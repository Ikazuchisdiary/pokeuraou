"""Does the equilibrium's heavier selection actually win more on the board?

`tools/ordering_check.py` emits, per opponent, two `--force-selection` arms: the four the
book weights most and the four it weights second. Both play the same equilibrium opponent,
and both arms seed game i from the same `[seed, i]`, so a game index means the same
opponent team and the same spread class in both. This reads those files back.

The question is the book's ordering, not its win rate. A rating cannot see this -- both
arms of `generation_match` draw from the same book, so an ordering error they share
cancels exactly -- and `book vs uniform` was +18.9 points because discarding the bad 87 of
90 is a different and much easier question than ordering the top three.

Two things are reported and they answer different questions:

  per opponent   the paired difference, heavy minus second, with its interval. This says
                 how much the ordering is worth on THIS opponent.
  across         a sign test on the direction. Six opponents measured earlier had the
                 heavier selection losing in four of five, which is a claim about the
                 book in general, and a claim about a direction is tested by counting
                 signs rather than by averaging differences that are not commensurable.

The mass each arm carries is re-derived from the book rather than taken from the order the
arms appear in the file. Reading a label's meaning off its position is how a measurement
ends up describing the wrong arm, which has happened twice on this project.

    uv run python tools/ordering_result.py
    uv run python tools/ordering_result.py --dir data/matches/ordering
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.selection_book import SelectionBook  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402


def wilson(wins: float, n: int) -> tuple[float, float]:
    """A proportion and its half-width. Normal approximation, as everywhere else here."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    return p, 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def sign_test(plus: int, minus: int) -> float:
    """Two-sided exact binomial p for `plus` successes in `plus + minus` at 1/2.

    Ties are dropped before this is called, which is the standard sign test and is why the
    denominator here is not the number of opponents measured.
    """
    n = plus + minus
    if n == 0:
        return 1.0
    def comb(k: int) -> float:
        return math.comb(n, k)
    extreme = min(plus, minus)
    tail = sum(comb(k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/matches/ordering"))
    ap.add_argument(
        "--book",
        type=Path,
        default=Path("data/selection/rizabanadohido-value-gen11L.jsonl.gz"),
        help="the book whose ordering is on trial; its masses label the arms",
    )
    ap.add_argument("--roster", default="rizabanadohido")
    args = ap.parse_args()

    book = SelectionBook.read(args.book)
    roster = load_roster(args.roster)
    reg = roster.reg
    species = [reg.species[s.species].name.lower().replace(" ", "") for s in roster.sets]

    # `SelectionBook.entries` is keyed by the opponent's TEAM SHEET, not by their place,
    # so `entries.get(place)` is always None and every mass comes back 0.0 -- which then
    # sorts the two arms arbitrarily and labels whichever it likes "heavy". That is the
    # exact error this tool's docstring says it re-derives the masses to avoid, so it is
    # indexed properly here and a missing place is refused rather than scored as zero.
    by_place = {entry.place: entry for entry in book.entries.values()}

    # label -> mass, per place. The label `selection_check` writes is
    # "front+front / back+back" over species ids, and the front pair is a set there
    # because lead order is not a choice the game exposes, so it is a set here too.
    def masses(place: int) -> dict[frozenset[tuple[str, ...]], float]:
        entry = by_place.get(place)
        if entry is None:
            return {}
        out: dict[frozenset[tuple[str, ...]], float] = {}
        for index, sel in enumerate(entry.selections):
            key = frozenset(
                {
                    tuple(sorted(species[i] for i in sel[:2])),
                    tuple(sorted(species[i] for i in sel[2:])),
                }
            )
            out[key] = out.get(key, 0.0) + float(entry.our_strategy[index])
        return out

    def key_of(label: str) -> frozenset[tuple[str, ...]]:
        front, _, back = label.partition(" / ")
        return frozenset(
            {tuple(sorted(front.split("+"))), tuple(sorted(back.split("+")))}
        )

    places = sorted(
        {int(p.name.split(".")[0].removeprefix("place")) for p in args.dir.glob("place*.jsonl")}
        | {int(p.name.split(".")[0].removeprefix("place")) for p in args.dir.glob("place*.part*.jsonl")}
    )
    if not places:
        raise SystemExit(f"no place*.jsonl under {args.dir}")

    print(f"  book {args.book.name}")
    print(
        f"\n  {'place':>5}  {'heavy four':<34} {'mass':>6}  "
        f"{'heavy':>6} {'second':>7}  {'diff':>7}  {'±':>5}  {'n':>4}"
    )
    signs: list[tuple[int, float]] = []
    rows = 0
    for place in places:
        by_arm: dict[str, dict[int, float]] = defaultdict(dict)
        for path in sorted(args.dir.glob(f"place{place}.part*.jsonl")) or sorted(
            args.dir.glob(f"place{place}.jsonl")
        ):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "header" in row or "arm" not in row:
                    continue
                by_arm[row["arm"]][int(row["game"])] = float(row["outcome"])
        arms = [a for a in by_arm if "/" in a]
        if len(arms) != 2:
            print(f"  {place:>5}  incomplete: {len(arms)} forced arms, skipped")
            continue
        weight = masses(place)
        found = {a: weight.get(key_of(a)) for a in arms}
        if any(w is None for w in found.values()):
            missing = [a for a, w in found.items() if w is None]
            print(
                f"  {place:>5}  cannot find {missing} in the book, skipped.\n"
                "         Without both masses there is no heavy arm to name, and sorting\n"
                "         by a default of zero would name one anyway."
            )
            continue
        labelled = sorted(arms, key=lambda a: -found[a])
        heavy, second = labelled
        if found[heavy] == found[second]:
            print(f"  {place:>5}  the two arms carry equal mass, so there is no ordering")
            continue
        # Paired on the game index: the same opponent and the same spread class.
        shared = sorted(set(by_arm[heavy]) & set(by_arm[second]))
        if not shared:
            print(f"  {place:>5}  no shared game indices, skipped")
            continue
        diffs = [by_arm[heavy][g] - by_arm[second][g] for g in shared]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
        half = 1.96 * math.sqrt(var / n)
        hp, _ = wilson(sum(by_arm[heavy][g] for g in shared), n)
        sp, _ = wilson(sum(by_arm[second][g] for g in shared), n)
        print(
            f"  {place:>5}  {heavy:<34} {weight.get(key_of(heavy), 0.0):>5.1%}  "
            f"{hp:>5.1%} {sp:>6.1%}  {mean:>+6.1%}  {half:>5.1%}  {n:>4}"
        )
        rows += 1
        if mean != 0.0:
            signs.append((1 if mean > 0 else -1, mean))

    plus = sum(1 for s, _ in signs if s > 0)
    minus = len(signs) - plus
    print(f"\n  {rows} opponents, {plus} where the heavier four won, {minus} where it lost")
    print(f"  sign test p = {sign_test(plus, minus):.3f}  (ties dropped: {rows - len(signs)})")
    if signs:
        overall = sum(m for _s, m in signs) / len(signs)
        print(f"  mean paired difference across opponents {overall:+.1%}")
    print(
        "\n  A negative difference means the book ordered the two backwards ON THAT\n"
        "  OPPONENT. It does not mean the book is worse than uniform: discarding the bad\n"
        "  87 of 90 and ordering the top three are different questions, and the first was\n"
        "  already measured at +18.9."
    )


if __name__ == "__main__":
    main()
