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
    # "front+front / back+back" over species ids. Lead order WITHIN a pair is not a choice
    # the game exposes, so each pair is a set -- but which pair leads is the whole point,
    # so the key is an ORDERED (front, back).
    #
    # It was a frozenset of the two pairs, which cannot tell a selection from the same
    # four with the leads and the reserves swapped: that collapses the 90 selections into
    # 45, and the two halves of a collision are the opposite decision. Nothing showed it
    # on the first opponent checked, because that selection's swap carried no mass.
    Key = tuple[tuple[str, ...], tuple[str, ...]]

    def masses(place: int) -> dict[Key, float]:
        entry = by_place.get(place)
        if entry is None:
            return {}
        out: dict[Key, float] = {}
        for index, sel in enumerate(entry.selections):
            key = (
                tuple(sorted(species[i] for i in sel[:2])),
                tuple(sorted(species[i] for i in sel[2:])),
            )
            out[key] = out.get(key, 0.0) + float(entry.our_strategy[index])
        return out

    def losses(place: int) -> dict[Key, float]:
        """The book's predicted EV loss per selection, keyed like `masses`.

        Masses are summed across the selections that collapse to one key; losses are
        minimised, because the collapsed selections differ only by the lead order within
        a pair, which the game does not expose -- their losses should be equal and the
        minimum says so without averaging two numbers that are meant to be one.
        """
        entry = by_place.get(place)
        if entry is None:
            return {}
        out: dict[Key, float] = {}
        for index, sel in enumerate(entry.selections):
            key = (
                tuple(sorted(species[i] for i in sel[:2])),
                tuple(sorted(species[i] for i in sel[2:])),
            )
            value = float(entry.our_ev_loss[index])
            out[key] = min(out.get(key, value), value)
        return out

    def key_of(label: str) -> Key:
        front, _, back = label.partition(" / ")
        return (tuple(sorted(front.split("+"))), tuple(sorted(back.split("+"))))

    places = sorted(
        {int(p.name.split(".")[0].removeprefix("place")) for p in args.dir.glob("place*.jsonl")}
        | {int(p.name.split(".")[0].removeprefix("place")) for p in args.dir.glob("place*.part*.jsonl")}
    )
    if not places:
        raise SystemExit(f"no place*.jsonl under {args.dir}")

    print(f"  book {args.book.name}")
    print(
        f"\n  {'place':>5}  {'heavy four':<34} {'mass':>6}  "
        f"{'heavy':>6} {'second':>7}  {'diff':>7}  {'±':>5}  {'pred':>6}  {'n':>4}"
    )
    signs: list[tuple[int, float]] = []
    calibration: list[tuple[float, float]] = []
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
        # What the book staked. The heavy arm is in the support so its loss is zero, and
        # the prediction for `heavy - second` is the second arm's loss. Both arms in the
        # support makes this 0.0 by construction, which is the point the docstring makes.
        loss = losses(place)
        predicted = loss.get(key_of(second), 0.0) - loss.get(key_of(heavy), 0.0)
        print(
            f"  {place:>5}  {heavy:<34} {weight.get(key_of(heavy), 0.0):>5.1%}  "
            f"{hp:>5.1%} {sp:>6.1%}  {mean:>+6.1%}  {half:>5.1%}  {predicted:>+5.1%}  {n:>4}"
        )
        rows += 1
        if mean != 0.0:
            signs.append((1 if mean > 0 else -1, mean))
        calibration.append((predicted, mean))

    plus = sum(1 for s, _ in signs if s > 0)
    minus = len(signs) - plus
    print(f"\n  {rows} opponents, {plus} where the heavier four won, {minus} where it lost")
    print(f"  sign test p = {sign_test(plus, minus):.3f}  (ties dropped: {rows - len(signs)})")
    # Two tests, because they need different things and fail differently.
    #
    # The sign test reads only the direction, so it survives an opponent whose difference
    # is enormous for a reason of its own -- and it is blunt: with twelve opponents it
    # needs ten of them one way to clear 0.05, and no number of games per opponent adds a
    # sign. The mean across opponents uses the magnitudes, which is the sharper question
    # and assumes they are commensurable: a 26-point difference on one opponent and a
    # 6-point one on another are being averaged as though a point is a point.
    if len(signs) > 1:
        values = [m for _s, m in signs]
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        half = 1.96 * math.sqrt(var / n)
        print(
            f"  mean paired difference across opponents {mean:+.1%} +-{half:.1%}"
            + ("  (excludes zero)" if abs(mean) > half else "  (includes zero)")
        )
    elif signs:
        print(f"  one opponent only: {signs[0][1]:+.1%}, no interval across opponents")
    # The prediction column decides what the rest of this output is allowed to mean, so
    # it is read before anything is concluded rather than printed as one more number.
    staked = [p for p, _m in calibration if abs(p) > 1e-9]
    if not staked:
        print(
            "\n  THE BOOK PREDICTED NOTHING HERE. Every pair has both arms in the support,\n"
            "  where complementary slackness makes each of them worth exactly the game\n"
            "  value against the opponent's equilibrium -- so the true difference is zero\n"
            "  by construction and the signs above should be a coin flip. Mass inside the\n"
            "  support is set by the OPPONENT's indifference conditions; it is not a\n"
            "  ranking of our payoffs, and a departure from zero here is the leaf\n"
            "  mispricing selections rather than the book ordering them backwards.\n"
            "  For a comparison the book can lose, run `ordering_check.py --second-by\n"
            "  evloss`, which plays the heaviest against a selection it REJECTED."
        )
    else:
        pairs = [(p, m) for p, m in calibration if abs(p) > 1e-9]
        mean_p = sum(p for p, _ in pairs) / len(pairs)
        mean_m = sum(m for _, m in pairs) / len(pairs)
        agree = sum(1 for p, m in pairs if (m > 0) == (p > 0))
        print(
            f"\n  {len(pairs)} opponents where the book staked a number: it predicted "
            f"{mean_p:+.1%} on average and the board gave {mean_m:+.1%}, with the sign "
            f"matching on {agree} of {len(pairs)}."
        )
        denom = sum(p * p for p, _ in pairs)
        if denom > 0:
            slope = sum(p * m for p, m in pairs) / denom
            print(
                f"  Through the origin the board is {slope:.2f}x what the book claimed. "
                "One would be\n  calibrated, zero would mean its EV losses carry no board "
                "information at all, and\n  a negative slope would mean it has the "
                "selections backwards."
            )
    print(
        "\n  Whatever the sign, this is not a verdict on the book against uniform:\n"
        "  discarding the bad 87 of 90 and ordering the top few are different questions,\n"
        "  and the first was already measured at +18.9."
    )


if __name__ == "__main__":
    main()
