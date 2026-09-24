"""What changes when the search stops being told the opponent's back two.

47.4% of recorded move decisions, and every opening one, were solved from a position that
contained the opponent's whole four. This plays the same position both ways and prints
both answers:

- **omniscient** -- today's search, the true bench in the matrix;
- **belief** -- one matrix per completion the sheet allows (:mod:`pokeuraou.hidden`),
  solved as the Bayesian game the position really is.

Not the average of the matrices, and not the average of the strategies. Averaging the
matrices solves a game where the opponent has to move before learning their own bench,
which is a thing that never happens -- they packed it -- and it returns a value we cannot
collect; measured on the first eight openings it claimed 0.139 *more* than the omniscient
solve, which is how the mistake announces itself. Averaging the strategies is worse: it
mixes plans that each answer a different opponent and answers none of them.
:func:`~pokeuraou.equilibrium.solve_bayesian`, written for the spread belief, is the same
game with a different type space -- one strategy for us, one per class for them.

Three numbers come out of it, and they are different questions:

- **the mixture** -- what the solver would tell a person to play. The product is a mixed
  strategy, so a change here is a change in the advice whether or not the value moves;
- **the value of knowing the bench** -- each completion solved with both sides informed,
  averaged, minus what one strategy can hold across all six. Non-negative by
  construction. Comparing the omniscient solve to the belief solve instead does *not*
  measure this: those are games against different opponents, one bench against a mixture
  of six, and the difference came out negative;
- **the regret of playing the omniscient mixture into the belief matrix** -- how much the
  advice costs when the thing it assumed is not known. This is the one that says whether
  the leak matters, and it is zero exactly when the advice happens to be robust.

    uv run --group learn python tools/what_the_leak_buys.py --games 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve, solve_bayesian  # noqa: E402
from pokeuraou.hidden import completions  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.port import batched_payoff  # noqa: E402 - Python's resolver until IKA-212
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import SampledSet  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def _set_from_json(entry: dict) -> SampledSet:
    """A recorded team member back into the shape the position builder wants."""
    return SampledSet(
        species=entry["species"],
        ability=entry["ability"],
        item=entry.get("item") or None,
        nature=entry["nature"],
        sp={k: int(v) for k, v in (entry.get("sp") or {}).items()},
        moves=list(entry["moves"]),
    )


def openings(paths: list[Path], cap: int) -> list[dict]:
    """The first move decision of games that recorded both the six and the four.

    The opening is where the whole four is hidden, so it is where the difference is
    largest -- and it is the decision the selection book's advice is about, which makes it
    the one the product answers.
    """
    found: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(found) >= cap:
                    return found
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("outcome") is None:
                    continue
                six = record.get("foeSix") or []
                four = record.get("foeTeam") or []
                if len(six) < 6 or len(four) < 4:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") != "move":
                        continue
                    found.append({
                        "position": decision["position"],
                        "foeSix": six,
                        "foeTeam": four,
                    })
                    break
    return found


def _sheet(reg, six: list, four: list[dict]) -> list[SampledSet] | None:  # noqa: ANN001
    """The opponent's six as sets. `foeSix` is species names, so the two they did not
    bring have no spread recorded -- they are filled from the usage prior, which is what
    an observer would have to do anyway."""
    from pokeuraou.priors import find_cached_chaos, load_chaos, sample_set

    names = [s if isinstance(s, str) else s.get("species") for s in six]
    brought = {entry["species"]: _set_from_json(entry) for entry in four}
    path = find_cached_chaos(reg.meta.format_id)
    if path is None:
        return None
    prior = load_chaos(path, reg)
    rng = np.random.default_rng(0)
    sheet: list[SampledSet] = []
    for name in names:
        if name in brought:
            sheet.append(brought[name])
            continue
        entry = prior.species.get(name)
        if entry is None:
            return None
        try:
            sheet.append(sample_set(rng, reg, entry))
        except (KeyError, ValueError):
            return None
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen9"))
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--value", type=Path, default=Path("data/models/value-all.pt"))
    ap.add_argument("--second-value", type=Path, default=Path("data/models/value-all-s1.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = sorted(args.dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no recorded games under {args.dir}")
    starts = openings(paths, args.games)
    if not starts:
        raise SystemExit("no opening decision carried both the six and the four")
    print(f"{len(starts)} opening decisions", file=sys.stderr)

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_ensemble

    reg = load_regulation(starts[0]["position"]["format"])
    register_mega_stones(reg)
    torch.set_num_threads(1)
    encoder = Encoder(reg)
    files = [args.value] + ([args.second_value] if args.second_value.exists() else [])
    nets, _ = load_ensemble(files, encoder)
    device = torch.device(args.device)
    leaf = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    budget = Budget.matrix()

    moved = []
    value_gap = []
    regrets = []
    fanouts = []
    skipped = 0
    for index, start in enumerate(starts):
        position = Position.from_json(start["position"])
        sheet = _sheet(reg, start["foeSix"], start["foeTeam"])
        if sheet is None:
            skipped += 1
            continue
        ours = narrow(reg, position, 0, limit=args.limit).actions
        theirs = narrow(reg, position, 1, limit=args.limit).actions
        if not ours or not theirs:
            skipped += 1
            continue
        try:
            spread = completions(reg, position, 1, sheet)
        except ValueError:
            skipped += 1
            continue
        fanouts.append(len(spread))

        truth, _ = batched_payoff(reg, position, ours, theirs, leaf, budget=budget)
        # One matrix per completion. The action lists are the same in every one --
        # "switch to slot 3" is an action we can name without knowing what it brings --
        # which is exactly the abstraction an observer is stuck with.
        parts = [
            batched_payoff(reg, item.position, ours, theirs, leaf, budget=budget)[0]
            for item in spread
        ]
        weights = np.array([item.weight for item in spread], dtype=np.float64)

        omniscient = solve(truth)
        # Not `solve` on the average of `parts`. That game has the opponent moving before
        # they learn their own bench, which is a thing that never happens -- they packed
        # it -- and it hands us a value we cannot collect. `solve_bayesian` gives them
        # their own type and keeps the uncertainty on the side that really has it.
        belief = solve_bayesian(parts, weights)
        x_true = np.asarray(omniscient.row_strategy, dtype=np.float64)
        x_belief = np.asarray(belief.row_strategy, dtype=np.float64)
        moved.append(0.5 * float(np.abs(x_true - x_belief).sum()))
        # What knowing the bench is worth, against the *same* opponents: solve each
        # completion with both sides informed, average those values, and subtract the
        # value one strategy can hold across all of them. This is non-negative by
        # construction and is the quantity "the value of information" names.
        #
        # `omniscient.value - belief.value` is not that quantity, and the first version
        # of this tool printed it as though it were. The omniscient game is against the
        # one bench they actually brought; the belief game is against a mixture of six.
        # A team that packs a strong back two makes the true game the harder one, so the
        # difference came out *negative* and said nothing about information.
        informed = sum(
            float(weight) * float(solve(part).value)
            for weight, part in zip(weights, parts, strict=True)
        )
        value_gap.append(informed - float(belief.value))
        # What the omniscient advice is worth in the game an observer is actually in: the
        # opponent answers it knowing their own bench, class by class. `belief.value` is
        # the most any single strategy could have guaranteed there.
        guaranteed = sum(
            float(weight) * float(np.min(x_true @ part))
            for weight, part in zip(weights, parts, strict=True)
        )
        regrets.append(float(belief.value) - guaranteed)
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(starts)}", file=sys.stderr, flush=True)

    if not moved:
        raise SystemExit("nothing solvable")
    played = len(moved)

    def line(name: str, values: list[float], scale: float = 1.0) -> None:
        arr = np.asarray(values) * scale
        print(f"  {name:<44} {arr.mean():>8.3f}   median {np.median(arr):>7.3f}   "
              f"max {arr.max():>7.3f}")

    print(f"\n  {played} opening decisions solved twice"
          f"{f', {skipped} skipped' if skipped else ''}, "
          f"{np.mean(fanouts):.1f} completions each\n")
    line("advice moved (total variation, 0-1)", moved)
    line("value of knowing the bench (points)", value_gap, 100.0)
    line("regret of the omniscient advice (points)", regrets, 100.0)
    unchanged = sum(1 for m in moved if m < 1e-6)
    print(f"\n  advice unchanged in {unchanged} of {played} openings "
          f"({unchanged / played:.0%}); replaced outright in "
          f"{sum(1 for m in moved if m > 0.999)}")
    print("\n  The first says whether the product's answer changes at all. The third is\n"
          "  what it costs to follow advice that assumed the bench was known: it is the\n"
          "  value the belief game guarantees, minus what that advice actually gets\n"
          "  against a best reply in it.")


if __name__ == "__main__":
    main()
