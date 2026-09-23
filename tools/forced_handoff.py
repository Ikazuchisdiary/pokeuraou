"""Is the equilibrium right to price the handoff at zero?

The switch to the sweeper survives `narrow` in 98.5% of the positions where it is
available, and the equilibrium gives it no weight at all in 60% of them. The 14% where it
is taken are the positions the equilibrium already liked, so they say the leaf ranks
*those* correctly and nothing about the 60%. The zero is the untested claim.

This tests it the only way it can be tested: take the positions the equilibrium priced at
zero, play each one twice from the same start with the same seed -- once as the search
wants, once with the handoff forced as the opening move -- and compare who wins.

Paired, because the alternative is not. Two independent samples of a position whose value
is near a coin flip would need thousands of games to separate a few points; the same
position played both ways differs only in the thing under test, and the seed makes the
opponent's draws line up as far as the two games stay in step.

Only the opening move is forced. After it the search plays its own game, so this measures
"was that move worth playing here", not "is a policy that always switches any good".

    uv run --group learn python tools/forced_handoff.py --games 300

**Open game only: a reference** (IKA-123). This tool has no hidden-bench path, so its
search is shown the opponent's four -- not the game that ships. It says so on stderr
when it runs, and `tools/agent_drift.py` lists it under KNOWN_DRIFT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.benchflags import say_open_reference  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402

CHIPPERS = {"toxapex", "incineroar"}


def _bench_slot(position: dict, species: str) -> int | None:
    for index, mon in enumerate((position.get("sides") or [{}])[0].get("pokemon") or [],
                                start=1):
        if (
            str(mon.get("species") or "") == species
            and mon.get("activeIndex") is None
            and not mon.get("fainted")
        ):
            return index
    return None


def collect(directories: list[Path], sweeper: str, cap: int) -> list[dict]:
    """Positions where the switch was on the menu and the equilibrium gave it zero."""
    found: list[dict] = []
    for directory in directories:
        for path in sorted(directory.glob("*.jsonl")):
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
                    for decision in record.get("decisions", ()):
                        if decision.get("kind") != "move":
                            continue
                        position = decision.get("position") or {}
                        sides = position.get("sides") or []
                        if len(sides) < 2:
                            continue
                        actives = {
                            str(m.get("species") or "")
                            for m in sides[0].get("pokemon") or []
                            if m.get("activeIndex") is not None
                        }
                        if not (CHIPPERS & actives):
                            continue
                        slot = _bench_slot(position, sweeper)
                        if slot is None:
                            continue
                        names = decision.get("ownActions") or []
                        policy = np.asarray(decision.get("ownPolicy") or [],
                                            dtype=np.float64)
                        if len(names) != len(policy) or not len(names):
                            continue
                        want = f"switch {slot}"
                        hits = [
                            i for i, n in enumerate(names)
                            if any(part.strip() == want for part in n.split(","))
                        ]
                        if not hits:
                            continue
                        if float(policy[hits].sum()) > 1e-9:
                            continue  # the equilibrium liked it; not the case under test
                        found.append({
                            "position": position,
                            "action": names[hits[0]],
                            "turn": int(decision.get("turn", 0)),
                            "foeArchetype": record.get("foeArchetype", "?"),
                        })
                        break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, nargs="+",
                    default=[Path("data/selfplay-gen9")])
    ap.add_argument("--sweeper", default="venusaur")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--value", type=Path, default=Path("data/models/value-gen9.pt"))
    ap.add_argument("--baseline-value", type=Path, default=None,
                    help="second member of the leaf ensemble; the floor wants two")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    say_open_reference()

    starts = collect(list(args.dir), args.sweeper, args.games * 3)
    if not starts:
        raise SystemExit("no positions where the handoff was on the menu and priced at 0")
    print(f"{len(starts):,} positions collected, {args.games} to play twice each",
          file=sys.stderr)

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(1)
    reg = load_regulation(starts[0]["position"]["format"])
    register_mega_stones(reg)
    encoder = Encoder(reg)
    paths = [args.value] + ([args.baseline_value] if args.baseline_value else [])
    nets, _ = load_ensemble(paths, encoder)
    device = torch.device(args.device)
    value = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    objective = OBJECTIVES["hp-share"]

    rng = np.random.default_rng(args.seed)
    free_wins = forced_wins = pairs = 0
    both_finished = 0
    started = time.perf_counter()
    out = args.out.open("w", encoding="utf-8") if args.out else None

    for index in range(args.games):
        pick = starts[int(rng.integers(len(starts)))]
        results = {}
        for label, forced in (("free", None), ("forced", pick["action"])):
            # The same seed for both arms of a pair: the two games differ in the opening
            # move and in nothing else that this can control.
            game_rng = np.random.default_rng([args.seed, index])
            record = play_game(
                reg, game_rng, [], [], pick["foeArchetype"],
                objective=objective, search_limit=args.limit,
                max_turns=args.max_turns, evaluate=value, rank_by_leaf=True,
                start=Position.from_json(pick["position"]),
                first_action=forced,
                open_information=True,
            )
            results[label] = record.outcome
        pairs += 1
        if results["free"] is None or results["forced"] is None:
            continue
        both_finished += 1
        free_wins += int(results["free"] > 0.5)
        forced_wins += int(results["forced"] > 0.5)
        if out:
            out.write(json.dumps({
                "turn": pick["turn"], "action": pick["action"],
                "free": results["free"], "forced": results["forced"],
            }) + "\n")
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{args.games} pairs, "
                  f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    if out:
        out.close()

    if not both_finished:
        raise SystemExit("no pair finished both games")
    free = free_wins / both_finished
    forced = forced_wins / both_finished
    # Wilson on the difference is not right for paired data, so the interval here is the
    # crude one and the honest statement is the count.
    print(f"\n  {both_finished} pairs where both games finished "
          f"(of {pairs} started)")
    print(f"    search's own opening   {free_wins:>5}  {free:>6.1%}")
    print(f"    handoff forced         {forced_wins:>5}  {forced:>6.1%}")
    print(f"    difference             {100 * (forced - free):>+6.1f} points")
    print("\n  positions were chosen because the equilibrium gave the handoff zero "
          "weight,\n  so a positive difference means the zero was wrong.")


if __name__ == "__main__":
    main()
