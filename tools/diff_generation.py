"""Does a generated game come out the same with the Rust node as without it?

The node differential compares matrices. This compares the only thing that is finally
used: the game. Both sides play the same seeds, and because every action is sampled from
the equilibrium of a matrix this fills, any difference in that matrix large enough to move
an LP solution would show up here as a different game -- a different move chosen, a
different winner, a different number of turns.

    POKEURAOU_RUST_NODE=1 uv run python tools/diff_generation.py --games 4
    POKEURAOU_RUST_NODE=1 uv run --group learn python tools/diff_generation.py --games 2 \
        --value data/models/value-gen234.pt

It also times both, which is the number the port exists for. With `--value` the games are
played by the learned value function, which is what generation actually runs on -- and the
crossing is the other one: the leaves go across as the encoder's arrays rather than the
payoff coming back as a number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def _report(index: int, python: dict, rust: dict) -> None:
    """Says *what* differs, because the two kinds mean opposite things.

    A different action, winner or turn count is a divergence: the port answered something
    that moved a decision. The same decisions with a search value a few places apart is
    float arithmetic, and the game the generation run stores is the same game.
    """
    for field in ("turns", "outcome", "unmodelled"):
        if python[field] != rust[field]:
            print(f"\n  game {index}: {field} {python[field]!r} against {rust[field]!r}")
    if len(python["decisions"]) != len(rust["decisions"]):
        print(
            f"\n  game {index}: {len(python['decisions'])} decisions against "
            f"{len(rust['decisions'])}"
        )
        return
    worst = 0.0
    for step, (mine, theirs) in enumerate(
        zip(python["decisions"], rust["decisions"], strict=True)
    ):
        if mine[:2] != theirs[:2]:
            print(
                f"\n  game {index} decision {step}: chose {mine[:2]!r} in python and "
                f"{theirs[:2]!r} through the port"
            )
            return
        worst = max(worst, abs(mine[2] - theirs[2]))
    print(
        f"  game {index}: the same decisions throughout; the search value differs by at "
        f"most {worst:.1e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--value",
        default=None,
        help="play the games with a trained value function (data/models/*.pt) instead of "
        "hp-share",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--solve-sparsely",
        action="store_true",
        help="solve nodes by proving the equilibrium rather than filling the matrix. The "
        "port must not change the answer with it on either.",
    )
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        device = torch.device(args.device)
        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        objective = BatchedValue(net.to(device), encoder, device=device).objective("win")
    else:
        objective = OBJECTIVES["hp-share"]

    def run(use_rust: bool) -> tuple[list[str], float, int]:
        os.environ[rustnode.ENV_ENABLE] = "1" if use_rust else "0"
        # A toggle mid-process must not keep a process from the other setting.
        rustnode.reset()
        records: list[str] = []
        turns = 0
        rng = np.random.default_rng(args.seed)
        started = time.perf_counter()
        for _ in range(args.games):
            team = pool[int(rng.integers(len(pool)))]
            foe_six = sample_standings_team(rng, reg, prior, team)
            own_pick = selections[int(rng.integers(len(selections)))]
            foe_pick = selections[int(rng.integers(len(selections)))]
            record = play_game(
                reg,
                rng,
                [roster.sets[i] for i in own_pick],
                [foe_six[j] for j in foe_pick],
                "diff-generation",
                objective=objective,
                search_limit=args.limit,
                max_turns=args.max_turns,
                solve_sparsely=args.solve_sparsely,
                open_information=True,
            )
            turns += record.turns
            # The decisions are what a difference would show up in first.
            records.append(
                json.dumps(
                    {
                        "turns": record.turns,
                        "outcome": record.outcome,
                        # The reported effects are part of a game record, and a caller
                        # reads them; a bridged game must not report a different set.
                        "unmodelled": sorted(set(record.unmodelled)),
                        "decisions": [
                            [d.own_chosen, d.foe_chosen, round(d.search_value, 12)]
                            for d in record.decisions
                        ],
                    },
                    sort_keys=True,
                    default=str,
                )
            )
        return records, time.perf_counter() - started, turns

    build = rustnode.require_current_binary()
    print(f"binary {build['sha256']} built {build['built']}")

    python_records, python_seconds, turns = run(False)
    rust_records, rust_seconds, rust_turns = run(True)

    same = sum(1 for a, b in zip(python_records, rust_records, strict=True) if a == b)
    print(f"{args.games} games, {turns} turns")
    print(f"  identical games {same}/{args.games}")
    for index, (a, b) in enumerate(zip(python_records, rust_records, strict=True)):
        if a != b:
            _report(index, json.loads(a), json.loads(b))
    print(f"  python {python_seconds:.1f} s ({python_seconds / max(turns, 1):.3f} s/turn)")
    print(f"  rust   {rust_seconds:.1f} s ({rust_seconds / max(rust_turns, 1):.3f} s/turn)")
    if rust_seconds > 0:
        print(f"  generation is {python_seconds / rust_seconds:.1f}x faster")


if __name__ == "__main__":
    main()
