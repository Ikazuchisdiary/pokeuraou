"""Does a whole node come out the same through the Rust port as without it?

The turn differential compares one `resolve_turn` at a time. This compares what the caller
actually asks for -- a filled matrix -- and it asks for it the way production does, through
`batched_payoffs`, with the bridge on and then off. Cells the port refuses are filled in
Python by the bridge itself, so what is compared is the finished matrix.

    uv run python tools/diff_node.py --games 2 --nodes 20
    uv run python tools/diff_node.py --games 2 --nodes 8 --value data/models/value-gen234.pt

With `--value` the leaves cross as the encoder's arrays rather than the payoff crossing as
a number, because a learned leaf's input *is* the leaves. The forward pass stays in torch
either way.

Exactness: the resolver is bit-identical, so a cell differs only where a weighted mean is
summed in a different order -- numpy's dot product against a sequential loop. The tool
reports how many cells are bit-identical, the worst difference, and what it does to the
equilibrium, which is the number anyone actually reads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def collect_positions(reg, roster, prior, pool, selections, args) -> list:  # noqa: ANN001
    """Positions a real game visits, collected with the bridge off.

    With it on the search never calls Python's resolver, so the hook that collects them
    would see almost nothing. Every cell of a node resolves against the same position, so
    one call in a few hundred is taken rather than all of them.
    """
    import pokeuraou.resolve as resolve_mod
    import pokeuraou.search as search_mod

    was = os.environ.get(rustnode.ENV_ENABLE, "")
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()

    real = resolve_mod.resolve_turn
    seen: list = []
    calls = [0]

    def recording(reg_, pos, actions, *, budget):  # noqa: ANN001
        calls[0] += 1
        if calls[0] % 137 == 0 and len(seen) < args.nodes * 4:
            seen.append(pos.copy())
        return real(reg_, pos, actions, budget=budget)

    resolve_mod.resolve_turn = recording
    search_mod.resolve_turn = recording
    rng = np.random.default_rng(args.seed)
    for _ in range(args.games):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        play_game(
            reg,
            rng,
            [roster.sets[i] for i in own_pick],
            [foe_six[j] for j in foe_pick],
            "diff-node",
            objective=OBJECTIVES["hp-share"],
            search_limit=args.limit,
            max_turns=args.max_turns,
        )
    resolve_mod.resolve_turn = real
    search_mod.resolve_turn = real
    os.environ[rustnode.ENV_ENABLE] = was or "1"

    distinct: list = []
    keys: set[str] = set()
    for pos in seen:
        key = str(pos.to_json())
        if key in keys:
            continue
        keys.add(key)
        distinct.append(pos)
        if len(distinct) >= args.nodes:
            break
    return distinct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=20, help="stop after this many nodes")
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--value",
        default=None,
        help="a trained value function (data/models/*.pt). The leaves then cross as the "
        "encoder's arrays instead of the payoff crossing as a number.",
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

        torch.set_num_threads(1)
        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        leaf = BatchedValue(net.to(torch.device("cpu")), encoder, device=torch.device("cpu"))
        # The analyser's own pair: the learned objective and a parameter-free one beside
        # it as a cross-check. Both scored from the same leaves, one here and one there.
        names = ["the learned leaf", "hp-share"]
        evaluators = [leaf, OBJECTIVES["hp-share"].batch]
    else:
        names = ["hp-share", "faints"]
        evaluators = [OBJECTIVES[name].batch for name in names]

    positions = collect_positions(reg, roster, prior, pool, selections, args)
    if not rustnode.binary_path().exists():
        raise SystemExit(f"no Rust binary at {rustnode.binary_path()}; build it first")

    # A refused cell is filled by Python, and for a learned leaf that means its leaves are
    # scored in a forward pass of their own rather than with the rest of the node. float32
    # matrix arithmetic is not shape-independent, so that alone can move a cell -- which is
    # why the count is reported next to the worst difference rather than left implicit.
    refused = [0]
    for name in ("fill", "fill_encoded"):
        original = getattr(rustnode.RustNode, name)

        def counting(self, *args, _original=original, **kwargs):  # noqa: ANN001
            filled = _original(self, *args, **kwargs)
            refused[0] += len(filled.refused)
            return filled

        # On the class, so the count survives the `reset()` between the two runs.
        setattr(rustnode.RustNode, name, counting)

    budget = Budget.matrix()
    checked = cells = identical = differing = 0
    worst = worst_value = worst_strategy = 0.0
    rust_seconds = python_seconds = 0.0

    for pos in positions:
        row = narrow(reg, pos, 0, limit=args.limit).actions
        col = narrow(reg, pos, 1, limit=args.limit).actions
        if not row or not col:
            continue

        os.environ[rustnode.ENV_ENABLE] = "0"
        rustnode.reset()
        started = time.perf_counter()
        expected, _notes, python_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=budget
        )
        python_seconds += time.perf_counter() - started

        os.environ[rustnode.ENV_ENABLE] = "1"
        rustnode.reset()
        started = time.perf_counter()
        got, _rust_notes, rust_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=budget
        )
        rust_seconds += time.perf_counter() - started

        checked += 1
        cells += len(row) * len(col)
        for index, name in enumerate(names):
            gap = np.abs(np.asarray(got[index]) - np.asarray(expected[index]))
            worst = max(worst, float(gap.max()))
            identical += int((gap == 0).sum())
            differing += int((gap > 0).sum())
            if gap.max() > 1e-9:
                where = np.unravel_index(int(np.argmax(gap)), gap.shape)
                print(
                    f"  {name} differs by {gap.max():.3e} at cell {where}: "
                    f"python {expected[index][where]!r} rust {got[index][where]!r}"
                )
            python_eq = solve(np.asarray(expected[index]))
            rust_eq = solve(np.asarray(got[index]))
            worst_value = max(worst_value, abs(python_eq.value - rust_eq.value))
            worst_strategy = max(
                worst_strategy,
                float(np.abs(python_eq.row_strategy - rust_eq.row_strategy).max()),
                float(np.abs(python_eq.col_strategy - rust_eq.col_strategy).max()),
            )
        if not np.array_equal(np.asarray(rust_exact), np.asarray(python_exact)):
            print("  the exact mask differs")

    rustnode.reset()
    scored = identical + differing
    leaf_kind = "a learned leaf and hp-share" if args.value else "hp-share and faints"
    print(f"\n{checked} nodes, {cells} cells, scored by {leaf_kind}")
    print(
        f"  bit-identical {identical}/{scored} scored cells "
        f"({identical / max(scored, 1) * 100:.2f}%)"
    )
    print(f"  cells the port refused {refused[0]}")
    print(f"  worst cell difference {worst:.3e}")
    print(f"  equilibrium value moved at most {worst_value:.3e}")
    print(f"  equilibrium frequency moved at most {worst_strategy:.3e}")
    print(f"  python {python_seconds:.2f} s   rust {rust_seconds:.2f} s")
    if rust_seconds > 0:
        print(f"  end to end {python_seconds / rust_seconds:.1f}x")


if __name__ == "__main__":
    main()
