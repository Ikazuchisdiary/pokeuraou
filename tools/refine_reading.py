"""Where does depth 2 lose the game it wins on paper -- in the numbers, or in the reading?

IKA-12 played the shipped `depth=2` out and it lost, 48.28% +-0.71 over 16,000 games,
while every number it produced got better: it moves side 0's own equilibrium value by
+0.00715 +-0.00095, the direction of that move predicts the result (r = +0.147), and its
forecast beats depth 1's on Brier. `search.py` names the suspect itself -- the strategy is
read off a matrix of 16 refined cells and 560 unrefined ones, which is "the equilibrium of
neither game".

This measures the suspect before anything is changed, on the positions the search meets
(recorded mid-game decisions, not turn 1 and not a mirror), and it answers three separate
questions that the win rate cannot tell apart:

  bias        `depth2 - depth1` per refined cell. If it is a near-constant shift, the
              refined cells are attractive to the row player for a reason that is about
              depth and not about play -- and the column player is pushed out of exactly
              the region that was refined. If its sign varies by position, it is
              information and the mixing is not what breaks.
  reading     the same refined cells read two ways: the shipped mixed matrix (24x24, 16
              cells deep), and the restricted game (4x4, every cell deep, which is the
              double-oracle's own answer). The strategies are compared directly, so
              "would this change any move at all" is answered before a single game is
              played. A reading that changes no move is a reading no match can measure.
  truncation  `--sweep` recomputes the same cells at wider sub-games. The refinement
              truncates the opponent's reply inside a cell (`DEFAULT_SUB_LIMIT` = 8), and a
              truncated reply flatters the row player. If the cell's value falls as the
              sub-game widens, part of +0.0072 is that truncation rather than depth.

The menus come from `selfplay._menus` and the cells from `search._refined_value`: the
shipped functions, called here, rather than a copy of what they do. A measurement tool
that reimplements the thing it measures answers a question about the copy.

    uv run --group learn python tools/refine_reading.py --positions 40 \\
        --value data/models/value-gen11L.pt data/models/value-gen11L-s1.pt
    uv run --group learn python tools/refine_reading.py --positions 8 --sweep 8,16,24
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from budget_effect import load_positions, total_variation  # noqa: E402

from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.port import batched_payoff  # noqa: E402 - Python's resolver until IKA-212
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.search import (  # noqa: E402
    DEFAULT_PASSES,
    DEFAULT_REFINE,
    DEFAULT_SUB_BRANCHES,
    DEFAULT_SUB_LIMIT,
    _refined_value,
    _top,
    search,
)
from pokeuraou.selfplay import _menus  # noqa: E402


def mean_pm(values: list[float]) -> str:
    """Mean and its 95% interval, the way every other tool in here prints one."""
    if not values:
        return "n/a"
    array = np.asarray(values, dtype=np.float64)
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size) if array.size > 1 else 0.0
    return f"{array.mean():+.5f} +-{half:.5f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--limit", type=int, default=24, help="the shipped width")
    ap.add_argument("--refine", type=int, default=DEFAULT_REFINE)
    ap.add_argument("--passes", type=int, default=DEFAULT_PASSES)
    ap.add_argument("--sub-limit", type=int, default=DEFAULT_SUB_LIMIT)
    ap.add_argument("--sub-branches", type=int, default=DEFAULT_SUB_BRANCHES)
    ap.add_argument(
        "--sweep",
        default="",
        help="comma-separated sub-game widths; recomputes the same cells at each",
    )
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument(
        "--max-turn",
        type=int,
        default=None,
        help="with --min-turn 1 --max-turn 1, the turn the +0.00715 was measured on",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--value",
        nargs="*",
        default=[],
        help="model files; several are an ensemble. Empty runs hp-share, which is a "
        "smoke test and not the agent",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--rank-leaf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the shipped narrowing order",
    )
    args = ap.parse_args()

    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")

    positions = load_positions(
        args.games_dir, args.min_turn, args.seed, args.max_turn
    )
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)

    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_ensemble

        encoder = Encoder(reg)
        nets, _meta = load_ensemble(list(args.value), encoder)
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        leaf = BatchedValue([n.to(device) for n in nets], encoder, device=device)
        leaf_name = "+".join(Path(v).stem for v in args.value)
    else:
        from pokeuraou.payoff import HP_SHARE

        leaf = HP_SHARE.batch
        leaf_name = "hp-share (smoke: not the agent)"

    budget = Budget.matrix()
    sweep = [int(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else []

    print(
        f"  leaf {leaf_name}, width {args.limit}, refine {args.refine} per side, "
        f"sub-game {args.sub_limit}x{args.sub_limit} over {args.sub_branches} branches"
    )
    print(
        f"  {args.positions} recorded positions from {args.games_dir} "
        f"(turn >= {args.min_turn}"
        f"{f', turn <= {args.max_turn}' if args.max_turn else ''}), menu ranked by "
        f"{'the leaf' if args.rank_leaf else 'damage'}\n"
    )
    header = (
        f"  {'#':>3} {'cells':>5} {'mean d':>9} {'sd d':>8} {'neg':>5} "
        f"{'v1':>7} {'mixed-v1':>9} {'restr-v1':>9} {'TV mix':>7} {'TV res':>7} "
        f"{'moved':>6} {'sec':>6}"
    )
    print(header)

    cell_deltas: list[float] = []
    per_position_delta: list[float] = []
    mixed_shift: list[float] = []
    restricted_shift: list[float] = []
    tv_mixed: list[float] = []
    tv_restricted: list[float] = []
    tv_between: list[float] = []
    gave_up_mixed: list[float] = []
    gave_up_restricted: list[float] = []
    arm_tv: list[float] = []
    arm_moved = 0
    arm_gave_up: dict[str, list[float]] = {"mixed": [], "restricted": []}
    arm_seconds: dict[str, list[float]] = {"mixed": [], "restricted": []}
    arm_subgames: dict[str, list[int]] = {"mixed": [], "restricted": []}
    arm_converged: dict[str, int] = {"mixed": 0, "restricted": 0}
    arm_shape: list[tuple[int, int]] = []
    arm_optimism: list[float] = []
    moved_mixed = moved_restricted = moved_between = 0
    sweep_deltas: dict[int, list[float]] = {width: [] for width in sweep}
    #: width -> (position, row, column) -> the cell's depth-2 value at that sub-game width.
    #: The means above are each over their own cells; the question is the change in the
    #: SAME cell, and a cell's value is far more alike across widths than across cells.
    sweep_cells: dict[int, dict[tuple[int, int, int], float]] = {
        width: {} for width in {*sweep, args.sub_limit}
    }
    done = 0
    taken = 0
    #: Why a position taken from the record was not measured. Every share printed below is
    #: a share of `done`, so the denominator has to be printed next to it -- "88.3% of
    #: decisions" was once written from this tool when it was 88.3% of the decisions that
    #: survived these filters.
    skipped: dict[str, int] = {
        "a side with fewer than 2 actions": 0,
        "depth 1 pure on a side": 0,
        "a cell that cannot be refined": 0,
        "an LP that failed": 0,
    }

    while done < args.positions and taken < len(positions):
        pos = positions[taken]
        taken += 1
        started = time.perf_counter()
        ours, theirs = _menus(
            reg, pos, (args.limit, args.limit), leaf, budget, args.rank_leaf
        )
        if len(ours) < 2 or len(theirs) < 2:
            skipped["a side with fewer than 2 actions"] += 1
            continue
        try:
            payoff, _notes = batched_payoff(reg, pos, ours, theirs, leaf, budget=budget)
            depth1 = solve(payoff)
        except EquilibriumError:
            skipped["an LP that failed"] += 1
            continue

        rows = _top(depth1.row_strategy, args.refine)
        cols = _top(depth1.col_strategy, args.refine)
        if rows.size < 2 or cols.size < 2:
            # A pure-against-pure cell has no mixture to read differently, and the two
            # readings are the same object. Counting those would dilute the share of
            # positions where the reading matters with positions where it cannot.
            skipped["depth 1 pure on a side"] += 1
            continue

        refined = np.array(payoff, dtype=np.float64, copy=True)
        deltas: list[float] = []
        ok = True
        for i in rows:
            for j in cols:
                value, _notes, _solved = _refined_value(
                    reg, pos, ours[int(i)], theirs[int(j)], leaf,
                    budget=budget,
                    sub_limit=args.sub_limit,
                    sub_branches=args.sub_branches,
                )
                if value is None:
                    # One unrefinable cell leaves the rectangle mixed-depth, which is the
                    # very thing under test -- so the position is dropped rather than
                    # measured with a hole in it.
                    ok = False
                    break
                refined[int(i), int(j)] = value
                deltas.append(value - float(payoff[int(i), int(j)]))
            if not ok:
                break
        if not ok:
            skipped["a cell that cannot be refined"] += 1
            continue

        try:
            mixed = solve(refined)
            restricted = solve(refined[np.ix_(rows, cols)])
        except EquilibriumError:
            skipped["an LP that failed"] += 1
            continue
        seconds = time.perf_counter() - started
        done += 1

        # The restricted game's answer, written on the full menu: zero outside the
        # rectangle, which is what playing it means.
        padded = np.zeros(len(ours), dtype=np.float64)
        padded[rows] = restricted.row_strategy

        tvm = total_variation(depth1.row_strategy, mixed.row_strategy)
        tvr = total_variation(depth1.row_strategy, padded)
        tvb = total_variation(mixed.row_strategy, padded)
        moved_mixed += tvm > 1e-6
        moved_restricted += tvr > 1e-6
        moved_between += tvb > 1e-6

        # What each reading still guarantees against the columns it did not look at,
        # priced by the only values that exist for all of them. The restricted game
        # restricts the OPPONENT too, so its own value is not a forecast of anything --
        # this is. `depth1.value` is the maximin of these same prices, so it is the
        # ceiling by construction and both readings sit at or below it.
        gave_up_mixed.append(float(np.min(mixed.row_strategy @ payoff)) - depth1.value)
        gave_up_restricted.append(float(np.min(padded @ payoff)) - depth1.value)

        cell_deltas.extend(deltas)
        per_position_delta.append(float(np.mean(deltas)))
        mixed_shift.append(float(mixed.value - depth1.value))
        restricted_shift.append(float(restricted.value - depth1.value))
        tv_mixed.append(tvm)
        tv_restricted.append(tvr)
        tv_between.append(tvb)

        print(
            f"  {done:>3} {len(deltas):>5} {np.mean(deltas):>+9.4f} {np.std(deltas):>8.4f} "
            f"{sum(d < 0 for d in deltas) / len(deltas) * 100:>4.0f}% "
            f"{depth1.value:>7.3f} {mixed.value - depth1.value:>+9.4f} "
            f"{restricted.value - depth1.value:>+9.4f} {tvm:>7.3f} {tvr:>7.3f} "
            f"{'yes' if tvb > 1e-6 else 'no':>6} {seconds:>6.1f}",
            flush=True,
        )

        # The same two readings as the shipped code runs them: `passes` passes, the
        # best-response step included. The block above is one pass with the rectangle
        # fixed, which isolates the reading; this is what a game would actually play.
        arms = {}
        for label in ("mixed", "restricted"):
            arm_started = time.perf_counter()
            arms[label] = search(
                reg, pos, ours, theirs, leaf,
                budget=budget,
                depth=2,
                refine=args.refine,
                passes=args.passes,
                sub_limit=args.sub_limit,
                sub_branches=args.sub_branches,
                solve_restricted=label == "restricted",
            )
            arm_seconds[label].append(time.perf_counter() - arm_started)
            arm_subgames[label].append(arms[label].subgames)
            arm_converged[label] += bool(arms[label].converged)
            arm_gave_up[label].append(
                float(np.min(arms[label].equilibrium.row_strategy @ payoff))
                - depth1.value
            )
        arm_tv.append(
            total_variation(
                arms["mixed"].equilibrium.row_strategy,
                arms["restricted"].equilibrium.row_strategy,
            )
        )
        arm_moved += arm_tv[-1] > 1e-6
        arm_shape.append(arms["restricted"].restricted)
        arm_optimism.append(arms["restricted"].optimism)

        if sweep:
            for i in rows:
                for j in cols:
                    sweep_cells[args.sub_limit][(done, int(i), int(j))] = float(
                        refined[int(i), int(j)]
                    )
        for width in sweep:
            if width == args.sub_limit:
                sweep_deltas[width].extend(deltas)
                continue
            for i in rows:
                for j in cols:
                    value, _notes, _solved = _refined_value(
                        reg, pos, ours[int(i)], theirs[int(j)], leaf,
                        budget=budget,
                        sub_limit=width,
                        sub_branches=args.sub_branches,
                    )
                    if value is not None:
                        sweep_deltas[width].append(
                            value - float(payoff[int(i), int(j)])
                        )
                        sweep_cells[width][(done, int(i), int(j))] = value

    if not done:
        raise SystemExit("no position had a mixed equilibrium to refine")

    print(f"\n  {done} positions, {len(cell_deltas)} refined cells")
    print(
        f"  measured {done} of the {taken} recorded positions taken; the rest were skipped "
        "for " + ", ".join(f"{why} ({count})" for why, count in skipped.items()) + "\n"
    )
    print("  the cell bias -- is `depth2 - depth1` a shift or a signal?")
    print(f"    per cell          {mean_pm(cell_deltas)}  sd {np.std(cell_deltas):.4f}")
    print(f"    per position      {mean_pm(per_position_delta)}  "
          f"sd {np.std(per_position_delta):.4f}")
    print(f"    cells below zero  {sum(d < 0 for d in cell_deltas) / len(cell_deltas) * 100:.1f}%")
    print(
        f"    positions whose mean is below zero  "
        f"{sum(d < 0 for d in per_position_delta) / done * 100:.1f}%"
    )
    print("\n  the reading -- the same refined cells, two ways")
    print(f"    mixed value  - depth1   {mean_pm(mixed_shift)}")
    print(f"    restricted   - depth1   {mean_pm(restricted_shift)}")
    print(f"    TV(depth1, mixed)       {np.mean(tv_mixed):.3f}  "
          f"moved {moved_mixed / done * 100:.1f}% of positions")
    print(f"    TV(depth1, restricted)  {np.mean(tv_restricted):.3f}  "
          f"moved {moved_restricted / done * 100:.1f}% of positions")
    print(f"    TV(mixed, restricted)   {np.mean(tv_between):.3f}  "
          f"moved {moved_between / done * 100:.1f}% of positions")
    print(
        "\n  the last line is the one that decides whether a match can see this at all:\n"
        "  it is the share of the measured positions -- depth 1 mixing on both sides --\n"
        "  where changing the reading changes the mixture. It is not a share of every\n"
        "  decision; `pair_divergence` on the board is."
    )
    print("\n  what each reading still guarantees against the columns it never priced")
    print(f"    mixed       - depth1's own guarantee   {mean_pm(gave_up_mixed)}")
    print(f"    restricted  - depth1's own guarantee   {mean_pm(gave_up_restricted)}")
    print(
        "    both are <= 0 by construction: depth 1's strategy is the maximin of exactly\n"
        "    these prices. The gap between the two lines is what the restricted game's\n"
        "    blindness to the unrefined columns costs, before any best-response step."
    )

    print(f"\n  the two arms as they would play, {args.passes} passes each")
    print(f"    TV(mixed, restricted)   {np.mean(arm_tv):.3f}  "
          f"moved {arm_moved / done * 100:.1f}% of positions")
    for label in ("mixed", "restricted"):
        print(
            f"    {label:<11} converged {arm_converged[label] / done * 100:5.1f}%  "
            f"{np.mean(arm_subgames[label]):6.2f} sub-games  "
            f"{np.mean(arm_seconds[label]) * 1000:7.1f} ms  "
            f"gives up {mean_pm(arm_gave_up[label])}"
        )
    shapes = [f"{r}x{c}" for r, c in arm_shape]
    print(
        f"    restricted game   rows {np.mean([r for r, _ in arm_shape]):.2f} "
        f"cols {np.mean([c for _, c in arm_shape]):.2f}, commonest "
        f"{max(set(shapes), key=shapes.count)}, "
        f"optimism left {np.mean(arm_optimism):+.5f}"
    )
    print(
        "    'gives up' is again priced at depth 1, where depth 1's own strategy is the\n"
        "    maximin: it is a deviation measure, not a verdict. The verdict is the board."
    )

    if sweep:
        print("\n  truncation -- the same cells at wider sub-games")
        for width in sweep:
            values = sweep_deltas[width]
            if not values:
                continue
            print(f"    sub-limit {width:>3}   {mean_pm(values)}  n={len(values)}")
        print(
            "    a delta that shrinks as the sub-game widens is the truncated reply,\n"
            "    not the extra ply."
        )
        base = sweep_cells[args.sub_limit]
        print(f"\n  the same cells, paired: value at the wider sub-game - value at {args.sub_limit}")
        for width in sweep:
            if width == args.sub_limit:
                continue
            common = sorted(set(base) & set(sweep_cells[width]))
            if not common:
                continue
            diffs = [sweep_cells[width][key] - base[key] for key in common]
            print(
                f"    {width:>3} - {args.sub_limit:<3}  {mean_pm(diffs)}  "
                f"sd {np.std(diffs):.4f}  lower in {sum(d < 0 for d in diffs) / len(diffs) * 100:.1f}%"
                f"  n={len(diffs)}"
            )
        print(
            "    the column player's reply is what a narrow sub-game truncates, so a\n"
            "    truncation that flatters the row player shows up here as a negative mean."
        )


if __name__ == "__main__":
    main()
