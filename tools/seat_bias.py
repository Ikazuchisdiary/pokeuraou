"""Is either seat structurally favoured? Measured exactly, with no games played.

A mirror -- the same four with the same spreads on both sides -- makes the answer an
identity rather than a statistic. ``hp-share`` is antisymmetric by construction
(``ours / (ours + theirs)``, and 1 or 0 once decided), so for any pair of actions

    M[i][j] + M[j][i] = 1

must hold to floating point, where row action ``i`` and column action ``i`` are the *same*
choice on opposite sides of the field. Anything else is a seat asymmetry in the machinery:
a speed tie broken by side index, a residual applied in party order, a narrowing that
offers one seat a candidate it does not offer the other.

This matters beyond the mirror. Every training game has our roster on side 0 and a
tournament team on side 1, so a seat effect does not cancel -- it is a constant added to
every label in the dataset.

Playing games measures the same thing far more expensively: 420 games per arm put the
estimate at +-2.7 points, which was not enough to tell -0.07 from zero. This is exact.

    uv run python tools/seat_bias.py --limit 16 --text /c/tmp/seat.txt
    uv run python tools/seat_bias.py --positions data/selfplay-gen2 --sample 200

The mirror only checks turn 1, where the position is symmetric by construction. The
stronger form needs no symmetry at all: **swapping the two sides must swap the answer.**
For any position P and any action pair (a, b),

    V(P, [a, b]) + V(swap(P), [b, a]) = 1

because the resolver has no business knowing which seat is which. ``--positions`` runs that
identity over recorded mid-game positions -- statuses, boosts, weather, depleted benches in
the proportions the search really meets -- which is where an asymmetry would hide after
turn 1 looked clean.

Three numbers come out, and they localise the fault:

- **the action lists**: if narrowing offers the two seats different candidates, nothing
  below is meaningful and that is the bug;
- **max |M[i][j] + M[j][i] - 1|**: the resolver and the objective together. Zero is the
  requirement;
- **the equilibrium value**: forced to exactly 0.500 for a symmetric game, so it is a
  check on the LP as well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.regulation import Regulation
from pokeuraou.resolve import Budget, resolve_turn, turn_expectation
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import all_selections, load_roster


def budget_for(name: str) -> Budget:
    if name == "exact":
        return Budget.exact()
    if name == "fast":
        return Budget.fast()
    return Budget.matrix()


def swap_check(reg: Regulation, args: argparse.Namespace) -> None:
    """The swap identity over recorded mid-game positions.

    No symmetry is assumed of the position, only of the machinery: whatever happens when
    our actions sit on side 0 must happen mirrored when they sit on side 1. A violation is
    a seat asymmetry in the resolver -- a tie broken by side index, a residual applied in
    party order -- and it lands on every training label, because side 0 is always ours.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from budget_effect import load_positions

    objective = OBJECTIVES[args.objective]
    budget = budget_for(args.budget)
    positions = load_positions(args.positions, args.min_turn, args.seed)[: args.sample]
    if not positions:
        raise SystemExit(f"no recorded positions in {args.positions}")

    rng = np.random.default_rng(args.seed)
    residuals: list[float] = []
    worst: list[tuple[float, str]] = []
    checked = 0
    for pos in positions:
        row = narrow(reg, pos, 0, limit=args.limit).actions
        col = narrow(reg, pos, 1, limit=args.limit).actions
        if not row or not col:
            continue
        mirror = pos.swapped()
        # The swapped position's own narrowing must offer the swapped candidates; looking
        # them up by choice string rather than by index is what makes that a check too.
        mirror_row = {a.to_choice(): a for a in narrow(reg, mirror, 0, limit=args.limit).actions}
        mirror_col = {a.to_choice(): a for a in narrow(reg, mirror, 1, limit=args.limit).actions}
        for _ in range(args.pairs):
            ours = row[int(rng.integers(len(row)))]
            theirs = col[int(rng.integers(len(col)))]
            a, b = ours.to_choice(), theirs.to_choice()
            if a not in mirror_col or b not in mirror_row:
                worst.append((float("nan"), f"絞り込みが入れ替えで一致しない: {a} / {b}"))
                continue
            forward, _ = turn_expectation(
                reg, resolve_turn(reg, pos, [ours, theirs], budget=budget), objective
            )
            backward, _ = turn_expectation(
                reg,
                resolve_turn(reg, mirror, [mirror_row[b], mirror_col[a]], budget=budget),
                objective,
            )
            residual = forward + backward - 1.0
            residuals.append(residual)
            checked += 1
            worst.append((abs(residual), f"{residual:+.5f}  ターン{pos.turn}  自陣 {a} × 相手 {b}"))

    out: list[str] = []
    out.append(
        f"■ 座席の入れ替え検査（記録局面 {len(positions)} 個、{checked} 対、"
        f"{args.objective}、{args.budget} 予算、幅 {args.limit}）"
    )
    out.append("  V(P,[a,b]) + V(入れ替えP,[b,a]) = 1 が要件。局面の対称性は仮定しない")
    if residuals:
        arr = np.array(residuals)
        out.append(
            f"  最大 |ずれ| {np.abs(arr).max():.3e}、平均 {arr.mean():+.3e}、"
            f"0 でないセル {int((np.abs(arr) > 1e-9).sum())}/{arr.size}"
        )
    named = [w for w in worst if not np.isnan(w[0]) and w[0] > 1e-9]
    named.sort(key=lambda w: -w[0])
    for _size, line in named[: args.worst]:
        out.append(f"    {line}")
    broken = [w for w in worst if np.isnan(w[0])]
    for _size, line in broken[:5]:
        out.append(f"    ! {line}")
    if not named and not broken:
        out.append("    ずれているセルは無い")
    text = "\n".join(out)
    print(text, file=sys.stderr)
    if args.text is not None:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if residuals and np.abs(np.array(residuals)).max() < 1e-9 else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--selection", type=int, default=None, help="index into the ordered 90")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    ap.add_argument("--budget", default="matrix", choices=("matrix", "fast", "exact"))
    ap.add_argument("--worst", type=int, default=6)
    ap.add_argument(
        "--positions",
        type=Path,
        default=None,
        help="recorded games directory: run the swap identity on mid-game positions "
        "instead of the turn-1 mirror",
    )
    ap.add_argument("--sample", type=int, default=100, help="positions to check")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=4, help="action pairs per position")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--text", type=Path, default=None)
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    if args.positions is not None:
        swap_check(reg, args)
        return
    selections = list(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    which = selections[args.selection % len(selections)] if args.selection is not None else selections[0]
    sets = [roster.sets[i] for i in which]
    pos = position_from_sets(reg, sets, sets)
    objective = OBJECTIVES[args.objective]
    budget = budget_for(args.budget)

    row = narrow(reg, pos, 0, limit=args.limit)
    col = narrow(reg, pos, 1, limit=args.limit)
    ours = {a.to_choice(): a for a in row.actions}
    theirs = {a.to_choice(): a for a in col.actions}

    out: list[str] = []
    out.append(
        f"■ 座席バイアス（ミラー、{args.objective}、{args.budget} 予算、幅 {args.limit}）"
    )
    out.append(
        "  自陣 = 相手 = "
        + "+".join(reg.species[sets[i].species].name for i in range(2))
        + " / "
        + "+".join(reg.species[sets[i].species].name for i in range(2, len(sets)))
    )
    only_ours = sorted(set(ours) - set(theirs))
    only_theirs = sorted(set(theirs) - set(ours))
    out.append(f"  候補手 自陣 {len(ours)} / 相手 {len(theirs)}")
    if only_ours or only_theirs:
        out.append(
            "  ! 絞り込みが座席で違う。以下が片側にしか出ていない（これ自体が不具合）:"
        )
        for choice in (only_ours + only_theirs)[:10]:
            side = "自陣のみ" if choice in ours else "相手のみ"
            out.append(f"      {side}: {choice}")
    shared = [c for c in ours if c in theirs]
    if not shared:
        out.append("  共通の候補手が無いので比較できない")
        print("\n".join(out), file=sys.stderr)
        raise SystemExit(1)

    size = len(shared)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, a in enumerate(shared):
        for j, b in enumerate(shared):
            result = resolve_turn(reg, pos, [ours[a], theirs[b]], budget=budget)
            value, _flags = turn_expectation(reg, result, objective)
            matrix[i, j] = value

    residual = matrix + matrix.T - 1.0
    worst = np.dstack(np.unravel_index(np.argsort(-np.abs(residual).ravel()), residual.shape))[0]
    out.append(
        f"  最大 |M[i][j] + M[j][i] - 1| = {np.abs(residual).max():.3e}"
        "（0 が要件。反対称な評価軸とミラーなら恒等式）"
    )
    out.append(f"  対角の平均 {matrix.diagonal().mean():.6f}（同じ手同士なので 0.5 が要件）")
    shown = 0
    for i, j in worst:
        if i == j or abs(residual[i, j]) < 1e-9:
            continue
        out.append(
            f"    {residual[i, j]:+.4f}  自陣 {shared[i]}  ×  相手 {shared[j]}"
            f"  (M={matrix[i, j]:.4f}, 鏡={matrix[j, i]:.4f})"
        )
        shown += 1
        if shown >= args.worst:
            break
    if shown == 0:
        out.append("    ずれているセルは無い")

    equilibrium = solve(matrix)
    out.append(
        f"  均衡値 {equilibrium.value:.6f}（対称ゲームなので 0.500000 が要件、"
        f"誤差 {abs(equilibrium.value - 0.5):.2e}）"
    )
    out.append(f"  双対ギャップ {equilibrium.duality_gap:.2e}")

    text = "\n".join(out)
    print(text, file=sys.stderr)
    if args.text is not None:
        args.text.parent.mkdir(parents=True, exist_ok=True)
        args.text.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if np.abs(residual).max() < 1e-9 else 1)


if __name__ == "__main__":
    main()
