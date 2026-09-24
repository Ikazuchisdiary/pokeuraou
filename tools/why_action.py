"""Where does an action's value come from? A column-by-column decomposition.

`human_baseline.py` answers "did the leaf rank these two actions correctly". It does not
answer "why", and the difference matters: on 2026-09-19 a case was written saying Yawn
could not be justified, and the first thing a person asked on reading the verdict was
whether Yawn was a switch read -- a question the pass/fail line cannot address, because
an equilibrium EV is a sum over the opponent's whole mixture and the sum hides which term
carries it.

So: the same solve, reported per column. For two rows A and B,

    EV(A) - EV(B) = sum_j  y[j] * (payoff[A][j] - payoff[B][j])

and every term is printed with the column that produced it, the opponent's mass on it,
and whether that column contains a switch. If A - B is carried by the switch columns,
"A is a switch read" is a measurement rather than a story; if it is carried by the
columns where nobody switches, the story is wrong however well it reads.

Columns with no mass are printed too when they contain a switch. A switch read is
precisely a bet on a column the solve currently thinks the opponent will not play, so
restricting the table to the support would hide the only evidence that bears on it. For
those the break-even is the number that matters -- how often the opponent must actually
take that line before A overtakes B -- and it is reported for every switch column where
A wins, because "not bad if it lands" is a claim about a probability and deserves one.

`--resolve` runs the turn for the printed cells and says what actually happened: who
fainted, who is drowsy, who is on the field. A decomposition saying "Yawn wins here" is
worth little if Protect blocked the Yawn, and only the resolver knows.

`--save` writes the payoff matrix, so asking a second question of the same solve costs
nothing -- the 24x24 matrix is ten minutes of resolver time on a busy machine, and the
questions arrive one at a time.

    uv run python tools/why_action.py --value data/models/value-gen10.pt --case sash-ko \\
        --resolve --save data/analysis/sash-ko-gen10.npz
    uv run python tools/why_action.py --case sash-ko --load data/analysis/sash-ko-gen10.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from human_baseline import CASES, load_leaf, position_of  # noqa: E402

from pokeuraou import port  # noqa: E402 - Python's resolver until IKA-212
from pokeuraou.actions import SideAction, SwitchAction, target_names  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.names import localiser  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.search import leaf_ranking, search  # noqa: E402


def switch_slots(action: SideAction) -> tuple[int, ...]:
    """Which of this side's slots switch out. Empty for a pure move action."""
    return tuple(
        slot for slot, sub in enumerate(action.slots) if isinstance(sub, SwitchAction)
    )


def outcome(before: Position, after: Position) -> str:
    """One line on what the turn did, in the terms the claims are argued in."""
    notes = []
    for side_index, side in enumerate(after.sides):
        who = "こちら" if side_index == 0 else "相手"
        for mon in side.pokemon:
            was = next(
                (m for m in before.sides[side_index].pokemon if m.species == mon.species),
                None,
            )
            if mon.fainted and (was is None or not was.fainted):
                notes.append(f"{who}{mon.species}が倒れた")
        active = [m.species for m in side.active_pokemon() if m is not None]
        was_active = [
            m.species for m in before.sides[side_index].active_pokemon() if m is not None
        ]
        for species in active:
            if species not in was_active:
                notes.append(f"{who}{species}が出てきた")
    for side_index, side in enumerate(after.sides):
        who = "こちら" if side_index == 0 else "相手"
        for mon in side.pokemon:
            if mon.fainted:
                continue
            before_mon = next(
                (m for m in before.sides[side_index].pokemon if m.species == mon.species),
                None,
            )
            if mon.has_volatile("yawn"):
                notes.append(f"{who}{mon.species}があくび状態")
            if mon.status and (before_mon is None or not before_mon.status):
                notes.append(f"{who}{mon.species}が{mon.status}")
            if before_mon is not None and mon.hp < before_mon.hp:
                notes.append(f"{who}{mon.species} {before_mon.hp}→{mon.hp}")
    return "、".join(notes) if notes else "何も起きない"


def main() -> None:  # noqa: PLR0912, PLR0915 - a report, and splitting it hides the order
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    ap.add_argument("--case", required=True, help="a case name from human_baseline.py")
    ap.add_argument(
        "--rows",
        nargs="+",
        default=None,
        help="choices to compare; defaults to the case's at_least and at_most",
    )
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--resolve", action="store_true", help="run the turn for each cell")
    ap.add_argument("--save", type=Path, default=None, help="write the payoff matrix")
    ap.add_argument("--load", type=Path, default=None, help="reuse a saved matrix")
    ap.add_argument("--locale", default="ja")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    args = ap.parse_args()
    if args.load is None and args.value is None:
        raise SystemExit("--value is required unless --load is given")

    case = next((c for c in CASES if c.name == args.case), None)
    if case is None:
        raise SystemExit(f"no case named {args.case!r}; have {[c.name for c in CASES]}")

    reg = load_regulation(args.format)
    register_mega_stones(reg)
    loc = localiser(reg, args.locale)
    pos, _decision = position_of(case)
    budget = Budget()

    if args.load is not None:
        blob = np.load(args.load, allow_pickle=False)
        payoff = np.asarray(blob["payoff"], dtype=np.float64)
        name = str(blob["name"])
        # The menu is stored as choice strings and the action objects are looked up again
        # from the live legal set, so a saved file that no longer describes this position
        # fails on the lookup instead of silently mislabelling a column.
        ours = _menu_named(reg, pos, 0, [str(s) for s in blob["ours"]])
        theirs = _menu_named(reg, pos, 1, [str(s) for s in blob["theirs"]])
        eq = solve(payoff)
    else:
        encoder = Encoder(reg)
        leaf = load_leaf(list(args.value), encoder)
        name = "+".join(p.stem for p in args.value)
        rank_ours = leaf_ranking(reg, pos, 0, leaf, budget=budget)
        rank_theirs = leaf_ranking(reg, pos, 1, leaf, budget=budget)
        ours = narrow(reg, pos, 0, limit=args.limit, rank=rank_ours).actions
        theirs = narrow(reg, pos, 1, limit=args.limit, rank=rank_theirs).actions
        result = search(reg, pos, ours, theirs, leaf, budget=budget, depth=1)
        eq = result.equilibrium
        payoff = np.asarray(result.payoff, dtype=np.float64)
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.save,
                payoff=payoff,
                name=name,
                ours=np.asarray([a.to_choice() for a in ours]),
                theirs=np.asarray([a.to_choice() for a in theirs]),
            )
            print(f"  saved the matrix to {args.save}")

    y = np.asarray(eq.col_strategy, dtype=np.float64)
    x = np.asarray(eq.row_strategy, dtype=np.float64)
    our_names = [a.to_choice() for a in ours]
    our_targets = target_names(pos, 0)
    their_targets = target_names(pos, 1)

    wanted = list(args.rows) if args.rows else [case.at_least, case.at_most]
    missing = [c for c in wanted if c not in our_names]
    if missing:
        raise SystemExit(f"{missing} not in this leaf's {len(our_names)}-action menu")
    idx = [our_names.index(c) for c in wanted]

    print(f"\n=== {case.name}  ({name})   ゲーム値 {eq.value:.4f}")
    print("\n【1】相手の均衡の支持（質量のある列）")
    _table(idx, wanted, payoff, y, theirs, reg, loc, their_targets, mask=y >= 1e-4)

    # A switch read bets on a column the solve says will not be played. Printing only the
    # support would hide exactly the evidence the claim rests on.
    is_switch = np.asarray([bool(switch_slots(t)) for t in theirs])
    unplayed = is_switch & (y < 1e-4)
    if unplayed.any():
        print("\n【2】交代を含むが質量0の列（交代読みが賭けている先）")
        _table(idx, wanted, payoff, y, theirs, reg, loc, their_targets, mask=unplayed)

    print("\n【3】各手の均衡での位置")
    for i, choice in zip(idx, wanted, strict=True):
        print(
            f"  {choice:22s}  EV {eq.row_ev[i]:.4f}  質量 {x[i]:6.1%}  "
            f"損失 {eq.row_ev_loss[i]:+.4f}   "
            f"{ours[i].describe(reg, loc, our_targets)}"
        )

    if len(idx) == 2:
        a, b = idx
        diff = payoff[a] - payoff[b]
        total = float(y @ diff)
        on_switch = float(y[is_switch] @ diff[is_switch])
        on_stay = float(y[~is_switch] @ diff[~is_switch])
        print(f"\n【4】分解: EV({wanted[0]}) − EV({wanted[1]}) = {total:+.4f}")
        print(f"  相手が交代する列から  {on_switch:+.4f}  （質量 {y[is_switch].sum():.1%}）")
        print(f"  交代しない列から      {on_stay:+.4f}  （質量 {y[~is_switch].sum():.1%}）")
        if abs(total) > 1e-6:
            print(f"  → 交代列の寄与が全体の {on_switch / total:+.0%}")
        else:
            print("  → 両者の EV が等しい（どちらも均衡の台にある）")

        # "Not bad if it lands on the switch-in" is a claim about a frequency. This is the
        # frequency: mix the opponent's equilibrium with the column in question and ask
        # where the two rows cross. Everything else about their play is held proportional,
        # which is the reading that makes the number a statement about that one line.
        print(f"\n【5】損益分岐: 相手がその列を何%で選べば {wanted[0]} が上回るか")
        any_line = False
        for j in np.argsort(-diff):
            if diff[j] <= 1e-6 or y[j] >= 1e-4:
                continue
            p = total / (total - diff[j]) if total < 0 else 0.0
            sw = switch_slots(theirs[j])
            tag = f"[交代 slot {','.join(str(s) for s in sw)}]" if sw else "[交代なし]"
            print(
                f"  {p:6.1%}  差 {diff[j]:+.4f}  {tag} "
                f"{theirs[j].describe(reg, loc, their_targets)}"
            )
            any_line = True
        if not any_line:
            print(f"  {wanted[0]} が上回る列が1つも無い（質量0の列も含めて）")

    if args.resolve:
        print("\n【6】実際に何が起きるか（最頻分岐）")
        shown = np.flatnonzero((y >= 1e-4) | unplayed)
        for j in shown[np.argsort(-y[shown])]:
            print(f"\n  相手: {theirs[j].describe(reg, loc, their_targets)}  （質量 {y[j]:.1%}）")
            for i, choice in zip(idx, wanted, strict=True):
                res = port.turn(reg, pos, [ours[i], theirs[j]], Budget(), full=True)
                best = max(res.outcomes, key=lambda br: br.probability)
                print(
                    f"    {choice:22s} p={best.probability:.2f}  {outcome(pos, best.position)}"
                )


def _table(idx, wanted, payoff, y, theirs, reg, loc, targets, *, mask) -> None:  # noqa: ANN001
    print("   質量  " + "  ".join(f"{c:>9.9s}" for c in wanted) + "   相手の行動")
    for j in np.argsort(-y):
        if not mask[j]:
            continue
        cells = "  ".join(f"{payoff[i, j]:9.4f}" for i in idx)
        sw = switch_slots(theirs[j])
        tag = f"  [交代 slot {','.join(str(s) for s in sw)}]" if sw else ""
        print(f"  {y[j]:5.1%}  {cells}   {theirs[j].describe(reg, loc, targets)}{tag}")


def _menu_named(reg, pos, side, names):  # noqa: ANN001, ANN201
    """The saved menu, as action objects from this position's legal set."""
    from pokeuraou.actions import side_actions

    by_choice = {a.to_choice(): a for a in side_actions(reg, pos, side)}
    missing = [n for n in names if n not in by_choice]
    if missing:
        raise SystemExit(
            f"the saved menu names {missing} which are not legal here; "
            "the file is for a different position -- re-run without --load"
        )
    return [by_choice[n] for n in names]


if __name__ == "__main__":
    main()
