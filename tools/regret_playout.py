"""Was the action the narrowing dropped actually better, or only better-looking?

`tools/narrow_regret.py` says the candidate menu often omits the reply the search's own
leaf likes best -- by more than ten points of win probability in one decision in fourteen.
That measurement is made *by the leaf*, so it cannot distinguish two opposite stories:

- the narrowing is dropping real strength, and ranking candidates by the leaf would fix it;
- the leaf over-values actions it has rarely seen, so the ones it prefers are the ones
  self-play never produced, and the narrowing is quietly protecting the search from its
  own blind spot.

They call for opposite work, so this settles which before either is built. For a decision
where the two disagree, the game is played out from that position twice -- once with the
side forced to take the best reply *on the menu*, once with the best reply *in the whole
legal set* -- and the real outcomes are counted. Everything after that one turn is played
normally by both sides.

Paired on purpose. The opponent's action on the forced turn is drawn once from its
equilibrium and used in both arms, and both arms start from the same seed, so the only
deliberate difference between the two games is the action under test.

The count that matters is side 1's win rate per arm. If the leaf-preferred action does not
actually win more, its extra value was the leaf talking about positions it has not seen.

    uv run --group learn python tools/regret_playout.py --shard 0 --shards 14
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import SideAction, side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import EquilibriumError, solve
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE, Objective
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, batched_payoff, replacements_needed, resolve_turn
from pokeuraou.search import search
from pokeuraou.selfplay import (
    GameRecord,
    _advance,
    _do_replacement_node,
    _sample_index,
)
from pokeuraou.teams import load_roster


def playout(
    reg,  # noqa: ANN001
    rng: np.random.Generator,
    pos: Position,
    evaluate,  # noqa: ANN001
    objective: Objective,
    *,
    limit: int,
    forced: tuple[SideAction, SideAction] | None,
    max_turns: int = 40,
) -> float | None:
    """Plays to a result from `pos`, taking `forced` on the first turn.

    Returns side 0's outcome, or `None` if the game did not finish -- which is thrown
    away rather than counted, the same rule the generator uses.

    Reuses self-play's own replacement and branch-sampling helpers rather than
    reimplementing them: a playout that advanced positions differently from the generator
    would be measuring the difference between two loops.
    """
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="playout")
    leaves = (evaluate, evaluate)
    first = True
    for _step in range(max_turns * 2):
        if pos.ended:
            break
        owed = replacements_needed(pos)
        if any(owed[0]) or any(owed[1]):
            pos = _do_replacement_node(reg, rng, pos, owed, record, evaluate)
            continue
        if first and forced is not None:
            chosen = list(forced)
            first = False
        else:
            ours = narrow(reg, pos, 0, limit=limit).actions
            theirs = narrow(reg, pos, 1, limit=limit).actions
            if not ours or not theirs:
                break
            try:
                found = search(
                    reg, pos, ours, theirs, evaluate, budget=Budget.matrix()
                )
            except EquilibriumError:
                break
            chosen = [
                ours[_sample_index(rng, found.equilibrium.row_strategy)],
                theirs[_sample_index(rng, found.equilibrium.col_strategy)],
            ]
        result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
        advanced = _advance(reg, rng, result, record, leaves, objective)
        if advanced is None:
            break
        pos = advanced
    if pos.ended and pos.winner is not None:
        return 1.0 if pos.winner == pos.sides[0].id else 0.0
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen4"))
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--games", type=int, default=15, help="recorded games screened")
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--playouts", type=int, default=16, help="per arm, per position")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(1)
    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    reg = load_roster(args.roster).reg
    register_mega_stones(reg)

    files = sorted(glob.glob(str(args.dir / "games-seed*.jsonl")))
    if not files:
        raise SystemExit(f"no games under {args.dir}")
    records = []
    with open(files[0], encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % args.shards != args.shard:
                continue
            records.append(json.loads(line))
            if len(records) >= args.games:
                break

    leaf_name = records[0]["searchObjective"].split(":", 1)[1]
    encoder = Encoder(reg)
    net, _meta = load_model(Path("data/models") / f"{leaf_name}.pt", encoder)
    evaluate = BatchedValue(net, encoder).__call__
    objective = HP_SHARE

    rng = np.random.default_rng(args.seed + args.shard)
    rows = []
    for record in records:
        moves = [d for d in record["decisions"] if d.get("kind") == "move"]
        if not moves:
            continue
        pick = rng.choice(len(moves), size=min(args.per_game, len(moves)), replace=False)
        for which in pick:
            decision = moves[int(which)]
            pos = Position.from_json(decision["position"])
            row = narrow(reg, pos, 0, limit=args.limit).actions
            col = narrow(reg, pos, 1, limit=args.limit).actions
            if not row or not col:
                continue
            payoff, _n = batched_payoff(
                reg, pos, row, col, evaluate, budget=Budget.matrix()
            )
            try:
                equilibrium = solve(payoff)
            except EquilibriumError:
                continue
            everything = side_actions(reg, pos, 1)
            full, _n = batched_payoff(
                reg, pos, row, everything, evaluate, budget=Budget.matrix()
            )
            ev = equilibrium.row_strategy @ full
            offered = {a.to_choice() for a in col}
            on_menu = [i for i, a in enumerate(everything) if a.to_choice() in offered]
            if not on_menu:
                continue
            best_menu = int(min(on_menu, key=lambda i: ev[i]))
            best_legal = int(np.argmin(ev))
            gap = float(ev[best_menu] - ev[best_legal])
            if gap <= args.threshold:
                continue

            # Side 0's action is drawn once and reused, so the arms differ only in the
            # action under test rather than in what it was answering.
            ours = row[_sample_index(rng, equilibrium.row_strategy)]
            wins = {"menu": [0.0, 0], "legal": [0.0, 0]}
            for trial in range(args.playouts):
                for tag, index in (("menu", best_menu), ("legal", best_legal)):
                    # Same seed for the two arms of one trial: the games diverge from the
                    # forced action onwards, but they start from the same stream.
                    inner = np.random.default_rng(
                        [args.seed, args.shard, len(rows), trial]
                    )
                    outcome = playout(
                        reg,
                        inner,
                        pos.copy(),
                        evaluate,
                        objective,
                        limit=args.limit,
                        forced=(ours, everything[index]),
                    )
                    if outcome is None:
                        continue
                    # Side 1's result, because side 1 is the one choosing here.
                    wins[tag][0] += 1.0 - outcome
                    wins[tag][1] += 1
            rows.append(
                {
                    "gap": gap,
                    "ev_menu": float(ev[best_menu]),
                    "ev_legal": float(ev[best_legal]),
                    "menu_wins": wins["menu"][0],
                    "menu_games": wins["menu"][1],
                    "legal_wins": wins["legal"][0],
                    "legal_games": wins["legal"][1],
                    "menu_action": everything[best_menu].describe(reg),
                    "legal_action": everything[best_legal].describe(reg),
                }
            )
            print(
                f"  gap {gap:.3f}  menu {wins['menu'][0]:.0f}/{wins['menu'][1]}  "
                f"legal {wins['legal'][0]:.0f}/{wins['legal'][1]}",
                flush=True,
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    menu = sum(r["menu_wins"] for r in rows), sum(r["menu_games"] for r in rows)
    legal = sum(r["legal_wins"] for r in rows), sum(r["legal_games"] for r in rows)
    print(f"\n{len(rows)} positions with gap > {args.threshold}")
    if menu[1] and legal[1]:
        print(f"  menu-best  {menu[0] / menu[1] * 100:.1f}%  ({menu[1]} playouts)")
        print(f"  legal-best {legal[0] / legal[1] * 100:.1f}%  ({legal[1]} playouts)")


if __name__ == "__main__":
    main()
