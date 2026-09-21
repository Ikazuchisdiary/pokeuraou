"""The same dominance question, asked under the information the players actually had.

`human_baseline.py` rebuilds a case's position from the log and solves it. The log
records the TRUE position, so that check knows the opponent's bench -- and the game it
came from was played with `information: ["hidden-bench", "hidden-bench"]`, where neither
side knew it. A claim checked against the true bench is a claim about a different game
than the one the model was playing, and a model that loses such a check may simply have
been answering the harder question correctly.

That is not a hypothetical here. On 2026-09-19 `sash-ko` was checked against all 79 legal
replies and Hyper Voice never did worse than Yawn -- against Grimmsnarl and Archaludon,
the two the opponent actually had. Their sheet also listed Swampert and Pelipper, which
our side could not rule out: Flare Blitz is 0.25x on Swampert, and Pelipper's Drizzle
replaces the sun as it lands, taking another half off the Fire damage and switching off
Venusaur's Chlorophyll. A kill that depends on which of four Pokemon walks in is not a
kill, and it is Yawn that does not care.

So: run the comparison once per completion of the opponent's sheet, weighted as the
observer would weight them, and report where the answer changes. The point is not to
overturn the earlier number but to say what it was a number about.

    uv run python tools/hidden_dominance.py --case sash-ko
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from human_baseline import CASES, position_of  # noqa: E402
from what_the_leak_buys import _sheet  # noqa: E402

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.hidden import completions  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402


def score(reg, pos: Position, legal, ours: str, theirs: str, slot: int):  # noqa: ANN001, ANN201
    """Faint mass in the contested slot, damage dealt, damage taken -- and who was hit."""
    res = resolve_turn(reg, pos, [legal[0][ours], legal[1][theirs]], budget=Budget())
    before_them = {m.species: m.hp for m in pos.sides[1].pokemon}
    before_us = {m.species: m.hp for m in pos.sides[0].pokemon}
    dead = total = dealt = taken = 0.0
    name = "-"
    for branch in res.branches:
        total += branch.probability
        occupant = branch.position.sides[1].active_pokemon()[slot]
        if occupant is not None:
            name = occupant.species
            if occupant.fainted:
                dead += branch.probability
        dealt += branch.probability * sum(
            max(0, before_them.get(m.species, m.hp) - m.hp)
            for m in branch.position.sides[1].pokemon
        )
        taken += branch.probability * sum(
            max(0, before_us.get(m.species, m.hp) - m.hp)
            for m in branch.position.sides[0].pokemon
        )
    n = total or 1.0
    return dead / n, dealt / n, taken / n, name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="sash-ko")
    ap.add_argument("--slot", type=int, default=1, help="the contested opponent slot")
    args = ap.parse_args()

    case = next((c for c in CASES if c.name == args.case), None)
    if case is None:
        raise SystemExit(f"no case named {args.case!r}")

    with open(case.source, encoding="utf-8") as handle:
        game = [json.loads(line) for line in handle][case.game_index]
    if game.get("information") != "hidden-bench":
        print(f"  note: this game's information is {game.get('information')!r}")

    reg = load_regulation(game["position"]["format"] if "position" in game else
                          "gen9championsvgc2026regmb")
    register_mega_stones(reg)
    pos, _ = position_of(case)

    sheet = _sheet(reg, game["foeSix"], game["foeTeam"])
    if sheet is None:
        raise SystemExit("could not build the opponent's sheet from the usage prior")
    spread = completions(reg, pos, 1, sheet)
    on_field = [m.species for m in pos.sides[1].active_pokemon() if m is not None]
    print(f"  相手の6匹: {[s.species for s in sheet]}")
    print(f"  場に見えている: {on_field}")
    print(f"  控えの可能性: {len(spread)} 通り\n")

    rows = {"ハイボ+フレドラ": case.at_least, "あくび+フレドラ": case.at_most}
    truth = {m.species for m in pos.sides[1].pokemon} - set(on_field)

    print(f"  {'控え':28s} {'重み':>6s}  {'退き列のハイボKO':>16s} {'あくびKO':>9s}  "
          f"{'あくびが勝つ列':>13s}")
    total_worse = 0
    for item in spread:
        cpos = item.position
        legal = {
            s: {a.to_choice(): a for a in side_actions(reg, cpos, s)} for s in (0, 1)
        }
        bench = [
            m.species
            for m in cpos.sides[1].pokemon
            if m.species not in on_field
        ]
        retreat = [c for c in legal[1] if c.split(", ")[1].startswith("switch")]
        hv_ko = [score(reg, cpos, legal, rows["ハイボ+フレドラ"], c, args.slot)[0]
                 for c in retreat]
        yw_ko = [score(reg, cpos, legal, rows["あくび+フレドラ"], c, args.slot)[0]
                 for c in retreat]
        worse = []
        for c in legal[1]:
            hi = score(reg, cpos, legal, rows["ハイボ+フレドラ"], c, args.slot)
            lo = score(reg, cpos, legal, rows["あくび+フレドラ"], c, args.slot)
            if hi[0] + 1e-9 < lo[0] or hi[1] + 0.5 < lo[1] or hi[2] > lo[2] + 0.5:
                worse.append(c)
        total_worse += len(worse)
        mark = "  ← 実際の控え" if set(bench) == truth else ""
        print(
            f"  {', '.join(sorted(bench)):28s} {item.weight:6.3f}  "
            f"{np.mean(hv_ko):15.1%} {np.mean(yw_ko):9.1%}  "
            f"{len(worse):6d}/{len(legal[1])}{mark}"
        )
        if worse:
            for c in worse[:3]:
                print(f"       あくび有利: {c}")

    print(
        f"\n  合計 {total_worse} 列で、あくびがハイボを上回る（真の控えだけで測ると 0 列）"
    )


if __name__ == "__main__":
    main()
