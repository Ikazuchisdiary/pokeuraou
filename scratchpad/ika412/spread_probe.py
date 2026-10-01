"""IKA-412: does a spread move's damage to the survivor depend on where the first target stands?

Showdown (oracle, main checkout's dist) and the port, the foe's two positions both ways round.
Foe: a 1-HP Shedinja (dies to the move) and a bulky Blissey. The mover's Hyper Voice hits both.
"""
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "C:/tmp/ika412/wt/src")
sys.path.insert(0, "C:/tmp/ika412/wt")
from pokeuraou import rustnode  # noqa: E402
from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
reg = load_regulation(FORMAT)
register_mega_stones(reg)
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False,
                 enumerate_accuracy=False).with_fixed_roll(0)


def mon(species, ability, moves, spe=32, **sp):
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


def run(first, second, mover_move="makeitrain"):
    mine = [mon("Gholdengo", "Good as Gold", spa=32, moves=[mover_move, "protect", "fakeout", "helpinghand"]),
            mon("Garchomp", "Rough Skin", ["protect", "helpinghand", "fakeout", "protect"], spe=10),
            mon("Milotic", "Marvel Scale", ["protect"] * 4)]
    sh = mon("Maushold", "Friend Guard", ["calmmind", "protect", "fakeout", "protect"], spe=0, hp=0, spd=0, **{"def": 0})
    bl = mon("Snorlax", "Immunity", ["calmmind", "protect", "fakeout", "protect"], spe=0, hp=32)
    foes = {"sh": sh, "bl": bl}
    theirs = [foes[first], foes[second], mon("Milotic", "Marvel Scale", ["protect"] * 4)]
    choices = ["move 1, move 3 %d" % (1 if first == "sh" else 2), "move 1, move 1"]
    with Oracle(script=Path("C:/Users/Ikazuchi/repos/pokeuraou/packages/sim-bridge/dist/cli/oracle.js")) as o:
        h = o.create(FORMAT, mine, theirs, policy=RandomnessPolicy())
        h.step(["team 12", "team 12"])
        before = Position.from_json(h.position)
        h.step(choices)
        assert h.choice_errors == [], h.choice_errors
        log = list(h.log)
        h.close()
    dmg = [line for line in log if "|-damage|" in line and "p2" in line.split("|")[2]]
    for side in before.sides:
        for m in side.pokemon:
            m.stats_override = None
    node = rustnode.node_for(reg)
    acts = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, before, side)}
        acts.append(menu[choices[side]])
    picked = node.resolve(before, acts, BUDGET, select=0)
    assert picked is not None, node.refusal
    after = picked.position
    port = {m.species: (m.hp, m.maxhp) for m in after.sides[1].pokemon[:2]}
    return dmg, port


import os  # noqa: E402

os.environ[rustnode.ENV_ENABLE] = "1"
rustnode.reset()
for order in (("sh", "bl"), ("bl", "sh")):
    dmg, port = run(*order)
    print(order, "showdown:", [re.sub(r"\|-damage\|", "", d) for d in dmg], "port hp:", port)
