"""IKA-413: an older build, read through `binary_scope`, answers as the older build does.

    PYTHONPATH=<tree>/src python binary_scope_check.py <tree> <old exe>

The first case of tests/test_spread_view_oracle.py (a Friend Guard holder knocked out first by
Hyper Voice, the second target Snorlax): Showdown 209, the port before IKA-413's fix 189. The
process's own port is the fixed one; inside `binary_scope(old)` the same call must give the
old answer, and outside it the new one again.
"""
import os
import sys

tree, old = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
sys.path.insert(0, tree + "/src")
from pokeuraou import rustnode
from pokeuraou.oracle import Oracle
from pokeuraou.regulation import load_regulation
from tests import test_spread_view_oracle as t

reg = load_regulation(t.FORMAT_ID)
os.environ[rustnode.ENV_ENABLE] = "1"
rustnode.reset()
name = "friend-guard-holder-hit-first-and-knocked-out"
teams, setup, choice, shown, victim = t.CASES[name]
with Oracle() as o:
    before, showdown, log = t._play(o, teams, setup, choice)
actions = t._actions(reg, before, [choice, t.FOES])


def port_snorlax():
    node = rustnode.node_for(reg)
    picked = node.resolve(before, actions, t.BUDGET, select=0)
    return t._state(picked.position)["p2.snorlax"][0], node.binary


print("showdown", showdown["p2.snorlax"][0])
print("own port", *port_snorlax())
with rustnode.binary_scope(old):
    print("scoped old", *port_snorlax())
print("own port again", *port_snorlax())
