import sys, os
sys.path.insert(0, "C:/tmp/ika413/wt"); sys.path.insert(0, "C:/tmp/ika413/wt/src")
from tests import test_spread_view_oracle as t
from pokeuraou.oracle import Oracle
from pokeuraou import rustnode
from pokeuraou.regulation import load_regulation
reg = load_regulation(t.FORMAT_ID)
os.environ[rustnode.ENV_ENABLE] = "1"; rustnode.reset()
def run(teams, setup, choice):
    with Oracle() as o:
        before, after, log = t._play(o, teams, setup, choice)
    node = rustnode.node_for(reg)
    acts = t._actions(reg, before, [choice, t.FOES])
    picked = node.resolve(before, acts, t.BUDGET, select=0)
    return after, t._state(picked.position)
for name, user, choice in [("makeitrain", t._user(), t.MOVE2),
                           ("clangingscales", t._mon("Kommo-o","Soundproof",["clangingscales","protect","nastyplot","protect"],32,"lifeorb",spa=32), "move 1, move 2 -1"),
                           ("matchagotcha", t._user(["matchagotcha","protect","nastyplot","protect"]), "move 1, move 2 -1")]:
    a, b = run(t._teams(t._sturdy(), t._sturdy(), user), [], choice)
    print(name, "MATCH" if a == b else "DIFF", a["p2.snorlax"], b["p2.snorlax"])
