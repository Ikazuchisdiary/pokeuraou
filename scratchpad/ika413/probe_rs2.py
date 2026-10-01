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
    return after, t._state(picked.position), log
fang = t._mon("Snorlax", "Immunity", t.FANG, hp=32)
pika = t._mon("Pikachu", "Static", t.SWIPE, 32, atk=32)
# B also Rough Skin (Garchomp with Super Fang as move1)
g2 = t._mon("Garchomp", "Rough Skin", t.FANG, hp=32, **{"def": 32})
for name, teams in [("both rough skin", t._teams(t._rough(), g2, pika)),
                    ("B rocky helmet", t._teams(t._rough(), t._mon("Snorlax","Immunity", t.FANG, item="rockyhelmet", hp=32), pika)),
                    ("A helmet B rough", t._teams(t._mon("Snorlax","Immunity", t.IDLE_MOVES, item="rockyhelmet", hp=32,**{"def":32}), g2, pika)),
                   ]:
    a, b, log = run(teams, [t.FANGED]*4, "move 1, move 1")
    print(name, "MATCH" if a == b else "DIFF")
    if a != b:
        print(" sd", a); print(" rs", b)
    print([l for l in log if "damage" in l and ("p1a" in l or "faint" in l)][-6:])
