import sys, os
sys.path.insert(0, "C:/tmp/ika413/wt")
sys.path.insert(0, "C:/tmp/ika413/wt/src")
from tests import test_spread_view_oracle as t
from pokeuraou.oracle import Oracle, RandomnessPolicy
from pokeuraou import rustnode
from pokeuraou.regulation import load_regulation
reg = load_regulation(t.FORMAT_ID)
os.environ[rustnode.ENV_ENABLE] = "1"
rustnode.reset()
def case(user, a, b, choice="move 1, move 1"):
    teams = t._teams(a, b, user)
    with Oracle() as o:
        before, after, log = t._play(o, teams, [], choice)
    node = rustnode.node_for(reg)
    acts = t._actions(reg, before, [choice, t.FOES])
    picked = node.resolve(before, acts, t.BUDGET, select=0)
    return after, t._state(picked.position), log
user = t._mon("Excadrill", "Mold Breaker", ["rockslide", "protect", "swordsdance", "protect"], 32, "lifeorb", atk=32)
for fg in ("Friend Guard", "Technician"):
    a, b, log = case(user, t._sturdy(fg, "Maushold"), t._sturdy())
    print(fg, "showdown snorlax", a["p2.snorlax"], "port", b["p2.snorlax"])
user2 = t._mon("Excadrill", "Mold Breaker", ["rockslide", "protect", "swordsdance", "protect"], 32, "lifeorb", atk=32)
for ab in ("Aura Break", "Technician"):
    # Fairy Aura on B's side vs Fairy move: use dazzlinggleam with Mold Breaker user
    u = t._mon("Gholdengo", "Mold Breaker", ["dazzlinggleam", "protect", "swordsdance", "protect"], 32, "lifeorb", spa=32)
    a_, b_, log = case(u, t._sturdy(ab, "Maushold"), t._sturdy("Fairy Aura"))
    print("auraBreakHolderAlive", ab, "showdown snorlax", a_["p2.snorlax"], "port", b_["p2.snorlax"])
