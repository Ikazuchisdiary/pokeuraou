"""IKA-413: the default path is byte-identical before and after the per-arm binary.

    PYTHONPATH=<tree>/src python identity.py <tree> OUT.json

Plays 2 x N pool-match games (stub leaves, hide_bench, both seats) in the tree's own code
and writes each game's JSON (less the clock) so two trees can be diffed.
"""
import hashlib
import json
import sys

tree, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
sys.path.insert(0, tree)
sys.path.insert(0, tree + "/src")
import tempfile
from pathlib import Path

from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from tests.test_board_belief_once import _other
from tests.test_poolplay import _stub, _variants, _write_pool

path = _write_pool(Path(tempfile.mkdtemp()) / "p.json", _variants())
pool = load_pool(path)
register_mega_stones(pool.reg)


def arms():
    a = PoolArm(name="a", evaluate=_stub, solver=SolvedSelections(pool.reg, pool.teams, _stub),
                limit=4, rank_by_leaf=True, rank_fill="refs2")
    b = PoolArm(name="b", evaluate=_other, solver=SolvedSelections(pool.reg, pool.teams, _other),
                limit=4, rank_by_leaf=True, rank_fill="refs2")
    return a, b


rows = []
for g in range(n):
    for which in (0, 1):
        record, sides = pool_match_game(
            pool.reg, pool, arms(), seed=7, game_index=g, which=which, hide_bench=True, max_turns=6,
        )
        payload = record.to_json(objective="value:arm", search_limit=(4, 4))
        payload.pop("searchSeconds", None)
        payload.pop("engine", None)
        text = json.dumps(payload, sort_keys=True)
        rows.append({"g": g, "which": which, "turns": record.turns, "outcome": record.outcome,
                     "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
json.dump(rows, open(out, "w"))
print(len(rows), "games", sum(r["turns"] for r in rows), "turns")
