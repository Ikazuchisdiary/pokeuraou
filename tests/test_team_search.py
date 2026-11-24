"""IKA-443: tools/team_search.py builds legal neighbours of pool teams and reads a store."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("team_search", ROOT / "tools" / "team_search.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)



def _data() -> dict:
    """Two real M-B teams as a pool (they share Garchomp and Choice Scarf, so the clauses
    refuse some swaps)."""
    teams = []
    for name in ("place1", "rizabanadohido"):
        raw = json.loads((ROOT / "configs" / "teams" / f"{name}.json").read_bytes().decode("utf-8"))
        teams.append({"id": name, "name": name, "team": raw["team"]})
    return {"id": "test", "regulation": "gen9championsvgc2026regmb", "teams": teams}


def test_neighbours_are_one_swap_away_legal_and_new() -> None:
    data = _data()
    base = data["teams"][0]
    cands, counts = ts.neighbors(data, [base["id"]], 1)
    assert cands and counts["refused"] > 0  # the clauses refuse some swaps
    own = {ts.set_key(m) for m in base["team"]}
    pool_keys = {tuple(sorted(ts.set_key(m) for m in t["team"])) for t in data["teams"]}
    for cand in cands:
        keys = [ts.set_key(m) for m in cand["team"]]
        assert sum(k not in own for k in keys) == 1
        assert tuple(sorted(keys)) not in pool_keys
        species = [m["species"] for m in cand["team"]]
        items = [m["item"] for m in cand["team"]]
        assert len(set(species)) == 6 and len(set(items)) == 6
    assert len({c["id"] for c in cands}) == len(cands)


def test_the_clauses_refuse_a_duplicate_item_or_species() -> None:
    data = _data()
    base = data["teams"][0]
    cands, _ = ts.neighbors(data, [base["id"]], 1)
    taken_items = {m["item"] for m in base["team"]}
    for cand in cands:
        swap = cand["swap"][0]
        new = cand["team"][swap["slot"]]
        others = [m for i, m in enumerate(base["team"]) if i != swap["slot"]]
        assert new["item"] not in {m["item"] for m in others}
        assert new["item"] not in taken_items - {base["team"][swap["slot"]]["item"]}


def test_team_id_is_stable_and_order_free_after_sorting() -> None:
    members = _data()["teams"][1]["team"]
    a = ts.team_id(sorted(members, key=ts.set_key))
    b = ts.team_id(sorted(reversed(members), key=ts.set_key))
    assert a == b and a.startswith("x")


def test_base_values_count_the_mirror_half_and_both_seats(tmp_path) -> None:  # noqa: ANN001
    for a, b, v in [(0, 0, 0.5), (0, 1, 0.8), (1, 1, 0.5)]:
        (tmp_path / f"{a:03d}-{b:03d}.json").write_bytes(json.dumps({"entry": {"value": v}}).encode())
    assert ts.base_values(tmp_path, 2) == pytest.approx([0.65, 0.35])
