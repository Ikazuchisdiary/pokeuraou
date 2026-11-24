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


def test_a_set_with_its_moves_reordered_or_a_gender_is_the_same_play() -> None:
    member = _data()["teams"][0]["team"][0]
    other = {**member, "moves": list(reversed(member["moves"])), "gender": "M"}
    assert ts.play_key(member) == ts.play_key(other)
    assert ts.set_key(member) != ts.set_key(other)  # the sheet's own identity still differs


def test_extra_sets_join_the_swaps_and_new_only_keeps_just_them() -> None:
    data = _data()
    base, other = data["teams"]
    extra = [{"id": "x", "name": "x", "team": other["team"]}]
    pool = {**data, "teams": [base]}
    plain, _ = ts.neighbors(pool, [base["id"]], 1)
    assert plain == []  # one team alone has nothing to swap in
    cands, _ = ts.neighbors(pool, [base["id"]], 1, extra=extra)
    assert cands and {c["swap"][0]["from"] for c in cands} == {"x"}
    # new_only drops every set the pool already has: the extra team's sets are all new here
    same, _ = ts.neighbors(pool, [base["id"]], 1, extra=extra, new_only=True)
    assert [c["id"] for c in same] == [c["id"] for c in cands]
    none, _ = ts.neighbors(data, [base["id"]], 1, extra=extra, new_only=True)
    assert none == []  # every extra set is a pool set once the pool holds that team
    # an extra team can be a base
    from_extra, _ = ts.neighbors(pool, ["x"], 1, extra=extra)
    assert from_extra and all(c["base"] == "x" for c in from_extra)
