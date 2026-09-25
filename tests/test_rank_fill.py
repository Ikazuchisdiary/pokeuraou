"""IKA-268: how a leaf ranking fills its cells is a per-agent setting.

What these tests hold:

- **the label**: ``refs<N>`` and ``refs<N>-fast`` parse, anything else stops;
- **it reaches the ranking**: `_menus` hands `leaf_ranking` the reply count and the budget
  the label names, and the default is exactly what ranked before (two replies, the node's
  own budget object);
- **it follows the arm into either seat**: in a pool match the record says each side
  played its own arm's fill, and two arms that differ only in the fill build two menus per
  decision where identical arms build one;
- **it is recorded only when it differs**: a record at the old fill is byte for byte what
  it was, and the agent's name changes only for a leaf ranking filled another way.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.provenance import LEGACY_RANK_FILL, agent_name, provenance
from pokeuraou.search import DEFAULT_RANK_FILL, DEFAULT_REFERENCES, parse_rank_fill

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _recording_ranking(monkeypatch):  # noqa: ANN001, ANN202
    """Stub `leaf_ranking`: records (side, references, budget) and orders nothing."""
    asked: list[tuple[int, int, Budget]] = []

    def stub(reg, at, side, evaluate, *, budget, references=DEFAULT_REFERENCES):  # noqa: ANN001, ANN202, ARG001
        asked.append((side, references, budget))
        return lambda pool, _scored=None: np.zeros(len(pool))

    monkeypatch.setattr(selfplay, "leaf_ranking", stub)
    return asked


def test_the_label_parses_and_a_bad_one_stops() -> None:
    assert parse_rank_fill("refs2") == (2, False)
    assert parse_rank_fill("refs1") == (1, False)
    assert parse_rank_fill("refs2-fast") == (2, True)
    assert f"refs{DEFAULT_REFERENCES}" == DEFAULT_RANK_FILL
    for bad in ("refs0", "fast", "refs", "refs2-slow", "2"):
        with pytest.raises(ValueError, match="rank fill"):
            parse_rank_fill(bad)


def test_the_menus_rank_with_the_fill_they_are_given(pool, monkeypatch) -> None:  # noqa: ANN001
    asked = _recording_ranking(monkeypatch)
    reg = pool.reg
    team = list(pool.teams[0].sets)
    pos = selfplay.position_from_sets(reg, team[:4], team[:4], rng=np.random.default_rng(0))
    budget = Budget.matrix()

    # Null: the default is what ranked before -- the default reply count and the very
    # budget object the node fills with.
    selfplay._menus(reg, pos, (4, 4), _stub, budget, True)
    assert [(s, r) for s, r, _b in asked] == [(0, DEFAULT_REFERENCES), (1, DEFAULT_REFERENCES)]
    assert all(b is budget for _s, _r, b in asked)

    asked.clear()
    selfplay._menus(reg, pos, (4, 4), _stub, budget, True, rank_fill="refs1")
    assert [(s, r) for s, r, _b in asked] == [(0, 1), (1, 1)]
    assert all(b is budget for _s, _r, b in asked)

    asked.clear()
    selfplay._menus(reg, pos, (4, 4), _stub, budget, True, rank_fill="refs2-fast")
    assert [r for _s, r, _b in asked] == [2, 2]
    assert all(b == Budget.fast() for _s, _r, b in asked)
    # Positive control: the fast budget is not the matrix one, so the assertion above
    # could have failed.
    assert Budget.fast() != budget

    # The damage ranking fills nothing, whatever the label.
    asked.clear()
    selfplay._menus(reg, pos, (4, 4), _stub, budget, False, rank_fill="refs1-fast")
    assert asked == []


def _arm(pool, fill: str, solver) -> PoolArm:  # noqa: ANN001
    return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2, rank_by_leaf=True,
                   rank_fill=fill)


def test_the_fill_follows_its_arm_into_either_seat(pool, monkeypatch) -> None:  # noqa: ANN001
    asked = _recording_ranking(monkeypatch)
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm(pool, "refs1-fast", solver), _arm(pool, "refs2", solver)
    for which in (0, 1):
        asked.clear()
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=268, game_index=0, which=which,
            hide_bench=True, max_turns=2,
        )
        assert record.rank_fill[which] == "refs1-fast"
        assert record.rank_fill[1 - which] == "refs2"
        assert sides["rank_fills"][which] == "refs1-fast"
        moves = sum(d.kind == "move" for d in record.decisions)
        assert moves > 0
        # Two constructions per move decision, one per arm, each ranking both sides.
        assert len(asked) == 4 * moves
        assert sorted({(r, b == Budget.fast()) for _s, r, b in asked}) == [(1, True), (2, False)]
        payload = record.to_json(objective="value:arm", search_limit=(2, 2))
        assert payload["rankFill"] == record.rank_fill

    # Null: identical arms build one menu per decision, at the one fill.
    asked.clear()
    same = _arm(pool, "refs2", solver)
    record, _ = pool_match_game(
        pool.reg, pool, (same, same), seed=268, game_index=0, which=0,
        hide_bench=True, max_turns=2,
    )
    moves = sum(d.kind == "move" for d in record.decisions)
    assert moves > 0 and len(asked) == 2 * moves
    assert {r for _s, r, _b in asked} == {2}
    assert "rankFill" not in record.to_json(objective="value:arm", search_limit=(2, 2))


def test_the_fill_is_recorded_only_when_it_differs() -> None:
    base = dict(seat="s", leaves=("m", "m"), limits=(12, 12), rankings=("leaf", "leaf"),
                information=("hidden-bench", "hidden-bench"))
    old = provenance("pool-match", **base)
    assert "rankFills" not in old
    assert provenance("pool-match", **base, rank_fills=(LEGACY_RANK_FILL,) * 2) == old
    new = provenance("pool-match", **base, rank_fills=("refs1", LEGACY_RANK_FILL))
    assert new["rankFills"] == ["refs1", LEGACY_RANK_FILL]
    assert agent_name(new, 0) == agent_name(old, 0) + "/rankfill:refs1"
    assert agent_name(new, 1) == agent_name(old, 1)
    # A damage ranking fills no cells, so the label does not make it another agent.
    damage = provenance("pool-match", **{**base, "rankings": ("damage", "damage")},
                        rank_fills=("refs1", "refs1"))
    assert "/rankfill" not in agent_name(damage, 0)


def test_the_menu_tool_counts_what_a_fill_drops() -> None:
    from ._harness import load_tool

    tool = load_tool("rank_fill_menus")
    played = ["a", "b", "c", "d"]
    # Null: the same menu drops nothing.
    same = tool.compare(played, list(played), [0.5, 0.5, 0.0, 0.0])
    assert same == {"same_set": True, "same_order": True, "kept": 4, "size": 4,
                    "dropped_mass": 0.0}
    # Reordered: the same set, another order, nothing dropped.
    moved = tool.compare(played, ["b", "a", "c", "d"], [0.5, 0.5, 0.0, 0.0])
    assert moved["same_set"] and not moved["same_order"] and moved["dropped_mass"] == 0.0
    # Positive control: dropping a supported action counts its mass, an unsupported one not.
    dropped = tool.compare(played, ["a", "c", "d", "e"], [0.25, 0.75, 0.0, 0.0])
    assert dropped["kept"] == 3 and not dropped["same_set"]
    assert dropped["dropped_mass"] == pytest.approx(0.75)


def test_the_menu_tool_counts_leaves_once_a_decision() -> None:
    """IKA-270: the fill's cells and leaves ride on side 0's row, so a sum counts them once."""
    from ._harness import load_tool

    tool = load_tool("rank_fill_menus")
    base = {"same_set": True, "same_order": True, "kept": 2, "size": 2, "dropped_mass": 0.0}
    rows = [
        {**base, "game": 0, "decision": d, "side": s, "fill": fill,
         "cells": (cells if s == 0 else 0), "leaves": (leaves if s == 0 else 0)}
        for fill, cells, leaves in (("refs2", 10, 28), ("refs1", 5, 14))
        for d in (0, 1)
        for s in (0, 1)
    ]
    lines = tool.summarise(rows, ["refs2", "refs1"])
    assert lines[0].endswith("cells/decision  leaves/decision  leaves/cell")
    assert lines[1].split()[-3:] == ["10.0", "28.0", "2.800"]
    assert lines[2].split()[-3:] == ["5.0", "14.0", "2.800"]
