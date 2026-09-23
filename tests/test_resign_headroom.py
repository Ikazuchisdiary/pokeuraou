"""`tools/resign_headroom.py` replays a resignation rule over games that played on.

Three things it computes could be wrong without any error, and each is checked against
something that is not itself:

- which decision a rule fires at, and whether the resignation was false -- on hand-built
  games whose answers are written down, including a replacement inside a streak and a
  resignation by each side;
- the hidden slots and legal pools it replays from a position's JSON -- against
  `hidden.seen_slots` and `narrow` on real positions, including one where a benched
  Pokemon has been revealed, so an agreement is not two functions that both ignore it;
- the split of a game's seconds over its decisions -- on synthetic pools where the cost of
  every decision is known. The fitted split must recover the true saving; the uniform
  split must *not*, because a test that the uniform split also passed would show that the
  pools do not have the shape the fit exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._harness import load_tool

tool = load_tool("resign_headroom")


# ------------------------------------------------------------------- hand-built games


def _mon(slot: int, *, active: bool) -> dict:
    return {
        "slot": slot,
        "hp": 100,
        "maxhp": 100,
        "activeIndex": slot if active else None,
        "fainted": False,
        "status": None,
        "boosts": {},
        "volatiles": [],
        "isMega": False,
    }


def _position() -> dict:
    side = {"pokemon": [_mon(i, active=i < 2) for i in range(4)]}
    return {"sides": [side, {"pokemon": [dict(m) for m in side["pokemon"]]}]}


def _decision(kind: str, turn: int, value: float, rows: int = 12, cols: int = 12) -> dict:
    return {
        "kind": kind,
        "turn": turn,
        "searchValue": value,
        "ownActions": ["a"] * rows,
        "foeActions": ["b"] * cols,
        "position": _position(),
    }


def _game(values: list[float | str], outcome: float, seconds: float) -> dict:
    """Move decisions carry their value; the string "r" is a replacement decision."""
    decisions = []
    turn = 1
    for v in values:
        if v == "r":
            decisions.append(_decision("replacement", turn, 0.5, 2, 2))
            continue
        decisions.append(_decision("move", turn, float(v)))
        turn += 1
    return {
        "outcome": outcome,
        "searchSeconds": [seconds / 2, seconds / 2],
        "decisions": decisions,
        "gameIndex": 0,
    }


def _pool(games: list[dict]):  # noqa: ANN202
    pool, flagged = tool.build_pool(games, legal=lambda _position: (10, 10))
    assert flagged["unlabelled"] == 0
    return pool


def _uniform(pool) -> dict:  # noqa: ANN001
    fit = tool.Fit("uniform", ("decision",), np.array([1.0]), 1.0, 1.0)
    return {"uniform": tool.split_seconds(pool, fit)}


def test_a_replacement_neither_counts_nor_breaks_a_streak() -> None:
    # Side 0 wins. 0.96 and 0.98 are two consecutive MOVE decisions at or over 0.95, with a
    # replacement between them, so k=2 fires on the 0.98 and side 1 is the one resigning.
    game = _game([0.5, 0.96, "r", 0.98, 0.99, 0.995], outcome=1.0, seconds=5.0)
    pool = _pool([game])
    trig = tool.find_triggers(pool, 0.95, 2)
    assert trig.at.tolist() == [2]
    assert trig.resigner.tolist() == [1]

    row = tool.replay(pool, 0.95, 2, _uniform(pool))
    assert row["fired"] == 1
    assert row["false"] == 0
    # Five move decisions at 1 s each (uniform), two of them after the deciding one.
    assert row["saved"]["uniform"] == pytest.approx(2 / 5)
    assert row["moves_cut"] == pytest.approx(2 / 5)
    # Six decisions of any kind; the deciding one is the fourth, so two are lost.
    assert row["rows_cut"] == 2


def test_a_resignation_by_the_side_that_won_is_false_and_a_broken_streak_does_not_fire() -> None:
    # Side 0 dips to 0.04 once and wins anyway. At k=1 side 0 resigns there -- falsely; at
    # k=2 the 0.30 after it breaks the streak and nothing fires.
    game = _game([0.5, 0.04, 0.3, 0.6, 0.9], outcome=1.0, seconds=4.0)
    pool = _pool([game])

    once = tool.replay(pool, 0.95, 1, _uniform(pool))
    assert (once["fired"], once["false"]) == (1, 1)
    assert once["false_by_side"][0] == (1, 1)
    assert once["false_by_side"][1] == (0, 0)
    assert once["saved"]["uniform"] == pytest.approx(3 / 5)

    twice = tool.replay(pool, 0.95, 2, _uniform(pool))
    assert twice["fired"] == 0
    assert twice["saved"]["uniform"] == 0.0


def test_the_threshold_is_inclusive_on_both_sides() -> None:
    lost = _game([0.5, 0.05, 0.2], outcome=0.0, seconds=3.0)
    won = _game([0.5, 0.95, 0.2], outcome=1.0, seconds=3.0)
    pool = _pool([lost, won])
    trig = tool.find_triggers(pool, 0.95, 1)
    assert trig.at.tolist() == [1, 1]
    assert trig.resigner.tolist() == [0, 1]


def test_the_whole_pool_saving_is_weighted_by_what_each_game_cost() -> None:
    # One expensive game that resigns halfway, one cheap game that never does.
    halfway = _game([0.5, 0.99, 0.99, 0.99], outcome=1.0, seconds=8.0)
    never = _game([0.5, 0.5, 0.5, 0.5], outcome=0.0, seconds=2.0)
    pool = _pool([halfway, never])
    row = tool.replay(pool, 0.97, 1, _uniform(pool))
    assert row["saved"]["uniform"] == pytest.approx((8.0 * 2 / 4) / 10.0)
    assert row["moves_cut"] == pytest.approx(2 / 8)


def test_the_interval_is_clopper_pearson() -> None:
    lo, hi = tool.clopper_pearson(0, 50)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 50))
    lo, hi = tool.clopper_pearson(5, 100)
    assert (lo, hi) == pytest.approx((0.01643, 0.11284), abs=5e-5)


def test_the_index_filter_reads_the_tail_and_agrees_with_the_parse(tmp_path) -> None:  # noqa: ANN001
    import json

    path = tmp_path / "games.jsonl"
    games = [_game([0.5, 0.6], 1.0, 1.0) | {"gameIndex": i} for i in range(6)]
    # CRLF, the way the generator's text-mode writes arrive on Windows.
    path.write_bytes(b"".join(json.dumps(g).encode() + b"\r\n" for g in games))
    kept = [g["gameIndex"] for g in tool.iter_games([path], min_index=2, max_index=5)]
    assert kept == [2, 3, 4]
    assert [g["gameIndex"] for g in tool.iter_games([path], limit=2)] == [0, 1]


# --------------------------------------------------------------- against the engine


@pytest.fixture(scope="module")
def engine():  # noqa: ANN201
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.regulation import load_regulation
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    return reg, sheet, position_from_sets


def test_seen_slots_read_off_the_json_are_hidden_seen_slots(engine) -> None:  # noqa: ANN001
    from pokeuraou.hidden import seen_slots

    reg, sheet, position_from_sets = engine
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    for side in (0, 1):
        assert tool.seen_in(pos.to_json()["sides"][side]) == set(seen_slots(pos, side))
        assert set(seen_slots(pos, side)) == {0, 1}, "the fixture's leads moved"

    # Reveal a benched Pokemon four different ways; each must be seen by both functions,
    # and the untouched one beside it must stay hidden in both.
    mon = pos.sides[0].pokemon[2]
    before = (mon.hp, mon.status, dict(mon.boosts), mon.fainted)
    for reveal in ("hp", "status", "boosts", "fainted"):
        if reveal == "hp":
            mon.hp = mon.maxhp - 1
        elif reveal == "status":
            mon.status = "par"
        elif reveal == "boosts":
            mon.boosts = {"atk": 1}
        else:
            mon.fainted = True
        replayed = tool.seen_in(pos.to_json()["sides"][0])
        assert replayed == set(seen_slots(pos, 0)), reveal
        assert 2 in replayed, reveal
        assert 3 not in replayed, reveal
        mon.hp, mon.status, mon.boosts, mon.fainted = before[0], before[1], dict(before[2]), before[3]
    assert tool.seen_in(pos.to_json()["sides"][0]) == {0, 1}


def test_legal_pools_are_the_pools_narrow_builds(engine) -> None:  # noqa: ANN001
    from pokeuraou.narrow import narrow

    reg, sheet, position_from_sets = engine
    count = tool.engine_legal_counter()
    for own, foe in ((sheet[:4], sheet[2:6]), (sheet[1:5], sheet[:4])):
        pos = position_from_sets(reg, own, foe)
        sizes = count(pos.to_json())
        assert sizes == tuple(narrow(reg, pos, s, limit=12).considered for s in (0, 1))
        assert min(sizes) > 12, "a turn-1 pool that fits the menu would not test the ranking term"


# ------------------------------------------------------------- the split of the seconds


def _synthetic_pool(rng: np.random.Generator, games: int, noise: float):  # noqa: ANN202
    """Games whose decisions get cheaper as the benches are revealed and the pools shrink,
    with a known cost per decision -- the shape the workload split exists to recover."""
    starts, values, own, foe, l0, l1, h0, h1, turns, at = [0], [], [], [], [], [], [], [], [], []
    all_turns, true_cost, seconds, outcome = [], [], [], []
    coef = {"decision": 0.02, "legal": 0.0002, "cells": 0.0001, "scored": 0.0003, "twice": 0.00005}
    for _g in range(games):
        n = int(rng.integers(3, 20))
        cost = 0.0
        for t in range(1, n + 1):
            hidden = [int(max(0, 2 - (t - 1) // int(rng.integers(2, 6)))) for _ in (0, 1)]
            legal = [int(max(4, 110 - 9 * t + rng.integers(-10, 10))) for _ in (0, 1)]
            rows = [min(12, legal[0]), min(12, legal[1])]
            cells = rows[0] * rows[1]
            comps = sum(tool.completions_for(h) for h in hidden)
            c = (
                coef["decision"]
                + coef["legal"] * sum(legal)
                + coef["cells"] * cells
                + coef["scored"] * cells * comps
                + coef["twice"] * cells * (hidden == [0, 0])
            )
            true_cost.append(c)
            cost += c
            values.append(min(1.0, 0.5 + 0.06 * t))
            own.append(rows[0])
            foe.append(rows[1])
            l0.append(legal[0])
            l1.append(legal[1])
            h0.append(hidden[0])
            h1.append(hidden[1])
            turns.append(t)
            at.append(t - 1)
            all_turns.append(t)
        starts.append(len(values))
        seconds.append(cost * (1.0 + noise * rng.standard_normal()))
        outcome.append(1)
    pool = tool.Pool(
        seconds=np.asarray(seconds),
        outcome=np.asarray(outcome, dtype=np.int8),
        index=np.arange(games),
        move_start=np.asarray(starts),
        all_start=np.asarray(starts),
        value=np.asarray(values),
        own=np.asarray(own, dtype=np.int16),
        foe=np.asarray(foe, dtype=np.int16),
        legal0=np.asarray(l0, dtype=np.int16),
        legal1=np.asarray(l1, dtype=np.int16),
        hidden0=np.asarray(h0, dtype=np.int8),
        hidden1=np.asarray(h1, dtype=np.int8),
        move_turn=np.asarray(turns, dtype=np.int16),
        move_at=np.asarray(at, dtype=np.int32),
        turn=np.asarray(all_turns, dtype=np.int16),
        kind=np.zeros(len(all_turns), dtype=np.int8),
    )
    return pool, np.asarray(true_cost)


def _true_saving(pool, true_cost: np.ndarray, threshold: float, streak: int) -> float:  # noqa: ANN001
    trig = tool.find_triggers(pool, threshold, streak)
    saved = 0.0
    for g in np.flatnonzero(trig.at >= 0):
        saved += true_cost[pool.move_start[g] + trig.at[g] + 1 : pool.move_start[g + 1]].sum()
    return saved / true_cost.sum()


def test_the_workload_split_recovers_a_known_saving_and_the_uniform_one_does_not() -> None:
    rng = np.random.default_rng(88)
    pool, true_cost = _synthetic_pool(rng, games=3000, noise=0.0)
    fits = {name: tool.fit_cost(pool, name) for name in ("uniform", "workload")}
    assert fits["workload"].r2 == pytest.approx(1.0, abs=1e-9)
    split = {name: tool.split_seconds(pool, fit) for name, fit in fits.items()}
    # A game's decisions always add back up to what the game recorded.
    per_game = np.zeros(pool.games)
    np.add.at(per_game, pool.game_of_move(), split["workload"])
    assert per_game == pytest.approx(pool.seconds)

    truth = _true_saving(pool, true_cost, 0.95, 1)
    row = tool.replay(pool, 0.95, 1, split)
    assert row["saved"]["workload"] == pytest.approx(truth, rel=1e-6)
    # The control: late decisions are cheap here, so counting decisions overstates it.
    assert row["saved"]["uniform"] > truth * 1.5


def test_the_workload_split_survives_per_game_noise() -> None:
    """Per-game noise (the machine's load, a game's branching) is what the real fit sees;
    it costs R^2 but must not move the split, since it does not depend on which decision."""
    rng = np.random.default_rng(89)
    pool, true_cost = _synthetic_pool(rng, games=6000, noise=0.3)
    fit = tool.fit_cost(pool, "workload")
    assert fit.cv_r2 < 0.99
    split = {"workload": tool.split_seconds(pool, fit)}
    row = tool.replay(pool, 0.95, 1, split)
    assert row["saved"]["workload"] == pytest.approx(_true_saving(pool, true_cost, 0.95, 1), rel=0.05)
