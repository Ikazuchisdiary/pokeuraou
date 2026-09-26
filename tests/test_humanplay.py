"""IKA-330: one game against a person, on a clock (`pokeuraou.humanplay`).

What these hold:

- **the budget rule** (`plan_move`): wider with more seconds, never past the legal
  actions, the rest to the deepening, none under `MIN_DEEPEN_MS` or with ``width_only``;
  an unmeasured core count reads the nearest measured one below it;
- **the count clock replays**: the same seed and the same stand-in person give the same
  record, byte for byte; the record's own inputs, read back as a script, give it again;
  another seed does not (the control that the comparison can fail), and the deepening
  fired in the game compared (the control that the clock's road was on it);
- **the wall clock stops at the budget**: `WallCost` reads the elapsed milliseconds, and
  a game on it spends the budget on some move;
- **the person sees their side only**: the opponent's four not yet shown are a count, not
  names;
- the script and terminal persons read what a person would type.

The pool is two variants of our M-B roster in the pool file's shape (test_poolplay's). The
games are played with hp-share scored in the port and damage-ranked menus, to stay cheap
(a leaf in Python costs seconds a node); the selection solve is checked apart, with an
antisymmetric stub leaf.
"""

from __future__ import annotations

import copy
import io
import json
import time

import numpy as np
import pytest

from pokeuraou import humanplay
from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import seen_identities
from pokeuraou.humanplay import (
    Agent,
    NodeTime,
    PolicyPerson,
    ScriptPerson,
    TerminalPerson,
    WallCost,
    parse_choice,
    parse_selection,
    plan_move,
    render_position,
)
from pokeuraou.pool import load_pool
from pokeuraou.regulation import repo_root
from pokeuraou.selfplay import position_from_sets

#: A fixed price table, so these tests do not move when `NODE_TIME` is re-measured.
PRICES = {("local", 1): NodeTime(fixed_ms=10.0, cell_ms=0.05),
          ("local", 8): NodeTime(fixed_ms=10.0, cell_ms=0.02)}


@pytest.fixture(autouse=True)
def _prices(monkeypatch):  # noqa: ANN001, ANN202
    monkeypatch.setattr(humanplay, "NODE_TIME", PRICES)


def _stub(positions):  # noqa: ANN001, ANN202
    """Antisymmetric: HP summed per side, the active pair counted twice (test_poolplay's)."""
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [
            sum(m.hp * (2 if m.active_index is not None else 1) for m in side.pokemon)
            for side in pos.sides
        ]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 150.0))
    return out


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    base = json.loads(
        (repo_root() / "configs" / "teams" / "rizabanadohido.json").read_text(encoding="utf-8")
    )["team"]
    rotated = copy.deepcopy(base[2:] + base[:2])
    data = {
        "id": "test-pool", "name": "test", "regulation": "gen9championsvgc2026regmb",
        "character": "a test pool",
        "validatedAgainst": {"formatId": "gen9championsvgc2026regmb"},
        "teams": [{"id": f"t{i}", "name": f"team {i}", "team": t}
                  for i, t in enumerate([base, rotated])],
    }
    path = tmp_path_factory.mktemp("pool") / "p.json"
    path.write_bytes(json.dumps(data).encode("utf-8"))
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _agent(pool, **kwargs) -> Agent:  # noqa: ANN001
    """hp-share, scored in the port: a stub leaf in Python costs seconds a node here."""
    settings = {"seconds": 0.4, "cores": 1, "clock": "count", "rank_by_leaf": False,
                "rank_fill": "refs2"}
    settings.update(kwargs)
    return Agent(reg=pool.reg, evaluate=None, name="hp-share", **settings)


def _play(pool, person, *, seed: int, agent: Agent | None = None, side: int = 1,  # noqa: ANN001
          turns: int = 3):  # noqa: ANN202
    agent = agent or _agent(pool)
    return humanplay.play(
        agent, person, (pool.teams[0], pool.teams[1]), agent_side=side, seed=seed,
        max_turns=turns,
    )


# ------------------------------------------------------------------------ the rule


def test_the_rule_widens_with_the_budget_and_gives_the_rest_to_depth() -> None:
    widths = [plan_move(s, 1, 40, 40, 3).width for s in (0.01, 0.2, 1.0, 5.0, 45.0)]
    assert widths == sorted(widths)
    assert widths[0] == humanplay.WIDTHS[0]
    big = plan_move(45.0, 1, 40, 40, 3)
    # The whole legal list on both sides fits: 64 would be the same menu as 48.
    assert big.width == 48
    assert big.deepen_ms == pytest.approx(45_000 - big.predicted_ms)
    assert big.predicted_ms <= humanplay.WIDTH_SHARE * 45_000


def test_no_deepening_below_the_floor_or_when_asked_for_width_only() -> None:
    tiny = plan_move(0.02, 1, 40, 40, 3)
    assert tiny.deepen_ms == 0.0
    assert plan_move(45.0, 1, 40, 40, 3, width_only=True).deepen_ms == 0.0


def test_an_unmeasured_core_count_reads_the_one_below() -> None:
    assert humanplay.node_time(4) is PRICES["local", 1]
    assert humanplay.node_time(16) is PRICES["local", 8]
    # More cores, cheaper cells: the same budget reaches at least as wide.
    assert plan_move(0.3, 8, 40, 40, 3).width >= plan_move(0.3, 1, 40, 40, 3).width


def test_the_count_clock_needs_measured_prices(pool) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="no measured prices|measured prices"):
        _agent(pool, clock="count", cores=8)
    _agent(pool, clock="wall", cores=8)  # the wall clock needs none


def test_the_wall_cost_reads_elapsed_milliseconds() -> None:
    cost = WallCost(time.perf_counter() - 0.25)
    assert cost.cell == 1.0
    assert 240.0 <= cost.ms(0, 0, 0) < 5_000.0
    assert cost.ms(10**6, 10**6, 10**9) < 5_000.0  # counts are not read


# ------------------------------------------------------------------------ the person


def test_selection_and_choice_lines() -> None:
    assert parse_selection("3 1 5 6", 6, 4) == (2, 0, 4, 5)
    for bad in ("1 2 3", "1 1 2 3", "0 1 2 3", "1 2 3 x"):
        with pytest.raises(ValueError):
            parse_selection(bad, 6, 4)


def test_the_person_is_shown_only_what_their_side_has_seen(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    own = list(pool.teams[0].sets[:4])
    foe = list(pool.teams[1].sets[:4])
    pos = position_from_sets(reg, own, foe)
    text = render_position(reg, pos, 0, seen_identities(pos, 1), None)
    for s in foe[:2]:
        assert s.species in text
    for s in foe[2:]:
        assert s.species not in text
    assert "まだ見ていない 2 体" in text
    for s in own:
        assert s.species in text
    # Exact HP for one's own side, the displayed percentage for the other's.
    assert f"HP {pos.sides[0].pokemon[0].hp}/{pos.sides[0].pokemon[0].maxhp}" in text
    assert "HP 100%" in text


def test_a_choice_is_a_number_or_its_string_and_the_terminal_asks_slot_by_slot(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    pos = position_from_sets(reg, list(pool.teams[0].sets[:4]), list(pool.teams[1].sets[:4]))
    legal = side_actions(reg, pos, 0)
    assert parse_choice("1", legal) is legal[0]
    assert parse_choice(legal[5].to_choice().replace(" ", "  "), legal) is legal[5]
    with pytest.raises(ValueError, match="not legal"):
        parse_choice("move 9 9, move 9 9", legal)
    answers = iter(["1", "1"])
    person = TerminalPerson(reg, None, read=lambda _p: next(answers), out=io.StringIO())
    got = person.choose("move", legal, "")
    assert got.slots[0].to_choice() == legal[0].slots[0].to_choice()
    assert got in legal
    whole = iter([f"#{legal[3].to_choice()}"])
    person = TerminalPerson(reg, None, read=lambda _p: next(whole), out=io.StringIO())
    assert person.choose("move", legal, "") is legal[3]


# ------------------------------------------------------------------------ the games


def test_the_agent_draws_its_four_from_its_own_side_of_the_solve(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    teams = (pool.teams[0], pool.teams[1])
    entry = humanplay.solve_entry(reg, teams, _stub, "stub")
    six = list(teams[1].sets)
    support = {
        tuple(entry.selections[i])
        for i, p in enumerate(entry.their_mixture(0, epsilon=0.0, temperature=1.0)) if p > 0
    }
    for seed in range(20):
        pick = humanplay.agent_pick(entry, 1, six, 4, np.random.default_rng(seed))
        assert pick in support
    # Without a leaf: four of six, uniformly.
    pick = humanplay.agent_pick(None, 1, six, 4, np.random.default_rng(0))
    assert len(set(pick)) == 4


def test_the_count_clock_replays_byte_for_byte(pool) -> None:  # noqa: ANN001
    first, clock, _ = _play(pool, PolicyPerson("random", 5), seed=5)
    again, _, _ = _play(pool, PolicyPerson("random", 5), seed=5)
    a = json.dumps(first, ensure_ascii=False)
    assert a == json.dumps(again, ensure_ascii=False)
    moves = [d for d in first["decisions"] if d["kind"] == "move"]
    assert moves, "no move decision; the comparison would be vacuous"
    # The positive control: the deepening ran in the game compared, on the count clock.
    deep = [d for d in moves if d.get("deepened") and d["plan"]["deepenBudget"] > 0]
    assert deep, "no move deepened; the clock's road was not on this game"
    assert all(d["plan"]["clock"] == "count" for d in moves)
    assert "searchSeconds" not in first
    assert clock["moves"] == len(moves) and all(
        row["budget"] == 0.4 for row in clock["decisions"] if row["kind"] == "move"
    )
    # The person's own inputs, read back as a script, are the same game again.
    script = ScriptPerson(first["human"]["inputs"])
    replay, _, _ = _play(pool, script, seed=5)
    assert script.at == len(script.lines)
    replay["human"]["person"] = first["human"]["person"]
    assert json.dumps(replay, ensure_ascii=False) == a
    # And the comparison can fail: another seed is another game.
    other, _, _ = _play(pool, PolicyPerson("random", 5), seed=6)
    assert json.dumps(other, ensure_ascii=False) != a


def test_the_record_holds_both_menus_the_agent_mixture_and_the_person_choice(pool) -> None:  # noqa: ANN001
    record, _, _ = _play(pool, PolicyPerson("first"), seed=2, side=0)
    assert record["human"]["side"] == 1
    assert record["provenance"]["kind"] == "human-play"
    assert record["provenance"]["leaves"] == ["hp-share", "person"]
    assert record["human"]["inputs"][0] == "1 2 3 4"
    assert record["foePick"] == [0, 1, 2, 3]
    for d in record["decisions"]:
        if d["kind"] != "move":
            continue
        assert abs(sum(d["ownPolicy"]) - 1.0) < 1e-6
        assert d["ownChosen"] in d["ownActions"]
        assert d["foeChosen"] is not None
        assert d["humanOffMenu"] == (d["foeChosen"] not in d["foeActions"])


def test_the_wall_clock_spends_the_budget(pool, tmp_path) -> None:  # noqa: ANN001
    agent = _agent(pool, clock="wall", seconds=0.3)
    record, clock, _ = _play(pool, PolicyPerson("first"), seed=3, agent=agent, turns=2)
    moves = [row for row in clock["decisions"] if row["kind"] == "move"]
    assert moves
    assert max(row["ratio"] for row in moves) >= 1.0
    assert all(row["ratio"] < 5.0 for row in moves)
    # The files: one line each, LF only.
    out = tmp_path / "g.jsonl"
    humanplay.write_line(out, record)
    humanplay.write_line(humanplay.clock_path(out), clock)
    assert humanplay.clock_path(out).name == "g.clock.jsonl"
    for path in (out, humanplay.clock_path(out)):
        raw = path.read_bytes()
        assert raw.count(b"\n") == 1 and b"\r" not in raw
