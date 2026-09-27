"""IKA-333: two clocks against each other (`pokeuraou.timematch`, `tools/time_match.py`).

What these hold:

- **a condition is the human-play agent** unless it says otherwise: every key left out is
  `humanplay`'s default, and `tools/agent_drift.py` fails when the time match's `Agent(...)`
  or a condition's default parts from `tools/play_human.py` (a fault injected in each);
- **a pair swaps the conditions** and **each seat reads its own move** under its own
  condition: two move rows a decision, one a side, each with its condition's budget, and
  no read falls back to a first legal action;
- **the count clock replays**: the same pair twice is the same two games to the byte (the
  wall-clock fields aside), and another seed is not (the control that the comparison can
  fail), with the deepening fired in the games compared;
- **the person's seat solves its replacements** as the agent does (a replacement row
  from that seat, with more than one option);
- **the scoring**: a pair is 0, 1/2 or 1 for the tested condition, a pair with a game
  nobody won is left out and said, and the in-order prefix stops at the first pair not
  yet in;

Played as `test_humanplay` plays: hp-share scored in the port, damage-ranked menus, the
count clock, on two variants of our M-B roster.
"""

from __future__ import annotations

import copy
import json

import pytest

from pokeuraou import humanplay, timematch
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import DEFAULT_BENCH_DROP
from pokeuraou.humanplay import NodeTime
from pokeuraou.pool import load_pool
from pokeuraou.regulation import repo_root
from pokeuraou.timematch import Condition, Match, parse_condition

PRICES = {("local", 1): NodeTime(fixed_ms=10.0, cell_ms=0.05),
          ("local", 8): NodeTime(fixed_ms=10.0, cell_ms=0.02)}


@pytest.fixture(autouse=True)
def _prices(monkeypatch):  # noqa: ANN001, ANN202
    monkeypatch.setattr(humanplay, "NODE_TIME", PRICES)


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


def _cond(name: str, seconds: float, **kwargs) -> Condition:  # noqa: ANN003
    return Condition(name=name, seconds=seconds, threads=1, clock="count", **kwargs)


def _match(pool, tested: Condition, other: Condition, *, seed: int = 3, turns: int = 3) -> Match:  # noqa: ANN001
    return Match(
        reg=pool.reg, evaluate=None, leaf_name="hp-share", rank_fill="refs2",
        bench_drop=DEFAULT_BENCH_DROP, tested=tested, other=other, seed=seed, max_turns=turns,
        rank_by_leaf=False,
    )


_TIMES = ("seconds", "ratio", "menuSeconds", "selectionSeconds")


def _timeless(value):  # noqa: ANN001, ANN202
    """A game line without its wall-clock fields."""
    if isinstance(value, dict):
        return {k: _timeless(v) for k, v in value.items() if k not in _TIMES}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


# ------------------------------------------------------------------------ conditions


def test_a_condition_left_alone_is_the_human_play_agent() -> None:
    got = parse_condition("long:seconds=10")
    assert got.seconds == 10.0 and got.clock == "wall"
    assert got.threads == humanplay.default_threads()
    assert got.oracle == humanplay.PLAY_ORACLE
    assert got.max_levels == humanplay.PLAY_MAX_LEVELS
    assert not got.width_only and got.child_q is None
    assert got.price_cores == got.threads
    old = parse_condition("old:seconds=1,threads=1,oracle=none,levels=16,width_only=on")
    assert (old.threads, old.oracle, old.max_levels, old.width_only) == (1, None, 16, True)
    assert parse_condition("g:seconds=1,levels=0").max_levels is None
    assert parse_condition("c:seconds=1,clock=count,threads=4").price_cores == 1
    counted = parse_condition("c:seconds=1,clock=count")
    assert (counted.threads, counted.price_cores) == (1, 1)
    assert parse_condition("o:seconds=1,oracle=s24").oracle == 24
    for bad in ("x", "x:threads=4", "x:seconds=1,speed=2", "x:seconds=1,seconds=2",
                "x:seconds=0", "x:seconds=1,oracle=all", "x y:seconds=1"):
        with pytest.raises(ValueError):
            parse_condition(bad)


def _drift():  # noqa: ANN202
    from ._harness import load_tool

    return load_tool("agent_drift")


def test_the_time_match_builds_play_humans_agent(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    drift = _drift()
    assert drift.play_agent_drift() == []
    # A fault in each of the two shapes the check names, or it could be vacuous.
    # (1) a condition's default parts from play_human's.
    monkeypatch.setattr(humanplay, "default_threads", lambda cpus=None: 7)  # noqa: ARG005
    assert any("--threads" in p for p in drift.play_agent_drift())
    monkeypatch.undo()
    # (2) the time match's Agent(...) stops passing an argument play_human passes.
    root = tmp_path / "tree"
    (root / "tools").mkdir(parents=True)
    (root / "src" / "pokeuraou").mkdir(parents=True)
    (root / "tools" / "play_human.py").write_bytes((repo_root() / "tools/play_human.py").read_bytes())
    source = (repo_root() / "src/pokeuraou/timematch.py").read_bytes()
    assert source.count(b" halt=self.halt,") == 1
    (root / "src" / "pokeuraou" / "timematch.py").write_bytes(source.replace(b" halt=self.halt,", b""))
    monkeypatch.setattr(drift, "ROOT", root)
    assert any("omits halt" in p for p in drift.play_agent_drift())


# ------------------------------------------------------------------------ a pair


def test_a_pair_swaps_the_conditions_and_each_seat_reads_its_own_move(pool) -> None:  # noqa: ANN001
    slow, fast = _cond("slow", 0.4), _cond("fast", 0.1)
    lines = timematch.play_pair(_match(pool, slow, fast), 0, (pool.teams[0], pool.teams[1]))
    assert [ln["conditions"] for ln in lines] == [["slow", "fast"], ["fast", "slow"]]
    assert [ln["testedSide"] for ln in lines] == [0, 1]
    assert lines[0]["picks"] == lines[1]["picks"]
    budgets = {"slow": 0.4, "fast": 0.1}
    for ln in lines:
        assert ln["fallbacks"] == [0, 0]
        by_decision: dict[int, list] = {}
        for row in ln["moves"]:
            by_decision.setdefault(row["decision"], []).append(row)
            assert row["condition"] == ln["conditions"][row["side"]]
            assert row["budget"] == budgets[row["condition"]]
        assert by_decision, "no move decision; the comparison would be vacuous"
        for rows in by_decision.values():
            assert sorted(r["side"] for r in rows) == [0, 1]
    # The positive control: the conditions reached their seats -- the slow one's deepening
    # was given more of the count clock.
    rows = [r for ln in lines for r in ln["moves"]]
    slow_budget = max(r["deepenBudget"] for r in rows if r["condition"] == "slow")
    fast_budget = max(r["deepenBudget"] for r in rows if r["condition"] == "fast")
    assert slow_budget > fast_budget > 0


def test_the_count_clock_pair_replays_byte_for_byte(pool) -> None:  # noqa: ANN001
    a, b = _cond("a", 0.3), _cond("b", 0.2, width_only=True)
    teams = (pool.teams[0], pool.teams[1])
    first = timematch.play_pair(_match(pool, a, b), 1, teams)
    again = timematch.play_pair(_match(pool, a, b), 1, teams)
    text = json.dumps(_timeless(first))
    assert text == json.dumps(_timeless(again))
    deep = [r for ln in first for r in ln["moves"] if (r.get("deepened") or {}).get("expanded")]
    assert deep, "no move deepened; the clock's road was not on these games"
    assert all(r["deepenBudget"] == 0 for ln in first for r in ln["moves"] if r["condition"] == "b")
    other = timematch.play_pair(_match(pool, a, b, seed=4), 1, teams)
    assert json.dumps(_timeless(other)) != text


def test_the_person_seat_solves_its_replacements(pool, monkeypatch) -> None:  # noqa: ANN001
    solved = []
    original = humanplay.HumanGame._replacement_mixture

    def spy(self, pos, options, side, shown, leads):  # noqa: ANN001, ANN202
        got = original(self, pos, options, side, shown, leads)
        solved.append((side, self.me, len(options[side])))
        return got

    monkeypatch.setattr(humanplay.HumanGame, "_replacement_mixture", spy)
    lines = timematch.play_pair(
        _match(pool, _cond("a", 0.1), _cond("b", 0.1), turns=12), 2,
        (pool.teams[0], pool.teams[1]),
    )
    rows = [r for ln in lines for r in ln["others"] if r["kind"] == "replacement"]
    person = [r for r in rows if "side" in r]
    assert person, "no replacement from the person's seat; the check would be vacuous"
    assert any(side != me and n > 1 for side, me, n in solved), (
        "the person's seat never solved a replacement with a choice in it"
    )


# ------------------------------------------------------------------------ scoring


def _line(pair: int, game: int, score) -> dict:  # noqa: ANN001
    return {"pair": pair, "game": game, "testedScore": score, "endReason": "turn-limit"}


def test_a_pair_scores_zero_half_or_one_and_a_game_nobody_won_is_left_out() -> None:
    lines = [_line(0, 0, 1.0), _line(0, 1, 1.0), _line(1, 0, 1.0), _line(1, 1, 0.0),
             _line(2, 0, 0.0), _line(2, 1, 0.0), _line(3, 0, None), _line(3, 1, 1.0),
             _line(5, 0, 1.0), _line(5, 1, 1.0), _line(6, 0, 1.0)]
    scores, left = timematch.pair_scores(lines)
    assert scores == {0: 1.0, 1: 0.5, 2: 0.0, 5: 1.0}
    assert list(left) == [3] and "game 0" in left[3]
    # In order, up to the first pair not in (4): 5 waits.
    assert timematch.prefix(scores, left) == [1.0, 0.5, 0.0]
    got = timematch.elo_interval([1.0, 0.5, 0.5, 0.5])
    assert got["score"] == 0.625 and got["counts"] == [0, 3, 1]
    assert got["low"] < got["elo"] < got["high"]
    assert timematch.elo_interval([]) == {"pairs": 0}


def test_a_fixed_width_replaces_the_rule(pool) -> None:  # noqa: ANN001
    wide = _cond("w36", 0.1, width_only=True, width=36, oracle=None)
    narrow = _cond("w8", 0.1, width_only=True, width=8, oracle=None)
    lines = timematch.play_pair(_match(pool, wide, narrow), 0, (pool.teams[0], pool.teams[1]))
    rows = [r for ln in lines for r in ln["moves"]]
    assert {r["width"] for r in rows if r["condition"] == "w36"} == {36}
    assert {r["width"] for r in rows if r["condition"] == "w8"} == {8}
    assert all(r["deepenBudget"] == 0 for r in rows)
    assert parse_condition("w:seconds=1,width=36").width == 36
    # The control: the rule at 0.1 s would not have chosen 36 there.
    from pokeuraou.humanplay import plan_move

    assert plan_move(0.1, 1, 40, 40, 3).width < 36
    assert plan_move(0.1, 1, 40, 40, 3, width=36).width == 36
