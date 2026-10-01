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

import numpy as np
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


def test_the_restricted_reading_deepens_the_rectangle(pool) -> None:  # noqa: ANN001
    """IKA-362: ``restricted`` reads the deepened root as IKA-68's ``r`` -- on the Bayesian
    root too -- and plays; the count clock replays it."""
    r = _cond("r", 0.4, restricted=True)
    m = _cond("m", 0.4)
    teams = (pool.teams[0], pool.teams[1])
    first = timematch.play_pair(_match(pool, r, m, turns=4), 3, teams)
    again = timematch.play_pair(_match(pool, r, m, turns=4), 3, teams)
    assert json.dumps(_timeless(first)) == json.dumps(_timeless(again))
    rows = [row for ln in first for row in ln["moves"] if row["condition"] == "r"]
    deep = [row for row in rows if (row.get("deepened") or {}).get("expanded")]
    assert deep, "no restricted move deepened"
    hidden = [row for row in deep if row["classes"] > 1]
    assert hidden, "no Bayesian root was read restricted"
    # Nothing probed by the swap oracle: the restricted reading asks the matrices itself.
    assert all("probed" not in (row.get("deepened") or {}) for row in rows)


def test_the_port_threads_leave_a_node_time_pair_as_it_was(pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-362's `--port-threads`: more cell threads in the port, the same games."""
    a, b = _cond("a", 0.3), _cond("b", 0.2)
    teams = (pool.teams[0], pool.teams[1])
    one = timematch.play_pair(_match(pool, a, b), 4, teams)
    monkeypatch.setattr(timematch, "PORT_THREADS", 3)
    three = timematch.play_pair(_match(pool, a, b), 4, teams)
    from pokeuraou import rustnode

    assert rustnode.port_threads() == 3
    assert json.dumps(_timeless(three)) == json.dumps(_timeless(one))
    monkeypatch.setattr(timematch, "PORT_THREADS", None)
    timematch.spread_threads(pool.reg, 1)


def test_a_fixed_depth_two_read_plays_and_replays(pool) -> None:  # noqa: ANN001
    """IKA-362's D: ``depth=2`` with ``width_only`` reads the support rectangle a ply
    deeper on both kinds of root; the count clock replays it, and it plays another game
    than the depth-1 agent at the same width (the read reached the answer)."""
    d = _cond("d", 0.3, width_only=True, depth=2)
    flat = _cond("flat", 0.3, width_only=True)
    teams = (pool.teams[0], pool.teams[1])
    first = timematch.play_pair(_match(pool, d, flat, turns=4), 5, teams)
    again = timematch.play_pair(_match(pool, d, flat, turns=4), 5, teams)
    assert json.dumps(_timeless(first)) == json.dumps(_timeless(again))
    same = timematch.play_pair(_match(pool, flat, _cond("flat2", 0.3, width_only=True), turns=4), 5,
                               teams)
    assert json.dumps(_timeless([ln["moves"] for ln in first])) != json.dumps(
        _timeless([ln["moves"] for ln in same])) or [ln["outcome"] for ln in first] != [
        ln["outcome"] for ln in same]
    assert parse_condition("d:seconds=3,clock=count,width_only=on,depth=2").depth == 2


def test_the_openings_root_at_every_legal_action(pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-366's ``root_all``: with ``depth2_auto``, a move where a side has more legal
    actions than the widest menu reads every legal action of both sides at the root when
    that node fits the budget; the count clock replays it; each move row carries its read's
    value. The control: the same condition without it never reads past the widest menu.
    The depth-2 read after it ranks its children by the Q, which this test has not: it is
    left out (none fits), the root is what is checked."""
    monkeypatch.setattr(humanplay, "depth2_children", lambda *_args: None)
    wide = _cond("all", 1.0, depth2_auto=True, root_all=True)
    rule = _cond("rule", 1.0, depth2_auto=True)
    teams = (pool.teams[0], pool.teams[1])
    first = timematch.play_pair(_match(pool, wide, rule, turns=2), 6, teams)
    again = timematch.play_pair(_match(pool, wide, rule, turns=2), 6, teams)
    assert json.dumps(_timeless(first)) == json.dumps(_timeless(again))
    rows = [r for ln in first for r in ln["moves"]]
    widened = [r for r in rows if r.get("rootAll")]
    assert widened, "no root was widened; the check would be vacuous"
    assert all(r["condition"] == "all" for r in widened)
    assert any(max(r["rows"], r["cols"]) > humanplay.WIDTHS[-1] for r in widened)
    assert all(max(r["rows"], r["cols"]) <= humanplay.WIDTHS[-1]
               for r in rows if r["condition"] == "rule")
    assert all(0.0 <= r["value0"] <= 1.0 for r in rows)
    assert parse_condition("a:seconds=1,depth2_auto=on,root_all=on").root_all
    with pytest.raises(ValueError, match="depth2_auto"):
        _match(pool, wide, rule).agent(_cond("x", 1.0, root_all=True))


# ------------------------------------------------------------------------ IKA-384


def test_a_decided_game_is_stopped_where_the_turns_reads_agree(pool) -> None:  # noqa: ANN001
    """`Match.adjudication`: a game whose turn's two reads are away from 1/2 by the
    threshold ends there, scored 0/1 by the side of 1/2 the reads' mean is on; without it
    the same pair is played on (control: nothing is stopped, and the reads up to the stop
    are the same)."""
    a, b = _cond("a", 0.3), _cond("b", 0.2)
    teams = (pool.teams[0], pool.teams[1])
    plain = timematch.play_pair(_match(pool, a, b, turns=4), 1, teams)
    assert all("adjudicated" not in ln for ln in plain)
    assert all(ln["endReason"] != "adjudicated" for ln in plain)
    match = _match(pool, a, b, turns=4)
    match.adjudication = (1, 0.02)
    cut = timematch.play_pair(match, 1, teams)
    stopped = [ln for ln in cut if "adjudicated" in ln]
    assert stopped, "nothing was adjudicated; the test would be vacuous"
    for ln in stopped:
        assert ln["endReason"] == "adjudicated"
        assert ln["outcome"] in (0.0, 1.0)
        assert ln["turns"] == ln["adjudicated"]["turn"] + 1
        assert (ln["outcome"] == 1.0) == (ln["adjudicated"]["value"] > 0.5)
        assert abs(ln["adjudicated"]["value"] - 0.5) >= 0.02 - 1e-9
        mine = ln["moves"]
        other = plain[ln["game"]]["moves"]
        assert len(mine) < len(other)
        assert _timeless(mine) == _timeless(other[:len(mine)])
    # A threshold nothing reaches is the game played out, the same to the byte.
    match.adjudication = (1, 0.51)
    never = timematch.play_pair(match, 1, teams)
    assert json.dumps(_timeless(never)) == json.dumps(_timeless(plain))


# ------------------------------------------------------------------------ the transcript


def _played_with_transcript(pool):  # noqa: ANN001, ANN202
    a, b = _cond("a", 0.3), _cond("b", 0.2)
    teams = (pool.teams[0], pool.teams[1])
    plain = timematch.play_pair(_match(pool, a, b, turns=4), 1, teams)
    match = _match(pool, a, b, turns=4)
    match.transcript = True
    return plain, timematch.play_pair(match, 1, teams), match, teams


def test_the_transcript_leaves_the_games_as_they_were(pool, monkeypatch) -> None:  # noqa: ANN001
    """With it on, the games are the same to the byte (the wall-clock fields aside) as with it
    off, and off writes no transcript; the account has an event trace for every move turn, each
    the drawn outcome's (its position the game's own), and a different seed's is another."""
    plain, on, match, teams = _played_with_transcript(pool)
    assert all("transcript" not in ln for ln in plain)
    stripped = [{k: v for k, v in ln.items() if k != "transcript"} for ln in on]
    assert json.dumps(_timeless(stripped)) == json.dumps(_timeless(plain))
    moves = 0
    for ln in on:
        for d in ln["transcript"]["decisions"]:
            if d["kind"] == "move":
                moves += 1
                assert d["events"] and d["events"]["lines"], "a move turn without its trace"
                assert d["events"]["matched"], "the port's re-asked outcome is not the game's"
    assert moves >= 4, "no turns were played; the test would be vacuous"
    match.seed = 4
    other = timematch.play_pair(match, 1, teams)
    assert json.dumps(_timeless([ln["transcript"] for ln in other])) != json.dumps(
        _timeless([ln["transcript"] for ln in on])
    )
    # The control that the identity check can fail: a note that draws from the game's rng.
    original = timematch.TimedGame._note_turn

    def touching(self, *args):  # noqa: ANN001, ANN002, ANN202
        self.rng.random()
        return original(self, *args)

    monkeypatch.setattr(timematch.TimedGame, "_note_turn", touching)
    match.seed = 3
    broken = timematch.play_pair(match, 1, teams)
    assert json.dumps(_timeless([{k: v for k, v in ln.items() if k != "transcript"} for ln in broken])) != (
        json.dumps(_timeless(plain))
    )


def test_a_transcript_is_shown_as_one_japanese_page(pool) -> None:  # noqa: ANN001
    """`show_game.py --transcript`: one column (no toggles), a card a move turn, the seats named,
    Japanese names from the dump; a game with no result reads as stopped."""
    from ._harness import load_tool

    show_game = load_tool("show_game")
    import game_page

    _plain, on, _match_, _teams = _played_with_transcript(pool)
    loc = show_game.Localiser(pool.reg, show_game.load_names("ja"))
    line = on[0]
    page = game_page.render_html(pool.reg, loc, line)
    turns = [d for d in line["transcript"]["decisions"] if d["kind"] == "move"]
    assert page.count('<section class="turn"') == len(turns)
    # The log is one long column: the only thing that opens and closes is a turn's mixtures.
    assert page.count("<details") == page.count('<details class="mix"') and "<summary" in page
    assert "席 0" in page and "席 1" in page
    first = line["transcript"]["ownSix"][0]
    assert loc.species(first) in page and loc.species(first) != first
    assert "打ち切り" in page
    # The control that the page can fail: a page of another game is not this one.
    assert game_page.render_html(pool.reg, loc, on[1]) != page
    # The mixtures of each seat's read are in the account, one per seat of every move turn.
    for d in turns:
        assert sorted(d["reads"]) == ["0", "1"]
        assert d["reads"]["0"]["rows"] and d["reads"]["0"]["menu"][0] >= 1
    assert page.count('<details class="mix"') == len(turns)


def test_the_pages_rank_is_among_the_moves_played_not_the_menu() -> None:
    """The denominator of "2 位 / N 通り" was the menu (64 candidates); it is the moves the
    mixture plays (probability >= `game_page.SUPPORT_MIN`), the menu beside, and a drawn move
    below the threshold is said to be outside the equilibrium. Counted from the whole vector."""
    from ._harness import load_tool

    load_tool("show_game")
    import game_page

    vector = [0.6, 0.3, 0.09, 0.006, 0.004, 0.0, 0.0]  # four moves at 0.5% or more, seven candidates
    shown = [["a", 0.6], ["b", 0.3], ["c", 0.09]]
    sup = game_page.support_of(vector, shown)
    assert sup["n"] == 4 and sup["menu"] == 7
    assert sup["rest"][0] == 1 and abs(sup["rest"][1] - 0.006) < 1e-9
    assert sup["small"][0] == 3 and abs(sup["small"][1] - (1 - 0.996)) < 1e-9
    inside = game_page.place_text({"sup": sup, "chosenP": 0.3, "chosenRank": 2})
    assert inside == "（4 通り中 2 位）"  # not "of 7": the menu is not the moves played
    outside = game_page.place_text({"sup": sup, "chosenP": 0.004, "chosenRank": 5})
    assert "均衡の外" in outside and "0.4%" in outside
    # A record from before the vectors were kept: the rank alone, no invented count.
    assert game_page.support_of(None, shown) is None
    assert game_page.place_text({"sup": None, "chosenP": 0.3, "chosenRank": 2}) == "（2 位）"


def test_the_page_says_which_field_moves_fail(pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-395's verdicts are shown as small marks on the move in a seat's row and in the tables
    (the candidates are not changed): every grade has words, and a verdict reaches the page --
    with none reported, the page has no such mark (the control)."""
    from ._harness import load_tool

    show_game = load_tool("show_game")
    import game_page

    from pokeuraou import narrow

    assert set(game_page.DEAD_TEXT) == {
        narrow.DEAD, narrow.DEAD_BUT_CHANGEABLE, narrow.DEAD_BUT_DODGES_SUCKER_PUNCH,
        narrow.DEAD_BUT_FEEDS_STOMPING_TANTRUM, narrow.DEAD_IF_FIRST_ACTS,
    }
    _plain, on, _m, _t = _played_with_transcript(pool)
    loc = show_game.Localiser(pool.reg, show_game.load_names("ja"))
    clean = game_page.render_html(pool.reg, loc, on[0])
    assert 'class="tag dead"' not in clean
    monkeypatch.setattr(narrow, "dead_field_moves", lambda reg, pos, side, action: [narrow.DEAD, None])
    game_page._ACTIONS.clear()
    marked = game_page.render_html(pool.reg, loc, on[0])
    assert marked.count('class="tag dead"') > 0 and "失敗が決まっている" in marked
    assert "失敗の札の意味" in marked and "失敗の札の意味" not in clean
    game_page._ACTIONS.clear()


def _action(index: int, target: int | None = None):  # noqa: ANN202
    from pokeuraou.actions import MoveAction, SideAction

    return SideAction(slots=(MoveAction(slot=0, move_index=index, move_id="x", target=target),))


def test_a_reads_summary_says_what_the_seat_would_have_played_and_what_hurt_it() -> None:
    """`read_summary`: the mixture's heaviest rows and the sum of the rest, where the drawn move
    stands, the other side's modelled mixture, and from the ladder's matrices the opponent's moves
    that take most from the mixture (and from the drawn move). Side 1's matrices are side 0's
    negated, so its own win rate is one more than the entry (the control below breaks that)."""
    from types import SimpleNamespace

    mine = [_action(1), _action(2), _action(3)]
    other = [_action(1, 1), _action(2, 1), _action(3, 1)]
    x = np.array([0.6, 0.3, 0.1])
    y = np.array([0.5, 0.5, 0.0])
    payoff = np.array([[0.9, 0.5, 0.2], [0.4, 0.6, 0.7], [0.5, 0.5, 0.5]])
    for me, offset in ((0, 0.0), (1, 1.0)):
        matrix = payoff if me == 0 else -payoff
        ladder = SimpleNamespace(prices=[matrix], replies=(y,))
        got = timematch.read_summary((me, mine, other, x, y, ladder, True), [1.0], mine[1].to_choice())
        assert got["rows"][0] == [mine[0].to_choice(), 0.6] and got["chosenRank"] == 2
        assert got["chosenP"] == 0.3 and got["rowsRest"][0] == 0
        assert got["p"] == [0.6, 0.3, 0.1] and got["q"] == [0.5, 0.5, 0.0]
        assert [c for c, _p in got["supp"]] == [m.to_choice() for m in mine]
        assert [c for c, _p in got["oppSupp"]] == [o.to_choice() for o in other[:2]]  # 0.0 is outside
        assert got["opp"][0][1] == 0.5
        by_col = offset + x @ matrix
        worst = int(np.argmin(by_col))
        assert got["hard"][0] == [other[worst].to_choice(), round(float(by_col[worst]), 6)]
        assert got["hardChosen"][1] == round(float((offset + matrix[1]).min()), 6)
        assert got["eq"] == round(float(offset + x @ matrix @ y), 6)
        # One completion: the lowest column is the guarantee.
        assert got["guarantee"] == got["hard"][0][1]
        # Among the columns the modelled mixture plays (here the first two; the third has 0):
        # the hardest and the best reply to the drawn move.
        against = offset + matrix[1]
        assert got["vs"] == [round(float(v), 6) for v in against]
        assert got["cols"] == [o.to_choice() for o in other]
        assert got["chosenEv"] == round(float(offset + matrix[1] @ y), 6)
        low, high = int(np.argmin(against[:2])), int(np.argmax(against[:2]))
        assert got["hardIn"] == [other[low].to_choice(), round(float(against[low]), 6)]
        assert got["bestIn"] == [other[high].to_choice(), round(float(against[high]), 6)]
        assert got["hardIn"][1] >= got["hardChosen"][1] and got["bestIn"][1] <= float(against.max())
    # Two completions of the hidden bench: the opponent that knows its own bench takes its own
    # hardest column in each, so the guarantee is the weighted sum of the lowest columns, and
    # the lowest column over both at once (`hard`) is never below it (here above it).
    second = np.array([[0.2, 0.9, 0.5], [0.6, 0.3, 0.5], [0.5, 0.5, 0.5]])
    both = timematch.read_summary(
        (0, mine, other, x, y, SimpleNamespace(prices=[payoff, second], replies=(y, y)), False),
        [0.5, 0.5], mine[0].to_choice(),
    )
    lows = [float((x @ m).min()) for m in (payoff, second)]
    assert both["guarantee"] == round(0.5 * lows[0] + 0.5 * lows[1], 6)
    assert both["hard"][0][1] > both["guarantee"]
    # No matrices (a read without a ladder): the mixtures only, and nothing made up.
    plain = timematch.read_summary((0, mine, other, x, y, None, True), [1.0], mine[0].to_choice())
    assert plain["hard"] is None and plain["hardChosen"] is None and "eq" not in plain
    # The control: with side 1's offset dropped the number differs.
    wrong = timematch.read_summary(
        (1, mine, other, x, y, SimpleNamespace(prices=[-payoff], replies=(y,)), True), [1.0],
        mine[0].to_choice(),
    )
    assert wrong["eq"] == round(float(1.0 + x @ -payoff @ y), 6) != round(float(x @ -payoff @ y), 6)


# ------------------------------------------------- IKA-398: a condition with a leaf of its own


class _Stop(Exception):  # noqa: N818
    pass


def _leaf_match(pool, other_evaluate):  # noqa: ANN001, ANN202
    tested = _cond("A", 1.0)
    other = _cond("B", 1.0)
    return Match(
        reg=pool.reg, evaluate="leaf-A", leaf_name="A-name", rank_fill="refs2",
        bench_drop=DEFAULT_BENCH_DROP, tested=tested, other=other, seed=3, max_turns=3,
        rank_by_leaf=False, other_evaluate=other_evaluate,
        other_leaf_name=None if other_evaluate is None else "B-name",
    )


def test_each_condition_reads_and_solves_its_own_leaf(monkeypatch, pool) -> None:  # noqa: ANN001
    from pokeuraou.pool import draw_pair

    def run(match):  # noqa: ANN001, ANN202
        solved, played = [], []
        monkeypatch.setattr(
            humanplay, "solve_entry",
            lambda reg, teams, evaluate, model, **kw: solved.append((evaluate, model)),  # noqa: ARG005
        )

        def play(seat, person, teams, **kw):  # noqa: ANN001, ANN003, ANN202, ARG001
            played.append(kw["make_game"].keywords if hasattr(kw["make_game"], "keywords") else None)
            raise _Stop

        monkeypatch.setattr(humanplay, "play", play)
        _k, a, b = draw_pair(np.random.default_rng([3, 0]), pool.pairs)
        with pytest.raises(_Stop):
            timematch.play_pair(match, 0, (pool.teams[a], pool.teams[b]))
        return solved

    # Own leaves: each condition's selection is solved by its own leaf, once each.
    own = _leaf_match(pool, "leaf-B")
    assert run(own) == [("leaf-A", "A-name"), ("leaf-B", "B-name")]
    assert own.agent(own.tested).evaluate == "leaf-A" and own.agent(own.tested).name == "A-name"
    assert own.agent(own.other).evaluate == "leaf-B" and own.agent(own.other).name == "B-name"
    assert own.solve_key(own.tested) != own.solve_key(own.other)
    # The control that this can fail: no leaf of its own is every run before it, one solve.
    same = _leaf_match(pool, None)
    assert run(same) == [("leaf-A", "A-name")]
    assert same.agent(same.other).evaluate == "leaf-A"
    assert same.solve_key(same.tested) == same.solve_key(same.other)


# ------------------------------------------------------------------------ a showcase game


def test_a_single_game_pair_plays_only_game_zero_and_the_same_game(pool) -> None:  # noqa: ANN001
    a, b = _cond("a", 0.3), _cond("b", 0.2)
    teams = (pool.teams[0], pool.teams[1])
    both = timematch.play_pair(_match(pool, a, b), 0, teams)
    one = timematch.play_pair(_match(pool, a, b), 0, teams, games=(0,))
    assert len(both) == 2, "the control: the default plays both games"
    assert len(one) == 1 and one[0]["game"] == 0
    assert json.dumps(_timeless(one[0])) == json.dumps(_timeless(both[0]))


def test_selection_seconds_is_a_condition_key_and_reaches_the_agent(pool) -> None:  # noqa: ANN001
    got = parse_condition("s:seconds=1,selection=default,selection_seconds=90")
    assert got.selection_seconds == 90.0
    assert "for 90 s" in got.describe()
    assert parse_condition("s:seconds=1").selection_seconds is None
    match = _match(pool, got, parse_condition("t:seconds=1"))
    assert match.agent(got).selection_seconds == 90.0
    assert match.agent(match.other).selection_seconds is None


def test_a_list_of_servers_gives_worker_k_the_kth() -> None:
    from pokeuraou.ladder import _worker_args

    spec = ("served", "a:1,b:2", "value", False, "q.pt", "q")
    assert [_worker_args(spec, k)[1] for k in range(4)] == ["a:1", "b:2", "a:1", "b:2"]
    one = ("served", "a:1", "value", False, "q.pt", "q")
    assert _worker_args(one, 3) == one
