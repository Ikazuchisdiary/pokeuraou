"""IKA-423: a match begun at positions where two conditions play different moves.

What these hold:

- **a start is where the pair begins** (`humanplay.GameStart`, `timematch.play_pair(start=)`):
  the first decision of both games is the start's position, the selection is not solved, the
  picks are the start's; the same pair played without it begins at turn 1 (the control that the
  start path was on);
- **the same condition twice is the same game** on a start (the A/A control: every pair 1/2, the
  two games the same moves), and two different conditions are not (the control that the
  comparison can fail);
- **a probe reads both seats under both conditions** and writes the answer's most played move;
- **the choice of starts asks nothing of which condition is which** (`select_positions`): the
  same starts when the two names are swapped; a tie is not a difference; a decided position is
  left out; a real difference is kept (the controls that the rule can keep and can leave out);
- **`tools/targeted_positions.py candidates`** leaves out a position where a bench is unseen.

Played as `test_timematch` plays: the hp-share leaf scored in the port, damage-ranked menus,
the count clock.
"""

# ruff: noqa: F811 - the `pool` fixture is test_timematch's, used by name

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pokeuraou import humanplay, timematch
from pokeuraou.hidden import identity
from pokeuraou.position import Position
from pokeuraou.regulation import repo_root

from .test_timematch import PRICES, _cond, _match, pool  # noqa: F401 - the fixture

sys.path.insert(0, str(repo_root() / "tools"))
import targeted_positions as tp  # noqa: E402


@pytest.fixture(autouse=True)
def _prices(monkeypatch):  # noqa: ANN001, ANN202
    monkeypatch.setattr(humanplay, "NODE_TIME", PRICES)


def _recorded_start(pool, turn_index: int = 2):  # noqa: ANN001, ANN201
    """A start from a played game's decision: its position and the four picked."""
    match = _match(pool, _cond("a", 0.3), _cond("b", 0.2), turns=5)
    match.transcript = True
    teams = (pool.teams[0], pool.teams[1])
    line = timematch.play_pair(match, 1, teams, games=(0,))[0]
    moves = [d for d in line["transcript"]["decisions"] if d["kind"] == "move"]
    assert len(moves) > turn_index, "the recorded game is too short for the test"
    pos = Position.from_json(moves[turn_index]["position"])
    shown = [sorted(identity(m) for m in pos.sides[s].pokemon) for s in (0, 1)]
    start = humanplay.GameStart.from_json({
        "id": "t", "teams": [teams[0].id, teams[1].id], "picks": line["picks"],
        "seenIds": shown, "position": moves[turn_index]["position"],
    })
    return start, moves[turn_index]["position"], teams, line


def test_a_start_is_where_the_pair_begins(pool) -> None:  # noqa: ANN001
    start, recorded, teams, plain = _recorded_start(pool)
    assert recorded["turn"] > 1, "the start is at turn 1; it would not tell the paths apart"
    match = _match(pool, _cond("a", 0.3), _cond("b", 0.2), turns=3)
    match.transcript = True
    lines = timematch.play_pair(match, 5, teams, start=start)
    assert [ln["start"] for ln in lines] == ["t", "t"]
    for ln in lines:
        first = next(d for d in ln["transcript"]["decisions"] if d["kind"] == "move")
        assert first["position"] == recorded
        assert ln["picks"] == plain["picks"]
        assert ln["moves"], "no move was read; the comparison would be vacuous"
        assert "selection" not in ln
    # The control: the pair played with no start begins at turn 1, on the same teams.
    normal = timematch.play_pair(match, 5, teams)
    assert normal[0]["transcript"]["decisions"][0]["position"]["turn"] == 1


def test_the_same_condition_twice_is_one_game_and_two_conditions_are_not(pool) -> None:  # noqa: ANN001
    start, _recorded, teams, _plain = _recorded_start(pool)
    equal = _match(pool, _cond("a", 0.3), _cond("same", 0.3), turns=6)
    equal.adjudication = (1, 0.02)  # a result from the reads, so that a pair is scored
    same = timematch.play_pair(equal, 5, teams, start=start)
    keys = ("outcome", "turns", "endReason")
    assert [same[0][k] for k in keys] == [same[1][k] for k in keys]
    for g0, g1 in zip(same[0]["moves"], same[1]["moves"], strict=True):
        assert g0["width"] == g1["width"] and g0["rows"] == g1["rows"]
        assert g0["value0"] == g1["value0"]
    assert same[0]["outcome"] in (0.0, 1.0), "no result; the pair would not be scored"
    assert timematch.pair_scores(same)[0][5] == 0.5
    # The control that the comparison can fail: another condition on the same start reads
    # something else (its menu is one row wide).
    diff = timematch.play_pair(
        _match(pool, _cond("a", 0.3), _cond("narrow", 0.3, width=1), turns=4), 5, teams,
        start=start)
    widths = {r["width"] for ln in diff for r in ln["moves"]}
    assert 1 in widths and max(widths) > 1
    assert 1 not in {r["width"] for ln in same for r in ln["moves"]}


def test_a_probe_reads_both_seats_under_both_conditions(pool) -> None:  # noqa: ANN001
    start, _recorded, teams, _plain = _recorded_start(pool)
    match = _match(pool, _cond("a", 0.3), _cond("b", 0.2, width=1), turns=4)
    lines = timematch.play_pair(match, 5, teams, start=start, probe=True)
    reads = tp.reads_of(lines, "a", "b")
    assert sorted(reads[5]) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    for read in reads[5].values():
        assert read["top"] in read["menu"] and abs(sum(read["p"]) - 1.0) < 1e-4
        assert read["p"][read["menu"].index(read["top"])] == pytest.approx(read["topP"], abs=1e-5)
    # One decision a game and nobody played it: the game stopped after the read.
    assert all(len([r for r in ln["moves"]]) == 2 and ln["outcome"] is None for ln in lines)
    # Each condition's read came under its own seat's condition (the clock rows' width).
    assert {(r["condition"], r["width"]) for ln in lines for r in ln["moves"]
            } >= {("b", 1)} and any(r["width"] > 1 for ln in lines for r in ln["moves"]
                                    if r["condition"] == "a")
    with pytest.raises(ValueError, match="needs a start"):
        timematch.play_pair(match, 5, teams, probe=True)


# ---------------------------------------------------------------------- the choice of starts


def _read(top: str, p: float = 1.0, value: float = 0.5, menu=("x", "y", "z")) -> dict:  # noqa: ANN001
    other = (1.0 - p) / max(sum(m != top for m in menu), 1)
    probs = [other if m != top else p for m in menu]
    return {"side": 0, "top": top, "topP": p, "menu": list(menu), "p": probs, "value0": value,
            "stage": "d2"}


def _reads(spec: dict[int, tuple]) -> dict:  # noqa: ANN001
    """{pair: ((cond a seat 0, a seat 1, b seat 0, b seat 1)) reads}."""
    return {pair: {("a", 0): r[0], ("a", 1): r[1], ("b", 0): r[2], ("b", 1): r[3]}
            for pair, r in spec.items()}


def test_the_choice_of_starts_asks_nothing_of_which_condition_is_which() -> None:
    starts = [{"id": str(i)} for i in range(6)]
    same = _read("x")
    reads = _reads({
        0: (same, same, same, same),                                  # no difference
        1: (_read("x"), same, _read("y"), same),                      # seat 0 differs
        2: (same, _read("x"), same, _read("z")),                      # seat 1 differs
        3: (_read("x"), _read("x"), _read("y"), _read("z")),          # both seats differ
        4: (_read("x", value=0.02), same, _read("y", value=0.02), same),  # decided
        5: (_read("x", p=0.5), same, _read("y", p=0.5), same),        # seat 0 differs, mixed
    })
    kept, stats = tp.select_positions(starts, reads, ("a", "b"), band=0.1)
    assert [k["id"] for k in kept] == ["1", "2", "3", "5"]
    swapped_reads = {pair: {(("b" if name == "a" else "a"), seat): read
                            for (name, seat), read in got.items()}
                     for pair, got in reads.items()}
    other, other_stats = tp.select_positions(starts, swapped_reads, ("b", "a"), band=0.1)
    assert [k["id"] for k in other] == [k["id"] for k in kept]
    assert {k: v for k, v in stats.items() if k != "tvMean"} == {
        k: v for k, v in other_stats.items() if k != "tvMean"}
    assert (stats["seat0"], stats["seat1"], stats["both"]) == (2, 1, 1)
    # The controls that the rule can leave out and keep: the decided position, and the
    # position that differs, are each in the counts.
    assert stats["decided"] == 1
    kept_all, _ = tp.select_positions(starts, reads, ("a", "b"), band=0.0)
    assert "4" in [k["id"] for k in kept_all] and "4" not in [k["id"] for k in kept]


def test_a_tie_between_equal_moves_is_not_a_difference() -> None:
    a = {"top": "x", "topP": 0.5, "menu": ["x", "y"], "p": [0.5, 0.5]}
    b = {"top": "y", "topP": 0.5, "menu": ["x", "y"], "p": [0.5, 0.5]}
    assert tp.compare(a, b) == {"differ": False, "tv": 0.0, "tops": ["x", "y"], "tie": True}
    c = {"top": "y", "topP": 0.9, "menu": ["x", "y"], "p": [0.1, 0.9]}
    got = tp.compare(a, c)
    assert got["differ"] and abs(got["tv"] - 0.4) < 1e-9
    assert tp.compare(c, a)["differ"]


def test_candidates_leave_out_a_position_with_a_bench_nobody_has_seen(pool) -> None:  # noqa: ANN001
    start, recorded, teams, _plain = _recorded_start(pool)
    entry = {"game": 7, "turn": recorded["turn"], "teams": [teams[0].id, teams[1].id],
             "position": recorded, "outcome": 1.0,
             "seen": [list(range(4)), list(range(4))]}
    line = tp.start_line(entry, teams, pool.reg)
    assert line is not None
    again = humanplay.GameStart.from_json(line)
    assert again.position.to_json() == recorded
    # The picks are the position's slot order (the opening order is gone): the same four.
    assert [sorted(p) for p in again.picks] == [sorted(p) for p in start.picks]
    entry["seen"] = [list(range(4)), [0, 1]]
    assert tp.start_line(entry, teams, pool.reg) is None


def test_time_match_refuses_a_probe_without_starts_and_a_start_file_too_short(tmp_path: Path) -> None:
    import time_match

    arms = ["--arm", "a:seconds=1,clock=count", "--arm", "b:seconds=2,clock=count"]
    out = ["--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="needs --starts"):
        time_match.conditions(time_match.parse_args([*arms, *out, "--pairs", "1", "--probe"]))
    starts = tmp_path / "s.jsonl"
    starts.write_bytes(b'{"id": "x"}\n')
    with pytest.raises(SystemExit, match="go past them"):
        time_match.conditions(time_match.parse_args(
            [*arms, *out, "--pairs", "2", "--starts", str(starts)]))
    time_match.conditions(time_match.parse_args(
        [*arms, *out, "--pairs", "1", "--starts", str(starts)]))
    assert json.loads(starts.read_bytes())["id"] == "x"
