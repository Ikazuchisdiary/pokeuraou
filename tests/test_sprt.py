"""The sequential test is only worth its early stop if it is the test fishtest runs.

Two things are checked against something independent rather than against themselves. The
ratio is compared with fishtest's own construction -- regularise, solve the secular
equation with a root finder, sum the log ratios -- because the closed form here replaces
that root finder, and a quadratic with the wrong root chosen would still return a number.
And the fast record reader is compared with parsing the record whole, on records written by
the real `write_game`, because it finds its fields by position and a layout change would
otherwise pair the wrong games without any error at all.
"""

from __future__ import annotations

import io
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from pokeuraou import sprt
from pokeuraou.provenance import provenance, write_game
from pokeuraou.selfplay import Decision, GameRecord
from pokeuraou.workqueue import WorkQueue, run_workers

from ._harness import load_tool


def _fishtest_llr(counts: list[float], elo0: float, elo1: float) -> float:
    """`LLRcalc.LLR_logistic`, transcribed: regularize, results_to_pdf, MLE_expected by
    brentq on the secular equation, then N times the mean log ratio."""
    results = [c if c != 0 else 1e-3 for c in counts]
    n = sum(results)
    pdf = [(i / (len(results) - 1), r / n) for i, r in enumerate(results)]

    def mle(s: float) -> list[float]:
        shifted = [(a - s, p) for a, p in pdf]
        v = min(a for a, _ in shifted)
        w = max(a for a, _ in shifted)

        def f(x: float) -> float:
            return sum(p * a / (1 + x * a) for a, p in shifted)

        x = brentq(f, -1 / w + 1e-9, -1 / v - 1e-9)
        return [p / (1 + x * (a - s)) for a, p in pdf]

    s0, s1 = sprt.expected_score(elo0), sprt.expected_score(elo1)
    p0, p1 = mle(s0), mle(s1)
    return n * sum(p * (math.log(q1) - math.log(q0)) for (_, p), q0, q1 in zip(pdf, p0, p1, strict=True))


@pytest.mark.parametrize("elos", [(0, 10), (0, 15), (-5, 5), (-10, 0), (2, 20)])
def test_the_ratio_is_fishtest_s(elos: tuple[float, float]) -> None:
    rng = np.random.default_rng(89)
    cases = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 100, 0], [3, 0, 5], [250, 3, 1]]
    cases += [list(rng.integers(0, 3000, size=3)) for _ in range(40)]
    for counts in cases:
        mine = float(sprt.llr(counts, *elos))
        theirs = _fishtest_llr([float(c) for c in counts], *elos)
        assert mine == pytest.approx(theirs, rel=1e-7, abs=1e-9), counts


def test_the_ratio_is_zero_where_the_hypotheses_are_equally_likely() -> None:
    """The midpoint of H0 and H1 by symmetry: equal counts of lost and won pairs put the
    mean at exactly 1/2, and with elo0 = -elo1 the two constrained fits are mirror images."""
    assert float(sprt.llr([40, 120, 40], -7.0, 7.0)) == pytest.approx(0.0, abs=1e-12)


def test_the_normal_approximation_shadows_it_at_this_project_s_sizes() -> None:
    """fishtest's `LLR_alt2` is second order in the effect, so on pairs from a +20 Elo arm
    it should sit close to the exact ratio -- close enough to be a check, not so close that
    a wrong exact ratio could hide behind it."""
    counts = np.array([1250, 3140, 1610])  # mean 0.530, about +21 Elo
    exact = float(sprt.llr(counts, 0, 10))
    approx = float(sprt.llr_normal(counts, 0, 10))
    assert exact > sprt.bounds(0.05, 0.05)[1]  # a clear H1
    assert approx == pytest.approx(exact, rel=0.02)


def test_the_bounds_are_wald_s() -> None:
    lower, upper = sprt.bounds(0.05, 0.05)
    assert lower == pytest.approx(-2.944439, abs=1e-6)
    assert upper == pytest.approx(2.944439, abs=1e-6)
    assert sprt.bounds(0.05, 0.10)[0] == pytest.approx(math.log(0.10 / 0.95))


def test_the_logistic_scale() -> None:
    assert sprt.expected_score(0) == 0.5
    assert sprt.expected_score(10) == pytest.approx(0.51439, abs=1e-5)
    assert sprt.elo_of(sprt.expected_score(-23.8)) == pytest.approx(-23.8)


def test_a_replay_stops_at_the_first_pair_a_live_test_would() -> None:
    """The vectorised replay and the one-pair-at-a-time `Sprt` are the same test."""
    rng = np.random.default_rng(7)
    scores = list(rng.choice(sprt.PAIR_SCORES, size=3000, p=[0.21, 0.55, 0.24]))
    stop = sprt.replay(scores, 0, 10)
    live = sprt.Sprt(0, 10)
    for score in scores:
        live.add(score)
        if live.decision is not None:
            break
    assert live.decision is not None
    assert stop.decision == live.decision
    assert stop.pairs == live.pairs
    assert stop.llr == pytest.approx(live.llr)


def test_identical_arms_are_turned_down_within_a_hundred_pairs() -> None:
    """The null control of every match: identical arms split every pair. Nothing about
    that is evidence of a +10 arm, and the test should say so quickly rather than never."""
    stop = sprt.replay([0.5] * 1000, 0, 10)
    assert stop.decision == "H0"
    assert 90 <= stop.pairs <= 110


def test_simulate_is_the_replay_run_many_times() -> None:
    """`simulate` grows paths a block at a time and drops a run once it crosses; the
    shortcut must not change where any run stops, least of all one that crosses in a later
    block and depends on the counts carried over from the earlier ones."""
    probabilities = [0.14, 0.67, 0.19]  # about +17 Elo: most runs decide, not all at once
    stops, decisions = sprt.simulate(
        probabilities, 0, 10, cap=3000, runs=40, rng=np.random.default_rng(5), block=700
    )
    # Redraw the same generator calls -- one per block, for the runs still going -- so each
    # run's pairs can be replayed one at a time and compared.
    rng = np.random.default_rng(5)
    alive = list(range(40))
    paths: dict[int, list[float]] = {run: [] for run in alive}
    step = 0
    while alive and step < 3000:
        width = min(700, 3000 - step)
        draws = rng.choice(3, size=(len(alive), width), p=probabilities)
        for row, run in enumerate(alive):
            paths[run].extend(sprt.PAIR_SCORES[d] for d in draws[row])
        alive = [run for run in alive if stops[run] > step + width or decisions[run] == 0]
        step += width
    for run in range(40):
        stop = sprt.replay(paths[run], 0, 10)
        if decisions[run] == 0:
            assert stop.decision is None and stops[run] == 3000
        else:
            assert stop.pairs == stops[run]
            assert stop.decision == ("H1" if decisions[run] == 1 else "H0")
    # The comparison has to contain stops, and stops past the first block.
    assert (decisions != 0).sum() >= 30
    assert ((decisions != 0) & (stops > 700)).sum() >= 5


def test_the_error_rates_are_the_registered_ones() -> None:
    """Wald's bounds are approximate, so check the operating characteristic by simulation:
    a change worth exactly elo0 should pass at most about alpha of the time, and one worth
    exactly elo1 fail at most about beta. The split rate is this project's: IKA-66's width
    match is +-0.73 at 6,000 pairs, a pair variance of 0.083, which is 67% of pairs split.

    The mean run is about 2,350 pairs with a long right tail (the first-passage time of a
    drifting walk). A run undecided at the cap is not an error -- the fixed count answers
    it -- but it should be rare.
    """
    rng = np.random.default_rng(2026)
    runs = 600
    for elo, wrong_decision in ((0.0, 1), (10.0, -1)):
        mean = sprt.expected_score(elo)
        split = 0.67
        p_won = mean - split / 2
        _stops, decisions = sprt.simulate(
            [1 - split - p_won, split, p_won], 0, 10, cap=12_000, runs=runs, rng=rng
        )
        wrong = float((decisions == wrong_decision).mean())
        undecided = float((decisions == 0).mean())
        # 5% nominal; 600 runs put one standard error at 0.9 points.
        assert wrong < 0.08, (elo, wrong)
        assert undecided < 0.02, (elo, undecided)


# -- reading pairs out of records --------------------------------------------------------


def _record_line(*, game: int, seat: int, outcome: float, nested: bool = False) -> bytes:
    """One line exactly as a match worker writes it."""
    record = GameRecord(
        own_team=[{"species": "incineroar"}],
        foe_team=[{"species": "sneasler"}],
        foe_archetype=f"arm = side {seat}",
        outcome=outcome,
        turns=9,
    )
    if nested:
        # Keys the fast reader looks for, planted where it must not find them: inside the
        # decisions, which sit between the real `outcome` and the real `provenance`.
        record.decisions.append(
            Decision(
                turn=1,
                kind="move",
                position={"outcome": 1.0 - outcome, "provenance": {"seat": "x = side 1"}},
                own_actions=[],
                own_policy=[],
                foe_actions=[],
                foe_policy=[],
                search_value=0.5,
            )
        )
    handle = io.StringIO()
    write_game(
        handle,
        record,
        objective="value:arm",
        search_limit=(24, 24),
        source=provenance("generation-match", seat=f"arm = side {seat}", leaves=("a", "b"), limits=(24, 24),
                          information=("open", "open")),
        extra={"gameIndex": game, "seatIndex": seat},
    )
    return handle.getvalue().encode("utf-8")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(("seat", "outcome"), [(0, 1.0), (0, 0.0), (1, 1.0), (1, 0.0), (1, 0.5)])
def test_the_fast_reader_reads_what_json_reads(seat: int, outcome: float, nested: bool) -> None:
    line = _record_line(game=41, seat=seat, outcome=outcome, nested=nested).rstrip(b"\n")
    assert sprt.fields_of_line(line) == sprt.fields_of_record(json.loads(line))
    index, side, won = sprt.entry(sprt.fields_of_line(line))
    assert (index, side) == (41, seat)
    # outcome is side 0's result: the tested arm at side 1 wins when side 0 loses, and a
    # tie is nobody's win.
    assert won == float(outcome > 0.5 if seat == 0 else outcome < 0.5)


def _reordered(line: bytes, first: list[str]) -> str:
    """The same record with `first` moved to the front, as a changed writer would put it."""
    game = json.loads(line)
    moved = {key: game[key] for key in first}
    moved.update({k: v for k, v in game.items() if k not in first})
    return json.dumps(moved, ensure_ascii=False) + "\n"


def test_a_changed_layout_is_refused_rather_than_misread(tmp_path) -> None:  # noqa: ANN001
    """The controls that break the named shape. Read by position, a nested `outcome` ahead
    of the real one is the silent failure -- it pairs a flipped result with no error at all
    -- so the check against a whole parse of each file's first record has to be what stops
    it. A `provenance` that is no longer last fails on its own, and has to say why."""
    line = _record_line(game=3, seat=0, outcome=1.0, nested=True)
    head = tmp_path / "head"
    head.mkdir()
    (head / "games-worker0.jsonl").write_text(_reordered(line, ["decisions"]), encoding="utf-8")
    tail = tmp_path / "tail"
    tail.mkdir()
    (tail / "games-worker0.jsonl").write_text(
        _reordered(line, ["provenance", "gameIndex", "seatIndex"]), encoding="utf-8"
    )
    for directory in (head, tail):
        with pytest.raises(ValueError, match="layout has changed"):
            sprt.PairTail(directory).poll()
    # And the check is what catches the first: without it the flipped result goes through.
    unchecked = sprt.PairTail(head, verify=False)
    unchecked.poll()
    assert unchecked.seats[3] == {0: 0.0}


def test_the_tail_reads_only_complete_lines(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "games-worker0.jsonl"
    first = _record_line(game=0, seat=0, outcome=1.0)
    second = _record_line(game=0, seat=1, outcome=1.0)
    third = _record_line(game=1, seat=0, outcome=1.0)
    path.write_bytes(first + second + third[:500])  # a worker half way through a line
    tail = sprt.PairTail(tmp_path)
    assert tail.poll() == 2
    assert tail.pair(0) == 0.5  # side 0 won both times: the arm won one seat, lost one
    assert tail.pair(1) is None
    with path.open("ab") as handle:
        handle.write(third[500:] + _record_line(game=1, seat=1, outcome=0.0))
    assert tail.poll() == 2
    assert tail.pair(1) == 1.0  # side 0 won with the arm there, lost with it opposite
    assert tail.complete() == [(0, 0.5), (1, 1.0)]


# -- stopping a live run -----------------------------------------------------------------


class _Resolved:
    """Stands in for the queue: a monitor only ever asks it what is final."""

    def __init__(self) -> None:
        self.indices: set[int] = set()

    def resolved(self) -> set[int]:
        return set(self.indices)


def test_the_monitor_reads_pairs_in_index_order_and_only_once_final(tmp_path) -> None:  # noqa: ANN001
    """A finished game whose partner is still being played, or a later game that finished
    first, must not move the test: then a live run reads the same prefix a replay reads.
    A game resolved with a seat missing -- discarded at the turn cap -- is skipped."""
    lines = b""
    for game in range(6):
        for seat in (0, 1):
            if game == 3 and seat == 1:
                continue  # hit the turn cap: finished, never written
            lines += _record_line(game=game, seat=seat, outcome=1.0 if seat == 0 else 0.0)
    (tmp_path / "games-worker0.jsonl").write_bytes(lines)
    queue = _Resolved()
    monitor = sprt.StopWhenDecided(tmp_path, sprt.Sprt(0, 10))
    # Game 2 has both records on disk but only its first seat is final -- its second is
    # being replayed after a worker died, say -- and game 3 is final.
    queue.indices = {0, 1, 2, 3, 4, 6, 7}
    assert monitor(queue) is None
    assert (monitor.next_game, monitor.test.pairs) == (2, 2)
    queue.indices |= {5}
    monitor(queue)
    assert (monitor.next_game, monitor.test.pairs, monitor.skipped) == (4, 3, 1)
    assert monitor.test.counts == [0, 0, 3]  # won both seats every time


_FAKE_WORKER = """
import json, sys, time
sys.path.insert(0, {src!r})
from pokeuraou.workqueue import WorkClient
client = WorkClient(sys.argv[1])
with open(sys.argv[2], "a", encoding="utf-8") as out:
    while (index := client.take()) is not None:
        game, seat = divmod(index, 2)
        won = (game * 2654435761 + seat * 40503) % 100 < {percent}
        outcome = float(won) if seat == 0 else float(not won)
        # write_game's layout: outcome first, decisions, then provenance and the index.
        out.write(json.dumps({{
            "outcome": outcome, "decisions": [],
            "provenance": {{"seat": f"arm = side {{seat}}"}},
            "gameIndex": game, "seatIndex": seat,
        }}) + "\\n")
        out.flush()
        time.sleep(0.002)
        client.finish(index)
client.close()
"""


def _fake_match(tmp_path: Path, *, percent: int, games: int, monitor) -> int:  # noqa: ANN001
    script = tmp_path / "worker.py"
    src = str(Path(sprt.__file__).resolve().parents[1])
    script.write_text(textwrap.dedent(_FAKE_WORKER.format(src=src, percent=percent)), encoding="utf-8")

    def build(worker: int, address: str) -> list[str]:
        return [sys.executable, str(script), address, str(tmp_path / f"games-worker{worker}.jsonl")]

    return run_workers(
        range(2 * games), build, workers=3, out_dir=tmp_path, label="match",
        monitor=monitor, poll=0.05,
    )


def test_a_match_stops_at_the_pair_its_replay_names(tmp_path) -> None:  # noqa: ANN001
    """End to end through `run_workers`: real worker processes, a queue, a monitor. The run
    has to stop early, exit 0, and have stopped at exactly the pair a replay of the records
    it left behind names -- including the games that finished after the decision."""
    record = tmp_path / "sprt.json"
    monitor = sprt.StopWhenDecided(tmp_path, sprt.Sprt(0, 10), record=record)
    status = _fake_match(tmp_path, percent=70, games=3000, monitor=monitor)
    assert status == 0
    state = json.loads(record.read_text(encoding="utf-8"))
    assert state["decision"] == "H1" and state["stoppedEarly"]
    tail = sprt.PairTail(tmp_path)
    tail.poll()
    assert tail.records < 2 * 3000, "the run played everything"
    stop = sprt.replay([score for _index, score in tail.complete()], 0, 10)
    assert (stop.decision, stop.pairs) == ("H1", state["pairs"])
    assert len(state["llrTrail"]) == state["pairs"]


def test_a_monitor_that_fails_leaves_the_run_whole_and_says_so(tmp_path) -> None:  # noqa: ANN001
    calls = []

    def broken(queue: WorkQueue) -> str | None:
        calls.append(queue)
        raise ValueError("the record layout has changed")

    status = _fake_match(tmp_path, percent=50, games=40, monitor=broken)
    assert status == 1
    tail = sprt.PairTail(tmp_path)
    tail.poll()
    assert tail.records == 80  # every game played: a broken stop is no stop at all
    assert len(calls) == 1  # not called again once it had failed


def test_match_queue_registers_its_test_before_the_first_game(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """`match_queue.py --sprt` without playing a game: the bounds are on disk before the
    driver starts a worker, the driver is handed the monitor, and a second run into the same
    directory is refused -- a test whose bounds could be rewritten after games exist is not
    a test fixed in advance."""
    match_queue = load_tool("match_queue")
    seen: dict[str, object] = {}

    def fake_run_workers(indices, build, *, monitor, **_kwargs) -> int:  # noqa: ANN001
        seen["before"] = json.loads((tmp_path / "sprt.json").read_text(encoding="utf-8"))
        seen["monitor"] = monitor
        seen["jobs"] = len(list(indices))
        return 0

    monkeypatch.setattr(match_queue, "run_workers", fake_run_workers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "match_queue.py", "--out", str(tmp_path), "--games", "5", "--value", "a.pt",
            "--sprt", "-5", "15", "--sprt-beta", "0.1", "--uniform-selection",
            "--hide-bench", "--",
        ],
    )
    with pytest.raises(SystemExit) as ended:
        match_queue.main()
    assert ended.value.code == 0
    before = seen["before"]
    assert before["decision"] is None and before["pairs"] == 0
    assert (before["registered"]["elo0"], before["registered"]["elo1"]) == (-5, 15)
    assert before["registered"]["bounds"] == pytest.approx(list(sprt.bounds(0.05, 0.1)))
    assert isinstance(seen["monitor"], sprt.StopWhenDecided)
    assert seen["jobs"] == 10  # two seats a game
    with pytest.raises(SystemExit, match="already exists"):
        match_queue.main()
