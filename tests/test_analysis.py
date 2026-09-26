"""IKA-337: the analysis mode (`pokeuraou.analysis`) -- a recorded position read with no
budget until it is stopped.

What these hold:

- **the same number of steps is the same answer**: a read stopped after N steps of the
  deepening gives the mixture and the value bit for bit again -- with the page listening,
  and with the cells expanded ahead on helper threads -- and the deepening did move the
  answer on the way (the control: N steps is not the depth-1 answer);
- **a stop from elsewhere ends the read** at the top of the next step, and the read says
  why it stopped; the memory watch stops it before a limit and says which;
- **the guard's count** during the read is the one the deepening's own report ends with;
- **the sources**: a record line of a game against a person, and the point file a game
  being played writes, read back as the positions they were;
- **the page's loop**: a command from the socket starts a read, the status frames come,
  the read ends on its answer.

The pool, the agent and the games are test_liveview's: two variants of our M-B roster,
hp-share scored in the port, damage-ranked menus.
"""

from __future__ import annotations

import copy
import json
import threading
import time

import numpy as np
import pytest

from pokeuraou import analysis, deepen, humanplay, liveview
from pokeuraou.damage import register_mega_stones
from pokeuraou.humanplay import Agent, NodeTime, PolicyPerson
from pokeuraou.pool import load_pool
from pokeuraou.regulation import repo_root

PRICES = {("local", 1): NodeTime(fixed_ms=10.0, cell_ms=0.05)}


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


def _agent(pool) -> Agent:  # noqa: ANN001
    return Agent(reg=pool.reg, evaluate=None, name="hp-share", seconds=0.4, cores=1,
                 clock="count", rank_by_leaf=False, rank_fill="refs2")


@pytest.fixture(scope="module")
def record(pool, tmp_path_factory):  # noqa: ANN001, ANN201
    """A game against a stand-in person, written as play_human writes it."""
    payload, _clock, _game = humanplay.play(
        _agent(pool), PolicyPerson("random", 11), (pool.teams[0], pool.teams[1]),
        agent_side=1, seed=11, max_turns=4,
    )
    payload["pool"] = {"id": pool.id, "sha256": pool.sha256}
    path = tmp_path_factory.mktemp("rec") / "games.jsonl"
    humanplay.write_line(path, payload)
    return path


def _settings(**kwargs) -> analysis.Settings:  # noqa: ANN003
    got = {"width": 8, "oracle": 12, "levels": 8, "rank_by_leaf": False, "rank_fill": "refs2",
           "interval_ms": 0.0}
    got.update(kwargs)
    return analysis.Settings(**got)


def _hidden_point(pool, record):  # noqa: ANN001, ANN202
    """The first move decision where the read side's opponent still hides a bench."""
    game = analysis.load_games(record, pools={pool.id: pool})[0]
    analyzer = analysis.Analyzer(pool.reg, None, "hp-share", settings=_settings())
    for point in game.points:
        spreads = analyzer.spreads(game, point, point.pos(), analyzer.settings, [])
        if spreads is not None and len(spreads[1 - game.side]) > 1:
            return analyzer, game, point
    raise AssertionError("no decision with a hidden bench in the test game")


# ------------------------------------------------------------------------ the sources


def test_a_record_reads_back_as_its_move_decisions(pool, record) -> None:  # noqa: ANN001
    raw = json.loads(record.read_bytes())
    game = analysis.load_games(record, pools={pool.id: pool})[0]
    moves = [d for d in raw["decisions"] if d["kind"] == "move"]
    assert [p.turn for p in game.points] == [d["turn"] for d in moves]
    assert game.side == raw["human"]["side"] == 0
    assert game.teams is not None and [t.id for t in game.teams] == ["t0", "t1"]
    assert game.leads[0] == frozenset(raw["leads"][0])
    # The position is the record's own, and what was shown is the decision's.
    assert game.points[0].pos().to_json() == moves[0]["position"]
    assert game.points[-1].seen[1] == frozenset(moves[-1]["shownIdentities"][1])
    # A record naming no pool is read open, and says so.
    del raw["pool"]
    bare = analysis.game_from_record(raw, 0)
    assert bare.teams is None and bare.note


def test_the_point_file_of_a_game_being_played_reads_back(pool, tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "current.json"
    written = []

    def on_move(pos, seen, leads):  # noqa: ANN001, ANN202
        analysis.write_point(path, analysis.point_json(
            pool.reg, pos, 0, seen, leads, (pool.teams[0], pool.teams[1]), label="now",
        ))
        written.append((pos.to_json(), [sorted(s) for s in seen]))

    humanplay.play(
        _agent(pool), PolicyPerson("random", 3), (pool.teams[0], pool.teams[1]), agent_side=1,
        seed=3, max_turns=2, on_move=on_move,
    )
    assert written, "the game made no move"
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    game = analysis.load_point(pool.reg, path)
    (point,) = game.points
    assert point.position == written[-1][0]
    assert [sorted(s) for s in point.seen] == written[-1][1]
    assert game.side == 0 and [t.id for t in game.teams] == ["t0", "t1"]
    assert [s.species for s in game.teams[1].sets] == [s.species for s in pool.teams[1].sets]
    got = analysis.Analyzer(pool.reg, None, "hp-share", settings=_settings()).run(
        game, point, max_steps=3
    )
    assert got.steps == 3 and got.stop == "steps"


# ------------------------------------------------------------------------ the read


def test_the_same_steps_give_the_same_answer_bit_for_bit(pool, record) -> None:  # noqa: ANN001
    analyzer, game, point = _hidden_point(pool, record)
    steps = 30
    first = analyzer.run(game, point, max_steps=steps)
    assert first.stop == "steps" and first.steps == steps
    assert not first.exact and first.classes > 1
    again = analyzer.run(game, point, max_steps=steps)
    heard = []
    watched = analyzer.run(game, point, max_steps=steps, listener=lambda k, p: heard.append((k, p)))
    before = deepen.ahead_counts()["expanded"]
    humanplay.use_threads(2)  # the cells expanded ahead by helper threads (IKA-32 stage 2)
    try:
        spread = analyzer.run(game, point, max_steps=steps)
        helped = deepen.ahead_counts()["expanded"] - before
    finally:
        humanplay.use_threads(1)
    for other in (again, watched, spread):
        assert other.strategy.tobytes() == first.strategy.tobytes()
        assert other.value0 == first.value0
        assert other.ours == first.ours and other.theirs == first.theirs
        assert other.model == first.model
        assert other.deepened == first.deepened
    assert helped > 0, "the helpers expanded nothing: the threads' road was not taken"
    # The control: the steps moved the answer (else the comparison could not fail).
    start = analyzer.run(game, point, max_steps=0)
    assert start.steps == 0
    assert first.deepened["expanded"] > 0
    assert (start.strategy.tobytes(), start.value0) != (first.strategy.tobytes(), first.value0)
    # The page heard the read: its last step is the answer.
    steps_heard = [p for k, p in heard if k == "step"]
    assert steps_heard[0].kind == "start" and steps_heard[-1].kind == "done"
    assert np.asarray(steps_heard[-1].our_p).tobytes() == first.strategy.tobytes()
    done = [p for k, p in heard if k == "analysis"][-1]
    assert done["state"] == "done" and done["stop"] == "steps" and done["steps"] == steps


def test_a_stop_from_another_thread_ends_the_read(pool, record) -> None:  # noqa: ANN001
    analyzer, game, point = _hidden_point(pool, record)
    held = []
    timer = threading.Timer(0.5, lambda: held[0].halt("person"))

    def keep(session):  # noqa: ANN001, ANN202
        held.append(session)
        timer.start()

    started = time.perf_counter()
    got = analyzer.run(game, point, settings=_settings(levels=64), on_session=keep)
    took = time.perf_counter() - started
    assert got.stop == "person"
    assert got.steps > 0
    assert took < 30.0
    # A read that runs out of cells worth a step stops by itself.
    small = analyzer.run(game, point, settings=_settings(width=2, oracle=None, levels=1))
    assert small.stop == "exhausted"


def test_the_seconds_limit_and_the_memory_watch_stop_a_read(pool, record) -> None:  # noqa: ANN001
    analyzer, game, point = _hidden_point(pool, record)
    timed = analyzer.run(game, point, settings=_settings(levels=64), max_seconds=0.4)
    assert timed.stop == "time" and timed.steps > 0
    held = []
    got = analyzer.run(
        game, point, settings=_settings(levels=64),
        limits=analysis.Limits(rss_gb=1e-3, free_gb=0, gpu_gb=0), on_session=held.append,
    )
    assert got.stop == "memory"
    assert "GB" in held[0].why
    assert got.memory is not None and got.memory.rss_gb > 1e-3


def test_the_guard_count_is_the_reports(pool, record) -> None:  # noqa: ANN001
    analyzer, game, point = _hidden_point(pool, record)
    got = analyzer.run(game, point, settings=_settings(levels=2), max_steps=60)
    assert got.deepened["levels"] == 2
    assert got.guard_lines == got.deepened["lines"]["guard"]
    assert got.guard_lines > 0, "no line reached the guard: the count was not exercised"
    assert got.nodes > 0


def test_the_status_frame_reads_back() -> None:
    frame = liveview.Wire.status(
        state=1, elapsed=12.5, steps=345, guard_lines=6, nodes=789, guard=8, depth=9, threads=4,
        rss=1.5, free=20.0, gpu=5.0, gpu_total=12.0, rss_limit=10.0, free_floor=2.0,
        gpu_limit=11.0, warn=True,
    )
    got = liveview.Decoder().feed(frame)
    assert got["type"] == "status" and got["state"] == 1 and got["flags"] == 1
    assert (got["steps"], got["guardLines"], got["nodes"], got["guard"], got["depth"]) == (345, 6, 789, 8, 9)
    assert got["elapsed"] == 12.5 and got["gpuLimit"] == pytest.approx(11.0)


def test_the_page_starts_a_read_by_command_and_hears_it_end(pool, record) -> None:  # noqa: ANN001
    analyzer = analysis.Analyzer(pool.reg, None, "hp-share", settings=_settings())
    service = None
    server = liveview.LiveServer(
        on_command=lambda message: service.command(message), keep_steps=50
    ).start()
    try:
        service = analysis.Service(
            analyzer, server, [analysis.Source("games", record)], max_steps=15,
            limits=analysis.Limits(rss_gb=0, free_gb=0, gpu_gb=0), pools={pool.id: pool},
        )
        loop = threading.Thread(target=service.serve, daemon=True)
        loop.start()
        host, port = server.address
        sock = liveview.ws_connect(host, port)
        decoder = liveview.Decoder()
        seen: list[dict] = []

        def until(test, limit: float = 60.0) -> dict:  # noqa: ANN001
            deadline = time.perf_counter() + limit
            while time.perf_counter() < deadline:
                _op, payload = liveview.ws_read(sock)
                got = decoder.feed(payload)
                if got is None:
                    continue
                seen.append(got)
                if test(got):
                    return got
            raise AssertionError("timed out")

        catalogue = until(lambda e: e.get("type") == "catalogue")
        (source,) = catalogue["sources"]
        assert source["games"][0]["decisions"]
        liveview.ws_send_text(sock, json.dumps(
            {"cmd": "analyze", "source": 0, "game": 0, "decision": 1, "width": 6}
        ))
        done = until(lambda e: e.get("type") == "analysis" and e.get("state") == "done")
        assert done["stop"] == "steps" and done["steps"] == 15 and done["width"] == 6
        kinds = {e.get("type") for e in seen}
        assert {"sheets", "board", "think", "step", "answer", "status"} <= kinds
        statuses = [e for e in seen if e["type"] == "status"]
        assert statuses[-1]["state"] == analysis.STATES.index("done")
        assert statuses[-1]["steps"] == 15
        liveview.ws_send_text(sock, json.dumps({"cmd": "quit"}))
        loop.join(timeout=30)
        assert not loop.is_alive()
        sock.close()
    finally:
        server.close()
