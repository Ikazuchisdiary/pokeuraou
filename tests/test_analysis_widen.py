"""IKA-354: the analysis mode widens the read it is running (`analysis.Widen`).

What these hold:

- **a wider width does not restart the read**: asked for mid-read, the menus at the new
  width join the root at the next step and the read goes on; the answer's menus hold
  the old ones first; the report counts the added actions; the same request at the same
  step gives the same answer bit for bit, with the cells expanded ahead on helper
  threads too (the count clock does not read the threads);
- **the page's request is routed**: the same position, side and guard with a wider width
  widens; a narrower width, another decision or another guard reads again from the
  start; a request that comes as the read ends is read from the start at that width;
- **over the socket**: the page's command widens a running read, and the page hears it.
"""

from __future__ import annotations

import json
import threading
import time

from pokeuraou import analysis, humanplay, liveview

from .test_analysis import _hidden_point, _prices, _settings, pool, record  # noqa: F401


def _widen_at(step_index: int, width: int):  # noqa: ANN202
    """An ``on_session`` that asks the read for ``width`` once its step ``step_index`` is
    heard (taken at the next step's top: a count, so the same step every run)."""

    def on_session(session):  # noqa: ANN001, ANN202
        heard = session.recorder

        def recorder(step):  # noqa: ANN001, ANN202
            if heard is not None:
                heard(step)
            if step.index == step_index and step.kind not in ("start", "done"):
                assert session.widen.request(width)

        session.recorder = recorder

    return on_session


def test_a_wider_width_mid_read_goes_on_from_the_tree(pool, record) -> None:  # noqa: ANN001, F811
    analyzer, game, point = _hidden_point(pool, record)
    settings = _settings(width=4, oracle=None)
    narrow = analyzer.run(game, point, settings=settings, max_steps=12)
    wide = analyzer.run(game, point, settings=settings, max_steps=40, on_session=_widen_at(12, 9))
    assert wide.width == 9 and wide.widened_to == [9]
    assert wide.deepened["grown"] > 0 and wide.deepened["grownCells"] > 0
    # The width-4 menus first, what joined after them.
    assert wide.ours[: len(narrow.ours)] == narrow.ours
    assert wide.theirs[: len(narrow.theirs)] == narrow.theirs
    assert len(wide.ours) > len(narrow.ours) or len(wide.theirs) > len(narrow.theirs)
    # The same request at the same step: the same answer, alone and on helper threads.
    again = analyzer.run(game, point, settings=settings, max_steps=40, on_session=_widen_at(12, 9))
    humanplay.use_threads(2)
    try:
        spread = analyzer.run(
            game, point, settings=settings, max_steps=40, on_session=_widen_at(12, 9)
        )
    finally:
        humanplay.use_threads(1)
    for other in (again, spread):
        assert other.strategy.tobytes() == wide.strategy.tobytes()
        assert other.value0 == wide.value0 and other.ours == wide.ours
        assert other.deepened == wide.deepened
    # The control: without the request the same 40 steps read another answer.
    plain = analyzer.run(game, point, settings=settings, max_steps=40)
    assert plain.width == 4 and not plain.widened_to and "grown" not in plain.deepened
    assert (plain.strategy.tobytes(), plain.ours) != (wide.strategy.tobytes(), wide.ours)


class _FakeSession:
    def __init__(self, width: int) -> None:
        self.widen = analysis.Widen(width, lambda w: ([], []))
        self.key = {"source": 0, "game": 1, "decision": 2, "side": 1, "guard": 8}


def test_the_page_request_widens_only_the_same_read() -> None:
    service = analysis.Service.__new__(analysis.Service)
    service.analyzer = type("A", (), {"settings": _settings(levels=8)})()
    base = {"cmd": "analyze", "source": 0, "game": 1, "decision": 2, "side": 1, "guard": 8}
    session = _FakeSession(8)
    assert service._widens(session, {**base, "width": 12})
    assert session.widen.pending == 12
    session = _FakeSession(8)
    for other in (
        {**base, "width": 8},  # the same width: read again
        {**base, "width": 6},  # narrower: read again
        {**base, "width": 12, "decision": 3},
        {**base, "width": 12, "side": 0},
        {**base, "width": 12, "guard": 16},
        {k: v for k, v in {**base, "width": 12}.items() if k != "side"},
    ):
        assert not service._widens(session, other), other
    assert session.widen.pending is None
    # No guard in the message: the service's own.
    assert service._widens(session, {**{k: v for k, v in base.items() if k != "guard"}, "width": 10})
    # A request the read never took is handed back at its end, and a closed read takes none.
    assert session.widen.close() == 10
    assert not session.widen.request(20)


def test_the_page_widens_a_running_read(pool, record) -> None:  # noqa: ANN001, F811
    analyzer = analysis.Analyzer(pool.reg, None, "hp-share", settings=_settings(width=4, oracle=None))
    service = None
    server = liveview.LiveServer(
        on_command=lambda message: service.command(message), keep_steps=50
    ).start()
    try:
        service = analysis.Service(
            analyzer, server, [analysis.Source("games", record)], max_steps=400,
            limits=analysis.Limits(rss_gb=0, free_gb=0, gpu_gb=0), pools={pool.id: pool},
        )
        loop = threading.Thread(target=service.serve, daemon=True)
        loop.start()
        host, port = server.address
        sock = liveview.ws_connect(host, port)
        decoder = liveview.Decoder()
        seen: list[dict] = []

        def until(test, limit: float = 120.0) -> dict:  # noqa: ANN001
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

        until(lambda e: e.get("type") == "catalogue")
        ask = {"cmd": "analyze", "source": 0, "game": 0, "decision": 1, "side": 0, "width": 4}
        liveview.ws_send_text(sock, json.dumps(ask))
        until(lambda e: e.get("type") == "step" and e.get("kind") == "refine")
        liveview.ws_send_text(sock, json.dumps({**ask, "width": 9}))
        done = until(lambda e: e.get("type") == "analysis" and e.get("state") == "done")
        assert done["width"] == 9
        started = [e for e in seen if e.get("type") == "analysis" and e.get("state") == "running"]
        grown = [e for e in started if e.get("grown")]
        # One read, widened once: the page heard the widening, not a second start.
        assert len(grown) == 1 and len(started) == 2, [e.get("grown") for e in started]
        assert grown[0]["width"] == 9 and grown[0]["run"] == started[0]["run"]
        assert done["steps"] == 400
        liveview.ws_send_text(sock, json.dumps({"cmd": "quit"}))
        loop.join(timeout=30)
        assert not loop.is_alive()
        sock.close()
    finally:
        server.close()


def test_a_request_as_the_read_ends_reads_from_the_start(pool, record) -> None:  # noqa: ANN001, F811
    """The request comes after the read's last step: the service reads the position again
    at the new width (nothing is lost)."""
    analyzer = analysis.Analyzer(pool.reg, None, "hp-share", settings=_settings(width=4, oracle=None))
    heard: list = []
    server = type("S", (), {
        "listener": staticmethod(lambda kind, payload: heard.append((kind, payload))),
        "begin_run": staticmethod(lambda: None),
        "status": staticmethod(lambda frame: None),
        "wire": liveview.Wire(),
    })()
    service = analysis.Service(
        analyzer, server, [analysis.Source("games", record)], max_steps=5,
        limits=analysis.Limits(rss_gb=0, free_gb=0, gpu_gb=0), pools={pool.id: pool},
    )
    ask = {"cmd": "analyze", "source": 0, "game": 0, "decision": 1, "side": 0, "width": 4}
    original = analyzer.run

    def run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        keep = kwargs["on_session"]

        def on_session(session):  # noqa: ANN001, ANN202
            keep(session)
            base = session.recorder

            def recorder(step):  # noqa: ANN001, ANN202
                if base is not None:
                    base(step)
                if step.kind == "done":
                    assert service._widens(session, {**ask, "width": 9})

            session.recorder = recorder

        kwargs["on_session"] = on_session
        return original(*args, **kwargs)

    analyzer.run = run
    first = service.analyze(ask)
    assert first.width == 4 and not first.widened_to
    again = service.commands.get_nowait()
    assert again["width"] == 9 and again["decision"] == 1
