"""Human play and the analysis take the port's roads, and start the Rust inference server (IKA-460).

`portlp.default_on_for_play` turns on stage 2.4's roads (the hidden-bench node, the menus and
everything below them) for a person's game and the analysis, not the game loop's (stage 3).
`humanplay.inference_address_or_start` starts the Rust server when no server is named and
leaves the leaf in the process otherwise. The answers on recorded decisions (roads off and
on, the Python server and the Rust one, one thread and four) are in records/IKA-460.md.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pokeuraou import humanplay, inference, portlp


def _reg() -> SimpleNamespace:
    return SimpleNamespace(meta=SimpleNamespace(format_id="gen9championsvgc2026regmc"))


def test_play_takes_stage_2_4_and_not_the_game_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (portlp.NODE_ENV, portlp.DECIDE_ENV):
        monkeypatch.delenv(name, raising=False)
    portlp.set_node(False)
    portlp.default_on_for_play()
    assert portlp.NODE[0] and portlp.MENUS[0] and portlp.DECIDE[0]
    assert not portlp.PLAY[0] and not portlp.HIDDEN[0] and not portlp.PREP[0]


def test_the_environment_turns_the_play_roads_off(monkeypatch: pytest.MonkeyPatch) -> None:
    portlp.set_node(False)
    monkeypatch.setenv(portlp.NODE_ENV, "0")
    portlp.default_on_for_play()
    assert not portlp.NODE[0]


def test_a_named_server_or_local_is_not_started_over(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(inference, "start_server", lambda *a, **k: started.append(1))
    said: list[str] = []
    monkeypatch.delenv(inference.ENV_SERVER, raising=False)
    assert humanplay.inference_address_or_start(
        "local", _reg(), [Path("v.pt")], None, "cuda", said.append) is None
    monkeypatch.setenv(inference.ENV_SERVER, "local")
    assert humanplay.inference_address_or_start(
        None, _reg(), [Path("v.pt")], None, "cuda", said.append) is None
    assert not started


def test_no_leaf_or_a_cpu_device_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(inference, "start_server", lambda *a, **k: started.append(1))
    monkeypatch.delenv(inference.ENV_SERVER, raising=False)
    assert humanplay.inference_address_or_start(None, _reg(), None, None, "cuda", print) is None
    assert humanplay.inference_address_or_start(
        None, _reg(), [Path("v.pt")], None, "cpu", print) is None
    assert not started


def test_the_server_is_started_for_a_leaf_and_stops_with_the_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(inference.ENV_SERVER, raising=False)
    seen: dict = {}

    class Process:
        def terminate(self) -> None:
            seen["terminated"] = True

    def fake_start(arms, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.update(arms=arms, **kwargs)
        return Process(), "127.0.0.1:1"

    registered: list[object] = []
    monkeypatch.setattr(inference, "start_server", fake_start)
    import atexit

    monkeypatch.setattr(atexit, "register", registered.append)
    q = tmp_path / "q.pt"
    q.write_bytes(b"")
    address = humanplay.inference_address_or_start(
        None, _reg(), [Path("v.pt")], q, "cuda", print)
    assert address == "127.0.0.1:1"
    assert seen["regulation"] == "gen9championsvgc2026regmc"
    assert seen["q_arms"] == {"q": q} and list(seen["arms"]) == ["value"]
    assert registered, "the server has to stop with the process"
    # A server that cannot take the net leaves the leaf in this process, with a note.
    said: list[str] = []

    def refuse(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise SystemExit("the Rust inference server cannot take this net")

    monkeypatch.setattr(inference, "start_server", refuse)
    assert humanplay.inference_address_or_start(
        None, _reg(), [Path("v.pt")], None, "cuda", said.append) is None
    assert said and "no Rust inference server" in said[0]
