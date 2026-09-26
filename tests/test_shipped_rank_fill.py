"""IKA-338: M-C generation and the board play q-nocover by the default Q unless told otherwise.

IKA-331 judged it (the leaf learned from q-nocover's pool is not weaker, and the pool has
1.18x the games). What these tests hold:

- **the default resolves to q-nocover**: a leaf-ranked arm that names no fill plays
  q-nocover -- in the generation worker, in both arms of a board worker -- and a
  damage-ranked arm keeps refs2, which it never reads;
- **the Q is the default one**: an unnamed Q is `qrank.DEFAULT_Q`, loaded by the worker or
  handed by the launchers to the workers;
- **a missing Q stops**: generation and the board stop before any game (and the launchers
  before any worker) with a message that names the file and ``--rank-fill refs2``; they
  never fall back to refs2 on their own;
- **refs2 named is refs2**, with no Q;
- **agent_drift sees a tool that defaults its fill apart from generation**.

The games under the new default are byte for byte the master's with
``--rank-fill q-nocover --q-model data/models/q-mc0.pt`` spelled out (records/IKA-338.md);
that needs the port, the leaf and the Q, so it is measured there rather than here.
Nothing here needs torch, CUDA or data/: the Q is a stand-in and the games are recorded
calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pokeuraou import poolplay, qrank
from pokeuraou.search import DEFAULT_RANK_FILL, SHIPPED_RANK_FILL, resolve_rank_fill

from ._harness import load_tool
from .test_pool_match import Counted, _recorder
from .test_poolplay import _stub, _variants, _write_pool


class FakeQ:
    """Stands in for `qrank.LocalQ`: remembers the file it was given, loads nothing."""

    loaded: list[Path] = []

    def __init__(self, path: Path, encoder, device: str = "cpu") -> None:  # noqa: ANN001
        self.path = Path(path)
        FakeQ.loaded.append(self.path)

    def describe(self) -> list[str]:
        return [self.path.name]


@pytest.fixture
def q_here(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    """A default Q that exists (an empty file; `FakeQ` never reads it)."""
    path = tmp_path / "models" / "q-mc0.pt"
    path.parent.mkdir()
    path.write_bytes(b"")
    FakeQ.loaded = []
    monkeypatch.setattr(qrank, "default_q", lambda: path)
    monkeypatch.setattr(qrank, "LocalQ", FakeQ)
    return path


@pytest.fixture
def no_q(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    path = tmp_path / "nowhere" / "q-mc0.pt"
    FakeQ.loaded = []
    monkeypatch.setattr(qrank, "default_q", lambda: path)
    monkeypatch.setattr(qrank, "LocalQ", FakeQ)
    return path


@pytest.fixture
def pool_file(tmp_path):  # noqa: ANN001, ANN201
    return str(_write_pool(tmp_path / "pool.json", _variants()))


def test_the_default_fill_is_q_nocover() -> None:
    assert SHIPPED_RANK_FILL == "q-nocover"
    assert qrank.DEFAULT_Q == "data/models/q-mc0.pt"
    assert resolve_rank_fill(None, True) == "q-nocover"
    # A damage-ranked arm reads no fill: it keeps the library's, so it needs no Q.
    assert resolve_rank_fill(None, False) == DEFAULT_RANK_FILL == "refs2"
    assert resolve_rank_fill("refs2", True) == "refs2"
    assert resolve_rank_fill("q", False) == "q"


def test_the_tail_decides_the_launchers_q() -> None:
    wants = qrank.tail_wants_default_q
    assert wants(["--rank-leaf"])
    assert wants(["--selection-store", "s", "--rank-leaf", "--limit", "12"])
    assert wants(["--baseline-rank-leaf", "--rank-leaf", "--rank-fill", "refs2"])
    assert not wants([])
    assert not wants(["--rank-leaf", "--rank-fill", "refs2"])
    assert not wants(["--rank-leaf", "--rank-fill", "q-nocover.long"])
    # A tail that names its own Q is handed none.
    assert not wants(["--rank-leaf", "--q-model", "x.pt"])


# ------------------------------------------------------------------ generation


def _generation_worker(monkeypatch, pool_file: str, tmp_path: Path, extra: list[str]) -> dict:  # noqa: ANN001
    import pokeuraou.pool as pool_module
    from pokeuraou.pool import load_pool

    tool = load_tool("selfplay")
    called: dict = {}

    def fake_generate(reg, pool_arg, **kwargs):  # noqa: ANN001, ANN003, ANN202
        called.update(kwargs)
        return {"games": 0, "finished": 0, "discarded_unfinished": 0, "wins": 0,
                "decisions": 0, "turns": 0, "mirror_games": 0, "mirror_wins": 0,
                "mirror_draws": 0, "path": "x"}

    pool = load_pool(pool_file)
    monkeypatch.setattr(pool_module, "load_pool", lambda name: pool)
    monkeypatch.setattr(poolplay, "generate_pool", fake_generate)
    monkeypatch.setattr(tool, "build_leaf", lambda args, reg: (_stub, "value:stub"))
    monkeypatch.setattr(sys, "argv", ["selfplay.py", "--pool", "p", "--games", "1",
                                      "--out", str(tmp_path / "g.jsonl"), *extra])
    qrank._INSTALLED.clear()
    tool.main()
    return called


def test_generation_plays_q_nocover_by_the_default_q(
    monkeypatch, pool_file, tmp_path, q_here  # noqa: ANN001
) -> None:
    called = _generation_worker(monkeypatch, pool_file, tmp_path, ["--rank-leaf"])
    assert called["rank_fill"] == "q-nocover"
    assert FakeQ.loaded == [q_here]
    assert qrank.installed().path == q_here
    # Without a leaf ranking there is no fill to read, and no Q is loaded.
    FakeQ.loaded = []
    called = _generation_worker(monkeypatch, pool_file, tmp_path, [])
    assert called["rank_fill"] == "refs2" and not FakeQ.loaded


def test_a_missing_q_stops_generation(monkeypatch, pool_file, tmp_path, no_q, capsys) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit) as stopped:
        _generation_worker(monkeypatch, pool_file, tmp_path, ["--rank-leaf"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert str(no_q) in err and "--rank-fill refs2" in err
    assert not FakeQ.loaded


def test_refs2_named_is_refs2_without_a_q(monkeypatch, pool_file, tmp_path, no_q) -> None:  # noqa: ANN001
    called = _generation_worker(monkeypatch, pool_file, tmp_path,
                                ["--rank-leaf", "--rank-fill", "refs2"])
    assert called["rank_fill"] == "refs2" and not FakeQ.loaded


def _generate_queue(monkeypatch, tmp_path: Path, argv: list[str]):  # noqa: ANN001, ANN202
    """Runs `generate_queue.main` to its `run_workers`; (exit code, worker 0's command)."""
    driver = load_tool("generate_queue")
    seen: dict[str, list[str]] = {}

    def fake_run_workers(indices, build, **_kwargs) -> int:  # noqa: ANN001
        seen["command"] = build(0, "127.0.0.1:1")
        return 0

    monkeypatch.setattr(driver, "run_workers", fake_run_workers)
    monkeypatch.setattr(sys, "argv", ["generate_queue.py", "--out", str(tmp_path / "out"),
                                      "--games", "1", "--value", "a.pt", *argv])
    with pytest.raises(SystemExit) as ended:
        driver.main()
    return ended.value.code, seen.get("command")


def test_the_generation_launcher_hands_the_default_q(
    monkeypatch, tmp_path, pool_file, q_here  # noqa: ANN001
) -> None:
    code, command = _generate_queue(monkeypatch, tmp_path,
                                    ["--pool", pool_file, "--", "--rank-leaf"])
    assert code == 0
    at = command.index("--q-model")
    assert command[at + 1] == str(q_here)
    # refs2 named: no Q.
    code, command = _generate_queue(monkeypatch, tmp_path,
                                    ["--pool", pool_file, "--", "--rank-leaf",
                                     "--rank-fill", "refs2"])
    assert code == 0 and "--q-model" not in command and "--q-arm" not in command


def test_a_missing_q_stops_the_generation_launcher(
    monkeypatch, tmp_path, pool_file, no_q  # noqa: ANN001
) -> None:
    code, command = _generate_queue(monkeypatch, tmp_path,
                                    ["--pool", pool_file, "--", "--rank-leaf"])
    assert command is None, "a worker was built for a run whose Q is missing"
    assert isinstance(code, str) and str(no_q) in code and "--rank-fill refs2" in code


# ------------------------------------------------------------------ the board


def _board_worker(tmp_path, monkeypatch, extra: list[str]) -> list[str]:  # noqa: ANN001
    tool = load_tool("pool_match")
    fills: list[str] = []
    record = _recorder([])

    def fake(reg, rng, own, foe, label, **kwargs):  # noqa: ANN001, ANN003, ANN202
        fills.append(kwargs["rank_fill"])
        return record(reg, rng, own, foe, label, **kwargs)

    monkeypatch.setattr(tool, "build_leaves",
                        lambda args, encoder: (Counted(_stub), None, "stub", None))
    monkeypatch.setattr(poolplay, "play_game", fake)
    qrank._INSTALLED.clear()
    tool.main([
        "--pool", str(_write_pool(tmp_path / "pool.json", _variants())),
        "--value", "unused.pt", "--hide-bench", "--games", "1", "--seed", "9",
        "--games-out", str(tmp_path / "m" / "games-worker0.jsonl"), "--limit", "2",
        "--baseline-hp-share", *extra,
    ])
    return fills


def test_both_board_arms_play_q_nocover_by_the_default_q(tmp_path, monkeypatch, q_here) -> None:  # noqa: ANN001
    fills = _board_worker(tmp_path, monkeypatch, ["--rank-leaf", "--baseline-rank-leaf"])
    # Two games (one per seat); play_game gets the pair of the arms' fills each time.
    assert fills and all(tuple(f) == ("q-nocover", "q-nocover") for f in fills)
    assert FakeQ.loaded == [q_here]


def test_a_missing_q_stops_the_board(tmp_path, monkeypatch, no_q, capsys) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit) as stopped:
        _board_worker(tmp_path, monkeypatch, ["--rank-leaf"])
    assert stopped.value.code == 2
    assert str(no_q) in capsys.readouterr().err
    assert not (tmp_path / "m" / "games-worker0.jsonl").exists()


def _match_queue(monkeypatch, tmp_path: Path, argv: list[str]):  # noqa: ANN001, ANN202
    driver = load_tool("match_queue")
    seen: dict[str, list[str]] = {}

    def fake_run_workers(indices, build, **_kwargs) -> int:  # noqa: ANN001
        seen["command"] = build(0, "127.0.0.1:1")
        return 0

    monkeypatch.setattr(driver, "run_workers", fake_run_workers)
    monkeypatch.setattr(sys, "argv", ["match_queue.py", "--out", str(tmp_path / "out"),
                                      "--games", "1", "--value", "a.pt", *argv])
    with pytest.raises(SystemExit) as ended:
        driver.main()
    return ended.value.code, seen.get("command")


def test_the_board_launcher_hands_the_default_q(
    monkeypatch, tmp_path, pool_file, q_here  # noqa: ANN001
) -> None:
    code, command = _match_queue(monkeypatch, tmp_path,
                                 ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt",
                                  "--", "--rank-leaf", "--baseline-rank-leaf"])
    assert code == 0
    at = command.index("--q-model")
    assert Path(command[at + 1]) == q_here


def test_a_missing_q_stops_the_board_launcher(
    monkeypatch, tmp_path, pool_file, no_q  # noqa: ANN001
) -> None:
    code, command = _match_queue(monkeypatch, tmp_path,
                                 ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt",
                                  "--", "--rank-leaf", "--baseline-rank-leaf"])
    assert command is None
    assert isinstance(code, str) and str(no_q) in code


# ------------------------------------------------------------------ agent_drift


def test_agent_drift_sees_a_fill_defaulted_apart_from_generation(tmp_path) -> None:  # noqa: ANN001
    drift = load_tool("agent_drift")
    tool = tmp_path / "board.py"
    tool.write_bytes(
        b"from pokeuraou.search import DEFAULT_RANK_FILL\n"
        b"ap.add_argument('--rank-fill', default=DEFAULT_RANK_FILL)\n"
        b"ap.add_argument('--q-model', default=None)\n"
    )
    problems = drift.resolution_drift(tool)
    assert "--rank-fill defaults to DEFAULT_RANK_FILL" in problems
    assert "--rank-fill is not resolved by resolve_rank_fill" in problems
    assert "--q-model is not resolved by tail_wants_default_q" in problems
    for name in ("selfplay", "pool_match", "generate_queue", "match_queue"):
        assert drift.resolution_drift(Path(drift.ROOT / "tools" / f"{name}.py")) == []
