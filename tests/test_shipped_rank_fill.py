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
Nothing here needs torch, CUDA or data/: the Q is a stand-in, and the tools stop at the
Rust worker's config or at the first inference server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pokeuraou import qrank, rustworker
from pokeuraou.search import DEFAULT_RANK_FILL, SHIPPED_RANK_FILL, resolve_rank_fill

from ._harness import load_tool
from .test_poolplay import _stub, _variants, _write_pool


class FakeQ:
    """Stands in for `qrank.LocalQ`: remembers the file it was given, loads nothing."""

    loaded: list[Path] = []

    def __init__(self, path: Path, encoder, device: str = "cpu") -> None:  # noqa: ANN001
        self.path = Path(path)
        FakeQ.loaded.append(self.path)

    def describe(self) -> list[str]:
        return [self.path.name]

    def digests(self) -> list[str]:
        return [qrank.file_sha256(self.path)]


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
    assert qrank.DEFAULT_Q == "data/models/q-mc6.pt"
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
    """`tools/selfplay.py` to the config it would write: its flags and the Q it named."""
    import pokeuraou.pool as pool_module
    from pokeuraou.pool import load_pool

    tool = load_tool("selfplay")
    called: dict = {}

    def fake_config(args, pool_arg, evaluate, leaf_label, q_record, pairs, store):  # noqa: ANN001, ANN202, PLR0913
        called.update(rank_fill=args.rank_fill, q_record=q_record)
        return {}

    pool = load_pool(pool_file)
    monkeypatch.setattr(pool_module, "load_pool", lambda name: pool)
    monkeypatch.setattr(rustworker, "build_config", fake_config)
    monkeypatch.setattr(rustworker, "write_config", lambda config, path: None)
    monkeypatch.setattr(tool, "build_leaf", lambda args, reg: (_stub, "value:stub"))
    monkeypatch.setattr(sys, "argv", ["selfplay.py", "--pool", "p", "--inference", "h:1",
                                      "--rust-config", str(tmp_path / "c.json"), *extra])
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
    # The config's records name the Q and its sha256 (IKA-340).
    assert called["q_record"] == {"qModel": ["q-mc0.pt"],
                                  "qModelSha256": [qrank.file_sha256(q_here)]}


def test_a_missing_q_stops_generation(monkeypatch, pool_file, tmp_path, no_q, capsys) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit) as stopped:
        _generation_worker(monkeypatch, pool_file, tmp_path, ["--rank-leaf"])
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    assert str(no_q) in err and "--rank-fill refs2" in err
    assert not FakeQ.loaded


def test_refs2_named_and_the_damage_ranking_are_refused(
    monkeypatch, pool_file, tmp_path, no_q, capsys  # noqa: ANN001
) -> None:
    """The Rust worker plays a q-nocover fill only (IKA-458 stage 3.7): refs2 named, and no
    leaf ranking, stop before any Q is loaded."""
    for extra in (["--rank-leaf", "--rank-fill", "refs2"], []):
        with pytest.raises(SystemExit):
            _generation_worker(monkeypatch, pool_file, tmp_path, extra)
        assert "the Rust worker does not play" in capsys.readouterr().err
    assert not FakeQ.loaded


def _launcher(tool: str, monkeypatch, tmp_path: Path, argv: list[str]):  # noqa: ANN001, ANN202
    """Runs a launcher's `main` to its first inference server; (exit code, the server's Qs)."""
    from pokeuraou import rustnet

    driver = load_tool(tool)
    seen: dict[str, dict] = {}

    def fake_server(regulation, arms, qs, out_dir, *rest):  # noqa: ANN001, ANN002, ANN202
        seen["qs"] = qs
        raise SystemExit(0)

    monkeypatch.setattr(rustnet, "server_command", fake_server)
    monkeypatch.setattr(sys, "argv", [f"{tool}.py", "--out", str(tmp_path / "out"),
                                      "--games", "1", "--value", "a.pt", "--served", *argv])
    with pytest.raises(SystemExit) as ended:
        driver.main()
    return ended.value.code, seen.get("qs")


def test_the_generation_launcher_hands_the_default_q(
    monkeypatch, tmp_path, pool_file, q_here  # noqa: ANN001
) -> None:
    code, qs = _launcher("generate_queue", monkeypatch, tmp_path,
                         ["--pool", pool_file, "--", "--rank-leaf"])
    assert code == 0 and qs == {"q": str(q_here)}
    # refs2 named: no Q.
    code, qs = _launcher("generate_queue", monkeypatch, tmp_path,
                         ["--pool", pool_file, "--", "--rank-leaf", "--rank-fill", "refs2"])
    assert code == 0 and qs == {}


def test_a_missing_q_stops_the_generation_launcher(
    monkeypatch, tmp_path, pool_file, no_q  # noqa: ANN001
) -> None:
    code, qs = _launcher("generate_queue", monkeypatch, tmp_path,
                         ["--pool", pool_file, "--", "--rank-leaf"])
    assert qs is None, "a server was started for a run whose Q is missing"
    assert isinstance(code, str) and str(no_q) in code and "--rank-fill refs2" in code


# ------------------------------------------------------------------ the board


def _board_worker(tmp_path, monkeypatch, extra: list[str]) -> list[str]:  # noqa: ANN001
    """`tools/pool_match.py` to the config it would write: the arms' fills."""
    tool = load_tool("pool_match")
    fills: list[str] = []

    def fake_config(args, pool, arms, arm_label, q_record, pairs):  # noqa: ANN001, ANN202, PLR0913
        fills.extend(arm.rank_fill for arm in arms)
        return {}

    monkeypatch.setattr(tool, "build_leaves",
                        lambda args, encoder: (_stub, _stub, "stub", "stub"))
    monkeypatch.setattr(rustworker, "board_config", fake_config)
    monkeypatch.setattr(rustworker, "write_config", lambda config, path: None)
    qrank._INSTALLED.clear()
    tool.main([
        "--pool", str(_write_pool(tmp_path / "pool.json", _variants())),
        "--inference", "h:1", "--baseline-inference-arm", "value", "--hide-bench",
        "--seed", "9", "--limit", "2", "--rust-config", str(tmp_path / "c.json"), *extra,
    ])
    return fills


def test_both_board_arms_play_q_nocover_by_the_default_q(tmp_path, monkeypatch, q_here) -> None:  # noqa: ANN001
    fills = _board_worker(tmp_path, monkeypatch, ["--rank-leaf", "--baseline-rank-leaf"])
    assert fills == ["q-nocover", "q-nocover"]
    assert FakeQ.loaded == [q_here]


def test_a_missing_q_stops_the_board(tmp_path, monkeypatch, no_q, capsys) -> None:  # noqa: ANN001
    with pytest.raises(SystemExit) as stopped:
        _board_worker(tmp_path, monkeypatch, ["--rank-leaf", "--baseline-rank-leaf"])
    assert stopped.value.code == 2
    assert str(no_q) in capsys.readouterr().err


def test_the_board_launcher_hands_the_default_q(
    monkeypatch, tmp_path, pool_file, q_here  # noqa: ANN001
) -> None:
    code, qs = _launcher("match_queue", monkeypatch, tmp_path,
                         ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt",
                          "--", "--rank-leaf", "--baseline-rank-leaf"])
    assert code == 0 and Path(qs["q"]) == q_here


def test_a_missing_q_stops_the_board_launcher(
    monkeypatch, tmp_path, pool_file, no_q  # noqa: ANN001
) -> None:
    code, qs = _launcher("match_queue", monkeypatch, tmp_path,
                         ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt",
                          "--", "--rank-leaf", "--baseline-rank-leaf"])
    assert qs is None
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
