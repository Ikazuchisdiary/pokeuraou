"""IKA-335: a board names its other arm; leaving the leaf out no longer means hp-share.

IKA-296's null control -- an arm against itself -- ran as the arm against hp-share,
because its driver left `--baseline` out and `tools/pool_match.py` took that to mean the
M-C origin. Both board entry points now stop instead, and hp-share is a flag that says so
(`--baseline-hp-share`). An explicit call is unchanged.

The refusals are tested where the old tools would have gone on: the worker plays its
games with a stub leaf and a recorder, and the queue driver reaches a fake `run_workers`.
Under the old code both calls complete, so these tests fail on them.
"""

from __future__ import annotations

import json
import sys

import pytest

from pokeuraou import poolplay

from ._harness import load_tool
from .test_pool_match import Counted, _recorder
from .test_poolplay import _stub, _variants, _write_pool

# ------------------------------------------------------------------ the worker


def _worker(tmp_path, monkeypatch):  # noqa: ANN001, ANN202
    tool = load_tool("pool_match")
    leaf = Counted(_stub)
    monkeypatch.setattr(tool, "build_leaves", lambda args, encoder: (leaf, None, "stub", None))
    monkeypatch.setattr(poolplay, "play_game", _recorder([]))
    out = tmp_path / "m" / "games-worker0.jsonl"
    argv = [
        "--pool", str(_write_pool(tmp_path / "pool.json", _variants())),
        "--value", "unused.pt", "--hide-bench", "--games", "1", "--seed", "9",
        "--games-out", str(out), "--limit", "2",
    ]
    return tool, argv, out


def test_the_worker_refuses_an_unnamed_other_arm(tmp_path, monkeypatch, capsys) -> None:  # noqa: ANN001
    tool, argv, out = _worker(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        tool.main(argv)
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    for flag in ("--baseline", "--baseline-inference-arm", "--baseline-hp-share"):
        assert flag in err
    # Nothing was played: the refusal comes before the first game.
    assert not out.exists()


@pytest.mark.parametrize(
    "named", [["--baseline", "b.pt"], ["--baseline-inference-arm", "baseline"]]
)
def test_the_worker_refuses_hp_share_beside_a_leaf(tmp_path, monkeypatch, capsys, named) -> None:  # noqa: ANN001
    tool, argv, _ = _worker(tmp_path, monkeypatch)
    if named[0] == "--baseline-inference-arm":
        # `--baseline-inference-arm` belongs to a served worker.
        argv = [a for a in argv if a not in ("--value", "unused.pt")] + [
            "--inference", "127.0.0.1:1"]
    with pytest.raises(SystemExit) as stopped:
        tool.main([*argv, *named, "--baseline-hp-share"])
    assert stopped.value.code == 2
    assert "contradicts" in capsys.readouterr().err


def test_the_worker_plays_hp_share_when_told(tmp_path, monkeypatch, capsys) -> None:  # noqa: ANN001
    tool, argv, out = _worker(tmp_path, monkeypatch)
    tool.main([*argv, "--baseline-hp-share"])
    records = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert len(records) == 2
    for record in records:
        which = record["seatIndex"]
        assert record["provenance"]["leaves"][1 - which] == "hp-share"
    # The head of the echo names both arms, hp-share included.
    err = capsys.readouterr().err
    assert "  tested arm: leaf stub" in err and "  other arm: leaf hp-share" in err


# ------------------------------------------------------------------ the queue driver


def _driver(tmp_path, monkeypatch, argv):  # noqa: ANN001, ANN202
    """Runs `match_queue.main` to its `run_workers`, and returns worker 0's command."""
    match_queue = load_tool("match_queue")
    seen: dict[str, list[str]] = {}

    def fake_run_workers(indices, build, **_kwargs) -> int:  # noqa: ANN001
        seen["command"] = build(0, "127.0.0.1:1")
        return 0

    monkeypatch.setattr(match_queue, "run_workers", fake_run_workers)
    monkeypatch.setattr(sys, "argv", ["match_queue.py", "--out", str(tmp_path / "out"),
                                      "--games", "1", "--value", "a.pt", *argv])
    with pytest.raises(SystemExit) as ended:
        match_queue.main()
    return ended.value.code, seen.get("command")


@pytest.fixture
def pool_file(tmp_path):  # noqa: ANN001, ANN201
    return str(_write_pool(tmp_path / "pool.json", _variants()))


def test_the_driver_refuses_an_unnamed_other_arm(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    for argv in (["--pool", pool_file, "--hide-bench"],
                 ["--uniform-selection", "--hide-bench"]):
        code, command = _driver(tmp_path, monkeypatch, argv)
        assert command is None, "a worker was built for a match with no named other arm"
        assert isinstance(code, str) and "--baseline-hp-share" in code
        assert "--baseline" in code


def test_the_driver_passes_hp_share_on(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    code, command = _driver(tmp_path, monkeypatch,
                            ["--pool", pool_file, "--hide-bench", "--baseline-hp-share"])
    assert code == 0
    assert "--baseline-hp-share" in command and "--baseline" not in command
    # M-B: generation_match.py has no such flag; its other arm is its --objective.
    code, command = _driver(tmp_path, monkeypatch,
                            ["--uniform-selection", "--hide-bench", "--baseline-hp-share"])
    assert code == 0
    assert "--baseline-hp-share" not in command and "--baseline" not in command


def test_an_explicit_baseline_is_passed_as_before(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    code, command = _driver(tmp_path, monkeypatch,
                            ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt"])
    assert code == 0
    at = command.index("--baseline")
    assert command[at + 1] == "b.pt" and "--baseline-hp-share" not in command
    # A leaf named in the tail counts: a served null control names the tested arm's own
    # server arm there (`-- --baseline-inference-arm value`, as IKA-307's boards do).
    # refs2 named: the shipped fill would want the default Q (IKA-338), not this test's point.
    tail = ["--baseline-inference-arm", "value", "--rank-leaf", "--rank-fill", "refs2"]
    code, command = _driver(tmp_path, monkeypatch,
                            ["--pool", pool_file, "--hide-bench", "--", *tail])
    assert code == 0
    assert command[-len(tail):] == tail and "--baseline-hp-share" not in command


@pytest.mark.parametrize(
    "argv",
    [
        ["--baseline", "b.pt", "--baseline-hp-share"],
        ["--baseline-hp-share", "--", "--objective", "faints"],
    ],
)
def test_the_driver_refuses_a_contradiction(tmp_path, monkeypatch, argv) -> None:  # noqa: ANN001
    code, command = _driver(tmp_path, monkeypatch,
                            ["--uniform-selection", "--hide-bench", *argv])
    assert command is None
    assert isinstance(code, str) and "contradicts" in code
