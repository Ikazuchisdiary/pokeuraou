"""IKA-335: a board names its other arm; leaving the leaf out no longer means hp-share.

IKA-296's null control -- an arm against itself -- ran as the arm against hp-share,
because its driver left `--baseline` out and `tools/pool_match.py` took that to mean the
M-C origin. Both board entry points now stop instead, and hp-share is a flag that says so
(`--baseline-hp-share`). An explicit call is unchanged.

The refusals are tested where the tools would otherwise go on: `tools/pool_match.py` writes
the Rust board worker's config (IKA-458 stage 3.7) and the queue driver reaches the command
that writes it, with stand-in servers and a fake `run_workers`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ._harness import load_tool
from .test_poolplay import _variants, _write_pool

# ------------------------------------------------------------------ the worker's config


def _worker(tmp_path):  # noqa: ANN001, ANN202
    tool = load_tool("pool_match")
    config = tmp_path / "m" / "worker-config.json"
    argv = [
        "--pool", str(_write_pool(tmp_path / "pool.json", _variants())),
        "--inference", "127.0.0.1:1", "--hide-bench", "--seed", "9", "--limit", "2",
        "--rust-config", str(config),
    ]
    return tool, argv, config


def test_the_worker_refuses_an_unnamed_other_arm(tmp_path, capsys) -> None:  # noqa: ANN001
    tool, argv, config = _worker(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        tool.main(argv)
    assert stopped.value.code == 2
    err = capsys.readouterr().err
    for flag in ("--baseline-inference-arm", "--baseline-hp-share"):
        assert flag in err
    # Nothing was written: the refusal comes before any server is asked.
    assert not config.exists()


def test_the_worker_refuses_hp_share_beside_a_leaf(tmp_path, capsys) -> None:  # noqa: ANN001
    tool, argv, _ = _worker(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        tool.main([*argv, "--baseline-inference-arm", "baseline", "--baseline-hp-share"])
    assert stopped.value.code == 2
    assert "contradicts" in capsys.readouterr().err


def test_the_worker_takes_hp_share_from_the_servers_arm() -> None:
    """IKA-460 step 4: `--baseline-hp-share` is the inference server's hp-share arm."""
    tool = load_tool("pool_match")
    args = tool.argparse.Namespace(
        baseline_inference_arm=None, baseline_hp_share=True, inference="127.0.0.1:1",
    )
    tool.check_other_arm(args, lambda text: pytest.fail(text))
    assert args.baseline_inference_arm == tool.HP_SHARE_ARM == "hp-share"


# ------------------------------------------------------------------ the queue driver


def _driver(tmp_path, monkeypatch, argv):  # noqa: ANN001, ANN202
    """Runs `match_queue.main` with stand-in servers to its `run_workers`; (exit code, the
    `tools/pool_match.py --rust-config` command it ran, the servers' hp-share arms)."""
    from pokeuraou import rustnet

    match_queue = load_tool("match_queue")
    seen: dict = {"hp": []}

    def fake_server_command(regulation, arms, qs, out_dir, hp_arms=()):  # noqa: ANN001, ANN202
        seen["hp"].append(tuple(hp_arms))
        return [sys.executable, "-c", "print('127.0.0.1:1', flush=True)"]

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
        seen["config"] = list(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(rustnet, "server_command", fake_server_command)
    monkeypatch.setattr(match_queue.subprocess, "run", fake_run)
    monkeypatch.setattr(match_queue, "run_workers", lambda indices, build, **_kwargs: 0)
    monkeypatch.setattr(sys, "argv", ["match_queue.py", "--out", str(tmp_path / "out"),
                                      "--games", "1", "--value", "a.pt", "--served",
                                      "--servers", "1", *argv])
    with pytest.raises(SystemExit) as ended:
        match_queue.main()
    return ended.value.code, seen.get("config"), seen["hp"]


@pytest.fixture
def pool_file(tmp_path):  # noqa: ANN001, ANN201
    return str(_write_pool(tmp_path / "pool.json", _variants()))


def test_the_driver_refuses_an_unnamed_other_arm(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    code, command, servers = _driver(tmp_path, monkeypatch, ["--pool", pool_file, "--hide-bench"])
    assert command is None and servers == [], "a board with no named other arm was started"
    assert isinstance(code, str) and "--baseline-hp-share" in code
    assert "--baseline" in code


def test_the_driver_gives_the_servers_an_hp_share_arm(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    """With `--baseline-hp-share` every server is started with an hp-share arm and the config is
    written with `--baseline-hp-share` (the control: no flag, no arm)."""
    code, command, servers = _driver(tmp_path, monkeypatch,
                                     ["--pool", pool_file, "--hide-bench", "--baseline-hp-share"])
    assert code == 0 and servers == [("hp-share",)]
    assert "--baseline-hp-share" in command and "--inference" in command
    code, command, servers = _driver(tmp_path, monkeypatch,
                                     ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt"])
    assert code == 0 and servers == [()] and "--baseline-hp-share" not in command


def test_an_explicit_baseline_is_passed_as_before(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    code, command, _ = _driver(tmp_path, monkeypatch,
                               ["--pool", pool_file, "--hide-bench", "--baseline", "b.pt"])
    assert code == 0
    at = command.index("--baseline-inference-arm")
    assert command[at + 1] == "baseline" and "--baseline-hp-share" not in command
    # A leaf named in the tail counts: a served null control names the tested arm's own
    # server arm there (`-- --baseline-inference-arm value`, as IKA-307's boards do).
    tail = ["--baseline-inference-arm", "value", "--rank-leaf", "--rank-fill", "refs2"]
    code, command, _ = _driver(tmp_path, monkeypatch,
                               ["--pool", pool_file, "--hide-bench", "--", *tail])
    assert code == 0
    assert command[-len(tail) - 2:-2] == tail and "--baseline-hp-share" not in command


@pytest.mark.parametrize("argv", [["--baseline", "b.pt", "--baseline-hp-share"]])
def test_the_driver_refuses_a_contradiction(tmp_path, monkeypatch, pool_file, argv) -> None:  # noqa: ANN001
    code, command, _ = _driver(tmp_path, monkeypatch,
                               ["--pool", pool_file, "--hide-bench", *argv])
    assert command is None
    assert isinstance(code, str) and "contradicts" in code


def test_a_failed_worker_config_ends_the_servers(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    """The Rust board worker's config is written after the servers are up; when that child fails
    the driver stops, and the servers it started must not stay behind holding the GPU."""
    import subprocess

    from pokeuraou import rustnet

    match_queue = load_tool("match_queue")
    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    def fake_server_command(regulation, arms, qs, out_dir, hp_arms=()):  # noqa: ANN001, ANN202
        # Names an address, then stays alive like a real server until it is terminated.
        return [sys.executable, "-c",
                "import time; print('127.0.0.1:1', flush=True); time.sleep(120)"]

    def failing_config(command, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        return subprocess.CompletedProcess(command, 1, stdout="refused: nothing to play\n")

    monkeypatch.setattr(rustnet, "server_command", fake_server_command)
    monkeypatch.setattr(match_queue.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(match_queue.subprocess, "run", failing_config)
    monkeypatch.setattr(sys, "argv", [
        "match_queue.py", "--out", str(tmp_path / "out"), "--games", "1", "--value", "a.pt",
        "--pool", pool_file, "--hide-bench", "--baseline", "b.pt", "--served", "--servers", "1"])
    try:
        with pytest.raises(SystemExit) as ended:
            match_queue.main()
        assert isinstance(ended.value.code, str) and "config was not written" in ended.value.code
        assert len(started) == 1, "the fake server was not the one started"
        assert started[0].poll() is not None, "the server was left running"
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()


def _recorded_servers(monkeypatch, match_queue, commands):  # noqa: ANN001, ANN202
    """Stand-in servers, one command per server in turn; the processes started."""
    import subprocess

    from pokeuraou import rustnet

    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen
    turn = iter(commands)

    def recording_popen(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(rustnet, "server_command", lambda *args, **kwargs: next(turn))
    monkeypatch.setattr(match_queue.subprocess, "Popen", recording_popen)
    return started


ALIVE = [sys.executable, "-c", "import time; print('127.0.0.1:1', flush=True); time.sleep(120)"]
SILENT = [sys.executable, "-c", "pass"]


def test_a_server_that_names_no_address_ends_the_ones_before_it(
    tmp_path, monkeypatch, pool_file  # noqa: ANN001
) -> None:
    """The second server exits without an address: the first is ended and waited for."""
    match_queue = load_tool("match_queue")
    started = _recorded_servers(monkeypatch, match_queue, [ALIVE, SILENT])
    monkeypatch.setattr(sys, "argv", [
        "match_queue.py", "--out", str(tmp_path / "out"), "--games", "1", "--value", "a.pt",
        "--pool", pool_file, "--hide-bench", "--baseline", "b.pt", "--served", "--servers", "2"])
    try:
        with pytest.raises(SystemExit) as ended:
            match_queue.main()
        assert isinstance(ended.value.code, str) and "before naming an address" in ended.value.code
        assert len(started) == 2
        assert all(p.poll() is not None for p in started), "a server was left running"
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()


def test_an_error_after_the_servers_ends_them(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    """An exception that is not SystemExit between the servers and the board (here an OSError
    from the config's child) still ends the servers."""
    match_queue = load_tool("match_queue")
    started = _recorded_servers(monkeypatch, match_queue, [ALIVE])

    def broken(command, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        raise OSError("no such interpreter")

    monkeypatch.setattr(match_queue.subprocess, "run", broken)
    monkeypatch.setattr(sys, "argv", [
        "match_queue.py", "--out", str(tmp_path / "out"), "--games", "1", "--value", "a.pt",
        "--pool", pool_file, "--hide-bench", "--baseline", "b.pt", "--served", "--servers", "1"])
    try:
        with pytest.raises(OSError, match="no such interpreter"):
            match_queue.main()
        assert len(started) == 1
        assert started[0].poll() is not None, "the server was left running"
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
