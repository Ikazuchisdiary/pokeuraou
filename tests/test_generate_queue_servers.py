"""IKA-458 stage 3.7: generation's launcher ends the inference servers it started, however it stops.

`tools/generate_queue.py` starts the servers, then writes the Rust workers' config with
`tools/selfplay.py --rust-config`, then runs the workers. A leftover server holds VRAM and answers
the next run with the previous run's model. What these tests hold, with stand-in servers (a
process that names an address and sleeps): the servers are ended when the config's child fails
(a removed flag after `--` stops it), when starting that child raises an OSError, and when a
later server names no address.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ._harness import load_tool
from .test_explicit_baseline import ALIVE, SILENT, _recorded_servers
from .test_poolplay import _variants, _write_pool


@pytest.fixture
def pool_file(tmp_path):  # noqa: ANN001, ANN201
    return str(_write_pool(tmp_path / "pool.json", _variants()))


def _argv(tmp_path, pool_file: str, servers: int = 1) -> list[str]:  # noqa: ANN001
    return ["generate_queue.py", "--out", str(tmp_path / "out"), "--games", "1", "--value", "a.pt",
            "--pool", pool_file, "--hide-bench", "--served", "--servers", str(servers)]


def _ended(started: list) -> None:  # noqa: ANN001
    try:
        assert started, "no stand-in server was started"
        assert all(p.poll() is not None for p in started), "a server was left running"
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()


def test_a_failed_worker_config_ends_the_servers(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    driver = load_tool("generate_queue")
    started = _recorded_servers(monkeypatch, driver, [ALIVE])
    monkeypatch.setattr(driver.subprocess, "run", lambda command, **kwargs: (
        subprocess.CompletedProcess(command, 2, stdout="unrecognized arguments: --deepen m8\n")))
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, pool_file))
    with pytest.raises(SystemExit) as stopped:
        driver.main()
    assert isinstance(stopped.value.code, str) and "config was not written" in stopped.value.code
    _ended(started)


def test_an_error_after_the_servers_ends_them(tmp_path, monkeypatch, pool_file) -> None:  # noqa: ANN001
    driver = load_tool("generate_queue")
    started = _recorded_servers(monkeypatch, driver, [ALIVE])

    def broken(command, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        raise OSError("no such interpreter")

    monkeypatch.setattr(driver.subprocess, "run", broken)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, pool_file))
    with pytest.raises(OSError, match="no such interpreter"):
        driver.main()
    _ended(started)


def test_a_server_that_names_no_address_ends_the_ones_before_it(
    tmp_path, monkeypatch, pool_file  # noqa: ANN001
) -> None:
    driver = load_tool("generate_queue")
    started = _recorded_servers(monkeypatch, driver, [ALIVE, SILENT])
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, pool_file, servers=2))
    with pytest.raises(SystemExit) as stopped:
        driver.main()
    assert isinstance(stopped.value.code, str) and "before naming an address" in stopped.value.code
    assert len(started) == 2
    _ended(started)
