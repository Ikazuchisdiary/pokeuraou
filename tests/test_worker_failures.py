"""A worker that dies must not leave a run looking whole (IKA-336).

The queue hands a dead worker's games to the living, which is right for the games -- a game
is seeded by its number, so the replay writes the same bytes -- and wrong for everything a
run measures besides them. IKA-307's cost runs lost 18 to 23 of 24 workers to CUDA OOM, the
rest replayed the games, 300 games arrived, and the CPU per game and the wall clock were
quietly those of a run nobody configured. The positive control is the first test: on the
driver before this change the run played every game and the failing worker was one line at
the end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pokeuraou import workqueue
from pokeuraou.workqueue import RUN_RECORD, run_workers, worker_error

SRC = str(Path(workqueue.__file__).resolve().parents[1])

# Worker 1 raises the error a served worker gets when its server is out of memory; the
# others play slowly enough that the run is still going when it does.
_WORKER = """
import sys, time
sys.path.insert(0, {src!r})
from pokeuraou.workqueue import WorkClient
client = WorkClient(sys.argv[1])
failing = sys.argv[2] == "1"
if failing:
    from pokeuraou.inference import ServerOutOfMemory
with open(sys.argv[3], "a", encoding="utf-8") as out:
    while (index := client.take()) is not None:
        if failing:
            print(f"playing game {{index}}", flush=True)
            raise ServerOutOfMemory(
                "inference failed: the inference server ran out of CUDA memory (its cap: "
                "3.5 GB) -- OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB"
            )
        out.write(f"{{index}}\\n")
        out.flush()
        time.sleep({pause})
        client.finish(index)
client.close()
"""


def _run(tmp_path: Path, *, jobs: int, pause: float, **kwargs: Any) -> tuple[int, list[int]]:
    script = tmp_path / "worker.py"
    script.write_bytes(textwrap.dedent(_WORKER.format(src=SRC, pause=pause)).encode("utf-8"))

    def build(worker: int, address: str) -> list[str]:
        return [sys.executable, str(script), address, "1" if worker == 1 else "0",
                str(tmp_path / f"games-worker{worker}.jsonl")]

    status = run_workers(range(jobs), build, workers=3, out_dir=tmp_path, label="test",
                         **kwargs)
    played = [
        int(line)
        for path in tmp_path.glob("games-worker*.jsonl")
        for line in path.read_text(encoding="utf-8").split()
    ]
    return status, played


def test_a_failed_worker_stops_the_run_by_default(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    """The positive control: before IKA-336 this played all 300 games and exited 1 with
    "1 worker(s) failed" on the last line, and nothing recorded the failure."""
    status, played = _run(tmp_path, jobs=300, pause=0.02)
    assert status != 0
    assert len(played) < 300, "the run played everything: the failure did not stop it"
    assert len(played) == len(set(played))
    record = json.loads((tmp_path / RUN_RECORD).read_bytes())
    assert record["failed"] == 1 and record["outOfMemory"] == 1
    assert record["stoppedForFailures"] and record["finished"]
    assert record["dropped"] > 0
    failure = record["failures"][0]
    assert failure["worker"] == 1 and failure["exit"] != 0 and failure["oom"]
    assert "ServerOutOfMemory" in failure["error"] and "3.5 GB" in failure["error"]
    err = capsys.readouterr().err
    assert "WORKER FAILED: worker 1" in err
    assert "STOPPED FOR A FAILED WORKER" in err


def test_generation_s_budget_lets_the_run_finish_and_says_so(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    """Allowed failures (generation's budget) keep the run going: every game is played, the
    failure is still counted and recorded, and the exit code is still not 0."""
    status, played = _run(tmp_path, jobs=300, pause=0.02, max_failures=1)
    assert status == 1
    assert sorted(played) == list(range(300))
    record = json.loads((tmp_path / RUN_RECORD).read_bytes())
    assert record["failed"] == 1 and record["stoppedForFailures"] is None
    assert record["replayed"] >= 1  # the game the dead worker held went back on the queue
    assert "1 worker(s) failed and the run went on" in capsys.readouterr().err


def test_a_clean_run_records_zero_failures(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    script = tmp_path / "worker.py"
    script.write_bytes(textwrap.dedent(_WORKER.format(src=SRC, pause=0.0)).encode("utf-8"))

    def build(worker: int, address: str) -> list[str]:
        return [sys.executable, str(script), address, "0",
                str(tmp_path / f"games-worker{worker}.jsonl")]

    assert run_workers(range(20), build, workers=2, out_dir=tmp_path, label="test") == 0
    record = json.loads((tmp_path / RUN_RECORD).read_bytes())
    assert (record["failed"], record["failures"], record["stoppedForFailures"]) == (0, [], None)
    assert (tmp_path / RUN_RECORD).read_bytes().count(b"\r") == 0
    assert ", 0 worker(s) failed" in capsys.readouterr().err


def test_the_error_is_the_traceback_s_last_line_and_oom_comes_first(tmp_path: Path) -> None:
    log = tmp_path / "worker0.log"
    log.write_bytes(
        b"Traceback (most recent call last):\n"
        b'  File "x.py", line 1, in <module>\n'
        b"ValueError: something else first\n"
        b"Traceback (most recent call last):\n"
        b"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        b"12 games in 30.0s\n"  # stdout, flushed after the traceback
    )
    assert worker_error(log) == (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", True
    )
    log.write_bytes(b"Traceback (most recent call last):\nKeyError: 'x'\nbye\n")
    assert worker_error(log) == ("KeyError: 'x'", False)
    log.write_bytes(b"just a line\n")
    assert worker_error(log) == ("just a line", False)


def test_a_server_oom_reaches_the_worker_as_what_it_is(capsys) -> None:  # noqa: ANN001
    """The server says an OOM is one, counts it in its log, and the worker raises
    `ServerOutOfMemory` rather than a RuntimeError like any other."""
    from pokeuraou.inference import RemoteValue, ServerOutOfMemory, serve

    class OutOfMemoryError(RuntimeError):  # torch.OutOfMemoryError's name
        pass

    def full(_arrays: dict, _rows: int) -> np.ndarray:
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB; 3.26 GiB allowed")

    def broken(_arrays: dict, _rows: int) -> np.ndarray:
        raise ValueError("not a memory problem")

    server, address = serve({"full": full, "broken": broken})
    server.memory_cap_gb = 3.5
    encoded = types.SimpleNamespace(**{
        name: np.zeros((4, 2), dtype=np.int32) for name in
        ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    })
    def failure(model: str) -> tuple[type, str]:
        # Caught here and not by `pytest.raises`, whose traceback would keep the frame's
        # views of the shared block alive and make closing it a BufferError.
        with RemoteValue(address, model, None, buffer_bytes=1 << 20) as remote:
            try:
                remote.from_encoded(encoded)
            except RuntimeError as error:
                caught = (type(error), str(error))
            else:
                caught = (type(None), "")
        return caught

    try:
        kind, said = failure("full")
        assert kind is ServerOutOfMemory
        assert "out of CUDA memory" in said and "3.5 GB" in said and "3.26 GiB allowed" in said
        kind, said = failure("broken")
        assert kind is RuntimeError
        assert said == "inference failed: ValueError: not a memory problem"
    finally:
        server.shutdown()
    assert server.oom_replies == 1
    err = capsys.readouterr().err
    assert err.count(workqueue.SERVER_OOM_MARK) == 1


def test_server_ooms_are_counted_in_the_logs(tmp_path: Path) -> None:
    (tmp_path / "inference0.log").write_bytes(
        f"  {workqueue.SERVER_OOM_MARK} #1 to a 'score' request\nother\n".encode()
    )
    (tmp_path / "inference1.log").write_bytes(
        f"  {workqueue.SERVER_OOM_MARK} #1\n  {workqueue.SERVER_OOM_MARK} #2\n".encode()
    )
    logs = [tmp_path / f"inference{i}.log" for i in range(3)]  # the third never written
    assert workqueue.server_ooms(logs) == 3


def test_a_q_request_s_oom_is_named_too() -> None:
    from pokeuraou.inference import ServerOutOfMemory, request_failed

    oom = request_failed({"ok": False, "oom": True, "capGb": 3.5, "error": "OutOfMemoryError: x"},
                         "Q request failed")
    assert isinstance(oom, ServerOutOfMemory) and str(oom).startswith("Q request failed: ")
    other = request_failed({"ok": False, "error": "KeyError: 'q'"}, "Q request failed")
    assert type(other) is RuntimeError and str(other) == "Q request failed: KeyError: 'q'"


@pytest.fixture(scope="module")
def profile_stages() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "profile_stages.py"
    spec = importlib.util.spec_from_file_location("_profile_stages_ika336", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_timing_run_stops_at_its_first_failure(profile_stages: Any) -> None:
    import argparse

    args = argparse.Namespace(
        workload="generation", out=Path("out"), games=24, first_game=None, workers=None,
        seed=None, roster=None, value="v.pt", limit=None, selection_book=None,
        uniform_selection=False, force_lead=None, device="cuda", no_bridge=False,
        served=True, servers=2, hide_bench=True, rest=["--rank-leaf"],
    )
    command = profile_stages.build(args)
    at = command.index("--max-worker-failures")
    assert command[at + 1] == "0" and at < command.index("--")


_FAILED_DRIVER = """
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "workers.json").write_bytes(json.dumps({
    "failed": 1, "outOfMemory": 1, "serverOutOfMemory": 0,
    "failures": [{"worker": 3, "exit": 1, "error": "ServerOutOfMemory: full", "oom": True}],
}).encode())
sys.exit(1)
"""


def test_profile_stages_exits_with_its_driver_s_failure(
    profile_stages: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through `main`, with a driver whose worker failed. The positive control:
    master's profile_stages printed an aside, wrote the summary and exited 0, so an ABBA
    driver reading the exit code took the run for a clean one."""
    driver = tmp_path / "driver.py"
    driver.write_bytes(_FAILED_DRIVER.encode())
    out = tmp_path / "games"
    monkeypatch.setattr(profile_stages, "build",
                        lambda _args: [sys.executable, str(driver), str(out)])
    monkeypatch.setattr(sys, "argv", [
        "profile_stages.py", "generation", "--no-timing", "--hide-bench", "--games", "1",
        "--timing-dir", str(tmp_path / "timing"), "--out", str(out),
    ])
    with pytest.raises(SystemExit) as stopped:
        profile_stages.main()
    assert stopped.value.code == 1
    summary = json.loads((tmp_path / "timing" / "summary.json").read_bytes())
    assert summary["exit"] == 1 and summary["workers"]["failed"] == 1
    assert "worker 3" in summary["failed"]


def test_a_failed_driver_or_worker_is_a_failed_timing(profile_stages: Any, tmp_path: Path) -> None:
    """Before IKA-336 profile_stages exited 0 whatever its driver did."""
    verdict = profile_stages.run_verdict
    assert verdict(0, {"failed": 0, "failures": []}) is None
    assert verdict(0, None) is None
    assert "exited 1" in verdict(1, None)
    record = {"failed": 2, "outOfMemory": 1, "serverOutOfMemory": 5,
              "failures": [{"worker": 7, "error": "ServerOutOfMemory: ..."}]}
    said = verdict(1, record)
    assert "2 worker(s) failed" in said and "worker 7" in said and "5 out-of-memory" in said
    (tmp_path / "workers.json").write_bytes(json.dumps(record).encode())
    assert profile_stages.worker_record(tmp_path) == record
    # An earlier run's record in the same directory is not this run's.
    assert profile_stages.worker_record(tmp_path, since=4e9) is None
