"""The stage timers, which are read as a breakdown and so have to add up.

Three properties, and each one is a way the table could lie while still printing numbers:

- **off is free and inert.** Unset, `timed` hands the function straight back and `stage`
  hands back a shared no-op, so a decorated hot path is the undecorated one. A timer that
  cost anything in a normal run would change the thing it was built to report.
- **a stage holds its own time, never its nestings.** Narrowing that ranks by the leaf
  calls the encoder and the net from inside itself; if the narrowing row contained them,
  encode and forward would appear twice in a table whose whole purpose is to add up, and
  the total would exceed the run.
- **threads do not share a stack.** The inference server answers each connection on its
  own thread. One shared stack would pop another thread's entry, which is not a slightly
  wrong number -- it is a negative one.

The module is loaded from its file under a private name rather than reloaded in place,
because `resolve`, `narrow` and the rest hold a reference to the real one and a reload
would leave half the package pointing at a module the test had configured.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "pokeuraou" / "timing.py"


def load(
    monkeypatch: pytest.MonkeyPatch,
    directory: Path | None,
    *,
    dupes: bool = False,
    sample_hz: float = 0.0,
    regions: bool = False,
    sample_cpu: bool = False,
) -> Any:
    """A private copy of the module, with the environment it reads at import time."""
    monkeypatch.setenv("POKEURAOU_SAMPLE_CPU", "1" if sample_cpu else "")
    if directory is None:
        monkeypatch.delenv("POKEURAOU_TIMING", raising=False)
    else:
        monkeypatch.setenv("POKEURAOU_TIMING", str(directory))
    monkeypatch.setenv("POKEURAOU_TIMING_DUPES", "1" if dupes else "")
    monkeypatch.setenv("POKEURAOU_TIMING_REGIONS", "1" if regions else "")
    monkeypatch.setenv("POKEURAOU_SAMPLE_HZ", f"{sample_hz:g}" if sample_hz else "")
    spec = importlib.util.spec_from_file_location("_timing_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def burn(seconds: float) -> None:
    """Busy-wait, so the CPU clock moves as well as the wall clock."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_off_is_the_undecorated_function(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = load(monkeypatch, None)
    assert timing.ON is False

    def work(x: int) -> int:
        return x + 1

    assert timing.timed("branch")(work) is work
    # And the same no-op object every time, rather than one built per call.
    assert timing.stage("branch") is timing.stage("encode")
    with timing.stage("branch"):
        pass
    assert timing.snapshot()["stages"] == {}


def test_off_records_nothing_through_add(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = load(monkeypatch, None)
    timing.add("rust.child.encode", 1.0)
    assert timing.snapshot()["stages"] == {}


def test_a_stage_excludes_what_it_nests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timing = load(monkeypatch, tmp_path)
    with timing.stage("narrow"):
        burn(0.05)
        with timing.stage("encode"):
            burn(0.10)
        burn(0.05)

    stages = timing.snapshot()["stages"]
    # The outer stage kept its own 0.10 and not the 0.10 it contained.
    assert stages["encode"]["wall"] == pytest.approx(0.10, abs=0.04)
    assert stages["narrow"]["wall"] == pytest.approx(0.10, abs=0.04)
    # The property that matters is the one the table depends on: the rows add to the
    # elapsed time rather than to more than it.
    total = sum(row["wall"] for row in stages.values())
    assert total <= timing.snapshot()["elapsed"] + 1e-6


def test_recursion_counts_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`batched_payoffs` fills a refused cell by calling itself; so does `resolve_turn`."""
    timing = load(monkeypatch, tmp_path)

    @timing.timed("branch")
    def outer(depth: int) -> None:
        burn(0.02)
        if depth:
            outer(depth - 1)

    outer(3)
    row = timing.snapshot()["stages"]["branch"]
    assert row["calls"] == 4
    # Four nested 0.02 s bodies: 0.08 s of stage, not 0.08 + 0.06 + 0.04 + 0.02.
    assert row["wall"] == pytest.approx(0.08, abs=0.03)


def test_borrowed_rows_are_not_this_process_s_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    with timing.stage("rust.fill"):
        burn(0.05)
    # What the Rust child said it spent, which is inside the wait above rather than beside
    # it. Adding both into one total would count the same seconds twice.
    timing.add("rust.child.resolve", 0.03)
    timing.add("rust.child.encode", 0.01)

    stages = timing.snapshot()["stages"]
    assert stages["rust.child.resolve"]["wall"] == pytest.approx(0.03)
    own = sum(
        row["wall"] for name, row in stages.items() if name not in timing.BORROWED
    )
    assert own == pytest.approx(stages["rust.fill"]["wall"])
    assert {"rust.child.resolve", "rust.child.encode"} <= timing.BORROWED


def test_threads_keep_their_own_stacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Four threads nesting at once, with no wall-clock burning anywhere.

    The first version had each thread busy-wait 40 x 2 ms. On its own that is 0.3 s and
    it looked free; run beside the rest of the suite under `-n auto` it added **84
    seconds** to a 76-second suite, reproducibly, while still reporting 0.30 s for
    itself. The mechanism was never established -- this repo has a second unexplained
    xdist anomaly on record (`-n 4` matching the serial run) -- so it is not written down
    as one. What is established is that a unit test has no business holding four threads
    in a wall-clock spin, and that removing it put the suite back to 76 s.

    The barrier is what makes the test mean anything: without it the threads can finish
    one after another and never overlap. With it they interleave, and a shared stack pops
    another thread's entry -- which shows up as a negative total, or as an IndexError on
    an empty pop. Neither needs the clock to have moved.
    """
    timing = load(monkeypatch, tmp_path)
    errors: list[BaseException] = []
    names = ("encode", "lp", "branch", "narrow")
    start = threading.Barrier(len(names))

    def run(inner: str) -> None:
        try:
            start.wait(timeout=10)
            for _ in range(200):
                with timing.stage("forward"), timing.stage(inner):
                    pass
        except BaseException as error:  # noqa: BLE001 - the point is that none escapes
            errors.append(error)

    threads = [threading.Thread(target=run, args=(name,)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert not any(thread.is_alive() for thread in threads)
    stages = timing.snapshot()["stages"]
    # A stack shared between threads pops someone else's entry, and the arithmetic then
    # subtracts a start time that belongs to another thread: the totals go negative.
    assert all(row["wall"] >= 0.0 for row in stages.values())
    assert all(row["cpu"] >= 0.0 for row in stages.values())
    assert stages["forward"]["calls"] == 200 * len(names)
    for name in names:
        assert stages[name]["calls"] == 200


def test_the_report_names_every_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stage missing because it never ran and one nobody instrumented must not match."""
    timing = load(monkeypatch, tmp_path)
    with timing.stage("lp"):
        burn(0.01)
    path = timing.write_report("a tag")
    assert path is not None
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["stages"]) >= set(timing.STAGES)
    assert report["stages"]["branch"]["calls"] == 0
    assert report["stages"]["lp"]["calls"] == 1
    assert report["tag"] == "a tag"
    assert report["process"]["wall"] > 0


# IKA-98: purposes, decision stretches, startup. Each is off by the same rule as a stage.


def test_off_the_ika98_additions_are_free(monkeypatch: pytest.MonkeyPatch) -> None:
    timing = load(monkeypatch, None)

    def work(x: int) -> int:
        return x + 1

    assert timing.labelled("matrix")(work) is work
    # The same shared no-op a stage hands back, not a purpose object per call.
    assert timing.purpose("rank") is timing.stage("lp")
    with timing.purpose("rank"):
        timing.count("leaves", 3)
        timing.decided("move")
        timing.refine("hidden")
        timing.ready()
    assert timing.clock() == 0.0
    report = timing.snapshot()
    assert report["stages"] == {}
    assert report["counts"] == {}
    assert report["decisions"] == {}
    assert report["startup_process_cpu"] is None


def test_a_purposed_count_goes_to_the_innermost_purpose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dirty cells of a hidden bench are filled from inside a node's matrix."""
    timing = load(monkeypatch, tmp_path)
    timing.count("leaves", 1)
    with timing.purpose("matrix"):
        timing.count("leaves", 10)
        with timing.purpose("dirty"):
            timing.count("leaves", 100)
            timing.count("lp.rows", 7)  # not in PURPOSED: kept once, with no split
        timing.count("leaves", 1000)
    with timing.purpose("not-a-purpose"):
        timing.count("fills")

    counts = timing.snapshot()["counts"]
    assert counts["leaves"] == 1111
    assert counts["leaves@other"] == 1
    assert counts["leaves@matrix"] == 1010
    assert counts["leaves@dirty"] == 100
    # The split adds up to the whole, because a name outside PURPOSES is `other`.
    assert sum(counts[f"leaves@{p}"] for p in timing.PURPOSES if f"leaves@{p}" in counts) == 1111
    assert counts["fills@other"] == 1
    assert "lp.rows@dirty" not in counts
    assert timing.current_purpose() == "other"


def test_labelled_names_a_whole_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timing = load(monkeypatch, tmp_path)

    @timing.labelled("rank")
    def rank() -> str:
        return timing.current_purpose()

    assert rank() == "rank"
    assert timing.current_purpose() == "other"


def test_purpose_rows_are_borrowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`rust.fill@rank` is `rust.fill` and its nestings again, so it is never in a total."""
    timing = load(monkeypatch, tmp_path)
    assert set(timing.PURPOSE_ROWS) <= timing.BORROWED
    assert {f"rust.fill@{p}" for p in timing.PURPOSES} <= set(timing.PURPOSE_ROWS)
    assert "startup" not in timing.BORROWED
    assert "belief" not in timing.BORROWED


def test_decision_stretches_are_charged_to_the_kind_that_closes_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    with timing.stage("narrow"):  # before the first game: not startup, and no decision's
        burn(0.03)
    timing.decided("between")  # the first game starts: startup ends, nothing is closed
    with timing.stage("lp"):
        pass
    timing.count("forward.passes", 4)
    timing.refine("hidden")
    timing.decided("move")
    with timing.stage("lp"):
        pass
    with timing.stage("lp"):
        pass
    timing.count("forward.passes", 1)
    timing.decided("move")
    timing.count("fills", 2)
    timing.decided("replacement")
    timing.decided("between")  # the next game
    timing.count("leaves", 5)  # after the last record: the open stretch, reported as between

    report = timing.snapshot()
    found = report["decisions"]
    assert set(found) == {"move.hidden", "move", "replacement", "between"}
    assert found["move.hidden"]["n"] == 1
    assert found["move.hidden"]["stages"]["lp"][1] == 1
    assert found["move.hidden"]["counts"]["forward.passes"] == 4
    assert found["move"]["stages"]["lp"][1] == 2
    assert found["move"]["counts"]["forward.passes"] == 1
    assert found["replacement"]["counts"]["fills"] == 2
    assert "lp" not in found["replacement"]["stages"]
    # One closed between (game 1 -> 2) and the open one after it.
    assert found["between"]["n"] == 2
    assert found["between"]["counts"]["leaves"] == 5
    # The narrowing before the first game is in no stretch.
    assert all("narrow" not in row["stages"] for row in found.values())

    startup = report["stages"]["startup"]
    assert startup["calls"] == 1
    # Its own time, less the stage that ran inside it: well under the 0.03 s burned there.
    assert startup["wall"] < 0.03
    assert report["startup_process_cpu"] is not None
    stretches = sum(row["wall"] for row in found.values())
    assert stretches + startup["wall"] + report["stages"]["narrow"]["wall"] == pytest.approx(
        report["elapsed"], abs=0.01
    )


def test_decided_twice_at_once_is_an_empty_stretch_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    timing.decided("between")
    timing.decided("move")
    timing.decided("move")
    found = timing.snapshot()["decisions"]
    assert found["move"]["n"] == 2
    assert found["move"]["counts"] == {}


def test_ready_ends_startup_once_and_opens_no_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    timing.ready()
    first = timing.snapshot()
    timing.ready()
    timing.decided("between")  # startup is already over: this opens the first stretch
    second = timing.snapshot()
    assert first["stages"]["startup"]["calls"] == 1
    assert first["decisions"] == {}
    assert second["startup_process_cpu"] == first["startup_process_cpu"]
    assert second["stages"]["startup"]["wall"] == first["stages"]["startup"]["wall"]


def test_the_report_says_which_checkout_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    assert Path(timing.snapshot()["source"]) == SOURCE.parent


def test_served_is_the_leaf_this_process_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """IKA-144: `served` follows the leaf, not an environment variable nobody sets.

    generate_queue.py and match_queue.py hand a worker the server as `--inference`, and
    the worker builds a `RemoteValue` from it; `POKEURAOU_INFERENCE` is never set. The
    report read the variable, so every served run said `served: false`. The real module is
    used here, because the flag is set by `inference.py` on the module it imported.
    """
    import socket

    from pokeuraou import timing as real
    from pokeuraou.inference import RemoteValue

    monkeypatch.setattr(real, "_SERVED", [False], raising=False)
    # The variable alone is not a served leaf: nothing was built from it.
    monkeypatch.setenv("POKEURAOU_INFERENCE", "127.0.0.1:1")
    assert real.snapshot()["served"] is False
    monkeypatch.delenv("POKEURAOU_INFERENCE")

    # A listening socket is all `RemoteValue` needs to be built; nothing is asked of it.
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
        with RemoteValue(f"127.0.0.1:{port}", "value", encoder=None, buffer_bytes=1 << 12):
            pass
    assert real.snapshot()["served"] is True


# -- IKA-258: counting repeated inputs, and sampling where a worker's main thread is.


def test_repeats_are_counted_within_a_stretch_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A key seen twice in one decision is one repeat; the next decision starts empty."""
    timing = load(monkeypatch, tmp_path, dupes=True)
    assert timing.DUPES is True
    timing.decided("between")
    assert timing.repeat("port.score", b"a") is False
    assert timing.repeat("port.score", b"b") is False
    assert timing.repeat("port.score", b"a") is True
    # Another kind with the same key is another question.
    assert timing.repeat("port.fill", b"a") is False
    timing.decided("move")
    assert timing.repeat("port.score", b"a") is False
    counts = timing.snapshot()["counts"]
    assert counts["dup.port.score.calls"] == 4
    assert counts["dup.port.score.repeat"] == 1
    assert counts["dup.port.fill.calls"] == 1
    assert "dup.port.fill.repeat" not in counts
    # And per decision, as every other count.
    move = timing.snapshot()["decisions"]["move"]["counts"]
    assert move["dup.port.score.repeat"] == 1


def test_repeats_need_their_own_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The null control: timing on, the switch off -- nothing is hashed or counted."""
    timing = load(monkeypatch, tmp_path)
    assert timing.DUPES is False
    assert timing.repeat("port.score", b"a") is False
    assert timing.repeat("port.score", b"a") is False
    assert timing.snapshot()["counts"] == {}
    off = load(monkeypatch, None, dupes=True)
    assert off.DUPES is False
    assert off.repeat("port.score", b"a") is False
    assert off.snapshot()["counts"] == {}


def _spin_here(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_the_sampler_finds_the_function_that_is_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: a main thread busy in one function is sampled there."""
    timing = load(monkeypatch, tmp_path, sample_hz=500)
    try:
        assert timing.SAMPLE_HZ == 500
        _spin_here(0.4)
    finally:
        timing._SAMPLER_STOP.set()
    samples = timing.snapshot()["samples"]
    assert samples["ticks"] > 20
    here = samples["self"].get("test_timing.py:_spin_here", 0)
    assert here > 0.5 * samples["ticks"]
    assert samples["inclusive"]["test_timing.py:_spin_here"] >= here


def _hash_here(seconds: float) -> None:
    """Busy outside the GIL, so a second busy thread does not take turns with it."""
    import hashlib

    block = bytes(1 << 20)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        hashlib.sha256(block).digest()


def test_the_cpu_sampler_charges_the_thread_that_burns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IKA-339: every thread's CPU seconds, by thread. A thread hashing is charged most of
    its 0.6 s; a thread asleep the whole time next to nothing (the control: a wall-clock
    sampler would give it as many ticks as the busy one)."""
    timing = load(monkeypatch, tmp_path, sample_hz=500, sample_cpu=True)
    try:
        assert timing.SAMPLE_CPU
        # Before this, every tick is charged to "(startup)".
        timing.ready()
        busy = threading.Thread(target=_hash_here, args=(0.6,), name="busy-7")
        idle = threading.Thread(target=time.sleep, args=(0.6,), name="idle-7")
        busy.start()
        idle.start()
        busy.join()
        idle.join()
    finally:
        timing._SAMPLER_STOP.set()
    threads = timing.snapshot()["cpu_samples"]["thread"]
    assert threads.get("busy-N", 0) > 0.2
    assert threads.get("idle-N", 0) < 0.05 * threads["busy-N"]
    assert not any(name.startswith("(startup)") for name in threads)
    assert timing.snapshot()["samples"]["ticks"] > 20


def test_no_cpu_samples_unless_asked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timing = load(monkeypatch, tmp_path, sample_hz=500)
    try:
        assert not timing.SAMPLE_CPU
        _spin_here(0.05)
    finally:
        timing._SAMPLER_STOP.set()
    assert timing.snapshot()["cpu_samples"] is None


def test_no_sampler_without_the_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The null control: no rate, no thread and no samples in the report; timing off too."""
    before = {t.name for t in threading.enumerate()}
    timing = load(monkeypatch, tmp_path)
    assert timing.SAMPLE_HZ == 0
    assert timing.snapshot()["samples"] is None
    off = load(monkeypatch, None, sample_hz=500)
    assert off.SAMPLE_HZ == 0
    after = [t.name for t in threading.enumerate() if t.name not in before]
    assert "pokeuraou-sampler" not in after


def test_the_worker_stages_are_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IKA-258's rows are in STAGES, so a report has them even when they never ran."""
    timing = load(monkeypatch, tmp_path)
    assert set(timing.WORKER_STAGES) <= set(timing.STAGES)
    assert not set(timing.WORKER_STAGES) & timing.BORROWED
    assert set(timing.WORKER_STAGES) <= set(timing.snapshot()["stages"])


def test_regions_off_are_the_shared_no_op(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IKA-32: without their own switch, parts are the no-op and nothing is kept -- with
    timing on as well as off."""
    for timing in (load(monkeypatch, tmp_path), load(monkeypatch, None, regions=True)):
        assert timing.REGIONS is False
        assert timing.region("d2.fills") is timing.region("lp.side")
        with timing.region("d2.fills"):
            pass
        timing.child_process(12345)
        assert timing._CHILDREN == []
        assert timing.snapshot()["regions"] is None


def test_a_part_splits_the_stage_running_across_its_edge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IKA-32: a stage running when a part opens is cut at the edge -- its total is the same,
    and each side of the edge is charged to its own part."""
    timing = load(monkeypatch, tmp_path, regions=True)
    timing.decided("between")
    with timing.stage("belief"):
        burn(0.05)
        with timing.region("belief.fold"):
            burn(0.10)
            with timing.region("d2.fold"):
                burn(0.05)
            burn(0.02)
        burn(0.03)
    timing.decided("move")
    report = timing.snapshot()
    parts = report["decisions"]["move"]["regions"]
    total = report["stages"]["belief"]["wall"]
    outside = parts["-|belief"][0]
    fold = parts["belief.fold|belief"][0]
    inner = parts["d2.fold|belief"][0]
    assert outside + fold + inner == pytest.approx(total, abs=1e-6)
    assert outside == pytest.approx(0.08, abs=0.02)
    assert fold == pytest.approx(0.12, abs=0.02)
    assert inner == pytest.approx(0.05, abs=0.02)
    # A part's own time excludes the part nested in it.
    assert parts["belief.fold"][0] == pytest.approx(0.12, abs=0.02)
    assert parts["belief.fold"][2] == 1
    assert parts["d2.fold"][0] == pytest.approx(0.05, abs=0.02)
    # The parts, "-" included, hold the whole stretch.
    own = sum(v[0] for k, v in parts.items() if "|" not in k)
    assert own == pytest.approx(report["decisions"]["move"]["wall"], abs=0.005)


def test_a_part_is_charged_its_child_s_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """IKA-32: the Rust child's CPU, read at the edges, goes to the part it was spent in.
    Positive control: a child that burns only while the part is open."""
    import subprocess
    import sys

    timing = load(monkeypatch, tmp_path, regions=True)
    # The base interpreter: a venv's python.exe on Windows is a launcher that runs the real
    # one as its own child, and the launcher's CPU is next to nothing.
    child = subprocess.Popen(
        [getattr(sys, "_base_executable", sys.executable), "-c",
         "import sys, time\n"
         "sys.stdin.readline()\n"
         "end = time.perf_counter() + 0.4\n"
         "while time.perf_counter() < end: pass\n"
         "print('done', flush=True)\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        timing.child_process(child.pid)
        timing.decided("between")
        with timing.region("belief.jobs"):
            child.stdin.write("go\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "done"
        burn(0.05)
        timing.decided("move")
    finally:
        child.wait(timeout=10)
    parts = timing.snapshot()["decisions"]["move"]["regions"]
    assert parts["belief.jobs"][3] == pytest.approx(0.4, abs=0.1)
    assert parts.get("-", [0.0, 0.0, 0, 0.0])[3] < 0.05


def test_a_part_left_open_by_a_raise_is_closed_at_the_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path, regions=True)
    timing.decided("between")
    part = timing.region("d2.turns")
    part.__enter__()  # as a stage whose body raises before the exit
    burn(0.02)
    timing.decided("move")
    assert timing._REGION_STACK == []
    found = timing.snapshot()["decisions"]["move"]
    assert found["counts"]["region.unclosed"] == 1
    assert found["regions"]["d2.turns"][0] == pytest.approx(0.02, abs=0.01)


# --- IKA-454: the stages' CPU is a thread's cycle count on Windows ---------------------

WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="the thread cycle clock exists on Windows only"
)


def _echo_pipes() -> tuple[Any, tuple[int, int, int, int]]:
    """A thread that answers every byte on one pipe with a byte on another, and the fds."""
    ask_r, ask_w = os.pipe()
    answer_r, answer_w = os.pipe()

    def echo() -> None:
        while os.read(ask_r, 1) == b"x":
            os.write(answer_w, b"y")

    thread = threading.Thread(target=echo)
    thread.start()
    return thread, (ask_r, ask_w, answer_r, answer_w)


def _finish(thread: Any, fds: tuple[int, int, int, int]) -> None:
    os.write(fds[1], b"q")
    thread.join()
    for fd in fds:
        os.close(fd)


def _wait_calls(clock: Any, calls: int) -> tuple[float, float, int]:
    """Sum of a clock's and the wall clock's differences over short waits on another thread,
    and how many single calls had more CPU than wall by over a millisecond.

    Each call writes a byte down one pipe and blocks on the answer from a thread on the other
    end, which is what a worker does while the port answers. The call takes tens of
    microseconds of wall time, so a clock that moves in 15.6 ms steps puts a whole step
    (a thousand times the call) on whichever call it lands on.
    """
    thread, fds = _echo_pipes()
    cpu = wall = 0.0
    over = 0
    for _ in range(calls):
        c0, w0 = clock(), time.perf_counter()
        os.write(fds[1], b"x")
        os.read(fds[2], 1)
        spent_wall = time.perf_counter() - w0
        spent_cpu = clock() - c0
        wall += spent_wall
        cpu += spent_cpu
        over += spent_cpu > spent_wall + 1e-3
    _finish(thread, fds)
    return cpu, wall, over


@WINDOWS_ONLY
def test_thread_time_overcounts_short_waits_and_the_cycle_clock_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The positive control (IKA-452, IKA-454): the old clock really does exceed the wall.

    The same pipe round trips are timed on `time.thread_time` and on the module's cycle
    clock. One thread cannot spend more CPU in a call than the call took, so a call whose CPU
    is over its wall by a millisecond is the clock (a 15.6 ms step landing on a 13 us call).
    The old clock must show such calls, or the new clock showing none proves nothing. (Summed
    over a quiet machine the old clock's total can still be below the wall: it is an unbiased
    estimator there. IKA-452 saw the sum above the wall on the busy worker, per row.)
    """
    timing = load(monkeypatch, tmp_path)
    assert timing.CPU_CLOCK == "thread_cycles"
    calls = 20_000
    _, _, old_over = _wait_calls(time.thread_time, calls)
    new_cpu, new_wall, new_over = _wait_calls(timing.thread_cpu, calls)
    assert old_over > 0, "the old clock no longer steps on this machine; the control is void"
    assert new_over == 0
    assert new_cpu <= new_wall * (1.0 + timing.CPU_OVER_WALL_TOLERANCE), (new_cpu, new_wall)


@WINDOWS_ONLY
def test_a_stage_around_a_wait_holds_no_more_cpu_than_wall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Through the module's own stage, not the bare clock."""
    timing = load(monkeypatch, tmp_path)
    thread, fds = _echo_pipes()
    for _ in range(5_000):
        with timing.stage("rust.score"):
            os.write(fds[1], b"x")
            os.read(fds[2], 1)
    _finish(thread, fds)
    row = timing.snapshot()["stages"]["rust.score"]
    assert row["calls"] == 5_000
    assert row["cpu"] <= row["wall"] * (1.0 + timing.CPU_OVER_WALL_TOLERANCE), row


@WINDOWS_ONLY
def test_the_cycle_clock_reads_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    timing = load(monkeypatch, tmp_path)
    seen: dict[str, float] = {}

    def spinner() -> None:
        c0 = timing.thread_cpu()
        burn(0.2)
        seen["spinner"] = timing.thread_cpu() - c0

    thread = threading.Thread(target=spinner)
    c0 = timing.thread_cpu()
    thread.start()
    thread.join()
    waiting = timing.thread_cpu() - c0
    assert seen["spinner"] == pytest.approx(0.2, rel=0.1)
    assert waiting < 0.02


@WINDOWS_ONLY
def test_the_cycle_rate_agrees_with_a_long_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 1 s spin is read on the cycle clock: its CPU is its wall, to the calibration's error."""
    timing = load(monkeypatch, tmp_path)
    c0, w0 = timing.thread_cpu(), time.perf_counter()
    burn(1.0)
    cpu, wall = timing.thread_cpu() - c0, time.perf_counter() - w0
    assert cpu / wall == pytest.approx(1.0, abs=timing.CPU_OVER_WALL_TOLERANCE)


def test_the_clock_is_thread_time_where_timing_is_off_or_not_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = load(monkeypatch, None)
    assert off.thread_cpu is time.thread_time
    assert off.CPU_CLOCK == "thread_time"
    assert off.CYCLES_PER_SECOND == 0.0
    if sys.platform != "win32":
        on = load(monkeypatch, tmp_path)
        assert on.thread_cpu is time.thread_time
        assert on.snapshot()["cpu_clock"] == "thread_time"


def test_no_module_code_calls_thread_time_directly() -> None:
    """The clock is one function; a second call site would put two scales in one table."""
    calls = [
        line.strip() for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if "thread_time(" in line and not line.lstrip().startswith("#")
    ]
    assert calls == [], calls


def test_calibration_takes_the_least_interrupted_spin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A counter at a known 3 GHz that loses 60% of its cycles on four spins in five, as a
    descheduled thread does. The fastest spin is the true rate; one interrupted spin is not."""
    timing = load(monkeypatch, None)
    rate = 3.0e9
    state = {"lost": 0.0, "reads": 0}

    def counter() -> int:
        state["reads"] += 1
        if state["reads"] % 2 == 0 and (state["reads"] // 2) % 5 != 0:
            state["lost"] += 0.6 * 0.005 * rate
        return int(time.perf_counter() * rate - state["lost"])

    got = timing.calibrate_cycles_per_second(counter, spans=10, span=0.005)
    assert got == pytest.approx(rate, rel=0.02)
    # The control: the first spin of the same counter is interrupted and reads low.
    state.update(lost=0.0, reads=0)
    c0, w0 = counter(), time.perf_counter()
    burn(0.005)
    interrupted = (counter() - c0) / (time.perf_counter() - w0)
    assert interrupted < rate * 0.6


def test_calibration_fails_on_a_slow_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check can fail: a counter at half the rate reads half the rate."""
    timing = load(monkeypatch, None)
    got = timing.calibrate_cycles_per_second(
        lambda: int(time.perf_counter() * 1.5e9), spans=5, span=0.005
    )
    assert got != pytest.approx(3.0e9, rel=0.02)


def test_profile_stages_warns_on_a_row_whose_cpu_exceeds_its_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "_profile_stages_for_clock", SOURCE.parents[2] / "tools" / "profile_stages.py"
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    # IKA-452's `rust.score` on thread_time: 0.603 ms CPU against 0.209 ms wall a call.
    bad = {"stages": {"rust.score": {"wall": 20.9, "cpu": 60.3, "calls": 100_000.0}}}
    assert len(tool.cpu_clock_warnings(bad)) == 1
    assert "rust.score" in tool.cpu_clock_warnings(bad)[0]
    # And with the cycle clock's 0.100 ms, nothing; so does a LP row 3% over (inside 5%).
    good = {"stages": {
        "rust.score": {"wall": 20.9, "cpu": 10.0, "calls": 100_000.0},
        "lp": {"wall": 100.0, "cpu": 103.0, "calls": 100_000.0},
    }}
    assert tool.cpu_clock_warnings(good) == []
    # IKA-452's mis-calibrated LP: 1.938 ms against 1.745 ms is 11% over.
    lp = {"stages": {"lp": {"wall": 174.5, "cpu": 193.8, "calls": 100_000.0}}}
    assert len(tool.cpu_clock_warnings(lp)) == 1
    # A borrowed row carries no CPU of its own and is never judged.
    borrowed = {"stages": {"server.held": {"wall": 1.0, "cpu": 9.0, "calls": 10.0}}}
    assert tool.cpu_clock_warnings(borrowed) == []
