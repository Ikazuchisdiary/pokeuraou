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
import threading
import time
from pathlib import Path
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "pokeuraou" / "timing.py"


def load(monkeypatch: pytest.MonkeyPatch, directory: Path | None) -> Any:
    """A private copy of the module, with the environment it reads at import time."""
    if directory is None:
        monkeypatch.delenv("POKEURAOU_TIMING", raising=False)
    else:
        monkeypatch.setenv("POKEURAOU_TIMING", str(directory))
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
