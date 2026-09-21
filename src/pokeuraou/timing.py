"""Where a run's time goes, by stage, measured in the processes that actually run it.

The alternative was `cProfile`, and it answers a different question. A profile attributes
time to *functions*, and the stages here do not line up with functions: the branch
generation of a node is `resolve_turn` when the bridge is off and a Rust child when it is
on, and the forward pass is in this process, or in a server, or nowhere visible at all.
A profiler also stops at the process boundary, which is where two of the three things
being measured live.

So this is a handful of named stages, accumulated where the work is, and written out by
every process that took part::

    POKEURAOU_TIMING=/some/dir uv run python tools/generate_queue.py ...

Off unless that variable is set, and off is the default everywhere. A stage that is not
enabled costs a module-global check and a shared no-op object, which is why the timers can
sit on `resolve_turn` -- a hot path at 576 calls a node -- rather than only on the outside
of a run.

**Time is charged to one stage at a time.** A stage entered inside another suspends the
outer one and resumes it on the way out, so what each row holds is its own time and not
its nestings: narrowing that ranks by the leaf calls the encoder and the net, and a
narrowing row that quietly contained them would make encode and forward appear twice in a
table whose whole purpose is to add up. What is left over -- elapsed minus the rows -- is
the simulator bridge, the I/O and the interpreter, and it is reported rather than assumed
to be zero.

Two clocks, and they are not the same measurement:

    wall    `perf_counter`, exact. Includes time this process spent waiting for another
    cpu     `thread_time`, and **it advances in 15.625 ms steps on Windows** whatever
            `get_clock_info` claims (measured: 19 distinct values in a 0.3 s busy loop).
            Per call it is noise; over the thousands of calls a run makes it is a
            randomised-rounding estimator of the total, and it is read for one thing only
            -- telling a stage that is burning CPU from a stage that is blocked on someone
            else, where that difference is the whole answer

The process totals come from `psutil` instead, which reads the kernel's own cumulative
counters and does not have that problem. Children are counted separately, because the Rust
node is a child and its CPU is not this process's.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Directory each process drops its report into. Unset means every timer here is a no-op.
ENV_DIR = "POKEURAOU_TIMING"

#: The stages, in the order a report prints them. Named here rather than created on first
#: use so that a report always has every row -- a stage missing because it never ran and a
#: stage missing because nobody instrumented it look identical otherwise, and the second
#: one is the mistake this module exists to avoid.
STAGES = (
    "narrow",        # building the candidate menus, minus any leaf call inside them
    "branch",        # resolve_turn, in Python
    "encode",        # Encoder.encode_positions / Encoder.encode, in Python
    "forward",       # the net, in this process
    "lp",            # solve / solve_bayesian
    "rust.fill",     # a node asked of the Rust child, and the wait for its header
    "rust.ask",      # building the request's JSON, here
    "rust.header",   # reading the answer's JSON, here
    "rust.body",     # getting the encoded leaves: off the pipe, or out of the shared block
    "rust.unpack",   # turning the header and blob into arrays, spans and folds
    "rust.resolve",  # advancing a real turn through the child
    "rust.score",    # the damage ranking, through the child
    "rust.child.resolve",  # what the child says it spent resolving  (its own clock)
    "rust.child.encode",   # what the child says it spent encoding   (its own clock)
    "rust.child.parse",    # ... reading the request's JSON          (its own clock)
    "rust.child.header",   # ... building and serialising the header (its own clock)
    "serve.copy",    # laying an encoded batch into the shared block
    "serve.wait",    # from sending the control line to having the answer
    "server.held",   # the serving side, working  (its own counter)
    "server.queue",  # the serving side, queueing (its own counter)
    "refused",       # filling the cells the port declined, in one call a node
)

#: Rows that repeat time another row already holds, and so are never added into a total.
#: `rust.child.*` is the child's own account of a span this process spent inside
#: `rust.fill`, and `server.*` is the serving thread's own account of what it spent inside
#: `forward`. Both are worth printing -- they say what the wait was *for* -- and adding
#: either to the rows beside it would count the same seconds twice.
BORROWED = frozenset(
    {"rust.child.resolve", "rust.child.encode", "rust.child.parse", "rust.child.header",
     "server.held", "server.queue", "refused"}
)


class _Off:
    """What `stage()` hands back when timing is off: enter, exit, nothing else."""

    __slots__ = ()

    def __enter__(self) -> _Off:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


_OFF = _Off()

#: Per thread: [stage, wall it last resumed at, cpu it last resumed at], innermost last.
#:
#: Thread-local because the inference server answers each connection on its own thread and
#: `thread_time` is per thread. One shared stack would interleave two threads' nestings and
#: pop the wrong entry, which is not a slightly wrong number -- it is a negative one.
_LOCAL = threading.local()


def _stack() -> list[list[Any]]:
    found = getattr(_LOCAL, "stack", None)
    if found is None:
        found = _LOCAL.stack = []
    return found


class _Stage:
    """One named stage's running totals, and its own context manager.

    No per-entry state lives here: the clock a nesting started on is on the stack, so one
    shared object can be entered from inside itself without losing the outer entry.
    """

    __slots__ = ("calls", "cpu", "name", "wall")

    def __init__(self, name: str) -> None:
        self.name = name
        self.wall = 0.0
        self.cpu = 0.0
        self.calls = 0

    def __enter__(self) -> _Stage:
        wall = time.perf_counter()
        cpu = time.thread_time()
        stack = _stack()
        if stack:
            outer = stack[-1]
            outer[0].wall += wall - outer[1]
            outer[0].cpu += cpu - outer[2]
        stack.append([self, wall, cpu])
        self.calls += 1
        return self

    def __exit__(self, *_exc: object) -> bool:
        wall = time.perf_counter()
        cpu = time.thread_time()
        stack = _stack()
        mine = stack.pop()
        mine[0].wall += wall - mine[1]
        mine[0].cpu += cpu - mine[2]
        if stack:
            outer = stack[-1]
            outer[1] = wall
            outer[2] = cpu
        return False


ON = bool(os.environ.get(ENV_DIR))

_STAGES: dict[str, _Stage] = {name: _Stage(name) for name in STAGES} if ON else {}
_COUNTS: dict[str, int] = {}
_STARTED = time.perf_counter()


def stage(name: str) -> Any:
    """A context manager that charges its body to `name`, or a no-op when timing is off."""
    if not ON:
        return _OFF
    found = _STAGES.get(name)
    if found is None:
        found = _STAGES[name] = _Stage(name)
    return found


def timed(name: str) -> Callable[[Callable], Callable]:
    """Charge a whole function to `name`.

    When timing is off this hands the function straight back, so a decorated hot path --
    `resolve_turn` is 576 calls a node -- costs literally nothing in a normal run rather
    than a check per call.

    Recursion is safe: an inner entry charges to the same stage, and the stack only
    accumulates once per entry, so a row is the time spent inside the stage and never the
    sum of its nestings.
    """

    def wrap(function: Callable) -> Callable:
        if not ON:
            return function
        here = stage(name)

        @functools.wraps(function)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with here:
                return function(*args, **kwargs)

        return inner

    return wrap


def add(name: str, seconds: float, *, calls: int = 1) -> None:
    """Charge `seconds` to a stage that was timed somewhere this process cannot stand.

    The Rust child reports its own resolve and encode microseconds in the node header, and
    that is the only honest source for them: from here the two are one wait on a pipe.
    These rows are listed in `BORROWED` and are never added into this process's total.
    """
    if not ON or seconds <= 0:
        return
    found = _STAGES.get(name)
    if found is None:
        found = _STAGES[name] = _Stage(name)
    found.wall += seconds
    found.calls += calls


def count(name: str, n: int = 1) -> None:
    """Tally something that is not a duration: refusals by reason, rows per call.

    Kept apart from the stages so that nothing here can end up in a column of
    seconds. A share of the clock and a count of events answer different
    questions, and the table prints them under different headings.
    """
    if not ON:
        return
    _COUNTS[name] = _COUNTS.get(name, 0) + n


def set_total(name: str, seconds: float, *, calls: int = 0) -> None:
    """Set a borrowed row to a running total, rather than adding to it.

    For a counter somebody else keeps cumulatively. The inference server is killed with
    `TerminateProcess` when its launcher is done -- on Windows that is not a signal and
    no `atexit` handler runs -- so it writes its report over and over while it lives, and
    a row it re-reports must replace the last one rather than pile on top of it.
    """
    if not ON:
        return
    found = _STAGES.get(name)
    if found is None:
        found = _STAGES[name] = _Stage(name)
    found.wall = seconds
    found.calls = calls


def snapshot() -> dict[str, Any]:
    """The report as it stands, for a caller that runs the workload in its own process."""
    return {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "elapsed": time.perf_counter() - _STARTED,
        "rust_node": os.environ.get("POKEURAOU_RUST_NODE", ""),
        "served": bool(os.environ.get("POKEURAOU_INFERENCE")),
        "stages": {
            name: {"wall": s.wall, "cpu": s.cpu, "calls": s.calls}
            for name, s in _STAGES.items()
        },
        "counts": dict(_COUNTS),
        "process": _process_times(),
    }


def _process_times() -> dict[str, Any]:
    """This process's CPU seconds, and its children's, from the kernel's own counters.

    `thread_time` is quantised at a scheduler tick and the per-stage numbers wear that;
    these do not, so the shares can be checked against a total that is not made of them.
    Children are separate because the Rust node is a child: counting its CPU as this
    process's is exactly the confusion that made "Rust 84% / Python 4%" read as a CPU
    split when it was a wall-clock one.
    """
    out: dict[str, Any] = {
        "process_time": time.process_time(),
        "wall": time.perf_counter() - _STARTED,
    }
    try:
        import psutil
    except ImportError:
        return out
    try:
        me = psutil.Process()
        mine = me.cpu_times()
        out["cpu_user"] = float(mine.user)
        out["cpu_system"] = float(mine.system)
        child_user = child_system = 0.0
        names: dict[str, float] = {}
        for child in me.children(recursive=True):
            try:
                times = child.cpu_times()
                child_user += float(times.user)
                child_system += float(times.system)
                label = child.name()
                names[label] = names.get(label, 0.0) + float(times.user + times.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        out["children_user"] = child_user
        out["children_system"] = child_system
        out["children_by_name"] = names
        out["rss"] = int(me.memory_info().rss)
    except Exception as error:  # noqa: BLE001 - a report is never worth failing a run for
        out["psutil_error"] = f"{type(error).__name__}: {error}"
    return out


def write_report(tag: str = "") -> Path | None:
    """Drop this process's report into the directory the environment named.

    Registered at import when timing is on, and registered *early* on purpose: `atexit`
    runs handlers in reverse, so the first one registered is the last one to run, and a
    child read after it has been reaped reports nothing.
    """
    directory = os.environ.get(ENV_DIR)
    if not directory:
        return None
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    report = snapshot()
    report["tag"] = tag
    path = target / f"timing-{os.getpid()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return path


if ON:
    atexit.register(write_report)


__all__ = [
    "BORROWED",
    "ENV_DIR",
    "ON",
    "STAGES",
    "add",
    "count",
    "set_total",
    "snapshot",
    "stage",
    "timed",
    "write_report",
]
