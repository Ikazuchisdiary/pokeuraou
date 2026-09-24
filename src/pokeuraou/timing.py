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

Three things were added for IKA-98, each because a table had been read for something it
could not say, and each off by the same rule as the rest -- a decorator hands the function
back, a context manager is the shared no-op, a call is a module-global check:

    purpose     what a stage's work was *for*, when one stage serves several callers.
                `rust.fill` is one row for the leaf ranking, a node's matrix and a hidden
                bench's dirty cells, and those are three different fixes (IKA-108, -105)
    decision    which decision the work belongs to, so a count can be read per decision
                (forward passes, completions, fills, LPs). Counts do not move with how
                busy the machine is, which is why they are the part of a shared-machine
                run that can be quoted
    startup     the wall clock from this module's import to the first game, as a row
                of its own. It was inside "the rest", and IKA-97 named 28.8% of a 40-game
                run "startup" without having measured it; it was, and it would not have
                been in a two-hour run, and the table could not tell the two apart
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

#: What a fill of the Rust node can be for. `other` is everything no caller named, so the
#: purposes always add up to the whole: the replacement and self-switch nodes, the analyser.
PURPOSES = ("rank", "matrix", "dirty", "other")

#: IKA-258's rows: where a pool worker's clock went that no earlier row named.
WORKER_STAGES = (
    "selection",        # a game's four each side and bench priors: the solve read, the draw
    "selection.solve",  # solving a pair's selection game, minus the encode/forward inside it
    "record",           # a finished game's JSON and its line in the file
    "completions",      # the hidden bench's completions at a node (hidden.completions)
    "position.json",    # Position.to_json, wherever it is called (a request, a record)
    "position.parse",   # Position.from_json, wherever it is called (an answer)
    "rust.turn",        # a turn through the port (port.turn), the wait included
    "rust.needed",      # the replacement check before every decision, the wait included
    "rust.leads",       # the leads' switch-ins (a game's start, the selection solve)
    "rust.replacements",  # the replacement phase
    "rust.resume",      # a paused turn resumed
    "rust.alternatives",  # a pause's alternatives, encoded or not
)

#: The stages, in the order a report prints them. Named here rather than created on first
#: use so that a report always has every row -- a stage missing because it never ran and a
#: stage missing because nobody instrumented it look identical otherwise, and the second
#: one is the mistake this module exists to avoid.
STAGES = (
    "startup",       # this module's import to the first game (see `decided`)
    "narrow",        # building the candidate menus, minus any leaf call inside them
    "branch",        # resolve_turn, in Python
    "encode",        # Encoder.encode_positions / Encoder.encode, in Python
    "forward",       # the net, in this process
    "lp",            # solve / solve_bayesian
    "belief",        # belief_payoffs' own Python: the per-completion copies, spans, folds
    # The self-switch node's own Python (IKA-150), minus the encode/forward inside it:
    "selfswitch.resume",    # resume_alternatives: every option's rest of the turn, per world
    "selfswitch.complete",  # the opponent's completions, and the pause rebuilt in each
    "selfswitch.leaves",    # turn_leaves (a nested pause's options included), fast-path copies
    "selfswitch.fold",      # folding the leaf values into scores, the argmax, the record
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
    # What a worker does outside the search, and the Python around every crossing that no
    # row above held (IKA-258): each of these was in "the rest" or inside `rust.fill`.
    *WORKER_STAGES,
    # One fill's whole call here (header, body, unpack and the wait), by what it was for...
    *(f"rust.fill@{name}" for name in PURPOSES),
    # ...and what the child says that fill cost it, all four of its clocks together.
    *(f"rust.child@{name}" for name in PURPOSES),
)

#: The purpose rows, which split `rust.fill` and the `rust.child.*` rows by caller.
PURPOSE_ROWS = tuple(name for name in STAGES if "@" in name)

#: Rows that repeat time another row already holds, and so are never added into a total.
#: `rust.child.*` is the child's own account of a span this process spent inside
#: `rust.fill`, and `server.*` is the serving thread's own account of what it spent inside
#: `forward`. Both are worth printing -- they say what the wait was *for* -- and adding
#: either to the rows beside it would count the same seconds twice. The purpose rows are
#: `rust.fill` and its nestings cut the other way, so they are borrowed for the same reason.
BORROWED = frozenset(
    {"rust.child.resolve", "rust.child.encode", "rust.child.parse", "rust.child.header",
     "server.held", "server.queue", "refused", *PURPOSE_ROWS}
)

#: Counts that are also kept per purpose, as `name@purpose`: the rows the net scored and
#: the passes it took to score them, and the fills of the Rust node with their size.
PURPOSED = frozenset({"leaves", "forward.passes", "fills", "fill.cells", "fill.leaves"})


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


class _Purpose:
    """One named purpose, as a context manager. Per-entry state lives on the thread's stack."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> _Purpose:
        _purposes().append(self.name)
        return self

    def __exit__(self, *_exc: object) -> bool:
        _purposes().pop()
        return False


def _purposes() -> list[str]:
    found = getattr(_LOCAL, "purposes", None)
    if found is None:
        found = _LOCAL.purposes = []
    return found


ON = bool(os.environ.get(ENV_DIR))

_STAGES: dict[str, _Stage] = {name: _Stage(name) for name in STAGES} if ON else {}
_COUNTS: dict[str, int] = {}
_PURPOSE_OBJECTS: dict[str, _Purpose] = {}
_STARTED = time.perf_counter()
_STARTED_CPU = time.thread_time()

#: The decision now being charged: [kind, details, wall it opened at, the stage totals and
#: the counts as they stood then]. None before the first one and between none at all.
_OPEN: list[Any] = [None]
#: Closed decisions, summed by kind: {"n", "wall", "stages": {name: [wall, calls]}, "counts"}.
_DECISIONS: dict[str, dict[str, Any]] = {}
#: The process's CPU seconds (every thread, from the kernel) when startup ended.
_STARTUP_CPU: list[float] = []
#: Whether this process's leaf is a server's: set by `RemoteValue` when one is built.
_SERVED: list[bool] = [False]


#: IKA-258: count calls that repeat an input already seen in the same decision stretch.
#: Its own switch on top of `POKEURAOU_TIMING`, because hashing every request and every
#: leaf row is not free: a run that counts is a run whose clock is not read.
ENV_DUPES = "POKEURAOU_TIMING_DUPES"
DUPES = ON and bool(os.environ.get(ENV_DUPES))
#: kind -> the keys seen since the stretch opened. Cleared by `decided`.
_SEEN: dict[str, set[Any]] = {}


def repeat(kind: str, key: Any, n: int = 1) -> bool:
    """Count `n` calls of `kind` on input `key`; True when this stretch saw `key` already.

    Counted as `dup.<kind>.calls` and `dup.<kind>.repeat`, so they land in the per-decision
    table beside everything else. A stage timer cannot see a call made twice on the same
    input -- both calls look like work -- and that is the waste a cache would remove.
    """
    if not DUPES:
        return False
    seen = _SEEN.setdefault(kind, set())
    count(f"dup.{kind}.calls", n)
    if key in seen:
        count(f"dup.{kind}.repeat", n)
        return True
    seen.add(key)
    return False


#: IKA-264: which caller sent a key first in this stretch, so a repeat names both ends.
_FIRST: dict[str, dict[Any, str]] = {}
#: The exchange's own modules: a caller is the first frame outside them.
_PLUMBING = frozenset(
    {"timing.py", "rustnode.py", "port.py", "position.py", "inference.py", "functools.py"}
)

#: IKA-264: called at every decision boundary, timing on or off -- what a cache that
#: lives one decision forgets on.
_ON_DECIDED: list[Callable[[], None]] = []


def on_decided(hook: Callable[[], None]) -> None:
    """Run `hook` at every `decided` call from now on (once, however often asked)."""
    if hook not in _ON_DECIDED:
        _ON_DECIDED.append(hook)


def caller(depth: int = 3) -> str:
    """`module.function` of the first `depth` frames outside the exchange's own modules,
    innermost first, joined by `<`. For the repeat counts only: walking the stack is not free.
    """
    frame = sys._getframe(1)
    names: list[str] = []
    while frame is not None and len(names) < depth:
        name = os.path.basename(frame.f_code.co_filename)
        if name not in _PLUMBING:
            names.append(f"{name.removesuffix('.py')}.{frame.f_code.co_name}")
        frame = frame.f_back
    return "<".join(names) or "?"


def repeat_where(kind: str, key: Any, n: int = 1, where: str | None = None) -> bool:
    """`repeat`, and the same counts by caller (IKA-264): `dup.<kind>.calls@<caller>` and
    `dup.<kind>.repeat@<first caller> >> <this caller>`. `where` saves the stack walk
    when one call counts many keys."""
    if not DUPES:
        return False
    if where is None:
        where = caller()
    again = repeat(kind, key, n)
    first = _FIRST.setdefault(kind, {})
    count(f"dup.{kind}.calls@{where}", n)
    if again:
        count(f"dup.{kind}.repeat@{first.get(key, '?')} >> {where}", n)
    else:
        first[key] = where
    return again


#: IKA-258: sample the main thread's stack this many times a second, into the report.
#: Zero (the default) starts no thread. Only with timing on.
ENV_SAMPLE = "POKEURAOU_SAMPLE_HZ"
_SAMPLES: dict[str, dict[str, int]] = {"self": {}, "own": {}, "inclusive": {}}
_SAMPLE_TICKS = [0]
#: Set to stop the sampler (a test's; a worker's stops with the process).
_SAMPLER_STOP = threading.Event()
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


#: code object -> (its name in a report, whether it is this package's). Once per code.
_CODES: dict[Any, tuple[str, bool]] = {}


def _frame_name(code: Any) -> tuple[str, bool]:
    found = _CODES.get(code)
    if found is None:
        path = code.co_filename
        found = _CODES[code] = (
            f"{os.path.basename(path)}:{code.co_name}",
            os.path.dirname(os.path.abspath(path)) == _PACKAGE_DIR,
        )
    return found


def _sample_loop(hz: float, target: int) -> None:
    """Every 1/hz seconds, where the main thread is: its innermost frame (`self`), its
    innermost frame in this package (`own`, so numpy and json are charged to the caller
    that asked for them) and every function on its stack once (`inclusive`)."""
    every = 1.0 / hz
    selfs, owns, inclusive = _SAMPLES["self"], _SAMPLES["own"], _SAMPLES["inclusive"]
    while not _SAMPLER_STOP.wait(every):
        frame = sys._current_frames().get(target)  # noqa: SLF001 - the sampler's whole job
        if frame is None:
            continue
        _SAMPLE_TICKS[0] += 1
        leaf = _frame_name(frame.f_code)[0]
        selfs[leaf] = selfs.get(leaf, 0) + 1
        mine = None
        names: set[str] = set()
        while frame is not None:
            name, ours = _frame_name(frame.f_code)
            names.add(name)
            if mine is None and ours:
                mine = name
            frame = frame.f_back
        if mine is not None:
            owns[mine] = owns.get(mine, 0) + 1
        for name in names:
            inclusive[name] = inclusive.get(name, 0) + 1


def _start_sampler() -> float:
    hz = float(os.environ.get(ENV_SAMPLE) or 0) if ON else 0.0
    if hz > 0:
        threading.Thread(
            target=_sample_loop, args=(hz, threading.main_thread().ident), daemon=True,
            name="pokeuraou-sampler",
        ).start()
    return hz


SAMPLE_HZ = _start_sampler()


def stage(name: str) -> Any:
    """A context manager that charges its body to `name`, or a no-op when timing is off."""
    if not ON:
        return _OFF
    found = _STAGES.get(name)
    if found is None:
        found = _STAGES[name] = _Stage(name)
    return found


def purpose(name: str) -> Any:
    """A context manager naming what the work inside it is for, or a no-op when off.

    Nothing is timed by this. A stage that serves several callers reads the innermost
    purpose in force when it charges a purpose row (`rust.fill@rank`) or a purposed count
    (`leaves@rank`), so the stage keeps one row and the split sits beside it. Innermost
    wins: the dirty cells of a hidden bench are filled from inside a node's matrix, and
    they are the dirty cells.
    """
    if not ON:
        return _OFF
    found = _PURPOSE_OBJECTS.get(name)
    if found is None:
        found = _PURPOSE_OBJECTS[name] = _Purpose(name)
    return found


def labelled(name: str) -> Callable[[Callable], Callable]:
    """`purpose` for a whole function; the function itself, untouched, when timing is off."""

    def wrap(function: Callable) -> Callable:
        if not ON:
            return function
        here = purpose(name)

        @functools.wraps(function)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with here:
                return function(*args, **kwargs)

        return inner

    return wrap


def current_purpose() -> str:
    """The innermost purpose in force on this thread, as one of `PURPOSES`."""
    found = getattr(_LOCAL, "purposes", None)
    if not found:
        return "other"
    name = found[-1]
    return name if name in PURPOSES else "other"


def clock() -> float:
    """`perf_counter` when timing is on, 0.0 when it is off -- for a caller's own span."""
    return time.perf_counter() if ON else 0.0


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

    A name in `PURPOSED` is also kept as `name@purpose`, under the purpose in force.
    """
    if not ON:
        return
    _COUNTS[name] = _COUNTS.get(name, 0) + n
    if name in PURPOSED:
        key = f"{name}@{current_purpose()}"
        _COUNTS[key] = _COUNTS.get(key, 0) + n


def decided(kind: str) -> None:
    """Close the stretch since the last call as one decision of `kind`, and open the next.

    Called right after a decision is recorded, when its kind is known -- `move`,
    `selfswitch`, `replacement` -- and at the start of each game as `between`, which is
    what the end of the last game, its record and the next game's set-up cost. Every
    second and every count between two calls therefore belongs to exactly one stretch.
    A move's stretch begins where the previous record ended, so it holds advancing the
    game to this decision as well as searching it. What is kept per kind is the change in
    every stage's clock and every count across the stretch, so a table can say "8
    forward passes a hidden decision" without any site having to know which it is in.

    Callers close one at a point where no stage is running; a stage still open would have
    its time since it last resumed charged to the next stretch instead.

    The first call only ends `startup`: the wall clock from this module's import to here,
    less any stage that ran in between, becomes that row. It is not part of "the rest".
    """
    for hook in _ON_DECIDED:
        hook()
    if not ON:
        return
    _SEEN.clear()
    _FIRST.clear()
    now = time.perf_counter()
    if not _STARTUP_CPU:
        _end_startup(now)
    else:
        _close(now, kind)
    _OPEN[0] = [
        set(),
        now,
        {name: (s.wall, s.calls) for name, s in _STAGES.items()},
        dict(_COUNTS),
    ]


def ready() -> None:
    """End `startup` without opening a decision, for a process that makes none.

    The inference server: its CPU over the time it held a request is read as whether it
    spins (IKA-106), and the seconds it spent importing torch and building a CUDA context
    are in its CPU and not in its holding. With this the reader can take them out.
    """
    if not ON or _STARTUP_CPU:
        return
    _end_startup(time.perf_counter())


def refine(detail: str) -> None:
    """Qualify the open stretch's kind: `move` becomes `move.hidden` or `move.exact`.

    A set rather than a replacement, because two nodes of one decision can disagree -- a
    match whose arms hold different leaves solves each side's game over the other's bench
    alone -- and `move.exact+hidden` says so where a last-write would not.
    """
    if not ON or _OPEN[0] is None:
        return
    _OPEN[0][0].add(detail)


def _kind(kind: str, opened: list[Any]) -> str:
    details = opened[0]
    return kind if not details else f"{kind}.{'+'.join(sorted(details))}"


def _delta(opened: list[Any], now: float) -> dict[str, Any]:
    """What the open stretch has cost so far: its wall clock, stage clocks and counts."""
    _details, started, stages, counts = opened
    moved: dict[str, list[float]] = {}
    for name, found in _STAGES.items():
        wall, calls = stages.get(name, (0.0, 0))
        if found.wall != wall or found.calls != calls:
            moved[name] = [found.wall - wall, found.calls - calls]
    tallied = {
        name: value - counts.get(name, 0)
        for name, value in _COUNTS.items()
        if value != counts.get(name, 0)
    }
    return {"wall": now - started, "stages": moved, "counts": tallied}


def _merge(into: dict[str, Any], delta: dict[str, Any]) -> None:
    into["n"] += 1
    into["wall"] += delta["wall"]
    for name, (wall, calls) in delta["stages"].items():
        row = into["stages"].setdefault(name, [0.0, 0])
        row[0] += wall
        row[1] += calls
    for name, value in delta["counts"].items():
        into["counts"][name] = into["counts"].get(name, 0) + value


def _empty() -> dict[str, Any]:
    return {"n": 0, "wall": 0.0, "stages": {}, "counts": {}}


def _close(now: float, kind: str) -> None:
    opened = _OPEN[0]
    if opened is None:
        return
    _merge(_DECISIONS.setdefault(_kind(kind, opened), _empty()), _delta(opened, now))
    _OPEN[0] = None


def _end_startup(now: float) -> None:
    """Turn the stretch before the first decision into the `startup` row."""
    own_wall = sum(s.wall for name, s in _STAGES.items() if name not in BORROWED)
    own_cpu = sum(s.cpu for name, s in _STAGES.items() if name not in BORROWED)
    row = _STAGES["startup"]
    row.wall = max(now - _STARTED - own_wall, 0.0)
    row.cpu = max(time.thread_time() - _STARTED_CPU - own_cpu, 0.0)
    row.calls = 1
    _STARTUP_CPU.append(time.process_time())


def _decisions() -> dict[str, dict[str, Any]]:
    """The closed stretches by kind, with the open one's cost so far as `between`.

    The stretch still open when a report is written is the one after the last record --
    the last game's end and the process's exit -- which is what `between` holds.
    """
    out = {
        kind: {
            "n": row["n"],
            "wall": row["wall"],
            "stages": {name: list(pair) for name, pair in row["stages"].items()},
            "counts": dict(row["counts"]),
        }
        for kind, row in _DECISIONS.items()
    }
    opened = _OPEN[0]
    if opened is not None:
        trailing = _kind("between", opened)
        _merge(out.setdefault(trailing, _empty()), _delta(opened, time.perf_counter()))
    return out


def serving() -> None:
    """Record that this process scores its leaves on an inference server.

    Called by `RemoteValue` when one is built, so a report's `served` says which leaf the
    process actually held. It used to be read off `POKEURAOU_INFERENCE`, which nothing
    sets: generate_queue.py and match_queue.py hand the address over as `--inference`, so
    every served run reported `served: false` and read as direct (IKA-144). Set whether or
    not timing is on -- it is one assignment, made once a process.
    """
    _SERVED[0] = True


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
        # Which checkout's code this process ran. A worktree run without PYTHONPATH does
        # not fail -- it runs the main checkout's src and reports it as the worktree's.
        "source": str(Path(__file__).resolve().parent),
        "elapsed": time.perf_counter() - _STARTED,
        "rust_node": os.environ.get("POKEURAOU_RUST_NODE", ""),
        # From the leaf this process built, not from the environment (IKA-144).
        "served": _SERVED[0],
        "stages": {
            name: {"wall": s.wall, "cpu": s.cpu, "calls": s.calls}
            for name, s in _STAGES.items()
        },
        "counts": dict(_COUNTS),
        "decisions": _decisions() if ON else {},
        "startup_process_cpu": _STARTUP_CPU[0] if _STARTUP_CPU else None,
        "process": _process_times(),
        # IKA-258: whether this process counted repeats, and its stack samples if any.
        "dupes": DUPES,
        "sample_hz": SAMPLE_HZ,
        "samples": (
            {"ticks": _SAMPLE_TICKS[0], **{k: dict(v) for k, v in _SAMPLES.items()}}
            if SAMPLE_HZ > 0 else None
        ),
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
    "PURPOSED",
    "PURPOSES",
    "PURPOSE_ROWS",
    "STAGES",
    "WORKER_STAGES",
    "add",
    "clock",
    "count",
    "current_purpose",
    "decided",
    "labelled",
    "purpose",
    "ready",
    "refine",
    "repeat",
    "serving",
    "set_total",
    "snapshot",
    "stage",
    "timed",
    "write_report",
]
