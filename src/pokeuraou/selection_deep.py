"""The selection game read deeper than the leaf (IKA-392).

`selection.solve_selection` fills the 90 x 90 matrix of one pair with the value function's
one estimate of each turn-1 position (about a second). This reads the cells that matter as
the depth-1 node of their turn-1 position (`humanplay.solve_move`: the menus and the matrix
of the move, not the leaf's one estimate), and solves the game again on the port's LP:

* the prices start as the leaf matrix; a cell read replaces its price with the read's value;
* a stage takes a rectangle of the current answer -- its support first, then the rows that
  do best against the other side's answer (and the columns that do worst against ours) --
  reads the cells of it not read yet, all in the same way, and solves the whole game on the
  prices in hand;
* a confirmation reads the cells of the new answer's support and best replies that are
  still unread, and solves again, until none is or the passes are used;
* the answer is the last stage or confirmation that *completed*, so it is the equilibrium of
  prices of which every cell it plays and is played against was read the same way.

**A read is not a leaf.** The deep value of a cell is systematically off the leaf's (the
depth-1 read of a turn-1 position measured +0.04 above the leaf on average, +0.09..+0.13
with the bench hidden; IKA-392 plan). Solving a game in which the cells read are deep and
the rest leaf would find the read cells better than they are: the price of a cell not read
is the leaf plus a shift estimated from the cells read (``shift``: ``none``, ``const``: the
mean of (read - leaf), ``add``: c + a_i + b_j fitted by ridge regression on the read cells;
a row or column with few reads keeps its own effect near zero, so ``add`` falls back to
``const``). Whether the shift helps or distorts is measured, not assumed: the B0 sweep
compares the three.

The cells are read with the bench open: both fours are known. A real turn 1 hides the other
side's two benched members, so this hands both players more than they have (the direction
is not known); reading it hidden costs 3-4 times as much (`records/IKA-392.md`).

`read_cell` is the one reading of a cell; `SerialReader` reads them here, `PoolReader` on
worker processes (each with the leaf and the Q, one port thread), for a person's 90 s.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import portlp
from .budget import Budget
from .equilibrium import BayesianEquilibrium
from .priors import SampledSet
from .selection import SelectionAnalysis, SpreadClass, solve_selection

SHIFTS = ("none", "const", "add")
#: How a cell is read: the depth-1 node of its turn-1 position, the only reading left. A spec
#: may still name it as the ladder's ``L6@0`` (no stage of the ladder: the depth-1 answer),
#: which is what a person's game played (IKA-440). IKA-464 removed the ladder and the deeper
#: stages a spec could name (``d2r4b3k8``, the reading ``default`` was before).
DEPTH1_STAGES = ("L6@0", "d1")

#: The reader a person's game reads its selection on (`humanplay.play`; None: this process,
#: `SerialReader`). A tool that has worker processes sets it (`use_reader`).
READER: Any = None


def use_reader(reader: Any) -> None:  # noqa: ANN401
    """The reader `humanplay.play` hands `solve_selection_deep` (None: read in the process)."""
    global READER  # noqa: PLW0603 - the process's one reader
    READER = reader


@dataclass(frozen=True, slots=True)
class Reading:
    """How the selection is read deeper: a spec like ``rects=8-11,confirm=2,shift=add``.

    Each cell is the depth-1 node of its turn-1 position (`DEPTH1_STAGES`). ``rects``: the
    rectangle's side of each stage, ``confirm``: confirmation passes after each stage,
    ``shift``: how the unread cells are priced (module docstring), ``width``: the menus'
    width at each cell, ``fill``: with a deadline, keep widening (by 8) after ``rects``.
    """

    rects: tuple[int, ...] = (8, 11)
    confirm: int = 2
    shift: str = "add"
    width: int = 64
    fill: bool = True
    #: The ridge weight of the additive shift, in cells.
    ridge: float = 2.0

    @property
    def label(self) -> str:
        return (f"rects={'-'.join(str(r) for r in self.rects)},"
                f"confirm={self.confirm},shift={self.shift},width={self.width}")


def parse_reading(spec: str) -> Reading:
    """`Reading` from its spec; ``default`` is the reading a person's game plays
    (`humanplay.PLAY_SELECTION_READING`)."""
    base = Reading()
    if spec in ("", "default"):
        return base
    given: dict[str, Any] = {}
    # A reading is written with commas or, inside a condition (whose keys are comma
    # separated), semicolons.
    for part in spec.replace(";", ",").split(","):
        key, _sep, value = part.partition("=")
        if key == "stage":
            if value not in DEPTH1_STAGES:
                raise ValueError(
                    f"a selection cell is read at depth 1 ({' or '.join(DEPTH1_STAGES)}); the "
                    f"ladder's stages are gone (IKA-464), not {value!r}"
                )
        elif key == "rects":
            given["rects"] = tuple(int(v) for v in value.split("-"))
        elif key in ("confirm", "width"):
            given[key] = int(value)
        elif key == "ridge":
            given["ridge"] = float(value)
        elif key == "shift":
            if value not in SHIFTS:
                raise ValueError(f"shift is one of {SHIFTS}, not {value!r}")
            given["shift"] = value
        elif key == "fill":
            given["fill"] = value not in ("0", "off", "no")
        else:
            raise ValueError(f"unknown selection reading key {key!r} in {spec!r}")
    return replace(base, **given)


# --------------------------------------------------------------------------- one cell


def read_cell(  # noqa: PLR0913 - a cell and how it is read
    reg: Any,  # noqa: ANN401
    our_sets: Sequence[SampledSet],
    their_sets: Sequence[SampledSet],
    leaf: Any,  # noqa: ANN401
    reading: Reading,
    *,
    rank_fill: str | None = "q-nocover",
    rank_by_leaf: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Side 0's value of the turn-1 position of two ordered fours: the depth-1 node of that
    position, both benches open (`humanplay.solve_move`, the port's `beliefNode`).

    A draw among the leads' switch-ins (`selfplay.lead_branches_from_sets`) is every outcome
    read at its weight, the value their mean, as `solve_selection` treats it (IKA-352).
    """
    from . import humanplay
    from .selfplay import _menus, lead_branches_from_sets

    branches = lead_branches_from_sets(reg, [(list(our_sets), list(their_sets))])[0]
    total = 0.0
    weight = 0.0
    for w, pos in branches:
        ours, theirs = _menus(reg, pos, (reading.width, reading.width), leaf, Budget.matrix(),
                              rank_by_leaf, None, None, rank_fill=rank_fill)
        if not ours or not theirs:
            raise RuntimeError("a selection cell has no menu")
        solved = humanplay.solve_move(
            reg, pos, 0, list(ours), list(theirs), None, leaf, budget=Budget.matrix(), exact=True,
        )
        total += float(w) * float(solved.value)
        weight += float(w)
    return total / weight, {"branches": len(branches)}


# --------------------------------------------------------------------------- readers


class SerialReader:
    """Reads the cells in this process, one after the other."""

    workers = 1

    def __init__(self, reg: Any, leaf: Any, *, rank_fill: str | None = "q-nocover",  # noqa: ANN401
                 rank_by_leaf: bool = True) -> None:
        self.reg, self.leaf, self.rank_fill, self.rank_by_leaf = reg, leaf, rank_fill, rank_by_leaf
        #: Cells read, and the seconds they took (a positive control that the path ran).
        self.cells = 0
        self.seconds = 0.0

    def read(
        self, reading: Reading, our_six: Sequence[SampledSet], their_six: Sequence[SampledSet],
        selections: Sequence[tuple[int, ...]], cells: Sequence[tuple[int, int]],
        deadline: float | None = None,
    ) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for i, j in cells:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            began = time.perf_counter()
            value, _info = read_cell(
                self.reg, [our_six[k] for k in selections[i]],
                [their_six[k] for k in selections[j]], self.leaf, reading,
                rank_fill=self.rank_fill, rank_by_leaf=self.rank_by_leaf)
            self.seconds += time.perf_counter() - began
            self.cells += 1
            out[(i, j)] = value
        return out

    def close(self) -> None:
        return None


def _worker(format_id: str, spec: tuple, rank_fill: str | None, rank_by_leaf: bool,
            tasks: Any, results: Any, stop: Any) -> None:  # noqa: ANN401
    """A pool worker: its own regulation, leaf and Q (`humanplay.worker_leaf`) and a
    port of one thread; reads the cells it is sent, or hands one back unread once ``stop``."""
    import traceback

    from . import humanplay, rustnode
    from .damage import register_mega_stones
    from .regulation import load_regulation

    reg = load_regulation(format_id)
    register_mega_stones(reg)
    rustnode.set_port_threads(1)
    try:
        leaf = humanplay.worker_leaf(reg, *spec)
    except Exception as error:  # noqa: BLE001 - said to the parent
        results.put(("failed", f"{type(error).__name__}: {error}"))
        return
    results.put(("ready", None))
    while True:
        task = tasks.get()
        if task is None:
            return
        generation, i, j, ours, theirs, reading = task
        if stop.is_set():
            results.put((generation, i, j, None, "skipped"))
            continue
        try:
            value, _info = read_cell(reg, ours, theirs, leaf, reading, rank_fill=rank_fill,
                                     rank_by_leaf=rank_by_leaf)
            results.put((generation, i, j, value, None))
        except Exception:  # noqa: BLE001 - said to the parent, which stops
            results.put((generation, i, j, None, traceback.format_exc()))


def _worker_args(args: tuple[Any, ...], index: int) -> tuple[Any, ...]:
    """IKA-390: a served leaf's address may name several inference servers (``a,b``): worker
    ``index`` asks the ``index mod n``th, its leaf, its Q and its port's direct requests
    alike (they all read the address the leaf was made with). One address: unchanged."""
    if len(args) > 1 and args[0] == "served" and isinstance(args[1], str) and "," in args[1]:
        addresses = [a for a in args[1].split(",") if a]
        return (args[0], addresses[index % len(addresses)], *args[2:])
    return args


class PoolReader:
    """Reads the cells on ``count`` worker processes (spawned, kept for the reader's life)."""

    def __init__(self, reg: Any, count: int, spec: tuple, *,  # noqa: ANN401
                 rank_fill: str | None = "q-nocover", rank_by_leaf: bool = True) -> None:
        context = multiprocessing.get_context("spawn")
        self.tasks = context.Queue()
        self.results = context.Queue()
        self.stop = context.Event()
        self.generation = 0
        self.cells = 0
        self.seconds = 0.0
        self.processes = []

        for k in range(count):
            process = context.Process(
                target=_worker, args=(reg.meta.format_id, _worker_args(tuple(spec), k),
                                      rank_fill, rank_by_leaf,
                                      self.tasks, self.results, self.stop), daemon=True)
            process.start()
            self.processes.append(process)
        for _ in range(count):
            state, why = self.results.get()
            if state != "ready":
                self.close()
                raise RuntimeError(f"a selection reading worker did not start: {why}")

    @property
    def workers(self) -> int:
        return len(self.processes)

    def read(
        self, reading: Reading, our_six: Sequence[SampledSet], their_six: Sequence[SampledSet],
        selections: Sequence[tuple[int, ...]], cells: Sequence[tuple[int, int]],
        deadline: float | None = None,
    ) -> dict[tuple[int, int], float]:
        self.generation += 1
        self.stop.clear()
        began = time.perf_counter()
        for i, j in cells:
            self.tasks.put((self.generation, i, j, [our_six[k] for k in selections[i]],
                            [their_six[k] for k in selections[j]], reading))
        out: dict[tuple[int, int], float] = {}
        replies = 0
        while replies < len(cells):
            wait = 0.2 if deadline is None else max(0.0, min(0.2, deadline - time.perf_counter()))
            try:
                generation, i, j, value, error = self.results.get(timeout=max(wait, 0.01))
            except queue.Empty:
                if deadline is not None and time.perf_counter() >= deadline:
                    self.stop.set()
                if not any(p.is_alive() for p in self.processes):
                    raise RuntimeError("every selection reading worker died") from None
                continue
            if generation != self.generation:
                continue
            replies += 1
            if error == "skipped":
                continue
            if error is not None:
                self.stop.set()
                raise RuntimeError(f"a selection cell failed in a worker:\n{error}")
            out[(i, j)] = value
        self.stop.clear()
        self.cells += len(out)
        self.seconds += time.perf_counter() - began
        return out

    def close(self) -> None:
        for _ in self.processes:
            self.tasks.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.processes = []


class LazyPoolReader:
    """A `PoolReader` made when a selection is read and closed when it is done (`release`), so
    the workers do not sit on memory and cores through the game's moves. Made again for the
    next game (a second or so on a served leaf)."""

    def __init__(self, reg: Any, count: int, spec: tuple, *,  # noqa: ANN401
                 rank_fill: str | None = "q-nocover", rank_by_leaf: bool = True) -> None:
        self.reg, self.count, self.spec = reg, count, spec
        self.rank_fill, self.rank_by_leaf = rank_fill, rank_by_leaf
        self.inner: PoolReader | None = None
        self.workers = count

    @property
    def cells(self) -> int:
        return 0 if self.inner is None else self.inner.cells

    @property
    def seconds(self) -> float:
        return 0.0 if self.inner is None else self.inner.seconds

    def read(self, *args: Any, **kwargs: Any) -> dict[tuple[int, int], float]:  # noqa: ANN401
        if self.inner is None:
            self.inner = PoolReader(self.reg, self.count, self.spec, rank_fill=self.rank_fill,
                                    rank_by_leaf=self.rank_by_leaf)
        return self.inner.read(*args, **kwargs)

    def release(self) -> None:
        if self.inner is not None:
            self.inner.close()
            self.inner = None

    def close(self) -> None:
        self.release()


class MemoReader:
    """``reader`` with the cells it read kept in ``memo`` (IKA-416): two readings of one pair
    (two conditions of a match) read a cell once. A cell is named by what its read depends on
    -- ``tag`` (the leaf and the menus, the caller's), the reading's width and the two ordered
    fours -- so a hit is the value the reader would give again (a cell's read is the depth-1
    node of one position: the same every time). The answer keeps the
    order of ``cells``, as the reader's does."""

    def __init__(self, reader: Any, memo: dict, tag: Any = None) -> None:  # noqa: ANN401
        self.reader, self.memo, self.tag = reader, memo, tag
        self.workers = getattr(reader, "workers", 1)
        #: Cells answered from the memo, and cells read (a positive control that it ran).
        self.hits = 0
        self.cells = 0

    def read(
        self, reading: Reading, our_six: Sequence[SampledSet], their_six: Sequence[SampledSet],
        selections: Sequence[tuple[int, ...]], cells: Sequence[tuple[int, int]],
        deadline: float | None = None,
    ) -> dict[tuple[int, int], float]:
        head = (self.tag, reading.width)

        def name(cell: tuple[int, int]) -> tuple:
            return (head, tuple(selections[cell[0]]), tuple(selections[cell[1]]))

        fresh = [c for c in cells if name(c) not in self.memo]
        got = self.reader.read(reading, our_six, their_six, selections, fresh, deadline) if fresh else {}
        for c, v in got.items():
            self.memo[name(c)] = v
        self.cells += len(got)
        out: dict[tuple[int, int], float] = {}
        for c in cells:
            if name(c) in self.memo:
                out[c] = self.memo[name(c)]
        self.hits += len(out) - len(got)
        return out

    def release(self) -> None:
        if hasattr(self.reader, "release"):
            self.reader.release()

    def close(self) -> None:
        if hasattr(self.reader, "close"):
            self.reader.close()


# --------------------------------------------------------------------------- the prices


def fit_shift(
    read: dict[tuple[int, int], float], leaf: np.ndarray, mode: str, ridge: float = 2.0
) -> np.ndarray:
    """The shift added to every unread cell's leaf price (module docstring)."""
    shape = leaf.shape
    if mode == "none" or not read:
        return np.zeros(shape)
    cells = list(read)
    d = np.array([read[c] - leaf[c] for c in cells])
    if mode == "const":
        return np.full(shape, float(d.mean()))
    n, m = shape
    a = np.zeros((len(cells) + n + m, 1 + n + m))
    a[:len(cells), 0] = 1.0
    for k, (i, j) in enumerate(cells):
        a[k, 1 + i] = 1.0
        a[k, 1 + n + j] = 1.0
    root = float(np.sqrt(ridge))
    for k in range(n + m):
        a[len(cells) + k, 1 + k] = root
    y = np.concatenate([d, np.zeros(n + m)])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return coef[0] + coef[1:1 + n][:, None] + coef[1 + n:][None, :]


def prices(
    leaf: np.ndarray, read: dict[tuple[int, int], float], mode: str, ridge: float = 2.0
) -> np.ndarray:
    """Every cell's price: the value where read, else the leaf plus the shift, kept in [0, 1]."""
    p = np.clip(leaf + fit_shift(read, leaf, mode, ridge), 0.0, 1.0)
    for (i, j), v in read.items():
        p[i, j] = v
    return p


def _rectangle(
    p: np.ndarray, x: np.ndarray, y: np.ndarray, count: int
) -> tuple[list[int], list[int]]:
    """The rows and columns of a stage: the support first, then the rows best against ``y``
    and the columns worst against ``x`` (`ladder`'s `_order`)."""
    def order(strength: np.ndarray, value: np.ndarray, larger: bool) -> list[int]:
        support = [k for k in np.argsort(-strength) if strength[k] > 1e-6]
        rest = [k for k in np.argsort(-value if larger else value) if strength[k] <= 1e-6]
        return [int(k) for k in (support + rest)[:count]]

    return order(x, p @ y, True), order(y, x @ p, False)


def _replies(p: np.ndarray, x: np.ndarray, y: np.ndarray, count: int) -> tuple[list[int], list[int]]:
    """The support and the ``count`` best replies of each side."""
    rows_ev, cols_ev = p @ y, x @ p
    rows = {int(k) for k in np.flatnonzero(x > 1e-6)} | {int(k) for k in np.argsort(-rows_ev)[:count]}
    cols = {int(k) for k in np.flatnonzero(y > 1e-6)} | {int(k) for k in np.argsort(cols_ev)[:count]}
    return sorted(rows), sorted(cols)


# --------------------------------------------------------------------------- the solve


@dataclass(slots=True)
class DeepReport:
    """What a deeper selection did: for the record and the sweep's controls."""

    reading: str
    leaf_value: float
    value: float
    #: One row per completed step: label, cells read so far, seconds, value, row strategy.
    steps: list[dict[str, Any]] = field(default_factory=list)
    cells: int = 0
    seconds: float = 0.0
    completed: str = ""
    leaf_row: np.ndarray | None = None
    leaf_col: np.ndarray | None = None
    read: dict[tuple[int, int], float] = field(default_factory=dict)
    leaf_matrix: np.ndarray | None = None


def solve_selection_deep(  # noqa: PLR0913, PLR0912, PLR0915, C901 - the stages, the clock, the record
    reg: Any,  # noqa: ANN401
    our_six: Sequence[SampledSet],
    their_six: Sequence[SampledSet],
    evaluate: Callable[[list], np.ndarray],
    reader: Any,  # noqa: ANN401
    reading: Reading,
    *,
    deadline: float | None = None,
    base: SelectionAnalysis | None = None,
) -> tuple[SelectionAnalysis, DeepReport]:
    """The selection game of one pair read as ``reading`` says (the module's docstring).

    ``deadline`` is a `time.perf_counter` time: stages are begun while it is not past (and
    ``reading.fill`` widens beyond ``reading.rects``); None reads exactly the stages named.
    """
    started = time.perf_counter()
    first = base or solve_selection(
        reg, our_six, [SpreadClass(weight=1.0, sets=tuple(their_six), label="sheet")], evaluate)
    leaf = np.asarray(first.matrices[0], dtype=np.float64)
    selections = first.ours
    x = np.asarray(first.equilibrium.row_strategy, dtype=np.float64)
    y = np.asarray(first.equilibrium.col_strategies[0], dtype=np.float64)
    report = DeepReport(reading=reading.label, leaf_value=float(first.value), value=float(first.value),
                        leaf_row=x.copy(), leaf_col=y.copy(), leaf_matrix=leaf.copy())
    read: dict[tuple[int, int], float] = {}
    answer = (x, y, leaf, first.equilibrium)
    completed = "leaf"

    def past() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def resolve() -> tuple[np.ndarray, BayesianEquilibrium]:
        p = prices(leaf, read, reading.shift, reading.ridge)
        # The port's LP (IKA-464), as the selection's own solve (`selectionSolve`) is.
        got = portlp.solve_bayesian_many(reg, [([p], np.ones(1))])[0]
        if isinstance(got, Exception):
            raise got
        return p, got

    def fetch(cells: list[tuple[int, int]]) -> bool:
        fresh = [c for c in cells if c not in read]
        if not fresh:
            return True
        got = reader.read(reading, our_six, their_six, selections, fresh, deadline)
        read.update(got)
        return len(got) == len(fresh)

    def record(label: str, eq: BayesianEquilibrium) -> None:
        report.steps.append({
            "step": label, "cells": len(read), "seconds": round(time.perf_counter() - started, 2),
            "value": float(eq.value),
            "support": [int((np.asarray(eq.row_strategy) > 1e-6).sum()),
                        int((np.asarray(eq.col_strategies[0]) > 1e-6).sum())],
        })

    sides = list(reading.rects)
    stage = 0
    stopped = False
    while not stopped:
        if stage >= len(sides):
            if not (reading.fill and deadline is not None) or sides[-1] >= len(selections):
                break
            sides.append(min(len(selections), sides[-1] + 8))
        if past():
            break
        p, eq = resolve()
        x, y = np.asarray(eq.row_strategy), np.asarray(eq.col_strategies[0])
        rows, cols = _rectangle(p, x, y, sides[stage])
        if not fetch([(i, j) for i in rows for j in cols]):
            break
        p, eq = resolve()
        x, y = np.asarray(eq.row_strategy), np.asarray(eq.col_strategies[0])
        answer, completed = (x, y, p, eq), f"stage {stage + 1} (rect {sides[stage]})"
        record(completed, eq)
        for k in range(reading.confirm):
            rows, cols = _replies(p, x, y, 2)
            cells = [(i, j) for i in rows for j in cols if (i, j) not in read]
            if not cells:
                break
            if not fetch(cells):
                stopped = True
                break
            p, eq = resolve()
            x, y = np.asarray(eq.row_strategy), np.asarray(eq.col_strategies[0])
            answer, completed = (x, y, p, eq), f"stage {stage + 1} confirm {k + 1}"
            record(completed, eq)
        stage += 1

    x, y, p, eq = answer
    seconds = time.perf_counter() - started
    report.value = float(eq.value)
    report.cells = len(read)
    report.seconds = seconds
    report.completed = completed
    report.read = dict(read)
    notes = first.notes + (
        f"選出を深く読んだ（{reading.label}、最後に完了した段: {completed}、読んだセル "
        f"{len(read)}、{seconds:.1f} 秒）。セルは裏を開いた読みで、実際の 1 ターン目は相手の裏 2 体が"
        "見えないので、両者に情報を与えすぎた値になる（向きは不明）。",
    )
    analysis = replace(first, matrices=(p,), equilibrium=eq, notes=notes,
                       seconds=first.seconds + seconds)
    return analysis, report


__all__ = [
    "DeepReport", "LazyPoolReader", "MemoReader", "PoolReader", "Reading", "SerialReader", "fit_shift",
    "parse_reading", "prices", "read_cell", "solve_selection_deep",
]
