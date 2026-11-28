"""IKA-464: a move's best-first deepening, read in the port (`rust/src/deepen.rs`).

The only deepening there is (IKA-464 stage A3 removed Python's). The port holds the read:
`deepenStart` takes the `beliefNode` request of the root (`beliefnode.node_request`) with the
deepening's settings, `deepenRun` steps it for a slice of wall time or of counted cells,
`deepenEnd` lets it go. This module runs the slices, reads the move's budget between them
with `deepen._Meter` (a `deepen.Cost`, `humanplay.WallCost`, `HaltingCost`, `PonderCost`,
`analysis.StopCost`), reports each slice to a `progress` callback as a `deepen.Step`, and
returns what `search.belief_solve` returns for a side that is not deepened.

The children's menus are each side's `CHILD_WIDTH` best by the Q (the root's `q-nocover`
without the cover), not `narrow`'s damage order: the port has no `narrow`. So the Q must be
the inference server's (`qrank.RemoteQ`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import port, portlp
from .actions import MoveAction, PassAction, SideAction, SwitchAction
from .budget import Budget
from .deepen import Deepened, Root, Step, _lost_options, _Meter
from .equilibrium import BayesianEquilibrium
from .position import Position
from .regulation import Regulation

#: Each side's menu in a refined cell's child: the Q's best (IKA-322 §18 measured half the
#: NashConv of `narrow`'s 8 at the same width).
CHILD_WIDTH = 8
#: The branches a refined cell keeps (as the search's depth 2 kept, IKA-12).
BRANCHES = 3
#: Cells a step refines (IKA-467): the port picks this many best cells, fills their children
#: on its cell threads and solves the tree again once. The same at any thread count, so a read
#: on counted cells is the same read on 1 thread or 16; 1 is the read before IKA-467. At 8 the
#: analysis mode's 10 s read refined 1.5 to 4.1 times the cells of 1 on two recorded positions
#: (1, 4 and 16 threads), but at equal time (4 s a move, 1 thread) 8 against 1 was undecided
#: after 1,200 pairs: -7.5 Elo [-19.6, +4.5], SPRT(-10, 0) LLR -0.67.
#: The default is 8. The reason is the user's decision (11/28), made on that undecided result.
BATCH = 8
#: A slice of wall time between two readings of the move's budget, in milliseconds.
SLICE_MS = 250.0
#: Steps in a slice where the budget is read off counted work (a `deepen.Cost`).
SLICE_STEPS = 4
#: The shortest slice of wall time, in milliseconds.
SLICE_MIN_MS = 25.0

#: The least wall time between two walks of the tree for its lines (the guard's count a page
#: shows), in milliseconds; Python's deepening walked its tree no more often than once a
#: second either (`analysis.Session`'s ``guard_every_ms`` before IKA-464).
LINES_EVERY_MS = 1000.0

#: The process's reads and slices, and the walks asked for after a read's last slice: the
#: positive control a test reads.
COUNTS = {"reads": 0, "slices": 0, "walks": 0}


def _ask_run(reg: Regulation, run: dict[str, Any]) -> dict[str, Any]:
    """One `deepenRun` and its answer, a fault raised."""
    got = port.ask(reg, lambda node: node._exchange(run))  # noqa: SLF001
    if "no deepening is held" in str(got.get("fault", "")):
        raise portlp.PortFault(
            "deepenRun", "the read was lost: the port was started again mid-read, and a "
            "read lives only in the process that began it")
    if "fault" in got or "serverError" in got or "tooBig" in got:
        raise portlp.PortFault("deepenRun", str(got))
    return got


def _code(action: SideAction) -> list:
    out = []
    for slot in action.slots:
        if isinstance(slot, MoveAction):
            out.append(["m", slot.slot, slot.move_index, slot.move_id, slot.target, bool(slot.mega)])
        elif isinstance(slot, SwitchAction):
            out.append(["s", slot.slot, slot.party_index, slot.species])
        elif isinstance(slot, PassAction):
            out.append(["p", slot.slot])
        else:
            raise TypeError(f"no port code for {slot!r}")
    return out


def _codes(actions: Sequence[SideAction]) -> list:
    return [_code(a) for a in actions]


def _installed_q(reg: Regulation) -> Any:  # noqa: ANN401 - the installed Q
    """The installed Q (the tests give a stand-in where none is installed)."""
    from . import qrank

    del reg
    return qrank._INSTALLED.get("")  # noqa: SLF001


def _q_server(reg: Regulation) -> dict[str, Any]:
    """The installed Q's server, as the port's Q menus read it (`qmenus.rs`)."""
    from . import qrank

    model = _installed_q(reg)
    if type(model) is not qrank.RemoteQ:
        raise ValueError(
            "the port's deepening ranks the children's menus by the Q on the inference server "
            f"(qrank.RemoteQ); the installed Q is {type(model).__name__}"
        )
    return {
        "qServer": {
            "address": model.address,
            "model": model.model,
            "shm": model._block.name,  # noqa: SLF001 - the Q's own block, idle while the port asks
            "bytes": int(model.buffer_bytes),
        },
        "properties": bool(model.properties),
        "megaFromSlots": bool(model.encoder.rules.mega_from_slots),
    }


def _is_wall(cost: Any) -> bool:  # noqa: ANN401
    """Whether the budget's reading is the wall clock (`humanplay.WallCost`, perhaps inside a
    `HaltingCost` or `PonderCost`, or a reading that says ``wall``: `analysis.StopCost` when
    no step count stops the read): then a slice is wall time, else a few counted steps."""
    from .humanplay import WallCost

    seen = cost
    for _ in range(8):
        if seen is None:
            return False
        if isinstance(seen, WallCost) or getattr(seen, "wall", False) is True:
            return True
        seen = getattr(seen, "inner", None)
    return False


def _prices(cost: Any) -> Any:  # noqa: ANN401
    """The `deepen.Cost` a reading of the budget wraps (`Tally`, `HaltingCost`, `PonderCost`
    keep theirs as ``inner``), or None: read for its prices only, so no reading's own state
    (a brake's, a tally's) is touched."""
    from .deepen import Cost

    seen = cost
    for _ in range(8):
        if isinstance(seen, Cost):
            return seen
        if seen is None:
            return None
        seen = getattr(seen, "__dict__", {}).get("inner")
    return None


@dataclass
class Answer:
    """`search.BeliefAnswer`'s parts a deepened side reads, and the port's last snapshot."""

    strategy: np.ndarray
    replies: list[np.ndarray]
    value: float
    ours: list[SideAction]
    theirs: list[SideAction]
    deepened: Deepened
    unmodelled: set[str]
    snapshot: dict[str, Any]


def _equilibrium(
    snap: dict[str, Any], weights: np.ndarray, m: int, n: int,
) -> tuple[BayesianEquilibrium, list[np.ndarray]]:
    """The snapshot's root: its Bayesian equilibrium and its prices (one matrix a completion,
    in the side's own orientation)."""
    import base64

    flat = np.frombuffer(base64.b64decode(snap["prices"]), dtype="<f8")
    prices = [flat[k * m * n:(k + 1) * m * n].reshape(m, n) for k in range(len(weights))]
    x = np.asarray(snap["x"], dtype=np.float64)
    ys = tuple(np.asarray(y, dtype=np.float64) for y in snap["ys"])
    value = float(snap["value"])
    row_ev = sum(w * (p @ y) for w, p, y in zip(weights, prices, ys, strict=True))
    col_loss = []
    for p in prices:
        per = x @ p
        col_loss.append(np.maximum(per - per.min(), 0.0))
    return BayesianEquilibrium(
        value=value, row_strategy=x, col_strategies=ys, col_marginal=None,
        row_ev=np.asarray(row_ev), row_ev_loss=np.maximum(value - np.asarray(row_ev), 0.0),
        col_ev_loss=tuple(col_loss), weights=weights, duality_gap=0.0,
    ), prices


def deepen_move(  # noqa: PLR0913, PLR0915, C901 - the node, the budget and the report
    reg: Regulation,
    pos: Position,
    me: int,
    ours: list[SideAction],
    theirs: list[SideAction],
    items: list[Any],
    leaf: Any,  # noqa: ANN401 - an inference.RemoteValue
    *,
    budget: Budget,
    cells: float,
    cost: Any = None,  # noqa: ANN401 - deepen.Cost, WallCost or a wrapper
    levels: int | None = None,
    discount: float | None = None,
    outside: tuple[Sequence[SideAction], Sequence[SideAction]] | None = None,
    progress: Callable[[Step], None] | None = None,
    hidden: bool = True,
    grow: Callable[[float], Any] | None = None,
    batch: int | None = None,
) -> Answer:
    """Side ``me``'s deepened answer on the menus ``ours`` (side 0's) x ``theirs`` over the
    other side's completions ``items`` (one exact item for an open node), the budget
    ``cells`` read by ``cost`` as `deepen._Meter` reads it (None: counted cells). ``grow``
    (`deepen.Grow`, IKA-354) is asked before each slice for actions to add to the root.
    ``batch`` is the cells a step refines (None: `BATCH`)."""
    from .beliefnode import _encoder_for, node_request, served_leaf  # noqa: PLC2701
    from .deepen import MAX_LEVELS
    from .encode import rules_of

    you = 1 - me
    served = served_leaf(leaf, "the port's deepening")
    rules = rules_of(served)
    encoder = getattr(served, "encoder", None) or _encoder_for(reg)
    spreads = {you: list(items)}
    slots = items[0].slots if items and not items[0].exact else ()
    request, _jobs, _asks = node_request(reg, served, pos, list(ours), list(theirs), budget, rules,
                                         {you: tuple(slots)}, spreads, (), encoder)
    weights = np.asarray([float(item.weight) for item in items], dtype=np.float64)
    guard = MAX_LEVELS if levels is None else int(levels)
    # The node's request with the deepening's keys beside them: `RustNode._defined` defines the
    # held positions and actions a request names at its top level only.
    start = {
        **request, "kind": "deepenStart", "side": me,
        "menus": [_codes(ours), _codes(theirs)],
        "weights": [float(w) for w in weights],
        "outside": None if outside is None else [_codes(outside[0]), _codes(outside[1])],
        "levels": guard, "discount": discount, "branches": BRANCHES, "childWidth": CHILD_WIDTH,
        "batch": max(1, int(BATCH if batch is None else batch)),
        **_q_server(reg),
    }
    answer = port.ask(reg, lambda node: node._exchange(start))  # noqa: SLF001
    if "fault" in answer or "serverError" in answer or "tooBig" in answer:
        raise portlp.PortFault("deepenStart", str(answer))
    COUNTS["reads"] += 1
    ident = int(answer["id"])
    try:
        return _run(reg, ident, answer, me, ours, theirs, items, weights, cells=cells, cost=cost,
                    levels=levels, outside=outside, progress=progress, hidden=hidden, grow=grow)
    finally:
        # Whatever stopped the read, the port lets it go (`deepenCounts.held` back to 0).
        port.ask(reg, lambda node: node._exchange({"kind": "deepenEnd", "id": ident}))  # noqa: SLF001


def _run(  # noqa: PLR0913, PLR0915, C901 - the read's slices and its report
    reg: Regulation, ident: int, answer: dict[str, Any], me: int, ours: list[SideAction],
    theirs: list[SideAction], items: list[Any], weights: np.ndarray, *, cells: float,
    cost: Any, levels: int | None, outside: Any, progress: Any, hidden: bool,  # noqa: ANN401
    grow: Any,  # noqa: ANN401
) -> Answer:
    """`deepen_move`'s slices on the read ``ident`` the port holds, and its report."""
    from .portmenus import _action  # noqa: PLC2701

    def meter(snap: dict[str, Any]) -> float:
        """`deepen._Meter.spent` of the port's counts: the counted cells, or the cost's units
        of the work so far (fills, refinements, cells and probed cells, the levels at each
        refinement and the root's cells at each step, IKA-467; the port's probe asks no Q, so
        ``qs`` is 0)."""
        read = _Meter(cost)
        read.refines = int(snap["refines"])
        read.cells = int(snap["cells"]) - read.refines
        read.fills = int(snap["fills"])
        read.probed = int(snap["probed"])
        read.levels = int(snap["levelSum"])
        read.roots = int(snap["rootSum"])
        return float(read.spent)

    weights_n = weights / weights.sum()
    index = [0]

    def announce(kind: str, snap: dict[str, Any]) -> None:
        if progress is None:
            return
        own = [_action(a) for a in snap["own"]]
        other = [_action(a) for a in snap["other"]]
        eq, prices = _equilibrium(snap, weights_n, len(own), len(other))
        root = Root(side=me, own=own, other=other, items=list(items), w=weights_n,
                    prices=prices, equilibrium=eq,
                    deep={tuple(int(v) for v in c) for c in snap.get("deep", ())},
                    refused={tuple(int(v) for v in c) for c in snap.get("refusedCells", ())},
                    port=snap)
        if kind not in ("start", "done"):
            index[0] += 1
        progress(Step(
            index=index[0], kind=kind, root=root, budget=int(cells), spent=float(meter(snap)),
            cells=int(snap["cells"]), fills=int(snap["fills"]), refines=int(snap["refines"]),
            expanded=int(snap["expanded"]), depth=int(snap["depth"]), refused=int(snap["refused"]),
            probed=int(snap["probed"]), widened=int(snap["widened"]), swapped=int(snap["swapped"]),
            grown=int(snap["grown"]),
        ))

    wall = _is_wall(cost)
    #: The budget's reading and the refined cells at the last slice's start (a priced step's cap).
    last: list[float] = [0.0, 0]
    walked = [time.perf_counter()]
    snap = answer
    announce("start", snap)
    stop = ""
    while True:
        if snap.get("exhausted"):
            stop = "exhausted"
            break
        if meter(snap) >= cells:
            stop = "budget"
            break
        run: dict[str, Any] = {"kind": "deepenRun", "id": ident}
        if cost is None:
            run["cells"] = int(cells)
            if grow is not None or progress is not None:
                # Slices of a few steps, so the caller hears and widens as the read goes;
                # the steps are the same either way (the budget is read before each).
                run["steps"] = SLICE_STEPS
        elif wall:
            # The wall clock: this slice, or what is left of the budget, at least
            # `SLICE_MIN_MS` (a pondering move's gate reads just under its budget).
            run["ms"] = max(SLICE_MIN_MS, min(SLICE_MS, float(cells) - meter(snap)))
        else:
            # A cost's units are read before every step, as `deepen._Meter` reads them. A
            # step of several cells (IKA-467) takes no more of them than what is left of the
            # budget pays for at the units a refined cell has cost: the larger of the read's
            # mean so far (the oracle's units in it) and the last step's, since a deeper tree's
            # cells cost more. The port cannot read these units between a step's cells.
            run["steps"] = 1
            refines = int(snap["refines"])
            spent = meter(snap)
            if refines:
                per = spent / refines
                if refines > last[1]:
                    per = max(per, (spent - last[0]) / (refines - last[1]))
                last[:] = [spent, refines]
            else:
                # The first step has no refined cell to read: one priced refinement of a full
                # child (a fill, its cells) and the root's cells solved again, in units, at
                # the prices the readings wrap (`_prices`; none: the step is not capped).
                prices = _prices(cost)
                per = None
                if prices is not None:
                    root_cells = len(snap["own"]) * len(snap["other"]) * max(1, int(snap["classes"]))
                    ms = prices.ms(1, 1, 1 + CHILD_WIDTH ** 2) + prices.root * root_cells
                    per = ms / prices.cell
            if per is not None:
                run["batch"] = max(1, math.ceil((float(cells) - spent) / max(per, 1e-9)))
        wanted = None if grow is None else grow(meter(snap))
        if wanted is not None:
            run["grow"] = [_codes(wanted[0]), _codes(wanted[1])]
        # The lines are counted by walking the tree, at most every `LINES_EVERY_MS` (they
        # change nothing the read does; a page shows them).
        now = time.perf_counter()
        walk = (now - walked[0]) * 1000.0 >= LINES_EVERY_MS
        run["walk"] = walk
        if walk:
            walked[0] = now
        snap = _ask_run(reg, run)
        COUNTS["slices"] += 1
        announce("refine", snap)
    if snap.get("linesFresh") is False:
        # The report's lines are the tree's as it ended: one walk, no step.
        snap = _ask_run(reg, {"kind": "deepenRun", "id": ident, "steps": 0, "walk": True})
        COUNTS["walks"] += 1
    own = [_action(a) for a in snap["own"]]
    other = [_action(a) for a in snap["other"]]
    lines = tuple(int(v) for v in snap["lines"])
    uncovered: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    if int(snap["swapped"]):
        mine_before, theirs_before = (ours, theirs) if me == 0 else (theirs, ours)
        mine = _lost_options(reg, mine_before, own)
        others = _lost_options(reg, theirs_before, other)
        uncovered = (mine, others) if me == 0 else (others, mine)
    report = Deepened(
        budget=int(cells), cells=int(snap["cells"]), expanded=int(snap["expanded"]),
        depth=int(snap["depth"]), refused=int(snap["refused"]), oracle=outside is not None,
        probed=int(snap["probed"]), widened=int(snap["widened"]), fills=int(snap["fills"]),
        swap=outside is not None, swapped=int(snap["swapped"]), uncovered=uncovered,
        classes=len(items) if hidden else 0, levels=levels, stop=stop,
        lines=(lines[0], lines[1], lines[2], lines[3]), guarded=int(snap["guarded"]),
        child_q=True, child_passes=int(snap["childPasses"]), child_ranked=int(snap["childRanked"]),
        grown=int(snap["grown"]), grown_cells=int(snap["grownCells"]),
        batch=int(snap["batch"]), steps=int(snap["steps"]),
    )
    announce("done", snap)
    ours_now, theirs_now = (own, other) if me == 0 else (other, own)
    return Answer(
        strategy=np.asarray(snap["x"], dtype=np.float64),
        replies=[np.asarray(y, dtype=np.float64) for y in snap["ys"]],
        value=float(snap["value"]), ours=ours_now, theirs=theirs_now, deepened=report,
        unmodelled=set(snap["unmodelled"]), snapshot=snap,
    )


__all__ = ["BATCH", "BRANCHES", "CHILD_WIDTH", "COUNTS", "SLICE_MS", "Answer", "deepen_move"]
