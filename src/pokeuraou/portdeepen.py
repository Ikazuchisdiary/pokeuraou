"""IKA-464: a move's best-first deepening read in the port (`rust/src/deepen.rs`).

`deepen.deepen_belief` / `deepen.deepen_root` held the tree, the priorities, the swap oracle
and every re-solve in Python (scipy's LP), with the turns and fills in the port. Here the
port holds the read: `deepenStart` takes the `beliefNode` request of the root
(`beliefnode.node_request`) with the deepening's settings, `deepenRun` steps it for a slice
of wall time or of counted cells, `deepenEnd` lets it go. This module runs the slices,
reads the move's budget between them as `deepen._Meter` read it (a `deepen.Cost`,
`humanplay.WallCost`, `HaltingCost`, `PonderCost`), reports each slice to a `progress`
callback as a `deepen.Step`, and returns what `search.belief_solve` returned for a deepened
side.

The children's menus are each side's `CHILD_WIDTH` best by the Q (the root's `q-nocover`
without the cover, `deepen.py`'s ``c<k>``), not `narrow`'s damage order: the port has no
`narrow`. So the Q must be the inference server's (`qrank.RemoteQ`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import port, portlp
from .actions import MoveAction, PassAction, SideAction, SwitchAction
from .budget import Budget
from .deepen import Deepened, Step, _lost_options
from .equilibrium import BayesianEquilibrium
from .position import Position
from .regulation import Regulation

#: Each side's menu in a refined cell's child: the Q's best (IKA-322 §18 measured half the
#: NashConv of `narrow`'s 8 at the same width).
CHILD_WIDTH = 8
#: The branches a refined cell keeps (`search.DEFAULT_SUB_BRANCHES`).
BRANCHES = 3
#: A slice of wall time between two readings of the move's budget, in milliseconds.
SLICE_MS = 250.0
#: Steps in a slice where the budget is read off counted work (a `deepen.Cost`).
SLICE_STEPS = 4

#: The process's reads and slices: the positive control a test reads.
COUNTS = {"reads": 0, "slices": 0}


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


def _q_server() -> dict[str, Any]:
    """The installed Q's server, as `qMenus` reads it."""
    from . import qrank

    model = qrank._INSTALLED.get("")  # noqa: SLF001
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
    `HaltingCost` or `PonderCost`): then a slice is wall time, else a few counted steps."""
    from .humanplay import WallCost

    seen = cost
    for _ in range(8):
        if seen is None:
            return False
        if isinstance(seen, WallCost):
            return True
        seen = getattr(seen, "inner", None)
    return False


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
    """The snapshot's root as `deepen._BeliefRoot` held it: its equilibrium and its prices."""
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
) -> Answer:
    """Side ``me``'s deepened answer on the menus ``ours`` (side 0's) x ``theirs`` over the
    other side's completions ``items`` (one exact item for an open node), the budget
    ``cells`` read by ``cost`` as `deepen._Meter` reads it (None: counted cells). ``grow``
    (`deepen.Grow`, IKA-354) is asked before each slice for actions to add to the root."""
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
        **_q_server(),
    }
    answer = port.ask(reg, lambda node: node._exchange(start))  # noqa: SLF001
    if "fault" in answer or "serverError" in answer or "tooBig" in answer:
        raise portlp.PortFault("deepenStart", str(answer))
    COUNTS["reads"] += 1
    ident = int(answer["id"])
    from .portmenus import _action  # noqa: PLC2701

    def meter(snap: dict[str, Any]) -> float:
        refines = int(snap["refines"])
        counted = int(snap["cells"]) - refines
        if cost is None:
            return float(refines + counted)
        ms = cost.ms(int(snap["fills"]), refines, counted, int(snap["probed"]), 0)
        units = ms / cost.cell
        gate = getattr(cost, "gate", None)
        return units if gate is None else gate(units, int(snap["fills"]) + refines + counted)

    weights_n = weights / weights.sum()
    index = [0]

    def announce(kind: str, snap: dict[str, Any]) -> None:
        if progress is None:
            return
        own = [_action(a) for a in snap["own"]]
        other = [_action(a) for a in snap["other"]]
        eq, prices = _equilibrium(snap, weights_n, len(own), len(other))
        root = SimpleNamespace(side=me, own=own, other=other, prices=prices, w=weights_n,
                               equilibrium=eq, children={}, refused=set(), level=0, port=snap)
        if kind not in ("start", "done"):
            index[0] += 1
        progress(Step(
            index=index[0], kind=kind, root=root, budget=int(cells), spent=float(meter(snap)),
            cells=int(snap["cells"]), fills=int(snap["fills"]), refines=int(snap["refines"]),
            expanded=int(snap["expanded"]), depth=int(snap["depth"]), refused=int(snap["refused"]),
            probed=int(snap["probed"]), widened=int(snap["widened"]), swapped=int(snap["swapped"]),
        ))

    wall = _is_wall(cost)
    snap = answer
    announce("start", snap)
    stop = ""
    try:
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
                # The wall clock: this slice, or what is left of the budget.
                run["ms"] = max(1.0, min(SLICE_MS, float(cells) - meter(snap)))
            else:
                run["steps"] = SLICE_STEPS
            wanted = None if grow is None else grow(meter(snap))
            if wanted is not None:
                run["grow"] = [_codes(wanted[0]), _codes(wanted[1])]
            got = port.ask(reg, lambda node, run=run: node._exchange(run))  # noqa: SLF001
            if "fault" in got or "serverError" in got or "tooBig" in got:
                raise portlp.PortFault("deepenRun", str(got))
            COUNTS["slices"] += 1
            snap = got
            announce("refine", snap)
    finally:
        port.ask(reg, lambda node: node._exchange({"kind": "deepenEnd", "id": ident}))  # noqa: SLF001
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
    )
    announce("done", snap)
    ours_now, theirs_now = (own, other) if me == 0 else (other, own)
    return Answer(
        strategy=np.asarray(snap["x"], dtype=np.float64),
        replies=[np.asarray(y, dtype=np.float64) for y in snap["ys"]],
        value=float(snap["value"]), ours=ours_now, theirs=theirs_now, deepened=report,
        unmodelled=set(snap["unmodelled"]), snapshot=snap,
    )


__all__ = ["BRANCHES", "CHILD_WIDTH", "COUNTS", "SLICE_MS", "Answer", "deepen_move"]
