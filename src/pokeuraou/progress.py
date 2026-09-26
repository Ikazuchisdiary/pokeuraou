"""What the agent's answer looks like while it forms (IKA-332): a deepening's `Step` read
into plain Python objects -- numbers, numpy arrays, actions and short texts -- for a host
that shows it.

The engine (`deepen`) calls back with a `Step` holding a live reference to the root; this
module reads it *there and then* (the next step moves it) and keeps copies. Nothing is
serialised here: the host decides how to send it (`liveview` packs it into binary frames).
Nothing is written to the tree and nothing is computed that the search did not already
compute -- the mixtures, prices and children are read as they stand.

**What a snapshot holds** (`Snapshot`), always from the agent's side (``me``):

- the counts of the step (cells, fills, refinements, levels, what the oracle did);
- the agent's mixture over its menu, and each action's EV loss against the current answer;
- the value, in side 0's units (``value0``, the number the record writes) and as the
  agent's own (``value``);
- the opponent's mixture: where the agent cannot see the opponent's bench, one per
  completion of that bench (裏の決定化) with the belief's weight, and their average;
- ``read`` -- the share of the equilibrium's joint play whose cell has been read deeper than
  its leaf: a cheap measure of how much of what matters the deepening has reached;
- the principal variation as a tree (`pv`), below.

**The principal variation.** A simultaneous-move game has no single line. At a node the
likeliest pairs of the two mixtures (``p`` = x_i * y_j, over completions where the bench
is hidden) each head a row with the pair's value. A pair whose cell was deepened opens its
chance branches -- the turn's outcomes, the likeliest `search.DEFAULT_SUB_BRANCHES` kept
and renormalised, each described by what changed (HP, faints, status, who came in) and
valued; a branch that opened a node shows that node's two mixtures, its value, and its own
likeliest pairs, as deep as the tree was read. A pair whose cell was not deepened is marked
as the leaf's value (``read`` "leaf"). The tree is cut to `PV_PAIRS` pairs at the root and
`PV_PAIRS_BELOW` below it, and `PV_LEVELS` levels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .actions import SideAction, target_names
from .deepen import Step, _BeliefRoot, _Node
from .hidden import identity
from .position import Position
from .regulation import Regulation

#: Pairs shown under the root and under every node below it, and the levels shown.
PV_PAIRS = 4
PV_PAIRS_BELOW = 2
PV_LEVELS = 6
#: A pair lighter than this is not shown (its row would say nothing).
PV_MIN_P = 0.005
#: The top actions of each side shown at a node below the root.
PV_TOP = 3

#: Between the parts of an action's label, one per active slot (the page splits on it).
SLOT_SEPARATOR = " ／ "


def action_label(reg: Regulation, action: SideAction, pos: Position, side: int, loc: Any = None) -> str:  # noqa: ANN401
    """An action as the page shows it: each slot's part (`describe`, targets named), in slot
    order, joined by `SLOT_SEPARATOR` -- slot k is the side's k-th active Pokemon."""
    targets = target_names(pos, side)
    return SLOT_SEPARATOR.join(slot.describe(reg, loc, targets) for slot in action.slots)


@dataclass(slots=True)
class PvNode:
    """A node below the root: a branch position whose next turn was solved."""

    value: float
    ours: list[tuple[str, float]]
    theirs: list[tuple[str, float]]
    pairs: list[PvPair]
    #: The node has deepened pairs past `PV_LEVELS` that are not shown.
    more: bool = False


@dataclass(slots=True)
class PvBranch:
    """One chance outcome of a deepened pair's turn."""

    weight: float
    value: float
    what: str
    ended: bool = False
    node: PvNode | None = None


@dataclass(slots=True)
class PvPair:
    """One pair of actions at a node, with its joint probability and value."""

    ours: str
    theirs: str
    p: float
    value: float
    #: ``deep`` (the cell was deepened: `branches`), ``leaf`` (the leaf's value) or
    #: ``refused`` (it could not be deepened and keeps the leaf's value).
    read: str
    #: The completion of the opponent's bench (index into `Snapshot.classes`), or -1.
    klass: int = -1
    branches: list[PvBranch] = field(default_factory=list)
    #: The cell in its node: (row, column), or (completion, row, column) at a Bayesian root.
    cell: tuple[int, ...] = ()


@dataclass(slots=True)
class ClassView:
    """One completion of the opponent's bench: its weight, who it puts there, and the
    opponent's mixture if the bench is that one."""

    weight: float
    #: The species it puts on the bench: names (localised) and ids (the dex's).
    bench: tuple[str, ...]
    p: np.ndarray
    value: float
    bench_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class Snapshot:
    """The answer as it stands at one step (the module's docstring)."""

    decision: int
    turn: int
    step: int
    kind: str
    ms: float
    budget: int
    spent: float
    cells: int
    fills: int
    refines: int
    expanded: int
    depth: int
    refused: int
    probed: int
    widened: int
    swapped: int
    ours: list[SideAction]
    #: The two root menus' labels (localised, targets named), in menu order.
    our_labels: list[str]
    their_labels: list[str]
    our_p: np.ndarray
    our_loss: np.ndarray
    theirs: list[SideAction]
    their_p: np.ndarray
    their_loss: np.ndarray
    classes: list[ClassView]
    value0: float
    value: float
    read: float
    gap: float
    pv: list[PvPair]

    @property
    def support(self) -> int:
        return int(np.count_nonzero(self.our_p > 1e-9))


def _mine(value0: float, me: int) -> float:
    """A side-0 value as side ``me``'s own (the game is constant-sum at 1)."""
    return value0 if me == 0 else 1.0 - value0


class Reader:
    """Reads a `Step` for side ``me`` at ``pos``: labels (localised when ``loc`` is
    given), the principal variation, the belief. ``names`` names the two sides in the
    branch texts, by side index."""

    def __init__(
        self,
        reg: Regulation,
        pos: Position,
        me: int,
        *,
        loc: Any = None,  # noqa: ANN401 - names.Localiser or None
        names: tuple[str, str] = ("側0", "側1"),
        pairs: int = PV_PAIRS,
        pairs_below: int = PV_PAIRS_BELOW,
        levels: int = PV_LEVELS,
    ) -> None:
        self.reg = reg
        self.pos = pos
        self.me = me
        self.loc = loc
        self.names = names
        self.pairs = pairs
        self.pairs_below = pairs_below
        self.levels = levels
        self._labels: dict[tuple[str, int, str], str] = {}

    # -- labels
    def label(self, action: SideAction, pos: Position, side: int) -> str:
        actives = tuple(
            None if m is None else m.species for s in pos.sides for m in s.active_pokemon()
        )
        key = (repr(actives), side, action.to_choice())
        got = self._labels.get(key)
        if got is None:
            got = action_label(self.reg, action, pos, side, self.loc)
            self._labels[key] = got
        return got

    def species(self, species: str) -> str:
        return self.loc.species(species) if self.loc is not None else species

    # -- the snapshot
    def read(self, step: Step, *, decision: int = 0, turn: int = 0, ms: float = 0.0) -> Snapshot:
        root = step.root
        me = self.me
        classes: list[ClassView] = []
        if isinstance(root, _BeliefRoot):
            eq = root.equilibrium
            own = root.side == me
            if not own:
                raise ValueError("a Bayesian root is read from its own side")
            value0 = float(eq.value) if root.side == 0 else -float(eq.value)
            ours = list(root.own)
            our_p = np.array(eq.row_strategy, dtype=np.float64)
            our_loss = np.array(eq.row_ev_loss, dtype=np.float64)
            theirs = list(root.other)
            w = root.w
            their_p = sum(wk * np.asarray(y) for wk, y in zip(w, eq.col_strategies, strict=True))
            their_loss = sum(
                wk * np.asarray(loss) for wk, loss in zip(w, eq.col_ev_loss, strict=True)
            )
            for k, item in enumerate(root.items):
                y = np.array(eq.col_strategies[k], dtype=np.float64)
                v_own = float(our_p @ root.prices[k] @ y)
                v0 = v_own if root.side == 0 else -v_own
                classes.append(ClassView(
                    weight=float(w[k]),
                    bench=tuple(self.species(s) for s in getattr(item, "species", ())),
                    p=y, value=_mine(v0, me),
                    bench_ids=tuple(getattr(item, "species", ())),
                ))
            read = 0.0
            for (k, i, j) in root.children:
                read += float(w[k] * our_p[i] * eq.col_strategies[k][j])
            gap = float(eq.duality_gap)
        else:
            eq = root.equilibrium
            value0 = float(eq.value)
            x = np.array(eq.row_strategy, dtype=np.float64)
            y = np.array(eq.col_strategy, dtype=np.float64)
            if me == 0:
                ours, theirs = list(root.rows), list(root.cols)
                our_p, their_p = x, y
                our_loss = np.array(eq.row_ev_loss, dtype=np.float64)
                their_loss = np.array(eq.col_ev_loss, dtype=np.float64)
            else:
                ours, theirs = list(root.cols), list(root.rows)
                our_p, their_p = y, x
                our_loss = np.array(eq.col_ev_loss, dtype=np.float64)
                their_loss = np.array(eq.row_ev_loss, dtype=np.float64)
            read = float(sum(x[i] * y[j] for (i, j) in root.children))
            gap = float(eq.duality_gap)
        return Snapshot(
            decision=decision, turn=turn, step=step.index, kind=step.kind, ms=ms,
            budget=step.budget, spent=step.spent, cells=step.cells, fills=step.fills,
            refines=step.refines, expanded=step.expanded, depth=step.depth,
            refused=step.refused, probed=step.probed, widened=step.widened,
            swapped=step.swapped, ours=ours,
            our_labels=[self.label(a, self.pos, me) for a in ours],
            their_labels=[self.label(a, self.pos, 1 - me) for a in theirs],
            our_p=our_p, our_loss=our_loss, theirs=theirs,
            their_p=np.asarray(their_p, dtype=np.float64),
            their_loss=np.asarray(their_loss, dtype=np.float64), classes=classes,
            value0=value0, value=_mine(value0, me), read=read, gap=gap,
            pv=self.pv(root),
        )

    # -- the principal variation
    def pv(self, root: Any) -> list[PvPair]:  # noqa: ANN401
        if isinstance(root, _BeliefRoot):
            return self._belief_pairs(root)
        return self._pairs(root, self.pairs, 0)

    def _belief_pairs(self, root: _BeliefRoot) -> list[PvPair]:
        eq = root.equilibrium
        x = np.asarray(eq.row_strategy, dtype=np.float64)
        joint = []
        for k, y in enumerate(eq.col_strategies):
            for i in np.flatnonzero(x > 1e-9):
                for j in np.flatnonzero(np.asarray(y) > 1e-9):
                    joint.append((float(root.w[k] * x[i] * y[j]), k, int(i), int(j)))
        joint.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
        out = []
        for p, k, i, j in joint[: self.pairs]:
            if p < PV_MIN_P:
                break
            pos = root.items[k].position
            price = float(root.prices[k][i, j])
            v0 = price if root.side == 0 else -price
            cell = (k, i, j)
            pair = PvPair(
                ours=self.label(root.own[i], pos, root.side),
                theirs=self.label(root.other[j], pos, 1 - root.side),
                p=p, value=_mine(v0, self.me), read=self._read_of(root, cell), klass=k,
                cell=cell,
            )
            if cell in root.children:
                pair.branches = self._branches(pos, root.children[cell], 1)
            out.append(pair)
        return out

    @staticmethod
    def _read_of(node: Any, cell: tuple[int, ...]) -> str:  # noqa: ANN401
        if cell in node.children:
            return "deep"
        return "refused" if cell in node.refused else "leaf"

    def _pairs(self, node: _Node, count: int, level: int) -> list[PvPair]:
        eq = node.equilibrium
        x = np.asarray(eq.row_strategy, dtype=np.float64)
        y = np.asarray(eq.col_strategy, dtype=np.float64)
        rows = np.flatnonzero(x > 1e-9)
        cols = np.flatnonzero(y > 1e-9)
        joint = sorted(
            ((float(x[i] * y[j]), int(i), int(j)) for i in rows for j in cols),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        out = []
        for p, i, j in joint[:count]:
            if p < PV_MIN_P:
                break
            ours_side, theirs_side = (0, 1) if self.me == 0 else (1, 0)
            acts = (node.rows[i], node.cols[j])
            pair = PvPair(
                ours=self.label(acts[ours_side], node.pos, self.me),
                theirs=self.label(acts[theirs_side], node.pos, 1 - self.me),
                p=p, value=_mine(float(node.payoff[i, j]), self.me),
                read=self._read_of(node, (i, j)), cell=(i, j),
            )
            if (i, j) in node.children:
                pair.branches = self._branches(node.pos, node.children[(i, j)], level + 1)
            out.append(pair)
        return out

    def _branches(
        self, before: Position, kept: Sequence[tuple[float, Any]], level: int
    ) -> list[PvBranch]:
        out = []
        for weight, child in kept:
            if isinstance(child, _Node):
                branch = PvBranch(
                    weight=float(weight), value=_mine(child.value, self.me),
                    what=self.describe(before, child.pos),
                )
                if level < self.levels:
                    branch.node = self._node(child, level)
                else:
                    branch.node = PvNode(
                        value=_mine(child.value, self.me), ours=[], theirs=[], pairs=[],
                        more=bool(child.children),
                    )
            else:
                branch = PvBranch(
                    weight=float(weight), value=_mine(float(child), self.me),
                    what="決着", ended=True,
                )
            out.append(branch)
        return out

    def _node(self, node: _Node, level: int) -> PvNode:
        eq = node.equilibrium
        x = np.asarray(eq.row_strategy, dtype=np.float64)
        y = np.asarray(eq.col_strategy, dtype=np.float64)
        mine, other = (x, y) if self.me == 0 else (y, x)
        menus = (node.rows, node.cols) if self.me == 0 else (node.cols, node.rows)

        def top(p: np.ndarray, acts: Sequence[SideAction], side: int) -> list[tuple[str, float]]:
            order = sorted(np.flatnonzero(p > 1e-9), key=lambda i: (-p[i], i))[:PV_TOP]
            return [(self.label(acts[i], node.pos, side), float(p[i])) for i in order]

        return PvNode(
            value=_mine(node.value, self.me),
            ours=top(mine, menus[0], self.me),
            theirs=top(other, menus[1], 1 - self.me),
            pairs=self._pairs(node, self.pairs_below, level),
        )

    # -- a branch in words
    def describe(self, before: Position, after: Position) -> str:
        """What a turn changed, side by side: HP (as a share of the maximum), faints,
        status, who came in. The agent's side is named first."""
        parts = []
        for side in (self.me, 1 - self.me):
            old = {identity(m): m for m in before.sides[side].pokemon}
            bits = []
            for mon in after.sides[side].pokemon:
                prev = old.get(identity(mon))
                if prev is None:
                    continue
                name = self.species(mon.species)
                if mon.fainted and not prev.fainted:
                    bits.append(f"{name} ひんし")
                    continue
                change = mon.hp - prev.hp
                if change and mon.maxhp:
                    bits.append(f"{name} {100.0 * change / mon.maxhp:+.0f}%")
                if mon.status and mon.status != prev.status:
                    status = self.loc.status(mon.status) if self.loc is not None else mon.status
                    bits.append(f"{name} {status}")
                if mon.active_index is not None and prev.active_index is None:
                    bits.append(f"{name} 登場")
            if bits:
                parts.append(f"{self.names[side]}: " + "・".join(bits))
        return " / ".join(parts) if parts else "変化なし"


class Recorder:
    """The callback a host hands the search: reads each `Step` it is given into a
    `Snapshot` and passes it to ``sink`` -- the first and the last always, the ones in
    between at most every ``interval_ms`` (reading a snapshot costs a little; ten a second
    is plenty to watch). ``clock`` gives seconds; ``started`` is the decision's start."""

    def __init__(
        self,
        reader: Reader,
        sink: Callable[[Snapshot], None],
        *,
        decision: int,
        turn: int,
        started: float,
        interval_ms: float = 100.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.reader = reader
        self.sink = sink
        self.decision = decision
        self.turn = turn
        self.started = started
        self.interval_ms = interval_ms
        self.clock = clock or time.perf_counter
        self.last: float | None = None
        self.calls = 0
        self.sent = 0

    def __call__(self, step: Step) -> None:
        self.calls += 1
        now = self.clock()
        if (
            step.kind not in ("start", "done")
            and self.last is not None
            and (now - self.last) * 1000.0 < self.interval_ms
        ):
            return
        self.last = now
        self.sent += 1
        self.sink(self.reader.read(
            step, decision=self.decision, turn=self.turn, ms=(now - self.started) * 1000.0
        ))


__all__ = [
    "PV_LEVELS",
    "PV_MIN_P",
    "PV_PAIRS",
    "PV_PAIRS_BELOW",
    "SLOT_SEPARATOR",
    "ClassView",
    "PvBranch",
    "PvNode",
    "PvPair",
    "Reader",
    "Recorder",
    "Snapshot",
    "action_label",
]
