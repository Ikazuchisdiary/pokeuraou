"""The production roads' resolver: the Rust port, and nothing behind it (IKA-209).

Generation, the analyser, the node solver and the selection solve used to reach the port
through `resolve`, which answered in Python whatever the port declined or could not be
asked: a turn that paused for a replacement, the replacement phase, the leads, depth 2, an
objective the port does not implement, a bridge that broke. Each of those is now a port
command (IKA-211) or a stop, never a Python resolve:

* `turn` / `resume_alternatives` / `turn_leaves` -- a turn, every replacement of a paused
  one, and a turn flattened into leaves and a fold (Python's `resolve_turn`,
  `resume_alternatives` and `turn_leaves`);
* `replacements_needed` / `resolve_replacements` / `apply_lead_abilities`;
* `batched_payoff(s)` -- a node's matrices: filled over there for a ported objective,
  encoded over there for a learned leaf, and for anything else (a hand-written objective,
  two learned leaves with different encoding rules) resolved over there and scored here;
* a cell the port refuses raises `PortRefused`; a missing, stale or switched-off binary
  raises `rustnode.PortUnavailable`; a broken process is restarted and asked again, and
  once the restarts run out that raises too (`ask`).

Python's resolver itself is gone (IKA-212); `tools/count_resolver_calls.py`, which said the
roads did not reach it, went with it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import rustnode, timing
from .actions import SideAction
from .budget import Budget
from .fold import Average, BestOf, Fold, LeafRef, TurnLeaves, _fold_from_json, _shift, fold_value
from .position import Position
from .regulation import Regulation
from .rustnode import (
    EncodedAlternatives,
    PortPause,
    PortPhase,
    PortTurn,
    PortUnavailable,
    RustNode,
)

#: The objectives the port scores itself.
PORTED_OBJECTIVES = frozenset({"hp-share", "faints"})

#: How many leaves the scored-here road holds before scoring them (`resolve.LEAF_CHUNK`'s
#: knob and default, read the same way: a typo stops the process).
LEAF_CHUNK = int(os.environ.get("POKEURAOU_LEAF_CHUNK", "32768"))


class PortRefused(RuntimeError):
    """The port declined a turn, and there is no Python resolver to hand it to."""


def ask[T](reg: Regulation, call: Callable[[RustNode], T]) -> T:
    """`call` on the warm node; a failure restarts the process and asks again.

    A node keeps nothing between requests (`rustnode.RESTARTS_ALLOWED`), so the same call on
    a fresh process is the same answer. Once the restarts run out this raises: the road it
    used to fall back to was the Python resolver, and there is none (IKA-209).
    """
    while True:
        node = rustnode.require_node(reg)
        try:
            return call(node)
        except (PortRefused, PortUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 - restarted, then re-raised
            if not rustnode.disable(str(exc)):
                raise PortUnavailable(f"the Rust node kept failing: {exc}") from exc


def _refused(node: RustNode, what: str) -> PortRefused:
    return PortRefused(f"the port refused {what}: {node.refusal}")


# -- One turn, and what only Python's resolver used to answer (IKA-211's commands).


def turn(
    reg: Regulation,
    pos: Position,
    actions: Sequence[SideAction],
    budget: Budget,
    *,
    full: bool = False,
    select: int | None = None,
) -> PortTurn:
    """`resolve_turn`: every branch and pause with `full`, the one at `select` with that."""

    def call(node: RustNode) -> PortTurn:
        answer = node.turn(pos, list(actions), budget, full=full, select=select)
        if answer is None:
            raise _refused(node, "a turn")
        return answer

    return ask(reg, call)


def weights(reg: Regulation, pos: Position, actions: Sequence[SideAction], budget: Budget):  # noqa: ANN201
    """The turn's branch and pause weights, without the positions (`RustNode.resolve`)."""

    def call(node: RustNode):  # noqa: ANN202
        answer = node.resolve(pos, list(actions), budget)
        if answer is None:
            raise _refused(node, "a turn")
        return answer

    return ask(reg, call)


def branch(
    reg: Regulation, pos: Position, actions: Sequence[SideAction], budget: Budget, index: int
) -> Position:
    """The position of branch `index` (branches first, as `weights` lists them)."""

    def call(node: RustNode) -> Position:
        answer = node.resolve(pos, list(actions), budget, select=index)
        if answer is None or answer.position is None:
            raise _refused(node, f"branch {index} of a turn")
        return answer.position

    return ask(reg, call)


def resume_alternatives(
    reg: Regulation, pause: PortPause, *, world: tuple[Position, int] | None = None
) -> tuple[int | None, list[tuple[SideAction, PortTurn]]]:
    """Every replacement the paused side could send in, and the whole turn each produces;
    with `world`, of the pause rebuilt in that completion (`paused_in`)."""

    def call(node: RustNode) -> tuple[int | None, list[tuple[SideAction, PortTurn]]]:
        answer = node.resume_alternatives(pause, world=world, full=True)
        if answer is None:
            raise _refused(node, "a paused turn")
        return answer

    return ask(reg, call)


def alternatives_encoded(
    reg: Regulation,
    pause: PortPause,
    *,
    world: tuple[Position, int] | None = None,
    want: Sequence[int] | None = None,
    shared: tuple[int, Sequence[int]] | None = None,
    rules: Any = None,  # noqa: ANN401 - EncodingRules
    objectives: Sequence[str] = (),
    encode: bool = True,
) -> EncodedAlternatives:
    """`resume_alternatives` with the wanted options' turns flattened and encoded over there
    (`RustNode.alternatives_encoded`): the self-switch node's leaves as arrays (IKA-209)."""

    def call(node: RustNode) -> EncodedAlternatives:
        answer = node.alternatives_encoded(
            pause, world=world, want=want, shared=shared, rules=rules,
            objectives=objectives, encode=encode,
        )
        if answer is None:
            raise _refused(node, "a paused turn")
        return answer

    return ask(reg, call)


def resume(reg: Regulation, pause: PortPause, choices: Sequence[SideAction]) -> PortTurn:
    """`resume_turn`: the rest of a paused turn with both sides' choices, every branch."""

    def call(node: RustNode) -> PortTurn:
        answer = node.resume(pause, list(choices), full=True)
        if answer is None:
            raise _refused(node, "a paused turn")
        return answer

    return ask(reg, call)


def fold_from_json(node: dict) -> Fold:
    """The port's fold tree with `numpy.float64` weights, as `turn_leaves` builds its own:
    `fold_value` sums the parts with `sum()`, which adds exact floats differently (IKA-209)."""
    if "leaf" in node:
        return LeafRef(index=int(node["leaf"]))
    if "best" in node:
        return BestOf(
            chooser=int(node["best"]),
            options=[fold_from_json(option) for option in node["options"]],
        )
    return Average(parts=[(np.float64(w), fold_from_json(part)) for w, part in node["avg"]])


def turn_leaves(reg: Regulation, result: PortTurn, *, depth: int = 0) -> TurnLeaves:
    """A whole turn -- pauses included -- as leaf positions and a fold (`resolve.turn_leaves`).

    The same walk in the same order: the branches, then each pause as a `BestOf` over its
    replacements, each resumed turn flattened in turn. `result` must be a full answer.

    The weights are `numpy.float64`, as the Python resolver's were. Not for the precision --
    it is the same double -- but for the sum: `fold_value` adds the parts with `sum()`, which
    since Python 3.12 compensates a run of exact `float`s and adds anything else plainly, so
    the same weights as `float` folded to a value 2 ulp away (IKA-209, M-C self-switches).
    """
    if result.outcomes is None or result.pauses is None:
        raise ValueError("turn_leaves needs a full turn (full=True)")
    positions: list[Position] = []
    parts: list[tuple[float, Fold]] = []
    unmodelled: set[str] = set(result.unmodelled)

    def add_leaf(position: Position) -> LeafRef:
        positions.append(position)
        return LeafRef(index=len(positions) - 1)

    for outcome in result.outcomes:
        parts.append((np.float64(outcome.probability), add_leaf(outcome.position)))
    for pause in result.pauses:
        if depth >= 4:
            unmodelled.add("more than four mid-turn replacements in one turn")
            parts.append((np.float64(pause.probability), add_leaf(pause.position)))
            continue
        chooser, alternatives = resume_alternatives(reg, pause)
        if chooser is None or not alternatives:
            unmodelled.add("a suspended turn offered no replacement")
            parts.append((np.float64(pause.probability), add_leaf(pause.position)))
            continue
        options: list[Fold] = []
        for _option, resumed in alternatives:
            sub = turn_leaves(reg, resumed, depth=depth + 1)
            offset = len(positions)
            positions.extend(sub.positions)
            options.append(_shift(sub.root, offset))
            unmodelled |= set(sub.unmodelled)
        parts.append((np.float64(pause.probability), BestOf(chooser=chooser, options=options)))
    return TurnLeaves(
        positions=positions, root=Average(parts=parts), unmodelled=tuple(sorted(unmodelled))
    )


def remaining_switches(pause: PortPause, side: int) -> bool:
    """Whether `side` still has a switch queued in the rest of the paused turn."""
    return any(
        int(queued["side"]) == side and queued["kind"] == "switch"
        for queued in pause.raw["state"]["remaining"]
    )


def self_switches_needed(pos: Position) -> tuple[tuple[bool, ...], ...]:
    """Per side, per active slot, whether the port left a self-switch waiting on a choice.

    Not a rule: it reads the `pendingselfswitch` flag the port wrote onto a paused position,
    and the bench it can be answered from (`resolve::self_switches_needed` in the port reads
    the same). Narrower than `replacements_needed` on purpose: a faint is answered after the
    turn and a forced switch is a random drag, but a self-switch interrupts the turn *now*.
    The one copy on this side (IKA-212): it was also `resolve.self_switches_needed`,
    `tests/_port.self_switches_needed` and `tools/diff_turn._owed_self_switches`.
    """
    out: list[tuple[bool, ...]] = []
    for side in pos.sides:
        bench = sum(1 for mon in side.pokemon if not mon.fainted and not mon.is_active)
        flags: list[bool] = []
        for party_index in side.active:
            mon = side.pokemon[party_index] if party_index is not None else None
            flags.append(
                bench > 0
                and mon is not None
                and not mon.fainted
                and mon.has_volatile("pendingselfswitch")
            )
        out.append(tuple(flags))
    return tuple(out)


def turn_expectation(
    reg: Regulation, result: PortTurn, value: Callable[[Position], float]
) -> tuple[float, tuple[str, ...]]:
    """The turn's value under `value` (side 0's view), every mid-turn replacement chosen
    by the fold `turn_leaves` builds; `result` must be a full turn.

    Python's `resolve.turn_expectation`, for the analysis tools that score one turn at a
    time (IKA-212). A turn without a pause is the probability-weighted mean of its branches,
    normalised by their total, as `TurnResult.expected` was.
    """
    if result.outcomes is None or result.pauses is None:
        raise ValueError("turn_expectation needs a full turn (full=True)")
    if not result.pauses:
        total = sum(o.probability for o in result.outcomes)
        if total <= 0:
            return 0.0, tuple(result.unmodelled)
        return (
            sum(o.probability * value(o.position) for o in result.outcomes) / total,
            tuple(result.unmodelled),
        )
    plan = turn_leaves(reg, result)
    return plan.value([value(p) for p in plan.positions]), plan.unmodelled


def replacements_needed(reg: Regulation, pos: Position) -> tuple[tuple[bool, ...], ...]:
    def call(node: RustNode) -> tuple[tuple[bool, ...], ...]:
        answer = node.replacements_needed(pos)
        if answer is None:
            raise _refused(node, "a replacement check")
        return answer

    return ask(reg, call)


def resolve_replacements(
    reg: Regulation,
    pos: Position,
    choices: Sequence[SideAction],
    *,
    rng: Any = None,  # noqa: ANN401 - numpy.random.Generator
) -> PortPhase:
    """The replacement phase. Its draws come from `rng` as Python's `_draw` took them.

    Not retried after a failure part-way: a draw already taken from `rng` would be taken
    again. A failure with `rng` stops; without one the request is stateless and is asked
    again on a fresh process.
    """

    def call(node: RustNode) -> PortPhase:
        answer = node.resolve_replacements(pos, list(choices), rng=rng)
        if answer is None:
            raise _refused(node, "a replacement phase")
        return answer

    return _phase(reg, call, rng)


def apply_lead_abilities(
    reg: Regulation, pos: Position, *, rng: Any = None  # noqa: ANN401
) -> PortPhase:
    """The leads' switch-ins before `|turn|1`, with draws as `resolve_replacements`'."""

    def call(node: RustNode) -> PortPhase:
        answer = node.apply_lead_abilities(pos, rng=rng)
        if answer is None:
            raise _refused(node, "the leads")
        return answer

    return _phase(reg, call, rng)


def apply_lead_abilities_many(reg: Regulation, positions: Sequence[Position]) -> list[PortPhase]:
    """`apply_lead_abilities` without a generator, for many positions: pipelined, so the
    selection solve's 8,100 turn-1 positions do not wait on 8,100 round trips."""

    def call(node: RustNode) -> list[PortPhase]:
        answers = node.apply_lead_abilities_many(positions)
        if any(answer is None for answer in answers):
            raise _refused(node, "the leads")
        return answers

    return ask(reg, call)


def _phase(reg: Regulation, call: Callable[[RustNode], PortPhase], rng: Any) -> PortPhase:  # noqa: ANN401
    if rng is None:
        return ask(reg, call)
    node = rustnode.require_node(reg)
    try:
        return call(node)
    except (PortRefused, PortUnavailable):
        raise
    except Exception as exc:
        rustnode.disable(str(exc))
        raise PortUnavailable(
            f"the Rust node failed inside a phase that draws from the game's generator: {exc}"
        ) from exc


# -- A node's matrices.


def batched_payoff(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: Callable[[list[Position]], np.ndarray],
    *,
    budget: Budget,
) -> tuple[np.ndarray, set[str]]:
    """One payoff matrix; see `batched_payoffs`."""
    matrices, unmodelled, _exact = batched_payoffs(
        reg, pos, ours, theirs, [evaluate], budget=budget
    )
    return matrices[0], unmodelled


def batched_payoffs(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluators: Sequence[Callable[[list[Position]], np.ndarray]],
    *,
    budget: Budget,
    cells: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    """One payoff matrix per evaluator, and which cells the budget resolved exactly.

    `resolve.batched_payoffs` with the bridge on, less its Python: the same three roads in
    the same arithmetic -- the ported objectives filled over there (`fill`), a learned leaf
    over the port's encoded leaves (`fill_encoded`) -- and a third where the Python fill was
    (`_scored_here`). A refused cell raises.
    """
    names = objective_names(evaluators)
    if names is not None and set(names) <= PORTED_OBJECTIVES:
        return _filled(reg, pos, ours, theirs, names, budget, cells)
    plan = encoded_leaf_plan(evaluators)
    if plan is not None and any(scorer is not None for _name, scorer in plan):
        from .encode import rules_of

        learned = [scorer for _name, scorer in plan if scorer is not None]
        wanted_rules = {rules_of(scorer) for scorer in learned}
        if len(wanted_rules) == 1:
            return _encoded(reg, pos, ours, theirs, plan, learned, budget, cells)
    # A hand-written objective, or two learned leaves wanting different encodings.
    return _scored_here(reg, pos, ours, theirs, evaluators, budget, cells)


def raise_refused(refused: Sequence[tuple[int, int, str]]) -> None:
    if not refused:
        return
    reasons = sorted({why for _i, _j, why in refused})
    raise PortRefused(
        f"the port refused {len(refused)} cell(s) of a node ({'; '.join(reasons)}), and "
        "there is no Python resolver to fill them (IKA-209)"
    )


def _filled(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    names: list[str],
    budget: Budget,
    cells: Sequence[tuple[int, int]] | None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    filled = ask(reg, lambda node: node.fill(pos, list(ours), list(theirs), names, budget, cells))
    raise_refused(filled.refused)
    payoffs = [np.array(matrix, dtype=np.float64) for matrix in filled.payoffs]
    return payoffs, set(filled.unmodelled), np.array(filled.exact, dtype=bool)


def note_port_rule(scorers: Sequence[Callable], filled: object) -> None:
    """Book the rule the port says it encoded with against each asking leaf's encoder."""
    echo = filled.mega_from_slots
    what = "rust can_mega=" + {True: "slots", False: "holder", None: "unechoed"}[echo]
    booked: list[object] = []
    for scorer in scorers:
        encoder = getattr(getattr(scorer, "__self__", None), "encoder", None)
        if encoder is None or not hasattr(encoder, "note") or any(encoder is e for e in booked):
            continue
        booked.append(encoder)
        encoder.note(what, len(filled.encoded.species))


def _encoded(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    plan: list[tuple[str | None, Callable | None]],
    learned: list[Callable],
    budget: Budget,
    cells: Sequence[tuple[int, int]] | None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    """`resolve._rust_encoded_payoffs`, the same folds in the same order."""
    from .encode import rules_of

    rules = rules_of(learned[0])
    named = [name for name, _scorer in plan if name is not None]
    filled = ask(
        reg,
        lambda node: node.fill_encoded(
            pos, list(ours), list(theirs), budget, named, cells, rules=rules
        ),
    )
    raise_refused(filled.refused)
    note_port_rule(learned, filled)
    payoffs = [np.zeros((len(ours), len(theirs)), dtype=np.float64) for _ in plan]
    exact = np.array(filled.exact, dtype=bool)
    timing.count("leaves.node", len(filled.encoded.species))
    for index, (name, score) in enumerate(plan):
        values = (
            np.asarray(filled.leaf_values[name], dtype=np.float64)
            if score is None
            else np.asarray(score(filled.encoded), dtype=np.float64)
        )
        for i, j, indices, weights in filled.spans:
            if not weights:
                continue
            payoffs[index][i, j] = float(values[indices] @ np.asarray(weights))
        for i, j, root in filled.folded:
            payoffs[index][i, j] = fold_value(_fold_from_json(root), values)
    return payoffs, set(filled.unmodelled), exact


class HeldLeaves:
    """Cells resolved by the port and scored here: `resolve.HeldLeaves` on the port's turns."""

    def __init__(self) -> None:
        self.leaves: list[Position] = []
        self.weights: list[np.ndarray] = []
        self.spans: list[tuple[int, int, int, int]] = []
        self.folded: list[tuple[int, int, Fold]] = []

    def resolve_cell(
        self,
        reg: Regulation,
        pos: Position,
        ours: Sequence[SideAction],
        theirs: Sequence[SideAction],
        i: int,
        j: int,
        budget: Budget,
        exact: np.ndarray,
        unmodelled: set[str],
    ) -> None:
        leaves = self.leaves
        result = turn(reg, pos, [ours[i], theirs[j]], budget, full=True)
        exact[i, j] = result.exact
        if result.suspended:
            plan = turn_leaves(reg, result)
            unmodelled.update(plan.unmodelled)
            if plan.positions:
                self.folded.append((i, j, plan.shifted(len(leaves))))
                leaves.extend(plan.positions)
            return
        unmodelled.update(result.unmodelled)
        outcomes = result.outcomes or []
        total = sum(outcome.probability for outcome in outcomes)
        if not outcomes or total <= 0:
            self.spans.append((i, j, len(leaves), 0))
            self.weights.append(np.zeros(0))
            return
        start = len(leaves)
        leaves.extend(outcome.position for outcome in outcomes)
        self.weights.append(np.array([outcome.probability for outcome in outcomes]) / total)
        self.spans.append((i, j, start, len(outcomes)))

    def write(self, values: np.ndarray, payoff: np.ndarray) -> None:
        for (i, j, start, count), w in zip(self.spans, self.weights, strict=True):
            if count:
                payoff[i, j] = float(values[start : start + count] @ w)
        for i, j, root in self.folded:
            payoff[i, j] = fold_value(root, values)

    def clear(self) -> None:
        self.leaves.clear()
        self.weights.clear()
        self.spans.clear()
        self.folded.clear()


def _scored_here(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluators: Sequence[Callable[[list[Position]], np.ndarray]],
    budget: Budget,
    cells: Sequence[tuple[int, int]] | None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    """Each cell's turn from the port, its leaves scored here by every evaluator.

    Where `resolve.batched_payoffs` resolved in Python for an evaluator the port cannot
    score. The leaves are held a chunk at a time, split on cell boundaries, as there.
    """
    payoffs = [np.zeros((len(ours), len(theirs)), dtype=np.float64) for _ in evaluators]
    exact = np.zeros((len(ours), len(theirs)), dtype=bool)
    unmodelled: set[str] = set()
    held = HeldLeaves()

    def score_held() -> None:
        leaves = held.leaves
        timing.count("leaves.node", len(leaves))
        for payoff, evaluate in zip(payoffs, evaluators, strict=True):
            values = np.asarray(evaluate(leaves), dtype=np.float64) if leaves else np.zeros(0)
            held.write(values, payoff)
        held.clear()

    wanted = (
        [(i, j) for i in range(len(ours)) for j in range(len(theirs))]
        if cells is None
        else [(i, j) for i, j in cells if i < len(ours) and j < len(theirs)]
    )
    for i, j in wanted:
        held.resolve_cell(reg, pos, ours, theirs, i, j, budget, exact, unmodelled)
        if LEAF_CHUNK and len(held.leaves) >= LEAF_CHUNK:
            score_held()
    score_held()
    return payoffs, unmodelled, exact


def objective_names(evaluators: Sequence[Callable]) -> list[str] | None:
    """The objective behind each evaluator, when every one is a plain named objective."""
    names: list[str] = []
    for evaluate in evaluators:
        name = getattr(getattr(evaluate, "__self__", None), "name", None)
        if not isinstance(name, str):
            return None
        names.append(name)
    return names


def encoded_leaf_plan(
    evaluators: Sequence[Callable],
) -> list[tuple[str | None, Callable | None]] | None:
    """How each evaluator scores the port's encoded leaves, or None if one cannot."""
    plan: list[tuple[str | None, Callable | None]] = []
    for evaluate in evaluators:
        owner = getattr(evaluate, "__self__", evaluate)
        scorer = getattr(owner, "from_encoded", None)
        if scorer is not None:
            plan.append((None, scorer))
            continue
        names = objective_names([evaluate])
        if names is None or names[0] not in PORTED_OBJECTIVES:
            return None
        plan.append((names[0], None))
    return plan


__all__ = [
    "PORTED_OBJECTIVES",
    "HeldLeaves",
    "PortRefused",
    "alternatives_encoded",
    "apply_lead_abilities",
    "ask",
    "batched_payoff",
    "batched_payoffs",
    "branch",
    "replacements_needed",
    "resolve_replacements",
    "resume",
    "resume_alternatives",
    "turn",
    "turn_leaves",
    "weights",
]
