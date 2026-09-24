"""The port, asked the questions the tests used to put to Python's resolver (IKA-210).

Python's `resolve_turn` and its neighbours are going away (IKA-204): the port is the only
engine, and `pokeuraou.port` is the production roads' door to it (IKA-209). This is a thin
layer over that door. It keeps the names and shapes the rule tests were written against --
a `TurnResult`'s `branches` / `suspended` / `exact` / `unmodelled`, a pause that
`resume_turn` and `resume_alternatives` take back -- so a test's assertions read as they
did; the asking is `pokeuraou.port`'s (`port.ask`: one warm process, restarted on a broken
pipe, `PortUnavailable` without a binary).

* **events and acts** come from the port (IKA-215) when a turn is asked with
  `events=True`: `Branch.events` is then the port's trace, and a pause asked that way
  carries it into `resume_turn` / `resume_alternatives`. Off by default -- the trace
  costs 3-4% of the rule tests' time and most read none -- and reading one from a turn
  asked without raises rather than handing a test an empty list. Phases always carry it.
* `reductions` and `merged` are not carried -- the port reports `exact` only.

A refusal is `pokeuraou.port.PortRefused` naming the port's reason, never a skip or a `None`
for the test to misread. There is no fallback: without a release binary every function here
fails, and so does the test that called it (IKA-210: the suite assumes the binary).

`POKEURAOU_RUST_NODE_BIN` points the process at another build -- which is how a positive
control runs the same tests against an older or deliberately broken port.
`POKEURAOU_PORT_BEFORE_TURN=1` asks `resolve_turn` of a binary older than the `turn` command.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from pokeuraou import port, rustnode
from pokeuraou.actions import SideAction
from pokeuraou.budget import Budget
from pokeuraou.fold import TurnLeaves
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation, load_regulation

__all__ = [
    "Budget",
    "Branch",
    "PortRefused",
    "ReplacementResult",
    "SuspendedTurn",
    "TurnResult",
    "apply_lead_abilities",
    "given",
    "paused_in",
    "replacements_needed",
    "resolve_replacements",
    "resolve_turn",
    "resume_alternatives",
    "resume_turn",
    "self_switches_needed",
    "turn_expectation",
    "turn_leaves",
]

#: Set to 1 to ask `resolve_turn` of a binary built before the `turn` command, as a
#: positive control against an older port does (IKA-210). Never set by the suite itself.
ENV_BEFORE_TURN = "POKEURAOU_PORT_BEFORE_TURN"

#: The regulation for the helpers Python took without one (`replacements_needed`).
DEFAULT_FORMAT = "gen9championsvgc2026regmc"

#: The port's refusal. The same class the production roads raise.
PortRefused = port.PortRefused


def require_binary() -> None:
    """Fail -- not skip -- when there is no release binary: the port is the only engine."""
    if not rustnode.binary_path().exists():
        pytest.fail(
            f"no Rust binary at {rustnode.binary_path()}; `cargo build --release` "
            "(the port is the only engine the rule tests have, IKA-210)"
        )


def _before_turn(kind: str) -> None:
    if os.environ.get(ENV_BEFORE_TURN) == "1":
        # An older binary reads an unknown kind as a fill and can wait on the pipe for good.
        raise PortRefused(f"the binary predates the {kind!r} command")


def _asked[T](reg: Regulation, call: Callable[[rustnode.RustNode], T | None], what: str,
              pos: Position | None = None, *, retry: bool = True) -> T:
    """`port.ask`, with a `None` answer turned into the port's reason. Without `retry` (a
    phase drawing from a generator: a second try would draw twice) it asks once."""
    require_binary()

    def ask(node: rustnode.RustNode) -> T:
        answer = call(node)
        if answer is None:
            reason = node.refusal or "no reason given"
            if reason == "position carries unmodelled volatiles" and pos is not None:
                names = {v for s in pos.sides for m in s.pokemon for v in m.unmodelled_volatiles}
                reason = f"{reason}: {','.join(sorted(names))}"
            raise PortRefused(f"the port refused {what}: {reason}")
        return answer

    return port.ask(reg, ask) if retry else ask(rustnode.require_node(reg))


def given(pos: Position) -> Position:
    """A copy of ``pos`` as the port is given it (`_port_showdown.given`): the Showdown
    stats written onto an oracle position are dropped where neither engine reads them."""
    out = Position.from_json(pos.to_json())
    for side in out.sides:
        for mon in side.pokemon:
            if not mon.transformed and mon.sp is not None:
                mon.stats_override = None
    return out


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def _no_log(what: str) -> AssertionError:
    return AssertionError(f"this {what} was asked without events (pass events=True)")


@dataclass
class Branch:
    """One finished outcome, with the port's trace when it was asked for."""

    probability: float
    position: Position
    _events: list[str] | None = None
    _acts: list[tuple[int, str]] | None = None

    @property
    def events(self) -> list[str]:
        if self._events is None:
            raise _no_log("turn")
        return self._events

    @property
    def acts(self) -> list[tuple[int, str]]:
        if self._acts is None:
            raise _no_log("turn")
        return self._acts


@dataclass
class SuspendedTurn:
    """A turn the port stopped for a mid-turn replacement: the port's own pause, handed back
    whole to resume it; `world` is `paused_in`'s completion, if any."""

    probability: float
    position: Position
    pause: rustnode.PortPause | None
    reg: Regulation
    world: tuple[Position, int] | None = None
    logged: bool = True

    @property
    def raw(self) -> dict[str, Any]:
        return {} if self.pause is None else self.pause.raw

    @property
    def events(self) -> list[str]:
        if not self.logged or self.pause is None:
            raise _no_log("pause")
        return self.pause.events

    @property
    def acts(self) -> list[tuple[int, str]]:
        if not self.logged or self.pause is None:
            raise _no_log("pause")
        return self.pause.acts


@dataclass
class TurnResult:
    branches: list[Branch]
    exact: bool
    unmodelled: tuple[str, ...] = ()
    suspended: tuple[SuspendedTurn, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)
    #: The port's own answer, for `port.turn_leaves`. None from a pre-`turn` binary.
    port: rustnode.PortTurn | None = None

    @property
    def total_probability(self) -> float:
        return sum(b.probability for b in self.branches) + sum(
            s.probability for s in self.suspended
        )

    def expected(self, value: Callable[[Position], float]) -> float:
        total = self.total_probability
        if total <= 0:
            return 0.0
        if self.suspended:
            raise ValueError("expected() on a turn with suspended outcomes")
        return sum(b.probability * value(b.position) for b in self.branches) / total

    def collapse(self, key: Callable[[Position], object]) -> dict[object, float]:
        if self.suspended:
            raise ValueError("collapse() on a turn with suspended outcomes")
        out: dict[object, float] = {}
        for b in self.branches:
            k = key(b.position)
            out[k] = out.get(k, 0.0) + b.probability
        total = sum(out.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in out.items()), key=lambda kv: -kv[1]))


def _result(reg: Regulation, answer: rustnode.PortTurn, *, events: bool) -> TurnResult:
    return TurnResult(
        branches=[
            Branch(
                o.probability,
                o.position,
                list(o.events) if events else None,
                list(o.acts) if events else None,
            )
            for o in answer.outcomes or []
        ],
        exact=answer.exact,
        unmodelled=answer.unmodelled,
        suspended=tuple(
            SuspendedTurn(p.probability, p.position, p, reg, logged=events)
            for p in answer.pauses or []
        ),
        port=answer,
    )


def resolve_turn(
    reg: Regulation,
    pos: Position,
    side_actions: Sequence[SideAction],
    *,
    budget: Budget | None = None,
    events: bool = False,
) -> TurnResult:
    """`resolve.resolve_turn`, answered by the port: every branch and every pause."""
    budget = budget or Budget.exact()
    if os.environ.get(ENV_BEFORE_TURN) == "1":
        return _resolve_before_turn(reg, pos, side_actions, budget)
    shown = given(pos)
    answer = _asked(
        reg,
        lambda node: node.turn(shown, list(side_actions), budget, full=True, events=events),
        "a turn",
        pos,
    )
    return _result(reg, answer, events=events)


def _resolve_before_turn(
    reg: Regulation, pos: Position, side_actions: Sequence[SideAction], budget: Budget
) -> TurnResult:
    """`resolve_turn` from a binary older than the `turn` command (IKA-211), for a positive
    control only: the weights, then each branch by `select` (`RustNode.resolve`). A pause
    has its weight but no position (`resolve` never sent one), so it comes back without one."""
    shown = given(pos)
    weights = _asked(
        reg, lambda node: node.resolve(shown, list(side_actions), budget), "a turn", pos
    )
    branches = []
    for index, weight in enumerate(weights.branches):
        chosen = _asked(
            reg,
            lambda node, index=index: node.resolve(
                shown, list(side_actions), budget, select=index
            ),
            "a turn",
            pos,
        )
        branches.append(Branch(float(weight), chosen.position))
    return TurnResult(
        branches=branches,
        exact=weights.exact,
        unmodelled=weights.unmodelled,
        suspended=tuple(
            SuspendedTurn(float(w), None, None, reg)  # type: ignore[arg-type]
            for w in weights.suspended
        ),
    )


def _world(world: tuple[Position, int] | None) -> tuple[Position, int] | None:
    return None if world is None else (given(world[0]), int(world[1]))


def resume_turn(
    reg: Regulation, paused: SuspendedTurn, choices: Sequence[SideAction]
) -> TurnResult:
    """`resolve.resume_turn`: the rest of the paused turn with both sides' choices."""
    _before_turn("turn")
    answer = _asked(
        reg,
        lambda node: node.resume(
            paused.pause, list(choices), full=True, world=_world(paused.world),
            events=paused.logged,
        ),
        "a paused turn",
    )
    return _result(reg, answer, events=paused.logged)


def resume_alternatives(
    reg: Regulation, paused: SuspendedTurn
) -> tuple[int | None, list[tuple[SideAction, TurnResult]]]:
    """`resolve.resume_alternatives`: who chooses, and every option with its turn."""
    _before_turn("alternatives")
    chooser, options = _asked(
        reg,
        lambda node: node.resume_alternatives(
            paused.pause, world=_world(paused.world), full=True, events=paused.logged
        ),
        "a paused turn",
    )
    return chooser, [
        (option, _result(reg, resumed, events=paused.logged)) for option, resumed in options
    ]


def paused_in(paused: SuspendedTurn, position: Position, side: int) -> SuspendedTurn:
    """`resolve.paused_in`: the same pause, resumed in another completion of it."""
    return SuspendedTurn(
        probability=paused.probability,
        position=position,
        pause=paused.pause,
        reg=paused.reg,
        world=(position, side),
        logged=paused.logged,
    )


def turn_leaves(reg: Regulation, result: TurnResult) -> TurnLeaves:
    """`resolve.turn_leaves`: `port.turn_leaves` of the port's own answer."""
    if result.port is None:
        raise ValueError("turn_leaves needs a turn the `turn` command answered")
    return port.turn_leaves(reg, result.port)


def turn_expectation(
    reg: Regulation, result: TurnResult, value: Callable[[Position], float]
) -> tuple[float, tuple[str, ...]]:
    """`resolve.turn_expectation`: the turn's value under `value` (side 0's view), every
    mid-turn replacement chosen by the fold `port.turn_leaves` builds."""
    if not result.suspended:
        return result.expected(value), result.unmodelled
    plan = turn_leaves(reg, result)
    return plan.value([value(p) for p in plan.positions]), plan.unmodelled


# ---------------------------------------------------------------------------
# The replacement phase and the leads
# ---------------------------------------------------------------------------


@dataclass
class ReplacementResult:
    position: Position
    unmodelled: tuple[str, ...] = ()
    events: list[str] = field(default_factory=list)


def _phase_result(phase: rustnode.PortPhase) -> ReplacementResult:
    return ReplacementResult(
        position=phase.position, unmodelled=phase.unmodelled, events=list(phase.events)
    )


def resolve_replacements(
    reg: Regulation,
    pos: Position,
    choices: Sequence[SideAction],
    *,
    rng: Any = None,  # noqa: ANN401 - numpy.random.Generator
) -> ReplacementResult:
    """`resolve.resolve_replacements`, with its trace. A draw is sampled from `rng` exactly
    as Python sampled it (`RustNode._phase`)."""
    _before_turn("replacements")
    shown = given(pos)
    return _phase_result(
        _asked(
            reg,
            lambda node: node.resolve_replacements(shown, list(choices), rng=rng, events=True),
            "a replacement phase",
            pos,
            retry=rng is None,
        )
    )


def apply_lead_abilities(
    reg: Regulation, pos: Position, *, rng: Any = None  # noqa: ANN401
) -> ReplacementResult:
    """`resolve.apply_lead_abilities`, with its trace."""
    _before_turn("leads")
    shown = given(pos)
    return _phase_result(
        _asked(
            reg,
            lambda node: node.apply_lead_abilities(shown, rng=rng, events=True),
            "the leads",
            pos,
            retry=rng is None,
        )
    )


def replacements_needed(
    pos: Position, reg: Regulation | None = None
) -> tuple[tuple[bool, ...], ...]:
    """`resolve.replacements_needed`: `port.replacements_needed`."""
    _before_turn("needed")
    require_binary()
    return port.replacements_needed(reg or load_regulation(DEFAULT_FORMAT), pos)


#: `port.self_switches_needed`: the flag the port wrote, one copy for tests and tools (IKA-212).
self_switches_needed = port.self_switches_needed
