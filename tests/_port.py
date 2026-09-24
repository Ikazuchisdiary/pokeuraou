"""The port, asked the questions the tests used to put to Python's resolver (IKA-210).

Python's `resolve_turn` and its neighbours are going away (IKA-204): the port is the only
engine. A rule test that built a position and asked `resolve_turn` what happens now asks
the port the same question through these functions, which keep Python's names and
shapes -- a `TurnResult`'s `branches` / `suspended` / `exact` / `unmodelled`, a pause that
`resume_turn` and `resume_alternatives` take back -- so a test's assertions read as they
did. Only the engine behind them changed.

What the port does not carry, these do not pretend to:

* **events and acts** (`Branch.events`, IKA-215). Reading one raises, so a test that
  asserts on the log fails loudly instead of passing on an empty list.
* `reductions` and `merged` -- the port reports `exact` only.

A refusal is a failure naming the port's reason, never a skip or a `None` for the test to
misread. There is no fallback: without a release binary every function here fails, and so
does the test that called it (IKA-210: the suite assumes the binary).

The process is shared for the session, one per regulation and binary.
`POKEURAOU_RUST_NODE_BIN` points it at another build -- which is how a positive control
runs the same tests against an older or deliberately broken port.
"""

from __future__ import annotations

import atexit
import contextlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import SideAction
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation, load_regulation

#: The port's budget, as `rustnode` sends it. Imported from here so no test names the
#: resolver's module for it (it moves with IKA-212).
Budget = rustnode.Budget

#: Set to 1 to ask `resolve_turn` of a binary built before the `turn` command, as a
#: positive control against an older port does (IKA-210). Never set by the suite itself.
ENV_BEFORE_TURN = "POKEURAOU_PORT_BEFORE_TURN"

#: The regulation for the helpers Python took without one (`replacements_needed`).
DEFAULT_FORMAT = "gen9championsvgc2026regmc"

_NODES: dict[tuple[str, str], rustnode.RustNode] = {}


def require_binary() -> None:
    """Fail -- not skip -- when there is no release binary: the port is the only engine."""
    if not rustnode.binary_path().exists():
        pytest.fail(
            f"no Rust binary at {rustnode.binary_path()}; `cargo build --release` "
            "(the port is the only engine the rule tests have, IKA-210)"
        )


def node(reg: Regulation) -> rustnode.RustNode:
    """The session's port process for `reg`."""
    require_binary()
    key = (reg.meta.format_id, str(rustnode.binary_path()))
    live = _NODES.get(key)
    if live is None or live._process.poll() is not None:  # noqa: SLF001
        live = rustnode.RustNode(reg)
        _NODES[key] = live
    return live


@atexit.register
def _close_all() -> None:
    for live in _NODES.values():
        # The interpreter is going away; a child that is already gone is not an error.
        with contextlib.suppress(Exception):
            live.close()
    _NODES.clear()


def given(pos: Position) -> Position:
    """A copy of ``pos`` as the port is given it (`_port_showdown.given`): the Showdown
    stats written onto an oracle position are dropped where neither engine reads them."""
    out = Position.from_json(pos.to_json())
    for side in out.sides:
        for mon in side.pokemon:
            if not mon.transformed and mon.sp is not None:
                mon.stats_override = None
    return out


class PortRefused(AssertionError):
    """The port declined the question; the message is its reason."""


def _ask(reg: Regulation, request: dict[str, Any], pos: Position | None = None) -> dict[str, Any]:
    if os.environ.get(ENV_BEFORE_TURN) == "1" and request.get("kind") != "resolve":
        # An older binary reads an unknown kind as a fill and can wait on the pipe for good.
        raise PortRefused(f"the binary predates the {request.get('kind')!r} command")
    reply = node(reg)._exchange(request)  # noqa: SLF001
    reason = reply.get("refused")
    if reason:
        if reason == "position carries unmodelled volatiles" and pos is not None:
            names = {v for s in pos.sides for m in s.pokemon for v in m.unmodelled_volatiles}
            reason = f"{reason}: {','.join(sorted(names))}"
        raise PortRefused(f"the port refused: {reason}")
    return reply


def _actions(actions: Sequence[SideAction]) -> list[list[dict[str, Any]]]:
    return [[rustnode.dump_action(a) for a in side.slots] for side in actions]


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


@dataclass
class Branch:
    """One finished outcome. `events` is not the port's (IKA-215)."""

    probability: float
    position: Position

    @property
    def events(self) -> list[str]:
        raise AssertionError("the port keeps no event log yet (IKA-215)")

    @property
    def acts(self) -> list[tuple[int, str]]:
        raise AssertionError("the port keeps no event log yet (IKA-215)")


@dataclass
class SuspendedTurn:
    """A turn the port stopped for a mid-turn replacement. `raw` is the port's own pause,
    handed back whole to resume it; `world` is `paused_in`'s completion, if any."""

    probability: float
    position: Position
    raw: dict[str, Any]
    reg: Regulation
    world: tuple[Position, int] | None = None

    @property
    def events(self) -> list[str]:
        raise AssertionError("the port keeps no event log yet (IKA-215)")


@dataclass
class TurnResult:
    branches: list[Branch]
    exact: bool
    unmodelled: tuple[str, ...] = ()
    suspended: tuple[SuspendedTurn, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

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


def _result(reg: Regulation, reply: dict[str, Any]) -> TurnResult:
    return TurnResult(
        branches=[
            Branch(float(b["probability"]), Position.from_json(b["position"]))
            for b in reply["branches"]
        ],
        exact=bool(reply["exact"]),
        unmodelled=tuple(reply["unmodelled"]),
        suspended=tuple(
            SuspendedTurn(float(p["probability"]), Position.from_json(p["position"]), p, reg)
            for p in reply["suspended"]
        ),
    )


def resolve_turn(
    reg: Regulation,
    pos: Position,
    side_actions: Sequence[SideAction],
    *,
    budget: Budget | None = None,
) -> TurnResult:
    """`resolve.resolve_turn`, answered by the port: every branch and every pause."""
    budget = budget or Budget.exact()
    if os.environ.get(ENV_BEFORE_TURN) == "1":
        return _resolve_before_turn(reg, pos, side_actions, budget)
    reply = _ask(
        reg,
        {
            "kind": "turn",
            "position": given(pos).to_json(),
            "actions": _actions(side_actions),
            "budget": rustnode.dump_budget(budget),
            "full": True,
        },
        pos,
    )
    return _result(reg, reply)


def _resolve_before_turn(
    reg: Regulation, pos: Position, side_actions: Sequence[SideAction], budget: Budget
) -> TurnResult:
    """`resolve_turn` from a binary older than the `turn` command (IKA-211), for a positive
    control only: the weights, then each branch by `select`. A pause has its weight but no
    position (`resolve` never sent one), so it comes back without one."""
    request = {
        "kind": "resolve",
        "position": given(pos).to_json(),
        "actions": _actions(side_actions),
        "budget": rustnode.dump_budget(budget),
        "select": None,
    }
    reply = _ask(reg, request, pos)
    branches = []
    for index, weight in enumerate(reply["branches"]):
        chosen = _ask(reg, {**request, "select": index}, pos)
        branches.append(Branch(float(weight), Position.from_json(chosen["position"])))
    return TurnResult(
        branches=branches,
        exact=bool(reply["exact"]),
        unmodelled=tuple(reply["unmodelled"]),
        suspended=tuple(
            SuspendedTurn(float(w), None, {}, reg)  # type: ignore[arg-type]
            for w in reply.get("suspended", [])
        ),
    )


def _world(world: tuple[Position, int] | None) -> dict[str, Any] | None:
    if world is None:
        return None
    position, side = world
    return {"position": given(position).to_json(), "side": int(side)}


def resume_turn(
    reg: Regulation, paused: SuspendedTurn, choices: Sequence[SideAction]
) -> TurnResult:
    """`resolve.resume_turn`: the rest of the paused turn with both sides' choices."""
    reply = _ask(
        reg,
        {
            "kind": "turn",
            "pause": paused.raw,
            "choices": _actions(choices),
            "full": True,
            "in": _world(paused.world),
        },
    )
    return _result(reg, reply)


def resume_alternatives(
    reg: Regulation, paused: SuspendedTurn
) -> tuple[int | None, list[tuple[SideAction, TurnResult]]]:
    """`resolve.resume_alternatives`: who chooses, and every option with its turn."""
    reply = _ask(
        reg, {"kind": "alternatives", "pause": paused.raw, "in": _world(paused.world), "full": True}
    )
    chooser = reply["chooser"]
    return (
        None if chooser is None else int(chooser),
        [
            (rustnode._side_action(option), _result(reg, result))  # noqa: SLF001
            for option, result in zip(reply["options"], reply["results"], strict=True)
        ],
    )


def paused_in(paused: SuspendedTurn, position: Position, side: int) -> SuspendedTurn:
    """`resolve.paused_in`: the same pause, resumed in another completion of it."""
    return SuspendedTurn(
        probability=paused.probability,
        position=position,
        raw=paused.raw,
        reg=paused.reg,
        world=(position, side),
    )


# ---------------------------------------------------------------------------
# The replacement phase and the leads
# ---------------------------------------------------------------------------


@dataclass
class ReplacementResult:
    position: Position
    unmodelled: tuple[str, ...] = ()

    @property
    def events(self) -> list[str]:
        raise AssertionError("the port keeps no event log yet (IKA-215)")


def _phase(
    reg: Regulation, request: dict[str, Any], rng: Any, pos: Position  # noqa: ANN401
) -> ReplacementResult:
    """As `RustNode._phase`: a draw is sampled from `rng` exactly as Python sampled it."""
    if rng is None:
        reply = _ask(reg, request, pos)
    else:
        presets: list[int] = []
        while True:
            reply = _ask(reg, {**request, "presets": presets}, pos)
            weights = reply.get("draw")
            if not weights:
                break
            total = float(sum(weights))
            presets.append(int(rng.choice(len(weights), p=[w / total for w in weights])))
    return ReplacementResult(
        position=Position.from_json(reply["position"]), unmodelled=tuple(reply["unmodelled"])
    )


def resolve_replacements(
    reg: Regulation,
    pos: Position,
    choices: Sequence[SideAction],
    *,
    rng: Any = None,  # noqa: ANN401 - numpy.random.Generator
) -> ReplacementResult:
    """`resolve.resolve_replacements`."""
    return _phase(
        reg,
        {"kind": "replacements", "position": given(pos).to_json(), "choices": _actions(choices)},
        rng,
        pos,
    )


def apply_lead_abilities(
    reg: Regulation, pos: Position, *, rng: Any = None  # noqa: ANN401
) -> ReplacementResult:
    """`resolve.apply_lead_abilities`."""
    return _phase(reg, {"kind": "leads", "position": given(pos).to_json()}, rng, pos)


def replacements_needed(
    pos: Position, reg: Regulation | None = None
) -> tuple[tuple[bool, ...], ...]:
    """`resolve.replacements_needed`, answered by the port."""
    reg = reg or load_regulation(DEFAULT_FORMAT)
    reply = _ask(reg, {"kind": "needed", "position": pos.to_json()})
    return tuple(tuple(bool(flag) for flag in side) for side in reply["needed"])


def self_switches_needed(pos: Position) -> tuple[tuple[bool, ...], ...]:
    """Per side, per active slot, whether the port left a self-switch waiting on a choice.

    Not a rule: it reads the `pendingselfswitch` flag the port wrote onto the position (and
    the bench it can be answered from), which is what `resolve.self_switches_needed` read.
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
