"""The port held to Showdown directly, for an oracle test's port twin (IKA-207).

An oracle test that checks Python's `resolve_turn` against Showdown gets a twin that asks
`RustNode.resolve` the same question and checks the answer against the same Showdown
state -- not against Python, so the twin survives Python's resolver (IKA-204).

The oracle cannot start from an arbitrary position, so the position is the one the test
already built by playing turns; `given` only drops the Showdown stats the oracle writes
onto every Pokemon, which the port declines and both engines ignore unless the Pokemon is
transformed or has no stat points (`view.py`, `battler.rs`).

The `port` fixture (conftest) is the process; it skips when there is no release binary.
"""

from __future__ import annotations

from typing import Any

from pokeuraou import rustnode
from pokeuraou.position import Position

from ._port import Budget


def given(pos: Position) -> Position:
    """A copy of ``pos`` as the port is given it."""
    out = Position.from_json(pos.to_json())
    for side in out.sides:
        for mon in side.pokemon:
            if not mon.transformed and mon.sp is not None:
                mon.stats_override = None
    return out


def _refuse(reply: dict[str, Any], pos: Position) -> None:
    """Fail on a refusal, naming the volatiles when that is the reason."""
    reason = reply.get("refused")
    if not reason:
        return
    if reason == "position carries unmodelled volatiles":
        names = {v for side in pos.sides for m in side.pokemon for v in m.unmodelled_volatiles}
        reason = f"{reason}: {','.join(sorted(names))}"
    raise AssertionError(f"the port refused the turn: {reason}")


def port_reply(
    node: rustnode.RustNode,
    pos: Position,
    actions: list[Any],
    budget: Budget,
    select: int | None = None,
) -> dict[str, Any]:
    """The port's raw answer; a refusal keeps its reason, which `RustNode.resolve` drops."""
    return node._exchange(  # noqa: SLF001
        {
            "kind": "resolve",
            "position": given(pos).to_json(),
            "actions": [[rustnode.dump_action(a) for a in side.slots] for side in actions],
            "budget": rustnode.dump_budget(budget),
            "select": select,
        }
    )


def port_weights(
    node: rustnode.RustNode, pos: Position, actions: list[Any], budget: Budget
) -> dict[str, Any]:
    """Branch weights, suspended weights and the unmodelled notes; fails on a refusal."""
    reply = port_reply(node, pos, actions, budget)
    _refuse(reply, pos)
    return reply


def port_branch(
    node: rustnode.RustNode,
    pos: Position,
    actions: list[Any],
    budget: Budget,
    index: int,
) -> Position:
    reply = port_reply(node, pos, actions, budget, select=index)
    _refuse(reply, pos)
    return Position.from_json(reply["position"])


def port_branches(
    node: rustnode.RustNode,
    pos: Position,
    actions: list[Any],
    budget: Budget,
) -> list[tuple[float, Position]]:
    """Every finished branch as (probability, position); fails on a refusal or a pause."""
    reply = port_weights(node, pos, actions, budget)
    assert not reply.get("suspended"), "the port stopped at a mid-turn replacement"
    return [
        (weight, port_branch(node, pos, actions, budget, index))
        for index, weight in enumerate(reply["branches"])
    ]


def port_turn(
    node: rustnode.RustNode,
    pos: Position,
    actions: list[Any],
    budget: Budget | None = None,
) -> Position:
    """The one finished outcome under ``budget`` (default: Showdown's pins, roll 0).

    Fails when the port refuses, stops at a mid-turn replacement, or branches: the twin
    of a Python test that took the single branch has to get a single branch too.
    """
    budget = budget or Budget.deterministic(0)
    reply = port_reply(node, pos, actions, budget, select=0)
    if reply.get("refused") == "branch index out of range":
        reply = port_reply(node, pos, actions, budget)
    _refuse(reply, pos)
    assert not reply.get("suspended"), "the port stopped at a mid-turn replacement"
    assert len(reply["branches"]) == 1, f"{len(reply['branches'])} branches"
    return Position.from_json(reply["position"])
