"""Differential test of the whole turn resolver -- the Rust port -- against Showdown.

For each turn: take the pre-turn position, run the port with every source of chance pinned
to the same outcome Showdown's policy forces, step Showdown, and compare the two resulting
positions field by field.

Under ``Budget.deterministic`` the port must produce exactly one branch, so this is an
equality test on states rather than a comparison of distributions. Divergences are counted
per field and attributed to the moves, abilities and items in play, so "what to fix next"
is a measurement.

``--self-switch`` aims the random chooser at U-turn and friends. Uniform play reaches an
interrupted turn on 4.6% of compared turns, which is enough to say the class is covered and
thin for putting a number on it -- and F1 was a question about exactly that class. At 0.9
the same 1,200 battles reach 14.6%, three times the sample for the same wall clock.

    uv run python tools/diff_turn.py --battles 40 --roll 8
    uv run python tools/diff_turn.py --battles 400 --self-switch 0.8
    uv run python tools/diff_turn.py --battles 400 --exes old=C:/tmp/old.exe,new

Until IKA-212 a Python column (Python's resolver) ran beside the port's; it went with the
resolver. A mid-turn replacement is drawn from the port's own pause (`answer_port_pauses`),
the draw Python's column used to take, so a seed plays the battles it played beside it.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import (  # noqa: E402
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.port import self_switches_needed  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import (  # noqa: E402
    MetagamePrior,
    find_cached_chaos,
    load_chaos,
    sample_team,
)
from pokeuraou.regulation import Regulation, load_regulation, to_id  # noqa: E402
from pokeuraou.speed import action_overriding_effects  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"

#: Position fields compared, and how much each matters. HP and fainted decide the game;
#: the rest shape the next turn.
COMPARED_FIELDS = (
    "hp",
    "fainted",
    "status",
    "boosts",
    "species",
    "item",
    "active",
    "side_conditions",
    "weather",
    "terrain",
    "pseudo_weather",
    "mega_used",
)


def canonical(pos: Position) -> dict[str, Any]:
    """The comparable projection of a position.

    Deliberately not everything: volatile bookkeeping and effect durations are compared
    only where the resolver claims to model them, and PP is left out because Showdown
    spends it in places the resolver does not yet reach (called moves, Pressure).
    """
    out: dict[str, Any] = {
        "weather": pos.field.weather,
        "terrain": pos.field.terrain,
        "pseudo_weather": sorted(p.id for p in pos.field.pseudo_weather),
    }
    for side_index, side in enumerate(pos.sides):
        prefix = f"p{side_index + 1}"
        out[f"{prefix}.active"] = list(side.active)
        out[f"{prefix}.side_conditions"] = sorted(c.id for c in side.side_conditions)
        out[f"{prefix}.mega_used"] = side.mega_used
        for mon in side.pokemon:
            key = f"{prefix}.{mon.slot}"
            out[f"{key}.hp"] = mon.hp
            out[f"{key}.fainted"] = mon.fainted
            # A fainted Pokemon's status is bookkeeping, not mechanics: Showdown writes
            # 'fnt' when it falls and clears it again when its replacement swaps in, so
            # comparing the field would report an artefact as a divergence.
            out[f"{key}.status"] = "fnt" if mon.fainted else mon.status
            out[f"{key}.boosts"] = dict(sorted(mon.boosts.items()))
            out[f"{key}.species"] = mon.species
            out[f"{key}.item"] = mon.item
    return out


def field_kind(key: str) -> str:
    return key.rsplit(".", 1)[-1]


@dataclass
class Report:
    """One run: a column per binary, and the turns no column was asked about.

    The per-turn counts live in each column (`PortReport`). Until IKA-212 the fields of this
    class were Python's column; it went with Python's resolver.
    """

    #: Turns no column was asked about (a replacement turn, a request Showdown made
    #: without both sides choosing).
    skipped: Counter[str] = field(default_factory=Counter)
    #: The port's columns, one per binary (IKA-207).
    ports: dict[str, PortReport] = field(default_factory=dict)
    #: Where the current battle came from, for a port example to be replayed.
    where: str = ""
    #: Every Showdown step of the current battle: a port that stopped at a replacement is
    #: carried on with the choices Showdown was given.
    showdown: StepTap | None = None

    @property
    def port(self) -> PortReport | None:
        """The first port column: the only one unless `--exes` named several."""
        return next(iter(self.ports.values()), None)

    def render(self) -> str:
        out: list[str] = []
        if self.skipped:
            out.append(
                "skipped before any column: "
                + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(8))
            )
        for label, port in self.ports.items():
            if out:
                out.append("")
            out.append(port.render(label))
        return "\n".join(out)


def describe_actions(reg: Regulation, chosen: list[SideAction]) -> str:
    return " | ".join(a.describe(reg) for a in chosen)


def self_switch_moves(reg: Regulation) -> frozenset[str]:
    """Move ids that send their own user out -- U-turn, Volt Switch, Parting Shot, Flip Turn.

    Read off the regulation rather than listed here: ``selfSwitch`` is the field the
    resolver itself branches on, so a hand-written list could disagree with the thing it is
    meant to be aiming at.
    """
    return frozenset(
        move_id for move_id, move in reg.moves.items() if move.raw.get("selfSwitch")
    )


def pick_action(
    reg: Regulation,
    pos: Position,
    side_index: int,
    py_rng: random.Random,
    wanted: frozenset[str],
    bias: float,
) -> SideAction:
    """One side's choice, with ``bias`` probability of preferring a self-switching move.

    Falls back to the uniform draw whenever no such move is available, so the bias changes
    which turns are reached and never which turns are legal.
    """
    options = side_actions(reg, pos, side_index)
    if bias <= 0 or py_rng.random() >= bias:
        return py_rng.choice(options)
    switching = [
        option
        for option in options
        if any(
            isinstance(slot, MoveAction) and slot.move_id in wanted
            for slot in option.slots
        )
    ]
    return py_rng.choice(switching or options)


#: The `skipped` reason every Showdown-driven diff tool counts a refused choice under: the
#: battle stops there and its remaining turns go uncompared, so it is named, not silent
#: (IKA-309 here; IKA-316 the other tools, which share `showdown_choice` below).
REFUSED_CHOICE = "battle stopped: Showdown refused a choice"


def showdown_choice(pick: SideAction, request: dict[str, Any] | None) -> str:
    """``pick`` as Showdown's choice string, its move numbers read off Showdown's request.

    `MoveAction.move_index` is the move's place among the Pokemon's four, which is what
    the port and the records read. Showdown numbers a choice by its place in the request
    instead, and those differ when a move locks: a charging move's second turn, Outrage's
    rampage and Hyper Beam's recharge offer that one move alone (`getMoves(lockedMove)`),
    so the charging move in slot 3 is `move 1` there, and `move 3` is refused with "doesn't
    have a move 3" (IKA-309). The menu already gives the lock no target (IKA-176); this
    gives it the request's number. A move the request does not list keeps its own.
    """
    if not request or not request.get("active"):
        return pick.to_choice()
    parts: list[str] = []
    for action in pick.slots:
        if isinstance(action, MoveAction) and action.slot < len(request["active"]):
            listed = [m.get("id") for m in (request["active"][action.slot] or {}).get("moves", [])]
            if action.move_id in listed:
                action = replace(action, move_index=listed.index(action.move_id) + 1)
        parts.append(action.to_choice())
    return ", ".join(parts)


def showdown_paused_mid_turn(handle: Any) -> bool:
    """Whether Showdown stopped inside the turn rather than at the end of it.

    A mid-turn interrupt and an end-of-turn faint replacement both present as a
    ``forceSwitch`` request with no ``|turn|`` line. What separates them is the residual
    phase: it runs before the end-of-turn request and is behind the mid-turn one, so
    ``|upkeep`` is present in the second case and absent in the first.
    """
    if not any(r and r.get("forceSwitch") for r in handle.requests):
        return False
    return not any(line.startswith("|upkeep") for line in handle.log)


def compare_turn(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    handle: Any,
    lines: list[str],
    roll: int,
    report: Report,
    py_rng: random.Random,
    nodes: dict[str, Any],
) -> None:
    """One port column per binary in ``nodes``, each held to the position Showdown reached
    at the end of the choices it was given; a mid-turn replacement is then drawn from the
    port's pause (`answer_port_pauses`) and each column follows the step."""
    given = before.to_json()
    for label, node in nodes.items():
        compare_port_turn(
            reg, node, report.ports[label], report.where, given, chosen, handle, lines, roll
        )
    for label in nodes:
        pending = report.ports[label].pending
        if pending is not None and report.showdown is not None:
            pending.cursor = len(report.showdown.steps)
    answer_port_pauses(reg, handle, report, nodes, py_rng)


# ---------------------------------------------------------------------------
# The port's column (IKA-207): the same turn, the same pins, `RustNode.resolve`.


@dataclass
class PortReport:
    """The port held to Showdown, turn by turn, one column per binary.

    A refusal is not a divergence and not a match: it is counted by its reason, because
    until the port answers everything (IKA-208) "how often does it decline, and why" is
    half of what this column is for. A turn both the port and Showdown stopped inside, at
    a mid-turn replacement, is carried on: the port's pause is resumed (`turn` from a
    pause, IKA-211) with the replacements Showdown was given, as many times as the turn
    stops, and the end of the turn is compared (IKA-217).
    """

    binary: dict[str, Any] = field(default_factory=dict)
    compared: int = 0
    matched: int = 0
    flagged: int = 0
    silent: int = 0
    #: Stops at a mid-turn replacement shared with Showdown, and how the state there
    #: compared.
    paused: int = 0
    paused_matched: int = 0
    #: Compared turns that went through at least one resume, and those that diverged at
    #: the end of the turn.
    continued: int = 0
    paused_divergences: int = 0
    paused_by_field: Counter[str] = field(default_factory=Counter)
    #: The turn being carried on, between Showdown's steps.
    pending: PortPending | None = None
    by_field: Counter[str] = field(default_factory=Counter)
    attributed: Counter[tuple[str, ...]] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    refused: Counter[str] = field(default_factory=Counter)
    unmodelled: Counter[str] = field(default_factory=Counter)
    branch_counts: Counter[int] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    #: Every divergent turn, with Showdown's log, for `--port-json`.
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    @property
    def silent_rate(self) -> float:
        return 0.0 if not self.compared else self.silent / self.compared

    def render(self, label: str = "port") -> str:
        sha = self.binary.get("sha256", "?")
        out = [
            f"{label} ({sha}, built {self.binary.get('built', '?')}): "
            f"compared {self.compared} turns, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}% "
            f"(silent {self.silent_rate * 100:.3f}%, flagged {self.flagged})"
        ]
        refused = sum(self.refused.values())
        out.append(f"  refused {refused} turns")
        for reason, count in self.refused.most_common(12):
            out.append(f"    {count:5d}  {reason}")
        if self.paused:
            out.append(
                f"  mid-turn replacement requests: {self.paused}, "
                f"state at the interrupt matched {self.paused_matched}, "
                f"turns carried on and compared {self.continued}, "
                f"diverging after resuming {self.paused_divergences}"
            )
            if self.paused_by_field:
                out.append(
                    "    on interrupted turns: "
                    + ", ".join(f"{k} x{v}" for k, v in self.paused_by_field.most_common(8))
                )
        if self.branch_counts:
            out.append(
                "  branches produced: "
                + ", ".join(f"{k}x{v}" for k, v in sorted(self.branch_counts.items()))
            )
        if self.by_field:
            out.append(
                "  divergent fields: "
                + ", ".join(f"{k} x{v}" for k, v in self.by_field.most_common(12))
            )
        if self.attributed:
            out.append("  attributed (field / move / ability / item):")
            for key, count in self.attributed.most_common(18):
                out.append(f"    {count:5d}  {' / '.join(k or '-' for k in key)}")
        if self.skipped:
            out.append(
                "  skipped: " + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(8))
            )
        if self.unmodelled:
            out.append("  port reported unmodelled:")
            for name, count in self.unmodelled.most_common(15):
                out.append(f"    {count:5d}  {name}")
        for line in self.examples[:10]:
            out.append("  " + line)
        return "\n".join(out)


def port_position(raw: dict[str, Any]) -> Position:
    """Showdown's position as the port is given it.

    The oracle writes Showdown's own stats onto every Pokemon, and the port declines a
    position that carries them. Both engines read `stats_override` only for a transformed
    Pokemon or one with no stat points (`view.py`, `battler.rs`), so dropping it anywhere
    else changes no number -- which is what the port tests do too.
    """
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            if not mon.transformed and mon.sp is not None:
                mon.stats_override = None
    return pos


def port_exchange(
    node: Any,
    pos: Position,
    actions: list[SideAction],
    budget: Budget,
    select: int | None = None,
) -> dict[str, Any]:
    """`RustNode.resolve`'s request, returning the reply itself so a refusal keeps its reason."""
    from pokeuraou import rustnode

    return node._exchange(  # noqa: SLF001 -- `resolve` drops the reason; this tool counts it
        {
            "kind": "resolve",
            "position": pos.to_json(),
            "actions": [[rustnode.dump_action(a) for a in side.slots] for side in actions],
            "budget": rustnode.dump_budget(budget),
            "select": select,
        }
    )


def refusal_reason(reply: dict[str, Any], before: Position) -> str:
    """The port's reason, with the volatiles named when it is "unmodelled volatiles".

    The port says only that there are some; which ones is what the fix needs, and the
    position that was sent has them.
    """
    reason = str(reply["refused"])
    if reason == "position carries unmodelled volatiles":
        names = sorted(
            {v for side in before.sides for m in side.pokemon for v in m.unmodelled_volatiles}
        )
        reason += ": " + ",".join(names)
    return reason


def in_play(before: Position, chosen: list[SideAction]) -> tuple[str, str, str]:
    """The moves, abilities and items a divergence is attributed to."""
    moves = sorted(s.move_id for a in chosen for s in a.slots if isinstance(s, MoveAction))
    active = [m for side in before.sides for m in side.active_pokemon() if m is not None]
    return (
        ",".join(moves),
        ",".join(sorted({m.ability for m in active})),
        ",".join(sorted({m.item or "" for m in active})),
    )


def compare_port_turn(
    reg: Regulation,
    node: Any,
    port: PortReport,
    where: str,
    given: dict[str, Any],
    chosen: list[SideAction],
    handle: Any,
    lines: list[str],
    roll: int,
) -> str:
    """One turn through the port; returns its verdict (match, diverge, refused, pending, skip).

    ``pending``: the port stopped with Showdown at a replacement; `follow_port` carries it
    on and gives the verdict once the turn ends.

    ``given`` is the position before the turn; ``handle`` is Showdown after it, still
    standing where the choices left it.
    A turn where the port and Showdown disagree about stopping for a mid-turn replacement
    is a compared, divergent turn with the field ``mid-turn interrupt``.

    One request a turn: a deterministic budget leaves one outcome, so branch 0 is asked
    for together with the weights. Only when there is no finished branch -- the turn
    stopped at a replacement -- does the port refuse the index, and only then is it asked
    again for the weights alone.
    """
    settle_port(port, "a new turn began inside a paused one")
    if action_overriding_effects(lines, only_unmodelled=True):
        port.skipped["action overridden mid-turn"] += 1
        return "skip"
    before = port_position(given)
    budget = Budget.deterministic(roll)
    reply = port_exchange(node, before, chosen, budget, select=0)
    if reply.get("refused") == "branch index out of range":
        # No finished branch: the turn stopped. `turn` hands the pause itself back with
        # the weights, so it can be carried on -- still two requests, as before (IKA-217).
        reply = port_turn_exchange(
            node,
            {
                "position": before.to_json(),
                "actions": _dumped(chosen),
                "budget": _dumped_budget(budget),
            },
        )
    if reply.get("refused"):
        port.refused[refusal_reason(reply, before)] += 1
        return "refused"
    weights = list(reply.get("branches", []))
    stopped = list(reply.get("suspended", []))
    port.branch_counts[len(weights) + len(stopped)] += 1
    unmodelled = tuple(reply.get("unmodelled", []))
    for name in unmodelled:
        port.unmodelled[name] += 1
    if len(weights) + len(stopped) != 1:
        port.skipped[
            f"{len(weights) + len(stopped)} branches under a deterministic budget"
        ] += 1
        return "skip"

    showdown_stopped = showdown_paused_mid_turn(handle)
    theirs = canonical(Position.from_json(handle.position))
    if stopped and showdown_stopped:
        port.paused += 1
        _port_at_interrupt(port, reply["pause"], handle.position)
        port.pending = PortPending(
            where=where,
            before=before,
            chosen=chosen,
            pause=reply["pause"],
            unmodelled=set(unmodelled),
            log=list(handle.log),
        )
        return "pending"
    if bool(stopped) != showdown_stopped:
        differences = {
            "mid-turn interrupt": (
                "port stopped" if stopped else "port carried on",
                "showdown stopped" if showdown_stopped else "showdown carried on",
            )
        }
        _port_divergence(reg, port, where, before, chosen, unmodelled, differences, handle.log)
        return "diverge"
    if any(name.startswith("forceSwitch") for name in unmodelled):
        port.skipped["pending replacement (resolver defers to a choice)"] += 1
        return "skip"

    ours = canonical(Position.from_json(reply["position"]))
    keys = [k for k in ours if ours[k] != theirs.get(k)]
    if not keys:
        port.compared += 1
        port.matched += 1
        return "match"
    differences = {k: (ours[k], theirs.get(k)) for k in keys}
    _port_divergence(reg, port, where, before, chosen, unmodelled, differences, handle.log)
    return "diverge"


def _port_divergence(
    reg: Regulation,
    port: PortReport,
    where: str,
    before: Position,
    chosen: list[SideAction],
    unmodelled: tuple[str, ...],
    differences: dict[str, tuple[Any, Any]],
    log: list[str],
) -> None:
    port.compared += 1
    if unmodelled:
        port.flagged += 1
    else:
        port.silent += 1
    cast = in_play(before, chosen)
    for key in differences:
        kind = field_kind(key)
        port.by_field[kind] += 1
        port.attributed[(kind, *cast)] += 1
    detail = "; ".join(
        f"{k}: port {ours!r} vs showdown {theirs!r}"
        for k, (ours, theirs) in list(differences.items())[:5]
    )
    if not unmodelled and len(port.examples) < 10:
        port.examples.append(
            f"{where} turn {before.turn} [{describe_actions(reg, chosen)}] -> {detail}"
        )
    port.records.append(
        {
            "where": where,
            "turn": before.turn,
            "actions": describe_actions(reg, chosen),
            "cast": list(cast),
            "unmodelled": list(unmodelled),
            "differences": {k: [repr(a), repr(b)] for k, (a, b) in differences.items()},
            "log": list(log),
        }
    )


# ---------------------------------------------------------------------------
# Carrying a paused turn on in the port (IKA-217): Showdown's own replacements, one resume
# per stop, the end of the turn compared as any other.


@dataclass
class ShowdownStep:
    """One `handle.step`: the requests it answered, the choices, and what came back."""

    requests: list[dict[str, Any] | None]
    choices: list[str | None]
    after: dict[str, Any]


class StepTap:
    """Records every step of one battle, whoever takes it.

    `answer_port_pauses` steps Showdown through the pauses, and a turn it did not carry
    through is finished by the run's next "default" step. Each column follows whichever it
    was, so no column steps Showdown itself and one column's pause never moves another's.
    """

    def __init__(self, handle: Any) -> None:  # noqa: ANN401 - oracle.BattleHandle
        self.steps: list[ShowdownStep] = []
        inner = handle.step

        def step(choices: Any) -> Any:  # noqa: ANN401
            requests = list(handle.requests)
            out = inner(choices)
            self.steps.append(ShowdownStep(requests, list(choices), handle.last))
            return out

        handle.step = step


@dataclass
class PortPending:
    """A turn the port stopped inside with Showdown, waiting for Showdown's next step."""

    where: str
    before: Position
    chosen: list[SideAction]
    pause: dict[str, Any]
    unmodelled: set[str]
    log: list[str]
    #: Index of the next Showdown step to answer the pause with.
    cursor: int = 0
    #: The end of the turn as `_resume_port` compared it -- (ours, theirs, unmodelled) --
    #: once it matched or diverged; `diverge_report` ranks from it (IKA-227).
    decided: tuple[dict[str, Any], dict[str, Any], tuple[str, ...]] | None = None


def _dumped(actions: list[SideAction]) -> list[list[dict[str, Any]]]:
    from pokeuraou import rustnode

    return [[rustnode.dump_action(a) for a in side.slots] for side in actions]


def _dumped_budget(budget: Budget) -> dict[str, Any]:
    from pokeuraou import rustnode

    return rustnode.dump_budget(budget)


def port_turn_exchange(node: Any, request: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
    """The `turn` command with `select` 0 (`RustNode.turn`/`resume`), keeping the refusal's
    reason. A deterministic budget leaves one outcome, so index 0 is it, branch or pause."""
    return node._exchange({"kind": "turn", "select": 0, **request})  # noqa: SLF001


def _port_at_interrupt(port: PortReport, pause: dict[str, Any], showdown: dict[str, Any]) -> None:
    """The state at the stop, compared as the end of a turn is."""
    ours = canonical(Position.from_json(pause["position"]))
    theirs = canonical(Position.from_json(showdown))
    keys = [k for k in ours if ours[k] != theirs.get(k)]
    if keys:
        port.by_field["at the interrupt"] += len(keys)
    else:
        port.paused_matched += 1


def showdown_switch_ins(
    flags: list[bool], team: list[dict[str, Any]], choice: str
) -> dict[int, int] | str:
    """Active slot -> index into the request's team, as Showdown reads ``choice``.

    `Side.choose` reads the parts in order: a `switch N` goes to the next slot that owes a
    switch (`getChoiceIndex` passes over the others), a `pass` takes the next slot, and
    `default` (`autoChoose` -> `chooseSwitch()`) gives every remaining owed slot the first
    Pokemon after the actives that is neither fainted nor already chosen.
    """
    width = len(flags)
    out: dict[int, int] = {}
    index = 0
    for part in (p.strip() for p in choice.split(",") if p.strip()):
        if part == "default":
            for slot in range(index, width):
                if not flags[slot]:
                    continue
                k = width
                while k < len(team) and (
                    k in out.values() or str(team[k].get("condition", "")).endswith(" fnt")
                ):
                    k += 1
                if k < len(team):
                    out[slot] = k
            index = width
        elif part == "pass":
            index += 1
        elif part.startswith("switch "):
            while index < width and not flags[index]:
                index += 1
            if index >= width:
                return "a replacement choice with no slot owing one"
            out[index] = int(part.split()[1]) - 1
            index += 1
        else:
            return f"unread replacement choice {part!r}"
    return out


def showdown_picks(pause: Position, step: ShowdownStep) -> list[SideAction] | str:
    """The replacements Showdown was given in ``step``, as actions on the port's pause.

    Read off the request (which Pokemon stood at the chosen index) rather than off the
    port's own numbering, so the port is carried on with Showdown's choice even where the
    two number the party differently. A reason instead when it cannot be.
    """
    picks: list[SideAction] = []
    for side_index, side in enumerate(pause.sides):
        request = step.requests[side_index] or {}
        width = len(side.active)
        flags = [bool(f) for f in (request.get("forceSwitch") or [])][:width]
        flags += [False] * (width - len(flags))
        if request.get("wait") or not any(flags):
            picks.append(SideAction(slots=tuple(PassAction(slot=i) for i in range(width))))
            continue
        team = list(request["side"]["pokemon"])
        chosen = showdown_switch_ins(flags, team, step.choices[side_index] or "default")
        if isinstance(chosen, str):
            return chosen
        slots: list[Any] = []
        for slot in range(width):
            index = chosen.get(slot)
            if index is None:
                slots.append(PassAction(slot=slot))
                continue
            species = to_id(str(team[index]["details"]).split(",")[0])
            mon = next(
                (
                    m
                    for m in side.pokemon
                    if not m.is_active and not m.fainted and species in (m.species, m.base_species)
                ),
                None,
            )
            if mon is None:
                return "Showdown's replacement is not on the port's bench"
            slots.append(SwitchAction(slot=slot, party_index=mon.slot + 1, species=mon.species))
        picks.append(SideAction(slots=tuple(slots)))
    return picks


def answer_port_pauses(
    reg: Regulation,
    handle: Any,  # noqa: ANN401 - oracle.BattleHandle
    report: Report,
    nodes: dict[str, Any],
    py_rng: random.Random,
) -> None:
    """The replacement draws Python's column used to take: at each mid-turn stop Showdown
    and a port column share, one `py_rng.choice` per side that owes one, from the first
    pending column's pause. So a seed plays the battles it played beside Python (IKA-210),
    wherever the two pauses agree; every column then follows the step.
    """
    for _ in range(4):
        pending = next(
            (report.ports[label].pending for label in nodes if report.ports[label].pending),
            None,
        )
        if pending is None or not showdown_paused_mid_turn(handle):
            return
        paused_at = Position.from_json(pending.pause["position"])
        owed = self_switches_needed(paused_at)
        told: list[str | None] = []
        for side_index in range(2):
            request = handle.requests[side_index]
            if not request or request.get("wait") or not request.get("forceSwitch"):
                told.append(None)
                continue
            options = switch_actions_after_faint(
                reg, paused_at, side_index, list(owed[side_index])
            )
            told.append(py_rng.choice(options).to_choice())
        handle.step(told)
        for label, node in nodes.items():
            follow_port(reg, node, report.ports[label], report.showdown)
        if handle.choice_errors:
            report.skipped["Showdown refused a mid-turn replacement"] += 1
            return


def follow_port(reg: Regulation, node: Any, port: PortReport, tap: StepTap | None) -> None:  # noqa: ANN401
    """Answers the port's pause with each Showdown step not yet answered, until the turn
    ends or the steps run out (the rest come on the run's next step)."""
    pending = port.pending
    if pending is None or tap is None:
        return
    while pending.cursor < len(tap.steps):
        step = tap.steps[pending.cursor]
        pending.cursor += 1
        verdict = _resume_port(reg, node, port, pending, step)
        if verdict is not None:
            port.pending = None
            return


def settle_port(port: PortReport, reason: str) -> None:
    """A battle that ended with the port's pause still unanswered: named, not compared."""
    if port.pending is None:
        return
    port.skipped[reason] += 1
    port.pending = None


def _resume_port(
    reg: Regulation, node: Any, port: PortReport, pending: PortPending, step: ShowdownStep  # noqa: ANN401
) -> str | None:
    """One resume; a verdict when the turn is decided, None when it stopped again."""
    after = step.after
    if after.get("choiceErrors"):
        port.skipped["replacement rejected by Showdown"] += 1
        return "skip"
    if not any((r or {}).get("forceSwitch") for r in step.requests):
        port.skipped["Showdown moved on without a replacement"] += 1
        return "skip"
    paused_at = Position.from_json(pending.pause["position"])
    owed = self_switches_needed(paused_at)
    theirs_owed = [
        [bool(x) for x in ((r or {}).get("forceSwitch") or [])] for r in step.requests
    ]
    for side_index, flags in enumerate(owed):
        want = theirs_owed[side_index][: len(flags)]
        if want and list(flags) != want:
            differences = {"mid-turn interrupt": (f"port owes {list(flags)}", f"showdown {want}")}
            pending.decided = _decided(differences, tuple(sorted(pending.unmodelled)))
            _port_divergence(
                reg, port, pending.where, pending.before, pending.chosen,
                tuple(sorted(pending.unmodelled)), differences, pending.log,
            )
            return "diverge"
    picks = showdown_picks(paused_at, step)
    if isinstance(picks, str):
        port.skipped[picks] += 1
        return "skip"
    reply = port_turn_exchange(node, {"pause": pending.pause, "choices": _dumped(picks)})
    if reply.get("refused"):
        port.refused[f"resume: {reply['refused']}"] += 1
        return "refused"
    pending.log += list(after["log"])
    pending.unmodelled.update(reply.get("unmodelled", []))
    unmodelled = tuple(sorted(pending.unmodelled))
    outcomes = len(reply.get("branches", [])) + len(reply.get("suspended", []))
    if outcomes != 1:
        port.skipped["resuming produced more than one branch"] += 1
        return "skip"
    stopped = "pause" in reply
    showdown_stopped = showdown_paused_mid_turn(
        _Stepped(requests=after["requests"], log=after["log"])
    )
    if stopped and showdown_stopped:
        port.paused += 1
        _port_at_interrupt(port, reply["pause"], after["position"])
        pending.pause = reply["pause"]
        return None
    if stopped != showdown_stopped:
        differences = {
            "mid-turn interrupt": (
                "port stopped" if stopped else "port carried on",
                "showdown stopped" if showdown_stopped else "showdown carried on",
            )
        }
        pending.decided = _decided(differences, unmodelled)
        _port_divergence(
            reg, port, pending.where, pending.before, pending.chosen, unmodelled,
            differences, pending.log,
        )
        return "diverge"
    if any(name.startswith("forceSwitch") for name in unmodelled):
        port.skipped["pending replacement (resolver defers to a choice)"] += 1
        return "skip"
    port.continued += 1
    ours = canonical(Position.from_json(reply["position"]))
    theirs = canonical(Position.from_json(after["position"]))
    pending.decided = (ours, theirs, unmodelled)
    keys = [k for k in ours if ours[k] != theirs.get(k)]
    if not keys:
        port.compared += 1
        port.matched += 1
        return "match"
    port.paused_divergences += 1
    for key in keys:
        port.paused_by_field[field_kind(key)] += 1
    _port_divergence(
        reg, port, pending.where, pending.before, pending.chosen, unmodelled,
        {k: (ours[k], theirs.get(k)) for k in keys}, pending.log,
    )
    return "diverge"


def _decided(
    differences: dict[str, tuple[Any, Any]], unmodelled: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """A disagreement about stopping, as the (ours, theirs) pair `PortPending.decided` holds."""
    ours = {k: a for k, (a, _) in differences.items()}
    theirs = {k: b for k, (_, b) in differences.items()}
    return ours, theirs, unmodelled


@dataclass
class _Stepped:
    """What `showdown_paused_mid_turn` reads, for a step other than the handle's last."""

    requests: list[dict[str, Any] | None]
    log: list[str]


def parse_exes(text: str) -> list[tuple[str, Path]]:
    """`--exes old=C:/x.exe,new` -> labelled binaries; a bare `new`/`current` is this tree's.

    Named as `tools/diff_node.py --exes` named them (IKA-206; that tool went with Python's
    resolver in IKA-212).
    """
    from pokeuraou import rustnode

    out: list[tuple[str, Path]] = []
    for index, entry in enumerate(e.strip() for e in text.split(",") if e.strip()):
        label, _, path = entry.partition("=")
        if not path:
            label, path = (entry, "") if entry in ("new", "current") else (f"exe{index}", entry)
        out.append((label, Path(path) if path else rustnode.binary_path()))
    return out


def fingerprint(path: Path) -> dict[str, Any]:
    """What a binary is, so a column means the code that ran (`rustnode.binary_fingerprint`)."""
    import hashlib
    from datetime import datetime

    if not path.exists():
        raise SystemExit(f"no Rust binary at {path}; `cd rust && cargo build --release`")
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()[:16],
        "built": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def open_ports(
    reg: Regulation, report: Report, exes: list[tuple[str, Path]] | None
) -> dict[str, Any]:
    """One warm process per binary for the whole run.

    With no list, this tree's binary, refused if it is older than the sources: a column
    against a stale build says nothing about the code on disk.
    """
    from pokeuraou import rustnode

    if not exes:
        report.ports["port"] = PortReport(binary=rustnode.require_current_binary())
        return {"port": rustnode.RustNode(reg)}
    nodes: dict[str, Any] = {}
    for label, path in exes:
        report.ports[label] = PortReport(binary=fingerprint(path))
        nodes[label] = rustnode.RustNode(reg, binary=path)
    return nodes


def hash_free_prior(reg: Regulation, prior: MetagamePrior) -> MetagamePrior:
    """``prior`` with every ability table in an order that does not depend on the hash seed.

    `load_chaos` fills the table of a species seen only as its mega forme with a uniform
    distribution built by iterating a *set* of ability ids, so the table's order -- and so
    which ability `weighted_choice` draws for the same generator state -- changed with
    PYTHONHASHSEED (18 species in the cached usage file, Delphox and Mawile the commonest).
    The same `--seed` then drew other teams and played other turns (IKA-207: 2,726 and
    2,727 compared turns; IKA-218).
    Those tables (the species `prior.abilities_unobserved` names) are put in the dex's order
    of the species' abilities; the probabilities are untouched, and every other table keeps
    the usage file's order, which never depended on the hash seed. Done here rather than in
    `load_chaos` because generation reads the same function (IKA-218's record).
    """
    unobserved = set(prior.abilities_unobserved)
    for entry in prior.species.values():
        if entry.name not in unobserved:
            continue
        order = [to_id(a) for a in reg.species[entry.species_id].abilities]
        entry.abilities = {a: entry.abilities[a] for a in order if a in entry.abilities}
    return prior


def run(
    battles: int,
    roll: int,
    seed: int,
    max_turns: int,
    quiet: bool = True,
    self_switch: float = 0.0,
    exes: list[tuple[str, Path]] | None = None,
) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = hash_free_prior(reg, load_chaos(chaos, reg))
    register_mega_stones(reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    wanted = self_switch_moves(reg)
    report = Report()
    policy = RandomnessPolicy(
        damage_roll=roll,
        accuracy="hit",
        crit=False,
        secondary=False,
        multihit="min",
        speed_tie="keep",
    )
    nodes = open_ports(reg, report, exes)

    with Oracle() as oracle:
        for battle in range(battles):
            report.where = f"seed {seed} battle {battle}"
            teams = [
                [TeamSet.from_json(s.to_team_set_json(reg)) for s in sample_team(rng, reg, prior)]
                for _ in range(2)
            ]
            handle = oracle.create(
                FORMAT_ID,
                teams[0],
                teams[1],
                seed=tuple(int(rng.integers(1, 60000)) for _ in range(4)),  # type: ignore[arg-type]
                policy=policy,
            )
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])
            report.showdown = StepTap(handle)

            for _ in range(max_turns):
                before = Position.from_json(handle.position)
                if before.ended:
                    break
                chosen: list[SideAction] = []
                choices: list[str | None] = []
                forced = False
                for side_index in range(2):
                    request = handle.requests[side_index]
                    if request and request.get("forceSwitch"):
                        forced = True
                        choices.append("default")
                        continue
                    if not request or request.get("wait"):
                        choices.append(None)
                        continue
                    pick = pick_action(
                        reg, before, side_index, py_rng, wanted, self_switch
                    )
                    chosen.append(pick)
                    choices.append(showdown_choice(pick, request))
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                for label, node in nodes.items():
                    # A turn left at the stop goes on with this step.
                    follow_port(reg, node, report.ports[label], report.showdown)
                if handle.choice_errors:
                    # Named, not silent: the rest of the battle goes uncompared (IKA-309).
                    report.skipped[REFUSED_CHOICE] += 1
                    break
                if forced or len(chosen) != 2:
                    report.skipped["replacement turn"] += 1
                    continue
                compare_turn(
                    reg, before, chosen, handle, handle.log, roll, report, py_rng, nodes
                )
            for port_column in report.ports.values():
                settle_port(port_column, "the battle stopped inside a paused turn")
            handle.close()
    for node in nodes.values():
        node.close()

    if not quiet:
        print(report.render())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=25)
    ap.add_argument("--roll", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument(
        "--self-switch",
        type=float,
        default=0.0,
        help="probability of preferring a self-switching move when one is legal, so "
        "interrupted turns are sampled on purpose (0 = uniform play)",
    )
    ap.add_argument(
        "--port-json",
        help="write every turn the port diverged on, with Showdown's log, to this file",
    )
    ap.add_argument(
        "--exes",
        help="several port columns in one run: `old=C:/tmp/old.exe,new` (a bare `new` is "
        "this tree's rust/target/release; POKEURAOU_RUST_NODE_BIN names the default one)",
    )
    args = ap.parse_args()
    # No PYTHONHASHSEED is needed: the hash-ordered ability tables that made the same seed
    # play other turns (IKA-207) are put in the dex's order by `hash_free_prior` (IKA-218).
    report = run(
        args.battles,
        args.roll,
        args.seed,
        args.max_turns,
        quiet=False,
        self_switch=args.self_switch,
        exes=parse_exes(args.exes) if args.exes else None,
    )
    if args.port_json and report.ports:
        import json

        Path(args.port_json).write_bytes(
            json.dumps(
                {
                    label: {"binary": port.binary, "divergences": port.records}
                    for label, port in report.ports.items()
                },
                indent=1,
            ).encode("utf-8")
        )


if __name__ == "__main__":
    main()
