"""Differential test of the whole turn resolver against Showdown.

For each turn: take the pre-turn position, run our resolver with every source of chance
pinned to the same outcome Showdown's policy forces, step Showdown, and compare the two
resulting positions field by field.

Under ``Budget.deterministic`` our resolver must produce exactly one branch, so this is an
equality test on states rather than a comparison of distributions. Divergences are counted
per field and attributed to the moves, abilities and items in play, so "what to fix next"
is a measurement.

``--self-switch`` aims the random chooser at U-turn and friends. Uniform play reaches an
interrupted turn on 4.6% of compared turns, which is enough to say the class is covered and
thin for putting a number on it -- and F1 was a question about exactly that class. At 0.9
the same 1,200 battles reach 14.6%, three times the sample for the same wall clock.

    uv run python tools/diff_turn.py --battles 40 --roll 8
    uv run python tools/diff_turn.py --battles 400 --self-switch 0.8
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import (  # noqa: E402
    MoveAction,
    PassAction,
    SideAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import (  # noqa: E402
    Budget,
    TurnResult,
    resolve_turn,
    resume_turn,
    self_switches_needed,
)
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
    compared: int = 0
    matched: int = 0
    #: Divergent turns where the resolver had already reported an unmodelled effect. These
    #: are documented gaps, not wrong answers.
    flagged_divergences: int = 0
    #: Divergent turns where the resolver reported nothing. This is the number that
    #: matters: a silent divergence means a printed value is quietly wrong.
    silent_divergences: int = 0
    #: Divergences per compared field kind.
    by_field: Counter[str] = field(default_factory=Counter)
    #: (field, move, ability, item) -> count.
    attributed: Counter[tuple[str, ...]] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    unmodelled: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    branch_counts: Counter[int] = field(default_factory=Counter)
    #: Turns the resolver suspended for a mid-turn replacement, and how the pause itself
    #: compared. Tracked separately because the class used to be skipped entirely: these
    #: counts are the evidence that it is now covered.
    paused: int = 0
    paused_matched: int = 0
    #: Divergences on turns that were interrupted, split out because the whole class used
    #: to be skipped: a change in the headline rate says nothing without this.
    paused_divergences: int = 0
    paused_by_field: Counter[str] = field(default_factory=Counter)
    #: The port's columns, one per binary, when the run was given any (IKA-207). The
    #: fields above stay Python's, so every caller that reads them reads what it always did.
    ports: dict[str, PortReport] = field(default_factory=dict)
    #: Where the current battle came from, for a port example to be replayed.
    where: str = ""

    @property
    def port(self) -> PortReport | None:
        """The first port column: the only one unless `--exes` named several."""
        return next(iter(self.ports.values()), None)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    @property
    def silent_rate(self) -> float:
        """The rate that matters: divergence with no warning attached."""
        return 0.0 if not self.compared else self.silent_divergences / self.compared

    def render(self) -> str:
        out = [
            f"compared {self.compared} turns, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}% "
            f"(silent {self.silent_rate * 100:.3f}%, "
            f"flagged {self.flagged_divergences})"
        ]
        if self.paused:
            out.append(
                f"  mid-turn replacement requests: {self.paused}, "
                f"state at the interrupt matched {self.paused_matched}, "
                f"turns diverging after resuming {self.paused_divergences}"
            )
            if self.paused_by_field:
                out.append(
                    "    on interrupted turns: "
                    + ", ".join(
                        f"{k} x{v}" for k, v in self.paused_by_field.most_common(8)
                    )
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
            out.append("  resolver reported unmodelled:")
            for name, count in self.unmodelled.most_common(15):
                out.append(f"    {count:5d}  {name}")
        for line in self.examples[:10]:
            out.append("  " + line)
        for label, port in self.ports.items():
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


def resolve_pauses(
    reg: Regulation,
    result: TurnResult,
    handle: Any,
    py_rng: random.Random,
    roll: int,
    report: Report,
) -> TurnResult | None:
    """Answers each mid-turn replacement identically on both sides and resumes.

    Returns the finished turn, or ``None`` when the turn could not be carried through --
    in which case the reason has already been recorded.
    """
    del roll
    for _ in range(4):
        if not result.suspended:
            return result
        if len(result.suspended) != 1 or result.branches:
            report.skipped["a deterministic turn split at the interrupt"] += 1
            return None
        pause = result.suspended[0]
        report.paused += 1
        if not showdown_paused_mid_turn(handle):
            report.silent_divergences += 1
            report.by_field["mid-turn interrupt"] += 1
            report.examples.append(
                "resolver suspended but Showdown did not stop: "
                + " / ".join(pause.events[-4:])
            )
            if os.environ.get("DIFF_DUMP_PAUSE"):
                print("=== resolver suspended, Showdown did not ===")
                print("  our events: " + " / ".join(pause.events))
                print("  requests: " + repr(handle.requests))
                for line in handle.log:
                    print("    " + line)
            return None

        # The two must agree about *which* slots owe a replacement before the choice can be
        # made identically on both sides.
        owed = self_switches_needed(pause.position)
        theirs = [
            list((r or {}).get("forceSwitch") or []) for r in handle.requests
        ]
        for side_index, flags in enumerate(owed):
            want = [bool(x) for x in theirs[side_index]][: len(flags)]
            if want and list(flags) != want:
                report.silent_divergences += 1
                report.by_field["mid-turn interrupt"] += 1
                report.examples.append(
                    f"forceSwitch slots differ on p{side_index + 1}: "
                    f"ours {list(flags)} vs showdown {want}"
                )
                return None

        # Compare the state at the pause: this is what the chooser can see, so if it is
        # wrong the choice is being made on a wrong position.
        at_pause = canonical(pause.position)
        showdown_pause = canonical(Position.from_json(handle.position))
        pause_diffs = [k for k in at_pause if at_pause[k] != showdown_pause.get(k)]
        if pause_diffs:
            report.by_field["at the interrupt"] += len(pause_diffs)
            report.examples.append(
                "at the interrupt: "
                + ", ".join(
                    f"{k}: ours {at_pause[k]!r} vs showdown {showdown_pause.get(k)!r}"
                    for k in pause_diffs[:3]
                )
            )
        else:
            report.paused_matched += 1

        picks: list[SideAction] = []
        told: list[str | None] = []
        for side_index in range(2):
            request = handle.requests[side_index]
            slots = range(len(pause.position.sides[side_index].active))
            if not request or request.get("wait") or not request.get("forceSwitch"):
                picks.append(SideAction(slots=tuple(PassAction(slot=i) for i in slots)))
                told.append(None)
                continue
            options = switch_actions_after_faint(
                reg, pause.position, side_index, list(owed[side_index])
            )
            pick = py_rng.choice(options)
            picks.append(pick)
            told.append(pick.to_choice())
        handle.step(told)
        if handle.choice_errors:
            report.skipped[f"replacement rejected: {handle.choice_errors}"] += 1
            return None
        result = resume_turn(reg, pause, picks)
    report.skipped["more than four mid-turn interrupts"] += 1
    return None


def compare_turn(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    handle: Any,
    lines: list[str],
    roll: int,
    report: Report,
    py_rng: random.Random,
    nodes: dict[str, Any] | None = None,
) -> None:
    """Python's column, and one port column per binary in ``nodes`` beside it.

    The ports go first: Python's mid-turn replacement steps Showdown on, and a port has to
    be held to the position Showdown reached at the end of the choices it was given.
    Python is solved once however many binaries are compared.
    """
    if not nodes:
        _compare_python(reg, before, chosen, handle, lines, roll, report, py_rng)
        return
    given = before.to_json()
    verdicts = {
        label: compare_port_turn(
            reg, node, report.ports[label], report.where, given, chosen, handle, lines, roll
        )
        for label, node in nodes.items()
    }
    matched = report.matched
    wrong = report.silent_divergences + report.flagged_divergences
    _compare_python(reg, before, chosen, handle, lines, roll, report, py_rng)
    # Python's interrupt checks count a divergence without counting a compared turn, so
    # the verdict is read off the divergence counters rather than off `compared`.
    if report.matched > matched:
        python_verdict = "match"
    elif report.silent_divergences + report.flagged_divergences > wrong:
        python_verdict = "diverge"
    else:
        python_verdict = "skip"
    for label, verdict in verdicts.items():
        report.ports[label].joint[(python_verdict, verdict)] += 1


def _compare_python(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    handle: Any,
    lines: list[str],
    roll: int,
    report: Report,
    py_rng: random.Random,
) -> None:
    # An Encore-style action override replaces a queued action after the turn starts,
    # which the resolver does not model; those turns are named rather than scored.
    # An action override replaces a queued action after the turn starts. Encore is
    # modelled now, so only the ones that are not are excluded -- a skip is a blind spot
    # and this class had been hiding Encore entirely.
    if action_overriding_effects(lines, only_unmodelled=True):
        report.skipped["action overridden mid-turn"] += 1
        return

    result = resolve_turn(reg, before, chosen, budget=Budget.deterministic(roll))
    report.branch_counts[len(result.branches) + len(result.suspended)] += 1
    for name in result.unmodelled:
        report.unmodelled[name] += 1
    if len(result.branches) + len(result.suspended) != 1:
        report.skipped[
            f"{len(result.branches) + len(result.suspended)} branches under a "
            "deterministic budget"
        ] += 1
        return

    # A self-switching move interrupts the turn, so the replacement is chosen and the turn
    # carries on -- answered the same way on both sides so this stays an equality test.
    was_paused = bool(result.suspended)
    if result.suspended:
        finished = resolve_pauses(reg, result, handle, py_rng, roll, report)
        if finished is None:
            return
        result = finished
        if len(result.branches) != 1:
            report.skipped["resuming produced more than one branch"] += 1
            return
        lines = handle.log

    # A forced switch (Roar, Whirlwind, Dragon Tail) drags in a *random* replacement rather
    # than asking, so it is still deferred to the post-turn phase and classified here.
    undetermined = [name for name in result.unmodelled if name.startswith("forceSwitch")]
    if undetermined:
        report.skipped["pending replacement (resolver defers to a choice)"] += 1
        return

    ours = canonical(result.branches[0].position)
    theirs = canonical(Position.from_json(handle.position))

    report.compared += 1
    differences = [k for k in ours if ours[k] != theirs.get(k)]
    if not differences:
        report.matched += 1
        return

    if result.unmodelled:
        report.flagged_divergences += 1
    else:
        report.silent_divergences += 1
    if was_paused:
        report.paused_divergences += 1
        for key in differences:
            report.paused_by_field[field_kind(key)] += 1

    moves = [
        s.move_id
        for a in chosen
        for s in a.slots
        if isinstance(s, MoveAction)
    ]
    abilities = sorted(
        {
            m.ability
            for side in before.sides
            for m in side.active_pokemon()
            if m is not None
        }
    )
    items = sorted(
        {
            m.item or ""
            for side in before.sides
            for m in side.active_pokemon()
            if m is not None
        }
    )
    for key in differences:
        kind = field_kind(key)
        report.by_field[kind] += 1
        report.attributed[(kind, ",".join(sorted(moves)), ",".join(abilities), ",".join(items))] += 1

    if not result.unmodelled and len(report.examples) < 10:
        detail = "; ".join(
            f"{k}: ours {ours[k]!r} vs showdown {theirs.get(k)!r}" for k in differences[:5]
        )
        report.examples.append(
            f"turn {before.turn} [{describe_actions(reg, chosen)}] -> {detail}\n"
            f"      our events: {' / '.join(result.branches[0].events[:12])}"
        )


# ---------------------------------------------------------------------------
# The port's column (IKA-207): the same turn, the same pins, `RustNode.resolve`.


@dataclass
class PortReport:
    """The port held to Showdown on the turns Python is held to, counted apart.

    A refusal is not a divergence and not a match: it is counted by its reason, because
    until the port answers everything (IKA-208) "how often does it decline, and why" is
    half of what this column is for. A turn both the port and Showdown stopped inside, at
    a mid-turn replacement, is counted apart too: the continuation stays in the Rust
    process and there is no command to carry it on yet (IKA-211).
    """

    binary: dict[str, Any] = field(default_factory=dict)
    compared: int = 0
    matched: int = 0
    flagged: int = 0
    silent: int = 0
    paused: int = 0
    by_field: Counter[str] = field(default_factory=Counter)
    attributed: Counter[tuple[str, ...]] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    refused: Counter[str] = field(default_factory=Counter)
    unmodelled: Counter[str] = field(default_factory=Counter)
    branch_counts: Counter[int] = field(default_factory=Counter)
    #: (Python's verdict, the port's verdict) per turn both were asked about.
    joint: Counter[tuple[str, str]] = field(default_factory=Counter)
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
                f"  stopped at a mid-turn replacement with Showdown: {self.paused} "
                "(no command to continue it yet; not compared)"
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
        if self.joint:
            out.append("  the same turns, python / port:")
            for (python, port), count in sorted(self.joint.items()):
                out.append(f"    {count:5d}  python {python:<8} port {port}")
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
    """One turn through the port; returns its verdict (match, diverge, refused, paused, skip).

    ``given`` is the position before the turn; ``handle`` is Showdown after it. Called
    before Python's column, while Showdown still stands where the choices left it.
    A turn where the port and Showdown disagree about stopping for a mid-turn replacement
    is a compared, divergent turn with the field ``mid-turn interrupt``.

    One request a turn: a deterministic budget leaves one outcome, so branch 0 is asked
    for together with the weights. Only when there is no finished branch -- the turn
    stopped at a replacement -- does the port refuse the index, and only then is it asked
    again for the weights alone.
    """
    if action_overriding_effects(lines, only_unmodelled=True):
        port.skipped["action overridden mid-turn"] += 1
        return "skip"
    before = port_position(given)
    budget = Budget.deterministic(roll)
    reply = port_exchange(node, before, chosen, budget, select=0)
    if reply.get("refused") == "branch index out of range":
        reply = port_exchange(node, before, chosen, budget)
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
        return "paused"
    if bool(stopped) != showdown_stopped:
        differences = {
            "mid-turn interrupt": (
                "port stopped" if stopped else "port carried on",
                "showdown stopped" if showdown_stopped else "showdown carried on",
            )
        }
        _port_divergence(reg, port, where, before, chosen, unmodelled, differences, handle)
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
    _port_divergence(reg, port, where, before, chosen, unmodelled, differences, handle)
    return "diverge"


def _port_divergence(
    reg: Regulation,
    port: PortReport,
    where: str,
    before: Position,
    chosen: list[SideAction],
    unmodelled: tuple[str, ...],
    differences: dict[str, tuple[Any, Any]],
    handle: Any,
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
            "log": list(handle.log),
        }
    )


def parse_exes(text: str) -> list[tuple[str, Path]]:
    """`--exes old=C:/x.exe,new` -> labelled binaries; a bare `new`/`current` is this tree's.

    Named as `tools/diff_node.py --exes` names them (IKA-206), so one list of binaries
    reads the same in both tools.
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


def run(
    battles: int,
    roll: int,
    seed: int,
    max_turns: int,
    quiet: bool = True,
    self_switch: float = 0.0,
    port: bool = False,
    exes: list[tuple[str, Path]] | None = None,
) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
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
    nodes = open_ports(reg, report, exes) if port or exes else {}

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
                    choices.append(pick.to_choice())
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    break
                if forced or len(chosen) != 2:
                    report.skipped["replacement turn"] += 1
                    continue
                compare_turn(
                    reg, before, chosen, handle, handle.log, roll, report, py_rng,
                    nodes=nodes,
                )
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
        "--no-port",
        action="store_true",
        help="Python's column only (the port's needs rust/target/release; "
        "POKEURAOU_RUST_NODE_BIN names another binary)",
    )
    ap.add_argument(
        "--port-json",
        help="write every turn the port diverged on, with Showdown's log, to this file",
    )
    ap.add_argument(
        "--exes",
        help="several port columns in one run, Python solved once: "
        "`old=C:/tmp/old.exe,new` (a bare `new` is this tree's rust/target/release)",
    )
    args = ap.parse_args()
    if not os.environ.get("PYTHONHASHSEED"):
        # Two runs of the same seed differ by a turn or two without it (IKA-207 measured
        # 2,726 and 2,727 compared turns at 400 battles), so a pair of runs is not a pair.
        print("[diff_turn] PYTHONHASHSEED is unset: the same --seed can play other turns",
              file=sys.stderr)
    report = run(
        args.battles,
        args.roll,
        args.seed,
        args.max_turns,
        quiet=False,
        self_switch=args.self_switch,
        port=not args.no_port,
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
