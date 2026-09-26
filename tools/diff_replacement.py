"""Differential test of the replacement phase against Showdown.

This is the part of a battle the single-turn tool was allowed to stop at: after a faint,
which Pokemon comes in is the player's decision, so the resolver recorded it and handed it
back. The turn differential harness classified those turns as "pending replacement" and
skipped them -- 327 of them in a 40-battle run -- which means the switch-in half of the
game has never been compared with anything.

Self-play to a win or a loss goes through this phase constantly, so it gets the same
treatment as everything else: play real battles, make the choice explicitly on both sides,
apply the identical choice through the port's replacement phase (`replacements`, IKA-211;
Python's `resolve_replacements` until IKA-212), and compare the resulting positions field
by field.

Two things get checked, not one. Whether the port *knows* a replacement is owed --
`replacements_needed` against Showdown's own `forceSwitch` flags -- and whether applying
it produces the same position. The first failing silently would be worse: a game loop that
does not notice it owes a replacement simply stops.

    uv run python tools/diff_replacement.py --battles 25
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diff_turn import (  # noqa: E402
    REFUSED_CHOICE,
    canonical,
    field_kind,
    port_position,
    showdown_choice,
)

from pokeuraou import port
from pokeuraou.actions import PassAction, SideAction, side_actions, switch_actions_after_faint
from pokeuraou.damage import register_mega_stones
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team
from pokeuraou.regulation import Regulation, load_regulation

FORMAT_ID = "gen9championsvgc2026regmc"


# The port's replacement phase (IKA-212). A Showdown position is handed over as the port
# is given one (`diff_turn.port_position`: Showdown's stats dropped where nothing reads them).
def replacements_needed(reg: Regulation, pos: Position) -> tuple[tuple[bool, ...], ...]:
    return port.replacements_needed(reg, port_position(pos.to_json()))


def resolve_replacements(reg: Regulation, pos: Position, choices: list) -> object:
    """The phase with its trace, for the examples; a refusal raises `port.PortRefused`."""
    shown = port_position(pos.to_json())

    def call(node):  # noqa: ANN001, ANN202
        answer = node.resolve_replacements(shown, list(choices), events=True)
        if answer is None:
            raise port.PortRefused(f"the port refused a replacement phase: {node.refusal}")
        return answer

    return port.ask(reg, call)


@dataclass
class Report:
    compared: int = 0
    matched: int = 0
    flag_mismatches: int = 0
    by_field: Counter[str] = field(default_factory=Counter)
    unmodelled: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    def render(self) -> str:
        lines = [
            f"compared {self.compared} replacement phases, matched {self.matched}, "
            f"divergence {self.divergence_rate * 100:.3f}%, "
            f"forceSwitch flag mismatches {self.flag_mismatches}"
        ]
        for name, count in self.by_field.most_common(10):
            lines.append(f"  {count:>5}  {name}")
        for name, count in self.unmodelled.most_common(8):
            lines.append(f"  unmodelled {count:>4}  {name}")
        for name, count in self.skipped.most_common(8):
            lines.append(f"  skipped {count:>4}  {name}")
        lines.extend(self.examples[:6])
        return "\n".join(lines)


def _pass_side(pos: Position, side_index: int) -> SideAction:
    return SideAction(
        slots=tuple(
            PassAction(slot=slot) for slot in range(len(pos.sides[side_index].active))
        )
    )


def run(battles: int, seed: int, max_turns: int, quiet: bool = True) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    report = Report()
    policy = RandomnessPolicy(
        damage_roll=0, accuracy="hit", crit=False, secondary=False,
        multihit="min", speed_tie="keep",
    )

    with Oracle() as oracle:
        for _ in range(battles):
            teams = [
                [TeamSet.from_json(s.to_team_set_json(reg)) for s in sample_team(rng, reg, prior)]
                for _ in range(2)
            ]
            handle = oracle.create(
                FORMAT_ID, teams[0], teams[1],
                seed=tuple(int(rng.integers(1, 60000)) for _ in range(4)),  # type: ignore[arg-type]
                policy=policy,
            )
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])

            for _step in range(max_turns * 2):
                before = Position.from_json(handle.position)
                if before.ended:
                    break
                forced = [
                    bool(request and request.get("forceSwitch"))
                    for request in handle.requests
                ]
                # A *pure* replacement phase is one where no side has an ordinary turn to
                # choose. Anything else is a turn that happens to contain a switch, and
                # stepping it here would compare a whole turn's worth of change against a
                # replacement -- which is what the first version of this harness did, and
                # it read as a 7.5% divergence that had nothing to do with replacements.
                mixed = any(
                    request
                    and not request.get("wait")
                    and not request.get("forceSwitch")
                    for request in handle.requests
                )
                if any(forced) and mixed:
                    report.skipped["not a pure replacement phase"] += 1
                if not any(forced) or mixed:
                    choices: list[str | None] = []
                    for side_index in range(2):
                        request = handle.requests[side_index]
                        if not request or request.get("wait"):
                            choices.append(None)
                            continue
                        options = side_actions(reg, before, side_index)
                        # Numbered by Showdown's request: a locked move is `move 1` (IKA-316).
                        choices.append(
                            showdown_choice(py_rng.choice(options), request) if options else None
                        )
                    if all(c is None for c in choices):
                        break
                    handle.step(choices)
                    if handle.choice_errors:
                        report.skipped[REFUSED_CHOICE] += 1
                        break
                    continue

                _compare(reg, before, handle, py_rng, report)
                if handle.choice_errors:
                    # `_compare` counted it under "choice rejected: ..."; the battle stops.
                    report.skipped[REFUSED_CHOICE] += 1
                    break
            handle.close()

    if not quiet:
        print(report.render())
    return report


def _compare(
    reg: Regulation,
    before: Position,
    handle: object,  # noqa: ANN401
    py_rng: random.Random,
    report: Report,
) -> None:
    ours_flags, theirs_flags = replacements_needed(reg, before)
    our_view = [ours_flags, theirs_flags]

    chosen: list[SideAction] = []
    step: list[str | None] = []
    for side_index in range(2):
        request = handle.requests[side_index]  # type: ignore[attr-defined]
        force = list(request.get("forceSwitch") or []) if request else []
        if not force:
            chosen.append(_pass_side(before, side_index))
            step.append(None if not request or request.get("wait") else "pass")
            continue

        must = [bool(f) for f in force]
        # Showdown is the authority on whether a replacement is owed; a disagreement means
        # a self-play loop would either stall or send an illegal choice.
        ours_view = list(our_view[side_index][: len(must)])
        if tuple(must) != tuple(ours_view):
            alive_pending = [
                slot
                for slot, wanted in enumerate(must)
                if wanted
                and not ours_view[slot]
                and (mon := before.sides[side_index].pokemon[
                    before.sides[side_index].active[slot]
                ]) is not None
                and not mon.fainted
            ]
            if alive_pending and len(alive_pending) == sum(
                1 for slot, wanted in enumerate(must) if wanted and not ours_view[slot]
            ):
                # A self-switch or a forced switch is pending on a *conscious* Pokemon.
                # The resolver records that as a volatile on its own output; a position
                # read back from Showdown cannot carry it, because Showdown expresses it
                # through the request instead. Not a disagreement -- an invisible cause.
                report.skipped["pending switch invisible in a Showdown snapshot"] += 1
            else:
                report.flag_mismatches += 1
                report.examples.append(
                    f"  forceSwitch p{side_index + 1}: showdown {must} vs ours {ours_view}"
                )
        options = switch_actions_after_faint(reg, before, side_index, must)
        if not options:
            report.skipped["no legal replacement enumerated"] += 1
            chosen.append(_pass_side(before, side_index))
            step.append("pass")
            continue
        pick = py_rng.choice(options)
        chosen.append(pick)
        step.append(pick.to_choice())

    if all(c is None for c in step):
        report.skipped["nothing to send"] += 1
        return

    handle.step(step)  # type: ignore[attr-defined]
    if handle.choice_errors:  # type: ignore[attr-defined]
        report.skipped[f"choice rejected: {handle.choice_errors}"] += 1  # type: ignore[attr-defined]
        return

    # Not every `forceSwitch` request is an end-of-turn replacement: Eject Button and
    # Emergency Exit ask for one *mid*-turn, and answering it makes Showdown finish the
    # rest of the turn in the same step -- the remaining moves and the residuals. Those
    # steps showed up as an 8.6% divergence made entirely of other Pokemon's HP and an
    # expiring Trick Room. The step's own protocol says which kind it was.
    #
    # `|upkeep` is the marker for the residual phase, which is the other half of the same
    # story: a faint mid-turn with no moves left still leaves the residuals to run after
    # the replacement, and Leftovers healing two Pokemon by 12 each is not a replacement
    # bug. An end-of-turn replacement happens *after* upkeep, so its own step has none.
    if any(
        line.startswith("|move|") or line.startswith("|upkeep")
        for line in handle.log  # type: ignore[attr-defined]
    ):
        report.skipped["mid-turn replacement: the step finished the turn"] += 1
        return

    result = resolve_replacements(reg, before, chosen)
    for name in result.unmodelled:
        report.unmodelled[name] += 1

    ours = canonical(result.position)
    theirs = canonical(Position.from_json(handle.position))  # type: ignore[attr-defined]
    report.compared += 1
    differences = [k for k in ours if ours[k] != theirs.get(k)]
    if not differences:
        report.matched += 1
        return
    for key in differences:
        report.by_field[field_kind(key)] += 1
    if len(report.examples) < 12:
        detail = "; ".join(
            f"{k}: ours {ours[k]!r} vs sd {theirs.get(k)!r}" for k in differences[:4]
        )
        report.examples.append(
            f"  replacement [{' | '.join(str(c) for c in step)}] -> {detail}\n"
            f"      our events: {' / '.join(result.events[:8])}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=14)
    args = ap.parse_args()
    run(args.battles, args.seed, args.max_turns, quiet=False)


if __name__ == "__main__":
    main()
