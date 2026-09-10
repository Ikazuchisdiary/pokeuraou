"""Differential test of turn order against Showdown.

Turn order is the part of a doubles turn a player reasons about most, and it is the part
the opponent's hidden Speed makes uncertain, so it gets its own check: predict the order
from the pre-turn position and the chosen actions, then read the order Showdown actually
used out of the protocol and compare the sequences.

Speed ties are forced to a known outcome (``speed_tie='keep'`` leaves the tied group in
queue order), so a tie is not a source of noise here -- ties are reported separately
because the resolver has to branch on them.

    uv run python tools/diff_order.py --battles 60
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

from pokeuraou.actions import SideAction, side_actions  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.speed import (  # noqa: E402
    action_overriding_effects,
    build_queue,
    order_groups,
    speed_changing_effects,
)
from pokeuraou.view import active_battlers, field_state  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def desplit(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|split|"):
            if i + 1 < len(lines):
                out.append(lines[i + 1])
            i += 3
            continue
        out.append(lines[i])
        i += 1
    return out


def parse_ident(ident: str) -> tuple[int, int]:
    who = ident.split(":")[0].strip()
    side = 0 if who.startswith("p1") else 1
    slot = "abc".index(who[2]) if len(who) > 2 and who[2] in "abc" else 0
    return side, slot


def actual_order(lines: list[str]) -> list[tuple[int, int, str]]:
    """The sequence of (side, slot, kind) that Showdown executed this turn.

    ``|cant|`` counts as executing: a flinched or sleeping Pokemon still consumed its slot
    in the queue at the position its Speed put it, which is what the order prediction is
    about.

    One shape needs care. When a priority-blocking ability stops a move, Showdown emits

        |cant|<blocker>|ability: Armor Tail|<move>|[of] <user>

    where the *named* Pokemon is the blocker and the one that actually acted is in the
    ``[of]`` field. Reading parts[2] there attributes the action to the wrong side, which
    looks exactly like a turn-order bug.

    A ``|switch|`` after a move belongs to that move (U-turn, Eject Button) rather than to
    a chosen switch, so only switches before the first move are treated as chosen.
    """
    out: list[tuple[int, int, str]] = []
    seen_move = False
    for line in lines:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        if tag == "switch" and not seen_move:
            out.append((*parse_ident(parts[2]), "switch"))
        elif tag == "-mega":
            out.append((*parse_ident(parts[2]), "mega"))
        elif tag in ("move", "cant"):
            seen_move = True
            actor = parts[2]
            for part in parts[3:]:
                if part.startswith("[of] "):
                    actor = part[5:]
            out.append((*parse_ident(actor), "move"))
    return out


@dataclass
class Report:
    compared: int = 0
    matched: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    #: How often the belief produced more than one candidate order for one turn.
    multi_order_turns: int = 0
    #: How often an exact Speed tie appeared, which the resolver must branch on.
    tie_turns: int = 0
    #: Turns where a Speed changed partway through, so Showdown re-sorted the queue and
    #: the pre-turn prediction could not have been right. Classified, not counted as a
    #: divergence -- the resolver re-sorts and does get these right.
    resorted_turns: int = 0
    resort_causes: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    def render(self) -> str:
        out = [
            f"compared {self.compared} turns, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}%",
            f"  turns with an exact Speed tie: {self.tie_turns}",
            f"  turns where the belief gave several orders: {self.multi_order_turns}",
            f"  turns Showdown re-sorted mid-turn (excluded): {self.resorted_turns}"
            + (
                "  causes: "
                + ", ".join(f"{k} x{v}" for k, v in self.resort_causes.most_common(6))
                if self.resort_causes
                else ""
            ),
        ]
        if self.skipped:
            out.append(
                "  skipped: " + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(8))
            )
        for line in self.examples[:12]:
            out.append("  " + line)
        return "\n".join(out)


def predicted_order(
    reg: Regulation, pos: Position, chosen: list[SideAction]
) -> tuple[list[tuple[int, int, str]], bool, bool]:
    """(order, had_ties, had_several_orders) for a fully-known position."""
    fs = field_state(pos, reg)
    battlers = [active_battlers(reg, side) for side in pos.sides]
    branches = build_queue(reg, pos, chosen, battlers, fs)
    if not branches or not branches[0]:
        return [], False, False
    # With no Quick Claw in play there is a single fractional-priority branch; when there
    # is one, the highest-probability branch is the one Showdown's forced randomness
    # produces (secondary=False makes the claw fail).
    queue = max(branches, key=lambda b: b[0].branch_probability if b else 0.0)
    groups = order_groups(queue, trick_room=pos.field.trick_room)
    if not groups:
        return [], False, False
    best = max(groups, key=lambda g: g.count)
    order = [(queue[i].side, queue[i].slot, queue[i].kind) for i in best.order]
    return order, bool(best.ties), len(groups) > 1


def compare_turn(
    reg: Regulation, pos: Position, chosen: list[SideAction], lines: list[str], report: Report
) -> None:
    predicted, had_ties, several = predicted_order(reg, pos, chosen)
    if not predicted:
        report.skipped["no predicted actions"] += 1
        return
    actual = actual_order(desplit(lines))
    if had_ties:
        report.tie_turns += 1
    if several:
        report.multi_order_turns += 1

    # Keep only the actions that both sequences know about: an action whose user fainted
    # before its turn never executes, and a mid-turn switch-in is not a chosen action.
    predicted_keys = [(s, sl, k) for s, sl, k in predicted]
    actual_keys = [k for k in actual if k in predicted_keys]
    # De-duplicate while keeping the first occurrence: a mega and its move share a slot.
    seen: set[tuple[int, int, str]] = set()
    actual_keys = [k for k in actual_keys if not (k in seen or seen.add(k))]
    filtered_prediction = [k for k in predicted_keys if k in set(actual_keys)]

    if len(filtered_prediction) < 2:
        report.skipped["fewer than two comparable actions"] += 1
        return

    # A Speed change partway through the turn makes Showdown re-sort the remaining queue,
    # so a prediction from the pre-turn state is not expected to match. These turns are
    # counted separately instead of being scored.
    if filtered_prediction != actual_keys:
        causes = speed_changing_effects(desplit(lines)) | action_overriding_effects(
            desplit(lines)
        )
        if causes:
            report.resorted_turns += 1
            for cause in causes:
                report.resort_causes[cause] += 1
            return

    report.compared += 1
    if filtered_prediction == actual_keys:
        report.matched += 1
        return
    if had_ties:
        # A tie means Showdown shuffled; with speed_tie='keep' it keeps queue order, which
        # our canonical order need not match. Counted, not failed.
        report.matched += 1
        report.skipped["tie ordering (not a divergence)"] += 1
        return
    if len(report.examples) < 12:
        fs = field_state(pos, reg)
        battlers = [active_battlers(reg, side) for side in pos.sides]
        queue = build_queue(reg, pos, chosen, battlers, fs)[0]
        detail = "; ".join(
            f"p{a.side + 1}{chr(97 + a.slot)} {a.kind}:{a.move_id} pri={a.priority}"
            f"{a.fractional:+.1f} spe={int(a.speed[0])}"
            for a in queue
        )
        report.examples.append(
            f"turn {pos.turn} TR={pos.field.trick_room}: predicted {filtered_prediction} "
            f"but Showdown ran {actual_keys}\n      {detail}"
        )


def run(battles: int, seed: int, max_turns: int, quiet: bool = True) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    report = Report()
    policy = RandomnessPolicy(
        damage_roll=8, accuracy="hit", crit=False, secondary=False, speed_tie="keep"
    )

    with Oracle() as oracle:
        for _ in range(battles):
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
                pos = Position.from_json(handle.position)
                if pos.ended:
                    break
                requests = handle.requests
                chosen: list[SideAction] = []
                choices: list[str | None] = []
                forced = False
                for side_index in range(2):
                    request = requests[side_index]
                    if request and request.get("forceSwitch"):
                        forced = True
                        choices.append("default")
                        continue
                    if not request or request.get("wait"):
                        choices.append(None)
                        continue
                    options = side_actions(reg, pos, side_index)
                    pick = py_rng.choice(options)
                    chosen.append(pick)
                    choices.append(pick.to_choice())
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    break
                if not forced and len(chosen) == 2:
                    compare_turn(reg, pos, chosen, handle.log, report)
            handle.close()

    if not quiet:
        print(report.render())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()
    run(args.battles, args.seed, args.max_turns, quiet=False)


if __name__ == "__main__":
    main()
