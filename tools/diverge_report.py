"""Ranks the causes of silent resolver divergence by statistical lift.

The per-turn report names the whole cast of a divergent turn -- four moves, four abilities,
four items -- which makes a common effect look guilty just for being common. This tool
runs many seeds and, for each individual effect, compares how often it appears in divergent
turns with how often it appears in matching ones:

    lift = P(effect | divergent) / P(effect | matched)

An effect with high lift and a decent count is a cause. An effect with lift near 1 is
merely popular. That turns "what to fix next" into a ranking rather than a guess.

    uv run python tools/diverge_report.py --seeds 12 --battles 10
    uv run python tools/diverge_report.py --seeds 20 --battles 20 --jobs 8

The engine ranked is the port (`RustNode.resolve`, the same pins, IKA-207); a turn it
refuses is counted under `skipped` by its reason, and `POKEURAOU_RUST_NODE_BIN` names
another binary. Python's ranking ran beside it until IKA-212 deleted Python's resolver.
A turn the port and Showdown both stopped inside at a mid-turn replacement is carried on
as `diff_turn` carries it (IKA-217): Showdown's run answers the replacement with `default`,
the port's pause is resumed with the Pokemon Showdown sent in, and the end of the turn is
scored like any other (IKA-227; before, such turns were set aside). The battles played do
not change with it: the port never steps Showdown.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import diff_turn  # noqa: E402
from diff_turn import canonical, field_kind  # noqa: E402

from pokeuraou.actions import MoveAction, SideAction, side_actions  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.speed import action_overriding_effects  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def effects_in_play(
    reg: Regulation, pos: Position, chosen: list[SideAction]
) -> set[str]:
    """Every named effect that could plausibly be responsible for one turn."""
    out: set[str] = set()
    for action in chosen:
        for slot_action in action.slots:
            if isinstance(slot_action, MoveAction):
                out.add(f"move:{slot_action.move_id}")
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None:
                continue
            out.add(f"ability:{mon.ability}")
            if mon.item:
                out.add(f"item:{mon.item}")
            if mon.status:
                out.add(f"status:{mon.status}")
            for volatile in mon.volatiles:
                out.add(f"volatile:{volatile.id}")
            for volatile in mon.unmodelled_volatiles:
                out.add(f"unmodelled-volatile:{volatile}")
        for condition in side.side_conditions:
            out.add(f"side:{condition.id}")
    if pos.field.weather:
        out.add(f"weather:{pos.field.weather}")
    if pos.field.terrain:
        out.add(f"terrain:{pos.field.terrain}")
    for pseudo in pos.field.pseudo_weather:
        out.add(f"pseudo:{pseudo.id}")
    del reg
    return out


@dataclass
class Aggregate:
    matched: int = 0
    silent: int = 0
    flagged: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    #: effect -> how many matching turns it appeared in.
    in_matched: Counter[str] = field(default_factory=Counter)
    #: effect -> how many silently divergent turns it appeared in.
    in_silent: Counter[str] = field(default_factory=Counter)
    by_field: Counter[str] = field(default_factory=Counter)
    #: Bucketed ratio of the damage we dealt to the damage Showdown dealt. A missing
    #: modifier shows up as a pile at 0.67 (a missing 1.5x), 0.77 (1.3x), 0.5 or 2.0,
    #: which names the mechanic without reading examples one at a time.
    damage_ratio: Counter[str] = field(default_factory=Counter)
    #: effect -> the fields that diverged alongside it, for reading the mechanism off.
    fields_with: dict[str, Counter[str]] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)
    #: Turns the port and Showdown both stopped inside at a mid-turn replacement, and of
    #: those, the ones carried on to the end and scored, with their silent and flagged
    #: divergences (IKA-227; until then they were set aside unscored).
    paused: int = 0
    carried_on: int = 0
    carried_on_silent: int = 0
    carried_on_flagged: int = 0

    @property
    def compared(self) -> int:
        return self.matched + self.silent + self.flagged

    def merge(self, other: Aggregate) -> None:
        """Adds ``other`` (the next seed) as if its turns had been scored here after these.

        Every counter and dict keeps first-seen order, which the ranking's ties and the
        examples read, so merging seeds in order gives the report one process gives.
        """
        self.matched += other.matched
        self.silent += other.silent
        self.flagged += other.flagged
        self.paused += other.paused
        self.carried_on += other.carried_on
        self.carried_on_silent += other.carried_on_silent
        self.carried_on_flagged += other.carried_on_flagged
        for mine, theirs in (
            (self.skipped, other.skipped),
            (self.in_matched, other.in_matched),
            (self.in_silent, other.in_silent),
            (self.by_field, other.by_field),
            (self.damage_ratio, other.damage_ratio),
        ):
            mine.update(theirs)
        for effect, fields in other.fields_with.items():
            self.fields_with.setdefault(effect, Counter()).update(fields)
        for effect, lines in other.examples.items():
            bucket = self.examples.setdefault(effect, [])
            bucket.extend(lines[: max(0, 2 - len(bucket))])

    def lift(self, effect: str) -> float:
        """P(effect | silent divergence) / P(effect | match)."""
        if not self.silent or not self.matched:
            return 0.0
        p_silent = self.in_silent[effect] / self.silent
        p_matched = self.in_matched[effect] / self.matched
        # An effect never seen in a matching turn is maximally suspicious, but only if it
        # occurred often enough to mean anything; the caller filters on count.
        return p_silent / p_matched if p_matched > 0 else float(p_silent * self.matched)

    def render(self, top: int, min_count: int) -> str:
        out = [
            f"compared {self.compared} turns: {self.matched} matched, "
            f"{self.silent} silent ({self.silent / max(self.compared, 1) * 100:.2f}%), "
            f"{self.flagged} flagged"
        ]
        out.append(
            f"  stopped with Showdown at a mid-turn replacement {self.paused} times; carried on "
            f"and scored {self.carried_on} turns: {self.carried_on_silent} silent, "
            f"{self.carried_on_flagged} flagged"
        )
        if self.by_field:
            out.append(
                "  divergent fields: "
                + ", ".join(f"{k} x{v}" for k, v in self.by_field.most_common(10))
            )
        if self.damage_ratio:
            out.append(
                "  our damage / Showdown's, bucketed: "
                + ", ".join(f"{k} x{v}" for k, v in self.damage_ratio.most_common(12))
            )
        out.append("")
        out.append(
            f"  causes ranked by lift (count >= {min_count}); lift near 1 means the effect "
            "is merely common"
        )
        out.append(f"  {'effect':<42} {'silent':>7} {'matched':>8} {'lift':>7}  fields")
        ranked = [
            (effect, self.lift(effect))
            for effect, count in self.in_silent.items()
            if count >= min_count
        ]
        ranked.sort(key=lambda pair: -pair[1])
        for effect, lift in ranked[:top]:
            fields = self.fields_with.get(effect, Counter())
            field_text = ", ".join(f"{k}x{v}" for k, v in fields.most_common(3))
            out.append(
                f"  {effect:<42} {self.in_silent[effect]:>7} {self.in_matched[effect]:>8} "
                f"{lift:>7.2f}  {field_text}"
            )
        if self.skipped:
            out.append("")
            out.append(
                "  skipped: "
                + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(6))
            )
        for effect, lines in list(self.examples.items())[:6]:
            out.append("")
            out.append(f"  -- {effect}")
            for line in lines[:2]:
                out.append(f"     {line}")
        return "\n".join(out)


def run(seeds: int, battles: int, roll: int, max_turns: int, jobs: int = 1) -> Aggregate:
    """Every seed's battles, ranked together.

    A seed is the unit of work: its teams and choices come from its own generators, so the
    seeds are independent and ``jobs`` processes can play them side by side. The per-seed
    tallies are merged in seed order, which is the order one process plays them in, so the
    report does not depend on ``jobs`` (`test_diverge_report`).
    """
    from pokeuraou import rustnode

    rustnode.require_current_binary()
    order = list(range(1, seeds + 1))
    chunks = [(order[k::jobs], battles, roll, max_turns) for k in range(max(1, min(jobs, seeds)))]
    if len(chunks) == 1:
        parts = _run_seeds(chunks[0])
    else:
        from concurrent.futures import ProcessPoolExecutor

        # By the name a worker can import (tools/ is on the path it inherits): run as a
        # script this module is `__main__`, and loaded by the tests it has another name.
        import diverge_report as importable

        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            parts = [part for done in pool.map(importable._run_seeds, chunks) for part in done]
    agg = Aggregate()
    for _, part in sorted(parts, key=lambda pair: pair[0]):
        agg.merge(part)
    return agg


def _run_seeds(task: tuple[list[int], int, int, int]) -> list[tuple[int, Aggregate]]:
    """One process's seeds: one regulation, one Showdown and one port for all of them."""
    seeds, battles, roll, max_turns = task
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    # The same teams whatever PYTHONHASHSEED is, as in diff_turn (IKA-218).
    prior = diff_turn.hash_free_prior(reg, load_chaos(chaos, reg))
    register_mega_stones(reg)
    policy = RandomnessPolicy(
        damage_roll=roll, accuracy="hit", crit=False, secondary=False,
        multihit="min", speed_tie="keep",
    )
    from pokeuraou import rustnode

    node = rustnode.RustNode(reg)
    out: list[tuple[int, Aggregate]] = []
    try:
        with Oracle() as oracle:
            for seed in seeds:
                agg = Aggregate()
                _play_seed(reg, prior, policy, oracle, node, seed, battles, roll, max_turns, agg)
                out.append((seed, agg))
    finally:
        node.close()
    return out


def _play_seed(
    reg: Regulation,
    prior: Any,  # noqa: ANN401 - priors.MetagamePrior
    policy: RandomnessPolicy,
    oracle: Any,  # noqa: ANN401 - oracle.Oracle
    node: Any,  # noqa: ANN401 - rustnode.RustNode
    seed: int,
    battles: int,
    roll: int,
    max_turns: int,
    agg: Aggregate,
) -> None:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    for battle in range(battles):
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
        # Every Showdown step from here on, and the port's paused turn if there is one:
        # the run's own `default` replacement carries the port on (IKA-227).
        carry = _Carry(tap=diff_turn.StepTap(handle), where=f"seed {seed} battle {battle}")
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
                pick = py_rng.choice(side_actions(reg, before, side_index))
                chosen.append(pick)
                # Numbered by Showdown's request: a locked move is `move 1` (IKA-316).
                choices.append(diff_turn.showdown_choice(pick, request))
            if all(c is None for c in choices):
                break
            handle.step(choices)
            _carry_on(reg, node, carry, agg)
            if handle.choice_errors:
                agg.skipped[diff_turn.REFUSED_CHOICE] += 1
                break
            if forced or len(chosen) != 2:
                agg.skipped["replacement turn"] += 1
                continue
            _score_port(reg, node, before, chosen, handle, roll, agg, carry)
        # The last turn allowed can stop at a replacement the loop will not answer. The
        # same `default` the next iteration would have sent finishes it: nothing after it
        # is scored and no generator is drawn, so the battles played are unchanged.
        for _ in range(4):
            if carry.column.pending is None or handle.choice_errors:
                break
            owed = [bool(r and r.get("forceSwitch")) for r in handle.requests]
            if not any(owed):
                break
            handle.step(["default" if o else None for o in owed])
            _carry_on(reg, node, carry, agg)
        _settle(carry, agg, "the battle stopped inside a paused turn")
        handle.close()


@dataclass
class _Carry:
    """A battle's Showdown steps and the port's column for a turn being carried on."""

    tap: Any  # diff_turn.StepTap
    where: str
    column: Any = field(default_factory=lambda: diff_turn.PortReport())


def _carry_on(reg: Regulation, node: Any, carry: _Carry, agg: Aggregate) -> None:  # noqa: ANN401
    """Resumes a paused turn with the Showdown steps taken since (`diff_turn.follow_port`),
    and scores it once the turn has ended."""
    pending = carry.column.pending
    if pending is None:
        return
    diff_turn.follow_port(reg, node, carry.column, carry.tap)
    if carry.column.pending is None and pending.decided is not None:
        ours, theirs, unmodelled = pending.decided
        _tally(reg, pending.before, pending.chosen, ours, theirs, bool(unmodelled), agg, True)


def _settle(carry: _Carry, agg: Aggregate, reason: str) -> None:
    """A paused turn left unanswered is named, and what `follow_port` set aside is moved over."""
    diff_turn.settle_port(carry.column, reason)
    for name, count in carry.column.skipped.items():
        agg.skipped[f"carried on, then {name}"] += count
    for name, count in carry.column.refused.items():
        agg.skipped[f"refused: {name}"] += count
    carry.column.skipped.clear()
    carry.column.refused.clear()


def _tally(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    ours: dict[str, Any],
    theirs: dict[str, Any],
    flagged: bool,
    agg: Aggregate,
    carried: bool = False,
) -> None:
    """One scored turn into the ranking; ``carried``: a turn resumed past a replacement."""
    # Sorted: a set's order follows PYTHONHASHSEED, and the counters' first-seen order is
    # what breaks ties in the ranking and picks the examples (IKA-218).
    effects = sorted(effects_in_play(reg, before, chosen))
    differences = [k for k in ours if ours[k] != theirs.get(k)]
    agg.carried_on += carried

    if not differences:
        agg.matched += 1
        for effect in effects:
            agg.in_matched[effect] += 1
        return
    if flagged:
        agg.flagged += 1
        agg.carried_on_flagged += carried
        return

    agg.silent += 1
    agg.carried_on_silent += carried
    _record_damage_ratios(before, ours, theirs, differences, agg)
    kinds = sorted({field_kind(k) for k in differences})
    for kind in kinds:
        agg.by_field[kind] += 1
    for effect in effects:
        agg.in_silent[effect] += 1
        agg.fields_with.setdefault(effect, Counter()).update(kinds)
        bucket = agg.examples.setdefault(effect, [])
        if len(bucket) < 2:
            detail = "; ".join(
                f"{k}: ours {ours[k]!r} vs sd {theirs.get(k)!r}" for k in differences[:3]
            )
            bucket.append(
                f"turn {before.turn} [{' | '.join(a.describe(reg) for a in chosen)}] {detail}"
            )


def _score_port(
    reg: Regulation,
    node: Any,
    before: Position,
    chosen: list[SideAction],
    handle: Any,
    roll: int,
    agg: Aggregate,
    carry: _Carry,
) -> None:
    """The port on one turn, before Showdown is stepped past it.

    `diff_turn.compare_port_turn`'s rules: one request with branch 0, a refusal counted by
    its reason, a disagreement about stopping scored as a divergence on the field
    ``mid-turn interrupt``, and a turn both stopped inside at a replacement carried on
    with Showdown's replacements (`_carry_on`, IKA-227) and scored when it ends.
    """
    diff_turn.settle_port(carry.column, "a new turn began inside a paused one")
    if action_overriding_effects(handle.log, only_unmodelled=True):
        agg.skipped["action overridden mid-turn"] += 1
        return
    given = diff_turn.port_position(before.to_json())
    budget = Budget.deterministic(roll)
    reply = diff_turn.port_exchange(node, given, chosen, budget, select=0)
    if reply.get("refused") == "branch index out of range":
        # The turn stopped: `turn` hands the pause back with the weights (IKA-217).
        from pokeuraou import rustnode

        reply = diff_turn.port_turn_exchange(
            node,
            {
                "position": given.to_json(),
                "actions": [[rustnode.dump_action(a) for a in side.slots] for side in chosen],
                "budget": rustnode.dump_budget(budget),
            },
        )
    if reply.get("refused"):
        agg.skipped[f"refused: {diff_turn.refusal_reason(reply, given)}"] += 1
        return
    stopped = bool(reply.get("suspended"))
    if len(reply.get("branches", [])) + len(reply.get("suspended", [])) != 1:
        agg.skipped["not a single branch"] += 1
        return
    unmodelled = tuple(reply.get("unmodelled", []))
    showdown_stopped = diff_turn.showdown_paused_mid_turn(handle)
    if stopped and showdown_stopped:
        agg.paused += 1
        carry.column.pending = diff_turn.PortPending(
            where=carry.where,
            before=given,
            chosen=chosen,
            pause=reply["pause"],
            unmodelled=set(unmodelled),
            log=list(handle.log),
            cursor=len(carry.tap.steps),
        )
        return
    theirs = canonical(Position.from_json(handle.position))
    if stopped != showdown_stopped:
        ours = dict(theirs)
        ours["mid-turn interrupt"] = "port stopped" if stopped else "port carried on"
        theirs["mid-turn interrupt"] = "showdown carried on" if stopped else "showdown stopped"
        _tally(reg, given, chosen, ours, theirs, bool(unmodelled), agg)
        return
    if any(n.startswith("forceSwitch") for n in unmodelled):
        agg.skipped["pending replacement"] += 1
        return
    ours = canonical(Position.from_json(reply["position"]))
    _tally(reg, given, chosen, ours, theirs, bool(unmodelled), agg)


#: Ratios worth naming, and the modifier each implies is missing.
_NAMED_RATIOS: tuple[tuple[float, str], ...] = (
    (0.5, "we are half (missing a 2x)"),
    (0.67, "we are two thirds (missing a 1.5x)"),
    (0.77, "we are ~0.77 (missing a 1.3x)"),
    (0.83, "we are ~0.83 (missing a 1.2x)"),
    (1.2, "we are 1.2x too high"),
    (1.3, "we are 1.3x too high"),
    (1.5, "we are 1.5x too high"),
    (2.0, "we are 2x too high"),
)


def _record_damage_ratios(
    before: Position,
    ours: dict[str, Any],
    theirs: dict[str, Any],
    differences: list[str],
    agg: Aggregate,
) -> None:
    """Buckets how far off each HP divergence is, as a ratio of damage dealt."""
    start = {}
    for side_index, side in enumerate(before.sides):
        for mon in side.pokemon:
            start[f"p{side_index + 1}.{mon.slot}.hp"] = mon.hp
    for key in differences:
        if not key.endswith(".hp") or key not in start:
            continue
        our_damage = start[key] - ours[key]
        their_damage = start[key] - theirs.get(key, ours[key])
        if their_damage == 0:
            agg.damage_ratio["Showdown dealt none, we dealt some"] += 1
            continue
        if our_damage == 0:
            agg.damage_ratio["we dealt none, Showdown dealt some"] += 1
            continue
        ratio = our_damage / their_damage
        label = f"other ({ratio:.2f})"
        for value, name in _NAMED_RATIOS:
            if abs(ratio - value) <= 0.04:
                label = name
                break
        agg.damage_ratio[label] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--roll", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="processes playing seeds side by side (a seed is the unit); the report is the "
        "same as with 1",
    )
    args = ap.parse_args()
    agg = run(args.seeds, args.battles, args.roll, args.max_turns, jobs=args.jobs)
    print(agg.render(args.top, args.min_count))


if __name__ == "__main__":
    main()
