"""Differential test of the damage calculator against Showdown.

Randomness is *pinned*, not merely seeded: the damage roll, crit, accuracy, secondaries,
multi-hit count and speed ties are all forced to chosen outcomes, so a hit has exactly one
correct damage number and the comparison is an equality test.

Teams are sampled from the metagame usage distribution rather than generated at random, so
the effects that get exercised are the ones that actually occur, at roughly the rate they
occur. Divergences are attributed to the named ability/item/move involved and reported
usage-weighted, which is what makes "what to implement next" a measurement rather than a
guess.

    uv run python tools/diff_damage.py --battles 200
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

from diff_turn import REFUSED_CHOICE, showdown_choice  # noqa: E402

from pokeuraou.actions import MoveAction, SwitchAction, side_actions  # noqa: E402
from pokeuraou.damage import (  # noqa: E402
    calculate,
    effective_damage,
    register_mega_stones,
)
from pokeuraou.moveinfo import (  # noqa: E402
    KNOWN_UNIMPLEMENTED_VARIABLE_BP,
    MoveContext,
)
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import (  # noqa: E402
    MetagamePrior,
    find_cached_chaos,
    load_chaos,
    sample_team,
)
from pokeuraou.regulation import Regulation, load_regulation, to_id  # noqa: E402
from pokeuraou.view import battler, field_state, move_hits_multiple  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------


def desplit(lines: list[str]) -> list[str]:
    """Resolves Showdown's ``|split|`` pairs to the omniscient (secret) version.

    After ``|split|<side>`` the next line is the version only that side sees, which
    carries exact HP, and the line after is the public version with HP as a percentage.
    Comparing damage needs the exact one.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|split|"):
            if i + 1 < len(lines):
                out.append(lines[i + 1])
            i += 3  # split marker, secret line, public line
            continue
        out.append(lines[i])
        i += 1
    return out


def parse_ident(ident: str) -> tuple[int, int]:
    """``p2a: Sinistcha`` -> (side index, active slot index)."""
    who = ident.split(":")[0].strip()
    side = 0 if who.startswith("p1") else 1
    slot = "abc".index(who[2]) if len(who) > 2 and who[2] in "abc" else 0
    return side, slot


@dataclass(slots=True)
class Hit:
    attacker: tuple[int, int]
    move_id: str
    target: tuple[int, int]
    new_hp: int
    crit: bool
    spread: bool
    immune: bool


def first_move_hits(lines: list[str]) -> tuple[list[Hit], str | None]:
    """Damage dealt by the first move of the turn.

    Only the first move is compared, and only when nothing changed the field before it:
    the pre-turn position is then exactly the state the hit was computed against. Turns
    with a switch, a form change or an ability trigger before the first move are skipped
    rather than compared against a state we would have to reconstruct.
    """
    hits: list[Hit] = []
    seen_move = False
    attacker: tuple[int, int] | None = None
    move_id = ""
    spread = False
    crit_targets: set[tuple[int, int]] = set()

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if not seen_move:
            if tag in ("switch", "drag", "replace", "detailschange", "-formechange", "-mega"):
                return [], f"state change before first move: {tag}"
            if tag == "move":
                seen_move = True
                attacker = parse_ident(parts[2])
                move_id = to_id(parts[3])
                spread = any(p.startswith("[spread]") for p in parts)
                continue
            continue

        if tag == "move":
            break  # only the first move is compared
        if tag == "-crit":
            crit_targets.add(parse_ident(parts[2]))
            continue
        if tag == "-immune":
            assert attacker is not None
            hits.append(
                Hit(attacker, move_id, parse_ident(parts[2]), 0, False, spread, immune=True)
            )
            continue
        if tag == "-damage":
            if any(p.startswith("[from]") for p in parts):
                continue  # residual damage (item, weather, status), not the move
            target = parse_ident(parts[2])
            hp_text = parts[3].split(" ")[0]
            new_hp = 0 if hp_text.startswith("0") else int(hp_text.split("/")[0])
            assert attacker is not None
            hits.append(
                Hit(attacker, move_id, target, new_hp, target in crit_targets, spread, False)
            )
            continue
        if tag in ("-miss", "-fail", "-activate", "-block", "-supereffective", "-resisted"):
            continue
        if tag in ("-heal", "-status", "-boost", "-unboost", "-sethp", "-end", "-start"):
            continue

    return hits, None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class Report:
    compared: int = 0
    matched: int = 0
    #: A divergence the calculator declared first (it reported an unmodelled effect on
    #: this very hit) versus one it produced silently. Only the second kind can quietly
    #: corrupt a printed number, which is the split the turn harness has always used and
    #: this one was missing.
    flagged: int = 0
    silent: int = 0
    skipped_turns: Counter[str] = field(default_factory=Counter)
    #: (move, attacker ability, attacker item, defender ability, defender item) -> count
    divergences: Counter[tuple[str, ...]] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    unmodelled: Counter[str] = field(default_factory=Counter)
    #: Absolute error histogram, for telling "off by one rounding" from "wrong formula".
    error_sizes: Counter[int] = field(default_factory=Counter)
    #: When set, the first few divergent positions are written here for offline debugging.
    dump_dir: Path | None = None

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1.0 - self.matched / self.compared

    @property
    def silent_rate(self) -> float:
        return 0.0 if not self.compared else self.silent / self.compared

    def render(self, prior_usage: dict[str, float] | None = None) -> str:
        out = [
            f"compared {self.compared} hits, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}% "
            f"(silent {self.silent_rate * 100:.3f}%, flagged {self.flagged})"
        ]
        if self.error_sizes:
            sizes = ", ".join(
                f"{'exact' if k == 0 else f'+/-{k}'}: {v}"
                for k, v in sorted(self.error_sizes.items())[:8]
            )
            out.append(f"  error sizes: {sizes}")
        if self.skipped_turns:
            out.append(
                "  skipped turns: "
                + ", ".join(f"{k} x{v}" for k, v in self.skipped_turns.most_common(6))
            )
        if self.divergences:
            out.append("  divergences by (move / atk ability / atk item / def ability / def item):")
            ranked = self.divergences.most_common(20)
            if prior_usage:
                ranked = sorted(
                    self.divergences.items(),
                    key=lambda kv: -(kv[1] * _usage_weight(kv[0], prior_usage)),
                )[:20]
            for key, count in ranked:
                out.append(f"    {count:5d}  {' / '.join(k or '-' for k in key)}")
        if self.unmodelled:
            out.append("  effects present but not modelled (by occurrence):")
            for name, count in self.unmodelled.most_common(20):
                out.append(f"    {count:5d}  {name}")
        return "\n".join(out)


def _usage_weight(key: tuple[str, ...], usage: dict[str, float]) -> float:
    return max((usage.get(k, 0.0) for k in key), default=0.0) or 1.0


def compare_turn(
    reg: Regulation,
    pos: Position,
    lines: list[str],
    roll: int,
    report: Report,
) -> None:
    hits, skip = first_move_hits(desplit(lines))
    if skip:
        report.skipped_turns[skip] += 1
        return
    if not hits:
        report.skipped_turns["no damage"] += 1
        return

    fs = field_state(pos, reg)
    for hit in hits:
        a_side, a_slot = hit.attacker
        d_side, d_slot = hit.target
        a_party = pos.sides[a_side].active[a_slot]
        d_party = pos.sides[d_side].active[d_slot]
        if a_party is None or d_party is None:
            report.skipped_turns["empty slot"] += 1
            continue
        a_mon = pos.sides[a_side].pokemon[a_party]
        d_mon = pos.sides[d_side].pokemon[d_party]
        if hit.move_id not in reg.moves:
            report.skipped_turns[f"unknown move {hit.move_id}"] += 1
            continue
        move = reg.moves[hit.move_id]
        if move.raw.get("multihit") or move.raw.get("ohko") or move.raw.get("damage"):
            report.skipped_turns["fixed/multi-hit damage"] += 1
            continue
        if hit.move_id in KNOWN_UNIMPLEMENTED_VARIABLE_BP:
            # Declared unimplemented rather than quietly compared and counted as a miss.
            report.skipped_turns[f"unimplemented base power: {hit.move_id}"] += 1
            report.unmodelled[f"move.basePowerCallback:{hit.move_id}"] += 1
            continue

        atk = battler(reg, a_mon)
        dfn = battler(reg, d_mon)
        live_foes = sum(
            1
            for m in pos.sides[d_side].active_pokemon()
            if m is not None and not m.fainted
        )
        # This is the first move of the turn, so nothing has been hurt yet and no move has
        # landed on the attacker: the two "earlier this turn" flags are false by
        # construction rather than by assumption.
        move_ctx = MoveContext(
            weather=pos.field.weather,
            terrain=pos.field.terrain,
            side_total_fainted=sum(1 for m in pos.sides[a_side].pokemon if m.fainted),
            times_attacked=a_mon.times_attacked,
            target_hurt_this_turn=False,
            damaged_by_target=False,
        )
        result = calculate(
            reg,
            atk,
            dfn,
            hit.move_id,
            fs,
            defender_side=d_side,
            spread=hit.spread and move_hits_multiple(reg, hit.move_id, live_foes),
            crit=hit.crit,
            move_ctx=move_ctx,
        )
        for name in result.unmodelled:
            report.unmodelled[name] += 1

        expected = 0 if hit.immune else d_mon.hp - hit.new_hp
        ours = 0 if result.immune else int(effective_damage(result, dfn)[0, roll])

        report.compared += 1
        err = abs(ours - expected)
        report.error_sizes[min(err, 99)] += 1
        if err == 0:
            report.matched += 1
            continue

        key = (
            hit.move_id,
            a_mon.ability,
            a_mon.item or "",
            d_mon.ability,
            d_mon.item or "",
        )
        report.divergences[key] += 1
        if result.unmodelled:
            report.flagged += 1
        else:
            report.silent += 1
        if report.dump_dir is not None and len(report.examples) < 8:
            import json as _json

            report.dump_dir.mkdir(parents=True, exist_ok=True)
            path = report.dump_dir / f"div{len(report.examples):02d}_{hit.move_id}.json"
            with path.open("w", encoding="utf-8") as fh:
                _json.dump(
                    {
                        "position": pos.to_json(),
                        "attacker": list(hit.attacker),
                        "target": list(hit.target),
                        "move": hit.move_id,
                        "crit": hit.crit,
                        "spread": hit.spread,
                        "roll": roll,
                        "showdown_damage": expected,
                        "our_damage": ours,
                        "log": lines,
                    },
                    fh,
                    indent=1,
                )
        if len(report.examples) < 25:
            report.examples.append(
                f"{a_mon.species}({a_mon.ability},{a_mon.item}) {hit.move_id} -> "
                f"{d_mon.species}({d_mon.ability},{d_mon.item}) "
                f"hp {d_mon.hp}/{d_mon.maxhp} crit={hit.crit} spread={hit.spread} "
                f"eff={result.effectiveness}: ours {ours}, showdown {expected}"
            )


def run(
    battles: int,
    roll: int,
    seed: int,
    max_turns: int,
    dump_dir: Path | None = None,
    quiet: bool = True,
) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit(
            "No cached usage stats. Run: uv run python tools/fetch_priors.py --month 2026-08"
        )
    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    report = Report(dump_dir=dump_dir)

    policy = RandomnessPolicy(
        damage_roll=roll,
        accuracy="hit",
        crit=False,
        secondary=False,
        multihit="min",
        speed_tie="keep",
    )

    with Oracle() as oracle:
        for _ in range(battles):
            teams = [
                [
                    TeamSet.from_json(s.to_team_set_json(reg))
                    for s in sample_team(rng, reg, prior)
                ]
                for _ in range(2)
            ]
            handle = oracle.create(FORMAT_ID, teams[0], teams[1], seed=(
                int(rng.integers(1, 60000)), int(rng.integers(1, 60000)),
                int(rng.integers(1, 60000)), int(rng.integers(1, 60000)),
            ), policy=policy)
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])

            for _ in range(max_turns):
                pos = Position.from_json(handle.position)
                if pos.ended:
                    break
                requests = handle.requests
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
                    # Prefer damaging moves and never switch: a switch resolves before
                    # moves and would change the state the hit is computed against.
                    actions = side_actions(reg, pos, side_index, allow_switch=False)

                    def _damaging(slot) -> bool:  # noqa: ANN001
                        # `.get`: a recharging Pokemon's only action is a fake move with
                        # no dex entry, and it is certainly not a damaging one.
                        if not isinstance(slot, MoveAction):
                            return False
                        move = reg.moves.get(slot.move_id)
                        return move is not None and move.category != "Status"

                    damaging = [
                        a
                        for a in actions
                        if all(_damaging(s) for s in a.slots)
                        and not any(isinstance(s, SwitchAction) for s in a.slots)
                    ]
                    pick = py_rng.choice(damaging or actions)
                    # Numbered by Showdown's request: a locked move is `move 1` (IKA-316).
                    choices.append(showdown_choice(pick, request))

                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    # Until IKA-316 this was not looked at: the refused turn was compared as
                    # an empty log, and the same refused choice came back every turn after.
                    report.skipped_turns[REFUSED_CHOICE] += 1
                    break
                if not forced:
                    compare_turn(reg, pos, handle.log, roll, report)
            handle.close()

    if not quiet:
        _print_report(report, prior)
    return report


def _print_report(report: Report, prior: MetagamePrior) -> None:
    report_usage = {}
    for sp in prior.species.values():
        for table in (sp.abilities, sp.items, sp.moves):
            for k, v in table.items():
                report_usage[k] = report_usage.get(k, 0.0) + v * sp.usage
    print(report.render(report_usage))
    if report.examples:
        print("\nfirst divergences:")
        for line in report.examples:
            print("  " + line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=60)
    ap.add_argument("--roll", type=int, default=0, help="damage roll index, 0 = 100%%")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--dump-dir", type=Path, default=None, help="write divergent positions here")
    args = ap.parse_args()
    run(args.battles, args.roll, args.seed, args.max_turns, args.dump_dir, quiet=False)


if __name__ == "__main__":
    main()
