"""The accuracy a move rolls against is Showdown's integer, not a product of floats (IKA-231).

Showdown (sim/battle-actions.ts `hitStepAccuracy`, vendor a5df827; the champions mod does
not override it)::

    accuracy = this.battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy);
    if (!move.ignoreAccuracy) boost = clampIntRange(boosts['accuracy'], -6, 6);
    if (!move.ignoreEvasion) boost = clampIntRange(boost - boosts['evasion'], -6, 6);
    if (boost > 0) accuracy = this.battle.trunc(accuracy * (3 + boost) / 3);
    else if (boost < 0) accuracy = this.battle.trunc(accuracy * 3 / (3 - boost));
    ...
    if (accuracy !== true && !this.battle.randomChance(accuracy, 100)) { miss }

The ModifyAccuracy handlers `chainModify` one 4096-based modifier (Compound Eyes 5325,
Hustle 3277 on a physical move and Snow Cloak 3277 at priority -1, Wide Lens 4505 and Bright
Powder 3686 at -2), which `runEvent` applies once with `modify` (a half rounded down).
`randomChance(n, 100)` is `random(100) < n`, so the chance is n/100.

The port multiplied floats (1.3, 0.8, 1.1, 0.9 and the stage ratio) and never truncated:
a 90% move at -1 was 67.5% where Showdown rolls against 67. An evasion-ignoring move also
skipped the user's own accuracy stage, which Showdown still reads.

Each case sets the stages with moves of the regulation (Coil, Mud-Slap on the user, Double
Team, Defog on the target; Calm Mind to idle) and then reads the numerator of the attack's
roll. The port's chance of the hit is the weight of its branches that took the target's
HP, which must be that numerator (capped at 100) over 100.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

#: No Attack points: +6 from Coil must not knock the target out (a faint is a pause).
FAST = {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 32, "spe": 0}

#: move id -> (base accuracy, physical)
MOVES = {
    "playrough": (90, True),
    "hydropump": (80, False),
    "fireblast": (85, False),
    "focusblast": (70, False),
    "rocktomb": (95, True),
    "irontail": (75, True),
    "hurricane": (70, False),
    "sacredsword": (100, True),
    "crunch": (100, True),
    "willowisp": (85, False),
}
IGNORES_EVASION = {"sacredsword"}


def _mon(species: str, ability: str, moves: list[str], sp: dict, item: str | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=dict(sp),
                   item=item)


@dataclasses.dataclass(frozen=True)
class Case:
    move: str
    #: The user's accuracy stage and the target's evasion stage when the move is used.
    accuracy: int = 0
    evasion: int = 0
    ability: str = "Inner Focus"
    item: str | None = None
    target_ability: str = "Thick Fat"
    target_item: str | None = None
    #: The target's partner's ability: the weather.
    weather: str = "Blaze"

    def team_p1(self) -> list[TeamSet]:
        user = _mon("Garchomp", self.ability, ["coil", self.move, "protect", "calmmind"], FAST,
                    self.item)
        helper = _mon("Whimsicott", "Infiltrator", ["defog", "protect", "calmmind"],
                      BULKY)
        return [user, helper, *BENCH]

    def team_p2(self) -> list[TeamSet]:
        target = _mon("Snorlax", self.target_ability, ["doubleteam", "calmmind", "protect"],
                      BULKY, self.target_item)
        partner = _mon("Incineroar", self.weather, ["mudslap", "protect", "calmmind"],
                       BULKY)
        return [target, partner, *BENCH]


BENCH = [
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"], BULKY),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], BULKY),
]


# -- Showdown's arithmetic, written out to check the reading against the oracle ---------------

def _chain(modifier: int, nxt: int) -> int:
    return (modifier * nxt + 2048) >> 12


def _modify(value: int, modifier: int) -> int:
    return (value * modifier + 2047) // 4096


def showdown_number(case: Case) -> int:
    base, physical = MOVES[case.move]
    if case.move == "hurricane" and case.weather == "Drought":
        base = 50
    modifier = 4096
    if case.ability == "Compound Eyes":
        modifier = _chain(modifier, 5325)
    if case.ability == "Hustle" and physical:
        modifier = _chain(modifier, 3277)
    if case.target_ability == "Snow Cloak" and case.weather == "Snow Warning":
        modifier = _chain(modifier, 3277)
    # The -2 pair in the holders' speed order: the user (Speed 32 points) is faster.
    if case.item == "Wide Lens":
        modifier = _chain(modifier, 4505)
    if case.target_item == "Bright Powder":
        modifier = _chain(modifier, 3686)
    accuracy = _modify(base, modifier)
    boost = max(-6, min(6, case.accuracy))
    if case.move not in IGNORES_EVASION:
        boost = max(-6, min(6, boost - case.evasion))
    if boost > 0:
        return accuracy * (3 + boost) // 3
    if boost < 0:
        return accuracy * 3 // (3 - boost)
    return accuracy


def old_port_chance(case: Case) -> float:
    """The port before IKA-231, for naming which cases the fix moves."""
    base, physical = MOVES[case.move]
    acc = 50.0 if case.move == "hurricane" and case.weather == "Drought" else float(base)
    if case.ability == "Compound Eyes":
        acc *= 1.3
    if case.ability == "Hustle" and physical:
        acc *= 0.8
    if case.item == "Wide Lens":
        acc *= 1.1
    if case.target_item == "Bright Powder":
        acc *= 0.9
    if case.target_ability == "Snow Cloak" and case.weather == "Snow Warning":
        acc = float((int(acc * 3277.0 + 2047.0)) // 4096)
    if case.move not in IGNORES_EVASION:
        stages = max(-6, min(6, case.accuracy - case.evasion))
        acc *= (3 + stages) / 3 if stages >= 0 else 3 / (3 - stages)
    return min(1.0, max(0.0, acc / 100))


def _cases() -> dict[str, Case]:
    out: dict[str, Case] = {}
    # Every stage of each side alone, on a 90% move.
    for stage in [*range(1, 7), *range(-6, 0)]:
        out[f"play rough, accuracy {stage:+d}"] = Case("playrough", accuracy=stage)
        out[f"play rough, evasion {stage:+d}"] = Case("playrough", evasion=stage)
    # Both sides: the user's stage is clamped first, then the difference.
    for acc, eva in [(2, 3), (-6, 6), (6, -6), (3, -4), (-2, -1), (1, 1)]:
        out[f"play rough, accuracy {acc:+d} evasion {eva:+d}"] = Case("playrough", acc, eva)
    # Other base accuracies.
    for move, stages in {
        "hydropump": [0, -1, -2, -4, 1],
        "fireblast": [-1, -3],
        "focusblast": [-1, 1],
        "rocktomb": [-1, -5],
        "irontail": [-1, -2],
        "crunch": [-1, 1],
        "willowisp": [0, -1, -2],
    }.items():
        for stage in stages:
            out[f"{move}, accuracy {stage:+d}"] = Case(move, accuracy=stage)
    # Sacred Sword ignores the target's evasion and still reads the user's accuracy.
    out["sacred sword, accuracy -1"] = Case("sacredsword", accuracy=-1)
    out["sacred sword, evasion +2"] = Case("sacredsword", evasion=2)
    out["sacred sword, accuracy -2 evasion +2"] = Case("sacredsword", accuracy=-2, evasion=2)
    # The modifiers, alone and chained, with and without a stage.
    ce, hustle = "Compound Eyes", "Hustle"
    wl, bp = "Wide Lens", "Bright Powder"
    snow, cloak = "Snow Warning", "Snow Cloak"
    mods = {
        "compound eyes": dict(ability=ce),
        "hustle": dict(ability=hustle),
        "wide lens": dict(item=wl),
        "bright powder": dict(target_item=bp),
        "snow cloak": dict(target_ability=cloak, weather=snow),
        "compound eyes + wide lens": dict(ability=ce, item=wl),
        "compound eyes + bright powder": dict(ability=ce, target_item=bp),
        "hustle + wide lens + bright powder": dict(ability=hustle, item=wl, target_item=bp),
        "compound eyes + snow cloak": dict(ability=ce, target_ability=cloak, weather=snow),
        "snow cloak + bright powder": dict(target_ability=cloak, weather=snow, target_item=bp),
        "compound eyes + wide lens + bright powder": dict(ability=ce, item=wl, target_item=bp),
    }
    for label, kw in mods.items():
        for move, stage in [("playrough", 0), ("fireblast", 0), ("focusblast", -1),
                            ("hydropump", 2), ("irontail", 0)]:
            out[f"{label}, {move}, accuracy {stage:+d}"] = Case(move, accuracy=stage, **kw)
    # Hurricane is 50 in the sun, before the modifiers.
    out["hurricane in sun"] = Case("hurricane", weather="Drought")
    out["hurricane in sun, accuracy -1"] = Case("hurricane", accuracy=-1, weather="Drought")
    out["hurricane in sun, compound eyes"] = Case("hurricane", ability=ce, weather="Drought")
    out["hurricane in sun, wide lens, evasion +1"] = Case("hurricane", evasion=1, item=wl,
                                                         weather="Drought")
    return out


CASES = _cases()
#: Which of CASES the fix changes: the port before it read another chance.
MOVED = sorted(n for n, c in CASES.items()
               if abs(old_port_chance(c) - min(showdown_number(c), 100) / 100) > 1e-9)


def _setup_choices(case: Case, turn: int) -> list[str]:
    p1a = "move 1" if turn < case.accuracy else "move 4"
    p1b = "move 1 1" if turn < -case.evasion else "move 3"
    p2a = "move 1" if turn < case.evasion else "move 2"
    p2b = "move 1 1" if turn < -case.accuracy else "move 3"
    return [f"{p1a}, {p1b}", f"{p2a}, {p2b}"]


ATTACK = ["move 2 1, move 2", "move 2, move 2"]


def _play(oracle: Oracle, case: Case) -> tuple[dict, dict, list[dict]]:
    handle = oracle.create(FORMAT_ID, case.team_p1(), case.team_p2(),
                           policy=RandomnessPolicy(damage_roll=0, accuracy="hit"))
    handle.step(["team 1234", "team 1234"])
    for turn in range(max(abs(case.accuracy), abs(case.evasion))):
        handle.step(_setup_choices(case, turn))
        assert handle.choice_errors == [], handle.choice_errors
    before = handle.position
    handle.step(ATTACK)
    assert handle.choice_errors == [], handle.choice_errors
    after, rolls = handle.position, handle.rolls
    handle.close()
    return before, after, rolls


def _p(pos: Position, side: int, slot: int) -> Pokemon:
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _numerator(rolls: list[dict]) -> int:
    hundreds = [r["numerator"] for r in rolls if r.get("kind") == "chance" and r.get("denominator") == 100]
    assert hundreds, rolls
    return hundreds[0]


def _staged(before: dict, case: Case) -> None:
    pos = Position.from_json(before)
    assert _p(pos, 0, 0).boost("accuracy") == max(-6, min(6, case.accuracy))
    assert _p(pos, 1, 0).boost("evasion") == max(-6, min(6, case.evasion))


def test_the_fix_moves_cases_and_leaves_some() -> None:
    """Both kinds are in the table: the ones the fix changes and the controls."""
    assert len(MOVED) >= 40 and len(CASES) - len(MOVED) >= 40, (len(MOVED), len(CASES))


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_accuracy(oracle: Oracle, name: str) -> None:
    case = CASES[name]
    before, _, rolls = _play(oracle, case)
    _staged(before, case)
    assert _numerator(rolls) == showdown_number(case)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown_accuracy(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_branches

    case = CASES[name]
    before, after, rolls = _play(oracle, case)
    chance = min(_numerator(rolls), 100) / 100
    start = Position.from_json(before)
    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
    chosen = []
    for side, choice in enumerate(ATTACK):
        menu = {a.to_choice(): a for a in side_actions(reg, start, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    budget = replace(Budget.deterministic(0), enumerate_accuracy=True, max_branches=8)
    start_hp = _p(start, 1, 0).hp
    hit_hp = _p(Position.from_json(after), 1, 0).hp
    assert hit_hp < start_hp
    by_hp: dict[int, float] = {}
    for weight, pos in port_branches(port, start, chosen, budget):
        by_hp[_p(pos, 1, 0).hp] = by_hp.get(_p(pos, 1, 0).hp, 0.0) + weight
    assert set(by_hp) <= {start_hp, hit_hp}, (by_hp, start_hp, hit_hp)
    assert by_hp.get(hit_hp, 0.0) == pytest.approx(chance, abs=1e-12), by_hp
    assert by_hp.get(start_hp, 0.0) == pytest.approx(1 - chance, abs=1e-12), by_hp
