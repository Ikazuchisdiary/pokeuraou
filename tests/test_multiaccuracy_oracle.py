"""Each later hit of Triple Axel and Population Bomb rolls its own accuracy (IKA-235).

Showdown (`hitStepMoveHitLoop`, sim/battle-actions.ts 911-940, vendor a5df827; the
champions mod's copy is data/mods/champions/scripts.ts 482-511), for a `multiaccuracy` move
and every hit after the first::

    let accuracy = move.accuracy;
    const boostTable = [1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3];
    if (!move.ignoreAccuracy) { boost = clamp(user accuracy);
        if (boost > 0) accuracy *= boostTable[boost]; else accuracy /= boostTable[-boost]; }
    if (!move.ignoreEvasion) { boost = clamp(target evasion);
        if (boost > 0) accuracy /= boostTable[boost]; else if (boost < 0) accuracy *= boostTable[-boost]; }
    accuracy = this.battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy);
    if (!move.alwaysHit) { accuracy = runEvent('Accuracy', ...);
        if (accuracy !== true && !this.battle.randomChance(accuracy, 100)) break; }

The first hit is `hitStepAccuracy` (IKA-231: modifiers first, then one integer stage). The
later ones are floats, and `runEvent` applies its modifier chain only to a non-negative
integer, so a staged 67.5 keeps no Compound Eyes; `random(100) < 67.5` is 68%. A miss ends
the move with the hits so far. Skill Link (`onModifyMove`) deletes `multiaccuracy`.
Triple Axel's power is `20 * move.hit`.

The regulation dump did not carry `multiaccuracy` until this change, so the port's gate for it
never fired and the port read the first roll as all the hits. With the field dumped, the
exe before this change refuses these turns ("move field multiaccuracy"): the positive
control. The ordinary multi-hit move (Dual Wingbeat) and the single hit (Play Rough) are the
controls, and pass on it.

The oracle's `accuracy_script` answers the step's accuracy rolls in order, so a later hit
can miss after the first landed; each stop is played in Showdown and its target HP is the
port branch that must carry that stop's weight.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 32, "atk": 0, "def": 2, "spa": 0, "spd": 0, "spe": 32}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 32, "spe": 0}

#: move id -> (base accuracy, hits, rolls each hit, physical)
MOVES = {
    "tripleaxel": (90, 3, True, True),
    "populationbomb": (90, 10, True, True),
    "dualwingbeat": (90, 2, False, True),
    "playrough": (90, 1, False, True),
}
BOOST_TABLE = [1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3]


def _mon(species: str, ability: str, moves: list[str], sp: dict, item: str | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=dict(sp),
                   item=item)


@dataclasses.dataclass(frozen=True)
class Case:
    move: str
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
        helper = _mon("Whimsicott", "Infiltrator", ["defog", "protect", "calmmind"], BULKY)
        return [user, helper, *BENCH]

    def team_p2(self) -> list[TeamSet]:
        target = _mon("Snorlax", self.target_ability, ["doubleteam", "calmmind", "protect"],
                      BULKY, self.target_item)
        partner = _mon("Incineroar", self.weather, ["mudslap", "protect", "calmmind"], BULKY)
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


def _modifier(case: Case) -> int:
    physical = MOVES[case.move][3]
    modifier = 4096
    if case.ability == "Compound Eyes":
        modifier = _chain(modifier, 5325)
    if case.ability == "Hustle" and physical:
        modifier = _chain(modifier, 3277)
    if case.target_ability == "Snow Cloak" and case.weather == "Snow Warning":
        modifier = _chain(modifier, 3277)
    if case.item == "Wide Lens":
        modifier = _chain(modifier, 4505)
    if case.target_item == "Bright Powder":
        modifier = _chain(modifier, 3686)
    return modifier


def first_number(case: Case) -> int:
    """`hitStepAccuracy` (IKA-231)."""
    accuracy = _modify(MOVES[case.move][0], _modifier(case))
    boost = max(-6, min(6, max(-6, min(6, case.accuracy)) - case.evasion))
    if boost > 0:
        return accuracy * (3 + boost) // 3
    if boost < 0:
        return accuracy * 3 // (3 - boost)
    return accuracy


def later_number(case: Case) -> float:
    """The loop's roll before hit 2, 3, ...: float stages, the chain only on an integer."""
    accuracy = float(MOVES[case.move][0])
    boost = max(-6, min(6, case.accuracy))
    accuracy = accuracy * BOOST_TABLE[boost] if boost > 0 else accuracy / BOOST_TABLE[-boost]
    boost = max(-6, min(6, case.evasion))
    if boost > 0:
        accuracy /= BOOST_TABLE[boost]
    elif boost < 0:
        accuracy *= BOOST_TABLE[-boost]
    if accuracy >= 0 and accuracy == math.floor(accuracy):
        return _modify(int(accuracy), _modifier(case))
    return accuracy


def showdown_numbers(case: Case) -> list[float]:
    """Every accuracy roll of the move when each one hits."""
    if case.ability == "No Guard":
        return []
    _, hits, each, _ = MOVES[case.move]
    later = hits - 1 if each and case.ability != "Skill Link" else 0
    return [first_number(case)] + [later_number(case)] * later


def chance(number: float) -> float:
    """`random(100) < number` over 0-99."""
    return min(100, max(0, math.ceil(number))) / 100


def stop_weights(case: Case) -> list[float]:
    """P(the move lands exactly k hits), k = 0..hits, from the numbers."""
    hits = MOVES[case.move][1]
    numbers = showdown_numbers(case)
    chances = [chance(n) for n in numbers] + [1.0] * (hits - len(numbers))
    out = []
    reach = 1.0
    for k in range(hits):
        out.append(reach * (1 - chances[k]))
        reach *= chances[k]
    out.append(reach)
    return out


def _cases() -> dict[str, Case]:
    out: dict[str, Case] = {}
    for stage in (0, 1, 2, -1, -2, 6, -6):
        out[f"triple axel, accuracy {stage:+d}"] = Case("tripleaxel", accuracy=stage)
    for stage in (1, 2, -1, -3, 6):
        out[f"triple axel, evasion {stage:+d}"] = Case("tripleaxel", evasion=stage)
    for acc, eva in ((1, 1), (2, 1), (-1, -1), (-2, -4), (3, 1)):
        out[f"triple axel, accuracy {acc:+d} evasion {eva:+d}"] = Case("tripleaxel", acc, eva)
    ce, hustle, wl, bp = "Compound Eyes", "Hustle", "Wide Lens", "Bright Powder"
    snow, cloak = "Snow Warning", "Snow Cloak"
    mods = {
        "compound eyes": dict(ability=ce),
        "hustle": dict(ability=hustle),
        "wide lens": dict(item=wl),
        "bright powder": dict(target_item=bp),
        "snow cloak": dict(target_ability=cloak, weather=snow),
        "compound eyes + bright powder": dict(ability=ce, target_item=bp),
        "hustle + wide lens + bright powder": dict(ability=hustle, item=wl, target_item=bp),
    }
    for label, kw in mods.items():
        for acc, eva in ((0, 0), (-1, 0), (0, 1), (1, 0)):
            if "hustle" in label and acc == 1:
                # Coil's +1 Attack with Hustle into Thick Fat: the port's damage is one short
                # of Showdown's on a 60-power ice hit (Avalanche alone too), which is not
                # this rule; left to its own issue (records/IKA-235.md).
                continue
            out[f"{label}, triple axel, accuracy {acc:+d} evasion {eva:+d}"] = Case(
                "tripleaxel", acc, eva, **kw)
    out["skill link, triple axel"] = Case("tripleaxel", ability="Skill Link")
    out["skill link, triple axel, evasion +1"] = Case("tripleaxel", evasion=1, ability="Skill Link")
    out["no guard, triple axel, evasion +2"] = Case("tripleaxel", evasion=2, ability="No Guard")
    for acc, eva in ((0, 0), (0, 1), (-1, 0), (1, 1)):
        out[f"population bomb, accuracy {acc:+d} evasion {eva:+d}"] = Case("populationbomb", acc, eva)
    out["population bomb, compound eyes"] = Case("populationbomb", ability=ce)
    out["population bomb, skill link, evasion +1"] = Case("populationbomb", evasion=1,
                                                          ability="Skill Link")
    # Controls: one roll for the whole move.
    for move in ("dualwingbeat", "playrough"):
        for acc, eva in ((0, 0), (0, 1), (-1, 0)):
            out[f"control {move}, accuracy {acc:+d} evasion {eva:+d}"] = Case(move, acc, eva)
    out["control dualwingbeat, compound eyes, accuracy -1"] = Case("dualwingbeat", -1, ability=ce)
    return out


CASES = _cases()
#: The ones the fix changes: a later hit can miss (the old port landed every hit).
MOVED = sorted(n for n, c in CASES.items() if any(w > 0 for w in stop_weights(c)[1:-1]))


def _setup_choices(case: Case, turn: int) -> list[str]:
    p1a = "move 1" if turn < case.accuracy else "move 4"
    p1b = "move 1 1" if turn < -case.evasion else "move 3"
    p2a = "move 1" if turn < case.evasion else "move 2"
    p2b = "move 1 1" if turn < -case.accuracy else "move 3"
    return [f"{p1a}, {p1b}", f"{p2a}, {p2b}"]


ATTACK = ["move 2 1, move 2", "move 2, move 2"]


def _p(pos: Position, side: int, slot: int) -> Pokemon:
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _play(
    oracle: Oracle, case: Case, script: tuple[str, ...] = ()
) -> tuple[dict, dict, list[dict], list[str]]:
    handle = oracle.create(FORMAT_ID, case.team_p1(), case.team_p2(),
                           policy=RandomnessPolicy(damage_roll=0, accuracy="hit",
                                                   accuracy_script=script))
    handle.step(["team 1234", "team 1234"])
    for turn in range(max(abs(case.accuracy), abs(case.evasion))):
        handle.step(_setup_choices(case, turn))
        assert handle.choice_errors == [], handle.choice_errors
    before = handle.position
    handle.step(ATTACK)
    assert handle.choice_errors == [], handle.choice_errors
    after, rolls, log = handle.position, handle.rolls, list(handle.log)
    handle.close()
    staged = Position.from_json(before)
    assert _p(staged, 0, 0).boost("accuracy") == max(-6, min(6, case.accuracy))
    assert _p(staged, 1, 0).boost("evasion") == max(-6, min(6, case.evasion))
    return before, after, rolls, log


def _numbers(rolls: list[dict]) -> list[float]:
    return [r["numerator"] for r in rolls if r.get("kind") == "chance" and r.get("denominator") == 100]


def _scripts(case: Case) -> dict[int, tuple[str, ...]]:
    """k hits landed -> the accuracy answers that stop the move there (when they can)."""
    numbers = showdown_numbers(case)
    out: dict[int, tuple[str, ...]] = {MOVES[case.move][1]: ()}
    for k in range(len(numbers)):
        if chance(numbers[k]) >= 1.0:
            continue
        if k == 0 and case.accuracy < 0:
            continue  # the setup's Mud-Slap would take the scripted miss
        out[k] = ("hit",) * k + ("miss",)
    return out


def test_the_table_has_both_kinds() -> None:
    assert len(MOVED) >= 30 and len(CASES) - len(MOVED) >= 10, (len(MOVED), len(CASES))


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_rolls(oracle: Oracle, name: str) -> None:
    """The rolls Showdown asks for are the ones written out above, and the power climbs."""
    case = CASES[name]
    before, after, rolls, log = _play(oracle, case)
    assert _numbers(rolls) == showdown_numbers(case)
    maxhp = _p(Position.from_json(before), 1, 0).maxhp
    hp = [int(line.split("|")[3].split("/")[0]) for line in log
          if line.startswith("|-damage|p2a: Snorlax|") and len(line.split("|")) == 4
          and line.endswith(f"/{maxhp}")]
    assert len(hp) == MOVES[case.move][1], log
    assert _p(Position.from_json(after), 1, 0).hp == hp[-1] > 0
    start = _p(Position.from_json(before), 1, 0).hp
    dealt = [a - b for a, b in zip([start, *hp], hp, strict=False)]
    if case.move == "tripleaxel":
        # 20, 40, 60: each hit more than the one before.
        assert dealt[0] < dealt[1] < dealt[2], dealt
    if case.move == "populationbomb":
        assert max(dealt) - min(dealt) <= 1, dealt


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_branches

    case = CASES[name]
    weights = stop_weights(case)
    before, _, rolls, _ = _play(oracle, case)
    assert _numbers(rolls) == showdown_numbers(case)
    start = Position.from_json(before)
    start_hp = _p(start, 1, 0).hp
    expected: dict[int, float] = {}
    for k, script in _scripts(case).items():
        _, after, _, _ = _play(oracle, case, script)
        hp = _p(Position.from_json(after), 1, 0).hp
        assert (hp == start_hp) == (k == 0), (k, hp, start_hp)
        expected[hp] = expected.get(hp, 0.0) + weights[k]
    # The stops the scripts could not reach (a first-hit miss behind a Mud-Slap) leave the HP.
    if 0 not in _scripts(case) and weights[0] > 0:
        expected[start_hp] = expected.get(start_hp, 0.0) + weights[0]
    assert sum(expected.values()) == pytest.approx(1.0, abs=1e-12)

    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
    chosen = []
    for side, choice in enumerate(ATTACK):
        menu = {a.to_choice(): a for a in side_actions(reg, start, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    budget = replace(Budget.deterministic(0), enumerate_accuracy=True, max_branches=64)
    got: dict[int, float] = {}
    for weight, pos in port_branches(port, start, chosen, budget):
        got[_p(pos, 1, 0).hp] = got.get(_p(pos, 1, 0).hp, 0.0) + weight
    assert set(got) == {hp for hp, w in expected.items() if w > 0}, (got, expected)
    for hp, w in expected.items():
        assert got.get(hp, 0.0) == pytest.approx(w, abs=1e-12), (hp, got, expected)


@pytest.mark.parametrize("budget_name", ["matrix", "exact"])
def test_a_ten_hit_move_under_the_search_budgets(reg, oracle: Oracle, port, budget_name: str) -> None:  # noqa: ANN001
    """Population Bomb at 68% a later hit: eleven stops, which the matrix cap of 16 prunes
    with the damage rolls and crits on top; the weights still sum to one."""
    from ._port_showdown import port_weights

    case = CASES["population bomb, accuracy +0 evasion +1"]
    before, _, _, _ = _play(oracle, case)
    start = Position.from_json(before)
    for side in start.sides:
        for mon in side.pokemon:
            mon.trapped = False
    chosen = []
    for side, choice in enumerate(ATTACK):
        menu = {a.to_choice(): a for a in side_actions(reg, start, side)}
        chosen.append(menu[choice])
    budget = Budget.matrix() if budget_name == "matrix" else Budget.exact()
    reply = port_weights(port, start, chosen, budget)
    total = sum(reply["branches"]) + sum(reply.get("suspended") or [])
    assert total == pytest.approx(1.0, abs=1e-9)
    assert len(reply["branches"]) <= budget.max_branches
