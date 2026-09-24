"""Seismic Toss and Night Shade (`damage: 'level'`) and Struggle's type and recoil in the port,
played against Showdown (IKA-239).

Showdown a5df827 (the champions mod overrides none of it)::

    data/moves.ts
      nightshade:  basePower: 0, damage: 'level', category: "Special", type: "Ghost"
      seismictoss: basePower: 0, damage: 'level', category: "Physical", type: "Fighting",
                   flags: { contact: 1, protect: 1, mirror: 1, nonsky: 1, metronome: 1 }
      struggle:    basePower: 50, category: "Physical", type: "Normal", target: "randomNormal",
                   onModifyMove(move, pokemon, target) { move.type = '???'; ... },
                   struggleRecoil: true
    sim/battle-actions.ts getDamage (1602-1612)
      if (!target.runImmunity(move, !suppressMessages)) return false;
      ...
      if (move.damage === 'level') { return source.level; }
    sim/battle-actions.ts applyRecoilDamage (1379-1389)
      if (move.struggleRecoil) recoilDamage = this.battle.clampIntRange(Math.round(pokemon.baseMaxhp / 4), 1);
      ...
      if (move.struggleRecoil) {
          this.battle.directDamage(recoilDamage, pokemon, pokemon, { id: 'strugglerecoil' } as Condition);

`getDamage` answers a level move after the type immunity and before the critical hit, the
rolls and `modifyDamage` (so no resist berry). `applyRecoilDamage` runs from
`hitStepMoveHitLoop` when `move.totalDamage` is truthy and from Substitute's `onTryPrimaryHit`
when the doll took something. `directDamage` runs no `Damage` event, so Rock Head and Magic
Guard do not stop it. Struggle's `onModifyMove` runs before the abilities' `ModifyType`, so it
is a `???` move: no immunity, no STAB, no Chilan Berry.

The port hit both level moves for 0 (base power 0, noted `move.basePowerCallback`) and struggled
as a Normal move without recoil (noted `damaging move: struggle`). Each case plays in Showdown
and holds the port's one outcome to it; the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"substitute", "taunt"})
#: What the port noted for these moves before it modelled them.
NOTED = ("move.basePowerCallback:seismictoss", "move.basePowerCallback:nightshade", "damaging move: struggle")


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None,
    level: int | None = None, **sp: int,
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread, level=level
    )


#: Machamp: 1 Seismic Toss, 2 Night Shade, 3 Protect, 4 Swords Dance.
LEVEL = ["seismictoss", "nightshade", "protect", "swordsdance"]
#: Snorlax knows only status moves: taunted, it can only Struggle.
STATUS_ONLY = ["protect", "swordsdance", "curse", "rest"]
#: The partner: 1 Protect, 2 Body Slam, 3 Helping Hand (the setup turns: a second Protect
#: in a row would be a 1-in-3 draw).
PARTNER = ["protect", "bodyslam", "helpinghand"]
#: Foe A: 1 Dragon Claw, 2 Flamethrower, 3 Protect, 4 Substitute.
FOE_A = ["dragonclaw", "flamethrower", "protect", "substitute"]
#: Foe B: 1 Taunt, 2 Flamethrower, 3 Protect, 4 Swords Dance.
FOE_B = ["taunt", "flamethrower", "protect", "swordsdance"]


def _teams(
    user: TeamSet,
    foe_a: str = "Garchomp",
    foe_a_ability: str = "Rough Skin",
    foe_a_item: str | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [user, _mon("Kangaskhan", "Inner Focus", PARTNER, 10, atk=32)]
    theirs = [
        _mon(foe_a, foe_a_ability, FOE_A, 30, foe_a_item),
        _mon("Incineroar", "Blaze", FOE_B, 32),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


def _machamp(level: int | None = None) -> TeamSet:
    return _mon("Machamp", "Inner Focus", LEVEL, 0, level=level)


def _snorlax(ability: str = "Thick Fat", hp: int = 20) -> TeamSet:
    return _mon("Snorlax", ability, STATUS_ONLY, 0, hp=hp)


TOSS, SHADE = "move 1 1, move 1", "move 2 1, move 1"
#: Garchomp claws the partner (which Protects); Incineroar Protects. Nothing reaches the user.
AWAY = "move 1 2, move 3"
#: Incineroar taunts the user; Garchomp claws the Protecting partner.
TAUNT = "move 1 2, move 1 1"
#: Snorlax's only choice under Taunt, beside a Protecting partner.
STRUGGLE = "move 1, move 1"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- the level moves (Machamp, level 50 unless said)
    "seismic-toss-deals-the-users-level": (
        _teams(_machamp()), [], [TOSS, AWAY], "|-damage|p2a: Garchomp",
    ),
    "night-shade-deals-the-users-level": (
        _teams(_machamp()), [], [SHADE, AWAY], "|-damage|p2a: Garchomp",
    ),
    "seismic-toss-at-level-37": (
        _teams(_machamp(level=37)), [], [TOSS, AWAY], "|-damage|p2a: Garchomp",
    ),
    # Night Shade is Ghost, 2x on a Psychic type: a resist berry is `onSourceModifyDamage`,
    # which the level never reaches.
    "night-shade-leaves-a-kasib-berry": (
        _teams(_machamp(), foe_a="Alakazam", foe_a_ability="Inner Focus", foe_a_item="Kasib Berry"),
        [], [SHADE, AWAY], "|-damage|p2a: Alakazam",
    ),
    "seismic-toss-into-a-substitute": (
        _teams(_machamp()), [["move 4, move 3 -1", "move 4, move 4"]], [TOSS, AWAY],
        "|-end|p2a: Garchomp|Substitute",
    ),
    "control-night-shade-into-a-normal-type": (
        _teams(_machamp(), foe_a="Snorlax", foe_a_ability="Thick Fat"), [], [SHADE, AWAY],
        "|-immune|p2a: Snorlax",
    ),
    "control-seismic-toss-into-a-ghost": (
        _teams(_machamp(), foe_a="Dragapult", foe_a_ability="Clear Body"), [], [TOSS, AWAY],
        "|-immune|p2a: Dragapult",
    ),
    # -- Struggle (Snorlax, taunted by Incineroar on the setup turn)
    "struggle-recoils-a-quarter-rounded": (
        _teams(_snorlax()), [["move 2, move 3 -1", TAUNT]], [STRUGGLE, AWAY], "|-damage|p1a: Snorlax",
    ),
    "struggle-recoils-a-quarter-of-an-even-hp": (
        _teams(_snorlax(hp=21)), [["move 2, move 3 -1", TAUNT]], [STRUGGLE, AWAY], "|-damage|p1a: Snorlax",
    ),
    "struggle-recoils-through-rock-head": (
        _teams(_snorlax("Rock Head")), [["move 2, move 3 -1", TAUNT]], [STRUGGLE, AWAY],
        "|-damage|p1a: Snorlax",
    ),
    # Garchomp without Rough Skin: the port lets Rough Skin through Magic Guard, which is
    # not this case's question (a separate gap).
    "struggle-recoils-through-magic-guard": (
        _teams(_snorlax("Magic Guard"), foe_a_ability="Sand Veil"), [["move 2, move 3 -1", TAUNT]],
        [STRUGGLE, AWAY],
        "|-damage|p1a: Snorlax",
    ),
    "struggle-hits-a-ghost": (
        _teams(_snorlax(), foe_a="Dragapult", foe_a_ability="Clear Body"), [["move 2, move 3 -1", TAUNT]],
        [STRUGGLE, AWAY], "|-damage|p2a: Dragapult",
    ),
    "struggle-leaves-a-chilan-berry": (
        _teams(_snorlax(), foe_a_item="Chilan Berry"), [["move 2, move 3 -1", TAUNT]], [STRUGGLE, AWAY],
        "|-damage|p2a: Garchomp",
    ),
    "struggle-into-a-substitute-recoils": (
        _teams(_snorlax()), [["move 2, move 3 -1", "move 4, move 1 1"]], [STRUGGLE, AWAY],
        "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    ),
    "control-struggle-into-protect-does-not-recoil": (
        _teams(_snorlax()), [["move 2, move 3 -1", TAUNT]], [STRUGGLE, "move 3, move 3"],
        "|-activate|p2a: Garchomp|move: Protect",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            mon.item,
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(
    oracle: Oracle, teams: tuple, setup: list, choices: list[str], policy: RandomnessPolicy
) -> tuple[Position, dict, list[str]]:
    mine, theirs = teams
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=policy)
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, after, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    teams, setup, choices, shown = CASES[name]
    before, theirs, log = _play(oracle, teams, setup, choices, RandomnessPolicy())
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) == 1, there.branches
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None
    rust_now = _state(picked.position)
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
    notes = [n for n in there.unmodelled if n in NOTED]
    assert notes == [], notes


def test_the_struggle_cases_cover_both_roundings(oracle: Oracle) -> None:
    """`Math.round(maxhp / 4)`: one Snorlax whose quarter rounds up (a floor would differ), and
    one whose quarter is whole, so the cases hold the rounding and not just the fraction."""
    remainders = set()
    for name in ("struggle-recoils-a-quarter-rounded", "struggle-recoils-a-quarter-of-an-even-hp"):
        teams, setup, choices, _shown = CASES[name]
        before, _after, _log = _play(oracle, teams, setup, choices, RandomnessPolicy())
        remainders.add(before.sides[0].pokemon[0].maxhp % 4)
    assert remainders & {2, 3} and 0 in remainders, remainders


def test_a_level_move_never_crits(reg, oracle: Oracle, bridged: None) -> None:  # noqa: ANN001
    """With every crit forced, Showdown's Body Slam crits and the Seismic Toss beside it does
    not (`getDamage` returns the level before the crit roll): Garchomp still loses 50. The port,
    enumerating crits, gives that same Garchomp in every branch."""
    teams = _teams(_machamp())
    choices = ["move 1 1, move 2 2", "move 1 2, move 4"]
    before, theirs, log = _play(oracle, teams, [], choices, RandomnessPolicy(crit=True))
    assert any(line.startswith("|-crit|p2b: Incineroar") for line in log), log
    assert not any(line.startswith("|-crit|p2a") for line in log), log
    lost = before.sides[1].pokemon[0].hp - theirs["p2.garchomp"][0]
    # Rough Skin answers the contact; the level is what Garchomp lost.
    assert lost == 50, lost
    node = rustnode.node_for(reg)
    assert node is not None
    crits = replace(BUDGET, enumerate_crit=True)
    actions = _actions(reg, before, choices)
    there = node.resolve(before, actions, crits)
    assert there is not None and len(there.branches) >= 2, there
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, crits, select=index)
        assert picked is not None and picked.position is not None
        assert _state(picked.position)["p2.garchomp"] == theirs["p2.garchomp"]


ROOT = Path(__file__).resolve().parents[1]


def _level_moves() -> set[str]:
    source = (ROOT / "rust" / "src" / "level_struggle.rs").read_text(encoding="utf-8")
    body = source[source.index("fn deals_level(") :]
    body = body[: body.index("\n}\n")]
    return set(re.findall(r'"([a-z0-9]+)"', body))


@pytest.mark.parametrize("regulation", ["gen9championsvgc2026regmb", "gen9championsvgc2026regmc"])
def test_every_damage_field_in_the_dump_is_a_level_move_the_port_knows(regulation: str) -> None:
    """`damage` is a field no code read: a new move with it (Dragon Rage's number, or another
    'level') shows up here before it is a wrong answer."""
    dump = json.loads((ROOT / "configs" / "regulations" / f"{regulation}.json").read_text(encoding="utf-8"))
    fixed = {m["id"]: m["damage"] for m in dump["moves"] if m.get("damage") is not None}
    assert fixed, "the dump lists no `damage` field at all"
    assert set(fixed.values()) == {"level"}, fixed
    assert set(fixed) == _level_moves(), sorted(set(fixed) ^ _level_moves())
