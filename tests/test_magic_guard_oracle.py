"""Magic Guard in the port, played against Showdown (IKA-248).

Showdown a5df827 (the champions mod does not override it)::

    data/abilities.ts 2465
      magicguard: {
          onDamage(damage, target, source, effect) {
              if (effect.effectType !== 'Move') {
                  if (effect.effectType === 'Ability') this.add('-activate', source, ...);
                  return false;
              }
          },
    sim/battle-actions.ts 1389 (Struggle's recoil)
      this.battle.directDamage(recoilDamage, pokemon, pokemon, { id: 'strugglerecoil' } as Condition);

Rough Skin, Rocky Helmet, Spiky Shield, Life Orb, recoil and Stealth Rock are `this.damage`
with an ability, an item, a condition or the `'recoil'` condition as the effect, so the holder
takes none of it. The port stopped only the weather, Leech Seed and the status residuals; the
rest went through. Struggle's recoil is `directDamage`, which runs no `Damage` event: it still
lands, and is the control that the gate is not simply "no self-damage".

Each case plays in Showdown and holds the port's one outcome to it; the positive control is the
exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: Clefable: 1 Moonblast, 2 Play Rough (contact), 3 Protect, 4 Double-Edge (contact, 33% recoil).
CLEFABLE = ["moonblast", "playrough", "protect", "doubleedge"]
#: Snorlax knows only status moves: taunted, it can only Struggle.
STATUS_ONLY = ["protect", "swordsdance", "curse", "rest"]
#: The partner: 1 Protect, 2 Body Slam, 3 Helping Hand.
PARTNER = ["protect", "bodyslam", "helpinghand"]
#: Foe A: 1 Dragon Claw, 2 Spiky Shield, 3 Stealth Rock, 4 Swords Dance.
FOE_A = ["dragonclaw", "spikyshield", "stealthrock", "swordsdance"]
#: Foe B: 1 Toxic, 2 Leech Seed, 3 Taunt, 4 Swords Dance.
FOE_B = ["toxic", "leechseed", "taunt", "swordsdance"]


def _clefable(ability: str = "Magic Guard", item: str | None = None) -> TeamSet:
    return _mon("Clefable", ability, CLEFABLE, 0, item)


def _teams(
    lead: TeamSet,
    foe_ability: str = "Sand Veil",
    foe_item: str | None = None,
    bench: TeamSet | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [lead, _mon("Kangaskhan", "Inner Focus", PARTNER, 10, atk=32)]
    if bench is not None:
        mine.append(bench)
    theirs = [
        _mon("Garchomp", foe_ability, FOE_A, 30, foe_item),
        _mon("Incineroar", "Blaze", FOE_B, 32),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


#: Clefable attacks Garchomp beside a Helping Hand; both foes Swords Dance.
PLAY_ROUGH, MOONBLAST, DOUBLE_EDGE = "move 2 1, move 3 -1", "move 1 1, move 3 -1", "move 4 1, move 3 -1"
IDLE = "move 4, move 4"
#: Nothing reaches anyone: Clefable Protects, the partner Helps, both foes Swords Dance.
QUIET = ["move 3, move 3 -1", IDLE]
#: Snorlax's only choice under Taunt, beside a Protecting partner.
STRUGGLE = "move 1, move 1"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- what Magic Guard stops and the port let through (these fail on the old exe)
    "rough-skin": (_teams(_clefable(), foe_ability="Rough Skin"), [], [PLAY_ROUGH, IDLE],
                   "|-activate|p2a: Garchomp|ability: Rough Skin"),
    "rocky-helmet": (_teams(_clefable(), foe_item="Rocky Helmet"), [], [PLAY_ROUGH, IDLE],
                     "|-damage|p2a: Garchomp"),
    "life-orb": (_teams(_clefable(item="Life Orb")), [], [MOONBLAST, IDLE], "|-damage|p2a: Garchomp"),
    "recoil": (_teams(_clefable()), [], [DOUBLE_EDGE, IDLE], "|-damage|p2a: Garchomp"),
    "spiky-shield": (_teams(_clefable()), [], [PLAY_ROUGH, "move 2, move 4"],
                     "|-activate|p2a: Garchomp|move: Protect"),
    # Snorlax leads, Garchomp lays the rocks, Clefable comes in on the compared turn.
    "stealth-rock": (
        _teams(_mon("Snorlax", "Thick Fat", STATUS_ONLY, 0), bench=_clefable()),
        [["move 2, move 3 -1", "move 3, move 4"]], ["switch 3, move 3 -1", IDLE], "|switch|p1a: Clefable",
    ),
    # Struggle's recoil lands (directDamage); the Rough Skin beside it does not.
    "struggle-into-rough-skin": (
        _teams(_mon("Snorlax", "Magic Guard", STATUS_ONLY, 0), foe_ability="Rough Skin"),
        [["move 2, move 3 -1", "move 4, move 3 1"]], [STRUGGLE, IDLE],
        "|-activate|p2a: Garchomp|ability: Rough Skin",
    ),
    # -- what the port already stopped (pass on both exes)
    "control-sandstorm": (_teams(_clefable(), foe_ability="Sand Stream"), [], QUIET,
                          "|-weather|Sandstorm|[upkeep]"),
    # The setup turn badly poisons or seeds Clefable; the compared turn's log shows no residual,
    # so what shows the case is the position it starts from.
    "control-toxic": (_teams(_clefable()), [["move 1 2, move 3 -1", "move 4, move 1 1"]], QUIET,
                      lambda before: before.sides[0].pokemon[0].status == "tox"),
    "control-leech-seed": (_teams(_clefable()), [["move 1 2, move 3 -1", "move 4, move 2 1"]], QUIET,
                           lambda before: before.sides[0].pokemon[0].volatile("leechseed") is not None),
    # -- without Magic Guard the same Life Orb and Rough Skin land (pass on both exes)
    "control-life-orb-without-magic-guard": (
        _teams(_clefable("Unaware", "Life Orb")), [], [MOONBLAST, IDLE], "|-damage|p1a: Clefable",
    ),
    "control-rough-skin-without-magic-guard": (
        _teams(_clefable("Unaware"), foe_ability="Rough Skin"), [], [PLAY_ROUGH, IDLE],
        "|-damage|p1a: Clefable",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.fainted, mon.item, mon.status)
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, teams: tuple, setup: list, choices: list[str]) -> tuple[Position, dict, list[str]]:
    mine, theirs = teams
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 123" if len(mine) > 2 else "team 12", "team 12"])
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
    before, theirs, log = _play(oracle, teams, setup, choices)
    if callable(shown):
        assert shown(before), f"Showdown did not set up {name}"
    else:
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


def test_the_magic_guard_user_is_hurt_only_where_showdown_says(oracle: Oracle) -> None:
    """The cases ask the right question: in Showdown the Magic Guard user keeps its HP through
    the compared turn in every stopped case, and loses HP to Struggle's recoil and in the two
    cases without Magic Guard (so a gate that stopped everything would fail those)."""
    kept, lost = set(), set()
    for name, (teams, setup, choices, _shown) in CASES.items():
        before, after, _log = _play(oracle, teams, setup, choices)
        user = next(m for m in before.sides[0].pokemon if m.species in {"clefable", "snorlax"}
                    and (m.species == "clefable" or name.startswith("struggle")))
        (kept if after[f"p1.{user.species}"][0] == user.hp else lost).add(name)
    assert lost == {"struggle-into-rough-skin", "control-life-orb-without-magic-guard",
                    "control-rough-skin-without-magic-guard"}, (sorted(kept), sorted(lost))
