"""Life Orb, drain, freeze and what counts as a failed move (IKA-161, IKA-171).

Each case is played by Showdown first, and the Python turn and the port's turn are both
held to it -- the HP, the status and `moveLastTurnFailed` of every Pokemon.

IKA-161, `_after_move`:

- **Sheer Force and Life Orb.** Sheer Force's `onModifyMove` deletes a move's secondaries
  and sets `hasSheerForce` (vendor/pokemon-showdown/data/abilities.ts, `sheerforce`), and
  `useMoveInner` skips `AfterMoveSecondarySelf` for such a move -- Life Orb's recoil and
  Shell Bell's heal. Both engines took the recoil.
- **Drain.** `spreadDamage` (sim/battle.ts) heals `Math.round(targetDamage * drain)` per
  target, inside the damage step: a Matcha Gotcha that deals 55 and 37 heals 28 + 19, not
  `round(92 / 2)`; and a Drain Punch at full HP heals nothing *before* Rocky Helmet takes
  its sixth, where both engines took the helmet first and then healed it back.

IKA-171, `_can_act` and `_after_hit`:

- **Freeze.** The champions mod's `frz.onBeforeMove` returns at once for a `defrost` move
  (Scald, Flare Blitz), whose `onModifyMove` then clears the status; `onDamagingHit` thaws
  a target hit by a Fire move and `onAfterMoveSecondary` one hit by a `thawsTarget` move
  (data/conditions.ts). Neither engine had any of it.
- **Failures.** A status move that did nothing returns `false` -- a Recover at full HP, a
  Toxic into a Poison type, a second Sunny Day -- and Stomping Tantrum reads `false`; a
  move every target of which Protected is `NOT_FAIL`, which is not. Both engines had both
  backwards, and a Pokemon that switched out kept last turn's flag, which `clearVolatile`
  resets.
- **A hit for 0.** A hit into Endure at 1 HP deals 0 and is still a hit: Rough Skin and
  Life Orb both run. Python skipped Rough Skin, and the port skipped both -- it read
  Life Orb as "damage above 0" where Python reads "a hit reached a target".

The controls were right before and have to stay right.

The policy answers every chance roll the same way for the whole battle, and Ice Beam's
freeze is a chance roll, so the freeze cases play with chance rolls on -- which also thaws a
frozen Pokemon the moment it tries to move. Toxapex is kept faster than the Pokemon that
freezes it so that the next turn is the first it tries to move frozen.
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
    spread = {"hp": 32, "atk": 20, "def": 2, "spa": 20, "spd": 2, "spe": spe}
    spread.update(sp)
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread
    )


SYLVEON = _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 1)
GARCHOMP = _mon("Garchomp", "Sand Veil", ["earthquake", "protect", "rockslide", "dragonclaw"], 8)
PAWMOT = _mon("Pawmot", "Iron Fist", ["protect", "thunderpunch", "closecombat", "doubleshock"], 8)
FOES = [
    _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0, item="Rocky Helmet"),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
    _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0),
    _mon("Garchomp", "Sand Veil", ["earthquake", "protect", "rockslide", "dragonclaw"], 0),
]
#: Toxapex Scalds the slot-b Pokemon, which Protects; Milotic Protects.
QUIET = "move 3 2, move 2"
#: Toxapex Scalds and Milotic Ice Beams the slot-b Pokemon.
AT_B = "move 3 2, move 4 2"

SHEER = [
    _mon("Feraligatr", "Sheer Force", ["waterfall", "aquajet", "icepunch", "protect"], 20, "Life Orb"),
    SYLVEON, GARCHOMP, PAWMOT,
]
DRAIN = [
    _mon("Conkeldurr", "Guts", ["drainpunch", "protect", "machpunch", "icepunch"], 20),
    SYLVEON, GARCHOMP, PAWMOT,
]


def _matcha(spa: int) -> list[TeamSet]:
    return [
        _mon("Sinistcha", "Heatproof", ["matchagotcha", "protect", "calmmind", "shadowball"], 0, spa=spa),
        SYLVEON, GARCHOMP, PAWMOT,
    ]


#: Garchomp hits Sinistcha first; both take the Matcha Gotcha neutrally.
MATCHA_FOES = [
    _mon("Garchomp", "Sand Veil", ["dragonclaw", "protect", "rockslide", "earthquake"], 32, atk=32),
    _mon("Toxapex", "Regenerator", ["recover", "protect", "scald", "toxic"], 0),
    _mon("Arcanine", "Justified", ["flareblitz", "protect", "scald", "extremespeed"], 0),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
]
STATUS = [
    _mon("Toxapex", "Regenerator", ["recover", "toxic", "protect", "scald"], 0),
    _mon("Garchomp", "Sand Veil", ["stompingtantrum", "protect", "rockslide", "dragonclaw"], 10),
    SYLVEON, PAWMOT,
]
SUN = [
    _mon("Charizard", "Blaze", ["sunnyday", "protect", "heatwave", "airslash"], 20),
    SYLVEON, GARCHOMP, PAWMOT,
]
TERRAIN = [
    _mon("Metagross", "Clear Body", ["bulletpunch", "stompingtantrum", "protect", "ironhead"], 0),
    SYLVEON, GARCHOMP, PAWMOT,
]
TERRAIN_FOES = [
    _mon("Indeedee-F", "Psychic Surge", ["protect", "psychic", "followme", "helpinghand"], 0),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
    _mon("Arcanine", "Justified", ["flareblitz", "protect", "scald", "extremespeed"], 0),
    _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0),
]
FROZEN = [
    _mon("Toxapex", "Regenerator", ["scald", "recover", "protect", "flareblitz"], 32),
    SYLVEON, GARCHOMP, PAWMOT,
]
FREEZERS = [
    _mon("Abomasnow", "Soundproof", ["icebeam", "protect", "woodhammer", "earthpower"], 0, spa=0),
    _mon("Arcanine", "Justified", ["flareblitz", "protect", "scald", "extremespeed"], 30, atk=0, spa=0),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
    _mon("Garchomp", "Sand Veil", ["earthquake", "protect", "rockslide", "dragonclaw"], 0),
]
#: Toxapex's Recover fails at full HP, then Abomasnow's Ice Beam freezes it.
FREEZE = ["move 2, move 2", "move 1 1, move 2"]
ENDURE = [
    _mon(
        "Kingambit", "Defiant", ["kowtowcleave", "protect", "ironhead", "suckerpunch"], 20,
        "Life Orb", atk=32,
    ),
    SYLVEON, GARCHOMP, PAWMOT,
]
ENDURERS = [
    _mon("Garchomp", "Rough Skin", ["endure", "protect", "dragonclaw", "earthquake"], 0, hp=0),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "icebeam", "recover"], 0),
    _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0),
    _mon("Arcanine", "Justified", ["flareblitz", "protect", "scald", "extremespeed"], 0),
]
TO_ONE_HP = [
    ["move 1 1, move 2", "move 3 -2, move 4"],  # Kowtow Cleave, no Endure
    ["move 1 1, move 2", "move 1, move 4"],  # Endure: Garchomp is left at 1 HP
    ["move 2, move 2", "move 3 -2, move 4"],  # a turn off, so the next Endure is certain
]
CHANCE = RandomnessPolicy(secondary=True)

#: name -> (teams, turns played first, the turn compared, the policy, a Showdown log line
#: that shows the case, a line that must be absent or None).
CASES = {
    "sheer-force-waterfall-skips-life-orb": (
        (SHEER, FOES), [], ["move 1 1, move 1", QUIET], None,
        "|-damage|p1a: Feraligatr|160/192|[from] item: Rocky Helmet|[of] p2a: Toxapex",
        "[from] item: Life Orb",
    ),
    "sheer-force-ice-punch-skips-life-orb": (
        (SHEER, FOES), [], ["move 3 1, move 1", QUIET], None,
        "|move|p1a: Feraligatr|Ice Punch|p2a: Toxapex", "[from] item: Life Orb",
    ),
    "control-sheer-force-aqua-jet-takes-life-orb": (
        (SHEER, FOES), [], ["move 2 1, move 1", QUIET], None,
        "|-damage|p1a: Feraligatr|141/192|[from] item: Life Orb", None,
    ),
    "drain-punch-heals-before-rocky-helmet": (
        (DRAIN, FOES), [], ["move 1 1, move 1", QUIET], None,
        "|-damage|p1a: Conkeldurr|177/212|[from] item: Rocky Helmet|[of] p2a: Toxapex",
        "|-heal|p1a: Conkeldurr",
    ),
    "control-drain-punch-without-a-helmet": (
        (DRAIN, FOES), [], ["move 1 2, move 1", AT_B], None,
        "|move|p1a: Conkeldurr|Drain Punch|p2b: Milotic", None,
    ),
    # 55 and 37: 28 + 19 = 47, where round(92 / 2) = 46.
    "matcha-gotcha-rounds-each-target": (
        (_matcha(4), MATCHA_FOES), [], ["move 1, move 2", "move 1 1, move 1"], None,
        "|-heal|p1a: Sinistcha|147/", None,
    ),
    # 54 and 36: the same either way.
    "control-matcha-gotcha-even-damage": (
        (_matcha(0), MATCHA_FOES), [], ["move 1, move 2", "move 1 1, move 1"], None,
        "|-heal|p1a: Sinistcha", None,
    ),
    "recover-at-full-hp-fails": (
        (STATUS, FOES), [], ["move 1, move 2", AT_B], None, "|-fail|p1a: Toxapex|heal", None,
    ),
    "toxic-into-a-poison-type-fails": (
        (STATUS, FOES), [], ["move 2 1, move 2", AT_B], None, "|-immune|p2a: Toxapex", None,
    ),
    "second-sunny-day-fails": (
        (SUN, FOES), [["move 1, move 1", AT_B]], ["move 1, move 2", AT_B], None,
        "|-fail|p1a: Charizard", None,
    ),
    "toxic-into-protect-is-not-a-failure": (
        (STATUS, FOES), [], ["move 2 1, move 2", "move 1, move 4 2"], None,
        "|-activate|p2a: Toxapex|move: Protect", None,
    ),
    "switching-out-forgets-the-failure": (
        (FROZEN, FREEZERS), [FREEZE], ["switch 3, move 2", "move 2, move 4 2"], CHANCE,
        "|switch|p1a: Garchomp|", None,
    ),
    "control-priority-into-psychic-terrain-fails": (
        (TERRAIN, TERRAIN_FOES), [], ["move 1 1, move 2", "move 2 2, move 4 2"], None,
        "|-activate|p2a: Indeedee|move: Psychic Terrain", None,
    ),
    "control-attack-into-protect": (
        (STATUS, FOES), [], ["move 4 1, move 2", "move 1, move 4 2"], None,
        "|-activate|p2a: Toxapex|move: Protect", None,
    ),
    "control-recover-when-hurt": (
        (STATUS, FOES), [["move 4 2, move 2", "move 3 1, move 4 1"]], ["move 1, move 4 1", "move 1, move 2"],
        None, "|-heal|p1a: Toxapex|157/157", None,
    ),
    "control-toxic-lands": (
        (STATUS, FOES), [], ["move 2 2, move 2", AT_B], None, "|-status|p2b: Milotic|tox", None,
    ),
    "control-first-sunny-day": (
        (SUN, FOES), [], ["move 1, move 2", AT_B], None, "|-weather|SunnyDay", None,
    ),
    "frozen-scald-thaws-its-user": (
        (FROZEN, FREEZERS), [FREEZE], ["move 1 2, move 2", "move 2, move 4 2"], CHANCE,
        "|-curestatus|p1a: Toxapex|frz|[from] move: Scald", None,
    ),
    "frozen-flare-blitz-thaws-its-user": (
        (FROZEN, FREEZERS), [FREEZE], ["move 4 2, move 2", "move 2, move 4 2"], CHANCE,
        "|-curestatus|p1a: Toxapex|frz|[from] move: Flare Blitz", None,
    ),
    "flare-blitz-thaws-its-target": (
        (FROZEN, FREEZERS), [FREEZE], ["move 2, move 2", "move 2, move 1 1"], CHANCE,
        "|-curestatus|p1a: Toxapex|frz", None,
    ),
    "scald-thaws-its-target": (
        (FROZEN, FREEZERS), [FREEZE], ["move 2, move 2", "move 2, move 3 1"], CHANCE,
        "|-curestatus|p1a: Toxapex|frz", None,
    ),
    "endure-at-one-hp-is-still-a-hit": (
        (ENDURE, ENDURERS), TO_ONE_HP, ["move 1 1, move 2", "move 1, move 3 2"], None,
        "|-damage|p1a: Kingambit|72/207|[from] item: Life Orb", None,
    ),
    "control-endure-from-full": (
        (ENDURE, ENDURERS), TO_ONE_HP[:1], ["move 1 1, move 2", "move 1, move 3 2"], None,
        "|-damage|p1a: Kingambit|117/207|[from] item: Life Orb", None,
    ),
}

#: No crit, the maximum roll -- what the oracle's policy pins. Chance-based secondaries are
#: branched only where the policy lets them happen.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)
CHANCE_BUDGET = replace(Budget.exact(), enumerate_crit=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{slot}.{mon.species}": (mon.hp, mon.status, mon.move_last_turn_failed)
        for index, side in enumerate(pos.sides)
        for slot, mon in enumerate(side.pokemon)
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
    (team_a, team_b), setup, choices, policy, _shown, _absent = CASES[name]
    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=policy or RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    shown, absent = CASES[name][4], CASES[name][5]
    assert any(line.startswith(shown) for line in log), f"Showdown did not do what {name} says: {log}"
    assert absent is None or not any(absent in line for line in log), f"{name}: {absent} in {log}"
    return before, choices, theirs, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


def _budget(name: str) -> Budget:
    return CHANCE_BUDGET if CASES[name][3] is CHANCE else BUDGET


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, _log = _play(oracle, name)
    # Showdown's position carries each Pokemon's stats, and the port declines any position
    # with an override (it reads that as a Transform).
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    # Every move hits, as the policy has it. The rest stays enumerated: without the chance
    # policy every branch has to be Showdown's; with it, the one where the chances Showdown
    # took all happened has to be among them (IKA-210: the branch used to be picked by
    # Python's events, which the port does not keep).
    budget = replace(_budget(name), enumerate_accuracy=False)
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    assert there.branches, "no finished outcome"
    states = []
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, budget, select=index)
        assert picked is not None and picked.position is not None
        states.append(_state(picked.position))
    if CASES[name][3] is CHANCE:
        assert theirs in states, f"{name}: showdown {theirs} not among the port's {states}"
    else:
        for index, rust_now in enumerate(states):
            assert rust_now == theirs, f"{name}, branch {index}: showdown {theirs} != rust {rust_now}"
