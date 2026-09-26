"""Three rules the port applied to foes only, played against Showdown (IKA-313, 314, 315).

Showdown a5df827 (the champions mod overrides none of them). IKA-309 met each in
diff_turn once its charging-move fix let the battles run on.

IKA-313, Lightning Rod and Storm Drain (data/abilities.ts)::

    lightningrod: {
        onAnyRedirectTarget(target, source, source2, move) {
            if (move.type !== 'Electric' || move.flags['pledgecombo']) return;
            const redirectTarget =
                ['randomNormal', 'adjacentFoe'].includes(move.target) ? 'normal' : move.target;
            if (this.validTarget(this.effectState.target, source, redirectTarget)) {
                ...
                return this.effectState.target;
            }
        },
    },

`onAny`: the holder draws the move whichever side it is on, the user's partner included,
and the user itself is no valid target. Two holders are met fastest first (`speedSort`).
Follow Me and Rage Powder are `onFoeRedirectTarget` -- the user's foes only -- but the
event's target is the user, so they draw a move aimed at the user's own partner too.

IKA-314, Wide Guard (data/moves.ts)::

    wideguard: { sideCondition: 'wideguard', condition: {
        onTryHit(target, source, move) {
            if (move?.target !== 'allAdjacent' && move.target !== 'allAdjacentFoes') return;
            if (this.checkMoveBypassesProtect(move, source, target)) return;
            ...
            return this.NOT_FAIL;
        },
    } },

A side condition's `onTryHit` runs for any target on that side, so a partner's Earthquake
is stopped too. Quick Guard's `onTryHit` has the same shape (`move.priority <= 0.1`).

IKA-315, Roost (data/moves.ts)::

    roost: { self: { volatileStatus: 'roost' }, condition: {
        duration: 1, onResidualOrder: 25,
        onType(types, pokemon) { return types.filter(type => type !== 'Flying'); },
    } },

For the rest of the turn the user is not a Flying type: type effectiveness, and
`isGrounded` (`this.hasType('Flying')`).

Each case plays in Showdown and holds the port's one outcome to it. The positive control
is the exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`); the
`control-` cases pass on it too.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, SideAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

STATS = ("atk", "def", "spa", "spd", "spe")


def _mon(species: str, ability: str, moves: list[str], spe: int = 10) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


#: The user: 1 Thunderbolt, 2 Scald, 3 Electroweb, 4 Earthquake.
USER = ["thunderbolt", "scald", "electroweb", "earthquake"]
#: The user with Extreme Speed in place of Earthquake.
QUICK_USER = ["thunderbolt", "scald", "electroweb", "extremespeed"]
#: The partner: 1 Swords Dance, 2 Wide Guard, 3 Quick Guard, 4 Follow Me.
PARTNER = ["swordsdance", "wideguard", "quickguard", "followme"]
#: The foes: 1 Swords Dance, 2 Follow Me, 3 Wide Guard, 4 Thunderbolt.
FOE = ["swordsdance", "followme", "wideguard", "thunderbolt"]
#: The roosting foe: 1 Swords Dance, 2 Roost, 3 Protect, 4 Follow Me.
ROOSTER = ["swordsdance", "roost", "protect", "followme"]


def _redirect_teams(
    partner: tuple[str, str] = ("Kommo-o", "Bulletproof"),
    foe_b: tuple[str, str] = ("Incineroar", "Blaze"),
    user_ability: str = "Static",
    user_moves: list[str] | None = None,
    partner_spe: int = 10,
    foe_b_spe: int = 10,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon("Pikachu", user_ability, user_moves or USER, 32),
        _mon(partner[0], partner[1], PARTNER, partner_spe),
    ]
    theirs = [
        _mon("Kommo-o", "Bulletproof", FOE, 0),
        _mon(foe_b[0], foe_b[1], FOE, foe_b_spe),
        _mon("Milotic", "Marvel Scale", FOE, 0),
    ]
    return mine, theirs


def _roost_teams(rooster_spe: int = 32) -> tuple[list[TeamSet], list[TeamSet]]:
    """Corviknight (Flying/Steel) roosts against Pikachu's Thunderbolt and Swampert's
    Earth Power; Kommo-o is the Corviknight's partner. At 32 Speed the Corviknight moves
    before both, at 0 after Pikachu."""
    mine = [
        _mon("Pikachu", "Static", ["thunderbolt", "earthpower", "swordsdance", "protect"], 0),
        _mon("Swampert", "Damp", ["earthpower", "thunderbolt", "swordsdance", "protect"], 0),
    ]
    theirs = [
        _mon("Corviknight", "Pressure", ROOSTER, rooster_spe),
        _mon("Kommo-o", "Bulletproof", ROOSTER, 0),
        _mon("Milotic", "Marvel Scale", ROOSTER, 0),
    ]
    return mine, theirs


QUIET = "move 1, move 1"
#: Pikachu's Thunderbolt on the Corviknight, Swampert's Swords Dance.
ZAP = "move 1 1, move 3"
#: Pikachu's Swords Dance, Swampert's Earth Power on the Corviknight.
QUAKE = "move 3, move 1 1"
#: The Corviknight roosts, its partner dances.
ROOST = "move 2, move 1"
#: A turn that takes some HP off the Corviknight first, so Roost has something to heal.
HURT = [ZAP, QUIET]

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- IKA-313: the redirecting abilities draw a partner's move
    "lightning-rod-draws-a-partners-thunderbolt": (
        _redirect_teams(partner=("Raichu", "Lightning Rod")), [], ["move 1 1, move 1", QUIET],
        "|-activate|p1b: Raichu|ability: Lightning Rod",
    ),
    "storm-drain-draws-a-partners-scald": (
        _redirect_teams(partner=("Milotic", "Storm Drain")), [], ["move 2 1, move 1", QUIET],
        "|-activate|p1b: Milotic|ability: Storm Drain",
    ),
    "a-foes-lightning-rod-draws-a-move-aimed-at-the-partner": (
        _redirect_teams(foe_b=("Raichu", "Lightning Rod")), [], ["move 1 -2, move 1", QUIET],
        "|-activate|p2b: Raichu|ability: Lightning Rod",
    ),
    "follow-me-draws-a-move-aimed-at-the-partner": (
        _redirect_teams(), [], ["move 2 -2, move 1", "move 2, move 1"],
        "|move|p1a: Pikachu|Scald|p2a: Kommo-o",
    ),
    "the-faster-lightning-rod-draws-it-the-partner": (
        _redirect_teams(partner=("Raichu", "Lightning Rod"), foe_b=("Incineroar", "Lightning Rod"),
                        partner_spe=20, foe_b_spe=0),
        [], ["move 1 2, move 1", QUIET], "|-activate|p1b: Raichu|ability: Lightning Rod",
    ),
    "control-the-faster-lightning-rod-draws-it-the-foe": (
        _redirect_teams(partner=("Milotic", "Lightning Rod"), foe_b=("Raichu", "Lightning Rod"),
                        partner_spe=0, foe_b_spe=20),
        [], ["move 1 1, move 1", QUIET], "|-activate|p2b: Raichu|ability: Lightning Rod",
    ),
    "control-a-partners-lightning-rod-leaves-a-spread-move": (
        _redirect_teams(partner=("Raichu", "Lightning Rod")), [], ["move 3, move 1", QUIET],
        "|move|p1a: Pikachu|Electroweb",
    ),
    "control-the-users-own-lightning-rod-draws-nothing": (
        _redirect_teams(user_ability="Lightning Rod"), [], ["move 1 1, move 1", QUIET],
        "|move|p1a: Pikachu|Thunderbolt|p2a: Kommo-o",
    ),
    "control-a-partners-lightning-rod-leaves-a-water-move": (
        _redirect_teams(partner=("Raichu", "Lightning Rod")), [], ["move 2 1, move 1", QUIET],
        "|move|p1a: Pikachu|Scald|p2a: Kommo-o",
    ),
    # -- IKA-314: the user's own side's guards stop the partner's move
    "wide-guard-stops-a-partners-earthquake": (
        _redirect_teams(partner=("Raichu", "Static")), [], ["move 4, move 2", QUIET],
        "|-activate|p1b: Raichu|move: Wide Guard",
    ),
    "quick-guard-stops-a-partners-extreme-speed": (
        _redirect_teams(partner=("Raichu", "Static"), user_moves=QUICK_USER), [],
        ["move 4 -2, move 3", QUIET], "|-activate|p1b: Raichu|move: Quick Guard",
    ),
    "control-a-foes-wide-guard-leaves-the-partner-hit": (
        _redirect_teams(partner=("Raichu", "Static")), [], ["move 4, move 1", "move 3, move 1"],
        "|-activate|p2a: Kommo-o|move: Wide Guard",
    ),
    "control-wide-guard-leaves-a-partners-single-target-move": (
        _redirect_teams(partner=("Raichu", "Static")), [], ["move 2 -2, move 2", QUIET],
        "|move|p1a: Pikachu|Scald|p1b: Raichu",
    ),
    "control-no-wide-guard-earthquake-hits-the-partner": (
        _redirect_teams(partner=("Raichu", "Static")), [], ["move 4, move 1", QUIET],
        "|move|p1a: Pikachu|Earthquake",
    ),
    # -- IKA-315: Roost takes the Flying type away for the rest of the turn
    "roost-drops-the-flying-weakness": (
        _roost_teams(), [HURT], [ZAP, ROOST], "|-singleturn|p2a: Corviknight|move: Roost",
    ),
    "roost-grounds-for-a-ground-move": (
        _roost_teams(), [HURT], [QUAKE, ROOST], "|-singleturn|p2a: Corviknight|move: Roost",
    ),
    "control-a-hit-before-roost": (
        _roost_teams(rooster_spe=0), [HURT], [ZAP, ROOST], "|-singleturn|p2a: Corviknight|move: Roost",
    ),
    "control-flying-again-the-next-turn": (
        _roost_teams(), [HURT, ["move 3, move 3", ROOST]], [ZAP, QUIET], "|move|p1a: Pikachu|Thunderbolt",
    ),
    "control-roost-at-full-hp": (
        _roost_teams(), [], [ZAP, ROOST], "|move|p2a: Corviknight|Roost",
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
            tuple(mon.boosts.get(stat, 0) for stat in STATS) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
    (mine, theirs), setup, choices, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
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
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, choices, after, log


def _actions(reg, pos: Position, choices: list[str]) -> list[SideAction]:  # noqa: ANN001
    """The menu's action where there is one. A move aimed at the user's own partner is not
    in the menu (IKA-181), so that one is built from the choice string."""
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        if choices[side] in menu:
            out.append(menu[choices[side]])
            continue
        slots = []
        for slot, part in enumerate(choices[side].split(", ")):
            words = part.split()
            assert words[0] == "move", part
            index = int(words[1])
            mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
            target = int(words[2]) if len(words) > 2 else None
            move_id = mon.moves[index - 1].id
            slots.append(MoveAction(slot=slot, move_index=index, move_id=move_id, target=target))
        out.append(SideAction(slots=tuple(slots)))
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
    before, choices, theirs, _log = _play(oracle, name)
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
