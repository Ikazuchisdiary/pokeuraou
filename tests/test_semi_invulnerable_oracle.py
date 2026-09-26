"""Semi-invulnerability in the port, played against Showdown (IKA-241).

Showdown a5df827 (the champions mod overrides none of it). A charging move's first turn
adds `twoturnmove`, whose `onStart` adds the move's own volatile; `onTryMove` removes it as
the move fires, `twoturnmove.onEnd` / `onMoveAborted` with it::

    fly / bounce: condition: { duration: 2,
        onInvulnerability(target, source, move) {
            if (['gust', 'twister', 'skyuppercut', 'thunder', 'hurricane', 'smackdown',
                 'thousandarrows'].includes(move.id)) return;
            return false;
        }, ... }
    dig:  earthquake, magnitude reach it;  onImmunity(type) { sandstorm, hail: false }
    dive: surf, whirlpool reach it;        onImmunity(type) { sandstorm, hail: false }
    phantomforce / shadowforce: condition: { duration: 2, onInvulnerability: false }
    noguard: onAnyInvulnerability(...) { if (source or target holds it) return 0; }

    // sim/battle-actions.ts, step 0 of trySpreadMoveHit (after `spreadHit` is set)
    hitStepInvulnerabilityEvent(targets, pokemon, move) {
        if (move.id === 'helpinghand') return new Array(targets.length).fill(true);
        ... else if (gen >= 8 && move.id === 'toxic' && pokemon.hasType('Poison')) true;
        else hitResults[i] = this.battle.runEvent('Invulnerability', target, pokemon, move);
        if (hitResults[i] === false) { if (move.smartTarget) move.smartTarget = false; ...

    // Perish Song's onHitField: `if (runEvent('Invulnerability', ...) === false) -miss`
    // Grassy Terrain's onResidual: `if (isGrounded() && !isSemiInvulnerable()) heal`

The port had none of it: a Phantom Force user was hit on its charge turn and on its second
turn before it fired (IKA-205, IKA-309, IKA-316: 14 silent divergences). The positive
control is the exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`):
the `dodges`, `shelters` and `skips` cases fail there, the `reaches` and `control` cases pass.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"perishsong", "twoturnmove", "helpinghand"})
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
FAST, SLOW = 32, 0


def _mon(species: str, ability: str, moves: list[str], spe: int, **sp: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


def _user(charge: str) -> list[str]:
    """The user: 1 the charging move, 2 Protect, 3 Bulk Up, 4 Close Combat."""
    return [charge, "protect", "bulkup", "closecombat"]


#: The partner: 1 Protect, 2 Earthquake, 3 Helping Hand, 4 Surf.
PARTNER = ["protect", "earthquake", "helpinghand", "surf"]
#: Foe A: 1 Dragon Claw, 2 Rock Slide, 3 Swords Dance, 4 Hurricane.
FOE_A = ["dragonclaw", "rockslide", "swordsdance", "hurricane"]
#: Foe A with Dragon Darts first.
DARTS = ["dragondarts", "rockslide", "swordsdance", "hurricane"]
#: Foe A with Imprison and the user's Phantom Force.
IMPRISONER = ["imprison", "dragonclaw", "swordsdance", "phantomforce"]
#: Foe B: 1 Thunder Wave, 2 Perish Song, 3 Helping Hand, 4 Toxic.
FOE_B = ["thunderwave", "perishsong", "helpinghand", "toxic"]


def _teams(
    charge: str = "phantomforce",
    user_spe: int = FAST,
    foe_a: tuple[str, str] = ("Garchomp", "Rough Skin"),
    foe_a_moves: list[str] | None = None,
    foe_b: tuple[str, str] = ("Incineroar", "Blaze"),
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon("Annihilape", "Defiant", _user(charge), user_spe),
        _mon("Swampert", "Damp", PARTNER, 20, hp=32, **{"def": 20, "spd": 20}),
        _mon("Milotic", "Marvel Scale", PARTNER, 0),
    ]
    theirs = [
        _mon(foe_a[0], foe_a[1], foe_a_moves or FOE_A, 10),
        _mon(foe_b[0], foe_b[1], FOE_B, 10),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


#: The user charges at foe A; the partner's Helping Hand on it (which reaches it).
CHARGE = "move 1 1, move 3 -1"
#: The second turn: the locked move, the only choice Showdown takes.
FIRE = "move 1, move 3 -1"
#: Foes that change nothing the user's move reads: A Swords Dance, B Helping Hand.
QUIET = "move 3, move 3 -1"
CLAW = "move 1 1, move 3 -1"
MISS = "|-miss|p2a:"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    "dodges-a-single-target-move-on-the-charge-turn": (
        _teams(), [], [CHARGE, CLAW], "|-miss|p2a: Garchomp|p1a: Annihilape",
    ),
    "dodges-a-spread-move-and-the-partner-takes-the-spread-hit": (
        _teams(), [], [CHARGE, "move 2, move 3 -1"], "|-miss|p2a: Garchomp|p1a: Annihilape",
    ),
    "dodges-its-partners-earthquake": (
        _teams(), [], ["move 1 1, move 2", QUIET], "|-miss|p1b: Swampert|p1a: Annihilape",
    ),
    "dodges-on-the-second-turn-until-it-fires": (
        _teams(user_spe=SLOW), [[CHARGE, QUIET]], [FIRE, CLAW], "|-miss|p2a: Garchomp|p1a: Annihilape",
    ),
    "dodges-thunder-wave": (
        _teams(), [], [CHARGE, "move 3, move 1 1"], "|-miss|p2b: Incineroar|p1a: Annihilape",
    ),
    "dodges-toxic-from-a-non-poison-type": (
        _teams(), [], [CHARGE, "move 3, move 4 1"], "|-miss|p2b: Incineroar|p1a: Annihilape",
    ),
    "dodges-perish-song": (
        _teams(), [], [CHARGE, "move 3, move 2"], "|-miss|p2b: Incineroar|p1a: Annihilape",
    ),
    "dodges-dragon-darts-which-hits-the-partner-twice": (
        # Dragapult outspeeds the user: the second turn, before it fires.
        _teams(foe_a=("Dragapult", "Clear Body"), foe_a_moves=DARTS), [[CHARGE, QUIET]], [FIRE, CLAW],
        "|-anim|p2a: Dragapult|Dragon Darts|p1b: Swampert",
    ),
    "dodges-dragon-claw-while-flying": (
        _teams(charge="fly"), [], [CHARGE, CLAW], "|-miss|p2a: Garchomp|p1a: Annihilape",
    ),
    "dodges-dragon-claw-underwater": (
        _teams(charge="dive"), [], [CHARGE, CLAW], "|-miss|p2a: Garchomp|p1a: Annihilape",
    ),
    "shelters-from-sand-underground": (
        _teams(charge="dig", foe_a=("Tyranitar", "Sand Stream")), [], [CHARGE, QUIET],
        "|-prepare|p1a: Annihilape|Dig",
    ),
    "shelters-from-sand-underwater": (
        _teams(charge="dive", foe_a=("Tyranitar", "Sand Stream")), [], [CHARGE, QUIET],
        "|-prepare|p1a: Annihilape|Dive",
    ),
    "skips-the-grassy-terrain-heal": (
        _teams(user_spe=SLOW, foe_a=("Rillaboom", "Grassy Surge")),
        [["move 3, move 3 -1", CLAW]], [CHARGE, QUIET], "|-prepare|p1a: Annihilape|Phantom Force",
    ),
    # -- the moves and holders that reach it, and the controls
    "reaches-with-hurricane-while-flying": (
        _teams(charge="fly"), [], [CHARGE, "move 4 1, move 3 -1"],
        "|move|p2a: Garchomp|Hurricane|p1a: Annihilape",
    ),
    "reaches-with-toxic-from-a-poison-type": (
        _teams(foe_b=("Toxapex", "Regenerator")), [], [CHARGE, "move 3, move 4 1"],
        "|-status|p1a: Annihilape|tox",
    ),
    "reaches-through-no-guard": (
        _teams(foe_a=("Garchomp", "No Guard")), [], [CHARGE, CLAW],
        "|move|p2a: Garchomp|Dragon Claw|p1a: Annihilape",
    ),
    "control-in-reach-before-it-charges": (
        _teams(user_spe=SLOW), [], [CHARGE, CLAW], "|move|p2a: Garchomp|Dragon Claw|p1a: Annihilape",
    ),
    "control-phantom-force-takes-sand": (
        _teams(foe_a=("Tyranitar", "Sand Stream")), [], [CHARGE, QUIET],
        "|-damage|p1a: Annihilape|",
    ),
    "control-in-reach-once-imprison-stops-the-second-turn": (
        _teams(foe_a_moves=IMPRISONER), [["move 1 1, move 3 -1", "move 1, move 3 -1"]],
        [FIRE, "move 2 1, move 3 -1"], "|cant|p1a: Annihilape|move: Imprison",
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
            mon.status if not mon.fainted else None,
            tuple(mon.boosts.get(stat, 0) for stat in STATS) if not mon.fainted else (),
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
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


def test_a_doubled_reach_is_named(reg, oracle: Oracle, bridged: None) -> None:  # noqa: ANN001
    """Earthquake into a Dig user doubles (`onSourceModifyDamage`), which the port does not
    model: it answers with the ordinary hit and names it."""
    mine, theirs = _teams(charge="dig")
    choices = ["move 1 1, move 2", QUIET]  # the user digs first, then its partner's Earthquake
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    log = list(handle.log)
    handle.close()
    assert any(line.startswith("|move|p1b: Swampert|Earthquake|") for line in log), log
    assert "|-damage|p1a: Annihilape|" in "\n".join(log[log.index("|-prepare|p1a: Annihilape|Dig") :])
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert "semi-invulnerable: earthquake into dig" in result.unmodelled, result.unmodelled
