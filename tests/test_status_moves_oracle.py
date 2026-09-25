"""Clangorous Soul, Imprison and Psych Up in the port, played against Showdown (IKA-256).

Showdown a5df827, data/moves.ts (the champions mod only makes Clangorous Soul
`accuracy: true`)::

    clangoroussoul: {
        onTry(source) {
            if (source.hp <= (source.maxhp * 33 / 100) || source.maxhp === 1) return false;
        },
        onTryHit(pokemon, target, move) {
            if (!this.boost(move.boosts!)) return null;
            delete move.boosts;
        },
        onHit(pokemon) { this.directDamage(pokemon.maxhp * 33 / 100); },
        boosts: { atk: 1, def: 1, spa: 1, spd: 1, spe: 1 },
    },
    imprison: {
        volatileStatus: 'imprison',
        condition: {
            onFoeDisableMove(pokemon) {
                for (const moveSlot of this.effectState.source.moveSlots) {
                    if (moveSlot.id === 'struggle') continue;
                    pokemon.disableMove(moveSlot.id, true);
                }
            },
            onFoeBeforeMovePriority: 4,
            onFoeBeforeMove(attacker, defender, move) {
                if (move.id !== 'struggle' && this.effectState.source.hasMove(move.id) && ...) {
                    this.add('cant', attacker, 'move: Imprison', move);
                    return false;
                }
            },
        },
    },
    psychup: {
        flags: { bypasssub: 1, allyanim: 1, metronome: 1 },
        onHit(target, source) {
            for (i in target.boosts) source.boosts[i] = target.boosts[i];
            // then the crit volatiles: the user's removed, the target's copied
        },
    },

The port gave Clangorous Soul its boosts but no cost, and did nothing for the other two,
with a `status move: <id>` note on each. Each case plays in Showdown and holds the port's
one outcome to it; the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`). Imprison's other half, the menu, is
Python's (`actions.imprisoned_moves`) and is held to Showdown's request.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, is_struggling, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"imprison", "focusenergy", "dragoncheer", "substitute"})
IDS = ("clangoroussoul", "imprison", "psychup")
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user: 1 Clangorous Soul, 2 Imprison, 3 Psych Up, 4 Dragon Claw.
USER = ["clangoroussoul", "imprison", "psychup", "dragonclaw"]
#: The partner: 1 Protect, 2 Swords Dance, 3 Helping Hand, 4 Focus Energy.
PARTNER = ["protect", "swordsdance", "helpinghand", "focusenergy"]
#: Foe A: 1 Swords Dance, 2 Dragon Claw, 3 Protect, 4 Substitute.
FOE_A = ["swordsdance", "dragonclaw", "protect", "substitute"]
#: Foe B: 1 Focus Energy, 2 Growl, 3 Helping Hand, 4 Protect.
FOE_B = ["focusenergy", "growl", "helpinghand", "protect"]
#: Foe B with Assurance in place of Growl.
ASSURER = ["focusenergy", "assurance", "helpinghand", "protect"]


def _teams(
    user_ability: str = "Bulletproof",
    user_item: str | None = None,
    user_spe: int = 0,
    foe_a_moves: list[str] | None = None,
    foe_b_moves: list[str] | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon("Kommo-o", user_ability, USER, user_spe, user_item),
        _mon("Swampert", "Damp", PARTNER, 20, hp=32, **{"def": 20, "spd": 20}),
    ]
    theirs = [
        _mon("Garchomp", "Rough Skin", foe_a_moves or FOE_A, 10),
        _mon("Incineroar", "Blaze", foe_b_moves or FOE_B, 10),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


SOUL, IMPRISON = "move 1", "move 2"
#: The partner's Helping Hand on the user: nothing it changes is compared.
HAND = "move 3 -1"


def _me(move: str) -> str:
    return f"{move}, {HAND}"


#: The foes' turn that changes nothing the user's move reads: A Swords Dance, B Helping Hand.
QUIET = "move 1, move 3 -1"
#: The user's turn that changes nothing: Psych Up on the unboosted partner.
STILL = _me("move 3 -2")
FAST = 32

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- Clangorous Soul
    "clangorous-soul-raises-and-pays": (
        _teams(), [], [_me(SOUL), QUIET], "|-damage|p1a: Kommo-o",
    ),
    "clangorous-soul-twice": (
        _teams(), [[_me(SOUL), QUIET]], [_me(SOUL), QUIET], "|-damage|p1a: Kommo-o",
    ),
    "clangorous-soul-fails-at-a-third": (
        _teams(), [[_me(SOUL), QUIET]] * 3, [_me(SOUL), QUIET], "|-fail|p1a: Kommo-o",
    ),
    "clangorous-soul-sets-off-a-sitrus-berry": (
        _teams(user_item="Sitrus Berry"), [[_me(SOUL), QUIET]], [_me(SOUL), QUIET],
        "|-enditem|p1a: Kommo-o|Sitrus Berry|[eat]",
    ),
    "clangorous-soul-through-magic-guard": (
        _teams(user_ability="Magic Guard"), [], [_me(SOUL), QUIET], "|-damage|p1a: Kommo-o",
    ),
    "clangorous-soul-with-contrary": (
        _teams(user_ability="Contrary"), [], [_me(SOUL), QUIET], "|-unboost|p1a: Kommo-o",
    ),
    # `directDamage` does not set `hurtThisTurn` (only `spreadDamage` does), so a later
    # Assurance stays 60 (PR #1 review).
    "assurance-after-clangorous-soul-stays-60": (
        _teams(user_spe=FAST, foe_b_moves=ASSURER), [], [_me(SOUL), "move 1, move 2 1"],
        "|move|p2b: Incineroar|Assurance|p1a: Kommo-o",
    ),
    "control-assurance-after-psych-up": (
        _teams(user_spe=FAST, foe_b_moves=ASSURER), [], [_me("move 3 2"), "move 1, move 2 1"],
        "|move|p2b: Incineroar|Assurance|p1a: Kommo-o",
    ),
    # -- Imprison (the user moves first unless it says otherwise)
    "imprison-stops-a-shared-move-chosen-before": (
        _teams(user_spe=FAST), [], [_me(IMPRISON), "move 2 2, move 3 -1"],
        "|cant|p2a: Garchomp|move: Imprison",
    ),
    "imprison-stops-a-shared-status-move": (
        _teams(user_spe=FAST, foe_a_moves=["swordsdance", "psychup", "protect", "substitute"]), [],
        [_me(IMPRISON), "move 2 1, move 3 -1"], "|cant|p2a: Garchomp|move: Imprison",
    ),
    "imprison-fails-when-already-up": (
        _teams(user_spe=FAST), [[_me(IMPRISON), QUIET]], [_me(IMPRISON), QUIET], "|-fail|p1a: Kommo-o",
    ),
    "control-imprison-leaves-other-moves": (
        _teams(user_spe=FAST), [], [_me(IMPRISON), QUIET], "|-start|p1a: Kommo-o|move: Imprison",
    ),
    "control-a-slower-imprison-stops-nothing-this-turn": (
        _teams(), [], [_me(IMPRISON), "move 2 2, move 3 -1"], "|move|p2a: Garchomp|Dragon Claw",
    ),
    # -- Psych Up (the user moves last; foe B's Growl gave it -1 first)
    "psych-up-copies-a-foes-boosts": (
        _teams(), [[STILL, "move 1, move 2"]], [_me("move 3 1"), QUIET], "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-through-protect": (
        _teams(), [[STILL, "move 1, move 2"]], [_me("move 3 1"), "move 3, move 3 -1"],
        "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-through-a-substitute": (
        _teams(), [[STILL, "move 4, move 2"], [STILL, QUIET]], [_me("move 3 1"), QUIET],
        "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-clears-its-own-drops": (
        _teams(), [[STILL, "move 1, move 2"]], [_me("move 3 2"), "move 1, move 4"],
        "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-copies-focus-energy": (
        _teams(), [[STILL, "move 1, move 1"]], [_me("move 3 2"), "move 1, move 4"],
        "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-drops-its-own-focus-energy": (
        _teams(), [[STILL, "move 1, move 1"], [_me("move 3 2"), "move 1, move 4"]],
        [_me("move 3 1"), QUIET], "|-copyboost|p1a: Kommo-o",
    ),
    "psych-up-with-contrary": (
        _teams(user_ability="Contrary"), [[STILL, "move 1, move 2"]], [_me("move 3 1"), QUIET],
        "|-copyboost|p1a: Kommo-o",
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
            mon.item,
            tuple(mon.boosts.get(stat, 0) for stat in STATS) if not mon.fainted else (),
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            tuple(m.pp for m in mon.moves),
            mon.move_last_turn_failed if not mon.fainted else None,
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


@pytest.mark.parametrize("move", IDS)
def test_the_moves_are_no_longer_reported(reg, oracle: Oracle, bridged: None, move: str) -> None:  # noqa: ANN001
    """`modelled::status_move_is_fully_modelled` lists them (`tools/port_coverage.py` reads
    the names in `moves.rs`), so using one notes nothing."""
    mine, theirs = _teams()
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    choice = {"clangoroussoul": SOUL, "imprison": IMPRISON, "psychup": "move 3 1"}[move]
    result = resolve_turn(reg, before, _actions(reg, before, [_me(choice), QUIET]), budget=BUDGET)
    assert f"status move: {move}" not in result.unmodelled, result.unmodelled


def _showdown_moves(handle, side: int, slot: int) -> set[str]:  # noqa: ANN001
    active = handle.requests[side]["active"][slot]
    return {m["id"] for m in active["moves"] if not m.get("disabled")}


def _our_moves(reg, pos: Position, side: int, slot: int) -> set[str]:  # noqa: ANN001
    return {
        a.slots[slot].move_id
        for a in side_actions(reg, pos, side)
        if isinstance(a.slots[slot], MoveAction)
    }


#: name -> (foe A's moves, whether the user Imprisons, then foe A's menu as Showdown offers it).
MENUS: dict[str, tuple[list[str], bool, set[str]]] = {
    "shared-moves-go": (FOE_A, True, {"swordsdance", "protect", "substitute"}),
    "all-shared-is-struggle": (USER, True, {"struggle"}),
    "control-no-imprison": (FOE_A, False, set(FOE_A)),
}


@pytest.mark.parametrize("name", sorted(MENUS))
def test_the_menu_matches_showdowns_request(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The menu the search offers after the port's turn, against Showdown's next request.

    Showdown's own position already carries `disabled` on the move slots `onFoeDisableMove`
    marked, so the menu is built from the position the port resolved, as the search's is.
    """
    foe_a_moves, imprisons, expected = MENUS[name]
    mine, theirs = _teams(user_spe=FAST, foe_a_moves=foe_a_moves)
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    # Imprison, or Clangorous Soul; the foes' Swords Dance (or the shared Imprison, stopped).
    foe_turn = "move 1, move 3 -1" if foe_a_moves == FOE_A else "move 2, move 3 -1"
    choices = [_me(IMPRISON if imprisons else SOUL), foe_turn]
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    shown = {(side, slot): _showdown_moves(handle, side, slot) for side in (0, 1) for slot in (0, 1)}
    handle.close()
    assert shown[1, 0] == expected, shown[1, 0]
    node = rustnode.node_for(reg)
    assert node is not None
    picked = node.resolve(before, _actions(reg, before, choices), BUDGET, select=0)
    assert picked is not None and picked.position is not None
    # Foe B shares nothing, and the imprisoner's own side is untouched.
    for (side, slot), moves in shown.items():
        assert _our_moves(reg, picked.position, side, slot) == moves, (side, slot)


def test_a_charging_move_is_not_struggle_under_imprison(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """A charging move's second turn is fired whatever is disabled (`chooseMove` takes
    `getLockedMove()`, sim/side.ts:675) and Imprison stops it at `BeforeMove`, so the move
    is not Struggle (`tools/show_game.py` reads `is_struggling`). A Choice lock into an
    Imprisoned move is (PR #1 review)."""
    mine, theirs = _teams()
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    pos = Position.from_json(handle.position)
    handle.close()
    foe = pos.sides[1].pokemon[pos.sides[1].active[0]]
    imprisoned = frozenset({"dragonclaw"})
    foe.locked_move = "dragonclaw"
    foe.volatiles.append(Effect(id="twoturnmove", duration=1, move="dragonclaw"))
    assert not is_struggling(foe, reg, imprisoned)
    foe.volatiles = [v for v in foe.volatiles if v.id != "twoturnmove"]
    assert is_struggling(foe, reg, imprisoned)
