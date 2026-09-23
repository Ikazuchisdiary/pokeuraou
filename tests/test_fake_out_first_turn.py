"""Fake Out and First Impression are off the menu once their user has moved (IKA-166).

The champions mod gives both moves (vendor/pokemon-showdown/data/mods/champions/moves.ts)

    onDisableMove(pokemon) {
        if (pokemon.activeMoveActions) pokemon.disableMove('fakeout');
    },

and `endTurn` runs every `DisableMove` handler before each request (sim/battle.ts:1691), so
from the second turn out the request marks them disabled and Showdown refuses the choice.
The base game's `onTry` (`activeMoveActions > 1` fails the move) is still there, but in this
dex nobody reaches it. `side_actions` offered them on every turn, and `narrow` only dropped
them at a counter above 1, so the second turn's Fake Out stayed on the search's menu.

The counter is `runMove`'s first line (sim/battle-actions.ts:217), before `BeforeMove`
(:255), so a Pokemon that was flinched, slept or fully paralysed has still spent its first
turn out. The resolver and the port counted only a move that started, so the Incineroar
that lost a Fake Out war kept a counter of 0 and was offered -- and landed -- a Fake Out
the next turn.

The bridge does not export the counter, so each Python position here is the resolver's
own child of the one before, which is exactly what the search enumerates at. Showdown's
request is the reference; the controls are the first turn out and a Pokemon that switched
out and back in.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, side_actions
from pokeuraou.narrow import drop_dead_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


TEAM_A = [
    _mon("Incineroar", "Intimidate", ["fakeout", "protect", "flareblitz", "uturn"], 4),
    _mon("Kleavor", "Sharpness", ["firstimpression", "protect", "swordsdance", "xscissor"], 6),
    _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"], 8),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"], 2),
]

TEAM_B = [
    # Faster than Incineroar, so its Fake Out lands first and flinches it.
    _mon("Lopunny", "Limber", ["fakeout", "protect", "return", "icepunch"], 30),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"], 10),
    _mon("Corviknight", "Unnerve", ["bulkup", "protect", "bravebird", "roost"], 16),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 20),
]

#: name -> the turns played after team preview, both sides' choices per turn.
CASES = {
    # Both of ours move on turn 1 (Protect): turn 2 has neither Fake Out nor First Impression.
    "moved": [
        ["move 2, move 2", "move 2, move 1"],
    ],
    # Lopunny's Fake Out flinches Incineroar, whose Fake Out never starts. The counter is
    # still spent.
    "flinched": [
        ["move 1 1, move 2", "move 1 1, move 1"],
    ],
    # Control: Incineroar goes out on turn 2 and comes back on turn 3; turn 4 is its first
    # turn out again. Kleavor, which stayed in, still may not use First Impression.
    "switched-back": [
        ["move 2, move 2", "move 2, move 1"],
        ["switch 3, move 3", "move 4 2, move 1"],
        ["switch 3, move 3", "move 4 2, move 1"],
    ],
}

#: Deterministic: the fixed roll and no enumerated secondaries or crits.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _showdown_menu(request: dict) -> list[set[str]]:
    return [
        {move["id"] for move in active["moves"] if not move.get("disabled")} for active in request["active"]
    ]


def _python_menu(reg, pos: Position, side: int, *, narrowed: bool) -> list[set[str]]:  # noqa: ANN001
    pool = side_actions(reg, pos, side)
    if narrowed:
        pool = drop_dead_actions(reg, pos, side, pool)
    out: list[set[str]] = [set() for _ in pos.sides[side].active]
    for action in pool:
        for piece in action.slots:
            if isinstance(piece, MoveAction):
                out[piece.slot].add(piece.move_id)
    return out


def _find(reg, pos: Position, side: int, choice: str):  # noqa: ANN001, ANN202
    found = [a for a in side_actions(reg, pos, side) if a.to_choice() == choice]
    assert found, (side, choice, [a.to_choice() for a in side_actions(reg, pos, side)])
    return found[0]


def _python_turn(reg, pos: Position, choices: list[str]) -> Position:  # noqa: ANN001
    actions = [_find(reg, pos, side, choices[side]) for side in (0, 1)]
    result = resolve_turn(reg, pos, actions, budget=BUDGET)
    assert result.branches
    counters = {
        tuple(mon.active_move_actions for side in branch.position.sides for mon in side.pokemon)
        for branch in result.branches
    }
    assert len(counters) == 1, counters
    return result.branches[0].position


def _play(reg, oracle: Oracle, name: str):  # noqa: ANN001, ANN202
    """Showdown's last request for side 0, and the Python position the search would be at."""
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    pos = Position.from_json(handle.position)
    first = _showdown_menu(handle.requests[0])
    for choices in CASES[name]:
        handle.step(choices)
        assert handle.choice_errors == [], handle.choice_errors
        pos = _python_turn(reg, pos, choices)
    request = handle.requests[0]
    log = list(handle.log)
    handle.close()
    return first, request, pos, log


#: name -> the moves Showdown disables on side 0's last request, per active slot.
DISABLED = {
    "moved": [{"fakeout"}, {"firstimpression"}],
    "flinched": [{"fakeout"}, {"firstimpression"}],
    "switched-back": [set(), {"firstimpression"}],
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_disables_fake_out_after_the_first_move(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The fact, with its control: the first turn out offers both moves."""
    first, request, _pos, log = _play(reg, oracle, name)
    assert {"fakeout"} <= first[0] and {"firstimpression"} <= first[1], first
    if name == "flinched":
        assert any(line.startswith("|cant|p1a: Incineroar|flinch") for line in log), log
    disabled = [
        {move["id"] for move in active["moves"] if move.get("disabled")} for active in request["active"]
    ]
    assert disabled == DISABLED[name], disabled


@pytest.mark.oracle
@pytest.mark.parametrize("narrowed", [False, True], ids=["side_actions", "narrowed"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_our_menu_is_showdowns(reg, oracle: Oracle, name: str, narrowed: bool) -> None:  # noqa: ANN001
    """Every move Showdown offers, and nothing it disables, at the resolver's own child."""
    _first, request, pos, _log = _play(reg, oracle, name)
    assert _python_menu(reg, pos, 0, narrowed=narrowed) == _showdown_menu(request)


def test_the_counter_decides_the_menu(reg) -> None:  # noqa: ANN001
    """No oracle: the same position at counter 0 and 1, and a Choice lock into Fake Out."""
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, TEAM_A)
    incineroar = next(m for m in pos.sides[0].pokemon if m.species == "incineroar")
    assert incineroar.active_index == 0
    assert "fakeout" in _python_menu(reg, pos, 0, narrowed=False)[0]
    incineroar.active_move_actions = 1
    assert "fakeout" not in _python_menu(reg, pos, 0, narrowed=False)[0]
    assert "flareblitz" in _python_menu(reg, pos, 0, narrowed=False)[0]

    # Choice-locked into Fake Out: the item disables the other three and the counter this
    # one, so the request is Struggle.
    incineroar.item = "choicescarf"
    incineroar.volatiles.append(Effect(id="choicelock", move="fakeout"))
    assert _python_menu(reg, pos, 0, narrowed=False)[0] == {"struggle"}
    incineroar.active_move_actions = 0
    assert _python_menu(reg, pos, 0, narrowed=False)[0] == {"fakeout"}


def test_a_dex_without_the_hook_still_offers_it(reg) -> None:  # noqa: ANN001
    """The dump decides: without champions' `onDisableMove`, Fake Out stays on the menu."""
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, TEAM_A)
    pos.sides[0].pokemon[0].active_move_actions = 1
    fakeout = reg.moves["fakeout"]
    assert "onDisableMove" in fakeout.custom_hooks
    hooks = [h for h in fakeout.raw["customHooks"] if h != "onDisableMove"]
    reg.moves["fakeout"] = replace(fakeout, raw={**fakeout.raw, "customHooks": hooks})
    try:
        assert "fakeout" in _python_menu(reg, pos, 0, narrowed=False)[0]
        # ...and `narrow` drops it there, because `onTry` will fail it.
        assert "fakeout" not in _python_menu(reg, pos, 0, narrowed=True)[0]
    finally:
        reg.moves["fakeout"] = fakeout


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.oracle
def test_the_port_counts_the_flinched_move(reg, oracle: Oracle, bridged: None) -> None:  # noqa: ANN001
    """The port's child carries the same counter as Python's, flinched Incineroar included."""
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    choices = CASES["flinched"][0]
    actions = [_find(reg, before, side, choices[side]) for side in (0, 1)]
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET, select=0)
    assert there is not None and there.position is not None, "the port refused the turn"
    incineroar = next(m for m in there.position.sides[0].pokemon if m.species == "incineroar")
    assert incineroar.active_move_actions == 1
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    here = resolve_turn(reg, before, actions, budget=BUDGET)
    assert there.position.to_json() == here.branches[0].position.to_json()
