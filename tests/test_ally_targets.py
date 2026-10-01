"""A `normal` move aimed at the user's ally (IKA-181).

Showdown lets a `normal` single-target move name the ally (`Battle.validTargetLoc`)::

    case 'randomNormal':
    case 'scripted':
    case 'normal':
        return isAdjacent;

and in doubles the ally is adjacent. The menu never offered it (`TARGETS_REQUIRING_FOE`), so
Stamina, Flash Fire, Unburden and the like were never set off by a partner. The mode
`actions.ALLY_TARGETS` lists them: ``off`` (ships), ``benefit`` (where the ally gains,
`actions.ally_benefits`), ``all``.

The port had them as a target too, and Sucker Punch aimed at an attacking ally failed
there: it asked for a waiting *foe* (`moves.rs`), where Showdown's `onTry` asks only that
the target will use a damaging move::

    onTry(source, target) {
        const action = this.queue.willMove(target);
        const move = action?.choice === 'move' ? action.move : null;
        if (!move || (move.category === 'Status' && move.id !== 'mefirst') || ...) return false;
    },
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou import actions
from pokeuraou.actions import MoveAction, names_ally, side_actions
from pokeuraou.budget import Budget
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


GOLISOPOD = _mon("Golisopod", "Emergency Exit", ["suckerpunch", "leechlife", "protect", "swordsdance"])
ARCHALUDON = _mon("Archaludon", "Stamina", ["flashcannon", "protect", "dracometeor", "bodypress"])
ARMAROUGE = _mon("Armarouge", "Flash Fire", ["armorcannon", "protect", "psychic", "trickroom"])
DRAGAPULT = _mon("Dragapult", "Infiltrator", ["willowisp", "dragondarts", "protect", "phantomforce"])
METAGROSS = _mon("Metagross", "Clear Body", ["psychup", "ironhead", "protect", "bulletpunch"])
FILL = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
]
FOES = [
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
    _mon("Kingambit", "Defiant", ["protect", "ironhead", "suckerpunch", "swordsdance"]),
    *FILL,
]
FOES_PROTECT = "move 2, move 1"


def _battle(oracle: Oracle, ours: list[TeamSet]):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, ours, FOES, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    return handle


def _position(oracle: Oracle, ours: list[TeamSet]) -> Position:
    handle = _battle(oracle, ours)
    pos = Position.from_json(handle.position)
    handle.close()
    return pos


def _ally_choices(reg, pos: Position, mode: str) -> set[str]:  # noqa: ANN001
    with actions.ally_targets(mode):
        return {
            f"{s.slot}:{s.move_id}"
            for a in side_actions(reg, pos, 0)
            for s in a.slots
            if isinstance(s, MoveAction) and s.target is not None and s.target < 0
            and reg.moves[s.move_id].target == "normal"
        }


# ---------------------------------------------------------------------------
# The menu.


@pytest.mark.oracle
def test_off_lists_no_ally_target_and_all_lists_every_one(reg, oracle: Oracle) -> None:  # noqa: ANN001
    pos = _position(oracle, [GOLISOPOD, ARCHALUDON, *FILL])
    assert _ally_choices(reg, pos, "off") == set()
    every = {
        f"{slot}:{m}"
        for slot, mon in ((0, GOLISOPOD), (1, ARCHALUDON))
        for m in mon.moves
        if reg.moves[m].target == "normal"
    }
    assert _ally_choices(reg, pos, "all") == every
    # Stamina: every damaging `normal` move on Archaludon, Swords Dance and Protect not;
    # nothing on Golisopod (Emergency Exit gains nothing).
    assert _ally_choices(reg, pos, "benefit") == {"0:suckerpunch", "0:leechlife"}
    # The mode is put back after the block.
    assert actions.ALLY_TARGETS[0] == "off"


@pytest.mark.oracle
def test_benefit_follows_the_allys_ability(reg, oracle: Oracle) -> None:  # noqa: ANN001
    # Flash Fire takes Will-O-Wisp, a status move of its type; Dragon Darts is not `normal`.
    pos = _position(oracle, [DRAGAPULT, ARMAROUGE, *FILL])
    assert _ally_choices(reg, pos, "benefit") == {"0:willowisp"}
    # Psych Up is for an ally whatever it holds; Iron Head on Milotic gains nothing.
    pos = _position(oracle, [METAGROSS, FILL[0], ARCHALUDON, FILL[1]])
    assert _ally_choices(reg, pos, "benefit") == {"0:psychup"}


@pytest.mark.oracle
def test_a_fainted_ally_is_not_a_target(reg, oracle: Oracle) -> None:  # noqa: ANN001
    pos = _position(oracle, [GOLISOPOD, ARCHALUDON, *FILL])
    ally = pos.sides[0].pokemon[pos.sides[0].active[1]]
    ally.hp = 0
    ally.fainted = True
    assert _ally_choices(reg, pos, "all") == set()


def test_an_unknown_mode_stops(reg) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="ally target mode"):
        actions.set_ally_targets("some")
    with pytest.raises(ValueError, match="ally_targets"):
        from pokeuraou.selfplay import play_game

        play_game(reg, None, [], [], "x", open_information=True, ally_targets="some")  # type: ignore[arg-type]


@pytest.mark.oracle
def test_showdown_takes_every_ally_target_the_menu_lists(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Every choice `all` lists, Showdown accepts; and the ally targets `off` leaves out are
    choices Showdown accepts (the gap `tests/test_actions.py`'s near-misses never probed)."""
    handle = _battle(oracle, [GOLISOPOD, ARCHALUDON, *FILL])
    pos = Position.from_json(handle.position)
    with actions.ally_targets("all"):
        listed = [a.to_choice() for a in side_actions(reg, pos, 0)]
    with_ally = [c for c in listed if "-" in c]
    assert with_ally, "no ally target listed"
    with actions.ally_targets("off"):
        off = {a.to_choice() for a in side_actions(reg, pos, 0)}
    answers = handle.probe(0, listed)
    handle.close()
    refused = [r for r in answers if not r["ok"]]
    assert not refused, refused[:4]
    assert set(with_ally) - off == set(with_ally)


# ---------------------------------------------------------------------------
# The port against Showdown.


def _find(reg, pos: Position, side: int, choice: str):  # noqa: ANN001, ANN202
    with actions.ally_targets("all"):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
    assert choice in menu, (choice, sorted(menu)[:8])
    return menu[choice]


CASES = {
    # Sucker Punch on Archaludon about to use Flash Cannon: it lands, and Stamina raises
    # Defense. The port failed it before IKA-181.
    "sucker punch on an attacking ally": "move 1 -2, move 1 1",
    # The control: on an ally using Protect it fails, in both, before and after.
    "sucker punch on a protecting ally": "move 1 -2, move 2",
}


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_resolves_an_ally_target_as_showdown(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn

    handle = _battle(oracle, [GOLISOPOD, ARCHALUDON, *FILL])
    start = Position.from_json(handle.position)
    ours = CASES[name]
    handle.step([ours, FOES_PROTECT])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    chosen = [_find(reg, start, 0, ours), _find(reg, start, 1, FOES_PROTECT)]
    assert names_ally(reg, chosen[0])
    ported = port_turn(port, start, chosen, BUDGET)

    def ally(pos: Position):  # noqa: ANN202
        return pos.sides[0].pokemon[pos.sides[0].active[1]]

    hit = name == "sucker punch on an attacking ally"
    # Showdown's own facts first: the case is the one it names.
    assert (ally(after).hp < ally(start).hp) == hit
    assert ally(after).boosts.get("def", 0) == (1 if hit else 0)
    assert (ally(ported).hp, ally(ported).boosts.get("def", 0)) == (
        ally(after).hp, ally(after).boosts.get("def", 0)
    )
