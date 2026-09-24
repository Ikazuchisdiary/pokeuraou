"""A charge's stored target, Gigaton Hammer's repeat lock, and two fields the bridge
did not export (IKA-176).

**The charge's target.** A charging move picks its target on the first turn. The
`twoturnmove` condition stores it on the move's own volatile
(vendor/pokemon-showdown/data/conditions.ts, `twoturnmove.onStart`)::

    attacker.volatiles[effect.id].targetLoc = moveTargetLoc;

and on the second turn `chooseMove` fires at it whatever the choice says
(sim/side.ts:675)::

    const lockedMove = pokemon.getLockedMove() || pokemon.getSemiLockedMove();
    if (lockedMove) {
        let lockedMoveTargetLoc = pokemon.lastMoveTargetLoc || 0;
        ...
        if (pokemon.volatiles[lockedMoveID]?.targetLoc) {
            lockedMoveTargetLoc = pokemon.volatiles[lockedMoveID].targetLoc;

The request's entry for the locked move has no `target` (`getMoves(lockedMove)`), so
the check above it refuses a choice that names one: "You can't choose a target for
Electro Shot". IKA-169 made the second turn's menu that move alone but kept one action per
target, every one of which Showdown refuses, and the resolver fired at whichever the
search picked. Now the resolver stores the target on `twoturnmove` (`extra.targetLoc`,
Showdown's own name, in Python and the port), the bridge exports it, and the menu is the
one choice Showdown takes: the move with no target.

**Gigaton Hammer** has the `cantusetwice` flag, and `endTurn` disables it when it was the
last move (sim/battle.ts:1695)::

    if (activeMove.flags['cantusetwice'] && pokemon.lastMove?.id === moveSlot.id) {
        pokemon.disableMove(pokemon.lastMove.id);
    }

`_usable_move_slots` did not read it. A Choice item beside it leaves nothing: Struggle.

**The bridge** exported neither `activeMoveActions` (so a search started from an oracle
position offered Fake Out on every turn, IKA-166) nor No Retreat's volatile (it was not in
`MODELLED_VOLATILES`, IKA-169).
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from ._port import Budget
from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


ARCHALUDON = _mon("Archaludon", "Stamina", ["electroshot", "protect", "dracometeor", "flashcannon"])
TINKATON = _mon("Tinkaton", "Mold Breaker", ["gigatonhammer", "protect", "playrough", "swordsdance"])
MILOTIC = _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"])
INCINEROAR = _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"])
SYLVEON = _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"])
#: No weather and nothing that heals or hurts a foe on its own, so the Electro Shot is the
#: only thing that moves a foe's HP on the second turn. Both leads take Electro Shot.
FOES = [
    _mon("Dragonite", "Inner Focus", ["dragondance", "protect", "extremespeed", "earthquake"]),
    _mon("Kingambit", "Defiant", ["swordsdance", "protect", "kowtowcleave", "suckerpunch"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
    _mon("Hippowdon", "Sand Stream", ["slackoff", "protect", "earthquake", "yawn"]),
]
#: Two self-boosts: no target, no chance, no HP.
FOES_QUIET = "move 1, move 1"
#: Both foes Protect (the first turn only: a second one in a row is a coin flip).
FOES_PROTECT = "move 2, move 2"

#: Deterministic: the fixed roll and no enumerated secondaries or crits.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _lead(first: TeamSet) -> list[TeamSet]:
    return [first, MILOTIC, INCINEROAR, SYLVEON]


def _start(oracle: Oracle, first: TeamSet, foes: list[TeamSet] = FOES):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, _lead(first), foes, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    return handle


def _find(reg, pos: Position, side: int, choice: str):  # noqa: ANN001, ANN202
    found = [a for a in side_actions(reg, pos, side) if a.to_choice() == choice]
    assert found, (side, choice, [a.to_choice() for a in side_actions(reg, pos, side)])
    return found[0]


def _slot0_choices(reg, pos: Position) -> set[str]:  # noqa: ANN001
    return {a.slots[0].to_choice() for a in side_actions(reg, pos, 0)}


def _foe_hp(pos: dict | Position) -> tuple[int, int]:
    if isinstance(pos, dict):
        side = pos["sides"][1]
        return tuple(side["pokemon"][i]["hp"] for i in side["active"])  # type: ignore[return-value]
    side = pos.sides[1]
    return tuple(side.pokemon[i].hp for i in side.active)  # type: ignore[return-value]


def _hurt(before: dict | Position, after: dict | Position) -> set[int]:
    """Showdown target locs (1, 2) of the foes that lost HP."""
    return {i + 1 for i, (b, a) in enumerate(zip(_foe_hp(before), _foe_hp(after), strict=True)) if a < b}


def _charge_turn(target: int) -> list[str]:
    return [f"move 1 {target}, move 2", FOES_PROTECT]


# ---------------------------------------------------------------------------
# The charge's target.


@pytest.mark.oracle
@pytest.mark.parametrize("target", [1, 2])
def test_showdown_fires_the_first_turns_target(oracle: Oracle, target: int) -> None:
    """The fact: the second turn takes no target and hits the one chosen on the first."""
    for second, refused in [("move 1", False), ("move 1 1", True), ("move 1 2", True)]:
        handle = _start(oracle, ARCHALUDON)
        handle.step(_charge_turn(target))
        assert handle.choice_errors == []
        entry = handle.requests[0]["active"][0]["moves"]
        assert entry == [{"move": "Electro Shot", "id": "electroshot"}], entry
        before = handle.position
        handle.step([f"{second}, move 1", FOES_QUIET])
        errors = handle.choice_errors
        if refused:
            assert any("can't choose a target" in e for e in errors), errors
        else:
            assert errors == [], errors
            assert _hurt(before, handle.position) == {target}
        handle.close()


@pytest.mark.oracle
@pytest.mark.parametrize("target", [1, 2])
def test_the_bridge_exports_the_target(oracle: Oracle, target: int) -> None:
    handle = _start(oracle, ARCHALUDON)
    handle.step(_charge_turn(target))
    pos = Position.from_json(handle.position)
    handle.close()
    charging = pos.sides[0].pokemon[pos.sides[0].active[0]].volatile("twoturnmove")
    assert charging is not None and charging.move == "electroshot"
    assert charging.extra.get("targetLoc") == target, charging


@pytest.mark.oracle
@pytest.mark.parametrize("target", [1, 2])
def test_our_second_turn_menu_is_showdowns(reg, oracle: Oracle, target: int) -> None:  # noqa: ANN001
    """One choice on the menu of Showdown's charging position (no engine involved)."""
    handle = _start(oracle, ARCHALUDON)
    handle.step(_charge_turn(target))
    pos = Position.from_json(handle.position)
    handle.close()
    pos.sides[0].pokemon[pos.sides[0].active[0]].trapped = False
    assert _slot0_choices(reg, pos) == {"move 1"}


def test_a_charge_without_a_stored_target_keeps_every_target(reg) -> None:  # noqa: ANN001
    """The control, and records from before: no target on the marker, one choice per target."""
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, _lead(ARCHALUDON))
    foes = _synthetic_position(reg, FOES)
    pos.sides[1] = foes.sides[0]
    mon = pos.sides[0].pokemon[0]
    mon.volatiles.append(Effect(id="twoturnmove", duration=1, move="electroshot"))
    assert _slot0_choices(reg, pos) == {"move 1 1", "move 1 2"}
    mon.volatiles[-1] = Effect(
        id="twoturnmove", duration=1, move="electroshot", extra={"targetLoc": 2}
    )
    assert _slot0_choices(reg, pos) == {"move 1"}


# ---------------------------------------------------------------------------
# Gigaton Hammer.

#: name -> (Tinkaton's item, the turns after team preview, the moves Showdown offers
#: Tinkaton on the last request).
HAMMERED = ["move 1 1, move 2", FOES_PROTECT]
HAMMER = {
    "hammered": (None, [HAMMERED], {"protect", "playrough", "swordsdance"}),
    # The control: one other move in between and the hammer is back.
    "then-protect": (
        None,
        [HAMMERED, ["move 2, move 1", FOES_QUIET]],
        {"gigatonhammer", "protect", "playrough", "swordsdance"},
    ),
    # A Choice item beside it disables the other three: Struggle.
    "choice-scarf": ("Choice Scarf", [HAMMERED], {"struggle"}),
    # The control: switching out clears `lastMove`.
    "switched-back": (
        None,
        [HAMMERED, ["switch 3, move 1", FOES_QUIET], ["switch 3, move 1", FOES_QUIET]],
        {"gigatonhammer", "protect", "playrough", "swordsdance"},
    ),
}


def _hammer(oracle: Oracle, name: str):  # noqa: ANN202
    item, steps, _ = HAMMER[name]
    handle = _start(oracle, replace(TINKATON, item=item))
    first = handle.position
    for step in steps:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
    return handle, first


def _showdown_moves(handle) -> set[str]:  # noqa: ANN001
    active = handle.requests[0]["active"][0]
    return {m["id"] for m in active["moves"] if not m.get("disabled")}


def _our_moves(reg, pos: Position) -> set[str]:  # noqa: ANN001
    return {
        a.slots[0].move_id for a in side_actions(reg, pos, 0) if isinstance(a.slots[0], MoveAction)
    }


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(HAMMER))
def test_showdown_disables_the_hammer_after_the_hammer(oracle: Oracle, name: str) -> None:
    handle, _ = _hammer(oracle, name)
    assert _showdown_moves(handle) == HAMMER[name][2]
    if name == "hammered":
        handle.step(["move 1 1, move 1", FOES_QUIET])
        assert any("Gigaton Hammer is disabled" in e for e in handle.choice_errors)
    handle.close()


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(HAMMER))
def test_our_hammer_menu_is_showdowns(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The menu on Showdown's own position; on the port's child it is
    `test_the_ports_hammer_child_has_showdowns_menu`."""
    handle, _first = _hammer(oracle, name)
    theirs = _showdown_moves(handle)
    last = handle.position
    handle.close()
    assert _our_moves(reg, Position.from_json(last)) == theirs


def test_the_flag_comes_from_the_dump(reg) -> None:  # noqa: ANN001
    """Without `cantusetwice` in the dump, the hammer stays on the menu."""
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, _lead(TINKATON))
    pos.sides[0].pokemon[0].last_move = "gigatonhammer"
    assert "gigatonhammer" not in _our_moves(reg, pos)
    hammer = reg.moves["gigatonhammer"]
    assert "cantusetwice" in hammer.flags
    reg.moves["gigatonhammer"] = replace(hammer, flags=hammer.flags - {"cantusetwice"})
    try:
        assert "gigatonhammer" in _our_moves(reg, pos)
    finally:
        reg.moves["gigatonhammer"] = hammer
    # The control: another last move leaves it on.
    pos.sides[0].pokemon[0].last_move = "protect"
    assert "gigatonhammer" in _our_moves(reg, pos)


# ---------------------------------------------------------------------------
# The bridge's two fields.


@pytest.mark.oracle
def test_the_bridge_exports_the_move_counter(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """A search started from Showdown's position knows Incineroar has moved."""
    lead = [INCINEROAR, MILOTIC, ARCHALUDON, SYLVEON]
    handle = oracle.create(FORMAT_ID, lead, FOES, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    first = Position.from_json(handle.position)
    # The control: nobody has moved yet, and Fake Out is on offer.
    assert first.sides[0].pokemon[0].active_move_actions == 0
    assert "fakeout" in _our_moves(reg, first)
    handle.step(["move 2 1, move 2", FOES_PROTECT])
    assert handle.choice_errors == []
    pos = Position.from_json(handle.position)
    request = handle.requests[0]["active"][0]
    handle.close()
    assert pos.sides[0].pokemon[0].active_move_actions == 1
    assert pos.sides[0].pokemon[1].active_move_actions == 1
    assert _our_moves(reg, pos) == {m["id"] for m in request["moves"] if not m.get("disabled")}
    assert "fakeout" not in _our_moves(reg, pos)


@pytest.mark.oracle
def test_the_bridge_exports_no_retreat(oracle: Oracle) -> None:
    falinks = _mon("Falinks", "Battle Armor", ["noretreat", "protect", "closecombat", "rockslide"])
    handle = _start(oracle, falinks)
    handle.step(["move 1, move 2", FOES_PROTECT])
    raw = handle.position
    handle.close()
    mon = raw["sides"][0]["pokemon"][0]
    assert "noretreat" not in mon["unmodelledVolatiles"]
    assert Position.from_json(raw).sides[0].pokemon[0].has_volatile("noretreat")


# ---------------------------------------------------------------------------
# The port.


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.oracle
@pytest.mark.parametrize("target", [1, 2])
def test_the_port_stores_and_fires_the_target(reg, oracle: Oracle, bridged: None, target: int) -> None:  # noqa: ANN001
    """Both turns in the port: the stored target is the one the second turn hits."""
    handle = _start(oracle, ARCHALUDON)
    start = Position.from_json(handle.position)
    handle.close()
    for side in start.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    node = rustnode.node_for(reg)
    assert node is not None
    pos = start
    for choices in [_charge_turn(target), ["move 1, move 1", FOES_QUIET]]:
        actions = [_find(reg, pos, side, choices[side]) for side in (0, 1)]
        there = node.resolve(pos, actions, BUDGET, select=0)
        assert there is not None and there.position is not None, "the port refused the turn"
        assert len(there.branches) == 1, there.branches
        pos = there.position
    assert _hurt(start, pos) == {target}


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


def _port_turn(reg, port, pos: Position, choices: list[str]) -> Position:  # noqa: ANN001
    from ._port_showdown import port_turn

    actions = [_find(reg, pos, side, choices[side]) for side in (0, 1)]
    return port_turn(port, pos, actions, BUDGET)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "form",
    [
        # Showdown's charging Pokemon carries the move's own `electroshot` volatile, which
        # Python ignores and the port refused until IKA-208 (IKA-207 found it); it is the
        # charge `twoturnmove` already carries, and the port drops it.
        "showdown position",
        "the port's child",
    ],
)
@pytest.mark.parametrize("target", [1, 2])
def test_the_ports_second_turn_is_showdowns(reg, oracle: Oracle, port, target: int, form: str) -> None:  # noqa: ANN001
    """`test_our_second_turn_is_showdowns` with the port resolving both turns."""
    handle = _start(oracle, ARCHALUDON)
    start = Position.from_json(handle.position)
    handle.step(_charge_turn(target))
    if form == "showdown position":
        pos = Position.from_json(handle.position)
        pos.sides[0].pokemon[pos.sides[0].active[0]].trapped = False
    else:
        pos = _port_turn(reg, port, start, _charge_turn(target))
        charging = pos.sides[0].pokemon[0].volatile("twoturnmove")
        assert charging is not None and charging.extra.get("targetLoc") == target
    assert _slot0_choices(reg, pos) == {"move 1"}
    before = handle.position
    handle.step([side_actions(reg, pos, 0)[0].slots[0].to_choice() + ", move 1", FOES_QUIET])
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _hurt(before, handle.position)
    handle.close()
    after = _port_turn(reg, port, pos, ["move 1, move 1", FOES_QUIET])
    assert _hurt(pos, after) == theirs == {target}


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(HAMMER))
def test_the_ports_hammer_child_has_showdowns_menu(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    """`test_our_hammer_menu_is_showdowns` on the position the port's turns build."""
    handle, first = _hammer(oracle, name)
    theirs = _showdown_moves(handle)
    handle.close()
    pos = Position.from_json(first)
    for step in HAMMER[name][1]:
        pos = _port_turn(reg, port, pos, step)
    assert _our_moves(reg, pos) == theirs
