"""Hyper Beam's recharge turn: the whole request, not just the move.

Reported from a game log, and the log was the reason it was invisible for so long -- the
volatile was being *set* all along (`self: {volatileStatus: 'mustrecharge'}` comes straight
from the move data and the generic self-effect path applied it), so a position dump looked
right. Nothing read it. The recharging Pokemon kept its four moves, kept the option to
switch out, and carried the volatile for the rest of the game. Hyper Beam was a 150-power
move with no cost, and one worker's 500 games contained 569 of them.

Showdown does not express this as "the move fails". It replaces the request:

    if (lockedMove === 'recharge') return [{ move: 'Recharge', id: 'recharge' }];

together with `trapped: true`. That distinction is the whole bug. A resolver-side check
that made the move fail would still leave the search choosing among four moves and a
switch, and the action set is what the equilibrium is computed over -- so the mixture would
put weight on choices Showdown rejects, and the search could price its way out of the
downside by switching.

The oracle test is the one that matters here, because every claim above is a claim about
what Showdown offers.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import MoveAction, SwitchAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

#: Snorlax leads with Hyper Beam; everything else has a target-free move so that a turn
#: can be driven without Protect accidentally blocking the thing under test. That is not a
#: hypothetical: the first two runs of this probe concluded "Showdown has no recharge
#: mechanic", once because the other side's choice was illegal and the whole step was
#: rejected, and once because Blissey happened to Protect -- and a move blocked by Protect
#: genuinely gives no recharge, so the wrong conclusion looked confirmed.
SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str]) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP)
    )


TEAM_A = [
    _mon("Snorlax", "Immunity", ["hyperbeam", "bodyslam", "protect", "yawn"]),
    _mon("Toxapex", "Regenerator", ["toxic", "recover", "scald", "protect"]),
    _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
TEAM_B = [
    # Milotic rather than something with Soft-Boiled: the `reg` fixture is regulation
    # M-C, and a species outside it has no entry to build a Battler from.
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"]),
    _mon("Venusaur", "Chlorophyll", ["sludgebomb", "gigadrain", "protect", "leechseed"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "darkestlariat"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]

#: Snorlax Hyper Beam at the foe's first slot, Toxapex Protect; the foe heals and Protects,
#: so nothing blocks the Hyper Beam.
CONNECTS = ["move 1 1, move 4", "move 1, move 3"]
#: The same, except the target Protects.
BLOCKED = ["move 1 1, move 4", "move 2, move 3"]


def _battle(oracle: Oracle, policy: RandomnessPolicy | None = None):  # noqa: ANN202
    handle = oracle.create(
        FORMAT_ID, TEAM_A, TEAM_B, policy=policy or RandomnessPolicy()
    )
    handle.step(["team 1234", "team 1234"])
    return handle


def _slot_actions(reg, pos: Position, side: int, slot: int):  # noqa: ANN001, ANN202
    """Every action our generator offers for one slot, as the slot-level pieces."""
    out = []
    for action in side_actions(reg, pos, side):
        for piece in action.slots:
            if piece.slot == slot:
                out.append(piece)
    return out


def test_showdown_offers_only_the_recharge(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """What Showdown's request actually is, so the rest of the file rests on a fact."""
    handle = _battle(oracle)
    handle.step(CONNECTS)
    assert handle.choice_errors == [], handle.choice_errors
    active = handle.requests[0]["active"][0]
    assert [m["id"] for m in active["moves"]] == ["recharge"], active
    assert active.get("trapped") is True, f"a recharging Pokemon cannot switch: {active}"
    handle.close()


def test_we_offer_only_the_recharge_too(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Our enumeration, on the position Showdown produced."""
    handle = _battle(oracle)
    handle.step(CONNECTS)
    pos = Position.from_json(handle.position)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert mon.has_volatile("mustrecharge"), (
        f"the volatile has to be in the position for anything else to work: {mon.volatiles}"
    )
    offered = _slot_actions(reg, pos, 0, 0)
    assert all(isinstance(a, MoveAction) for a in offered), (
        f"no switch may be offered on a recharge turn: {[type(a).__name__ for a in offered]}"
    )
    assert not any(isinstance(a, SwitchAction) for a in offered)
    assert {a.move_id for a in offered} == {"recharge"}, (
        f"one fake move and nothing else: {[a.move_id for a in offered]}"
    )
    # Distinct, not raw: a side action is a pair, so slot 0's one action appears once per
    # action the partner has. What matters is that the slot itself has a single choice.
    assert len(set(offered)) == 1, f"and no target variants: {set(offered)}"
    handle.close()


def test_a_blocked_hyper_beam_costs_nothing(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Protect blocks the move, so the self-effect never applies and there is no lock.

    This is the case that made the mechanic look absent, and it is genuine behaviour
    rather than an artefact: `self` is applied from the hit, so a move that connected with
    nothing applies nothing.
    """
    handle = _battle(oracle)
    handle.step(BLOCKED)
    assert handle.choice_errors == [], handle.choice_errors
    active = handle.requests[0]["active"][0]
    assert [m["id"] for m in active["moves"]] != ["recharge"], (
        f"a blocked Hyper Beam gives no recharge turn: {active}"
    )
    pos = Position.from_json(handle.position)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert not mon.has_volatile("mustrecharge"), mon.volatiles
    offered = _slot_actions(reg, pos, 0, 0)
    assert any(isinstance(a, SwitchAction) for a in offered), (
        "and the Pokemon is not trapped"
    )
    handle.close()


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


@pytest.mark.oracle
def test_the_ports_recharge_turn_lifts_the_lock_and_spends_no_pp(reg, oracle: Oracle, port) -> None:  # noqa: ANN001
    """`test_the_recharge_turn_lifts_the_lock_and_spends_no_pp` with the port, and the
    port's pinned outcome held to Showdown's own position after the recharge."""
    from ._port_showdown import port_branches, port_turn

    handle = _battle(oracle)
    handle.step(CONNECTS)
    pos = Position.from_json(handle.position)
    before = [(m.id, m.pp) for m in pos.sides[0].pokemon[pos.sides[0].active[0]].moves]
    ours = next(
        a for a in side_actions(reg, pos, 0) if any(
            isinstance(p, MoveAction) and p.move_id == "recharge" for p in a.slots
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    branches = port_branches(port, pos, [ours, theirs], Budget.exact())
    assert branches, "the recharge turn has to resolve"
    after_pos = max(branches, key=lambda b: b[0])[1]
    mon = after_pos.sides[0].pokemon[after_pos.sides[0].active[0]]
    assert not mon.has_volatile("mustrecharge"), mon.volatiles
    assert [(m.id, m.pp) for m in mon.moves] == before

    handle.step([ours.to_choice(), theirs.to_choice()])
    assert handle.choice_errors == [], handle.choice_errors
    active = handle.requests[0]["active"][0]
    assert [m["id"] for m in active["moves"]] == [
        "hyperbeam", "bodyslam", "protect", "yawn"
    ], f"the full move set is back the turn after: {active}"
    assert any("cant" in line and "recharge" in line for line in handle.log), (
        f"Showdown logs the spent turn: {handle.log}"
    )
    showdown = Position.from_json(handle.position)
    handle.close()
    pinned = port_turn(port, pos, [ours, theirs])
    for side, theirs_side in zip(pinned.sides, showdown.sides, strict=True):
        for a, b in zip(side.pokemon, theirs_side.pokemon, strict=True):
            assert (a.species, a.hp, a.has_volatile("mustrecharge"), [(m.id, m.pp) for m in a.moves]) == (
                b.species, b.hp, b.has_volatile("mustrecharge"), [(m.id, m.pp) for m in b.moves]
            )
