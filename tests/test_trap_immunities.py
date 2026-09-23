"""Ghost types and Shed Shell holders are not trapped (IKA-163).

Every volatile in `TRAPPING_VOLATILES` below traps through `pokemon.tryTrap()`
(`trapped.onTrapPokemon`, `partiallytrapped.onTrapPokemon`, `octolock`'s condition), and
`tryTrap` begins

    if (!this.runStatusImmunity('trapped')) return false;

which is the type chart's `ghost: { damageTaken: { trapped: 3 } }` -- the dump's
`effectImmunities.trapped`. Shed Shell has `onTrapPokemonPriority: -10` and sets
`pokemon.trapped = false` after every trapper has run, the same shape as the champions
mod's Run Away (IKA-136).

A position straight from Showdown carries Showdown's own verdict in `trapped`. A position
our resolver builds carries the trapper's volatile and a `trapped` flag nobody computes,
and `actions._is_trapped` read the volatile as a trap: a Dragapult under Infestation, or
anything holding Shed Shell under Mean Look, was never offered a switch. Generation plays
its games in that resolver (`selfplay.play_game` starts from `position_from_sets`), so
there it was every such decision, the root of the search included, not only its children.

Mean Look, Block and Octolock do not land on a Ghost at all (`addVolatile` runs the same
immunity, Octolock's `onTryImmunity` asks for it by name), so the Ghost cases that reach a
child are the binding moves. The oracle tests pin that too.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou.actions import TRAPPING_CONDITIONS, SwitchAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from .conftest import FORMAT_ID

#: Every trapping volatile: the two conditions, and the three a move's condition sets
#: (IKA-169; `test_trap_sources.test_the_trap_sources_come_from_the_dump` pins them).
TRAPPING_VOLATILES = TRAPPING_CONDITIONS | {"ingrain", "noretreat", "octolock"}

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


#: Each subject spends turn 1 on a status move that is not Protect.
SUBJECTS = {
    "ghost": _mon("Dragapult", "Clear Body", ["dragondance", "protect", "dragondarts", "uturn"]),
    "shedshell": _mon(
        "Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"],
        "Shed Shell",
    ),
    # The control: the same Garchomp with nothing in its hand.
    "control": _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"]),
}

#: One user of each trapping volatile the regulation has, with its first move the trap.
TRAPPERS = {
    "partiallytrapped": _mon(
        "Venusaur", "Chlorophyll", ["infestation", "protect", "sludgebomb", "gigadrain"]
    ),
    "trapped": _mon("Umbreon", "Synchronize", ["meanlook", "protect", "foulplay", "moonlight"]),
    "octolock": _mon("Grapploct", "Limber", ["octolock", "protect", "drainpunch", "bulkup"]),
}


def _team_a(subject: str) -> list[TeamSet]:
    return [
        SUBJECTS[subject],
        _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
        _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
        _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
    ]


def _team_b(trap: str) -> list[TeamSet]:
    return [
        TRAPPERS[trap],
        _mon("Hippowdon", "Sand Stream", ["slackoff", "protect", "earthquake", "yawn"]),
        _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
        _mon("Kingambit", "Defiant", ["kowtowcleave", "protect", "suckerpunch", "ironhead"]),
    ]


#: (subject, trapping volatile, does the volatile land, is the subject trapped)
CASES = [
    ("ghost", "partiallytrapped", True, False),
    ("ghost", "trapped", False, False),
    ("ghost", "octolock", False, False),
    ("shedshell", "partiallytrapped", True, False),
    ("shedshell", "trapped", True, False),
    ("shedshell", "octolock", True, False),
    ("control", "partiallytrapped", True, True),
    ("control", "trapped", True, True),
    ("control", "octolock", True, True),
]


def _after_the_trap(oracle: Oracle, subject: str, trap: str):  # noqa: ANN202
    handle = oracle.create(
        FORMAT_ID, _team_a(subject), _team_b(trap), policy=RandomnessPolicy()
    )
    handle.step(["team 1234", "team 1234"])
    # The subject and Milotic use their first move; the trapper aims its first at p1a.
    handle.step(["move 1, move 1", "move 1 1, move 2"])
    assert handle.choice_errors == [], handle.choice_errors
    return handle


def _switches_for_slot0(reg, pos: Position) -> set[int]:  # noqa: ANN001
    return {
        piece.party_index
        for action in side_actions(reg, pos, 0)
        for piece in action.slots
        if isinstance(piece, SwitchAction) and piece.slot == 0
    }


@pytest.mark.oracle
@pytest.mark.parametrize(("subject", "volatile", "lands", "trapped"), CASES)
def test_showdown_verdict(
    oracle: Oracle, subject: str, volatile: str, lands: bool, trapped: bool
) -> None:
    """The fact: which volatiles land, and whether Showdown's request says trapped."""
    handle = _after_the_trap(oracle, subject, volatile)
    pos = Position.from_json(handle.position)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert mon.has_volatile(volatile) is lands, [v.id for v in mon.volatiles]
    active = handle.requests[0]["active"][0]
    assert bool(active.get("trapped")) is trapped, active
    handle.close()


@pytest.mark.oracle
@pytest.mark.parametrize(("subject", "volatile", "lands", "trapped"), CASES)
def test_we_agree_once_the_flag_is_gone(
    reg,  # noqa: ANN001
    oracle: Oracle,
    subject: str,
    volatile: str,
    lands: bool,
    trapped: bool,
) -> None:
    """Showdown's position with its verdict cleared, as a searched child carries it."""
    del lands
    handle = _after_the_trap(oracle, subject, volatile)
    pos = Position.from_json(handle.position)
    handle.close()
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.trapped = False
    offered = _switches_for_slot0(reg, pos)
    assert (not offered) is trapped, (subject, volatile, offered)


def _hand_built(reg, subject: str):  # noqa: ANN001, ANN202
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, _team_a(subject))
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert mon.species == SUBJECTS[subject].species.lower()
    return pos, mon


@pytest.mark.parametrize("volatile", sorted(TRAPPING_VOLATILES))
def test_ghost_escapes_every_trapping_volatile(reg, volatile: str) -> None:  # noqa: ANN001
    """No oracle: a Ghost is offered its switches, and loses them with the Ghost type."""
    pos, mon = _hand_built(reg, "ghost")
    mon.volatiles.append(Effect(id=volatile))
    assert _switches_for_slot0(reg, pos), "a Ghost must be offered its switches"
    # The control: Soak or Reflect Type leaves Dragapult without its Ghost type.
    mon.types = ("Water",)
    assert not _switches_for_slot0(reg, pos), "without the Ghost type the volatile traps"


@pytest.mark.parametrize("volatile", sorted(TRAPPING_VOLATILES))
def test_shed_shell_escapes_every_trapping_volatile(reg, volatile: str) -> None:  # noqa: ANN001
    """No oracle: Shed Shell frees its holder until the item is gone."""
    pos, mon = _hand_built(reg, "shedshell")
    mon.volatiles.append(Effect(id=volatile))
    assert _switches_for_slot0(reg, pos), "a Shed Shell holder must be offered its switches"
    # The control: Knock Off or Trick has taken the item.
    mon.item = None
    assert not _switches_for_slot0(reg, pos), "without Shed Shell the volatile traps"


def test_the_immunities_come_from_the_dump(reg, monkeypatch) -> None:  # noqa: ANN001
    """Both escapes follow the regulation's data, not a list written here."""
    pos, ghost = _hand_built(reg, "ghost")
    ghost.volatiles.append(Effect(id="partiallytrapped"))
    assert _switches_for_slot0(reg, pos)
    monkeypatch.setattr(reg, "effect_immunities", {})
    assert not _switches_for_slot0(reg, pos), "no `trapped` immunity in the dump, no escape"
    monkeypatch.undo()

    pos, holder = _hand_built(reg, "shedshell")
    holder.volatiles.append(Effect(id="partiallytrapped"))
    assert _switches_for_slot0(reg, pos)
    item = reg.items["shedshell"]
    hookless = dataclasses.replace(item, raw={**item.raw, "customHooks": []})
    monkeypatch.setitem(reg.items, "shedshell", hookless)
    assert not _switches_for_slot0(reg, pos), "a Shed Shell with no hook in the dump does nothing"


def test_the_oracle_flag_still_wins(reg) -> None:  # noqa: ANN001
    """Showdown's own `trapped` stays authoritative: the escapes only read the volatiles."""
    pos, ghost = _hand_built(reg, "ghost")
    ghost.trapped = True
    assert not _switches_for_slot0(reg, pos)
