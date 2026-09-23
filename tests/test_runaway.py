"""Run Away frees its holder from traps in Champions (IKA-136).

Showdown d3de52a17's parent d849b2200 ("Champions: Implement Run Away's immunity to
trapping") gives the champions mod's Run Away

    onTrapPokemonPriority: -10,
    onTrapPokemon(pokemon) { pokemon.trapped = false; },

so it runs after every trapper (Infestation's `partiallytrapped`, Mean Look's `trapped`,
Octolock) and undoes it. Before the bump the dump had Run Away as an ability with no code,
and our enumeration still has no reason to know: a position straight from Showdown carries
Showdown's own `trapped` verdict, which already accounts for it, but a position the search
builds carries the trapper's volatile and a `trapped` flag nobody recomputed, and
`actions._is_trapped` read the volatile as a trap. In M-C the holder is Thievul.

The oracle tests pin the fact first, with a control that is trapped by the same move.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import SwitchAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str]) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP))


def _team_a(ability: str) -> list[TeamSet]:
    return [
        _mon("Thievul", ability, ["protect", "nastyplot", "darkpulse", "foulplay"]),
        _mon("Toxapex", "Regenerator", ["recover", "protect", "scald", "toxic"]),
        _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"]),
        _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
    ]


TEAM_B = [
    # Infestation never misses at 100 accuracy under the pinned policy, and is Bug so the
    # Dark Thievul takes it; Milotic only Recovers so nothing else happens.
    _mon("Venusaur", "Chlorophyll", ["infestation", "protect", "sludgebomb", "gigadrain"]),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "darkestlariat"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]

#: Thievul Nasty Plots, Toxapex Recovers; Venusaur Infests Thievul, Milotic Recovers.
INFEST = ["move 2, move 1", "move 1 1, move 1"]


def _infested(oracle: Oracle, ability: str):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, _team_a(ability), TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    handle.step(INFEST)
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
@pytest.mark.parametrize(("ability", "trapped"), [("Run Away", False), ("Unburden", True)])
def test_showdown_frees_run_away_from_infestation(oracle: Oracle, ability: str, trapped: bool) -> None:
    """The fact, with its control: the same Infestation traps Thievul without Run Away."""
    handle = _infested(oracle, ability)
    pos = Position.from_json(handle.position)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert mon.has_volatile("partiallytrapped"), mon.volatiles
    active = handle.requests[0]["active"][0]
    assert bool(active.get("trapped")) is trapped, active
    handle.close()


@pytest.mark.oracle
@pytest.mark.parametrize(("ability", "trapped"), [("Run Away", False), ("Unburden", True)])
def test_we_agree_once_the_flag_is_gone(
    reg,  # noqa: ANN001
    oracle: Oracle,
    ability: str,
    trapped: bool,
) -> None:
    """Showdown's position with its verdict cleared, as a searched child carries it."""
    handle = _infested(oracle, ability)
    pos = Position.from_json(handle.position)
    handle.close()
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.trapped = False
    offered = _switches_for_slot0(reg, pos)
    assert (not offered) is trapped, (ability, offered)


@pytest.mark.parametrize("volatile", ["partiallytrapped", "trapped", "octolock"])
def test_run_away_escapes_every_trapping_volatile(reg, volatile: str) -> None:  # noqa: ANN001
    """No oracle: a hand-built position, Run Away against the control ability."""
    from .test_actions import _synthetic_position

    team = _team_a("Run Away")
    pos = _synthetic_position(reg, team)
    thievul = next(m for m in pos.sides[0].pokemon if m.species == "thievul")
    assert thievul.active_index is not None, "the fixture has to put Thievul on the field"
    thievul.volatiles.append(Effect(id=volatile))

    def switches() -> set[int]:
        return {
            piece.party_index
            for action in side_actions(reg, pos, 0)
            for piece in action.slots
            if isinstance(piece, SwitchAction) and piece.slot == thievul.active_index
        }

    assert switches(), "Run Away holder must be offered its switches"
    thievul.ability = "unburden"
    assert not switches(), "the control: without Run Away the volatile traps"
