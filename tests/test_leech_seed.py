"""Leech Seed's drained HP goes to whoever stands in the planter's slot (IKA-56).

Showdown, vendor/pokemon-showdown/data/moves.ts:10218-10227:

    onResidual(pokemon) {
        const target = this.getAtSlot(pokemon.volatiles['leechseed'].sourceSlot);
        if (!target || target.fainted || target.hp <= 0) { return; }
        const damage = this.damage(pokemon.baseMaxhp / 8, pokemon, target);
        if (damage) { this.heal(damage, target, pokemon); }
    },

and the slot is written when the volatile is added (sim/pokemon.ts:2008,
``sourceSlot = source.getSlot()``). The resolver never wrote it, so every seed it planted
had ``source_slot = None``, the planter lookup always failed, and the drained HP went
nowhere. The port wrote ``"10"`` and healed; a bridge ON/OFF pair disagreed exactly here.

Three consequences of the quoted rule are pinned:

* the planter heals by what was drained (against the oracle, on a seed *our resolver*
  planted -- a seed read from Showdown carries ``p2a`` and was always parsed);
* it is a slot, so a Pokemon that replaced the planter is healed instead;
* an empty or fainted slot makes the seed do nothing at all -- not even the drain.
"""

from __future__ import annotations

import copy

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


# Distinct Speeds so no turn has a tie to roll; no move below draws on chance except
# Leech Seed's accuracy, and the Python branches are filtered to the one where it hit.
TEAM_A = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"], 10),
    _mon("Toxapex", "Regenerator", ["recover", "protect", "scald", "toxic"], 6),
    _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"], 20),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 4),
]
TEAM_B = [
    _mon("Venusaur", "Chlorophyll", ["leechseed", "protect", "growth", "sludgebomb"], 14),
    _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 8),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 24),
    _mon("Arcanine", "Intimidate", ["flareblitz", "extremespeed", "protect", "willowisp"], 30),
]

#: Milotic Scalds Venusaur so it has HP to get back; everyone else does nothing to HP.
DAMAGE_THE_PLANTER = ["move 3 1, move 2", "move 3, move 1"]
#: Venusaur seeds Milotic; the rest do nothing to HP (full-HP Recover fails, Calm Mind).
PLANT = ["move 1, move 1", "move 1 1, move 2"]
#: Nobody touches HP and nobody plants anything: the residual is the whole turn.
QUIET = ["move 1, move 1", "move 3, move 2"]


def _hp(pos: Position) -> dict[str, int]:
    out: dict[str, int] = {}
    for side_index, side in enumerate(pos.sides):
        for party in side.pokemon:
            out[f"p{side_index + 1} {party.species}"] = party.hp
    return out


def _resolve(reg, pos: Position, choices: list[str]) -> list[Position]:  # noqa: ANN001
    actions = [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side])
        for side in (0, 1)
    ]
    result = resolve_turn(reg, pos, actions, budget=Budget.exact())
    assert result.branches, "the turn has to resolve"
    return [b.position for b in result.branches]


def _seeded(pos: Position) -> bool:
    milotic = pos.sides[0].pokemon[pos.sides[0].active[0]]
    return milotic.has_volatile("leechseed")


def _battle(oracle: Oracle):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    handle.step(DAMAGE_THE_PLANTER)
    assert handle.choice_errors == [], handle.choice_errors
    return handle


def _after_planting(oracle: Oracle) -> Position:
    handle = _battle(oracle)
    handle.step(PLANT)
    pos = Position.from_json(handle.position)
    handle.close()
    assert _seeded(pos)
    return pos


@pytest.mark.parametrize("encoding", ["p2a", "10"])
def test_a_fainted_planter_drains_nothing(reg, oracle: Oracle, encoding: str) -> None:  # noqa: ANN001
    """``if (!target || target.fainted || target.hp <= 0) return;`` -- before the damage."""
    pos = _after_planting(oracle)
    seed = pos.sides[0].pokemon[pos.sides[0].active[0]].volatile("leechseed")
    seed.source_slot = encoding
    # Venusaur down in its slot, as a mid-turn KO leaves it for the residual.
    venusaur = pos.sides[1].pokemon[pos.sides[1].active[0]]
    venusaur.hp, venusaur.fainted, venusaur.status = 0, True, "fnt"
    milotic_hp = pos.sides[0].pokemon[pos.sides[0].active[0]].hp
    choices = ["move 2, move 1", "move 2"]
    actions = [
        next(a for a in side_actions(reg, pos, side) if a.to_choice().endswith(choices[side]))
        for side in (0, 1)
    ]
    for branch in resolve_turn(reg, pos, actions, budget=Budget.exact()).branches:
        milotic = branch.position.sides[0].pokemon[branch.position.sides[0].active[0]]
        assert milotic.hp == milotic_hp, "nothing to leech into, so nothing is drained"


def test_whoever_took_the_slot_is_healed(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """A slot, not a Pokemon: Venusaur switches out and Charizard gets the HP."""
    pos = copy.deepcopy(_after_planting(oracle))
    side_b = pos.sides[1]
    charizard = next(m for m in side_b.pokemon if m.species == "charizard")
    charizard.hp = charizard.maxhp // 2
    before = charizard.hp
    milotic = pos.sides[0].pokemon[pos.sides[0].active[0]]
    drain = milotic.maxhp // 8
    for after in _resolve(reg, pos, ["move 2, move 1", "switch 3, move 2"]):
        healed = next(m for m in after.sides[1].pokemon if m.species == "charizard")
        assert healed.hp == before + drain, (
            f"the replacement in the planter's slot is healed: {before} -> {healed.hp}"
        )


@pytest.mark.parametrize("encoding", ["10", "p2a"])
@pytest.mark.parametrize("planter", ["alive", "fainted"])
def test_the_port_drains_by_either_encoding_and_not_into_a_fainted_slot(
    reg, oracle: Oracle, encoding: str, planter: str  # noqa: ANN001
) -> None:
    """Both encodings of the planter's slot drain, and a fainted planter drains nothing
    (the `test_a_fainted_planter_drains_nothing` rule), under the matrix budget.

    Before IKA-56 the port read only ``"10"`` (so a Showdown ``"p2a"`` seed healed nobody
    there) and drained a seeded Pokemon whose planter's slot had fainted.
    """
    pos = _after_planting(oracle)
    # Showdown's position carries each Pokemon's stats, and the port declines any
    # position with an override (it reads that as a Transform). Both engines compute the
    # same stats from the spreads, so dropping them loses nothing here.
    for side in pos.sides:
        for party in side.pokemon:
            party.stats_override = None
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.volatile("leechseed").source_slot = encoding
    venusaur = pos.sides[1].pokemon[pos.sides[1].active[0]]
    choices = ["move 2, move 1", "move 2, move 2"]
    if planter == "fainted":
        venusaur.hp, venusaur.fainted, venusaur.status = 0, True, "fnt"
        choices[1] = "move 2"
    actions = [
        next(a for a in side_actions(reg, pos, side) if a.to_choice().endswith(choices[side]))
        for side in (0, 1)
    ]
    here = resolve_turn(reg, pos, actions, budget=Budget.matrix())
    assert here.branches
    drained = [b.position.sides[0].pokemon[b.position.sides[0].active[0]].hp for b in here.branches]
    if planter == "fainted":
        assert drained == [mon.hp] * len(drained)
    else:
        assert all(hp < mon.hp for hp in drained), (drained, mon.hp)


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


def _port_resolve(reg, port, pos: Position, choices: list[str]) -> list[Position]:  # noqa: ANN001
    from ._port_showdown import port_branches

    actions = [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side])
        for side in (0, 1)
    ]
    out = [p for _, p in port_branches(port, pos, actions, Budget.exact())]
    assert out, "the turn has to resolve"
    return out


@pytest.mark.oracle
def test_the_port_heals_the_planter_by_what_was_drained(reg, oracle: Oracle, port) -> None:  # noqa: ANN001
    """`test_the_planter_is_healed_by_what_was_drained` with the port."""
    handle = _battle(oracle)
    before = Position.from_json(handle.position)
    handle.step(PLANT)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _hp(Position.from_json(handle.position))
    handle.close()
    hit = [p for p in _port_resolve(reg, port, before, PLANT) if _seeded(p)]
    assert hit, "some branch has the seed landing"
    for pos in hit:
        assert _hp(pos) == theirs, f"showdown {theirs} != port {_hp(pos)}"


@pytest.mark.oracle
def test_the_port_heals_from_a_seed_read_from_showdown(reg, oracle: Oracle, port) -> None:  # noqa: ANN001
    """`test_a_seed_read_from_showdown_heals_too` with the port."""
    handle = _battle(oracle)
    handle.step(PLANT)
    before = Position.from_json(handle.position)
    assert _seeded(before)
    handle.step(QUIET)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _hp(Position.from_json(handle.position))
    handle.close()
    for pos in _port_resolve(reg, port, before, QUIET):
        assert _hp(pos) == theirs, f"showdown {theirs} != port {_hp(pos)}"
