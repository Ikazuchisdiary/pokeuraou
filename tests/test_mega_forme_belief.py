"""A Mega of a forme keeps its sheet member out of the hidden bench (IKA-411).

`hidden.shown_species` tells the sheet member behind a Mega by the Pokemon's `base_species`.
`position.ts` writes that as the *set's* species id, and Showdown keeps it through the Mega
(`floettemega` stays `baseSpecies: floetteeternal`). `selfplay._make_pokemon` wrote the dex's
`baseSpecies` instead, which is the same thing for every M-C Mega but two: Floette-Eternal
and Meowstic-F, whose dex base is "Floette" and "Meowstic". Once those Mega Evolved, neither
the board's species nor its base named the sheet member, so the belief put a second Floette
on the bench -- C(5,2) = 10 completions where there were 6, four of them worlds Species
Clause forbids -- and the bench prior, asked about a board that no longer named Floette,
priced none of them, so the belief fell back to uniform without a note. Found in IKA-410's
replays of generation games (6 -> 10 at turn 2, three of 396 examples).
"""

from __future__ import annotations

import itertools

import pytest

from pokeuraou import port
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions, seen_identities, seen_slots, shown_species
from pokeuraou.priors import SampledSet
from pokeuraou.regulation import Regulation, to_id
from pokeuraou.selection_book import bench_weights
from pokeuraou.selfplay import position_from_sets

#: (set species, stone): the two formes the old base species lost, and Charizard as the
#: control whose dex base is its set species (it passed before the fix too).
MEGAS = [
    ("floetteeternal", "floettite"),
    ("meowsticf", "meowsticite"),
    ("charizard", "charizarditey"),
]
FOE_FILL = ["incineroar", "kingambit", "sneasler", "gholdengo", "whimsicott"]
OWN = ["rillaboom", "garchomp", "basculegion", "archaludon"]


def _set(reg: Regulation, species: str, item: str | None = None) -> SampledSet:
    return SampledSet(
        species=species, ability=to_id(reg.species[species].abilities[0]), item=item,
        nature="Hardy", sp={}, moves=["protect"],
    )


@pytest.fixture(scope="module")
def mreg(reg: Regulation) -> Regulation:
    register_mega_stones(reg)
    return reg


def _opening(reg: Regulation, species: str, stone: str):  # noqa: ANN202
    sheet = [_set(reg, species, stone)] + [_set(reg, s) for s in FOE_FILL]
    own = [_set(reg, s) for s in OWN]
    return position_from_sets(reg, own, sheet[:4]), sheet


@pytest.mark.parametrize(("species", "stone"), MEGAS)
def test_a_party_member_carries_its_set_species_as_base_species(
    mreg: Regulation, species: str, stone: str
) -> None:
    """The same `baseSpecies` `position.ts` writes (`dex.species.get(set.species).id`)."""
    position, _sheet = _opening(mreg, species, stone)
    assert position.sides[1].pokemon[0].base_species == species


@pytest.mark.parametrize(("species", "stone"), MEGAS)
def test_the_completions_do_not_grow_when_a_lead_mega_evolves(
    mreg: Regulation, species: str, stone: str
) -> None:
    position, sheet = _opening(mreg, species, stone)
    carried = seen_identities(position, 1)
    before = completions(mreg, position, 1, sheet, seen=seen_slots(position, 1, carried))
    assert len(before) == 6

    ours = next(a for a in side_actions(mreg, position, 0) if not a.declares_mega)
    theirs = next(a for a in side_actions(mreg, position, 1) if a.declares_mega)
    after_position = port.turn(mreg, position, [ours, theirs], Budget.exact(), full=True).outcomes[0].position
    mega = after_position.sides[1].pokemon[0]
    assert mega.is_mega and mega.species == mreg.mega_target(species, sheet[0].item), "no Mega"

    carried = seen_identities(after_position, 1, carried)
    shown = seen_slots(after_position, 1, carried)
    after = completions(mreg, after_position, 1, sheet, seen=shown)
    assert len(after) == 6
    assert all(species not in item.species for item in after)

    # The bench prior must still explain the board: a uniform book over every ordered four
    # of the six prices each of the six completions, not none of them.
    selections = list(itertools.permutations(range(6), 4))
    leads = shown_species(position, 1, frozenset({0, 1}))
    weights = bench_weights(
        selections, [1.0 / len(selections)] * len(selections), [s.species for s in sheet],
        shown_species(after_position, 1, shown), leads=leads,
    )
    priced = sum(weights.get(tuple(sorted(item.species)), 0.0) for item in after)
    assert priced == pytest.approx(1.0)


@pytest.mark.oracle
@pytest.mark.parametrize(("species", "stone"), MEGAS)
def test_showdown_keeps_the_set_species_as_base_species_through_the_mega(
    oracle, mreg: Regulation, species: str, stone: str  # noqa: ANN001
) -> None:
    """The fact the fix rests on, from Showdown's own snapshot."""
    from pokeuraou.oracle import TeamSet

    def team_set(sid: str, item: str | None = None) -> TeamSet:
        found = mreg.species[sid]
        return TeamSet(
            species=found.name, ability=found.abilities[0], nature="Serious",
            moves=["protect", "helpinghand", "tackle", "growl"],
            item=mreg.items[item].name if item else None,
            sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 10},
        )

    battle = oracle.create(
        "gen9championsvgc2026regmc",
        [team_set("incineroar"), team_set("rillaboom")],
        [team_set(species, stone), team_set("kingambit"), team_set("gholdengo")],
    )
    battle.step(["team 12", "team 123"])
    battle.step(["move 1, move 1", "move 1 mega, move 1"])
    assert battle.choice_errors == []
    snap = battle.position["sides"][1]["pokemon"][0]
    assert snap["isMega"] and snap["species"] == mreg.mega_target(species, stone)
    assert snap["baseSpecies"] == species

    position, _sheet = _opening(mreg, species, stone)
    assert position.sides[1].pokemon[0].base_species == snap["baseSpecies"]
