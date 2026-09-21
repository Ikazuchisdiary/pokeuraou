"""The opponent's unseen bench, and the promise that a completion is a real position.

Everything downstream -- `narrow`, the payoff matrix, the value function -- takes a
`Position` and asks no questions. So a completion that is subtly malformed does not fail,
it evaluates, and the number it returns is wrong in a way nothing reports. These tests are
about the substitution being total (the slot is rebuilt, not patched) and about it leaving
the rest of the board exactly where it was.

The count is the other half. There are C(4,2) = 6 ways to fill two unseen slots from four
unaccounted sheet members, and a caller that averages over them is paying six times for
the turn -- so a bug that returns one, or twenty-four, is a cost bug and a correctness bug
at once.
"""

from __future__ import annotations

import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions, seen_slots
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    return reg, roster


def _sheet(roster):  # noqa: ANN001, ANN202
    return list(roster.sets)[:6]


def _opening(reg, sheet):  # noqa: ANN001, ANN202
    """Turn 1 with the first four of the sheet brought, slots 0 and 1 leading."""
    return position_from_sets(reg, sheet[:4], sheet[:4]), sheet[:4]


def test_only_the_leads_count_as_seen_at_turn_one(setup) -> None:  # noqa: ANN001
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    assert seen_slots(position, 1) == frozenset({0, 1})


def test_two_unseen_slots_from_four_candidates_make_six_completions(setup) -> None:  # noqa: ANN001
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    made = completions(reg, position, 1, sheet)
    assert len(made) == 6
    assert {c.slots for c in made} == {(2, 3)}
    # Every one of them is a distinct pair, and none repeats a Pokemon already on board.
    leads = {position.sides[1].pokemon[i].species for i in (0, 1)}
    assert len({tuple(sorted(c.species)) for c in made}) == 6
    assert all(not (set(c.species) & leads) for c in made)


def test_a_mega_on_the_board_is_not_also_a_candidate(setup) -> None:  # noqa: ANN001
    """The board calls it `charizardmegay`; the sheet calls it `charizard`.

    Found on 2026-09-19 while re-checking a human-baseline case under the information the
    players had. The opponent's Charizard had Mega Evolved, so the sheet's `charizard`
    matched nothing on the board and stayed in the candidate pool: ten completions instead
    of six, four of them holding a second Charizard while the first was still out. Those
    are positions Species Clause forbids, they carried 40% of the belief, and nothing
    complained -- the arity check below only fires when candidates are too FEW, and this
    bug makes them too many.
    """
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    lead = position.sides[1].pokemon[0]
    assert lead.species == "charizard", "the fixture's first sheet member moved"
    lead.species = "charizardmegay"
    # Showdown hands `baseSpecies` back as a display name, which is half the bug.
    lead.base_species = "Charizard"
    lead.is_mega = True

    made = completions(reg, position, 1, sheet)
    assert len(made) == 6
    assert all("charizard" not in c.species for c in made)
    assert all("charizardmegay" not in c.species for c in made)


def test_the_weights_are_a_distribution(setup) -> None:  # noqa: ANN001
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    made = completions(reg, position, 1, sheet)
    assert sum(c.weight for c in made) == pytest.approx(1.0)
    assert all(c.weight == pytest.approx(1 / 6) for c in made), "no book means uniform"


def test_a_book_weight_survives_normalisation(setup) -> None:  # noqa: ANN001
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    plain = completions(reg, position, 1, sheet)
    favoured = tuple(sorted(plain[0].species))
    made = completions(reg, position, 1, sheet, weights={favoured: 3.0})
    by_key = {tuple(sorted(c.species)): c.weight for c in made}
    assert by_key[favoured] == pytest.approx(1.0)
    assert sum(by_key.values()) == pytest.approx(1.0)


def test_nothing_hidden_returns_the_position_itself(setup) -> None:  # noqa: ANN001
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    made = completions(reg, position, 1, sheet, seen=frozenset({0, 1, 2, 3}))
    assert len(made) == 1
    assert made[0].exact
    assert made[0].position is position
    assert made[0].weight == pytest.approx(1.0)


def test_our_own_side_is_untouched(setup) -> None:  # noqa: ANN001
    """A completion edits one side. Side 0 is ours and we know it exactly."""
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    before = position.sides[0].to_json()
    for made in completions(reg, position, 1, sheet):
        assert made.position.sides[0].to_json() == before


def test_the_seen_pokemon_keep_their_state(setup) -> None:  # noqa: ANN001
    """Substitution is per slot. Everything the opponent has shown stays as it was."""
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    position.sides[1].pokemon[0].hp = position.sides[1].pokemon[0].maxhp // 2
    position.sides[1].pokemon[1].status = "brn"
    before = [position.sides[1].pokemon[i].to_json() for i in (0, 1)]
    for made in completions(reg, position, 1, sheet):
        assert [made.position.sides[1].pokemon[i].to_json() for i in (0, 1)] == before


def test_a_substituted_pokemon_is_fresh(setup) -> None:  # noqa: ANN001
    """Never having been active means full HP, no status, full PP -- from the sheet, not
    inherited from whoever the slot used to hold."""
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    for made in completions(reg, position, 1, sheet):
        for slot in made.slots:
            mon = made.position.sides[1].pokemon[slot]
            assert mon.hp == mon.maxhp
            assert mon.status is None
            assert not mon.boosts
            assert not mon.volatiles
            assert mon.active_index is None
            assert all(m.pp == m.maxpp for m in mon.moves)


def test_mega_capability_follows_the_substituted_species(setup) -> None:  # noqa: ANN001
    """A stone belongs to the Pokemon, not to the slot. Carrying the old side's list over
    would let the search declare a Mega on a Pokemon that cannot hold the stone."""
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    for made in completions(reg, position, 1, sheet):
        side = made.position.sides[1]
        expected = [
            mon.slot
            for mon in side.pokemon
            if reg.mega_target(mon.species, mon.item) is not None
        ]
        assert side.mega_capable_slots == expected


def test_every_completion_reads_back(setup) -> None:  # noqa: ANN001
    """The wire format is what the simulator and the encoder consume. A completion that
    cannot round-trip is a position nothing downstream can be trusted with."""
    from pokeuraou.position import Position, validate_position

    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    for made in completions(reg, position, 1, sheet):
        again = Position.from_json(made.position.to_json())
        assert again.to_json() == made.position.to_json()
        problems = validate_position(again, reg.meta.active_per_side)
        assert not problems, problems


def test_a_sheet_that_cannot_explain_the_board_is_refused(setup) -> None:  # noqa: ANN001
    """Silently carrying on would mean searching the truth, which is the leak."""
    reg, roster = setup
    sheet = _sheet(roster)
    position, _brought = _opening(reg, sheet)
    with pytest.raises(ValueError, match="unseen slots"):
        completions(reg, position, 1, sheet[:3])
