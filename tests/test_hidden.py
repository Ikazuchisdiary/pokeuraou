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


# -------------------------------------------------- carried across a switch (IKA-117)
#
# Every test above builds its position by hand, so none of them ever went through
# `_do_switch`, which swaps party indices the way Showdown does. The ones below get
# their second position from the resolver: side 1's Charizard goes back for Garchomp
# while everyone else Protects, so it reaches the bench with full HP, no status and no
# boosts -- nothing on the board says it was ever out -- and it now sits at the index
# Garchomp vacated.

#: Turn 1. Both of our leads Protect; theirs send Charizard back for Garchomp (party
#: index 3, 1-based) and Venusaur Protects. Nothing takes damage.
TURN_ONE = {0: "move 4, move 4", 1: "switch 3, move 4"}


def _action(reg, position, side, choice):  # noqa: ANN001, ANN202
    from pokeuraou.actions import side_actions

    for action in side_actions(reg, position, side):
        if action.to_choice() == choice:
            return action
    raise AssertionError(f"side {side} has no {choice!r} here")


def _charizard_went_back(reg, position):  # noqa: ANN001, ANN202
    """The turn-2 position, resolved from `position` by the real resolver."""
    from pokeuraou.resolve import Budget, resolve_turn

    result = resolve_turn(
        reg,
        position,
        [_action(reg, position, side, TURN_ONE[side]) for side in (0, 1)],
        budget=Budget.exact(),
    )
    assert result.branches and not result.suspended
    after = max(result.branches, key=lambda branch: branch.probability).position
    _assert_it_went_back_unharmed(after)
    return after


def _assert_it_went_back_unharmed(position) -> None:  # noqa: ANN001
    """The precondition: without it the tests below pass against a turn that did nothing."""
    theirs = position.sides[1].pokemon
    assert [mon.species for mon in theirs] == [
        "garchomp", "venusaur", "charizard", "sylveon",
    ], "the switch did not swap party indices the way the tests below assume"
    back = theirs[2]
    assert back.active_index is None
    assert back.hp == back.maxhp and back.status is None and not back.boosts
    assert not back.volatiles and not back.is_mega


def test_a_seen_pokemon_is_followed_through_a_switch_not_its_old_number(setup) -> None:  # noqa: ANN001
    from pokeuraou.hidden import seen_identities

    reg, roster = setup
    first, _brought = _opening(reg, _sheet(roster))
    second = _charizard_went_back(reg, first)
    carried = seen_identities(second, 1, seen_identities(first, 1))
    assert carried == frozenset({"charizard", "venusaur", "garchomp"})
    assert seen_slots(second, 1, carried) == frozenset({0, 1, 2})
    # The board alone no longer shows the Charizard at all.
    assert seen_slots(second, 1) == frozenset({0, 1})


def test_a_carried_set_of_slot_numbers_is_refused(setup) -> None:  # noqa: ANN001
    """The shape IKA-117 removed. A set of numbers passes every set operation a set of
    names does, so without this it would just forget again, quietly."""
    from pokeuraou.hidden import seen_identities

    reg, roster = setup
    first, _brought = _opening(reg, _sheet(roster))
    second = _charizard_went_back(reg, first)
    with pytest.raises(TypeError, match="party slots"):
        seen_slots(second, 1, seen_slots(first, 1))
    with pytest.raises(TypeError, match="party slots"):
        seen_identities(second, 1, frozenset({0, 1}))


def test_play_game_keeps_a_lead_that_went_back_in_every_world(setup, monkeypatch) -> None:  # noqa: ANN001
    """What `play_game` hands `completions` after the switch, read off the real driver.

    `data/ika73/w12` game 49, side 1, is this exact shape: by turn 2 the belief was six
    completions over two slots, three of them without the Garchomp that had led turn 1.
    The only Pokemon really unseen here is the Sylveon, so the belief is three worlds, and
    every one of them still holds the Charizard where it stands.

    The menus are scripted so the turn is the one above; everything else -- the carrying,
    the resolver, `_do_switch` -- is the driver's own.
    """
    import types

    import numpy as np

    from pokeuraou import selfplay
    from pokeuraou.actions import side_actions

    reg, roster = setup
    sheet = _sheet(roster)

    def scripted(reg, position, side, *, limit, rank=None):  # noqa: ANN001, ANN202, ARG001
        if position.turn == 1:
            actions = [_action(reg, position, side, TURN_ONE[side])]
        else:
            actions = side_actions(reg, position, side)[:1]
        return types.SimpleNamespace(actions=actions)

    calls = []
    real = selfplay.completions

    def spy(reg, position, side, sheet, *, seen=None, weights=None):  # noqa: ANN001, ANN202
        made = real(reg, position, side, sheet, seen=seen, weights=weights)
        calls.append((position.turn, side, seen, position, made))
        return made

    monkeypatch.setattr(selfplay, "narrow", scripted)
    monkeypatch.setattr(selfplay, "completions", spy)
    selfplay.play_game(
        reg, np.random.default_rng(0), sheet[:4], sheet[:4], "test",
        sheets=(sheet, sheet), max_turns=1,
    )

    at_two = [call for call in calls if call[0] == 2 and call[1] == 1]
    assert at_two, "play_game never reached turn 2's belief about side 1"
    _turn, _side, seen, position, made = at_two[0]
    _assert_it_went_back_unharmed(position)
    without = [
        world.species
        for world in made
        if all(mon.species != "charizard" for mon in world.position.sides[1].pokemon)
    ]
    assert not without, (
        f"{len(without)} of {len(made)} worlds have no Charizard, which led turn 1: "
        f"{without}"
    )
    assert seen == frozenset({0, 1, 2})
    assert len(made) == 3, [world.species for world in made]
    for world in made:
        assert world.slots == (3,)
        assert world.position.sides[1].pokemon[2].species == "charizard"


def test_play_game_hands_the_bench_prior_the_turn_one_leads_after_a_switch(
    setup,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
) -> None:
    """IKA-118: the lead pair reaches the bench prior, and is still the lead pair after
    one of the leads has gone back to the bench.

    Side 1 led Charizard and Venusaur; by turn 2 Garchomp stands where Charizard did and
    Charizard sits at another index. What the prior must be told is who LED -- read once
    at turn 1 and carried as names -- not who is active now, and not whatever stands at
    the indices that led.
    """
    import types

    import numpy as np

    from pokeuraou import selfplay
    from pokeuraou.actions import side_actions
    from pokeuraou.regulation import to_id

    reg, roster = setup
    sheet = _sheet(roster)

    def scripted(reg, position, side, *, limit, rank=None):  # noqa: ANN001, ANN202, ARG001
        if position.turn == 1:
            actions = [_action(reg, position, side, TURN_ONE[side])]
        else:
            actions = side_actions(reg, position, side)[:1]
        return types.SimpleNamespace(actions=actions)

    class Recording:
        """Stands in for a `BenchPrior`: records what it is asked, answers uniform."""

        def __init__(self) -> None:
            self.leads: list[frozenset[str] | None] = []

        def weights(self, seen, leads=None):  # noqa: ANN001, ANN202, ARG002
            self.leads.append(leads)
            return {}

    priors = (Recording(), Recording())
    positions = []
    real = selfplay._bench_weights

    def spy(bench_prior, side, pos, seen, record, leads=None):  # noqa: ANN001, ANN202
        if side == 1:
            positions.append(pos)
        return real(bench_prior, side, pos, seen, record, leads)

    monkeypatch.setattr(selfplay, "narrow", scripted)
    monkeypatch.setattr(selfplay, "_bench_weights", spy)
    selfplay.play_game(
        reg, np.random.default_rng(0), sheet[:4], sheet[:4], "test",
        sheets=(sheet, sheet), bench_prior=priors, max_turns=1,
    )

    asked = list(zip(positions, priors[1].leads, strict=True))
    assert any(pos.turn == 2 for pos, _ in asked), "never reached turn 2's belief"
    names = {to_id(s.species) for s in sheet}
    led = {to_id(sheet[0].species), to_id(sheet[1].species)}
    assert led == {"charizard", "venusaur"}
    for pos, leads in asked:
        assert leads is not None, f"turn {pos.turn}: the prior was not told who led"
        assert {name for name in leads if name in names} == led, (pos.turn, leads)
    # The turn-2 board really is the one with a lead on the bench.
    at_two = next(pos for pos, _ in asked if pos.turn == 2)
    _assert_it_went_back_unharmed(at_two)
