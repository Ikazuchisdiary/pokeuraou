"""The cached selection equilibria, and the property they exist to preserve.

Self-play now draws both sides' four-of-six from a solved equilibrium instead of
uniformly. That is only sound if our draw cannot see the opponent's -- otherwise the
generated games are 後出しジャンケン and every number learned from them is a fantasy where
we already knew what they brought. The leak has four possible entrances and each has a test
here:

- **the key is their sheet**, so a cached row cannot be a response to their four;
- **our draw is independent of their spread class**, tested by conditioning on the class
  and comparing our distribution -- this is the one that would silently pass if the two
  draws shared a code path;
- **our strategy has no per-class axis at all**, checked on the signature, because the
  cheapest way to reintroduce the leak is to add a convenience argument;
- **the game plays the spreads the drawn class holds**, so the column strategy the
  opponent used belongs to the opponent who actually turned up.

The exploration mixture gets its own tests for a different reason: with a pure equilibrium
on both sides, 7,000 games would visit one selection pair per opponent and the *next* solve
needs cells for all 8,100 pairs. Coverage is a correctness property of the loop, not a
nicety, so ``perplexity`` is asserted and not merely printed.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pokeuraou.selection import SpreadClass, book_entry, solve_selection
from pokeuraou.selection_book import (
    BenchPrior,
    BookEntry,
    bench_weights,
    entry_bench_weights,
    explore_mixture,
    perplexity,
)
from pokeuraou.standin import RowStub
from pokeuraou.standings import Standings, TeamMember, TournamentTeam
from pokeuraou.teams import all_selections, load_roster

SELECTIONS = tuple(all_selections(6, 4))


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    return load_roster("rizabanadohido")


def _team(sets, *, player: str = "Opponent", place: int = 1) -> TournamentTeam:  # noqa: ANN001
    """A tournament entry whose sheet is these sets, investment excluded."""
    return TournamentTeam(
        player=player,
        place=place,
        country="jp",
        wins=1,
        losses=0,
        made_cut=False,
        phase_two=False,
        members=tuple(
            TeamMember(
                species=s.species,
                ability=s.ability,
                item=s.item,
                nature=s.nature,
                moves=tuple(s.moves),
            )
            for s in sets
        ),
    )


def _entry(*, ours: np.ndarray, theirs: list[np.ndarray], sets) -> BookEntry:  # noqa: ANN001
    """An entry with hand-written strategies, so the draw is checkable by hand."""
    return BookEntry(
        key="k",
        player="Opponent",
        place=1,
        selections=SELECTIONS,
        our_strategy=ours,
        our_ev_loss=np.zeros(len(SELECTIONS)),
        class_weights=np.full(len(theirs), 1.0 / len(theirs)),
        class_sets=tuple(tuple(sets) for _ in theirs),
        their_strategies=tuple(theirs),
        their_ev_loss=tuple(np.zeros(len(SELECTIONS)) for _ in theirs),
        value=0.5,
    )


def _point_mass(index: int) -> np.ndarray:
    out = np.zeros(len(SELECTIONS))
    out[index] = 1.0
    return out


# ------------------------------------------------------------------ the key is the sheet


# ------------------------------------------------------------- no 後出しジャンケン


def test_our_draw_is_independent_of_the_opponents_spread_class(roster) -> None:  # noqa: ANN001
    """Conditioning on their class must not move our distribution at all.

    The two classes here play opposite selections -- 0 against 89 -- so if our draw were
    reading the class in any way, our conditional distributions would differ. They must be
    the same distribution, sampling noise aside, and our recorded equilibrium must be
    literally the same vector.
    """
    ours = np.zeros(len(SELECTIONS))
    ours[:4] = 0.25
    entry = _entry(
        ours=ours,
        theirs=[_point_mass(0), _point_mass(len(SELECTIONS) - 1)],
        sets=roster.sets,
    )
    rng = np.random.default_rng(7)
    per_class: dict[int, list[int]] = {0: [], 1: []}
    for _ in range(4000):
        drawn = entry.draw(rng, epsilon=0.0)
        per_class[drawn.class_index].append(SELECTIONS.index(drawn.our_pick))
        # Their draw is the class's, ours is not indexed by it.
        assert drawn.foe_pick == (
            SELECTIONS[0] if drawn.class_index == 0 else SELECTIONS[-1]
        )
        assert np.array_equal(drawn.our_equilibrium, ours)

    # The same precise pin as the arm test: our selection is the one the stream's first
    # uniform names against our own mixture. Independence alone would survive a reorder
    # (a PRNG's successive draws are independent either way); this pins the claim the
    # docstring actually makes, that nothing downstream of their type precedes our draw.
    first = np.random.default_rng(11).random()
    expected = int(np.searchsorted(np.cumsum(ours / ours.sum()), first, side="right"))
    assert (
        SELECTIONS.index(entry.draw(np.random.default_rng(11), epsilon=0.0).our_pick)
        == expected
    )

    assert len(per_class[0]) > 1500 and len(per_class[1]) > 1500
    counts = []
    for picks in per_class.values():
        seen = np.bincount(picks, minlength=4)[:4] / len(picks)
        counts.append(seen)
    assert np.abs(counts[0] - counts[1]).max() < 0.05
    for seen in counts:
        assert np.abs(seen - 0.25).max() < 0.05


def test_our_mixture_takes_no_spread_class_argument() -> None:
    """The leak's cheapest re-entry is a convenience argument, so the shape is asserted.

    ``their_mixture`` needs a class index because they know their own investment;
    ``our_mixture`` must not accept one, because we do not. This fails the moment someone
    adds it -- unlike the volatile-usage guard that was deleted for being unfailable.
    """
    ours = inspect.signature(BookEntry.our_mixture).parameters
    theirs = inspect.signature(BookEntry.their_mixture).parameters
    assert "class_index" in theirs
    assert "class_index" not in ours
    assert set(ours) == {"self", "epsilon", "temperature"}


# --------------------------------------------------------------- the exploration mixture


def test_no_exploration_leaves_the_equilibrium_untouched() -> None:
    eq = np.array([0.6, 0.3, 0.1, 0.0])
    loss = np.array([0.0, 0.0, 0.0, 0.4])
    assert np.allclose(explore_mixture(eq, loss, epsilon=0.0), eq)


def test_exploration_prefers_the_selections_that_lose_least() -> None:
    """Shaped by EV loss, because the cells worth measuring are the plausible ones."""
    eq = _point_mass(0)[:4]
    loss = np.array([0.0, 0.002, 0.05, 0.45])
    mixed = explore_mixture(eq, loss, epsilon=0.5, temperature=0.05)
    assert mixed.sum() == pytest.approx(1.0)
    assert mixed[1] > mixed[2] > mixed[3] > 0.0
    # Read the exploration share on its own: mixed = (1-eps)*eq + eps*shaped.
    shaped = (mixed - 0.5 * eq) / 0.5
    # The near-tie is nearly as likely as the "best" -- 0.002 of EV is inside the value
    # function's own error, and generating only the argmax would be false precision.
    assert shaped[1] > 0.9 * shaped[0]
    # A selection that gives up 45 points is generated, but rarely: the next solve needs a
    # cell there and nothing more.
    assert shaped[3] < 1e-3 * shaped[0]


def test_an_infinite_temperature_is_the_old_uniform_draw() -> None:
    eq = _point_mass(0)[:4]
    loss = np.array([0.0, 0.1, 0.2, 0.3])
    mixed = explore_mixture(eq, loss, epsilon=0.25, temperature=float("inf"))
    assert np.allclose(mixed, 0.75 * eq + 0.25 * 0.25)


def test_a_zero_temperature_is_refused_rather_than_read_as_the_equilibrium() -> None:
    with pytest.raises(ValueError, match="temperature"):
        explore_mixture(np.ones(4) / 4, np.zeros(4), temperature=0.0)


def test_a_pure_equilibrium_alone_would_generate_one_selection_pair(roster) -> None:  # noqa: ANN001
    """Why exploration is mandatory rather than a flourish.

    The first solved team's equilibrium was pure: one of 90. Generating from it visits one
    cell of the 90x90 game, and the next generation's solve needs all 8,100 -- so the
    value function would be asked about selections it had never seen. The mixture's
    effective width is the number that says whether the loop can close.
    """
    entry = _entry(
        ours=_point_mass(0),
        theirs=[_point_mass(0)],
        sets=roster.sets,
    )
    assert perplexity(entry.our_strategy) == pytest.approx(1.0)
    # With the default epsilon and every selection tied on EV loss -- the widest the
    # exploration share can be -- the mixture plays 5.3 of the 90 effectively. That is the
    # ceiling, not a typical value: real losses concentrate it further. Asserted so that a
    # change to the defaults which quietly returns to a pure draw fails here.
    assert perplexity(entry.our_mixture()) > 4.0


# ------------------------------------------------------------------------ persistence


def test_the_effective_width_of_a_pooled_distribution_is_the_normalised_one() -> None:
    """It is called on strategies *summed* over the field, which do not sum to 1.

    Entropy on a vector summing to 394 comes out at zero, and zero reads as "the whole
    field plays one selection" -- the exact conclusion the number exists to test for. So
    it normalises, and this is the test that says so.
    """
    one = np.array([0.5, 0.25, 0.25, 0.0])
    assert perplexity(one * 394) == pytest.approx(perplexity(one))
    assert perplexity(np.ones(90)) == pytest.approx(90.0)
    assert perplexity(_point_mass(0)) == pytest.approx(1.0)


def test_the_entry_carries_the_lps_own_numbers(roster) -> None:  # noqa: ANN001
    """``book_entry`` must copy the solve, not re-derive anything."""
    reg = roster.reg

    def score(rows):  # noqa: ANN001, ANN202
        """Antisymmetric by construction: strength is a sum over each side's members.

        The active pair counts double, so leads matter and the 90 selections are not all
        the same cell -- otherwise the LP would return an arbitrary tie and this test
        would check nothing. Read off the encoded rows (`standin.RowStub`): the port scores
        the selection game on the inference server (IKA-463).
        """
        strengths = (rows.maxhp * np.where(rows.active, 2, 1)).sum(axis=-1)
        return 1.0 / (1.0 + np.exp(-(strengths[:, 0] - strengths[:, 1]) / 200.0))

    evaluate = RowStub(reg, score)

    classes = [
        SpreadClass(weight=0.4, sets=tuple(roster.sets), label="a"),
        SpreadClass(weight=0.6, sets=tuple(roster.sets), label="b"),
    ]
    analysis = solve_selection(reg, roster.sets, classes, evaluate)
    entry = book_entry(analysis, key="k", player="P", place=3, model="stub")

    eq = analysis.equilibrium
    assert np.allclose(entry.our_strategy, eq.row_strategy)
    assert np.allclose(entry.our_ev_loss, eq.row_ev_loss)
    assert np.allclose(entry.class_weights, eq.weights)
    assert len(entry.their_strategies) == 2
    assert np.allclose(entry.their_strategies[0], eq.col_strategies[0])
    assert entry.value == pytest.approx(analysis.value)
    assert entry.place == 3
    assert entry.model == "stub"
    # The classes' sets travel with the entry: a draw has to hand generation the very
    # spreads whose column strategy it drew from.
    assert entry.class_sets[0] == classes[0].sets


# ------------------------------------------------------------------ generation uses it


def _standings(team: TournamentTeam) -> Standings:
    return Standings(
        event="test",
        event_format="Reg M-B",
        player_count=1,
        teams=[team],
    )


def _species(roster) -> list[str]:  # noqa: ANN001
    return [entry.species for entry in roster.sets]


def test_bench_weights_name_the_two_a_pure_selection_leaves_behind(roster) -> None:  # noqa: ANN001
    """The whole point: what the unseen slots hold, given what is already out."""
    names = _species(roster)
    first = SELECTIONS[0]
    seen = [names[i] for i in first[:2]]
    got = bench_weights(SELECTIONS, _point_mass(0), names, seen)
    assert got == {tuple(sorted(names[i] for i in first[2:])): 1.0}


def test_bench_weights_drop_a_selection_that_cannot_explain_the_board(roster) -> None:  # noqa: ANN001
    """A Pokemon on the field is one they brought, so a selection without it is out.

    Uniform keeps those worlds and that is the bias: the opponent inside the belief is
    weaker than the one across the table, because their real selections are the ones the
    board has already ruled in.
    """
    names = _species(roster)
    missing = next(name for name in names if name not in {names[i] for i in SELECTIONS[0]})
    assert bench_weights(SELECTIONS, _point_mass(0), names, [missing]) == {}


def test_bench_weights_survive_a_form_whose_base_is_not_on_the_sheet(roster) -> None:  # noqa: ANN001
    """`shown_species` hands over a revealed Pokemon's own id AND its base form.

    The candidate filter needs both, so that a Mega on the field does not leave its base
    available to fill a hidden slot. Conditioning needs exactly one of them -- whichever
    is on the sheet. Requiring both made this explain nothing in 591 of 600 games:
    Floette-Eternal is a sheet member and `floette` is not, so every selection was thrown
    out for missing a Pokemon the opponent never had, and the belief fell back to the
    uniform one the whole change exists to replace.
    """
    names = _species(roster)
    first = SELECTIONS[0]
    seen = [names[first[0]], f"{names[first[0]]}megax", names[first[1]]]
    got = bench_weights(SELECTIONS, _point_mass(0), names, seen)
    assert got == {tuple(sorted(names[i] for i in first[2:])): 1.0}


def test_bench_weights_are_a_distribution_over_the_pairs_that_survive(roster) -> None:  # noqa: ANN001
    names = _species(roster)
    uniform = np.full(len(SELECTIONS), 1.0 / len(SELECTIONS))
    seen = [names[i] for i in SELECTIONS[0][:2]]
    got = bench_weights(SELECTIONS, uniform, names, seen)
    assert got, "some selection must contain the two on the board"
    assert all(len(key) == 2 for key in got)
    assert sum(got.values()) == pytest.approx(1.0)
    # Renormalised over the survivors, not over all ninety: the selections that cannot
    # explain the board are impossible, not merely unlikely.
    assert max(got.values()) > 1.0 / len(SELECTIONS)


def test_bench_weights_ignore_the_order_within_a_pair(roster) -> None:  # noqa: ANN001
    """`completions` keys on a sorted species tuple because two unseen slots are
    interchangeable until one of them is switched in."""
    names = _species(roster)
    got = bench_weights(
        SELECTIONS, np.full(len(SELECTIONS), 1.0 / len(SELECTIONS)), names, []
    )
    assert all(list(key) == sorted(key) for key in got)


def test_entry_bench_weights_read_our_vector_and_their_class_average(roster) -> None:  # noqa: ANN001
    """Ours is one distribution; theirs is one per spread class and we do not know which.

    Averaging theirs by `class_weights` is the same marginalisation the selection solve
    already makes when it prices a cell, and indexing ours by class would hand us their
    private information -- constraint 3 in this module's docstring.
    """
    names = _species(roster)
    entry = _entry(ours=_point_mass(0), theirs=[_point_mass(7)], sets=roster.sets)
    # epsilon 0: the equilibrium unsoftened, so the masses are checkable by hand.
    ours = BenchPrior.of(entry, 0, names, epsilon=0.0).weights(
        [names[i] for i in SELECTIONS[0][:2]]
    )
    theirs = BenchPrior.of(entry, 1, names, epsilon=0.0).weights(
        [names[i] for i in SELECTIONS[7][:2]]
    )
    assert ours == {tuple(sorted(names[i] for i in SELECTIONS[0][2:])): 1.0}
    assert theirs == {tuple(sorted(names[i] for i in SELECTIONS[7][2:])): 1.0}


def test_entry_bench_weights_average_two_classes(roster) -> None:  # noqa: ANN001
    names = _species(roster)
    entry = _entry(
        ours=_point_mass(0), theirs=[_point_mass(0), _point_mass(7)], sets=roster.sets
    )
    got = BenchPrior.of(entry, 1, names, epsilon=0.0).weights([])
    assert sum(got.values()) == pytest.approx(1.0)
    assert len(got) == 2, "two classes putting their mass on different fours"
    assert sorted(got.values()) == pytest.approx([0.5, 0.5])


def test_the_bench_prior_is_softened_like_the_draw_it_models(roster) -> None:  # noqa: ANN001
    """`BookEntry.draw` samples the softened mixture, so the belief must too.

    A belief on the raw equilibrium would be confidently wrong about the opponent this
    very game has -- and confidently is the word, because these supports run to two or
    three selections of ninety, chosen on cells G28 measured at +-11 points.
    """
    names = _species(roster)
    entry = _entry(ours=_point_mass(0), theirs=[_point_mass(0)], sets=roster.sets)
    seen = [names[i] for i in SELECTIONS[0][:2]]
    sharp = BenchPrior.of(entry, 0, names, epsilon=0.0).weights(seen)
    soft = BenchPrior.of(entry, 0, names, epsilon=0.25).weights(seen)
    assert len(sharp) == 1
    assert len(soft) > 1, "exploration puts mass on fours the equilibrium does not"
    assert max(soft.values()) < 1.0
    assert sum(soft.values()) == pytest.approx(1.0)


def test_entry_bench_weights_is_the_same_call(roster) -> None:  # noqa: ANN001
    """The convenience wrapper must not be a second implementation."""
    names = _species(roster)
    entry = _entry(ours=_point_mass(0), theirs=[_point_mass(3)], sets=roster.sets)
    seen = [names[i] for i in SELECTIONS[0][:2]]
    assert entry_bench_weights(entry, 0, names, seen) == BenchPrior.of(
        entry, 0, names
    ).weights(seen)


# ------------------------------------------------------ the lead pair (IKA-118)
#
# A selection is ordered: its first two lead. At turn 1 the seen species are exactly the
# two leads, so "the four contain what was seen" also keeps selections that brought a
# lead but planned it for the back -- and led something that is not on the field.


def _same_four_other_leads(names):  # noqa: ANN001, ANN202
    """Two selections of the same four that differ only in which two lead."""
    first = SELECTIONS[0]
    swapped = (first[0], first[2], first[1], first[3])
    assert swapped in SELECTIONS and sorted(swapped) == sorted(first)
    return first, SELECTIONS.index(swapped)


def test_two_selections_that_differ_only_in_their_leads_are_different_beliefs(
    roster,  # noqa: ANN001
) -> None:
    """Fails before IKA-118: both returned the same belief for the same board."""
    names = _species(roster)
    first, other = _same_four_other_leads(names)
    leads = [names[i] for i in first[:2]]
    led_it = bench_weights(SELECTIONS, _point_mass(0), names, leads, leads=leads)
    brought_it = bench_weights(SELECTIONS, _point_mass(other), names, leads, leads=leads)
    assert led_it == {tuple(sorted(names[i] for i in first[2:])): 1.0}
    # The same four, but it would have led names[first[2]] alongside names[first[0]]:
    # it is not what this side played, and nothing else explains the board.
    assert brought_it == {}
    assert led_it != brought_it


def test_a_selection_that_benched_a_lead_loses_its_back_pair(roster) -> None:  # noqa: ANN001
    """The shape of the measured bias: the mixture's mass on a selection that holds one of
    the leads in the back used to land on that selection's back pair."""
    names = _species(roster)
    first = SELECTIONS[0]
    leads = [names[i] for i in first[:2]]
    elsewhere = next(
        index
        for index, selection in enumerate(SELECTIONS)
        if set(first[:2]) <= set(selection) and set(selection[:2]) != set(first[:2])
        and set(selection) != set(first)
    )
    half = (_point_mass(0) + _point_mass(elsewhere)) / 2
    before = bench_weights(SELECTIONS, half, names, leads)
    after = bench_weights(SELECTIONS, half, names, leads, leads=leads)
    assert len(before) == 2, "the unconditioned belief keeps both back pairs"
    assert after == {tuple(sorted(names[i] for i in first[2:])): 1.0}


def test_leads_are_read_like_seen_so_a_mega_or_forme_id_still_matches(roster) -> None:  # noqa: ANN001
    names = _species(roster)
    first = SELECTIONS[0]
    leads = [names[first[0]], f"{names[first[0]]}megax", names[first[1]]]
    got = BenchPrior.of(
        _entry(ours=_point_mass(0), theirs=[_point_mass(0)], sets=roster.sets),
        0,
        names,
        epsilon=0.0,
    ).weights(leads, leads)
    assert got == {tuple(sorted(names[i] for i in first[2:])): 1.0}


def test_leads_that_are_not_two_sheet_members_are_refused(roster) -> None:  # noqa: ANN001
    """A lead pair the sheet cannot name is a caller bug, not an empty belief."""
    names = _species(roster)
    with pytest.raises(ValueError, match="not the 2 that led"):
        bench_weights(SELECTIONS, _point_mass(0), names, [], leads=[names[0]])
    with pytest.raises(ValueError, match="not the 2 that led"):
        bench_weights(SELECTIONS, _point_mass(0), names, [], leads=["mew", "mewtwo"])

