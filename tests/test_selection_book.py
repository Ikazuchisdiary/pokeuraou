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
import json

import numpy as np
import pytest

from pokeuraou.payoff import HP_SHARE
from pokeuraou.priors import SampledSet
from pokeuraou.selection import SpreadClass, book_entry, solve_selection
from pokeuraou.selection_book import (
    ARMS,
    BookEntry,
    SelectionBook,
    append_entry,
    draw_arm,
    explore_mixture,
    key_for_team,
    perplexity,
    start_book,
)
from pokeuraou.selfplay import generate
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


def test_the_key_is_the_sheet_and_not_the_hidden_investment(roster) -> None:  # noqa: ANN001
    """Two draws of the same team differ only in SP, which the key must ignore.

    The investment is the opponent's private information. A key that moved with it would
    split one team into a new entry per draw -- every game a cache miss, and the solve
    would have been conditioned on a spread we are not allowed to know.
    """
    team = _team(roster.sets)
    assert key_for_team(team) == key_for_team(_team(roster.sets, player="Someone", place=9))

    changed = list(roster.sets)
    changed[0] = SampledSet(
        species=changed[0].species,
        ability=changed[0].ability,
        item="leftovers" if changed[0].item != "leftovers" else "lifeorb",
        nature=changed[0].nature,
        sp=dict(changed[0].sp),
        moves=list(changed[0].moves),
    )
    assert key_for_team(_team(changed)) != key_for_team(team)


def test_an_ability_the_sheet_never_showed_is_keyed_as_unknown(roster) -> None:  # noqa: ANN001
    """Champions sheets show the pre-mega ability, so the recorded one may be private.

    Keying on it would key on something we cannot see, and would also split the team by
    whatever the sampler happened to draw.
    """
    base = _team(roster.sets)
    hidden = TournamentTeam(
        player=base.player,
        place=base.place,
        country=base.country,
        wins=base.wins,
        losses=base.losses,
        made_cut=base.made_cut,
        phase_two=base.phase_two,
        members=(
            TeamMember(
                species=base.members[0].species,
                ability="somethingelse",
                item=base.members[0].item,
                nature=base.members[0].nature,
                moves=base.members[0].moves,
                ability_is_post_mega=True,
            ),
            *base.members[1:],
        ),
    )
    other = TournamentTeam(
        player=base.player,
        place=base.place,
        country=base.country,
        wins=base.wins,
        losses=base.losses,
        made_cut=base.made_cut,
        phase_two=base.phase_two,
        members=(
            TeamMember(
                species=base.members[0].species,
                ability="yetanother",
                item=base.members[0].item,
                nature=base.members[0].nature,
                moves=base.members[0].moves,
                ability_is_post_mega=True,
            ),
            *base.members[1:],
        ),
    )
    assert key_for_team(hidden) == key_for_team(other)
    assert key_for_team(hidden) != key_for_team(base)


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


def test_no_arm_lets_our_draw_see_the_opponents_spread_class(roster) -> None:  # noqa: ANN001
    """The head-to-head harness mixes rules, which is where the leak would come back.

    Two sides drawn by different code paths invite passing the class into both. So for
    every arm, with the stream fixed, our index must be identical whichever class the
    opponent turns out to hold -- while theirs moves, because it is allowed to.
    """
    ours = np.zeros(len(SELECTIONS))
    ours[:6] = 1.0 / 6.0
    entry = _entry(
        ours=ours,
        theirs=[_point_mass(0), _point_mass(len(SELECTIONS) - 1)],
        sets=roster.sets,
    )
    for arm in ARMS:
        picks = [
            draw_arm(arm, entry, k, np.random.default_rng(4), epsilon=0.3)
            for k in (0, 1)
        ]
        assert picks[0][0] == picks[1][0], f"{arm} moved our draw with their class"
    # Stronger, because the assertion above would also pass if the two draws were
    # *reordered* -- each consumes exactly one uniform, so a swap is invisible to it.
    # Our index must be the one the stream's FIRST uniform names against our own
    # distribution, which pins the order as well as the independence.
    for arm, mine in (
        ("book/uniform", entry.our_strategy),
        ("book/book", entry.our_strategy),
        ("gen/gen", entry.our_mixture(epsilon=0.3)),
    ):
        first = np.random.default_rng(4).random()
        expected = int(
            np.searchsorted(np.cumsum(mine / mine.sum()), first, side="right")
        )
        got, _ = draw_arm(arm, entry, 1, np.random.default_rng(4), epsilon=0.3)
        assert got == expected, f"{arm} did not draw ours from the first uniform"

    # And the control: the arms that read their strategy do see the difference, so the
    # test above is not passing because nothing is wired up.
    for arm in ("book/book", "gen/gen"):
        picks = [
            draw_arm(arm, entry, k, np.random.default_rng(4), epsilon=0.0)
            for k in (0, 1)
        ]
        assert picks[0][1] != picks[1][1], f"{arm} ignored their class entirely"


def test_an_unknown_arm_is_refused() -> None:
    entry = _entry(ours=_point_mass(0), theirs=[_point_mass(0)], sets=[])
    with pytest.raises(ValueError, match="unknown arm"):
        draw_arm("book/best-response", entry, 0, np.random.default_rng(0))


def test_the_uniform_arm_really_is_uniform(roster) -> None:  # noqa: ANN001
    """The baseline has to be the old behaviour, or the comparison measures nothing."""
    entry = _entry(
        ours=_point_mass(0), theirs=[_point_mass(0)], sets=roster.sets
    )
    rng = np.random.default_rng(2)
    seen = np.zeros(len(SELECTIONS))
    for _ in range(9000):
        ours, theirs = draw_arm("uniform/uniform", entry, 0, rng)
        seen[ours] += 1
        seen[theirs] += 1
    share = seen / seen.sum()
    assert np.abs(share - 1.0 / len(SELECTIONS)).max() < 0.004


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


def test_a_book_round_trips_through_disk(roster, tmp_path) -> None:  # noqa: ANN001
    entry = _entry(
        ours=np.full(len(SELECTIONS), 1.0 / len(SELECTIONS)),
        theirs=[_point_mass(3), _point_mass(4)],
        sets=roster.sets,
    )
    book = SelectionBook(roster="rizabanadohido", model="value-gen2.pt", format_id="x")
    book.add(entry)
    path = tmp_path / "book.jsonl.gz"
    book.write(path)

    back = SelectionBook.read(path)
    assert back.roster == "rizabanadohido"
    assert back.model == "value-gen2.pt"
    assert len(back) == 1
    got = back.entries[entry.key]
    assert got.selections == entry.selections
    assert np.allclose(got.our_strategy, entry.our_strategy)
    assert len(got.their_strategies) == 2
    assert np.allclose(got.their_strategies[1], entry.their_strategies[1])
    assert [s.species for s in got.class_sets[0]] == [s.species for s in roster.sets]
    assert got.class_sets[0][0].sp == roster.sets[0].sp


def test_appending_one_entry_at_a_time_leaves_a_readable_book(roster, tmp_path) -> None:  # noqa: ANN001
    """An 80-minute solve must be readable after every entry, not only at the end."""
    path = tmp_path / "book.jsonl.gz"
    start_book(path, roster="rizabanadohido", model="m", format_id="f")
    for i in range(3):
        entry = _entry(
            ours=_point_mass(i), theirs=[_point_mass(i)], sets=roster.sets
        )
        entry.key = f"key-{i}"
        append_entry(path, entry)
        partial = SelectionBook.read(path)
        assert len(partial) == i + 1
        assert partial.roster == "rizabanadohido"


def test_a_book_refuses_the_roster_it_was_not_solved_for() -> None:
    book = SelectionBook(roster="rizabanadohido")
    book.require_roster("rizabanadohido")
    with pytest.raises(ValueError, match="party order"):
        book.require_roster("something-else")


def test_the_entry_carries_the_lps_own_numbers(roster) -> None:  # noqa: ANN001
    """``book_entry`` must copy the solve, not re-derive anything."""
    reg = roster.reg

    def evaluate(positions):  # noqa: ANN001, ANN202
        """Antisymmetric by construction: strength is a sum over each side's members.

        The active pair counts double, so leads matter and the 90 selections are not all
        the same cell -- otherwise the LP would return an arbitrary tie and this test
        would check nothing.
        """
        out = np.empty(len(positions), dtype=np.float64)
        for i, pos in enumerate(positions):
            strengths = []
            for side in pos.sides:
                strengths.append(
                    sum(
                        m.maxhp * (2 if m.active_index is not None else 1)
                        for m in side.pokemon
                    )
                )
            out[i] = 1.0 / (1.0 + np.exp(-(strengths[0] - strengths[1]) / 200.0))
        return out

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


def test_generation_draws_both_sides_from_the_book(roster, tmp_path) -> None:  # noqa: ANN001
    """End to end: every game is a book hit, and the record says what it was drawn from.

    The opponent's sets in the record must be the drawn class's, spreads included. If
    generation resampled the investment after drawing their selection, their four would
    have been chosen by a player holding a different team.
    """
    reg = roster.reg
    team = _team(roster.sets)
    spreads = [
        SampledSet(
            species=s.species,
            ability=s.ability,
            item=s.item,
            nature=s.nature,
            sp={"hp": 8 + k, "atk": 4, "spe": 12},
            moves=list(s.moves),
        )
        for k, s in enumerate(roster.sets)
    ]
    entry = _entry(
        ours=_point_mass(0), theirs=[_point_mass(5)], sets=spreads
    )
    entry.key = key_for_team(team)
    book = SelectionBook(roster="rizabanadohido")
    book.add(entry)

    out = tmp_path / "games.jsonl"
    stats = generate(
        reg,
        None,
        roster,
        [],
        games=2,
        seed=3,
        out=out,
        objective=HP_SHARE,
        search_limit=2,
        max_turns=3,
        standings=_standings(team),
        book=book,
        explore_epsilon=0.0,
    )
    assert stats["book_hits"] == 2
    assert stats["book_misses"] == 0

    # Asserted rather than skipped-if-empty: the rng is seeded and the resolver is
    # deterministic here, so "nothing finished" would be a real change and not weather.
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    for line in lines:
        record = json.loads(line)
        assert record["selectionSource"] == "book"
        assert record["ownPick"] == list(SELECTIONS[0])
        assert record["foePick"] == list(SELECTIONS[5])
        assert record["selectionValue"] == pytest.approx(0.5)
        assert len(record["ownSelectionPolicy"]) == len(SELECTIONS)
        assert sum(record["foeSelectionPolicy"]) == pytest.approx(1.0)
        brought = [m["sp"] for m in record["foeTeam"]]
        assert all("hp" in sp for sp in brought)


def test_a_mirror_game_is_our_own_six_with_our_own_spreads(roster, tmp_path) -> None:  # noqa: ANN001
    """Not "a team like ours" -- the same sets, or the 50% assertion is worth nothing.

    A mirror whose opponent had resampled spreads is not antisymmetric, so its win rate
    would no longer be forced to 50% and the free calibration check would quietly become
    a number with no requirement attached to it.
    """
    reg = roster.reg
    out = tmp_path / "games.jsonl"
    stats = generate(
        reg,
        None,
        roster,
        [],
        games=2,
        seed=5,
        out=out,
        objective=HP_SHARE,
        search_limit=2,
        # Mirrors of this composition are stally -- two Toxapex and two Incineroar on the
        # field -- so a three-turn cap finishes nothing and the assertions below would all
        # be skipped over an empty file.
        max_turns=25,
        mirror_share=1.0,
    )
    assert stats["mirror_games"] == 2
    assert stats["book_hits"] == 0
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    for line in lines:
        record = json.loads(line)
        assert record["foeArchetype"] == "mirror"
        assert record["foeSix"] == [entry.species for entry in roster.sets]
        # Both sides brought four of the *same* six sets. They are different fours -- the
        # picks are independent -- so the teams are not equal; what must hold is that every
        # member of either is one of our roster's sets, spreads included.
        exact = {
            (entry.species, tuple(sorted((k, v) for k, v in entry.sp.items() if v)))
            for entry in roster.sets
        }
        for side in ("ownTeam", "foeTeam"):
            members = {
                (m["species"], tuple(sorted(m["sp"].items()))) for m in record[side]
            }
            assert members <= exact, f"{side} is not our own six"


def test_a_mirror_share_must_be_a_probability(roster) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="mirror_share"):
        generate(roster.reg, None, roster, [], games=1, mirror_share=1.5)


def test_a_team_the_book_does_not_cover_falls_back_to_a_uniform_draw(roster) -> None:  # noqa: ANN001
    """Counted, not silently mixed in: two distributions in one dataset need labels."""
    reg = roster.reg
    team = _team(roster.sets)
    book = SelectionBook(roster="rizabanadohido")
    hit, total = book.covered([team])
    assert (hit, total) == (0, 1)
    with pytest.raises(ValueError, match="standings"):
        generate(reg, None, roster, [], games=1, book=book)


def test_the_merge_reads_the_directory_the_shards_wrote_to(tmp_path) -> None:  # noqa: ANN001
    """A book solved outside the default directory has to be mergeable.

    The two halves disagreed: ``part_path`` wrote beside ``--out`` while ``--merge``
    globbed the default ``data/selection``. A sharded solve into any other directory --
    the seed comparison writes into ``data/selection/seedcheck`` -- therefore solved all
    eight shards and then refused, saying it found no part files "next to" an ``out`` it
    had never looked next to.
    """
    from tests._harness import load_tool

    tool = load_tool("solve_selection_book")
    out = tmp_path / "elsewhere" / "rizabanadohido-value-gen8-s2.jsonl.gz"
    out.parent.mkdir(parents=True)

    written = [tool.part_path(out, shard) for shard in range(3)]
    for path in written:
        path.write_bytes(b"")
    # A decoy at the default location, and the merged book itself, are not parts of it.
    out.write_bytes(b"")
    (out.parent / "rizabanadohido-value-gen8-s2-other.part0.jsonl.gz").write_bytes(b"")

    assert [p.parent for p in written] == [out.parent] * 3
    assert tool.part_files(out) == written
