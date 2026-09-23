"""The teams self-play will be played with, and where every number in them came from.

The load-bearing test is :func:`test_our_stats_match_the_game_itself`. Every other stat
check in this project compares against Showdown, which is a reimplementation; the recorded
team carries the final stats the game's own status screen displayed, so this is the one
place the SP formula is checked against the real thing.

The rest are refusals. A team file is where a wrong premise gets in, and a set that is
silently repaired -- an illegal item accepted, an unstated spread invented -- becomes a
training example and then a printed win probability with nothing to trace it to.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import Regulation
from pokeuraou.teams import (
    TeamError,
    all_picks,
    all_selections,
    load_archetypes,
    load_roster,
    pick_four,
    sample_archetype,
    teams_dir,
    usable_archetypes,
    verify_shown_stats,
)

ROSTER = "rizabanadohido"
ARCHETYPES = "wcs2026-regmb"


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    if not (teams_dir() / f"{ROSTER}.json").exists():
        pytest.skip(f"{ROSTER}.json missing")
    return load_roster(ROSTER)


@pytest.fixture(scope="module")
def archetypes():  # noqa: ANN201
    return load_archetypes(ARCHETYPES)


@pytest.fixture(scope="module")
def prior(archetypes):  # noqa: ANN001, ANN201
    reg, _ = archetypes
    path = find_cached_chaos(reg.meta.format_id) or find_cached_chaos(
        "gen9championsvgc2026regmc"
    )
    if path is None:
        pytest.skip("no cached usage stats")
    return load_chaos(path, reg)


def test_our_stats_match_the_game_itself(roster) -> None:  # noqa: ANN001
    """The SP formula against the game's own status screen, not against Showdown.

    Both should agree, and they do -- but only one of them is the actual game.
    """
    mismatches = verify_shown_stats(roster)
    assert mismatches == [], "\n".join(
        f"{species}: ours {ours} vs the game's screen {shown}"
        for species, ours, shown in mismatches
    )
    recorded = sum(1 for shown in roster.shown_stats if shown is not None)
    assert recorded == len(roster.sets), "every recorded Pokemon should carry its stats"


def test_the_roster_is_legal(roster) -> None:  # noqa: ANN001
    reg: Regulation = roster.reg
    assert len(roster.sets) == reg.meta.team_size
    items = [s.item for s in roster.sets if s.item]
    assert len(set(items)) == len(items), "Item Clause"
    species = [reg.species[s.species].base_species for s in roster.sets]
    assert len(set(species)) == len(species), "Species Clause"
    for entry in roster.sets:
        assert sum(entry.sp.values()) <= reg.meta.sp_limit
        assert max(entry.sp.values()) <= reg.meta.sp_per_stat_max
        assert len(entry.moves) == 4


def test_a_recorded_team_must_state_its_abilities(tmp_path) -> None:  # noqa: ANN001
    """A roster is "recorded exactly"; a null field there would be a guess in disguise."""
    data = json.loads((teams_dir() / f"{ROSTER}.json").read_text(encoding="utf-8"))
    data["team"][0]["ability"] = None
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TeamError, match="state the ability"):
        load_roster(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nature", None),
        ("nature", ""),
        ("nature", "<absent>"),
        ("ability", "<absent>"),
        ("item", "<absent>"),
        ("moves", None),
        ("moves", []),
        ("moves", "<absent>"),
        ("sp", None),
        ("sp", "<absent>"),
        ("species", "<absent>"),
    ],
)
def test_a_missing_field_stops_the_load_and_names_itself(tmp_path, field, value) -> None:  # noqa: ANN001
    """IKA-137: a member with no nature used to load as ``nature='None'`` and fail only
    later, in ``nature_multipliers`` (``KeyError: Unknown nature: 'None'``), which points at
    the stats and not at the file. The entry is where the missing field is known."""
    data = json.loads((teams_dir() / f"{ROSTER}.json").read_text(encoding="utf-8"))
    data["id"] = "broken-team"
    member = data["team"][2]
    if value == "<absent>":
        del member[field]
    else:
        member[field] = value
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TeamError, match=rf"broken-team\[2\].*'{field}'"):
        load_roster(path)


def test_no_item_is_a_statement_not_a_missing_field(tmp_path) -> None:  # noqa: ANN001
    """``"item": null`` says the Pokemon holds nothing, which a sheet can show; only an
    absent key is unrecorded."""
    data = json.loads((teams_dir() / f"{ROSTER}.json").read_text(encoding="utf-8"))
    data["team"][2]["item"] = None
    path = tmp_path / "noitem.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_roster(path).sets[2].item is None


def test_an_illegal_item_is_refused(tmp_path) -> None:  # noqa: ANN001
    data = json.loads((teams_dir() / f"{ROSTER}.json").read_text(encoding="utf-8"))
    data["team"][0]["item"] = "Choice Specs Of Nowhere"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TeamError, match="not legal"):
        load_roster(path)


def test_an_over_budget_spread_is_refused(tmp_path) -> None:  # noqa: ANN001
    data = json.loads((teams_dir() / f"{ROSTER}.json").read_text(encoding="utf-8"))
    data["team"][0]["sp"] = {"hp": 32, "atk": 32, "def": 32}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TeamError, match="exceeds"):
        load_roster(path)


def test_every_archetype_is_legal(archetypes) -> None:  # noqa: ANN001
    reg, entries = archetypes
    assert entries
    for archetype in entries:
        assert len(archetype.species) == reg.meta.team_size
        assert len(set(archetype.species)) == len(archetype.species)
        assert archetype.source, f"{archetype.id} has no source"
        for species, item in archetype.mega_stones.items():
            assert reg.mega_target(species, item) is not None


def test_sampling_an_archetype_keeps_the_composition_and_the_megas(
    archetypes, prior  # noqa: ANN001
) -> None:
    """The cited part stays cited: the six species, and the stones the name commits to."""
    reg, entries = archetypes
    usable, _blocked = usable_archetypes(prior, entries)
    assert usable
    rng = np.random.default_rng(3)
    for archetype in usable:
        six = sample_archetype(rng, reg, prior, archetype)
        assert [s.species for s in six] == list(archetype.species)
        for species, stone in archetype.mega_stones.items():
            held = next(s.item for s in six if s.species == species)
            assert held == stone, (
                f"{archetype.id}: {species} must hold {stone}; the archetype's own name "
                "says it megas, so drawing an item from usage would erase that"
            )
        items = [s.item for s in six if s.item]
        assert len(set(items)) == len(items), f"{archetype.id}: Item Clause"
        for entry in six:
            assert sum(entry.sp.values()) <= reg.meta.sp_limit
            assert entry.ability, "an ability must end up set, stated or sampled"


def test_an_archetype_with_no_usage_data_is_reported_not_invented(
    archetypes, prior  # noqa: ANN001
) -> None:
    """A composition whose member never appears in the data cannot be built.

    Filling it in from nothing would put an untraceable set into training, so the loader
    refuses and the caller is told which species is missing.
    """
    reg, entries = archetypes
    usable, blocked = usable_archetypes(prior, entries)
    assert len(usable) + len(blocked) == len(entries)
    rng = np.random.default_rng(0)
    for archetype, missing in blocked:
        assert missing
        with pytest.raises(TeamError, match="no usage data"):
            sample_archetype(rng, reg, prior, archetype)


def test_picking_four_is_uniform_over_all_fifteen(roster) -> None:  # noqa: ANN001
    """Not the real selection distribution, but it has to cover every combination."""
    assert len(all_picks(roster.sets)) == 15
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(600):
        four = pick_four(rng, roster.sets)
        assert len(four) == 4
        assert len({s.species for s in four}) == 4
        seen.add(tuple(sorted(s.species for s in four)))
    assert len(seen) == 15, f"only {len(seen)} of 15 selections were ever drawn"


def test_every_member_can_lead(roster) -> None:  # noqa: ANN001
    """Selection is ordered, and the team file must not decide who leads.

    This was wrong: `pick_four` returned the four in party order and the first two go on
    the field, so the leads came from the file's ordering. Over 4,539 generated games only
    6 of the 15 lead pairs appeared, one of them 39% of the time, and the two members
    listed last could never lead at all -- Fake Out never opened a game.

    It matters because the turn-1 position *is* the selection, and turn 1 is where the
    proxy is weakest and the value function has the most to gain. The dataset was
    systematically wrong in the one region it existed to improve, so this is asserted
    rather than assumed.
    """
    reg = roster.reg
    rng = np.random.default_rng(0)
    lead_pairs: set[tuple[str, ...]] = set()
    leaders: set[str] = set()
    counts: dict[tuple[str, ...], int] = {}
    draws = 4000
    for _ in range(draws):
        four = pick_four(rng, roster.sets, size=reg.meta.picked_team_size)
        assert len(four) == reg.meta.picked_team_size
        assert len({m.species for m in four}) == len(four)
        pair = tuple(sorted(m.species for m in four[:2]))
        lead_pairs.add(pair)
        leaders.update(pair)
        counts[pair] = counts.get(pair, 0) + 1

    assert len(leaders) == len(roster.sets), f"only {sorted(leaders)} ever lead"
    assert len(lead_pairs) == 15, f"{len(lead_pairs)} of 15 lead pairs"
    # Roughly uniform: no pair may be more than twice as likely as another, which a draw
    # biased by party order would fail immediately.
    assert max(counts.values()) < 2 * min(counts.values())


def test_the_ordered_selection_space_is_ninety(roster) -> None:  # noqa: ANN001
    """C(6,2) lead pairs x C(4,2) bench pairs, which is what the selection solver enumerates."""
    reg = roster.reg
    selections = all_selections(reg.meta.team_size, reg.meta.picked_team_size)
    assert len(selections) == 90
    assert len(set(selections)) == 90
    assert len({s[:2] for s in selections}) == 15
    assert len({tuple(sorted(s)) for s in selections}) == 15
    for selection in selections:
        assert len(set(selection)) == reg.meta.picked_team_size
        # Leads and bench are each recorded as an ordered pair of ascending indices, so
        # the same selection has exactly one representation.
        assert list(selection[:2]) == sorted(selection[:2])
        assert list(selection[2:]) == sorted(selection[2:])
