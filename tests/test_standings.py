"""Real tournament teams, which are now what the value function is trained against.

The opponent pool decides what the tool is calibrated for, so it sits with the other
load-bearing pieces. Three things have to hold, and each has a specific failure it prevents:

- **teams are real and legal.** Every member is team-legal with legal moves and a legal
  item, Species Clause and Item Clause hold, and a team with one unparseable member is
  rejected whole rather than silently becoming a five-Pokemon team nobody brought.
- **only the hidden field is invented.** Species, ability, item, nature and all four moves
  come from the sheet; the SP spread comes from usage *conditioned on the stated nature*,
  because that is exactly the information a Champions open team sheet gives.
- **nothing is dropped quietly.** The one rejected entry and the two Pokemon whose recorded
  ability was the mega forme's are reported by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import load_regulation, to_id
from pokeuraou.standings import (
    StandingsError,
    cluster_core,
    cluster_teams,
    find_cached_standings,
    load_standings,
    sample_standings_team,
    species_frequency,
)

REGULATION = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def bundle():  # noqa: ANN201
    reg = load_regulation(REGULATION)
    cached = find_cached_standings("2026", "worlds")
    if cached is None:
        pytest.skip("no cached standings; run tools/fetch_standings.py 2026 worlds")
    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        pytest.skip("no cached usage stats")
    return reg, load_standings(cached, reg), load_chaos(chaos, reg)


def test_the_field_loads_and_is_the_right_format(bundle) -> None:  # noqa: ANN001
    reg, standings, _prior = bundle
    assert "M-B" in standings.event_format
    assert standings.player_count > 300
    # 394 of 395 entries validate; the one rejection is an entry with no team submitted.
    assert len(standings.teams) >= 380
    assert len(standings.rejected) <= 12
    assert all(len(t.members) == reg.meta.team_size for t in standings.teams)


def test_every_member_is_legal(bundle) -> None:  # noqa: ANN001
    reg, standings, _prior = bundle
    for team in standings.teams:
        for member in team.members:
            species = reg.species[member.species]
            assert species.team_legal, (team.player, member.species)
            assert member.nature in reg.natures
            assert len(member.moves) == 4
            assert all(m in reg.moves for m in member.moves)
            if member.item is not None:
                assert member.item in reg.items
            if member.ability is not None:
                assert member.ability in {to_id(a) for a in species.abilities}


def test_the_clauses_hold_on_every_kept_team(bundle) -> None:  # noqa: ANN001
    reg, standings, _prior = bundle
    for team in standings.teams:
        bases = [reg.species[m.species].base_species for m in team.members]
        assert len(set(bases)) == len(bases), team.player
        if reg.meta.item_clause is not None:
            items = [m.item for m in team.members if m.item]
            assert len(set(items)) == len(items), team.player


def test_a_post_mega_ability_becomes_unknown_and_is_reported(bundle) -> None:  # noqa: ANN001
    """Drought is Mega Charizard Y's ability, so it cannot be what the sheet showed.

    The sheet shows the pre-mega ability, which this file does not carry. Keeping "Drought"
    would put an illegal ability on the base forme; guessing Blaze silently would invent
    information. So it becomes unknown, is filled from usage, and is named in the summary.
    """
    _reg, standings, _prior = bundle
    flagged = [m for t in standings.teams for m in t.members if m.ability_is_post_mega]
    assert all(m.ability is None for m in flagged)
    if flagged:
        assert standings.post_mega_abilities
        assert "Charizard" in " ".join(standings.post_mega_abilities)


def test_sampling_keeps_the_sheet_and_only_fills_the_spread(bundle) -> None:  # noqa: ANN001
    reg, standings, prior = bundle
    rng = np.random.default_rng(0)
    for team in standings.pool("cut"):
        sets = sample_standings_team(rng, reg, prior, team)
        assert len(sets) == len(team.members)
        for got, member in zip(sets, team.members, strict=True):
            assert got.species == member.species
            assert got.item == member.item
            assert got.nature == member.nature
            assert list(got.moves) == list(member.moves)
            if member.ability is not None:
                assert got.ability == member.ability
            else:
                # Unknown ability: whatever is chosen has to be legal on the base forme.
                assert got.ability in {to_id(a) for a in reg.species[got.species].abilities}
            assert sum(got.sp.values()) <= reg.meta.sp_limit
            assert all(0 <= v <= reg.meta.sp_per_stat_max for v in got.sp.values())


def test_the_spread_is_conditioned_on_the_stated_nature(bundle) -> None:  # noqa: ANN001
    """The nature is public, so the spread has to be drawn from particles matching it.

    Drawing from the unconditioned distribution would produce a spread nobody played with
    that nature -- a Modest Pokemon with the Adamant crowd's investment -- and would throw
    away the one piece of information the team sheet actually reveals.
    """
    reg, standings, prior = bundle
    rng = np.random.default_rng(1)
    checked = 0
    for team in standings.pool("phase2")[:40]:
        for member in team.members:
            entry = prior.species.get(member.species)
            if entry is None:
                continue
            matching = np.asarray(entry.natures) == member.nature
            if not matching.any():
                continue  # a nature nobody in the usage sample played; reported, not faked
            observed = {
                tuple(int(v) for v in entry.spreads[i])
                for i in np.flatnonzero(matching)
            }
            for _ in range(3):
                sets = sample_standings_team(rng, reg, prior, team)
                got = next(s for s in sets if s.species == member.species)
                spread = tuple(
                    int(got.sp.get(stat, 0))
                    for stat in ("hp", "atk", "def", "spa", "spd", "spe")
                )
                assert spread in observed, (member.species, member.nature, spread)
                checked += 1
    assert checked > 100


def test_the_pools_nest(bundle) -> None:  # noqa: ANN001
    _reg, standings, _prior = bundle
    everyone = standings.pool("all")
    phase2 = standings.pool("phase2")
    cut = standings.pool("cut")
    assert len(cut) <= len(phase2) <= len(everyone)
    assert {t.player for t in cut} <= {t.player for t in phase2}
    assert {t.player for t in phase2} <= {t.player for t in everyone}
    with pytest.raises(StandingsError):
        standings.pool("winners")


def test_the_field_is_concentrated_into_a_few_archetypes(bundle) -> None:  # noqa: ANN001
    """The claim that motivated using real teams: the field is a mixture of a few decks.

    A pairwise co-occurrence model cannot represent a mixture, and a curated list of eleven
    archetypes cannot cover one this concentrated. Both were replaced because of this
    shape, so it is worth asserting that the shape is really there.
    """
    _reg, standings, _prior = bundle
    teams = standings.pool("all")
    groups = cluster_teams(teams, min_shared=5)
    biggest = len(groups[0]) / len(teams)
    top_three = sum(len(g) for g in groups[:3]) / len(teams)
    assert biggest > 0.10, f"largest cluster only {biggest * 100:.1f}% of the field"
    assert top_three > 0.35, f"top three clusters only {top_three * 100:.1f}%"
    core = cluster_core(groups[0])
    assert core and core[0][1] > 0.5


def test_the_field_disagrees_with_ladder_usage(bundle) -> None:  # noqa: ANN001
    """Tournament and ladder are different distributions, which is why the pool changed.

    Measured: Charizard 45.2% at Worlds against 27.0% on the ladder. If these two ever
    agreed closely, the ladder-derived pool would have been fine and this module would be
    unnecessary -- so the disagreement is the thing to keep an eye on.
    """
    _reg, standings, prior = bundle
    field = species_frequency(standings.pool("all"))
    ladder = {s.species_id: s.usage for s in prior.species.values()}
    shared = [s for s in field if s in ladder]
    assert len(shared) > 50
    gaps = {s: field[s] - ladder[s] for s in shared}
    assert max(gaps.values()) > 0.10, "no species differs by more than 10 points"
    # And something the ladder plays is absent from the field entirely.
    assert any(field.get(s, 0.0) == 0.0 and u > 0.005 for s, u in ladder.items())
