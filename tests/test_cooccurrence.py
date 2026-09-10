"""The measured pairwise team structure, and the reading of the table it rests on.

The opponent pool decides what the value function is calibrated for, so it belongs with
the other load-bearing pieces. It also rests on an *interpretation* of an undocumented
field: that `Teammates` holds weighted counts of teams containing both Pokemon. Two
consequences of that reading are checkable, and if a future stats file stops satisfying
them the interpretation is wrong and every sampled opponent is wrong with it:

- ``sum(Teammates) / 5 / usage`` is the same for every species, because each team has
  exactly five other members;
- the matrix is symmetric where it carries mass, because counts are.

The rest of the tests are about the sampler obeying the rules of the format, and about the
one thing that must never happen quietly: a sampled Pokemon with no way to deal damage.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.priors import (
    build_cooccurrence,
    find_cached_chaos,
    load_chaos,
    sample_species_by_cooccurrence,
)
from pokeuraou.regulation import load_regulation
from pokeuraou.teams import sample_metagame_team


@pytest.fixture(scope="module")
def bundle():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(cached, reg)
    return reg, prior, build_cooccurrence(prior)


def test_the_teammate_table_really_holds_counts(bundle) -> None:  # noqa: ANN001
    """The evidence for reading `Teammates` as co-occurrence counts.

    If the values were conditional shares instead, this ratio would not be constant and
    the lift matrix would be wrong by a per-species factor.
    """
    _reg, prior, _cooc = bundle
    assert prior.weighted_teams > 0
    # Measured at 6.66e-04 for the M-B 1760 file; anything of order 1 means the reading is
    # wrong rather than merely noisy.
    assert prior.weighted_teams_spread < 1e-2


def test_the_matrix_is_symmetric_where_it_has_mass(bundle) -> None:  # noqa: ANN001
    _reg, _prior, cooc = bundle
    # The unweighted maximum is dominated by species at 0.0% usage, where one shared team
    # is the entire count and the two directions round differently, so the check is on the
    # mass-weighted figure and on the heaviest pairs.
    assert cooc.asymmetry < 1e-3
    assert cooc.asymmetry_heavy < 0.2


def test_lift_is_a_ratio_against_chance(bundle) -> None:  # noqa: ANN001
    _reg, _prior, cooc = bundle
    n = len(cooc.species)
    assert cooc.lift.shape == (n, n)
    assert np.all(cooc.lift >= 0.0)
    # Species Clause: a Pokemon is never its own teammate.
    assert np.all(np.diag(cooc.lift) == 0.0)
    # Pair frequency = lift * usage * usage, and a frequency cannot exceed either usage.
    freq = cooc.lift * np.outer(cooc.usage, cooc.usage)
    bound = np.minimum.outer(cooc.usage, cooc.usage)
    assert np.all(freq <= bound + 1e-9)


def test_sampled_species_obey_species_clause(bundle) -> None:  # noqa: ANN001
    reg, _prior, cooc = bundle
    rng = np.random.default_rng(1)
    for _ in range(80):
        team = sample_species_by_cooccurrence(rng, reg, cooc)
        assert len(team) == reg.meta.team_size
        bases = [reg.species[s].base_species for s in team]
        assert len(set(bases)) == len(bases)
        assert all(reg.species[s].team_legal for s in team)


def test_sampling_reproduces_the_measured_usage(bundle) -> None:  # noqa: ANN001
    """The sampler is a pairwise approximation, so its error has to be bounded.

    Measured at 4.5% total variation over 5,000 teams by ``tools/cooc_report.py``. The
    bound here is loose enough for 600 teams' sampling noise and tight enough that a
    sampler which collapsed onto the most-used species would fail it.
    """
    reg, _prior, cooc = bundle
    rng = np.random.default_rng(2)
    counts = np.zeros(len(cooc.species))
    index = cooc.index
    teams = 600
    for _ in range(teams):
        for species_id in sample_species_by_cooccurrence(rng, reg, cooc):
            counts[index[species_id]] += 1
    sampled = counts / teams
    tv = 0.5 * np.abs(sampled / sampled.sum() - cooc.usage / cooc.usage.sum()).sum()
    assert tv < 0.15, f"total variation {tv:.3f} against the measured metagame"
    # The most-used species has to come out roughly as often as it is actually played.
    top = int(np.argmax(cooc.usage))
    assert 0.75 < sampled[top] / cooc.usage[top] < 1.25


def test_the_pool_covers_the_species_the_archetypes_missed(bundle) -> None:  # noqa: ANN001
    """The reason this pool exists at all.

    Eleven cited archetypes use 28 species between them and contain no Sneasler (31.4%
    usage) and no Sinistcha (26.1%). A pool that still misses them would not have fixed
    anything.
    """
    reg, _prior, cooc = bundle
    rng = np.random.default_rng(4)
    seen: set[str] = set()
    for _ in range(400):
        seen.update(sample_species_by_cooccurrence(rng, reg, cooc))
    for species_id in ("sneasler", "sinistcha", "kingambit"):
        if species_id in cooc.index:
            assert species_id in seen, species_id
    assert len(seen) > 60


def test_every_sampled_pokemon_can_deal_damage(bundle) -> None:  # noqa: ANN001
    """Marginal move sampling used to hand out sets that cannot win.

    2.3% of sampled Pokemon came out with four status moves -- Whimsicott with
    Charm/Encore/Protect/Tailwind was the most common -- and the opponent lost 12 points of
    win rate in the games where it happened, so the label recorded a loss that was really
    a broken set.
    """
    reg, prior, cooc = bundle
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(120):
        for member in sample_metagame_team(rng, reg, prior, cooc):
            checked += 1
            attacks = [
                m
                for m in member.moves
                if reg.moves[m].raw.get("category") != "Status"
                and (
                    reg.moves[m].raw.get("basePower")
                    or reg.moves[m].raw.get("damageCallback")
                    or reg.moves[m].raw.get("damage")
                )
            ]
            assert attacks, f"{member.species} has no damaging move: {member.moves}"
    assert checked == 120 * reg.meta.team_size


def test_sampled_teams_obey_the_item_clause(bundle) -> None:  # noqa: ANN001
    reg, prior, cooc = bundle
    if reg.meta.item_clause is None:
        pytest.skip("this regulation has no item clause")
    rng = np.random.default_rng(6)
    for _ in range(60):
        team = sample_metagame_team(rng, reg, prior, cooc)
        items = [m.item for m in team if m.item]
        assert len(set(items)) == len(items)
        assert all(i in reg.items for i in items)


def test_a_mega_stone_only_lands_on_its_own_species(bundle) -> None:  # noqa: ANN001
    reg, prior, cooc = bundle
    rng = np.random.default_rng(7)
    for _ in range(80):
        for member in sample_metagame_team(rng, reg, prior, cooc):
            if member.item and reg.items[member.item].mega_stone:
                assert reg.mega_target(member.species, member.item) is not None, (
                    f"{member.species} holding {member.item}"
                )
