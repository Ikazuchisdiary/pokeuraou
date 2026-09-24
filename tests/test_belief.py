"""The belief layer, and the one property it is not allowed to get wrong.

Reducing the particle set is only legitimate if it is lossless: every member of a class
has to produce the same resolved turn, or the reduction is quietly changing the numbers
the tool prints. So the central test here resolves the *same* action pair against several
members of one class and asserts the payoffs are equal -- not close, equal.

The rest guards the inputs to that: that a revealed nature is actually used, that a belief
is never invented when the data has nothing to say, and that no move in the regulation
reads a stat in a way the liveness analysis does not know about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokeuraou.actions import side_actions
from pokeuraou.belief import (
    BeliefError,
    build_belief,
    faster_probability,
    live_stats_for,
    reduce_for_position,
    unmodelled_stat_readers,
)
from pokeuraou.payoff import HP_SHARE
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import Regulation
from pokeuraou.setup import load_scenario, with_spreads

from ._port import Budget, resolve_turn, turn_expectation

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"


@pytest.fixture(scope="module")
def prior(reg: Regulation):  # noqa: ANN201
    path = find_cached_chaos(reg.meta.format_id)
    if path is None:
        pytest.skip("no cached usage stats; run tools/fetch_priors.py")
    return load_chaos(path, reg)


@pytest.fixture(scope="module")
def scenario():  # noqa: ANN201
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE.name} missing")
    return load_scenario(EXAMPLE)


def _species_with_particles(prior, reg: Regulation) -> tuple[str, str]:  # noqa: ANN001
    for entry in prior.by_usage():
        if entry.n_particles < 50:
            continue
        nature = str(entry.natures[int(np.argmax(entry.weights))])
        if nature.lower() in {n.lower() for n in reg.natures}:
            return entry.species_id, nature
    pytest.skip("no species with enough particles")


def test_a_revealed_nature_is_actually_used(prior, reg: Regulation) -> None:  # noqa: ANN001
    """The open team sheet shows nature, which is a 20-fold cut for free.

    Ignoring it would leave the belief reasoning about spreads the sheet has already ruled
    out.
    """
    species_id, nature = _species_with_particles(prior, reg)
    belief = build_belief(reg, prior, species_id, nature)
    assert belief.size > 0
    assert belief.size <= prior.species[species_id].n_particles
    assert float(belief.weights.sum()) == pytest.approx(1.0)
    assert belief.nature == nature
    assert "nature" in belief.provenance


def test_a_species_with_no_data_is_refused(prior, reg: Regulation) -> None:  # noqa: ANN001
    """Inventing a spread distribution would make every number downstream fiction."""
    absent = next(
        (s for s in reg.species if s not in prior.species and reg.species[s].team_legal),
        None,
    )
    if absent is None:
        pytest.skip("every legal species appears in the usage data")
    with pytest.raises(BeliefError, match="no usage data"):
        build_belief(reg, prior, absent, "Adamant")


def test_impossible_evidence_raises_rather_than_renormalising(
    prior, reg: Regulation  # noqa: ANN001
) -> None:
    species_id, nature = _species_with_particles(prior, reg)
    belief = build_belief(reg, prior, species_id, nature)
    with pytest.raises(BeliefError, match="probability zero"):
        belief.reweight(np.zeros(belief.size))


def test_a_likelihood_of_the_wrong_length_raises(prior, reg: Regulation) -> None:  # noqa: ANN001
    species_id, nature = _species_with_particles(prior, reg)
    belief = build_belief(reg, prior, species_id, nature)
    with pytest.raises(BeliefError):
        belief.reweight(np.ones(belief.size + 1))


def test_no_move_reads_a_stat_the_analysis_does_not_know_about(reg: Regulation) -> None:
    """The completeness check the reduction's losslessness depends on.

    A regulation that adds a move computing its power from a stat has to fail here, not
    silently merge particles that resolve differently.
    """
    assert unmodelled_stat_readers(reg) == ()


def test_liveness_is_per_pokemon_not_per_position(scenario) -> None:  # noqa: ANN001
    """A Pokemon with no special move does not have Special Attack read by anything.

    The first version asked the position-wide question and every stat came out live, which
    made the reduction collapse nothing at all.
    """
    reg = scenario.reg
    key = scenario.hidden_actives()[0]
    live, reasons = live_stats_for(reg, scenario.position, key)
    assert "hp" in live
    assert set(live) <= {"hp", "atk", "def", "spa", "spd"}
    for stat in live:
        assert reasons[stat].strip(), f"{stat} is kept for no stated reason"


def _reductions(scenario):  # noqa: ANN001
    from pokeuraou.cli import _modal, build_beliefs

    beliefs = build_beliefs(scenario)
    modal = {key: _modal(b) for key, b in beliefs.items()}
    base = with_spreads(scenario, modal)
    active = {}
    for side_index, side in enumerate(base.sides):
        for slot, party_index in enumerate(side.active):
            if (side_index, party_index) in beliefs:
                active[(side_index, slot)] = beliefs[(side_index, party_index)]
    return base, modal, {k: reduce_for_position(scenario.reg, base, k, active) for k in active}


def test_the_reduction_merges_something(scenario) -> None:  # noqa: ANN001
    base, _modal, reductions = _reductions(scenario)
    del base
    assert reductions
    for reduction in reductions.values():
        assert reduction.size <= reduction.original_size
        assert sum(len(m) for m in reduction.members) == reduction.original_size
        assert float(reduction.belief.weights.sum()) == pytest.approx(1.0)


def test_the_reduction_is_lossless(scenario) -> None:  # noqa: ANN001
    """Members of one class must resolve identically. This is the load-bearing claim.

    If it fails, the merged particle set is not a substitute for the full one and every
    equilibrium computed over it is wrong by an unknown amount.
    """
    reg = scenario.reg
    base, modal, reductions = _reductions(scenario)

    checked = 0
    for (side_index, slot), reduction in sorted(reductions.items()):
        party_index = base.sides[side_index].active[slot]
        groups = [g for g in reduction.members if len(g) > 1]
        if not groups:
            continue
        group = max(groups, key=len)[:4]
        ours = side_actions(reg, base, 0)[:3]
        theirs = side_actions(reg, base, 1)[:3]

        values: list[list[float]] = []
        for member in group:
            assignment = dict(modal)
            assignment[(side_index, party_index)] = reduction.original.spreads[member]
            pos = with_spreads(scenario, assignment)
            row = []
            for a in ours:
                for b in theirs:
                    result = resolve_turn(reg, pos, [a, b], budget=Budget.deterministic(0))
                    # `turn_expectation` rather than `expected`: a self-switching move
                    # suspends the turn for a replacement choice, and two members of one
                    # equivalence class have to agree on that choice too.
                    value, _flags = turn_expectation(reg, result, HP_SHARE)
                    row.append(value)
            values.append(row)
        for other in values[1:]:
            assert other == pytest.approx(values[0], abs=1e-12), (
                "two members of one equivalence class resolved differently, so the "
                "reduction is lossy and every equilibrium built on it is wrong"
            )
        checked += 1

    if not checked:
        pytest.skip("no class in this position merged more than one particle")


def test_faster_probability_is_a_probability(scenario) -> None:  # noqa: ANN001
    reg = scenario.reg
    _base, _modal, reductions = _reductions(scenario)
    for key, reduction in sorted(reductions.items()):
        faster, tie = faster_probability(
            reg, scenario.position, key, reduction.belief, (0, 0)
        )
        assert 0.0 <= faster <= 1.0
        assert 0.0 <= tie <= 1.0
        assert faster + tie <= 1.0 + 1e-9
