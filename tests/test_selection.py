"""The 6->4 selection game, which is the largest single lever measured so far.

Across 13,395 self-play games our win rate ran from 43.4% to 87.3% depending on which four
were brought, so an error here outweighs anything inside a turn. Three properties carry the
correctness, and none of them needs a trained model -- the solver takes the value function
as a callable, so a stub with a known shape pins the machinery exactly.

- **the action space is the ordered 90**, leads first, because the first two start on the
  field;
- **a mirror is exactly 0.500.** With the same six and the same spreads on both sides, the
  matrix satisfies ``M[i][j] = 1 - M[j][i]`` and the game is symmetric, so its value is
  forced. Any other answer is a wiring bug -- an orientation flip, a mislabelled side --
  and this is the test that catches it;
- **the matrix orientation is ours-by-theirs.** Transposing it would silently solve the
  opponent's problem and report it as ours.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.selection import SpreadClass, cyclic_share, render, solve_selection
from pokeuraou.teams import all_selections, load_roster

REGULATION = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    return load_roster("rizabanadohido")


def _antisymmetric_stub(scores: dict[str, float]):  # noqa: ANN202
    """A stand-in value function: each side's strength is the sum of its members' scores.

    Antisymmetric by construction -- ``V = sigmoid(ours - theirs)`` -- so it satisfies the
    one property the real network guarantees, which is what the solver relies on.
    """

    def evaluate(positions):  # noqa: ANN001, ANN202
        out = np.empty(len(positions), dtype=np.float64)
        for i, pos in enumerate(positions):
            strengths = []
            for side in pos.sides:
                strengths.append(
                    sum(scores.get(m.species, 0.0) for m in side.pokemon)
                )
            out[i] = 1.0 / (1.0 + np.exp(-(strengths[0] - strengths[1])))
        return out

    return evaluate


def test_the_action_space_is_the_ordered_ninety(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    calls: list[int] = []

    def evaluate(positions):  # noqa: ANN001, ANN202
        calls.append(len(positions))
        return np.full(len(positions), 0.5)

    analysis = solve_selection(reg, roster.sets, classes, evaluate)
    assert len(analysis.ours) == 90
    assert len(analysis.theirs) == 90
    assert analysis.matrices[0].shape == (90, 90)
    # 90x90 for the matrix, then the antisymmetry sample and its mirror.
    assert calls[0] == 8100


def test_a_mirror_is_exactly_even(roster) -> None:  # noqa: ANN001
    """Same six, same spreads: the game is symmetric and the value is forced."""
    reg = roster.reg
    scores = {m.species: float(i) for i, m in enumerate(roster.sets)}
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    analysis = solve_selection(reg, roster.sets, classes, _antisymmetric_stub(scores))
    assert analysis.value == pytest.approx(0.5, abs=1e-9)
    assert analysis.antisymmetry_error < 1e-9
    assert analysis.equilibrium.duality_gap < 1e-9


def test_the_matrix_is_ours_by_theirs(roster) -> None:  # noqa: ANN001
    """A transposed matrix would solve the opponent's problem and label it ours.

    Built so the answer is unambiguous: one member is worth far more than the rest, so our
    equilibrium must bring it and the value must exceed one half.
    """
    reg = roster.reg
    star = roster.sets[0].species
    scores = {m.species: 0.0 for m in roster.sets}
    scores[star] = 6.0
    # The opponent's six is the same species with the star worth nothing to them, so our
    # advantage is real rather than symmetric.
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="weak")]

    def evaluate(positions):  # noqa: ANN001, ANN202
        out = np.empty(len(positions), dtype=np.float64)
        for i, pos in enumerate(positions):
            ours = sum(scores.get(m.species, 0.0) for m in pos.sides[0].pokemon)
            out[i] = 1.0 / (1.0 + np.exp(-ours))
        return out

    analysis = solve_selection(reg, roster.sets, classes, evaluate)
    assert analysis.value > 0.5
    for selection, frequency, _loss in analysis.our_frequencies():
        if frequency > 1e-6:
            assert star in {roster.sets[i].species for i in selection}


def test_the_leads_are_distinguished_from_the_bench(roster) -> None:  # noqa: ANN001
    """Being on the field has to change the value, or selection is only 15 choices.

    The stub scores a member only while it is active, so the equilibrium must lead the
    valuable one. If `position_from_sets` ignored the order, every selection containing it
    would be equally good and the support would be far wider.
    """
    reg = roster.reg
    star = roster.sets[0].species
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]

    def evaluate(positions):  # noqa: ANN001, ANN202
        out = np.empty(len(positions), dtype=np.float64)
        for i, pos in enumerate(positions):
            def active_score(side) -> float:  # noqa: ANN001
                return sum(
                    3.0
                    for m in side.pokemon
                    if m.species == star and m.active_index is not None
                )

            delta = active_score(pos.sides[0]) - active_score(pos.sides[1])
            out[i] = 1.0 / (1.0 + np.exp(-delta))
        return out

    analysis = solve_selection(reg, roster.sets, classes, evaluate)
    # Symmetric game, so the value is still a half; what matters is *which* selections.
    assert analysis.value == pytest.approx(0.5, abs=1e-9)
    for selection, frequency, _loss in analysis.our_frequencies():
        if frequency > 1e-6:
            leads = {roster.sets[i].species for i in selection[:2]}
            assert star in leads, "the valuable member has to be led, not benched"


def test_the_cyclic_decomposition_separates_rps_from_a_ladder() -> None:
    """The diagnostic that says whether a pure recommendation is credible."""
    rps = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    t_norm, c_norm, share = cyclic_share(rps)
    assert share > 0.99, f"rock-paper-scissors came out {share:.3f} cyclic"
    assert t_norm < 1e-9 < c_norm

    strength = np.array([0.0, 1.0, 2.0, 3.0])
    ladder = 0.5 + (strength[:, None] - strength[None, :]) * 0.1
    t_norm, c_norm, share = cyclic_share(ladder)
    assert share < 1e-9, f"a ladder came out {share:.3f} cyclic"
    assert c_norm < 1e-9 < t_norm

    with pytest.raises(ValueError, match="square"):
        cyclic_share(np.zeros((3, 4)))


def test_a_pure_equilibrium_is_reported_as_a_dominance_claim(roster) -> None:  # noqa: ANN001
    """The output has to say which kind of claim it is making.

    "Bring this four" and "mix over these four" are different advice, and a reader cannot
    tell them apart from a frequency table alone when the frequency happens to be 100%.
    """
    reg = roster.reg
    scores = {m.species: float(i) for i, m in enumerate(roster.sets)}
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    analysis = solve_selection(reg, roster.sets, classes, _antisymmetric_stub(scores))
    text = render(analysis)
    assert "均衡値" in text
    if len(analysis.equilibrium.row_support()) == 1:
        assert "支配的" in text
    assert "巡回の割合" in text


def test_it_refuses_to_solve_with_no_spread_classes(roster) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="no spread classes"):
        solve_selection(roster.reg, roster.sets, [], lambda p: np.zeros(len(p)))


def test_every_selection_appears_once(roster) -> None:  # noqa: ANN001
    """No duplicate representations, or the equilibrium would split its own mass."""
    selections = all_selections(roster.reg.meta.team_size, roster.reg.meta.picked_team_size)
    assert len(selections) == len(set(selections))
    seen = {(frozenset(s[:2]), frozenset(s[2:])) for s in selections}
    assert len(seen) == len(selections)
