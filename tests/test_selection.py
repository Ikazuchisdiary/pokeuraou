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
from pokeuraou.standin import RowStub
from pokeuraou.teams import all_selections, load_roster

REGULATION = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    return load_roster("rizabanadohido")


def EncodedStub(reg, score) -> RowStub:  # noqa: ANN001, N802
    """A stand-in leaf read off the encoded rows, so the port's `selectionSolve` can score it
    on a stand-in inference server (IKA-463). ``score(species, active)`` gets each row's
    species ids and whether each is on the field, both shaped (rows, 2, mons a side), absent
    mons ``""``."""
    return RowStub(reg, lambda rows: score(rows.species, rows.active))


def _sigmoid(x):  # noqa: ANN001, ANN202
    return 1.0 / (1.0 + np.exp(-x))


def _antisymmetric_stub(reg, scores: dict[str, float]) -> RowStub:  # noqa: ANN001
    """A stand-in value function: each side's strength is the sum of its members' scores.

    Antisymmetric by construction -- ``V = sigmoid(ours - theirs)`` -- so it satisfies the
    one property the real network guarantees, which is what the solver relies on.
    """
    value = np.vectorize(lambda s: scores.get(s, 0.0), otypes=[np.float64])

    def score(species, _active):  # noqa: ANN001, ANN202
        strength = value(species).sum(axis=-1)
        return _sigmoid(strength[:, 0] - strength[:, 1])

    return EncodedStub(reg, score)


def test_the_action_space_is_the_ordered_ninety(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    stub = EncodedStub(reg, lambda species, _active: np.full(len(species), 0.5))
    analysis = solve_selection(reg, roster.sets, classes, stub)
    assert len(analysis.ours) == 90
    assert len(analysis.theirs) == 90
    assert analysis.matrices[0].shape == (90, 90)
    # 90x90 for the matrix, then the antisymmetry sample and its mirror.
    assert stub.calls[0] == 8100
    assert stub.calls[1:] == [8, 8]
    assert analysis.positions_evaluated == 8116


def test_a_mirror_is_exactly_even(roster) -> None:  # noqa: ANN001
    """Same six, same spreads: the game is symmetric and the value is forced."""
    reg = roster.reg
    scores = {m.species: float(i) for i, m in enumerate(roster.sets)}
    classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    analysis = solve_selection(reg, roster.sets, classes, _antisymmetric_stub(reg, scores))
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

    value = np.vectorize(lambda s: scores.get(s, 0.0), otypes=[np.float64])
    evaluate = EncodedStub(reg, lambda species, _active: _sigmoid(value(species[:, 0]).sum(axis=-1)))
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

    def score(species, active):  # noqa: ANN001, ANN202
        led = 3.0 * ((species == star) & active).sum(axis=-1)
        return _sigmoid(led[:, 0] - led[:, 1])

    analysis = solve_selection(reg, roster.sets, classes, EncodedStub(reg, score))
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
    analysis = solve_selection(reg, roster.sets, classes, _antisymmetric_stub(reg, scores))
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


def _python_solve(reg, our_six, classes, evaluate):  # noqa: ANN001, ANN202
    """The reference: `solve_selection` as it was before IKA-463 -- the openings' branches
    scored in Python and folded by weight, and Python's `solve_bayesian`."""
    from pokeuraou.equilibrium import solve_bayesian
    from pokeuraou.selfplay import lead_branches_from_sets

    selections = all_selections(reg.meta.team_size, reg.meta.picked_team_size)
    matrices = []
    for c in classes:
        openings = lead_branches_from_sets(
            reg, [([our_six[i] for i in ours], [c.sets[j] for j in theirs])
                  for ours in selections for theirs in selections])
        positions = [pos for branches in openings for _w, pos in branches]
        owner = np.repeat(np.arange(len(openings)), [len(b) for b in openings])
        weights = np.array([w for branches in openings for w, _pos in branches])
        values = np.bincount(owner, weights=weights * evaluate(positions), minlength=len(openings))
        matrices.append(values.reshape(len(selections), len(selections)))
    return matrices, solve_bayesian(matrices, np.array([c.weight for c in classes]))


def test_the_port_s_solve_is_the_python_solve_it_replaced(roster) -> None:  # noqa: ANN001
    """IKA-463: `solve_selection` is the port's `selectionSolve`, one request. On two spread
    classes that differ, its matrices, mixtures and value are the Python solve's
    (`_python_solve`). The comparison can fail: the classes' matrices differ, so a class taken
    for the other moves them, and the equilibrium is mixed."""
    import dataclasses

    from pokeuraou import portlp

    reg = roster.reg
    scores = {m.species: float(i) * 1.1 for i, m in enumerate(roster.sets)}
    value = np.vectorize(lambda s: scores.get(s, 0.0), otypes=[np.float64])

    def score(rows):  # noqa: ANN001, ANN202
        # Cyclic in the leads (sin of the difference), so the game mixes; the HP, which the
        # second class moves, makes the two classes' matrices differ.
        x = (value(rows.species) * rows.active + rows.maxhp / 500.0 * ~rows.active).sum(axis=-1)
        return _sigmoid(3.0 * np.sin(x[:, 0] - x[:, 1]))

    stub = RowStub(reg, score)
    moved = tuple(dataclasses.replace(s, sp={"hp": 32, "def": 32, "spd": 2}) for s in roster.sets)
    classes = [SpreadClass(0.3, tuple(roster.sets), "sheet"), SpreadClass(0.7, moved, "moved")]
    before = portlp.COUNTS["crossings"]
    analysis = solve_selection(reg, roster.sets, classes, stub)
    assert portlp.COUNTS["crossings"] == before + 1
    matrices, reference = _python_solve(reg, roster.sets, classes, stub)
    assert float(np.max(np.abs(matrices[0] - matrices[1]))) > 0.01
    for got, want in zip(analysis.matrices, matrices, strict=True):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    eq = analysis.equilibrium
    assert len(eq.row_support()) > 1, "a pure answer would not tell the mixtures apart"
    np.testing.assert_allclose(eq.row_strategy, reference.row_strategy, atol=1e-9)
    for got, want in zip(eq.col_strategies, reference.col_strategies, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)
    assert analysis.value == pytest.approx(float(reference.value), abs=1e-12)
