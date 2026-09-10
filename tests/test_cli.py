"""The tool end to end, and the honesty of what it prints.

Two kinds of check. The first is that the equilibrium it reports is actually an
equilibrium: frequencies that sum to one, zero EV loss on the support, a duality gap at
machine precision. The second is that the output keeps its promises -- it never labels
anything a win probability, it says how much of the belief the matrix covered, and it
names what narrowing left out. Those are load-bearing claims about the numbers, so they
are tested like the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.cli import analyse, main, render, to_json
from pokeuraou.payoff import FAINTS, HP_SHARE
from pokeuraou.setup import load_scenario

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"


@pytest.fixture(scope="module")
def analysis():  # noqa: ANN201
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE.name} missing")
    scenario = load_scenario(EXAMPLE)
    return analyse(
        scenario, objective=HP_SHARE, cross=FAINTS, limit=6, belief_classes=2
    )


def test_the_reported_strategies_are_an_equilibrium(analysis) -> None:  # noqa: ANN001
    eq = analysis.equilibrium
    assert float(eq.row_strategy.sum()) == pytest.approx(1.0)
    assert (eq.row_strategy >= -1e-12).all()
    assert eq.duality_gap < 1e-6, "both LPs solve the same game; a gap means one is wrong"

    # Zero EV loss on the support is the definition of an equilibrium, and the printed
    # "EV損" column is exactly this quantity.
    for index in eq.row_support():
        assert eq.row_ev_loss[index] == pytest.approx(0.0, abs=1e-9)

    # The opponent knows their own class, so they get one strategy per class and each is
    # a distribution supported on that class's best replies.
    assert len(eq.col_strategies) == len(analysis.matrices)
    for strategy, loss in zip(eq.col_strategies, eq.col_ev_loss, strict=True):
        assert float(strategy.sum()) == pytest.approx(1.0)
        for index in np.flatnonzero(strategy > 1e-6):
            assert loss[index] == pytest.approx(0.0, abs=1e-9)


def test_the_matrix_is_not_averaged_before_solving(analysis) -> None:  # noqa: ANN001
    """Averaging first would hand the opponent our uncertainty about their own spread.

    Information cannot hurt the player holding it, so the Bayesian value can only be at
    or below the averaged one from our point of view.
    """
    from pokeuraou.equilibrium import solve

    averaged = solve(analysis.payoff)
    assert analysis.equilibrium.value <= averaged.value + 1e-9


def test_the_value_is_between_the_matrix_extremes(analysis) -> None:  # noqa: ANN001
    assert analysis.payoff.min() - 1e-9 <= analysis.equilibrium.value
    assert analysis.equilibrium.value <= analysis.payoff.max() + 1e-9


def test_the_matrix_is_the_size_it_says(analysis) -> None:  # noqa: ANN001
    assert analysis.payoff.shape == (len(analysis.row.kept), len(analysis.col.kept))
    assert analysis.cells == analysis.payoff.size


def test_the_belief_weights_are_a_distribution(analysis) -> None:  # noqa: ANN001
    total = sum(c.weight for c in analysis.classes)
    assert total == pytest.approx(1.0), "the truncated classes are renormalised"
    assert 0.0 < analysis.covered_mass <= 1.0


def test_the_output_never_calls_anything_a_win_probability(analysis) -> None:  # noqa: ANN001
    """The one claim the whole objective module exists to avoid making."""
    text = render(analysis)
    assert "勝率ではありません" in text
    payload = to_json(analysis)
    assert payload["objective"]["isWinProbability"] is False
    for entry in payload["belief"].values():
        assert entry["isPosterior"] is False, "there is no evidence yet; it is a prior"


def test_the_output_states_its_own_coverage(analysis) -> None:  # noqa: ANN001
    text = render(analysis)
    assert "覆った同値類の質量" in text
    assert "候補" in text
    payload = to_json(analysis)
    assert payload["coverage"]["beliefMass"] == pytest.approx(analysis.covered_mass)
    assert payload["coverage"]["candidatesOwn"]["considered"] >= len(analysis.row.kept)


def test_the_json_output_is_serialisable(analysis) -> None:  # noqa: ANN001
    text = json.dumps(to_json(analysis), ensure_ascii=False)
    assert json.loads(text)["regulation"]


def test_the_frequencies_and_ev_losses_line_up(analysis) -> None:  # noqa: ANN001
    payload = to_json(analysis)
    assert len(payload["own"]) == len(analysis.row.kept)
    frequencies = np.array([e["frequency"] for e in payload["own"]])
    assert frequencies.sum() == pytest.approx(1.0)
    for entry in payload["own"]:
        if entry["frequency"] > 1e-6:
            assert entry["evLoss"] == pytest.approx(0.0, abs=1e-9)
    opponent = np.array([e["frequency"] for e in payload["opponent"]])
    assert opponent.sum() == pytest.approx(1.0), "the marginal is what a player observes"
    assert len(payload["opponentBestReplies"]) == len(analysis.matrices)


WITH_OBSERVATIONS = (
    Path(__file__).resolve().parents[1] / "examples" / "scenario-turn5.json"
)


def test_observations_produce_a_posterior_not_a_prior() -> None:
    """The same position with evidence attached must report itself as a posterior.

    And the evidence has to actually do something: a damage number that eliminates no
    candidate would mean the likelihood is not being applied.
    """
    if not WITH_OBSERVATIONS.exists():
        pytest.skip(f"{WITH_OBSERVATIONS.name} missing")
    scenario = load_scenario(WITH_OBSERVATIONS)
    assert scenario.raw_observations, "the example is supposed to carry observations"
    analysis = analyse(
        scenario, objective=HP_SHARE, cross=FAINTS, limit=4, belief_classes=1
    )
    assert analysis.update_report is not None
    assert analysis.update_report.entries, (
        f"no observation was applied: {analysis.update_report.skipped}"
    )
    narrowed = [
        (before, after) for _l, _k, before, after, _b in analysis.update_report.entries
    ]
    assert any(after < before for before, after in narrowed), (
        "no observation eliminated a single candidate; the likelihood is not biting"
    )

    text = render(analysis)
    assert "事後分布" in text
    payload = to_json(analysis)
    assert payload["observations"]
    assert all(entry["isPosterior"] for entry in payload["belief"].values())
    assert payload["update"]["applied"]


def test_the_command_line_runs(capsys: pytest.CaptureFixture[str]) -> None:
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE.name} missing")
    code = main([str(EXAMPLE), "--limit", "4", "--belief-classes", "1", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["equilibrium"]["dualityGap"] < 1e-6
    assert payload["cost"]["cells"] == 16
