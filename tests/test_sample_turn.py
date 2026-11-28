"""IKA-469: a played turn is drawn along one path and draws from the branches' distribution.

The Rust worker's `playGame` used to resolve a played turn's every branch (`resolve_turn`,
the exact budget) and then draw one by its weight. It now draws each chance point as it is
met and resolves only the drawn path (`resolve::sample_turn`). The game changes (other draws
are made from the generator), and the distribution it draws from must not. These hold that
on the move decisions of the Rust worker's games (`tests/_worker_games.py`), with the port's
diagnostic `sampleCheck` (`rust/src/drawcheck.rs`). The draws are made under the exact budget
a game plays. The branches are enumerated under the same budget with its cap raised to
`ENUMERATION_CAP`, and only a turn whose enumeration is exact (nothing narrowed or dropped)
is compared. Under the cap of 512 most turns with two damaging moves are narrowed, and there
the old draw was from the narrowed branches while the new one is from the full ones.

- **every path**: where the paths are few enough to walk, the paths' outcomes and their
  probabilities are `resolve_turn`'s branches and pauses (largest difference below 1e-9).
- **draws**: 2,000 draws per decision land on `resolve_turn`'s outcomes only, at frequencies
  a chi-square test does not reject. The test can fail: the same counts are rejected against
  a distribution with half of the second heaviest outcome's weight moved to the heaviest.

Against a build with `ika469-control` (each chance point drawn at its weights squared)
both tests fail.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pokeuraou import port
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.position import Position
from pokeuraou.rustnode import _candidates, _position, dump_budget

from ._worker_games import worker_games

#: Paths walked per decision at most; a decision with more is left to the draws.
MAX_PATHS = 5_000
SAMPLES = 2_000
GAMES = 6
#: The enumeration's branch cap. The exact budget's 512 narrows most turns with two damaging
#: moves (its result is then not exact), and a turn drawn along one path never narrows.
ENUMERATION_CAP = 1 << 12


@pytest.fixture(scope="module")
def decisions(tmp_path_factory) -> tuple[Any, list[tuple[Position, list]]]:  # noqa: ANN001
    """(the regulation, every move decision of six worker games as (position, chosen))."""
    pool, games = worker_games(tmp_path_factory.mktemp("sample"), GAMES)
    reg = pool.reg
    found = []
    for game in games:
        for decision in game["decisions"]:
            if decision["kind"] != "move":
                continue
            pos = Position.from_json(decision["position"])
            menus = [{a.to_choice(): a for a in side_actions(reg, pos, i)} for i in (0, 1)]
            found.append((pos, [menus[0][decision["ownChosen"]], menus[1][decision["foeChosen"]]]))
    return reg, found


def _check(  # noqa: ANN001
    reg, pos: Position, chosen: list, *, max_paths: int, samples: int, seed: int, cap: int = ENUMERATION_CAP
) -> dict:
    """`sampleCheck`: draws under the exact budget a game plays, the enumeration under the same
    budget with the branch cap at ``cap``."""

    def call(node):  # noqa: ANN001, ANN202
        return node._exchange({
            "kind": "sampleCheck",
            "position": _position(pos),
            "actions": _candidates(chosen),
            "budget": dump_budget(replace(Budget.exact(), max_branches=cap)),
            "sampleBudget": dump_budget(Budget.exact()),
            "maxPaths": max_paths,
            "samples": samples,
            "seed": seed,
        })

    answer = port.ask(reg, call)
    assert "fault" not in answer and "refused" not in answer, answer
    return answer


def _chi_square(counts: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    """Pearson's statistic and degrees of freedom, outcomes expected under 5 pooled into one."""
    expected = p * counts.sum()
    small = expected < 5
    if small.any():
        counts = np.append(counts[~small], counts[small].sum())
        expected = np.append(expected[~small], expected[small].sum())
    keep = expected > 0
    stat = float((((counts - expected) ** 2)[keep] / expected[keep]).sum())
    return stat, int(keep.sum()) - 1


@pytest.fixture(scope="module")
def checked(decisions) -> list[dict]:  # noqa: ANN001
    """`sampleCheck`'s answer for each decision: its paths walked and its draws, in one call."""
    reg, found = decisions
    return [
        _check(reg, pos, chosen, max_paths=MAX_PATHS, samples=SAMPLES, seed=469 + k)
        for k, (pos, chosen) in enumerate(found)
    ]


def test_every_path_is_the_branches(checked) -> None:  # noqa: ANN001
    walked = [a for a in checked if a["paths"] is not None and a["exact"]]
    # The comparison ran, on turns that branch.
    assert len(walked) >= 20 and sum(a["outcomes"] > 1 for a in walked) >= 12, len(walked)
    worst = max(a["maxDiff"] for a in walked)
    assert worst < 1e-9, worst


def test_draws_land_at_the_weights(checked) -> None:  # noqa: ANN001
    from scipy.stats import chi2

    stats: list[tuple[float, int]] = []
    moved: list[tuple[float, int]] = []
    for answer in checked:
        if not answer["exact"] or answer["outcomes"] < 2:
            continue
        assert answer["unmatched"] == 0 and answer["nothing"] == 0, answer
        counts = np.array(answer["counts"], dtype=np.float64)
        p = np.array(answer["probabilities"], dtype=np.float64)
        stats.append(_chi_square(counts, p))
        # The control: half of the second heaviest outcome's weight moved to the heaviest.
        top = np.argsort(p)[::-1][:2]
        wrong = p.copy()
        shift = 0.5 * wrong[top[1]]
        wrong[top[0]] += shift
        wrong[top[1]] -= shift
        moved.append(_chi_square(counts, wrong))
    assert len(stats) >= 10, len(stats)
    total, dof = sum(s for s, _ in stats), sum(d for _, d in stats)
    assert chi2.sf(total, dof) > 1e-4, (total, dof)
    wrong_total, wrong_dof = sum(s for s, _ in moved), sum(d for _, d in moved)
    assert chi2.sf(wrong_total, wrong_dof) < 1e-4, (wrong_total, wrong_dof)
