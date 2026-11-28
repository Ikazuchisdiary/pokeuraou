"""The port's LPs (IKA-381, `portlp`, `rust/src/lp.rs`) against `equilibrium`'s.

What these hold:

- **the same games**: `solve_many` gives an `Equilibrium` of `solve`'s game -- the value
  within `VALUE_TOL`, strategies that are an equilibrium (`tests/_equilibrium.py`), the EVs
  of those strategies -- on random games, degenerate ones (ties, dominated actions, a single
  row or column) and Bayesian ones; a non-finite matrix is `solve`'s ValueError, returned in
  its place. The port's own count of LPs is the positive control. Since IKA-470 the port's
  HiGHS options are not scipy's, so on a degenerate game the strategy may be another optimal
  vertex: the strategies are not held to the bit any more.

The depth-2 refine's sub-games (`solve_subs`) and the one-game switch (`solve_one`,
`solve_bayesian_one`) went with the depth-2 refine (IKA-464). No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import equilibrium, portlp, rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.teams import load_roster
from tests._equilibrium import VALUE_TOL, is_bayesian_equilibrium, is_the_solve


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _games(rng: np.random.Generator) -> list[np.ndarray]:
    games = []
    for k in range(300):
        m, n = (int(v) for v in rng.integers(1, 26, 2))
        a = rng.random((m, n))
        if k % 3 == 0:
            a = np.round(a, 1)  # ties: many optimal vertices, the pivots decide which
        if k % 5 == 0:
            a[0] = a.max(axis=0)  # a dominating row
        games.append(a)
    return games


TOL = 1e-12


def _close(a, b) -> bool:  # noqa: ANN001
    a, b = (np.zeros(0) if v is None else np.asarray(v, dtype=np.float64) for v in (a, b))
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= TOL))


def test_the_ports_games_are_solves(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    games = _games(np.random.default_rng(381))
    before = portlp.COUNTS["lps"]
    got = portlp.solve_many(reg, games)
    assert portlp.COUNTS["lps"] - before == 2 * len(games)
    for a, e in zip(games, got, strict=True):
        assert isinstance(e, equilibrium.Equilibrium)
        assert is_the_solve(a, e, equilibrium.solve(a))
    bad = np.array([[0.5, np.nan], [0.1, 0.2]])
    errors = portlp.solve_many(reg, [bad, np.zeros((0, 3)), games[0]])
    assert isinstance(errors[0], ValueError) and isinstance(errors[1], ValueError)
    assert is_the_solve(games[0], errors[2], equilibrium.solve(games[0]))


def test_the_ports_bayesian_games_are_solves(roster) -> None:  # noqa: ANN001
    """The `lp` command's Bayesian games (the deep children's rectangles, `_kid_solve`)."""
    import base64
    import json

    rng = np.random.default_rng(382)
    node = rustnode.require_node(roster.reg)
    for _ in range(40):
        m = int(rng.integers(1, 12))
        mats = [rng.random((m, int(rng.integers(1, 12)))) for _ in range(int(rng.integers(1, 4)))]
        w = rng.random(len(mats))
        want = equilibrium.solve_bayesian(mats, w)
        game = {"bayes": [portlp._game(a) for a in mats], "weights": portlp._b64(w)}
        one = node.lp({"kind": "lp", "games": [game]})["answers"][0]
        assert abs(0.5 * (one["valueRow"] + one["valueCol"]) - want.value) <= VALUE_TOL
        x = np.frombuffer(base64.b64decode(one["x"]), dtype="<f8")
        ys = [np.frombuffer(base64.b64decode(y), dtype="<f8") for y in one["ys"]]
        assert len(ys) == len(want.col_strategies)
        assert is_bayesian_equilibrium(mats, w, x, ys, want.value)
        json.dumps(one)  # the answer is plain JSON


def _is_the_bayesian_solve(mats, w, got, want) -> bool:  # noqa: ANN001
    """`got` is an equilibrium of the game `want = solve_bayesian(mats, w)` solved."""
    ws = np.asarray(w, dtype=np.float64) / np.sum(w)
    return (abs(got.value - want.value) <= VALUE_TOL
            and is_bayesian_equilibrium(mats, w, got.row_strategy, list(got.col_strategies),
                                        want.value)
            and len(got.col_strategies) == len(want.col_strategies)
            and _close(got.weights, ws)
            and np.allclose(got.row_ev, sum(ws[k] * (mats[k] @ got.col_strategies[k])
                                            for k in range(len(mats))), atol=1e-9))


def test_the_ports_bayesian_equilibria_are_solve_bayesians(roster) -> None:  # noqa: ANN001
    """IKA-387: `solve_bayesian_many` gives an equilibrium of `solve_bayesian`'s game
    (the value, the strategies as an equilibrium, the EVs of those strategies) -- one class
    (a deep child's pass rectangle, `ladder._kids_solve`) and
    several (a stage's rectangle, `ladder.read`), with ties, a dominating row, a shared or
    uneven column count and weights not summing to 1 -- in one crossing; an invalid game is
    `solve_bayesian`'s ValueError in its place. The positive control: the port's LPs."""
    rng = np.random.default_rng(387)
    games = []
    for n in range(200):
        m = int(rng.integers(1, 20))
        k = 1 if n % 2 == 0 else int(rng.integers(2, 5))
        shared = int(rng.integers(1, 20))
        mats = []
        for _ in range(k):
            a = rng.random((m, shared if n % 3 else int(rng.integers(1, 20))))
            if n % 5 == 0:
                a = np.round(a, 1)
            if n % 7 == 0:
                a[0] = a.max(axis=0)
            mats.append(a)
        w = rng.random(k) * 3.0 if k > 1 else np.asarray([1.0])
        games.append((mats, w))
    before = (portlp.COUNTS["lps"], portlp.COUNTS["crossings"])
    got = portlp.solve_bayesian_many(roster.reg, games)
    assert portlp.COUNTS["lps"] - before[0] == 2 * len(games)
    assert portlp.COUNTS["crossings"] - before[1] == 1
    for (mats, w), e in zip(games, got, strict=True):
        assert isinstance(e, equilibrium.BayesianEquilibrium)
        assert _is_the_bayesian_solve(mats, w, e, equilibrium.solve_bayesian(mats, w))
    good = games[1]
    bad = [([np.array([[0.5, np.nan]])], np.asarray([1.0])), ([], np.asarray([])),
           ([np.ones((2, 2)), np.ones((3, 2))], np.asarray([0.5, 0.5])),
           ([np.ones((2, 2))], np.asarray([-1.0])), good]
    errors = portlp.solve_bayesian_many(roster.reg, bad)
    assert all(isinstance(e, ValueError) for e in errors[:4])
    assert _is_the_bayesian_solve(*good, errors[4], equilibrium.solve_bayesian(*good))


def test_the_equilibrium_check_can_fail() -> None:
    """The control of the tests above (IKA-470): a strategy that is not an equilibrium, or a
    value that is off, is refused."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    good = equilibrium.solve(a)
    assert is_the_solve(a, good, good)
    from dataclasses import replace
    assert not is_the_solve(a, replace(good, row_strategy=np.array([1.0, 0.0])), good)
    assert not is_the_solve(a, replace(good, value=good.value + 1e-4), good)
