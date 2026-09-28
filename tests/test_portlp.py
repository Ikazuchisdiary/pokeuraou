"""The port's LPs (IKA-381, `portlp`, `rust/src/lp.rs`) against `equilibrium`'s.

What these hold:

- **the same games**: `solve_many` gives `solve`'s `Equilibrium` to the bit -- value,
  strategies, both sides' EVs -- on random games, degenerate ones (ties, dominated actions,
  a single row or column) and Bayesian ones; a non-finite matrix is `solve`'s ValueError,
  returned in its place. The port's own count of LPs is the positive control.
- **the same sub-games**: `solve_subs` on nodes the port filled and encoded (the learned
  leaf's road, here with random leaf values): folded in the port from the span block and
  solved, each value is `solve(PendingPayoff.finish()).value` -- to the bit where numpy's dot
  takes the order the port copies (this machine's OpenBLAS core), to the last places where it
  does not; a node given whole is the same to the bit. The positive control: sub-games folded
  in the port.

No torch: the nodes are encoded by the port and scored with made-up values.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import equilibrium, port, portlp, rustnode, search
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.teams import load_roster

from .test_ladder import _played


def openblas_core() -> str | None:
    """The core numpy's OpenBLAS runs on this machine ("SkylakeX" on the 9800X3D), or None.

    `lp.rs::numpy_dot` copies the SkylakeX `ddot`'s order, so where numpy takes that core the
    port's fold is numpy's to the bit and the tests hold it to that; elsewhere the last place
    may move."""
    import ctypes
    import glob
    import os

    root = os.path.dirname(np.__file__)
    for lib in glob.glob(os.path.join(root, "..", "numpy.libs", "*openblas*")):
        try:
            loaded = ctypes.CDLL(lib)
        except OSError:
            continue
        for name in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename",
                     "openblas_get_corename64_", "openblas_get_corename"):
            fn = getattr(loaded, name, None)
            if fn is not None:
                fn.restype = ctypes.c_char_p
                return fn().decode()
    return None


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


def _same_equilibrium(a: equilibrium.Equilibrium, b: equilibrium.Equilibrium) -> bool:
    return (a.value == b.value and a.duality_gap == b.duality_gap
            and all(np.asarray(getattr(a, f)).tobytes() == np.asarray(getattr(b, f)).tobytes()
                    for f in ("row_strategy", "col_strategy", "row_ev", "col_ev",
                              "row_ev_loss", "col_ev_loss")))


def test_the_ports_games_are_solves(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    games = _games(np.random.default_rng(381))
    before = portlp.COUNTS["lps"]
    got = portlp.solve_many(reg, games)
    assert portlp.COUNTS["lps"] - before == 2 * len(games)
    for a, e in zip(games, got, strict=True):
        assert isinstance(e, equilibrium.Equilibrium)
        assert _same_equilibrium(e, equilibrium.solve(a))
    bad = np.array([[0.5, np.nan], [0.1, 0.2]])
    errors = portlp.solve_many(reg, [bad, np.zeros((0, 3)), games[0]])
    assert isinstance(errors[0], ValueError) and isinstance(errors[1], ValueError)
    assert _same_equilibrium(errors[2], equilibrium.solve(games[0]))


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
        assert 0.5 * (one["valueRow"] + one["valueCol"]) == want.value
        x = np.frombuffer(base64.b64decode(one["x"]), dtype="<f8")
        assert x.tobytes() == np.asarray(want.row_strategy).tobytes()
        for y, theirs in zip(one["ys"], want.col_strategies, strict=True):
            assert np.frombuffer(base64.b64decode(y), dtype="<f8").tobytes() == theirs.tobytes()
        json.dumps(one)  # the answer is plain JSON


#: (game seed, turn) of `_played` positions whose width-8 nodes hold a cell of 16 leaves
#: (the dot's blocks) and cells a replacement stopped (9 and 14 fold trees).
FILLED = ((0, 0), (1, 2), (4, 3))


def _filled(roster, count: int = 3):  # noqa: ANN001, ANN202
    """Nodes the port filled and encoded, with made-up leaf values: `search._Sub`s pending."""
    reg = roster.reg
    budget = Budget.matrix()
    asks = []
    for seed, turn in FILLED[:count]:
        pos = _played(roster, turns=6, seed=seed)[turn]
        ours = narrow(reg, pos, 0, limit=8).actions
        theirs = narrow(reg, pos, 1, limit=8).actions
        asks.append((pos, list(ours), list(theirs)))
    was = rustnode._SPAN_BLOCKS[0]
    rustnode._SPAN_BLOCKS[0] = True
    try:
        nodes = port.ask(reg, lambda node: node.fill_encoded_many(asks, budget))
    finally:
        rustnode._SPAN_BLOCKS[0] = was
    rng = np.random.default_rng(383)
    subs = []
    for (_pos, ours, theirs), node in zip(asks, nodes, strict=True):
        assert node.span_block is not None and not node.refused
        pending = port.PendingPayoff(node, (len(ours), len(theirs)))
        pending.scored(rng.random(pending.rows))
        subs.append(search._Sub(pending=pending))
    return subs


def test_the_ports_sub_games_are_folds_and_solves(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    subs = _filled(roster)
    # The roads are met: a span of 16 leaves, and fold trees.
    assert max(len(span[2]) for s in subs for span in s.pending.filled.spans) >= 16
    assert sum(len(s.pending.filled.folded) for s in subs) >= 20
    want = [float(equilibrium.solve(s.pending.finish()).value) for s in subs]
    folded = portlp.COUNTS["folded"]
    portlp.solve_subs(reg, subs)
    assert portlp.COUNTS["folded"] - folded == len(subs)
    got = [s.solved for s in subs]
    assert all(s.done for s in subs)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    if openblas_core() == "SkylakeX":
        assert got == want
    # Whole matrices (the hp-share road): the same to the bit, wherever it runs.
    whole = [search._Sub(payoff=s.pending.finish()) for s in _filled(roster)]
    portlp.solve_subs(reg, whole)
    assert [s.solved for s in whole] == [
        float(equilibrium.solve(s.payoff).value) for s in whole]


def test_a_sub_game_solved_already_or_elsewhere_is_left_alone(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    subs = _filled(roster, 2)
    subs[0].done, subs[0].solved = True, 0.25
    alias = search._Sub(alias=subs[1])
    shared = search._Sub(value=0.5, shared=True)
    before = portlp.COUNTS["folded"]
    portlp.solve_subs(reg, [subs[0], alias, shared, subs[1]])
    assert portlp.COUNTS["folded"] - before == 1  # subs[1], once, through its alias
    assert subs[0].solved == 0.25 and subs[1].done and not alias.done and not shared.done
