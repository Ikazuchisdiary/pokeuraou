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


def _same_bayesian(a: equilibrium.BayesianEquilibrium,
                   b: equilibrium.BayesianEquilibrium) -> bool:
    def raw(v) -> bytes:  # noqa: ANN001
        return b"" if v is None else np.asarray(v).tobytes()

    return (a.value == b.value and a.duality_gap == b.duality_gap
            and all(raw(getattr(a, f)) == raw(getattr(b, f))
                    for f in ("row_strategy", "col_marginal", "row_ev", "row_ev_loss", "weights"))
            and len(a.col_strategies) == len(b.col_strategies)
            and all(raw(x) == raw(y) for x, y in zip(a.col_strategies, b.col_strategies,
                                                     strict=True))
            and all(raw(x) == raw(y) for x, y in zip(a.col_ev_loss, b.col_ev_loss, strict=True)))


def test_the_ports_bayesian_equilibria_are_solve_bayesians(roster) -> None:  # noqa: ANN001
    """IKA-387: `solve_bayesian_many` gives `solve_bayesian`'s `BayesianEquilibrium` to the bit
    (every field) -- one class (a deep child's pass rectangle, `ladder._kids_solve`) and
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
        assert _same_bayesian(e, equilibrium.solve_bayesian(mats, w))
    good = games[1]
    bad = [([np.array([[0.5, np.nan]])], np.asarray([1.0])), ([], np.asarray([])),
           ([np.ones((2, 2)), np.ones((3, 2))], np.asarray([0.5, 0.5])),
           ([np.ones((2, 2))], np.asarray([-1.0])), good]
    errors = portlp.solve_bayesian_many(roster.reg, bad)
    assert all(isinstance(e, ValueError) for e in errors[:4])
    assert _same_bayesian(errors[4], equilibrium.solve_bayesian(*good))
    # One at a time, on and off: the same answer.
    was = portlp.ON[0]
    try:
        for on in (True, False):
            portlp.set_on(on)
            one = portlp.solve_bayesian_one(roster.reg, *good)
            assert _same_bayesian(one, equilibrium.solve_bayesian(*good))
            assert _same_equilibrium(portlp.solve_one(roster.reg, good[0][0]),
                                     equilibrium.solve(good[0][0]))
    finally:
        portlp.set_on(was)


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


def _kids(rng: np.random.Generator, count: int) -> list:
    """Deep children (`ladder._Child`) begun on random prices, their first pass read."""
    from pokeuraou import ladder

    kids = []
    for _ in range(count):
        m, n = (int(v) for v in rng.integers(2, 9, 2))
        prices = rng.random((m, n))
        if rng.random() < 0.3:
            prices = np.round(prices, 1)
        start = equilibrium.solve(prices)
        kid = ladder._Child(None, list(range(m)), list(range(n)), prices,
                            np.asarray(start.row_strategy), np.asarray(start.col_strategy))
        kid.rows = [int(i) for i in rng.permutation(m)[:max(1, m // 2)]]
        kid.cols = [int(j) for j in rng.permutation(n)[:max(1, n // 2)]]
        kid.trial = prices.copy()
        kids.append(kid)
    return kids


def _read_pass(rng: np.random.Generator, kids: list) -> None:
    for kid in kids:
        for i in kid.rows:
            for j in kid.cols:
                kid.memo.setdefault((i, j), float(rng.random()) if rng.random() < 0.9 else None)


def test_the_kids_passes_solved_together_are_solved_alone(roster) -> None:  # noqa: ANN001
    """IKA-387 (`ladder._kids_solve`): a pass of many deep children, their rectangles solved in
    the port in one crossing, is each child's `_kid_solve` with scipy -- answer, the oracle's
    rows and columns, whether it goes on -- pass after pass, each child its own. The positive
    control: the Bayesian games the port solved, one crossing a pass."""
    from pokeuraou import ladder

    was = portlp.ON[0]
    try:
        for seed in range(4):
            got = {}
            for on in (False, True):
                rng = np.random.default_rng(3870 + seed)
                kids = _kids(rng, 7)
                portlp.set_on(on)
                before = (portlp.COUNTS["bayes"], portlp.COUNTS["crossings"])
                passes = 3
                for attempt in range(passes + 1):
                    _read_pass(rng, kids)
                    live = [k for k in kids if k.active]
                    ladder._kids_solve(roster.reg, [(k, attempt, passes) for k in live])
                    if on and live:
                        assert portlp.COUNTS["bayes"] - before[0] == len(live)
                        assert portlp.COUNTS["crossings"] - before[1] == 1
                        before = (portlp.COUNTS["bayes"], portlp.COUNTS["crossings"])
                got[on] = [(k.rows, k.cols, k.active, None if k.answer is None else (
                    k.answer[0].tobytes(), [y.tobytes() for y in k.answer[1]], k.answer[2]))
                    for k in kids]
            assert got[True] == got[False]
            # The children differ (a child's answer given to another would show).
            assert len({str(a) for a in got[True]}) == len(got[True])
    finally:
        portlp.set_on(was)


def test_the_kids_matrices_solved_together_are_solved_alone(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-387 (`ladder._kids_fill`): the deep children's own matrices, solved in the port in
    one crossing, are each `solve` with scipy -- prices, both strategies, notes and counted
    work. The leaf is the port's filled nodes with made-up values (`_filled`). The positive
    control: the games the port solved."""
    from pokeuraou import ladder

    subs = _filled(roster)
    pendings = [s.pending for s in subs]
    monkeypatch.setattr(port, "pending_payoffs", lambda *_a, **_k: pendings)
    monkeypatch.setattr(port, "score_stacked", lambda _leaf, encoded: [p.values for p in pendings])
    monkeypatch.setattr(ladder, "STACK", True)
    was = portlp.ON[0]
    got = {}
    try:
        for on in (False, True):
            portlp.set_on(on)
            kids = [ladder._Child(None, [], [], None, None, None) for _ in pendings]
            games = portlp.COUNTS["games"]
            notes: set[str] = set()
            assert ladder._kids_fill(roster.reg, kids, None, Budget.matrix(), notes) is None
            if on:
                assert portlp.COUNTS["games"] - games == len(kids)
            got[on] = ([(k.prices.tobytes(), k.x.tobytes(), k.y.tobytes(), sorted(k.notes),
                         k.work) for k in kids], sorted(notes))
    finally:
        portlp.set_on(was)
    assert got[True] == got[False]
    assert len({a[1] + a[2] for a in got[True][0]}) == len(pendings)


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
