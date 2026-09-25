"""tools/refine_signals.py: the offline half of IKA-281's refine-selection measurement.

The collect half needs the port and the recorded games; what is tested here is that the
numbers it stores are read the way the shipped code reads them (the chance-branch
arithmetic of `search._refined_value`, the refine loop of `search.search`), and that the
statistics and the selections have working controls: an oracle that must win, a random
score that must not, and a full budget that must agree with the reference exactly.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import search as search_mod

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("refine_signals", ROOT / "tools" / "refine_signals.py")
refine_signals = importlib.util.module_from_spec(spec)
sys.modules["refine_signals"] = refine_signals  # its dataclasses look themselves up there
spec.loader.exec_module(refine_signals)


def _cell(rng: random.Random, branches: int) -> dict:
    p = sorted((rng.random() + 0.01 for _ in range(branches)), reverse=True)
    total = sum(p)
    p = [v / total for v in p]
    return {
        "p": p,
        "sub": [rng.random() for _ in p],
        "solved": [1] * len(p),
        "leaf": [rng.random() for _ in p],
    }


def test_the_shipped_branch_arithmetic_is_what_is_recomputed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`branch_estimate(.., 3, "prob", fill=False)` is `_refined_value`, bit for bit."""
    rng = random.Random(3)
    compared = 0
    moved = 0
    for trial in range(40):
        cell = _cell(rng, 1 + trial % 7)
        outcomes = [SimpleNamespace(probability=p, position=k) for k, p in enumerate(cell["p"])]
        # The port hands branches in its own order; the shipped code sorts them.
        rng.shuffle(outcomes)
        monkeypatch.setattr(
            search_mod.port, "turn",
            lambda *_a, outcomes=outcomes, **_k: SimpleNamespace(
                unmodelled=set(), suspended=False, outcomes=outcomes),
        )
        monkeypatch.setattr(
            search_mod, "_subgame_value",
            lambda _reg, position, *_a, cell=cell, **_k: (cell["sub"][position], set(), 1),
        )
        shipped, _notes, _solved = search_mod._refined_value(
            None, None, None, None, None, budget=None, sub_limit=8, sub_branches=3
        )
        mine = refine_signals.branch_estimate(cell, 3, "prob", fill=False)
        assert mine == shipped
        compared += 1
        # Positive control: pricing the dropped branches at their leaf is a different
        # number whenever something was dropped, so the equality above is not vacuous.
        if len(cell["p"]) > 3:
            moved += refine_signals.branch_estimate(cell, 3, "prob", fill=True) != shipped
    assert compared == 40 and moved > 0


def test_every_branch_kept_is_the_same_value_however_chosen() -> None:
    rng = random.Random(5)
    for _ in range(20):
        cell = _cell(rng, rng.randint(1, 9))
        n = len(cell["p"])
        whole = refine_signals.branch_estimate(cell, n, "prob", fill=False)
        for how in ("prob", "spread"):
            for fill in (False, True):
                assert refine_signals.branch_estimate(cell, n, how, fill=fill) == pytest.approx(
                    whole, abs=1e-12)


def test_a_missing_sub_game_gives_up_only_when_kept() -> None:
    cell = {"p": [0.5, 0.3, 0.15, 0.05], "sub": [0.4, 0.5, 0.6, None],
            "solved": [1, 1, 1, 1], "leaf": [0.4, 0.5, 0.6, 0.7]}
    assert refine_signals.branch_estimate(cell, 3, "prob", fill=False) is not None
    assert refine_signals.branch_estimate(cell, 4, "prob", fill=False) is None


def test_auc_and_spearman_controls() -> None:
    rng = np.random.default_rng(0)
    target = rng.random(5000)
    label = refine_signals.top_label(target)
    assert label.mean() == pytest.approx(refine_signals.TOP_SHARE, abs=0.01)
    assert refine_signals.auc(target, label) == 1.0
    assert refine_signals.spearman(target, target**3) == pytest.approx(1.0)
    noise = rng.random(5000)
    assert abs(refine_signals.auc(noise, label) - 0.5) < 0.04
    assert abs(refine_signals.spearman(noise, target)) < 0.05
    # partialling a signal out of itself leaves nothing; out of noise leaves it whole
    mixed = target + 0.3 * noise
    assert abs(refine_signals.partial_spearman(mixed, target, mixed)) < 1e-6 or np.isnan(
        refine_signals.partial_spearman(mixed, target, mixed))
    assert refine_signals.partial_spearman(target, target, noise) == pytest.approx(1.0, abs=1e-6)


def _node_row(seed: int, size: int = 6, moved: int = 6) -> dict:
    """A position whose depth-2 values differ from depth 1 on `moved` cells only."""
    rng = np.random.default_rng(seed)
    d1 = rng.uniform(0.2, 0.8, (size, size))
    delta = np.zeros((size, size))
    picks = rng.choice(size * size, moved, replace=False)
    delta.flat[picks] = rng.choice([-0.3, 0.3], moved)
    cells = [
        [
            {"p": [1.0], "sub": [float(d1[i, j] + delta[i, j])], "solved": [1],
             "leaf": [float(d1[i, j])]}
            for j in range(size)
        ]
        for i in range(size)
    ]
    noise = rng.normal(0, 0.01, (size, size))
    return {"index": seed, "d1": d1.tolist(), "d1a": (d1 + noise).tolist(),
            "d1b": (d1 - noise).tolist(), "cells": cells}


def test_budgets_have_their_controls() -> None:
    oracle = []
    null = []
    for seed in range(12):
        node = refine_signals.build_node(_node_row(seed))
        got = refine_signals.analyse_node(node, seed=1)
        # 36 cells: a budget of 64 refines all of them, so every rule is the reference.
        for name in refine_signals.RULES:
            assert got["cells"][(name, 64)]["exploit"] == pytest.approx(0.0, abs=1e-7)
            assert got["cells"][(name, 64)]["value_err"] == pytest.approx(0.0, abs=1e-7)
        # Six cells moved and sixteen can be refined: the oracle finds all of them.
        assert got["cells"][("oracle/gap", 16)]["exploit"] == pytest.approx(0.0, abs=1e-7)
        oracle.append(got["cells"][("oracle/gap", 16)]["exploit"])
        null.append(got["cells"][("random (null)", 16)]["exploit"])
    assert np.mean(null) > np.mean(oracle) + 0.01


def test_the_shipped_loop_runs_on_the_stored_numbers() -> None:
    row = _node_row(4, size=8, moved=0)
    node = refine_signals.build_node(row)
    for restricted in (False, True):
        got, used = refine_signals.shipped_search(node, restricted=restricted, passes=2)
        # Nothing moved, so depth 2 is depth 1: the mixed reading is depth 1's answer, and
        # the restricted one guarantees no more than it (a guarantee, not a game value).
        depth1 = refine_signals.read_mixed(node.d1, node.d2, np.zeros_like(node.refinable))
        if restricted:
            assert got.value <= depth1.value + 1e-9
        else:
            assert got.value == pytest.approx(depth1.value, abs=1e-9)
        assert 1 <= used <= 32
    # The stubs are put back.
    assert search_mod.batched_payoff is not None
    assert search_mod._refined_value.__module__ == "pokeuraou.search"
