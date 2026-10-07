"""IKA-440: an arm's `bench_oracle` makes its bench belief the opponent's drawn four.

A measuring arm, off by default: it bounds what any better bench belief could be worth.
What these tests hold:

- **it fires**: every belief the oracle arm builds about the side across is one completion
  with weight 1, and that completion is the side's real unseen members;
- **it changes the belief**: the same game without the flag builds beliefs that are not all
  point masses (so the first check could have failed);
- **only the belief**: both sides draw the same four with and without the flag;
- **off ships**: a default arm has no oracle and reports a solved belief.
"""

from __future__ import annotations

import pytest

from pokeuraou import selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.regulation import to_id

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _arm(pool, *, oracle: bool) -> PoolArm:  # noqa: ANN001
    return PoolArm(
        name="a", evaluate=_stub, solver=SolvedSelections(pool.reg, pool.teams, _stub),
        limit=4, rank_by_leaf=True, rank_fill="refs2", bench_oracle=oracle,
    )


def _play(pool, monkeypatch, oracle: bool, which: int):  # noqa: ANN001, ANN202
    """Plays one game with the tested arm at `which`; returns the beliefs built per priced
    side, each as (weights, the priced side's real identities), and `sides`."""
    calls: dict[int, list] = {0: [], 1: []}
    real = selfplay._bench_weights

    def logged(bench_prior, side, pos, seen, record, leads=None):  # noqa: ANN001, ANN202
        got = real(bench_prior, side, pos, seen, record, leads)
        real_ids = {to_id(m.base_species) for m in pos.sides[side].pokemon}
        real_ids |= {to_id(m.species) for m in pos.sides[side].pokemon}
        calls[side].append((got, real_ids))
        return got

    with monkeypatch.context() as patch:
        patch.setattr(selfplay, "_bench_weights", logged)
        arms = (_arm(pool, oracle=oracle), _arm(pool, oracle=False))
        _, sides = pool_match_game(
            pool.reg, pool, arms, seed=440, game_index=0, which=which,
            hide_bench=True, max_turns=4, epsilon=0.25, temperature=0.5,
        )
    return calls, sides


@pytest.mark.parametrize("which", [0, 1])
def test_the_oracle_believes_the_drawn_four_and_draws_as_before(pool, monkeypatch, which) -> None:  # noqa: ANN001
    priced = 1 - which  # the side whose bench the oracle arm (at `which`) believes about
    on, sides_on = _play(pool, monkeypatch, oracle=True, which=which)
    off, sides_off = _play(pool, monkeypatch, oracle=False, which=which)

    assert sides_on["beliefs"][which] == "oracle"
    assert sides_off["beliefs"][which] == "solved"
    assert sides_on["picks"] == sides_off["picks"], "the flag must not move any selection"

    hidden = [(w, ids) for w, ids in on[priced] if w and () not in w]
    assert hidden, "the oracle side never believed about a hidden bench: nothing was tested"
    for weights, real_ids in hidden:
        assert len(weights) == 1, weights
        (key, weight), = weights.items()
        assert weight == pytest.approx(1.0)
        assert {to_id(s) for s in key} <= real_ids, (key, real_ids)

    spread = [w for w, _ in off[priced] if w and () not in w and len(w) > 1]
    assert spread, (
        "without the flag every belief was already a point mass, so the check above could "
        "not have failed"
    )


def test_a_default_arm_has_no_oracle(pool) -> None:  # noqa: ANN001
    arm = PoolArm(name="a", evaluate=_stub, solver=None, limit=4, rank_by_leaf=False)
    assert arm.bench_oracle is False
