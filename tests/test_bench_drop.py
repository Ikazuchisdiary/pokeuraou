"""IKA-283: an agent's belief may leave out the completions its bench prior barely weighs.

What these tests hold:

- **the label**: ``none``, ``w<P>`` and ``m<P>`` parse, anything else stops;
- **what is dropped**: under P% (``w``) or beyond the heaviest P% of the mass (``m``); the
  heaviest completion always stays, the order of the rest is kept, and what is left sums
  to one. A rule with nothing to drop hands back the very list, so its game is the
  default's bit for bit;
- **it reaches both halves of the search**: the Bayesian solve over what is left is the
  full node's matrices of the kept completions, solved under the renormalised weights, and
  the menu is ranked from the same (heaviest) completion as without the drop;
- **each agent drops from its own belief**: side `s`'s completions are side `1 - s`'s
  belief, so it is side `1 - s`'s rule that drops from them, in either seat of a match;
- **it is recorded only when it differs**, and names another agent only under a hidden
  bench.

The positive control for the renormalisation is `_assert_a_belief`: a drop that forgets
to renormalise fails it (`test_forgetting_to_renormalise_is_caught`), and the same edit
made in `hidden.drop_light` itself fails this file (records/IKA-283.md).
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from pokeuraou import rustnode, selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve_bayesian
from pokeuraou.hidden import (
    DEFAULT_BENCH_DROP,
    Completion,
    completions,
    drop_light,
    parse_bench_drop,
)
from pokeuraou.narrow import narrow
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.provenance import LEGACY_BENCH_DROP, agent_name, provenance
from pokeuraou.regulation import load_regulation
from pokeuraou.search import belief_solve
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from .test_beliefnode import _Leaf
from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    return reg, sheet, position_from_sets(reg, sheet[:4], sheet[:4])


def _worlds(weights: list[float]) -> list[Completion]:
    """Completions with these weights and nothing else; positions are not read here."""
    return [
        Completion(position=None, species=(f"s{i}", f"t{i}"), slots=(2, 3), weight=w)  # type: ignore[arg-type]
        for i, w in enumerate(weights)
    ]


def _assert_a_belief(kept: list[Completion], full: list[Completion], label: str) -> None:
    """What any drop must hand the search: a subset in order, the heaviest in it, and
    weights that are the full belief's conditioned on the subset."""
    rule, fraction = parse_bench_drop(label)
    names = [c.species for c in full]
    at = [names.index(c.species) for c in kept]
    assert at == sorted(at), "the enumeration order moved"
    heaviest = max(range(len(full)), key=lambda i: full[i].weight)
    assert heaviest in at, "the heaviest completion was dropped"
    assert sum(c.weight for c in kept) == pytest.approx(1.0, abs=1e-12), "not renormalised"
    mass = sum(full[i].weight for i in at)
    for c, i in zip(kept, at, strict=True):
        assert c.weight == pytest.approx(full[i].weight / mass, rel=1e-12)
    if rule == "w":
        assert all(full[i].weight >= fraction for i in at if i != heaviest)


def test_the_label_parses_and_a_bad_one_stops() -> None:
    assert DEFAULT_BENCH_DROP == LEGACY_BENCH_DROP == "none"
    assert parse_bench_drop("none") == ("none", 0.0)
    assert parse_bench_drop("w5") == ("w", 0.05)
    assert parse_bench_drop("w10") == ("w", 0.10)
    assert parse_bench_drop("w0.5") == ("w", 0.005)
    assert parse_bench_drop("m95") == ("m", 0.95)
    for bad in ("", "w", "m", "w0", "w100", "m100", "x5", "w-1", "5", "W5", "none5"):
        with pytest.raises(ValueError, match="bench drop"):
            parse_bench_drop(bad)


def test_a_weight_bar_drops_the_light_and_renormalises() -> None:
    full = _worlds([0.02, 0.9, 0.05, 0.03])
    kept = drop_light(full, "w5")
    assert [c.species for c in kept] == [full[1].species, full[2].species]
    _assert_a_belief(kept, full, "w5")
    assert kept[0].weight == 0.9 / (0.9 + 0.05)
    # The completions are the same worlds, only reweighted.
    assert kept[0] == replace(full[1], weight=0.9 / (0.9 + 0.05))


def test_a_mass_bar_keeps_the_heaviest_until_it_is_reached() -> None:
    full = _worlds([0.06, 0.6, 0.04, 0.3])
    kept = drop_light(full, "m95")
    # 0.6 + 0.3 = 0.90 < 0.95, so 0.06 joins; 0.04 is past the bar.
    assert [c.species for c in kept] == [full[0].species, full[1].species, full[3].species]
    _assert_a_belief(kept, full, "m95")
    assert drop_light(full, "m99") is full
    assert [c.species for c in drop_light(full, "m50")] == [full[1].species]


def test_the_heaviest_stays_whatever_the_bar() -> None:
    # Fifteen worlds at 1/15 each: every one is under 10%, and the first of the equal
    # maxima -- the one the menu ranks from -- is what is left.
    full = _worlds([1.0 / 15] * 15)
    kept = drop_light(full, "w10")
    assert [c.species for c in kept] == [full[0].species]
    assert kept[0].weight == 1.0
    _assert_a_belief(kept, full, "w10")
    # Zero-weight worlds (a pair the book never brings) go under any bar.
    zeros = _worlds([0.0, 0.0, 1.0, 0.0])
    assert [c.species for c in drop_light(zeros, "w0.5")] == [zeros[2].species]


def test_nothing_to_drop_is_the_same_list() -> None:
    full = _worlds([0.2, 0.3, 0.5])
    assert drop_light(full, "none") is full
    assert drop_light(full, "w5") is full
    assert drop_light(full, "m99") is full
    exact = [Completion(position=None, species=(), slots=(), weight=1.0, exact=True)]  # type: ignore[arg-type]
    assert drop_light(exact, "w10") is exact


def test_forgetting_to_renormalise_is_caught() -> None:
    """Positive control: a drop that keeps the right worlds at their old weights fails."""
    full = _worlds([0.02, 0.9, 0.05, 0.03])

    def forgetful(items, label):  # noqa: ANN001, ANN202
        rule, fraction = parse_bench_drop(label)
        return [c for c in items if c.weight >= fraction]

    with pytest.raises(AssertionError, match="not renormalised"):
        _assert_a_belief(forgetful(full, "w5"), full, "w5")


def test_each_side_is_dropped_by_the_agent_blind_to_it() -> None:
    light, heavy = _worlds([0.02, 0.98]), _worlds([0.03, 0.97])
    spreads = {0: light, 1: heavy}
    # Null: neither agent drops, the dict itself.
    assert selfplay._believed(spreads, ("none", "none")) is spreads
    # Side 0's agent drops from side 1's completions, side 1's from side 0's.
    got = selfplay._believed(spreads, ("w5", "none"))
    assert got[0] is light and len(got[1]) == 1
    got = selfplay._believed(spreads, ("none", "w5"))
    assert len(got[0]) == 1 and got[1] is heavy


def _skewed(reg, position, sheet, side):  # noqa: ANN001, ANN202
    """`side`'s six completions with one at 0.9 and the rest light, two of them under 5%."""
    plain = completions(reg, position, side, sheet)
    keys = [tuple(sorted(c.species)) for c in plain]
    weights = dict(zip(keys, [0.01, 0.02, 0.9, 0.03, 0.02, 0.02], strict=True))
    made = completions(reg, position, side, sheet, weights=weights)
    assert len(made) == 6
    return made


def test_the_solve_over_what_is_left_is_the_full_node_restricted(setup) -> None:  # noqa: ANN001
    """Both sides' Bayesian solves over the dropped belief are the full node's matrices of
    the kept completions under the renormalised weights -- to the bit, with a row-wise
    leaf -- and the node really builds fewer matrices."""
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    if not rustnode.available():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}")
    from pokeuraou.beliefnode import belief_payoffs

    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=6).actions
    theirs = narrow(reg, position, 1, limit=6).actions
    budget = Budget.matrix()
    leaf = _Leaf(Encoder(reg))
    full = {side: _skewed(reg, position, sheet, side) for side in (0, 1)}
    dropped = selfplay._believed(full, ("m90", "w3"))
    # Side 1's completions are side 0's belief (m90: 0.9 alone reaches it), and side 0's
    # are side 1's (w3: 0.9 and 0.03 stay).
    assert len(dropped[1]) == 1 and len(dropped[0]) == 2
    for side in (0, 1):
        _assert_a_belief(dropped[side], full[side], ("w3", "m90")[side])

    node = belief_payoffs(reg, position, ours, theirs, leaf, budget=budget, spreads=full)
    small = belief_payoffs(reg, position, ours, theirs, leaf, budget=budget, spreads=dropped)
    assert [len(small.matrices[s]) for s in (0, 1)] == [1, 2]
    got = belief_solve(reg, position, ours, theirs, dropped, {0: leaf, 1: leaf}, budget=budget)
    for side in (0, 1):
        names = [c.species for c in full[1 - side]]
        at = [names.index(c.species) for c in dropped[1 - side]]
        kept = [node.matrices[side][k] for k in at]
        for mine, theirs_ in zip(small.matrices[side], kept, strict=True):
            assert np.array_equal(mine, theirs_), f"side {side}"
        weights = np.asarray([c.weight for c in dropped[1 - side]])
        want = solve_bayesian([m if side == 0 else -m.T for m in kept], weights)
        assert np.array_equal(got[side].strategy, want.row_strategy), f"side {side}"
        assert got[side].value == want.value
        assert got[side].classes == len(at)
    # Positive control: the full belief is a different game here.
    whole = belief_solve(reg, position, ours, theirs, full, {0: leaf, 1: leaf}, budget=budget)
    assert whole[0].classes == 6 and whole[0].value != got[0].value


def test_the_menu_ranks_from_the_same_completion(setup, monkeypatch) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    full = {side: _skewed(reg, position, sheet, side) for side in (0, 1)}
    asked: list[tuple[int, tuple]] = []

    def stub(reg, at, side, evaluate, *, budget, references=2):  # noqa: ANN001, ANN202, ARG001
        asked.append((side, tuple(m.species for m in at.sides[1 - side].pokemon)))
        return lambda pool, _scored=None: np.zeros(len(pool))

    monkeypatch.setattr(selfplay, "leaf_ranking", stub)
    menus, views = [], []
    for drops in (("none", "none"), ("w5", "w5")):
        used: dict = {}
        asked.clear()
        spreads = selfplay._believed(full, drops)
        menus.append(selfplay._menus(
            reg, position, (4, 4), _stub, Budget.matrix(), True, spreads=spreads, used=used,
        ))
        views.append((list(asked), {s: used[s][1] for s in used}))
    assert menus[0] == menus[1]
    assert views[0] == views[1]


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _arm(drop: str, solver) -> PoolArm:  # noqa: ANN001
    return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2, rank_by_leaf=False,
                   bench_drop=drop)


def _counting(monkeypatch):  # noqa: ANN001, ANN202
    """Records the completions each `belief_solve` was handed, per side."""
    seen: list[tuple[int, int]] = []
    real = selfplay.belief_solve

    def spy(reg, pos, ours, theirs, spreads, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        seen.append((len(spreads[0]), len(spreads[1])))
        return real(reg, pos, ours, theirs, spreads, *args, **kwargs)

    monkeypatch.setattr(selfplay, "belief_solve", spy)
    return seen


def test_the_drop_follows_its_arm_into_either_seat(pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm("w20", solver), _arm("none", solver)
    for which in (0, 1):
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=283, game_index=0, which=which,
            hide_bench=True, max_turns=2,
        )
        assert record.bench_drop[which] == "w20"
        assert record.bench_drop[1 - which] == "none"
        assert sides["bench_drops"][which] == "w20"
        payload = record.to_json(objective="value:arm", search_limit=(2, 2))
        assert payload["benchDrop"] == record.bench_drop

    # Null: identical default arms record nothing new.
    same = _arm("none", solver)
    record, _ = pool_match_game(
        pool.reg, pool, (same, same), seed=283, game_index=0, which=0,
        hide_bench=True, max_turns=2,
    )
    assert "benchDrop" not in record.to_json(objective="value:arm", search_limit=(2, 2))


def _game(setup, drop):  # noqa: ANN001, ANN202
    reg, sheet, _position = setup
    return selfplay.play_game(
        reg, np.random.default_rng(283), sheet[:4], sheet[:4], "test",
        search_limit=3, max_turns=3, sheets=(sheet, sheet), bench_drop=drop,
    )


def _without_drop(record) -> dict:  # noqa: ANN001
    payload = record.to_json(objective="hp-share", search_limit=3)
    payload.pop("benchDrop", None)
    payload.pop("searchSeconds", None)
    payload.pop("engine", None)
    return payload


def test_a_bar_nothing_falls_under_plays_the_default_game(setup, monkeypatch) -> None:  # noqa: ANN001
    """Without a prior every completion weighs 1/6, which clears 5%: the same game."""
    seen = _counting(monkeypatch)
    base = _game(setup, "none")
    base_seen = list(seen)
    seen.clear()
    same = _game(setup, "w5")
    assert seen == base_seen and base_seen[0] == (6, 6)
    assert _without_drop(same) == _without_drop(base)
    assert "benchDrop" not in base.to_json(objective="hp-share", search_limit=3)
    assert same.to_json(objective="hp-share", search_limit=3)["benchDrop"] == ["w5", "w5"]
    # Positive control: 1/6 is under 20%, so each belief keeps its heaviest alone.
    seen.clear()
    _game(setup, "w20")
    assert seen[0] == (1, 1)


def test_the_drop_is_recorded_only_when_it_differs() -> None:
    base = dict(seat="s", leaves=("m", "m"), limits=(12, 12), rankings=("leaf", "leaf"),
                information=("hidden-bench", "hidden-bench"))
    old = provenance("pool-match", **base)
    assert "benchDrops" not in old
    assert provenance("pool-match", **base, bench_drops=(LEGACY_BENCH_DROP,) * 2) == old
    new = provenance("pool-match", **base, bench_drops=("w5", LEGACY_BENCH_DROP))
    assert new["benchDrops"] == ["w5", LEGACY_BENCH_DROP]
    assert agent_name(new, 0) == agent_name(old, 0) + "/benchdrop:w5"
    assert agent_name(new, 1) == agent_name(old, 1)
    # The open game has no completions, so the label does not make it another agent.
    open_ = provenance("pool-match", **{**base, "information": ("open", "open")},
                       bench_drops=("w5", "w5"))
    assert "/benchdrop" not in agent_name(open_, 0)
