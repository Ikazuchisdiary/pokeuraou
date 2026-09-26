"""IKA-323: a leaf-ranked root menu can be built without the cover.

What these tests hold:

- **the label**: ``-nocover`` rides on a rank-fill label (``refs2-nocover``,
  ``refs2-fast-nocover``) and says only whether the menu is covered;
- **narrow**: ``cover=False`` keeps the top ``limit`` by the ranking and nothing else, and
  every option that leaves off is in ``uncovered`` (nothing is dropped quietly); the damage
  score alone may not skip the cover;
- **the root menu**: `_menus` builds a ``-nocover`` agent's menus that way and counts them,
  and the default is the covered menu, uncounted;
- **it follows its arm into either seat**: in a pool match only the ``-nocover`` arm's
  menus are counted, in both seats, and identical covered arms count nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import narrow as narrowing
from pokeuraou import selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow, slot_options
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.search import DEFAULT_REFERENCES, parse_rank_fill, rank_fill_covers

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture()
def pos(pool):  # noqa: ANN001, ANN201
    team = list(pool.teams[0].sets)
    return selfplay.position_from_sets(
        pool.reg, team[:4], team[:4], rng=np.random.default_rng(0)
    )


def _by_damage(_pool, scored):  # noqa: ANN001, ANN202
    """A ranking that is the damage score: it puts every Protect and switch last."""
    return [c.score for c in scored]


def test_the_label_says_whether_the_menu_is_covered() -> None:
    assert parse_rank_fill("refs2-nocover") == (2, False)
    assert parse_rank_fill("refs1-fast-nocover") == (1, True)
    assert rank_fill_covers("refs2") and rank_fill_covers("refs2-fast")
    assert not rank_fill_covers("refs2-nocover")
    assert not rank_fill_covers("refs2-fast-nocover")
    for bad in ("refs2-nocover-fast", "nocover", "refs2-cover"):
        with pytest.raises(ValueError, match="rank fill"):
            rank_fill_covers(bad)


def test_without_the_cover_the_menu_is_the_ranking_top_and_says_what_it_left(
    pool, pos  # noqa: ANN001
) -> None:
    reg = pool.reg
    whole = narrow(reg, pos, 0, limit=10_000, rank=_by_damage)
    assert whole.considered > 4
    bare = narrow(reg, pos, 0, limit=4, rank=_by_damage, cover=False)
    # The top four of the whole ranking, in its order.
    assert bare.actions == whole.actions[:4]
    assert bare.for_coverage == 0 and bare.for_score == 4
    # Nothing is dropped quietly: every option of the pool is on the menu or reported.
    every = slot_options(reg, whole.actions)
    on_menu = slot_options(reg, bare.actions)
    assert set(bare.uncovered) == set(every.values()) - set(on_menu.values())
    assert bare.uncovered and len(bare.uncovered_options) == len(bare.uncovered)
    # Positive control: the covered menu of the same ranking is another menu.
    covered = narrow(reg, pos, 0, limit=4, rank=_by_damage)
    assert covered.for_coverage > 0
    assert covered.actions != bare.actions
    assert len(covered.uncovered) < len(bare.uncovered)
    # A pool within the width is the whole pool either way.
    assert narrow(reg, pos, 0, limit=10_000, rank=_by_damage, cover=False).actions == (
        whole.actions
    )


def test_the_damage_score_alone_keeps_the_cover(pool, pos) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="cover"):
        narrow(pool.reg, pos, 0, limit=4, cover=False)


def _ranking_by_order(monkeypatch):  # noqa: ANN001, ANN202
    """Stub `leaf_ranking`: the later a candidate in the pool, the higher it ranks."""

    def stub(reg, at, side, evaluate, *, budget, references=DEFAULT_REFERENCES):  # noqa: ANN001, ANN202, ARG001
        return lambda pool, _scored=None: np.arange(len(pool), dtype=float)

    monkeypatch.setattr(selfplay, "leaf_ranking", stub)


def test_the_root_menu_is_built_without_the_cover_only_when_asked(
    pool, pos, monkeypatch  # noqa: ANN001
) -> None:
    _ranking_by_order(monkeypatch)
    reg = pool.reg
    budget = Budget.matrix()
    ranking = lambda pool_, _scored=None: np.arange(len(pool_), dtype=float)  # noqa: E731
    expect = {
        cover: tuple(
            narrow(reg, pos, side, limit=4, rank=ranking, cover=cover).actions
            for side in (0, 1)
        )
        for cover in (True, False)
    }
    assert expect[True] != expect[False]  # the flag can be seen in the menus

    before = {k: dict(v) for k, v in narrowing.COVERLESS.items()}
    assert selfplay._menus(reg, pos, (4, 4), _stub, budget, True) == expect[True]
    assert before == narrowing.COVERLESS  # the default counts nothing

    menus = selfplay._menus(reg, pos, (4, 4), _stub, budget, True, rank_fill="refs2-nocover")
    assert menus == expect[False]
    got = dict(narrowing.COVERLESS["refs2-nocover"])
    was = before.get("refs2-nocover", {"menus": 0, "dropping": 0, "dropped": 0})
    assert got["menus"] - was["menus"] == 2
    assert got["dropping"] - was["dropping"] >= 1

    # The wider menus of the root's double oracle are the ranking's top too, uncounted.
    wider: dict = {}
    menus = selfplay._menus(
        reg, pos, (4, 4), _stub, budget, True, rank_fill="refs2-nocover",
        wide=[6], wider=wider,
    )
    assert menus == expect[False]
    assert wider[6] == tuple(
        narrow(reg, pos, side, limit=6, rank=ranking, cover=False).actions for side in (0, 1)
    )
    assert narrowing.COVERLESS["refs2-nocover"]["menus"] - got["menus"] == 2

    # The damage ranking has no leaf to trust, so the label changes nothing there.
    damage = selfplay._menus(reg, pos, (4, 4), _stub, budget, False, rank_fill="refs2-nocover")
    assert damage == tuple(narrow(reg, pos, side, limit=4).actions for side in (0, 1))


def _arm(fill: str, solver) -> PoolArm:  # noqa: ANN001
    return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2, rank_by_leaf=True,
                   rank_fill=fill)


def _counted() -> dict[str, int]:
    return {k: v["menus"] for k, v in narrowing.COVERLESS.items()}


def test_the_uncovered_menu_follows_its_arm_into_either_seat(pool, monkeypatch) -> None:  # noqa: ANN001
    _ranking_by_order(monkeypatch)
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm("refs2-nocover", solver), _arm("refs2", solver)
    for which in (0, 1):
        before = _counted()
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=323, game_index=0, which=which,
            hide_bench=True, max_turns=2,
        )
        assert record.rank_fill[which] == "refs2-nocover"
        assert record.rank_fill[1 - which] == "refs2"
        moves = sum(d.kind == "move" for d in record.decisions)
        assert moves > 0
        after = _counted()
        # One construction per move decision by the uncovered arm, both sides' menus.
        assert after["refs2-nocover"] - before.get("refs2-nocover", 0) == 2 * moves
        assert after.get("refs2", 0) == before.get("refs2", 0)

    # Null: identical covered arms build nothing without the cover.
    before = _counted()
    same = _arm("refs2", solver)
    pool_match_game(pool.reg, pool, (same, same), seed=323, game_index=0, which=0,
                    hide_bench=True, max_turns=2)
    assert _counted() == before
