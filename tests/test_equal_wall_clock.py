"""Each arm's own wall clock, so "is depth worth it" is not answered by the afternoon.

The cost side of the depth-2 question has only ever been quoted by dividing one run's
`s/game` by another run's decisions per game -- and the record already carries the
correction that made necessary: one of those runs had the test suite on the same machine,
so a figure of "about 2.1x" was partly the load. GENERATIONS.md says it in the tool's own
words: *"別の run の壁時間を比べて、ツール自身の1決定あたりの数字を見ていなかった"*.

`GameRecord.search_seconds` puts both arms on one clock inside one game, so the ratio is
read from the same machine, the same minute and the same positions. Two properties make
it a measurement rather than a stopwatch reading:

- **A shared construction is split, not awarded.** When one menu and one solve serve both
  agents, each is charged half. Charging it to side 0 -- the side the code happens to run
  first -- would make every symmetric match report a seat-shaped cost difference that no
  setting caused.
- **It covers the move nodes and says so.** The replacement node has no depth and no
  width; both agents do the same work there, so folding it in would drag every ratio
  toward 1 by an amount that depends on how often somebody fainted.

The equalities below are exact where the arithmetic is exact. The one inequality is the
one the field is for: a side looking a ply further has to be carrying more seconds than
the side that is not, and if it is not, the flag did not reach the search.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.selfplay import play_game
from pokeuraou.teams import (
    load_archetypes,
    load_roster,
    pick_four,
    sample_archetype,
    usable_archetypes,
)

TOOLS = __import__("pathlib").Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    from pokeuraou.priors import find_cached_chaos, load_chaos

    roster = load_roster("rizabanadohido")
    reg, archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    path = find_cached_chaos(reg.meta.format_id)
    if path is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(path, reg)
    usable, _blocked = usable_archetypes(prior, archetypes)
    if not usable:
        pytest.skip("no usable archetype")
    return reg, prior, roster, usable


def a_game(setup, seed: int, **kwargs):  # noqa: ANN001, ANN003, ANN201
    """One real game, narrow enough to be a test and wide enough to have a matrix."""
    reg, prior, roster, usable = setup
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    record = play_game(
        reg,
        rng,
        pick_four(rng, roster.sets, size=4),
        pick_four(rng, sample_archetype(rng, reg, prior, usable[0]), size=4),
        usable[0].id,
        objective=HP_SHARE,
        search_limit=4,
        **kwargs,
    )
    return record, time.perf_counter() - started


def test_one_menu_and_one_solve_are_split_and_not_awarded(setup) -> None:  # noqa: ANN001
    """Identical settings: neither agent spent it alone, so neither is charged for it."""
    record, _elapsed = a_game(setup, seed=4)
    assert record.decisions, "a game with no decision point cannot time one"
    assert record.search_seconds[0] > 0.0
    assert record.search_seconds[0] == record.search_seconds[1], (
        "the same construction served both agents, so the halves have to be equal "
        "exactly -- 'on average' would leave a seat term in every cost comparison"
    )


def test_the_deeper_side_carries_the_extra_seconds(setup) -> None:  # noqa: ANN001
    """The one inequality: depth 2 refines, depth 1 does not, in the same game.

    Both sides pay the same depth-1 solve over the same matrix, so the difference is the
    refine and nothing else. An equality here would mean `--depth` reached the record and
    not the search, which is the shape every measurement defect in this project has had.
    """
    record, elapsed = a_game(setup, seed=4, depth=(2, 1))
    moves = [d for d in record.decisions if d.kind == "move"]
    assert moves, "a game with no move node cannot show the refine"
    assert record.search_seconds[1] > 0.0, "the shallower side still solves its own matrix"
    assert record.search_seconds[0] > record.search_seconds[1], (
        "the side looking a ply further spent no longer than the side that did not, so "
        "the depth pair did not reach `search`"
    )
    assert sum(record.search_seconds) < elapsed, (
        "the move nodes cannot account for more than the whole game; the resolver and "
        "the replacement nodes are in `elapsed` and must not be in here"
    )


def test_the_shallower_side_is_the_same_game_either_way(setup) -> None:  # noqa: ANN001
    """Sanity on the split: the pair is per side, so swapping it swaps the seconds."""
    deep_first, _a = a_game(setup, seed=11, depth=(2, 1))
    deep_second, _b = a_game(setup, seed=11, depth=(1, 2))
    assert deep_first.search_seconds[0] > deep_first.search_seconds[1]
    assert deep_second.search_seconds[1] > deep_second.search_seconds[0]


def test_the_record_carries_both_numbers_out(setup) -> None:  # noqa: ANN001
    """A run whose per-arm clock lives only in a printed line cannot be re-read."""
    record, _elapsed = a_game(setup, seed=7)
    out = record.to_json(objective=HP_SHARE.name, search_limit=4)
    assert len(out["searchSeconds"]) == 2
    assert out["searchSeconds"] == list(record.search_seconds)


def test_the_two_arms_stay_two_objects() -> None:
    """One file named twice is still two leaves, and the clock is why it has to be.

    Aliasing them is the obvious saving -- the same weights rank the same candidates to
    the same menu twice -- and `tools/generation_match.py` already refuses it, because
    five places used to name the tested arm by `leaves[0] is value` and an alias makes
    all five answer True in both seats.

    The clock adds a second reason, and it is the one that matters to a cost comparison.
    Distinct objects make `same_menu` False, so each side builds the menus it searches
    over and `search_seconds` reads what that agent alone would spend on the position:
    the whole menu, plus its own solve. Aliased, each arm would be charged half a menu,
    which is what neither of them would pay in a game it played by itself.
    """
    body = (TOOLS / "generation_match.py").read_text(encoding="utf-8")
    assert "assert baseline is not value" in body, (
        "the two arms may not be aliased: the seat flip and each arm's own wall clock "
        "both rest on them being distinct"
    )
    # Comments out: the module explains the five defects at length, in this spelling.
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    assert "leaves[0] is value" not in code, (
        "an arm identified by object identity is an arm that reads as the tested one in "
        "both seats the moment anything aliases them"
    )


def test_the_match_reports_the_arms_and_not_only_the_game() -> None:
    """`s/game` holds both arms; a per-arm row is what an equal-time claim rests on."""
    body = (TOOLS / "generation_match.py").read_text(encoding="utf-8")
    assert "record.search_seconds[which]" in body, (
        "the tested arm's seconds have to follow the seat swap, or seat 1 reports the "
        "baseline's clock under the tested arm's name"
    )
    assert "record.search_seconds[1 - which]" in body
    assert '"armSeconds"' in body and '"moveDecisions"' in body, (
        "the two numbers and the count they are per have to reach the written row"
    )
