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
  sides of one agent, each is charged half. Charging it to side 0 -- the side the code
  happens to run first -- would make every symmetric match report a seat-shaped cost
  difference that no setting caused. Two agents that hold one leaf object are each
  charged all of it instead, because each would have paid all of it alone and the field
  is what one agent spends; `one_agent` is how the caller says which it has.
- **It covers the move nodes and says so.** The replacement node has no depth and no
  width; both agents do the same work there, so folding it in would drag every ratio
  toward 1 by an amount that depends on how often somebody fainted.

The equalities below are exact where the arithmetic is exact. The one inequality is the
one the field is for: a side looking a ply further has to be carrying more seconds than
the side that is not, and if it is not, the flag did not reach the search.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from pokeuraou import selfplay
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

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


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
    foe_six = sample_archetype(rng, reg, prior, usable[0])
    started = time.perf_counter()
    record = play_game(
        reg,
        rng,
        pick_four(rng, roster.sets, size=4),
        pick_four(rng, foe_six, size=4),
        usable[0].id,
        objective=HP_SHARE,
        search_limit=4,
        sheets=(list(roster.sets), foe_six),
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


def test_the_record_carries_both_numbers_out(setup) -> None:  # noqa: ANN001
    """A run whose per-arm clock lives only in a printed line cannot be re-read."""
    record, _elapsed = a_game(setup, seed=7)
    out = record.to_json(objective=HP_SHARE.name, search_limit=4)
    assert len(out["searchSeconds"]) == 2
    assert out["searchSeconds"] == list(record.search_seconds)


def test_two_agents_holding_one_leaf_are_each_charged_all_of_it(setup, monkeypatch) -> None:  # noqa: ANN001
    """The saving is the machine's; the clock still reports one agent's move.

    One leaf object for both sides is what a match of a configuration against itself now
    builds, and it is worth about half the solves. But a symmetric run is the only way a
    width or a depth can be priced -- the matrix is one, so a per-side knob grows both
    arms at once and the in-run ratio is 0.99 by construction -- and the number it is
    priced by is this field. Split between the two, it reads half of what the agent
    would spend in a game it played alone, and half of every figure recorded before the
    arms could be shared, with the ratio it is usually quoted as intact to hide it.
    """
    # The flag is a division, not a measurement: the two branches add `menu + own` and
    # `(menu + own) / 2` to the same field. Only `play_game` takes it, so the two
    # accountings need two runs -- and two runs of a real clock is the comparison this
    # module's own docstring calls a defect, which is what it was doing here. The earlier
    # bound was `> 1.5x` of a ratio that is 2 by construction, and under the parallel
    # suite it came back 1.4984 and failed three times in five (IKA-74). Holding the
    # clock still makes the assertion the exact one the arithmetic supports.
    def ticking() -> object:
        ticks = itertools.count(1.0, 1.0)
        return lambda: next(ticks)

    monkeypatch.setattr(selfplay, "perf_counter", ticking())
    shared, _a = a_game(setup, seed=4)
    monkeypatch.setattr(selfplay, "perf_counter", ticking())
    apart, _b = a_game(setup, seed=4, one_agent=False)
    # The control for the ratio below. A held clock and a repeated game make "2.0x" cheap
    # to assert and cheap to fake: a counter that drifted between the two runs would push
    # the ratio around on its own, and a factor read off a harness rather than off the
    # flag is the failure this whole module is about. The same call with the flag NOT
    # moved has to come back at exactly 1.0x.
    monkeypatch.setattr(selfplay, "perf_counter", ticking())
    again, _c = a_game(setup, seed=4)
    assert again.search_seconds == shared.search_seconds, (
        "two runs of the same game on the same held clock disagreed, so the ratio below "
        "would be measuring the counter rather than `one_agent`"
    )

    assert shared.search_seconds[0] == shared.search_seconds[1]
    assert apart.search_seconds[0] == apart.search_seconds[1], (
        "two agents over one leaf do the same work as each other, whoever is charged"
    )
    assert [d.kind for d in apart.decisions] == [d.kind for d in shared.decisions], (
        "the flag charges the clock and decides nothing; a different game means it did"
    )
    assert shared.search_seconds[0] > 0.0, (
        "a stopped clock would satisfy the ratio below and measure nothing"
    )
    assert apart.search_seconds[0] == 2.0 * shared.search_seconds[0], (
        "one agent is charged half of a shared construction and two agents are each "
        "charged all of it, so with the same game on the same clock the two runs differ "
        "by exactly two -- anything else means `one_agent` did not reach the accounting"
    )


def test_the_pool_match_says_its_two_arms_are_two_agents() -> None:
    """A match holds two agents whether or not they hold one leaf object.

    Left to default, a symmetric pricing run reports half the cost it is measuring
    (`generation_match.py`, which held this, was removed in IKA-460 with M-B).
    """
    body = (ROOT / "src" / "pokeuraou" / "poolplay.py").read_text(encoding="utf-8")
    assert "one_agent=False" in body
