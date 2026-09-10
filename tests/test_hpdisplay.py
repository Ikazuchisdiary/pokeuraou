"""The opponent's HP bar, checked against Showdown's own output.

The rule matters because it is the only view of the opponent's HP a player ever gets, and
Champions does not use the one every other modern generation uses: it *floors* where they
round up. A point estimate taken from a percentage is off by a point or two, which is the
margin a KO turns on, so the inversion has to be right.

Showdown emits both views for every HP change -- the secret ``hp/maxhp`` line and the
shared ``percent/100`` line, paired by ``|split|`` -- which makes this a differential test
rather than a restatement of the source: the pairs come from real battles.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.hpdisplay import (
    HpDisplayError,
    band,
    displayed_colour,
    displayed_percent,
    uses_floor_display,
)
from pokeuraou.oracle import ORACLE_JS, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team
from pokeuraou.regulation import Regulation

FORMAT_ID = "gen9championsvgc2026regmc"


def test_the_regulation_is_on_the_flooring_mod(reg: Regulation) -> None:
    assert uses_floor_display(reg), (
        "the display rule is read from the dumped mod id; a regulation on another mod "
        "must not silently inherit Champions' rule"
    )


def test_a_living_pokemon_is_never_shown_as_zero() -> None:
    """`|| 1` in the source: 1 HP out of 400 floors to 0% and is shown as 1%."""
    assert displayed_percent(1, 400) == 1
    assert displayed_percent(3, 400) == 1
    assert displayed_percent(0, 400) == 0


def test_it_floors_rather_than_rounding_up() -> None:
    """The difference from every other modern generation, and the reason for this module."""
    # 99/100 of the way there still shows as 99, and one point short of half shows 49.
    assert displayed_percent(137, 200) == 68  # 68.5 -> 68, not 69
    assert displayed_percent(99, 100) == 99
    assert displayed_percent(100, 100) == 100
    assert displayed_percent(1, 2) == 50


def test_only_full_hp_shows_a_hundred() -> None:
    for maxhp in (100, 137, 250):
        assert displayed_percent(maxhp, maxhp) == 100
        assert displayed_percent(maxhp - 1, maxhp) < 100


def test_the_band_inverts_the_display_exactly() -> None:
    """Every HP in the band displays as the band's percentage, and none outside it does."""
    for maxhp in (100, 137, 160, 207, 250, 331):
        for hp in range(1, maxhp + 1):
            percent = displayed_percent(hp, maxhp)
            got = band(percent, maxhp)
            assert got.low <= hp <= got.high
            for candidate in got.values:
                assert displayed_percent(candidate, maxhp) == percent


def test_the_band_is_a_few_hp_wide_not_one() -> None:
    """Which is the whole point: a percentage is a set, not a number."""
    widths = [band(displayed_percent(hp, 250), 250).width for hp in range(2, 250)]
    assert max(widths) >= 2, "a 250 HP Pokemon cannot have a 1-HP-wide band everywhere"


def test_the_bar_colour_narrows_the_band() -> None:
    """At 20% and 50% the colour is free information, so it has to be usable."""
    maxhp = 250
    wide = band(50, maxhp)
    assert wide.width > 1
    above = band(50, maxhp, colour="g")
    at_or_below = band(50, maxhp, colour="y")
    assert above.width + at_or_below.width == wide.width
    assert all(hp * 2 > maxhp for hp in above.values)
    assert all(hp * 2 <= maxhp for hp in at_or_below.values)

    for hp in wide.values:
        assert displayed_colour(hp, maxhp) in ("g", "y")


def test_a_colour_where_there_is_none_is_refused() -> None:
    with pytest.raises(HpDisplayError, match="means nothing"):
        band(62, 250, colour="g")


def test_an_impossible_observation_is_refused_not_clamped() -> None:
    """A percentage no HP can produce is evidence against the spread, not an error to hide.

    Milestone 2's update wants exactly this: a particle whose max HP cannot produce the
    observed bar is ruled out.
    """
    with pytest.raises(HpDisplayError, match="no HP out of"):
        band(99, 3)


@pytest.mark.oracle
def test_it_matches_showdown_on_real_battles(reg: Regulation) -> None:
    """The differential check: Showdown emits both views, so compare them.

    ``|split|SIDE`` is followed by the secret line and then the shared line, so each
    triple gives one (hp, maxhp) and the percentage the opponent would have seen.
    """
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(chaos, reg)
    rng = np.random.default_rng(3)

    compared = 0
    with Oracle() as oracle:
        for _ in range(4):
            teams = [
                [TeamSet.from_json(s.to_team_set_json(reg)) for s in sample_team(rng, reg, prior)]
                for _ in range(2)
            ]
            handle = oracle.create(
                FORMAT_ID, teams[0], teams[1],
                seed=tuple(int(rng.integers(1, 60000)) for _ in range(4)),  # type: ignore[arg-type]
                policy=RandomnessPolicy(damage_roll=8, secondary=False),
            )
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])
            for _turn in range(10):
                if handle.ended:
                    break
                choices = []
                for side_index in range(2):
                    request = handle.requests[side_index]
                    if not request or request.get("wait"):
                        choices.append(None)
                    else:
                        choices.append("default")
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    break
                compared += _compare_log(handle.log)
            handle.close()

    assert compared >= 20, f"only {compared} HP displays compared"


def _compare_log(log: list[str]) -> int:
    compared = 0
    for index, line in enumerate(log):
        if not line.startswith("|split|") or index + 2 >= len(log):
            continue
        secret, shared = log[index + 1], log[index + 2]
        secret_hp = _health(secret)
        shared_hp = _health(shared)
        if secret_hp is None or shared_hp is None:
            continue
        hp, maxhp = secret_hp
        percent, scale = shared_hp
        if scale != 100 or maxhp == 100:
            # A Pokemon whose max HP really is 100 makes the two views identical, so it
            # cannot tell a right rule from a wrong one.
            continue
        assert displayed_percent(hp, maxhp) == percent, (
            f"{hp}/{maxhp} displays as {percent}% in Showdown but "
            f"{displayed_percent(hp, maxhp)}% here"
        )
        compared += 1
    return compared


def _health(line: str) -> tuple[int, int] | None:
    """The (current, max) pair from the last field of a protocol line, if it has one."""
    parts = line.split("|")
    for field in reversed(parts):
        text = field.strip()
        if "/" not in text:
            continue
        left, _, right = text.partition("/")
        right = "".join(ch for ch in right if ch.isdigit())
        if not left.isdigit() or not right:
            continue
        return int(left), int(right)
    return None
