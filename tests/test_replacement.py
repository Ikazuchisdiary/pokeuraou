"""The replacement phase after a faint.

This is the half of the game the single-turn tool was allowed to stop at, and therefore
the half that had never been compared with Showdown: the turn harness classified 327
"pending replacement" turns as skipped. Self-play to a win or a loss walks through this
phase constantly, so it gets held to the same standard as a turn -- an equality against
Showdown's own resulting position, with every source of chance pinned.

The threshold is zero. Unlike a turn, a replacement has no unmodelled mechanics left in
it: placement, the party-slot swap, hazards and the switch-in abilities are all things the
resolver already does. A divergence here is a bug, not a documented gap.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import switch_actions_after_faint
from pokeuraou.oracle import ORACLE_JS, TeamSet
from pokeuraou.regulation import Regulation

from . import _diff_replacement_entry as diff
from ._port import replacements_needed, resolve_replacements
from .test_actions import _synthetic_position

pytestmark = pytest.mark.oracle


def test_nothing_is_owed_from_a_fresh_position(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """A position with nobody fainted owes no replacement, or a game loop would stall."""
    pos = _synthetic_position(reg, team_a)
    assert replacements_needed(pos) == ((False, False), (False, False))


def test_a_faint_is_owed_and_can_be_answered(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.hp = 0
    mon.fainted = True
    mon.status = "fnt"

    ours, theirs = replacements_needed(pos)
    assert ours == (True, False)
    assert theirs == (False, False)

    options = switch_actions_after_faint(reg, pos, 0, [True, False])
    assert options, "a faint with a live bench has to offer a replacement"
    result = resolve_replacements(reg, pos, [options[0], _pass(pos, 1)])

    after = result.position.sides[0]
    incoming = after.pokemon[after.active[0]]
    assert not incoming.fainted
    assert incoming.slot == 0, (
        "Showdown swaps party slots on any switch, fainted or not, so the active Pokemon "
        "sits at its own active index"
    )
    # `if (oldActive.fainted) oldActive.status = ''`: the marker is cleared on the way out.
    benched = next(m for m in after.pokemon if m.species == mon.species)
    assert benched.fainted and benched.status is None


def _pass(pos, side_index: int):  # noqa: ANN001, ANN202
    from pokeuraou.actions import PassAction, SideAction

    return SideAction(
        slots=tuple(
            PassAction(slot=slot) for slot in range(len(pos.sides[side_index].active))
        )
    )


@pytest.fixture()
def on_the_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness in tools/ asks Python's replacement phase; here it asks the port's
    (IKA-210). The tool itself moves to the port with IKA-212."""
    monkeypatch.setattr(diff._MODULE, "replacements_needed", replacements_needed)  # noqa: SLF001
    monkeypatch.setattr(diff._MODULE, "resolve_replacements", resolve_replacements)  # noqa: SLF001


#: The port refuses a phase the sampled battles reach (IKA-208): a Throat Chop'd Pokemon
#: (seed 1, and the large sample) and a Disguise holder (seed 2). Strict, so the day the
#: port answers them these come off (IKA-210).
REFUSED = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="the port refuses throatchop / disguise (IKA-208)"
)


@REFUSED
@pytest.mark.parametrize("seed", [1, 2])
def test_replacements_match_showdown(seed: int, on_the_port: None) -> None:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    report = diff.run(battles=10, seed=seed, max_turns=14)
    assert report.compared >= 15, f"only {report.compared} phases compared"
    assert report.flag_mismatches == 0, (
        f"the resolver disagreed with Showdown about whether a replacement is owed "
        f"{report.flag_mismatches} times; a self-play loop would stall or send an "
        f"illegal choice\n{report.render()}"
    )
    assert report.divergence_rate == 0.0, report.render()


@REFUSED
@pytest.mark.slow
def test_replacement_divergence_over_a_large_sample(on_the_port: None) -> None:
    """The number quoted in the README, measured rather than remembered."""
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    compared = matched = flags = 0
    for seed in (1, 2, 3, 4):
        report = diff.run(battles=15, seed=seed, max_turns=14)
        compared += report.compared
        matched += report.matched
        flags += report.flag_mismatches
    rate = 0.0 if not compared else 1 - matched / compared
    print(f"\nreplacement divergence over {compared} phases: {rate * 100:.3f}%")
    assert compared > 100
    assert flags == 0
    assert rate == 0.0
