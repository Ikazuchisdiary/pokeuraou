"""Which half of an interrupted turn the log reproduces.

A self-switching move stops the turn for a replacement choice, and `Budget.exact()`
enumerates the damage rolls, so an attack that lands *before* the interrupt splits the
pause into hundreds of suspensions. The record names one of them: the `selfswitch`
decision's position is the state at the pause, exactly as generation sampled it.

The renderer used to take `suspended[0]` and carry on. Index 0 is the outcome where every
roll came out maximum, so the log printed the hardest possible version of the turn and
then said it could not match the record -- which read as the two halves of the project
ordering a resumed turn differently (TODO F1). The order was never the problem: measured
over 200 recorded games, the pre-interrupt order agreed every time and index 0 was the
wrong suspension in 66 of 105 interrupted turns.

So what is pinned here is the matching, in the shape the bug was found in: Stomping
Tantrum in front of Parting Shot, where the record's roll and the top of the list are 50
HP apart.
"""

from __future__ import annotations

import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.resolve import Budget, resolve_turn, resume_alternatives
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._harness import load_tool

#: Garchomp's Stomping Tantrum at foe 1, Incineroar's Parting Shot at foe 1.
OURS = "move 3 1, move 3 1"
#: Sludge Bomb and Quick Attack back, so the turn has damage on both sides of the pause.
THEIRS = "move 2 1, move 2 1"


@pytest.fixture(scope="module")
def tool():  # noqa: ANN201
    return load_tool("show_game")


@pytest.fixture(scope="module")
def interrupted():  # noqa: ANN201
    """The turn, its suspensions, and a roster that produces them."""
    roster = load_roster("rizabanadohido")
    reg = roster.reg
    register_mega_stones(reg)
    # Garchomp and Incineroar lead, Charizard and Sylveon on the bench: Parting Shot needs
    # somewhere to go or the resolver never marks the self-switch at all.
    own = [roster.sets[2], roster.sets[5], roster.sets[0], roster.sets[3]]
    foe = [roster.sets[1], roster.sets[3], roster.sets[4], roster.sets[0]]
    pos = position_from_sets(reg, own, foe)
    ours = {a.to_choice(): a for a in narrow(reg, pos, 0, limit=512).actions}
    theirs = {a.to_choice(): a for a in narrow(reg, pos, 1, limit=512).actions}
    result = resolve_turn(reg, pos, [ours[OURS], theirs[THEIRS]], budget=Budget.exact())
    return reg, pos, result


def record_of(reg, pos, played, action, ending):  # noqa: ANN001, ANN201
    """Three decisions in the shape `selfplay` writes them: move, selfswitch, next move."""

    def node(turn: int, kind: str, position, chosen: str | None) -> dict:  # noqa: ANN001
        return {
            "turn": turn,
            "kind": kind,
            "position": position,
            "ownActions": [chosen] if chosen else [],
            "ownPolicy": [1.0] if chosen else [],
            "foeActions": [THEIRS if kind == "move" else "pass"] if chosen else [],
            "foePolicy": [1.0] if chosen else [],
            "searchValue": 0.5,
            "ownChosen": chosen,
            "foeChosen": (THEIRS if kind == "move" else "pass") if chosen else None,
        }

    return [
        node(1, "move", pos.to_json(), OURS),
        node(1, "selfswitch", played.position.to_json(), action.to_choice()),
        node(2, "move", ending.position.to_json(), None),
    ]


def test_the_turn_has_many_pauses_to_choose_between(interrupted) -> None:  # noqa: ANN001
    """Without this the test below would pass on a turn that offers no wrong answer."""
    _reg, _pos, result = interrupted
    assert not result.branches, "the turn has to stop at the interrupt"
    assert len(result.suspended) > 100, (
        f"only {len(result.suspended)} suspensions: the damage rolls before the interrupt "
        "are what makes picking one a decision"
    )
    first = [e for e in result.suspended[0].events if "stompingtantrum" in e]
    last = [e for e in result.suspended[-1].events if "stompingtantrum" in e]
    assert first and last and first != last, (
        f"the ends of the list must disagree about the damage: {first} vs {last}"
    )


def test_the_log_reproduces_the_recorded_pause_not_the_first_one(  # noqa: ANN001
    tool, interrupted
) -> None:
    """The record names a pause; the log has to render that one.

    Asserted on the damage line rather than on the absence of a warning alone, because a
    log that prints the wrong turn and admits it is still printing the wrong turn.
    """
    reg, pos, result = interrupted
    # The last suspension is the minimum roll -- as far from `suspended[0]` as the list
    # goes, which is where a wrong pick is visible.
    played = result.suspended[-1]
    chooser, alternatives = resume_alternatives(reg, played)
    assert chooser == 0, "Incineroar owes the replacement"
    action, resumed = alternatives[0]
    ending = max(resumed.branches, key=lambda b: b.probability)

    decisions = record_of(reg, pos, played, action, ending)
    loc = tool.Localiser(reg, tool.load_names("ja"))
    groups = tool.turn_events(reg, loc, decisions[0], decisions[1:], 1.0)
    text = "\n".join(
        (header or "") + " " + " ".join(lines) for header, lines in groups
    )

    recorded = next(e for e in played.events if "stompingtantrum" in e).split()[1]
    rejected = next(e for e in result.suspended[0].events if "stompingtantrum" in e).split()[1]
    assert recorded != rejected
    assert recorded in text, f"the recorded roll {recorded} is missing from:\n{text}"
    assert rejected not in text, (
        f"the log printed {rejected}, which is suspended[0]'s roll, not the record's:\n{text}"
    )
    assert "⚠" not in text, f"a reproducible turn must not be flagged:\n{text}"


def test_an_unmatchable_pause_is_flagged_rather_than_guessed(tool, interrupted) -> None:  # noqa: ANN001
    """A record whose pause no suspension explains still has to say so.

    The matcher returns the nearest candidate, which is the right answer when the record
    is reproducible and a lie when it is not, so the gap is reported.
    """
    reg, pos, result = interrupted
    played = result.suspended[-1]
    chooser, alternatives = resume_alternatives(reg, played)
    action, resumed = alternatives[0]
    ending = max(resumed.branches, key=lambda b: b.probability)
    decisions = record_of(reg, pos, played, action, ending)
    # Move the pause somewhere the resolver cannot reach: one HP off every roll.
    for side in decisions[1]["position"]["sides"]:
        for mon in side["pokemon"]:
            mon["hp"] = max(1, mon["hp"] - 3)

    loc = tool.Localiser(reg, tool.load_names("ja"))
    groups = tool.turn_events(reg, loc, decisions[0], decisions[1:], 1.0)
    text = "\n".join(
        (header or "") + " " + " ".join(lines) for header, lines in groups
    )
    assert "⚠" in text and "中断時の局面" in text, (
        f"an unreachable pause has to be named as one:\n{text}"
    )
