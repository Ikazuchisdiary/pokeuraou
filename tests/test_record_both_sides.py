"""IKA-127: each decision records what each side's search was conditioned on.

Under a hidden bench the two sides solve different games, and the record used to keep
side 0's value and the true position only. Now it also keeps, per decision, each side's
shown identities and side 1's own value, and per game both lead pairs. These tests hold
the recorded "shown" to the replay a reader would otherwise have had to do, on the games
of the Rust generation worker (IKA-463; `play_game`'s games until then).

What these tests no longer hold (IKA-463 removed `play_game` with its scripted games): that
side 1's value in a mirror is side 0's mirrored, and that a lead that went back at full HP
stays shown. The worker's records carry both fields; the scripted turns that forced the two
cases were `play_game`'s `narrow` patched in Python, which the Rust worker does not read.
"""

from __future__ import annotations

import pytest

from pokeuraou.hidden import identity
from pokeuraou.position import Position
from pokeuraou.selfplay import replay_shown

from ._worker_games import worker_games


@pytest.fixture(scope="module")
def games(tmp_path_factory) -> list[dict]:  # noqa: ANN001
    """Three finished hidden-bench games of the worker, as their JSON."""
    _pool, played = worker_games(tmp_path_factory.mktemp("both"), 3)
    return played


def test_the_recorded_shown_is_the_replayed_shown(games) -> None:  # noqa: ANN001
    hidden = 0
    for game in games:
        assert game["information"] == "hidden-bench"
        decisions = game["decisions"]
        assert len(decisions) >= 3, "the game ended before it could say anything"
        recorded = [d["shownIdentities"] for d in decisions]
        assert recorded == replay_shown(game)
        hidden += sum(len(shown[1]) < 4 for shown in recorded)
    # Something was hidden, or the comparison is the open game's in disguise.
    assert hidden > 0


def test_the_replay_can_fail(games) -> None:  # noqa: ANN001
    """The control: a record whose shown identities are changed is not its replay."""
    game = dict(games[0])
    decisions = [dict(d) for d in game["decisions"]]
    late = next(i for i, d in enumerate(decisions) if len(d["shownIdentities"][1]) >= 3)
    decisions[late]["shownIdentities"] = [decisions[late]["shownIdentities"][0], []]
    game["decisions"] = decisions
    assert [d["shownIdentities"] for d in decisions] != replay_shown(game)


def test_side_ones_own_value_is_recorded_where_it_solved_one(games) -> None:  # noqa: ANN001
    seen = 0
    for game in games:
        for decision in game["decisions"]:
            if decision["kind"] == "selfswitch":
                assert "foeSearchValue" not in decision
                continue
            value = decision["foeSearchValue"]
            # Side 0's units: a share in [0, 1], not the negated game's [-1, 0].
            assert 0.0 <= value <= 1.0, value
            assert 0.0 <= decision["searchValue"] <= 1.0
            seen += 1
    assert seen > 0


def test_leads_are_the_turn_one_pairs_in_active_order(games) -> None:  # noqa: ANN001
    for game in games:
        first = Position.from_json(game["decisions"][0]["position"])
        assert first.turn == 1
        assert game["leads"] == [
            [identity(side.pokemon[k]) for k in side.active] for side in first.sides
        ]
