"""A menu belongs to the agent that built it, in both seats and both information sets.

`_menus`' own docstring says so -- "handing one agent's menu to the other would make a
ranking comparison measure nothing" -- and three paths handed it over anyway:

  * `same_menu` compared the ranking FLAG and not the leaf, so two arms both passing
    --rank-leaf with different models counted as one menu. About 25,000 recorded games.
  * under `sheets` the rebuild was unreachable, so side 1 always played side 0's menu and
    `ranked[1]`/`policies[1]` were dead arguments the provenance recorded anyway.
  * the replacement node took both strategies off `matrix(pos, leaves[0])`.

These tests are written against the shape of the fix rather than against a whole game,
because a game needs a regulation, a node and several seconds; what is under test is which
construction each side gets, and that is decidable from the arguments.
"""

from __future__ import annotations

import inspect

from pokeuraou import selfplay


def source_of(name: str) -> str:
    return inspect.getsource(getattr(selfplay, name))


def test_same_menu_consults_the_leaf() -> None:
    """Two arms ranking BY the leaf with different leaves do not share a menu."""
    body = source_of("play_game")
    start = body.index("same_menu = (")
    condition = body[start : body.index(")", body.index("leaves[1] is leaves[0]", start))]
    assert "leaves[1] is leaves[0]" in condition, (
        "the leaf is not in the condition, so --rank-leaf on both arms with different "
        "models counts as one menu"
    )
    assert "policies[0] is not None" in condition, (
        "a shared policy ranks without consulting the leaf, so it must still short-circuit"
    )
    assert "not ranked[0]" in condition, (
        "damage ordering does not consult the leaf either"
    )


def test_the_hidden_bench_path_rebuilds_the_other_side() -> None:
    """The `spreads` branch has to do what the open branch does when the arms differ."""
    body = source_of("play_game")
    hidden = body[body.index("if spreads is not None:") :]
    hidden = hidden[: hidden.index("        else:")]
    assert "if not same_menu:" in hidden, "the hidden branch never rebuilds"
    assert hidden.count("belief_solve(") == 2, (
        "the column player's answer has to come from its own solve, so there are two"
    )
    assert "foe_answers[1].strategy" in hidden, (
        "side 1's strategy must come from the solve over side 1's menu"
    )
    # IKA-282: each of the two solves is asked only for the side it is read on
    # (`tests/test_board_belief_once.py` counts it in a game).
    assert "sides=(1,)" in hidden, "the solve over side 1's menu builds side 0's node too"


def test_the_replacement_node_solves_each_side_with_its_own_leaf() -> None:
    body = source_of("_do_replacement_node")
    assert "matrix(pos, leaves[1])" in body, (
        "the column player's replacement is still chosen by the row player's evaluator"
    )
    assert "if leaves[1] is not leaves[0]:" in body, (
        "one leaf on both sides must stay one solve, or every self-play game pays twice"
    )


def test_a_hidden_bench_game_refuses_depth_and_sparse() -> None:
    """`belief_solve` has no parameter for either, so accepting them is a lie.

    Since IKA-111 it has one for depth 2 in the restricted reading, and only that
    (`tests/test_hidden_depth2.py`): depth 2 without it, the flag at depth 1, and the
    sparse solve are still refused.

    Raised before any work happens, so the caller learns at the first game rather than
    finding the flags missing from a finished run's behaviour and present in its record.
    """
    import numpy as np
    import pytest

    from pokeuraou.payoff import HP_SHARE

    with pytest.raises(ValueError, match="hidden bench"):
        selfplay.play_game(
            None,  # never reached
            np.random.default_rng(0),
            [],
            [],
            "probe",
            objective=HP_SHARE,
            depth=(2, 1),
            sheets=([], []),
        )
    with pytest.raises(ValueError, match="hidden bench"):
        selfplay.play_game(
            None,
            np.random.default_rng(0),
            [],
            [],
            "probe",
            objective=HP_SHARE,
            solve_sparsely=(True, False),
            sheets=([], []),
        )
    with pytest.raises(ValueError, match="hidden bench"):
        # The depth-2 reading is the third one. It reaches `search` through the same
        # branch the other two do, which `belief_solve` does not take at all.
        selfplay.play_game(
            None,
            np.random.default_rng(0),
            [],
            [],
            "probe",
            objective=HP_SHARE,
            solve_restricted=(True, False),
            sheets=([], []),
        )
