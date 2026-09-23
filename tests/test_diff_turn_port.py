"""The port held to Showdown over whole battles, as `test_resolve` holds Python (IKA-207).

`tools/diff_turn.py` plays Showdown and hands every turn's position and choices to Python's
`resolve_turn` and, beside it, to `RustNode.resolve` under the same pins. These are the
port's column of `test_resolved_positions_match_showdown`, with the same thresholds, plus
the guard that matters while both engines exist: on a turn both answered, the port never
diverges where Python matched.

Measured when written (seed 1, 400 battles, PYTHONHASHSEED=0): Python diverged on 156 of
2,723 compared turns, the port on 127 of 2,456, and on the 2,456 both answered the two
diverged on the same 127 turns. The port refused 145 turns and stopped with Showdown at
126 mid-turn replacements it cannot continue yet (IKA-208, IKA-211).
"""

from __future__ import annotations

import pytest

from pokeuraou import rustnode
from pokeuraou.oracle import ORACLE_JS, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget

from . import _diff_turn_entry as diff_turn
from .conftest import FORMAT_ID
from .test_resolve import (
    _INFLICTOR_TEAM,
    _VICTIM_TEAM,
    MAX_SILENT_DIVERGENCE,
    MAX_TOTAL_DIVERGENCE_PER_SEED,
    _move_action,
)

pytestmark = pytest.mark.oracle


def _need_port() -> None:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")


@pytest.mark.parametrize("seed", [1, 5])
def test_the_port_matches_showdown_over_whole_battles(seed: int) -> None:
    _need_port()
    report = diff_turn.run(battles=8, roll=8, seed=seed, max_turns=10, port=True)
    port = report.port
    assert port is not None
    assert port.compared >= 20, f"only {port.compared} turns compared\n{port.render()}"
    assert port.silent_rate <= MAX_SILENT_DIVERGENCE, port.render()
    assert port.divergence_rate <= MAX_TOTAL_DIVERGENCE_PER_SEED, port.render()
    # While Python exists: nothing the port gets wrong that Python gets right.
    assert port.joint[("match", "diverge")] == 0, port.render()


def test_adding_the_port_changes_no_python_number() -> None:
    """The null control: the port's column is read beside Python's and moves none of it."""
    _need_port()
    alone = diff_turn.run(battles=4, roll=8, seed=3, max_turns=10)
    beside = diff_turn.run(battles=4, roll=8, seed=3, max_turns=10, port=True)
    assert (alone.compared, alone.matched, alone.silent_divergences) == (
        beside.compared, beside.matched, beside.silent_divergences
    )
    assert alone.port is None and beside.port is not None


def test_the_port_keeps_light_clays_screen_as_showdown_does(reg, oracle: Oracle, port) -> None:  # noqa: ANN001
    """`test_light_clay_extends_a_screen_and_the_dump_says_so` with the port: 8 with the
    clay, 5 without, as Showdown stores it after the turn."""
    from pokeuraou.actions import SideAction

    from ._port_showdown import port_turn

    for item, expected in (("lightclay", 8), ("leftovers", 5)):
        held = list(_INFLICTOR_TEAM)
        first = held[0]
        held[0] = TeamSet(
            first.species, first.ability, first.nature, list(first.moves), dict(first.sp), item=item
        )
        handle = oracle.create(FORMAT_ID, held, _VICTIM_TEAM, policy=RandomnessPolicy(damage_roll=8))
        handle.step(["team 1,2,3,4", "team 1,2,3,4"])
        before = Position.from_json(handle.position)
        ours = SideAction(
            slots=(
                _move_action(before, 0, 0, "lightscreen", None),
                _move_action(before, 0, 1, "protect", None),
            )
        )
        theirs = SideAction(
            slots=(
                _move_action(before, 1, 0, "protect", None),
                _move_action(before, 1, 1, "protect", None),
            )
        )
        handle.step([ours.to_choice(), theirs.to_choice()])
        assert not handle.choice_errors, handle.choice_errors
        after = Position.from_json(handle.position)
        handle.close()

        def screen(pos: Position) -> int | None:
            return next((c.duration for c in pos.sides[0].side_conditions if c.id == "lightscreen"), None)

        mine = screen(port_turn(port, before, [ours, theirs], Budget.deterministic(8)))
        assert mine == screen(after) == expected - 1, (item, mine, screen(after))
