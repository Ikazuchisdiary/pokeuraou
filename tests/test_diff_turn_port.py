"""The port held to Showdown over whole battles (IKA-207).

`tools/diff_turn.py` plays Showdown and hands every turn's position and choices to the port
under Showdown's pins. The tool still has Python's column beside it (IKA-212 takes it);
these run the port's column alone (`python=False`, IKA-210), with the thresholds
`test_resolve` held Python to. Until IKA-210 they ran both and also held the guard that
mattered while both engines existed: on a turn both answered, the port never diverged
where Python matched. Beside Python gone, a mid-turn replacement is now Showdown's
`default` on the run's next step rather than one Python's column drew, so the battles
are not the ones the numbers below were measured on.

Measured when written (seed 1, 400 battles, PYTHONHASHSEED=0): Python diverged on 156 of
2,723 compared turns, the port on 127 of 2,456, and on the 2,456 both answered the two
diverged on the same 127 turns. The port refused 145 turns and stopped with Showdown at
126 mid-turn replacements it cannot continue yet (IKA-208, IKA-211).

Since IKA-217 those 126 are resumed with Showdown's replacements and compared: 2,582 turns,
135 divergent, and still none where Python matched. The binary built with
`--features ika211-control` (the rest of a resumed turn loses its last action) diverges
alone on 52 of them.
"""

from __future__ import annotations

import pytest

from pokeuraou import rustnode
from pokeuraou.budget import Budget
from pokeuraou.oracle import ORACLE_JS, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

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
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")


@pytest.mark.parametrize("seed", [1, 5])
def test_the_port_matches_showdown_over_whole_battles(seed: int) -> None:
    _need_port()
    report = diff_turn.run(
        battles=8, roll=8, seed=seed, max_turns=10, port=True, python=False
    )
    port = report.port
    assert port is not None and report.compared == 0, "Python's column ran"
    assert port.compared >= 20, f"only {port.compared} turns compared\n{port.render()}"
    assert port.silent_rate <= MAX_SILENT_DIVERGENCE, port.render()
    assert port.divergence_rate <= MAX_TOTAL_DIVERGENCE_PER_SEED, port.render()


def _team(*conditions: str) -> list[dict[str, str]]:
    return [{"details": f"Mon{k}, L50", "condition": c} for k, c in enumerate(conditions)]


@pytest.mark.parametrize(
    ("flags", "conditions", "choice", "expected"),
    [
        # Python's replacement: one part per active slot.
        ([False, True], ("100/100", "90/100", "50/100", "80/100"), "pass, switch 4", {1: 3}),
        ([True, True], ("100/100", "90/100", "50/100", "80/100"), "switch 4, switch 3", {0: 3, 1: 2}),
        # A bare switch goes to the slot that owes one, as `getChoiceIndex` passes the rest.
        ([False, True], ("100/100", "90/100", "50/100", "80/100"), "switch 3", {1: 2}),
        # The run's own step: `autoChoose`, the first after the actives not fainted.
        ([True, False], ("100/100", "90/100", "0 fnt", "80/100"), "default", {0: 3}),
        ([True, True], ("100/100", "90/100", "50/100", "80/100"), "default", {0: 2, 1: 3}),
    ],
)
def test_the_port_is_given_the_replacement_showdown_read(
    flags: list[bool], conditions: tuple[str, ...], choice: str, expected: dict[int, int]
) -> None:
    """IKA-217: a paused turn is carried on with Showdown's choice, read as `Side.choose`
    reads it -- not with the port's own numbering of the party."""
    got = diff_turn._MODULE.showdown_switch_ins(flags, _team(*conditions), choice)  # noqa: SLF001
    assert got == expected


def test_the_port_carries_paused_turns_on_with_showdown() -> None:
    """IKA-217: a turn the port and Showdown both stopped inside is resumed and compared,
    not set aside; aimed at U-turn and friends so there are stops to carry on."""
    _need_port()
    report = diff_turn.run(
        battles=8, roll=8, seed=2, max_turns=10, port=True, self_switch=0.8, python=False
    )
    port = report.port
    assert port is not None
    assert port.paused > 0 and port.continued > 0, port.render()
    assert port.pending is None
    assert not any("paused" in reason for reason in port.skipped), port.render()


def test_two_columns_of_one_binary_agree() -> None:
    """The null control: two port columns of the same binary, in one run, count the same
    turns the same way -- paused turns carried on included. (Replaces the two controls
    that held Python's column still beside the port's, IKA-210.)"""
    _need_port()
    same = rustnode.binary_path()
    report = diff_turn.run(
        battles=4, roll=8, seed=2, max_turns=10, self_switch=0.8,
        exes=[("a", same), ("b", same)], python=False,
    )
    a, b = report.ports["a"], report.ports["b"]
    assert a.compared > 0 and a.continued > 0, a.render("a")
    assert a.render("x") == b.render("x")


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
