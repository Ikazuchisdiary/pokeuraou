"""Eject Button, Red Card and Emergency Exit switch their Pokemon out (IKA-191).

Showdown a5df827, the champions mod (data/mods/champions/items.ts, abilities.ts,
scripts.ts `hitStepMoveHitLoop`) and data/items.ts for Red Card:

* Eject Button, `onAfterMoveSecondary`: a damaging move from another Pokemon that reached
  the holder, the holder still standing, its side able to switch, no forced switch on it
  and no active Pokemon's `switchFlag === true` -- then the item is used and the holder's
  `switchFlag` is set. Sheer Force skips `afterMoveSecondaryEvent`, so a move it stripped
  of its secondaries ejects nobody.
* Red Card, `onAfterMoveSecondary` (priority 0, after Eject Button's 2): the attacker still
  standing and able to switch; the card is used and the attacker is dragged out at random
  at the end of the action (`dragIn`), with no request.
* Emergency Exit (and Wimp Out, the same handler): the holder's HP crossing half -- on a
  target after `afterMoveSecondaryEvent` (not under Sheer Force), on the user after
  `DamagingHit`, recoil and Life Orb, and on anyone after the residual phase.

Neither engine did any of it. A `switchFlag` mid-turn is what U-turn leaves, so the
holder's replacement goes through the same suspension; after the residual phase it is owed
with the faint replacements, which is the one request Showdown makes there. Red Card's drag
is a random replacement, which is recorded the way Dragon Tail's is -- a forced switch the
resolver does not draw -- and the port refuses it as it refuses Dragon Tail.

Each case is played by Showdown first; the controls are the hits that do not cross half
and the Sheer Force move with secondaries.

A hit that a Substitute takes (IKA-180) sets none of them off: `spreadMoveHit` turns the
target it met into `null` (`HIT_SUBSTITUTE`), `afterMoveSecondaryEvent` is given only the
targets left, and the doll's damage is not `totalDamage` -- so no Eject Button, no Red
Card and no Emergency Exit. Both engines reach `move_hit` only for a hit on the Pokemon.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import SideAction, side_actions, switch_actions_after_faint
from pokeuraou.beliefnode import reaches_bench
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import Position
from pokeuraou.resolve import (
    Budget,
    batched_payoffs,
    replacements_needed,
    resolve_turn,
    resume_alternatives,
    resume_turn,
    self_switches_needed,
)

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 0, "spe": 0}
MID = {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def _mon(
    species: str, ability: str, moves: list[str], sp: dict[str, int], item: str | None = None
) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, item=item, sp=dict(sp))


BENCH = [
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], FAST),
    _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], FAST),
]
RILLABOOM = _mon("Rillaboom", "Overgrow", ["uturn", "protect", "swordsdance", "woodhammer"], FAST)
INCINEROAR = _mon("Incineroar", "Blaze", ["protect", "swordsdance", "knockoff", "fakeout"], FAST)
FERALIGATR = _mon("Feraligatr", "Sheer Force", ["crunch", "protect", "aquajet", "swordsdance"], FAST)

EJECT = _mon("Rillaboom", "Overgrow", ["protect", "swordsdance"], BULKY, item="Eject Button")
RED = _mon("Rillaboom", "Overgrow", ["protect", "swordsdance"], BULKY, item="Red Card")
GOLISOPOD = _mon("Golisopod", "Emergency Exit", ["protect", "swordsdance", "liquidation"], MID)
TYRANITAR = _mon("Tyranitar", "Sand Stream", ["protect", "rockslide", "crunch"], SLOW)
#: The same holders with a Substitute to put up first (IKA-180).
DOLL_MOVES = ["protect", "swordsdance", "substitute"]
EJECT_DOLL = _mon("Rillaboom", "Overgrow", DOLL_MOVES, BULKY, item="Eject Button")
RED_DOLL = _mon("Rillaboom", "Overgrow", DOLL_MOVES, BULKY, item="Red Card")
GOLISOPOD_DOLL = _mon("Golisopod", "Emergency Exit", ["protect", "swordsdance", "substitute"], MID)


def _team_b(lead: TeamSet, partner: TeamSet | None = None) -> list[TeamSet]:
    return [
        lead,
        partner or _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], FAST),
        _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], FAST),
        _mon("Incineroar", "Blaze", ["protect", "swordsdance", "knockoff"], FAST),
    ]


#: The foe's lead uses Swords Dance (slower than everything) and its partner Protects;
#: our partner Protects too, so the one hit is the case's.
WOOD_HAMMER = ["move 4 1, move 1", "move 2, move 1"]
KNOCK_OFF = ["move 2, move 3 1", "move 2, move 1"]
SAND_TURN = ["move 3, move 2", "move 2, move 1"]
#: The holder puts up a Substitute while everyone else Protects.
DOLL_UP = ["move 2, move 1", "move 3, move 1"]

#: name -> (our team, their team, turns played first, the turn compared, what Showdown
#: does: "eject" p2a owes a switch mid-turn, "drag" p1a is dragged out, "residual" p2a owes
#: one after the turn, None nothing).
CASES: dict[str, tuple] = {
    "eject-button": ([RILLABOOM, INCINEROAR, *BENCH], _team_b(EJECT), [], WOOD_HAMMER, "eject"),
    "red-card": ([RILLABOOM, INCINEROAR, *BENCH], _team_b(RED), [], WOOD_HAMMER, "drag"),
    "emergency-exit": ([RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD), [], WOOD_HAMMER, "eject"),
    # The second Knock Off takes Golisopod from 71% to 43%.
    "emergency-exit-second-hit": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD), [KNOCK_OFF],
        ["move 3, move 3 1", "move 2, move 1"], "eject",
    ),
    # Aqua Jet has no secondaries, so Sheer Force leaves `afterMoveSecondaryEvent` alone.
    "sheer-force-without-secondaries": (
        [FERALIGATR, INCINEROAR, *BENCH], _team_b(EJECT), [], ["move 3 1, move 1", "move 2, move 1"],
        "eject",
    ),
    # Sand from 98/182 to 87/182 at the end of the fourth turn.
    "emergency-exit-in-the-residual": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD, TYRANITAR),
        [KNOCK_OFF, SAND_TURN, SAND_TURN], SAND_TURN, "residual",
    ),
    # Controls.
    "control-first-hit-stays-above-half": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD), [], KNOCK_OFF, None,
    ),
    "control-sheer-force-crunch-into-eject-button": (
        [FERALIGATR, INCINEROAR, *BENCH], _team_b(EJECT), [], ["move 1 1, move 1", "move 2, move 1"],
        None,
    ),
    "control-sheer-force-crunch-into-emergency-exit": (
        [FERALIGATR, INCINEROAR, *BENCH], _team_b(GOLISOPOD), [], ["move 1 1, move 1", "move 2, move 1"],
        None,
    ),
    "control-sand-above-half": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD, TYRANITAR), [KNOCK_OFF], SAND_TURN, None,
    ),
    # The Wood Hammer breaks the doll and nothing else (IKA-180 x IKA-191).
    "doll-takes-the-hit-eject-button": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(EJECT_DOLL), [DOLL_UP], WOOD_HAMMER, None,
    ),
    "doll-takes-the-hit-red-card": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(RED_DOLL), [DOLL_UP], WOOD_HAMMER, None,
    ),
    "doll-takes-the-hit-emergency-exit": (
        [RILLABOOM, INCINEROAR, *BENCH], _team_b(GOLISOPOD_DOLL), [DOLL_UP], WOOD_HAMMER, None,
    ),
}

#: No crit, no chance-based secondary, the maximum roll -- what the oracle's policy pins.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _force(request: dict | None) -> tuple[bool, ...]:
    return tuple(bool(f) for f in ((request or {}).get("forceSwitch") or []))


def _play(oracle: Oracle, name: str):  # noqa: ANN202
    ours, theirs, setup, choices, _what = CASES[name]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    asked = (_force(handle.requests[0]), _force(handle.requests[1]))
    log = list(handle.log)
    return handle, before, after, asked, log


def _actions(reg, pos: Position, choices: list[str]) -> list[SideAction]:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


def _lead_item(pos: Position, side: int) -> str | None:
    return pos.sides[side].pokemon[pos.sides[side].active[0]].item


def _what_python_does(pos: Position, suspended: bool) -> str | None:
    lead0 = pos.sides[0].pokemon[pos.sides[0].active[0]]
    if lead0.has_volatile("pendingforceswitch"):
        return "drag"
    if suspended and self_switches_needed(pos) == ((False, False), (True, False)):
        return "eject"
    if not suspended and replacements_needed(pos)[1] == (True, False):
        return "residual"
    return None


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_switches_the_holder(oracle: Oracle, name: str) -> None:
    """The fact the case is named for."""
    handle, before, after, asked, log = _play(oracle, name)
    handle.close()
    what = CASES[name][4]
    dragged = any(line.startswith("|drag|p1a:") for line in log)
    if name.startswith("doll-"):
        # The case is only a case if the doll is what the Wood Hammer met.
        assert any(
            line.startswith("|-end|p2a:") and "Substitute" in line for line in log
        ), log[-12:]
    if what == "drag":
        assert dragged and asked == ((), ()), (asked, log[-12:])
        assert _lead_item(after, 1) is None
    elif what in ("eject", "residual"):
        assert asked == ((), (True, False)), (asked, log[-12:])
        assert not dragged
    else:
        assert asked == ((), ()) and not dragged, (asked, log[-12:])
        assert _lead_item(after, 1) == _lead_item(before, 1)


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_switches_the_holder_as_showdown_does(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    handle, before, after, _asked, _log = _play(oracle, name)
    handle.close()
    choices, what = CASES[name][3], CASES[name][4]
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    outcomes = [(b.position, False, b.events) for b in result.branches]
    outcomes += [(s.position, True, s.events) for s in result.suspended]
    assert outcomes
    for pos, suspended, events in outcomes:
        ours = _what_python_does(pos, suspended)
        assert ours == what, f"{name}: showdown {what}, python {ours}; " + " / ".join(events)
        assert _lead_item(pos, 1) == _lead_item(after, 1), (name, _lead_item(pos, 1))


def test_the_ejected_holders_move_goes_with_it(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The slower holder had chosen Swords Dance. Its replacement does not use it."""
    handle, before, _after, _asked, _log = _play(oracle, "eject-button")
    handle.step([None, "switch 3, pass"])
    assert handle.choice_errors == [], handle.choice_errors
    theirs = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    assert not any(line.startswith("|move|p2a:") for line in log), log[-10:]
    standing = theirs.sides[1].pokemon[theirs.sides[1].active[0]]
    assert standing.species == "kingambit" and not standing.boosts

    result = resolve_turn(reg, before, _actions(reg, before, WOOD_HAMMER), budget=BUDGET)
    assert result.suspended and not result.branches
    for pause in result.suspended:
        option = next(
            o for o in switch_actions_after_faint(reg, pause.position, 1, [True, False])
            if o.to_choice() == "switch 3, pass"
        )
        passes = SideAction(slots=tuple(o for o in _passes(reg, pause.position, 0)))
        resumed = resume_turn(reg, pause, [passes, option])
        for branch in resumed.branches:
            mon = branch.position.sides[1].pokemon[branch.position.sides[1].active[0]]
            assert mon.species == "kingambit" and not mon.boosts, (mon.boosts, branch.events)


def _passes(reg, pos: Position, side: int):  # noqa: ANN001, ANN202
    from pokeuraou.actions import PassAction

    return [PassAction(slot=i) for i in range(len(pos.sides[side].active))]


def test_a_u_turn_into_an_eject_button_asks_both_sides_in_turn(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Showdown asks both sides at once (tests/test_eject_selfswitch.py). The node asks the
    U-turn's side first and then the holder's, before anything else in the turn runs, and
    says that it did."""
    ours = [RILLABOOM, INCINEROAR, *BENCH]
    handle = oracle.create(FORMAT_ID, ours, _team_b(EJECT), policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    choices = ["move 1 1, move 1", "move 2, move 1"]
    handle.step(choices)
    asked = (_force(handle.requests[0]), _force(handle.requests[1]))
    handle.close()
    assert asked == ((True, False), (True, False))

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert result.suspended and not result.branches
    for pause in result.suspended:
        assert self_switches_needed(pause.position) == ((True, False), (True, False))
        chooser, alternatives = resume_alternatives(reg, pause)
        assert chooser == 0 and alternatives
        for _option, resumed in alternatives:
            assert "simultaneous mid-turn replacements" in resumed.unmodelled
            assert resumed.suspended and not resumed.branches
            for again in resumed.suspended:
                assert self_switches_needed(again.position) == ((False, False), (True, False))


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_switches_the_holder_as_python_does(
    reg, oracle: Oracle, bridged: None, name: str  # noqa: ANN001
) -> None:
    handle, before, _after, _asked, _log = _play(oracle, name)
    handle.close()
    # The port declines a position with a stats override (it reads that as a Transform);
    # both engines compute the same stats from the spreads.
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    choices, what = CASES[name][3], CASES[name][4]
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    if what == "drag":
        # Dragon Tail's shape: a replacement drawn at random, which the port refuses.
        assert there is None
        return
    assert there is not None, "the port refused the turn"
    here = resolve_turn(reg, before, actions, budget=BUDGET)
    assert there.branches == pytest.approx([b.probability for b in here.branches], abs=1e-12)
    assert there.suspended == pytest.approx([s.probability for s in here.suspended], abs=1e-12)
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        assert _what_python_does(chosen.position, False) == _what_python_does(branch.position, False)


def test_the_port_folds_a_node_where_both_sides_owe_as_python_does(
    reg, oracle: Oracle, bridged: None  # noqa: ANN001
) -> None:
    """The node's matrix, through the port and without it, where a U-turn meets an Eject
    Button (both sides owe, one after the other), where Wood Hammer ejects the holder (one
    side), and the controls beside them. The folds through both suspensions are the port's
    own; a port that never ejects gives other numbers (the old binary, in the record)."""
    ours = [RILLABOOM, INCINEROAR, *BENCH]
    handle = oracle.create(FORMAT_ID, ours, _team_b(EJECT), policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    wanted = {
        0: ["move 1 1, move 1", "move 4 1, move 1", "move 1 1, move 3 1", "move 2, move 1"],
        1: ["move 2, move 1", "move 1, move 2", "move 2, move 3"],
    }
    menus = {
        side: [a for a in side_actions(reg, before, side) if a.to_choice() in wanted[side]]
        for side in (0, 1)
    }
    assert [len(menus[0]), len(menus[1])] == [4, 3]
    budget = Budget.matrix()
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]
    os.environ.pop(rustnode.ENV_ENABLE, None)
    rustnode.reset()
    python, _notes, _exact = batched_payoffs(reg, before, menus[0], menus[1], evaluators, budget=budget)
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    port, _port_notes, _port_exact = batched_payoffs(
        reg, before, menus[0], menus[1], evaluators, budget=budget
    )
    for mine, theirs in zip(python, port, strict=True):
        assert abs(mine - theirs).max() < 1e-12, (mine, theirs)


def test_a_red_card_reaches_the_attackers_hidden_bench(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """IKA-128's mask with the item's path: a damaging move into a Red Card holder can drag
    the attacker's unseen Pokemon in, a Protect cannot, and without the card nothing does.
    Dragon Tail is read by its move id (the choice string only names an index)."""
    ours = [RILLABOOM, INCINEROAR, *BENCH]
    handle = oracle.create(FORMAT_ID, ours, _team_b(RED), policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.close()
    row = [a for a in side_actions(reg, before, 0) if a.to_choice() in ("move 4 1, move 1", "move 2, move 1")]
    col = [a for a in side_actions(reg, before, 1) if a.to_choice() == "move 2, move 1"]
    assert len(row) == 2
    attacking = [any(getattr(s, "move_id", "") == "woodhammer" for s in a.slots) for a in row]
    hidden = {0: (2, 3), 1: ()}
    mask = reaches_bench(reg, row, col, hidden, before)
    assert [bool(mask[i, 0]) for i in range(len(row))] == attacking
    assert not reaches_bench(reg, row, col, {0: (), 1: (2, 3)}, before).any()
    bare = before.copy()
    bare.sides[1].pokemon[bare.sides[1].active[0]].item = None
    assert not reaches_bench(reg, row, col, hidden, bare).any()

    from pokeuraou.actions import MoveAction, PassAction

    tail = SideAction(
        slots=(MoveAction(slot=0, move_index=1, move_id="dragontail", target=1), PassAction(slot=1))
    )
    assert reaches_bench(reg, [tail], col, {0: (), 1: (2, 3)})[0, 0]
