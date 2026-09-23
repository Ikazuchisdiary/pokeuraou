"""The residual phase sorts on the Speed the Pokemon had when the phase began, before the
weather's own residual ends it (IKA-190).

vendor/pokemon-showdown/sim/battle.ts, `runAction`::

    case 'residual':
        ...
        this.updateSpeed();
        ...
        this.fieldEvent('Residual');

and `fieldEvent` collects every handler (each carrying its holder's `speed`), calls
`this.speedSort(handlers)` once, and only then walks them -- the weather's handler first,
which decrements its duration and ends it. So on the turn the sun runs out, Chlorophyll's
doubled Speed still orders the burn, Leftovers and the rest, and the phase is shuffled only
where those speeds tie.

Our resolver computed the residual order lazily, at the first residual that asked for it,
which is after the weather ended: a Chlorophyll Venusaur at 200 in the sun was sorted at
100. Against a 100 Speed foe that noted a random tie Showdown does not roll (the port sorted
at 200 and did not note it: the gen11L node where the notes disagreed); against a faster foe
it put the foe's residuals first. Each case is played by Showdown twice, with speed ties
kept and reversed, so a tie shows up as the residual order flipping between the two.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

# Python's resolver only for the one test that reads the event log (IKA-215).
from pokeuraou.resolve import resolve_turn as python_resolve_turn

from ._port import Budget
from .conftest import FORMAT_ID

TIE_NOTE = "residual speed tie (Showdown breaks it at random)"


def _mon(species: str, ability: str, moves: list[str], spe: int = 0) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp={"hp": 20, "atk": 0, "def": 10, "spa": 0, "spd": 10, "spe": spe})


#: Our Venusaur is 100 Speed, 200 in Torkoal's sun (five turns, from the lead). Turn 1 burns
#: both Venusaurs, so each residual phase shows both in the log.
OURS = [_mon("Venusaur", "Chlorophyll", ["swordsdance", "protect", "sludgebomb", "gigadrain"]),
        _mon("Torkoal", "Drought", ["willowisp", "protect", "flamethrower", "yawn"])]
#: The foe Venusaur's Speed points: 0 is 100 Speed, a tie once the sun is gone; 10 is
#: faster than ours without the sun and slower with it.
FOES = {"tie": 0, "faster": 10}
BURN = ["move 1, move 1 1", "move 1, move 1 1"]
WAIT = ["move 1, move 2", "move 1, move 2"]
STEPS = [BURN] + [WAIT] * 5
#: Turn 4: sun with two turns left. Turn 5: the sun ends in the residual phase. Turn 6: none.
TURNS = (4, 5, 6)
#: Whether Showdown shuffles the two Venusaurs in each turn's residual phase.
TIED = {"tie": {4: False, 5: False, 6: True}, "faster": {4: False, 5: False, 6: False}}


def _theirs(case: str) -> list[TeamSet]:
    return [_mon("Venusaur", "Overgrow", ["swordsdance", "protect", "sludgebomb", "gigadrain"],
                 spe=FOES[case]),
            _mon("Sableye", "Keen Eye", ["willowisp", "protect", "shadowball", "calmmind"])]


def _play(oracle: Oracle, case: str, tie: str = "keep") -> tuple[list[dict], list[list[str]]]:
    """Showdown's position before each turn (index 0 before turn 1) and, for each turn, the
    slots its burns hit, in the order the residual phase dealt them."""
    handle = oracle.create(FORMAT_ID, OURS, _theirs(case), policy=RandomnessPolicy(speed_tie=tie))
    handle.step(["team 12", "team 12"])
    positions = [handle.position]
    burns = []
    for step in STEPS:
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(handle.position)
        slots = [line.split("|")[2].split(":")[0] for line in handle.log
                 if line.startswith("|-damage|") and line.endswith("[from] brn")]
        burns.append(list(dict.fromkeys(slots)))
    handle.close()
    return positions, burns


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
            mon.stats_override = None
    return pos


def _chosen(reg, pos: Position, step: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _resolved(reg, positions: list[dict], turn: int):  # noqa: ANN001, ANN202
    start = _loaded(positions[turn - 1])
    chosen = _chosen(reg, start, STEPS[turn - 1])
    return start, chosen, python_resolve_turn(reg, start, chosen, budget=Budget.matrix())


@pytest.mark.oracle
@pytest.mark.parametrize("case", sorted(FOES))
def test_showdown_sorts_before_the_sun_ends(oracle: Oracle, case: str) -> None:
    """The facts: the sun ends in turn 5's residual phase; that phase is ordered at our
    sunny 200 (ours first, not shuffled); turn 6's is at 100 against the foe's Speed."""
    positions, kept = _play(oracle, case)
    _, reversed_ = _play(oracle, case, "reverse")
    before = Position.from_json(positions[4]).field
    assert (before.weather, before.weather_duration) == ("sunnyday", 1)
    assert Position.from_json(positions[5]).field.weather is None
    assert {turn: kept[turn - 1] != reversed_[turn - 1] for turn in TURNS} == TIED[case]
    assert kept[4] == reversed_[4] == ["p1a", "p2a"]
    if case == "faster":
        assert kept[5] == ["p2a", "p1a"]


@pytest.mark.oracle
@pytest.mark.parametrize("case", sorted(FOES))
def test_python_orders_the_residuals_as_showdown(reg, oracle: Oracle, case: str) -> None:  # noqa: ANN001
    """On every turn Showdown does not shuffle, our burn events come in its order.

    Still Python's: the port keeps no events (IKA-215). Its HP after the residual is held
    to Showdown's in `test_the_port_ends_the_residual_where_showdown_does`.
    """
    positions, kept = _play(oracle, case)
    for turn in TURNS:
        if TIED[case][turn]:
            continue
        _, _, result = _resolved(reg, positions, turn)
        assert result.branches
        for branch in result.branches:
            burns = [event.split()[0] for event in branch.events if event.endswith("(brn)")]
            assert burns == kept[turn - 1], (turn, branch.events)


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207). The port keeps no events, so
# the order is held through what it decides: every Pokemon's HP after the residual, on the
# turns Showdown does not shuffle.


@pytest.mark.oracle
@pytest.mark.parametrize("case", sorted(FOES))
def test_the_port_notes_the_ties_showdown_rolls(reg, oracle: Oracle, port, case: str) -> None:  # noqa: ANN001
    """`test_python_notes_the_ties_showdown_rolls` with the port's notes."""
    from ._port_showdown import port_weights

    positions, _ = _play(oracle, case)
    for turn in TURNS:
        start = _loaded(positions[turn - 1])
        chosen = _chosen(reg, start, STEPS[turn - 1])
        notes = set(port_weights(port, start, chosen, Budget.matrix())["unmodelled"])
        assert notes == ({TIE_NOTE} if TIED[case][turn] else set()), turn


@pytest.mark.oracle
@pytest.mark.parametrize("case", sorted(FOES))
def test_the_port_ends_the_residual_where_showdown_does(reg, oracle: Oracle, port, case: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn

    positions, _ = _play(oracle, case)
    for turn in TURNS:
        if TIED[case][turn]:
            continue
        start = _loaded(positions[turn - 1])
        after = port_turn(port, start, _chosen(reg, start, STEPS[turn - 1]))
        theirs = Position.from_json(positions[turn])
        hp = [[m.hp for m in side.pokemon] for side in after.sides]
        assert hp == [[m.hp for m in side.pokemon] for side in theirs.sides], turn
