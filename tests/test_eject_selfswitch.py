"""U-turn still switches its user out when the target ejects (IKA-167).

Showdown a5df827 changed two Champions handlers (IKA-164):

* aa6d5f085, Eject Button: the base item's `onAfterMoveSecondary` sets
  `source.switchFlag = false` when the holder is switched out by it, so the attacker's
  U-turn was cancelled. The champions mod overrides the item without that line.
* 57ecb348b, Emergency Exit / Wimp Out: the base ability cleared every other active
  Pokemon's `switchFlag` when it fired. The champions mod leaves them alone.

So under the mod both the holder and the attacker are asked to switch. Neither engine
here models the holder's exit (Python lists Emergency Exit as unmodelled, Eject Button not
at all), and both always honour the attacker's self-switch -- which the old Showdown did not
and the new one does. The tests pin the fact first (the holder's effect fired, and Showdown
asks for both switches), then hold the Python turn to Showdown's answer for the attacker
only, which is all the resolver claims. The controls are the same U-turn into a target
without the item or the ability.

Positive control (in the record, IKA-167): with the vendor at d3de52a17 the fact tests fail
because Showdown does not ask the attacker to switch, and the agreement tests fail because
Python still suspends on the U-turn.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 0, "spe": 0}


def _mon(
    species: str, ability: str, moves: list[str], sp: dict[str, int], item: str | None = None
) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, item=item, sp=dict(sp))


BENCH = [
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], FAST),
    _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], FAST),
]
TEAM_A = [
    _mon("Rillaboom", "Overgrow", ["uturn", "protect", "swordsdance", "woodhammer"], FAST),
    _mon("Incineroar", "Blaze", ["protect", "swordsdance", "knockoff"], FAST),
    *BENCH,
]


def _team_b(lead: TeamSet) -> list[TeamSet]:
    return [
        lead,
        _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], FAST),
        _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], FAST),
        _mon("Incineroar", "Blaze", ["protect", "swordsdance", "knockoff"], FAST),
    ]


#: The lead Rillaboom U-turns into the foe's lead on the last turn; every partner stays out
#: of it. Emergency Exit needs the U-turn to be the hit that crosses half, so that case
#: spends a turn on Swords Dance (the foe's lead and both partners Protect).
SETUP = ["move 3, move 1", "move 1, move 1"]
UTURN = ["move 1 1, move 2", "move 2, move 2"]

#: name -> (the foe's lead, the setup turns, the log line that shows the holder's effect,
#: or None for a control).
CASES = {
    "eject-button": (
        _mon("Rillaboom", "Overgrow", ["protect", "swordsdance"], BULKY, item="Eject Button"),
        [],
        "|-enditem|p2a: Rillaboom|Eject Button",
    ),
    "emergency-exit": (
        _mon("Golisopod", "Emergency Exit", ["protect", "swordsdance"], SLOW),
        [SETUP],
        "|-activate|p2a: Golisopod|ability: Emergency Exit",
    ),
    "control-no-item": (
        _mon("Rillaboom", "Overgrow", ["protect", "swordsdance"], BULKY),
        [],
        None,
    ),
    "control-no-emergency-exit": (
        _mon("Golisopod", "Shell Armor", ["protect", "swordsdance"], SLOW),
        [SETUP],
        None,
    ),
}

#: No crit, no chance-based secondary, the maximum roll -- what the oracle's policy pins.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _force(request: dict | None) -> bool:
    return bool(((request or {}).get("forceSwitch") or [False])[0])


def _play(oracle: Oracle, name: str) -> tuple[Position, tuple[bool, bool], list[str]]:
    """The position before the U-turn turn, Showdown's two slot-0 switch requests, the log."""
    lead, setup, _line = CASES[name]
    handle = oracle.create(FORMAT_ID, TEAM_A, _team_b(lead), policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for choices in setup:
        handle.step(choices)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(UTURN)
    assert handle.choice_errors == [], handle.choice_errors
    asked = (_force(handle.requests[0]), _force(handle.requests[1]))
    log = list(handle.log)
    handle.close()
    return before, asked, log


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_switches_the_attacker_out_as_well(oracle: Oracle, name: str) -> None:
    """The fact: the holder's effect fired and Showdown asks both sides for a switch."""
    _before, (attacker, holder), log = _play(oracle, name)
    line = CASES[name][2]
    if line is None:
        assert not any("Eject Button" in entry or "Emergency Exit" in entry for entry in log), log
        assert (attacker, holder) == (True, False), (name, attacker, holder)
    else:
        assert line in log, f"the holder's effect did not fire: {log}"
        assert (attacker, holder) == (True, True), (name, attacker, holder)


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207): under Showdown's pins the
# U-turn hits, and the port stops for the attacker's switch exactly when Showdown asks.


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_switches_the_attacker_as_showdown_does(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_weights

    before, (attacker, _holder), _log = _play(oracle, name)
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == UTURN[side]) for side in (0, 1)
    ]
    reply = port_weights(port, before, actions, Budget.deterministic(0))
    assert bool(reply["suspended"]) == attacker, (name, reply)
