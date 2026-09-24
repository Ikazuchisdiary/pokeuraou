"""Last Resort in the port, played against Showdown (IKA-208).

The port refused it ("item-swapping or history move"); Python has the rule. `data/moves.ts`:

    onTry(source) {
        if (source.moveSlots.length < 2) return false;
        let hasLastResort = false;
        for (const moveSlot of source.moveSlots) {
            if (moveSlot.id === 'lastresort') { hasLastResort = true; continue; }
            if (!moveSlot.used) return false;
        }
        return hasLastResort;
    },

`moveSlot.used` is set by `moveUsed` and cleared on a switch. The positive control is the
exe before this change, which refuses these turns; the control is a Last Resort with one
move still unused, which fails.
"""

from __future__ import annotations

import pytest

from pokeuraou import rustnode
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from .conftest import FORMAT_ID
from .test_self_destruct_oracle import BUDGET, _actions, _state, bridged  # noqa: F401

pytestmark = pytest.mark.oracle

FILL = ["protect", "helpinghand", "substitute", "irondefense"]
USER = ["lastresort", "protect", "helpinghand", "irondefense"]


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=spread)


TEAMS = (
    [_mon("Kangaskhan", "Scrappy", USER, 20), _mon("Alakazam", "Inner Focus", FILL, 10)],
    [_mon("Garchomp", "Rough Skin", FILL, 2), _mon("Incineroar", "Blaze", FILL, 0)],
)
QUIET = "move 2 -2, move 2 -1"
MINE = "move 2 -1"
LAST_RESORT = [f"move 1 1, {MINE}", QUIET]
SETUP = [
    [f"move 2, {MINE}", QUIET],
    [f"move 3 -2, {MINE}", QUIET],
    [f"move 4, {MINE}", QUIET],
]

#: name -> (setup turns, a Showdown log line that shows the case).
CASES = {
    "last-resort-after-every-other-move": (SETUP, "|move|p1a: Kangaskhan|Last Resort|p2a: Garchomp"),
    "control-last-resort-with-one-move-unused": (SETUP[:2], "|-fail|p1a: Kangaskhan"),
    "control-last-resort-first": ([], "|-fail|p1a: Kangaskhan"),
}


def _play(oracle: Oracle, name: str) -> tuple[Position, dict]:
    setup, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, *TEAMS, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(LAST_RESORT)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    assert any(line.startswith(shown) for line in log), f"Showdown did not do what {name} says: {log}"
    return before, after


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    before, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, LAST_RESORT)
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) == 1, there.branches
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None
    rust_now = _state(picked.position)
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
