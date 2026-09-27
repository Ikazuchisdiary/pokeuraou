"""IKA-345: the port's chance tags (`EventLog::chance`, `PortBranch.chance`).

- **asking changes nothing**: a turn asked with its draws has the same branches, in the same
  order, with the same weights and positions, as the turn asked without;
- **the tags tell branches apart**: two branches of one turn never carry the same tags --
  where branches merged, the tags left are the ones all of them share, and rolls that came
  to the same thing widen to a range (``roll ... 0-2 16 0-2``) rather than vanish;
- **positive control**: on the exact budget the turns below draw crits, accuracy and damage
  rolls, and the tags say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import port
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def turns():  # noqa: ANN201
    roster = load_roster("rizabanadohido")
    register_mega_stones(roster.reg)
    reg = roster.reg
    pos = position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    rng = np.random.default_rng(3)
    out = []
    for _ in range(3):
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        for i in range(3):
            pair = [ours[i % len(ours)], theirs[(2 * i) % len(theirs)]]
            out.append((reg, pos, pair))
        res = port.turn(reg, pos, [ours[0], theirs[0]], Budget.exact(), full=True)
        w = np.array([b.probability for b in res.outcomes])
        pos = res.outcomes[int(rng.choice(len(w), p=w / w.sum()))].position
    return out


def test_asking_for_the_draws_changes_no_branch(turns) -> None:  # noqa: ANN001
    for reg, pos, pair in turns:
        plain = port.turn(reg, pos, pair, Budget.exact(), full=True)
        told = port.turn(reg, pos, pair, Budget.exact(), full=True, events=True)
        assert len(plain.outcomes) == len(told.outcomes)
        for a, b in zip(plain.outcomes, told.outcomes, strict=True):
            assert a.probability == b.probability and a.position == b.position
            assert a.chance == []


def test_the_tags_tell_a_turns_branches_apart(turns) -> None:  # noqa: ANN001
    kinds: set[str] = set()
    widened = 0
    for reg, pos, pair in turns:
        told = port.turn(reg, pos, pair, Budget.exact(), full=True, events=True)
        tags = [tuple(b.chance) for b in told.outcomes]
        assert len(set(tags)) == len(tags), "two branches of one turn with the same draws"
        for branch in told.outcomes:
            for tag in branch.chance:
                kinds.add(tag.split(" ")[0])
                widened += tag.startswith("roll ") and "-" in tag.split(" ")[-1]
    # Positive control: the exact budget forks on crits, accuracy and rolls here.
    assert {"crit", "nocrit", "hit", "miss", "roll"} <= kinds, kinds
    assert widened, "no merged roll widened to a range"
