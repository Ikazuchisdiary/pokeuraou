"""Explosion, Self-Destruct, Misty Explosion, Memento, Healing Wish and the always-crit moves
in the port, played against Showdown (IKA-208).

The port refused all of them by their dex field (`selfdestruct`, `willCrit`); Python filled
the cells. Python has no `selfdestruct` at all -- its Explosion leaves the user standing --
so it is not asserted here.

`sim/battle-actions.ts`, `useMoveInner`:

    let tryMoveResult = this.battle.singleEvent('TryMove', ...) && this.battle.runEvent('TryMove', ...);
    if (!tryMoveResult) { ...; return tryMoveResult; }
    ...
    if (this.battle.gen !== 4 && move.selfdestruct === 'always') {
        this.battle.faint(pokemon, pokemon, move);
    }

so the user goes before any target is looked at -- into a Protect as well -- unless Damp's
`onAnyTryMove` (data/abilities.ts, `breakable`) stops the move first. `faint()` zeroes the
HP only, so the blast is computed from the user as it was. `runMoveEffects`:

    if (moveData.selfdestruct === 'ifHit' && damage[i] !== false) this.faint(source, source, move);

and a status move's `damage[i]` is `undefined` once it reaches the target: Memento faints
its user into Clear Body, not into Protect or a Substitute. Healing Wish's `onTryHit` returns
`NOT_FAIL` with no one to switch in, and otherwise leaves its `slotCondition` behind.

`willCrit: true` (Storm Throw, Flower Trick, Frost Breath) sets `moveHit.crit` in `getDamage`
without a roll, and `runEvent('CriticalHit')` -- Battle Armor, Shell Armor -- can still stop it.

The positive control is the exe before this change, which refuses all of these turns
(`POKEURAOU_RUST_NODE_BIN=<old exe>`); the controls are the cases where the effect must not
fire.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user's four moves are set per case; everyone else: 1 Protect, 2 Helping Hand,
#: 3 Substitute, 4 Iron Defense.
FILL = ["protect", "helpinghand", "substitute", "irondefense"]


def _teams(
    moves: list[str],
    ability: str = "Inner Focus",
    foe: tuple[str, str] = ("Garchomp", "Rough Skin"),
    bench: bool = True,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon("Alakazam", ability, moves, 20),
        _mon("Kangaskhan", "Inner Focus", FILL, 10),
    ]
    if bench:
        mine.append(_mon("Milotic", "Marvel Scale", FILL, 0))
    theirs = [
        _mon(foe[0], foe[1], FILL, 2),  # never tied with Incineroar (Swampert is 60 too)
        _mon("Incineroar", "Blaze", FILL, 0),
        _mon("Milotic", "Marvel Scale", FILL, 0),
    ]
    return mine, theirs


BLASTS = ["explosion", "selfdestruct", "mistyexplosion", "memento"]
CRITS = ["stormthrow", "flowertrick", "frostbreath", "healingwish"]
#: Everyone but the case's user Helps its partner.
QUIET = "move 2 -2, move 2 -1"
MINE = "move 2 -1"  # Kangaskhan Helps Alakazam
GUARDED = "move 1, move 1"  # both foes Protect

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    "explosion-hits-every-adjacent-and-faints-its-user": (
        _teams(BLASTS), [], [f"move 1, {MINE}", QUIET], "|faint|p1a: Alakazam",
    ),
    "self-destruct-hits-every-adjacent-and-faints-its-user": (
        _teams(BLASTS), [], [f"move 2, {MINE}", QUIET], "|faint|p1a: Alakazam",
    ),
    "misty-explosion-faints-its-user": (
        _teams(BLASTS), [], [f"move 3, {MINE}", QUIET], "|faint|p1a: Alakazam",
    ),
    # Every target Protected, and the user goes all the same.
    "explosion-into-protect-still-faints-its-user": (
        _teams(BLASTS), [], ["move 1, move 1", GUARDED], "|faint|p1a: Alakazam",
    ),
    "control-explosion-under-damp": (
        _teams(BLASTS, foe=("Swampert", "Damp")), [], [f"move 1, {MINE}", QUIET],
        "|cant|p2a: Swampert|ability: Damp",
    ),
    "explosion-by-a-mold-breaker-through-damp": (
        _teams(BLASTS, ability="Mold Breaker", foe=("Swampert", "Damp")), [], [f"move 1, {MINE}", QUIET],
        "|faint|p1a: Alakazam",
    ),
    "memento-drops-and-faints-its-user": (
        _teams(BLASTS), [], [f"move 4 1, {MINE}", QUIET], "|faint|p1a: Alakazam",
    ),
    "memento-into-clear-body-still-faints-its-user": (
        _teams(BLASTS, foe=("Metagross", "Clear Body")), [], [f"move 4 1, {MINE}", QUIET],
        "|faint|p1a: Alakazam",
    ),
    "control-memento-into-protect": (
        _teams(BLASTS), [], [f"move 4 1, {MINE}", GUARDED], "|-activate|p2a: Garchomp|move: Protect",
    ),
    "control-memento-into-a-substitute": (
        _teams(["memento", "protect", "explosion", "helpinghand"]),
        [["move 2, move 2 -1", "move 3, move 1"]], [f"move 1 1, {MINE}", QUIET],
        "|-fail|p1a: Alakazam",
    ),
    "healing-wish-faints-its-user-and-leaves-the-wish": (
        _teams(CRITS), [], [f"move 4, {MINE}", QUIET], "|faint|p1a: Alakazam",
    ),
    "control-healing-wish-with-no-one-to-come-in": (
        _teams(CRITS, bench=False), [], [f"move 4, {MINE}", QUIET], "|-fail|p1a: Alakazam",
    ),
    "storm-throw-always-crits": (
        _teams(CRITS), [], [f"move 1 1, {MINE}", QUIET], "|-crit|p2a: Garchomp",
    ),
    "flower-trick-always-crits": (
        _teams(CRITS), [], [f"move 2 1, {MINE}", QUIET], "|-crit|p2a: Garchomp",
    ),
    "frost-breath-always-crits": (
        _teams(CRITS), [], [f"move 3 2, {MINE}", QUIET], "|-crit|p2b: Incineroar",
    ),
    "control-storm-throw-into-shell-armor": (
        _teams(CRITS, foe=("Torkoal", "Shell Armor")), [], [f"move 1 1, {MINE}", QUIET],
        "|-damage|p2a: Torkoal|",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    out: dict[str, tuple] = {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }
    for index, side in enumerate(pos.sides):
        slots = side.slot_conditions
        out[f"p{index + 1}.slots"] = tuple(tuple(sorted(c.id for c in group)) for group in slots)
    out["winner"] = (pos.ended, pos.winner)
    return out


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict]:
    (mine, theirs), setup, choices, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 123" if len(mine) > 2 else "team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    assert any(line.startswith(shown) for line in log), f"Showdown did not do what {name} says: {log}"
    for side in before.sides:
        for party in side.pokemon:
            # The port declines a position with a stats override (it reads it as a
            # Transform); both engines compute the stats from the set.
            party.stats_override = None
    return before, choices, after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [next(a for a in side_actions(reg, pos, s) if a.to_choice() == choices[s]) for s in (0, 1)]


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    before, choices, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) == 1, there.branches
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None
    rust_now = _state(picked.position)
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
