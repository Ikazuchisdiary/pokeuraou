"""Roar, Whirlwind, Dragon Tail and Circle Throw in the port, played against Showdown (IKA-208).

The port refused every `forceSwitch` move ("forceSwitch move" / "forceSwitch status move"),
and Red Card when it fired; Python marks the target `pendingforceswitch` and reports that
the replacement is drawn at random, and never switches it. Python is not asserted here.

`sim/battle-actions.ts`, `spreadMoveHit` step 6:

    forceSwitch(damage, targets, source, move) {
        for (const [i, target] of targets.entries()) {
            if (target && target.hp > 0 && source.hp > 0 && this.battle.canSwitch(target.side)) {
                const hitResult = this.battle.runEvent('DragOut', target, source, move);
                if (hitResult) target.forceSwitchFlag = true;
                else if (hitResult === false && move.category === 'Status') { ...fail }
            }
        }

(`runMoveEffects` fails a status one first when `canSwitch` is false), and at the end of
`runAction` (sim/battle.ts):

    for (const side of this.sides) for (const pokemon of side.active)
        if (pokemon.forceSwitchFlag) { if (pokemon.hp) this.actions.dragIn(...); ... }

`dragIn` takes `getRandomSwitchable` -- a `sample` over the bench in party order, which the
oracle's policy answers with the first -- and `switchIn(..., isDrag)` runs the newcomer's
switch-in at once. The move queued for the Pokemon dragged out is skipped (`runAction`
skips a Pokemon that is not active). Guard Dog, Suction Cups and Ingrain answer `DragOut`
with `null`: no drag, no failure.

Each case plays in Showdown; the port resolves the compared turn from Showdown's position
under a budget that does not branch chance (so the first on the bench, as the oracle), and
every active species, HP and boost is held to Showdown's. A second test lets the draw branch
and asks for one equal-weight branch per bench Pokemon, Showdown's among them. The
positive control is the exe before this change, which refuses every one of these turns.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget

from .conftest import FORMAT_ID
from .test_self_destruct_oracle import bridged  # noqa: F401

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user: 1 Roar, 2 Whirlwind, 3 Dragon Tail, 4 Circle Throw.
USER = _mon("Garchomp", "Rough Skin", ["roar", "whirlwind", "dragontail", "circlethrow"], 20)
FILL = ["protect", "helpinghand", "ingrain", "roar"]
PARTNER = _mon("Alakazam", "Inner Focus", FILL, 5)


def _foes(lead: TeamSet | None = None, bench: bool = True) -> list[TeamSet]:
    team = [
        lead or _mon("Kangaskhan", "Scrappy", FILL, 0),
        _mon("Milotic", "Marvel Scale", FILL, 1),
    ]
    if bench:
        team += [
            _mon("Incineroar", "Intimidate", FILL, 2),
            _mon("Sylveon", "Pixilate", FILL, 3),
        ]
    return team


MINE = "move 2 -1"  # Alakazam Helps Garchomp
QUIET = "move 2 -2, move 2 -1"

#: name -> (their team, setup turns, the compared turn, a Showdown log line that shows it).
CASES: dict[str, tuple] = {
    "roar-drags-the-first-on-the-bench": (
        _foes(), [], [f"move 1 1, {MINE}", QUIET], "|drag|p2a: Incineroar",
    ),
    "whirlwind-drags-the-first-on-the-bench": (
        _foes(), [], [f"move 2 1, {MINE}", QUIET], "|drag|p2a: Incineroar",
    ),
    "dragon-tail-drags-after-the-hit": (
        _foes(), [], [f"move 3 1, {MINE}", QUIET], "|drag|p2a: Incineroar",
    ),
    "circle-throw-drags-after-the-hit": (
        _foes(), [], [f"move 4 1, {MINE}", QUIET], "|drag|p2a: Incineroar",
    ),
    # Kangaskhan Roars too, slower: it is dragged out before its turn, and the Incineroar
    # that comes in does not act.
    "a-dragged-pokemon-loses-its-move": (
        _foes(), [], [f"move 1 1, {MINE}", "move 4 1, move 2 -1"], "|drag|p2a: Incineroar",
    ),
    "control-roar-into-guard-dog": (
        _foes(_mon("Mabosstiff", "Guard Dog", FILL, 0)), [], [f"move 1 1, {MINE}", QUIET],
        "|-activate|p2a: Mabosstiff|ability: Guard Dog",
    ),
    "control-roar-into-ingrain": (
        _foes(), [[f"move 3 1, {MINE}", "move 3, move 2 -1"]], [f"move 1 1, {MINE}", QUIET],
        "|-activate|p2a: Kangaskhan|move: Ingrain",
    ),
    "control-roar-with-no-one-to-come-in": (
        _foes(bench=False), [], [f"move 1 1, {MINE}", QUIET], "|-fail|p1a: Garchomp",
    ),
}

#: Cases whose HP is not compared, and why: only who stands where is.
HP_UNMODELLED = {
    # Ingrain's `onResidual` heals 1/16, which neither engine does (it is only a trap here).
    "control-roar-into-ingrain": "ingrain heal",
}

#: Showdown's pins and a pinned roll; chance is not branched, so the draw takes the first.
BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)
#: The same with chance branched: the draw is one branch per bench Pokemon.
DRAWN = replace(BUDGET, enumerate_secondary=True)


def _state(pos: Position) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for index, side in enumerate(pos.sides):
        for slot, at in enumerate(side.active):
            mon = side.pokemon[at] if at is not None else None
            out[f"p{index + 1}.{slot}"] = None if mon is None else (
                mon.species, mon.hp, tuple(sorted((k, v) for k, v in mon.boosts.items() if v))
            )
        for mon in side.pokemon:
            out[f"p{index + 1}.{mon.species}.hp"] = mon.hp
    return out


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict]:
    theirs, setup, choices, shown = CASES[name]
    # A bench of our own, so a Roar at Garchomp would have someone to drag in.
    ours = [USER, PARTNER, _mon("Metagross", "Clear Body", FILL, 4)]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy())
    handle.step(["team 123", "team 1234" if len(theirs) > 2 else "team 12"])
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
    return before, choices, after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [next(a for a in side_actions(reg, pos, s) if a.to_choice() == choices[s]) for s in (0, 1)]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    before, choices, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) == 1 and not there.suspended, (there.branches, there.suspended)
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None
    ours = _state(picked.position)
    if name in HP_UNMODELLED:
        who = lambda state: {k: v[0] for k, v in state.items() if k.count('.') == 1 and v}  # noqa: E731
        assert who(ours) == who(theirs)
        return
    assert ours == theirs


@pytest.mark.parametrize("name", ["roar-drags-the-first-on-the-bench", "dragon-tail-drags-after-the-hit"])
def test_the_draw_is_one_branch_per_bench_pokemon(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    before, choices, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    there = node.resolve(before, actions, DRAWN)
    assert there is not None, "the port refused the turn"
    assert there.branches == pytest.approx([0.5, 0.5]), there.branches
    states = []
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, DRAWN, select=index)
        assert picked is not None and picked.position is not None
        states.append(_state(picked.position))
    assert theirs in states
    assert {s["p2.0"][0] for s in states} == {"incineroar", "sylveon"}
