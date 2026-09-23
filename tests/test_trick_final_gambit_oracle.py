"""Trick, Switcheroo and Final Gambit in the port, played against Showdown (IKA-208).

The port refused all three -- Trick and Switcheroo as "status move" (Python models them),
Final Gambit as "move field selfdestruct" -- and Python filled the cells: 948 and 1,308 of
the 2,256 refused cells in the M-C g600 sample, the only two reasons left there. Python is
not the reference here and is not asserted: it swaps any two items but a mega stone, and
it never faints Final Gambit's user (Python has no `selfdestruct` at all).

`data/moves.ts`, trick (switcheroo is the same body):

    onTryImmunity(target) { return !target.hasAbility('stickyhold'); },
    onHit(target, source, move) {
        const yourItem = target.takeItem(source);
        const myItem = source.takeItem();
        if (yourItem === false || myItem === false || (!yourItem && !myItem)) { ... return false; }
        if ((myItem && !this.singleEvent('TakeItem', myItem, ..., target, source, move, myItem)) ||
            (yourItem && !this.singleEvent('TakeItem', yourItem, ..., source, target, move, yourItem))) {
            ... return false;
        }
        ... target.setItem(myItem) ... source.setItem(yourItem) ...

`takeItem` runs `TakeItem` on the holder, and a mega stone answers it with
`onTakeItem(item, source) { return !item.megaStone?.[source.baseSpecies.baseSpecies]; }`
(data/items.ts); the `singleEvent` asks the same of each item's *receiver*. `setItem` runs
the new item's `onStart` (Choice Scarf: `removeVolatile('choicelock')`; White Herb: use at
once on a lowered stat), and a berry is eaten at the next `Update`.

finalgambit:

    damageCallback(pokemon) { const damage = pokemon.hp; pokemon.faint(); return damage; },
    selfdestruct: "ifHit",

`getDamage` (sim/battle-actions.ts) runs the immunity first and only then the callback, so
a Ghost or a Protect leaves the user standing; a doll's `onTryPrimaryHit` calls `getDamage`
too, so the user faints into a Substitute as well.

Each case plays in Showdown, and the port resolves the compared turn from Showdown's
position before it under a budget with one outcome; HP, status, item, the volatiles named
below and the boosts of every Pokemon are held to Showdown's. The positive control is the
exe before this change, which refuses every one of these turns (the test fails with "the
port refused the turn"): `POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

#: The volatiles a case can move. Showdown's position carries others (`stall` after a
#: Protect) that these turns do not touch.
VOLATILES = frozenset({"choicelock", "substitute", "unburden", "confusion"})


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: Alakazam: 1 Trick, 2 Protect, 3 Final Gambit, 4 Switcheroo.
USER_MOVES = ["trick", "protect", "finalgambit", "switcheroo"]
#: The partners and foes: 1 Protect, 2 Helping Hand, 3 Super Fang, 4 Substitute.
FILL = ["protect", "helpinghand", "superfang", "substitute"]


def _user(item: str | None, spe: int = 10) -> TeamSet:
    return _mon("Alakazam", "Inner Focus", USER_MOVES, spe, item)


def _teams(
    user_item: str | None,
    foe_item: str | None,
    foe_ability: str = "Rough Skin",
    foe: str = "Garchomp",
    user_spe: int = 10,
    foe_spe: int = 0,
    partner_moves: list[str] | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _user(user_item, user_spe),
        _mon("Kangaskhan", "Inner Focus", partner_moves or FILL, 20, atk=32),
    ]
    theirs = [
        _mon(foe, foe_ability, FILL, foe_spe, foe_item),
        _mon("Incineroar", "Blaze", FILL, 0),
        _mon("Milotic", "Marvel Scale", FILL, 0),
    ]
    return mine, theirs


QUIET = "move 2 -2, move 2 -1"
TRICK = "move 1 1, move 2 -1"
SWITCHEROO = "move 4 1, move 2 -1"
GAMBIT = "move 3 1, move 2 -1"
#: Kangaskhan Super Fangs the foe in slot a while Alakazam Protects.
FANG = "move 2, move 3 1"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    "trick-swaps-scarf-and-leftovers": (
        _teams("Choice Scarf", "Leftovers"), [], [TRICK, QUIET],
        "|-item|p2a: Garchomp|Choice Scarf|[from] move: Trick",
    ),
    "switcheroo-swaps-scarf-and-leftovers": (
        _teams("Choice Scarf", "Leftovers"), [], [SWITCHEROO, QUIET],
        "|-item|p2a: Garchomp|Choice Scarf|[from] move: Switcheroo",
    ),
    # Garchomp moves first and locks itself into Helping Hand; it gives the Scarf away, and the
    # lock goes at the end of the turn. Alakazam's new Scarf locks nothing yet.
    "trick-takes-a-locked-scarf": (
        _teams("Leftovers", "Choice Scarf", foe_spe=32, user_spe=0), [], [TRICK, QUIET],
        "|-item|p1a: Alakazam|Choice Scarf|[from] move: Trick",
    ),
    # Both hold a Scarf and Garchomp moves first: each receiver's Scarf `onStart` drops the
    # lock it had, which the end of the turn would not (the item is still a Choice item).
    "trick-swapping-two-scarves-unlocks-both": (
        _teams("Choice Scarf", "Choice Scarf", foe_spe=32, user_spe=0), [], [TRICK, QUIET],
        "|-item|p1a: Alakazam|Choice Scarf|[from] move: Trick",
    ),
    "trick-with-nothing-takes-an-item": (
        _teams(None, "Leftovers"), [], [TRICK, QUIET], "|-item|p1a: Alakazam|Leftovers|[from] move: Trick",
    ),
    "trick-gives-an-item-for-nothing": (
        _teams("Leftovers", None), [], [TRICK, QUIET], "|-item|p2a: Garchomp|Leftovers|[from] move: Trick",
    ),
    # A stone for neither species moves both ways; Python refuses every mega stone.
    "trick-hands-over-a-foreign-stone": (
        _teams("Gengarite", "Leftovers"), [], [TRICK, QUIET],
        "|-item|p2a: Garchomp|Gengarite|[from] move: Trick",
    ),
    "trick-fails-on-the-targets-own-stone": (
        _teams("Leftovers", "Garchompite"), [], [TRICK, QUIET], "|-fail|p1a: Alakazam",
    ),
    "trick-fails-to-hand-a-stone-to-its-species": (
        _teams("Garchompite", "Leftovers"), [], [TRICK, QUIET], "|-fail|p1a: Alakazam",
    ),
    # Mega-evolved, Garchomp-Mega's base species is still Garchomp.
    "trick-fails-on-a-mega-evolved-holder": (
        _teams("Leftovers", "Garchompite"), [["move 2, move 2 -1", "move 2 -2 mega, move 2 -1"]],
        [TRICK, QUIET], "|-fail|p1a: Alakazam",
    ),
    # Floettite has its own `onTakeItem`, on `source.baseSpecies.name`: Floette-Eternal's base
    # species is Floette, which the other stones' `megaStone[baseSpecies.baseSpecies]` would
    # miss, and it keeps its stone all the same.
    "trick-fails-on-floette-eternals-floettite": (
        _teams("Leftovers", "Floettite", foe="Floette-Eternal", foe_ability="Flower Veil"), [],
        [TRICK, QUIET], "|-fail|p1a: Alakazam",
    ),
    # And a Meowsticite moves between two species it has nothing to do with.
    "trick-hands-over-a-meowsticite": (
        _teams("Meowsticite", "Leftovers"), [], [TRICK, QUIET],
        "|-item|p2a: Garchomp|Meowsticite|[from] move: Trick",
    ),
    "trick-fails-with-nothing-on-either-side": (
        _teams(None, None), [], [TRICK, QUIET], "|-fail|p1a: Alakazam",
    ),
    "trick-into-sticky-hold": (
        _teams("Choice Scarf", "Leftovers", foe_ability="Sticky Hold"), [], [TRICK, QUIET],
        "|-immune|p2a: Garchomp",
    ),
    # Two Super Fangs leave Garchomp near a quarter; the Sitrus it is handed is eaten at once.
    "trick-hands-a-sitrus-below-half": (
        _teams("Sitrus Berry", "Leftovers"), [[FANG, QUIET], [FANG, QUIET]], [TRICK, QUIET],
        "|-enditem|p2a: Garchomp|Sitrus Berry|[eat]",
    ),
    "trick-hands-a-white-herb-to-a-lowered-stat": (
        _teams("White Herb", "Leftovers", partner_moves=["protect", "helpinghand", "screech", "substitute"]),
        [["move 2, move 3 1", QUIET]], [TRICK, QUIET],
        "|-enditem|p2a: Garchomp|White Herb",
    ),
    "control-trick-into-protect": (
        _teams("Choice Scarf", "Leftovers"), [], [TRICK, "move 1, move 2 -1"],
        "|-activate|p2a: Garchomp|move: Protect",
    ),
    "final-gambit-faints-its-user": (
        _teams(None, None), [], [GAMBIT, QUIET], "|faint|p1a: Alakazam",
    ),
    "final-gambit-into-a-substitute": (
        _teams(None, None), [["move 2, move 2 -1", "move 4, move 2 -1"]], [GAMBIT, QUIET],
        "|-end|p2a: Garchomp|Substitute",
    ),
    "final-gambit-into-a-focus-sash": (
        _teams(None, "Focus Sash", foe="Alakazam", foe_ability="Inner Focus"), [], [GAMBIT, QUIET],
        "|-enditem|p2a: Alakazam|Focus Sash",
    ),
    "control-final-gambit-into-a-ghost": (
        _teams(None, None, foe="Gengar", foe_ability="Cursed Body"), [], [GAMBIT, QUIET],
        "|-immune|p2a: Gengar",
    ),
    "control-final-gambit-into-protect": (
        _teams(None, None), [], [GAMBIT, "move 1, move 2 -1"], "|-activate|p2a: Garchomp|move: Protect",
    ),
    "control-no-trick": (
        _teams("Choice Scarf", "Leftovers"), [], ["move 2, move 2 -1", QUIET], "|move|p1a: Alakazam|Protect",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            mon.status if not mon.fainted else None,
            mon.item,
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict]:
    (mine, theirs), setup, choices, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
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
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
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
