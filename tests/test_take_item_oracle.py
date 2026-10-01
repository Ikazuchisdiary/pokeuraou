"""Knock Off, Thief, Covet, Magician, Pickpocket and Corrosive Gas in the port, played against
Showdown (IKA-214).

Showdown a5df827 (the champions mod overrides none of them). Every route that takes an item
off a Pokemon goes through `Pokemon#takeItem(source)` (sim/pokemon.ts), which runs the
`TakeItem` event on the holder: Sticky Hold answers it (data/abilities.ts)

    onTakeItem(item, pokemon, source) {
        if (!pokemon.hp || pokemon.item === 'stickybarb') return;
        if ((source && source !== pokemon) || this.activeMove.id === 'knockoff') {
            this.add('-activate', pokemon, 'ability: Sticky Hold'); return false;
        }
    },   // flags: { breakable: 1 }

and a mega stone answers it (data/items.ts, 81 items in Reg M-C)

    onTakeItem(item, source) { return !item.megaStone?.[source.baseSpecies.baseSpecies]; },

where `source` is the *holder* (`runEvent` passes the target first), and `baseSpecies` is the
species the Pokemon had before it Mega Evolved, so a Garchomp-Mega still keeps Garchompite.
(Floettite and Meowsticite read `baseSpecies.name`.) The routes:

* knockoff: `onBasePower` asks `singleEvent('TakeItem', item, ..., target, target)` -- the
  item's own handler only, never Sticky Hold -- and boosts by 1.5 when the item may leave;
  `onAfterHit` is `target.takeItem()`, so Sticky Hold (not for a Mold Breaker user, not for a
  fainted holder) and the stone both stop it.
* thief / covet: `onAfterHit` -- nothing when the user already holds an item;
  `target.takeItem(source)`; then `singleEvent('TakeItem', yourItem, ..., source, target)`,
  the item's handler asked about the *receiver*, and `source.setItem(yourItem)`, which fails
  for a fainted user. When either refuses, `target.item = yourItem.id` puts it back.
* magician: `onAfterMoveSecondarySelf` -- a user holding nothing takes the first hit
  target's item (`pokemon.takeItem(source)`, then `source.setItem`) after a damaging move.
* pickpocket: `onAfterMoveSecondary` -- a holder of nothing hit by a contact move takes the
  attacker's item (`source.takeItem(target)`).
* corrosivegas: `onHit` is `target.takeItem(source)` on every adjacent Pokemon, the ally too.

Each case plays in Showdown, and the port resolves the compared turn from Showdown's
position before it under a budget with one outcome; HP, status, item and the volatiles named
below are held to Showdown's. The positive control is the exe before the change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
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

VOLATILES = frozenset({"substitute", "confusion", "unburden"})


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user: 1 Knock Off, 2 Thief, 3 Covet, 4 Splash.
USER = ["knockoff", "thief", "covet", "splash"]
#: The partner: 1 Protect, 2 Splash, 3 Reflect, 4 Will-O-Wisp.
PARTNER = ["protect", "splash", "reflect", "willowisp"]
#: Foe A: 1 Splash, 2 Dragon Claw, 3 Protect, 4 Substitute.
FOE_A = ["splash", "dragonclaw", "protect", "substitute"]
#: Foe B: 1 Protect, 2 Super Fang, 3 Splash, 4 Will-O-Wisp.
FOE_B = ["protect", "superfang", "splash", "willowisp"]


def _teams(
    user_moves: list[str] | None = None,
    user: str = "Kangaskhan",
    user_ability: str = "Inner Focus",
    user_item: str | None = None,
    foe: str = "Garchomp",
    foe_ability: str = "Rough Skin",
    foe_item: str | None = None,
    foe_sp: dict[str, int] | None = None,
    partner_item: str | None = None,
    foe_b_item: str | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon(user, user_ability, user_moves or USER, 0, user_item, atk=32),
        _mon("Swampert", "Damp", PARTNER, 20, partner_item, hp=32, **{"def": 20, "spd": 20}),
    ]
    theirs = [
        _mon(foe, foe_ability, FOE_A, 10, foe_item, **(foe_sp or {})),
        _mon("Incineroar", "Blaze", FOE_B, 5, foe_b_item),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


KNOCK, THIEF, COVET = "move 1 1, move 1", "move 2 1, move 1", "move 3 1, move 1"
#: Foe A claws the Protecting partner, foe B Protects.
HOLD = "move 2 2, move 1"
#: A setup turn: nothing happens (the partner Splashes, so the compared turn's Protect works).
IDLE = ["move 4, move 2", "move 1, move 1"]
#: A setup turn: foe A Mega Evolves behind a Protect.
MEGA = ["move 4, move 2", "move 3 mega, move 3"]
#: A setup turn: foe B Super Fangs its partner, foe A.
FANG_PARTNER = ["move 4, move 2", "move 1, move 2 -1"]
#: A setup turn: foe A puts up a Substitute.
SUB = ["move 4, move 2", "move 4, move 3"]
#: A setup turn: foe B Super Fangs the user, which Splashes.
FANG = ["move 4, move 2", "move 1, move 2 1"]

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- Knock Off
    "control-knock-off-takes-leftovers": (
        _teams(foe_item="Leftovers"),
        [],
        [KNOCK, HOLD],
        "|-enditem|p2a: Garchomp|Leftovers|[from] move: Knock Off",
    ),
    "knock-off-into-sticky-hold": (
        _teams(foe_item="Leftovers", foe_ability="Sticky Hold"),
        [],
        [KNOCK, HOLD],
        "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "knock-off-takes-from-sticky-hold-with-mold-breaker": (
        _teams(user_ability="Mold Breaker", foe_item="Leftovers", foe_ability="Sticky Hold"),
        [],
        [KNOCK, HOLD],
        "|-enditem|p2a: Garchomp|Leftovers|[from] move: Knock Off",
    ),
    "knock-off-into-sticky-hold-that-faints": (
        _teams(foe="Gengar", foe_item="Leftovers", foe_ability="Sticky Hold", foe_sp={"hp": 0, "def": 0}),
        [],
        [KNOCK, HOLD],
        "|faint|p2a: Gengar",
    ),
    "knock-off-leaves-a-stone-with-its-holder": (
        _teams(foe_item="Garchompite"),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    # Mega-evolved, Garchomp-Mega's base species is still Garchomp: the stone stays, and the
    # move gets no 1.5x (`mon.species` is Garchomp-Mega).
    "knock-off-leaves-a-mega-evolved-holders-stone": (
        _teams(foe_item="Garchompite"),
        [MEGA],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    "knock-off-leaves-charizard-mega-xs-stone": (
        _teams(foe="Charizard", foe_ability="Blaze", foe_item="Charizardite X"),
        [MEGA],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    "knock-off-takes-a-foreign-stone": (
        _teams(foe_item="Gengarite"),
        [],
        [KNOCK, HOLD],
        "|-enditem|p2a: Garchomp|Gengarite|[from] move: Knock Off",
    ),
    "knock-off-leaves-floette-eternals-stone": (
        _teams(foe="Floette-Eternal", foe_ability="Flower Veil", foe_item="Floettite"),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    "knock-off-leaves-meowstic-fs-stone": (
        _teams(foe="Meowstic-F", foe_ability="Keen Eye", foe_item="Meowsticite"),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    # -- Thief and Covet
    "control-thief-takes-leftovers": (
        _teams(foe_item="Leftovers"),
        [],
        [THIEF, HOLD],
        "|-item|p1a: Kangaskhan|Leftovers|[from] move: Thief",
    ),
    "control-covet-takes-leftovers": (
        _teams(foe_item="Leftovers"),
        [],
        [COVET, HOLD],
        "|-item|p1a: Kangaskhan|Leftovers|[from] move: Covet",
    ),
    "control-thief-takes-nothing-when-the-user-holds-an-item": (
        _teams(user_item="Sitrus Berry", foe_item="Leftovers"),
        [],
        [THIEF, HOLD],
        "|move|p1a: Kangaskhan|Thief",
    ),
    "thief-into-sticky-hold": (
        _teams(foe_item="Leftovers", foe_ability="Sticky Hold"),
        [],
        [THIEF, HOLD],
        "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "covet-into-sticky-hold": (
        _teams(foe_item="Leftovers", foe_ability="Sticky Hold"),
        [],
        [COVET, HOLD],
        "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "thief-takes-from-sticky-hold-with-mold-breaker": (
        _teams(user_ability="Mold Breaker", foe_item="Leftovers", foe_ability="Sticky Hold"),
        [],
        [THIEF, HOLD],
        "|-item|p1a: Kangaskhan|Leftovers|[from] move: Thief",
    ),
    "thief-leaves-a-mega-evolved-holders-stone": (
        _teams(foe_item="Garchompite"),
        [MEGA],
        [THIEF, HOLD],
        "|move|p1a: Kangaskhan|Thief",
    ),
    "covet-leaves-a-mega-evolved-holders-stone": (
        _teams(foe_item="Garchompite"),
        [MEGA],
        [COVET, HOLD],
        "|move|p1a: Kangaskhan|Covet",
    ),
    "thief-takes-a-foreign-stone": (
        _teams(foe_item="Gengarite"),
        [],
        [THIEF, HOLD],
        "|-item|p1a: Kangaskhan|Gengarite|[from] move: Thief",
    ),
    # The foe's Kangaskhanite cannot be handed to a Kangaskhan: the item's own handler is
    # asked about the receiver, and the foe keeps it.
    "thief-cannot-hand-a-stone-to-its-species": (
        _teams(foe="Kangaskhan", foe_ability="Early Bird", foe_item="Kangaskhanite"),
        [],
        [THIEF, HOLD],
        "|move|p1a: Kangaskhan|Thief",
    ),
    "covet-cannot-hand-a-stone-to-its-species": (
        _teams(foe="Kangaskhan", foe_ability="Early Bird", foe_item="Kangaskhanite"),
        [],
        [COVET, HOLD],
        "|move|p1a: Kangaskhan|Covet",
    ),
    # Rough Skin takes the user's last HP before the Thief's `onAfterHit`: `setItem` fails for a
    # fainted user and the item goes back to Garchomp.
    "thief-by-a-user-that-faints-on-rough-skin": (
        _teams(foe_item="Leftovers"),
        [FANG, FANG, FANG],
        [THIEF, HOLD],
        "|faint|p1a: Kangaskhan",
    ),
    # -- Magician
    "magician-takes-the-first-targets-item": (
        _teams(["bodyslam", "protect", "splash", "splash"], user_ability="Magician", foe_item="Leftovers"),
        [],
        [KNOCK, HOLD],
        "|-item|p1a: Kangaskhan|Leftovers|[from] ability: Magician",
    ),
    "magician-into-sticky-hold": (
        _teams(
            ["bodyslam", "protect", "splash", "splash"],
            user_ability="Magician",
            foe_item="Leftovers",
            foe_ability="Sticky Hold",
        ),
        [],
        [KNOCK, HOLD],
        "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "magician-holding-an-item-takes-nothing": (
        _teams(
            ["bodyslam", "protect", "splash", "splash"],
            user_ability="Magician",
            user_item="Sitrus Berry",
            foe_item="Leftovers",
        ),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Body Slam",
    ),
    "magician-leaves-a-mega-evolved-holders-stone": (
        _teams(["bodyslam", "protect", "splash", "splash"], user_ability="Magician", foe_item="Garchompite"),
        [MEGA],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Body Slam",
    ),
    "magician-hands-no-stone-to-its-species": (
        _teams(
            ["bodyslam", "protect", "splash", "splash"],
            user_ability="Magician",
            foe="Kangaskhan",
            foe_ability="Early Bird",
            foe_item="Kangaskhanite",
        ),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Body Slam",
    ),
    # -- Pickpocket
    "pickpocket-takes-a-contact-attackers-item": (
        _teams(user_item="Leftovers", foe_ability="Pickpocket"),
        [],
        [KNOCK, HOLD],
        "|-item|p2a: Garchomp|Leftovers|[from] ability: Pickpocket",
    ),
    "pickpocket-steals-once-knock-off-emptied-the-holder": (
        _teams(user_item="Leftovers", foe_ability="Pickpocket", foe_item="Sitrus Berry"),
        [],
        [KNOCK, HOLD],
        "|move|p1a: Kangaskhan|Knock Off",
    ),
    "pickpocket-into-sticky-hold": (
        _teams(
            ["splash", "protect", "bodyslam", "splash"],
            user_item="Leftovers",
            user_ability="Sticky Hold",
            foe_ability="Pickpocket",
        ),
        [],
        ["move 3 1, move 1", HOLD],
        "|-activate|p1a: Kangaskhan|ability: Sticky Hold",
    ),
    "pickpocket-cannot-take-the-users-own-stone": (
        _teams(
            ["splash", "protect", "bodyslam", "splash"], user_item="Kangaskhanite", foe_ability="Pickpocket"
        ),
        [],
        ["move 3 1, move 1", HOLD],
        "|move|p1a: Kangaskhan|Body Slam",
    ),
    "pickpocket-ignores-a-non-contact-hit": (
        _teams(
            ["splash", "protect", "earthquake", "splash"], user_item="Leftovers", foe_ability="Pickpocket"
        ),
        [],
        ["move 3, move 1", HOLD],
        "|move|p1a: Kangaskhan|Earthquake",
    ),
    # -- Behind a Substitute, a fainted holder, Unburden, and a spread move
    "knock-off-into-a-substitute": (
        _teams(foe_item="Leftovers"),
        [SUB],
        [KNOCK, HOLD],
        "|-end|p2a: Garchomp|Substitute",
    ),
    "thief-into-a-substitute-that-holds": (
        _teams(foe_item="Leftovers"),
        [SUB],
        [THIEF, HOLD],
        "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    ),
    "magician-into-a-substitute": (
        _teams(["bodyslam", "protect", "splash", "splash"], user_ability="Magician", foe_item="Leftovers"),
        [SUB],
        [KNOCK, HOLD],
        "|-end|p2a: Garchomp|Substitute",
    ),
    "pickpocket-behind-a-substitute-that-holds": (
        _teams(user_item="Leftovers", foe_ability="Pickpocket"),
        [SUB],
        [KNOCK, HOLD],
        "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    ),
    "pickpocket-behind-a-substitute-that-breaks": (
        _teams(["bodyslam", "protect", "splash", "splash"], user_item="Leftovers", foe_ability="Pickpocket"),
        [SUB],
        [KNOCK, HOLD],
        "|-end|p2a: Garchomp|Substitute",
    ),
    "magician-into-a-substitute-that-holds": (
        _teams(user_ability="Magician", foe_item="Leftovers"),
        [SUB],
        [THIEF, HOLD],
        "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    ),
    "thief-takes-a-fainted-holders-item": (
        _teams(foe="Alakazam", foe_ability="Inner Focus", foe_item="Leftovers", foe_sp={"hp": 0, "def": 0}),
        [FANG_PARTNER],
        [THIEF, HOLD],
        "|faint|p2a: Alakazam",
    ),
    "pickpocket-of-a-fainted-holder": (
        _teams(user_item="Leftovers", foe="Alakazam", foe_ability="Pickpocket", foe_sp={"hp": 0, "def": 0}),
        [],
        [KNOCK, HOLD],
        "|faint|p2a: Alakazam",
    ),
    "knock-off-starts-unburden": (
        _teams(foe_item="Leftovers", foe_ability="Unburden"),
        [],
        [KNOCK, HOLD],
        "|-enditem|p2a: Garchomp|Leftovers|[from] move: Knock Off",
    ),
    "thief-starts-unburden": (
        _teams(foe_item="Leftovers", foe_ability="Unburden"),
        [],
        [THIEF, HOLD],
        "|-item|p1a: Kangaskhan|Leftovers|[from] move: Thief",
    ),
    # Rock Slide hits both foes: Magician takes the faster one's item (Garchomp, 10 Speed
    # points against 5) and leaves the other.
    "magician-takes-only-the-first-targets-item": (
        _teams(
            ["rockslide", "protect", "splash", "splash"],
            user_ability="Magician",
            foe_item="Leftovers",
            foe_b_item="Sitrus Berry",
        ),
        [],
        ["move 1, move 1", "move 2 2, move 2 2"],
        "|-item|p1a: Kangaskhan|Leftovers|[from] ability: Magician",
    ),
    "magician-takes-the-second-target-when-the-first-holds-a-stone": (
        _teams(
            ["rockslide", "protect", "splash", "splash"],
            user_ability="Magician",
            foe_item="Garchompite",
            foe_b_item="Sitrus Berry",
        ),
        [MEGA],
        ["move 1, move 1", "move 2 2, move 2 2"],
        "|-item|p1a: Kangaskhan|Sitrus Berry|[from] ability: Magician",
    ),
    # -- Corrosive Gas (every adjacent Pokemon, the ally too)
    "corrosive-gas-takes-a-foes-item": (
        _teams(
            ["corrosivegas", "protect", "splash", "splash"], foe_item="Leftovers", partner_item="Sitrus Berry"
        ),
        [],
        ["move 1, move 3", HOLD],
        "|-enditem|p2a: Garchomp|Leftovers|[from] move: Corrosive Gas",
    ),
    "corrosive-gas-into-sticky-hold": (
        _teams(
            ["corrosivegas", "protect", "splash", "splash"], foe_item="Leftovers", foe_ability="Sticky Hold"
        ),
        [],
        ["move 1, move 3", HOLD],
        "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "corrosive-gas-leaves-a-mega-evolved-holders-stone": (
        _teams(["corrosivegas", "protect", "splash", "splash"], foe_item="Garchompite"),
        [MEGA],
        ["move 1, move 3", HOLD],
        "|-fail|p2a: Garchomp|move: Corrosive Gas",
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
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
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
            party.stats_override = None
    return before, choices, after, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


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
    before, choices, theirs, _log = _play(oracle, name)
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
