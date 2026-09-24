"""Brick Break, Psychic Fangs, Poltergeist, Bug Bite and Pluck in the port, played against
Showdown (IKA-240).

Showdown a5df827, data/moves.ts (the champions mod overrides none of them)::

    brickbreak / psychicfangs: {
        onTryHit(pokemon) {
            // will shatter screens through sub, before you hit
            pokemon.side.removeSideCondition('reflect');
            pokemon.side.removeSideCondition('lightscreen');
            pokemon.side.removeSideCondition('auroraveil');
        },
    },
    poltergeist: {
        onTry(source, target) { return !!target.item; },
    },
    bugbite / pluck: {
        onHit(target, source, move) {
            const item = target.getItem();
            if (source.hp && item.isBerry && target.takeItem(source)) {
                ...
                if (this.singleEvent('Eat', item, target.itemState, source, source, move)) { ... }
            }
        },
    },

The move's own `onTryHit` runs in `spreadMoveHit`, after Protect, the type immunity and the
accuracy and before the Substitute and the damage; `onTry` runs before any hit step;
`onHit` runs after the damage and before the `Update` at which a hit target eats its own
Sitrus Berry. Sticky Hold keeps the berry unless its holder fainted or the user has Mold
Breaker. The port hit all five as plain moves, noted `damaging move: <id>` since IKA-213.
Each case plays in Showdown and holds the port's one outcome to it; the positive control is
the exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"substitute", "confusion"})
SCREENS = frozenset({"reflect", "lightscreen", "auroraveil"})
IDS = ("brickbreak", "psychicfangs", "poltergeist", "bugbite", "pluck")


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user: 1 Brick Break, 2 Psychic Fangs, 3 Poltergeist, 4 Protect (Bug Bite and Pluck by `user_moves`).
USER = ["brickbreak", "psychicfangs", "poltergeist", "protect"]
BITER = ["bugbite", "pluck", "protect", "substitute"]
#: The partner: 1 Protect, 2 Knock Off, 3 Reflect, 4 Will-O-Wisp.
PARTNER = ["protect", "knockoff", "reflect", "willowisp"]
#: The foes: A 1 Reflect, 2 Dragon Claw, 3 Protect, 4 Substitute; B 1 Reflect, 2 Light Screen,
#: 3 Protect, 4 Will-O-Wisp.
FOE_A = ["reflect", "dragonclaw", "protect", "substitute"]
FOE_B = ["reflect", "lightscreen", "protect", "willowisp"]


def _teams(
    user: str = "Kangaskhan",
    user_ability: str = "Inner Focus",
    user_moves: list[str] | None = None,
    user_spe: int = 0,
    foe_a: str = "Garchomp",
    foe_a_ability: str = "Rough Skin",
    foe_a_item: str | None = None,
    foe_a_moves: list[str] | None = None,
    foe_b: str = "Incineroar",
    foe_b_ability: str = "Blaze",
    foe_b_moves: list[str] | None = None,
    **user_sp: int,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon(user, user_ability, user_moves or USER, user_spe, atk=32, **user_sp),
        _mon("Swampert", "Damp", PARTNER, 20, hp=32, **{"def": 20, "spd": 20}),
    ]
    theirs = [
        _mon(foe_a, foe_a_ability, foe_a_moves or FOE_A, 10, foe_a_item),
        _mon(foe_b, foe_b_ability, foe_b_moves or FOE_B, 10),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


BREAK, FANGS, POLTER, PROTECT = "move 1 1", "move 2 1", "move 3 1", "move 4"
BITE, PLUCK = "move 1 1", "move 2 1"
#: The user's move beside a Protecting partner.
def _me(move: str) -> str:
    return f"{move}, move 1"


#: Setup, mine: the user's move 4 (Protect, or the biter's Substitute) and the partner's Reflect
#: -- not its Protect, which would leave the compared turn's Protect a 1-in-3.
MINE = "move 4, move 3"
#: Setup: foe B puts up Reflect (A claws the partner), or A Reflect and B Light Screen.
REFLECT_UP = [MINE, "move 2 2, move 1"]
BOTH_UP = [MINE, "move 1, move 2"]
#: The compared turn's foes: A claws the Protecting partner, B Protects.
A_STAYS = "move 2 2, move 3"
#: Setup: the user and foe A put up Substitutes (B Reflect).
A_SUB = [MINE, "move 4, move 1"]
#: Setup: the biter's Substitute costs it a quarter, so a Sitrus Berry's heal shows.
HURT = [MINE, "move 2 2, move 2"]

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- Brick Break and Psychic Fangs
    "brick-break-breaks-reflect": (
        _teams(), [REFLECT_UP], [_me(BREAK), A_STAYS], "|-sideend|p2: ",
    ),
    "brick-break-breaks-both-screens": (
        _teams(), [BOTH_UP], [_me(BREAK), A_STAYS], "|-sideend|p2: ",
    ),
    "psychic-fangs-breaks-reflect": (
        _teams(foe_a="Machamp", foe_a_ability="No Guard"), [REFLECT_UP], [_me(FANGS), A_STAYS],
        "|-sideend|p2: ",
    ),
    "psychic-fangs-breaks-aurora-veil": (
        _teams(foe_b="Ninetales-Alola", foe_b_ability="Snow Warning",
               foe_b_moves=["auroraveil", "lightscreen", "protect", "willowisp"]),
        [[MINE, "move 3, move 1"]], [_me(FANGS), A_STAYS], "Aurora Veil",
    ),
    "brick-break-breaks-reflect-through-a-substitute": (
        _teams(), [A_SUB], [_me(BREAK), A_STAYS], "|-sideend|p2: ",
    ),
    "control-brick-break-into-protect": (
        _teams(), [REFLECT_UP], [_me(BREAK), "move 3, move 3"], "|-activate|p2a: Garchomp|move: Protect",
    ),
    "control-brick-break-into-a-ghost": (
        _teams(foe_a="Dragapult", foe_a_ability="Clear Body"), [REFLECT_UP], [_me(BREAK), A_STAYS],
        "|-immune|p2a: Dragapult",
    ),
    "control-psychic-fangs-into-a-dark-type": (
        _teams(foe_a="Tyranitar", foe_a_ability="Unnerve"), [REFLECT_UP], [_me(FANGS), A_STAYS],
        "|-immune|p2a: Tyranitar",
    ),
    "control-brick-break-with-no-screen": (
        _teams(), [], [_me(BREAK), A_STAYS], "|-damage|p2a: Garchomp",
    ),
    # -- Poltergeist
    "poltergeist-fails-into-no-item": (
        _teams(), [], [_me(POLTER), A_STAYS], "|-fail|p1a: Kangaskhan",
    ),
    "poltergeist-fails-after-knock-off": (
        # Snorlax, slower than the partner's Knock Off.
        _teams(user="Snorlax", user_ability="Thick Fat", foe_a_item="Leftovers"), [],
        [f"{POLTER}, move 2 1", A_STAYS], "|-fail|p1a: Snorlax",
    ),
    "poltergeist-fails-into-protect-with-no-item": (
        _teams(), [], [_me(POLTER), "move 3, move 3"], "|-fail|p1a: Kangaskhan",
    ),
    "control-poltergeist-into-an-item": (
        _teams(foe_a_item="Leftovers"), [], [_me(POLTER), A_STAYS],
        "|-activate|p2a: Garchomp|move: Poltergeist",
    ),
    # -- Bug Bite and Pluck (the user took a Will-O-Wisp burn or a hit first where it matters)
    "bug-bite-eats-a-sitrus-berry": (
        _teams(user_moves=BITER, foe_a_item="Sitrus Berry"),
        [HURT],
        [_me(BITE), A_STAYS], "[from] stealeat",
    ),
    "pluck-eats-a-sitrus-berry": (
        _teams(user_moves=BITER, foe_a_item="Sitrus Berry"),
        [HURT],
        [_me(PLUCK), A_STAYS], "[from] stealeat",
    ),
    "bug-bite-takes-the-berry-before-the-target-eats-it": (
        _teams(user_moves=BITER, foe_a="Meowscarada", foe_a_ability="Overgrow", foe_a_item="Sitrus Berry"),
        [], [_me(BITE), A_STAYS], "[from] stealeat",
    ),
    # Incineroar's Fake Out takes a fifth first; `takeItem` does not look at the HP.
    "bug-bite-takes-the-berry-of-a-target-it-knocks-out": (
        _teams(user_moves=BITER, foe_a="Meowscarada", foe_a_ability="Overgrow", foe_a_item="Sitrus Berry",
               foe_b_moves=["reflect", "lightscreen", "protect", "fakeout"]),
        [[MINE, "move 2 2, move 4 -1"]], [_me(BITE), A_STAYS], "|faint|p2a: Meowscarada",
    ),
    "bug-bite-eats-a-lum-berry-and-cures-the-burn": (
        _teams(user_moves=BITER, foe_a_item="Lum Berry"),
        # The partner burns the user, which Plucks foe B (no item).
        [["move 2 2, move 4 -1", "move 2 2, move 1"]],
        [_me(BITE), A_STAYS], "[from] stealeat",
    ),
    "bug-bite-takes-from-sticky-hold-with-mold-breaker": (
        _teams(user_moves=BITER, user_ability="Mold Breaker", foe_a_ability="Sticky Hold",
               foe_a_item="Sitrus Berry"),
        [], [_me(BITE), A_STAYS], "[from] stealeat",
    ),
    "control-bug-bite-into-sticky-hold": (
        _teams(user_moves=BITER, foe_a_ability="Sticky Hold", foe_a_item="Sitrus Berry"),
        [], [_me(BITE), A_STAYS], "|-activate|p2a: Garchomp|ability: Sticky Hold",
    ),
    "control-bug-bite-into-a-substitute": (
        _teams(user_moves=BITER, foe_a_item="Sitrus Berry"), [A_SUB], [_me(BITE), A_STAYS],
        "|-activate|p2a: Garchomp|move: Substitute|[damage]",
    ),
    "control-bug-bite-into-leftovers": (
        _teams(user_moves=BITER, foe_a_item="Leftovers"), [], [_me(BITE), A_STAYS], "|-damage|p2a: Garchomp",
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
            mon.status if not mon.fainted else None,
            mon.item,
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            mon.move_last_turn_failed if not mon.fainted else None,
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }
    for index, side in enumerate(pos.sides):
        out[f"p{index + 1}.screens"] = tuple(sorted(c.id for c in side.side_conditions if c.id in SCREENS))
    return out


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
    if not any(line.startswith("|turn|") for line in log):
        # Waiting on a replacement, Showdown has not reached `nextTurn`, which is where
        # `moveLastTurnResult` takes this turn's result; the port's position already has it.
        after = {
            key: (*value[:-1], "not yet") if not key.endswith(".screens") else value
            for key, value in after.items()
        }
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"
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
    if any(value[-1] == "not yet" for key, value in theirs.items() if not key.endswith(".screens")):
        rust_now = {
            key: (*value[:-1], "not yet") if not key.endswith(".screens") else value
            for key, value in rust_now.items()
        }
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
    # The berries here are all ones `move_hooks::eat` implements. (The move's own note is
    # `test_the_moves_are_no_longer_reported`, so a control reads the same on the old exe.)
    notes = [n for n in there.unmodelled if n.startswith("berry eaten")]
    assert notes == [], notes


@pytest.mark.parametrize("move", IDS)
def test_the_moves_are_no_longer_reported(reg, oracle: Oracle, bridged: None, move: str) -> None:  # noqa: ANN001
    """`modelled::damaging_move_is_unmodelled` no longer lists them (`tools/port_coverage.py`
    reads the names in `move_hooks.rs`), so using one notes nothing."""
    mine = [_mon("Kangaskhan", "Inner Focus", [move, "protect"], 10, atk=32),
            _mon("Swampert", "Damp", ["protect", "earthquake"], 0)]
    theirs = [_mon("Garchomp", "Rough Skin", FOE_A, 32, "Leftovers"), _mon("Incineroar", "Blaze", FOE_B, 20)]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    result = resolve_turn(reg, before, _actions(reg, before, ["move 1 1, move 1", A_STAYS]), budget=BUDGET)
    assert f"damaging move: {move}" not in result.unmodelled, result.unmodelled
