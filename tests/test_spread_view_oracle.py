"""A spread move's damage is computed for every target before any is dealt (IKA-413).

Showdown a5df827, `sim/battle-actions.ts` `spreadMoveHit`::

    damage = this.getSpreadDamage(damage, targets, pokemon, move, moveData, isSecondary, isSelf);  // 1
    damage = this.battle.spreadDamage(damage, targets, pokemon, move);                              // 2
    damage = this.runMoveEffects(...);                                                              // 3
    ... selfDrops, secondaries, forceSwitch, then `runEvent('DamagingHit', damagedTargets, ...)`

So when the second target's damage is computed, the first has not been hit: it has its HP, and
everything a field-wide ability does to the second target's damage is still there -- Friend Guard
(`onAnyModifyDamage`), Fairy Aura / Dark Aura (`onAnyBasePower`), Aura Break
(`onAnyTryPrimaryHit`), Cloud Nine / Air Lock (`suppressWeather`), the Ruin abilities
(`onAnyModify*`; tested in test_ignored_abilities.py). `onAny*` handlers are looked for through
`Side.allies()`, which lists only Pokemon with HP, so a holder is gone from the first moment its
HP is 0: a user that explodes (`faint()` zeroes its HP before the hits) gives its blast no aura.
Unaware (`onAnyModifyBoost`) reads the move's `activeTarget`, so a knocked-out Unaware first
target changes nothing for the second: a control. And the first target's *hit* has not happened
either when the second is computed: its Rough Skin takes the user's HP only after the damage is
dealt to both, so a user the first target's Rough Skin knocks out still hits the second target.

The port hit the targets one by one: the first target's knock-out removed its ability from the
field and the effects of its hit were in the state when the second was computed. The positive
control is the exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
The cases that pass on both are the controls: the same Pokemon where the effect has nothing to
change (the holder is hit last, or lives, or the move is not a spread of two foes).

Not here: Sand Spit, Mummy and Wandering Spirit change the field or the user's ability after the
first target's hit and so, in Showdown, not the second target's damage; the port does not model
them (a note, `modelled.rs`), so there is nothing to compare.
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
    species: str, ability: str, moves: list[str], spe: int = 0, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user is the fastest: it moves first and the foes do nothing that shows (Calm Mind after it).
VOICE = ["hypervoice", "makeitrain", "nastyplot", "protect"]
GLEAM = ["dazzlinggleam", "muddywater", "nastyplot", "protect"]
SWIPE = ["breakingswipe", "rockslide", "swordsdance", "protect"]
BOOM = ["mistyexplosion", "protect", "nastyplot", "protect"]
IDLE_MOVES = ["protect", "calmmind", "swordsdance", "fakeout"]
PARTNER = ["protect", "helpinghand", "fakeout", "swordsdance"]
FANG = ["superfang", "calmmind", "protect", "swordsdance"]


def _user(
    moves: list[str] = VOICE, species: str = "Gholdengo", ability: str = "Illuminate", **sp: int
) -> TeamSet:
    return _mon(species, ability, moves, 32, "lifeorb", **({"spa": 32} | sp))


def _frail(ability: str, species: str = "Pikachu") -> TeamSet:
    """A first target that dies to the moves above (with the partner's Helping Hand)."""
    return _mon(species, ability, IDLE_MOVES, hp=0, spd=0, **{"def": 0})


def _sturdy(ability: str = "Immunity", species: str = "Snorlax") -> TeamSet:
    return _mon(species, ability, IDLE_MOVES, hp=32, spd=10)


def _rough(moves: list[str] = IDLE_MOVES) -> TeamSet:
    return _mon("Garchomp", "Rough Skin", moves, hp=32, **{"def": 32})


PELIPPER = _mon("Pelipper", "Drizzle", PARTNER, 10)
SLIDE = ["rockslide", "protect", "swordsdance", "fakeout"]
SLIDER = _mon("Garchomp", "Sand Veil", SLIDE, 32, "lifeorb", atk=32)


def _teams(a: TeamSet, b: TeamSet, user: TeamSet | None = None, partner: TeamSet | None = None):  # noqa: ANN202
    garchomp = _mon("Garchomp", "Sand Veil", PARTNER, 10)
    mine = [user or _user(), partner or garchomp, _mon("Milotic", "Marvel Scale", PARTNER)]
    theirs = [a, b, _mon("Milotic", "Marvel Scale", IDLE_MOVES)]
    return mine, theirs


#: p1 uses move N (a spread move, no target) with the partner's Helping Hand (so the frail first
#: target dies); p2 uses Calm Mind with both, after it.
MOVE1 = "move 1, move 2 -1"
MOVE2 = "move 2, move 2 -1"
FOES = "move 2, move 2"
NASTY_PLOT = ["move 3, move 1", FOES]
#: The turn Super Fang (Snorlax's move 1, at p1a) halves the user's HP while the user sets up.
FANGED = ["move 3, move 1", "move 2, move 1 1"]


def _case(teams: tuple, setup: list, choice: str, shown: str, victim: str) -> tuple:
    """teams, setup turns, my choice, a Showdown log line that shows the case, whose HP is the answer."""
    return (teams, setup, choice, shown, victim)


def _fainted_first(ability: str, **kw) -> tuple:  # noqa: ANN003
    """The frail first target (p2a) with `ability` is knocked out; Snorlax (p2b) is the answer."""
    teams = _teams(_frail(ability), _sturdy(), kw.pop("user", None), kw.pop("partner", None))
    return _case(teams, kw.pop("setup", []), kw.pop("choice", MOVE1), "|faint|p2a: Pikachu", "p2b: Snorlax")


def _alive_first(ability: str, species: str, **kw) -> tuple:  # noqa: ANN003
    """The first target (p2a) holds `ability` and lives; Snorlax (p2b) is the answer."""
    teams = _teams(_sturdy(ability, species), _sturdy(), kw.pop("user", None))
    return _case(teams, [], MOVE1, "|-damage|p2a: " + species, "p2b: Snorlax")


#: A Pikachu (110 HP) with Super Fang halving it each turn: 110, 55, 28, 14, 7; Rough Skin takes 13.
def _rough_skin(fangs: int, shown: str) -> tuple:
    snorlax = _mon("Snorlax", "Immunity", FANG, hp=32)
    teams = _teams(_rough(), snorlax, _mon("Pikachu", "Static", SWIPE, 32, atk=32))
    return _case(teams, [FANGED] * fangs, "move 1, move 1", shown, "p2b: Snorlax")


#: name -> `_case(...)`.
CASES: dict[str, tuple] = {
    # Friend Guard: the first target's knock-out leaves it up.
    "friend-guard-holder-hit-first-and-knocked-out": _fainted_first("Friend Guard"),
    "friend-guard-with-the-issue's-move": _fainted_first("Friend Guard", choice=MOVE2),
    "friend-guard-holder-hit-last-and-knocked-out": _case(
        _teams(_sturdy(), _frail("Friend Guard")), [], MOVE1, "|faint|p2b: Pikachu", "p2a: Snorlax"
    ),
    "control-friend-guard-holder-lives": _alive_first("Friend Guard", "Maushold"),
    "control-no-friend-guard": _fainted_first("Technician"),
    # Auras: Fairy Aura boosts and Aura Break turns it round, for a move the knocked-out holder was
    # in front of.
    "fairy-aura-holder-knocked-out-first": _fainted_first("Fairy Aura", user=_user(GLEAM)),
    "aura-break-holder-knocked-out-first": _case(
        _teams(_frail("Aura Break"), _sturdy("Fairy Aura"), _user(GLEAM)),
        [NASTY_PLOT], MOVE1, "|faint|p2a: Pikachu", "p2b: Snorlax",
    ),
    "control-fairy-aura-holder-lives": _alive_first("Fairy Aura", "Maushold", user=_user(GLEAM)),
    # Cloud Nine and Air Lock: rain (the partner's Drizzle) does nothing while one is on the field.
    "cloud-nine-holder-knocked-out-first": _fainted_first(
        "Cloud Nine", user=_user(GLEAM), partner=PELIPPER, choice=MOVE2
    ),
    "air-lock-holder-knocked-out-first": _fainted_first(
        "Air Lock", user=_user(GLEAM), partner=PELIPPER, choice=MOVE2
    ),
    "control-rain-without-cloud-nine": _fainted_first(
        "Technician", user=_user(GLEAM), partner=PELIPPER, choice=MOVE2
    ),
    # Unaware reads the move's target: the knocked-out Unaware first target changes nothing for the
    # second. (Mr. Mime: Make It Rain is super effective, the Unaware holder dies at +2 ignored.)
    "control-unaware-first-target-knocked-out-user-boosted": _case(
        _teams(_frail("Unaware", "Mr. Mime"), _sturdy()), [NASTY_PLOT], MOVE2,
        "|faint|p2a: Mr. Mime", "p2b: Snorlax",
    ),
    # A user that explodes has no HP (`faint()` zeroes it) and `Side.allies()`, where `onAny*`
    # handlers are looked for, lists the Pokemon with HP: its own aura gives its blast nothing.
    "control-misty-explosion-user-with-fairy-aura-gives-none": _case(
        _teams(_sturdy(), _sturdy(), _user(BOOM, ability="Fairy Aura")),
        [], "move 1, move 1", "|faint|p1a: Gholdengo", "p2b: Snorlax",
    ),
    "control-misty-explosion-user-without-an-aura": _case(
        _teams(_sturdy(), _sturdy(), _user(BOOM)),
        [], "move 1, move 1", "|faint|p1a: Gholdengo", "p2b: Snorlax",
    ),
    # Spicy Spray burns the user after the damage is dealt to both targets: the second target's
    # damage is not halved (a burn halves a physical hit), in Showdown.
    "spicy-spray-first-target-burns-the-user": _case(
        _teams(_sturdy("Spicy Spray", "Scovillain"), _sturdy(), SLIDER),
        [], "move 1, move 1", "|-status|p1a: Garchomp|brn|[from] ability: Spicy Spray", "p2b: Snorlax",
    ),
    "control-spicy-spray-last-target": _case(
        _teams(_sturdy(), _sturdy("Spicy Spray", "Scovillain"), SLIDER),
        [], "move 1, move 1", "|-status|p1a: Garchomp|brn|[from] ability: Spicy Spray", "p2a: Snorlax",
    ),
    # The first target's hit has not happened when the second is computed: its Rough Skin takes the
    # user's last HP after the damage is dealt to both, and the second target is still hit.
    "rough-skin-first-target-knocks-out-the-user": _rough_skin(4, "|faint|p1a: Pikachu"),
    "control-rough-skin-first-target-the-user-lives": _rough_skin(3, "|-damage|p1a: Pikachu|"),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.fainted, mon.ability)
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, teams: tuple, setup: list, choice: str) -> tuple[Position, dict, list[str]]:
    mine, theirs = teams
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step([choice, FOES])
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, after, log


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
    teams, setup, choice, shown, victim = CASES[name]
    before, theirs, log = _play(oracle, teams, setup, choice)
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"
    assert any(line.startswith("|-damage|" + victim) for line in log), f"{victim} was not hit: {log}"
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, [choice, FOES])
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None, f"the port refused the turn: {node.refusal}"
    rust_now = _state(picked.position)
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
