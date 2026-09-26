"""Ripen, Cheek Pouch and Gluttony where a berry is eaten, played against Showdown (IKA-329).

Showdown a5df827, data/abilities.ts (the champions mod overrides none of the three)::

    ripen: {
        onTryHeal(damage, target, source, effect) {
            if ((effect as Item).isBerry) return this.chainModify(2);
        },
        onSourceModifyDamagePriority: -1,
        onSourceModifyDamage(damage, source, target, move) {
            if (target.abilityState.berryWeaken) {
                target.abilityState.berryWeaken = false;
                return this.chainModify(0.5);
            }
        },
        onEatItem(item, pokemon) {
            pokemon.abilityState.berryWeaken = weakenBerries.includes(item.name);  // the resist berries
        },
    },
    cheekpouch: {
        onEatItem(item, pokemon) { this.heal(pokemon.baseMaxhp / 3); },
    },
    gluttony: { ... pokemon.abilityState.gluttony = true; },  // read by the 1/4 pinch berries

`Pokemon#eatItem` runs the berry's `Eat` and then `EatItem`, so Cheek Pouch heals after the
berry's own heal; a resist berry is eaten inside the hit's `ModifyDamage`, so its Cheek Pouch
heal lands before the hit's damage. Bug Bite and Pluck run both on the user. Gluttony changes
only the 1/4 pinch berries (Figy, Liechi, ...), none of which either regulation has
(`test_no_gluttony_berry_in_the_regulations`).

The port ignored all three (`inert.rs`), so it was silent whenever the holder ate a berry
without being hit -- poison, burn, sandstorm, Leech Seed, a status move and Lum Berry -- and
noted only when it was the defender or attacker of a hit. Each case plays in Showdown and holds
the port's one outcome to it; the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import regulation_dir

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"confusion", "leechseed"})

#: The holder: 1 Swords Dance, 2 Bug Bite, 3 Protect, 4 Substitute.
HOLDER = ["swordsdance", "bugbite", "protect", "substitute"]
#: The partner: 1 Helping Hand, 2 Protect.
PARTNER = ["helpinghand", "protect"]
#: Foe A: 1 Super Fang, 2 Toxic, 3 Will-O-Wisp, 4 Leech Seed.
FOE_A = ["superfang", "toxic", "willowisp", "leechseed"]
#: Foe B: 1 Helping Hand, 2 Sandstorm, 3 Confuse Ray, 4 Ice Beam.
FOE_B = ["helpinghand", "sandstorm", "confuseray", "icebeam"]


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, hp: int = 21
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": hp, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


def _teams(
    holder: str, ability: str, item: str | None, hp: int = 21, foe_a_item: str | None = None
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [_mon(holder, ability, HOLDER, 0, item, hp), _mon("Swampert", "Damp", PARTNER, 20)]
    theirs = [_mon("Garchomp", "Rough Skin", FOE_A, 31, foe_a_item), _mon("Incineroar", "Blaze", FOE_B, 30)]
    return mine, theirs


#: Setup: foe A's Super Fang leaves the holder at ceil(maxhp / 2) -- just above half for an
#: odd maxhp, so a pinch berry waits for the compared turn.
FANG = ["move 1, move 1 -1", "move 1 1, move 1 -1"]
IDLE = "move 1, move 1 -1"
TOXIC = [IDLE, "move 2 1, move 1 -1"]
WISP = [IDLE, "move 3 1, move 1 -1"]
SEED = [IDLE, "move 4 1, move 1 -1"]
SAND = [IDLE, "move 2 2, move 2"]
CONFUSE = [IDLE, "move 2 2, move 3 1"]
ICE = [IDLE, "move 2 2, move 4 1"]
BITE = ["move 2 1, move 1 -1", "move 2 2, move 1 -1"]

EAT = "[eat]"
STEAL = "[from] stealeat"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- A pinch berry eaten at the residual, with nobody hitting the holder
    "ripen-sitrus-toxic": (_teams("Flapple", "Ripen", "Sitrus Berry", hp=22), [FANG], TOXIC, EAT),
    "ripen-oran-burn": (_teams("Appletun", "Ripen", "Oran Berry", hp=22), [FANG], WISP, EAT),
    "cheekpouch-sitrus-sandstorm": (_teams("Dedenne", "Cheek Pouch", "Sitrus Berry"), [FANG], SAND, EAT),
    "cheekpouch-sitrus-leechseed": (_teams("Dedenne", "Cheek Pouch", "Sitrus Berry"), [FANG], SEED, EAT),
    "cheekpouch-oran-toxic": (_teams("Diggersby", "Cheek Pouch", "Oran Berry"), [FANG], TOXIC, EAT),
    # -- A cure berry eaten on a status move
    "cheekpouch-lum-toxic": (_teams("Dedenne", "Cheek Pouch", "Lum Berry"), [FANG], TOXIC, EAT),
    "cheekpouch-persim-confuse-ray": (_teams("Dedenne", "Cheek Pouch", "Persim Berry"), [FANG], CONFUSE, EAT),
    # -- A resist berry: Ripen halves again; Cheek Pouch heals before the damage, so at full
    # HP it heals nothing
    "ripen-yache-ice-beam": (_teams("Flapple", "Ripen", "Yache Berry"), [], ICE, EAT),
    "cheekpouch-yache-ice-beam-full-hp": (_teams("Diggersby", "Cheek Pouch", "Yache Berry"), [], ICE, EAT),
    "cheekpouch-yache-ice-beam-hurt": (_teams("Diggersby", "Cheek Pouch", "Yache Berry"), [FANG], ICE, EAT),
    # -- Bug Bite: the user's own Ripen and Cheek Pouch
    "ripen-bug-bite-sitrus": (
        _teams("Flapple", "Ripen", None, foe_a_item="Sitrus Berry"), [FANG], BITE, STEAL,
    ),
    "cheekpouch-bug-bite-sitrus": (
        _teams("Dedenne", "Cheek Pouch", None, foe_a_item="Sitrus Berry"), [FANG], BITE, STEAL,
    ),
    "cheekpouch-bug-bite-lum": (
        _teams("Dedenne", "Cheek Pouch", None, foe_a_item="Lum Berry"), [FANG], BITE, STEAL,
    ),
    # -- Gluttony: Sitrus Berry at half as without it
    "gluttony-sitrus-toxic": (_teams("Snorlax", "Gluttony", "Sitrus Berry", hp=22), [FANG], TOXIC, EAT),
    # -- Controls: the same berries without the abilities
    "control-sitrus-toxic": (_teams("Dedenne", "Plus", "Sitrus Berry"), [FANG], TOXIC, EAT),
    "control-lum-toxic": (_teams("Dedenne", "Plus", "Lum Berry"), [FANG], TOXIC, EAT),
    "control-yache-ice-beam": (_teams("Flapple", "Hustle", "Yache Berry"), [], ICE, EAT),
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
    # Showdown first: the compared turn is the one the berry is eaten on.
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
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
    # The port applies the three, so it says nothing of them (on a hit either).
    notes = [n for n in there.unmodelled if any(a in n for a in ("ripen", "cheekpouch", "gluttony"))]
    assert notes == [], notes


#: Showdown a5df827 data/items.ts: the berries whose `onUpdate`/`onResidual` reads
#: `pokemon.hasAbility('gluttony')` (the 1/4 pinch berries).
GLUTTONY_BERRIES = frozenset({
    "aguavberry", "apicotberry", "custapberry", "figyberry", "ganlonberry", "iapapaberry",
    "lansatberry", "liechiberry", "magoberry", "micleberry", "petayaberry", "salacberry",
    "starfberry", "wikiberry",
})


@pytest.mark.parametrize("fmt", ["gen9championsvgc2026regmb", "gen9championsvgc2026regmc"])
def test_no_gluttony_berry_in_the_regulations(fmt: str) -> None:
    """The port names Gluttony as doing nothing (`resolve::ability_handled`): true only while
    no berry it acts on can be held. A regulation that adds one must teach the port first."""
    dump = json.loads((regulation_dir() / f"{fmt}.json").read_text(encoding="utf-8"))
    items = {item["id"] if isinstance(item, dict) else item for item in dump["items"]}
    assert not items & GLUTTONY_BERRIES, sorted(items & GLUTTONY_BERRIES)
