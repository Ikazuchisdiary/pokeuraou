"""Counter, Mirror Coat, Metal Burst, Comeuppance, Endeavor and Super Fang in the port,
played against Showdown (IKA-213).

Showdown a5df827, data/moves.ts (the champions mod overrides none of them)::

    counter: {
        damageCallback(pokemon) {
            if (!pokemon.volatiles['counter']) return 0;
            return pokemon.volatiles['counter'].damage || 1;
        },
        priority: -5,
        beforeTurnCallback(pokemon) { pokemon.addVolatile('counter'); },
        onTry(source) {
            if (!source.volatiles['counter']) return false;
            if (source.volatiles['counter'].slot === null) return false;
        },
        condition: {
            duration: 1,
            onStart(target, source, move) { this.effectState.slot = null; this.effectState.damage = 0; },
            onRedirectTargetPriority: -1,
            onRedirectTarget(target, source, source2, move) {
                if (move.id !== 'counter') return;
                if (source !== this.effectState.target || !this.effectState.slot) return;
                return this.getAtSlot(this.effectState.slot);
            },
            onDamagingHit(damage, target, source, move) {
                if (!source.isAlly(target) && this.getCategory(move) === 'Physical') {
                    this.effectState.slot = source.getSlot();
                    this.effectState.damage = 2 * damage;
                }
            },
        },
        target: "scripted",
    },
    mirrorcoat: the same, with 'Special'
    metalburst: {
        damageCallback(pokemon) {
            const lastDamagedBy = pokemon.getLastDamagedBy(true);
            if (lastDamagedBy !== undefined) return (lastDamagedBy.damage * 1.5) || 1;
            return 0;
        },
        onTry(source) {
            const lastDamagedBy = source.getLastDamagedBy(true);
            if (!lastDamagedBy?.thisTurn) return false;
        },
        onModifyTarget(targetRelayVar, source, target, move) {
            const lastDamagedBy = source.getLastDamagedBy(true);
            if (lastDamagedBy) targetRelayVar.target = this.getAtSlot(lastDamagedBy.slot);
        },
        target: "scripted",
    },
    comeuppance: the same body as metalburst (Dark, contact)
    endeavor: {
        damageCallback(pokemon, target) { return target.getUndynamaxedHP() - pokemon.hp; },
        onTryImmunity(target, pokemon) { return pokemon.hp < target.hp; },
    },
    superfang: {
        damageCallback(pokemon, target) { return this.clampIntRange(target.getUndynamaxedHP() / 2, 1); },
    },

`getLastDamagedBy(true)` is the last entry of `attackedBy` from a foe whose `damageValue` is
a number; `hitStepMoveHitLoop` pushes one per target at the end of a move with the last hit's
damage (sim/battle-actions.ts:990). `getDamage` runs the callback after the type immunity
and before any modifier; `spreadDamage` floors a fraction (`clampIntRange(damage, 1)`). A
target that fainted is retargeted in `getMoveTargets` before `RedirectTarget` runs, so Metal
Burst hits the other foe while Counter's own redirect sends it back to the fainted one and
it fails; Follow Me (`onFoeRedirectTargetPriority: 1`) outranks Counter's (-1).

The port gave all four reply moves a base power of 0 and hit the first foe for nothing,
with the note `move.basePowerCallback:<id>`; Endeavor at no lower HP was a 0-damage hit
rather than an immunity. Each case plays in Showdown and holds the port's one outcome to
it; the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import PassAction, SideAction, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn, resume_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

VOLATILES = frozenset({"substitute", "confusion"})
#: Notes the port gave these moves before it modelled them.
REPLY_IDS = ("counter", "mirrorcoat", "metalburst", "comeuppance", "endeavor", "superfang")


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: Swampert: 1 Counter, 2 Mirror Coat, 3 Metal Burst, 4 Comeuppance.
REPLY = ["counter", "mirrorcoat", "metalburst", "comeuppance"]
#: Houndoom: 1 Endeavor, 2 Super Fang, 3 Protect, 4 Comeuppance.
FIXED = ["endeavor", "superfang", "protect", "comeuppance"]
#: The partner: 1 Protect, 2 Earthquake (hits its own side too), 3 Double-Edge, 4 Helping Hand.
PARTNER = ["protect", "earthquake", "doubleedge", "helpinghand"]
#: The foes: 1 Dragon Claw (physical, contact), 2 Flamethrower (special), 3 Protect, 4 the slot's own.
FOE_A = ["dragonclaw", "flamethrower", "protect", "substitute"]
FOE_B = ["knockoff", "flamethrower", "protect", "followme"]


def _teams(
    user: str = "Swampert",
    user_moves: list[str] | None = None,
    user_spe: int = 0,
    foe_a: str = "Garchomp",
    foe_a_ability: str = "Rough Skin",
    foe_b: str = "Incineroar",
    foe_spe: tuple[int, int] = (32, 20),
    foe_a_item: str | None = None,
    foe_b_item: str | None = None,
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon(user, "Damp", user_moves or REPLY, user_spe, hp=32, **{"def": 20, "spd": 20}),
        _mon("Kangaskhan", "Inner Focus", PARTNER, 10, atk=32),
    ]
    theirs = [
        _mon(foe_a, foe_a_ability, FOE_A, foe_spe[0], foe_a_item),
        _mon(foe_b, "Blaze", FOE_B, foe_spe[1], foe_b_item),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


def _me(user: str) -> str:
    """The user's choice beside a Protecting partner."""
    return f"{user}, move 1"


COUNTER, MIRROR, BURST, COMEUPPANCE = "move 1", "move 2", "move 3", "move 4"
ENDEAVOR, FANG = "move 1 1", "move 2 1"
#: Both foes Protect.
QUIET = "move 3, move 3"
A_CLAW = "move 1 1, move 3"
A_FLAME = "move 2 1, move 3"
#: Garchomp flames the Protecting partner, Incineroar Protects: nothing reaches the user.
AWAY = "move 2 2, move 3"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # -- Counter
    "counter-returns-a-physical-hit": (
        _teams(), [], [_me(COUNTER), A_CLAW], "|-damage|p2a: Garchomp",
    ),
    "counter-returns-the-last-physical-hit": (
        _teams(), [], [_me(COUNTER), "move 1 1, move 1 1"], "|-damage|p2b: Incineroar",
    ),
    "counter-keeps-the-physical-hit-over-a-later-special": (
        _teams(), [], [_me(COUNTER), "move 1 1, move 2 1"], "|move|p1a: Swampert|Counter|p2a: Garchomp",
    ),
    "counter-fails-after-a-special-hit": (
        _teams(), [], [_me(COUNTER), A_FLAME], "|-fail|p1a: Swampert",
    ),
    "counter-fails-on-its-partners-hit": (
        _teams(), [], [f"{COUNTER}, move 2", QUIET], "|-fail|p1a: Swampert",
    ),
    "counter-into-a-ghost": (
        _teams(foe_a="Dragapult", foe_a_ability="Clear Body"), [], [_me(COUNTER), A_CLAW],
        "|-immune|p2a: Dragapult",
    ),
    "counter-drawn-by-follow-me": (
        _teams(), [], [_me(COUNTER), "move 1 1, move 4"], "|-damage|p2b: Incineroar",
    ),
    "counter-into-a-substitute": (
        _teams(), [[f"{COUNTER}, move 4 -1", "move 4, move 4"]], [_me(COUNTER), A_CLAW],
        "|-end|p2a: Garchomp|Substitute",
    ),
    # Kangaskhan's Double-Edge takes Alakazam down before Counter: its redirect sends it back
    # to the fainted slot, and it fails.
    "counter-fails-when-the-attacker-fainted": (
        _teams(foe_a="Alakazam", foe_a_ability="Inner Focus"), [],
        [f"{COUNTER}, move 3 1", "move 1 1, move 2 2"], "|faint|p2a: Alakazam",
    ),
    "control-counter-with-nothing-to-return": (
        _teams(), [], [_me(COUNTER), QUIET], "|-fail|p1a: Swampert",
    ),
    # A resist berry is `onSourceModifyDamage`, which a `damageCallback` never reaches: a
    # super-effective Counter leaves Incineroar's Chople Berry (the port ate it).
    "counter-leaves-a-resist-berry": (
        _teams(foe_b_item="Chople Berry"),
        [],
        [_me(COUNTER), "move 3, move 1 1"],
        "|-damage|p2b: Incineroar",
    ),
    # -- Mirror Coat
    "mirror-coat-returns-a-special-hit": (
        _teams(), [], [_me(MIRROR), A_FLAME], "|-damage|p2a: Garchomp",
    ),
    "mirror-coat-fails-after-a-physical-hit": (
        _teams(), [], [_me(MIRROR), A_CLAW], "|-fail|p1a: Swampert",
    ),
    "mirror-coat-into-a-dark-type": (
        _teams(foe_a="Hydreigon", foe_a_ability="Levitate"), [], [_me(MIRROR), A_FLAME],
        "|-immune|p2a: Hydreigon",
    ),
    # -- Metal Burst (priority 0: Swampert is the slowest)
    "metal-burst-returns-a-physical-hit": (
        _teams(), [], [_me(BURST), A_CLAW], "|-damage|p2a: Garchomp",
    ),
    "metal-burst-returns-a-special-hit": (
        _teams(), [], [_me(BURST), A_FLAME], "|-damage|p2a: Garchomp",
    ),
    "metal-burst-returns-the-last-hit-of-either-kind": (
        _teams(), [], [_me(BURST), "move 1 1, move 2 1"], "|move|p1a: Swampert|Metal Burst|p2b: Incineroar",
    ),
    "metal-burst-fails-before-it-is-hit": (
        _teams(user_spe=32, foe_a="Snorlax", foe_a_ability="Thick Fat", foe_spe=(0, 0)), [],
        [_me(BURST), A_CLAW], "|-fail|p1a: Swampert",
    ),
    "metal-burst-fails-on-its-partners-hit": (
        _teams(), [], [f"{BURST}, move 2", QUIET], "|-fail|p1a: Swampert",
    ),
    # The attacker fainted: `getMoveTargets` retargets the other foe, and nothing sends it back.
    "metal-burst-retargets-when-the-attacker-fainted": (
        _teams(foe_a="Alakazam", foe_a_ability="Inner Focus"), [],
        [f"{BURST}, move 3 1", "move 1 1, move 2 2"], "|move|p1a: Swampert|Metal Burst|p2b: Incineroar",
    ),
    "control-metal-burst-with-nothing-to-return": (
        _teams(), [], [_me(BURST), QUIET], "|-fail|p1a: Swampert",
    ),
    # -- Comeuppance (contact: Rough Skin answers it, which Metal Burst does not meet)
    "comeuppance-returns-a-physical-hit": (
        _teams(), [], [_me(COMEUPPANCE), A_CLAW], "[from] ability: Rough Skin",
    ),
    "comeuppance-returns-a-special-hit": (
        _teams(), [], [_me(COMEUPPANCE), A_FLAME], "|-damage|p2a: Garchomp",
    ),
    # -- Endeavor and Super Fang (Houndoom)
    "endeavor-cuts-to-the-users-hp": (
        _teams(user="Houndoom", user_moves=FIXED), [["move 2 2, move 4 -1", "move 1 1, move 4"]],
        [_me(ENDEAVOR), AWAY],
        "|-damage|p2a: Garchomp",
    ),
    "endeavor-is-immune-at-no-lower-hp": (
        _teams(user="Houndoom", user_moves=FIXED), [["move 3, move 3 1", "move 2 2, move 2 2"]],
        [_me(ENDEAVOR), AWAY], "|-immune|p2a: Garchomp",
    ),
    "control-endeavor-into-a-ghost": (
        _teams(user="Houndoom", user_moves=FIXED, foe_a="Dragapult", foe_a_ability="Clear Body"), [],
        [_me(ENDEAVOR), AWAY], "|-immune|p2a: Dragapult",
    ),
    "super-fang-halves-the-target": (
        _teams(user="Houndoom", user_moves=FIXED), [], [_me(FANG), AWAY], "|-damage|p2a: Garchomp",
    ),
    "super-fang-leaves-a-chilan-berry": (
        _teams(user="Houndoom", user_moves=FIXED, foe_a_item="Chilan Berry"),
        [],
        [_me(FANG), AWAY],
        "|-damage|p2a: Garchomp",
    ),
    "super-fang-halves-an-odd-hp": (
        _teams(user="Houndoom", user_moves=FIXED), [["move 2 1, move 4 -1", "move 2 2, move 2 2"]],
        [_me(FANG), AWAY],
        "|-damage|p2a: Garchomp",
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
            # A reply with nothing to reply to, and Endeavor's immunity, fail: Stomping
            # Tantrum reads it next turn.
            mon.move_last_turn_failed if not mon.fainted else None,
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
    if not any(line.startswith("|turn|") for line in log):
        # Waiting on a replacement, Showdown has not reached `nextTurn`, which is where
        # `moveLastTurnResult` takes this turn's result; the port's position already has it.
        after = {key: (*value[:-1], "not yet") for key, value in after.items()}
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


@pytest.mark.parametrize(("reply", "shown"), [("move 3", "Metal Burst"), ("move 1", "Counter")])
def test_a_reply_after_a_u_turn_goes_through_the_pause(
    reg, oracle: Oracle, bridged: None, reply: str, shown: str  # noqa: ANN001
) -> None:
    """Garchomp U-turns the user and Milotic comes in before the user moves: the turn stops
    at the replacement, and the record crosses the pause as JSON (`damage_callback::to_json`).
    `getAtSlot` is the slot, so the reply lands on Milotic."""
    mine, theirs = _teams()
    theirs[0] = _mon("Garchomp", "Rough Skin", ["uturn", "flamethrower", "protect", "substitute"], 32)
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    choices = [_me(reply), "move 1 1, move 3"]
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    handle.step([None, "switch 3, pass"])
    assert handle.choice_errors == [], handle.choice_errors
    log = list(handle.log)
    assert any(line.startswith(f"|move|p1a: Swampert|{shown}|p2a: Milotic") for line in log), log
    theirs_after = _state(Position.from_json(handle.position))
    handle.close()

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(result.suspended) == 1 and not result.branches
    pause = result.suspended[0]
    option = next(
        o
        for o in switch_actions_after_faint(reg, pause.position, 1, [True, False])
        if o.to_choice() == "switch 3, pass"
    )
    passes = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    finished = resume_turn(reg, pause, [passes, option])
    assert len(finished.branches) == 1
    assert _state(finished.branches[0].position) == theirs_after
    assert [n for n in finished.unmodelled if any(i in n for i in REPLY_IDS)] == []


@pytest.mark.parametrize(("move", "noted"), [("brickbreak", True), ("dragonclaw", False), ("counter", False)])
def test_a_damaging_move_the_port_never_names_is_reported(
    reg, oracle: Oracle, bridged: None, move: str, noted: bool  # noqa: ANN001
) -> None:
    """The mechanism behind the fix (`modelled::damaging_move_is_unmodelled`): Brick Break's
    screen-breaking is an `onTryHit` the port never names, so the turn says so; Dragon Claw has
    no custom code, and Counter has custom code the port now names."""
    mine = [_mon("Kangaskhan", "Inner Focus", [move, "protect"], 10, atk=32),
            _mon("Swampert", "Damp", ["protect", "earthquake"], 0)]
    theirs = [_mon("Garchomp", "Rough Skin", FOE_A, 32), _mon("Incineroar", "Blaze", FOE_B, 20)]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    choice = "move 1, move 1" if move == "counter" else "move 1 1, move 1"
    result = resolve_turn(reg, before, _actions(reg, before, [choice, A_CLAW]), budget=BUDGET)
    assert (f"damaging move: {move}" in result.unmodelled) is noted, result.unmodelled


ROOT = Path(__file__).resolve().parents[1]


def _ported() -> set[str]:
    source = (ROOT / "rust" / "src" / "damage_callback.rs").read_text(encoding="utf-8")
    body = source[source.index("fn ported(") :]
    body = body[: body.index("\n}\n")]
    return set(re.findall(r'"([a-z0-9]+)"', body))


@pytest.mark.parametrize("regulation", ["gen9championsvgc2026regmb", "gen9championsvgc2026regmc"])
def test_every_damage_callback_in_the_dump_is_ported(regulation: str) -> None:
    """The hook is not a field, so no gate sees it: this is where a new one shows up. A move
    listed here and not implemented would be a quiet wrong answer, which is why the list is
    the port's own (`damage_callback::ported`) and not this file's."""
    dump = json.loads((ROOT / "configs" / "regulations" / f"{regulation}.json").read_text(encoding="utf-8"))
    hooked = {m["id"] for m in dump["moves"] if "damageCallback" in (m.get("customHooks") or [])}
    assert hooked, "the dump lists no damageCallback at all: its customHooks are not being read"
    assert hooked <= _ported(), sorted(hooked - _ported())


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
    if all(value[-1] == "not yet" for value in theirs.values()):
        rust_now = {key: (*value[:-1], "not yet") for key, value in rust_now.items()}
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
    notes = [n for n in there.unmodelled if any(i in n for i in REPLY_IDS)]
    assert notes == [], notes
