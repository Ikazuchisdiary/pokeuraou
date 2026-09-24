"""The two causes behind the Parting Shot turns diverge_report carried on (IKA-238), played
against Showdown.

**A Speed tie's canonical order.** Showdown a5df827, sim/battle.ts `speedSort` is a
selection sort: it takes the first of the remaining actions, with every action tied to it
in list order, and swaps them to the front::

    while (sorted + 1 < list.length) {
        let nextIndexes = [sorted];
        for (let i = sorted + 1; i < list.length; i++) {
            const delta = comparator(list[nextIndexes[0]], list[i]);
            if (delta < 0) continue;
            if (delta > 0) nextIndexes = [i];
            if (delta === 0) nextIndexes.push(i);
        }
        for (let i = 0; i < nextIndexes.length; i++) {
            const index = nextIndexes[i];
            if (index !== sorted + i) {
                [list[sorted + i], list[index]] = [list[index], list[sorted + i]];
            }
        }
        if (nextIndexes.length > 1) this.prng.shuffle(list, sorted, sorted + nextIndexes.length);
        sorted += nextIndexes.length;
    }

The oracle's `speedTie: 'keep'` makes the shuffle do nothing, so the tied group is left as
the swaps put it. The port sorted stably (a tie in queue order) and, with ties not
enumerated -- the deterministic budget every comparison uses -- that is the one order it
plays. Two Incineroar tied behind two faster partners, queue `[p1a, p1b, p2a, p2b]`: the
swaps give `[.., .., p2a, p1a]`, the stable sort `p1a` first. A Parting Shot that should
come last went first, and the foe's Throat Chop hit at -1 (seed 1 battle 2 turn 2).

**Mirror Armor.** data/abilities.ts::

    mirrorarmor: {
        onTryBoost(boost, target, source, effect) {
            if (!source || target === source || !boost || effect.name === 'Mirror Armor') return;
            for (b in boost) {
                if (boost[b]! < 0) {
                    if (target.boosts[b] === -6) continue;
                    const negativeBoost = {}; negativeBoost[b] = boost[b]; delete boost[b];
                    if (source.hp) {
                        this.add('-ability', target, 'Mirror Armor');
                        this.boost(negativeBoost, source, target, null, true);
                    }
                }
            }
        },
        flags: { breakable: 1 },
    },

and data/moves.ts partingshot::

    onHit(target, source, move) {
        const success = this.boost({ atk: -1, spa: -1 }, target, source);
        if (!success && !target.hasAbility('mirrorarmor')) delete move.selfSwitch;
    },
    selfSwitch: true,

The port let the user leave but lowered Corviknight anyway (seed 20 battle 17 turn 3).

Each case plays in Showdown and holds the port's one outcome (deterministic budget, the
paused turn resumed with Showdown's replacement) to it. The positive control is the exe
before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import os

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import PassAction, SideAction, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn, resume_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

BUDGET = Budget.deterministic(0)


def _mon(species: str, ability: str, moves: list[str], spe: int, **sp: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 10, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=spread)


#: The Incineroar: 1 Charm, 2 Body Slam, 3 Protect, 4 Parting Shot. Both at 0 Speed SP.
#: Body Slam and not the Throat Chop of the divergent turn: a Throat Chop that lands first
#: stops Parting Shot, a sound move, in Showdown (`onBeforeMove`), which the port does not
#: model -- a separate hole (IKA-238 records it).
ROAR = ["charm", "bodyslam", "protect", "partingshot"]
#: A partner: 1 Protect, 2 Swords Dance.
PARTNER = ["protect", "swordsdance"]


def _teams(
    fast_partners: bool = True,
    foe: str = "Incineroar",
    foe_ability: str = "Blaze",
    user_ability: str = "Blaze",
) -> tuple[list[TeamSet], list[TeamSet]]:
    """Faster partners (Garchomp, Dragapult) sort ahead of the Incineroar and swap them;
    slower ones (Snorlax, Torkoal) leave the queue order alone."""
    mine_partner, their_partner = ("Garchomp", "Dragapult") if fast_partners else ("Snorlax", "Torkoal")
    mine = [
        _mon("Incineroar", user_ability, ROAR, 0),
        _mon(mine_partner, "Inner Focus", PARTNER, 0),
        _mon("Milotic", "Marvel Scale", PARTNER, 0, hp=32),
    ]
    theirs = [
        _mon(foe, foe_ability, ROAR if foe == "Incineroar" else ["bravebird", "roost"], 0),
        _mon(their_partner, "Inner Focus", PARTNER, 0),
        _mon("Milotic", "Marvel Scale", PARTNER, 0),
    ]
    return mine, theirs


CHARM, SLAM, PARTING = "move 1 1", "move 2 1", "move 4 1"

#: name -> (teams, [p1 choice, p2 choice], a Showdown log line that shows the case).
CASES: dict[str, tuple] = {
    # The partners Protect first; Showdown's swaps put p2a's Body Slam ahead of Charm.
    "tie-behind-faster-partners-charm": (
        _teams(), [f"{CHARM}, move 1", f"{SLAM}, move 1"],
        "|move|p2a: Incineroar|Body Slam|p1a: Incineroar",
    ),
    # The divergent turn's shape: Parting Shot comes after the foe's hit.
    "tie-behind-faster-partners-parting-shot": (
        _teams(), [f"{PARTING}, move 1", f"{SLAM}, move 1"],
        "|move|p2a: Incineroar|Body Slam|p1a: Incineroar",
    ),
    # Control: slower partners never swap the Incineroar, and p1a goes first in both sorts.
    "control-tie-in-queue-order-charm": (
        _teams(fast_partners=False), [f"{CHARM}, move 2", f"{SLAM}, move 2"],
        "|move|p1a: Incineroar|Charm|p2a: Incineroar",
    ),
    "control-tie-in-queue-order-parting-shot": (
        _teams(fast_partners=False), [f"{PARTING}, move 2", f"{SLAM}, move 2"],
        "|move|p1a: Incineroar|Parting Shot|p2a: Incineroar",
    ),
    # Mirror Armor sends both drops back; the user still leaves.
    "mirror-armor-sends-parting-shot-back": (
        _teams(foe="Corviknight", foe_ability="Mirror Armor"), [f"{PARTING}, move 1", "move 1 2, move 1"],
        "|-ability|p2a: Corviknight|Mirror Armor",
    ),
    # Mold Breaker breaks it (`flags: {breakable: 1}`): the drops land.
    "control-mold-breaker-parting-shot-into-mirror-armor": (
        _teams(foe="Corviknight", foe_ability="Mirror Armor", user_ability="Mold Breaker"),
        [f"{PARTING}, move 1", "move 1 2, move 1"],
        "|-unboost|p2a: Corviknight|atk|1",
    ),
    # Control: Parting Shot into an ordinary target.
    "control-parting-shot-into-an-ordinary-target": (
        _teams(foe="Corviknight", foe_ability="Pressure"), [f"{PARTING}, move 1", "move 1 2, move 1"],
        "|-unboost|p2a: Corviknight|atk|1",
    ),
}


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


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
    (mine, theirs), choices, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 123", "team 123"])
    assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    log = list(handle.log)
    replaced = bool((handle.requests[0] or {}).get("forceSwitch"))
    if replaced:
        handle.step(["switch 3, pass", None])
        assert handle.choice_errors == [], handle.choice_errors
        log += handle.log
    theirs_after = _state(Position.from_json(handle.position))
    handle.close()
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    if replaced:
        assert len(result.suspended) == 1 and not result.branches, (result.branches, result.suspended)
        pause = result.suspended[0]
        pick = next(
            o
            for o in switch_actions_after_faint(reg, pause.position, 0, [True, False])
            if o.to_choice() == "switch 3, pass"
        )
        passes = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
        result = resume_turn(reg, pause, [pick, passes])
    assert len(result.branches) == 1 and not result.suspended, (result.branches, result.suspended)
    ours_after = _state(result.branches[0].position)
    assert ours_after == theirs_after, f"{name}: showdown {theirs_after} != port {ours_after}"
