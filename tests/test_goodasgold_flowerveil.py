"""Good as Gold and Flower Veil (IKA-202).

vendor/pokemon-showdown (a5df827; the champions mod overrides neither)::

    // data/abilities.ts
    goodasgold: {
        onTryHit(target, source, move) {
            if (move.category === 'Status' && target !== source) { ...; return null; }
        },
        flags: { breakable: 1 },
    },
    flowerveil: {
        onAllyTryBoost(boost, target, source, effect) {
            if ((source && target === source) || !target.hasType('Grass')) return;
            for (i in boost) if (boost[i]! < 0) { delete boost[i]; ... }
        },
        onAllySetStatus(status, target, source, effect) {
            if (target.hasType('Grass') && source && target !== source && effect
                && effect.id !== 'yawn') { ...; return null; }
        },
        onAllyTryAddVolatile(status, target) {
            if (target.hasType('Grass') && status.id === 'yawn') { ...; return null; }
        },
        flags: { breakable: 1 },
    },

Good as Gold's `onTryHit` runs in `hitStepTryHitEvent`, so it meets every status move aimed
at the holder by another Pokemon -- a foe's, a partner's (Decorate), each target of a spread
one -- and Perish Song's own `runEvent('TryHit', pokemon, ...)` for each active Pokemon.
Side and field moves (`foeSide`, `allySide`, `all`, `allyTeam`) go through `TryHitSide` /
`TryHitField` instead and pass. The `null` is a failure (`hitResults[i] || false`).

Flower Veil's `onAlly*` handlers run for the holder and its partner (`alliesAndSelf`), for a
Grass type only: every lowering another Pokemon causes (a foe's move, its secondary,
Intimidate, a partner's Charm) and every status another causes, except Yawn's sleep -- Yawn
is refused as the volatile instead. A Pokemon's own drops (Leaf Storm) pass. Both abilities
are `breakable`: a Mold Breaker move passes them.

Neither engine had either ability: the port listed Flower Veil as having no effect a turn
can observe and Good as Gold as inert, so neither refused and nothing was reported.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget
from .conftest import FORMAT_ID

FAST = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 32}
SLOW = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 4, "spe": 0}


def _mon(species: str, ability: str, moves: list[str], sp: dict | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp=dict(sp or SLOW))


STATUSER = ["willowisp", "charm", "taunt", "yawn"]
SPREADER = ["snarl", "perishsong", "willowisp", "charm"]
GENGAR = _mon("Gengar", "Cursed Body", STATUSER, FAST)
BREAKER = _mon("Gengar", "Mold Breaker", STATUSER, FAST)
SINGER = _mon("Gengar", "Cursed Body", SPREADER, FAST)
P1_REST = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "protect"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]
P2_REST = [
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]

GHOLDENGO = _mon("Gholdengo", "Good as Gold", ["nastyplot", "shadowball", "protect", "makeitrain"])
#: The control: the same Pokemon without the ability.
PLAIN_GHOLDENGO = _mon("Gholdengo", "Pressure", ["nastyplot", "shadowball", "protect", "makeitrain"])
RILLABOOM = _mon("Rillaboom", "Overgrow", ["swordsdance", "leafstorm", "woodhammer", "protect"])
FLORGES_VEIL = _mon("Florges", "Flower Veil", ["calmmind", "charm", "decorate", "moonblast"])
FLORGES_PLAIN = _mon("Florges", "Symbiosis", ["calmmind", "charm", "decorate", "moonblast"])


def _p2a(pos: Position) -> Pokemon:
    return pos.sides[1].pokemon[pos.sides[1].active[0]]


def _p2b(pos: Position) -> Pokemon:
    return pos.sides[1].pokemon[pos.sides[1].active[1]]


@dataclasses.dataclass(frozen=True)
class Case:
    p1a: TeamSet
    p2: tuple[TeamSet, TeamSet]
    step: tuple[str, str]
    #: Whether the effect under test reached its Pokemon, read from a position.
    landed: Callable[[Position], bool]
    #: Showdown's answer, asserted so a change in the oracle is seen.
    want: bool
    secondary: bool = False


def _burned(pos: Position) -> bool:
    return _p2a(pos).status == "brn"


def _cases() -> dict[str, Case]:
    gold = (GHOLDENGO, FLORGES_PLAIN)
    plain = (PLAIN_GHOLDENGO, FLORGES_PLAIN)
    veil = (RILLABOOM, FLORGES_VEIL)
    unveiled = (RILLABOOM, FLORGES_PLAIN)
    # p2a Nasty Plots / Swords Dances and p2b Calm Minds unless the case says otherwise;
    # p1b Protects.
    gold_step = "move 1, move 1"
    out = {
        "gold, will-o-wisp": Case(GENGAR, gold, ("move 1 1, move 2", gold_step), _burned, False),
        "gold, charm": Case(GENGAR, gold, ("move 2 1, move 2", gold_step),
                            lambda p: _p2a(p).boosts.get("atk", 0) < 0, False),
        # Taunted first, its Nasty Plot fails: +2 means the taunt never landed.
        "gold, taunt": Case(GENGAR, gold, ("move 3 1, move 2", gold_step),
                            lambda p: _p2a(p).boosts.get("spa", 0) < 2, False),
        "gold, yawn": Case(GENGAR, gold, ("move 4 1, move 2", gold_step),
                           lambda p: _p2a(p).has_volatile("yawn"), False),
        "gold, perish song": Case(SINGER, gold, ("move 2, move 2", gold_step),
                                  lambda p: _p2a(p).has_volatile("perishsong"), False),
        # The partner's Decorate is another Pokemon's status move too.
        "gold, partner's decorate": Case(GENGAR, gold, ("move 2 2, move 2", "move 1, move 3 -1"),
                                         lambda p: _p2a(p).boosts.get("atk", 0) > 0, False),
        # Controls: Mold Breaker, and the Pokemon without the ability.
        "gold, mold breaker will-o-wisp": Case(BREAKER, gold, ("move 1 1, move 2", gold_step),
                                               _burned, True),
        "gold, mold breaker charm": Case(BREAKER, gold, ("move 2 1, move 2", gold_step),
                                         lambda p: _p2a(p).boosts.get("atk", 0) < 0, True),
        "plain, will-o-wisp": Case(GENGAR, plain, ("move 1 1, move 2", gold_step), _burned, True),
        "plain, perish song": Case(SINGER, plain, ("move 2, move 2", gold_step),
                                   lambda p: _p2a(p).has_volatile("perishsong"), True),
        "plain, partner's decorate": Case(GENGAR, plain, ("move 2 2, move 2", "move 1, move 3 -1"),
                                          lambda p: _p2a(p).boosts.get("atk", 0) > 0, True),
        # Flower Veil guards the Grass partner.
        "veil, will-o-wisp": Case(GENGAR, veil, ("move 1 1, move 2", gold_step), _burned, False),
        # Charmed first, its Swords Dance nets 0: +2 means the drop never landed.
        "veil, charm": Case(GENGAR, veil, ("move 2 1, move 2", gold_step),
                            lambda p: _p2a(p).boosts.get("atk", 0) < 2, False),
        "veil, yawn": Case(GENGAR, veil, ("move 4 1, move 2", gold_step),
                           lambda p: _p2a(p).has_volatile("yawn"), False),
        "veil, snarl": Case(SINGER, veil, ("move 1, move 2", gold_step),
                            lambda p: _p2a(p).boosts.get("spa", 0) < 0, False, secondary=True),
        "veil, partner's charm": Case(GENGAR, veil, ("move 1 2, move 2", "move 1, move 2 -1"),
                                      lambda p: _p2a(p).boosts.get("atk", 0) < 2, False),
        "veil, intimidate": Case(GENGAR, veil, ("switch 3, move 2", gold_step),
                                 lambda p: _p2a(p).boosts.get("atk", 0) < 2, False),
        # Controls: its own Leaf Storm, the Fairy holder itself, Mold Breaker, no veil.
        "veil, own leaf storm": Case(GENGAR, veil, ("move 2 2, move 2", "move 2 1, move 1"),
                                     lambda p: _p2a(p).boosts.get("spa", 0) < 0, True),
        "veil, the holder is no grass type": Case(
            GENGAR, veil, ("move 1 2, move 2", gold_step), lambda p: _p2b(p).status == "brn", True
        ),
        "veil, intimidate on the holder": Case(
            GENGAR, veil, ("switch 3, move 2", gold_step),
            lambda p: _p2b(p).boosts.get("atk", 0) < 0, True,
        ),
        "veil, mold breaker will-o-wisp": Case(BREAKER, veil, ("move 1 1, move 2", gold_step),
                                               _burned, True),
        "unveiled, will-o-wisp": Case(GENGAR, unveiled, ("move 1 1, move 2", gold_step),
                                      _burned, True),
        "unveiled, yawn": Case(GENGAR, unveiled, ("move 4 1, move 2", gold_step),
                               lambda p: _p2a(p).has_volatile("yawn"), True),
        "unveiled, charm": Case(GENGAR, unveiled, ("move 2 1, move 2", gold_step),
                                lambda p: _p2a(p).boosts.get("atk", 0) < 2, True),
    }
    return out


CASES = _cases()


def _policy(case: Case) -> RandomnessPolicy:
    return RandomnessPolicy(secondary=case.secondary)


def _play(oracle: Oracle, case: Case) -> tuple[dict, dict]:
    """Showdown's positions before the step and after it."""
    handle = oracle.create(FORMAT_ID, [case.p1a, *P1_REST], [*case.p2, *P2_REST],
                           policy=_policy(case))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    handle.step(list(case.step))
    assert handle.choice_errors == [], handle.choice_errors
    after = handle.position
    handle.close()
    return before, after


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
            mon.stats_override = None
    return pos


def _chosen(reg, pos: Position, step: tuple[str, str]):  # noqa: ANN001, ANN202
    """The menu's actions. The menu never aims a move at the partner, so a `-1` / `-2`
    target is the menu's foe-aimed action with the target put back."""
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        parts = choice.split(", ")
        ally = {i: int(p.rsplit(" ", 1)[1]) for i, p in enumerate(parts) if p.endswith((" -1", " -2"))}
        aimed = ", ".join(p[:-3] + " 1" if i in ally else p for i, p in enumerate(parts))
        assert aimed in menu, (aimed, sorted(menu))
        action = menu[aimed]
        if ally:
            slots = tuple(dataclasses.replace(a, target=ally[i]) if i in ally else a
                          for i, a in enumerate(action.slots))
            action = dataclasses.replace(action, slots=slots)
        assert action.to_choice() == choice
        chosen.append(action)
    return chosen


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown(oracle: Oracle, name: str) -> None:
    case = CASES[name]
    _, after = _play(oracle, case)
    assert case.landed(Position.from_json(after)) == case.want


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_ports_turn_from_showdowns_position(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    """`test_our_turn_from_showdowns_position` with the port's branches, and the pinned
    outcome held to Showdown's own after-state."""
    from ._port_showdown import port_branches, port_turn

    case = CASES[name]
    before, after = _play(oracle, case)
    start = _loaded(before)
    chosen = _chosen(reg, start, case.step)
    landed = {case.landed(p) for _, p in port_branches(port, start, chosen, Budget.matrix())}
    assert (True in landed) == case.want, landed
    if case.secondary:
        return  # Showdown's policy fires the chance secondary; the pinned budget never does
    pinned = port_turn(port, start, chosen)
    assert case.landed(pinned) == case.landed(Position.from_json(after)), name
