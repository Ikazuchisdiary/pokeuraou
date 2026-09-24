"""Substitute: what it costs, what it takes in place of its user, and what gets past it
(IKA-180).

vendor/pokemon-showdown (a5df827), data/moves.ts `substitute`, which the champions mod
keeps -- quoted in full at the head of the Substitute section of rust/src/moves.rs (it was
above `_hits_substitute` in src/pokeuraou/resolve.py until IKA-212):

- **Use.** `onTryHit` refuses a second doll and a user at or below a quarter of its HP
  (`NOT_FAIL`: a `null` result, which Stomping Tantrum does not count); `onHit` pays
  `directDamage(maxhp / 4)`; the doll's `onStart` gives it `floor(maxhp / 4)` HP and ends a
  `partiallytrapped`.
- **A hit.** `onTryPrimaryHit`, for any move but its user's own, one without `bypasssub`
  and not an Infiltrator's: the damage is capped at the doll's HP and taken off it, the
  doll goes at 0, and the recoil (`applyRecoilDamage`) and the drain (`Math.ceil`) come from
  that damage. `spreadMoveHit` (data/mods/champions/scripts.ts) makes the target `null`
  from there on: no secondary, no contact ability, no Rocky Helmet, no resist berry
  (`hitSub`), no Infestation trap -- while `selfDrops` and a secondary's `self` still reach
  the user, and a spread move treats only the target behind the doll this way.
- **A status move** gets `null` from `getDamage` and does nothing, and is not a failure.
- **Past it.** A sound move (every one has `bypasssub`), a `bypasssub` status move (Taunt,
  Encore) and an Infiltrator's move.
- **Intimidate** skips a Pokemon behind a doll (data/abilities.ts, `intimidate`).

Before IKA-180 both engines marked `substitute` on the user and did nothing else: no HP
paid, every move went to the user.

Each case is played by Showdown first, and the Python turn and the port's turn are both
held to it, as in tests/test_after_move_oracle.py.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 2, "spa": 20, "spd": 2, "spe": spe}
    spread.update(sp)
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread
    )


# The side with the doll: Garchomp (p2a) moves first and holds Rough Skin and a Rocky
# Helmet, so a contact move into it shows both; Toxapex (p2b) moves last.
GARCHOMP = _mon(
    "Garchomp", "Rough Skin", ["substitute", "swordsdance", "protect", "dragonclaw"], 32,
    "Rocky Helmet",
)
YACHE = _mon("Garchomp", "Rough Skin", ["substitute", "swordsdance", "protect", "dragonclaw"], 32,
             "Yache Berry")
#: Slower than Dragapult, so a doll broken this turn is put up again after the hit; 212 HP,
#: so three dolls leave it at exactly a quarter.
SLOW = _mon("Garchomp", "Rough Skin", ["substitute", "swordsdance", "protect", "dragonclaw"], 0,
            hp=29)
TOXAPEX = _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0)
BENCH = [
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
    _mon("Arcanine", "Justified", ["flareblitz", "protect", "scald", "extremespeed"], 0),
]
SUBBERS = [GARCHOMP, TOXAPEX, *BENCH]


def _team(attacker: TeamSet, *bench: TeamSet) -> list[TeamSet]:
    """The attacking side: `attacker` (p1a), a Sylveon beside it that Protects or Calm
    Minds, and a bench."""
    rest = list(bench) or [
        _mon("Kingambit", "Defiant", ["ironhead", "protect", "suckerpunch", "swordsdance"], 0)
    ]
    sylveon = _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "yawn"], 10)
    while len(rest) < 2:
        rest.append(_mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0))
    return [attacker, sylveon, *rest]


ARCANINE = _mon("Arcanine", "Justified", ["flareblitz", "protect", "willowisp", "extremespeed"], 10)
CONKELDURR = _mon("Conkeldurr", "Guts", ["drainpunch", "protect", "machpunch", "icepunch"], 10)
PAWMOT = _mon("Pawmot", "Iron Fist", ["nuzzle", "protect", "closecombat", "thunderpunch"], 10)
TOXAPEX_P1 = _mon("Toxapex", "Regenerator", ["infestation", "protect", "toxic", "recover"], 10)
MILOTIC = _mon("Milotic", "Marvel Scale", ["icebeam", "protect", "scald", "recover"], 10)
CHARIZARD = _mon("Charizard", "Blaze", ["heatwave", "protect", "flamecharge", "airslash"], 10)
GENGAR = _mon("Gengar", "Cursed Body", ["taunt", "protect", "shadowball", "encore"], 10)
DRAGAPULT = _mon("Dragapult", "Infiltrator", ["willowisp", "protect", "shadowball", "dragonclaw"], 10)
DRAGAPULT_CLAW = _mon("Dragapult", "Clear Body", ["dragonclaw", "protect", "willowisp", "shadowball"], 10)
DRAGAPULT_PLAIN = _mon("Dragapult", "Clear Body", ["willowisp", "protect", "shadowball", "breakingswipe"], 10)
MAMOSWINE = _mon("Mamoswine", "Oblivious", ["iciclespear", "protect", "earthquake", "iceshard"], 10)
INCINEROAR = _mon("Incineroar", "Intimidate", ["fakeout", "protect", "flareblitz", "partingshot"], 10)

#: p1 Protects twice; Garchomp puts up its doll and Toxapex Recovers (and fails).
SUB = ["move 2, move 1", "move 1, move 2"]
#: Conkeldurr Mach Punches Toxapex, which Scalds it; then SUB, with the Sylveon Calm Minding.
HURT_THEN_SUB = [["move 3 2, move 1", "move 2, move 3 1"], ["move 2, move 2", "move 1, move 2"]]
#: The compared turn's p2 side: Garchomp Swords Dances, Toxapex Protects.
QUIET = "move 2, move 1"
#: ... or Toxapex Recovers, for a spread move that has to reach it.
OPEN = "move 2, move 2"


def _at(move: int, target: int = 1, partner: str = "move 2") -> str:
    """p1a's move `move` at p2's slot `target`; the Sylveon Calm Minds."""
    return f"move {move} {target}, {partner}"


#: name -> (teams, turns played first, the turn compared, the policy, a Showdown log line
#: the case needs, a line that must be absent or None).
CASES: dict[str, tuple] = {
    # --- Use ---------------------------------------------------------------------------
    "substitute-costs-a-quarter": (
        (_team(ARCANINE), SUBBERS), [], SUB, None, "|-start|p2a: Garchomp|Substitute", None,
    ),
    "second-substitute-fails": (
        (_team(ARCANINE), SUBBERS), [SUB], ["move 3 2, move 2", "move 1, move 1"], None,
        "|-fail|p2a: Garchomp|move: Substitute", None,
    ),
    "control-third-substitute-after-two-broken": (
        (_team(DRAGAPULT_CLAW), [SLOW, TOXAPEX, *BENCH]),
        [["move 2, move 1", "move 1, move 2"], [_at(1), "move 1, move 2"]],
        [_at(1), "move 1, move 2"], None, "|-start|p2a: Garchomp|Substitute", None,
    ),
    "weak-substitute-fails": (
        (_team(DRAGAPULT_CLAW), [SLOW, TOXAPEX, *BENCH]),
        [["move 2, move 1", "move 1, move 2"], [_at(1), "move 1, move 2"],
         [_at(1), "move 1, move 2"]],
        [_at(1), "move 1, move 2"], None, "|-fail|p2a: Garchomp|move: Substitute|[weak]", None,
    ),
    "substitute-ends-infestation": (
        (_team(TOXAPEX_P1), SUBBERS), [[_at(1), QUIET]], ["move 2, move 1", "move 1, move 2"],
        None, "|-start|p2a: Garchomp|Substitute", None,
    ),
    # --- A hit -----------------------------------------------------------------------
    "flare-blitz-hits-the-doll": (
        (_team(ARCANINE), SUBBERS), [SUB], [_at(1), QUIET], None,
        "|-activate|p2a: Garchomp|move: Substitute|[damage]", "Rocky Helmet",
    ),
    "control-flare-blitz-hits-garchomp": (
        (_team(ARCANINE), SUBBERS), [], [_at(1), QUIET], None,
        "|-damage|p1a: Arcanine|", None,
    ),
    # Toxapex Scalds Conkeldurr first, so the drain has HP to fill: the doll's 53 heal 27.
    "drain-punch-drains-from-the-doll": (
        (_team(CONKELDURR), SUBBERS), HURT_THEN_SUB, [_at(1), QUIET], None,
        "|-heal|p1a: Conkeldurr|", None,
    ),
    "control-drain-punch": (
        (_team(CONKELDURR), SUBBERS), [], [_at(1), QUIET], None, "|-damage|p2a: Garchomp|", None,
    ),
    # A spread move's 100% drop: on Toxapex, not on the doll's holder.
    "breaking-swipe-drops-only-toxapex": (
        (_team(DRAGAPULT_PLAIN), SUBBERS), [SUB], ["move 4, move 2", OPEN], None,
        "|-unboost|p2b: Toxapex|atk|1", "|-unboost|p2a: Garchomp|atk",
    ),
    "control-breaking-swipe": (
        (_team(DRAGAPULT_PLAIN), SUBBERS), [], ["move 4, move 2", OPEN], None,
        "|-unboost|p2a: Garchomp|atk|1", None,
    ),
    "close-combat-keeps-its-drops": (
        (_team(PAWMOT), SUBBERS), [SUB], [_at(3), QUIET], None, "|-unboost|p1a: Pawmot|def|1", None,
    ),
    "flame-charge-keeps-its-boost": (
        (_team(CHARIZARD), SUBBERS), [SUB], [_at(3), QUIET], None, "|-boost|p1a: Charizard|spe|1",
        None,
    ),
    "infestation-traps-no-doll": (
        (_team(TOXAPEX_P1), SUBBERS), [SUB], [_at(1), QUIET], None,
        "|-activate|p2a: Garchomp|move: Substitute|[damage]", "|-activate|p2a: Garchomp|move: Infestation",
    ),
    "yache-berry-stays-behind-the-doll": (
        (_team(MILOTIC), [YACHE, TOXAPEX, *BENCH]), [SUB], [_at(1), QUIET], None,
        "|-end|p2a: Garchomp|Substitute", "|-enditem|p2a: Garchomp|Yache Berry",
    ),
    "control-yache-berry-eaten": (
        (_team(MILOTIC), [YACHE, TOXAPEX, *BENCH]), [], [_at(1), QUIET], None,
        "|-enditem|p2a: Garchomp|Yache Berry|[eat]", None,
    ),
    "heat-wave-hits-the-doll-and-toxapex": (
        (_team(CHARIZARD), SUBBERS), [SUB], ["move 1, move 2", OPEN], None,
        "|-damage|p2b: Toxapex|", None,
    ),
    "icicle-spear-breaks-the-doll-then-hits": (
        (_team(MAMOSWINE), SUBBERS), [SUB], [_at(1), QUIET], None, "|-end|p2a: Garchomp|Substitute",
        None,
    ),
    # --- A status move ---------------------------------------------------------------
    "toxic-fails-on-the-doll": (
        (_team(TOXAPEX_P1), SUBBERS), [SUB], [_at(3), QUIET], None, "|-fail|p1a: Toxapex", None,
    ),
    "control-toxic": (
        (_team(TOXAPEX_P1), SUBBERS), [], [_at(3), QUIET], None, "|-status|p2a: Garchomp|tox", None,
    ),
    "will-o-wisp-fails-on-the-doll": (
        (_team(DRAGAPULT_PLAIN), SUBBERS), [SUB], [_at(1), QUIET], None, "|-fail|p1a: Dragapult", None,
    ),
    "yawn-fails-on-the-doll": (
        (_team(ARCANINE), SUBBERS), [SUB], ["move 3 2, move 4 1", QUIET], None, "|-fail|p1b: Sylveon",
        None,
    ),
    # --- Past it ---------------------------------------------------------------------
    "hyper-voice-passes-the-doll": (
        (_team(ARCANINE), SUBBERS), [SUB], ["move 3 2, move 3", QUIET], None,
        "|-damage|p2a: Garchomp|", None,
    ),
    "taunt-passes-the-doll": (
        (_team(GENGAR), SUBBERS), [SUB], [_at(1), QUIET], None, "|-start|p2a: Garchomp|move: Taunt",
        None,
    ),
    "infiltrator-will-o-wisp-passes-the-doll": (
        (_team(DRAGAPULT), SUBBERS), [SUB], [_at(1), QUIET], None, "|-status|p2a: Garchomp|brn", None,
    ),
    # --- Intimidate ------------------------------------------------------------------
    "intimidate-skips-the-doll": (
        (_team(ARCANINE, INCINEROAR), SUBBERS), [SUB], ["switch 3, move 2", OPEN], None,
        "|-immune|p2a: Garchomp", None,
    ),
    "control-intimidate": (
        (_team(ARCANINE, INCINEROAR), SUBBERS), [], ["switch 3, move 2", OPEN], None,
        "|-unboost|p2a: Garchomp|atk|1", None,
    ),
}

#: No crit, the maximum roll, no chance-based secondary -- what the oracle's policy pins.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)
#: What each case's compared turn is held to, per Pokemon.
TRACKED_VOLATILES = ("partiallytrapped", "taunt", "encore", "yawn")


def _doll(mon) -> int | None:  # noqa: ANN001
    doll = mon.volatile("substitute")
    if doll is None:
        return None
    return int(doll.extra.get("hp", -1))


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{slot}.{mon.species}": (
            mon.hp, mon.status, mon.move_last_turn_failed, mon.item,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)), _doll(mon),
            tuple(v for v in TRACKED_VOLATILES if mon.has_volatile(v)),
        )
        for index, side in enumerate(pos.sides)
        for slot, mon in enumerate(side.pokemon)
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
    (team_a, team_b), setup, choices, policy, shown, absent = CASES[name]
    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=policy or RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    assert any(line.startswith(shown) for line in log), f"Showdown did not do what {name} says: {log}"
    assert absent is None or not any(absent in line for line in log), f"{name}: {absent} in {log}"
    return before, choices, theirs, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


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
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    # Every move hits, as the policy has it; every other branch has to be Showdown's
    # (IKA-210: the branches used to be picked by Python's events).
    budget = replace(BUDGET, enumerate_accuracy=False)
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    assert there.branches, "no finished outcome"
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, budget, select=index)
        assert picked is not None and picked.position is not None
        rust_now = _state(picked.position)
        assert rust_now == theirs, f"{name}, branch {index}: showdown {theirs} != rust {rust_now}"


# ---------------------------------------------------------------------------
# No oracle: the parts a position has to be built for.


def _hand_built(reg, p1: list[TeamSet], p2: list[TeamSet]) -> Position:  # noqa: ANN001
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, p1)
    other = _synthetic_position(reg, p2)
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    return pos


def test_a_doll_without_its_hp_starts_at_a_quarter(reg) -> None:  # noqa: ANN001
    """A hand-built doll carries no `extra.hp`; it is taken to be the `floor(maxhp / 4)`
    the doll starts with, and a hit smaller than that leaves the rest on it."""
    pos = _hand_built(reg, _team(ARCANINE), SUBBERS)
    garchomp = pos.sides[1].pokemon[pos.sides[1].active[0]]
    garchomp.volatiles.append(Effect(id="substitute"))
    before = garchomp.hp
    result = resolve_turn(reg, pos, _actions(reg, pos, ["move 4 1, move 2", QUIET]), budget=BUDGET)
    for branch in result.branches:
        after = branch.position.sides[1].pokemon[pos.sides[1].active[0]]
        assert after.hp == before
        doll = after.volatile("substitute")
        assert doll is None or 0 < doll.extra["hp"] < garchomp.maxhp // 4
