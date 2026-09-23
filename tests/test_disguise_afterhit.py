"""A hit Disguise takes is still a hit (IKA-157).

Showdown's Disguise is an `onDamage` handler that returns 0 (vendor/pokemon-showdown/data/
abilities.ts, `disguise`). 0 is a number, so `spreadMoveHit` (sim/battle-actions.ts) keeps
the target through every step after the damage: the move's own effects, the self drops,
the secondaries and `DamagingHit`, then `onAfterHit`. The move `didAnything`, so U-turn
switches, and `afterMoveSecondaryEvent` runs, so Life Orb costs its tenth. Only the damage
is gone -- and drain and recoil, which are computed from it. The forme changes at the
`Update` after the first hit, so a multi-hit move's later hits land on Mimikyu-Busted.

The resolver returned from an absorbed hit before any of that, and absorbed every hit of a
multi-hit move. Each case is played by Showdown first and the Python turn is held to it.
The controls -- a Rock Slide and a Shadow Ball, which have nothing after the damage --
were right before and have to stay right.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn, self_switches_needed
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        item=item,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


TEAM_A = [
    _mon("Kingambit", "Defiant", ["kowtowcleave", "ironhead", "suckerpunch", "protect"], 12, "Life Orb"),
    _mon("Incineroar", "Blaze", ["uturn", "knockoff", "snarl", "throatchop"], 11),
    _mon("Dragonite", "Inner Focus", ["dualwingbeat", "protect", "extremespeed", "firepunch"], 10),
    _mon("Gholdengo", "Good as Gold", ["makeitrain", "protect", "shadowball", "nastyplot"], 9),
    _mon("Garchomp", "Sand Veil", ["rockslide", "protect", "earthquake", "dragonclaw"], 8),
]
# Mimikyu holds Rocky Helmet and leads beside a Protecting Kingambit, so a spread move
# reaches Mimikyu alone and every contact move meets the helmet.
TEAM_B = [
    _mon("Mimikyu", "Disguise", ["swordsdance", "protect", "playrough", "shadowsneak"], 2, "Rocky Helmet"),
    _mon("Kingambit", "Defiant", ["protect", "kowtowcleave", "suckerpunch", "swordsdance"], 3),
    _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 1),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 24),
]
SIDE_B = "move 1, move 1"  # Swords Dance, Protect

#: name -> (side 0's team order, side 0's choice, the Showdown log line that shows the
#: effect after the absorbed hit).
CASES = {
    # Contact into Rocky Helmet, then Life Orb's tenth.
    "rocky-helmet-and-life-orb": (
        "team 1534",
        "move 1 1, move 2",
        "|-damage|p1a: Kingambit|144/195|[from] item: Life Orb",
    ),
    # Helmet, then the switch request.
    "u-turn": (
        "team 2513",
        "move 1 1, move 2",
        "|-damage|p1a: Incineroar|159/190|[from] item: Rocky Helmet|[of] p2a: Mimikyu",
    ),
    # Helmet, then the item is knocked off.
    "knock-off": (
        "team 2513",
        "move 2 1, move 2",
        "|-enditem|p2a: Mimikyu|Rocky Helmet|[from] move: Knock Off|[of] p1a: Incineroar",
    ),
    # The move's own volatile, from a 100% secondary.
    "throat-chop": ("team 2513", "move 4 1, move 2", "|-start|p2a: Mimikyu|Throat Chop|[silent]"),
    # A 100% secondary on a spread move.
    "snarl": ("team 2513", "move 3, move 2", "|-unboost|p2a: Mimikyu|spa|1"),
    # The user's own drop, on a spread move whose only other target Protected.
    "make-it-rain": ("team 4513", "move 1, move 2", "|-unboost|p1a: Gholdengo|spa|2"),
    # The second hit lands on the busted forme; both meet the helmet.
    "dual-wingbeat": ("team 3514", "move 1 1, move 2", "|-hitcount|p2a: Mimikyu|2"),
    # Controls: nothing follows the absorbed damage.
    "control-rock-slide": ("team 5413", "move 1, move 2", "|-activate|p2a: Mimikyu|ability: Disguise"),
    "control-shadow-ball": ("team 4513", "move 3 1, move 2", "|-activate|p2a: Mimikyu|ability: Disguise"),
}

#: No crit, no chance-based secondary, the maximum roll -- what the oracle's policy pins.
#: Accuracy is enumerated, and the branches where everything hit are the ones compared.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)

#: The volatiles a case can leave behind. The rest are bookkeeping one side or the other
#: keeps and the other does not. The bridge dumps Throat Chop under `unmodelledVolatiles`,
#: so both lists are read.
VOLATILES = frozenset({"throatchop"})


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{slot}": (
            mon.species,
            mon.hp,
            mon.item,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)),
            tuple(
                sorted(VOLATILES.intersection([v.id for v in mon.volatiles] + list(mon.unmodelled_volatiles)))
            ),
        )
        for index, side in enumerate(pos.sides)
        for slot, mon in enumerate(side.pokemon)
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str], bool]:
    order, choice, _line = CASES[name]
    choices = [choice, SIDE_B]
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step([order, "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    switch = bool(((handle.requests[0] or {}).get("forceSwitch") or [False])[0])
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log, switch


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207). The port refuses Disguise
# ("forme change and 1/8 not ported"), so every case is an expected failure until it
# does; `strict` makes the first one that passes say so. Accuracy is pinned to a hit, as
# Showdown's policy has it, and U-turn's pause is an outcome like the finished ones
# (IKA-210: the hit branches used to be picked by Python's events).


@pytest.mark.xfail(strict=True, reason="the port refuses Disguise (IKA-208)")
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_keeps_everything_but_the_absorbed_damage(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, log, switch = _play(oracle, name)
    assert CASES[name][2] in log, f"Showdown did not do what the case says: {log}"
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]
    result = resolve_turn(reg, before, actions, budget=replace(BUDGET, enumerate_accuracy=False))
    outcomes = [b.position for b in result.branches] + [s.position for s in result.suspended]
    assert outcomes, "no outcome"
    for pos in outcomes:
        assert _state(pos) == theirs, f"{name}: showdown {theirs} != port {_state(pos)}"
        # U-turn's switch: Showdown asks for it, the port pauses on it.
        assert self_switches_needed(pos)[0][0] == switch, (name, switch)

