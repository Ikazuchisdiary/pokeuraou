"""When two Pokemon come in at once, whose Intimidate hits whom.

There are two phases that bring two Pokemon in together, and Showdown orders them
differently. Reported as one rule -- "both switches finish, then Intimidate" -- which is
right for one of them and wrong for the other, so both are pinned here against the
simulator rather than against a reading of it.

**An in-turn double switch is interleaved.** `switchIn` queues a `runSwitch` action at
order 101 and a `switch` is order 103, so the `runSwitch` is spliced *ahead* of the other
side's pending switch and its `SwitchIn` event fires immediately:

    switch p1a Incineroar
    Incineroar's Intimidate -> p2a Venusaur -1, p2b Sylveon -1
    switch p2a Arcanine                      (Venusaur leaves, taking the drop with it)
    Arcanine's Intimidate   -> p1a Incineroar -1, p1b Toxapex -1

So the side that switches *second* brings its Pokemon in clean, and the first switcher's
Intimidate is spent on a Pokemon that is about to leave. Which side goes first is decided
by the Speed of the Pokemon going *out*, not the one coming in.

**A replacement phase is not.** Both replacements are placed before any switch-in effect
runs -- Showdown's `runSwitch` drains every consecutive `runSwitch` from the queue and
fires one `SwitchIn` field event over all of them, in Speed order -- so here both
newcomers really do take the other's Intimidate:

    switch p1a Incineroar
    switch p2a Arcanine
    Arcanine's Intimidate   -> p1a Incineroar -1, p1b Froslass -1      (Speed 30 first)
    Incineroar's Intimidate -> p2a Arcanine -1, p2b Polteageist -1

The difference is worth a test because it is a real asymmetry in play -- whether a switch
eats an Intimidate depends on which phase it happens in -- and because a single
"place everything, then run the effects" implementation would pass one of these and fail
the other.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, replacements_needed, resolve_replacements, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


#: Slot 3 of each side holds the Intimidate, at Speeds far enough apart that the order it
#: comes in at is decided rather than a coin flip.
INTIMIDATE_A = _mon(
    "Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 4
)
INTIMIDATE_B = _mon(
    "Arcanine", "Intimidate", ["flareblitz", "extremespeed", "protect", "willowisp"], 30
)


def _attack_boosts(pos: Position) -> dict[str, int]:
    """Attack stage per active slot, keyed by slot and species."""
    out: dict[str, int] = {}
    for side_index, side in enumerate(pos.sides):
        for slot, party in enumerate(side.active):
            if party is None:
                continue
            mon = side.pokemon[party]
            out[f"p{side_index + 1}{'ab'[slot]} {mon.species}"] = mon.boosts.get("atk", 0)
    return out


def test_an_in_turn_double_switch_is_interleaved(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Showdown's answer, and ours, for two switches chosen in the same turn."""
    team_a = [
        _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"], 20),
        _mon("Toxapex", "Regenerator", ["recover", "protect", "scald", "toxic"], 20),
        INTIMIDATE_A,
        _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"], 20),
    ]
    team_b = [
        _mon("Venusaur", "Chlorophyll", ["sludgebomb", "gigadrain", "protect", "leechseed"], 20),
        _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"], 20),
        INTIMIDATE_B,
        _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 20),
    ]
    choices = ["switch 3, move 2", "switch 3, move 2"]

    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs_answer = _attack_boosts(Position.from_json(handle.position))
    log = [line for line in handle.log if "-ability" in line or "|switch|" in line]
    handle.close()

    # The log shape is the claim: a switch, then its own ability, then the other switch.
    order = [
        ("switch" if "|switch|" in line else "ability")
        for line in log
        # The oracle emits the split-log pair for each switch; one is enough.
        if not line.startswith("|switch|p") or "/100" not in line
    ]
    assert order[:3] == ["switch", "ability", "switch"], (
        f"an in-turn double switch is interleaved, not batched: {log}"
    )

    ours = next(a for a in side_actions(reg, before, 0) if a.to_choice() == choices[0])
    theirs = next(a for a in side_actions(reg, before, 1) if a.to_choice() == choices[1])
    result = resolve_turn(reg, before, [ours, theirs], budget=Budget.exact())
    mine = _attack_boosts(max(result.branches, key=lambda b: b.probability).position)
    assert mine == theirs_answer, f"showdown {theirs_answer} != ours {mine}"

    # And the consequence, stated directly: exactly one of the two newcomers escaped.
    newcomers = {k: v for k, v in mine.items() if "incineroar" in k or "arcanine" in k}
    assert sorted(newcomers.values()) == [-1, 0], (
        f"the second switcher comes in clean, the first one's Intimidate is wasted on a "
        f"Pokemon that is leaving: {newcomers}"
    )


def test_a_replacement_phase_places_both_before_either_ability(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Showdown's answer, and ours, for two replacements after a double faint.

    Both leads use Explosion. `selfdestruct: 'always'` faints the user whatever the blast
    hits, and every other active is a Ghost type, so the Normal-type move hits nothing:
    two faints, no collateral, one replacement owed on each side. Engineering the same
    thing out of damage rolls takes tuning and can still fail on a resistance.
    """
    team_a = [
        _mon("Gengar", "Cursed Body", ["explosion", "protect", "shadowball", "sludgebomb"], 20),
        _mon("Froslass", "Cursed Body", ["protect", "shadowball", "icebeam", "willowisp"], 20),
        INTIMIDATE_A,
        _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"], 20),
    ]
    team_b = [
        _mon("Chandelure", "Flash Fire", ["explosion", "protect", "shadowball", "flamethrower"], 20),
        _mon("Polteageist", "Cursed Body", ["protect", "shadowball", "storedpower", "strengthsap"], 20),
        INTIMIDATE_B,
        _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 20),
    ]

    handle = oracle.create(
        FORMAT_ID, team_a, team_b, policy=RandomnessPolicy(damage_roll=0)
    )
    handle.step(["team 1234", "team 1234"])
    handle.step(["move 1, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    owed = replacements_needed(before)
    assert any(owed[0]) and any(owed[1]), (
        f"the setup has to leave both sides owing a replacement: {owed}"
    )

    handle.step(["switch 3", "switch 3"])
    assert handle.choice_errors == [], handle.choice_errors
    theirs_answer = _attack_boosts(Position.from_json(handle.position))
    log = [line for line in handle.log if "-ability" in line or "|switch|" in line]
    handle.close()

    order = [
        ("switch" if "|switch|" in line else "ability")
        for line in log
        if not line.startswith("|switch|p") or "/100" not in line
    ]
    assert order[:3] == ["switch", "switch", "ability"], (
        f"a replacement phase places both, then runs the abilities: {log}"
    )

    picks = [
        next(
            a
            for a in switch_actions_after_faint(reg, before, side, owed[side])
            if "switch 3" in a.to_choice()
        )
        for side in (0, 1)
    ]
    mine = _attack_boosts(resolve_replacements(reg, before, picks).position)
    assert mine == theirs_answer, f"showdown {theirs_answer} != ours {mine}"

    # The consequence, and the difference from the in-turn case: neither escapes.
    newcomers = {k: v for k, v in mine.items() if "incineroar" in k or "arcanine" in k}
    assert sorted(newcomers.values()) == [-1, -1], (
        f"both newcomers are on the field when the other's Intimidate fires: {newcomers}"
    )
