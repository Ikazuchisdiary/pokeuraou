"""The user's Electric Terrain and Baton Pass build (IKA-419), played against Showdown.

The build: Mega Raichu X (Electric Surge on the Mega Evolution) sets Electric Terrain,
Espathra (Speed Boost, Electric Seed) raises its stats with Calm Mind, the Seed and Speed
Boost, and Baton Pass hands them to a partner; Politoed's rain and Mega Charizard Y's sun
play into Solar Beam, Electro Shot and Weather Ball.

Showdown a5df827, the pieces each case looks at:

* data/pokedex.ts `raichumegax: abilities: { 0: "Electric Surge" }`; the Mega Evolution
  runs the new ability's `onStart`, which sets the terrain before any move.
* data/items.ts `electricseed`: `onStart` and `onTerrainChange` use the item under
  Electric Terrain, `boosts: { def: 1 }` (Defense, not Special Defense).
* data/abilities.ts `speedboost`: `onResidual(pokemon) { if (pokemon.activeTurns)
  this.boost({ spe: 1 }); }`.
* data/moves.ts `batonpass`: `selfSwitch: 'copyvolatile'`, so the replacement gets the
  user's boosts and its copyable volatiles (`Pokemon#copyVolatileFrom`).
* data/moves.ts `risingvoltage`: 140 power on a grounded target in Electric Terrain;
  `luminacrash`: a 100% `spd: -2` secondary.
* Solar Beam fires at once in the sun and charges (and hits at half power) in the rain;
  Electro Shot fires at once in the rain; Weather Ball is a 100-power Fire move in the sun.

Each case plays in Showdown and holds the port's one outcome (deterministic budget, a
paused turn resumed with Showdown's replacement) to it. A case named ``known-gap-*``
records a divergence the port has today: it is marked ``xfail(strict=True)``, so a port
that starts to agree with Showdown fails it and the mark has to come off.
"""

from __future__ import annotations

import os

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import ally_targets, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, resolve_turn, resume_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

BUDGET = Budget.deterministic(0)

#: The user's sets (examples/diag-team-ika419.json), with moves picked for the cases.
RAICHU = TeamSet(
    species="Raichu", ability="Lightning Rod", nature="Timid", item="Raichunite X",
    moves=["Fake Out", "Rising Voltage", "Reflect", "Protect"],
    sp={"hp": 29, "def": 5, "spe": 32},
)
ESPATHRA = TeamSet(
    species="Espathra", ability="Speed Boost", nature="Timid", item="Electric Seed",
    moves=["Lumina Crash", "Baton Pass", "Calm Mind", "Protect"],
    sp={"hp": 2, "spa": 32, "spe": 32},
)
CHARIZARD = TeamSet(
    species="Charizard", ability="Blaze", nature="Timid", item="Charizardite Y",
    moves=["Heat Wave", "Weather Ball", "Solar Beam", "Protect"],
    sp={"hp": 22, "def": 15, "spa": 11, "spd": 1, "spe": 17},
)
WHIMSICOTT = TeamSet(
    species="Whimsicott", ability="Chlorophyll", nature="Timid", item="Life Orb",
    moves=["Moonblast", "Solar Beam", "Psychic", "Protect"],
    sp={"hp": 2, "spa": 32, "spe": 32},
)
POLITOED = TeamSet(
    species="Politoed", ability="Drizzle", nature="Calm", item="Sitrus Berry",
    moves=["Muddy Water", "Encore", "Rain Dance", "Psych Up"],
    sp={"hp": 32, "def": 24, "spd": 7, "spe": 3},
)
ARCHALUDON = TeamSet(
    species="Archaludon", ability="Stamina", nature="Bold", item="Leftovers",
    moves=["Electro Shot", "Dragon Pulse", "Snarl", "Protect"],
    sp={"hp": 32, "def": 10, "spa": 5, "spd": 18, "spe": 1},
)


def _foe(species: str, ability: str, item: str | None = None) -> TeamSet:
    """Bulky foes that only Protect or Helping Hand (Splash is not in Reg M-C), so the
    user's side acts on a known board."""
    return TeamSet(
        species=species, ability=ability, nature="Serious", item=item,
        moves=["Protect", "Helping Hand"],
        sp={"hp": 32, "def": 16, "spd": 16},
    )


#: Incineroar is grounded and immune to Lumina Crash; Corviknight (Flying) is not grounded.
FOES = [_foe("Incineroar", "Blaze"), _foe("Corviknight", "Pressure"), _foe("Milotic", "Marvel Scale")]
#: Foe choice: both Helping Hand (nothing reaches my side).
IDLE = "move 2 -2, move 2 -1"

#: The volatiles a case compares (the charging turn's, and Baton Pass's copyable ones).
VOLATILES = {"twoturnmove", "solarbeam", "electroshot", "focusenergy", "substitute"}


def _case(mine, team, setup, turn, shown, replace=None, port_turn=None):  # noqa: ANN001, ANN202
    return (mine, team, setup, turn, shown, replace, port_turn or turn)


#: The divergences the port has today, each with its reason (see the module docstring).
KNOWN_GAPS = {
    "known-gap-baton-pass-hands-the-boosts-on": (
        "IKA-419: the port reads Baton Pass's selfSwitch 'copyvolatile' as a plain self-switch "
        "and notes 'status move: batonpass'; the boosts stay behind"
    ),
    "known-gap-rising-voltage-terrain-ungrounded": (
        "IKA-419: moveinfo.rs doubles Rising Voltage in Electric Terrain without "
        "Showdown's target.isGrounded(); no note"
    ),
}

#: name -> (my six or fewer, team order, setup turns, compared turn, Showdown log line,
#: my replacement choice when the compared turn asks for one, the compared turn as the
#: port's menu names it when that differs: a charging move's second turn is Showdown's
#: only "move 1" but the port keeps the move's own index).
CASES: dict[str, tuple] = {
    # Mega Raichu X sets the terrain at the Mega Evolution; Espathra beside it eats its Seed
    # at once (Defense +1), and Speed Boost raises Speed at the end of the turn.
    "mega-raichu-x-surge-seed-speed-boost": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 3 mega, move 4", IDLE], "|-enditem|p1b: Espathra|Electric Seed",
    ),
    # Control: without the Mega Evolution there is no terrain and the Seed stays.
    "control-no-mega-no-terrain": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 3, move 4", IDLE], "|move|p1a: Raichu|Reflect",
    ),
    # Rising Voltage in the terrain into grounded Incineroar: 140 power.
    "rising-voltage-terrain-grounded": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 2 1 mega, move 4", IDLE], "|move|p1a: Raichu|Rising Voltage|p2a: Incineroar",
    ),
    # Rising Voltage in the terrain into Corviknight, which is not grounded: 70 power.
    "known-gap-rising-voltage-terrain-ungrounded": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 2 2 mega, move 4", IDLE], "|move|p1a: Raichu|Rising Voltage|p2b: Corviknight",
    ),
    # Control: the same hit into Corviknight with no terrain (70 power in both engines).
    "control-rising-voltage-no-terrain-ungrounded": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 2 2, move 4", IDLE], "|move|p1a: Raichu|Rising Voltage|p2b: Corviknight",
    ),
    # Control: Rising Voltage with no terrain.
    "control-rising-voltage-no-terrain": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 2 1, move 4", IDLE], "|move|p1a: Raichu|Rising Voltage|p2a: Incineroar",
    ),
    # Lumina Crash into Corviknight: damage and Special Defense -2.
    "lumina-crash-lowers-spd": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 4, move 1 2", IDLE], "|-unboost|p2b: Corviknight|spd|2",
    ),
    # ... in the terrain the Mega just set, with the Seed and Speed Boost on the user.
    "lumina-crash-in-terrain": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 4 mega, move 1 2", IDLE], "|-unboost|p2b: Corviknight|spd|2",
    ),
    # Calm Mind (with Seed and Speed Boost) on turn 1, Baton Pass to Charizard on turn 2:
    # the boosts go with it.
    "known-gap-baton-pass-hands-the-boosts-on": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [["move 3 mega, move 3", IDLE]],
        ["move 4, move 2", IDLE], "|switch|p1b: Charizard", replace="pass, switch 3",
    ),
    # Control: the same switch by an ordinary switch action drops the boosts.
    "control-plain-switch-drops-the-boosts": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [["move 3 mega, move 3", IDLE]],
        ["move 4, switch 3", IDLE], "|switch|p1b: Charizard",
    ),
    # Baton Pass with no boosts: the same switch as any self-switch.
    "baton-pass-with-nothing-to-hand-on": _case(
        [RAICHU, ESPATHRA, CHARIZARD], "team 123", [],
        ["move 4, move 2", IDLE], "|switch|p1b: Charizard", replace="pass, switch 3",
    ),
    # Espathra comes in under the terrain: the Seed's onStart.
    "seed-on-switch-in-under-terrain": _case(
        [RAICHU, CHARIZARD, ESPATHRA], "team 123", [["move 3 mega, move 4", IDLE]],
        ["move 4, switch 3", IDLE], "|-enditem|p1b: Espathra|Electric Seed",
    ),
    # Politoed's Psych Up copies Espathra's boosts (the user's team has it). The menu lists
    # the ally target only under `ally_targets('all'|'benefit')`; the test asks for 'all'.
    "psych-up-copies-espathra": _case(
        [ESPATHRA, POLITOED, RAICHU], "team 123", [["move 3, move 3", IDLE]],
        ["move 4, move 4 -1", IDLE], "|-copyboost|p1b: Politoed|p1a: Espathra",
    ),
    # Mega Charizard Y's Drought, then Whimsicott's Solar Beam fires at once in the sun.
    "solar-beam-in-sun": _case(
        [CHARIZARD, WHIMSICOTT, RAICHU], "team 123", [],
        ["move 4 mega, move 2 2", IDLE], "|-anim|p1b: Whimsicott|Solar Beam|p2b: Corviknight",
    ),
    # Weather Ball in the sun: a 100-power Fire move.
    "weather-ball-in-sun": _case(
        [CHARIZARD, WHIMSICOTT, RAICHU], "team 123", [],
        ["move 2 2 mega, move 4", IDLE], "|move|p1a: Charizard|Weather Ball|p2b: Corviknight",
    ),
    # Weather Ball in the rain: a 100-power Water move (Politoed's Drizzle).
    "weather-ball-in-rain": _case(
        [POLITOED, CHARIZARD, RAICHU], "team 123", [],
        ["move 3, move 2 1", IDLE], "|move|p1b: Charizard|Weather Ball|p2a: Incineroar",
    ),
    # Solar Beam in the rain: the charging turn.
    "solar-beam-in-rain-charges": _case(
        [POLITOED, WHIMSICOTT, RAICHU], "team 123", [],
        ["move 3, move 2 1", IDLE], "|-prepare|p1b: Whimsicott|Solar Beam",
    ),
    # ... and the hit at half power on the next turn.
    "solar-beam-in-rain-half-power": _case(
        [POLITOED, WHIMSICOTT, RAICHU], "team 123", [["move 3, move 2 2", IDLE]],
        ["move 3, move 1", IDLE], "|move|p1b: Whimsicott|Solar Beam|p2b: Corviknight",
        port_turn=["move 3, move 2", IDLE],
    ),
    # Solar Beam in Electric Terrain and no weather: it charges; the terrain plays no part.
    "solar-beam-in-electric-terrain-charges": _case(
        [RAICHU, WHIMSICOTT, CHARIZARD], "team 123", [],
        ["move 3 mega, move 2 1", IDLE], "|-prepare|p1b: Whimsicott|Solar Beam",
    ),
    # Electro Shot in the rain: Special Attack +1 and the hit on the same turn.
    "electro-shot-in-rain": _case(
        [POLITOED, ARCHALUDON, RAICHU], "team 123", [],
        ["move 3, move 1 1", IDLE], "|-anim|p1b: Archaludon|Electro Shot|p2a: Incineroar",
    ),
    # Electro Shot in Electric Terrain and no rain: it charges (Special Attack +1).
    "electro-shot-in-terrain-charges": _case(
        [RAICHU, ARCHALUDON, ESPATHRA], "team 123", [],
        ["move 3 mega, move 1 1", IDLE], "|-prepare|p1b: Archaludon|Electro Shot",
    ),
}


def _state(pos: Position) -> dict[str, object]:
    out: dict[str, object] = {
        "weather": pos.field.weather,
        "terrain": pos.field.terrain,
    }
    for index, side in enumerate(pos.sides):
        out[f"p{index + 1}.active"] = tuple(
            side.pokemon[a].species if a is not None else None for a in side.active
        )
        for mon in side.pokemon:
            out[f"p{index + 1}.{mon.base_species}"] = (
                mon.species,
                mon.hp,
                mon.fainted,
                mon.item,
                tuple(sorted((k, v) for k, v in mon.boosts.items() if v)) if not mon.fainted else (),
                tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            )
    return out


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        with ally_targets("all"):
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


def _params() -> list:
    out = []
    for name in sorted(CASES):
        marks = []
        if name.startswith("known-gap-"):
            marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[name]))
        out.append(pytest.param(name, marks=marks))
    return out


def test_every_known_gap_is_a_case() -> None:
    assert set(KNOWN_GAPS) == {n for n in CASES if n.startswith("known-gap-")}


def play(reg, oracle: Oracle, name: str) -> tuple[dict, dict, tuple[str, ...]]:  # noqa: ANN001
    """Showdown's state after the compared turn, the port's, and the port's notes."""
    mine, team, setup, choices, shown, replace, port_choices = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, FOES, policy=RandomnessPolicy())
    handle.step([team, "team 123"])
    assert handle.choice_errors == [], handle.choice_errors
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    log = list(handle.log)
    owed = (handle.requests[0] or {}).get("forceSwitch")
    if owed:
        assert replace is not None, f"{name}: Showdown asks for a replacement: {owed}"
        handle.step([replace, None])
        assert handle.choice_errors == [], handle.choice_errors
        log += handle.log
    theirs = _state(Position.from_json(handle.position))
    handle.close()
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"

    result = resolve_turn(reg, before, _actions(reg, before, port_choices), budget=BUDGET)
    notes = tuple(result.unmodelled)
    if owed:
        assert len(result.suspended) == 1 and not result.branches, (result.branches, result.suspended)
        pause = result.suspended[0]
        pick = next(
            o
            for o in switch_actions_after_faint(reg, pause.position, 0, list(owed))
            if o.to_choice() == replace
        )
        foe_pass = next(iter(switch_actions_after_faint(reg, pause.position, 1, [False, False])))
        result = resume_turn(reg, pause, [pick, foe_pass])
        notes += tuple(result.unmodelled)
    assert len(result.branches) == 1 and not result.suspended, (result.branches, result.suspended)
    return theirs, _state(result.branches[0].position), notes


@pytest.mark.parametrize("name", _params())
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    theirs, ours, _notes = play(reg, oracle, name)
    diff = {k: (theirs[k], ours.get(k)) for k in theirs if theirs[k] != ours.get(k)}
    assert not diff, f"{name}: (showdown, port) {diff}"
