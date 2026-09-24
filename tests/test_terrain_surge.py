"""The Surge abilities put their terrain up, and the terrain does what it says (IKA-201).

Showdown a5df827 (the champions mod changes none of it):

* Electric / Grassy / Misty / Psychic Surge, data/abilities.ts: `onStart(source) {
  this.field.setTerrain('...terrain') }`. `onStart` runs in the `SwitchIn` field event
  (sim/battle.ts `getCallback`), so it fires for the leads before `|turn|1`, for a Pokemon
  switched in, and for a mega evolution that gains the ability (Raichu-Mega-X).
* `Field#setTerrain` (sim/field.ts:130): a terrain already up returns `false` and keeps its
  duration; otherwise the duration is the terrain's `durationCallback(source)` -- 8 when
  the source holds a Terrain Extender, 5 otherwise -- and `eachEvent('TerrainChange')`
  runs at once.
* The Seeds (data/items.ts): `onTerrainChange` / `onStart` (`onSwitchInPriority: -1`) use
  the item when the matching terrain is up; `useItem` applies its `boosts` (Electric and
  Grassy Seed Def +1, Misty and Psychic Seed SpD +1).
* Grassy Terrain (data/moves.ts): `onResidual` (order 5, sub-order 2, ahead of Leftovers)
  heals a grounded Pokemon `baseMaxhp / 16`; `onBasePower` halves Earthquake, Bulldoze and
  Magnitude into a grounded target.
* Misty Terrain's `onBasePower` halves a Dragon move into a grounded *defender* -- the
  attacker's footing does not matter.
* Grassy Glide's `onModifyPriority`: +1 under Grassy Terrain for a grounded user.

Neither engine put the terrain up from an ability (Python had no handler, the port listed
the Surges as inert), so M-C's trial run had no terrain in any of the 1,954 decisions where
a Surge holder stood on the field. Each case is played by Showdown first; the controls are
the same teams without the Surge, where every engine already agreed.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import SideAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.priors import SampledSet
from pokeuraou.selfplay import position_from_sets

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 0, "spe": 0}
MID = {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def _mon(
    species: str, ability: str, moves: list[str], sp: dict[str, int], item: str | None = None
) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, item=item, sp=dict(sp))


RILLABOOM = _mon("Rillaboom", "Grassy Surge", ["protect", "grassyglide", "swordsdance", "woodhammer"], SLOW)
RILLABOOM_PLAIN = _mon("Rillaboom", "Overgrow", ["protect", "grassyglide", "swordsdance", "woodhammer"], SLOW)
RILLABOOM_FAST = _mon("Rillaboom", "Grassy Surge", ["protect", "grassyglide", "swordsdance"], FAST)
RILLABOOM_EXTENDER = _mon(
    "Rillaboom", "Grassy Surge", ["protect", "grassyglide", "swordsdance"], SLOW, item="Terrain Extender"
)
#: Attack and no Speed: slower than Greninja, strong enough to knock it out.
GLIDER_SP = {"hp": 0, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 0}
GLIDER = _mon("Rillaboom", "Grassy Surge", ["protect", "grassyglide", "swordsdance"], GLIDER_SP)
GLIDER_BALLOON = _mon(
    "Rillaboom", "Grassy Surge", ["protect", "grassyglide", "swordsdance"], GLIDER_SP, item="Air Balloon"
)
GRENINJA = _mon("Greninja", "Torrent", ["protect", "darkpulse"], FAST)
INDEEDEE = _mon("Indeedee-F", "Psychic Surge", ["protect", "followme", "psychic"], SLOW)
INDEEDEE_FAST = _mon("Indeedee-F", "Psychic Surge", ["protect", "followme", "psychic"], FAST)
SNEASLER_MOVES = ["protect", "closecombat", "swordsdance"]
SNEASLER_GRASSY = _mon("Sneasler", "Poison Touch", SNEASLER_MOVES, FAST, item="Grassy Seed")
SNEASLER_PSYCHIC = _mon("Sneasler", "Poison Touch", SNEASLER_MOVES, FAST, item="Psychic Seed")
SNEASLER = _mon("Sneasler", "Poison Touch", SNEASLER_MOVES, FAST)
GARCHOMP = _mon("Garchomp", "Rough Skin", ["protect", "earthquake", "dragonclaw", "swordsdance"], FAST)
GARCHOMP_SLOW = _mon("Garchomp", "Rough Skin", ["protect", "earthquake", "dragonclaw", "swordsdance"], MID)
KINGAMBIT = _mon("Kingambit", "Defiant", ["protect", "ironhead", "swordsdance"], FAST)
KINGAMBIT_BULKY = _mon("Kingambit", "Defiant", ["protect", "ironhead", "swordsdance"], BULKY)
INCINEROAR = _mon("Incineroar", "Blaze", ["protect", "knockoff", "swordsdance"], BULKY)
CORVIKNIGHT = _mon("Corviknight", "Pressure", ["protect", "bravebird", "swordsdance"], BULKY)
DRAGONITE = _mon("Dragonite", "Inner Focus", ["protect", "dragonclaw", "swordsdance"], MID)
SYLVEON = _mon("Sylveon", "Pixilate", ["protect", "mistyterrain", "hypervoice"], FAST)
RAICHU_X = _mon("Raichu", "Lightning Rod", ["protect", "thunderbolt", "fakeout"], FAST, item="Raichunite X")
GLIMMORA = _mon("Glimmora", "Toxic Debris", ["protect", "powergem"], BULKY)


def _team(*mons: TeamSet) -> list[TeamSet]:
    """Pads to four with Pokemon nobody in the case uses."""
    out = list(mons)
    for filler in (KINGAMBIT_BULKY, INCINEROAR, GLIMMORA, CORVIKNIGHT):
        if len(out) == 4:
            break
        if all(m.species != filler.species for m in out):
            out.append(filler)
    return out


#: Lead cases: the turn-1 position Showdown starts from, and what Python builds from the
#: same sets. name -> (our team, their team).
LEADS: dict[str, tuple[list[TeamSet], list[TeamSet]]] = {
    "grassy-surge-lead": (_team(RILLABOOM, SNEASLER), _team(GARCHOMP, KINGAMBIT)),
    "terrain-extender-lead": (_team(RILLABOOM_EXTENDER, SNEASLER), _team(GARCHOMP, KINGAMBIT)),
    # Both seeds are used the moment their terrain goes up; the slower Surge's terrain stays.
    "two-surges-slower-wins-with-seeds": (
        _team(RILLABOOM_FAST, SNEASLER_GRASSY), _team(INDEEDEE, SNEASLER_PSYCHIC)
    ),
    "two-surges-the-other-way": (_team(RILLABOOM, SNEASLER_GRASSY), _team(INDEEDEE_FAST, SNEASLER_PSYCHIC)),
    # A Seed for another terrain is kept.
    "seed-kept-under-another-terrain": (_team(RILLABOOM, SNEASLER_PSYCHIC), _team(GARCHOMP, KINGAMBIT)),
    # Control: no Surge, no terrain, the Seed kept.
    "control-no-surge-lead": (_team(RILLABOOM_PLAIN, SNEASLER_GRASSY), _team(GARCHOMP, KINGAMBIT)),
}

#: Mid-game cases. name -> (our team, their team, turns played first, the turn compared).
TURNS: dict[str, tuple] = {
    # Rillaboom comes in; the foe's Grassy Seed is used at once.
    "switch-in-sets-terrain": (
        _team(GARCHOMP, KINGAMBIT, RILLABOOM), _team(SNEASLER_GRASSY, INCINEROAR), [],
        ["switch 3, move 1", "move 1, move 1"],
    ),
    # A Seed holder comes in under a terrain already up and uses its Seed.
    "seed-on-switch-in": (
        _team(RILLABOOM, GARCHOMP), _team(KINGAMBIT, INCINEROAR, SNEASLER_GRASSY), [],
        ["move 1, move 1", "switch 3, move 1"],
    ),
    # Raichu-Mega-X gains Electric Surge.
    "mega-gains-electric-surge": (
        _team(RAICHU_X, GARCHOMP), _team(KINGAMBIT, INCINEROAR), [],
        ["move 1 mega, move 1", "move 1, move 1"],
    ),
    # Rillaboom is healed at the end of the turn; Corviknight is not grounded.
    "grassy-terrain-heals-the-grounded": (
        _team(RILLABOOM, CORVIKNIGHT), _team(KINGAMBIT, SNEASLER), [],
        ["move 3, move 3", "move 2 1, move 2 2"],
    ),
    # Earthquake into grounded foes under Grassy Terrain, halved.
    "grassy-terrain-halves-earthquake": (
        _team(GARCHOMP, RILLABOOM), _team(KINGAMBIT_BULKY, INCINEROAR), [],
        ["move 2, move 1", "move 3, move 3"],
    ),
    "control-earthquake-without-terrain": (
        _team(GARCHOMP, RILLABOOM_PLAIN), _team(KINGAMBIT_BULKY, INCINEROAR), [],
        ["move 2, move 1", "move 3, move 3"],
    ),
    # Misty Terrain (by the move), then Dragon Claws: Dragonite is airborne, the grounded
    # Garchomp it hits is what halves the claw; Corviknight is airborne and is not halved.
    "misty-terrain-reads-the-defender": (
        _team(SYLVEON, DRAGONITE), _team(GARCHOMP_SLOW, CORVIKNIGHT),
        [["move 2, move 3", "move 4, move 3"]],
        ["move 1, move 2 1", "move 3 2, move 1"],
    ),
    # A second Grassy Terrain on a terrain already up fails and keeps its duration.
    "same-terrain-keeps-its-duration": (
        _team(_mon("Whimsicott", "Infiltrator", ["protect", "grassyterrain", "moonblast"], FAST), RILLABOOM),
        _team(KINGAMBIT, INCINEROAR),
        [["move 1, move 3", "move 3, move 3"]],
        ["move 2, move 3", "move 3, move 3"],
    ),
    # An airborne Rillaboom (Air Balloon) gets no priority from Grassy Glide: the faster
    # Greninja's Dark Pulse lands on Garchomp before the Glide knocks Greninja out. (Nothing
    # hits the balloon: neither engine pops it, noted in IKA-201.)
    "grassy-glide-needs-ground": (
        _team(GLIDER_BALLOON, GARCHOMP), _team(GRENINJA, INCINEROAR), [],
        ["move 2 1, move 4", "move 2 2, move 1"],
    ),
    # The same Rillaboom on the ground: the Glide goes first and Garchomp is not hit.
    "grassy-glide-grounded-goes-first": (
        _team(GLIDER, GARCHOMP), _team(GRENINJA, INCINEROAR), [],
        ["move 2 1, move 4", "move 2 2, move 1"],
    ),
    # Control: Misty Terrain by the move, a grounded Dragon Claw into a grounded target,
    # halved by both readings.
    "control-misty-grounded-into-grounded": (
        _team(SYLVEON, GARCHOMP_SLOW), _team(GARCHOMP, KINGAMBIT_BULKY),
        [["move 2, move 4", "move 4, move 3"]],
        ["move 1, move 3 1", "move 1, move 1"],
    ),
}

#: No crit, no chance-based secondary, the maximum roll -- what the oracle's policy pins.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _board(pos: Position) -> tuple:
    """What a terrain case can change: the field and every Pokemon's HP, item and boosts."""
    mons = []
    for side in pos.sides:
        for mon in side.pokemon:
            boosts = tuple(sorted((k, v) for k, v in mon.boosts.items() if v))
            mons.append((mon.species, mon.hp, mon.item, boosts))
    return (pos.field.terrain, pos.field.terrain_duration, tuple(sorted(mons)))


def _sampled(reg, team: list[TeamSet]) -> list[SampledSet]:  # noqa: ANN001
    def ident(name: str | None) -> str | None:
        return None if not name else "".join(ch for ch in name.lower() if ch.isalnum())

    return [
        SampledSet(
            species=ident(t.species), ability=ident(t.ability), item=ident(t.item),
            nature=t.nature, sp=dict(t.sp), moves=list(t.moves),
        )
        for t in team
    ]


def _lead(oracle: Oracle, name: str) -> Position:
    ours, theirs = LEADS[name]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    pos = Position.from_json(handle.position)
    handle.close()
    return pos


def _play(oracle: Oracle, name: str) -> tuple[Position, Position, list[str]]:
    ours, theirs, setup, choices = TURNS[name]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    return before, after, log


def _actions(reg, pos: Position, choices: list[str]) -> list[SideAction]:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


EXPECTED_TERRAIN = {
    "grassy-surge-lead": ("grassyterrain", 5),
    "terrain-extender-lead": ("grassyterrain", 8),
    "two-surges-slower-wins-with-seeds": ("psychicterrain", 5),
    "two-surges-the-other-way": ("grassyterrain", 5),
    "seed-kept-under-another-terrain": ("grassyterrain", 5),
    "control-no-surge-lead": (None, None),
}


@pytest.mark.parametrize("name", sorted(LEADS))
def test_showdown_starts_with_the_leads_terrain(oracle: Oracle, name: str) -> None:
    pos = _lead(oracle, name)
    assert (pos.field.terrain, pos.field.terrain_duration) == EXPECTED_TERRAIN[name]


@pytest.mark.parametrize("name", sorted(LEADS))
def test_the_port_starts_from_the_position_showdown_does(
    reg, oracle: Oracle, name: str, monkeypatch: pytest.MonkeyPatch  # noqa: ANN001
) -> None:
    """`position_from_sets` with the leads' switch-ins run by the port (IKA-210)."""
    from pokeuraou import selfplay

    from . import _port

    monkeypatch.setattr(selfplay, "apply_lead_abilities", _port.apply_lead_abilities, raising=False)
    showdown = _lead(oracle, name)
    ours, theirs = LEADS[name]
    opened = position_from_sets(reg, _sampled(reg, ours), _sampled(reg, theirs))
    assert _board(opened) == _board(showdown)


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(TURNS))
def test_the_port_plays_the_turn_as_showdown_does(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    before, after, log = _play(oracle, name)
    moves = [line for line in log if line.startswith("|move|")]
    if name.startswith("grassy-glide-"):
        # The airborne case is a case only if Greninja moved first, and the grounded one
        # only if the Glide knocked it out before it could.
        airborne = name == "grassy-glide-needs-ground"
        assert any("Dark Pulse" in line for line in moves) == airborne, moves
        assert any(line.startswith("|faint|p2a: Greninja") for line in log), log[-12:]
    # The port declines a position with a stats override (it reads that as a Transform);
    # both engines compute the same stats from the spreads.
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    actions = _actions(reg, before, TURNS[name][3])
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) >= 1 and not there.suspended
    for index in range(len(there.branches)):
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        assert _board(chosen.position) == _board(after)
