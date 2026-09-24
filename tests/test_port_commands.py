"""The port's own mid-turn replacement, replacement phase and leads, held to Showdown (IKA-211).

`resolve` could hand back only the weight of a turn that a self-switching move stopped, and
the replacement phase and the leads' switch-ins were Python's alone. The node's `turn`,
`alternatives`, `replacements` and `leads` commands answer them now. Each is played by
Showdown first and the port is held to it, as the other `test_the_port_matches_showdown`
tests hold `resolve`:

- **Mid-turn.** Rillaboom U-turns out and a Trace Gardevoir comes in: the pause, and the
  rest of the turn resumed from it (`turn` with a pause), is one outcome per foe Trace can
  copy, each Showdown's for that `sample`. The pause crosses the pipe as data and comes
  back; `select` names the same pause and the same branch as the full answer does.
- **Replacement phase.** Two Explosions leave both sides owing one: both Intimidates are
  placed before either fires (`runSwitch` is order 101, after both switches).
- **Leads.** A lead Trace between Intimidate and Drought: the first foe, noted, without a
  generator; Showdown's two answers with one, drawn in the same order as Python draws them.

`tools/diff_commands.py` is the Python comparison over recorded games.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.actions import PassAction, SideAction, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from . import _port
from ._port import Budget, replacements_needed
from .conftest import FORMAT_ID

FAST = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def _mon(species: str, ability: str, moves: list[str], sp: dict[str, int]) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, sp=dict(sp))


GARDE = _mon("Gardevoir", "Trace", ["protect", "thunderwave", "hypervoice", "calmmind"], FAST)
INCIN = _mon("Incineroar", "Intimidate", ["protect", "fakeout", "willowisp", "knockoff"], SLOW)
TORK = _mon("Torkoal", "Drought", ["protect", "willowisp", "eruption", "toxic"], SLOW)
KING = _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], SLOW)
CHOMP = _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], SLOW)
WHIM = _mon("Whimsicott", "Prankster", ["protect", "thunderwave", "moonblast", "tailwind"], FAST)
RILLA = _mon("Rillaboom", "Overgrow", ["uturn", "protect", "woodhammer", "fakeout"], FAST)
INTIMIDATE_AND_DROUGHT = [INCIN, TORK, KING, CHOMP]

#: What the oracle's policy pins, and the same with the draws branched.
BUDGET = replace(
    Budget.exact(),
    enumerate_crit=False,
    enumerate_secondary=False,
    enumerate_accuracy=False,
    enumerate_status_checks=False,
).with_fixed_roll(0)
BRANCHING = replace(BUDGET, enumerate_secondary=True)
PASSES = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))


@pytest.fixture(autouse=True)
def _leads_on_the_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """`position_from_sets` runs the leads' switch-ins through the port, not Python (IKA-210)."""
    from pokeuraou import selfplay

    monkeypatch.setattr(selfplay, "apply_lead_abilities", _port.apply_lead_abilities)


@pytest.fixture()
def node(reg):  # noqa: ANN001, ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    opened = rustnode.RustNode(reg)
    yield opened
    opened.close()


def _board(pos: Position) -> tuple:
    out: list[tuple | None] = []
    for side in pos.sides:
        for party in side.active:
            if party is None:
                out.append(None)
                continue
            mon = side.pokemon[party]
            boosts = tuple(sorted((k, v) for k, v in mon.boosts.items() if v))
            out.append((mon.species, mon.ability, mon.status, boosts, mon.hp))
    return (tuple(out), pos.field.weather)


def _unstat(pos: Position) -> Position:
    """The port declines a stats override (it reads one as a Transform)."""
    for side in pos.sides:
        for party in side.pokemon:
            party.stats_override = None
    return pos


def _actions(reg, pos: Position, choices: list[str]) -> list[SideAction]:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side])
        for side in (0, 1)
    ]


# ---------------------------------------------------------------------------
# Mid-turn: a U-turn into a Trace
# ---------------------------------------------------------------------------

U_TURN = ["move 1 2, move 1", "move 1, move 2 2"]


def _u_turn_boards(oracle: Oracle) -> tuple[Position, dict[str, tuple]]:
    boards = {}
    before = None
    for sample in ("first", "last"):
        handle = oracle.create(
            FORMAT_ID, [RILLA, WHIM, GARDE, CHOMP], INTIMIDATE_AND_DROUGHT,
            policy=RandomnessPolicy(sample=sample),
        )
        handle.step(["team 1234", "team 1234"])
        before = Position.from_json(handle.position)
        handle.step(U_TURN)
        assert handle.choice_errors == [], handle.choice_errors
        assert (handle.requests[0] or {}).get("forceSwitch") == [True, False]
        handle.step(["switch 3, pass", None])
        assert handle.choice_errors == [], handle.choice_errors
        boards[sample] = _board(Position.from_json(handle.position))
        handle.close()
    assert boards["first"] != boards["last"]
    return _unstat(before), boards


@pytest.mark.oracle
def test_the_port_resumes_a_mid_turn_replacement_as_showdown_does(reg, oracle: Oracle, node) -> None:  # noqa: ANN001
    before, boards = _u_turn_boards(oracle)
    actions = _actions(reg, before, U_TURN)
    turn = node.turn(before, actions, BRANCHING, full=True)
    assert turn is not None, "the port refused the turn"
    assert turn.pauses and not turn.outcomes
    for k, pause in enumerate(turn.pauses):
        # `select` names this pause as the full answer does, and it crossed as data.
        chosen = node.turn(before, actions, BRANCHING, select=len(turn.outcomes) + k)
        assert chosen is not None and chosen.pause is not None
        assert chosen.pause.raw == pause.raw
        option = next(
            o for o in switch_actions_after_faint(reg, pause.position, 0, [True, False])
            if o.to_choice() == "switch 3, pass"
        )
        resumed = node.resume(pause, [option, PASSES], full=True)
        assert resumed is not None, "the port refused the rest of the turn"
        got = sorted((b.probability, _board(b.position)) for b in resumed.outcomes)
        assert [p for p, _b in got] == pytest.approx([pause.probability / 2] * 2, abs=1e-15)
        assert {b for _p, b in got} == {boards["first"], boards["last"]}
        for index, outcome in enumerate(resumed.outcomes):
            one = node.resume(pause, [option, PASSES], select=index)
            assert one is not None and one.position is not None
            assert one.position.to_json() == outcome.position.to_json()
        # The node's own walk through the pause: the same chooser and the same turn.
        answered = node.resume_alternatives(pause)
        assert answered is not None
        chooser, alternatives = answered
        assert chooser == 0
        assert [o.to_choice() for o, _ in alternatives] == [
            o.to_choice() for o in switch_actions_after_faint(reg, pause.position, 0, [True, False])
        ]
        picked = next(r for o, r in alternatives if o.to_choice() == "switch 3, pass")
        assert [b.position.to_json() for b in picked.outcomes] == [
            b.position.to_json() for b in resumed.outcomes
        ]


@pytest.mark.oracle
def test_the_pinned_budget_resumes_into_the_first_foe_and_says_so(reg, oracle: Oracle, node) -> None:  # noqa: ANN001
    before, boards = _u_turn_boards(oracle)
    actions = _actions(reg, before, U_TURN)
    turn = node.turn(before, actions, BUDGET, full=True)
    assert turn is not None and len(turn.pauses) == 1
    pause = turn.pauses[0]
    option = next(
        o for o in switch_actions_after_faint(reg, pause.position, 0, [True, False])
        if o.to_choice() == "switch 3, pass"
    )
    resumed = node.resume(pause, [option, PASSES], full=True)
    assert resumed is not None and len(resumed.outcomes) == 1
    assert _board(resumed.outcomes[0].position) == boards["first"]
    assert "trace target (the first; not branched)" in resumed.unmodelled


# ---------------------------------------------------------------------------
# The replacement phase: both placed before either Intimidate
# ---------------------------------------------------------------------------


def _intimidated(pos: Position) -> dict[str, int]:
    return {
        f"p{side_index + 1}{'ab'[slot]} {side.pokemon[party].species}":
            side.pokemon[party].boosts.get("atk", 0)
        for side_index, side in enumerate(pos.sides)
        for slot, party in enumerate(side.active)
        if party is not None
    }


def _spread(spe: int) -> dict[str, int]:
    return {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}


@pytest.mark.oracle
def test_the_port_places_both_replacements_before_either_ability(reg, oracle: Oracle, node) -> None:  # noqa: ANN001
    """Both leads Explode into Ghosts: two faints and nothing else, one owed on each side."""

    def mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
        return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=_spread(spe))

    team_a = [
        mon("Gengar", "Cursed Body", ["explosion", "protect", "shadowball", "sludgebomb"], 20),
        mon("Froslass", "Cursed Body", ["protect", "shadowball", "icebeam", "willowisp"], 20),
        mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 4),
        mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"], 20),
    ]
    team_b = [
        mon("Chandelure", "Flash Fire", ["explosion", "protect", "shadowball", "flamethrower"], 20),
        mon("Polteageist", "Cursed Body", ["protect", "shadowball", "storedpower", "strengthsap"], 20),
        mon("Arcanine", "Intimidate", ["flareblitz", "extremespeed", "protect", "willowisp"], 30),
        mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 20),
    ]
    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    handle.step(["move 1, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    before = _unstat(Position.from_json(handle.position))
    owed = replacements_needed(before)
    assert any(owed[0]) and any(owed[1]), owed
    handle.step(["switch 3", "switch 3"])
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _intimidated(Position.from_json(handle.position))
    handle.close()

    assert node.replacements_needed(before) == tuple(tuple(side) for side in owed)
    picks = [
        next(
            a for a in switch_actions_after_faint(reg, before, side, list(owed[side]))
            if "switch 3" in a.to_choice()
        )
        for side in (0, 1)
    ]
    phase = node.resolve_replacements(before, picks)
    assert phase is not None, "the port refused the phase"
    assert _intimidated(phase.position) == theirs, f"showdown {theirs} != port {_intimidated(phase.position)}"


# ---------------------------------------------------------------------------
# The leads: a Trace between two abilities
# ---------------------------------------------------------------------------


@pytest.mark.oracle
def test_the_port_runs_the_leads_as_showdown_does(reg, oracle: Oracle, node) -> None:  # noqa: ANN001
    from pokeuraou.priors import SampledSet
    from pokeuraou.regulation import to_id
    from pokeuraou.selfplay import position_from_sets

    ours = [GARDE, WHIM, KING, CHOMP]
    boards = {}
    for sample in ("first", "last"):
        handle = oracle.create(
            FORMAT_ID, ours, INTIMIDATE_AND_DROUGHT, policy=RandomnessPolicy(sample=sample)
        )
        handle.step(["team 1234", "team 1234"])
        boards[sample] = _board(Position.from_json(handle.position))
        handle.close()
    assert boards["first"] != boards["last"]

    def sampled(sets: list[TeamSet]) -> list[SampledSet]:
        return [
            SampledSet(
                species=to_id(t.species), ability=to_id(t.ability),
                item=to_id(t.item) if t.item else None, nature=t.nature,
                moves=[to_id(m) for m in t.moves], sp=dict(t.sp),
            )
            for t in sets
        ]

    # The leads before any ability ran.
    fresh = position_from_sets(reg, sampled(ours), sampled(INTIMIDATE_AND_DROUGHT))
    for side in fresh.sides:
        for mon in side.pokemon:
            mon.boosts = {}
    fresh.field.weather = None
    fresh.field.weather_duration = None
    fresh.sides[0].pokemon[0].ability = "trace"

    phase = node.apply_lead_abilities(fresh)
    assert phase is not None, "the port refused the leads"
    assert _board(phase.position) == boards["first"]
    assert "trace target (the first; not branched)" in phase.unmodelled
    seen = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        drawn = node.apply_lead_abilities(fresh, rng=rng)
        assert drawn is not None and drawn.unmodelled == ()
        seen.add(_board(drawn.position))
    assert seen == {boards["first"], boards["last"]}


@pytest.mark.oracle
def test_the_port_runs_the_leads_fastest_first(reg, oracle: Oracle, node) -> None:  # noqa: ANN001
    """A slow Drought lead against a fast Drizzle one: the faster sets rain and the slower
    replaces it with sun, so the order the port runs the leads in is the whole answer."""
    ours = [TORK, WHIM, KING, CHOMP]
    theirs = [
        _mon("Pelipper", "Drizzle", ["scald", "protect", "hurricane", "tailwind"], FAST),
        INCIN, KING, CHOMP,
    ]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    showdown = _unstat(Position.from_json(handle.position))
    handle.close()
    assert showdown.field.weather == "sunnyday", showdown.field.weather
    fresh = showdown.copy()
    fresh.field.weather = None
    fresh.field.weather_duration = None
    for side in fresh.sides:
        for mon in side.pokemon:
            mon.boosts = {}
    phase = node.apply_lead_abilities(fresh)
    assert phase is not None, "the port refused the leads"
    assert _board(phase.position) == _board(showdown)


# ---------------------------------------------------------------------------
# `paused_in`: the pause resumed in another completion of the hidden bench
# ---------------------------------------------------------------------------


def test_a_pause_resumed_in_another_world_uses_that_world(reg, node) -> None:  # noqa: ANN001
    """IKA-120's shape: the opponent's unseen back two rebuilt, and every option resumed there."""
    from pokeuraou.priors import SampledSet
    from pokeuraou.regulation import to_id
    from pokeuraou.selfplay import position_from_sets

    def sampled(sets: list[TeamSet]) -> list[SampledSet]:
        return [
            SampledSet(
                species=to_id(t.species), ability=to_id(t.ability),
                item=to_id(t.item) if t.item else None, nature=t.nature,
                moves=[to_id(m) for m in t.moves], sp=dict(t.sp),
            )
            for t in sets
        ]

    start = position_from_sets(reg, sampled([RILLA, WHIM, KING, CHOMP]), sampled(INTIMIDATE_AND_DROUGHT))
    actions = _actions(reg, start, U_TURN)
    turn = node.turn(start, actions, BUDGET, full=True)
    assert turn is not None and turn.pauses
    # Another world: the opponent's bench Kingambit is at half HP and holds Leftovers.
    world = turn.pauses[0].position.copy()
    bench = world.sides[1].pokemon[2]
    assert not bench.is_active
    bench.hp = bench.maxhp // 2
    bench.item = "leftovers"
    answered = node.resume_alternatives(turn.pauses[0], world=(world, 1))
    assert answered is not None
    chooser, port = answered
    assert chooser == 0 and port
    # The control: in the pause's own world nobody holds Leftovers there.
    plain = node.resume_alternatives(turn.pauses[0])
    assert plain is not None and plain[1]
    assert not any(
        b.position.sides[1].pokemon[2].item == "leftovers" for _o, r in plain[1] for b in r.outcomes
    )
    for _theirs, port_resumed in port:
        assert port_resumed.outcomes
        # The world was used: its bench is in every leaf.
        assert all(b.position.sides[1].pokemon[2].item == "leftovers" for b in port_resumed.outcomes)
