"""Trace copies a foe's ability as it comes in, Synchronize passes a status back, and Aura
Guard halves contact damage (IKA-203).

Showdown a5df827, data/abilities.ts (the champions mod overrides none of the three):

* Trace, `onStart` then `onUpdate`: `sample` over the foes standing whose ability has no
  `notrace` flag, then `setAbility(ability, target)`, which ends in
  `singleEvent('Start', ability)` -- so a traced Intimidate lowers the foes and a traced
  Drought puts the sun up, at Trace's own place in the switch-in order. It runs on any
  switch-in (the leads, a switch, a replacement) and on a mega evolution into Trace (Mega
  Meowstic). Two foes with different abilities are two outcomes of one half each; the
  pinned policy answers `sample` with the first foe and `sample: "last"` with the second.
* Synchronize, `onAfterSetStatus`: a status from another Pokemon, not Toxic Spikes, not
  sleep or freeze, is `trySetStatus`'d on the source -- a Thunder Wave, a Toxic (an
  ally's too), Scald's burn. It fires before a Lum Berry cures the holder.
* Aura Guard (Mega Lucario Z), `onSourceModifyDamage`: contact halves the damage.

Neither engine did any of it: both are in `inert.rs` as abilities Python never names, and
Aura Guard was refused by the port and noted by the calculator. Light Clay was listed as
inert as well, but only because Python reads it from the dump (`durations.byItem`) rather
than by name; both engines already give a screen 8 turns (`test_resolve`'s
`test_light_clay_extends_a_screen_and_the_dump_says_so`, and the port case below).

Each case is played by Showdown first. The controls: a foe's Stance Change (`notrace`)
and two foes with the same ability, which draw nothing; Thunder Wave from an Electric type and
Sleep Powder, which Synchronize does not pass; a non-contact hit into Aura Guard.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.actions import SideAction, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import Position
from pokeuraou.resolve import (
    Budget,
    apply_lead_abilities,
    batched_payoffs,
    resolve_turn,
    resume_alternatives,
    resume_turn,
)

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}


def _mon(
    species: str, ability: str, moves: list[str], sp: dict[str, int], item: str | None = None
) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Adamant", moves=moves, item=item, sp=dict(sp))


GARDE = _mon("Gardevoir", "Trace", ["protect", "thunderwave", "hypervoice", "calmmind"], FAST)
GARDE_SYNC = _mon("Gardevoir", "Synchronize", ["protect", "calmmind", "hypervoice", "psychic"], SLOW)
GARDE_LUM = _mon(
    "Gardevoir", "Synchronize", ["protect", "calmmind", "hypervoice", "psychic"], SLOW, "Lum Berry"
)
MEOW = _mon("Meowstic", "Prankster", ["protect", "fakeout", "psychic", "reflect"], FAST, "Meowsticite")
INCIN = _mon("Incineroar", "Intimidate", ["protect", "fakeout", "willowisp", "knockoff"], SLOW)
INCIN_B = _mon("Incineroar", "Intimidate", ["protect", "fakeout", "willowisp", "knockoff"], SLOW)
TORK = _mon("Torkoal", "Drought", ["protect", "willowisp", "eruption", "toxic"], SLOW)
KING = _mon("Kingambit", "Defiant", ["protect", "swordsdance", "ironhead"], SLOW)
CHOMP = _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake"], SLOW)
GYARA = _mon("Gyarados", "Intimidate", ["protect", "waterfall", "dragondance"], SLOW)
WHIM = _mon("Whimsicott", "Prankster", ["protect", "thunderwave", "moonblast", "tailwind"], FAST)
RAICHU = _mon("Raichu", "Static", ["protect", "thunderwave", "nuzzle", "fakeout"], FAST)
PELI = _mon("Pelipper", "Drizzle", ["scald", "protect", "hurricane", "tailwind"], FAST)
VENU = _mon("Venusaur", "Chlorophyll", ["sleeppowder", "protect", "gigadrain", "sludgebomb"], FAST)
RILLA = _mon("Rillaboom", "Overgrow", ["uturn", "protect", "woodhammer", "fakeout"], FAST)
LUCARIO = _mon(
    "Lucario", "Inner Focus", ["protect", "swordsdance", "closecombat", "meteormash"], SLOW,
    "Lucarionite Z",
)
AEGIS = _mon("Aegislash", "Stance Change", ["kingsshield", "shadowball", "protect"], SLOW)

INTIMIDATE_AND_DROUGHT = [INCIN, TORK, KING, CHOMP]
SYNC_SIDE = [GARDE_SYNC, TORK, KING, CHOMP]

#: name -> (our team, their team, turns played first, the turn compared, oracle policy,
#: whether Python/the port must draw (two outcomes under a branching budget)).
CASES: dict[str, tuple] = {
    # Trace. The pinned policy's first foe is Incineroar (Intimidate); `last` is Torkoal.
    "trace-switch-in-first": (
        [WHIM, KING, GARDE, CHOMP], INTIMIDATE_AND_DROUGHT, [],
        ["switch 3, move 2", "move 1, move 1"], RandomnessPolicy(), True,
    ),
    "trace-switch-in-last": (
        [WHIM, KING, GARDE, CHOMP], INTIMIDATE_AND_DROUGHT, [],
        ["switch 3, move 2", "move 1, move 1"], RandomnessPolicy(sample="last"), True,
    ),
    "trace-mega-first": (
        [MEOW, WHIM, KING, CHOMP], INTIMIDATE_AND_DROUGHT, [],
        ["move 1 mega, move 1", "move 1, move 1"], RandomnessPolicy(), True,
    ),
    "trace-mega-last": (
        [MEOW, WHIM, KING, CHOMP], INTIMIDATE_AND_DROUGHT, [],
        ["move 1 mega, move 1", "move 1, move 1"], RandomnessPolicy(sample="last"), True,
    ),
    # Controls: Stance Change is `notrace` (the first foe, passed over), and two
    # Intimidates are one outcome.
    "control-trace-passes-over-stance-change": (
        [WHIM, KING, GARDE, CHOMP], [AEGIS, INCIN, KING, CHOMP], [],
        ["switch 3, move 2", "move 1, move 1"], RandomnessPolicy(), False,
    ),
    "control-trace-two-foes-with-one-ability": (
        [WHIM, KING, GARDE, CHOMP], [INCIN_B, GYARA, KING, CHOMP], [],
        ["switch 3, move 2", "move 1, move 1"], RandomnessPolicy(sample="last"), False,
    ),
    # Synchronize.
    "sync-thunder-wave": (
        [WHIM, KING, CHOMP, INCIN], SYNC_SIDE, [], ["move 2 1, move 1", "move 2, move 1"],
        RandomnessPolicy(), False,
    ),
    "sync-toxic": (
        [TORK, KING, CHOMP, INCIN], SYNC_SIDE, [], ["move 4 1, move 1", "move 2, move 1"],
        RandomnessPolicy(), False,
    ),
    "sync-lum-berry": (
        [WHIM, KING, CHOMP, INCIN], [GARDE_LUM, TORK, KING, CHOMP], [],
        ["move 2 1, move 1", "move 2, move 1"], RandomnessPolicy(), False,
    ),
    "control-sync-electric-source": (
        [RAICHU, KING, CHOMP, INCIN], SYNC_SIDE, [], ["move 2 1, move 1", "move 2, move 1"],
        RandomnessPolicy(), False,
    ),
    "control-sync-sleep": (
        [VENU, KING, CHOMP, INCIN], SYNC_SIDE, [], ["move 1 1, move 2", "move 2, move 1"],
        RandomnessPolicy(), False,
    ),
    # Aura Guard: Iron Head is contact, Earthquake is not.
    "aura-guard-contact": (
        [KING, CHOMP, WHIM, INCIN], [LUCARIO, TORK, GARDE, WHIM], [],
        ["move 3 1, move 1", "move 2 mega, move 1"], RandomnessPolicy(), False,
    ),
    "control-aura-guard-no-contact": (
        [CHOMP, KING, WHIM, INCIN], [LUCARIO, TORK, GARDE, WHIM], [],
        ["move 3, move 1", "move 2 mega, move 1"], RandomnessPolicy(), False,
    ),
}

#: No crit, no miss, no status check, no chance-based secondary, the maximum roll -- what
#: the oracle's policy pins.
BUDGET = replace(
    Budget.exact(),
    enumerate_crit=False,
    enumerate_secondary=False,
    enumerate_accuracy=False,
    enumerate_status_checks=False,
).with_fixed_roll(0)
#: The same with the draws branched: what a matrix does.
BRANCHING = replace(BUDGET, enumerate_secondary=True)


def _play(oracle: Oracle, name: str):  # noqa: ANN202
    ours, theirs, setup, choices, policy, _draws = CASES[name]
    handle = oracle.create(FORMAT_ID, ours, theirs, policy=policy)
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


def _board(pos: Position) -> tuple:
    """What the cases are about: each active's species, ability, status, boosts and HP, and
    the weather."""
    out: list[tuple] = []
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
    """The port declines a stats override (it reads one as a Transform); both engines compute
    the same stats from the spreads."""
    for side in pos.sides:
        for party in side.pokemon:
            party.stats_override = None
    return pos


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_does_what_the_case_is_named_for(oracle: Oracle, name: str) -> None:
    before, after, log = _play(oracle, name)
    traced = [line for line in log if "[from] ability: Trace" in line]
    synced = [line for line in log if "ability: Synchronize" in line]
    if name.startswith("trace-"):
        copied = "Drought" if name.endswith("last") else "Intimidate"
        assert traced and f"|{copied}|Trace|" in traced[0], log[-15:]
        if copied == "Drought":
            assert after.field.weather == "sunnyday"
    elif name.startswith("control-trace-"):
        assert traced and "|Intimidate|Trace|" in traced[0], log[-15:]
    elif name.startswith("sync-"):
        assert synced, log[-15:]
        source = after.sides[0].pokemon[after.sides[0].active[0]]
        assert source.status is not None, log[-15:]
    elif name == "control-sync-electric-source":
        assert synced and after.sides[0].pokemon[after.sides[0].active[0]].status is None
    elif name == "control-sync-sleep":
        assert not synced and after.sides[0].pokemon[after.sides[0].active[0]].status is None
    del before


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_does_what_showdown_does(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, after, _log = _play(oracle, name)
    choices = CASES[name][3]
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(result.branches) == 1 and not result.suspended
    branch = result.branches[0]
    if name.endswith("-last"):
        # The collapsed budget takes the first foe; the branching one has the second.
        result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BRANCHING)
        boards = [_board(b.position) for b in result.branches]
        assert _board(after) in boards, (boards, _board(after))
        return
    assert _board(branch.position) == _board(after), " / ".join(branch.events)


@pytest.mark.parametrize("name", sorted(n for n in CASES if CASES[n][5]))
def test_a_trace_between_two_abilities_is_two_halves(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """Under a branching budget both foes' abilities are outcomes of one half each; the
    collapsed one takes the first and says so."""
    before, after, _log = _play(oracle, name)
    choices = CASES[name][3]
    actions = _actions(reg, before, choices)
    result = resolve_turn(reg, before, actions, budget=BRANCHING)
    abilities = sorted(
        (b.probability, b.position.sides[0].pokemon[b.position.sides[0].active[0]].ability)
        for b in result.branches
    )
    assert [a for _p, a in abilities] == ["drought", "intimidate"], abilities
    assert [p for p, _a in abilities] == pytest.approx([0.5, 0.5])
    assert not any("trace target" in u for u in result.unmodelled)
    collapsed = resolve_turn(reg, before, actions, budget=BUDGET)
    assert "trace target (the first; not branched)" in collapsed.unmodelled
    del after


def test_the_leads_trace_the_first_foe_or_a_drawn_one(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """`apply_lead_abilities` hands back one position: the first foe with a note, or a draw
    from the generator it is given, which lands on one of Showdown's two answers."""
    from pokeuraou.priors import SampledSet
    from pokeuraou.regulation import to_id

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

    mine = position_from_sets(reg, sampled(ours), sampled(INTIMIDATE_AND_DROUGHT))
    assert _board(mine) == boards["first"]
    # The phase itself, on the leads before any ability ran: the first foe, noted.
    fresh = mine.copy()
    for side in fresh.sides:
        for mon in side.pokemon:
            mon.boosts = {}
    fresh.field.weather = None
    fresh.field.weather_duration = None
    fresh.sides[0].pokemon[0].ability = "trace"
    phase = apply_lead_abilities(reg, fresh)
    assert _board(phase.position) == boards["first"]
    assert "trace target (the first; not branched)" in phase.unmodelled
    assert apply_lead_abilities(reg, fresh, rng=np.random.default_rng(0)).unmodelled == ()
    seen = set()
    for seed in range(12):
        drawn = position_from_sets(
            reg, sampled(ours), sampled(INTIMIDATE_AND_DROUGHT), rng=np.random.default_rng(seed)
        )
        seen.add(_board(drawn))
    assert seen == {boards["first"], boards["last"]}


def test_a_replacement_after_u_turn_traces_in_each_outcome(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Rillaboom U-turns out and Gardevoir comes in mid-turn: `resume_turn` runs the switch-in
    once per foe, and each outcome is Showdown's for that `sample`."""
    ours = [RILLA, WHIM, GARDE, CHOMP]
    choices = ["move 1 2, move 1", "move 1, move 2 2"]
    boards = {}
    for sample in ("first", "last"):
        handle = oracle.create(
            FORMAT_ID, ours, INTIMIDATE_AND_DROUGHT, policy=RandomnessPolicy(sample=sample)
        )
        handle.step(["team 1234", "team 1234"])
        before = Position.from_json(handle.position)
        handle.step(choices)
        assert handle.choice_errors == [], handle.choice_errors
        assert (handle.requests[0] or {}).get("forceSwitch") == [True, False]
        handle.step(["switch 3, pass", None])
        assert handle.choice_errors == [], handle.choice_errors
        boards[sample] = _board(Position.from_json(handle.position))
        handle.close()
    assert boards["first"] != boards["last"]

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BRANCHING)
    assert result.suspended and not result.branches
    for pause in result.suspended:
        option = next(
            o for o in switch_actions_after_faint(reg, pause.position, 0, [True, False])
            if o.to_choice() == "switch 3, pass"
        )
        from pokeuraou.actions import PassAction

        passes = SideAction(slots=tuple(PassAction(slot=i) for i in range(2)))
        resumed = resume_turn(reg, pause, [option, passes])
        got = sorted((b.probability, _board(b.position)) for b in resumed.branches)
        assert [p for p, _b in got] == pytest.approx([pause.probability / 2] * 2)
        assert {b for _p, b in got} == {boards["first"], boards["last"]}
        # The node's own walk through the pause agrees.
        chooser, alternatives = resume_alternatives(reg, pause)
        assert chooser == 0 and alternatives


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("budget_name", ["collapsed", "branching"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_resolves_the_case_as_python_does(
    reg, oracle: Oracle, bridged: None, name: str, budget_name: str  # noqa: ANN001
) -> None:
    before, _after, _log = _play(oracle, name)
    before = _unstat(before)
    budget = BUDGET if budget_name == "collapsed" else BRANCHING
    actions = _actions(reg, before, CASES[name][3])
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    here = resolve_turn(reg, before, actions, budget=budget)
    assert there.branches == pytest.approx([b.probability for b in here.branches], abs=1e-12)
    assert there.suspended == pytest.approx([s.probability for s in here.suspended], abs=1e-12)
    # The notes this change adds. The calculator's item notes differ between the two
    # already (Python names an unused mega stone the port knows from the regulation).
    assert {u for u in there.unmodelled if "trace" in u} == {
        u for u in here.unmodelled if "trace" in u
    }
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(before, actions, budget, select=index)
        assert chosen is not None and chosen.position is not None
        assert _board(chosen.position) == _board(branch.position)


def test_the_port_folds_a_traced_replacement_as_python_does(
    reg, oracle: Oracle, bridged: None  # noqa: ANN001
) -> None:
    """The node's matrix through the port and without it, where a U-turn brings Gardevoir
    in to trace mid-turn (the draw inside `resume_turn`), where Gardevoir switches in, and
    the moves beside them."""
    ours = [RILLA, WHIM, GARDE, CHOMP]
    handle = oracle.create(FORMAT_ID, ours, INTIMIDATE_AND_DROUGHT, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = _unstat(Position.from_json(handle.position))
    handle.close()
    wanted = {
        0: ["move 1 2, move 1", "move 1 1, move 2 1", "switch 3, move 1", "move 3 1, move 1"],
        1: ["move 3 2, move 1", "move 1, move 1", "move 4 1, move 3"],
    }
    menus = {
        side: [a for a in side_actions(reg, before, side) if a.to_choice() in wanted[side]]
        for side in (0, 1)
    }
    assert [len(menus[0]), len(menus[1])] == [4, 3]
    budget = Budget.matrix()
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]
    os.environ.pop(rustnode.ENV_ENABLE, None)
    rustnode.reset()
    python, notes, _exact = batched_payoffs(reg, before, menus[0], menus[1], evaluators, budget=budget)
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    port, port_notes, _port_exact = batched_payoffs(
        reg, before, menus[0], menus[1], evaluators, budget=budget
    )
    for mine, theirs in zip(python, port, strict=True):
        assert abs(mine - theirs).max() < 1e-12, (mine, theirs)
    assert not any("trace target" in n for n in notes)


def test_the_port_keeps_light_clay_screens_eight_turns(reg, bridged: None) -> None:  # noqa: ANN001
    """Light Clay is `inert` to the name scan but read from the dump by both engines."""
    from pokeuraou.priors import SampledSet
    from pokeuraou.selfplay import position_from_sets

    grimm = SampledSet(species="grimmsnarl", ability="prankster", item="lightclay", nature="Careful",
                       moves=["lightscreen", "reflect", "protect", "spiritbreak"], sp={"hp": 32})
    rest = [
        SampledSet(species=t.species.lower(), ability=t.ability.lower().replace(" ", ""), item=None,
                   nature=t.nature, moves=list(t.moves), sp=dict(t.sp))
        for t in (KING, CHOMP, WHIM)
    ]
    pos = _unstat(position_from_sets(reg, [grimm, *rest], [*rest[::-1], grimm]))
    actions = [
        next(a for a in side_actions(reg, pos, 0) if a.to_choice() == "move 1, move 1"),
        next(a for a in side_actions(reg, pos, 1) if a.to_choice() == "move 1, move 1"),
    ]
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(pos, actions, BUDGET, select=0)
    here = resolve_turn(reg, pos, actions, budget=BUDGET).branches[0].position
    assert there is not None and there.position is not None
    screen = there.position.sides[0].side_condition("lightscreen")
    assert screen is not None and screen.duration == here.sides[0].side_condition("lightscreen").duration
    assert screen.duration == 7, screen  # 8, one counted down at the end of this turn
