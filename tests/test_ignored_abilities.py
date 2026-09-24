"""The four Ruin abilities, Sniper, Supreme Overlord and Snow Cloak (IKA-222).

Showdown a5df827, data/abilities.ts (the champions mod overrides none of them)::

    tabletsofruin: {
        onAnyModifyAtk(atk, source, target, move) {
            const abilityHolder = this.effectState.target;
            if (source.hasAbility('Tablets of Ruin')) return;
            if (!move.ruinedAtk) move.ruinedAtk = abilityHolder;
            if (move.ruinedAtk !== abilityHolder) return;
            return this.chainModify(0.75);
        },
    },
    vesselofruin:  onAnyModifySpA, the same with SpA
    swordofruin:   onAnyModifyDef(def, target, source, move), `if (target.hasAbility(...)) return`
    beadsofruin:   onAnyModifySpD, the same with SpD
    sniper: {
        onModifyDamage(damage, source, target, move) {
            if (target.getMoveHitData(move).crit) return this.chainModify(1.5);
        },
    },
    supremeoverlord: {
        onStart(pokemon) {
            if (pokemon.side.totalFainted) {
                ...; this.effectState.fallen = Math.min(pokemon.side.totalFainted, 5);
            }
        },
        onBasePowerPriority: 21,
        onBasePower(basePower, attacker, defender, move) {
            if (this.effectState.fallen) {
                const powMod = [4096, 4506, 4915, 5325, 5734, 6144];
                return this.chainModify([powMod[this.effectState.fallen], 4096]);
            }
        },
    },
    snowcloak: {
        onModifyAccuracyPriority: -1,
        onModifyAccuracy(accuracy) {
            if (typeof accuracy !== 'number') return;
            if (this.field.isWeather(['hail', 'snowscape'])) return this.chainModify([3277, 4096]);
        },
        flags: { breakable: 1 },
    },

A Ruin ability is an `onAny` handler: every active Pokemon's ability runs on every stat
event, so the holder lowers its partner's and its foes' stat alike and only a Pokemon with
the same ability is spared. Which events run is `getDamage`'s (sim/battle-actions.ts):

    attackStat = (category === 'Physical' ? 'atk' : 'spa');
    attack = this.battle.runEvent('Modify' + statTable[attackStat], source, target, move, attack);
    defense = this.battle.runEvent('Modify' + statTable[defenseStat], target, source, move, defense);

The attacking event is the category's and always the user's, whatever stat was read: Body
Press's Def and Foul Play's borrowed Atk both meet Tablets, and only Tablets on the user
spares them. The defending event is the stat read: Psyshock meets Sword, not Beads.
Supreme Overlord counts its side's faints when
it comes in and keeps that count while it stays in; the count lives in `abilityState`,
which the bridge exports.

The port named none of the seven: all were in `inert.rs`, so its gates took them and did
nothing, and the calculator noted each one on every hit. Snow Cloak was in the gates'
"no effect a turn can observe" group, so it was not even noted.

Every case is played by Showdown first, and asserts what Showdown did (the effect fired or
did not) before the port is held to it. The controls: the same Pokemon with another
ability, the holder's own stat, the stat the event does not read, Mold Breaker into Snow
Cloak, and no snow.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget, apply_lead_abilities, resolve_replacements
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 10, "def": 10, "spa": 10, "spd": 4, "spe": 0}
#: The ids this file is about: the port's notes must not name them once it models them.
IDS = (
    "tabletsofruin", "vesselofruin", "swordofruin", "beadsofruin", "sniper",
    "supremeoverlord", "snowcloak",
)


def _mon(species: str, ability: str, moves: list[str], sp: dict | None = None,
         item: str | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp=dict(sp or SLOW), item=item)


def _p(pos: Position, side: int, slot: int) -> Pokemon:
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _chosen(reg, pos: Position, step: tuple[str, str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
    return pos


def _notes(reply: dict) -> list[str]:
    return [n for n in reply.get("unmodelled") or [] if any(i in n for i in IDS)]


# ---------------------------------------------------------------------------
# The Ruin abilities: one hit into p2a, compared with the same hit and no Ruin anywhere.

ATTACKER = ["closecombat", "aurasphere", "psyshock", "bodypress"]
CC, SPHERE, PSYSHOCK, BODYPRESS = ("move 1 1", "move 2 1", "move 3 1", "move 4 1")
CHOMP = _mon("Garchomp", "Rough Skin", ATTACKER, FAST)
# Foul Play reads the target's Atk; the target carries an Attack worth reading.
FOUL = _mon("Garchomp", "Rough Skin", ["foulplay", "protect", "closecombat", "aurasphere"], FAST)
TARGET_MOVES = ["calmmind", "protect", "scald", "icebeam"]
TARGET_SP = {"hp": 32, "atk": 32, "def": 1, "spa": 0, "spd": 1, "spe": 0}


def _target(ability: str = "Marvel Scale") -> TeamSet:
    return _mon("Milotic", ability, TARGET_MOVES, TARGET_SP)


def _guard(ability: str = "Blaze") -> TeamSet:
    """The partner on either side, which Protects; a holder when given a Ruin ability.

    Not Intimidate, which would lower the attacker in the plain case and not in the Ruin
    one: the difference would be the Intimidate's."""
    return _mon("Incineroar", ability, ["protect", "fakeout", "flareblitz", "partingshot"])


P1_BENCH = [
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]
P2_BENCH = [
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]


@dataclasses.dataclass(frozen=True)
class RuinCase:
    p1: tuple[TeamSet, TeamSet]
    p2: tuple[TeamSet, TeamSet]
    #: p1a's choice; p1b and p2b Protect, p2a Calm Minds after the hit.
    attack: str
    #: The same hit with no Ruin on the field, and how this one compares with it.
    plain: str
    want: str  # "<", ">", "="


def _ruin_cases() -> dict[str, RuinCase]:
    chomp, foul, target, guard = CHOMP, FOUL, _target(), _guard()
    return {
        # The plain hits, each its own baseline.
        "plain, physical": RuinCase((chomp, guard), (target, guard), CC, "plain, physical", "="),
        "plain, special": RuinCase((chomp, guard), (target, guard), SPHERE, "plain, special", "="),
        "plain, psyshock": RuinCase((chomp, guard), (target, guard), PSYSHOCK, "plain, psyshock", "="),
        "plain, body press": RuinCase((chomp, guard), (target, guard), BODYPRESS, "plain, body press", "="),
        "plain, foul play": RuinCase((foul, guard), (target, guard), CC, "plain, foul play", "="),
        # Tablets lowers every other Pokemon's Atk: a foe's, the holder's partner's.
        "tablets on the foe side, physical": RuinCase(
            (chomp, guard), (target, _guard("Tablets of Ruin")), CC, "plain, physical", "<"),
        "tablets beside the attacker, physical": RuinCase(
            (chomp, _guard("Tablets of Ruin")), (target, guard), CC, "plain, physical", "<"),
        "tablets on the attacker, physical": RuinCase(
            (replace(chomp, ability="Tablets of Ruin"), guard), (target, guard), CC,
            "plain, physical", "="),
        "tablets, special": RuinCase(
            (chomp, guard), (target, _guard("Tablets of Ruin")), SPHERE, "plain, special", "="),
        # Foul Play reads the target's Atk, but the ModifyAtk event is the user's
        # (`runEvent('ModifyAtk', source, ...)`): Tablets on the target lowers it too, and
        # only Tablets on the user spares it.
        "tablets beside the foul play target": RuinCase(
            (foul, guard), (target, _guard("Tablets of Ruin")), CC, "plain, foul play", "<"),
        "tablets on the foul play target": RuinCase(
            (foul, guard), (_target("Tablets of Ruin"), guard), CC, "plain, foul play", "<"),
        "tablets on the foul play user": RuinCase(
            (replace(foul, ability="Tablets of Ruin"), guard), (_target("Tablets of Ruin"), guard),
            CC, "plain, foul play", "="),
        "vessel on the foe side, special": RuinCase(
            (chomp, guard), (target, _guard("Vessel of Ruin")), SPHERE, "plain, special", "<"),
        "vessel on the attacker, special": RuinCase(
            (replace(chomp, ability="Vessel of Ruin"), guard), (target, guard), SPHERE,
            "plain, special", "="),
        "vessel, physical": RuinCase(
            (chomp, guard), (target, _guard("Vessel of Ruin")), CC, "plain, physical", "="),
        # Sword lowers the target's Def, and the attacker's own for Body Press.
        "sword on the attacker, physical": RuinCase(
            (replace(chomp, ability="Sword of Ruin"), guard), (target, guard), CC,
            "plain, physical", ">"),
        "sword beside the target, physical": RuinCase(
            (chomp, guard), (target, _guard("Sword of Ruin")), CC, "plain, physical", ">"),
        "sword on the target, physical": RuinCase(
            (chomp, _guard("Sword of Ruin")), (_target("Sword of Ruin"), guard), CC,
            "plain, physical", "="),
        "sword, psyshock": RuinCase(
            (replace(chomp, ability="Sword of Ruin"), guard), (target, guard), PSYSHOCK,
            "plain, psyshock", ">"),
        "sword, special": RuinCase(
            (replace(chomp, ability="Sword of Ruin"), guard), (target, guard), SPHERE,
            "plain, special", "="),
        # Body Press reads the user's Def, but the event is ModifyAtk (`attackStat` is reset
        # to the category's before the events run): Sword lowers only the target's Def, and
        # Tablets lowers the pressed Def.
        "sword on the foe side, body press": RuinCase(
            (chomp, guard), (target, _guard("Sword of Ruin")), BODYPRESS, "plain, body press", ">"),
        "tablets on the foe side, body press": RuinCase(
            (chomp, guard), (target, _guard("Tablets of Ruin")), BODYPRESS, "plain, body press", "<"),
        "beads on the attacker, special": RuinCase(
            (replace(chomp, ability="Beads of Ruin"), guard), (target, guard), SPHERE,
            "plain, special", ">"),
        "beads on the target, special": RuinCase(
            (chomp, _guard("Beads of Ruin")), (_target("Beads of Ruin"), guard), SPHERE,
            "plain, special", "="),
        "beads, psyshock": RuinCase(
            (replace(chomp, ability="Beads of Ruin"), guard), (target, guard), PSYSHOCK,
            "plain, psyshock", "="),
    }


RUIN = _ruin_cases()
RUIN_STEP_P2 = "move 1, move 1"  # Calm Mind, Protect


def _play_ruin(oracle: Oracle, case: RuinCase) -> tuple[dict, dict]:
    handle = oracle.create(FORMAT_ID, [*case.p1, *P1_BENCH], [*case.p2, *P2_BENCH],
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    handle.step([f"{case.attack}, move 1", RUIN_STEP_P2])
    assert handle.choice_errors == [], handle.choice_errors
    after = handle.position
    handle.close()
    return before, after


def _dealt(before: dict, after: dict) -> int:
    return _p(Position.from_json(before), 1, 0).hp - _p(Position.from_json(after), 1, 0).hp


@pytest.mark.parametrize("name", sorted(RUIN))
def test_showdown_ruin(oracle: Oracle, name: str) -> None:
    case = RUIN[name]
    dealt = _dealt(*_play_ruin(oracle, case))
    plain = _dealt(*_play_ruin(oracle, RUIN[case.plain]))
    assert dealt > 0
    assert {"<": dealt < plain, ">": dealt > plain, "=": dealt == plain}[case.want], (dealt, plain)


@pytest.mark.parametrize("name", sorted(RUIN))
def test_the_port_matches_showdown_ruin(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn, port_weights

    case = RUIN[name]
    before, after = _play_ruin(oracle, case)
    start = _loaded(before)
    chosen = _chosen(reg, start, (f"{case.attack}, move 1", RUIN_STEP_P2))
    pinned = port_turn(port, start, chosen)
    assert _p(pinned, 1, 0).hp == _p(Position.from_json(after), 1, 0).hp, name
    assert _notes(port_weights(port, start, chosen, Budget.deterministic(0))) == []


# A spread move's damage is computed for every target before any is dealt, and a Pokemon it
# knocks out stays on the field until the move is over: a Tablets holder the Earthquake
# knocks out still lowers it for the other target. The port hits targets one by one.

QUAKER = _mon("Garchomp", "Rough Skin", ["earthquake", "protect", "closecombat", "aurasphere"], FAST)
#: Fire/Rock, 4x weak to Ground: the Earthquake knocks it out. (Not a low level: the port
#: computes stats at the format's level 50.)
FRAIL = _mon("Arcanine-Hisui", "Tablets of Ruin", ["calmmind", "protect", "flareblitz", "rockslide"],
             {"hp": 0, "def": 0, "spd": 0})


def _spread_cases() -> dict[str, tuple[tuple[TeamSet, TeamSet], int]]:
    """p2 and the slot (0 or 1) of the Pokemon whose damage is read."""
    target = _target()
    return {
        "frail tablets holder at p2a, target at p2b": ((FRAIL, target), 1),
        "frail tablets holder at p2b, target at p2a": ((target, FRAIL), 0),
        "frail plain at p2a, target at p2b": ((replace(FRAIL, ability="Blaze"), target), 1),
    }


SPREAD = _spread_cases()
SPREAD_STEP = ("move 1, move 1", "move 1, move 1")  # Earthquake, Protect / Calm Mind x2


def _play_spread(oracle: Oracle, p2: tuple[TeamSet, TeamSet]) -> tuple[dict, dict]:
    handle = oracle.create(FORMAT_ID, [QUAKER, _guard(), *P1_BENCH], [*p2, *P2_BENCH],
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    handle.step(list(SPREAD_STEP))
    assert handle.choice_errors == [], handle.choice_errors
    after = handle.position
    handle.close()
    return before, after


def test_showdown_ruin_outlasts_its_knockout(oracle: Oracle) -> None:
    dealt = {}
    for name, (p2, slot) in SPREAD.items():
        before, after = _play_spread(oracle, p2)
        b, a = Position.from_json(before), Position.from_json(after)
        assert _p(a, 1, 1 - slot).fainted, name
        dealt[name] = _p(b, 1, slot).hp - _p(a, 1, slot).hp
    plain = dealt["frail plain at p2a, target at p2b"]
    assert dealt["frail tablets holder at p2a, target at p2b"] < plain, dealt
    assert dealt["frail tablets holder at p2b, target at p2a"] < plain, dealt


@pytest.mark.parametrize("name", sorted(SPREAD))
def test_the_port_matches_showdown_ruin_spread(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn

    p2, slot = SPREAD[name]
    before, after = _play_spread(oracle, p2)
    start = _loaded(before)
    pinned = port_turn(port, start, _chosen(reg, start, SPREAD_STEP))
    assert _p(pinned, 1, slot).hp == _p(Position.from_json(after), 1, slot).hp, name


# ---------------------------------------------------------------------------
# Sniper: a crit into p2a, and the same hit uncritted.

SNIPER = _mon("Inteleon", "Sniper", ["scald", "protect", "icebeam", "uturn"], FAST)
PLAIN_SNIPER = replace(SNIPER, ability="Torrent")
SNIPER_TARGET = _mon("Snorlax", "Thick Fat", ["curse", "protect", "bodyslam", "rest"],
                     {"hp": 32, "def": 32, "spd": 32})


def _play_crit(oracle: Oracle, p1a: TeamSet, crit: bool) -> tuple[dict, dict, list[dict]]:
    handle = oracle.create(FORMAT_ID, [p1a, _guard(), *P1_BENCH],
                           [SNIPER_TARGET, _guard(), *P2_BENCH],
                           policy=RandomnessPolicy(damage_roll=0, crit=crit))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    handle.step(["move 1 1, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    after, rolls = handle.position, handle.rolls
    handle.close()
    return before, after, rolls


CRIT_CASES = {
    "sniper, crit": (SNIPER, True),
    "sniper, no crit": (SNIPER, False),
    "torrent, crit": (PLAIN_SNIPER, True),
}


def test_showdown_sniper(oracle: Oracle) -> None:
    dealt = {name: _dealt(*_play_crit(oracle, mon, crit)[:2]) for name, (mon, crit) in CRIT_CASES.items()}
    # 1.5x on the crit's own 1.5x, and nothing without a crit.
    assert dealt["sniper, crit"] > dealt["torrent, crit"] > dealt["sniper, no crit"] > 0, dealt


@pytest.mark.parametrize("name", sorted(CRIT_CASES))
def test_the_port_matches_showdown_sniper(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_branches, port_weights

    mon, crit = CRIT_CASES[name]
    before, after, _ = _play_crit(oracle, mon, crit)
    start = _loaded(before)
    chosen = _chosen(reg, start, ("move 1 1, move 1", "move 1, move 1"))
    budget = replace(Budget.deterministic(0), enumerate_crit=True, max_branches=4)
    hps = {_p(p, 1, 0).hp for _, p in port_branches(port, start, chosen, budget)}
    assert len(hps) == 2, hps  # a crit and a non-crit
    assert _p(Position.from_json(after), 1, 0).hp in hps, (name, hps)
    assert _notes(port_weights(port, start, chosen, budget)) == []


# ---------------------------------------------------------------------------
# Supreme Overlord: p1a faints to its own Memento, Kingambit comes in, and hits p2a.

MEMENTO = _mon("Gengar", "Cursed Body", ["memento", "protect", "shadowball", "sludgebomb"], FAST)
KINGAMBIT = _mon("Kingambit", "Supreme Overlord", ["ironhead", "protect", "swordsdance", "kowtowcleave"],
                 {"hp": 20, "atk": 32, "spe": 14})
PLAIN_KINGAMBIT = replace(KINGAMBIT, ability="Defiant")
SO_TARGET = _mon("Milotic", "Marvel Scale", TARGET_MOVES, {"hp": 32, "def": 32, "spd": 1})
SO_P2 = [SO_TARGET, _guard(), *P2_BENCH]


def _p1(kingambit: TeamSet, lead: bool) -> list[TeamSet]:
    if lead:
        return [kingambit, _guard(), *P1_BENCH]
    return [MEMENTO, _guard(), kingambit, P1_BENCH[0]]


def _play_overlord(oracle: Oracle, kingambit: TeamSet, lead: bool) -> dict[str, dict]:
    """Showdown's positions: before the replacement, after it, and around Kingambit's hit."""
    out: dict[str, dict] = {}
    handle = oracle.create(FORMAT_ID, _p1(kingambit, lead), SO_P2,
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    if not lead:
        handle.step(["move 1 1, move 1", "move 1, move 1"])  # Memento; everyone else Protects
        assert handle.choice_errors == [], handle.choice_errors
        out["replacement"] = handle.position
        handle.step(["switch 3, pass", None])
        assert handle.choice_errors == [], handle.choice_errors
    out["before"] = handle.position
    handle.step(["move 1 1, move 1", "move 1, move 1"])  # Iron Head; p2a Calm Minds
    assert handle.choice_errors == [], handle.choice_errors
    out["after"] = handle.position
    handle.close()
    return out


def _fallen(pos: Position, side: int, slot: int) -> object:
    return _p(pos, side, slot).ability_state.get("fallen")


OVERLORD = {
    "overlord, one fallen": (KINGAMBIT, False),
    "overlord, none fallen": (KINGAMBIT, True),
    "defiant, one fallen": (PLAIN_KINGAMBIT, False),
}


def test_showdown_overlord(oracle: Oracle) -> None:
    played = {name: _play_overlord(oracle, mon, lead) for name, (mon, lead) in OVERLORD.items()}
    dealt = {name: _dealt(p["before"], p["after"]) for name, p in played.items()}
    one, none, plain = (dealt[k] for k in ("overlord, one fallen", "overlord, none fallen",
                                           "defiant, one fallen"))
    assert one > none == plain > 0, dealt
    fallen = {name: _fallen(Position.from_json(p["before"]), 0, 0) for name, p in played.items()}
    assert fallen == {"overlord, one fallen": 1, "overlord, none fallen": None,
                      "defiant, one fallen": None}, fallen


@pytest.mark.parametrize("name", sorted(OVERLORD))
def test_the_port_matches_showdown_overlord_hit(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn, port_weights

    mon, lead = OVERLORD[name]
    played = _play_overlord(oracle, mon, lead)
    start = _loaded(played["before"])
    chosen = _chosen(reg, start, ("move 1 1, move 1", "move 1, move 1"))
    pinned = port_turn(port, start, chosen)
    assert _p(pinned, 1, 0).hp == _p(Position.from_json(played["after"]), 1, 0).hp, name
    assert _notes(port_weights(port, start, chosen, Budget.deterministic(0))) == []


@pytest.mark.parametrize("name", ["overlord, one fallen", "defiant, one fallen"])
def test_the_ports_replacement_counts_the_fallen(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The count is taken as it comes in: the port's own replacement writes it."""
    mon, lead = OVERLORD[name]
    played = _play_overlord(oracle, mon, lead)
    pos = _loaded(played["replacement"])
    picks = [
        next(a for a in switch_actions_after_faint(reg, pos, 0, [True, False])
             if "switch 3" in a.to_choice()),
        switch_actions_after_faint(reg, pos, 1, [False, False])[0],
    ]
    # A stale count on the bench must not survive the switch-in either. (Only the holder's:
    # nothing reads another ability's `fallen`, and the port leaves its state alone.)
    for bench in pos.sides[0].pokemon:
        if bench.ability == "supremeoverlord":
            bench.ability_state = {"fallen": 4}
    ours = resolve_replacements(reg, pos, picks).position
    assert _fallen(ours, 0, 0) == _fallen(Position.from_json(played["before"]), 0, 0)


def test_the_ports_leads_count_no_fallen(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """A lead has nobody fallen: a count planted on it goes when it comes in."""
    played = _play_overlord(oracle, KINGAMBIT, lead=True)
    pos = _loaded(played["before"])
    assert _fallen(pos, 0, 0) is None
    _p(pos, 0, 0).ability_state = {"fallen": 3}
    assert _fallen(apply_lead_abilities(reg, pos).position, 0, 0) is None


def test_the_ports_switch_counts_the_fallen(reg, oracle: Oracle, port) -> None:  # noqa: ANN001
    """The same count from a switch within a turn: Memento, a replacement, then a switch."""
    from ._port_showdown import port_turn

    handle = oracle.create(FORMAT_ID, [MEMENTO, _guard(), KINGAMBIT, P1_BENCH[0]], SO_P2,
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    handle.step(["move 1 1, move 1", "move 1, move 1"])
    handle.step(["switch 4, pass", None])
    assert handle.choice_errors == [], handle.choice_errors
    before = handle.position
    handle.step(["move 2, switch 3", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    assert _fallen(after, 0, 1) == 1
    start = _loaded(before)
    ours = port_turn(port, start, _chosen(reg, start, ("move 2, switch 3", "move 1, move 1")))
    assert _fallen(ours, 0, 1) == 1


# ---------------------------------------------------------------------------
# Snow Cloak: Glaceon in Abomasnow's snow, hit by a 100% and an 80% move.

GLACEON = _mon("Glaceon", "Snow Cloak", ["calmmind", "protect", "icebeam", "freezedry"],
               {"hp": 32, "def": 32, "spd": 32})
PLAIN_GLACEON = replace(GLACEON, ability="Ice Body")
SNOW = _mon("Abomasnow", "Snow Warning", ["protect", "blizzard", "gigadrain", "iceshard"])
NO_SNOW = replace(SNOW, ability="Soundproof")
MILOTIC = _mon("Milotic", "Marvel Scale", ["scald", "hydropump", "protect", "recover"], FAST)
BREAKER = replace(MILOTIC, ability="Mold Breaker")


@dataclasses.dataclass(frozen=True)
class CloakCase:
    attacker: TeamSet
    target: TeamSet
    weather: TeamSet
    attack: str
    #: Showdown's `randomChance(n, 100)` for the hit.
    accuracy: int


CLOAK = {
    "cloak, scald": CloakCase(MILOTIC, GLACEON, SNOW, "move 1 1", 80),
    "cloak, hydro pump": CloakCase(MILOTIC, GLACEON, SNOW, "move 2 1", 64),
    "cloak, mold breaker": CloakCase(BREAKER, GLACEON, SNOW, "move 1 1", 100),
    "cloak, no snow": CloakCase(MILOTIC, GLACEON, NO_SNOW, "move 1 1", 100),
    "ice body, scald": CloakCase(MILOTIC, PLAIN_GLACEON, SNOW, "move 1 1", 100),
}


def _play_cloak(oracle: Oracle, case: CloakCase, accuracy: str) -> tuple[dict, dict, list[dict]]:
    handle = oracle.create(FORMAT_ID, [case.attacker, _guard(), *P1_BENCH],
                           [case.target, case.weather, *P2_BENCH],
                           policy=RandomnessPolicy(damage_roll=0, accuracy=accuracy))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    handle.step([f"{case.attack}, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    after, rolls = handle.position, handle.rolls
    handle.close()
    return before, after, rolls


@pytest.mark.parametrize("name", sorted(CLOAK))
def test_showdown_snow_cloak(oracle: Oracle, name: str) -> None:
    case = CLOAK[name]
    _, _, rolls = _play_cloak(oracle, case, "hit")
    hundreds = [r["numerator"] for r in rolls if r.get("kind") == "chance" and r.get("denominator") == 100]
    assert hundreds[:1] == [case.accuracy], rolls


@pytest.mark.parametrize("name", sorted(CLOAK))
def test_the_port_matches_showdown_snow_cloak(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_branches, port_weights

    case = CLOAK[name]
    before, hit, _ = _play_cloak(oracle, case, "hit")
    _, missed, _ = _play_cloak(oracle, case, "miss")
    start = _loaded(before)
    chosen = _chosen(reg, start, (f"{case.attack}, move 1", "move 1, move 1"))
    budget = replace(Budget.deterministic(0), enumerate_accuracy=True, max_branches=4)
    branches = port_branches(port, start, chosen, budget)
    by_hp: dict[int, float] = {}
    for weight, pos in branches:
        by_hp[_p(pos, 1, 0).hp] = by_hp.get(_p(pos, 1, 0).hp, 0.0) + weight
    hit_hp = _p(Position.from_json(hit), 1, 0).hp
    missed_hp = _p(Position.from_json(missed), 1, 0).hp
    assert missed_hp > hit_hp
    assert by_hp.get(hit_hp, 0.0) == pytest.approx(case.accuracy / 100), by_hp
    if case.accuracy < 100:
        assert by_hp.get(missed_hp, 0.0) == pytest.approx(1 - case.accuracy / 100), by_hp
    assert _notes(port_weights(port, start, chosen, budget)) == []
