"""Transform and Imposter (IKA-219).

Showdown a5df827 (the champions mod overrides neither)::

    transform: {                                           // data/moves.ts
        accuracy: true, category: "Status", pp: 10, target: "normal",
        flags: { allyanim: 1, failencore: 1, noassist: 1, failcopycat: 1, failmimic: 1, failinstruct: 1 },
        onHit(target, pokemon) { return pokemon.transformInto(target); },
    },
    imposter: {                                            // data/abilities.ts
        onSwitchIn(pokemon) {
            const target = pokemon.side.foe.active[pokemon.side.foe.active.length - 1 - pokemon.position];
            if (target) pokemon.transformInto(target, this.dex.abilities.get('imposter'));
        },
    },

`Pokemon#transformInto` (sim/pokemon.ts:1270) fails on a fainted target, an Illusion, a
target behind a Substitute, a target or a user already transformed; otherwise it copies the
species, the types, the stored stats but HP, the moves at 5 PP, the boosts, `timesAttacked`
and the crit volatiles, and last the ability, whose `Start` runs (a copied Intimidate
lands). `clearVolatile` -- a switch out, a faint -- gives back `baseAbility`,
`baseMoveSlots` and the species.

The port neither transformed nor reverted: a Ditto fought as itself, and a position with a
transformed Pokemon (which only Showdown made) was refused. Every case is played by
Showdown first; the Showdown tests assert what it did, and the port twins hold the port to
the same positions. The controls: a Limber Ditto at the lead, a Ditto that Protects.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position
from pokeuraou.priors import SampledSet
from pokeuraou.regulation import to_id
from pokeuraou.selfplay import _opening

from ._port import Budget, apply_lead_abilities
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

SLOW = {"hp": 32, "atk": 10, "def": 10, "spa": 10, "spd": 4, "spe": 0}
FAST = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 0, "spe": 32}


def _set(species: str, ability: str, moves: list[str], sp: dict | None = None,
         item: str | None = None) -> SampledSet:
    """Not legal sets: the bridge checks neither learnsets nor abilities."""
    return SampledSet(species=species, ability=ability, item=item, nature="Serious",
                      sp=dict(sp or SLOW), moves=moves)


DITTO = _set("ditto", "imposter", ["transform"], {"hp": 32, "spe": 32}, item="choicescarf")
SLOW_DITTO = _set("ditto", "imposter", ["transform"], {"hp": 32, "def": 32})
LIMBER = _set("ditto", "limber", ["transform", "protect"], {"hp": 32, "def": 32})
INCIN = _set("incineroar", "intimidate", ["fakeout", "flareblitz", "protect", "partingshot"])
FAST_INCIN = _set("incineroar", "intimidate", ["fakeout", "flareblitz", "protect", "partingshot"], FAST)
BLAZE = _set("incineroar", "blaze", ["fakeout", "flareblitz", "protect", "partingshot"])
CHOMP = _set("garchomp", "roughskin", ["earthquake", "protect", "dragonclaw", "swordsdance"], FAST)
SUB_CHOMP = _set("garchomp", "roughskin", ["substitute", "protect", "dragonclaw", "swordsdance"], FAST)
SYLV = _set("sylveon", "pixilate", ["hypervoice", "protect", "moonblast", "calmmind"])
ZARD = _set("charizard", "blaze", ["heatwave", "airslash", "protect", "solarbeam"])
GENGAR = _set("gengar", "cursedbody", ["memento", "protect", "shadowball", "sludgebomb"])


def _team(reg: Any, sets: list[SampledSet]) -> list[TeamSet]:  # noqa: ANN401
    return [TeamSet.from_json(s.to_team_set_json(reg)) for s in sets]


def _p(pos: Position, side: int, slot: int) -> Pokemon:
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _types(reg: Any, mon: Pokemon) -> tuple[str, ...]:  # noqa: ANN401
    return tuple(mon.types) or tuple(reg.species[mon.species].types)


def _state(reg: Any, mon: Pokemon, pp: bool = True) -> dict[str, Any]:  # noqa: ANN401
    """What a Transform copies and a switch out gives back, read off one Pokemon.

    ``pp=False`` for an opening built from the sets, whose own PP are the dex's and not the
    champions mod's (Protect 5 against 8): not the Transform's to answer for.
    """
    out: dict[str, Any] = {
        "species": mon.species,
        "ability": mon.ability,
        "types": _types(reg, mon),
        "boosts": {k: v for k, v in sorted(mon.boosts.items()) if v},
        "hp": mon.hp,
        "fainted": mon.fainted,
        "active": mon.active_index,
        "transformed": mon.transformed,
        "moves": [(m.id, m.pp, m.maxpp) if pp or mon.transformed else m.id for m in mon.moves],
        "isMega": mon.is_mega,
    }
    if mon.transformed:
        # Copied from the target. (The port counts no hit into it: a gap of its own.)
        out["timesAttacked"] = mon.times_attacked
        assert mon.stats_override is not None
        out["stats"] = {k: v for k, v in mon.stats_override.items()}
        out["base"] = (mon.base_ability, [(m.id, m.pp) if pp else m.id for m in mon.base_moves or []])
    return out


def _everyone(reg: Any, pos: Position, pp: bool = True) -> dict[str, dict[str, Any]]:  # noqa: ANN401
    return {
        f"p{s + 1}.{to_id(mon.base_species)}": _state(reg, mon, pp)
        for s, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
    return pos


def _chosen(reg: Any, pos: Position, step: tuple[str, str]) -> list[Any]:  # noqa: ANN401
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        if choice.startswith("move 1 -2,") and choice not in menu:
            # The menu offers no move at an ally but a spread or an ally-only one; Showdown
            # takes "move 1 -2" for Transform, and so does the port.
            foe = menu[choice.replace("move 1 -2,", "move 1 1,")]
            first = dataclasses.replace(foe.slots[0], target=-2)
            menu[choice] = dataclasses.replace(foe, slots=(first, *foe.slots[1:]))
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


# ---------------------------------------------------------------------------
# Imposter at the lead: which foe, in what order, and the copied ability's Start.

LEADS: dict[str, tuple[list[SampledSet], list[SampledSet]]] = {
    # p1a (position 0) takes foe.active[1] -- across, not in front.
    "imposter at p1a copies p2b": ([DITTO, INCIN, SYLV, ZARD], [CHOMP, INCIN, SYLV, ZARD]),
    # p1b takes foe.active[0].
    "imposter at p1b copies p2a": ([INCIN, DITTO, SYLV, ZARD], [CHOMP, BLAZE, SYLV, ZARD]),
    # A slower Ditto: the foe's Intimidate lands first, and the Transform copies the
    # target's boosts over the drop.
    "slow imposter after the foe's intimidate": (
        [SLOW_DITTO, SYLV, INCIN, ZARD], [CHOMP, FAST_INCIN, SYLV, ZARD]),
    # A Ditto copying a Limber Ditto: Limber and the Transform move come across.
    "imposter copies an untransformed ditto": (
        [DITTO, SYLV, INCIN, ZARD], [CHOMP, LIMBER, SYLV, ZARD]),
    # Controls: Limber does nothing at the lead; no Ditto at all.
    "control: limber at the lead": ([LIMBER, INCIN, SYLV, ZARD], [CHOMP, INCIN, SYLV, ZARD]),
    "control: no ditto": ([BLAZE, INCIN, SYLV, ZARD], [CHOMP, INCIN, SYLV, ZARD]),
}


def _play_leads(reg: Any, oracle: Oracle, name: str) -> dict:  # noqa: ANN401
    p1, p2 = LEADS[name]
    handle = oracle.create(FORMAT_ID, _team(reg, p1), _team(reg, p2),
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    out = handle.position
    handle.close()
    return out


def test_showdown_imposter_at_the_lead(reg, oracle: Oracle) -> None:  # noqa: ANN001
    across = Position.from_json(_play_leads(reg, oracle, "imposter at p1a copies p2b"))
    ditto = _p(across, 0, 0)
    assert ditto.transformed and ditto.species == "incineroar" and ditto.ability == "intimidate"
    assert ditto.base_ability == "imposter" and [m.id for m in ditto.base_moves or []] == ["transform"]
    assert [m.pp for m in ditto.moves] == [5, 5, 5, 5]
    assert ditto.hp == ditto.maxhp < _p(across, 1, 1).maxhp  # HP stays Ditto's
    # The copied Intimidate landed on both foes, with p1b's own: -2 each.
    assert _p(across, 1, 0).boost("atk") == _p(across, 1, 1).boost("atk") == -2

    other = Position.from_json(_play_leads(reg, oracle, "imposter at p1b copies p2a"))
    assert _p(other, 0, 1).species == "garchomp" and _p(other, 0, 1).ability == "roughskin"

    slow = Position.from_json(_play_leads(reg, oracle, "slow imposter after the foe's intimidate"))
    # The Intimidate's -1 on the Ditto is gone under the target's boosts; p1b keeps it.
    assert _p(slow, 0, 0).transformed and _p(slow, 0, 0).boost("atk") == 0
    assert _p(slow, 0, 1).boost("atk") == -1
    assert _p(slow, 1, 0).boost("atk") == -1  # the copied Intimidate

    same = Position.from_json(_play_leads(reg, oracle, "imposter copies an untransformed ditto"))
    assert _p(same, 0, 0).transformed and _p(same, 0, 0).species == "ditto"
    assert _p(same, 0, 0).ability == "limber"

    limber = Position.from_json(_play_leads(reg, oracle, "control: limber at the lead"))
    assert not _p(limber, 0, 0).transformed and _p(limber, 0, 0).species == "ditto"


@pytest.mark.parametrize("name", sorted(LEADS))
def test_the_ports_leads_match_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The generation path: the opening from the sets, then the port's lead switch-ins."""
    want = Position.from_json(_play_leads(reg, oracle, name))
    p1, p2 = LEADS[name]
    ours = apply_lead_abilities(reg, _opening(reg, p1, p2)).position
    assert _everyone(reg, ours, pp=False) == _everyone(reg, want, pp=False)


def test_a_transformed_pokemon_does_not_show_the_species_it_copied(reg) -> None:  # noqa: ANN001
    """The belief's "what they have shown" (`hidden.shown_species`) reads species off the
    board. A Ditto wearing the foe's Incineroar has shown a Ditto, not an Incineroar: its
    own Incineroar, unseen on the bench, stays a candidate for the hidden slots. Without
    this the candidates lost it, and a sheet left short of candidates raises."""
    from pokeuraou.hidden import seen_slots, shown_species

    ours = apply_lead_abilities(reg, _opening(reg, [DITTO, SYLV, INCIN, ZARD],
                                              [CHOMP, INCIN, SYLV, ZARD])).position
    assert _p(ours, 0, 0).transformed and _p(ours, 0, 0).species == "incineroar"
    shown = shown_species(ours, 0, seen_slots(ours, 0))
    assert "ditto" in shown and "incineroar" not in shown, shown


# ---------------------------------------------------------------------------
# Turns: the move (whom it copies, what, when it fails), then a transformed Pokemon's own
# turns -- a copied move, a switch out and back in, a faint.

LIMBER_P1 = [LIMBER, SYLV, INCIN, ZARD]
IMPOSTER_P1 = [DITTO, INCIN, SYLV, ZARD]
SD_FIRST = ("move 1 1, move 2", "move 4, move 3")  # Transform p2a; Swords Dance first


@dataclasses.dataclass(frozen=True)
class TurnCase:
    p1: list[SampledSet]
    p2: list[SampledSet]
    #: Turns played before the one compared, and the one compared.
    steps: tuple[tuple[str, str], ...]


TURNS: dict[str, TurnCase] = {
    # The move copies p2a after its Swords Dance: species, types, stats, 5 PP, +2 Atk.
    "transform copies a boosted foe": TurnCase(LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD], (SD_FIRST,)),
    # The copied Intimidate starts at once and lowers both foes (through Protect).
    "transform into an intimidate holder": TurnCase(
        LIMBER_P1, [CHOMP, INCIN, SYLV, ZARD], (("move 1 2, move 2", "move 2, move 3"),)),
    # The ally, after its Calm Mind.
    "transform into the ally": TurnCase(
        LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD], (("move 1 -2, move 4", "move 2, move 3"),)),
    # No `protect` flag: Protect does not stop it.
    "transform through protect": TurnCase(
        LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD], (("move 1 1, move 2", "move 2, move 3"),)),
    # The failures.
    "transform fails on a substitute": TurnCase(
        LIMBER_P1, [SUB_CHOMP, BLAZE, SYLV, ZARD], (("move 1 1, move 2", "move 1, move 3"),)),
    "transform fails on a transformed foe": TurnCase(
        LIMBER_P1, [DITTO, BLAZE, SYLV, ZARD], (("move 1 1, move 2", "move 2, move 3"),)),
    "a transformed user cannot transform again": TurnCase(
        IMPOSTER_P1, [CHOMP, LIMBER, SYLV, ZARD], (("move 1 1, move 3", "move 2, move 2"),)),
    # A transformed Pokemon's turns.
    "a copied move at 5 pp": TurnCase(
        IMPOSTER_P1, [CHOMP, INCIN, SYLV, ZARD], (("move 2 1, move 3", "move 4, move 3"),)),
    "a transformed ditto switches out": TurnCase(
        IMPOSTER_P1, [CHOMP, INCIN, SYLV, ZARD], (("switch 3, move 3", "move 4, move 3"),)),
    "a ditto switches back in and transforms again": TurnCase(
        IMPOSTER_P1, [CHOMP, INCIN, SYLV, ZARD],
        (("switch 3, move 3", "move 4, move 3"), ("switch 3, move 2 1", "move 4, move 2 2"))),
    "a transformed ditto faints": TurnCase(
        IMPOSTER_P1, [CHOMP, GENGAR, SYLV, ZARD], (("move 1 1, move 3", "move 4, move 2"),)),
    # A switch names its Pokemon by species: the Ditto that copied the foe's Garchomp is
    # not the Garchomp its own side calls in (the port's lookup took the Ditto).
    "the own garchomp comes in beside a ditto that copied one": TurnCase(
        [DITTO, SYLV, CHOMP, ZARD], [BLAZE, CHOMP, SYLV, ZARD], (("move 2, switch 3", "move 3, move 2"),)),
    # The Transform's own PP, spent before it, comes back with the base moves.
    "the base moves keep the pp spent before": TurnCase(
        LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD],
        (SD_FIRST, ("switch 3, move 4", "move 4, move 2 2"))),
    # Controls: a Ditto that Protects, and one that stays untransformed beside the rest.
    "control: the ditto protects": TurnCase(
        LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD], (("move 2, move 4", "move 4, move 3"),)),
    "control: the limber ditto is hit": TurnCase(
        LIMBER_P1, [CHOMP, BLAZE, SYLV, ZARD], (("move 2, move 4", "move 3 1, move 2 1"),)),
}


def _play_turns(reg: Any, oracle: Oracle, case: TurnCase) -> tuple[dict, dict]:  # noqa: ANN401
    """Showdown's positions before and after the last step."""
    handle = oracle.create(FORMAT_ID, _team(reg, case.p1), _team(reg, case.p2),
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    for step in case.steps:
        before = handle.position
        handle.step(list(step))
        assert handle.choice_errors == [], handle.choice_errors
    after = handle.position
    handle.close()
    return before, after


def test_showdown_transform_turns(reg, oracle: Oracle) -> None:  # noqa: ANN001
    played = {name: [Position.from_json(p) for p in _play_turns(reg, oracle, case)]
              for name, case in TURNS.items()}

    _, after = played["transform copies a boosted foe"]
    ditto = _p(after, 0, 0)
    assert ditto.transformed and ditto.species == "garchomp" and ditto.boost("atk") == 2
    assert [(m.id, m.pp) for m in ditto.moves][:1] == [("earthquake", 5)]
    assert ditto.base_ability == "limber"

    before, after = played["transform into an intimidate holder"]
    assert _p(after, 1, 0).boost("atk") == _p(before, 1, 0).boost("atk") - 1
    _, after = played["transform into the ally"]
    assert _p(after, 0, 0).species == "sylveon" and _p(after, 0, 0).boost("spa") == 1
    _, after = played["transform through protect"]
    assert _p(after, 0, 0).species == "garchomp"
    for name in ("transform fails on a substitute", "transform fails on a transformed foe"):
        _, after = played[name]
        assert not _p(after, 0, 0).transformed, name
    _, after = played["a transformed user cannot transform again"]
    assert _p(after, 0, 0).species == "ditto" and _p(after, 0, 0).ability == "limber"

    _, after = played["a copied move at 5 pp"]
    assert [m.pp for m in _p(after, 0, 0).moves][1] == 4
    _, after = played["a transformed ditto switches out"]
    benched = next(m for m in after.sides[0].pokemon if to_id(m.base_species) == "ditto")
    assert (benched.species, benched.ability, benched.transformed) == ("ditto", "imposter", False)
    assert [m.id for m in benched.moves] == ["transform"]
    _, after = played["a ditto switches back in and transforms again"]
    assert _p(after, 0, 0).transformed and _p(after, 0, 0).species == "incineroar"
    _, after = played["a transformed ditto faints"]
    gone = next(m for m in after.sides[0].pokemon if to_id(m.base_species) == "ditto")
    assert gone.fainted and gone.species == "ditto" and gone.ability == "imposter"
    before, after = played["the base moves keep the pp spent before"]
    benched = next(m for m in after.sides[0].pokemon if to_id(m.base_species) == "ditto")
    assert [(m.id, m.pp) for m in benched.moves][0][1] == _p(before, 0, 0).base_moves[0].pp


@pytest.mark.parametrize("name", sorted(TURNS))
def test_the_port_matches_showdown_transform(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn, port_weights

    case = TURNS[name]
    before, after = _play_turns(reg, oracle, case)
    start = _loaded(before)
    chosen = _chosen(reg, start, case.steps[-1])
    ours = port_turn(port, start, chosen)
    assert _everyone(reg, ours) == _everyone(reg, Position.from_json(after))
    notes = port_weights(port, start, chosen, Budget.deterministic(0)).get("unmodelled") or []
    assert [n for n in notes if "transform" in n or "imposter" in n] == []

