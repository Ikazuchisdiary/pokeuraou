"""The traps generation never saw: Shadow Tag, the move locks, Ingrain, No Retreat and
Fairy Lock (IKA-169).

Generation plays its games in our resolver (`selfplay.play_game`), so Showdown's
`trapped` flag is never set there and `actions._is_trapped` decides every menu from the
position alone. IKA-163 found it knew only the three volatiles in its list. Showdown has
more sources, every one of them in the champions dex:

- **Shadow Tag** (`data/abilities.ts`, `shadowtag.onFoeTrapPokemon`)::

      if (!pokemon.hasAbility('shadowtag') && pokemon.isAdjacent(this.effectState.target)) {
          pokemon.tryTrap(true);
      }

  Mega Gengar is its only holder. `tryTrap` is where the Ghost immunity lives, and Shed
  Shell and the champions Run Away undo it at priority -10, so all three escapes of
  IKA-163 apply; a foe that has Shadow Tag itself is the fourth.
- **The move locks** (`sim/pokemon.ts`, `getMoveRequestData`)::

      let lockedMove = this.getLockedMove();
      const hardLocked = !!lockedMove;
      if (lockedMove) { this.trapped = true; }

  `getLockedMove` is the `LockMove` event: `twoturnmove` (the charge turn of Solar Beam,
  Electro Shot, Meteor Beam, Fly...), `lockedmove` (Outrage) and `mustrecharge`. It runs
  after the `TrapPokemon` event, so no escape undoes it, and the request then offers the
  one move and no Mega (`if (!lockedMove) { if (this.canMegaEvo) ... }`).
- **Ingrain, No Retreat, Fairy Lock**: each condition's `onTrapPokemon` calls
  `pokemon.tryTrap()`. Fairy Lock is a pseudo-weather, so it traps every active Pokemon on
  both sides for the one turn after it is used.

Each case is checked three ways: Showdown's own verdict; Showdown's position with that
verdict cleared, as the searched children carry it; and the position *our resolver* builds
from Showdown's previous turn, which is the form generation meets.
"""

from __future__ import annotations

import dataclasses

import pytest

from pokeuraou.actions import MoveAction, SwitchAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str], item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP), item=item
    )


SUBJECTS = {
    "control": _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"]),
    "ghost": _mon("Dragapult", "Clear Body", ["dragondance", "protect", "dragondarts", "uturn"]),
    "shedshell": _mon(
        "Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"],
        "Shed Shell",
    ),
    "runaway": _mon("Thievul", "Run Away", ["nastyplot", "protect", "darkpulse", "foulplay"]),
}
MILOTIC = _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"])
INCINEROAR = _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"])
SYLVEON = _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"])
#: Trace copies Shadow Tag: the only way a non-Ghost can hold it in this dex.
GARDEVOIR = _mon("Gardevoir", "Trace", ["calmmind", "protect", "moonblast", "psychic"])
GENGAR = _mon("Gengar", "Cursed Body", ["protect", "shadowball", "sludgebomb", "willowisp"], "Gengarite")
#: Forecast cannot be traced, so Gardevoir's Trace has one candidate: Mega Gengar.
CASTFORM = _mon("Castform", "Forecast", ["protect", "weatherball", "icebeam", "thunderbolt"])
HIPPOWDON = _mon("Hippowdon", "Sand Stream", ["slackoff", "protect", "earthquake", "yawn"])
KINGAMBIT = _mon("Kingambit", "Defiant", ["kowtowcleave", "protect", "suckerpunch", "ironhead"])
CHARIZARD = _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"])
MILOTIC_B = _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"])

TAGGERS = [GENGAR, CASTFORM, HIPPOWDON, KINGAMBIT]
QUIET_B = [HIPPOWDON, MILOTIC_B, KINGAMBIT, CHARIZARD]
#: Gengar Protects and Mega Evolves, Castform Protects; turn 2 is Shadow Tag's first.
MEGA_TURN = ["move 1, move 1", "move 1 mega, move 1"]
#: Everything in slot 0 uses its first move at the foe in slot 1 if it takes a target.
FIRST_MOVE = ["move 1 1, move 1", "move 1, move 1"]
SELF_MOVE = ["move 1, move 1", "move 1, move 1"]


def _lead(first: TeamSet) -> list[TeamSet]:
    return [first, MILOTIC, INCINEROAR, SYLVEON]


@dataclasses.dataclass(frozen=True)
class Case:
    team_a: list[TeamSet]
    team_b: list[TeamSet]
    steps: list[list[str]]
    #: Showdown's verdict for (side, slot) after the last step.
    trapped: dict[tuple[int, int], bool]
    #: Whether our resolver can build the last turn's position (the generation form).
    generation: bool = True


CASES: dict[str, Case] = {
    # Milotic, the last active, is trapped too: Showdown only *reports* it as
    # `maybeTrapped`, because a hidden trap on the last active would leak the foe's
    # ability, but `chooseSwitch` refuses it (`test_the_hidden_trap_is_a_trap`).
    "shadowtag/control": Case(
        _lead(SUBJECTS["control"]), TAGGERS, [MEGA_TURN], {(0, 0): True, (0, 1): True}
    ),
    "shadowtag/ghost": Case(
        _lead(SUBJECTS["ghost"]), TAGGERS, [MEGA_TURN], {(0, 0): False, (0, 1): True}
    ),
    "shadowtag/shedshell": Case(
        _lead(SUBJECTS["shedshell"]), TAGGERS, [MEGA_TURN], {(0, 0): False, (0, 1): True}
    ),
    "shadowtag/runaway": Case(
        _lead(SUBJECTS["runaway"]), TAGGERS, [MEGA_TURN], {(0, 0): False, (0, 1): True}
    ),
    # The Shed Shell Garchomp leaves for Gardevoir, which traces Shadow Tag. Our resolver
    # has no Trace, so only Showdown's position can say this.
    "shadowtag/holder": Case(
        [SUBJECTS["shedshell"], MILOTIC, GARDEVOIR, SYLVEON],
        TAGGERS,
        [MEGA_TURN, ["switch 3, move 1", "move 1, move 1"]],
        {(0, 0): False, (0, 1): True},
        generation=False,
    ),
    # The control: Gengar has Cursed Body until it evolves.
    "shadowtag/before-mega": Case(
        _lead(SUBJECTS["control"]), TAGGERS, [SELF_MOVE], {(0, 0): False, (0, 1): False}
    ),
    "twoturnmove/electroshot": Case(
        _lead(_mon("Archaludon", "Stamina", ["electroshot", "protect", "dracometeor", "flashcannon"])),
        QUIET_B,
        [FIRST_MOVE],
        {(0, 0): True, (0, 1): False},
    ),
    # With its stone: a charging Pokemon may not Mega Evolve either.
    "twoturnmove/solarbeam": Case(
        _lead(
            _mon(
                "Venusaur", "Chlorophyll", ["solarbeam", "protect", "sludgebomb", "gigadrain"],
                "Venusaurite",
            )
        ),
        QUIET_B,
        [FIRST_MOVE],
        {(0, 0): True, (0, 1): False},
    ),
    "lockedmove/outrage": Case(
        _lead(_mon("Garchomp", "Rough Skin", ["outrage", "protect", "earthquake", "swordsdance"])),
        QUIET_B,
        [SELF_MOVE],
        {(0, 0): True, (0, 1): False},
    ),
    # The positive control: the recharge turn was already a one-action menu.
    "mustrecharge/hyperbeam": Case(
        _lead(_mon("Snorlax", "Thick Fat", ["hyperbeam", "protect", "bodyslam", "curse"])),
        QUIET_B,
        [FIRST_MOVE],
        {(0, 0): True, (0, 1): False},
        # Hyper Beam can miss in our resolver's branches; the menu is pinned above.
        generation=False,
    ),
    "ingrain": Case(
        _lead(_mon("Venusaur", "Chlorophyll", ["ingrain", "protect", "sludgebomb", "gigadrain"])),
        QUIET_B,
        [SELF_MOVE],
        {(0, 0): True, (0, 1): False},
    ),
    "noretreat": Case(
        _lead(_mon("Falinks", "Battle Armor", ["noretreat", "protect", "closecombat", "rockslide"])),
        QUIET_B,
        [SELF_MOVE],
        {(0, 0): True, (0, 1): False},
    ),
    # Fairy Lock traps both sides, the Ghost Dragapult excepted.
    "fairylock": Case(
        _lead(_mon("Klefki", "Prankster", ["fairylock", "protect", "dazzlinggleam", "thunderwave"])),
        [HIPPOWDON, SUBJECTS["ghost"], KINGAMBIT, CHARIZARD],
        [SELF_MOVE],
        {(0, 0): True, (0, 1): True, (1, 0): True, (1, 1): False},
    ),
    # ... for one turn: the turn after, nobody is.
    "fairylock/next-turn": Case(
        _lead(_mon("Klefki", "Prankster", ["fairylock", "protect", "dazzlinggleam", "thunderwave"])),
        [HIPPOWDON, SUBJECTS["ghost"], KINGAMBIT, CHARIZARD],
        [SELF_MOVE, ["move 2, move 1", "move 1, move 1"]],
        {(0, 0): False, (0, 1): False, (1, 0): False, (1, 1): False},
        generation=False,
    ),
}

#: A move lock: which move Showdown's request leaves on offer, and whether it may Mega.
LOCKS = {
    "twoturnmove/electroshot": "electroshot",
    "twoturnmove/solarbeam": "solarbeam",
    "lockedmove/outrage": "outrage",
    "mustrecharge/hyperbeam": "recharge",
}

#: Cases whose trap Showdown's position does not carry. No Retreat was one until IKA-176
#: put its volatile in the bridge's `MODELLED_VOLATILES`.
NOT_IN_THE_DUMP: set[str] = set()


def _play(oracle: Oracle, case: Case):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, case.team_a, case.team_b, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    for step in case.steps:
        before = handle.position
        handle.step(step)
        assert handle.choice_errors == [], handle.choice_errors
    return handle, before


def _slot_menu(reg, pos: Position, side: int, slot: int):  # noqa: ANN001, ANN202
    pieces = [action.slots[slot] for action in side_actions(reg, pos, side)]
    switches = {p.party_index for p in pieces if isinstance(p, SwitchAction)}
    moves = {p.move_id for p in pieces if isinstance(p, MoveAction)}
    mega = any(isinstance(p, MoveAction) and p.mega for p in pieces)
    return switches, moves, mega


def _bench(pos: Position, side: int) -> set[int]:
    return {m.slot + 1 for m in pos.sides[side].pokemon if not m.fainted and not m.is_active}


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown_verdict(oracle: Oracle, name: str) -> None:
    """The fact: Showdown's internal `trapped` after the turn, and the lock's one move."""
    case = CASES[name]
    handle, _ = _play(oracle, case)
    pos = Position.from_json(handle.position)
    for (side, slot), trapped in case.trapped.items():
        mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
        assert mon.trapped is trapped, (name, side, slot, mon.species)
        active = handle.requests[side]["active"][slot]
        # The request says `trapped` except for a hidden trap on the last active.
        assert bool(active.get("trapped") or active.get("maybeTrapped")) is trapped, active
        if name in LOCKS and (side, slot) == (0, 0):
            assert [m["id"] for m in active["moves"]] == [LOCKS[name]], active
            assert not active.get("canMegaEvo"), active
    handle.close()


@pytest.mark.oracle
def test_the_hidden_trap_is_a_trap(oracle: Oracle) -> None:
    """Milotic's request says only `maybeTrapped`; the switch is refused all the same."""
    handle, _ = _play(oracle, CASES["shadowtag/ghost"])
    active = handle.requests[0]["active"]
    assert not active[1].get("trapped") and active[1].get("maybeTrapped"), active
    handle.step(["move 1, switch 3", "move 1, move 1"])
    assert any("trapped" in e for e in handle.choice_errors), handle.choice_errors
    handle.close()
    # The control: the Ghost beside it may switch.
    handle, _ = _play(oracle, CASES["shadowtag/ghost"])
    handle.step(["switch 3, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    handle.close()


def _check(reg, pos: Position, case: Case, name: str, form: str) -> None:  # noqa: ANN001
    for (side, slot), trapped in case.trapped.items():
        switches, moves, mega = _slot_menu(reg, pos, side, slot)
        bench = _bench(pos, side)
        assert bench, "the case needs somebody to switch to"
        assert (not switches) is trapped, (form, name, side, slot, switches)
        if not trapped:
            assert switches == bench, (form, name, side, slot, switches, bench)
        if name in LOCKS and (side, slot) == (0, 0):
            assert moves == {LOCKS[name]}, (form, name, moves)
            assert not mega, (form, name)


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(set(CASES) - NOT_IN_THE_DUMP))
def test_we_agree_once_the_flag_is_gone(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """Showdown's position with its verdict cleared, as a searched child carries it."""
    case = CASES[name]
    handle, _ = _play(oracle, case)
    pos = Position.from_json(handle.position)
    handle.close()
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
    _check(reg, pos, case, name, "showdown position")


GENERATION = sorted(n for n, c in CASES.items() if c.generation)


@pytest.mark.oracle
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "our resolver adds Outrage's `lockedmove` with no move and no duration "
                    "and never locks the move (IKA-169's report); the menu cannot know"
                ),
            ),
        )
        if n == "lockedmove/outrage"
        else n
        for n in GENERATION
    ],
)
def test_the_position_our_resolver_builds(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The generation form: our resolver plays Showdown's last turn, then we build the menu."""
    case = CASES[name]
    handle, before = _play(oracle, case)
    after = Position.from_json(handle.position)
    handle.close()
    start = Position.from_json(before)
    chosen = []
    for side, choice in enumerate(case.steps[-1]):
        menu = {a.to_choice(): a for a in side_actions(reg, start, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    result = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    assert not result.suspended
    assert result.branches
    for branch in result.branches:
        child = branch.position
        assert all(not m.trapped for s in child.sides for m in s.pokemon)
        _check(reg, child, case, name, "our resolver's child")
        if name.startswith("twoturnmove/"):
            # The marker itself is Showdown's: the move, and one turn left of its two.
            ours = child.sides[0].pokemon[child.sides[0].active[0]].volatile("twoturnmove")
            theirs = after.sides[0].pokemon[after.sides[0].active[0]].volatile("twoturnmove")
            assert theirs is not None and ours is not None
            assert (ours.move, ours.duration) == (theirs.move, theirs.duration), (ours, theirs)


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["twoturnmove/electroshot", "twoturnmove/solarbeam"])
def test_the_port_charges_the_same_way(
    reg,  # noqa: ANN001
    oracle: Oracle,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """The port stores the charge's move and duration as Python does.

    The port builds no menus, but it resolves the turns generation plays, and the next
    turn's menu is read off the position it hands back. The binary built from master
    before IKA-169 returns `twoturnmove` with neither.
    """
    from pokeuraou import rustnode

    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    case = CASES[name]
    handle, before = _play(oracle, case)
    handle.close()
    start = Position.from_json(before)
    # Showdown's stats ride along as an override, which the port refuses as a
    # transformed Pokemon; the spreads are known, so the stats are too.
    for side in start.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    chosen = []
    for side, choice in enumerate(case.steps[-1]):
        menu = {a.to_choice(): a for a in side_actions(reg, start, side)}
        chosen.append(menu[choice])
    try:
        node = rustnode.node_for(reg)
        assert node is not None
        ported = node.resolve(start, chosen, Budget.matrix(), select=0)
    finally:
        rustnode.reset()
    assert ported is not None and ported.position is not None, "the port refused the turn"
    ours = resolve_turn(reg, start, chosen, budget=Budget.matrix())
    assert len(ours.branches) == len(ported.branches) == 1
    python_mark = ours.branches[0].position.sides[0].pokemon[0].volatile("twoturnmove")
    port_mark = ported.position.sides[0].pokemon[0].volatile("twoturnmove")
    assert python_mark is not None and port_mark is not None
    assert (port_mark.move, port_mark.duration) == (python_mark.move, python_mark.duration)
    assert port_mark.move == LOCKS[name]


# ---------------------------------------------------------------------------
# No oracle: hand-built positions.


def _hand_built(reg, first: TeamSet, foes: list[TeamSet]):  # noqa: ANN001, ANN202
    from .test_actions import _synthetic_position

    pos = _synthetic_position(reg, _lead(first))
    other = _synthetic_position(reg, foes)
    pos.sides[1] = other.sides[0]
    pos.sides[1].id = pos.sides[1].name = "p2"
    return pos


def _switches(reg, pos: Position, side: int = 0, slot: int = 0) -> set[int]:  # noqa: ANN001
    return _slot_menu(reg, pos, side, slot)[0]


def _tagger(reg, pos: Position) -> None:  # noqa: ANN001
    gengar = pos.sides[1].pokemon[pos.sides[1].active[0]]
    assert gengar.species == "gengar"
    gengar.species = "gengarmega"
    gengar.types = reg.species["gengarmega"].types
    gengar.ability = "shadowtag"
    gengar.is_mega = True


def test_shadow_tag_needs_its_holder_on_the_field(reg) -> None:  # noqa: ANN001
    pos = _hand_built(reg, SUBJECTS["control"], TAGGERS)
    assert _switches(reg, pos), "plain Gengar has Cursed Body"
    _tagger(reg, pos)
    assert not _switches(reg, pos) and not _switches(reg, pos, 0, 1)
    gengar = pos.sides[1].pokemon[pos.sides[1].active[0]]
    gengar.fainted = True
    gengar.hp = 0
    assert _switches(reg, pos) and _switches(reg, pos, 0, 1), "a fainted holder traps nobody"


def test_shadow_tag_does_not_trap_its_own_side(reg) -> None:  # noqa: ANN001
    pos = _hand_built(reg, SUBJECTS["control"], TAGGERS)
    _tagger(reg, pos)
    # Castform, Gengar's partner, is not its foe.
    assert _switches(reg, pos, 1, 1)


@pytest.mark.parametrize("escape", ["ghost", "shedshell", "runaway", "shadowtag"])
def test_the_escapes_from_shadow_tag(reg, escape: str) -> None:  # noqa: ANN001
    subject = SUBJECTS["control"] if escape == "shadowtag" else SUBJECTS[escape]
    pos = _hand_built(reg, subject, TAGGERS)
    _tagger(reg, pos)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    if escape == "shadowtag":
        mon.ability = "shadowtag"
    assert _switches(reg, pos), escape
    # The control: without the escape the same Pokemon is trapped.
    if escape == "ghost":
        mon.types = ("Water",)
    elif escape == "shedshell":
        mon.item = None
    else:
        mon.ability = "roughskin"
    assert not _switches(reg, pos), escape


@pytest.mark.parametrize(
    ("ability", "subject", "trapped"),
    [
        # Magnet Pull traps a Steel type.
        (
            "magnetpull",
            _mon("Kingambit", "Defiant", ["protect", "kowtowcleave", "ironhead", "swordsdance"]),
            True,
        ),
        ("magnetpull", SUBJECTS["control"], False),
        # Arena Trap traps whatever is grounded.
        ("arenatrap", SUBJECTS["control"], True),
        ("arenatrap", _mon("Charizard", "Blaze", ["protect", "heatwave", "airslash", "solarbeam"]), False),
    ],
)
def test_the_other_trapping_abilities(reg, ability: str, subject: TeamSet, trapped: bool) -> None:  # noqa: ANN001
    """No holder in this dex, so hand-built only: Showdown's two conditions."""
    pos = _hand_built(reg, subject, TAGGERS)
    pos.sides[1].pokemon[pos.sides[1].active[0]].ability = ability
    assert (not _switches(reg, pos)) is trapped


def test_arena_trap_reads_groundedness(reg) -> None:  # noqa: ANN001
    pos = _hand_built(reg, SUBJECTS["control"], TAGGERS)
    pos.sides[1].pokemon[pos.sides[1].active[0]].ability = "arenatrap"
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    assert not _switches(reg, pos)
    mon.item = "airballoon"
    assert _switches(reg, pos), "an Air Balloon holder is not grounded"
    pos.field.pseudo_weather.append(Effect(id="gravity", duration=5))
    assert not _switches(reg, pos), "under Gravity everything is grounded"


def test_the_trap_sources_come_from_the_dump(reg) -> None:  # noqa: ANN001
    """Every move whose condition has `onTrapPokemon`, and every foe-trapping ability."""
    from pokeuraou.actions import TRAPPING_ABILITIES

    volatiles = set()
    pseudo = set()
    for move in reg.moves.values():
        if "condition.onTrapPokemon" not in move.raw.get("customHooks", ()):
            continue
        if move.raw.get("volatileStatus"):
            volatiles.add(move.raw["volatileStatus"])
        if move.raw.get("pseudoWeather"):
            pseudo.add(move.raw["pseudoWeather"])
    assert volatiles == {"ingrain", "noretreat", "octolock"}
    assert pseudo == {"fairylock"}
    assert reg.trapping_volatiles == volatiles
    assert reg.trapping_pseudo_weather == pseudo
    foe_trappers = {
        a.id for a in reg.abilities.values() if "onFoeTrapPokemon" in a.raw.get("customHooks", ())
    }
    assert set(TRAPPING_ABILITIES) == foe_trappers


@pytest.mark.parametrize("volatile", ["twoturnmove", "lockedmove"])
def test_a_locked_move_is_the_whole_menu(reg, volatile: str) -> None:  # noqa: ANN001
    venusaur = _mon(
        "Venusaur", "Chlorophyll", ["solarbeam", "protect", "sludgebomb", "outrage"], "Venusaurite"
    )
    pos = _hand_built(reg, venusaur, QUIET_B)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    move = "solarbeam" if volatile == "twoturnmove" else "outrage"
    switches, moves, mega = _slot_menu(reg, pos, 0, 0)
    assert switches and len(moves) == 4 and mega, "the control: nothing locked yet"
    mon.volatiles.append(Effect(id=volatile, duration=1, move=move))
    switches, moves, mega = _slot_menu(reg, pos, 0, 0)
    assert (switches, moves, mega) == (set(), {move}, False)
    # Shed Shell does not undo a lock: `trapped = true` comes after the TrapPokemon event.
    mon.item = "shedshell"
    assert _slot_menu(reg, pos, 0, 0)[0] == set()


def test_a_recorded_charge_without_its_move(reg) -> None:  # noqa: ANN001
    """Records made before IKA-169 carry `twoturnmove` with no move: the last move is it,
    if that move charges; a leaked marker beside another last move locks nothing."""
    venusaur = _mon("Venusaur", "Chlorophyll", ["solarbeam", "protect", "sludgebomb", "gigadrain"])
    pos = _hand_built(reg, venusaur, QUIET_B)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.volatiles.append(Effect(id="twoturnmove"))
    mon.last_move = "solarbeam"
    assert _slot_menu(reg, pos, 0, 0)[:2] == (set(), {"solarbeam"})
    mon.last_move = "sludgebomb"
    switches, moves, _ = _slot_menu(reg, pos, 0, 0)
    assert switches and len(moves) == 4
