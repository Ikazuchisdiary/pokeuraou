"""Legal action enumeration, checked against Showdown.

This is the third of the three components whose failure would invalidate every number the
tool prints: if the action set is wrong, the matrix is wrong, and so is the equilibrium.

The test does not assert a hand-written list of legal choices. It plays real battles and,
at every decision point, asks Showdown two questions:

  1. does it accept every choice string we generate?  (no false positives)
  2. does it reject every near-miss we deliberately exclude?  (no missing restrictions)

Question 2 is the one that catches an enumerator that is merely permissive.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pytest

from pokeuraou.actions import (
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.oracle import BattleHandle, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import (
    TARGETS_REQUIRING_ALLY,
    TARGETS_REQUIRING_FOE,
    TARGETS_WITHOUT_CHOICE,
    Regulation,
)

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _near_misses(reg: Regulation, pos: Position, side_index: int, legal: set[str]) -> list[str]:
    """Choice strings that must all be rejected, built by breaking one rule at a time."""
    side = pos.sides[side_index]
    n_slots = len(side.active)
    out: set[str] = set()

    per_slot_moves: list[list[MoveAction]] = []
    per_slot_switches: list[list[SwitchAction]] = []
    for slot in range(n_slots):
        moves: list[MoveAction] = []
        switches: list[SwitchAction] = []
        for a in side_actions(reg, pos, side_index):
            for sa in a.slots:
                if sa.slot != slot:
                    continue
                if isinstance(sa, MoveAction) and sa not in moves:
                    moves.append(sa)
                elif isinstance(sa, SwitchAction) and sa not in switches:
                    switches.append(sa)
        per_slot_moves.append(moves)
        per_slot_switches.append(switches)

    def pair(a: str, b: str) -> str:
        return f"{a}, {b}"

    baseline = [
        (m[0].to_choice() if m else "pass") for m in per_slot_moves
    ]

    for slot in range(n_slots):
        other = 1 - slot if n_slots == 2 else slot
        for mv in per_slot_moves[slot]:
            move = reg.moves.get(mv.move_id)
            if move is None:
                continue
            # Wrong target arity.
            if move.target in TARGETS_WITHOUT_CHOICE:
                broken = f"move {mv.move_index} 1"
            elif move.target in TARGETS_REQUIRING_FOE or move.target in TARGETS_REQUIRING_ALLY:
                broken = f"move {mv.move_index}"
            else:
                continue
            parts = list(baseline)
            parts[slot] = broken
            out.add(pair(*parts) if n_slots == 2 else parts[0])

        # A move index past the last slot is never legal.
        parts = list(baseline)
        parts[slot] = "move 5"
        out.add(pair(*parts) if n_slots == 2 else parts[0])

        # Switching to an active or fainted party member.
        for mon in side.pokemon:
            if not (mon.is_active or mon.fainted):
                continue
            parts = list(baseline)
            parts[slot] = f"switch {mon.slot + 1}"
            cand = pair(*parts) if n_slots == 2 else parts[0]
            if cand not in legal:
                out.add(cand)

        # Mega on a slot that cannot mega (no stone, wrong species, or already used).
        for mv in per_slot_moves[slot]:
            if mv.mega:
                continue
            parts = list(baseline)
            parts[slot] = f"{mv.to_choice()} mega"
            cand = pair(*parts) if n_slots == 2 else parts[0]
            if cand not in legal:
                out.add(cand)

        del other

    # Both slots declaring Mega.
    if n_slots == 2:
        mega0 = [m for m in per_slot_moves[0] if m.mega]
        mega1 = [m for m in per_slot_moves[1] if m.mega]
        if mega0 and mega1:
            out.add(pair(mega0[0].to_choice(), mega1[0].to_choice()))
        # Both slots switching to the same Pokemon.
        shared = {s.party_index for s in per_slot_switches[0]} & {
            s.party_index for s in per_slot_switches[1]
        }
        for idx in sorted(shared):
            out.add(pair(f"switch {idx}", f"switch {idx}"))
        # Passing a healthy Pokemon.
        if per_slot_moves[0] and per_slot_moves[1]:
            out.add(pair("pass", per_slot_moves[1][0].to_choice()))

    return sorted(out - legal)


def _assert_enumeration_matches(
    reg: Regulation, handle: BattleHandle, pos: Position, side_index: int
) -> tuple[int, int]:
    actions = side_actions(reg, pos, side_index)
    assert actions, f"no legal actions enumerated for side {side_index} at turn {pos.turn}"

    choices = [a.to_choice() for a in actions]
    assert len(choices) == len(set(choices)), "duplicate choice strings enumerated"

    accepted = handle.probe(side_index, choices)
    rejected = [r for r in accepted if not r["ok"]]
    assert not rejected, (
        f"turn {pos.turn} side {side_index}: Showdown rejected "
        f"{len(rejected)}/{len(choices)} of our choices, e.g. "
        + "; ".join(f"{r['choice']!r} -> {r.get('error')}" for r in rejected[:6])
    )

    misses = _near_misses(reg, pos, side_index, set(choices))
    if misses:
        probed = handle.probe(side_index, misses)
        wrongly_ok = [r["choice"] for r in probed if r["ok"]]
        assert not wrongly_ok, (
            f"turn {pos.turn} side {side_index}: Showdown ACCEPTS choices we exclude, so our "
            f"enumeration is missing {len(wrongly_ok)} legal action(s): {wrongly_ok[:8]}"
        )
    return len(choices), len(misses)


def _request_forced_switch(request: dict[str, Any] | None) -> list[bool] | None:
    if not request:
        return None
    fs = request.get("forceSwitch")
    if not fs:
        return None
    return [bool(x) for x in fs]


def _cross_check_against_request(
    reg: Regulation, pos: Position, side_index: int, request: dict[str, Any] | None
) -> None:
    """Our per-slot move set must match the move set Showdown offers the client."""
    if not request or not request.get("active"):
        return
    side = pos.sides[side_index]
    for slot, active in enumerate(request["active"]):
        if active is None or slot >= len(side.active):
            continue
        party = side.active[slot]
        if party is None or side.pokemon[party].fainted:
            continue
        theirs = {
            m["move"].lower().replace(" ", "").replace("-", "")
            for i, m in enumerate(active.get("moves", []))
            if not m.get("disabled")
        }
        ours = {
            a.move_id
            for action in side_actions(reg, pos, side_index)
            for a in action.slots
            if isinstance(a, MoveAction) and a.slot == slot
        }
        # Showdown reports move names; compare on ids by normalising both.
        theirs_ids = {t for t in theirs}
        ours_ids = {o.replace(" ", "").replace("-", "") for o in ours}
        assert ours_ids == theirs_ids or ours_ids == {"struggle"}, (
            f"turn {pos.turn} side {side_index} slot {slot}: our usable moves {sorted(ours_ids)} "
            f"differ from Showdown's request {sorted(theirs_ids)}"
        )


def _pick(rng: random.Random, actions: Sequence[SideAction]) -> SideAction:
    """Prefer actions that keep the battle interesting: megas and switches sometimes."""
    weights = []
    for a in actions:
        w = 1.0
        if a.declares_mega:
            w = 6.0
        elif any(isinstance(s, SwitchAction) for s in a.slots):
            w = 2.0
        weights.append(w)
    return rng.choices(list(actions), weights=weights, k=1)[0]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_enumeration_agrees_with_showdown_over_whole_battles(
    reg: Regulation,
    oracle: Oracle,
    team_a: list[TeamSet],
    team_b: list[TeamSet],
    seed: int,
) -> None:
    rng = random.Random(seed)
    handle = oracle.create(
        FORMAT_ID,
        team_a,
        team_b,
        seed=(seed, seed + 1, seed + 2, seed + 3),
        policy=RandomnessPolicy(damage_roll=8, accuracy="hit", crit=False, secondary=True),
    )
    # Team preview: pick leads. Enumeration of the team-preview choice is not part of the
    # solver (selection is out of scope), so drive it directly.
    order = [1, 2, 3, 4]
    rng.shuffle(order)
    handle.step([f"team {','.join(map(str, order))}", "team 1,2,3,4"])

    total_choices = 0
    total_misses = 0
    decision_points = 0

    for _ in range(40):
        pos = Position.from_json(handle.position)
        if pos.ended:
            break
        requests = handle.requests
        choices: list[str | None] = []
        for side_index in range(2):
            request = requests[side_index]
            forced = _request_forced_switch(request)
            if forced is not None:
                actions = switch_actions_after_faint(reg, pos, side_index, forced)
                # Replacement choices are also probed for acceptance.
                probed = handle.probe(side_index, [a.to_choice() for a in actions])
                bad = [r for r in probed if not r["ok"]]
                assert not bad, (
                    f"turn {pos.turn} side {side_index}: replacement choices rejected: "
                    + "; ".join(f"{r['choice']!r} -> {r.get('error')}" for r in bad[:5])
                )
                choices.append(_pick(rng, actions).to_choice() if actions else None)
                continue
            if not request or request.get("wait"):
                choices.append(None)
                continue

            _cross_check_against_request(reg, pos, side_index, request)
            n_choices, n_misses = _assert_enumeration_matches(reg, handle, pos, side_index)
            total_choices += n_choices
            total_misses += n_misses
            decision_points += 1

            actions = side_actions(reg, pos, side_index)
            choices.append(_pick(rng, actions).to_choice())

        if all(c is None for c in choices):
            break
        handle.step(choices)
        if handle.choice_errors:
            raise AssertionError(
                f"turn {pos.turn}: our chosen action was rejected: {handle.choice_errors}"
            )

    handle.close()
    print(
        f"[seed {seed}] decision points {decision_points}, "
        f"legal choices probed {total_choices}, near-misses probed {total_misses}"
    )
    assert decision_points >= 3, f"battle ended too early to be a real test ({decision_points})"
    assert total_choices > 200
    assert total_misses > 100


def test_mega_is_a_once_per_side_resource(reg: Regulation, team_a: list[TeamSet]) -> None:
    """With the stone holder active, exactly one slot at a time may declare Mega."""
    pos = _synthetic_position(reg, team_a)
    side = pos.sides[0]
    assert side.mega_capable_slots, "fixture should have a mega-capable Pokemon"

    actions = side_actions(reg, pos, 0)
    assert any(a.declares_mega for a in actions), "expected some Mega actions"
    assert all(sum(1 for s in a.slots if isinstance(s, MoveAction) and s.mega) <= 1 for a in actions)

    side.mega_used = True
    after = side_actions(reg, pos, 0)
    assert not any(a.declares_mega for a in after), "Mega must vanish once the side has used it"


def test_no_duplicate_switch_targets(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    for action in side_actions(reg, pos, 0):
        idx = action.switch_indices
        assert len(idx) == len(set(idx))


def test_trapped_pokemon_cannot_switch(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.trapped = True
    for action in side_actions(reg, pos, 0):
        for slot_action in action.slots:
            assert not (isinstance(slot_action, SwitchAction) and slot_action.slot == 0)


def test_fainted_slot_only_passes(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[1]]
    mon.hp = 0
    mon.fainted = True
    for action in side_actions(reg, pos, 0):
        assert isinstance(action.slots[1], PassAction)


def test_choice_lock_restricts_to_one_move(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    locked = mon.moves[1].id
    mon.locked_move = locked
    ids = {
        s.move_id
        for a in side_actions(reg, pos, 0)
        for s in a.slots
        if isinstance(s, MoveAction) and s.slot == 0
    }
    assert ids == {locked}


def test_taunt_removes_status_moves(reg: Regulation, team_a: list[TeamSet]) -> None:
    from pokeuraou.position import Effect

    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    status_moves = {m.id for m in mon.moves if reg.moves[m.id].category == "Status"}
    assert status_moves, "fixture slot should have at least one status move"
    mon.volatiles.append(Effect(id="taunt", duration=3))
    ids = {
        s.move_id
        for a in side_actions(reg, pos, 0)
        for s in a.slots
        if isinstance(s, MoveAction) and s.slot == 0
    }
    assert not (ids & status_moves)


def _synthetic_position(reg: Regulation, team: list[TeamSet]) -> Position:
    """A hand-built turn-1 position, used for unit checks that need no oracle."""
    from pokeuraou.position import Pokemon, Side
    from pokeuraou.regulation import STAT_IDS, to_id
    from pokeuraou.stats import stats_for

    def make(i: int, s: TeamSet) -> Pokemon:
        species_id = to_id(s.species)
        stats = stats_for(reg, species_id, s.nature, s.sp)
        return Pokemon(
            slot=i,
            species=species_id,
            base_species=species_id,
            types=reg.species[species_id].types,
            ability=to_id(s.ability),
            nature=s.nature,
            moves=[
                __import__("pokeuraou.position", fromlist=["MoveSlot"]).MoveSlot(
                    id=to_id(m), pp=reg.moves[to_id(m)].pp, maxpp=reg.moves[to_id(m)].pp
                )
                for m in s.moves
            ],
            hp=stats["hp"],
            maxhp=stats["hp"],
            item=to_id(s.item) if s.item else None,
            base_item=to_id(s.item) if s.item else None,
            sp={k: s.sp.get(k, 0) for k in STAT_IDS},
            active_index=i if i < 2 else None,
        )

    # Put the mega stone holder on the field so Mega actions exist.
    ordered = sorted(team, key=lambda s: 0 if reg.mega_target(to_id(s.species), to_id(s.item or "")) else 1)
    mons = [make(i, s) for i, s in enumerate(ordered)]
    mega_slots = [
        m.slot for m in mons if reg.mega_target(m.species, m.item) is not None
    ]
    side = Side(
        id="p1",
        name="p1",
        active=[0, 1],
        pokemon=mons,
        slot_conditions=[[], []],
        mega_capable_slots=mega_slots,
    )
    foe = Side(
        id="p2",
        name="p2",
        active=[0, 1],
        pokemon=[make(i, s) for i, s in enumerate(ordered)],
        slot_conditions=[[], []],
        mega_capable_slots=mega_slots,
    )
    foe.id = "p2"
    return Position(format=FORMAT_ID, sides=[side, foe], turn=1)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_a_target_is_named_rather_than_numbered(reg: Regulation) -> None:
    """"相手1" is a slot number; the reader wants to know which Pokemon.

    The target index in a choice string is relative to whoever is choosing, so the same
    "foe1" is a different Pokemon depending on the side. `target_names` resolves it per
    side, and getting that backwards would mislabel every line in the output -- so both
    directions are checked here.
    """
    from pokeuraou.actions import target_names
    from pokeuraou.names import localiser
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    roster = load_roster("rizabanadohido")
    four = roster.sets[: reg.meta.picked_team_size]
    # Two different sixes, so a mislabelled side is visible rather than symmetric.
    other = list(reversed(roster.sets))[: reg.meta.picked_team_size]
    pos = position_from_sets(reg, four, other)

    ours = target_names(pos, 0)
    theirs = target_names(pos, 1)
    assert ours.foes[0] == other[0].species
    assert ours.allies[0] == four[0].species
    # Symmetry: our foe slot 1 is their own slot 1.
    assert theirs.allies[0] == ours.foes[0]
    assert theirs.foes[0] == ours.allies[0]

    move = MoveAction(slot=0, move_index=1, move_id="tackle", target=1)
    named = move.describe(reg, None, ours)
    assert reg.species[other[0].species].name in named
    assert "foe1" not in named
    # Without the resolver, the numeric form is still what comes out.
    assert "foe1" in move.describe(reg)

    ally = MoveAction(slot=0, move_index=1, move_id="tackle", target=-2)
    ally_named = ally.describe(reg, None, ours)
    assert reg.species[four[1].species].name in ally_named
    # The side has to survive the substitution: an ally-targeting move reads completely
    # differently from a foe-targeting one, and a bare name does not say which.
    assert "ally" in ally_named

    loc = localiser(reg, "ja")
    if loc is not None:
        japanese = move.describe(reg, loc, ours)
        assert loc.species(other[0].species) in japanese
        assert "相手1" not in japanese
        assert "味方" in ally.describe(reg, loc, ours)


def test_an_empty_slot_falls_back_to_the_index(reg: Regulation) -> None:
    """There is no Pokemon to name, so the honest answer is the number."""
    from pokeuraou.actions import TargetNames

    empty = TargetNames(foes=(None, None), allies=(None, None))
    move = MoveAction(slot=0, move_index=1, move_id="tackle", target=1)
    assert "foe1" in move.describe(reg, None, empty)
