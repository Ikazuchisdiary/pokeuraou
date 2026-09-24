"""The single-turn resolver.

Two kinds of check. Unit tests pin the mechanics that are easy to get subtly wrong and
whose failure is silent -- the stratified damage rolls summing to one, Protect's repeat
penalty, the party-slot swap on a switch. Then the whole-turn differential test compares
resulting positions against Showdown field by field, with every source of chance pinned.

The differential threshold is a *measured* number and is split in two. A "flagged"
divergence is a turn where the resolver reported an unmodelled effect before producing an
answer: a documented gap. A "silent" divergence is a turn where it reported nothing and
was still wrong, which is the only kind that can quietly corrupt a printed value. The
silent rate is what the threshold holds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pokeuraou.actions import (
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, MoveSlot, Position
from pokeuraou.regulation import Regulation, to_id

# Python's resolver only where the search's own fold or fill is what is tested (IKA-209,
# IKA-212 take them): `turn_expectation`, `turn_leaves` and `batched_payoffs`' chunking.
from pokeuraou.resolve import Average, BestOf, turn_expectation, turn_leaves
from pokeuraou.resolve import resolve_turn as python_resolve_turn
from pokeuraou.resolve import resume_alternatives as python_resume_alternatives

from ._port import (
    Budget,
    PortRefused,
    resolve_turn,
    resume_turn,
    self_switches_needed,
)
from .conftest import FORMAT_ID
from .test_actions import _synthetic_position

#: Measured over 498 turns against the pinned Showdown commit: 7.8% total, 3.4% silent.
#: The headroom above the measurement is deliberate but small; a regression that pushes
#: past it fails here and ``tools/diverge_report.py`` names the cause by statistical lift.
#:
#: The total was 0.08, measured at 4.9%, while the harness skipped every turn containing a
#: self-switching move -- 75 turns in this same sweep. That skip hid the class completely,
#: and five real bugs with it: the mid-turn interrupt itself, Defiant answering a two-stat
#: drop once instead of twice, Parting Shot leaving when its drops did nothing, a
#: Prankster-boosted status move landing on a Dark type, and Throat Chop failing to lock
#: sound moves. On the same 16 seeds the old code scored 863 turns at 8.57% total with those
#: 75 skipped; this code scores 963 at 10.38% with none skipped, and 46 of the 58
#: newly-scored turns match exactly. The remaining gap is smaller than the spread between
#: two blocks of the same code (6.82% on seeds 1-8, 10.40% on 9-16).
#:
#: The silent rate keeps the threshold it had: it is the number that matters, it did not
#: need loosening (3.41% against 0.05), and giving it away here would cost the guard.
#:
#: The total came back down to 7.83% once the resolver learned the mechanics the newly
#: scored turns exposed -- secondary effects, the status probabilities the champions mod
#: overrides, item-extended durations, Cursed Body, Disable, Stance Change, Speed Boost and
#: the Prankster immunity. That is *below* the 8.57% the old code managed while skipping 75
#: turns, so the tightening is real coverage rather than a reclassification: the flagged
#: count fell from 31 to 23 at the same time, because thirteen effects were being declared
#: as "Showdown rolls it" when they are extended by an item and not rolled at all.
MAX_SILENT_DIVERGENCE = 0.05
MAX_TOTAL_DIVERGENCE = 0.10


# ---------------------------------------------------------------------------
# Randomness accounting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Turn resolution
# ---------------------------------------------------------------------------


def _move_action(pos: Position, side: int, slot: int, move_id: str, target: int | None) -> MoveAction:
    mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
    index = next(i for i, m in enumerate(mon.moves, start=1) if m.id == move_id)
    return MoveAction(slot=slot, move_index=index, move_id=move_id, target=target)


def test_probabilities_sum_to_one(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    for budget in (Budget.matrix(), Budget.fast()):
        for side_index in (0, 1):
            actions = side_actions(reg, pos, side_index)
            assert actions
        both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
        result = resolve_turn(reg, pos, both, budget=budget)
        assert result.branches
        assert result.total_probability == pytest.approx(1.0), budget


def _everyone_attacks(reg: Regulation, pos: Position, side: int) -> SideAction:
    """A choice in which both of the side's Pokemon use a damaging move, no mega."""
    for choice in side_actions(reg, pos, side):
        if all(
            isinstance(a, MoveAction)
            and not a.mega
            and reg.moves[a.move_id].category != "Status"
            for a in choice.slots
        ):
            return choice
    raise AssertionError(f"side {side} has no all-damaging choice")


@pytest.mark.parametrize("holder", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize(("kind", "name", "chance"), [
    ("item", "quickclaw", 0.2),
    ("ability", "quickdraw", 0.3),
])
def test_a_priority_roll_weighs_the_same_whoever_is_queued_first(
    reg: Regulation,
    team_a: list[TeamSet],
    holder: tuple[int, int],
    kind: str,
    name: str,
    chance: float,
) -> None:
    """Quick Claw's two queue branches weigh 0.2 and 0.8 wherever its holder is queued.

    They weighed 0.8 and 0.8 unless the holder's action was the first one queued. The
    weight is written onto each branch's first action, and the branches shared that object
    whenever it was queued before the split -- so the second write replaced the first. A
    turn with a claw anywhere but p1a carried a total probability of 1.6 and the claw fired
    half the time; Quick Draw's 0.3 became 1.4 the same way. The port clones its queues
    and rolled one in five all along, which is how `tools/diff_node.py` found it (IKA-70).
    """
    from pokeuraou.speed import build_queue
    from pokeuraou.view import battler, field_state

    pos = _synthetic_position(reg, team_a)
    side, slot = holder
    mon = pos.sides[side].active_pokemon()[slot]
    assert mon is not None
    if kind == "item":
        mon.item = mon.base_item = name
    else:
        mon.ability = name
    actions = [_everyone_attacks(reg, pos, 0), _everyone_attacks(reg, pos, 1)]
    fighters = [
        [battler(reg, m) if m is not None and not m.fainted else None for m in s.active_pokemon()]
        for s in pos.sides
    ]
    queues = build_queue(reg, pos, actions, fighters, field_state(pos, reg))
    assert sorted(q[0].branch_probability for q in queues) == pytest.approx(
        [chance, 1 - chance]
    )
    result = resolve_turn(reg, pos, actions, budget=Budget.matrix())
    assert result.total_probability == pytest.approx(1.0)


#: Showdown's own answer, measured by running `vendor/pokemon-showdown/dist/sim` on 2,000
#: seeds per row (IKA-145): the share of turns with `-activate|item: Quick Claw` and with
#: `-activate|ability: Quick Draw`, and the fractional value each leaves.
#: calmmind is status at priority 0, protect status at +4, quickattack physical at +1,
#: earthquake physical at 0. Quick Claw fired 401/2000 on every one of the four.
_SHOWDOWN_FRACTIONAL = [
    # item, ability, move -> {fractional value: probability}
    ("quickclaw", "owntempo", "calmmind", {0.1: 0.2, 0.0: 0.8}),
    ("quickclaw", "owntempo", "protect", {0.1: 0.2, 0.0: 0.8}),
    ("quickclaw", "owntempo", "quickattack", {0.1: 0.2, 0.0: 0.8}),
    ("quickclaw", "owntempo", "earthquake", {0.1: 0.2, 0.0: 0.8}),
    # Quick Draw runs first (handler priority -1 before -2); the claw rolls only when the
    # value it sees is still <= 0: draw 601/2000, claw 283/2000 (0.7 x 0.2).
    ("quickclaw", "quickdraw", "earthquake", {0.1: 0.3 + 0.7 * 0.2, 0.0: 0.7 * 0.8}),
    ("quickclaw", "quickdraw", "calmmind", {0.1: 0.2, 0.0: 0.8}),
    # The -0.1 constants run first and the later handlers replace them: 601/2000, 401/2000.
    ("laggingtail", "quickdraw", "earthquake", {0.1: 0.3, -0.1: 0.7}),
    ("quickclaw", "stall", "earthquake", {0.1: 0.2, -0.1: 0.8}),
    # The claw's own guard: 0/2000 on a status move under Mycelium Might, 401 otherwise.
    ("quickclaw", "myceliummight", "calmmind", {-0.1: 1.0}),
    ("quickclaw", "myceliummight", "earthquake", {0.1: 0.2, 0.0: 0.8}),
]


@pytest.mark.parametrize(("item", "ability", "move_id", "expected"), _SHOWDOWN_FRACTIONAL)
def test_fractional_priority_is_showdowns_event(
    reg: Regulation,
    team_a: list[TeamSet],
    item: str,
    ability: str,
    move_id: str,
    expected: dict[float, float],
) -> None:
    """Quick Claw's `priority <= 0` is the event's relay value, not the move's priority.

    Both engines skipped the claw on every status move and let Quick Draw, Stall and the
    -0.1 items shut it out, so a claw holder's Parting Shot never went first (IKA-145).
    """
    from pokeuraou.speed import fractional_priority
    from pokeuraou.view import battler

    mon = _synthetic_position(reg, team_a).sides[0].active_pokemon()[0]
    assert mon is not None
    mon.item = mon.base_item = item
    mon.ability = ability
    got = fractional_priority(reg, move_id, battler(reg, mon))
    assert sum(p for _v, p in got) == pytest.approx(1.0)
    assert len({v for v, _p in got}) == len(got)
    assert {round(v, 1): p for v, p in got} == pytest.approx(expected)


def test_a_claw_holders_status_move_splits_the_queue(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """A status move rolls the claw in the queue too, not only in `fractional_priority`."""
    from pokeuraou.speed import build_queue
    from pokeuraou.view import battler, field_state

    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].active_pokemon()[0]
    assert mon is not None
    mon.item = mon.base_item = "quickclaw"
    status = next(
        c
        for c in side_actions(reg, pos, 0)
        if isinstance(c.slots[0], MoveAction)
        and not c.slots[0].mega
        and reg.moves[c.slots[0].move_id].category == "Status"
    )
    actions = [status, _everyone_attacks(reg, pos, 1)]
    fighters = [
        [battler(reg, m) if m is not None and not m.fainted else None for m in s.active_pokemon()]
        for s in pos.sides
    ]
    queues = build_queue(reg, pos, actions, fighters, field_state(pos, reg))
    assert sorted(q[0].branch_probability for q in queues) == pytest.approx([0.2, 0.8])


def test_a_deterministic_budget_gives_one_branch(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    result = resolve_turn(reg, pos, both, budget=Budget.deterministic(8))
    assert len(result.branches) == 1
    assert result.branches[0].probability == pytest.approx(1.0)


def test_the_resolver_does_not_mutate_the_position(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Every branch works on its own copy, so a caller can evaluate a whole matrix."""
    pos = _synthetic_position(reg, team_a)
    before = pos.to_json()
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    resolve_turn(reg, pos, both, budget=Budget.fast())
    assert pos.to_json() == before


def test_a_hidden_spread_is_refused_not_guessed(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The resolver needs a concrete spread; inventing one would be a silent fabrication."""
    pos = _synthetic_position(reg, team_a)
    pos.sides[1].pokemon[0].sp = None
    pos.sides[1].pokemon[0].stats_override = None
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    with pytest.raises(PortRefused, match="hidden SP spread"):
        resolve_turn(reg, pos, both, budget=Budget.deterministic(0))


def test_a_switch_swaps_party_slots(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Showdown swaps positions so an active Pokemon sits at its own active index.

    The party index is what ``switch N`` counts, so getting this wrong makes every later
    choice refer to a different Pokemon.
    """
    pos = _synthetic_position(reg, team_a)
    side = pos.sides[0]
    bench = next(m for m in side.pokemon if not m.is_active)
    outgoing = side.pokemon[side.active[0]]
    switch = SideAction(
        slots=(
            SwitchAction(slot=0, party_index=bench.slot + 1, species=bench.species),
            PassAction(slot=1),
        )
    )
    opposing = side_actions(reg, pos, 1)[0]

    result = resolve_turn(reg, pos, [switch, opposing], budget=Budget.deterministic(0))
    after = result.branches[0].position.sides[0]
    assert after.active[0] == 0
    assert after.pokemon[0].species == bench.species
    assert after.pokemon[0].slot == 0
    # The outgoing Pokemon took the index the incoming one vacated.
    moved = next(m for m in after.pokemon if m.species == outgoing.species)
    assert moved.slot == bench.slot
    assert after.pokemon[moved.slot].species == outgoing.species


def test_protect_blocks_a_damaging_move(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    defender = pos.sides[1].pokemon[pos.sides[1].active[0]]
    defender.volatiles.append(Effect(id="protect"))
    hp_before = defender.hp

    attacker_side = side_actions(reg, pos, 0)
    attacking = next(
        a
        for a in attacker_side
        if isinstance(a.slots[0], MoveAction)
        and reg.moves[a.slots[0].move_id].category != "Status"
        and a.slots[0].target == 1
    )
    result = resolve_turn(
        reg, pos, [attacking, side_actions(reg, pos, 1)[0]], budget=Budget.deterministic(0)
    )
    # Every move hits under the pinned budget, so an unchanged HP is the block.
    assert len(result.branches) == 1
    after = result.branches[0].position.sides[1].pokemon[0]
    assert after.hp == hp_before


def test_protect_used_twice_is_a_branch(reg: Regulation, team_a: list[TeamSet]) -> None:
    """A repeated Protect succeeds one time in three, which is a real branch."""
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    protect_id = next(
        (m.id for m in mon.moves if m.id in ("protect", "detect")), None
    )
    if protect_id is None:
        pytest.skip("fixture slot has no Protect")
    mon.volatiles.append(Effect(id="stall", duration=2, counter=3))

    action = SideAction(
        slots=(_move_action(pos, 0, 0, protect_id, None), PassAction(slot=1))
    )
    result = resolve_turn(
        reg,
        pos,
        [action, side_actions(reg, pos, 1)[0]],
        budget=Budget(
            damage_rolls=1, enumerate_crit=False, enumerate_accuracy=False,
            enumerate_secondary=False, max_branches=16,
        ),
    )
    weights = sorted(b.probability for b in result.branches)
    assert len(result.branches) >= 2
    assert sum(weights) == pytest.approx(1.0)
    assert any(w == pytest.approx(1 / 3, abs=0.05) for w in weights)


def branch_target_fainted(branch) -> bool:  # noqa: ANN001
    """Whether side 1's first slot fainted in this branch.

    A fainted Pokemon also fails to act, so a test about flinching has to exclude it or it
    would count a knock-out as a flinch.
    """
    side = branch.position.sides[1]
    party = side.active[0]
    return party is None or side.pokemon[party].fainted


def _protect_both(pos: Position, side: int) -> SideAction:
    """Both of a side's actives use the move in their first slot.

    Paired with `_install_move(..., "protect")` this makes a turn in which nothing can
    faint, which is what a test about how long a status lasts needs -- otherwise the
    subject dies to an incidental attack and the test skips.
    """
    return SideAction(
        slots=tuple(
            _move_action(pos, side, slot, pos.sides[side].pokemon[party].moves[0].id, None)
            for slot, party in enumerate(pos.sides[side].active)
            if party is not None
        )
    )


def _install_move(pos: Position, side: int, slot: int, move_id: str) -> None:
    """Puts a move in the active Pokemon's first slot.

    The fixture teams come from usage priors, so whether one happens to carry Wide Guard
    is chance. Legality is ``side_actions``' job and is tested separately; here the point
    is what the move does once chosen.
    """
    mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
    mon.moves[0] = MoveSlot(id=move_id, pp=10, maxpp=10)


def test_wide_guard_raises_the_protect_counter(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Showdown's Wide Guard has `addVolatile('stall')` but no `stallingMove` flag.

    So it never fails itself, and it still makes the next Protect a one-in-three. Missing
    that scores a Wide Guard -> Protect line as certain, which it is not.
    """
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "wideguard")
    action = SideAction(
        slots=(_move_action(pos, 0, 0, "wideguard", None), PassAction(slot=1))
    )
    result = resolve_turn(
        reg, pos, [action, side_actions(reg, pos, 1)[0]], budget=Budget.deterministic(0)
    )
    after = result.branches[0].position.sides[0]
    user = after.pokemon[after.active[0]]
    stall = next((v for v in user.volatiles if v.id == "stall"), None)
    assert stall is not None, "Wide Guard did not raise the Protect counter"
    assert stall.counter == 3


def test_wide_guard_never_fails_itself(reg: Regulation, team_a: list[TeamSet]) -> None:
    """It raises the counter but does not read it, so a repeat is still a single branch."""
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "wideguard")
    user = pos.sides[0].pokemon[pos.sides[0].active[0]]
    user.volatiles.append(Effect(id="stall", duration=2, counter=9))
    action = SideAction(
        slots=(_move_action(pos, 0, 0, "wideguard", None), PassAction(slot=1))
    )
    result = resolve_turn(
        reg,
        pos,
        [action, side_actions(reg, pos, 1)[0]],
        budget=Budget(
            damage_rolls=1, enumerate_crit=False, enumerate_accuracy=False,
            enumerate_secondary=False, max_branches=16,
        ),
    )
    # The side condition itself lasts one turn and is gone by the end of it; what
    # persists, and what the next Protect reads, is the counter.
    for branch in result.branches:
        after = branch.position.sides[0]
        stall = next(v for v in after.pokemon[after.active[0]].volatiles if v.id == "stall")
        assert stall.counter == 27


def test_endure_consults_and_raises_the_counter(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Endure is a stalling move: it shares Protect's counter even though it blocks nothing.

    Which moves consult the counter comes from the regulation dump's ``stallingMove``
    flag, so this also checks that the flag is being read rather than a set of ids being
    kept in step by hand.
    """
    assert reg.moves["endure"].raw.get("stallingMove"), "regulation dump lost the flag"
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "endure")
    action = SideAction(
        slots=(_move_action(pos, 0, 0, "endure", None), PassAction(slot=1))
    )
    result = resolve_turn(
        reg, pos, [action, side_actions(reg, pos, 1)[0]], budget=Budget.deterministic(0)
    )
    after = result.branches[0].position.sides[0]
    user = after.pokemon[after.active[0]]
    # Showdown's Endure condition is `duration: 1`, so the volatile is gone by the end of
    # the turn it was used and only the counter persists. This test used to assert the
    # opposite, which was the bug: with no duration the volatile stayed for the rest of
    # the battle and the Pokemon survived every lethal hit at 1 HP.
    assert not any(v.id == "endure" for v in user.volatiles)
    stall = next((v for v in user.volatiles if v.id == "stall"), None)
    assert stall is not None and stall.counter == 3


def test_perish_song_counts_down_and_kills(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The one volatile whose *expiry* faints its holder.

    Weather's handler is skipped when its duration runs out; Perish Song's runs. Ticking
    it as an ordinary duration leaves a Pokemon alive that the game has killed, and every
    later turn of the battle is then scored against a position with an extra Pokemon.
    """
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[1].pokemon[pos.sides[1].active[0]]
    mon.volatiles.append(Effect(id="perishsong", duration=2))
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]

    result = resolve_turn(reg, pos, both, budget=Budget.deterministic(0))
    after = result.branches[0].position.sides[1]
    survivor = after.pokemon[after.active[0]]
    perish = next(v for v in survivor.volatiles if v.id == "perishsong")
    assert perish.duration == 1, "the counter is ticked once per turn, not twice"
    assert not survivor.fainted

    result = resolve_turn(
        reg, result.branches[0].position, both, budget=Budget.deterministic(0)
    )
    after = result.branches[0].position.sides[1]
    assert after.pokemon[after.active[0]].fainted


def test_perish_song_hits_both_sides(reg: Regulation, team_a: list[TeamSet]) -> None:
    """`onHitField` gives the count to every active Pokemon, the user's own side included."""
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "perishsong")
    action = SideAction(
        slots=(_move_action(pos, 0, 0, "perishsong", None), PassAction(slot=1))
    )
    result = resolve_turn(
        reg, pos, [action, side_actions(reg, pos, 1)[0]], budget=Budget.deterministic(0)
    )
    after = result.branches[0].position
    for side in (0, 1):
        for slot in after.sides[side].active:
            mon = after.sides[side].pokemon[slot]
            if mon.fainted or mon.ability == "soundproof":
                continue
            perish = next((v for v in mon.volatiles if v.id == "perishsong"), None)
            assert perish is not None, f"p{side + 1} slot {slot} escaped Perish Song"
            # Applied with duration 4 and ticked once at the end of the same turn.
            assert perish.duration == 3


def test_collapse_groups_outcomes_readably(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    result = resolve_turn(reg, pos, both, budget=Budget.fast())

    def who_fainted(p: Position) -> tuple[str, ...]:
        return tuple(
            f"p{i + 1}.{m.slot}"
            for i, side in enumerate(p.sides)
            for m in side.pokemon
            if m.fainted
        )

    grouped = result.collapse(who_fainted)
    assert sum(grouped.values()) == pytest.approx(1.0)
    assert len(grouped) <= len(result.branches)


def test_expected_value_is_a_probability_weighted_mean(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    pos = _synthetic_position(reg, team_a)
    both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    result = resolve_turn(reg, pos, both, budget=Budget.fast())

    def our_hp_fraction(p: Position) -> float:
        mons = [m for m in p.sides[0].pokemon]
        return sum(m.hp / max(m.maxhp, 1) for m in mons) / len(mons)

    got = result.expected(our_hp_fraction)
    values = [our_hp_fraction(b.position) for b in result.branches]
    assert min(values) - 1e-9 <= got <= max(values) + 1e-9


# ---------------------------------------------------------------------------
# Differential test: the thresholds. The port's run against Showdown is
# tests/test_diff_turn_port.py; Python's went with IKA-210.
# ---------------------------------------------------------------------------


#: A single seed compares about 50 turns, so one divergence moves the rate by two whole
#: points and a threshold set at the aggregate's value would be testing which teams the
#: usage sampler happened to draw. The *silent* rate is still held tight per seed, because
#: that is the load-bearing number and a new silent divergence is a regression whatever
#: the sample; the total rate is held tight only on the large sample below, where the
#: count of *declared* gaps averages out.
MAX_TOTAL_DIVERGENCE_PER_SEED = 0.14


def test_the_matrix_budget_gives_the_lp_a_zero_sum_game(reg: Regulation) -> None:
    """`M[i][j] + M[j][i] = 1` in a mirror, or the two sides solve different games.

    The LP is handed one matrix and reads a maximin off it for the row player and a
    minimax for the column player. That is only coherent if the matrix is zero-sum, and in
    a mirror -- same four, same spreads, same action list -- the identity above is forced.

    `Budget.matrix()` used to resolve Speed ties one way instead of enumerating them, which
    broke it: measured at up to 0.042 per cell, mean 0.011, biased +0.005 toward side 0,
    because the side the canonical order puts first plans as though it wins every tie while
    the other plans as though it loses every one. Compounded over a game that showed up as
    a 59.1% side-0 win rate in a mirror, where symmetry forces 50%.

    Every other collapse in that budget (crit, accuracy, secondary, status) was measured to
    leave the identity exact, so ties were the only one that had to be paid for -- 1.06x on
    real matchups, where ties occur at 5.8% of decisions.
    """
    import numpy as np

    from pokeuraou.narrow import narrow
    from pokeuraou.payoff import HP_SHARE
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    roster = load_roster("rizabanadohido")
    if roster.reg.meta.format_id != reg.meta.format_id:
        reg = roster.reg
    four = roster.sets[: reg.meta.picked_team_size]
    pos = position_from_sets(reg, four, four)

    ours = narrow(reg, pos, 0, limit=5).actions
    theirs = narrow(reg, pos, 1, limit=5).actions
    assert [a.to_choice() for a in ours] == [b.to_choice() for b in theirs], (
        "a mirror has to offer both sides the same actions for this check to mean anything"
    )

    matrix = np.zeros((len(ours), len(theirs)))
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            matrix[i, j] = resolve_turn(reg, pos, [a, b], budget=Budget.matrix()).expected(
                HP_SHARE
            )
    error = np.abs(matrix + matrix.T - 1.0)
    assert error.max() < 1e-9, (
        f"the search matrix is not zero-sum: max |M[i][j]+M[j][i]-1| = {error.max():.4f}"
    )
    # Same action on both sides of a mirror is an even position, exactly.
    assert np.abs(np.diag(matrix) - 0.5).max() < 1e-9


def test_a_volatile_that_should_expire_has_a_duration(reg: Regulation, team_a: list[TeamSet]) -> None:
    """A volatile added with no duration is kept for the rest of the battle.

    Showdown keeps most durations on the condition in conditions.ts, and reading only the
    move's own `condition` once made four effects permanent in Python's resolver --
    partiallytrapped (Infestation and the other binding moves), Endure, Magnet Rise and
    Electrify. The dump carries every one under `durations`; the port reads it there
    (`moves.rs`, `effect_duration`). Asked of the port as a turn (IKA-210; the audit over the
    whole dump was of Python's `_duration`): Tailwind is Showdown's 4, one spent at the end
    of the turn it was used, and Infestation's bind has a duration at all. Endure's single
    turn is `test_endure_survives_a_lethal_move_from_any_hp` and the one after it.
    """
    assert reg.moves["tailwind"].raw["durations"]["tailwind"]["duration"] == 4
    pos = _synthetic_position(reg, team_a)
    _set_move(pos, 0, 0, 0, "infestation")
    _set_move(pos, 0, 1, 0, "tailwind")
    _set_move(pos, 1, 0, 0, "swordsdance")
    _set_move(pos, 1, 1, 0, "protect")
    after = _one(
        reg, pos,
        _act(pos, 0, ("infestation", 1), ("tailwind", None)),
        _act(pos, 1, ("swordsdance", None), ("protect", None)),
    )
    tailwind = after.sides[0].side_condition("tailwind")
    assert tailwind is not None and tailwind.duration == 3, tailwind
    bound = _slot(after, 1, 0).volatile("partiallytrapped")
    assert bound is not None and bound.duration is not None and bound.duration >= 4, bound


def test_a_bind_expires_and_blocks_switching_until_it_does(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """End to end: the volatile counts down, and the target cannot switch while it is on."""
    from pokeuraou.actions import SwitchAction

    pos = _synthetic_position(reg, team_a)
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.volatiles.append(Effect(id="partiallytrapped", duration=5, source_slot="00"))

    # Trapped: no switch is offered for that slot.
    offered = [
        s
        for action in side_actions(reg, pos, 1)
        for s in action.slots
        if isinstance(s, SwitchAction) and s.slot == 0
    ]
    assert not offered, "a bound Pokemon must not be offered a switch"

    # The duration has to fall by exactly one per turn. Asserting the decrement rather
    # than only the eventual removal is what makes this test independent of how long the
    # trapper survives -- and a `None` duration fails it on the first step, which is
    # precisely the bug. Removal is asserted too, when the battle lasts long enough.
    seen: list[int | None] = [5]
    for _ in range(6):
        both = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
        result = resolve_turn(reg, pos, both, budget=Budget.matrix())
        pos = max(result.branches, key=lambda b: b.probability).position
        volatile = pos.sides[1].pokemon[pos.sides[1].active[0]].volatile("partiallytrapped")
        seen.append(volatile.duration if volatile else None)
        trapper_gone = pos.sides[0].pokemon[0].fainted
        if volatile is None:
            # A bind also ends when its source leaves or faints, which is correct Showdown
            # behaviour and a different event from expiry. Only the second one says
            # anything about the duration.
            released_early = trapper_gone
            break
        if trapper_gone:
            released_early = True
            break
    else:
        released_early = False

    counted = [d for d in seen if d is not None]
    assert len(counted) >= 2, f"never got two turns of bind to compare: {seen}"
    steps = [a - b for a, b in zip(counted, counted[1:], strict=False)]
    # A `None` duration never falls, so this is the assertion that catches the bug.
    assert all(step == 1 for step in steps), f"duration did not fall by one a turn: {seen}"
    if seen[-1] is None and not released_early:
        assert counted[-1] == 1, f"expired from the wrong duration: {seen}"


# ---------------------------------------------------------------------------
# Mid-turn replacement (self-switching moves)
# ---------------------------------------------------------------------------


#: A Prankster user and a Dark type, so the immunity is exercised by name rather than by
#: whatever the usage sampler happens to draw.
_PRANKSTER_TEAM = [
    TeamSet(
        "Whimsicott", "Prankster", "Timid",
        ['charm', 'tailwind', 'moonblast', 'protect'],
        {'spe': 32},
    ),
    TeamSet(
        "Incineroar", "Intimidate", "Brave",
        ['partingshot', 'throatchop', 'flareblitz', 'fakeout'],
        {'hp': 32},
    ),
    TeamSet(
        "Garchomp", "Rough Skin", "Jolly",
        ['earthquake', 'dragonclaw', 'protect', 'rockslide'],
        {'spe': 32},
    ),
    TeamSet(
        "Venusaur", "Chlorophyll", "Modest",
        ['sludgebomb', 'gigadrain', 'protect', 'sleeppowder'],
        {'spa': 32},
    ),
]

_DARK_TEAM = [
    TeamSet(
        "Kingambit", "Defiant", "Adamant",
        ['ironhead', 'suckerpunch', 'swordsdance', 'protect'],
        {'atk': 32},
    ),
    TeamSet(
        "Tyranitar", "Sand Stream", "Jolly",
        ['rockslide', 'crunch', 'protect', 'earthquake'],
        {'spe': 32},
    ),
    TeamSet(
        "Toxapex", "Regenerator", "Relaxed",
        ['infestation', 'toxic', 'wideguard', 'protect'],
        {'hp': 32},
    ),
    TeamSet(
        "Charizard", "Blaze", "Timid",
        ['heatwave', 'airslash', 'protect', 'flamethrower'],
        {'spe': 32},
    ),
]


def test_a_self_switch_suspends_the_turn_instead_of_finishing_it(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The resolver must hand the turn back at the interrupt, not run past it.

    Showdown checks `switchFlag` at the end of `runAction`, which runs after every action,
    so a self-switching move stops the turn there. Finishing the turn instead left the
    Pokemon that used the move standing in the slot for the rest of it -- the opposite of
    what the move does.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)

    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))

    assert result.suspended, "a self-switching move has to suspend the turn"
    assert result.total_probability == pytest.approx(1.0), (
        "branches and suspensions together are the whole turn"
    )
    # The turn number has not advanced and the user is still in its slot: the position is
    # mid-turn, which is exactly what the chooser is looking at.
    pause = result.suspended[0]
    assert pause.position.turn == pos.turn
    owed = self_switches_needed(pause.position)
    assert owed[0][0] and not any(owed[1]), f"only the mover owes a replacement: {owed}"


def test_summarising_a_suspended_turn_is_refused(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """A partial expectation is worse than an error, because it looks like a number.

    Averaging over `branches` while a suspension holds part of the mass gives a value short
    by that fraction and perfectly ordinary-looking. Every caller has to answer the
    replacement first.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    # Python's: the fold under test is the search's `turn_expectation` (IKA-209 moves it).
    result = python_resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended

    with pytest.raises(ValueError, match="suspended"):
        result.expected(lambda _p: 0.0)

    # ...and the fold does produce a number, once the choice is part of it.
    value, _flags = turn_expectation(reg, result, lambda _p: 0.5)
    assert value == pytest.approx(0.5)


def test_the_mid_turn_replacement_is_a_choice_and_not_an_average(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The interrupted side gets its best option, not the mean of all of them.

    This is the whole reason `TurnLeaves` exists. Which Pokemon comes in is a decision, so
    averaging over the bench would price a Parting Shot as if the player brought in
    something at random. Zero-sum means side 0 maximises and side 1 minimises, and the
    direction is asserted both ways because getting the sign backwards would be invisible
    in aggregate.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    # Python's: the fold under test is the search's `turn_leaves` (IKA-209 moves it).
    result = python_resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended

    chooser, alternatives = python_resume_alternatives(reg, result.suspended[0])
    assert chooser == 0
    assert len(alternatives) >= 2, "a bench of two is the point of the test"

    # Score each candidate by which species ended up in the slot, so the values are
    # distinct and the best one is known independently of the value function.
    plan = turn_leaves(reg, result)
    species = [
        (p.sides[0].pokemon[p.sides[0].active[0]].species if p.sides[0].active[0] is not None else "")
        for p in plan.positions
    ]
    scores = {name: float(i + 1) for i, name in enumerate(sorted(set(species)))}
    values = [scores[name] for name in species]

    assert plan.value(values) == pytest.approx(max(scores.values())), (
        "side 0 chooses, so the fold must take the option it likes most"
    )
    # Flip the fold's owner and the same leaves must produce the worst value instead.
    flipped = replace(
        plan,
        root=Average(
            parts=[
                (w, BestOf(chooser=1, options=node.options) if isinstance(node, BestOf) else node)
                for w, node in plan.root.parts
            ]
        ),
    )
    assert flipped.value(values) == pytest.approx(min(scores.values()))


def test_a_chunked_fill_scores_the_same_node(
    reg: Regulation, team_a: list[TeamSet], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Releasing a node's leaves part way through must not change what it is worth.

    `batched_payoffs` held every leaf of every cell alive until the whole node was
    resolved, and a 24x24 depth-1 node measured that way took this machine to 0.3 GB free
    (IKA-27). `LEAF_CHUNK` bounds the list instead. The leaves it scores are the same
    leaves and the folds are the same folds, so for a per-position objective the matrix has
    to come back identical to the bit -- not close, identical, because a payoff that moves
    is an equilibrium that moves.

    The cell list includes a U-turn, so at least one cell stops for a mid-turn replacement
    and the fold path is the one crossing a chunk boundary rather than a plain average.

    A learned leaf is the case this cannot claim: its forward pass is float32 matrix
    arithmetic whose last places depend on the number of rows in the batch, and chunking
    changes that number. That difference is measured where it belongs, against the value
    function, and not asserted here.
    """
    from pokeuraou import resolve as resolve_module
    from pokeuraou.payoff import HP_SHARE

    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
    uturn = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    ours = [uturn, *side_actions(reg, pos, 0)[:3]]
    theirs = side_actions(reg, pos, 1)[:4]
    budget = Budget.matrix()
    # Python's own fill (`batched_payoffs`, `LEAF_CHUNK`) is what is under test; it goes with
    # the resolver (IKA-212).
    assert python_resolve_turn(reg, pos, [uturn, theirs[0]], budget=budget).suspended, (
        "the node has to contain a suspended turn for the fold to be under test"
    )

    calls: list[int] = []

    def evaluate(positions: list[Position]) -> np.ndarray:
        calls.append(len(positions))
        return HP_SHARE.batch(positions)

    monkeypatch.setattr(resolve_module, "LEAF_CHUNK", 0)
    whole, whole_notes, whole_exact = resolve_module.batched_payoffs(
        reg, pos, ours, theirs, [evaluate], budget=budget
    )
    at_once = calls.copy()
    calls.clear()

    monkeypatch.setattr(resolve_module, "LEAF_CHUNK", 64)
    chunked, chunked_notes, chunked_exact = resolve_module.batched_payoffs(
        reg, pos, ours, theirs, [evaluate], budget=budget
    )

    assert len(at_once) == 1, "unchunked is one call holding the whole node"
    assert len(calls) > 1, "chunked has to have actually split this node"
    assert sum(calls) == at_once[0], "the same leaves were scored, not fewer"
    assert max(calls) < at_once[0], "and never all of them at the same time"
    assert np.array_equal(chunked[0], whole[0]), "the payoff moved when the leaves were released"
    assert np.array_equal(chunked_exact, whole_exact)
    assert chunked_notes == whole_notes


def test_a_chunk_boundary_never_falls_inside_a_cell(
    reg: Regulation, team_a: list[TeamSet], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell's leaves are scored by one call, whatever the chunk size.

    The fold of a suspended turn indexes into the array of values it was scored with, so a
    cell split across two calls would be folded from half an array. `LEAF_CHUNK` is a
    floor checked between cells rather than a ceiling enforced inside one, and that is the
    reason: with the chunk set to 1 every cell flushes on its own, and each call is exactly
    one cell's leaves.
    """
    from pokeuraou import resolve as resolve_module
    from pokeuraou.payoff import HP_SHARE

    pos = _synthetic_position(reg, team_a)
    ours = side_actions(reg, pos, 0)[:3]
    theirs = side_actions(reg, pos, 1)[:3]
    budget = Budget.matrix()
    def leaves_of(a: SideAction, b: SideAction) -> int:
        # Python's own fill is under test, as above (IKA-212).
        result = python_resolve_turn(reg, pos, [a, b], budget=budget)
        if result.suspended:
            return len(turn_leaves(reg, result).positions)
        return len(result.branches)

    per_cell = [leaves_of(a, b) for a in ours for b in theirs]

    calls: list[int] = []

    def evaluate(positions: list[Position]) -> np.ndarray:
        calls.append(len(positions))
        return HP_SHARE.batch(positions)

    monkeypatch.setattr(resolve_module, "LEAF_CHUNK", 1)
    resolve_module.batched_payoffs(reg, pos, ours, theirs, [evaluate], budget=budget)

    assert [n for n in calls if n] == [n for n in per_cell if n]


@pytest.mark.oracle
def test_the_replacement_takes_the_residual_and_not_the_departing_pokemon(
    reg: Regulation, oracle: Oracle
) -> None:
    """Against Showdown: the interrupt is in front of the residual phase.

    This is the case that made deferring the choice to the post-turn phase wrong even when
    the self-switching move resolved *last*. Incineroar is deliberately the slowest thing
    on the field, so nothing but the residual phase is queued behind its Parting Shot --
    and Showdown still stops, brings the replacement in, and only then applies the
    sandstorm. Deferring meant Incineroar took weather it never sees in the real game.
    """
    handle = oracle.create(
        FORMAT_ID, _PRANKSTER_TEAM, _DARK_TEAM, policy=RandomnessPolicy(damage_roll=8)
    )
    handle.step(["team 2,1,3,4", "team 2,1,3,4"])
    before = Position.from_json(handle.position)

    # p1a Incineroar uses Parting Shot on p2b Kingambit (Dark, but Incineroar is not a
    # Prankster user, so it lands); everything else attacks so nothing blocks it.
    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="partingshot", target=2),
            MoveAction(slot=1, move_index=3, move_id="moonblast", target=1),
        )
    )
    theirs = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="rockslide", target=None),
            MoveAction(slot=1, move_index=1, move_id="ironhead", target=1),
        )
    )
    handle.step([ours.to_choice(), theirs.to_choice()])
    assert not handle.choice_errors, handle.choice_errors

    result = resolve_turn(reg, before, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended, "Parting Shot must suspend the turn"
    assert any(r and r.get("forceSwitch") for r in handle.requests), (
        "Showdown must be asking for a replacement at the same point"
    )
    assert not any(line.startswith("|upkeep") for line in handle.log), (
        "the residual phase must still be ahead of us, or the premise of the fix is wrong"
    )

    # Answer both the same way, then compare the finished turn.
    owed = self_switches_needed(result.suspended[0].position)
    options = switch_actions_after_faint(reg, result.suspended[0].position, 0, list(owed[0]))
    pick = options[0]
    handle.step([pick.to_choice(), None])
    assert not handle.choice_errors, handle.choice_errors

    passes = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    finished = resume_turn(reg, result.suspended[0], [pick, passes])
    assert len(finished.branches) == 1
    ours_after = finished.branches[0].position
    theirs_after = Position.from_json(handle.position)

    for side in range(2):
        for mon_index in range(4):
            mine = ours_after.sides[side].pokemon[mon_index]
            yours = theirs_after.sides[side].pokemon[mon_index]
            assert (mine.species, mine.hp) == (yours.species, yours.hp), (
                f"p{side + 1} slot {mon_index}: ours {mine.species} {mine.hp} "
                f"vs showdown {yours.species} {yours.hp}"
            )


def test_parting_shot_stays_in_when_its_drops_do_nothing(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`if (!success && !target.hasAbility('mirrorarmor')) delete move.selfSwitch`.

    A target already at the floor keeps the user on the field. We used to suspend the turn
    for a replacement Showdown never asks for, which the differential test sees as a
    disagreement about whether the turn stopped at all.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="partingshot", pp=20, maxpp=20)
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.boosts = {"atk": -6, "spa": -6}

    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="partingshot", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert not result.suspended, (
        "Parting Shot that lowered nothing must not switch its user out"
    )


def test_defiant_answers_every_stat_a_foe_lowers(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`runEvent('AfterEachBoost')` is inside Showdown's per-stat loop, not after it.

    Parting Shot drops Attack and Special Attack, so Defiant answers twice. The raise also
    lands *between* the two drops, so the arithmetic from neutral is
    ``-1 -> +2 (defiant) -> (spa -1) -> +2 (defiant) = +3``; firing once per call gave +1.
    Measured against Showdown from +1 Attack it was ours {'atk': 2} vs theirs {'atk': 4},
    and Incineroar's Parting Shot into Kingambit is an ordinary line in this format.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="partingshot", pp=20, maxpp=20)
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.ability = "defiant"

    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="partingshot", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = (result.suspended[0].position if result.suspended else result.branches[0].position)
    hit = after.sides[1].pokemon[after.sides[1].active[0]]
    assert hit.boosts.get("atk") == 3, (
        f"two drops means two Defiant triggers, not one: {hit.boosts}"
    )
    assert hit.boosts.get("spa") == -1


def test_a_prankster_status_move_does_not_reach_a_dark_type(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Since gen 7 a Prankster-boosted status move simply fails against a foe Dark type.

        gen >= 7 && move.pranksterBoosted && pokemon.hasAbility('prankster') &&
            !targets[i].isAlly(pokemon) && !this.dex.getImmunity('prankster', target)

    Whimsicott and Grimmsnarl are the format's Prankster users and Incineroar is in 41% of
    its teams, so this decides real turns. The immunity is read from the regulation dump
    rather than written here, because a hand-kept copy of what the simulator declares is
    what made four effects permanent.
    """
    assert reg.immune_to_effect("prankster", ("Dark",))
    assert not reg.immune_to_effect("prankster", ("Fairy", "Grass"))

    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.ability = "prankster"
    mover.moves[0] = MoveSlot(id="charm", pp=20, maxpp=20)
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.types = ("Dark",)
    target.boosts = {}

    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="charm", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = result.branches[0].position
    hit = after.sides[1].pokemon[after.sides[1].active[0]]
    assert not hit.boosts, f"Charm must not reach a Dark type from Prankster: {hit.boosts}"

    # Without Prankster the same Charm lands, so the test is about the ability and not
    # about Charm being broken.
    mover.ability = "chlorophyll"
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = result.branches[0].position
    hit = after.sides[1].pokemon[after.sides[1].active[0]]
    assert hit.boosts.get("atk") == -2, f"Charm should have landed: {hit.boosts}"


def test_throat_chop_locks_sound_moves(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Two turns without sound moves, which is how Throat Chop answers Parting Shot.

    Showdown blocks it twice -- `onDisableMove` keeps a sound move out of the request and
    `onBeforeMove` refuses it if used anyway -- so both are checked. The volatile is added
    from a 100%-chance `secondary.onHit`, so nothing declarative in the dump pointed at it
    and we applied neither.
    """
    pos = _synthetic_position(reg, team_a)
    attacker = pos.sides[0].pokemon[pos.sides[0].active[0]]
    attacker.moves[0] = MoveSlot(id="throatchop", pp=15, maxpp=15)
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.moves[0] = MoveSlot(id="partingshot", pp=20, maxpp=20)
    assert "sound" in reg.moves["partingshot"].flags

    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="throatchop", target=1),
            PassAction(slot=1),
        )
    )
    theirs = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = max(result.branches, key=lambda b: b.probability).position
    hit = after.sides[1].pokemon[after.sides[1].active[0]]
    if hit.fainted:
        pytest.skip("the target did not survive the hit, so there is no lock to check")

    lock = hit.volatile("throatchop")
    assert lock is not None, "Throat Chop has to apply its own volatile"
    # Declared as 2 and decremented by the residual phase of the same turn, so one turn of
    # lock is left. What matters is that it is a number at all: a volatile added without a
    # duration never expires, and Throat Chop's is only reachable through the move's own
    # condition, which the dump did not carry until now.
    assert lock.duration == 1, f"declared 2, one residual spent, got {lock.duration}"

    # And the legal action set no longer offers the sound move.
    offered = {
        s.move_id
        for action in side_actions(reg, after, 1)
        for s in action.slots
        if isinstance(s, MoveAction) and s.slot == 0
    }
    assert "partingshot" not in offered, (
        f"a throat-chopped Pokemon must not be offered a sound move: {sorted(offered)}"
    )


# ---------------------------------------------------------------------------
# Status odds, read from the simulator rather than written down
# ---------------------------------------------------------------------------


#: A Prankster paralyser, a sleeper and a Poison/Water target that can take both.
_INFLICTOR_TEAM = [
    TeamSet(
        "Grimmsnarl",
        "Prankster",
        "Careful",
        ["thunderwave", "protect", "lightscreen", "reflect"],
        {"hp": 32},
        item="lightclay",
    ),
    TeamSet(
        "Venusaur",
        "Chlorophyll",
        "Modest",
        ["sleeppowder", "protect", "gigadrain", "sludgebomb"],
        {"spa": 32},
        item="lifeorb",
    ),
    TeamSet(
        "Kingambit",
        "Defiant",
        "Adamant",
        ["protect", "ironhead", "suckerpunch", "swordsdance"],
        {"atk": 32},
        item="focussash",
    ),
    TeamSet(
        "Tyranitar",
        "Sand Stream",
        "Jolly",
        ["protect", "crunch", "rockslide", "earthquake"],
        {"spe": 32},
        item="chopleberry",
    ),
]

_VICTIM_TEAM = [
    TeamSet(
        "Toxapex",
        "Regenerator",
        "Relaxed",
        ["protect", "wideguard", "toxic", "infestation"],
        {"hp": 32},
        item="leftovers",
    ),
    TeamSet(
        "Venusaur",
        "Chlorophyll",
        "Modest",
        ["protect", "gigadrain", "sludgebomb", "sleeppowder"],
        {"spa": 32},
        item="lifeorb",
    ),
    TeamSet(
        "Garchomp",
        "Rough Skin",
        "Jolly",
        ["protect", "dragonclaw", "earthquake", "rockslide"],
        {"spe": 32},
        item="choicescarf",
    ),
    TeamSet(
        "Charizard",
        "Blaze",
        "Timid",
        ["protect", "airslash", "heatwave", "flamethrower"],
        {"spe": 32},
        item="charcoal",
    ),
]


def _chance_denominators(rolls: list[dict[str, object]], numerator: int = 1) -> set[int]:
    return {
        int(r["denominator"])
        for r in rolls
        if r["kind"] == "chance" and int(r["numerator"]) == numerator
    }


@pytest.mark.oracle
def test_the_paralysis_chance_is_the_one_the_simulator_rolls(oracle: Oracle) -> None:
    """`FULL_PARALYSIS_CHANCE` against the odds Showdown asks the policy for.

    The champions mod overrides the base game's 1/4 with `randomChance(1, 8)`, and we
    carried 0.25 -- twice the real rate, multiplied through every branch of every turn a
    paralysed Pokemon was on the field.

    It cannot be tested by counting outcomes, because the policy pins every roll's answer;
    that is what the policy is for. The *arguments* are the evidence, and the oracle records
    them, so this reads the denominator instead of trusting a number in a comment. It also
    fails if a future regulation changes the rate, which a hand-written constant cannot.
    """
    handle = oracle.create(
        FORMAT_ID, _VICTIM_TEAM, _INFLICTOR_TEAM, policy=RandomnessPolicy(damage_roll=8)
    )
    handle.step(["team 1,2,3,4", "team 1,2,3,4"])
    # Thunder Wave from Prankster Grimmsnarl onto Toxapex, which is neither Ground nor
    # Electric nor Dark, so it lands. The victim must not Protect on this turn.
    handle.step(["move 3 1, move 1", "move 1 1, move 2"])
    assert not handle.choice_errors, handle.choice_errors
    assert any(
        line.startswith("|-status|p1a") and line.endswith("par") for line in handle.log
    ), "the setup did not paralyse anything, so there is nothing to measure"

    # Now it tries to move, which is when `par.onBeforeMove` rolls.
    handle.step(["move 1, move 1", "move 2, move 2"])
    assert not handle.choice_errors, handle.choice_errors
    denominators = _chance_denominators(handle.rolls)
    assert denominators, f"no 1-in-N roll recorded at all: {handle.rolls}"
    # The champions mod's `par.onBeforeMove` is `randomChance(1, 8)`; the port's weight for
    # it is held in tests/test_confusion_duration.py (IKA-210: was Python's constant).
    expected = 8
    assert expected in denominators, (
        f"the full paralysis is 1/{expected} but the simulator rolled "
        f"1 in {sorted(denominators)} on a paralysed Pokemon's turn"
    )
    handle.close()


@pytest.mark.oracle
def test_the_sleep_counter_is_the_distribution_the_simulator_samples(oracle: Oracle) -> None:
    """Champions sleep is `sample([2, 3, 3])`, and we pinned the 1-in-3 outcome.

    "The first action is always asleep, the second wakes one time in three, the third always
    acts" -- so the counter is 2 a third of the time and 3 the rest, and the modal outcome
    is two turns of sleep rather than one. The base game rolls `random(2, 5)` instead, which
    is where our pinned 2 came from.

    A `sample` is the one distribution no denominator reveals, which is why the oracle
    records its values.
    """
    handle = oracle.create(
        FORMAT_ID, _VICTIM_TEAM, _INFLICTOR_TEAM, policy=RandomnessPolicy(damage_roll=8)
    )
    handle.step(["team 1,2,3,4", "team 1,2,3,4"])
    # Sleep Powder from p2b onto p1a.
    handle.step(["move 3 1, move 1", "move 2, move 1 1"])
    assert not handle.choice_errors, handle.choice_errors

    samples = [r["values"] for r in handle.rolls if r["kind"] == "sample"]
    assert samples, f"no sample recorded on the turn sleep landed: {handle.rolls}"
    counters = sorted(int(v) for v in samples[0])
    assert counters, samples
    # `sample([2, 3, 3])`: modal 3, and the policy's pinned answer (the first element) 2.
    # The port's turns are held to the modal one in
    # `test_a_sleeping_pokemon_wakes_and_the_counter_is_not_the_lucky_one` (IKA-210).
    modal = max(set(counters), key=counters.count)
    assert (modal, counters[0]) == (3, 2), counters
    handle.close()


def test_speed_boost_raises_speed_every_turn_but_not_on_arrival(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`onResidual(pokemon) { if (pokemon.activeTurns) this.boost({spe: 1}); }`.

    The ability was not implemented at all, so a Pokemon the search believed had a fixed
    Speed was in fact getting faster every turn -- and turn order is the part of a doubles
    turn that decides the rest of it.

    `activeTurns` is zero on the turn its holder switches in, which is the half of the rule
    that is easy to miss: Speed Boost does not fire immediately.
    """
    pos = _synthetic_position(reg, team_a)
    for side in (0, 1):
        for slot in (0, 1):
            _install_move(pos, side, slot, "protect")
    holder = pos.sides[0].pokemon[pos.sides[0].active[0]]
    holder.ability = "speedboost"

    # Fresh off a switch: no raise.
    holder.newly_switched = True
    result = resolve_turn(
        reg, pos, [_protect_both(pos, 0), _protect_both(pos, 1)], budget=Budget.matrix()
    )
    after = max(result.branches, key=lambda b: b.probability).position
    assert after.sides[0].pokemon[after.sides[0].active[0]].boosts.get("spe", 0) == 0, (
        "Speed Boost must not fire on the turn its holder arrives"
    )

    # Settled in: one stage a turn, and it keeps going.
    pos = after
    for expected in (1, 2, 3):
        result = resolve_turn(
            reg, pos, [_protect_both(pos, 0), _protect_both(pos, 1)], budget=Budget.matrix()
        )
        pos = max(result.branches, key=lambda b: b.probability).position
        mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
        assert not mon.fainted, "nothing should faint in a turn of Protects"
        assert mon.boosts.get("spe", 0) == expected, (
            f"expected +{expected} Speed after {expected} turns, got {mon.boosts}"
        )


def test_a_secondary_effect_is_a_branch_with_the_right_weight(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    '''A sub-100% secondary never happened, in any budget.

    `enumerate_secondary` read like a switch between branching and collapsing and was
    neither: with the flag on it reported the secondary, with it off it said nothing, and
    both discarded the effect. So Sludge Bomb never poisoned and Rock Slide never flinched.
    '''
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "sludgebomb")
    _install_move(pos, 0, 1, "protect")
    # Not Protect on the target: Protect blocks the move whose secondary is under test,
    # which is how the first version of this test managed to assert nothing.
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    secondary = reg.moves["sludgebomb"].raw.get("secondaries") or []
    chance = float(secondary[0]["chance"]) / 100.0 if secondary else 0.0
    assert 0.0 < chance < 1.0, f"the fixture move needs a real secondary: {secondary}"

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "sludgebomb", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)  # both use the move in slot 1, i.e. Swords Dance
    budget = replace(Budget.matrix(), enumerate_secondary=True, max_branches=64)
    result = resolve_turn(reg, pos, [ours, theirs], budget=budget)
    assert result.total_probability == pytest.approx(1.0)

    poisoned = sum(
        b.probability
        for b in result.branches
        if b.position.sides[1].pokemon[b.position.sides[1].active[0]].status == "psn"
    )
    assert poisoned == pytest.approx(chance, abs=1e-9), (
        f"the secondary should carry its own {chance:.0%}, got {poisoned:.3f} over "
        f"{len(result.branches)} branches"
    )

    # And with the knob off it is collapsed to "did not happen" *and* said out loud. That
    # is what the cheap budgets buy; the search's own budget now enumerates them.
    collapsed = resolve_turn(
        reg, pos, [ours, theirs], budget=replace(Budget.matrix(), enumerate_secondary=False)
    )
    assert not any(
        b.position.sides[1].pokemon[b.position.sides[1].active[0]].status == "psn"
        for b in collapsed.branches
    )
    assert any("not branched" in f for f in collapsed.unmodelled), collapsed.unmodelled


def test_a_flinch_actually_costs_the_target_its_action(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    '''Flinch is the secondary that matters: the target loses its turn, not some HP.

    Iron Head is on 36% of the tournament field and Rock Slide on 62%, and neither could
    flinch anything. Asserting that the flinched branch *did not resolve the target's move*
    is the point -- applying the volatile and then ignoring it would pass a weaker test.
    '''
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "ironhead")
    _install_move(pos, 0, 1, "protect")
    # Swords Dance rather than Protect: Protect would block Iron Head outright, and its
    # `atk +1` event is the witness for whether the target got to act.
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    # The flincher has to move first, or a flinch cannot land.
    pos.sides[0].pokemon[pos.sides[0].active[0]].boosts = {"spe": 6}
    pos.sides[1].pokemon[pos.sides[1].active[0]].boosts = {"spe": -6}

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "ironhead", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)
    budget = replace(Budget.matrix(), enumerate_secondary=True, max_branches=64)
    result = resolve_turn(reg, pos, [ours, theirs], budget=budget)

    # The witness is the consequence, not the volatile: the volatile is single-turn and is
    # gone by the time the branch is handed back, and asserting on it would pass even if
    # `_can_act` ignored it entirely.
    def acted(branch) -> bool:  # noqa: ANN001
        target = branch.position.sides[1].pokemon[branch.position.sides[1].active[0]]
        return target.boosts.get("atk", 0) > 0

    flinched = [b for b in result.branches if not acted(b) and not
                branch_target_fainted(b)]
    got_to_move = [b for b in result.branches if acted(b)]
    assert flinched, "no branch has the target losing its action"
    assert got_to_move, "and it must sometimes get to move, or this is not a branch"

    chance = float((reg.moves["ironhead"].raw["secondaries"] or [{}])[0]["chance"]) / 100.0
    total = sum(b.probability for b in flinched)
    assert total == pytest.approx(chance, abs=1e-9), (
        f"Iron Head flinches {chance:.0%} of the time, got {total:.3f}"
    )


def test_aegislash_takes_the_forme_that_matches_its_move(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Stance Change, which turned out to be implemented all along -- but not *declared*.

    I read the differential's `species: ours 'aegislashblade' vs showdown 'aegislash'` and
    the coverage report's "stancechange uncovered" together and concluded the ability did
    nothing. `_use_move` had handled it since before this session. What was missing was its
    entry in `all_modelled_abilities`, so the calculator reported
    `attacker.ability:stancechange` on every hit -- 158 times in a 13,000-game run -- for an
    ability that works. "Reported as unmodelled" is not "unimplemented": the reporting can
    be the broken half. The duplicate implementation I added on that reading is gone.

    The test earns its place anyway: the formes are 50/140 and 140/50 in attack and defence,
    so a Shield-forme Aegislash swinging Iron Head at a third of its real attack would be a
    large silent error, and nothing covered it before.
    """
    if "aegislashblade" not in reg.species:
        pytest.skip("this regulation has no Aegislash-Blade")

    pos = _synthetic_position(reg, team_a)
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    _install_move(pos, 0, 1, "protect")
    attacker = pos.sides[0].pokemon[pos.sides[0].active[0]]
    attacker.species = "aegislash"
    attacker.base_species = "aegislash"
    attacker.types = reg.species["aegislash"].types
    attacker.ability = "stancechange"
    attacker.item = None
    _install_move(pos, 0, 0, "ironhead")

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "ironhead", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)
    budget = replace(Budget.matrix(), enumerate_secondary=False)
    result = resolve_turn(reg, pos, [ours, theirs], budget=budget)
    after = max(result.branches, key=lambda b: b.probability).position
    changed = after.sides[0].pokemon[after.sides[0].active[0]]
    assert changed.species == "aegislashblade", (
        f"a damaging move has to put Aegislash in Blade forme, got {changed.species}"
    )
    blade_damage = (
        pos.sides[1].pokemon[pos.sides[1].active[0]].hp
        - after.sides[1].pokemon[after.sides[1].active[0]].hp
    )

    # King's Shield puts it back, and a status move that is not King's Shield leaves it.
    _install_move(pos, 0, 0, "kingsshield")
    shielded = resolve_turn(
        reg,
        pos,
        [
            SideAction(
                slots=(
                    _move_action(pos, 0, 0, "kingsshield", None),
                    _move_action(pos, 0, 1, "protect", None),
                )
            ),
            theirs,
        ],
        budget=budget,
    )
    kept = max(shielded.branches, key=lambda b: b.probability).position
    assert kept.sides[0].pokemon[kept.sides[0].active[0]].species == "aegislash", (
        "King's Shield is the one status move that changes the forme, back to Shield"
    )

    # The stats really followed the forme: the same Iron Head from a Pokemon stuck in
    # Shield forme does far less. 140 against 50 is a factor near three.
    attacker_shield = pos.sides[0].pokemon[pos.sides[0].active[0]]
    attacker_shield.ability = "blaze"  # anything but Stance Change
    _install_move(pos, 0, 0, "ironhead")
    stuck = resolve_turn(reg, pos, [ours, theirs], budget=budget)
    stuck_after = max(stuck.branches, key=lambda b: b.probability).position
    shield_damage = (
        pos.sides[1].pokemon[pos.sides[1].active[0]].hp
        - stuck_after.sides[1].pokemon[stuck_after.sides[1].active[0]].hp
    )
    assert blade_damage > shield_damage * 2, (
        f"Blade forme should hit far harder: {blade_damage} vs {shield_damage}"
    )


def test_cursed_body_can_disable_the_move_that_hit_it(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`MoveSlot.disabled` existed and `usable` honoured it, and nothing ever set it.

    Cursed Body is on 43 of the 394 tournament teams -- the most common ability the resolver
    did not model -- and it is `onDamagingHit` with `randomChance(3, 10)`, so it belongs in
    the same fan-out as any other sub-100% chance. The half that decides games is the
    legality half: a disabled move must not be offered next turn.
    """
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "ironhead")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    defender = pos.sides[1].pokemon[pos.sides[1].active[0]]
    defender.ability = "cursedbody"
    defender.hp = defender.maxhp

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "ironhead", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    budget = replace(Budget.matrix(), max_branches=64)
    result = resolve_turn(reg, pos, [ours, _protect_both(pos, 1)], budget=budget)
    assert result.total_probability == pytest.approx(1.0)

    def attacker_of(branch):  # noqa: ANN001, ANN202
        side = branch.position.sides[0]
        return side.pokemon[side.active[0]]

    hit = sum(
        b.probability
        for b in result.branches
        if attacker_of(b).volatile("disable") is not None
    )
    assert hit == pytest.approx(0.3, abs=1e-9), (
        f"Cursed Body is 3 in 10, got {hit:.3f} over {len(result.branches)} branches"
    )

    disabled_branch = next(
        b for b in result.branches if attacker_of(b).volatile("disable") is not None
    )
    volatile = attacker_of(disabled_branch).volatile("disable")
    assert volatile is not None and volatile.move == "ironhead", (
        f"the disabled move has to be the one that hit: {volatile}"
    )

    # The legality half: Iron Head is no longer on offer, and the rest of the moves are.
    offered = {
        a.move_id
        for action in side_actions(reg, disabled_branch.position, 0)
        for a in action.slots
        if isinstance(a, MoveAction) and a.slot == 0
    }
    assert "ironhead" not in offered, f"a disabled move must not be offered: {offered}"
    assert offered, "and the rest of the moves must still be there"


@pytest.mark.oracle
def test_light_clay_extends_a_screen_and_the_dump_says_so(
    reg: Regulation, oracle: Oracle
) -> None:
    """`durationCallback` does not mean "Showdown rolls it" -- usually it means an item.

        lightscreen: durationCallback(target, source) {
            if (source?.hasItem('lightclay')) return 8;
            return 5;
        }

    We used the unextended number and reported it as a roll, which was wrong twice over. 34
    of the 394 tournament teams hold Light Clay -- about seven in ten of the teams carrying a
    screen at all -- so a screen lasting 5 turns instead of 8 is an ordinary occurrence.

    Compared against the simulator's own stored duration, and the same battle without the
    item, so the test fails if the extension is applied unconditionally as well as if it is
    not applied at all.
    """
    entry = (reg.moves["lightscreen"].raw.get("durations") or {}).get("lightscreen") or {}
    assert entry.get("byItem", {}).get("lightclay") == 8, (
        f"the dump has to carry the extension, not just that a callback exists: {entry}"
    )
    assert not entry.get("rolled"), "a screen is extended, not rolled"

    def team(item: str) -> list[TeamSet]:
        held = list(_INFLICTOR_TEAM)
        first = held[0]
        held[0] = TeamSet(
            first.species, first.ability, first.nature, list(first.moves), dict(first.sp),
            item=item,
        )
        return held

    for item, expected in (("lightclay", 8), ("leftovers", 5)):
        handle = oracle.create(
            FORMAT_ID, team(item), _VICTIM_TEAM, policy=RandomnessPolicy(damage_roll=8)
        )
        handle.step(["team 1,2,3,4", "team 1,2,3,4"])
        before = Position.from_json(handle.position)
        # Grimmsnarl's Light Screen is slot 3; the victim uses Swords Dance so nothing is
        # blocked and nothing faints.
        ours = SideAction(
            slots=(
                _move_action(before, 0, 0, "lightscreen", None),
                _move_action(before, 0, 1, "protect", None),
            )
        )
        theirs = SideAction(
            slots=(
                _move_action(before, 1, 0, "protect", None),
                _move_action(before, 1, 1, "protect", None),
            )
        )
        handle.step([ours.to_choice(), theirs.to_choice()])
        assert not handle.choice_errors, handle.choice_errors

        showdown = next(
            (
                c.duration
                for c in Position.from_json(handle.position).sides[0].side_conditions
                if c.id == "lightscreen"
            ),
            None,
        )
        result = resolve_turn(reg, before, [ours, theirs], budget=Budget.deterministic(8))
        mine = next(
            (
                c.duration
                for c in result.branches[0].position.sides[0].side_conditions
                if c.id == "lightscreen"
            ),
            None,
        )
        # Both have spent one turn of it by the time the turn ends, so the stored number is
        # one less than the duration the callback returned.
        assert mine == showdown == expected - 1, (
            f"with {item}: ours {mine}, showdown {showdown}, expected {expected - 1}"
        )
        assert not any("lightscreen duration" in f for f in result.unmodelled), (
            f"an extended screen is not an approximation: {result.unmodelled}"
        )
        handle.close()


def test_every_duration_the_dump_carries_is_keyed_the_way_it_is_looked_up(
    reg: Regulation,
) -> None:
    """A duration nobody can find is a duration that silently falls back to a literal.

    Showdown's `weather` field is inconsistently cased -- 'sunnyday' but 'RainDance' and
    'Sandstorm' -- and the resolver normalises before looking the entry up, so an unnormalised
    key means rain and sand quietly keep the hardcoded 5 while sun gets its 8. Checking every
    key rather than the three that happened to be noticed.
    """
    for move in reg.moves.values():
        durations = move.raw.get("durations")
        if not isinstance(durations, dict):
            continue
        for key in durations:
            assert key == key.lower() and key.isalnum(), (
                f"{move.id} carries a duration keyed {key!r}, which is not how any caller "
                "spells an effect id"
            )

    # And the item extensions are actually reachable for the effect kinds that have them.
    for move_id, effect, item, extended in (
        ("lightscreen", "lightscreen", "lightclay", 8),
        ("sunnyday", "sunnyday", "heatrock", 8),
        ("raindance", "raindance", "damprock", 8),
        ("electricterrain", "electricterrain", "terrainextender", 8),
    ):
        if move_id not in reg.moves:
            continue
        entry = (reg.moves[move_id].raw.get("durations") or {}).get(effect) or {}
        assert entry.get("byItem", {}).get(item) == extended, (
            f"{move_id}: {item} should make {effect} last {extended}, dump says {entry}"
        )


def test_a_weather_duration_follows_its_rock(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The three lines under the side-condition fix still read `= 5`.

    No holder of a weather rock is in the current field, so this changes no number today --
    which is exactly why it went unnoticed, and exactly why a literal duration is the wrong
    shape. "No team happens to run Heat Rock this season" is not a property of the code.
    """
    entry = (reg.moves["sunnyday"].raw.get("durations") or {}).get("sunnyday") or {}
    rock = next(iter(entry.get("byItem", {})), None)
    if rock is None:
        pytest.skip("this regulation has no weather-extending item")

    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "sunnyday")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "protect")
    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "sunnyday", None),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)

    seen = {}
    for item in (None, rock):
        pos.sides[0].pokemon[pos.sides[0].active[0]].item = item
        result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
        after = result.branches[0].position.field
        assert after.weather == "sunnyday"
        seen[item] = after.weather_duration

    # One turn of it is spent by the residual phase, so the stored numbers are one less.
    assert seen[None] == entry["base"] - 1, seen
    assert seen[rock] == entry["byItem"][rock] - 1, (
        f"{rock} has to extend the weather: {seen}"
    )


def test_a_grass_type_ignores_a_powder_move(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The immunity was in the dump and only Rage Powder consulted it.

        gen >= 6 && move.flags['powder'] && target !== pokemon &&
            !this.dex.getImmunity('powder', target)

    So Sleep Powder put Grass types to sleep. 61 of the 394 tournament teams carry Sleep
    Powder and 304 of them field a Grass type, and `diverge_report.py` ranked
    `move:sleeppowder` first by lift with every divergence on `status` -- the fingerprint of
    a status landing that should not have.
    """
    assert reg.immune_to_effect("powder", ("Grass",))
    assert "powder" in reg.moves["sleeppowder"].flags

    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "sleeppowder")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "protect")
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "sleeppowder", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    # Swords Dance on the target, not Protect: Protect would block the move for the wrong
    # reason and the baseline case would fail while looking like the immunity works.
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    theirs = _protect_both(pos, 1)

    def slept(types: tuple[str, ...], **overrides: object) -> bool:
        target.types = types
        target.status = None
        target.ability = str(overrides.get("ability", "blaze"))
        target.item = overrides.get("item")  # type: ignore[assignment]
        result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
        after = result.branches[0].position.sides[1]
        return after.pokemon[after.active[0]].status == "slp"

    assert slept(("Fire",)), "the setup has to be able to land the move at all"
    assert not slept(("Grass",)), "a Grass type ignores a powder move"
    assert not slept(("Grass", "Poison")), "one immune type is enough"
    assert not slept(("Fire",), ability="overcoat"), "Overcoat blocks powder"
    assert not slept(("Fire",), item="safetygoggles"), "Safety Goggles blocks powder"


def test_prankster_immunity_survived_being_folded_together(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`_prankster_immune` became `_immune_to_move`; the rule it held has to still hold.

    Two per-target immunities in one loop want one place to live, but a refactor that drops
    a rule on the way is the reason this is checked separately from the powder one.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.ability = "prankster"
    _install_move(pos, 0, 0, "charm")
    _install_move(pos, 0, 1, "protect")
    # Nasty Plot, not Swords Dance: Prankster Charm resolves first at +1 priority, and a
    # target raising its own Attack afterwards would cancel the drop exactly -- which is how
    # the first version of this test managed to fail on its own baseline.
    for slot in (0, 1):
        _install_move(pos, 1, slot, "nastyplot")
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.boosts = {}

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "charm", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)

    target.types = ("Fire",)
    landed = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    hit = landed.branches[0].position.sides[1]
    assert hit.pokemon[hit.active[0]].boosts.get("atk", 0) < 0, (
        "the setup has to be able to land Charm at all"
    )

    target.types = ("Dark",)
    target.boosts = {}
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = result.branches[0].position.sides[1]
    blocked = after.pokemon[after.active[0]].boosts
    assert blocked.get("atk", 0) >= 0, (
        f"Charm must not reach a Dark type from Prankster: {blocked}"
    )


def test_feint_tears_the_guard_down_for_the_rest_of_the_turn(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Letting the move through was only half of `hitStepBreakProtect`.

    Showdown *removes* the Protect volatile and the side's Wide Guard, so the target's
    partner is exposed too -- which is the entire reason to bring Feint to a doubles game:
    break the Protect, then land the partner's move. We returned "not blocked" for the
    breaking move and left the volatile standing, so the partner was still blocked.
    `diverge_report.py` ranked `move:feint` second by lift with its divergences on `hp`.

    Breaking anything also clears `stall`, so the target's next Protect is certain again
    rather than one in three.
    """
    assert reg.moves["feint"].raw.get("breaksProtect")

    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "feint")
    _install_move(pos, 0, 1, "ironhead")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "protect")
    # Feint has +2 priority so it lands before the partner's Iron Head either way, but the
    # Protect has to already be up when Feint arrives -- which is what a Protect chosen the
    # same turn gives, since it resolves at +4.
    target = pos.sides[1].pokemon[pos.sides[1].active[0]]
    target.hp = target.maxhp
    partner_target = pos.sides[1].pokemon[pos.sides[1].active[1]]
    del partner_target

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "feint", 1),
            _move_action(pos, 0, 1, "ironhead", 1),
        )
    )
    theirs = _protect_both(pos, 1)
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    after = result.branches[0].position
    hit = after.sides[1].pokemon[after.sides[1].active[0]]

    assert hit.hp < target.maxhp, "Feint itself has to get through a Protect"
    assert hit.volatile("protect") is None and hit.volatile("detect") is None, (
        "the Protect has to be gone, not merely bypassed"
    )
    assert hit.volatile("stall") is None, "breaking a guard also resets the Protect counter"
    # And the consequence that matters: the partner's Iron Head landed too -- the target
    # loses more than to Feint alone, with the partner Protecting instead (IKA-210: this
    # was read off Python's events).
    alone = pos.copy()
    alone.sides[0].pokemon[alone.sides[0].active[1]].moves[1] = MoveSlot(id="protect", pp=10, maxpp=10)
    only_feint = SideAction(
        slots=(
            _move_action(alone, 0, 0, "feint", 1),
            _move_action(alone, 0, 1, "protect", None),
        )
    )
    single = resolve_turn(reg, alone, [only_feint, _protect_both(alone, 1)], budget=Budget.deterministic(8))
    feinted = single.branches[0].position.sides[1].pokemon[alone.sides[1].active[0]]
    assert hit.hp < feinted.hp < target.maxhp, (hit.hp, feinted.hp, target.maxhp)


def test_feint_also_strips_wide_guard_from_the_side(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`hitStepBreakProtect` removes the side conditions as well, from gen 6 regardless of side.

    Wide Guard is the doubles-relevant one: it is what stops the partner's spread move, so
    stripping only the single-target Protect would leave the interaction half modelled.
    """
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "feint")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    side = pos.sides[1]
    side.side_conditions.append(Effect(id="wideguard", duration=1))
    assert side.has_side_condition("wideguard")

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "feint", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    result = resolve_turn(
        reg, pos, [ours, _protect_both(pos, 1)], budget=Budget.deterministic(8)
    )
    after = result.branches[0].position
    assert not after.sides[1].has_side_condition("wideguard"), "Feint has to strip Wide Guard"


@pytest.mark.oracle
def test_the_leads_switch_in_abilities_fire_before_turn_one(
    reg: Regulation, oracle: Oracle
) -> None:
    """Showdown has applied Intimidate, Defiant and a lead's weather before `|turn|1`.

        |-ability|p1a: Incineroar|Intimidate|boost
        |-unboost|p2a: Kingambit|atk|1
        |-ability|p2a: Kingambit|Defiant|boost
        |-boost|p2a: Kingambit|atk|2
        |-unboost|p2b: Torkoal|atk|1
        |-weather|SunnyDay|[from] ability: Drought|[of] p2b: Torkoal
        |turn|1

    `position_from_sets` returned the raw position, so none of that existed: Intimidate is
    on 41% of the field through Incineroar alone, Defiant and Competitive answer it on
    another 17%, and every sun or rain team was being searched with no weather at all.

    The differential test cannot catch this, because it takes its positions *from* Showdown
    -- our own opening construction had never been compared to anything. So the comparison
    is made here, against the position Showdown reports at turn 1.
    """
    from pokeuraou.priors import SampledSet
    from pokeuraou.selfplay import position_from_sets

    ours = [
        TeamSet("Incineroar", "Intimidate", "Brave",
                ["fakeout", "flareblitz", "partingshot", "protect"], {"hp": 32},
                item="sitrusberry"),
        TeamSet("Charizard", "Blaze", "Timid",
                ["heatwave", "airslash", "protect", "flamethrower"], {"spe": 32},
                item="charcoal"),
        TeamSet("Garchomp", "Rough Skin", "Jolly",
                ["earthquake", "dragonclaw", "protect", "rockslide"], {"spe": 32},
                item="choicescarf"),
        TeamSet("Venusaur", "Chlorophyll", "Modest",
                ["sludgebomb", "gigadrain", "protect", "sleeppowder"], {"spa": 32},
                item="lifeorb"),
    ]
    theirs = [
        TeamSet("Kingambit", "Defiant", "Adamant",
                ["ironhead", "suckerpunch", "swordsdance", "protect"], {"atk": 32},
                item="focussash"),
        TeamSet("Torkoal", "Drought", "Bold",
                ["eruption", "protect", "helpinghand", "weatherball"], {"hp": 32},
                item="charcoal"),
        TeamSet("Toxapex", "Regenerator", "Relaxed",
                ["infestation", "toxic", "wideguard", "protect"], {"hp": 32},
                item="leftovers"),
        TeamSet("Sylveon", "Pixilate", "Modest",
                ["hypervoice", "protect", "yawn", "quickattack"], {"spa": 32},
                item="lifeorb"),
    ]

    def sampled(sets: list[TeamSet]) -> list[SampledSet]:
        return [
            SampledSet(
                species=to_id(t.species),
                ability=to_id(t.ability),
                item=to_id(t.item) if t.item else None,
                nature=t.nature,
                moves=[to_id(m) for m in t.moves],
                sp=dict(t.sp),
            )
            for t in sets
        ]

    handle = oracle.create(FORMAT_ID, ours, theirs, policy=RandomnessPolicy(damage_roll=8))
    handle.step(["team 1,2,3,4", "team 1,2,3,4"])
    showdown = Position.from_json(handle.position)
    mine = position_from_sets(reg, sampled(ours), sampled(theirs))

    assert mine.field.weather == showdown.field.weather == "sunnyday", (
        f"a lead's Drought has to put the sun up: ours {mine.field.weather}, "
        f"showdown {showdown.field.weather}"
    )
    assert mine.field.weather_duration == showdown.field.weather_duration

    for side in range(2):
        for slot in range(2):
            ours_mon = mine.sides[side].pokemon[mine.sides[side].active[slot]]
            their_mon = showdown.sides[side].pokemon[showdown.sides[side].active[slot]]
            assert ours_mon.species == their_mon.species, "the leads must line up"
            assert ours_mon.boosts == their_mon.boosts, (
                f"p{side + 1} slot {slot} ({ours_mon.species}): ours {ours_mon.boosts} vs "
                f"showdown {their_mon.boosts}"
            )
    # And specifically the interaction the user reported: Intimidate lowers it, Defiant
    # answers with +2, so Kingambit is a net +1 before anyone has chosen anything.
    kingambit = mine.sides[1].pokemon[mine.sides[1].active[0]]
    assert kingambit.boosts.get("atk") == 1, kingambit.boosts
    handle.close()


def test_fake_out_only_works_on_the_turn_its_user_came_in(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`if (source.activeMoveActions > 1) return false` -- and we tracked no such counter.

    Fake Out is the most common move in the tournament field, on 234 of 394 teams, and it
    worked on every turn. A recorded game had Incineroar using it on turn 4 having been on
    the field since turn 1.

    The counter is reset by a switch, so a Pokemon that leaves and comes back may use it
    again -- asserted too, because gating on "turn 1" instead of "first move since coming
    in" would pass the first half of this test and fail the game.
    """
    pos = _synthetic_position(reg, team_a)
    _install_move(pos, 0, 0, "fakeout")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")
    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "fakeout", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    theirs = _protect_both(pos, 1)

    def flinched(position: Position) -> bool:
        """Whether the target lost its action -- Swords Dance never raised its Attack."""
        result = resolve_turn(reg, position, [ours, theirs], budget=Budget.deterministic(8))
        after = result.branches[0].position.sides[1]
        return after.pokemon[after.active[0]].boosts.get("atk", 0) <= 0

    assert flinched(pos), "the first move out has to work"

    # Second turn on the field: refused.
    user = pos.sides[0].pokemon[pos.sides[0].active[0]]
    user.active_move_actions = 1
    assert not flinched(pos), "Fake Out must fail once its user has already moved"

    # And a switch re-arms it, which is why the counter is on the Pokemon and not the turn.
    user.active_move_actions = 0
    assert flinched(pos), "coming back in re-arms Fake Out"


def test_a_choice_item_locks_its_holder_into_one_move(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """`choicelock` was written by the resolver and read by nothing.

    107 of the 394 tournament teams carry a Choice Scarf, and being locked is most of what
    the Scarf costs -- so a Garchomp that switched moves every turn was getting the Speed
    for free.

    The existing guards could not see it, and the reason is structural: they take their
    positions *from* Showdown, whose snapshot already marks the locked-out moves
    `disabled`, so our own `usable` filtered them without ever consulting the volatile. The
    bug only existed in positions self-play builds itself. That makes this test's starting
    point -- a position we constructed -- the point of it.
    """
    assert "choicescarf" in reg.choice_items, "the dump has to say which items lock"

    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.item = "choicescarf"
    assert len(mon.moves) >= 2, "the fixture needs a second move to be locked out of"
    locked_move = mon.moves[0].id
    other = mon.moves[1].id

    def offered() -> set[str]:
        return {
            a.move_id
            for action in side_actions(reg, pos, 0)
            for a in action.slots
            if isinstance(a, MoveAction) and a.slot == 0
        }

    assert {locked_move, other} <= offered(), "unlocked, everything is on offer"

    mon.volatiles.append(Effect(id="choicelock", move=locked_move))
    assert offered() == {locked_move}, (
        f"a Choice holder may only repeat its move, got {sorted(offered())}"
    )

    # `onDisableMove` drops the lock when the item goes, so Knock Off frees the holder.
    mon.item = None
    assert other in offered(), "losing the item has to free the holder"

    # ...and when the move itself is gone, which is the other half of the same guard.
    mon.item = "choicescarf"
    mon.volatiles = [Effect(id="choicelock", move="somemovenotknown")]
    assert other in offered(), "a lock naming a move it does not know cannot hold"


def test_using_a_move_with_a_choice_item_sets_the_lock(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The lock has to be applied by the turn, not just honoured when present.

    Set for any item the dump marks `isChoice` rather than for `choicescarf` by name -- the
    old code named the Scarf, which is the only Choice item in this regulation's pool but
    is not what the rule says.
    """
    pos = _synthetic_position(reg, team_a)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.item = "choicescarf"
    _install_move(pos, 0, 0, "ironhead")
    _install_move(pos, 0, 1, "protect")
    for slot in (0, 1):
        _install_move(pos, 1, slot, "swordsdance")

    ours = SideAction(
        slots=(
            _move_action(pos, 0, 0, "ironhead", 1),
            _move_action(pos, 0, 1, "protect", None),
        )
    )
    result = resolve_turn(
        reg, pos, [ours, _protect_both(pos, 1)], budget=Budget.deterministic(8)
    )
    after = result.branches[0].position
    holder = after.sides[0].pokemon[after.sides[0].active[0]]
    lock = holder.volatile("choicelock")
    assert lock is not None and lock.move == "ironhead", (
        f"using a move on a Choice item has to record it: {lock}"
    )
    offered = {
        a.move_id
        for action in side_actions(reg, after, 0)
        for a in action.slots
        if isinstance(a, MoveAction) and a.slot == 0
    }
    assert offered == {"ironhead"}, f"next turn only that move is legal: {sorted(offered)}"


# ---------------------------------------------------------------------------
# Rules that were tested on Python's `_Turn` (IKA-210)
# ---------------------------------------------------------------------------
#
# Endure, Focus Sash and Focus Band were tested by calling Python's `_Turn.deal_damage`
# directly, the sleep and freeze counters by `_Turn.apply_status`, Disable and Encore by
# `_apply_disable` / `_apply_encore`. Those are Python's insides and go with it; the rules
# are asked of the port here as whole turns. The synthetic position is a mirror: p1a and
# p2a Charizard (slow), p1b and p2b Absol (fast). Every turn is pinned (`deterministic`)
# unless the rule is a roll, and whoever is not under test Protects.


def _slot(pos: Position, side: int, slot: int):  # noqa: ANN202
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _set_move(pos: Position, side: int, slot: int, index: int, move_id: str, pp: int = 10) -> None:
    _slot(pos, side, slot).moves[index] = MoveSlot(id=move_id, pp=pp, maxpp=max(pp, 1))


def _act(pos: Position, side: int, *chosen: tuple[str, int | None]) -> SideAction:
    return SideAction(
        slots=tuple(
            _move_action(pos, side, slot, move_id, target)
            for slot, (move_id, target) in enumerate(chosen)
        )
    )


def _one(reg: Regulation, pos: Position, ours: SideAction, theirs: SideAction) -> Position:
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(0))
    assert len(result.branches) == 1 and not result.suspended, result
    return result.branches[0].position


def _hit_by_air_slash(reg: Regulation, team_a: list[TeamSet], move_id: str, item: str | None,
                      hp: int, maxhp: int) -> tuple[Position, object]:
    """p1a Charizard's Air Slash into p2a, which uses `move_id`, holds `item` and stands at
    `hp` of `maxhp` -- a small maximum, so the hit is lethal from full."""
    pos = _synthetic_position(reg, team_a)
    target = _slot(pos, 1, 0)
    target.maxhp, target.hp, target.item = maxhp, hp, item
    _set_move(pos, 1, 0, 0, move_id)
    _set_move(pos, 1, 1, 0, "protect")
    _set_move(pos, 0, 1, 0, "protect")
    after = _one(
        reg, pos,
        _act(pos, 0, ("airslash", 1), ("protect", None)),
        _act(pos, 1, (move_id, None), ("protect", None)),
    )
    return after, _slot(after, 1, 0)


@pytest.mark.parametrize("endures", [True, False], ids=["endure", "control"])
def test_endure_survives_a_lethal_move_from_any_hp(
    reg: Regulation, team_a: list[TeamSet], endures: bool
) -> None:
    """Endure caps a move's damage at 1 HP from wherever the Pokemon stands; the control
    without it faints to the same hit."""
    _after, mon = _hit_by_air_slash(reg, team_a, "endure" if endures else "swordsdance", None, 10, 185)
    if endures:
        assert mon.hp == 1 and not mon.fainted, mon.hp
    else:
        assert mon.fainted


def test_endure_does_not_survive_residual_damage(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Showdown's Endure tests `effect.effectType === 'Move'`: a sandstorm goes through it."""
    pos = _synthetic_position(reg, team_a)
    pos.field.weather, pos.field.weather_duration = "sandstorm", 5
    target = _slot(pos, 1, 0)
    target.hp, target.item = 4, None
    _set_move(pos, 1, 0, 0, "endure")
    for side, slot in ((0, 0), (0, 1), (1, 1)):
        _set_move(pos, side, slot, 0, "protect")
    after = _one(
        reg, pos,
        _act(pos, 0, ("protect", None), ("protect", None)),
        _act(pos, 1, ("endure", None), ("protect", None)),
    )
    assert _slot(after, 1, 0).fainted


def test_focus_sash_needs_full_hp_and_is_consumed(reg: Regulation, team_a: list[TeamSet]) -> None:
    """At full HP the Sash leaves 1 HP and is spent; one short of full it does nothing."""
    pos = _synthetic_position(reg, team_a)
    full, short = _slot(pos, 1, 0), _slot(pos, 1, 1)
    full.maxhp = full.hp = 10
    short.maxhp, short.hp = 10, 9
    full.item = short.item = "focussash"
    _set_move(pos, 1, 0, 0, "swordsdance")
    _set_move(pos, 1, 1, 0, "swordsdance")
    after = _one(
        reg, pos,
        _act(pos, 0, ("airslash", 1), ("playrough", 2)),
        _act(pos, 1, ("swordsdance", None), ("swordsdance", None)),
    )
    saved, gone = _slot(after, 1, 0), _slot(after, 1, 1)
    assert saved.hp == 1 and not saved.fainted and saved.item is None, (saved.hp, saved.item)
    assert gone.fainted


def test_endure_takes_precedence_over_the_sash(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Endure runs at onDamagePriority -10 and the Sash at -40: Endure caps the damage
    first, so the Sash sees a survivable hit and is not spent."""
    _after, mon = _hit_by_air_slash(reg, team_a, "endure", "focussash", 10, 10)
    assert mon.hp == 1
    assert mon.item == "focussash", "Endure absorbed the hit, so the Sash is still held"


def test_focus_band_is_a_chance_and_is_reported_not_guessed(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Focus Band is `randomChance(1, 10)` from any HP and is not consumed. Under the pinned
    budget the port takes the reading the pinned oracle policy takes, and says so."""
    pos = _synthetic_position(reg, team_a)
    target = _slot(pos, 1, 0)
    target.maxhp = target.hp = 10
    target.item = "focusband"
    _set_move(pos, 1, 0, 0, "swordsdance")
    _set_move(pos, 1, 1, 0, "protect")
    _set_move(pos, 0, 1, 0, "protect")
    result = resolve_turn(
        reg, pos,
        [
            _act(pos, 0, ("airslash", 1), ("protect", None)),
            _act(pos, 1, ("swordsdance", None), ("protect", None)),
        ],
        budget=Budget.deterministic(0),
    )
    mon = _slot(result.branches[0].position, 1, 0)
    assert mon.fainted
    assert mon.item == "focusband", "Focus Band is not consumed"
    assert any("focusband" in note for note in result.unmodelled), result.unmodelled


def _quiet(pos: Position, target_move: str = "swordsdance") -> list[SideAction]:
    """Everyone Protects but p2a, which uses `target_move`."""
    return [
        _act(pos, 0, ("protect", None), ("protect", None)),
        _act(pos, 1, (target_move, None), ("protect", None)),
    ]


def _heaviest(reg: Regulation, pos: Position, actions: list[SideAction]) -> Position:
    result = resolve_turn(reg, pos, actions, budget=Budget.matrix())
    assert result.branches and not result.suspended
    return max(result.branches, key=lambda b: b.probability).position


def test_a_sleeping_pokemon_wakes_and_the_counter_is_not_the_lucky_one(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Champions sleep is `sample([2, 3, 3])` (data/mods/champions/conditions.ts): the modal
    counter is 3, and sleep does end. Absol (faster) Spores Charizard, and the heaviest
    branch is followed from there -- the modal counter, if the port rolls it.

    Charizard tries to move on the Spore turn itself, which spends one (`time--` in
    `slp.onBeforeMove`), so the modal 3 leaves it asleep through one more turn and awake on
    the next; the lucky 2 would wake it on the first.
    """
    pos = _synthetic_position(reg, team_a)
    _set_move(pos, 0, 1, 0, "spore")
    _set_move(pos, 0, 0, 0, "protect")
    _set_move(pos, 1, 1, 0, "protect")
    _set_move(pos, 1, 0, 0, "swordsdance")
    pos = _heaviest(
        reg, pos,
        [
            _act(pos, 0, ("protect", None), ("spore", 1)),
            _act(pos, 1, ("swordsdance", None), ("protect", None)),
        ],
    )
    assert _slot(pos, 1, 0).status == "slp"
    asleep = 0
    for _ in range(5):
        pos = _heaviest(reg, pos, _quiet(pos))
        mon = _slot(pos, 1, 0)
        assert not mon.fainted, "nothing should be able to faint in a turn of Protects"
        if mon.status != "slp":
            break
        asleep += 1
    assert asleep == 1, f"the modal counter keeps it asleep one turn past the Spore's, counted {asleep}"


def test_freeze_cannot_last_the_whole_battle(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The champions mod caps a freeze at three attempts (`startTime = 3`); following the
    heaviest branch -- the 1-in-4 thaw roll answered "no" every time -- it still ends."""
    pos = _synthetic_position(reg, team_a)
    for side, slot in ((0, 0), (0, 1), (1, 1)):
        _set_move(pos, side, slot, 0, "protect")
    _set_move(pos, 1, 0, 0, "swordsdance")
    target = _slot(pos, 1, 0)
    target.status, target.status_counter = "frz", 3
    for _ in range(5):
        pos = _heaviest(reg, pos, _quiet(pos))
        mon = _slot(pos, 1, 0)
        assert not mon.fainted
        if mon.status != "frz":
            break
    else:
        raise AssertionError("still frozen after 5 turns with the thaw roll answered no")


def test_a_disable_expires_and_the_move_comes_back(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The flag lives on the move slot, so expiry has to clear it or it is permanent."""
    pos = _synthetic_position(reg, team_a)
    for side, slot in ((0, 1), (1, 0), (1, 1)):
        _set_move(pos, side, slot, 0, "protect")
    _set_move(pos, 0, 0, 0, "protect")
    _set_move(pos, 0, 0, 1, "swordsdance")
    mon = _slot(pos, 0, 0)
    mon.last_move = "protect"
    mon.moves[0].disabled = True
    mon.volatiles.append(Effect(id="disable", duration=4, move="protect"))
    seen: list[int | None] = [4]
    for _ in range(6):
        pos = _heaviest(
            reg, pos,
            [
                _act(pos, 0, ("swordsdance", None), ("protect", None)),
                _act(pos, 1, ("protect", None), ("protect", None)),
            ],
        )
        mon = _slot(pos, 0, 0)
        current = mon.volatile("disable")
        seen.append(current.duration if current else None)
        if current is None:
            break
    assert seen[-1] is None, f"the disable never expired: durations {seen}"
    assert all(m.usable or m.pp <= 0 for m in mon.moves), [(m.id, m.disabled, m.pp) for m in mon.moves]


@pytest.mark.parametrize(
    ("order", "locked", "left"),
    [("encore first", "airslash", 2), ("target first", "extremespeed", 3)],
)
def test_encore_leaves_exactly_one_move_on_offer(
    reg: Regulation, team_a: list[TeamSet], order: str, locked: str, left: int
) -> None:
    """Encore locks the target's last move for 3 turns, 4 when the target has already moved
    this turn (`if (!this.queue.willMove(target)) this.effectState.duration++`), one of them
    spent at the end of this one; the menu after is that move alone."""
    pos = _synthetic_position(reg, team_a)
    _set_move(pos, 0, 1, 0, "encore")
    _set_move(pos, 0, 0, 0, "protect")
    _set_move(pos, 1, 1, 0, "protect")
    target = _slot(pos, 1, 0)
    target.last_move = "airslash"
    if order == "target first":
        _set_move(pos, 1, 0, 3, "extremespeed")
        theirs = _act(pos, 1, ("extremespeed", 1), ("protect", None))
    else:
        _set_move(pos, 1, 0, 3, "swordsdance")
        theirs = _act(pos, 1, ("swordsdance", None), ("protect", None))
    result = resolve_turn(
        reg, pos, [_act(pos, 0, ("protect", None), ("encore", 1)), theirs], budget=Budget.deterministic(0)
    )
    assert result.branches
    for branch in result.branches:
        after = branch.position
        held = _slot(after, 1, 0).volatile("encore")
        assert held is not None and held.move == locked, held
        assert held.duration == left, (order, held.duration)
        offered = {
            a.move_id
            for action in side_actions(reg, after, 1)
            for a in action.slots
            if isinstance(a, MoveAction) and a.slot == 0
        }
        assert offered == {locked}, sorted(offered)


@pytest.mark.parametrize("why", ["never moved", "failencore", "no pp"])
def test_encore_refuses_what_showdown_refuses(reg: Regulation, team_a: list[TeamSet], why: str) -> None:
    """`onStart` returns false three ways, and each means no volatile at all."""
    pos = _synthetic_position(reg, team_a)
    _set_move(pos, 0, 1, 0, "encore")
    _set_move(pos, 0, 0, 0, "protect")
    _set_move(pos, 1, 1, 0, "protect")
    _set_move(pos, 1, 0, 3, "swordsdance")
    target = _slot(pos, 1, 0)
    if why == "never moved":
        target.last_move = None
    elif why == "failencore":
        assert "failencore" in reg.moves["encore"].flags
        _set_move(pos, 1, 0, 0, "encore", pp=5)
        target.last_move = "encore"
    else:
        _set_move(pos, 1, 0, 0, "protect", pp=0)
        target.last_move = "protect"
    after = _one(
        reg, pos,
        _act(pos, 0, ("protect", None), ("encore", 1)),
        _act(pos, 1, ("swordsdance", None), ("protect", None)),
    )
    assert _slot(after, 1, 0).volatile("encore") is None
