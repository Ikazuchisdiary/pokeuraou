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

import pytest

from pokeuraou.actions import (
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    side_actions,
    switch_actions_after_faint,
)
from pokeuraou.oracle import ORACLE_JS, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Effect, MoveSlot, Position
from pokeuraou.regulation import Regulation
from pokeuraou.resolve import (
    Average,
    BestOf,
    Budget,
    _Turn,
    multihit_counts,
    pending_attacks,
    resolve_turn,
    resume_alternatives,
    resume_turn,
    self_switches_needed,
    stall_success_chance,
    stratified_rolls,
    turn_expectation,
    turn_leaves,
)

from . import _diff_turn_entry as diff_turn
from .conftest import FORMAT_ID
from .test_actions import _synthetic_position

#: Measured over 963 turns against the pinned Showdown commit: 10.4% total, 4.0% silent.
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
#: need loosening (3.95% against 0.05), and giving it away here would cost the guard.
MAX_SILENT_DIVERGENCE = 0.05
MAX_TOTAL_DIVERGENCE = 0.12


# ---------------------------------------------------------------------------
# Randomness accounting
# ---------------------------------------------------------------------------


def test_stratified_rolls_are_a_probability_distribution() -> None:
    for count in range(1, 17):
        rolls = stratified_rolls(Budget(damage_rolls=count))
        assert sum(w for _, w in rolls) == pytest.approx(1.0)
        assert all(0 <= r < 16 for r, _ in rolls)
        assert len({r for r, _ in rolls}) == len(rolls)


def test_sixteen_rolls_is_the_exact_distribution() -> None:
    rolls = stratified_rolls(Budget(damage_rolls=16))
    assert [r for r, _ in rolls] == list(range(16))
    assert all(w == pytest.approx(1 / 16) for _, w in rolls)


def test_stratification_keeps_the_mean_roll() -> None:
    """A reduced roll set is a quantisation, not a bias.

    Each representative carries the weight of the block it stands for, so the expected
    roll index is preserved -- which is what stops a cheap budget from systematically
    over- or under-estimating damage.
    """
    exact = sum(r * w for r, w in stratified_rolls(Budget(damage_rolls=16)))
    for count in (2, 4, 8):
        got = sum(r * w for r, w in stratified_rolls(Budget(damage_rolls=count)))
        assert got == pytest.approx(exact, abs=0.5)


def test_a_pinned_roll_is_a_single_branch() -> None:
    assert stratified_rolls(Budget.deterministic(8)) == [(8, 1.0)]
    assert stratified_rolls(Budget.deterministic(0)) == [(0, 1.0)]
    assert stratified_rolls(Budget.deterministic(15)) == [(15, 1.0)]


def test_stall_chance_matches_showdown(reg: Regulation) -> None:
    """Protect is certain the first time and one in three the second."""
    del reg
    assert stall_success_chance(1) == 1.0
    assert stall_success_chance(3) == pytest.approx(1 / 3)
    assert stall_success_chance(9) == pytest.approx(1 / 9)


def test_multihit_distribution(reg: Regulation) -> None:
    exact = Budget.exact()
    fixed = Budget.deterministic(0)

    single = reg.moves["ironhead"]
    assert multihit_counts(single, exact) == [(1, 1.0)]

    if "populationbomb" in reg.moves:
        assert multihit_counts(reg.moves["populationbomb"], exact) == [(10, 1.0)]

    if "dualwingbeat" in reg.moves:
        assert multihit_counts(reg.moves["dualwingbeat"], exact) == [(2, 1.0)]

    # A [2, 5] move is not uniform: Showdown samples [2, 2, 3, 3, 4, 5].
    two_to_five = next(
        (m for m in reg.moves.values() if m.raw.get("multihit") == [2, 5]), None
    )
    if two_to_five is not None:
        spread = dict(multihit_counts(two_to_five, exact))
        assert spread[2] == pytest.approx(1 / 3)
        assert spread[3] == pytest.approx(1 / 3)
        assert spread[4] == pytest.approx(1 / 6)
        assert spread[5] == pytest.approx(1 / 6)
        assert sum(spread.values()) == pytest.approx(1.0)
        # A pinned budget takes the minimum, which is what Showdown's policy forces.
        assert multihit_counts(two_to_five, fixed) == [(2, 1.0)]


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
    with pytest.raises(ValueError, match="belief layer"):
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
    after = result.branches[0].position.sides[1].pokemon[0]
    assert after.hp == hp_before
    assert any("blocked by protect" in e for e in result.branches[0].events)


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


def _install_move(pos: Position, side: int, slot: int, move_id: str) -> None:
    """Puts a move in the active Pokemon's first slot.

    The fixture teams come from usage priors, so whether one happens to carry Wide Guard
    is chance. Legality is ``side_actions``' job and is tested separately; here the point
    is what the move does once chosen.
    """
    mon = pos.sides[side].pokemon[pos.sides[side].active[slot]]
    mon.moves[0] = MoveSlot(id=move_id, pp=10, maxpp=10)


def _turn_for(reg: Regulation, pos: Position) -> _Turn:
    return _Turn(reg, pos, Budget.deterministic(0), {})


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


def test_endure_survives_a_lethal_move_from_any_hp(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    pos = _synthetic_position(reg, team_a)
    turn = _turn_for(reg, pos)
    mon = turn.mon_at(1, 0)
    assert mon is not None
    mon.hp = mon.maxhp // 2
    mon.item = None
    mon.volatiles.append(Effect(id="endure", duration=1))

    dealt = turn.deal_damage(1, 0, mon.maxhp * 4, reason="testmove", from_move=True)
    assert dealt == mon.maxhp // 2 - 1
    assert mon.hp == 1 and not mon.fainted


def test_endure_does_not_survive_residual_damage(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Showdown's Endure tests `effect.effectType === 'Move'`.

    Sandstorm, recoil and Life Orb go straight through it. Applying the cap to every
    source turns a lost Pokemon into a surviving one.
    """
    pos = _synthetic_position(reg, team_a)
    turn = _turn_for(reg, pos)
    mon = turn.mon_at(1, 0)
    assert mon is not None
    mon.hp = 4
    mon.item = None
    mon.volatiles.append(Effect(id="endure", duration=1))

    turn.deal_damage(1, 0, 40, reason="sandstorm")
    assert mon.fainted


def test_focus_sash_needs_full_hp_and_is_consumed(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    pos = _synthetic_position(reg, team_a)
    turn = _turn_for(reg, pos)
    mon = turn.mon_at(1, 0)
    assert mon is not None
    mon.hp = mon.maxhp
    mon.item = "focussash"

    turn.deal_damage(1, 0, mon.maxhp * 4, reason="testmove", from_move=True)
    assert mon.hp == 1 and not mon.fainted
    assert mon.item is None

    # One short of full is not full.
    other = turn.mon_at(1, 1)
    assert other is not None
    other.hp = other.maxhp - 1
    other.item = "focussash"
    turn.deal_damage(1, 1, other.maxhp * 4, reason="testmove", from_move=True)
    assert other.fainted


def test_endure_takes_precedence_over_the_sash(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Endure runs at onDamagePriority -10 and the Sash at -40.

    Endure caps the damage first, so the Sash sees a survivable hit and is not spent.
    """
    pos = _synthetic_position(reg, team_a)
    turn = _turn_for(reg, pos)
    mon = turn.mon_at(1, 0)
    assert mon is not None
    mon.hp = mon.maxhp
    mon.item = "focussash"
    mon.volatiles.append(Effect(id="endure", duration=1))

    turn.deal_damage(1, 0, mon.maxhp * 4, reason="testmove", from_move=True)
    assert mon.hp == 1
    assert mon.item == "focussash", "Endure absorbed the hit, so the Sash is still held"


def test_focus_band_is_a_chance_and_is_reported_not_guessed(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Focus Band is `randomChance(1, 10)` from any HP and is not consumed.

    Resolving it as a certain save was wrong; resolving it silently as no save would be
    wrong in the other direction on a tenth of these hits. The resolver takes the
    deterministic reading the pinned oracle policy takes and declares it, so the turn
    lands in the flagged bucket rather than the silent one.
    """
    pos = _synthetic_position(reg, team_a)
    turn = _turn_for(reg, pos)
    mon = turn.mon_at(1, 0)
    assert mon is not None
    mon.hp = mon.maxhp
    mon.item = "focusband"

    turn.deal_damage(1, 0, mon.maxhp * 4, reason="testmove", from_move=True)
    assert mon.fainted
    assert mon.item == "focusband", "Focus Band is not consumed"
    assert any("focusband" in flag for flag in turn.unmodelled)


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


def test_pending_attacks_reads_the_chosen_actions(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Sucker Punch is decidable here because the resolver sees both sides' choices."""
    pos = _synthetic_position(reg, team_a)
    actions = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    attacks = pending_attacks(reg, actions)
    assert attacks
    for (side, slot), attacking in attacks.items():
        chosen = next(s for s in actions[side].slots if s.slot == slot)
        if isinstance(chosen, MoveAction):
            assert attacking == (reg.moves[chosen.move_id].category != "Status")
        else:
            assert attacking is False


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
# Differential test
# ---------------------------------------------------------------------------


#: A single seed compares about 50 turns, so one divergence moves the rate by two whole
#: points and a threshold set at the aggregate's value would be testing which teams the
#: usage sampler happened to draw. The *silent* rate is still held tight per seed, because
#: that is the load-bearing number and a new silent divergence is a regression whatever
#: the sample; the total rate is held tight only on the large sample below, where the
#: count of *declared* gaps averages out.
MAX_TOTAL_DIVERGENCE_PER_SEED = 0.14


@pytest.mark.oracle
@pytest.mark.parametrize("seed", [1, 5])
def test_resolved_positions_match_showdown(seed: int) -> None:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    report = diff_turn.run(battles=8, roll=8, seed=seed, max_turns=10)
    assert report.compared >= 25, f"only {report.compared} turns compared"
    assert report.silent_rate <= MAX_SILENT_DIVERGENCE, report.render()
    assert report.divergence_rate <= MAX_TOTAL_DIVERGENCE_PER_SEED, report.render()


@pytest.mark.oracle
@pytest.mark.slow
def test_resolver_divergence_over_a_large_sample() -> None:
    """The number quoted in the README, measured rather than asserted from memory."""
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    compared = matched = silent = flagged = 0
    for seed in range(1, 9):
        report = diff_turn.run(battles=10, roll=8, seed=seed, max_turns=10)
        compared += report.compared
        matched += report.matched
        silent += report.silent_divergences
        flagged += report.flagged_divergences
    total_rate = 1 - matched / compared
    silent_rate = silent / compared
    print(
        f"\nresolver divergence over {compared} turns: {total_rate * 100:.2f}% total, "
        f"{silent_rate * 100:.2f}% silent, {flagged} flagged"
    )
    assert compared > 350
    assert silent_rate <= MAX_SILENT_DIVERGENCE
    assert total_rate <= MAX_TOTAL_DIVERGENCE


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


def test_every_volatile_that_should_expire_has_a_duration(reg: Regulation) -> None:
    """A volatile added with no duration is kept for the rest of the battle.

    The expiry loop reads ``if volatile.duration is not None: volatile.duration -= 1``, so
    ``None`` means permanent. `_duration` used to look only at the move's own ``condition``
    field, while Showdown keeps most durations on the condition in conditions.ts -- and that
    silently made four effects permanent:

    - **partiallytrapped** (bind, Fire Spin, Infestation, Sand Tomb, Snap Trap, Whirlpool,
      Thunder Cage): the target could never switch again and lost an eighth of its HP every
      turn for the rest of the game. Infestation is played on 40% of the turns Toxapex is
      out, so this was frequent rather than exotic.
    - **Endure**: a stalling move whose volatile is not in ``PROTECT_VOLATILES``, so it
      missed the unconditional one-turn removal too. Once used, the Pokemon survived every
      lethal hit at 1 HP for the rest of the battle.
    - magnetrise, electrify: same shape, rarer moves.

    This is the audit that would have caught all four, so it lives here as a test rather
    than as a one-off script: every move whose volatile Showdown gives a duration must get
    one from us, unless the volatile is in the set the residual removes unconditionally.
    """
    from pokeuraou.resolve import PROTECT_VOLATILES, _duration

    # Removed every turn regardless of duration, so `None` is harmless for these.
    single_turn = set(PROTECT_VOLATILES) | {
        "flinch", "helpinghand", "followme", "ragepowder", "spotlight", "glaiverush",
    }
    # The expectation comes from the dump rather than from a list here, so it cannot drift
    # from Showdown -- which is the whole point: the hand-written list this replaced was
    # both incomplete (four permanent effects) and wrong (Tailwind 5, Showdown says 4).
    permanent: list[str] = []
    for move in reg.moves.values():
        volatile = move.raw.get("volatileStatus")
        if not isinstance(volatile, str) or volatile in single_turn:
            continue
        declared = (move.raw.get("durations") or {}).get(volatile)
        if not isinstance(declared, dict) or "duration" not in declared:
            continue  # Showdown leaves it permanent too, so None is correct.
        if _duration(move, volatile) is None:
            permanent.append(f"{move.id} -> {volatile}")

    assert not permanent, (
        "these would be added with no duration and never expire: " + ", ".join(sorted(permanent))
    )
    # And the two that mattered, by name.
    assert _duration(reg.moves["infestation"], "partiallytrapped") == 5
    assert _duration(reg.moves["tailwind"], "tailwind") == 4, "Tailwind is 4 turns, not 5"
    assert _duration(reg.moves["endure"], "endure") == 1


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
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
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
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended

    chooser, alternatives = resume_alternatives(reg, result.suspended[0])
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
