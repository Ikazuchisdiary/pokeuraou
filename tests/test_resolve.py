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

import pytest

from pokeuraou.actions import MoveAction, PassAction, SideAction, SwitchAction, side_actions
from pokeuraou.oracle import ORACLE_JS, TeamSet
from pokeuraou.position import Effect, MoveSlot, Position
from pokeuraou.regulation import Regulation
from pokeuraou.resolve import (
    Budget,
    _Turn,
    multihit_counts,
    pending_attacks,
    resolve_turn,
    stall_success_chance,
    stratified_rolls,
)

from . import _diff_turn_entry as diff_turn
from .test_actions import _synthetic_position

#: Measured over 445 turns against the pinned Showdown commit: 4.9% total, 2.9% silent.
#: The headroom above the measurement is deliberate but small; a regression that pushes
#: past it fails here and ``tools/diverge_report.py`` names the cause by statistical lift.
MAX_SILENT_DIVERGENCE = 0.05
MAX_TOTAL_DIVERGENCE = 0.08


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
    assert any(v.id == "endure" for v in user.volatiles)
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
