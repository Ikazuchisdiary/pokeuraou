"""The posterior update, and the one failure mode that would be fatal.

A likelihood that is merely loose makes the belief weak. A likelihood that assigns zero to
the truth makes it *wrong*, permanently and silently: once a particle is dead it never
comes back, so from then on the tool is reasoning about a set of spreads that excludes the
opponent's actual one, and every frequency it prints is an answer to a different question.

So the central test is self-consistency over the whole particle set: for each of a sample
of particles, compute the damage that particle would really deal, feed it in as the
observation, and assert that particle keeps positive weight. That is checked for every
channel, because each one reads a different mechanic and each could be wrong in a
different direction.

The oracle test closes the loop against Showdown: a real turn, the exact damage taken read
off the simulator's own positions, and the requirement that the opponent's true spread
survives the update.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.actions import side_actions
from pokeuraou.belief import battler_for
from pokeuraou.damage import calculate, effective_damage, register_mega_stones
from pokeuraou.hpdisplay import displayed_percent
from pokeuraou.observe import (
    DamageDealt,
    DamageTaken,
    FractionalLoss,
    MovedFirst,
    damage_taken_likelihood,
    likelihood_for,
    update,
)
from pokeuraou.oracle import ORACLE_JS, Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team
from pokeuraou.regulation import Regulation
from pokeuraou.setup import load_scenario, with_spreads
from pokeuraou.view import battler, field_state

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"
FORMAT_ID = "gen9championsvgc2026regmc"


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    """A materialised position plus a belief per hidden active Pokemon."""
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE.name} missing")
    from pokeuraou.cli import _modal, build_beliefs

    scenario = load_scenario(EXAMPLE)
    register_mega_stones(scenario.reg)
    beliefs = build_beliefs(scenario)
    modal = {key: _modal(b) for key, b in beliefs.items()}
    base = with_spreads(scenario, modal)
    active = {}
    for side_index, side in enumerate(base.sides):
        for slot, party_index in enumerate(side.active):
            if (side_index, party_index) in beliefs:
                active[(side_index, slot)] = beliefs[(side_index, party_index)]
    return scenario, base, active


def _damaging_move(reg: Regulation, pos: Position, key: SlotKey) -> str | None:  # type: ignore[name-defined]
    side_index, slot = key
    mon = pos.sides[side_index].pokemon[pos.sides[side_index].active[slot]]
    for move_slot in mon.moves:
        move = reg.moves.get(move_slot.id)
        if move is None or move.category == "Status":
            continue
        if move.target not in ("normal", "any", "adjacentFoe"):
            continue
        if move.raw.get("multihit") or move.raw.get("overrideOffensivePokemon"):
            continue
        return move_slot.id
    return None


SlotKey = tuple[int, int]


def test_no_particle_is_ruled_out_by_its_own_damage(setup) -> None:  # noqa: ANN001
    """The property that matters most: the truth must survive.

    For a sample of particles, the damage that particle would actually deal is fed back in
    as the observation. Every one of them must keep positive weight -- if any is zeroed,
    the update can permanently exclude the opponent's real spread.
    """
    _scenario, base, active = setup
    reg = _scenario.reg
    attacker_key = next(iter(sorted(active)))
    move_id = _damaging_move(reg, base, attacker_key)
    if move_id is None:
        pytest.skip("no single-target damaging move on the hidden attacker")

    defender_key: SlotKey = (0, 0)
    belief = active[attacker_key]
    attacker = battler_for(reg, base, attacker_key, belief)
    defender = battler(reg, base.sides[0].pokemon[base.sides[0].active[0]])

    result = calculate(
        reg, attacker, defender, move_id, field_state(base, reg), defender_side=0
    )
    if result.immune or result.unmodelled:
        pytest.skip(f"the calculator flags {move_id}: {result.unmodelled}")
    dealt = effective_damage(result, defender)

    rng = np.random.default_rng(0)
    sample = rng.choice(belief.size, size=min(40, belief.size), replace=False)
    for index in sample:
        for roll in (0, 8, 15):
            observed = int(dealt[index, roll])
            values = damage_taken_likelihood(
                reg,
                base,
                belief,
                DamageTaken(
                    attacker=attacker_key,
                    defender=defender_key,
                    move_id=move_id,
                    damage=observed,
                ),
            )
            assert values is not None
            assert values[index] > 0, (
                f"particle {index} was ruled out by the damage it would itself deal "
                f"({observed}); the update can now never recover the true spread"
            )


def test_the_posterior_is_sharper_than_the_prior(setup) -> None:  # noqa: ANN001
    """An observation that carries information has to actually narrow the belief."""
    _scenario, base, active = setup
    reg = _scenario.reg
    attacker_key = next(iter(sorted(active)))
    move_id = _damaging_move(reg, base, attacker_key)
    if move_id is None:
        pytest.skip("no usable move")

    belief = active[attacker_key]
    attacker = battler_for(reg, base, attacker_key, belief)
    defender = battler(reg, base.sides[0].pokemon[base.sides[0].active[0]])
    result = calculate(
        reg, attacker, defender, move_id, field_state(base, reg), defender_side=0
    )
    if result.immune or result.unmodelled:
        pytest.skip("calculator flags this move")
    modal = int(np.argmax(belief.weights))
    observed = int(effective_damage(result, defender)[modal, 8])

    posterior, report = update(
        reg,
        base,
        active,
        [
            DamageTaken(
                attacker=attacker_key,
                defender=(0, 0),
                move_id=move_id,
                damage=observed,
            )
        ],
    )
    assert report.entries, f"the observation was dropped: {report.skipped}"
    label, key, before, after, bits = report.entries[0]
    assert key == attacker_key
    assert after <= before
    assert bits >= -1e-9, "an update cannot increase entropy"
    assert float(posterior[attacker_key].weights.sum()) == pytest.approx(1.0)
    assert posterior[attacker_key].weights[modal] > 0
    del label


def test_an_impossible_damage_is_reported_not_renormalised(setup) -> None:  # noqa: ANN001
    """Zero probability everywhere means the input is wrong, and hiding it would be worse.

    Renormalising nothing produces a confident-looking belief with no evidence behind it.
    """
    _scenario, base, active = setup
    reg = _scenario.reg
    attacker_key = next(iter(sorted(active)))
    move_id = _damaging_move(reg, base, attacker_key)
    if move_id is None:
        pytest.skip("no usable move")

    posterior, report = update(
        reg,
        base,
        active,
        [
            DamageTaken(
                attacker=attacker_key,
                defender=(0, 0),
                move_id=move_id,
                damage=99_999,
            )
        ],
    )
    assert not report.entries
    assert report.skipped and "確率 0" in report.skipped[0]
    # The belief is left exactly as it was rather than quietly replaced.
    assert posterior[attacker_key] is active[attacker_key]


def test_no_particle_is_ruled_out_by_the_bar_it_would_show(setup) -> None:  # noqa: ANN001
    """The same survival property for the defensive channel, which reads a band.

    Here both ends of the observation are percentages, so the likelihood has to average
    over the starting HP the band allows. Getting that wrong zeroes particles that were
    perfectly consistent.
    """
    _scenario, base, active = setup
    reg = _scenario.reg
    defender_key = next(iter(sorted(active)))
    move_id = _damaging_move(reg, base, (0, 0))
    if move_id is None:
        pytest.skip("no single-target damaging move on our side")

    belief = active[defender_key]
    attacker = battler(reg, base.sides[0].pokemon[base.sides[0].active[0]])
    defender = battler_for(reg, base, defender_key, belief)
    result = calculate(
        reg, attacker, defender, move_id, field_state(base, reg), defender_side=1
    )
    if result.immune or result.unmodelled:
        pytest.skip(f"the calculator flags {move_id}: {result.unmodelled}")
    dealt = effective_damage(result, defender)

    rng = np.random.default_rng(1)
    sample = rng.choice(belief.size, size=min(30, belief.size), replace=False)
    for index in sample:
        maxhp = int(defender.maxhp[index])
        hp0 = int(defender.hp[index])
        before = displayed_percent(hp0, maxhp)
        for roll in (0, 8, 15):
            after_hp = max(hp0 - int(dealt[index, roll]), 0)
            values = likelihood_for(
                reg,
                base,
                active,
                DamageDealt(
                    attacker=(0, 0),
                    defender=defender_key,
                    move_id=move_id,
                    percent_before=before,
                    percent_after=displayed_percent(after_hp, maxhp),
                ),
            ).get(defender_key)
            assert values is not None
            assert values[index] > 0, (
                f"particle {index} was ruled out by the bar its own HP would have shown"
            )


def test_turn_order_bounds_speed(setup) -> None:  # noqa: ANN001
    """Moving first at equal priority is an inequality, and a tie is a known coin flip."""
    from pokeuraou.speed import effective_speed

    _scenario, base, active = setup
    reg = _scenario.reg
    hidden = next(iter(sorted(active)))
    belief = active[hidden]
    field = field_state(base, reg)

    ours = battler(reg, base.sides[0].pokemon[base.sides[0].active[0]])
    our_speed = int(effective_speed(reg, ours, field, frozenset())[0])
    their_speeds = np.asarray(
        effective_speed(
            reg, battler_for(reg, base, hidden, belief), field, frozenset()
        ),
        dtype=np.int64,
    )

    values = likelihood_for(
        reg, base, active, MovedFirst(first=hidden, second=(0, 0))
    )[hidden]
    assert (values[their_speeds > our_speed] == 1.0).all()
    assert (values[their_speeds < our_speed] == 0.0).all()
    assert (values[their_speeds == our_speed] == 0.5).all()

    # And the reverse observation is the complement.
    reverse = likelihood_for(
        reg, base, active, MovedFirst(first=(0, 0), second=hidden)
    )[hidden]
    assert values + reverse == pytest.approx(np.ones_like(values))


def test_priority_difference_carries_no_speed_information(setup) -> None:  # noqa: ANN001
    """Fake Out going first says nothing about Speed, so it must not reweight anything."""
    _scenario, base, active = setup
    reg = _scenario.reg
    hidden = next(iter(sorted(active)))
    priority_move = next(
        (
            m.id
            for m in base.sides[hidden[0]].pokemon[
                base.sides[hidden[0]].active[hidden[1]]
            ].moves
            if reg.moves[m.id].priority > 0
        ),
        None,
    )
    if priority_move is None:
        pytest.skip("the hidden Pokemon has no priority move")
    slow_move = _damaging_move(reg, base, (0, 0))
    result = likelihood_for(
        reg,
        base,
        active,
        MovedFirst(
            first=hidden, second=(0, 0),
            first_move=priority_move, second_move=slow_move,
        ),
    )
    assert hidden not in result, "a priority bracket difference was mistaken for Speed"


def test_a_residual_tick_pins_max_hp(setup) -> None:  # noqa: ANN001
    """Sandstorm takes maxhp/16, so watching the bar move constrains HP investment alone.

    Every particle whose max HP reproduces both percentages must survive, and every one
    that cannot must not.
    """
    _scenario, base, active = setup
    reg = _scenario.reg
    hidden = next(iter(sorted(active)))
    belief = active[hidden]

    maxhp = belief.stats[:, 0].astype(np.int64)
    modal = int(np.argmax(belief.weights))
    start = int(maxhp[modal])
    amount = max(start // 16, 1)
    before = displayed_percent(start, start)
    after = displayed_percent(start - amount, start)

    values = likelihood_for(
        reg,
        base,
        active,
        FractionalLoss(
            slot=hidden, numerator=1, denominator=16,
            percent_before=before, percent_after=after, reason="sandstorm",
        ),
    )[hidden]
    assert values[modal] > 0

    for index in range(0, belief.size, max(1, belief.size // 50)):
        m = int(maxhp[index])
        reproduced = displayed_percent(
            m - max(m // 16, 1), m
        ) == after and displayed_percent(m, m) == before
        assert (values[index] > 0) == reproduced, (
            f"particle {index} (max HP {m}) disagrees with the bar it would show"
        )


def _secret_hp_events(log: list[str], ident: str) -> list[tuple[int, int, int]]:
    """(log index, hp, maxhp) for each secret HP line about one Pokemon.

    Showdown emits `|split|SIDE` followed by the secret line and then the shared one, so
    the line right after a split is the exact HP. Reading these rather than diffing the
    position is what makes the damage *attributable*: a drain move on our own side heals us
    in the same turn, so the net HP change is not the damage we took.
    """
    out: list[tuple[int, int, int]] = []
    for index, line in enumerate(log):
        if not line.startswith("|split|") or index + 1 >= len(log):
            continue
        secret = log[index + 1]
        if f"|{ident}:" not in secret:
            continue
        if not (secret.startswith("|-damage|") or secret.startswith("|-heal|")):
            continue
        tail = secret.split("|")[-1].strip()
        if "/" not in tail:
            out.append((index, 0, 0))
            continue
        left, _, right = tail.partition("/")
        right = "".join(ch for ch in right if ch.isdigit())
        if left.isdigit() and right:
            out.append((index, int(left), int(right)))
    return out


def _damage_from_move(
    log: list[str], attacker_ident: str, defender_ident: str, move_name: str, hp_before: int
) -> tuple[int, bool] | None:
    """Damage the named move did, taken from the log rather than from an HP diff."""
    move_index = next(
        (
            i
            for i, line in enumerate(log)
            if line.startswith(f"|move|{attacker_ident}:")
            and move_name.lower().replace(" ", "").replace("-", "")
            in line.split("|")[3].lower().replace(" ", "").replace("-", "")
        ),
        None,
    )
    if move_index is None:
        return None
    events = _secret_hp_events(log, defender_ident)
    previous = hp_before
    for index, hp, maxhp in events:
        del maxhp
        if index > move_index:
            fainted = hp <= 0
            return previous - hp, fainted
        previous = hp
    return None


@pytest.mark.oracle
def test_the_true_spread_survives_a_real_turn(reg: Regulation) -> None:
    """End to end against Showdown: the opponent's actual spread must keep weight.

    The damage is read out of the simulator's own positions, so this is the update facing
    a number the game produced rather than one our calculator produced. If the true
    spread is ever zeroed here, the belief layer is unusable no matter how sharp it looks.
    """
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        pytest.skip("no cached usage stats")
    from pokeuraou.belief import build_belief

    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    rng = np.random.default_rng(19)
    py_rng = random.Random(19)

    checked = 0
    with Oracle() as oracle:
        for _ in range(6):
            if checked:
                break
            teams = [
                [TeamSet.from_json(s.to_team_set_json(reg)) for s in sample_team(rng, reg, prior)]
                for _ in range(2)
            ]
            handle = oracle.create(
                FORMAT_ID, teams[0], teams[1],
                seed=tuple(int(rng.integers(1, 60000)) for _ in range(4)),  # type: ignore[arg-type]
                policy=RandomnessPolicy(damage_roll=8, crit=False, secondary=False),
            )
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])
            before = Position.from_json(handle.position)

            # One attack from their left slot at our left slot, and something harmless
            # from us, so the HP change we see has exactly one cause.
            theirs = next(
                (
                    a
                    for a in side_actions(reg, before, 1)
                    if getattr(a.slots[0], "move_id", None)
                    and reg.moves[a.slots[0].move_id].category != "Status"
                    and getattr(a.slots[0], "target", None) == 1
                    and not reg.moves[a.slots[0].move_id].raw.get("multihit")
                ),
                None,
            )
            if theirs is None:
                handle.close()
                continue
            # A switch or a Protect would leave the HP change with a different cause, so
            # our own action is pinned to an ordinary attack.
            our_options = [
                a
                for a in side_actions(reg, before, 0)
                if getattr(a.slots[0], "move_id", None)
                and reg.moves[a.slots[0].move_id].category != "Status"
                and not reg.moves[a.slots[0].move_id].raw.get("stallingMove")
            ]
            if not our_options:
                handle.close()
                continue
            ours = py_rng.choice(our_options)
            handle.step([ours.to_choice(), theirs.to_choice()])
            if handle.choice_errors:
                handle.close()
                continue
            our_mon_before = before.sides[0].pokemon[before.sides[0].active[0]]
            move_id = theirs.slots[0].move_id
            attributed = _damage_from_move(
                # `handle.log` holds this step's protocol only, not the whole battle.
                handle.log,
                "p2a",
                "p1a",
                reg.moves[move_id].name,
                our_mon_before.hp,
            )
            if attributed is None or attributed[0] <= 0:
                handle.close()
                continue
            # A KO reports the HP that was there, not the roll, so the observation becomes
            # "at least this much".
            damage, capped = attributed

            # Their real set is known here because we built the teams, so the belief can
            # be checked against ground truth.
            attacker_true = before.sides[1].pokemon[before.sides[1].active[0]]
            belief = build_belief(reg, prior, attacker_true.species, attacker_true.nature)
            truth = np.array(
                [attacker_true.sp[k] for k in ("hp", "atk", "def", "spa", "spd", "spe")],
                dtype=np.int64,
            )
            matches = np.flatnonzero((belief.spreads == truth).all(axis=1))
            if matches.size == 0:
                handle.close()
                continue

            hidden = before.copy()
            hidden.sides[1].pokemon[hidden.sides[1].active[0]].sp = None
            values = damage_taken_likelihood(
                reg, hidden, belief,
                DamageTaken(
                    attacker=(1, 0), defender=(0, 0), move_id=move_id,
                    damage=damage, capped=capped,
                ),
            )
            if values is None:
                handle.close()
                continue
            assert values[matches[0]] > 0, (
                f"{attacker_true.species}'s true spread {truth.tolist()} was ruled out by "
                f"the {damage} damage its own {move_id} did in Showdown"
            )
            checked += 1
            handle.close()

    if not checked:
        pytest.skip("no clean single-cause damage turn was produced")
