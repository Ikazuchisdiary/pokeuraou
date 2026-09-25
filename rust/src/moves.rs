//! Move execution and the end-of-turn phase, transcribed from `src/pokeuraou/resolve.py`.
//!
//! Split from `resolve.rs` only for length; the state it works on is `resolve::Turn`.
//! Every path that meets something this port does not model returns `Err(reason)` so the
//! caller stops rather than take a wrong answer (it fell back to Python until IKA-209; the
//! Python resolver is gone since IKA-212 -- see the head of `resolve.rs` for what "Python's
//! `_x`" in these comments names).

use crate::battler::Battler;
use crate::damage::{calculate, crit_probability};
use crate::effects::{is_mold_breaker, resist_berry};
use crate::id::Id;
use crate::moveinfo::MoveContext;
use crate::position::{Effect, Position, Types};
use crate::reg::{
    Move, Reg, F_BYPASSSUB, F_CONTACT, F_DEFROST, F_FAILENCORE, F_POWDER, F_PROTECT,
};
use crate::resolve::{
    change_forme, check_white_herb, grounded_ignoring, stratified_rolls, Budget, Label, Name,
    Outcome, Slot, Turn,
    CONFUSION_SELF_HIT_CHANCE, FREEZE_COUNTER, FULL_PARALYSIS_CHANCE, THAW_CHANCE,
    TWO_TURN_MOVES,
};
use crate::speed::QueuedAction;
use serde_json::{json, Value};

const RECHARGE: &str = "recharge";

/// Protect-family volatiles, and what each blocks. Endure is deliberately absent: it is a
/// stalling move that shares the counter, but its volatile caps damage rather than blocking
/// the hit, so it is handled in `deal_damage` instead.
const PROTECT_VOLATILES: [(&str, &str); 9] = [
    ("protect", "all"),
    ("detect", "all"),
    ("banefulbunker", "all"),
    ("burningbulwark", "all"),
    ("spikyshield", "all"),
    ("kingsshield", "damaging"),
    ("obstruct", "damaging"),
    ("silktrap", "damaging"),
    ("maxguard", "all"),
];

const FIRST_TURN_OUT_MOVES: [&str; 3] = ["fakeout", "firstimpression", "matblock"];

/// Python's `TYPE_SPENDING_MOVES` (IKA-162): the move fails unless its user has the type,
/// and once it has hit, the type is rewritten to `SPENT_TYPE`.
fn spent_type(move_id: &str) -> Option<&'static str> {
    match move_id {
        "doubleshock" => Some("Electric"),
        "burnup" => Some("Fire"),
        _ => None,
    }
}

/// What Showdown writes in place of a spent type. The type chart has no slot for it, so
/// `type_slot_of` makes it neutral, as it is in Showdown.
const SPENT_TYPE: &str = "???";
const STALL_BUMPING_MOVES: [&str; 2] = ["wideguard", "quickguard"];
const MAX_BRANCHED_SECONDARIES: usize = 2;

const CHARGE_SPA: &[(&str, i64)] = &[("spa", 1)];
const CHARGE_GEOMANCY: &[(&str, i64)] = &[("spa", 2), ("spd", 2), ("spe", 2)];

/// Stat changes a charge turn brings with it. Electro Shot and Meteor Beam raise Special
/// Attack while winding up, which is most of the reason to use them.
fn charge_turn_boosts(move_id: &str) -> Option<&'static [(&'static str, i64)]> {
    match move_id {
        "electroshot" | "meteorbeam" => Some(CHARGE_SPA),
        "geomancy" => Some(CHARGE_GEOMANCY),
        _ => None,
    }
}

fn is(value: Option<Id>, name: &str) -> bool {
    matches!(value, Some(v) if v.as_str() == name)
}

// ---------------------------------------------------------------------------
// One move
// ---------------------------------------------------------------------------

pub(crate) fn do_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.ok_or("a move action with no move")?;
    // The turn after Hyper Beam: nothing happens, and the lock lifts. Showdown's
    // `mustrecharge` condition has `onBeforeMovePriority: 11` and an `onBeforeMove` that
    // adds the `cant` line, removes itself (and Truant's volatile) and returns null.
    // Priority 11 is above sleep's 10 and flinch's 8, so the recharge is spent even by a
    // Pokemon that could not have moved anyway -- which is why this sits ahead of `can_act`.
    // No PP is spent: there is no move slot to spend it from, and Showdown's request
    // confirms it (Hyper Beam stays at 7 of 8 across the recharge turn). Truant is not
    // modelled; the volatile's `duration: 2` is not tracked either, since the only way to
    // hold it past this point is to be forced out, and a switch clears volatiles anyway.
    // (Showdown's quoted ids are left unquoted here on purpose: `tools/port_coverage.py`
    // reads a quoted id anywhere in the port as the port naming it.)
    if move_id.as_str() == RECHARGE {
        if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
            mon.volatiles.retain(|v| v.id.as_str() != "mustrecharge");
            // IKA-210's positive control puts the recharge lock back, for good.
            if cfg!(feature = "ika210-control") {
                mon.volatiles.push(Effect::new(Id::new("mustrecharge")));
            }
        }
        log_event!(turn, "{} must recharge", Name(action.side, action.slot));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    let mv = reg.moves.get(move_id.as_str()).ok_or("move not in the regulation")?;

    // `_roll_rampage`: a rampage's length, branched on its second turn (IKA-174).
    if let Some(rolled) = roll_rampage(&mut turn, action, &budget) {
        let mut outcomes: Vec<Outcome<'a>> = Vec::new();
        for (weight, state) in rolled {
            for (inner, sub_state) in do_move(reg, state, action, budget)? {
                outcomes.push((weight * inner, sub_state));
            }
        }
        return Ok(outcomes);
    }
    // `_roll_confusion`: whether this try ends an unrolled confusion (IKA-177).
    if let Some(tried) = roll_confusion(&mut turn, action, mv, &budget) {
        let mut outcomes: Vec<Outcome<'a>> = Vec::new();
        for (weight, state) in tried {
            for (inner, sub_state) in do_move(reg, state, action, budget)? {
                outcomes.push((weight * inner, sub_state));
            }
        }
        return Ok(outcomes);
    }

    let started = crate::resolve::phase_start();
    let checks = can_act(&mut turn, action, mv, &budget)?;
    crate::resolve::phase_end(7, started);
    let mut outcomes: Vec<Outcome<'a>> = Vec::new();
    // The last check cannot hand `turn` over instead of copying it, however tempting: the
    // fallback below needs it if every branch turns out to weigh nothing, and a move that
    // produced no outcome at all is a thing Python answers rather than refuses.
    for (probability, blocked) in checks.iter() {
        if *probability <= 0.0 {
            continue;
        }
        let mut state = turn.clone();
        match blocked {
            Some(reason) => {
                // `runMove` bumps `activeMoveActions` before `BeforeMove`, so a Pokemon
                // that could not move has still spent its first turn out (IKA-166).
                if reason.as_str() != "fainted" {
                    if let Some(mon) = state.mon_at_mut(action.side, action.slot) {
                        mon.active_move_actions += 1;
                    }
                }
                if reason.as_str() == "flinch" {
                    if let Some(mon) = state.mon_at_mut(action.side, action.slot) {
                        mon.volatiles.retain(|v| v.id.as_str() != "flinch");
                    }
                }
                let hits = if reason.as_str() == "confusion" {
                    confusion_self_hits(state, action, &budget)?
                } else {
                    vec![(1.0, state)]
                };
                for (hit_weight, mut hit_state) in hits {
                    log_event!(hit_state, "{} did not happen ({})", Label(reg, action), reason);
                    hit_state.move_failed[action.side][action.slot] = true;
                    outcomes.push((*probability * hit_weight, hit_state));
                }
            }
            None => {
                for (drawn, drawn_state, drawn_action) in
                    draw_random_target(state, action, mv, &budget)
                {
                    let started = crate::resolve::phase_start();
                    let produced = use_move(reg, drawn_state, &drawn_action, mv, budget)?;
                    crate::resolve::phase_end(8, started);
                    for (weight, mut sub_state) in produced {
                        rampage_after_move(&mut sub_state, &drawn_action);
                        outcomes.push((probability * drawn * weight, sub_state));
                    }
                }
            }
        }
    }
    if outcomes.is_empty() {
        outcomes.push((1.0, turn));
    }
    Ok(outcomes)
}

/// The foe a `randomNormal` move is used at, as Python's `_draw_random_target` (IKA-178):
/// `getTarget` ignores the chosen location for it and `sample`s the foes standing, every
/// time the move is used. Each is a branch of equal weight, carried as the action's target
/// for `resolve_targets`, which redirects it as any other. One foe standing is no draw; a
/// budget that collapses the random ranges takes the first (with a note), and so does the
/// pinned one.
fn draw_random_target<'a>(
    turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Vec<(f64, Turn<'a>, QueuedAction)> {
    if mv.target.as_str() != "randomNormal" {
        return vec![(1.0, turn, action.clone())];
    }
    let mut turn = turn;
    let foe_side = 1 - action.side;
    let mut standing: Vec<usize> = (0..turn.pos.sides[foe_side].active.len())
        .filter(|s| matches!(turn.mon_at(foe_side, *s), Some(mon) if !mon.fainted))
        .collect();
    if standing.len() > 1 && !(budget.enumerate_secondary && !budget.pinned_policy) {
        if !budget.pinned_policy {
            turn.report("randomNormal target (the first foe; not branched)");
        }
        standing.truncate(1);
    }
    if standing.is_empty() {
        let mut aimed = action.clone();
        aimed.target = None;
        return vec![(1.0, turn, aimed)];
    }
    let share = 1.0 / standing.len() as f64;
    let aimed = |slot: usize| {
        let mut aimed = action.clone();
        aimed.target = Some(slot as i64 + 1);
        aimed
    };
    let (&last, rest) = standing.split_last().expect("not empty");
    let mut out: Vec<(f64, Turn<'a>, QueuedAction)> =
        rest.iter().map(|slot| (share, turn.clone(), aimed(*slot))).collect();
    out.push((share, turn, aimed(last)));
    out
}

/// `getConfusionDamage`: then `trunc(baseDamage, 16)`, `randomizer` at roll `r` (`100 - r`
/// percent) and at least 1, as Python's `_confusion_damage` (IKA-189).
fn confusion_damage(turn: &Turn, side: usize, slot: usize, roll: usize) -> Result<i64, String> {
    let Some(mon) = turn.battler_at(side, slot)? else { return Ok(0) };
    let attack = mon.stat("atk", false);
    let defence = mon.stat("def", false).max(1);
    let level_term = (2.0 * mon.level as f64 / 5.0 + 2.0).trunc() as i64;
    let inner = (level_term * 40 * attack) as f64;
    let base = ((inner.trunc() / defence as f64).trunc() / 50.0).trunc() as i64;
    let damage = (base + 2).rem_euclid(65536);
    Ok((damage * (100 - roll as i64) / 100).max(1))
}

/// The self-hit once per damage roll the budget keeps (`stratified_rolls`), as Python's
/// `_confusion_self_hits` (IKA-189).
fn confusion_self_hits<'a>(
    turn: Turn<'a>,
    action: &QueuedAction,
    budget: &Budget,
) -> Result<Vec<(f64, Turn<'a>)>, String> {
    let rolls = stratified_rolls(budget);
    let mut out = Vec::with_capacity(rolls.len());
    let mut last = Some(turn);
    for (index, (roll, weight)) in rolls.iter().enumerate() {
        let mut state = if index + 1 == rolls.len() {
            last.take().expect("the last roll takes the turn")
        } else {
            last.as_ref().expect("taken only at the last roll").clone()
        };
        let amount = confusion_damage(&state, action.side, action.slot, *roll)?;
        state.deal_damage(action.side, action.slot, amount, false, "confusion")?;
        out.push((*weight, state));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Outrage's rampage (`lockedmove`), as Python's `_start_rampage` and the rest (IKA-174)
// ---------------------------------------------------------------------------

/// Showdown's own `effectState` key for the rampage's length, carried in `extra`.
const RAMPAGE_LEFT: &str = "trueDuration";

fn rampage_left(effect: &Effect) -> Option<i64> {
    effect.extra.get(RAMPAGE_LEFT).and_then(Value::as_i64)
}

fn set_rampage_left(effect: &mut Effect, left: i64) {
    let mut extra = (*effect.extra).clone();
    extra.insert(RAMPAGE_LEFT.into(), json!(left));
    effect.extra = std::rc::Rc::new(extra);
}

/// The move a rampage holds; a bare recorded `lockedmove` holds none.
pub(crate) fn rampage_move(mon: &crate::position::Pokemon) -> Option<Id> {
    mon.volatile("lockedmove").and_then(|held| held.move_id)
}

/// `onStart` (duration 2, the move; the roll waits for `roll_rampage`) or `onRestart`
/// (duration back to 2 while `trueDuration >= 2`).
///
/// `self: {volatileStatus: 'lockedmove'}` landing: Outrage, Petal Dance, Raging Fury,
/// Thrash. `after_move` calls it only once the move reached a target, which is `selfDrops`.
/// `onStart` also rolls `trueDuration = random(2, 4)`, but nothing reads it until the second
/// turn, so `roll_rampage` branches it there, where the two lengths first differ (continue,
/// or stop and be confused) -- rather than doubling every first turn's leaves into pairs no
/// one-ply payoff or encoding can tell apart. A bare recorded one (before IKA-174, with no
/// move or length) is replaced, not restarted.
pub(crate) fn start_rampage(turn: &mut Turn, side: usize, slot: usize) {
    let Some(mon) = turn.mon_at_mut(side, slot) else { return };
    if mon.fainted {
        return;
    }
    if let Some(held) = mon.volatile_mut("lockedmove") {
        if held.move_id.is_some() {
            if matches!(rampage_left(held), Some(left) if left >= 2) {
                held.duration = Some(2);
            }
            return;
        }
    }
    mon.volatiles.retain(|v| v.id.as_str() != "lockedmove");
    let mut effect = Effect::new(Id::new("lockedmove"));
    effect.duration = Some(2);
    effect.move_id = mon.last_move;
    let last_move = mon.last_move;
    mon.volatiles.push(effect);
    log_event!(turn, "{} is rampaging ({})", Name(side, slot), OrNone(last_move));
}

/// The length on the second turn when the position lacks it: 1 or 2 left, a half each.
///
/// `random(2, 4)` at the start, one taken off by the first turn's residual: 1 left is a
/// two-turn rampage, 2 a three-turn one. Showdown's own positions carry it and never branch
/// here. A budget that collapses the random ranges takes the short one, which is also what
/// the oracle's pinned `random(a, b)` answers (as `multihit_counts`).
fn roll_rampage<'a>(
    turn: &mut Turn<'a>,
    action: &QueuedAction,
    budget: &Budget,
) -> Option<Vec<(f64, Turn<'a>)>> {
    let unrolled = match turn.mon_at(action.side, action.slot) {
        Some(mon) if !mon.fainted => matches!(
            mon.volatile("lockedmove"),
            Some(held) if held.move_id.is_some()
                && held.duration == Some(1)
                && rampage_left(held).is_none()
        ),
        _ => false,
    };
    if !unrolled {
        return None;
    }
    let options: Vec<(f64, i64)> = if budget.enumerate_secondary && !budget.pinned_policy {
        vec![(0.5, 1), (0.5, 2)]
    } else {
        if !budget.pinned_policy {
            turn.report("rampage length (the two-turn one of 2-or-3; not branched)");
        }
        vec![(1.0, 1)]
    };
    let mut out = Vec::new();
    for (weight, left) in options {
        let mut state = turn.clone();
        if let Some(lock) = state
            .mon_at_mut(action.side, action.slot)
            .and_then(|m| m.volatile_mut("lockedmove"))
        {
            set_rampage_left(lock, left);
        }
        out.push((weight, state));
    }
    Some(out)
}

/// `onEnd`'s test, `trueDuration > 1` returning early: whether the rampage ran its full
/// length, which is when it confuses.
fn rampage_last_turn(turn: &mut Turn, left: Option<i64>) -> bool {
    match left {
        Some(left) => left <= 1,
        None => {
            turn.report("rampage length (not on the position; read as its last turn)");
            true
        }
    }
}

/// `onAfterMove`: `if (duration === 1) removeVolatile('lockedmove')`, then `onEnd`.
///
/// It runs after every move the Pokemon made, hit or not -- a Protect on the second turn
/// skipped the restart, so the rampage ends there. A Pokemon that could not move (flinch,
/// sleep, paralysis, its own confusion) never gets here; its rampage runs out at the
/// residual instead (`rampage_runs_out`).
fn rampage_after_move(turn: &mut Turn, action: &QueuedAction) {
    let left = match turn.mon_at(action.side, action.slot).and_then(|m| m.volatile("lockedmove")) {
        Some(held) if held.move_id.is_some() && held.duration == Some(1) => rampage_left(held),
        _ => return,
    };
    if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
        mon.volatiles.retain(|v| v.id.as_str() != "lockedmove");
    }
    log_event!(turn, "{}'s rampage ended", Name(action.side, action.slot));
    if rampage_last_turn(turn, left) {
        confused_by_fatigue(turn, action.side, action.slot);
    }
}

/// The rampage's `onEnd` confusion: Own Tempo and grounded under Misty Terrain refuse it;
/// the berries are every confusion's, in `start_confusion` (IKA-177). Safeguard does not,
/// since the rampage has no source other than itself.
fn confused_by_fatigue(turn: &mut Turn, side: usize, slot: usize) {
    match turn.mon_at(side, slot) {
        Some(mon) if !mon.fainted && !mon.has_volatile("confusion") => {}
        _ => return,
    }
    log_event!(turn, "{} tires (fatigue)", Name(side, slot));
    // `addVolatile` fills in `source = this`: the rampager is its own source (IKA-189).
    confuse(turn, side, slot, Some((side, slot)));
}

/// `addVolatile('confusion', source)`: `TryAddVolatile`, `start_confusion`, then Own
/// Tempo's `onUpdate` for a Mold Breaker move that got past it -- Python's `_confuse`.
///
/// Every confusion goes through here -- a status move's, a secondary's, the fatigue's. Only
/// a Mold Breaker move gets past Own Tempo's `onTryAddVolatile`: the confusion then starts,
/// a Persim or Lum Berry eats it, and otherwise `onUpdate` cures it at once
/// (vendor/pokemon-showdown/data/abilities.ts `owntempo`; the oracle eats the berry).
pub(crate) fn confuse(turn: &mut Turn, side: usize, slot: usize, source: Option<Slot>) {
    match turn.mon_at(side, slot) {
        Some(mon) if !mon.fainted && !mon.has_volatile("confusion") => {}
        _ => return,
    }
    if let Some(refused) = confusion_refused(turn, side, slot, source) {
        log_event!(turn, "{} is not confused ({})", Name(side, slot), refused);
        return;
    }
    start_confusion(turn, side, slot);
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        if mon.ability == "owntempo" && mon.has_volatile("confusion") {
            mon.volatiles.retain(|v| v.id.as_str() != "confusion");
            log_event!(turn, "{} snapped out of its confusion (owntempo)", Name(side, slot));
        }
    }
}

/// Python's `f"{x}"` of an optional id: the id, or `None`.
pub(crate) struct OrNone(pub Option<Id>);

impl std::fmt::Display for OrNone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(id) => write!(f, "{id}"),
            None => f.write_str("None"),
        }
    }
}

/// Disguise's `onUpdate` after the hit it took: `formeChange('Mimikyu-Busted', ..., true)`
/// and `this.damage(pokemon.baseMaxhp / 8, ...)` (data/abilities.ts; Python's
/// `_bust_disguise`, IKA-208).
fn bust_disguise(reg: &Reg, turn: &mut Turn, target: Slot) -> Result<(), String> {
    let maxhp = match turn.mon_at(target.0, target.1) {
        Some(mon) if !mon.fainted && mon.species.as_str() == "mimikyu" => mon.maxhp,
        _ => return Ok(()),
    };
    change_forme(reg, turn, target.0, target.1, "mimikyubusted")?;
    turn.deal_damage(target.0, target.1, (maxhp / 8).max(1), false, "disguise")?;
    Ok(())
}

/// What Dragon Darts does with its target (IKA-208).
enum SmartHits {
    /// The ordinary two-hit move at these targets: one foe, or none that it can reach.
    Ordinary(Vec<Slot>),
    /// A hit on each of two foes, or -- where one of them fails a hit step -- both hits on
    /// the other, one plan per accuracy outcome with its weight.
    Split(Vec<(f64, Vec<(Slot, usize)>)>),
}

/// `smartTarget` (sim/pokemon.ts `getSmartTargets`, sim/battle-actions.ts): the target and
/// its adjacent ally, when that ally stands and is not the user; the hit steps run on both,
/// and any failure among them -- Protect, an immunity, a miss -- turns `smartTarget` off,
/// leaving the move's two hits to whoever is left. With both still there, hit 1 goes to
/// the first and hit 2 to the second, each a single-target hit (no spread).
fn smart_hits(
    reg: &Reg,
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
    budget: &Budget,
) -> Result<SmartHits, String> {
    let ordinary = || Ok(SmartHits::Ordinary(targets.to_vec()));
    let [first] = targets else { return ordinary() };
    let first = *first;
    let second = (first.0, 1 - first.1);
    let standing = |at: Slot| matches!(turn.mon_at(at.0, at.1), Some(m) if !m.fainted && m.hp > 0);
    if second == (action.side, action.slot) || !standing(second) || !standing(first) {
        return ordinary();
    }
    let Some(attacker) = turn.battler_at(action.side, action.slot)? else { return ordinary() };
    // Protect (step 3) and the type immunity (step 2), per target.
    let mut reached: Vec<(Slot, f64)> = Vec::new();
    for at in [first, second] {
        if blocked_by_protect(turn, action, mv, at).is_some() {
            continue;
        }
        let Some(defender) = turn.battler_at(at.0, at.1)? else { continue };
        let field = turn.field();
        let probe = calculate(reg, &attacker, &defender, mv.id.as_str(), &field, at.0, false, false, None, None, false);
        if probe.immune {
            continue;
        }
        let accuracy = accuracy_of(turn, mv, &attacker, &defender);
        let accuracy = if budget.enumerate_accuracy { accuracy } else if accuracy > 0.0 { 1.0 } else { 0.0 };
        reached.push((at, accuracy));
    }
    match reached.as_slice() {
        [(a, pa), (b, pb)] => {
            let (a, b, pa, pb) = (*a, *b, *pa, *pb);
            let plans = vec![
                (pa * pb, vec![(a, 1), (b, 1)]),
                (pa * (1.0 - pb), vec![(a, 2)]),
                ((1.0 - pa) * pb, vec![(b, 2)]),
                ((1.0 - pa) * (1.0 - pb), vec![]),
            ];
            Ok(SmartHits::Split(plans.into_iter().filter(|(w, _)| *w > 0.0).collect()))
        }
        // One left: the ordinary move at it, whose accuracy `hit_target` rolls.
        [(only, _)] => Ok(SmartHits::Ordinary(vec![*only])),
        _ => ordinary(),
    }
}

/// Damp's `onAnyTryMove`: any active Pokemon with it stops Explosion, Self-Destruct,
/// Misty Explosion (and Mind Blown) -- `breakable`, so not against a Mold Breaker's own
/// blast (IKA-208).
fn damp_stops(turn: &Turn, action: &QueuedAction) -> bool {
    let source = Some((action.side, action.slot));
    (0..turn.pos.sides.len()).any(|side| {
        (0..turn.pos.sides[side].active.len()).any(|slot| {
            matches!(turn.mon_at(side, slot), Some(mon)
                if !mon.fainted && mon.ability == "damp" && !ability_broken_by(turn, mon, source))
        })
    })
}

/// `suppressingAbility(mon)` under `source`'s move: Python's `_ability_broken_by`.
fn ability_broken_by(turn: &Turn, mon: &crate::position::Pokemon, source: Option<Slot>) -> bool {
    let Some((side, slot)) = source else { return false };
    let Some(user) = turn.mon_at(side, slot) else { return false };
    if std::ptr::eq(user, mon) || !is_mold_breaker(user.ability.as_str()) {
        return false;
    }
    if is(mon.item, "abilityshield") {
        return false;
    }
    if user.ability == "myceliummight" {
        return user
            .last_move
            .and_then(|m| turn.reg.moves.get(m.as_str()))
            .is_some_and(|mv| mv.category == "Status");
    }
    true
}

/// Own Tempo, Misty Terrain on the grounded, Safeguard against another's move unless it
/// infiltrates: Python's `_confusion_refused` (IKA-189).
fn confusion_refused(turn: &Turn, side: usize, slot: usize, source: Option<Slot>) -> Option<&'static str> {
    let mon = turn.mon_at(side, slot)?;
    let broken = ability_broken_by(turn, mon, source);
    if mon.ability == "owntempo" && !broken {
        return Some("owntempo");
    }
    if is(turn.pos.field.terrain, "mistyterrain") && grounded_ignoring(turn, mon, broken) {
        return Some("mistyterrain");
    }
    match source {
        Some(from) if from != (side, slot) && turn.pos.sides[side].has_side_condition("safeguard") => {
            let infiltrates = from.0 != side
                && turn.mon_at(from.0, from.1).is_some_and(|user| user.ability == "infiltrator");
            (!infiltrates).then_some("safeguard")
        }
        _ => None,
    }
}

/// Good as Gold's `onTryHit` for another Pokemon's status move, unless a Mold Breaker
/// move: Python's `_good_as_gold_blocks` (IKA-202).
pub(crate) fn good_as_gold_blocks(turn: &Turn, action: &QueuedAction, mv: &Move, target: Slot) -> bool {
    let me = (action.side, action.slot);
    if mv.category != "Status" || target == me {
        return false;
    }
    turn.mon_at(target.0, target.1).is_some_and(|mon| {
        mon.ability == "goodasgold" && !ability_broken_by(turn, mon, Some(me))
    })
}

/// A Grass type beside (or holding) a Flower Veil no `current_actor` Mold Breaker move
/// passes: Python's `_flower_veil` (IKA-202). The callers check the source. The slot of
/// the ally that holds it, which the trace names (IKA-215).
pub(crate) fn flower_veil_holder(turn: &Turn, side: usize, slot: usize) -> Option<usize> {
    let mon = turn.mon_at(side, slot)?;
    if mon.fainted || !turn.types_of(mon).contains("Grass") {
        return None;
    }
    (0..turn.pos.sides[side].active.len()).find(|&ally| {
        turn.mon_at(side, ally).is_some_and(|holder| {
            !holder.fainted
                && holder.ability == "flowerveil"
                && !ability_broken_by(turn, holder, turn.current_actor)
        })
    })
}

/// The residual's `duration--` reaching zero: `onEnd` before the loop takes it off.
fn rampage_runs_out(turn: &mut Turn, order: &[Slot]) {
    for (side, slot) in order.iter().copied() {
        let left = match turn.mon_at(side, slot) {
            Some(mon) if !mon.fainted => match mon.volatile("lockedmove") {
                Some(held) if held.move_id.is_some() && held.duration == Some(1) => {
                    rampage_left(held)
                }
                _ => continue,
            },
            _ => continue,
        };
        log_event!(turn, "{}'s rampage ran out", Name(side, slot));
        if rampage_last_turn(turn, left) {
            confused_by_fatigue(turn, side, slot);
        }
    }
}

/// `onResidual` for a rampage that goes on: asleep it stops unconfused, else
/// `trueDuration--`. After the loop, so Yawn's sleep has landed.
fn rampage_residual(turn: &mut Turn, order: &[Slot]) {
    for (side, slot) in order.iter().copied() {
        let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
        if mon.fainted {
            continue;
        }
        let asleep = is(mon.status, "slp");
        let Some(held) = mon.volatile_mut("lockedmove") else { continue };
        if held.move_id.is_none() {
            continue;
        }
        if asleep {
            mon.volatiles.retain(|v| v.id.as_str() != "lockedmove");
            log_event!(turn, "{}'s rampage ended (asleep)", Name(side, slot));
            continue;
        }
        if let Some(left) = rampage_left(held) {
            set_rampage_left(held, left - 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Confusion's length and the berries, as Python's `_start_confusion` and the rest (IKA-177)
// ---------------------------------------------------------------------------

/// Showdown's own `effectState.time`: the tries left, cured at zero.
const CONFUSION_LEFT: &str = "time";
/// Ours while the length is not rolled: the tries so far, and Axe Kick's `min = 3`.
const CONFUSION_TRIES: &str = "tries";
const CONFUSION_MIN: &str = "min";
/// `roll_confusion`'s branch where this try does not cure; `confusion_try` takes it off.
const CONFUSION_GOES_ON: &str = "goesOn";
/// `random(min, 6)` is at most 5.
const CONFUSION_LONGEST: i64 = 5;

fn extra_int(effect: &Effect, key: &str) -> Option<i64> {
    effect.extra.get(key).and_then(Value::as_i64)
}

fn set_extra(effect: &mut Effect, key: &str, value: Option<Value>) {
    let mut extra = (*effect.extra).clone();
    match value {
        Some(value) => {
            extra.insert(key.into(), value);
        }
        None => {
            extra.remove(key);
        }
    }
    effect.extra = std::rc::Rc::new(extra);
}

/// `addVolatile('confusion')`: `onStart` (the roll waits for `roll_confusion`; tries are
/// counted meanwhile), then a Persim or Lum Berry's `onUpdate` unless Unnerve forbids.
///
/// `onStart` rolls `time = random(2, 6)`, but nothing reads it before a try where the
/// lengths first differ, so the roll waits -- as the rampage's length waits for its second
/// turn (IKA-174). The berries' `onUpdate` eats whenever `volatiles['confusion']`, whatever
/// confused the holder (the resolver once ate them only for a rampage's fatigue).
pub(crate) fn start_confusion(turn: &mut Turn, side: usize, slot: usize) {
    let berry = {
        let Some(mon) = turn.mon_at_mut(side, slot) else { return };
        if mon.fainted || mon.has_volatile("confusion") {
            return;
        }
        let mut effect = Effect::new(Id::new("confusion"));
        set_extra(&mut effect, CONFUSION_TRIES, Some(json!(0)));
        mon.volatiles.push(effect);
        mon.item.filter(|i| i.as_str() == "persimberry" || i.as_str() == "lumberry")
    };
    log_event!(turn, "{} became confused", Name(side, slot));
    if let Some(berry) = berry.filter(|_| !turn.berries_blocked(side)) {
        turn.consume_item(side, slot, berry.as_str());
        if let Some(mon) = turn.mon_at_mut(side, slot) {
            mon.volatiles.retain(|v| v.id.as_str() != "confusion");
        }
    }
}

/// `const min = sourceEffect?.id === 'axekick' ? 3 : 2`.
fn confused_by_axe_kick(turn: &mut Turn, target: Slot) {
    if let Some(held) = turn
        .mon_at_mut(target.0, target.1)
        .and_then(|m| m.volatile_mut("confusion"))
    {
        set_extra(held, CONFUSION_MIN, Some(json!(3)));
    }
}

/// `move.flags['defrost'] && !(move.id === 'burnup' && !pokemon.hasType('Fire'))`.
fn defrosts(turn: &Turn, mon: &crate::position::Pokemon, mv: &Move) -> bool {
    mv.has_flag(F_DEFROST) && !(mv.id == "burnup" && !turn.types_of(mon).contains("Fire"))
}

/// Whether this try gets as far as confusion's `onBeforeMove`: past a flinch, and past a
/// sleep or a freeze only when it wakes or thaws for certain.
///
/// Confusion's `onBeforeMove` is priority 3, below a flinch (8), a sleep (10) and a freeze
/// (10). A freeze left to its 1-in-4 is not reached, as `can_act` does not look at
/// confusion there.
fn confusion_reached(turn: &Turn, mon: &crate::position::Pokemon, mv: &Move) -> bool {
    if mon.fainted || mon.has_volatile("flinch") || taunt_stops(mon, mv) {
        return false;
    }
    if is(mon.status, "slp") {
        let step = if mon.ability == "earlybird" { 2 } else { 1 };
        return mon.status_counter.unwrap_or(0) - step <= 0;
    }
    if is(mon.status, "frz") {
        return defrosts(turn, mon, mv) || mon.status_counter.unwrap_or(FREEZE_COUNTER) - 1 <= 0;
    }
    true
}

/// The chance this try ends the confusion given the ones before did not: 0 before `min`,
/// then one in `6 - k` at try k; the shortest length when the checks are collapsed.
///
/// `time` is uniform on min..5 and try k cures when `time == k`, so after k - 1 tries it is
/// uniform on max(min, k)..5, and try k cures one time in `6 - k` once `k >= min`: 0, 1/4,
/// 1/3, 1/2, 1 for the plain confusion. The shortest length is also the oracle's pinned
/// `random(a, b)`, `a`.
fn confusion_cure_chance(held: &Effect, budget: &Budget) -> f64 {
    let tries = extra_int(held, CONFUSION_TRIES).unwrap_or(0) + 1;
    let low = extra_int(held, CONFUSION_MIN).unwrap_or(2);
    if tries < low {
        return 0.0;
    }
    if !budget.enumerate_status_checks || budget.pinned_policy {
        return 1.0;
    }
    1.0 / ((CONFUSION_LONGEST + 1 - tries).max(1) as f64)
}

/// Python's `_roll_confusion`: cured now (`time` 1) or one more try, when the position
/// does not carry the length.
///
/// At the head of the move, as `roll_rampage`, because the two outcomes are different
/// states and `can_act` answers weights. Branching per try rather than rolling the whole
/// length once keeps the leaves to two a try; a length rolled at the first try that can end
/// would split the rest into three or four states no encoding tells apart. Showdown's own
/// positions carry `time` and never branch.
fn roll_confusion<'a>(
    turn: &mut Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Option<Vec<(f64, Turn<'a>)>> {
    let chance = {
        let mon = turn.mon_at(action.side, action.slot)?;
        let held = mon.volatile("confusion")?;
        let goes_on = held.extra.get(CONFUSION_GOES_ON).and_then(Value::as_bool).unwrap_or(false);
        if extra_int(held, CONFUSION_LEFT).is_some() || goes_on || !confusion_reached(turn, mon, mv) {
            return None;
        }
        confusion_cure_chance(held, budget)
    };
    if chance >= 1.0 && !budget.enumerate_status_checks && !budget.pinned_policy {
        turn.report("confusion length (the shortest of 2-to-5; not branched)");
    }
    if chance <= 0.0 {
        return None;
    }
    let options: Vec<(f64, bool)> = if chance >= 1.0 {
        vec![(chance, true)]
    } else {
        vec![(chance, true), (1.0 - chance, false)]
    };
    let mut out = Vec::new();
    for (weight, cures) in options {
        let mut state = turn.clone();
        if let Some(held) = state
            .mon_at_mut(action.side, action.slot)
            .and_then(|m| m.volatile_mut("confusion"))
        {
            if cures {
                set_extra(held, CONFUSION_LEFT, Some(json!(1)));
            } else {
                set_extra(held, CONFUSION_GOES_ON, Some(json!(true)));
            }
        }
        out.push((weight, state));
    }
    Some(out)
}

/// `time--`, cured at zero; whether the Pokemon is still confused for this try.
fn confusion_try(turn: &mut Turn, side: usize, slot: usize) -> bool {
    let Some(mon) = turn.mon_at_mut(side, slot) else { return false };
    let left = match mon.volatile("confusion") {
        Some(held) => extra_int(held, CONFUSION_LEFT),
        None => return false,
    };
    if matches!(left, Some(left) if left - 1 <= 0) {
        mon.volatiles.retain(|v| v.id.as_str() != "confusion");
        log_event!(turn, "{} snapped out of its confusion", Name(side, slot));
        return false;
    }
    let Some(held) = mon.volatile_mut("confusion") else { return false };
    match left {
        Some(left) => set_extra(held, CONFUSION_LEFT, Some(json!(left - 1))),
        None => {
            let tries = extra_int(held, CONFUSION_TRIES).unwrap_or(0) + 1;
            set_extra(held, CONFUSION_GOES_ON, None);
            set_extra(held, CONFUSION_TRIES, Some(json!(tries)));
        }
    }
    true
}

/// Confusion's `onBeforeMove` in `can_act`: the try, then `randomChance(33, 100)`.
fn confusion_stage(
    turn: &mut Turn,
    action: &QueuedAction,
    budget: &Budget,
) -> Vec<(f64, Option<String>)> {
    if confusion_try(turn, action.side, action.slot) && budget.enumerate_status_checks {
        return vec![
            (1.0 - CONFUSION_SELF_HIT_CHANCE, None),
            (CONFUSION_SELF_HIT_CHANCE, Some("confusion".into())),
        ];
    }
    vec![(1.0, None)]
}

/// Python's `_taunt_stops`: Taunt's `onBeforeMove` (priority 5) stops a status move but
/// Me First, before PP or a Choice lock (IKA-188).
///
/// ```text
/// if (!(move.isZ && move.isZOrMaxPowered) && move.category === 'Status' && move.id !== 'mefirst') {
///     this.add('cant', attacker, 'move: Taunt', move);
///     return false;
/// }
/// ```
///
/// `onDisableMove` only shapes the next request, so this is what stops a status move chosen
/// before the Taunt landed -- a Prankster Taunt, then the Tailwind. No PP is spent and no
/// Choice lock is set: both come after `BeforeMove`.
fn taunt_stops(mon: &crate::position::Pokemon, mv: &Move) -> bool {
    mon.has_volatile("taunt") && mv.category == "Status" && mv.id.as_str() != "mefirst"
}

/// Python's `_taunt_stage`: Taunt's check, then confusion's (5 is above confusion's 3).
/// 5 is below sleep and freeze (10) and flinch (8), and above confusion (3) and paralysis
/// (1), so a taunted status move spends no confused try.
fn taunt_stage(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Vec<(f64, Option<String>)> {
    if turn.mon_at(action.side, action.slot).is_some_and(|mon| taunt_stops(mon, mv)) {
        return vec![(1.0, Some("taunt".into()))];
    }
    if imprison_stops(turn, action, mv) {
        return vec![(1.0, Some("imprison".into()))];
    }
    confusion_stage(turn, action, budget)
}

/// Imprison's `onFoeBeforeMove` (priority 4, between Taunt's 5 and confusion's 3), IKA-256:
///
/// ```text
/// if (move.id !== 'struggle' && this.effectState.source.hasMove(move.id) && !move.isZOrMaxPowered) {
///     this.add('cant', attacker, 'move: Imprison', move);
///     return false;
/// }
/// ```
///
/// The volatile is on the imprisoner (`target: "self"`), which is its own `source`, and a
/// `Foe` handler is every opposing active Pokemon's. Its `onFoeDisableMove` takes the moves
/// off the next menu (`actions._usable_move_slots`); this stops the ones chosen before the
/// Imprison went up. No PP is spent: `BeforeMove` comes first.
fn imprison_stops(turn: &Turn, action: &QueuedAction, mv: &Move) -> bool {
    if mv.id == "struggle" || mv.id == "recharge" {
        return false;
    }
    let foe = 1 - action.side;
    (0..turn.pos.sides[foe].active.len()).any(|slot| {
        turn.mon_at(foe, slot).is_some_and(|mon| {
            !mon.fainted && mon.has_volatile("imprison") && mon.moves.iter().any(|m| m.id.as_str() == mv.id)
        })
    })
}

/// Python's `_taunt_lasts_longer`: `if (target.activeTurns && !this.queue.willMove(target))
/// duration++` -- 4 on a Pokemon that has already moved, 3 on one that came in this turn.
fn taunt_lasts_longer(turn: &mut Turn, side: usize, slot: usize) {
    if !turn.acted[side][slot] {
        return;
    }
    let Some(mon) = turn.mon_at_mut(side, slot) else { return };
    if mon.newly_switched {
        return;
    }
    if let Some(held) = mon.volatile_mut("taunt") {
        if let Some(duration) = held.duration {
            held.duration = Some(duration + 1);
        }
    }
}

/// (probability, reason it could not act) for the pre-move checks.
fn can_act(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Result<Vec<(f64, Option<String>)>, String> {
    let (fainted, has_flinch, status) = {
        let Some(mon) = turn.mon_at(action.side, action.slot) else {
            return Ok(vec![(1.0, Some("fainted".into()))]);
        };
        (mon.fainted, mon.has_volatile("flinch"), mon.status)
    };
    if fainted {
        return Ok(vec![(1.0, Some("fainted".into()))]);
    }
    if has_flinch {
        return Ok(vec![(1.0, Some("flinch".into()))]);
    }

    if is(status, "slp") {
        // `slp.onBeforeMove` decrements the counter when the Pokemon tries to move and cures
        // it at zero, so waking happens here rather than at end of turn. The duration was
        // set when sleep was applied and is part of the state, so this is deterministic.
        let woke = {
            let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
            let mut counter = mon.status_counter.unwrap_or(0) - 1;
            if mon.ability == "earlybird" {
                counter -= 1;
            }
            mon.status_counter = Some(counter);
            if counter <= 0 {
                mon.status = None;
                mon.status_counter = None;
                true
            } else {
                false
            }
        };
        if woke {
            log_event!(turn, "{} woke up", Name(action.side, action.slot));
        }
        return Ok(if woke {
            taunt_stage(turn, action, mv, budget)
        } else {
            vec![(1.0, Some("slp".into()))]
        });
    }

    if is(status, "frz") {
        // Python's `_can_act`: a `defrost` move skips the counter and the roll, and its
        // `onModifyMove` clears the status; Burn Up only for a Fire type (IKA-171).
        let defrosts = turn
            .mon_at(action.side, action.slot)
            .is_some_and(|mon| defrosts(turn, mon, mv));
        if defrosts {
            if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
                mon.status = None;
                mon.status_counter = None;
            }
            log_event!(turn, "{} thawed ({})", Name(action.side, action.slot), mv.id);
            return Ok(taunt_stage(turn, action, mv, budget));
        }
        // `time--; if (time <= 0 || randomChance(1, 4))` -- the counter is spent on the
        // attempt to move, and reaching zero thaws regardless of the roll.
        let thawed = {
            let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
            let counter = mon.status_counter.unwrap_or(FREEZE_COUNTER) - 1;
            mon.status_counter = Some(counter);
            if counter <= 0 {
                mon.status = None;
                mon.status_counter = None;
                true
            } else {
                false
            }
        };
        if thawed {
            log_event!(turn, "{} thawed (counter)", Name(action.side, action.slot));
            return Ok(taunt_stage(turn, action, mv, budget));
        }
        if !budget.enumerate_status_checks {
            return Ok(vec![(1.0, Some("frz".into()))]);
        }
        // The two outcomes differ in more than "did it act": one of them is no longer
        // frozen next turn, and this returns weights rather than states, so it cannot
        // express that. The weights are right and the cured state is reported as missing.
        turn.report("thaw roll (1 in 4; the cured state is not branched)");
        return Ok(vec![(THAW_CHANCE, None), (1.0 - THAW_CHANCE, Some("frz".into()))]);
    }

    // Neither the priority-blocking abilities nor Psychic Terrain are here: both act after
    // the move has started, so PP is spent (`priority_blocked_by`, IKA-158, and
    // `stopped_by_psychic_terrain`, IKA-156).

    // Confusion's priority 3 is above paralysis's 1: the self-hit first (IKA-177).
    let mut outcomes = taunt_stage(turn, action, mv, budget);
    if is(status, "par") && budget.enumerate_status_checks {
        let mut expanded = Vec::new();
        for (weight, reason) in outcomes {
            match reason {
                Some(reason) => expanded.push((weight, Some(reason))),
                None => {
                    expanded.push((weight * (1.0 - FULL_PARALYSIS_CHANCE), None));
                    expanded.push((weight * FULL_PARALYSIS_CHANCE, Some("par".into())));
                }
            }
        }
        outcomes = expanded;
    }
    Ok(outcomes)
}

/// `_stopped_by_psychic_terrain`: the terrain's `onTryHit` for one target
/// (`data/moves.ts:14116`). Only a positive-priority move, not a self-targeting one, into
/// a grounded foe; Mold Breaker reads Levitate as absent (IKA-156).
fn stopped_by_psychic_terrain(turn: &Turn, action: &QueuedAction, mv: &Move, target: Slot) -> bool {
    if !is(turn.pos.field.terrain, "psychicterrain") || action.priority <= 0 {
        return false;
    }
    if mv.target.as_str() == "self" || target.0 == action.side {
        return false;
    }
    let Some(defender) = turn.mon_at(target.0, target.1) else {
        return false;
    };
    if defender.fainted {
        return false;
    }
    let ignore_ability = turn
        .mon_at(action.side, action.slot)
        .is_some_and(|attacker| is_mold_breaker(attacker.ability.as_str()))
        && !matches!(defender.item, Some(i) if i.as_str() == "abilityshield");
    grounded_ignoring(turn, defender, ignore_ability)
}

/// `_priority_blocked_by`: Armor Tail / Queenly Majesty / Dazzling's `onFoeTryMove`
/// (`data/abilities.ts`, armortail), after the PP. Stops a positive-priority move whose
/// target is on the holder's side, never a `foeSide` one, and of the `all` moves only
/// Perish Song, Flower Shield and Rototiller. `breakable`: Mold Breaker passes (Mycelium
/// Might only with a status move) unless the holder has an Ability Shield (IKA-158).
fn priority_blocked_by(
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
) -> Option<Slot> {
    if action.priority <= 0 || mv.target.as_str() == "foeSide" {
        return None;
    }
    if mv.target.as_str() == "all" {
        if !matches!(mv.id.as_str(), "perishsong" | "flowershield" | "rototiller") {
            return None;
        }
    } else if !targets.iter().any(|t| t.0 != action.side) {
        return None;
    }
    let ignores_ability = turn.mon_at(action.side, action.slot).is_some_and(|attacker| {
        let ability = attacker.ability.as_str();
        is_mold_breaker(ability) && (ability != "myceliummight" || mv.category == "Status")
    });
    let foe_side = 1 - action.side;
    for slot in 0..turn.pos.sides[foe_side].active.len() {
        let Some(foe) = turn.mon_at(foe_side, slot) else {
            continue;
        };
        if foe.fainted
            || !matches!(
                foe.ability.as_str(),
                "armortail" | "queenlymajesty" | "dazzling"
            )
        {
            continue;
        }
        if ignores_ability && !matches!(foe.item, Some(i) if i.as_str() == "abilityshield") {
            continue;
        }
        return Some((foe_side, slot));
    }
    None
}

/// A Choice item's `onModifyMove`: `addVolatile('choicelock')`, whose `onStart` records the
/// move only when the lock is new -- a Struggle while locked leaves it on the locked move
/// (IKA-179). A lock naming no move slot (a record's lock on `struggle`) is one Showdown's
/// end of turn had dropped, so it locks afresh. Python's `_live_choice_lock`.
///
/// Any Choice item, not just the Scarf: the dump says which. Only the move that *started*
/// the lock is recorded, because `addVolatile` on a volatile already there does not run
/// `onStart` again. Overwriting it every move once let a Choice item stop being one: the
/// locked move ran out of PP, the holder Struggled, Struggle rewrote the lock to `struggle`
/// -- in nobody's move list -- and the legality rule dropped it as stale and offered the
/// whole moveset back (a Choice Scarf Garchomp locked into Earthquake for ten turns
/// Struggled on the eleventh and picked Stomping Tantrum on the twelfth). Showdown keeps
/// the lock on Earthquake and Struggles for the rest of the game.
fn lock_choice(turn: &mut Turn, side: usize, slot: usize, move_id: Id) {
    let Some(mon) = turn.mon_at_mut(side, slot) else { return };
    let stale = mon
        .volatile_mut("choicelock")
        .map(|held| held.move_id)
        .is_some_and(|held| !held.is_some_and(|id| mon.moves.get(id).is_some()));
    if stale {
        mon.volatiles.retain(|v| v.id.as_str() != "choicelock");
    }
    let fresh = !mon.has_volatile("choicelock");
    turn.add_volatile(side, slot, "choicelock", None);
    if fresh {
        if let Some(locked) = turn
            .mon_at_mut(side, slot)
            .and_then(|mon| mon.volatile_mut("choicelock"))
        {
            locked.move_id = Some(move_id);
        }
    }
}

/// `choicelock.onDisableMove`, run by `endTurn` for every active Pokemon: the lock goes when
/// the holder's item is no Choice item or the move is in none of its slots (IKA-179).
/// Python's `_choice_lock_ends`.
fn choice_lock_ends(reg: &Reg, turn: &mut Turn, order: &[Slot]) {
    for (side, slot) in order.iter().copied() {
        let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
        if mon.fainted {
            continue;
        }
        let Some(held) = mon.volatile_mut("choicelock").map(|held| held.move_id) else {
            continue;
        };
        let choice = mon.item.is_some_and(|i| reg.choice_items.contains(i.as_str()));
        let known = held.is_some_and(|id| mon.moves.get(id).is_some());
        if !choice || !known {
            mon.volatiles.retain(|v| v.id.as_str() != "choicelock");
        }
    }
}

fn use_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.unwrap();
    spend_pp(&mut turn, action);
    turn.current_actor = Some((action.side, action.slot));

    {
        let choice_items = &reg.choice_items;
        let locks = match turn.mon_at(action.side, action.slot) {
            None => false,
            Some(mon) => mon
                .item
                .map(|i| choice_items.contains(i.as_str()))
                .unwrap_or(false),
        };
        if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
            mon.last_move = Some(move_id);
        }
        if locks {
            lock_choice(&mut turn, action.side, action.slot, move_id);
        }
    }

    // Stance Change: Aegislash takes its Blade forme to attack and its Shield forme back
    // with King's Shield. The forme decides its stats, so leaving it alone puts every
    // later damage number on the wrong Pokemon.
    {
        let wanted = match turn.mon_at(action.side, action.slot) {
            Some(mon) if mon.ability.as_str() == "stancechange" => {
                if move_id.as_str() == "kingsshield" {
                    Some("aegislash")
                } else if mv.category != "Status" {
                    Some("aegislashblade")
                } else {
                    None
                }
            }
            _ => None,
        };
        let changes = match (wanted, turn.mon_at(action.side, action.slot)) {
            (Some(forme), Some(mon)) => mon.species.as_str() != forme,
            _ => false,
        };
        if changes {
            change_forme(reg, &mut turn, action.side, action.slot, wanted.unwrap())?;
        }
    }

    // `runMove` increments this before `onTry` runs, so the counter is 1 during the first
    // move a Pokemon makes after coming in. Last Resort fails until every *other* move the
    // Pokemon knows has been used, and needs at least two moves to begin with.
    let first_turn_failure = {
        let Some(mon) = turn.mon_at_mut(action.side, action.slot) else {
            return Ok(vec![(1.0, turn)]);
        };
        mon.active_move_actions += 1;
        FIRST_TURN_OUT_MOVES.contains(&move_id.as_str()) && mon.active_move_actions > 1
    };
    if first_turn_failure {
        log_event!(turn, "{} failed (only on the first turn out)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    if move_id.as_str() == "lastresort" && last_resort_fails(&turn, action) {
        log_event!(turn, "{} failed (other moves not all used)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    // The target must still be *waiting* to attack. One that already moved this turn --
    // commonly a faster Aqua Jet in the same priority bracket -- leaves nothing to counter,
    // and Showdown's `willMove` returns nothing.
    if move_id.as_str() == "suckerpunch" {
        let candidates = resolve_targets(reg, &mut turn, action, mv)?;
        let pending = candidates.iter().any(|slot| {
            slot.0 != action.side
                && turn.attacks[slot.0][slot.1]
                && !turn.acted[slot.0][slot.1]
        });
        if !pending {
            log_event!(turn, "{} failed (nothing left to counter)", Label(reg, action));
            turn.move_failed[action.side][action.slot] = true;
            return Ok(vec![(1.0, turn)]);
        }
    }

    // A move that spends a turn winding up. Exactly this much of them is modelled and no
    // more -- there is no semi-invulnerability anywhere in the engine -- so Fly and Dig have
    // the same shape here as Solar Beam. (This was written when the port's oracle was
    // Python; since IKA-212 it is Showdown, against which Fly and Dig are not modelled.)
    if let Some((_, skip_weather)) =
        TWO_TURN_MOVES.iter().find(|(id, _)| *id == move_id.as_str())
    {
        let charged = turn
            .mon_at(action.side, action.slot)
            .map(|mon| mon.has_volatile("twoturnmove"))
            .unwrap_or(false);
        if charged {
            // The charge already happened last turn: drop the marker and attack. Showdown's
            // `onTryMove` starts with `if (attacker.removeVolatile(move.id)) return;`, so
            // there is no second boost.
            if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
                mon.volatiles.retain(|v| v.id.as_str() != "twoturnmove");
            }
        } else {
            // The charge-turn boost applies whether or not the charge is skipped: Showdown
            // boosts first and only then checks the weather. Electro Shot in rain both
            // raises Special Attack and attacks in the same turn, so leaving the boost out
            // understates its damage by a whole stage.
            if let Some(boosts) = charge_turn_boosts(move_id.as_str()) {
                turn.apply_boosts(action.side, action.slot, boosts, false, move_id.as_str());
            }
            let skipped = turn
                .pos
                .field
                .weather
                .as_ref()
                .map(|w| skip_weather.contains(&w.as_str()))
                .unwrap_or(false);
            if !skipped {
                // `duration: 2`, and the move stored as Showdown's `effectState.move`: the
                // next turn's menu is that move alone (IKA-169), and a charge that never
                // fires does not keep the marker for good.
                turn.add_volatile(action.side, action.slot, "twoturnmove", Some(2));
                if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
                    if let Some(charging) = mon.volatile_mut("twoturnmove") {
                        charging.move_id = Some(move_id);
                        // ...and the target, which the next turn fires at (IKA-176).
                        if let Some(loc) = action.target.filter(|t| *t != 0) {
                            std::rc::Rc::make_mut(&mut charging.extra)
                                .insert("targetLoc".into(), json!(loc));
                        }
                    }
                }
                log_event!(turn, "{} is charging", Label(reg, action));
                return Ok(vec![(1.0, turn)]);
            }
        }
    }

    let targets = resolve_targets(reg, &mut turn, action, mv)?;
    // Explosion, Self-Destruct, Misty Explosion (IKA-208): `useMoveInner` faints the user
    // right after `TryMove` -- Damp's `onAnyTryMove` -- and before it looks at the targets,
    // so it goes whatever the blast meets. The hits are then computed from the user as it
    // was: Showdown's `faint()` only zeroes the HP, and the boosts and volatiles stay until
    // `faintMessages`.
    let exploded = if mv.raw.get("selfdestruct").and_then(Value::as_str) == Some("always") {
        if damp_stops(&turn, action) {
            turn.move_failed[action.side][action.slot] = true;
            return Ok(vec![(1.0, turn)]);
        }
        let as_used = turn.mon_at(action.side, action.slot).cloned();
        turn.faint(action.side, action.slot);
        as_used
    } else {
        None
    };
    let no_target_needed = matches!(
        mv.target.as_str(),
        "self" | "allySide" | "allyTeam" | "all" | "foeSide"
    );
    if targets.is_empty() && !no_target_needed {
        log_event!(turn, "{} had no target", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    // Double Shock and Burn Up: the move's own `onTryMove`, after the target and the PP,
    // and before the `runEvent('TryMove')` the priority block answers. It returns `null`,
    // so the move is not one `move_failed` records.
    if let Some(spent) = spent_type(move_id.as_str()) {
        let lacks = match turn.mon_at(action.side, action.slot) {
            Some(mon) => !turn.types_of(mon).contains(spent),
            None => false,
        };
        if lacks {
            log_event!(turn, "{} failed (no {} type)", Label(reg, action), spent);
            return Ok(vec![(1.0, turn)]);
        }
    }

    // `TryMove`: after the PP, the target and the charge turn, before any hit step.
    if let Some(holder) = priority_blocked_by(&turn, action, mv, &targets) {
        if turn.log.is_some() {
            let ability = turn.mon_at(holder.0, holder.1).map(|m| m.ability.as_str().to_string());
            let ability = ability.unwrap_or_else(|| "?".into());
            log_event!(turn, "{} did not happen (ability: {})", Label(reg, action), ability);
        }
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    // Counter, Mirror Coat, Metal Burst, Comeuppance: `onTry`, after `TryMove` (IKA-213).
    if crate::damage_callback::fails_on_try(&turn, action, move_id.as_str()) {
        log_event!(turn, "{} failed (nothing to return)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    // Poltergeist: `onTry`, the target holds nothing (IKA-240).
    if crate::move_hooks::fails_on_try(&turn, move_id.as_str(), &targets) {
        log_event!(turn, "{} failed (the target holds nothing)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    // The protection a `breaksProtect` move tears down is torn down per target, inside
    // `hit_target`, once the hit is known to land (IKA-153). Every such move is damaging
    // (Feint, Phantom Force), so the status path never needs it.

    if mv.category == "Status" {
        return do_status_move(reg, turn, action, mv, &targets, budget);
    }
    // A damaging move with custom code the port never names (`modelled.rs`, IKA-213).
    if crate::modelled::damaging_move_is_unmodelled(move_id.as_str()) {
        turn.report(format!("damaging move: {move_id}"));
    }

    // `move.spreadHit` is decided from every target before the hit steps run, and Psychic
    // Terrain (step 1, ahead of Protect) then drops the grounded ones.
    let spread = move_hits_multiple(reg, move_id.as_str(), targets.len())
        || (targets.len() > 1 && crate::airborne::expanding_force_spreads(&turn, action, mv));
    if turn.log.is_some() {
        let stopped: Vec<Slot> =
            targets.iter().copied().filter(|t| stopped_by_psychic_terrain(&turn, action, mv, *t)).collect();
        for t in stopped {
            log_event!(turn, "{} protected by psychicterrain", Name(t.0, t.1));
        }
    }
    let targets: Vec<Slot> = targets
        .iter()
        .copied()
        .filter(|t| !stopped_by_psychic_terrain(&turn, action, mv, *t))
        .collect();
    if targets.is_empty() {
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    turn.move_damage_total = 0;
    turn.move_connected = false;
    begin_move_watch(&mut turn);
    let targets = if mv.raw.get("smartTarget").and_then(Value::as_bool) == Some(true) {
        match smart_hits(reg, &turn, action, mv, &targets, &budget)? {
            SmartHits::Split(splits) => {
                let mut branches: Vec<Outcome<'a>> = Vec::new();
                for (weight, plan) in splits {
                    let mut here: Vec<Outcome<'a>> = vec![(weight, turn.clone())];
                    if plan.is_empty() {
                        here[0].1.move_failed[action.side][action.slot] = true;
                    }
                    for (target, hits) in plan {
                        let mut expanded: Vec<Outcome<'a>> = Vec::new();
                        for (w, state) in here {
                            for (inner, next) in
                                hit_target(reg, state, action, mv, target, false, budget, None, Some(hits))?
                            {
                                expanded.push((w * inner, next));
                            }
                        }
                        here = expanded;
                    }
                    branches.extend(here);
                }
                for (_weight, state) in branches.iter_mut() {
                    after_move(state, action, mv)?;
                }
                return Ok(branches);
            }
            SmartHits::Ordinary(targets) => targets,
        }
    } else {
        targets
    };
    let mut branches: Vec<Outcome<'a>> = vec![(1.0, turn)];
    for target in &targets {
        let mut expanded: Vec<Outcome<'a>> = Vec::new();
        for (weight, state) in branches.into_iter() {
            let started = crate::resolve::phase_start();
            let hit = hit_target(reg, state, action, mv, *target, spread, budget, exploded.as_ref(), None)?;
            crate::resolve::phase_end(9, started);
            for (inner_weight, inner_state) in hit {
                expanded.push((weight * inner_weight, inner_state));
            }
        }
        branches = expanded;
    }
    for (_weight, state) in branches.iter_mut() {
        after_move(state, action, mv)?;
    }
    Ok(branches)
}

/// Last Resort's `onTry` (data/moves.ts, IKA-208): `false` with fewer than two moves, or
/// while any other move slot is not `used` -- which `moveUsed` sets and a switch clears.
fn last_resort_fails(turn: &Turn, action: &QueuedAction) -> bool {
    let Some(mon) = turn.mon_at(action.side, action.slot) else { return true };
    let slots: Vec<_> = mon.moves.iter().collect();
    slots.len() < 2
        || !slots.iter().any(|slot| slot.id.as_str() == "lastresort")
        || slots.iter().any(|slot| slot.id.as_str() != "lastresort" && !slot.used)
}

fn spend_pp(turn: &mut Turn, action: &QueuedAction) {
    let Some(move_id) = action.move_id else { return };
    let Some(mon) = turn.mon_at_mut(action.side, action.slot) else { return };
    // `if (!lockedMove) deductPP(...)`: a rampage's later turns are free (IKA-174).
    let rampaging = rampage_move(mon) == Some(move_id);
    if let Some(slot) = mon.moves.get_mut(move_id) {
        if slot.pp > 0 && !rampaging {
            slot.pp -= 1;
        }
        slot.used = true;
    }
}

fn move_hits_multiple(reg: &Reg, move_id: &str, live_targets: usize) -> bool {
    match reg.moves.get(move_id).map(|m| m.target.as_str()) {
        Some("allAdjacentFoes") => live_targets > 1,
        Some("allAdjacent") => live_targets >= 1,
        _ => false,
    }
}

/// Which (side, slot) the move actually hits, after redirection. "If a targeted foe faints,
/// the move is retargeted" (`Battle#getTarget`): a move whose target has already fallen this
/// turn hits the other one rather than failing.
fn resolve_targets(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
) -> Result<Vec<Slot>, String> {
    let me = (action.side, action.slot);
    let foe_side = 1 - action.side;
    let slots = 0..turn.pos.sides[foe_side].active.len();

    let live = |turn: &Turn, side: usize, slot: usize| -> bool {
        matches!(turn.mon_at(side, slot), Some(mon) if !mon.fainted)
    };

    match crate::airborne::move_target(turn, action, mv) {
        "self" | "allySide" | "allyTeam" | "all" | "foeSide" => return Ok(vec![me]),
        "allAdjacentFoes" => {
            return Ok(slots.filter(|s| live(turn, foe_side, *s)).map(|s| (foe_side, s)).collect())
        }
        "allAdjacent" => {
            let mut out: Vec<Slot> =
                slots.filter(|s| live(turn, foe_side, *s)).map(|s| (foe_side, s)).collect();
            let ally = 1 - action.slot;
            if live(turn, action.side, ally) {
                out.push((action.side, ally));
            }
            return Ok(out);
        }
        "allies" => {
            return Ok((0..turn.pos.sides[action.side].active.len())
                .filter(|s| live(turn, action.side, *s))
                .map(|s| (action.side, s))
                .collect())
        }
        "adjacentAlly" | "adjacentAllyOrSelf" => {
            let Some(target) = action.target else { return Ok(vec![me]) };
            let slot = (-target - 1) as usize;
            return Ok(if live(turn, action.side, slot) {
                vec![(action.side, slot)]
            } else {
                Vec::new()
            });
        }
        // `randomNormal` falls through: `draw_random_target` put the drawn foe in the
        // action's target, and it is redirected as any other (IKA-178).
        _ => {}
    }
    if let Some(recorded) = crate::damage_callback::target(turn, action, mv.id.as_str()) {
        return Ok(reply_targets(turn, action, mv, recorded));
    }

    let mut chosen = match action.target {
        None => {
            match (0..turn.pos.sides[foe_side].active.len())
                .find(|s| live(turn, foe_side, *s))
            {
                None => return Ok(Vec::new()),
                Some(slot) => (foe_side, slot),
            }
        }
        Some(target) if target > 0 => (foe_side, (target - 1) as usize),
        Some(target) => (action.side, (-target - 1) as usize),
    };

    if !live(turn, chosen.0, chosen.1) && chosen.0 != action.side {
        if let Some(slot) = (0..turn.pos.sides[foe_side].active.len())
            .find(|s| live(turn, foe_side, *s))
        {
            chosen = (foe_side, slot);
            log_event!(turn, "{} retargeted to {}", Label(reg, action), Name(chosen.0, chosen.1));
        }
    }

    if let Some(redirected) = redirection_target(turn, action, mv, chosen) {
        if redirected != chosen {
            log_event!(turn, "{} redirected to {}", Label(reg, action), Name(redirected.0, redirected.1));
        }
        chosen = redirected;
    }
    Ok(if live(turn, chosen.0, chosen.1) { vec![chosen] } else { Vec::new() })
}

/// Counter, Mirror Coat, Metal Burst and Comeuppance aim at the slot that hit the user
/// (IKA-213). `getMoveTargets` retargets a fainted foe first and then runs `RedirectTarget`,
/// where Follow Me (priority 1) outranks Counter's own redirect (-1), which sends Counter and
/// Mirror Coat back to the recorded slot -- so they fail on a fainted attacker, while Metal
/// Burst and Comeuppance (`onModifyTarget`, before all of it) hit the other foe.
fn reply_targets(turn: &Turn, action: &QueuedAction, mv: &Move, recorded: Slot) -> Vec<Slot> {
    let live = |slot: Slot| matches!(turn.mon_at(slot.0, slot.1), Some(mon) if !mon.fainted);
    let mut chosen = recorded;
    if !live(chosen) {
        if let Some(slot) = (0..turn.pos.sides[recorded.0].active.len()).find(|s| live((recorded.0, *s))) {
            chosen = (recorded.0, slot);
        }
    }
    match redirection_target(turn, action, mv, chosen) {
        Some(redirected) => chosen = redirected,
        None if crate::damage_callback::redirects_back(mv.id.as_str()) => chosen = recorded,
        None => {}
    }
    if live(chosen) {
        vec![chosen]
    } else {
        Vec::new()
    }
}

/// Follow Me / Rage Powder / Spotlight, then the type-drawing abilities. Rage Powder is a
/// powder effect, so it does not pull in a move used by a Grass type, a Safety Goggles
/// holder or an Overcoat Pokemon; the exemption is on the move's user, not on the
/// redirector, and Follow Me has no such exemption.
fn redirection_target(
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    chosen: Slot,
) -> Option<Slot> {
    if chosen.0 == action.side {
        return None;
    }
    let foe_side = chosen.0;
    let powder_immune = match turn.mon_at(action.side, action.slot) {
        None => false,
        Some(user) => {
            turn.types_of(user).contains("Grass")
                || is(user.item, "safetygoggles")
                || user.ability == "overcoat"
        }
    };
    for slot in 0..turn.pos.sides[foe_side].active.len() {
        let Some(mon) = turn.mon_at(foe_side, slot) else { continue };
        if mon.fainted {
            continue;
        }
        let drawing: Vec<&str> = ["followme", "ragepowder", "spotlight"]
            .into_iter()
            .filter(|v| mon.has_volatile(v))
            .collect();
        if drawing.is_empty() {
            continue;
        }
        if drawing == ["ragepowder"] && powder_immune {
            continue;
        }
        return Some((foe_side, slot));
    }
    for slot in 0..turn.pos.sides[foe_side].active.len() {
        let Some(mon) = turn.mon_at(foe_side, slot) else { continue };
        if mon.fainted || (foe_side, slot) == chosen {
            continue;
        }
        let draws = match mon.ability.as_str() {
            "lightningrod" => Some("Electric"),
            "stormdrain" => Some("Water"),
            _ => None,
        };
        if draws == Some(mv.mtype.as_str()) {
            return Some((foe_side, slot));
        }
    }
    None
}

/// Side conditions a `breaksProtect` move removes. From gen 6 it strips them regardless of
/// whose side they are on, which is why there is no ally test here. Python's
/// `BREAKABLE_SIDE_CONDITIONS`.
const BREAKABLE_SIDE_CONDITIONS: [&str; 4] =
    ["craftyshield", "matblock", "quickguard", "wideguard"];

fn is_protect_volatile(id: &str) -> bool {
    PROTECT_VOLATILES.iter().any(|(vid, _)| *vid == id)
}

/// Strips the guards a `breaksProtect` move tears down, for the rest of the turn.
///
/// Letting the move through was only half of it: if the volatile survived, the target's
/// *partner* was still protected and the point of Feint in doubles -- break the Protect,
/// then hit with the partner -- never happened. Breaking anything also clears `stall`, so
/// the target's next Protect is certain again instead of one in three.
///
/// Python's `_break_protection`, line for line: every protect-family volatile in
/// `PROTECT_VOLATILES` (not Showdown's literal seven, because Detect stays `detect` here),
/// then the side's breakable guards, and when anything broke, the target's `stall`.
/// A target slot with no Pokemon in it is skipped whole, side conditions included.
///
/// Reads before it writes: `mon_at_mut` unshares the Pokemon, and most Feints break
/// nothing.
fn break_protection(turn: &mut Turn, action: &QueuedAction, targets: &[Slot]) {
    for &(side, slot) in targets {
        let Some(mon) = turn.mon_at(side, slot) else {
            continue;
        };
        let broke = mon.volatiles.iter().any(|v| is_protect_volatile(v.id.as_str()));
        // Python's `broke + stripped`, for the trace (IKA-215).
        let mut names: Vec<Id> = Vec::new();
        if turn.log.is_some() {
            names.extend(mon.volatiles.iter().map(|v| v.id).filter(|id| is_protect_volatile(id.as_str())));
            names.extend(
                turn.pos.sides[side]
                    .side_conditions
                    .iter()
                    .map(|c| c.id)
                    .filter(|id| BREAKABLE_SIDE_CONDITIONS.contains(&id.as_str())),
            );
        }
        if broke {
            if let Some(mon) = turn.mon_at_mut(side, slot) {
                mon.volatiles.retain(|v| !is_protect_volatile(v.id.as_str()));
            }
        }

        let conditions = &mut turn.pos.sides[side].side_conditions;
        let before = conditions.len();
        conditions.retain(|c| !BREAKABLE_SIDE_CONDITIONS.contains(&c.id.as_str()));
        let stripped = conditions.len() != before;
        if broke || stripped {
            let joined = names.iter().map(|id| id.as_str()).collect::<Vec<_>>().join(", ");
            log_event!(turn, "{} broke {} on {}", Label(turn.reg, action), joined, Name(side, slot));
        }

        let stalling = turn.mon_at(side, slot).is_some_and(|m| m.has_volatile("stall"));
        if (broke || stripped) && stalling {
            if let Some(mon) = turn.mon_at_mut(side, slot) {
                mon.volatiles.retain(|v| v.id.as_str() != "stall");
            }
        }
    }
}

/// Whether a Protect-family effect or a guard blocks this hit. A same-side target is still
/// protected: Earthquake hits its own partner, and a partner behind Protect is not hit.
fn blocked_by_protect(
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
) -> Option<String> {
    if !mv.has_flag(F_PROTECT) || mv.breaks_protect {
        return None;
    }
    let side = &turn.pos.sides[target.0];
    let from_foe = target.0 != action.side;
    if from_foe
        && matches!(crate::airborne::move_target(turn, action, mv), "allAdjacentFoes" | "allAdjacent")
        && side.has_side_condition("wideguard")
    {
        return Some("wideguard".into());
    }
    if from_foe && action.priority > 0 && side.has_side_condition("quickguard") {
        return Some("quickguard".into());
    }
    let mon = turn.mon_at(target.0, target.1)?;
    for (vid, blocks) in PROTECT_VOLATILES {
        if !mon.has_volatile(vid) {
            continue;
        }
        if blocks == "damaging" && mv.category == "Status" {
            continue;
        }
        return Some(vid.to_string());
    }
    None
}

/// Python's `multihit_counts`. A [2, 5] move is 35-35-15-15, Showdown's
/// `sample([2 x7, 3 x7, 4 x3, 5 x3])`, and Skill Link takes the upper end before anything
/// is drawn, under every budget (IKA-160).
///
/// The source is `hitStepMoveHitLoop`'s
/// `sample([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5])`
/// (vendor/pokemon-showdown/sim/battle-actions.ts:869-870, and the champions mod's copy in
/// data/mods/champions/scripts.ts:440-441). It was 1/3, 1/3, 1/6, 1/6 -- the older
/// `[2, 2, 3, 3, 4, 5]` -- until IKA-160; 200,000 real turns of Bullet Seed in the Champions
/// format came out 35.0 / 35.1 / 14.9 / 14.9 (and a Skill Link user's 200,000 all 5). Under a
/// budget that does not enumerate it the count is the minimum, which is what Showdown's
/// `multihit='min'` policy produces (`sample` returns the first element, `random(a, b)`
/// returns `a`). Loaded Dice is `isNonstandard: "Past"` in both Champions mods and is not
/// modelled.
fn multihit_counts(mv: &Move, budget: &Budget, ability: &str) -> Vec<(usize, f64)> {
    let Some(multihit) = mv.multihit.as_ref() else { return vec![(1, 1.0)] };
    if let Some(fixed) = multihit.as_u64() {
        return vec![(fixed as usize, 1.0)];
    }
    let Some(list) = multihit.as_array() else { return vec![(1, 1.0)] };
    let low = list.first().and_then(Value::as_u64).unwrap_or(1) as usize;
    let high = list.last().and_then(Value::as_u64).unwrap_or(low as u64) as usize;
    if ability == "skilllink" {
        return vec![(high, 1.0)];
    }
    if !budget.enumerate_secondary {
        return vec![(low, 1.0)];
    }
    // IKA-210's positive control: the older `[2, 2, 3, 3, 4, 5]` (IKA-160's bug).
    if cfg!(feature = "ika210-control") && (low, high) == (2, 5) {
        return vec![(2, 1.0 / 3.0), (3, 1.0 / 3.0), (4, 1.0 / 6.0), (5, 1.0 / 6.0)];
    }
    if (low, high) == (2, 5) {
        return vec![
            (2, 7.0 / 20.0),
            (3, 7.0 / 20.0),
            (4, 3.0 / 20.0),
            (5, 3.0 / 20.0),
        ];
    }
    let span = high - low + 1;
    (0..span).map(|i| (low + i, 1.0 / span as f64)).collect()
}

/// One damaging hit on one target, enumerating accuracy, crit and damage roll.
///
/// With accuracy not enumerated the move hits: that is what the budget means, and it is
/// what Showdown's pinned `accuracy: 'hit'` policy does (deciding by whether the accuracy
/// exceeds 50% would make Hurricane miss in sun). The damage passed on is the raw roll, not
/// one capped at the target's HP: `deal_damage` owns the cap *and* the Focus Sash / Sturdy
/// consumption that goes with it, and capping here as well would leave the item on the
/// field. Between the hits of a multi-hit move the defender's state has moved on -- Stamina
/// has raised its Defence, a berry has fired, its HP is lower for a fraction-of-HP move --
/// so each later hit's damage is computed again rather than reused.
#[allow(clippy::too_many_arguments)]
fn hit_target<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
    spread: bool,
    budget: Budget,
    exploded: Option<&crate::position::Pokemon>,
    forced_hits: Option<usize>,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.unwrap();
    if let Some(blocked) = blocked_by_protect(&turn, action, mv, target) {
        log_event!(turn, "{} blocked by {}", Label(reg, action), blocked);
        protect_punish(&mut turn, action, mv, &blocked)?;
        return Ok(vec![(1.0, turn)]);
    }

    let attacker = match exploded {
        Some(mon) => Battler::from_pokemon(reg, mon)?,
        None => match turn.battler_at(action.side, action.slot)? {
            Some(attacker) => attacker,
            None => return Ok(vec![(1.0, turn)]),
        },
    };
    let Some(defender) = turn.battler_at(target.0, target.1)? else {
        return Ok(vec![(1.0, turn)]);
    };
    if defender.ability.as_str() == "iceface" {
        return Err(format!("forme guard: {}", defender.ability));
    }
    // Python's `_hit_target`: a doll in front takes each hit while it stands (IKA-180).
    let subbed = hits_substitute(&turn, action, mv, target);
    // Disguise (IKA-208), Python's `_forme_guard` / `_bust_disguise` (IKA-155, IKA-157):
    // `onDamage` returns 0 for the first hit's damage, at the damage step -- after the
    // immunity, the accuracy and the break -- and 0 is still a hit, so everything the hit
    // carries happens; `onCriticalHit` is false and `onEffectiveness` 0 for it; the forme
    // changes at the `Update` after that hit and costs `baseMaxhp / 8`. A doll in front
    // takes the hit instead (`_substitute_in_front`).
    let multihit = mv.raw.get("multihit").is_some_and(|v| !v.is_null());
    // `flags: { breakable: 1 }`: not against a Mold Breaker's move.
    let mut guarded = mv.category != "Status"
        && defender.ability.as_str() == "disguise"
        && defender.species.as_str() == "mimikyu"
        && !is_mold_breaker(attacker.ability.as_str());
    if subbed && guarded {
        if multihit {
            turn.report(format!("substitute: {} past a broken Substitute into a forme guard", mv.id));
        }
        guarded = false;
    }

    let accuracy = accuracy_of(&turn, mv, &attacker, &defender);
    let crit_p = crit_probability(reg, &attacker, &defender, move_id.as_str());

    let accuracy_branches: Vec<(f64, bool)> = if forced_hits.is_some() {
        // `smart_hits` has rolled it already.
        vec![(1.0, true)]
    } else if budget.enumerate_accuracy && accuracy > 0.0 && accuracy < 1.0 {
            vec![(accuracy, true), (1.0 - accuracy, false)]
        } else {
            vec![(1.0, accuracy > 0.0)]
        };
    let mut crit_branches: Vec<(f64, bool)> =
        if budget.enumerate_crit && crit_p > 0.0 && crit_p < 1.0 {
            vec![(crit_p, true), (1.0 - crit_p, false)]
        } else {
            vec![(1.0, crit_p >= 1.0)]
        };
    let mut rolls = stratified_rolls(&budget);
    if guarded && !multihit {
        // The one hit is absorbed: neither a crit nor a roll changes anything.
        crit_branches = vec![(1.0, false)];
        rolls = vec![(rolls[0].0, 1.0)];
    }
    // Python's `_hit_target` returns every outcome from here on with the note "damage
    // rolls stratified" unless the roll is pinned or all sixteen are kept, and `_run_queue`
    // makes the turn inexact for it. Setting it on `turn` puts it on every clone below;
    // the one outcome that is `turn` itself, the empty fallback, is cleared again (IKA-151).
    if budget.fixed_roll().is_none() && budget.damage_rolls < 16 {
        turn.rolls_stratified = true;
    }
    // Does not depend on the roll, and the exact budget enumerates sixteen of them -- so
    // building this inside the loop was fifteen wasted allocations per hit on the path that
    // advances a game. Under the matrix budget the roll is fixed and it costs nothing,
    // which is why the allocation count barely moved and the clock did.
    let hit_counts = match forced_hits {
        Some(hits) => vec![(hits, 1.0)],
        None => multihit_counts(mv, &budget, attacker.ability.as_str()),
    };
    let rolls_later_hits = rolls_each_hit(mv, &attacker);

    let ctx_started = crate::resolve::phase_start();
    let move_ctx = MoveContext {
        weather: turn.pos.field.weather,
        terrain: turn.pos.field.terrain,
        side_total_fainted: turn.pos.sides[action.side]
            .pokemon
            .iter()
            .filter(|m| m.fainted)
            .count() as i64,
        times_attacked: turn
            .mon_at(action.side, action.slot)
            .map(|m| m.times_attacked)
            .unwrap_or(0),
        target_hurt_this_turn: turn.hurt_this_turn[target.0][target.1],
        damaged_by_target: false,
        previous_move_failed: turn
            .mon_at(action.side, action.slot)
            .map(|m| m.move_last_turn_failed)
            .unwrap_or(false),
        hit_index: 1,
        moving_last: false,
        ally_used_same_move: false,
        reply_damage: crate::damage_callback::damage(&turn, action, move_id.as_str()),
    };

    crate::resolve::phase_end(12, ctx_started);

    let mut outcomes: Vec<Outcome<'a>> = Vec::new();
    for (acc_weight, hit) in accuracy_branches {
        if acc_weight <= 0.0 {
            continue;
        }
        if !hit {
            let mut state = turn.clone();
            log_event!(state, "{} missed", Label(reg, action));
            state.move_failed[action.side][action.slot] = true;
            outcomes.push((acc_weight, state));
            continue;
        }
        let field = field_for_hit(&turn);
        // Brick Break, Psychic Fangs: the screens are gone before `getDamage` (IKA-240).
        let field = crate::move_hooks::field_past_screens(field, move_id.as_str(), target);
        for (crit_weight, crit) in crit_branches.iter().copied() {
        if crit_weight <= 0.0 {
            continue;
        }
        let started = crate::resolve::phase_start();
        // A guarded first hit never crits; `crit` then speaks for the later hits only.
        let result = calculate(
            reg,
            &attacker,
            &defender,
            move_id.as_str(),
            &field,
            target.0,
            spread,
            crit && !guarded,
            Some(&move_ctx),
            None,
            false,
        );
        crate::resolve::phase_end(10, started);
        for note in &result.unmodelled {
            turn.report(note.clone());
        }
        if result.immune {
            let mut state = turn.clone();
            log_event!(state, "{} had no effect", Label(reg, action));
            state.move_failed[action.side][action.slot] = true;
            absorb(&mut state, mv, target);
            outcomes.push((acc_weight * crit_weight, state));
            continue;
        }
        let doll_rolls = if subbed {
            doll_damage(reg, &attacker, &defender, move_id.as_str(), &field, target, spread, crit, &move_ctx)
        } else {
            None
        };
        for (roll, roll_weight) in &rolls {
            for (hits, hit_weight) in hit_counts.iter().copied() {
                let mut state = turn.clone();
                // `hitStepBreakProtect` is step 5 of `trySpreadMoveHit`, after the type
                // immunity (2) and the accuracy (4), and only for the targets they left: a
                // Feint into a Protecting Ghost, or one that misses, breaks nothing
                // (IKA-153). Python's `_hit_target`, at the same place.
                if mv.breaks_protect {
                    break_protection(&mut state, action, &[target]);
                }
                // The move's own `onTryHit`, after the immunity and the accuracy (IKA-240).
                crate::move_hooks::break_screens(&mut state, move_id.as_str(), target);
                let mut reached = false;
                // Python's `total` and its loop variable, for "hit Nx for T" (IKA-215).
                let mut total = 0i64;
                let mut last_index = 0usize;
                // A later hit of a multiaccuracy move that misses ends the move there
                // (IKA-235): each such stop, its weight and its hits so far.
                let mut stops: Vec<(f64, Turn<'a>, usize, i64, bool)> = Vec::new();
                let mut stop_weight = 1.0f64;
                for hit_index in 0..hits {
                    last_index = hit_index;
                    let gone = match state.mon_at(target.0, target.1) {
                        None => true,
                        Some(mon) => mon.fainted,
                    };
                    if gone {
                        break;
                    }
                    if rolls_later_hits && hit_index > 0 {
                        // A fainted user stops the loop before the roll, as below.
                        let Some(chance) = later_hit_chance(&state, mv, action, target)? else {
                            break;
                        };
                        if chance <= 0.0 {
                            break;
                        }
                        if chance < 1.0 && budget.enumerate_accuracy {
                            let stopped = state.clone();
                            stops.push((stop_weight * (1.0 - chance), stopped, hit_index - 1, total, reached));
                            stop_weight *= chance;
                        }
                    }
                    let on_doll = subbed && hits_substitute(&state, action, mv, target);
                    let amount = if hit_index == 0 {
                        match (&doll_rolls, on_doll) {
                            (Some(rolls), true) => rolls[*roll],
                            _ => result.rolls[*roll],
                        }
                    } else {
                        let Some(live_attacker) =
                            state.battler_at(action.side, action.slot)?
                        else {
                            break;
                        };
                        let Some(mut live_defender) = state.battler_at(target.0, target.1)? else {
                            break;
                        };
                        if on_doll {
                            live_defender = behind_substitute(live_defender);
                        }
                        let mut again_ctx = move_ctx.clone();
                        again_ctx.hit_index = hit_index as i64 + 1;
                        let field = field_for_hit(&state);
                        let again = calculate(
                            reg,
                            &live_attacker,
                            &live_defender,
                            move_id.as_str(),
                            &field,
                            target.0,
                            spread,
                            crit,
                            Some(&again_ctx),
                            None,
                            false,
                        );
                        if again.immune {
                            break;
                        }
                        again.rolls[*roll]
                    };
                    // Final Gambit (IKA-208): `damageCallback(pokemon) { const damage =
                    // pokemon.hp; pokemon.faint(); return damage; }` runs in `getDamage`,
                    // after the immunity, on a doll as on the Pokemon, so the user is in
                    // the faint queue before the target (`selfdestruct: "ifHit"` then
                    // finds it fainted already).
                    if hit_index == 0 && mv.id == "finalgambit" {
                        state.faint(action.side, action.slot);
                    }
                    if on_doll {
                        hit_substitute(&mut state, action, mv, target, amount, &budget)?;
                        continue;
                    }
                    let absorbed = guarded && hit_index == 0;
                    let berry = crate::move_hooks::set_berry_aside(&mut state, move_id.as_str(), target);
                    let dealt = if absorbed {
                        0
                    } else {
                        state.deal_damage(target.0, target.1, amount, true, move_id.as_str())?
                    };
                    total += dealt;
                    // Bug Bite, Pluck: `onHit`, before `DamagingHit` and the `Update` (IKA-240).
                    crate::move_hooks::steal_berry(&mut state, action, target, berry, mv.mtype.as_str(), result.type_mod);
                    crate::damage_callback::record(&mut state, (action.side, action.slot), target, dealt, mv.category.as_str());
                    reached = true;
                    state.move_hit[target.0][target.1] = true;
                    let after_started = crate::resolve::phase_start();
                    // A hit into Endure at 1 HP deals 0 and is still a hit (IKA-171), and
                    // so is one Disguise took, which is neutral (no resist berry).
                    let landed = absorbed || amount > 0;
                    after_hit(
                        &mut state,
                        action,
                        mv,
                        target,
                        dealt,
                        landed,
                        &budget,
                        if absorbed { 0 } else { result.type_mod },
                    )?;
                    crate::resolve::phase_end(11, after_started);
                    if absorbed {
                        bust_disguise(reg, &mut state, target)?;
                        if state.log.is_some() {
                            let busted = state.mon_at(target.0, target.1).map(|m| m.ability);
                            log_event!(state, "{} absorbed by {}", Label(reg, action), OrNone(busted));
                        }
                    }
                }
                stops.push((stop_weight, state, last_index, total, reached));
                for (stop_weight, mut state, last_index, total, reached) in stops {
                if hits > 1 {
                    log_event!(state, "{} hit {}x for {}", Label(reg, action), last_index + 1, total);
                }
                let weight = acc_weight * crit_weight * roll_weight * hit_weight * stop_weight;
                for (extra, mut expanded) in spread_secondaries(state, action, hits > 1)? {
                    if reached {
                        thaw_on_hit(&mut expanded, action, mv, target);
                    }
                    outcomes.push((weight * extra, expanded));
                }
                }
            }
        }
        }
    }
    if outcomes.is_empty() {
        // Python's fallback here carries no note.
        turn.rolls_stratified = false;
        outcomes.push((1.0, turn));
    }
    Ok(outcomes)
}

/// Hit chance in [0, 1].
///
/// Toxic never misses when a Poison type uses it, from gen 8 on. The rule is not in the
/// move's data -- `toxic` still says `accuracy: 90`, with a comment pointing at the hook --
/// so reading the dump alone leaves a Poison type's Toxic failing one time in ten:
/// `move.alwaysHit || (move.id === 'toxic' && this.battle.gen >= 8 &&
/// pokemon.hasType('Poison')) || ... accuracy = true`.
fn accuracy_of(turn: &Turn, mv: &Move, attacker: &Battler, defender: &Battler) -> f64 {
    if mv.accuracy.is_none() || mv.always_hit {
        return 1.0;
    }
    if attacker.ability == "noguard" || defender.ability == "noguard" {
        return 1.0;
    }
    if mv.id == "toxic" && attacker.types.contains("Poison") {
        return 1.0;
    }
    let mut accuracy = mv.accuracy.unwrap() as f64;
    let weather = turn.pos.field.weather.as_ref().map(|w| w.as_str());
    if mv.id == "blizzard" && matches!(weather, Some("hail" | "snowscape" | "snow")) {
        return 1.0;
    }
    if mv.id == "thunder" || mv.id == "hurricane" {
        if matches!(weather, Some("raindance" | "primordialsea")) {
            return 1.0;
        }
        if matches!(weather, Some("sunnyday" | "desolateland")) {
            accuracy = 50.0;
        }
    }
    let accuracy = showdown_accuracy(turn, mv, attacker, defender, accuracy as i64);
    // `randomChance(accuracy, 100)` is `this.random(100) < accuracy` (sim/prng.ts): an
    // integer from 0 to 99, so the chance is `accuracy / 100` and anything past 100 hits.
    accuracy.clamp(0, 100) as f64 / 100.0
}

/// The number `hitStepAccuracy` rolls against (IKA-231), all in integers as Showdown has it
/// (sim/battle-actions.ts, vendor a5df827; the champions mod does not override the step):
///
/// ```text
/// accuracy = this.battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy);
/// if (!move.ignoreAccuracy) boost = clampIntRange(boosts['accuracy'], -6, 6);
/// if (!move.ignoreEvasion) boost = clampIntRange(boost - boosts['evasion'], -6, 6);
/// if (boost > 0) accuracy = trunc(accuracy * (3 + boost) / 3);
/// else if (boost < 0) accuracy = trunc(accuracy * 3 / (3 - boost));
/// ```
///
/// The ModifyAccuracy handlers each `chainModify` one 4096-based modifier, and `runEvent`
/// applies it once at the end with `modify` (half rounded down). The handlers run by
/// priority: -1 (Compound Eyes 5325, Hustle 3277 on a physical move, Snow Cloak 3277), then
/// -2 (Wide Lens 4505, Bright Powder 3686). The chain rounds at each step, and the first
/// step from 4096 is exact, so the order inside a priority only matters for three or more
/// modifiers with two at -2: Wide Lens and Bright Powder after Compound Eyes or Hustle.
/// Showdown orders those two by the holders' speed; this takes the user's first.
///
/// An accuracy-ignoring move (`ignoreEvasion`) still reads the user's own accuracy stage.
fn showdown_accuracy(
    turn: &Turn,
    mv: &Move,
    attacker: &Battler,
    defender: &Battler,
    accuracy: i64,
) -> i64 {
    let accuracy = accuracy_modifiers(turn, mv, attacker, defender).apply(accuracy);
    let mut boost = attacker.boost("accuracy").clamp(-6, 6);
    if !mv.ignore_evasion {
        boost = (boost - defender.boost("evasion")).clamp(-6, 6);
    }
    if boost > 0 {
        accuracy * (3 + boost) / 3
    } else if boost < 0 {
        accuracy * 3 / (3 - boost)
    } else {
        accuracy
    }
}

/// The `ModifyAccuracy` handlers' chain (IKA-231), for `showdown_accuracy` and
/// `later_hit_chance`.
fn accuracy_modifiers(
    turn: &Turn,
    mv: &Move,
    attacker: &Battler,
    defender: &Battler,
) -> crate::fixedpoint::Chain {
    let mut chain = crate::fixedpoint::Chain::new();
    // onSourceModifyAccuracyPriority / onModifyAccuracyPriority: -1.
    if attacker.ability == "compoundeyes" {
        chain.add_fp(5325, "compoundeyes");
    }
    if attacker.ability == "hustle" && mv.category == "Physical" {
        chain.add_fp(3277, "hustle");
    }
    if snow_cloak_applies(turn, attacker, defender) {
        chain.add_fp(3277, "snowcloak");
    }
    // Priority -2.
    if is(attacker.item, "widelens") {
        chain.add_fp(4505, "widelens");
    }
    if is(defender.item, "brightpowder") {
        chain.add_fp(3686, "brightpowder");
    }
    chain
}

/// Whether each hit after the first rolls accuracy again: `multiaccuracy` (Triple Axel,
/// Population Bomb; Triple Kick is not in either regulation), unless Skill Link's
/// `onModifyMove` deleted it. Loaded Dice deletes it too, and is `isNonstandard: "Past"`
/// in both Champions mods.
fn rolls_each_hit(mv: &Move, attacker: &Battler) -> bool {
    mv.raw.get("multiaccuracy").is_some_and(|v| v.as_bool() == Some(true))
        && attacker.ability != "skilllink"
}

/// The chance that a later hit of a `multiaccuracy` move lands (IKA-235), from the live
/// state before that hit; None when the user is gone, which ends the move before the roll.
///
/// `hitStepMoveHitLoop` (sim/battle-actions.ts 911-940, vendor a5df827; the champions
/// mod's copy is data/mods/champions/scripts.ts 482-511) does not use `hitStepAccuracy`:
///
/// ```text
/// let accuracy = move.accuracy;
/// const boostTable = [1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3];
/// if (!move.ignoreAccuracy) { boost = clamp(user accuracy);
///     if (boost > 0) accuracy *= boostTable[boost]; else accuracy /= boostTable[-boost]; }
/// if (!move.ignoreEvasion) { boost = clamp(target evasion);
///     if (boost > 0) accuracy /= boostTable[boost]; else if (boost < 0) accuracy *= boostTable[-boost]; }
/// accuracy = this.battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy);
/// if (!move.alwaysHit) { accuracy = runEvent('Accuracy', ...);
///     if (accuracy !== true && !this.battle.randomChance(accuracy, 100)) break; }
/// ```
///
/// So the two stages are floats, each clamped and applied on its own, not the first hit's
/// one combined integer stage. `runEvent` applies its modifier chain only to a
/// non-negative integer (`relayVar === Math.abs(Math.floor(relayVar))`, sim/battle.ts
/// 929), so a staged 67.5 keeps no Compound Eyes or Wide Lens. `randomChance(x, 100)` is
/// `random(100) < x` over the integers 0-99: the chance is `ceil(x) / 100`, 68% for 67.5.
/// The arithmetic is f64 in the same order, so the float residue is Showdown's too (for a
/// 90% move it only shows past 100: -2 accuracy against -4 evasion is 126.00000000000001).
/// No Guard answers the `Accuracy` event with true.
fn later_hit_chance(
    turn: &Turn,
    mv: &Move,
    action: &QueuedAction,
    target: Slot,
) -> Result<Option<f64>, String> {
    let Some(attacker) = turn.battler_at(action.side, action.slot)? else {
        return Ok(None);
    };
    let Some(defender) = turn.battler_at(target.0, target.1)? else {
        return Ok(None);
    };
    let Some(base) = mv.accuracy else { return Ok(Some(1.0)) };
    if mv.always_hit || attacker.ability == "noguard" || defender.ability == "noguard" {
        return Ok(Some(1.0));
    }
    const BOOST_TABLE: [f64; 7] = [1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0, 8.0 / 3.0, 3.0];
    let mut accuracy = base as f64;
    let boost = attacker.boost("accuracy").clamp(-6, 6);
    if boost > 0 {
        accuracy *= BOOST_TABLE[boost as usize];
    } else {
        accuracy /= BOOST_TABLE[(-boost) as usize];
    }
    if !mv.ignore_evasion {
        let boost = defender.boost("evasion").clamp(-6, 6);
        if boost > 0 {
            accuracy /= BOOST_TABLE[boost as usize];
        } else if boost < 0 {
            accuracy *= BOOST_TABLE[(-boost) as usize];
        }
    }
    if accuracy >= 0.0 && accuracy == accuracy.floor() {
        accuracy = accuracy_modifiers(turn, mv, &attacker, &defender).apply(accuracy as i64) as f64;
    }
    Ok(Some(accuracy.ceil().clamp(0.0, 100.0) / 100.0))
}

/// Snow Cloak (IKA-222): `onModifyAccuracy`, `chainModify([3277, 4096])` while
/// `this.field.isWeather(['hail', 'snowscape'])`, `flags: { breakable: 1 }`. `isWeather`
/// reads the effective weather, which an active Cloud Nine or Air Lock clears. Its modifier
/// joins the event's chain in `showdown_accuracy`.
fn snow_cloak_applies(turn: &Turn, attacker: &Battler, defender: &Battler) -> bool {
    if defender.ability != "snowcloak" || is_mold_breaker(attacker.ability.as_str()) {
        return false;
    }
    let weather = turn.pos.field.weather.as_ref().map(|w| w.as_str());
    if !matches!(weather, Some("hail" | "snowscape" | "snow")) {
        return false;
    }
    !turn.pos.sides.iter().any(|side| {
        (0..side.active.len()).any(|slot| {
            matches!(side.active_pokemon(slot), Some(mon)
                if !mon.fainted && crate::effects::suppresses_weather(mon.ability.as_str()))
        })
    })
}

/// Spiky Shield and friends punish the blocked attacker (contact moves only).
fn protect_punish(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    blocked: &str,
) -> Result<(), String> {
    if !mv.has_flag(F_CONTACT) {
        return Ok(());
    }
    let me = (action.side, action.slot);
    let maxhp = match turn.mon_at(me.0, me.1) {
        None => return Ok(()),
        Some(mon) => mon.maxhp,
    };
    match blocked {
        "spikyshield" => {
            turn.deal_damage(me.0, me.1, (maxhp / 8).max(1), false, "spikyshield")?;
        }
        "banefulbunker" => {
            turn.apply_status(me.0, me.1, "psn", "banefulbunker")?;
        }
        "burningbulwark" => {
            turn.apply_status(me.0, me.1, "brn", "burningbulwark")?;
        }
        "kingsshield" => {
            turn.apply_boosts(me.0, me.1, &[("atk", -1)], true, "kingsshield");
        }
        "obstruct" => {
            turn.apply_boosts(me.0, me.1, &[("def", -2)], true, "obstruct");
        }
        "silktrap" => {
            turn.apply_boosts(me.0, me.1, &[("spe", -1)], true, "silktrap");
        }
        _ => {}
    }
    Ok(())
}

/// What an immune defender gains from the hit it just shrugged off. Treating these purely
/// as immunities loses half the mechanic: Dry Skin heals off a Water move and Lightning Rod
/// gains Special Attack from an Electric one, and either can decide the next turn.
fn absorb(turn: &mut Turn, mv: &Move, target: Slot) {
    let (ability, maxhp) = match turn.mon_at(target.0, target.1) {
        None => return,
        Some(mon) if mon.fainted => return,
        Some(mon) => (mon.ability, mon.maxhp),
    };
    let heals = match ability.as_str() {
        "waterabsorb" | "dryskin" => Some("Water"),
        "voltabsorb" => Some("Electric"),
        "eartheater" => Some("Ground"),
        _ => None,
    };
    if heals == Some(mv.mtype.as_str()) {
        turn.heal(target.0, target.1, (maxhp / 4).max(1), ability.as_str());
        return;
    }
    let boosts: Option<(&str, &[(&str, i64)])> = match ability.as_str() {
        "lightningrod" => Some(("Electric", &[("spa", 1)])),
        "stormdrain" => Some(("Water", &[("spa", 1)])),
        "motordrive" => Some(("Electric", &[("spe", 1)])),
        "sapsipper" => Some(("Grass", &[("atk", 1)])),
        "windrider" => Some(("Flying", &[("atk", 1)])),
        "wellbakedbody" => Some(("Fire", &[("def", 2)])),
        "steamengine" => Some(("Fire", &[("spe", 6)])),
        _ => None,
    };
    if let Some((wanted, table)) = boosts {
        if wanted == mv.mtype.as_str() {
            turn.apply_boosts(target.0, target.1, table, false, ability.as_str());
            return;
        }
    }
    if ability == "flashfire" && mv.mtype == "Fire" {
        turn.add_volatile(target.0, target.1, "flashfire", None);
    }
}

/// Drain, item reactions, contact effects and secondaries, for one hit on one target.
///
/// `landed` is any hit that reached the target, including the two that deal 0: a hit
/// Disguise took, and a hit into Endure at 1 HP. 0 is still a damaging hit to Showdown --
/// `DamagingHit` runs for any numeric damage -- so the handlers gated on `landed` fire for
/// them too (IKA-157, IKA-171). A damaging move can also carry a volatile or a status
/// outright, not only as a chance-based secondary: Infestation's trap, Salt Cure, Nuzzle's
/// paralysis; a trap records who applied it, since it ends when that Pokemon leaves.
///
/// Spicy Spray (Mega Scovillain) is an `onDamagingHit` too, but *not* contact-gated and not
/// rolled -- `onDamagingHit(damage, target, source, move) { source.trySetStatus('brn',
/// target); }` -- so any damaging hit burns the attacker, Moonblast included. It was once
/// declared modelled because it changes no damage number, which was true of the calculator
/// and wrong of the resolver; unlike Static and Flame Body (1-in-3, reported rather than
/// branched) it is certain and simply applied. The contact effects run from `damage()`,
/// before the faint is processed, so Rough Skin still hurts the attacker when the Pokemon
/// holding it is knocked out by that very hit.
///
/// A secondary under a budget that does not branch them is collapsed to "it did not
/// happen" and said out loud, because under such a budget a Rock Slide never flinches --
/// except under the pinned policy, where the oracle answers every secondary roll with no,
/// so the collapse is exact and declaring it would flag turns that are right. A branched one
/// is pushed to `pending_secondaries`: this function holds one state, and the hit loop fans
/// it out.
fn after_hit(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
    dealt: i64,
    landed: bool,
    budget: &Budget,
    type_mod: i64,
) -> Result<(), String> {
    let me = (action.side, action.slot);
    turn.move_damage_total += dealt;
    turn.move_connected = true;

    // Drain heals inside `spreadDamage`, per target and rounded per target, before any
    // `DamagingHit` handler (IKA-161): Python's `_after_hit`.
    if let Some(drain) = mv.drain.as_ref() {
        if dealt > 0 {
            let amount = round_fraction(dealt, drain);
            turn.heal(me.0, me.1, amount, "drain");
        }
    }

    let defender_alive = matches!(turn.mon_at(target.0, target.1), Some(m) if !m.fainted);

    if defender_alive {
        if let Some(vid) = mv.volatile_status.as_deref() {
            let vid = vid.to_string();
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("move volatile: {vid}"));
            }
            let duration = effect_duration(turn, mv, &vid, action.side, action.slot);
            if duration_is_rolled(mv, &vid) {
                turn.report(format!(
                    "{vid} duration (Showdown rolls it; pinned to the low end)"
                ));
            }
            turn.add_volatile(target.0, target.1, &vid, duration);
            if vid == "partiallytrapped" {
                if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
                    if let Some(applied) = mon.volatile_mut("partiallytrapped") {
                        applied.source_slot = Some(Id::new(&format!("{}{}", me.0, me.1)));
                    }
                }
            }
        }
        if let Some(status) = mv.status.as_deref() {
            let status = status.to_string();
            crate::resolve::apply_status_from(turn, target, &status, me, mv.id.as_str())?;
        }
    }

    let defender_ability = turn.mon_at(target.0, target.1).map(|m| m.ability);
    if landed && matches!(defender_ability, Some(a) if a.as_str() == "spicyspray") {
        crate::resolve::apply_status_from(turn, me, "brn", target, "spicyspray")?;
    }

    // Throat Chop adds its own condition from a 100%-chance `secondary.onHit`, so there is
    // nothing declarative in the dump to drive it.
    if mv.id == "throatchop" && landed && defender_alive {
        let duration = effect_duration(turn, mv, "throatchop", action.side, action.slot);
        turn.add_volatile(target.0, target.1, "throatchop", duration);
        log_event!(turn, "{} cannot use sound moves (throatchop)", Name(target.0, target.1));
    }

    // Cursed Body: `onDamagingHit` with `randomChance(3, 10)`, gated on neither contact
    // nor the target surviving -- the handler runs from `damage()`, before the faint is
    // processed. 43 of the 394 tournament teams carry it.
    if landed && matches!(defender_ability, Some(a) if a.as_str() == "cursedbody") {
        let attacker_free = turn
            .mon_at(me.0, me.1)
            .map(|mon| !mon.has_volatile("disable"))
            .unwrap_or(false);
        if attacker_free {
            if budget.enumerate_secondary {
                turn.pending_secondaries.push((0.3, json!({ "disable": true }), me));
            } else if !budget.pinned_policy {
                // Same reasoning as a secondary: under the pinned policy `randomChance`
                // is answered with no, so not applying it is exact rather than approximate.
                turn.report("cursedbody (30% disable, not branched)");
            }
        }
    }

    if mv.has_flag(F_CONTACT) && landed {
        let (ability, item) = match turn.mon_at(target.0, target.1) {
            None => (None, None),
            Some(mon) => (Some(mon.ability), mon.item),
        };
        let attacker_maxhp = turn.mon_at(me.0, me.1).map(|m| m.maxhp).unwrap_or(0);
        if let Some(barbs) = ability.filter(|a| matches!(a.as_str(), "roughskin" | "ironbarbs")) {
            turn.deal_damage(me.0, me.1, (attacker_maxhp / 8).max(1), false, barbs.as_str())?;
        }
        if is(item, "rockyhelmet") {
            turn.deal_damage(me.0, me.1, (attacker_maxhp / 6).max(1), false, "rockyhelmet")?;
        }
        if let Some(ability) = ability {
            if matches!(
                ability.as_str(),
                "static" | "flamebody" | "effectspore" | "poisonpoint" | "cutecharm"
            ) {
                turn.report(format!("contact ability: {ability}"));
            }
        }
    }

    if matches!(attacker_ability_of(turn, me), Some(a) if a.as_str() == "poisontouch")
        && mv.has_flag(F_CONTACT)
    {
        turn.report("ability: poisontouch (30% poison not branched)");
    }

    // A resist berry is eaten only by a hit it actually weakened. Occa Berry's handler is
    // `if (move.type === 'Fire' && typeMod > 0) { if (target.eatItem()) ... }`, so a Fire
    // move that is *not* super effective leaves the berry alone. Chilan Berry is the one
    // exception: it halves Normal regardless of effectiveness.
    let eats_berry = match turn.mon_at(target.0, target.1) {
        None => false,
        Some(mon) if mon.fainted => false,
        Some(mon) => match mon.item.and_then(|i| resist_berry(i.as_str())) {
            None => false,
            Some(berry_type) => {
                berry_type == mv.mtype.as_str() && (berry_type == "Normal" || type_mod > 0)
                    // The berry is `onSourceModifyDamage`, which a `damageCallback` never
                    // reaches (IKA-213), nor a level move; Struggle is `???` (IKA-239).
                    && !crate::level_struggle::misses_resist_berry(mv.id.as_str())
            }
        },
    };
    if eats_berry && !turn.berries_blocked(target.0) {
        let berry = turn.mon_at(target.0, target.1).and_then(|m| m.item);
        turn.consume_item(target.0, target.1, berry.as_ref().map(|b| b.as_str()).unwrap_or(""));
    }

    if landed {
        crate::airborne::pop_air_balloon(turn, target);
    }

    // Knock Off removes what it hit; Thief and Covet take it when the attacker has
    // nothing. A mega stone refuses to leave the species it belongs to, and only that
    // species, which `item_is_removable` reads from the regulation's own mega map.
    if matches!(mv.id.as_str(), "knockoff" | "thief" | "covet") {
        let target_item = match turn.mon_at(target.0, target.1) {
            None => None,
            Some(mon) if mon.fainted => None,
            Some(mon) => mon.item,
        };
        if let Some(item) = target_item {
            let species = turn.mon_at(target.0, target.1).unwrap().species;
            let removable = turn
                .reg
                .item_is_removable(species.as_str(), Some(item.as_str()));
            let attacker_empty =
                matches!(turn.mon_at(me.0, me.1), Some(mon) if mon.item.is_none());
            if removable && mv.id == "knockoff" {
                turn.consume_item(target.0, target.1, "knockoff");
            } else if removable && attacker_empty {
                turn.consume_item(target.0, target.1, mv.id.as_str());
                if let Some(mon) = turn.mon_at_mut(me.0, me.1) {
                    mon.item = Some(item);
                }
                log_event!(turn, "{} stole {}", Name(me.0, me.1), item);
            }
        }
    }

    let attacker_ability = turn.mon_at(me.0, me.1).map(|m| m.ability);
    if !mv.secondaries.is_empty() {
        for secondary in &mv.secondaries {
            if matches!(attacker_ability, Some(a) if a.as_str() == "sheerforce") {
                continue;
            }
            let chance = secondary
                .get("chance")
                .and_then(Value::as_f64)
                .unwrap_or(100.0)
                / 100.0;
            check_secondary_supported(secondary)?;
            if chance >= 1.0 {
                apply_secondary(turn, action, secondary, target)?;
                continue;
            }
            if !budget.enumerate_secondary {
                if !budget.pinned_policy {
                    turn.report(format!(
                        "secondary {}%: {} (not branched)",
                        (chance * 100.0) as i64,
                        mv.id
                    ));
                }
                continue;
            }
            turn.pending_secondaries.push((chance, secondary.clone(), target));
        }
    }

    // Dragon Tail, Circle Throw: `spreadMoveHit`'s step 6, `forceSwitch` (IKA-208).
    if mv.force_switch {
        let _ = raise_force_switch(turn, (action.side, action.slot), target);
    }

    on_being_hit(turn, mv, target, action.side)?;
    lay_hazard_after_hit(turn, action, mv, dealt > 0);
    Ok(())
}

/// Stone Axe's Stealth Rock and Ceaseless Edge's Spikes, from the move's own `onAfterHit`
/// (IKA-173): on the user's foe's side, whoever was hit, after `DamagingHit` and only while
/// the user stands, and not under Sheer Force. Python's `_lay_hazard_after_hit`.
fn lay_hazard_after_hit(turn: &mut Turn, action: &QueuedAction, mv: &Move, landed: bool) {
    let cid = match mv.id.as_str() {
        "stoneaxe" => "stealthrock",
        "ceaselessedge" => "spikes",
        _ => return,
    };
    if !landed {
        return;
    }
    let stands = matches!(
        turn.mon_at(action.side, action.slot),
        Some(mon) if !mon.fainted && mon.ability.as_str() != "sheerforce"
    );
    if stands {
        turn.add_side_condition(1 - action.side, cid, None);
    }
}

fn check_secondary_supported(secondary: &Value) -> Result<(), String> {
    for key in secondary.as_object().map(|o| o.keys().collect::<Vec<_>>()).unwrap_or_default() {
        if !matches!(key.as_str(), "chance" | "status" | "volatileStatus" | "boosts" | "self")
        {
            return Err(format!("secondary field: {key}"));
        }
    }
    Ok(())
}

fn apply_secondary(
    turn: &mut Turn,
    action: &QueuedAction,
    secondary: &Value,
    target: Slot,
) -> Result<(), String> {
    if secondary.get("disable").and_then(Value::as_bool).unwrap_or(false) {
        // Cursed Body, pushed through the secondary fan-out because it is the same shape:
        // a chance whose consequence outlives the turn.
        apply_disable(turn, target.0, target.1, None);
        return Ok(());
    }
    if let Some(status) = secondary.get("status").and_then(Value::as_str) {
        let status = status.to_string();
        crate::resolve::apply_status_from(turn, target, &status, (action.side, action.slot), "secondary")?;
    }
    if let Some(vid) = secondary.get("volatileStatus").and_then(Value::as_str) {
        let vid = vid.to_string();
        if !crate::resolve::volatile_is_handled(&vid) {
            return Err(format!("secondary volatile: {vid}"));
        }
        let fresh = !turn
            .mon_at(target.0, target.1)
            .is_some_and(|m| m.has_volatile("confusion"));
        turn.add_volatile(target.0, target.1, &vid, None);
        if is(action.move_id, "axekick") && fresh {
            confused_by_axe_kick(turn, target);
        }
    }
    if let Some(boosts) = secondary.get("boosts").and_then(Value::as_object) {
        let table: Vec<(&str, i64)> = boosts
            .iter()
            .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
            .collect();
        turn.apply_boosts(target.0, target.1, &table, true, "secondary");
    }
    if let Some(self_boosts) = secondary
        .get("self")
        .and_then(|s| s.get("boosts"))
        .and_then(Value::as_object)
    {
        let table: Vec<(&str, i64)> = self_boosts
            .iter()
            .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
            .collect();
        turn.apply_boosts(action.side, action.slot, &table, false, "secondary");
    }
    Ok(())
}

/// Fans one resolved hit out over the secondaries it would roll: (weight, state) pairs
/// summing to one, and the state itself when nothing is pending, which is the
/// overwhelmingly common case.
///
/// Each secondary is independent, so the fan-out is a cross product, capped at
/// `MAX_BRANCHED_SECONDARIES` per hit: each doubles the states for that hit and the turn's
/// tree is the product over every hit, so four targets each carrying two secondaries would
/// be 256 states from this alone. Two is enough for every real move (a spread move's two
/// targets each carry one); past the cap the rest are collapsed to "did not happen" and
/// reported.
fn spread_secondaries<'a>(
    mut state: Turn<'a>,
    action: &QueuedAction,
    multihit: bool,
) -> Result<Vec<(f64, Turn<'a>)>, String> {
    if state.pending_secondaries.is_empty() {
        return Ok(vec![(1.0, state)]);
    }
    let mut pending = std::mem::take(&mut state.pending_secondaries);
    if multihit {
        // Cursed Body is pushed once per hit, and `if (source.volatiles['disable']) return;`
        // makes every try after the one that lands a no-op: n tries of p are one chance of
        // 1 - (1 - p)^n, and what it does does not depend on which hit it was (IKA-208).
        let mut kept: Vec<(f64, Value, Slot)> = Vec::new();
        for (chance, secondary, target) in pending {
            let disable = secondary.get("disable").and_then(Value::as_bool) == Some(true);
            let merged = kept.iter_mut().find(|(_, s, t)| {
                disable && *t == target && s.get("disable").and_then(Value::as_bool) == Some(true)
            });
            match merged {
                Some(entry) => entry.0 = 1.0 - (1.0 - entry.0) * (1.0 - chance),
                None => {
                    if !disable {
                        // No move in either regulation has one; Python's answer and note.
                        state.report("secondary on a multi-hit move (applied after the last hit)");
                    }
                    kept.push((chance, secondary, target));
                }
            }
        }
        pending = kept;
    }
    if pending.len() > MAX_BRANCHED_SECONDARIES {
        // Python's `_spread_secondaries`: the tail past the cap did not happen, and says so.
        // No move in either regulation reaches it (IKA-208).
        for (chance, _secondary, _target) in pending.drain(MAX_BRANCHED_SECONDARIES..) {
            state.report(format!(
                "secondary {}%: beyond the {MAX_BRANCHED_SECONDARIES} branched on one hit (not branched)",
                (chance * 100.0) as i64
            ));
        }
    }
    let mut out: Vec<(f64, Turn<'a>)> = vec![(1.0, state)];
    for (chance, secondary, target) in pending {
        let mut expanded: Vec<(f64, Turn<'a>)> = Vec::new();
        for (weight, current) in out.into_iter() {
            let mut fired = current.clone();
            fired.pending_secondaries.clear();
            apply_secondary(&mut fired, action, &secondary, target)?;
            expanded.push((weight * chance, fired));
            expanded.push((weight * (1.0 - chance), current));
        }
        out = expanded;
    }
    Ok(out)
}

/// Abilities that trigger on the defender taking a hit: (stat changes, the move types that
/// trigger it -- empty means any type). The ones whose effect depends on how much HP was
/// lost or on a chance are not modelled and are reported.
///
/// Toxic Debris (IKA-173):
///
/// ```text
/// onDamagingHit(damage, target, source, move) {
///     const side = source.isAlly(target) ? source.side.foe : source.side;
///     const toxicSpikes = side.sideConditions['toxicspikes'];
///     if (move.category === 'Physical' && (!toxicSpikes || toxicSpikes.layers < 2)) {
/// ```
///
/// The attacker's side, or its foe's for a partner's hit: the side across from Glimmora
/// either way, so the attacker's side is not read. `onDamagingHit` runs before the faint is
/// processed, so a Glimmora knocked out by the hit lays them too.
fn on_being_hit(
    turn: &mut Turn,
    mv: &Move,
    target: Slot,
    _attacker_side: usize,
) -> Result<(), String> {
    let (ability, fainted) = match turn.mon_at(target.0, target.1) {
        None => return Ok(()),
        Some(mon) => (mon.ability, mon.fainted),
    };
    // Toxic Debris runs from `onDamagingHit`, before the faint, and lays them across from
    // Glimmora: the attacker's side, or its foe's for a partner's hit (IKA-173).
    if ability == "toxicdebris" && mv.category == "Physical" {
        let across = 1 - target.0;
        let layers = turn.pos.sides[across]
            .side_condition("toxicspikes")
            .and_then(|c| c.layers)
            .unwrap_or(0);
        if layers < 2 {
            turn.add_side_condition(across, "toxicspikes", None);
        }
    }
    if fainted {
        return Ok(());
    }
    let entry: Option<(&[(&str, i64)], &[&str])> = match ability.as_str() {
        "stamina" => Some((&[("def", 1)], &[])),
        "weakarmor" => Some((&[("def", -1), ("spe", 2)], &[])),
        "justified" => Some((&[("atk", 1)], &["Dark"])),
        "rattled" => Some((&[("spe", 1)], &["Bug", "Dark", "Ghost"])),
        "steamengine" => Some((&[("spe", 6)], &["Fire", "Water"])),
        "watercompaction" => Some((&[("def", 2)], &["Water"])),
        _ => None,
    };
    if let Some((boosts, types)) = entry {
        if types.is_empty() || types.contains(&mv.mtype.as_str()) {
            turn.apply_boosts(target.0, target.1, boosts, false, ability.as_str());
        }
    }
    // Cursed Body stays on this list even though the Disable is branched: Python reports
    // it here regardless, and the report is part of what a turn returns.
    if matches!(ability.as_str(), "angerpoint" | "berserk" | "angershell" | "cursedbody") {
        turn.report(format!("on-hit ability: {ability}"));
    }
    Ok(())
}

fn attacker_ability_of(turn: &Turn, me: Slot) -> Option<Id> {
    turn.mon_at(me.0, me.1).map(|mon| mon.ability)
}

/// Records that the user of a self-switching move has to be replaced.
///
/// Which Pokemon comes in is the player's choice, so nothing is picked here. The mark is what
/// suspends the turn: `run_queue` sees it and hands the branch back with its remaining
/// queue, and `resume_turn` continues once the choice is made. A Pokemon with an empty bench
/// is not marked at all -- Showdown's `switchFlag` has nothing to answer it with, and the
/// move simply leaves it in place.
fn mark_self_switch(turn: &mut Turn, action: &QueuedAction) {
    let alive = matches!(turn.mon_at(action.side, action.slot), Some(mon) if !mon.fainted);
    if !alive {
        return;
    }
    // Red Card's drag runs first and takes the U-turn's flag with it (IKA-191).
    if matches!(turn.mon_at(action.side, action.slot), Some(mon) if mon.has_volatile("pendingforceswitch")) {
        return;
    }
    let bench = turn.pos.sides[action.side]
        .pokemon
        .iter()
        .filter(|mon| !mon.fainted && !mon.is_active())
        .count();
    if bench == 0 {
        return;
    }
    // IKA-210's positive control: the old Showdown (before a5df827), whose Eject Button and
    // Emergency Exit cancelled the attacker's self-switch -- here, whenever a foe holds one.
    if cfg!(feature = "ika210-control") {
        let foe = 1 - action.side;
        let exits = (0..turn.pos.sides[foe].active.len()).any(|slot| {
            matches!(turn.mon_at(foe, slot), Some(mon) if !mon.fainted
                && (mon.item.map(|i| i.as_str() == "ejectbutton").unwrap_or(false)
                    || mon.ability.as_str() == "emergencyexit"))
        });
        if exits {
            return;
        }
    }
    turn.add_volatile(action.side, action.slot, "pendingselfswitch", None);
    turn.self_switch_pending = true;
    log_event!(turn, "{} must switch out", Name(action.side, action.slot));
}

// ---------------------------------------------------------------------------
// Eject Button, Red Card, Emergency Exit and Wimp Out (IKA-191)
// ---------------------------------------------------------------------------

/// `resolve.EMERGENCY_EXIT_ABILITIES`: the abilities that switch their holder out when it
/// drops to half its HP or below. The champions mod gives both the same `onEmergencyExit`
/// (data/mods/champions/abilities.ts), which no longer clears anyone else's `switchFlag`
/// (57ecb348b, IKA-164).
fn is_emergency_exit(ability: &str) -> bool {
    matches!(ability, "emergencyexit" | "wimpout")
}

/// Python's `_active_hp`: every conscious active Pokemon's HP, -1 elsewhere.
fn active_hp(turn: &Turn) -> [[i64; 2]; 2] {
    let mut out = [[-1; 2]; 2];
    for (side, row) in out.iter_mut().enumerate() {
        for (slot, hp) in row.iter_mut().enumerate().take(turn.pos.sides[side].active.len()) {
            if let Some(mon) = turn.mon_at(side, slot) {
                if !mon.fainted {
                    *hp = mon.hp;
                }
            }
        }
    }
    out
}

/// The field one target's damage is computed on (IKA-222).
///
/// Showdown computes a spread move's damage for every target before any is dealt
/// (`getSpreadDamage`, then `spreadDamage`), and a Pokemon knocked out by the move stays on
/// the field until `faintMessages` after it. This port hits the targets one by one, so a
/// Ruin holder the move has already knocked out is put back for the targets after it --
/// it still lowers their stat in Showdown. Only the Ruin abilities are put back: the other
/// field abilities (Friend Guard, the auras, Cloud Nine) keep the reading they had.
fn field_for_hit(turn: &Turn) -> crate::battler::FieldState {
    let mut field = turn.field();
    let Some(start) = turn.move_start_hp else { return field };
    for (side, row) in start.iter().enumerate() {
        for (slot, before) in row.iter().enumerate() {
            if *before <= 0 {
                continue;
            }
            if let Some(mon) = turn.mon_at(side, slot) {
                if mon.fainted && mon.ability.as_str().ends_with("ofruin") {
                    field.active_abilities[side].push(mon.ability);
                }
            }
        }
    }
    field
}

/// Python's `_begin_move_watch`.
fn begin_move_watch(turn: &mut Turn) {
    turn.move_start_hp = Some(active_hp(turn));
    turn.move_hit = [[false; 2]; 2];
}

/// `battle.canSwitch(side)` (Python's `_can_switch`).
fn can_switch(turn: &Turn, side: usize) -> bool {
    turn.pos.sides[side].pokemon.iter().any(|mon| !mon.fainted && !mon.is_active())
}

/// `hp && hp <= maxhp / 2 && before > maxhp / 2` (Python's `_crossed_half`).
fn crossed_half(hp: i64, maxhp: i64, before: i64) -> bool {
    hp > 0 && 2 * hp <= maxhp && 2 * before > maxhp
}

/// `runEvent('DragOut', target, source)`: Guard Dog and Suction Cups (both `breakable`)
/// and Ingrain's volatile answer `null` -- the Pokemon stays, and nothing fails (IKA-208).
fn drag_out_stopped(turn: &Turn, at: Slot, source: Option<Slot>) -> bool {
    let Some(mon) = turn.mon_at(at.0, at.1) else { return true };
    if mon.has_volatile("ingrain") {
        return true;
    }
    matches!(mon.ability.as_str(), "guarddog" | "suctioncups") && !ability_broken_by(turn, mon, source)
}

/// `forceSwitch` in `spreadMoveHit` (sim/battle-actions.ts): `if (target.hp > 0 &&
/// source.hp > 0 && this.battle.canSwitch(target.side))` and `DragOut` lets it, the target's
/// `forceSwitchFlag` -- here `pendingforceswitch`, which `resolve::drag_in` answers at the
/// end of the action. Says whether it was raised (IKA-208).
fn raise_force_switch(turn: &mut Turn, source: Slot, target: Slot) -> bool {
    let standing = |turn: &Turn, at: Slot| matches!(turn.mon_at(at.0, at.1), Some(m) if !m.fainted && m.hp > 0);
    if !standing(turn, target) || !standing(turn, source) || !can_switch(turn, target.0) {
        return false;
    }
    if drag_out_stopped(turn, target, Some(source)) {
        return false;
    }
    if !turn.mon_at(target.0, target.1).is_some_and(|m| m.has_volatile("pendingforceswitch")) {
        turn.add_volatile(target.0, target.1, "pendingforceswitch", None);
        log_event!(turn, "{} is forced out", Name(target.0, target.1));
    }
    true
}

fn switch_flagged(mon: &crate::position::Pokemon) -> bool {
    mon.has_volatile("pendingselfswitch") || mon.has_volatile("pendingforceswitch")
}

/// Python's `_emergency_exit`: the holder's `switchFlag`, answered mid-turn like U-turn's
/// and after the residual phase with the faint replacements.
///
/// The champions mod's `onEmergencyExit`, for a holder that has just crossed half:
/// `if (!this.canSwitch(target.side) || target.forceSwitchFlag || target.switchFlag)
/// return; target.switchFlag = true;`. The flag is the same one U-turn sets, so it is
/// answered the same way: mid-turn it suspends the turn for the holder's player to choose;
/// after the residual phase it is owed with the faint replacements, which is the one
/// request Showdown makes there.
fn emergency_exit(turn: &mut Turn, slot: Slot, mid_turn: bool) {
    let fires = matches!(turn.mon_at(slot.0, slot.1), Some(mon)
        if !mon.fainted && is_emergency_exit(mon.ability.as_str()) && !switch_flagged(mon));
    if !fires || !can_switch(turn, slot.0) {
        return;
    }
    turn.add_volatile(slot.0, slot.1, "pendingselfswitch", None);
    if mid_turn {
        turn.self_switch_pending = true;
    }
    if turn.log.is_some() {
        let ability = turn.mon_at(slot.0, slot.1).map(|m| m.ability).unwrap_or_default();
        log_event!(turn, "{} must switch out ({})", Name(slot.0, slot.1), ability);
    }
}

/// Python's `_exits_if_crossed`.
fn exits_if_crossed(turn: &mut Turn, slot: Slot, flagged: bool) {
    let Some(start) = turn.move_start_hp else { return };
    let before = start[slot.0][slot.1];
    if before < 0 || flagged {
        return;
    }
    let crossed = matches!(turn.mon_at(slot.0, slot.1), Some(mon) if crossed_half(mon.hp, mon.maxhp, before));
    if crossed {
        emergency_exit(turn, slot, true);
    }
}

/// Python's `_user_self_switches`: U-turn's flag, up before any of these checks run.
fn user_self_switches(turn: &Turn, mv: &Move) -> bool {
    mv.self_switch && turn.move_connected
}

/// Python's `_by_speed`: fastest first (Trick Room reversed), a tie reported.
fn by_speed(turn: &mut Turn, slots: Vec<Slot>) -> Result<Vec<Slot>, String> {
    if slots.len() < 2 {
        return Ok(slots);
    }
    let trick_room = turn.pos.field.trick_room();
    let field = turn.field();
    let mut keyed: Vec<(i64, String, usize, usize)> = Vec::new();
    for (side, slot) in slots {
        let mut speed = match turn.battler_at(side, slot)? {
            Some(fighter) => {
                let conditions = turn.pos.sides[side].side_conditions.clone();
                crate::speed::effective_speed(&fighter, &field, &conditions)
            }
            None => 0,
        };
        if trick_room {
            speed = 10000 - speed;
        }
        let species = turn
            .mon_at(side, slot)
            .map(|mon| mon.species.as_str().to_string())
            .unwrap_or_default();
        keyed.push((-speed, species, slot, side));
    }
    let mut speeds: Vec<i64> = keyed.iter().map(|k| k.0).collect();
    speeds.sort_unstable();
    speeds.dedup();
    if speeds.len() != keyed.len() {
        turn.report("item speed tie (Showdown breaks it at random)");
    }
    keyed.sort();
    Ok(keyed.into_iter().map(|(_, _, slot, side)| (side, slot)).collect())
}

/// Python's `_after_move_secondary_switches`: the user's Emergency Exit from `DamagingHit`
/// and recoil, then (not under Sheer Force) Eject Button, Red Card and the targets'
/// Emergency Exit.
///
/// Everything between a damaging move's recoil and its Life Orb that switches a Pokemon out
/// (vendor/pokemon-showdown/data/mods/champions/scripts.ts `hitStepMoveHitLoop`,
/// data/mods/champions/items.ts, data/items.ts):
///
/// 1. The user's Emergency Exit, for Rough Skin, Rocky Helmet (`DamagingHit`) or recoil
///    taking it across half. A U-turn user's flag is already up, so it never fires.
/// 2. `afterMoveSecondaryEvent`, which Sheer Force skips: Eject Button (priority 2) on a
///    target the move reached, unless any active Pokemon's `switchFlag === true` --
///    U-turn's is the move id, so a U-turn does not stop it, and the champions mod no
///    longer clears the attacker's flag (aa6d5f085). Then Red Card (priority 0), which drags
///    the attacker out at random: the forced-switch mark Dragon Tail leaves (IKA-191).
/// 3. Each target's Emergency Exit, also skipped by Sheer Force.
///
/// Holders act in the order `runEvent` sorts their handlers (`by_speed`). The user's
/// Emergency Exit after `AfterMoveSecondarySelf` (Life Orb) is checked again by
/// `useMoveInner`; `after_move` does that, and skips it under Sheer Force as Showdown does.
fn after_move_secondary_switches(turn: &mut Turn, action: &QueuedAction, mv: &Move) -> Result<(), String> {
    if turn.move_start_hp.is_none() {
        return Ok(());
    }
    let me = (action.side, action.slot);
    let flagged = user_self_switches(turn, mv);
    exits_if_crossed(turn, me, flagged);
    if mv.category == "Status" || sheer_forced(turn, me, mv) {
        return Ok(());
    }
    let mut hit: Vec<Slot> = Vec::new();
    for side in 0..2 {
        for slot in 0..2 {
            if turn.move_hit[side][slot] && (side, slot) != me {
                hit.push((side, slot));
            }
        }
    }
    let holding = |turn: &Turn, item: &str| -> Vec<Slot> {
        hit.iter()
            .copied()
            .filter(|t| matches!(turn.mon_at(t.0, t.1), Some(mon) if !mon.fainted && is(mon.item, item)))
            .collect()
    };

    let ejecting = holding(turn, "ejectbutton");
    for target in by_speed(turn, ejecting)? {
        let flag_up = (0..2).any(|side| {
            (0..turn.pos.sides[side].active.len()).any(|slot| {
                matches!(turn.mon_at(side, slot), Some(mon) if mon.has_volatile("pendingselfswitch"))
            })
        });
        if flag_up {
            break;
        }
        let forced = matches!(turn.mon_at(target.0, target.1), Some(mon) if mon.has_volatile("pendingforceswitch"));
        if !can_switch(turn, target.0) || forced {
            continue;
        }
        turn.consume_item(target.0, target.1, "ejectbutton");
        turn.add_volatile(target.0, target.1, "pendingselfswitch", None);
        turn.self_switch_pending = true;
        log_event!(turn, "{} must switch out (ejectbutton)", Name(target.0, target.1));
    }

    let carding = holding(turn, "redcard");
    for target in by_speed(turn, carding)? {
        let user_can_go = matches!(turn.mon_at(me.0, me.1), Some(mon)
            if !mon.fainted && mon.is_active() && !mon.has_volatile("pendingforceswitch"));
        if !user_can_go || !can_switch(turn, me.0) {
            break;
        }
        if matches!(turn.mon_at(target.0, target.1), Some(mon) if mon.has_volatile("pendingforceswitch")) {
            continue;
        }
        // `if (target.useItem(source)) { if (this.runEvent('DragOut', source, target, move))
        // source.forceSwitchFlag = true; }` -- the card goes even when the drag is stopped
        // (IKA-208). The drag itself is `resolve::drag_in`, at the end of the action.
        turn.consume_item(target.0, target.1, "redcard");
        if !drag_out_stopped(turn, me, None) {
            turn.add_volatile(me.0, me.1, "pendingforceswitch", None);
            log_event!(turn, "{} is forced out", Name(me.0, me.1));
        }
    }

    for target in hit {
        exits_if_crossed(turn, target, false);
    }
    Ok(())
}

/// Python's `_residuals_then_emergency_exit`: the residual phase, then Emergency Exit for
/// every Pokemon active at its start that crossed half in it (`residualPokemon`).
pub(crate) fn residuals_then_emergency_exit(reg: &Reg, turn: &mut Turn) -> Result<(), String> {
    let before = active_hp(turn);
    residuals(reg, turn)?;
    for (side, row) in before.iter().enumerate() {
        for (slot, hp) in row.iter().enumerate() {
            if *hp < 0 {
                continue;
            }
            let crossed = matches!(turn.mon_at(side, slot), Some(mon) if crossed_half(mon.hp, mon.maxhp, *hp));
            if crossed {
                emergency_exit(turn, (side, slot), false);
            }
        }
    }
    Ok(())
}

/// `resolve.WEATHER_RECOVERY_MOVES`: the recovery moves whose amount is in `onHit`, so the
/// dump has no `heal` field for them (IKA-187).
const WEATHER_RECOVERY_MOVES: [&str; 3] = ["moonlight", "synthesis", "morningsun"];

/// The factor of Moonlight's `onHit` in 4096ths (`resolve._weather_recovery_modifier`):
/// `0.667` in sun, `0.25` in any other weather, `0.5` without one.
fn weather_recovery_modifier(weather: Option<&str>) -> i64 {
    match weather {
        Some("sunnyday" | "desolateland") => 2732,
        None => 2048,
        Some(_) => 1024,
    }
}

/// `this.heal(this.modify(pokemon.maxhp, factor))` (`resolve._weather_recovery`). `modify`
/// rounds a half down, where Recover's `heal` field is `Math.round` (`round_fraction`).
fn weather_recovery(turn: &mut Turn, me: Slot, mv: &Move) {
    let Some(maxhp) = turn.mon_at(me.0, me.1).map(|m| m.maxhp) else { return };
    let modifier =
        weather_recovery_modifier(turn.pos.field.weather.as_ref().map(|w| w.as_str()));
    turn.heal(me.0, me.1, modify(maxhp, modifier), mv.id.as_str());
}

/// Showdown's `battle.modify(value, modifier / 4096)`: `tr((tr(value * modifier) + 2048 - 1)
/// / 4096)` (sim/battle.ts), a half rounded down (`resolve._modify`).
fn modify(value: i64, modifier: i64) -> i64 {
    (value * modifier + 2047) / 4096
}

/// `clampIntRange(Math.round(amount * num / den), 1)`, as Showdown does for recoil, drain
/// and a `heal` field. JavaScript's `Math.round` rounds halves up (Python's `round` rounds
/// them to even, which the resolver had to work around), and generation 5 on rounds rather
/// than floors: a quarter of 227 heals 57, not 56.
fn round_fraction(amount: i64, ratio: &Value) -> i64 {
    let list = ratio.as_array();
    let (numerator, denominator) = match list {
        Some(values) if values.len() >= 2 => (
            values[0].as_i64().unwrap_or(1),
            values[1].as_i64().unwrap_or(1),
        ),
        _ => (1, 1),
    };
    let scaled = amount * numerator;
    ((scaled * 2 + denominator) / (2 * denominator)).max(1)
}

/// Python's `_sheer_forced`: `move.hasSheerForce && pokemon.hasAbility('sheerforce')`.
///
/// Sheer Force's `onModifyMove` sets `hasSheerForce` only on a move that has secondaries
/// (`if (move.secondaries && !move.hasSheerForceBoost)`), and deletes them; the flag is what
/// `useMoveInner` and `afterMoveSecondaryEvent` read to skip `AfterMoveSecondarySelf` (Life
/// Orb, Shell Bell) and `AfterMoveSecondary` (Scald's thaw).
fn sheer_forced(turn: &Turn, me: Slot, mv: &Move) -> bool {
    mv.has_secondaries
        && !mv.raw.get("hasSheerForceBoost").and_then(Value::as_bool).unwrap_or(false)
        && matches!(turn.mon_at(me.0, me.1), Some(m) if m.ability == "sheerforce")
}

/// Python's `_thaw_on_hit`: the `frz` condition's `onDamagingHit` (a damaging Fire move
/// other than Polar Flare) and `onAfterMoveSecondary` (`thawsTarget`, skipped by Sheer
/// Force), after the move's secondaries (IKA-171).
fn thaw_on_hit(turn: &mut Turn, action: &QueuedAction, mv: &Move, target: Slot) {
    let frozen = matches!(turn.mon_at(target.0, target.1), Some(m) if !m.fainted && is(m.status, "frz"));
    if !frozen {
        return;
    }
    let fire = mv.mtype == "Fire" && mv.category != "Status" && mv.id != "polarflare";
    let thaws = mv.raw.get("thawsTarget").and_then(Value::as_bool).unwrap_or(false)
        && !sheer_forced(turn, (action.side, action.slot), mv);
    if fire || thaws {
        if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
            mon.status = None;
            mon.status_counter = None;
        }
        log_event!(turn, "{} thawed ({})", Name(target.0, target.1), mv.id);
    }
}

// ---------------------------------------------------------------------------
// Substitute (IKA-180)
//
// vendor/pokemon-showdown data/moves.ts, substitute (the champions mod keeps it):
//
//     onTryHit(source) {
//         if (source.volatiles['substitute']) { ...; return this.NOT_FAIL; }
//         if (source.hp <= source.maxhp / 4 || source.maxhp === 1) { ...; return this.NOT_FAIL; }
//     },
//     onHit(target) { this.directDamage(target.maxhp / 4); },
//     condition: {
//         onStart(target) {
//             this.effectState.hp = Math.floor(target.maxhp / 4);
//             if (target.volatiles['partiallytrapped']) { ...; delete target.volatiles['partiallytrapped']; }
//         },
//         onTryPrimaryHit(target, source, move) {
//             if (target === source || move.flags['bypasssub'] || move.infiltrates) return;
//             let damage = this.actions.getDamage(source, target, move);
//             if (!damage && damage !== 0) { ...; return null; }
//             if (damage > target.volatiles['substitute'].hp) damage = target.volatiles['substitute'].hp;
//             target.volatiles['substitute'].hp -= damage;
//             if (target.volatiles['substitute'].hp <= 0) target.removeVolatile('substitute');
//             if (damage) this.actions.applyRecoilDamage(damage, move, source);
//             if (move.drain) this.heal(Math.ceil(damage * move.drain[0] / move.drain[1]), ...);
//             this.singleEvent('AfterSubDamage', ...); this.runEvent('AfterSubDamage', ...);
//             return this.HIT_SUBSTITUTE;
//         },
//     },
//
// and data/mods/champions/scripts.ts `spreadMoveHit`: the check runs for neither a secondary
// nor a self hit, nor a move whose target is `all`, `allyTeam`, `allySide` or `foeSide`; a
// HIT_SUBSTITUTE target is `null` from there on, so its damage, the move's own effects, its
// secondaries, `DamagingHit` (the contact abilities, Rocky Helmet, Cursed Body) and
// `AfterHit` (Knock Off, Stone Axe) skip it, while `selfDrops` and a secondary's `self` still
// reach the user. `hitStepMoveHitLoop` counts it 0 towards `totalDamage` (recoil, Shell
// Bell) and still as a hit (Life Orb, U-turn). A status move gets `null` from `getDamage` and
// does nothing to that target, and the move is not a failure. The resist berries return
// early for a hit the doll takes (`hitSub`), and Intimidate skips a Pokemon behind one.
// A Disguise behind a doll is not the first hit's guard; a multi-hit move that breaks the
// doll would meet the guard on a later hit, which the hit loop does not follow, so that is
// said (`hit_target`).
// ---------------------------------------------------------------------------

const SUBSTITUTE: &str = "substitute";

/// Python's `AFTER_SUB_DAMAGE_UNMODELLED`.
fn after_sub_damage_unmodelled(move_id: &str) -> bool {
    matches!(
        move_id,
        "icespinner" | "steelroller" | "rapidspin" | "mortalspin" | "coreenforcer" | "flameburst"
    )
}

/// Python's `_hits_substitute`: the gate of `onTryPrimaryHit` and the one `spreadMoveHit`
/// puts around it.
fn hits_substitute(turn: &Turn, action: &QueuedAction, mv: &Move, target: Slot) -> bool {
    if target == (action.side, action.slot)
        || matches!(mv.target.as_str(), "all" | "allyTeam" | "allySide" | "foeSide")
    {
        return false;
    }
    if mv.has_flag(F_BYPASSSUB) {
        return false;
    }
    if turn.mon_at(action.side, action.slot).is_some_and(|m| m.ability == "infiltrator") {
        return false;
    }
    matches!(turn.mon_at(target.0, target.1), Some(m) if !m.fainted && m.has_volatile(SUBSTITUTE))
}

/// Python's `_substitute_hp`: `extra.hp`, or the `floor(maxhp / 4)` a doll starts with.
fn substitute_hp(mon: &crate::position::Pokemon) -> i64 {
    match mon.volatile(SUBSTITUTE) {
        None => 0,
        Some(doll) => match doll.extra.get("hp") {
            Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
            _ => mon.maxhp / 4,
        },
    }
}

/// Python's `_intimidate_meets_substitute`.
pub(crate) fn intimidate_meets_substitute(turn: &Turn, target: Slot) -> bool {
    turn.mon_at(target.0, target.1).is_some_and(|m| m.has_volatile(SUBSTITUTE))
}

/// Python's `_use_substitute`: the two refusals (a failure for Stomping Tantrum), then the
/// doll at `floor(maxhp / 4)` and the same HP paid by `directDamage`.
///
/// `directDamage` meets no Endure, Sash or Magic Guard and marks no hurt. A refusal is
/// `NOT_FAIL` from `onTryHit`, which the champions mod's `spreadMoveHit` meets first as the
/// move's own `singleEvent('TryHit')` and turns into `[false]`: a failure Stomping Tantrum
/// reads. Showdown's position after a second Substitute, or one at a quarter of the HP,
/// carries `moveLastTurnFailed` (tests/test_substitute.py).
fn use_substitute(turn: &mut Turn, action: &QueuedAction) {
    let me = (action.side, action.slot);
    let Some(mon) = turn.mon_at_mut(me.0, me.1) else { return };
    if mon.fainted {
        return;
    }
    if mon.has_volatile(SUBSTITUTE) || mon.hp * 4 <= mon.maxhp || mon.maxhp == 1 {
        let why = if mon.has_volatile(SUBSTITUTE) { "a Substitute is already up" } else { "too weak" };
        log_event!(turn, "{} failed ({})", Label(turn.reg, action), why);
        turn.move_failed[me.0][me.1] = true;
        return;
    }
    let cost = (mon.maxhp / 4).max(1);
    mon.volatiles.retain(|v| v.id.as_str() != "partiallytrapped");
    let mut doll = Effect::new(Id::new(SUBSTITUTE));
    set_extra(&mut doll, "hp", Some(json!(cost)));
    mon.volatiles.push(doll);
    mon.hp -= cost;
    log_event!(turn, "{} -{} (substitute)", Name(me.0, me.1), cost);
    turn.check_berry(me.0, me.1);
}

/// Clangorous Soul (IKA-256; the champions mod makes it `accuracy: true`):
///
/// ```text
/// onTry(source) {
///     if (source.hp <= (source.maxhp * 33 / 100) || source.maxhp === 1) return false;
/// },
/// onTryHit(pokemon, target, move) {
///     if (!this.boost(move.boosts!)) return null;
///     delete move.boosts;
/// },
/// onHit(pokemon) {
///     this.directDamage(pokemon.maxhp * 33 / 100);
/// },
/// ```
///
/// The boosts go up in `onTryHit`, before the HP is paid; a user that can raise nothing
/// (all five at +6, or Contrary at -6) fails and pays nothing. `directDamage` floors the
/// cost and skips the `Damage` event (Magic Guard does not stop it) and `hurtThisTurn`;
/// the berry is eaten at the next `Update`.
fn clangorous_soul(turn: &mut Turn, action: &QueuedAction, mv: &Move) {
    let me = (action.side, action.slot);
    let Some(mon) = turn.mon_at(me.0, me.1) else { return };
    if mon.fainted {
        return;
    }
    let (hp, maxhp) = (mon.hp, mon.maxhp);
    // `hp <= maxhp * 33 / 100` in floats, exactly this in integers.
    if hp * 100 <= maxhp * 33 || maxhp == 1 {
        log_event!(turn, "{} failed (too weak)", Label(turn.reg, action));
        turn.move_failed[me.0][me.1] = true;
        return;
    }
    let boosts: Vec<(&str, i64)> = mv
        .raw
        .get("boosts")
        .and_then(Value::as_object)
        .map(|b| b.iter().map(|(stat, v)| (stat.as_str(), v.as_i64().unwrap_or(0))).collect())
        .unwrap_or_default();
    let mut table = boosts;
    if let Some(order) = turn.reg.boost_order.get(mv.id.as_str()) {
        table.sort_by_key(|(stat, _)| order.iter().position(|s| s == stat));
    }
    if !turn.apply_boosts_by(me.0, me.1, &table, false, false, mv.id.as_str()) {
        log_event!(turn, "{} failed (nothing to raise)", Label(turn.reg, action));
        turn.move_failed[me.0][me.1] = true;
        return;
    }
    let cost = (maxhp * 33 / 100).max(1);
    let fainted = {
        let mon = turn.mon_at_mut(me.0, me.1).unwrap();
        mon.hp = (mon.hp - cost).max(0);
        mon.hp <= 0
    };
    // No `hurt_this_turn`: `directDamage` goes through `Pokemon.damage`, and only
    // `spreadDamage` sets `hurtThisTurn` (sim/battle.ts:2138), so a later Assurance stays 60.
    log_event!(turn, "{} -{} (clangoroussoul)", Name(me.0, me.1), cost);
    if fainted {
        turn.faint(me.0, me.1);
    } else {
        turn.check_berry(me.0, me.1);
    }
}

/// Psych Up's `onHit` (IKA-256): the user takes the target's stages as they are -- an
/// assignment, not a `boost`, so Contrary, Simple and Clear Body do nothing -- and the crit
/// volatiles, its own removed first:
///
/// ```text
/// for (i in target.boosts) source.boosts[i] = target.boosts[i];
/// const volatilesToCopy = ['dragoncheer', 'focusenergy', 'gmaxchistrike', 'laserfocus'];
/// for (const volatile of volatilesToCopy) source.removeVolatile(volatile);
/// for (const volatile of volatilesToCopy) {
///     if (target.volatiles[volatile]) { source.addVolatile(volatile); ... hasDragonType ... }
/// }
/// ```
///
/// The move has no `protect` flag and `bypasssub`, so Protect and a Substitute do not stop it.
fn psych_up(turn: &mut Turn, me: Slot, targets: &[Slot]) {
    const CRIT_VOLATILES: [&str; 4] = ["dragoncheer", "focusenergy", "gmaxchistrike", "laserfocus"];
    for target in targets {
        let Some(foe) = turn.mon_at(target.0, target.1) else { continue };
        if foe.fainted {
            continue;
        }
        let boosts = foe.boosts;
        let copied: Vec<Effect> = foe
            .volatiles
            .iter()
            .filter(|v| CRIT_VOLATILES.contains(&v.id.as_str()))
            .cloned()
            .collect();
        let Some(mon) = turn.mon_at_mut(me.0, me.1) else { continue };
        if mon.fainted {
            continue;
        }
        mon.boosts = boosts;
        mon.volatiles.retain(|v| !CRIT_VOLATILES.contains(&v.id.as_str()));
        mon.volatiles.extend(copied);
        log_event!(turn, "{} copied {}'s stat changes (psychup)", Name(me.0, me.1), Name(target.0, target.1));
    }
}

/// Python's `_behind_substitute`: a resist berry halves nothing the doll takes.
fn behind_substitute(defender: Battler) -> Battler {
    match defender.item {
        Some(item) if resist_berry(item.as_str()).is_some() => Battler { item: None, ..defender },
        _ => defender,
    }
}

/// Python's `_doll_damage`: the first hit's rolls against the doll, when a resist berry
/// makes them differ from the target's own.
#[allow(clippy::too_many_arguments)]
fn doll_damage(
    reg: &Reg,
    attacker: &Battler,
    defender: &Battler,
    move_id: &str,
    field: &crate::battler::FieldState,
    target: Slot,
    spread: bool,
    crit: bool,
    move_ctx: &MoveContext,
) -> Option<[i64; crate::battler::N_ROLLS]> {
    let behind = behind_substitute(*defender);
    if behind.item == defender.item {
        return None;
    }
    let result = calculate(
        reg, attacker, &behind, move_id, field, target.0, spread, crit, Some(move_ctx), None, false,
    );
    Some(result.rolls)
}

/// Python's `_hit_substitute`: one hit the doll takes -- its HP, the recoil and the
/// (rounded-up) drain from what it took, Stone Axe's and Ceaseless Edge's
/// `onAfterSubDamage`, a secondary's `self`, and nothing that reaches the target.
fn hit_substitute(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
    amount: i64,
    budget: &Budget,
) -> Result<(), String> {
    let Some(mon) = turn.mon_at_mut(target.0, target.1) else { return Ok(()) };
    let held = substitute_hp(mon);
    let dealt = amount.max(0).min(held);
    let left = held - dealt;
    if left <= 0 {
        mon.volatiles.retain(|v| v.id.as_str() != SUBSTITUTE);
        log_event!(turn, "{}'s Substitute broke ({})", Name(target.0, target.1), mv.id);
    } else {
        if let Some(doll) = mon.volatile_mut(SUBSTITUTE) {
            set_extra(doll, "hp", Some(json!(left)));
        }
        log_event!(turn, "{}'s Substitute -{} ({})", Name(target.0, target.1), dealt, mv.id);
    }
    turn.move_connected = true;

    let me = (action.side, action.slot);
    let rockhead = matches!(turn.mon_at(me.0, me.1), Some(m) if m.ability == "rockhead");
    if let Some(recoil) = mv.recoil.as_ref() {
        if dealt > 0 && !rockhead && turn.mon_at(me.0, me.1).is_some() {
            let amount = round_fraction(dealt, recoil);
            turn.deal_damage(me.0, me.1, amount, false, "recoil")?;
        }
    }
    crate::level_struggle::struggle_recoil(turn, me, mv, dealt)?;
    if let Some(drain) = mv.drain.as_ref().and_then(Value::as_array) {
        if dealt > 0 && drain.len() >= 2 {
            let numerator = drain[0].as_i64().unwrap_or(1);
            let denominator = drain[1].as_i64().unwrap_or(1);
            turn.heal(me.0, me.1, (dealt * numerator + denominator - 1) / denominator, "drain");
        }
    }

    if matches!(mv.id.as_str(), "stoneaxe" | "ceaselessedge") {
        lay_hazard_after_hit(turn, action, mv, true);
    } else if after_sub_damage_unmodelled(mv.id.as_str()) {
        turn.report(format!("substitute: {} onAfterSubDamage", mv.id));
    }
    crate::airborne::pop_air_balloon(turn, target);

    if matches!(turn.mon_at(me.0, me.1), Some(m) if m.ability == "sheerforce") {
        return Ok(());
    }
    for secondary in &mv.secondaries {
        let Some(own) = secondary.get("self") else { continue };
        if own.as_object().is_none_or(|o| o.is_empty()) {
            continue;
        }
        let chance =
            secondary.get("chance").and_then(Value::as_f64).unwrap_or(100.0) / 100.0;
        let kept = json!({ "self": own });
        if chance >= 1.0 {
            apply_secondary(turn, action, &kept, target)?;
        } else if budget.enumerate_secondary {
            turn.pending_secondaries.push((chance, kept, target));
        } else if !budget.pinned_policy {
            turn.report(format!(
                "secondary {}%: {} (not branched)",
                (chance * 100.0) as i64,
                mv.id
            ));
        }
    }
    Ok(())
}

/// Python's `_apply_status_move_past_substitutes`: a status move on the targets no doll
/// stopped, and no "did nothing" judgement while any target was a doll's.
fn apply_status_move_past_substitutes(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    reachable: &[Slot],
    subbed: &[Slot],
) -> Result<(), String> {
    if subbed.is_empty() {
        return apply_status_move_and_judge(reg, turn, action, mv, reachable);
    }
    for target in subbed {
        log_event!(turn, "{}'s Substitute blocked {}", Name(target.0, target.1), mv.id);
    }
    let applied: Vec<Slot> = reachable.iter().copied().filter(|t| !subbed.contains(t)).collect();
    if applied.is_empty() {
        return Ok(());
    }
    apply_status_move(reg, turn, action, mv, &applied)
}

/// Effects that fire once per move use, not once per target.
///
/// Showdown computes recoil from `damageDealt` -- the total across every target -- and Life
/// Orb's recoil and a move's own `self` effect are single `onAfterMoveSecondary` handlers;
/// running them per target multiplies them by the number of Pokemon hit. Shell Bell heals an
/// eighth of the move's *total* damage, once, truncated -- so a move dealing under eight
/// heals nothing. `selfBoost` is a separate field from `self`, applied once after the move
/// succeeds (Clanging Scales, Clangorous Soul, Scale Shot). The self-switch is `else if
/// (move.selfSwitch && source.hp && !source.volatiles['commanded'])`, reached only when the
/// move `didAnything`: a U-turn into a Ghost type, or one that missed, leaves its user in
/// place. White Herb is `onAnyAfterMove`: every holder on the field is checked.
fn after_move(turn: &mut Turn, action: &QueuedAction, mv: &Move) -> Result<(), String> {
    let me = (action.side, action.slot);
    let total = turn.move_damage_total;

    // Drain is in `after_hit`, per target (IKA-161).
    let rockhead = matches!(turn.mon_at(me.0, me.1), Some(m) if m.ability == "rockhead");
    if let Some(recoil) = mv.recoil.as_ref() {
        if total > 0 && !rockhead {
            let amount = round_fraction(total, recoil);
            turn.deal_damage(me.0, me.1, amount, false, "recoil")?;
        }
    }
    crate::level_struggle::struggle_recoil(turn, me, mv, total)?;
    after_move_secondary_switches(turn, action, mv)?;
    let (item, maxhp) = match turn.mon_at(me.0, me.1) {
        None => (None, 0),
        Some(mon) => (mon.item, mon.maxhp),
    };
    // Sheer Force skips `AfterMoveSecondarySelf` and deletes `move.self` (IKA-161), and
    // Life Orb runs for any hit that reached a target, a 0-damage one included -- Python's
    // `move_connected`, not the total (IKA-171).
    let sheer = sheer_forced(turn, me, mv);
    if is(item, "lifeorb") && turn.move_connected && !sheer {
        turn.deal_damage(me.0, me.1, (maxhp / 10).max(1), false, "lifeorb")?;
    }
    if is(item, "shellbell") && total >= 8 && !sheer {
        turn.heal(me.0, me.1, total / 8, "shellbell");
    }
    if !sheer && mv.category != "Status" && turn.move_start_hp.is_some() {
        let flagged = user_self_switches(turn, mv);
        exits_if_crossed(turn, me, flagged);
    }

    let connected = turn.move_connected;
    let alive = matches!(turn.mon_at(me.0, me.1), Some(m) if !m.fainted);
    if alive && connected {
        if let Some(self_effect) = mv.self_effect.as_ref().filter(|_| !sheer) {
            if let Some(boosts) = self_effect.get("boosts").and_then(Value::as_object) {
                let table: Vec<(&str, i64)> = boosts
                    .iter()
                    .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                    .collect();
                turn.apply_boosts(me.0, me.1, &table, false, mv.id.as_str());
            }
            if let Some(vid) = self_effect.get("volatileStatus").and_then(Value::as_str) {
                let vid = vid.to_string();
                if !crate::resolve::volatile_is_handled(&vid) {
                    return Err(format!("self volatile: {vid}"));
                }
                turn.add_volatile(me.0, me.1, &vid, None);
            }
        }
        // Double Shock's and Burn Up's `self.onHit`: the spent type becomes `SPENT_TYPE`.
        if let Some(spent) = spent_type(mv.id.as_str()) {
            let current = match turn.mon_at(me.0, me.1) {
                Some(mon) => turn.types_of(mon),
                None => Types::default(),
            };
            let replaced: Vec<Id> = current
                .as_slice()
                .iter()
                .map(|t| if t.as_str() == spent { Id::new(SPENT_TYPE) } else { *t })
                .collect();
            if let Some(mon) = turn.mon_at_mut(me.0, me.1) {
                mon.types = Types::from_slice(&replaced);
            }
            if turn.log.is_some() {
                let joined = replaced.iter().map(|t| t.as_str()).collect::<Vec<_>>().join("/");
                log_event!(turn, "{} is {} ({})", Name(me.0, me.1), joined, mv.id);
            }
        }
        if let Some(boosts) = mv.self_boost_boosts.as_ref() {
            let table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            turn.apply_boosts(me.0, me.1, &table, false, mv.id.as_str());
        }
    }

    if mv.self_switch && turn.move_connected {
        mark_self_switch(turn, action);
    }

    check_white_herb(turn);
    turn.move_damage_total = 0;
    turn.move_connected = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Status moves
// ---------------------------------------------------------------------------

/// Python's `_helping_hand_fails` (IKA-184): Helping Hand's `onTryHit`,
/// `if (!target.newlySwitched && !this.queue.willMove(target)) return false`. A partner that
/// has already moved this turn is not helped, unless it came in this turn.
fn helping_hand_fails(turn: &Turn, targets: &[Slot]) -> bool {
    for target in targets {
        let Some(mon) = turn.mon_at(target.0, target.1) else { continue };
        if mon.newly_switched || !turn.acted[target.0][target.1] {
            return false;
        }
    }
    true
}

/// A status move, which can miss (Hypnosis 60, Will-O-Wisp 85, Thunder Wave 90).
///
/// Protect also blocks status moves that carry the `protect` flag, and the accuracy check
/// is per target. Psychic Terrain comes before Protect, as on the damaging path (IKA-156).
/// Wide Guard and Quick Guard carry Protect's `onTry() { return !!queue.willAct(); }`
/// gate: a guard that resolves after everything else has moved has nothing to guard.
///
/// Protect's `onTryHit` returns `NOT_FAIL`, which `hitStepTryHitEvent` keeps, so a move
/// every target of which Protected ends with `moveThisTurnResult` null -- not the `false`
/// Stomping Tantrum reads. Psychic Terrain's `null` becomes `false` there (`hitResults[i] ||
/// false`), and an immunity is `false` from the start, so either makes it a failure
/// (IKA-171).
fn do_status_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let Some(attacker) = turn.battler_at(action.side, action.slot)? else {
        return Ok(vec![(1.0, turn)]);
    };

    let mut reachable: Vec<Slot> = Vec::new();
    let mut failed_any = false;
    for target in targets {
        if stopped_by_psychic_terrain(&turn, action, mv, *target) {
            log_event!(turn, "{} protected by psychicterrain", Name(target.0, target.1));
            failed_any = true;
            continue;
        }
        if *target != (action.side, action.slot) {
            if let Some(blocked) = blocked_by_protect(&turn, action, mv, *target) {
                log_event!(turn, "{} blocked by {}", Label(reg, action), blocked);
                continue;
            }
        }
        if let Some(immunity) = immune_to_move(reg, &turn, action, mv, *target) {
            log_event!(turn, "{} immune ({})", Name(target.0, target.1), immunity);
            failed_any = true;
            continue;
        }
        reachable.push(*target);
    }
    // Rolled for like the rest, then met by the doll (IKA-180).
    let subbed: Vec<Slot> =
        reachable.iter().copied().filter(|t| hits_substitute(&turn, action, mv, *t)).collect();

    // Python's `_do_status_move`: a move every target of which Protected is `null`, not a
    // failure; the terrain and an immunity are `false` (IKA-171).
    if reachable.is_empty() && !targets.is_empty() {
        if failed_any {
            turn.move_failed[action.side][action.slot] = true;
        }
        return Ok(vec![(1.0, turn)]);
    }

    let mut accuracy = 1.0f64;
    for target in &reachable {
        if *target == (action.side, action.slot) {
            continue;
        }
        if let Some(defender) = turn.battler_at(target.0, target.1)? {
            accuracy = accuracy.min(accuracy_of(&turn, mv, &attacker, &defender));
        }
    }

    if mv.stalling_move {
        return do_protect(turn, action, mv, &budget);
    }

    if STALL_BUMPING_MOVES.contains(&mv.id.as_str()) && turn.actions_remaining == 0 {
        log_event!(turn, "{} failed (nothing left to act)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    if mv.id == "helpinghand" && helping_hand_fails(&turn, &reachable) {
        log_event!(turn, "{} failed (partner already moved)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    if mv.id == "substitute" {
        use_substitute(&mut turn, action);
        return Ok(vec![(1.0, turn)]);
    }

    if mv.id == "clangoroussoul" {
        clangorous_soul(&mut turn, action, mv);
        return Ok(vec![(1.0, turn)]);
    }

    if !budget.enumerate_accuracy || accuracy >= 1.0 || accuracy <= 0.0 {
        if accuracy <= 0.0 {
            log_event!(turn, "{} missed", Label(reg, action));
            turn.move_failed[action.side][action.slot] = true;
            return Ok(vec![(1.0, turn)]);
        }
        apply_status_move_past_substitutes(turn.reg, &mut turn, action, mv, &reachable, &subbed)?;
        return Ok(vec![(1.0, turn)]);
    }

    let mut hit_state = turn.clone();
    apply_status_move_past_substitutes(reg, &mut hit_state, action, mv, &reachable, &subbed)?;
    log_event!(turn, "{} missed", Label(reg, action));
    turn.move_failed[action.side][action.slot] = true;
    Ok(vec![(accuracy, hit_state), (1.0 - accuracy, turn)])
}

/// Python's `JUDGED_STATUS_EFFECTS` and `UNJUDGED_STATUS_EFFECTS`.
const JUDGED_STATUS_EFFECTS: [&str; 5] = ["boosts", "heal", "status", "weather", "terrain"];
const UNJUDGED_STATUS_EFFECTS: [&str; 9] = [
    "sideCondition",
    "volatileStatus",
    "pseudoWeather",
    "self",
    "selfSwitch",
    "forceSwitch",
    "slotCondition",
    "selfBoost",
    "selfdestruct",
];

/// Python's `_apply_status_move_and_judge`: `move_failed` for a status move made only of
/// declarative effects when none of them did anything to any target (IKA-171).
///
/// `runMoveEffects` combines each effect's result per target: a heal on a target at full HP
/// is `false` outright, a status that did not take is `false` (`if (!hitResult &&
/// move.status)`), a boost that moved nothing is `null`, and `null` becomes `false` at the
/// end; a weather or terrain already in place returns `false` from `setWeather` /
/// `setTerrain`. The move's result is `false` when every target's is, which is a failure for
/// Stomping Tantrum -- a Recover at full HP (`-fail heal`), a Toxic into a Poison type, a
/// second Sunny Day. Only the moves made entirely of those declarative effects are judged; a
/// move with a side condition, a volatile or custom code keeps the old reading.
fn apply_status_move_and_judge(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    reachable: &[Slot],
) -> Result<(), String> {
    let truthy = |key: &str| match mv.raw.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(s)) => !s.is_empty(),
    };
    let judged = JUDGED_STATUS_EFFECTS.iter().any(|key| truthy(key))
        && !truthy("hasCustomCode")
        && !UNJUDGED_STATUS_EFFECTS.iter().any(|key| truthy(key));
    if !judged {
        return apply_status_move(reg, turn, action, mv, reachable);
    }
    let weather_before = turn.pos.field.weather;
    let terrain_before = turn.pos.field.terrain;
    let mut before = Vec::new();
    for target in reachable {
        if let Some(mon) = turn.mon_at(target.0, target.1) {
            before.push((*target, mon.hp, mon.maxhp, mon.status, mon.boosts));
        }
    }
    apply_status_move(reg, turn, action, mv, reachable)?;

    let mut did = false;
    if truthy("weather") && turn.pos.field.weather != weather_before {
        did = true;
    }
    if truthy("terrain") && turn.pos.field.terrain != terrain_before {
        did = true;
    }
    for (target, hp, maxhp, status, boosts) in before {
        let Some(mon) = turn.mon_at(target.0, target.1) else { continue };
        if truthy("heal") && hp >= maxhp {
            continue;
        }
        if truthy("heal") {
            did = true;
        }
        if truthy("status") && mon.status != status {
            did = true;
        }
        if truthy("boosts") && mon.boosts != boosts {
            did = true;
        }
    }
    if !did {
        log_event!(turn, "{} failed (did nothing)", Label(reg, action));
        turn.move_failed[action.side][action.slot] = true;
    }
    Ok(())
}

/// Why this target ignores this move outright, or None if it does not.
///
/// Two immunities Showdown checks per target before anything else happens, both read from
/// the regulation's `effectImmunities` rather than written down here.
///
/// **Powder.** `hitStepTryImmunity`: `gen >= 6 && move.flags['powder'] && target !== pokemon
/// && !this.dex.getImmunity('powder', target)`, so a Grass type ignores Sleep Powder, Spore
/// and Stun Spore. Overcoat and Safety Goggles block them by a separate `onTryHit`, included
/// for completeness.
///
/// **Prankster.** `gen >= 7 && move.pranksterBoosted && pokemon.hasAbility('prankster') &&
/// !targets[i].isAlly(pokemon) && !this.dex.getImmunity('prankster', target)`.
/// `pranksterBoosted` is set by the ability's own `onModifyPriority`, so it means exactly "a
/// Status move used by a Prankster holder". Allies are exempt, so Prankster Tailwind and
/// screens are unaffected.
fn immune_to_move(
    reg: &Reg,
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
) -> Option<String> {
    let defender = turn.mon_at(target.0, target.1)?;
    if defender.fainted {
        return None;
    }
    if good_as_gold_blocks(turn, action, mv, target) {
        return Some("goodasgold".into());
    }
    // Trick and Switcheroo: `onTryImmunity(target) { return !target.hasAbility('stickyhold'); }`
    // (data/moves.ts, IKA-208).
    if matches!(mv.id.as_str(), "trick" | "switcheroo") && defender.ability == "stickyhold" {
        return Some("stickyhold".into());
    }
    let self_targeted = target == (action.side, action.slot);
    let types = turn.types_of(defender);

    if mv.has_flag(F_POWDER) && !self_targeted {
        let immune = reg
            .effect_immunities
            .get("powder")
            .map(|set| types.as_slice().iter().any(|t| set.contains(t.as_str())))
            .unwrap_or(false);
        if immune {
            return Some("powder vs Grass".into());
        }
        if defender.ability == "overcoat" {
            return Some("overcoat".into());
        }
        if is(defender.item, "safetygoggles") {
            return Some("safetygoggles".into());
        }
    }

    if mv.category == "Status" && target.0 != action.side {
        let attacker_is_prankster = turn
            .mon_at(action.side, action.slot)
            .map(|m| m.ability == "prankster")
            .unwrap_or(false);
        if attacker_is_prankster {
            let immune = reg
                .effect_immunities
                .get("prankster")
                .map(|set| types.as_slice().iter().any(|t| set.contains(t.as_str())))
                .unwrap_or(false);
            if immune {
                return Some("prankster vs Dark".into());
            }
        }
    }
    None
}

/// Disables the target's last move, as Showdown's `disable` condition does.
///
/// Fails -- with no volatile at all -- when the target has not moved yet, or when the move
/// it last used has no PP left; both are `return false` in `onStart`, which means no
/// volatile rather than a volatile that disables nothing.
///
/// The duration is decremented immediately when the target still owes an action this turn
/// or when it is the Pokemon whose own move triggered this, which covers Cursed Body and
/// the ordinary Disable; five turns is left for disabling something that has already acted.
fn apply_disable(turn: &mut Turn, side: usize, slot: usize, mv: Option<&Move>) -> bool {
    let (last_move, already) = match turn.mon_at(side, slot) {
        None => return false,
        Some(mon) if mon.fainted => return false,
        Some(mon) => (mon.last_move, mon.has_volatile("disable")),
    };
    let Some(last_move) = last_move else { return false };
    if already {
        return false;
    }
    let has_pp = match turn.mon_at(side, slot) {
        None => false,
        Some(mon) => mon.moves.get(last_move).map(|m| m.pp > 0).unwrap_or(false),
    };
    if !has_pp {
        return false;
    }
    let mut duration = match mv {
        Some(mv) => effect_duration(turn, mv, "disable", side, slot).unwrap_or(5),
        None => 5,
    };
    if !turn.acted[side][slot] || turn.current_actor == Some((side, slot)) {
        duration -= 1;
    }
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        let mut effect = Effect::new(Id::new("disable"));
        effect.duration = Some(duration);
        effect.move_id = Some(last_move);
        mon.volatiles.push(effect);
        // The flag lives on the move slot, which is what the legality half reads.
        if let Some(move_slot) = mon.moves.get_mut(last_move) {
            move_slot.disabled = true;
        }
    }
    log_event!(turn, "{} cannot use {} (disable)", Name(side, slot), last_move);
    true
}

/// Locks the target into the move it last used. Fails -- with no volatile at all --
/// when the target has not moved, when that move cannot be encored (`failencore`: Struggle,
/// Sleep Talk, Copycat, Transform and Encore itself), or when it is out of PP, all three of
/// which are `return false` in Showdown's `onStart`.
///
/// Duration 3, or 4 when the target has already moved this turn: `if
/// (!queue.willMove(target)) duration++`. The volatile is decremented at the end of this
/// turn either way, so the increment is what gives a Pokemon that has already acted its
/// full three turns.
fn apply_encore(turn: &mut Turn, side: usize, slot: usize, mv: &Move) -> bool {
    let (last_move, already) = match turn.mon_at(side, slot) {
        None => return false,
        Some(mon) if mon.fainted => return false,
        Some(mon) => (mon.last_move, mon.has_volatile("encore")),
    };
    let Some(last_move) = last_move else { return false };
    if already {
        return false;
    }
    match turn.reg.moves.get(last_move.as_str()) {
        None => return false,
        Some(last) if last.has_flag(F_FAILENCORE) => return false,
        Some(_) => {}
    }
    let has_pp = match turn.mon_at(side, slot) {
        None => false,
        Some(mon) => mon.moves.get(last_move).map(|m| m.pp > 0).unwrap_or(false),
    };
    if !has_pp {
        return false;
    }
    let mut duration = effect_duration(turn, mv, "encore", side, slot).unwrap_or(3);
    if turn.acted[side][slot] {
        duration += 1;
    }
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        let mut effect = Effect::new(Id::new("encore"));
        effect.duration = Some(duration);
        effect.move_id = Some(last_move);
        mon.volatiles.push(effect);
    }
    log_event!(turn, "{} is locked into {} (encore)", Name(side, slot), last_move);
    true
}

/// `randomChance(1, counter)`: certain on the first use, a third on the second.
fn stall_success_chance(counter: i64) -> f64 {
    1.0 / counter.max(1) as f64
}

/// Raises the Protect counter, as Showdown's `addVolatile('stall')` does. The volatile lasts
/// two turns, so a turn spent on anything else lets it lapse and the next Protect is certain
/// again. Wide Guard and Quick Guard come through here without ever having consulted the
/// counter: Showdown gives them `source.addVolatile('stall')` in `onHitSide` but no
/// `stallingMove` flag, so they always succeed themselves and still make the next Protect a
/// 1-in-3. Which moves *consult* the counter comes from the dumped `stallingMove` flag.
fn bump_stall(turn: &mut Turn, side: usize, slot: usize) {
    let Some(mon) = turn.mon_at_mut(side, slot) else { return };
    match mon.volatile_mut("stall") {
        None => {
            let mut effect = Effect::new(Id::new("stall"));
            effect.duration = Some(2);
            effect.counter = Some(3);
            mon.volatiles.push(effect);
        }
        Some(existing) => {
            existing.counter = Some((existing.counter.unwrap_or(1) * 3).min(729));
            existing.duration = Some(2);
        }
    }
}

/// A Protect-family move, which fails more often the more it is repeated (the `stall`
/// counter, `bump_stall`).
///
/// Showdown gates Protect on `!!this.queue.willAct()`: a Protect that resolves last in the
/// turn fails outright. In doubles that is the common case for a slow side, so letting it
/// succeed is not a rounding error. The protection lasts one turn, stated here rather than
/// by membership of `PROTECT_VOLATILES`: Endure is a stalling move and is *not* in that
/// table (its volatile caps damage rather than blocking the hit, in `deal_damage`), so it
/// once got no duration, missed the unconditional removal, and lasted the rest of the
/// battle -- the Pokemon surviving every lethal hit at 1 HP forever. Under a budget that
/// does not enumerate status checks the reading is failure, which is also what the pinned
/// policy answers `randomChance(1, counter)` with for any counter above 1.
fn do_protect<'a>(
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let me = (action.side, action.slot);
    let Some(mon) = turn.mon_at(me.0, me.1) else { return Ok(vec![(1.0, turn)]) };
    if turn.actions_remaining == 0 {
        log_event!(turn, "{} failed (nothing left to act)", Label(turn.reg, action));
        turn.move_failed[me.0][me.1] = true;
        return Ok(vec![(1.0, turn)]);
    }
    let counter = mon.volatile("stall").and_then(|s| s.counter).unwrap_or(1);
    let chance = stall_success_chance(counter);

    let succeed = |state: &mut Turn, move_id: &str| {
        state.add_volatile(me.0, me.1, move_id, Some(1));
        bump_stall(state, me.0, me.1);
        log_event!(state, "{} protected (1 in {})", Label(state.reg, action), counter);
    };
    let fail = |state: &mut Turn| {
        log_event!(state, "{} failed (1 in {})", Label(state.reg, action), counter);
        state.move_failed[me.0][me.1] = true;
        if let Some(mon) = state.mon_at_mut(me.0, me.1) {
            mon.volatiles.retain(|v| v.id.as_str() != "stall");
        }
    };

    if chance >= 1.0 {
        succeed(&mut turn, &mv.id);
        return Ok(vec![(1.0, turn)]);
    }
    if !budget.enumerate_status_checks {
        fail(&mut turn);
        return Ok(vec![(1.0, turn)]);
    }
    let mut hit_state = turn.clone();
    succeed(&mut hit_state, &mv.id);
    fail(&mut turn);
    Ok(vec![(chance, hit_state), (1.0 - chance, turn)])
}

/// Whether Showdown really *rolls* this duration. A `durationCallback` alone does not
/// mean that -- most of them apply an item or ability extension -- so the dumper records
/// whether calling it consumed randomness, and that is what this reads. Thirteen of the
/// fourteen effects that have one use it for an extension, and only `partiallytrapped`
/// calls `this.random`; reading the mere presence of a function once had Tailwind and every
/// screen reported as "Showdown rolls it".
fn duration_is_rolled(mv: &Move, effect_id: &str) -> bool {
    mv.raw
        .get("durations")
        .and_then(Value::as_object)
        .and_then(|durations| durations.get(effect_id))
        .and_then(Value::as_object)
        .and_then(|entry| entry.get("rolled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// How long this effect lasts for *this* user, extensions included, from the regulation
/// dump.
///
/// Keyed by the effect actually being added, because one move can name a volatile, a side
/// condition and a slot condition and they need not share a duration. `durationCallback
/// (target, source)` reads the source's item and ability, and the dump carries what it
/// returns for each -- so Light Clay's screens, Grip Claw's binds, Persistent's rooms and
/// Terrain Extender's terrains follow from the position rather than from a number written
/// here. When the callback rolls (`partiallytrapped` declares 5 and returns 5 or 6) the
/// fixed value is the pinned reading and the caller reports the approximation.
///
/// This used to be a hand-written list, and the list was the root cause of a whole class of
/// bug: anything missing from it became a *permanent* effect. Four were -- including a bind
/// the target could never escape and an Endure that survived every lethal hit for the rest
/// of the battle -- and it gave Tailwind 5 turns where Showdown gives 4.
fn effect_duration(
    turn: &Turn,
    mv: &Move,
    effect_id: &str,
    side: usize,
    slot: usize,
) -> Option<i64> {
    let durations = mv.raw.get("durations")?.as_object()?;
    let entry = durations.get(effect_id)?.as_object()?;
    if let Some(mon) = turn.mon_at(side, slot) {
        if let (Some(by_item), Some(item)) = (entry.get("byItem").and_then(Value::as_object), mon.item)
        {
            if let Some(value) = by_item.get(item.as_str()).and_then(Value::as_i64) {
                return Some(value);
            }
        }
        if let Some(by_ability) = entry.get("byAbility").and_then(Value::as_object) {
            if let Some(value) = by_ability.get(mon.ability.as_str()).and_then(Value::as_i64) {
                return Some(value);
            }
        }
    }
    entry
        .get("base")
        .and_then(Value::as_i64)
        .or_else(|| entry.get("duration").and_then(Value::as_i64))
}

/// Parting Shot's `this.boost({atk: -1, spa: -1}, target, source)` on one target; true when
/// a drop landed on it (IKA-238).
///
/// Mirror Armor's `onTryBoost` (data/abilities.ts) takes each drop off its holder and sends
/// it back at the user, one stat at a time, unless the holder is already at -6 there:
///
/// ```text
/// if (!source || target === source || !boost || effect.name === 'Mirror Armor') return;
/// for (b in boost) {
///     if (boost[b]! < 0) {
///         if (target.boosts[b] === -6) continue;
///         const negativeBoost = {}; negativeBoost[b] = boost[b]; delete boost[b];
///         if (source.hp) {
///             this.add('-ability', target, 'Mirror Armor');
///             this.boost(negativeBoost, source, target, null, true);
///         }
///     }
/// }
/// ```
///
/// The ability is `breakable`, so a Mold Breaker user's drops land. The holder's own drops
/// that are left (a stat at -6) change nothing. Nothing lands, so the caller's `landed` is
/// Mirror Armor's own exception to `delete move.selfSwitch`.
fn parting_shot_drops(turn: &mut Turn, target: Slot, me: Slot) -> bool {
    const DROPS: [(&str, i64); 2] = [("atk", -1), ("spa", -1)];
    let bounces = matches!(turn.mon_at(target.0, target.1), Some(mon)
        if mon.ability == "mirrorarmor" && !mon.fainted && !ability_broken_by(turn, mon, Some(me)));
    if !bounces || target == me {
        return turn.apply_boosts(target.0, target.1, &DROPS, true, "partingshot");
    }
    for (stat, delta) in DROPS {
        let at_floor = turn
            .mon_at(target.0, target.1)
            .zip(crate::position::boost_index(stat))
            .is_some_and(|(mon, index)| mon.boosts[index] == -6);
        let user_up = turn.mon_at(me.0, me.1).is_some_and(|m| m.hp > 0);
        if at_floor || !user_up {
            continue;
        }
        log_event!(turn, "{} mirrorarmor sent the {stat} drop back", Name(target.0, target.1));
        turn.apply_boosts(me.0, me.1, &[(stat, delta)], true, "mirrorarmor");
    }
    false
}

fn apply_status_move(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
) -> Result<(), String> {
    let me = (action.side, action.slot);
    let mut suppress_self_switch = false;

    // Healing Wish's `onTryHit`: `if (!this.canSwitch(source.side)) { ...; return
    // this.NOT_FAIL; }` -- nothing happens and the user stays (IKA-208).
    if mv.id == "healingwish" && !can_switch(turn, me.0) {
        return Ok(());
    }

    if let Some(condition) = mv.side_condition.as_deref() {
        let condition = condition.to_string();
        let duration = effect_duration(turn, mv, &condition, action.side, action.slot);
        if duration_is_rolled(mv, &condition) {
            turn.report(format!(
                "{condition} duration (Showdown rolls it; pinned to the low end)"
            ));
        }
        // The target's side (`moveHit`: `target.side.addSideCondition`), which
        // `getMoveTargets` makes a foe's for `foeSide` -- the hazards. IKA-165: this was
        // always the user's. The duration above stays the user's (Light Clay).
        let condition_side = if mv.target == "foeSide" { 1 - action.side } else { action.side };
        if !turn.add_side_condition(condition_side, &condition, duration) {
            // Nothing else in these moves does anything, so `didSomething` is false and
            // the move fails (IKA-173): a second Stealth Rock or Tailwind, a fourth Spikes.
            log_event!(turn, "{} failed ({} already up)", Label(reg, action), condition);
            turn.move_failed[me.0][me.1] = true;
        }
    }
    if let Some(weather) = mv.weather.as_deref() {
        let weather = weather.to_lowercase().replace(' ', "");
        turn.pos.field.weather = Some(Id::new(&weather));
        turn.pos.field.weather_duration =
            Some(effect_duration(turn, mv, &weather, action.side, action.slot).unwrap_or(5));
        log_event!(turn, "weather -> {}", weather);
    }
    if let Some(terrain) = mv.terrain.as_deref() {
        let terrain = terrain.to_lowercase().replace(' ', "");
        // The same terrain again changes nothing (IKA-201).
        let duration = effect_duration(turn, mv, &terrain, action.side, action.slot).unwrap_or(5);
        crate::terrain::set_terrain(turn, &terrain, duration);
    }
    if let Some(pseudo) = mv.pseudo_weather.as_deref() {
        let pid = pseudo.to_lowercase().replace(' ', "");
        let already = turn.pos.field.has_pseudo_weather(&pid);
        // Only the room moves switch themselves off when used again (`onFieldRestart`);
        // Gravity and the rest just fail. Persistent makes Trick Room and Gravity 7.
        let toggling = matches!(pid.as_str(), "trickroom" | "magicroom" | "wonderroom");
        if already && toggling {
            turn.pos.field.pseudo_weather.retain(|p| p.id.as_str() != pid);
            log_event!(turn, "{} ended", pid);
        } else if !already {
            let duration =
                effect_duration(turn, mv, &pid, action.side, action.slot).unwrap_or(5);
            let mut effect = Effect::new(Id::new(&pid));
            effect.duration = Some(duration);
            turn.pos.field.pseudo_weather.push(effect);
            log_event!(turn, "{} started", pid);
        } else {
            log_event!(turn, "{} failed (already active)", pid);
        }
    }

    if let Some(self_effect) = mv.self_effect.as_ref() {
        if let Some(boosts) = self_effect.get("boosts").and_then(Value::as_object) {
            let table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            turn.apply_boosts(me.0, me.1, &table, false, mv.id.as_str());
        }
        if let Some(vid) = self_effect.get("volatileStatus").and_then(Value::as_str) {
            let vid = vid.to_string();
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("self volatile: {vid}"));
            }
            turn.add_volatile(me.0, me.1, &vid, None);
        }
        if let Some(condition) = self_effect.get("sideCondition").and_then(Value::as_str) {
            let condition = condition.to_string();
            turn.add_side_condition(action.side, &condition, None);
        }
    }

    for target in targets {
        let own_side = target.0 == action.side;
        if let Some(boosts) = mv.raw.get("boosts").and_then(Value::as_object) {
            let mut table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            // In the dump's order, as Showdown's `boost()` walks it (IKA-215).
            if let Some(order) = reg.boost_order.get(mv.id.as_str()) {
                table.sort_by_key(|(stat, _)| order.iter().position(|s| s == stat));
            }
            turn.apply_boosts_by(
                target.0,
                target.1,
                &table,
                !own_side,
                (target.0, target.1) != me,
                mv.id.as_str(),
            );
        }
        if let Some(status) = mv.status.as_deref() {
            let status = status.to_string();
            crate::resolve::apply_status_from(turn, *target, &status, me, mv.id.as_str())?;
        }
        if let Some(vid) = mv.volatile_status.as_deref() {
            let vid = vid.to_string();
            if vid == "encore" {
                if !apply_encore(turn, target.0, target.1, mv) {
                    log_event!(turn, "{} failed (nothing to encore)", Label(reg, action));
                    turn.move_failed[action.side][action.slot] = true;
                }
                continue;
            }
            if vid == "disable" {
                // Which move is the whole effect, and it fails outright when the target has
                // not moved, so the generic path cannot express it.
                if !apply_disable(turn, target.0, target.1, Some(mv)) {
                    log_event!(turn, "{} failed (nothing to disable)", Label(reg, action));
                    turn.move_failed[action.side][action.slot] = true;
                }
                continue;
            }
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("status move volatile: {vid}"));
            }
            let duration = effect_duration(turn, mv, &vid, action.side, action.slot);
            // Re-seeding a seeded target fails in `addVolatile` (no `onRestart`) and keeps
            // the first planter's slot (IKA-56).
            let already = turn.mon_at(target.0, target.1).is_some_and(|m| m.has_volatile(&vid));
            turn.add_volatile(target.0, target.1, &vid, duration);
            if vid == "taunt" && !already {
                taunt_lasts_longer(turn, target.0, target.1);
            }
            // A second Imprison: `addVolatile` returns false, the move's only effect, so
            // the move fails (IKA-256).
            if vid == "imprison" && already {
                log_event!(turn, "{} failed (already imprisoning)", Label(reg, action));
                turn.move_failed[action.side][action.slot] = true;
            }
            // Leech Seed heals whoever stands in the planter's *slot* at the end of the turn,
            // so the slot is written down: `this.volatiles[status.id].sourceSlot =
            // source.getSlot();` (sim/pokemon.ts:2008).
            if vid == "leechseed" && !already {
                if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
                    if let Some(applied) = mon.volatile_mut("leechseed") {
                        applied.source_slot = Some(Id::new(&format!("{}{}", me.0, me.1)));
                    }
                }
            }
        }
        if let Some(heal) = mv.raw.get("heal") {
            let maxhp = turn.mon_at(target.0, target.1).map(|m| m.maxhp).unwrap_or(0);
            if maxhp > 0 {
                let amount = round_fraction(maxhp, heal);
                turn.heal(target.0, target.1, amount, mv.id.as_str());
            }
        }
    }

    if WEATHER_RECOVERY_MOVES.contains(&mv.id.as_str()) {
        weather_recovery(turn, me, mv);
    }

    // Parting Shot's drops are in an `onHit` handler, so they are not in the dumped
    // declarative fields:
    //     const success = this.boost({atk: -1, spa: -1}, target, source);
    //     if (!success && !target.hasAbility('mirrorarmor')) delete move.selfSwitch;
    // A target that cannot be lowered any further, or that is behind Clear Body, leaves the
    // user standing. Mirror Armor is the exception: it bounces the drops and the user leaves.
    if mv.id == "partingshot" {
        let mut landed = false;
        for target in targets {
            if parting_shot_drops(turn, *target, me) {
                landed = true;
            }
            if matches!(turn.mon_at(target.0, target.1), Some(m) if m.ability == "mirrorarmor") {
                landed = true;
            }
        }
        if !landed {
            suppress_self_switch = true;
            log_event!(turn, "{} did nothing, so nobody switched", Label(reg, action));
        }
    }

    if matches!(mv.id.as_str(), "followme" | "ragepowder" | "spotlight") {
        turn.add_volatile(me.0, me.1, &mv.id, None);
    }
    if mv.id == "helpinghand" {
        for target in targets {
            turn.add_volatile(target.0, target.1, "helpinghand", None);
        }
    }
    if STALL_BUMPING_MOVES.contains(&mv.id.as_str()) {
        turn.add_side_condition(action.side, &mv.id, Some(1));
        bump_stall(turn, action.side, action.slot);
    }
    if mv.id == "perishsong" {
        perish_song(turn, me, action);
    }
    // `runMoveEffects`: `target.side.addSlotCondition(target, moveData.slotCondition)` --
    // Healing Wish's, which heals whoever comes into the slot next (IKA-208).
    if let Some(cid) = mv.raw.get("slotCondition").and_then(Value::as_str) {
        for target in targets {
            let conditions = &mut turn.pos.sides[target.0].slot_conditions;
            if conditions.len() <= target.1 {
                conditions.resize_with(target.1 + 1, Vec::new);
            }
            if !conditions[target.1].iter().any(|c| c.id.as_str() == cid) {
                conditions[target.1].push(Effect::new(Id::new(cid)));
            }
        }
    }
    if matches!(mv.id.as_str(), "trick" | "switcheroo") {
        for target in targets {
            if !swap_items(reg, turn, me, *target) {
                turn.move_failed[me.0][me.1] = true;
            }
        }
    }
    if mv.id == "psychup" {
        psych_up(turn, me, targets);
    }
    // Transform's `onHit: return pokemon.transformInto(target)` (IKA-219).
    if mv.id == "transform" {
        transform_move(turn, me, targets)?;
    }

    if mv.raw.get("hasCustomCode").and_then(Value::as_bool).unwrap_or(false)
        && !crate::modelled::status_move_is_fully_modelled(&mv.id)
    {
        turn.report(format!("status move: {}", mv.id));
    }
    if mv.self_switch && !suppress_self_switch {
        mark_self_switch(turn, action);
    }
    // Memento, Healing Wish: `if (moveData.selfdestruct === 'ifHit' && damage[i] !== false)
    // this.faint(source)` -- a status move's `damage[i]` is `undefined` once it reached a
    // target, so a Memento into -6 or Clear Body still faints its user; Protect, a miss,
    // an immunity or a doll keep the target out of `targets` (IKA-208).
    if mv.raw.get("selfdestruct").and_then(Value::as_str) == Some("ifHit") && !targets.is_empty() {
        turn.faint(me.0, me.1);
    }
    // Roar, Whirlwind (IKA-208): `runMoveEffects` answers `forceSwitch` with
    // `canSwitch(target.side)` -- no one to come in is a failure -- and `forceSwitch` then
    // raises the flag unless `DragOut` stops it (which, for a status move, is a failure
    // only on `false`; Guard Dog, Suction Cups and Ingrain all answer `null`).
    if mv.force_switch {
        for target in targets {
            if !can_switch(turn, target.0) {
                turn.move_failed[me.0][me.1] = true;
                continue;
            }
            let _ = raise_force_switch(turn, me, *target);
        }
    }
    let _ = reg;
    Ok(())
}

/// Transform on its one target: a failed `transformInto` fails the move, and a new
/// ability's `Start` runs at once, as `setAbility` does it (a copied Intimidate lands).
fn transform_move(turn: &mut Turn, me: Slot, targets: &[Slot]) -> Result<(), String> {
    use crate::transform::{transform_into, Transformed};
    for target in targets {
        match transform_into(turn, me, *target, "transform")? {
            Transformed::Failed => turn.move_failed[me.0][me.1] = true,
            Transformed::SameAbility => {}
            Transformed::NewAbility => crate::resolve::switch_in_ability(turn, me.0, me.1),
        }
    }
    Ok(())
}

/// `takeItem` (sim/pokemon.ts): the `TakeItem` event on the holder, which the item's own
/// `onTakeItem` answers -- in Reg M-C only the mega stones have one -- and Unburden's
/// `onTakeItem(item, pokemon) { pokemon.addVolatile('unburden'); }` hears. `None` is
/// `takeItem`'s `undefined` (nothing held), `Some(None)` its `false`.
fn take_item(reg: &Reg, turn: &mut Turn, holder: Slot) -> Option<Option<Id>> {
    let mon = turn.mon_at_mut(holder.0, holder.1)?;
    let item = mon.item?;
    if reg.mega_stone_stays(mon.species.as_str(), item.as_str()) {
        return Some(None);
    }
    mon.item = None;
    if mon.ability == "unburden" && !mon.has_volatile("unburden") {
        mon.volatiles.push(Effect::new(Id::new("unburden")));
    }
    Some(Some(item))
}

/// Trick and Switcheroo's `onHit` (data/moves.ts, IKA-208):
///
///     const yourItem = target.takeItem(source);
///     const myItem = source.takeItem();
///     if (yourItem === false || myItem === false || (!yourItem && !myItem)) { ...restore; return false; }
///     if ((myItem && !this.singleEvent('TakeItem', myItem, ..., target, source, move, myItem)) ||
///         (yourItem && !this.singleEvent('TakeItem', yourItem, ..., source, target, move, yourItem))) {
///         ...restore; return false;
///     }
///     if (myItem) target.setItem(myItem); ...  if (yourItem) source.setItem(yourItem); ...
///
/// The second test asks each item's `onTakeItem` about its *receiver*, so a mega stone cannot
/// be handed to its own species either. `setItem` then runs the new item's `onStart`: a
/// Choice item drops `choicelock`, a terrain seed on its terrain is used at once, a White
/// Herb clears lowered stats. A berry is eaten at the next `Update`, which comes before
/// anything else moves. The giver of a Choice item keeps `choicelock` until
/// `onDisableMove` at the end of the turn (`choice_lock_ends`). Returns false for a failure.
fn swap_items(reg: &Reg, turn: &mut Turn, me: Slot, target: Slot) -> bool {
    let alive = |turn: &Turn, at: Slot| matches!(turn.mon_at(at.0, at.1), Some(m) if !m.fainted);
    if !alive(turn, me) || !alive(turn, target) {
        return false;
    }
    let yours = take_item(reg, turn, target);
    let mine = take_item(reg, turn, me);
    let restore = |turn: &mut Turn, yours: Option<Option<Id>>, mine: Option<Option<Id>>| {
        if let Some(Some(item)) = yours {
            turn.mon_at_mut(target.0, target.1).unwrap().item = Some(item);
        }
        if let Some(Some(item)) = mine {
            turn.mon_at_mut(me.0, me.1).unwrap().item = Some(item);
        }
    };
    if matches!(yours, Some(None)) || matches!(mine, Some(None)) || (yours.is_none() && mine.is_none()) {
        restore(turn, yours, mine);
        return false;
    }
    let (yours, mine) = (yours.flatten(), mine.flatten());
    let receiver_species = |turn: &Turn, at: Slot| turn.mon_at(at.0, at.1).unwrap().species;
    let refused = mine.is_some_and(|i| reg.mega_stone_stays(receiver_species(turn, target).as_str(), i.as_str()))
        || yours.is_some_and(|i| reg.mega_stone_stays(receiver_species(turn, me).as_str(), i.as_str()));
    if refused {
        restore(turn, yours.map(Some), mine.map(Some));
        return false;
    }
    log_event!(turn, "{} and {} swapped items", Name(me.0, me.1), Name(target.0, target.1));
    for (at, item) in [(target, mine), (me, yours)] {
        let Some(item) = item else { continue };
        turn.mon_at_mut(at.0, at.1).unwrap().item = Some(item);
        if reg.choice_items.contains(item.as_str()) {
            let mon = turn.mon_at_mut(at.0, at.1).unwrap();
            mon.volatiles.retain(|v| v.id.as_str() != "choicelock");
        }
        crate::terrain::use_terrain_seed(turn, at.0, at.1);
    }
    crate::resolve::check_white_herb(turn);
    for at in [target, me] {
        eat_received_berry(turn, at);
    }
    true
}

/// The `onUpdate` of a berry just received: Sitrus and Oran below half, Lum on any status or
/// confusion, Persim on confusion -- the berries this port eats elsewhere (IKA-208).
fn eat_received_berry(turn: &mut Turn, at: Slot) {
    turn.check_berry(at.0, at.1);
    let cures = match turn.mon_at(at.0, at.1) {
        Some(mon) if !mon.fainted => {
            let confused = mon.has_volatile("confusion");
            (is(mon.item, "lumberry") && (mon.status.is_some() || confused))
                || (is(mon.item, "persimberry") && confused)
        }
        _ => false,
    };
    if !cures || turn.berries_blocked(at.0) {
        return;
    }
    let berry = turn.mon_at(at.0, at.1).unwrap().item;
    let lum = is(berry, "lumberry");
    turn.consume_item(at.0, at.1, berry.as_ref().map(|b| b.as_str()).unwrap_or(""));
    let mon = turn.mon_at_mut(at.0, at.1).unwrap();
    mon.volatiles.retain(|v| v.id.as_str() != "confusion");
    if lum {
        mon.status = None;
        mon.status_counter = None;
    }
}

/// Perish Song's `onHitField` (`data/moves.ts`): every active Pokemon on both sides, the
/// singer's included, gets `perishsong` at duration 4, which the residual below counts
/// down from the end of this same turn, so the faint lands three turns later. Soundproof's
/// `onTryHit` (`target !== source`) is the `null` that still counts as a result, and it is
/// `breakable`: a Mold Breaker singer (Mycelium Might too, this being a status move)
/// reaches it unless the holder has an Ability Shield. Nobody reached and nobody
/// Soundproof -- everyone already counting -- is `return false`, a failure. IKA-172: the
/// port had no such path while `modelled.rs` listed the move, so its turn left nobody
/// counting down and reported nothing.
///
/// ```text
/// for (const pokemon of this.getAllActive()) {
///     if (this.runEvent('Invulnerability', pokemon, source, move) === false) { ...
///     } else if (this.runEvent('TryHit', pokemon, source, move) === null) {
///         result = true;
///     } else if (!pokemon.volatiles['perishsong']) {
///         pokemon.addVolatile('perishsong');
/// ...
/// if (!result) return false;
/// ```
///
/// Nothing is ever semi-invulnerable here. Good as Gold's `onTryHit` (a status move,
/// IKA-202) answers `null` as Soundproof's does.
fn perish_song(turn: &mut Turn, me: Slot, action: &QueuedAction) {
    let ignores_ability = turn
        .mon_at(me.0, me.1)
        .is_some_and(|singer| is_mold_breaker(singer.ability.as_str()));
    let mut result = false;
    for side in 0..2 {
        for slot in 0..turn.pos.sides[side].active.len() {
            let Some(mon) = turn.mon_at(side, slot) else { continue };
            if mon.fainted {
                continue;
            }
            let shielded = matches!(mon.item, Some(i) if i.as_str() == "abilityshield");
            if (mon.ability == "soundproof" || mon.ability == "goodasgold")
                && (side, slot) != me
                && !(ignores_ability && !shielded)
            {
                let ability = mon.ability;
                log_event!(turn, "{} immune ({})", Name(side, slot), ability);
                result = true;
                continue;
            }
            if mon.has_volatile("perishsong") {
                continue;
            }
            turn.add_volatile(side, slot, "perishsong", Some(4));
            log_event!(turn, "{} perish3", Name(side, slot));
            result = true;
        }
    }
    if !result {
        log_event!(turn, "{} failed (everyone is already counting)", Label(turn.reg, action));
        turn.move_failed[me.0][me.1] = true;
    }
}

// ---------------------------------------------------------------------------
// End of turn
// ---------------------------------------------------------------------------

/// Turns a `source_slot` into our (side, slot) pair, in either encoding in use. Reading only
/// one of the two once left Leech Seed's heal unreachable for every seed the resolver
/// planted (IKA-56).
fn slot_of(turn: &Turn, source_slot: Option<Id>) -> Option<Slot> {
    let source = source_slot?;
    let text = source.as_str();
    let bytes = text.as_bytes();
    // Two encodings are in use: the resolver writes "10" (side digit, slot digit), a
    // position read from Showdown carries its own label "p2a" (IKA-56).
    let (side, slot) = if bytes.len() >= 3 && bytes[0] == b'p' {
        let side = ((bytes[1] as char).to_digit(10)? as usize).checked_sub(1)?;
        let slot = (bytes[2] as usize).checked_sub(b'a' as usize)?;
        (side, slot)
    } else if bytes.len() >= 2 {
        let side = (bytes[0] as char).to_digit(10)? as usize;
        let slot = (bytes[1] as char).to_digit(10)? as usize;
        (side, slot)
    } else {
        return None;
    };
    if side > 1 || slot >= turn.pos.sides[side].active.len() {
        return None;
    }
    Some((side, slot))
}

/// Whether the Pokemon that applied a trap has left the field. A trap whose source is gone
/// ends without dealing damage, so keeping it going costs the trapped Pokemon an eighth of
/// its HP a turn that it should not lose.
fn trapper_gone(turn: &Turn, source_slot: Option<Id>) -> bool {
    let Some((side, slot)) = slot_of(turn, source_slot) else { return false };
    match turn.mon_at(side, slot) {
        None => true,
        Some(mon) => mon.fainted,
    }
}

/// Active slots in Showdown's residual order: by Speed, fastest first.
///
/// ```text
/// eachEvent(eventid, effect, relayVar) {
///     const actives = this.getAllActive();
///     ...
///     this.speedSort(actives, (a, b) => b.speed - a.speed);
/// ```
///
/// `pokemon.speed` is the Trick-Room-inverted value, so under Trick Room the order reverses
/// -- the same quantity the action queue sorts on. This once iterated side 0 then side 1,
/// which made the residual phase seat-dependent: in a mirrored position our burn and trap
/// resolved before their poison in *both* orientations, so a faint the residuals cause
/// landed on a different side depending on which seat we held (4.8 points on one cell of a
/// turn-14 position). Empty and fainted slots keep their place in the list and sort last.
///
/// It is computed once for the whole phase, for two reasons that usually disagree.
/// Faithfulness: `eachEvent('Residual')` speed-sorts the actives a single time and walks that
/// list, so a Speed change *during* the phase (Speed Boost is itself an `onResidual`) does
/// not reorder what is left of it. Cost: each computation is four Speed calculations, each
/// of which builds a battler; eight per phase once made it 30% of generation time.
fn residual_order(turn: &Turn) -> Result<(Vec<Slot>, bool), String> {
    let trick_room = turn.pos.field.trick_room();
    let field = turn.field();
    let mut entries: Vec<(i64, i64, String, usize, usize)> = Vec::new();
    let mut speeds: Vec<i64> = Vec::new();
    for side in 0..2 {
        let conditions = turn.pos.sides[side].side_conditions.clone();
        for slot in 0..turn.pos.sides[side].active.len() {
            let mon = turn.mon_at(side, slot);
            let fighter = turn.battler_at(side, slot)?;
            match (mon, fighter) {
                (Some(mon), Some(fighter)) => {
                    let mut speed = crate::speed::effective_speed(&fighter, &field, &conditions);
                    if trick_room {
                        speed = 10000 - speed;
                    }
                    speeds.push(speed);
                    entries.push((0, -speed, mon.species.as_str().to_string(), slot, side));
                }
                _ => entries.push((1, 0, String::new(), slot, side)),
            }
        }
    }
    let mut unique = speeds.clone();
    unique.sort_unstable();
    unique.dedup();
    let tied = unique.len() != speeds.len();
    // A Speed tie here is reported by Python rather than branched, and broken by
    // (-speed, species, slot, side) -- deliberately not by side, so the residual phase
    // cannot become seat-dependent. The same sort gives the same order, so the tie is not
    // a reason to refuse; it is a reason to sort on exactly the same key.
    entries.sort();
    Ok((
        entries.into_iter().map(|(_, _, _, slot, side)| (side, slot)).collect(),
        tied,
    ))
}

/// End-of-turn effects, in Showdown's residual order.
pub(crate) fn residuals(reg: &Reg, turn: &mut Turn) -> Result<(), String> {
    crate::resolve::RESIDUALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = reg;
    #[cfg(not(feature = "ika215-control"))]
    turn.begin(|| "residual".to_string());
    // Sorted before anything ends, as Showdown's `updateSpeed()` and `fieldEvent`'s one
    // `speedSort` run before the weather's handler decrements it: on the turn the sun runs
    // out, Chlorophyll's doubled Speed still orders the phase (IKA-190).
    let (order, tied) = residual_order(turn)?;
    if tied {
        turn.report("residual speed tie (Showdown breaks it at random)");
    }

    // Residual order 1: weather. Its duration is decremented *before* its handler runs and
    // the handler is skipped when it expires, so the last turn of a sandstorm deals no
    // damage at all.
    let mut weather_expired = false;
    if turn.pos.field.weather.is_some() {
        if let Some(duration) = turn.pos.field.weather_duration {
            let left = duration - 1;
            turn.pos.field.weather_duration = Some(left);
            if left <= 0 {
                if let Some(weather) = turn.pos.field.weather {
                    log_event!(turn, "{} ended", weather);
                }
                turn.pos.field.weather = None;
                turn.pos.field.weather_duration = None;
                weather_expired = true;
            }
        }
    }
    // IKA-210's positive control orders the residuals after the weather ended, and notes a
    // tie there (IKA-190's bug in Python).
    #[cfg(feature = "ika210-control")]
    let (order, tied_after) = residual_order(turn)?;
    #[cfg(feature = "ika210-control")]
    if tied_after && !tied {
        turn.report("residual speed tie (Showdown breaks it at random)");
    }

    if matches!(turn.pos.field.weather, Some(w) if w.as_str() == "sandstorm") && !weather_expired
    {
        for (side, slot) in order.iter().copied() {
            let skip = match turn.mon_at(side, slot) {
                None => true,
                Some(mon) => {
                    mon.fainted
                        || turn
                            .types_of(mon)
                            .as_slice()
                            .iter()
                            .any(|t| matches!(t.as_str(), "Rock" | "Ground" | "Steel"))
                        || matches!(
                            mon.ability.as_str(),
                            "sandveil" | "sandrush" | "sandforce" | "overcoat" | "magicguard"
                        )
                        || is(mon.item, "safetygoggles")
                }
            };
            if skip {
                continue;
            }
            let amount = turn.fraction_of_max(side, slot, SANDSTORM_DAMAGE);
            turn.deal_damage(side, slot, amount, false, "sandstorm")?;
        }
    }

    // Abilities that trade HP with the weather. Showdown hangs these on `onWeather`, which
    // runs with the weather's own residual, so they go here beside the sandstorm.
    if turn.pos.field.weather.is_some() && !weather_expired {
        let weather = turn.pos.field.weather.unwrap();
        for (side, slot) in order.iter().copied() {
            let ability = match turn.mon_at(side, slot) {
                None => continue,
                Some(mon) if mon.fainted => continue,
                Some(mon) => mon.ability,
            };
            let ratio: Option<(i64, i64)> = match (ability.as_str(), weather.as_str()) {
                ("solarpower", "sunnyday") | ("solarpower", "desolateland") => Some((-1, 8)),
                ("dryskin", "sunnyday") | ("dryskin", "desolateland") => Some((-1, 8)),
                ("dryskin", "raindance") | ("dryskin", "primordialsea") => Some((1, 8)),
                ("raindish", "raindance") | ("raindish", "primordialsea") => Some((1, 16)),
                ("icebody", "hail") | ("icebody", "snowscape") => Some((1, 16)),
                _ => None,
            };
            let Some((numerator, denominator)) = ratio else { continue };
            let amount = turn.fraction_of_max(side, slot, (numerator.abs(), denominator));
            if numerator < 0 {
                turn.deal_damage(side, slot, amount, false, ability.as_str())?;
            } else {
                turn.heal(side, slot, amount, ability.as_str());
            }
        }
    }

    crate::terrain::grassy_terrain_heal(turn, &order);

    for (side, slot) in order.iter().copied() {
        let leftovers = matches!(turn.mon_at(side, slot), Some(mon)
            if !mon.fainted && is(mon.item, "leftovers"));
        if leftovers {
            let amount = turn.fraction_of_max(side, slot, LEFTOVERS_HEAL);
            turn.heal(side, slot, amount, "leftovers");
        }
    }

    for (side, slot) in order.iter().copied() {
        let seed_source = match turn.mon_at(side, slot) {
            None => continue,
            Some(mon) if mon.fainted || mon.ability == "magicguard" => continue,
            Some(mon) => mon.volatile("leechseed").map(|s| s.source_slot),
        };
        let Some(source_slot) = seed_source else { continue };
        // data/moves.ts:10218-10227: `getAtSlot(sourceSlot)`, and when that slot is empty
        // or fainted the seed returns before `this.damage` -- no drain at all (IKA-56).
        //     const target = this.getAtSlot(pokemon.volatiles['leechseed'].sourceSlot);
        //     if (!target || target.fainted || target.hp <= 0) { return; }
        //     const damage = this.damage(pokemon.baseMaxhp / 8, pokemon, target);
        //     if (damage) { this.heal(damage, target, pokemon); }
        // A *slot*, not a Pokemon: if the planter switched out, whoever replaced it is
        // healed. (Big Root and Liquid Ooze act on the heal; neither is modelled.)
        let Some(planter) = slot_of(turn, source_slot) else { continue };
        let alive =
            matches!(turn.mon_at(planter.0, planter.1), Some(m) if !m.fainted && m.hp > 0);
        if !alive {
            continue;
        }
        let amount = turn.fraction_of_max(side, slot, LEECH_SEED_DRAIN);
        let drained = turn.deal_damage(side, slot, amount, false, "leechseed")?;
        if drained > 0 {
            turn.heal(planter.0, planter.1, drained, "leechseed");
        }
    }

    for (side, slot) in order.iter().copied() {
        let (status, has_trap, trap_source, has_saltcure, maxhp, counter) =
            match turn.mon_at(side, slot) {
                None => continue,
                Some(mon) if mon.fainted || mon.ability == "magicguard" => continue,
                Some(mon) => (
                    mon.status,
                    mon.has_volatile("partiallytrapped"),
                    mon.volatile("partiallytrapped").and_then(|t| t.source_slot),
                    mon.has_volatile("saltcure"),
                    mon.maxhp,
                    mon.status_counter,
                ),
            };
        if is(status, "brn") {
            let amount = turn.fraction_of_max(side, slot, BURN_DAMAGE);
            turn.deal_damage(side, slot, amount, false, "brn")?;
        } else if is(status, "psn") {
            let amount = turn.fraction_of_max(side, slot, POISON_DAMAGE);
            turn.deal_damage(side, slot, amount, false, "psn")?;
        } else if is(status, "tox") {
            // Showdown: `clampIntRange(baseMaxhp / 16, 1) * stage` -- the sixteenth is
            // truncated first and then multiplied, which is not truncating the product.
            let stage = (counter.unwrap_or(0) + 1).min(15);
            if let Some(mon) = turn.mon_at_mut(side, slot) {
                mon.status_counter = Some(stage);
            }
            let per_stage = (maxhp / 16).max(1);
            turn.deal_damage(side, slot, per_stage * stage, false, "tox")?;
        }
        // Read again, as Python's `mon.volatile(...)` is: a faint to the status damage
        // above has cleared it, and a freed line would follow (IKA-215).
        let has_trap =
            has_trap && turn.mon_at(side, slot).is_some_and(|m| m.has_volatile("partiallytrapped"));
        if has_trap {
            if trapper_gone(turn, trap_source) {
                if let Some(mon) = turn.mon_at_mut(side, slot) {
                    mon.volatiles.retain(|v| v.id.as_str() != "partiallytrapped");
                }
                log_event!(turn, "{} freed (trapper left)", Name(side, slot));
            } else {
                let amount = turn.fraction_of_max(side, slot, PARTIAL_TRAP_DAMAGE);
                turn.deal_damage(side, slot, amount, false, "partiallytrapped")?;
            }
        }
        if has_saltcure {
            let weak = match turn.mon_at(side, slot) {
                None => false,
                Some(mon) => {
                    let types = turn.types_of(mon);
                    types.contains("Water") || types.contains("Steel")
                }
            };
            let ratio = if weak { SALT_CURE_DAMAGE_WEAK } else { SALT_CURE_DAMAGE };
            let amount = turn.fraction_of_max(side, slot, ratio);
            turn.deal_damage(side, slot, amount, false, "saltcure")?;
        }
    }

    // Residual order 24: Perish Song. The counter is decremented like any other duration,
    // but on expiry the effect *runs* -- `onEnd` faints the Pokemon -- where weather's handler
    // is skipped instead. Missing it leaves a Pokemon alive that the game killed. (It is not
    // decremented again in the volatile loop below, which would kill a turn and a half early.)
    for (side, slot) in order.iter().copied() {
        let left = {
            let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
            if mon.fainted {
                continue;
            }
            match mon.volatile_mut("perishsong") {
                None => continue,
                Some(perish) => match perish.duration {
                    None => continue,
                    Some(duration) => {
                        perish.duration = Some(duration - 1);
                        duration - 1
                    }
                },
            }
        };
        log_event!(turn, "{} perish{}", Name(side, slot), left.max(0));
        if left <= 0 {
            turn.faint(side, slot);
        }
    }

    // Residual order 28: Speed Boost. `if (pokemon.activeTurns)` is what stops it firing on
    // the turn its holder came in, and `newly_switched` is that flag.
    for (side, slot) in order.iter().copied() {
        let boosts = matches!(turn.mon_at(side, slot), Some(mon)
            if !mon.fainted && mon.ability == "speedboost" && !mon.newly_switched);
        if boosts {
            turn.apply_boosts(side, slot, &[("spe", 1)], false, "speedboost");
        }
    }

    // Residual order 29: the last of White Herb's four chances to fire.
    check_white_herb(turn);

    for side in 0..2 {
        let mut kept: Vec<Effect> = Vec::new();
        for mut condition in std::mem::take(&mut turn.pos.sides[side].side_conditions) {
            if let Some(duration) = condition.duration {
                condition.duration = Some(duration - 1);
                if duration - 1 <= 0 {
                    log_event!(turn, "p{} side -{}", side + 1, condition.id);
                    continue;
                }
            }
            kept.push(condition);
        }
        turn.pos.sides[side].side_conditions = kept;
    }

    if let Some(duration) = turn.pos.field.terrain_duration {
        turn.pos.field.terrain_duration = Some(duration - 1);
        if duration - 1 <= 0 {
            if turn.log.is_some() {
                let terrain = turn.pos.field.terrain;
                log_event!(turn, "{} ended", crate::moves::OrNone(terrain));
            }
            turn.pos.field.terrain = None;
            turn.pos.field.terrain_duration = None;
        }
    }
    let mut kept_pseudo: Vec<Effect> = Vec::new();
    for mut pseudo in std::mem::take(&mut turn.pos.field.pseudo_weather) {
        if let Some(duration) = pseudo.duration {
            pseudo.duration = Some(duration - 1);
            if duration - 1 <= 0 {
                log_event!(turn, "{} ended", pseudo.id);
                continue;
            }
        }
        kept_pseudo.push(pseudo);
    }
    turn.pos.field.pseudo_weather = kept_pseudo;

    rampage_runs_out(turn, &order);
    for (side, slot) in order.iter().copied() {
        let failed = turn.move_failed[side][slot];
        let logging = turn.log.is_some();
        let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
        let mut kept: Vec<Effect> = Vec::new();
        let mut yawn_expired = false;
        // Python's order of the loop's lines: Yawn's sleep (true) and an Encore that ran
        // out of PP (false), as the volatiles meet them (IKA-215).
        let mut said: Vec<bool> = Vec::new();
        for mut volatile in std::mem::take(&mut mon.volatiles) {
            // The Protect-family volatiles last one turn. `stall` is deliberately not here:
            // it carries the repeat counter and expires on its own two-turn duration. Yawn's
            // whole effect is on expiry: the target falls asleep at the end of the turn after
            // it lands. An Encore whose move ran out of PP ends early (`onResidual`), or the
            // only legal move would be an unusable one.
            let single_turn = PROTECT_VOLATILES.iter().any(|(v, _)| *v == volatile.id.as_str())
                || matches!(
                    volatile.id.as_str(),
                    "flinch" | "helpinghand" | "followme" | "ragepowder" | "spotlight"
                        | "glaiverush"
                );
            if single_turn {
                continue;
            }
            if volatile.id.as_str() == "perishsong" {
                kept.push(volatile);
                continue;
            }
            if let Some(duration) = volatile.duration {
                volatile.duration = Some(duration - 1);
                if duration - 1 <= 0 {
                    if volatile.id.as_str() == "yawn" {
                        yawn_expired = true;
                        if logging {
                            said.push(true);
                        }
                    }
                    if volatile.id.as_str() == "disable" {
                        // The flag lives on the move slot, so it has to be cleared here or
                        // the move stays unusable for the rest of the battle.
                        if let Some(move_id) = volatile.move_id {
                            if let Some(move_slot) = mon.moves.get_mut(move_id) {
                                move_slot.disabled = false;
                            }
                        }
                    }
                    continue;
                }
            }
            if volatile.id.as_str() == "encore" {
                // `onResidual`: an Encore whose move has run out of PP ends early.
                let spent = volatile
                    .move_id
                    .and_then(|id| mon.moves.get(id))
                    .map(|slot| slot.pp > 0)
                    .unwrap_or(false);
                if !spent {
                    if logging && volatile.move_id.is_some() {
                        said.push(false);
                    }
                    continue;
                }
            }
            kept.push(volatile);
        }
        mon.volatiles = kept;
        mon.newly_switched = false;
        mon.move_last_turn_failed = failed;
        if !logging && yawn_expired {
            turn.apply_status_unveiled(side, slot, "slp", "yawn")?;
        }
        for yawn in said {
            if yawn {
                turn.apply_status_unveiled(side, slot, "slp", "yawn")?;
            } else {
                log_event!(turn, "{} is free of encore (no PP)", Name(side, slot));
            }
        }
    }
    rampage_residual(turn, &order);
    choice_lock_ends(reg, turn, &order);

    settle_outcome(&mut turn.pos, &turn.wipe_order);
    Ok(())
}

/// Marks the battle over and names the winner, the way Showdown's `checkWin` does:
///
/// ```text
/// checkWin(faintData) {
///     if (this.sides.every(side => !side.pokemonLeft)) {
///         this.win(faintData && this.gen > 4 ? faintData.target.side : null);
/// ```
///
/// Two cases and one rule. One side out: the other wins, and Showdown ends the battle at the
/// faint that emptied it rather than at the end of the turn. Both sides out: gen 5 and later
/// give it to the side of the Pokemon that fainted *last*. Both readings are "whichever
/// side's wipe-out completed last wins", so `wipe_order` -- recorded as the faints happen,
/// because by now both sides just look empty -- answers them together.
///
/// This was once wrong in a way only a rare position exposed: a loop over the sides that
/// assigned a winner per wiped side, so with both wiped the second iteration overwrote the
/// first and side 0 won every mutual knockout -- a seat-dependent result on the closest
/// games there are, every one labelled a win for our roster in the training data. Without
/// `wipe_order` (a position that arrived already empty) a mutual wipe-out cannot be
/// attributed and is left as a draw rather than guessed.
pub fn settle_outcome(pos: &mut Position, wipe_order: &[usize]) {
    let wiped: Vec<usize> = (0..pos.sides.len())
        .filter(|index| pos.sides[*index].pokemon.iter().all(|m| m.fainted))
        .collect();
    if wiped.is_empty() {
        return;
    }
    pos.ended = true;
    if wiped.len() == 1 {
        pos.winner = Some(pos.sides[1 - wiped[0]].id.clone());
        return;
    }
    let ordered: Vec<usize> = wipe_order
        .iter()
        .copied()
        .filter(|index| wiped.contains(index))
        .collect();
    pos.winner = ordered.last().map(|index| pos.sides[*index].id.clone());
}

const BURN_DAMAGE: (i64, i64) = (1, 16);
const POISON_DAMAGE: (i64, i64) = (1, 8);
const SANDSTORM_DAMAGE: (i64, i64) = (1, 16);
const LEECH_SEED_DRAIN: (i64, i64) = (1, 8);
const LEFTOVERS_HEAL: (i64, i64) = (1, 16);
const PARTIAL_TRAP_DAMAGE: (i64, i64) = (1, 8);
// The champions mod's Salt Cure, half the base game's (IKA-159); see resolve.rs.
const SALT_CURE_DAMAGE: (i64, i64) = (1, 16);
const SALT_CURE_DAMAGE_WEAK: (i64, i64) = (1, 8);

// The port's own unit tests of what Python's test_resolve held (IKA-210).
#[cfg(test)]
#[path = "moves_unit_tests.rs"]
mod unit_tests;
