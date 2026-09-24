//! Effective Speed, move priority, and the order actions resolve in.
//!
//! Transcribed from `src/pokeuraou/speed.py`, scalar rather than vectorised: the resolver
//! works on a fully-known position, so there is one Speed per Pokemon rather than one per
//! belief particle.

use crate::battler::{Battler, FieldState, V_UNBURDEN};
use crate::fixedpoint::Chain;
use crate::id::Id;
use crate::reg::Reg;

pub const ORDER_SWITCH: i64 = 103;
pub const ORDER_MEGA: i64 = 104;
pub const ORDER_MOVE: i64 = 200;

/// Showdown caps Speed at 10000 unless the format sets `battle.trunc`, which these do not.
pub const SPEED_CAP: i64 = 10000;

pub const QUICK_CLAW_CHANCE: f64 = 0.2;
pub const QUICK_DRAW_CHANCE: f64 = 0.3;

fn speed_weather_abilities(ability: &str) -> &'static [&'static str] {
    match ability {
        "chlorophyll" => &["sunnyday", "desolateland"],
        "swiftswim" => &["raindance", "primordialsea"],
        "sandrush" => &["sandstorm"],
        "slushrush" => &["hail", "snowscape", "snow"],
        _ => &[],
    }
}

fn speed_item_ratio(item: &str) -> Option<(f64, f64)> {
    match item {
        "choicescarf" => Some((1.5, 1.0)),
        "ironball" | "machobrace" | "powerweight" | "powerbracer" | "powerbelt" | "powerlens"
        | "powerband" | "poweranklet" => Some((0.5, 1.0)),
        _ => None,
    }
}

fn is(value: Option<Id>, name: &str) -> bool {
    matches!(value, Some(v) if v.as_str() == name)
}

/// Weather as this Pokemon experiences it, for the Speed abilities.
fn effective_weather(field: &FieldState, mon: &Battler) -> Option<Id> {
    let weather = field.weather?;
    let suppressed = field
        .active_abilities
        .iter()
        .flatten()
        .any(|a| matches!(a.as_str(), "cloudnine" | "airlock"));
    if suppressed {
        return None;
    }
    if is(mon.item, "utilityumbrella")
        && matches!(
            weather.as_str(),
            "sunnyday" | "raindance" | "desolateland" | "primordialsea"
        )
    {
        return None;
    }
    Some(weather)
}

/// `Pokemon#getStat('spe')`: boosts, then one accumulated ModifySpe chain, then paralysis.
/// The side's conditions are taken as they are stored rather than as a list of ids: the
/// caller was collecting one per action, and the queue is re-sorted five times a turn.
pub fn effective_speed(
    mon: &Battler,
    field: &FieldState,
    side_conditions: &[crate::position::Effect],
) -> i64 {
    let mut spe = mon.stat("spe", false);
    let mut chain = Chain::new();

    let has = |name: &str| side_conditions.iter().any(|c| c.id.as_str() == name);
    if has("tailwind") {
        chain.add(2.0, 1.0, "tailwind");
    }
    if has("grasspledge") {
        chain.add(0.25, 1.0, "grasspledge");
    }

    let weathers = speed_weather_abilities(mon.ability.as_str());
    if !weathers.is_empty() {
        if let Some(weather) = effective_weather(field, mon) {
            if weathers.contains(&weather.as_str()) {
                chain.add(2.0, 1.0, "speedweather");
            }
        }
    }
    if mon.ability == "surgesurfer" && is(field.terrain, "electricterrain") {
        chain.add(2.0, 1.0, "surgesurfer");
    }
    if mon.ability == "quickfeet" && mon.status.is_some() {
        chain.add(1.5, 1.0, "quickfeet");
    }
    if mon.ability == "unburden" && mon.item.is_none() && mon.volatiles.has(V_UNBURDEN) {
        chain.add(2.0, 1.0, "unburden");
    }
    // Slow Start's counter lives in `abilityState`, which this port does not carry into
    // the Battler; a holder is refused by the resolver instead of being sped up wrongly.
    if let Some(item) = mon.item {
        if let Some((num, den)) = speed_item_ratio(item.as_str()) {
            chain.add(num, den, "speeditem");
        } else if item.as_str() == "quickpowder" && mon.species == "ditto" {
            chain.add(2.0, 1.0, "quickpowder");
        }
    }

    spe = chain.apply(spe);
    if is(mon.status, "par") && mon.ability != "quickfeet" {
        spe = spe * 50 / 100;
    }
    spe.min(SPEED_CAP)
}

pub fn move_priority(reg: &Reg, move_id: &str, mon: &Battler, field: &FieldState) -> i64 {
    let Some(mv) = reg.moves.get(move_id) else {
        // The recharge turn's fake move: queued at plain priority 0.
        return 0;
    };
    let mut priority = mv.priority;
    if mon.ability == "prankster" {
        if mv.category == "Status" {
            priority += 1;
        }
    } else if mon.ability == "galewings" {
        if mv.mtype == "Flying" && mon.at_full_hp() {
            priority += 1;
        }
    } else if mon.ability == "triage" && mv.has_flag(crate::reg::F_HEAL) {
        priority += 3;
    }
    // `source.isGrounded()` too (IKA-201).
    if move_id == "grassyglide"
        && is(field.terrain, "grassyterrain")
        && crate::damage::is_grounded(mon)
    {
        priority += 1;
    }
    priority
}

/// Fractional priority outcomes as (value, probability).
///
/// Showdown's `FractionalPriority` event with relay value 0, handlers in
/// `onFractionalPriorityPriority` order, each seeing the value the previous ones left
/// (IKA-145): Stall / Lagging Tail / Full Incense set -0.1; Mycelium Might sets -0.1 on a
/// status move; Quick Draw rolls 3/10 for 0.1 on a move that is not a status move; Quick
/// Claw (skipped on a status move under Mycelium Might) rolls 1/5 for 0.1 if the value so
/// far is <= 0. That `priority` is the relay value, not the move's priority.
pub fn fractional_priority(reg: &Reg, move_id: &str, mon: &Battler) -> Vec<(f64, f64)> {
    let Some(mv) = reg.moves.get(move_id) else {
        return vec![(0.0, 1.0)];
    };
    let status = mv.category == "Status";
    let mut base = 0.0;
    if mon.ability == "stall" || is(mon.item, "laggingtail") || is(mon.item, "fullincense") {
        base = -0.1;
    }
    if mon.ability == "myceliummight" && status {
        base = -0.1;
    }
    let mut outcomes = vec![(base, 1.0)];
    if mon.ability == "quickdraw" && !status {
        outcomes = vec![(0.1, QUICK_DRAW_CHANCE), (base, 1.0 - QUICK_DRAW_CHANCE)];
    }
    if is(mon.item, "quickclaw") && !(status && mon.ability == "myceliummight") {
        // Merged by value in first-seen order, as the Python dict does.
        let mut rolled: Vec<(f64, f64)> = Vec::with_capacity(3);
        fn add(rolled: &mut Vec<(f64, f64)>, value: f64, chance: f64) {
            if let Some(entry) = rolled.iter_mut().find(|(v, _)| *v == value) {
                entry.1 += chance;
            } else {
                rolled.push((value, chance));
            }
        }
        for (value, chance) in outcomes {
            if value <= 0.0 {
                add(&mut rolled, 0.1, chance * QUICK_CLAW_CHANCE);
                add(&mut rolled, value, chance * (1.0 - QUICK_CLAW_CHANCE));
            } else {
                add(&mut rolled, value, chance);
            }
        }
        outcomes = rolled;
    }
    outcomes
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueuedAction {
    pub side: usize,
    pub slot: usize,
    pub kind: ActionKind,
    pub order: i64,
    pub priority: i64,
    pub fractional: f64,
    pub speed: i64,
    pub move_id: Option<Id>,
    pub target: Option<i64>,
    pub switch_to: Option<usize>,
    pub switch_species: Option<Id>,
    pub branch_probability: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Switch,
    Mega,
    Move,
}

/// The three comparison columns Showdown sorts on, as one key per action.
fn sort_key(action: &QueuedAction, trick_room: bool) -> (i64, i64, i64) {
    let speed = if trick_room { -action.speed } else { action.speed };
    (
        action.order,
        ((action.priority as f64 + action.fractional) * 10.0).round() as i64,
        speed,
    )
}

/// Indices in resolution order, plus the groups that compare exactly equal.
///
/// Showdown shuffles a tied group, so a tie is a coin flip the caller must branch on.
pub fn order_actions(
    actions: &[QueuedAction],
    trick_room: bool,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut indices: Vec<usize> = (0..actions.len()).collect();
    // Ascending order, descending priority, descending speed -- by Showdown's own selection
    // sort, so the canonical order of a tie is the one Showdown's queue is left in when its
    // shuffle keeps the tied group as it lies (the oracle's `speedTie: 'keep'`, IKA-238).
    showdown_speed_sort(&mut indices, |a, b| {
        let (oa, pa, sa) = sort_key(&actions[a], trick_room);
        let (ob, pb, sb) = sort_key(&actions[b], trick_room);
        oa.cmp(&ob).then(pb.cmp(&pa)).then(sb.cmp(&sa))
    });

    let mut ties: Vec<Vec<usize>> = Vec::new();
    let mut run: Vec<usize> = vec![];
    for (position, index) in indices.iter().enumerate() {
        if position == 0 {
            run.push(*index);
            continue;
        }
        let previous = indices[position - 1];
        if sort_key(&actions[previous], trick_room) == sort_key(&actions[*index], trick_room) {
            run.push(*index);
        } else {
            if run.len() > 1 {
                ties.push(run.clone());
            }
            run = vec![*index];
        }
    }
    if run.len() > 1 {
        ties.push(run);
    }
    (indices, ties)
}

/// `Battle#speedSort` (sim/battle.ts), with the tied group's shuffle left out (IKA-238).
///
/// A selection sort: it picks the first of the remaining entries (with every entry tied to
/// it, in list order) and swaps them to the front. The swaps move the entries they pass
/// over, so a tie comes out in an order a stable sort does not give: with Incineroar
/// and Incineroar tied behind two faster Pokemon, `[p1a, p1b, p2a, p2b]` sorts to
/// `[p1b, p2b, p2a, p1a]`. `ties` and the tie permutations are unchanged by it.
///
/// ```text
/// while (sorted + 1 < list.length) {
///     let nextIndexes = [sorted];
///     for (let i = sorted + 1; i < list.length; i++) {
///         const delta = comparator(list[nextIndexes[0]], list[i]);
///         if (delta < 0) continue;
///         if (delta > 0) nextIndexes = [i];
///         if (delta === 0) nextIndexes.push(i);
///     }
///     for (let i = 0; i < nextIndexes.length; i++) {
///         const index = nextIndexes[i];
///         if (index !== sorted + i) {
///             [list[sorted + i], list[index]] = [list[index], list[sorted + i]];
///         }
///     }
///     if (nextIndexes.length > 1) this.prng.shuffle(list, sorted, sorted + nextIndexes.length);
///     sorted += nextIndexes.length;
/// }
/// ```
pub fn showdown_speed_sort(
    list: &mut [usize],
    comparator: impl Fn(usize, usize) -> std::cmp::Ordering,
) {
    use std::cmp::Ordering;
    let mut sorted = 0;
    let mut next: Vec<usize> = Vec::with_capacity(list.len());
    while sorted + 1 < list.len() {
        next.clear();
        next.push(sorted);
        for i in sorted + 1..list.len() {
            match comparator(list[next[0]], list[i]) {
                Ordering::Less => {}
                Ordering::Greater => {
                    next.clear();
                    next.push(i);
                }
                Ordering::Equal => next.push(i),
            }
        }
        for (k, index) in next.iter().enumerate() {
            if *index != sorted + k {
                list.swap(sorted + k, *index);
            }
        }
        sorted += next.len();
    }
}
