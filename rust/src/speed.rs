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
pub fn effective_speed(mon: &Battler, field: &FieldState, side_conditions: &[Id]) -> i64 {
    let mut spe = mon.stat("spe", false);
    let mut chain = Chain::new();

    let has = |name: &str| side_conditions.iter().any(|c| c.as_str() == name);
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
    } else if mon.ability == "triage" && mv.has_flag("heal") {
        priority += 3;
    }
    if move_id == "grassyglide" && is(field.terrain, "grassyterrain") {
        priority += 1;
    }
    priority
}

/// Fractional priority outcomes as (value, probability).
pub fn fractional_priority(reg: &Reg, move_id: &str, mon: &Battler) -> Vec<(f64, f64)> {
    let Some(mv) = reg.moves.get(move_id) else {
        return vec![(0.0, 1.0)];
    };
    if mon.ability == "stall" {
        return vec![(-0.1, 1.0)];
    }
    if mon.ability == "myceliummight" && mv.category == "Status" {
        return vec![(-0.1, 1.0)];
    }
    if is(mon.item, "laggingtail") || is(mon.item, "fullincense") {
        return vec![(-0.1, 1.0)];
    }
    if mon.ability == "quickdraw" && mv.category != "Status" {
        return vec![(0.1, QUICK_DRAW_CHANCE), (0.0, 1.0 - QUICK_DRAW_CHANCE)];
    }
    if is(mon.item, "quickclaw") && mv.category != "Status" {
        return vec![(0.1, QUICK_CLAW_CHANCE), (0.0, 1.0 - QUICK_CLAW_CHANCE)];
    }
    vec![(0.0, 1.0)]
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    // Ascending order, descending priority, descending speed -- and stable, so the
    // canonical order of a tie is the queue's own order, as Python's lexsort gives.
    indices.sort_by(|a, b| {
        let (oa, pa, sa) = sort_key(&actions[*a], trick_room);
        let (ob, pb, sb) = sort_key(&actions[*b], trick_room);
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
