//! End-of-turn abilities the port does not apply, noted where Showdown fires them (IKA-317).
//!
//! Each of these is an `onResidual` (or a residual damage or heal) in Showdown a5df827's
//! data/abilities.ts that the port ignores. Their only note used to be the one every hit
//! carries for an ability the port ignores (`damage::unmodelled_effects`), so a turn whose
//! holder used status moves and was not hit missed the effect and said nothing -- Moody's
//! shape (IKA-308). Naming them here makes them `modelled.rs`'s (the hit note goes), so every
//! effect each has is noted here instead, on a turn it can fire:
//!
//! ```text
//! order 5.3  hydration   status && rain            -> cureStatus()
//! order 5.3  shedskin    status, 33%               -> cureStatus()
//! order 9/10 poisonheal  psn/tox damage            -> heal 1/8 instead
//! order 28.2 harvest     no item, lastItem a berry -> the berry back (sun, or 50%)
//! order 28.2 cudchew     a berry eaten last turn   -> eaten again
//! order 28.2 pickup      no item, a foe or ally used one this turn -> takes it
//! order 29   hungerswitch Morpeko                  -> Full Belly <-> Hangry every turn
//! order 29   opportunist boosts copied from foes (also after every move and switch-in)
//! weather    forecast    Castform's form follows the weather (and reverts when it ends)
//! drains     liquidooze  Leech Seed, a drain move, Strength Sap hurt instead of healing
//! ```
//!
//! The notes are in the turn's union over branches, so one said at the residual covers the
//! whole turn -- an Opportunist copy after a foe's Swords Dance included -- on any branch its
//! holder is still there for. What the port cannot see (Showdown's `lastItem`,
//! `usedItemThisTurn`, a Cud Chew berry eaten inside the port's turn) is approximated from the
//! set's item (`base_item`) and the held one, erring towards a note.

use crate::resolve::{Slot, Turn};

fn live<'t>(turn: &'t Turn, side: usize, slot: usize) -> Option<&'t crate::position::Pokemon> {
    turn.mon_at(side, slot).filter(|mon| !mon.fainted && mon.hp > 0)
}

fn lost_a_berry(mon: &crate::position::Pokemon) -> bool {
    mon.item.is_none() && mon.base_item.is_some_and(|item| item.as_str().ends_with("berry"))
}

/// Residual order 5 to 10, before the status damage: the status curers and Poison Heal. Said
/// before the port's poison damage, which a Poison Heal holder may faint to.
pub(crate) fn before_status_damage(turn: &mut Turn, order: &[Slot]) {
    let rain = matches!(turn.pos.field.weather, Some(w)
        if matches!(w.as_str(), "raindance" | "primordialsea"));
    let mut notes: Vec<&'static str> = Vec::new();
    for (side, slot) in order.iter().copied() {
        let Some(mon) = live(turn, side, slot) else { continue };
        let Some(status) = mon.status else { continue };
        match mon.ability.as_str() {
            "hydration" if rain => notes.push("ability: hydration (status cured in rain, not applied)"),
            "shedskin" => notes.push("ability: shedskin (end-of-turn 33% status cure not applied)"),
            "poisonheal" if matches!(status.as_str(), "psn" | "tox") => {
                notes.push("ability: poisonheal (poison heals 1/8 instead of hurting, not applied)")
            }
            _ => {}
        }
    }
    for note in notes {
        turn.report(note);
    }
}

/// Residual order 28 and 29, and Forecast: after Speed Boost, so a foe's boost from it is one
/// Opportunist would copy. `weather_seen` is whether a weather was up when the phase began
/// (it may have ended in it, which is a form change too).
pub(crate) fn late_residual(turn: &mut Turn, order: &[Slot], weather_seen: bool) {
    let mut notes: Vec<&'static str> = Vec::new();
    for (side, slot) in order.iter().copied() {
        let Some(mon) = live(turn, side, slot) else { continue };
        match mon.ability.as_str() {
            "harvest" if lost_a_berry(mon) => {
                notes.push("ability: harvest (end-of-turn berry restore not applied)")
            }
            "cudchew" if lost_a_berry(mon) || mon.ability_state.get("counter").is_some() => {
                notes.push("ability: cudchew (berry eaten again not applied)")
            }
            "pickup" if mon.item.is_none() => {
                let used = order.iter().copied().any(|(s, t)| {
                    (s, t) != (side, slot)
                        && turn
                            .mon_at(s, t)
                            .is_some_and(|other| other.item.is_none() && other.base_item.is_some())
                });
                if used {
                    notes.push("ability: pickup (end-of-turn item pickup not applied)");
                }
            }
            "hungerswitch" if mon.species.as_str().starts_with("morpeko") => {
                notes.push("ability: hungerswitch (end-of-turn form change not applied)")
            }
            "opportunist" => {
                let foe_boosted = (0..turn.pos.sides[1 - side].active.len()).any(|t| {
                    turn.mon_at(1 - side, t)
                        .is_some_and(|foe| !foe.fainted && foe.boosts.iter().any(|&b| b > 0))
                });
                if foe_boosted {
                    notes.push("ability: opportunist (copying foes' boosts not applied)");
                }
            }
            "forecast" if weather_seen && mon.species.as_str().starts_with("castform") => {
                notes.push("ability: forecast (form change with the weather not applied)")
            }
            _ => {}
        }
    }
    for note in notes {
        turn.report(note);
    }
}

/// Liquid Ooze's `onSourceTryHeal`: a drain from its holder (`drain`, `leechseed`,
/// `strengthsap`) hurts the drainer instead. Called where the port heals from a drain.
pub(crate) fn drained_from(turn: &mut Turn, holder: Slot) {
    if turn.mon_at(holder.0, holder.1).is_some_and(|mon| mon.ability == "liquidooze") {
        turn.report("ability: liquidooze (drain hurts instead of healing, not applied)");
    }
}
