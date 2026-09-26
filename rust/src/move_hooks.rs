//! Damaging moves whose effect is a hook the port did not name (IKA-240): the screen breakers,
//! Poltergeist, and the berry eaters. Until IKA-213 they hit as plain moves with no note;
//! since then `modelled::damaging_move_is_unmodelled` listed them and `use_move` noted them.
//!
//! Showdown a5df827, data/moves.ts (the champions mod overrides none of them):
//!
//! ```text
//! brickbreak (1822) / psychicfangs (14060):
//!     onTryHit(pokemon) {
//!         // will shatter screens through sub, before you hit
//!         pokemon.side.removeSideCondition('reflect');
//!         pokemon.side.removeSideCondition('lightscreen');
//!         pokemon.side.removeSideCondition('auroraveil');
//!     },
//! poltergeist (13595):
//!     onTry(source, target) { return !!target.item; },
//!     onTryHit(target, source, move) { this.add('-activate', target, 'move: Poltergeist', ...); },
//! bugbite (1911) / pluck (13439):
//!     onHit(target, source, move) {
//!         const item = target.getItem();
//!         if (source.hp && item.isBerry && target.takeItem(source)) {
//!             this.add('-enditem', target, item.name, '[from] stealeat', ...);
//!             if (this.singleEvent('Eat', item, target.itemState, source, source, move)) {
//!                 this.runEvent('EatItem', source, source, move, item); ...
//!             }
//!             if (item.onEat) source.ateBerry = true;
//!         }
//!     },
//! ```
//!
//! Where each runs (sim/battle-actions.ts):
//!
//! * The move's own `onTryHit` is `spreadMoveHit`'s `singleEvent('TryHit', ...)` (1044), inside
//!   the hit loop: after Protect (`runEvent('TryHit')`, step 1), the type immunity (2) and the
//!   accuracy (4), and before the Substitute (`tryPrimaryHitEvent`) and `getDamage`. So a
//!   screen breaker that is protected against, misses, or meets a Ghost / Dark target breaks
//!   nothing, and one that connects breaks the target's side's screens before its own damage
//!   -- through a Substitute too.
//! * `onTry` is `trySpreadMoveHit`'s `singleEvent('Try', move, null, pokemon, targets[0])`
//!   (590), after `TryMove` and before any hit step: Poltergeist into a target with no item is
//!   `-fail` (a failure, `moveThisTurnResult` false) even against Protect. `target.item` is
//!   the raw field, so Klutz and Magic Room do not matter.
//! * `onHit` is `runMoveEffects` (step 3, 1086), after `spreadDamage` and before `DamagingHit`
//!   and the `Update` that lets a hit target eat its own Sitrus Berry (967): Bug Bite takes the
//!   berry first. A Substitute that took the hit runs no `onHit`. `takeItem` does not look at
//!   the holder's HP, so a berry is taken from a target the hit knocked out; Sticky Hold's
//!   `onTakeItem` gives way then (`!pokemon.hp`) and to Mold Breaker (`breakable`). A resist
//!   berry this hit met was eaten in `getDamage` (`onSourceModifyDamage`) and is gone.
//!   `singleEvent('Eat')` is not `eatItem`: Unnerve's `onFoeTryEatItem` is never asked.

use crate::battler::FieldState;
use crate::effects::{is_mold_breaker, resist_berry, SCREEN_CONDITIONS};
use crate::id::Id;
use crate::resolve::{Name, Slot, Turn};
use crate::speed::QueuedAction;

// The moves this file implements. Naming them here is what takes them out of
// `modelled::damaging_move_is_unmodelled` (`tools/port_coverage.py` reads this file).

fn breaks_screens(id: &str) -> bool {
    matches!(id, "brickbreak" | "psychicfangs")
}

fn eats_berry(id: &str) -> bool {
    matches!(id, "bugbite" | "pluck")
}

/// Poltergeist's `onTry`: the first target holds nothing.
pub(crate) fn fails_on_try(turn: &Turn, move_id: &str, targets: &[Slot]) -> bool {
    if move_id != "poltergeist" {
        return false;
    }
    let Some(first) = targets.first() else { return false };
    turn.mon_at(first.0, first.1).is_some_and(|mon| mon.item.is_none())
}

/// The field a screen breaker's damage is computed on: the target's side without its screens,
/// which its `onTryHit` removed before `getDamage`. Any other move's field is unchanged.
pub(crate) fn field_past_screens(mut field: FieldState, move_id: &str, target: Slot) -> FieldState {
    if breaks_screens(move_id) {
        field.side_conditions[target.0]
            .retain(|c| !SCREEN_CONDITIONS.iter().any(|(screen, _)| c.as_str() == *screen));
    }
    field
}

/// A screen breaker's `onTryHit`, for a hit that got past Protect, the immunity and the
/// accuracy: the target's side loses Reflect, Light Screen and Aurora Veil.
pub(crate) fn break_screens(turn: &mut Turn, move_id: &str, target: Slot) {
    if !breaks_screens(move_id) {
        return;
    }
    let conditions = &mut turn.pos.sides[target.0].side_conditions;
    let before = conditions.len();
    conditions.retain(|c| !SCREEN_CONDITIONS.iter().any(|(screen, _)| c.id.as_str() == *screen));
    if conditions.len() != before {
        log_event!(turn, "{} shattered the screens on side {}", move_id, target.0);
    }
}

/// Before a berry eater's damage: the target's berry, set aside so that `deal_damage` does
/// not let the target eat it at half HP -- in Showdown the `onHit` takes it before the
/// `Update` does. `steal_berry` puts it back or takes it.
pub(crate) fn set_berry_aside(turn: &mut Turn, move_id: &str, target: Slot) -> Option<Id> {
    if !eats_berry(move_id) {
        return None;
    }
    let mon = turn.mon_at_mut(target.0, target.1)?;
    let berry = mon.item.filter(|item| item.as_str().ends_with("berry"))?;
    mon.item = None;
    Some(berry)
}

/// Bug Bite and Pluck's `onHit`, with the berry `set_berry_aside` took. `type_mod` is the
/// hit's effectiveness, which decides whether a resist berry was eaten by the target first.
pub(crate) fn steal_berry(
    turn: &mut Turn,
    action: &QueuedAction,
    target: Slot,
    berry: Option<Id>,
    move_type: &str,
    type_mod: i64,
) {
    let Some(berry) = berry else { return };
    let me = (action.side, action.slot);
    if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
        mon.item = Some(berry);
    }
    let (target_hp, sticky) = match turn.mon_at(target.0, target.1) {
        Some(mon) => (mon.hp, mon.ability.as_str() == "stickyhold"),
        None => return,
    };
    let (user_hp, user_ability) = match turn.mon_at(me.0, me.1) {
        Some(mon) => (if mon.fainted { 0 } else { mon.hp }, mon.ability),
        None => (0, Id::new("")),
    };
    // The resist berry this hit met: eaten by the target in `getDamage`, which `after_hit`
    // does for the port.
    let resisted = resist_berry(berry.as_str())
        .is_some_and(|t| t == move_type && (t == "Normal" || type_mod > 0))
        && !turn.berries_blocked(target.0);
    let held = sticky && target_hp > 0 && !is_mold_breaker(user_ability.as_str());
    if user_hp <= 0 || resisted || held {
        // Not taken: the target eats its own pinch berry at the `Update`, as `deal_damage`
        // would have let it.
        turn.check_berry(target.0, target.1);
        return;
    }
    turn.consume_item(target.0, target.1, "stealeat");
    log_event!(turn, "{} ate {}'s {}", Name(me.0, me.1), Name(target.0, target.1), berry);
    eat(turn, me, berry.as_str());
}

/// The berry's `onEat` (data/items.ts) on the eater, for the berries whose own handlers the
/// port implements: Sitrus and Oran heal, Lum and Persim cure, a resist berry does nothing.
/// Any other berry is reported rather than named here: a name in this file would tell
/// `tools/port_coverage.py` that the port acts on the berry wherever it is held. The eater's
/// Ripen doubles the heal (`onTryHeal`, the berry being the effect) and its Cheek Pouch
/// heals after (`runEvent('EatItem', source, ...)`, IKA-329); Cud Chew is reported where
/// the port meets the ability, as it is for any ability `modelled::ability_is_modelled`
/// does not list.
fn eat(turn: &mut Turn, at: Slot, berry: &str) {
    match berry {
        "sitrusberry" => {
            let amount = turn.mon_at(at.0, at.1).map(|m| (m.maxhp / 4).max(1)).unwrap_or(0);
            turn.berry_heal(at.0, at.1, amount);
        }
        "oranberry" => {
            turn.berry_heal(at.0, at.1, 10);
        }
        "lumberry" | "persimberry" => {
            if let Some(mon) = turn.mon_at_mut(at.0, at.1) {
                mon.volatiles.retain(|v| v.id.as_str() != "confusion");
                if berry == "lumberry" && !mon.fainted {
                    mon.status = None;
                    mon.status_counter = None;
                }
            }
        }
        _ if resist_berry(berry).is_some() => {}
        _ => turn.report(format!("berry eaten by another: {berry}")),
    }
    turn.ate_berry(at.0, at.1);
}
