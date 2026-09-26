//! Three small rules the port had wrong without a note (IKA-325, IKA-326, IKA-328), each a
//! function its one call site asks, kept out of moves.rs so the parallel work there merges.
//!
//! * `disable_stops`: Disable's `onBeforeMove` (IKA-325). The port kept the volatile and
//!   took the move off the next menu, but a move already on its way went on: a charging
//!   move's second turn (fired whatever the menu says) and a move chosen before a faster
//!   Disable landed.
//! * `resist_berry_eaten`: a resist berry reads the hit's type after the skins and the
//!   moves' own type changes (IKA-326). It read the move's declared type, so a Pixilate
//!   Quick Attack left a Roseli Berry uneaten (and a Chilan Berry eaten) while the damage
//!   was halved (or not) by the calculator's type.
//! * `heatproof_burn`: Heatproof halves the burn's residual damage (IKA-328).

use crate::effects::resist_berry;
use crate::reg::Move;
use crate::resolve::Turn;
use crate::speed::QueuedAction;

/// Disable's `onBeforeMove` (priority 7: below flinch's 8, above Taunt's 5), with the
/// champions mod's `cantusetwice` exception:
///
/// ```text
/// // data/moves.ts, disable.condition (Showdown a5df827)
/// onBeforeMovePriority: 7,
/// // data/mods/champions/moves.ts, disable.condition
/// onBeforeMove(attacker, defender, move) {
///     if (!(move.isZ && move.isZOrMaxPowered) && move.id === this.effectState.move && !move.flags['cantusetwice']) {
///         this.add('cant', attacker, 'Disable', move);
///         return false;
///     }
/// },
/// ```
///
/// `onDisableMove` shapes the next request (`MoveSlot.disabled`); this stops what the
/// request did not: a charging move's second turn, which `twoturnmove`'s `onLockMove` fires
/// whatever is disabled, and a move chosen before a faster Disable landed. The volatile's
/// `effectState.move` is the port's `Effect::move_id` (`apply_disable`). No PP is spent
/// and `twoturnmove` goes at the `MoveAborted` its `onMoveAborted` answers, as for every
/// other `BeforeMove` stop.
pub(crate) fn disable_stops(turn: &Turn, action: &QueuedAction, mv: &Move) -> bool {
    let Some(mon) = turn.mon_at(action.side, action.slot) else { return false };
    let disabled = mon.volatile("disable").and_then(|held| held.move_id);
    if !matches!(disabled, Some(id) if id.as_str() == mv.id.as_str()) {
        return false;
    }
    let cant_use_twice = mv
        .raw
        .get("flags")
        .and_then(|flags| flags.get("cantusetwice"))
        .and_then(serde_json::Value::as_i64)
        .is_some_and(|flag| flag != 0);
    !cant_use_twice
}

/// Whether a resist berry held by the target is eaten by this hit: the berry's
/// `onSourceModifyDamage` reads `move.type`, the active move's type after `ModifyType`
/// (Pixilate, Aerilate, Refrigerate, Galvanize, Normalize, Liquid Voice, and Weather Ball,
/// Terrain Pulse and the rest of the moves' own), which is `hit_type`, the type the
/// calculator used for this hit:
///
/// ```text
/// // data/items.ts, roseliberry (every resist berry but Chilan has this shape)
/// onSourceModifyDamage(damage, source, target, move) {
///     if (move.type === 'Fairy' && target.getMoveHitData(move).typeMod > 0) { ... target.eatItem() ...
/// // chilanberry
///     if (move.type === 'Normal' && (!target.volatiles['substitute'] || ...)) { ... target.eatItem() ...
/// ```
pub(crate) fn resist_berry_eaten(item: &str, hit_type: &str, type_mod: i64) -> bool {
    match resist_berry(item) {
        None => false,
        Some(berry_type) => berry_type == hit_type && (berry_type == "Normal" || type_mod > 0),
    }
}

/// The burn's residual damage on this Pokemon: a sixteenth (`fraction_of_max`), halved by
/// Heatproof.
///
/// ```text
/// // data/conditions.ts, brn
/// onResidual(pokemon) { this.damage(pokemon.baseMaxhp / 16); },
/// // data/abilities.ts, heatproof (no champions override)
/// onDamage(damage, target, source, effect) {
///     if (effect && effect.id === 'brn') { return damage / 2; }
/// },
/// ```
///
/// `spreadDamage` clamps to at least 1 (flooring) before the `Damage` event and again after
/// it, so a sixteenth of 1 or 2 stays 1 under Heatproof. `breakable` does not reach a
/// residual: no Mold Breaker is acting.
pub(crate) fn burn_damage(turn: &Turn, side: usize, slot: usize, sixteenth: i64) -> i64 {
    let heatproof = turn.mon_at(side, slot).is_some_and(|mon| mon.ability == "heatproof");
    if heatproof {
        (sixteenth / 2).max(1)
    } else {
        sixteenth
    }
}
