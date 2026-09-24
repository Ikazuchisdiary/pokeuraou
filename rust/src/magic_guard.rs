//! Magic Guard stops every damage that is not a move's (IKA-248).
//!
//! Showdown a5df827 (the champions mod does not override it):
//!
//! ```text
//! data/abilities.ts 2465
//!   magicguard: {
//!       onDamage(damage, target, source, effect) {
//!           if (effect.effectType !== 'Move') {
//!               if (effect.effectType === 'Ability') this.add('-activate', source, 'ability: ' + effect.name);
//!               return false;
//!           }
//!       },
//!       flags: {},
//! data/conditions.ts 193 (confusion's self-hit)
//!   const activeMove = { id: this.toID('confused'), effectType: 'Move', type: '???' };
//!   this.damage(damage, pokemon, pokemon, activeMove as ActiveMove);
//! sim/battle-actions.ts 1389 (Struggle's recoil)
//!   this.battle.directDamage(recoilDamage, pokemon, pokemon, { id: 'strugglerecoil' } as Condition);
//! ```
//!
//! Every non-move entry of `Turn::deal_damage` is `this.damage` with an ability, an item, a
//! condition or the `'recoil'` condition as its effect, so Magic Guard stops it: Rough Skin,
//! Iron Barbs, Rocky Helmet, Spiky Shield, Life Orb, recoil, Stealth Rock, Spikes, the weather,
//! Solar Power and Dry Skin, Leech Seed, burn and poison, binding and Salt Cure. Two are not:
//! confusion hits itself with an `effectType: 'Move'`, and Struggle's recoil is `directDamage`,
//! which runs no `Damage` event. The residual blocks in `moves::end_of_turn` already skipped the
//! holder for the weather, Leech Seed and the status and trap loop; this one gate covers the rest.
//! `flags: {}` -- Mold Breaker does not break it.

use crate::resolve::Turn;

/// The non-move damage that Magic Guard lets through: an `effectType: 'Move'` or `directDamage`.
fn passes_magic_guard(reason: &str) -> bool {
    matches!(reason, "confusion" | "strugglerecoil")
}

/// Whether Magic Guard stops this damage to the Pokemon at `(side, slot)`.
pub(crate) fn stops(turn: &Turn, side: usize, slot: usize, from_move: bool, reason: &str) -> bool {
    !from_move
        && !passes_magic_guard(reason)
        && turn.mon_at(side, slot).is_some_and(|mon| mon.ability == "magicguard")
}
