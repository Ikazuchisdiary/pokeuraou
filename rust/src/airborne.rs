//! Rules that turn on the user standing on the ground, or on a balloon holding a Pokemon
//! off it (IKA-205).
//!
//! Showdown a5df827, data/moves.ts and data/items.ts (the champions mod overrides none):
//!
//! ```text
//! expandingforce: {
//!     onBasePower(basePower, source) {
//!         if (this.field.isTerrain('psychicterrain') && source.isGrounded()) return this.chainModify(1.5);
//!     },
//!     onModifyMove(move, source, target) {
//!         if (this.field.isTerrain('psychicterrain') && source.isGrounded()) move.target = 'allAdjacentFoes';
//!     },
//!     target: "normal",
//! },
//! terrainpulse: {
//!     onModifyType(move, pokemon) { if (!pokemon.isGrounded()) return; switch (this.field.terrain) ... },
//!     onModifyMove(move, pokemon) { if (this.field.terrain && pokemon.isGrounded()) move.basePower *= 2; },
//! },
//! airballoon: {
//!     onDamagingHit(damage, target, source, move) {
//!         this.add('-enditem', target, 'Air Balloon'); target.item = ''; ...;
//!         this.runEvent('AfterUseItem', target, null, null, this.dex.items.get('airballoon'));
//!     },
//!     onAfterSubDamage(damage, target, source, effect) { if (effect.effectType === 'Move') { ...the same } },
//! },
//! ```
//!
//! The target Expanding Force takes is decided once, in `useMoveInner`'s `ModifyMove`, before
//! `getMoveTargets`: a spread move from there on, so it is not redirected, it takes the
//! spread 0.75 when two foes are left (`spreadHit`), and Wide Guard (`move.target ===
//! 'allAdjacentFoes'`) stops it even when only one is. The request still asks for a target
//! (`target: "normal"`), so the menu keeps one choice per target.

use crate::battler::Battler;
use crate::moveinfo::MoveContext;
use crate::reg::Move;
use crate::resolve::{grounded, Slot, Turn};
use crate::speed::QueuedAction;

/// Expanding Force's `onModifyMove`: Psychic Terrain and a grounded user.
pub(crate) fn expanding_force_spreads(turn: &Turn, action: &QueuedAction, mv: &Move) -> bool {
    if mv.id != "expandingforce" {
        return false;
    }
    if !matches!(turn.pos.field.terrain, Some(t) if t.as_str() == "psychicterrain") {
        return false;
    }
    turn.mon_at(action.side, action.slot).is_some_and(|mon| grounded(turn, mon))
}

/// The move's `target` after its own `onModifyMove`.
pub(crate) fn move_target<'m>(turn: &Turn, action: &QueuedAction, mv: &'m Move) -> &'m str {
    if expanding_force_spreads(turn, action, mv) {
        "allAdjacentFoes"
    } else {
        mv.target.as_str()
    }
}

/// The terrain a move's own handlers see when they ask `source.isGrounded()` too:
/// Expanding Force's 1.5 and Terrain Pulse's type and doubled power.
pub(crate) fn terrain_under<'c>(attacker: &Battler, ctx: &'c MoveContext) -> Option<&'c str> {
    if crate::damage::is_grounded(attacker) {
        ctx.terrain_name()
    } else {
        None
    }
}

/// Air Balloon's `onDamagingHit` / `onAfterSubDamage`: any damaging hit pops it, 0 damage
/// included, before the move's `onAfterHit` (Knock Off then finds nothing to take). It
/// is not `useItem`, but `AfterUseItem` runs, so Unburden starts -- `consume_item`'s.
pub(crate) fn pop_air_balloon(turn: &mut Turn, target: Slot) {
    let holds = turn
        .mon_at(target.0, target.1)
        .is_some_and(|mon| matches!(mon.item, Some(i) if i.as_str() == "airballoon"));
    if holds {
        turn.consume_item(target.0, target.1, "airballoon");
    }
}
