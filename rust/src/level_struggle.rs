//! Seismic Toss and Night Shade (`damage: 'level'`), and Struggle's type and recoil (IKA-239).
//!
//! Both were answered wrong without a refusal. `damage` is a field the dump carries and no
//! code read, so the two level moves reached the calculator with a base power of 0 and hit for
//! nothing (noted `move.basePowerCallback`). `struggleRecoil` is a field the dump does not
//! carry, so `UNHANDLED_MOVE_FIELDS` never saw it, and Struggle's `onModifyMove` is a hook:
//! Struggle hit as a Normal move without recoil (noted `damaging move: struggle`).
//!
//! Showdown a5df827 (the champions mod overrides none of it):
//!
//! ```text
//! data/moves.ts
//!   nightshade:  basePower 0, damage: 'level', Special, Ghost
//!   seismictoss: basePower 0, damage: 'level', Physical, Fighting, contact
//!   struggle:    basePower 50, Physical, Normal, contact, target randomNormal, struggleRecoil: true,
//!                onModifyMove(move, pokemon, target) { move.type = '???'; ... }
//! sim/battle-actions.ts getDamage
//!   if (!target.runImmunity(move, !suppressMessages)) return false;
//!   if (move.ohko) ...; if (move.damageCallback) return ...;
//!   if (move.damage === 'level') return source.level;      // before crit, rolls, modifyDamage
//! sim/battle-actions.ts applyRecoilDamage (from hitStepMoveHitLoop `if (move.totalDamage)`,
//! and from Substitute's onTryPrimaryHit `if (damage)`)
//!   if (move.struggleRecoil) recoilDamage = clampIntRange(Math.round(pokemon.baseMaxhp / 4), 1);
//!   ... this.battle.directDamage(recoilDamage, pokemon, pokemon, { id: 'strugglerecoil' });
//! ```
//!
//! `directDamage` runs no `Damage` event, so neither Rock Head nor Magic Guard stops Struggle's
//! recoil. Struggle's own `ModifyMove` runs before the abilities' `ModifyType`
//! (useMoveInner: singleEvent ModifyMove, then runEvent ModifyType), so an -ate ability finds
//! `???` rather than Normal and neither retypes nor boosts it; Normalize lists `struggle` in
//! its `noModifyType`. A `???` move meets no immunity, no STAB and no resist berry.

use crate::battler::Battler;
use crate::id::Id;
use crate::reg::Move;
use crate::resolve::{Slot, Turn};

/// The moves whose damage is `damage: 'level'`. `tests/test_level_struggle_oracle.py` holds
/// this list to every move with a `damage` field in the regulation dumps.
pub(crate) fn deals_level(move_id: &str) -> bool {
    matches!(move_id, "nightshade" | "seismictoss")
}

/// The user's level, as `moveinfo::fixed_damage` answers for a level move.
pub(crate) fn level_damage(move_id: &str, attacker: &Battler) -> Option<i64> {
    deals_level(move_id).then_some(attacker.level)
}

/// Struggle's type and type-change boost: `???` and none, whatever the ability.
pub(crate) fn struggle_type(mv: &Move) -> Option<(Id, i64)> {
    (mv.id == "struggle").then(|| (Id::new("???"), 4096))
}

/// Whether a hit of this move never meets a resist berry: its damage skips `modifyDamage`
/// (a `damageCallback` or a level move), or its type is `???` (Struggle, so not even Chilan
/// Berry's Normal).
pub(crate) fn misses_resist_berry(move_id: &str) -> bool {
    crate::damage_callback::ported(move_id) || deals_level(move_id) || move_id == "struggle"
}

/// Struggle's recoil after a hit that dealt `dealt` (the move's total, or what a doll took):
/// a quarter of the user's max HP, rounded half up, at least 1, through nothing.
pub(crate) fn struggle_recoil(turn: &mut Turn, me: Slot, mv: &Move, dealt: i64) -> Result<(), String> {
    if mv.id != "struggle" || dealt <= 0 {
        return Ok(());
    }
    let Some(maxhp) = turn.mon_at(me.0, me.1).map(|mon| mon.maxhp) else { return Ok(()) };
    turn.deal_damage(me.0, me.1, ((maxhp + 2) / 4).max(1), false, "strugglerecoil")?;
    Ok(())
}
