//! Steel Beam's (and Mind Blown's) half-HP recoil, `mindBlownRecoil` (IKA-406).
//!
//! Showdown a5df827 (the champions mod overrides neither the move nor `applyRecoilDamage`):
//!
//! ```text
//! data/moves.ts
//!   steelbeam: { accuracy: 95, basePower: 140, category: "Special", target: "normal",
//!       flags: { protect: 1, mirror: 1 }, mindBlownRecoil: true,
//!       onMoveFail(target, source, move) {
//!           if (move.multihit) return;
//!           this.damage(Math.round(source.maxhp / 2), source, source, this.dex.conditions.get('Steel Beam'));
//!       }, ... }
//! sim/battle-actions.ts applyRecoilDamage (1379-1395)
//!   else if (move.mindBlownRecoil || move.chloroblastRecoil) recoilDamage = Math.round(pokemon.maxhp / 2);
//!   ...
//!   const effect = move.mindBlownRecoil ? this.dex.conditions.get(move.name) : 'recoil';
//!   this.battle.damage(recoilDamage, pokemon, pokemon, effect);
//! sim/battle-actions.ts useMoveInner (526)
//!   if (!moveResult) { ...this.battle.singleEvent('MoveFail', move, null, target, pokemon, move); ... }
//! data/abilities.ts
//!   magicguard: onDamage(...) { if (effect.effectType !== 'Move') { ...return false; } }
//! sim/dex-conditions.ts 650
//!   this.effectType = (['Weather', 'Status', 'Terrain'].includes(data.effectType) ? data.effectType : 'Condition');
//!   rockhead:   onDamage(...) { if (effect.id === 'recoil') { ... } }
//! ```
//!
//! Two paths, one amount. A move that dealt damage (`applyRecoilDamage` from
//! `hitStepMoveHitLoop`, and from Substitute's `onTryPrimaryHit` when the doll took something)
//! costs the user `Math.round(maxhp / 2)` -- half, rounded up -- whatever the move dealt. A
//! move that failed (`trySpreadMoveHit` false: a miss, Protect, an immunity, a semi-invulnerable
//! target) costs the same through `onMoveFail`, unless it is a multi-hit move. A move that
//! never reached `trySpreadMoveHit` (no target, `TryMove`) costs nothing. The effect is the
//! `dex.conditions.get('Steel Beam')`, whose `effectType` is `'Condition'` (sim/dex-conditions.ts
//! 650: only Weather, Status and Terrain keep theirs), not `'Move'`: Magic Guard stops it (checked
//! against Showdown, not read off the `'Move'` a first reading of the move suggests), Rock Head
//! (which tests the id `recoil`) does not, and `from_move` is false because Endure, Focus Sash and
//! Sturdy test `effectType === 'Move'`.

use crate::resolve::Turn;

type Slot = (usize, usize);

pub(crate) fn has_field(mv: &crate::reg::Move) -> bool {
    mv.raw.get("mindBlownRecoil").and_then(serde_json::Value::as_bool) == Some(true)
}

fn hurt(turn: &mut Turn, me: Slot) -> Result<(), String> {
    let Some(maxhp) = turn.mon_at(me.0, me.1).map(|mon| mon.maxhp) else { return Ok(()) };
    turn.deal_damage(me.0, me.1, (maxhp + 1) / 2, false,"mindblownrecoil")?;
    Ok(())
}

/// The recoil after a hit that dealt `dealt` (the move's total, or what a doll took).
pub(crate) fn recoil_after_hit(turn: &mut Turn, me: Slot, mv: &crate::reg::Move, dealt: i64) -> Result<(), String> {
    if !has_field(mv) || dealt <= 0 {
        return Ok(());
    }
    hurt(turn, me)
}

/// `onMoveFail`: the move reached `trySpreadMoveHit` and nothing came of it.
pub(crate) fn recoil_on_fail(turn: &mut Turn, me: Slot, mv: &crate::reg::Move) -> Result<(), String> {
    if !has_field(mv) || mv.raw.get("multihit").is_some_and(|v| !v.is_null()) {
        return Ok(());
    }
    hurt(turn, me)
}
