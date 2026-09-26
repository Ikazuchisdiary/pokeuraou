//! Semi-invulnerability: the turn a Fly, Bounce, Dig, Dive, Phantom Force or Shadow Force
//! user spends out of reach (IKA-241).
//!
//! Showdown a5df827 (the champions mod overrides none of it). The `twoturnmove` condition
//! adds the move's own volatile as it charges, and takes it off when it ends:
//!
//! ```text
//! data/conditions.ts twoturnmove
//!   onStart(attacker, defender, effect) { ... attacker.addVolatile(effect.id); ... },
//!   onEnd(target) { target.removeVolatile(this.effectState.move); },
//!   onMoveAborted(pokemon) { pokemon.removeVolatile('twoturnmove'); },
//! data/moves.ts (the move's own volatile; `onTryMove` removes it as the move fires)
//!   fly, bounce: onInvulnerability(target, source, move) {
//!       if (['gust', 'twister', 'skyuppercut', 'thunder', 'hurricane', 'smackdown',
//!            'thousandarrows'].includes(move.id)) return;
//!       return false; },
//!     onSourceModifyDamage / onSourceBasePower: gust, twister x2
//!   dig:  earthquake, magnitude reach it (x2); onImmunity: sandstorm, hail
//!   dive: surf, whirlpool reach it (x2);       onImmunity: sandstorm, hail
//!   phantomforce, shadowforce: onInvulnerability: false
//! data/abilities.ts noguard
//!   onAnyInvulnerabilityPriority: 1,
//!   onAnyInvulnerability(target, source, move) {
//!       if (move && (source === this.effectState.target || target === this.effectState.target)) return 0; },
//! sim/battle-actions.ts hitStepInvulnerabilityEvent (step 0 of trySpreadMoveHit, after
//! `move.spreadHit` is set from every target)
//!   if (move.id === 'helpinghand') return new Array(targets.length).fill(true);
//!   ... else if (gen >= 8 && move.id === 'toxic' && pokemon.hasType('Poison')) hitResults[i] = true;
//!   else hitResults[i] = this.battle.runEvent('Invulnerability', target, pokemon, move);
//!   if (hitResults[i] === false) { ... this.battle.add('-miss', pokemon, target); }
//! sim/pokemon.ts isSemiInvulnerable (Grassy Terrain's heal skips such a Pokemon)
//! ```
//!
//! The position carries all of it already: `twoturnmove` with its move (the bridge drops the
//! move's own volatile beside it, `resolve::showdown_volatiles`), and the port takes the marker
//! off as the move fires. Moves whose targets are a side or the field (`tryMoveHit`) run no
//! Invulnerability event. Not modelled, and named when they happen: the doubled damage of the
//! moves that reach a Fly, Bounce, Dig or Dive user, and Smack Down / Thousand Arrows pulling a
//! flier down. Lock-On is not modelled at all. Sky Drop is not a charging move here.

use crate::position::Pokemon;
use crate::reg::Move;
use crate::resolve::{Name, Slot, Turn};
use crate::speed::QueuedAction;

/// The charging moves that hide their user, and the moves that still reach it.
const OUT_OF_REACH: [(&str, &[&str]); 6] = [
    ("fly", &["gust", "twister", "skyuppercut", "thunder", "hurricane", "smackdown", "thousandarrows"]),
    ("bounce", &["gust", "twister", "skyuppercut", "thunder", "hurricane", "smackdown", "thousandarrows"]),
    ("dig", &["earthquake", "magnitude"]),
    ("dive", &["surf", "whirlpool"]),
    ("phantomforce", &[]),
    ("shadowforce", &[]),
];

/// What reaches a hidden Pokemon and does something the port does not: double damage, or
/// grounding a flier (`smackdown`'s `onHit` removes `fly` / `bounce` and cancels the move).
const REACHES_WITH_MORE: [(&str, &[&str]); 4] = [
    ("fly", &["gust", "twister", "smackdown", "thousandarrows"]),
    ("bounce", &["gust", "twister", "smackdown", "thousandarrows"]),
    ("dig", &["earthquake", "magnitude"]),
    ("dive", &["surf", "whirlpool"]),
];

/// The charging move that has `mon` out of reach, if any: `isSemiInvulnerable`.
pub(crate) fn hidden_by(mon: &Pokemon) -> Option<&'static str> {
    let charging = mon.volatile("twoturnmove")?.move_id?;
    OUT_OF_REACH.iter().find(|(id, _)| *id == charging.as_str()).map(|(id, _)| *id)
}

/// Whether `mv` from `source` misses `target` at the Invulnerability step.
pub(crate) fn dodges(turn: &Turn, source: Slot, mv: &Move, target: Slot) -> bool {
    if mv.id == "helpinghand" || target == source {
        return false;
    }
    let Some(mon) = turn.mon_at(target.0, target.1) else { return false };
    let Some(hidden) = hidden_by(mon) else { return false };
    let reach = OUT_OF_REACH.iter().find(|(id, _)| *id == hidden).map(|(_, r)| *r).unwrap_or(&[]);
    if reach.contains(&mv.id.as_str()) {
        return false;
    }
    let user = turn.mon_at(source.0, source.1);
    if mon.ability == "noguard" || user.is_some_and(|u| u.ability == "noguard") {
        return false;
    }
    if mv.id == "toxic" && user.is_some_and(|u| turn.types_of(u).contains("Poison")) {
        return false;
    }
    true
}

/// The note for a move that reaches a hidden Pokemon and does more there than the port does.
pub(crate) fn unmodelled_reach(turn: &Turn, mv: &Move, target: Slot) -> Option<String> {
    let mon = turn.mon_at(target.0, target.1)?;
    let hidden = hidden_by(mon)?;
    let (_, more) = REACHES_WITH_MORE.iter().find(|(id, _)| *id == hidden)?;
    more.contains(&mv.id.as_str()).then(|| format!("semi-invulnerable: {} into {}", mv.id, hidden))
}

/// The targets a move reaches past the Invulnerability step, logging each it misses; the
/// flag says whether any missed. Only for moves that target Pokemon (`trySpreadMoveHit`).
pub(crate) fn reachable(turn: &mut Turn, action: &QueuedAction, mv: &Move, targets: &[Slot]) -> (Vec<Slot>, bool) {
    if matches!(mv.target.as_str(), "all" | "foeSide" | "allySide" | "allyTeam") {
        return (targets.to_vec(), false);
    }
    let source = (action.side, action.slot);
    let mut kept = Vec::with_capacity(targets.len());
    let mut missed = false;
    for &target in targets {
        if dodges(turn, source, mv, target) {
            log_event!(turn, "{} missed {} (semi-invulnerable)", Name(source.0, source.1), Name(target.0, target.1));
            missed = true;
            continue;
        }
        if let Some(note) = unmodelled_reach(turn, mv, target) {
            turn.report(note);
        }
        kept.push(target);
    }
    (kept, missed)
}

/// `reachable` on a move's targets; `None`, with the move failed (`moveThisTurnResult` is
/// `false` when every target missed), when none is left.
pub(crate) fn reached_or_fail(turn: &mut Turn, action: &QueuedAction, mv: &Move, targets: &[Slot]) -> Option<Vec<Slot>> {
    let (kept, missed) = reachable(turn, action, mv, targets);
    if kept.is_empty() && missed {
        turn.move_failed[action.side][action.slot] = true;
        return None;
    }
    Some(kept)
}

/// `twoturnmove.onMoveAborted`: a charged Pokemon stopped before its second turn's move
/// (asleep, fully paralysed, confused, Imprison) drops the charge and is back in reach.
pub(crate) fn abort_charge(turn: &mut Turn, side: usize, slot: usize) {
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        if mon.has_volatile("twoturnmove") {
            mon.volatiles.retain(|v| v.id.as_str() != "twoturnmove");
        }
    }
}

/// Dig's and Dive's `onImmunity`: no sandstorm damage underground or underwater.
pub(crate) fn sheltered_from_sand(mon: &Pokemon) -> bool {
    matches!(hidden_by(mon), Some("dig" | "dive"))
}
