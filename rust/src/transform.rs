//! Transform and Imposter (IKA-219).
//!
//! Showdown a5df827 (the champions mod overrides neither). The move, data/moves.ts:
//!
//! ```text
//! transform: { accuracy: true, category: "Status", pp: 10, target: "normal",
//!     flags: { allyanim: 1, failencore: 1, noassist: 1, failcopycat: 1, failmimic: 1, failinstruct: 1 },
//!     onHit(target, pokemon) { return pokemon.transformInto(target); } }
//! ```
//!
//! No `protect` flag, so Protect does not stop it; no `bypasssub`, so a Substitute does
//! (`moves::apply_status_move_past_substitutes`). The ability, data/abilities.ts:
//!
//! ```text
//! imposter: { onSwitchIn(pokemon) {
//!     const target = pokemon.side.foe.active[pokemon.side.foe.active.length - 1 - pokemon.position];
//!     if (target) pokemon.transformInto(target, this.dex.abilities.get('imposter'));
//! } }
//! ```
//!
//! `onSwitchIn` runs in the switch-in's speed order with every other ability's `onStart`
//! (`fieldEvent('SwitchIn')`), which is where `resolve::switch_in_ability` calls it.
//!
//! `Pokemon#transformInto` (sim/pokemon.ts:1270) fails on a fainted target, an Illusion on
//! either side, a target behind a Substitute, a target already transformed, a user already
//! transformed (and Eternamax, a Terastallized Ogerpon/Terapagos and Stellar, none of which
//! this format can meet). Otherwise it copies the species, the types (`getTypes(true, true)`,
//! Roost's `typeWas`), the stored stats except HP, the moves at `Math.min(5, move.pp)` PP,
//! every boost, `timesAttacked`, Focus Energy / Dragon Cheer / Laser Focus, and the ability
//! last, by `setAbility(..., isTransform)`, whose `Start` runs unless the ability is the
//! one it already had. HP, the item, the status, the gender, the nature and the spread stay.
//! `clearVolatile` (a switch out, a faint) puts back `baseAbility`, `baseMoveSlots` (their
//! PP spent before the Transform included) and `setSpecies(baseSpecies)`: `revert`.

use std::rc::Rc;

use crate::battler::Battler;
use crate::position::{MoveSlot, Moves, Pokemon, TransformBase};
use crate::resolve::{Name, Turn};

type Slot = (usize, usize);

/// The crit volatiles `transformInto` removes from the user and copies from the target.
const CRIT_VOLATILES: [&str; 4] = ["dragoncheer", "focusenergy", "gmaxchistrike", "laserfocus"];

/// Why `transformInto` would return false for this user and target, if it would.
fn fails(user: &Pokemon, target: &Pokemon) -> Option<&'static str> {
    if target.fainted {
        return Some("the target fainted");
    }
    if target.has_volatile("substitute") {
        return Some("the target is behind a Substitute");
    }
    if target.transformed {
        return Some("the target is transformed");
    }
    if user.transformed {
        return Some("the user is transformed");
    }
    None
}

/// `pokemon.transformInto(target)`, `from` the move or the ability. `Failed` is its
/// `false`; otherwise whether the copied ability is a new one, whose `Start` the caller
/// runs. An error is a target whose stats cannot be built (a hidden spread).
pub(crate) fn transform_into(turn: &mut Turn, me: Slot, target: Slot, from: &str) -> Result<Transformed, String> {
    let reg = turn.reg;
    let (Some(user), Some(foe)) = (turn.mon_at(me.0, me.1), turn.mon_at(target.0, target.1)) else {
        return Ok(Transformed::Failed);
    };
    if let Some(why) = fails(user, foe) {
        log_event!(turn, "{} could not transform ({}: {})", Name(me.0, me.1), from, why);
        return Ok(Transformed::Failed);
    }
    // `this.illusion || pokemon.illusion`: Illusion is inert in this port and the position
    // does not say whether one is up, so a Zoroark is taken as showing itself (noted).
    let illusion = user.ability == "illusion" || foe.ability == "illusion";
    let stats = Battler::from_pokemon(reg, foe)?.stats;
    let types = turn.base_types_of(foe);
    let mut moves = Moves::default();
    for slot in foe.moves.iter() {
        let pp = reg
            .moves
            .get(slot.id.as_str())
            .and_then(|mv| mv.raw.get("pp").and_then(serde_json::Value::as_i64))
            .unwrap_or(5)
            .min(5);
        moves.push(MoveSlot { id: slot.id, pp, maxpp: pp, disabled: false, used: false });
    }
    let copied: Vec<_> = foe
        .volatiles
        .iter()
        .filter(|v| CRIT_VOLATILES.contains(&v.id.as_str()))
        .cloned()
        .collect();
    let (species, is_mega, boosts, times_attacked, ability) =
        (foe.species, foe.is_mega, foe.boosts, foe.times_attacked, foe.ability);
    if illusion {
        turn.report("transform: Illusion (assumed not up)");
    }

    let mon = turn.mon_at_mut(me.0, me.1).unwrap();
    let old_ability = mon.ability;
    mon.transform_base = Some(Box::new(TransformBase { ability: mon.ability, moves: mon.moves }));
    mon.transformed = true;
    mon.species = species;
    mon.is_mega = is_mega;
    mon.types = types;
    let mut override_stats = stats;
    override_stats[0] = mon.maxhp;
    mon.stats_override = Some(override_stats);
    mon.moves = moves;
    mon.boosts = boosts;
    mon.times_attacked = times_attacked;
    mon.volatiles.retain(|v| !CRIT_VOLATILES.contains(&v.id.as_str()));
    mon.volatiles.extend(copied);
    // `setAbility`: a fresh `abilityState` (Protean's flag, Supreme Overlord's count).
    mon.ability = ability;
    mon.ability_state = Rc::new(serde_json::Map::new());
    log_event!(turn, "{} transformed into {} ({})", Name(me.0, me.1), species, from);
    Ok(if old_ability == ability { Transformed::SameAbility } else { Transformed::NewAbility })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transformed {
    Failed,
    /// `setAbility` skips `Start` when the ability is the one it had (`isTransform`).
    SameAbility,
    NewAbility,
}

/// Imposter's `onSwitchIn`: the foe in the opposite position, `foe.active[length - 1 -
/// position]` -- across the field in a double battle, not the one in front.
pub(crate) fn imposter(turn: &mut Turn, side: usize, slot: usize) {
    match turn.mon_at(side, slot) {
        Some(mon) if !mon.fainted && mon.ability == "imposter" => {}
        _ => return,
    }
    let foe = 1 - side;
    let count = turn.pos.sides[foe].active.len();
    if slot >= count || turn.mon_at(foe, count - 1 - slot).is_none() {
        return;
    }
    if let Err(why) = transform_into(turn, (side, slot), (foe, count - 1 - slot), "imposter") {
        turn.report(format!("imposter: {why}"));
    }
}

/// Whether `mon` is the party member a queued switch names by `species`: by species or
/// base species, as before, but never a transformed Pokemon by the species it copied. A
/// Ditto that copied a foe's Garchomp is still a Ditto when its own Garchomp is called in
/// (found in diff_turn with a Ditto in every team: the switch found the Ditto, active,
/// and did nothing). A switch never brings in a transformed Pokemon: it is on the field.
pub(crate) fn switch_names(mon: &Pokemon, species: crate::id::Id) -> bool {
    (mon.species == species && !mon.transformed) || mon.base_species == species
}

/// `clearVolatile`'s share of a Transform, at a switch out or a faint: the ability and the
/// moves it had, the species it is. The types follow from the species (`restore_types`).
pub(crate) fn revert(mon: &mut Pokemon) {
    if !mon.transformed {
        return;
    }
    mon.transformed = false;
    mon.species = mon.base_species;
    mon.is_mega = false;
    mon.stats_override = None;
    if let Some(base) = mon.transform_base.take() {
        mon.ability = base.ability;
        mon.moves = base.moves;
    }
    mon.ability_state = Rc::new(serde_json::Map::new());
}
