//! The Surges' terrain, the Seeds and Grassy Terrain's heal (IKA-201).
//!
//! Python's `_surge`, `_set_terrain`, `_use_terrain_seed` and `_grassy_terrain_heal` in
//! resolve.py, kept together so the resolver and the move layer each call one function.

use crate::id::Id;
use crate::resolve::{grounded, Turn};

/// The Surges' `onStart`: `this.field.setTerrain(...)`.
fn surge_terrain(ability: &str) -> Option<&'static str> {
    match ability {
        "electricsurge" => Some("electricterrain"),
        "grassysurge" => Some("grassyterrain"),
        "mistysurge" => Some("mistyterrain"),
        "psychicsurge" => Some("psychicterrain"),
        _ => None,
    }
}

/// The terrain each Seed is used under and the stat `useItem` raises.
fn seed(item: &str) -> Option<(&'static str, &'static str)> {
    match item {
        "electricseed" => Some(("electricterrain", "def")),
        "grassyseed" => Some(("grassyterrain", "def")),
        "mistyseed" => Some(("mistyterrain", "spd")),
        "psychicseed" => Some(("psychicterrain", "spd")),
        _ => None,
    }
}

/// `_surge`: the holder is the source, so a Terrain Extender makes it 8 turns.
pub(crate) fn surge(turn: &mut Turn, side: usize, slot: usize) {
    let Some(mon) = turn.mon_at(side, slot) else { return };
    let Some(terrain) = surge_terrain(mon.ability.as_str()) else { return };
    let extended = matches!(mon.item, Some(i) if i.as_str() == "terrainextender");
    set_terrain(turn, terrain, if extended { 8 } else { 5 });
}

/// `Field#setTerrain`: the same terrain again changes nothing; otherwise every active
/// Seed holder answers `TerrainChange` at once.
pub(crate) fn set_terrain(turn: &mut Turn, terrain: &str, duration: i64) -> bool {
    if matches!(turn.pos.field.terrain, Some(t) if t.as_str() == terrain) {
        return false;
    }
    turn.pos.field.terrain = Some(Id::new(terrain));
    turn.pos.field.terrain_duration = Some(duration);
    for side in 0..turn.pos.sides.len() {
        for slot in 0..turn.pos.sides[side].active.len() {
            use_terrain_seed(turn, side, slot);
        }
    }
    true
}

/// `_use_terrain_seed`: the boost from the holder itself, then the item goes.
pub(crate) fn use_terrain_seed(turn: &mut Turn, side: usize, slot: usize) {
    let Some(mon) = turn.mon_at(side, slot) else { return };
    if mon.fainted {
        return;
    }
    let Some((terrain, stat)) = mon.item.and_then(|i| seed(i.as_str())) else { return };
    if !matches!(turn.pos.field.terrain, Some(t) if t.as_str() == terrain) {
        return;
    }
    turn.apply_boosts(side, slot, &[(stat, 1)], false);
    turn.consume_item(side, slot);
}

/// `_grassy_terrain_heal`: `baseMaxhp / 16` for each grounded Pokemon, in residual order.
pub(crate) fn grassy_terrain_heal(turn: &mut Turn, order: &[(usize, usize)]) {
    if !matches!(turn.pos.field.terrain, Some(t) if t.as_str() == "grassyterrain") {
        return;
    }
    for &(side, slot) in order {
        let heals = match turn.mon_at(side, slot) {
            Some(mon) => !mon.fainted && grounded(turn, mon),
            None => false,
        };
        if heals {
            let amount = turn.fraction_of_max(side, slot, (1, 16));
            turn.heal(side, slot, amount);
        }
    }
}
