//! Move-specific mechanics, transcribed from `src/pokeuraou/moveinfo.py`.
//!
//! Variable base power, a move's own type change, and the two effectiveness overrides.
//! `base_power` returns `None` for a move it cannot determine, exactly as the Python does:
//! the caller reports that rather than printing a number with no basis.

use crate::battler::Battler;
use crate::position::Types;
use crate::reg::Reg;

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct MoveContext {
    pub weather: Option<String>,
    pub terrain: Option<String>,
    pub side_total_fainted: i64,
    pub times_attacked: i64,
    pub target_hurt_this_turn: bool,
    pub damaged_by_target: bool,
    pub previous_move_failed: bool,
    pub hit_index: i64,
    pub moving_last: bool,
    pub ally_used_same_move: bool,
}

const WEIGHT_BP: [(f64, i64); 5] =
    [(200.0, 120), (100.0, 100), (50.0, 80), (25.0, 60), (10.0, 40)];
const RELATIVE_WEIGHT_BP: [(f64, i64); 4] = [(5.0, 120), (4.0, 100), (3.0, 80), (2.0, 60)];

fn weather_ball_type(weather: Option<&str>) -> Option<&'static str> {
    match weather {
        Some("sunnyday") | Some("desolateland") => Some("Fire"),
        Some("raindance") | Some("primordialsea") => Some("Water"),
        Some("sandstorm") => Some("Rock"),
        Some("hail") | Some("snowscape") | Some("snow") => Some("Ice"),
        _ => None,
    }
}

fn terrain_pulse_type(terrain: Option<&str>) -> Option<&'static str> {
    match terrain {
        Some("electricterrain") => Some("Electric"),
        Some("grassyterrain") => Some("Grass"),
        Some("mistyterrain") => Some("Fairy"),
        Some("psychicterrain") => Some("Psychic"),
        _ => None,
    }
}

fn solar_weak(weather: Option<&str>) -> bool {
    matches!(
        weather,
        Some("raindance")
            | Some("primordialsea")
            | Some("sandstorm")
            | Some("hail")
            | Some("snowscape")
            | Some("snow")
    )
}

fn weight_bp(kg: f64) -> i64 {
    for (threshold, bp) in WEIGHT_BP {
        if kg >= threshold {
            return bp;
        }
    }
    20
}

fn relative_weight_bp(attacker_kg: f64, target_kg: f64) -> i64 {
    if target_kg <= 0.0 {
        return 120;
    }
    for (ratio, bp) in RELATIVE_WEIGHT_BP {
        if attacker_kg >= target_kg * ratio {
            return bp;
        }
    }
    40
}

pub fn fixed_damage(move_id: &str, attacker: &Battler, defender: &Battler) -> Option<i64> {
    match move_id {
        "superfang" | "naturesmadness" | "ruination" => Some((defender.hp / 2).max(1)),
        "finalgambit" => Some(attacker.hp.max(1)),
        "endeavor" => Some((defender.hp - attacker.hp).max(0)),
        _ => None,
    }
}

/// A move's own `onModifyType`, or None when the declared type stands.
pub fn effective_type(move_id: &str, attacker: &Battler, ctx: &MoveContext) -> Option<String> {
    match move_id {
        "weatherball" => weather_ball_type(ctx.weather.as_deref()).map(String::from),
        "terrainpulse" => terrain_pulse_type(ctx.terrain.as_deref()).map(String::from),
        "aurawheel" => {
            if attacker.species.as_str() == "morpekohangry" {
                Some("Dark".to_string())
            } else {
                None
            }
        }
        "ragingbull" => match attacker.species.as_str() {
            "taurospaldeacombat" => Some("Fighting".to_string()),
            "taurospaldeablaze" => Some("Fire".to_string()),
            "taurospaldeaaqua" => Some("Water".to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// A move's `onEffectiveness`, as (multiplier, Showdown's typeMod).
pub fn effectiveness_override(
    reg: &Reg,
    move_id: &str,
    move_type: &str,
    defender_types: &Types,
) -> Option<(f64, i64)> {
    let step = |mult: f64| -> i64 {
        if mult == 2.0 {
            1
        } else if mult == 0.5 {
            -1
        } else {
            0
        }
    };
    match move_id {
        "freezedry" => {
            let row = reg.typechart.get(move_type);
            let mut mult = 1.0;
            let mut steps = 0i64;
            for t in defender_types.as_slice() {
                if t.as_str() == "Water" {
                    mult *= 2.0;
                    steps += 1;
                    continue;
                }
                let value = row.and_then(|r| r.get(t.as_str())).copied().unwrap_or(1.0);
                mult *= value;
                steps += step(value);
            }
            Some((mult, steps.clamp(-6, 6)))
        }
        "flyingpress" => {
            let mut mult = 1.0;
            let mut steps = 0i64;
            for t in defender_types.as_slice() {
                for name in ["Fighting", "Flying"] {
                    let value = reg
                        .typechart
                        .get(name)
                        .and_then(|r| r.get(t.as_str()))
                        .copied()
                        .unwrap_or(1.0);
                    mult *= value;
                    steps += step(value);
                }
            }
            Some((mult, steps.clamp(-6, 6)))
        }
        _ => None,
    }
}

pub fn is_approximate(move_id: &str) -> bool {
    matches!(
        move_id,
        "beatup"
            | "ficklebeam"
            | "lashout"
            | "mistyexplosion"
            | "gravapple"
            | "shellsidearm"
            | "boltbeak"
            | "fishiousrend"
            | "punishment"
            | "trumpcard"
            | "wringout"
            | "crushgrip"
            | "spitup"
            | "naturalgift"
            | "fling"
            | "brine"
            | "retaliate"
            | "smellingsalts"
            | "wakeupslap"
            | "watershuriken"
    )
}

/// Base power for one hit, or None when this module cannot determine it.
pub fn base_power(
    reg: &Reg,
    move_id: &str,
    attacker: &Battler,
    defender: &Battler,
    ctx: &MoveContext,
) -> Option<i64> {
    let mv = reg.moves.get(move_id)?;
    let declared = mv.base_power;

    match move_id {
        "lowkick" | "grassknot" => {
            let kg = reg.species.get(defender.species.as_str()).map(|s| s.weightkg).unwrap_or(0.0);
            return Some(weight_bp(kg));
        }
        "heavyslam" | "heatcrash" => {
            let a = reg.species.get(attacker.species.as_str()).map(|s| s.weightkg).unwrap_or(0.0);
            let d = reg.species.get(defender.species.as_str()).map(|s| s.weightkg).unwrap_or(0.0);
            return Some(relative_weight_bp(a, d));
        }
        "electroball" => {
            let atk = attacker.stat("spe", false);
            let def = defender.stat("spe", false).max(1);
            let ratio = (atk / def).min(4) as usize;
            return Some([40, 60, 80, 120, 150][ratio]);
        }
        "gyroball" => {
            let atk = attacker.stat("spe", false).max(1);
            let power = (25 * defender.stat("spe", false)) / atk + 1;
            return Some(power.min(150));
        }
        "lastrespects" => return Some(50 + 50 * ctx.side_total_fainted),
        "ragefist" => return Some((50 + 50 * ctx.times_attacked).min(350)),
        "eruption" | "waterspout" => {
            let bp = (declared as f64 * attacker.hp as f64 / attacker.maxhp.max(1) as f64).trunc();
            return Some((bp as i64).max(1));
        }
        "hardpress" => {
            let bp = (100.0 * defender.hp as f64 / defender.maxhp.max(1) as f64).trunc();
            return Some((bp as i64).max(1));
        }
        "flail" | "reversal" => {
            let ratio =
                (48.0 * attacker.hp as f64 / attacker.maxhp.max(1) as f64).trunc() as i64;
            return Some(if ratio < 2 {
                200
            } else if ratio < 5 {
                150
            } else if ratio < 10 {
                100
            } else if ratio < 17 {
                80
            } else if ratio < 33 {
                40
            } else {
                20
            });
        }
        "acrobatics" => {
            return Some(if attacker.item.is_none() { declared * 2 } else { declared })
        }
        "assurance" => {
            return Some(if ctx.target_hurt_this_turn { declared * 2 } else { declared })
        }
        "avalanche" => {
            return Some(if ctx.damaged_by_target { declared * 2 } else { declared })
        }
        "storedpower" | "powertrip" => {
            let positive: i64 = attacker.boosts.iter().filter(|v| **v > 0).map(|v| *v as i64).sum();
            return Some(20 + 20 * positive);
        }
        "weatherball" => {
            let doubled = weather_ball_type(ctx.weather.as_deref()).is_some();
            return Some(if doubled { declared * 2 } else { declared });
        }
        "terrainpulse" => {
            let doubled = terrain_pulse_type(ctx.terrain.as_deref()).is_some();
            return Some(if doubled { declared * 2 } else { declared });
        }
        "risingvoltage" => {
            return Some(if ctx.terrain.as_deref() == Some("electricterrain") {
                declared * 2
            } else {
                declared
            })
        }
        "expandingforce" => return Some(declared),
        "tripleaxel" => return Some(20 * ctx.hit_index.max(1)),
        "triplekick" => return Some(10 * ctx.hit_index.max(1)),
        _ => {}
    }

    let conditional = match move_id {
        "stompingtantrum" | "temperflare" => Some(ctx.previous_move_failed),
        "payback" => Some(ctx.moving_last),
        "round" => Some(ctx.ally_used_same_move),
        _ => None,
    };
    if let Some(holds) = conditional {
        return Some(if holds { declared * 2 } else { declared });
    }

    if matches!(move_id, "hex" | "infernalparade" | "barbbarrage" | "venoshock") {
        let statused = if move_id == "venoshock" {
            matches!(defender.status.map(|s| s.as_str().to_string()).as_deref(), Some("psn") | Some("tox"))
        } else {
            defender.status.is_some()
        };
        return Some(if statused { declared * 2 } else { declared });
    }

    if declared > 0 {
        return Some(declared);
    }
    None
}

/// A move's own `onBasePower` chain entries, as (label, numerator, denominator).
pub fn base_power_modifiers(
    reg: &Reg,
    move_id: &str,
    attacker: &Battler,
    defender: &Battler,
    ctx: &MoveContext,
) -> Vec<(&'static str, f64, f64)> {
    if move_id == "facade"
        && attacker.status.is_some()
        && attacker.status.map(|s| s.as_str() != "slp").unwrap_or(false)
    {
        return vec![("facade", 2.0, 1.0)];
    }
    if move_id == "knockoff"
        && reg.item_is_removable(defender.species.as_str(), defender.item.map(|i| i.as_str().to_string()).as_deref())
    {
        return vec![("knockoff", 1.5, 1.0)];
    }
    if matches!(move_id, "solarbeam" | "solarblade") && solar_weak(ctx.weather.as_deref()) {
        return vec![("solar", 0.5, 1.0)];
    }
    if move_id == "expandingforce" && ctx.terrain.as_deref() == Some("psychicterrain") {
        return vec![("expandingforce", 1.5, 1.0)];
    }
    Vec::new()
}

/// Terrain's `onBasePower`, for a grounded user's matching type.
pub fn terrain_modifiers(
    move_type: &str,
    attacker_grounded: bool,
    terrain: Option<&str>,
) -> Vec<(&'static str, f64, f64)> {
    if !attacker_grounded {
        return Vec::new();
    }
    match (terrain, move_type) {
        (Some("electricterrain"), "Electric") => vec![("electricterrain", 5325.0, 4096.0)],
        (Some("grassyterrain"), "Grass") => vec![("grassyterrain", 5325.0, 4096.0)],
        (Some("psychicterrain"), "Psychic") => vec![("psychicterrain", 5325.0, 4096.0)],
        (Some("mistyterrain"), "Dragon") => vec![("mistyterrain", 2048.0, 4096.0)],
        _ => Vec::new(),
    }
}
