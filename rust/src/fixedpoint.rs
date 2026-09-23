//! Showdown's fixed-point modifier arithmetic, transcribed from `src/pokeuraou/fixedpoint.py`.
//!
//! The Python side is itself a transcription of `sim/battle.ts`; this is a second one, so
//! the rounding rules are copied literally rather than re-derived. `trunc` goes through
//! f64 because the Python does -- `np.trunc(...).astype(np.int64)` -- and a division that
//! rounds differently is worth one point of damage, which is worth a KO threshold.

pub const ONE: i64 = 4096;
const UINT32: i64 = 1 << 32;
const UINT16: i64 = 1 << 16;

/// `Dex#trunc`: floor to an integer, wrapped to 32 bits.
#[inline]
pub fn trunc(x: f64) -> i64 {
    (x.trunc() as i64).rem_euclid(UINT32)
}

/// The final truncation in `modifyDamage`, which can wrap a huge value to 0.
#[inline]
pub fn trunc16(x: i64) -> i64 {
    x.rem_euclid(UINT16)
}

/// `trunc(num * 4096 / den)` -- a ratio as a fixed-point modifier.
#[inline]
pub fn to_fp(num: f64, den: f64) -> i64 {
    (num * ONE as f64 / den) as i64
}

/// `modify(value, modifier / 4096)` for an integer value.
#[inline]
pub fn apply_fp(value: i64, modifier: i64) -> i64 {
    if modifier == ONE {
        value
    } else {
        (value * modifier + 2047) >> 12
    }
}

/// `Battle#modify`: a single modifier applied directly, with no chaining.
#[inline]
pub fn modify(value: i64, num: f64, den: f64) -> i64 {
    apply_fp(value, to_fp(num, den))
}

/// One event's modifier accumulator. Chaining rounds at each step, so order matters.
///
/// The label names the entry for a reader of the call site and is not kept: the list of
/// applied labels this used to push to was read by nothing, and it was an allocation on
/// every chain with an entry (IKA-101). A report that wants it back should collect it
/// behind the `profile` feature rather than on every hit.
#[derive(Debug, Clone)]
pub struct Chain {
    pub modifier: i64,
}

impl Chain {
    pub fn new() -> Self {
        Chain { modifier: ONE }
    }

    pub fn add(&mut self, num: f64, den: f64, label: &'static str) {
        self.add_fp(to_fp(num, den), label);
    }

    pub fn add_fp(&mut self, next: i64, _label: &'static str) {
        self.modifier = (self.modifier * next + 2048) >> 12;
    }

    #[inline]
    pub fn apply(&self, value: i64) -> i64 {
        apply_fp(value, self.modifier)
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.modifier == ONE
    }
}

/// Showdown's exact integer stat-stage ratios.
const BOOST_NUM: [i64; 13] = [2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8];
const BOOST_DEN: [i64; 13] = [8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2];

#[inline]
pub fn apply_boost(stat: i64, stage: i64) -> i64 {
    let idx = (stage.clamp(-6, 6) + 6) as usize;
    stat * BOOST_NUM[idx] / BOOST_DEN[idx]
}
