//! IKA-458 stage 3.1 `advanceTurn`: one turn's weights, the draw among them and the drawn
//! branch's position, in one request.
//!
//! `selfplay._advance_turn` asked for the turn's weights, drew an outcome with numpy's
//! `Generator.choice(n, p=...)`, and asked again for the drawn branch (the same position
//! and actions went over the pipe twice), and the next loop asked `needed` of the position it
//! got. Here the caller hands over the generator's PCG64 state, the port resolves the turn,
//! draws with the same arithmetic as numpy's `choice` (a normalised cumulative sum, one
//! `random()`, `searchsorted(side="right")`) and answers the new state, the index, the branch
//! position and the replacements it owes. A pause is answered by its index alone: the caller
//! resumes it with the request it always made.
//!
//! The draw is the one numpy takes, so the same seed plays the same game as the Python road.
//! A draw numpy would refuse (a negative or NaN weight, weights that do not sum to one) is a
//! `fault`: the caller then does the turn as before, with its generator untouched.

use crate::lp::sum_f64;
use crate::position::Position;
use crate::reg::Reg;
use serde_json::{json, Value};

/// The turns this process advanced, the positive control of the stage.
pub static ADVANCED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const MULTIPLIER: u128 = 47026247687942121848144207491837523525;

/// numpy's `PCG64` (`pcg64_next64`): the state steps first, and the output is the XSL-RR of
/// the new state. `has_uint32` and `uinteger` are numpy's own buffer of a 32-bit draw
/// (`pcg64_next32`: the low half is returned and the high half kept for the next call).
#[derive(Clone)]
pub(crate) struct Pcg64 {
    pub(crate) state: u128,
    pub(crate) inc: u128,
    pub(crate) has_uint32: bool,
    pub(crate) uinteger: u32,
}

impl Pcg64 {
    pub(crate) fn from_json(value: &Value) -> Option<Self> {
        let read = |key: &str| u128::from_str_radix(value.get(key)?.as_str()?, 16).ok();
        Some(Pcg64 {
            state: read("state")?,
            inc: read("inc")?,
            has_uint32: value.get("hasUint32").and_then(Value::as_bool).unwrap_or(false),
            uinteger: value.get("uinteger").and_then(Value::as_u64).unwrap_or(0) as u32,
        })
    }

    /// numpy's `default_rng(entropy)` for an entropy list (`SeedSequence`, then `PCG64`'s
    /// `pcg64_srandom_r` of the first two and the last two of the four 64-bit words).
    pub(crate) fn seeded(entropy: &[u64]) -> Pcg64 {
        let words = seed_sequence_state(entropy);
        let seed = ((words[0] as u128) << 64) | words[1] as u128;
        let seq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut rng = Pcg64 { state: 0, inc: (seq << 1) | 1, has_uint32: false, uinteger: 0 };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// `pcg64_next32`.
    fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// `Generator.random()`: 53 bits in [0, 1).
    pub(crate) fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// `Generator.integers(n)` for `1 <= n <= 2**32` (the 32-bit Lemire road numpy takes:
    /// `buffered_bounded_lemire_uint32`).
    pub(crate) fn integers(&mut self, n: u64) -> u64 {
        let rng = n - 1;
        if rng == 0 {
            return 0;
        }
        if rng == 0xFFFF_FFFF {
            return self.next_u32() as u64;
        }
        let rng = rng as u32;
        let excl = rng as u64 + 1;
        let mut m = self.next_u32() as u64 * excl;
        let mut leftover = m & 0xFFFF_FFFF;
        if leftover < excl {
            let threshold = (u32::MAX as u64 - rng as u64) % excl;
            while leftover < threshold {
                m = self.next_u32() as u64 * excl;
                leftover = m & 0xFFFF_FFFF;
            }
        }
        m >> 32
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({ "state": format!("{:x}", self.state), "inc": format!("{:x}", self.inc) })
    }

    /// With numpy's 32-bit buffer, which a caller that goes on drawing integers needs.
    pub(crate) fn to_json_full(&self) -> Value {
        json!({
            "state": format!("{:x}", self.state),
            "inc": format!("{:x}", self.inc),
            "hasUint32": self.has_uint32,
            "uinteger": self.uinteger,
        })
    }
}

/// numpy's `SeedSequence(entropy).generate_state(4, np.uint64)`: the entropy's 32-bit words
/// (little end first; 0 is one word) mixed into a pool of four, then four words made from it.
fn seed_sequence_state(entropy: &[u64]) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0d7e5;
    const MULT_A: u32 = 0x931e8875;
    const INIT_B: u32 = 0x8b51f9dd;
    const MULT_B: u32 = 0x58f38ded;
    const MIX_MULT_L: u32 = 0xca01f9dd;
    const MIX_MULT_R: u32 = 0x4973f715;
    const XSHIFT: u32 = 16;
    let mut words: Vec<u32> = Vec::new();
    for value in entropy {
        let mut n = *value;
        if n == 0 {
            words.push(0);
        }
        while n > 0 {
            words.push((n & 0xFFFF_FFFF) as u32);
            n >>= 32;
        }
    }
    let hashmix = |value: u32, hash_const: &mut u32| -> u32 {
        let value = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        let value = value.wrapping_mul(*hash_const);
        value ^ (value >> XSHIFT)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    };
    let mut hash_const = INIT_A;
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                let hashed = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], hashed);
            }
        }
    }
    for word in words.iter().skip(4) {
        for slot in pool.iter_mut() {
            let hashed = hashmix(*word, &mut hash_const);
            *slot = mix(*slot, hashed);
        }
    }
    let mut hash_const = INIT_B;
    let mut out32 = [0u32; 8];
    for (i, slot) in out32.iter_mut().enumerate() {
        let data = pool[i % 4] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        let data = data.wrapping_mul(hash_const);
        *slot = data ^ (data >> XSHIFT);
    }
    let mut out = [0u64; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = out32[2 * i] as u64 | ((out32[2 * i + 1] as u64) << 32);
    }
    out
}

/// numpy's `Generator.choice(len(counts), p=counts / counts.sum())`: the index, or why numpy
/// would have raised. The caller has checked that the total is positive.
pub(crate) fn choice(counts: &[f64], rng: &mut Pcg64) -> Result<usize, String> {
    let total = sum_f64(counts);
    let p: Vec<f64> = counts.iter().map(|c| c / total).collect();
    choose_p(&p, rng)
}

/// numpy's `Generator.choice(len(p), p=p)`: a normalised cumulative sum, one `random()` and
/// `searchsorted(side="right")`; the error numpy would have raised for a bad `p`.
pub(crate) fn choose_p(p: &[f64], rng: &mut Pcg64) -> Result<usize, String> {
    if p.iter().any(|v| !(*v >= 0.0)) {
        return Err("a weight is negative or not a number".into());
    }
    // numpy's tolerance on the probabilities' sum (`sqrt(finfo(double).eps)`).
    if (p.iter().sum::<f64>() - 1.0).abs() > 1.4901161193847656e-8 {
        return Err("the weights do not sum to one".into());
    }
    let mut cdf = Vec::with_capacity(p.len());
    let mut run = 0.0;
    for v in p {
        run += v;
        cdf.push(run);
    }
    let last = *cdf.last().ok_or("no weights to draw from")?;
    for c in &mut cdf {
        *c /= last;
    }
    let u = rng.next_double();
    Ok(cdf.partition_point(|c| *c <= u).min(cdf.len() - 1))
}

pub fn advance_turn(reg: &Reg, encoder: &crate::encode::Encoder, value: &Value) -> Value {
    let Some(mut rng) = Pcg64::from_json(&value["rng"]) else {
        return json!({ "fault": "advanceTurn without a PCG64 state" });
    };
    // IKA-458 stage 3.2: with `selfSwitch` a drawn pause is carried on here, node by node,
    // to the turn's end (`selfswitch.rs`).
    let context = match value.get("selfSwitch").filter(|c| !c.is_null()) {
        None => None,
        Some(c) => match crate::selfswitch::Context::from_json(c) {
            Ok(context) => Some(context),
            Err(why) => return json!({ "fault": why }),
        },
    };
    let result = match crate::node::resolve_raw(reg, value) {
        Err(refused) => return refused,
        Ok(result) => result,
    };
    let advanced = match advance_core(reg, encoder, result, context.as_ref(), &mut rng) {
        Err(answer) => return answer,
        Ok(advanced) => advanced,
    };
    let mut answer = json!({
        "branches": advanced.weights,
        "suspended": advanced.paused,
        "exact": advanced.exact,
        "unmodelled": advanced.unmodelled,
        "index": advanced.index,
        "position": Value::Null,
    });
    answer["rng"] = rng.to_json();
    if let Some(landed) = &advanced.position {
        answer["needed"] = json!(crate::resolve::commands::replacements_needed(landed));
        answer["position"] = landed.to_json();
    }
    if let Some(chain) = &advanced.chain {
        answer["chain"] = chain.to_json();
    }
    answer
}

/// What a turn did once its outcome was drawn (`advance_core`).
pub(crate) struct Advanced {
    pub(crate) weights: Vec<f64>,
    pub(crate) paused: Vec<f64>,
    pub(crate) exact: bool,
    pub(crate) unmodelled: Vec<String>,
    /// The drawn outcome (None: nothing to draw from, no draw was made).
    pub(crate) index: Option<usize>,
    /// The position the turn ended in (None: no outcome to go on from).
    pub(crate) position: Option<Position>,
    /// The self-switch nodes a drawn pause played on the way.
    pub(crate) chain: Option<crate::selfswitch::Chain>,
}

/// `advanceTurn` after the turn is resolved: the draw among the branches and pauses with numpy's
/// own arithmetic, the drawn branch's position, or the drawn pause carried on to the turn's end
/// when ``context`` says where the leaves are. Err is the answer to hand back (a fault, a refusal).
pub(crate) fn advance_core(
    reg: &Reg,
    encoder: &crate::encode::Encoder,
    mut result: crate::resolve::TurnResult<'_>,
    context: Option<&crate::selfswitch::Context>,
    rng: &mut Pcg64,
) -> Result<Advanced, Value> {
    let weights: Vec<f64> = result.branches.iter().map(|b| b.probability).collect();
    let paused: Vec<f64> = result.suspended.iter().map(|s| s.probability).collect();
    let branches = weights.len();
    let counts: Vec<f64> = weights.iter().chain(paused.iter()).copied().collect();
    let mut advanced = Advanced {
        exact: result.exact,
        unmodelled: result.unmodelled.iter().cloned().collect(),
        weights,
        paused,
        index: None,
        position: None,
        chain: None,
    };
    // Nothing to draw from: the caller ends the game, as it did, without a draw.
    if counts.is_empty() || sum_f64(&counts) <= 0.0 {
        return Ok(advanced);
    }
    let index = choice(&counts, rng).map_err(|why| json!({ "fault": why }))?;
    ADVANCED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    advanced.index = Some(index);
    if index < branches {
        advanced.position = Some(result.branches[index].position.clone());
    } else if let Some(context) = context {
        let first = result.suspended.swap_remove(index - branches);
        match crate::selfswitch::run(reg, encoder, context, first, rng) {
            Err(crate::selfswitch::Failure::Fault(why)) => return Err(json!({ "fault": why })),
            Err(crate::selfswitch::Failure::Answer(refusal)) => return Err(refusal),
            Ok(chain) => {
                advanced.position = chain.position.clone();
                advanced.chain = Some(chain);
            }
        }
    }
    Ok(advanced)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy 2.x, `default_rng(12345)`: the first three `random()`s, then the eight
    /// `choice(4, p=[.1, .2, .3, .4])` of a fresh generator and the state after them.
    #[test]
    fn the_generator_is_numpys() {
        let fresh = || Pcg64 {
            state: 0x1905e0335aae96349199b0d09775add5,
            inc: 0xc9c7353e6e2b1f287d761f2d4027fae7,
            has_uint32: false,
            uinteger: 0,
        };
        let mut rng = fresh();
        let got: Vec<f64> = (0..3).map(|_| rng.next_double()).collect();
        assert_eq!(got, [0.22733602246716966, 0.31675833970975287, 0.7973654573327341]);
        let mut rng = fresh();
        let drawn: Vec<usize> =
            (0..8).map(|_| choice(&[0.1, 0.2, 0.3, 0.4], &mut rng).unwrap()).collect();
        assert_eq!(drawn, [1, 2, 3, 3, 2, 2, 2, 1]);
        assert_eq!(rng.state, 216098700122296263478885786697608317821);
    }

    #[test]
    fn a_draw_numpy_would_refuse_is_an_error() {
        let mut rng = Pcg64 { state: 1, inc: 3, has_uint32: false, uinteger: 0 };
        assert!(choice(&[0.5, f64::NAN], &mut rng).is_err());
        assert!(choice(&[1.0, -0.5, 0.5], &mut rng).is_err());
    }

    /// numpy's `default_rng(entropy)` states, and `integers(7260)`'s draws with the 32-bit
    /// buffer left behind (IKA-458 stage 3.5).
    #[test]
    fn a_seed_is_numpys() {
        let of = |entropy: &[u64]| {
            let rng = Pcg64::seeded(entropy);
            (format!("{:x}", rng.state), format!("{:x}", rng.inc))
        };
        let want = |a: &str, b: &str| (a.to_string(), b.to_string());
        assert_eq!(
            of(&[12345]),
            want("1905e0335aae96349199b0d09775add5", "c9c7353e6e2b1f287d761f2d4027fae7")
        );
        assert_eq!(
            of(&[5, 17]),
            want("784a2264511b082818cb8eb6dbe595c", "ad30a2f9f2e28d7748c4cba6b1deef5d")
        );
        assert_eq!(
            of(&[45801, 123456]),
            want("2fba315d0aea08d9f22d7fa7eb4cb3a5", "d39a1bcab3742beee260dc8f040e220d")
        );
        // A word past 32 bits and a zero.
        assert_eq!(
            of(&[(1u64 << 40) + 3, 0]),
            want("46b9e1f7d584d9b4b5ec56f67c9fed54", "cf4ef350c4a7a5226b829e3da8ec123f")
        );
        let mut rng = Pcg64::seeded(&[45801, 123456]);
        let drawn: Vec<u64> = (0..3).map(|_| rng.integers(7260)).collect();
        assert_eq!(drawn, [5949, 5768, 2856]);
        assert_eq!(rng.state, 152734038657690699424913589981853995307);
        assert!(rng.has_uint32 && rng.uinteger == 3461578269);
        let mut rng = Pcg64::seeded(&[45801, 7]);
        let drawn: Vec<u64> = (0..5).map(|_| rng.integers(7260)).collect();
        assert_eq!(drawn, [1652, 2312, 5898, 1166, 6564]);
        assert_eq!(rng.next_double(), 0.059068876037850604);
        assert_eq!(rng.state, 10283358025631968964606158021676308445);
        assert!(rng.has_uint32 && rng.uinteger == 659541193);
    }
}
