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

use crate::lp::numpy_sum;
use crate::reg::Reg;
use serde_json::{json, Value};

/// The turns this process advanced, the positive control of the stage.
pub static ADVANCED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const MULTIPLIER: u128 = 47026247687942121848144207491837523525;

/// numpy's `PCG64` (`pcg64_next64`): the state steps first, and the output is the XSL-RR of
/// the new state.
struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    fn from_json(value: &Value) -> Option<Self> {
        let read = |key: &str| u128::from_str_radix(value.get(key)?.as_str()?, 16).ok();
        Some(Pcg64 { state: read("state")?, inc: read("inc")? })
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// `Generator.random()`: 53 bits in [0, 1).
    fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn to_json(&self) -> Value {
        json!({ "state": format!("{:x}", self.state), "inc": format!("{:x}", self.inc) })
    }
}

/// numpy's `Generator.choice(len(counts), p=counts / counts.sum())`: the index, or why numpy
/// would have raised. The caller has checked that the total is positive.
fn choice(counts: &[f64], rng: &mut Pcg64) -> Result<usize, String> {
    let total = numpy_sum(counts);
    let p: Vec<f64> = counts.iter().map(|c| c / total).collect();
    if p.iter().any(|v| !(*v >= 0.0)) {
        return Err("a weight is negative or not a number".into());
    }
    // numpy's tolerance on the probabilities' sum (`sqrt(finfo(double).eps)`).
    if (p.iter().sum::<f64>() - 1.0).abs() > 1.4901161193847656e-8 {
        return Err("the weights do not sum to one".into());
    }
    let mut cdf = Vec::with_capacity(p.len());
    let mut run = 0.0;
    for v in &p {
        run += v;
        cdf.push(run);
    }
    let last = *cdf.last().expect("a positive total has a weight");
    for c in &mut cdf {
        *c /= last;
    }
    let u = rng.next_double();
    Ok(cdf.partition_point(|c| *c <= u).min(cdf.len() - 1))
}

pub fn advance_turn(reg: &Reg, value: &Value) -> Value {
    let Some(mut rng) = Pcg64::from_json(&value["rng"]) else {
        return json!({ "fault": "advanceTurn without a PCG64 state" });
    };
    let turn = match crate::node::resolve_fresh(reg, value, String::new()) {
        Err(refused) => return refused,
        Ok(turn) => turn,
    };
    let branches = turn.weights.len();
    let counts: Vec<f64> = turn.weights.iter().chain(turn.paused.iter()).copied().collect();
    let mut answer = json!({
        "branches": turn.weights,
        "suspended": turn.paused,
        "exact": turn.exact,
        "unmodelled": turn.unmodelled,
        "index": Value::Null,
        "position": Value::Null,
    });
    // Nothing to draw from: the caller ends the game, as it did, without a draw.
    if counts.is_empty() || numpy_sum(&counts) <= 0.0 {
        answer["rng"] = rng.to_json();
        return answer;
    }
    let index = match choice(&counts, &mut rng) {
        Ok(index) => index,
        Err(why) => return json!({ "fault": why }),
    };
    ADVANCED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    answer["index"] = json!(index);
    answer["rng"] = rng.to_json();
    if index < branches {
        let landed = &turn.positions[index];
        answer["needed"] = json!(crate::resolve::commands::replacements_needed(landed));
        answer["position"] = landed.to_json();
    }
    answer
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
        let mut rng = Pcg64 { state: 1, inc: 3 };
        assert!(choice(&[0.5, f64::NAN], &mut rng).is_err());
        assert!(choice(&[1.0, -0.5, 0.5], &mut rng).is_err());
    }
}
