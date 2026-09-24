//! Unit tests of the port's randomness accounting (IKA-210).
//!
//! These were Python's `tests/test_resolve.py` (`stratified_rolls`, `stall_success_chance`,
//! `multihit_counts`), testing Python's copies of the three. Python's resolver is going
//! (IKA-204), so the same assertions are held of the port's own functions here. A child
//! module of `moves`, so the private helpers are in reach without widening them.
//!
//!     cargo test --release --manifest-path rust/Cargo.toml

use super::*;
use crate::resolve::stratified_rolls;

fn budget(damage_rolls: i64, enumerate_secondary: bool) -> Budget {
    Budget {
        damage_rolls,
        enumerate_crit: true,
        enumerate_accuracy: true,
        enumerate_status_checks: true,
        enumerate_secondary,
        enumerate_speed_ties: true,
        pinned_policy: false,
        max_branches: 512,
        merge_duplicates: true,
    }
}

fn reg() -> Reg {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../configs/regulations/gen9championsvgc2026regmc.json"
    );
    Reg::load(path).expect("the committed M-C regulation dump")
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn stratified_rolls_are_a_probability_distribution() {
    for count in 1..=16 {
        let rolls = stratified_rolls(&budget(count, true));
        assert!(close(rolls.iter().map(|(_, w)| w).sum::<f64>(), 1.0), "{count}");
        assert!(rolls.iter().all(|(r, _)| *r < 16), "{count}");
        let mut seen: Vec<usize> = rolls.iter().map(|(r, _)| *r).collect();
        seen.dedup();
        assert_eq!(seen.len(), rolls.len(), "{count}: a roll twice");
    }
}

#[test]
fn sixteen_rolls_is_the_exact_distribution() {
    let rolls = stratified_rolls(&budget(16, true));
    assert_eq!(rolls.iter().map(|(r, _)| *r).collect::<Vec<_>>(), (0..16).collect::<Vec<_>>());
    assert!(rolls.iter().all(|(_, w)| close(*w, 1.0 / 16.0)));
}

/// A reduced roll set is a quantisation, not a bias: the mean roll index is kept.
#[test]
fn stratification_keeps_the_mean_roll() {
    let mean = |count| stratified_rolls(&budget(count, true)).iter().map(|(r, w)| *r as f64 * w).sum::<f64>();
    let exact = mean(16);
    for count in [2, 4, 8] {
        assert!((mean(count) - exact).abs() <= 0.5, "{count}: {} vs {exact}", mean(count));
    }
}

#[test]
fn a_pinned_roll_is_a_single_branch() {
    for roll in [0usize, 8, 15] {
        // `Budget.with_fixed_roll`: -1 - roll.
        assert_eq!(stratified_rolls(&budget(-1 - roll as i64, true)), vec![(roll, 1.0)]);
    }
}

/// Protect is certain the first time and one in three the second (`counter` 3).
#[test]
fn stall_chance_matches_showdown() {
    assert!(close(stall_success_chance(1), 1.0));
    assert!(close(stall_success_chance(3), 1.0 / 3.0));
    assert!(close(stall_success_chance(9), 1.0 / 9.0));
}

#[test]
fn multihit_distribution() {
    let reg = reg();
    let exact = budget(16, true);
    let pinned = budget(-1, false);
    let mv = |id: &str| reg.moves.get(&Id::new(id)).unwrap_or_else(|| panic!("{id} not in M-C"));
    assert_eq!(multihit_counts(mv("ironhead"), &exact, "keeneye"), vec![(1, 1.0)]);
    for (id, hits) in [("populationbomb", 10), ("dualwingbeat", 2)] {
        if reg.moves.contains_key(&Id::new(id)) {
            assert_eq!(multihit_counts(mv(id), &exact, "keeneye"), vec![(hits, 1.0)], "{id}");
            // A fixed count is not a range, and Skill Link leaves it alone.
            assert_eq!(multihit_counts(mv(id), &exact, "skilllink"), vec![(hits, 1.0)], "{id}");
        }
    }
    // A [2, 5] move: Showdown's `sample([2 x7, 3 x7, 4 x3, 5 x3])`, 35-35-15-15 (IKA-160).
    let two_to_five = reg
        .moves
        .values()
        .find(|m| m.raw.get("multihit") == Some(&serde_json::json!([2, 5])))
        .expect("M-C has a [2, 5] move");
    let spread = multihit_counts(two_to_five, &exact, "keeneye");
    let want = [(2, 0.35), (3, 0.35), (4, 0.15), (5, 0.15)];
    assert_eq!(spread.len(), want.len());
    for ((hits, w), (want_hits, want_w)) in spread.iter().zip(want) {
        assert_eq!(*hits, want_hits);
        assert!(close(*w, want_w), "{hits}: {w}");
    }
    // A pinned budget takes the minimum, which is what Showdown's policy forces; Skill Link
    // takes the upper end under every budget.
    assert_eq!(multihit_counts(two_to_five, &pinned, "keeneye"), vec![(2, 1.0)]);
    for b in [&exact, &pinned] {
        assert_eq!(multihit_counts(two_to_five, b, "skilllink"), vec![(5, 1.0)]);
    }
}
