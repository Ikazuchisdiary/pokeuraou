//! The readable trace of a turn: Python's `Branch.events` and `acts` (IKA-215).
//!
//! Off unless a command asks for it (`events: true`). A turn that is not logging carries
//! `None` and nothing else: no line is formatted, no vector is allocated, and a clone copies
//! one null pointer. Every line goes through `log_event!`, which tests for the log before it
//! evaluates its format arguments, so the generation road pays one branch per call site.
//!
//! The lines are Python's `_Turn.log` strings, character for character, and `acts` cuts
//! them the way `_Turn.begin` does: once per queued action, just before it executes, and
//! once for the residual phase. Merging keeps the first contributor's trace, as Python's
//! `_fold` does, so a merged branch may say "-31" where the roll folded into it dealt 34.

/// What one branch has said so far.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct EventLog {
    pub events: Vec<String>,
    /// `(index into events, action label)`, in the order the turn ran the actions.
    pub acts: Vec<(usize, String)>,
    /// The chances this branch took (IKA-345), one tag per draw that had more than one
    /// outcome: `<kind> <user> <move id> <target> [<arg>...]`, e.g. `crit p1a flareblitz p2b`
    /// or `roll p1a flareblitz p2b 0 3 2` (the kept roll's place, the rolls kept, Showdown's
    /// roll 0..15). Both sides of a draw are tagged (`crit`/`nocrit`, `hit`/`miss`), so two
    /// branches of one turn differ in their tags wherever they differ in a draw. Unlike
    /// `events`, merging keeps only the tags every merged branch shares (`common_chance`):
    /// what is left is true of the whole branch.
    pub chance: Vec<String>,
}

impl EventLog {
    /// Keeps the tags of `self.chance` that `other` has too (as a multiset, in `self`'s
    /// order): the branch they merge into is every one of them, so only what they all
    /// share describes it.
    pub fn common_chance(&mut self, other: &EventLog) {
        let mut left: Vec<&str> = other.chance.iter().map(String::as_str).collect();
        let mut kept: Vec<String> = Vec::with_capacity(self.chance.len());
        for tag in self.chance.iter() {
            if let Some(at) = left.iter().position(|o| *o == tag) {
                left.swap_remove(at);
                kept.push(tag.clone());
            } else if let Some((at, merged)) = widen_roll(tag, &left) {
                // Two rolls of one hit that came to the same thing are the rolls between
                // them: `roll <user> <move> <target> <place> <kept> <roll>` widens to ranges.
                left.swap_remove(at);
                kept.push(merged);
            }
        }
        self.chance = kept;
    }
}

/// `tag` (a roll) widened by the same hit's roll in `left`, and where that was.
fn widen_roll(tag: &str, left: &[&str]) -> Option<(usize, String)> {
    let mine: Vec<&str> = tag.split(' ').collect();
    if mine.len() != 7 || mine[0] != "roll" {
        return None;
    }
    let at = left.iter().position(|o| {
        let theirs: Vec<&str> = o.split(' ').collect();
        theirs.len() == 7 && theirs[..4] == mine[..4] && theirs[5] == mine[5]
    })?;
    let theirs: Vec<&str> = left[at].split(' ').collect();
    let bounds = |s: &str| -> (i64, i64) {
        let mut it = s.split('-').map(|x| x.parse::<i64>().unwrap_or(0));
        let lo = it.next().unwrap_or(0);
        (lo, it.next().unwrap_or(lo))
    };
    let span = |a: &str, b: &str| {
        let ((a0, a1), (b0, b1)) = (bounds(a), bounds(b));
        let (lo, hi) = (a0.min(b0), a1.max(b1));
        if lo == hi { lo.to_string() } else { format!("{lo}-{hi}") }
    };
    Some((
        at,
        format!(
            "roll {} {} {} {} {} {}",
            mine[1], mine[2], mine[3], span(mine[4], theirs[4]), mine[5], span(mine[6], theirs[6])
        ),
    ))
}

/// Merges `other`'s log into `into`'s for a merged branch (see `EventLog::chance`).
pub fn merge_logs(into: &mut Option<Box<EventLog>>, other: &Option<Box<EventLog>>) {
    if let (Some(a), Some(b)) = (into.as_mut(), other.as_ref()) {
        a.common_chance(b);
    }
}

/// A chance tag on a logging turn (`EventLog::chance`); nothing at all otherwise.
macro_rules! chance_tag {
    ($turn:expr, $($arg:tt)*) => {
        if $turn.log.is_some() {
            let tag = format!($($arg)*);
            if let Some(log) = $turn.log.as_mut() {
                log.chance.push(tag);
            }
        }
    };
}

/// `_Turn.name`: `p1a`, `p2b`.
pub struct Name(pub usize, pub usize);

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "p{}{}", self.0 + 1, if self.1 == 0 { 'a' } else { 'b' })
    }
}

/// `turn.log(f"...")`. The arguments are evaluated only when the turn is logging.
macro_rules! log_event {
    ($turn:expr, $($arg:tt)*) => {
        if $turn.log.is_some() {
            let line = format!($($arg)*);
            if let Some(log) = $turn.log.as_mut() {
                log.events.push(line);
            }
        }
    };
}
