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
