//! IKA-458 stage 3.2: a self-switch node (a mid-turn replacement) chosen and played on in
//! the port, inside `advanceTurn`.
//!
//! `selfplay._do_self_switch_node` made one request for each step of this: the pause's
//! options and the turn each produces (`alternativesEncoded`, with a `shared` question and a
//! per-completion `world`), the leaves scored on the inference server, the fold of every
//! option, the choice, and the resumed turn's weights and drawn outcome. Here one pass does
//! all of it, in the same order, for the whole chain of pauses a turn can have (at most five
//! nodes, as `_advance`'s loop): the options are resumed once, a hidden bench's completions
//! come from `hiddenbench::step_of`, the leaves of every world and option are encoded and
//! scored in one batch, each option's fold is taken with the arithmetic of Python's
//! `fold_value` over numpy weights (plain sums), the best option is the first of the best, and
//! the outcome is drawn from the resumed turn with numpy's own draw (`advance::choice`).
//!
//! What Python reads back per node: who chose, the options, the best and its value, the pause's
//! position and what the chooser's opponent had shown (`Decision` of kind `selfswitch`).
//! A leaf without an encoded form, a move among the options and a request the port cannot
//! read are a `fault`: Python then does the turn as before, its generator untouched.

use crate::advance::{choice, Pcg64};
use crate::beliefnode::server_failed;
use crate::encode::{EncodeRules, Encoder};
use crate::hiddenbench::{step_of, Item};
use crate::position::{Pokemon, Position};
use crate::reg::{to_id, Reg};
use crate::resolve::commands::{alternatives_wanted, paused_in};
use crate::resolve::{SlotAction, Suspended, TurnResult};
use crate::served::{Server, Stats};
use crate::speed::ActionKind;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

/// The turns this process carried through a self-switch, the positive control of the stage.
pub static NODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const MOST_NODES: usize = 5;

pub enum Failure {
    /// The caller does the turn itself.
    Fault(String),
    /// An answer to hand back as it is (the inference server's failure).
    Answer(Value),
}

impl From<String> for Failure {
    fn from(why: String) -> Self {
        Failure::Fault(why)
    }
}

pub(crate) struct Hidden {
    pub(crate) game: u64,
    pub(crate) seen: [Vec<String>; 2],
    pub(crate) leads: [Option<Vec<String>>; 2],
}

/// The caller's side of the request: the leaf each side scores with, and what a hidden bench
/// is read from.
pub struct Context {
    pub(crate) servers: Vec<Server>,
    pub(crate) net_scores_ends: Vec<bool>,
    pub(crate) rules: Vec<EncodeRules>,
    pub(crate) leaf: [usize; 2],
    pub(crate) hidden: Option<Hidden>,
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default()
}

impl Context {
    pub fn from_json(value: &Value) -> Result<Context, String> {
        let list = value["servers"].as_array().ok_or("selfSwitch without servers")?;
        let mut servers = Vec::new();
        let mut net_scores_ends = Vec::new();
        let mut rules = Vec::new();
        for s in list {
            servers.push(Server::from_json(s)?);
            net_scores_ends.push(s.get("netScoresEnds").and_then(Value::as_bool).unwrap_or(false));
            rules.push(EncodeRules {
                mega_from_slots: s
                    .get("encoding")
                    .and_then(|e| e.get("megaFromSlots"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
        }
        let leaf = [0, 1].map(|side| value["leaf"][side].as_u64().unwrap_or(0) as usize);
        if leaf.iter().any(|k| *k >= servers.len()) {
            return Err("selfSwitch names a leaf that is not a server".into());
        }
        let hidden = match value.get("hidden").filter(|h| !h.is_null()) {
            None => None,
            Some(h) => Some(Hidden {
                game: h["game"].as_u64().ok_or("selfSwitch.hidden without a game")?,
                seen: [strings(&h["seen"][0]), strings(&h["seen"][1])],
                leads: [0, 1].map(|side| {
                    if h["leads"][side].is_null() {
                        None
                    } else {
                        Some(strings(&h["leads"][side]))
                    }
                }),
            }),
        };
        Ok(Context { servers, net_scores_ends, rules, leaf, hidden })
    }
}

/// One node as Python records it.
pub struct NodeRecord {
    turn: i64,
    position: Value,
    chooser: usize,
    options: Vec<String>,
    best: usize,
    value: f64,
    shown: Vec<Vec<String>>,
}

/// What the chain did: its nodes, the position the turn ended in (None: the turn had no
/// outcome to go on from), the notes to add to the record and the server's work.
pub struct Chain {
    pub position: Option<Position>,
    nodes: Vec<NodeRecord>,
    notes: BTreeSet<String>,
    stats: Vec<Stats>,
    ended: Vec<u64>,
}

impl Chain {
    pub fn to_json(&self) -> Value {
        json!({
            "nodes": self.nodes.iter().map(|n| json!({
                "turn": n.turn,
                "position": n.position,
                "chooser": n.chooser,
                "options": n.options,
                "best": n.best,
                "value": n.value,
                "shown": n.shown,
            })).collect::<Vec<_>>(),
            "notes": self.notes.iter().collect::<Vec<_>>(),
            "served": self.stats.iter().zip(&self.ended).map(|(s, e)| json!({
                "requests": s.requests,
                "rows": s.rows,
                "ended": e,
                "waitUs": s.wait_ns as f64 / 1e3,
                "copyUs": s.copy_ns as f64 / 1e3,
            })).collect::<Vec<_>>(),
        })
    }
}

/// A fold: chance averages and a mid-turn choice takes the chooser's best
/// (`fold.Average`, `fold.BestOf`).
enum Tree {
    Leaf(usize),
    Best(usize, Vec<Tree>),
    Avg(Vec<(f64, Tree)>),
}

/// `fold.fold_value` over numpy float64 weights: plain sums (`sum()` compensates only exact
/// floats, and a numpy weight is not one), first of the best.
fn fold_value(tree: &Tree, values: &[f64]) -> f64 {
    match tree {
        Tree::Leaf(index) => values[*index],
        Tree::Best(chooser, options) => {
            let mut best: Option<f64> = None;
            for option in options {
                let v = fold_value(option, values);
                best = Some(match best {
                    None => v,
                    Some(b) if (*chooser == 0 && v > b) || (*chooser != 0 && v < b) => v,
                    Some(b) => b,
                });
            }
            best.unwrap_or(0.0)
        }
        Tree::Avg(parts) => {
            let mut total = 0.0;
            for (w, _) in parts {
                total += w;
            }
            if total <= 0.0 {
                return 0.0;
            }
            let mut sum = 0.0;
            for (w, part) in parts {
                sum += w * fold_value(part, values);
            }
            sum / total
        }
    }
}

/// `port.turn_leaves`: a turn's leaves and its fold, every pause a `Best` over its replacements
/// resumed in turn (`commands::flatten`, whose JSON this is the in-process twin of).
fn flatten<'a>(
    reg: &'a Reg,
    result: &TurnResult<'a>,
    depth: usize,
    leaves: &mut Vec<Position>,
    notes: &mut BTreeSet<String>,
) -> Result<Tree, String> {
    notes.extend(result.unmodelled.iter().cloned());
    let mut parts: Vec<(f64, Tree)> = Vec::new();
    for branch in &result.branches {
        leaves.push(branch.position.clone());
        parts.push((branch.probability, Tree::Leaf(leaves.len() - 1)));
    }
    for pause in &result.suspended {
        if depth >= 4 {
            notes.insert("more than four mid-turn replacements in one turn".into());
            leaves.push(pause.turn.pos.clone());
            parts.push((pause.probability, Tree::Leaf(leaves.len() - 1)));
            continue;
        }
        // The pauses of this turn share its budget (IKA-284).
        let shared = crate::resolve::sharing_budget(pause, result.suspended.len());
        match alternatives_wanted(reg, shared.as_ref().unwrap_or(pause), None)? {
            Some((chooser, resumed)) if !resumed.is_empty() => {
                let mut options = Vec::new();
                for (_option, one) in &resumed {
                    options.push(flatten(reg, one, depth + 1, leaves, notes)?);
                }
                parts.push((pause.probability, Tree::Best(chooser, options)));
            }
            _ => {
                notes.insert("a suspended turn offered no replacement".into());
                leaves.push(pause.turn.pos.clone());
                parts.push((pause.probability, Tree::Leaf(leaves.len() - 1)));
            }
        }
    }
    Ok(Tree::Avg(parts))
}

fn copy<'a>(pause: &Suspended<'a>) -> Suspended<'a> {
    Suspended { probability: pause.probability, turn: pause.turn.clone(), remaining: pause.remaining.clone() }
}

/// `SideAction.to_choice` of a replacement: what the record writes.
pub(crate) fn choice_string(option: &[SlotAction]) -> Result<String, String> {
    let mut parts = Vec::new();
    for action in option {
        parts.push(match action {
            SlotAction::Switch { party_index, .. } => format!("switch {party_index}"),
            SlotAction::Pass { .. } => "pass".to_string(),
            SlotAction::Move { .. } => return Err("a replacement option is a move".into()),
        });
    }
    Ok(parts.join(", "))
}

/// Everything one world needs of an option: its leaves, local to the plan, and the fold.
struct Plan {
    leaves: Vec<Position>,
    fold: Tree,
}

fn plan_of<'a>(
    reg: &'a Reg,
    result: &TurnResult<'a>,
    notes: &mut BTreeSet<String>,
) -> Result<Plan, String> {
    let mut leaves = Vec::new();
    let fold = flatten(reg, result, 0, &mut leaves, notes)?;
    Ok(Plan { leaves, fold })
}

/// `_with_bench`: `plan`'s leaves with the completion's Pokemon in the unseen slots.
fn patched(plan: &Plan, other: usize, item: &Item) -> Plan {
    let completed = &item.position.as_ref().expect("a completion has its position").sides[other];
    let fresh: HashMap<usize, Rc<Pokemon>> = completed
        .pokemon
        .iter()
        .filter(|mon| item.slots.contains(&mon.slot))
        .map(|mon| (mon.slot, mon.clone()))
        .collect();
    let leaves = plan
        .leaves
        .iter()
        .map(|leaf| {
            let mut leaf = leaf.clone();
            leaf.sides[other].pokemon =
                leaf.sides[other].pokemon.iter().map(|mon| fresh.get(&mon.slot).unwrap_or(mon).clone()).collect();
            leaf.sides[other].mega_capable_slots = completed.mega_capable_slots.clone();
            leaf
        })
        .collect();
    Plan { leaves, fold: clone_tree(&plan.fold) }
}

fn clone_tree(tree: &Tree) -> Tree {
    match tree {
        Tree::Leaf(i) => Tree::Leaf(*i),
        Tree::Best(c, options) => Tree::Best(*c, options.iter().map(clone_tree).collect()),
        Tree::Avg(parts) => Tree::Avg(parts.iter().map(|(w, t)| (*w, clone_tree(t))).collect()),
    }
}

fn identity(mon: &Pokemon) -> String {
    to_id(mon.base_species.as_str())
}

/// One node: who chooses, the options, each option's value and the best, and the turn that
/// choice resumes (so the draw reads it without resolving again). None where the node has
/// nothing to choose or score (the turn then has no outcome to go on from).
struct Scored<'a> {
    chooser: usize,
    options: Vec<Vec<SlotAction>>,
    scores: Vec<f64>,
    best: usize,
    resumed: TurnResult<'a>,
    shown: Vec<Vec<String>>,
}

#[allow(clippy::too_many_lines)]
fn node<'a>(
    reg: &'a Reg,
    encoder: &Encoder,
    context: &Context,
    pause: &Suspended<'a>,
    chain: &mut Chain,
) -> Result<Option<Scored<'a>>, Failure> {
    let Some((chooser, alternatives)) = alternatives_wanted(reg, pause, None)? else {
        return Ok(None);
    };
    if alternatives.is_empty() {
        return Ok(None);
    }
    chain
        .notes
        .insert("mid-turn replacement chosen against the opponent's already-committed action".into());
    let other = 1 - chooser;
    let options: Vec<Vec<SlotAction>> = alternatives.iter().map(|(o, _)| o.clone()).collect();
    let position = &pause.turn.pos;

    // The other side's completions where its bench is hidden (`_self_switch_spread`).
    let mut seen_now: Option<[Vec<String>; 2]> = None;
    let mut spread: Option<Vec<Item>> = None;
    if let Some(hidden) = &context.hidden {
        let step = step_of(reg, hidden.game, position, &hidden.seen, &hidden.leads, Some(other))
            .ok_or_else(|| Failure::Fault(format!("selfSwitch: no game {} is defined", hidden.game)))?;
        chain.notes.extend(step.notes.iter().cloned());
        seen_now = Some([step.seen[0].clone(), step.seen[1].clone()]);
        match step.sides.into_iter().nth(other).flatten() {
            Some(Ok(items)) => {
                if !(items.len() == 1 && items[0].exact) {
                    spread = Some(items);
                }
            }
            Some(Err(why)) => {
                chain.notes.insert(format!("hidden bench at a self-switch: {why}"));
                return Ok(None);
            }
            None => return Err(Failure::Fault("selfSwitch: the other side's completions are missing".into())),
        }
    }

    // The plans, per world and option, in the order Python stacks them.
    let mut true_plans: Vec<Plan> = Vec::new();
    let mut plan_notes = BTreeSet::new();
    for (_option, resumed) in &alternatives {
        true_plans.push(plan_of(reg, resumed, &mut plan_notes)?);
    }
    chain.notes.extend(plan_notes);
    let mut worlds: Vec<(f64, Vec<Plan>)> = Vec::new();
    match &spread {
        None => worlds.push((1.0, true_plans)),
        Some(items) => {
            let slots = &items[0].slots;
            // `_shared_self_switch_plans`: an option whose rest of the turn never reaches the
            // unseen slots is the true plan with each completion's Pokemon swapped in.
            let queued_switch = pause
                .remaining
                .iter()
                .any(|queued| queued.side == other && queued.kind == ActionKind::Switch);
            let before = &pause.turn.pos.sides[other].pokemon;
            let shared: Vec<bool> = alternatives
                .iter()
                .map(|(_o, one)| {
                    if queued_switch || !one.suspended.is_empty() {
                        return false;
                    }
                    one.branches.iter().all(|branch| {
                        before.iter().enumerate().filter(|(_, mon)| slots.contains(&mon.slot)).all(
                            |(index, mon)| {
                                branch.position.sides[other]
                                    .pokemon
                                    .get(index)
                                    .is_some_and(|leaf| **leaf == **mon && leaf.active_index.is_none())
                            },
                        )
                    })
                })
                .collect();
            let unshared: Vec<usize> =
                shared.iter().enumerate().filter(|(_, flag)| !**flag).map(|(k, _)| k).collect();
            for item in items {
                let mut found: Option<Vec<(Vec<SlotAction>, TurnResult<'a>)>> = None;
                if !unshared.is_empty() {
                    let world = paused_in(
                        copy(pause),
                        item.position.clone().expect("a completion has its position"),
                        other,
                    );
                    let (seen_as, resumed) = alternatives_wanted(reg, &world, Some(&unshared))?
                        .ok_or_else(|| Failure::Fault("a completion has no replacement owed".into()))?;
                    let same = resumed.len() == options.len()
                        && resumed.iter().zip(&options).all(|((o, _), p)| choice_string(o) == choice_string(p));
                    if seen_as != chooser || !same {
                        return Err(Failure::Fault(format!(
                            "a completion of side {other}'s bench changed side {chooser}'s self-switch options"
                        )));
                    }
                    found = Some(resumed);
                }
                let mut world_plans = Vec::new();
                let mut notes = BTreeSet::new();
                for k in 0..options.len() {
                    if shared[k] {
                        world_plans.push(patched(&true_plans[k], other, item));
                    } else {
                        let resumed = &found.as_ref().expect("an unshared option was resumed")[k].1;
                        world_plans.push(plan_of(reg, resumed, &mut notes)?);
                    }
                }
                chain.notes.extend(notes);
                worlds.push((item.weight, world_plans));
            }
        }
    }

    // Every leaf of every world and option, scored in one batch.
    let total: usize = worlds.iter().flat_map(|(_, plans)| plans).map(|p| p.leaves.len()).sum();
    if total == 0 {
        return Ok(None);
    }
    let k = context.leaf[chooser];
    let server = &context.servers[k];
    let all: Vec<&Position> = worlds
        .iter()
        .flat_map(|(_, plans)| plans)
        .flat_map(|plan| plan.leaves.iter())
        .collect();
    let encoded = encoder.encode_positions_with(&all, context.rules[k]);
    let decided = crate::objective::decided_leaves(all.iter().copied());
    let batch = crate::served::begin(server, &encoder.widths, &[&encoded], &[total], &mut chain.stats[k])
        .map_err(|f| Failure::Answer(server_failed(f)))?;
    let mut values =
        crate::served::finish(server, batch, &mut chain.stats[k]).map_err(|f| Failure::Answer(server_failed(f)))?;
    for (i, v) in decided {
        if i < values.len() && !v.is_nan() {
            chain.ended[k] += 1;
            if !context.net_scores_ends[k] {
                values[i] = v;
            }
        }
    }
    drop(all);

    let mut scores = vec![0.0f64; options.len()];
    let mut start = 0usize;
    for (weight, plans) in &worlds {
        for (option, plan) in plans.iter().enumerate() {
            let count = plan.leaves.len();
            scores[option] += weight * fold_value(&plan.fold, &values[start..start + count]);
            start += count;
        }
    }
    // Side 0 is the maximiser: the first of the best.
    let mut best = 0usize;
    for (i, v) in scores.iter().enumerate().skip(1) {
        let better = if (chooser == 0) != cfg!(feature = "ika458-selfswitch-control") {
            *v > scores[best]
        } else {
            *v < scores[best]
        };
        if better {
            best = i;
        }
    }
    let shown: Vec<Vec<String>> = match &seen_now {
        Some(seen) => seen.to_vec(),
        None => (0..2)
            .map(|i| {
                let mut ids: Vec<String> = position.sides[i].pokemon.iter().map(|m| identity(m)).collect();
                ids.sort();
                ids
            })
            .collect(),
    };
    let resumed = alternatives.into_iter().nth(best).expect("the best option exists").1;
    Ok(Some(Scored { chooser, options, scores, best, resumed, shown }))
}

/// `_advance` from a drawn pause: every node of the chain, each followed by the draw of what the
/// chosen replacement's turn did. The position the turn ended in, or None when it had no
/// outcome to go on from.
pub fn run<'a>(
    reg: &'a Reg,
    encoder: &Encoder,
    context: &Context,
    first: Suspended<'a>,
    rng: &mut Pcg64,
) -> Result<Chain, Failure> {
    let mut chain = Chain {
        position: None,
        nodes: Vec::new(),
        notes: BTreeSet::new(),
        stats: context.servers.iter().map(|_| Stats::default()).collect(),
        ended: vec![0; context.servers.len()],
    };
    let mut pause = first;
    for attempt in 0..MOST_NODES {
        let Some(scored) = node(reg, encoder, context, &pause, &mut chain)? else {
            return Ok(chain);
        };
        NODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        chain.nodes.push(NodeRecord {
            turn: pause.turn.pos.turn,
            position: pause.turn.pos.to_json(),
            chooser: scored.chooser,
            options: scored.options.iter().map(|o| choice_string(o)).collect::<Result<_, _>>()?,
            best: scored.best,
            value: scored.scores[scored.best],
            shown: scored.shown,
        });
        let mut resumed = scored.resumed;
        if attempt + 1 == MOST_NODES {
            chain.notes.insert("more than five mid-turn replacements in one turn".into());
            return Ok(chain);
        }
        let counts: Vec<f64> = resumed
            .branches
            .iter()
            .map(|b| b.probability)
            .chain(resumed.suspended.iter().map(|s| s.probability))
            .collect();
        if counts.is_empty() || counts.iter().sum::<f64>() <= 0.0 {
            return Ok(chain);
        }
        let index = choice(&counts, rng)?;
        if index < resumed.branches.len() {
            chain.position = Some(resumed.branches.swap_remove(index).position);
            return Ok(chain);
        }
        pause = resumed.suspended.swap_remove(index - resumed.branches.len());
    }
    Ok(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fold_adds_weights_plainly_and_takes_the_first_best() {
        let tree = Tree::Avg(vec![
            (0.5, Tree::Leaf(0)),
            (0.25, Tree::Best(1, vec![Tree::Leaf(1), Tree::Leaf(2)])),
            (0.25, Tree::Best(0, vec![Tree::Leaf(1), Tree::Leaf(2)])),
        ]);
        let value = fold_value(&tree, &[1.0, 0.5, 0.25]);
        assert_eq!(value, (0.5 * 1.0 + 0.25 * 0.25 + 0.25 * 0.5) / 1.0);
        assert_eq!(fold_value(&Tree::Best(0, vec![]), &[]), 0.0);
    }

    #[test]
    fn a_replacement_is_written_as_its_choice() {
        let id = crate::id::Id::new("pikachu");
        let option = [
            SlotAction::Switch { slot: 0, party_index: 3, species: id },
            SlotAction::Pass { slot: 1 },
        ];
        assert_eq!(choice_string(&option).unwrap(), "switch 3, pass");
    }
}
