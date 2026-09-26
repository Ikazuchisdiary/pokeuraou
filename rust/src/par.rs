//! A node's cells on several threads (IKA-32).
//!
//! A game played against a person on a clock is one game on a machine of sixteen cores,
//! and the port resolves a node's cells one after another. The cells are independent --
//! each is `resolve_turn` of one position and two actions -- so they can be resolved on
//! as many threads as the machine lends, and gathered back in the order they were asked
//! for. Everything after the gathering (sharing a leaf the node already has, encoding,
//! the header) stays on one thread and in the order it always had, so the answer is the
//! same bytes at any thread count.
//!
//! Off by default: `threads()` is 1 unless `--threads N` or `POKEURAOU_PORT_THREADS`
//! says otherwise, and at 1 no pool exists and the callers run their old loops.
//! Generation keeps it at 1 -- its machine is already full of workers.
//!
//! No crate for this (rayon would be the obvious one): the pool is fifty lines, and the
//! part that needs care -- a resolved turn holds `Rc`s and is not `Send` -- is ours to
//! reason about either way. See `Handoff`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

static THREADS: OnceLock<usize> = OnceLock::new();

/// Fixes the thread count, before anything asks for it. `--threads N` on the command line.
pub fn set_threads(threads: usize) {
    let _ = THREADS.set(threads.max(1));
}

/// Threads a node's cells are resolved on: `--threads`, else `POKEURAOU_PORT_THREADS`,
/// else 1. Anything that does not parse as a positive number is 1.
pub fn threads() -> usize {
    *THREADS.get_or_init(|| {
        std::env::var("POKEURAOU_PORT_THREADS")
            .ok()
            .and_then(|text| text.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1)
    })
}

/// Maps run on the pool, and items of them done by a thread other than the one that
/// asked. The positive control: a check that the answers agree at 1 and at N threads is
/// only a check if the N-thread road was taken, and these say how often it was.
pub static PARALLEL_MAPS: AtomicUsize = AtomicUsize::new(0);
pub static PARALLEL_ITEMS: AtomicUsize = AtomicUsize::new(0);
pub static OFF_MAIN_ITEMS: AtomicUsize = AtomicUsize::new(0);
/// Nodes of a `fills` crossing resolved one per thread (IKA-32 stage 2), the positive
/// control of that road.
pub static NODE_ITEMS: AtomicUsize = AtomicUsize::new(0);

/// Counts `n` nodes of a crossing that went one per thread.
pub fn count_node_items(n: usize) {
    NODE_ITEMS.fetch_add(n, Ordering::Relaxed);
}

/// A value built on one thread and read on another, although it holds `Rc`s.
///
/// Sound here, and only here, because of how the pool hands it over. A worker builds the
/// value's whole `Rc` graph itself -- from its own parse of the request's position, never
/// from a clone of the asker's -- and drops everything else it held of that graph before
/// the job ends. The asker does not touch any `Rc` while a job runs (it is either inside
/// the job, working on its own graph, or waiting), and reads the handed-over values only
/// after every worker has said it is done, under the pool's mutex. So no two threads
/// ever touch one reference count at once, and each touch is ordered after the last one.
///
/// One `Rc` is shared across jobs: the empty blob `Position::from_json` hands out from a
/// thread-local. Each worker has its own and a worker never exits (the pool is never torn
/// down), so its blob is only touched by it inside a job and by the asker between jobs.
pub struct Handoff<T>(pub T);

// SAFETY: see the type's documentation; `map_with` is the only place one crosses.
unsafe impl<T> Send for Handoff<T> {}

/// The job the workers are running: a type-erased pointer to the asker's closure.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn() + Sync));

// SAFETY: the closure it points to is `Sync`, and the asker keeps it alive until every
// worker has finished with it (`run` waits for `busy` to reach 0 before returning).
unsafe impl Send for Job {}

struct State {
    generation: u64,
    job: Option<Job>,
    busy: usize,
}

pub struct Pool {
    workers: usize,
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

thread_local! {
    /// Set on a pool's own threads, for counting what ran off the asker's thread.
    static ON_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Pool {
    /// A pool of `threads` in all: the asker and `threads - 1` workers. Leaked: its
    /// workers never exit (see `Handoff`).
    pub fn leaked(threads: usize) -> &'static Pool {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            workers: threads.max(1) - 1,
            state: Mutex::new(State { generation: 0, job: None, busy: 0 }),
            work: Condvar::new(),
            done: Condvar::new(),
        }));
        for index in 0..pool.workers {
            std::thread::Builder::new()
                .name(format!("cells-{index}"))
                .spawn(move || pool.serve())
                .expect("a cell thread");
        }
        pool
    }

    /// The process's pool, of `threads()`; None at 1, where nothing is shared.
    pub fn global() -> Option<&'static Pool> {
        static GLOBAL: OnceLock<Option<&'static Pool>> = OnceLock::new();
        *GLOBAL.get_or_init(|| (threads() > 1).then(|| Pool::leaked(threads())))
    }

    /// Threads in all, the asker's included.
    pub fn threads(&self) -> usize {
        self.workers + 1
    }

    fn serve(&self) {
        ON_WORKER.with(|w| w.set(true));
        let mut seen = 0u64;
        loop {
            let job = {
                let mut state = self.state.lock().unwrap();
                while state.generation == seen {
                    state = self.work.wait(state).unwrap();
                }
                seen = state.generation;
                state.job.expect("a generation always carries its job")
            };
            // SAFETY: `run` keeps the closure alive until `busy` is back to 0.
            let task = unsafe { &*job.0 };
            // A panic inside is caught by `map_with`'s own guard; this one only keeps the
            // thread and the count alive if something slipped past it.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
            let mut state = self.state.lock().unwrap();
            state.busy -= 1;
            if state.busy == 0 {
                self.done.notify_all();
            }
        }
    }

    /// Runs `task` on every worker and on this thread, and returns when all are done.
    fn run(&self, task: &(dyn Fn() + Sync)) {
        // SAFETY (lifetime): the pointer is dropped from `state.job` below, after every
        // worker has decremented `busy`, i.e. has returned from `task`.
        let erased: *const (dyn Fn() + Sync) = unsafe { std::mem::transmute(task) };
        {
            let mut state = self.state.lock().unwrap();
            state.job = Some(Job(erased));
            state.busy = self.workers;
            state.generation += 1;
            self.work.notify_all();
        }
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
        let mut state = self.state.lock().unwrap();
        while state.busy > 0 {
            state = self.done.wait(state).unwrap();
        }
        state.job = None;
        drop(state);
        if let Err(panic) = mine {
            std::panic::resume_unwind(panic);
        }
    }

    /// `(0..n).map(|k| each(&mut local, k))`, the items spread over the pool's threads and
    /// returned in index order.
    ///
    /// `local` is made once per thread that takes an item, by `init` on that thread, and
    /// dropped there before the thread says it is done -- this is where a worker parses
    /// its own copy of a position. Which thread takes which item is not fixed (they take
    /// the next index as they come free), so `each` must answer an index the same way
    /// whichever thread and whatever it did before; the callers' items are one cell each,
    /// resolved from the position and the two actions alone.
    pub fn map_with<S, T, I, F>(&self, n: usize, init: I, each: F) -> Vec<T>
    where
        I: Fn() -> S + Sync,
        F: Fn(&mut S, usize) -> T + Sync,
    {
        PARALLEL_MAPS.fetch_add(1, Ordering::Relaxed);
        PARALLEL_ITEMS.fetch_add(n, Ordering::Relaxed);
        let next = AtomicUsize::new(0);
        let gathered: Mutex<Vec<(usize, Handoff<T>)>> = Mutex::new(Vec::with_capacity(n));
        let failed: Mutex<Option<Box<dyn std::any::Any + Send>>> = Mutex::new(None);
        let task = || {
            let mut mine: Vec<(usize, Handoff<T>)> = Vec::new();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut local: Option<S> = None;
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= n {
                        break;
                    }
                    let state = local.get_or_insert_with(&init);
                    mine.push((k, Handoff(each(state, k))));
                }
                // `local` goes here, on this thread, before the job is over.
            }));
            if ON_WORKER.with(|w| w.get()) {
                OFF_MAIN_ITEMS.fetch_add(mine.len(), Ordering::Relaxed);
            }
            gathered.lock().unwrap().extend(mine);
            if let Err(panic) = outcome {
                failed.lock().unwrap().get_or_insert(panic);
            }
        };
        self.run(&task);
        if let Some(panic) = failed.into_inner().unwrap() {
            std::panic::resume_unwind(panic);
        }
        let mut gathered = gathered.into_inner().unwrap();
        // The order the cells were asked in, not the order they finished in. IKA-32's
        // control build leaves the finishing order, which is what forgetting this looks like.
        if !cfg!(feature = "ika32-control") {
            gathered.sort_unstable_by_key(|(k, _)| *k);
        }
        gathered.into_iter().map(|(_, Handoff(value))| value).collect()
    }
}

/// The counters, for the `parallel` request and the report at exit.
pub fn report() -> serde_json::Value {
    serde_json::json!({
        "kind": "parallel",
        "threads": threads(),
        "maps": PARALLEL_MAPS.load(Ordering::Relaxed),
        "items": PARALLEL_ITEMS.load(Ordering::Relaxed),
        "offMain": OFF_MAIN_ITEMS.load(Ordering::Relaxed),
        "nodes": NODE_ITEMS.load(Ordering::Relaxed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_come_back_in_index_order_and_off_the_asking_thread() {
        let pool = Pool::leaked(4);
        let before = OFF_MAIN_ITEMS.load(Ordering::Relaxed);
        for _ in 0..20 {
            let out = pool.map_with(
                257,
                || std::rc::Rc::new(7usize),
                |seven, k| {
                    // Uneven work, so the finishing order is not the index order.
                    let mut spin = 0u64;
                    for step in 0..((k * 7919) % 2000) as u64 {
                        spin = spin.wrapping_add(step);
                    }
                    std::hint::black_box(spin);
                    (k, **seven, std::rc::Rc::clone(seven))
                },
            );
            let indices: Vec<usize> = out.iter().map(|(k, _, _)| *k).collect();
            assert_eq!(indices, (0..257).collect::<Vec<_>>());
            assert!(out.iter().all(|(_, seven, _)| *seven == 7));
        }
        assert!(OFF_MAIN_ITEMS.load(Ordering::Relaxed) > before, "no item ran on a worker");
    }

    #[test]
    fn a_panic_in_an_item_reaches_the_asker_and_the_pool_survives() {
        let pool = Pool::leaked(3);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.map_with(50, || (), |_, k| if k == 31 { panic!("cell 31") } else { k })
        }));
        assert!(caught.is_err());
        let again = pool.map_with(50, || (), |_, k| k * 2);
        assert_eq!(again, (0..50).map(|k| k * 2).collect::<Vec<_>>());
    }
}
