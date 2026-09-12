//! A global allocator that counts, behind `--features count-allocations`.
//!
//! The resolver's cost kept coming back to allocation and the evidence kept being indirect
//! -- a clone got cheaper when its Vecs went away, a benchmark moved when a regulation was
//! loaded beside it. Counting is direct: how many allocations does a turn make, and how
//! much would removing a class of them be worth.
//!
//! Off by default. An atomic increment on every allocation is small but it is not nothing,
//! and a number measured with the counter in place is a number about the counter too.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
pub static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
pub static BYTES: AtomicUsize = AtomicUsize::new(0);

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

pub fn reset() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}

pub fn report(turns: f64) {
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) as f64;
    let reallocations = REALLOCATIONS.load(Ordering::Relaxed) as f64;
    let bytes = BYTES.load(Ordering::Relaxed) as f64;
    println!(
        "  allocations:     {allocations:.0} = {:.1} per turn ({:.0} B each, {:.1} reallocs)",
        allocations / turns,
        bytes / allocations.max(1.0),
        reallocations / turns
    );
}
