//! Allocation counting and the allocation-zero assertion (P17).
//!
//! Hot loops (physics step, observation pipeline, control step) must not touch the global
//! allocator: allocation latency is unbounded and shows up as a deadline miss (§9, §18.5).
//! [`assert_no_alloc`] turns that requirement into a test.
//!
//! The counter is per thread and only ticks while [`AllocCounter`] is installed as the global
//! allocator, which `es-core` does under `cfg(test)` or the `alloc-count` feature.

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

/// Global allocator wrapper that counts allocations on the calling thread.
#[derive(Debug)]
pub struct AllocCounter<A>(A);

impl<A> AllocCounter<A> {
    pub const fn new(inner: A) -> Self {
        Self(inner)
    }
}

fn bump() {
    // `try_with` because an allocation can outlive the thread-local during TLS teardown.
    let _ = ALLOCS.try_with(|n| n.set(n.get().wrapping_add(1)));
}

// SAFETY: every method forwards the caller's contract unchanged to the inner allocator and
// only adds a counter bump, which neither allocates nor touches the returned memory.
unsafe impl<A: GlobalAlloc> GlobalAlloc for AllocCounter<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: same layout, same contract as our caller's.
        unsafe { self.0.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: same layout, same contract as our caller's.
        unsafe { self.0.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        // SAFETY: `ptr`/`layout` come from this allocator, as our caller guarantees.
        unsafe { self.0.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Frees are not counted: only acquiring memory can block.
        // SAFETY: `ptr`/`layout` come from this allocator, as our caller guarantees.
        unsafe { self.0.dealloc(ptr, layout) }
    }
}

/// Allocations made by the calling thread since it started.
pub fn allocation_count() -> u64 {
    ALLOCS.try_with(Cell::get).unwrap_or(0)
}

/// Whether the counting allocator is installed. When it is not, [`assert_no_alloc`] cannot
/// observe anything and passes.
pub const fn counting_enabled() -> bool {
    cfg!(any(test, feature = "alloc-count"))
}

/// Runs `f` and panics if it allocated.
///
/// # Panics
/// If the global allocator was called on this thread inside `f`.
pub fn assert_no_alloc<R>(f: impl FnOnce() -> R) -> R {
    let before = allocation_count();
    let value = f();
    let after = allocation_count();
    assert!(
        after == before,
        "expected no allocation, observed {} (counting_enabled = {})",
        after - before,
        counting_enabled()
    );
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::BumpArena;

    #[test]
    fn counter_is_installed_in_tests() {
        assert!(counting_enabled());
        let before = allocation_count();
        let v = vec![1u8, 2, 3];
        assert!(allocation_count() > before);
        drop(v);
    }

    #[test]
    fn arena_allocation_is_allocation_free() {
        let arena = BumpArena::with_capacity(1024);
        let value = assert_no_alloc(|| {
            let slot = arena.alloc(7u64);
            let slice = arena.alloc_slice(16, 1u32);
            *slot + u64::from(slice.iter().sum::<u32>())
        });
        assert_eq!(value, 23);
    }

    #[test]
    #[should_panic(expected = "expected no allocation")]
    fn heap_allocation_is_caught() {
        assert_no_alloc(|| Vec::<u8>::with_capacity(1));
    }
}
