//! Fixed-capacity bump arena (P17).
//!
//! Scratch memory for a step: allocate through `&self`, then wipe the whole thing with
//! [`BumpArena::reset`]. Capacity is fixed at construction so the arena never grows behind the
//! caller's back — growth would both invalidate handed-out references and hit the global
//! allocator inside a hot loop (see [`crate::alloc_count::assert_no_alloc`]).

use std::alloc::{self, Layout};
use std::cell::Cell;
use std::fmt;
use std::ptr::NonNull;

/// Alignment of the backing block, and the largest alignment the arena can hand out.
const MAX_ALIGN: usize = 16;

pub struct BumpArena {
    base: NonNull<u8>,
    capacity: usize,
    cursor: Cell<usize>,
}

impl BumpArena {
    /// Reserves `bytes` of scratch memory. This is the only allocation the arena ever makes.
    ///
    /// # Panics
    /// If `bytes` is zero or the allocation fails.
    pub fn with_capacity(bytes: usize) -> Self {
        assert!(bytes > 0, "arena capacity must be non-zero");
        let layout = Layout::from_size_align(bytes, MAX_ALIGN).expect("valid arena layout");
        // SAFETY: `layout` has non-zero size.
        let ptr = unsafe { alloc::alloc(layout) };
        let base = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self {
            base,
            capacity: bytes,
            cursor: Cell::new(0),
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes handed out so far, including alignment padding.
    pub fn used(&self) -> usize {
        self.cursor.get()
    }

    /// Drops everything allocated so far and reuses the memory.
    ///
    /// Taking `&mut self` is what makes this sound: every reference handed out by `alloc`
    /// borrows the arena, so none can be alive here.
    pub fn reset(&mut self) {
        self.cursor.set(0);
    }

    /// Stores `value` in the arena.
    ///
    /// # Panics
    /// If the arena is full or `T` needs more than 16-byte alignment.
    // `&self` handing out `&mut` is the point of a bump arena: the block is freshly reserved
    // and reached by nobody else, and `reset` needs `&mut self`, so no reference outlives it.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T: Copy>(&self, value: T) -> &mut T {
        let ptr = self.bump::<T>(1).cast::<T>();
        // SAFETY: `bump` returned a block of at least `size_of::<T>()` bytes, aligned for `T`
        // and not overlapping any earlier allocation, so writing and then handing out a `&mut`
        // borrowed from `&self` aliases nothing. `T: Copy` means no destructor is skipped when
        // `reset` reclaims the block.
        unsafe {
            ptr.write(value);
            &mut *ptr
        }
    }

    /// Stores `len` copies of `value` in the arena.
    ///
    /// # Panics
    /// If the arena is full or `T` needs more than 16-byte alignment.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice<T: Copy>(&self, len: usize, value: T) -> &mut [T] {
        let ptr = self.bump::<T>(len).cast::<T>();
        // SAFETY: as in `alloc`, for `len` consecutive `T`-sized slots.
        unsafe {
            for i in 0..len {
                ptr.add(i).write(value);
            }
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Reserves `count * size_of::<T>()` aligned bytes and returns the start of the block.
    fn bump<T>(&self, count: usize) -> *mut u8 {
        let align = std::mem::align_of::<T>();
        assert!(
            align <= MAX_ALIGN,
            "arena alignment limit is {MAX_ALIGN} bytes"
        );
        let size = std::mem::size_of::<T>()
            .checked_mul(count)
            .expect("arena block size overflow");
        let start = (self.cursor.get() + align - 1) & !(align - 1);
        let end = start.checked_add(size).expect("arena block size overflow");
        assert!(
            end <= self.capacity,
            "bump arena exhausted: need {end} bytes, capacity is {}",
            self.capacity
        );
        self.cursor.set(end);
        // SAFETY: `start <= end <= capacity`, so the offset stays inside the block.
        unsafe { self.base.as_ptr().add(start) }
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, MAX_ALIGN).expect("valid arena layout");
        // SAFETY: `base`/`layout` are exactly what `with_capacity` allocated, and no reference
        // into the block can be alive while `self` is being dropped.
        unsafe { alloc::dealloc(self.base.as_ptr(), layout) };
    }
}

impl fmt::Debug for BumpArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BumpArena")
            .field("used", &self.used())
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_and_resets() {
        let mut arena = BumpArena::with_capacity(256);
        assert_eq!(arena.used(), 0);

        let a = arena.alloc(1u32);
        let b = arena.alloc_slice(4, 2u64);
        *a += 1;
        b[0] = 9;
        assert_eq!(*a, 2);
        assert_eq!(b, &[9, 2, 2, 2]);
        // 4 bytes for the u32, 4 bytes of padding to align the u64s, then 32 bytes.
        assert_eq!(arena.used(), 40);

        arena.reset();
        assert_eq!(arena.used(), 0);
        // The block is reused, not leaked.
        assert_eq!(*arena.alloc(5u32), 5);
        assert_eq!(arena.used(), 4);
    }

    #[test]
    fn allocations_do_not_overlap() {
        let arena = BumpArena::with_capacity(64);
        let a = arena.alloc(1u8);
        let b = arena.alloc(2u8);
        *a = 10;
        assert_eq!(*b, 2);
        assert_ne!(std::ptr::from_mut(a), std::ptr::from_mut(b));
    }

    #[test]
    #[should_panic(expected = "bump arena exhausted")]
    fn over_capacity_panics() {
        let arena = BumpArena::with_capacity(16);
        let _ = arena.alloc_slice(8, 1u64);
    }
}
