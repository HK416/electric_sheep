//! Fixed-capacity, overwrite-oldest ring buffer (spec 9.6: `es-runtime-embedded` ships a
//! telemetry ring buffer and must not touch the heap after startup).
//!
//! Lives here (layer 1) rather than in `es-telemetry` (layer 10) so that both
//! `es-runtime-embedded` (layer 9) and `es-telemetry` can use the same implementation without
//! either crossing the other's layer boundary (spec 4.2). `es_telemetry::ring` re-exports this
//! type so existing callers of the telemetry crate see no change.
//!
//! [`RingBuffer::with_capacity`] is the only place that allocates; `push` only ever writes
//! into an existing slot.
//!
//! Two shapes, one behaviour: [`RingBuffer`] owns a `Vec` sized once at startup (`std`), and
//! [`ArrayRing`] owns `[Option<T>; CAP]` inline, so it needs neither `std` nor `alloc` and is
//! what `es-runtime-embedded` uses on the embedded target (spec 9.6). Sequence numbering,
//! overwrite order and the `dropped` count are identical; `ArrayRing` is the subset of the
//! API the deployment runtime actually reads.

/// A fixed-size ring of the most recent `capacity` items, oldest overwritten first.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct RingBuffer<T> {
    slots: Vec<Option<T>>,
    capacity: usize,
    /// Sequence number of the next item [`push`](Self::push) will write.
    next_seq: u64,
    /// Items overwritten before anything ever read them.
    dropped: u64,
}

#[cfg(feature = "std")]
impl<T> RingBuffer<T> {
    /// # Panics
    /// If `capacity` is zero.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be non-zero");
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        Self {
            slots,
            capacity,
            next_seq: 0,
            dropped: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Items currently held (at most `capacity`).
    pub fn len(&self) -> usize {
        self.next_seq.min(self.capacity as u64) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.next_seq == 0
    }

    /// Sequence number of the oldest item still held, i.e. the smallest `seq` for which
    /// [`drain_since`](Self::drain_since) returns anything beyond what has already been
    /// dropped.
    pub fn oldest_seq(&self) -> u64 {
        self.next_seq.saturating_sub(self.capacity as u64)
    }

    /// Writes `value` into the next slot, overwriting the oldest item once full. Never
    /// allocates.
    pub fn push(&mut self, value: T) {
        let idx = (self.next_seq as usize) % self.capacity;
        if self.next_seq >= self.capacity as u64 {
            self.dropped += 1;
        }
        self.slots[idx] = Some(value);
        self.next_seq += 1;
    }

    /// Count of items overwritten before ever being read via [`drain_since`](Self::drain_since)
    /// or [`iter_newest`](Self::iter_newest) at their sequence number.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Iterates currently-held items, most recently pushed first.
    pub fn iter_newest(&self) -> impl Iterator<Item = &T> {
        let held = self.len() as u64;
        (0..held).map(move |i| {
            let seq = self.next_seq - 1 - i;
            self.slots[(seq as usize) % self.capacity]
                .as_ref()
                .expect("seq within [oldest_seq, next_seq) is always occupied")
        })
    }

    /// Iterates items with sequence number `>= seq`, oldest first. Non-destructive (a shared
    /// ring buffer can have more than one reader): `seq` values already overwritten are
    /// silently clamped to [`oldest_seq`](Self::oldest_seq) — that data is gone, see
    /// [`dropped`](Self::dropped).
    pub fn drain_since(&self, seq: u64) -> impl Iterator<Item = &T> {
        let start = seq.max(self.oldest_seq());
        (start..self.next_seq).map(move |s| {
            self.slots[(s as usize) % self.capacity]
                .as_ref()
                .expect("seq within [oldest_seq, next_seq) is always occupied")
        })
    }
}

/// The heap-free ring: `CAP` slots inline, no `std`, no `alloc`, nothing to size at startup
/// (spec 9.6, Appendix B.4).
///
/// `CAP` is a const generic rather than a runtime length precisely so that the storage is part
/// of the struct — on a microcontroller the whole deployment runtime is one static.
///
/// # Panics
/// Every method panics-by-construction never: `CAP == 0` is rejected at compile time by the
/// assertion in [`ArrayRing::new`], which is evaluated in a const context.
#[derive(Debug)]
pub struct ArrayRing<T, const CAP: usize> {
    slots: [Option<T>; CAP],
    /// Sequence number of the next item [`push`](Self::push) will write.
    next_seq: u64,
    /// Items overwritten before anything ever read them.
    dropped: u64,
}

impl<T, const CAP: usize> Default for ArrayRing<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAP: usize> ArrayRing<T, CAP> {
    /// # Panics
    /// If `CAP` is zero. The check is a const assertion, so it fires at compile time.
    pub fn new() -> Self {
        const {
            assert!(CAP > 0, "ArrayRing capacity must be non-zero");
        }
        Self {
            slots: [const { None }; CAP],
            next_seq: 0,
            dropped: 0,
        }
    }

    pub const fn capacity(&self) -> usize {
        CAP
    }

    /// Items currently held (at most `CAP`).
    pub fn len(&self) -> usize {
        self.next_seq.min(CAP as u64) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.next_seq == 0
    }

    /// Sequence number of the oldest item still held.
    pub fn oldest_seq(&self) -> u64 {
        self.next_seq.saturating_sub(CAP as u64)
    }

    /// Count of items overwritten before ever being read.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Writes `value` into the next slot, overwriting the oldest item once full. Never
    /// allocates and never fails: this runs inside the control tick (spec 9.6).
    pub fn push(&mut self, value: T) {
        let idx = (self.next_seq as usize) % CAP;
        if self.next_seq >= CAP as u64 {
            self.dropped += 1;
        }
        self.slots[idx] = Some(value);
        self.next_seq += 1;
    }

    /// Iterates currently-held items, most recently pushed first.
    pub fn iter_newest(&self) -> impl Iterator<Item = &T> {
        let held = self.len() as u64;
        (0..held).filter_map(move |i| {
            let seq = self.next_seq - 1 - i;
            self.slots[(seq as usize) % CAP].as_ref()
        })
    }

    /// Iterates items with sequence number `>= seq`, oldest first. Non-destructive; `seq`
    /// values already overwritten are clamped to [`oldest_seq`](Self::oldest_seq).
    pub fn drain_since(&self, seq: u64) -> impl Iterator<Item = &T> {
        let start = seq.max(self.oldest_seq());
        (start..self.next_seq).filter_map(move |s| self.slots[(s as usize) % CAP].as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_ring_matches_ring_buffer() {
        let mut a: ArrayRing<u64, 3> = ArrayRing::new();
        let mut r: RingBuffer<u64> = RingBuffer::with_capacity(3);
        assert!(a.is_empty());
        for v in 0..7u64 {
            a.push(v);
            r.push(v);
            assert_eq!(a.len(), r.len());
            assert_eq!(a.dropped(), r.dropped());
            assert_eq!(a.oldest_seq(), r.oldest_seq());
            assert_eq!(
                a.iter_newest().copied().collect::<Vec<_>>(),
                r.iter_newest().copied().collect::<Vec<_>>()
            );
            assert_eq!(
                a.drain_since(2).copied().collect::<Vec<_>>(),
                r.drain_since(2).copied().collect::<Vec<_>>()
            );
        }
        assert_eq!(a.capacity(), 3);
    }

    #[test]
    fn array_ring_push_never_allocates() {
        if !crate::alloc_count::counting_enabled() {
            return;
        }
        let mut a: ArrayRing<u64, 8> = ArrayRing::new();
        crate::alloc_count::assert_no_alloc(|| {
            for v in 0..20u64 {
                a.push(v);
            }
        });
    }

    #[test]
    fn push_and_iter_newest_before_wrap() {
        let mut r = RingBuffer::with_capacity(4);
        r.push(1);
        r.push(2);
        r.push(3);
        assert_eq!(r.len(), 3);
        assert_eq!(r.iter_newest().copied().collect::<Vec<_>>(), vec![3, 2, 1]);
        assert_eq!(r.dropped(), 0);
    }

    #[test]
    fn overwrites_oldest_and_counts_dropped() {
        let mut r = RingBuffer::with_capacity(3);
        for v in 0..3 {
            r.push(v);
        }
        assert_eq!(r.dropped(), 0);
        r.push(3); // overwrites the item at seq 0
        r.push(4); // overwrites seq 1
        assert_eq!(r.dropped(), 2);
        assert_eq!(r.len(), 3);
        assert_eq!(r.iter_newest().copied().collect::<Vec<_>>(), vec![4, 3, 2]);
    }

    #[test]
    fn drain_since_clamps_to_oldest_held() {
        let mut r = RingBuffer::with_capacity(3);
        for v in 0..7 {
            r.push(v); // seqs 0..7; only 4,5,6 survive
        }
        assert_eq!(r.oldest_seq(), 4);
        assert_eq!(r.drain_since(0).copied().collect::<Vec<_>>(), vec![4, 5, 6]);
        assert_eq!(r.drain_since(5).copied().collect::<Vec<_>>(), vec![5, 6]);
        assert_eq!(
            r.drain_since(100).copied().collect::<Vec<_>>(),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn empty_buffer_has_no_items() {
        let r: RingBuffer<u8> = RingBuffer::with_capacity(2);
        assert!(r.is_empty());
        assert_eq!(r.iter_newest().count(), 0);
        assert_eq!(r.drain_since(0).count(), 0);
        assert_eq!(r.dropped(), 0);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_capacity_panics() {
        let _: RingBuffer<u8> = RingBuffer::with_capacity(0);
    }

    #[test]
    fn push_after_construction_never_allocates() {
        // The allocation counter is only installed process-wide when this crate is built with
        // `alloc-count` (see `Cargo.toml`'s [dev-dependencies] equivalent: it is this same
        // crate, re-enabled with the feature); skip quietly otherwise so this test still
        // passes (vacuously) rather than lying about a build that can't see it.
        if !crate::alloc_count::counting_enabled() {
            return;
        }
        let mut r: RingBuffer<u64> = RingBuffer::with_capacity(8);
        crate::alloc_count::assert_no_alloc(|| {
            for v in 0..20u64 {
                r.push(v);
            }
        });
    }
}
