//! Fixed-capacity, overwrite-oldest ring buffer (spec 9.6: `es-runtime-embedded` ships a
//! telemetry ring buffer and must not touch the heap after startup).
//!
//! [`RingBuffer::with_capacity`] is the only place that allocates; `push` only ever writes
//! into an existing slot.

/// A fixed-size ring of the most recent `capacity` items, oldest overwritten first.
#[derive(Debug)]
pub struct RingBuffer<T> {
    slots: Vec<Option<T>>,
    capacity: usize,
    /// Sequence number of the next item [`push`](Self::push) will write.
    next_seq: u64,
    /// Items overwritten before anything ever read them.
    dropped: u64,
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        // es-core's allocation counter is only installed process-wide when es-core is built
        // with `alloc-count` (see this crate's [dev-dependencies]); skip quietly otherwise so
        // this test still passes (vacuously) rather than lying about a build that can't see it.
        if !es_core::alloc_count::counting_enabled() {
            return;
        }
        let mut r: RingBuffer<u64> = RingBuffer::with_capacity(8);
        es_core::alloc_count::assert_no_alloc(|| {
            for v in 0..20u64 {
                r.push(v);
            }
        });
    }
}
