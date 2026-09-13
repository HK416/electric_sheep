//! The telemetry ring buffer of spec 9.6.
//!
//! **Duplication, on purpose.** `es_telemetry::ring::RingBuffer` is the real, general version
//! of this, and it is a better one — it has sequence numbers, `drain_since` and a dropped
//! count. `es-telemetry` is layer 10 and this crate is layer 9, so depending on it is a
//! layering violation (spec 4.2). The lazy fix is the few lines below; the correct fix, when a
//! second crate at layer <= 9 wants one, is to move `RingBuffer` into `es-core` (layer 1) and
//! have both crates use it. Until then this stays small enough that divergence cannot hurt.

use es_core::PhysTick;
use es_ir::deployment::Micros;
use es_safety::ActionSource;

/// One control tick, as the deployment records it. Plain `Copy` data: pushing one must not
/// allocate, so nothing here owns a heap object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickRecord {
    pub tick: PhysTick,
    pub source: ActionSource,
    /// `es_safety::EventSet` as its bitset, so the record stays a POD a transport can memcpy.
    pub events: u32,
    /// Whether this tick ran the policy or reused the buffered chunk (spec 8.6).
    pub replanned: bool,
    pub obs_age: Micros,
}

/// Fixed-capacity, overwrite-oldest ring. [`TelemetryRing::with_capacity`] is the only place
/// that allocates.
#[derive(Debug)]
pub struct TelemetryRing {
    slots: Vec<TickRecord>,
    cap: usize,
    /// Pushes since construction; the write index is this modulo `cap`.
    pushed: u64,
}

impl TelemetryRing {
    /// # Panics
    /// If `capacity` is zero.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "TelemetryRing capacity must be non-zero");
        Self {
            slots: Vec::with_capacity(capacity),
            cap: capacity,
            pushed: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Records currently held, at most `capacity`.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Total pushes since construction, including ones already overwritten.
    pub fn pushed(&self) -> u64 {
        self.pushed
    }

    /// Never allocates: the `Vec` reaches `cap` once and from then on only slots are
    /// overwritten.
    pub fn push(&mut self, record: TickRecord) {
        if self.slots.len() < self.cap {
            self.slots.push(record);
        } else {
            self.slots[(self.pushed as usize) % self.cap] = record;
        }
        self.pushed += 1;
    }

    /// Held records, most recent first.
    pub fn iter_newest(&self) -> impl Iterator<Item = &TickRecord> {
        let (len, cap, pushed) = (self.slots.len(), self.cap, self.pushed as usize);
        (0..len).map(move |i| &self.slots[(pushed + cap - 1 - i) % cap])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(tick: u64) -> TickRecord {
        TickRecord {
            tick: PhysTick(tick),
            source: ActionSource::Policy,
            events: 0,
            replanned: false,
            obs_age: Micros(0),
        }
    }

    #[test]
    fn overwrites_oldest_and_reports_newest_first() {
        let mut r = TelemetryRing::with_capacity(3);
        assert!(r.is_empty());
        for t in 0..5 {
            r.push(rec(t));
        }
        assert_eq!(r.len(), 3);
        assert_eq!(r.pushed(), 5);
        let ticks: Vec<u64> = r.iter_newest().map(|x| x.tick.0).collect();
        assert_eq!(ticks, vec![4, 3, 2]);
    }

    #[test]
    fn push_never_allocates_after_construction() {
        if !es_core::alloc_count::counting_enabled() {
            return;
        }
        let mut r = TelemetryRing::with_capacity(8);
        es_core::alloc_count::assert_no_alloc(|| {
            for t in 0..40 {
                r.push(rec(t));
            }
        });
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_capacity_panics() {
        let _ = TelemetryRing::with_capacity(0);
    }
}
