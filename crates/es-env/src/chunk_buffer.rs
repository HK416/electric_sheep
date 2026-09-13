//! The per-env action chunk buffer (§8.5, §8.6).
//!
//! A policy emits `H` predicted actions and the controller consumes one per control tick, so
//! between two arrivals the buffer *is* the controller's action source. Three things follow,
//! and they are the whole module:
//!
//! - **Fixed storage.** [`ChunkBuffer::next_action`] runs on the control path, so it allocates
//!   nothing: [`CHUNK_SLOTS`] chunks live in an inline array and the overlap set is an inline
//!   index array that is insertion-sorted. `push` allocates nothing either.
//! - **Fixed order.** Overlapping chunks are combined in ascending arrival order — the push
//!   sequence number, never wall-clock and never slot order — so the ensemble sum is
//!   reproducible bitwise (§12.3).
//! - **Underrun is signalled, never papered over.** When no chunk covers the tick,
//!   `next_action` returns `None`. The caller hands the Safety Plane an *empty* chunk and the
//!   plane produces the fallback (§8.6, §9.4). Fabricating an action here would put an
//!   unchecked value on the actuator path, which `INV-12` forbids.

use es_ir::learning::ChunkBlendPolicy;
use es_safety::ActionChunk;

/// How many overlapping chunks one buffer keeps.
///
/// The live overlap is `ceil(H / replan_interval)`; for the §8.6 worked example (H = 20,
/// replanning every 4 control ticks) that is 5. Eight slots leave headroom and keep the
/// scan cheap; a ninth chunk evicts the oldest, which is the one the ACT weights have
/// already decayed to nothing.
///
/// It lives in `es-core` because `es-compile`'s memory budget sizes this buffer too and
/// cannot see this crate (§4.2, §20.2).
pub use es_core::sizing::CHUNK_SLOTS;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot<const NJ: usize, const H: usize> {
    /// Control tick at which row 0 of this chunk executes.
    start: u64,
    /// Rows the policy filled.
    valid: usize,
    /// Push order. Monotone, so "newest" never depends on where the slot landed.
    seq: u64,
    actions: [[f64; NJ]; H],
}

impl<const NJ: usize, const H: usize> Slot<NJ, H> {
    const EMPTY: Self = Self {
        start: 0,
        valid: 0,
        seq: 0,
        actions: [[0.0; NJ]; H],
    };
}

/// The chunk buffer of one env (§8.6).
///
/// `H` is the prediction horizon and `execute_chunk` is `K ≤ H` of §8.5. `K` bounds how long a
/// single chunk may drive the controller under [`ChunkBlendPolicy::HardSwitch`] and
/// [`ChunkBlendPolicy::LinearBlend`] — past it the chunk is stale and an underrun is the
/// correct, recorded outcome. [`ChunkBlendPolicy::TemporalEnsemble`] is the exception: ACT
/// averages *all* overlapping predictions, so there the span is `valid` rows.
#[derive(Clone, Debug)]
pub struct ChunkBuffer<const NJ: usize, const H: usize> {
    slots: [Slot<NJ, H>; CHUNK_SLOTS],
    execute_chunk: usize,
    blend: ChunkBlendPolicy,
    next_seq: u64,
    served: u64,
    underruns: u64,
}

impl<const NJ: usize, const H: usize> ChunkBuffer<NJ, H> {
    /// `execute_chunk` is clamped into `1..=H`: `K = 0` would underrun on every tick and
    /// `K > H` would read past the prediction.
    pub fn new(execute_chunk: usize, blend: ChunkBlendPolicy) -> Self {
        Self {
            slots: [Slot::EMPTY; CHUNK_SLOTS],
            execute_chunk: execute_chunk.clamp(1, H.max(1)),
            blend,
            next_seq: 1,
            served: 0,
            underruns: 0,
        }
    }

    pub fn execute_chunk(&self) -> usize {
        self.execute_chunk
    }

    pub fn blend(&self) -> ChunkBlendPolicy {
        self.blend
    }

    /// Chunks pushed since construction — one per policy invocation result delivered to this
    /// env. It is what stamps the `ActionChunk::seq` the Safety Plane judges freshness by
    /// (§8.6, §9.4): a tick that adds no arrival must not look like a new chunk.
    pub fn arrivals(&self) -> u64 {
        self.next_seq - 1
    }

    /// Chunks currently held (live slots), for tests and telemetry.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.valid > 0).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ticks served from a chunk, and ticks that found none (§12.4 `chunk_underrun_rate`).
    pub fn served(&self) -> u64 {
        self.served
    }

    pub fn underruns(&self) -> u64 {
        self.underruns
    }

    /// `underruns / (served + underruns)`, or `None` before the first call — never a
    /// fabricated zero (§12.4).
    pub fn underrun_rate(&self) -> Option<f64> {
        let total = self.served + self.underruns;
        (total > 0).then(|| self.underruns as f64 / total as f64)
    }

    /// Drops every chunk. Called on episode reset: a chunk predicted for the old episode has
    /// no meaning in the new one. Counters survive, because they are run statistics.
    pub fn clear(&mut self) {
        self.slots = [Slot::EMPTY; CHUNK_SLOTS];
    }

    /// Stores `chunk`, whose row 0 executes at `arrival_tick` (control ticks).
    ///
    /// An empty chunk (`valid == 0`) is dropped: it carries no action, and storing it would
    /// occupy a slot that a real prediction needs.
    pub fn push(&mut self, chunk: &ActionChunk<NJ, H>, arrival_tick: u64) {
        if chunk.valid == 0 {
            return;
        }
        // Evict the oldest by sequence number, so eviction order is arrival order.
        let victim = (0..CHUNK_SLOTS)
            .min_by_key(|i| (self.slots[*i].valid > 0, self.slots[*i].seq))
            .unwrap_or(0);
        self.slots[victim] = Slot {
            start: arrival_tick,
            valid: chunk.valid.min(H),
            seq: self.next_seq,
            actions: chunk.actions,
        };
        self.next_seq += 1;
    }

    /// How many rows of a slot may drive the controller.
    fn span(&self, valid: usize) -> usize {
        match self.blend {
            ChunkBlendPolicy::TemporalEnsemble { .. } => valid,
            _ => valid.min(self.execute_chunk),
        }
    }

    /// The action for `tick`, or `None` on underrun (§8.6), counted into `served`/`underruns`.
    ///
    /// Call once per control tick per env: the counters are §12.4's `chunk_underrun_rate`.
    /// Use [`ChunkBuffer::action_at`] for a lookahead that must not be counted.
    pub fn next_action(&mut self, tick: u64) -> Option<[f64; NJ]> {
        let out = self.action_at(tick);
        if out.is_some() {
            self.served += 1;
        } else {
            self.underruns += 1;
        }
        out
    }

    /// The action for `tick`, or `None` on underrun (§8.6), without touching the counters.
    ///
    /// No allocation and no copy of a slot: the overlap set is an inline `[usize;
    /// CHUNK_SLOTS]` of *borrowed* slots, insertion-sorted by arrival so the reduction order
    /// is fixed (§12.3).
    pub fn action_at(&self, tick: u64) -> Option<[f64; NJ]> {
        let mut order = [0usize; CHUNK_SLOTS];
        let mut n = 0usize;
        for i in 0..CHUNK_SLOTS {
            let s = &self.slots[i];
            if s.valid == 0 || tick < s.start || tick - s.start >= self.span(s.valid) as u64 {
                continue;
            }
            // Insertion sort by arrival sequence: `order[0]` is the oldest overlapping chunk,
            // which is ACT's `i = 0`.
            let mut at = n;
            while at > 0 && self.slots[order[at - 1]].seq > s.seq {
                order[at] = order[at - 1];
                at -= 1;
            }
            order[at] = i;
            n += 1;
        }
        if n == 0 {
            return None;
        }
        let row = |i: usize| {
            let s = &self.slots[i];
            s.actions[(tick - s.start) as usize]
        };
        Some(match self.blend {
            ChunkBlendPolicy::HardSwitch => row(order[n - 1]),
            ChunkBlendPolicy::LinearBlend { steps } => {
                let new = order[n - 1];
                if steps == 0 || n == 1 {
                    row(new)
                } else {
                    // Ramp from the previous chunk to the newest over `steps` control ticks,
                    // counted from the newest chunk's own start.
                    let elapsed = tick - self.slots[new].start + 1;
                    let a = (elapsed.min(u64::from(steps)) as f64) / f64::from(steps);
                    let (prev, new) = (row(order[n - 2]), row(new));
                    let mut out = [0.0; NJ];
                    for j in 0..NJ {
                        out[j] = (1.0 - a) * prev[j] + a * new[j];
                    }
                    out
                }
            }
            // ACT temporal ensembling: `w_i = exp(-m * i)` with `i = 0` the **oldest**
            // overlapping prediction, so a smaller `m` incorporates new observations faster.
            // `es_math::approx::exp` rather than `f64::exp`: std transcendentals are forbidden
            // on a deterministic path (§3.4).
            ChunkBlendPolicy::TemporalEnsemble { weight_decay } => {
                let mut out = [0.0; NJ];
                let mut total = 0.0f64;
                for (i, slot) in order[..n].iter().enumerate() {
                    let w = f64::from(es_math::approx::exp(-weight_decay * i as f32));
                    let a = row(*slot);
                    for j in 0..NJ {
                        out[j] += w * a[j];
                    }
                    total += w;
                }
                if total > 0.0 {
                    for v in &mut out {
                        *v /= total;
                    }
                }
                out
            }
        })
    }
}

// The property under test is bitwise reproducibility, so exact comparisons are deliberate.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use es_ir::deployment::ExecutionMode;

    const H: usize = 4;

    fn chunk(rows: [f64; H], valid: usize) -> ActionChunk<1, H> {
        ActionChunk::new(rows.map(|v| [v]), valid, ExecutionMode::RecedingHorizon)
    }

    fn ensemble(m: f32) -> ChunkBlendPolicy {
        ChunkBlendPolicy::TemporalEnsemble { weight_decay: m }
    }

    #[test]
    fn hard_switch_serves_k_rows_then_underruns() {
        let mut b = ChunkBuffer::<1, H>::new(2, ChunkBlendPolicy::HardSwitch);
        assert!(b.is_empty());
        assert_eq!(b.next_action(0), None, "nothing pushed yet");
        b.push(&chunk([1.0, 2.0, 3.0, 4.0], 4), 10);
        assert_eq!(b.len(), 1);
        assert_eq!(b.next_action(10), Some([1.0]));
        assert_eq!(b.next_action(11), Some([2.0]));
        // K = 2: rows 2 and 3 exist but are past the replanning point (§8.5).
        assert_eq!(b.next_action(12), None);
        assert_eq!(b.next_action(9), None, "before the chunk starts");
        assert_eq!(b.underruns(), 3);
        assert_eq!(b.served(), 2);
        assert_eq!(b.underrun_rate(), Some(3.0 / 5.0));
    }

    #[test]
    fn a_later_chunk_wins_regardless_of_slot_order() {
        let mut b = ChunkBuffer::<1, H>::new(4, ChunkBlendPolicy::HardSwitch);
        b.push(&chunk([1.0, 1.0, 1.0, 1.0], 4), 0);
        b.push(&chunk([9.0, 9.0, 9.0, 9.0], 4), 2);
        assert_eq!(
            b.next_action(1),
            Some([1.0]),
            "only the first covers tick 1"
        );
        assert_eq!(b.next_action(2), Some([9.0]), "the newest wins the overlap");
        assert_eq!(b.next_action(3), Some([9.0]));
    }

    /// Hand-computed ACT ensembling, `m = 0.01` as in the paper.
    ///
    /// Chunk A starts at 0 with rows `[1, 2, 3, 4]`, chunk B starts at 2 with rows
    /// `[10, 20, 30, 40]`. At tick 2 the overlap is `{A[2] = 3, B[0] = 10}` with A older, so
    /// `w = (exp(0), exp(-0.01)) = (1, 0.99004983…)` and
    /// `out = (1*3 + 0.99004983*10) / (1 + 0.99004983) = 6.4832…`.
    #[test]
    fn temporal_ensembling_matches_the_hand_computed_weights() {
        let mut b = ChunkBuffer::<1, H>::new(2, ensemble(0.01));
        b.push(&chunk([1.0, 2.0, 3.0, 4.0], 4), 0);
        b.push(&chunk([10.0, 20.0, 30.0, 40.0], 4), 2);
        let w1 = f64::from(es_math::approx::exp(-0.01));
        let want = (3.0 + w1 * 10.0) / (1.0 + w1);
        let got = b.next_action(2).expect("both chunks cover tick 2")[0];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!((got - 6.4832).abs() < 1e-3, "sanity: {got}");
        // Tick 3: A[3] = 4 and B[1] = 20, same weights.
        let got = b.next_action(3).expect("still overlapping")[0];
        assert!((got - (4.0 + w1 * 20.0) / (1.0 + w1)).abs() < 1e-12);
        // Tick 4: A is exhausted, so B alone — the average degenerates to B[2].
        assert_eq!(b.next_action(4), Some([30.0]));
        // Ensembling reads all H rows, not just K: that is what ACT averages (§8.5).
        assert!(b.execute_chunk() < H);
    }

    #[test]
    fn ensembling_is_order_independent_of_slot_placement() {
        let rows = [
            ([1.0, 2.0, 3.0, 4.0], 0u64),
            ([5.0, 6.0, 7.0, 8.0], 1),
            ([9.0, 10.0, 11.0, 12.0], 2),
        ];
        let mut a = ChunkBuffer::<1, H>::new(4, ensemble(0.5));
        // Fill and wrap the ring so the same three chunks land in different slots.
        let mut c = ChunkBuffer::<1, H>::new(4, ensemble(0.5));
        for _ in 0..CHUNK_SLOTS {
            c.push(&chunk([0.0; H], 4), 100);
        }
        for (r, t) in rows {
            a.push(&chunk(r, 4), t);
            c.push(&chunk(r, 4), t);
        }
        for tick in 0..4 {
            assert_eq!(a.next_action(tick), c.next_action(tick), "tick {tick}");
        }
    }

    #[test]
    fn linear_blend_ramps_from_the_previous_chunk() {
        let mut b = ChunkBuffer::<1, H>::new(4, ChunkBlendPolicy::LinearBlend { steps: 2 });
        b.push(&chunk([0.0, 0.0, 0.0, 0.0], 4), 0);
        b.push(&chunk([4.0, 4.0, 4.0, 4.0], 4), 1);
        assert_eq!(b.next_action(0), Some([0.0]));
        assert_eq!(b.next_action(1), Some([2.0]), "halfway after one step");
        assert_eq!(b.next_action(2), Some([4.0]), "fully switched");
        assert_eq!(b.next_action(3), Some([4.0]));
    }

    #[test]
    fn an_empty_chunk_is_dropped_and_clear_forgets_everything() {
        let mut b = ChunkBuffer::<1, H>::new(4, ChunkBlendPolicy::HardSwitch);
        b.push(&ActionChunk::empty(ExecutionMode::RecedingHorizon), 0);
        assert!(b.is_empty(), "an empty chunk never occupies a slot");
        assert_eq!(b.next_action(0), None);
        b.push(&chunk([7.0; H], 4), 0);
        assert_eq!(b.next_action(0), Some([7.0]));
        b.clear();
        assert_eq!(b.next_action(0), None, "reset drops stale predictions");
        assert!(b.underrun_rate().is_some_and(|r| r > 0.0));
    }

    /// `action_at` is the lookahead `DomainRunner` builds one chunk per arrival from: same
    /// rows, but it must not move `served`/`underruns` — those are per control tick (§12.4).
    #[test]
    fn action_at_is_next_action_without_the_counters() {
        let mut b = ChunkBuffer::<1, H>::new(4, ensemble(0.3));
        assert_eq!(b.arrivals(), 0);
        b.push(&chunk([1.0, 2.0, 3.0, 4.0], 4), 5);
        assert_eq!(b.arrivals(), 1);
        for tick in 4..10 {
            assert_eq!(b.action_at(tick), b.action_at(tick), "pure");
        }
        assert_eq!((b.served(), b.underruns()), (0, 0));
        for tick in 4..10 {
            assert_eq!(b.next_action(tick), b.action_at(tick));
        }
        assert_eq!((b.served(), b.underruns()), (4, 2));
    }

    #[test]
    fn eviction_drops_the_oldest_arrival() {
        let mut b = ChunkBuffer::<1, H>::new(4, ChunkBlendPolicy::HardSwitch);
        for i in 0..=CHUNK_SLOTS {
            b.push(&chunk([i as f64; H], 4), 0);
        }
        assert_eq!(b.len(), CHUNK_SLOTS);
        // The first push is gone; the newest still wins.
        assert_eq!(b.next_action(0), Some([CHUNK_SLOTS as f64]));
    }
}
