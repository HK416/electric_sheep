//! Asynchronous inference, simulated deterministically (§8.6, §12.1, §12.3).
//!
//! Inference latency larger than the control period is the normal case, not an error (§8.6),
//! so the runtime models it. What it must **not** model is wall-clock: §12.3 is explicit that
//! the apply time of a chunk is decided by "which tick's observation produced it", never by
//! when it happened to arrive. So this pipeline is a queue over *ticks*:
//!
//! ```text
//! submit(env, t)  ──▶  queue  ──▶  poll(t') releases when t' >= t + latency_ticks
//! ```
//!
//! and `latency_ticks` comes from `RuntimeHints::expected_latency_ms` × the control rate, an
//! integer computed once. Two runs of the same seed therefore release the same submissions on
//! the same ticks on any machine, however fast or slow the host is.
//!
//! **Upgrade path to real threading.** A real inference thread belongs behind this same
//! interface: `submit` hands the batch to a worker and `poll` takes finished work, but in
//! *deterministic* mode a result that arrives early is still held until `submit_tick +
//! latency_ticks` (§12.3), and only real-time mode releases it immediately and records the
//! divergence in the replay. That is why the release rule lives here rather than in the
//! worker — swapping in a thread must not be able to change it. Deterministic first (§12.3);
//! threads are a throughput change, not a semantics change.

use std::collections::{BTreeMap, VecDeque};

use es_compile::Tensor;
use es_core::TickRate;

/// One observation waiting for (or released from) inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Submission {
    pub env: u32,
    /// Control tick whose observation this is — the `computed_from` of Appendix B.5.
    pub submit_tick: u64,
    /// Policy inputs, already preprocessed and named per `PolicyContract::inputs` (§8.7).
    pub inputs: BTreeMap<String, Tensor>,
}

/// Whole control ticks of latency for `expected_latency_ms` at `control` (§8.4, §12.3).
///
/// Rounded **up**: a policy that takes 1.2 control periods is late by two ticks, not one. The
/// `f32` millisecond field is converted to whole microseconds once, here, at the edge — no
/// float time is accumulated anywhere downstream (§18.1).
pub fn latency_ticks(expected_latency_ms: f32, control: TickRate) -> u64 {
    if !expected_latency_ms.is_finite() || expected_latency_ms <= 0.0 {
        return 0;
    }
    let micros = f64::from(expected_latency_ms) * 1000.0;
    let micros = micros.min(u64::MAX as f64 / 2.0) as u64;
    // ticks = ceil(micros * num / (den * 1e6)).
    micros
        .saturating_mul(control.num())
        .div_ceil(control.den().saturating_mul(1_000_000))
}

/// The deterministic simulated inference pipeline of §12.3.
///
/// FIFO by construction: submissions are released in submit order, and the caller submits in
/// ascending env id within a tick (§12.3, "inference batch composition is env-id ascending"),
/// so the released order is a pure function of the schedule.
#[derive(Clone, Debug)]
pub struct AsyncInference {
    latency_ticks: u64,
    batch: u32,
    queue: VecDeque<Submission>,
    submitted: u64,
    released: u64,
}

impl AsyncInference {
    /// `batch` is the inference domain's batch size (§12.1); it is clamped to at least 1,
    /// because a zero-width batch would never drain the queue.
    pub fn new(latency_ticks: u64, batch: u32) -> Self {
        Self {
            latency_ticks,
            batch: batch.max(1),
            queue: VecDeque::new(),
            submitted: 0,
            released: 0,
        }
    }

    pub fn latency_ticks(&self) -> u64 {
        self.latency_ticks
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn submitted(&self) -> u64 {
        self.submitted
    }

    pub fn released(&self) -> u64 {
        self.released
    }

    pub fn submit(&mut self, env: u32, submit_tick: u64, inputs: BTreeMap<String, Tensor>) {
        self.queue.push_back(Submission {
            env,
            submit_tick,
            inputs,
        });
        self.submitted += 1;
    }

    /// Releases at most `batch` submissions whose latency has elapsed by `tick`, in submit
    /// order.
    ///
    /// A submission that is ready but does not fit in this batch stays at the head of the
    /// queue and goes out first next time — the queue never reorders, so back-pressure delays
    /// work without changing what the policy sees or in which order.
    pub fn poll(&mut self, tick: u64) -> Vec<Submission> {
        let mut out = Vec::new();
        while out.len() < self.batch as usize {
            let ready = self
                .queue
                .front()
                .is_some_and(|s| s.submit_tick.saturating_add(self.latency_ticks) <= tick);
            if !ready {
                break;
            }
            out.push(self.queue.pop_front().expect("front was just inspected"));
        }
        self.released += out.len() as u64;
        out
    }

    /// Drops queued work for `env` — its episode ended, so the result would be applied to a
    /// state that no longer exists.
    pub fn drop_env(&mut self, env: u32) {
        self.queue.retain(|s| s.env != env);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_ir::types::ElemType;

    fn obs(v: f32) -> BTreeMap<String, Tensor> {
        [(
            "state".to_owned(),
            Tensor {
                dtype: ElemType::F32,
                shape: vec![1],
                data: v.to_le_bytes().to_vec(),
            },
        )]
        .into_iter()
        .collect()
    }

    /// The full release trace of a run: `(release_tick, env, submit_tick)`.
    fn run(batch: u32, latency: u64, ticks: u64, envs: u32) -> Vec<(u64, u32, u64)> {
        let mut inf = AsyncInference::new(latency, batch);
        let mut trace = Vec::new();
        for t in 0..ticks {
            for env in 0..envs {
                inf.submit(env, t, obs(t as f32));
            }
            for s in inf.poll(t) {
                trace.push((t, s.env, s.submit_tick));
            }
        }
        trace
    }

    #[test]
    fn latency_is_ticks_and_rounds_up() {
        let khz = TickRate::hz(1000);
        assert_eq!(latency_ticks(5.0, khz), 5, "ACT: 5 ms at 1 kHz");
        assert_eq!(latency_ticks(369.8, khz), 370, "pi-zero, §8.4");
        assert_eq!(
            latency_ticks(0.5, TickRate::hz(100)),
            1,
            "rounds up, never 0"
        );
        assert_eq!(latency_ticks(0.0, khz), 0);
        assert_eq!(latency_ticks(f32::NAN, khz), 0, "no panic on a bad hint");
        assert_eq!(
            latency_ticks(20.0, TickRate::rational(30000, 1001).unwrap()),
            1
        );
    }

    #[test]
    fn a_result_is_released_exactly_latency_ticks_after_submission() {
        let trace = run(8, 3, 8, 2);
        // With batch >= the arrival rate nothing queues up, so release == submit + 3.
        assert!(
            trace.iter().all(|(rel, _, sub)| *rel == sub + 3),
            "{trace:?}"
        );
        // Within a tick, env order is ascending (§12.3).
        assert_eq!(trace[0], (3, 0, 0));
        assert_eq!(trace[1], (3, 1, 0));
    }

    #[test]
    fn the_released_order_does_not_depend_on_the_batch_size() {
        let order = |batch| -> Vec<(u32, u64)> {
            run(batch, 2, 40, 4)
                .into_iter()
                .map(|(_, env, sub)| (env, sub))
                .collect()
        };
        // A batch narrower than the arrival rate cannot drain the queue inside the run, so it
        // gets *through less* — but what it does get through is the same sequence, in the same
        // order, never a reshuffle. That is the invariant: composition is schedule-determined,
        // batch size only rations throughput.
        let reference = order(16);
        for batch in [1, 2, 3, 4, 8, 16] {
            let got = order(batch);
            assert!(reference.starts_with(&got), "batch {batch}: {got:?}");
        }
        assert_eq!(order(4), reference, "a balanced batch keeps up exactly");
        // A batch smaller than the arrival rate delays work; it never reorders or drops it.
        let small = run(1, 2, 40, 4);
        let big = run(16, 2, 40, 4);
        assert!(
            small.len() < big.len(),
            "a narrower batch gets through less"
        );
        assert!(
            small.iter().zip(&big).all(|(s, b)| s.0 >= b.0),
            "a narrower batch is never earlier"
        );
    }

    #[test]
    fn nothing_is_released_before_its_time_and_a_reset_drops_its_work() {
        let mut inf = AsyncInference::new(5, 4);
        inf.submit(0, 10, obs(1.0));
        inf.submit(1, 10, obs(2.0));
        assert!(inf.poll(14).is_empty(), "one tick short");
        assert_eq!(inf.pending(), 2);
        inf.drop_env(1);
        let out = inf.poll(15);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].env, 0);
        assert_eq!((inf.submitted(), inf.released()), (2, 1));
        assert_eq!(inf.pending(), 0);
    }

    #[test]
    fn a_run_replays_bitwise() {
        assert_eq!(run(3, 7, 50, 5), run(3, 7, 50, 5));
    }
}
