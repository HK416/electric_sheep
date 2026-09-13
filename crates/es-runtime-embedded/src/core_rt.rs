//! [`EmbeddedCore`]: the control loop of spec 9.6 with nothing `std` in it.
//!
//! Spec 9.6 lists what ships on the robot: an Observation IR evaluator, the Learning IR's
//! pre/post-processing, a `PolicyRuntime`, the whole Safety Plane and a telemetry ring, in a
//! single static binary with zero heap allocation. The first three are *plug-ins*; the last
//! two plus the chunk bookkeeping are the loop itself, and that loop is what lives here:
//!
//! ```text
//! sensors -> [observe] -> obs -> [infer] -> ActionChunk -> SafetyPlane -> SafeAction
//!                                                             |
//!                                                         TickRecord -> ArrayRing
//! ```
//!
//! The two brackets are the plug-ins. On the embedded target they are plain function pointers
//! passed to [`EmbeddedCore::tick`] — no trait, no vtable, no `Box`, so nothing allocates and
//! INV-17's list of extension points is untouched. In the `std` build the same two steps are
//! `es_compile::CpuPlan` and `dyn es_policy::PolicyRuntime`, and [`EmbeddedRuntime`] drives
//! them itself and calls [`EmbeddedCore::step`] instead. One loop, two front ends: that is the
//! spec 9.5 identity claim in code.
//!
//! [`EmbeddedRuntime`]: crate::EmbeddedRuntime

use es_core::ring::ArrayRing;
use es_core::PhysTick;
use es_safety::{
    ActionChunk, ActionSource, ExecutionMode, Micros, SafeAction, SafetyConfig, SafetyCounters,
    SafetyPlane,
};

/// Telemetry depth. One second of history at 1 kHz, which is the highest control rate spec 9.2
/// contemplates; the ring is part of the struct and never grows.
pub const TELEMETRY_TICKS: usize = 1024;

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

/// The Observation IR evaluation step: sensor samples in, the policy's input tensor out.
///
/// Both buffers belong to the caller, so the step allocates nothing. In the `std` build this
/// is `es_compile::CpuPlan::run`; on the embedded target it is the compiled plan's entry point
/// (see `docs/design/embedded-runtime.md` for the `StaticPlan` that will fill this slot).
pub type ObserveFn = fn(sensors: &[f32], obs: &mut [f32]);

/// The inference step, including the Learning IR's pre/post-processing: observation tensor in,
/// action rows out, returning how many rows were filled (spec 8.5).
///
/// Returning the row count is what lets an inference failure be reported as `0` and turn into
/// the plane's chunk underrun and configured fallback, rather than into an error the caller
/// could ignore (INV-13).
pub type InferFn<const NJ: usize, const H: usize> =
    fn(obs: &[f32], actions: &mut [[f64; NJ]; H]) -> usize;

/// The deployment control loop: Safety Plane + chunk cursor + telemetry ring, no `std`, no
/// heap, no trait objects (spec 9.6, Appendix B.4).
///
/// `NJ` is the joint count, `H` the prediction horizon and `CAP` the telemetry depth; all
/// three are const generics because the whole thing is meant to be one `static` on a
/// microcontroller.
///
/// **The plane is not optional** (INV-12): [`EmbeddedCore::from_config`] is the only
/// constructor, it always builds a [`SafetyPlane`], and every way out of the loop returns a
/// [`SafeAction`], which only the plane can produce.
// Not `Clone`: a cloned control loop would fork the Safety Plane's state and its counters.
#[derive(Debug)]
pub struct EmbeddedCore<const NJ: usize, const H: usize, const CAP: usize = TELEMETRY_TICKS> {
    plane: SafetyPlane<NJ, H>,
    telemetry: ArrayRing<TickRecord, CAP>,
    /// The buffered chunk, re-submitted on every tick that does not replan (spec 8.6).
    chunk: ActionChunk<NJ, H>,
    mode: ExecutionMode,
    /// Control ticks between two inferences (spec 8.6).
    replan_every: u64,
    /// Actions consumed from the buffered chunk since it was produced.
    consumed: u64,
    /// Handed to the plane once per policy invocation, never per control tick (spec 8.6).
    seq: u64,
    started: bool,
}

impl<const NJ: usize, const H: usize, const CAP: usize> EmbeddedCore<NJ, H, CAP> {
    /// `replan_every` is control ticks per inference; `0` is read as `1`.
    pub fn from_config(config: &SafetyConfig<NJ>, mode: ExecutionMode, replan_every: u64) -> Self {
        Self::with_plane(SafetyPlane::from_config(config), mode, replan_every)
    }

    /// Wraps a plane the caller already built — what `EmbeddedRuntime` does with the one
    /// `SafetyPlane::from_ir` returns. A plane is only constructible from a complete
    /// [`SafetyConfig`], so this is not a way around INV-12.
    pub fn with_plane(plane: SafetyPlane<NJ, H>, mode: ExecutionMode, replan_every: u64) -> Self {
        Self {
            plane,
            telemetry: ArrayRing::new(),
            chunk: ActionChunk::empty(mode),
            mode,
            replan_every: replan_every.max(1),
            consumed: 0,
            seq: 0,
            started: false,
        }
    }

    /// Whether this tick must run the policy (spec 8.6).
    pub fn should_replan(&self) -> bool {
        !self.started || self.consumed >= self.replan_every
    }

    /// One control tick, given the chunk the caller's policy produced (`Some` on a replan tick,
    /// `None` to keep consuming the buffered one).
    ///
    /// **Cannot fail** (INV-13). A caller whose inference went wrong submits
    /// `Some(ActionChunk::empty(..))`, which the plane reports as a chunk underrun and turns
    /// into the configured fallback.
    pub fn step(
        &mut self,
        chunk: Option<ActionChunk<NJ, H>>,
        now: PhysTick,
        obs_age: Micros,
    ) -> SafeAction<NJ> {
        let replanned = chunk.is_some();
        if let Some(c) = chunk {
            self.seq += 1;
            self.chunk = c.with_seq(self.seq);
            self.consumed = 0;
            self.started = true;
        }
        let action = self.plane.validate(&self.chunk, obs_age, now);
        self.consumed += 1;
        self.telemetry.push(TickRecord {
            tick: now,
            source: action.source,
            events: action.events.bits(),
            replanned,
            obs_age,
        });
        action
    }

    /// One control tick, driving the two plug-ins itself: the `no_std` entry point.
    ///
    /// `obs` is the caller's scratch buffer for the observation tensor — sized once, reused
    /// every tick, never resized — so this path touches no allocator at all.
    pub fn tick(
        &mut self,
        sensors: &[f32],
        obs: &mut [f32],
        now: PhysTick,
        obs_age: Micros,
        observe: ObserveFn,
        infer: InferFn<NJ, H>,
    ) -> SafeAction<NJ> {
        let chunk = if self.should_replan() {
            observe(sensors, obs);
            let mut actions = [[0.0f64; NJ]; H];
            let valid = infer(obs, &mut actions);
            Some(ActionChunk::new(actions, valid, self.mode))
        } else {
            None
        };
        self.step(chunk, now, obs_age)
    }

    pub fn plane(&self) -> &SafetyPlane<NJ, H> {
        &self.plane
    }

    /// Mutable access for the operator-side calls — `observe_state`, `heartbeat`,
    /// `sensor_seen`, `reset_latch`. None of them can weaken the envelope (INV-12).
    pub fn plane_mut(&mut self) -> &mut SafetyPlane<NJ, H> {
        &mut self.plane
    }

    pub fn counters(&self) -> &SafetyCounters {
        self.plane.counters()
    }

    pub fn telemetry(&self) -> &ArrayRing<TickRecord, CAP> {
        &self.telemetry
    }

    /// Control ticks between two inferences (spec 8.6).
    pub fn replan_interval(&self) -> u64 {
        self.replan_every
    }

    /// The `seq` of the last chunk handed to the plane: one per policy invocation (spec 8.6).
    pub fn chunk_seq(&self) -> u64 {
        self.seq
    }
}
