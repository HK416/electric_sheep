//! [`EmbeddedRuntime`]: one `policy.esb`, one control tick at a time (spec 9.6).

use std::collections::BTreeMap;

use es_compile::bundle::{BundleError, PolicyBundle};
use es_compile::{CpuPlan, Tensor, TensorRef};
use es_core::PhysTick;
use es_ir::deployment::{ExecutionMode, Micros};
use es_ir::hash::{DatasetHash, HardwareCapability, HashChain};
use es_ir::types::ElemType;
use es_policy::{PolicyError, PolicyRuntime, WeightsSource};
use es_safety::{ActionChunk, SafeAction, SafetyConfigError, SafetyCounters, SafetyPlane};

use crate::hardware::hardware_capability;
use crate::ring::{TelemetryRing, TickRecord};

/// Telemetry depth. One second of history at 1 kHz, which is the highest control rate spec 9.2
/// contemplates; the ring is allocated once in `from_bundle` and never grows.
const TELEMETRY_TICKS: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("the bundle is not usable: {0}")]
    Bundle(#[from] BundleError),
    #[error("the deployment does not fit this runtime: {0}")]
    Safety(#[from] SafetyConfigError),
    #[error("loading the policy failed: {0}")]
    Policy(#[from] PolicyError),
    #[error("the learning graph declares {found} outputs; exactly one action output is required")]
    ActionOutputs { found: usize },
}

/// The deployment runtime of spec 9.6: compiled observation plan, policy runtime, Safety
/// Plane, telemetry ring. No physics, no renderer, no Python, no filesystem.
///
/// `NJ` is the joint count and `H` the prediction horizon; both are checked against the
/// Deployment IR at construction, so a bundle for a different robot cannot load.
///
/// **The plane is not optional** (`INV-12`). [`EmbeddedRuntime::from_bundle`] is the only
/// constructor, it always builds a [`SafetyPlane`], and [`tick`](Self::tick) is the only way
/// to get an action out — it returns [`SafeAction`], which only the plane can produce.
///
/// # Allocation
///
/// Everything is sized in `from_bundle`. Inside `tick`:
///
/// - a **reuse** tick (the buffered chunk still has executable rows, spec 8.6) allocates
///   nothing at all;
/// - a **replan** tick allocates in two places that this crate does not own: `CpuPlan::run`
///   takes a fresh arena per call and builds a borrowed input map, and `PolicyRuntime::infer`
///   is a trait boundary whose implementations own their buffers (the `Torch` one talks to a
///   subprocess). Spec 9.6's "zero heap allocation" is therefore met by the plane and the
///   chunk buffer, and the two named boundaries are what an M2 packet has to move.
pub struct EmbeddedRuntime<const NJ: usize, const H: usize> {
    bundle: PolicyBundle,
    plan: CpuPlan,
    plane: SafetyPlane<NJ, H>,
    policy: Box<dyn PolicyRuntime>,
    telemetry: TelemetryRing,
    hardware: HardwareCapability,
    /// The Learning IR's single output port: the action chunk (spec 8.5).
    action_out: String,
    mode: ExecutionMode,
    /// Control ticks between two inferences (spec 8.6).
    replan_every: u64,
    /// Actions consumed from the buffered chunk since it was produced.
    consumed: u64,
    chunk: ActionChunk<NJ, H>,
    started: bool,
}

impl<const NJ: usize, const H: usize> std::fmt::Debug for EmbeddedRuntime<NJ, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedRuntime")
            .field("action_out", &self.action_out)
            .field("replan_every", &self.replan_every)
            .field("consumed", &self.consumed)
            .field("telemetry_pushed", &self.telemetry.pushed())
            .finish_non_exhaustive()
    }
}

impl<const NJ: usize, const H: usize> EmbeddedRuntime<NJ, H> {
    /// Open a `policy.esb` and bring up everything it describes.
    ///
    /// `PolicyBundle::open` has already re-validated the IRs, re-run the cross-IR pass and
    /// checked every hash the manifest claims; what is left here is turning them into running
    /// objects. `SafetyPlane::from_ir` is what rejects a bundle whose joint count is not `NJ`
    /// or whose action horizon is not `H`.
    pub fn from_bundle(
        bytes: &[u8],
        mut policy: Box<dyn PolicyRuntime>,
    ) -> Result<Self, RuntimeError> {
        let bundle = PolicyBundle::open(bytes)?;
        let plan = bundle.compile_plan()?;
        let plane = SafetyPlane::<NJ, H>::from_ir(&bundle.deployment)?;
        policy.load(
            &bundle.learning,
            &WeightsSource::InMemory(bundle.weights.clone()),
        )?;

        if bundle.learning.outputs.len() != 1 {
            return Err(RuntimeError::ActionOutputs {
                found: bundle.learning.outputs.len(),
            });
        }
        let action_out = bundle.learning.outputs[0].name.clone();
        let mode = bundle.deployment.execution;
        let replan_every = replan_interval(&bundle);

        Ok(Self {
            plan,
            plane,
            policy,
            telemetry: TelemetryRing::with_capacity(TELEMETRY_TICKS),
            hardware: hardware_capability(),
            action_out,
            mode,
            replan_every,
            consumed: 0,
            chunk: ActionChunk::empty(mode),
            started: false,
            bundle,
        })
    }

    /// One control tick: observation plan -> policy -> chunk -> **Safety Plane** -> action.
    ///
    /// `sensors` is keyed by the plan's input names (the `StableId` hex of the sensor or body,
    /// as `CpuPlan::inputs` records them). `obs_age` is how old the sample is and `now` the
    /// control tick; both are the plane's watchdog inputs, not this function's.
    ///
    /// The policy runs only on a replan tick (spec 8.6); otherwise the buffered chunk is
    /// re-submitted and the plane advances its own cursor. Every failure on the way — a
    /// missing input, an inference error, an action tensor of the wrong shape — produces an
    /// **empty chunk**, which the plane turns into a chunk underrun and the configured
    /// fallback. There is no error return, because there is no tick without an action
    /// (`INV-13`).
    pub fn tick(
        &mut self,
        sensors: &BTreeMap<String, Tensor>,
        now: PhysTick,
        obs_age: Micros,
    ) -> SafeAction<NJ> {
        let replanned = !self.started || self.consumed >= self.replan_every;
        if replanned {
            self.chunk = self.infer(sensors);
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

    /// Run the observation plan and the policy, and pack the result into an action chunk.
    /// Anything that goes wrong becomes an empty chunk: the plane is the error handler.
    fn infer(&mut self, sensors: &BTreeMap<String, Tensor>) -> ActionChunk<NJ, H> {
        let inputs: BTreeMap<String, TensorRef<'_>> = sensors
            .iter()
            .map(|(name, t)| (name.clone(), t.as_ref()))
            .collect();
        let Ok(observation) = self.plan.run(&inputs) else {
            return ActionChunk::empty(self.mode);
        };
        let Ok(out) = self.policy.infer(&observation) else {
            return ActionChunk::empty(self.mode);
        };
        let Some(actions) = out.get(&self.action_out) else {
            return ActionChunk::empty(self.mode);
        };
        pack(actions, self.mode)
    }

    /// `H(task, observation, learning, policy, dataset, deployment, compiler, runtime,
    /// hardware_capability)` (spec 5.3), from the manifest plus what only this machine knows:
    /// the loaded backend's `runtime_hash` and the hardware capability.
    ///
    /// A slot the manifest leaves empty hashes as zeros. For a `policy.esb` that is `dataset`
    /// — training is not on the deployment path — so this value equals the training-time
    /// `execution_hash` only when the bundle carries a dataset hash, which an evidence bundle
    /// (spec 27.1) does and a deployment bundle does not.
    pub fn execution_hash(&self) -> [u8; 32] {
        let h = &self.bundle.manifest.hashes;
        let or_zero = |v: Option<[u8; 32]>| v.unwrap_or([0u8; 32]);
        let zero = DatasetHash {
            content: [0u8; 32],
            schema: [0u8; 32],
            split: [0u8; 32],
        };
        HashChain {
            asset: Vec::new(),
            scene: [0u8; 32],
            task_graph: [0u8; 32],
            task: or_zero(h.task),
            observation: or_zero(h.observation),
            learning: or_zero(h.learning),
            policy: or_zero(h.policy),
            dataset: h.dataset.unwrap_or(zero),
            deployment: or_zero(h.deployment),
            evaluation: None,
            compiler: or_zero(h.compiler),
            runtime: self.policy.runtime_hash(),
            hardware: self.hardware,
        }
        .execution_hash()
    }

    pub fn bundle(&self) -> &PolicyBundle {
        &self.bundle
    }

    pub fn telemetry(&self) -> &TelemetryRing {
        &self.telemetry
    }

    pub fn hardware(&self) -> HardwareCapability {
        self.hardware
    }

    /// Control ticks between two inferences (spec 8.6).
    pub fn replan_interval(&self) -> u64 {
        self.replan_every
    }

    pub fn counters(&self) -> &SafetyCounters {
        self.plane.counters()
    }

    /// Seed the plane with the measured joint state, so its hold target is where the robot
    /// actually is. Call before the first [`tick`](Self::tick).
    pub fn observe_state(&mut self, q: &[f64; NJ], qd: &[f64; NJ]) {
        self.plane.observe_state(q, qd);
    }

    /// Forwarded to the plane: the controller is alive as of `now` (spec 9.4).
    pub fn heartbeat(&mut self, now: PhysTick) {
        self.plane.heartbeat(now);
    }

    /// Forwarded to the plane: a sample from `sensor` arrived at `now` (spec 9.4).
    pub fn sensor_seen(&mut self, sensor: &str, now: PhysTick) {
        self.plane.sensor_seen(sensor, now);
    }

    /// Forwarded to the plane: clear the emergency-stop latch. An operator action — the plane
    /// never clears its own latch (spec 9.4).
    pub fn reset_latch(&mut self) {
        self.plane.reset_latch();
    }

    pub fn is_latched(&self) -> bool {
        self.plane.is_latched()
    }
}

/// Control ticks per inference.
///
/// `rate.control / rate.inference` is exact (both are rational, spec 3.4 forbids accumulating
/// a float period), and `XIR-023` has already checked that it agrees with
/// `PolicyContract::replanning_hz`. Capped by `execute_chunk`: rows past K are predictions,
/// not commands (spec 8.5), so replanning cannot be rarer than the chunk runs out.
fn replan_interval(bundle: &PolicyBundle) -> u64 {
    let r = bundle.deployment.rate;
    let ticks = (u128::from(r.control.num()) * u128::from(r.inference.den()))
        / (u128::from(r.control.den()) * u128::from(r.inference.num()));
    let ticks = u64::try_from(ticks).unwrap_or(1).max(1);
    ticks
        .min(bundle.deployment.action.execute_chunk as u64)
        .max(1)
}

/// `[rows, NJ]` f32 -> an action chunk. A tensor that is not that shape or dtype yields an
/// empty chunk, which the plane reports as an underrun.
fn pack<const NJ: usize, const H: usize>(t: &Tensor, mode: ExecutionMode) -> ActionChunk<NJ, H> {
    if t.dtype != ElemType::F32 || t.shape.len() != 2 || t.shape[1] != NJ as u64 {
        return ActionChunk::empty(mode);
    }
    let rows = (t.shape[0] as usize).min(H);
    if t.data.len() < rows * NJ * 4 {
        return ActionChunk::empty(mode);
    }
    let mut actions = [[0.0f64; NJ]; H];
    for (r, row) in actions.iter_mut().take(rows).enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            let at = (r * NJ + c) * 4;
            let bits = [t.data[at], t.data[at + 1], t.data[at + 2], t.data[at + 3]];
            *v = f64::from(f32::from_le_bytes(bits));
        }
    }
    ActionChunk::new(actions, rows, mode)
}
