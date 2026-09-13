//! Turning `PerturbationKind` (§10.2) into something that happens during a run.
//!
//! Every kind is resolved **once**, in [`PerturbationPlan::compile`], so the per-episode path
//! does no string matching and cannot fail. A kind this build has no kernel for is
//! [`EvalError::Unsupported`] at compile time, naming it — the same rule
//! `es_env::RandomizationPlan` follows for targets and `PhysicsBackend::load` follows for
//! scene features (§17.2). `docs/design/evaluation-execution.md` section 3 is the support
//! table and says what each unsupported kind is blocked on.
//!
//! Fairness (§10.4): every draw comes from `EnvRng::new(seed, suite_id, episode_idx, stream)`,
//! which is `TaskRng(seed_base, suite_id, episode_idx, stream)`. Nothing on this path reads
//! the policy, the backend, or the wall clock, so two runs of the same `evaluation_hash` draw
//! the same numbers.

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_env::rng::EnvRng;
use es_ir::evaluation::{CountRange, EvaluationIr, PerturbationKind, Range};
use es_ir::task::Distribution;
use es_physics_core::backend::ModelInfo;

use crate::EvalError;

/// Domain separator for perturbation RNG streams, so a `Perturbation::stream` index can never
/// collide with a Task IR `Randomization` stream name.
const STREAM_TAG: &str = "es.eval.perturbation";

fn stream_id(stream: u32) -> StableId {
    StableId::from_path(&format!("{STREAM_TAG}.{stream}"))
}

/// A per-step frame-drop process (§10.2 `frame_drop`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDrop {
    pub prob: f64,
    pub burst: CountRange,
    pub stream: StableId,
}

/// A per-step multiplicative actuator noise (§10.2 `torque_noise`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorqueNoise {
    pub rel_sigma: f64,
    pub stream: StableId,
}

/// What one episode of one cell was set up with.
///
/// Named for the hook it will become: when `es-env` grows a reset that takes state, the
/// `object_pose` group (section 3.2 of the design note) writes its `qpos` offsets here.
/// Today every state-mutating kind is `Unsupported`, so this carries only the knobs the step
/// loop reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResetOverrides {
    /// Observation age, in milliseconds, before the runner converts it to control steps.
    pub observation_delay_ms: u32,
    pub action_delay_ms: u32,
    pub frame_drop: Option<FrameDrop>,
    pub torque_noise: Option<TorqueNoise>,
    /// Actuator deadband in radians: a commanded change smaller than this does not move.
    pub backlash_rad: f64,
}

/// One perturbation, resolved against the streams it draws from.
#[derive(Clone, Debug, PartialEq)]
enum Entry {
    ObservationDelay { ms: Vec<f64>, stream: StableId },
    ActionDelay { ms: Vec<f64>, stream: StableId },
    FrameDrop(FrameDrop),
    TorqueNoise(TorqueNoise),
    Backlash { rad: Range, stream: StableId },
}

/// One row of the §10.1 table: the suite's name and its resolved perturbations.
#[derive(Clone, Debug, PartialEq)]
struct Cell {
    name: String,
    entries: Vec<Entry>,
}

/// Every suite of an [`EvaluationIr`], resolved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerturbationPlan {
    cells: Vec<Cell>,
}

impl PerturbationPlan {
    /// Resolves every perturbation of every suite.
    ///
    /// `scene` and `model` are not read today; they are in the signature because every kind in
    /// the scene-mutation group resolves a target against them the moment one is implemented,
    /// and changing the signature later would touch every caller.
    pub fn compile(
        ir: &EvaluationIr,
        _scene: &SceneDesc,
        _model: &ModelInfo,
    ) -> Result<Self, EvalError> {
        let mut cells = Vec::with_capacity(ir.suites.len());
        for suite in &ir.suites {
            let mut entries = Vec::with_capacity(suite.perturbations.len());
            for p in &suite.perturbations {
                entries.push(resolve(&p.kind, stream_id(p.stream))?);
            }
            cells.push(Cell {
                name: suite.name.clone(),
                entries,
            });
        }
        Ok(Self { cells })
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn suite_name(&self, cell: usize) -> &str {
        &self.cells[cell].name
    }

    /// Draws the per-episode knobs of one cell into `out`.
    ///
    /// `cell` is the `suite_id` of §10.4 and `episode` is `episode_idx`, so the draw is fixed
    /// by the evaluation document alone.
    pub fn apply_at_reset(&self, cell: usize, seed: u64, episode: u64, out: &mut ResetOverrides) {
        *out = ResetOverrides::default();
        let Some(c) = self.cells.get(cell) else {
            return;
        };
        let suite_id = cell as u32;
        for entry in &c.entries {
            match entry {
                Entry::ObservationDelay { ms, stream } => {
                    let mut rng = EnvRng::new(seed, suite_id, episode, *stream);
                    out.observation_delay_ms = rng.sample(&Distribution::Choice(ms.clone())) as u32;
                }
                Entry::ActionDelay { ms, stream } => {
                    let mut rng = EnvRng::new(seed, suite_id, episode, *stream);
                    out.action_delay_ms = rng.sample(&Distribution::Choice(ms.clone())) as u32;
                }
                Entry::Backlash { rad, stream } => {
                    let mut rng = EnvRng::new(seed, suite_id, episode, *stream);
                    out.backlash_rad = rng
                        .sample(&Distribution::Uniform {
                            lo: rad.lo,
                            hi: rad.hi,
                        })
                        .abs();
                }
                // Per-step processes: the parameters travel to the step loop, the draws happen
                // there so every step consumes from the same stream in order.
                Entry::FrameDrop(f) => out.frame_drop = Some(*f),
                Entry::TorqueNoise(n) => out.torque_noise = Some(*n),
            }
        }
    }
}

fn resolve(kind: &PerturbationKind, stream: StableId) -> Result<Entry, EvalError> {
    const NO_RENDERER: &str =
        "es-render (layer 5) is not implemented, so there is nothing to perturb";
    match kind {
        PerturbationKind::ObservationDelay { ms } => Ok(Entry::ObservationDelay {
            ms: ms.iter().map(|v| f64::from(*v)).collect(),
            stream,
        }),
        PerturbationKind::ActionDelay { ms } => Ok(Entry::ActionDelay {
            ms: ms.iter().map(|v| f64::from(*v)).collect(),
            stream,
        }),
        PerturbationKind::FrameDrop { prob, burst } => Ok(Entry::FrameDrop(FrameDrop {
            prob: *prob,
            burst: *burst,
            stream,
        })),
        PerturbationKind::TorqueNoise { rel_sigma } => Ok(Entry::TorqueNoise(TorqueNoise {
            rel_sigma: *rel_sigma,
            stream,
        })),
        PerturbationKind::Backlash { rad } => Ok(Entry::Backlash { rad: *rad, stream }),
        PerturbationKind::LightIntensity { .. }
        | PerturbationKind::LightDirection { .. }
        | PerturbationKind::ColorTemperature { .. }
        | PerturbationKind::Occluder { .. } => {
            Err(EvalError::unsupported(kind.name(), NO_RENDERER))
        }
        PerturbationKind::CameraExtrinsic { .. } | PerturbationKind::CameraIntrinsic { .. } => {
            Err(EvalError::unsupported(
                kind.name(),
                "no renderer, and INV-14 requires the ImageSpec intrinsics to move with the \
                 camera rather than be approximated",
            ))
        }
        PerturbationKind::ObjectPose { .. } => Err(EvalError::unsupported(
            kind.name(),
            "Env::reset takes no state override, so the initial pose cannot be written; the \
             hook is Env::reset_with in a follow-up es-env packet",
        )),
    }
}

/// The per-step state of one episode's perturbations.
///
/// Held by the runner, which owns the loop; `apply_per_step` lives here rather than on
/// [`PerturbationPlan`] because every one of these processes is stateful (a ring, a burst
/// counter, an RNG cursor) and a `&self` method would have nowhere to keep it.
#[derive(Clone, Debug)]
pub struct StepState {
    drop_rng: Option<EnvRng>,
    drop_prob: f64,
    burst: CountRange,
    drop_left: u32,
    noise_rng: Option<EnvRng>,
    rel_sigma: f64,
    backlash_rad: f64,
    /// Ring of commanded control vectors; `len = action_delay_steps + 1`.
    ring: Vec<Vec<f64>>,
    cursor: usize,
    /// Last control vector actually sent, for the backlash deadband.
    last_sent: Vec<f64>,
}

impl StepState {
    /// `hold` is the control vector the episode starts from: the action-delay ring is filled
    /// with it, so the delayed steps command the hold pose rather than zeros.
    pub fn new(
        ov: &ResetOverrides,
        action_delay_steps: usize,
        hold: &[f64],
        seed: u64,
        cell: usize,
        episode: u64,
    ) -> Self {
        let suite_id = cell as u32;
        Self {
            drop_rng: ov
                .frame_drop
                .map(|f| EnvRng::new(seed, suite_id, episode, f.stream)),
            drop_prob: ov.frame_drop.map_or(0.0, |f| f.prob),
            burst: ov.frame_drop.map_or(CountRange::new(1, 1), |f| f.burst),
            drop_left: 0,
            noise_rng: ov
                .torque_noise
                .map(|n| EnvRng::new(seed, suite_id, episode, n.stream)),
            rel_sigma: ov.torque_noise.map_or(0.0, |n| n.rel_sigma),
            backlash_rad: ov.backlash_rad,
            ring: vec![hold.to_vec(); action_delay_steps + 1],
            cursor: 0,
            last_sent: hold.to_vec(),
        }
    }

    /// Whether this control step's observation is dropped and the previous one reused
    /// (§10.2 `frame_drop`). Consumes one draw per step so the stream stays aligned with the
    /// step index whatever the outcome.
    pub fn drop_observation(&mut self) -> bool {
        let Some(rng) = self.drop_rng.as_mut() else {
            return false;
        };
        if self.drop_left > 0 {
            self.drop_left -= 1;
            return true;
        }
        if rng.next_f64() >= self.drop_prob {
            return false;
        }
        let span = u64::from(self.burst.hi.saturating_sub(self.burst.lo)) + 1;
        let len = self.burst.lo + (rng.next_u64() % span) as u32;
        // This step is the first of the burst.
        self.drop_left = len.saturating_sub(1);
        true
    }

    /// Applies action delay, then the backlash deadband, then actuator noise, in that order:
    /// the delay is a transport property of the command, the deadband and the noise are
    /// properties of the actuator that receives it.
    pub fn apply_per_step(&mut self, ctrl: &mut [f64]) {
        // Action delay: push the new command, emit the oldest.
        self.ring[self.cursor].clear();
        self.ring[self.cursor].extend_from_slice(ctrl);
        self.cursor = (self.cursor + 1) % self.ring.len();
        ctrl.copy_from_slice(&self.ring[self.cursor]);

        if self.backlash_rad > 0.0 {
            for (i, v) in ctrl.iter_mut().enumerate() {
                let prev = self.last_sent[i];
                if (*v - prev).abs() < self.backlash_rad {
                    *v = prev;
                }
            }
        }
        self.last_sent.copy_from_slice(ctrl);

        if let Some(rng) = self.noise_rng.as_mut() {
            for v in ctrl.iter_mut() {
                let n = rng.sample(&Distribution::Normal {
                    mean: 0.0,
                    std: self.rel_sigma,
                });
                *v *= 1.0 + n;
            }
        }
    }
}
