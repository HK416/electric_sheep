//! What a collection's caller hands [`Collector::run_perturbed`](super::Collector::run_perturbed):
//! the scripted intervener, the frame and event sinks, the perturbation hook and the observer --
//! closures all, none an extension point (`INV-17`) -- and the ring a perturbed observation is
//! held in.

use std::collections::BTreeMap;

use es_compile::Tensor;
use es_core::PhysTick;
use es_env::randomize::RenderOverrides;
use es_env::Termination;
use es_physics_core::backend::{ModelInfo, StateView};

use crate::intervention::ActionSourceCode;

/// What a scripted intervener does with one control tick.
///
/// [`Abort`](Self::Abort) exists because a scripted driver can run out of answers — a waypoint
/// outside the robot's workspace, say — and the honest record of that is a demonstration that
/// ended in failure, not one clamped to something reachable (spec 17.2).
#[derive(Clone, Debug, PartialEq)]
pub enum Intervention<const NJ: usize> {
    /// Leave this tick to the policy.
    Policy,
    /// Drive this tick with this action, held for the whole chunk horizon. It still travels
    /// chunk buffer -> `SafetyPlane` -> `ctrl` like any other (`INV-12`).
    Action([f64; NJ]),
    /// Drive the next ticks with these actions, one row per control tick. A row short of the
    /// horizon repeats the last one; rows past it are dropped. A scripted driver that paces
    /// itself to the Safety Plane's envelope emits a chunk rather than a step, because a step
    /// is what the envelope has to clamp (design note section 5.3).
    Chunk(Vec<[f64; NJ]>),
    /// Stop here: the episode is recorded as a failed demonstration.
    Abort,
}

/// The scripted-intervention hook: `Fn(episode, frame, &model, &obs) -> Intervention<NJ>`.
///
/// `obs` is the observation the policy was handed for that control tick (the `qpos ‖ qvel` row
/// of spec 12.2's raw path) and `model` says where each joint sits in it, so a scripted
/// intervener is a pure function of the state and the run stays bit-reproducible for a seed —
/// which is what makes it usable as an oracle.
pub type Intervener<'a, const NJ: usize> =
    &'a mut dyn FnMut(u32, u32, &ModelInfo, &[f64]) -> Intervention<NJ>;

/// Called once per control step with the state that step is entered with -- the same
/// instant the `observation.state` row and the `.estraj` pose of that step carry (M5/V12) --
/// and the env's render draws for the episode that step is in (`Env::render_overrides`,
/// packet M11/X5), so a frame is drawn under the draws the episode records.
///
/// A closure, not a renderer: `es-data` is layer 10 and `es-render` layer 5, and this is the
/// same trade `es_eval::runner::FrameSource` makes — the caller owns
/// `es_env::render::EnvRenderer` (feature `render`) and hands its `frames_with` in through
/// this, so nothing here links Vulkan. One call is one step of **every** camera: a task with
/// several image channels writes one frame per channel per call, each into its own directory
/// (packet M15/N2). With a sink, `info.json`'s video features stop being dangling references.
pub type FrameSink<'a> =
    &'a mut dyn FnMut(&ModelInfo, &StateView<'_>, &RenderOverrides) -> Result<(), String>;

/// One moment of a running collection, for a [`CollectSink`] (packet M7/E7).
///
/// The shape [`es_eval::runner::RunEvent`] has for an evaluation, for the same reason: a
/// collection that publishes what it is doing must publish what it *already had*, so a viewer
/// sees the plane's own verdict and not a second derivation of it. `es-data` links no
/// transport (layer 10 beside `es-telemetry`, spec 4.2 forbids the dependency); the caller
/// turns these into wire frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectEvent {
    /// One episode starts, with the seed the whole run was drawn from.
    EpisodeBegin { episode: u32, seed: u64 },
    /// One control tick, as the dataset's own `action_source` column records it and with the
    /// `es_safety::EventSet` bits of that step — the two counters' delta, which is exactly
    /// the set the plane raised (one env, one `validate` per step).
    Tick {
        episode: u32,
        frame: u32,
        tick: PhysTick,
        source: ActionSourceCode,
        events: u32,
    },
    /// The episode is over and its rows are written.
    EpisodeEnd {
        episode: u32,
        outcome: Termination,
        steps: usize,
    },
}

/// The collection sink: a closure, not an eighth extension point (`INV-17`). `None` is the
/// run of before — nothing is computed for a sink that is not there.
pub type CollectSink<'a> = &'a mut dyn FnMut(CollectEvent);

/// One moment of a collection under an Evaluation IR's perturbations (packet M13/Z2), for the
/// caller's [`Perturber`] to answer.
///
/// `es_eval::PerturbationPlan` draws them and is layer 10 beside this crate (spec 4.2 forbids
/// the dependency), so the caller owns the plan and its per-step state the way it owns the
/// renderer behind a [`FrameSink`]. The moments are the ones `es_eval::runner` applies the same
/// draws at: an episode's reset, the observation of a control step, and the plane's answer on
/// its way to the actuator -- *after* `SafetyPlane::validate`, which still runs on every step
/// (`INV-12`). The perturbation is the plant's, never the policy's.
#[derive(Debug)]
pub enum PerturbAt<'c> {
    /// Episode `episode` begins: draw what is fixed for it, and write how many control steps
    /// the policy's observation lags in it (§10.2 `observation_delay`).
    Episode {
        episode: u32,
        observation_delay: &'c mut usize,
    },
    /// A control step is about to be observed: write `true` when its observation is lost and
    /// the held one is handed on instead (§10.2 `frame_drop`).
    Observe { dropped: &'c mut bool },
    /// The plane's answer for this step, on its way to the actuator (§10.2 `action_delay`,
    /// `backlash`, `torque_noise`).
    Actuate(&'c mut [f64]),
}

/// The perturbation hook: a closure, not an eighth extension point (`INV-17`).
pub type Perturber<'a> = &'a mut dyn FnMut(PerturbAt<'_>);

/// The policy's observation of one control step (packet M16/H5):
/// `(first, previous, model, state) -> the policy's input tensors`.
///
/// Without one, the policy is handed the raw `qpos ‖ qvel` row under `"state"` -- the
/// plan-free path the scripted demonstrations are collected through, and what an
/// [`Intervener`] reads. A trained policy was trained on its Observation IR's output instead
/// (joint blocks, poses, velocities, the previous action, normalized and concatenated), which
/// the raw row is not: the Shadow Hand teacher wants 88 values and the row is 74. The caller
/// runs that plan here with the capture `es eval run` and `Rollout` use
/// (`es_eval::LiveObservation`, over `es_eval::runner::capture_at`), which `es-data` cannot
/// name (layer 10 beside `es-eval`).
///
/// `first` is the first observation of an episode (reset the plan's history); `previous` is
/// the policy row the last control step executed, before the plane -- what a
/// `PreviousAction` channel reads -- and `None` until this episode has one.
pub type Observer<'a> = &'a mut dyn FnMut(
    bool,
    Option<&[f64]>,
    &ModelInfo,
    &StateView<'_>,
) -> Result<BTreeMap<String, Tensor>, String>;

/// What [`Collector::run_perturbed`] is given beyond [`Collector::run_with_sink`]. What a
/// perturbation adds to the ledger step (`perturb.config`, ...) travels in `run_perturbed`'s
/// `ledger`, beside the expert's (packet M14/Q2).
///
/// [`Collector::run_perturbed`]: super::Collector::run_perturbed
/// [`Collector::run_with_sink`]: super::Collector::run_with_sink
pub struct Perturbation<'a> {
    pub hook: Perturber<'a>,
}

impl std::fmt::Debug for Perturbation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Perturbation").finish_non_exhaustive()
    }
}

/// What the policy observes on a perturbed control step: the ring `es_eval::runner` keeps for
/// §10.2 `observation_delay` and `frame_drop`, index rule for index rule. It holds raw `qpos`
/// / `qvel` rows rather than plan outputs because this path's observation *is* the raw row
/// (`DomainRunner::observe_window` with no plans).
#[derive(Debug, Default)]
pub(super) struct Held {
    delay: usize,
    ring: Vec<(Vec<f64>, Vec<f64>)>,
    cursor: usize,
}

impl Held {
    pub(super) fn new(delay: usize) -> Self {
        Self {
            delay,
            ..Self::default()
        }
    }

    /// Captures `state` unless its observation was dropped -- never the first one, there being
    /// nothing to hold yet -- and returns what the policy is handed this step.
    pub(super) fn observe(&mut self, state: &StateView<'_>, dropped: bool) -> StateView<'_> {
        if !dropped || self.ring.is_empty() {
            let row = (state.qpos_of(0).to_vec(), state.qvel_of(0).to_vec());
            if self.ring.len() <= self.delay {
                self.ring.push(row);
            } else {
                self.ring[self.cursor] = row;
                self.cursor = (self.cursor + 1) % self.ring.len();
            }
        }
        let oldest = if self.ring.len() > self.delay {
            self.cursor
        } else {
            0
        };
        let (qpos, qvel) = &self.ring[oldest];
        StateView {
            n_envs: 1,
            tick: state.tick,
            qpos,
            qvel,
            ..StateView::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packet M13/Z2: the observation a perturbed step hands on is `es_eval::runner`'s -- step
    /// `s` sees step `s - delay` once the ring is full, step 0 until then, and a dropped step
    /// reuses the held one without moving the ring.
    #[test]
    #[allow(clippy::float_cmp)] // exact: the ring hands on the captured value itself
    fn a_delayed_or_dropped_observation_is_the_evaluations() {
        let seen = |held: &mut Held, s: f64, dropped: bool| {
            let q = [s];
            let state = StateView {
                n_envs: 1,
                qpos: &q,
                qvel: &q,
                ..StateView::default()
            };
            held.observe(&state, dropped).qpos[0]
        };
        let mut held = Held::new(2);
        let got: Vec<f64> = (0..5)
            .map(|s| seen(&mut held, f64::from(s), false))
            .collect();
        assert_eq!(got, [0.0, 0.0, 0.0, 1.0, 2.0]);

        let mut held = Held::new(0);
        assert_eq!(
            seen(&mut held, 0.0, true),
            0.0,
            "the first step has nothing to hold"
        );
        assert_eq!(seen(&mut held, 1.0, false), 1.0);
        assert_eq!(
            seen(&mut held, 2.0, true),
            1.0,
            "a drop reuses the held observation"
        );
        assert_eq!(seen(&mut held, 3.0, false), 3.0);
    }
}
