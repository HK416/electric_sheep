//! The learning loop's collect and distill steps, and the `loop.jsonl` ledger that links
//! them (spec 13.1, spec 13.3).
//!
//! Read `docs/design/learning-loop.md` first. Two of its rules are the whole module:
//!
//! - **The Safety Plane is on the only actuator path** (`INV-12`). [`Collector::run`] drives
//!   the phases of [`es_env::Env::step_with_policy`], and an injected human action is wired
//!   in as a `PolicyRuntime` wrapper — so it becomes *just another chunk*, bounded by the same
//!   envelope and counted by the same counters as a policy chunk. There is no collect-mode
//!   branch around `validate`, tests included.
//! - **The training run is not here.** [`distill`] produces the *input identity* of a
//!   training run — merged dataset, deterministic split, `training_identity.json` — and the
//!   slots this side of the boundary cannot know are all-zero digests, never fabricated
//!   (spec 19.3, spec 2.3: training is on the Python path).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use es_assets::scene::SceneDesc;
use es_compile::{PolicyBundle, Tensor};
use es_core::TickRate;
use es_env::scheduler::BatchDomains;
use es_env::traj::Trajectory;
use es_env::{DomainRunner, Env, Termination};
use es_ir::learning::{ChunkBlendPolicy, LearningGraph, LearningNode};
use es_ir::types::ElemType;
use es_physics_core::backend::{ModelInfo, PhysicsBackend};
use es_policy::{PolicyError, PolicyInfo, PolicyRuntime, WeightsSource};
use es_safety::SafetyPlane;

use crate::intervention::{identity_of, segments_of, InterventionSegment, InterventionSource};
use crate::lerobot::meta::Info;
use crate::lerobot::{LeRobotDataset, LeRobotWriter};
use crate::DataError;

mod distillation;
mod hooks;
mod ledger;
mod record;

pub use distillation::{
    distill, successful_episodes, SplitSpec, OUTCOMES_FILE, PERTURBATIONS_FILE,
    TRAINING_IDENTITY_FILE,
};
pub use hooks::{
    CollectEvent, CollectSink, FrameSink, Intervener, Intervention, Observer, PerturbAt,
    Perturbation, Perturber,
};
pub use ledger::{
    append_loop_step, check_chain, last_evaluation_hash, read_loop_steps, LoopKind, LoopStep,
    CHECKPOINT, LOOP_FILE,
};
pub use record::ACTION_COMMANDED;

use distillation::write_outcomes;
use hooks::Held;
use record::{classify, collect_features, counters_of, raised, to_lerobot, Recorded};

pub(crate) fn hex(d: &[u8; 32]) -> String {
    d.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

// --- collection -----------------------------------------------------------------------------

/// What [`Collector::run`] is given that the bundle does not say.
#[derive(Debug)]
pub struct CollectSpec<'a> {
    pub bundle: &'a PolicyBundle,
    pub scene: &'a SceneDesc,
    pub n_episodes: u32,
    pub seed: u64,
    /// Episode step budget; `0` uses the task's own `max_episode_steps`.
    pub max_steps: u32,
    pub out_root: &'a Path,
    /// Where to write each episode's `.estraj` state trajectory (`es_env::traj`, packet
    /// M5/V9). `None` writes none. Not inside the `LeRobot` dataset's own files: nothing in
    /// `dataset_content_hash` or `dataset_schema_hash` moves because of it.
    pub traj_dir: Option<PathBuf>,
}

/// What one collect run produced.
#[derive(Clone, Debug)]
pub struct CollectReport {
    pub root: PathBuf,
    pub episodes: u32,
    pub frames: u64,
    pub intervention_frames: u64,
    pub segments: Vec<InterventionSegment>,
    /// How each episode ended, in episode order — the demonstration's own verdict, which the
    /// `LeRobot` columns have no place for.
    pub terminations: Vec<Termination>,
    /// Frames handed to the [`FrameSink`], if there was one.
    pub rendered: u64,
    /// What the Safety Plane did across the run: spec 10.3's `failure_mode_histogram`, plus
    /// the clamp and fallback counts `action_source` is classified from.
    pub safety: es_safety::SafetyCounters,
    pub content: [u8; 32],
    pub schema: [u8; 32],
    /// Honest gaps, one line each — currently the image channels for which `info.json`
    /// declares a video feature but no mp4 was written (design note section 3.1).
    pub warnings: Vec<String>,
}

/// A `PolicyRuntime` that lets a scripted intervener take over a control tick.
///
/// The injected action is emitted as the policy's action chunk, so it travels the ordinary
/// path: chunk buffer -> `SafetyPlane::validate` -> `ctrl`. Nothing is bypassed (`INV-12`).
struct Intervened<'a, const NJ: usize, const H: usize> {
    inner: &'a mut dyn PolicyRuntime,
    intervener: Intervener<'a, NJ>,
    action_port: String,
    /// The loaded model, so the intervener can find a joint inside the observation row.
    model: ModelInfo,
    episode: u32,
    frame: u32,
    /// Set by the last `infer` call: did the intervener take this tick?
    injected: bool,
    /// Set by the last `infer` call: did the intervener give up?
    aborted: bool,
    /// The last chunk the intervener asked for, repeated on an abort.
    last: Vec<[f64; NJ]>,
}

impl<const NJ: usize, const H: usize> Intervened<'_, NJ, H> {
    /// Episode `episode` begins as the first one did: nothing the last episode's driver said
    /// carries into it (packet P-M14-R1). The three flags are set by `infer`, which runs on the
    /// inference tick -- with a declared latency never on frame 0 -- so an abort left set here
    /// ended every later episode after one frame, and a stale `injected` marked frames no chunk
    /// of this episode drove as the human's.
    fn begin_episode(&mut self, episode: u32) {
        self.episode = episode;
        self.frame = 0;
        self.injected = false;
        self.aborted = false;
        self.last = vec![[0.0; NJ]];
    }
}

impl<const NJ: usize, const H: usize> PolicyRuntime for Intervened<'_, NJ, H> {
    fn load(
        &mut self,
        graph: &LearningGraph,
        weights: &WeightsSource,
    ) -> Result<PolicyInfo, PolicyError> {
        self.inner.load(graph, weights)
    }

    fn infer(
        &mut self,
        inputs: &BTreeMap<String, Tensor>,
    ) -> Result<BTreeMap<String, Tensor>, PolicyError> {
        let obs = inputs.values().next().map_or_else(Vec::new, tensor_to_f64);
        self.injected = false;
        let taken = (self.intervener)(self.episode, self.frame, &self.model, &obs);
        self.aborted = taken == Intervention::Abort;
        // A driver that gave up repeats what it last asked for -- through the plane, like every
        // other chunk (`INV-12`) -- and the collector ends the episode on the flag above. The
        // alternative, an empty chunk, is a watchdog event the demonstration did not have.
        let rows = match taken {
            Intervention::Action(a) => Some(vec![a]),
            Intervention::Chunk(rows) if !rows.is_empty() => Some(rows),
            // An empty chunk is nothing to execute, which is the policy's tick.
            Intervention::Policy | Intervention::Chunk(_) => None,
            Intervention::Abort => Some(self.last.clone()),
        };
        if let Some(rows) = &rows {
            if !self.aborted {
                self.injected = true;
                self.last.clone_from(rows);
            }
        }
        if let Some(rows) = rows {
            let last = *rows.last().unwrap_or(&[0.0; NJ]);
            let mut data = Vec::with_capacity(H * NJ * 4);
            for k in 0..H {
                for v in rows.get(k).unwrap_or(&last) {
                    data.extend_from_slice(&(*v as f32).to_le_bytes());
                }
            }
            return Ok(BTreeMap::from([(
                self.action_port.clone(),
                Tensor {
                    dtype: ElemType::F32,
                    shape: vec![H as u64, NJ as u64],
                    data,
                },
            )]));
        }
        // A policy's only output is its action chunk, whatever the Learning IR names it -- every
        // committed one says "actions" -- which is `es eval run`'s rule (`RunConfig::
        // action_output` = `None`) too (packet M16/H5).
        let mut out = self.inner.infer(inputs)?;
        if out.len() == 1 && !out.contains_key(&self.action_port) {
            let only = out.pop_first().map(|(_, t)| t).expect("len == 1");
            out.insert(self.action_port.clone(), only);
        }
        Ok(out)
    }

    fn info(&self) -> Option<&PolicyInfo> {
        self.inner.info()
    }

    fn runtime_hash(&self) -> [u8; 32] {
        self.inner.runtime_hash()
    }
}

fn tensor_to_f64(t: &Tensor) -> Vec<f64> {
    match t.dtype {
        ElemType::F32 => t
            .data
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        ElemType::F64 => t
            .data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        _ => Vec::new(),
    }
}

/// The blend the Learning IR's `ActionChunker` declares (spec 8.5). A graph without one
/// switches hard, which is the behaviour of a policy that replans every chunk.
fn chunk_blend(learning: &LearningGraph) -> ChunkBlendPolicy {
    learning
        .nodes
        .nodes
        .values()
        .find_map(|n| match n {
            LearningNode::ActionChunker { blend, .. } => Some(*blend),
            _ => None,
        })
        .unwrap_or(ChunkBlendPolicy::HardSwitch)
}

/// Namespace for [`Collector::run`]. Not a trait and not state: `INV-17` allows seven
/// extension points and this is none of them.
#[derive(Debug)]
pub struct Collector;

impl Collector {
    /// Rolls out `n_episodes` episodes and writes them as a `LeRobot` dataset (spec 13.1).
    ///
    /// `NJ` / `H` are the deployment's joint count and chunk horizon; `SafetyPlane::from_ir`
    /// and `DomainRunner::new` both reject a mismatch, so they cannot silently disagree with
    /// the bundle.
    pub fn run<B, F, const NJ: usize, const H: usize>(
        spec: &CollectSpec<'_>,
        policy: &mut dyn PolicyRuntime,
        new_backend: F,
        intervener: Intervener<'_, NJ>,
        frame_sink: Option<FrameSink<'_>>,
    ) -> Result<CollectReport, DataError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        Self::run_with_sink::<B, F, NJ, H>(spec, policy, new_backend, intervener, frame_sink, None)
    }

    /// [`Self::run`] with a [`CollectSink`]: the same run, saying what it does as it does it
    /// (packet M7/E7).
    ///
    /// With `None` this is the run of before, byte for byte — nothing is computed for a sink
    /// that is not there, and the two per-step numbers a sink is given (the source code and
    /// the violation bits) come out of counters this loop already snapshots.
    pub fn run_with_sink<B, F, const NJ: usize, const H: usize>(
        spec: &CollectSpec<'_>,
        policy: &mut dyn PolicyRuntime,
        new_backend: F,
        intervener: Intervener<'_, NJ>,
        frame_sink: Option<FrameSink<'_>>,
        sink: Option<CollectSink<'_>>,
    ) -> Result<CollectReport, DataError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        Self::run_perturbed::<B, F, NJ, H>(
            spec,
            policy,
            new_backend,
            intervener,
            frame_sink,
            sink,
            None,
            None,
            &[],
        )
    }

    /// [`Self::run_with_sink`] under an Evaluation IR's perturbations (packet M13/Z2): the
    /// [`PerturbAt`] moments go to `perturb.hook`. `None` is the run of before, byte for byte.
    ///
    /// The frames, the `.estraj` poses and the `observation.state` rows stay what the world
    /// was at each step -- a camera sees the scene, whatever reaches the policy late -- and
    /// the `action` column stays the plane's answer: a delayed, deadbanded or noisy actuator
    /// is the plant, and a demonstration labelled with it would teach the plant's error.
    ///
    /// `ledger` is the extra inputs of this run's `collect` ledger step: the expert and the
    /// blake3 of its program (packet M14/Q2), the perturbation's `perturb.config`, ... (packet
    /// M13/Z2). Provenance, which like every ledger value feeds no hash (spec 13.3); `&[]` is
    /// the step of before.
    #[allow(clippy::too_many_arguments)]
    pub fn run_perturbed<B, F, const NJ: usize, const H: usize>(
        spec: &CollectSpec<'_>,
        policy: &mut dyn PolicyRuntime,
        mut new_backend: F,
        intervener: Intervener<'_, NJ>,
        mut frame_sink: Option<FrameSink<'_>>,
        mut sink: Option<CollectSink<'_>>,
        mut perturb: Option<Perturbation<'_>>,
        mut observe: Option<Observer<'_>>,
        ledger: &[(String, String)],
    ) -> Result<CollectReport, DataError>
    where
        B: PhysicsBackend,
        F: FnMut() -> B,
    {
        let bundle = spec.bundle;
        let deploy = &bundle.deployment;
        let contract = &bundle.learning.policy.contract;
        let bad = |e: &dyn std::fmt::Display| DataError::Loop(e.to_string());

        // One control step is one control period (packet M5/V11): the schedule is derived from
        // the scene's own timestep and the Deployment IR's `rate.control`, so a recorded row is
        // 1 / `rate.control` seconds of simulated time and the dataset's `fps` below is true.
        let domains = BatchDomains::single_env_at(
            TickRate::from_period_secs(spec.scene.options.timestep).map_err(|e| bad(&e))?,
            deploy.rate.control,
        )
        .map_err(|e| bad(&e))?;
        let mut env: Env<B> =
            Env::new(&bundle.task, spec.scene, new_backend(), &domains, spec.seed)
                .map_err(|e| bad(&e))?;
        let (nq, nv, nu) = {
            let m = env.model();
            (m.nq as usize, m.nv as usize, m.nu as usize)
        };
        if nu != NJ {
            return Err(DataError::Loop(format!(
                "the deployment declares {NJ} joints and the loaded model has {nu} actuators"
            )));
        }
        let control = deploy.rate.control;
        // `DomainRunner` reads both of the Deployment IR's rates: it replans every
        // `rate.control / rate.inference` control ticks and the chunk drives the ticks in
        // between (packet M5/V17). A rate that does not divide is refused here, by name.
        let mut runner = DomainRunner::<NJ, H>::new(
            env.schedule(),
            contract,
            chunk_blend(&bundle.learning),
            deploy.rate,
        )
        .map_err(|e| bad(&e))?
        // The deployment's action space decides whether a chunk row is a target or an
        // increment the runner integrates before the plane sees it (packet M9/T1, spec 8.5);
        // absent, `JointPosition`, is the runner's own default.
        .with_action_space(deploy.action.space);
        let latency = runner.inference().latency_ticks() as usize;
        let execute = (contract.execute_chunk as usize).clamp(1, H.max(1));
        let mut planes = vec![SafetyPlane::<NJ, H>::from_ir(deploy).map_err(|e| bad(&e))?];

        let max_steps = if spec.max_steps == 0 {
            bundle.task.config.max_episode_steps
        } else {
            spec.max_steps
        };
        let fps = control.num() as f64 / control.den() as f64;
        let (features, warnings) = collect_features(bundle, nq + nv, nu, frame_sink.is_some());
        let task_name = format!(
            "es:task:{}",
            hex(&bundle.task.task_hash().map_err(DataError::Canon)?)
        );

        let mut wrapper = Intervened::<NJ, H> {
            inner: policy,
            intervener,
            action_port: "action".to_owned(),
            model: env.model().clone(),
            episode: 0,
            frame: 0,
            injected: false,
            aborted: false,
            last: vec![[0.0; NJ]],
        };

        let mut writer = LeRobotWriter::create(spec.out_root, Info::new(fps, features))?;
        let mut segments = Vec::new();
        let mut frames = 0u64;
        let mut intervention_frames = 0u64;
        let mut terminations = Vec::with_capacity(spec.n_episodes as usize);
        let mut rendered = 0u64;

        for index in 0..spec.n_episodes {
            wrapper.begin_episode(index);
            if let Some(sink) = sink.as_deref_mut() {
                sink(CollectEvent::EpisodeBegin {
                    episode: index,
                    seed: spec.seed,
                });
            }
            let mut delay = 0;
            if let Some(p) = perturb.as_mut() {
                (p.hook)(PerturbAt::Episode {
                    episode: index,
                    observation_delay: &mut delay,
                });
            }
            let mut held = Held::new(delay);
            // The policy row the last control step executed, for an [`Observer`].
            let mut previous: Option<[f64; NJ]> = None;
            let mut traj = spec.traj_dir.as_ref().map(|_| Trajectory::new(env.model()));
            let mut sources: Vec<i64> = Vec::with_capacity(max_steps as usize);
            let mut commanded: Vec<f64> = Vec::with_capacity(max_steps as usize * NJ);
            // The plane's answer per frame: the `action` column (see `run_perturbed`).
            let mut answered: Vec<f64> = Vec::with_capacity(max_steps as usize * NJ);
            let mut human = vec![false; max_steps as usize + latency + execute + 1];
            let mut closed = None;
            let mut aborted = false;
            for frame in 0..max_steps {
                wrapper.frame = frame;
                // Every consumer of the plane observes before every `validate` and none of
                // them decides what that means: `SafetyPlane::observe_state` seeds the
                // command chain on the first call after `begin_episode` and ignores the rest,
                // so collection, evaluation, HIL and the embedded runtime all measure the
                // envelope against the same thing -- the plane's own commands (packet M5/V6,
                // design note section 7.12). Before V6 this was `if frame == 0` here and
                // unconditional in `es_eval::runner`, and the two paths disagreed about what
                // `velocity_max` bounds.
                {
                    let state = env.backend().state();
                    let (mut q, mut qd) = ([0.0; NJ], [0.0; NJ]);
                    q.copy_from_slice(&state.qpos_of(0)[..NJ]);
                    qd.copy_from_slice(&state.qvel_of(0)[..NJ]);
                    planes[0].observe_state(&q, &qd);
                    // Image, `.estraj` pose and `observation.state` row are one instant: the
                    // state this control step is entered with, which is the state the expert
                    // is about to compute its action from and the state the policy will be
                    // handed at inference (packet M5/V12, design note section 7.20).
                    // `Env::step` records the same instant, so trajectory index, frame index
                    // and row index still agree -- and the terminal step no longer renders
                    // the *next* episode's reset state, which is what the post-step read did
                    // once `Env::step` auto-reset a done env. The state the episode ended in
                    // is the trajectory's one extra row, pushed when it closes.
                    if let Some(t) = traj.as_mut() {
                        t.push(env.model(), &state, 0).map_err(|e| bad(&e))?;
                    }
                    if let Some(sink) = frame_sink.as_deref_mut() {
                        sink(env.model(), &state, env.render_overrides(0))
                            .map_err(DataError::Loop)?;
                        rendered += 1;
                    }
                }
                let before = counters_of(&planes[0]);
                // Only when someone is watching: the per-kind array is 14 words and the
                // stream-2 sample is the only reader of it (packet M7/E7).
                let kinds_before = sink.as_ref().map(|_| planes[0].counters().violations);
                let tick = env.tick();
                // `Env::step_with_policy`, phase for phase, opened up where a perturbation
                // enters (packet M13/Z2): observe -> infer -> chunk buffer -> SafetyPlane ->
                // plant. The plane is on the path either way (`INV-12`).
                {
                    let state = env.backend().state();
                    let delayed;
                    let seen = match perturb.as_mut() {
                        None => &state,
                        Some(p) => {
                            let mut dropped = false;
                            (p.hook)(PerturbAt::Observe {
                                dropped: &mut dropped,
                            });
                            delayed = held.observe(&state, dropped);
                            &delayed
                        }
                    };
                    let model = env.model();
                    match observe.as_deref_mut() {
                        None => runner.observe_window(tick.0, model, seen, &mut []),
                        // ponytail: under `observation_delay` / `frame_drop` this re-captures
                        // the held state with the current previous action and advances the
                        // plan's history, where `es_eval::runner` replays the held plan output;
                        // equal for a plan without a TemporalWindow or PreviousAction input.
                        Some(observe) => runner.observe_window_with(tick.0, &mut |_| {
                            observe(frame == 0, previous.as_ref().map(|r| &r[..]), model, seen)
                                .map_err(|e| {
                                    es_env::EnvError::Unsupported(format!("observation: {e}"))
                                })
                        }),
                    }
                    .map_err(|e| bad(&e))?;
                }
                runner
                    .infer_window(tick.0, &mut wrapper)
                    .map_err(|e| bad(&e))?;
                // This step's policy row before the plane, which the next observation reads
                // (`es_eval::runner::run_episode` reads `action_at` at the same point). An
                // underrun serves no row and the last one stands.
                if observe.is_some() {
                    if let Some(row) = runner
                        .buffer(0)
                        .and_then(|b| b.action_at(runner.control_tick()))
                    {
                        previous = Some(row);
                    }
                }
                let mut ctrl = [0.0; NJ];
                runner
                    .emit_actions(&mut planes, &mut ctrl)
                    .map_err(|e| bad(&e))?;
                answered.extend_from_slice(&ctrl);
                if let Some(p) = perturb.as_mut() {
                    (p.hook)(PerturbAt::Actuate(&mut ctrl));
                }
                let outcome = env.step(&ctrl).map_err(|e| bad(&e))?;
                if outcome.dones.first() == Some(&true) {
                    runner.reset_env(0);
                }
                runner.advance();
                // Provenance for the row `env.step` just recorded: what the plane was asked
                // for, beside what it allowed (spec 13.2).
                commanded.extend_from_slice(&runner.commanded()[..NJ]);
                // An injection decided at tick `t` reaches the actuator at `t + latency` and
                // drives `execute_chunk` ticks (design note section 5.1) -- so mark first,
                // then classify this frame against the marks.
                if wrapper.injected {
                    let start = frame as usize + latency;
                    for f in start..start + execute {
                        if let Some(slot) = human.get_mut(f) {
                            *slot = true;
                        }
                    }
                }
                let after = counters_of(&planes[0]);
                let source = classify(before, after, human[frame as usize]);
                sources.push(source.as_i64());
                if let (Some(sink), Some(kinds_before)) = (sink.as_deref_mut(), kinds_before) {
                    sink(CollectEvent::Tick {
                        episode: index,
                        frame,
                        tick,
                        source,
                        events: raised(&kinds_before, &planes[0].counters().violations).bits(),
                    });
                }
                if let Some(ep) = outcome.episodes.into_iter().next() {
                    closed = Some(ep);
                    break;
                }
                if wrapper.aborted {
                    aborted = true;
                    break;
                }
            }
            // The budget ran out before a terminal condition: close the open episode. Exactly
            // one reset happens per episode either way, so the task's own randomization draws
            // stay a pure function of `seed` and the episode index.
            let mut episode = match closed {
                Some(ep) => ep,
                None => env
                    .reset(None)
                    .map_err(|e| bad(&e))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        DataError::Loop(format!("episode {index} closed no episode on reset"))
                    })?,
            };
            // A scripted driver that gave up is a failed demonstration, and the episode is
            // still written: dropping it would hide what the expert cannot do.
            if aborted {
                episode.termination = Termination::Failure;
            }
            terminations.push(episode.termination);
            // The state the episode ended in, which no step was entered with: the last row,
            // past the last frame (review M17 F-7).
            if let Some(t) = traj.as_mut() {
                t.push(env.model(), &env.terminal_state(), 0)
                    .map_err(|e| bad(&e))?;
            }
            if let (Some(dir), Some(t)) = (spec.traj_dir.as_ref(), &traj) {
                t.write(&dir.join(format!("ep-{index:03}.estraj")))
                    .map_err(|e| bad(&e))?;
            }
            runner.reset_env(0);
            // Clears the latch *and* re-arms the seed, so the next episode's first
            // `observe_state` puts the command chain back on the arm (packet M5/V6).
            planes[0].begin_episode();

            let n = episode.steps();
            if sources.len() < n {
                return Err(DataError::Loop(format!(
                    "episode {index}: {n} recorded steps but only {} action sources",
                    sources.len()
                )));
            }
            sources.truncate(n);
            commanded.truncate(n * NJ);
            answered.truncate(n * NJ);
            human.truncate(n);
            human.resize(n, false);
            intervention_frames += human.iter().filter(|h| **h).count() as u64;
            frames += n as u64;
            if let Some(sink) = sink.as_deref_mut() {
                sink(CollectEvent::EpisodeEnd {
                    episode: index,
                    outcome: episode.termination,
                    steps: n,
                });
            }
            segments.extend(segments_of(index, &human, InterventionSource::Scripted));
            writer.write_episode(&to_lerobot(
                index,
                &episode,
                (nq, nv, nu),
                &Recorded {
                    sources: &sources,
                    human: &human,
                    commanded: &commanded,
                    answered: &answered,
                },
                fps,
                &task_name,
            ))?;
        }
        writer.finish()?;
        crate::intervention::write_segments(spec.out_root, &segments)?;
        write_outcomes(spec.out_root, &terminations)?;

        let dataset = LeRobotDataset::open(spec.out_root)?;
        let (content, schema) = identity_of(&dataset)?;
        let h = &bundle.manifest.hashes;
        let slot = |d: Option<[u8; 32]>| d.as_ref().map_or_else(|| "-".to_owned(), hex);
        let ended = |t: Termination| terminations.iter().filter(|e| **e == t).count();
        // How the demonstrations ended, which the `LeRobot` columns have no place for: what
        // `es loop cycle` reads before it trains on them (packet P-M14-R1).
        let mut step = LoopStep::new(LoopKind::Collect)
            .input("task", &slot(h.task))
            .input("observation", &slot(h.observation))
            .input("learning", &slot(h.learning))
            .input("deployment", &slot(h.deployment))
            .input("seed", &spec.seed)
            .input("episodes", &spec.n_episodes)
            .output("content", &hex(&content))
            .output("schema", &hex(&schema))
            .output("success", &ended(Termination::Success))
            .output("failure", &ended(Termination::Failure))
            .output("timeout", &ended(Termination::Timeout));
        for (key, value) in ledger {
            step = step.input(key, value);
        }
        append_loop_step(spec.out_root, &step)?;

        Ok(CollectReport {
            root: spec.out_root.to_path_buf(),
            episodes: spec.n_episodes,
            frames,
            intervention_frames,
            segments,
            terminations,
            rendered,
            safety: *planes[0].counters(),
            content,
            schema,
            warnings,
        })
    }
}
