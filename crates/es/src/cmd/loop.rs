//! `es loop collect / intervene / distill` (spec 13.1: each step of the learning loop is a
//! CLI command and an artifact).
//!
//! Read `docs/design/learning-loop.md` first; the packet is
//! `docs/packets/M3/W7-learning-loop.md`. `es loop train` deliberately does not exist: the
//! training run is on the Python side of spec 2.3's split, and `distill` produces its input
//! identity rather than pretending to run it.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use es_assets::scene::{JointKind, SceneDesc};
use es_compile::PolicyBundle;
use es_data::collect::{
    CollectEvent, CollectSink, CollectSpec, Collector, Intervention, PerturbAt, Perturbation,
    SplitSpec, PERTURBATIONS_FILE,
};
use es_data::{CollectReport, InterventionSegment};
use es_env::expert::{demo_cfg, ScriptedExpert};
use es_env::program::{Program, ProgramError, SO101_PICK_PLACE};
use es_env::Termination;
use es_eval::{LightOverride, PerturbationPlan, ResetOverrides, StepState};
use es_ir::deployment::{DeploymentIr, Watchdog};
use es_ir::evaluation::{EvaluationIr, PerturbationKind, SeedPlan};
use es_ir::types::ElemType;
use es_physics_backend::{BackendKind, MjWarpBackend, MuJoCoCpuBackend, PhysXBackend};
use es_physics_core::backend::ModelInfo;
use es_physics_core::backend::PhysicsBackend;
use es_policy::{PolicyInfo, PolicyRuntime, TorchRuntime, WeightsSource};

/// The one scripted expert the CLI knows: SO-101, cube into the bin
/// (`docs/design/visible-learning.md` section 5).
const EXPERT_NAME: &str = "so101-pick-place";

use crate::cmd::telemetry::{Publisher, TelemetryArgs, STAGE_COLLECT};
use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es loop collect --policy <policy.esb> --scene <file.xml|urdf> --episodes <N> --seed <S>
                --out <root> [--backend <name>] [--runtime torch] [--max-steps <N>]
                [--expert <name | program.toml>] [--frames <dir>] [--traj <dir>]
                [--perturb <evaluation.toml> --suites <a,b>]
                [--telemetry <addr>] [--telemetry-token <t>] [--telemetry-image-every <N>]
es loop intervene --dataset <root> --segments <segments.json>
es loop distill --in <root> [--in-frames <dir>] [--in <root> [--in-frames <dir>]...]
                [--train 0.8] [--val 0.1] [--test 0.1] [--seed <S>] --out <root>
                [--frames <dir>]
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]     (see `es loop cycle --help`)

The three steps of the spec 13.1 learning loop that produce datasets.

collect    Opens the policy bundle (spec 9.6), rolls out <N> episodes through the env
           runtime with the Safety Plane from the bundle's Deployment IR -- the only
           actuator path (INV-12) -- and writes a LeRobot dataset with an `action_source`
           and an `intervention` column per frame (spec 13.2). Checks that the requested
           backend and runtime are available first; when either is not, prints
           `SKIPPED (<reason>)` and exits 3 without faking a run (spec 1.4).
           --backend (packet M11/X1; the names `es eval run --help` lists):
             mujoco-cpu  the reference, bitwise run to run (the default);
             mjwarp      MuJoCo Warp on the GPU, needs `mujoco_warp`; never bitwise;
             newton      refused by its mapping report before anything spawns (no
                         actuators, sensors or contacts in its adapter);
             physx       PhysX through Isaac Sim (M11/I1), needs ES_ISAAC_PYTHON;
                         never bitwise.
           Without --frames no pixels are written: an image channel becomes a declared
           video feature with VideoRef placeholders, plus a warning. --frames <dir> renders
           the Task IR's one image channel from its own camera, once per control step, as
           <dir>/<NNNNNN>.bin + .json -- the raw-tile format `es video mosaic` and the render
           goldens already use. It needs the `render` feature and a Vulkan device; a build
           without it refuses the flag rather than writing a dataset with a hole in it.
           Every episode's per-tick state (qpos, qvel and every body's world pose) is
           written to <root>/traj/ep-NNN.estraj, or to --traj <dir>; `es video showcase`
           re-renders a run from those files with an independent camera.
           --expert replaces the policy with a scripted demonstration: it drives every
           control tick through the same chunk buffer and the same Safety Plane (INV-12),
           records action_source=Human, and needs no Torch runtime. --policy is still
           required -- the bundle carries the Task, Observation and Deployment IR the
           collector reads -- but its weights are never loaded. A waypoint the arm cannot
           reach ends that episode as a failed demonstration (spec 17.2), written, not dropped.
           The demonstration is a program of move and grip blocks (packet M14/Q2; design note
           docs/design/editor-redesign.md section 11): a value ending in .toml is a program
           file's path, anything else a built-in name -- so101-pick-place, which is
           templates/teach/so101-pick-place.toml compiled in. A missing file, a file that does
           not parse and an unknown name are refused by name before anything opens. The
           ledger's collect step records `expert` (the value as given) and `expert_program`
           (the blake3 of the program's bytes; the built-in's for a name).
           With --telemetry <addr> the collection publishes what it is doing, on the streams
           `es eval run --telemetry` uses (packet M7/E7): 1 the episode.begin / episode.end
           events, 2 one [frame, tick, source, violation bits] sample per control tick -- the
           plane's own verdict, the same the dataset's action_source column records -- and 4
           the rendered observation every --telemetry-image-every ticks, which needs --frames.
           Publishing is non-blocking and computes nothing extra: the dataset under --out is
           byte-identical with and without the flag.
           --perturb <evaluation.toml> --suites <a,b> (packet M13/Z2) collects under that
           Evaluation IR's own perturbations: episode i runs under suite suites[i % len], drawn
           by es_eval::PerturbationPlan with the key `es eval run` gives episode i of an
           evaluation whose seed_base is --seed, so its scene, light and actuator are that
           evaluation episode's. The seeds [S, S+N) must not meet the Evaluation IR's own; an
           overlap is refused by name (spec 13.3). Frames, trajectories and state rows are the
           world as it was; the action column is the plane's answer, never the perturbed
           actuator's. <root>/meta/perturbations.jsonl records each episode's suite, seed and
           draws, and the ledger's collect step perturb.config, perturb.evaluation_hash and
           perturb.suites. A light suite needs --frames; an observation delay or frame drop
           the Deployment IR's stale_observation watchdog could see is refused (the
           collector's runner stamps an observation's age itself).

intervene  Applies intervention segments to a dataset that is already on disk. <segments.json>
           is a JSON array of
             { \"episode\": u32, \"start_frame\": u32, \"end_frame\": u32,
               \"source\": \"teleop\"|\"scripted\"|\"corrective\",
               \"operator_id\": string?, \"note\": string? }
           with `end_frame` inclusive. Segments are merged with any already recorded in
           meta/interventions.jsonl; the per-frame `intervention` column is rebuilt from the
           merged set. `action_source` is never rewritten -- it is collection-time
           provenance. dataset_content_hash moves; dataset_schema_hash does not, unless a
           column had to be added.

distill    Merges datasets (episodes re-indexed, intervention labels and the rows of
           meta/perturbations.jsonl remapped), computes the spec 19.2 deterministic split and
           writes training_identity.json (spec 19.3), split.json and a loop step. The training
           run itself is PyTorch-side and is not run here, so every TrainingIdentity slot but
           `dataset` is an all-zero digest.
           --frames <dir> also merges the inputs' flat frame tiles (`es loop collect --frames`)
           into <dir>, in the merged dataset's global frame order; each input's tiles are the
           --in-frames after its --in. Tiles that already are <dir> stay, when their --in is
           first -- what `es loop cycle`'s [collect] merge does with collect/frames.

cycle      Runs collect -> train -> eval -> showcase from one document, appending a step per
           stage to one ledger (spec 13.1, spec 13.3). It re-implements no stage: each one is
           the command above, called in-process with the words its own plan prints.

    --telemetry <addr> publish the run live on this address (spec 23.1), e.g. 127.0.0.1:7777
    --telemetry-token <t>  required in every client's Hello (spec 25.1); none by default
    --telemetry-image-every <N>  publish the observation image every N ticks (default 0, never)

Every step appends a LoopStep to <root>/loop.jsonl so the loop is reproducible (spec 13.3);
a distill step is appended to each input root as well as to the output root, and a cycle's
train and evaluate steps to the dataset root and to its own <out>.

Exit codes: 0 success, 1 runtime failure, 2 usage error, 3 skipped (nothing ran).
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("collect") => collect(&args[1..], None),
        // The whole loop under one document (packet M7/T2); its own help.
        Some("cycle") => crate::cmd::cycle::run(&args[1..]),
        Some("intervene") => intervene(&args[1..]),
        Some("distill") => distill(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es loop: unknown subcommand '{other}'\n\n{HELP}"
        ))),
    }
}

/// Pulls `--flag value` pairs off the argument list, rejecting anything unexpected.
fn parse(args: &[String], flags: &[&str]) -> Result<Vec<(String, String)>, CliError> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--help" || a == "-h" {
            return Err(CliError::Usage(HELP.to_owned()));
        }
        if !flags.contains(&a.as_str()) {
            return Err(CliError::Usage(format!("unknown flag '{a}'\n\n{HELP}")));
        }
        let value = it
            .next()
            .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{HELP}")))?;
        out.push((a.clone(), value.clone()));
    }
    Ok(out)
}

fn one<'a>(pairs: &'a [(String, String)], flag: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(f, _)| f == flag)
        .map(|(_, v)| v.as_str())
}

fn required<'a>(pairs: &'a [(String, String)], flag: &str) -> Result<&'a str, CliError> {
    one(pairs, flag).ok_or_else(|| CliError::Usage(format!("{flag} is required\n\n{HELP}")))
}

fn number<T: std::str::FromStr>(
    pairs: &[(String, String)],
    flag: &str,
    default: T,
) -> Result<T, CliError> {
    match one(pairs, flag) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| CliError::Usage(format!("{flag}: '{v}' is not a number\n\n{HELP}"))),
    }
}

// --- collect ----------------------------------------------------------------------------------

/// `Collector::run` (and the `SafetyPlane` inside it) is generic over the joint count and the
/// chunk horizon, which a `policy.esb` only reveals at run time.
/// ponytail: the same fixed dispatch table `es eval run` uses -- add a pair here when a new
/// robot/horizon combination needs `es loop collect`.
macro_rules! dispatch_nj_h {
    ($b:ty; $nj:expr, $h:expr, $($args:expr),+ $(,)?) => {
        match ($nj, $h) {
            (1, 1) => collect_typed::<$b, 1, 1>($($args),+),
            (2, 2) => collect_typed::<$b, 2, 2>($($args),+),
            (6, 1) => collect_typed::<$b, 6, 1>($($args),+),
            (6, 8) => collect_typed::<$b, 6, 8>($($args),+),
            (6, 16) => collect_typed::<$b, 6, 16>($($args),+),
            (6, 50) => collect_typed::<$b, 6, 50>($($args),+),
            (7, 1) => collect_typed::<$b, 7, 1>($($args),+),
            (7, 8) => collect_typed::<$b, 7, 8>($($args),+),
            (7, 16) => collect_typed::<$b, 7, 16>($($args),+),
            (7, 50) => collect_typed::<$b, 7, 50>($($args),+),
            (8, 50) => collect_typed::<$b, 8, 50>($($args),+),
            (nj, h) => Err(CliError::Runtime(format!(
                "unsupported (n_joints={nj}, horizon={h}); es loop collect supports a fixed \
                 table of pairs (crates/es/src/cmd/loop.rs) -- add one for this robot"
            ))),
        }
    };
}

/// The renderer `--frames` needs, built from what the bundle's Task IR already declares.
///
/// One image channel: `MultiViewPack` is rejected upstream anyway, and a second camera would be
/// a second frame directory this flag does not have a name for. The `ImageSpec` is the Task
/// IR's, so a scene whose camera does not produce what the IR declares is refused by
/// `EnvRenderer::check` rather than silently rendered at the wrong size (`INV-14`), and the
/// render path is the channel's own `render` declaration (packet M7/R5) -- there is no flag
/// for it, because the document decides.
#[cfg(feature = "render")]
fn renderer_cfg(
    bundle: &PolicyBundle,
    frames: &std::path::Path,
) -> Result<es_env::EnvRendererCfg, CliError> {
    let (name, frame, spec, render) = crate::cmd::eval::image_channel(&bundle.task)?;
    let es_ir::types::Frame::Camera(camera) = frame else {
        return Err(CliError::Runtime(format!(
            "image channel {name:?} is not in a camera frame, so there is no camera to render \
             it from"
        )));
    };
    Ok(es_env::render::sensor_cfg(
        camera,
        &spec,
        &render,
        Some(frames.to_path_buf()),
    ))
}

fn collect_typed<B: PhysicsBackend + Default, const NJ: usize, const H: usize>(
    spec: &CollectSpec<'_>,
    policy: &mut dyn PolicyRuntime,
    mut expert: Option<&mut ScriptedExpert>,
    frames: Option<&std::path::Path>,
    publisher: Option<&mut Publisher>,
    perturb: Option<&mut Perturb>,
    ledger: &[(String, String)],
) -> Result<CollectReport, CliError> {
    // The running episode's suite light, set by the perturbation hook at each episode's reset
    // and read by the frame sink below (packet M13/Z2); the identity without `--perturb`.
    let light = Cell::new(LightOverride::default());
    let mut at = perturb.map(|p| {
        let light = &light;
        move |at: PerturbAt<'_>| p.at(at, light)
    });
    let perturbation = at.as_mut().map(|hook| Perturbation { hook });
    // One publisher, two hooks: the collector's own sink says what the plane did and the
    // frame sink has the pixels. A `RefCell` because both closures live at once and the run
    // is single-threaded -- neither hook can be entered from inside the other (packet M7/E7).
    let publishing = publisher.is_some();
    let publisher = publisher.map(std::cell::RefCell::new);
    // Where this path learns that an episode began, for the `Tick` seed stream (packet
    // M10/W1a). `CollectEvent::EpisodeBegin` is emitted before the episode's first control
    // step and therefore before its first frame; the frame sink below clears the flag as it
    // renders. The collector's own loop counter is not reachable from here and the env's tick
    // runs across episodes, so this is the one signal that means "tick 0 of an episode".
    let new_episode = std::cell::Cell::new(Some(0u32));
    let mut publish = |event: CollectEvent| {
        if let CollectEvent::EpisodeBegin { episode, .. } = event {
            new_episode.set(Some(episode));
        }
        if let Some(p) = &publisher {
            p.borrow_mut().on_collect(event);
        }
    };
    // With `--frames` the sink is wired up even with nobody listening: it costs one bitset
    // copy per control step and it is what keeps the seed stream aligned with the episode.
    let sink: Option<CollectSink<'_>> = (publishing || frames.is_some()).then_some(&mut publish);
    // Teleop is real-robot I/O (M3 W1) and stays off this path; the one scripted intervener
    // the CLI offers is `--expert`. `es loop intervene` labels afterwards.
    //
    // The episode index, not `frame == 0`, is what says a new episode began: this hook is
    // `PolicyRuntime::infer`, which runs on the *inference* tick (5 Hz against a 50 Hz control
    // rate), so it sees frame 0 only when the episode happens to start on one. An episode
    // whose length is not a multiple of the inference period puts every later episode off
    // phase, and the expert then carries the finished episode's stage and latched cube into
    // the next one -- which is why `es loop collect --episodes N` only solved episode 0
    // (design note section 7.6 finding 5, packet M5/V1c).
    let mut started: Option<u32> = None;
    let mut hook = |episode: u32, _frame: u32, model: &ModelInfo, obs: &[f64]| {
        let Some(expert) = expert.as_deref_mut() else {
            return Intervention::Policy;
        };
        if started != Some(episode) {
            started = Some(episode);
            expert.reset();
        }
        let state = es_env::expert::state_of_row(model, obs);
        match expert.chunk(model, &state, 0) {
            Some(rows) => Intervention::Chunk(
                rows.iter()
                    .filter_map(|r| <[f64; NJ]>::try_from(r.as_slice()).ok())
                    .collect(),
            ),
            // Out of reach: a failed demonstration, never a clamped approximation (spec 17.2).
            None => Intervention::Abort,
        }
    };
    // With a renderer, the image channel the Task IR declares stops being a dangling video
    // reference: `EnvRenderer::frame` writes `<dir>/<NNNNNN>.bin` + `.json` per control step.
    // The `Gpu` and the renderer live here because `EnvRenderer<'gpu>` borrows the device, and
    // putting that borrow on `Env` would put a lifetime on a type `es-data` and `es-eval` both
    // name (design note section 7.4).
    #[cfg(feature = "render")]
    if let Some(dir) = frames {
        let cfg = renderer_cfg(spec.bundle, dir)?;
        let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
            .map_err(|e| CliError::Runtime(format!("no Vulkan device for --frames: {e}")))?;
        let mut renderer = es_env::EnvRenderer::new(&gpu, spec.scene, cfg)
            .map_err(|e| CliError::Runtime(e.to_string()))?;
        std::fs::create_dir_all(dir)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;
        // `drawn` is the episode's render draws as the collector's `Env` recorded them (packet
        // M11/X5): the identity for a task with no render target.
        let mut frame_sink =
            |model: &ModelInfo,
             state: &es_physics_core::backend::StateView<'_>,
             drawn: &es_env::randomize::RenderOverrides| {
                // Tick 0 of the episode (packet M10/W1a): under `seed = "tick"` this is where
                // the sample keys restart, so the evaluator -- which runs one `Env` per
                // episode and therefore starts at tick 0 by construction -- renders the same
                // grain at the same `(episode, tick)`.
                if new_episode.take().is_some() {
                    renderer.begin_episode();
                }
                let drawn = episode_draws(drawn, &light.get());
                let tile = renderer
                    .frame_with(model, state, 0, &drawn)
                    .map_err(|e| e.to_string())?;
                // The tile the run already rendered, borrowed, not a second render (packet
                // M7/E7); the publisher decides whether this is one of the published ones.
                if let (Some(p), Some(bytes)) = (&publisher, tile.as_u8()) {
                    p.borrow_mut().observation(tile.shape, bytes);
                }
                Ok(())
            };
        return Collector::run_perturbed::<B, _, NJ, H>(
            spec,
            policy,
            B::default,
            &mut hook,
            Some(&mut frame_sink),
            sink,
            perturbation,
            ledger,
        )
        .map_err(|e| CliError::Runtime(e.to_string()));
    }
    #[cfg(not(feature = "render"))]
    if frames.is_some() {
        return Err(CliError::Runtime(
            "--frames needs the `render` feature; this build links no renderer (it was built \
             with `--no-default-features`). Rebuild with the default features \
             (`cargo build -p es`)."
                .to_owned(),
        ));
    }
    Collector::run_perturbed::<B, _, NJ, H>(
        spec,
        policy,
        B::default,
        &mut hook,
        None,
        sink,
        perturbation,
        ledger,
    )
    .map_err(|e| CliError::Runtime(e.to_string()))
}

/// `es_eval::runner::episode_draws`, which that module keeps private: this episode's render
/// draws with the suite's light folded in -- intensities multiply and yaws add, and a suite
/// with no light perturbation leaves the draws exactly as recorded.
#[cfg(feature = "render")]
fn episode_draws(
    recorded: &es_env::randomize::RenderOverrides,
    light: &LightOverride,
) -> es_env::randomize::RenderOverrides {
    let mut out = recorded.clone();
    if !light.is_identity() {
        out.light.intensity *= light.intensity;
        out.light.yaw_deg += light.yaw_deg;
    }
    out
}

// --- collection under an Evaluation IR's perturbations (packet M13/Z2) ----------------------

/// `--perturb <evaluation.toml> --suites <a,b>`.
///
/// Episode `i` runs under suite `suites[i % len]` with the key `es eval run` gives episode
/// `i` of an evaluation whose `seed_base` is `--seed`: `apply_at_reset(suite index, seed + i,
/// i)` and `StepState::new(.., seed + i, suite index, i)`, as `es_eval::runner::run_episode`
/// calls them. The collector's `Env` is seeded `seed` at episode `i`, which is that evaluation
/// episode's scene as well (`Env::new(seeds[0])` then `seek_episode(i)` there).
#[derive(Debug)]
struct Perturb {
    config: String,
    evaluation_hash: [u8; 32],
    plan: PerturbationPlan,
    /// `--suites` as given, and each one's index in the Evaluation IR: the draw's `suite_id`.
    names: Vec<String>,
    cells: Vec<usize>,
    seed: u64,
    /// The Deployment IR's control period, which turns a drawn delay into control steps.
    control_us: u64,
    nj: usize,
    /// The running episode's per-step processes.
    step: Option<StepState>,
}

/// A drawn delay in control steps: `es_eval::runner`'s own conversion, truncating.
fn ms_to_steps(ms: u32, control_us: u64) -> usize {
    (u64::from(ms) * 1000 / control_us.max(1)) as usize
}

impl Perturb {
    /// Reads the Evaluation IR and refuses, before anything opens, what this collection cannot
    /// do faithfully.
    fn open(
        config: &str,
        suites: &str,
        seed: u64,
        episodes: u32,
        deploy: &DeploymentIr,
        has_renderer: bool,
    ) -> Result<Self, CliError> {
        let fail = |e: &dyn std::fmt::Display| CliError::Runtime(format!("{config}: {e}"));
        let raw = std::fs::read_to_string(config).map_err(|e| fail(&e))?;
        let mut ir = es_ir::serial::evaluation_from_toml(&raw).map_err(|e| fail(&e))?;
        if let Some(d) = ir.validate().first() {
            return Err(fail(d));
        }
        let evaluation_hash = ir.evaluation_hash().map_err(|d| fail(&d))?;
        let names: Vec<String> = suites
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();
        let cells = suite_cells(&ir, config, &names)?;
        refuse_evaluation_seeds(&ir, config, seed, episodes)?;
        let control_us = deploy.rate.control_period().0;
        refuse_unseen_age(&ir, &cells, deploy, control_us)?;
        // Only the chosen suites are realised, so an unchosen light suite does not demand
        // `--frames`. Emptied rather than removed: every suite keeps its index, which is the
        // `suite_id` its draws are keyed on (spec 10.4).
        for (k, suite) in ir.suites.iter_mut().enumerate() {
            if !cells.contains(&k) {
                suite.perturbations.clear();
            }
        }
        // ponytail: compiled before any backend opens, against an empty scene and model --
        // `compile` reads neither today (perturb.rs keeps them for the state-mutation kinds,
        // all `Unsupported`). When one of those gets a kernel, compile after `Env::new`.
        let plan = PerturbationPlan::compile(
            &ir,
            &SceneDesc::default(),
            &ModelInfo::default(),
            has_renderer,
        )
        .map_err(|e| fail(&e))?;
        Ok(Self {
            config: config.to_owned(),
            evaluation_hash,
            plan,
            names,
            cells,
            seed,
            control_us,
            nj: deploy.robot.n_joints,
            step: None,
        })
    }

    /// Episode `episode`'s suite index, seed and reset draws.
    fn draw(&self, episode: u32) -> (usize, u64, ResetOverrides) {
        let cell = self.cells[episode as usize % self.cells.len()];
        let seed = self.seed.wrapping_add(u64::from(episode));
        let mut ov = ResetOverrides::default();
        self.plan
            .apply_at_reset(cell, seed, u64::from(episode), &mut ov);
        (cell, seed, ov)
    }

    /// The collector's [`PerturbAt`] moments, answered as `es_eval::runner::run_episode`
    /// answers them.
    fn at(&mut self, at: PerturbAt<'_>, light: &Cell<LightOverride>) {
        match at {
            PerturbAt::Episode {
                episode,
                observation_delay,
            } => {
                let (cell, seed, ov) = self.draw(episode);
                light.set(ov.light);
                *observation_delay = ms_to_steps(ov.observation_delay_ms, self.control_us);
                // The runner fills its action-delay ring with this hold command.
                let hold = vec![0.0; self.nj];
                self.step = Some(StepState::new(
                    &ov,
                    ms_to_steps(ov.action_delay_ms, self.control_us),
                    &hold,
                    seed,
                    cell,
                    u64::from(episode),
                ));
            }
            PerturbAt::Observe { dropped } => {
                *dropped = self.step.as_mut().is_some_and(StepState::drop_observation);
            }
            PerturbAt::Actuate(ctrl) => {
                if let Some(step) = self.step.as_mut() {
                    step.apply_per_step(ctrl);
                }
            }
        }
    }

    /// The collect step's extra ledger inputs.
    fn ledger(&self) -> Vec<(String, String)> {
        vec![
            ("perturb.config".to_owned(), self.config.clone()),
            (
                "perturb.evaluation_hash".to_owned(),
                hex(&self.evaluation_hash),
            ),
            ("perturb.suites".to_owned(), self.names.join(",")),
        ]
    }

    fn write_meta(&self, root: &Path, episodes: u32) -> Result<(), CliError> {
        let mut text = String::new();
        for i in 0..episodes {
            let (cell, seed, ov) = self.draw(i);
            let line = serde_json::json!({
                "episode": i,
                "suite": self.plan.suite_name(cell),
                "seed": seed,
                "observation_delay_ms": ov.observation_delay_ms,
                "action_delay_ms": ov.action_delay_ms,
                "backlash_rad": ov.backlash_rad,
                "light_intensity": ov.light.intensity,
                "light_yaw_deg": ov.light.yaw_deg,
            });
            text.push_str(&line.to_string());
            text.push('\n');
        }
        let path = root.join(PERTURBATIONS_FILE);
        std::fs::write(&path, text)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
    }

    /// Successes per suite, in `--suites` order.
    fn summary(&self, terminations: &[Termination]) {
        let mut said: Vec<&str> = Vec::new();
        for name in &self.names {
            if said.contains(&name.as_str()) {
                continue;
            }
            said.push(name);
            let mine: Vec<&Termination> = terminations
                .iter()
                .enumerate()
                .filter(|(i, _)| self.names[i % self.names.len()] == *name)
                .map(|(_, t)| t)
                .collect();
            let won = mine.iter().filter(|t| ***t == Termination::Success).count();
            println!("suite {name}: success {won} of {}", mine.len());
        }
    }
}

/// `--suites` against the Evaluation IR: each name's index, which is its draws' `suite_id`,
/// and a name the IR lacks refused beside the ones it has. `es loop cycle` asks it too, so a
/// cycle's `[collect] perturb` is refused by `--dry-run` (packet M13/Z3).
pub(crate) fn suite_cells(
    ir: &EvaluationIr,
    config: &str,
    names: &[String],
) -> Result<Vec<usize>, CliError> {
    let known = || {
        let all: Vec<&str> = ir.suites.iter().map(|s| s.name.as_str()).collect();
        all.join(", ")
    };
    if names.is_empty() {
        return Err(CliError::Usage(format!(
            "--suites names no suite; {config} has {}",
            known()
        )));
    }
    names
        .iter()
        .map(|name| {
            ir.suites
                .iter()
                .position(|s| s.name == *name)
                .ok_or_else(|| {
                    CliError::Usage(format!(
                        "--suites: {config} has no suite {name:?}; it has {}",
                        known()
                    ))
                })
        })
        .collect()
}

/// Spec 13.3: a re-collection never draws what the policy will be judged on. The collection's
/// seeds `[seed, seed + episodes)` against the Evaluation IR's resolved ones (spec 10.2: the
/// explicit list, or `n_episodes` from `seed_base`, as `es_eval::runner::resolve_seeds` has it).
pub(crate) fn refuse_evaluation_seeds(
    ir: &EvaluationIr,
    config: &str,
    seed: u64,
    episodes: u32,
) -> Result<(), CliError> {
    let mut judged: Vec<u64> = match &ir.episodes.seeds {
        SeedPlan::Base(b) => (0..u64::from(ir.episodes.n_episodes))
            .map(|i| b.wrapping_add(i))
            .collect(),
        SeedPlan::Explicit(list) => list.clone(),
    };
    if !judged
        .iter()
        .any(|s| s.wrapping_sub(seed) < u64::from(episodes))
    {
        return Ok(());
    }
    judged.sort_unstable();
    judged.dedup();
    let span = match (judged.first(), judged.last()) {
        (Some(lo), Some(hi)) if hi - lo == judged.len() as u64 - 1 => format!("{lo}-{hi}"),
        _ => judged
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
    };
    let last = seed.wrapping_add(u64::from(episodes) - 1);
    Err(CliError::Usage(format!(
        "--seed {seed} --episodes {episodes} collects seeds {seed}-{last}, and --seed overlaps \
         evaluation seeds {span} of {config}: a re-collection never trains on what it is judged \
         on (spec 13.3). Pick a --seed whose range misses them."
    )))
}

/// The one thing this path cannot hand the Safety Plane that `es eval run` does: the
/// collector's `DomainRunner` stamps an observation's age itself, so a held observation
/// reaches `SafetyPlane::validate` as age 0 where the evaluation passes its true age. The plane
/// reads that age only against a `stale_observation` watchdog, so a suite is realised when no
/// age it can produce crosses that watchdog and refused otherwise, rather than judged by a
/// different plane (spec 17.2) -- `frame_drop` whenever there is one, its bursts having no bound.
fn refuse_unseen_age(
    ir: &EvaluationIr,
    cells: &[usize],
    deploy: &DeploymentIr,
    control_us: u64,
) -> Result<(), CliError> {
    let Some(max_age) = deploy.watchdogs.0.iter().find_map(|w| match w {
        Watchdog::StaleObservation { max_age } => Some(max_age.0),
        _ => None,
    }) else {
        return Ok(());
    };
    for suite in cells.iter().map(|k| &ir.suites[*k]) {
        for p in &suite.perturbations {
            let worst = match &p.kind {
                PerturbationKind::ObservationDelay { ms } => ms
                    .iter()
                    .map(|m| ms_to_steps(*m, control_us) as u64 * control_us)
                    .max()
                    .unwrap_or(0),
                PerturbationKind::FrameDrop { .. } => u64::MAX,
                _ => 0,
            };
            if worst > max_age {
                return Err(CliError::Runtime(format!(
                    "suite {:?}: {} can age the observation past the deployment's \
                     stale_observation max_age of {max_age} us, and `es loop collect` cannot \
                     hand the Safety Plane that age (its runner stamps its own), so the plane \
                     here would not be the evaluation's -- refused rather than approximated \
                     (spec 17.2)",
                    suite.name,
                    p.kind.name()
                )));
            }
        }
    }
    Ok(())
}

/// The `PolicyRuntime` slot under `--expert`: the bundle's Task, Observation and Deployment IR
/// are what the collector needs, and its weights are never loaded, so nothing is behind this.
/// Every call is a named error rather than a zero tensor — if the collector ever reached the
/// policy under `--expert`, that would be a bug worth a message (spec 17.2).
#[derive(Debug, Default)]
struct NoPolicy;

impl PolicyRuntime for NoPolicy {
    fn load(
        &mut self,
        _graph: &es_ir::learning::LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<es_policy::PolicyInfo, es_policy::PolicyError> {
        Err(es_policy::PolicyError::Backend(
            "es loop collect --expert loads no policy".to_owned(),
        ))
    }

    fn infer(
        &mut self,
        _inputs: &std::collections::BTreeMap<String, es_compile::Tensor>,
    ) -> Result<std::collections::BTreeMap<String, es_compile::Tensor>, es_policy::PolicyError>
    {
        Err(es_policy::PolicyError::Backend(
            "es loop collect --expert has no policy to infer with".to_owned(),
        ))
    }

    fn info(&self) -> Option<&es_policy::PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        [0; 32]
    }
}

// --- the expert as a `PolicyRuntime` (packet M7/T2, spec 28.9 rule 1) -----------------------

/// The loaded model and the raw `qpos ‖ qvel` row the evaluation's frame source last saw.
///
/// `es loop collect` hands the expert its state through the intervener hook, which carries the
/// episode index; `es_eval::Evaluation` has no such hook and calls `PolicyRuntime::infer` with
/// the observation tensors alone — and the demo's Observation IR carries no cube pose. The
/// frame source is handed the full `StateView` immediately before the policy is called, so the
/// row travels through here. That is the scaffold `expert_passes_the_evaluation_harness` has
/// used since packet M5/V6, promoted from the test file to the command that needs it.
#[derive(Clone, Debug, Default)]
pub(crate) struct SeenState(std::rc::Rc<std::cell::RefCell<Seen>>);

#[derive(Debug, Default)]
pub(crate) struct Seen {
    model: Option<ModelInfo>,
    row: Vec<f64>,
    /// How many episodes have begun. `Env::reset` fills `qpos` and `qvel` with zeros before
    /// the Task IR's `Randomization` node writes the cube's pose, and an episode's first tick
    /// always reaches the frame source (the observation ring is empty there, so
    /// `observation_delay` cannot drop it) — so a state whose every velocity is *exactly*
    /// zero is that tick and no other, and counting them is an episode boundary the runner
    /// never had to be asked for.
    ///
    /// ponytail: an exact-zero test, not a threshold. A mid-episode state with every velocity
    /// exactly 0.0 would restart the expert's stage machine, which fails that episode loudly
    /// rather than passing the gate quietly — the safe direction for a gate to be wrong in.
    episodes: u64,
}

impl SeenState {
    /// Only the render build has a frame source to call this from (`es eval run --expert`
    /// refuses without one), so a build without the feature never reaches it.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn capture(
        &self,
        model: &ModelInfo,
        state: &es_physics_core::backend::StateView<'_>,
    ) {
        let mut seen = self.0.borrow_mut();
        if state.qvel_of(0).iter().all(|v| *v == 0.0) {
            seen.episodes += 1;
        }
        seen.row.clear();
        seen.row.extend_from_slice(state.qpos_of(0));
        seen.row.extend_from_slice(state.qvel_of(0));
        seen.model = Some(model.clone());
    }
}

/// [`ScriptedExpert`] wearing the `PolicyRuntime` the evaluation runner drives.
///
/// Not a new extension point (`INV-17`): `PolicyRuntime` is one of the seven and this is an
/// impl of it. The joint count and chunk horizon are runtime fields rather than const
/// generics because the runtime is built before the bundle's `(NJ, H)` pair picks a typed
/// path, and the tensor it returns is shaped, not typed.
pub(crate) struct ExpertPolicy {
    expert: ScriptedExpert,
    seen: SeenState,
    nj: usize,
    horizon: usize,
    /// The episode this instance last reset for.
    episode: u64,
}

impl PolicyRuntime for ExpertPolicy {
    fn load(
        &mut self,
        _graph: &es_ir::learning::LearningGraph,
        _weights: &WeightsSource,
    ) -> Result<PolicyInfo, es_policy::PolicyError> {
        Err(es_policy::PolicyError::Backend(
            "es eval run --expert loads no policy".to_owned(),
        ))
    }

    fn infer(
        &mut self,
        _inputs: &std::collections::BTreeMap<String, es_compile::Tensor>,
    ) -> Result<std::collections::BTreeMap<String, es_compile::Tensor>, es_policy::PolicyError>
    {
        let seen = self.seen.0.borrow();
        let model = seen.model.as_ref().ok_or_else(|| {
            es_policy::PolicyError::Backend(
                "no state captured yet: es eval run --expert needs --frames".to_owned(),
            )
        })?;
        if self.episode != seen.episodes {
            self.episode = seen.episodes;
            self.expert.reset();
        }
        let state = es_env::expert::state_of_row(model, &seen.row);
        // Out of reach ends the demonstration in `es loop collect`; here the arm holds, so the
        // episode runs out its budget and is scored a failure rather than an error (spec 17.2).
        let rows = self
            .expert
            .chunk(model, &state, 0)
            .unwrap_or_else(|| vec![state.qpos_of(0)[..self.nj].to_vec(); self.horizon]);
        let mut data = Vec::with_capacity(self.horizon * self.nj * 8);
        for r in rows.iter().take(self.horizon) {
            for j in 0..self.nj {
                data.extend_from_slice(&r.get(j).copied().unwrap_or(0.0).to_le_bytes());
            }
        }
        Ok(std::collections::BTreeMap::from([(
            "action".to_owned(),
            es_compile::Tensor {
                dtype: ElemType::F64,
                shape: vec![self.horizon as u64, self.nj as u64],
                data,
            },
        )]))
    }

    fn info(&self) -> Option<&PolicyInfo> {
        None
    }

    fn runtime_hash(&self) -> [u8; 32] {
        *blake3::hash(b"es-env::ScriptedExpert").as_bytes()
    }
}

/// The expert `es eval run --expert <name | program.toml>` drives with, paced by the same
/// Deployment IR the collection path paces it with.
pub(crate) fn expert_policy(
    expert: &ExpertProgram,
    scene: &SceneDesc,
    deploy: &es_ir::deployment::DeploymentIr,
    seen: &SeenState,
) -> Result<ExpertPolicy, CliError> {
    Ok(ExpertPolicy {
        expert: build_expert(expert, scene, deploy)?,
        seen: seen.clone(),
        nj: deploy.robot.n_joints,
        horizon: deploy.action.horizon,
        episode: 0,
    })
}

/// What `--expert <name | program.toml>` names (packet M14/Q2), resolved once for `es loop
/// collect` and `es eval run` alike: a value ending in `.toml` is a demonstration program's path
/// (`docs/design/editor-redesign.md` section 11), anything else a built-in's name -- and the
/// one built-in, `so101-pick-place`, is the committed program compiled in.
#[derive(Debug)]
pub(crate) struct ExpertProgram {
    /// The flag's value as given: the collect step's `expert`.
    given: String,
    program: Program,
    /// blake3 of the program's bytes, the built-in's compiled-in bytes for a name: the collect
    /// step's `expert_program`.
    digest: [u8; 32],
}

impl ExpertProgram {
    /// Refuses, by name, an unknown name, a file that cannot be read and a file that does not
    /// parse (in `ProgramError`'s words) -- before any bundle, scene or backend is opened.
    pub(crate) fn resolve(given: &str, help: &str) -> Result<Self, CliError> {
        let text = if Path::new(given).extension().is_some_and(|e| e == "toml") {
            std::fs::read_to_string(given).map_err(|e| {
                CliError::Runtime(format!(
                    "--expert {given}: cannot read this demonstration program: {e}"
                ))
            })?
        } else if given == EXPERT_NAME {
            SO101_PICK_PLACE.to_owned()
        } else {
            return Err(CliError::Usage(format!(
                "unknown --expert '{given}': the built-in expert is {EXPERT_NAME}, and a \
                 demonstration program is a path ending in .toml\n\n{help}"
            )));
        };
        let program = Program::parse(&text)
            .map_err(|e| CliError::Runtime(format!("--expert {given}: {e}")))?;
        Ok(Self {
            given: given.to_owned(),
            program,
            digest: *blake3::hash(text.as_bytes()).as_bytes(),
        })
    }

    /// The collect step's ledger inputs: provenance, which feeds no hash (spec 13.3).
    fn ledger(&self) -> [(String, String); 2] {
        [
            ("expert".to_owned(), self.given.clone()),
            ("expert_program".to_owned(), hex(&self.digest)),
        ]
    }
}

/// The scripted expert running `expert`'s program on the scene it will drive.
///
/// The object is the free-joint body the program names (`object = "cube"`): a demonstration
/// that picks something up needs something that can be picked up, and the program -- not a
/// flag -- says which.
fn build_expert(
    expert: &ExpertProgram,
    scene: &SceneDesc,
    deploy: &es_ir::deployment::DeploymentIr,
) -> Result<ScriptedExpert, CliError> {
    let refuse = |e: ProgramError| CliError::Runtime(format!("--expert {}: {e}", expert.given));
    let object = &expert.program.object;
    let joint = scene
        .joints
        .iter()
        .find(|j| {
            j.kind == JointKind::Free
                && scene
                    .bodies
                    .iter()
                    .any(|b| b.id == j.body && b.name == *object)
        })
        .ok_or_else(|| refuse(ProgramError::UnknownObject(object.clone())))?;
    let mut cfg = demo_cfg(joint.id);
    // Both paths replan at the deployment's inference rate and execute the chunk's rows in
    // between, so the rows that actually execute per chunk are the re-plan period -- one
    // definition, one number, on collection and evaluation alike (packet M5/V17).
    let replan = es_env::replan_interval(deploy.rate)
        .map_err(|e| CliError::Runtime(e.to_string()))?
        .min(deploy.action.execute_chunk as u64);
    cfg.pace_to(deploy, replan as u32);
    ScriptedExpert::with_program(scene, cfg, &expert.program).map_err(refuse)
}

/// Packet M14/Q2 oracles: the resolver `es loop collect` and `es eval run` share.
#[cfg(test)]
mod expert_program_tests {
    use super::*;

    fn repo(path: &str) -> String {
        format!("{}/../../{path}", env!("CARGO_MANIFEST_DIR"))
    }

    /// A program file in this test's own directory.
    fn written(name: &str, text: &str) -> String {
        let dir = std::env::temp_dir().join(format!("es-q2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join(name);
        std::fs::write(&path, text).expect("write the program");
        path.display().to_string()
    }

    fn resolve(given: &str) -> Result<ExpertProgram, CliError> {
        ExpertProgram::resolve(given, HELP)
    }

    /// The name and the committed file are one program with one digest; only `expert`, the
    /// value as given, tells them apart in the ledger.
    #[test]
    fn expert_name_and_committed_file_are_one_program() {
        let digest = hex(blake3::hash(SO101_PICK_PLACE.as_bytes()).as_bytes());
        let name = resolve(EXPERT_NAME).expect("the built-in resolves");
        assert_eq!(name.program, Program::builtin());
        assert_eq!(
            name.ledger(),
            [
                ("expert".to_owned(), EXPERT_NAME.to_owned()),
                ("expert_program".to_owned(), digest.clone()),
            ]
        );
        let path = repo("templates/teach/so101-pick-place.toml");
        let file = resolve(&path).expect("the committed program resolves");
        assert_eq!(file.program, name.program);
        assert_eq!(
            file.ledger(),
            [
                ("expert".to_owned(), path),
                ("expert_program".to_owned(), digest),
            ]
        );
    }

    /// An unknown name, a missing file and a file that does not parse are each refused by
    /// name, the last in `ProgramError`'s own words.
    #[test]
    fn expert_refuses_by_name() {
        match resolve("so101-pick") {
            Err(CliError::Usage(m)) => assert!(m.contains("unknown --expert 'so101-pick'"), "{m}"),
            other => panic!("an unknown name was not a usage error: {other:?}"),
        }
        let missing = repo("templates/teach/no-such-program.toml");
        let e = resolve(&missing).expect_err("a missing file");
        assert!(matches!(e, CliError::Runtime(_)), "{e}");
        assert!(
            e.to_string()
                .contains(&format!("--expert {missing}: cannot read")),
            "{e}"
        );

        let both = written(
            "both.toml",
            &SO101_PICK_PLACE.replacen("above = 0.045", "above = 0.045\nheight = 0.2", 1),
        );
        let e = resolve(&both).expect_err("both heights");
        let words = ProgramError::BothHeights(0).to_string();
        assert!(
            e.to_string().contains(&format!("--expert {both}: {words}")),
            "{e}"
        );
        let junk = written("junk.toml", "kind = [");
        let e = resolve(&junk).expect_err("not TOML");
        assert!(e.to_string().contains("demonstration program:"), "{e}");
    }

    /// The program's object and places resolve against the scene it will drive; a program that
    /// names what the scene lacks is refused by name, never approximated.
    #[test]
    fn expert_builds_on_the_scene_the_program_names() {
        let scene =
            crate::cmd::backend::load_scene(&repo("tests/fixtures/mjcf/so101_pick_place.xml"))
                .expect("the demo scene");
        let deploy = es_ir::serial::deployment_from_toml(
            &std::fs::read_to_string(repo("tests/fixtures/visible-learning/deployment.toml"))
                .expect("deployment.toml"),
        )
        .expect("the demo deployment");
        let build = |name: &str, text: &str| {
            let program = resolve(&written(name, text)).expect("parses");
            build_expert(&program, &scene, &deploy)
        };
        build("builtin.toml", SO101_PICK_PLACE).expect("the built-in builds");
        let e = build(
            "ball.toml",
            &SO101_PICK_PLACE.replace("object = \"cube\"", "object = \"ball\""),
        )
        .expect_err("no ball");
        assert!(e.to_string().contains("object = \"ball\""), "{e}");
        let e = build(
            "shelf.toml",
            &SO101_PICK_PLACE.replacen("move = \"bin\"", "move = \"shelf\"", 1),
        )
        .expect_err("no shelf");
        assert!(e.to_string().contains("block 5: no geom"), "{e}");
    }
}

/// One collection.
///
/// `cycle` is `es loop cycle`'s own publisher (packet M7/E7): a cycle binds one socket before
/// its first stage, so `--telemetry` is not on the argv of a nested `es loop collect` and
/// this command binds nothing for it.
pub(crate) fn collect(args: &[String], cycle: Option<&mut Publisher>) -> Result<u8, CliError> {
    let pairs = parse(
        args,
        &[
            "--policy",
            "--scene",
            "--episodes",
            "--seed",
            "--out",
            "--backend",
            "--runtime",
            "--max-steps",
            "--expert",
            "--frames",
            "--traj",
            "--telemetry",
            "--telemetry-token",
            "--telemetry-image-every",
            "--perturb",
            "--suites",
        ],
    )?;
    let perturb_args = match (one(&pairs, "--perturb"), one(&pairs, "--suites")) {
        (Some(config), Some(suites)) => Some((config, suites)),
        (None, None) => None,
        _ => {
            return Err(CliError::Usage(format!(
                "--perturb and --suites go together: the Evaluation IR, and which of its suites \
                 to collect under\n\n{HELP}"
            )))
        }
    };
    let program = one(&pairs, "--expert")
        .map(|e| ExpertProgram::resolve(e, HELP))
        .transpose()?;
    let policy_path = required(&pairs, "--policy")?.to_owned();
    let scene_path = required(&pairs, "--scene")?.to_owned();
    let out = PathBuf::from(required(&pairs, "--out")?);
    let episodes: u32 = number(&pairs, "--episodes", 1)?;
    let seed: u64 = number(&pairs, "--seed", 0)?;
    let max_steps: u32 = number(&pairs, "--max-steps", 0)?;
    let frames = one(&pairs, "--frames").map(PathBuf::from);
    // Always recorded, never a flag to remember (packet M5/V9): the per-tick state is what
    // `es video showcase` replays and what "what the robot did" means for provenance. It sits
    // beside the `LeRobot` files, not inside them, so no dataset hash moves.
    let traj_dir = one(&pairs, "--traj").map_or_else(|| out.join("traj"), PathBuf::from);
    let kind =
        crate::cmd::eval::parse_backend(one(&pairs, "--backend").unwrap_or("mujoco-cpu"), HELP)?;
    let runtime = one(&pairs, "--runtime").unwrap_or("torch");
    if runtime != "torch" {
        return Err(CliError::Usage(format!(
            "unknown --runtime '{runtime}': only torch is supported\n\n{HELP}"
        )));
    }

    // Bound before the bundle, the scene or the Torch runtime is opened, so a viewer that
    // attaches on the printed address is subscribed before the first episode (packet M7/E7).
    let telemetry = TelemetryArgs {
        addr: match one(&pairs, "--telemetry") {
            Some(v) => Some(TelemetryArgs::parse_addr(v, HELP)?),
            None => None,
        },
        token: one(&pairs, "--telemetry-token").map(ToOwned::to_owned),
        image_every: match one(&pairs, "--telemetry-image-every") {
            Some(v) => TelemetryArgs::parse_image_every(v, HELP)?,
            None => 0,
        },
    };
    let mut owned = telemetry.bind(STAGE_COLLECT)?;
    let publisher: Option<&mut Publisher> = cycle.or(owned.as_mut());
    // Said once, rather than by a stream that never carries an image: nothing renders without
    // `--frames`, so there is no observation to publish.
    if publisher.is_some() && frames.is_none() && telemetry.image_every > 0 {
        println!(
            "telemetry: --telemetry-image-every is set and --frames is not, so no observation \
             image is published (nothing renders)"
        );
    }

    let bytes = std::fs::read(&policy_path)
        .map_err(|e| CliError::Runtime(format!("{policy_path}: {e}")))?;
    let bundle = PolicyBundle::open(&bytes).map_err(|e| CliError::Runtime(e.to_string()))?;
    // Before any backend is probed, so a collection that would reuse the evaluation's seeds is
    // refused on the documents alone (packet M13/Z2).
    let mut perturb = perturb_args
        .map(|(config, suites)| {
            Perturb::open(
                config,
                suites,
                seed,
                episodes,
                &bundle.deployment,
                frames.is_some(),
            )
        })
        .transpose()?;

    // A backend other than the reference is gated on its mapping report first (spec 14.4),
    // so the refusal names the rows even where the engine is not installed; `mujoco-cpu`
    // keeps today's order. Then: a run this machine cannot really do is skipped, never faked.
    let early_scene = if kind == BackendKind::MuJoCoCpu {
        None
    } else {
        let scene = super::backend::load_scene(&scene_path)?;
        crate::cmd::eval::mapping_gate(kind, &scene)?;
        Some(scene)
    };
    if let Err(reason) = es_physics_backend::is_available(kind) {
        println!("SKIPPED ({kind} backend unavailable: {reason})");
        return Ok(3);
    }
    // `--expert` drives every tick itself, so the bundle's weights are never loaded and the
    // Torch runtime is not needed at all (design note section 5.1).
    if program.is_none() {
        if let Err(reason) = es_policy::torch_runtime::is_available() {
            println!("SKIPPED (torch runtime unavailable: {reason})");
            return Ok(3);
        }
    }

    let scene = match early_scene {
        Some(scene) => scene,
        None => super::backend::load_scene(&scene_path)?,
    };
    let mut expert = program
        .as_ref()
        .map(|p| build_expert(p, &scene, &bundle.deployment))
        .transpose()?;
    let mut torch = TorchRuntime::new();
    let mut no_policy = NoPolicy;
    let policy: &mut dyn PolicyRuntime = if expert.is_some() {
        &mut no_policy
    } else {
        torch
            .load(
                &bundle.learning,
                &WeightsSource::InMemory(bundle.weights.clone()),
            )
            .map_err(|e| CliError::Runtime(e.to_string()))?;
        &mut torch
    };

    let spec = CollectSpec {
        bundle: &bundle,
        scene: &scene,
        n_episodes: episodes,
        seed,
        max_steps,
        out_root: &out,
        traj_dir: Some(traj_dir.clone()),
    };
    let nj = bundle.deployment.robot.n_joints;
    let h = bundle.deployment.action.horizon;
    // One dispatch on the backend, monomorphized: `Env<B>` stays generic (spec 3.4).
    let (expert, frames) = (expert.as_mut(), frames.as_deref());
    // The collect step's inputs beyond the bundle's (spec 13.3): what drove it (packet M14/Q2)
    // and what it ran under (packet M13/Z2).
    let ledger: Vec<(String, String)> = program
        .iter()
        .flat_map(ExpertProgram::ledger)
        .chain(perturb.iter().flat_map(Perturb::ledger))
        .collect();
    let (p, l) = (perturb.as_mut(), ledger.as_slice());
    let report = match kind {
        BackendKind::MuJoCoCpu => dispatch_nj_h!(
            MuJoCoCpuBackend; nj, h, &spec, policy, expert, frames, publisher, p, l
        ),
        BackendKind::MjWarp => dispatch_nj_h!(
            MjWarpBackend; nj, h, &spec, policy, expert, frames, publisher, p, l
        ),
        BackendKind::PhysX => dispatch_nj_h!(
            PhysXBackend; nj, h, &spec, policy, expert, frames, publisher, p, l
        ),
        BackendKind::Newton => Err(crate::cmd::eval::no_closed_loop(kind)),
    }?;
    if let Some(p) = &perturb {
        p.write_meta(&out, episodes)?;
    }

    for w in &report.warnings {
        println!("warning: {w}");
    }
    println!("wrote {}", report.root.display());
    println!("episodes: {}   frames: {}", report.episodes, report.frames);
    println!("trajectories: {}", traj_dir.display());
    println!("{}", crate::cmd::eval::determinism_tier(kind));
    let count = |t: Termination| report.terminations.iter().filter(|x| **x == t).count();
    println!(
        "terminations: success {}   failure {}   timeout {}   running {}",
        count(Termination::Success),
        count(Termination::Failure),
        count(Termination::Timeout),
        count(Termination::Running)
    );
    if let Some(p) = &perturb {
        p.summary(&report.terminations);
    }
    println!("intervention frames: {}", report.intervention_frames);
    // What the plane did, per spec 10.3's failure-mode histogram: a demonstration the envelope
    // had to correct is worth seeing, not hiding (INV-12).
    let kinds: Vec<String> = es_safety::ViolationKind::ALL
        .iter()
        .filter(|k| report.safety.count(**k) > 0)
        .map(|k| format!("{k:?}={}", report.safety.count(*k)))
        .collect();
    println!(
        "safety: {} steps, {} clamped, {} fallback{}{}",
        report.safety.steps,
        report.safety.clamped_steps,
        report.safety.fallback_activations,
        if kinds.is_empty() { "" } else { "   " },
        kinds.join(" ")
    );
    println!("content: {}", hex(&report.content));
    println!("schema:  {}", hex(&report.schema));
    // Only the command that bound the socket closes the account of it.
    if let Some(p) = &owned {
        println!("{}", p.summary());
    }
    Ok(0)
}

// --- intervene --------------------------------------------------------------------------------

fn intervene(args: &[String]) -> Result<u8, CliError> {
    let pairs = parse(args, &["--dataset", "--segments"])?;
    let dataset = PathBuf::from(required(&pairs, "--dataset")?);
    let segments_path = required(&pairs, "--segments")?.to_owned();

    let raw = std::fs::read_to_string(&segments_path)
        .map_err(|e| CliError::Runtime(format!("{segments_path}: {e}")))?;
    let segments: Vec<InterventionSegment> = serde_json::from_str(&raw).map_err(|e| {
        CliError::Runtime(format!(
            "{segments_path}: not an array of intervention segments: {e}"
        ))
    })?;

    let report =
        es_data::label(&dataset, &segments).map_err(|e| CliError::Runtime(e.to_string()))?;
    println!(
        "labelled {} frames across {} episodes",
        report.frames,
        report.episodes.len()
    );
    println!("content before: {}", hex(&report.content_before));
    println!("content after:  {}", hex(&report.content_after));
    println!(
        "schema:         {} ({})",
        hex(&report.schema),
        if report.schema_changed {
            "changed: a column was added"
        } else {
            "unchanged"
        }
    );
    Ok(0)
}

// --- distill ----------------------------------------------------------------------------------

/// One merge. `es loop cycle`'s merge stage is this call, with the words its plan prints.
pub(crate) fn distill(args: &[String]) -> Result<u8, CliError> {
    let pairs = parse(
        args,
        &[
            "--in",
            "--in-frames",
            "--out",
            "--frames",
            "--train",
            "--val",
            "--test",
            "--seed",
        ],
    )?;
    // Each `--in-frames` belongs to the `--in` before it (packet M13/Z3).
    let mut inputs: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    for (flag, value) in &pairs {
        match (flag.as_str(), inputs.last_mut()) {
            ("--in", _) => inputs.push((PathBuf::from(value), None)),
            ("--in-frames", Some((_, tiles @ None))) => *tiles = Some(PathBuf::from(value)),
            ("--in-frames", _) => {
                return Err(CliError::Usage(format!(
                    "--in-frames {value}: it follows the --in <root> whose tiles it holds, once \
                     per --in\n\n{HELP}"
                )))
            }
            _ => {}
        }
    }
    if inputs.is_empty() {
        return Err(CliError::Usage(format!(
            "at least one --in <root> is required\n\n{HELP}"
        )));
    }
    let out = PathBuf::from(required(&pairs, "--out")?);
    let frames = one(&pairs, "--frames").map(PathBuf::from);
    let tiles: Vec<(&Path, &Path)> = inputs
        .iter()
        .filter_map(|(root, t)| Some((root.as_path(), t.as_deref()?)))
        .collect();
    if tiles.len() != if frames.is_some() { inputs.len() } else { 0 } {
        return Err(CliError::Usage(format!(
            "--frames <dir> and --in-frames go together: every --in names its tiles when the \
             merge writes frames, and none does when it does not\n\n{HELP}"
        )));
    }
    let split = SplitSpec {
        ratios: [
            number(&pairs, "--train", 0.8)?,
            number(&pairs, "--val", 0.1)?,
            number(&pairs, "--test", 0.1)?,
        ],
        seed: number(&pairs, "--seed", 0)?,
    };

    // Before the dataset, so a missing tile stops the merge before any ledger records it.
    if let Some(dir) = &frames {
        let n = merge_frames(&tiles, dir)?;
        println!("frames: {n} tile(s) in {}", dir.display());
    }
    let inputs: Vec<&Path> = inputs.iter().map(|(root, _)| root.as_path()).collect();
    let identity =
        es_data::distill(&inputs, &split, &out).map_err(|e| CliError::Runtime(e.to_string()))?;
    let training_hash = identity
        .training_hash()
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    println!("wrote {}", out.display());
    println!("dataset identity (spec 19.2):");
    println!("  content: {}", hex(&identity.dataset.content));
    println!("  schema:  {}", hex(&identity.dataset.schema));
    println!("  split:   {}", hex(&identity.dataset.split));
    println!("training_hash: {}", hex(&training_hash));
    println!(
        "note: every other TrainingIdentity slot is an all-zero digest -- the training run is \
         PyTorch-side (spec 19.3, spec 2.3)."
    );
    Ok(0)
}

/// The inputs' flat `<NNNNNN>.bin` tiles (and `.json` beside them) in the merged dataset's
/// global frame order: input `k`'s tile `i` becomes `<out>/<offset + i>`, the offset being the
/// frames of the inputs before it -- `es_data::distill`'s episode order. Hard links where the
/// volume allows, copies where not. Tiles that already are `<out>` at offset 0 (a cycle's new
/// collection, merged first) stay where they are.
fn merge_frames(inputs: &[(&Path, &Path)], out: &Path) -> Result<u64, CliError> {
    let fail =
        |p: &Path, e: &dyn std::fmt::Display| CliError::Runtime(format!("{}: {e}", p.display()));
    std::fs::create_dir_all(out).map_err(|e| fail(out, &e))?;
    let here = out.canonicalize().map_err(|e| fail(out, &e))?;
    // Every offset first, so a refusal leaves every tile where it was.
    let mut next = 0u64;
    let mut moves = Vec::new();
    for (root, tiles) in inputs {
        let dataset = es_data::LeRobotDataset::open(root).map_err(|e| fail(root, &e))?;
        let n: u64 = dataset.episodes().iter().map(|m| m.length).sum();
        if tiles.canonicalize().ok().as_ref() != Some(&here) {
            moves.push((*tiles, next, n));
        } else if next != 0 {
            return Err(CliError::Usage(format!(
                "--in-frames {}: these tiles are --frames itself, so its --in has to come \
                 first\n\n{HELP}",
                tiles.display()
            )));
        }
        next += n;
    }
    for (tiles, offset, n) in moves {
        for i in 0..n {
            for ext in ["bin", "json"] {
                let src = tiles.join(format!("{i:06}.{ext}"));
                let dst = out.join(format!("{:06}.{ext}", offset + i));
                if ext == "json" && !src.exists() {
                    continue;
                }
                // A tile of an earlier merge into the same directory.
                let _ = std::fs::remove_file(&dst);
                if std::fs::hard_link(&src, &dst).is_err() {
                    std::fs::copy(&src, &dst).map_err(|e| {
                        CliError::Runtime(format!("{} -> {}: {e}", src.display(), dst.display()))
                    })?;
                }
            }
        }
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!(
            "{}/../../tests/fixtures/visible-learning/{name}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn evaluation() -> EvaluationIr {
        es_ir::serial::evaluation_from_toml(
            &std::fs::read_to_string(fixture("evaluation.toml")).expect("evaluation.toml"),
        )
        .expect("the committed Evaluation IR parses")
    }

    fn deployment() -> DeploymentIr {
        es_ir::serial::deployment_from_toml(
            &std::fs::read_to_string(fixture("deployment.toml")).expect("deployment.toml"),
        )
        .expect("the committed Deployment IR parses")
    }

    fn open(suites: &str, seed: u64, episodes: u32) -> Result<Perturb, CliError> {
        Perturb::open(
            &fixture("evaluation.toml"),
            suites,
            seed,
            episodes,
            &deployment(),
            true,
        )
    }

    /// Packet M13/Z2 oracle: episode `i` runs under `suites[i % len]`, at seed `seed + i`.
    #[test]
    fn collect_assigns_suites_by_episode_index() {
        let p = open("light_intensity,nominal,torque_noise", 500, 7).expect("opens");
        let suites: Vec<&str> = (0..7).map(|i| p.plan.suite_name(p.draw(i).0)).collect();
        assert_eq!(
            suites,
            [
                "light_intensity",
                "nominal",
                "torque_noise",
                "light_intensity",
                "nominal",
                "torque_noise",
                "light_intensity"
            ]
        );
        assert_eq!(p.draw(3).1, 503);
    }

    /// Packet M13/Z2 oracle: what collect draws for a (suite, seed) is what `es eval run`
    /// draws for it -- the reset draws and the per-step streams -- because collect's episode
    /// `i` is episode `i` of an evaluation whose `seed_base` is `--seed`. The evaluation side
    /// is written out the way `es_eval::runner::run_shard_with_sink` and `run_episode` make
    /// it: the plan over *every* suite, `apply_at_reset(cell, seeds[i], i)` and
    /// `StepState::new(.., zeros, seeds[i], cell, i)`. Collect is handed a subset of the
    /// suites in another order, so the suite index and the emptied suites are covered too.
    #[test]
    #[allow(clippy::float_cmp)] // bit for bit is the claim
    fn collect_draws_what_eval_draws_for_a_suite_and_seed() {
        const S: u64 = 500;
        let mut ir = evaluation();
        ir.episodes.seeds = SeedPlan::Base(S);
        let eval =
            PerturbationPlan::compile(&ir, &SceneDesc::default(), &ModelInfo::default(), true)
                .expect("the evaluation's plan");
        let control_us = deployment().rate.control_period().0;
        let chosen = [
            "torque_noise",
            "light_intensity",
            "backlash",
            "observation_delay",
        ];
        let mut p = open(&chosen.join(","), S, 8).expect("opens");
        let light = Cell::new(LightOverride::default());
        let mut moved = 0;
        for i in 0..8u32 {
            let cell = ir
                .suites
                .iter()
                .position(|s| s.name == chosen[i as usize % chosen.len()])
                .expect("a suite of the IR");
            // `resolve_seeds` of `SeedPlan::Base(S)`, entry `i`.
            let seed = S + u64::from(i);
            let mut want = ResetOverrides::default();
            eval.apply_at_reset(cell, seed, u64::from(i), &mut want);
            assert_eq!(p.draw(i), (cell, seed, want), "episode {i}");
            moved += usize::from(want != ResetOverrides::default());

            let mut delay = usize::MAX;
            p.at(
                PerturbAt::Episode {
                    episode: i,
                    observation_delay: &mut delay,
                },
                &light,
            );
            assert_eq!(delay, ms_to_steps(want.observation_delay_ms, control_us));
            assert_eq!(light.get(), want.light);
            let mut eval_step = StepState::new(
                &want,
                ms_to_steps(want.action_delay_ms, control_us),
                &[0.0; 6],
                seed,
                cell,
                u64::from(i),
            );
            for t in 0..5 {
                let mut a = [0.3, -0.2, 0.1, 0.5, -0.4, f64::from(t)];
                let mut b = a;
                p.at(PerturbAt::Actuate(&mut a), &light);
                eval_step.apply_per_step(&mut b);
                assert_eq!(a, b, "episode {i} step {t}");
            }
        }
        assert_eq!(moved, 8, "every chosen suite moves its episodes' overrides");
    }

    /// Packet M13/Z2 oracle: seeds `[S, S+N)` that meet the Evaluation IR's are refused by
    /// name; a range that misses them is not.
    #[test]
    fn collect_refuses_the_evaluations_seeds() {
        let usage = |seed, n| match open("nominal", seed, n) {
            Err(CliError::Usage(m)) => m,
            other => panic!("--seed {seed} --episodes {n} was not refused: {other:?}"),
        };
        let m = usage(100, 5);
        assert!(m.contains("overlaps evaluation seeds 101-116"), "{m}");
        assert!(m.contains("collects seeds 100-104"), "{m}");
        usage(116, 1);
        usage(1, 200);
        assert!(open("nominal", 96, 5).is_ok(), "96-100 misses 101");
        assert!(open("nominal", 117, 200).is_ok());
        // A `seed_base` plan is its base and the next `n_episodes - 1`.
        let mut ir = evaluation();
        ir.episodes.seeds = SeedPlan::Base(40);
        assert!(refuse_evaluation_seeds(&ir, "e.toml", 30, 10).is_ok());
        assert!(refuse_evaluation_seeds(&ir, "e.toml", 56, 1).is_ok());
        assert!(refuse_evaluation_seeds(&ir, "e.toml", 30, 11).is_err());
        assert!(refuse_evaluation_seeds(&ir, "e.toml", 55, 1).is_err());
        assert!(refuse_evaluation_seeds(&ir, "e.toml", 45, 0).is_ok());
    }

    /// Packet M13/Z2 oracle: the collect step's ledger row names the Evaluation IR by path and
    /// by hash, and the suites as given.
    #[test]
    fn collect_ledger_names_the_evaluation_and_its_suites() {
        let p = open("light_intensity,nominal", 500, 4).expect("opens");
        let row: std::collections::BTreeMap<String, String> = p.ledger().into_iter().collect();
        assert_eq!(row["perturb.config"], fixture("evaluation.toml"));
        assert_eq!(
            row["perturb.evaluation_hash"],
            hex(&evaluation().evaluation_hash().expect("hash"))
        );
        assert_eq!(row["perturb.suites"], "light_intensity,nominal");
        assert_eq!(row.len(), 3);
    }

    /// Only the chosen suites are realised: an unchosen light suite needs no renderer, a
    /// chosen one does, and a suite the IR lacks is named beside the ones it has.
    #[test]
    fn collect_realises_only_the_chosen_suites() {
        let open = |suites: &str, renderer: bool| {
            Perturb::open(
                &fixture("evaluation.toml"),
                suites,
                500,
                4,
                &deployment(),
                renderer,
            )
        };
        assert!(open("torque_noise,backlash", false).is_ok());
        let e = open("light_intensity", false).expect_err("no frame source");
        assert!(e.to_string().contains("light_intensity"), "{e}");
        let e = open("nominal,dusk", true).expect_err("unknown suite");
        assert!(matches!(e, CliError::Usage(_)), "{e}");
        assert!(
            e.to_string().contains("\"dusk\"") && e.to_string().contains("backlash"),
            "{e}"
        );
    }

    /// An observation delay the plane's `stale_observation` watchdog could see is refused,
    /// since this path hands the plane age 0; one below it is collected.
    #[test]
    fn collect_refuses_a_delay_the_plane_would_see() {
        let mut deploy = deployment();
        let ir = evaluation();
        let delay = ir
            .suites
            .iter()
            .position(|s| s.name == "observation_delay")
            .expect("suite");
        let control_us = deploy.rate.control_period().0;
        assert!(refuse_unseen_age(&ir, &[delay], &deploy, control_us).is_ok());
        for w in &mut deploy.watchdogs.0 {
            if let Watchdog::StaleObservation { max_age } = w {
                max_age.0 = 30_000;
            }
        }
        let e = refuse_unseen_age(&ir, &[delay], &deploy, control_us).expect_err("40 ms > 30 ms");
        assert!(e.to_string().contains("stale_observation"), "{e}");
        assert!(
            refuse_unseen_age(&ir, &[0], &deploy, control_us).is_ok(),
            "nominal"
        );
    }
}
