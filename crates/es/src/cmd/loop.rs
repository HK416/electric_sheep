//! `es loop collect / intervene / distill` (spec 13.1: each step of the learning loop is a
//! CLI command and an artifact).
//!
//! Read `docs/design/learning-loop.md` first; the packet is
//! `docs/packets/M3/W7-learning-loop.md`. `es loop train` deliberately does not exist: the
//! training run is on the Python side of spec 2.3's split, and `distill` produces its input
//! identity rather than pretending to run it.

use std::path::PathBuf;

use es_assets::scene::{JointKind, SceneDesc};
use es_compile::PolicyBundle;
use es_data::collect::{
    CollectEvent, CollectSink, CollectSpec, Collector, Intervention, SplitSpec,
};
use es_data::{CollectReport, InterventionSegment};
use es_env::expert::{demo_cfg, ScriptedExpert};
use es_env::Termination;
use es_ir::types::ElemType;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::ModelInfo;
use es_policy::{PolicyInfo, PolicyRuntime, TorchRuntime, WeightsSource};

/// The one scripted expert the CLI knows: SO-101, cube into the bin
/// (`docs/design/visible-learning.md` section 5).
const EXPERT_NAME: &str = "so101-pick-place";

use crate::cmd::telemetry::{Publisher, TelemetryArgs, STAGE_COLLECT};
use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es loop collect --policy <policy.esb> --scene <file.xml|urdf> --episodes <N> --seed <S>
                --out <root> [--backend mujoco-cpu] [--runtime torch] [--max-steps <N>]
                [--expert so101-pick-place] [--frames <dir>] [--traj <dir>]
                [--telemetry <addr>] [--telemetry-token <t>] [--telemetry-image-every <N>]
es loop intervene --dataset <root> --segments <segments.json>
es loop distill --in <root> [--in <root>...] [--train 0.8] [--val 0.1] [--test 0.1]
                [--seed <S>] --out <root>
es loop cycle --recipe <cycle.toml> [--out <dir>] [--dry-run] [--from <stage>]
              [--allow-new-evaluation] [--skip-expert-gate]     (see `es loop cycle --help`)

The three steps of the spec 13.1 learning loop that produce datasets.

collect    Opens the policy bundle (spec 9.6), rolls out <N> episodes through the env
           runtime with the Safety Plane from the bundle's Deployment IR -- the only
           actuator path (INV-12) -- and writes a LeRobot dataset with an `action_source`
           and an `intervention` column per frame (spec 13.2). Checks that the requested
           backend and runtime are available first; when either is not, prints
           `SKIPPED (<reason>)` and exits 3 without faking a run (spec 1.4).
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
           With --telemetry <addr> the collection publishes what it is doing, on the streams
           `es eval run --telemetry` uses (packet M7/E7): 1 the episode.begin / episode.end
           events, 2 one [frame, tick, source, violation bits] sample per control tick -- the
           plane's own verdict, the same the dataset's action_source column records -- and 4
           the rendered observation every --telemetry-image-every ticks, which needs --frames.
           Publishing is non-blocking and computes nothing extra: the dataset under --out is
           byte-identical with and without the flag.

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

distill    Merges datasets (episodes re-indexed, intervention labels remapped), computes the
           spec 19.2 deterministic split and writes training_identity.json (spec 19.3),
           split.json and a loop step. The training run itself is PyTorch-side and is not
           run here, so every TrainingIdentity slot but `dataset` is an all-zero digest.

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
    ($nj:expr, $h:expr, $($args:expr),+ $(,)?) => {
        match ($nj, $h) {
            (1, 1) => collect_typed::<1, 1>($($args),+),
            (2, 2) => collect_typed::<2, 2>($($args),+),
            (6, 1) => collect_typed::<6, 1>($($args),+),
            (6, 8) => collect_typed::<6, 8>($($args),+),
            (6, 16) => collect_typed::<6, 16>($($args),+),
            (6, 50) => collect_typed::<6, 50>($($args),+),
            (7, 1) => collect_typed::<7, 1>($($args),+),
            (7, 8) => collect_typed::<7, 8>($($args),+),
            (7, 16) => collect_typed::<7, 16>($($args),+),
            (7, 50) => collect_typed::<7, 50>($($args),+),
            (8, 50) => collect_typed::<8, 50>($($args),+),
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

fn collect_typed<const NJ: usize, const H: usize>(
    spec: &CollectSpec<'_>,
    policy: &mut dyn PolicyRuntime,
    mut expert: Option<&mut ScriptedExpert>,
    frames: Option<&std::path::Path>,
    publisher: Option<&mut Publisher>,
) -> Result<CollectReport, CliError> {
    // One publisher, two hooks: the collector's own sink says what the plane did and the
    // frame sink has the pixels. A `RefCell` because both closures live at once and the run
    // is single-threaded -- neither hook can be entered from inside the other (packet M7/E7).
    let publishing = publisher.is_some();
    let publisher = publisher.map(std::cell::RefCell::new);
    let mut publish = |event: CollectEvent| {
        if let Some(p) = &publisher {
            p.borrow_mut().on_collect(event);
        }
    };
    let sink: Option<CollectSink<'_>> = publishing.then_some(&mut publish);
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
        let mut frame_sink =
            |model: &ModelInfo, state: &es_physics_core::backend::StateView<'_>| {
                let tile = renderer.frame(model, state, 0).map_err(|e| e.to_string())?;
                // The tile the run already rendered, borrowed, not a second render (packet
                // M7/E7); the publisher decides whether this is one of the published ones.
                if let (Some(p), Some(bytes)) = (&publisher, tile.as_u8()) {
                    p.borrow_mut().observation(tile.shape, bytes);
                }
                Ok(())
            };
        return Collector::run_with_sink::<MuJoCoCpuBackend, _, NJ, H>(
            spec,
            policy,
            MuJoCoCpuBackend::new,
            &mut hook,
            Some(&mut frame_sink),
            sink,
        )
        .map_err(|e| CliError::Runtime(e.to_string()));
    }
    #[cfg(not(feature = "render"))]
    if frames.is_some() {
        return Err(CliError::Runtime(
            "--frames needs the `render` feature; this build links no renderer (spec 4.2: \
             es-render is layer 5 and the default build of `es` does not pull it in). Rebuild \
             with `cargo build -p es --features render`."
                .to_owned(),
        ));
    }
    Collector::run_with_sink::<MuJoCoCpuBackend, _, NJ, H>(
        spec,
        policy,
        MuJoCoCpuBackend::new,
        &mut hook,
        None,
        sink,
    )
    .map_err(|e| CliError::Runtime(e.to_string()))
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

/// The expert `es eval run --expert <name>` drives with, paced by the same Deployment IR the
/// collection path paces it with.
pub(crate) fn expert_policy(
    name: &str,
    scene: &SceneDesc,
    deploy: &es_ir::deployment::DeploymentIr,
    seen: &SeenState,
) -> Result<ExpertPolicy, CliError> {
    Ok(ExpertPolicy {
        expert: build_expert(name, scene, deploy)?,
        seen: seen.clone(),
        nj: deploy.robot.n_joints,
        horizon: deploy.action.horizon,
        episode: 0,
    })
}

/// The scripted expert for `--expert <name>`, built from the scene it will drive.
///
/// The cube is the scene's one free-joint body: a demonstration that picks something up needs
/// something that can be picked up, and naming it by id would put a scene detail in a flag.
fn build_expert(
    name: &str,
    scene: &SceneDesc,
    deploy: &es_ir::deployment::DeploymentIr,
) -> Result<ScriptedExpert, CliError> {
    if name != EXPERT_NAME {
        return Err(CliError::Usage(format!(
            "unknown --expert '{name}': only {EXPERT_NAME} exists\n\n{HELP}"
        )));
    }
    let free: Vec<_> = scene
        .joints
        .iter()
        .filter(|j| j.kind == JointKind::Free)
        .collect();
    let [joint] = free.as_slice() else {
        return Err(CliError::Runtime(format!(
            "--expert {EXPERT_NAME} needs exactly one free-joint body to pick up; the scene has {}",
            free.len()
        )));
    };
    let mut cfg = demo_cfg(joint.id);
    // Both paths replan at the deployment's inference rate and execute the chunk's rows in
    // between, so the rows that actually execute per chunk are the re-plan period -- one
    // definition, one number, on collection and evaluation alike (packet M5/V17).
    let replan = es_env::replan_interval(deploy.rate)
        .map_err(|e| CliError::Runtime(e.to_string()))?
        .min(deploy.action.execute_chunk as u64);
    cfg.pace_to(deploy, replan as u32);
    ScriptedExpert::new(scene, cfg).map_err(|e| CliError::Runtime(e.to_string()))
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
        ],
    )?;
    let expert_name = one(&pairs, "--expert").map(ToOwned::to_owned);
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
    let backend = one(&pairs, "--backend").unwrap_or("mujoco-cpu");
    let runtime = one(&pairs, "--runtime").unwrap_or("torch");
    if backend != "mujoco-cpu" {
        return Err(CliError::Usage(format!(
            "unknown --backend '{backend}': only mujoco-cpu is supported\n\n{HELP}"
        )));
    }
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

    // Before touching the scene: a run this machine cannot really do is skipped, never faked.
    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIPPED (mujoco-cpu backend unavailable: {reason})");
        return Ok(3);
    }
    // `--expert` drives every tick itself, so the bundle's weights are never loaded and the
    // Torch runtime is not needed at all (design note section 5.1).
    if expert_name.is_none() {
        if let Err(reason) = es_policy::torch_runtime::is_available() {
            println!("SKIPPED (torch runtime unavailable: {reason})");
            return Ok(3);
        }
    }

    let scene = super::backend::load_scene(&scene_path)?;
    let mut expert = match &expert_name {
        Some(name) => Some(build_expert(name, &scene, &bundle.deployment)?),
        None => None,
    };
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
    let report = dispatch_nj_h!(
        nj,
        h,
        &spec,
        policy,
        expert.as_mut(),
        frames.as_deref(),
        publisher
    )?;

    for w in &report.warnings {
        println!("warning: {w}");
    }
    println!("wrote {}", report.root.display());
    println!("episodes: {}   frames: {}", report.episodes, report.frames);
    println!("trajectories: {}", traj_dir.display());
    let count = |t: Termination| report.terminations.iter().filter(|x| **x == t).count();
    println!(
        "terminations: success {}   failure {}   timeout {}   running {}",
        count(Termination::Success),
        count(Termination::Failure),
        count(Termination::Timeout),
        count(Termination::Running)
    );
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

fn distill(args: &[String]) -> Result<u8, CliError> {
    let pairs = parse(
        args,
        &["--in", "--out", "--train", "--val", "--test", "--seed"],
    )?;
    let inputs: Vec<PathBuf> = pairs
        .iter()
        .filter(|(f, _)| f == "--in")
        .map(|(_, v)| PathBuf::from(v))
        .collect();
    if inputs.is_empty() {
        return Err(CliError::Usage(format!(
            "at least one --in <root> is required\n\n{HELP}"
        )));
    }
    let out = PathBuf::from(required(&pairs, "--out")?);
    let split = SplitSpec {
        ratios: [
            number(&pairs, "--train", 0.8)?,
            number(&pairs, "--val", 0.1)?,
            number(&pairs, "--test", 0.1)?,
        ],
        seed: number(&pairs, "--seed", 0)?,
    };

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
