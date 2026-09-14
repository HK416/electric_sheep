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
use es_data::collect::{CollectSpec, Collector, Intervention, SplitSpec};
use es_data::{CollectReport, InterventionSegment};
use es_env::expert::{demo_cfg, ScriptedExpert};
use es_env::Termination;
use es_physics_backend::MuJoCoCpuBackend;
use es_physics_core::backend::ModelInfo;
use es_policy::{PolicyRuntime, TorchRuntime, WeightsSource};

/// The one scripted expert the CLI knows: SO-101, cube into the bin
/// (`docs/design/visible-learning.md` section 5).
const EXPERT_NAME: &str = "so101-pick-place";

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es loop collect --policy <policy.esb> --scene <file.xml|urdf> --episodes <N> --seed <S>
                --out <root> [--backend mujoco-cpu] [--runtime torch] [--max-steps <N>]
                [--expert so101-pick-place]
es loop intervene --dataset <root> --segments <segments.json>
es loop distill --in <root> [--in <root>...] [--train 0.8] [--val 0.1] [--test 0.1]
                [--seed <S>] --out <root>

The three steps of the spec 13.1 learning loop that produce datasets.

collect    Opens the policy bundle (spec 9.6), rolls out <N> episodes through the env
           runtime with the Safety Plane from the bundle's Deployment IR -- the only
           actuator path (INV-12) -- and writes a LeRobot dataset with an `action_source`
           and an `intervention` column per frame (spec 13.2). Checks that the requested
           backend and runtime are available first; when either is not, prints
           `SKIPPED (<reason>)` and exits 3 without faking a run (spec 1.4).
           No mp4 is written: there is no renderer in this build, so an image channel
           becomes a declared video feature with VideoRef placeholders, plus a warning.
           --expert replaces the policy with a scripted demonstration: it drives every
           control tick through the same chunk buffer and the same Safety Plane (INV-12),
           records action_source=Human, and needs no Torch runtime. --policy is still
           required -- the bundle carries the Task, Observation and Deployment IR the
           collector reads -- but its weights are never loaded. A waypoint the arm cannot
           reach ends that episode as a failed demonstration (spec 17.2), written, not dropped.

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

Every step appends a LoopStep to <root>/loop.jsonl so the loop is reproducible (spec 13.3);
a distill step is appended to each input root as well as to the output root.

Exit codes: 0 success, 1 runtime failure, 2 usage error, 3 skipped (nothing ran).
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("collect") => collect(&args[1..]),
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

fn collect_typed<const NJ: usize, const H: usize>(
    spec: &CollectSpec<'_>,
    policy: &mut dyn PolicyRuntime,
    mut expert: Option<&mut ScriptedExpert>,
) -> Result<CollectReport, CliError> {
    // Teleop is real-robot I/O (M3 W1) and stays off this path; the one scripted intervener
    // the CLI offers is `--expert`. `es loop intervene` labels afterwards.
    let mut hook = |_episode: u32, frame: u32, model: &ModelInfo, obs: &[f64]| {
        let Some(expert) = expert.as_deref_mut() else {
            return Intervention::Policy;
        };
        if frame == 0 {
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
    Collector::run::<MuJoCoCpuBackend, _, NJ, H>(
        spec,
        policy,
        MuJoCoCpuBackend::new,
        &mut hook,
        None,
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
    pace(&mut cfg, deploy);
    ScriptedExpert::new(scene, cfg).map_err(|e| CliError::Runtime(e.to_string()))
}

/// Paces the expert to the envelope it will be recorded through (`INV-12`: the demonstration
/// obeys the Safety Plane rather than being corrected by it, and the plane is still the only
/// actuator path). A tenth is held back so nothing lands exactly on a limit.
fn pace(cfg: &mut es_env::ExpertCfg, deploy: &es_ir::deployment::DeploymentIr) {
    let s = &deploy.safety;
    let dt = deploy.rate.control.period_secs_f64();
    let least = |values: &[f64]| {
        values
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(f64::INFINITY, f64::min)
    };
    let step = least(&s.velocity_max) * dt;
    let step = step.min(least(&s.action_rate.first_diff_max));
    let accel = least(&s.acceleration_max) * dt * dt;
    let accel = accel.min(least(&s.action_rate.second_diff_max));
    if step.is_finite() {
        cfg.step_max = 0.9 * step;
    }
    if accel.is_finite() {
        cfg.accel_max = 0.9 * accel;
    }
    cfg.horizon = deploy.action.horizon as u32;
    cfg.execute = deploy.action.execute_chunk as u32;
}

fn collect(args: &[String]) -> Result<u8, CliError> {
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
        ],
    )?;
    let expert_name = one(&pairs, "--expert").map(ToOwned::to_owned);
    let policy_path = required(&pairs, "--policy")?.to_owned();
    let scene_path = required(&pairs, "--scene")?.to_owned();
    let out = PathBuf::from(required(&pairs, "--out")?);
    let episodes: u32 = number(&pairs, "--episodes", 1)?;
    let seed: u64 = number(&pairs, "--seed", 0)?;
    let max_steps: u32 = number(&pairs, "--max-steps", 0)?;
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
    };
    let nj = bundle.deployment.robot.n_joints;
    let h = bundle.deployment.action.horizon;
    let report = dispatch_nj_h!(nj, h, &spec, policy, expert.as_mut())?;

    for w in &report.warnings {
        println!("warning: {w}");
    }
    println!("wrote {}", report.root.display());
    println!("episodes: {}   frames: {}", report.episodes, report.frames);
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
