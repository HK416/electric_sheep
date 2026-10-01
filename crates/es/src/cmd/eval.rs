//! `es eval compare` / `es eval run` (spec 10.5).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use es_compile::PolicyBundle;
use es_eval::runner::{RunEvent, RunSink};
use es_eval::{Evaluation, RunConfig};
use es_ir::evaluation::{AcceptanceResult, EvaluationReport};
use es_physics_backend::{BackendKind, MjWarpBackend, MuJoCoCpuBackend, PhysXBackend};
use es_physics_core::backend::PhysicsBackend;
use es_policy::{PolicyRuntime, TorchRuntime, WeightsSource};

use crate::cmd::telemetry::{Publisher, TelemetryArgs, STAGE_EVAL};
use crate::error::CliError;

mod args;
mod compare;
#[cfg(feature = "render")]
mod frames;
mod workers;

pub(crate) use compare::compare;

use args::parse_run_args;
use compare::value_repr;
#[cfg(feature = "render")]
use frames::{renderer_cfgs, LightRig};
use workers::{shard_thread_env, spawn_shards};

const HELP: &str = "\
es eval compare <A.json> <B.json>

Prints a per-suite/per-metric table comparing two EvaluationReport JSON files (spec
10.5): A's value, B's value, the delta B-A, and a significance column.

EvaluationReport (spec 10.5) carries only per-cell aggregates (MetricValue::Scalar or
::Histogram), never raw per-episode samples, so significance prints `n/a` unless a report
also carries a non-standard `samples: [f64, ...]` array alongside a cell -- in which case
a two-sided Welch t-test p-value is computed (plain Rust, no stats crate) and flagged
when |p| < 0.05.
";

const RUN_HELP: &str = "\
es eval run --config <eval.toml> --policy <policy.esb> --scene <file.xml|urdf> [OPTIONS]
            [--frames <dir>]

Opens the policy bundle (spec 9.6, `PolicyBundle::open`), parses the Evaluation IR from
--config, and checks that the requested physics backend and policy runtime are actually
available before doing anything else -- an evaluation this machine cannot really run is
refused, never faked (spec 1.4). When either is unavailable, prints `SKIPPED (<reason>)`
and exits 3.

Otherwise loads the scene, runs the evaluation (`es_eval::Evaluation::run`) and writes,
under --out:
  report.json         spec 10.5, via `es_eval::write_artifacts`
  evaluation.lock      spec 10.5, via `es_eval::write_artifacts`
  episodes.json        one row per episode: suite, cell, seed, termination, plane steps and
                        changed steps, its own failure-mode buckets (`es_eval::episodes`)
  report.html          a minimal static HTML table rendered from report.json (escaped,
                        no template crate)
`episodes/` replay is not produced (M2 packet CLI-eval-run-import; --frames below is what
renders a run); `report.html` carries no failure-episode links because of that.

With --frames <dir> the run also renders each image input's channel from the scene's own
camera, which is what lets an Observation IR with image inputs be evaluated at all
(spec 7.2). It writes one subdirectory per cell -- a cell being one episode of one suite,
named `<suite>-<NN>` -- holding `<NNNNNN>.bin` plus one `layout.json` (with several image
inputs, one `<cell>/<channel>/` of those per input), and an `events.json` under --out with
one { frame, tick, source, events } record per frame. That is exactly what `es video
mosaic` reads. It needs the `render` feature and a Vulkan device; a build without the
feature refuses the flag rather than running with no frames.

With --jobs N (default 1) the evaluation's (suite, episode) units are partitioned round-robin
over N worker processes: this same binary, re-invoked as `es eval run --shard i/N --shard-out
<file>`. A worker runs the units it owns and judges nothing; the parent sums each suite's
episodes back in episode order and computes the report, the `evaluation_hash` and
`evaluation.lock` itself. So --jobs is a scheduling choice and not a different evaluation: at
the same seeds, and **at the same policy-runtime thread count, which `evaluation.lock` records
and `execution_hash` covers**, the artifacts are byte-identical to --jobs 1
(`evaluation.lock`'s `created` timestamp excepted). The thread count is the caveat and not a
formality: the cap below is `cores/N`, a Torch CPU inference is not bitwise-reproducible
across intra-op thread counts, and a `--jobs` large enough to push `cores/N` below what the
policy's tensors parallelise over therefore moves the trajectory. Measured on the demo's
nominal suite on a 16-core box: --jobs 2 and --jobs 4 are byte-identical to --jobs 1, --jobs 8
is not, and --jobs 1 with OMP_NUM_THREADS=2 exported reproduces the --jobs 8 artifacts exactly
(`docs/design/evaluation-execution.md` 2.7). Since packet M10/W0a the torch runtime reports
the count it runs with, `runtime_threads` in `evaluation.lock` says it in plain text and
`execution_hash` differs when it differs: two reports at two counts are two conditions, never
one broken promise. Export the thread vars yourself to pin it.

The split is by episode, so a one-suite evaluation parallelises too: the demo's 16-episode
nominal suite runs 16-wide. Each unit builds its own `Env` -- seeked to its episode with
`Env::seek_episode`, which draws exactly what replaying the resets before it would have drawn
(spec 6.3) -- its own Safety Plane, chunk buffer and inference queue, so nothing crosses an
episode boundary. --jobs 1 is that same partition with one worker and not a second code path.
N is clamped to suites x episodes.

Each worker's own subprocess (the torch runtime, and whatever math library backs it) is capped
to cores/N threads unless the caller already exported OMP_NUM_THREADS, MKL_NUM_THREADS,
OPENBLAS_NUM_THREADS or TORCH_NUM_THREADS, in which case that choice wins. Left uncapped, N
workers each size their own pool to every core on the box: measured on a 16-core server,
`--jobs 6` ran slower than `--jobs 1` for exactly that reason (design note section 7.11).

A worker that fails stops the run with one error naming the shard, its exit code and its last
line of stderr. A partial report is never written.

With --expert <name> the scripted demonstrator drives instead of the bundle's weights, which
is spec 28.9 rule 1 made runnable: a harness the expert cannot pass is a harness no policy can
pass, and `es loop cycle` runs exactly this before it trains anything. --policy is still
required -- the bundle carries the Task, Observation and Deployment IR the run judges against
-- but its weights are never loaded and no Torch runtime is needed. It needs --frames: the
expert reads the cube's pose out of the state the frame source is handed, which is the same
scaffold the `expert_passes_the_evaluation_harness` oracle uses, and nothing else on this path
hands a policy privileged state. The expert runs a demonstration program (packet M14/Q2): a
value ending in .toml is a program file's path, anything else a built-in name --
so101-pick-place, which is templates/teach/so101-pick-place.toml compiled in -- resolved
exactly as `es loop collect --expert` resolves it. A missing file, a file that does not parse
and an unknown name are refused by name before anything opens.

    --out <dir>        output directory (default: ./eval-out)
    --expert <name | program.toml>
                       drive with the scripted expert instead of the policy's weights
    --frames <dir>     render every step here (needs the `render` feature)
    --traj <dir>       per-episode `.estraj` state trajectories (default <out>/traj)
    --backend <name>   physics backend (default mujoco-cpu); see BACKENDS below
    --runtime <name>   policy runtime; only `torch` is supported (default, spec 2.4)
    --jobs <N>         worker processes for the (suite, episode) units (default 1); 0 is refused
    --shard <i/N>      run only the units of shard i (worker mode); needs --shard-out
    --shard-out <file> where a worker writes its units; implies no report and no lock
    --telemetry <addr> publish the run live on this address (spec 23.1), e.g. 127.0.0.1:7777
    --telemetry-token <t>  required in every client's Hello (spec 25.1); none by default
    --telemetry-image-every <N>  publish the observation frame every N ticks (default 0, never)

With --telemetry <addr> the run binds an `es_telemetry::transport::Server` before it opens
anything -- so a viewer can attach and be subscribed before the first cell -- and publishes,
on four streams: 1 the `cell.begin` / `cell.end` / `suite.end` events, 2 one
[frame, tick, source, violation bits] sample per control tick, 3 one `PerfMetrics` per
finished episode, 4 the observation image every --telemetry-image-every ticks. Publishing is
non-blocking (spec 23.4 gate 9): a subscriber that stops reading loses frames and is counted,
and the run never waits for a socket. What the run *computes* is untouched -- `report.json`
and `events.json` are byte-identical with and without the flag.

--telemetry needs --jobs 1: the units of a --jobs N run are separate processes and only one
of them could own the address.

BACKENDS (spec 17.2, packet M11/X1):
  mujoco-cpu  MuJoCo on the CPU, the reference (spec 17.1): tier 1, bitwise run to run. The
              default; its runs keep the hardware_capability slot every committed lock has.
  mjwarp      MuJoCo Warp on the GPU (needs `mujoco_warp` under ES_PYTHON): tier 2, never
              bitwise against mujoco-cpu. Writes H(\"es.backend.v1\", name, engine version,
              float, tier) into execution_hash's hardware_capability slot (spec 28.14 rule 2).
  newton      refused by its mapping report before anything spawns: the adapter declares no
              actuators or sensors and wires no contacts, so it runs open loop only
              (`es backend compare`).
  physx       NVIDIA PhysX through Isaac Sim, headless (M11/I1; needs ES_ISAAC_PYTHON, the
              Isaac Sim interpreter; ES_PHYSX_DEVICE=cpu|cuda:0 picks the pipeline, cpu by
              default): tier 2, the scene through Isaac Sim's MJCF importer, every importer gap
              a mapping-report row (warnings; a blocking row refuses the run). Writes the same
              identity slot as mjwarp, the pipeline named in the engine version.
evaluation.lock's `backend` block carries the engine version the backend's load reply named.

Exit code: 0 when every acceptance result is Determined{passed: true}; 1 when any failed or
is Unavailable (both printed); 2 on a usage error; 3 when the backend or runtime is
unavailable (distinct from 1: nothing ran).
";

/// `--backend <name>` of `es eval run` and `es loop collect` (packet M11/X1): one of the four
/// spec 17.2 names, or a usage error that lists them.
pub(crate) fn parse_backend(name: &str, help: &str) -> Result<BackendKind, CliError> {
    match BackendKind::from_name(name) {
        Some(kind) => Ok(kind),
        None => Err(CliError::Usage(format!(
            "unknown --backend '{name}': one of {}\n\n{help}",
            BackendKind::ALL.map(BackendKind::name).join(", ")
        ))),
    }
}

/// The stdout line that says which spec 3.5 tier produced a run (`docs/reviews/M11.md` S-4):
/// the tier `kind` declares in its capabilities, by number and name. A line and not a
/// `report.json` field, because a new field would move every committed report.
pub(crate) fn determinism_tier(kind: BackendKind) -> String {
    let tier = match kind {
        BackendKind::MuJoCoCpu => es_physics_backend::mujoco::capabilities(),
        BackendKind::MjWarp => es_physics_backend::mjwarp::capabilities(),
        BackendKind::Newton => es_physics_backend::newton::capabilities(),
        BackendKind::PhysX => es_physics_backend::physx::capabilities(),
    }
    .determinism;
    format!("determinism tier: {} {tier:?} (backend {kind})", tier as u8)
}

/// The spec 14.4 gate for a backend other than the reference, run on the scene before any
/// availability probe or process: an unmapped row with severity `error` refuses the run and
/// names the rows. It is the same report the backend's own `load` checks first.
pub(crate) fn mapping_gate(
    kind: BackendKind,
    scene: &es_assets::scene::SceneDesc,
) -> Result<(), CliError> {
    let report = es_physics_backend::mapping_report(scene, kind);
    if report.blocked {
        return Err(CliError::Runtime(format!("--backend {kind}: {report}")));
    }
    Ok(())
}

/// What a closed-loop verb says about a backend it has no path for (Newton, today).
pub(crate) fn no_closed_loop(kind: BackendKind) -> CliError {
    CliError::Runtime(format!(
        "--backend {kind}: no closed-loop path; the closed-loop verbs run mujoco-cpu, mjwarp \
         and physx"
    ))
}

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("compare") => compare(&args[1..]),
        Some("run") => run(&args[1..], None),
        Some("--help" | "-h") | None => {
            println!("{HELP}\n{RUN_HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es eval: unknown subcommand '{other}'"
        ))),
    }
}

/// `Evaluation::run` (and the `SafetyPlane` inside it) is generic over the joint count and
/// chunk horizon, which a `policy.esb` only reveals at runtime. Rather than a speculative
/// type-erased `PhysicsBackend`/`SafetyPlane` path, this enumerates the (joints, horizon)
/// pairs the fixtures and real robots in this repo actually use.
/// ponytail: a fixed dispatch table, not a runtime-generic solver -- add a pair here when a
/// new robot/horizon combination needs `es eval run`.
macro_rules! dispatch_nj_h {
    ($b:ty; $nj:expr, $h:expr, $($args:expr),+ $(,)?) => {
        match ($nj, $h) {
            (1, 1) => run_typed::<$b, 1, 1>($($args),+),
            (6, 1) => run_typed::<$b, 6, 1>($($args),+),
            (6, 8) => run_typed::<$b, 6, 8>($($args),+),
            (6, 16) => run_typed::<$b, 6, 16>($($args),+),
            (6, 50) => run_typed::<$b, 6, 50>($($args),+),
            (7, 1) => run_typed::<$b, 7, 1>($($args),+),
            // Unitree Go1, packet M6/B1: twelve joints, chunk 1.
            (12, 1) => run_typed::<$b, 12, 1>($($args),+),
            (7, 8) => run_typed::<$b, 7, 8>($($args),+),
            (7, 16) => run_typed::<$b, 7, 16>($($args),+),
            (7, 50) => run_typed::<$b, 7, 50>($($args),+),
            (8, 50) => run_typed::<$b, 8, 50>($($args),+),
            // The Shadow Hand's 20 actuators, one action per tick (plan H, packet M16/H2).
            (20, 1) => run_typed::<$b, 20, 1>($($args),+),
            // The student: 16-row chunks over the same 20 (packet M16/H3).
            (20, 16) => run_typed::<$b, 20, 16>($($args),+),
            (nj, h) => Err(CliError::Runtime(format!(
                "unsupported (n_joints={nj}, horizon={h}); es eval run supports a fixed table \
                 of pairs (crates/es/src/cmd/eval.rs) -- add one for this robot"
            ))),
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn run_typed<B: PhysicsBackend + Default, const NJ: usize, const H: usize>(
    bundle: &PolicyBundle,
    eval_ir: &es_ir::evaluation::EvaluationIr,
    scene: &es_assets::scene::SceneDesc,
    policy: &mut dyn PolicyRuntime,
    cfg: &RunConfig,
    frames_dir: Option<&Path>,
    shard: (u32, u32),
    // `mut` only under `render`, where the sink is offered to the frame-rendering call first.
    #[cfg_attr(not(feature = "render"), allow(unused_mut))] mut sink: Option<&mut RunSink<'_>>,
    seen: Option<&super::r#loop::SeenState>,
) -> Result<es_eval::Shard, CliError> {
    let mut run = |frames: Option<&mut es_eval::runner::DrawnFrameSource<'_>>,
                   sink: Option<&mut RunSink<'_>>| {
        Evaluation::run_shard_with_sink::<B, _, NJ, H>(
            eval_ir,
            &bundle.task,
            scene,
            &bundle.observation,
            policy,
            &bundle.deployment,
            B::default,
            cfg,
            frames,
            frames_dir,
            shard,
            sink,
        )
        .map_err(|e| CliError::Runtime(e.to_string()))
    };
    // The `Gpu` and the renderer live here because a renderer borrows the device, and putting
    // that borrow on `Env` would put a lifetime on a type `es-data` and `es-eval` both name
    // (design note section 7.4).
    #[cfg(feature = "render")]
    if frames_dir.is_some() {
        let cameras = renderer_cfgs(bundle)?;
        let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
            .map_err(|e| CliError::Runtime(format!("no Vulkan device for --frames: {e}")))?;
        // One rig per image input, keyed by the plan input's name (packet M15/N2). A rig
        // builds its renderer on its first frame, so a camera no input reads costs nothing.
        let mut rigs: std::collections::BTreeMap<String, LightRig<'_>> = cameras
            .into_iter()
            .map(|(input, cfg)| (input, LightRig::new(&gpu, scene.clone(), cfg)))
            .collect();
        let mut source =
            |input: &str,
             drawn: &es_env::randomize::RenderOverrides,
             model: &es_physics_core::backend::ModelInfo,
             state: &es_physics_core::backend::StateView<'_>| {
                // `--expert`: the raw `qpos ‖ qvel` row on its way past, because the runner
                // hands the frame source the full state immediately before it calls the
                // policy and the demo's Observation IR carries no cube pose. Once per image
                // input of a step; the row is the same state each time.
                if let Some(seen) = seen {
                    seen.capture(model, state);
                }
                let rig = rigs.get_mut(input).ok_or_else(|| {
                    format!("image input {input} reads no image channel of the Task IR")
                })?;
                rig.frame(drawn, model, state)
            };
        return run(Some(&mut source), sink.as_deref_mut());
    }
    // No renderer, no frame source, so no state reaches an expert -- which is why `es eval
    // run` refuses `--expert` without `--frames` before it opens anything.
    let _ = &seen;
    run(None, sink)
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `report.html` (spec 10.5): a table over the same data as `report.json`, no template crate.
fn write_report_html(report: &EvaluationReport, path: &Path) -> Result<(), CliError> {
    use std::fmt::Write as _;

    let mut html = String::new();
    html.push_str("<!doctype html>\n<meta charset=\"utf-8\">\n<title>Evaluation report</title>\n");
    let _ = write!(
        html,
        "<h1>Evaluation report</h1>\n<p>evaluation_hash: {}<br>execution_hash: {}<br>passed: {}</p>\n",
        crate::util::hex(&report.evaluation_hash),
        crate::util::hex(&report.execution_hash),
        report.passed,
    );
    html.push_str(
        "<h2>cells</h2>\n<table border=\"1\"><tr><th>suite</th><th>metric</th><th>value</th><th>n_episodes</th></tr>\n",
    );
    for c in &report.cells {
        let _ = writeln!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&c.suite),
            escape_html(c.metric.name()),
            escape_html(&value_repr(&c.value)),
            c.n_episodes,
        );
    }
    html.push_str(
        "</table>\n<h2>acceptance</h2>\n<table border=\"1\"><tr><th>suite</th><th>metric</th><th>observed</th><th>passed</th></tr>\n",
    );
    for a in &report.acceptance {
        match a {
            AcceptanceResult::Determined {
                criterion,
                observed,
                passed,
            } => {
                let _ = writeln!(
                    html,
                    "<tr><td>{}</td><td>{}</td><td>{observed}</td><td>{passed}</td></tr>",
                    escape_html(criterion.suite.as_deref().unwrap_or("*")),
                    escape_html(criterion.metric.name()),
                );
            }
            AcceptanceResult::Unavailable { metric, reason } => {
                let _ = writeln!(
                    html,
                    "<tr><td>*</td><td>{}</td><td colspan=\"2\">unavailable: {}</td></tr>",
                    escape_html(metric.name()),
                    escape_html(reason),
                );
            }
        }
    }
    html.push_str("</table>\n");
    std::fs::write(path, html).map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// One evaluation.
///
/// `cycle` is `es loop cycle`'s own publisher (packet M7/E7): a cycle binds one socket before
/// its first stage and every stage speaks on it, so `--telemetry` is not on the argv of a
/// nested `es eval run` and this command binds nothing for it.
pub(crate) fn run(args: &[String], cycle: Option<&mut Publisher>) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{RUN_HELP}");
        return Ok(0);
    }
    let a = parse_run_args(args)?;
    let kind = parse_backend(&a.backend, RUN_HELP)?;
    if a.runtime != "torch" {
        return Err(CliError::Usage(format!(
            "unknown --runtime '{}': only torch is supported\n\n{RUN_HELP}",
            a.runtime
        )));
    }
    // Before anything is loaded: a build that cannot render says so instead of running the
    // whole evaluation and writing no frames.
    #[cfg(not(feature = "render"))]
    if a.frames.is_some() {
        return Err(CliError::Usage(
            "--frames needs the `render` feature; this build links no renderer (it was built \
             with `--no-default-features`). Rebuild with the default features \
             (`cargo build -p es`)."
                .to_owned(),
        ));
    }
    // The expert gate's argument check (packet M14/Q2): the program is read and parsed before
    // anything opens, by the resolver `es loop collect --expert` uses.
    let program = a
        .expert
        .as_deref()
        .map(|e| super::r#loop::ExpertProgram::resolve(e, RUN_HELP))
        .transpose()?;

    // Bound **before** the bundle, the scene or either Python interpreter is opened (packet
    // M7/E4): a viewer that attaches on the printed address is subscribed well before the
    // first cell, which is the difference between watching a run and reading its tail.
    let mut owned = TelemetryArgs {
        addr: a.telemetry,
        token: a.telemetry_token.clone(),
        image_every: a.telemetry_image_every,
    }
    .bind(STAGE_EVAL)?;
    // The cycle's, when there is one; otherwise this command's own. One or the other, never
    // two sockets for one run.
    let mut publisher: Option<&mut Publisher> = cycle.or(owned.as_mut());

    let bytes =
        std::fs::read(&a.policy).map_err(|e| CliError::Runtime(format!("{}: {e}", a.policy)))?;
    let bundle = PolicyBundle::open(&bytes).map_err(|e| CliError::Runtime(e.to_string()))?;
    let config_raw = std::fs::read_to_string(&a.config)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.config)))?;
    let eval_ir = es_ir::serial::evaluation_from_toml(&config_raw)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.config)))?;
    // Spec 13.3 / spec 10.4: an Evaluation IR names the documents it judges, and `es eval run`
    // puts its `evaluation_hash` on the report. Running a bundle built from a different Task
    // or Observation IR would therefore stamp the right hash on numbers from the wrong
    // documents -- a moved render path is exactly that case (packet M7/R5). `PolicyBundle::open`
    // has already checked the other four IRs against each other, so the only rules this can
    // add are the Evaluation IR's own.
    let diags: Vec<_> = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &bundle.task,
        observation: &bundle.observation,
        learning: &bundle.learning,
        deployment: &bundle.deployment,
        evaluation: Some(&eval_ir),
    })
    .into_iter()
    .filter(es_ir::diag::Diagnostic::is_error)
    .collect();
    if !diags.is_empty() {
        return Err(CliError::Runtime(format!(
            "{} does not judge {}:\n{}",
            a.config,
            a.policy,
            diags
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        )));
    }

    // Any backend but the reference is gated on its mapping report before an interpreter is
    // probed (spec 14.4), so a refusal names the rows even where the engine is not installed.
    // `mujoco-cpu` keeps today's order: availability first, the scene after.
    let early_scene = if kind == BackendKind::MuJoCoCpu {
        None
    } else {
        let scene = super::backend::load_scene(&a.scene)?;
        mapping_gate(kind, &scene)?;
        Some(scene)
    };
    if let Err(reason) = es_physics_backend::is_available(kind) {
        println!("SKIPPED ({kind} backend unavailable: {reason})");
        return Ok(3);
    }
    // `--expert` drives every tick itself, so the bundle's weights are never loaded and the
    // Torch runtime is not needed at all -- the same trade `es loop collect --expert` makes.
    if a.expert.is_none() {
        if let Err(reason) = es_policy::torch_runtime::is_available() {
            println!("SKIPPED (torch runtime unavailable: {reason})");
            return Ok(3);
        }
    }
    if a.expert.is_some() && a.frames.is_none() {
        return Err(CliError::Usage(format!(
            "--expert needs --frames: the expert is handed its state through the run's frame \
             source, and nothing else on this path gives a policy the cube's pose\n\n{RUN_HELP}"
        )));
    }

    let scene = match early_scene {
        Some(scene) => scene,
        None => super::backend::load_scene(&a.scene)?,
    };
    // The backend opened once on this scene, for the engine version its load reply names and
    // the `hardware_capability` slot that follows from it (spec 28.14 rule 2). A worker's
    // cells carry no hash, so only the process that merges asks.
    let identity = match a.shard {
        Some(_) => None,
        None => Some(
            es_physics_backend::identify(kind, &scene)
                .map_err(|e| CliError::Runtime(format!("--backend {kind}: {e}")))?,
        ),
    };

    // More workers than units would start interpreters that own nothing; the partition N has
    // to be the one the workers are actually told, so it is clamped before either is decided.
    // The unit is the `(suite, episode)` pair since packet M7/R1, so a one-suite evaluation
    // clamps to its episode count and not to 1. `n_episodes` is the count `es_eval` resolves
    // the seed list to (spec 10.2): an explicit list is its own length, a `seed_base` is
    // `n_episodes`.
    let n_episodes = match &eval_ir.episodes.seeds {
        es_ir::evaluation::SeedPlan::Explicit(list) => list.len(),
        es_ir::evaluation::SeedPlan::Base(_) => eval_ir.episodes.n_episodes as usize,
    };
    let units = eval_ir.suites.len() * n_episodes;
    let jobs = a.jobs.min(units.max(1) as u32);
    // The workers' pool cap (`shard_thread_env`, below), decided once. The parent of a
    // `--jobs N` run never infers, but its runtime is the one `merge` hashes and the one the
    // lock quotes, so its reference process is spawned under the same variables the workers
    // get and reports the count that produced the numbers (packet M10/W0a). `--jobs 1`
    // exports nothing and inherits the ambient pool, as before.
    let thread_env = if jobs > 1 {
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        shard_thread_env(jobs, cores, |name| std::env::var_os(name).is_some())
    } else {
        Vec::new()
    };

    let mut torch = TorchRuntime::with_env(thread_env.iter().cloned());
    // The episode counter the expert's `reset` keys off, shared with the frame source below.
    let seen = super::r#loop::SeenState::default();
    let mut expert = program
        .as_ref()
        .map(|p| super::r#loop::expert_policy(p, &scene, &bundle.deployment, &seen))
        .transpose()?;
    let policy: &mut dyn PolicyRuntime = if let Some(e) = &mut expert {
        e
    } else {
        torch
            .load(
                &bundle.learning,
                &WeightsSource::InMemory(bundle.weights.clone()),
            )
            .map_err(|e| CliError::Runtime(e.to_string()))?;
        &mut torch
    };

    // Always recorded, never a flag to remember: a run that cannot say what the arm did is a
    // run nobody can re-render or audit, and an episode of it is under a megabyte (packet
    // M5/V9). `--traj` only moves it.
    let traj_dir = a.traj.clone().unwrap_or_else(|| a.out.join("traj"));
    let cfg = RunConfig {
        created: now_unix(),
        traj_dir: Some(traj_dir.clone()),
        // The bundle's own `expected_latency_ms`, carried across because this is the only
        // place that has the `LearningGraph`: `Evaluation::run` is handed the four IRs it
        // judges and never the graph (packet M7/T7). The evaluator then turns it into whole
        // control ticks with `es_env::latency_ticks` -- the one latency model, the same one
        // `es loop collect` drives `AsyncInference` with.
        expected_latency_ms: bundle.learning.policy.contract.runtime.expected_latency_ms,
        // All zeros on mujoco-cpu, as before; the backend's identity everywhere else.
        hardware: es_ir::HardwareCapability(identity.as_ref().map_or([0; 32], |i| i.hardware)),
        ..RunConfig::default()
    };
    let nj = bundle.deployment.robot.n_joints;
    let h = bundle.deployment.action.horizon;
    let mut shards = if jobs > 1 {
        println!("es eval run --jobs {jobs}: {units} unit(s) over {jobs} worker(s)");
        spawn_shards(&a, jobs, &thread_env)?
    } else {
        // The sink is a closure and not a trait object of this crate's invention (INV-17):
        // `es-eval` calls it, `Publisher::on` turns what it says into wire frames.
        let publishing = publisher.is_some();
        let mut publish = |event: RunEvent<'_>| {
            if let Some(p) = publisher.as_deref_mut() {
                p.on(event);
            }
        };
        let sink: Option<&mut RunSink<'_>> = if publishing { Some(&mut publish) } else { None };
        let seen = a.expert.is_some().then_some(&seen);
        let shard = a.shard.unwrap_or((0, 1));
        let frames = a.frames.as_deref();
        // One dispatch on the backend, monomorphized: `Env<B>` stays generic (spec 3.4).
        vec![match kind {
            BackendKind::MuJoCoCpu => dispatch_nj_h!(
                MuJoCoCpuBackend; nj, h, &bundle, &eval_ir, &scene, policy, &cfg, frames, shard,
                sink, seen
            ),
            BackendKind::MjWarp => dispatch_nj_h!(
                MjWarpBackend; nj, h, &bundle, &eval_ir, &scene, policy, &cfg, frames, shard,
                sink, seen
            ),
            BackendKind::PhysX => dispatch_nj_h!(
                PhysXBackend; nj, h, &bundle, &eval_ir, &scene, policy, &cfg, frames, shard,
                sink, seen
            ),
            BackendKind::Newton => Err(no_closed_loop(kind)),
        }?]
    };

    // Worker mode stops here: the cells go to the parent and nothing else is written. A
    // report over a subset of the suites would carry a correct `evaluation_hash` (spec 10.4).
    if let Some(path) = &a.shard_out {
        let (i, n) = a.shard.expect("--shard-out implies --shard");
        let text = serde_json::to_string(&shards[0])
            .map_err(|e| CliError::Runtime(format!("serializing shard {i}/{n}: {e}")))?;
        std::fs::write(path, text)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
        println!("shard {i}/{n}: {} unit(s)", shards[0].cells.len());
        return Ok(0);
    }

    let mut events = std::collections::BTreeMap::new();
    for s in &mut shards {
        events.append(&mut s.events);
    }
    let (report, mut lock) = Evaluation::merge(
        &eval_ir,
        &bundle.task,
        &bundle.observation,
        &bundle.deployment,
        policy,
        &cfg,
        &shards,
    )
    .map_err(|e| CliError::Runtime(e.to_string()))?;
    // In plain text beside the `execution_hash` that covers it (spec 5.3, packet M10/W0a).
    // `None` under `--expert`: the torch runtime was never loaded and nothing has a pool.
    lock.runtime_threads = torch.threads();
    // Beside the `backend` block it names (packets M11/X1, M11/R1).
    if let Some(i) = identity {
        lock.backend.engine_version = Some(i.engine_version);
        lock.backend.script_blake3 = i.script_blake3;
    }

    std::fs::create_dir_all(&a.out)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.out.display())))?;
    es_eval::write_artifacts(&report, &lock, &a.out)
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    es_eval::episodes::write_episodes(&es_eval::episodes::episode_rows(&eval_ir, &shards), &a.out)
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    write_report_html(&report, &a.out.join("report.html"))?;
    if let Some(dir) = &a.frames {
        let sink = es_eval::FrameSink {
            dir: dir.clone(),
            events,
        };
        sink.write_events(&a.out.join("events.json"))
            .map_err(|e| CliError::Runtime(e.to_string()))?;
        let frames: usize = sink.events.values().map(Vec::len).sum();
        println!(
            "frames: {frames} in {} cell(s) under {}",
            sink.events.len(),
            sink.dir.display()
        );
    }

    println!("trajectories: {}", traj_dir.display());
    println!("{}", determinism_tier(kind));
    // Only the command that bound the socket closes the account of it: a stage of a cycle
    // would print a running total three more stages are still adding to.
    if let Some(p) = &owned {
        println!("{}", p.summary());
    }

    let mut ok = true;
    for r in &report.acceptance {
        match r {
            AcceptanceResult::Determined { passed: true, .. } => {}
            AcceptanceResult::Determined {
                criterion,
                observed,
                passed: false,
            } => {
                ok = false;
                println!(
                    "FAILED suite={:?} metric={} observed={observed}",
                    criterion.suite,
                    criterion.metric.name()
                );
            }
            AcceptanceResult::Unavailable { metric, reason } => {
                ok = false;
                println!("UNAVAILABLE metric={}: {reason}", metric.name());
            }
        }
    }
    println!("wrote {}", a.out.display());
    Ok(u8::from(!ok))
}
