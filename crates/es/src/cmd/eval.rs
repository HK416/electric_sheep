//! `es eval compare` / `es eval run` (spec 10.5).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use es_compile::PolicyBundle;
use es_eval::runner::{RunEvent, RunSink};
use es_eval::{Evaluation, RunConfig};
use es_ir::evaluation::{AcceptanceResult, EvaluationReport, MetricValue};
use es_physics_backend::MuJoCoCpuBackend;
use es_policy::{PolicyRuntime, TorchRuntime, WeightsSource};

use crate::cmd::telemetry::{Publisher, TelemetryArgs, STAGE_EVAL};
use crate::error::CliError;

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
  report.html          a minimal static HTML table rendered from report.json (escaped,
                        no template crate)
`episodes/` replay is not produced by this build -- there is no renderer yet (M2 packet
CLI-eval-run-import); `report.html` carries no failure-episode links because of that.

With --frames <dir> the run also renders the Task IR's one image channel from the scene's
own camera, which is what lets an Observation IR with an image input be evaluated at all
(spec 7.2). It writes one subdirectory per cell -- a cell being one episode of one suite,
named `<suite>-<NN>` -- holding `<NNNNNN>.bin` plus one `layout.json`, and an `events.json`
under --out with one { frame, tick, source, events } record per frame. That is exactly what
`es video mosaic` reads. It needs the `render` feature and a Vulkan device; a build without
the feature refuses the flag rather than running with no frames.

With --jobs N (default 1) the evaluation's cells -- a cell being one suite, run as its whole
episode list -- are partitioned round-robin over N worker processes: this same binary,
re-invoked as `es eval run --shard i/N --shard-out <file>`. A worker runs the cells it owns
and judges nothing; the parent merges them back into the cell order the sequential run would
have written, and computes the report, the `evaluation_hash` and `evaluation.lock` itself. So
--jobs is a scheduling choice and not a different evaluation: at the same seeds the artifacts
are byte-identical to --jobs 1 (`evaluation.lock`'s `created` timestamp excepted).

The split is by suite and not by episode on purpose. Inside one suite the runner keeps one
`Env`, one `SafetyPlane` and one monotonic chunk sequence for all of the episodes, and
`Env::reset` keys the task's randomization by an episode counter that cannot be seeked -- so
episode 5 of a suite is not reproducible without having run episodes 0..4. An evaluation with
one suite gets no speedup from --jobs, and N is clamped to the suite count.

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
hands a policy privileged state.

    --out <dir>        output directory (default: ./eval-out)
    --expert <name>    drive with the scripted expert instead of the policy's weights
    --frames <dir>     render every step here (needs the `render` feature)
    --traj <dir>       per-episode `.estraj` state trajectories (default <out>/traj)
    --backend <name>   physics backend; only `mujoco-cpu` is supported (default, spec 17.1)
    --runtime <name>   policy runtime; only `torch` is supported (default, spec 2.4)
    --jobs <N>         worker processes for the cells (default 1); 0 is refused
    --shard <i/N>      run only the cells of shard i (worker mode); needs --shard-out
    --shard-out <file> where a worker writes its cells; implies no report and no lock
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

--telemetry needs --jobs 1: the cells of a --jobs N run are separate processes and only one
of them could own the address.

Exit code: 0 when every acceptance result is Determined{passed: true}; 1 when any failed or
is Unavailable (both printed); 2 on a usage error; 3 when the backend or runtime is
unavailable (distinct from 1: nothing ran).
";

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

fn load(path: &str) -> Result<(EvaluationReport, serde_json::Value), CliError> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let report: EvaluationReport = serde_json::from_str(&raw)
        .map_err(|e| CliError::Runtime(format!("{path}: not an EvaluationReport: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    Ok((report, value))
}

/// The non-standard `cells[i].samples` extension a report may carry; the typed
/// [`EvaluationReport`] has no such field (spec 10.5 only defines aggregates).
fn samples_of(raw: &serde_json::Value, idx: usize) -> Option<Vec<f64>> {
    let arr = raw.get("cells")?.get(idx)?.get("samples")?.as_array()?;
    Some(arr.iter().filter_map(serde_json::Value::as_f64).collect())
}

fn scalar(v: &MetricValue) -> Option<f64> {
    match v {
        MetricValue::Scalar(x) => Some(*x),
        MetricValue::Histogram(_) | MetricValue::Unavailable { .. } => None,
    }
}

fn value_repr(v: &MetricValue) -> String {
    match v {
        MetricValue::Scalar(x) => format!("{x:.6}"),
        MetricValue::Histogram(h) => format!("{h:?}"),
        MetricValue::Unavailable { reason } => format!("unavailable ({reason})"),
    }
}

fn print_row(suite: &str, metric: &str, a: &str, b: &str, delta: &str, sig: &str) {
    println!("{suite:<20} {metric:<26} {a:>14} {b:>14} {delta:>14}  {sig}");
}

pub(crate) fn compare(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let [a_path, b_path] = args else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let (a, a_raw) = load(a_path)?;
    let (b, b_raw) = load(b_path)?;

    print_row("SUITE", "METRIC", "A", "B", "DELTA", "SIGNIFICANT");
    for (ia, ca) in a.cells.iter().enumerate() {
        let Some((ib, cb)) = b
            .cells
            .iter()
            .enumerate()
            .find(|(_, cb)| cb.suite == ca.suite && cb.metric == ca.metric)
        else {
            print_row(
                &ca.suite,
                ca.metric.name(),
                &value_repr(&ca.value),
                "-",
                "n/a",
                "n/a",
            );
            continue;
        };
        match (scalar(&ca.value), scalar(&cb.value)) {
            (Some(av), Some(bv)) => {
                let sig = match (samples_of(&a_raw, ia), samples_of(&b_raw, ib)) {
                    (Some(sa), Some(sb)) if sa.len() > 1 && sb.len() > 1 => {
                        let p = welch_t_test(&sa, &sb);
                        format!("p={p:.4}{}", if p < 0.05 { " *" } else { "" })
                    }
                    _ => "n/a (aggregate-only report)".to_owned(),
                };
                print_row(
                    &ca.suite,
                    ca.metric.name(),
                    &format!("{av:.6}"),
                    &format!("{bv:.6}"),
                    &format!("{:+.6}", bv - av),
                    &sig,
                );
            }
            _ => print_row(
                &ca.suite,
                ca.metric.name(),
                &value_repr(&ca.value),
                &value_repr(&cb.value),
                "n/a",
                "n/a",
            ),
        }
    }
    for cb in &b.cells {
        if !a
            .cells
            .iter()
            .any(|ca| ca.suite == cb.suite && ca.metric == cb.metric)
        {
            print_row(
                &cb.suite,
                cb.metric.name(),
                "-",
                &value_repr(&cb.value),
                "n/a",
                "n/a",
            );
        }
    }
    println!();
    println!("A: passed={}   B: passed={}", a.passed, b.passed);
    Ok(0)
}

// --- `es eval run` ---------------------------------------------------------------------------

#[derive(Debug)]
struct RunArgs {
    config: String,
    policy: String,
    scene: String,
    out: PathBuf,
    frames: Option<PathBuf>,
    /// The scripted demonstrator, driving instead of the bundle's weights (spec 28.9 rule 1).
    expert: Option<String>,
    /// Where the per-episode `.estraj` state trajectories go; `<out>/traj` unless named.
    traj: Option<PathBuf>,
    backend: String,
    runtime: String,
    /// Worker processes to partition the cells over. 1 is the sequential path, which is the
    /// same code with one shard.
    jobs: u32,
    /// `(index, count)` when this process *is* a worker.
    shard: Option<(u32, u32)>,
    shard_out: Option<PathBuf>,
    /// Where to publish the run live (packet M7/E4). `None` publishes nothing and binds
    /// nothing.
    telemetry: Option<std::net::SocketAddr>,
    telemetry_token: Option<String>,
    /// Control ticks between two observation images on stream 4; `0` publishes none.
    telemetry_image_every: u64,
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, CliError> {
    let (mut config, mut policy, mut scene, mut out) = (None, None, None, None);
    let (mut backend, mut runtime) = ("mujoco-cpu".to_owned(), "torch".to_owned());
    let (mut frames, mut jobs, mut shard, mut shard_out) = (None, 1u32, None, None);
    let (mut traj, mut expert) = (None, None);
    let (mut telemetry, mut telemetry_token, mut telemetry_image_every) = (None, None, 0u64);

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{RUN_HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(RUN_HELP.to_owned())),
            "--config" => config = Some(val()?.clone()),
            "--policy" => policy = Some(val()?.clone()),
            "--scene" => scene = Some(val()?.clone()),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--frames" => frames = Some(PathBuf::from(val()?)),
            "--expert" => expert = Some(val()?.clone()),
            "--traj" => traj = Some(PathBuf::from(val()?)),
            "--backend" => backend.clone_from(val()?),
            "--runtime" => runtime.clone_from(val()?),
            "--jobs" => {
                let v = val()?;
                jobs = v.parse().map_err(|_| {
                    CliError::Usage(format!("--jobs {v:?} is not a number\n\n{RUN_HELP}"))
                })?;
            }
            "--shard" => shard = Some(parse_shard(val()?)?),
            "--shard-out" => shard_out = Some(PathBuf::from(val()?)),
            "--telemetry" => {
                let v = val()?;
                telemetry = Some(v.parse().map_err(|e| {
                    CliError::Usage(format!(
                        "--telemetry {v:?} is not an address: {e}

{RUN_HELP}"
                    ))
                })?);
            }
            "--telemetry-token" => telemetry_token = Some(val()?.clone()),
            "--telemetry-image-every" => {
                let v = val()?;
                telemetry_image_every = v.parse().map_err(|_| {
                    CliError::Usage(format!(
                        "--telemetry-image-every {v:?} is not a number

{RUN_HELP}"
                    ))
                })?;
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{RUN_HELP}"
                )))
            }
        }
    }
    // `--jobs 0` would be "run nothing and report on it", which is a lie with an exit code.
    if jobs == 0 {
        return Err(CliError::Usage(format!(
            "--jobs 0 runs no cell; the sequential run is --jobs 1\n\n{RUN_HELP}"
        )));
    }
    if shard.is_some() && jobs > 1 {
        return Err(CliError::Usage(format!(
            "--shard is worker mode and --jobs spawns workers; a worker is never a parent\n\n\
             {RUN_HELP}"
        )));
    }
    // A worker that wrote a report would write one covering a subset of the suites, under a
    // correct `evaluation_hash` (spec 10.4). The two flags travel together or not at all.
    if shard.is_some() != shard_out.is_some() {
        return Err(CliError::Usage(format!(
            "--shard and --shard-out go together: a worker's cells are not a report\n\n\
             {RUN_HELP}"
        )));
    }
    // One process, one listener: the cells of a `--jobs N` run happen in N child processes,
    // and a worker that inherited the address would fight the parent for the port.
    if telemetry.is_some() && (jobs > 1 || shard.is_some()) {
        return Err(CliError::Usage(format!(
            "--telemetry publishes one process's run; use --jobs 1 (a --jobs N run's cells              happen in worker processes, which cannot share the address)

{RUN_HELP}"
        )));
    }
    let req = |v: Option<String>, name: &str| {
        v.ok_or_else(|| CliError::Usage(format!("{name} is required\n\n{RUN_HELP}")))
    };
    Ok(RunArgs {
        config: req(config, "--config")?,
        policy: req(policy, "--policy")?,
        scene: req(scene, "--scene")?,
        out: out.unwrap_or_else(|| PathBuf::from("eval-out")),
        frames,
        expert,
        traj,
        backend,
        runtime,
        jobs,
        shard,
        shard_out,
        telemetry,
        telemetry_token,
        telemetry_image_every,
    })
}

/// `i/N`, with the range checked here so a worker cannot be asked for a shard that is not part
/// of the partition.
fn parse_shard(s: &str) -> Result<(u32, u32), CliError> {
    let bad = || CliError::Usage(format!("--shard {s:?} is not `i/N`\n\n{RUN_HELP}"));
    let (i, n) = s.split_once('/').ok_or_else(bad)?;
    let (i, n): (u32, u32) = (i.parse().map_err(|_| bad())?, n.parse().map_err(|_| bad())?);
    if n == 0 || i >= n {
        return Err(CliError::Usage(format!(
            "--shard {i}/{n}: the count must be at least 1 and the index below it\n\n{RUN_HELP}"
        )));
    }
    Ok((i, n))
}

/// `Evaluation::run` (and the `SafetyPlane` inside it) is generic over the joint count and
/// chunk horizon, which a `policy.esb` only reveals at runtime. Rather than a speculative
/// type-erased `PhysicsBackend`/`SafetyPlane` path, this enumerates the (joints, horizon)
/// pairs the fixtures and real robots in this repo actually use.
/// ponytail: a fixed dispatch table, not a runtime-generic solver -- add a pair here when a
/// new robot/horizon combination needs `es eval run`.
macro_rules! dispatch_nj_h {
    ($nj:expr, $h:expr, $($args:expr),+ $(,)?) => {
        match ($nj, $h) {
            (1, 1) => run_typed::<1, 1>($($args),+),
            (6, 1) => run_typed::<6, 1>($($args),+),
            (6, 8) => run_typed::<6, 8>($($args),+),
            (6, 16) => run_typed::<6, 16>($($args),+),
            (6, 50) => run_typed::<6, 50>($($args),+),
            (7, 1) => run_typed::<7, 1>($($args),+),
            // Unitree Go1, packet M6/B1: twelve joints, chunk 1.
            (12, 1) => run_typed::<12, 1>($($args),+),
            (7, 8) => run_typed::<7, 8>($($args),+),
            (7, 16) => run_typed::<7, 16>($($args),+),
            (7, 50) => run_typed::<7, 50>($($args),+),
            (8, 50) => run_typed::<8, 50>($($args),+),
            (nj, h) => Err(CliError::Runtime(format!(
                "unsupported (n_joints={nj}, horizon={h}); es eval run supports a fixed table \
                 of pairs (crates/es/src/cmd/eval.rs) -- add one for this robot"
            ))),
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn run_typed<const NJ: usize, const H: usize>(
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
    let mut run = |frames: Option<&mut es_eval::runner::FrameSource<'_>>,
                   sink: Option<&mut RunSink<'_>>| {
        Evaluation::run_shard_with_sink::<MuJoCoCpuBackend, _, NJ, H>(
            eval_ir,
            &bundle.task,
            scene,
            &bundle.observation,
            policy,
            &bundle.deployment,
            MuJoCoCpuBackend::new,
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
        let rcfg = renderer_cfg(bundle)?;
        let gpu = es_gpu::Gpu::open(es_gpu::GpuOptions::default())
            .map_err(|e| CliError::Runtime(format!("no Vulkan device for --frames: {e}")))?;
        let mut rig = LightRig::new(&gpu, scene.clone(), rcfg);
        let mut source =
            |light: &es_eval::LightOverride,
             model: &es_physics_core::backend::ModelInfo,
             state: &es_physics_core::backend::StateView<'_>| {
                // `--expert`: the raw `qpos ‖ qvel` row on its way past, because the runner
                // hands the frame source the full state immediately before it calls the
                // policy and the demo's Observation IR carries no cube pose.
                if let Some(seen) = seen {
                    seen.capture(model, state);
                }
                rig.frame(light, model, state)
            };
        return run(Some(&mut source), sink.as_deref_mut());
    }
    // No renderer, no frame source, so no state reaches an expert -- which is why `es eval
    // run` refuses `--expert` without `--frames` before it opens anything.
    let _ = &seen;
    run(None, sink)
}

/// The renderer `--frames` needs, built from what the bundle's Task IR already declares.
///
/// The same rule `es loop collect --frames` follows (`crates/es/src/cmd/loop.rs`): one image
/// channel, rendered from the camera its `Frame` names, at the `ImageSpec` the IR declares --
/// so a scene whose camera produces something else is refused rather than silently resampled
/// (`INV-14`).
#[cfg(feature = "render")]
fn renderer_cfg(bundle: &PolicyBundle) -> Result<es_env::EnvRendererCfg, CliError> {
    let images: Vec<_> = bundle
        .task
        .observation_spec
        .channels
        .iter()
        .filter_map(|(name, c)| c.ty.image.as_ref().map(|spec| (name, &c.ty.frame, spec)))
        .collect();
    let [(name, frame, spec)] = images.as_slice() else {
        return Err(CliError::Runtime(format!(
            "--frames needs exactly one image channel in the Task IR's ObservationSpec; it \
             declares {}",
            images.len()
        )));
    };
    let es_ir::types::Frame::Camera(camera) = frame else {
        return Err(CliError::Runtime(format!(
            "image channel {name:?} is not in a camera frame, so there is no camera to render \
             it from"
        )));
    };
    Ok(es_env::EnvRendererCfg::rgb(
        *camera,
        spec.width,
        spec.height,
    ))
}

/// One camera, rendered under one episode's lighting (spec 10.2).
///
/// Not `es_env::EnvRenderer`: a light perturbation sets `RenderConfig::light_dir`, which is
/// fixed when a renderer is built and which `EnvRendererCfg` does not carry -- and V0b owns
/// that surface, so this composes the same public pieces (`es_env::render::render_config`,
/// `body_poses`, `camera_view`) instead of widening it. Everything else is identical, which is
/// why the frames are still the ones the render goldens pin.
///
/// The renderer is rebuilt only when the lighting changes, so a suite with no light
/// perturbation builds exactly one for the whole run.
#[cfg(feature = "render")]
struct LightRig<'gpu> {
    gpu: &'gpu es_gpu::Gpu,
    /// The scene as authored; every episode's scene is derived from this one.
    scene: es_assets::scene::SceneDesc,
    cfg: es_env::EnvRendererCfg,
    lit: Option<(
        es_eval::LightOverride,
        es_assets::scene::SceneDesc,
        es_render::Renderer<'gpu>,
    )>,
}

#[cfg(feature = "render")]
impl<'gpu> LightRig<'gpu> {
    fn new(
        gpu: &'gpu es_gpu::Gpu,
        scene: es_assets::scene::SceneDesc,
        cfg: es_env::EnvRendererCfg,
    ) -> Self {
        Self {
            gpu,
            scene,
            cfg,
            lit: None,
        }
    }

    fn frame(
        &mut self,
        light: &es_eval::LightOverride,
        model: &es_physics_core::backend::ModelInfo,
        state: &es_physics_core::backend::StateView<'_>,
    ) -> Result<Vec<u8>, String> {
        if self.lit.as_ref().is_none_or(|(l, _, _)| l != light) {
            let scene = light.scene(&self.scene);
            let mut rc = es_env::render::render_config(&self.cfg);
            let d = light.rotate_dir([rc.light_dir.x, rc.light_dir.y, rc.light_dir.z]);
            (rc.light_dir.x, rc.light_dir.y, rc.light_dir.z) = (d[0], d[1], d[2]);
            let renderer =
                es_render::Renderer::new(self.gpu, rc).map_err(|e| format!("renderer: {e}"))?;
            self.lit = Some((*light, scene, renderer));
        }
        let (_, scene, renderer) = self.lit.as_mut().expect("just built");
        let world = es_env::render::body_poses(model, state, 0);
        let tri = es_render::TriScene::from_scene_with_poses(scene, &world)
            .map_err(|e| format!("tessellation: {e}"))?;
        let view = es_env::render::camera_view(scene, &self.cfg, &world)
            .map_err(|e| format!("camera: {e}"))?;
        renderer
            .upload_tris(tri)
            .map_err(|e| format!("scene upload: {e}"))?;
        let mut atlas = renderer
            .render(&[view])
            .map_err(|e| format!("render: {e}"))?;
        let tile = atlas
            .read_tile(0, self.cfg.channel)
            .map_err(|e| format!("readback: {e}"))?;
        Ok(tile.to_bytes())
    }
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
    if a.backend != "mujoco-cpu" {
        return Err(CliError::Usage(format!(
            "unknown --backend '{}': only mujoco-cpu is supported\n\n{RUN_HELP}",
            a.backend
        )));
    }
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
            "--frames needs the `render` feature; this build links no renderer (spec 4.2: \
             es-render is layer 5 and the default build of `es` does not pull it in). Rebuild \
             with `cargo build -p es --features render`."
                .to_owned(),
        ));
    }

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

    if let Err(reason) = MuJoCoCpuBackend::is_available() {
        println!("SKIPPED (mujoco-cpu backend unavailable: {reason})");
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

    let scene = super::backend::load_scene(&a.scene)?;

    let mut torch = TorchRuntime::new();
    // The episode counter the expert's `reset` keys off, shared with the frame source below.
    let seen = super::r#loop::SeenState::default();
    let mut expert = match &a.expert {
        Some(name) => Some(super::r#loop::expert_policy(
            name,
            &scene,
            &bundle.deployment,
            &seen,
        )?),
        None => None,
    };
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
        ..RunConfig::default()
    };
    let nj = bundle.deployment.robot.n_joints;
    let h = bundle.deployment.action.horizon;
    // More workers than cells would start interpreters that own nothing; the partition N has
    // to be the one the workers are actually told, so it is clamped before either is decided.
    let jobs = a.jobs.min(eval_ir.suites.len().max(1) as u32);
    let mut shards = if jobs > 1 {
        println!(
            "es eval run --jobs {jobs}: {} cell(s) over {jobs} worker(s)",
            eval_ir.suites.len()
        );
        spawn_shards(&a, jobs)?
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
        vec![dispatch_nj_h!(
            nj,
            h,
            &bundle,
            &eval_ir,
            &scene,
            policy,
            &cfg,
            a.frames.as_deref(),
            a.shard.unwrap_or((0, 1)),
            sink,
            a.expert.is_some().then_some(&seen)
        )?]
    };

    // Worker mode stops here: the cells go to the parent and nothing else is written. A
    // report over a subset of the suites would carry a correct `evaluation_hash` (spec 10.4).
    if let Some(path) = &a.shard_out {
        let (i, n) = a.shard.expect("--shard-out implies --shard");
        let text = serde_json::to_string(&shards[0])
            .map_err(|e| CliError::Runtime(format!("serializing shard {i}/{n}: {e}")))?;
        std::fs::write(path, text)
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
        println!("shard {i}/{n}: {} cell(s)", shards[0].cells.len());
        return Ok(0);
    }

    let mut events = std::collections::BTreeMap::new();
    for s in &mut shards {
        events.append(&mut s.events);
    }
    let (report, lock) = Evaluation::merge(
        &eval_ir,
        &bundle.task,
        &bundle.observation,
        &bundle.deployment,
        policy,
        &cfg,
        &shards,
    )
    .map_err(|e| CliError::Runtime(e.to_string()))?;

    std::fs::create_dir_all(&a.out)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", a.out.display())))?;
    es_eval::write_artifacts(&report, &lock, &a.out)
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

// --- `--jobs N`: one worker process per shard (packet M5/V5) --------------------------------

/// Spawns one `es eval run --shard i/N --shard-out <file>` per shard, all at once, and
/// collects their cells in shard order.
///
/// Processes and not threads, because the physics backend is itself a Python subprocess with
/// one env in it and `TorchRuntime` holds another: threads here would queue behind the same
/// two interpreters (design note section 7.11). The same binary and the same documents, so a
/// worker is this code with a different shard, never a second implementation.
///
/// The children write their frames straight into the shared `--frames` directory: cell names
/// are globally unique and shards own disjoint cells, so there is nothing to merge and nothing
/// to collide.
/// The env vars that size a CPU math-library thread pool to every core on the box by default.
/// `TORCH_NUM_THREADS` is not an official `PyTorch` var but is harmless to set; `PyTorch`'s own
/// intra-op pool falls back to `OMP_NUM_THREADS`/`MKL_NUM_THREADS` when neither
/// `torch.set_num_threads` nor a build-time default has run yet, which is true at process start.
const THREAD_ENV_VARS: [&str; 4] = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCH_NUM_THREADS",
];

/// One shard's fair share of the box's cores, floored at 1. Measured on the 16-core oracle
/// server (design note section 7.11): six shards left uncapped each sized their own subprocess's
/// thread pool to all 16 cores, and `--jobs 6` ran *slower* than `--jobs 1` for it.
fn shard_thread_cap(jobs: u32, cores: usize) -> usize {
    (cores / jobs.max(1) as usize).max(1)
}

/// Which of [`THREAD_ENV_VARS`] this shard should set, and to what -- skipping any the caller
/// already chose a value for, since a value the user exported wins over every shard's guess
/// (`already_set` reads the real environment in production; the test below stubs it).
fn shard_thread_env(
    jobs: u32,
    cores: usize,
    already_set: impl Fn(&str) -> bool,
) -> Vec<(&'static str, String)> {
    let cap = shard_thread_cap(jobs, cores).to_string();
    THREAD_ENV_VARS
        .into_iter()
        .filter(|name| !already_set(name))
        .map(|name| (name, cap.clone()))
        .collect()
}

fn spawn_shards(a: &RunArgs, jobs: u32) -> Result<Vec<es_eval::Shard>, CliError> {
    let exe = std::env::current_exe()
        .map_err(|e| CliError::Runtime(format!("cannot find this executable to re-run it: {e}")))?;
    let dir = a.out.join("shards");
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", dir.display())))?;

    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let thread_env = shard_thread_env(jobs, cores, |name| std::env::var_os(name).is_some());

    let mut running = Vec::new();
    for i in 0..jobs {
        let path = dir.join(format!("{i}.json"));
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["eval", "run", "--config"])
            .arg(&a.config)
            .arg("--policy")
            .arg(&a.policy)
            .arg("--scene")
            .arg(&a.scene)
            .arg("--backend")
            .arg(&a.backend)
            .arg("--runtime")
            .arg(&a.runtime)
            .arg("--shard")
            .arg(format!("{i}/{jobs}"))
            .arg("--shard-out")
            .arg(&path)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(f) = &a.frames {
            cmd.arg("--frames").arg(f);
        }
        if let Some(e) = &a.expert {
            cmd.arg("--expert").arg(e);
        }
        // Cell names are globally unique and shards own disjoint cells, so the trajectories
        // share one directory exactly as the frames do, with nothing to merge.
        cmd.arg("--traj")
            .arg(a.traj.clone().unwrap_or_else(|| a.out.join("traj")));
        for (name, value) in &thread_env {
            cmd.env(name, value);
        }
        let child = cmd
            .spawn()
            .map_err(|e| CliError::Runtime(format!("spawning {}: {e}", exe.display())))?;
        running.push((i, path, child));
    }

    // Every child is waited on before anything is returned, so a failure does not leave the
    // rest of them orphaned behind a `?`.
    let mut shards = Vec::with_capacity(running.len());
    let mut failed: Option<CliError> = None;
    for (i, path, child) in running {
        let out = match child.wait_with_output() {
            Ok(out) => out,
            Err(e) => {
                failed.get_or_insert(CliError::Runtime(format!(
                    "waiting for shard {i}/{jobs}: {e}"
                )));
                continue;
            }
        };
        if !out.status.success() {
            failed.get_or_insert(shard_failed(
                i,
                jobs,
                out.status.code(),
                &String::from_utf8_lossy(&out.stderr),
            ));
            continue;
        }
        match std::fs::read(&path)
            .map_err(|e| format!("{}: {e}", path.display()))
            .and_then(|b| {
                serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))
            }) {
            Ok(s) => shards.push(s),
            Err(e) => {
                failed.get_or_insert(CliError::Runtime(format!(
                    "shard {i}/{jobs} exited 0 but its cells are unreadable: {e}"
                )));
            }
        }
    }
    if let Some(e) = failed {
        return Err(e);
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(shards)
}

/// The one error a failed worker becomes: which shard, what it exited with, and the last thing
/// it said. Never a partial report -- a merge missing a suite would wear a correct
/// `evaluation_hash` over numbers nobody measured (spec 10.4).
fn shard_failed(index: u32, count: u32, code: Option<i32>, stderr: &str) -> CliError {
    let last = stderr
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)");
    CliError::Runtime(format!(
        "es eval run --shard {index}/{count} failed ({}): {last}",
        code.map_or_else(|| "killed by a signal".to_owned(), |c| format!("exit {c}")),
    ))
}

// --- Welch's t-test, in plain Rust (no stats crate) -----------------------------------------

/// Two-sided Welch t-test p-value for two independent samples of unequal variance.
fn welch_t_test(a: &[f64], b: &[f64]) -> f64 {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let var = |xs: &[f64], m: f64| {
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0)
    };
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let se2 = va / na + vb / nb;
    if se2 <= 0.0 {
        return if (ma - mb).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    let t = (ma - mb) / se2.sqrt();
    // Welch-Satterthwaite degrees of freedom.
    let df = se2 * se2 / ((va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0));
    (student_t_sf(t.abs(), df) * 2.0).min(1.0)
}

/// Student-t survival function `P(T > t)` for `t >= 0`, via the regularized incomplete beta
/// function: `sf(t, df) = 0.5 * I_x(df/2, 1/2)` with `x = df / (df + t^2)`.
fn student_t_sf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * reg_incomplete_beta(x, df / 2.0, 0.5)
}

/// The regularized incomplete beta function `I_x(a, b)`, via its continued fraction
/// (Numerical Recipes 6.4), with the standard symmetry swap for faster convergence.
fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(x, a, b) / a
    } else {
        1.0 - front * betacf(1.0 - x, b, a) / b
    }
}

/// Lentz's continued fraction for the incomplete beta function. Parameter names follow
/// Numerical Recipes' own notation, so the short names stay.
#[allow(clippy::many_single_char_names)]
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAXIT: u32 = 200;
    const EPS: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let mf = f64::from(m);
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Lanczos approximation of `ln(gamma(x))`, g=7, n=9 -- accurate to ~1e-13 for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: keeps the argument in the domain the series was fit for.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let g = 7.0_f64;
    let mut acc = 0.999_999_999_999_809_9_f64;
    for (i, c) in COEF.iter().enumerate() {
        acc += c / (x + i as f64 + 1.0);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut v = vec![
            "--config".to_owned(),
            "e.toml".to_owned(),
            "--policy".to_owned(),
            "p.esb".to_owned(),
            "--scene".to_owned(),
            "s.xml".to_owned(),
        ];
        v.extend(extra.iter().map(|s| (*s).to_owned()));
        v
    }

    fn usage(extra: &[&str]) -> String {
        match parse_run_args(&args(extra)) {
            Err(CliError::Usage(text)) => text,
            other => panic!("{extra:?} was not refused: {other:?}"),
        }
    }

    /// The default is the sequential run, and it is the same partition the sharded path uses.
    #[test]
    fn jobs_defaults_to_one_and_names_no_shard() {
        let a = parse_run_args(&args(&[])).expect("the bare form parses");
        assert_eq!(a.jobs, 1);
        assert_eq!(a.shard, None);
        assert_eq!(a.shard_out, None);
    }

    /// `--jobs 0` is "run no cell and report on it". Refused before anything is opened.
    #[test]
    fn jobs_zero_is_refused() {
        assert!(usage(&["--jobs", "0"]).contains("--jobs 0 runs no cell"));
    }

    /// A worker's cells are not a report (spec 10.4): the two flags travel together, and a
    /// worker is never also a parent.
    #[test]
    fn a_shard_needs_a_shard_out_and_refuses_to_be_a_parent() {
        assert!(usage(&["--shard", "0/4"]).contains("go together"));
        assert!(usage(&["--shard-out", "s.json"]).contains("go together"));
        assert!(
            usage(&["--shard", "0/4", "--shard-out", "s.json", "--jobs", "2"])
                .contains("never a parent")
        );
    }

    /// The partition is checked where it is parsed, so no worker is ever asked for a shard
    /// outside it.
    #[test]
    fn a_shard_outside_the_partition_is_refused() {
        assert!(usage(&["--shard", "4/4", "--shard-out", "s.json"]).contains("index below it"));
        assert!(usage(&["--shard", "0/0", "--shard-out", "s.json"]).contains("index below it"));
        assert!(usage(&["--shard", "0-4", "--shard-out", "s.json"]).contains("is not `i/N`"));
    }

    /// A lost worker is one named error carrying enough to act on: which shard, how it died,
    /// and the last thing it printed.
    #[test]
    fn a_failed_shard_names_itself() {
        let CliError::Runtime(text) =
            shard_failed(2, 6, Some(3), "loading\nSKIPPED (no torch)\n\n")
        else {
            panic!("a failed shard must be a runtime error");
        };
        assert!(text.contains("--shard 2/6"), "{text}");
        assert!(text.contains("exit 3"), "{text}");
        assert!(text.contains("SKIPPED (no torch)"), "{text}");

        let CliError::Runtime(text) = shard_failed(0, 2, None, "") else {
            panic!("a killed shard must be a runtime error");
        };
        assert!(text.contains("killed by a signal"), "{text}");
        assert!(text.contains("(no output)"), "{text}");
    }

    /// Six shards on a 16-core box get 2 cores each, one shard gets the whole box, and a shard
    /// count above the core count still gets at least one (never a zero-thread pool).
    #[test]
    fn shard_thread_cap_divides_the_box_and_floors_at_one() {
        assert_eq!(shard_thread_cap(6, 16), 2);
        assert_eq!(shard_thread_cap(1, 16), 16);
        assert_eq!(shard_thread_cap(32, 16), 1);
    }

    /// The fix caps a pool the user left unset; it does not override one they chose. Regression
    /// for the oracle server measurement (design note section 7.11): uncapped, `--jobs 6` each
    /// sized their own torch subprocess to all 16 cores and the run was slower than `--jobs 1`.
    #[test]
    fn shard_thread_env_skips_a_var_the_user_already_set() {
        let env = shard_thread_env(6, 16, |name| name == "OMP_NUM_THREADS");
        assert!(!env.iter().any(|(name, _)| *name == "OMP_NUM_THREADS"));
        assert_eq!(env.len(), THREAD_ENV_VARS.len() - 1);
        assert!(env
            .iter()
            .any(|(name, value)| *name == "MKL_NUM_THREADS" && value == "2"));
    }

    /// Nothing set by the caller: every var is capped to the same fair share.
    #[test]
    fn shard_thread_env_caps_every_var_by_default() {
        let env = shard_thread_env(4, 16, |_| false);
        assert_eq!(env.len(), THREAD_ENV_VARS.len());
        assert!(env.iter().all(|(_, value)| value == "4"));
    }
}
