//! `es dataset info` / `export` / `bake` (spec 19: `LeRobot` dataset, Dataset Identity).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use es_compile::PolicyBundle;
use es_data::{export_v3, Column, DatasetIdentity, ExportOptions, LeRobotDataset, Split};
use es_eval::ObservationBake;
use es_ir::types::ElemType;

use crate::error::CliError;
use crate::util::hex;

const HELP: &str = "\
es dataset info <root>
es dataset export --lerobot-v3 <root> --out <dir> [--frames <dir>] [--drop <a,b>]
                  [--state-dim <n>]
es dataset bake --policy <bundle.esb> --out <dir> [--frames <dir>] [--scene <file.xml>] <root>

`info` opens a LeRobot dataset at <root>, and prints its features (as the spec 5.4 PortType
they present at an Observation IR port), episode count, and the dataset identity's three
hashes (spec 19: content/schema/split) -- computed over a display-only all-train split,
since `dataset info` has no split the caller actually intends to train with.

`export --lerobot-v3` converts the v2.1 dataset at <root> into the v3.0 layout `lerobot`
0.6.1 reads, under <dir>; the source is not modified. <dir>/meta/es_provenance.json records
the source's content/schema/split hashes (spec 19.2), and <dir>/meta/stats.json the
per-feature min/max/mean/std/count `lerobot`'s normalizers read (packet M5/V8).

`bake` runs every frame of the dataset at <root> through the Observation IR the bundle
carries -- the same `CpuPlan`, the same input resolution and the same encoding `es eval
run` uses at inference (`es_eval::ObservationBake`, spec 7.2) -- and writes, under <dir>:
  episode_<NNNNNN>.safetensors   one tensor per Observation IR output plus `action`,
                                 each [frames, ...], F32 (INV-16: safetensors only)
  manifest.json                  observation_hash, task_hash, compiler_hash, the
                                 dataset's content/schema hashes, shapes and frame counts
So training reads what inference computes instead of re-implementing it. `python/es/
train_act.py --baked <dir>` is the consumer.

--frames <dir>  For `export`: raw rendered frames to embed as PNG for each
                `observation.images.<name>` feature, <dir>/<name>/<NNNNNN>.bin, row-major
                u8, the feature's own [height, width, 3], in dataset-global frame order --
                the layout `es_env::render::EnvRenderer` writes. Without it, a declared
                camera is dropped from the export and reported, because there are no
                pixels for it.
                For `bake`: the flat <dir>/<NNNNNN>.bin tiles `es loop collect --frames`
                writes, in dataset-global frame order. An Observation IR with an image
                input and no --frames is refused, never baked with zeros.
--drop <a,b>    For `export`: source columns to leave out. `lerobot` classifies a policy
                feature by name alone, so every `action*` column it sees becomes an action
                head -- `--drop action_commanded,action_source` is what leaves one.
--state-dim <n> For `export`: keep only the leading n values of `observation.state`. The
                recorded row is env 0's whole qpos followed by its whole qvel, and only the
                qpos part can be served back at inference (`es_eval`'s state capture reads
                qpos; there is no qvel arm), so a policy trained on the whole row could
                never be run.
--scene <file>  For `bake`: the scene to load when a channel needs the model's `qpos`
                ranges (a second JointState channel, packet M5/V7a) -- the same file
                `es eval run --scene` and `es loop collect --scene` take. Without it the
                Task IR's `scene.path` is opened relative to the working directory.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("info") => info(&args[1..]),
        Some("export") => export(&args[1..]),
        Some("bake") => bake(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es dataset: unknown subcommand '{other}'"
        ))),
    }
}

pub(crate) fn export(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let (mut root, mut out, mut frames) = (None, None, None);
    let (mut drop, mut state_dim) = (None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let slot = match arg.as_str() {
            "--lerobot-v3" => &mut root,
            "--out" => &mut out,
            "--frames" => &mut frames,
            "--drop" => &mut drop,
            "--state-dim" => &mut state_dim,
            other => {
                return Err(CliError::Usage(format!(
                    "es dataset export: unexpected argument '{other}'\n\n{HELP}"
                )))
            }
        };
        *slot = Some(PathBuf::from(rest.next().ok_or_else(|| {
            CliError::Usage(format!("es dataset export: {arg} needs a value\n\n{HELP}"))
        })?));
    }
    let (Some(root), Some(out)) = (root, out) else {
        return Err(CliError::Usage(format!(
            "es dataset export: --lerobot-v3 <root> and --out <dir> are both required\n\n{HELP}"
        )));
    };

    let opts = ExportOptions {
        drop: drop
            .iter()
            .flat_map(|p| {
                p.to_string_lossy()
                    .split(',')
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|s| !s.is_empty())
            .collect(),
        state_dim: state_dim
            .as_ref()
            .map(|p| {
                p.to_string_lossy().parse::<usize>().map_err(|e| {
                    CliError::Usage(format!(
                        "es dataset export: --state-dim: {e}

{HELP}"
                    ))
                })
            })
            .transpose()?,
    };

    let dataset = LeRobotDataset::open(root).map_err(|e| CliError::Runtime(e.to_string()))?;
    let report = export_v3(&dataset, &out, frames.as_deref(), &opts)
        .map_err(|e| CliError::Runtime(e.to_string()))?;
    println!("wrote: {}", out.display());
    println!("episodes: {}   frames: {}", report.episodes, report.frames);
    println!("cameras: {}", camera_list(&report.cameras));
    for dropped in &report.dropped {
        println!(
            "warning: camera {dropped:?} dropped -- no frames under --frames for it, and an \
             image feature with no pixels is a dangling reference"
        );
    }
    println!("provenance: {}", provenance(&out).display());
    Ok(0)
}

/// `es dataset bake` -- the training set, through the Observation IR (packet M5/V2b).
///
/// The defect this closes: `train_act.py` fed the graph's state port the raw
/// `observation.state` row and re-implemented `Op::Dequantize` for the image, while `capture`
/// ran the compiled plan, which puts a `Normalize` on both. Two implementations of one node
/// is the root cause, so the fix is that there is only one: everything between a recorded
/// value and a baked tensor here is produced by `CpuPlan::run`.
pub(crate) fn bake(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let (mut policy, mut out, mut frames, mut scene, mut root) = (None, None, None, None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let slot = match arg.as_str() {
            "--policy" => &mut policy,
            "--out" => &mut out,
            "--frames" => &mut frames,
            "--scene" => &mut scene,
            other if other.starts_with("--") => {
                return Err(CliError::Usage(format!(
                    "es dataset bake: unexpected argument '{other}'\n\n{HELP}"
                )))
            }
            other => {
                if root.replace(PathBuf::from(other)).is_some() {
                    return Err(CliError::Usage(format!(
                        "es dataset bake: more than one dataset root\n\n{HELP}"
                    )));
                }
                continue;
            }
        };
        *slot = Some(PathBuf::from(rest.next().ok_or_else(|| {
            CliError::Usage(format!("es dataset bake: {arg} needs a value\n\n{HELP}"))
        })?));
    }
    let (Some(policy), Some(out), Some(root)) = (policy, out, root) else {
        return Err(CliError::Usage(format!(
            "es dataset bake: --policy <bundle.esb>, --out <dir> and <root> are all \
             required\n\n{HELP}"
        )));
    };

    let bytes = std::fs::read(&policy)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", policy.display())))?;
    let bundle = PolicyBundle::open(&bytes)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", policy.display())))?;
    // No model first: a channel that reads the leading `dof` of the recorded row needs
    // none, and asking for one would make every bake spawn a physics backend. Only a channel
    // that names a joint needs its `qpos` range, and the sole authority on that layout is the
    // backend that will also run the scene -- deriving it from the MJCF here would be a
    // second implementation of exactly the kind packet M5/V2b removed.
    let mut plan = match ObservationBake::new(&bundle.observation, &bundle.task, None) {
        Ok(plan) => plan,
        Err(model_free) => {
            // `--scene` names the file the way `es eval run` and `es loop collect` do;
            // the Task IR's `scene.path` is relative to the repository and only resolves
            // from its root, which a bake started elsewhere (a test, a server tree) is
            // not.
            let scene_path = scene.as_ref().map_or_else(
                || bundle.task.scene.path.clone(),
                |p| p.to_string_lossy().into_owned(),
            );
            let model = load_model(&scene_path).map_err(|why| {
                CliError::Runtime(format!(
                    "{model_free}\n  and the scene \"{scene_path}\" could not be loaded to \
                     resolve it: {why}"
                ))
            })?;
            ObservationBake::new(&bundle.observation, &bundle.task, Some(&model))
                .map_err(|e| CliError::Runtime(e.to_string()))?
        }
    };
    // Every output is stacked into an F32 tensor, which is the one dtype `write_safetensors`
    // emits (spec 8.4's `runtime.dtype` on this path). A plan that ends in anything else is
    // named here rather than silently narrowed.
    let mut shapes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (name, dtype, shape) in plan.outputs() {
        if dtype != ElemType::F32 {
            return Err(CliError::Runtime(format!(
                "observation output \"{name}\" is {dtype:?}; a baked set is F32 (INV-16)"
            )));
        }
        shapes.insert(name.to_owned(), shape.to_vec());
    }

    let dataset = LeRobotDataset::open(&root).map_err(|e| CliError::Runtime(e.to_string()))?;
    std::fs::create_dir_all(&out)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;

    let mut episodes = Vec::new();
    let mut global = 0u64;
    for meta in dataset.episodes() {
        let episode = dataset
            .read_episode(meta.episode_index)
            .map_err(|e| CliError::Runtime(e.to_string()))?;
        let n = episode.len();
        let state = rows(&episode.columns, "observation.state", n)?;
        let action = rows(&episode.columns, "action", n)?;
        let mut stacked: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        // The episode is where an observation stream ends (spec 7.5), exactly as
        // `run_episode` resets the plan per episode -- otherwise a `TemporalWindow` would read
        // across a boundary in training that it cannot read across at inference.
        plan.reset();
        for (t, row) in state.iter().enumerate() {
            let index = global + t as u64;
            let mut tile = |port: &str| match frames.as_deref() {
                Some(dir) => std::fs::read(dir.join(format!("{index:06}.bin")))
                    .map_err(|e| format!("{}/{index:06}.bin: {e}", dir.display())),
                None => Err(format!(
                    "observation input \"{port}\" is an image and --frames was not given; a \
                     baked set with a zero-filled image channel trains a policy that looks fine"
                )),
            };
            let out = plan
                .frame(row, &mut tile)
                .map_err(|e| CliError::Runtime(format!("episode {}: {e}", meta.episode_index)))?;
            for (name, tensor) in &out {
                let values = tensor
                    .data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                stacked.entry(name.clone()).or_default().extend(values);
            }
        }
        global += n as u64;

        let mut file: es_policy::weights::Checkpoint = stacked
            .into_iter()
            .map(|(name, values)| {
                let mut shape = vec![n as u64];
                shape.extend(shapes[&name].iter().copied());
                (name, (shape, values))
            })
            .collect();
        let dim = action.first().map_or(0, Vec::len) as u64;
        file.insert(
            "action".to_owned(),
            (
                vec![n as u64, dim],
                action.iter().flatten().map(|v| *v as f32).collect(),
            ),
        );
        let name = format!("episode_{:06}.safetensors", meta.episode_index);
        let path = out.join(&name);
        std::fs::write(&path, es_policy::weights::write_safetensors(&file))
            .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
        episodes.push(serde_json::json!({ "file": name, "frames": n }));
        shapes.entry("action".to_owned()).or_insert(vec![dim]);
    }

    // The split is display-only, the way `dataset info` computes one: `content` and `schema`
    // do not depend on it, and they are what names the rows this set was baked from (spec 19.2).
    let split = Split::deterministic(dataset.episodes().len() as u32, [1.0, 0.0, 0.0], 0);
    let identity =
        DatasetIdentity::compute(&dataset, &split).map_err(|e| CliError::Runtime(e.to_string()))?;
    let hashes = &bundle.manifest.hashes;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "observation_hash": hashes.observation.as_ref().map(hex),
        "task_hash": hashes.task.as_ref().map(hex),
        "compiler_hash": hex(&plan.compiler_hash()),
        "dataset_content_hash": hex(&identity.content),
        "dataset_schema_hash": hex(&identity.schema),
        "frames": global,
        "episodes": episodes,
        "tensors": shapes.iter().map(|(name, shape)| {
            (name.clone(), serde_json::json!({ "dtype": "F32", "shape": shape }))
        }).collect::<serde_json::Map<_, _>>(),
    });
    let path = out.join("manifest.json");
    let mut text =
        serde_json::to_string_pretty(&manifest).map_err(|e| CliError::Runtime(e.to_string()))?;
    text.push('\n');
    std::fs::write(&path, text)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;

    println!("wrote: {}", out.display());
    println!("episodes: {}   frames: {global}", dataset.episodes().len());
    for (name, shape) in &shapes {
        println!("  {name:<28} {shape:?}");
    }
    println!(
        "observation_hash: {}",
        hashes.observation.as_ref().map_or("(none)".to_owned(), hex)
    );
    Ok(0)
}

/// One feature column as per-frame rows. A column that is not `n` equal-width rows of floats
/// is a refusal: a ragged or short one reshaped to fit would train on the wrong numbers.
fn rows(
    columns: &BTreeMap<String, Column>,
    name: &str,
    n: usize,
) -> Result<Vec<Vec<f64>>, CliError> {
    let column = columns.get(name).ok_or_else(|| {
        CliError::Runtime(format!(
            "the dataset has no `{name}` column; `es dataset bake` needs the rows the \
             Observation IR and the action target are read from"
        ))
    })?;
    let flat: Vec<f64> = match column {
        Column::F32(v) => v.iter().map(|x| f64::from(*x)).collect(),
        Column::F64(v) => v.clone(),
        other => {
            return Err(CliError::Runtime(format!(
                "`{name}` is {:?}; a baked set is built from floats",
                other.dtype()
            )))
        }
    };
    // `flat.is_empty()` is not covered by the modulo: `0 % n == 0`, and `chunks_exact(0)`
    // panics rather than yielding nothing.
    if n == 0 || flat.is_empty() || flat.len() % n != 0 {
        return Err(CliError::Runtime(format!(
            "`{name}` holds {} values across {n} frames, which is not a whole row per frame",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(flat.len() / n)
        .map(<[f64]>::to_vec)
        .collect())
}

fn camera_list(cameras: &[String]) -> String {
    if cameras.is_empty() {
        "(none)".to_owned()
    } else {
        cameras.join(", ")
    }
}

fn provenance(out: &Path) -> PathBuf {
    out.join("meta").join("es_provenance.json")
}

fn info(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let [root] = args else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let dataset =
        LeRobotDataset::open(root.as_str()).map_err(|e| CliError::Runtime(e.to_string()))?;

    println!("root: {}", dataset.root().display());
    println!("episodes: {}", dataset.episodes().len());
    println!();
    println!("features:");
    for (name, spec) in dataset.features() {
        match spec.port_type() {
            Ok(pt) => println!("  {name:<28} {pt:?}"),
            Err(e) => println!("  {name:<28} <no PortType: {e}>"),
        }
    }

    // `dataset info` has no caller-chosen split, so identity is computed over the whole
    // dataset as a single (all-train) split purely for display.
    let split = Split::deterministic(dataset.episodes().len() as u32, [1.0, 0.0, 0.0], 0);
    let identity =
        DatasetIdentity::compute(&dataset, &split).map_err(|e| CliError::Runtime(e.to_string()))?;
    println!();
    println!("dataset identity (spec 19; split shown is display-only, all-train):");
    println!("  content: {}", hex(&identity.content));
    println!("  schema:  {}", hex(&identity.schema));
    println!("  split:   {}", hex(&identity.split));
    Ok(0)
}

/// The `ModelInfo` of the scene the Task IR names — the `qpos` ranges a joint-named
/// observation channel resolves against (packet M5/V7a).
///
/// `MuJoCoCpuBackend` is the same loader `es loop collect` and `es eval run` use, so the
/// ranges a bake indexes the recorded row with are the ranges the run that recorded it
/// indexed `qpos` with.
fn load_model(scene: &str) -> Result<es_physics_core::ModelInfo, String> {
    use es_physics_core::{LoadConfig, PhysicsBackend};

    let xml = std::fs::read_to_string(scene).map_err(|e| format!("{scene}: {e}"))?;
    let parsed = es_assets::parse_mjcf(&xml).map_err(|e| format!("{scene}: {e}"))?;
    es_physics_backend::MuJoCoCpuBackend::new()
        .load(&parsed.scene, &LoadConfig::default())
        .map_err(|e| e.to_string())
}
