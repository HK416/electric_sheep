//! Distillation (spec 13.1, spec 19.2): merge collections, split them deterministically and
//! write the input identity of the training run they feed -- with the per-episode outcome and
//! perturbation rows that ride along outside every hash.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use es_env::Termination;
use serde::Serialize;

use super::{append_loop_step, hex, LoopKind, LoopStep};
use crate::identity::{BaseModel, DatasetIdentity, Split, TrainingIdentity};
use crate::intervention::identity_of;
use crate::lerobot::{LeRobotDataset, LeRobotWriter};
use crate::{write_file, DataError};

/// Where [`distill`] writes the spec 19.3 identity.
pub const TRAINING_IDENTITY_FILE: &str = "training_identity.json";

// --- distillation ---------------------------------------------------------------------------

/// The spec 19.2 split, as the two things that decide it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitSpec {
    /// `[train, val, test]`. `test` takes the remainder, so the three are always an exact
    /// partition.
    pub ratios: [f64; 3],
    pub seed: u64,
}

impl Default for SplitSpec {
    fn default() -> Self {
        Self {
            ratios: [0.8, 0.1, 0.1],
            seed: 0,
        }
    }
}

/// Each episode's suite, seed and reset draws under `es loop collect --perturb` (packet
/// M13/Z2), one JSON line per episode beside `meta/interventions.jsonl` and, like it, outside
/// every hash (`dataset_content_hash` reads the parquet files, `dataset_schema_hash`
/// `info.json`).
pub const PERTURBATIONS_FILE: &str = "meta/perturbations.jsonl";

/// How each episode of a collection ended, one JSON line per episode --
/// `{"episode":0,"termination":"success"}` -- beside [`PERTURBATIONS_FILE`] and, like it,
/// outside every hash (packet M16/H3). The `LeRobot` columns have no place for it and the
/// ledger's collect step only counts; this is what [`distill`]'s `success_only` keeps by.
pub const OUTCOMES_FILE: &str = "meta/outcomes.jsonl";

pub(super) fn write_outcomes(root: &Path, terminations: &[Termination]) -> Result<(), DataError> {
    let mut text = String::new();
    for (i, t) in terminations.iter().enumerate() {
        let t = format!("{t:?}").to_lowercase();
        let _ = writeln!(
            text,
            "{}",
            serde_json::json!({ "episode": i, "termination": t })
        );
    }
    write_file(&root.join(OUTCOMES_FILE), text.as_bytes())
}

/// The episodes of the dataset at `root` whose collect-time outcome was a success
/// ([`OUTCOMES_FILE`]). Refused, not guessed, for a dataset without the file: one collected
/// before packet M16/H3 records only counts.
pub fn successful_episodes(root: &Path) -> Result<BTreeSet<u32>, DataError> {
    if !root.join(OUTCOMES_FILE).is_file() {
        return Err(DataError::Loop(format!(
            "{} has no {OUTCOMES_FILE}, so which of its episodes succeeded is not recorded (a \
             collection older than packet M16/H3 counts them only); collect it again to keep \
             its successes",
            root.display()
        )));
    }
    Ok(read_rows(root, OUTCOMES_FILE)?
        .iter()
        .filter(|row| row["termination"] == "success")
        .filter_map(|row| row["episode"].as_u64().and_then(|e| u32::try_from(e).ok()))
        .collect())
}

/// Merges datasets, splits deterministically, and writes the spec 19.3 identity.
///
/// Episodes are re-indexed `0..N` in input order then original index — a total order, so two
/// runs of the same inputs produce byte-identical parquet and the same `TrainingIdentity`.
/// `meta/interventions.jsonl` of each input is merged with its episode indices remapped, so
/// labels survive the merge, and so is [`PERTURBATIONS_FILE`], so each episode keeps the suite
/// it was collected under (packet M13/Z3); a merge of inputs that have none writes none.
///
/// [`OUTCOMES_FILE`] is carried the same way. With `success_only` (packet M16/H3) only the
/// episodes whose collect-time outcome was a success are merged -- re-indexed, their labels,
/// perturbation and outcome rows carried -- and the step records `keep = success`; an input
/// without [`OUTCOMES_FILE`], or inputs with no success at all, are refused before anything is
/// written. Without it the merge is exactly what it always was.
///
/// The `Distill` loop step is appended to the output root **and** every input root: a
/// dataset's own ledger should record that it was consumed (spec 13.3).
pub fn distill<P: AsRef<Path>>(
    inputs: &[P],
    split: &SplitSpec,
    out_root: &Path,
    success_only: bool,
) -> Result<TrainingIdentity, DataError> {
    if inputs.is_empty() {
        return Err(DataError::Loop(
            "distill needs at least one input dataset".to_owned(),
        ));
    }
    if split.ratios.iter().sum::<f64>() > 1.0 + 1e-9 {
        return Err(DataError::Loop(format!(
            "split ratios {:?} sum above 1.0; that is silent truncation of the test set, \
             not a clamp",
            split.ratios
        )));
    }
    let datasets: Vec<LeRobotDataset> = inputs
        .iter()
        .map(|p| LeRobotDataset::open(p.as_ref()))
        .collect::<Result<_, _>>()?;
    let first = datasets[0].info();
    for (i, ds) in datasets.iter().enumerate().skip(1) {
        if ds.info().features != first.features {
            return Err(DataError::Inconsistent(format!(
                "input {i} ({}) has a different feature set than input 0; merging them would \
                 produce a schema hash that describes neither",
                ds.root().display()
            )));
        }
        if ds.info().fps.to_bits() != first.fps.to_bits() {
            return Err(DataError::Inconsistent(format!(
                "input {i} ({}) runs at {} fps, input 0 at {}",
                ds.root().display(),
                ds.info().fps,
                first.fps
            )));
        }
    }

    let kept: Vec<Option<BTreeSet<u32>>> = datasets
        .iter()
        .map(|ds| {
            success_only
                .then(|| successful_episodes(ds.root()))
                .transpose()
        })
        .collect::<Result<_, _>>()?;
    if success_only && kept.iter().flatten().all(BTreeSet::is_empty) {
        return Err(DataError::Loop(
            "no episode of the inputs succeeded; there is nothing to keep".to_owned(),
        ));
    }

    let mut info = first.clone();
    crate::intervention::ensure_columns(&mut info);
    let mut writer = LeRobotWriter::create(out_root, info)?;
    let mut segments = Vec::new();
    let (mut perturbations, mut outcomes) = (String::new(), String::new());
    let mut next = 0u32;
    for (ds, kept) in datasets.iter().zip(&kept) {
        let mut remap = BTreeMap::new();
        for meta in ds.episodes() {
            if kept
                .as_ref()
                .is_some_and(|k| !k.contains(&meta.episode_index))
            {
                continue;
            }
            let mut ep = ds.read_episode(meta.episode_index)?;
            remap.insert(meta.episode_index, next);
            ep.index = next;
            next += 1;
            writer.write_episode(&ep)?;
        }
        for mut s in crate::intervention::read_segments(ds.root())? {
            if let Some(index) = remap.get(&s.episode) {
                s.episode = *index;
                segments.push(s);
            }
        }
        for (file, text) in [
            (PERTURBATIONS_FILE, &mut perturbations),
            (OUTCOMES_FILE, &mut outcomes),
        ] {
            for mut row in read_rows(ds.root(), file)? {
                let episode = row["episode"].as_u64().and_then(|e| u32::try_from(e).ok());
                if let Some(index) = episode.and_then(|e| remap.get(&e)) {
                    row["episode"] = (*index).into();
                    text.push_str(&row.to_string());
                    text.push('\n');
                }
            }
        }
    }
    writer.finish()?;
    crate::intervention::write_segments(out_root, &segments)?;
    for (file, text) in [
        (PERTURBATIONS_FILE, perturbations),
        (OUTCOMES_FILE, outcomes),
    ] {
        let path = out_root.join(file);
        if text.is_empty() {
            // An earlier merge into the same root must not name rows for these episodes.
            let _ = std::fs::remove_file(&path);
        } else {
            write_file(&path, text.as_bytes())?;
        }
    }

    let merged = LeRobotDataset::open(out_root)?;
    let lists = Split::deterministic(next, split.ratios, split.seed);
    let identity = DatasetIdentity::compute(&merged, &lists)?;
    let training = TrainingIdentity {
        // Everything the training run owns is unknown on this side of spec 2.3's boundary. A
        // zero digest reads as "unset"; a fabricated one would make `training_hash` a lie.
        config: [0; 32],
        optimizer: [0; 32],
        scheduler: [0; 32],
        seed: [0; 32],
        dataset: identity.to_dataset_hash(),
        base_model: BaseModel::default(),
        augmentation: [0; 32],
        precision: [0; 32],
        topology: [0; 32],
        checkpoint_manifest: [0; 32],
        metrics: [0; 32],
        hardware: [0; 32],
    };
    write_json(&out_root.join(TRAINING_IDENTITY_FILE), &training)?;
    // The split lists themselves, so the `dataset_split_hash` above is reproducible by hand
    // and not just by re-running this function (spec 19.2).
    write_json(&out_root.join("split.json"), &lists)?;

    let step = LoopStep::new(LoopKind::Distill)
        .input("seed", &split.seed)
        .input("ratios", &format!("{:?}", split.ratios))
        .output("content", &hex(&identity.content))
        .output("schema", &hex(&identity.schema))
        .output("split", &hex(&identity.split))
        .output("training_hash", &hex(&training.training_hash()?));
    let mut step = if success_only {
        step.input("keep", &"success").output("episodes", &next)
    } else {
        step
    };
    for (i, ds) in datasets.iter().enumerate() {
        let (content, schema) = identity_of(ds)?;
        step = step
            .input(&format!("input.{i}.content"), &hex(&content))
            .input(&format!("input.{i}.schema"), &hex(&schema));
    }
    append_loop_step(out_root, &step)?;
    for ds in &datasets {
        if ds.root() != out_root {
            append_loop_step(ds.root(), &step)?;
        }
    }
    Ok(training)
}

/// The rows of a JSON-lines `file` under `root` ([`PERTURBATIONS_FILE`], [`OUTCOMES_FILE`]);
/// an absent file is none, as with the intervention segments.
fn read_rows(root: &Path, file: &str) -> Result<Vec<serde_json::Value>, DataError> {
    let path = root.join(file);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(DataError::io(&path, e)),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).map_err(|source| DataError::Json {
                path: path.clone(),
                source,
            })
        })
        .collect()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), DataError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|source| DataError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    text.push('\n');
    write_file(path, text.as_bytes())
}
