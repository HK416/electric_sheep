//! Checkpoints: each mark's bundle and `checkpoint.manifest` row, and which marks the trainer
//! has finished writing, so they are bundled while it still runs (packet M13/Z1).

use std::path::{Path, PathBuf};

use es_compile::PolicyBundle;
use es_data::training::{Plan, Step, StepKind};
use serde_json::{json, Value};

use super::{bad, one_line, TrainWatch};
use crate::error::CliError;
use crate::util::hex;

/// The checkpoints bundled while the trainer is still running (packet M13/Z1): the plan's own
/// pack or import step for a mark, run as soon as [`CheckpointWatch`] says the mark is written,
/// and skipped by the plan loop afterwards.
///
/// A bundle that fails here is not the run's failure: the mark stays in the plan, and the loop
/// after the trainer bundles it the way it always has, and fails there if it still does.
pub(super) struct Early<'p> {
    pub(super) plan: &'p Plan,
    pub(super) out: &'p Path,
    pub(super) marks: CheckpointWatch,
    pub(super) rows: &'p mut Vec<Value>,
}

impl Early<'_> {
    /// The trainer says it is past optimizer step (or iteration) `step`.
    pub(super) fn past(&mut self, step: u64, watch: &mut TrainWatch<'_>) {
        for mark in self.marks.past(step) {
            let Some(s) = self.plan.steps.iter().find(|s| {
                s.step == Some(mark)
                    && matches!(s.kind, StepKind::PolicyPack | StepKind::PolicyImportLerobot)
            }) else {
                continue;
            };
            println!("$ {}", one_line(s));
            let made = if s.kind == StepKind::PolicyPack {
                crate::cmd::policy::pack(&s.args)
            } else {
                crate::cmd::policy::import_lerobot(&s.args)
            };
            match made.and_then(|_| published_row(s, self.out, watch)) {
                Ok(row) => self.rows.push(row),
                Err(e) => println!(
                    "note: checkpoint {mark} was not bundled while training ({e}); it is \
                     bundled again after the trainer exits"
                ),
            }
        }
    }
}

/// [`manifest_row`], announced: a viewer learns that a mark was packed and which policy it
/// is, at the moment the bundle exists on disk (packet M7/E7).
pub(super) fn published_row(
    step: &Step,
    out: &Path,
    watch: &mut TrainWatch<'_>,
) -> Result<Value, CliError> {
    let row = manifest_row(step, out)?;
    if let Some(p) = watch.publisher.as_deref_mut() {
        p.checkpoint(
            &row["step"].to_string(),
            row["bundle_policy_hash"].as_str().unwrap_or("-"),
        );
    }
    if let (Some(hook), Some(mark)) = (watch.on_checkpoint.as_deref_mut(), step.step) {
        hook(mark);
    }
    Ok(row)
}

/// The word after `flag` on a step's command line.
fn arg<'s>(step: &'s Step, flag: &str) -> Option<&'s String> {
    step.args.iter().skip_while(|a| *a != flag).nth(1)
}

/// One `checkpoint.manifest` row, read back from the bundle that was just written so the
/// digest names bytes on disk rather than bytes in memory.
fn manifest_row(step: &Step, out: &Path) -> Result<Value, CliError> {
    let path = arg(step, "--out").ok_or_else(|| bad("a checkpoint step with no --out"))?;
    let bytes = std::fs::read(path).map_err(|e| bad(format!("{path}: {e}")))?;
    let bundle = PolicyBundle::open(&bytes).map_err(|e| bad(format!("{path}: {e}")))?;
    let relative = Path::new(path)
        .strip_prefix(out)
        .unwrap_or(Path::new(path))
        .to_string_lossy()
        .replace('\\', "/");
    Ok(json!({
        "step": step.step,
        "bundle": relative,
        // The bundle's own spec 5.3 `policy_hash`, not spec 19.3's: the latter is
        // H(training_hash, checkpoint) and cannot live inside a `training_hash` input.
        // `training.lock` carries that one.
        "bundle_policy_hash": bundle.manifest.hashes.policy.as_ref().map(hex),
        "weights_blake3": hex(blake3::hash(&bytes).as_bytes()),
    }))
}

/// Which checkpoint marks the trainer has finished writing (packet M13/Z1). One rule on every
/// route: a mark is complete when its files are on disk **and** the trainer has said it is
/// past the mark -- because each trainer says where it is *before* it writes the mark:
///
/// * `train_act.py` / `train_ppo.py` print `{"progress": {"step": N}}` and then write
///   `weights/model-N.safetensors`, so the first progress line past N comes after the write
///   returned;
/// * `lerobot-train` 0.6.1 draws the bar for step N (`progbar.update(1)`) and then saves step
///   N, writing `model.safetensors` and `config.json` before the processor files
///   `import-lerobot` also reads -- a directory holding both can still be half written, and a
///   stats file missing then is read as the older layout. The next bar past N is drawn after
///   the save returned.
///
/// The last mark is the run's length: nothing is said past it, so it is bundled after the
/// trainer exits, as before.
pub(super) struct CheckpointWatch {
    /// `(mark, the files it is written as)`, not yet reported.
    pub(super) pending: Vec<(u32, Vec<PathBuf>)>,
}

impl CheckpointWatch {
    /// Every mark the plan bundles, with what the trainer writes for it.
    pub(super) fn of(plan: &Plan) -> Self {
        let pending = plan
            .steps
            .iter()
            .filter_map(|s| {
                let files = match s.kind {
                    StepKind::PolicyPack => vec![PathBuf::from(arg(s, "--weights")?)],
                    StepKind::PolicyImportLerobot => {
                        let dir = Path::new(arg(s, "--checkpoint")?);
                        vec![dir.join("model.safetensors"), dir.join("config.json")]
                    }
                    _ => return None,
                };
                Some((s.step?, files))
            })
            .collect();
        Self { pending }
    }

    /// The marks complete now that the trainer is past `step`, each reported once.
    pub(super) fn past(&mut self, step: u64) -> Vec<u32> {
        let (ready, pending): (Vec<_>, Vec<_>) =
            self.pending.drain(..).partition(|(mark, files)| {
                step > u64::from(*mark) && files.iter().all(|f| f.is_file())
            });
        self.pending = pending;
        ready.into_iter().map(|(mark, _)| mark).collect()
    }
}
