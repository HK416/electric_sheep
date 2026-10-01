//! What a run reads from its dataset: the spec 19.2 facts, the task check, and the camera
//! frames mirrored for the external route's export.

use std::path::Path;

use es_data::training::{camera_dirs, DatasetFacts};
use es_data::{DatasetIdentity, LeRobotDataset, Split};

use super::bad;
use crate::error::CliError;
use crate::util::hex;

/// The dataset's own numbers: the spec 19.2 three-way hash, the split `es loop distill` left
/// beside it when there is one, and the `es:task:<hex>` name collection recorded.
pub(super) fn dataset_facts(dataset: &LeRobotDataset) -> Result<DatasetFacts, CliError> {
    let n = dataset.episodes().len() as u32;
    let split_file = dataset.root().join("split.json");
    let (split, split_source) = match std::fs::read_to_string(&split_file) {
        Ok(text) => (
            serde_json::from_str::<Split>(&text)
                .map_err(|e| bad(format!("{}: {e}", split_file.display())))?,
            "loop distill split.json",
        ),
        // `es train` trains on every episode of the root it was pointed at, so the honest
        // split for a root with no `split.json` is the all-train partition -- the same
        // display-only convention `es dataset info` and `es dataset bake` use.
        Err(_) => (Split::deterministic(n, [1.0, 0.0, 0.0], 0), "all-train"),
    };
    let identity = DatasetIdentity::compute(dataset, &split).map_err(|e| bad(e.to_string()))?;
    Ok(DatasetFacts {
        hashes: identity.to_dataset_hash(),
        episodes: n,
        frames: dataset.episodes().iter().map(|m| m.length).sum(),
        recorded_task: dataset
            .tasks()
            .iter()
            .map(|t| t.task.clone())
            .find(|t| t.starts_with("es:task:")),
        split_source,
    })
}

/// M5 review S-3/R4, surfaced here: demonstrations of one predicate and documents of another
/// measure nothing, and packet M5/V19 measured 0/16 by construction because of it.
pub(super) fn check_task(
    facts: &DatasetFacts,
    task_hash: &[u8; 32],
    retired: Option<&str>,
) -> Result<(), CliError> {
    let Some(recorded) = &facts.recorded_task else {
        return Ok(());
    };
    let want = format!("es:task:{}", hex(task_hash));
    if *recorded == want {
        return Ok(());
    }
    let recorded_hex = recorded.trim_start_matches("es:task:");
    if retired == Some(recorded_hex) {
        println!("note: accepting the retired task {recorded_hex} (--allow-retired-task)");
        return Ok(());
    }
    Err(bad(format!(
        "the dataset was collected under task_hash {recorded_hex} and this recipe's Task IR \
         is {}.\nA policy trained on demonstrations of one predicate and judged against \
         another measures nothing (M5 review S-3).\nPass \
         `--allow-retired-task {recorded_hex}` to accept it deliberately.",
        hex(task_hash)
    )))
}

/// Hard-links each camera's `<NNNNNN>.bin` tiles into the `<camera>/` layout `es dataset
/// export` reads: the flat tiles of a one-camera collection, `<frames>/<camera>/` for each
/// camera of several (packet M15/N3, [`camera_dirs`]) -- never one camera's frames under
/// another's name. Nothing is copied unless the link fails, and an existing file is left
/// alone.
///
/// ponytail: one link per frame, ~60k for a 200-demonstration set; a directory symlink would
/// be one syscall, but it needs privileges on Windows and this runs on both.
pub(super) fn mirror_frames(
    frames: &Path,
    dataset: &LeRobotDataset,
    into: &Path,
) -> Result<(), CliError> {
    let total: u64 = dataset.episodes().iter().map(|m| m.length).sum();
    for (camera, from) in camera_dirs(dataset.info(), frames) {
        let dir = into.join(camera);
        std::fs::create_dir_all(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        for i in 0..total {
            let (src, dst) = (
                from.join(format!("{i:06}.bin")),
                dir.join(format!("{i:06}.bin")),
            );
            if dst.exists() {
                continue;
            }
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst)
                    .map_err(|e| bad(format!("{} -> {}: {e}", src.display(), dst.display())))?;
            }
        }
        println!("frames:        {} <- {}", dir.display(), from.display());
    }
    Ok(())
}
