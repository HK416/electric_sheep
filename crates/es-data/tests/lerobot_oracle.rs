//! The real `lerobot` package reads what this crate wrote (packet `docs/packets/M5/V1`).
//!
//! Spec 1.4: the dataset's oracle is the package it claims compatibility with, not our own
//! reader agreeing with itself (`crates/es-data/tests/lerobot.rs` is that, and it is a
//! different test). `crates/es-data/python/lerobot_read_ref.py` opens the dataset with
//! `lerobot.datasets.lerobot_dataset.LeRobotDataset` and prints what it found; this compares
//! that with `es-data`'s own reader.
//!
//!     ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python \
//!       cargo test -p es-data --test lerobot_oracle -- --nocapture
//!
//! No interpreter, or one without `lerobot[dataset]`, prints `SKIP lerobot_oracle: <why>` and
//! returns (spec 1.4: never fake a run). Spec 25.1: the dataset root is written here and read
//! back **out of process**; nothing the script says feeds a trust decision.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use es_data::{Column, Dtype, Episode, FeatureSpec, Info, LeRobotDataset, LeRobotWriter};

const LENGTHS: [usize; 2] = [4, 3];
const NJ: usize = 3;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-data-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// A dataset with the columns `es loop collect` writes, minus the image feature: a `video`
/// feature with no mp4 behind it is a dangling reference the real reader would rightly refuse
/// (design note section 3.1), and that gap is V4's, not this oracle's.
fn write_fixture(root: &Path) -> Vec<Episode> {
    let mut features = BTreeMap::new();
    features.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [NJ as u64]),
    );
    features.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [NJ as u64]),
    );
    features.insert("reward".to_owned(), FeatureSpec::new(Dtype::Float64, [1]));
    features.insert(
        es_data::ACTION_SOURCE.to_owned(),
        FeatureSpec::new(Dtype::Int64, [1]),
    );
    features.insert(
        es_data::INTERVENTION.to_owned(),
        FeatureSpec::new(Dtype::Int64, [1]),
    );

    let mut writer = LeRobotWriter::create(root, Info::new(50.0, features)).expect("writer");
    let episodes: Vec<Episode> = LENGTHS
        .iter()
        .enumerate()
        .map(|(index, n)| {
            let n = *n;
            let base = index as f32;
            let columns = BTreeMap::from([
                (
                    "observation.state".to_owned(),
                    Column::F32((0..n * NJ).map(|i| base + i as f32 * 0.25).collect()),
                ),
                (
                    "action".to_owned(),
                    Column::F32((0..n * NJ).map(|i| base - i as f32 * 0.5).collect()),
                ),
                (
                    "reward".to_owned(),
                    Column::F64((0..n).map(|i| f64::from(i as u32) * 0.1).collect()),
                ),
                (es_data::ACTION_SOURCE.to_owned(), Column::I64(vec![1; n])),
                (es_data::INTERVENTION.to_owned(), Column::I64(vec![1; n])),
            ]);
            Episode {
                index: index as u32,
                tasks: vec!["es:task:oracle".to_owned()],
                timestamps: (0..n).map(|i| i as f64 / 50.0).collect(),
                task_index: vec![0; n],
                columns,
                video: BTreeMap::new(),
            }
        })
        .collect();
    for ep in &episodes {
        writer.write_episode(ep).expect("write episode");
    }
    writer.finish().expect("finish");
    episodes
}

/// The one value of a flat JSON field, without a JSON dependency (this crate has one, but the
/// script's output is machine-written and flat, so a scan is enough and a malformed line
/// fails the test loudly).
fn field<'a>(json: &'a str, name: &str) -> Option<&'a str> {
    let after = json.split_once(&format!("\"{name}\":"))?.1.trim_start();
    let end = match after.as_bytes().first()? {
        b'"' => after[1..].find('"')? + 2,
        b'[' => after.find(']')? + 1,
        b'{' => after.find('}')? + 1,
        _ => after.find([',', '}'])?,
    };
    Some(after[..end].trim().trim_matches('"'))
}

fn numbers(list: &str) -> Vec<f64> {
    list.trim_matches(['[', ']'])
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

#[test]
fn lerobot_reads_what_we_wrote() {
    let Ok(python) = std::env::var("ES_LEROBOT_PYTHON") else {
        println!(
            "SKIP lerobot_oracle: ES_LEROBOT_PYTHON is unset; point it at an interpreter with \
             lerobot[dataset]"
        );
        return;
    };
    let root = scratch("lerobot-oracle");
    let written = write_fixture(&root);
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("python/lerobot_read_ref.py");

    let out = match Command::new(&python).arg(&script).arg(&root).output() {
        Ok(out) => out,
        Err(e) => {
            println!("SKIP lerobot_oracle: {python}: {e}");
            return;
        }
    };
    let text = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    assert!(
        out.status.success(),
        "{}: exited {:?}\nstdout:\n{text}\nstderr:\n{}",
        script.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    if let Some(why) = field(&text, "skip") {
        println!("SKIP lerobot_oracle: {why}");
        return;
    }

    // What our own reader says about the same bytes.
    let ours = LeRobotDataset::open(&root).expect("our reader opens it");
    let episodes: usize = field(&text, "episodes")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no episode count in {text}"));
    let frames: usize = field(&text, "frames")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no frame count in {text}"));
    let lengths: Vec<f64> = numbers(field(&text, "lengths").expect("lengths"));

    assert_eq!(episodes, ours.episodes().len(), "episode count: {text}");
    assert_eq!(frames, LENGTHS.iter().sum::<usize>(), "frame count: {text}");
    assert_eq!(
        lengths.iter().map(|v| *v as usize).collect::<Vec<_>>(),
        LENGTHS.to_vec(),
        "per-episode frame counts: {text}"
    );

    // Every feature we declared, with the shape we declared, and the state rows at both ends.
    let features = field(&text, "features").expect("features");
    for name in ours.features().keys() {
        assert!(features.contains(name), "{name} is missing from {features}");
    }
    let first = numbers(field(&text, "first_state").expect("first_state"));
    let last = numbers(field(&text, "last_state").expect("last_state"));
    let ours_first = match written[0].columns.get("observation.state") {
        Some(Column::F32(v)) => v[..NJ].iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        other => panic!("{other:?}"),
    };
    let tail = &written[LENGTHS.len() - 1];
    let ours_last = match tail.columns.get("observation.state") {
        Some(Column::F32(v)) => v[v.len() - NJ..]
            .iter()
            .map(|v| f64::from(*v))
            .collect::<Vec<_>>(),
        other => panic!("{other:?}"),
    };
    for (a, b) in first.iter().zip(&ours_first) {
        assert!(
            (a - b).abs() < 1e-6,
            "first row: {first:?} vs {ours_first:?}"
        );
    }
    for (a, b) in last.iter().zip(&ours_last) {
        assert!((a - b).abs() < 1e-6, "last row: {last:?} vs {ours_last:?}");
    }
    println!("RAN lerobot_oracle: {text}");
}
