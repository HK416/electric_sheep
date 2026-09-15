//! `es dataset export --lerobot-v3` writes something `lerobot` 0.6.1 opens (packet
//! `docs/packets/M5/V1b-lerobot-v3-export.md`).
//!
//! Spec 1.4: the oracle for a format claim is the package that owns the format. V1 measured
//! that 0.6.1 refuses our v2.1 outright; this checks the converter's output against the real
//! reader, frame by frame.
//!
//!     ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python \
//!       cargo test -p es-data --test lerobot_v3 -- --nocapture
//!
//! No interpreter, or one without `lerobot[dataset]`, prints `SKIP lerobot_v3_export: <why>`
//! and returns (spec 1.4: never fake a run). A *refusal* by `lerobot` is a failure, not a skip.
//! Spec 25.1: the export is written here and read back **out of process**.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use es_data::{
    export_v3, Column, DatasetIdentity, Dtype, Episode, FeatureSpec, Info, LeRobotDataset,
    LeRobotWriter, Split,
};

const LENGTHS: [usize; 2] = [4, 3];
const NJ: usize = 3;
const H: u64 = 4;
const W: u64 = 6;
const CAMERA: &str = "observation.images.top";
const TASK: &str = "es:task:oracle";
const FPS: f64 = 50.0;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("es-data-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn total_frames() -> usize {
    LENGTHS.iter().sum()
}

/// One raw rendered frame, the bytes `es_env::render::EnvRenderer` dumps.
fn frame_bytes(frame: usize) -> Vec<u8> {
    (0..(H * W * 3) as usize)
        .map(|i| ((frame * 7 + i) % 256) as u8)
        .collect()
}

/// A V1-style v2.1 dataset: the columns `es loop collect` writes plus one declared camera.
/// `frames`, when given, also receives the raw frame dump the camera's pixels come from.
fn write_fixture(root: &Path, frames: Option<&Path>) -> Vec<Episode> {
    let mut features = BTreeMap::new();
    for name in ["observation.state", "action", es_data::ACTION_COMMANDED] {
        features.insert(
            name.to_owned(),
            FeatureSpec::new(Dtype::Float32, [NJ as u64]),
        );
    }
    features.insert("reward".to_owned(), FeatureSpec::new(Dtype::Float64, [1]));
    for name in [es_data::ACTION_SOURCE, es_data::INTERVENTION] {
        features.insert(name.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
    }
    features.insert(CAMERA.to_owned(), FeatureSpec::new(Dtype::Video, [H, W, 3]));

    let mut writer = LeRobotWriter::create(root, Info::new(FPS, features)).expect("writer");
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
                    // Packet M5/V1c: the raw command, which differs from the executed action
                    // wherever the plane corrected it.
                    es_data::ACTION_COMMANDED.to_owned(),
                    Column::F32((0..n * NJ).map(|i| base - i as f32 * 0.75).collect()),
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
                tasks: vec![TASK.to_owned()],
                timestamps: (0..n).map(|i| i as f64 / FPS).collect(),
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

    if let Some(frames) = frames {
        let cell = frames.join("top");
        std::fs::create_dir_all(&cell).expect("create cell dir");
        for frame in 0..total_frames() {
            std::fs::write(cell.join(format!("{frame:06}.bin")), frame_bytes(frame))
                .expect("write frame");
        }
    }
    episodes
}

/// The `observation.state` and `action` values, flattened over every frame in order —
/// the same order `lerobot_read_ref.py` iterates in.
fn flat(episodes: &[Episode], name: &str) -> Vec<f64> {
    episodes
        .iter()
        .flat_map(|ep| match ep.columns.get(name) {
            Some(Column::F32(v)) => v.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            other => panic!("{name}: {other:?}"),
        })
        .collect()
}

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn nonempty(path: &Path) {
    let len = std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len();
    assert!(len > 0, "{} is empty", path.display());
}

/// The layout, template by template, against `docs/api-notes/lerobot-dataset.md`'s
/// "`LeRobot` v3.0" section.
#[test]
fn export_layout_is_v3() {
    let dir = scratch("v3-layout");
    let (src, frames, out) = (dir.join("src"), dir.join("frames"), dir.join("out"));
    write_fixture(&src, Some(&frames));

    let dataset = LeRobotDataset::open(&src).expect("open the v2.1 source");
    let report = export_v3(&dataset, &out, Some(&frames)).expect("export");
    assert_eq!(report.episodes, LENGTHS.len() as u32);
    assert_eq!(report.frames, total_frames() as u64);
    assert_eq!(report.cameras, vec![CAMERA.to_owned()]);
    assert!(report.dropped.is_empty(), "{:?}", report.dropped);

    let info = json(&out.join("meta/info.json"));
    assert_eq!(info["codebase_version"], "v3.0");
    assert_eq!(info["fps"], 50);
    assert_eq!(info["total_frames"], total_frames() as u64);
    assert_eq!(info["total_tasks"], 1);
    assert_eq!(
        info["data_path"],
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    );
    assert!(info["video_path"].is_null(), "no mp4 is written");

    let features = &info["features"];
    // v3.0 requires the five bookkeeping columns in `features`; v2.1 did not carry them.
    for name in [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ] {
        assert_eq!(features[name]["shape"], serde_json::json!([1]), "{name}");
    }
    assert_eq!(features[CAMERA]["dtype"], "image", "not video: no decoder");
    assert_eq!(features[CAMERA]["shape"], serde_json::json!([H, W, 3]));
    assert_eq!(features["observation.state"]["dtype"], "float32");
    // Both action columns cross the export: what was executed and what was asked for
    // (packet M5/V1c).
    for name in ["action", es_data::ACTION_COMMANDED] {
        assert_eq!(features[name]["dtype"], "float32", "{name}");
        assert_eq!(features[name]["shape"], serde_json::json!([NJ]), "{name}");
    }

    // The source is untouched: V2's training script still reads v2.1 (packet `forbidden`).
    assert_eq!(
        json(&src.join("meta/info.json"))["codebase_version"],
        "v2.1"
    );

    for name in [
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/tasks.parquet",
    ] {
        nonempty(&out.join(name));
    }
}

/// Spec 19.2: the export is derived, and says what it was derived from.
#[test]
fn provenance_records_the_source_identity() {
    let dir = scratch("v3-provenance");
    let (src, out) = (dir.join("src"), dir.join("out"));
    write_fixture(&src, None);

    let dataset = LeRobotDataset::open(&src).expect("open");
    export_v3(&dataset, &out, None).expect("export");

    let split = Split::deterministic(LENGTHS.len() as u32, [1.0, 0.0, 0.0], 0);
    let identity = DatasetIdentity::compute(&dataset, &split).expect("identity");
    let written = json(&out.join("meta/es_provenance.json"));
    assert_eq!(written["content"], hex(&identity.content));
    assert_eq!(written["schema"], hex(&identity.schema));
    assert_eq!(written["split"], hex(&identity.split));
    assert_eq!(written["source_codebase_version"], "v2.1");
}

/// A camera with no pixels behind it is dropped and reported, not written as a feature
/// `lerobot` would fail on.
#[test]
fn images_without_frames_are_dropped_not_dangled() {
    let dir = scratch("v3-no-frames");
    let (src, out) = (dir.join("src"), dir.join("out"));
    write_fixture(&src, None);

    let dataset = LeRobotDataset::open(&src).expect("open");
    let report = export_v3(&dataset, &out, None).expect("export");
    assert!(report.cameras.is_empty(), "{:?}", report.cameras);
    assert_eq!(report.dropped, vec![CAMERA.to_owned()]);
    assert!(
        json(&out.join("meta/info.json"))["features"][CAMERA].is_null(),
        "the dropped camera must not be declared"
    );
}

/// The one value of a flat JSON field, without leaning on `serde_json` for the script's
/// machine-written output (mirrors `tests/lerobot_oracle.rs`).
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

fn close(a: &[f64], b: &[f64], what: &str, text: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length\n{text}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 1e-5, "{what}[{i}]: {x} vs {y}\n{text}");
    }
}

#[test]
fn lerobot_v3_export() {
    let Ok(python) = std::env::var("ES_LEROBOT_PYTHON") else {
        println!(
            "SKIP lerobot_v3_export: ES_LEROBOT_PYTHON is unset; point it at an interpreter \
             with lerobot[dataset]"
        );
        return;
    };
    let dir = scratch("v3-export");
    let (src, frames, out) = (dir.join("src"), dir.join("frames"), dir.join("out"));
    let written = write_fixture(&src, Some(&frames));

    let dataset = LeRobotDataset::open(&src).expect("open the v2.1 source");
    let report = export_v3(&dataset, &out, Some(&frames)).expect("export");
    assert_eq!(report.cameras, vec![CAMERA.to_owned()]);

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("python/lerobot_read_ref.py");
    let output = match Command::new(&python).arg(&script).arg(&out).output() {
        Ok(output) => output,
        Err(e) => {
            println!("SKIP lerobot_v3_export: {python}: {e}");
            return;
        }
    };
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    assert!(
        output.status.success(),
        "{}: exited {:?}\nstdout:\n{text}\nstderr:\n{}",
        script.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(why) = field(&text, "skip") {
        // An import failure is a machine without the package; anything else is `lerobot`
        // refusing what we wrote, and that is the failure this packet exists to prevent.
        assert!(
            why.starts_with("import lerobot.datasets"),
            "lerobot refused the export: {why}"
        );
        println!("SKIP lerobot_v3_export: {why}");
        return;
    }

    let episodes: usize = field(&text, "episodes")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no episode count in {text}"));
    let frames_seen: usize = field(&text, "frames")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no frame count in {text}"));
    let iterated: usize = field(&text, "iterated")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no iterated count in {text}"));
    assert_eq!(episodes, LENGTHS.len(), "episode count\n{text}");
    assert_eq!(frames_seen, total_frames(), "frame count\n{text}");
    assert_eq!(iterated, total_frames(), "frames actually read\n{text}");
    assert_eq!(
        numbers(field(&text, "lengths").expect("lengths"))
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>(),
        LENGTHS.to_vec(),
        "per-episode lengths\n{text}"
    );

    // Every frame's values, not the two ends.
    close(
        &numbers(field(&text, "states_flat").expect("states_flat")),
        &flat(&written, "observation.state"),
        "observation.state",
        &text,
    );
    close(
        &numbers(field(&text, "actions_flat").expect("actions_flat")),
        &flat(&written, "action"),
        "action",
        &text,
    );

    // The camera: CHW as `lerobot` returns it, and the pixels we handed the encoder.
    assert_eq!(
        numbers(field(&text, "image_shape").expect("image_shape"))
            .iter()
            .map(|v| *v as u64)
            .collect::<Vec<_>>(),
        vec![3, H, W],
        "image shape\n{text}"
    );
    let ours: f64 = (0..total_frames())
        .flat_map(frame_bytes)
        .map(|b| f64::from(b) / 255.0)
        .sum();
    let theirs = numbers(field(&text, "image_sum").expect("image_sum"));
    assert_eq!(theirs.len(), 1, "one camera\n{text}");
    assert!(
        (theirs[0] - ours).abs() < 1e-2,
        "image pixel sum: {} vs {ours}\n{text}",
        theirs[0]
    );
    assert_eq!(
        field(&text, "tasks"),
        Some(format!("[\"{TASK}\"]").as_str())
    );

    println!("RAN lerobot_v3_export: {text}");
}
