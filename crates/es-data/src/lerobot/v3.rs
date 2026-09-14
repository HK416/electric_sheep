//! `LeRobot` **v3.0** export (spec 19.1) — the format `lerobot` 0.6.1 actually reads.
//!
//! `lerobot` 0.6.1 refuses `codebase_version: "v2.1"` outright (design note
//! `docs/design/visible-learning.md` section 7.5), so this module converts what
//! [`super::LeRobotWriter`] produces into the layout that package opens. The v2.1 reader and
//! writer next door are deliberately untouched: V2's training script reads them directly.
//!
//! Every field, template and physical encoding below was read out of the installed package and
//! then executed against it; `docs/api-notes/lerobot-dataset.md`, "`LeRobot` v3.0", is the
//! write-up and names the exact files. The three findings that decide the shape of this module:
//!
//! - a `shape: [1]` feature is a **scalar** parquet column, not a length-1 list, and the five
//!   bookkeeping columns must be declared in `info.json`'s `features`;
//! - image pixels live **inside** the data parquet as `struct<bytes, path>` holding an encoded
//!   image file, which is the one visual storage 0.6.1 decodes with no `ffmpeg` and no
//!   `torchcodec` (neither of which loads on the oracle server);
//! - `meta/tasks.parquet` is read with `pandas`, and the task string must be its *index*.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parquet::basic::{LogicalType, Repetition, Type as PhysicalType};
use parquet::data_type::{BoolType, ByteArray, ByteArrayType, DoubleType, FloatType, Int64Type};
use parquet::errors::ParquetError;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{Type, TypePtr};

use super::meta::{Dtype, EpisodeMeta, FeatureSpec, Info};
use super::{Column, Episode, LeRobotDataset};
use crate::identity::{DatasetIdentity, Split};
use crate::{read_file, write_file, DataError};

/// `lerobot.datasets.dataset_metadata.CODEBASE_VERSION`.
const CODEBASE_VERSION: &str = "v3.0";
const DATA_PATH: &str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet";
const DATA_FILE: &str = "data/chunk-000/file-000.parquet";
const EPISODES_FILE: &str = "meta/episodes/chunk-000/file-000.parquet";
const TASKS_FILE: &str = "meta/tasks.parquet";

/// The five names that are bookkeeping rather than learning features. Unlike v2.1, v3.0
/// **requires** them in `info.json`'s `features` (api-note, "features -> the parquet schema").
const BOOKKEEPING: [&str; 5] = [
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
];

/// What the converter produced, for the CLI to print.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportReport {
    pub episodes: u32,
    pub frames: u64,
    /// Camera keys written as `dtype: "image"`.
    pub cameras: Vec<String>,
    /// Camera keys dropped — declared by the source but with no frames behind them.
    pub dropped: Vec<String>,
}

fn pq(path: &Path, source: ParquetError) -> DataError {
    DataError::Parquet {
        path: path.to_path_buf(),
        source,
    }
}

fn bad(what: impl Into<String>) -> DataError {
    DataError::Inconsistent(what.into())
}

fn hex(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The next leaf column of the open row group.
macro_rules! next_col {
    ($rg:expr, $path:expr) => {
        $rg.next_column()
            .map_err(|e| pq($path, e))?
            .ok_or_else(|| bad("schema ran out of columns"))?
    };
}

/// One output column: where its values come from, not what type they are.
#[derive(Clone, Debug)]
enum Col {
    /// One of [`BOOKKEEPING`] — derived from position, never read from `Episode::columns`.
    Reserved(String),
    /// A learning feature, `elems` values per frame.
    Feature { name: String, elems: usize },
    /// One PNG per frame, encoded from `dir/<NNNNNN>.bin`.
    Image {
        name: String,
        dir: PathBuf,
        height: u32,
        width: u32,
    },
}

/// Converts the v2.1 dataset `src` into `lerobot` 0.6.1's v3.0 layout under `out`.
///
/// `frames` is the root of the raw frame dump `es_env::render::EnvRenderer` writes: one
/// subdirectory per camera, named after the suffix of `observation.images.<name>`, holding
/// `<NNNNNN>.bin` (row-major `u8`, the feature's own `[h, w, 3]`) in dataset-global frame
/// order. A camera with no directory behind it is **dropped and reported**, never written as a
/// feature nothing can load.
pub fn export_v3(
    src: &LeRobotDataset,
    out: &Path,
    frames: Option<&Path>,
) -> Result<ExportReport, DataError> {
    let info = src.info();
    let metas = src.episodes().to_vec();
    if metas.is_empty() {
        return Err(bad("source dataset has no episodes"));
    }
    if src.tasks().is_empty() {
        return Err(DataError::Unsupported(
            "source dataset has no meta/tasks.jsonl; lerobot v3.0 resolves a frame's task by \
             position in meta/tasks.parquet and cannot be written without one"
                .into(),
        ));
    }
    let fps = info.fps;
    if fps <= 0.0 || (fps - fps.round()).abs() > 1e-9 {
        return Err(DataError::Unsupported(format!(
            "fps {fps} is not a positive integer; lerobot v3.0's DatasetInfo declares `fps: int`"
        )));
    }

    // Column values are gathered whole before the parquet writer sees them, so an export holds
    // one dataset in memory.
    // ponytail: whole-dataset buffering, ~30 MB per 1k 96x96 frames; stream per row group if a
    // demonstration set ever outgrows RAM.
    let episodes: Vec<Episode> = metas
        .iter()
        .map(|m| src.read_episode(m.episode_index))
        .collect::<Result<_, _>>()?;
    let total_frames: u64 = metas.iter().map(|m| m.length).sum();

    let mut report = ExportReport {
        episodes: u32::try_from(metas.len()).unwrap_or(u32::MAX),
        frames: total_frames,
        ..ExportReport::default()
    };
    let mut features: BTreeMap<String, FeatureSpec> = BTreeMap::new();
    let mut cameras: Vec<Col> = Vec::new();

    for (name, spec) in info.columnar() {
        features.insert(name.clone(), spec.clone());
    }
    for camera in info.cameras() {
        let Some(([height, width], dir)) = camera_source(info, camera, frames) else {
            report.dropped.push(camera.clone());
            continue;
        };
        let mut spec = FeatureSpec::new(Dtype::Image, [u64::from(height), u64::from(width), 3]);
        spec.names = Some(vec![
            "height".to_owned(),
            "width".to_owned(),
            "channels".to_owned(),
        ]);
        features.insert(camera.clone(), spec);
        report.cameras.push(camera.clone());
        cameras.push(Col::Image {
            name: camera.clone(),
            dir,
            height,
            width,
        });
    }
    features.insert(
        "timestamp".to_owned(),
        FeatureSpec::new(Dtype::Float64, [1]),
    );
    for name in ["frame_index", "episode_index", "index", "task_index"] {
        features.insert(name.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
    }

    // One pass over the (sorted) features fixes both the parquet schema and the order the
    // columns are written in, so the two cannot drift apart.
    let mut fields = Vec::with_capacity(features.len());
    let mut plan = Vec::with_capacity(features.len());
    for (name, spec) in &features {
        let col = if BOOKKEEPING.contains(&name.as_str()) {
            Col::Reserved(name.clone())
        } else if spec.dtype == Dtype::Image {
            cameras
                .iter()
                .find(|c| matches!(c, Col::Image { name: n, .. } if n == name))
                .cloned()
                .ok_or_else(|| bad(format!("{name:?}: image feature with no frame directory")))?
        } else {
            Col::Feature {
                name: name.clone(),
                elems: spec.elem_count() as usize,
            }
        };
        fields.push(field_of(name, spec, &col)?);
        plan.push(col);
    }

    write_data(&out.join(DATA_FILE), fields, &plan, &episodes, &metas)?;
    write_episodes(&out.join(EPISODES_FILE), &metas)?;
    write_tasks(&out.join(TASKS_FILE), src)?;

    let value = serde_json::json!({
        "codebase_version": CODEBASE_VERSION,
        "robot_type": info.extra.get("robot_type").cloned().unwrap_or(serde_json::Value::Null),
        "fps": fps.round() as i64,
        "total_episodes": report.episodes,
        "total_frames": total_frames,
        "total_tasks": src.tasks().len(),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": DATA_PATH,
        "video_path": serde_json::Value::Null,
        "features": features,
        "splits": serde_json::Map::new(),
    });
    write_json(&out.join("meta/info.json"), &value)?;
    write_provenance(out, src)?;
    Ok(report)
}

/// A camera's `[height, width]` and the directory its raw frames are in, or `None` when the
/// feature is not a `[h, w, 3]` image or nothing on disk backs it.
fn camera_source(info: &Info, camera: &str, frames: Option<&Path>) -> Option<([u32; 2], PathBuf)> {
    let shape = info.features.get(camera)?.shape.as_slice();
    let [height, width, 3] = shape else {
        return None;
    };
    let (height, width) = (u32::try_from(*height).ok()?, u32::try_from(*width).ok()?);
    if height == 0 || width == 0 {
        return None;
    }
    let suffix = camera.rsplit_once('.').map_or(camera, |(_, tail)| tail);
    let dir = frames?.join(suffix);
    dir.is_dir().then_some(([height, width], dir))
}

fn physical(dtype: Dtype) -> Result<PhysicalType, DataError> {
    Ok(match dtype {
        Dtype::Float32 => PhysicalType::FLOAT,
        Dtype::Float64 => PhysicalType::DOUBLE,
        Dtype::Int64 => PhysicalType::INT64,
        Dtype::Bool => PhysicalType::BOOLEAN,
        other => {
            return Err(DataError::Unsupported(format!(
                "{other:?} has no v3.0 parquet column"
            )))
        }
    })
}

fn scalar(name: &str, phys: PhysicalType) -> Result<TypePtr, ParquetError> {
    Ok(Arc::new(
        Type::primitive_type_builder(name, phys)
            .with_repetition(Repetition::OPTIONAL)
            .build()?,
    ))
}

fn string(name: &str) -> Result<TypePtr, ParquetError> {
    Ok(Arc::new(
        Type::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(Some(LogicalType::String))
            .build()?,
    ))
}

/// `optional group <name> (LIST) { repeated group list { optional <t> item; } }` — accepted
/// where a `fixed_size_list` is declared, because `datasets` casts it (api-note).
fn list(name: &str, item: TypePtr) -> Result<TypePtr, ParquetError> {
    let inner = Type::group_type_builder("list")
        .with_repetition(Repetition::REPEATED)
        .with_fields(vec![item])
        .build()?;
    Ok(Arc::new(
        Type::group_type_builder(name)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(Some(LogicalType::List))
            .with_fields(vec![Arc::new(inner)])
            .build()?,
    ))
}

/// `optional group <name> { optional binary bytes; optional binary (String) path; }` — what
/// `datasets.Image()` reads.
fn image_group(name: &str) -> Result<TypePtr, ParquetError> {
    let bytes = Type::primitive_type_builder("bytes", PhysicalType::BYTE_ARRAY)
        .with_repetition(Repetition::OPTIONAL)
        .build()?;
    Ok(Arc::new(
        Type::group_type_builder(name)
            .with_repetition(Repetition::OPTIONAL)
            .with_fields(vec![Arc::new(bytes), string("path")?])
            .build()?,
    ))
}

fn field_of(name: &str, spec: &FeatureSpec, col: &Col) -> Result<TypePtr, DataError> {
    let here = Path::new(name);
    match col {
        Col::Image { .. } => image_group(name).map_err(|e| pq(here, e)),
        Col::Reserved(_) | Col::Feature { elems: 1, .. } => {
            scalar(name, physical(spec.dtype)?).map_err(|e| pq(here, e))
        }
        Col::Feature { .. } => {
            let item = scalar("item", physical(spec.dtype)?).map_err(|e| pq(here, e))?;
            list(name, item).map_err(|e| pq(here, e))
        }
    }
}

/// Repetition levels for `rows` lists of `d` items each: a new list starts at 0.
fn rep_levels(rows: usize, d: usize) -> Vec<i16> {
    (0..rows * d).map(|i| i16::from(i % d != 0)).collect()
}

/// One feature's values across every episode, in episode order.
fn concat(episodes: &[Episode], name: &str) -> Result<Column, DataError> {
    let mut out: Option<Column> = None;
    for ep in episodes {
        let col = ep
            .columns
            .get(name)
            .ok_or_else(|| bad(format!("episode {} has no column {name:?}", ep.index)))?;
        match (&mut out, col) {
            (None, col) => out = Some(col.clone()),
            (Some(Column::F32(a)), Column::F32(b)) => a.extend_from_slice(b),
            (Some(Column::F64(a)), Column::F64(b)) => a.extend_from_slice(b),
            (Some(Column::I64(a)), Column::I64(b)) => a.extend_from_slice(b),
            (Some(Column::Bool(a)), Column::Bool(b)) => a.extend_from_slice(b),
            (Some(a), b) => {
                return Err(bad(format!(
                    "{name:?}: episode {} is {:?}, earlier episodes are {:?}",
                    ep.index,
                    b.dtype(),
                    a.dtype()
                )))
            }
        }
    }
    out.ok_or_else(|| bad("no episodes"))
}

/// `frame_index`, `episode_index`, `index` and `task_index`, derived from position.
fn reserved_i64(name: &str, episodes: &[Episode]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut global = 0i64;
    for ep in episodes {
        let n = i64::try_from(ep.len()).unwrap_or(i64::MAX);
        match name {
            "frame_index" => out.extend(0..n),
            "episode_index" => out.resize(out.len() + ep.len(), i64::from(ep.index)),
            "index" => out.extend(global..global + n),
            _ => out.extend_from_slice(&ep.task_index),
        }
        global += n;
    }
    out
}

/// `data/chunk-000/file-000.parquet`: every episode, one row group.
fn write_data(
    path: &Path,
    fields: Vec<TypePtr>,
    plan: &[Col],
    episodes: &[Episode],
    metas: &[EpisodeMeta],
) -> Result<(), DataError> {
    let n: usize = metas.iter().map(|m| m.length as usize).sum();
    let schema = Type::group_type_builder("lerobot")
        .with_fields(fields)
        .build()
        .map(Arc::new)
        .map_err(|e| pq(path, e))?;
    let file = create(path)?;
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|e| pq(path, e))?;
    let mut rg = writer.next_row_group().map_err(|e| pq(path, e))?;

    for col in plan {
        match col {
            Col::Reserved(name) => {
                let def = vec![1i16; n];
                let mut w = next_col!(rg, path);
                if name == "timestamp" {
                    let v: Vec<f64> = episodes.iter().flat_map(|e| e.timestamps.clone()).collect();
                    w.typed::<DoubleType>().write_batch(&v, Some(&def), None)
                } else {
                    w.typed::<Int64Type>().write_batch(
                        &reserved_i64(name, episodes),
                        Some(&def),
                        None,
                    )
                }
                .map_err(|e| pq(path, e))?;
                w.close().map_err(|e| pq(path, e))?;
            }
            Col::Feature { name, elems } => {
                let values = concat(episodes, name)?;
                if values.len() != n * elems {
                    return Err(bad(format!(
                        "{name:?}: {} values for {n} frames of {elems}",
                        values.len()
                    )));
                }
                let (def, rep) = if *elems == 1 {
                    (vec![1i16; n], None)
                } else {
                    (vec![3i16; n * elems], Some(rep_levels(n, *elems)))
                };
                let rep = rep.as_deref();
                let mut w = next_col!(rg, path);
                match &values {
                    Column::F32(v) => w.typed::<FloatType>().write_batch(v, Some(&def), rep),
                    Column::F64(v) => w.typed::<DoubleType>().write_batch(v, Some(&def), rep),
                    Column::I64(v) => w.typed::<Int64Type>().write_batch(v, Some(&def), rep),
                    Column::Bool(v) => w.typed::<BoolType>().write_batch(v, Some(&def), rep),
                }
                .map_err(|e| pq(path, e))?;
                w.close().map_err(|e| pq(path, e))?;
            }
            Col::Image {
                name,
                dir,
                height,
                width,
            } => {
                let frame_bytes = *height as usize * *width as usize * 3;
                let mut pngs = Vec::with_capacity(n);
                for frame in 0..n {
                    let src = dir.join(format!("{frame:06}.bin"));
                    let raw = read_file(&src)?;
                    if raw.len() != frame_bytes {
                        return Err(bad(format!(
                            "{}: {} byte(s), {name:?} declares {frame_bytes}",
                            src.display(),
                            raw.len()
                        )));
                    }
                    pngs.push(ByteArray::from(png_rgb(*width, *height, &raw).as_slice()));
                }
                let mut w = next_col!(rg, path);
                w.typed::<ByteArrayType>()
                    .write_batch(&pngs, Some(&vec![2i16; n]), None)
                    .map_err(|e| pq(path, e))?;
                w.close().map_err(|e| pq(path, e))?;
                // `path` is null on every row: `datasets.Image()` decodes `bytes` and never
                // looks at it (api-note, "Image features without ffmpeg or torchcodec").
                let mut w = next_col!(rg, path);
                w.typed::<ByteArrayType>()
                    .write_batch(&[], Some(&vec![1i16; n]), None)
                    .map_err(|e| pq(path, e))?;
                w.close().map_err(|e| pq(path, e))?;
            }
        }
    }
    rg.close().map_err(|e| pq(path, e))?;
    writer.close().map_err(|e| pq(path, e))?;
    Ok(())
}

/// `meta/episodes/chunk-000/file-000.parquet`. Column names containing `/` are flat top-level
/// fields, not nested groups (api-note).
fn write_episodes(path: &Path, metas: &[EpisodeMeta]) -> Result<(), DataError> {
    let names = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ];
    let mut fields: Vec<TypePtr> = names
        .iter()
        .map(|n| scalar(n, PhysicalType::INT64))
        .collect::<Result<_, _>>()
        .map_err(|e| pq(path, e))?;
    fields.push(list("tasks", string("item").map_err(|e| pq(path, e))?).map_err(|e| pq(path, e))?);

    let n = metas.len();
    let mut from = 0i64;
    let mut bounds = Vec::with_capacity(n);
    let length = |m: &EpisodeMeta| i64::try_from(m.length).unwrap_or(i64::MAX);
    for m in metas {
        let to = from + length(m);
        bounds.push((from, to));
        from = to;
    }
    let columns: [Vec<i64>; 8] = [
        metas.iter().map(|m| i64::from(m.episode_index)).collect(),
        metas.iter().map(length).collect(),
        bounds.iter().map(|(a, _)| *a).collect(),
        bounds.iter().map(|(_, b)| *b).collect(),
        vec![0; n],
        vec![0; n],
        vec![0; n],
        vec![0; n],
    ];

    let (mut def, mut rep, mut tasks) = (Vec::new(), Vec::new(), Vec::new());
    for m in metas {
        if m.tasks.is_empty() {
            def.push(1);
            rep.push(0);
            continue;
        }
        for (i, task) in m.tasks.iter().enumerate() {
            def.push(3);
            rep.push(i16::from(i != 0));
            tasks.push(ByteArray::from(task.as_str()));
        }
    }

    let schema = Type::group_type_builder("episodes")
        .with_fields(fields)
        .build()
        .map(Arc::new)
        .map_err(|e| pq(path, e))?;
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer =
        SerializedFileWriter::new(create(path)?, schema, props).map_err(|e| pq(path, e))?;
    let mut rg = writer.next_row_group().map_err(|e| pq(path, e))?;
    for column in &columns {
        let mut w = next_col!(rg, path);
        w.typed::<Int64Type>()
            .write_batch(column, Some(&vec![1i16; n]), None)
            .map_err(|e| pq(path, e))?;
        w.close().map_err(|e| pq(path, e))?;
    }
    let mut w = next_col!(rg, path);
    w.typed::<ByteArrayType>()
        .write_batch(&tasks, Some(&def), Some(&rep))
        .map_err(|e| pq(path, e))?;
    w.close().map_err(|e| pq(path, e))?;
    rg.close().map_err(|e| pq(path, e))?;
    writer.close().map_err(|e| pq(path, e))?;
    Ok(())
}

/// `meta/tasks.parquet`. `lerobot` reads it with `pandas` and resolves a frame's task by
/// position with `tasks.iloc[i].name`, so the task string has to be the row's **index** —
/// which is the `pandas` footer metadata below, not a column property (api-note).
fn write_tasks(path: &Path, src: &LeRobotDataset) -> Result<(), DataError> {
    let mut tasks = src.tasks().to_vec();
    tasks.sort_by_key(|t| t.task_index);
    if tasks
        .iter()
        .enumerate()
        .any(|(i, t)| t.task_index as usize != i)
    {
        return Err(bad(format!(
            "task indices are {:?}, but lerobot resolves a task by position and needs 0..n",
            tasks.iter().map(|t| t.task_index).collect::<Vec<_>>()
        )));
    }
    let pandas = serde_json::json!({
        "index_columns": ["task"],
        "column_indexes": [{
            "name": serde_json::Value::Null,
            "field_name": serde_json::Value::Null,
            "pandas_type": "unicode",
            "numpy_type": "object",
            "metadata": { "encoding": "UTF-8" },
        }],
        "columns": [
            { "name": "task_index", "field_name": "task_index", "pandas_type": "int64",
              "numpy_type": "int64", "metadata": serde_json::Value::Null },
            { "name": "task", "field_name": "task", "pandas_type": "unicode",
              "numpy_type": "object", "metadata": serde_json::Value::Null },
        ],
        "creator": { "library": "es-data", "version": env!("CARGO_PKG_VERSION") },
        "pandas_version": "2.3.3",
    });
    let fields = vec![
        scalar("task_index", PhysicalType::INT64).map_err(|e| pq(path, e))?,
        string("task").map_err(|e| pq(path, e))?,
    ];
    let schema = Type::group_type_builder("schema")
        .with_fields(fields)
        .build()
        .map(Arc::new)
        .map_err(|e| pq(path, e))?;
    let props = Arc::new(
        WriterProperties::builder()
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "pandas".to_owned(),
                pandas.to_string(),
            )]))
            .build(),
    );
    let mut writer =
        SerializedFileWriter::new(create(path)?, schema, props).map_err(|e| pq(path, e))?;
    let mut rg = writer.next_row_group().map_err(|e| pq(path, e))?;
    let n = tasks.len();
    let def = vec![1i16; n];

    let mut w = next_col!(rg, path);
    let indices: Vec<i64> = tasks.iter().map(|t| i64::from(t.task_index)).collect();
    w.typed::<Int64Type>()
        .write_batch(&indices, Some(&def), None)
        .map_err(|e| pq(path, e))?;
    w.close().map_err(|e| pq(path, e))?;

    let mut w = next_col!(rg, path);
    let strings: Vec<ByteArray> = tasks
        .iter()
        .map(|t| ByteArray::from(t.task.as_str()))
        .collect();
    w.typed::<ByteArrayType>()
        .write_batch(&strings, Some(&def), None)
        .map_err(|e| pq(path, e))?;
    w.close().map_err(|e| pq(path, e))?;
    rg.close().map_err(|e| pq(path, e))?;
    writer.close().map_err(|e| pq(path, e))?;
    Ok(())
}

/// Spec 19.2: the export is a derived artifact, so it names the dataset it came from. A
/// sidecar and not `info.json` keys, because `DatasetInfo.from_dict` drops what it does not
/// know and `to_dict` would not write it back.
fn write_provenance(out: &Path, src: &LeRobotDataset) -> Result<(), DataError> {
    let split = Split::deterministic(src.episodes().len() as u32, [1.0, 0.0, 0.0], 0);
    let identity = DatasetIdentity::compute(src, &split)?;
    let value = serde_json::json!({
        "source_root": src.root().display().to_string(),
        "source_codebase_version": src.info().codebase_version,
        "content": hex(&identity.content),
        "schema": hex(&identity.schema),
        "split": hex(&identity.split),
        "exported_by": "es dataset export --lerobot-v3",
    });
    write_json(&out.join("meta/es_provenance.json"), &value)
}

fn create(path: &Path) -> Result<File, DataError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    }
    File::create(path).map_err(|e| DataError::io(path, e))
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), DataError> {
    let json = serde_json::to_vec_pretty(value).map_err(|source| DataError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    write_file(path, &json)
}

// --- PNG ---------------------------------------------------------------------------------
//
// 8-bit truecolor, no filtering, a zlib stream of *stored* (uncompressed) deflate blocks.
// ponytail: stored deflate costs ~0.1% over the raw frame; swap in a real deflate only if an
// export's size becomes the complaint. There is no image or deflate crate in the workspace and
// the packet forbids adding one.

fn crc_table() -> &'static [u32; 256] {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (n, entry) in table.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *entry = c;
        }
        table
    })
}

fn crc32(bytes: &[u8]) -> u32 {
    let table = crc_table();
    let crc = bytes.iter().fold(0xFFFF_FFFFu32, |c, b| {
        table[((c ^ u32::from(*b)) & 0xFF) as usize] ^ (c >> 8)
    });
    crc ^ 0xFFFF_FFFF
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in bytes {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// `width * height * 3` bytes of row-major RGB as a PNG file.
fn png_rgb(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let stride = (width as usize * 3).max(1);
    let mut raw = Vec::with_capacity(height as usize * (1 + stride));
    for row in rgb.chunks(stride) {
        raw.push(0); // filter type 0 (None)
        raw.extend_from_slice(row);
    }

    let mut zlib = vec![0x78, 0x01];
    let mut rest = raw.as_slice();
    loop {
        let take = rest.len().min(0xFFFF);
        let last = rest.len() <= 0xFFFF;
        zlib.push(u8::from(last));
        zlib.extend_from_slice(&(take as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(take as u16)).to_le_bytes());
        zlib.extend_from_slice(&rest[..take]);
        rest = &rest[take..];
        if last {
            break;
        }
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bpc, truecolor, deflate, no filter, no interlace
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    chunk(&mut out, *b"IHDR", &ihdr);
    chunk(&mut out, *b"IDAT", &zlib);
    chunk(&mut out, *b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads a PNG this module wrote back: every chunk CRC, the IHDR fields, the stored
    /// deflate blocks and the Adler-32. The encoder's own oracle when no interpreter is around
    /// (packet `docs/packets/M5/V1b-lerobot-v3-export.md`, oracle 2).
    fn decode(png: &[u8]) -> (u32, u32, Vec<u8>) {
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let (mut at, mut ihdr, mut idat) = (8usize, None, Vec::new());
        while at < png.len() {
            let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let kind = &png[at + 4..at + 8];
            let data = &png[at + 8..at + 8 + len];
            let crc = u32::from_be_bytes(png[at + 8 + len..at + 12 + len].try_into().unwrap());
            assert_eq!(crc, crc32(&png[at + 4..at + 8 + len]), "CRC of {kind:?}");
            match kind {
                b"IHDR" => ihdr = Some(data.to_vec()),
                b"IDAT" => idat.extend_from_slice(data),
                _ => {}
            }
            at += 12 + len;
        }
        let ihdr = ihdr.expect("IHDR");
        let width = u32::from_be_bytes(ihdr[..4].try_into().unwrap());
        let height = u32::from_be_bytes(ihdr[4..8].try_into().unwrap());
        assert_eq!(
            &ihdr[8..],
            &[8, 2, 0, 0, 0],
            "8-bit truecolor, no interlace"
        );

        assert_eq!(&idat[..2], &[0x78, 0x01], "zlib header");
        let (mut at, mut raw) = (2usize, Vec::new());
        loop {
            let last = idat[at] & 1 == 1;
            assert_eq!(idat[at] >> 1 & 3, 0, "stored block");
            let n = u16::from_le_bytes(idat[at + 1..at + 3].try_into().unwrap()) as usize;
            let nlen = u16::from_le_bytes(idat[at + 3..at + 5].try_into().unwrap());
            assert_eq!(nlen, !(n as u16), "LEN/NLEN");
            raw.extend_from_slice(&idat[at + 5..at + 5 + n]);
            at += 5 + n;
            if last {
                break;
            }
        }
        let adler = u32::from_be_bytes(idat[at..at + 4].try_into().unwrap());
        assert_eq!(adler, adler32(&raw), "Adler-32");

        let stride = width as usize * 3;
        let mut pixels = Vec::with_capacity(height as usize * stride);
        for row in raw.chunks(1 + stride) {
            assert_eq!(row[0], 0, "filter type");
            pixels.extend_from_slice(&row[1..]);
        }
        (width, height, pixels)
    }

    #[test]
    fn png_round_trips() {
        for (w, h) in [(1u32, 1u32), (4, 3), (97, 5)] {
            let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
            let (dw, dh, back) = decode(&png_rgb(w, h, &pixels));
            assert_eq!((dw, dh), (w, h));
            assert_eq!(back, pixels, "{w}x{h}");
        }
    }

    /// One frame bigger than a single stored deflate block (65535 bytes), so the multi-block
    /// path is exercised rather than assumed.
    #[test]
    fn png_spans_several_stored_blocks() {
        let (w, h) = (160u32, 160u32);
        let pixels: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
        let png = png_rgb(w, h, &pixels);
        assert!(pixels.len() > 0xFFFF, "the fixture must exceed one block");
        let (dw, dh, back) = decode(&png);
        assert_eq!((dw, dh, back), (w, h, pixels));
    }
}
