//! Parquet episode files, through the low-level `parquet` column API.
//!
//! No `arrow` and no compression codec are enabled (see `crates/es-data/Cargo.toml`), which
//! keeps the dependency to one crate and the check time to a couple of seconds. The physical
//! layout — 3-level lists for features, plain primitives for the five reserved columns — is
//! written up in `docs/api-notes/lerobot-dataset.md`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use parquet::basic::{LogicalType, Repetition, Type as PhysicalType};
use parquet::column::reader::get_typed_column_reader;
use parquet::data_type::{BoolType, DataType, DoubleType, FloatType, Int64Type};
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::{Type, TypePtr};

use super::meta::{Dtype, Info};
use super::{Episode, VideoRef};
use crate::DataError;

/// One feature's values for a whole episode: flat, row-major, `length * elem_count` long.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            Self::F32(_) => Dtype::Float32,
            Self::F64(_) => Dtype::Float64,
            Self::I64(_) => Dtype::Int64,
            Self::Bool(_) => Dtype::Bool,
        }
    }
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

fn physical(dtype: Dtype) -> Result<PhysicalType, DataError> {
    Ok(match dtype {
        Dtype::Float32 => PhysicalType::FLOAT,
        Dtype::Float64 => PhysicalType::DOUBLE,
        Dtype::Int64 => PhysicalType::INT64,
        Dtype::Bool => PhysicalType::BOOLEAN,
        other => {
            return Err(DataError::Unsupported(format!(
                "{other:?} has no parquet column"
            )))
        }
    })
}

/// `optional group <name> (LIST) { repeated group list { optional <t> item; } }`.
fn list_field(name: &str, phys: PhysicalType) -> Result<TypePtr, ParquetError> {
    let item = Type::primitive_type_builder("item", phys)
        .with_repetition(Repetition::OPTIONAL)
        .build()?;
    let list = Type::group_type_builder("list")
        .with_repetition(Repetition::REPEATED)
        .with_fields(vec![Arc::new(item)])
        .build()?;
    Ok(Arc::new(
        Type::group_type_builder(name)
            .with_repetition(Repetition::OPTIONAL)
            .with_logical_type(Some(LogicalType::List))
            .with_fields(vec![Arc::new(list)])
            .build()?,
    ))
}

fn scalar_field(name: &str, phys: PhysicalType) -> Result<TypePtr, ParquetError> {
    Ok(Arc::new(
        Type::primitive_type_builder(name, phys)
            .with_repetition(Repetition::OPTIONAL)
            .build()?,
    ))
}

/// Leaves in declaration order: every columnar feature as a list, then the five reserved
/// bookkeeping primitives.
fn schema(info: &Info, path: &Path) -> Result<TypePtr, DataError> {
    let mut fields = Vec::new();
    for (name, feat) in info.columnar() {
        let phys = physical(feat.dtype)?;
        fields.push(list_field(name, phys).map_err(|e| pq(path, e))?);
    }
    fields.push(scalar_field("timestamp", PhysicalType::DOUBLE).map_err(|e| pq(path, e))?);
    for name in ["frame_index", "episode_index", "index", "task_index"] {
        fields.push(scalar_field(name, PhysicalType::INT64).map_err(|e| pq(path, e))?);
    }
    Type::group_type_builder("lerobot")
        .with_fields(fields)
        .build()
        .map(Arc::new)
        .map_err(|e| pq(path, e))
}

/// Repetition levels for `rows` lists of `d` items each: a new list starts at 0.
fn rep_levels(rows: usize, d: usize) -> Vec<i16> {
    (0..rows * d).map(|i| i16::from(i % d != 0)).collect()
}

pub(crate) fn write_episode(
    path: &Path,
    info: &Info,
    ep: &Episode,
    global_offset: u64,
) -> Result<(), DataError> {
    let n = ep.len();
    let schema = schema(info, path)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    }
    let file = File::create(path).map_err(|e| DataError::io(path, e))?;
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|e| pq(path, e))?;
    let mut rg = writer.next_row_group().map_err(|e| pq(path, e))?;

    for (name, feat) in info.columnar() {
        let d = feat.elem_count() as usize;
        let col = ep
            .columns
            .get(name)
            .ok_or_else(|| bad(format!("episode {} has no column {name:?}", ep.index)))?;
        if col.dtype() != feat.dtype {
            return Err(bad(format!(
                "{name:?}: column is {:?}, info.json says {:?}",
                col.dtype(),
                feat.dtype
            )));
        }
        if col.len() != n * d {
            return Err(bad(format!(
                "{name:?}: {} values for {n} frames of {d}",
                col.len()
            )));
        }
        let def = vec![3i16; n * d];
        let rep = rep_levels(n, d);
        let mut w = rg
            .next_column()
            .map_err(|e| pq(path, e))?
            .ok_or_else(|| bad("schema ran out of columns"))?;
        match col {
            Column::F32(v) => w
                .typed::<FloatType>()
                .write_batch(v, Some(&def), Some(&rep)),
            Column::F64(v) => w
                .typed::<DoubleType>()
                .write_batch(v, Some(&def), Some(&rep)),
            Column::I64(v) => w
                .typed::<Int64Type>()
                .write_batch(v, Some(&def), Some(&rep)),
            Column::Bool(v) => w.typed::<BoolType>().write_batch(v, Some(&def), Some(&rep)),
        }
        .map_err(|e| pq(path, e))?;
        w.close().map_err(|e| pq(path, e))?;
    }

    let def = vec![1i16; n];
    {
        let mut w = rg
            .next_column()
            .map_err(|e| pq(path, e))?
            .ok_or_else(|| bad("schema ran out of columns"))?;
        w.typed::<DoubleType>()
            .write_batch(&ep.timestamps, Some(&def), None)
            .map_err(|e| pq(path, e))?;
        w.close().map_err(|e| pq(path, e))?;
    }
    // `frame_index`, `episode_index` and `index` are pure functions of position, so they are
    // derived here and discarded on read (api-note, "Column shapes").
    let at = |i: u64| i64::try_from(i).unwrap_or(i64::MAX);
    let frame_index: Vec<i64> = (0..n as u64).map(at).collect();
    let episode_index = vec![i64::from(ep.index); n];
    let index: Vec<i64> = (0..n as u64).map(|i| at(global_offset + i)).collect();
    for values in [&frame_index, &episode_index, &index, &ep.task_index] {
        let mut w = rg
            .next_column()
            .map_err(|e| pq(path, e))?
            .ok_or_else(|| bad("schema ran out of columns"))?;
        w.typed::<Int64Type>()
            .write_batch(values, Some(&def), None)
            .map_err(|e| pq(path, e))?;
        w.close().map_err(|e| pq(path, e))?;
    }

    rg.close().map_err(|e| pq(path, e))?;
    writer.close().map_err(|e| pq(path, e))?;
    Ok(())
}

/// The leaf column index whose root field is `name`, plus its max definition level.
fn locate(reader: &SerializedFileReader<File>, name: &str) -> Option<(usize, i16)> {
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .find(|(_, d)| d.path().parts().first().is_some_and(|p| p == name))
        .map(|(i, d)| (i, d.max_def_level()))
}

/// A leaf column as parquet hands it over: values, definition levels, repetition levels.
type Raw<T> = (Vec<T>, Vec<i16>, Vec<i16>);

/// Reads one leaf column across every row group.
fn read_all<T: DataType>(
    reader: &SerializedFileReader<File>,
    leaf: usize,
    path: &Path,
) -> Result<Raw<T::T>, DataError>
where
    T::T: Default + Clone,
{
    let (mut values, mut def, mut rep) = (Vec::new(), Vec::new(), Vec::new());
    for g in 0..reader.num_row_groups() {
        let rg = reader.get_row_group(g).map_err(|e| pq(path, e))?;
        let rows = rg.metadata().num_rows().max(0) as usize;
        let cr = rg.get_column_reader(leaf).map_err(|e| pq(path, e))?;
        let mut typed = get_typed_column_reader::<T>(cr);
        let mut read = 0;
        while read < rows {
            let (records, _, _) = typed
                .read_records(rows - read, Some(&mut def), Some(&mut rep), &mut values)
                .map_err(|e| pq(path, e))?;
            if records == 0 {
                break;
            }
            read += records;
        }
    }
    Ok((values, def, rep))
}

/// `values.len() == rows * d`, every level at its maximum, one list boundary every `d`.
fn check_levels(
    name: &str,
    got: usize,
    rows: usize,
    d: usize,
    def: &[i16],
    max_def: i16,
    rep: &[i16],
) -> Result<(), DataError> {
    if got != rows * d {
        return Err(bad(format!(
            "{name:?}: {got} values for {rows} frames of {d}"
        )));
    }
    if def.iter().any(|l| *l != max_def) {
        return Err(bad(format!("{name:?}: null values are not supported")));
    }
    if rep
        .iter()
        .enumerate()
        .any(|(i, r)| (*r == 0) != (i % d == 0))
    {
        return Err(bad(format!("{name:?}: ragged lists are not supported")));
    }
    Ok(())
}

/// Everything a parquet episode file contributes: the feature columns, the timestamps and the
/// per-frame task label (spec 13.2).
pub(crate) type EpisodeData = (BTreeMap<String, Column>, Vec<f64>, Vec<i64>);

pub(crate) fn read_episode(
    path: &Path,
    info: &Info,
    index: u32,
    length: u64,
) -> Result<EpisodeData, DataError> {
    let file = File::open(path).map_err(|e| DataError::io(path, e))?;
    let reader = SerializedFileReader::new(file).map_err(|e| pq(path, e))?;
    let total: u64 = reader
        .metadata()
        .row_groups()
        .iter()
        .map(|g| g.num_rows().max(0) as u64)
        .sum();
    if total != length {
        return Err(bad(format!(
            "episode {index}: {total} parquet rows, episodes.jsonl says {length}"
        )));
    }
    let n = length as usize;

    let mut columns = BTreeMap::new();
    for (name, feat) in info.columnar() {
        let d = feat.elem_count() as usize;
        let (leaf, max_def) = locate(&reader, name).ok_or_else(|| {
            bad(format!(
                "{name:?} is in info.json but not in {}",
                path.display()
            ))
        })?;
        macro_rules! read {
            ($t:ty, $variant:ident) => {{
                let (v, def, rep) = read_all::<$t>(&reader, leaf, path)?;
                check_levels(name, v.len(), n, d, &def, max_def, &rep)?;
                Column::$variant(v)
            }};
        }
        let col = match feat.dtype {
            Dtype::Float32 => read!(FloatType, F32),
            Dtype::Float64 => read!(DoubleType, F64),
            Dtype::Int64 => read!(Int64Type, I64),
            Dtype::Bool => read!(BoolType, Bool),
            other => return Err(DataError::Unsupported(format!("{name:?}: {other:?}"))),
        };
        columns.insert(name.clone(), col);
    }

    let timestamps = read_timestamps(&reader, n, path)?;
    let task_index = match locate(&reader, "task_index") {
        Some((leaf, max_def)) => {
            let (v, def, rep) = read_all::<Int64Type>(&reader, leaf, path)?;
            check_levels("task_index", v.len(), n, 1, &def, max_def, &rep)?;
            v
        }
        // A single-task dataset need not carry the column.
        None => vec![0; n],
    };
    Ok((columns, timestamps, task_index))
}

/// `timestamp` is `float32` in some `LeRobot` versions and `float64` in others (`unverified`),
/// so both are accepted and widened.
fn read_timestamps(
    reader: &SerializedFileReader<File>,
    n: usize,
    path: &Path,
) -> Result<Vec<f64>, DataError> {
    let (leaf, max_def) =
        locate(reader, "timestamp").ok_or_else(|| bad("no `timestamp` column"))?;
    let phys = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .column(leaf)
        .physical_type();
    match phys {
        PhysicalType::DOUBLE => {
            let (v, def, rep) = read_all::<DoubleType>(reader, leaf, path)?;
            check_levels("timestamp", v.len(), n, 1, &def, max_def, &rep)?;
            Ok(v)
        }
        PhysicalType::FLOAT => {
            let (v, def, rep) = read_all::<FloatType>(reader, leaf, path)?;
            check_levels("timestamp", v.len(), n, 1, &def, max_def, &rep)?;
            Ok(v.into_iter().map(f64::from).collect())
        }
        other => Err(DataError::Unsupported(format!(
            "timestamp is {other}, expected FLOAT or DOUBLE"
        ))),
    }
}

/// Video refs for one episode: the `video_path` template rendered per camera, one entry per
/// frame. No mp4 is opened — decoding is a later packet.
pub(crate) fn video_refs(
    info: &Info,
    index: u32,
    n: usize,
) -> Result<BTreeMap<String, Vec<VideoRef>>, DataError> {
    let mut out = BTreeMap::new();
    for camera in info.cameras() {
        let Some(path) = info.video_path(index, camera)? else {
            continue;
        };
        out.insert(
            camera.clone(),
            (0..n as u32)
                .map(|frame_index| VideoRef {
                    path: path.clone().into(),
                    frame_index,
                })
                .collect(),
        );
    }
    Ok(out)
}
