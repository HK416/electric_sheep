//! `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl` and the path templates.
//!
//! Field-by-field uncertainty is recorded in `docs/api-notes/lerobot-dataset.md`; no
//! `lerobot` version is pinned, so most of this is `unverified` by spec 1.7's standard.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_ir::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use serde::{Deserialize, Serialize};

use crate::DataError;

/// Bookkeeping columns that are not learning features. They live as plain parquet primitives
/// rather than as per-frame lists, and three of the five are pure functions of position, so
/// they never reach [`crate::Episode::columns`].
pub const RESERVED: [&str; 5] = [
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
];

pub fn is_reserved(name: &str) -> bool {
    RESERVED.contains(&name)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Float32,
    Float64,
    Int64,
    Bool,
    Video,
    Image,
    #[serde(rename = "string")]
    Str,
}

impl Dtype {
    /// Does this feature occupy a parquet column? `video`/`image` frames live in mp4 files
    /// and are surfaced as [`crate::VideoRef`]s instead.
    pub fn is_columnar(self) -> bool {
        matches!(
            self,
            Self::Float32 | Self::Float64 | Self::Int64 | Self::Bool
        )
    }

    pub fn is_video(self) -> bool {
        matches!(self, Self::Video | Self::Image)
    }
}

/// One entry of `info.json`'s `features` map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub names: Option<Vec<String>>,
    /// Everything else `LeRobot` writes here (video codec info, …): carried through untouched
    /// so a round-trip does not drop keys this crate does not model.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl FeatureSpec {
    pub fn new(dtype: Dtype, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            dtype,
            shape: shape.into(),
            names: None,
            extra: BTreeMap::new(),
        }
    }

    /// Values per frame. `1` for a scalar feature.
    pub fn elem_count(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// The spec 5.4 type this feature presents at an IR port.
    ///
    /// Three edges are lossy because the format does not record what the type system wants;
    /// the table in `docs/api-notes/lerobot-dataset.md` says which and why. In short:
    /// `int64` narrows to [`ElemType::I32`] (the spec has no 64-bit integer element, while
    /// [`crate::Column::I64`] keeps full width in memory), the frame is [`Frame::Policy`]
    /// (the dataset declares none, and `Policy` is the spec's "no frame checking" frame),
    /// and `image` is `None` because an [`es_ir::ImageSpec`] cannot be reconstructed from
    /// `info.json`.
    pub fn port_type(&self) -> Result<PortType, DataError> {
        let elem = match self.dtype {
            Dtype::Float32 => ElemType::F32,
            Dtype::Float64 => ElemType::F64,
            Dtype::Int64 => ElemType::I32,
            Dtype::Bool => ElemType::Bool,
            Dtype::Video | Dtype::Image => ElemType::U8,
            Dtype::Str => {
                return Err(DataError::Unsupported(
                    "string features have no PortType".into(),
                ))
            }
        };
        Ok(PortType {
            elem,
            shape: Shape::new(self.shape.clone()),
            unit: if self.dtype.is_video() {
                Unit::Pixel
            } else {
                Unit::Dimensionless
            },
            frame: Frame::Policy,
            time: TimeRef::Tick,
            image: None,
        })
    }
}

/// `meta/info.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub codebase_version: String,
    pub fps: f64,
    pub total_episodes: u32,
    pub total_frames: u64,
    pub total_tasks: u32,
    pub total_chunks: u32,
    pub chunks_size: u32,
    pub data_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_path: Option<String>,
    pub features: BTreeMap<String, FeatureSpec>,
    /// `robot_type`, `total_videos`, `splits`, … — preserved, never interpreted. `splits` in
    /// particular is *not* read: splits are owned by [`crate::Split`] (spec 19.2).
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Info {
    /// A minimal `info.json` for [`crate::LeRobotWriter`]; totals are recomputed by
    /// `finish()`, so the values here are placeholders.
    pub fn new(fps: f64, features: BTreeMap<String, FeatureSpec>) -> Self {
        let has_video = features.values().any(|f| f.dtype.is_video());
        Self {
            codebase_version: "v2.1".into(),
            fps,
            total_episodes: 0,
            total_frames: 0,
            total_tasks: 0,
            total_chunks: 0,
            chunks_size: 1000,
            data_path: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet".into(),
            video_path: has_video.then(|| {
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                    .into()
            }),
            features,
            extra: BTreeMap::new(),
        }
    }

    pub fn chunk_of(&self, episode: u32) -> u32 {
        episode / self.chunks_size.max(1)
    }

    /// `data_path` rendered for one episode, relative to the dataset root.
    pub fn data_path(&self, episode: u32) -> Result<String, DataError> {
        render(&self.data_path, self.chunk_of(episode), episode, None)
    }

    /// `video_path` rendered for one episode and camera, relative to the dataset root.
    pub fn video_path(&self, episode: u32, video_key: &str) -> Result<Option<String>, DataError> {
        let Some(t) = &self.video_path else {
            return Ok(None);
        };
        render(t, self.chunk_of(episode), episode, Some(video_key)).map(Some)
    }

    /// Feature names with a parquet column, in sorted order.
    pub fn columnar(&self) -> impl Iterator<Item = (&String, &FeatureSpec)> {
        self.features
            .iter()
            .filter(|(name, f)| f.dtype.is_columnar() && !is_reserved(name))
    }

    /// Camera keys, in sorted order.
    pub fn cameras(&self) -> impl Iterator<Item = &String> {
        self.features
            .iter()
            .filter(|(_, f)| f.dtype.is_video())
            .map(|(name, _)| name)
    }
}

/// One line of `meta/episodes.jsonl`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeMeta {
    pub episode_index: u32,
    pub tasks: Vec<String>,
    pub length: u64,
}

/// One line of `meta/tasks.jsonl`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub task_index: u32,
    pub task: String,
}

/// Renders a Python `str.format` template restricted to the three placeholders `LeRobot` uses,
/// each optionally zero-padded (`{episode_index:06d}`). Anything else is rejected rather than
/// guessed at.
fn render(
    template: &str,
    chunk: u32,
    episode: u32,
    video_key: Option<&str>,
) -> Result<String, DataError> {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after
            .find('}')
            .ok_or_else(|| DataError::Unsupported(format!("unterminated `{{` in {template:?}")))?;
        let field = &after[..close];
        rest = &after[close + 1..];

        let (name, pad) = match field.split_once(':') {
            Some((name, spec)) => {
                let width = spec
                    .strip_prefix('0')
                    .and_then(|s| s.strip_suffix('d'))
                    .and_then(|s| s.parse::<usize>().ok())
                    .ok_or_else(|| {
                        DataError::Unsupported(format!("unsupported format spec {spec:?}"))
                    })?;
                (name, width)
            }
            None => (field, 0),
        };
        match name {
            "episode_chunk" => write!(out, "{chunk:0pad$}").expect("String never fails"),
            "episode_index" => write!(out, "{episode:0pad$}").expect("String never fails"),
            "video_key" => out.push_str(video_key.ok_or_else(|| {
                DataError::Unsupported("{video_key} outside a video path".into())
            })?),
            other => {
                return Err(DataError::Unsupported(format!(
                    "unknown path placeholder {{{other}}}"
                )))
            }
        }
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_render_like_python_format() {
        let info = Info::new(30.0, BTreeMap::new());
        assert_eq!(
            info.data_path(7).unwrap(),
            "data/chunk-000/episode_000007.parquet"
        );
        let mut info = info;
        info.chunks_size = 4;
        info.video_path = Some("videos/chunk-{episode_chunk:03d}/{video_key}/e.mp4".into());
        assert_eq!(
            info.video_path(9, "cam").unwrap().unwrap(),
            "videos/chunk-002/cam/e.mp4"
        );
    }

    #[test]
    fn unknown_placeholder_is_rejected_not_guessed() {
        let err = render("data/{robot_type}/x", 0, 0, None).unwrap_err();
        assert!(matches!(err, DataError::Unsupported(_)), "{err}");
        let err = render("data/{episode_index:x}", 0, 0, None).unwrap_err();
        assert!(matches!(err, DataError::Unsupported(_)), "{err}");
        let err = render("data/{episode_index", 0, 0, None).unwrap_err();
        assert!(matches!(err, DataError::Unsupported(_)), "{err}");
    }

    #[test]
    fn port_type_mapping_is_the_documented_one() {
        let state = FeatureSpec::new(Dtype::Float32, [6]).port_type().unwrap();
        assert_eq!(state.elem, ElemType::F32);
        assert_eq!(state.unit, Unit::Dimensionless);
        assert_eq!(state.shape, Shape::new([6]));
        assert!(state.image.is_none());

        let cam = FeatureSpec::new(Dtype::Video, [480, 640, 3])
            .port_type()
            .unwrap();
        assert_eq!(cam.elem, ElemType::U8);
        assert_eq!(cam.unit, Unit::Pixel);
        assert!(cam.image.is_none());

        assert_eq!(FeatureSpec::new(Dtype::Int64, [1]).elem_count(), 1);
        assert!(FeatureSpec::new(Dtype::Str, [1]).port_type().is_err());
    }
}
