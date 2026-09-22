//! The `LeRobot` dataset's `meta/` schema (`info.json`, `episodes.jsonl`, `tasks.jsonl`, the
//! path templates): the one piece of `es_data::lerobot` that `lerobot_config` needs (the
//! dataset's `fps`) and that `es-data`'s reader and writer share. `es-data` re-exports this
//! module as `es_data::lerobot::meta`, so both paths name the same types.

pub mod meta;

pub use meta::{Dtype, EpisodeMeta, FeatureSpec, Info, MetaError, Task};
