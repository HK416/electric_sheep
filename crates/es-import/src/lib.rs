//! `es-import` (layer 9): external policy and config importers. A `LeRobot` `config.json`
//! (`lerobot_config`, spec 14.4), a `RoboVerse` task (`roboverse`, spec 14.4) and a brax /
//! `rsl_rl` / `rl_games` PPO actor (`rl_import`, spec 28.11) are each converted into IR the rest
//! of the system reads; `lerobot::meta` is the `LeRobot` dataset's `meta/` schema, which
//! `lerobot_config` (the dataset's `fps`) and `es-data`'s reader and writer share.
//!
//! These modules lived in `es-data` until packet `docs/packets/M10/W3a-es-import-split.md`
//! moved them here to bring that crate under the spec 1.5 line target; `es-data` re-exports
//! every one of them under its old path, so `es_data::lerobot_config::…`,
//! `es_data::roboverse::…`, `es_data::rl_import::…` and `es_data::lerobot::meta::…` keep
//! resolving and no caller changed.
//!
//! Layer rule (spec 4.2): layer 9, so this crate may depend on `es-core`, `es-ir`, `es-math`
//! and `es-policy` (layer 8), and never on `es-data` (layer 10).
#![forbid(unsafe_code)]

pub mod lerobot;
pub mod lerobot_config;
pub mod rl_import;
pub mod roboverse;
