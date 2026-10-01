//! `import.json`, the manifest `python/es/import_rl.py` writes beside `weights.safetensors`:
//! the source network's shape, its observation statistics and what it says about its action.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::ImportError;

// --- import.json ------------------------------------------------------------------------------

/// `import.json`, as `python/es/import_rl.py` writes it.
///
/// Unknown fields are captured rather than refused, and reported as warnings: no framework
/// version is pinned in this workspace (spec 1.7), so a manifest from a newer exporter must
/// still convert — the same rule [`crate::lerobot_config`] follows for `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct ImportManifest {
    pub framework: String,
    #[serde(default)]
    pub versions: BTreeMap<String, String>,
    pub obs_dim: u32,
    pub action_dim: u32,
    pub hidden: Vec<u32>,
    pub activation: String,
    #[serde(default)]
    pub activate_output: bool,
    pub squash: String,
    #[serde(default)]
    pub obs_mean: Option<Vec<f64>>,
    #[serde(default)]
    pub obs_std: Option<Vec<f64>>,
    /// Free text naming the formula the exporter used to derive `obs_std`; recorded, not read.
    #[serde(default)]
    pub normalizer: Option<String>,
    /// The source's state-independent log-std, `None` when it has none (brax's second output
    /// half is a function of the observation). Training-only: `train_ppo.py --init-log-std`
    /// is fed from it, and it never enters a bundle.
    #[serde(default)]
    pub log_std: Option<Vec<f64>>,
    /// Whatever the source stored in place of a log-std, as metadata.
    #[serde(default)]
    pub source_std: Option<serde_json::Value>,
    /// What the source framework says its action vector *is* (`"position_target"`,
    /// `"joint_delta"`, `"torque"`), when its exporter recorded it. `None` is not a licence to
    /// guess: it means only the adapter declares, and the adapter always has to.
    #[serde(default)]
    pub action_kind: Option<String>,
    #[serde(default)]
    pub action_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub action_offset: Option<Vec<f64>>,
    #[serde(default)]
    pub joint_order: Option<Vec<String>>,
    #[serde(default)]
    pub synthetic: Option<String>,
    /// The source's rest pose, in its own joint order, when `import_rl.py` was handed the
    /// source config (`--isaac-env-cfg` / `--playground-config`, packet M11/X2). Read by an
    /// adapter's `use_default_offset` and `offset = "default_pos"` when the adapter itself
    /// does not state the pose.
    #[serde(default)]
    pub default_joint_pos: Option<Vec<f64>>,
    /// `decimation` physics steps of `sim_dt` seconds per policy step, from the same config:
    /// checked against the Deployment IR's control period (`IMP-006`).
    #[serde(default)]
    pub decimation: Option<u32>,
    #[serde(default)]
    pub sim_dt: Option<f64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl ImportManifest {
    pub fn parse(raw: &str) -> Result<Self, ImportError> {
        serde_json::from_str(raw).map_err(|e| ImportError::Manifest(e.to_string()))
    }
}
