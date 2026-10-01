//! `adapter.toml`, the per-robot adapter document a human writes (spec 14.4): joint order and
//! units, the action's kind and tail, the observation layout, and adapter v2's I/O conventions
//! (packet M11/X2).

use std::collections::BTreeMap;

use serde::Deserialize;

use super::ImportError;

// --- adapter.toml -----------------------------------------------------------------------------

/// The per-robot adapter document (spec 14.4). `deny_unknown_fields` everywhere: a key nobody
/// reads is a mapping nobody declared, and silently ignoring it is exactly the guess rule 3
/// forbids.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adapter {
    pub robot: RobotBlock,
    pub joints: JointBlock,
    pub action: ActionBlock,
    pub observation: ObservationBlock,
    /// v2 (packet M11/X2): the source's policy period, checked, never converted (`IMP-006`).
    #[serde(default)]
    pub timing: Option<TimingBlock>,
    /// v2: the source's actuator model, compared to the scene's and reported, never converted.
    #[serde(default)]
    pub actuators: Option<ActuatorBlock>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotBlock {
    pub name: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointBlock {
    /// v1: the framework's own action order, by our actuator names. Recorded, not permuted.
    /// Empty when absent; exactly one of this and `source_names` is declared (`IMP-008`).
    #[serde(default)]
    pub source_order: Vec<String>,
    /// v2: the **source's** joint names in its articulation order -- what Isaac Lab's
    /// `resolve_matching_names` returns (`isaac-lab.md` section 2) -- resolved by name against
    /// our actuators, through `rename` where the two disagree. Every per-joint vector in the
    /// adapter and the manifest is in this order; the importer permutes the first Dense's
    /// input columns and the head's rows into ours, which is exact.
    #[serde(default)]
    pub source_names: Option<Vec<String>>,
    /// `[joints.rename]`: source name -> our actuator name.
    #[serde(default)]
    pub rename: BTreeMap<String, String>,
    /// `"rad"` or `"deg"`. `"deg"` is refused (`IMP-003`), never converted.
    pub units: String,
    /// v2: the source's rest pose, radians, source order (Isaac's `default_joint_pos`).
    #[serde(default)]
    pub default_pos: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionBlock {
    /// `"position_target"` or `"torque"`.
    pub kind: String,
    #[serde(default)]
    pub scale: Option<Vec<f64>>,
    #[serde(default)]
    pub offset: Option<Vec<f64>>,
    /// v2: `offset = default_pos`, Isaac's `JointPositionActionCfg.use_default_offset`.
    #[serde(default)]
    pub use_default_offset: bool,
    /// v2: the source clamps its raw action to `[lo, hi]`. Exact, and accepted, only where it
    /// cannot bind -- under a `tanh` squash with `lo <= -1` and `hi >= 1`; the Learning IR has
    /// no clamp node, so anything tighter is `IMP-009`.
    #[serde(default)]
    pub clip: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingBlock {
    /// Seconds per policy step in the source: Isaac's `decimation * sim.dt`, Playground's
    /// `ctrl_dt`.
    pub policy_dt: f64,
}

/// Per joint, source order. Reported next to the scene's own numbers; a disagreement is a
/// warning row, because the physics of *our* scene is what the policy will meet.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActuatorBlock {
    #[serde(default)]
    pub stiffness: Option<Vec<f64>>,
    #[serde(default)]
    pub damping: Option<Vec<f64>>,
    #[serde(default)]
    pub armature: Option<Vec<f64>>,
    #[serde(default)]
    pub effort_limit: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationBlock {
    pub channels: Vec<ChannelMap>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelMap {
    /// Where the framework read this block from, e.g. `"qpos[0:6]"`. Recorded for a human --
    /// the meaning comes from `channel` -- and read for one thing only: a term the IR cannot
    /// compute (`"mdp.projected_gravity"`, ...) is refused by its name (`IMP-007`).
    pub source: String,
    /// `[start, end)` in the source's flat observation vector.
    pub slice: [u32; 2],
    /// The Task IR `ObservationSpec` channel this block feeds.
    pub channel: String,
    /// v2: the source computed `(x - offset) * scale` for this block. Folded into the
    /// `Normalize{MeanStd}` (`mean += offset`, `std /= scale`), which is exact algebra.
    #[serde(default)]
    pub scale: Option<Scale>,
    #[serde(default)]
    pub offset: Option<Offset>,
    /// v2: a per-term clip. The Observation IR has no clamp node, so it is `IMP-009`.
    #[serde(default)]
    pub clip: Option<[f64; 2]>,
    /// v2: the block is the last `history` frames of the channel, flattened frame by frame:
    /// a `TemporalWindowNode` of that many steps (`Align::Hold`, the first frame repeated
    /// before the ring fills -- Isaac's `CircularBuffer` on reset).
    #[serde(default)]
    pub history: Option<u32>,
    /// v2: the frame order inside the block. Absent = `newest_last`, Isaac's flattening.
    #[serde(default)]
    pub history_order: Option<HistoryOrder>,
}

/// A per-term scale: one number for the whole block or one per element (source order).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Scale {
    One(f64),
    Each(Vec<f64>),
}

/// A per-term offset: `"default_pos"` (Isaac's `joint_pos_rel`) or one number per element.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Offset {
    Named(String),
    Each(Vec<f64>),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HistoryOrder {
    NewestLast,
    NewestFirst,
}

impl Adapter {
    pub fn parse(raw: &str) -> Result<Self, ImportError> {
        es_ir::serial::parse_toml(raw).map_err(|e| ImportError::Adapter(e.to_string()))
    }
}
