//! `es policy import-rl` — an externally trained PPO actor becomes our two documents and a
//! bundle (spec 14.4 "RL policy import", spec 13.4 "the import never guesses", packet M8/S2b).
//!
//! The Python half (`python/es/import_rl.py`) opens the pickle or the orbax checkpoint and
//! leaves behind `weights.safetensors` with framework-neutral keys plus an `import.json`
//! manifest (INV-16: no pickle reaches Rust). This half reads those two, reads a per-robot
//! **adapter document** written by a human, and builds:
//!
//! * an Observation IR — one `StateInput` per adapter channel, in the source's own slice
//!   order, `Concat`ed and `Normalize`d with the source's running statistics (or a `Range`
//!   identity when the source carried none);
//! * a Learning IR — `StateEncoder{Mlp} -> PolicyHead{Regression, horizon 1, squash} ->
//!   ActionChunker -> Normalizer{Inverse, MeanStd}`, the shape
//!   `docs/design/rl-continuation.md` rule 1 fixes for every RL policy;
//! * the weights, remapped from the neutral keys onto the keys `lower_to_torch` declares;
//! * a **Semantic Mapping Report** (spec 14.4): one row per joint and per observation channel.
//!
//! **Where the network is cut matters.** Every framework here activates each of its hidden
//! Dense layers and leaves its output Dense linear, so `import.json` says
//! `activate_output = false`. Our `StateEncoder{Mlp}` stops at the last *hidden* layer and our
//! `PolicyHead{Regression}` is the output Dense — so the encoder we build carries
//! `activate_output = true` and the head carries the source's `squash`. It is the same
//! function, cut in a different place; [`learning_graph`] is where that happens.
//!
//! **Nothing is guessed** (rule 3). Joint order, units, action kind and the observation layout
//! come from the adapter or they do not come, and a mismatch is one of five named refusals,
//! `IMP-001` .. `IMP-005`.
//!
//! **Adapter v2** (packet M11/X2, spec 28.14 rule 3) declares an Isaac Lab or Playground
//! policy's I/O conventions -- joint names in the source's order, the rest pose, per-term
//! offset/scale, history, the previous action, the action offset and clip, the policy period and
//! the actuator model -- and compiles each into the bundle where the algebra is exact
//! (`fold_channels`, and `remap`'s column gather and row permutation), or refuses it by name
//! (`IMP-006` .. `IMP-009`). Every v2 field is optional; absent, the conversion is v1's, byte
//! for byte. `docs/design/rl-continuation.md` section 8a is the table.

use std::collections::BTreeMap;

use es_core::StableId;
use es_ir::codes::{
    IMP_001, IMP_002, IMP_003, IMP_004, IMP_005, IMP_006, IMP_007, IMP_008, IMP_009,
};
use es_ir::deployment::{ActionSpace as DeployedSpace, DeploymentIr, ExecutionMode};
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, HeadKind, LearningGraph,
    LearningNode, NormalizeDir, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, StatsSource, WeightsRef,
};
use es_ir::observation::{
    in_port, Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput, TemporalWindow,
    OUT,
};
use es_ir::task::{ActionSpace, ObsSource, TaskIr, TaskNode};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_policy::lower::lower_to_torch;
use es_policy::weights::{parse_header, validate_keys, write_safetensors, Checkpoint};
use serde::{Deserialize, Serialize};

/// The name the Observation IR exposes its one concatenated vector under, and therefore the
/// name the policy contract binds (`XIR-010`). An RL policy reads one flat state vector.
pub const STATE_PORT: &str = "state";

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

// --- refusals ---------------------------------------------------------------------------------

/// Every way an import can be refused. The five `IMP-0xx` are the adapter mismatches spec 14.4
/// names; the rest are malformed inputs, which have no code because they never got as far as
/// describing a mapping.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("{IMP_001}: the adapter lists {adapter} joints and the manifest {manifest}, but the Task IR's ActionSpec declares dim {task}")]
    JointCount {
        adapter: usize,
        manifest: u32,
        task: u32,
    },
    #[error("{IMP_002}: the adapter names joint \"{joint}\", which is not an actuator of {scene}; the scene has {actuators:?}")]
    UnknownJoint {
        joint: String,
        scene: String,
        actuators: Vec<String>,
    },
    #[error("{IMP_003}: [joints] units = \"{units}\"; this import reads radians only. Convert the checkpoint's own angles before exporting -- a silent degree-to-radian scaling inside the importer is a mapping nobody declared (spec 13.4)")]
    Units { units: String },
    #[error(
        "{IMP_004}: [action] kind = \"{kind}\" but the Task IR's ActionSpec declares {space:?}"
    )]
    ActionKind { kind: String, space: ActionSpace },
    #[error("{IMP_004}: the checkpoint's manifest says action.kind = \"{manifest}\" and the adapter declares \"{adapter}\". The adapter says what the source's numbers mean for *our* robot; it cannot re-interpret what they are. Fix whichever is wrong -- an increment imported as a position target drives the arm to 0.05 rad and stays there")]
    ManifestActionKind { adapter: String, manifest: String },
    #[error("{IMP_005}: {detail}")]
    Channels { detail: String },
    #[error("{IMP_006}: {detail}")]
    Timing { detail: String },
    #[error("{IMP_007}: channel \"{channel}\" is the source term `{term}`, which the IR cannot compute on this robot (spec 28.14 rule 3: refused by name, never zero-filled). The locomotion terms -- projected_gravity, base_lin_vel, base_ang_vel, velocity_commands -- stay with the parked quadruped track, and a generated command is admitted only onto a channel the Task IR's ObservationSpec declares")]
    Uncomputable { channel: String, term: String },
    #[error("{IMP_008}: {detail}")]
    JointOrder { detail: String },
    #[error("{IMP_009}: {detail}")]
    Clip { detail: String },
    #[error("import.json: {0}")]
    Manifest(String),
    #[error("adapter: {0}")]
    Adapter(String),
    #[error("weights.safetensors: {0}")]
    Weights(String),
    #[error("{0}")]
    Ir(String),
}

// --- the mapping report (spec 14.4) ------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warning,
}

#[derive(Clone, Debug, Serialize)]
pub struct JointRow {
    pub source_index: u32,
    pub name: String,
    pub unit: String,
    pub severity: Severity,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChannelRow {
    pub source: String,
    pub source_range: [u32; 2],
    pub name: String,
    pub dim: u32,
    pub unit: String,
    pub severity: Severity,
    pub note: String,
}

/// The Semantic Mapping Report spec 14.4 requires: every joint and every observation channel,
/// with where it came from, what it became, in what unit, and how sure the importer is.
#[derive(Clone, Debug, Serialize)]
pub struct MappingReport {
    pub framework: String,
    pub versions: BTreeMap<String, String>,
    pub robot: String,
    pub obs_dim: u32,
    pub action_dim: u32,
    pub hidden: Vec<u32>,
    pub activation: String,
    pub squash: String,
    /// `null` unless the source carried a state-independent one; `train_ppo.py
    /// --init-log-std` is fed from this and from nothing else.
    pub log_std: Option<Vec<f64>>,
    /// What the source stored in place of a log-std, by kind only -- brax's is a `[hidden,
    /// action_dim]` kernel and the rows belong in `import.json`, not in a report a human reads.
    pub source_std: Option<String>,
    /// The action tail the import actually resolved, `ctrl = offset + scale * a` — the
    /// manifest's, unless the adapter overrode it. Written as numbers rather than only inside
    /// the joint rows' prose because the `--reference` oracle reads them back: the reference
    /// and our runtime have to be the same function, and the rule that picks between the two
    /// sources lives here and is not restated in Python.
    pub action_scale: Vec<f64>,
    pub action_offset: Vec<f64>,
    pub joints: Vec<JointRow>,
    pub channels: Vec<ChannelRow>,
    pub warnings: Vec<String>,
    /// v2: the policy period the import checked, when either the adapter or the manifest
    /// stated one. Absent from a v1 report, whose bytes this packet does not move.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<String>,
    /// v2: `[actuators]` next to the scene's own numbers, filled by [`actuator_rows`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub actuators: Vec<ActuatorRow>,
}

/// One `[actuators]` quantity of one joint: the source's number and our scene's.
#[derive(Clone, Debug, Serialize)]
pub struct ActuatorRow {
    pub name: String,
    pub quantity: String,
    pub source: f64,
    pub scene: Option<f64>,
    pub severity: Severity,
    pub note: String,
}

/// What our scene says about one actuator, read by the caller out of the MJCF (this crate
/// opens no files): a position servo's `kp` / `kv`, the joint's armature, the force range.
#[derive(Clone, Debug, Default)]
pub struct SceneActuator {
    pub name: String,
    pub stiffness: Option<f64>,
    pub damping: Option<f64>,
    pub armature: Option<f64>,
    pub effort_limit: Option<f64>,
}

/// `[actuators]` against the scene, one row per declared joint and quantity. Never
/// converted: a source trained on a stiffer servo is a fact a human deploying it must see,
/// and rescaling the policy to hide it would be the guess rule 3 forbids. The rows are in
/// the source's joint order, named by our actuators.
pub fn actuator_rows(adapter: &Adapter, scene: &[SceneActuator]) -> Vec<ActuatorRow> {
    let Some(block) = &adapter.actuators else {
        return Vec::new();
    };
    let names = adapter
        .joints
        .source_names
        .clone()
        .unwrap_or_else(|| adapter.joints.source_order.clone());
    let mut rows = Vec::new();
    for (quantity, values, read) in [
        (
            "stiffness",
            &block.stiffness,
            (|s: &SceneActuator| s.stiffness) as fn(&SceneActuator) -> Option<f64>,
        ),
        ("damping", &block.damping, |s| s.damping),
        ("armature", &block.armature, |s| s.armature),
        ("effort_limit", &block.effort_limit, |s| s.effort_limit),
    ] {
        for (source_name, value) in names.iter().zip(values.iter().flatten()) {
            let ours = adapter
                .joints
                .rename
                .get(source_name)
                .unwrap_or(source_name);
            let found = scene.iter().find(|s| &s.name == ours).and_then(read);
            let same = found.is_some_and(|v| (v - value).abs() <= 1e-9 * v.abs().max(1.0));
            rows.push(ActuatorRow {
                name: ours.clone(),
                quantity: quantity.to_owned(),
                source: *value,
                scene: found,
                severity: if same {
                    Severity::Ok
                } else {
                    Severity::Warning
                },
                note: match found {
                    Some(v) if same => format!("{quantity} {value} in the source and {v} here"),
                    Some(v) => format!(
                        "{quantity} {value} in the source, {v} in this scene: the policy meets \
                         a different actuator (reported, never converted)"
                    ),
                    None => format!(
                        "{quantity} {value} in the source; this scene states none for \
                         \"{ours}\""
                    ),
                },
            });
        }
    }
    rows
}

/// What [`convert`] produced.
#[derive(Debug)]
pub struct RlImport {
    pub observation: ObservationIr,
    pub learning: LearningGraph,
    /// The weights under the keys `lower_to_torch` declares, ready for `PolicyBundle::build`.
    pub weights: Vec<u8>,
    pub report: MappingReport,
}

// --- the conversion ----------------------------------------------------------------------------

/// `import.json` + `weights.safetensors` + `adapter.toml` + the Task and Deployment IR the
/// import targets → the two documents, the remapped weights and the mapping report.
///
/// `actuators` is the scene's actuator names in declaration order, read by the caller out of
/// the Task IR's own `scene.path` (`IMP-002` needs them and this crate does not open files).
pub fn convert(
    manifest: &ImportManifest,
    adapter: &Adapter,
    task: &TaskIr,
    deployment: &DeploymentIr,
    weights: &[u8],
    actuators: &[String],
) -> Result<RlImport, ImportError> {
    let mut warnings: Vec<String> = manifest
        .extra
        .keys()
        .map(|k| format!("import.json: unrecognized field \"{k}\", ignored"))
        .collect();

    let action_spec = task
        .graph
        .nodes
        .values()
        .find_map(|n| match n {
            TaskNode::ActionSpec { space, dim, .. } => Some((*space, *dim)),
            _ => None,
        })
        .ok_or_else(|| ImportError::Ir("the Task IR declares no ActionSpec node".to_owned()))?;
    let (space, task_dim) = action_spec;

    // IMP-008 / IMP-001 / IMP-002 / IMP-003 / IMP-004: the joint block. `names` is our
    // actuator name of each source joint, in the source's order; `perm[i]` is where source
    // joint `i` sits in ours (v2 only -- a v1 `source_order` records and does not permute).
    let joints = &adapter.joints;
    let (names, perm) = resolve_joints(joints, actuators, &task.scene.path, task_dim)?;
    if names.len() as u32 != task_dim || manifest.action_dim != task_dim {
        return Err(ImportError::JointCount {
            adapter: names.len(),
            manifest: manifest.action_dim,
            task: task_dim,
        });
    }
    for name in &names {
        if !actuators.contains(name) {
            return Err(ImportError::UnknownJoint {
                joint: name.clone(),
                scene: task.scene.path.clone(),
                actuators: actuators.to_vec(),
            });
        }
    }
    if joints.units != "rad" {
        return Err(ImportError::Units {
            units: joints.units.clone(),
        });
    }
    let expected = match space {
        ActionSpace::JointPosition => "position_target",
        ActionSpace::JointDelta => "joint_delta",
        ActionSpace::JointTorque => "torque",
        _ => "",
    };
    if adapter.action.kind != expected {
        return Err(ImportError::ActionKind {
            kind: adapter.action.kind.clone(),
            space,
        });
    }
    // The same `IMP-004` from the other side: the source recorded what its numbers are, and
    // the adapter said it again. Two declarations that disagree are not a tie to break.
    if let Some(kind) = &manifest.action_kind {
        if *kind != adapter.action.kind {
            return Err(ImportError::ManifestActionKind {
                adapter: adapter.action.kind.clone(),
                manifest: kind.clone(),
            });
        }
    }
    if let Some(order) = &manifest.joint_order {
        let declared = joints.source_names.as_ref().unwrap_or(&joints.source_order);
        if order != declared {
            warnings.push(format!(
                "the checkpoint recorded joint order {order:?}, the adapter declares {declared:?}; \
                 the adapter wins (spec 13.4) -- check that this permutation is intended"
            ));
        }
    }
    let timing = check_timing(adapter, manifest, deployment)?;

    // IMP-007 / IMP-009 / IMP-005: the channels must tile `obs_dim` exactly, in order, and
    // each must be a channel the Task IR declares at the same width (times its history).
    let mut cursor = 0u32;
    let mut channels = Vec::new();
    for map in &adapter.observation.channels {
        let [start, end] = map.slice;
        let term = term_name(&map.source);
        let declared_here = task.observation_spec.channels.contains_key(&map.channel);
        if UNCOMPUTABLE.contains(&term) || (term == "generated_commands" && !declared_here) {
            return Err(ImportError::Uncomputable {
                channel: map.channel.clone(),
                term: term.to_owned(),
            });
        }
        if let Some([lo, hi]) = map.clip {
            return Err(ImportError::Clip {
                detail: format!(
                    "channel \"{}\" declares clip = [{lo}, {hi}]. The Observation IR has no \
                     clamp node (spec 7.3), and this packet adds none; a clip that never binds \
                     on the states this policy meets can be left out of the adapter, and one \
                     that binds is a function the IR cannot express",
                    map.channel
                ),
            });
        }
        if start != cursor || end <= start {
            return Err(ImportError::Channels {
                detail: format!(
                    "channel \"{}\" is [{start}, {end}), which does not continue the tiling at \
                     {cursor}. The channels must cover [0, {}) in order, with no gap and no \
                     overlap.",
                    map.channel, manifest.obs_dim
                ),
            });
        }
        let declared = task
            .observation_spec
            .channels
            .get(&map.channel)
            .ok_or_else(|| ImportError::Channels {
                detail: format!(
                    "channel \"{}\" is not declared by the Task IR's ObservationSpec, which \
                     declares {:?}",
                    map.channel,
                    task.observation_spec.channels.keys().collect::<Vec<_>>()
                ),
            })?;
        let width: u64 = declared.ty.shape.dims().iter().product();
        let history = map.history.unwrap_or(1);
        if history == 0 || width * u64::from(history) != u64::from(end - start) {
            return Err(ImportError::Channels {
                detail: format!(
                    "channel \"{}\" takes {} source values but the Task IR declares it {width} \
                     wide{}",
                    map.channel,
                    end - start,
                    if history == 1 {
                        String::new()
                    } else {
                        format!(" over a history of {history} frames")
                    }
                ),
            });
        }
        channels.push((map, declared));
        cursor = end;
    }
    if cursor != manifest.obs_dim {
        return Err(ImportError::Channels {
            detail: format!(
                "the channels cover [0, {cursor}) but the manifest declares obs_dim {}",
                manifest.obs_dim
            ),
        });
    }

    // The manifest's architecture, translated once.
    let activation = match manifest.activation.as_str() {
        "relu" => Activation::Relu,
        "elu" => Activation::Elu,
        "swish" => Activation::Swish,
        "tanh" => Activation::Tanh,
        other => {
            return Err(ImportError::Manifest(format!(
                "activation \"{other}\" is not one of relu, elu, swish, tanh"
            )))
        }
    };
    let squash = match manifest.squash.as_str() {
        "none" => Squash::None,
        "tanh" => Squash::Tanh,
        other => {
            return Err(ImportError::Manifest(format!(
                "squash \"{other}\" is not one of none, tanh"
            )))
        }
    };
    if manifest.activate_output {
        return Err(ImportError::Manifest(
            "activate_output = true: an activation after the *output* Dense is not expressible \
             by PolicyHead{Regression}, whose only output nonlinearity is `squash` (spec 8.3). \
             All three supported frameworks leave their output layer linear."
                .to_owned(),
        ));
    }
    let Some((&out_dim, hidden)) = manifest.hidden.split_last().map(|(l, h)| (l, h.to_vec()))
    else {
        return Err(ImportError::Manifest(
            "hidden is empty: a StateEncoder{Mlp} needs at least one hidden layer (the output \
             Dense becomes PolicyHead{Regression}, which is always a Linear)"
                .to_owned(),
        ));
    };

    // The action tail. The adapter overrides the manifest, because the adapter is about *our*
    // robot and the manifest is about the source's (spec 13.4).
    let dim = task_dim as usize;
    let scale = pick(
        "scale",
        adapter.action.scale.as_ref(),
        manifest.action_scale.as_ref(),
        dim,
        1.0,
        &mut warnings,
    )?;
    // v2: the rest pose, source order -- stated by the adapter or the manifest, or not at all.
    let default_pos = match (&joints.default_pos, &manifest.default_joint_pos) {
        (None, None) => None,
        (a, m) => Some(pick(
            "default_pos",
            a.as_ref(),
            m.as_ref(),
            dim,
            0.0,
            &mut warnings,
        )?),
    };
    let offset = if adapter.action.use_default_offset {
        if adapter.action.offset.is_some() {
            return Err(ImportError::Adapter(
                "[action] states both `offset` and `use_default_offset = true`; one declaration \
                 of the offset, not two"
                    .to_owned(),
            ));
        }
        default_pos.clone().ok_or_else(|| {
            ImportError::Adapter(
                "[action] use_default_offset = true, but neither [joints] default_pos nor the \
                 manifest's default_joint_pos states the pose, and the import does not guess \
                 one (spec 13.4)"
                    .to_owned(),
            )
        })?
    } else {
        pick(
            "offset",
            adapter.action.offset.as_ref(),
            manifest.action_offset.as_ref(),
            dim,
            0.0,
            &mut warnings,
        )?
    };
    if let Some([lo, hi]) = adapter.action.clip {
        if !(squash == Squash::Tanh && lo <= -1.0 && hi >= 1.0) {
            return Err(ImportError::Clip {
                detail: format!(
                    "[action] clip = [{lo}, {hi}] under squash \"{}\". A clip is folded only \
                     where it cannot bind -- a tanh output inside [-1, 1] -- because the \
                     Learning IR has no clamp node (spec 8.3) and this packet adds none",
                    manifest.squash
                ),
            });
        }
    }

    let fold = fold_channels(
        manifest,
        &channels,
        perm.as_deref(),
        &scale,
        &offset,
        default_pos.as_deref(),
    )?;
    let observation = observation_ir(manifest, task, &channels, fold.stats, &mut warnings)?;
    // The head and its unnormalizer in *our* actuator order: exact, a row permutation.
    let (scale_ours, offset_ours) = match &perm {
        Some(p) => (permuted(&scale, p), permuted(&offset, p)),
        None => (scale.clone(), offset.clone()),
    };
    let learning = learning_graph(
        manifest,
        deployment,
        &observation,
        hidden,
        out_dim,
        activation,
        squash,
        &scale_ours,
        &offset_ours,
    )?;
    let weights = remap(
        &learning,
        manifest,
        weights,
        fold.columns.as_deref(),
        perm.as_deref(),
    )?;

    let report = MappingReport {
        framework: manifest.framework.clone(),
        versions: manifest.versions.clone(),
        robot: adapter.robot.name.clone(),
        obs_dim: manifest.obs_dim,
        action_dim: manifest.action_dim,
        hidden: manifest.hidden.clone(),
        activation: manifest.activation.clone(),
        squash: manifest.squash.clone(),
        log_std: manifest.log_std.clone(),
        source_std: manifest.source_std.as_ref().map(|v| {
            format!(
                "{} (the rows stay in import.json)",
                v.get("kind")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            )
        }),
        action_scale: scale.clone(),
        action_offset: offset.clone(),
        joints: names
            .iter()
            .enumerate()
            .map(|(i, name)| JointRow {
                source_index: i as u32,
                name: name.clone(),
                unit: joint_unit(space).to_owned(),
                severity: Severity::Ok,
                note: match (&perm, &joints.source_names) {
                    (Some(p), Some(src)) => format!(
                        "action[{i}] (source joint \"{}\") -> actuator \"{name}\" (ours {}); \
                         {} = {:.6} + {:.6} * a",
                        src[i],
                        p[i],
                        joint_unit(space),
                        offset[i],
                        scale[i]
                    ),
                    _ => format!(
                        "action[{i}] -> actuator \"{name}\"; {} = {:.6} + {:.6} * a",
                        joint_unit(space),
                        offset[i],
                        scale[i]
                    ),
                },
            })
            .collect(),
        channels: channels
            .iter()
            .map(|(map, declared)| {
                let dim = map.slice[1] - map.slice[0];
                // A channel that barely moved during training gets a tiny `obs_std`, and the
                // normalizer then amplifies anything off that scale by `1 / std`. brax clips
                // such a channel at `std_min_value = 1e-6` (`brax-ppo-so101.md` section 6);
                // the measured S2c run bottoms out at 1.8e-4 for the cube's z, which is not
                // the floor but is still 1,000x narrower than its neighbours. So the test is
                // *relative*: it catches both, and it is the fact a human deploying this has
                // to know, not a threshold anybody tuned.
                let narrow = manifest.obs_std.as_ref().is_some_and(|s| {
                    let widest = s.iter().fold(0.0f64, |a, b| a.max(*b));
                    s[map.slice[0] as usize..map.slice[1] as usize]
                        .iter()
                        .any(|v| *v < 1e-3 * widest)
                });
                ChannelRow {
                    source: map.source.clone(),
                    source_range: map.slice,
                    name: map.channel.clone(),
                    dim,
                    unit: format!("{:?}", declared.ty.unit),
                    severity: if narrow {
                        Severity::Warning
                    } else {
                        Severity::Ok
                    },
                    note: if narrow {
                        format!(
                            "{} -> ObsSource {}; at least one component's obs_std is over 1000x \
                             narrower than the widest channel's -- it hardly moved in training, \
                             and the normalizer amplifies anything off that scale by 1/std",
                            map.source,
                            source_name(&declared.source)
                        )
                    } else {
                        format!(
                            "{} -> ObsSource {}{}",
                            map.source,
                            source_name(&declared.source),
                            fold_note(map, &declared.source)
                        )
                    },
                }
            })
            .collect(),
        warnings,
        timing,
        actuators: Vec::new(),
    };
    Ok(RlImport {
        observation,
        learning,
        weights,
        report,
    })
}

/// The adapter's value, else the manifest's, else a constant — and a warning when the two
/// disagree, because a silent override is how a robot ends up driven in the wrong units.
fn pick(
    what: &str,
    adapter: Option<&Vec<f64>>,
    manifest: Option<&Vec<f64>>,
    dim: usize,
    default: f64,
    warnings: &mut Vec<String>,
) -> Result<Vec<f64>, ImportError> {
    let chosen = match (adapter, manifest) {
        (Some(a), Some(m)) => {
            if a != m {
                warnings.push(format!(
                    "the adapter's action {what} overrides the checkpoint's: {a:?} replaces {m:?}"
                ));
            }
            a.clone()
        }
        (Some(v), None) | (None, Some(v)) => v.clone(),
        (None, None) => {
            warnings.push(format!(
                "neither the adapter nor the checkpoint declares an action {what}; the \
                 unnormalizer uses {default} and the policy's output is the network's own"
            ));
            vec![default; dim]
        }
    };
    if chosen.len() != dim {
        return Err(ImportError::Manifest(format!(
            "action {what} has {} entries for a {dim}-joint action",
            chosen.len()
        )));
    }
    Ok(chosen)
}

// --- adapter v2 (packet M11/X2) --------------------------------------------------------------

/// Source terms the IR cannot compute on this robot (spec 28.14 "not on the ladder").
const UNCOMPUTABLE: [&str; 4] = [
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "velocity_commands",
];

/// The term a channel's free-text `source` names: `"mdp.projected_gravity"` and
/// `"projected_gravity(robot)"` are both `projected_gravity`. A v1 `source` such as
/// `"qpos[0:6]"` reads as `qpos`, which names nothing refused.
fn term_name(source: &str) -> &str {
    let s = source.trim();
    let s = s.strip_prefix("mdp.").unwrap_or(s);
    s.split(['(', '[', ' ']).next().unwrap_or(s)
}

/// `[joints]` -> (our actuator name of each source joint in source order, the permutation).
/// Exactly one of `source_order` (v1, recorded) and `source_names` (v2, permuted) (`IMP-008`).
fn resolve_joints(
    joints: &JointBlock,
    actuators: &[String],
    scene: &str,
    dim: u32,
) -> Result<(Vec<String>, Option<Vec<usize>>), ImportError> {
    let order = |detail: String| ImportError::JointOrder { detail };
    let Some(source) = &joints.source_names else {
        if joints.source_order.is_empty() {
            return Err(order(
                "[joints] declares neither `source_order` nor `source_names`; the source's \
                 joint order is not something the import may assume"
                    .to_owned(),
            ));
        }
        if !joints.rename.is_empty() {
            return Err(order(
                "[joints.rename] renames source names, and a v1 `source_order` is already in \
                 our names; declare `source_names` instead"
                    .to_owned(),
            ));
        }
        return Ok((joints.source_order.clone(), None));
    };
    if !joints.source_order.is_empty() {
        return Err(order(
            "[joints] declares both `source_order` and `source_names`: two statements of one \
             order, and nothing here may pick between them"
                .to_owned(),
        ));
    }
    if let Some(unused) = joints.rename.keys().find(|k| !source.contains(k)) {
        return Err(order(format!(
            "[joints.rename] renames \"{unused}\", which is not one of source_names {source:?}"
        )));
    }
    let mut names = Vec::with_capacity(source.len());
    let mut perm = Vec::with_capacity(source.len());
    for name in source {
        let ours = joints.rename.get(name).unwrap_or(name);
        let at =
            actuators
                .iter()
                .position(|a| a == ours)
                .ok_or_else(|| ImportError::UnknownJoint {
                    joint: ours.clone(),
                    scene: scene.to_owned(),
                    actuators: actuators.to_vec(),
                })?;
        if at >= dim as usize || perm.contains(&at) {
            return Err(order(format!(
                "source joint \"{name}\" resolves to actuator \"{ours}\" (index {at}), which is \
                 {} -- the resolution must be a permutation of the {dim} actuators the \
                 ActionSpec drives",
                if perm.contains(&at) {
                    "already taken by another source joint"
                } else {
                    "past the ActionSpec's width"
                }
            )));
        }
        names.push(ours.clone());
        perm.push(at);
    }
    Ok((names, Some(perm)))
}

/// `IMP-006`: the adapter's `policy_dt` and the manifest's `decimation * sim_dt`, each against
/// the Deployment IR's control period. Returns the report line, `None` when neither states one.
fn check_timing(
    adapter: &Adapter,
    manifest: &ImportManifest,
    deployment: &DeploymentIr,
) -> Result<Option<String>, ImportError> {
    let period = 1.0 / deployment.rate.control.as_hz_f64();
    let manifest_dt = match (manifest.decimation, manifest.sim_dt) {
        (Some(n), Some(dt)) => Some((f64::from(n) * dt, format!("decimation {n} x sim_dt {dt}"))),
        _ => None,
    };
    let adapter_dt = adapter
        .timing
        .as_ref()
        .map(|t| (t.policy_dt, "[timing] policy_dt".to_owned()));
    let mut said = Vec::new();
    for (dt, what) in [adapter_dt, manifest_dt].into_iter().flatten() {
        if (dt - period).abs() > 1e-9 * period {
            return Err(ImportError::Timing {
                detail: format!(
                    "{what} = {dt} s per policy step, but the Deployment IR's control period is \
                     {period} s. A policy run at another rate than it was trained at sees a \
                     different world per step; the import checks the period and never \
                     resamples it"
                ),
            });
        }
        said.push(format!("{what} = {dt} s"));
    }
    Ok((!said.is_empty()).then(|| {
        format!(
            "{} == the Deployment IR's control period {period} s",
            said.join(", ")
        )
    }))
}

/// The observation side of the fold: the `Normalize{MeanStd}` stats in *our* flat order
/// (`None`: no source statistics and nothing folded -- v1's identity `Range`), and the first
/// Dense's input column for each of our positions (`None`: the identity).
struct Fold {
    stats: Option<(Vec<f64>, Vec<f64>)>,
    columns: Option<Vec<usize>>,
}

/// Every channel's `(x - offset) * scale`, the joint permutation, the history order and the
/// previous action's un-normalization, folded into one per-element affine map and the
/// source's `(o - mean) / std`:
///
/// ```text
/// n_j = ((x - a_j) * b_j - mean_j) / std_j = (x - (a_j + mean_j / b_j)) / (std_j / b_j)
/// ```
///
/// exact algebra, evaluated once in f64. For a `PreviousAction` channel `x` is our row in
/// actuator units and the source saw its raw action, so `a` and `b` also carry the action
/// tail's inverse: `raw = (x - offset) / scale`. An element with `a = 0, b = 1` keeps the
/// source's numbers bit for bit, which is what leaves every v1 conversion unmoved.
// The names are the formula's above; the float comparisons test for exactly the identity.
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
fn fold_channels(
    manifest: &ImportManifest,
    channels: &[(&ChannelMap, &es_ir::task::ObsChannel)],
    perm: Option<&[usize]>,
    action_scale: &[f64],
    action_offset: &[f64],
    default_pos: Option<&[f64]>,
) -> Result<Fold, ImportError> {
    let n = manifest.obs_dim as usize;
    let dim = action_scale.len();
    let bad = |detail: String| ImportError::Channels { detail };
    let (mu, sigma) = match (&manifest.obs_mean, &manifest.obs_std) {
        (Some(m), Some(s)) => {
            if m.len() != n || s.len() != n {
                return Err(ImportError::Manifest(format!(
                    "obs_mean/obs_std are {}/{} long for obs_dim {n}",
                    m.len(),
                    s.len()
                )));
            }
            (Some(m), Some(s))
        }
        _ => (None, None),
    };
    let mut src_of: Vec<usize> = (0..n).collect();
    let mut a = vec![0.0; n];
    let mut b = vec![1.0; n];
    let mut folded = false;
    for (map, declared) in channels {
        let start = map.slice[0] as usize;
        let width = declared.ty.shape.dims().iter().product::<u64>() as usize;
        let history = map.history.unwrap_or(1) as usize;
        let previous = matches!(declared.source, ObsSource::PreviousAction { .. });
        let per_joint = previous
            || matches!(declared.source, ObsSource::JointState { .. })
                && matches!(declared.ty.frame, Frame::Joint(_));
        if (per_joint && (perm.is_some() || previous)) && width != dim {
            return Err(bad(format!(
                "channel \"{}\" is a per-joint channel {width} wide, and the action is {dim}: \
                 a joint permutation or an action tail cannot be applied to it",
                map.channel
            )));
        }
        let term_scale: Vec<f64> = match &map.scale {
            None => vec![1.0; width],
            Some(Scale::One(s)) => vec![*s; width],
            Some(Scale::Each(v)) if v.len() == width => v.clone(),
            Some(Scale::Each(v)) => {
                return Err(bad(format!(
                    "channel \"{}\" has {} scale entries for a {width}-wide channel",
                    map.channel,
                    v.len()
                )))
            }
        };
        let term_offset: Vec<f64> = match &map.offset {
            None => vec![0.0; width],
            Some(Offset::Each(v)) if v.len() == width => v.clone(),
            Some(Offset::Each(v)) => {
                return Err(bad(format!(
                    "channel \"{}\" has {} offset entries for a {width}-wide channel",
                    map.channel,
                    v.len()
                )))
            }
            Some(Offset::Named(name)) if name == "default_pos" => {
                let Some(pose) = default_pos.filter(|_| per_joint && width == dim) else {
                    return Err(bad(format!(
                        "channel \"{}\" declares offset = \"default_pos\", which needs a \
                         per-joint channel as wide as the action and a pose stated by \
                         [joints] default_pos or the manifest's default_joint_pos",
                        map.channel
                    )));
                };
                pose.to_vec()
            }
            Some(Offset::Named(other)) => {
                return Err(ImportError::Adapter(format!(
                    "channel \"{}\": offset = \"{other}\" is not \"default_pos\" or a vector",
                    map.channel
                )))
            }
        };
        if term_scale.iter().any(|s| *s == 0.0 || !s.is_finite()) {
            return Err(bad(format!(
                "channel \"{}\" has a zero or non-finite scale: a term the source multiplied by \
                 zero carries nothing, and no normalizer can divide it back out",
                map.channel
            )));
        }
        if previous {
            if let ObsSource::PreviousAction { initial } = &declared.source {
                // Isaac's `last_action` is zero on the first tick of an episode; ours is
                // `initial` in actuator units, which folds to zero only if it is the offset.
                let want = match perm {
                    Some(p) => permuted(action_offset, p),
                    None => action_offset.to_vec(),
                };
                let have = initial.clone().unwrap_or_else(|| vec![0.0; dim]);
                if have.len() != dim || have.iter().zip(&want).any(|(h, w)| (h - w).abs() > 1e-9) {
                    return Err(bad(format!(
                        "channel \"{}\" is the previous action, whose raw value is zero at an \
                         episode's first tick in the source; ours is `initial` in actuator \
                         units, so the Task IR must declare initial = {want:?} (the action \
                         offset, in our order), not {have:?}",
                        map.channel
                    )));
                }
            }
        }
        for e in 0..width * history {
            let (f, k) = (e / width, e % width);
            let our_k = match perm {
                Some(p) if per_joint => p[k],
                _ => k,
            };
            let our_f = match map.history_order {
                Some(HistoryOrder::NewestFirst) => history - 1 - f,
                _ => f,
            };
            let j = start + e;
            src_of[start + our_f * width + our_k] = j;
            let (aj, bj) = if previous {
                (
                    action_offset[k] + action_scale[k] * term_offset[k],
                    term_scale[k] / action_scale[k],
                )
            } else {
                (term_offset[k], term_scale[k])
            };
            folded |= aj != 0.0 || bj != 1.0;
            a[j] = aj;
            b[j] = bj;
        }
    }
    let identity = src_of.iter().enumerate().all(|(i, j)| i == *j);
    let stats = if mu.is_some() || folded {
        let mut mean = vec![0.0; n];
        let mut std = vec![0.0; n];
        for (ours, &j) in src_of.iter().enumerate() {
            let (m, s) = (mu.map_or(0.0, |v| v[j]), sigma.map_or(1.0, |v| v[j]));
            // Exactly the identity, so v1 keeps its bits.
            let untouched = a[j] == 0.0 && b[j] == 1.0;
            mean[ours] = if untouched { m } else { a[j] + m / b[j] };
            std[ours] = if untouched { s } else { s / b[j] };
        }
        Some((mean, std))
    } else {
        None
    };
    Ok(Fold {
        stats,
        columns: (!identity).then_some(src_of),
    })
}

/// `out[perm[i]] = v[i]`: a source-order vector in our order.
fn permuted(v: &[f64], perm: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (i, p) in perm.iter().enumerate() {
        out[*p] = v[i];
    }
    out
}

/// The v2 part of a channel row's note; empty for a v1 channel, whose note is unchanged.
fn fold_note(map: &ChannelMap, source: &ObsSource) -> String {
    let mut parts = Vec::new();
    if let Some(o) = &map.offset {
        parts.push(format!("offset {o:?}"));
    }
    if let Some(s) = &map.scale {
        parts.push(format!("scale {s:?}"));
    }
    if let Some(n) = map.history {
        parts.push(format!(
            "history {n} ({:?})",
            map.history_order.unwrap_or(HistoryOrder::NewestLast)
        ));
    }
    if matches!(source, ObsSource::PreviousAction { .. }) {
        parts.push("un-normalized by the action tail: raw = (row - offset) / scale".to_owned());
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("; folded into Normalize: {}", parts.join(", "))
    }
}

fn joint_unit(space: ActionSpace) -> &'static str {
    match space {
        ActionSpace::JointTorque => "N m",
        // An increment, not a pose: the same radians, added by `es-env` to the previous
        // command once per control tick (spec 8.5, packet M9/T1). The mapping report says so
        // because what a human checks against the datasheet here is a rate.
        ActionSpace::JointDelta => "rad per control tick",
        _ => "rad",
    }
}

fn source_name(source: &ObsSource) -> String {
    match source {
        ObsSource::Sensor { id, .. } => format!("Sensor({id})"),
        ObsSource::JointState {
            body,
            dof,
            quantity,
        } => {
            format!("JointState({body}, dof {dof}, {quantity:?})")
        }
        ObsSource::BodyPose(id) => format!("BodyPose({id})"),
        ObsSource::Language => "Language".to_owned(),
        ObsSource::PreviousAction { .. } => "PreviousAction".to_owned(),
    }
}

/// One `StateInput` per adapter channel, in slice order → `Concat` → `Normalize`.
///
/// The concatenated vector is `Dimensionless` for the reason
/// `tests/fixtures/rl/observation-reach.toml` gives: it mixes radians, metres and a
/// quaternion, and spec 5.4's unit algebra has no mixed unit.
fn observation_ir(
    manifest: &ImportManifest,
    task: &TaskIr,
    channels: &[(&ChannelMap, &es_ir::task::ObsChannel)],
    stats: Option<(Vec<f64>, Vec<f64>)>,
    warnings: &mut Vec<String>,
) -> Result<ObservationIr, ImportError> {
    let task_hash = task
        .task_hash()
        .map_err(|d| ImportError::Ir(format!("hashing the Task IR: {d}")))?;
    let mut ir = ObservationIr::new(1, task_hash);

    let n = channels.len() as u32;
    let concat_id = NodeId(n);
    let normalize_id = NodeId(n + 1);
    let mut parts = Vec::new();
    for (i, (map, declared)) in channels.iter().enumerate() {
        let id = NodeId(i as u32);
        ir.graph.insert(
            id,
            ObservationNode::StateInput {
                source: source_id(&declared.source),
                io: Io::source(declared.ty.clone()),
            },
        );
        // v2 `history = N`: a `TemporalWindowNode` of N frames between the input and the
        // concat, oldest first; a newest-first source was already reordered by the fold.
        match map.history.filter(|h| *h > 1) {
            None => {
                ir.graph.connect(id, OUT, concat_id, &in_port(i));
                parts.push(declared.ty.clone());
            }
            Some(h) => {
                let window_id = NodeId(n + 2 + i as u32);
                let mut stacked = declared.ty.clone();
                let width: u64 = stacked.shape.dims().iter().product();
                stacked.shape = Shape::new([width * u64::from(h)]);
                ir.graph.insert(
                    window_id,
                    ObservationNode::TemporalWindowNode {
                        window: TemporalWindow {
                            n_steps: h,
                            stride: 1,
                            align: Align::Hold,
                        },
                        io: Io {
                            inputs: vec![declared.ty.clone()],
                            output: stacked.clone(),
                        },
                    },
                );
                ir.graph.connect(id, OUT, window_id, &in_port(0));
                ir.graph.connect(window_id, OUT, concat_id, &in_port(i));
                parts.push(stacked);
            }
        }
    }

    let flat = PortType {
        elem: ElemType::F32,
        shape: Shape::new([u64::from(manifest.obs_dim)]),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    };
    ir.graph.insert(
        concat_id,
        ObservationNode::Concat {
            axis: 0,
            time_align: None,
            io: Io {
                inputs: parts,
                output: flat.clone(),
            },
        },
    );
    ir.graph.connect(concat_id, OUT, normalize_id, &in_port(0));

    // The source's own running statistics when it carried them, an identity `Range` when it
    // did not (the packet's rule): either way the output is `Normalized`, which is what makes
    // the tensor admissible as a policy input (`OBS-040`).
    // `fold_channels` already checked the lengths and folded every v2 term into them.
    let stats = if let Some((mean, std)) = stats {
        NormalizeStats::MeanStd { mean, std }
    } else {
        warnings.push(
            "the checkpoint carries no observation normalizer; the Observation IR gets an \
             identity Range{-1, 1} and the policy sees the raw channels"
                .to_owned(),
        );
        NormalizeStats::Range { lo: -1.0, hi: 1.0 }
    };
    let mut normalized = flat.clone();
    normalized.unit = Unit::Normalized { lo: -1.0, hi: 1.0 };
    ir.graph.insert(
        normalize_id,
        ObservationNode::Normalize {
            stats,
            io: Io {
                inputs: vec![flat],
                output: normalized.clone(),
            },
        },
    );
    ir.graph.outputs.push(PortRef::new(normalize_id, OUT));
    ir.outputs.insert(
        STATE_PORT.to_owned(),
        ObservationOutput {
            port: PortRef::new(normalize_id, OUT),
            ty: normalized,
        },
    );
    // Layer 2 of spec 7.5: one frame, no stacking. PPO acts on the current observation.
    ir.temporal.window = Some(TemporalWindow {
        n_steps: 1,
        stride: 1,
        align: Align::Hold,
    });
    Ok(ir)
}

/// The id an `ObsSource` is keyed by — which is what `XIR-002` matches a `StateInput` against,
/// so it is the source's own id and never a new one.
fn source_id(source: &ObsSource) -> StableId {
    match source {
        ObsSource::Sensor { id, .. } | ObsSource::BodyPose(id) => *id,
        ObsSource::JointState { body, .. } => *body,
        ObsSource::Language => StableId::from_path("language"),
        ObsSource::PreviousAction { .. } => ObsSource::previous_action_id(),
    }
}

/// `StateEncoder{Mlp} -> PolicyHead{Regression} -> ActionChunker -> Normalizer{Inverse}`.
///
/// `activate_output = true` on the encoder and never the manifest's value: the manifest
/// describes the *source* network, whose output Dense is linear, and that output Dense is our
/// head. What the encoder's flag decides is whether the last **hidden** layer is activated,
/// and in every one of these frameworks it is (module docs).
#[allow(clippy::too_many_arguments)]
fn learning_graph(
    manifest: &ImportManifest,
    deployment: &DeploymentIr,
    observation: &ObservationIr,
    hidden: Vec<u32>,
    out_dim: u32,
    activation: Activation,
    squash: Squash,
    scale: &[f64],
    offset: &[f64],
) -> Result<LearningGraph, ImportError> {
    let action = &deployment.action;
    if action.horizon != 1 || action.execute_chunk != 1 {
        return Err(ImportError::Ir(format!(
            "this deployment buffers a horizon of {} and executes {} of it; an imported PPO \
             actor emits exactly one action per control step (docs/design/rl-continuation.md \
             section 3)",
            action.horizon, action.execute_chunk
        )));
    }
    let state = observation
        .outputs
        .get(STATE_PORT)
        .ok_or_else(|| ImportError::Ir("the observation has no state output".to_owned()))?;
    let input = Port::new(STATE_PORT, state.ty.clone());
    let policy_ty = |shape: Shape, unit: Unit| PortType {
        elem: ElemType::F32,
        shape,
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    };
    let dim = u64::from(manifest.action_dim);
    let chunk_unit = Unit::Normalized { lo: -1.0, hi: 1.0 };
    let chunk = |unit: Unit| policy_ty(Shape::new([1, dim]), unit);
    let out_unit = match action.space {
        DeployedSpace::JointTorque => Unit::Torque,
        // An increment is rad *per control tick*, and `es_ir::cross` refuses `Unit::Angle` on
        // a `JointDelta` deployment by name (`XIR-031`): the documents would read as a
        // position policy and the runtime would add a position to a position.
        DeployedSpace::JointVelocity | DeployedSpace::JointDelta => Unit::AngularVelocity,
        _ => Unit::Angle,
    };

    let mut nodes: Graph<LearningNode> = Graph::new(1);
    nodes.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![input.clone()],
            kind: StateEncoderKind::Mlp {
                hidden,
                activation,
                activate_output: true,
            },
            out_dim,
        },
    );
    nodes.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![Port::new(
                "feat",
                policy_ty(Shape::new([u64::from(out_dim)]), Unit::Dimensionless),
            )],
            kind: HeadKind::Regression,
            action_dim: manifest.action_dim,
            horizon: 1,
            squash,
        },
    );
    nodes.insert(
        NodeId(2),
        LearningNode::ActionChunker {
            inputs: vec![Port::new("chunk", chunk(chunk_unit.clone()))],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: deployment.rate.control.as_hz_f64() as f32,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    nodes.insert(
        NodeId(3),
        LearningNode::Normalizer {
            inputs: vec![Port::new("actions", chunk(chunk_unit))],
            direction: NormalizeDir::Inverse,
            stats: StatsSource::MeanStd {
                mean: offset.to_vec(),
                std: scale.to_vec(),
            },
            out_unit: out_unit.clone(),
        },
    );
    nodes.connect(NodeId(0), OUT, NodeId(1), "feat");
    nodes.connect(NodeId(1), "chunk", NodeId(2), "chunk");
    nodes.connect(NodeId(2), "actions", NodeId(3), "actions");
    nodes.inputs.push(PortRef::new(NodeId(0), STATE_PORT));
    nodes.outputs.push(PortRef::new(NodeId(3), OUT));

    let control_hz = deployment.rate.control.as_hz_f64();
    let budget_ms = deployment.deadlines.inference_budget.0 as f32 / 1000.0;
    let replanning_hz = control_hz as f32;
    Ok(LearningGraph {
        schema_version: 1,
        inputs: vec![input.clone()],
        nodes,
        outputs: vec![Port::new("actions", chunk(out_unit))],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: [(STATE_PORT.to_owned(), input)].into_iter().collect(),
                action_dim: manifest.action_dim,
                horizon: 1,
                execute_chunk: 1,
                observation_window: 1,
                replanning_hz,
                execution_mode: match deployment.execution {
                    ExecutionMode::OpenLoopChunk => ActionExecutionMode::OpenLoopChunk,
                    ExecutionMode::RecedingHorizon => ActionExecutionMode::RecedingHorizon,
                    ExecutionMode::TemporalEnsemble { .. } => ActionExecutionMode::TemporalEnsemble,
                    ExecutionMode::RealTimeChunking => ActionExecutionMode::RealTimeChunking,
                },
                runtime: RuntimeHints {
                    // The same bound `es policy import-lerobot` declares: a checkpoint carries
                    // no measurement, so what is declared is the tighter of the deployment's
                    // own budget and its re-plan period (`LRN-052`).
                    expected_latency_ms: budget_ms.min(1000.0 / replanning_hz),
                    deadline_ms: budget_ms,
                    dtype: ElemType::F32,
                },
            },
        },
    })
}

/// The neutral keys onto the keys the lowering declares.
///
/// The pairing is by **order**, not by a formula: `lower_to_torch` numbers an `nn.Sequential`'s
/// members including the activation modules, and recomputing that index here would be a second
/// copy of `torch.rs`'s layout. Every pair is shape-checked, and `validate_keys` re-checks the
/// whole file against the lowering afterwards, so a misalignment is refused rather than packed.
///
/// v2 moves no number, only positions (packet M11/X2): `columns[k]` is the source input the
/// first Dense's column `k` reads, and `rows[i]` is where the head's row `i` goes -- the same
/// function of our state as the source's of its own, exactly. Both `None` for v1.
fn remap(
    learning: &LearningGraph,
    manifest: &ImportManifest,
    weights: &[u8],
    columns: Option<&[usize]>,
    rows: Option<&[usize]>,
) -> Result<Vec<u8>, ImportError> {
    let module = lower_to_torch(learning)
        .map_err(|e| ImportError::Ir(format!("lowering the Learning IR: {e}")))?;
    let header = parse_header(weights).map_err(|e| ImportError::Weights(e.to_string()))?;

    let mut neutral: Vec<String> = Vec::new();
    for i in 0..manifest.hidden.len() {
        neutral.push(format!("mlp.{i}.weight"));
        neutral.push(format!("mlp.{i}.bias"));
    }
    neutral.push("head.weight".to_owned());
    neutral.push("head.bias".to_owned());

    let ours: Vec<&String> = module
        .weight_keys
        .iter()
        .filter(|k| !k.ends_with(".*"))
        .collect();
    if ours.len() != neutral.len() {
        return Err(ImportError::Weights(format!(
            "the checkpoint carries {} tensors and the lowered module declares {}: {neutral:?} \
             against {ours:?}",
            neutral.len(),
            ours.len()
        )));
    }

    let mut out: Checkpoint = BTreeMap::new();
    let last = neutral.len() - 2;
    for (index, (from, to)) in neutral.iter().zip(&ours).enumerate() {
        let entry = header
            .get(from)
            .ok_or_else(|| ImportError::Weights(format!("no tensor named \"{from}\"")))?;
        let want = module.weight_shapes.get(*to);
        if want.is_some_and(|w| *w != entry.shape) {
            return Err(ImportError::Weights(format!(
                "\"{from}\" is {:?} but the lowering declares \"{to}\" as {:?}",
                entry.shape,
                want.expect("checked")
            )));
        }
        let mut values = tensor_f32(weights, entry)?;
        // The first hidden Dense's weight `[out, in]`: gather its input columns.
        if let (0, Some(cols)) = (index, columns) {
            let width = entry.shape.get(1).copied().unwrap_or(0) as usize;
            if cols.len() != width {
                return Err(ImportError::Weights(format!(
                    "\"{from}\" reads {width} inputs and the observation folds {}",
                    cols.len()
                )));
            }
            values = values
                .chunks_exact(width)
                .flat_map(|row| cols.iter().map(|c| row[*c]))
                .collect();
        }
        // The head's weight `[action, hidden]` and bias `[action]`: move source row `i` to
        // our row `rows[i]`.
        if let (true, Some(rows)) = (index >= last, rows) {
            let stride = values.len() / rows.len().max(1);
            let mut moved = vec![0.0f32; values.len()];
            for (i, to) in rows.iter().enumerate() {
                moved[to * stride..(to + 1) * stride]
                    .copy_from_slice(&values[i * stride..(i + 1) * stride]);
            }
            values = moved;
        }
        out.insert((*to).clone(), (entry.shape.clone(), values));
    }

    let bytes = write_safetensors(&out);
    let written = parse_header(&bytes).map_err(|e| ImportError::Weights(e.to_string()))?;
    validate_keys(&module, &written)
        .map_err(|e| ImportError::Weights(format!("the remapped checkpoint: {e}")))?;
    Ok(bytes)
}

/// One tensor's f32 values out of a safetensors blob.
fn tensor_f32(
    blob: &[u8],
    entry: &es_policy::weights::SafetensorsEntry,
) -> Result<Vec<f32>, ImportError> {
    if entry.dtype != "F32" {
        return Err(ImportError::Weights(format!(
            "dtype {} is not F32 (spec 8.4's runtime.dtype on this path)",
            entry.dtype
        )));
    }
    let len = u64::from_le_bytes(
        blob.get(..8)
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| ImportError::Weights("file is shorter than its length prefix".into()))?,
    ) as usize;
    let base = 8 + len;
    let (a, b) = (
        base + entry.offsets.0 as usize,
        base + entry.offsets.1 as usize,
    );
    let bytes = blob
        .get(a..b)
        .ok_or_else(|| ImportError::Weights("a tensor runs past the end of the file".into()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
