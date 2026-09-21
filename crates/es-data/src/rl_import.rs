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

use std::collections::BTreeMap;

use es_core::StableId;
use es_ir::codes::{IMP_001, IMP_002, IMP_003, IMP_004, IMP_005};
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
    #[serde(default)]
    pub action_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub action_offset: Option<Vec<f64>>,
    #[serde(default)]
    pub joint_order: Option<Vec<String>>,
    #[serde(default)]
    pub synthetic: Option<String>,
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
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotBlock {
    pub name: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointBlock {
    /// The framework's own action order, by our actuator names.
    pub source_order: Vec<String>,
    /// `"rad"` or `"deg"`. `"deg"` is refused (`IMP-003`), never converted.
    pub units: String,
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
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationBlock {
    pub channels: Vec<ChannelMap>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelMap {
    /// Where the framework read this block from, e.g. `"qpos[0:6]"`. Recorded for a human,
    /// never interpreted — the meaning comes from `channel`.
    pub source: String,
    /// `[start, end)` in the source's flat observation vector.
    pub slice: [u32; 2],
    /// The Task IR `ObservationSpec` channel this block feeds.
    pub channel: String,
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
    #[error("{IMP_005}: {detail}")]
    Channels { detail: String },
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

    // IMP-001 / IMP-002 / IMP-003 / IMP-004: the joint block.
    let joints = &adapter.joints;
    if joints.source_order.len() as u32 != task_dim || manifest.action_dim != task_dim {
        return Err(ImportError::JointCount {
            adapter: joints.source_order.len(),
            manifest: manifest.action_dim,
            task: task_dim,
        });
    }
    for name in &joints.source_order {
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
        ActionSpace::JointTorque => "torque",
        _ => "",
    };
    if adapter.action.kind != expected {
        return Err(ImportError::ActionKind {
            kind: adapter.action.kind.clone(),
            space,
        });
    }
    if let Some(order) = &manifest.joint_order {
        if order != &joints.source_order {
            warnings.push(format!(
                "the checkpoint recorded joint order {order:?}, the adapter declares {:?}; the \
                 adapter wins (spec 13.4) -- check that this permutation is intended",
                joints.source_order
            ));
        }
    }

    // IMP-005: the channels must tile `obs_dim` exactly, in order, and each must be a channel
    // the Task IR declares at the same width.
    let mut cursor = 0u32;
    let mut channels = Vec::new();
    for map in &adapter.observation.channels {
        let [start, end] = map.slice;
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
        if width != u64::from(end - start) {
            return Err(ImportError::Channels {
                detail: format!(
                    "channel \"{}\" takes {} source values but the Task IR declares it {width} \
                     wide",
                    map.channel,
                    end - start
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
    let offset = pick(
        "offset",
        adapter.action.offset.as_ref(),
        manifest.action_offset.as_ref(),
        dim,
        0.0,
        &mut warnings,
    )?;

    let observation = observation_ir(manifest, task, &channels, &mut warnings)?;
    let learning = learning_graph(
        manifest,
        deployment,
        &observation,
        hidden,
        out_dim,
        activation,
        squash,
        &scale,
        &offset,
    )?;
    let weights = remap(&learning, manifest, weights)?;

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
        joints: joints
            .source_order
            .iter()
            .enumerate()
            .map(|(i, name)| JointRow {
                source_index: i as u32,
                name: name.clone(),
                unit: joint_unit(space).to_owned(),
                severity: Severity::Ok,
                note: format!(
                    "action[{i}] -> actuator \"{name}\"; {} = {:.6} + {:.6} * a",
                    joint_unit(space),
                    offset[i],
                    scale[i]
                ),
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
                            "{} -> ObsSource {}",
                            map.source,
                            source_name(&declared.source)
                        )
                    },
                }
            })
            .collect(),
        warnings,
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

fn joint_unit(space: ActionSpace) -> &'static str {
    match space {
        ActionSpace::JointTorque => "N m",
        _ => "rad",
    }
}

fn source_name(source: &ObsSource) -> String {
    match source {
        ObsSource::Sensor { id, .. } => format!("Sensor({id})"),
        ObsSource::JointState { body, dof } => format!("JointState({body}, dof {dof})"),
        ObsSource::BodyPose(id) => format!("BodyPose({id})"),
        ObsSource::Language => "Language".to_owned(),
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
    warnings: &mut Vec<String>,
) -> Result<ObservationIr, ImportError> {
    let task_hash = task
        .task_hash()
        .map_err(|d| ImportError::Ir(format!("hashing the Task IR: {d}")))?;
    let mut ir = ObservationIr::new(1, task_hash);

    let concat_id = NodeId(channels.len() as u32);
    let normalize_id = NodeId(channels.len() as u32 + 1);
    let mut parts = Vec::new();
    for (i, (_, declared)) in channels.iter().enumerate() {
        let id = NodeId(i as u32);
        ir.graph.insert(
            id,
            ObservationNode::StateInput {
                source: source_id(&declared.source),
                io: Io::source(declared.ty.clone()),
            },
        );
        ir.graph.connect(id, OUT, concat_id, &in_port(i));
        parts.push(declared.ty.clone());
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
    let stats = if let (Some(mean), Some(std)) = (&manifest.obs_mean, &manifest.obs_std) {
        let n = manifest.obs_dim as usize;
        if mean.len() != n || std.len() != n {
            return Err(ImportError::Manifest(format!(
                "obs_mean/obs_std are {}/{} long for obs_dim {n}",
                mean.len(),
                std.len()
            )));
        }
        NormalizeStats::MeanStd {
            mean: mean.clone(),
            std: std.clone(),
        }
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
        DeployedSpace::JointVelocity => Unit::AngularVelocity,
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
fn remap(
    learning: &LearningGraph,
    manifest: &ImportManifest,
    weights: &[u8],
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
    for (from, to) in neutral.iter().zip(&ours) {
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
        out.insert(
            (*to).clone(),
            (entry.shape.clone(), tensor_f32(weights, entry)?),
        );
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
