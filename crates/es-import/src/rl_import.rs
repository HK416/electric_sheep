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

use es_ir::codes::{
    IMP_001, IMP_002, IMP_003, IMP_004, IMP_005, IMP_006, IMP_007, IMP_008, IMP_009,
};
use es_ir::deployment::DeploymentIr;
use es_ir::learning::{Activation, LearningGraph, Squash};
use es_ir::observation::ObservationIr;
use es_ir::task::{ActionSpace, TaskIr, TaskNode};

mod adapter;
mod documents;
mod manifest;
mod report;
mod resolve;
mod weights;

pub use adapter::{
    ActionBlock, ActuatorBlock, Adapter, ChannelMap, HistoryOrder, JointBlock, ObservationBlock,
    Offset, RobotBlock, Scale, TimingBlock,
};
pub use manifest::ImportManifest;
pub use report::{
    actuator_rows, ActuatorRow, ChannelRow, JointRow, MappingReport, SceneActuator, Severity,
};

use documents::{learning_graph, observation_ir};
use report::{fold_note, joint_unit, source_name};
use resolve::{check_timing, fold_channels, permuted, resolve_joints, term_name, UNCOMPUTABLE};
use weights::remap;

/// The name the Observation IR exposes its one concatenated vector under, and therefore the
/// name the policy contract binds (`XIR-010`). An RL policy reads one flat state vector.
pub const STATE_PORT: &str = "state";

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
