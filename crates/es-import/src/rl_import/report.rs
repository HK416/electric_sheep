//! The Semantic Mapping Report (spec 14.4): a row per joint, per observation channel and per
//! declared actuator quantity, and the prose those rows carry.

use std::collections::BTreeMap;

use es_ir::task::{ActionSpace, ObsSource};
use serde::Serialize;

use super::adapter::{Adapter, ChannelMap, HistoryOrder};

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

/// The v2 part of a channel row's note; empty for a v1 channel, whose note is unchanged.
pub(super) fn fold_note(map: &ChannelMap, source: &ObsSource) -> String {
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

pub(super) fn joint_unit(space: ActionSpace) -> &'static str {
    match space {
        ActionSpace::JointTorque => "N m",
        // An increment, not a pose: the same radians, added by `es-env` to the previous
        // command once per control tick (spec 8.5, packet M9/T1). The mapping report says so
        // because what a human checks against the datasheet here is a rate.
        ActionSpace::JointDelta => "rad per control tick",
        _ => "rad",
    }
}

pub(super) fn source_name(source: &ObsSource) -> String {
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
