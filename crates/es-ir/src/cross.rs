//! Cross-IR checks: the five boundaries of spec 11.1, from `Task.ObservationSpec` through to
//! `Deployment.envelope` (P26).
//!
//! Each IR validates itself; nothing there can see across a boundary. This module is the pass
//! spec 11.1 calls "the most valuable check in v1.0": shape, normalization space, action
//! dimension and inference latency disagreements between two IRs, all at compile time.
//!
//! Boundaries, in the order spec 11.1 lists them:
//!
//! 1. `Task.ObservationSpec` <-> `Observation` inputs (spec 7.4)
//! 2. `Observation.outputs` <-> `Learning.inputs` (spec 8.4)
//! 3. `Learning.outputs` <-> `Task.ActionSpec` (spec 8.5)
//! 4. `Learning.runtime` <-> `Deployment.rate` (spec 8.4 `LRN-052`, spec 9.2)
//! 5. `Deployment.envelope` <-> robot capability (spec 9.3 `DEP-031`)
//!
//! plus `Evaluation` <-> `Task` / `Observation` (spec 10.4, `INV-15`).

use std::collections::BTreeSet;

use es_core::id::StableId;

use crate::codes;
use crate::deployment::{ActionSpace as DepSpace, DeploymentIr, ExecutionMode};
use crate::diag::Diagnostic;
use crate::evaluation::{AugmentationPolicy, EvaluationIr};
use crate::graph::IrNode;
use crate::learning::{ActionExecutionMode, LearningGraph};
use crate::observation::{ObservationIr, ObservationNode};
use crate::task::{ActionSpace as TaskSpace, ObsSource, TaskIr, TaskNode};
use crate::types::TimeRef;

// The `XIR-0xx` codes this module reports (severity and title in `codes.rs`):
// XIR-001 observation IR belongs to a different Task IR (spec 7.4)
// XIR-002 an Observation source node reads a channel the Task IR does not declare (spec 7.4)
// XIR-010 observation outputs and `PolicyContract::inputs` do not name the same tensors (spec 8.4)
// XIR-011 `TemporalWindow::n_steps` disagrees with `observation_window` (spec 7.5, spec 8.4)
// XIR-020 `action_dim` disagrees with the deployed action contract (spec 8.5, spec 9.2)
// XIR-021 execution mode disagrees between Learning IR and Deployment IR (spec 8.5, spec 9.2)
// XIR-022 `horizon` / `execute_chunk` disagree with the deployed action contract (spec 8.5)
// XIR-023 `replanning_hz` is not an integer divisor of the control rate (spec 8.4)
// XIR-024 `runtime.deadline_ms` does not fit the deployment inference budget (spec 8.4, spec 9.4)
// XIR-030 task `ActionSpec::dim` disagrees with `action_dim` (spec 8.4, spec 8.5)
// XIR-031 task `ActionSpec::space` disagrees with the deployed action space (spec 8.5)
// XIR-032 task `ActionSpec::control_rate_hz` disagrees with `Deployment.rate.control` (spec 9.2)
// XIR-040 Evaluation IR references a different Task or Observation IR (spec 10.4)
// XIR-050 `INV-15`: an augmentation node would stay on during evaluation (spec 7.3, spec 10.4)
// XIR-051 the evaluation allow-list names a node that is not an Observation `Augment` node

/// The five IRs of one execution, as `check` sees them. Evaluation is optional: a deployment
/// bundle has no Evaluation IR, and the first four boundaries are checkable without it.
#[derive(Clone, Copy, Debug)]
pub struct IrBundle<'a> {
    pub task: &'a TaskIr,
    pub observation: &'a ObservationIr,
    pub learning: &'a LearningGraph,
    pub deployment: &'a DeploymentIr,
    pub evaluation: Option<&'a EvaluationIr>,
}

/// Every rule spec 11.1 states across an IR boundary. An empty result means the five IRs agree;
/// it says nothing about whether each of them is valid on its own (`TaskIr::validate` and its
/// siblings answer that, and run first).
pub fn check(bundle: &IrBundle) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    task_observation(bundle, &mut out);
    observation_learning(bundle, &mut out);
    learning_deployment(bundle, &mut out);
    task_action(bundle, &mut out);
    robot_capability(bundle, &mut out);
    if let Some(ev) = bundle.evaluation {
        evaluation(bundle, ev, &mut out);
    }
    out
}

// --- 1. Task.ObservationSpec <-> Observation inputs (spec 7.4, lines 813-838) --------------

/// What a declared channel and a source node have to agree on to be the same channel.
type SourceKey = (&'static str, Option<StableId>);

fn channel_key(src: &ObsSource) -> SourceKey {
    match src {
        ObsSource::Sensor { id, .. } => ("sensor", Some(*id)),
        ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => ("state", Some(*body)),
        ObsSource::Language => ("language", None),
    }
}

fn node_key(node: &ObservationNode) -> Option<SourceKey> {
    match node {
        ObservationNode::ImageInput { sensor, .. } => Some(("sensor", Some(*sensor))),
        ObservationNode::StateInput { source, .. } => Some(("state", Some(*source))),
        ObservationNode::LanguageInput { .. } => Some(("language", None)),
        _ => None,
    }
}

fn task_observation(b: &IrBundle, out: &mut Vec<Diagnostic>) {
    match b.task.task_hash() {
        Ok(h) if h == b.observation.task_ref => {}
        Ok(h) => out.push(
            Diagnostic::new(
                codes::XIR_001,
                format!(
                    "observation.task_ref is {}, the task hashes to {}",
                    short(&b.observation.task_ref),
                    short(&h)
                ),
            )
            .with_hint("spec 7.4: several Observation IRs share one task_hash; this is not one"),
        ),
        Err(d) => out.push(d),
    }

    let channels = &b.task.observation_spec.channels;
    for (id, node) in &b.observation.graph.nodes {
        let Some(key) = node_key(node) else { continue };
        let declared: Vec<_> = channels
            .iter()
            .filter(|(_, ch)| channel_key(&ch.source) == key)
            .collect();
        match declared.as_slice() {
            [] => out.push(
                Diagnostic::new(
                    codes::XIR_002,
                    format!("{} reads {:?}, which no channel declares", node.kind(), key),
                )
                .at(*id)
                .with_hint("spec 7.4: Task IR declares the channel, Observation IR implements it"),
            ),
            // Task declares, Observation implements: the declared type is what flows in.
            [(name, ch)] => {
                if let Err(d) = ch.ty.compatible(&node.io().output) {
                    out.push(d.at(*id).with_hint(format!(
                        "spec 7.4: channel \"{name}\" is declared with a different type"
                    )));
                }
            }
            // Several channels name one source id: one joint block read at two quantities,
            // a position channel and a velocity one (packet M8/S4e). `StateInput` carries no
            // quantity of its own, so what tells them apart is the declared **type** — and
            // `JointQuantity::unit()` is exactly that type's unit, so the pair is keyed by
            // `(id, quantity)` after all. Two channels a node's type cannot tell apart are
            // refused by name rather than resolved by map order.
            many => {
                let fits: Vec<_> = many
                    .iter()
                    .filter(|(_, ch)| ch.ty.compatible(&node.io().output).is_ok())
                    .collect();
                if fits.len() != 1 {
                    let names: Vec<&str> = many.iter().map(|(n, _)| n.as_str()).collect();
                    out.push(
                        Diagnostic::new(
                            codes::XIR_002,
                            format!(
                                "{} reads {:?}, which {} channels declare ({}), and its type \
                                 matches {} of them",
                                node.kind(),
                                key,
                                many.len(),
                                names.join(", "),
                                fits.len()
                            ),
                        )
                        .at(*id)
                        .with_hint(
                            "spec 7.4: two channels on one source id are told apart by the \
                             quantity they declare, which is the unit of their type",
                        ),
                    );
                }
            }
        }
    }
}

// --- 2. Observation.outputs <-> Learning.inputs (spec 8.4, lines 973-1024) -----------------

fn observation_learning(b: &IrBundle, out: &mut Vec<Diagnostic>) {
    let contract = &b.learning.policy.contract;
    let specs = b.observation.propagate_image_specs().unwrap_or_default();

    for (name, port) in &contract.inputs {
        let Some(produced) = b.observation.outputs.get(name) else {
            out.push(
                Diagnostic::new(
                    codes::XIR_010,
                    format!("policy input \"{name}\" is not produced by the Observation IR"),
                )
                .on_port(name.clone())
                .with_hint("spec 8.4: `input` names the Observation IR outputs verbatim"),
            );
            continue;
        };

        // The boundary is the policy tick: a sensor clock and a camera frame legitimately
        // become `TimeRef::Tick` / `Frame::Policy` here, and a tensor port need not restate
        // the ImageSpec. Everything else -- elem, shape, unit, frame -- is compared strictly.
        let mut crossing = produced.ty.clone();
        if matches!(port.ty.time, TimeRef::Tick) {
            crossing.time = TimeRef::Tick;
        }
        if port.ty.image.is_none() {
            crossing.image = None;
        }
        if let Err(d) = crossing.compatible(&port.ty) {
            out.push(d.on_port(name.clone()).with_hint(format!(
                "spec 8.4: observation output \"{name}\" does not match the policy contract"
            )));
        }
        // Propagated geometry (INV-14) against the one the output advertises. Only the
        // geometry: dtype, colour space and channel layout are the tensor's business, the
        // resolution and the intrinsics are the camera's.
        if let Some(spec) = specs.get(&produced.port) {
            let stale = produced.ty.image.as_ref().is_none_or(|declared| {
                (declared.width, declared.height, declared.intrinsics)
                    != (spec.width, spec.height, spec.intrinsics)
            });
            if stale {
                out.push(
                    Diagnostic::new(
                        codes::TYPE_020,
                        format!(
                            "output \"{name}\" advertises an ImageSpec the graph does not \
                             produce ({}x{} after propagation)",
                            spec.width, spec.height
                        ),
                    )
                    .at(produced.port.node)
                    .on_port(name.clone())
                    .with_hint("spec 7.2 INV-14: Resize/Crop transform the intrinsics"),
                );
            }
        }
        if let Err(d) = port.ty.check_policy_input() {
            out.push(d.on_port(name.clone()).with_hint(
                "spec 5.4: a policy input is Normalized, Dimensionless or Token -- insert a \
                 Normalize node",
            ));
        }
    }

    for name in b.observation.outputs.keys() {
        if !contract.inputs.contains_key(name) {
            out.push(
                Diagnostic::new(
                    codes::XIR_010,
                    format!("observation output \"{name}\" is not a policy input"),
                )
                .on_port(name.clone())
                .with_hint("spec 8.4: the two sides name the same tensors"),
            );
        }
    }

    // Layer 2 of spec 7.5 fixes the tensor's time extent; layer 3 has to agree with it.
    let n_steps = b.observation.temporal.window.map_or(1, |w| w.n_steps);
    if n_steps != contract.observation_window {
        out.push(
            Diagnostic::new(
                codes::XIR_011,
                format!(
                    "TemporalWindow n_steps {n_steps} against observation_window {}",
                    contract.observation_window
                ),
            )
            .with_hint("spec 7.5 / spec 8.4: ACT is n_steps = 1, Diffusion Policy is 2"),
        );
    }
}

// --- 3./4. Learning <-> Deployment (spec 8.4, spec 8.5, spec 9.2, lines 1161-1191) ---------

fn same_mode(learning: ActionExecutionMode, deployed: &ExecutionMode) -> bool {
    matches!(
        (learning, deployed),
        (
            ActionExecutionMode::OpenLoopChunk,
            ExecutionMode::OpenLoopChunk
        ) | (
            ActionExecutionMode::RecedingHorizon,
            ExecutionMode::RecedingHorizon
        ) | (
            ActionExecutionMode::TemporalEnsemble,
            ExecutionMode::TemporalEnsemble { .. }
        ) | (
            ActionExecutionMode::RealTimeChunking,
            ExecutionMode::RealTimeChunking
        )
    )
}

fn learning_deployment(b: &IrBundle, out: &mut Vec<Diagnostic>) {
    let c = &b.learning.policy.contract;
    let a = b.deployment.action;

    if c.action_dim as usize != a.dim {
        out.push(
            Diagnostic::new(
                codes::XIR_020,
                format!(
                    "policy action_dim {} against deployment action.dim {} ({} envelope joints)",
                    c.action_dim,
                    a.dim,
                    b.deployment.safety.n_joints()
                ),
            )
            .with_hint("spec 9.2: the Safety Plane validates exactly the vector the policy emits"),
        );
    }
    if !same_mode(c.execution_mode, &b.deployment.execution) {
        out.push(
            Diagnostic::new(
                codes::XIR_021,
                format!(
                    "policy {:?} against deployment {:?}",
                    c.execution_mode, b.deployment.execution
                ),
            )
            .with_hint("spec 8.5 / spec 9.2: one chunk, one execution mode"),
        );
    }
    if c.horizon as usize != a.horizon || c.execute_chunk as usize != a.execute_chunk {
        out.push(
            Diagnostic::new(
                codes::XIR_022,
                format!(
                    "policy horizon {} / execute_chunk {} against deployment {} / {}",
                    c.horizon, c.execute_chunk, a.horizon, a.execute_chunk
                ),
            )
            .with_hint("spec 8.5: the chunk the runtime buffers is the chunk the policy predicts"),
        );
    }

    let control_hz = b.deployment.rate.control.as_hz_f64();
    let ratio = control_hz / f64::from(c.replanning_hz);
    if !(c.replanning_hz > 0.0 && ratio >= 1.0 && (ratio - ratio.round()).abs() < 1e-6) {
        out.push(
            Diagnostic::new(
                codes::XIR_023,
                format!(
                    "replanning_hz {} against dt_ctrl {control_hz} Hz",
                    c.replanning_hz
                ),
            )
            .with_hint("spec 8.4: replanning_hz and dt_ctrl stand in an integer relation"),
        );
    }

    let budget_ms = b.deployment.deadlines.inference_budget.0 as f64 / 1000.0;
    let period_ms = b.deployment.rate.control_period().0 as f64 / 1000.0;
    let latency_ms = f64::from(c.runtime.expected_latency_ms);
    if latency_ms > budget_ms {
        out.push(
            Diagnostic::new(
                codes::LRN_052,
                format!(
                    "measured latency {latency_ms} ms exceeds the inference budget {budget_ms} ms"
                ),
            )
            .with_hint("spec 8.4 LRN-052: async inference (spec 8.6) or a wider budget"),
        );
    } else if latency_ms > period_ms && a.execute_chunk <= 1 {
        out.push(
            Diagnostic::new(
                codes::LRN_052,
                format!(
                    "latency {latency_ms} ms exceeds the control period {period_ms} ms with \
                     execute_chunk {}",
                    a.execute_chunk
                ),
            )
            .with_hint("spec 8.4 LRN-052 (a): chunk buffering needs execute_chunk > 1"),
        );
    }
    if f64::from(c.runtime.deadline_ms) > budget_ms {
        out.push(
            Diagnostic::new(
                codes::XIR_024,
                format!(
                    "runtime.deadline_ms {} exceeds deadlines.inference_budget {budget_ms} ms",
                    c.runtime.deadline_ms
                ),
            )
            .with_hint("spec 9.4: the watchdog budget is what actually triggers the fallback"),
        );
    }
}

// --- 3. Task.ActionSpec <-> Learning / Deployment (spec 8.5, lines 1025-1065) --------------

fn maps_to(task: TaskSpace, deployed: DepSpace) -> bool {
    matches!(
        (task, deployed),
        (TaskSpace::JointPosition, DepSpace::JointPosition)
            | (TaskSpace::JointVelocity, DepSpace::JointVelocity)
            | (TaskSpace::JointTorque, DepSpace::JointTorque)
            | (TaskSpace::EePose, DepSpace::EePose)
            | (TaskSpace::EeDelta, DepSpace::EeDelta)
            | (TaskSpace::Gripper, DepSpace::Gripper)
            // A composite action is assembled from several declarations; the Task IR
            // declares one of them, so any space may feed it.
            | (_, DepSpace::Composite)
    )
}

fn task_action(b: &IrBundle, out: &mut Vec<Diagnostic>) {
    let c = &b.learning.policy.contract;
    for (id, node) in &b.task.graph.nodes {
        let TaskNode::ActionSpec {
            space,
            dim,
            control_rate_hz,
        } = node
        else {
            continue;
        };
        if *dim != c.action_dim {
            out.push(
                Diagnostic::new(
                    codes::XIR_030,
                    format!(
                        "task ActionSpec dim {dim} against action_dim {}",
                        c.action_dim
                    ),
                )
                .at(*id)
                .with_hint("spec 8.4: the compiler checks ActionSpec(Task IR) against `output`"),
            );
        }
        if !maps_to(*space, b.deployment.action.space) {
            out.push(
                Diagnostic::new(
                    codes::XIR_031,
                    format!(
                        "task declares {space:?}, deployment executes {:?}",
                        b.deployment.action.space
                    ),
                )
                .at(*id)
                .with_hint("spec 8.5: the action space fixes the unit the actuator receives"),
            );
        }
        let control_hz = b.deployment.rate.control.as_hz_f64();
        if (f64::from(*control_rate_hz) - control_hz).abs() > 1e-3 {
            out.push(
                Diagnostic::new(
                    codes::XIR_032,
                    format!("task control_rate_hz {control_rate_hz} against rate.control {control_hz} Hz"),
                )
                .at(*id)
                .with_hint("spec 9.2: dt_ctrl is one number, declared twice"),
            );
        }
    }
}

// --- 5. Deployment.envelope <-> robot capability (spec 9.3, spec 11.1) ---------------------

/// `DEP-031` as far as the IRs can see it. `RobotRef` carries a name, a target and a joint
/// count -- no torque or velocity capability table -- so the envelope's *values* cannot be
/// checked against the hardware here; only its width can, against the joint count the Task IR declares
/// for the same robot. The value check needs a robot capability description (spec 11.1
/// `Deployment.envelope <-> Robot capability`) that no IR carries yet.
fn robot_capability(b: &IrBundle, out: &mut Vec<Diagnostic>) {
    let declared: Option<u32> =
        b.task
            .observation_spec
            .channels
            .values()
            .find_map(|ch| match ch.source {
                ObsSource::JointState { dof, .. } => Some(dof),
                _ => None,
            });
    let Some(dof) = declared else { return };
    let n = b.deployment.safety.n_joints();
    if n > dof as usize {
        out.push(
            Diagnostic::new(
                codes::DEP_031,
                format!("the envelope covers {n} joints, the task's robot has {dof}"),
            )
            .with_hint("spec 9.3: every envelope entry is per joint of the robot it deploys to"),
        );
    }
}

// --- Evaluation <-> Task / Observation (spec 10.4, lines 1368-1374) ------------------------

fn evaluation(b: &IrBundle, ev: &EvaluationIr, out: &mut Vec<Diagnostic>) {
    // `task` / `observation` are references: an authoring path, or the hash itself. Only the
    // second form is checkable here.
    check_ref("task", &ev.task, b.task.task_hash(), out);
    check_ref(
        "observation",
        &ev.observation,
        b.observation.observation_hash(),
        out,
    );

    // INV-15. An `Augment` node with `training_only = true` is switched off structurally
    // (spec 7.3); one that is not has to be named in the allow-list, with a justification,
    // or evaluation runs with augmentation on.
    let empty = BTreeSet::new();
    let allowed = match &ev.augmentation {
        AugmentationPolicy::Disabled => &empty,
        AugmentationPolicy::AllowList { nodes, .. } => nodes,
    };
    let mut augment_nodes = BTreeSet::new();
    for (id, node) in &b.observation.graph.nodes {
        let ObservationNode::Augment { training_only, .. } = node else {
            continue;
        };
        augment_nodes.insert(id.0.to_string());
        if !*training_only && !allowed.contains(&id.0.to_string()) {
            out.push(
                Diagnostic::new(
                    codes::XIR_050,
                    format!("augmentation node {} would run during evaluation", id.0),
                )
                .at(*id)
                .with_hint(
                    "INV-15 (spec 7.3, spec 10.4): set training_only, or name the node in the \
                     evaluation allow-list with a justification",
                ),
            );
        }
    }
    for name in allowed {
        if !augment_nodes.contains(name) {
            out.push(
                Diagnostic::new(
                    codes::XIR_051,
                    format!("allow-list names \"{name}\", which is no Augment node of this graph"),
                )
                .with_hint("INV-15: the allow-list names Observation IR nodes by node id"),
            );
        }
    }
}

// --- Helpers -------------------------------------------------------------------------------

/// The first four bytes of a digest, for a message a human reads.
fn short(d: &[u8; 32]) -> String {
    use std::fmt::Write;
    d[..4].iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// One `Evaluation` reference against the hash the bundle actually has.
fn check_ref(
    what: &str,
    reference: &str,
    actual: Result<[u8; 32], Diagnostic>,
    out: &mut Vec<Diagnostic>,
) {
    let (Some(want), Ok(have)) = (digest_of(reference), actual) else {
        return;
    };
    if want != have {
        out.push(
            Diagnostic::new(
                codes::XIR_040,
                format!(
                    "evaluation {what} reference is {}, the bundle hashes to {}",
                    short(&want),
                    short(&have)
                ),
            )
            .with_hint("spec 10.4: equal evaluation_hash means equal conditions"),
        );
    }
}

/// A reference string that *is* a digest: 64 hex characters, optionally `b3:`-prefixed
/// (spec 8.4 writes hashes that way). Anything else is an authoring path.
fn digest_of(reference: &str) -> Option<[u8; 32]> {
    let hex = reference.strip_prefix("b3:").unwrap_or(reference);
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_parsing_accepts_only_a_real_digest() {
        assert_eq!(digest_of("tasks/pick_cube.toml"), None);
        assert_eq!(digest_of(&"z".repeat(64)), None);
        let hex = "ab".repeat(32);
        assert_eq!(digest_of(&hex), Some([0xabu8; 32]));
        assert_eq!(digest_of(&format!("b3:{hex}")), Some([0xabu8; 32]));
    }

    #[test]
    fn execution_modes_pair_up() {
        assert!(same_mode(
            ActionExecutionMode::TemporalEnsemble,
            &ExecutionMode::TemporalEnsemble { decay: 0.01 }
        ));
        assert!(!same_mode(
            ActionExecutionMode::RecedingHorizon,
            &ExecutionMode::OpenLoopChunk
        ));
    }
}
