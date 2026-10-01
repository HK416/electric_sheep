//! The two documents an import builds: the Observation IR, one `StateInput` per adapter channel,
//! concatenated and normalized; and the Learning IR,
//! `StateEncoder{Mlp} -> PolicyHead{Regression} -> ActionChunker -> Normalizer{Inverse}`.

use es_core::StableId;
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
use es_ir::task::{ObsSource, TaskIr};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};

use super::adapter::ChannelMap;
use super::manifest::ImportManifest;
use super::{ImportError, STATE_PORT};

/// One `StateInput` per adapter channel, in slice order → `Concat` → `Normalize`.
///
/// The concatenated vector is `Dimensionless` for the reason
/// `tests/fixtures/rl/observation-reach.toml` gives: it mixes radians, metres and a
/// quaternion, and spec 5.4's unit algebra has no mixed unit.
pub(super) fn observation_ir(
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
pub(super) fn learning_graph(
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
