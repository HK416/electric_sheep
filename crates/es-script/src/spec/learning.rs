//! The two Learning IR families `es project generate` writes: the teacher's state MLP (plan H's
//! `learning-teacher.toml`, `rl_games`' Shadow Hand actor) and the student's camera ACT
//! (plan N's `learning-views.toml`; `mad` its shared-encoder `Sum`, `learning-mad.toml`).

use std::collections::BTreeMap;

use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, FusionKind, HeadKind,
    LearningGraph, LearningNode, NormalizeDir, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, StatsSource, TemporalKind, VisionBackbone, WeightsRef,
};
use es_ir::observation::ObservationIr;
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

use super::generate::Gen;
use super::project::{Family, Preset, Student};
use super::{refuse, SpecError};

fn policy(shape: Shape, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape,
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

fn feature(name: &str, dim: u64) -> Port {
    Port::new(name, policy(Shape::new([dim]), Unit::Dimensionless))
}

fn signed() -> Unit {
    Unit::Normalized { lo: -1.0, hi: 1.0 }
}

/// The observation output `name` as a policy input: its shape and unit, in the policy frame.
fn input(obs: &ObservationIr, name: &str) -> Port {
    let ty = &obs.outputs[name].ty;
    Port::new(name, policy(ty.shape.clone(), ty.unit.clone()))
}

struct Builder {
    nodes: Graph<LearningNode>,
}

impl Builder {
    fn add(&mut self, node: LearningNode) -> NodeId {
        let id = NodeId(self.nodes.nodes.len() as u32);
        self.nodes.insert(id, node);
        id
    }
}

/// `(lo + hi) / 2` and `(hi - lo) / 2` of each ctrlrange: the head's `[-1, 1]` is the envelope.
fn unnormalizer(g: &Gen, rows: u32, dim: u32) -> Result<LearningNode, SpecError> {
    let range = g.ctrlrange("learning")?;
    // Written as the generator it replaces wrote it, so the bits are the committed ones.
    #[allow(clippy::manual_midpoint)]
    let mean = range.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
    Ok(LearningNode::Normalizer {
        inputs: vec![Port::new(
            "actions",
            policy(Shape::new([u64::from(rows), u64::from(dim)]), signed()),
        )],
        direction: NormalizeDir::Inverse,
        stats: StatsSource::MeanStd {
            mean,
            std: range.iter().map(|(lo, hi)| (hi - lo) / 2.0).collect(),
        },
        out_unit: Unit::Angle,
    })
}

fn handle(
    inputs: &[Port],
    dim: u32,
    horizon: u32,
    execute: u32,
    hz: f32,
    mode: ActionExecutionMode,
    (latency, deadline): (f32, f32),
) -> PolicyHandle {
    PolicyHandle {
        architecture: ArchKind::Act,
        base_model: None,
        weights: WeightsRef::Safetensors {
            path: "policy.safetensors".to_owned(),
            hash: [0; 32],
        },
        contract: PolicyContract {
            inputs: inputs
                .iter()
                .map(|p| (p.name.clone(), p.clone()))
                .collect::<BTreeMap<_, _>>(),
            observation_window: 1,
            action_dim: dim,
            horizon,
            execute_chunk: execute,
            replanning_hz: hz,
            execution_mode: mode,
            runtime: RuntimeHints {
                dtype: ElemType::F32,
                expected_latency_ms: latency,
                deadline_ms: deadline,
            },
        },
    }
}

fn checked(g: LearningGraph) -> Result<LearningGraph, SpecError> {
    let errors: Vec<_> = g
        .validate()
        .into_iter()
        .filter(es_ir::diag::Diagnostic::is_error)
        .collect();
    if errors.is_empty() {
        Ok(g)
    } else {
        refuse("learning", "validate", format!("{errors:?}"))
    }
}

/// The teacher: the `state` output through an ELU MLP `[512, 256] -> 128`, a `tanh` regression
/// head over every actuator, one row per control tick, unnormalized by the ctrlranges. Its
/// deadline is one control period in whole milliseconds.
pub(super) fn teacher(g: &Gen, obs: &ObservationIr) -> Result<LearningGraph, SpecError> {
    let state = input(obs, "state");
    let dim = g.scene.actuators.len() as u32;
    let hz = g.control_hz() as f32;
    let chunk = policy(Shape::new([1, u64::from(dim)]), signed());
    let mut b = Builder {
        nodes: Graph::new(1),
    };
    let enc = b.add(LearningNode::StateEncoder {
        inputs: vec![state.clone()],
        kind: StateEncoderKind::Mlp {
            hidden: vec![512, 256],
            activation: Activation::Elu,
            activate_output: true,
        },
        out_dim: 128,
    });
    let head = b.add(LearningNode::PolicyHead {
        inputs: vec![feature("feat", 128)],
        kind: HeadKind::Regression,
        action_dim: dim,
        horizon: 1,
        squash: Squash::Tanh,
    });
    let chunker = b.add(LearningNode::ActionChunker {
        inputs: vec![Port::new("chunk", chunk.clone())],
        horizon: 1,
        execute_chunk: 1,
        replan_hz: hz,
        mode: ActionExecutionMode::RecedingHorizon,
        blend: ChunkBlendPolicy::HardSwitch,
        buffer_chunks: 2,
    });
    let norm = b.add(unnormalizer(g, 1, dim)?);
    let n = &mut b.nodes;
    n.connect(enc, "out", head, "feat");
    n.connect(head, "chunk", chunker, "chunk");
    n.connect(chunker, "actions", norm, "actions");
    n.inputs.push(PortRef::new(enc, "state"));
    n.outputs.push(PortRef::new(norm, "out"));
    let deadline = (1000.0 / g.control_hz()).floor() as f32;
    checked(LearningGraph {
        schema_version: 1,
        inputs: vec![state.clone()],
        nodes: b.nodes,
        outputs: vec![Port::new(
            "actions",
            PortType {
                unit: Unit::Angle,
                ..chunk
            },
        )],
        policy: handle(
            &[state],
            dim,
            1,
            1,
            hz,
            ActionExecutionMode::RecedingHorizon,
            (2.0, deadline),
        ),
    })
}

/// The student: one `ImageNet` `ResNet18` per view (`mad`: the first view's, shared, the features
/// summed) and an MLP `[256]` per state input into a `Concat`, a one-frame transformer, a
/// regression head over `horizon` rows, a temporal-ensemble chunker executing `execute`; the
/// `h3` preset puts a `tanh` on the head and the ctrlrange unnormalizer after the chunker
/// (H6's invariant).
pub(super) fn student(
    g: &Gen,
    s: &Student,
    obs: &ObservationIr,
    views: &[String],
    state: &[String],
) -> Result<LearningGraph, SpecError> {
    let preset = s.preset.unwrap_or_default();
    let (horizon, execute) = (s.horizon, s.execute);
    let dim = g.scene.actuators.len() as u32;
    let hz = g.control_hz() / f64::from(execute);
    if !(execute >= 1 && execute <= horizon && hz.fract() == 0.0) {
        return refuse(
            "student",
            "execute",
            format!("1 ≤ execute ≤ horizon and control_hz / execute whole (XIR-023); {hz} Hz"),
        );
    }
    let hz = hz as f32;
    let images: Vec<Port> = views
        .iter()
        .map(|v| input(obs, &format!("rgb_{v}")))
        .collect();
    let states: Vec<Port> = match preset {
        Preset::H3 => vec![input(obs, "state")],
        Preset::U3 => state.iter().map(|c| input(obs, c)).collect(),
    };
    let mut b = Builder {
        nodes: Graph::new(1),
    };
    let encoders: Vec<NodeId> = images
        .iter()
        .map(|p| {
            b.add(LearningNode::VisionEncoder {
                inputs: vec![p.clone()],
                backbone: VisionBackbone::ResNet18,
                pretrained: true,
                frozen: false,
                out_dim: 512,
                token_count: 0,
                share: None,
            })
        })
        .collect();
    let state_enc: Vec<NodeId> = states
        .iter()
        .map(|p| {
            b.add(LearningNode::StateEncoder {
                inputs: vec![p.clone()],
                kind: StateEncoderKind::Mlp {
                    hidden: vec![256],
                    activation: Activation::default(),
                    activate_output: false,
                },
                out_dim: 512,
            })
        })
        .collect();
    // The fusion lists the first view, the state, then the other views (the committed order).
    let first = match preset {
        Preset::H3 => views[0].clone(),
        Preset::U3 => "image".to_owned(),
    };
    let mut fused: Vec<(String, NodeId)> = Vec::new();
    if s.family.unwrap_or_default() == Family::Mad {
        let sum = b.add(LearningNode::Fusion {
            inputs: views.iter().map(|v| feature(v, 512)).collect(),
            kind: FusionKind::Sum,
            out_dim: 512,
            token_count: 0,
        });
        for (v, e) in views.iter().zip(&encoders) {
            b.nodes.connect(*e, "out", sum, v);
            if *e != encoders[0] {
                if let Some(LearningNode::VisionEncoder { share, .. }) = b.nodes.nodes.get_mut(e) {
                    *share = Some(encoders[0]);
                }
            }
        }
        fused.push(("image".to_owned(), sum));
        fused.extend(state_enc.iter().map(|e| ("state".to_owned(), *e)));
    } else {
        fused.push((first, encoders[0]));
        fused.extend(state_enc.iter().map(|e| ("state".to_owned(), *e)));
        fused.extend(
            views
                .iter()
                .zip(&encoders)
                .skip(1)
                .map(|(v, e)| (v.clone(), *e)),
        );
    }
    let concat = b.add(LearningNode::Fusion {
        inputs: fused.iter().map(|(p, _)| feature(p, 512)).collect(),
        kind: FusionKind::Concat,
        out_dim: 512,
        token_count: 0,
    });
    for (port, from) in &fused {
        b.nodes.connect(*from, "out", concat, port);
    }
    let temporal = b.add(LearningNode::TemporalEncoder {
        inputs: vec![feature("seq", 512)],
        kind: TemporalKind::Transformer,
        n_frames: 1,
        out_dim: 512,
        token_count: 0,
    });
    let head = b.add(LearningNode::PolicyHead {
        inputs: vec![feature("feat", 512)],
        kind: HeadKind::Regression,
        action_dim: dim,
        horizon,
        squash: match preset {
            Preset::H3 => Squash::Tanh,
            Preset::U3 => Squash::None,
        },
    });
    let rows = |n: u32| Shape::new([u64::from(n), u64::from(dim)]);
    let chunker = b.add(LearningNode::ActionChunker {
        inputs: vec![Port::new("chunk", policy(rows(horizon), signed()))],
        horizon,
        execute_chunk: execute,
        replan_hz: hz,
        mode: ActionExecutionMode::TemporalEnsemble,
        blend: ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
        buffer_chunks: 2,
    });
    b.nodes.connect(concat, "out", temporal, "seq");
    b.nodes.connect(temporal, "out", head, "feat");
    b.nodes.connect(head, "chunk", chunker, "chunk");
    let (out, unit) = match preset {
        Preset::H3 => {
            let norm = b.add(unnormalizer(g, execute, dim)?);
            b.nodes.connect(chunker, "actions", norm, "actions");
            (PortRef::new(norm, "out"), Unit::Angle)
        }
        Preset::U3 => (PortRef::new(chunker, "actions"), signed()),
    };
    b.nodes.outputs.push(out);
    // Inputs in the fusion's order: the first view, the state, the other views.
    let mut inputs = vec![(images[0].clone(), encoders[0])];
    inputs.extend(states.iter().cloned().zip(state_enc));
    inputs.extend(images.iter().cloned().zip(encoders.iter().copied()).skip(1));
    for (p, node) in &inputs {
        b.nodes.inputs.push(PortRef::new(*node, &p.name));
    }
    let ports: Vec<Port> = inputs.into_iter().map(|(p, _)| p).collect();
    checked(LearningGraph {
        schema_version: 1,
        policy: handle(
            &ports,
            dim,
            horizon,
            execute,
            hz,
            ActionExecutionMode::TemporalEnsemble,
            (15.0, 40.0),
        ),
        inputs: ports,
        nodes: b.nodes,
        outputs: vec![Port::new("actions", policy(rows(execute), unit))],
    })
}
