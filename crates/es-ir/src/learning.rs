//! Learning IR: tensor to action chunk — encoders, fusion, temporal model, policy head and
//! action decoding (spec 8). Batch semantics are free here (spec 5.2): no shape in this module
//! carries a batch axis.
//!
//! The rule of spec 8.1 is "the network is opaque, its interface is typed". Everything the
//! compiler can check lives in [`PolicyContract`] (spec 8.4); the weights themselves are a
//! reference ([`WeightsRef`]), never inline tensors, and the reference has no pickle variant,
//! so `INV-16` is structural rather than a review rule.
//!
//! Not here: `ActionSpec`, `ActionSpace` and `ActionLimits` of spec 8.5 — those are declared by
//! Task IR and consumed by the Safety Plane (spec 9); the Learning IR only needs the execution
//! mode and the chunk-transition data, which are [`ActionExecutionMode`] and
//! [`ChunkBlendPolicy`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::codes;
use crate::diag::Diagnostic;
use crate::graph::{Graph, IrNode, Port};
use crate::hash::{canonical_hash, CanonWriter};
use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

/// Domain separators for the two hashes of spec 5.3.
const LEARNING_TAG: &str = "es.learning_hash.v1";
const POLICY_TAG: &str = "es.policy_hash.v1";

/// One named tensor port. Spec 8.2 calls this `TensorPort`; it is exactly [`Port`].
pub type TensorPort = Port;

// --- Node parameter enums (spec 8.3) ------------------------------------------------------

/// `Custom` carries the blake3 digest of the backbone definition, so an unnamed backbone is
/// still hashable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisionBackbone {
    ResNet18,
    ResNet34,
    ViT { size: String },
    DinoV2,
    SigLip,
    SmolVlm,
    Custom { hash: [u8; 32] },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateEncoderKind {
    Identity,
    Mlp { hidden: Vec<u32> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionKind {
    Concat,
    CrossAttention,
    FiLm,
    AdaLn,
    TokenConcat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalKind {
    None,
    TemporalConv,
    Transformer,
    Gru,
    Mamba,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffusionScheduler {
    Ddpm,
    Ddim,
    DpmSolver,
}

/// `diffusers`' `beta_schedule`. `LeRobot`'s Diffusion Policy default is `squaredcos_cap_v2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetaSchedule {
    Linear,
    #[default]
    SquaredcosCapV2,
}

/// `diffusers`' `variance_type`. Its (and `LeRobot`'s) default is `fixed_small`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceType {
    #[default]
    FixedSmall,
    FixedLarge,
}

/// `diffusers`' `prediction_type`: what the denoiser's output means.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionType {
    #[default]
    Epsilon,
    Sample,
    VPrediction,
}

fn default_train_timesteps() -> u32 {
    100
}

fn default_true() -> bool {
    true
}

fn default_clip_range() -> f32 {
    1.0
}

/// Spec 8.3. The head is the only node that turns features into an action chunk.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HeadKind {
    /// ACT, plain behaviour cloning.
    Regression,
    /// Diffusion Policy. The fields after `scheduler` are `diffusers`' `DDPMScheduler` /
    /// `DDIMScheduler` configuration, which a checkpoint was trained under and without which it
    /// cannot be reproduced; the `serde` defaults are `LeRobot`'s Diffusion Policy defaults
    /// (`lerobot/common/policies/diffusion/configuration_diffusion.py`; `Status: unverified`,
    /// recalled rather than fetched — see `docs/api-notes/lerobot-config.md`). `n_steps` stays
    /// the number of *inference* steps, subsampled out of `num_train_timesteps`.
    Diffusion {
        n_steps: u32,
        scheduler: DiffusionScheduler,
        #[serde(default = "default_train_timesteps")]
        num_train_timesteps: u32,
        #[serde(default)]
        beta_schedule: BetaSchedule,
        #[serde(default)]
        variance_type: VarianceType,
        #[serde(default)]
        prediction_type: PredictionType,
        #[serde(default = "default_true")]
        clip_sample: bool,
        #[serde(default = "default_clip_range")]
        clip_sample_range: f32,
    },
    /// pi0, `SmolVLA`.
    FlowMatching { n_steps: u32 },
    /// VQ-BET, RT-2.
    Discrete { vocab: u32 },
    /// IBC.
    Energy { n_samples: u32 },
}

/// Where the (un)normalization statistics come from (spec 8.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StatsSource {
    /// The dataset the policy was trained on, by `dataset_hash` (spec 5.3).
    Dataset {
        dataset_hash: [u8; 32],
    },
    MeanStd {
        mean: Vec<f64>,
        std: Vec<f64>,
    },
    MinMax {
        lo: Vec<f64>,
        hi: Vec<f64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizeDir {
    /// Into the policy's normalized space.
    Forward,
    /// Out of it — spec 8.3 calls this node `ActionUnnormalizer`.
    Inverse,
}

/// Spec 8.5. `RecedingHorizon` is the default; `TemporalEnsemble` is what ACT actually does,
/// so it is first class rather than a runtime flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionExecutionMode {
    OpenLoopChunk,
    RecedingHorizon,
    TemporalEnsemble,
    RealTimeChunking,
}

/// Spec 8.6: how a chunk arriving late replaces the one being executed. Data only — the
/// buffer itself lives in the runtime, not in the IR.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ChunkBlendPolicy {
    /// Immediate replacement; may be discontinuous.
    HardSwitch,
    LinearBlend {
        steps: u32,
    },
    /// Exponentially weighted average over the overlap, ACT style.
    TemporalEnsemble {
        weight_decay: f32,
    },
}

// --- Nodes --------------------------------------------------------------------------------

/// The node set of spec 8.3.
///
/// Every consuming node declares its input ports explicitly (`inputs`), because what an encoder
/// accepts is the Observation IR's business, not a constant of the encoder. Output ports are
/// derived from the parameters, so a shape cannot drift away from the node that produces it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LearningNode {
    VisionEncoder {
        inputs: Vec<TensorPort>,
        backbone: VisionBackbone,
        pretrained: bool,
        frozen: bool,
        out_dim: u32,
        /// `0` for a pooled feature vector, `n` for `n` tokens.
        token_count: u32,
    },
    StateEncoder {
        inputs: Vec<TensorPort>,
        kind: StateEncoderKind,
        out_dim: u32,
    },
    LanguageEncoder {
        inputs: Vec<TensorPort>,
        tokenizer: String,
        model: String,
        max_len: u32,
        out_dim: u32,
        token_count: u32,
    },
    Fusion {
        inputs: Vec<TensorPort>,
        kind: FusionKind,
        out_dim: u32,
        token_count: u32,
    },
    /// The third layer of the spec 7.5 time model: the network's own temporal mixing.
    TemporalEncoder {
        inputs: Vec<TensorPort>,
        kind: TemporalKind,
        /// Must equal `PolicyContract::observation_window`.
        n_frames: u32,
        out_dim: u32,
        token_count: u32,
    },
    PolicyHead {
        inputs: Vec<TensorPort>,
        kind: HeadKind,
        action_dim: u32,
        horizon: u32,
    },
    /// A large VLA referenced whole (spec 8.3). pi0 is not decomposed into nodes; only its
    /// interface is type-checked.
    PolicyBundle {
        inputs: Vec<TensorPort>,
        weights: WeightsRef,
        action_dim: u32,
        horizon: u32,
    },
    ActionChunker {
        inputs: Vec<TensorPort>,
        horizon: u32,
        execute_chunk: u32,
        replan_hz: f32,
        mode: ActionExecutionMode,
        blend: ChunkBlendPolicy,
        /// Chunks the async buffer of spec 8.6 holds. Underrun is a safety event, never a
        /// tunable, so it is not represented here.
        buffer_chunks: u32,
    },
    Normalizer {
        inputs: Vec<TensorPort>,
        direction: NormalizeDir,
        stats: StatsSource,
        /// The unit the node produces.
        out_unit: Unit,
    },
}

fn policy_ty(shape: Shape, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape,
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

/// A feature port: `[dim]` when `tokens == 0`, `[tokens, dim]` otherwise.
fn feature(name: &str, dim: u32, tokens: u32) -> TensorPort {
    let shape = if tokens == 0 {
        Shape::new([u64::from(dim)])
    } else {
        Shape::new([u64::from(tokens), u64::from(dim)])
    };
    TensorPort::new(name, policy_ty(shape, Unit::Dimensionless))
}

/// The space an action chunk lives in before unnormalization (spec 8.4).
pub fn action_unit() -> Unit {
    Unit::Normalized { lo: -1.0, hi: 1.0 }
}

fn chunk(name: &str, horizon: u32, action_dim: u32) -> TensorPort {
    TensorPort::new(
        name,
        policy_ty(
            Shape::new([u64::from(horizon), u64::from(action_dim)]),
            action_unit(),
        ),
    )
}

fn canon_ports(w: &mut CanonWriter, ports: &[TensorPort]) {
    w.seq(ports.len());
    for p in ports {
        w.str(&p.name);
        p.ty.canonical(w);
    }
}

/// `Unit` has no public canonical encoder; its `Debug` is derived, total and round-trips
/// floats, so it is a sound hash input.
fn canon_unit(w: &mut CanonWriter, unit: &Unit) {
    w.str(&format!("{unit:?}"));
}

impl LearningNode {
    fn input_ports(&self) -> &[TensorPort] {
        match self {
            Self::VisionEncoder { inputs, .. }
            | Self::StateEncoder { inputs, .. }
            | Self::LanguageEncoder { inputs, .. }
            | Self::Fusion { inputs, .. }
            | Self::TemporalEncoder { inputs, .. }
            | Self::PolicyHead { inputs, .. }
            | Self::PolicyBundle { inputs, .. }
            | Self::ActionChunker { inputs, .. }
            | Self::Normalizer { inputs, .. } => inputs,
        }
    }

    /// Inclusive `(min, max)` input count; `None` for "any number".
    fn arity(&self) -> (usize, Option<usize>) {
        match self {
            Self::Fusion { .. } => (2, None),
            Self::PolicyHead { .. } | Self::PolicyBundle { .. } => (1, None),
            _ => (1, Some(1)),
        }
    }

    /// The last dimension of the first input, which is the action width of the action nodes.
    fn trailing_dim(&self) -> u64 {
        self.input_ports()
            .first()
            .and_then(|p| p.ty.shape.dims().last().copied())
            .unwrap_or(0)
    }
}

impl IrNode for LearningNode {
    fn kind(&self) -> &'static str {
        match self {
            Self::VisionEncoder { .. } => "VisionEncoder",
            Self::StateEncoder { .. } => "StateEncoder",
            Self::LanguageEncoder { .. } => "LanguageEncoder",
            Self::Fusion { .. } => "Fusion",
            Self::TemporalEncoder { .. } => "TemporalEncoder",
            Self::PolicyHead { .. } => "PolicyHead",
            Self::PolicyBundle { .. } => "PolicyBundle",
            Self::ActionChunker { .. } => "ActionChunker",
            Self::Normalizer { .. } => "Normalizer",
        }
    }

    fn inputs(&self) -> Vec<Port> {
        self.input_ports().to_vec()
    }

    fn outputs(&self) -> Vec<Port> {
        match self {
            Self::VisionEncoder {
                out_dim,
                token_count,
                ..
            }
            | Self::LanguageEncoder {
                out_dim,
                token_count,
                ..
            }
            | Self::Fusion {
                out_dim,
                token_count,
                ..
            }
            | Self::TemporalEncoder {
                out_dim,
                token_count,
                ..
            } => vec![feature("out", *out_dim, *token_count)],
            Self::StateEncoder { out_dim, .. } => vec![feature("out", *out_dim, 0)],
            Self::PolicyHead {
                action_dim,
                horizon,
                ..
            }
            | Self::PolicyBundle {
                action_dim,
                horizon,
                ..
            } => vec![chunk("chunk", *horizon, *action_dim)],
            Self::ActionChunker { execute_chunk, .. } => {
                vec![chunk("actions", *execute_chunk, self.trailing_dim() as u32)]
            }
            Self::Normalizer {
                inputs, out_unit, ..
            } => match inputs.first() {
                None => Vec::new(),
                Some(p) => vec![TensorPort::new(
                    "out",
                    policy_ty(p.ty.shape.clone(), out_unit.clone()),
                )],
            },
        }
    }

    fn params_canonical(&self, w: &mut CanonWriter) {
        canon_ports(w, self.input_ports());
        match self {
            Self::VisionEncoder {
                backbone,
                pretrained,
                frozen,
                out_dim,
                token_count,
                ..
            } => {
                w.str(&format!("{backbone:?}"));
                w.bool(*pretrained);
                w.bool(*frozen);
                w.u32(*out_dim);
                w.u32(*token_count);
            }
            Self::StateEncoder { kind, out_dim, .. } => {
                w.str(&format!("{kind:?}"));
                w.u32(*out_dim);
            }
            Self::LanguageEncoder {
                tokenizer,
                model,
                max_len,
                out_dim,
                token_count,
                ..
            } => {
                w.str(tokenizer);
                w.str(model);
                w.u32(*max_len);
                w.u32(*out_dim);
                w.u32(*token_count);
            }
            Self::Fusion {
                kind,
                out_dim,
                token_count,
                ..
            } => {
                w.str(&format!("{kind:?}"));
                w.u32(*out_dim);
                w.u32(*token_count);
            }
            Self::TemporalEncoder {
                kind,
                n_frames,
                out_dim,
                token_count,
                ..
            } => {
                w.str(&format!("{kind:?}"));
                w.u32(*n_frames);
                w.u32(*out_dim);
                w.u32(*token_count);
            }
            Self::PolicyHead {
                kind,
                action_dim,
                horizon,
                ..
            } => {
                w.str(&format!("{kind:?}"));
                w.u32(*action_dim);
                w.u32(*horizon);
            }
            Self::PolicyBundle {
                weights,
                action_dim,
                horizon,
                ..
            } => {
                weights.canonical(w);
                w.u32(*action_dim);
                w.u32(*horizon);
            }
            Self::ActionChunker {
                horizon,
                execute_chunk,
                replan_hz,
                mode,
                blend,
                buffer_chunks,
                ..
            } => {
                w.u32(*horizon);
                w.u32(*execute_chunk);
                w.f32(*replan_hz);
                w.str(&format!("{mode:?}"));
                w.str(&format!("{blend:?}"));
                w.u32(*buffer_chunks);
            }
            Self::Normalizer {
                direction,
                stats,
                out_unit,
                ..
            } => {
                w.str(&format!("{direction:?}"));
                match stats {
                    StatsSource::Dataset { dataset_hash } => {
                        w.str("Dataset");
                        w.digest(dataset_hash);
                    }
                    StatsSource::MeanStd { mean: a, std: b }
                    | StatsSource::MinMax { lo: a, hi: b } => {
                        w.str(if matches!(stats, StatsSource::MeanStd { .. }) {
                            "MeanStd"
                        } else {
                            "MinMax"
                        });
                        w.seq(a.len());
                        for v in a {
                            w.f64(*v);
                        }
                        w.seq(b.len());
                        for v in b {
                            w.f64(*v);
                        }
                    }
                }
                canon_unit(w, out_unit);
            }
        }
    }
}

// --- Policy handle (spec 8.4, Appendix B.3) -----------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchKind {
    Act,
    Diffusion,
    FlowMatching,
    Discrete,
    Bundle,
}

/// A pretrained checkpoint the weights were derived from, e.g. `"lerobot/pi05_base"`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseModelRef {
    pub uri: String,
    pub hash: [u8; 32],
    pub license: String,
}

/// Where the weights live and in what format.
///
/// `INV-16`: there is no pickle variant and there never will be one — a format that executes
/// code on load has no business in a deployment artefact. Adding one is an API break, which is
/// the point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsRef {
    Safetensors { path: String, hash: [u8; 32] },
    Onnx { path: String, hash: [u8; 32] },
    SpirV { path: String, hash: [u8; 32] },
}

impl WeightsRef {
    pub fn hash(&self) -> &[u8; 32] {
        match self {
            Self::Safetensors { hash, .. } | Self::Onnx { hash, .. } | Self::SpirV { hash, .. } => {
                hash
            }
        }
    }

    pub fn path(&self) -> &str {
        match self {
            Self::Safetensors { path, .. } | Self::Onnx { path, .. } | Self::SpirV { path, .. } => {
                path
            }
        }
    }

    fn format(&self) -> &'static str {
        match self {
            Self::Safetensors { .. } => "Safetensors",
            Self::Onnx { .. } => "Onnx",
            Self::SpirV { .. } => "SpirV",
        }
    }

    fn canonical(&self, w: &mut CanonWriter) {
        w.str(self.format());
        w.str(self.path());
        w.digest(self.hash());
    }
}

/// `dtype`, `expected_latency_ms`, `deadline_ms` of spec 8.4. Latency is a measurement on the
/// reference device of spec 0.3, not a promise.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeHints {
    pub dtype: ElemType,
    pub expected_latency_ms: f32,
    /// Exceeding this triggers the spec 9.4 fallback.
    pub deadline_ms: f32,
}

/// The metadata contract of spec 8.4 — what the compiler checks about an opaque policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyContract {
    pub inputs: BTreeMap<String, TensorPort>,
    pub observation_window: u32,
    pub action_dim: u32,
    /// Prediction length `H`.
    pub horizon: u32,
    /// Execution length `K ≤ H`.
    pub execute_chunk: u32,
    pub replanning_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub runtime: RuntimeHints,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyHandle {
    pub architecture: ArchKind,
    pub base_model: Option<BaseModelRef>,
    pub weights: WeightsRef,
    pub contract: PolicyContract,
}

// --- Graph --------------------------------------------------------------------------------

/// Appendix B.3. `inputs` and `outputs` are the contract with the Observation IR and with
/// `ActionSpec`; the graph's own boundary (`nodes.inputs` / `nodes.outputs`) says which node
/// port each one lands on, positionally.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,
    pub nodes: Graph<LearningNode>,
    pub outputs: Vec<TensorPort>,
    pub policy: PolicyHandle,
}

impl LearningGraph {
    /// Every check of spec 8.4 that this IR can answer on its own. Cross-IR checks (dataset
    /// shapes, `dt_ctrl`, the Safety Plane fallback) belong to P26.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut diags = self.nodes.validate_declared_ports();
        if let Err(d) = self.nodes.topo_order() {
            diags.push(d);
        }
        if self.schema_version != self.nodes.schema_version {
            diags.push(Diagnostic::new(
                codes::LRN_001,
                format!(
                    "wrapper says schema {} but the graph says {}",
                    self.schema_version, self.nodes.schema_version
                ),
            ));
        }
        self.check_arity(&mut diags);
        self.check_boundary(&mut diags);
        self.check_contract(&mut diags);
        self.check_normalizers(&mut diags);
        diags
    }

    fn check_arity(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let (min, max) = node.arity();
            let n = node.input_ports().len();
            if n < min || max.is_some_and(|m| n > m) {
                diags.push(
                    Diagnostic::new(
                        codes::LRN_002,
                        format!("{} declares {n} input ports, expected {min}..", node.kind()),
                    )
                    .at(*id),
                );
            }
        }
    }

    /// The declared tensor ports, the graph boundary and `PolicyContract::inputs` must agree.
    /// Boundary inputs feed the network, so spec 5.4's policy-input rule applies to each.
    fn check_boundary(&self, diags: &mut Vec<Diagnostic>) {
        for (declared, refs, side) in [
            (&self.inputs, &self.nodes.inputs, "input"),
            (&self.outputs, &self.nodes.outputs, "output"),
        ] {
            if declared.len() != refs.len() {
                diags.push(Diagnostic::new(
                    codes::LRN_010,
                    format!(
                        "{} declared {side} ports but the graph boundary has {}",
                        declared.len(),
                        refs.len()
                    ),
                ));
                continue;
            }
            for (port, at) in declared.iter().zip(refs) {
                // A missing node is already reported as GRAPH-002.
                let Some(node) = self.nodes.nodes.get(&at.node) else {
                    continue;
                };
                let ports = if side == "input" {
                    node.inputs()
                } else {
                    node.outputs()
                };
                let Some(found) = ports.into_iter().find(|p| p.name == at.port) else {
                    continue; // already reported as GRAPH-010
                };
                if let Err(d) = port.ty.compatible(&found.ty) {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_010,
                            format!(
                                "{side} \"{}\" does not match the node: {}",
                                port.name, d.message
                            ),
                        )
                        .at(at.node)
                        .on_port(at.port.clone()),
                    );
                }
            }
        }

        for port in &self.inputs {
            if let Err(d) = port.ty.check_policy_input() {
                diags.push(d.on_port(port.name.clone()));
            }
            match self.policy.contract.inputs.get(&port.name) {
                None => diags.push(
                    Diagnostic::new(
                        codes::LRN_011,
                        format!("PolicyContract has no input named \"{}\"", port.name),
                    )
                    .on_port(port.name.clone()),
                ),
                Some(declared) => {
                    if let Err(d) = port.ty.compatible(&declared.ty) {
                        diags.push(
                            Diagnostic::new(
                                codes::LRN_011,
                                format!("contract input \"{}\": {}", port.name, d.message),
                            )
                            .on_port(port.name.clone()),
                        );
                    }
                }
            }
        }
    }

    fn check_contract(&self, diags: &mut Vec<Diagnostic>) {
        let c = &self.policy.contract;
        for (name, v) in [
            ("action_dim", c.action_dim),
            ("horizon", c.horizon),
            ("execute_chunk", c.execute_chunk),
            ("observation_window", c.observation_window),
        ] {
            if v == 0 {
                diags.push(Diagnostic::new(
                    codes::LRN_023,
                    format!("{name} must be positive"),
                ));
            }
        }
        if c.replanning_hz <= 0.0 || !c.replanning_hz.is_finite() {
            diags.push(Diagnostic::new(
                codes::LRN_023,
                format!(
                    "replanning_hz must be positive and finite, got {}",
                    c.replanning_hz
                ),
            ));
        }
        if c.execute_chunk > c.horizon {
            diags.push(
                Diagnostic::new(
                    codes::LRN_020,
                    format!(
                        "execute_chunk {} exceeds horizon {}",
                        c.execute_chunk, c.horizon
                    ),
                )
                .with_hint("K must be at most H (spec 8.5)"),
            );
        }

        let head = self.nodes.nodes.values().find_map(|n| match n {
            LearningNode::PolicyHead {
                action_dim,
                horizon,
                ..
            }
            | LearningNode::PolicyBundle {
                action_dim,
                horizon,
                ..
            } => Some((*action_dim, *horizon)),
            _ => None,
        });
        match head {
            None => diags.push(
                Diagnostic::new(codes::LRN_021, "the graph has no policy head")
                    .with_hint("add a PolicyHead node, or reference a whole VLA with PolicyBundle"),
            ),
            Some((action_dim, horizon)) => {
                if action_dim != c.action_dim {
                    diags.push(Diagnostic::new(
                        codes::LRN_021,
                        format!(
                            "contract action_dim {} but the head produces {action_dim}",
                            c.action_dim
                        ),
                    ));
                }
                if horizon != c.horizon {
                    diags.push(Diagnostic::new(
                        codes::LRN_021,
                        format!(
                            "contract horizon {} but the head produces {horizon}",
                            c.horizon
                        ),
                    ));
                }
            }
        }

        // The chunker, when present, must agree with the contract it implements.
        for (id, node) in &self.nodes.nodes {
            if let LearningNode::ActionChunker {
                horizon,
                execute_chunk,
                ..
            } = node
            {
                if *execute_chunk > *horizon {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_020,
                            format!("execute_chunk {execute_chunk} exceeds horizon {horizon}"),
                        )
                        .at(*id),
                    );
                }
                if (*horizon, *execute_chunk) != (c.horizon, c.execute_chunk) {
                    diags.push(
                        Diagnostic::new(
                            codes::LRN_021,
                            format!(
                                "chunker is ({horizon}, {execute_chunk}) but the contract is ({}, {})",
                                c.horizon, c.execute_chunk
                            ),
                        )
                        .at(*id),
                    );
                }
            }
        }

        let window = self
            .nodes
            .nodes
            .values()
            .find_map(|n| match n {
                LearningNode::TemporalEncoder { n_frames, .. } => Some(*n_frames),
                _ => None,
            })
            .unwrap_or(1);
        if window != c.observation_window {
            diags.push(
                Diagnostic::new(
                    codes::LRN_022,
                    format!(
                        "observation_window {} but the temporal node sees {window} frames",
                        c.observation_window
                    ),
                )
                .with_hint("the TemporalEncoder is layer 3 of the spec 7.5 time model"),
            );
        }

        // Spec 8.4: inference must fit inside the deadline and inside the replanning period.
        let replan_ms = 1000.0 / c.replanning_hz;
        let budget = c.runtime.deadline_ms.min(replan_ms);
        if c.runtime.expected_latency_ms > budget {
            diags.push(
                Diagnostic::new(
                    codes::LRN_052,
                    format!(
                        "measured latency {:.1} ms, budget {budget:.1} ms (deadline {:.1} ms, replanning {:.1} Hz)",
                        c.runtime.expected_latency_ms, c.runtime.deadline_ms, c.replanning_hz
                    ),
                )
                .with_hint(
                    "async inference with chunk buffering (spec 8.6), a Safety Plane fallback (spec 9.4), or a faster policy",
                ),
            );
        }
    }

    fn check_normalizers(&self, diags: &mut Vec<Diagnostic>) {
        for (id, node) in &self.nodes.nodes {
            let LearningNode::Normalizer {
                direction,
                out_unit,
                ..
            } = node
            else {
                continue;
            };
            if (*direction == NormalizeDir::Forward) != out_unit.is_policy_input() {
                diags.push(
                    Diagnostic::new(
                        codes::LRN_030,
                        format!("{direction:?} normalizer produces {out_unit:?}"),
                    )
                    .at(*id)
                    .with_hint(
                        "Forward produces Normalized/Dimensionless/Token; Inverse leaves that space",
                    ),
                );
            }
        }
    }

    /// `learning_hash` (spec 5.3): the architecture only. Independent of node ids and node
    /// order, and independent of the weights — changing a checkpoint must not change it.
    pub fn learning_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let graph = canonical_hash(&self.nodes)?;
        let c = &self.policy.contract;
        let mut w = CanonWriter::new();
        w.str(LEARNING_TAG);
        w.u32(self.schema_version);
        w.digest(&graph);
        canon_ports(&mut w, &self.inputs);
        canon_ports(&mut w, &self.outputs);
        w.str(&format!("{:?}", self.policy.architecture));
        w.seq(c.inputs.len());
        for (name, port) in &c.inputs {
            w.str(name);
            w.str(&port.name);
            port.ty.canonical(&mut w);
        }
        for v in [
            c.observation_window,
            c.action_dim,
            c.horizon,
            c.execute_chunk,
        ] {
            w.u32(v);
        }
        w.f32(c.replanning_hz);
        w.str(&format!("{:?}", c.execution_mode));
        w.str(&format!("{:?}", c.runtime.dtype));
        w.f32(c.runtime.expected_latency_ms);
        w.f32(c.runtime.deadline_ms);
        w.hash()
    }

    /// `policy_hash = H(weights, learning_hash, base_model)` (spec 5.3).
    pub fn policy_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let learning = self.learning_hash()?;
        let mut w = CanonWriter::new();
        w.str(POLICY_TAG);
        self.policy.weights.canonical(&mut w);
        w.digest(&learning);
        match &self.policy.base_model {
            None => w.bool(false),
            Some(m) => {
                w.bool(true);
                w.str(&m.uri);
                w.digest(&m.hash);
                w.str(&m.license);
            }
        }
        w.hash()
    }
}

// --- Test support -------------------------------------------------------------------------

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    //! Fixtures and the proptest generator for the Appendix B.7 properties (P25).

    // `use super::*` is how a test-support module reads; clippy only exempts `#[cfg(test)]`
    // ones automatically, and this one is also reachable through `feature = "testing"`.
    #![allow(clippy::wildcard_imports)]

    use super::*;
    use crate::graph::{NodeId, PortRef};
    use proptest::prelude::*;

    /// An ACT-shaped graph: image + state encoders, fusion, temporal encoder, regression head,
    /// chunker. Every generated graph validates clean, so it is usable as a "valid input"
    /// generator for downstream passes.
    pub fn act_like(
        state_dim: u32,
        feat: u32,
        action_dim: u32,
        horizon: u32,
        execute_chunk: u32,
        n_frames: u32,
    ) -> LearningGraph {
        let rgb = TensorPort::new(
            "rgb_front",
            policy_ty(
                Shape::new([3, 224, 224]),
                Unit::Normalized { lo: 0.0, hi: 1.0 },
            ),
        );
        let state = TensorPort::new(
            "joint_state",
            policy_ty(Shape::new([u64::from(state_dim)]), action_unit()),
        );

        let mut g = Graph::new(1);
        g.insert(
            NodeId(0),
            LearningNode::VisionEncoder {
                inputs: vec![rgb.clone()],
                backbone: VisionBackbone::ResNet18,
                pretrained: true,
                frozen: false,
                out_dim: feat,
                token_count: 0,
            },
        );
        g.insert(
            NodeId(1),
            LearningNode::StateEncoder {
                inputs: vec![state.clone()],
                kind: StateEncoderKind::Mlp { hidden: vec![256] },
                out_dim: feat,
            },
        );
        g.insert(
            NodeId(2),
            LearningNode::Fusion {
                inputs: vec![feature("image", feat, 0), feature("state", feat, 0)],
                kind: FusionKind::Concat,
                out_dim: feat,
                token_count: 0,
            },
        );
        g.insert(
            NodeId(3),
            LearningNode::TemporalEncoder {
                inputs: vec![feature("seq", feat, 0)],
                kind: TemporalKind::Transformer,
                n_frames,
                out_dim: feat,
                token_count: 0,
            },
        );
        g.insert(
            NodeId(4),
            LearningNode::PolicyHead {
                inputs: vec![feature("feat", feat, 0)],
                kind: HeadKind::Regression,
                action_dim,
                horizon,
            },
        );
        g.insert(
            NodeId(5),
            LearningNode::ActionChunker {
                inputs: vec![chunk("chunk", horizon, action_dim)],
                horizon,
                execute_chunk,
                replan_hz: 10.0,
                mode: ActionExecutionMode::TemporalEnsemble,
                blend: ChunkBlendPolicy::TemporalEnsemble { weight_decay: 0.01 },
                buffer_chunks: 2,
            },
        );
        g.connect(NodeId(0), "out", NodeId(2), "image");
        g.connect(NodeId(1), "out", NodeId(2), "state");
        g.connect(NodeId(2), "out", NodeId(3), "seq");
        g.connect(NodeId(3), "out", NodeId(4), "feat");
        g.connect(NodeId(4), "chunk", NodeId(5), "chunk");
        g.inputs = vec![
            PortRef::new(NodeId(0), "rgb_front"),
            PortRef::new(NodeId(1), "joint_state"),
        ];
        g.outputs = vec![PortRef::new(NodeId(5), "actions")];

        let contract = PolicyContract {
            inputs: [
                (rgb.name.clone(), rgb.clone()),
                (state.name.clone(), state.clone()),
            ]
            .into_iter()
            .collect(),
            observation_window: n_frames,
            action_dim,
            horizon,
            execute_chunk,
            replanning_hz: 10.0,
            execution_mode: ActionExecutionMode::TemporalEnsemble,
            runtime: RuntimeHints {
                dtype: ElemType::F32,
                expected_latency_ms: 5.0,
                deadline_ms: 50.0,
            },
        };
        LearningGraph {
            schema_version: 1,
            inputs: vec![rgb, state],
            outputs: vec![chunk("actions", execute_chunk, action_dim)],
            nodes: g,
            policy: PolicyHandle {
                architecture: ArchKind::Act,
                base_model: None,
                weights: WeightsRef::Safetensors {
                    path: "policy.safetensors".to_owned(),
                    hash: [7u8; 32],
                },
                contract,
            },
        }
    }

    pub fn arbitrary_learning_graph() -> impl Strategy<Value = LearningGraph> {
        (1u32..16, 1u32..512, 1u32..32, 1u32..64, 1u32..4).prop_flat_map(
            |(state_dim, feat, action_dim, horizon, n_frames)| {
                (1..=horizon).prop_map(move |execute_chunk| {
                    act_like(
                        state_dim,
                        feat,
                        action_dim,
                        horizon,
                        execute_chunk,
                        n_frames,
                    )
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{act_like, arbitrary_learning_graph};
    use super::*;
    use crate::graph::{NodeId, PortRef};
    use proptest::prelude::*;

    fn act() -> LearningGraph {
        act_like(8, 512, 8, 50, 20, 1)
    }

    fn errors(g: &LearningGraph) -> Vec<String> {
        g.validate()
            .into_iter()
            .filter(Diagnostic::is_error)
            .map(|d| format!("{} {}", d.code, d.message))
            .collect()
    }

    fn has(g: &LearningGraph, code: &str) -> bool {
        g.validate().iter().any(|d| d.code.as_str() == code)
    }

    #[test]
    fn act_fixture_validates_clean() {
        let g = act();
        assert!(errors(&g).is_empty(), "{:#?}", errors(&g));
    }

    #[test]
    fn serde_round_trip() {
        let g = act();
        let json = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<LearningGraph>(&json).unwrap(), g);
    }

    #[test]
    fn raw_unit_into_the_policy_is_type_011() {
        let mut g = act();
        g.inputs[1].ty.unit = Unit::Length;
        assert!(has(&g, codes::TYPE_011), "{:#?}", errors(&g));
    }

    #[test]
    fn execute_chunk_above_horizon_is_lrn_020() {
        let mut g = act();
        g.policy.contract.execute_chunk = 80;
        assert!(has(&g, codes::LRN_020), "{:#?}", errors(&g));
    }

    #[test]
    fn contract_disagreeing_with_the_head_is_lrn_021() {
        let mut g = act();
        g.policy.contract.action_dim = 9;
        assert!(has(&g, codes::LRN_021), "{:#?}", errors(&g));
    }

    #[test]
    fn observation_window_must_match_the_temporal_node() {
        let mut g = act();
        g.policy.contract.observation_window = 2;
        assert!(has(&g, codes::LRN_022), "{:#?}", errors(&g));
    }

    #[test]
    fn slow_policy_inside_a_fast_loop_is_lrn_052() {
        let mut g = act();
        // Diffusion Policy on an RTX 4090 at 10 Hz control (spec 8.4).
        g.policy.contract.runtime.expected_latency_ms = 369.8;
        assert!(has(&g, codes::LRN_052), "{:#?}", errors(&g));
    }

    #[test]
    fn inverse_normalizer_must_leave_the_normalized_space() {
        let mut g = act();
        g.nodes.insert(
            NodeId(9),
            LearningNode::Normalizer {
                inputs: vec![chunk("x", 20, 8)],
                direction: NormalizeDir::Inverse,
                stats: StatsSource::Dataset {
                    dataset_hash: [1u8; 32],
                },
                out_unit: Unit::Dimensionless,
            },
        );
        assert!(has(&g, codes::LRN_030), "{:#?}", errors(&g));
    }

    #[test]
    fn wrong_input_count_is_lrn_002() {
        let mut g = act();
        let LearningNode::Fusion { inputs, .. } = g.nodes.nodes.get_mut(&NodeId(2)).unwrap() else {
            unreachable!()
        };
        inputs.truncate(1);
        assert!(has(&g, codes::LRN_002), "{:#?}", errors(&g));
    }

    #[test]
    fn learning_hash_ignores_node_labels() {
        let a = act();
        let mut b = a.clone();
        // Relabel every node by +100, keeping the same structure.
        let bump = |id: NodeId| NodeId(id.0 + 100);
        b.nodes.nodes = a
            .nodes
            .nodes
            .iter()
            .map(|(k, v)| (bump(*k), v.clone()))
            .collect();
        for e in &mut b.nodes.edges {
            e.from.node = bump(e.from.node);
            e.to.node = bump(e.to.node);
        }
        for side in [&mut b.nodes.inputs, &mut b.nodes.outputs] {
            for p in side.iter_mut() {
                *p = PortRef::new(bump(p.node), p.port.clone());
            }
        }
        assert_ne!(a.nodes.nodes, b.nodes.nodes);
        assert_eq!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
        assert!(errors(&b).is_empty(), "{:#?}", errors(&b));
    }

    #[test]
    fn learning_hash_sees_architecture_changes() {
        let a = act();
        let b = act_like(8, 256, 8, 50, 20, 1);
        assert_ne!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
    }

    #[test]
    fn new_weights_move_only_the_policy_hash() {
        let a = act();
        let mut b = a.clone();
        b.policy.weights = WeightsRef::Safetensors {
            path: "policy.safetensors".to_owned(),
            hash: [9u8; 32],
        };
        assert_eq!(a.learning_hash().unwrap(), b.learning_hash().unwrap());
        assert_ne!(a.policy_hash().unwrap(), b.policy_hash().unwrap());

        // A base model is part of the policy identity too (spec 5.3).
        let mut c = a.clone();
        c.policy.base_model = Some(BaseModelRef {
            uri: "lerobot/pi05_base".to_owned(),
            hash: [3u8; 32],
            license: "apache-2.0".to_owned(),
        });
        assert_eq!(a.learning_hash().unwrap(), c.learning_hash().unwrap());
        assert_ne!(a.policy_hash().unwrap(), c.policy_hash().unwrap());
    }

    #[test]
    fn weights_ref_has_no_executable_format() {
        // INV-16 is structural: these are the only three constructors there are.
        for w in [
            WeightsRef::Safetensors {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
            WeightsRef::Onnx {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
            WeightsRef::SpirV {
                path: "a".to_owned(),
                hash: [0u8; 32],
            },
        ] {
            assert_eq!(w.path(), "a");
            assert_ne!(w.format(), "Pickle");
        }
    }

    proptest! {
        #[test]
        fn arbitrary_graphs_validate_and_round_trip(g in arbitrary_learning_graph()) {
            let diags: Vec<_> = g.validate().into_iter().filter(Diagnostic::is_error).collect();
            prop_assert!(diags.is_empty(), "{diags:#?}");
            let json = serde_json::to_string(&g).unwrap();
            prop_assert_eq!(serde_json::from_str::<LearningGraph>(&json).unwrap(), g);
        }
    }
}
