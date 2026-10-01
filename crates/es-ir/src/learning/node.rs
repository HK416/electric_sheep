//! The spec 8.3 node set: the parameter types, [`LearningNode`], the ports each node declares
//! and its canonical parameters (the `learning_hash` input).

use serde::{Deserialize, Serialize};

use super::{canon_ports, canon_unit, chunk, feature, policy_ty, TensorPort, WeightsRef};
use crate::graph::{IrNode, NodeId, Port};
use crate::hash::CanonWriter;
use crate::types::Unit;

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

/// The non-linearity a lowered `Mlp` puts between its layers (spec 8.3, packet M8/S2a).
///
/// brax's MLP is `swish`, `rsl_rl`'s is `ELU`, and the lowering emitted `ReLU` before this
/// parameter existed — so [`Self::Relu`] is the default and an absent `activation` is byte
/// for byte today's canonical form (see [`StateEncoderKind::canonical`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    #[default]
    Relu,
    Elu,
    /// `SiLU`, `x * sigmoid(x)` — `linen.swish`, `nn.SiLU`.
    Swish,
    Tanh,
}

impl Activation {
    /// Whether this is the activation an absent `activation` means.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_default(&self) -> bool {
        *self == Self::Relu
    }
}

/// Whether a `Regression` head squashes its output into `[-1, 1]` (spec 8.3, packet M8/S2a).
///
/// brax's deterministic inference is `tanh(location)`; without this the action port's
/// `Normalized { lo = -1, hi = 1 }` unit and the Safety Plane's clamp give the *range* of
/// `tanh` but not its shape (`quadruped-track.md` 3.4 item 3). Default [`Self::None`] = today's
/// canonical form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Squash {
    #[default]
    None,
    Tanh,
}

impl Squash {
    /// Whether this is the squash an absent `squash` means.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn is_default(&self) -> bool {
        *self == Self::None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateEncoderKind {
    Identity,
    Mlp {
        hidden: Vec<u32>,
        /// Absent = [`Activation::Relu`] = today's canonical form (packet M8/S2a).
        #[serde(default, skip_serializing_if = "Activation::is_default")]
        activation: Activation,
        /// Whether [`Self::Mlp::activation`] also follows the **last** `Linear`, the one that
        /// feeds the head. Upstream MLPs (brax, `rsl_rl`) activate every hidden layer including
        /// the last; ours did not. Absent = `false` = today's canonical form.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        activate_output: bool,
    },
}

impl StateEncoderKind {
    /// The canonical bytes of this kind.
    ///
    /// Written by hand rather than as `format!("{self:?}")` for one reason: the default
    /// `Mlp` must hash exactly as it did before `activation` and `activate_output` existed,
    /// i.e. as the string `Mlp { hidden: [256] }`, or every committed `learning_hash` moves.
    /// The two parameters are appended only when they are not the default — the same rule
    /// `SensorRender` follows in `task.rs` (packet M7/R5).
    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::Identity => w.str("Identity"),
            Self::Mlp {
                hidden,
                activation,
                activate_output,
            } => {
                w.str(&format!("Mlp {{ hidden: {hidden:?} }}"));
                if !activation.is_default() || *activate_output {
                    w.str(&format!("{activation:?}"));
                    w.bool(*activate_output);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionKind {
    Concat,
    CrossAttention,
    FiLm,
    AdaLn,
    TokenConcat,
    /// The element-wise sum of inputs that all have the node's own shape (`LRN-032`). With
    /// encoders that [`LearningNode::VisionEncoder::share`] weights, a view can be dropped
    /// without retraining the fusion: the remaining terms keep their meaning
    /// (`docs/design/multi-camera.md` section 3.3, packet M15/N5).
    Sum,
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
        /// The encoder whose weights this one uses (packet M15/N5): each view keeps its own
        /// node and its one input, and the lowering applies one module to each. It names the
        /// group's owner — a `VisionEncoder` of the same backbone, `out_dim`, `token_count`,
        /// `pretrained` and `frozen` that shares nothing itself — anywhere in the graph, before
        /// or after this node; anything else is `LRN-033`. Absent = no sharing = today's
        /// canonical form: it is never written into [`IrNode::params_canonical`] (a `NodeId`
        /// is not hash input), and `learning_hash` reads it as an edge instead.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        share: Option<NodeId>,
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
        /// Absent = [`Squash::None`] = today's canonical form (packet M8/S2a). Only a
        /// `Regression` head has an output to squash; anything else is `LRN-031`.
        #[serde(default, skip_serializing_if = "Squash::is_default")]
        squash: Squash,
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

impl LearningNode {
    pub(super) fn input_ports(&self) -> &[TensorPort] {
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
    pub(super) fn arity(&self) -> (usize, Option<usize>) {
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
                kind.canonical(w);
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
                squash,
                ..
            } => {
                w.str(&format!("{kind:?}"));
                w.u32(*action_dim);
                w.u32(*horizon);
                // Only when it is not the default: see [`Squash`].
                if !squash.is_default() {
                    w.str(&format!("{squash:?}"));
                }
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
