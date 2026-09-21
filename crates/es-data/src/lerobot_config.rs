//! `LeRobot` policy `config.json` → Electric Sheep IR conversion (spec §14.4 external
//! conversion, §14.2 Python builder shape, §7.4/§7.5 Observation IR, §8.4 `PolicyContract`).
//!
//! This is the Rust half of M2 Wave 6 only: parsing a policy `config.json` (plus an optional
//! `stats.json` and dataset `meta/info.json`) into a consistent `(ObservationIr,
//! LearningGraph)` pair. **The Python builder (`PyO3`, `es-py`) is the other half of W6 and is
//! not implemented here** — see `docs/packets/M2/W6-lerobot-config.md`.
//!
//! No `lerobot` version is pinned in this workspace (spec §1.7): every field below is
//! reconstructed from memory and marked `unverified` in `docs/api-notes/lerobot-config.md`.
//! Unknown JSON fields are never rejected — they are captured in `extra` and reported back as
//! [`Converted::warnings`], so a config from an unpinned newer/older `lerobot` still converts.

use std::collections::BTreeMap;
use std::time::Duration;

use es_core::StableId;
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, BetaSchedule, ChunkBlendPolicy, DiffusionScheduler,
    FusionKind, HeadKind, LearningGraph, LearningNode, NormalizeDir, PolicyContract, PolicyHandle,
    PredictionType, RuntimeHints, Squash, StateEncoderKind, StatsSource, TemporalKind,
    VarianceType, VisionBackbone, WeightsRef,
};
use es_ir::observation::{
    self, AugmentKind, History, Io, NormalizeStats, ObservationIr, ObservationNode,
    ObservationOutput, ResizeFilter, TemporalWindow,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_math::conventions::Pose;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lerobot::Info;

/// `ResNet18` (and most `torchvision` backbones) is pretrained at 224x224; a bare `config.json`
/// names no separate resize target, so this is the conversion's own assumption
/// (`docs/api-notes/lerobot-config.md`), applied after any crop.
const POLICY_RESOLUTION: u32 = 224;

/// `torchvision` `ImageNet` normalization constants — the fallback when no dataset `stats.json`
/// covers a visual feature.
const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];

// --- config.json ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FeatureKind {
    Visual,
    State,
    Action,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureCfg {
    #[serde(rename = "type")]
    pub kind: FeatureKind,
    pub shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NormMode {
    MeanStd,
    MinMax,
    Identity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActConfig {
    pub input_features: BTreeMap<String, FeatureCfg>,
    pub output_features: BTreeMap<String, FeatureCfg>,
    pub normalization_mapping: BTreeMap<String, NormMode>,
    pub chunk_size: u32,
    pub n_action_steps: u32,
    pub n_obs_steps: u32,
    pub vision_backbone: String,
    #[serde(default)]
    pub pretrained_backbone_weights: Option<String>,
    pub dim_model: u32,
    pub n_heads: u32,
    pub dim_feedforward: u32,
    pub n_encoder_layers: u32,
    pub n_decoder_layers: u32,
    pub use_vae: bool,
    pub latent_dim: u32,
    #[serde(default)]
    pub temporal_ensemble_coeff: Option<f64>,
    pub dropout: f64,
    pub kl_weight: f64,
    /// Everything not modeled above (`optimizer_lr`, `optimizer_weight_decay`, …):
    /// training-only fields this conversion does not need. Reported as warnings, never
    /// silently dropped.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffusionConfig {
    pub input_features: BTreeMap<String, FeatureCfg>,
    pub output_features: BTreeMap<String, FeatureCfg>,
    pub normalization_mapping: BTreeMap<String, NormMode>,
    pub horizon: u32,
    pub n_action_steps: u32,
    pub n_obs_steps: u32,
    #[serde(default)]
    pub crop_shape: Option<[u32; 2]>,
    #[serde(default)]
    pub crop_is_random: bool,
    pub use_group_norm: bool,
    pub down_dims: Vec<u32>,
    pub kernel_size: u32,
    pub noise_scheduler_type: String,
    pub num_train_timesteps: u32,
    pub beta_schedule: String,
    pub prediction_type: String,
    #[serde(default)]
    pub num_inference_steps: Option<u32>,
    /// `diffusers`' `clip_sample` / `clip_sample_range`, which `LeRobot` re-exports with the
    /// same defaults. Optional here because older configs predate them.
    #[serde(default = "default_clip_sample")]
    pub clip_sample: bool,
    #[serde(default = "default_clip_sample_range")]
    pub clip_sample_range: f32,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A `LeRobot` policy `config.json` (spec §14.4). `#[serde(tag = "type")]` dispatches on the
/// `"act"` / `"diffusion"` discriminator; [`LeRobotPolicyConfig::parse`] is the entry point
/// that turns an unrecognized `"type"` into [`ConfigError::Unsupported`] rather than a
/// generic serde error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LeRobotPolicyConfig {
    Act(ActConfig),
    Diffusion(DiffusionConfig),
}

impl LeRobotPolicyConfig {
    /// Parses a `config.json` document, reporting an unrecognized `"type"` as
    /// [`ConfigError::Unsupported`] (spec §14.4: severity=error blocks execution rather than
    /// guessing a mapping) instead of a generic deserialization failure.
    pub fn parse(json: &str) -> Result<Self, ConfigError> {
        let value: Value = serde_json::from_str(json)?;
        match value.get("type").and_then(Value::as_str) {
            Some("act" | "diffusion") | None => Ok(serde_json::from_value(value)?),
            Some(other) => Err(ConfigError::Unsupported(other.to_owned())),
        }
    }

    fn features(&self) -> (&BTreeMap<String, FeatureCfg>, &BTreeMap<String, FeatureCfg>) {
        match self {
            Self::Act(c) => (&c.input_features, &c.output_features),
            Self::Diffusion(c) => (&c.input_features, &c.output_features),
        }
    }

    fn n_obs_steps(&self) -> u32 {
        match self {
            Self::Act(c) => c.n_obs_steps,
            Self::Diffusion(c) => c.n_obs_steps,
        }
    }

    fn extra(&self) -> &BTreeMap<String, Value> {
        match self {
            Self::Act(c) => &c.extra,
            Self::Diffusion(c) => &c.extra,
        }
    }
}

// --- stats.json -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FeatureStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    #[serde(default)]
    pub min: Vec<f64>,
    #[serde(default)]
    pub max: Vec<f64>,
}

/// `meta/stats.json` (spec §19.1 documents this file as *ignored* by the dataset reader; this
/// packet is the first consumer of it, for Observation-IR normalization). Feature name -> the
/// per-channel/per-dimension moments `LeRobot` recorded over the training dataset.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats(pub BTreeMap<String, FeatureStats>);

// --- errors and result ------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unsupported LeRobot policy type: {0}")]
    Unsupported(String),
    #[error("malformed LeRobot config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing or malformed feature: {0}")]
    MissingFeature(String),
    #[error("feature dimension out of range: {0}")]
    OutOfRange(String),
    #[error("IR construction failed: {0}")]
    Ir(String),
}

/// Largest `shape` entry this crate will turn into a buffer. `config.json` is untrusted
/// input, and every dim is allocated at least twice (the mean and the std of an identity
/// statistic); 65,536 is orders of magnitude above any real robot's joint or action count.
const MAX_DIM: u64 = 65_536;

fn checked_dim(name: &str, what: &str, dim: u64) -> Result<u32, ConfigError> {
    if dim == 0 || dim > MAX_DIM {
        return Err(ConfigError::OutOfRange(format!(
            "\"{name}\": {what} dim is {dim}, outside 1..={MAX_DIM}"
        )));
    }
    Ok(dim as u32)
}

/// The result of [`convert`]: a consistent Observation IR / Learning IR pair, plus every
/// non-fatal thing the conversion had to guess or ignore.
#[derive(Debug)]
pub struct Converted {
    pub observation: ObservationIr,
    pub learning: LearningGraph,
    pub warnings: Vec<String>,
}

// --- shared IR-building helpers ---------------------------------------------------------------

/// `[feat]` when `token_count == 0` (a pooled feature vector), `[token_count, feat]` when the
/// stream still carries a temporal-window token axis (spec §7.5 layer 2 flowing into layer 3).
fn feat_shape(feat: u32, token_count: u32) -> Shape {
    if token_count == 0 {
        Shape::new([u64::from(feat)])
    } else {
        Shape::new([u64::from(token_count), u64::from(feat)])
    }
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

fn image_ty(sensor: StableId, spec: ImageSpec, unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([3, u64::from(spec.height), u64::from(spec.width)]),
        unit,
        frame: Frame::Camera(sensor),
        time: TimeRef::Sensor {
            id: sensor,
            align: Align::Hold,
        },
        image: Some(spec),
    }
}

/// A pinhole camera at the feature's declared `[C, H, W]` resolution. `LeRobot`'s `config.json`
/// carries no calibration (spec §7.2's `Intrinsics` is a dataset/robot concern), so `fx = fy =
/// width` and a centred principal point are a nominal placeholder, not a measurement.
fn nominal_camera(name: &str, shape: &[u64]) -> Result<ImageSpec, ConfigError> {
    let &[_, height, width] = shape else {
        return Err(ConfigError::MissingFeature(format!(
            "\"{name}\": VISUAL shape must be [channels, height, width], got {shape:?}"
        )));
    };
    let (width, height) = (width as u32, height as u32);
    Ok(ImageSpec {
        width,
        height,
        channels: ChannelFormat::Rgb,
        dtype: ImageDType::U8,
        color_space: ColorSpace::SRgb,
        camera_model: CameraModel::Pinhole,
        intrinsics: Intrinsics::new(
            f64::from(width),
            f64::from(width),
            f64::from(width) / 2.0,
            f64::from(height) / 2.0,
        ),
        extrinsics: Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: ShutterModel::Global,
        exposure: Duration::ZERO,
        rate_hz: 0.0,
        depth_scale: None,
    })
}

fn center_rect(spec: &ImageSpec, width: u32, height: u32) -> Rect {
    Rect {
        x: spec.width.saturating_sub(width) / 2,
        y: spec.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

/// The dataset moments for `name`, or a documented fallback with a warning when no `stats`
/// cover this feature (or their length disagrees with `dim`).
fn moments(
    name: &str,
    dim: usize,
    stats: Option<&Stats>,
    fallback: (&[f64], &[f64]),
    warnings: &mut Vec<String>,
) -> (Vec<f64>, Vec<f64>) {
    if let Some(fs) = stats.and_then(|s| s.0.get(name)) {
        if fs.mean.len() == dim && fs.std.len() == dim {
            return (fs.mean.clone(), fs.std.clone());
        }
        warnings.push(format!(
            "stats for \"{name}\" have {} value(s), expected {dim}; using the fallback instead",
            fs.mean.len()
        ));
    } else {
        warnings.push(format!(
            "no stats.json entry for \"{name}\"; using the documented fallback \
             (docs/api-notes/lerobot-config.md)"
        ));
    }
    (fallback.0.to_vec(), fallback.1.to_vec())
}

/// `MeanStd` normalization from `stats`, or the documented fallback (see [`moments`]).
fn mean_std(
    name: &str,
    dim: usize,
    stats: Option<&Stats>,
    fallback: (&[f64], &[f64]),
    warnings: &mut Vec<String>,
) -> NormalizeStats {
    let (mean, std) = moments(name, dim, stats, fallback, warnings);
    NormalizeStats::MeanStd { mean, std }
}

fn next_id(counter: &mut u32) -> NodeId {
    let id = NodeId(*counter);
    *counter += 1;
    id
}

/// Appends the layer-2 `TemporalWindow` node (spec §7.5) when `n_obs_steps > 1`, registering
/// the layer-1 `History` depth it needs. Returns the final port and type.
fn append_window(
    obs: &mut ObservationIr,
    counter: &mut u32,
    sensor: StableId,
    n_obs_steps: u32,
    last: NodeId,
    ty: PortType,
) -> (NodeId, PortType) {
    if n_obs_steps <= 1 {
        return (last, ty);
    }
    obs.temporal
        .history
        .insert(sensor, History { depth: n_obs_steps });
    let mut dims = vec![u64::from(n_obs_steps)];
    dims.extend(ty.shape.dims());
    let windowed = PortType {
        shape: Shape::new(dims),
        time: TimeRef::Window {
            base: Box::new(ty.time.clone()),
            n: n_obs_steps,
            stride: 1,
        },
        ..ty.clone()
    };
    let window = TemporalWindow {
        n_steps: n_obs_steps,
        stride: 1,
        align: Align::Hold,
    };
    let id = next_id(counter);
    obs.graph.insert(
        id,
        ObservationNode::TemporalWindowNode {
            window,
            io: Io::unary(ty, windowed.clone()),
        },
    );
    obs.graph
        .connect(last, observation::OUT, id, &observation::in_port(0));
    (id, windowed)
}

#[allow(clippy::too_many_arguments)]
fn build_visual_stream(
    obs: &mut ObservationIr,
    counter: &mut u32,
    name: &str,
    feature: &FeatureCfg,
    crop: Option<([u32; 2], bool)>,
    n_obs_steps: u32,
    stats: Option<&Stats>,
    warnings: &mut Vec<String>,
) -> Result<(), ConfigError> {
    let sensor = StableId::from_path(name);
    let raw_spec = nominal_camera(name, &feature.shape)?;
    let raw_ty = image_ty(sensor, raw_spec, Unit::Pixel);
    let input_id = next_id(counter);
    obs.graph.insert(
        input_id,
        ObservationNode::ImageInput {
            sensor,
            io: Io::source(raw_ty.clone()),
        },
    );

    let (mut cur_id, mut cur_ty, mut cur_spec) = (input_id, raw_ty, raw_spec);
    if let Some(([crop_h, crop_w], is_random)) = crop {
        let cropped_spec = cur_spec.cropped(center_rect(&cur_spec, crop_w, crop_h), true);
        let cropped_ty = image_ty(sensor, cropped_spec, Unit::Pixel);
        let crop_id = next_id(counter);
        obs.graph.insert(
            crop_id,
            ObservationNode::Crop {
                mode: observation::CropMode::Center {
                    width: crop_w,
                    height: crop_h,
                },
                rescale_intrinsics: true,
                io: Io::unary(cur_ty.clone(), cropped_ty.clone()),
            },
        );
        obs.graph
            .connect(cur_id, observation::OUT, crop_id, &observation::in_port(0));
        (cur_id, cur_ty, cur_spec) = (crop_id, cropped_ty, cropped_spec);
        if is_random {
            // `Augment` is geometry-preserving in this IR (it propagates the incoming
            // `ImageSpec` unchanged), so LeRobot's random crop is the centred `Crop` above
            // plus an offset jitter *of that window*, which is what this node names. Wired
            // into the chain rather than dangling: an unreachable node is a graph no planner
            // can lower. `training_only` means evaluation and deployment auto-disable it
            // (INV-15, spec §7.3/§10.4) and run the deterministic centred crop.
            let aug_id = next_id(counter);
            obs.graph.insert(
                aug_id,
                ObservationNode::Augment {
                    kind: AugmentKind::RandomCrop {
                        width: crop_w,
                        height: crop_h,
                    },
                    training_only: true,
                    io: Io::unary(cur_ty.clone(), cur_ty.clone()),
                },
            );
            obs.graph
                .connect(cur_id, observation::OUT, aug_id, &observation::in_port(0));
            cur_id = aug_id;
        }
    }

    let resized_spec = cur_spec.resized(POLICY_RESOLUTION, POLICY_RESOLUTION, true);
    let resized_ty = image_ty(sensor, resized_spec, Unit::Pixel);
    let resize_id = next_id(counter);
    obs.graph.insert(
        resize_id,
        ObservationNode::Resize {
            width: POLICY_RESOLUTION,
            height: POLICY_RESOLUTION,
            filter: ResizeFilter::Bilinear,
            rescale_intrinsics: true,
            io: Io::unary(cur_ty.clone(), resized_ty.clone()),
        },
    );
    obs.graph.connect(
        cur_id,
        observation::OUT,
        resize_id,
        &observation::in_port(0),
    );

    let normalized_spec = ImageSpec {
        dtype: ImageDType::F32,
        ..resized_spec
    };
    let unit = Unit::Normalized { lo: -1.0, hi: 1.0 };
    let normalized_ty = image_ty(sensor, normalized_spec, unit);
    let stats_fallback = mean_std(name, 3, stats, (&IMAGENET_MEAN, &IMAGENET_STD), warnings);
    let norm_id = next_id(counter);
    obs.graph.insert(
        norm_id,
        ObservationNode::Normalize {
            stats: stats_fallback,
            io: Io::unary(resized_ty, normalized_ty.clone()),
        },
    );
    obs.graph.connect(
        resize_id,
        observation::OUT,
        norm_id,
        &observation::in_port(0),
    );

    let (final_id, final_ty) =
        append_window(obs, counter, sensor, n_obs_steps, norm_id, normalized_ty);
    obs.outputs.insert(
        name.to_owned(),
        ObservationOutput {
            port: PortRef::new(final_id, observation::OUT),
            ty: final_ty,
        },
    );
    Ok(())
}

fn build_state_stream(
    obs: &mut ObservationIr,
    counter: &mut u32,
    name: &str,
    feature: &FeatureCfg,
    n_obs_steps: u32,
    stats: Option<&Stats>,
    warnings: &mut Vec<String>,
) -> Result<(), ConfigError> {
    let &[dim] = feature.shape.as_slice() else {
        return Err(ConfigError::MissingFeature(format!(
            "\"{name}\": STATE shape must be [dim], got {:?}",
            feature.shape
        )));
    };
    let dim = u64::from(checked_dim(name, "STATE", dim)?);
    let source = StableId::from_path(name);
    // Joint position control (spec §5.4's `Unit::Angle`) is the assumption this crate makes
    // for an unqualified STATE vector — LeRobot's config.json carries no physical unit
    // (docs/api-notes/lerobot-config.md).
    let raw_ty = PortType {
        elem: ElemType::F32,
        shape: Shape::new([dim]),
        unit: Unit::Angle,
        frame: Frame::Joint(source),
        time: TimeRef::Sensor {
            id: source,
            align: Align::Hold,
        },
        image: None,
    };
    let input_id = next_id(counter);
    obs.graph.insert(
        input_id,
        ObservationNode::StateInput {
            source,
            io: Io::source(raw_ty.clone()),
        },
    );

    let identity = (vec![0.0; dim as usize], vec![1.0; dim as usize]);
    let stats_fallback = mean_std(
        name,
        dim as usize,
        stats,
        (&identity.0, &identity.1),
        warnings,
    );
    let normalized_ty = PortType {
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        ..raw_ty.clone()
    };
    let norm_id = next_id(counter);
    obs.graph.insert(
        norm_id,
        ObservationNode::Normalize {
            stats: stats_fallback,
            io: Io::unary(raw_ty, normalized_ty.clone()),
        },
    );
    obs.graph.connect(
        input_id,
        observation::OUT,
        norm_id,
        &observation::in_port(0),
    );

    let (final_id, final_ty) =
        append_window(obs, counter, source, n_obs_steps, norm_id, normalized_ty);
    obs.outputs.insert(
        name.to_owned(),
        ObservationOutput {
            port: PortRef::new(final_id, observation::OUT),
            ty: final_ty,
        },
    );
    Ok(())
}

fn vision_backbone(name: &str, warnings: &mut Vec<String>) -> VisionBackbone {
    match name {
        "resnet18" => VisionBackbone::ResNet18,
        "resnet34" => VisionBackbone::ResNet34,
        other => {
            warnings.push(format!(
                "unrecognized vision_backbone \"{other}\"; assuming resnet18"
            ));
            VisionBackbone::ResNet18
        }
    }
}

fn default_clip_sample() -> bool {
    true
}

fn default_clip_sample_range() -> f32 {
    1.0
}

/// `diffusers`' `beta_schedule`. An unrecognized one falls back to `LeRobot`'s default with a
/// warning rather than to silence: the schedule decides every coefficient of the sampler.
fn beta_schedule(name: &str, warnings: &mut Vec<String>) -> BetaSchedule {
    match name {
        "linear" => BetaSchedule::Linear,
        "squaredcos_cap_v2" => BetaSchedule::SquaredcosCapV2,
        other => {
            warnings.push(format!(
                "unrecognized beta_schedule \"{other}\"; assuming \"squaredcos_cap_v2\""
            ));
            BetaSchedule::SquaredcosCapV2
        }
    }
}

/// `diffusers`' `prediction_type`. Only `epsilon` lowers today, but the other two are carried
/// through so `es-policy` can refuse them by name instead of misreading the denoiser.
fn prediction_type(name: &str, warnings: &mut Vec<String>) -> PredictionType {
    match name {
        "epsilon" => PredictionType::Epsilon,
        "sample" => PredictionType::Sample,
        "v_prediction" => PredictionType::VPrediction,
        other => {
            warnings.push(format!(
                "unrecognized prediction_type \"{other}\"; assuming \"epsilon\""
            ));
            PredictionType::Epsilon
        }
    }
}

fn diffusion_scheduler(name: &str, warnings: &mut Vec<String>) -> DiffusionScheduler {
    match name.to_ascii_uppercase().as_str() {
        "DDPM" => DiffusionScheduler::Ddpm,
        "DDIM" => DiffusionScheduler::Ddim,
        other => {
            warnings.push(format!(
                "unrecognized noise_scheduler_type \"{other}\"; assuming a DPM-Solver-family sampler"
            ));
            DiffusionScheduler::DpmSolver
        }
    }
}

/// Builds the Learning IR mirroring `crates/es-policy`'s expected ACT/Diffusion lowering
/// (spec §8.3): `VisionEncoder(s) + StateEncoder -> Fusion -> TemporalEncoder -> PolicyHead ->
/// ActionChunker -> Normalizer(Inverse)`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_learning(
    cfg: &LeRobotPolicyConfig,
    obs: &ObservationIr,
    visual_names: &[String],
    state_name: &str,
    action_name: &str,
    action_dim: u32,
    n_obs_steps: u32,
    dataset_info: Option<&Info>,
    stats: Option<&Stats>,
    warnings: &mut Vec<String>,
) -> Result<LearningGraph, ConfigError> {
    let feat = match cfg {
        LeRobotPolicyConfig::Act(c) => c.dim_model,
        LeRobotPolicyConfig::Diffusion(c) => c.down_dims.last().copied().unwrap_or(512),
    };
    let token_count = if n_obs_steps > 1 { n_obs_steps } else { 0 };

    let mut g: Graph<LearningNode> = Graph::new(1);
    let mut counter = 0u32;
    let mut boundary_ports: Vec<Port> = Vec::new();
    let mut fusion_inputs: Vec<Port> = Vec::new();
    let mut vision_ids: Vec<(NodeId, Port)> = Vec::new();

    let pretrained = match cfg {
        LeRobotPolicyConfig::Act(c) => c.pretrained_backbone_weights.is_some(),
        LeRobotPolicyConfig::Diffusion(_) => true,
    };
    let backbone_name = match cfg {
        LeRobotPolicyConfig::Act(c) => c.vision_backbone.as_str(),
        LeRobotPolicyConfig::Diffusion(_) => "resnet18",
    };
    let backbone = vision_backbone(backbone_name, warnings);

    for name in visual_names {
        let ty = &obs
            .outputs
            .get(name)
            .ok_or_else(|| ConfigError::Ir(format!("no observation output \"{name}\"")))?
            .ty;
        let in_port = Port::new(name.clone(), policy_ty(ty.shape.clone(), ty.unit.clone()));
        let vid = next_id(&mut counter);
        g.insert(
            vid,
            LearningNode::VisionEncoder {
                inputs: vec![in_port.clone()],
                backbone: backbone.clone(),
                pretrained,
                frozen: false,
                out_dim: feat,
                token_count,
            },
        );
        g.inputs.push(PortRef::new(vid, name.clone()));
        boundary_ports.push(in_port);
        let out_port = Port::new(
            format!("image_{name}"),
            policy_ty(feat_shape(feat, token_count), Unit::Dimensionless),
        );
        vision_ids.push((vid, out_port.clone()));
        fusion_inputs.push(out_port);
    }

    let state_ty = &obs
        .outputs
        .get(state_name)
        .ok_or_else(|| ConfigError::Ir(format!("no observation output \"{state_name}\"")))?
        .ty;
    let state_in = Port::new(
        state_name.to_owned(),
        policy_ty(state_ty.shape.clone(), state_ty.unit.clone()),
    );
    let state_id = next_id(&mut counter);
    g.insert(
        state_id,
        LearningNode::StateEncoder {
            inputs: vec![state_in.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![feat],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: feat,
        },
    );
    g.inputs.push(PortRef::new(state_id, state_name.to_owned()));
    boundary_ports.push(state_in);
    // `LearningNode::StateEncoder::outputs()` always pools to `[feat]` (spec §8.3: a state
    // vector, unlike a vision stream, carries no per-frame token axis out of its encoder).
    let state_out = Port::new(
        "state_feat",
        policy_ty(feat_shape(feat, 0), Unit::Dimensionless),
    );
    fusion_inputs.push(state_out.clone());

    let fusion_id = next_id(&mut counter);
    g.insert(
        fusion_id,
        LearningNode::Fusion {
            inputs: fusion_inputs.clone(),
            kind: FusionKind::Concat,
            out_dim: feat,
            token_count,
        },
    );
    for (vid, port) in &vision_ids {
        g.connect(*vid, "out", fusion_id, &port.name);
    }
    g.connect(state_id, "out", fusion_id, &state_out.name);

    let temporal_id = next_id(&mut counter);
    let temporal_kind = if n_obs_steps > 1 {
        TemporalKind::Transformer
    } else {
        TemporalKind::None
    };
    g.insert(
        temporal_id,
        LearningNode::TemporalEncoder {
            inputs: vec![Port::new(
                "seq",
                policy_ty(feat_shape(feat, token_count), Unit::Dimensionless),
            )],
            kind: temporal_kind,
            n_frames: n_obs_steps.max(1),
            out_dim: feat,
            token_count: 0,
        },
    );
    g.connect(fusion_id, "out", temporal_id, "seq");

    let (horizon, execute_chunk, head_kind, exec_mode, blend) = match cfg {
        LeRobotPolicyConfig::Act(c) => {
            let (mode, blend) = match c.temporal_ensemble_coeff {
                Some(coeff) => (
                    ActionExecutionMode::TemporalEnsemble,
                    ChunkBlendPolicy::TemporalEnsemble {
                        weight_decay: coeff as f32,
                    },
                ),
                None => (
                    ActionExecutionMode::RecedingHorizon,
                    ChunkBlendPolicy::HardSwitch,
                ),
            };
            (
                c.chunk_size,
                c.n_action_steps,
                HeadKind::Regression,
                mode,
                blend,
            )
        }
        LeRobotPolicyConfig::Diffusion(c) => {
            let n_steps = c.num_inference_steps.unwrap_or(c.num_train_timesteps);
            let scheduler = diffusion_scheduler(&c.noise_scheduler_type, warnings);
            (
                c.horizon,
                c.n_action_steps,
                HeadKind::Diffusion {
                    n_steps,
                    scheduler,
                    // The schedule is built over the *training* grid and subsampled to
                    // `n_steps`, so both numbers have to survive the conversion.
                    num_train_timesteps: c.num_train_timesteps,
                    beta_schedule: beta_schedule(&c.beta_schedule, warnings),
                    // Not a `LeRobot` field: `DDPMScheduler`'s default is `fixed_small`.
                    variance_type: VarianceType::FixedSmall,
                    prediction_type: prediction_type(&c.prediction_type, warnings),
                    clip_sample: c.clip_sample,
                    clip_sample_range: c.clip_sample_range,
                },
                ActionExecutionMode::RecedingHorizon,
                ChunkBlendPolicy::HardSwitch,
            )
        }
    };

    let head_id = next_id(&mut counter);
    g.insert(
        head_id,
        LearningNode::PolicyHead {
            inputs: vec![Port::new(
                "feat",
                policy_ty(Shape::new([u64::from(feat)]), Unit::Dimensionless),
            )],
            kind: head_kind,
            action_dim,
            horizon,
            squash: Squash::None,
        },
    );
    g.connect(temporal_id, "out", head_id, "feat");

    let replanning_hz = dataset_info.map_or_else(
        || {
            warnings.push(
                "no dataset meta/info.json given; assuming replanning_hz = 10.0 (spec §8.4 \
                 Target/Status: unverified)"
                    .to_owned(),
            );
            10.0
        },
        |info| info.fps as f32,
    );
    let chunk_action_unit = Unit::Normalized { lo: -1.0, hi: 1.0 };
    let chunk_shape = Shape::new([u64::from(execute_chunk), u64::from(action_dim)]);
    let chunker_id = next_id(&mut counter);
    g.insert(
        chunker_id,
        LearningNode::ActionChunker {
            inputs: vec![Port::new(
                "chunk",
                policy_ty(
                    Shape::new([u64::from(horizon), u64::from(action_dim)]),
                    chunk_action_unit.clone(),
                ),
            )],
            horizon,
            execute_chunk,
            replan_hz: replanning_hz,
            mode: exec_mode,
            blend,
            buffer_chunks: 2,
        },
    );
    g.connect(head_id, "chunk", chunker_id, "chunk");

    // Inverse-unnormalize the chunk back into the action feature's physical space. LeRobot's
    // config.json declares no unit for `action` either; joint position is assumed (same
    // caveat as the state stream).
    let action_stats = stats_source(action_name, action_dim, stats, warnings);
    let norm_id = next_id(&mut counter);
    g.insert(
        norm_id,
        LearningNode::Normalizer {
            inputs: vec![Port::new(
                "actions",
                policy_ty(chunk_shape.clone(), chunk_action_unit),
            )],
            direction: NormalizeDir::Inverse,
            stats: action_stats,
            out_unit: Unit::Angle,
        },
    );
    g.connect(chunker_id, "actions", norm_id, "actions");

    g.outputs = vec![PortRef::new(norm_id, "out")];
    let final_out = Port::new("actions", policy_ty(chunk_shape, Unit::Angle));

    let contract_inputs: BTreeMap<String, Port> = boundary_ports
        .iter()
        .cloned()
        .map(|p| (p.name.clone(), p))
        .collect();

    Ok(LearningGraph {
        schema_version: 1,
        inputs: boundary_ports,
        outputs: vec![final_out],
        nodes: g,
        policy: PolicyHandle {
            architecture: match cfg {
                LeRobotPolicyConfig::Act(_) => ArchKind::Act,
                LeRobotPolicyConfig::Diffusion(_) => ArchKind::Diffusion,
            },
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0u8; 32],
            },
            contract: PolicyContract {
                inputs: contract_inputs,
                observation_window: n_obs_steps.max(1),
                action_dim,
                horizon,
                execute_chunk,
                replanning_hz,
                execution_mode: exec_mode,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    // No measurement exists for a config-only conversion (spec §12.4: report
                    // as unverified rather than a fabricated benchmark).
                    expected_latency_ms: match cfg {
                        LeRobotPolicyConfig::Act(_) => 5.0,
                        LeRobotPolicyConfig::Diffusion(_) => 15.0,
                    },
                    deadline_ms: 100.0,
                },
            },
        },
    })
}

/// The action chunk's unnormalization stats from `stats.json`'s `"action"` entry, or the
/// identity with a warning (see [`moments`]).
fn stats_source(
    name: &str,
    dim: u32,
    stats: Option<&Stats>,
    warnings: &mut Vec<String>,
) -> StatsSource {
    let identity = (vec![0.0; dim as usize], vec![1.0; dim as usize]);
    let (mean, std) = moments(
        name,
        dim as usize,
        stats,
        (&identity.0, &identity.1),
        warnings,
    );
    StatsSource::MeanStd { mean, std }
}

// --- entry point ------------------------------------------------------------------------------

/// Converts a `LeRobot` policy `config.json` (plus optional dataset `stats.json` and
/// `meta/info.json`) into a consistent `(ObservationIr, LearningGraph)` pair (spec §14.4).
///
/// The pair is built to pass [`ObservationIr::validate`], [`LearningGraph::validate`] and the
/// Observation↔Learning half of `es_ir::cross::check` on its own; a full `IrBundle` also needs
/// a Task IR and a Deployment IR, which a bare policy config does not carry (see
/// `crates/es-data/tests/lerobot_config.rs` for a minimal synthetic pair that closes that
/// loop for testing).
pub fn convert(
    cfg: &LeRobotPolicyConfig,
    stats: Option<&Stats>,
    dataset_info: Option<&Info>,
) -> Result<Converted, ConfigError> {
    let mut warnings: Vec<String> = cfg
        .extra()
        .keys()
        .map(|k| format!("config field \"{k}\" is not used by this conversion (training-only or unrecognized)"))
        .collect();

    let (inputs, outputs) = cfg.features();
    let visuals: Vec<(String, FeatureCfg)> = inputs
        .iter()
        .filter(|(_, f)| f.kind == FeatureKind::Visual)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if visuals.is_empty() {
        return Err(ConfigError::MissingFeature(
            "input_features has no VISUAL entry".to_owned(),
        ));
    }
    let (state_name, state_cfg) = inputs
        .iter()
        .find(|(_, f)| f.kind == FeatureKind::State)
        .ok_or_else(|| {
            ConfigError::MissingFeature("input_features has no STATE entry".to_owned())
        })?;
    let (action_name, action_cfg) = outputs
        .iter()
        .find(|(_, f)| f.kind == FeatureKind::Action)
        .ok_or_else(|| {
            ConfigError::MissingFeature("output_features has no ACTION entry".to_owned())
        })?;
    let &[action_dim] = action_cfg.shape.as_slice() else {
        return Err(ConfigError::MissingFeature(format!(
            "\"{action_name}\": ACTION shape must be [dim], got {:?}",
            action_cfg.shape
        )));
    };
    let action_dim = checked_dim(action_name, "ACTION", action_dim)?;
    let n_obs_steps = cfg.n_obs_steps();

    let crop = match cfg {
        LeRobotPolicyConfig::Diffusion(c) => c.crop_shape.map(|shape| (shape, c.crop_is_random)),
        LeRobotPolicyConfig::Act(_) => None,
    };

    // A bare policy config names no Task IR; `task_ref` is the all-zero placeholder spec §7.4
    // uses nowhere else, documented here rather than invented as a fake hash.
    let mut obs = ObservationIr::new(1, [0u8; 32]);
    let mut counter = 0u32;
    let mut visual_names = Vec::with_capacity(visuals.len());
    for (name, feature) in &visuals {
        build_visual_stream(
            &mut obs,
            &mut counter,
            name,
            feature,
            crop,
            n_obs_steps,
            stats,
            &mut warnings,
        )?;
        visual_names.push(name.clone());
    }
    build_state_stream(
        &mut obs,
        &mut counter,
        state_name,
        state_cfg,
        n_obs_steps,
        stats,
        &mut warnings,
    )?;
    obs.temporal.window = Some(TemporalWindow {
        n_steps: n_obs_steps.max(1),
        stride: 1,
        align: Align::Hold,
    });

    let learning = build_learning(
        cfg,
        &obs,
        &visual_names,
        state_name,
        action_name,
        action_dim,
        n_obs_steps,
        dataset_info,
        stats,
        &mut warnings,
    )?;

    Ok(Converted {
        observation: obs,
        learning,
        warnings,
    })
}
