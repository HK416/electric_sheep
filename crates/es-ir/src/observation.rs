//! Observation IR (spec 7): sensor output to policy input tensor, as one typed graph.
//!
//! The point of the layer is that the *same* graph runs in simulation, over a recorded
//! dataset, and on the robot (spec 7.6), so preprocessing cannot drift between them. Two
//! things make that possible and both live here: `ImageSpec` travels with the tensor
//! (spec 7.2), and the geometric nodes are required to transform the intrinsics they carry
//! (`INV-14`) — [`ObservationIr::propagate_image_specs`] does that by calling
//! [`ImageSpec::resized`] / [`ImageSpec::cropped`] / [`ImageSpec::undistorted`] and never by
//! editing an [`Intrinsics`] by hand.
//!
//! One Task IR can carry several Observation IRs (spec 7.4): same `task_ref`, different
//! [`ObservationIr::observation_hash`].

use std::collections::{BTreeMap, BTreeSet};

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::codes;
use crate::diag::{Diagnostic, Severity};
use crate::graph::{Graph, IrNode, NodeId, Port, PortRef};
use crate::hash::{canonical_hash, CanonWriter};
use crate::image::{ChannelFormat, ColorSpace, ImageDType, ImageSpec, Intrinsics, Rect};
use crate::types::{Align, PortType, TimeRef};

/// Domain separator for [`ObservationIr::observation_hash`].
const OBS_TAG: &str = "es.observation_hash.v1";

/// The single output port every node has.
pub const OUT: &str = "out";

/// Name of the `i`-th input port. Uniform across the node set so wiring is mechanical.
pub fn in_port(i: usize) -> String {
    format!("in{i}")
}

// --- time model (spec 7.5) ----------------------------------------------------------------

/// Layer 1 of the spec 7.5 time model: the system ring buffer `History<T, N>`.
///
/// The buffer itself belongs to `es-core` / `es-sensor` and never appears in the graph; the IR
/// only declares how deep it has to be for the windows below to be satisfiable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct History {
    /// `N`: how many past samples of this sensor the system keeps.
    pub depth: u32,
}

/// Layer 2 of the spec 7.5 time model: what the policy is defined to look at.
///
/// "The policy sees the last 2 frames, 200 ms apart" — it fixes tensor shape, memory budget and
/// dataset sampling. Layer 3, `TemporalEncoder`, is a network architecture choice and belongs
/// to Learning IR (spec 7.5); it has no type here on purpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalWindow {
    pub n_steps: u32,
    pub stride: u32,
    pub align: Align,
}

impl TemporalWindow {
    /// How many buffered samples the window reaches back over.
    pub fn required_depth(&self) -> u32 {
        self.n_steps.saturating_sub(1).saturating_mul(self.stride) + 1
    }

    fn canonical(&self, w: &mut CanonWriter) {
        w.u32(self.n_steps);
        w.u32(self.stride);
        w.str(&format!("{:?}", self.align));
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalModel {
    /// Layer 1, per sensor.
    pub history: BTreeMap<StableId, History>,
    /// Layer 2. `None` is the single-frame case (ACT, `n_obs_steps = 1`).
    pub window: Option<TemporalWindow>,
}

impl TemporalModel {
    fn canonical(&self, w: &mut CanonWriter) {
        w.seq(self.history.len());
        for (id, h) in &self.history {
            w.bytes(id.as_bytes());
            w.u32(h.depth);
        }
        match &self.window {
            None => w.bool(false),
            Some(win) => {
                w.bool(true);
                win.canonical(w);
            }
        }
    }
}

// --- node parameters ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResizeFilter {
    Nearest,
    Bilinear,
    Bicubic,
    Area,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CropMode {
    Rect(Rect),
    Center {
        width: u32,
        height: u32,
    },
    /// Per-sample offset. The nominal geometry the type system carries is the centred one.
    Random {
        width: u32,
        height: u32,
    },
}

impl CropMode {
    fn rect(self, spec: &ImageSpec) -> Rect {
        match self {
            Self::Rect(r) => r,
            Self::Center { width, height } | Self::Random { width, height } => Rect {
                x: spec.width.saturating_sub(width) / 2,
                y: spec.height.saturating_sub(height) / 2,
                width,
                height,
            },
        }
    }

    fn canonical(self, w: &mut CanonWriter) {
        match self {
            Self::Rect(r) => {
                w.str("Rect");
                for v in [r.x, r.y, r.width, r.height] {
                    w.u32(v);
                }
            }
            Self::Center { width, height } | Self::Random { width, height } => {
                w.str(if matches!(self, Self::Center { .. }) {
                    "Center"
                } else {
                    "Random"
                });
                w.u32(width);
                w.u32(height);
            }
        }
    }
}

/// `Unit::Normalized` is produced either from dataset statistics or from a declared range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NormalizeStats {
    MeanStd { mean: Vec<f64>, std: Vec<f64> },
    Range { lo: f64, hi: f64 },
}

impl NormalizeStats {
    fn canonical(&self, w: &mut CanonWriter) {
        match self {
            Self::MeanStd { mean, std } => {
                w.str("MeanStd");
                for v in [mean, std] {
                    w.seq(v.len());
                    for x in v {
                        w.f64(*x);
                    }
                }
            }
            Self::Range { lo, hi } => {
                w.str("Range");
                w.f64(*lo);
                w.f64(*hi);
            }
        }
    }
}

/// The spec 7.3 augmentation family. One node kind, because the rule that matters is shared:
/// it is `training_only`, so Evaluation IR switches all of it off structurally.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AugmentKind {
    RandomCrop {
        width: u32,
        height: u32,
    },
    ColorJitter {
        brightness: f64,
        contrast: f64,
        saturation: f64,
        hue: f64,
    },
    RandomErasing {
        probability: f64,
    },
    GaussianNoise {
        sigma: f64,
    },
}

impl AugmentKind {
    fn canonical(self, w: &mut CanonWriter) {
        match self {
            Self::RandomCrop { width, height } => {
                w.str("RandomCrop");
                w.u32(width);
                w.u32(height);
            }
            Self::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
            } => {
                w.str("ColorJitter");
                for v in [brightness, contrast, saturation, hue] {
                    w.f64(v);
                }
            }
            Self::RandomErasing { probability } => {
                w.str("RandomErasing");
                w.f64(probability);
            }
            Self::GaussianNoise { sigma } => {
                w.str("GaussianNoise");
                w.f64(sigma);
            }
        }
    }
}

/// The types a node declares for its ports. Inputs are `in0..inN`, the output is `out`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Io {
    pub inputs: Vec<PortType>,
    pub output: PortType,
}

impl Io {
    /// A source node: no inputs.
    pub fn source(output: PortType) -> Self {
        Self {
            inputs: Vec::new(),
            output,
        }
    }

    /// A one-in one-out node.
    pub fn unary(input: PortType, output: PortType) -> Self {
        Self {
            inputs: vec![input],
            output,
        }
    }

    pub fn new(inputs: Vec<PortType>, output: PortType) -> Self {
        Self { inputs, output }
    }
}

// --- the node set (spec 7.3) ---------------------------------------------------------------

/// Every node kind of spec 7.3. No neural network appears here: an encoder is Learning IR
/// (spec 5.1), and a `TemporalEncoder` is not a preprocessing step (spec 7.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObservationNode {
    /// Takes the `ImageSpec` from the sensor registry; it is the head of every image chain.
    ImageInput {
        sensor: StableId,
        io: Io,
    },
    /// Joint, IMU, F/T and other vector state.
    StateInput {
        source: StableId,
        io: Io,
    },
    /// Task instruction or external text, for a VLA.
    LanguageInput {
        source: StableId,
        io: Io,
    },

    Resize {
        width: u32,
        height: u32,
        filter: ResizeFilter,
        rescale_intrinsics: bool,
        io: Io,
    },
    Crop {
        mode: CropMode,
        rescale_intrinsics: bool,
        io: Io,
    },
    Pad {
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        io: Io,
    },
    Undistort {
        rectified: Intrinsics,
        io: Io,
    },
    Rectify {
        rectified: Intrinsics,
        io: Io,
    },
    Warp {
        homography: [f64; 9],
        io: Io,
    },
    /// 3D point to pixel and back. Consumes intrinsics and extrinsics, so stale ones here are
    /// exactly the `OBS-034` bug of spec 7.2.
    CameraProjection {
        intrinsics: Intrinsics,
        io: Io,
    },

    ColorTransform {
        src: ColorSpace,
        dst: ColorSpace,
        io: Io,
    },
    ToGray {
        io: Io,
    },
    ChannelSelect {
        channels: Vec<u32>,
        io: Io,
    },
    Normalize {
        stats: NormalizeStats,
        io: Io,
    },
    QuantizeU8 {
        io: Io,
    },
    Dequantize {
        io: Io,
    },

    /// Layer 2 of spec 7.5 applied to one stream.
    TemporalWindowNode {
        window: TemporalWindow,
        io: Io,
    },
    /// Channel-axis concatenation of `n` frames — a preprocessing step, not an architecture.
    FrameStack {
        n: u32,
        io: Io,
    },
    Delta {
        n: u32,
        io: Io,
    },

    Concat {
        axis: u32,
        time_align: Option<Align>,
        io: Io,
    },
    Stack {
        axis: u32,
        io: Io,
    },
    Mask {
        source: StableId,
        io: Io,
    },
    MultiViewPack {
        cameras: Vec<StableId>,
        io: Io,
    },

    Augment {
        kind: AugmentKind,
        training_only: bool,
        io: Io,
    },
}

impl ObservationNode {
    pub fn io(&self) -> &Io {
        match self {
            Self::ImageInput { io, .. }
            | Self::StateInput { io, .. }
            | Self::LanguageInput { io, .. }
            | Self::Resize { io, .. }
            | Self::Crop { io, .. }
            | Self::Pad { io, .. }
            | Self::Undistort { io, .. }
            | Self::Rectify { io, .. }
            | Self::Warp { io, .. }
            | Self::CameraProjection { io, .. }
            | Self::ColorTransform { io, .. }
            | Self::ToGray { io }
            | Self::ChannelSelect { io, .. }
            | Self::Normalize { io, .. }
            | Self::QuantizeU8 { io }
            | Self::Dequantize { io }
            | Self::TemporalWindowNode { io, .. }
            | Self::FrameStack { io, .. }
            | Self::Delta { io, .. }
            | Self::Concat { io, .. }
            | Self::Stack { io, .. }
            | Self::Mask { io, .. }
            | Self::MultiViewPack { io, .. }
            | Self::Augment { io, .. } => io,
        }
    }
}

impl IrNode for ObservationNode {
    fn kind(&self) -> &'static str {
        match self {
            Self::ImageInput { .. } => "ImageInput",
            Self::StateInput { .. } => "StateInput",
            Self::LanguageInput { .. } => "LanguageInput",
            Self::Resize { .. } => "Resize",
            Self::Crop { .. } => "Crop",
            Self::Pad { .. } => "Pad",
            Self::Undistort { .. } => "Undistort",
            Self::Rectify { .. } => "Rectify",
            Self::Warp { .. } => "Warp",
            Self::CameraProjection { .. } => "CameraProjection",
            Self::ColorTransform { .. } => "ColorTransform",
            Self::ToGray { .. } => "ToGray",
            Self::ChannelSelect { .. } => "ChannelSelect",
            Self::Normalize { .. } => "Normalize",
            Self::QuantizeU8 { .. } => "QuantizeU8",
            Self::Dequantize { .. } => "Dequantize",
            Self::TemporalWindowNode { .. } => "TemporalWindow",
            Self::FrameStack { .. } => "FrameStack",
            Self::Delta { .. } => "Delta",
            Self::Concat { .. } => "Concat",
            Self::Stack { .. } => "Stack",
            Self::Mask { .. } => "Mask",
            Self::MultiViewPack { .. } => "MultiViewPack",
            Self::Augment { .. } => "Augment",
        }
    }

    fn inputs(&self) -> Vec<Port> {
        self.io()
            .inputs
            .iter()
            .enumerate()
            .map(|(i, ty)| Port::new(in_port(i), ty.clone()))
            .collect()
    }

    fn outputs(&self) -> Vec<Port> {
        vec![Port::new(OUT, self.io().output.clone())]
    }

    fn params_canonical(&self, w: &mut CanonWriter) {
        let io = self.io();
        w.seq(io.inputs.len());
        for ty in &io.inputs {
            ty.canonical(w);
        }
        io.output.canonical(w);
        match self {
            Self::ImageInput { sensor: id, .. }
            | Self::StateInput { source: id, .. }
            | Self::LanguageInput { source: id, .. }
            | Self::Mask { source: id, .. } => w.bytes(id.as_bytes()),
            Self::Resize {
                width,
                height,
                filter,
                rescale_intrinsics,
                ..
            } => {
                w.u32(*width);
                w.u32(*height);
                w.str(&format!("{filter:?}"));
                w.bool(*rescale_intrinsics);
            }
            Self::Crop {
                mode,
                rescale_intrinsics,
                ..
            } => {
                mode.canonical(w);
                w.bool(*rescale_intrinsics);
            }
            Self::Pad {
                left,
                top,
                right,
                bottom,
                ..
            } => {
                for v in [left, top, right, bottom] {
                    w.u32(*v);
                }
            }
            Self::Undistort { rectified: k, .. }
            | Self::Rectify { rectified: k, .. }
            | Self::CameraProjection { intrinsics: k, .. } => {
                for v in [k.fx, k.fy, k.cx, k.cy, k.skew] {
                    w.f64(v);
                }
            }
            Self::Warp { homography, .. } => {
                for v in homography {
                    w.f64(*v);
                }
            }
            Self::ColorTransform { src, dst, .. } => {
                w.str(&format!("{src:?}"));
                w.str(&format!("{dst:?}"));
            }
            Self::ToGray { .. } | Self::QuantizeU8 { .. } | Self::Dequantize { .. } => {}
            Self::ChannelSelect { channels, .. } => {
                w.seq(channels.len());
                for c in channels {
                    w.u32(*c);
                }
            }
            Self::Normalize { stats, .. } => stats.canonical(w),
            Self::TemporalWindowNode { window, .. } => window.canonical(w),
            Self::FrameStack { n, .. } | Self::Delta { n, .. } => w.u32(*n),
            Self::Concat {
                axis, time_align, ..
            } => {
                w.u32(*axis);
                w.str(&format!("{time_align:?}"));
            }
            Self::Stack { axis, .. } => w.u32(*axis),
            Self::MultiViewPack { cameras, .. } => {
                w.seq(cameras.len());
                for c in cameras {
                    w.bytes(c.as_bytes());
                }
            }
            Self::Augment {
                kind,
                training_only,
                ..
            } => {
                kind.canonical(w);
                w.bool(*training_only);
            }
        }
    }
}

// --- the IR ---------------------------------------------------------------------------------

/// A named tensor the graph exposes to Learning IR.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationOutput {
    pub port: PortRef,
    pub ty: PortType,
}

/// Sensor to policy-input tensor (spec 7).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationIr {
    pub schema_version: u32,
    /// The `task_hash` this IR is an observation of (spec 7.4).
    pub task_ref: [u8; 32],
    pub graph: Graph<ObservationNode>,
    pub temporal: TemporalModel,
    /// Keyed by the name Learning IR binds to, e.g. `"rgb_front"`.
    pub outputs: BTreeMap<String, ObservationOutput>,
}

impl ObservationIr {
    pub fn new(schema_version: u32, task_ref: [u8; 32]) -> Self {
        Self {
            schema_version,
            task_ref,
            graph: Graph::new(schema_version),
            temporal: TemporalModel::default(),
            outputs: BTreeMap::new(),
        }
    }

    /// The `ImageSpec` on every image-carrying output port, in topological order.
    ///
    /// Geometry changes go through [`ImageSpec`]'s own transforms (`INV-14`), so intrinsics
    /// cannot be forgotten. Returns the diagnostics only when one of them is an error;
    /// warnings such as an explicit `rescale_intrinsics = false` reach the caller through
    /// [`ObservationIr::validate`], which keeps them.
    pub fn propagate_image_specs(&self) -> Result<BTreeMap<PortRef, ImageSpec>, Vec<Diagnostic>> {
        let mut diags = Vec::new();
        let specs = self.propagate(&mut diags);
        if diags.iter().any(Diagnostic::is_error) {
            Err(diags)
        } else {
            Ok(specs)
        }
    }

    fn propagate(&self, diags: &mut Vec<Diagnostic>) -> BTreeMap<PortRef, ImageSpec> {
        let order = match self.graph.topo_order() {
            Ok(o) => o,
            Err(d) => {
                diags.push(d);
                return BTreeMap::new();
            }
        };
        let mut specs: BTreeMap<PortRef, ImageSpec> = BTreeMap::new();
        for id in order {
            let node = &self.graph.nodes[&id];
            // The first declared input port that carries an image is the geometric source.
            let mut feeds: Vec<(&str, &PortRef)> = self
                .graph
                .edges
                .iter()
                .filter(|e| e.to.node == id)
                .map(|e| (e.to.port.as_str(), &e.from))
                .collect();
            feeds.sort_unstable();
            let incoming = feeds.iter().find_map(|(_, from)| specs.get(*from).copied());
            if let Some(out) = image_out(node, id, incoming, diags) {
                specs.insert(PortRef::new(id, OUT), out);
            }
        }
        specs
    }

    /// Every check spec 7 mandates, plus the graph's own port and edge checks.
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut diags = self.graph.validate_declared_ports();
        self.propagate(&mut diags);

        for (id, node) in &self.graph.nodes {
            match node {
                // Normalize exists to make a tensor admissible as a policy input (spec 5.4).
                ObservationNode::Normalize { io, .. } => {
                    if !matches!(io.output.unit, crate::types::Unit::Normalized { .. }) {
                        diags.push(
                            Diagnostic::new(
                                codes::OBS_040,
                                format!("Normalize output carries {:?}", io.output.unit),
                            )
                            .at(*id)
                            .on_port(OUT)
                            .with_hint("a Normalize output is Unit::Normalized { lo, hi }"),
                        );
                    }
                }
                ObservationNode::Augment { training_only, .. } if !*training_only => {
                    diags.push(
                        Diagnostic::new(codes::OBS_041, "augmentation node is not training_only")
                            .at(*id)
                            .with_hint(
                                "augmentation must switch off under Evaluation IR (spec 7.3)",
                            ),
                    );
                }
                ObservationNode::Concat {
                    time_align: None,
                    io,
                    ..
                } => {
                    let clocks: BTreeSet<String> = io
                        .inputs
                        .iter()
                        .filter(|t| matches!(t.time, TimeRef::Sensor { .. }))
                        .map(|t| format!("{:?}", t.time))
                        .collect();
                    if clocks.len() > 1 {
                        diags.push(
                            Diagnostic::new(
                                codes::TYPE_014,
                                format!("Concat inputs: {}", clocks.iter().cloned().collect::<Vec<_>>().join(", ")),
                            )
                            .at(*id)
                            .with_hint(
                                "time_align = \"hold\" | \"interpolate\" | \"reject\" (ACT and Diffusion Policy usually use \"hold\")",
                            ),
                        );
                    }
                }
                ObservationNode::TemporalWindowNode { window, io } => {
                    let declared = io.inputs.first().and_then(|t| match &t.time {
                        TimeRef::Sensor { id, .. } => self.temporal.history.get(id),
                        _ => None,
                    });
                    if let Some(h) = declared {
                        if h.depth < window.required_depth() {
                            diags.push(
                                Diagnostic::new(
                                    codes::OBS_042,
                                    format!(
                                        "window reaches back {} samples, History keeps {}",
                                        window.required_depth(),
                                        h.depth
                                    ),
                                )
                                .at(*id),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        for (name, out) in &self.outputs {
            let declared = self
                .graph
                .nodes
                .get(&out.port.node)
                .map(IrNode::outputs)
                .and_then(|ports| ports.into_iter().find(|p| p.name == out.port.port));
            match declared {
                None => diags.push(
                    Diagnostic::new(
                        codes::OBS_043,
                        format!("output \"{name}\" names no node port"),
                    )
                    .at(out.port.node)
                    .on_port(out.port.port.clone()),
                ),
                Some(p) if p.ty != out.ty => diags.push(
                    Diagnostic::new(
                        codes::TYPE_020,
                        format!("output \"{name}\" does not match the node's declared type"),
                    )
                    .at(out.port.node)
                    .on_port(out.port.port.clone()),
                ),
                Some(_) => {}
            }
        }
        diags
    }

    /// Semantic identity (spec 5.3): the graph's canonical hash mixed with the fields that are
    /// not in the graph. Node ids do not reach it, so a relabelling leaves it unchanged.
    pub fn observation_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let graph = canonical_hash(&self.graph)?;
        let mut w = CanonWriter::new();
        w.str(OBS_TAG);
        w.u32(self.schema_version);
        w.digest(&self.task_ref);
        w.digest(&graph);
        self.temporal.canonical(&mut w);
        w.seq(self.outputs.len());
        for (name, out) in &self.outputs {
            w.str(name);
            out.ty.canonical(&mut w);
            // Position in the graph boundary, not the node id: relabel-invariant.
            let at = self.graph.outputs.iter().position(|p| *p == out.port);
            w.u32(u32::try_from(at.unwrap_or(usize::MAX)).unwrap_or(u32::MAX));
        }
        w.hash()
    }
}

/// The `ImageSpec` a node puts on its `out` port, or `None` when it carries no image.
fn image_out(
    node: &ObservationNode,
    id: NodeId,
    incoming: Option<ImageSpec>,
    diags: &mut Vec<Diagnostic>,
) -> Option<ImageSpec> {
    let stale = |diags: &mut Vec<Diagnostic>, what: &str| {
        diags.push(Diagnostic {
            severity: Severity::Warning,
            ..Diagnostic::new(
                codes::OBS_034,
                format!("{what} with rescale_intrinsics = false"),
            )
            .at(id)
            .with_hint("set rescale_intrinsics = true unless the stale intrinsics are intended")
        });
    };
    match node {
        ObservationNode::ImageInput { io, .. } => io.output.image,
        ObservationNode::StateInput { .. } | ObservationNode::LanguageInput { .. } => None,
        ObservationNode::Resize {
            width,
            height,
            rescale_intrinsics,
            ..
        } => {
            let spec = incoming?;
            if !rescale_intrinsics {
                stale(diags, &format!("Resize to {width}x{height}"));
            }
            Some(spec.resized(*width, *height, *rescale_intrinsics))
        }
        ObservationNode::Crop {
            mode,
            rescale_intrinsics,
            ..
        } => {
            let spec = incoming?;
            if !rescale_intrinsics {
                stale(diags, "Crop");
            }
            Some(spec.cropped(mode.rect(&spec), *rescale_intrinsics))
        }
        ObservationNode::Pad {
            left,
            top,
            right,
            bottom,
            ..
        } => {
            let spec = incoming?;
            // Padding grows the canvas and moves the principal point with the old origin.
            Some(ImageSpec {
                width: spec.width + left + right,
                height: spec.height + top + bottom,
                intrinsics: Intrinsics {
                    cx: spec.intrinsics.cx + f64::from(*left),
                    cy: spec.intrinsics.cy + f64::from(*top),
                    ..spec.intrinsics
                },
                ..spec
            })
        }
        ObservationNode::Undistort { rectified, .. }
        | ObservationNode::Rectify { rectified, .. } => Some(incoming?.undistorted(*rectified)),
        ObservationNode::CameraProjection { intrinsics, .. } => {
            let spec = incoming?;
            let declared = ImageSpec {
                intrinsics: *intrinsics,
                ..spec
            };
            if !declared.intrinsics_consistent_with(&spec) {
                diags.push(
                    Diagnostic::new(
                        codes::OBS_034,
                        format!(
                            "CameraProjection declares fx={} cx={}, the {}x{} input carries fx={} cx={}",
                            intrinsics.fx,
                            intrinsics.cx,
                            spec.width,
                            spec.height,
                            spec.intrinsics.fx,
                            spec.intrinsics.cx
                        ),
                    )
                    .at(id)
                    .with_hint("rescale the intrinsics through the Resize / Crop chain feeding this node"),
                );
            }
            None
        }
        ObservationNode::ColorTransform { src, dst, .. } => {
            let spec = incoming?;
            if spec.color_space != *src {
                diags.push(
                    Diagnostic::new(
                        codes::OBS_021,
                        format!("node reads {src:?}, the input is {:?}", spec.color_space),
                    )
                    .at(id),
                );
            }
            Some(ImageSpec {
                color_space: *dst,
                ..spec
            })
        }
        ObservationNode::ToGray { .. } => Some(ImageSpec {
            channels: ChannelFormat::Gray,
            ..incoming?
        }),
        ObservationNode::Normalize { .. } | ObservationNode::Dequantize { .. } => Some(ImageSpec {
            dtype: ImageDType::F32,
            ..incoming?
        }),
        ObservationNode::QuantizeU8 { .. } => Some(ImageSpec {
            dtype: ImageDType::U8,
            ..incoming?
        }),
        // Channel and time restructuring, and `Warp` (a homography is applied in pixel space):
        // the calibrated camera the image came from is untouched.
        _ => incoming,
    }
}

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    //! Fixtures and the proptest generator for the Appendix B.7 properties (P25).

    // `use super::*` is how a test-support module reads; clippy only exempts `#[cfg(test)]`
    // ones automatically, and this one is also reachable through `feature = "testing"`.
    #![allow(clippy::wildcard_imports)]

    use std::time::Duration;

    use es_math::conventions::Pose;
    use proptest::prelude::*;

    use super::*;
    use crate::image::{CameraModel, DistortionModel, ShutterModel};
    use crate::types::{ElemType, Frame, Shape, Unit};

    /// A 640x480 sRGB pinhole camera: `fx = fy = 600`, `cx = 320`, `cy = 240`.
    pub fn cam() -> ImageSpec {
        ImageSpec {
            width: 640,
            height: 480,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(600.0, 600.0, 320.0, 240.0),
            extrinsics: Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_micros(500),
            rate_hz: 30.0,
            depth_scale: None,
        }
    }

    /// Small valid graphs: one camera, one resize that rescales, one normalize.
    pub fn arbitrary_observation_ir() -> impl Strategy<Value = ObservationIr> {
        (
            (64u32..=1024),
            (64u32..=1024),
            (100.0f64..900.0),
            (16u32..=256),
            (16u32..=256),
        )
            .prop_map(|(w, h, f, tw, th)| build(w, h, f, tw, th))
    }

    fn build(w: u32, h: u32, f: f64, tw: u32, th: u32) -> ObservationIr {
        let sensor = StableId::from_path("cam_front");
        let src = ImageSpec {
            intrinsics: Intrinsics::new(f, f, f64::from(w) / 2.0, f64::from(h) / 2.0),
            width: w,
            height: h,
            ..cam()
        };
        let resized = src.resized(tw, th, true);
        let normalized = ImageSpec {
            dtype: ImageDType::F32,
            ..resized
        };

        let img = |spec: ImageSpec, unit: Unit| PortType {
            elem: ElemType::F32,
            shape: Shape::new([3, u64::from(spec.height), u64::from(spec.width)]),
            unit,
            frame: Frame::Camera(sensor),
            time: TimeRef::Sensor {
                id: sensor,
                align: Align::Hold,
            },
            image: Some(spec),
        };
        let t0 = img(src, Unit::Pixel);
        let t1 = img(resized, Unit::Pixel);
        let t2 = img(normalized, Unit::Normalized { lo: 0.0, hi: 1.0 });

        let mut ir = ObservationIr::new(1, [7u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::ImageInput {
                sensor,
                io: Io::source(t0.clone()),
            },
        );
        ir.graph.insert(
            NodeId(1),
            ObservationNode::Resize {
                width: tw,
                height: th,
                filter: ResizeFilter::Bilinear,
                rescale_intrinsics: true,
                io: Io::unary(t0, t1.clone()),
            },
        );
        ir.graph.insert(
            NodeId(2),
            ObservationNode::Normalize {
                stats: NormalizeStats::Range { lo: 0.0, hi: 1.0 },
                io: Io::unary(t1, t2.clone()),
            },
        );
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
        ir.graph.outputs.push(PortRef::new(NodeId(2), OUT));
        ir.outputs.insert(
            "rgb_front".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(2), OUT),
                ty: t2,
            },
        );
        ir
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::testing::{arbitrary_observation_ir, cam};
    use super::*;
    use crate::types::{ElemType, Frame, Shape, Unit};

    fn sensor() -> StableId {
        StableId::from_path("cam_front")
    }

    fn image_ty(spec: ImageSpec) -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([3, u64::from(spec.height), u64::from(spec.width)]),
            unit: Unit::Pixel,
            frame: Frame::Camera(sensor()),
            time: TimeRef::Sensor {
                id: sensor(),
                align: Align::Hold,
            },
            image: Some(spec),
        }
    }

    fn state_ty(id: StableId) -> PortType {
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([7]),
            unit: Unit::Angle,
            frame: Frame::World,
            time: TimeRef::Sensor {
                id,
                align: Align::Hold,
            },
            image: None,
        }
    }

    /// `ImageInput(640x480)` -> `Resize(224, 224)` -> `Crop(16, 16, 192, 192)`.
    fn resize_crop_chain(rescale: bool) -> ObservationIr {
        let src = cam();
        let resized = src.resized(224, 224, rescale);
        let rect = Rect {
            x: 16,
            y: 16,
            width: 192,
            height: 192,
        };
        let cropped = resized.cropped(rect, rescale);

        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::ImageInput {
                sensor: sensor(),
                io: Io::source(image_ty(src)),
            },
        );
        ir.graph.insert(
            NodeId(1),
            ObservationNode::Resize {
                width: 224,
                height: 224,
                filter: ResizeFilter::Bilinear,
                rescale_intrinsics: rescale,
                io: Io::unary(image_ty(src), image_ty(resized)),
            },
        );
        ir.graph.insert(
            NodeId(2),
            ObservationNode::Crop {
                mode: CropMode::Rect(rect),
                rescale_intrinsics: rescale,
                io: Io::unary(image_ty(resized), image_ty(cropped)),
            },
        );
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
        ir.graph.outputs.push(PortRef::new(NodeId(2), OUT));
        ir.outputs.insert(
            "rgb_front".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(2), OUT),
                ty: image_ty(cropped),
            },
        );
        ir
    }

    #[track_caller]
    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn resize_then_crop_propagates_intrinsics() {
        let ir = resize_crop_chain(true);
        let specs = ir.propagate_image_specs().expect("no errors");

        let resized = specs[&PortRef::new(NodeId(1), OUT)];
        assert_eq!((resized.width, resized.height), (224, 224));
        assert_close(resized.intrinsics.fx, 210.0); // 600 * 224 / 640
        assert_close(resized.intrinsics.fy, 280.0); // 600 * 224 / 480
        assert_close(resized.intrinsics.cx, 112.0); // 320 * 224 / 640
        assert_close(resized.intrinsics.cy, 112.0); // 240 * 224 / 480

        let cropped = specs[&PortRef::new(NodeId(2), OUT)];
        assert_eq!((cropped.width, cropped.height), (192, 192));
        assert_close(cropped.intrinsics.fx, 210.0);
        assert_close(cropped.intrinsics.fy, 280.0);
        assert_close(cropped.intrinsics.cx, 96.0); // 112 - 16
        assert_close(cropped.intrinsics.cy, 96.0);

        assert!(ir.validate().is_empty(), "{:?}", ir.validate());
    }

    #[test]
    fn rescale_false_is_an_obs_034_warning() {
        let ir = resize_crop_chain(false);
        let diags = ir.validate();
        let obs: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_str() == codes::OBS_034)
            .collect();
        assert_eq!(obs.len(), 2, "{diags:?}"); // the Resize and the Crop
        assert!(obs.iter().all(|d| d.severity == Severity::Warning));
        // An explicit opt-out is legal (INV-14), so propagation still succeeds.
        let specs = ir.propagate_image_specs().expect("warning only");
        assert_eq!(
            specs[&PortRef::new(NodeId(2), OUT)].intrinsics,
            cam().intrinsics
        );
    }

    #[test]
    fn camera_projection_rejects_stale_intrinsics() {
        let mut ir = resize_crop_chain(true);
        let cropped = ir.propagate_image_specs().unwrap()[&PortRef::new(NodeId(2), OUT)];
        // The classic bug of spec 7.2: the node still declares the original 640x480 intrinsics.
        let point = PortType {
            elem: ElemType::F32,
            shape: Shape::new([3]),
            unit: Unit::Length,
            frame: Frame::Camera(sensor()),
            time: TimeRef::Sensor {
                id: sensor(),
                align: Align::Hold,
            },
            image: None,
        };
        ir.graph.insert(
            NodeId(3),
            ObservationNode::CameraProjection {
                intrinsics: cam().intrinsics,
                io: Io::unary(image_ty(cropped), point.clone()),
            },
        );
        ir.graph.connect(NodeId(2), OUT, NodeId(3), &in_port(0));
        let err = ir.propagate_image_specs().unwrap_err();
        assert!(
            err.iter()
                .any(|d| d.code.as_str() == codes::OBS_034 && d.is_error()),
            "{err:?}"
        );

        // Declaring what actually arrives clears it.
        ir.graph.insert(
            NodeId(3),
            ObservationNode::CameraProjection {
                intrinsics: cropped.intrinsics,
                io: Io::unary(image_ty(cropped), point),
            },
        );
        assert!(ir.propagate_image_specs().is_ok());
    }

    #[test]
    fn concat_of_two_clocks_needs_an_explicit_align() {
        let encoder = StableId::from_path("encoder");
        let a = state_ty(encoder);
        let b = state_ty(sensor());
        let mut out = a.clone();
        out.shape = Shape::new([14]);

        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::Concat {
                axis: 0,
                time_align: None,
                io: Io::new(vec![a.clone(), b.clone()], out.clone()),
            },
        );
        let diags = ir.validate();
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert_eq!(diags[0].code.as_str(), codes::TYPE_014);
        assert_eq!(diags[0].node, Some(NodeId(0)));

        ir.graph.insert(
            NodeId(0),
            ObservationNode::Concat {
                axis: 0,
                time_align: Some(Align::Hold),
                io: Io::new(vec![a, b], out),
            },
        );
        assert!(ir.validate().is_empty(), "{:?}", ir.validate());
    }

    #[test]
    fn normalize_must_produce_a_normalized_unit() {
        let raw = state_ty(sensor());
        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::Normalize {
                stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
                io: Io::unary(raw.clone(), raw.clone()),
            },
        );
        assert_eq!(ir.validate()[0].code.as_str(), codes::OBS_040);

        let mut good = raw.clone();
        good.unit = Unit::Normalized { lo: -1.0, hi: 1.0 };
        ir.graph.insert(
            NodeId(0),
            ObservationNode::Normalize {
                stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
                io: Io::unary(raw, good),
            },
        );
        assert!(ir.validate().is_empty());
    }

    #[test]
    fn augmentation_must_be_training_only() {
        let ty = image_ty(cam());
        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::Augment {
                kind: AugmentKind::GaussianNoise { sigma: 0.01 },
                training_only: false,
                io: Io::unary(ty.clone(), ty),
            },
        );
        assert_eq!(ir.validate()[0].code.as_str(), codes::OBS_041);
    }

    #[test]
    fn history_must_cover_the_window() {
        let ty = state_ty(sensor());
        let window = TemporalWindow {
            n_steps: 4,
            stride: 3,
            align: Align::Hold,
        };
        assert_eq!(window.required_depth(), 10);

        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.temporal.history.insert(sensor(), History { depth: 4 });
        ir.graph.insert(
            NodeId(0),
            ObservationNode::TemporalWindowNode {
                window,
                io: Io::unary(ty.clone(), ty),
            },
        );
        assert_eq!(ir.validate()[0].code.as_str(), codes::OBS_042);

        ir.temporal.history.insert(sensor(), History { depth: 10 });
        assert!(ir.validate().is_empty());
    }

    #[test]
    fn color_transform_checks_the_input_color_space() {
        let src = cam(); // sRGB
        let linear = ImageSpec {
            color_space: ColorSpace::Linear,
            ..src
        };
        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::ImageInput {
                sensor: sensor(),
                io: Io::source(image_ty(src)),
            },
        );
        ir.graph.insert(
            NodeId(1),
            ObservationNode::ColorTransform {
                src: ColorSpace::Linear,
                dst: ColorSpace::SRgb,
                io: Io::unary(image_ty(src), image_ty(linear)),
            },
        );
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        let err = ir.propagate_image_specs().unwrap_err();
        assert_eq!(err[0].code.as_str(), codes::OBS_021);
    }

    #[test]
    fn an_output_that_names_no_port_is_obs_043() {
        let mut ir = resize_crop_chain(true);
        ir.outputs.get_mut("rgb_front").unwrap().port.port = "nope".to_owned();
        assert!(ir
            .validate()
            .iter()
            .any(|d| d.code.as_str() == codes::OBS_043));
    }

    #[test]
    fn observation_hash_is_relabel_invariant() {
        let ir = resize_crop_chain(true);
        let base = ir.observation_hash().unwrap();

        let mut relabelled = ObservationIr::new(1, [0u8; 32]);
        relabelled.temporal = ir.temporal.clone();
        let shift = |id: NodeId| NodeId(id.0 + 100);
        for (id, node) in &ir.graph.nodes {
            relabelled.graph.insert(shift(*id), node.clone());
        }
        for e in &ir.graph.edges {
            relabelled.graph.connect(
                shift(e.from.node),
                &e.from.port,
                shift(e.to.node),
                &e.to.port,
            );
        }
        relabelled.graph.outputs = ir
            .graph
            .outputs
            .iter()
            .map(|p| PortRef::new(shift(p.node), &p.port))
            .collect();
        relabelled.outputs = ir
            .outputs
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    ObservationOutput {
                        port: PortRef::new(shift(v.port.node), &v.port.port),
                        ty: v.ty.clone(),
                    },
                )
            })
            .collect();
        assert_eq!(relabelled.observation_hash().unwrap(), base);

        // A parameter change is not invariant.
        let mut other = ir.clone();
        other.graph.insert(
            NodeId(1),
            ObservationNode::Resize {
                width: 256,
                height: 256,
                filter: ResizeFilter::Bilinear,
                rescale_intrinsics: true,
                io: Io::unary(image_ty(cam()), image_ty(cam().resized(256, 256, true))),
            },
        );
        assert_ne!(other.observation_hash().unwrap(), base);

        // So is the task it binds to (spec 7.4).
        let mut other = ir.clone();
        other.task_ref = [1u8; 32];
        assert_ne!(other.observation_hash().unwrap(), base);
    }

    #[test]
    fn serde_round_trip() {
        let mut ir = resize_crop_chain(true);
        ir.temporal.history.insert(sensor(), History { depth: 4 });
        ir.temporal.window = Some(TemporalWindow {
            n_steps: 2,
            stride: 2,
            align: Align::Hold,
        });
        let json = serde_json::to_string(&ir).unwrap();
        assert_eq!(serde_json::from_str::<ObservationIr>(&json).unwrap(), ir);
    }

    proptest! {
        #[test]
        fn arbitrary_irs_are_valid(ir in arbitrary_observation_ir()) {
            prop_assert!(ir.validate().is_empty(), "{:?}", ir.validate());
            prop_assert!(ir.propagate_image_specs().is_ok());
            prop_assert!(ir.observation_hash().is_ok());
        }
    }
}
