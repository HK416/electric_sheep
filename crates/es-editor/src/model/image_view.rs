//! Before/after images (spec 23.3): what the camera produced next to what the policy sees.
//!
//! *"Showing the pre- and post-preprocessing image side by side is the practically powerful
//! one — half of vision debugging is seeing the difference with your own eyes"* (spec 23.3).
//! The "after" is not a re-implementation of the pipeline: it is [`es_compile::CpuPlan`], the
//! same reference execution the compiler is judged against (spec 11.3), run on one input.

use std::collections::BTreeMap;

use es_compile::{CpuPlan, PlanMode, Tensor, TensorRef};
use es_ir::observation::{ObservationIr, ObservationNode};
use es_ir::types::ElemType;

/// Packed RGB8, row-major — what an egui texture wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePair {
    /// The Observation IR output name (what Learning IR binds to).
    pub name: String,
    pub before: Rgb8Image,
    pub after: Rgb8Image,
}

#[derive(Debug)]
pub enum ImageError {
    /// The Observation IR did not compile; the rendered diagnostics.
    Compile(String),
    Exec(es_compile::ExecError),
    /// The tab runs one input through the plan; this graph wants a different number.
    InputCount(usize),
    /// A tensor whose shape is not an image this view can draw.
    NotAnImage {
        name: String,
        shape: Vec<u64>,
    },
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(s) => write!(f, "observation IR does not compile:\n{s}"),
            Self::Exec(e) => write!(f, "plan failed: {e}"),
            Self::InputCount(n) => {
                write!(f, "the before/after view runs one input, plan takes {n}")
            }
            Self::NotAnImage { name, shape } => {
                write!(f, "output \"{name}\" has shape {shape:?}, not an image")
            }
        }
    }
}

impl std::error::Error for ImageError {}

#[derive(Debug)]
pub struct BeforeAfter;

impl BeforeAfter {
    /// Compile `obs`, run it over one sensor frame, and pair the raw frame with every image
    /// the pipeline produces.
    ///
    /// Non-image outputs (a joint-state vector) are skipped rather than reported: a graph
    /// legitimately mixes both, and this tab is about the image half.
    pub fn run(obs: &ObservationIr, input: &Tensor) -> Result<Vec<ImagePair>, ImageError> {
        let mut plan = CpuPlan::compile(obs, PlanMode::Debug).map_err(|d| {
            ImageError::Compile(
                d.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        })?;
        if plan.inputs.len() != 1 {
            return Err(ImageError::InputCount(plan.inputs.len()));
        }
        let name = plan.inputs.keys().next().expect("checked above").clone();
        let inputs = BTreeMap::from([(name, input.as_ref())]);
        let outputs = plan.run(&inputs).map_err(ImageError::Exec)?;

        let before = to_rgb8(&input.as_ref()).ok_or_else(|| ImageError::NotAnImage {
            name: "input".to_owned(),
            shape: input.shape.clone(),
        })?;
        Ok(outputs
            .iter()
            .filter_map(|(name, t)| {
                Some(ImagePair {
                    name: name.clone(),
                    before: before.clone(),
                    after: to_rgb8(&t.as_ref())?,
                })
            })
            .collect())
    }

    /// [`run`](Self::run) over a deterministic gradient shaped like the graph's own image
    /// input. A bundle on disk carries no sample frame — until a telemetry image stream is
    /// attached (spec 23.3), this is what the tab shows.
    pub fn sample(obs: &ObservationIr) -> Result<Vec<ImagePair>, ImageError> {
        let input = sample_input(obs).ok_or(ImageError::InputCount(0))?;
        Self::run(obs, &input)
    }
}

/// A gradient of the dtype and shape the graph's first `ImageInput` declares.
fn sample_input(obs: &ObservationIr) -> Option<Tensor> {
    let ty = obs.graph.nodes.values().find_map(|n| match n {
        ObservationNode::ImageInput { .. } => Some(n.io().output.clone()),
        _ => None,
    })?;
    let elems: u64 = ty.shape.dims().iter().product();
    let elems = elems as usize;
    let data = match ty.elem {
        ElemType::U8 => (0..elems).map(|i| (i % 256) as u8).collect(),
        ElemType::F32 => (0..elems)
            .flat_map(|i| ((i % 256) as f32 / 255.0).to_le_bytes())
            .collect(),
        _ => return None,
    };
    Some(Tensor {
        dtype: ty.elem,
        shape: ty.shape.dims().to_vec(),
        data,
    })
}

/// Decode a plan tensor to RGB8 for display.
///
/// Layout follows what the CPU kernels define (`docs/design/observation-lowering.md`): a `u8`
/// tensor is the sensor's `HWC` frame, everything else is the pipeline's `CHW` f32. Float
/// values outside `[0, 1]` — a `Normalize` output, most of all — are linearly rescaled to the
/// tensor's own min/max, so a normalized image is visible rather than clipped to black.
fn to_rgb8(tensor: &TensorRef<'_>) -> Option<Rgb8Image> {
    if tensor.dtype == ElemType::U8 {
        let [rows, cols, chans] = <[u64; 3]>::try_from(tensor.shape.clone()).ok()?;
        let (rows, cols, chans) = (rows as usize, cols as usize, chans as usize);
        if !(chans == 1 || chans == 3 || chans == 4) || tensor.data.len() != rows * cols * chans {
            return None;
        }
        let mut data = Vec::with_capacity(rows * cols * 3);
        // Grey replicates, RGB(A) drops alpha.
        for px in tensor.data.chunks_exact(chans) {
            data.extend_from_slice(&[px[0], px[(chans - 1).min(1)], px[(chans - 1).min(2)]]);
        }
        return Some(Rgb8Image {
            width: cols,
            height: rows,
            data,
        });
    }

    let [chans, rows, cols] = <[u64; 3]>::try_from(tensor.shape.clone()).ok()?;
    let (chans, rows, cols) = (chans as usize, rows as usize, cols as usize);
    if !(chans == 1 || chans == 3 || chans == 4) {
        return None;
    }
    let plane = rows * cols;
    let values = decode_f32(tensor, chans * plane)?;
    let (lo, hi) = values.iter().fold((f32::MAX, f32::MIN), |(lo, hi), value| {
        (lo.min(*value), hi.max(*value))
    });
    // Already display-ranged: keep the absolute scale so 0.5 grey stays 0.5 grey.
    let (lo, hi) = if lo >= 0.0 && hi <= 1.0 {
        (0.0, 1.0)
    } else {
        (lo, hi)
    };
    let span = if hi > lo { hi - lo } else { 1.0 };
    let mut data = Vec::with_capacity(plane * 3);
    for pixel in 0..plane {
        for chan in 0..3 {
            let value = values[chan.min(chans - 1) * plane + pixel];
            data.push((((value - lo) / span).clamp(0.0, 1.0) * 255.0) as u8);
        }
    }
    Some(Rgb8Image {
        width: cols,
        height: rows,
        data,
    })
}

fn decode_f32(tensor: &TensorRef<'_>, elems: usize) -> Option<Vec<f32>> {
    let width = match tensor.dtype {
        ElemType::F32 => 4,
        ElemType::F16 | ElemType::Bf16 => 2,
        _ => return None,
    };
    if tensor.data.len() != elems * width {
        return None;
    }
    Some(
        tensor
            .data
            .chunks_exact(width)
            .map(|bytes| match tensor.dtype {
                ElemType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                ElemType::F16 => f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
                // bf16 is the top 16 bits of the f32.
                _ => f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16),
            })
            .collect(),
    )
}

/// IEEE half to single. Display only — not a numerics path (spec 3.4 keeps those in
/// `es-math`).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let frac = u32::from(bits & 0x3ff);
    let word = match exp {
        0 if frac == 0 => 0,
        0 => return f32::from_bits(sign | (frac << 13)) * 2.0f32.powi(-14) * 1024.0,
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(word)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use es_ir::graph::{NodeId, PortRef};
    use es_ir::image::{
        CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
        ShutterModel,
    };
    use es_ir::observation::{in_port, Io, ObservationOutput, ResizeFilter, OUT};
    use es_ir::types::{Align, Frame, PortType, Shape, TimeRef, Unit};

    use super::*;

    fn sensor() -> es_core::StableId {
        es_core::StableId::from_path("cam_front")
    }

    fn spec_8x6() -> ImageSpec {
        ImageSpec {
            width: 8,
            height: 6,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(6.0, 6.0, 4.0, 3.0),
            extrinsics: es_math::conventions::Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_micros(500),
            rate_hz: 30.0,
            depth_scale: None,
        }
    }

    fn ty(elem: ElemType, shape: [u64; 3], spec: ImageSpec) -> PortType {
        PortType {
            elem,
            shape: Shape::new(shape),
            unit: Unit::Pixel,
            frame: Frame::Camera(sensor()),
            time: TimeRef::Sensor {
                id: sensor(),
                align: Align::Hold,
            },
            image: Some(spec),
        }
    }

    /// The 8x6 RGB u8 HWC gradient of the `es-compile` oracle.
    fn gradient_8x6() -> Vec<u8> {
        let mut px = Vec::with_capacity(8 * 6 * 3);
        for y in 0..6u32 {
            for x in 0..8u32 {
                px.push((x * 32) as u8);
                px.push((y * 40) as u8);
                px.push(((x + y) * 16) as u8);
            }
        }
        px
    }

    /// `ImageInput(8x6 u8) -> Dequantize -> Resize(4x3)`.
    fn chain() -> ObservationIr {
        let src = spec_8x6();
        let deq = ImageSpec {
            dtype: ImageDType::F32,
            ..src
        };
        let small = deq.resized(4, 3, true);
        let t_in = ty(ElemType::U8, [6, 8, 3], src);
        let t_deq = ty(ElemType::F32, [3, 6, 8], deq);
        let t_small = ty(ElemType::F32, [3, 3, 4], small);

        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(
            NodeId(0),
            ObservationNode::ImageInput {
                sensor: sensor(),
                io: Io::source(t_in.clone()),
            },
        );
        ir.graph.insert(
            NodeId(1),
            ObservationNode::Dequantize {
                io: Io::unary(t_in, t_deq.clone()),
            },
        );
        ir.graph.insert(
            NodeId(2),
            ObservationNode::Resize {
                width: 4,
                height: 3,
                filter: ResizeFilter::Bilinear,
                rescale_intrinsics: true,
                io: Io::unary(t_deq, t_small.clone()),
            },
        );
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
        ir.graph.outputs.push(PortRef::new(NodeId(2), OUT));
        ir.outputs.insert(
            "rgb_front".to_owned(),
            ObservationOutput {
                port: PortRef::new(NodeId(2), OUT),
                ty: t_small,
            },
        );
        ir
    }

    #[test]
    fn gradient_through_a_resize_gives_a_before_and_an_after() {
        let ir = chain();
        assert!(ir.validate().is_empty(), "{:?}", ir.validate());
        let input = Tensor {
            dtype: ElemType::U8,
            shape: vec![6, 8, 3],
            data: gradient_8x6(),
        };
        let pairs = BeforeAfter::run(&ir, &input).expect("runs");
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.name, "rgb_front");
        assert_eq!((p.before.width, p.before.height), (8, 6));
        assert_eq!(p.before.data.len(), 8 * 6 * 3);
        // The raw frame passes through untouched; the pipeline output is the resized one.
        assert_eq!(&p.before.data[..3], &gradient_8x6()[..3]);
        assert_eq!((p.after.width, p.after.height), (4, 3));
        assert_eq!(p.after.data.len(), 4 * 3 * 3);
        // Dequantize keeps values in [0, 1], so the display scale is absolute: the top-left
        // pixel's red channel is still near black and its blue is brighter.
        assert!(p.after.data[0] < 64, "{:?}", &p.after.data[..3]);
    }

    #[test]
    fn a_bundle_without_a_frame_still_has_something_to_show() {
        let pairs = BeforeAfter::sample(&chain()).expect("runs on a synthetic gradient");
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].before.width, pairs[0].before.height), (8, 6));
    }

    #[test]
    fn a_non_image_tensor_is_not_drawn() {
        let t = Tensor {
            dtype: ElemType::F32,
            shape: vec![8],
            data: vec![0; 32],
        };
        assert!(to_rgb8(&t.as_ref()).is_none());
    }

    #[test]
    fn values_outside_the_display_range_are_rescaled_not_clipped() {
        // One 1x2 single-channel plane spanning [-2, 2]: without rescaling both pixels clip.
        let data: Vec<u8> = [-2.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let t = Tensor {
            dtype: ElemType::F32,
            shape: vec![1, 1, 2],
            data,
        };
        let img = to_rgb8(&t.as_ref()).expect("is an image");
        assert_eq!(img.data[0], 0);
        assert_eq!(img.data[3], 255);
    }
}
