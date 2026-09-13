//! Oracle for the Observation IR CPU reference plan (spec 11.3, spec 7.7).
//!
//! Two halves: byte-for-byte agreement with `tests/golden/observation/**`, which is CI
//! read-only (spec 1.4), and the properties that hold for every input.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use es_compile::{kernels, CpuPlan, PlanMode, TensorRef};
use es_core::StableId;
use es_ir::graph::{NodeId, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
use es_ir::observation::{
    in_port, CropMode, Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput,
    ResizeFilter, OUT,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_math::conventions::Pose;
use proptest::prelude::*;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn golden(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/observation")
        .join(format!("{name}.bin"));
    fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nrun `cargo test -p es-compile --test gen_goldens -- --ignored` once",
            path.display()
        )
    })
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The same 8x6 RGB u8 HWC gradient `gen_goldens` starts from.
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

fn dequantized() -> Vec<f32> {
    let mut chw = vec![0.0f32; 3 * 6 * 8];
    kernels::cast_u8_hwc_to_f32_chw(&gradient_8x6(), 6, 8, 3, &mut chw);
    chw
}

// --- goldens ------------------------------------------------------------------------------

#[test]
fn dequantize_matches_the_golden() {
    assert_eq!(f32_bytes(&dequantized()), golden("dequantize_8x6_rgb"));
}

#[test]
fn resize_matches_the_golden() {
    let mut small = vec![0.0f32; 3 * 3 * 4];
    kernels::resize_bilinear(&dequantized(), 8, 6, 3, 4, 3, &mut small);
    assert_eq!(f32_bytes(&small), golden("resize_bilinear_8x6_to_4x3"));
}

#[test]
fn crop_matches_the_golden() {
    let rect = Rect {
        x: 2,
        y: 1,
        width: 4,
        height: 4,
    };
    let mut cropped = vec![0.0f32; 3 * 4 * 4];
    kernels::crop(&dequantized(), 8, 6, 3, rect, &mut cropped);
    assert_eq!(f32_bytes(&cropped), golden("crop_8x6_at_2_1_4x4"));
}

#[test]
fn srgb_lut_matches_the_golden() {
    assert_eq!(
        f32_bytes(&kernels::srgb_to_linear_lut()),
        golden("srgb_to_linear_lut256")
    );
}

#[test]
fn normalize_matches_the_golden() {
    let mut small = vec![0.0f32; 3 * 3 * 4];
    kernels::resize_bilinear(&dequantized(), 8, 6, 3, 4, 3, &mut small);
    let mut norm = vec![0.0f32; 3 * 3 * 4];
    kernels::normalize_mean_std(&small, 12, &MEAN, &STD, &mut norm);
    assert_eq!(f32_bytes(&norm), golden("normalize_imagenet_4x3"));
}

#[test]
fn history_window_matches_the_golden() {
    let (slot, depth) = (4usize, 4usize);
    let mut ring = vec![0.0f32; slot * depth];
    for i in 0..3usize {
        let f = i as f32;
        kernels::history_push(&mut ring, slot, depth, i, &[f, f + 0.5, f + 1.0, f + 1.5]);
    }
    let mut win = vec![0.0f32; slot * 2];
    kernels::window_gather(&ring, slot, depth, 2, 3, 2, 1, &mut win);
    assert_eq!(f32_bytes(&win), golden("history_window_n2_s1"));
}

// --- the plan, end to end -----------------------------------------------------------------

fn sensor() -> StableId {
    StableId::from_path("cam_front")
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
        extrinsics: Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: ShutterModel::Global,
        exposure: Duration::from_micros(500),
        rate_hz: 30.0,
        depth_scale: None,
    }
}

fn ty(elem: ElemType, shape: [u64; 3], unit: Unit, spec: ImageSpec) -> PortType {
    PortType {
        elem,
        shape: Shape::new(shape),
        unit,
        frame: Frame::Camera(sensor()),
        time: TimeRef::Sensor {
            id: sensor(),
            align: Align::Hold,
        },
        image: Some(spec),
    }
}

/// `ImageInput(8x6 u8 HWC)` -> `Dequantize` -> `Resize(4x3)` -> `Normalize(ImageNet)`.
fn chain() -> ObservationIr {
    let src = spec_8x6();
    let deq = ImageSpec {
        dtype: ImageDType::F32,
        ..src
    };
    let small = deq.resized(4, 3, true);

    let t_in = ty(ElemType::U8, [6, 8, 3], Unit::Pixel, src);
    let t_deq = ty(ElemType::F32, [3, 6, 8], Unit::Pixel, deq);
    let t_small = ty(ElemType::F32, [3, 3, 4], Unit::Pixel, small);
    let t_out = ty(
        ElemType::F32,
        [3, 3, 4],
        Unit::Normalized { lo: -3.0, hi: 3.0 },
        small,
    );

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
    ir.graph.insert(
        NodeId(3),
        ObservationNode::Normalize {
            stats: NormalizeStats::MeanStd {
                mean: MEAN.iter().map(|v| f64::from(*v)).collect(),
                std: STD.iter().map(|v| f64::from(*v)).collect(),
            },
            io: Io::unary(t_small, t_out.clone()),
        },
    );
    for a in 0..3u32 {
        ir.graph.connect(NodeId(a), OUT, NodeId(a + 1), &in_port(0));
    }
    ir.graph.outputs.push(PortRef::new(NodeId(3), OUT));
    ir.outputs.insert(
        "rgb_front".to_owned(),
        ObservationOutput {
            port: PortRef::new(NodeId(3), OUT),
            ty: t_out,
        },
    );
    ir
}

#[test]
fn the_plan_reproduces_the_goldens_end_to_end() {
    let ir = chain();
    assert!(ir.validate().is_empty(), "{:?}", ir.validate());
    let mut plan = CpuPlan::compile(&ir, PlanMode::Debug).expect("compiles");
    assert_eq!(plan.steps.len(), 3); // dequantize, resize, normalize
    assert_eq!(plan.inputs.len(), 1);

    let px = gradient_8x6();
    let mut inputs = BTreeMap::new();
    inputs.insert(
        sensor().to_string(),
        TensorRef::new(ElemType::U8, [6, 8, 3], &px),
    );
    let out = plan.run(&inputs).expect("runs");

    let rgb = &out["rgb_front"];
    assert_eq!(rgb.dtype, ElemType::F32);
    assert_eq!(rgb.shape, vec![3, 3, 4]);
    assert_eq!(rgb.data, golden("normalize_imagenet_4x3"));
}

#[test]
fn compiler_hash_separates_the_two_plans() {
    let ir = chain();
    let debug = CpuPlan::compile(&ir, PlanMode::Debug).unwrap();
    let release = CpuPlan::compile(&ir, PlanMode::Release).unwrap();
    assert_ne!(debug.compiler_hash(), release.compiler_hash());
    // Spec 11.5 on this path: neither plan aliases, so both can show every node's value.
    assert_eq!(debug.arena_elems, release.arena_elems);
    assert!(debug.buffer_of(NodeId(2)).is_some());
}

#[test]
fn an_unsupported_node_is_a_diagnostic_not_a_silent_no_op() {
    let mut ir = chain();
    let small = ImageSpec {
        dtype: ImageDType::F32,
        ..spec_8x6()
    }
    .resized(4, 3, true);
    let gray = ImageSpec {
        channels: ChannelFormat::Gray,
        ..small
    };
    let unit = Unit::Normalized { lo: -3.0, hi: 3.0 };
    let t = ty(ElemType::F32, [3, 3, 4], unit.clone(), small);
    ir.graph.insert(
        NodeId(4),
        ObservationNode::ToGray {
            io: Io::unary(t, ty(ElemType::F32, [1, 3, 4], unit, gray)),
        },
    );
    ir.graph.connect(NodeId(3), OUT, NodeId(4), &in_port(0));
    let errs = CpuPlan::compile(&ir, PlanMode::Debug).unwrap_err();
    assert!(
        errs.iter().any(|d| d.code.as_str() == "COMPILE-002"),
        "{errs:?}"
    );
}

// --- properties ---------------------------------------------------------------------------

proptest! {
    /// A resize of a constant image is that constant — to within 2 ulp.
    ///
    /// No neighbour leaks in at the border, which is the classic half-pixel bug. It is not
    /// *bit*-exact, and deliberately so: the taps are combined as `l0 * a + l1 * b` with
    /// `l0 = 1 - l1`, which is PyTorch's association and does not sum the weights to exactly
    /// 1 in f32. Matching PyTorch bit for bit is the contract (spec 7.7); an exact-on-
    /// constants association would be a different kernel.
    #[test]
    fn resize_of_a_constant_is_constant(
        v in -10.0f32..10.0,
        sw in 1usize..17,
        sh in 1usize..17,
        dw in 1usize..17,
        dh in 1usize..17,
    ) {
        let src = vec![v; 2 * sh * sw];
        let mut dst = vec![f32::NAN; 2 * dh * dw];
        kernels::resize_bilinear(&src, sw, sh, 2, dw, dh, &mut dst);
        let tol = 2.0 * f32::EPSILON * v.abs();
        prop_assert!(dst.iter().all(|x| (*x - v).abs() <= tol), "{dst:?}");
    }

    /// `crop(crop(x, a), b) == crop(x, a o b)` in pixels, and `ImageSpec::cropped` chained
    /// gives the same intrinsics as the combined rectangle (`INV-14`).
    #[test]
    fn crop_of_crop_is_one_crop(
        ax in 0u32..4, ay in 0u32..3, bx in 0u32..2, by in 0u32..2,
        bw in 1u32..3, bh in 1u32..3,
    ) {
        let (w, h, c) = (8usize, 6usize, 3usize);
        let a = Rect { x: ax, y: ay, width: w as u32 - ax, height: h as u32 - ay };
        prop_assume!(bx + bw <= a.width && by + bh <= a.height);
        let b = Rect { x: bx, y: by, width: bw, height: bh };
        let combined = Rect { x: a.x + b.x, y: a.y + b.y, width: b.width, height: b.height };

        let src = dequantized();
        let mut mid = vec![0.0f32; c * a.height as usize * a.width as usize];
        kernels::crop(&src, w, h, c, a, &mut mid);
        let mut twice = vec![0.0f32; c * bh as usize * bw as usize];
        kernels::crop(&mid, a.width as usize, a.height as usize, c, b, &mut twice);
        let mut once = vec![0.0f32; c * bh as usize * bw as usize];
        kernels::crop(&src, w, h, c, combined, &mut once);
        prop_assert_eq!(&twice, &once);

        let spec = spec_8x6();
        let chained = spec.cropped(a, true).cropped(b, true);
        let direct = spec.cropped(combined, true);
        prop_assert_eq!(chained.intrinsics, direct.intrinsics);
        prop_assert_eq!((chained.width, chained.height), (direct.width, direct.height));
    }

    /// Whatever the camera geometry, the chain compiles and runs, and the plan's crop
    /// rectangle is the one the type system used.
    #[test]
    fn the_centre_crop_plan_agrees_with_the_propagated_spec(cw in 1u32..8, ch in 1u32..6) {
        let src = spec_8x6();
        let deq = ImageSpec { dtype: ImageDType::F32, ..src };
        let rect = CropMode::Center { width: cw, height: ch };
        let cropped = deq.cropped(
            Rect { x: (8 - cw) / 2, y: (6 - ch) / 2, width: cw, height: ch },
            true,
        );

        let t_in = ty(ElemType::U8, [6, 8, 3], Unit::Pixel, src);
        let t_deq = ty(ElemType::F32, [3, 6, 8], Unit::Pixel, deq);
        let t_out = ty(
            ElemType::F32,
            [3, u64::from(ch), u64::from(cw)],
            Unit::Pixel,
            cropped,
        );

        let mut ir = ObservationIr::new(1, [0u8; 32]);
        ir.graph.insert(NodeId(0), ObservationNode::ImageInput {
            sensor: sensor(), io: Io::source(t_in.clone()) });
        ir.graph.insert(NodeId(1), ObservationNode::Dequantize {
            io: Io::unary(t_in, t_deq.clone()) });
        ir.graph.insert(NodeId(2), ObservationNode::Crop {
            mode: rect, rescale_intrinsics: true, io: Io::unary(t_deq, t_out.clone()) });
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
        ir.graph.outputs.push(PortRef::new(NodeId(2), OUT));
        ir.outputs.insert("rgb".to_owned(), ObservationOutput {
            port: PortRef::new(NodeId(2), OUT), ty: t_out });

        let mut plan = CpuPlan::compile(&ir, PlanMode::Release).expect("compiles");
        let px = gradient_8x6();
        let mut inputs = BTreeMap::new();
        inputs.insert(sensor().to_string(), TensorRef::new(ElemType::U8, [6, 8, 3], &px));
        let out = plan.run(&inputs).expect("runs");
        prop_assert_eq!(
            out["rgb"].data.len(),
            3 * cw as usize * ch as usize * 4
        );
        prop_assert_eq!(
            ir.propagate_image_specs().unwrap()[&PortRef::new(NodeId(2), OUT)].intrinsics,
            cropped.intrinsics
        );
    }
}
