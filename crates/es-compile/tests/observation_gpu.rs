//! Oracle for the Observation IR **GPU** lowering (spec 7.6, spec 11.4, spec 28.3 W3).
//!
//! The CPU reference plan is the ground truth (spec 11.3), so almost everything here is an
//! equality: the same golden bytes `observation_cpu.rs` pins, the same tolerance the sidecar
//! declares, and `CpuPlan::run == GpuPlan::run` bit for bit where
//! `docs/design/observation-lowering.md` claims bit equality.
//!
//! Every test that needs a device opens one first and prints `SKIP <reason>` when there is
//! none or when `slangc` is missing (spec 1.4: an oracle must be runnable everywhere, even
//! when all it can report is that it did not run). A test that did run prints `RAN`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use es_compile::{CpuPlan, GpuPlan, PlanMode, Tensor, TensorRef};
use es_core::StableId;
use es_gpu::{Gpu, GpuOptions, SlangCompiler};
use es_ir::graph::{NodeId, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    Rect, ShutterModel,
};
use es_ir::observation::{
    in_port, CropMode, History, Io, NormalizeStats, ObservationIr, ObservationNode,
    ObservationOutput, ResizeFilter, TemporalWindow, OUT,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_math::conventions::Pose;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

// --- harness ------------------------------------------------------------------------------

/// `Some(gpu)`, or a printed SKIP and `None`.
fn device(test: &str) -> Option<Gpu> {
    if let Err(e) = SlangCompiler::new() {
        println!("SKIP {test}: no slangc ({e})");
        return None;
    }
    match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            println!("SKIP {test}: no Vulkan device ({e})");
            None
        }
    }
}

/// Runs `body` on a device, or prints SKIP. Saves repeating the guard in every test.
fn on_gpu(test: &str, body: impl FnOnce(&Gpu)) {
    if let Some(gpu) = device(test) {
        body(&gpu);
        println!("RAN {test} on {}", gpu.capabilities().device_name);
    }
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/observation")
}

fn golden(name: &str) -> Vec<u8> {
    let path = golden_dir().join(format!("{name}.bin"));
    fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The golden's own declared tolerance in ULP, or 0. Read from the sidecar, which is CI
/// read-only, exactly as `observation_cpu.rs` reads it.
fn tolerance_ulp(name: &str) -> u32 {
    let path = golden_dir().join(format!("{name}.json"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let sidecar: serde_json::Value = serde_json::from_str(&text).expect("sidecar is JSON");
    sidecar
        .get("tolerance_ulp")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0)
}

fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Worst ULP distance between two finite non-negative f32 slices, and where it is.
fn worst_ulp(got: &[f32], want: &[f32]) -> (u64, usize) {
    let mut worst = (0u64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite() && *g >= 0.0 && *w >= 0.0,
            "[{i}]: {g} vs {w} — the ULP count assumes one sign"
        );
        let d = (i64::from(g.to_bits()) - i64::from(w.to_bits())).unsigned_abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    worst
}

fn assert_golden(name: &str, got: &Tensor) {
    let want = golden(name);
    let tol = tolerance_ulp(name);
    assert_eq!(got.data.len(), want.len(), "{name}: length");
    if tol == 0 {
        assert_eq!(got.data, want, "{name}: not bit-equal to the golden");
        println!("{name}: GPU == golden bit for bit");
        return;
    }
    let (d, at) = worst_ulp(&as_f32(&got.data), &as_f32(&want));
    assert!(d <= u64::from(tol), "{name}[{at}]: {d} ULP > {tol}");
    println!("{name}: GPU within {d} ULP of the golden (sidecar allows {tol})");
}

// --- fixtures -----------------------------------------------------------------------------

fn sensor() -> StableId {
    StableId::from_path("cam_front")
}

fn spec(width: u32, height: u32, channels: ChannelFormat) -> ImageSpec {
    ImageSpec {
        width,
        height,
        channels,
        dtype: ImageDType::U8,
        color_space: ColorSpace::SRgb,
        camera_model: CameraModel::Pinhole,
        intrinsics: Intrinsics::new(6.0, 6.0, f64::from(width) / 2.0, f64::from(height) / 2.0),
        extrinsics: Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: ShutterModel::Global,
        exposure: Duration::from_micros(500),
        rate_hz: 30.0,
        depth_scale: None,
    }
}

fn ty(elem: ElemType, shape: [u64; 3], unit: Unit, image: ImageSpec) -> PortType {
    PortType {
        elem,
        shape: Shape::new(shape),
        unit,
        frame: Frame::Camera(sensor()),
        time: TimeRef::Sensor {
            id: sensor(),
            align: Align::Hold,
        },
        image: Some(image),
    }
}

fn finish(ir: &mut ObservationIr, last: NodeId, ty: PortType) {
    ir.graph.outputs.push(PortRef::new(last, OUT));
    ir.outputs.insert(
        "out".to_owned(),
        ObservationOutput {
            port: PortRef::new(last, OUT),
            ty,
        },
    );
}

/// What an image chain ends with. One builder for four goldens beats four builders.
enum Tail {
    Dequantize,
    Resize {
        w: u32,
        h: u32,
        filter: ResizeFilter,
    },
    Crop(Rect),
    Normalize,
}

/// `Tail::Resize` with the bilinear filter, which is what most cases want.
fn resize(w: u32, h: u32) -> Tail {
    Tail::Resize {
        w,
        h,
        filter: ResizeFilter::Bilinear,
    }
}

/// `ImageInput(w x h u8 HWC)` -> `Dequantize` -> `tail`.
fn image_ir(w: u32, h: u32, tail: &Tail) -> ObservationIr {
    let src = spec(w, h, ChannelFormat::Rgb);
    let deq = ImageSpec {
        dtype: ImageDType::F32,
        ..src
    };
    let t_in = ty(
        ElemType::U8,
        [u64::from(h), u64::from(w), 3],
        Unit::Pixel,
        src,
    );
    let t_deq = ty(
        ElemType::F32,
        [3, u64::from(h), u64::from(w)],
        Unit::Pixel,
        deq,
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
    ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));

    match tail {
        Tail::Dequantize => finish(&mut ir, NodeId(1), t_deq),
        Tail::Resize {
            w: dw,
            h: dh,
            filter,
        } => {
            let small = deq.resized(*dw, *dh, true);
            let t = ty(
                ElemType::F32,
                [3, u64::from(*dh), u64::from(*dw)],
                Unit::Pixel,
                small,
            );
            ir.graph.insert(
                NodeId(2),
                ObservationNode::Resize {
                    width: *dw,
                    height: *dh,
                    filter: *filter,
                    rescale_intrinsics: true,
                    io: Io::unary(t_deq, t.clone()),
                },
            );
            ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
            finish(&mut ir, NodeId(2), t);
        }
        Tail::Crop(rect) => {
            let cropped = deq.cropped(*rect, true);
            let t = ty(
                ElemType::F32,
                [3, u64::from(rect.height), u64::from(rect.width)],
                Unit::Pixel,
                cropped,
            );
            ir.graph.insert(
                NodeId(2),
                ObservationNode::Crop {
                    mode: CropMode::Rect(*rect),
                    rescale_intrinsics: true,
                    io: Io::unary(t_deq, t.clone()),
                },
            );
            ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
            finish(&mut ir, NodeId(2), t);
        }
        Tail::Normalize => {
            let small = deq.resized(4, 3, true);
            let t_small = ty(ElemType::F32, [3, 3, 4], Unit::Pixel, small);
            let t_norm = ty(
                ElemType::F32,
                [3, 3, 4],
                Unit::Normalized { lo: -3.0, hi: 3.0 },
                small,
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
                    io: Io::unary(t_small, t_norm.clone()),
                },
            );
            ir.graph.connect(NodeId(1), OUT, NodeId(2), &in_port(0));
            ir.graph.connect(NodeId(2), OUT, NodeId(3), &in_port(0));
            finish(&mut ir, NodeId(3), t_norm);
        }
    }
    ir
}

/// `ImageInput(256x1 gray u8)` -> [`Dequantize` ->] `ColorTransform{SRgb -> Linear}`.
///
/// Fused, the node reads u8 and indexes the LUT and its output over the pixels `0..=255`
/// *is* the 256-entry table the golden pins. Unfused, it reads f32 and evaluates the EOTF
/// per element through `approx.slang` — the two kernels behind one id (design note section 6).
fn srgb_ir(fused: bool) -> ObservationIr {
    let src = spec(256, 1, ChannelFormat::Gray);
    let linear = ImageSpec {
        color_space: ColorSpace::Linear,
        ..src
    };
    let deq = ImageSpec {
        dtype: ImageDType::F32,
        ..src
    };
    let t_in = ty(ElemType::U8, [1, 256, 1], Unit::Pixel, src);
    let t_deq = ty(ElemType::F32, [1, 1, 256], Unit::Pixel, deq);
    let t_out = ty(ElemType::F32, [1, 1, 256], Unit::Pixel, linear);

    let mut ir = ObservationIr::new(1, [0u8; 32]);
    ir.graph.insert(
        NodeId(0),
        ObservationNode::ImageInput {
            sensor: sensor(),
            io: Io::source(t_in.clone()),
        },
    );
    let (color_in, from) = if fused {
        (t_in, NodeId(0))
    } else {
        ir.graph.insert(
            NodeId(1),
            ObservationNode::Dequantize {
                io: Io::unary(t_in, t_deq.clone()),
            },
        );
        ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
        (t_deq, NodeId(1))
    };
    ir.graph.insert(
        NodeId(2),
        ObservationNode::ColorTransform {
            src: ColorSpace::SRgb,
            dst: ColorSpace::Linear,
            io: Io::unary(color_in, t_out.clone()),
        },
    );
    ir.graph.connect(from, OUT, NodeId(2), &in_port(0));
    finish(&mut ir, NodeId(2), t_out);
    ir
}

/// `StateInput([4])` -> `TemporalWindow(n = 2, stride = 1)` over a depth-4 history: the
/// graph behind `history_window_n2_s1`.
fn window_ir() -> ObservationIr {
    let raw = PortType {
        elem: ElemType::F32,
        shape: Shape::new([4]),
        unit: Unit::Angle,
        frame: Frame::Joint(sensor()),
        time: TimeRef::Sensor {
            id: sensor(),
            align: Align::Hold,
        },
        image: None,
    };
    let win = PortType {
        shape: Shape::new([2, 4]),
        time: TimeRef::Window {
            base: Box::new(raw.time.clone()),
            n: 2,
            stride: 1,
        },
        ..raw.clone()
    };
    let mut ir = ObservationIr::new(1, [0u8; 32]);
    ir.temporal.history.insert(sensor(), History { depth: 4 });
    ir.graph.insert(
        NodeId(0),
        ObservationNode::StateInput {
            source: sensor(),
            io: Io::source(raw.clone()),
        },
    );
    ir.graph.insert(
        NodeId(1),
        ObservationNode::TemporalWindowNode {
            window: TemporalWindow {
                n_steps: 2,
                stride: 1,
                align: Align::Hold,
            },
            io: Io::unary(raw, win.clone()),
        },
    );
    ir.graph.connect(NodeId(0), OUT, NodeId(1), &in_port(0));
    finish(&mut ir, NodeId(1), win);
    ir
}

/// The same 8x6 RGB u8 HWC gradient the goldens start from.
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

/// Deterministic pixels for an arbitrary size: no global RNG (spec 3.4).
fn pixels(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u8
        })
        .collect()
}

fn image_inputs(px: &[u8], w: u32, h: u32) -> BTreeMap<String, TensorRef<'_>> {
    BTreeMap::from([(
        sensor().to_string(),
        TensorRef::new(ElemType::U8, [u64::from(h), u64::from(w), 3], px),
    )])
}

fn run_gpu(gpu: &Gpu, ir: &ObservationIr, inputs: &BTreeMap<String, TensorRef<'_>>) -> Tensor {
    let mut plan = GpuPlan::compile(gpu, ir, PlanMode::Debug).expect("GPU plan compiles");
    plan.run(inputs).expect("GPU plan runs")["out"].clone()
}

// --- goldens ------------------------------------------------------------------------------

#[test]
fn the_gpu_plan_reproduces_every_image_golden() {
    on_gpu("the_gpu_plan_reproduces_every_image_golden", |gpu| {
        let px = gradient_8x6();
        let inputs = image_inputs(&px, 8, 6);
        for (name, tail) in [
            ("dequantize_8x6_rgb", Tail::Dequantize),
            ("resize_bilinear_8x6_to_4x3", resize(4, 3)),
            (
                "crop_8x6_at_2_1_4x4",
                Tail::Crop(Rect {
                    x: 2,
                    y: 1,
                    width: 4,
                    height: 4,
                }),
            ),
            ("normalize_imagenet_4x3", Tail::Normalize),
        ] {
            let ir = image_ir(8, 6, &tail);
            assert!(ir.validate().is_empty(), "{name}: {:?}", ir.validate());
            assert_golden(name, &run_gpu(gpu, &ir, &inputs));
        }
    });
}

/// The one golden with a tolerance. The GPU indexes the *same* uploaded table bytes the CPU
/// builds, so the 7 ULP the sidecar allows is the CPU kernel's distance from the f64
/// reference formula (design note section 6), not a GPU error on top of it.
#[test]
fn the_gpu_srgb_path_reproduces_the_lut_golden() {
    on_gpu("the_gpu_srgb_path_reproduces_the_lut_golden", |gpu| {
        let ir = srgb_ir(true);
        assert!(ir.validate().is_empty(), "{:?}", ir.validate());
        let px: Vec<u8> = (0..=255u8).collect();
        let inputs = BTreeMap::from([(
            sensor().to_string(),
            TensorRef::new(ElemType::U8, [1, 256, 1], px.as_slice()),
        )]);
        assert_golden("srgb_to_linear_lut256", &run_gpu(gpu, &ir, &inputs));

        // The unfused kernel evaluates the EOTF per element instead of indexing the table;
        // it is the CPU's own `approx` path, so the equality is with the CPU, not the golden.
        let unfused = srgb_ir(false);
        assert!(unfused.validate().is_empty(), "{:?}", unfused.validate());
        let mut cpu = CpuPlan::compile(&unfused, PlanMode::Debug).expect("CPU compiles");
        let want = cpu.run(&inputs).expect("CPU runs")["out"].clone();
        let got = run_gpu(gpu, &unfused, &inputs);
        let (d, at) = worst_ulp(&as_f32(&got.data), &as_f32(&want.data));
        println!("srgb_to_linear (elementwise) CPU vs GPU: worst {d} ULP at {at}");
        assert_eq!(got, want, "elementwise sRGB: CPU/GPU differ");
    });
}

#[test]
fn the_gpu_window_reproduces_the_history_golden() {
    on_gpu("the_gpu_window_reproduces_the_history_golden", |gpu| {
        let ir = window_ir();
        let mut plan = GpuPlan::compile(gpu, &ir, PlanMode::Debug).expect("compiles");
        let mut last = None;
        for i in 0..3usize {
            let f = i as f32;
            let bytes: Vec<u8> = [f, f + 0.5, f + 1.0, f + 1.5]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            let inputs = BTreeMap::from([(
                sensor().to_string(),
                TensorRef::new(ElemType::F32, [4], bytes.as_slice()),
            )]);
            last = Some(plan.run(&inputs).expect("runs")["out"].clone());
        }
        assert_golden("history_window_n2_s1", &last.expect("three runs"));
    });
}

// --- CPU / GPU equivalence ------------------------------------------------------------------

/// 50 random `(source, target)` size pairs, CPU plan vs GPU plan, bit for bit.
///
/// Not a `proptest!` block: each case compiles a pipeline and opens a command buffer, so the
/// shrinking machinery would be re-running Vulkan work, and a fixed-seed sweep reports the
/// same thing. A mismatch prints the size and the worst ULP rather than only failing.
#[test]
fn cpu_and_gpu_resize_agree_bit_for_bit() {
    on_gpu("cpu_and_gpu_resize_agree_bit_for_bit", |gpu| {
        let mut seed = 0x5eed_1234_u64;
        let mut next = move |lo: u32, hi: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            lo + (seed >> 33) as u32 % (hi - lo + 1)
        };
        let mut mismatches = Vec::new();
        let mut worst = 0u64;
        for case in 0..50u32 {
            let (sw, sh) = (next(1, 17), next(1, 17));
            let (dw, dh) = (next(1, 17), next(1, 17));
            let ir = image_ir(sw, sh, &resize(dw, dh));
            assert!(ir.validate().is_empty(), "{:?}", ir.validate());

            let px = pixels((sw * sh * 3) as usize, u64::from(case) + 1);
            let inputs = image_inputs(&px, sw, sh);
            let mut cpu = CpuPlan::compile(&ir, PlanMode::Debug).expect("CPU plan compiles");
            let want = cpu.run(&inputs).expect("CPU runs")["out"].clone();
            let got = run_gpu(gpu, &ir, &inputs);

            if got.data != want.data {
                let (d, at) = worst_ulp(&as_f32(&got.data), &as_f32(&want.data));
                worst = worst.max(d);
                mismatches.push(format!("{sw}x{sh}->{dw}x{dh}: {d} ULP at {at}"));
            }
        }
        println!(
            "resize CPU vs GPU: {}/50 sizes bit-equal, worst {worst} ULP",
            50 - mismatches.len()
        );
        assert!(
            mismatches.is_empty(),
            "resize mismatches: {}",
            mismatches.join(", ")
        );
    });
}

/// Every kernel the M1 W3 subset has, in one graph, CPU against GPU.
#[test]
fn cpu_and_gpu_agree_on_the_whole_chain() {
    on_gpu("cpu_and_gpu_agree_on_the_whole_chain", |gpu| {
        let px = gradient_8x6();
        let inputs = image_inputs(&px, 8, 6);
        for tail in [
            Tail::Dequantize,
            resize(4, 3),
            Tail::Resize {
                w: 5,
                h: 5,
                filter: ResizeFilter::Nearest,
            },
            Tail::Crop(Rect {
                x: 2,
                y: 1,
                width: 4,
                height: 4,
            }),
            Tail::Normalize,
        ] {
            let ir = image_ir(8, 6, &tail);
            let mut cpu = CpuPlan::compile(&ir, PlanMode::Debug).expect("compiles");
            let want = cpu.run(&inputs).expect("runs")["out"].clone();
            assert_eq!(run_gpu(gpu, &ir, &inputs), want, "CPU/GPU mismatch");
        }
        println!("chain CPU vs GPU: bit-equal on every tail");
    });
}

/// Spec 3.5 tier 1: the same plan, the same input, twice — the same bits.
#[test]
fn two_gpu_runs_are_bit_identical() {
    on_gpu("two_gpu_runs_are_bit_identical", |gpu| {
        let ir = image_ir(8, 6, &Tail::Normalize);
        let px = gradient_8x6();
        let inputs = image_inputs(&px, 8, 6);
        let mut plan = GpuPlan::compile(gpu, &ir, PlanMode::Debug).expect("compiles");
        let first = plan.run(&inputs).expect("runs")["out"].clone();
        let second = plan.run(&inputs).expect("runs")["out"].clone();
        assert_eq!(first, second, "two runs of one plan differ");
    });
}

/// P-M2-R1 on the device: `reset` must leave the rings the way `compile` did, and mean the
/// same thing the CPU plan's `reset` means.
#[test]
fn gpu_reset_matches_cpu_reset() {
    on_gpu("gpu_reset_matches_cpu_reset", |gpu| {
        let ir = window_ir();
        let frame = |v: f32| -> Vec<u8> {
            [v, v + 0.5, v + 1.0, v + 1.5]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect()
        };
        let run = |plan: &mut GpuPlan<'_>, v: f32| {
            let bytes = frame(v);
            let inputs = BTreeMap::from([(
                sensor().to_string(),
                TensorRef::new(ElemType::F32, [4], bytes.as_slice()),
            )]);
            plan.run(&inputs).expect("runs")["out"].clone()
        };

        let mut plan = GpuPlan::compile(gpu, &ir, PlanMode::Release).expect("compiles");
        for v in [1.0, 3.0, 5.0] {
            let _ = run(&mut plan, v);
        }
        let leaked = run(&mut plan, 7.0);
        plan.reset().expect("reset");
        let after_reset = run(&mut plan, 7.0);

        let mut fresh = GpuPlan::compile(gpu, &ir, PlanMode::Release).expect("compiles");
        let first = run(&mut fresh, 7.0);
        assert_eq!(after_reset, first, "reset must clear every ring");
        assert_ne!(
            leaked, first,
            "the fixture must be history-sensitive, or the test proves nothing"
        );

        // ... and it means what the CPU's reset means.
        let bytes = frame(7.0);
        let inputs = BTreeMap::from([(
            sensor().to_string(),
            TensorRef::new(ElemType::F32, [4], bytes.as_slice()),
        )]);
        let mut cpu = CpuPlan::compile(&ir, PlanMode::Release).expect("compiles");
        assert_eq!(cpu.run(&inputs).expect("runs")["out"], first);
    });
}

/// Editing a `.slang` file must move the hash even when no kernel id did (spec 3.4 item 7),
/// and the GPU hash must not collide with the CPU one.
#[test]
fn the_gpu_compiler_hash_covers_the_spirv() {
    on_gpu("the_gpu_compiler_hash_covers_the_spirv", |gpu| {
        let ir = image_ir(8, 6, &Tail::Normalize);
        let debug = GpuPlan::compile(gpu, &ir, PlanMode::Debug).expect("compiles");
        let release = GpuPlan::compile(gpu, &ir, PlanMode::Release).expect("compiles");
        assert_ne!(debug.compiler_hash(), release.compiler_hash());
        assert_ne!(debug.compiler_hash(), debug.cpu().compiler_hash());
        // The SPIR-V hashes are in there, one per pipeline.
        assert_eq!(debug.pipeline_count(), debug.keys().len());
    });
}

// --- CPU-only -------------------------------------------------------------------------------

/// The pipeline list is a pure function of the plan, so CI without a GPU still gates it: a
/// chain that uses each kernel once must compile one pipeline per kernel id.
#[test]
fn one_pipeline_per_distinct_kernel_id() {
    let ir = image_ir(8, 6, &Tail::Normalize);
    let keys = GpuPlan::pipeline_plan(&ir, PlanMode::Debug).expect("lowers");
    let ids: Vec<&str> = keys.iter().map(|k| k.kernel).collect();
    assert_eq!(
        ids,
        vec![
            "cast_u8_hwc_to_f32_chw.v1",
            "resize_bilinear.v1",
            "normalize_mean_std.v1"
        ]
    );
    assert_eq!(keys.len(), ids.len(), "one pipeline per distinct kernel id");

    // A window graph adds the two temporal ids, and the sRGB graph the colour one; together
    // they cover the M1 W3 subset with no id used twice.
    let win = GpuPlan::pipeline_plan(&window_ir(), PlanMode::Debug).expect("lowers");
    assert_eq!(
        win.iter().map(|k| k.kernel).collect::<Vec<_>>(),
        vec!["history_push.v1", "window_gather.v1"]
    );
    let lut = GpuPlan::pipeline_plan(&srgb_ir(true), PlanMode::Debug).expect("lowers");
    assert_eq!(
        lut.iter().map(|k| k.kernel).collect::<Vec<_>>(),
        vec!["srgb_to_linear.v1"]
    );
}

/// Two resizes to the same size share one pipeline; to different sizes they do not. The
/// defines *are* the specialisation (spec 2.3), so this is what "specialised by defines"
/// has to mean.
#[test]
fn defines_decide_pipeline_sharing() {
    let a =
        GpuPlan::pipeline_plan(&image_ir(8, 6, &resize(4, 3)), PlanMode::Debug).expect("lowers");
    let b =
        GpuPlan::pipeline_plan(&image_ir(8, 6, &resize(4, 3)), PlanMode::Debug).expect("lowers");
    let c =
        GpuPlan::pipeline_plan(&image_ir(8, 6, &resize(2, 2)), PlanMode::Debug).expect("lowers");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// `StateInput([2]) x2 -> Concat|Stack -> Normalize{Range}`, optionally narrowed at the
/// output. The kernels the image chain never reaches: `concat`, `stack`, `normalize_range`
/// and the two f16/bf16 casts.
fn join_ir(stack: bool, elem: ElemType) -> ObservationIr {
    let arm = |n: u32| {
        let id = StableId::from_path(&format!("joint{n}"));
        PortType {
            elem: ElemType::F32,
            shape: Shape::new([2]),
            unit: Unit::Angle,
            frame: Frame::Joint(id),
            time: TimeRef::Sensor {
                id,
                align: Align::Hold,
            },
            image: None,
        }
    };
    let joined = PortType {
        shape: Shape::new(if stack { vec![2u64, 2] } else { vec![4u64] }),
        ..arm(0)
    };
    let out = PortType {
        elem,
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        ..joined.clone()
    };

    let mut ir = ObservationIr::new(1, [0u8; 32]);
    for n in 0..2u32 {
        ir.graph.insert(
            NodeId(n),
            ObservationNode::StateInput {
                source: StableId::from_path(&format!("joint{n}")),
                io: Io::source(arm(n)),
            },
        );
        ir.graph
            .connect(NodeId(n), OUT, NodeId(2), &in_port(n as usize));
    }
    let io = Io::new(vec![arm(0), arm(1)], joined.clone());
    ir.graph.insert(
        NodeId(2),
        if stack {
            ObservationNode::Stack { axis: 0, io }
        } else {
            ObservationNode::Concat {
                axis: 0,
                // Two sensor clocks meet here; the IR insists the author says how (TYPE-014).
                time_align: Some(Align::Hold),
                io,
            }
        },
    );
    ir.graph.insert(
        NodeId(3),
        ObservationNode::Normalize {
            stats: NormalizeStats::Range { lo: -1.0, hi: 1.0 },
            io: Io::unary(joined, out.clone()),
        },
    );
    ir.graph.connect(NodeId(2), OUT, NodeId(3), &in_port(0));
    finish(&mut ir, NodeId(3), out);
    ir
}

#[test]
fn cpu_and_gpu_agree_on_join_range_and_narrowing() {
    on_gpu("cpu_and_gpu_agree_on_join_range_and_narrowing", |gpu| {
        let a: Vec<u8> = [0.25f32, -0.5]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let b: Vec<u8> = [0.125f32, 0.9]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let inputs = BTreeMap::from([
            (
                StableId::from_path("joint0").to_string(),
                TensorRef::new(ElemType::F32, [2], a.as_slice()),
            ),
            (
                StableId::from_path("joint1").to_string(),
                TensorRef::new(ElemType::F32, [2], b.as_slice()),
            ),
        ]);
        for (stack, elem) in [
            (false, ElemType::F32),
            (true, ElemType::F32),
            (false, ElemType::F16),
            (false, ElemType::Bf16),
        ] {
            let ir = join_ir(stack, elem);
            assert!(ir.validate().is_empty(), "{:?}", ir.validate());
            let mut cpu = CpuPlan::compile(&ir, PlanMode::Debug).expect("CPU compiles");
            let want = cpu.run(&inputs).expect("CPU runs")["out"].clone();
            assert_eq!(
                run_gpu(gpu, &ir, &inputs),
                want,
                "stack={stack} elem={elem:?}"
            );
        }
        println!("join/range/narrowing CPU vs GPU: bit-equal (f32, f16, bf16)");
    });
}
