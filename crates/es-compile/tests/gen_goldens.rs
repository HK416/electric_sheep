//! Generator for `tests/golden/observation/**` (spec 1.4).
//!
//! Run **once**:
//!
//! ```text
//! cargo test -p es-compile --test gen_goldens -- --ignored
//! ```
//!
//! After that the files are read-only for ever: `cargo xtask verify-goldens` fails on any
//! modification, and a golden is never edited to make a test pass. Regenerating one is a
//! deliberate act with a spec change and a new kernel id behind it (design note §13).
//!
//! Layout on disk: little-endian, tightly packed, exactly the arena bytes. Each `.bin` has a
//! `.json` sidecar naming shape, dtype, kernel and what the file pins.

use std::fs;
use std::path::PathBuf;

use es_compile::kernels;
use es_ir::image::Rect;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/observation")
}

/// The 8x6 RGB u8 HWC synthetic gradient every image golden starts from. Chosen so every
/// channel varies on a different axis: a channel swap or an HWC/CHW slip is visible.
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

fn write(name: &str, dtype: &str, shape: &[usize], kernel: &str, pins: &str, bytes: &[u8]) {
    let dir = golden_dir();
    fs::create_dir_all(&dir).expect("create tests/golden/observation");
    fs::write(dir.join(format!("{name}.bin")), bytes).expect("write .bin");
    let dims = shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        "{{\n  \"name\": \"{name}\",\n  \"dtype\": \"{dtype}\",\n  \"layout\": \"row-major, little-endian, tightly packed\",\n  \"shape\": [{dims}],\n  \"kernel\": \"{kernel}\",\n  \"pins\": \"{pins}\",\n  \"generator\": \"cargo test -p es-compile --test gen_goldens -- --ignored\",\n  \"spec\": \"docs/design/observation-lowering.md\"\n}}\n"
    );
    fs::write(dir.join(format!("{name}.json")), json).expect("write .json");
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore = "generator; goldens are written once and then read-only (spec 1.4)"]
fn generate() {
    let src = gradient_8x6();

    // 1. ToTensor: HWC u8 -> CHW f32 / 255.
    let mut chw = vec![0.0f32; 3 * 6 * 8];
    kernels::cast_u8_hwc_to_f32_chw(&src, 6, 8, 3, &mut chw);
    write(
        "dequantize_8x6_rgb",
        "f32",
        &[3, 6, 8],
        "cast_u8_hwc_to_f32_chw.v1",
        "the HWC u8 -> CHW f32 /255 boundary conversion (torchvision ToTensor)",
        &f32_bytes(&chw),
    );

    // 2. Bilinear downscale, align_corners=false, antialias=false.
    let mut small = vec![0.0f32; 3 * 3 * 4];
    kernels::resize_bilinear(&chw, 8, 6, 3, 4, 3, &mut small);
    write(
        "resize_bilinear_8x6_to_4x3",
        "f32",
        &[3, 3, 4],
        "resize_bilinear.v1",
        "half-pixel centres, align_corners=false, no antialias, PyTorch's tap association",
        &f32_bytes(&small),
    );

    // 3. Crop.
    let rect = Rect {
        x: 2,
        y: 1,
        width: 4,
        height: 4,
    };
    let mut cropped = vec![0.0f32; 3 * 4 * 4];
    kernels::crop(&chw, 8, 6, 3, rect, &mut cropped);
    write(
        "crop_8x6_at_2_1_4x4",
        "f32",
        &[3, 4, 4],
        "crop.v1",
        "origin top-left (OpenCV, spec 3.1); pairs with ImageSpec::cropped for the intrinsics",
        &f32_bytes(&cropped),
    );

    // 4. The sRGB EOTF at every u8 input.
    let lut = kernels::srgb_to_linear_lut();
    write(
        "srgb_to_linear_lut256",
        "f32",
        &[256],
        "srgb_to_linear.v1",
        "sRGB EOTF at k/255 via es_math::approx exp(2.4 * ln t); no std powf (DET-010)",
        &f32_bytes(&lut),
    );

    // 5. Per-channel normalize with the ImageNet statistics.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut norm = vec![0.0f32; 3 * 3 * 4];
    kernels::normalize_mean_std(&small, 12, &mean, &std, &mut norm);
    write(
        "normalize_imagenet_4x3",
        "f32",
        &[3, 3, 4],
        "normalize_mean_std.v1",
        "(x - mean[c]) / std[c] by division, not by a reciprocal multiply",
        &f32_bytes(&norm),
    );

    // 6. A two-frame history window over a 4-element state, after three pushes.
    let (slot, depth) = (4usize, 4usize);
    let mut ring = vec![0.0f32; slot * depth];
    for i in 0..3usize {
        let f = i as f32;
        kernels::history_push(&mut ring, slot, depth, i, &[f, f + 0.5, f + 1.0, f + 1.5]);
    }
    let mut win = vec![0.0f32; slot * 2];
    kernels::window_gather(&ring, slot, depth, 2, 3, 2, 1, &mut win);
    write(
        "history_window_n2_s1",
        "f32",
        &[2, 4],
        "window_gather.v1",
        "oldest -> newest, current frame last; Align::Hold before the ring fills",
        &f32_bytes(&win),
    );
}
