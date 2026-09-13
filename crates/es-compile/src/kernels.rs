//! The CPU reference kernels (spec 11.3). Ground truth, not a fast path.
//!
//! Every function here is pure over slices, single-threaded, and written in a fixed loop and
//! operator order, because the bits it produces are what the Slang lowering (spec 11.4) will
//! be judged against and what `tests/golden/observation/` pins. Rules from spec 3.4 /
//! `DET-010`: no atomics, no `f64` accumulators, no `std` transcendentals — the only one
//! needed is the sRGB 2.4 power, and it goes through `es_math::approx` so Rust and Slang
//! share both coefficients and evaluation order.
//!
//! Layout (design note `docs/design/observation-lowering.md` §2): u8 **HWC** at the sensor
//! boundary, f32 **CHW** everywhere downstream, tightly packed and row-major.

use es_ir::Rect;

/// Identifier of a kernel, in `compiler_hash` order. Append only: inserting one would change
/// the hash of every plan that does not use it (design note §10).
pub const KERNEL_IDS: &[&str] = &[
    "cast_u8_hwc_to_f32_chw.v1",
    "resize_bilinear.v1",
    "resize_nearest.v1",
    "crop.v1",
    "srgb_to_linear.v1",
    "normalize_mean_std.v1",
    "normalize_range.v1",
    "concat.v1",
    "stack.v1",
    "history_push.v1",
    "window_gather.v1",
    "cast_f32_to_f16.v1",
    "cast_f32_to_bf16.v1",
];

// --- casts --------------------------------------------------------------------------------

/// `ToTensor`: HWC u8 → CHW f32 divided by 255. The one layout change in the whole plan.
///
/// `src.len() == h * w * c`, `dst.len() == c * h * w`.
pub fn cast_u8_hwc_to_f32_chw(src: &[u8], h: usize, w: usize, c: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), h * w * c);
    debug_assert_eq!(dst.len(), c * h * w);
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                dst[ch * h * w + y * w + x] = f32::from(src[(y * w + x) * c + ch]) / 255.0;
            }
        }
    }
}

/// f32 → f16, round-to-nearest-even (what `half` and `.half()` both do).
pub fn cast_f32_to_f16(src: &[f32], dst: &mut [u16]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = half::f16::from_f32(*s).to_bits();
    }
}

/// f32 → bf16, round-to-nearest-even.
pub fn cast_f32_to_bf16(src: &[f32], dst: &mut [u16]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = half::bf16::from_f32(*s).to_bits();
    }
}

// --- geometry -----------------------------------------------------------------------------

/// The `align_corners = false` source coordinate for output index `d` (design note §4).
///
/// `PyTorch`'s `area_pixel_compute_source_index`: half-pixel centres, and the *negative* edge is
/// clamped while the positive one is handled by clamping the second tap instead.
#[inline]
fn src_index(scale: f32, d: usize, align_corners_false: bool) -> f32 {
    debug_assert!(align_corners_false);
    let s = scale * (d as f32 + 0.5) - 0.5;
    if s < 0.0 {
        0.0
    } else {
        s
    }
}

/// Bilinear resize of a CHW f32 image, matching
/// `F.interpolate(mode="bilinear", align_corners=False, antialias=False)`.
///
/// Antialiasing is deliberately absent: for a downscale it is a different filter, not a
/// refinement, and which one `LeRobot` uses is `unverified` (design note §12).
pub fn resize_bilinear(
    src: &[f32],
    sw: usize,
    sh: usize,
    c: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), c * sh * sw);
    debug_assert_eq!(dst.len(), c * dh * dw);
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    for ch in 0..c {
        let plane = &src[ch * sh * sw..(ch + 1) * sh * sw];
        for y in 0..dh {
            let sy = src_index(scale_y, y, true);
            let y0 = sy as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let ly1 = sy - y0 as f32;
            let ly0 = 1.0 - ly1;
            for x in 0..dw {
                let sx = src_index(scale_x, x, true);
                let x0 = sx as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let lx1 = sx - x0 as f32;
                let lx0 = 1.0 - lx1;
                // `PyTorch`'s association. Reassociating changes the last bits.
                dst[ch * dh * dw + y * dw + x] = ly0
                    * (lx0 * plane[y0 * sw + x0] + lx1 * plane[y0 * sw + x1])
                    + ly1 * (lx0 * plane[y1 * sw + x0] + lx1 * plane[y1 * sw + x1]);
            }
        }
    }
}

/// Nearest-neighbour resize, `PyTorch`'s `floor(scale * d)` rule (not the half-pixel one:
/// `mode="nearest"` in torch is the legacy, non-centred variant).
pub fn resize_nearest(
    src: &[f32],
    sw: usize,
    sh: usize,
    c: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;
    for ch in 0..c {
        for y in 0..dh {
            let sy = ((scale_y * y as f32) as usize).min(sh - 1);
            for x in 0..dw {
                let sx = ((scale_x * x as f32) as usize).min(sw - 1);
                dst[ch * dh * dw + y * dw + x] = src[ch * sh * sw + sy * sw + sx];
            }
        }
    }
}

/// Crop a CHW f32 image to `rect`. The caller has already checked that `rect` is inside the
/// image (`COMPILE-003` at compile time, never a run-time clamp).
pub fn crop(src: &[f32], sw: usize, sh: usize, c: usize, rect: Rect, dst: &mut [f32]) {
    let (rx, ry) = (rect.x as usize, rect.y as usize);
    let (rw, rh) = (rect.width as usize, rect.height as usize);
    debug_assert!(rx + rw <= sw && ry + rh <= sh);
    debug_assert_eq!(dst.len(), c * rh * rw);
    for ch in 0..c {
        for y in 0..rh {
            let from = ch * sh * sw + (ry + y) * sw + rx;
            let to = ch * rh * rw + y * rw;
            dst[to..to + rw].copy_from_slice(&src[from..from + rw]);
        }
    }
}

// --- colour (spec 3.1) --------------------------------------------------------------------

/// The sRGB EOTF, one channel value in `[0, 1]`.
///
/// `powf` is forbidden in an observation kernel (spec 3.2), so the 2.4 power is
/// `exp(2.4 * ln t)` through `es_math::approx`, whose Slang mirror shares the coefficients.
#[inline]
#[must_use]
pub fn srgb_eotf(x: f32) -> f32 {
    if x <= 0.040_449_936 {
        x / 12.92
    } else {
        let t = (x + 0.055) / 1.055;
        es_math::approx::exp(2.4 * es_math::approx::ln(t))
    }
}

/// `srgb_eotf` sampled at `k / 255`, for a `ColorTransform` that sits before `Dequantize`.
///
/// Built from the same scalar function as [`srgb_to_linear`], so the two paths agree by
/// construction (design note §6). This table is itself a golden.
#[must_use]
pub fn srgb_to_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (k, v) in lut.iter_mut().enumerate() {
        *v = srgb_eotf(k as f32 / 255.0);
    }
    lut
}

/// Element-wise sRGB → linear over f32 in `[0, 1]`.
pub fn srgb_to_linear(src: &[f32], dst: &mut [f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = srgb_eotf(*s);
    }
}

// --- photometric --------------------------------------------------------------------------

/// `(x - mean[c]) / std[c]` over a CHW plane stack. Division, not a reciprocal multiply:
/// `PyTorch` divides, and the two differ in the last bit.
pub fn normalize_mean_std(src: &[f32], plane: usize, mean: &[f32], std: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(mean.len(), std.len());
    for (ch, (m, s)) in mean.iter().zip(std).enumerate() {
        for i in 0..plane {
            dst[ch * plane + i] = (src[ch * plane + i] - m) / s;
        }
    }
}

/// `(x - lo) / (hi - lo)`, the same scalar for every channel.
pub fn normalize_range(src: &[f32], lo: f32, hi: f32, dst: &mut [f32]) {
    let span = hi - lo;
    for (d, s) in dst.iter_mut().zip(src) {
        *d = (*s - lo) / span;
    }
}

// --- structural ---------------------------------------------------------------------------

/// Copy `srcs` back to back into `dst`, `outer` times, taking `chunk[i]` elements of source
/// `i` per repetition. `outer` is the product of the axes before the join axis and `chunk[i]`
/// the product of the axis and everything after it — which makes one function do both
/// `Concat` (differing chunks) and `Stack` (a new axis, equal chunks).
pub fn concat(srcs: &[&[f32]], chunk: &[usize], outer: usize, dst: &mut [f32]) {
    debug_assert_eq!(srcs.len(), chunk.len());
    let mut w = 0;
    for o in 0..outer {
        for (src, n) in srcs.iter().zip(chunk) {
            dst[w..w + n].copy_from_slice(&src[o * n..(o + 1) * n]);
            w += n;
        }
    }
}

/// `Stack`: `concat` with one chunk per input and no leading axes to interleave.
pub fn stack(srcs: &[&[f32]], frame: usize, outer: usize, dst: &mut [f32]) {
    let chunks = vec![frame; srcs.len()];
    concat(srcs, &chunks, outer, dst);
}

// --- temporal (spec 7.5) ------------------------------------------------------------------

/// Write `frame` into ring slot `cursor % depth`. Layer 1 of the spec 7.5 time model; the
/// plan owns one only because the CPU reference has to run standalone.
pub fn history_push(ring: &mut [f32], slot: usize, depth: usize, cursor: usize, frame: &[f32]) {
    debug_assert_eq!(frame.len(), slot);
    debug_assert_eq!(ring.len(), slot * depth);
    let at = (cursor % depth) * slot;
    ring[at..at + slot].copy_from_slice(frame);
}

/// Gather `n` frames oldest→newest ending at `cursor`, `stride` apart, into `dst`.
///
/// The last frame out is always the current one. Frames from before the stream started are
/// the oldest available one repeated — `Align::Hold`, which is what `LeRobot`'s
/// `delta_timestamps` clamping does (`unverified`, design note §12).
// Strides and window parameters are all independent and all explicit; bundling them into a
// struct would only move the same eight values behind a name the GPU mirror does not have.
#[allow(clippy::too_many_arguments)]
pub fn window_gather(
    ring: &[f32],
    slot: usize,
    depth: usize,
    cursor: usize,
    pushed: usize,
    n: usize,
    stride: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), slot * n);
    let oldest = cursor.saturating_sub(pushed.min(depth) - 1);
    for k in 0..n {
        let back = (n - 1 - k) * stride;
        let at = cursor.saturating_sub(back).max(oldest);
        let from = (at % depth) * slot;
        dst[k * slot..(k + 1) * slot].copy_from_slice(&ring[from..from + slot]);
    }
}

#[cfg(test)]
mod tests {
    // Exact comparison is the point: these kernels are a bitwise determinism contract
    // (spec 3.4), so a tolerance here would hide exactly what the tests exist to catch.
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn resize_of_a_constant_is_that_constant() {
        let src = vec![0.25f32; 3 * 6 * 8];
        let mut dst = vec![0.0f32; 3 * 3 * 4];
        resize_bilinear(&src, 8, 6, 3, 4, 3, &mut dst);
        assert!(
            dst.iter().all(|v| (*v - 0.25).abs() < f32::EPSILON),
            "{dst:?}"
        );
    }

    #[test]
    fn resize_to_the_same_size_is_the_identity() {
        let src: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 48];
        resize_bilinear(&src, 8, 6, 1, 8, 6, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn crop_of_crop_is_one_crop() {
        let src: Vec<f32> = (0..2 * 6 * 8).map(|i| i as f32).collect();
        let a = Rect {
            x: 1,
            y: 1,
            width: 6,
            height: 4,
        };
        let b = Rect {
            x: 2,
            y: 1,
            width: 3,
            height: 2,
        };
        let mut mid = vec![0.0f32; 2 * 4 * 6];
        crop(&src, 8, 6, 2, a, &mut mid);
        let mut twice = vec![0.0f32; 2 * 2 * 3];
        crop(&mid, 6, 4, 2, b, &mut twice);

        let combined = Rect {
            x: a.x + b.x,
            y: a.y + b.y,
            width: b.width,
            height: b.height,
        };
        let mut once = vec![0.0f32; 2 * 2 * 3];
        crop(&src, 8, 6, 2, combined, &mut once);
        assert_eq!(twice, once);
    }

    #[test]
    fn the_lut_is_the_scalar_function_sampled() {
        let lut = srgb_to_linear_lut();
        for (k, v) in lut.iter().enumerate() {
            assert_eq!(*v, srgb_eotf(k as f32 / 255.0));
        }
        // Anchors: the linear segment is exact, the top of the range is 1.
        assert_eq!(lut[0], 0.0);
        assert!((lut[255] - 1.0).abs() < 2e-6, "{}", lut[255]);
        assert!((lut[10] - (10.0 / 255.0 / 12.92)).abs() < 1e-9);
        // Monotone: a colour transform that inverts anywhere is a bug you see as banding.
        assert!(lut.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn dequantize_moves_hwc_to_chw() {
        // 2x2 RGB, pixel (y, x) = (10y + x) in every channel offset by the channel.
        let src: Vec<u8> = (0..4)
            .flat_map(|p| [p * 10, p * 10 + 1, p * 10 + 2])
            .collect();
        let mut dst = vec![0.0f32; 12];
        cast_u8_hwc_to_f32_chw(&src, 2, 2, 3, &mut dst);
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[1], 10.0 / 255.0);
        assert_eq!(dst[4], 1.0 / 255.0); // channel 1, pixel 0
        assert_eq!(dst[8], 2.0 / 255.0); // channel 2, pixel 0
    }

    #[test]
    fn window_holds_the_oldest_frame_before_the_ring_fills() {
        let mut ring = vec![0.0f32; 2 * 4];
        history_push(&mut ring, 2, 4, 0, &[1.0, 1.5]);
        let mut dst = vec![0.0f32; 2 * 3];
        window_gather(&ring, 2, 4, 0, 1, 3, 1, &mut dst);
        assert_eq!(dst, vec![1.0, 1.5, 1.0, 1.5, 1.0, 1.5]);

        history_push(&mut ring, 2, 4, 1, &[2.0, 2.5]);
        window_gather(&ring, 2, 4, 1, 2, 3, 1, &mut dst);
        // oldest available repeated, then the two real frames, newest last
        assert_eq!(dst, vec![1.0, 1.5, 1.0, 1.5, 2.0, 2.5]);
    }
}
