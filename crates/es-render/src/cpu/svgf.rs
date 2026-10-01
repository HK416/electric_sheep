//! SVGF's spatial half (Schied et al. 2017): the per-pixel variance of the accumulated estimate
//! and the edge-aware a-trous wavelet filter it guides. `accumulate.slang` and `svgf.slang` are
//! the line-for-line mirrors.

use es_math::approx;

use super::{add, luminance, scale, GBuffer, History};

/// Depth edge-stopping scale of the a-trous filter, m.
const SVGF_SIGMA_Z: f32 = 0.1;
/// 5-tap B-spline wavelet row.
const SVGF_H: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
/// Luminance edge-stopping scale (packet M7/R4; Schied et al. 2017 section 4.3).
const SVGF_SIGMA_L: f32 = 4.0;
/// Keeps the luminance weight's denominator off zero where the variance is zero.
const SVGF_VAR_EPS: f32 = 1e-10;
/// Frames of history below which the moments are too few to trust and the 7x7 spatial
/// estimate stands in (Schied et al. 2017 section 4.2).
const SVGF_MIN_HISTORY: u32 = 4;
/// Radius of that spatial estimate: 7x7.
const SVGF_SPATIAL_RADIUS: i32 = 3;

/// Variance of the accumulated estimate, per pixel (packet M7/R4; Schied et al. 2017 section
/// 4.2). Mirror of `accumulate.slang`.
///
/// With `n >= 4` frames it comes from the moments: `var(l) = E[l^2] - E[l]^2` over the
/// per-frame luminances, divided by `n` once more because what the filter needs is the
/// variance of the *mean* of those `n` frames, which is what the pixel holds. Below 4 the
/// moments are too few to be worth anything and the paper's 7x7 depth/normal-weighted spatial
/// estimate stands in, divided by `n` for the same reason — so the quantity is continuous
/// across the switch and always means "how uncertain is this pixel".
pub(super) fn variance_estimate(
    radiance: &[f32],
    history: &History,
    g: &GBuffer,
    w: u32,
    h: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; (w as usize) * (h as usize)];
    for py in 0..h {
        for px in 0..w {
            let i = (py * w + px) as usize;
            let n = history.n[i];
            if n == 0 {
                continue;
            }
            let inv_n = 1.0 / n as f32;
            out[i] = if n >= SVGF_MIN_HISTORY {
                let m1 = history.moments[i * 2] * inv_n;
                let m2 = history.moments[i * 2 + 1] * inv_n;
                (m2 - m1 * m1).max(0.0) * inv_n
            } else {
                let (mut sw, mut swl, mut swl2) = (0.0f32, 0.0f32, 0.0f32);
                for dy in -SVGF_SPATIAL_RADIUS..=SVGF_SPATIAL_RADIUS {
                    for dx in -SVGF_SPATIAL_RADIUS..=SVGF_SPATIAL_RADIUS {
                        let (nx, ny) = (px as i32 + dx, py as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = (ny as u32 * w + nx as u32) as usize;
                        let (wz, wn) = edge_weights(&g.depth, &g.normal, i, j);
                        let weight = wz * wn;
                        let l =
                            luminance([radiance[j * 3], radiance[j * 3 + 1], radiance[j * 3 + 2]]);
                        sw += weight;
                        swl += weight * l;
                        swl2 += weight * l * l;
                    }
                }
                let inv = if sw > 0.0 { 1.0 / sw } else { 0.0 };
                let mean = swl * inv;
                (swl2 * inv - mean * mean).max(0.0) * inv_n
            };
        }
    }
    out
}

/// The depth and normal edge-stopping weights, returned separately so each caller multiplies
/// them in its own order — the a-trous filter's `wh * wz * wn` is the order its output bits
/// were measured in and must not move.
fn edge_weights(depth: &[f32], normal: &[f32], i: usize, j: usize) -> (f32, f32) {
    let wz = approx::exp(-(depth[i] - depth[j]).abs() / SVGF_SIGMA_Z);
    let mut wn = (normal[i * 3] * normal[j * 3]
        + normal[i * 3 + 1] * normal[j * 3 + 1]
        + normal[i * 3 + 2] * normal[j * 3 + 2])
        .max(0.0);
    // n^32 by five squarings: no transcendental, exact same ops in Slang.
    for _ in 0..5 {
        wn *= wn;
    }
    (wz, wn)
}

// --- SVGF (the a-trous half) -----------------------------------------------------------------

/// Edge-aware a-trous wavelet filter, `iterations` passes at stride `1 << i`, edge-stopping
/// on depth, normal and — when `variance` is `Some` (packet M7/R4) — luminance.
///
/// `variance` is the per-pixel variance of the accumulated estimate ([`History::variance`]).
/// With it the filter is variance-guided the way Schied et al. 2017 section 4.3 is: the
/// luminance weight `exp(-|l_p - l_q| / (sigma_l * sqrt(var_p) + eps))` narrows the kernel
/// wherever the estimate has converged, and the variance is filtered alongside the colour
/// with the squared weights so the next iteration sees the variance of what it is reading.
/// Without it — `temporal: None` — the weight is exactly `1.0` and the output is byte for
/// byte what M4's filter produced.
///
/// Still **not** the whole of SVGF: no history-length-driven kernel widening, and the
/// temporal half is `path_trace_accum`'s, not this function's. See
/// `docs/design/renderer.md` sections 4.3 and 11.
pub fn atrous(
    color: &[f32],
    variance: Option<&[f32]>,
    depth: &[f32],
    normal: &[f32],
    w: u32,
    h: u32,
    iterations: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut src = color.to_vec();
    let mut dst = src.clone();
    let mut vsrc = variance.map(<[f32]>::to_vec).unwrap_or_default();
    let mut vdst = vsrc.clone();
    let guided = !vsrc.is_empty();
    for it in 0..iterations {
        let stride = 1i32 << it;
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                let mut sum = [0.0f32; 3];
                let (mut wsum, mut vsum) = (0.0f32, 0.0f32);
                let l_p = luminance([src[i * 3], src[i * 3 + 1], src[i * 3 + 2]]);
                let sigma_l = if guided {
                    SVGF_SIGMA_L * approx::sqrt(vsrc[i]) + SVGF_VAR_EPS
                } else {
                    0.0
                };
                for ky in 0..5i32 {
                    for kx in 0..5i32 {
                        let nx = px as i32 + (kx - 2) * stride;
                        let ny = py as i32 + (ky - 2) * stride;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let j = (ny as u32 * w + nx as u32) as usize;
                        let wh = SVGF_H[kx as usize] * SVGF_H[ky as usize];
                        let (wz, wn) = edge_weights(depth, normal, i, j);
                        let wl = if guided {
                            let l_q = luminance([src[j * 3], src[j * 3 + 1], src[j * 3 + 2]]);
                            approx::exp(-(l_p - l_q).abs() / sigma_l)
                        } else {
                            1.0
                        };
                        let weight = wh * wz * wn * wl;
                        sum = add(
                            sum,
                            scale([src[j * 3], src[j * 3 + 1], src[j * 3 + 2]], weight),
                        );
                        wsum += weight;
                        if guided {
                            vsum += weight * weight * vsrc[j];
                        }
                    }
                }
                let inv = if wsum > 0.0 { 1.0 / wsum } else { 0.0 };
                dst[i * 3..i * 3 + 3].copy_from_slice(&scale(sum, inv));
                if guided {
                    vdst[i] = vsum * inv * inv;
                }
            }
        }
        std::mem::swap(&mut src, &mut dst);
        if guided {
            std::mem::swap(&mut vsrc, &mut vdst);
        }
    }
    (src, vsrc)
}
