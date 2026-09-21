//! SSIM between two [`Channel::Rgb8`](es_sensor::Channel) tiles (Wang et al. 2004).
//!
//! Spec 15.3 states the `RS`/`PT` colour contract as a similarity threshold rather than a
//! bit comparison, and until packet M7/R3 nothing in the repository could measure one. This
//! is that measurement and nothing else: pure Rust, `f64`, no GPU, deterministic, and the
//! threshold itself is *not* asserted here — it is set from the numbers in
//! `docs/design/renderer.md` section 10 once they exist.
//!
//! The simple windowed form of the 2004 paper: an 8x8 window slid one pixel at a time over
//! the luma plane, uniform weights (not the 11x11 Gaussian), and the mean of the per-window
//! SSIM. Uniform weights because the accumulation order then *is* the raster order and the
//! result is a function of the two images alone (spec 3.4).
//!
//! `a`, `b`, `w`, `h`, `n`, `mx`, `vx`, `cxy` are the 2004 paper's own names for the two
//! images, the window and its moments; renaming them would make this harder to check against
//! the paper than to read, so the single-character lint is off for this module.
#![allow(clippy::many_single_char_names)]

/// Window side, in pixels. Shrunk for an image smaller than this in either axis, so a tiny
/// tile needs no special case.
const WINDOW: usize = 8;
/// `K1 * L` squared, `K1 = 0.01`, `L = 255` (Wang et al. 2004).
const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
/// `K2 * L` squared, `K2 = 0.03`.
const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

/// Rec.709 luma of one `Rgb8` pixel, the same coefficients `cpu::luminance` uses on linear
/// radiance. On an sRGB-encoded tile this is a perceptual grey, which is what SSIM wants.
fn luma(p: &[u8], i: usize) -> f64 {
    0.2126 * f64::from(p[i * 3])
        + 0.7152 * f64::from(p[i * 3 + 1])
        + 0.0722 * f64::from(p[i * 3 + 2])
}

/// Mean SSIM of two `[h, w, 3]` `Rgb8` tiles, in `[-1, 1]`; `1.0` exactly for identical
/// input.
///
/// # Panics
///
/// If either slice is not `w * h * 3` bytes long.
#[must_use]
pub fn ssim(a: &[u8], b: &[u8], w: u32, h: u32) -> f64 {
    let (w, h) = (w as usize, h as usize);
    assert_eq!(a.len(), w * h * 3, "a is not {w}x{h}x3");
    assert_eq!(b.len(), w * h * 3, "b is not {w}x{h}x3");
    let win = WINDOW.min(w).min(h);
    if win == 0 {
        return 1.0;
    }
    let la: Vec<f64> = (0..w * h).map(|i| luma(a, i)).collect();
    let lb: Vec<f64> = (0..w * h).map(|i| luma(b, i)).collect();
    let n = (win * win) as f64;
    let denom = (n - 1.0).max(1.0);

    let mut sum = 0.0;
    let mut windows = 0u64;
    for y0 in 0..=(h - win) {
        for x0 in 0..=(w - win) {
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for dy in 0..win {
                for dx in 0..win {
                    let i = (y0 + dy) * w + x0 + dx;
                    let (x, y) = (la[i], lb[i]);
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                }
            }
            let (mx, my) = (sx / n, sy / n);
            // Written so that `x == y` gives the numerator and the denominator the *same*
            // bits: `vx`, `vy` and `cxy` are then the same expression over the same sums, and
            // `2 * m * m == m * m + m * m` exactly in binary floating point.
            let vx = (sxx - n * mx * mx) / denom;
            let vy = (syy - n * my * my) / denom;
            let cxy = (sxy - n * mx * my) / denom;
            let num = (2.0 * mx * my + C1) * (2.0 * cxy + C2);
            let den = (mx * mx + my * my + C1) * (vx + vy + C2);
            sum += num / den;
            windows += 1;
        }
    }
    sum / windows as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: u32, h: u32, seed: u32) -> Vec<u8> {
        (0..(w * h * 3))
            .map(|i| crate::rng::mix32(seed ^ i) as u8)
            .collect()
    }

    /// Identity is exact, not "within 1e-12": the ratio is a quotient of identical bits.
    #[test]
    #[allow(clippy::float_cmp)]
    fn identical_images_score_exactly_one() {
        let a = noise(16, 16, 7);
        assert_eq!(ssim(&a, &a, 16, 16), 1.0);
    }

    /// A tile smaller than the window shrinks the window instead of erroring.
    #[test]
    #[allow(clippy::float_cmp)]
    fn a_tile_smaller_than_the_window_still_scores() {
        let a = noise(4, 4, 3);
        assert_eq!(ssim(&a, &a, 4, 4), 1.0);
    }
}
