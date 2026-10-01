//! Shading and the display transfer: the `Rs` path's Lambert, [`Shading::Full`]'s shadow ray,
//! hemisphere and highlight (the glTF BRDF on a PBR surface), and linear radiance to `u8`
//! through a tone map and the exact sRGB curve. `common.slang` is the line-for-line mirror.

use es_math::approx;

use super::geometry::any_hit_but_panels;
use super::{add, dot, mul, normalize, scale, sub, RAY_EPS, SHADOW_FAR};
use crate::bvh::Bvh;
use crate::material::Surface;
use crate::scene::Tri;
use crate::view::{RenderConfig, Shading, Tonemap};

// --- shading -------------------------------------------------------------------------------

/// Flat + Lambert with one directional light. No shadow ray: that is a second scan of the
/// whole array per pixel, and the `Pt` path models occlusion properly.
///
/// `ponytail:` no shadows in `Rs`; add a shadow scan when a golden shows the missing contact
/// shadow costs a policy something.
pub(crate) fn shade_lambert(tri: &Tri, n: [f32; 3], cfg: &RenderConfig) -> [f32; 3] {
    shade_lambert_base(tri.albedo, tri.emission, n, cfg)
}

/// [`shade_lambert`] with a base colour that is not the triangle's flat albedo — a texel
/// times the factor, for a textured triangle (plan H, HT1) — and the emission at the hit (an
/// emissive map's texel, HT2). Metallic and roughness do not enter the `Lambert` look.
pub(super) fn shade_lambert_base(
    base: [f32; 3],
    emission: [f32; 3],
    n: [f32; 3],
    cfg: &RenderConfig,
) -> [f32; 3] {
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let ndl = dot(n, light).max(0.0);
    let lambert = cfg.ambient + ndl * (1.0 - cfg.ambient);
    add(scale(base, lambert), emission)
}

/// [`Shading::Full`]: one shadow ray, a hemisphere ambient, a Blinn-Phong highlight (packet
/// M7/R2). `es_shade_full` in `common.slang` is the line-for-line mirror; the order below is
/// the order there.
///
/// `n` is the world-space normal already face-forwarded, `p` the hit point, `d` the (not
/// normalised) view ray. The shadow ray runs on `(0, SHADOW_FAR)` from `p + n * RAY_EPS`
/// towards the light, so the surface cannot shadow itself, and passes through the `_light`
/// panels (packet M17/R6, [`Tri::light`]). `pow` is `exp(ln(x) * k)` guarded at
/// `x <= 0` — no `std`, no `GLSL.std.450` (spec 3.2 `DET-010`).
///
/// A [`Shading::Lambert`] config never reaches here; it is answered as itself so this is a
/// total function on `cfg`.
pub fn shade_full(
    tris: &[Tri],
    bvh: &Bvh,
    tri: &Tri,
    n: [f32; 3],
    p: [f32; 3],
    d: [f32; 3],
    cfg: &RenderConfig,
) -> [f32; 3] {
    shade_full_surface(tris, bvh, tri, &Surface::of(tri), n, p, d, cfg)
}

/// [`shade_full`] on a [`Surface`] (plan H, HT1). A non-PBR surface is the Blinn-Phong look
/// with its base colour in place of the flat albedo; a PBR one replaces the highlight and the
/// diffuse term by the glTF BRDF under the same directional light and hemisphere:
///
/// ```text
/// direct = pi * f(l, v) * max(0, n.l) * vis                 // per channel
/// A      = c_diff * (1 - F0) + F0                           // the hemisphere's albedo
/// rgb    = A * hemi + direct * (1 - hemi) + emission
/// ```
///
/// `pi * f * cos` is what makes a white Lambertian surface at normal incidence return its
/// albedo, the scale `Full`'s own `diffuse * (1 - hemi)` term has.
///
/// A normal-mapped surface (plan H, HT2) lights and tints its hemisphere with the shading
/// normal; the shadow ray still leaves along the geometric `n`. The emission is the surface's.
#[allow(clippy::too_many_arguments)]
pub fn shade_full_surface(
    tris: &[Tri],
    bvh: &Bvh,
    _tri: &Tri,
    surf: &Surface,
    n: [f32; 3],
    p: [f32; 3],
    d: [f32; 3],
    cfg: &RenderConfig,
) -> [f32; 3] {
    let Shading::Full {
        shadows,
        specular,
        shininess,
        sky_rgb,
        ground_rgb,
        ..
    } = cfg.shading
    else {
        return shade_lambert_base(surf.base, surf.emission, surf.normal.unwrap_or(n), cfg);
    };
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let o = add(p, scale(n, RAY_EPS));
    let vis = if shadows && any_hit_but_panels(tris, bvh, o, light, 0.0, SHADOW_FAR, true) {
        0.0
    } else {
        1.0
    };
    let n = surf.normal.unwrap_or(n);
    if surf.pbr {
        let v = normalize(scale(d, -1.0));
        let ndl = dot(n, light).max(0.0);
        let f = surf.brdf(n, v, light);
        let direct = scale(f, std::f32::consts::PI * ndl * vis);
        #[allow(clippy::manual_midpoint)]
        let t = (n[2] + 1.0) * 0.5;
        let hemi = add(ground_rgb, scale(sub(sky_rgb, ground_rgb), t));
        let (c_diff, f0) = surf.diffuse_and_f0();
        let rgb = [0, 1, 2].map(|c| {
            let a = c_diff[c] * (1.0 - f0[c]) + f0[c];
            a * hemi[c] + direct[c] * (1.0 - hemi[c])
        });
        return add(rgb, surf.emission);
    }
    let diffuse = dot(n, light).max(0.0) * vis;
    let h = normalize(sub(light, d));
    let ndh = dot(n, h).max(0.0);
    let spec = if ndh > 0.0 {
        specular * approx::exp(approx::ln(ndh) * shininess) * vis
    } else {
        0.0
    };
    // Written out rather than a `lerp`: HLSL's `lerp` and GLSL's `mix` are not the same
    // expression, and the two texts have to round the same way.
    // Not `f32::midpoint`: `common.slang` computes `(n.z + 1.0) * 0.5` and the two texts have
    // to be the same two operations, not two functions that usually agree.
    #[allow(clippy::manual_midpoint)]
    let t = (n[2] + 1.0) * 0.5;
    let hemi = add(ground_rgb, scale(sub(sky_rgb, ground_rgb), t));
    // Energy-conserving, the same mix `shade_lambert` uses for its constant ambient (amended
    // at review): `hemi + diffuse * (1 - hemi)` per channel, so a fully lit surface returns
    // its albedo and a shadowed one `albedo * hemi`, and nothing clips to white before the
    // sRGB transfer. `spec` stays additive: a highlight is allowed to blow out.
    let lit = [
        hemi[0] + diffuse * (1.0 - hemi[0]),
        hemi[1] + diffuse * (1.0 - hemi[1]),
        hemi[2] + diffuse * (1.0 - hemi[2]),
    ];
    add(add(mul(surf.base, lit), [spec; 3]), surf.emission)
}

/// Exact piecewise sRGB transfer (spec 3.1). `c^(1/2.4)` goes through `es_math::approx`, not
/// `std`, so the CPU and the GPU agree bit for bit (spec 3.2 `DET-010`, spec 28.7 gate 3).
pub fn srgb_encode(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * approx::exp(approx::ln(c) * (1.0 / 2.4)) - 0.055
    }
}

pub fn to_u8(c: f32) -> u8 {
    (255.0 * srgb_encode(c) + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Linear radiance -> display value, one channel (packet M7/R3).
///
/// Additions, multiplies and one division: **no transcendental at all**, which is why the
/// CPU and the GPU are asserted bit-equal here rather than to a ULP budget.
/// `es_tonemap` in `common.slang` is the line-for-line mirror.
pub fn tonemap(c: f32, exposure: f32, map: Tonemap) -> f32 {
    let x = c * exposure;
    match map {
        // Reinhard et al. 2002. Monotone on [0, inf), never reaches 1, so it never clips.
        Tonemap::Reinhard => x / (1.0 + x),
        // Narkowicz 2015's rational ACES fit. Clamped: the fit dips below 0 for x < 0 and
        // creeps past 1 at the top of its range.
        Tonemap::Aces => {
            let num = x * (2.51 * x + 0.03);
            let den = x * (2.43 * x + 0.59) + 0.14;
            (num / den).clamp(0.0, 1.0)
        }
    }
}

/// One linear RGB triple through [`tonemap`] and then the exact sRGB transfer of
/// [`to_u8`] — the `Pt` path's [`Channel::Rgb8`].
pub fn tonemap_to_u8(lin: [f32; 3], exposure: f32, map: Tonemap) -> [u8; 3] {
    [0, 1, 2].map(|c| to_u8(tonemap(lin[c], exposure, map)))
}
