//! Materials and textures on both render paths (plan H, packet HT1).
//!
//! A triangle whose `mat` is 0 is drawn exactly as before this module existed: its flat
//! `albedo`, Lambertian, not one extra operation on its path. A triangle with `mat = k + 1`
//! reads material `k` of the scene's [`Materials`]: a base colour factor (the triangle's
//! `albedo`) times a texel, and — when the material is `pbr` — glTF 2.0's metallic-roughness
//! BRDF (Appendix B): GGX `D`, the height-correlated Smith `V`, Schlick `F` with
//! `F0 = mix(0.04, base, metallic)`, and a Lambert lobe `(1 - F) * (1 - metallic) * base / pi`.
//!
//! This file is the reference; `material.slang` mirrors it expression for expression, the
//! way `cpu.rs` and `common.slang` mirror each other. Texels are 8-bit and decoded through a
//! 512-entry table built here with `es_math::approx` (linear, then sRGB), which both sides read
//! from the same buffer — so the decode cannot differ between them. Sampling is bilinear with
//! texel centres at +0.5, `repeat` for a 2D texture and clamp-to-edge inside a cube face, no
//! mipmaps. Barycentrics are recomputed at the hit with `dot` and `cross` written out, never
//! the `OpDot` whose summation order `renderer.md` 9.3 found unpinned.

// Same single-letter names as the ray code of `cpu.rs` (o, d, n, v, l, h, u), for the same
// reason: the Slang mirror uses them, and the two texts have to diff line by line.
// The mirrored Slang has to round as written: `(x + 1) * 0.5` is not `midpoint`, and a texel
// index is an `i32` the way the shader has it.
#![allow(
    clippy::many_single_char_names,
    clippy::manual_midpoint,
    clippy::cast_possible_wrap
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use es_assets::scene::{Material, SceneDesc};
use es_assets::texture::TexKind;
use es_core::StableId;
use es_math::approx;

use crate::error::RenderError;
use crate::scene::Tri;

/// Floats per material record in the upload buffer (mirrors `material.slang`).
pub const MAT_STRIDE: usize = 8;
/// Words per texture header: texel offset, width, face height, flags (bit 0 cube, bit 1 sRGB).
pub const TEX_HEADER: usize = 4;
/// The decode table: 256 linear entries, then 256 sRGB ones.
pub const LUT_LEN: usize = 512;
/// Roughness is clamped here from below: GGX at `alpha -> 0` is a delta the estimator cannot
/// sample (Filament's `MIN_ROUGHNESS` is of the same order).
pub const MIN_ROUGHNESS: f32 = 0.05;
const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
const PI: f32 = std::f32::consts::PI;

/// One material as both kernels read it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuMaterial {
    /// Texture index + 1 of the base colour map, 0 for none.
    pub rgb: u32,
    pub metallic: f32,
    pub roughness: f32,
    /// Texture index + 1 and channel (0 R, 1 G, 2 B) of the metallic map.
    pub metal_tex: u32,
    pub metal_ch: u32,
    pub rough_tex: u32,
    pub rough_ch: u32,
    /// Whether the BRDF above applies. A material that only brings a texture (or `emission`)
    /// is textured and Lambertian: `MuJoCo`'s *default* `specular` / `shininess` do not switch a
    /// geom to PBR, only an explicitly written attribute does.
    pub pbr: bool,
}

/// One texture: packed `r | g << 8 | b << 16` per texel, faces stacked for a cube.
#[derive(Clone, Debug, PartialEq)]
pub struct TexImage {
    pub cube: bool,
    pub srgb: bool,
    pub width: u32,
    pub height: u32,
    pub texels: Vec<u32>,
}

/// The scene's material table, shared by every frame of a [`crate::SceneCache`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Materials {
    pub mats: Vec<GpuMaterial>,
    pub textures: Vec<TexImage>,
}

/// What a geom wears, resolved once per scene: its material slot and how its texture
/// coordinates are generated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Look {
    pub mat: u32,
    pub cube: bool,
    pub texrepeat: [f64; 2],
    pub texuniform: bool,
}

/// The 8-bit decode table both kernels read: `i / 255`, then the exact sRGB EOTF through
/// `approx` (spec 3.2 `DET-010`).
#[must_use]
pub fn lut() -> [f32; LUT_LEN] {
    let mut out = [0.0f32; LUT_LEN];
    for i in 0..256 {
        let c = i as f32 / 255.0;
        out[i] = c;
        out[256 + i] = if c <= 0.040_45 {
            c / 12.92
        } else {
            approx::exp(approx::ln((c + 0.055) / 1.055) * 2.4)
        };
    }
    out
}

/// [`lut`], computed once per process.
fn shared_lut() -> &'static [f32; LUT_LEN] {
    static LUT: std::sync::OnceLock<[f32; LUT_LEN]> = std::sync::OnceLock::new();
    LUT.get_or_init(lut)
}

impl Materials {
    /// The table for `scene`'s drawn materials, and each material asset's slot. A material
    /// whose texture was not decoded is refused by name, as an unloaded mesh is.
    pub(crate) fn build(
        scene: &SceneDesc,
    ) -> Result<(Self, BTreeMap<StableId, Look>), RenderError> {
        let mut out = Self::default();
        let mut looks = BTreeMap::new();
        let mut tex_slot: BTreeMap<StableId, u32> = BTreeMap::new();
        for (id, m) in &scene.materials {
            let mut kinds = Vec::new();
            let mut slot = |tex: Option<StableId>| -> Result<u32, RenderError> {
                let Some(tex) = tex else { return Ok(0) };
                let name = || {
                    scene
                        .assets
                        .iter()
                        .find(|a| a.id == tex)
                        .map_or_else(|| tex.to_string(), |a| a.name.clone())
                };
                let data = scene
                    .textures
                    .get(&tex)
                    .and_then(|t| t.data.as_ref())
                    .ok_or_else(|| {
                        RenderError::Config(format!(
                            "texture `{}` is not loaded (es_assets::mesh::load)",
                            name()
                        ))
                    })?;
                kinds.push(data.kind);
                if let Some(i) = tex_slot.get(&tex) {
                    return Ok(i + 1);
                }
                let texels = data
                    .rgb
                    .chunks_exact(3)
                    .map(|p| u32::from(p[0]) | u32::from(p[1]) << 8 | u32::from(p[2]) << 16)
                    .collect();
                let i = u32::try_from(out.textures.len()).unwrap_or(u32::MAX);
                out.textures.push(TexImage {
                    cube: data.kind == TexKind::Cube,
                    srgb: data.srgb,
                    width: data.width,
                    height: data.height,
                    texels,
                });
                tex_slot.insert(tex, i);
                Ok(i + 1)
            };
            let rgb = slot(m.rgb)?;
            let (metal_tex, metal_ch) = match (m.metallic_map, m.orm) {
                (Some(t), _) => (slot(Some(t))?, 0),
                (None, orm) => (slot(orm)?, 2),
            };
            let (rough_tex, rough_ch) = match (m.roughness_map, m.orm) {
                (Some(t), _) => (slot(Some(t))?, 0),
                (None, orm) => (slot(orm)?, 1),
            };
            if kinds.windows(2).any(|k| k[0] != k[1]) {
                return Err(RenderError::Config(format!(
                    "material {id} mixes 2D and cube textures"
                )));
            }
            let mat = u32::try_from(out.mats.len() + 1).unwrap_or(u32::MAX);
            out.mats.push(GpuMaterial {
                rgb,
                metallic: m.metallic() as f32,
                roughness: m.roughness() as f32,
                metal_tex,
                metal_ch,
                rough_tex,
                rough_ch,
                pbr: is_pbr(m),
            });
            looks.insert(
                *id,
                Look {
                    mat,
                    cube: kinds.first() == Some(&TexKind::Cube),
                    texrepeat: m.texrepeat,
                    texuniform: m.texuniform,
                },
            );
        }
        Ok((out, looks))
    }

    /// `LUT | materials | headers | texels`, the region `Renderer` appends after the scene.
    /// Header offsets are relative to the region's start.
    #[must_use]
    pub fn to_floats(&self) -> Vec<f32> {
        let mut out: Vec<f32> = lut().to_vec();
        for m in &self.mats {
            out.extend_from_slice(&[
                f32::from_bits(m.rgb),
                m.metallic,
                m.roughness,
                f32::from_bits(m.metal_tex),
                f32::from_bits(m.metal_ch),
                f32::from_bits(m.rough_tex),
                f32::from_bits(m.rough_ch),
                f32::from_bits(u32::from(m.pbr)),
            ]);
        }
        let mut offset = out.len() + self.textures.len() * TEX_HEADER;
        for t in &self.textures {
            let flags = u32::from(t.cube) | u32::from(t.srgb) << 1;
            for w in [
                u32::try_from(offset).unwrap_or(u32::MAX),
                t.width,
                t.height,
                flags,
            ] {
                out.push(f32::from_bits(w));
            }
            offset += t.texels.len();
        }
        for t in &self.textures {
            out.extend(t.texels.iter().map(|w| f32::from_bits(*w)));
        }
        out
    }

    fn texel(&self, lut: &[f32; LUT_LEN], t: &TexImage, face: u32, row: u32, col: u32) -> [f32; 3] {
        let _ = self;
        let w = t.texels[((face * t.height + row) * t.width + col) as usize];
        let base = if t.srgb { 256 } else { 0 };
        [0, 8, 16].map(|s| lut[base + ((w >> s) & 0xff) as usize])
    }

    /// Bilinear sample of texture `index` (0-based) at `tc`: `(s, t)` of a 2D texture, the
    /// direction `(s, t, r)` of a cube one.
    #[must_use]
    pub fn sample(&self, index: u32, tc: [f32; 3]) -> [f32; 3] {
        let lut = shared_lut();
        let t = &self.textures[index as usize];
        let (w, h) = (t.width as i32, t.height as i32);
        let (face, s, tt) = if t.cube {
            cube_face(tc)
        } else {
            (0, tc[0], tc[1])
        };
        let x = s * w as f32 - 0.5;
        let y = tt * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (ix, iy) = (x0 as i32, y0 as i32);
        let (cx, cy): ([u32; 2], [u32; 2]) = if t.cube {
            (
                [ix.clamp(0, w - 1) as u32, (ix + 1).clamp(0, w - 1) as u32],
                [iy.clamp(0, h - 1) as u32, (iy + 1).clamp(0, h - 1) as u32],
            )
        } else {
            let a = ((ix % w) + w) % w;
            let b = ((iy % h) + h) % h;
            (
                [a as u32, ((a + 1) % w) as u32],
                [b as u32, ((b + 1) % h) as u32],
            )
        };
        let c00 = self.texel(lut, t, face, cy[0], cx[0]);
        let c10 = self.texel(lut, t, face, cy[0], cx[1]);
        let c01 = self.texel(lut, t, face, cy[1], cx[0]);
        let c11 = self.texel(lut, t, face, cy[1], cx[1]);
        [0, 1, 2].map(|c| {
            let top = c00[c] * (1.0 - fx) + c10[c] * fx;
            let bottom = c01[c] * (1.0 - fx) + c11[c] * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }

    /// The surface a ray `(o, d)` sees on `tri`, which it hits. `mat == 0` is the triangle's
    /// flat albedo and nothing else, which is what keeps every committed frame where it was.
    #[must_use]
    pub fn surface(&self, tri: &Tri, o: [f32; 3], d: [f32; 3]) -> Surface {
        if tri.mat == 0 {
            return Surface::flat(tri.albedo);
        }
        let m = self.mats[(tri.mat - 1) as usize];
        let mut base = tri.albedo;
        let (mut metallic, mut roughness) = (m.metallic, m.roughness);
        if m.rgb != 0 || m.metal_tex != 0 || m.rough_tex != 0 {
            let tc = tex_coord(tri, o, d);
            if m.rgb != 0 {
                let texel = self.sample(m.rgb - 1, tc);
                base = [base[0] * texel[0], base[1] * texel[1], base[2] * texel[2]];
            }
            if m.metal_tex != 0 {
                metallic *= self.sample(m.metal_tex - 1, tc)[m.metal_ch as usize];
            }
            if m.rough_tex != 0 {
                roughness *= self.sample(m.rough_tex - 1, tc)[m.rough_ch as usize];
            }
        }
        Surface {
            base,
            metallic,
            roughness: roughness.clamp(MIN_ROUGHNESS, 1.0),
            pbr: m.pbr,
        }
    }
}

/// Whether a drawn material is PBR: it writes one of the four BRDF attributes or a map.
fn is_pbr(m: &Material) -> bool {
    m.specular.is_some()
        || m.shininess.is_some()
        || m.metallic.is_some()
        || m.roughness.is_some()
        || m.orm.is_some()
        || m.metallic_map.is_some()
        || m.roughness_map.is_some()
}

/// `OpenGL`'s cube-map face selection (the core profile's table 8.19) and the face's `(s, t)`.
fn cube_face(r: [f32; 3]) -> (u32, f32, f32) {
    let (ax, ay, az) = (r[0].abs(), r[1].abs(), r[2].abs());
    let (face, ma, sc, tc) = if ax >= ay && ax >= az {
        if r[0] >= 0.0 {
            (0, ax, -r[2], -r[1])
        } else {
            (1, ax, r[2], -r[1])
        }
    } else if ay >= az {
        if r[1] >= 0.0 {
            (2, ay, r[0], r[2])
        } else {
            (3, ay, r[0], -r[2])
        }
    } else if r[2] >= 0.0 {
        (4, az, r[0], -r[1])
    } else {
        (5, az, -r[0], -r[1])
    };
    if ma <= 0.0 {
        return (0, 0.5, 0.5);
    }
    (face, (sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let n = approx::sqrt(dot(a, a));
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        [0.0; 3]
    }
}

/// The interpolated texture coordinate where `(o, d)` meets `tri`: Moller-Trumbore's
/// barycentrics, recomputed with the written-out `dot` / `cross`.
fn tex_coord(tri: &Tri, o: [f32; 3], d: [f32; 3]) -> [f32; 3] {
    let e1 = sub(tri.v[1], tri.v[0]);
    let e2 = sub(tri.v[2], tri.v[0]);
    let p = cross(d, e2);
    let inv = 1.0 / dot(e1, p);
    let tv = sub(o, tri.v[0]);
    let u = dot(tv, p) * inv;
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    let w = 1.0 - u - v;
    [0, 1, 2].map(|c| tri.tc[0][c] * w + tri.tc[1][c] * u + tri.tc[2][c] * v)
}

/// What a hit looks like to the shading code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub base: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub pbr: bool,
}

impl Surface {
    #[must_use]
    pub fn flat(albedo: [f32; 3]) -> Self {
        Self {
            base: albedo,
            metallic: 0.0,
            roughness: 1.0,
            pbr: false,
        }
    }

    /// The Lambert lobe's colour `base * (1 - metallic)` and `F0 = mix(0.04, base, metallic)`.
    #[must_use]
    pub fn diffuse_and_f0(&self) -> ([f32; 3], [f32; 3]) {
        let (c, f0, _) = self.lobes();
        (c, f0)
    }

    fn lobes(&self) -> ([f32; 3], [f32; 3], f32) {
        let k = 1.0 - self.metallic;
        let c_diff = scale(self.base, k);
        let f0 = [0, 1, 2].map(|c| 0.04 * k + self.base[c] * self.metallic);
        (c_diff, f0, self.roughness * self.roughness)
    }

    /// The glTF BRDF `f(l, v)`, no cosine. `n`, `v`, `l` unit, `n` facing `v`.
    #[must_use]
    pub fn brdf(&self, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> [f32; 3] {
        let nl = dot(n, l);
        let nv = dot(n, v);
        if nl <= 0.0 || nv <= 0.0 {
            return [0.0; 3];
        }
        let (c_diff, f0, a) = self.lobes();
        let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
        let nh = dot(n, h).max(0.0);
        let vh = dot(v, h).max(0.0);
        let a2 = a * a;
        let dd = nh * nh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * dd * dd);
        let gv = nl * approx::sqrt(nv * nv * (1.0 - a2) + a2);
        let gl = nv * approx::sqrt(nl * nl * (1.0 - a2) + a2);
        let vis = 0.5 / (gv + gl);
        let m = 1.0 - vh;
        let m5 = m * m * m * m * m;
        // The Lambert lobe keeps what Fresnel leaves at both ends, `(1 - F(n.l)) (1 - F(n.v))`:
        // reciprocal, and it holds the white furnace at or below one, where glTF's
        // `(1 - F(v.h))` measured 1.025 (renderer.md 15.3).
        let ml = 1.0 - nl;
        let mv = 1.0 - nv;
        let ml5 = ml * ml * ml * ml * ml;
        let mv5 = mv * mv * mv * mv * mv;
        [0, 1, 2].map(|c| {
            let f = f0[c] + (1.0 - f0[c]) * m5;
            let fl = f0[c] + (1.0 - f0[c]) * ml5;
            let fv = f0[c] + (1.0 - f0[c]) * mv5;
            (1.0 - fl) * (1.0 - fv) * c_diff[c] * INV_PI + f * d * vis
        })
    }

    /// Probability of sampling the specular lobe: the lobes' luminance at normal incidence
    /// of the view, `0.5` for a black surface.
    fn p_spec(&self, nv: f32) -> f32 {
        let (c_diff, f0, _) = self.lobes();
        let m = 1.0 - nv.max(0.0);
        let m5 = m * m * m * m * m;
        let f = [0, 1, 2].map(|c| f0[c] + (1.0 - f0[c]) * m5);
        let ws = lum(f);
        let wd = lum(c_diff);
        if ws + wd > 0.0 {
            ws / (ws + wd)
        } else {
            0.5
        }
    }

    /// Solid-angle pdf of [`Self::sample`] producing `l`: the lobe mixture.
    #[must_use]
    pub fn pdf(&self, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> f32 {
        let nl = dot(n, l);
        let nv = dot(n, v);
        if nl <= 0.0 || nv <= 0.0 {
            return 0.0;
        }
        let ps = self.p_spec(nv);
        let a = self.lobes().2;
        let a2 = a * a;
        let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
        let nh = dot(n, h).max(0.0);
        let dd = nh * nh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * dd * dd);
        // Heitz 2018: D_v(h) / (4 v.h) = G1(v) D(h) / (4 n.v).
        let g1 = 2.0 * nv / (nv + approx::sqrt(a2 + (1.0 - a2) * nv * nv));
        let spec = g1 * d / (4.0 * nv);
        ps * spec + (1.0 - ps) * nl * INV_PI
    }

    /// A direction from the lobe mixture: stream-7 index 0 picks the lobe, indices 1 and 2
    /// drive the GGX visible-normal sample (Heitz 2018); the diffuse lobe takes the
    /// cosine-weighted direction `cosine` the caller drew from stream 0.
    #[must_use]
    pub fn sample(
        &self,
        n: [f32; 3],
        v: [f32; 3],
        cosine: [f32; 3],
        u: [f32; 3],
        onb: ([f32; 3], [f32; 3]),
    ) -> [f32; 3] {
        let nv = dot(n, v);
        if u[0] >= self.p_spec(nv) {
            return cosine;
        }
        let (t, b) = onb;
        let a = self.lobes().2;
        let vl = [dot(v, t), dot(v, b), nv];
        let ve = normalize([a * vl[0], a * vl[1], vl[2]]);
        let lensq = ve[0] * ve[0] + ve[1] * ve[1];
        let t1 = if lensq > 0.0 {
            scale([-ve[1], ve[0], 0.0], approx::rsqrt(lensq))
        } else {
            [1.0, 0.0, 0.0]
        };
        let t2 = cross(ve, t1);
        let r = approx::sqrt(u[1]);
        let phi = 2.0 * PI * u[2];
        let p1 = r * approx::cos(phi);
        let s = 0.5 * (1.0 + ve[2]);
        let p2 = (1.0 - s) * approx::sqrt((1.0 - p1 * p1).max(0.0)) + s * (r * approx::sin(phi));
        let pz = approx::sqrt((1.0 - p1 * p1 - p2 * p2).max(0.0));
        let nh = [0, 1, 2].map(|c| p1 * t1[c] + p2 * t2[c] + pz * ve[c]);
        let ne = normalize([a * nh[0], a * nh[1], nh[2].max(0.0)]);
        let h = [0, 1, 2].map(|c| t[c] * ne[0] + b[c] * ne[1] + n[c] * ne[2]);
        let vh = dot(v, h);
        [0, 1, 2].map(|c| 2.0 * vh * h[c] - v[c])
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Shared ownership of a table: every frame of a replay carries the same one.
pub type SharedMaterials = Arc<Materials>;

#[cfg(test)]
mod tests {
    use super::*;

    fn surf(base: f32, metallic: f32, roughness: f32) -> Surface {
        Surface {
            base: [base; 3],
            metallic,
            roughness,
            pbr: true,
        }
    }

    fn dir(theta: f32, phi: f32) -> [f32; 3] {
        let (s, c) = (theta.sin(), theta.cos());
        [s * phi.cos(), s * phi.sin(), c]
    }

    /// Oracle 3: the BRDF is non-negative and reciprocal, `f(l, v) == f(v, l)` — to rounding,
    /// since `h` is normalised from `v + l` in either order.
    #[test]
    fn brdf_is_reciprocal_and_non_negative() {
        let n = [0.0, 0.0, 1.0];
        for &(m, r) in &[(0.0, 0.05), (0.0, 0.5), (1.0, 0.3), (0.5, 1.0)] {
            let s = surf(0.7, m, r);
            for i in 0..24 {
                for j in 0..24 {
                    let v = dir(0.06 * i as f32, 0.7 * i as f32);
                    let l = dir(0.065 * j as f32, 2.1 + 0.3 * j as f32);
                    let a = s.brdf(n, v, l);
                    let b = s.brdf(n, l, v);
                    for c in 0..3 {
                        assert!(a[c] >= 0.0 && a[c].is_finite(), "{a:?}");
                        let tol = 1e-5 * a[c].abs().max(1.0);
                        assert!((a[c] - b[c]).abs() <= tol, "m{m} r{r} {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn cube_faces_follow_opengl() {
        assert_eq!(cube_face([1.0, 0.0, 0.0]), (0, 0.5, 0.5));
        assert_eq!(cube_face([0.0, 0.0, -1.0]).0, 5);
        // +X: s runs along -z, t along -y.
        let (_, s, t) = cube_face([1.0, -0.5, -0.5]);
        assert_eq!((s, t), (0.75, 0.75));
    }

    #[test]
    fn the_decode_table_is_exact_at_the_ends() {
        let t = lut();
        assert_eq!(
            (t[0], t[255], t[256], t[511]),
            (0.0, 1.0, 0.0, 1.0_f32.min(t[511]))
        );
        assert!((t[511] - 1.0).abs() < 1e-5 && (t[256 + 128] - 0.2158).abs() < 1e-3);
    }
}
