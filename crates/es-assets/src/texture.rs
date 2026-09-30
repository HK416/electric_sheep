//! Textures: MJCF `<texture>` declarations decoded into texels, and each texture's
//! `AssetRef::hash` moved from its path to its content (plan H, packet HT1; spec 5.3).
//!
//! The parser records a [`TextureSpec`] per `<texture>`; [`load`] — called by
//! [`crate::mesh::load`], the one step that reads the files a scene names — generates the
//! builtins, decodes the PNGs, assembles the cube faces exactly as `MuJoCo` 3.13's
//! `mjCTexture::Compile` does, and writes the content digest into the texture's asset. A
//! skybox is not drawn by this renderer and is left undecoded, with its path digest.
//!
//! Every texture is stored as 8-bit RGB (`MuJoCo`'s default `nchannel = 3`: a PNG's alpha is
//! dropped, 16-bit channels keep their high byte, grey is replicated) plus a resolved colour
//! space. A cube texture is always six square faces in `MuJoCo`'s order — right, left, up,
//! down, front, back, i.e. +X, -X, +Y, -Y, +Z, -Z of the `OpenGL` cube map — whatever form it
//! was declared in.
//!
//! No host `libm` (spec 3.4 `DET-010`): the cube gradient's `asin`/`acos` go through
//! `es_math::approx::acos_f64`, and `random` marks come from a `mt19937_64` written out here
//! (seed 42, as `MuJoCo`'s `randomdot`), so the texels — and therefore the digest — are the
//! same on every host.

// `w`, `h`, `r`, `c`, `s` are width, height, row, column and spec, as in the `MuJoCo` source these
// functions transcribe; renaming them would make the two harder to read side by side.
#![allow(clippy::many_single_char_names)]

use std::path::Path;

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::mesh::MeshError;
use crate::scene::{AssetKind, SceneDesc};

/// Domain separator of a texture's content digest.
const TEXTURE_TAG: &str = "es.texture.v1";

/// `<texture type>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TexKind {
    TwoD,
    Cube,
    /// Declared, parsed and never drawn: the renderer has no environment map.
    Skybox,
}

/// `<texture colorspace>`. `Auto` is the PNG's own `sRGB` chunk, and linear for a builtin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    Auto,
    Srgb,
    Linear,
}

/// `<texture builtin>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Builtin {
    None,
    Gradient,
    Checker,
    Flat,
}

/// `<texture mark>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mark {
    None,
    Edge,
    Cross,
    Random,
}

/// What a `<texture>` element declares. Paths are the scene-relative ones the importer
/// resolved against `<compiler texturedir>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureSpec {
    pub kind: TexKind,
    pub colorspace: ColorSpace,
    pub builtin: Builtin,
    pub rgb1: [f64; 3],
    pub rgb2: [f64; 3],
    pub mark: Mark,
    pub markrgb: [f64; 3],
    pub random: f64,
    pub width: u32,
    pub height: u32,
    pub file: Option<String>,
    /// Rows, columns of the single-file cube grid.
    pub gridsize: [u32; 2],
    pub gridlayout: String,
    /// `fileright fileleft fileup filedown filefront fileback`.
    pub cubefiles: [Option<String>; 6],
}

impl Default for TextureSpec {
    /// `mjs_defaultTexture`.
    fn default() -> Self {
        Self {
            kind: TexKind::Cube,
            colorspace: ColorSpace::Auto,
            builtin: Builtin::None,
            rgb1: [0.8; 3],
            rgb2: [0.5; 3],
            mark: Mark::None,
            markrgb: [0.0; 3],
            random: 0.01,
            width: 0,
            height: 0,
            file: None,
            gridsize: [1, 1],
            gridlayout: "............".to_owned(),
            cubefiles: Default::default(),
        }
    }
}

/// How a 2D texture's coordinate outside `[0, 1)` is folded back (glTF's sampler `wrapS` /
/// `wrapT`; plan H, HT2). An MJCF texture always repeats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

/// Decoded texels: 8-bit RGB, row-major, row 0 the image's top row (`MuJoCo`'s and `OpenGL`'s
/// `t = 0`). A cube is six `width x width` faces stacked in face order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureData {
    pub kind: TexKind,
    pub width: u32,
    /// Face height: `height` of a 2D texture, `width` for a cube.
    pub height: u32,
    /// The resolved colour space: `true` decodes each texel from sRGB before filtering.
    pub srgb: bool,
    pub rgb: Vec<u8>,
    /// `(s, t)` wrap modes; `Repeat` for every MJCF texture.
    #[serde(default)]
    pub wrap: [Wrap; 2],
}

impl TextureData {
    /// The texel at `(face, row, col)` as three bytes.
    #[must_use]
    pub fn texel(&self, face: u32, row: u32, col: u32) -> [u8; 3] {
        let i = (((face * self.height + row) * self.width + col) * 3) as usize;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    /// blake3 over the kind, colour space, shape and texels — never the path (spec 5.3).
    #[must_use]
    pub fn content_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(TEXTURE_TAG.as_bytes());
        h.update(&[match self.kind {
            TexKind::TwoD => 0,
            TexKind::Cube => 1,
            TexKind::Skybox => 2,
        }]);
        h.update(&[u8::from(self.srgb)]);
        h.update(&self.width.to_le_bytes());
        h.update(&self.height.to_le_bytes());
        h.update(&self.rgb);
        // Plan H, HT2: a sampler other than repeat is part of what the texture looks like;
        // appended only then, so every HT1 digest stands.
        if self.wrap != [Wrap::Repeat; 2] {
            h.update(b"wrap");
            h.update(&self.wrap.map(|w| w as u8));
        }
        *h.finalize().as_bytes()
    }
}

/// A texture of the scene: what the file declares and, once [`load`] ran, its texels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub spec: TextureSpec,
    pub data: Option<TextureData>,
}

/// Decodes every texture `scene` declares and has not decoded yet, relative to `base_dir`,
/// and moves its asset's hash onto the texels. Idempotent, like the mesh loader; a scene with
/// no texture is untouched.
pub fn load(scene: &mut SceneDesc, base_dir: &Path) -> Result<(), MeshError> {
    let SceneDesc {
        assets, textures, ..
    } = scene;
    for asset in assets.iter_mut() {
        if asset.kind != AssetKind::Texture {
            continue;
        }
        let Some(tex) = textures.get_mut(&asset.id) else {
            continue;
        };
        if tex.data.is_some() || tex.spec.kind == TexKind::Skybox {
            continue;
        }
        let fail = |reason: String| MeshError::Texture {
            name: asset.name.clone(),
            path: asset.path.clone(),
            reason,
        };
        let data = decode(&tex.spec, base_dir).map_err(fail)?;
        asset.hash = data.content_hash();
        tex.data = Some(data);
    }
    Ok(())
}

/// The texels a spec describes.
fn decode(spec: &TextureSpec, base: &Path) -> Result<TextureData, String> {
    let cube = spec.kind != TexKind::TwoD;
    if spec.builtin != Builtin::None {
        let (w, h) = (spec.width, spec.height);
        if w < 1 || (!cube && h < 1) {
            return Err("a builtin texture needs width (and, 2D, height) >= 1".to_owned());
        }
        let srgb = spec.colorspace == ColorSpace::Srgb;
        return Ok(if cube {
            TextureData {
                kind: TexKind::Cube,
                width: w,
                height: w,
                srgb,
                rgb: builtin_cube(spec, w),
                wrap: [Wrap::Repeat; 2],
            }
        } else {
            TextureData {
                kind: TexKind::TwoD,
                width: w,
                height: h,
                srgb,
                rgb: builtin_2d(spec, w, h),
                wrap: [Wrap::Repeat; 2],
            }
        });
    }
    let resolve = |srgb_chunk: bool| match spec.colorspace {
        ColorSpace::Srgb => true,
        ColorSpace::Linear => false,
        ColorSpace::Auto => srgb_chunk,
    };
    if let Some(file) = &spec.file {
        let img = read_png(&base.join(file))?;
        let srgb = resolve(img.srgb);
        if !cube {
            return Ok(TextureData {
                kind: TexKind::TwoD,
                width: img.w,
                height: img.h,
                srgb,
                rgb: img.rgb,
                wrap: [Wrap::Repeat; 2],
            });
        }
        return cube_single(spec, &img, srgb);
    }
    if cube {
        return cube_separate(spec, base, resolve);
    }
    Err("a 2D texture needs `file` or `builtin`".to_owned())
}

/// A decoded PNG, reduced to 8-bit RGB.
pub(crate) struct Png {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) srgb: bool,
    pub(crate) rgb: Vec<u8>,
}

fn read_png(path: &Path) -> Result<Png, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_png(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// `lodepng` with `LCT_RGB`, as `MuJoCo` asks for it: palette and low bit depths expanded,
/// 16-bit channels to their high byte, grey replicated, alpha dropped. Also the glTF
/// importer's image decoder (plan H, HT2).
pub(crate) fn decode_png(bytes: Vec<u8>) -> Result<Png, String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let srgb = reader.info().srgb.is_some();
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "image too large".to_owned())?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let stride = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("palette not expanded".to_owned()),
    };
    let (w, h) = (info.width, info.height);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for row in 0..h as usize {
        let line = &buf[row * info.line_size..row * info.line_size + w as usize * stride];
        for px in line.chunks_exact(stride) {
            if stride <= 2 {
                rgb.extend_from_slice(&[px[0]; 3]);
            } else {
                rgb.extend_from_slice(&px[..3]);
            }
        }
    }
    Ok(Png { w, h, srgb, rgb })
}

/// `(std::byte)(255 * c)`: truncation toward zero, as the C++ cast does for `c` in `[0, 1]`.
fn byte(c: f64) -> u8 {
    (255.0 * c) as u8
}

fn bytes3(c: [f64; 3]) -> [u8; 3] {
    c.map(byte)
}

/// `MuJoCo`'s `interp`: a sigmoid between two colours on `pos` in `(-1, 1)`.
fn interp(rgb1: [f64; 3], rgb2: [f64; 3], pos: f64) -> [u8; 3] {
    let correction = 1.0 / 2f64.sqrt();
    // MuJoCo's expression as written, not `f64::midpoint`: the bytes are MuJoCo's.
    #[allow(clippy::manual_midpoint)]
    let alpha = (0.5 * (1.0 + pos / (1.0 + pos * pos).sqrt() / correction)).clamp(0.0, 1.0);
    [0, 1, 2].map(|j| byte(alpha * rgb1[j] + (1.0 - alpha) * rgb2[j]))
}

/// `MuJoCo`'s `checker` for one `w x h` side.
fn checker(out: &mut [u8], a: [u8; 3], b: [u8; 3], w: u32, h: u32) {
    for r in 0..h {
        for c in 0..w {
            let first = (r < h / 2) == (c < w / 2);
            let i = ((r * w + c) * 3) as usize;
            out[i..i + 3].copy_from_slice(if first { &a } else { &b });
        }
    }
}

fn put(out: &mut [u8], w: u32, r: u32, c: u32, v: [u8; 3]) {
    let i = ((r * w + c) * 3) as usize;
    out[i..i + 3].copy_from_slice(&v);
}

/// `mjCTexture::Builtin2D`.
fn builtin_2d(s: &TextureSpec, w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 3) as usize];
    match s.builtin {
        Builtin::Gradient => {
            for r in 0..h {
                for c in 0..w {
                    let x = 2.0 * f64::from(c) / f64::from(w.max(2) - 1) - 1.0;
                    let y = 1.0 - 2.0 * f64::from(r) / f64::from(h.max(2) - 1);
                    let pos = 2.0 * (x * x + y * y).sqrt() - 1.0;
                    put(&mut out, w, r, c, interp(s.rgb2, s.rgb1, pos));
                }
            }
        }
        Builtin::Checker => checker(&mut out, bytes3(s.rgb1), bytes3(s.rgb2), w, h),
        Builtin::Flat | Builtin::None => {
            for px in out.chunks_exact_mut(3) {
                px.copy_from_slice(&bytes3(s.rgb1));
            }
        }
    }
    marks(s, &mut out, w, h, 1);
    out
}

/// `mjCTexture::BuiltinCube`: six `w x w` faces.
fn builtin_cube(s: &TextureSpec, w: u32) -> Vec<u8> {
    let face = (w * w * 3) as usize;
    let mut out = vec![0u8; face * 6];
    match s.builtin {
        Builtin::Gradient => {
            let half_pi = std::f64::consts::FRAC_PI_2;
            for r in 0..w {
                for c in 0..w {
                    let x = 2.0 * f64::from(c) / f64::from(w.max(2) - 1) - 1.0;
                    let y = 1.0 - 2.0 * f64::from(r) / f64::from(w.max(2) - 1);
                    let len = (1.0 + x * x + y * y).sqrt();
                    // asin(a) = pi/2 - acos(a), through `approx` rather than the host libm.
                    let elside = (half_pi - es_math::approx::acos_f64(y / len)) / half_pi;
                    let elup = 1.0 - es_math::approx::acos_f64(1.0 / len) / half_pi;
                    let side = interp(s.rgb1, s.rgb2, elside);
                    for f in [0, 1, 4, 5] {
                        put(&mut out[f * face..], w, r, c, side);
                    }
                    put(&mut out[2 * face..], w, r, c, interp(s.rgb1, s.rgb2, elup));
                    put(&mut out[3 * face..], w, r, c, interp(s.rgb1, s.rgb2, -elup));
                }
            }
        }
        Builtin::Checker => {
            let (a, b) = (bytes3(s.rgb1), bytes3(s.rgb2));
            for f in 0..6 {
                let (x, y) = if f < 4 { (a, b) } else { (b, a) };
                checker(&mut out[f * face..(f + 1) * face], x, y, w, w);
            }
        }
        Builtin::Flat | Builtin::None => {
            for f in 0..6 {
                let v = bytes3(if f == 3 { s.rgb2 } else { s.rgb1 });
                for px in out[f * face..(f + 1) * face].chunks_exact_mut(3) {
                    px.copy_from_slice(&v);
                }
            }
        }
    }
    marks(s, &mut out, w, w, 6);
    out
}

/// Edge, cross or random marks over `faces` faces of `w x h`.
fn marks(s: &TextureSpec, out: &mut [u8], w: u32, h: u32, faces: usize) {
    let m = bytes3(s.markrgb);
    let face = (w * h * 3) as usize;
    match s.mark {
        Mark::Edge => {
            for f in 0..faces {
                let o = &mut out[f * face..(f + 1) * face];
                for r in 0..h {
                    put(o, w, r, 0, m);
                    put(o, w, r, w - 1, m);
                }
                for c in 0..w {
                    put(o, w, 0, c, m);
                    put(o, w, h - 1, c, m);
                }
            }
        }
        Mark::Cross => {
            for f in 0..faces {
                let o = &mut out[f * face..(f + 1) * face];
                for r in 0..h {
                    put(o, w, r, w / 2, m);
                }
                for c in 0..w {
                    put(o, w, h / 2, c, m);
                }
            }
        }
        // `randomdot` over every row of every face (`height` is `6 w` for a cube).
        Mark::Random if s.random > 0.0 => {
            let mut rng = Mt64::new(42);
            for px in out.chunks_exact_mut(3) {
                if rng.canonical() < s.random {
                    px.copy_from_slice(&m);
                }
            }
        }
        Mark::Random | Mark::None => {}
    }
}

/// `mjCTexture::LoadCubeSingle`: a `1 x 1` grid repeats one square image on every face; a
/// larger grid cuts `gridlayout`'s faces out of it, and an undeclared face is `rgb1`.
fn cube_single(spec: &TextureSpec, img: &Png, srgb: bool) -> Result<TextureData, String> {
    let [rows, cols] = spec.gridsize;
    if rows < 1 || cols < 1 || rows * cols > 12 {
        return Err("gridsize must be non-zero and no more than 12 squares".to_owned());
    }
    if img.w / cols != img.h / rows || img.w % cols != 0 || img.h % rows != 0 {
        return Err(format!(
            "PNG size {}x{} is not an integer multiple of gridsize {rows} {cols}",
            img.w, img.h
        ));
    }
    let w = img.w / cols;
    let face = (w * w * 3) as usize;
    let mut out = vec![0u8; face * 6];
    let mut loaded = [false; 6];
    if rows == 1 && cols == 1 {
        for f in 0..6 {
            out[f * face..(f + 1) * face].copy_from_slice(&img.rgb[..face]);
        }
        loaded = [true; 6];
    } else {
        let layout: Vec<char> = spec.gridlayout.chars().collect();
        for k in 0..(rows * cols) as usize {
            let f = match layout.get(k) {
                Some('R') => 0,
                Some('L') => 1,
                Some('U') => 2,
                Some('D') => 3,
                Some('F') => 4,
                Some('B') => 5,
                Some('.') | None => continue,
                Some(other) => return Err(format!("gridlayout symbol `{other}` is not .RLUDFB")),
            };
            let (r0, c0) = (w * (k as u32 / cols), w * (k as u32 % cols));
            for j in 0..w {
                let src = (((j + r0) * img.w + c0) * 3) as usize;
                let dst = f * face + (j * w * 3) as usize;
                out[dst..dst + (w * 3) as usize]
                    .copy_from_slice(&img.rgb[src..src + (w * 3) as usize]);
            }
            loaded[f] = true;
        }
    }
    fill_missing(&mut out, loaded, face, bytes3(spec.rgb1));
    Ok(TextureData {
        kind: TexKind::Cube,
        width: w,
        height: w,
        srgb,
        rgb: out,
        wrap: [Wrap::Repeat; 2],
    })
}

/// `mjCTexture::LoadCubeSeparate`: one square PNG per face, the first one's size for all.
fn cube_separate(
    spec: &TextureSpec,
    base: &Path,
    resolve: impl Fn(bool) -> bool,
) -> Result<TextureData, String> {
    let mut faces: Vec<Option<Png>> = Vec::new();
    for file in &spec.cubefiles {
        faces.push(match file {
            Some(f) => Some(read_png(&base.join(f))?),
            None => None,
        });
    }
    let Some(first) = faces.iter().flatten().next() else {
        return Err("a cube texture needs `file`, a cube file or `builtin`".to_owned());
    };
    let (w, srgb) = (first.w, resolve(first.srgb));
    let face = (w * w * 3) as usize;
    let mut out = vec![0u8; face * 6];
    let mut loaded = [false; 6];
    for (f, img) in faces.iter().enumerate() {
        let Some(img) = img else { continue };
        if img.w != img.h || img.w != w {
            return Err(format!("cube face {f} is {}x{}, not {w}x{w}", img.w, img.h));
        }
        out[f * face..(f + 1) * face].copy_from_slice(&img.rgb);
        loaded[f] = true;
    }
    fill_missing(&mut out, loaded, face, bytes3(spec.rgb1));
    Ok(TextureData {
        kind: TexKind::Cube,
        width: w,
        height: w,
        srgb,
        rgb: out,
        wrap: [Wrap::Repeat; 2],
    })
}

fn fill_missing(out: &mut [u8], loaded: [bool; 6], face: usize, rgb1: [u8; 3]) {
    for (f, done) in loaded.iter().enumerate() {
        if !done {
            for px in out[f * face..(f + 1) * face].chunks_exact_mut(3) {
                px.copy_from_slice(&rgb1);
            }
        }
    }
}

/// `std::mt19937_64`, and `uniform_real_distribution<double>(0, 1)` as libstdc++'s
/// `generate_canonical` computes it from one 64-bit draw: `double(x) / 2^64`, kept below 1.
struct Mt64 {
    mt: [u64; 312],
    i: usize,
}

impl Mt64 {
    fn new(seed: u64) -> Self {
        let mut mt = [0u64; 312];
        mt[0] = seed;
        for i in 1..312 {
            mt[i] = 6_364_136_223_846_793_005u64
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, i: 312 }
    }

    fn next(&mut self) -> u64 {
        const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
        const LOWER: u64 = 0x7FFF_FFFF;
        const A: u64 = 0xB502_6F5A_A966_19E9;
        if self.i >= 312 {
            for k in 0..312 {
                let x = (self.mt[k] & UPPER) | (self.mt[(k + 1) % 312] & LOWER);
                let mag = if x & 1 == 0 { 0 } else { A };
                self.mt[k] = self.mt[(k + 156) % 312] ^ (x >> 1) ^ mag;
            }
            self.i = 0;
        }
        let mut x = self.mt[self.i];
        self.i += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^ (x >> 43)
    }

    fn canonical(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let r = self.next() as f64 / 18_446_744_073_709_551_616.0;
        if r >= 1.0 {
            1.0 - f64::EPSILON / 2.0
        } else {
            r
        }
    }
}

/// The id of a `<texture>` named `name`.
#[must_use]
pub fn texture_id(name: &str) -> StableId {
    crate::scene::scene_id("asset", &format!("texture/{name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference sequence of `std::mt19937_64` default-seeded (5489): the 10000th output
    /// is 9981545732273789042 (C++11 [rand.predef]).
    #[test]
    fn mt19937_64_matches_the_standard() {
        let mut rng = Mt64::new(5489);
        let mut x = 0;
        for _ in 0..10_000 {
            x = rng.next();
        }
        assert_eq!(x, 9_981_545_732_273_789_042);
    }

    #[test]
    fn builtins_follow_mujoco() {
        let spec = TextureSpec {
            kind: TexKind::TwoD,
            builtin: Builtin::Checker,
            rgb1: [0.2, 0.3, 0.4],
            rgb2: [0.1, 0.15, 0.2],
            width: 4,
            height: 4,
            ..TextureSpec::default()
        };
        let d = decode(&spec, Path::new(".")).unwrap();
        // 0.3 * 255 = 76.5 truncates to 76.
        assert_eq!(d.texel(0, 0, 0), [51, 76, 102]);
        assert_eq!(d.texel(0, 0, 3), [25, 38, 51]);
        assert_eq!(d.texel(0, 3, 3), [51, 76, 102]);
        assert!(!d.srgb, "a builtin under `auto` is linear");

        let cube = TextureSpec {
            builtin: Builtin::Flat,
            mark: Mark::Cross,
            rgb1: [0.3, 0.6, 0.5],
            rgb2: [0.3, 0.6, 0.5],
            width: 5,
            ..TextureSpec::default()
        };
        let d = decode(&cube, Path::new(".")).unwrap();
        assert_eq!((d.width, d.height, d.rgb.len()), (5, 5, 5 * 5 * 3 * 6));
        assert_eq!(
            d.texel(3, 2, 0),
            [0, 0, 0],
            "the cross runs through row w/2"
        );
        assert_eq!(d.texel(3, 0, 0), [76, 153, 127]);
        let random = TextureSpec {
            mark: Mark::Random,
            random: 0.5,
            ..cube
        };
        let a = decode(&random, Path::new(".")).unwrap();
        assert_eq!(a, decode(&random, Path::new(".")).unwrap());
        assert!(a.rgb.chunks(3).any(|p| p == [0, 0, 0]));
    }
}
