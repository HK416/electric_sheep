//! [`SceneDesc`] to MJCF, full fidelity (plan G, packet G2).
//!
//! The export a person opens in `MuJoCo`'s viewer: everything the scene carries — bodies,
//! joints, geoms, sites, cameras, `_light` geoms, materials and textures, meshes, tendons,
//! actuators, sensors, contact pairs and excludes, `gravcomp`, `<option>` — written so that
//! [`super::parse_str`] of the text, then [`crate::mesh::load`] of `out_dir`, gives the scene
//! back field for field, with the same `scene_hash` and asset hashes. Unlike
//! `es-physics-backend`'s `mjcf_out`, which drops what carries no dynamics, nothing is
//! dropped: what MJCF cannot say is a [`WriteError::Inexpressible`] naming it.
//!
//! How the round trip is kept exact:
//!
//! * Angles are radians (`<compiler angle="radian">`); `fovy`, always degrees in MJCF, is the
//!   degree value whose product with `DEG_TO_RAD` is the stored radian value.
//! * The reader normalises every `quat` and joint `axis` (a pose's quaternion twice), and
//!   normalising is not bitwise idempotent (`es-math`), so what is written is a pre-image.
//! * Names the parser generated for unnamed elements (`geom3`) are left out again; every
//!   other name is written.
//! * Asset files are re-encoded from the decoded data (STL, OBJ, PNG) at the asset's own
//!   scene-relative path — the path is hash input (spec 5.3) — or, for an absolute or `..`
//!   path, under `<kind>/<name>.<ext>`, which moves that asset's path digest. A file already
//!   in `out_dir` with other bytes is never overwritten.
//! * Every number is Rust's shortest round-trip decimal; a value equal to the reader's default
//!   (`MuJoCo`'s) is left out.

// `w`, `h`, `x`, `y`, `z`: the texture and quaternion components, as everywhere in this crate.
#![allow(clippy::many_single_char_names)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Component, Path};

use es_core::StableId;
use es_math::units::DEG_TO_RAD;
use es_math::{Pose, Quat, Vec3};
use thiserror::Error;

use crate::scene::{
    ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Body, Geom, JointKind, SceneDesc,
    SensorKind, SensorTarget, Shape, TendonKind,
};
use crate::texture::{Builtin, ColorSpace, TexKind, TextureData};

/// Why a scene was not written.
#[derive(Debug, Error)]
pub enum WriteError {
    /// Something the scene carries that MJCF (or this crate's MJCF reader) cannot say.
    #[error("{0} cannot be written as MJCF")]
    Inexpressible(String),
    #[error("{path}: {reason}")]
    Io { path: String, reason: String },
}

macro_rules! no {
    ($($t:tt)*) => { WriteError::Inexpressible(format!($($t)*)) };
}

/// The enums' MJCF spellings, indexed by declaration order.
const INTEGRATORS: [&str; 4] = ["Euler", "RK4", "implicit", "implicitfast"];
const CONES: [&str; 2] = ["pyramidal", "elliptic"];
const JACOBIANS: [&str; 3] = ["dense", "sparse", "auto"];
const SOLVERS: [&str; 3] = ["PGS", "CG", "Newton"];
const TEX_KINDS: [&str; 3] = ["2d", "cube", "skybox"];
const COLORSPACES: [&str; 3] = ["auto", "sRGB", "linear"];
const BUILTINS: [&str; 4] = ["none", "gradient", "checker", "flat"];
const MARKS: [&str; 4] = ["none", "edge", "cross", "random"];
const SENSORS: [&str; 12] = [
    "jointpos",
    "jointvel",
    "actuatorfrc",
    "framepos",
    "framequat",
    "accelerometer",
    "gyro",
    "force",
    "torque",
    "touch",
    "rangefinder",
    "camprojection",
];
/// `<texture>`'s six single-face file attributes, in face order.
const SIDES: [&str; 6] = [
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
];

/// Writes `scene` as MJCF text and its mesh and texture files under `out_dir`; the caller
/// writes the returned text to a file in `out_dir`.
pub fn write_mjcf(scene: &SceneDesc, out_dir: &Path) -> Result<String, WriteError> {
    let mut w = Writer::new(scene);
    w.document()?;
    for (rel, bytes) in &w.files {
        let path = out_dir.join(rel);
        let fail = |reason: String| WriteError::Io {
            path: path.display().to_string(),
            reason,
        };
        match std::fs::read(&path) {
            Ok(old) if &old == bytes => continue,
            Ok(_) => return Err(fail("exists with other bytes; not overwritten".to_owned())),
            Err(_) => {}
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| fail(e.to_string()))?;
        }
        std::fs::write(&path, bytes).map_err(|e| fail(e.to_string()))?;
    }
    Ok(w.out)
}

/// Shortest round-trip decimal.
fn num(v: f64) -> String {
    format!("{v:?}")
}

fn nums(vs: &[f64]) -> String {
    vs.iter().map(|v| num(*v)).collect::<Vec<_>>().join(" ")
}

fn same(a: &[f64], b: &[f64]) -> bool {
    a.iter()
        .map(|v| v.to_bits())
        .eq(b.iter().map(|v| v.to_bits()))
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `v` moved by `k` in {0, +1, -1, +2, -2} units in the last place (by bit pattern).
fn nudge(v: f64, k: u32) -> f64 {
    let step = u64::from(k.div_ceil(2));
    f64::from_bits(if k % 2 == 1 {
        v.to_bits().wrapping_add(step)
    } else {
        v.to_bits().wrapping_sub(step)
    })
}

/// A value `normalize` maps onto `target` bit for bit, else `target`: `target` scaled by
/// `1 ± k * 1e-9`, then moved by up to two ULPs per component (a pose's quaternion is
/// normalised twice, so its pre-image can sit two ULPs off). Measured: every one of 2,000
/// random unit quaternions has a single-normalisation pre-image by `k = 42`, and of 3,000
/// random unit axes by `k = 69`; the search stops at the first.
fn preimage<const N: usize>(
    target: [f64; N],
    normalize: impl Fn([f64; N]) -> [f64; N],
) -> [f64; N] {
    for k in 0..=1000u32 {
        let scale = 1.0 + f64::from(k) * if k % 2 == 0 { 1e-9 } else { -1e-9 };
        for d in 0..5u32.pow(N as u32) {
            let mut c = target;
            for (i, v) in c.iter_mut().enumerate() {
                *v = nudge(*v * scale, d / 5u32.pow(i as u32) % 5);
            }
            if same(&normalize(c), &target) {
                return c;
            }
        }
    }
    target
}

/// One element's opening tag, built attribute by attribute.
struct El(String);

impl El {
    fn new(tag: &str) -> Self {
        Self(format!("<{tag}"))
    }

    fn s(mut self, k: &str, v: &str) -> Self {
        let _ = write!(self.0, " {k}=\"{}\"", esc(v));
        self
    }

    /// Written unless it is the reader's default, bit for bit.
    fn f(self, k: &str, v: &[f64], default: &[f64]) -> Self {
        if same(v, default) {
            self
        } else {
            self.s(k, &nums(v))
        }
    }

    /// An enum or integer attribute, written unless it is the default.
    #[allow(clippy::needless_pass_by_value)]
    fn e(self, k: &str, v: impl ToString, default: &str) -> Self {
        let v = v.to_string();
        if v == default {
            self
        } else {
            self.s(k, &v)
        }
    }

    fn opt(self, k: &str, v: Option<f64>) -> Self {
        match v {
            Some(v) => self.s(k, &num(v)),
            None => self,
        }
    }

    fn range(self, k: &str, v: Option<(f64, f64)>) -> Self {
        self.opt_vec(k, v.map(|(lo, hi)| vec![lo, hi]))
    }

    fn opt_vec(self, k: &str, v: Option<Vec<f64>>) -> Self {
        match v {
            Some(v) => self.s(k, &nums(&v)),
            None => self,
        }
    }

    fn pos(self, p: Vec3) -> Self {
        self.f("pos", &[p.x, p.y, p.z], &[0.0; 3])
    }

    /// `pos` and `quat` of a [`Pose`], whose constructor normalises the quaternion the
    /// orientation reader already normalised: the pre-image is of both.
    fn pose(self, p: Pose) -> Self {
        self.pos(p.position).quat(p.orientation, 2)
    }

    /// MJCF `quat` is wxyz; the identity is left out. `times` is how often the reader
    /// normalises it.
    fn quat(self, q: Quat, times: usize) -> Self {
        if same(&[q.x, q.y, q.z, q.w], &[0.0, 0.0, 0.0, 1.0]) {
            return self;
        }
        let [x, y, z, w] = preimage([q.x, q.y, q.z, q.w], |[x, y, z, w]| {
            let n = (0..times).fold(Quat::from_xyzw(x, y, z, w), |q, _| q.normalize());
            [n.x, n.y, n.z, n.w]
        });
        self.s("quat", &nums(&[w, x, y, z]))
    }
}

/// MJCF name namespaces: bodies, joints, geoms, sites, cameras, tendons, actuators, and one
/// per asset kind.
type Ns = u8;

struct Writer<'a> {
    scene: &'a SceneDesc,
    out: String,
    files: BTreeMap<String, Vec<u8>>,
    names: BTreeMap<StableId, (Ns, &'a str)>,
    counts: BTreeMap<(Ns, &'a str), usize>,
    /// Elements another element names: their name is always written.
    referenced: BTreeSet<StableId>,
    /// The parser's per-owner counters for unnamed elements.
    counters: BTreeMap<String, u32>,
}

impl<'a> Writer<'a> {
    fn new(scene: &'a SceneDesc) -> Self {
        let s = scene;
        let bodies = s.bodies.iter();
        let names: BTreeMap<StableId, (Ns, &str)> = (bodies.clone().map(|b| (b.id, 0, &b.name)))
            .chain(s.joints.iter().map(|j| (j.id, 1, &j.name)))
            .chain(
                bodies
                    .clone()
                    .flat_map(|b| &b.geoms)
                    .map(|g| (g.id, 2, &g.name)),
            )
            .chain(bodies.flat_map(|b| &b.sites).map(|x| (x.id, 3, &x.name)))
            .chain(s.cameras.iter().map(|c| (c.id, 4, &c.name)))
            .chain(s.tendons.iter().map(|t| (t.id, 5, &t.name)))
            .chain(s.actuators.iter().map(|a| (a.id, 6, &a.name)))
            .chain(s.assets.iter().map(|a| (a.id, 7 + a.kind as u8, &a.name)))
            .map(|(id, ns, name)| (id, (ns, name.as_str())))
            .collect();
        let mut counts = BTreeMap::new();
        for key in names.values() {
            *counts.entry(*key).or_insert(0) += 1;
        }
        let mut referenced: BTreeSet<StableId> = s
            .contact_excludes
            .iter()
            .flat_map(|(a, b)| [*a, *b])
            .collect();
        referenced.extend(s.contact_pairs.iter().flat_map(|p| [p.geom1, p.geom2]));
        for t in &s.tendons {
            match &t.kind {
                TendonKind::Fixed { joints } => referenced.extend(joints.iter().map(|j| j.0)),
                TendonKind::Spatial { sites } => referenced.extend(sites.iter().copied()),
            }
        }
        referenced.extend(s.actuators.iter().map(|a| actuator_target(a.target).1));
        referenced.extend(s.sensors.iter().map(|x| sensor_target(x.target).1));
        Self {
            scene,
            out: String::new(),
            files: BTreeMap::new(),
            names,
            counts,
            referenced,
            counters: BTreeMap::new(),
        }
    }

    /// The name another element refers to `id` by; it must be unique in its namespace, or the
    /// reader would resolve it to another element.
    fn r(&self, id: StableId) -> Result<&'a str, WriteError> {
        let Some(&(ns, name)) = self.names.get(&id) else {
            return Err(no!("a reference to {id}, which is not in the scene,"));
        };
        if self.counts[&(ns, name)] > 1 {
            return Err(no!("a reference to `{name}`, a name used twice,"));
        }
        Ok(name)
    }

    /// `name=`, unless `name` is the one the parser generates for an unnamed `tag` here. A name
    /// another element of its namespace also has (a URDF's per-link `visual1`) is qualified by
    /// its owner, `base_link/visual1`: `MuJoCo` refuses a repeated name, so such a scene could
    /// not be opened at all, and its read-back names (and ids) are the qualified ones.
    fn named(&mut self, el: El, tag: &str, owner: &str, id: StableId, name: &str) -> El {
        let key = format!("{owner}/{tag}");
        let next = self.counters.get(&key).copied().unwrap_or(0) + 1;
        if name == format!("{tag}{next}") && !self.referenced.contains(&id) {
            self.counters.insert(key, next);
            return el;
        }
        if self.names.get(&id).is_some_and(|key| self.counts[key] > 1) {
            let parent = owner.rsplit('/').next().unwrap_or(owner);
            return el.s("name", &format!("{parent}/{name}"));
        }
        el.s("name", name)
    }

    fn line(&mut self, depth: usize, El(el): El, close: bool) {
        let close = if close { "/" } else { "" };
        let _ = writeln!(self.out, "{:w$}{el}{close}>", "", w = 2 * depth);
    }

    fn end(&mut self, depth: usize, tag: &str) {
        let _ = writeln!(self.out, "{:w$}</{tag}>", "", w = 2 * depth);
    }

    fn document(&mut self) -> Result<(), WriteError> {
        let o = &self.scene.options;
        self.line(0, El::new("mujoco").s("model", &self.scene.name), false);
        self.line(
            1,
            El::new("compiler")
                .s("angle", "radian")
                .s("autolimits", "true"),
            true,
        );
        let option = El::new("option")
            .s("timestep", &num(o.timestep))
            .s("gravity", &nums(&[o.gravity.x, o.gravity.y, o.gravity.z]))
            .s("integrator", INTEGRATORS[o.integrator as usize])
            .s("cone", CONES[o.cone as usize])
            .s("jacobian", JACOBIANS[o.jacobian as usize])
            .s("solver", SOLVERS[o.solver as usize])
            .s("iterations", &o.iterations.to_string())
            .s("ls_iterations", &o.ls_iterations.to_string())
            .s("impratio", &num(o.impratio));
        self.line(1, option, o.eulerdamp);
        if !o.eulerdamp {
            self.line(2, El::new("flag").s("eulerdamp", "disable"), true);
            self.end(1, "option");
        }
        self.section("asset", Self::assets)?;
        self.worldbody()?;
        self.section("contact", Self::contact)?;
        self.section("tendon", Self::tendons)?;
        self.section("actuator", Self::actuators)?;
        self.section("sensor", Self::sensors)?;
        self.end(0, "mujoco");
        Ok(())
    }

    /// `<tag>` around what `body` writes, or nothing when it writes nothing.
    fn section(
        &mut self,
        tag: &str,
        body: fn(&mut Self) -> Result<(), WriteError>,
    ) -> Result<(), WriteError> {
        let mark = self.out.len();
        self.line(1, El::new(tag), false);
        let inner = self.out.len();
        body(self)?;
        if self.out.len() == inner {
            self.out.truncate(mark);
        } else {
            self.end(1, tag);
        }
        Ok(())
    }

    // ---- <asset> ----------------------------------------------------------------------

    /// Where an asset file goes: its own path when that stays inside `out_dir`.
    fn file_path(a: &AssetRef, path: &str, suffix: &str) -> String {
        let inside = |c: Component| matches!(c, Component::Normal(_) | Component::CurDir);
        if !path.is_empty() && Path::new(path).components().all(inside) {
            return path.to_owned();
        }
        let ext = Path::new(path)
            .extension()
            .map_or(String::new(), |e| format!(".{}", e.to_string_lossy()));
        let stem: String = a
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        // ponytail: two assets whose names sanitise alike collide here; `add_file` refuses
        // that instead of overwriting.
        format!("{}/{stem}{suffix}{ext}", a.kind.tag())
    }

    fn add_file(&mut self, rel: &str, bytes: Vec<u8>) -> Result<(), WriteError> {
        if self.files.get(rel).is_some_and(|old| *old != bytes) {
            return Err(no!("two different files at `{rel}`"));
        }
        self.files.insert(rel.to_owned(), bytes);
        Ok(())
    }

    fn assets(&mut self) -> Result<(), WriteError> {
        for a in &self.scene.assets {
            match a.kind {
                AssetKind::Mesh => self.mesh(a)?,
                AssetKind::Texture => self.texture(a)?,
                AssetKind::Material => self.material(a)?,
                AssetKind::HeightField => {
                    return Err(no!(
                        "height field `{}` (SceneDesc carries no samples)",
                        a.name
                    ))
                }
            }
        }
        Ok(())
    }

    fn mesh(&mut self, a: &AssetRef) -> Result<(), WriteError> {
        let Some(data) = self.scene.meshes.get(&a.id) else {
            return Err(no!("mesh `{}` (not loaded: run mesh::load first)", a.name));
        };
        let ext = Path::new(&a.path)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase());
        let bytes = match ext.as_deref() {
            Some("stl") => stl(&data.positions, &data.indices),
            Some("obj") => obj(&data.positions, data.uvs.as_deref(), &data.indices),
            _ => {
                return Err(no!(
                    "mesh `{}` from `{}` (only .stl and .obj)",
                    a.name,
                    a.path
                ))
            }
        };
        let rel = Self::file_path(a, &a.path, "");
        self.add_file(&rel, bytes)?;
        let scale = self.scene.mesh_scales.get(&a.id).unwrap_or(&[1.0; 3]);
        let el = El::new("mesh").s("name", &a.name).s("file", &rel);
        self.line(2, el.f("scale", scale, &[1.0; 3]), true);
        Ok(())
    }

    fn texture(&mut self, a: &AssetRef) -> Result<(), WriteError> {
        let Some(tex) = self.scene.textures.get(&a.id) else {
            return Err(no!("texture `{}` without a declaration", a.name));
        };
        let (s, d) = (&tex.spec, crate::texture::TextureSpec::default());
        let mut el = El::new("texture")
            .s("name", &a.name)
            .s("type", TEX_KINDS[s.kind as usize])
            .e("colorspace", COLORSPACES[s.colorspace as usize], "auto")
            .e("builtin", BUILTINS[s.builtin as usize], "none")
            .e("mark", MARKS[s.mark as usize], "none")
            .f("rgb1", &s.rgb1, &d.rgb1)
            .f("rgb2", &s.rgb2, &d.rgb2)
            .f("markrgb", &s.markrgb, &d.markrgb)
            .f("random", &[s.random], &[d.random])
            .e("width", s.width, "0")
            .e("height", s.height, "0")
            .e(
                "gridsize",
                format!("{} {}", s.gridsize[0], s.gridsize[1]),
                "1 1",
            )
            .e("gridlayout", &s.gridlayout, &d.gridlayout);
        let has_files = s.file.is_some() || s.cubefiles.iter().any(Option::is_some);
        if has_files && s.builtin != Builtin::None {
            return Err(no!("texture `{}`: a builtin with a file", a.name));
        }
        let data = match (&tex.data, has_files) {
            (_, false) => None,
            (Some(data), true) => Some(data),
            (None, true) => {
                return Err(no!(
                    "texture `{}` (not decoded: a skybox, or not loaded)",
                    a.name
                ))
            }
        };
        // `auto` reads the PNG's own sRGB chunk, so the chunk carries the resolved space.
        let srgb = s.colorspace == ColorSpace::Auto && data.is_some_and(|d| d.srgb);
        if let (Some(file), Some(data)) = (&s.file, data) {
            let (w, h, rgb) = texture_image(data, s.gridsize, &s.gridlayout);
            let rel = Self::file_path(a, file, "");
            self.add_file(&rel, png(w, h, &rgb, srgb)?)?;
            el = el.s("file", &rel);
        }
        for (f, (file, attr)) in s.cubefiles.iter().zip(SIDES).enumerate() {
            let (Some(file), Some(data)) = (file, data) else {
                continue;
            };
            let face = (data.width * data.width * 3) as usize;
            let rel = Self::file_path(a, file, &format!("_{attr}"));
            self.add_file(
                &rel,
                png(
                    data.width,
                    data.width,
                    &data.rgb[f * face..(f + 1) * face],
                    srgb,
                )?,
            )?;
            el = el.s(attr, &rel);
        }
        self.line(2, el, true);
        Ok(())
    }

    fn material(&mut self, a: &AssetRef) -> Result<(), WriteError> {
        let mut el = El::new("material").s("name", &a.name);
        let Some(m) = self.scene.materials.get(&a.id) else {
            // A material that only names a colour: the parser keeps nothing but its name.
            if !a.path.is_empty() {
                return Err(no!("material `{}` naming a texture but not drawn", a.name));
            }
            self.line(2, el, true);
            return Ok(());
        };
        if m.normal_scale.is_some() {
            return Err(no!("material `{}`: a normal-map scale", a.name));
        }
        el = el
            .f("rgba", &m.rgba, &[1.0; 4])
            .opt("specular", m.specular)
            .opt("shininess", m.shininess)
            .opt("metallic", m.metallic)
            .opt("roughness", m.roughness)
            .f("texrepeat", &m.texrepeat, &[1.0, 1.0])
            .e("texuniform", m.texuniform, "false");
        // `emission` is always written -- it is what makes the reader keep a drawn material --
        // except where an emissive layer's strength is the default 1.
        el = match (m.emissive_map, m.emissive) {
            (None, None) => el.s("emission", &num(m.emission)),
            (Some(_), Some([e, g, b])) if same(&[e, e], &[g, b]) && same(&[e], &[m.emission]) => {
                el.s("emission", &num(e))
            }
            (Some(_), Some(e)) if same(&[m.emission], &[0.0]) && same(&e, &[1.0; 3]) => el,
            _ => {
                return Err(no!(
                    "material `{}`: an emissive colour no MJCF emissive layer gives",
                    a.name
                ))
            }
        };
        let maps = [
            ("rgb", m.rgb),
            ("orm", m.orm),
            ("metallic", m.metallic_map),
            ("roughness", m.roughness_map),
            ("normal", m.normal_map),
            ("emissive", m.emissive_map),
        ];
        let layers: Vec<(&str, StableId)> = maps
            .into_iter()
            .filter_map(|(role, t)| Some((role, t?)))
            .collect();
        if !a.path.is_empty() {
            // The `texture` attribute: the asset path is that texture's name.
            if layers.len() != 1 || layers[0].0 != "rgb" || self.r(layers[0].1)? != a.path {
                return Err(no!(
                    "material `{}`: a `texture` attribute together with other maps",
                    a.name
                ));
            }
            self.line(2, el.s("texture", &a.path), true);
            return Ok(());
        }
        self.line(2, el, layers.is_empty());
        if !layers.is_empty() {
            for (role, t) in layers {
                let el = El::new("layer").s("role", role).s("texture", self.r(t)?);
                self.line(3, el, true);
            }
            self.end(2, "material");
        }
        Ok(())
    }

    // ---- <worldbody> ------------------------------------------------------------------

    fn worldbody(&mut self) -> Result<(), WriteError> {
        let s = self.scene;
        let mut children: BTreeMap<Option<StableId>, Vec<&Body>> = BTreeMap::new();
        for b in &s.bodies {
            children.entry(b.parent).or_default().push(b);
        }
        let world = s
            .bodies
            .iter()
            .find(|b| b.parent.is_none() && b.name == "world");
        self.line(1, El::new("worldbody"), false);
        match world {
            Some(world) => self.contents(world, "world", 2)?,
            None => self.cameras(None, "world", 2),
        }
        for root in children.get(&None).into_iter().flatten() {
            if world.is_some_and(|w| w.id == root.id) {
                for child in children.get(&Some(root.id)).into_iter().flatten() {
                    self.body(child, "world", &children, 2)?;
                }
            } else {
                self.body(root, "world", &children, 2)?;
            }
        }
        self.end(1, "worldbody");
        let homes: BTreeSet<StableId> = s.bodies.iter().map(|b| b.id).collect();
        if let Some(c) = s
            .cameras
            .iter()
            .find(|c| c.body.is_some_and(|b| !homes.contains(&b)))
        {
            return Err(no!("camera `{}` on a body not in the scene", c.name));
        }
        Ok(())
    }

    fn body(
        &mut self,
        b: &'a Body,
        parent_path: &str,
        children: &BTreeMap<Option<StableId>, Vec<&'a Body>>,
        depth: usize,
    ) -> Result<(), WriteError> {
        let el = self.named(El::new("body"), "body", parent_path, b.id, &b.name);
        let gravcomp = self.scene.gravcomp.get(&b.id).copied();
        self.line(depth, el.pose(b.pose).opt("gravcomp", gravcomp), false);
        let path = format!("{parent_path}/{}", b.name);
        self.contents(b, &path, depth + 1)?;
        for child in children.get(&Some(b.id)).into_iter().flatten() {
            self.body(child, &path, children, depth + 1)?;
        }
        self.end(depth, "body");
        Ok(())
    }

    /// A body's own elements: inertial, joints, geoms, sites, cameras.
    fn contents(&mut self, b: &'a Body, path: &str, depth: usize) -> Result<(), WriteError> {
        let world = path == "world";
        let joints: Vec<_> = self
            .scene
            .joints
            .iter()
            .filter(|j| j.body == b.id)
            .collect();
        if world {
            if let Some(j) = joints.first() {
                return Err(no!("joint `{}` on the world body", j.name));
            }
            let posed = b.pose.position != Vec3::ZERO || b.pose.orientation != Quat::IDENTITY;
            if posed || b.inertial.is_some() || self.scene.gravcomp.contains_key(&b.id) {
                return Err(no!("a pose, inertial or gravcomp on the world body"));
            }
        }
        if let Some(i) = &b.inertial {
            let m = i.inertia.matrix();
            let el = El::new("inertial")
                .s("pos", &nums(&[i.com.x, i.com.y, i.com.z]))
                .s("mass", &num(i.mass));
            let el = if same(&[m[0][1], m[0][2], m[1][2]], &[0.0; 3]) {
                el.s("diaginertia", &nums(&[m[0][0], m[1][1], m[2][2]]))
                    .quat(i.frame, 1)
            } else if i.frame == Quat::IDENTITY {
                el.s(
                    "fullinertia",
                    &nums(&[m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]),
                )
            } else {
                return Err(no!(
                    "body `{}`: a full inertia tensor in a rotated frame",
                    b.name
                ));
            };
            self.line(depth, el, true);
        }
        for j in joints {
            if j.kind == JointKind::Fixed {
                // Welded: MJCF spells it as a body without a joint.
                let name = esc(&j.name).replace("--", "- -");
                let _ = writeln!(
                    self.out,
                    "{:w$}<!-- fixed joint `{name}`: welded -->",
                    "",
                    w = 2 * depth
                );
                continue;
            }
            let plain = j.range.is_none()
                && j.anchor == Vec3::ZERO
                && j.axis == Vec3::new(0.0, 0.0, 1.0)
                && same(
                    &[
                        j.damping,
                        j.armature,
                        j.stiffness,
                        j.friction_loss,
                        j.spring_ref,
                    ],
                    &[0.0; 5],
                );
            let auto_free = format!(
                "freejoint{}",
                self.counters
                    .get(&format!("{path}/freejoint"))
                    .unwrap_or(&0)
                    + 1
            );
            let tag = if j.kind == JointKind::Free && plain && j.name == auto_free {
                "freejoint"
            } else {
                "joint"
            };
            let mut el = self.named(El::new(tag), tag, path, j.id, &j.name);
            if tag == "joint" {
                let axis = preimage([j.axis.x, j.axis.y, j.axis.z], |[x, y, z]| {
                    let n = Vec3::new(x, y, z).normalize();
                    [n.x, n.y, n.z]
                });
                el = el
                    .e(
                        "type",
                        ["free", "ball", "hinge", "slide"][j.kind as usize],
                        "hinge",
                    )
                    .pos(j.anchor)
                    .f("axis", &axis, &[0.0, 0.0, 1.0])
                    .range("range", j.range)
                    .f("damping", &[j.damping], &[0.0])
                    .f("armature", &[j.armature], &[0.0])
                    .f("stiffness", &[j.stiffness], &[0.0])
                    .f("frictionloss", &[j.friction_loss], &[0.0])
                    .f("springref", &[j.spring_ref], &[0.0]);
            }
            self.line(depth, el, true);
        }
        for g in &b.geoms {
            let el = self.geom(g, path)?;
            self.line(depth, el, true);
        }
        for site in &b.sites {
            let el = self.named(El::new("site"), "site", path, site.id, &site.name);
            let size = [site.size.x, site.size.y, site.size.z];
            self.line(
                depth,
                el.pose(site.pose).f("size", &size, &[0.005; 3]),
                true,
            );
        }
        self.cameras((!world).then_some(b.id), path, depth);
        Ok(())
    }

    fn cameras(&mut self, body: Option<StableId>, path: &str, depth: usize) {
        for c in self.scene.cameras.iter().filter(|c| c.body == body) {
            let el = self.named(El::new("camera"), "camera", path, c.id, &c.name);
            let deg = c.fovy / DEG_TO_RAD;
            let deg = (0..3)
                .map(|k| nudge(deg, k))
                .find(|d| same(&[d * DEG_TO_RAD], &[c.fovy]))
                .unwrap_or(deg);
            self.line(depth, el.pose(c.pose).f("fovy", &[deg], &[45.0]), true);
        }
    }

    fn geom(&mut self, g: &Geom, path: &str) -> Result<El, WriteError> {
        if g.visual_only != (g.contype == 0 && g.conaffinity == 0) {
            return Err(no!(
                "geom `{}`: visual_only {} with contype {} conaffinity {}",
                g.name,
                g.visual_only,
                g.contype,
                g.conaffinity
            ));
        }
        let el = self.named(El::new("geom"), "geom", path, g.id, &g.name);
        let (kind, size) = match g.shape {
            Shape::Plane {
                half_x,
                half_y,
                grid,
            } => ("plane", vec![half_x, half_y, grid]),
            Shape::Sphere { radius } => ("sphere", vec![radius]),
            Shape::Capsule {
                radius,
                half_length,
            } => ("capsule", vec![radius, half_length]),
            Shape::Cylinder {
                radius,
                half_length,
            } => ("cylinder", vec![radius, half_length]),
            Shape::Box { half_extents: v } => ("box", vec![v.x, v.y, v.z]),
            Shape::Ellipsoid { radii: v } => ("ellipsoid", vec![v.x, v.y, v.z]),
            Shape::Mesh { .. } | Shape::HeightField { .. } => ("", vec![]),
        };
        let mut el = match g.shape {
            Shape::Mesh { asset } => el.s("type", "mesh").s("mesh", self.r(asset)?),
            Shape::HeightField { asset } => el.s("type", "hfield").s("hfield", self.r(asset)?),
            _ => el.s("type", kind).s("size", &nums(&size)),
        };
        el = el
            .pose(g.pose)
            .f("friction", &g.friction, &[1.0, 0.005, 0.0001])
            .e("contype", g.contype, "1")
            .e("conaffinity", g.conaffinity, "1")
            .e("condim", g.condim, "3")
            .e("priority", g.priority, "0")
            .f("density", &[g.density], &[1000.0])
            .opt("mass", g.mass)
            .f("margin", &[g.margin], &[0.0])
            .f("gap", &[g.gap], &[0.0])
            .f("solref", &g.solref, &[0.02, 1.0])
            .f("solimp", &g.solimp, &[0.9, 0.95, 0.001, 0.5, 2.0])
            .f("rgba", &g.rgba, &[0.5, 0.5, 0.5, 1.0]);
        if let Some(m) = g.material {
            el = el.s("material", self.r(m)?);
        }
        Ok(el)
    }

    // ---- sections that name elements ---------------------------------------------------

    fn contact(&mut self) -> Result<(), WriteError> {
        let s = self.scene;
        for p in &s.contact_pairs {
            let el = El::new("pair")
                .s("geom1", self.r(p.geom1)?)
                .s("geom2", self.r(p.geom2)?)
                .e(
                    "condim",
                    p.condim.map_or(String::new(), |c| c.to_string()),
                    "",
                )
                .opt_vec("friction", p.friction.map(|v| v.to_vec()))
                .opt_vec("solref", p.solref.map(|v| v.to_vec()))
                .opt_vec("solimp", p.solimp.map(|v| v.to_vec()))
                .opt("margin", p.margin)
                .opt("gap", p.gap);
            self.line(2, el, true);
        }
        for (a, b) in &s.contact_excludes {
            let el = El::new("exclude")
                .s("body1", self.r(*a)?)
                .s("body2", self.r(*b)?);
            self.line(2, el, true);
        }
        Ok(())
    }

    fn tendons(&mut self) -> Result<(), WriteError> {
        for t in &self.scene.tendons {
            let tag = match t.kind {
                TendonKind::Fixed { .. } => "fixed",
                TendonKind::Spatial { .. } => "spatial",
            };
            let el = El::new(tag)
                .s("name", &t.name)
                .range("range", t.range)
                .f("stiffness", &[t.stiffness], &[0.0])
                .f("damping", &[t.damping], &[0.0]);
            self.line(2, el, false);
            match &t.kind {
                TendonKind::Fixed { joints } => {
                    for (j, coef) in joints {
                        let el = El::new("joint")
                            .s("joint", self.r(*j)?)
                            .s("coef", &num(*coef));
                        self.line(3, el, true);
                    }
                }
                TendonKind::Spatial { sites } => {
                    for site in sites {
                        let el = El::new("site").s("site", self.r(*site)?);
                        self.line(3, el, true);
                    }
                }
            }
            self.end(2, tag);
        }
        Ok(())
    }

    fn actuators(&mut self) -> Result<(), WriteError> {
        for a in &self.scene.actuators {
            let el =
                match a.kind {
                    ActuatorKind::Motor => El::new("motor"),
                    ActuatorKind::Position { kp, kv } => El::new("position")
                        .f("kp", &[kp], &[1.0])
                        .f("kv", &[kv], &[0.0]),
                    ActuatorKind::Velocity { kv } => El::new("velocity").f("kv", &[kv], &[1.0]),
                    // `MuJoCo` reads the affine terms only under the affine types.
                    ActuatorKind::General { gain, bias } => El::new("general")
                        .f("gainprm", &gain, &[1.0, 0.0, 0.0])
                        .f("biasprm", &bias, &[0.0; 3])
                        .e(
                            "gaintype",
                            if same(&gain[1..], &[0.0; 2]) {
                                "fixed"
                            } else {
                                "affine"
                            },
                            "fixed",
                        )
                        .e(
                            "biastype",
                            if same(&bias, &[0.0; 3]) {
                                "none"
                            } else {
                                "affine"
                            },
                            "none",
                        ),
                };
            let (attr, id) = actuator_target(a.target);
            let el = el
                .s("name", &a.name)
                .s(attr, self.r(id)?)
                .f("gear", &a.gear, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .range("ctrlrange", a.ctrl_range)
                .range("forcerange", a.force_range);
            self.line(2, el, true);
        }
        Ok(())
    }

    fn sensors(&mut self) -> Result<(), WriteError> {
        for s in &self.scene.sensors {
            let (kind, id) = sensor_target(s.target);
            let attr = match s.kind {
                SensorKind::JointPos | SensorKind::JointVel => "joint",
                SensorKind::ActuatorFrc => "actuator",
                SensorKind::Camera => "camera",
                SensorKind::FramePos | SensorKind::FrameQuat => "objname",
                _ => "site",
            };
            let fits = if attr == "objname" {
                !matches!(kind, "joint" | "actuator")
            } else {
                attr == kind
            };
            if !fits {
                return Err(no!(
                    "sensor `{}`: a {:?} sensor on a {kind}",
                    s.name,
                    s.kind
                ));
            }
            let mut el = El::new(SENSORS[s.kind as usize]);
            if attr == "objname" {
                el = el.s("objtype", kind);
            }
            let el = el
                .s(attr, self.r(id)?)
                .s("name", &s.name)
                .f("noise", &[s.noise], &[0.0])
                .f("cutoff", &[s.cutoff], &[0.0]);
            self.line(2, el, true);
        }
        Ok(())
    }
}

fn actuator_target(t: ActuatorTarget) -> (&'static str, StableId) {
    match t {
        ActuatorTarget::Joint(id) => ("joint", id),
        ActuatorTarget::Tendon(id) => ("tendon", id),
        ActuatorTarget::Site(id) => ("site", id),
    }
}

fn sensor_target(t: SensorTarget) -> (&'static str, StableId) {
    match t {
        SensorTarget::Body(id) => ("body", id),
        SensorTarget::Joint(id) => ("joint", id),
        SensorTarget::Geom(id) => ("geom", id),
        SensorTarget::Site(id) => ("site", id),
        SensorTarget::Camera(id) => ("camera", id),
        SensorTarget::Actuator(id) => ("actuator", id),
    }
}

// ---- asset files ------------------------------------------------------------------------

/// Binary STL with zero facet normals, so the reader keeps the stored winding; its
/// first-seen deduplication gives back the same positions and indices.
fn stl(positions: &[[f32; 3]], indices: &[u32]) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out.extend_from_slice(
        &u32::try_from(indices.len() / 3)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for tri in indices.chunks_exact(3) {
        out.extend_from_slice(&[0u8; 12]);
        for c in tri.iter().flat_map(|&i| positions[i as usize]) {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out.extend_from_slice(&[0u8; 2]);
    }
    out
}

/// OBJ: positions in order and the triangles; with UVs, one `vt` per vertex whose `1 - v` is
/// the stored (flipped) coordinate.
fn obj(positions: &[[f32; 3]], uvs: Option<&[[f32; 2]]>, indices: &[u32]) -> Vec<u8> {
    let mut out = String::new();
    for p in positions {
        let _ = writeln!(out, "v {:?} {:?} {:?}", p[0], p[1], p[2]);
    }
    for t in uvs.unwrap_or_default() {
        let v0 = 1.0 - t[1];
        let v = [0u32, 1, u32::MAX, 2, u32::MAX - 1]
            .map(|d| f32::from_bits(v0.to_bits().wrapping_add(d)))
            .into_iter()
            .find(|v| (1.0 - v).to_bits() == t[1].to_bits())
            .unwrap_or(v0);
        let _ = writeln!(out, "vt {:?} {v:?}", t[0]);
    }
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0] + 1, tri[1] + 1, tri[2] + 1];
        let _ = match uvs {
            Some(_) => writeln!(out, "f {a}/{a} {b}/{b} {c}/{c}"),
            None => writeln!(out, "f {a} {b} {c}"),
        };
    }
    out.into_bytes()
}

/// The single image a texture's `file` was cut from: a 2D texture as it is, a cube's
/// `gridsize` x `gridlayout` sheet (one face for a 1 x 1 grid; undeclared cells black).
fn texture_image(data: &TextureData, [rows, cols]: [u32; 2], layout: &str) -> (u32, u32, Vec<u8>) {
    let w = data.width;
    let face = (w * w * 3) as usize;
    if data.kind == TexKind::TwoD {
        return (w, data.height, data.rgb.clone());
    }
    if rows * cols == 1 {
        return (w, w, data.rgb[..face].to_vec());
    }
    let (width, row) = (cols * w, (w * 3) as usize);
    let mut rgb = vec![0u8; (width * rows * w * 3) as usize];
    for (k, symbol) in layout.chars().enumerate() {
        let Some(f) = "RLUDFB".find(symbol) else {
            continue;
        };
        let (r0, c0) = (w * (k as u32 / cols), w * (k as u32 % cols));
        for j in 0..w {
            let (dst, src) = (
                (((j + r0) * width + c0) * 3) as usize,
                f * face + j as usize * row,
            );
            rgb[dst..dst + row].copy_from_slice(&data.rgb[src..src + row]);
        }
    }
    (width, rows * w, rgb)
}

/// 8-bit RGB PNG, with an sRGB chunk when the texture's `auto` colour space resolved to it.
fn png(w: u32, h: u32, rgb: &[u8], srgb: bool) -> Result<Vec<u8>, WriteError> {
    let fail = |e: png::EncodingError| WriteError::Io {
        path: "<png>".to_owned(),
        reason: e.to_string(),
    };
    let mut buf = Vec::new();
    let mut enc = png::Encoder::new(&mut buf, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    if srgb {
        enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    }
    let mut writer = enc.write_header().map_err(fail)?;
    writer.write_image_data(rgb).map_err(fail)?;
    writer.finish().map_err(fail)?;
    Ok(buf)
}
