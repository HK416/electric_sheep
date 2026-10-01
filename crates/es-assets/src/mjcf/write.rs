//! [`SceneDesc`] to MJCF, full fidelity (plan G, packet G2).
//!
//! The export a person opens in `MuJoCo`'s viewer: everything the scene carries — bodies,
//! joints, geoms, sites, cameras, `_light` geoms, materials and textures, meshes, tendons,
//! actuators, sensors, contact pairs and excludes, `gravcomp`, `<option>` — written so that
//! [`super::parse_str`] of the text, followed by [`crate::mesh::load`] of `out_dir`, gives the
//! scene back field for field, with the same `scene_hash` and asset hashes. Unlike
//! `es-physics-backend`'s `mjcf_out`, which drops what carries no dynamics, nothing is
//! dropped: what MJCF cannot say is a [`WriteError::Inexpressible`] naming it.
//!
//! How the round trip is kept exact:
//!
//! * Angles are radians (`<compiler angle="radian">`), so ranges read back unscaled; `fovy`,
//!   always degrees in MJCF, is written as the degree value whose product with `DEG_TO_RAD` is
//!   the stored radian value.
//! * The parser normalises every `quat` and joint `axis` it reads, and normalising is not
//!   bitwise idempotent (`es-math`), so the value written is a pre-image: the stored one, or
//!   the neighbour within an ULP per component that normalises onto it.
//! * Names the parser generated for unnamed elements (`geom3`) are left out again, so `MuJoCo`
//!   sees no duplicate name; every other name is written.
//! * Asset files are re-encoded from the decoded data (STL, OBJ, PNG) at the asset's own
//!   scene-relative path — the path is hash input (spec 5.3) — or, for an absolute or `..`
//!   path, under `<kind>/<name>.<ext>`, which then moves that asset's path digest. A file
//!   already in `out_dir` with other bytes is never overwritten.
//! * Every number is Rust's shortest round-trip decimal.

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
    ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Body, FrictionCone, Geom, Integrator,
    Jacobian, JointKind, SceneDesc, SensorKind, SensorTarget, Shape, Solver, TendonKind,
};
use crate::texture::{Builtin, ColorSpace, Mark, TexKind, TextureData};

/// Why a scene was not written.
#[derive(Debug, Error)]
pub enum WriteError {
    /// Something the scene carries that MJCF (or this crate's MJCF reader) cannot say.
    #[error("{0} cannot be written as MJCF")]
    Inexpressible(String),
    #[error("{path}: {reason}")]
    Io { path: String, reason: String },
}

fn no(what: impl Into<String>) -> WriteError {
    WriteError::Inexpressible(what.into())
}

/// Writes `scene` as MJCF text and its mesh and texture files under `out_dir`; the caller
/// writes the returned text to a file in `out_dir`.
pub fn write_mjcf(scene: &SceneDesc, out_dir: &Path) -> Result<String, WriteError> {
    let mut w = Writer::new(scene);
    w.document()?;
    for (rel, bytes) in &w.files {
        let path = out_dir.join(rel);
        let io = |e: std::io::Error| WriteError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        };
        match std::fs::read(&path) {
            Ok(old) if &old == bytes => continue,
            Ok(_) => {
                return Err(WriteError::Io {
                    path: path.display().to_string(),
                    reason: "exists with other bytes; not overwritten".to_owned(),
                })
            }
            Err(_) => {}
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        std::fs::write(&path, bytes).map_err(io)?;
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
    if k % 2 == 1 {
        f64::from_bits(v.to_bits().wrapping_add(step))
    } else {
        f64::from_bits(v.to_bits().wrapping_sub(step))
    }
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
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let scale = 1.0 + sign * f64::from(k) * 1e-9;
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

/// `<texture>`'s six single-face file attributes, in face order.
const SIDES: [&str; 6] = [
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
];

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

    fn i(self, k: &str, v: i64, default: i64) -> Self {
        if v == default {
            self
        } else {
            self.s(k, &v.to_string())
        }
    }

    fn opt(self, k: &str, v: Option<f64>) -> Self {
        match v {
            Some(v) => self.s(k, &num(v)),
            None => self,
        }
    }

    fn range(self, k: &str, v: Option<(f64, f64)>) -> Self {
        match v {
            Some((lo, hi)) => self.s(k, &nums(&[lo, hi])),
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
            let mut n = Quat::from_xyzw(x, y, z, w);
            for _ in 0..times {
                n = n.normalize();
            }
            [n.x, n.y, n.z, n.w]
        });
        self.s("quat", &nums(&[w, x, y, z]))
    }
}

/// MJCF name namespaces, in the parser's order of `Names`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Ns {
    Body,
    Joint,
    Geom,
    Site,
    Camera,
    Tendon,
    Actuator,
    Asset(u8),
}

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
        let mut names = BTreeMap::new();
        for b in &scene.bodies {
            names.insert(b.id, (Ns::Body, b.name.as_str()));
            names.extend(b.geoms.iter().map(|g| (g.id, (Ns::Geom, g.name.as_str()))));
            names.extend(b.sites.iter().map(|s| (s.id, (Ns::Site, s.name.as_str()))));
        }
        names.extend(
            scene
                .joints
                .iter()
                .map(|j| (j.id, (Ns::Joint, j.name.as_str()))),
        );
        names.extend(
            scene
                .cameras
                .iter()
                .map(|c| (c.id, (Ns::Camera, c.name.as_str()))),
        );
        names.extend(
            scene
                .tendons
                .iter()
                .map(|t| (t.id, (Ns::Tendon, t.name.as_str()))),
        );
        names.extend(
            scene
                .actuators
                .iter()
                .map(|a| (a.id, (Ns::Actuator, a.name.as_str()))),
        );
        names.extend(
            scene
                .assets
                .iter()
                .map(|a| (a.id, (Ns::Asset(a.kind as u8), a.name.as_str()))),
        );
        let mut counts = BTreeMap::new();
        for key in names.values() {
            *counts.entry(*key).or_insert(0) += 1;
        }
        let mut referenced = BTreeSet::new();
        for p in &scene.contact_pairs {
            referenced.extend([p.geom1, p.geom2]);
        }
        for (a, b) in &scene.contact_excludes {
            referenced.extend([*a, *b]);
        }
        for t in &scene.tendons {
            match &t.kind {
                TendonKind::Fixed { joints } => referenced.extend(joints.iter().map(|j| j.0)),
                TendonKind::Spatial { sites } => referenced.extend(sites.iter().copied()),
            }
        }
        for a in &scene.actuators {
            let (ActuatorTarget::Joint(id) | ActuatorTarget::Tendon(id) | ActuatorTarget::Site(id)) =
                a.target;
            referenced.insert(id);
        }
        for s in &scene.sensors {
            referenced.insert(sensor_target(s.target).1);
        }
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
        let (ns, name) = self
            .names
            .get(&id)
            .copied()
            .ok_or_else(|| no(format!("a reference to {id}, which is not in the scene,")))?;
        if self.counts[&(ns, name)] > 1 {
            return Err(no(format!("a reference to `{name}`, a name used twice,")));
        }
        Ok(name)
    }

    /// `name=`, unless `name` is the one the parser generates for an unnamed `tag` here.
    fn named(&mut self, el: El, tag: &str, owner: &str, id: StableId, name: &str) -> El {
        let key = format!("{owner}/{tag}");
        let next = self.counters.get(&key).copied().unwrap_or(0) + 1;
        if name == format!("{tag}{next}") && !self.referenced.contains(&id) {
            self.counters.insert(key, next);
            el
        } else {
            el.s("name", name)
        }
    }

    fn line(&mut self, depth: usize, El(el): El, close: bool) {
        let _ = writeln!(
            self.out,
            "{:w$}{}{}>",
            "",
            el,
            if close { "/" } else { "" },
            w = 2 * depth
        );
    }

    fn end(&mut self, depth: usize, tag: &str) {
        let _ = writeln!(self.out, "{:w$}</{tag}>", "", w = 2 * depth);
    }

    fn document(&mut self) -> Result<(), WriteError> {
        let s = self.scene;
        self.line(0, El::new("mujoco").s("model", &s.name), false);
        self.line(
            1,
            El::new("compiler")
                .s("angle", "radian")
                .s("autolimits", "true"),
            true,
        );
        self.option();
        self.assets()?;
        self.worldbody()?;
        self.contact()?;
        self.tendons()?;
        self.actuators()?;
        self.sensors()?;
        self.end(0, "mujoco");
        Ok(())
    }

    fn option(&mut self) {
        let o = &self.scene.options;
        let el = El::new("option")
            .s("timestep", &num(o.timestep))
            .s("gravity", &nums(&[o.gravity.x, o.gravity.y, o.gravity.z]))
            .s(
                "integrator",
                match o.integrator {
                    Integrator::Euler => "Euler",
                    Integrator::Rk4 => "RK4",
                    Integrator::Implicit => "implicit",
                    Integrator::ImplicitFast => "implicitfast",
                },
            )
            .s(
                "cone",
                match o.cone {
                    FrictionCone::Pyramidal => "pyramidal",
                    FrictionCone::Elliptic => "elliptic",
                },
            )
            .s(
                "jacobian",
                match o.jacobian {
                    Jacobian::Dense => "dense",
                    Jacobian::Sparse => "sparse",
                    Jacobian::Auto => "auto",
                },
            )
            .s(
                "solver",
                match o.solver {
                    Solver::Pgs => "PGS",
                    Solver::Cg => "CG",
                    Solver::Newton => "Newton",
                },
            )
            .s("iterations", &o.iterations.to_string())
            .s("ls_iterations", &o.ls_iterations.to_string())
            .s("impratio", &num(o.impratio));
        if o.eulerdamp {
            self.line(1, el, true);
        } else {
            self.line(1, el, false);
            self.line(2, El::new("flag").s("eulerdamp", "disable"), true);
            self.end(1, "option");
        }
    }

    // ---- <asset> ----------------------------------------------------------------------

    /// Where an asset file goes: its own path when that stays inside `out_dir`.
    fn file_path(a: &AssetRef, path: &str, suffix: &str) -> String {
        let safe = !path.is_empty()
            && Path::new(path)
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
        if safe {
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
        // ponytail: two assets whose names sanitise alike collide here; the conflict check in
        // `add_file` refuses that instead of overwriting.
        format!("{}/{stem}{suffix}{ext}", a.kind.tag())
    }

    fn add_file(&mut self, rel: String, bytes: Vec<u8>) -> Result<(), WriteError> {
        match self.files.get(&rel) {
            Some(old) if *old != bytes => Err(no(format!("two different files at `{rel}`"))),
            _ => {
                self.files.insert(rel, bytes);
                Ok(())
            }
        }
    }

    fn assets(&mut self) -> Result<(), WriteError> {
        if self.scene.assets.is_empty() {
            return Ok(());
        }
        self.line(1, El::new("asset"), false);
        for a in &self.scene.assets {
            match a.kind {
                AssetKind::Mesh => self.mesh(a)?,
                AssetKind::Texture => self.texture(a)?,
                AssetKind::Material => self.material(a)?,
                AssetKind::HeightField => {
                    return Err(no(format!(
                        "height field `{}` (SceneDesc carries no samples)",
                        a.name
                    )))
                }
            }
        }
        self.end(1, "asset");
        Ok(())
    }

    fn mesh(&mut self, a: &AssetRef) -> Result<(), WriteError> {
        let data = self.scene.meshes.get(&a.id).ok_or_else(|| {
            no(format!(
                "mesh `{}` (not loaded: run mesh::load first)",
                a.name
            ))
        })?;
        let ext = Path::new(&a.path)
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let bytes = match ext.as_str() {
            "stl" => stl(&data.positions, &data.indices),
            "obj" => obj(&data.positions, data.uvs.as_deref(), &data.indices),
            _ => {
                return Err(no(format!(
                    "mesh `{}` from `{}` (MJCF meshes here are .stl or .obj)",
                    a.name, a.path
                )))
            }
        };
        let rel = Self::file_path(a, &a.path, "");
        self.add_file(rel.clone(), bytes)?;
        self.line(2, El::new("mesh").s("name", &a.name).s("file", &rel), true);
        Ok(())
    }

    fn texture(&mut self, a: &AssetRef) -> Result<(), WriteError> {
        let tex = self
            .scene
            .textures
            .get(&a.id)
            .ok_or_else(|| no(format!("texture `{}` without a declaration", a.name)))?;
        let s = &tex.spec;
        let d = crate::texture::TextureSpec::default();
        let mut el = El::new("texture")
            .s("name", &a.name)
            .s(
                "type",
                match s.kind {
                    TexKind::TwoD => "2d",
                    TexKind::Cube => "cube",
                    TexKind::Skybox => "skybox",
                },
            )
            .f("rgb1", &s.rgb1, &d.rgb1)
            .f("rgb2", &s.rgb2, &d.rgb2)
            .f("markrgb", &s.markrgb, &d.markrgb)
            .f("random", &[s.random], &[d.random]);
        for (attr, value, default) in [
            (
                "colorspace",
                match s.colorspace {
                    ColorSpace::Auto => "auto",
                    ColorSpace::Srgb => "sRGB",
                    ColorSpace::Linear => "linear",
                },
                "auto",
            ),
            (
                "builtin",
                match s.builtin {
                    Builtin::None => "none",
                    Builtin::Gradient => "gradient",
                    Builtin::Checker => "checker",
                    Builtin::Flat => "flat",
                },
                "none",
            ),
            (
                "mark",
                match s.mark {
                    Mark::None => "none",
                    Mark::Edge => "edge",
                    Mark::Cross => "cross",
                    Mark::Random => "random",
                },
                "none",
            ),
        ] {
            if value != default {
                el = el.s(attr, value);
            }
        }
        for (attr, v) in [("width", s.width), ("height", s.height)] {
            if v != 0 {
                el = el.s(attr, &v.to_string());
            }
        }
        if s.gridsize != d.gridsize {
            el = el.s("gridsize", &format!("{} {}", s.gridsize[0], s.gridsize[1]));
        }
        if s.gridlayout != d.gridlayout {
            el = el.s("gridlayout", &s.gridlayout);
        }
        let has_files = s.file.is_some() || s.cubefiles.iter().any(Option::is_some);
        if has_files && s.builtin != Builtin::None {
            return Err(no(format!("texture `{}`: a builtin with a file", a.name)));
        }
        let data = match (&tex.data, has_files) {
            (_, false) => None,
            (Some(data), true) => Some(data),
            (None, true) => {
                return Err(no(format!(
                    "texture `{}` (not decoded: a skybox, or the scene was not loaded)",
                    a.name
                )))
            }
        };
        // `auto` reads the PNG's own sRGB chunk, so the chunk carries the resolved space.
        let srgb_chunk = s.colorspace == ColorSpace::Auto && data.is_some_and(|d| d.srgb);
        if let (Some(file), Some(data)) = (&s.file, data) {
            let (w, h, rgb) = texture_image(data, s.gridsize, &s.gridlayout);
            let rel = Self::file_path(a, file, "");
            self.add_file(rel.clone(), png(w, h, &rgb, srgb_chunk)?)?;
            el = el.s("file", &rel);
        }
        for (f, (file, attr)) in s.cubefiles.iter().zip(SIDES).enumerate() {
            let (Some(file), Some(data)) = (file, data) else {
                continue;
            };
            let face = (data.width * data.width * 3) as usize;
            let rgb = data.rgb[f * face..(f + 1) * face].to_vec();
            let rel = Self::file_path(a, file, &format!("_{attr}"));
            self.add_file(rel.clone(), png(data.width, data.width, &rgb, srgb_chunk)?)?;
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
                return Err(no(format!(
                    "material `{}` naming a texture but not drawn",
                    a.name
                )));
            }
            self.line(2, el, true);
            return Ok(());
        };
        if m.normal_scale.is_some() {
            return Err(no(format!("material `{}`: a normal-map scale", a.name)));
        }
        el = el
            .f("rgba", &m.rgba, &[1.0; 4])
            .opt("specular", m.specular)
            .opt("shininess", m.shininess)
            .opt("metallic", m.metallic)
            .opt("roughness", m.roughness)
            .f("texrepeat", &m.texrepeat, &[1.0, 1.0]);
        if m.texuniform {
            el = el.s("texuniform", "true");
        }
        // `emission` is always written: it is what makes the reader keep a drawn material.
        el = match (m.emissive_map, m.emissive) {
            (None, None) => el.s("emission", &num(m.emission)),
            (Some(_), Some([e, g, b])) if same(&[e, e], &[g, b]) => {
                if same(&[e], &[m.emission]) {
                    el.s("emission", &num(e))
                } else if same(&[m.emission, e], &[0.0, 1.0]) {
                    el
                } else {
                    return Err(no(format!(
                        "material `{}`: emission {} with an emissive map of strength {e}",
                        a.name, m.emission
                    )));
                }
            }
            _ => {
                return Err(no(format!(
                    "material `{}`: an emissive colour other than an MJCF emissive layer's",
                    a.name
                )))
            }
        };
        let layers: Vec<(&str, StableId)> = [
            ("rgb", m.rgb),
            ("orm", m.orm),
            ("metallic", m.metallic_map),
            ("roughness", m.roughness_map),
            ("normal", m.normal_map),
            ("emissive", m.emissive_map),
        ]
        .into_iter()
        .filter_map(|(role, t)| t.map(|t| (role, t)))
        .collect();
        if !a.path.is_empty() {
            // The `texture` attribute: the asset path is that texture's name.
            if layers.len() != 1 || layers[0].0 != "rgb" || self.r(layers[0].1)? != a.path {
                return Err(no(format!(
                    "material `{}`: a `texture` attribute together with other maps",
                    a.name
                )));
            }
            self.line(2, el.s("texture", &a.path), true);
            return Ok(());
        }
        if layers.is_empty() {
            self.line(2, el, true);
            return Ok(());
        }
        self.line(2, el, false);
        for (role, t) in layers {
            let name = self.r(t)?;
            self.line(3, El::new("layer").s("role", role).s("texture", name), true);
        }
        self.end(2, "material");
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
        if let Some(world) = world {
            self.contents(world, "world", 2)?;
        } else {
            self.cameras(None, "world", 2);
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
            return Err(no(format!(
                "camera `{}` on a body not in the scene",
                c.name
            )));
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
        let mut el = El::new("body");
        el = self.named(el, "body", parent_path, b.id, &b.name);
        el = el.pose(b.pose);
        if let Some(g) = self.scene.gravcomp.get(&b.id) {
            el = el.s("gravcomp", &num(*g));
        }
        let path = format!("{parent_path}/{}", b.name);
        self.line(depth, el, false);
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
        if world && (!b.pose.position.eq(&Vec3::ZERO) || b.pose.orientation != Quat::IDENTITY) {
            return Err(no("a pose on the world body"));
        }
        if world && self.scene.gravcomp.contains_key(&b.id) {
            return Err(no("gravcomp on the world body"));
        }
        if let Some(i) = &b.inertial {
            if world {
                return Err(no("an inertial on the world body"));
            }
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
                return Err(no(format!(
                    "body `{}`: a full inertia tensor in a rotated frame",
                    b.name
                )));
            };
            self.line(depth, el, true);
        }
        for j in self.scene.joints.iter().filter(|j| j.body == b.id) {
            if world {
                return Err(no(format!("joint `{}` on the world body", j.name)));
            }
            let defaults = j.range.is_none()
                && same(
                    &[
                        j.axis.x,
                        j.axis.y,
                        j.axis.z,
                        j.anchor.x,
                        j.anchor.y,
                        j.anchor.z,
                        j.damping,
                        j.armature,
                        j.stiffness,
                        j.friction_loss,
                        j.spring_ref,
                    ],
                    &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                );
            let tag = match j.kind {
                JointKind::Fixed => {
                    // Welded: MJCF spells it as a body without a joint.
                    let _ = writeln!(
                        self.out,
                        "{:w$}<!-- fixed joint `{}`: welded, no MJCF joint -->",
                        "",
                        esc(&j.name).replace("--", "- -"),
                        w = 2 * depth
                    );
                    continue;
                }
                JointKind::Free
                    if defaults
                        && j.name
                            == format!(
                                "freejoint{}",
                                self.counters
                                    .get(&format!("{path}/freejoint"))
                                    .copied()
                                    .unwrap_or(0)
                                    + 1
                            ) =>
                {
                    "freejoint"
                }
                _ => "joint",
            };
            let mut el = El::new(tag);
            el = self.named(el, tag, path, j.id, &j.name);
            if tag == "joint" {
                let kind = match j.kind {
                    JointKind::Free => "free",
                    JointKind::Ball => "ball",
                    JointKind::Slide => "slide",
                    _ => "hinge",
                };
                if kind != "hinge" {
                    el = el.s("type", kind);
                }
                let axis = preimage([j.axis.x, j.axis.y, j.axis.z], |[x, y, z]| {
                    let n = Vec3::new(x, y, z).normalize();
                    [n.x, n.y, n.z]
                });
                el = el
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
            let mut el = El::new("site");
            el = self.named(el, "site", path, site.id, &site.name);
            el = el.pose(site.pose).f(
                "size",
                &[site.size.x, site.size.y, site.size.z],
                &[0.005; 3],
            );
            self.line(depth, el, true);
        }
        self.cameras((!world).then_some(b.id), path, depth);
        Ok(())
    }

    fn cameras(&mut self, body: Option<StableId>, path: &str, depth: usize) {
        for c in self.scene.cameras.iter().filter(|c| c.body == body) {
            let mut el = El::new("camera");
            el = self.named(el, "camera", path, c.id, &c.name);
            let deg = c.fovy / DEG_TO_RAD;
            let deg = (0..3)
                .map(|k| nudge(deg, k))
                .find(|d| same(&[d * DEG_TO_RAD], &[c.fovy]))
                .unwrap_or(deg);
            el = el.pose(c.pose).f("fovy", &[deg], &[45.0]);
            self.line(depth, el, true);
        }
    }

    fn geom(&mut self, g: &Geom, path: &str) -> Result<El, WriteError> {
        if g.visual_only != (g.contype == 0 && g.conaffinity == 0) {
            return Err(no(format!(
                "geom `{}`: visual_only {} with contype {} conaffinity {}",
                g.name, g.visual_only, g.contype, g.conaffinity
            )));
        }
        let mut el = El::new("geom");
        el = self.named(el, "geom", path, g.id, &g.name);
        el = match g.shape {
            Shape::Plane {
                half_x,
                half_y,
                grid,
            } => el
                .s("type", "plane")
                .s("size", &nums(&[half_x, half_y, grid])),
            Shape::Sphere { radius } => el.s("type", "sphere").s("size", &num(radius)),
            Shape::Capsule {
                radius,
                half_length,
            } => el
                .s("type", "capsule")
                .s("size", &nums(&[radius, half_length])),
            Shape::Cylinder {
                radius,
                half_length,
            } => el
                .s("type", "cylinder")
                .s("size", &nums(&[radius, half_length])),
            Shape::Box { half_extents: v } => {
                el.s("type", "box").s("size", &nums(&[v.x, v.y, v.z]))
            }
            Shape::Ellipsoid { radii: v } => {
                el.s("type", "ellipsoid").s("size", &nums(&[v.x, v.y, v.z]))
            }
            Shape::Mesh { asset } => el.s("type", "mesh").s("mesh", self.r(asset)?),
            Shape::HeightField { asset } => el.s("type", "hfield").s("hfield", self.r(asset)?),
        };
        el = el
            .pose(g.pose)
            .f("friction", &g.friction, &[1.0, 0.005, 0.0001])
            .i("contype", g.contype.into(), 1)
            .i("conaffinity", g.conaffinity.into(), 1)
            .i("condim", g.condim.into(), 3)
            .i("priority", g.priority.into(), 0)
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
        if s.contact_pairs.is_empty() && s.contact_excludes.is_empty() {
            return Ok(());
        }
        self.line(1, El::new("contact"), false);
        for p in &s.contact_pairs {
            let mut el = El::new("pair")
                .s("geom1", self.r(p.geom1)?)
                .s("geom2", self.r(p.geom2)?);
            if let Some(c) = p.condim {
                el = el.s("condim", &c.to_string());
            }
            for (k, v) in [
                ("friction", p.friction.map(|v| v.to_vec())),
                ("solref", p.solref.map(|v| v.to_vec())),
                ("solimp", p.solimp.map(|v| v.to_vec())),
                ("margin", p.margin.map(|v| vec![v])),
                ("gap", p.gap.map(|v| vec![v])),
            ] {
                if let Some(v) = v {
                    el = el.s(k, &nums(&v));
                }
            }
            self.line(2, el, true);
        }
        for (a, b) in &s.contact_excludes {
            let el = El::new("exclude")
                .s("body1", self.r(*a)?)
                .s("body2", self.r(*b)?);
            self.line(2, el, true);
        }
        self.end(1, "contact");
        Ok(())
    }

    fn tendons(&mut self) -> Result<(), WriteError> {
        if self.scene.tendons.is_empty() {
            return Ok(());
        }
        self.line(1, El::new("tendon"), false);
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
                        let el =
                            El::new("joint")
                                .s("joint", self.r(*j)?)
                                .f("coef", &[*coef], &[1.0]);
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
        self.end(1, "tendon");
        Ok(())
    }

    fn actuators(&mut self) -> Result<(), WriteError> {
        if self.scene.actuators.is_empty() {
            return Ok(());
        }
        self.line(1, El::new("actuator"), false);
        for a in &self.scene.actuators {
            let el =
                match a.kind {
                    ActuatorKind::Motor => El::new("motor"),
                    ActuatorKind::Position { kp, kv } => El::new("position")
                        .f("kp", &[kp], &[1.0])
                        .f("kv", &[kv], &[0.0]),
                    ActuatorKind::Velocity { kv } => El::new("velocity").f("kv", &[kv], &[1.0]),
                    ActuatorKind::General { gain, bias } => {
                        let mut el = El::new("general")
                            .f("gainprm", &gain, &[1.0, 0.0, 0.0])
                            .f("biasprm", &bias, &[0.0; 3]);
                        // `MuJoCo` reads the affine terms only under these types.
                        if !same(&gain[1..], &[0.0; 2]) {
                            el = el.s("gaintype", "affine");
                        }
                        if !same(&bias, &[0.0; 3]) {
                            el = el.s("biastype", "affine");
                        }
                        el
                    }
                };
            let (attr, id) = match a.target {
                ActuatorTarget::Joint(id) => ("joint", id),
                ActuatorTarget::Tendon(id) => ("tendon", id),
                ActuatorTarget::Site(id) => ("site", id),
            };
            let el = el
                .s("name", &a.name)
                .s(attr, self.r(id)?)
                .f("gear", &a.gear, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .range("ctrlrange", a.ctrl_range)
                .range("forcerange", a.force_range);
            self.line(2, el, true);
        }
        self.end(1, "actuator");
        Ok(())
    }

    fn sensors(&mut self) -> Result<(), WriteError> {
        if self.scene.sensors.is_empty() {
            return Ok(());
        }
        self.line(1, El::new("sensor"), false);
        for s in &self.scene.sensors {
            let (kind, id) = sensor_target(s.target);
            let name = self.r(id)?;
            let el = match (s.kind, kind) {
                (SensorKind::JointPos, "joint") => El::new("jointpos").s("joint", name),
                (SensorKind::JointVel, "joint") => El::new("jointvel").s("joint", name),
                (SensorKind::ActuatorFrc, "actuator") => El::new("actuatorfrc").s("actuator", name),
                (
                    SensorKind::FramePos | SensorKind::FrameQuat,
                    "body" | "site" | "geom" | "camera",
                ) => {
                    let tag = if s.kind == SensorKind::FramePos {
                        "framepos"
                    } else {
                        "framequat"
                    };
                    El::new(tag).s("objtype", kind).s("objname", name)
                }
                (SensorKind::Camera, "camera") => El::new("camprojection").s("camera", name),
                (
                    SensorKind::Accelerometer
                    | SensorKind::Gyro
                    | SensorKind::Force
                    | SensorKind::Torque
                    | SensorKind::Touch
                    | SensorKind::RangeFinder,
                    "site",
                ) => El::new(match s.kind {
                    SensorKind::Accelerometer => "accelerometer",
                    SensorKind::Gyro => "gyro",
                    SensorKind::Force => "force",
                    SensorKind::Torque => "torque",
                    SensorKind::Touch => "touch",
                    _ => "rangefinder",
                })
                .s("site", name),
                _ => {
                    return Err(no(format!(
                        "sensor `{}`: a {:?} sensor on a {kind}",
                        s.name, s.kind
                    )))
                }
            };
            let el = el.s("name", &s.name).f("noise", &[s.noise], &[0.0]).f(
                "cutoff",
                &[s.cutoff],
                &[0.0],
            );
            self.line(2, el, true);
        }
        self.end(1, "sensor");
        Ok(())
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
    let tris = indices.len() / 3;
    let mut out = vec![0u8; 80];
    out.extend_from_slice(&u32::try_from(tris).unwrap_or(u32::MAX).to_le_bytes());
    for tri in indices.chunks_exact(3) {
        out.extend_from_slice(&[0u8; 12]);
        for &i in tri {
            for c in positions[i as usize] {
                out.extend_from_slice(&c.to_le_bytes());
            }
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
            .iter()
            .map(|d| f32::from_bits(v0.to_bits().wrapping_add(*d)))
            .find(|v| (1.0 - v).to_bits() == t[1].to_bits())
            .unwrap_or(v0);
        let _ = writeln!(out, "vt {:?} {v:?}", t[0]);
    }
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0] + 1, tri[1] + 1, tri[2] + 1];
        let _ = if uvs.is_some() {
            writeln!(out, "f {a}/{a} {b}/{b} {c}/{c}")
        } else {
            writeln!(out, "f {a} {b} {c}")
        };
    }
    out.into_bytes()
}

/// The single image a texture's `file` was cut from: a 2D texture as it is, a cube's
/// `gridsize` x `gridlayout` sheet (one face for a 1 x 1 grid; undeclared cells black).
fn texture_image(data: &TextureData, grid: [u32; 2], layout: &str) -> (u32, u32, Vec<u8>) {
    if data.kind == TexKind::TwoD {
        return (data.width, data.height, data.rgb.clone());
    }
    let w = data.width;
    let face = (w * w * 3) as usize;
    let [rows, cols] = grid;
    if rows * cols == 1 {
        return (w, w, data.rgb[..face].to_vec());
    }
    let (width, height) = (cols * w, rows * w);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for (k, symbol) in layout.chars().enumerate() {
        let Some(f) = "RLUDFB".find(symbol) else {
            continue;
        };
        let (r0, c0) = (w * (k as u32 / cols), w * (k as u32 % cols));
        for j in 0..w {
            let dst = (((j + r0) * width + c0) * 3) as usize;
            let src = f * face + (j * w * 3) as usize;
            rgb[dst..dst + (w * 3) as usize]
                .copy_from_slice(&data.rgb[src..src + (w * 3) as usize]);
        }
    }
    (width, height, rgb)
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
