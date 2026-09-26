//! Domain randomization and reset-state distributions (§6.3 `Randomization` / `ResetState`).
//!
//! Target strings are resolved against the scene and the loaded model **once**, at compile
//! time, so the per-reset path has no string work and cannot fail. A target this runtime does
//! not implement is [`EnvError::Unsupported`] naming it — never silently skipped
//! (`docs/design/batch-domains.md` §5).
//!
//! Render targets (packet M11/X5, spec 28.14 rule 4) draw into a per-env [`RenderOverrides`]
//! that the frame source reads at render time ([`RandomizationPlan::apply_render`]); the
//! physics never sees them, and a task that declares none draws the identity.

use std::collections::BTreeMap;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_ir::image::ImageSpec;
use es_ir::task::{Distribution, ObsSource, SensorPath, TaskIr, TaskNode};
use es_math::{approx, Pose, Quat, Vec3};
use es_physics_core::backend::ModelInfo;

use crate::rng::EnvRng;
use crate::EnvError;

/// What one draw writes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// Index into one env's `qpos` row.
    Qpos(u32),
    /// Index into one env's `qvel` row.
    Qvel(u32),
    /// Multiplicative scale on a model parameter: recorded in the episode and pushed into the
    /// backend through `PhysicsBackend::set_params` at reset (packet M11/X4).
    Scale(Param, StableId),
    /// A render draw (packet M11/X5): recorded in the episode, read by the frame source.
    Render(Visual),
}

/// One field of [`RenderOverrides`] a render draw writes. The `usize` is a channel (`r g b`)
/// or an axis (`x y z`, `roll pitch yaw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Visual {
    Intensity,
    Yaw,
    Pitch,
    Color(usize),
    Kelvin,
    Ambient,
    Radiance,
    Sky,
    GeomRgb(StableId, usize),
    CameraOffset(StableId, usize),
    CameraRot(StableId, usize),
    CameraFocal(StableId),
}

/// The scene lighting one episode renders under (§10.2 `light_intensity`, `light_direction`;
/// moved here from `es-eval` by packet M11/X5 so the evaluation's two perturbations and the
/// Task IR's `light.intensity` / `light.direction` draws are one type).
///
/// Two scalars rather than a light model: the `Rs` path shades
/// `albedo * (ambient + n.l * (1 - ambient)) + emission` from **one** directional light
/// (`crates/es-render/src/cpu.rs`), so a light is exactly a gain and a direction.
///
/// On `Rs` the gain is the scene's own colours ([`Self::scene`]): that Lambert term is linear
/// in `albedo`, so scaling every geom's rgba by `k` is *identical* to scaling the incident
/// radiance by `k`. On `Pt` that would double-count (albedo *and* emission), so the render
/// draws scale the emitters, the directional light and the sky instead
/// (`crate::render::drawn_frame`, `docs/design/renderer.md` section 13). The evaluation keeps
/// [`Self::scene`] on both paths, byte for byte as it always ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightOverride {
    /// Multiplier on the light's radiance; `1.0` is the scene as authored.
    pub intensity: f64,
    /// Yaw of the light direction about `+Z`, in degrees; `0.0` is the scene as authored.
    pub yaw_deg: f64,
}

impl Default for LightOverride {
    /// The scene as authored: the identity, so a suite with no light perturbation renders
    /// exactly what every earlier packet rendered.
    fn default() -> Self {
        Self {
            intensity: 1.0,
            yaw_deg: 0.0,
        }
    }
}

impl LightOverride {
    /// Whether this leaves the scene as authored. A caller holding a renderer rebuilds it
    /// only when the override changes, so the nominal cell builds exactly one.
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// `base` with every geom's colour scaled by [`Self::intensity`] — the scene to
    /// tessellate and upload for this episode.
    ///
    /// Alpha is untouched: it is not radiance. A clone rather than an in-place edit, because
    /// the caller's scene is the *authored* one and every episode starts from it.
    pub fn scene(&self, base: &SceneDesc) -> SceneDesc {
        let mut out = base.clone();
        // Exact, not within a margin: this is the "nothing was drawn" path, and a draw that
        // really did land on 1.0 renders the same scene either way.
        if self.intensity.to_bits() == 1.0_f64.to_bits() {
            return out;
        }
        for body in &mut out.bodies {
            for geom in &mut body.geoms {
                for c in &mut geom.rgba[..3] {
                    *c *= self.intensity;
                }
            }
        }
        out
    }

    /// `dir` yawed about `+Z` by [`Self::yaw_deg`], for the renderer's one directional light.
    ///
    /// `es_math::approx`, not `std`: a perturbation draw is an input to the §10.1 table and
    /// two machines must agree on it bit for bit (§3.4).
    pub fn rotate_dir(&self, dir: [f64; 3]) -> [f64; 3] {
        if self.yaw_deg.to_bits() == 0.0_f64.to_bits() {
            return dir;
        }
        let a = (self.yaw_deg as f32).to_radians();
        let (s, c) = (f64::from(approx::sin(a)), f64::from(approx::cos(a)));
        [dir[0] * c - dir[1] * s, dir[0] * s + dir[1] * c, dir[2]]
    }
}

/// One camera's draw: a rigid offset in the camera's own `OpenCV` frame (`+X` right, `+Y`
/// down, `+Z` forward, spec 3.1) and a zoom about the principal point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraDraw {
    /// Translation, metres, along the camera's own `x y z`.
    pub offset: [f64; 3],
    /// Rotation, degrees: roll about `+Z`, pitch about `+X`, yaw about `+Y`.
    pub rot_deg: [f64; 3],
    /// `fx`, `fy` multiplier; `cx`, `cy` stay. `> 1` narrows the field of view.
    pub focal: f64,
}

impl Default for CameraDraw {
    fn default() -> Self {
        Self {
            offset: [0.0; 3],
            rot_deg: [0.0; 3],
            focal: 1.0,
        }
    }
}

impl CameraDraw {
    /// The offset as a pose in the camera frame, composed on the right of the camera's own:
    /// yaw, then pitch, then roll, each about the camera's axis (intrinsic), through
    /// `es_math::approx` so two machines agree on it (§3.4).
    pub fn pose(&self) -> Pose {
        let about = |axis: usize, deg: f64| {
            let h = (deg as f32).to_radians() * 0.5;
            let mut v = [0.0; 3];
            v[axis] = f64::from(approx::sin(h));
            Quat::from_xyzw(v[0], v[1], v[2], f64::from(approx::cos(h)))
        };
        let [roll, pitch, yaw] = self.rot_deg;
        let [x, y, z] = self.offset;
        Pose::new(
            Vec3::new(x, y, z),
            about(1, yaw) * about(0, pitch) * about(2, roll),
        )
    }

    /// A focal length (`fx` or `fy`, in the renderer's `f32`) under this draw's zoom: the one
    /// computation `render::drawn_frame` projects with and [`RenderOverrides::image_spec`]
    /// records, so the episode's intrinsics are the frame's to the bit (`INV-14`).
    pub fn zoom(&self, f: f32) -> f32 {
        f * self.focal as f32
    }

    fn moves_pose(&self) -> bool {
        self.offset.iter().chain(&self.rot_deg).any(|v| *v != 0.0)
    }
}

/// Every render draw of one env's episode (packet M11/X5, spec 28.14 rule 4): a small plain
/// struct the frame source reads per frame and the episode records.
///
/// [`Self::default`] is the scene, lights and cameras as authored, and a frame drawn under it
/// is today's frame byte for byte (`crate::render::drawn_frame` short-circuits on
/// [`Self::is_identity`]). What each field means on each render path is
/// `docs/design/renderer.md` section 13.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderOverrides {
    /// `light.intensity` and `light.direction.yaw`: the evaluation's own type.
    pub light: LightOverride,
    /// `light.direction.pitch`, degrees: raises the light towards `+Z` about the horizontal
    /// axis perpendicular to it.
    pub pitch_deg: f64,
    /// `light.color` (per-channel scale) or `light.color.kelvin` (a fixed table).
    pub color: [f64; 3],
    /// `light.ambient`: scale on the ambient term (`Rs` `ambient`, `Pt` sky).
    pub ambient: f64,
    /// `light.radiance`: the `Pt` directional light's radiance, white. `None` keeps the
    /// sensor's, which is zero — today's `Pt` sensor has no directional light.
    pub radiance: Option<f64>,
    /// `light.sky`: the `Pt` sky's radiance, white. `None` keeps the sensor's (zero).
    pub sky: Option<f64>,
    /// `geom.<name>.rgba`: per-channel RGB scale by geom; alpha untouched.
    pub geoms: BTreeMap<StableId, [f64; 3]>,
    /// `camera.<name>.pose.*` and `camera.<name>.fov`, by camera.
    pub cameras: BTreeMap<StableId, CameraDraw>,
}

impl Default for RenderOverrides {
    fn default() -> Self {
        Self {
            light: LightOverride::default(),
            pitch_deg: 0.0,
            color: [1.0; 3],
            ambient: 1.0,
            radiance: None,
            sky: None,
            geoms: BTreeMap::new(),
            cameras: BTreeMap::new(),
        }
    }
}

impl RenderOverrides {
    /// Whether nothing was drawn: the frame source renders the authored scene untouched.
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    /// Per-channel gain on every light source: `intensity * color`.
    pub fn light_gain(&self) -> [f64; 3] {
        self.color.map(|c| self.light.intensity * c)
    }

    /// `base` (unit, towards the light) yawed by [`LightOverride::rotate_dir`], then pitched
    /// by [`Self::pitch_deg`] about the horizontal axis perpendicular to it. A light straight
    /// overhead has no such axis and is left where it is.
    pub fn light_dir(&self, base: [f64; 3]) -> [f64; 3] {
        let d = self.light.rotate_dir(base);
        if self.pitch_deg.to_bits() == 0.0_f64.to_bits() {
            return d;
        }
        let horizontal = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if horizontal == 0.0 {
            return d;
        }
        // Rodrigues about h = (z x d) / |z x d|, which is perpendicular to d, so the term in
        // h (h . d) vanishes: d' = d cos a + (d x h) sin a.
        let axis = [-d[1] / horizontal, d[0] / horizontal];
        let dxh = [
            -d[2] * axis[1],
            d[2] * axis[0],
            d[0] * axis[1] - d[1] * axis[0],
        ];
        let angle = (self.pitch_deg as f32).to_radians();
        let (sin, cos) = (f64::from(approx::sin(angle)), f64::from(approx::cos(angle)));
        [0, 1, 2].map(|i| d[i] * cos + dxh[i] * sin)
    }

    /// The `ImageSpec` this episode's frames of `camera` really have (`INV-14`): `fx`, `fy`
    /// through the drawn zoom ([`CameraDraw::zoom`], in the `f32` the frame is projected in)
    /// about the principal point, and the drawn offset composed onto the extrinsics.
    /// `declared` itself when nothing was drawn for this camera.
    pub fn image_spec(&self, camera: StableId, declared: &ImageSpec) -> ImageSpec {
        let Some(d) = self.cameras.get(&camera) else {
            return *declared;
        };
        let mut out = *declared;
        out.intrinsics.fx = f64::from(d.zoom(declared.intrinsics.fx as f32));
        out.intrinsics.fy = f64::from(d.zoom(declared.intrinsics.fy as f32));
        if d.moves_pose() {
            out.extrinsics = declared.extrinsics.compose(d.pose());
        }
        out
    }

    fn set(&mut self, v: Visual, x: f64) {
        match v {
            Visual::Intensity => self.light.intensity = x.max(0.0),
            Visual::Yaw => self.light.yaw_deg = x,
            Visual::Pitch => self.pitch_deg = x,
            Visual::Color(c) => self.color[c] = x.max(0.0),
            Visual::Kelvin => self.color = kelvin_rgb(x),
            Visual::Ambient => self.ambient = x.max(0.0),
            Visual::Radiance => self.radiance = Some(x.max(0.0)),
            Visual::Sky => self.sky = Some(x.max(0.0)),
            Visual::GeomRgb(id, c) => self.geoms.entry(id).or_insert([1.0; 3])[c] = x.max(0.0),
            Visual::CameraOffset(id, i) => self.cameras.entry(id).or_default().offset[i] = x,
            Visual::CameraRot(id, i) => self.cameras.entry(id).or_default().rot_deg[i] = x,
            // Positive by construction: `check_positive` refused any other support.
            Visual::CameraFocal(id) => self.cameras.entry(id).or_default().focal = x,
        }
    }
}

/// `light.color.kelvin`'s fixed table: Tanner Helland's fit of the blackbody colour, sampled
/// every 1000 K from 2000 K to 10000 K as 8-bit sRGB and used as linear multipliers over
/// 255. Piecewise linear between rows, clamped at the ends. The table *is* the definition:
/// changing a row changes every draw that lands near it.
const KELVIN: [(f64, [f64; 3]); 9] = [
    (2000.0, [255.0, 137.0, 14.0]),
    (3000.0, [255.0, 177.0, 110.0]),
    (4000.0, [255.0, 206.0, 166.0]),
    (5000.0, [255.0, 228.0, 206.0]),
    (6000.0, [255.0, 246.0, 237.0]),
    (7000.0, [243.0, 242.0, 255.0]),
    (8000.0, [221.0, 230.0, 255.0]),
    (9000.0, [210.0, 223.0, 255.0]),
    (10000.0, [202.0, 218.0, 255.0]),
];

pub(crate) fn kelvin_rgb(k: f64) -> [f64; 3] {
    let (lo, hi) = (KELVIN[0], KELVIN[KELVIN.len() - 1]);
    let row = |(_, rgb): (f64, [f64; 3])| rgb.map(|c| c / 255.0);
    if k <= lo.0 || k.is_nan() {
        return row(lo);
    }
    if k >= hi.0 {
        return row(hi);
    }
    let row_at = KELVIN.iter().rposition(|(t, _)| *t <= k).unwrap_or(0);
    let ((t0, below), (t1, above)) = (KELVIN[row_at], KELVIN[row_at + 1]);
    let w = (k - t0) / (t1 - t0);
    [0, 1, 2].map(|c| (below[c] + (above[c] - below[c]) * w) / 255.0)
}

pub use es_physics_core::backend::Param;

/// One resolved node: a target, a distribution and the RNG stream it draws from.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    target: Target,
    dist: Distribution,
    stream: StableId,
}

/// The per-reset scale factors a plan drew, for the episode record.
pub type ParamScales = BTreeMap<(Param, StableId), f64>;

/// Where a reset writes: one env's state row plus the parameter scales it drew.
#[derive(Debug)]
pub struct ResetBuffer<'a> {
    pub qpos: &'a mut [f64],
    pub qvel: &'a mut [f64],
    pub scales: &'a mut ParamScales,
}

/// Every `ResetState` and `Randomization` node of a task, resolved and ordered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RandomizationPlan {
    entries: Vec<Entry>,
}

impl RandomizationPlan {
    /// Resolves every randomization node against `scene` and `model`.
    ///
    /// Nodes are taken in ascending `NodeId` (§6.4), which fixes the draw order for a given
    /// graph; `ResetState` nodes run before `Randomization` nodes so a randomizer can perturb
    /// the state a reset distribution just laid down.
    pub fn compile(task: &TaskIr, scene: &SceneDesc, model: &ModelInfo) -> Result<Self, EnvError> {
        let mut reset = Vec::new();
        let mut random = Vec::new();
        for node in task.graph.nodes.values() {
            let (target, dist, stream, into) = match node {
                TaskNode::ResetState {
                    target,
                    dist,
                    stream,
                } => (target, dist, stream, &mut reset),
                TaskNode::Randomization {
                    target,
                    dist,
                    stream,
                } => (target, dist, stream, &mut random),
                _ => continue,
            };
            check_distribution(dist, target)?;
            for (resolved, sub) in resolve(target, scene, model, has_rs_sensor(task))? {
                if matches!(resolved, Target::Render(Visual::CameraFocal(_))) {
                    check_positive(dist, target)?;
                }
                // A target that spans channels or angles draws each from its own stream,
                // `<stream>.<sub>`; a one-value target keeps the stream as declared, so no
                // draw a committed task already makes moves.
                let stream = if sub.is_empty() {
                    StableId::from_path(stream)
                } else {
                    StableId::from_path(&format!("{stream}.{sub}"))
                };
                into.push(Entry {
                    target: resolved,
                    dist: dist.clone(),
                    stream,
                });
            }
        }
        reset.append(&mut random);
        Ok(Self { entries: reset })
    }

    /// Draws every render entry for one env's episode into `out` (packet M11/X5).
    ///
    /// Keyed exactly as [`Self::apply`] is — `(seed, env, episode, stream)` — and separate
    /// from it, so the physical draws, `ResetBuffer` and `set_params` are untouched by a
    /// render target and the physics never sees one. `out` is not reset first: the caller
    /// hands in [`RenderOverrides::default`] at every reset.
    pub fn apply_render(&self, seed: u64, env: u32, episode: u64, out: &mut RenderOverrides) {
        for entry in &self.entries {
            if let Target::Render(v) = entry.target {
                let mut rng = EnvRng::new(seed, env, episode, entry.stream);
                out.set(v, rng.sample(&entry.dist));
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether any entry scales a model parameter, i.e. whether a reset has to call
    /// `PhysicsBackend::set_params`. A task without one never does (spec 28.14 rule 1).
    pub fn has_scales(&self) -> bool {
        self.entries
            .iter()
            .any(|e| matches!(e.target, Target::Scale(..)))
    }

    /// Draws every entry for one env and writes it into `buf`.
    ///
    /// Each entry gets its own stream keyed by `(seed, env, episode, stream_id)`, so the result
    /// depends on neither the order envs are reset in nor how many envs there are.
    pub fn apply(&self, seed: u64, env: u32, episode: u64, buf: &mut ResetBuffer<'_>) {
        for entry in &self.entries {
            let mut rng = EnvRng::new(seed, env, episode, entry.stream);
            let v = rng.sample(&entry.dist);
            match entry.target {
                Target::Qpos(i) => set(buf.qpos, i, v),
                Target::Qvel(i) => set(buf.qvel, i, v),
                Target::Scale(param, id) => {
                    buf.scales.insert((param, id), v);
                }
                // Drawn by `apply_render`, from the same key: nothing to write here.
                Target::Render(_) => {}
            }
        }
    }
}

fn set(row: &mut [f64], index: u32, value: f64) {
    // The index was bounds-checked against `ModelInfo` at compile time; a short row means the
    // caller passed a buffer that does not match the model, which `Env` prevents.
    if let Some(slot) = row.get_mut(index as usize) {
        *slot = value;
    }
}

fn check_distribution(dist: &Distribution, target: &str) -> Result<(), EnvError> {
    match dist {
        Distribution::LogUniform { lo, hi } if *lo <= 0.0 || *hi <= 0.0 => {
            Err(EnvError::Unsupported(format!(
                "LogUniform on \"{target}\" with a non-positive bound ({lo}, {hi})"
            )))
        }
        Distribution::Choice(vs) if vs.is_empty() => Err(EnvError::Unsupported(format!(
            "empty Choice on \"{target}\""
        ))),
        _ => Ok(()),
    }
}

/// A focal scale must stay positive whatever is drawn: a zero or negative `fx` is not a
/// camera. Only distributions whose whole support is positive are accepted.
fn check_positive(dist: &Distribution, target: &str) -> Result<(), EnvError> {
    let ok = match dist {
        Distribution::Constant(v) => *v > 0.0,
        Distribution::Uniform { lo, hi } => *lo > 0.0 && *hi > 0.0,
        Distribution::LogUniform { .. } => true, // positive bounds, checked above
        Distribution::Choice(vs) => vs.iter().all(|v| *v > 0.0),
        Distribution::Normal { .. } => false,
    };
    if ok {
        Ok(())
    } else {
        Err(EnvError::Unsupported(format!(
            "\"{target}\" draws a focal scale, whose distribution must be positive over its \
             whole support: {dist:?}"
        )))
    }
}

/// Whether the task declares a sensor on the rasterizer, which has no `Pt` light or sky.
fn has_rs_sensor(task: &TaskIr) -> bool {
    task.observation_spec.channels.values().any(|c| {
        matches!(
            c.source,
            ObsSource::Sensor { render, .. } if render.path == SensorPath::Rs
        )
    })
}

/// The render half of the grammar (packet M11/X5): `light.*` alone; `geom.` and `camera.`
/// come through [`resolve`]'s dotted split.
fn resolve_light(
    target: &str,
    rs_sensor: bool,
) -> Option<Result<Vec<(Target, &'static str)>, EnvError>> {
    let one = |v| Some(Ok(vec![(Target::Render(v), "")]));
    let rgb = |f: fn(usize) -> Visual| {
        Some(Ok(["r", "g", "b"]
            .into_iter()
            .enumerate()
            .map(|(c, s)| (Target::Render(f(c)), s))
            .collect()))
    };
    match target {
        "light.intensity" => one(Visual::Intensity),
        "light.direction" => Some(Ok(vec![
            (Target::Render(Visual::Yaw), "yaw"),
            (Target::Render(Visual::Pitch), "pitch"),
        ])),
        "light.direction.yaw" => one(Visual::Yaw),
        "light.direction.pitch" => one(Visual::Pitch),
        "light.color" => rgb(Visual::Color),
        "light.color.kelvin" => one(Visual::Kelvin),
        "light.ambient" => one(Visual::Ambient),
        "light.radiance" | "light.sky" if rs_sensor => Some(Err(EnvError::Unsupported(format!(
            "randomization target \"{target}\": the task declares an Rs sensor, and the \
             rasterizer has no Pt directional light or sky to draw it into"
        )))),
        "light.radiance" => one(Visual::Radiance),
        "light.sky" => one(Visual::Sky),
        _ => None,
    }
}

/// The target grammar. Anything else is `Unsupported`, by name.
///
/// One target can expand into several entries (`light.direction`, `light.color`,
/// `geom.<n>.rgba`); the `&str` is the stream suffix each draws from, empty for a one-value
/// target.
fn resolve(
    target: &str,
    scene: &SceneDesc,
    model: &ModelInfo,
    rs_sensor: bool,
) -> Result<Vec<(Target, &'static str)>, EnvError> {
    let unsupported = || EnvError::Unsupported(format!("randomization target \"{target}\""));
    let one = |t| Ok(vec![(t, "")]);

    if let Some(i) = index_of(target, "qpos") {
        return in_range(i, model.nq)
            .map(Target::Qpos)
            .ok_or_else(unsupported)
            .and_then(one);
    }
    if let Some(i) = index_of(target, "qvel") {
        return in_range(i, model.nv)
            .map(Target::Qvel)
            .ok_or_else(unsupported)
            .and_then(one);
    }
    if let Some(r) = resolve_light(target, rs_sensor) {
        return r;
    }

    let parts: Vec<&str> = target.split('.').collect();
    let camera = |name: &str| {
        scene
            .cameras
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.id)
            .ok_or_else(unsupported)
    };
    match parts[..] {
        ["camera", name, "fov"] => {
            return one(Target::Render(Visual::CameraFocal(camera(name)?)));
        }
        ["camera", name, "pose", axis] => {
            let id = camera(name)?;
            let v = match axis {
                "x" => Visual::CameraOffset(id, 0),
                "y" => Visual::CameraOffset(id, 1),
                "z" => Visual::CameraOffset(id, 2),
                "roll" => Visual::CameraRot(id, 0),
                "pitch" => Visual::CameraRot(id, 1),
                "yaw" => Visual::CameraRot(id, 2),
                _ => return Err(unsupported()),
            };
            return one(Target::Render(v));
        }
        _ => {}
    }
    let [kind, name, field] = parts[..] else {
        return Err(unsupported());
    };
    let geom = || {
        scene
            .bodies
            .iter()
            .flat_map(|b| &b.geoms)
            .find(|g| g.name == name)
            .map(|g| g.id)
            .ok_or_else(unsupported)
    };
    if (kind, field) == ("geom", "rgba") {
        let id = geom()?;
        return Ok(["r", "g", "b"]
            .into_iter()
            .enumerate()
            .map(|(c, s)| (Target::Render(Visual::GeomRgb(id, c)), s))
            .collect());
    }
    let physical = match (kind, field) {
        ("joint", "qpos" | "qvel") => {
            let joint = scene
                .joints
                .iter()
                .find(|j| j.name == name)
                .ok_or_else(unsupported)?;
            let map = if field == "qpos" {
                &model.qpos
            } else {
                &model.dof
            };
            let start = map.get(&joint.id).ok_or_else(unsupported)?.start;
            Ok(if field == "qpos" {
                Target::Qpos(start)
            } else {
                Target::Qvel(start)
            })
        }
        ("body", "mass") => scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .map(|b| Target::Scale(Param::BodyMass, b.id))
            .ok_or_else(unsupported),
        ("geom", "friction") => geom().map(|id| Target::Scale(Param::GeomFriction, id)),
        ("actuator", "gain") => scene
            .actuators
            .iter()
            .find(|a| a.name == name)
            .map(|a| Target::Scale(Param::ActuatorGain, a.id))
            .ok_or_else(unsupported),
        _ => Err(unsupported()),
    };
    physical.and_then(one)
}

/// `"qpos[3]"` -> `Some(3)`.
fn index_of(target: &str, prefix: &str) -> Option<u32> {
    target
        .strip_prefix(prefix)?
        .strip_prefix('[')?
        .strip_suffix(']')?
        .parse()
        .ok()
}

fn in_range(i: u32, n: u32) -> Option<u32> {
    (i < n).then_some(i)
}

// Bitwise reproducibility is the property under test: these comparisons are deliberate.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::env::tests::{fake_model, fake_scene, task_with};
    use es_ir::graph::NodeId;

    fn plan_for(nodes: &[TaskNode]) -> Result<RandomizationPlan, EnvError> {
        RandomizationPlan::compile(&task_with(nodes), &fake_scene(), &fake_model())
    }

    fn randomization(target: &str, dist: Distribution) -> TaskNode {
        TaskNode::Randomization {
            target: target.to_owned(),
            dist,
            stream: format!("s.{target}"),
        }
    }

    fn uniform(lo: f64, hi: f64) -> Distribution {
        Distribution::Uniform { lo, hi }
    }

    #[test]
    fn every_supported_target_resolves() {
        let plan = plan_for(&[
            randomization("qpos[0]", uniform(-0.1, 0.1)),
            randomization("qvel[1]", uniform(-1.0, 1.0)),
            randomization("joint.hinge.qpos", uniform(-0.2, 0.2)),
            randomization("joint.slide.qvel", uniform(-2.0, 2.0)),
            randomization("body.link.mass", uniform(0.9, 1.1)),
            randomization("geom.ball.friction", uniform(0.5, 1.5)),
            randomization("actuator.motor.gain", uniform(0.8, 1.2)),
        ])
        .unwrap();
        assert_eq!(plan.entries.len(), 7);
        assert_eq!(plan.entries[0].target, Target::Qpos(0));
        assert_eq!(plan.entries[1].target, Target::Qvel(1));
        assert!(matches!(
            plan.entries[4].target,
            Target::Scale(Param::BodyMass, _)
        ));
    }

    #[test]
    fn an_unknown_target_is_named_not_skipped() {
        for target in [
            "gravity",
            "qpos[99]",
            "body.nonexistent.mass",
            "body.link.colour",
            "joint.hinge.torque",
        ] {
            let err = plan_for(&[randomization(target, uniform(0.0, 1.0))]).unwrap_err();
            assert!(err.to_string().contains(target), "{target}: {err}");
        }
        let err = plan_for(&[randomization(
            "qpos[0]",
            Distribution::LogUniform { lo: -1.0, hi: 2.0 },
        )])
        .unwrap_err();
        assert!(err.to_string().contains("non-positive"), "{err}");
        let err =
            plan_for(&[randomization("qpos[0]", Distribution::Choice(Vec::new()))]).unwrap_err();
        assert!(err.to_string().contains("empty Choice"), "{err}");
    }

    #[test]
    fn reset_state_runs_before_randomization_whatever_the_node_ids() {
        let task = {
            let mut t = task_with(&[]);
            t.graph.insert(
                NodeId(0),
                randomization("qpos[0]", Distribution::Constant(5.0)),
            );
            t.graph.insert(
                NodeId(1),
                TaskNode::ResetState {
                    target: "qpos[0]".to_owned(),
                    dist: Distribution::Constant(1.0),
                    stream: "reset".to_owned(),
                },
            );
            t
        };
        let plan = RandomizationPlan::compile(&task, &fake_scene(), &fake_model()).unwrap();
        let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
        plan.apply(
            0,
            0,
            0,
            &mut ResetBuffer {
                qpos: &mut qpos,
                qvel: &mut qvel,
                scales: &mut scales,
            },
        );
        assert_eq!(qpos[0], 5.0, "the Randomization node writes last");
    }

    #[test]
    fn draws_depend_on_env_and_episode_but_not_on_reset_order() {
        let plan = plan_for(&[
            randomization("qpos[0]", uniform(-1.0, 1.0)),
            randomization("body.link.mass", uniform(0.5, 1.5)),
        ])
        .unwrap();
        let draw = |env: u32, episode: u64| {
            let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
            plan.apply(
                7,
                env,
                episode,
                &mut ResetBuffer {
                    qpos: &mut qpos,
                    qvel: &mut qvel,
                    scales: &mut scales,
                },
            );
            (qpos[0], *scales.values().next().unwrap())
        };
        assert_eq!(draw(0, 0), draw(0, 0));
        assert_ne!(draw(0, 0), draw(1, 0));
        assert_ne!(draw(0, 0), draw(0, 1));
        let (_, mass) = draw(3, 2);
        assert!((0.5..=1.5).contains(&mass), "{mass}");
    }

    // --- visual randomization (packet M11/X5), oracle 1 --------------------------------------

    use crate::env::tests::{camera_scene, sensor_channel};

    /// Every render target of `docs/design/batch-domains.md` section 5.
    const VISUAL: [&str; 18] = [
        "light.intensity",
        "light.direction",
        "light.direction.yaw",
        "light.direction.pitch",
        "light.color",
        "light.color.kelvin",
        "light.ambient",
        "light.radiance",
        "light.sky",
        "geom.ball.rgba",
        "camera.cam.pose.x",
        "camera.cam.pose.y",
        "camera.cam.pose.z",
        "camera.cam.pose.roll",
        "camera.cam.pose.pitch",
        "camera.cam.pose.yaw",
        "camera.cam.fov",
        "geom.cube.rgba",
    ];

    fn visual_plan(nodes: &[TaskNode]) -> Result<RandomizationPlan, EnvError> {
        RandomizationPlan::compile(&task_with(nodes), &camera_scene(), &fake_model())
    }

    fn visual_draw(plan: &RandomizationPlan, seed: u64, env: u32, ep: u64) -> RenderOverrides {
        let mut out = RenderOverrides::default();
        plan.apply_render(seed, env, ep, &mut out);
        out
    }

    fn physical_draw(plan: &RandomizationPlan, env: u32, ep: u64) -> (Vec<u64>, ParamScales) {
        let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
        plan.apply(
            3,
            env,
            ep,
            &mut ResetBuffer {
                qpos: &mut qpos,
                qvel: &mut qvel,
                scales: &mut scales,
            },
        );
        (
            qpos.iter().chain(&qvel).map(|v| v.to_bits()).collect(),
            scales,
        )
    }

    #[test]
    fn visual_randomization_every_target_parses_and_resolves() {
        for target in VISUAL {
            let dist = if target.ends_with("kelvin") {
                uniform(2500.0, 9000.0)
            } else {
                uniform(0.5, 1.5)
            };
            let plan = visual_plan(&[randomization(target, dist)])
                .unwrap_or_else(|e| panic!("{target}: {e}"));
            assert!(!plan.has_scales(), "{target} is not a physics parameter");
            let ov = visual_draw(&plan, 1, 0, 0);
            assert!(!ov.is_identity(), "{target} drew nothing");
        }
        // The multi-stream targets: two angles, three channels.
        let n = |t: &str| {
            visual_plan(&[randomization(t, uniform(0.5, 1.5))])
                .unwrap()
                .entries
                .len()
        };
        assert_eq!(n("light.direction"), 2);
        assert_eq!(n("light.color"), 3);
        assert_eq!(n("geom.ball.rgba"), 3);
        assert_eq!(n("camera.cam.fov"), 1);
        // Each channel from its own stream: a draw of `light.color` is not grey.
        let ov = visual_draw(
            &visual_plan(&[randomization("light.color", uniform(0.5, 1.5))]).unwrap(),
            1,
            0,
            0,
        );
        assert!(
            ov.color[0] != ov.color[1] && ov.color[1] != ov.color[2],
            "{:?}",
            ov.color
        );
    }

    #[test]
    fn visual_randomization_unknown_targets_are_named_not_skipped() {
        for target in [
            "light",
            "light.flux",
            "light.direction.roll",
            "light.color.hue",
            "geom.nope.rgba",
            "geom.ball.rgb",
            "camera.nope.fov",
            "camera.cam.pose",
            "camera.cam.pose.w",
            "camera.cam.zoom",
            "camera.cam.fov.x",
        ] {
            let err = visual_plan(&[randomization(target, uniform(0.5, 1.5))]).unwrap_err();
            assert!(err.to_string().contains(target), "{target}: {err}");
        }
        // A focal scale must stay positive: an unbounded distribution is refused.
        let err = visual_plan(&[randomization(
            "camera.cam.fov",
            Distribution::Normal {
                mean: 1.0,
                std: 0.1,
            },
        )])
        .unwrap_err();
        assert!(err.to_string().contains("positive"), "{err}");
        let err = visual_plan(&[randomization("camera.cam.fov", uniform(0.0, 1.0))]).unwrap_err();
        assert!(err.to_string().contains("positive"), "{err}");
        // The `Pt` light and sky mean nothing to the rasterizer, and a task with an `Rs` sensor
        // refuses them rather than drawing a value no frame shows (spec 17.2).
        for target in ["light.radiance", "light.sky"] {
            let mut task = task_with(&[randomization(target, uniform(0.5, 1.5))]);
            let scene = camera_scene();
            let (channel, _) =
                sensor_channel(scene.cameras[0].id, es_ir::task::SensorRender::default());
            task.observation_spec
                .channels
                .insert("rgb".to_owned(), channel);
            let err = RandomizationPlan::compile(&task, &scene, &fake_model()).unwrap_err();
            let text = err.to_string();
            assert!(text.contains(target) && text.contains("Rs"), "{text}");
        }
    }

    #[test]
    fn visual_randomization_draws_are_keyed_by_seed_env_episode_stream() {
        let nodes: Vec<TaskNode> = VISUAL
            .iter()
            .filter(|t| !t.ends_with("kelvin"))
            .map(|t| randomization(t, uniform(0.5, 1.5)))
            .collect();
        let plan = visual_plan(&nodes).unwrap();
        assert_eq!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 2, 3));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 2, 4));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 1, 3));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 8, 2, 3));
        // The stream name is part of the key.
        let renamed = |stream: &str| {
            let plan = visual_plan(&[TaskNode::Randomization {
                target: "light.intensity".to_owned(),
                dist: uniform(0.5, 1.5),
                stream: stream.to_owned(),
            }])
            .unwrap();
            visual_draw(&plan, 7, 2, 3).light.intensity
        };
        assert_eq!(renamed("a"), renamed("a"));
        assert_ne!(renamed("a"), renamed("b"));
    }

    #[test]
    fn visual_randomization_undeclared_targets_move_nothing() {
        let physical = [
            randomization("qpos[0]", uniform(-1.0, 1.0)),
            randomization("body.link.mass", uniform(0.5, 1.5)),
        ];
        let plain = visual_plan(&physical).unwrap();
        assert!(visual_draw(&plain, 3, 0, 0).is_identity());
        assert_eq!(visual_draw(&plain, 3, 0, 0), RenderOverrides::default());
        // Render entries beside the physical ones move none of the physical draws.
        let mut both = physical.to_vec();
        both.extend(
            VISUAL
                .iter()
                .filter(|t| !t.ends_with("kelvin"))
                .map(|t| randomization(t, uniform(0.5, 1.5))),
        );
        let both = visual_plan(&both).unwrap();
        for (env, ep) in [(0, 0), (1, 0), (0, 5)] {
            assert_eq!(
                physical_draw(&plain, env, ep),
                physical_draw(&both, env, ep)
            );
        }
        assert_eq!(plain.has_scales(), both.has_scales());
    }

    #[test]
    fn visual_randomization_light_and_camera_arithmetic() {
        let base = [0.3, 0.4, 0.866_025_4];
        let norm = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        // The identity is exact.
        assert_eq!(
            RenderOverrides::default().light_dir(base).map(f64::to_bits),
            base.map(f64::to_bits)
        );
        // Yaw is the evaluation's own kernel: `LightOverride::rotate_dir`.
        let yawed = RenderOverrides {
            light: LightOverride {
                intensity: 1.0,
                yaw_deg: 30.0,
            },
            ..RenderOverrides::default()
        };
        assert_eq!(yawed.light_dir(base), yawed.light.rotate_dir(base));
        // Pitch raises the light towards +Z and keeps it a unit vector and its azimuth.
        let up = RenderOverrides {
            pitch_deg: 20.0,
            ..RenderOverrides::default()
        }
        .light_dir(base);
        assert!(up[2] > base[2], "{up:?}");
        assert!((norm(up) - norm(base)).abs() < 1e-6, "{up:?}");
        assert!((up[1] / up[0] - base[1] / base[0]).abs() < 1e-6, "{up:?}");
        // The colour-temperature table: warm is red-heavy, cool is blue-heavy, clamped at the
        // ends, piecewise linear between its rows.
        let warm = kelvin_rgb(2000.0);
        let cool = kelvin_rgb(10_000.0);
        assert!(warm[0] > warm[2] && cool[2] > cool[0], "{warm:?} {cool:?}");
        assert_eq!(kelvin_rgb(500.0), warm);
        assert_eq!(kelvin_rgb(40_000.0), cool);
        let mid = kelvin_rgb(2500.0);
        let (a, b) = (kelvin_rgb(2000.0), kelvin_rgb(3000.0));
        for c in 0..3 {
            assert!((mid[c] - f64::midpoint(a[c], b[c])).abs() < 1e-12);
        }
        // The camera delta: a translation in the camera's own frame, and a yaw about its +Y.
        let d = CameraDraw {
            offset: [0.1, 0.0, 0.0],
            ..CameraDraw::default()
        };
        assert_eq!(d.pose().position.x, 0.1);
        assert_eq!(d.pose().orientation, es_math::Quat::IDENTITY);
        let turned = CameraDraw {
            rot_deg: [0.0, 0.0, 90.0],
            ..CameraDraw::default()
        }
        .pose()
        .orientation
        .rotate(es_math::Vec3::new(0.0, 0.0, 1.0));
        assert!((turned.x - 1.0).abs() < 1e-6, "{turned:?}");
    }
}
