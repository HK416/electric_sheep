//! The Replay panel's view-model: one `.estraj` played back on the CPU (spec 23.3, packet
//! M7/E2).
//!
//! Spec 23.3 asks for an editor that "locally replicates the scene and receives only the
//! pose/joints". A recorded trajectory is exactly that stream, offline: `es_env::Trajectory`
//! holds every body's world pose per tick, and `TriScene::from_scene_with_poses` re-poses the
//! scene from it with no physics backend, no GPU and no process anywhere. This projects those
//! triangles to the screen and [`Raster::draw`] rasterises them with a per-pixel depth test,
//! which `app.rs` uploads as one texture.
//!
//! The depth buffer is packet M7/E8. E2 sorted whole triangles by centroid depth and painted
//! them in order, which is exact for convex primitives that do not interpenetrate -- and the
//! SO-101 is links that interpenetrate at every joint, so a base plate drew over the shoulder
//! and a finger through the wrist. The sort stays, because it is what fixes the paint order
//! and therefore the tie-break, but it no longer decides what is visible.

use std::path::Path;

use es_assets::scene::SceneDesc;
use es_env::traj::Trajectory;
use es_math::{Pose, Quat, Vec3};
use es_render::cpu::{quat_rotate_inv, srgb_encode};
use es_render::view::ViewParams;
use es_render::{CameraView, ImageSpec, RenderConfig, TileAtlasCfg, Tri, TriScene};

/// Everything that stops a replay from opening, each naming its file.
#[derive(Debug)]
pub enum ReplayError {
    Scene {
        path: String,
        message: String,
    },
    Trajectory {
        path: String,
        message: String,
    },
    /// A camera whose direction has no roll-free orientation (see [`Camera::view`]).
    Camera(String),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scene { path, message } | Self::Trajectory { path, message } => {
                write!(f, "{path}: {message}")
            }
            Self::Camera(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ReplayError {}

/// The free camera of `es video showcase`: in no scene and in no document (spec 15.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub eye: [f64; 3],
    pub look_at: [f64; 3],
    /// Vertical field of view, radians.
    pub fov_y: f64,
    pub width: u32,
    pub height: u32,
}

/// How close to the pole [`Camera::orbit`] lets the pitch get. At the pole the view direction
/// is parallel to world up and [`Camera::view`] has no roll-free answer, so the orbit stops
/// just short of it rather than failing there.
const PITCH_LIMIT: f64 = 1.552; // ~88.9 degrees

/// What the Replay panel takes of the Run tab once a replay is loaded: this fraction, never
/// less than a canvas worth looking at, and never so much that the table above it is gone.
const PANEL_FRACTION: f32 = 0.45;
const PANEL_MIN: f32 = 320.0;
const PANEL_CEILING: f32 = 0.8;

/// The height the Replay panel should ask for inside a tab `available` points high, or `None`
/// for "as tall as its own control rows".
///
/// `None` is the panel that has opened nothing: an empty canvas taking half the window is half
/// the window wasted, and it was clipping the filmstrip above it. The question is answered
/// from the model — is there a loaded [`ReplayView`] — and not from what is on screen
/// (spec 28.10 rule 3).
pub fn panel_height(replay: Option<&ReplayView>, available: f32) -> Option<f32> {
    if !replay.is_some_and(ReplayView::is_loaded) {
        return None;
    }
    // `max` then `min`, never `clamp`: on a window too short for the minimum the ceiling wins,
    // and `clamp` with a lower bound above its upper bound panics.
    Some(
        (available * PANEL_FRACTION)
            .max(PANEL_MIN)
            .min(available * PANEL_CEILING),
    )
}

impl Camera {
    /// `T_world_camera` and the pinhole `ImageSpec`, in the `OpenCV` frame of spec 3.1.
    ///
    /// The arithmetic of `es_env::render::look_at`, which lives behind that crate's `render`
    /// feature and therefore behind Vulkan; the editor links no Vulkan (packet M7/E2), so it
    /// is repeated here over the same `es-render` types and pinned against
    /// `es_render::cpu::rasterize` by `flat_shade_matches_the_renderer`.
    pub fn view(&self) -> Result<CameraView, ReplayError> {
        let (eye, target) = (vec3(self.eye), vec3(self.look_at));
        let forward = (target - eye).normalize();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let right = forward.cross(up);
        if !forward.norm().is_finite() || right.norm() < 1e-9 {
            return Err(ReplayError::Camera(format!(
                "camera at {:?} looking at {:?}: the view direction is degenerate or parallel \
                 to world up (+Z), so there is no roll-free orientation",
                self.eye, self.look_at
            )));
        }
        let right = right.normalize();
        let down = forward.cross(right);
        Ok(CameraView {
            pose: Pose::new(eye, quat_from_basis(right, down, forward)),
            spec: ImageSpec::pinhole(self.width, self.height, self.fov_y),
        })
    }

    /// Moves the eye on the sphere about `look_at`. Pure: the caller keeps the result or does
    /// not.
    ///
    /// `f64::atan2`/`asin`/`sin`/`cos` rather than `es_math::approx` (spec 3.4): a camera the
    /// user drags is UI state, not a kernel, and no golden is a function of it -- the sort-order
    /// golden is taken at a camera built from literal coordinates.
    #[must_use]
    pub fn orbit(mut self, delta_yaw: f64, delta_pitch: f64) -> Self {
        let d = vec3(self.eye) - vec3(self.look_at);
        let r = d.norm();
        if r < 1e-9 {
            return self;
        }
        let yaw = d.y.atan2(d.x) + delta_yaw;
        let pitch = ((d.z / r).asin() + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.eye = [
            self.look_at[0] + r * pitch.cos() * yaw.cos(),
            self.look_at[1] + r * pitch.cos() * yaw.sin(),
            self.look_at[2] + r * pitch.sin(),
        ];
        self
    }

    /// Moves the eye towards (`factor < 1`) or away from (`factor > 1`) `look_at`. Pure.
    #[must_use]
    pub fn zoom(mut self, factor: f64) -> Self {
        let d = vec3(self.eye) - vec3(self.look_at);
        let r = (d.norm() * factor).clamp(1e-3, 1e4);
        let d = d.normalize().scale(r);
        self.eye = [
            self.look_at[0] + d.x,
            self.look_at[1] + d.y,
            self.look_at[2] + d.z,
        ];
        self
    }
}

/// One triangle ready to draw: screen-space points, the camera-space depth of each of them,
/// one flat colour, and the centroid depth it was sorted by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri2d {
    /// Pixel coordinates, origin top-left (spec 3.1). Off-screen values are kept: clipping
    /// sideways is the painter's job, and `egui` scissors the canvas anyway.
    pub p: [[f32; 2]; 3],
    /// Camera-space depth of each vertex, metres, in `p`'s order: what [`Raster::draw`]
    /// interpolates per pixel.
    pub z: [f32; 3],
    /// sRGB, the `Rgb8` the renderer would write for this surface.
    pub color: [u8; 3],
    /// Camera-space depth of the centroid, metres. The paint order's key, not the depth test's.
    pub depth: f32,
    /// Index into the tick's `TriScene::tris`, so the emitted order is itself checkable.
    pub index: u32,
}

/// What one tick looks like from one camera: back to front, so painting in order is correct.
pub type Projected = Vec<Tri2d>;

/// The canvas behind the scene, as a grey level. The raster carries its own background, so
/// the panel is one texture and not a texture over a rectangle.
pub const BACKGROUND: u8 = 18;

/// The largest raster the panel draws, whatever its size in points. Above this a frame costs
/// more than it shows: the picture is scaled to fill the panel either way.
const MAX_WIDTH: u32 = 960;
const MAX_HEIGHT: u32 = 540;

/// One drawn frame: `Rgb8` pixels and the camera-space depth each of them came from.
///
/// A pure function of ([`Projected`], `w`, `h`) -- triangles in the order they arrive, pixels
/// row-major, `f32` throughout, no threads and no atomics -- so the same tick from the same
/// camera is the same bytes, which is what makes a golden of it worth having (spec 1.4).
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub w: u32,
    pub h: u32,
    /// Row-major, tightly packed, `w * h * 3` bytes.
    pub rgb: Vec<u8>,
    /// Row-major, `w * h`; `f32::INFINITY` where nothing was drawn.
    pub depth: Vec<f32>,
}

impl Raster {
    /// The raster size for a panel `points` big: its own resolution, scaled down *uniformly*
    /// to fit [`MAX_WIDTH`] x [`MAX_HEIGHT`].
    ///
    /// Uniformly, because the picture is stretched back over the whole panel: capping the two
    /// axes independently would squash a wide panel's view of the arm. A degenerate or
    /// non-finite size is one pixel rather than zero -- `egui` hands out a collapsed panel
    /// while the window is being dragged, and a zero-sized texture is a device error.
    #[must_use]
    pub fn size_for(points: [f32; 2]) -> (u32, u32) {
        // `max(1.0)` first: it also turns a NaN into 1.0, since NaN loses every comparison.
        let (w, h) = (points[0].max(1.0), points[1].max(1.0));
        let scale = (MAX_WIDTH as f32 / w).min(MAX_HEIGHT as f32 / h).min(1.0);
        (
            ((w * scale) as u32).clamp(1, MAX_WIDTH),
            ((h * scale) as u32).clamp(1, MAX_HEIGHT),
        )
    }

    /// Draws `projected` into a `w` x `h` frame with a per-pixel depth test.
    #[must_use]
    pub fn draw(projected: &Projected, w: u32, h: u32) -> Self {
        let (w, h) = (w.max(1), h.max(1));
        let pixels = w as usize * h as usize;
        let mut raster = Self {
            w,
            h,
            rgb: vec![BACKGROUND; pixels * 3],
            depth: vec![f32::INFINITY; pixels],
        };
        for tri in projected {
            raster.triangle(tri);
        }
        raster
    }

    /// One triangle, by edge functions over its bounding box. Both faces are drawn: the
    /// scene's tessellation has no guaranteed winding, and dividing the three edge functions
    /// by the signed area flips all three signs with it, so "inside" is one test either way.
    fn triangle(&mut self, tri: &Tri2d) {
        let [a, b, c] = tri.p;
        let area = edge(a, b, c);
        if area == 0.0 || !area.is_finite() {
            return;
        }
        let inv_area = 1.0 / area;
        // Screen space is linear in 1/z, not in z: a triangle seen at a grazing angle is
        // metres wrong if the depths themselves are interpolated.
        let inv_z = tri.z.map(|z| 1.0 / z);
        let xs = span(a[0].min(b[0]).min(c[0]), a[0].max(b[0]).max(c[0]), self.w);
        let ys = span(a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]), self.h);
        for py in ys.clone() {
            for px in xs.clone() {
                // The pixel centre: the point `es_render::cpu::primary_dir` casts through.
                let centre = [px as f32 + 0.5, py as f32 + 0.5];
                let bary = [
                    edge(b, c, centre) * inv_area,
                    edge(c, a, centre) * inv_area,
                    edge(a, b, centre) * inv_area,
                ];
                if bary[0] < 0.0 || bary[1] < 0.0 || bary[2] < 0.0 {
                    continue;
                }
                let depth = 1.0 / (bary[0] * inv_z[0] + bary[1] * inv_z[1] + bary[2] * inv_z[2]);
                let at = py as usize * self.w as usize + px as usize;
                // Strictly nearer, so a shared edge -- drawn twice, since the inside test
                // keeps both sides of it -- is owned by whichever triangle came first.
                if depth >= self.depth[at] {
                    continue;
                }
                self.depth[at] = depth;
                self.rgb[at * 3..at * 3 + 3].copy_from_slice(&tri.color);
            }
        }
    }
}

/// Twice the signed area of `(a, b, p)`: zero on the line `a -> b`, and of one sign on each
/// side of it.
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// The pixel range `lo..=hi` touches, clipped to `0..n`. `as u32` saturates and sends NaN to
/// zero, so an off-screen or degenerate span is an empty range and never an index panic.
fn span(lo: f32, hi: f32, n: u32) -> std::ops::Range<u32> {
    let first = (lo.floor().max(0.0) as u32).min(n);
    let last = (hi.ceil().max(0.0) as u32).min(n);
    first..last
}

/// An opened scene + trajectory, and where playback is in it.
#[derive(Debug)]
pub struct ReplayView {
    scene: SceneDesc,
    traj: Trajectory,
    pub playing: bool,
    pub tick: usize,
    /// Playback rate multiplier; 1.0 is real time at the run's control rate.
    pub speed: f64,
    /// Ticks owed but not yet whole, so 0.5x does not round to a standstill.
    pending: f64,
}

impl ReplayView {
    /// Loads the scene the way `es video showcase` does (MJCF or URDF by extension) and the
    /// `.estraj` beside it. The scene is tessellated once here so an unsupported geom is an
    /// error at open rather than a blank canvas later.
    pub fn open(scene_path: &Path, traj_path: &Path) -> Result<Self, ReplayError> {
        let scene = load_scene(scene_path)?;
        let traj = Trajectory::read(traj_path).map_err(|e| ReplayError::Trajectory {
            path: traj_path.display().to_string(),
            message: e.to_string(),
        })?;
        let view = Self {
            scene,
            traj,
            playing: false,
            tick: 0,
            speed: 1.0,
            pending: 0.0,
        };
        view.scene_at(0)?;
        Ok(view)
    }

    pub fn ticks(&self) -> usize {
        self.traj.ticks()
    }

    /// Whether there is anything to play. What the panel's height turns on
    /// ([`panel_height`]): a trajectory with no ticks is not something to grow a canvas for.
    pub fn is_loaded(&self) -> bool {
        self.ticks() > 0
    }

    /// The joint state at `tick`, for the labels. Empty past the last tick.
    pub fn qpos(&self, tick: usize) -> &[f64] {
        if tick < self.ticks() {
            self.traj.qpos(tick)
        } else {
            &[]
        }
    }

    /// The world triangles of `tick` -- the same call `es video showcase` renders, so the
    /// replay draws exactly the geometry the showcase does (packet M5/V9).
    pub fn scene_at(&self, tick: usize) -> Result<TriScene, ReplayError> {
        let poses = if tick < self.ticks() {
            self.traj.poses(tick)
        } else {
            std::collections::BTreeMap::new()
        };
        TriScene::from_scene_with_poses(&self.scene, &poses).map_err(|e| ReplayError::Scene {
            path: self.scene.name.clone(),
            message: e.to_string(),
        })
    }

    /// `tick` seen from `camera`, back to front. A pure function of (trajectory, camera):
    /// same inputs, same `Vec`, bit for bit.
    pub fn project(&self, tick: usize, camera: &Camera) -> Projected {
        let (Ok(scene), Ok(view)) = (self.scene_at(tick), camera.view()) else {
            return Vec::new();
        };
        project_scene(&scene, &view)
    }

    /// Moves the scrubber by whole ticks and pauses. Clamped at both ends, so the buttons
    /// cannot leave the trajectory.
    pub fn step(&mut self, delta: i64) {
        self.playing = false;
        self.pending = 0.0;
        let by = delta.unsigned_abs() as usize;
        self.tick = if delta >= 0 {
            self.tick
                .saturating_add(by)
                .min(self.ticks().saturating_sub(1))
        } else {
            self.tick.saturating_sub(by)
        };
    }

    /// Advances playback by one wall-clock delta. `control_rate_hz` is the rate the
    /// trajectory was recorded at, so 1.0x is the speed the robot moved.
    pub fn advance(&mut self, dt_seconds: f64, control_rate_hz: f64) {
        if !self.playing || self.ticks() == 0 {
            return;
        }
        let steps = self.pending + dt_seconds * control_rate_hz * self.speed;
        let whole = steps.floor().max(0.0);
        self.pending = steps - whole;
        self.tick = self
            .tick
            .saturating_add(whole as usize)
            .min(self.ticks() - 1);
    }
}

/// The shading the `Rs` path uses, for its light direction and ambient floor. The atlas is
/// irrelevant here -- nothing in this file rasterizes -- but the constants must be the
/// renderer's own, and this is where they live.
fn shading() -> RenderConfig {
    RenderConfig::rs(TileAtlasCfg::row(1, 1, 1))
}

/// Project, shade, clip and sort. Split out of [`ReplayView::project`] so a test can drive it
/// from a hand-built `TriScene`.
fn project_scene(scene: &TriScene, view: &CameraView) -> Projected {
    let vp = ViewParams::new(view);
    let cfg = shading();
    let mut out: Projected = Vec::with_capacity(scene.tris.len());
    for (index, tri) in scene.tris.iter().enumerate() {
        let cam = tri.v.map(|v| {
            quat_rotate_inv(
                vp.quat,
                [v[0] - vp.pos[0], v[1] - vp.pos[1], v[2] - vp.pos[2]],
            )
        });
        // Near-plane clipping: a triangle with a vertex at or behind the eye has no screen
        // position for it, and dividing by that `z` would project it through the eye onto the
        // far side of the image. Whole triangles only -- one that straddles the plane
        // disappears instead of being split.
        if cam.iter().any(|c| c[2] <= vp.near) {
            continue;
        }
        let p = cam.map(|c| [vp.fx * c[0] / c[2] + vp.cx, vp.fy * c[1] / c[2] + vp.cy]);
        out.push(Tri2d {
            p,
            z: cam.map(|c| c[2]),
            color: flat_colour(tri, vp.pos, &cfg),
            depth: (cam[0][2] + cam[1][2] + cam[2][2]) / 3.0,
            index: index as u32,
        });
    }
    // Painter's algorithm: farthest first. A stable sort on a key that is only the depth
    // leaves ties in triangle order, which is what makes the emitted sequence a pure function
    // of the inputs (and a golden worth having).
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    out
}

/// `es_shade_lambert` on the flat triangle: the formula of `es_render::cpu::shade_lambert`
/// (private there) followed by its `to_u8`, pinned against `rasterize` by a test.
fn flat_colour(tri: &Tri, eye: [f32; 3], cfg: &RenderConfig) -> [u8; 3] {
    // `es_render::cpu::face_forward`: the normal is flipped towards the incoming ray. The ray
    // of a flat triangle is the one from the eye to its centroid, in world space, which is the
    // ray `rasterize` casts through the centroid's pixel.
    let dir = [
        (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0 - eye[0],
        (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0 - eye[1],
        (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0 - eye[2],
    ];
    let n = if dot(tri.n, dir) > 0.0 {
        [-tri.n[0], -tri.n[1], -tri.n[2]]
    } else {
        tri.n
    };
    let light = [
        cfg.light_dir.x as f32,
        cfg.light_dir.y as f32,
        cfg.light_dir.z as f32,
    ];
    let lambert = cfg.ambient + dot(n, light).max(0.0) * (1.0 - cfg.ambient);
    let mut out = [0u8; 3];
    for (c, (albedo, emission)) in out.iter_mut().zip(tri.albedo.iter().zip(&tri.emission)) {
        // Multiply then add, not `mul_add`: `es_render::cpu::shade_lambert` rounds twice and
        // a fused one would round once, which is a different last bit on an emissive geom.
        let lin = albedo * lambert + emission;
        *c = (255.0 * srgb_encode(lin) + 0.5).floor().clamp(0.0, 255.0) as u8;
    }
    out
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// The rotation whose matrix has `x`, `y`, `z` as its columns, by Shepperd's method: pick the
/// largest of the four denominators, so no branch divides by something near zero. The same
/// arithmetic as `es_env::render`'s, which is behind Vulkan (see [`Camera::view`]).
fn quat_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m01, m02) = (x.x, y.x, z.x);
    let (m10, m11, m12) = (x.y, y.y, z.y);
    let (m20, m21, m22) = (x.z, y.z, z.z);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::from_xyzw((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat::from_xyzw(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat::from_xyzw((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat::from_xyzw((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    }
}

/// MJCF or URDF by extension, as `es backend`'s `load_scene` and `es video showcase` do.
fn load_scene(path: &Path) -> Result<SceneDesc, ReplayError> {
    let bad = |message: String| ReplayError::Scene {
        path: path.display().to_string(),
        message,
    };
    let raw = std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?;
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("urdf"))
    {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        let import =
            es_assets::urdf::parse_urdf(&raw, &resolver).map_err(|e| bad(e.to_string()))?;
        return Ok(import.scene);
    }
    Ok(es_assets::parse_mjcf(&raw)
        .map_err(|e| bad(e.to_string()))?
        .scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use es_assets::scene::{JointKind, SceneDesc};
    use es_core::StableId;
    use es_physics_core::backend::{IndexRange, ModelInfo, StateView};
    use es_render::{Channel, TileData};

    fn root() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).to_path_buf()
    }

    fn scene_path() -> PathBuf {
        root().join("tests/fixtures/mjcf/so101_pick_place.xml")
    }

    fn traj_path() -> PathBuf {
        root().join("tests/fixtures/visible-learning/run/traj/nominal-00.estraj")
    }

    fn golden_path() -> PathBuf {
        root().join("tests/golden/editor/replay_tick0_order.json")
    }

    fn replay() -> ReplayView {
        ReplayView::open(&scene_path(), &traj_path()).expect("open the fixture replay")
    }

    /// The camera the showcase renders the demo from (`docs/packets/M5/V9`), in world metres.
    fn showcase_camera() -> Camera {
        Camera {
            eye: [0.55, -0.45, 0.42],
            look_at: [0.12, -0.02, 0.08],
            fov_y: 45.0_f64.to_radians(),
            width: 480,
            height: 320,
        }
    }

    // --- the .estraj fixture generator --------------------------------------------------------

    /// Ticks in the fixture trajectory. Short on purpose: the packet allows 60 and the file is
    /// `ticks x (nq + nv + nbody*7) x 8` bytes.
    const TICKS: usize = 48;

    /// The joint the fixture sweeps, and the arc it sweeps through (inside the MJCF range).
    const MOVING_JOINT: &str = "shoulder_pan";
    const SWEEP: (f64, f64) = (-0.6, 0.6);

    /// Regenerates `tests/fixtures/visible-learning/run/traj/nominal-00.estraj`. Run once,
    /// explicitly; the result is then read-only (spec 1.4).
    ///
    /// No physics backend is involved and none is needed: `es_env::Trajectory` is shaped by a
    /// `ModelInfo` and filled from a `StateView`, both plain `es-physics-core` structs, and
    /// what it stores is the body poses -- which this composes from the scene's own parent
    /// chain with one hinge turning, exactly the way `es_render::scene`'s `world_poses` does.
    #[test]
    #[ignore = "fixture generator; run explicitly"]
    fn generate_fixture_traj() {
        let scene = load_scene(&scene_path()).expect("the demo scene");
        let model = model_of(&scene);
        let mut traj = Trajectory::new(&model);
        for k in 0..TICKS {
            let angle = SWEEP.0 + (SWEEP.1 - SWEEP.0) * k as f64 / (TICKS - 1) as f64;
            let world = world_poses(&scene, angle);
            let mut xpos = vec![0.0; scene.bodies.len() * 3];
            let mut xquat = vec![0.0; scene.bodies.len() * 4];
            for (i, body) in scene.bodies.iter().enumerate() {
                let pose = world[&body.id];
                xpos[i * 3..i * 3 + 3].copy_from_slice(&[
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]);
                xquat[i * 4..i * 4 + 4].copy_from_slice(&[
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]);
            }
            let qpos = qpos_of(&scene, angle);
            let qvel = vec![0.0; model.nv as usize];
            let state = StateView {
                n_envs: 1,
                tick: es_core::PhysTick(k as u64),
                qpos: &qpos,
                qvel: &qvel,
                act: &[],
                sensordata: &[],
                xpos: &xpos,
                xquat: &xquat,
            };
            traj.push(&model, &state, 0).expect("push the tick");
        }
        traj.write(&traj_path()).expect("write the .estraj");
        println!("wrote {} ({TICKS} ticks)", traj_path().display());
    }

    /// Regenerates the sort-order golden. Run after the trajectory generator.
    #[test]
    #[ignore = "golden generator; run explicitly"]
    fn generate_sort_order_golden() {
        let order: Vec<u32> = replay()
            .project(0, &showcase_camera())
            .iter()
            .map(|t| t.index)
            .collect();
        let path = golden_path();
        std::fs::create_dir_all(path.parent().expect("golden dir")).expect("golden dir");
        std::fs::write(&path, serde_json::to_string(&order).expect("json")).expect("golden");
        println!("wrote {} ({} triangles)", path.display(), order.len());
    }

    /// `nq`/`nv` per joint in `MuJoCo`'s own layout, and one body index per scene body.
    fn model_of(scene: &SceneDesc) -> ModelInfo {
        let (mut nq, mut nv) = (0, 0);
        for joint in &scene.joints {
            let (q, v) = dofs(joint.kind);
            nq += q;
            nv += v;
        }
        let mut model = ModelInfo {
            nq,
            nv,
            nbody: scene.bodies.len() as u32,
            ..ModelInfo::default()
        };
        for (i, body) in scene.bodies.iter().enumerate() {
            model.body.insert(
                body.id,
                IndexRange {
                    start: i as u32,
                    len: 1,
                },
            );
        }
        model
    }

    fn dofs(kind: JointKind) -> (u32, u32) {
        match kind {
            JointKind::Free => (7, 6),
            JointKind::Ball => (4, 3),
            JointKind::Hinge | JointKind::Slide => (1, 1),
            JointKind::Fixed => (0, 0),
        }
    }

    /// The home pose with the swept hinge at `angle`: every other hinge at zero and the free
    /// joint at its body's scene pose.
    fn qpos_of(scene: &SceneDesc, angle: f64) -> Vec<f64> {
        let mut out = Vec::new();
        for joint in &scene.joints {
            match joint.kind {
                JointKind::Free => {
                    let pose = scene
                        .bodies
                        .iter()
                        .find(|b| b.id == joint.body)
                        .map_or(Pose::IDENTITY, |b| b.pose);
                    out.extend_from_slice(&[pose.position.x, pose.position.y, pose.position.z]);
                    out.extend_from_slice(&[
                        pose.orientation.w,
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                    ]);
                }
                JointKind::Ball => out.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]),
                JointKind::Hinge | JointKind::Slide => {
                    out.push(if joint.name == MOVING_JOINT {
                        angle
                    } else {
                        0.0
                    });
                }
                JointKind::Fixed => {}
            }
        }
        out
    }

    /// Every body's world pose with the swept hinge at `angle`, resolved through the parent
    /// chain -- `es_render::scene::world_poses` (private there) with one local frame turned.
    fn world_poses(scene: &SceneDesc, angle: f64) -> BTreeMap<StableId, Pose> {
        let by_id: BTreeMap<StableId, usize> = scene
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();
        let local = |i: usize| -> Pose {
            let body = &scene.bodies[i];
            let Some(joint) = scene
                .joints
                .iter()
                .find(|j| j.body == body.id && j.name == MOVING_JOINT)
            else {
                return body.pose;
            };
            // An MJCF hinge turns the child frame about `axis` through `anchor`, both in the
            // child's own frame: `body.pose * T(anchor) * R * T(-anchor)`.
            let (s, c) = ((angle * 0.5).sin(), (angle * 0.5).cos());
            let axis = joint.axis.normalize();
            let rot = Quat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c);
            body.pose
                .compose(Pose::new(joint.anchor, rot))
                .compose(Pose::new(-joint.anchor, Quat::default()))
        };
        let mut world = BTreeMap::new();
        for (i, body) in scene.bodies.iter().enumerate() {
            let mut chain = vec![local(i)];
            let mut cursor = body.parent;
            for _ in 0..scene.bodies.len() {
                let Some(parent) = cursor.and_then(|p| by_id.get(&p).copied()) else {
                    break;
                };
                chain.push(local(parent));
                cursor = scene.bodies[parent].parent;
            }
            world.insert(
                body.id,
                chain
                    .iter()
                    .rev()
                    .fold(Pose::IDENTITY, |acc, p| acc.compose(*p)),
            );
        }
        world
    }

    // --- oracles -------------------------------------------------------------------------------

    /// Oracle 1: the projected order is a function of (trajectory, camera) and nothing else.
    #[test]
    fn projection_is_a_pure_function() {
        let view = replay();
        let camera = showcase_camera();
        let first = view.project(0, &camera);
        assert_eq!(first, view.project(0, &camera), "twice is not the same");
        assert!(first.len() > 100, "{} triangles projected", first.len());

        let order: Vec<u32> = first.iter().map(|t| t.index).collect();
        let golden: Vec<u32> = serde_json::from_str(
            &std::fs::read_to_string(golden_path())
                .expect("the sort-order golden (run generate_sort_order_golden first)"),
        )
        .expect("parse the golden");
        assert_eq!(order, golden, "the emitted triangle order moved");

        // Back to front, and every index emitted once.
        for pair in first.windows(2) {
            assert!(pair[0].depth >= pair[1].depth, "not sorted back to front");
        }
        let mut unique = order.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), order.len(), "a triangle was emitted twice");

        // One degree of orbit is a different picture.
        let moved: Vec<u32> = view
            .project(0, &camera.orbit(1.0_f64.to_radians(), 0.0))
            .iter()
            .map(|t| t.index)
            .collect();
        assert_ne!(moved, order, "one degree changed nothing");
    }

    /// Oracle 2: the replay draws the showcase's geometry, not a second tessellation of it.
    #[test]
    fn a_replay_reposes_what_the_renderer_would() {
        let view = replay();
        let scene = load_scene(&scene_path()).expect("the demo scene");
        let traj = Trajectory::read(&traj_path()).expect("the fixture trajectory");
        assert!(traj.ticks() >= 2 && traj.ticks() <= 60, "{}", traj.ticks());

        for tick in [0, traj.ticks() / 2, traj.ticks() - 1] {
            let want = TriScene::from_scene_with_poses(&scene, &traj.poses(tick))
                .expect("the showcase's own tessellation");
            let got = view.scene_at(tick).expect("the replay's");
            assert_eq!(got.tris.len(), want.tris.len(), "tick {tick}");
            for (a, b) in got.tris.iter().zip(&want.tris) {
                assert_eq!(bits(a.v), bits(b.v), "tick {tick}: a vertex moved");
                assert_eq!(a.albedo.map(f32::to_bits), b.albedo.map(f32::to_bits));
                assert_eq!(a.seg, b.seg);
            }
            // Every projected triangle names one of them, unclipped, bitwise.
            let projected = view.project(tick, &showcase_camera());
            assert!(!projected.is_empty());
            for tri in &projected {
                let named = &want.tris[tri.index as usize];
                assert_eq!(bits(named.v), bits(want.tris[tri.index as usize].v));
            }
        }
        // A tick past the end is the static scene, not a panic.
        assert!(view.scene_at(traj.ticks() + 10).is_ok());
        assert!(view.qpos(traj.ticks() + 10).is_empty());
        assert_eq!(view.qpos(0).len(), 13, "6 hinges + one free joint");
    }

    fn bits(v: [[f32; 3]; 3]) -> [[u32; 3]; 3] {
        v.map(|p| p.map(f32::to_bits))
    }

    /// Oracle 3: the flat colour is the renderer's own, and the projection lands where the
    /// renderer draws it. One triangle, so no occlusion can make either question ambiguous.
    #[test]
    fn flat_shade_matches_the_renderer() {
        let scene = TriScene {
            tris: vec![Tri {
                v: [[-0.2, 0.9, 0.05], [0.25, 1.1, -0.15], [0.05, 1.0, 0.3]],
                n: [0.0, -1.0, 0.0],
                albedo: [0.8, 0.45, 0.2],
                emission: [0.0; 3],
                seg: 1,
            }],
            ..TriScene::default()
        };
        let camera = Camera {
            eye: [0.0, 0.0, 0.0],
            look_at: [0.0, 1.0, 0.0],
            fov_y: 45.0_f64.to_radians(),
            width: 64,
            height: 48,
        };
        let view = camera.view().expect("a camera looking along +Y");
        let cfg = shading();
        let projected = project_scene(&scene, &view);
        assert_eq!(projected.len(), 1, "the triangle is in front of the camera");
        let tri = projected[0];

        let frame = es_render::cpu::rasterize(&scene, &view, &cfg, 0);
        let Some(tile) = frame.tile(Channel::Rgb8) else {
            panic!("the Rs path writes Rgb8")
        };
        let TileData::U8(rgb) = &tile.data else {
            panic!("Rgb8 is u8")
        };
        let cx = (tri.p[0][0] + tri.p[1][0] + tri.p[2][0]) / 3.0;
        let cy = (tri.p[0][1] + tri.p[1][1] + tri.p[2][1]) / 3.0;
        assert!(
            (0.0..64.0).contains(&cx) && (0.0..48.0).contains(&cy),
            "the centroid projected off-image: {cx}, {cy}"
        );
        let i = (cy as usize * 64 + cx as usize) * 3;
        let pixel = [rgb[i], rgb[i + 1], rgb[i + 2]];
        assert_ne!(pixel, [0, 0, 0], "the rasterizer drew background there");
        assert_eq!(
            tri.color, pixel,
            "the replay's flat colour is not the renderer's"
        );
    }

    /// Oracle 4: playback advances by the control rate, scales with speed, and stops at the
    /// end.
    #[test]
    fn playback_advances_by_the_control_rate() {
        let mut view = replay();
        let last = view.ticks() - 1;
        view.playing = true;
        view.advance(0.02, 50.0);
        assert_eq!(view.tick, 1, "one control tick at 50 Hz");
        view.speed = 2.0;
        view.advance(0.02, 50.0);
        assert_eq!(view.tick, 3, "two ticks at 2x");
        // Half speed owes a tick and pays it on the second call, rather than standing still.
        view.speed = 0.5;
        view.advance(0.02, 50.0);
        assert_eq!(view.tick, 3);
        view.advance(0.02, 50.0);
        assert_eq!(view.tick, 4);
        // Paused, nothing moves.
        view.playing = false;
        view.advance(10.0, 50.0);
        assert_eq!(view.tick, 4);
        // Playing past the end clamps.
        view.playing = true;
        view.speed = 1.0;
        view.advance(10.0, 50.0);
        assert_eq!(view.tick, last);
        view.advance(10.0, 50.0);
        assert_eq!(view.tick, last);
        // Stepping pauses and clamps at both ends.
        view.step(1);
        assert_eq!((view.tick, view.playing), (last, false));
        view.step(-3);
        assert_eq!(view.tick, last - 3);
        view.step(-1000);
        assert_eq!(view.tick, 0);
    }

    /// The panel is its control rows until a replay is loaded, and a canvas afterwards.
    #[test]
    fn the_panel_only_grows_once_a_replay_is_loaded() {
        #[track_caller]
        fn close(got: Option<f32>, want: Option<f32>) {
            match (got, want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-3, "{g} vs {w}"),
                (None, None) => {}
                _ => panic!("{got:?} vs {want:?}"),
            }
        }
        close(panel_height(None, 900.0), None);

        let view = replay();
        assert!(view.is_loaded(), "the fixture has ticks");
        // Nearly half the tab when there is room for it.
        close(panel_height(Some(&view), 900.0), Some(405.0));
        // Never less than a canvas worth looking at...
        close(panel_height(Some(&view), 600.0), Some(320.0));
        // ...and never so much that the table above is gone.
        close(panel_height(Some(&view), 300.0), Some(240.0));
    }

    // --- the rasteriser (packet M7/E8) ---------------------------------------------------------

    /// The golden's camera: the showcase view at the golden's size.
    fn golden_camera() -> Camera {
        Camera {
            width: 320,
            height: 180,
            ..showcase_camera()
        }
    }

    fn raster_golden() -> PathBuf {
        root().join("tests/golden/editor/replay-tick0-320x180")
    }

    fn pixel(raster: &Raster, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * raster.w as usize + x as usize) * 3;
        [raster.rgb[i], raster.rgb[i + 1], raster.rgb[i + 2]]
    }

    /// Whether `p` is inside the screen-space triangle, by the same edge functions the
    /// rasteriser uses -- so "the painter would have covered this pixel" is not a guess.
    fn covers(tri: &Tri2d, p: [f32; 2]) -> bool {
        let [a, b, c] = tri.p;
        let inv = 1.0 / edge(a, b, c);
        edge(b, c, p) * inv >= 0.0 && edge(c, a, p) * inv >= 0.0 && edge(a, b, p) * inv >= 0.0
    }

    /// Regenerates the raster golden. Run once, explicitly; the result is then read-only
    /// (spec 1.4), and `ES_GENERATE_GOLDENS=1` is required on top of `--ignored` so that
    /// `cargo test -- --ignored` cannot rewrite the oracle as a side effect.
    #[test]
    #[ignore = "golden generator; run explicitly"]
    fn generate_raster_golden() {
        if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
            println!("SKIP generate_raster_golden: set ES_GENERATE_GOLDENS=1 to rewrite it");
            return;
        }
        let camera = golden_camera();
        let raster = Raster::draw(&replay().project(0, &camera), camera.width, camera.height);
        let stem = raster_golden();
        std::fs::create_dir_all(stem.parent().expect("golden dir")).expect("golden dir");
        std::fs::write(stem.with_extension("bin"), &raster.rgb).expect("golden bin");
        let meta = serde_json::json!({
            "name": "replay-tick0-320x180",
            "dtype": "u8",
            "layout": "row-major, tightly packed",
            "shape": [raster.h, raster.w, 3],
            "generator": "cargo test -p es-editor -- --ignored generate_raster_golden",
            "oracle": "es_editor::model::replay_view::Raster::draw, the Replay panel itself",
            "source": "tests/fixtures/mjcf/so101_pick_place.xml re-posed by \
                       tests/fixtures/visible-learning/run/traj/nominal-00.estraj, tick 0",
            "pins": "OpenCV camera frame, top-left image origin (spec 3.1); a per-pixel depth \
                     test on interpolated camera-space z, flat Lambert per triangle",
            "spec": "docs/design/editor-shell.md",
            "camera": {
                "eye": camera.eye,
                "look_at": camera.look_at,
                "fov_y": camera.fov_y,
                "width": camera.width,
                "height": camera.height,
            },
        });
        std::fs::write(
            stem.with_extension("json"),
            serde_json::to_string_pretty(&meta).expect("json"),
        )
        .expect("golden json");
        println!("wrote {}.{{bin,json}}", stem.display());
    }

    /// Oracle 1: two triangles that cross -- an X seen edge-on, each nearer on one side. The
    /// painter's algorithm has to draw one of them last and covers the other whole; the depth
    /// test splits the picture down the middle.
    #[test]
    fn a_depth_test_beats_the_painters_sort() {
        // The camera sits at the origin looking along +Y, so world +X is screen right. Both
        // triangles run from the bottom corners up to a shared apex; one is near on the left,
        // the other near on the right, and they cross in the plane x = 0.
        const NEAR: f32 = 1.5;
        const FAR: f32 = 2.5;
        let arm = |near_left: bool, albedo: [f32; 3]| Tri {
            v: [
                [-0.8, if near_left { NEAR } else { FAR }, -0.5],
                [0.8, if near_left { FAR } else { NEAR }, -0.5],
                [0.0, 2.0, 0.5],
            ],
            // The same normal for both, so the two flat colours differ only by albedo.
            n: [0.0, -1.0, 0.0],
            albedo,
            emission: [0.0; 3],
            seg: 1,
        };
        let scene = TriScene {
            tris: vec![arm(true, [0.8, 0.1, 0.1]), arm(false, [0.1, 0.1, 0.8])],
            ..TriScene::default()
        };
        let camera = Camera {
            eye: [0.0, 0.0, 0.0],
            look_at: [0.0, 1.0, 0.0],
            fov_y: 45.0_f64.to_radians(),
            width: 64,
            height: 48,
        };
        let projected = project_scene(&scene, &camera.view().expect("a camera along +Y"));
        assert_eq!(
            projected.len(),
            2,
            "both triangles are in front of the camera"
        );
        let named = |index: u32| {
            *projected
                .iter()
                .find(|t| t.index == index)
                .expect("the triangle survived the clip")
        };
        let (left, right) = (named(0), named(1));
        assert_ne!(left.color, right.color, "the two colours must differ");

        // What the painter's algorithm has to do here: the two centroids are at the same
        // depth, so no order separates them, and whichever is painted last covers both
        // sample pixels whole.
        assert_eq!(
            left.depth.to_bits(),
            right.depth.to_bits(),
            "the centroids are the same depth"
        );
        let samples = [[20.5, 30.5], [44.5, 30.5]];
        let last = *projected.last().expect("two triangles");
        for p in samples {
            assert!(covers(&left, p) && covers(&right, p), "{p:?} is in both");
        }

        let raster = Raster::draw(&projected, camera.width, camera.height);
        let (got_left, got_right) = (pixel(&raster, 20, 30), pixel(&raster, 44, 30));
        assert_eq!(
            got_left, left.color,
            "the left half is not the near-left arm"
        );
        assert_eq!(
            got_right, right.color,
            "the right half is not the near-right arm"
        );
        assert!(
            got_left != last.color || got_right != last.color,
            "the picture is exactly what the painter would have drawn"
        );
        // The depth buffer holds what was interpolated, not the centroid, and the nearer
        // surface won on each side.
        let depth = |x: u32, y: u32| raster.depth[y as usize * raster.w as usize + x as usize];
        assert!(depth(20, 30) < left.depth, "{}", depth(20, 30));
        assert!(depth(44, 30) < right.depth, "{}", depth(44, 30));
        // Background is background: a corner no triangle reaches keeps the canvas colour and
        // an untouched depth.
        assert_eq!(pixel(&raster, 0, 47), [BACKGROUND; 3]);
        assert!(depth(0, 47).is_infinite());
    }

    /// Oracle 2: the fixture at tick 0 from the golden's camera, bitwise.
    #[test]
    fn replay_raster_reproduces_its_golden() {
        let camera = golden_camera();
        let stem = raster_golden();
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(stem.with_extension("json"))
                .expect("the raster golden's json (run generate_raster_golden first)"),
        )
        .expect("parse the golden json");
        // The golden names its own camera, so a drifted constant fails here and not as an
        // unexplained pixel diff.
        assert_eq!(meta["camera"]["fov_y"], serde_json::json!(camera.fov_y));
        assert_eq!(meta["camera"]["eye"], serde_json::json!(camera.eye));
        assert_eq!(meta["camera"]["look_at"], serde_json::json!(camera.look_at));
        assert_eq!(meta["shape"], serde_json::json!([180, 320, 3]));

        let want = std::fs::read(stem.with_extension("bin")).expect("the raster golden");
        let raster = Raster::draw(&replay().project(0, &camera), camera.width, camera.height);
        assert_eq!(
            raster.rgb.len(),
            want.len(),
            "the golden is a different size"
        );
        let differing = raster.rgb.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{differing} of {} bytes moved", want.len());
        // A scene was actually drawn, not a canvas of background.
        let drawn = raster.depth.iter().filter(|d| d.is_finite()).count();
        assert!(drawn > 5_000, "only {drawn} pixels were covered");
    }

    /// Oracle 3: the picture is a pure function of (trajectory, tick, camera, w, h).
    #[test]
    fn replay_raster_is_a_function_of_its_inputs() {
        let view = replay();
        let camera = golden_camera();
        let draw = |tick: usize, camera: &Camera| {
            Raster::draw(&view.project(tick, camera), camera.width, camera.height).rgb
        };
        let first = draw(0, &camera);
        assert_eq!(first, draw(0, &camera), "twice is not the same");
        assert_ne!(first, draw(24, &camera), "a later tick is the same picture");
        assert_ne!(
            first,
            draw(0, &camera.orbit(5.0_f64.to_radians(), 0.0)),
            "five degrees changed nothing"
        );

        // The panel's size rule: the panel's own resolution, scaled down uniformly to fit.
        assert_eq!(Raster::size_for([640.0, 400.0]), (640, 400));
        assert_eq!(Raster::size_for([1920.0, 1080.0]), (960, 540));
        // Wider than the cap's aspect -- the width binds, and the aspect is kept.
        assert_eq!(Raster::size_for([1920.0, 600.0]), (960, 300));
        // Taller -- the height binds.
        assert_eq!(Raster::size_for([1080.0, 1080.0]), (540, 540));
        // A collapsed panel is still one pixel, never zero.
        assert_eq!(Raster::size_for([0.0, 0.0]), (1, 1));
        assert_eq!(Raster::size_for([f32::NAN, 10.0]), (1, 10));
    }

    /// Oracle 4: what a frame costs at the largest raster the panel draws. Printed, not
    /// asserted: the number is this box's, and a threshold in CI would be a flake (spec 12.4).
    #[test]
    fn replay_raster_is_fast_enough() {
        let view = replay();
        let camera = Camera {
            width: 960,
            height: 540,
            ..showcase_camera()
        };
        let projected = view.project(0, &camera);
        let mut ms: Vec<f64> = (0..20)
            .map(|_| {
                let t = std::time::Instant::now();
                let raster = Raster::draw(&projected, camera.width, camera.height);
                let elapsed = t.elapsed().as_secs_f64() * 1e3;
                assert_eq!(raster.rgb.len(), 960 * 540 * 3);
                elapsed
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        println!(
            "replay raster 960x540, {} triangles: median {:.2} ms (min {:.2}, max {:.2})",
            projected.len(),
            ms[ms.len() / 2],
            ms[0],
            ms[ms.len() - 1]
        );
    }

    /// Orbit stays on the sphere and off its poles; zoom only scales the radius.
    #[test]
    fn the_camera_moves_on_a_sphere_about_the_target() {
        let camera = showcase_camera();
        let radius = |c: &Camera| (vec3(c.eye) - vec3(c.look_at)).norm();
        let start = radius(&camera);
        let turned = camera.orbit(0.7, 0.3);
        assert!(
            (radius(&turned) - start).abs() < 1e-9,
            "orbit changed the radius"
        );
        assert_eq!(
            turned.look_at.map(f64::to_bits),
            camera.look_at.map(f64::to_bits)
        );
        assert!(turned.view().is_ok());
        // Straight up is where a roll-free orientation stops existing; the orbit stops short.
        let up = camera.orbit(0.0, 10.0);
        assert!(up.view().is_ok(), "the orbit walked into the pole");
        assert!((radius(&up) - start).abs() < 1e-9);

        let closer = camera.zoom(0.5);
        assert!((radius(&closer) - start * 0.5).abs() < 1e-9);
        assert_eq!(
            closer.look_at.map(f64::to_bits),
            camera.look_at.map(f64::to_bits)
        );

        // A camera looking straight down world up has no roll-free orientation, and says so.
        let degenerate = Camera {
            eye: [0.0, 0.0, 1.0],
            look_at: [0.0, 0.0, 0.0],
            ..camera
        };
        assert!(degenerate.view().is_err());
        assert!(degenerate.view().unwrap_err().to_string().contains("+Z"));
    }
}
