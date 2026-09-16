//! The Replay panel's view-model: one `.estraj` played back on the CPU (spec 23.3, packet
//! M7/E2).
//!
//! Spec 23.3 asks for an editor that "locally replicates the scene and receives only the
//! pose/joints". A recorded trajectory is exactly that stream, offline: `es_env::Trajectory`
//! holds every body's world pose per tick, and `TriScene::from_scene_with_poses` re-poses the
//! scene from it with no physics backend, no GPU and no process anywhere. This projects those
//! triangles to the screen, sorts them back to front and hands `app.rs` a flat list.
//!
//! `ponytail:` painter's algorithm. Sorting whole triangles by depth is exact for the scene's
//! convex primitives while they do not interpenetrate, and wrong exactly where they do (a
//! gripper closed around a cube can show the wrong face). The upgrade path is
//! `es_render::cpu::rasterize` per pixel at a low resolution, which R1's BVH is what makes
//! affordable; the camera this file builds is already an `es_render::CameraView`, so that
//! swap is a change to one function.

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

/// One triangle ready to draw: screen-space points, one flat colour, and the depth it was
/// sorted by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri2d {
    /// Pixel coordinates, origin top-left (spec 3.1). Off-screen values are kept: clipping
    /// sideways is the painter's job, and `egui` scissors the canvas anyway.
    pub p: [[f32; 2]; 3],
    /// sRGB, the `Rgb8` the renderer would write for this surface.
    pub color: [u8; 3],
    /// Camera-space depth of the centroid, metres.
    pub depth: f32,
    /// Index into the tick's `TriScene::tris`, so the emitted order is itself checkable.
    pub index: u32,
}

/// What one tick looks like from one camera: back to front, so painting in order is correct.
pub type Projected = Vec<Tri2d>;

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
