//! The Replay panel's view-model: one `.estraj` played back on the CPU (spec 23.3, packet
//! M7/E2).
//!
//! Spec 23.3 asks for an editor that "locally replicates the scene and receives only the
//! pose/joints". A recorded trajectory is exactly that stream, offline: `es_env::Trajectory`
//! holds every body's world pose per tick, and `TriScene::from_scene_with_poses` re-poses the
//! scene from it with no physics backend, no GPU and no process anywhere. `es_render::raster`
//! projects those triangles to the screen and rasterises them with a per-pixel depth test,
//! which `app.rs` uploads as one texture.

use std::path::Path;

use es_assets::scene::SceneDesc;
use es_env::traj::Trajectory;
use es_render::raster::{project_scene, Camera, Projected, RasterError};
use es_render::TriScene;

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
    Camera(RasterError),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scene { path, message } | Self::Trajectory { path, message } => {
                write!(f, "{path}: {message}")
            }
            Self::Camera(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ReplayError {}

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

/// MJCF or URDF by extension, as `es backend`'s `load_scene` and `es video showcase` do, then
/// the mesh files the scene names, relative to its directory (packet M10/W2b).
fn load_scene(path: &Path) -> Result<SceneDesc, ReplayError> {
    let bad = |message: String| ReplayError::Scene {
        path: path.display().to_string(),
        message,
    };
    let raw = std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?;
    let mut scene = if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("urdf"))
    {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        es_assets::urdf::parse_urdf(&raw, &resolver)
            .map_err(|e| bad(e.to_string()))?
            .scene
    } else {
        es_assets::parse_mjcf(&raw)
            .map_err(|e| bad(e.to_string()))?
            .scene
    };
    es_assets::mesh::load(&mut scene, path.parent().unwrap_or(Path::new(".")))
        .map_err(|e| bad(e.to_string()))?;
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use es_assets::scene::{JointKind, SceneDesc};
    use es_core::StableId;
    use es_math::{Pose, Quat};
    use es_physics_core::backend::{IndexRange, ModelInfo, StateView};
    use es_render::raster::{Raster, MAX_PIXELS};

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

        // The panel's size rule: its own resolution while it is inside the budget, and
        // scaled down uniformly once it is not.
        assert_eq!(Raster::size_for([640.0, 400.0]), (640, 400));
        assert_eq!(Raster::size_for([1920.0, 1080.0]), (960, 540));
        // The budget is on the area, so a panel of another shape keeps its own aspect and
        // spends all of it -- the Replay panel's own shape is wide and short.
        for panel in [[1920.0, 600.0], [1080.0, 1080.0], [1897.0, 283.0]] {
            let (w, h) = Raster::size_for(panel);
            let (w, h) = (f64::from(w), f64::from(h));
            assert!(w * h <= f64::from(MAX_PIXELS), "{panel:?} -> {w} x {h}");
            assert!(w * h > f64::from(MAX_PIXELS) * 0.99, "{panel:?} is timid");
            let (want, got) = (f64::from(panel[0]) / f64::from(panel[1]), w / h);
            assert!((want - got).abs() < 0.01, "{panel:?} changed shape");
        }
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
}
