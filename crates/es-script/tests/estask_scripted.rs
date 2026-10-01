//! Packet G3a, oracle (3): every relation of the vocabulary on scripted states, through
//! `es_env::Env` — its truth (the termination it raises) and its shaping reward, from states
//! built to satisfy and to violate it (as H2's `the_task_scores_scripted_states`). The states are
//! set directly on a stateless fake backend, so the cones read exactly what the test wrote and
//! no physics runs.

// The rewards are compared against the formulas exactly where the lowering is one operation.
#![allow(clippy::float_cmp)]
// The curve is written in the design note's symbols (`s`, `f`, `h`, `r`, `t`).
#![allow(clippy::many_single_char_names)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use es_assets::scene::{JointKind, SceneDesc};
use es_core::time::{PhysTick, TickRate};
use es_env::{BatchDomains, Env, Termination};
use es_ir::task::TaskIr;
use es_physics_core::backend::{
    IndexRange, LoadConfig, ModelInfo, PhysicsBackend, PhysicsError, StateView, StepReport,
};
use es_physics_core::caps::{BatchSupport, Capabilities, DeterminismTier, FloatPrecision};
use es_script::spec::{compile_task, TaskSpec};

const SCENE: &str = "tests/fixtures/mjcf/so101_pick_place_views.xml";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn scene() -> SceneDesc {
    let xml = std::fs::read_to_string(repo().join(SCENE)).expect("the scene");
    es_assets::parse_mjcf(&xml).expect("parses").scene
}

/// One env whose state is whatever was last written: `step` only moves the clock.
#[derive(Debug)]
struct Still {
    caps: Capabilities,
    model: Option<ModelInfo>,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    xpos: Vec<f64>,
    xquat: Vec<f64>,
    tick: PhysTick,
}

impl Still {
    fn new() -> Self {
        Self {
            caps: Capabilities {
                name: "still".to_owned(),
                determinism: DeterminismTier::Bitwise,
                batch: BatchSupport {
                    max_envs: 1,
                    gpu_resident: false,
                },
                joints: BTreeSet::new(),
                actuators: BTreeSet::new(),
                sensors: BTreeSet::new(),
                contact: BTreeSet::new(),
                float: FloatPrecision::F64,
                supports_reset_subset: true,
                supports_state_get_set: true,
                quirks: Vec::new(),
            },
            model: None,
            qpos: Vec::new(),
            qvel: Vec::new(),
            xpos: Vec::new(),
            xquat: Vec::new(),
            tick: PhysTick::ZERO,
        }
    }
}

impl PhysicsBackend for Still {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// `MuJoCo`'s layout: each joint's `qpos` / dof slice in scene order, one body row per
    /// scene body.
    fn load(&mut self, scene: &SceneDesc, cfg: &LoadConfig) -> Result<ModelInfo, PhysicsError> {
        let mut info = ModelInfo {
            n_envs: cfg.n_envs,
            rate: TickRate::from_period_secs(scene.options.timestep).expect("a timestep"),
            ..ModelInfo::default()
        };
        for j in &scene.joints {
            let (nq, nv) = match j.kind {
                JointKind::Free => (7, 6),
                JointKind::Ball => (4, 3),
                JointKind::Hinge | JointKind::Slide => (1, 1),
                JointKind::Fixed => (0, 0),
            };
            info.qpos.insert(j.id, IndexRange::new(info.nq, nq));
            info.dof.insert(j.id, IndexRange::new(info.nv, nv));
            (info.nq, info.nv) = (info.nq + nq, info.nv + nv);
        }
        for (i, a) in scene.actuators.iter().enumerate() {
            info.actuator.insert(a.id, IndexRange::new(i as u32, 1));
        }
        for (i, b) in scene.bodies.iter().enumerate() {
            info.body.insert(b.id, IndexRange::new(i as u32, 1));
        }
        (info.nu, info.nbody) = (scene.actuators.len() as u32, scene.bodies.len() as u32);
        self.qpos = vec![0.0; info.nq as usize];
        self.qvel = vec![0.0; info.nv as usize];
        self.xpos = vec![0.0; 3 * info.nbody as usize];
        self.xquat = [0.0, 0.0, 0.0, 1.0].repeat(info.nbody as usize);
        self.model = Some(info.clone());
        Ok(info)
    }

    fn model_info(&self) -> Option<&ModelInfo> {
        self.model.as_ref()
    }

    fn reset(
        &mut self,
        _: Option<&[u32]>,
        state: Option<&StateView<'_>>,
    ) -> Result<(), PhysicsError> {
        if let Some(s) = state {
            self.qpos.copy_from_slice(s.qpos);
            self.qvel.copy_from_slice(s.qvel);
        }
        Ok(())
    }

    fn set_ctrl(&mut self, _: &[f64]) -> Result<(), PhysicsError> {
        Ok(())
    }

    fn step(&mut self, n_substeps: u32) -> Result<StepReport, PhysicsError> {
        self.tick = self.tick.add_ticks(u64::from(n_substeps));
        Ok(StepReport {
            tick: self.tick,
            failures: Vec::new(),
        })
    }

    fn state(&self) -> StateView<'_> {
        StateView {
            n_envs: 1,
            tick: self.tick,
            qpos: &self.qpos,
            qvel: &self.qvel,
            xpos: &self.xpos,
            xquat: &self.xquat,
            ..StateView::default()
        }
    }

    fn set_state(&mut self, s: &StateView<'_>) -> Result<(), PhysicsError> {
        self.qpos.copy_from_slice(s.qpos);
        self.qvel.copy_from_slice(s.qvel);
        self.xpos.copy_from_slice(s.xpos);
        self.xquat.copy_from_slice(s.xquat);
        Ok(())
    }
}

/// The SO-101 views scene with `body` (TOML) after the header.
fn task(body: &str) -> TaskIr {
    let text = format!(
        "kind = \"task-spec\"\nschema = 1\nscene = \"{SCENE}\"\nrobot = \"base\"\n\
         control_hz = 50\ntimeout_s = 36.0\n{body}"
    );
    let spec = TaskSpec::from_toml(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    compile_task(&spec, &repo()).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// The state a step is scored on: `qpos`, `qvel`, `xpos`, `xquat` (xyzw).
struct State {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    xpos: Vec<f64>,
    xquat: Vec<f64>,
}

/// One control step entered with the reset's state as `edit` leaves it: the reward and how
/// the episode stands after it.
fn score(task: &TaskIr, edit: impl Fn(&mut State)) -> (f64, Termination) {
    let scene = scene();
    let domains = BatchDomains::single_env_at(
        TickRate::from_period_secs(scene.options.timestep).expect("200 Hz"),
        TickRate::hz(50),
    )
    .expect("four ticks");
    let mut env = Env::new(task, &scene, Still::new(), &domains, 7).expect("the task compiles");
    let mut s = {
        let v = env.backend().state();
        State {
            qpos: v.qpos.to_vec(),
            qvel: v.qvel.to_vec(),
            xpos: v.xpos.to_vec(),
            xquat: v.xquat.to_vec(),
        }
    };
    edit(&mut s);
    let view = StateView {
        n_envs: 1,
        qpos: &s.qpos,
        qvel: &s.qvel,
        xpos: &s.xpos,
        xquat: &s.xquat,
        ..StateView::default()
    };
    env.backend_mut().set_state(&view).expect("set");
    let out = env.step(&[0.0; 6]).expect("one step");
    let how = if out.dones[0] {
        out.episodes[0].termination
    } else {
        Termination::Running
    };
    (out.rewards[0], how)
}

/// The scene row of body `name` in `xpos` / `xquat`.
fn row(name: &str) -> usize {
    scene()
        .bodies
        .iter()
        .position(|b| b.name == name)
        .expect("a body")
}

// `qpos[5]` is the gripper, `qpos[6..13]` the cube's free joint; `qvel[6]` its x velocity.

#[test]
fn inside_holds_within_the_span_and_its_ramp_pays_toward_it() {
    let t = task(
        "[success]\nclauses = [{ subject = \"cube.x\", relation = \"inside\", range = [0.09, 0.19], \
         shaping = \"ramp\", ramp = [0.09, 0.3], weight = -1.0, term = \"t\" }]",
    );
    let ramp = |x: f64| -((0.0 + (x - 0.09) * (1.0 / (0.3 - 0.09))).clamp(0.0, 1.0));
    assert_eq!(
        score(&t, |s| s.qpos[6] = 0.14),
        (ramp(0.14), Termination::Success)
    );
    assert_eq!(
        score(&t, |s| s.qpos[6] = 0.25),
        (ramp(0.25), Termination::Running)
    );
    assert_eq!(
        score(&t, |s| s.qpos[6] = 0.05),
        (ramp(0.05), Termination::Running)
    );
}

#[test]
fn above_and_below_compare_a_joint_and_a_free_joints_x() {
    let joint = |rel: &str| {
        task(&format!(
        "[success]\nclauses = [{{ subject = \"gripper\", relation = \"{rel}\", value = 0.85 }}]"
    ))
    };
    let (above, below) = (joint("above"), joint("below"));
    assert_eq!(score(&above, |s| s.qpos[5] = 1.0).1, Termination::Success);
    assert_eq!(score(&above, |s| s.qpos[5] = 0.5).1, Termination::Running);
    assert_eq!(score(&below, |s| s.qpos[5] = 0.5).1, Termination::Success);
    assert_eq!(score(&below, |s| s.qpos[5] = 1.0).1, Termination::Running);
    // In `[failure]`: the cube pushed out of reach ends the attempt as a failure.
    let out = task(
        "[success]\nclauses = [{ subject = \"gripper\", relation = \"above\", value = 9.0 }]\n\
         [failure]\nclauses = [{ subject = \"cube.x\", relation = \"above\", value = 0.55 }]",
    );
    assert_eq!(score(&out, |s| s.qpos[6] = 0.6).1, Termination::Failure);
    assert_eq!(score(&out, |s| s.qpos[6] = 0.5).1, Termination::Running);
}

#[test]
fn still_bounds_the_velocity_both_ways() {
    let t =
        task("[success]\nclauses = [{ subject = \"cube.x\", relation = \"still\", speed = 0.05 }]");
    assert_eq!(score(&t, |s| s.qvel[6] = 0.01).1, Termination::Success);
    assert_eq!(score(&t, |s| s.qvel[6] = 0.2).1, Termination::Running);
    assert_eq!(score(&t, |s| s.qvel[6] = -0.2).1, Termination::Running);
}

#[test]
fn near_and_farther_than_measure_a_body_or_a_point_and_pay_the_distance() {
    let (cube, jaw) = (row("cube"), row("gripper"));
    let near = task(
        "[success]\nclauses = [{ subject = \"cube\", relation = \"near\", object = \"gripper\", \
         m = 0.06, shaping = \"distance\", weight = -2.0 }]",
    );
    let place = |c: [f64; 3], g: [f64; 3]| {
        move |s: &mut State| {
            s.xpos[cube * 3..cube * 3 + 3].copy_from_slice(&c);
            s.xpos[jaw * 3..jaw * 3 + 3].copy_from_slice(&g);
        }
    };
    let (r, how) = score(&near, place([0.1, 0.0, 0.0], [0.1, 0.03, 0.04]));
    assert_eq!(how, Termination::Success);
    assert!((r + 2.0 * 0.05).abs() < 1e-12, "{r}");
    let (r, how) = score(&near, place([0.5, 0.0, 0.0], [0.1, 0.0, 0.0]));
    assert_eq!(how, Termination::Running);
    assert!((r + 2.0 * 0.4).abs() < 1e-12, "{r}");

    let far = task(
        "[success]\nclauses = [{ subject = \"cube\", relation = \"farther_than\", \
         point = [0.2, 0.0, 0.02], m = 0.1 }]",
    );
    let at = |p: [f64; 3]| move |s: &mut State| s.xpos[cube * 3..cube * 3 + 3].copy_from_slice(&p);
    assert_eq!(score(&far, at([0.2, 0.0, 0.02])).1, Termination::Running);
    assert_eq!(score(&far, at([0.5, 0.0, 0.02])).1, Termination::Success);
}

#[test]
fn orientation_matches_within_the_angle_and_pays_the_inverse_angle_curve() {
    let (cube, base) = (row("cube"), row("base"));
    let t = task(
        "[success]\nclauses = [{ subject = \"cube\", relation = \"orientation_matches\", \
         object = \"base\", within_deg = 10.0, shaping = \"inverse_angle\", weight = 1.0 }]\n\
         [reward]\nsuccess = 5.0",
    );
    let turn = |q: [f64; 4]| {
        move |s: &mut State| {
            s.xquat[cube * 4..cube * 4 + 4].copy_from_slice(&q);
            s.xquat[base * 4..base * 4 + 4].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);
        }
    };
    // The same orientation: every ramp is 1, the curve pays `1 / 0.1`, and the bonus.
    let (r, how) = score(&t, turn([0.0, 0.0, 0.0, 1.0]));
    assert_eq!(how, Termination::Success);
    assert!((r - (10.0 + 5.0)).abs() < 1e-12, "{r}");
    // 5 deg about z: inside 10 deg.
    let h = 2.5_f64.to_radians();
    assert_eq!(
        score(&t, turn([0.0, 0.0, h.sin(), h.cos()])).1,
        Termination::Success
    );
    // A quarter turn: running, and the interpolant of `1 / (s + 0.1)` at `s = √(8 (1 − d))`.
    let h = std::f64::consts::FRAC_PI_4;
    let (r, how) = score(&t, turn([0.0, 0.0, h.sin(), h.cos()]));
    assert_eq!(how, Termination::Running);
    let s = (8.0 * (1.0 - h.cos())).sqrt();
    let f = |s: f64| 1.0 / (s + 0.1);
    let want = f(0.8) + (f(1.6) - f(0.8)) * (s - 0.8) / 0.8;
    assert!((r - want).abs() < 1e-9, "{r} vs {want}");
}

#[test]
fn the_reset_lays_down_what_the_start_items_say() {
    let t = task(
        "[success]\nclauses = [{ subject = \"gripper\", relation = \"above\", value = 9.0 }]\n\
         [start]\nitems = [\n\
           { what = \"gripper\", value = 0.5 },\n\
           { what = \"shoulder_pan\", value = 0.1, noise = 0.05 },\n\
           { what = \"cube.x\", range = [0.21, 0.27], dice = true },\n\
           { what = \"cube.z\", value = 0.02 },\n\
           { what = \"cube.orientation\", draw = \"yaw\", tilt = 0.0355 },\n\
         ]",
    );
    let scene = scene();
    let domains = BatchDomains::single_env_at(TickRate::hz(200), TickRate::hz(50)).expect("4");
    let env = Env::new(&t, &scene, Still::new(), &domains, 7).expect("compiles");
    let q = env.backend().state().qpos.to_vec();
    assert_eq!(q[5], 0.5);
    assert!((q[0] - 0.1).abs() <= 0.05 && q[0] != 0.1, "{}", q[0]);
    assert!((0.21..=0.27).contains(&q[6]), "{}", q[6]);
    assert_eq!(q[8], 0.02);
    // One draw sets the quaternion `(1, t, −t·u, u)`.
    let u = q[12];
    assert_eq!((q[9], q[10]), (1.0, 0.0355));
    assert!(
        (q[11] + 0.0355 * u).abs() < 1e-15 && u.abs() <= 1.0,
        "{q:?}"
    );
}

/// The rotation of the free joint's quaternion `qpos[9..13]` (w x y z, unnormalized, as the
/// backend normalizes it): `(tilt of the body's +z from world +z in degrees, heading of the
/// body's +x about world +z in degrees, which of the six faces is on top)`.
fn pose_of(q: &[f64]) -> (f64, f64, usize) {
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    // The rotation matrix's last row: each body axis's world z.
    let up = [
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ];
    let face = |i: usize| if i < 3 { up[i] } else { -up[i - 3] };
    let top = (0..6)
        .max_by(|a, b| face(*a).total_cmp(&face(*b)))
        .expect("six faces");
    let heading = (2.0 * (x * y + w * z)).atan2(1.0 - 2.0 * (y * y + z * z));
    (
        up[2].clamp(-1.0, 1.0).acos().to_degrees(),
        heading.to_degrees(),
        top,
    )
}

/// `draws` resets of the cube's orientation drawn as `item` says.
fn orientations(item: &str, draws: usize) -> Vec<(f64, f64, usize)> {
    let t = task(&format!(
        "[success]\nclauses = [{{ subject = \"gripper\", relation = \"above\", value = 9.0 }}]\n\
         [start]\nitems = [{{ what = \"cube.orientation\", {item} }}]"
    ));
    let domains = BatchDomains::single_env_at(TickRate::hz(200), TickRate::hz(50)).expect("4");
    let mut env = Env::new(&t, &scene(), Still::new(), &domains, 7).expect("compiles");
    (0..draws)
        .map(|_| {
            env.reset(None).expect("reset");
            pose_of(&env.backend().state().qpos[9..13])
        })
        .collect()
}

#[test]
fn orientation_draws_stay_in_their_stated_ranges() {
    // `yaw`: the top face stays at the resting tilt whatever the yaw, over a half turn.
    let rest = 2.0 * 0.0355_f64.atan().to_degrees();
    for (tilt, heading, _) in orientations("draw = \"yaw\", tilt = 0.0355", 500) {
        assert!((tilt - rest).abs() < 1e-9, "{tilt} vs {rest}");
        assert!(heading.abs() <= 90.0 + 1e-9, "{heading}");
    }
    // `tilt`: never past theta, close to it at the extreme, and the yaw past the half turn on
    // both sides (`2·atan(g)`, g ~ N(0, 1): 8 % of draws are beyond ±120°).
    for theta in [45.0, 90.0] {
        let d = orientations(&format!("draw = \"tilt\", tilt_max_deg = {theta}"), 2000);
        let most = d.iter().map(|p| p.0).fold(0.0, f64::max);
        assert!(most <= theta + 1e-9, "theta = {theta}: {most}");
        assert!(
            most > 0.8 * theta,
            "theta = {theta}: the cap is reached ({most})"
        );
        assert!(
            d.iter().any(|p| p.1 > 120.0),
            "theta = {theta}: yaw is free"
        );
        assert!(
            d.iter().any(|p| p.1 < -120.0),
            "theta = {theta}: yaw is free"
        );
    }
    // `any`: every face ends up on top.
    let tops: BTreeSet<usize> = orientations("draw = \"any\"", 500)
        .iter()
        .map(|p| p.2)
        .collect();
    assert_eq!(tops.len(), 6, "{tops:?}");
}

/// The drawn quaternion is written unnormalized; the real backends normalize it — `xquat` at
/// once, `qpos` after the first step (plan H's H2b measured it for the yaw draw; here for
/// `any`, whose norm is anything). Needs `ES_PYTHON` (`MuJoCo`, `MJWarp`); prints SKIP without.
#[test]
fn the_backends_normalize_a_drawn_quaternion() {
    use es_physics_backend::{MjWarpBackend, MuJoCoCpuBackend};
    if std::env::var_os("ES_PYTHON").is_none() {
        println!("SKIP the_backends_normalize_a_drawn_quaternion: ES_PYTHON is not set");
        return;
    }
    let t = task(
        "[success]\nclauses = [{ subject = \"gripper\", relation = \"above\", value = 9.0 }]\n\
         [start]\nitems = [\n\
           { what = \"cube.x\", value = 0.24 },\n\
           { what = \"cube.z\", value = 0.1 },\n\
           { what = \"cube.orientation\", draw = \"any\" },\n\
         ]",
    );
    normalizes("mujoco-cpu", &t, MuJoCoCpuBackend::new());
    normalizes("mjwarp", &t, MjWarpBackend::new());
}

fn normalizes<B: PhysicsBackend>(name: &str, t: &TaskIr, backend: B) {
    let scene = scene();
    let rate = TickRate::from_period_secs(scene.options.timestep).expect("200 Hz");
    let domains = BatchDomains::single_env_at(rate, TickRate::hz(50)).expect("4");
    let mut env = match Env::new(t, &scene, backend, &domains, 7) {
        Ok(env) => env,
        Err(e) => return println!("SKIP {name}: {e}"),
    };
    let cube = row("cube");
    // MJWarp computes in f32.
    let unit = |q: &[f64]| (q.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-6;
    for episode in 0..4 {
        env.reset(None).expect("reset");
        let s = env.backend().state();
        let (q, xq) = (
            s.qpos[9..13].to_vec(),
            s.xquat[cube * 4..cube * 4 + 4].to_vec(),
        );
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        // xquat is x y z w; the same rotation as q / |q| up to sign.
        let dot = (xq[3] * q[0] + xq[0] * q[1] + xq[1] * q[2] + xq[2] * q[3]) / n;
        println!("{name} episode {episode}: |q| = {n:.4}, xquat {xq:?}");
        assert!(unit(&xq), "{name}: xquat {xq:?}");
        assert!((dot.abs() - 1.0).abs() < 1e-6, "{name}: {dot}");
        env.step(&[0.0; 6]).expect("a step");
        let after = env.backend().state().qpos[9..13].to_vec();
        assert!(unit(&after), "{name}: qpos after a step {after:?}");
    }
}
