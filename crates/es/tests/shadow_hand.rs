//! Plan H, packet M16/H2 (`docs/packets/M16/plan-h.md`): the Shadow Hand reorientation task and
//! the state-based PPO teacher's documents, `tests/fixtures/shadow-hand/`.
//!
//! Every IR document there is written by [`generate_shadow_hand_documents`] from the scene
//! `tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml` and the committed reach documents it
//! derives the deployment and evaluation from, so no hash, range or ctrlrange in them is typed in;
//! `shadow_hand_documents_are_what_the_generator_writes` holds the files to it. The training
//! recipe is hand-written and parsed here.
//!
//! ```text
//! ES_GENERATE_GOLDENS=1 cargo test -p es --test shadow_hand -- --ignored generate_shadow_hand_documents
//! ES_PYTHON=.venv/Scripts/python.exe cargo test -p es --test shadow_hand -- --nocapture
//! ```

// The graph builder and the reward formula are written in the header's own symbols (`q`, `d`,
// `s`, `f`, `c`, `g`), so a reader can hold the two side by side.
#![allow(clippy::many_single_char_names)]
// A reset writes exact constants and draws; those are compared exactly on purpose.
#![allow(clippy::float_cmp)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use es_assets::scene::{JointKind, SceneDesc, TendonKind};
use es_core::{StableId, TickRate};
use es_ir::deployment::{DeploymentIr, Limit, RobotTarget, Workspace};
use es_ir::evaluation::{AcceptanceCriterion, Comparator, EpisodeBatch, EvaluationIr, SeedPlan};
use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    ShutterModel,
};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, HeadKind, LearningGraph,
    LearningNode, NormalizeDir, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, StatsSource, WeightsRef,
};
use es_ir::observation::{
    Io, NormalizeStats, ObservationIr, ObservationNode, ObservationOutput, TemporalWindow,
};
use es_ir::task::{
    ActionSpace, Aggregation, CmpOp, Distribution, JointQuantity, MathFunc, NormKind, ObsChannel,
    ObsSource, ObservationSpec, SceneRef, SeedStream, SensorPath, SensorRender, TaskConfig, TaskIr,
    TaskNode, TerminationKind, Tonemap,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};

const SCENE: &str = "tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml";

/// The 24 hand joints, `qpos[0..24]` and `qvel[0..24]` in `MuJoCo`'s order.
const HAND: [&str; 24] = [
    "robot0:WRJ1",
    "robot0:WRJ0",
    "robot0:FFJ3",
    "robot0:FFJ2",
    "robot0:FFJ1",
    "robot0:FFJ0",
    "robot0:MFJ3",
    "robot0:MFJ2",
    "robot0:MFJ1",
    "robot0:MFJ0",
    "robot0:RFJ3",
    "robot0:RFJ2",
    "robot0:RFJ1",
    "robot0:RFJ0",
    "robot0:LFJ4",
    "robot0:LFJ3",
    "robot0:LFJ2",
    "robot0:LFJ1",
    "robot0:LFJ0",
    "robot0:THJ4",
    "robot0:THJ3",
    "robot0:THJ2",
    "robot0:THJ1",
    "robot0:THJ0",
];

/// Control at 60 Hz: two physics ticks of the scene's 1/120 s (Isaac Lab's decimation 2).
const CONTROL_HZ: u64 = 60;
/// 8 s at 60 Hz.
const STEPS: u32 = 480;
/// Where the cube rests on the palm with every joint at 0, measured (`MuJoCo` 3.13, the cube
/// dropped from its MJCF pose and from 2 cm lower at six yaws, 1 s): (1.00005, 0.86699-0.86716,
/// 0.17721-0.17724). The distance term and the drop predicate are measured from here.
const P_REF: [f64; 3] = [1.0, 0.867, 0.1772];
/// The resting cube's tilt about world `+X`, as `tan(alpha / 2)` of the quaternion
/// `(1, TILT, 0, 0)`: measured 0.0326-0.0355 over the same six yaws (alpha 3.7-4.1 deg), the
/// palm's own slope under the cube. The cube's reset pose and every goal carry it.
const TILT: f64 = 0.0355;
/// `cos(0.05)`: success when the rotation between cube and goal is under 0.1 rad.
const COS_TOLERANCE: f64 = 0.998_750_260_394_966_3;
/// The drop predicate: the cube's centre 0.24 m from `P_REF` (Isaac Lab's `fall_dist`).
const FALL_M: f64 = 0.24;
/// `rl_games`' `scale_value` on Isaac Lab's terms.
const SCALE: f64 = 0.01;
/// The small-angle rotation distance `s = sqrt(8 (1 - d))` at which `1 / (s + 0.1)` is sampled;
/// the last is `sqrt(8)`, `d = 0`.
const KNOTS: [f64; 9] = [
    0.0,
    0.025,
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    2.828_427_124_746_190_3,
];
/// Isaac Lab's `reset_dof_pos_noise`: each joint drawn over this fraction of its range.
const JOINT_NOISE: f64 = 0.2;
/// Isaac Lab's `reset_position_noise` on the cube's `x` and `y`.
const CUBE_NOISE: f64 = 0.01;
const IMAGE: u32 = 96;
/// Chosen by rendering: at 64 (the X7 rerun's) 3-6 % of each camera's pixels are the
/// tonemap's white; at 8 none are, the brightest is 242 and the hand keeps its shading
/// (re-checked with packet H1b's textured cubes and `MatViz` hand: brightest 245).
const EXPOSURE: f32 = 8.0;
const CAMERAS: [(&str, &str); 3] = [
    ("top", "rgb_top"),
    ("front", "rgb_front"),
    ("side", "rgb_side"),
];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo().join("tests/fixtures/shadow-hand").join(name)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hex(d: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*d).to_hex().to_string()
}

/// The scene as every CLI verb loads it (`es_tools::backend::load_scene`): parsed, then its
/// twelve STLs read relative to the file, so `scene_hash` covers the mesh content.
fn scene() -> (SceneDesc, Vec<u8>) {
    let path = repo().join(SCENE);
    let xml = std::fs::read(&path).expect("the hand scene");
    let parsed = es_assets::parse_mjcf(std::str::from_utf8(&xml).expect("utf-8")).expect("parses");
    assert!(parsed.warnings.is_empty(), "{:?}", parsed.warnings);
    let mut scene = parsed.scene;
    es_assets::mesh::load(&mut scene, path.parent().expect("a directory")).expect("STLs load");
    (scene, xml)
}

fn body(scene: &SceneDesc, name: &str) -> StableId {
    scene
        .bodies
        .iter()
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no body {name}"))
        .id
}

fn joint<'a>(scene: &'a SceneDesc, name: &str) -> &'a es_assets::scene::Joint {
    scene
        .joints
        .iter()
        .find(|j| j.name == name)
        .unwrap_or_else(|| panic!("no joint {name}"))
}

fn ty(n: u64, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

/// Half the scene's physics rate, exactly: the rate every runtime path steps at is the
/// timestep rounded to whole nanoseconds (`TickRate::from_period_secs`), so the scene's
/// `1/120 s` is `1e9 / 8,333,333` Hz and a control rate of exactly 60 Hz would be 2.00000008
/// ticks (`BatchDomains::single_env_at` refuses it). Two ticks of that clock is
/// `5e8 / 8,333,333` Hz, 60.0000024 Hz.
fn control_rate(scene: &SceneDesc) -> TickRate {
    let physics = TickRate::from_period_secs(scene.options.timestep).expect("a timestep");
    assert_eq!(physics.num() % 2, 0, "{physics:?}");
    TickRate::rational(physics.num() / 2, physics.den()).expect("non-zero")
}

/// `(lo, hi)` of every actuator's `ctrlrange`, in actuator order.
fn ctrlrange(scene: &SceneDesc) -> Vec<(f64, f64)> {
    scene
        .actuators
        .iter()
        .map(|a| a.ctrl_range.expect("every servo declares a ctrlrange"))
        .collect()
}

/// `J0 = k * J1` for each coupled pair, read off the four fixed tendons: a tendon of length
/// `c1 * J1 + c0 * J0` held at zero couples them with `k = -c1 / c0`.
fn couplings(scene: &SceneDesc) -> Vec<(String, String, f64)> {
    let name = |id: StableId| {
        scene
            .joints
            .iter()
            .find(|j| j.id == id)
            .expect("a tendon joint")
            .name
            .clone()
    };
    scene
        .tendons
        .iter()
        .map(|t| match &t.kind {
            TendonKind::Fixed { joints } => {
                let (j1, c1) = joints[0];
                let (j0, c0) = joints[1];
                (name(j1), name(j0), -c1 / c0)
            }
            TendonKind::Spatial { .. } => panic!("the hand has fixed tendons only"),
        })
        .collect()
}

/// A world-fixed 96x96 RGB camera of vertical field of view `fovy` (rad), as the renderer
/// delivers it (`es_env::render::image_spec`; `the_cameras_declare_what_the_scene_renders`).
fn camera_ty(camera: StableId, fovy: f64) -> PortType {
    let f = f64::from(IMAGE) / 2.0 / (fovy / 2.0).tan();
    let c = f64::from(IMAGE) / 2.0;
    PortType {
        elem: ElemType::U8,
        shape: Shape::new([u64::from(IMAGE), u64::from(IMAGE), 3]),
        unit: Unit::Pixel,
        frame: Frame::Camera(camera),
        time: TimeRef::Sensor {
            id: camera,
            align: Align::Hold,
        },
        image: Some(ImageSpec {
            width: IMAGE,
            height: IMAGE,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(f, f, c, c),
            extrinsics: es_math::Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_millis(2),
            rate_hz: CONTROL_HZ as f32,
            depth_scale: None,
        }),
    }
}

/// `1 / (s + 0.1)`, Isaac Lab's rotation reward, at each knot.
fn rotation_at_knots() -> Vec<f64> {
    KNOTS.iter().map(|s| 1.0 / (s + 0.1)).collect()
}

// --- the Task IR --------------------------------------------------------------------------------

/// Builds the graph node by node; `add` returns the id it used.
struct Builder {
    graph: Graph<TaskNode>,
}

impl Builder {
    fn add(&mut self, node: TaskNode) -> NodeId {
        let id = NodeId(self.graph.nodes.len() as u32);
        self.graph.insert(id, node);
        id
    }

    fn link(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
        self.graph.connect(from, from_port, to, to_port);
    }

    fn reset(&mut self, target: String, lo: f64, hi: f64, stream: &str) {
        let dist = if lo.to_bits() == hi.to_bits() {
            Distribution::Constant(lo)
        } else {
            Distribution::Uniform { lo, hi }
        };
        self.add(TaskNode::ResetState {
            target,
            dist,
            stream: stream.to_owned(),
        });
    }
}

#[allow(clippy::too_many_lines)]
fn task() -> TaskIr {
    let (scene, xml) = scene();
    let (mount, object, target) = (
        body(&scene, "robot0:hand mount"),
        body(&scene, "object"),
        body(&scene, "target"),
    );
    let joints = |unit: Unit| ty(24, unit, Frame::Joint(mount));
    let pos3 = ty(3, Unit::Length, Frame::World);
    let quat4 = ty(4, Unit::Quaternion, Frame::World);
    let pose7 = ty(7, Unit::Length, Frame::World);
    let vel6 = ty(6, Unit::Velocity, Frame::World);
    let names: Vec<String> = HAND.iter().map(|s| (*s).to_owned()).collect();
    let signed = Unit::Normalized { lo: -1.0, hi: 1.0 };
    let unit01 = Unit::Normalized { lo: 0.0, hi: 1.0 };
    let flag = PortType {
        elem: ElemType::Bool,
        ..ty(1, Unit::Dimensionless, Frame::World)
    };

    let mut b = Builder {
        graph: Graph::new(2),
    };
    // --- the observation channels -----------------------------------------------------------
    let q = b.add(TaskNode::GetJointState {
        body: mount,
        joints: names.clone(),
        quantity: JointQuantity::Position,
    });
    let spec = b.add(TaskNode::ObservationSpec {
        channel: "joint_pos".to_owned(),
        ty: joints(Unit::Angle),
    });
    b.link(q, "value", spec, "value");
    let qd = b.add(TaskNode::GetJointState {
        body: mount,
        joints: names,
        quantity: JointQuantity::Velocity,
    });
    let spec = b.add(TaskNode::ObservationSpec {
        channel: "joint_vel".to_owned(),
        ty: joints(Unit::AngularVelocity),
    });
    b.link(qd, "value", spec, "value");
    let pose_channel = |b: &mut Builder, body: StableId, channel: &str| {
        let get = b.add(TaskNode::GetBodyPose {
            body,
            relative_to: Frame::World,
        });
        let cat = b.add(TaskNode::Concat {
            parts: vec![pos3.clone(), quat4.clone()],
            axis: 0,
        });
        let spec = b.add(TaskNode::ObservationSpec {
            channel: channel.to_owned(),
            ty: pose7.clone(),
        });
        b.link(get, "pos", cat, "in0");
        b.link(get, "quat", cat, "in1");
        b.link(cat, "value", spec, "value");
        get
    };
    let cube = pose_channel(&mut b, object, "cube_pose");
    let vel = b.add(TaskNode::GetBodyVelocity {
        body: object,
        relative_to: Frame::World,
    });
    let cat = b.add(TaskNode::Concat {
        parts: vec![
            ty(3, Unit::Velocity, Frame::World),
            ty(3, Unit::AngularVelocity, Frame::World),
        ],
        axis: 0,
    });
    let spec = b.add(TaskNode::ObservationSpec {
        channel: "object_vel".to_owned(),
        ty: vel6.clone(),
    });
    b.link(vel, "linear", cat, "in0");
    b.link(vel, "angular", cat, "in1");
    b.link(cat, "value", spec, "value");
    let goal = pose_channel(&mut b, target, "goal_pose");
    b.add(TaskNode::ActionSpec {
        space: ActionSpace::JointPosition,
        dim: 20,
        control_rate_hz: CONTROL_HZ as f32,
    });

    // --- distance: -10 ||p_cube - p_ref||, and the drop ----------------------------------------
    // `p_cube - p_ref` is a per-lane `Normalize` whose map is the identity shifted by `p_ref`
    // (`[p - 1, p + 1] -> [-1, 1]`): the IR has no constant node, and a lane-wise `Arith` needs
    // a second port. The clamp at 1 m per axis is past the 0.24 m drop.
    let offset = b.add(TaskNode::Normalize {
        lo: P_REF.iter().map(|p| p - 1.0).collect(),
        hi: P_REF.iter().map(|p| p + 1.0).collect(),
        out_lo: -1.0,
        out_hi: 1.0,
        ty: pos3.clone(),
    });
    b.link(cube, "pos", offset, "value");
    let dist = b.add(TaskNode::Norm {
        kind: NormKind::L2,
        ty: ty(3, signed.clone(), Frame::World),
    });
    b.link(offset, "value", dist, "value");
    let metres = b.add(TaskNode::Normalize {
        lo: vec![0.0],
        hi: vec![1.0],
        out_lo: 0.0,
        out_hi: 1.0,
        ty: ty(1, signed.clone(), Frame::World),
    });
    b.link(dist, "value", metres, "value");
    let term = b.add(TaskNode::Reward {
        name: "cube_distance".to_owned(),
        weight: -10.0 * SCALE,
        aggregation: Aggregation::Sum,
        ty: ty(1, unit01.clone(), Frame::World),
    });
    b.link(metres, "value", term, "value");
    let fell = b.add(TaskNode::Compare {
        op: CmpOp::Gt,
        rhs: Some(FALL_M),
        ty: ty(1, signed, Frame::World),
    });
    b.link(dist, "value", fell, "a");
    let stop = b.add(TaskNode::Terminate {
        kind: TerminationKind::Failure,
    });
    b.link(fell, "value", stop, "value");

    // --- rotation: d = |q_cube . q_goal| ------------------------------------------------------
    let dot = b.add(TaskNode::Dot { ty: quat4 });
    b.link(cube, "quat", dot, "a");
    b.link(goal, "quat", dot, "b");
    let q1 = ty(1, Unit::Quaternion, Frame::World);
    let d = b.add(TaskNode::MathFn {
        func: MathFunc::Abs,
        approx: false,
        ty: q1.clone(),
    });
    b.link(dot, "value", d, "value");
    let reached = b.add(TaskNode::Compare {
        op: CmpOp::Ge,
        rhs: Some(COS_TOLERANCE),
        ty: q1.clone(),
    });
    b.link(d, "value", reached, "a");
    let bonus = b.add(TaskNode::Reward {
        name: "success".to_owned(),
        weight: 250.0 * SCALE,
        aggregation: Aggregation::Sum,
        ty: flag.clone(),
    });
    b.link(reached, "value", bonus, "value");
    let done = b.add(TaskNode::Terminate {
        kind: TerminationKind::Success,
    });
    b.link(reached, "value", done, "value");
    // s = sqrt(8 (1 - d)), then `1 / (s + 0.1)` as the sum of one clamped ramp per knot
    // interval: exact at every knot, linear between (there is no reciprocal a `Reward` can
    // take -- `Arith { Div }` needs a `Dimensionless` divisor and nothing derived from a
    // quaternion is one).
    let eight = Unit::Normalized { lo: 0.0, hi: 8.0 };
    let gap = b.add(TaskNode::Normalize {
        lo: vec![1.0],
        hi: vec![0.0],
        out_lo: 0.0,
        out_hi: 8.0,
        ty: q1,
    });
    b.link(d, "value", gap, "value");
    let s = b.add(TaskNode::MathFn {
        func: MathFunc::Sqrt,
        approx: false,
        ty: ty(1, eight.clone(), Frame::World),
    });
    b.link(gap, "value", s, "value");
    let f = rotation_at_knots();
    for k in 0..KNOTS.len() - 1 {
        // 1 at s <= s_k, 0 at s >= s_{k+1}.
        let ramp = b.add(TaskNode::Normalize {
            lo: vec![KNOTS[k + 1]],
            hi: vec![KNOTS[k]],
            out_lo: 0.0,
            out_hi: 1.0,
            ty: ty(1, eight.clone(), Frame::World),
        });
        b.link(s, "value", ramp, "value");
        let term = b.add(TaskNode::Reward {
            name: format!("rotation_{k}"),
            weight: SCALE * (f[k] - f[k + 1]),
            aggregation: Aggregation::Sum,
            ty: ty(1, unit01.clone(), Frame::World),
        });
        b.link(ramp, "value", term, "value");
    }
    // The floor `1 / (sqrt(8) + 0.1)`, paid every step: `time since reset >= 0` is always true.
    let time = b.add(TaskNode::GetTime { since_reset: true });
    let always = b.add(TaskNode::Compare {
        op: CmpOp::Ge,
        rhs: Some(0.0),
        ty: ty(1, Unit::Time, Frame::World),
    });
    b.link(time, "value", always, "a");
    let floor = b.add(TaskNode::Reward {
        name: "rotation_floor".to_owned(),
        weight: SCALE * f[KNOTS.len() - 1],
        aggregation: Aggregation::Sum,
        ty: flag,
    });
    b.link(always, "value", floor, "value");
    let late = b.add(TaskNode::Compare {
        op: CmpOp::Ge,
        rhs: Some(f64::from(STEPS) / CONTROL_HZ as f64),
        ty: ty(1, Unit::Time, Frame::World),
    });
    b.link(time, "value", late, "a");
    let timeout = b.add(TaskNode::Terminate {
        kind: TerminationKind::Timeout,
    });
    b.link(late, "value", timeout, "value");

    // --- the three cameras -------------------------------------------------------------------
    let render = SensorRender {
        path: SensorPath::Pt {
            spp: 32,
            bounces: 3,
        },
        exposure: EXPOSURE,
        tonemap: Tonemap::Reinhard,
        seed: SeedStream::Tick,
        svgf: false,
    };
    let mut channels = BTreeMap::new();
    for (name, channel) in CAMERAS {
        let cam = scene
            .cameras
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no camera {name}"));
        let image = camera_ty(cam.id, cam.fovy);
        let get = b.add(TaskNode::GetSensor {
            sensor: cam.id,
            ty: image.clone(),
        });
        let spec = b.add(TaskNode::ObservationSpec {
            channel: channel.to_owned(),
            ty: image.clone(),
        });
        b.link(get, "value", spec, "value");
        channels.insert(
            channel.to_owned(),
            ObsChannel {
                source: ObsSource::Sensor {
                    id: cam.id,
                    format: ChannelFormat::Rgb,
                    render,
                },
                ty: image,
            },
        );
    }

    // --- reset -------------------------------------------------------------------------------
    let mut streams = BTreeSet::new();
    // Each actuated joint over 0.2 of its range around 0 (every range contains 0); each coupled
    // J0 from its J1's stream, so the same draw sets both and the tendon starts at length 0.
    let coupled = couplings(&scene);
    for name in HAND {
        let (lo, hi) = joint(&scene, name)
            .range
            .expect("every hand joint is limited");
        let (stream, k) = match coupled.iter().find(|(_, j0, _)| j0 == name) {
            Some((j1, _, k)) => {
                let (lo1, hi1) = joint(&scene, j1).range.expect("limited");
                assert!(lo1 * k >= lo && hi1 * k * JOINT_NOISE <= hi, "{name}");
                (format!("reset.{j1}"), Some((lo1 * k, hi1 * k)))
            }
            None => (format!("reset.{name}"), None),
        };
        let (lo, hi) = k.unwrap_or((lo, hi));
        b.reset(
            format!("joint.{name}.qpos"),
            JOINT_NOISE * lo,
            JOINT_NOISE * hi,
            &stream,
        );
        streams.insert(stream);
    }
    // The cube at `P_REF` (x, y within 1 cm), resting tilt, a yaw drawn from one `u ~ U(-1, 1)`:
    // `q = q_x(alpha) * q_z(2 atan u)` unnormalized is `(1, TILT, -TILT u, u)`, each lane linear
    // in `u`, so two `Uniform` nodes on one stream write it from the same draw. The backend
    // normalizes a free joint's quaternion (`MuJoCo` and `MJWarp` both, measured); the Task IR
    // and the Observation IR read the body's normalized `xquat`, never `qpos`.
    let free = |j: &str| {
        let start = scene
            .joints
            .iter()
            .position(|x| x.name == j)
            .expect("free joint");
        assert_eq!(scene.joints[start].kind, JointKind::Free);
        // 24 hinges before the cube's free joint, each one `qpos` entry.
        24 + 7 * (start - 24)
    };
    let (c, g) = (free("object:joint"), free("target:joint"));
    for (axis, noise) in [(0, CUBE_NOISE), (1, CUBE_NOISE), (2, 0.0)] {
        let stream = format!("cube.pos.{axis}");
        b.reset(
            format!("qpos[{}]", c + axis),
            P_REF[axis] - noise,
            P_REF[axis] + noise,
            &stream,
        );
        streams.insert(stream);
    }
    for (start, stream) in [(c + 3, "cube.yaw"), (g + 3, "goal.yaw")] {
        b.reset(format!("qpos[{start}]"), 1.0, 1.0, stream);
        b.reset(format!("qpos[{}]", start + 1), TILT, TILT, stream);
        b.reset(format!("qpos[{}]", start + 2), TILT, -TILT, stream);
        b.reset(format!("qpos[{}]", start + 3), -1.0, 1.0, stream);
        streams.insert(stream.to_owned());
    }

    channels.extend([
        (
            "joint_pos".to_owned(),
            ObsChannel {
                // Not a joint's id: the leading 24 of `qpos`, which are the hand's (as the
                // reach task's `base`).
                source: ObsSource::JointState {
                    body: mount,
                    dof: 24,
                    quantity: JointQuantity::Position,
                },
                ty: joints(Unit::Angle),
            },
        ),
        (
            "joint_vel".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: joint(&scene, HAND[0]).id,
                    dof: 24,
                    quantity: JointQuantity::Velocity,
                },
                ty: joints(Unit::AngularVelocity),
            },
        ),
        (
            "cube_pose".to_owned(),
            ObsChannel {
                source: ObsSource::BodyPose(object),
                ty: pose7.clone(),
            },
        ),
        (
            "object_vel".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: joint(&scene, "object:joint").id,
                    dof: 6,
                    quantity: JointQuantity::Velocity,
                },
                ty: vel6,
            },
        ),
        (
            "goal_pose".to_owned(),
            ObsChannel {
                source: ObsSource::BodyPose(target),
                ty: pose7,
            },
        ),
        (
            "last_action".to_owned(),
            ObsChannel {
                // Isaac Lab's `last_action` is the raw action, zero at reset; ours is in
                // actuator units, so zero-raw is the centre of each ctrlrange.
                source: ObsSource::PreviousAction {
                    initial: Some(
                        ctrlrange(&scene)
                            .iter()
                            .map(|(lo, hi)| (lo + hi) / 2.0)
                            .collect(),
                    ),
                },
                ty: ty(20, Unit::Angle, Frame::World),
            },
        ),
    ]);

    let task = TaskIr {
        schema_version: es_ir::task::SCHEMA_VERSION,
        scene: SceneRef {
            path: SCENE.to_owned(),
            scene_hash: scene.scene_hash(),
            asset_hash: *blake3::hash(&xml).as_bytes(),
        },
        graph: b.graph,
        observation_spec: ObservationSpec { channels },
        config: TaskConfig {
            max_episode_steps: STEPS,
            control_rate_hz: CONTROL_HZ as f32,
            deterministic: true,
            rng_streams: streams,
        },
        control: None,
    };
    let diags = task.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    task
}

// --- the teacher's Observation IR ---------------------------------------------------------------

/// The layout order of the teacher's state vector.
const STATE: [&str; 6] = [
    "joint_pos",
    "joint_vel",
    "cube_pose",
    "object_vel",
    "goal_pose",
    "last_action",
];

/// Per-element `(mean, std)` bringing channel `name` to about `[-1, 1]` with static ranges.
fn channel_stats(scene: &SceneDesc, name: &str) -> Vec<(f64, f64)> {
    let n = |k: usize, m: f64, s: f64| vec![(m, s); k];
    match name {
        "joint_pos" => HAND
            .iter()
            .map(|j| {
                let (lo, hi) = joint(scene, j).range.expect("limited");
                (f64::midpoint(lo, hi), (hi - lo) / 2.0)
            })
            .collect(),
        // Isaac Lab scales joint velocities by 0.2.
        "joint_vel" => n(24, 0.0, 5.0),
        // The cube: +-1 at the drop radius; the quaternion is already in [-1, 1].
        "cube_pose" => P_REF
            .iter()
            .map(|p| (*p, FALL_M))
            .chain(n(4, 0.0, 1.0))
            .collect(),
        // Linear m/s as is, angular rad/s by 0.2 as Isaac Lab does.
        "object_vel" => [n(3, 0.0, 1.0), n(3, 0.0, 5.0)].concat(),
        // The goal cube floats where the scene puts it; its position is a constant.
        "goal_pose" => {
            let target = scene
                .bodies
                .iter()
                .find(|b| b.name == "target")
                .expect("target");
            let p = target.pose.position;
            [p.x, p.y, p.z]
                .iter()
                .map(|v| (*v, 1.0))
                .chain(n(4, 0.0, 1.0))
                .collect()
        }
        "last_action" => ctrlrange(scene)
            .iter()
            .map(|(lo, hi)| (f64::midpoint(*lo, *hi), (hi - lo) / 2.0))
            .collect(),
        other => panic!("no statistics for {other}"),
    }
}

fn state_width(task: &TaskIr, names: &[&str]) -> u64 {
    names
        .iter()
        .map(|n| task.observation_spec.channels[*n].ty.shape.dims()[0])
        .sum()
}

/// `names`' channels as `StateInput`s from node `base` on, one `Concat` and one
/// `Normalize { MeanStd }` after them; returns the normalized port and its type.
fn add_state(
    ir: &mut ObservationIr,
    task: &TaskIr,
    names: &[&str],
    base: u32,
) -> (PortRef, PortType) {
    let (scene, _) = scene();
    let mut parts = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let ch = &task.observation_spec.channels[*name];
        let source = match ch.source {
            ObsSource::JointState { body, .. } | ObsSource::BodyPose(body) => body,
            ObsSource::PreviousAction { .. } => ObsSource::previous_action_id(),
            ref other => panic!("{other:?}"),
        };
        ir.graph.insert(
            NodeId(base + i as u32),
            ObservationNode::StateInput {
                source,
                io: Io::source(ch.ty.clone()),
            },
        );
        parts.push(ch.ty.clone());
    }
    let wide = ty(state_width(task, names), Unit::Dimensionless, Frame::World);
    let normalized = PortType {
        unit: Unit::Normalized { lo: -1.0, hi: 1.0 },
        ..wide.clone()
    };
    let k = names.len() as u32;
    let (cat, norm) = (NodeId(base + k), NodeId(base + k + 1));
    ir.graph.insert(
        cat,
        ObservationNode::Concat {
            axis: 0,
            time_align: None,
            io: Io::new(parts, wide.clone()),
        },
    );
    let (mean, std): (Vec<f64>, Vec<f64>) =
        names.iter().flat_map(|n| channel_stats(&scene, n)).unzip();
    assert_eq!(mean.len() as u64, state_width(task, names));
    ir.graph.insert(
        norm,
        ObservationNode::Normalize {
            stats: NormalizeStats::MeanStd { mean, std },
            io: Io::unary(wide, normalized.clone()),
        },
    );
    for i in 0..k {
        ir.graph
            .connect(NodeId(base + i), "out", cat, &format!("in{i}"));
    }
    ir.graph.connect(cat, "out", norm, "in0");
    (PortRef::new(norm, "out"), normalized)
}

fn observation(task: &TaskIr) -> ObservationIr {
    let mut ir = ObservationIr::new(1, task.task_hash().expect("hash"));
    let (port, normalized) = add_state(&mut ir, task, &STATE, 0);
    ir.graph.outputs.push(port.clone());
    ir.temporal.window = Some(TemporalWindow {
        n_steps: 1,
        stride: 1,
        align: Align::Hold,
    });
    ir.outputs = BTreeMap::from([(
        "state".to_owned(),
        ObservationOutput {
            port,
            ty: normalized,
        },
    )]);
    let diags = ir.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    ir
}

// --- the teacher's Learning IR ------------------------------------------------------------------

fn learning(task: &TaskIr) -> LearningGraph {
    let (scene, _) = scene();
    let signed = Unit::Normalized { lo: -1.0, hi: 1.0 };
    let state = Port::new(
        "state",
        ty(state_width(task, &STATE), signed.clone(), Frame::Policy),
    );
    let chunk = PortType {
        shape: Shape::new([1, 20]),
        ..ty(1, signed, Frame::Policy)
    };
    let mut nodes: Graph<LearningNode> = Graph::new(1);
    nodes.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            // rl_games' Shadow Hand actor: ELU over [512, 256, 128].
            kind: StateEncoderKind::Mlp {
                hidden: vec![512, 256],
                activation: Activation::Elu,
                activate_output: true,
            },
            out_dim: 128,
        },
    );
    nodes.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![Port::new(
                "feat",
                ty(128, Unit::Dimensionless, Frame::Policy),
            )],
            kind: HeadKind::Regression,
            action_dim: 20,
            horizon: 1,
            squash: Squash::Tanh,
        },
    );
    nodes.insert(
        NodeId(2),
        LearningNode::ActionChunker {
            inputs: vec![Port::new("chunk", chunk.clone())],
            horizon: 1,
            execute_chunk: 1,
            replan_hz: CONTROL_HZ as f32,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    let range = ctrlrange(&scene);
    nodes.insert(
        NodeId(3),
        LearningNode::Normalizer {
            inputs: vec![Port::new("actions", chunk.clone())],
            direction: NormalizeDir::Inverse,
            stats: StatsSource::MeanStd {
                mean: range.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect(),
                std: range.iter().map(|(lo, hi)| (hi - lo) / 2.0).collect(),
            },
            out_unit: Unit::Angle,
        },
    );
    nodes.connect(NodeId(0), "out", NodeId(1), "feat");
    nodes.connect(NodeId(1), "chunk", NodeId(2), "chunk");
    nodes.connect(NodeId(2), "actions", NodeId(3), "actions");
    nodes.inputs.push(PortRef::new(NodeId(0), "state"));
    nodes.outputs.push(PortRef::new(NodeId(3), "out"));
    let actions = PortType {
        unit: Unit::Angle,
        ..chunk
    };
    let g = LearningGraph {
        schema_version: 1,
        inputs: vec![state.clone()],
        nodes,
        outputs: vec![Port::new("actions", actions)],
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: BTreeMap::from([("state".to_owned(), state)]),
                observation_window: 1,
                action_dim: 20,
                horizon: 1,
                execute_chunk: 1,
                replanning_hz: CONTROL_HZ as f32,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 2.0,
                    deadline_ms: 16.0,
                },
            },
        },
    };
    let diags = g.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    g
}

// --- the Deployment IR --------------------------------------------------------------------------

/// The Safety Plane over the 20 servos, from `deployment-reach.toml`'s execution (one action
/// per tick, receding horizon, the watchdogs and the fallback) with the hand's own envelope.
fn deployment() -> DeploymentIr {
    let (scene, _) = scene();
    let mut dep = es_ir::serial::deployment_from_toml(&read(
        &repo().join("tests/fixtures/rl/deployment-reach.toml"),
    ))
    .expect("deployment-reach.toml");
    let range = ctrlrange(&scene);
    let n = range.len();
    let hz = CONTROL_HZ as f64;
    let width: Vec<f64> = range.iter().map(|(lo, hi)| hi - lo).collect();
    "shadow_hand".clone_into(&mut dep.robot.name);
    dep.robot.n_joints = n;
    dep.robot.target = RobotTarget::Simulated {
        scene: SCENE.to_owned(),
    };
    dep.action.dim = n;
    dep.rate.control = control_rate(&scene);
    dep.rate.inference = control_rate(&scene);
    let s = &mut dep.safety;
    s.position = range
        .iter()
        .map(|(lo, hi)| Limit {
            lower: *lo,
            upper: *hi,
        })
        .collect();
    s.position_soft_margin = vec![0.0; n];
    // Twice the widest jump inside the range in one control tick, so a target anywhere in the
    // range is never clamped by rate; the position stage is what binds (the owner's review).
    s.velocity_max = width.iter().map(|w| 2.0 * w * hz).collect();
    s.acceleration_max = width.iter().map(|w| 4.0 * w * hz * hz).collect();
    s.action_rate.first_diff_max = width.iter().map(|w| 2.0 * w).collect();
    s.action_rate.second_diff_max = width.iter().map(|w| 4.0 * w).collect();
    s.torque_max = scene
        .actuators
        .iter()
        .map(|a| a.force_range.expect("every servo declares a forcerange").1)
        .collect();
    s.jerk_max = None;
    // Joint space: the workspace stage does not apply (spec 9.3); a box around the hand.
    s.workspace = Workspace::Box {
        min: [0.7, 0.5, 0.0],
        max: [1.3, 1.5, 0.6],
    };
    let diags = dep.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    dep
}

// --- the Evaluation IR --------------------------------------------------------------------------

fn evaluation(task: &TaskIr, obs: &ObservationIr) -> EvaluationIr {
    let mut ev = es_ir::serial::evaluation_from_toml(&read(
        &repo().join("tests/fixtures/rl/evaluation-reach.toml"),
    ))
    .expect("evaluation-reach.toml");
    ev.task = hex(&task.task_hash().expect("hash"));
    ev.observation = hex(&obs.observation_hash().expect("hash"));
    ev.suites.retain(|s| s.name == "nominal");
    ev.episodes = EpisodeBatch {
        n_episodes: 16,
        seeds: SeedPlan::Explicit((301..=316).collect()),
    };
    ev.acceptance = vec![AcceptanceCriterion {
        suite: Some("nominal".to_owned()),
        metric: es_ir::evaluation::MetricSpec::SuccessRate,
        comparator: Comparator::Ge,
        threshold: 0.5,
        aggregation: es_ir::evaluation::Aggregation::Mean,
    }];
    let diags = ev.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    ev
}

// --- the student (packet M16/H3) ----------------------------------------------------------------

/// The student's state: the hand's joints and the goal. Never the cube: its pose is what the
/// cameras are for.
const STUDENT_STATE: [&str; 2] = ["joint_pos", "goal_pose"];
/// Plan N's three views onto this scene's three cameras: `observation-views.toml`'s chains and
/// `learning-views.toml`'s encoders, in that order.
const VIEWS: [(&str, &str); 3] = [
    ("rgb_overhead", "rgb_top"),
    ("rgb_wrist", "rgb_front"),
    ("rgb_side", "rgb_side"),
];
/// The student's chunk: 16 rows predicted (0.27 s at 60 Hz), 6 executed, so it replans at
/// 10 Hz with a temporal ensemble over the overlapping rows.
const CHUNK: u32 = 16;
const EXECUTE: u32 = 6;
/// The held-out seeds every Shadow Hand policy is judged on from packet M16/H3: 64, the
/// teacher's 16 (301-316) first, so the two documents' shared episodes are the same episodes.
const SEEDS_64: std::ops::RangeInclusive<u64> = 301..=364;

fn vl(name: &str) -> PathBuf {
    repo().join("tests/fixtures/visible-learning").join(name)
}

/// `value` through its serialized form with every string and map key in `names` renamed and,
/// with `image = Some((fx, hz))`, every `fx` / `fy` set to `fx` and every `rate_hz` to `hz`.
/// A camera appears in a node, a port type's frame, its time reference and its image spec;
/// the serialized form catches every one without naming each (`crates/es/tests/views.rs`'s
/// `retarget`, with a map).
fn rewrite<T: serde::Serialize + serde::de::DeserializeOwned>(
    value: &T,
    names: &BTreeMap<String, String>,
    image: Option<(f64, f64)>,
) -> T {
    fn walk(
        v: &mut serde_json::Value,
        names: &BTreeMap<String, String>,
        image: Option<(f64, f64)>,
    ) {
        match v {
            serde_json::Value::String(s) => {
                if let Some(to) = names.get(s.as_str()) {
                    to.clone_into(s);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    walk(item, names, image);
                }
            }
            serde_json::Value::Object(map) => {
                for (key, mut item) in std::mem::take(map) {
                    match (key.as_str(), image) {
                        ("fx" | "fy", Some((fx, _))) => item = serde_json::json!(fx),
                        ("rate_hz", Some((_, hz))) => item = serde_json::json!(hz),
                        _ => walk(&mut item, names, image),
                    }
                    map.insert(names.get(&key).cloned().unwrap_or(key), item);
                }
            }
            _ => {}
        }
    }
    let mut v = serde_json::to_value(value).expect("serializes");
    walk(&mut v, names, image);
    serde_json::from_value(v).expect("deserializes")
}

/// `ir` without output `name` and the nodes that exist only to produce it (views.rs's).
fn drop_output(ir: &mut ObservationIr, name: &str) {
    let out = ir.outputs.remove(name).expect("the output");
    let mut gone = vec![out.port.node];
    while let Some(e) = ir
        .graph
        .edges
        .iter()
        .find(|e| gone.contains(&e.to.node) && !gone.contains(&e.from.node))
    {
        gone.push(e.from.node);
    }
    ir.graph
        .edges
        .retain(|e| !gone.contains(&e.from.node) && !gone.contains(&e.to.node));
    for id in &gone {
        ir.graph.nodes.remove(id);
    }
}

fn id_string(id: StableId) -> String {
    serde_json::to_value(id)
        .expect("id")
        .as_str()
        .expect("an id serializes as a string")
        .to_owned()
}

/// observation-views.toml's three image chains on this scene's cameras, and the state of
/// [`STUDENT_STATE`] in place of `joint_state`.
fn observation_student(task: &TaskIr) -> ObservationIr {
    let views = es_ir::serial::observation_from_toml(&read(&vl("observation-views.toml")))
        .expect("observation-views.toml");
    let mut names = BTreeMap::new();
    let mut fx = None;
    for (from, to) in VIEWS {
        let Frame::Camera(old) = views.outputs[from].ty.frame else {
            panic!("{from} is not a camera port")
        };
        let ch = &task.observation_spec.channels[to];
        let ObsSource::Sensor { id, .. } = ch.source else {
            panic!("{to} is not a sensor")
        };
        names.insert(id_string(old), id_string(id));
        names.insert(from.to_owned(), to.to_owned());
        let f = ch.ty.image.as_ref().expect("image").intrinsics.fx;
        assert!(
            fx.is_none_or(|x| x == f),
            "the three cameras share one fovy"
        );
        fx = Some(f);
    }
    let hz = task.config.control_rate_hz.into();
    let mut obs = rewrite(&views, &names, Some((fx.expect("three cameras"), hz)));
    drop_output(&mut obs, "joint_state");
    obs.task_ref = task.task_hash().expect("hash");
    let base = obs.graph.nodes.keys().map(|n| n.0).max().expect("nodes") + 1;
    let (port, ty) = add_state(&mut obs, task, &STUDENT_STATE, base);
    obs.outputs
        .insert("state".to_owned(), ObservationOutput { port, ty });
    let diags = obs.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    obs
}

/// learning-views.toml on this task: the views and the state renamed, the state 31 wide, a
/// 20-wide action chunked 16 / 6 at 10 Hz, and the teacher's unnormalizer after the chunker.
fn learning_student(task: &TaskIr) -> LearningGraph {
    let (scene, _) = scene();
    let views = es_ir::serial::learning_from_toml(&read(&vl("learning-views.toml")))
        .expect("learning-views.toml");
    let names: BTreeMap<String, String> = VIEWS
        .iter()
        .chain(&[("joint_state", "state")])
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect();
    let mut g = rewrite(&views, &names, None);
    let width = Shape::new([state_width(task, &STUDENT_STATE)]);
    let chunk = Shape::new([u64::from(CHUNK), 20]);
    let fused = |p: &str| {
        match p {
            "image" => "top",
            "wrist" => "front",
            other => other,
        }
        .to_owned()
    };
    let mut fusion = None;
    let mut chunker = None;
    for (id, node) in &mut g.nodes.nodes {
        match node {
            LearningNode::StateEncoder { inputs, .. } => inputs[0].ty.shape = width.clone(),
            LearningNode::Fusion { inputs, .. } => {
                fusion = Some(*id);
                for p in inputs {
                    p.name = fused(&p.name);
                }
            }
            LearningNode::PolicyHead {
                action_dim,
                horizon,
                ..
            } => (*action_dim, *horizon) = (20, CHUNK),
            LearningNode::ActionChunker {
                inputs,
                horizon,
                execute_chunk,
                replan_hz,
                ..
            } => {
                chunker = Some(*id);
                inputs[0].ty.shape = chunk.clone();
                (*horizon, *execute_chunk) = (CHUNK, EXECUTE);
                *replan_hz = (CONTROL_HZ / u64::from(EXECUTE)) as f32;
            }
            _ => {}
        }
    }
    let (fusion, chunker) = (fusion.expect("a fusion"), chunker.expect("a chunker"));
    for e in &mut g.nodes.edges {
        if e.to.node == fusion {
            e.to.port = fused(&e.to.port);
        }
    }
    for p in &mut g.inputs {
        if p.name == "state" {
            p.ty.shape = width.clone();
        }
    }
    let contract = &mut g.policy.contract;
    contract.inputs.get_mut("state").expect("state").ty.shape = width;
    (
        contract.action_dim,
        contract.horizon,
        contract.execute_chunk,
    ) = (20, CHUNK, EXECUTE);
    contract.replanning_hz = (CONTROL_HZ / u64::from(EXECUTE)) as f32;

    // The teacher's unnormalizer: the head works in about [-1, 1], the chunk leaves in radians.
    let executed = PortType {
        shape: Shape::new([u64::from(EXECUTE), 20]),
        ..g.outputs[0].ty.clone()
    };
    let norm = NodeId(g.nodes.nodes.keys().map(|n| n.0).max().expect("nodes") + 1);
    let range = ctrlrange(&scene);
    g.nodes.insert(
        norm,
        LearningNode::Normalizer {
            inputs: vec![Port::new("actions", executed.clone())],
            direction: NormalizeDir::Inverse,
            stats: StatsSource::MeanStd {
                mean: range.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect(),
                std: range.iter().map(|(lo, hi)| (hi - lo) / 2.0).collect(),
            },
            out_unit: Unit::Angle,
        },
    );
    g.nodes.connect(chunker, "actions", norm, "actions");
    g.nodes.outputs = vec![PortRef::new(norm, "out")];
    g.outputs = vec![Port::new(
        "actions",
        PortType {
            unit: Unit::Angle,
            ..executed
        },
    )];
    let diags = g.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    g
}

/// deployment-hand.toml executing the student's chunks: 16 / 6 under a temporal ensemble,
/// asked for a chunk every 6 control ticks.
fn deployment_student() -> DeploymentIr {
    let (scene, _) = scene();
    let mut dep = deployment();
    dep.action.horizon = CHUNK as usize;
    dep.action.execute_chunk = EXECUTE as usize;
    dep.execution = es_ir::deployment::ExecutionMode::TemporalEnsemble { decay: 0.01 };
    let c = control_rate(&scene);
    dep.rate.inference =
        TickRate::rational(c.num(), c.den() * u64::from(EXECUTE)).expect("non-zero");
    let diags = dep.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    dep
}

/// evaluation-teacher.toml on [`SEEDS_64`].
fn evaluation_64(task: &TaskIr, obs: &ObservationIr) -> EvaluationIr {
    let mut ev = evaluation(task, obs);
    ev.episodes = EpisodeBatch {
        n_episodes: 64,
        seeds: SeedPlan::Explicit(SEEDS_64.collect()),
    };
    let diags = ev.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    ev
}

/// [`evaluation_64`] for the student with evaluation-augmented.toml's six suites, the
/// observation delay moved onto 60 Hz ticks.
fn evaluation_student(task: &TaskIr, obs: &ObservationIr) -> EvaluationIr {
    use es_ir::evaluation::PerturbationKind;
    let mut ev = evaluation_64(task, obs);
    let so101 = es_ir::serial::evaluation_from_toml(&read(&vl("evaluation-augmented.toml")))
        .expect("evaluation-augmented.toml");
    ev.suites = so101.suites;
    for p in ev.suites.iter_mut().flat_map(|s| &mut s.perturbations) {
        if let PerturbationKind::ObservationDelay { ms } = &mut p.kind {
            *ms = vec![17, 34];
        }
    }
    let diags = ev.validate();
    assert!(diags.is_empty(), "{diags:#?}");
    ev
}

// --- the generator ------------------------------------------------------------------------------

const GENERATED: &str = "`ES_GENERATE_GOLDENS=1 cargo test -p es --test shadow_hand -- --ignored \
                         generate_shadow_hand_documents` (crates/es/tests/shadow_hand.rs)";

const TASK_HEADER: &str = "\
# Task IR (spec 6) for the Shadow Hand cube reorientation -- plan H, packet M16/H2
# (docs/packets/M16/plan-h.md). Isaac Lab's repose-cube task (`ShadowHandEnv`) in
# tests/fixtures/mjcf/shadow_hand/shadow_hand_repose.xml, simplified where noted.
#
# Generated by GENERATED
# from the scene: `scene_hash` (the twelve STLs included, as `load_scene` reads them),
# `asset_hash`, the joint ranges, the tendon couplings and the ctrlranges are read from it.
#
# CONTROL 60 Hz (two physics ticks of 1/120 s), `JointPosition` over the 20 servos, 480 steps
# (8 s). Isaac Lab runs 10 s and draws a new goal each time one is reached; this document ends
# the episode on the first success (a Task IR has no mid-episode redraw).
#
# CHANNELS (spec 7.4):
#  * joint_pos[24], joint_vel[24] -- qpos[0..24] and qvel[0..24], the hand's 24 joints in
#    MuJoCo's order (joint_vel names WRJ1, the first; joint_pos the hand mount, which has no
#    joint, so it reads the leading 24 -- the reach task's two bindings).
#  * cube_pose[7], goal_pose[7] -- `BodyPose`: xpos || xquat (x y z w), the backend's
#    normalized orientation. SIMULATOR-PRIVILEGED (cube_pose): the teacher reads it, the
#    student (H3) does not. goal_pose's position is constant (the goal cube floats beside the
#    hand where the cameras see it); its orientation is the goal. A 4-wide goal_quat is not a
#    source an ObsSource can name (a channel starts at a joint's first qpos), so the goal is
#    the whole pose.
#  * object_vel[6] -- the cube's free joint's qvel: linear (world frame), angular (body frame,
#    MuJoCo's convention). Not `cube_vel`: the cross-IR width check (DEP-031) reads the robot's
#    joint count off the first JointState channel by name, and it must be joint_pos (24), not
#    this one (6).
#  * last_action[20] -- the previous policy row in actuator units; `initial` is each
#    ctrlrange's centre, which is Isaac Lab's zero raw action.
#  * rgb_top, rgb_front, rgb_side -- 96x96 Rgb8 on the path tracer: 32 spp, 3 bounces, no
#    SVGF, `seed = \"tick\"` (the X7 rerun's), exposure 8. EXPOSURE: at the X7 rerun's 64 the
#    white hand saturates (3-6 % of each camera's pixels at 255); at 8 none do and the hand
#    keeps its shading (chosen by rendering the three cameras at 64, 32, 16, 8, 4).
#    SAMPLES (packet M16/H3, `render_quality_sweep`): the three cameras at the reset pose and
#    three mid-episode poses, 4/8/16/32/64 spp x 2/3 bounces x SVGF off/on, against 256 spp.
#    The rule: the cheapest setting whose cube keeps block.png's letters on every visible face
#    and whose speckle -- pixels under half the reference's luminance -- is at most 5 % on
#    every camera. 4 spp (H2's) left 17-37 % speckle; SVGF removes it at any spp but also the
#    letters (single-frame SVGF smooths the textured radiance), so it fails the first clause.
#    32 spp / 3 bounces: speckle 0.2 / 3.2 / 4.8 % (top / front / side), 89 ms per camera on
#    an RTX 3060 shared with a PPO run (4 spp: 27 ms); 32 / 2 bounces left 7.4 % on `side`.
#
# REWARD (Isaac Lab's terms times 0.01, rl_games' `scale_value`), with p_ref = (1.0, 0.867,
# 0.1772) -- where the cube rests on the palm, measured -- and d = |q_cube . q_goal|:
#  * cube_distance  -0.1 * ||p_cube - p_ref||. The offset is a per-lane Normalize over
#    [p_ref - 1, p_ref + 1] -> [-1, 1] (the identity shifted by p_ref: the IR has no constant).
#  * rotation_0..7 + rotation_floor  0.01 * 1 / (s + 0.1), s = sqrt(8 (1 - d)) -- the
#    small-angle form of Isaac Lab's 1 / (rot_dist + 0.1). A Reward cannot take a reciprocal
#    (Arith Div needs a Dimensionless divisor and nothing derived from a quaternion is one),
#    so the curve is written as its piecewise-linear interpolant: one clamped ramp per knot
#    interval, s = 0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, sqrt(8), exact at every knot
#    (between them at most 0.11 of 10 above the curve, on [0, 0.025]), plus the floor
#    1 / (sqrt(8) + 0.1) paid on every step (`time since reset >= 0`).
#  * success  +2.5 when d >= cos(0.05), i.e. the rotation to the goal is under 0.1 rad --
#    exactly Isaac Lab's `rot_dist <= success_tolerance`.
#  * Isaac Lab's action penalty -0.0002 * ||a||^2 is NOT here: no Task IR node reads the
#    action (ObsSource::PreviousAction serves an observation, not a cone). At 0.01 scale it is
#    at most 4e-5 per step.
# TERMINATION: success (above); failure when ||p_cube - p_ref|| > 0.24 (Isaac Lab's
# fall_dist); timeout at 480 control steps (`max_episode_steps`; the `time >= 8 s` node is
# the same bound, reached one step later on the runtime's clock -- 960 ticks of 8,333,333 ns
# is 7.99999968 s, deployment-hand.toml's header).
#
# RESET (ResetState, drawn at every reset):
#  * every hand joint Uniform over 0.2 x its range (Isaac Lab's reset_dof_pos_noise; every
#    range contains 0, the default pose). The four tendon-coupled J0s draw from their J1's
#    stream with bounds scaled by k = 0.00805 / 0.00705, so one draw sets both and the tendon
#    starts at length 0: two nodes on one stream share its draw (es_env::rng addresses a draw
#    by (seed, env, episode, stream)).
#  * the cube at p_ref, x and y within +-0.01 m (Isaac Lab's reset_position_noise).
#  * cube and goal orientation: the resting tilt q_x(alpha) (measured 3.7-4.1 deg about world
#    +X, the palm's slope; tan(alpha / 2) = 0.0355) composed with a yaw about the palm normal
#    (world +Z, measured to 5e-6 rad): q = (1, 0.0355, -0.0355 u, u) with one u ~ U(-1, 1) per
#    cube (`cube.yaw`, `goal.yaw`), the yaw 2 atan(u) in [-90, 90] deg. Unnormalized on purpose:
#    each lane is linear in u, so four nodes on one stream write it from one draw; MuJoCo and
#    MJWarp both normalize a free joint's quaternion (in xquat at once, in qpos after the first
#    step, measured) and this document reads xquat only. Isaac Lab draws both from all of SO(3);
#    yaw goals are plan H's reduced set.
";

const OBSERVATION_HEADER: &str = "\
# Observation IR (spec 7) for the Shadow Hand teacher -- plan H, packet M16/H2. State only.
#
# Generated by GENERATED;
# `task_ref` is task-repose.toml's `task_hash`.
#
# Six StateInputs -> Concat -> one Normalize { MeanStd }, in the layout order
# joint_pos[24] || joint_vel[24] || cube_pose[7] || object_vel[6] || goal_pose[7] ||
# last_action[20] = 88, each element brought to about [-1, 1] by static statistics:
#   joint_pos    centre / half-range of each joint's range (from the scene)
#   joint_vel    0 / 5 rad/s (Isaac Lab's 0.2 velocity scale)
#   cube_pose    p_ref / 0.24 m (+-1 at the drop radius); the quaternion 0 / 1
#   object_vel   0 / 1 m/s linear, 0 / 5 rad/s angular
#   goal_pose    the goal cube's scene position / 1 (a constant: 0); the quaternion 0 / 1
#   last_action  centre / half of each ctrlrange (the policy's own [-1, 1])
# The three camera channels task-repose.toml declares are not read: the teacher is a state
# policy, so nothing is rendered while it trains (es_native.Rollout renders an image input only).
";

const LEARNING_HEADER: &str = "\
# Learning IR (spec 8) for the Shadow Hand PPO teacher -- plan H, packet M16/H2.
#
# Generated by GENERATED;
# the unnormalizer's mean and std are the scene's ctrlranges, never typed in.
#
#   StateEncoder{Mlp [512, 256] -> 128, ELU, last layer activated}
#     -> PolicyHead{Regression, tanh, 1 x 20} -> ActionChunker -> Normalizer{Inverse}
#
# rl_games' Shadow Hand actor shape (ELU, 512 / 256 / 128) on one 88-wide state port; the
# output is in actuator units (mean = ctrlrange centre, std = half-range), which is what
# es_native.Rollout::act takes. The value network and log_std are training-only state
# (learning-reach.toml's header).
";

const DEPLOYMENT_HEADER: &str = "\
# Deployment IR + Safety Plane (spec 9) for the Shadow Hand -- plan H, packet M16/H2.
#
# Generated by GENERATED
# from deployment-reach.toml (one action per control tick, receding horizon, the watchdogs,
# `envelope_violation_rate.max_frac = 1.0` and hold_position -- its header argues each) with the
# hand's own envelope, read from the scene:
#  * position = each servo's ctrlrange, soft margin 0: a tanh policy's mean is always inside,
#    and the plane clamps a Gaussian sample that lands outside.
#  * velocity_max = 2 x range x 60 Hz, acceleration_max = 4 x range x 60^2, action_rate first
#    difference 2 x range, second 4 x range: twice the widest jump a target inside the range
#    can make in one tick, so no rate stage ever binds and the position stage is the envelope.
#    M9/X7 measured the reach task clamped on every tick (`executed_ne_sampled_rate` 1.00) under
#    a physical rate envelope; a policy commanding absolute targets at 60 Hz jumps. What bounds
#    the hand's real motion is the servos (kp 1-5, forcerange). THESE NUMBERS ARE FOR THE
#    OWNER'S REVIEW: a physical Shadow Hand's joint-speed limits would be the other choice.
#  * torque_max = each servo's forcerange (not applied: the action is a position).
#  * the workspace box is not applied in joint space (spec 9.3).
#  * rate.control = rate.inference = 500000000 / 8333333 Hz (60.0000024 Hz): exactly two
#    physics ticks. Every runtime path reads the scene's timestep 1/120 s rounded to whole
#    nanoseconds (TickRate::from_period_secs: 8,333,333 ns, 120.0000048 Hz), against which
#    60 Hz is 2.00000008 ticks and BatchDomains::single_env_at refuses it.
";

const EVALUATION_HEADER: &str = "\
# Evaluation IR (spec 10) for the Shadow Hand teacher -- plan H, packet M16/H2.
#
# Generated by GENERATED
# from evaluation-reach.toml: `task` and `observation` are task-repose.toml's and
# observation-teacher.toml's hashes; the nominal suite only; 16 held-out seeds 301-316; metrics
# success_rate, episode_length, envelope_violation_rate, failure_mode_histogram. Run it with
# `--backend mujoco-cpu` (the reference) whatever backend trained the policy.
#
# ACCEPTANCE IS A PLACEHOLDER (success_rate >= 0.5 on nominal): the orchestrator sets it from
# E1's stop rule before the run.
";

const EVALUATION_64_HEADER: &str = "\
# Evaluation IR (spec 10) for the Shadow Hand teacher on 64 held-out seeds -- plan H, packet
# M16/H3.
#
# Generated by GENERATED;
# evaluation-teacher.toml with 64 episodes on seeds 301-364 in place of 16 on 301-316 (its first
# 16 are those), so a checkpoint comparison is not 16-episode noise: one episode is 1.6 points
# of success rate here, 6.25 there. Everything else -- `task`, `observation`, the nominal suite,
# the metrics, the placeholder acceptance -- is evaluation-teacher.toml's. evaluation-student.toml
# uses the same seeds, so teacher and student are judged on the same episodes.
";

const OBSERVATION_STUDENT_HEADER: &str = "\
# Observation IR (spec 7) for the Shadow Hand student -- plan H, packet M16/H3. Three cameras
# and the hand's own state; NO cube pose.
#
# Generated by GENERATED
# from ../visible-learning/observation-views.toml (plan N's three-view arm, M7/U3's chain per
# camera) and task-repose.toml:
#  * the three image chains, byte for byte plan N's -- ImageInput (U8 HWC) -> Dequantize ->
#    Normalize{0..1} -> Pad{4} -> Crop{Random 96x96} -> Augment{ColorJitter 0.2 / 0.2,
#    training only} -- with the cameras moved onto this scene's: rgb_overhead -> rgb_top,
#    rgb_wrist -> rgb_front, rgb_side -> rgb_side (sensor, frame and time reference), fx = fy
#    from this scene's fovy 45 deg (all three), and rate_hz 50 -> 60 (the control rate).
#    observation-augmented.toml's header says what the chain does in training and evaluation.
#  * `joint_state` replaced by `state` (31): joint_pos[24] || goal_pose[7] -> Concat ->
#    Normalize{MeanStd} with observation-teacher.toml's statistics for those two channels.
#    The goal is the task's input (which way to turn the cube), not privileged: a real hand
#    is told its goal. What the teacher also read and the student does not: joint_vel,
#    cube_pose (SIMULATOR-PRIVILEGED), object_vel and last_action.
# goal_pose is the goal cube's free-joint pose; a recorded row carries qpos, so the bake and
# the evaluator read it from the free joint's qpos range with the loaded model
# (es_eval::runner::input_sources).
";

const LEARNING_STUDENT_HEADER: &str = "\
# Learning IR (spec 8) for the Shadow Hand student -- plan H, packet M16/H3.
#
# Generated by GENERATED
# from ../visible-learning/learning-views.toml (plan N's three-view ACT-shaped graph):
#
#   rgb_top   -> VisionEncoder{ResNet18, ImageNet} 0 -.
#   rgb_front -> VisionEncoder{ResNet18, ImageNet} 6 -+
#   rgb_side  -> VisionEncoder{ResNet18, ImageNet} 7 -+- Fusion{Concat 4x512 -> 512} 2
#   state     -> StateEncoder{Mlp[256]} 1            -'   -> TemporalEncoder -> PolicyHead
#                                                         -> ActionChunker -> Normalizer{Inverse}
#
# What differs from learning-views.toml: the inputs renamed (rgb_top / rgb_front / rgb_side,
# `state` 31 wide) and the fusion's ports with them (top, state, front, side); action_dim 20;
# the unnormalizer learning-teacher.toml ends with (mean = each ctrlrange's centre, std = its
# half-range), so the head works in about [-1, 1] and the chunk leaves in radians -- the hand's
# servos span 0.4-1.6 rad where SO-101's six were alike.
#
# WHY CONCAT, NOT learning-mad.toml's shared encoder + Sum: MAD's gain is deploying on fewer
# cameras without retraining, and this student is deployed with the three it is trained on.
# On plan N's E1/E2 (NV-verification.md) the Concat graph scored best -- 7/16 nominal and
# 21/96 on one seed -- and MAD with three cameras 4/16 and 12/96, inside the seed spread
# but not above it; three encoders also let each camera's features differ (the three views
# here are all world-fixed but see different faces). The MAD graph stays one derivation away.
#
# WHY 16 / 6 AT 10 Hz: plan N kept SO-101's 16 / 10 at 50 Hz, replanning every 200 ms. The
# hand's contacts change within a few ticks and its teacher acts every tick, so the student
# replans every 6 ticks (100 ms, 10 Hz: an integer relation to 60 Hz, XIR-023) and keeps a
# 16-row chunk (267 ms) for the temporal ensemble to average over, as ACT does. Three
# ResNet18s at batch 1 fit the 100 ms between inferences (plan N declared 15 ms expected,
# 40 ms deadline; kept).
";

const DEPLOYMENT_STUDENT_HEADER: &str = "\
# Deployment IR + Safety Plane (spec 9) for the Shadow Hand student -- plan H, packet M16/H3.
#
# Generated by GENERATED
# from deployment-hand.toml (the teacher's): the same envelope, watchdogs, fallback and control
# rate, with the student's chunk execution -- action horizon 16, execute_chunk 6, a temporal
# ensemble (decay 0.01, ../visible-learning/deployment.toml's) and rate.inference = the control
# rate / 6 (500000000 / 49999998 Hz, 10.0000004 Hz: a chunk every six control ticks, exactly).
# deployment-hand.toml's header argues every envelope number, and they are still for the
# owner's review.
";

const EVALUATION_STUDENT_HEADER: &str = "\
# Evaluation IR (spec 10) for the Shadow Hand student -- plan H, packet M16/H3.
#
# Generated by GENERATED;
# `task` and `observation` are task-repose.toml's and observation-student.toml's hashes; 64
# held-out seeds 301-364 (evaluation-teacher-64.toml's, so the teacher and the student play the
# same episodes; the student's collection draws from 1001 on). Metrics as the teacher's.
# The suites are ../visible-learning/evaluation-augmented.toml's six (the SO-101 student's), each
# on its own stream:
#   nominal; light_intensity 0.5-1.5 and light_direction +-45 deg (the three cameras render
#   under them, so the collection and this evaluation need frames); observation_delay 17 or
#   34 ms -- one or two 60 Hz ticks (SO-101's 20 / 40 ms were one or two 50 Hz ticks; the
#   runner floors ms to ticks); torque_noise 5 %; backlash 0-0.01 rad.
# None is left out: every one is a property of the cameras or of the actuators, and none
# assumes a moving base.
# ACCEPTANCE IS A PLACEHOLDER (success_rate >= 0.5 on nominal): the orchestrator sets it from
# the teacher's own score on evaluation-teacher-64.toml -- the student imitates the teacher's
# successes and is not expected to pass it.
";

/// Every generated document as `(file, header, body)`.
fn documents() -> Vec<(&'static str, String, String)> {
    use es_ir::serial::{
        deployment_to_toml, evaluation_to_toml, learning_to_toml, observation_to_toml, task_to_toml,
    };
    let task = task();
    let obs = observation(&task);
    let student = observation_student(&task);
    let header = |h: &str| h.replace("GENERATED", GENERATED);
    vec![
        (
            "task-repose.toml",
            header(TASK_HEADER),
            task_to_toml(&task).expect("toml"),
        ),
        (
            "observation-teacher.toml",
            header(OBSERVATION_HEADER),
            observation_to_toml(&obs).expect("toml"),
        ),
        (
            "learning-teacher.toml",
            header(LEARNING_HEADER),
            learning_to_toml(&learning(&task)).expect("toml"),
        ),
        (
            "deployment-hand.toml",
            header(DEPLOYMENT_HEADER),
            deployment_to_toml(&deployment()).expect("toml"),
        ),
        (
            "evaluation-teacher.toml",
            header(EVALUATION_HEADER),
            evaluation_to_toml(&evaluation(&task, &obs)).expect("toml"),
        ),
        (
            "evaluation-teacher-64.toml",
            header(EVALUATION_64_HEADER),
            evaluation_to_toml(&evaluation_64(&task, &obs)).expect("toml"),
        ),
        (
            "observation-student.toml",
            header(OBSERVATION_STUDENT_HEADER),
            observation_to_toml(&student).expect("toml"),
        ),
        (
            "learning-student.toml",
            header(LEARNING_STUDENT_HEADER),
            learning_to_toml(&learning_student(&task)).expect("toml"),
        ),
        (
            "deployment-student.toml",
            header(DEPLOYMENT_STUDENT_HEADER),
            deployment_to_toml(&deployment_student()).expect("toml"),
        ),
        (
            "evaluation-student.toml",
            header(EVALUATION_STUDENT_HEADER),
            evaluation_to_toml(&evaluation_student(&task, &student)).expect("toml"),
        ),
    ]
}

/// Writes every IR document of `documents`. Run explicitly:
///
///     ES_GENERATE_GOLDENS=1 cargo test -p es --test shadow_hand -- --ignored generate_shadow_hand_documents
#[test]
#[ignore = "fixture generator; run explicitly"]
fn generate_shadow_hand_documents() {
    // Spec 1.4: a generator must refuse to run by accident (M7 review).
    if std::env::var("ES_GENERATE_GOLDENS").as_deref() != Ok("1") {
        println!("SKIP generate_shadow_hand_documents: set ES_GENERATE_GOLDENS=1 to regenerate");
        return;
    }
    std::fs::create_dir_all(fixture(".")).expect("tests/fixtures/shadow-hand");
    for (name, header, body) in documents() {
        std::fs::write(fixture(name), format!("{header}\n{body}")).expect("write");
        println!("wrote {}", fixture(name).display());
    }
}

// --- the documents ------------------------------------------------------------------------------

fn body_of(text: &str) -> String {
    text.split_once("\nes_schema")
        .map_or_else(|| text.to_owned(), |(_, rest)| format!("es_schema{rest}"))
}

/// The committed documents are exactly what the generator writes, validate, and agree across
/// every boundary the cross-IR pass sees; the policy reads 88 numbers and emits 20.
#[test]
fn shadow_hand_documents_are_what_the_generator_writes() {
    for (name, _, body) in documents() {
        assert_eq!(
            body_of(&read(&fixture(name))),
            body,
            "{name} is not what the generator writes; rerun {GENERATED}"
        );
    }
    let task = es_ir::serial::task_from_toml(&read(&fixture("task-repose.toml"))).expect("task");
    let obs = es_ir::serial::observation_from_toml(&read(&fixture("observation-teacher.toml")))
        .expect("observation");
    let learning = es_ir::serial::learning_from_toml(&read(&fixture("learning-teacher.toml")))
        .expect("learning");
    let deployment = es_ir::serial::deployment_from_toml(&read(&fixture("deployment-hand.toml")))
        .expect("deployment");
    let evaluation =
        es_ir::serial::evaluation_from_toml(&read(&fixture("evaluation-teacher.toml")))
            .expect("evaluation");
    let diags = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &task,
        observation: &obs,
        learning: &learning,
        deployment: &deployment,
        evaluation: Some(&evaluation),
    });
    assert!(diags.is_empty(), "cross-IR: {diags:#?}");
    assert_eq!(obs.outputs["state"].ty.shape.dims(), [88]);
    assert_eq!(learning.policy.contract.action_dim, 20);
    assert_eq!(deployment.robot.n_joints, 20);
    // The scene the documents pin is the file on disk, meshes included.
    let (scene, xml) = scene();
    assert_eq!(task.scene.scene_hash, scene.scene_hash());
    assert_eq!(task.scene.asset_hash, *blake3::hash(&xml).as_bytes());
    println!(
        "RAN shadow_hand_documents_are_what_the_generator_writes: task {} observation {} \
         learning {} deployment {} evaluation {}",
        hex(&task.task_hash().expect("hash")),
        hex(&obs.observation_hash().expect("hash")),
        hex(&learning.learning_hash().expect("hash")),
        hex(&deployment.deployment_hash().expect("hash")),
        hex(&evaluation.evaluation_hash().expect("hash")),
    );
}

fn es(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_es"))
        .current_dir(repo())
        .args(args)
        .output()
        .expect("run es");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

const DOCS: [&str; 5] = [
    "tests/fixtures/shadow-hand/task-repose.toml",
    "tests/fixtures/shadow-hand/observation-teacher.toml",
    "tests/fixtures/shadow-hand/learning-teacher.toml",
    "tests/fixtures/shadow-hand/deployment-hand.toml",
    "tests/fixtures/shadow-hand/evaluation-teacher.toml",
];

/// `es ir validate`, `es ir check` (the XIR pass, spec 11.1) and `es task compile` accept the
/// five documents; the compiled plan reads the 88-wide state.
#[test]
fn the_documents_validate_check_and_compile() {
    for verb in ["validate", "check"] {
        let (ok, text) = es(&[&["ir", verb][..], &DOCS].concat());
        assert!(ok && !text.contains("ERROR"), "es ir {verb}\n{text}");
    }
    let (ok, text) = es(&["task", "compile", DOCS[0], DOCS[1]]);
    assert!(ok, "{text}");
    assert!(text.contains("shape=[88]"), "{text}");
    println!("RAN the_documents_validate_check_and_compile");
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("es-shadow-hand-{tag}-{nanos}"))
}

/// `es policy init` builds the untrained teacher bundle from the four documents; it opens
/// (every IR re-validated, XIR-010), passes XIR-040 against the Evaluation IR, and the scene it
/// names -- STLs and all -- loads to the `scene_hash` it pins.
#[test]
fn policy_init_builds_the_teacher_bundle() {
    let out = scratch("init").join("teacher.esb");
    let out_s = out.to_string_lossy().into_owned();
    let (ok, text) = es(&[
        "policy",
        "init",
        "--task",
        DOCS[0],
        "--observation",
        DOCS[1],
        "--learning",
        DOCS[2],
        "--deployment",
        DOCS[3],
        "--out",
        &out_s,
    ]);
    assert!(ok, "{text}");
    let bundle =
        es_compile::PolicyBundle::open(&std::fs::read(&out).expect("written")).expect("opens");
    let evaluation =
        es_ir::serial::evaluation_from_toml(&read(&repo().join(DOCS[4]))).expect("evaluation");
    let errors: Vec<_> = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &bundle.task,
        observation: &bundle.observation,
        learning: &bundle.learning,
        deployment: &bundle.deployment,
        evaluation: Some(&evaluation),
    })
    .into_iter()
    .filter(es_ir::diag::Diagnostic::is_error)
    .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    // A bundle carries documents, not the scene: `scene.path` is repository-relative and the
    // runtime reads the XML and its `meshes/` from there (`es_tools::backend::load_scene`).
    let loaded = es_tools::backend::load_scene(&repo().join(SCENE).to_string_lossy())
        .expect("the scene and its STLs load");
    assert_eq!(loaded.meshes.len(), 12, "the twelve STLs");
    assert_eq!(bundle.task.scene.scene_hash, loaded.scene_hash());
    println!("RAN policy_init_builds_the_teacher_bundle: {text}");
}

/// The training recipe parses and `es train --dry-run` prints its plan: PPO on `mjwarp`.
#[test]
fn the_training_recipe_dry_runs() {
    let recipe = "tests/fixtures/shadow-hand/training-teacher.toml";
    let out = scratch("dry-run");
    let out = out.to_string_lossy();
    let (ok, text) = es(&["train", "--recipe", recipe, "--out", &out, "--dry-run"]);
    assert!(ok, "{text}");
    assert!(text.contains("train_ppo"), "{text}");
    assert!(text.contains("mjwarp"), "{text}");
    println!("RAN the_training_recipe_dry_runs:\n{text}");
}

// --- the student (packet M16/H3) ----------------------------------------------------------------

const STUDENT_DOCS: [&str; 5] = [
    "tests/fixtures/shadow-hand/task-repose.toml",
    "tests/fixtures/shadow-hand/observation-student.toml",
    "tests/fixtures/shadow-hand/learning-student.toml",
    "tests/fixtures/shadow-hand/deployment-student.toml",
    "tests/fixtures/shadow-hand/evaluation-student.toml",
];

/// The student's documents agree across every boundary the cross-IR pass sees (and so do the
/// teacher's under the 64-seed evaluation): three camera ports and a 31-wide state that holds
/// no cube, a 20-wide action chunked 16 / 6, and `es ir validate` / `es ir check` accept them.
#[test]
fn the_student_documents_agree_and_check() {
    let doc = |i: usize| read(&repo().join(STUDENT_DOCS[i]));
    let task = es_ir::serial::task_from_toml(&doc(0)).expect("task");
    let obs = es_ir::serial::observation_from_toml(&doc(1)).expect("observation");
    let learning = es_ir::serial::learning_from_toml(&doc(2)).expect("learning");
    let deployment = es_ir::serial::deployment_from_toml(&doc(3)).expect("deployment");
    let evaluation = es_ir::serial::evaluation_from_toml(&doc(4)).expect("evaluation");
    let diags = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &task,
        observation: &obs,
        learning: &learning,
        deployment: &deployment,
        evaluation: Some(&evaluation),
    });
    assert!(diags.is_empty(), "cross-IR: {diags:#?}");
    let ports: Vec<&str> = obs.outputs.keys().map(String::as_str).collect();
    assert_eq!(ports, ["rgb_front", "rgb_side", "rgb_top", "state"]);
    assert_eq!(obs.outputs["state"].ty.shape.dims(), [31]);
    let read_by_state: Vec<StableId> = obs
        .graph
        .nodes
        .values()
        .filter_map(|n| match n {
            ObservationNode::StateInput { source, .. } => Some(*source),
            _ => None,
        })
        .collect();
    let cube = body(&scene().0, "object");
    assert!(
        !read_by_state.contains(&cube),
        "the student reads no cube pose"
    );
    assert_eq!(read_by_state.len(), 2);
    let c = &learning.policy.contract;
    assert_eq!((c.action_dim, c.horizon, c.execute_chunk), (20, 16, 6));
    assert_eq!(evaluation.episodes.n_episodes, 64);
    assert_eq!(evaluation.suites.len(), 6);

    let teacher = |f: &str| read(&fixture(f));
    let diags = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &task,
        observation: &es_ir::serial::observation_from_toml(&teacher("observation-teacher.toml"))
            .expect("observation"),
        learning: &es_ir::serial::learning_from_toml(&teacher("learning-teacher.toml"))
            .expect("learning"),
        deployment: &es_ir::serial::deployment_from_toml(&teacher("deployment-hand.toml"))
            .expect("deployment"),
        evaluation: Some(
            &es_ir::serial::evaluation_from_toml(&teacher("evaluation-teacher-64.toml"))
                .expect("evaluation"),
        ),
    });
    assert!(
        diags.is_empty(),
        "cross-IR, teacher on 64 seeds: {diags:#?}"
    );

    for verb in ["validate", "check"] {
        let (ok, text) = es(&[&["ir", verb][..], &STUDENT_DOCS].concat());
        assert!(ok && !text.contains("ERROR"), "es ir {verb}\n{text}");
    }
    println!("RAN the_student_documents_agree_and_check");
}

/// `es policy init` builds the untrained student bundle, which opens and passes XIR-040
/// against the student's Evaluation IR.
#[test]
fn policy_init_builds_the_student_bundle() {
    let out = scratch("student-init").join("student.esb");
    let out_s = out.to_string_lossy().into_owned();
    let (ok, text) = es(&[
        "policy",
        "init",
        "--task",
        STUDENT_DOCS[0],
        "--observation",
        STUDENT_DOCS[1],
        "--learning",
        STUDENT_DOCS[2],
        "--deployment",
        STUDENT_DOCS[3],
        "--out",
        &out_s,
    ]);
    assert!(ok, "{text}");
    let bundle =
        es_compile::PolicyBundle::open(&std::fs::read(&out).expect("written")).expect("opens");
    let evaluation =
        es_ir::serial::evaluation_from_toml(&read(&repo().join(STUDENT_DOCS[4]))).expect("eval");
    let errors: Vec<_> = es_ir::cross::check(&es_ir::cross::IrBundle {
        task: &bundle.task,
        observation: &bundle.observation,
        learning: &bundle.learning,
        deployment: &bundle.deployment,
        evaluation: Some(&evaluation),
    })
    .into_iter()
    .filter(es_ir::diag::Diagnostic::is_error)
    .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    println!("RAN policy_init_builds_the_student_bundle: {text}");
}

/// `es train --dry-run` on the student's recipe (the IR route, ACT over the baked set) and `es
/// loop cycle --dry-run` on its cycle: collect with the teacher, keep the successes, train,
/// evaluate.
#[test]
fn the_student_recipe_and_cycle_dry_run() {
    let out = scratch("student-dry-run");
    let out = out.to_string_lossy();
    let (ok, text) = es(&[
        "train",
        "--recipe",
        "tests/fixtures/shadow-hand/training-student.toml",
        "--out",
        &out,
        "--dry-run",
    ]);
    assert!(ok, "{text}");
    assert!(text.contains("train_act"), "{text}");
    let (ok, text) = es(&[
        "loop",
        "cycle",
        "--recipe",
        "tests/fixtures/shadow-hand/cycle-student.toml",
        "--out",
        &out,
        "--dry-run",
    ]);
    assert!(ok, "{text}");
    for want in [
        "--policy runs/shadow-hand/teacher.esb",
        "--success-only",
        "collect/successes/frames",
        "evaluation-student.toml",
    ] {
        assert!(text.contains(want), "{want:?} not in\n{text}");
    }
    assert!(!text.contains("--expert"), "{text}");
    println!("RAN the_student_recipe_and_cycle_dry_run:\n{text}");
}

/// The student's Observation IR bakes a recorded row with the loaded hand: `joint_pos` is the
/// row's leading 24 and `goal_pose` the goal cube's free joint in `qpos` (a recorded row has no
/// `xpos` / `xquat`), so `es train` can bake what the teacher collects. Needs `ES_PYTHON`
/// (`MuJoCo`, for the model).
#[test]
fn the_student_state_bakes_from_a_recorded_row() {
    use es_physics_backend::MuJoCoCpuBackend;
    use es_physics_core::backend::{LoadConfig, PhysicsBackend};

    if std::env::var_os("ES_PYTHON").is_none() {
        println!("SKIP the_student_state_bakes_from_a_recorded_row: ES_PYTHON is not set");
        return;
    }
    let task = es_ir::serial::task_from_toml(&read(&repo().join(STUDENT_DOCS[0]))).expect("task");
    let obs = es_ir::serial::observation_from_toml(&read(&repo().join(STUDENT_DOCS[1])))
        .expect("observation");
    let (scene, _) = scene();
    let mut backend = MuJoCoCpuBackend::new();
    let model = backend
        .load(&scene, &LoadConfig::default())
        .expect("the hand loads");
    es_eval::ObservationBake::new(&obs, &task, None)
        .expect_err("goal_pose needs the model's qpos range");
    es_eval::ObservationBake::new(&obs, &task, Some(&model))
        .expect("the student's inputs resolve against the loaded hand");
    println!("RAN the_student_state_bakes_from_a_recorded_row");
}

// --- the task on scripted states (MuJoCo) --------------------------------------------------------

/// What the Task IR should pay for one post-step state, from `xpos` / `xquat` computed here.
fn expected_reward(cube_pos: &[f64], cube_q: &[f64], goal_q: &[f64]) -> (f64, bool, bool) {
    let dist = (0..3)
        .map(|i| (cube_pos[i] - P_REF[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let d = (0..4).map(|i| cube_q[i] * goal_q[i]).sum::<f64>().abs();
    let s = (8.0 * (1.0 - d)).max(0.0).sqrt();
    let f = rotation_at_knots();
    // The piecewise-linear interpolant of 1 / (s + 0.1) at the knots.
    let k = KNOTS
        .windows(2)
        .position(|w| s <= w[1])
        .unwrap_or(KNOTS.len() - 2);
    let t = ((s - KNOTS[k]) / (KNOTS[k + 1] - KNOTS[k])).clamp(0.0, 1.0);
    let rot = f[k] + (f[k + 1] - f[k]) * t;
    let success = d >= COS_TOLERANCE;
    let reward = -10.0 * SCALE * dist.min(1.0) + SCALE * rot + if success { 2.5 } else { 0.0 };
    (reward, success, dist > FALL_M)
}

/// The Task IR on states set by hand, through `es_env::Env` on `mujoco-cpu`: the reset lays
/// the hand, the cube and the goal down as the document says, and the reward, the success and
/// the drop come out of the lowered cones as the header's formulas compute them from the state.
/// Needs `ES_PYTHON` (`MuJoCo`); prints SKIP without.
#[test]
#[allow(clippy::too_many_lines)]
fn the_task_scores_scripted_states() {
    use es_env::{scheduler::BatchDomains, Env, Termination};
    use es_physics_backend::MuJoCoCpuBackend;
    use es_physics_core::backend::{PhysicsBackend, StateView};

    if std::env::var_os("ES_PYTHON").is_none() {
        println!("SKIP the_task_scores_scripted_states: ES_PYTHON is not set (MuJoCo)");
        return;
    }
    let task = es_ir::serial::task_from_toml(&read(&fixture("task-repose.toml"))).expect("task");
    let (scene, _) = scene();
    let domains = BatchDomains::single_env_at(
        TickRate::from_period_secs(scene.options.timestep).expect("120 Hz"),
        control_rate(&scene),
    )
    .expect("two ticks per control step");
    let mut env = Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, 7)
        .expect("the task compiles against the loaded hand");
    let model = env.model().clone();
    let row = |id: StableId| model.body[&id].start as usize;
    let (object, target) = (row(body(&scene, "object")), row(body(&scene, "target")));

    // --- the reset -----------------------------------------------------------------------------
    env.reset(None).expect("reset");
    let s = env.backend().state();
    let qpos = s.qpos.to_vec();
    for (i, name) in HAND.iter().enumerate() {
        let (lo, hi) = joint(&scene, name).range.expect("limited");
        assert!(qpos[i] >= lo && qpos[i] <= hi, "{name} = {}", qpos[i]);
        assert!(qpos[i].abs() <= 0.4, "{name} = {}: small noise", qpos[i]);
    }
    for (j1, j0, k) in couplings(&scene) {
        let at = |n: &str| HAND.iter().position(|h| *h == n).expect("hand joint");
        let (a, b) = (qpos[at(&j1)], qpos[at(&j0)]);
        assert!(
            (b - k * a).abs() < 1e-12,
            "{j0} = {b}, {k} x {j1} = {}",
            k * a
        );
    }
    let (c, g) = (24, 31);
    assert!(
        (qpos[c] - P_REF[0]).abs() <= CUBE_NOISE && (qpos[c + 1] - P_REF[1]).abs() <= CUBE_NOISE
    );
    assert_eq!(qpos[c + 2], P_REF[2]);
    // One draw per cube: (1, TILT, -TILT u, u).
    for start in [c + 3, g + 3] {
        let q = &qpos[start..start + 4];
        let u = q[3];
        assert_eq!((q[0], q[1]), (1.0, TILT), "{q:?}");
        assert!((q[2] + TILT * u).abs() < 1e-15 && u.abs() <= 1.0, "{q:?}");
    }
    let xq = s.xquat.to_vec();
    let norm = |q: &[f64]| q.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (norm(&xq[object * 4..object * 4 + 4]) - 1.0).abs() < 1e-12,
        "xquat is normalized"
    );

    // --- three scripted states, one control step each ---------------------------------------
    let ctrl = vec![0.0; 20];
    let base_qpos = qpos.clone();
    let base_qvel = s.qvel.to_vec();
    let base = |edit: &dyn Fn(&mut Vec<f64>)| {
        let mut q = base_qpos.clone();
        edit(&mut q);
        q
    };
    let cases: [(&str, Vec<f64>, Termination); 3] = [
        // The cube turned to the goal's orientation, resting where it is.
        (
            "at the goal",
            base(&|q: &mut Vec<f64>| {
                let goal: Vec<f64> = q[g + 3..g + 7].to_vec();
                q[c + 3..c + 7].copy_from_slice(&goal);
            }),
            Termination::Success,
        ),
        // The goal a quarter turn from the cube: running, the reward is the curve's.
        (
            "a quarter turn away",
            base(&|q: &mut Vec<f64>| {
                q[c + 3..c + 7].copy_from_slice(&[1.0, TILT, 0.0, 0.0]);
                q[g + 3..g + 7].copy_from_slice(&[1.0, TILT, -TILT, 1.0]);
            }),
            Termination::Running,
        ),
        // Off the hand: 0.3 m from p_ref.
        (
            "0.3 m away",
            base(&|q: &mut Vec<f64>| {
                q[c] = P_REF[0] + 0.3;
                q[c + 3..c + 7].copy_from_slice(&[1.0, TILT, 0.0, 0.0]);
                q[g + 3..g + 7].copy_from_slice(&[1.0, TILT, -TILT, 1.0]);
            }),
            Termination::Failure,
        ),
    ];
    for (what, q, want) in cases {
        env.reset(None).expect("reset");
        let (tick, act, sensordata, xpos, xquat) = {
            let now = env.backend().state();
            (
                now.tick,
                now.act.to_vec(),
                now.sensordata.to_vec(),
                now.xpos.to_vec(),
                now.xquat.to_vec(),
            )
        };
        let view = StateView {
            n_envs: 1,
            tick,
            qpos: &q,
            qvel: &base_qvel,
            act: &act,
            sensordata: &sensordata,
            xpos: &xpos,
            xquat: &xquat,
        };
        env.backend_mut().set_state(&view).expect("set_state");
        let out = env.step(&ctrl).expect("one control step");
        let got = if out.dones[0] {
            out.episodes[0].termination
        } else {
            Termination::Running
        };
        // The cones read the post-step state, so the formula is taken on it. A finished episode
        // was reset inside `step`, so its post-step state is made again: the same state and
        // controls through the backend alone, for the step's two physics ticks (packet H2b:
        // the formula on the state as set, two ticks earlier, drifts with the hand's motion).
        if out.dones[0] {
            env.reset(None).expect("reset");
            env.backend_mut().set_state(&view).expect("set_state");
            env.backend_mut().set_ctrl(&ctrl).expect("ctrl");
            env.backend_mut().step(2).expect("the step again");
        }
        let after = env.backend().state();
        let (xpos, xquat) = (after.xpos.to_vec(), after.xquat.to_vec());
        // `xquat` is x y z w; a dot product does not care about the order.
        let (reward, success, fell) = expected_reward(
            &xpos[object * 3..object * 3 + 3],
            &xquat[object * 4..object * 4 + 4],
            &xquat[target * 4..target * 4 + 4],
        );
        println!(
            "{what}: reward {:.6} (formula {reward:.6}), success {success}, fell {fell}, {got:?}",
            out.rewards[0]
        );
        assert_eq!(got, want, "{what}");
        assert_eq!(
            (success, fell),
            (want == Termination::Success, want == Termination::Failure)
        );
        assert!(
            (out.rewards[0] - reward).abs() < 1e-9,
            "{what}: the cones pay {}, the header's formula {reward}",
            out.rewards[0]
        );
    }
    println!("RAN the_task_scores_scripted_states");
}

// --- the cameras ---------------------------------------------------------------------------------

#[cfg(feature = "render")]
mod render {
    use super::*;
    use es_env::render::{camera_view, check_image_spec, image_spec, render_config, sensor_cfg};
    use es_render::{Channel, Renderer, TriScene};

    /// Each camera channel's declared `ImageSpec` is what the renderer produces (INV-14), and
    /// at the declared `Pt` settings -- 32 spp, 3 bounces, exposure 8 -- each camera sees the
    /// cube (segmentation) at the scene's static pose, with no pixel at the tonemap's white.
    /// `ES_HAND_DUMP=<dir>` writes each frame's Rgb8 bytes there. SKIP without a GPU.
    #[test]
    fn the_cameras_declare_what_the_scene_renders_and_see_the_cube() {
        let task =
            es_ir::serial::task_from_toml(&read(&fixture("task-repose.toml"))).expect("task");
        let (scene, _) = scene();
        let mut cfgs = Vec::new();
        for (name, channel) in CAMERAS {
            let ch = &task.observation_spec.channels[channel];
            let (id, render) = match &ch.source {
                ObsSource::Sensor { id, render, .. } => (*id, *render),
                other => panic!("{channel}: {other:?}"),
            };
            assert_eq!(
                scene.cameras.iter().find(|c| c.name == name).map(|c| c.id),
                Some(id)
            );
            let declared = ch.ty.image.as_ref().expect("an image channel");
            let cfg = sensor_cfg(id, declared, &render, None);
            let ours = image_spec(&scene, &cfg).expect("the camera resolves");
            check_image_spec(&ours, declared).unwrap_or_else(|e| panic!("{channel}: {e}"));
            assert!((ours.intrinsics.fx - declared.intrinsics.fx).abs() < 1e-3);
            cfgs.push((name, cfg));
        }

        let gpu = match es_gpu::SlangCompiler::new()
            .map_err(|e| e.to_string())
            .and_then(|_| {
                es_gpu::Gpu::open(es_gpu::GpuOptions::default()).map_err(|e| e.to_string())
            }) {
            Ok(gpu) => gpu,
            Err(e) => {
                println!("SKIP the_cameras_see_the_cube (Pt): {e}");
                return;
            }
        };
        let tri = TriScene::from_scene(&scene).expect("the hand tessellates");
        let mut rc = render_config(&cfgs[0].1);
        rc.channels = BTreeSet::from([Channel::Rgb8, Channel::SegmentationId]);
        let mut renderer = Renderer::new(&gpu, rc).expect("renderer");
        renderer.upload_tris(tri.clone()).expect("upload");
        for (name, cfg) in &cfgs {
            let view = camera_view(&scene, cfg, &BTreeMap::new()).expect("the camera");
            let mut atlas = renderer.render(&[view]).expect("render");
            let seg = atlas.read_tile(0, Channel::SegmentationId).expect("seg");
            let cube = seg
                .as_u32()
                .expect("u32")
                .iter()
                .filter(|s| tri.names.get(s).is_some_and(|n| n == "object"))
                .count();
            let rgb = atlas.read_tile(0, Channel::Rgb8).expect("rgb");
            let rgb = rgb.as_u8().expect("u8");
            let white = rgb
                .chunks(3)
                .filter(|p| p.iter().all(|c| *c == 255))
                .count();
            if let Ok(dir) = std::env::var("ES_HAND_DUMP") {
                std::fs::create_dir_all(&dir).expect("dump dir");
                std::fs::write(Path::new(&dir).join(format!("{name}_pt.rgb")), rgb).expect("dump");
            }
            println!("{name}: cube pixels {cube}, white pixels {white}");
            assert!(cube > 100, "`{name}` does not see the cube: {cube}");
            assert_eq!(white, 0, "`{name}` saturates at exposure {EXPOSURE}");
        }
        println!("RAN the_cameras_declare_what_the_scene_renders_and_see_the_cube");
    }

    /// Packet M16/H3: the sweep `task-repose.toml`'s render settings were chosen from. The
    /// three cameras at the reset pose and three mid-episode poses (the hand driven by a slow
    /// sinusoid on `mujoco-cpu`), rendered at 4 / 8 / 16 / 32 / 64 spp x 2 / 3 bounces x SVGF off / on
    /// at tick 0, each against a 256-spp reference: per camera the RMSE (sRGB bytes) and the
    /// share of speckle pixels (luminance under half the reference's where the reference is
    /// above 40), and the wall clock per camera (one dispatch renders all three, as `es loop
    /// collect` does; 8 frames after a warm-up). `ES_HAND_SHEET=<dir>` writes one PPM per pose
    /// (rows: the configs in the printed order, reference last; columns: top, front, side).
    /// Needs `ES_PYTHON` (`MuJoCo`) and a GPU; a measurement, so run explicitly.
    #[test]
    #[ignore = "measurement; run explicitly"]
    #[allow(clippy::too_many_lines)]
    fn render_quality_sweep() {
        use es_env::render::EnvRenderer;
        use es_env::{scheduler::BatchDomains, Env};
        use es_physics_backend::MuJoCoCpuBackend;
        use es_physics_core::backend::{PhysicsBackend, StateView};

        if std::env::var_os("ES_PYTHON").is_none() {
            println!("SKIP render_quality_sweep: ES_PYTHON is not set (MuJoCo)");
            return;
        }
        let gpu = match es_gpu::Gpu::open(es_gpu::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(e) => {
                println!("SKIP render_quality_sweep: {e}");
                return;
            }
        };
        let task =
            es_ir::serial::task_from_toml(&read(&fixture("task-repose.toml"))).expect("task");
        let (scene, _) = scene();
        let domains = BatchDomains::single_env_at(
            TickRate::from_period_secs(scene.options.timestep).expect("120 Hz"),
            control_rate(&scene),
        )
        .expect("two ticks per control step");
        let mut env = Env::new(&task, &scene, MuJoCoCpuBackend::new(), &domains, 7)
            .expect("the task compiles");
        let model = env.model().clone();
        env.reset(None).expect("reset");
        let range = ctrlrange(&scene);
        let snapshot = |env: &Env<MuJoCoCpuBackend>| {
            let s = env.backend().state();
            (s.xpos.to_vec(), s.xquat.to_vec())
        };
        let mut poses = vec![("reset", snapshot(&env))];
        for step in 1..=150u32 {
            let t = f64::from(step) / CONTROL_HZ as f64;
            let ctrl: Vec<f64> = range
                .iter()
                .enumerate()
                .map(|(j, (lo, hi))| {
                    let phase = 0.7 * j as f64;
                    f64::midpoint(*lo, *hi) + 0.4 * (hi - lo) / 2.0 * (2.0 * t + phase).sin()
                })
                .collect();
            env.step(&ctrl).expect("step");
            if [40, 90, 150].contains(&step) {
                poses.push((
                    ["", "mid-40", "mid-90", "mid-150"][poses.len()],
                    snapshot(&env),
                ));
            }
        }

        let cams = |spp: u32, bounces: u32, svgf: bool| -> Vec<(String, _)> {
            CAMERAS
                .iter()
                .map(|(_, channel)| {
                    let ch = &task.observation_spec.channels[*channel];
                    let ObsSource::Sensor { id, render, .. } = &ch.source else {
                        panic!("{channel}")
                    };
                    let render = SensorRender {
                        path: SensorPath::Pt { spp, bounces },
                        svgf,
                        ..*render
                    };
                    let spec = ch.ty.image.as_ref().expect("image");
                    ((*channel).to_owned(), sensor_cfg(*id, spec, &render, None))
                })
                .collect()
        };
        let render_all =
            |r: &mut EnvRenderer<'_>, (xpos, xquat): &(Vec<f64>, Vec<f64>)| -> Vec<Vec<u8>> {
                let state = StateView {
                    n_envs: 1,
                    xpos,
                    xquat,
                    ..StateView::default()
                };
                r.begin_episode();
                r.frames_with(
                    &model,
                    &state,
                    0,
                    &es_env::randomize::RenderOverrides::default(),
                )
                .expect("frames")
                .into_iter()
                .map(|(_, t)| t.as_u8().expect("rgb8").to_vec())
                .collect()
            };

        let reference: Vec<Vec<Vec<u8>>> = {
            let mut r = EnvRenderer::views(&gpu, &scene, cams(256, 3, false)).expect("renderer");
            poses.iter().map(|(_, p)| render_all(&mut r, p)).collect()
        };
        let lum = |p: &[u8]| {
            0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
        };
        let mut sheets: Vec<Vec<Vec<Vec<u8>>>> = vec![Vec::new(); poses.len()];
        println!("config            ms/camera  | RMSE top front side | speckle % top front side");
        for spp in [4, 8, 16, 32, 64] {
            for bounces in [2, 3] {
                for svgf in [false, true] {
                    let mut r = EnvRenderer::views(&gpu, &scene, cams(spp, bounces, svgf))
                        .expect("renderer");
                    render_all(&mut r, &poses[0].1);
                    let t0 = std::time::Instant::now();
                    for _ in 0..8 {
                        render_all(&mut r, &poses[1].1);
                    }
                    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 8.0 / 3.0;
                    let (mut rmse, mut speckle) = ([0.0f64; 3], [0.0f64; 3]);
                    for (k, (_, pose)) in poses.iter().enumerate() {
                        let frames = render_all(&mut r, pose);
                        for (c, (img, want)) in frames.iter().zip(&reference[k]).enumerate() {
                            let se: f64 = img
                                .iter()
                                .zip(want)
                                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                                .sum();
                            rmse[c] += (se / img.len() as f64).sqrt() / poses.len() as f64;
                            let lit: Vec<(f64, f64)> = img
                                .chunks(3)
                                .zip(want.chunks(3))
                                .map(|(a, b)| (lum(a), lum(b)))
                                .filter(|(_, b)| *b > 40.0)
                                .collect();
                            let dark = lit.iter().filter(|(a, b)| *a < 0.5 * b).count();
                            speckle[c] +=
                                100.0 * dark as f64 / lit.len().max(1) as f64 / poses.len() as f64;
                        }
                        sheets[k].push(frames);
                    }
                    println!(
                        "spp {spp:>2} b {bounces} svgf {:<5} {ms:>7.2}  | {:>5.1} {:>5.1} {:>5.1} | {:>5.2} {:>5.2} {:>5.2}",
                        svgf, rmse[0], rmse[1], rmse[2], speckle[0], speckle[1], speckle[2]
                    );
                }
            }
        }
        if let Ok(dir) = std::env::var("ES_HAND_SHEET") {
            std::fs::create_dir_all(&dir).expect("sheet dir");
            let w = IMAGE as usize;
            for (k, (name, _)) in poses.iter().enumerate() {
                let rows: Vec<&Vec<Vec<u8>>> = sheets[k]
                    .iter()
                    .chain(std::iter::once(&reference[k]))
                    .collect();
                let mut ppm = format!("P6\n{} {}\n255\n", 3 * w, rows.len() * w).into_bytes();
                for row in rows {
                    for y in 0..w {
                        for img in row {
                            ppm.extend_from_slice(&img[y * w * 3..(y + 1) * w * 3]);
                        }
                    }
                }
                std::fs::write(Path::new(&dir).join(format!("{name}.ppm")), ppm).expect("ppm");
            }
        }
        println!("RAN render_quality_sweep over {} poses", poses.len());
    }

    /// Packet H1b: the evaluation's `light_intensity` draw (`LightOverride::scene`) scales what
    /// the hand scene draws -- each triangle's base colour factor, the textured cubes' and the
    /// `MatViz` hand's included -- by the draw, and leaves the material table (the texels)
    /// alone. Before the fix a geom that wears its material's colour (its own `rgba` at
    /// `MuJoCo`'s default) was drawn as the default grey scaled: the cube at 0.25 instead of
    /// 0.5, the hand at 0.25 instead of 0.465. 0.5 is a power of two, so the products are exact.
    #[test]
    fn a_light_intensity_draw_scales_the_drawn_colours() {
        let (scene, _) = scene();
        let base = TriScene::from_scene(&scene).expect("the hand tessellates");
        let dim = es_eval::LightOverride {
            intensity: 0.5,
            yaw_deg: 0.0,
        };
        let dimmed = TriScene::from_scene(&dim.scene(&scene)).expect("the dimmed hand");
        assert_eq!(base.tris.len(), dimmed.tris.len());
        assert_eq!(*base.materials, *dimmed.materials, "the texels do not move");
        for (a, b) in base.tris.iter().zip(&dimmed.tris) {
            assert_eq!(a.mat, b.mat);
            assert_eq!(
                b.albedo,
                a.albedo.map(|c| c * 0.5),
                "{}",
                base.names[&a.seg]
            );
        }
        let albedo = |name: &str| {
            dimmed
                .tris
                .iter()
                .find(|t| dimmed.names[&t.seg] == name)
                .unwrap_or_else(|| panic!("no triangle of {name}"))
                .albedo
        };
        assert_eq!(albedo("object"), [0.5; 3], "block.png's factor, 1 x 0.5");
        assert_eq!(albedo("robot0:V_palm"), [(0.93_f64 * 0.5) as f32; 3]);
        println!("RAN a_light_intensity_draw_scales_the_drawn_colours");
    }
}
