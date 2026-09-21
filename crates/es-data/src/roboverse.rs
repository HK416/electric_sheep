//! `RoboVerse` / `MetaSim` task config → Electric Sheep IR conversion (spec §14.4 external
//! conversion, spec §0.3 "`RoboVerse` / `MetaSim`: simulator-agnostic config, 276 tasks", spec §6
//! Task IR, spec §7.4 Observation IR declaration link, spec §25.2 derivative-work provenance).
//!
//! Mirrors `crates/es-data/src/lerobot_config.rs`'s shape: a plain serde config type, a
//! `convert` entry point that never panics on an unmapped feature, and every guess this crate
//! makes written down in `docs/api-notes/roboverse.md` rather than assumed silently. Accepts
//! **JSON only** — `MetaSim`'s native config is a Python `ScenarioCfg`/`TaskCfg` dataclass pair;
//! turning that into the flattened JSON document this module reads is a one-line Python
//! export (`dataclasses.asdict` + `json.dump`), out of scope here and not implemented in this
//! crate (`serde_yaml` and a `PyO3` call-out are both avoidable, so neither is a dependency).
//!
//! Only the task *config* is converted. `MetaSim`'s own trajectory storage is not read here:
//! `INV-16` bans pickle-based loading anywhere in this workspace, so a `.pkl`/`.npz`
//! trajectory (if that is even the real format — `docs/api-notes/roboverse.md` marks it
//! `unverified`) needs its own Python-side export to `LeRobot` before `es_data::lerobot` reads
//! it.
//!
//! Spec §14.4: an item this converter cannot map is `severity: error` and blocks execution —
//! [`convert`] never hides that behind a warning or a guess; it comes back in
//! [`Converted::unmapped`] and the CLI (`es import roboverse`) exits 1 when any entry is
//! [`Severity::Error`].

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use es_core::StableId;
use es_ir::graph::{Graph, NodeId, PortRef};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
};
use es_ir::observation::{
    self, Io, ObservationIr, ObservationNode, ObservationOutput, ResizeFilter,
};
use es_ir::task::{
    ActionSpace, Aggregation, CmpOp, Distribution, JointQuantity, LogicOp, ObsChannel, ObsSource,
    ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_math::conventions::Pose;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `RoboVerse`'s `episode_length` is an env-step count, not seconds (spec §0.3 gives no
/// step-rate); this is this crate's own assumption when a config names no
/// `control_rate_hz` (`docs/api-notes/roboverse.md`: unverified).
const DEFAULT_CONTROL_RATE_HZ: f32 = 20.0;
/// `JointPosChecker.tolerance` fallback when the config omits it (`unverified`).
const DEFAULT_JOINT_TOLERANCE: f64 = 0.05;

// --- task config JSON (docs/api-notes/roboverse.md) ------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetFormat {
    Usd,
    Mjcf,
    Urdf,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssetPaths {
    #[serde(default)]
    pub usd_path: Option<String>,
    #[serde(default)]
    pub mjcf_path: Option<String>,
    #[serde(default)]
    pub urdf_path: Option<String>,
}

impl AssetPaths {
    /// Exactly one of the three paths, per `RobotCfg`'s fetched shape (`verified (fetched)`,
    /// `docs/api-notes/roboverse.md`). More or fewer than one is a malformed asset reference.
    fn resolve(&self) -> Option<(AssetFormat, &str)> {
        match (&self.usd_path, &self.mjcf_path, &self.urdf_path) {
            (Some(p), None, None) => Some((AssetFormat::Usd, p.as_str())),
            (None, Some(p), None) => Some((AssetFormat::Mjcf, p.as_str())),
            (None, None, Some(p)) => Some((AssetFormat::Urdf, p.as_str())),
            _ => None,
        }
    }
}

/// `RobotCfg` (spec fetched from `concept/config.html`): `name`, one asset path, and
/// `control_type` (`verified (fetched)`: "Dict of joint -> control mode", so its key set is
/// this crate's joint name list). `actuators` / `joint_limits` are not modeled (unverified
/// shape) and fall into `extra`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobotCfg {
    pub name: String,
    #[serde(flatten)]
    pub asset: AssetPaths,
    #[serde(default)]
    pub control_type: BTreeMap<String, String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `BaseObjCfg` — `unverified` (`docs/api-notes/roboverse.md`): no fetched page enumerated its
/// fields, so this is the `RobotCfg`-analogous guess.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectCfg {
    pub name: String,
    #[serde(flatten)]
    pub asset: AssetPaths,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Camera config — `unverified` shape. Absent `intrinsics` synthesizes a nominal pinhole
/// (same fallback as `lerobot_config.rs::nominal_camera`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraCfg {
    pub name: String,
    pub resolution: [u32; 2],
    #[serde(default)]
    pub intrinsics: Option<CameraIntrinsics>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One `randomization[]` entry — `unverified` shape (`docs/api-notes/roboverse.md`). Kept as
/// a raw `distribution` value so an unrecognized `kind` degrades to a warning rather than a
/// parse failure (spec §14.4 draws the error/warning line at "meaningless to run", and a
/// missing randomization term does not cross it the way an unmapped checker does).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandTerm {
    pub target: String,
    pub distribution: Value,
}

/// The flattened task config this crate accepts (`docs/api-notes/roboverse.md`): a `TaskCfg`
/// and its `ScenarioCfg` joined into one JSON document by the export step this crate assumes
/// but does not implement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoboVerseTask {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub scene: Option<String>,
    pub robots: Vec<RobotCfg>,
    #[serde(default)]
    pub objects: Vec<ObjectCfg>,
    #[serde(default)]
    pub cameras: Vec<CameraCfg>,
    pub episode_length: u32,
    #[serde(default)]
    pub control_rate_hz: Option<f32>,
    /// Raw JSON: the checker taxonomy is only partially `verified (fetched)` (class names, not
    /// field shapes), so this crate dispatches on `checker["kind"]` itself rather than a typed
    /// enum, and reports an unrecognized `kind` by name (see [`map_checker`]).
    pub checker: Value,
    #[serde(default)]
    pub randomization: Vec<RandTerm>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl RoboVerseTask {
    pub fn parse(json: &str) -> Result<Self, ConvertError> {
        Ok(serde_json::from_str(json)?)
    }
}

// --- conversion result ------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Warning,
    Error,
}

/// One item spec §14.4's "Semantic Mapping Report" could not place. `severity: Error` blocks
/// execution; `convert` still returns `Ok` so the caller sees the whole report at once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unmapped {
    pub item: String,
    pub severity: Severity,
}

/// An external asset this task refers to, not yet imported. `es import mjcf|urdf|usd` is the
/// actual mesh/kinematics converter (spec §14.4); this only maps config shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneAssetRef {
    pub path: String,
    pub format: AssetFormat,
    pub license: Option<String>,
}

/// Spec §25.2: "Isaac Lab / `RoboVerse` conversion output derivative-work status" has no v1.0
/// answer, only a requirement that the converter record what it started from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub name: String,
    pub version: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug)]
pub struct Converted {
    pub task: TaskIr,
    pub observation: ObservationIr,
    pub scene_refs: Vec<SceneAssetRef>,
    pub provenance: Provenance,
    pub warnings: Vec<String>,
    pub unmapped: Vec<Unmapped>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("malformed RoboVerse task JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no robots declared")]
    NoRobots,
    /// It divides the episode length into `limit_s` and lands in `TaskConfig`, so a `0.0`,
    /// a negative or a non-finite rate puts `+inf`/`NaN` into the Task IR and into
    /// `task_hash` (`CanonWriter` rejects `NaN`, but only much later and as an IR error).
    #[error("control_rate_hz must be finite and greater than 0, got {0}")]
    ControlRate(f32),
    #[error("IR construction failed: {0}")]
    Ir(String),
}

// --- shared IR-building helpers (mirrors task.rs's private helpers; duplicated rather than
// exposed, since es-ir is out of scope for this packet) --------------------------------------

fn f32v(n: u64, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

fn flag() -> PortType {
    PortType {
        elem: ElemType::Bool,
        shape: Shape::new([1]),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: TimeRef::Tick,
        image: None,
    }
}

fn boolish(ty: &PortType) -> PortType {
    PortType {
        elem: ElemType::Bool,
        shape: ty.shape.clone(),
        unit: Unit::Dimensionless,
        frame: Frame::World,
        time: ty.time.clone(),
        image: None,
    }
}

/// A nominal pinhole `ImageSpec` at `(width, height)`: `intr`, or `fx = fy = width` with a
/// centred principal point (same fallback `lerobot_config.rs::nominal_camera` uses).
fn nominal_camera(width: u32, height: u32, intr: Option<CameraIntrinsics>) -> ImageSpec {
    let i = intr.unwrap_or(CameraIntrinsics {
        fx: f64::from(width),
        fy: f64::from(width),
        cx: f64::from(width) / 2.0,
        cy: f64::from(height) / 2.0,
    });
    ImageSpec {
        width,
        height,
        channels: ChannelFormat::Rgb,
        dtype: ImageDType::U8,
        color_space: ColorSpace::SRgb,
        camera_model: CameraModel::Pinhole,
        intrinsics: Intrinsics::new(i.fx, i.fy, i.cx, i.cy),
        extrinsics: Pose::IDENTITY,
        distortion: DistortionModel::None,
        shutter: es_ir::image::ShutterModel::Global,
        exposure: Duration::ZERO,
        rate_hz: 0.0,
        depth_scale: None,
    }
}

fn image_ty(sensor: StableId, spec: &ImageSpec) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([3, u64::from(spec.height), u64::from(spec.width)]),
        unit: Unit::Pixel,
        frame: Frame::Camera(sensor),
        time: TimeRef::Sensor {
            id: sensor,
            align: Align::Hold,
        },
        image: Some(*spec),
    }
}

fn next_id(counter: &mut u32) -> NodeId {
    let id = NodeId(*counter);
    *counter += 1;
    id
}

/// Wires `gate`'s boolean output into a sparse `Reward` and a `Terminate(Success)` — the
/// common shape every checker of spec §14.4's mapping table reduces to.
fn add_success_nodes(
    graph: &mut TaskGraph,
    counter: &mut u32,
    gate: NodeId,
    gate_port: &str,
    gate_ty: PortType,
    name: &str,
) {
    let reward_id = next_id(counter);
    graph.insert(
        reward_id,
        TaskNode::Reward {
            name: name.to_owned(),
            weight: 1.0,
            aggregation: Aggregation::Sum,
            ty: gate_ty,
        },
    );
    graph.connect(gate, gate_port, reward_id, "value");

    let term_id = next_id(counter);
    graph.insert(
        term_id,
        TaskNode::Terminate {
            kind: TerminationKind::Success,
        },
    );
    graph.connect(gate, gate_port, term_id, "value");
}

// --- checker mapping (docs/api-notes/roboverse.md: class names verified, fields unverified) --

/// Maps `checker` (a `{"kind": ..., ...}` JSON object) to `Reward`/`Terminate(Success)` nodes.
/// An unrecognized `kind`, or a recognized one missing a required field, is `severity: Error`
/// (spec §14.4): this converter refuses to guess a success condition.
fn map_checker(
    checker: &Value,
    graph: &mut TaskGraph,
    counter: &mut u32,
    robots: &[RobotCfg],
    warnings: &mut Vec<String>,
) -> Vec<Unmapped> {
    let error = |item: String| {
        vec![Unmapped {
            item,
            severity: Severity::Error,
        }]
    };
    let Some(kind) = checker.get("kind").and_then(Value::as_str) else {
        return error("checker: missing \"kind\"".to_owned());
    };

    match kind {
        "DetectedChecker" => {
            let Some(object) = checker.get("object").and_then(Value::as_str) else {
                return error("DetectedChecker: missing \"object\"".to_owned());
            };
            let detector = checker
                .get("detector")
                .and_then(Value::as_str)
                .or_else(|| robots.first().map(|r| r.name.as_str()))
                .unwrap_or("ee");
            let a = StableId::from_path(&format!("object/{object}"));
            let b = StableId::from_path(&format!("robot/{detector}"));
            let id = next_id(counter);
            graph.insert(id, TaskNode::GetContact { a, b });
            add_success_nodes(graph, counter, id, "touching", flag(), "success");
            Vec::new()
        }
        "JointPosChecker" => {
            let (Some(robot), Some(joint), Some(target)) = (
                checker.get("robot").and_then(Value::as_str),
                checker.get("joint").and_then(Value::as_str),
                checker.get("target").and_then(Value::as_f64),
            ) else {
                return error("JointPosChecker: missing robot/joint/target".to_owned());
            };
            let tolerance = checker
                .get("tolerance")
                .and_then(Value::as_f64)
                .unwrap_or(DEFAULT_JOINT_TOLERANCE);
            let body = StableId::from_path(&format!("robot/{robot}"));
            let ty = f32v(1, Unit::Angle, Frame::Joint(body));

            let js_id = next_id(counter);
            graph.insert(
                js_id,
                TaskNode::GetJointState {
                    body,
                    joints: vec![joint.to_owned()],
                    quantity: JointQuantity::Position,
                },
            );
            let lo_id = next_id(counter);
            graph.insert(
                lo_id,
                TaskNode::Compare {
                    op: CmpOp::Ge,
                    rhs: Some(target - tolerance),
                    ty: ty.clone(),
                },
            );
            graph.connect(js_id, "value", lo_id, "a");
            let hi_id = next_id(counter);
            graph.insert(
                hi_id,
                TaskNode::Compare {
                    op: CmpOp::Le,
                    rhs: Some(target + tolerance),
                    ty: ty.clone(),
                },
            );
            graph.connect(js_id, "value", hi_id, "a");
            let and_id = next_id(counter);
            graph.insert(
                and_id,
                TaskNode::Logic {
                    op: LogicOp::And,
                    shape: Shape::new([1]),
                },
            );
            graph.connect(lo_id, "value", and_id, "a");
            graph.connect(hi_id, "value", and_id, "b");
            add_success_nodes(graph, counter, and_id, "value", boolish(&ty), "success");
            Vec::new()
        }
        "PositionShiftChecker" => {
            let Some(object) = checker.get("object").and_then(Value::as_str) else {
                return error("PositionShiftChecker: missing \"object\"".to_owned());
            };
            let Some(distance) = checker.get("distance").and_then(Value::as_f64) else {
                return error("PositionShiftChecker: missing \"distance\"".to_owned());
            };
            let axis = checker.get("axis").and_then(Value::as_str).unwrap_or("z");
            let axis_dim: u64 = match axis {
                "x" => 0,
                "y" => 1,
                _ => 2,
            };
            warnings.push(format!(
                "PositionShiftChecker \"{object}\": approximated as an absolute position \
                 threshold on axis \"{axis}\", not a shift from the episode's reset pose \
                 (Task IR's pure dataflow graph has no node reading a value captured at the \
                 last reset; docs/api-notes/roboverse.md)"
            ));
            let body = StableId::from_path(&format!("object/{object}"));
            let pos_ty = f32v(3, Unit::Length, Frame::World);
            let pose_id = next_id(counter);
            graph.insert(
                pose_id,
                TaskNode::GetBodyPose {
                    body,
                    relative_to: Frame::World,
                },
            );
            let scalar_ty = f32v(1, Unit::Length, Frame::World);
            let slice_id = next_id(counter);
            graph.insert(
                slice_id,
                TaskNode::Slice {
                    ty: pos_ty,
                    axis: axis_dim as u32,
                    start: axis_dim,
                    len: 1,
                },
            );
            graph.connect(pose_id, "pos", slice_id, "value");
            let (op, rhs) = if distance >= 0.0 {
                (CmpOp::Ge, distance)
            } else {
                (CmpOp::Le, distance)
            };
            let cmp_id = next_id(counter);
            graph.insert(
                cmp_id,
                TaskNode::Compare {
                    op,
                    rhs: Some(rhs),
                    ty: scalar_ty.clone(),
                },
            );
            graph.connect(slice_id, "value", cmp_id, "a");
            add_success_nodes(
                graph,
                counter,
                cmp_id,
                "value",
                boolish(&scalar_ty),
                "success",
            );
            Vec::new()
        }
        other => error(format!("checker.kind = \"{other}\"")),
    }
}

fn parse_distribution(v: &Value) -> Option<Distribution> {
    match v.get("kind").and_then(Value::as_str) {
        Some("uniform") => Some(Distribution::Uniform {
            lo: v.get("low").and_then(Value::as_f64)?,
            hi: v.get("high").and_then(Value::as_f64)?,
        }),
        _ => None,
    }
}

// --- Observation IR: cameras only (spec §7.4 declares, Observation IR implements) ------------

/// `ImageInput -> Resize(to the config's declared resolution)`. The config carries only one
/// resolution per camera, so this is a same-size pass in practice — it still exercises the
/// mandatory `INV-14` intrinsics-rescale path rather than inventing an unverified native
/// sensor resolution to resize down from.
fn add_camera_observation(obs: &mut ObservationIr, counter: &mut u32, cam: &CameraCfg) {
    let sensor = StableId::from_path(&format!("camera/{}", cam.name));
    let raw_spec = nominal_camera(cam.resolution[0], cam.resolution[1], cam.intrinsics);
    let raw_ty = image_ty(sensor, &raw_spec);
    let input_id = next_id(counter);
    obs.graph.insert(
        input_id,
        ObservationNode::ImageInput {
            sensor,
            io: Io::source(raw_ty.clone()),
        },
    );

    let resized_spec = raw_spec.resized(cam.resolution[0], cam.resolution[1], true);
    let resized_ty = image_ty(sensor, &resized_spec);
    let resize_id = next_id(counter);
    obs.graph.insert(
        resize_id,
        ObservationNode::Resize {
            width: cam.resolution[0],
            height: cam.resolution[1],
            filter: ResizeFilter::Bilinear,
            rescale_intrinsics: true,
            io: Io::unary(raw_ty, resized_ty.clone()),
        },
    );
    obs.graph.connect(
        input_id,
        observation::OUT,
        resize_id,
        &observation::in_port(0),
    );

    obs.outputs.insert(
        cam.name.clone(),
        ObservationOutput {
            port: PortRef::new(resize_id, observation::OUT),
            ty: resized_ty,
        },
    );
}

// --- entry point --------------------------------------------------------------------------

/// Converts a `RoboVerse` / `MetaSim` task config (spec §14.4) into a `(TaskIr, ObservationIr)`
/// pair, the task's external asset references, and its provenance.
///
/// Robots and objects become `SceneAssetRef`s for `es import mjcf|urdf|usd` to actually
/// import; cameras become both a Task IR `ObservationSpec` channel and an Observation IR
/// `ImageInput -> Resize` chain; the checker becomes `Reward`/`Terminate(Success)`; episode
/// length becomes `TaskConfig::max_episode_steps` plus a `Terminate(Timeout)` chain;
/// randomization terms become `Randomization` nodes where the distribution kind is
/// recognized.
pub fn convert(task: &RoboVerseTask) -> Result<Converted, ConvertError> {
    if task.robots.is_empty() {
        return Err(ConvertError::NoRobots);
    }

    let mut warnings: Vec<String> = task
        .extra
        .keys()
        .map(|k| format!("task field \"{k}\" is not used by this conversion"))
        .collect();
    let mut scene_refs = Vec::new();
    let mut graph: TaskGraph = Graph::new(1);
    let mut counter = 0u32;
    let mut channels: BTreeMap<String, ObsChannel> = BTreeMap::new();
    let mut rng_streams: BTreeSet<String> = BTreeSet::new();

    let control_rate_hz = match task.control_rate_hz {
        Some(hz) if hz.is_finite() && hz > 0.0 => hz,
        Some(hz) => return Err(ConvertError::ControlRate(hz)),
        None => {
            warnings.push(format!(
                "no control_rate_hz given; assuming {DEFAULT_CONTROL_RATE_HZ} Hz \
                 (docs/api-notes/roboverse.md: unverified)"
            ));
            DEFAULT_CONTROL_RATE_HZ
        }
    };

    for robot in &task.robots {
        match robot.asset.resolve() {
            Some((format, path)) => scene_refs.push(SceneAssetRef {
                path: path.to_owned(),
                format,
                license: task.license.clone(),
            }),
            None => warnings.push(format!(
                "robot \"{}\": expected exactly one of usd_path/mjcf_path/urdf_path",
                robot.name
            )),
        }
        if robot.control_type.is_empty() {
            warnings.push(format!(
                "robot \"{}\" has no control_type entries; no joint channel or ActionSpec added",
                robot.name
            ));
            continue;
        }
        let body = StableId::from_path(&format!("robot/{}", robot.name));
        let joints: Vec<String> = robot.control_type.keys().cloned().collect();
        let dof = u32::try_from(joints.len()).unwrap_or(u32::MAX);
        let ty = f32v(u64::from(dof), Unit::Angle, Frame::Joint(body));

        let get_id = next_id(&mut counter);
        graph.insert(
            get_id,
            TaskNode::GetJointState {
                body,
                joints,
                quantity: JointQuantity::Position,
            },
        );
        let channel_name = format!("{}_joint_state", robot.name);
        let obs_id = next_id(&mut counter);
        graph.insert(
            obs_id,
            TaskNode::ObservationSpec {
                channel: channel_name.clone(),
                ty: ty.clone(),
            },
        );
        graph.connect(get_id, "value", obs_id, "value");
        channels.insert(
            channel_name,
            ObsChannel {
                source: ObsSource::JointState {
                    body,
                    dof,
                    quantity: JointQuantity::Position,
                },
                ty,
            },
        );

        let action_id = next_id(&mut counter);
        graph.insert(
            action_id,
            TaskNode::ActionSpec {
                space: ActionSpace::JointPosition,
                dim: dof,
                control_rate_hz,
            },
        );
    }

    for obj in &task.objects {
        match obj.asset.resolve() {
            Some((format, path)) => scene_refs.push(SceneAssetRef {
                path: path.to_owned(),
                format,
                license: task.license.clone(),
            }),
            None => warnings.push(format!(
                "object \"{}\": names no usd_path/mjcf_path/urdf_path; skipped as a scene ref",
                obj.name
            )),
        }
    }

    for cam in &task.cameras {
        let sensor = StableId::from_path(&format!("camera/{}", cam.name));
        let spec = nominal_camera(cam.resolution[0], cam.resolution[1], cam.intrinsics);
        let ty = image_ty(sensor, &spec);
        let get_id = next_id(&mut counter);
        graph.insert(
            get_id,
            TaskNode::GetSensor {
                sensor,
                ty: ty.clone(),
            },
        );
        let obs_id = next_id(&mut counter);
        graph.insert(
            obs_id,
            TaskNode::ObservationSpec {
                channel: cam.name.clone(),
                ty: ty.clone(),
            },
        );
        graph.connect(get_id, "value", obs_id, "value");
        channels.insert(
            cam.name.clone(),
            ObsChannel {
                source: ObsSource::Sensor {
                    id: sensor,
                    format: ChannelFormat::Rgb,
                    render: es_ir::task::SensorRender::default(),
                },
                ty,
            },
        );
    }

    for term in &task.randomization {
        match parse_distribution(&term.distribution) {
            Some(dist) => {
                let stream = format!("rand_{}", term.target.replace(['.', ' '], "_"));
                let id = next_id(&mut counter);
                graph.insert(
                    id,
                    TaskNode::Randomization {
                        target: term.target.clone(),
                        dist,
                        stream: stream.clone(),
                    },
                );
                rng_streams.insert(stream);
            }
            None => warnings.push(format!(
                "randomization target \"{}\": unrecognized distribution, dropped",
                term.target
            )),
        }
    }

    // Episode length: the runtime budget (`TaskConfig::max_episode_steps`) plus an explicit
    // graph termination (spec §6.3 `Terminate(Timeout)`), converting the env-step count to
    // seconds via `control_rate_hz` (see the `unverified` note above).
    let time_id = next_id(&mut counter);
    graph.insert(time_id, TaskNode::GetTime { since_reset: true });
    let limit_s = f64::from(task.episode_length) / f64::from(control_rate_hz);
    let over_ty = f32v(1, Unit::Time, Frame::World);
    let over_id = next_id(&mut counter);
    graph.insert(
        over_id,
        TaskNode::Compare {
            op: CmpOp::Gt,
            rhs: Some(limit_s),
            ty: over_ty,
        },
    );
    graph.connect(time_id, "value", over_id, "a");
    let timeout_id = next_id(&mut counter);
    graph.insert(
        timeout_id,
        TaskNode::Terminate {
            kind: TerminationKind::Timeout,
        },
    );
    graph.connect(over_id, "value", timeout_id, "value");

    let unmapped = map_checker(
        &task.checker,
        &mut graph,
        &mut counter,
        &task.robots,
        &mut warnings,
    );

    let scene_path = task.scene.clone().unwrap_or_else(|| {
        warnings.push("no scene given; synthesizing a placeholder scene path".to_owned());
        format!("roboverse/{}.usd", task.name)
    });
    let scene_hash = *blake3::hash(scene_path.as_bytes()).as_bytes();

    let task_ir = TaskIr {
        schema_version: 1,
        scene: SceneRef {
            path: scene_path,
            scene_hash,
            asset_hash: scene_hash,
        },
        graph,
        observation_spec: ObservationSpec { channels },
        config: TaskConfig {
            max_episode_steps: task.episode_length,
            control_rate_hz,
            deterministic: true,
            rng_streams,
        },
        control: None,
    };

    let task_ref = task_ir
        .task_hash()
        .map_err(|d| ConvertError::Ir(d.to_string()))?;
    let mut obs = ObservationIr::new(1, task_ref);
    let mut obs_counter = 0u32;
    for cam in &task.cameras {
        add_camera_observation(&mut obs, &mut obs_counter, cam);
    }

    let provenance = Provenance {
        name: task.name.clone(),
        version: task.version.clone(),
        license: task.license.clone(),
    };

    Ok(Converted {
        task: task_ir,
        observation: obs,
        scene_refs,
        provenance,
        warnings,
        unmapped,
    })
}
