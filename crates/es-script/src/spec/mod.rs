//! The task specification `*.estask` (plan G, packet G3a; `docs/design/scene-authoring.md`
//! section 4): what the task is, in the sentence editor's terms — objects of the scene and
//! relations from a fixed vocabulary — compiled with the scene into a Task IR by
//! [`compile_task`]. A recipe, not an IR (spec 5.1 rule 6).
//!
//! [`TaskSpec`] is the document as written: every optional field an `Option`, so writing back
//! what was read changes nothing. Unknown keys are refused by name. Lengths are metres, angles
//! radians except `within_deg`, times seconds.

mod compile;
mod generate;
mod learning;
mod project;
mod recipes;
mod robot;
pub mod vocab;

use std::collections::BTreeMap;

use es_ir::task::{SeedStream, Tonemap};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use compile::{compile_clauses, compile_task, load_scene, ClauseNode};
pub use generate::{generate, Document};
pub use project::{BoxDoc, CycleDoc, Deploy, Evaluate, Family, Preset, Student, Teacher};

/// Why a specification did not read or compile. A refusal names where and which field.
#[derive(Debug, Error)]
pub enum SpecError {
    /// Not TOML, or not this schema: an unknown key, a missing one, a wrong type.
    #[error("{0}")]
    Toml(#[from] toml::de::Error),
    #[error("{0}")]
    Write(#[from] toml::ser::Error),
    /// The scene did not load.
    #[error("scene `{path}`: {reason}")]
    Scene { path: String, reason: String },
    /// `at` is the place in the document (`success[0] (cube.x inside)`, `start[2] (cube.z)`,
    /// `observe.state.joint_pos`), `field` the key.
    #[error("{at}: `{field}`: {reason}")]
    Field {
        at: String,
        field: String,
        reason: String,
    },
}

pub(crate) fn refuse<T>(
    at: impl Into<String>,
    field: impl Into<String>,
    reason: impl Into<String>,
) -> Result<T, SpecError> {
    Err(SpecError::Field {
        at: at.into(),
        field: field.into(),
        reason: reason.into(),
    })
}

/// A task specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    /// Always `"task-spec"`.
    pub kind: String,
    /// Always 1.
    pub schema: u32,
    /// The scene file (`.esscene`, MJCF, URDF), relative to the project root [`compile_task`]
    /// is given; written into `SceneRef.path` as it is written here.
    pub scene: String,
    /// The robot's root body, or an `.esscene` `[[include]]` naming it (its one root body whose
    /// subtree has joints). Its subtree's joints are `robot.joints`; the scene's actuators are
    /// its action.
    pub robot: String,
    pub control_hz: f64,
    /// The episode budget; `control_hz × timeout_s` control steps.
    pub timeout_s: f64,
    /// Every clause holds: the attempt succeeded.
    pub success: Clauses,
    /// Any clause holds: the attempt failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<Clauses>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<Start>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observe: Option<Observe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<RewardDoc>,
    // The other documents' sections (packet G3b, `project.rs`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<Teacher>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub student: Option<Student>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deploy: Option<Deploy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluate: Option<Evaluate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle: Option<CycleDoc>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clauses {
    pub clauses: Vec<Clause>,
}

/// The vocabulary of section 4.1: what `es-env` lowers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// A scalar inside `range`, or a body inside the region (site) `object`, all three axes.
    Inside,
    /// A scalar above `value`, or a body higher than the body `object` by more than `m`.
    Above,
    /// A scalar below `value`, or a body lower than the body `object` by more than `m`.
    Below,
    /// A body within `m` of `object` or `point`.
    Near,
    /// A body farther than `m` from `object` or `point`.
    FartherThan,
    /// A scalar's velocity within ±`speed`, or a free body's speed under `speed` (and its
    /// angular rate under `angular`).
    Still,
    /// A body's orientation within `within_deg` of `object`'s.
    OrientationMatches,
    /// Refused until `GetContact` lowers in `es-env`.
    Touches,
}

impl Relation {
    /// The name the document writes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Inside => "inside",
            Self::Above => "above",
            Self::Below => "below",
            Self::Near => "near",
            Self::FartherThan => "farther_than",
            Self::Still => "still",
            Self::OrientationMatches => "orientation_matches",
            Self::Touches => "touches",
        }
    }
}

/// The dense reward term a clause pays every step (section 4.1, "reward shaping per clause").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shaping {
    /// `near` / `farther_than`, and `inside` a region (the distance to its centre):
    /// `weight × distance`, the distance clamped to [0, 1] m.
    Distance,
    /// `inside` a range: `weight × (s − ramp[0]) / (ramp[1] − ramp[0])`, clamped to [0, 1].
    Ramp,
    /// `orientation_matches`: `weight / (s + 0.1)`, `s = √(8 (1 − |q·g|))` — Isaac Lab's
    /// rotation reward, as its piecewise-linear interpolant at nine knots (plan H) plus the
    /// floor paid every step: terms `<term>_0` .. `<term>_7` and `<term>_floor`.
    InverseAngle,
}

/// One sentence: `subject relation [object | point | range | value ...]`.
///
/// A **scalar** subject (`inside` a `range`, `above` / `below` a `value`, `still`) is a joint, or
/// `<body>.x|y|z` — a free body's `x` is its free joint's first lane, every other coordinate a
/// lane of the body's world position (or velocity). A **body** subject (`near`,
/// `farther_than`, `orientation_matches`, `inside` a region, `above` / `below` a body, `still`
/// of a body that is not also a joint's name) is a body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clause {
    pub subject: String,
    pub relation: Relation,
    /// The other body of `near`, `farther_than`, `orientation_matches`, `above` / `below`; the
    /// region (a site) of `inside`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// A fixed world point in place of `object` (`near`, `farther_than`); offsets clamp at 1 m
    /// per axis, so `m` is under 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<[f64; 3]>,
    /// `inside`: `[lo, hi]`, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    /// `above` / `below`: the bound, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// `near` / `farther_than`: the distance, exclusive; `above` / `below` a body: the margin,
    /// exclusive, absent 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<f64>,
    /// `still`: the velocity bound, exclusive, either way (a body: its speed `‖v‖`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// `still` of a body: the bound on its angular rate `‖w‖`, rad/s, exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angular: Option<f64>,
    /// `orientation_matches`: the largest rotation between the two, in degrees, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shaping: Option<Shaping>,
    /// The shaping term's weight, before `[reward] scale`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// `shaping = "ramp"`: where the term is 0 and where it is 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ramp: Option<[f64; 2]>,
    /// The shaping term's name; absent is `<subject>_<shaping>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub term: Option<String>,
}

/// The reset: placements, 🎲 items and how strongly they are drawn.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Start {
    /// Scales every 🎲 item's `range` (about its centre) and `noise`; absent leaves them as
    /// written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f64>,
    /// `true`: a coordinate of a free body (outside the robot) that no item sets starts at zero,
    /// its quaternion all zeros (the backend reads it as the identity), as in the documents
    /// committed before the rule. Absent or `false`: it starts where the scene puts it, the
    /// body's own pose (`qpos0`, design note section 4.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_unset: Option<bool>,
    pub items: Vec<StartItem>,
}

/// One placement. `what` is `robot.joints`, a joint, `<body>.x|y|z` (the body's free joint)
/// or `<body>.orientation`.
///
/// `value` alone is a constant; `value` with `noise` is uniform over `value ± noise`, a
/// constant at zero noise; `range` is uniform over it, always. On `robot.joints`, `noise` is
/// the fraction of each joint's range. An orientation is drawn as `draw` says; the free
/// joint's quaternion is written unnormalized and the backend normalizes it (`MuJoCo` and
/// `MJWarp`, measured by plan H's H2b), so each lane is one draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartItem {
    pub what: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[f64; 2]>,
    /// `<body>.orientation`: how the orientation is drawn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<Draw>,
    /// `draw = "yaw"`: `tan(alpha / 2)` of the resting tilt about world X; absent is 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tilt: Option<f64>,
    /// `draw = "tilt"`: the largest tilt from vertical, in degrees, under 180.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tilt_max_deg: Option<f64>,
    /// `robot.joints`: a joint coupled to another by a two-joint fixed tendon draws from the
    /// other's stream, scaled by the coupling, so the tendon starts at its rest length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupled: Option<bool>,
    /// 🎲: a `Randomization` node (domain randomization) rather than a `ResetState`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dice: Option<bool>,
    /// The draw's stream; absent is `what` (`reset.<joint>` for a joint). Items naming one
    /// stream share its draw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
}

/// How `<body>.orientation` is drawn. Each lane of the quaternion `(w, x, y, z)` is one draw
/// (a reset node writes one number), so the constructions are those whose lanes are each a
/// draw of their own or a fixed multiple of a shared one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Draw {
    /// Plan H's: `(1, tilt, −tilt·u, u)`, one `u ~ U(−1, 1)` — the resting tilt about world X
    /// composed with a yaw `2·atan(u)` in [−90°, 90°] about world Z. The top face stays at the
    /// resting tilt.
    Yaw,
    /// `(1, a, b, g)`, `a, b ~ U(−m, m)` with `m = tan(theta/2)/√2`, `g ~ N(0, 1)`: the tilt from
    /// vertical is at most `tilt_max_deg` (`a² + b² ≤ tan²(theta/2) ≤ tan²(theta/2)·(1 + g²)`, reached
    /// at `g = 0` and the corners), the yaw `2·atan(g)` takes every heading — 68 % within
    /// ±90°, 8 % beyond ±120°. Not uniform over the cap: a tilt bound with a free yaw needs the
    /// yaw pair `(w, z)` away from zero, which no bounded draw covering every heading is, so
    /// the yaw lane is unbounded instead.
    Tilt,
    /// Four `N(0, 1)` lanes, normalized by the backend: uniform over SO(3).
    Any,
}

/// The observation channels. `state` and `privileged` map a channel name to a source:
/// `robot.joint_pos`, `robot.joint_vel`, `robot.previous_action`, `<body>.pose` (xpos ‖ xquat),
/// `<body>.qpos` (the free joint's seven), `<body>.vel` (the free joint's six). `privileged`
/// is what the simulator knows and a real robot does not (the teacher's extra inputs).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observe {
    /// Each is channel `rgb_<camera>`, `camera_px` square, RGB8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cameras: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_px: Option<u32>,
    /// Absent is the rasterizer (the Task IR's default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<RenderDoc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privileged: Option<BTreeMap<String, String>>,
}

/// `SensorRender` as a person writes it: `path = "pt"` needs `spp` and `bounces`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderDoc {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<RenderPath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spp: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounces: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tonemap: Option<Tonemap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<SeedStream>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub svgf: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderPath {
    Rs,
    Pt,
}

/// The sparse terms and the scale on every term.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardDoc {
    /// Multiplies every weight (`rl_games`' `scale_value`); absent is 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    /// Term `success`, paid on the step the success clauses hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<f64>,
    /// Term `failure`, paid on the step a failure clause holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<f64>,
}

impl TaskSpec {
    pub fn from_toml(text: &str) -> Result<Self, SpecError> {
        let doc: Self = toml::from_str(text)?;
        if doc.kind != "task-spec" {
            return refuse(
                "task-spec",
                "kind",
                format!("expected `task-spec`, found `{}`", doc.kind),
            );
        }
        if doc.schema != 1 {
            return refuse(
                "task-spec",
                "schema",
                format!("expected 1, found {}", doc.schema),
            );
        }
        Ok(doc)
    }

    /// Writes the document back: what was read, field for field.
    pub fn to_toml(&self) -> Result<String, SpecError> {
        Ok(toml::to_string(self)?)
    }
}
