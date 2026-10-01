//! ①'s task part (packet M17/G8, `docs/design/scene-authoring.md` section 4.6): the task
//! specification as sentences — "[cube] is [inside] [bin]", "fails if not done within [8 s]" —
//! each slot a name of the scene, a relation, or a number in the person's units, and the edits a
//! person makes to them. `es-editor` hands each edited specification to [`SceneModel::apply`] as
//! one [`crate::Command::Spec`], which `compile_task` checks on the scene as edited.
//!
//! Headless: the subjects and relations offered (es-script's [`vocab`], the compiler's own
//! table), how a number is shown, the weight levels, the defaults of "say the task" and whether
//! an observed camera sees what a clause is about are all here, under test; `es-editor` turns a
//! [`Sentence`] into words of its tables and widgets. A number is shown converted (cm, °) and
//! written back only when the person changes it, so a field nobody touched keeps its bits.
//!
//! Layout: the sentences themselves (their slots, a clause, a start item and the time limit
//! as words) and the compiler's refusal are here; how a number is shown is
//! `sentence/numbers.rs`, the weight levels `sentence/weights.rs`, what the scene offers a slot
//! and what an edit implies `sentence/edit.rs`, the specification's lists `sentence/lists.rs`,
//! saying the task `sentence/new_task.rs` and the camera check `sentence/camera_check.rs`;
//! every item keeps its `es_editor_scene::sentence::` path.
//!
//! [`SceneModel::apply`]: crate::SceneModel::apply

use es_assets::scene::{JointKind, SceneDesc};
use es_script::spec::{SpecError, StartItem};

// What `es-editor` names through this module, which has no `es-script` of its own.
pub use es_script::spec::{vocab, Clause, Draw, Relation, TaskSpec};

use crate::check::{Refusal, OTHER};

mod camera_check;
mod edit;
mod lists;
mod new_task;
mod numbers;
mod weights;

pub use edit::{bodies, choices, edit_clause, edit_item, set_relation, sources};
pub use lists::{
    add_clause, add_item, add_source, look, move_clause, remove_clause, remove_item, remove_source,
    set_camera, set_look, swap_source, Look,
};
pub use new_task::{
    control_hz, new_spec, robots, CONTROL_HZ, EPISODES, HORIZON, REWARD_SCALE, SEED, TIMEOUT_S,
};
pub use numbers::{show, show_point, Unit};
pub use weights::{
    bonus, set_bonus, set_shaping, set_strength, shapable, shaping_level, strength, Level, BONUS,
    LEVELS, SHAPING, STRENGTH,
};

pub const REQUIRED: &str = "author.task.required";
pub const TICKS: &str = "author.task.ticks";
pub const TOUCHES: &str = "author.task.touches";
pub const NO_SUCCESS: &str = "author.task.no_success";
pub const HOLD_TICKS: &str = "author.task.hold_ticks";

/// The compiler's refusal in the editor's words. Its field is the clause's place and the key
/// (`success[1].within_deg`, `timeout_s`, `observe.cameras`), as G3a names them.
pub(crate) fn refusal(e: SpecError) -> Refusal {
    match e {
        SpecError::Field { at, field, reason } => {
            let place = at.split(" (").next().unwrap_or(&at);
            // `observe.state.goal_pose` names its channel already.
            let path = if place == "task-spec" {
                field.clone()
            } else if place.ends_with(&format!(".{field}")) {
                place.to_owned()
            } else {
                format!("{place}.{field}")
            };
            let key = match field.as_str() {
                "timeout_s" => TICKS,
                "hold_s" => HOLD_TICKS,
                "relation" if reason.contains("touches") => TOUCHES,
                "clauses" => NO_SUCCESS,
                _ if reason.starts_with("required") => REQUIRED,
                _ => OTHER,
            };
            let args = if key == OTHER { vec![reason] } else { vec![] };
            Refusal::new(path, key, args)
        }
        SpecError::Scene { reason, .. } => Refusal::new("scene", OTHER, vec![reason]),
        other => Refusal::new("task", OTHER, vec![other.to_string()]),
    }
}

// --- sentences ----------------------------------------------------------------------------

/// What a slot of a sentence says.
#[derive(Clone, Debug, PartialEq)]
pub enum Slot {
    /// A name of the scene: a subject, an object, a region, a placed thing.
    Name(String),
    /// A relation, and whether it is measured against something named (`inside` a region or
    /// `inside` a range read differently).
    Relation(Relation, bool),
    Draw(Draw),
    /// A number as the document keeps it (absent: shown as 0, which is what absent means:
    /// no margin, no noise), and how it is shown.
    Number(Option<f64>, Unit),
    /// A fixed world point, metres.
    Point([f64; 3]),
}

/// Which field of a clause, a start item or the specification a slot is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Subject,
    Relation,
    Object,
    Range(usize),
    Value,
    M,
    Speed,
    Angular,
    WithinDeg,
    What,
    Noise,
    Draw,
    Tilt,
    TiltMax,
    Timeout,
    /// A section's `hold_s` (packet M18/K7).
    Hold,
}

impl Field {
    /// The document's key, as the compiler's refusals name it.
    pub fn key(self) -> &'static str {
        match self {
            Self::Subject => "subject",
            Self::Relation => "relation",
            Self::Object => "object",
            Self::Range(_) => "range",
            Self::Value => "value",
            Self::M => "m",
            Self::Speed => "speed",
            Self::Angular => "angular",
            Self::WithinDeg => "within_deg",
            Self::What => "what",
            Self::Noise => "noise",
            Self::Draw => "draw",
            Self::Tilt => "tilt",
            Self::TiltMax => "tilt_max_deg",
            Self::Timeout => "timeout_s",
            Self::Hold => "hold_s",
        }
    }
}

/// One sentence: a key of the tables whose `{0}`, `{1}`, ... are its slots, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    pub key: &'static str,
    pub slots: Vec<(Field, Slot)>,
    /// 🎲: a start item drawn anew every attempt.
    pub dice: bool,
}

/// Where a clause is: its section and its place there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct At {
    pub failure: bool,
    pub index: usize,
}

impl At {
    /// As the compiler names it: `success[1]`.
    pub fn path(self) -> String {
        let section = if self.failure { "failure" } else { "success" };
        format!("{section}[{}]", self.index)
    }
}

/// A relation's word: `inside` a region and `inside` a range (between), `above` a value and a
/// body read differently.
pub fn relation_key(r: Relation, object: bool) -> &'static str {
    match (r, object) {
        (Relation::Inside, true) => "author.task.rel.inside",
        (Relation::Inside, false) => "author.task.rel.between",
        (Relation::Above, true) => "author.task.rel.higher",
        (Relation::Above, false) => "author.task.rel.above",
        (Relation::Below, true) => "author.task.rel.lower",
        (Relation::Below, false) => "author.task.rel.below",
        (Relation::Near, _) => "author.task.rel.near",
        (Relation::FartherThan, _) => "author.task.rel.farther",
        (Relation::Still, _) => "author.task.rel.still",
        (Relation::OrientationMatches, _) => "author.task.rel.turned",
        (Relation::Touches, _) => "author.task.rel.touches",
    }
}

pub const DRAWS: [Draw; 3] = [Draw::Yaw, Draw::Tilt, Draw::Any];

pub fn draw_key(d: Draw) -> &'static str {
    match d {
        Draw::Yaw => "author.task.draw.yaw",
        Draw::Tilt => "author.task.draw.tilt",
        Draw::Any => "author.task.draw.any",
    }
}

/// The relation of a clause and whether it is measured against something named.
pub fn form(c: &Clause) -> (Relation, bool) {
    let object = match c.relation {
        Relation::Inside | Relation::Above | Relation::Below => c.object.is_some(),
        Relation::Still => false,
        _ => true,
    };
    (c.relation, object)
}

/// A hinge's numbers are angles; everything else measures metres.
fn units(scene: &SceneDesc, subject: &str) -> (Unit, Unit) {
    let hinge = (scene.joints.iter()).any(|j| j.name == subject && j.kind == JointKind::Hinge);
    if hinge {
        (Unit::Deg, Unit::DegPerS)
    } else {
        (Unit::Cm, Unit::CmPerS)
    }
}

/// A clause as a sentence.
pub fn clause(scene: &SceneDesc, c: &Clause) -> Sentence {
    let (scalar, speed) = units(scene, &c.subject);
    let (relation, object) = form(c);
    let mut slots = vec![
        (Field::Subject, Slot::Name(c.subject.clone())),
        (Field::Relation, Slot::Relation(relation, object)),
    ];
    let num = |f: Field, v: Option<f64>, u: Unit| (f, Slot::Number(v, u));
    let named = match c.point {
        Some(p) => (Field::Object, Slot::Point(p)),
        None => (
            Field::Object,
            Slot::Name(c.object.clone().unwrap_or_default()),
        ),
    };
    let key = match relation {
        Relation::Inside if object => {
            slots.push(named);
            "author.task.inside_region"
        }
        Relation::Inside => {
            let [lo, hi] = c.range.unwrap_or_default();
            slots.push(num(Field::Range(0), Some(lo), scalar));
            slots.push(num(Field::Range(1), Some(hi), scalar));
            "author.task.inside_range"
        }
        Relation::Above | Relation::Below if object => {
            slots.extend([named, num(Field::M, c.m, Unit::Cm)]);
            "author.task.compare_body"
        }
        Relation::Above | Relation::Below => {
            slots.push(num(Field::Value, c.value, scalar));
            "author.task.compare_value"
        }
        Relation::Near | Relation::FartherThan => {
            slots.extend([named, num(Field::M, c.m, Unit::Cm)]);
            if relation == Relation::Near {
                "author.task.near"
            } else {
                "author.task.farther"
            }
        }
        Relation::Still => {
            slots.push(num(Field::Speed, c.speed, speed));
            match c.angular {
                Some(a) => {
                    slots.push(num(Field::Angular, Some(a), Unit::DegPerS));
                    "author.task.still_turning"
                }
                None => "author.task.still",
            }
        }
        Relation::OrientationMatches => {
            slots.extend([named, num(Field::WithinDeg, c.within_deg, Unit::DegAsIs)]);
            "author.task.orientation"
        }
        Relation::Touches => {
            slots.push(named);
            "author.task.touching"
        }
    };
    Sentence {
        key,
        slots,
        dice: false,
    }
}

/// "fails if not done within [8 s]".
pub fn timeout(spec: &TaskSpec) -> Sentence {
    Sentence {
        key: "author.task.timeout",
        slots: vec![(
            Field::Timeout,
            Slot::Number(Some(spec.timeout_s), Unit::Seconds),
        )],
        dice: false,
    }
}

/// A section's header (packet M18/K7): "It succeeds when all of these hold for [1 s]", "It fails
/// when one of these holds for [1 s]". An absent hold reads as 0 s, the tick they hold. A failure
/// section that has no clauses reads "It fails as soon as one of these holds", with no slot: there
/// is nothing to hold.
pub fn section(spec: &TaskSpec, failure: bool) -> Sentence {
    let held = if failure {
        spec.failure.as_ref().map(|f| f.hold_s)
    } else {
        Some(spec.success.hold_s)
    };
    let (key, slots) = match held {
        None => ("author.task.failure", vec![]),
        Some(h) => (
            if failure {
                "author.task.failure_hold"
            } else {
                "author.task.success"
            },
            vec![(Field::Hold, Slot::Number(h, Unit::Seconds))],
        ),
    };
    Sentence {
        key,
        slots,
        dice: false,
    }
}

/// Sets a section's hold; 0 s, or none, is the tick they hold and writes nothing, so the document
/// is today's again. A failure section that is not there gets none.
pub fn set_hold(spec: &mut TaskSpec, failure: bool, hold_s: Option<f64>) {
    let hold_s = hold_s.filter(|h| *h != 0.0);
    if failure {
        if let Some(f) = spec.failure.as_mut() {
            f.hold_s = hold_s;
        }
    } else {
        spec.success.hold_s = hold_s;
    }
}

/// A start item as a sentence. A placement by value always shows its noise (`± 0`), so noise
/// can be given where there was none.
pub fn start_item(scene: &SceneDesc, item: &StartItem) -> Sentence {
    let (unit, _) = units(scene, &item.what);
    let n = |f: Field, v: Option<f64>, u: Unit| (f, Slot::Number(v, u));
    let (key, slots) = if item.what == "robot.joints" {
        let noise = n(Field::Noise, item.noise, Unit::Percent);
        ("author.task.start_joints", vec![noise])
    } else if let Some(body) = item.what.strip_suffix(".orientation") {
        let draw = item.draw.unwrap_or(Draw::Yaw);
        let mut slots = vec![
            (Field::What, Slot::Name(body.to_owned())),
            (Field::Draw, Slot::Draw(draw)),
        ];
        let key = match draw {
            Draw::Yaw => {
                slots.push(n(Field::Tilt, item.tilt, Unit::Tilt));
                "author.task.start_yaw"
            }
            Draw::Tilt => {
                slots.push(n(Field::TiltMax, item.tilt_max_deg, Unit::DegAsIs));
                "author.task.start_tilt"
            }
            Draw::Any => "author.task.start_any",
        };
        (key, slots)
    } else {
        let what = (Field::What, Slot::Name(item.what.clone()));
        match item.range {
            Some([lo, hi]) => (
                "author.task.start_between",
                vec![
                    what,
                    n(Field::Range(0), Some(lo), unit),
                    n(Field::Range(1), Some(hi), unit),
                ],
            ),
            None => (
                "author.task.start_at",
                vec![
                    what,
                    n(Field::Value, item.value, unit),
                    n(Field::Noise, item.noise, unit),
                ],
            ),
        }
    };
    Sentence {
        key,
        slots,
        dice: item.dice == Some(true),
    }
}
