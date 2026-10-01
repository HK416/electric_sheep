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

use std::path::Path;

use es_assets::esscene::EsScene;
use es_assets::scene::{JointKind, SceneDesc};
use es_core::StableId;
use es_ir::task::SeedStream;
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};
use es_math::{Pose, Vec3};
use es_render::cpu::nearest_hit_flat;
use es_render::TriScene;
use es_script::spec::vocab::takes;
use es_script::spec::{
    Clauses, Observe, RenderDoc, RenderPath, RewardDoc, Shaping, SpecError, Start, StartItem,
};

// What `es-editor` names through this module, which has no `es-script` of its own.
pub use es_script::spec::{vocab, Clause, Draw, Relation, TaskSpec};

use crate::check::{Refusal, OTHER};
use crate::model::SceneModel;
use crate::view::world;
use crate::SCENE_FILE;

pub const REQUIRED: &str = "author.task.required";
pub const TICKS: &str = "author.task.ticks";
pub const TOUCHES: &str = "author.task.touches";
pub const NO_SUCCESS: &str = "author.task.no_success";

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

// --- numbers ------------------------------------------------------------------------------

/// How a number of the document is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Metres, shown in centimetres.
    Cm,
    /// Radians, shown in degrees.
    Deg,
    /// Degrees, as the document keeps them (`within_deg`, `tilt_max_deg`).
    DegAsIs,
    CmPerS,
    DegPerS,
    Seconds,
    /// A fraction, shown in per cent.
    Percent,
    /// `tan(a / 2)`, shown as the angle `a` (plan H's resting tilt).
    Tilt,
    Plain,
}

impl Unit {
    /// `v` of the document in the person's unit.
    pub fn shown(self, v: f64) -> f64 {
        match self {
            Self::Cm | Self::CmPerS | Self::Percent => v * 100.0,
            Self::Deg | Self::DegPerS => v * RAD_TO_DEG,
            // Display only: what is written back goes through `es_math::approx`.
            Self::Tilt => 2.0 * v.atan() * RAD_TO_DEG,
            Self::DegAsIs | Self::Seconds | Self::Plain => v,
        }
    }

    /// The document's value of `v` in the person's unit.
    pub fn stored(self, v: f64) -> f64 {
        match self {
            Self::Cm | Self::CmPerS | Self::Percent => v / 100.0,
            Self::Deg | Self::DegPerS => v * DEG_TO_RAD,
            Self::Tilt => es_math::approx::tan_f64(v * DEG_TO_RAD / 2.0),
            Self::DegAsIs | Self::Seconds | Self::Plain => v,
        }
    }

    pub fn suffix(self) -> &'static str {
        match self {
            Self::Cm => " cm",
            Self::Deg | Self::DegAsIs | Self::Tilt => "\u{b0}",
            Self::CmPerS => " cm/s",
            Self::DegPerS => "\u{b0}/s",
            Self::Seconds => " s",
            Self::Percent => " %",
            Self::Plain => "",
        }
    }
}

/// `v` shown in `unit`, at most two decimals, with its unit.
pub fn show(v: f64, unit: Unit) -> String {
    format!("{}{}", decimals(unit.shown(v)), unit.suffix())
}

fn decimals(v: f64) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0" } else { s }.to_owned()
}

/// A fixed point, in centimetres.
pub fn show_point(p: [f64; 3]) -> String {
    let [x, y, z] = p.map(|v| decimals(Unit::Cm.shown(v)));
    format!("({x}, {y}, {z}) cm")
}

// --- weights ------------------------------------------------------------------------------

/// A weight as the person picks it. A number of the document that is no level is `Custom` and
/// is kept as written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    Low,
    Medium,
    High,
    Custom(f64),
}

/// The levels a person picks from.
pub const LEVELS: [Level; 3] = [Level::Low, Level::Medium, Level::High];
/// A shaping term's weight at each level, before `[reward] scale`; its sign is the term's (a
/// distance or a ramp costs, an angle term pays). The committed references read as levels:
/// Shadow Hand's rotation `1` and cube distance `−10`, SO-101's ramp `−1`. The owner's to change.
pub const SHAPING: [f64; 3] = [0.1, 1.0, 10.0];
/// The success bonus at each level, before `[reward] scale` (Shadow Hand's `250` is high).
pub const BONUS: [f64; 3] = [25.0, 100.0, 250.0];
/// `[start] strength` at low and high; medium is no `strength` (every range as written).
pub const STRENGTH: [f64; 2] = [0.5, 1.5];

impl Level {
    fn of(v: f64, table: &[f64; 3]) -> Self {
        match table.iter().position(|x| x.to_bits() == v.to_bits()) {
            Some(0) => Self::Low,
            Some(1) => Self::Medium,
            Some(2) => Self::High,
            _ => Self::Custom(v),
        }
    }

    fn value(self, table: &[f64; 3]) -> Option<f64> {
        match self {
            Self::Low => Some(table[0]),
            Self::Medium => Some(table[1]),
            Self::High => Some(table[2]),
            Self::Custom(_) => None,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Low => "author.task.level.low",
            Self::Medium => "author.task.level.medium",
            Self::High => "author.task.level.high",
            Self::Custom(_) => "author.task.level.custom",
        }
    }
}

fn sign(s: Shaping) -> f64 {
    if s == Shaping::InverseAngle {
        1.0
    } else {
        -1.0
    }
}

/// The shaping term a clause can pay (design section 4.3), `None` when its relation pays none.
pub fn shapable(c: &Clause) -> Option<Shaping> {
    takes(c.relation, form(c).1).shaping
}

/// The clause's shaping weight as a level; `None` when it pays no shaping term.
pub fn shaping_level(c: &Clause) -> Option<Level> {
    let s = c.shaping?;
    Some(Level::of(c.weight.unwrap_or(0.0) * sign(s), &SHAPING))
}

/// Turns the clause's shaping off (`None`) or on at a level. A ramp starts at the range's low
/// edge and ends a range's width past its high one.
pub fn set_shaping(c: &mut Clause, level: Option<Level>) {
    let Some(form) = shapable(c) else { return };
    let Some(level) = level else {
        (c.shaping, c.weight, c.ramp, c.term) = (None, None, None, None);
        return;
    };
    c.shaping = Some(form);
    match level.value(&SHAPING) {
        Some(w) => c.weight = Some(w * sign(form)),
        None => {
            c.weight.get_or_insert(SHAPING[1] * sign(form));
        }
    }
    if form == Shaping::Ramp && c.ramp.is_none() {
        let [lo, hi] = c.range.unwrap_or_default();
        c.ramp = Some([lo, hi + (hi - lo)]);
    }
}

/// The success bonus as a level; `None` without one.
pub fn bonus(spec: &TaskSpec) -> Option<Level> {
    let v = spec.reward.as_ref()?.success?;
    Some(Level::of(v, &BONUS))
}

pub fn set_bonus(spec: &mut TaskSpec, level: Option<Level>) {
    let r = spec.reward.get_or_insert_with(RewardDoc::default);
    match level {
        None => r.success = None,
        Some(l) => r.success = l.value(&BONUS).or(r.success),
    }
    if spec.reward.as_ref() == Some(&RewardDoc::default()) {
        spec.reward = None;
    }
}

/// How strongly the 🎲 items are drawn.
pub fn strength(spec: &TaskSpec) -> Level {
    match spec.start.as_ref().and_then(|s| s.strength) {
        None => Level::Medium,
        Some(v) if v.to_bits() == STRENGTH[0].to_bits() => Level::Low,
        Some(v) if v.to_bits() == STRENGTH[1].to_bits() => Level::High,
        Some(v) => Level::Custom(v),
    }
}

pub fn set_strength(spec: &mut TaskSpec, level: Level) {
    let v = match level {
        Level::Low => Some(STRENGTH[0]),
        Level::Medium => None,
        Level::High => Some(STRENGTH[1]),
        Level::Custom(_) => return,
    };
    if v.is_some() || spec.start.is_some() {
        spec.start.get_or_insert_with(Start::default).strength = v;
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

// --- what the scene offers ----------------------------------------------------------------

/// The bodies of the scene but the world, in scene order: what a body is measured against.
pub fn bodies(scene: &SceneDesc) -> Vec<String> {
    (scene.bodies.iter())
        .filter(|b| b.name != "world")
        .map(|b| b.name.clone())
        .collect()
}

fn free_bodies(scene: &SceneDesc) -> Vec<String> {
    let free = |id| (scene.joints.iter()).any(|j| j.body == id && j.kind == JointKind::Free);
    (scene.bodies.iter())
        .filter(|b| free(b.id))
        .map(|b| b.name.clone())
        .collect()
}

/// The names a slot offers: the subjects, the objects of the clause's relation (regions for
/// `inside`, bodies otherwise), or what a start item places (a body's orientation is its body).
pub fn choices(scene: &SceneDesc, field: Field, of: &Sentence) -> Vec<String> {
    match field {
        Field::Subject => vocab::subjects(scene),
        Field::Object if of.key == "author.task.inside_region" => vocab::regions(scene),
        Field::Object => bodies(scene),
        Field::What => {
            if matches!(of.slots.get(1), Some((Field::Draw, _))) {
                return free_bodies(scene);
            }
            let joints = (scene.joints.iter())
                .filter(|j| matches!(j.kind, JointKind::Hinge | JointKind::Slide))
                .map(|j| j.name.clone());
            let axes = free_bodies(scene)
                .into_iter()
                .flat_map(|b| ["x", "y", "z"].map(|a| format!("{b}.{a}")));
            joints.chain(axes).collect()
        }
        _ => Vec::new(),
    }
}

/// The sources a channel can observe: the robot's three, and a free body's pose, coordinates
/// and velocity.
pub fn sources(scene: &SceneDesc) -> Vec<String> {
    let robot = [
        "robot.joint_pos",
        "robot.joint_vel",
        "robot.previous_action",
    ];
    let mut out: Vec<String> = robot.map(str::to_owned).into();
    for b in free_bodies(scene) {
        out.extend(["pose", "qpos", "vel"].map(|f| format!("{b}.{f}")));
    }
    out
}

/// A scalar subject's place in the scene: a coordinate's, or a joint's zero.
fn here(scene: &SceneDesc, subject: &str) -> f64 {
    let Some((b, axis)) = subject.rsplit_once('.') else {
        return 0.0;
    };
    let Some(b) = scene.bodies.iter().find(|x| x.name == b) else {
        return 0.0;
    };
    let p = b.pose.position;
    match axis {
        "x" => p.x,
        "y" => p.y,
        "z" => p.z,
        _ => 0.0,
    }
}

/// Another body to measure against: a free one if there is one.
fn other(scene: &SceneDesc, subject: &str) -> Option<String> {
    let mut all = free_bodies(scene).into_iter().chain(bodies(scene));
    all.find(|b| b != subject)
}

fn bare(subject: String, relation: Relation) -> Clause {
    Clause {
        subject,
        relation,
        object: None,
        point: None,
        range: None,
        value: None,
        m: None,
        speed: None,
        angular: None,
        within_deg: None,
        shaping: None,
        weight: None,
        ramp: None,
        term: None,
    }
}

/// The clause with relation `r` (measured against something named when `object`): the fields
/// the new relation takes are kept where they fit, the ones it needs get defaults, and the
/// shaping stays when its form does.
pub fn set_relation(scene: &SceneDesc, c: &mut Clause, r: Relation, object: bool) {
    let old = std::mem::replace(c, bare(c.subject.clone(), r));
    let t = takes(r, object);
    let keeps = |f: &str| t.fields.contains(&f);
    let region = r == Relation::Inside;
    let fits = |o: &String| {
        if region {
            vocab::regions(scene).contains(o)
        } else {
            bodies(scene).contains(o)
        }
    };
    if keeps("object") && object {
        c.object = old.object.clone().filter(fits).or_else(|| {
            if region {
                vocab::regions(scene).into_iter().next()
            } else {
                other(scene, &c.subject)
            }
        });
    }
    if keeps("point") && old.point.is_some() {
        (c.object, c.point) = (None, old.point);
    }
    let here = here(scene, &c.subject);
    let defaults: [(&str, Option<f64>, f64); 4] = [
        ("value", old.value, here),
        ("speed", old.speed, 0.05),
        ("within_deg", old.within_deg, 10.0),
        (
            "m",
            old.m,
            if r == Relation::FartherThan {
                0.25
            } else {
                0.05
            },
        ),
    ];
    for (f, kept, default) in defaults {
        let v = if keeps(f) { kept } else { None };
        let v = v.or_else(|| t.needs.contains(&f).then_some(default));
        match f {
            "value" => c.value = v,
            "speed" => c.speed = v,
            "within_deg" => c.within_deg = v,
            _ => c.m = v,
        }
    }
    if keeps("range") {
        c.range = old.range.or(Some([here - 0.05, here + 0.05]));
    }
    if keeps("angular") {
        c.angular = old.angular;
    }
    if old.shaping.is_some() && old.shaping == t.shaping {
        (c.shaping, c.weight, c.ramp, c.term) = (old.shaping, old.weight, old.ramp, old.term);
    }
}

/// Changes what a slot of a clause says, with what that implies: a new subject keeps its
/// relation when it takes it (else takes the first it does), a new relation brings its fields,
/// a body for a fixed point (or the other way) swaps `object` and `point`.
pub fn edit_clause(scene: &SceneDesc, c: &mut Clause, field: Field, slot: Slot) {
    match (field, slot) {
        (Field::Subject, Slot::Name(n)) => {
            c.subject = n;
            let offered = vocab::relations(scene, &c.subject);
            if !offered.contains(&form(c)) {
                let (r, o) = offered.first().copied().unwrap_or((Relation::Still, false));
                set_relation(scene, c, r, o);
            }
        }
        (Field::Relation, Slot::Relation(r, o)) => set_relation(scene, c, r, o),
        (Field::Object, Slot::Name(n)) => (c.object, c.point) = (Some(n), None),
        (Field::Object, Slot::Point(p)) => (c.object, c.point) = (None, Some(p)),
        (field, Slot::Number(Some(v), _)) => match field {
            Field::Range(i) => c.range.get_or_insert([0.0; 2])[i.min(1)] = v,
            Field::Value => c.value = Some(v),
            Field::M => c.m = Some(v),
            Field::Speed => c.speed = Some(v),
            Field::Angular => c.angular = Some(v),
            Field::WithinDeg => c.within_deg = Some(v),
            _ => {}
        },
        _ => {}
    }
}

/// Changes what a slot of a start item says. A new way to draw an orientation brings its own
/// field (a tilt bound of 30° for `tilt`) and drops the other's.
pub fn edit_item(item: &mut StartItem, field: Field, slot: Slot) {
    match (field, slot) {
        (Field::What, Slot::Name(n)) if item.what.ends_with(".orientation") => {
            item.what = format!("{n}.orientation");
        }
        (Field::What, Slot::Name(n)) => item.what = n,
        (Field::Draw, Slot::Draw(d)) => {
            item.draw = Some(d);
            item.tilt = None;
            item.tilt_max_deg = (d == Draw::Tilt).then_some(30.0);
        }
        (field, Slot::Number(Some(v), _)) => match field {
            Field::Value => item.value = Some(v),
            Field::Noise => item.noise = Some(v),
            Field::Range(i) => item.range.get_or_insert([0.0; 2])[i.min(1)] = v,
            Field::Tilt => item.tilt = Some(v),
            Field::TiltMax => item.tilt_max_deg = Some(v),
            _ => {}
        },
        _ => {}
    }
}

// --- the specification's lists ------------------------------------------------------------

fn list(spec: &mut TaskSpec, failure: bool) -> &mut Vec<Clause> {
    if failure {
        &mut spec.failure.get_or_insert_with(Clauses::default).clauses
    } else {
        &mut spec.success.clauses
    }
}

fn tidy(spec: &mut TaskSpec) {
    if spec.failure.as_ref().is_some_and(|f| f.clauses.is_empty()) {
        spec.failure = None;
    }
    if spec.observe.as_ref() == Some(&Observe::default()) {
        spec.observe = None;
    }
    let start = spec.start.as_ref();
    if start.is_some_and(|s| s.items.is_empty() && s.strength.is_none()) {
        spec.start = None;
    }
}

/// A new clause: the first free body (or subject), still if it can be, else its first relation.
pub fn add_clause(spec: &mut TaskSpec, scene: &SceneDesc, failure: bool) {
    let subject = (free_bodies(scene).into_iter().next())
        .or_else(|| vocab::subjects(scene).into_iter().next())
        .unwrap_or_default();
    let offered = vocab::relations(scene, &subject);
    let (r, o) = (offered.iter())
        .find(|(r, _)| *r == Relation::Still)
        .or(offered.first())
        .copied()
        .unwrap_or((Relation::Still, false));
    let mut c = bare(subject, r);
    set_relation(scene, &mut c, r, o);
    list(spec, failure).push(c);
}

pub fn remove_clause(spec: &mut TaskSpec, at: At) {
    let l = list(spec, at.failure);
    if at.index < l.len() {
        l.remove(at.index);
    }
    tidy(spec);
}

/// Moves a clause one place up (`down = false`) or down in its section.
pub fn move_clause(spec: &mut TaskSpec, at: At, down: bool) {
    let l = list(spec, at.failure);
    let to = if down {
        at.index + 1
    } else {
        at.index.wrapping_sub(1)
    };
    if at.index < l.len() && to < l.len() {
        l.swap(at.index, to);
    }
    tidy(spec);
}

fn item(what: String) -> StartItem {
    StartItem {
        what,
        value: None,
        noise: None,
        range: None,
        draw: None,
        tilt: None,
        tilt_max_deg: None,
        coupled: None,
        dice: None,
        stream: None,
    }
}

/// A new 🎲 placement: the first free body's x where it stands, ± 2 cm.
pub fn add_item(spec: &mut TaskSpec, scene: &SceneDesc) {
    let Some(body) = free_bodies(scene).into_iter().next() else {
        return;
    };
    let what = format!("{body}.x");
    let it = StartItem {
        value: Some(here(scene, &what)),
        noise: Some(0.02),
        dice: Some(true),
        ..item(what)
    };
    spec.start.get_or_insert_with(Start::default).items.push(it);
}

pub fn remove_item(spec: &mut TaskSpec, index: usize) {
    if let Some(s) = spec.start.as_mut().filter(|s| index < s.items.len()) {
        s.items.remove(index);
    }
    tidy(spec);
}

/// The camera named `camera` observed or not (a new one appended, `camera_px` 96 if unset); a
/// camera no longer observed leaves the student's views too.
pub fn set_camera(spec: &mut TaskSpec, camera: &str, on: bool) {
    let o = spec.observe.get_or_insert_with(Observe::default);
    let list = o.cameras.get_or_insert_with(Vec::new);
    if on && !list.iter().any(|c| c == camera) {
        list.push(camera.to_owned());
        o.camera_px.get_or_insert(96);
    }
    if !on {
        list.retain(|c| c != camera);
        if let Some(v) = spec.student.as_mut().and_then(|s| s.views.as_mut()) {
            v.retain(|c| c != camera);
        }
    }
    if o.cameras.as_ref().is_some_and(Vec::is_empty) {
        o.cameras = None;
    }
    tidy(spec);
}

/// The look of the observed pictures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    /// The rasterizer (no `render`).
    Quick,
    /// The viewport's material look: the Task IR has no such sensor path yet (offered disabled).
    Material,
    /// The path tracer at `spp` samples.
    Traced,
}

pub fn look(spec: &TaskSpec) -> Look {
    let o = spec.observe.as_ref();
    match o.and_then(|o| o.render.as_ref()).and_then(|r| r.path) {
        Some(RenderPath::Pt) => Look::Traced,
        _ => Look::Quick,
    }
}

/// Quick drops `render`; traced starts at plan H's 32 samples, 3 bounces and a seed per tick.
pub fn set_look(spec: &mut TaskSpec, to: Look) {
    if to == look(spec) || to == Look::Material {
        return;
    }
    let o = spec.observe.get_or_insert_with(Observe::default);
    o.render = (to == Look::Traced).then(|| RenderDoc {
        path: Some(RenderPath::Pt),
        spp: Some(32),
        bounces: Some(3),
        seed: Some(SeedStream::Tick),
        ..RenderDoc::default()
    });
    tidy(spec);
}

/// Observes `source` on a new channel named after it (`cube.pose` → `cube_pose`), in the
/// privileged table (the teacher's alone) or the state one.
pub fn add_source(spec: &mut TaskSpec, source: &str, privileged: bool) {
    let o = spec.observe.get_or_insert_with(Observe::default);
    let base = source.trim_start_matches("robot.").replace('.', "_");
    let taken = |n: &str| {
        [&o.state, &o.privileged]
            .iter()
            .any(|t| t.as_ref().is_some_and(|t| t.contains_key(n)))
    };
    let name = crate::command::unique(&base, &taken);
    let table = if privileged {
        &mut o.privileged
    } else {
        &mut o.state
    };
    table
        .get_or_insert_with(Default::default)
        .insert(name, source.to_owned());
}

/// Stops observing `channel`; the teacher and the student stop reading it too.
pub fn remove_source(spec: &mut TaskSpec, channel: &str) {
    if let Some(o) = spec.observe.as_mut() {
        for t in [&mut o.state, &mut o.privileged] {
            if let Some(map) = t.as_mut() {
                map.remove(channel);
            }
            if t.as_ref().is_some_and(std::collections::BTreeMap::is_empty) {
                *t = None;
            }
        }
    }
    let lists = [
        spec.teacher.as_mut().and_then(|t| t.state.as_mut()),
        spec.student.as_mut().and_then(|s| s.state.as_mut()),
    ];
    for l in lists.into_iter().flatten() {
        l.retain(|c| c != channel);
    }
    tidy(spec);
}

/// Moves `channel` between the state table and the privileged one.
pub fn swap_source(spec: &mut TaskSpec, channel: &str) {
    let Some(o) = spec.observe.as_mut() else {
        return;
    };
    let from_state = o.state.as_ref().is_some_and(|t| t.contains_key(channel));
    let (from, to) = if from_state {
        (&mut o.state, &mut o.privileged)
    } else {
        (&mut o.privileged, &mut o.state)
    };
    let Some(source) = from.as_mut().and_then(|t| t.remove(channel)) else {
        return;
    };
    if from
        .as_ref()
        .is_some_and(std::collections::BTreeMap::is_empty)
    {
        *from = None;
    }
    to.get_or_insert_with(Default::default)
        .insert(channel.to_owned(), source);
}

// --- a new task ---------------------------------------------------------------------------

/// The control rate when nothing says one: SO-101's committed documents' (`task.toml`).
pub const CONTROL_HZ: f64 = 50.0;
/// "Say the task"'s time limit, seconds.
pub const TIMEOUT_S: f64 = 8.0;

/// The control rate of the Task IR at `task` (a template's), for "say the task" on its copy.
pub fn control_hz(task: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(task).ok()?;
    let task = es_ir::serial::task_from_toml(&text).ok()?;
    Some(f64::from(task.config.control_rate_hz))
}

/// What can be the robot: the scene document's includes, then the bodies hanging from the world
/// with a hinge or slide joint at or below them.
pub fn robots(scene: &SceneDesc, doc: &EsScene) -> Vec<String> {
    let mut out: Vec<String> = doc.includes.iter().map(|i| i.name.clone()).collect();
    let world = scene
        .bodies
        .iter()
        .find(|b| b.name == "world")
        .map(|b| b.id);
    for root in (scene.bodies.iter()).filter(|b| b.parent.is_none() || b.parent == world) {
        if Some(root.id) == world {
            continue;
        }
        // Bodies come parent first.
        let mut tree = std::collections::BTreeSet::from([root.id]);
        for b in &scene.bodies {
            if b.parent.is_some_and(|p| tree.contains(&p)) {
                tree.insert(b.id);
            }
        }
        let moves = |j: &&es_assets::scene::Joint| {
            tree.contains(&j.body) && matches!(j.kind, JointKind::Hinge | JointKind::Slide)
        };
        if scene.joints.iter().any(|j| moves(&j)) {
            out.push(root.name.clone());
        }
    }
    out
}

/// The robot of a new task: the scene document's first include, else the root of the body that
/// holds the first hinge or slide joint.
fn robot(scene: &SceneDesc, doc: &EsScene) -> String {
    if let Some(i) = doc.includes.first() {
        return i.name.clone();
    }
    let jointed =
        (scene.joints.iter()).find(|j| matches!(j.kind, JointKind::Hinge | JointKind::Slide));
    let mut at = jointed.and_then(|j| scene.bodies.iter().find(|b| b.id == j.body));
    // Bounded by the body count: an expanded scene has no cycle.
    for _ in 0..scene.bodies.len() {
        let up = (at.and_then(|b| b.parent)).and_then(|p| scene.bodies.iter().find(|b| b.id == p));
        match up {
            Some(p) if p.name != "world" => at = Some(p),
            _ => break,
        }
    }
    at.map(|b| b.name.clone()).unwrap_or_default()
}

/// "Say the task": the robot, `control_hz`, 8 s, every camera observed at 96 px, and one clause
/// a fresh scene can satisfy — the first free body still (under 5 cm/s). The person edits from
/// there.
pub fn new_spec(scene: &SceneDesc, doc: &EsScene, control_hz: f64) -> TaskSpec {
    let mut spec = TaskSpec {
        kind: "task-spec".to_owned(),
        schema: 1,
        scene: SCENE_FILE.to_owned(),
        robot: robot(scene, doc),
        control_hz,
        timeout_s: TIMEOUT_S,
        success: Clauses::default(),
        failure: None,
        start: None,
        observe: None,
        reward: None,
        teacher: None,
        student: None,
        deploy: None,
        evaluate: None,
        cycle: None,
    };
    add_clause(&mut spec, scene, false);
    for c in &scene.cameras {
        set_camera(&mut spec, &c.name, true);
    }
    spec
}

// --- the camera check ---------------------------------------------------------------------

/// The bodies the clauses are about, in clause order: a body subject, or a coordinate's body
/// (a joint is the robot's, which no camera check is for).
fn subject_bodies(spec: &TaskSpec, scene: &SceneDesc) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let all = spec
        .success
        .clauses
        .iter()
        .chain(spec.failure.iter().flat_map(|f| &f.clauses));
    for c in all {
        if scene.joints.iter().any(|j| j.name == c.subject) {
            continue;
        }
        let name = c
            .subject
            .rsplit_once('.')
            .map_or(c.subject.as_str(), |(b, _)| b);
        if scene.bodies.iter().any(|b| b.name == name) && !out.iter().any(|o| o == name) {
            out.push(name.to_owned());
        }
    }
    out
}

/// Puts a free body where the start places it: each coordinate an item gives a value for (or a
/// range, whose middle it takes). A free body hangs from the world, so its pose is the world's.
fn place(scene: &mut SceneDesc, spec: &TaskSpec, body: &str) {
    let items: Vec<&StartItem> = spec.start.iter().flat_map(|s| &s.items).collect();
    let Some(b) = scene.bodies.iter_mut().find(|b| b.name == body) else {
        return;
    };
    for (axis, a) in ["x", "y", "z"].iter().enumerate() {
        let Some(it) = items.iter().find(|i| i.what == format!("{body}.{a}")) else {
            continue;
        };
        let v = match (it.value, it.range) {
            (Some(v), _) => v,
            (None, Some([lo, hi])) => f64::midpoint(lo, hi),
            (None, None) => continue,
        };
        let p = &mut b.pose.position;
        match axis {
            0 => p.x = v,
            1 => p.y = v,
            _ => p.z = v,
        }
    }
}

/// Whether `eye` (a camera of vertical field of view `fovy` and a square picture) sees body
/// `body`: its origin inside the picture, and the first thing a ray from the eye to it hits
/// one of its own shapes — G6's picking rule (`nearest_hit_flat`), segmentation id by id.
fn sees(
    scene: &SceneDesc,
    tris: &TriScene,
    owner: &[StableId],
    eye: Pose,
    fovy: f64,
    body: &str,
) -> bool {
    let Some(b) = scene.bodies.iter().find(|b| b.name == body) else {
        return false;
    };
    let target = world(scene, b.id).position;
    // The camera's own frame, MJCF's: it looks along −Z with +Y up.
    let c = eye.inverse().transform_point(target);
    let half = es_math::approx::tan_f64(fovy / 2.0);
    if !(c.z < 0.0 && c.x.abs() <= -c.z * half && c.y.abs() <= -c.z * half) {
        return false;
    }
    let f = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32];
    let ray = target - eye.position;
    let Some(hit) = nearest_hit_flat(&tris.tris, f(eye.position), f(ray), 0.0, f32::INFINITY)
    else {
        return false;
    };
    let seg = tris.tris[hit.tri as usize].seg as usize;
    seg.checked_sub(1).and_then(|k| owner.get(k)) == Some(&b.id)
}

impl SceneModel {
    /// For each camera the specification observes, the clause subjects it does not see at
    /// their start positions (section 2's "the front camera does not see the cube at its start
    /// position"), in the observed order. Empty without a specification.
    // ponytail: the scene is tessellated per call; `es-editor` calls it once per revision.
    pub fn unseen(&self) -> Vec<(String, Vec<String>)> {
        let Some(spec) = self.spec() else {
            return Vec::new();
        };
        let mut scene = (**self.scene()).clone();
        let subjects = subject_bodies(spec, &scene);
        for s in &subjects {
            place(&mut scene, spec, s);
        }
        let Ok(tris) = TriScene::from_scene(&scene) else {
            return Vec::new();
        };
        let owner: Vec<StableId> = (scene.bodies.iter())
            .flat_map(|b| b.geoms.iter().map(move |_| b.id))
            .collect();
        let observed = spec.observe.as_ref().and_then(|o| o.cameras.clone());
        (observed.unwrap_or_default().into_iter())
            .filter_map(|name| {
                let cam = scene.cameras.iter().find(|c| c.name == name)?;
                let at = cam.body.map_or(Pose::IDENTITY, |b| world(&scene, b));
                let eye = at.compose(cam.pose);
                let missing = (subjects.iter())
                    .filter(|s| !sees(&scene, &tris, &owner, eye, cam.fovy, s))
                    .cloned()
                    .collect();
                Some((name, missing))
            })
            .collect()
    }
}
