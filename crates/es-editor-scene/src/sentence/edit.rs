//! What the scene offers a slot (its bodies, the names a slot takes, the sources a channel
//! observes), and what changing a slot implies for the rest of its clause or start item.

use es_assets::scene::{JointKind, SceneDesc};
use es_script::spec::vocab::takes;
use es_script::spec::StartItem;

use super::{form, set_shaping, vocab, Clause, Draw, Field, Level, Relation, Sentence, Slot};

/// The bodies of the scene but the world, in scene order: what a body is measured against.
pub fn bodies(scene: &SceneDesc) -> Vec<String> {
    (scene.bodies.iter())
        .filter(|b| b.name != "world")
        .map(|b| b.name.clone())
        .collect()
}

pub(super) fn free_bodies(scene: &SceneDesc) -> Vec<String> {
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
pub(super) fn here(scene: &SceneDesc, subject: &str) -> f64 {
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

pub(super) fn bare(subject: String, relation: Relation) -> Clause {
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
/// shaping stays when its form does. A newly picked `inside` a region with no shaping to keep
/// pays its distance to the region's centre at the medium level (design note section 4.7.1).
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
    } else if form(c) == (Relation::Inside, true) && form(&old) != form(c) {
        set_shaping(c, Some(Level::Medium));
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
