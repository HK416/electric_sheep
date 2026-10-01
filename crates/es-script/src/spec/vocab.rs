//! The vocabulary as the compiler reads it (packet M17/G8, design note sections 4.1, 4.3, 4.4):
//! which subjects a scene offers, which relations each takes, and which fields a relation takes.
//! The sentence editor offers only these; [`super::compile_task`] checks every clause by the
//! same [`takes`] table, so what is offered and what compiles cannot drift apart.

use es_assets::scene::{JointKind, SceneDesc};

use super::{Relation, Shaping};

/// What a relation takes: the fields it may carry, the ones it needs, and its shaping form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Takes {
    pub fields: &'static [&'static str],
    pub needs: &'static [&'static str],
    pub shaping: Option<Shaping>,
}

/// The fields of `relation`, measured against `object` (a body, a region) or against numbers.
pub fn takes(relation: Relation, object: bool) -> Takes {
    let (fields, needs, shaping): (&'static [&'static str], &'static [&'static str], _) =
        match relation {
            Relation::Inside if object => (&["object"], &[], None),
            Relation::Inside => (&["range"], &["range"], Some(Shaping::Ramp)),
            Relation::Above | Relation::Below if object => (&["object", "m"], &[], None),
            Relation::Above | Relation::Below => (&["value"], &["value"], None),
            Relation::Still => (&["speed", "angular"], &["speed"], None),
            Relation::Near | Relation::FartherThan => {
                (&["object", "point", "m"], &["m"], Some(Shaping::Distance))
            }
            Relation::OrientationMatches => (
                &["object", "within_deg"],
                &["object", "within_deg"],
                Some(Shaping::InverseAngle),
            ),
            Relation::Touches => (&[], &[], None),
        };
    Takes {
        fields,
        needs,
        shaping,
    }
}

fn free(scene: &SceneDesc, body: &str) -> bool {
    let Some(b) = scene.bodies.iter().find(|b| b.name == body) else {
        return false;
    };
    (scene.joints.iter()).any(|j| j.body == b.id && j.kind == JointKind::Free)
}

/// What a clause may be about: every body but the world, every hinge and slide joint, and the
/// coordinates `<body>.x|y|z` of every free body, in scene order.
pub fn subjects(scene: &SceneDesc) -> Vec<String> {
    let bodies = scene.bodies.iter().filter(|b| b.name != "world");
    let mut out: Vec<String> = bodies.clone().map(|b| b.name.clone()).collect();
    out.extend(
        (scene.joints.iter())
            .filter(|j| matches!(j.kind, JointKind::Hinge | JointKind::Slide))
            .map(|j| j.name.clone()),
    );
    for b in bodies.filter(|b| free(scene, &b.name)) {
        out.extend(["x", "y", "z"].map(|a| format!("{}.{a}", b.name)));
    }
    out
}

/// The relations a clause about `subject` compiles with, each with whether it is measured
/// against something named (`object`: a body, a region) rather than numbers; `touches` is
/// listed for a body though the compiler refuses it until `GetContact` lowers. A joint or a
/// coordinate is a scalar (`inside` a range, `above` / `below` a value, `still`); a body is
/// measured against bodies and regions. A name that is a joint and a body (SO-101's `gripper`)
/// is read as the joint wherever both could be, as the compiler reads it.
pub fn relations(scene: &SceneDesc, subject: &str) -> Vec<(Relation, bool)> {
    let joint = (scene.joints.iter())
        .any(|j| j.name == subject && matches!(j.kind, JointKind::Hinge | JointKind::Slide));
    let coordinate = subject.rsplit_once('.').and_then(|(b, axis)| {
        let body = scene.bodies.iter().any(|x| x.name == b);
        (body && ["x", "y", "z"].contains(&axis)).then(|| free(scene, b))
    });
    let body = scene.bodies.iter().any(|b| b.name == subject);
    let mut out = Vec::new();
    if joint || coordinate.is_some() {
        out.extend([Relation::Inside, Relation::Above, Relation::Below].map(|r| (r, false)));
        if joint || coordinate == Some(true) {
            out.push((Relation::Still, false));
        }
    } else if body {
        out.extend([Relation::Inside, Relation::Above, Relation::Below].map(|r| (r, true)));
    }
    if body {
        out.extend([Relation::Near, Relation::FartherThan].map(|r| (r, true)));
        if !joint && free(scene, subject) {
            out.push((Relation::Still, false));
        }
        out.extend([Relation::OrientationMatches, Relation::Touches].map(|r| (r, true)));
    }
    out
}

/// The regions a body can be `inside`: every site on a body that does not move (no joint on it
/// or above it), in scene order. Whether it is unrotated the compiler says.
pub fn regions(scene: &SceneDesc) -> Vec<String> {
    let moves = |mut id| {
        // Bounded by the body count: an expanded scene has no cycle.
        for _ in 0..=scene.bodies.len() {
            if scene.joints.iter().any(|j| j.body == id) {
                return true;
            }
            match scene
                .bodies
                .iter()
                .find(|b| b.id == id)
                .and_then(|b| b.parent)
            {
                Some(p) => id = p,
                None => return false,
            }
        }
        false
    };
    (scene.bodies.iter())
        .filter(|b| !moves(b.id))
        .flat_map(|b| b.sites.iter().map(|s| s.name.clone()))
        .collect()
}
