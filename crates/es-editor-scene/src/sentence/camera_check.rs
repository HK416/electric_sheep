//! The camera check: which bodies the clauses are about each observed camera does not see at
//! their start positions.

use es_assets::scene::{JointKind, SceneDesc};
use es_core::StableId;
use es_math::{Pose, Vec3};
use es_render::cpu::nearest_hit_flat;
use es_render::TriScene;
use es_script::spec::StartItem;

use super::TaskSpec;
use crate::model::SceneModel;
use crate::view::world;

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
        // A free joint named as its body is the body's (packet M17/G9), as the compiler reads it.
        let moves = |j: &&es_assets::scene::Joint| {
            j.name == c.subject && matches!(j.kind, JointKind::Hinge | JointKind::Slide)
        };
        if scene.joints.iter().any(|j| moves(&j)) {
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
