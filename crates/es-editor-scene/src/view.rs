//! The viewport's headless half (packet M17/G6, `docs/design/scene-authoring.md` section 5): the
//! ray through a point of the picture, what a click there selects, where a thing sits, what the
//! selection draws (the viewport tints it) and the camera that frames it. `es-editor` turns the
//! pointer into points of the picture and these answers into lines; nothing here knows a device.
//!
//! **A pick is the segmentation channel at that point, read on the CPU.** The renderer gives
//! every drawn geom a segmentation id, its place in body-then-geom order (`es_render::scene`); a
//! click casts the ray the renderer casts through that point (`es_render::cpu::primary_dir`, here
//! in `f64`) at the same triangles with the renderer's own nearest-hit rule
//! (`es_render::cpu::nearest_hit_flat`: nearest, the lower index on a tie) and reads the id of
//! what it hits. That is the channel's value there without rendering a second channel or reading
//! one back from the device: the viewport draws `Rgb8` only, and a click is one ray, not a frame.

use std::collections::BTreeMap;

use es_assets::esscene::EsScene;
use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_math::approx::sin_cos_f64;
use es_math::{Pose, Quat, Vec3};
use es_render::cpu::nearest_hit_flat;
use es_render::TriScene;

pub use es_render::raster::Camera;

use crate::command::Entity;
use crate::model::SceneModel;
use crate::tree::Contents;

/// A ray in the world: where it starts and the way it goes (of any length).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

fn vec3(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// The ray from `camera`'s eye through `at`, a point of its picture in pixels (origin top left,
/// a pixel's centre at `+0.5`): the ray `es_render::cpu::primary_dir` casts, in `f64`. `None` for
/// a camera with no orientation (one looking straight up or down).
pub fn ray(camera: &Camera, at: [f64; 2]) -> Option<Ray> {
    let view = camera.view().ok()?;
    let k = view.spec.intrinsics;
    let d = Vec3::new(
        (at[0] - f64::from(k.cx)) / f64::from(k.fx),
        (at[1] - f64::from(k.cy)) / f64::from(k.fy),
        1.0,
    );
    Some(Ray {
        origin: view.pose.position,
        dir: view.pose.orientation.rotate(d),
    })
}

/// Where `p` lands in `camera`'s picture, in pixels; `None` at or behind its near plane.
pub fn project(camera: &Camera, p: Vec3) -> Option<[f64; 2]> {
    let view = camera.view().ok()?;
    let c = view.pose.inverse().transform_point(p);
    if c.z <= f64::from(view.spec.near) {
        return None;
    }
    let k = view.spec.intrinsics;
    Some([
        f64::from(k.fx) * c.x / c.z + f64::from(k.cx),
        f64::from(k.fy) * c.y / c.z + f64::from(k.cy),
    ])
}

/// `pos` and `quat` as the document writes them, as a pose: absent is the origin, unturned.
pub(crate) fn pose(pos: Option<[f64; 3]>, quat: Option<[f64; 4]>) -> Pose {
    let [x, y, z, w] = quat.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    Pose::new(vec3(pos.unwrap_or_default()), Quat::from_xyzw(x, y, z, w))
}

/// Body `id`'s world pose: its chain of poses from the world down, composed as the renderer
/// places a still scene's bodies.
fn world(scene: &SceneDesc, id: StableId) -> Pose {
    let mut chain = Vec::new();
    let mut at = Some(id);
    // Bounded by the body count: an expanded scene has no cycle, and this cannot hang on one.
    for _ in 0..=scene.bodies.len() {
        let Some(b) = at.and_then(|id| scene.bodies.iter().find(|b| b.id == id)) else {
            break;
        };
        chain.push(b.pose);
        at = b.parent;
    }
    chain
        .iter()
        .rev()
        .fold(Pose::IDENTITY, |acc, p| acc.compose(*p))
}

/// What a click on a geom selects, and for a body's own geom its place among the body's geoms.
type Owner = (Entity, Option<usize>);

/// The owner of every geom the document or an include put in `scene`, by geom id: a body's
/// geoms select the body, everything an include brings selects the include, a world geom of the
/// document is scenery and a light's panel is the light. The world body's geoms are in G1's
/// order (design section 3.4): the includes', then the document's scenery, then its lights.
fn owners(doc: &EsScene, includes: &[Contents], scene: &SceneDesc) -> BTreeMap<StableId, Owner> {
    let include = |brings: &dyn Fn(&Contents) -> bool| {
        (doc.includes.iter().zip(includes))
            .find(|(_, c)| brings(c))
            .map(|(i, _)| Entity::Include(i.name.clone()))
    };
    let mut out = BTreeMap::new();
    let world = &scene.bodies[0].geoms;
    let first = world
        .len()
        .saturating_sub(doc.geoms.len() + doc.lights.len());
    for (k, g) in world.iter().enumerate() {
        let e = match k.checked_sub(first) {
            None => include(&|c| c.scenery.contains(&g.name)),
            Some(i) if i < doc.geoms.len() => Some(Entity::Scenery(i)),
            Some(i) => (doc.lights.get(i - doc.geoms.len())).map(|l| Entity::Light(l.name.clone())),
        };
        out.extend(e.map(|e| (g.id, (e, None))));
    }
    for b in scene.bodies.iter().skip(1) {
        let own = doc.bodies.iter().any(|d| d.name == b.name);
        let e = if own {
            Some(Entity::Body(b.name.clone()))
        } else {
            include(&|c| c.bodies.iter().any(|(n, _)| *n == b.name))
        };
        let Some(e) = e else { continue };
        for (i, g) in b.geoms.iter().enumerate() {
            out.insert(g.id, (e.clone(), own.then_some(i)));
        }
    }
    out
}

/// Whether a geom owned by `owner` is part of `e`: its body's, its include's, or the geom itself.
fn part_of(owner: &Owner, e: &Entity) -> bool {
    match (e, owner) {
        (Entity::Geom { body, index }, (Entity::Body(b), Some(i))) => b == body && i == index,
        _ => owner.0 == *e,
    }
}

/// `camera` looking at the sphere (`centre`, `radius`) from the side it looks from now, near
/// enough that the sphere spans most of the picture's height (frame-selected). The viewport's
/// camera is the editor's, in no document; the sine is `es_math::approx`'s all the same.
pub fn frame(camera: &Camera, centre: Vec3, radius: f64) -> Camera {
    let back = vec3(camera.eye) - vec3(camera.look_at);
    let back = if back.norm() > 1e-9 {
        back.normalize()
    } else {
        Vec3::new(1.0, -1.0, 1.0).normalize()
    };
    let (sin, _) = sin_cos_f64(camera.fov_y * 0.5);
    let eye = centre + back.scale(radius / (0.8 * sin));
    Camera {
        eye: [eye.x, eye.y, eye.z],
        look_at: [centre.x, centre.y, centre.z],
        ..*camera
    }
}

impl SceneModel {
    /// The drawn scene's triangles, and the owner of each segmentation id (`seg - 1`).
    // ponytail: tessellated per call (a click, a new selection, frame-selected); keep it per
    // revision if picking on a large scene lags.
    fn tessellated(&self) -> Option<(TriScene, Vec<Option<Owner>>)> {
        let drawn = self.drawn();
        let tris = TriScene::from_scene(&drawn).ok()?;
        let owners = owners(self.doc(), self.contents(), self.scene());
        let by_seg = (drawn.bodies.iter().flat_map(|b| &b.geoms))
            .map(|g| owners.get(&g.id).cloned())
            .collect();
        Some((tris, by_seg))
    }

    /// What a click along `ray` selects: the owner of the nearest drawn geom it hits (a body, an
    /// include, a piece of scenery, a light), or `None` for empty space. A hidden thing is not
    /// drawn, so the ray goes on to what is behind it.
    pub fn hit(&self, ray: &Ray) -> Option<Entity> {
        let (tris, by_seg) = self.tessellated()?;
        let f = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32];
        let hit = nearest_hit_flat(&tris.tris, f(ray.origin), f(ray.dir), 0.0, f32::INFINITY)?;
        let seg = tris.tris[hit.tri as usize].seg as usize;
        by_seg
            .get(seg.checked_sub(1)?)?
            .as_ref()
            .map(|o| o.0.clone())
    }

    /// The world triangles `e` draws: what the viewport tints while it is selected.
    pub fn triangles(&self, e: &Entity) -> Vec<[[f32; 3]; 3]> {
        let Some((tris, by_seg)) = self.tessellated() else {
            return Vec::new();
        };
        let owner = |seg: u32| {
            (seg as usize)
                .checked_sub(1)
                .and_then(|k| by_seg.get(k)?.as_ref())
        };
        (tris.tris.iter())
            .filter(|t| owner(t.seg).is_some_and(|o| part_of(o, e)))
            .map(|t| t.v)
            .collect()
    }

    /// The sphere about what `e` draws, or a 10 cm one at its place when it draws nothing (a
    /// camera, a region, a hidden thing).
    pub fn bounds(&self, e: &Entity) -> Option<(Vec3, f64)> {
        let tris = self.triangles(e);
        if tris.is_empty() {
            let (parent, local) = self.placement(e)?;
            return Some((parent.compose(local).position, 0.1));
        }
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in tris.iter().flatten() {
            for k in 0..3 {
                (lo[k], hi[k]) = (lo[k].min(v[k]), hi[k].max(v[k]));
            }
        }
        let [lo, hi] = [lo, hi].map(|v| Vec3::new(v[0].into(), v[1].into(), v[2].into()));
        let half = (hi - lo).scale(0.5);
        Some((lo + half, half.norm().max(0.01)))
    }

    /// `camera` framing what `e` draws (frame-selected).
    pub fn framed(&self, e: &Entity, camera: &Camera) -> Option<Camera> {
        let (centre, radius) = self.bounds(e)?;
        Some(frame(camera, centre, radius))
    }

    /// The world frame `e` hangs from and its own place in it, as the document writes them (an
    /// absent `pos` is the origin, an absent `quat` no turn); `None` for what has no place.
    pub fn placement(&self, e: &Entity) -> Option<(Pose, Pose)> {
        let mut r = self.record(e)?;
        let (pos, quat) = r.pose_mut()?;
        let local = pose(*pos, *quat);
        let (doc, scene) = (self.doc(), self.scene());
        let body =
            |n: &str| (scene.bodies.iter().find(|b| b.name == n)).map(|b| world(scene, b.id));
        let parent = match e {
            Entity::Body(n) => {
                let b = scene.bodies.iter().find(|b| b.name == *n)?;
                b.parent.map(|p| world(scene, p))
            }
            Entity::Geom { body: n, .. } => body(n),
            Entity::Camera(n) => (doc.cameras.iter().find(|c| c.name == *n)?.parent)
                .as_deref()
                .and_then(body),
            Entity::Region(n) => (doc.regions.iter().find(|c| c.name == *n)?.parent)
                .as_deref()
                .and_then(body),
            _ => None,
        };
        Some((parent.unwrap_or(Pose::IDENTITY), local))
    }
}
