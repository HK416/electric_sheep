//! Move, turn and size handles (packet M17/G6, `docs/design/scene-authoring.md` section 5): where
//! each handle is in the picture, which one the pointer is on, and what a drag of it writes —
//! **one [`Command`] when it is let go**, so a drag is one undo step.
//!
//! **Frames.** Move and turn work in the frame the document writes the thing's place in (its
//! parent's; the world's for most things), so a move along a handle changes the one coordinate
//! of `pos` it names and nothing else, and a turn is about that frame's axis through the thing's
//! own origin. Size works in the shape's own frame, where its sizes are measured.
//!
//! **Snapping** rounds in the document's units: a moved coordinate to a whole centimetre (the
//! number written is `0.87`, not `0.87000000000000001`), a turn to a multiple of 15 degrees, a
//! size — the whole width, height or diameter the inspector shows — to a whole centimetre.
//!
//! **Determinism** (spec 3.2): a turn's angle is measured with `es_math::approx`'s `acos_f64`
//! (the inspector's `atan2`) and the quaternion written is built with its `sin_cos_f64`;
//! everything else is `+ - * / sqrt`. So what a drag writes does not depend on the host's `libm`.

// Geometry's own letters: points `p`, `a`, `b`, a ray's `t`, a quaternion's `x y z w`.
#![allow(clippy::many_single_char_names)]

use es_assets::esscene::ShapeDoc;
use es_math::approx::sin_cos_f64;
use es_math::units::{DEG_TO_RAD, RAD_TO_DEG};
use es_math::{Pose, Quat, Vec3};

use crate::command::{Command, Entity, Record};
use crate::euler::atan2;
use crate::model::SceneModel;
use crate::view::{pose, project, Camera, Ray};
use crate::{import, inspect};

/// Which handles the selection shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Move,
    Rotate,
    Scale,
}

impl Tool {
    pub const ALL: [Self; 3] = [Self::Move, Self::Rotate, Self::Scale];

    /// The key of its word in the editor's string tables.
    pub fn key(self) -> &'static str {
        match self {
            Self::Move => "author.tool.move",
            Self::Rotate => "author.tool.rotate",
            Self::Scale => "author.tool.scale",
        }
    }
}

/// A snapped length is a whole number of these per metre: centimetres.
const PER_M: f64 = 100.0;
/// A snapped turn is a multiple of this many degrees.
pub const SNAP_DEG: f64 = 15.0;
/// A handle's length as a share of its distance from the eye: one size on screen anywhere.
const SHARE: f64 = 0.15;
/// Points on a turn handle's ring.
const RING: usize = 48;

/// `v` metres on the nearest whole centimetre, as the document then writes it.
pub fn snap_cm(v: f64) -> f64 {
    (v * PER_M).round() / PER_M
}

fn unit(k: usize) -> Vec3 {
    let mut v = [0.0; 3];
    v[k] = 1.0;
    Vec3::new(v[0], v[1], v[2])
}

/// One entity's handles: three axes from an origin, each a line (move, size) or a ring (turn).
#[derive(Clone, Debug, PartialEq)]
pub struct Gizmo {
    pub tool: Tool,
    pub origin: Vec3,
    /// Unit axes in the world: of the frame the document writes the place in (move, turn), or of
    /// the shape (size).
    pub axes: [Vec3; 3],
    /// Which axes have a handle: all for move and turn, those a size follows for size.
    pub on: [bool; 3],
    /// A handle's length, metres.
    pub size: f64,
}

impl Gizmo {
    /// Each handle as a line in `camera`'s picture, in pixels: an axis from the origin out (move,
    /// size) or a ring about it (turn). A point at or behind the eye is left out.
    pub fn lines(&self, camera: &Camera) -> Vec<(usize, Vec<[f64; 2]>)> {
        let line = |k: usize| -> Vec<Vec3> {
            if self.tool != Tool::Rotate {
                return vec![self.origin, self.origin + self.axes[k].scale(self.size)];
            }
            let (u, v) = (self.axes[(k + 1) % 3], self.axes[(k + 2) % 3]);
            (0..=RING)
                .map(|i| {
                    let (s, c) = sin_cos_f64(std::f64::consts::TAU * i as f64 / RING as f64);
                    self.origin + (u.scale(c) + v.scale(s)).scale(self.size)
                })
                .collect()
        };
        (0..3)
            .filter(|&k| self.on[k])
            .map(|k| {
                let points = line(k).into_iter().filter_map(|p| project(camera, p));
                (k, points.collect())
            })
            .collect()
    }

    /// The handle nearest `at` (pixels) within `reach` pixels of it.
    pub fn handle(&self, camera: &Camera, at: [f64; 2], reach: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (k, line) in self.lines(camera) {
            for w in line.windows(2) {
                let d = distance(at, w[0], w[1]);
                if d <= reach && best.is_none_or(|(_, b)| d < b) {
                    best = Some((k, d));
                }
            }
        }
        best.map(|(k, _)| k)
    }
}

/// How far `p` is from the segment `a`-`b`.
pub(crate) fn distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (x, y) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    (x * x + y * y).sqrt()
}

/// The place along the line `o + a s` (`a` a unit) nearest `ray`; `None` when the ray runs along
/// the line.
fn along(o: Vec3, a: Vec3, ray: &Ray) -> Option<f64> {
    let w = o - ray.origin;
    let (b, c) = (a.dot(ray.dir), ray.dir.dot(ray.dir));
    let (d, e) = (a.dot(w), ray.dir.dot(w));
    let den = c - b * b;
    if den <= 1e-12 * c {
        return None;
    }
    Some(b * (e - b * d) / den - d)
}

/// Where `ray` meets the plane through `o` across `n`, from `o`; `None` when it runs along the
/// plane or meets it behind where it starts.
fn on_plane(o: Vec3, n: Vec3, ray: &Ray) -> Option<Vec3> {
    let den = n.dot(ray.dir);
    if den.abs() <= 1e-9 * ray.dir.norm() {
        return None;
    }
    let t = n.dot(o - ray.origin) / den;
    (t > 0.0).then(|| ray.origin + ray.dir.scale(t) - o)
}

/// What a size handle on `axis` changes of `s`: the index of the size in [`inspect::dims`] (for
/// a mesh, of its extent along that axis), and the shape's axes that size stretches. A mesh
/// grows the same along all three (packet M17/R3): its handles write one uniform factor.
fn shape_dim(s: &ShapeDoc, axis: usize) -> Option<(usize, [bool; 3])> {
    let one = |k: usize| [k == 0, k == 1, k == 2];
    match s {
        ShapeDoc::Box(_) | ShapeDoc::Ellipsoid(_) => Some((axis, one(axis))),
        ShapeDoc::Sphere(_) => Some((0, [true; 3])),
        ShapeDoc::Capsule(_) | ShapeDoc::Cylinder(_) if axis < 2 => Some((0, [true, true, false])),
        ShapeDoc::Capsule(_) | ShapeDoc::Cylinder(_) => Some((1, one(2))),
        ShapeDoc::Plane(_) => (axis < 2).then(|| (axis, one(axis))),
        ShapeDoc::Mesh { .. } => Some((axis, [true; 3])),
    }
}

/// Where the shape of a record sits in its entity's frame and which axes can be sized; `None`
/// for what has no size (a camera, an include, a body of several geoms).
fn sizes(r: &Record) -> Option<(Pose, [bool; 3])> {
    let on = |s: &ShapeDoc| Some([0, 1, 2].map(|k| shape_dim(s, k).is_some())).filter(|v| v[0]);
    match r {
        Record::Geom(g) | Record::Scenery(g) => Some((Pose::IDENTITY, on(&g.shape)?)),
        Record::Body(b) => match b.geoms.as_slice() {
            [g] => Some((pose(g.pos, g.quat), on(&g.shape)?)),
            _ => None,
        },
        Record::Region(_) => Some((Pose::IDENTITY, [true; 3])),
        Record::Light(_) => Some((Pose::IDENTITY, [true, true, false])),
        _ => None,
    }
}

/// A handle held: the entity, the gizmo as it was grabbed, the ray it was grabbed on, and the
/// entity's place and fields at that moment.
#[derive(Clone, Debug, PartialEq)]
pub struct Drag {
    pub entity: Entity,
    pub gizmo: Gizmo,
    pub axis: usize,
    start: Ray,
    pos: [f64; 3],
    quat: [f64; 4],
    record: Record,
    /// The whole extent of the record's mesh along each axis as it is drawn now; `None` when it
    /// has no mesh, or its file does not read.
    mesh: Option<[f64; 3]>,
}

/// How far a drag has gone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// The handle's coordinate of `pos` (metres, in the parent's frame) becomes `to`.
    Move { from: f64, to: f64 },
    /// A turn about the handle's axis, radians.
    Rotate { angle: f64 },
    /// The whole extent the handle sizes (a width, a height, a diameter, metres) becomes `to`.
    Scale { from: f64, to: f64 },
}

impl SceneModel {
    /// `e`'s handles for `tool` as `camera` sees them; `None` for what has no place, or for size
    /// on what has no size.
    pub fn gizmo(&self, e: &Entity, tool: Tool, camera: &Camera) -> Option<Gizmo> {
        let (parent, local) = self.placement(e)?;
        let at = parent.compose(local);
        let (frame, on) = match tool {
            Tool::Scale => {
                let (shape, on) = sizes(&self.record(e)?)?;
                (at.compose(shape), on)
            }
            _ => (
                Pose {
                    position: at.position,
                    orientation: parent.orientation,
                },
                [true; 3],
            ),
        };
        let eye = Vec3::new(camera.eye[0], camera.eye[1], camera.eye[2]);
        Some(Gizmo {
            tool,
            origin: frame.position,
            axes: [0, 1, 2].map(|k| frame.orientation.rotate(unit(k))),
            on,
            size: (frame.position - eye).norm() * SHARE,
        })
    }

    /// `gizmo`'s handle `axis` grabbed on `start`: the drag of `e` that follows.
    pub fn drag(&self, e: &Entity, gizmo: Gizmo, axis: usize, start: Ray) -> Option<Drag> {
        let mut record = self.record(e)?;
        let (pos, quat) = record.pose_mut()?;
        let (pos, quat) = (
            pos.unwrap_or_default(),
            quat.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        );
        let shape = match &record {
            Record::Geom(g) | Record::Scenery(g) => Some(&g.shape),
            Record::Body(b) => b.geoms.first().map(|g| &g.shape),
            _ => None,
        };
        let mesh = match shape {
            Some(s @ ShapeDoc::Mesh { file, .. }) => {
                let size = import::mesh_size(&self.root().join(file)).ok();
                let scale = inspect::scale(s).unwrap_or([1.0; 3]);
                size.map(|e| [0, 1, 2].map(|k| e[k] * scale[k]))
            }
            _ => None,
        };
        Some(Drag {
            entity: e.clone(),
            gizmo,
            axis,
            start,
            pos,
            quat,
            record,
            mesh,
        })
    }
}

impl Drag {
    /// The whole extent this drag's size handle sizes, and the axes that stretch with it.
    fn sized(&self) -> Option<(f64, [bool; 3])> {
        let k = self.axis;
        let shape = |s: &ShapeDoc| {
            let (i, axes) = shape_dim(s, k)?;
            match s {
                ShapeDoc::Mesh { .. } => Some((self.mesh?[i], axes)),
                _ => Some((inspect::dims(s)[i].1, axes)),
            }
        };
        let one = [k == 0, k == 1, k == 2];
        match &self.record {
            Record::Geom(g) | Record::Scenery(g) => shape(&g.shape),
            Record::Body(b) => shape(&b.geoms.first()?.shape),
            Record::Region(r) => Some((2.0 * r.size.unwrap_or(inspect::REGION)[k], one)),
            Record::Light(l) => (k < 2).then(|| (2.0 * l.size[k], one)),
            _ => None,
        }
    }

    /// How far the pointer on `now` has dragged, snapped when `snap` is on; `None` while the ray
    /// gives no answer (it runs along the handle).
    pub fn step(&self, now: &Ray, snap: bool) -> Option<Step> {
        let g = &self.gizmo;
        let a = g.axes[self.axis];
        Some(match g.tool {
            Tool::Move => {
                let d = along(g.origin, a, now)? - along(g.origin, a, &self.start)?;
                let from = self.pos[self.axis];
                let to = if snap { snap_cm(from + d) } else { from + d };
                Step::Move { from, to }
            }
            Tool::Rotate => {
                let (p0, p1) = (
                    on_plane(g.origin, a, &self.start)?,
                    on_plane(g.origin, a, now)?,
                );
                let angle = atan2(a.dot(p0.cross(p1)), p0.dot(p1));
                let angle = if snap {
                    (angle * RAD_TO_DEG / SNAP_DEG).round() * SNAP_DEG * DEG_TO_RAD
                } else {
                    angle
                };
                Step::Rotate { angle }
            }
            Tool::Scale => {
                let s0 = along(g.origin, a, &self.start)?;
                let s1 = along(g.origin, a, now)?;
                let from = self.sized()?.0;
                if s0.abs() < 1e-9 || from <= 0.0 {
                    return None;
                }
                let to = from * (s1 / s0);
                let to = if snap {
                    snap_cm(to).max(1.0 / PER_M)
                } else {
                    to.max(1e-3)
                };
                Step::Scale { from, to }
            }
        })
    }

    /// The one command the drag makes when let go on `now`; `None` when it changes nothing.
    #[allow(clippy::float_cmp)] // "changes nothing" is bit equality
    pub fn command(&self, now: &Ray, snap: bool) -> Option<Command> {
        let entity = self.entity.clone();
        match self.step(now, snap)? {
            Step::Move { from, to } => {
                let mut pos = self.pos;
                pos[self.axis] = to;
                (to != from).then_some(Command::SetPose {
                    entity,
                    pos,
                    quat: self.quat,
                })
            }
            Step::Rotate { angle } => {
                if angle == 0.0 {
                    return None;
                }
                // About the parent frame's axis, through the thing's own origin: applied after
                // its own turn.
                let (s, c) = sin_cos_f64(angle * 0.5);
                let e = unit(self.axis).scale(s);
                let [x, y, z, w] = self.quat;
                let q = Quat::from_xyzw(e.x, e.y, e.z, c) * Quat::from_xyzw(x, y, z, w);
                let q = q.normalize();
                Some(Command::SetPose {
                    entity,
                    pos: self.pos,
                    quat: [q.x, q.y, q.z, q.w],
                })
            }
            Step::Scale { from, to } => {
                if to == from {
                    return None;
                }
                let k = self.axis;
                let resize = |s: &mut ShapeDoc| {
                    // A mesh's extent goes from `from` to `to` by its scale, uniformly.
                    if let Some(v) = inspect::scale(s) {
                        *s = inspect::with_scale(s, v.map(|x| x * to / from));
                        return Some(());
                    }
                    let (i, _) = shape_dim(s, k)?;
                    let mut v: Vec<f64> = inspect::dims(s).iter().map(|d| d.1).collect();
                    v[i] = to;
                    *s = inspect::with_dims(s, &v);
                    Some(())
                };
                let mut r = self.record.clone();
                match &mut r {
                    Record::Geom(g) | Record::Scenery(g) => resize(&mut g.shape)?,
                    Record::Body(b) => resize(&mut b.geoms.first_mut()?.shape)?,
                    Record::Region(x) => {
                        let mut s = x.size.unwrap_or(inspect::REGION);
                        s[k] = to / 2.0;
                        x.size = Some(s);
                    }
                    Record::Light(l) => l.size[k] = to / 2.0,
                    _ => return None,
                }
                Some(Command::Set(entity, r))
            }
        }
    }

    /// Where the point `p` of what is dragged is drawn after `step`: moved along the handle,
    /// turned about it, or stretched along the axes its size follows, about the gizmo's origin.
    pub fn moved(&self, step: Step, p: Vec3) -> Vec3 {
        let g = &self.gizmo;
        let a = g.axes[self.axis];
        let r = p - g.origin;
        match step {
            Step::Move { from, to } => p + a.scale(to - from),
            Step::Rotate { angle } => {
                let (s, c) = sin_cos_f64(angle * 0.5);
                g.origin + Quat::from_xyzw(a.x * s, a.y * s, a.z * s, c).rotate(r)
            }
            Step::Scale { from, to } => {
                let stretched = self.sized().map_or([false; 3], |x| x.1);
                let f = |j: usize| if stretched[j] { to / from } else { 1.0 };
                (0..3).fold(g.origin, |acc, j| {
                    acc + g.axes[j].scale(r.dot(g.axes[j]) * f(j))
                })
            }
        }
    }
}
