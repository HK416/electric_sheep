//! Cameras and regions in the viewport (packet M17/G7): neither draws anything in the scene, so
//! the viewport draws each as lines — a camera's frustum, with a tick on its picture's top edge,
//! and a region's box — and a click within a few pixels of those lines picks it, before the ray
//! looks for a drawn geom. Selected, they get G6's handles: move and turn, and a region its size.

// Geometry's own letters: corners `k`, a tick `p`, sine and cosine `s c`, half-sizes `h`.
#![allow(clippy::many_single_char_names)]

use es_math::approx::sin_cos_f64;
use es_math::units::DEG_TO_RAD;
use es_math::{Pose, Vec3};

use crate::command::{Entity, Record};
use crate::gizmo::distance;
use crate::inspect::{FOVY, REGION};
use crate::model::SceneModel;
use crate::view::{project, Camera};

/// A frustum's depth as a share of its distance from the eye: one size on screen, as a handle.
const DEPTH: f64 = 0.12;

/// A line in the picture, pixels at both ends.
pub type Segment = [[f64; 2]; 2];

/// A camera at `at` seeing `fovy` degrees, drawn `depth` deep with a square picture.
fn frustum(at: Pose, fovy: f64, depth: f64) -> Vec<[Vec3; 2]> {
    let (s, c) = sin_cos_f64(fovy * DEG_TO_RAD * 0.5);
    let h = depth * s / c;
    // MJCF's camera frame: it looks along -Z, its picture's up is +Y.
    let p = |x: f64, y: f64| at.transform_point(Vec3::new(x, y, -depth));
    let k = [p(-h, -h), p(h, -h), p(h, h), p(-h, h)];
    let tick = p(0.0, 1.4 * h);
    let mut out: Vec<[Vec3; 2]> = k.iter().map(|&q| [at.position, q]).collect();
    out.extend((0..4).map(|i| [k[i], k[(i + 1) % 4]]));
    out.extend([[k[2], tick], [tick, k[3]]]);
    out
}

/// The twelve edges of the box of half-extents `h` at `at`.
fn cuboid(at: Pose, h: [f64; 3]) -> Vec<[Vec3; 2]> {
    let corner = |i: usize| {
        let s = |bit: usize, v: f64| if i & bit == 0 { -v } else { v };
        at.transform_point(Vec3::new(s(1, h[0]), s(2, h[1]), s(4, h[2])))
    };
    (0..8)
        .flat_map(|i| {
            [1, 2, 4]
                .into_iter()
                .filter(move |b| i & b == 0)
                .map(move |b| [corner(i), corner(i | b)])
        })
        .collect()
}

impl SceneModel {
    /// Every camera's and region's lines as `view` sees them, in pixels; a line with an end at
    /// or behind the eye is left out.
    pub fn markers(&self, view: &Camera) -> Vec<(Entity, Vec<Segment>)> {
        let eye = Vec3::new(view.eye[0], view.eye[1], view.eye[2]);
        let doc = self.doc();
        let cameras = doc.cameras.iter().map(|c| Entity::Camera(c.name.clone()));
        let regions = doc.regions.iter().map(|r| Entity::Region(r.name.clone()));
        cameras
            .chain(regions)
            .filter_map(|e| {
                let (parent, local) = self.placement(&e)?;
                let at = parent.compose(local);
                let lines = match self.record(&e)? {
                    Record::Camera(c) => {
                        let depth = (at.position - eye).norm() * DEPTH;
                        frustum(at, c.fovy.unwrap_or(FOVY), depth)
                    }
                    Record::Region(r) => cuboid(at, r.size.unwrap_or(REGION)),
                    _ => return None,
                };
                let px = (lines.into_iter())
                    .filter_map(|[a, b]| Some([project(view, a)?, project(view, b)?]))
                    .collect();
                Some((e, px))
            })
            .collect()
    }

    /// The camera or region whose lines pass nearest `at` (pixels), within `reach` pixels.
    pub fn marker_at(&self, view: &Camera, at: [f64; 2], reach: f64) -> Option<Entity> {
        let mut best: Option<(Entity, f64)> = None;
        for (e, lines) in self.markers(view) {
            for [a, b] in lines {
                let d = distance(at, a, b);
                if d <= reach && best.as_ref().is_none_or(|(_, x)| d < *x) {
                    best = Some((e.clone(), d));
                }
            }
        }
        best.map(|(e, _)| e)
    }
}
