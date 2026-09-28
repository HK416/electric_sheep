//! The CPU replay rasterizer: a `TriScene` seen from a free [`Camera`], drawn with no GPU
//! (packets M7/E2 and M7/E8; moved here from `es-editor` by packet M12/Y3).
//!
//! [`project_scene`] projects the triangles to the screen and flat-shades each with the `Rs`
//! path's own Lambert; [`Raster::draw`] rasterises them with a per-pixel depth test. The
//! editor's Replay panel feeds it a scene re-posed from an `.estraj` and uploads the result as
//! one texture.
//!
//! The depth buffer is packet M7/E8. E2 sorted whole triangles by centroid depth and painted
//! them in order, which is exact for convex primitives that do not interpenetrate -- and the
//! SO-101 is links that interpenetrate at every joint, so a base plate drew over the shoulder
//! and a finger through the wrist. The sort stays, because it is what fixes the paint order
//! and therefore the tie-break, but it no longer decides what is visible.

use es_math::{Pose, Quat, Vec3};
use thiserror::Error;

use crate::cpu::{face_forward, quat_rotate_inv, shade_lambert, to_u8};
use crate::scene::{Tri, TriScene};
use crate::view::{CameraView, ImageSpec, RenderConfig, TileAtlasCfg, ViewParams};

/// What stops a frame from being drawn.
#[derive(Debug, Error)]
pub enum RasterError {
    /// A camera whose direction has no roll-free orientation (see [`Camera::view`]).
    #[error("{0}")]
    Camera(String),
}

/// The free camera of `es video showcase`: in no scene and in no document (spec 15.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub eye: [f64; 3],
    pub look_at: [f64; 3],
    /// Vertical field of view, radians.
    pub fov_y: f64,
    pub width: u32,
    pub height: u32,
}

/// How close to the pole [`Camera::orbit`] lets the pitch get. At the pole the view direction
/// is parallel to world up and [`Camera::view`] has no roll-free answer, so the orbit stops
/// just short of it rather than failing there.
const PITCH_LIMIT: f64 = 1.552; // ~88.9 degrees

impl Camera {
    /// `T_world_camera` and the pinhole `ImageSpec`, in the `OpenCV` frame of spec 3.1.
    ///
    /// The arithmetic of `es_env::render::look_at`, which is layer 9 and behind Vulkan; this
    /// crate is layer 5, so it is repeated here over the same types and pinned against
    /// [`crate::cpu::rasterize`] by `flat_shade_matches_the_renderer`.
    pub fn view(&self) -> Result<CameraView, RasterError> {
        let (eye, target) = (vec3(self.eye), vec3(self.look_at));
        let forward = (target - eye).normalize();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let right = forward.cross(up);
        if !forward.norm().is_finite() || right.norm() < 1e-9 {
            return Err(RasterError::Camera(format!(
                "camera at {:?} looking at {:?}: the view direction is degenerate or parallel \
                 to world up (+Z), so there is no roll-free orientation",
                self.eye, self.look_at
            )));
        }
        let right = right.normalize();
        let down = forward.cross(right);
        Ok(CameraView {
            pose: Pose::new(eye, quat_from_basis(right, down, forward)),
            spec: ImageSpec::pinhole(self.width, self.height, self.fov_y),
        })
    }

    /// Moves the eye on the sphere about `look_at`. Pure: the caller keeps the result or does
    /// not.
    ///
    /// `f64::atan2`/`asin`/`sin`/`cos` rather than `es_math::approx` (spec 3.4): a camera the
    /// user drags is UI state, not a kernel, and no golden is a function of it -- the sort-order
    /// golden is taken at a camera built from literal coordinates.
    #[must_use]
    pub fn orbit(mut self, delta_yaw: f64, delta_pitch: f64) -> Self {
        let d = vec3(self.eye) - vec3(self.look_at);
        let r = d.norm();
        if r < 1e-9 {
            return self;
        }
        let yaw = d.y.atan2(d.x) + delta_yaw;
        let pitch = ((d.z / r).asin() + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.eye = [
            self.look_at[0] + r * pitch.cos() * yaw.cos(),
            self.look_at[1] + r * pitch.cos() * yaw.sin(),
            self.look_at[2] + r * pitch.sin(),
        ];
        self
    }

    /// Moves the eye towards (`factor < 1`) or away from (`factor > 1`) `look_at`. Pure.
    #[must_use]
    pub fn zoom(mut self, factor: f64) -> Self {
        let d = vec3(self.eye) - vec3(self.look_at);
        let r = (d.norm() * factor).clamp(1e-3, 1e4);
        let d = d.normalize().scale(r);
        self.eye = [
            self.look_at[0] + d.x,
            self.look_at[1] + d.y,
            self.look_at[2] + d.z,
        ];
        self
    }
}

/// One triangle ready to draw: screen-space points, the camera-space depth of each of them,
/// one flat colour, and the centroid depth it was sorted by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tri2d {
    /// Pixel coordinates, origin top-left (spec 3.1). Off-screen values are kept: clipping
    /// sideways is the painter's job, and `egui` scissors the canvas anyway.
    pub p: [[f32; 2]; 3],
    /// Camera-space depth of each vertex, metres, in `p`'s order: what [`Raster::draw`]
    /// interpolates per pixel.
    pub z: [f32; 3],
    /// sRGB, the `Rgb8` the renderer would write for this surface.
    pub color: [u8; 3],
    /// Camera-space depth of the centroid, metres. The paint order's key, not the depth test's.
    pub depth: f32,
    /// Index into the tick's `TriScene::tris`, so the emitted order is itself checkable.
    pub index: u32,
}

/// What one tick looks like from one camera: back to front, so painting in order is correct.
pub type Projected = Vec<Tri2d>;

/// The canvas behind the scene, as a grey level. The raster carries its own background, so
/// the panel is one texture and not a texture over a rectangle.
pub const BACKGROUND: u8 = 18;

/// What one frame may cost: a 960 x 540 raster's worth of pixels, whatever shape the panel
/// is. Above it a frame costs more than it shows, and the picture is scaled to fill the panel
/// either way.
pub const MAX_PIXELS: f32 = 960.0 * 540.0;

/// One drawn frame: `Rgb8` pixels and the camera-space depth each of them came from.
///
/// A pure function of ([`Projected`], `w`, `h`) -- triangles in the order they arrive, pixels
/// row-major, `f32` throughout, no threads and no atomics -- so the same tick from the same
/// camera is the same bytes, which is what makes a golden of it worth having (spec 1.4).
#[derive(Clone, Debug, PartialEq)]
pub struct Raster {
    pub w: u32,
    pub h: u32,
    /// Row-major, tightly packed, `w * h * 3` bytes.
    pub rgb: Vec<u8>,
    /// Row-major, `w * h`; `f32::INFINITY` where nothing was drawn.
    pub depth: Vec<f32>,
}

impl Raster {
    /// The raster size for a panel `points` big: its own resolution, scaled down *uniformly*
    /// until it fits within [`MAX_PIXELS`].
    ///
    /// Uniformly, because the picture is stretched back over the whole panel and
    /// `ImageSpec::pinhole` has square pixels: scaling both axes by one factor is the same
    /// view at a different resolution, while capping them independently would squash it. The
    /// budget is on the area and not on each axis for the same reason -- the Replay panel is
    /// wide and short (roughly 1900 x 280 on a maximised window), and a per-axis cap would
    /// spend a quarter of the budget it is allowed and show a quarter of the detail. A
    /// degenerate or non-finite size is one pixel rather than zero: `egui` hands out a
    /// collapsed panel while the window is being dragged, and a zero-sized texture is a
    /// device error.
    #[must_use]
    pub fn size_for(points: [f32; 2]) -> (u32, u32) {
        // `max(1.0)` first: it also turns a NaN into 1.0, since NaN loses every comparison.
        let (w, h) = (points[0].max(1.0), points[1].max(1.0));
        let scale = (MAX_PIXELS / (w * h)).sqrt().min(1.0);
        (((w * scale) as u32).max(1), ((h * scale) as u32).max(1))
    }

    /// Draws `projected` into a `w` x `h` frame with a per-pixel depth test.
    #[must_use]
    pub fn draw(projected: &Projected, w: u32, h: u32) -> Self {
        let (w, h) = (w.max(1), h.max(1));
        let pixels = w as usize * h as usize;
        let mut raster = Self {
            w,
            h,
            rgb: vec![BACKGROUND; pixels * 3],
            depth: vec![f32::INFINITY; pixels],
        };
        for tri in projected {
            raster.triangle(tri);
        }
        raster
    }

    /// One triangle, by edge functions over its bounding box. Both faces are drawn: the
    /// scene's tessellation has no guaranteed winding, and dividing the three edge functions
    /// by the signed area flips all three signs with it, so "inside" is one test either way.
    fn triangle(&mut self, tri: &Tri2d) {
        let [a, b, c] = tri.p;
        let area = edge(a, b, c);
        if area == 0.0 || !area.is_finite() {
            return;
        }
        let inv_area = 1.0 / area;
        // Screen space is linear in 1/z, not in z: a triangle seen at a grazing angle is
        // metres wrong if the depths themselves are interpolated.
        let inv_z = tri.z.map(|z| 1.0 / z);
        let xs = span(a[0].min(b[0]).min(c[0]), a[0].max(b[0]).max(c[0]), self.w);
        let ys = span(a[1].min(b[1]).min(c[1]), a[1].max(b[1]).max(c[1]), self.h);
        for py in ys.clone() {
            for px in xs.clone() {
                // The pixel centre: the point `crate::cpu::primary_dir` casts through.
                let centre = [px as f32 + 0.5, py as f32 + 0.5];
                let bary = [
                    edge(b, c, centre) * inv_area,
                    edge(c, a, centre) * inv_area,
                    edge(a, b, centre) * inv_area,
                ];
                if bary[0] < 0.0 || bary[1] < 0.0 || bary[2] < 0.0 {
                    continue;
                }
                let depth = 1.0 / (bary[0] * inv_z[0] + bary[1] * inv_z[1] + bary[2] * inv_z[2]);
                let at = py as usize * self.w as usize + px as usize;
                // Strictly nearer, so a shared edge -- drawn twice, since the inside test
                // keeps both sides of it -- is owned by whichever triangle came first.
                if depth >= self.depth[at] {
                    continue;
                }
                self.depth[at] = depth;
                self.rgb[at * 3..at * 3 + 3].copy_from_slice(&tri.color);
            }
        }
    }
}

/// Twice the signed area of `(a, b, p)`: zero on the line `a -> b`, and of one sign on each
/// side of it.
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// The pixel range `lo..=hi` touches, clipped to `0..n`. `as u32` saturates and sends NaN to
/// zero, so an off-screen or degenerate span is an empty range and never an index panic.
fn span(lo: f32, hi: f32, n: u32) -> std::ops::Range<u32> {
    let first = (lo.floor().max(0.0) as u32).min(n);
    let last = (hi.ceil().max(0.0) as u32).min(n);
    first..last
}

/// The shading the `Rs` path uses, for its light direction and ambient floor. The atlas is
/// irrelevant here -- nothing here writes a tile -- but the constants must be the renderer's
/// own, and this is where they live.
fn shading() -> RenderConfig {
    RenderConfig::rs(TileAtlasCfg::row(1, 1, 1))
}

/// Project, shade, clip and sort: `scene` seen from `view`, back to front. A pure function of
/// its inputs, so a test can drive it from a hand-built `TriScene`.
pub fn project_scene(scene: &TriScene, view: &CameraView) -> Projected {
    let vp = ViewParams::new(view);
    let cfg = shading();
    let mut out: Projected = Vec::with_capacity(scene.tris.len());
    for (index, tri) in scene.tris.iter().enumerate() {
        let cam = tri.v.map(|v| {
            quat_rotate_inv(
                vp.quat,
                [v[0] - vp.pos[0], v[1] - vp.pos[1], v[2] - vp.pos[2]],
            )
        });
        // Near-plane clipping: a triangle with a vertex at or behind the eye has no screen
        // position for it, and dividing by that `z` would project it through the eye onto the
        // far side of the image. Whole triangles only -- one that straddles the plane
        // disappears instead of being split.
        if cam.iter().any(|c| c[2] <= vp.near) {
            continue;
        }
        let p = cam.map(|c| [vp.fx * c[0] / c[2] + vp.cx, vp.fy * c[1] / c[2] + vp.cy]);
        out.push(Tri2d {
            p,
            z: cam.map(|c| c[2]),
            color: flat_colour(tri, vp.pos, &cfg),
            depth: (cam[0][2] + cam[1][2] + cam[2][2]) / 3.0,
            index: index as u32,
        });
    }
    // Painter's algorithm: farthest first. A stable sort on a key that is only the depth
    // leaves ties in triangle order, which is what makes the emitted sequence a pure function
    // of the inputs (and a golden worth having).
    out.sort_by(|a, b| b.depth.total_cmp(&a.depth));
    out
}

/// `es_shade_lambert` on the flat triangle, pinned against `rasterize` by a test. The ray of a
/// flat triangle is the one from the eye to its centroid, in world space, which is the ray
/// `rasterize` casts through the centroid's pixel; its normal is flipped towards that ray.
fn flat_colour(tri: &Tri, eye: [f32; 3], cfg: &RenderConfig) -> [u8; 3] {
    let dir = [
        (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0 - eye[0],
        (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0 - eye[1],
        (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0 - eye[2],
    ];
    shade_lambert(tri, face_forward(tri, dir), cfg).map(to_u8)
}

fn vec3(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// The rotation whose matrix has `x`, `y`, `z` as its columns, by Shepperd's method: pick the
/// largest of the four denominators, so no branch divides by something near zero. The same
/// arithmetic as `es_env::render`'s, which is behind Vulkan (see [`Camera::view`]).
fn quat_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m01, m02) = (x.x, y.x, z.x);
    let (m10, m11, m12) = (x.y, y.y, z.y);
    let (m20, m21, m22) = (x.z, y.z, z.z);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::from_xyzw((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat::from_xyzw(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat::from_xyzw((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat::from_xyzw((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Channel, TileData};

    /// The camera the showcase renders the demo from (`docs/packets/M5/V9`), in world metres.
    fn showcase_camera() -> Camera {
        Camera {
            eye: [0.55, -0.45, 0.42],
            look_at: [0.12, -0.02, 0.08],
            fov_y: 45.0_f64.to_radians(),
            width: 480,
            height: 320,
        }
    }

    fn pixel(raster: &Raster, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * raster.w as usize + x as usize) * 3;
        [raster.rgb[i], raster.rgb[i + 1], raster.rgb[i + 2]]
    }

    /// Whether `p` is inside the screen-space triangle, by the same edge functions the
    /// rasteriser uses -- so "the painter would have covered this pixel" is not a guess.
    fn covers(tri: &Tri2d, p: [f32; 2]) -> bool {
        let [a, b, c] = tri.p;
        let inv = 1.0 / edge(a, b, c);
        edge(b, c, p) * inv >= 0.0 && edge(c, a, p) * inv >= 0.0 && edge(a, b, p) * inv >= 0.0
    }

    /// The flat colour is the renderer's own, and the projection lands where the renderer
    /// draws it. One triangle, so no occlusion can make either question ambiguous.
    #[test]
    fn flat_shade_matches_the_renderer() {
        let scene = TriScene {
            tris: vec![Tri {
                v: [[-0.2, 0.9, 0.05], [0.25, 1.1, -0.15], [0.05, 1.0, 0.3]],
                n: [0.0, -1.0, 0.0],
                albedo: [0.8, 0.45, 0.2],
                emission: [0.0; 3],
                seg: 1,
            }],
            ..TriScene::default()
        };
        let camera = Camera {
            eye: [0.0, 0.0, 0.0],
            look_at: [0.0, 1.0, 0.0],
            fov_y: 45.0_f64.to_radians(),
            width: 64,
            height: 48,
        };
        let view = camera.view().expect("a camera looking along +Y");
        let cfg = shading();
        let projected = project_scene(&scene, &view);
        assert_eq!(projected.len(), 1, "the triangle is in front of the camera");
        let tri = projected[0];

        let frame = crate::cpu::rasterize(&scene, &view, &cfg, 0);
        let Some(tile) = frame.tile(Channel::Rgb8) else {
            panic!("the Rs path writes Rgb8")
        };
        let TileData::U8(rgb) = &tile.data else {
            panic!("Rgb8 is u8")
        };
        let cx = (tri.p[0][0] + tri.p[1][0] + tri.p[2][0]) / 3.0;
        let cy = (tri.p[0][1] + tri.p[1][1] + tri.p[2][1]) / 3.0;
        assert!(
            (0.0..64.0).contains(&cx) && (0.0..48.0).contains(&cy),
            "the centroid projected off-image: {cx}, {cy}"
        );
        let i = (cy as usize * 64 + cx as usize) * 3;
        let pixel = [rgb[i], rgb[i + 1], rgb[i + 2]];
        assert_ne!(pixel, [0, 0, 0], "the rasterizer drew background there");
        assert_eq!(
            tri.color, pixel,
            "the replay's flat colour is not the renderer's"
        );
    }

    /// Two triangles that cross -- an X seen edge-on, each nearer on one side. The painter's
    /// algorithm has to draw one of them last and covers the other whole; the depth test
    /// splits the picture down the middle.
    #[test]
    fn a_depth_test_beats_the_painters_sort() {
        // The camera sits at the origin looking along +Y, so world +X is screen right. Both
        // triangles run from the bottom corners up to a shared apex; one is near on the left,
        // the other near on the right, and they cross in the plane x = 0.
        const NEAR: f32 = 1.5;
        const FAR: f32 = 2.5;
        let arm = |near_left: bool, albedo: [f32; 3]| Tri {
            v: [
                [-0.8, if near_left { NEAR } else { FAR }, -0.5],
                [0.8, if near_left { FAR } else { NEAR }, -0.5],
                [0.0, 2.0, 0.5],
            ],
            // The same normal for both, so the two flat colours differ only by albedo.
            n: [0.0, -1.0, 0.0],
            albedo,
            emission: [0.0; 3],
            seg: 1,
        };
        let scene = TriScene {
            tris: vec![arm(true, [0.8, 0.1, 0.1]), arm(false, [0.1, 0.1, 0.8])],
            ..TriScene::default()
        };
        let camera = Camera {
            eye: [0.0, 0.0, 0.0],
            look_at: [0.0, 1.0, 0.0],
            fov_y: 45.0_f64.to_radians(),
            width: 64,
            height: 48,
        };
        let projected = project_scene(&scene, &camera.view().expect("a camera along +Y"));
        assert_eq!(
            projected.len(),
            2,
            "both triangles are in front of the camera"
        );
        let named = |index: u32| {
            *projected
                .iter()
                .find(|t| t.index == index)
                .expect("the triangle survived the clip")
        };
        let (left, right) = (named(0), named(1));
        assert_ne!(left.color, right.color, "the two colours must differ");

        // What the painter's algorithm has to do here: the two centroids are at the same
        // depth, so no order separates them, and whichever is painted last covers both
        // sample pixels whole.
        assert_eq!(
            left.depth.to_bits(),
            right.depth.to_bits(),
            "the centroids are the same depth"
        );
        let samples = [[20.5, 30.5], [44.5, 30.5]];
        let last = *projected.last().expect("two triangles");
        for p in samples {
            assert!(covers(&left, p) && covers(&right, p), "{p:?} is in both");
        }

        let raster = Raster::draw(&projected, camera.width, camera.height);
        let (got_left, got_right) = (pixel(&raster, 20, 30), pixel(&raster, 44, 30));
        assert_eq!(
            got_left, left.color,
            "the left half is not the near-left arm"
        );
        assert_eq!(
            got_right, right.color,
            "the right half is not the near-right arm"
        );
        assert!(
            got_left != last.color || got_right != last.color,
            "the picture is exactly what the painter would have drawn"
        );
        // The depth buffer holds what was interpolated, not the centroid, and the nearer
        // surface won on each side.
        let depth = |x: u32, y: u32| raster.depth[y as usize * raster.w as usize + x as usize];
        assert!(depth(20, 30) < left.depth, "{}", depth(20, 30));
        assert!(depth(44, 30) < right.depth, "{}", depth(44, 30));
        // Background is background: a corner no triangle reaches keeps the canvas colour and
        // an untouched depth.
        assert_eq!(pixel(&raster, 0, 47), [BACKGROUND; 3]);
        assert!(depth(0, 47).is_infinite());
    }

    /// Orbit stays on the sphere and off its poles; zoom only scales the radius.
    #[test]
    fn the_camera_moves_on_a_sphere_about_the_target() {
        let camera = showcase_camera();
        let radius = |c: &Camera| (vec3(c.eye) - vec3(c.look_at)).norm();
        let start = radius(&camera);
        let turned = camera.orbit(0.7, 0.3);
        assert!(
            (radius(&turned) - start).abs() < 1e-9,
            "orbit changed the radius"
        );
        assert_eq!(
            turned.look_at.map(f64::to_bits),
            camera.look_at.map(f64::to_bits)
        );
        assert!(turned.view().is_ok());
        // Straight up is where a roll-free orientation stops existing; the orbit stops short.
        let up = camera.orbit(0.0, 10.0);
        assert!(up.view().is_ok(), "the orbit walked into the pole");
        assert!((radius(&up) - start).abs() < 1e-9);

        let closer = camera.zoom(0.5);
        assert!((radius(&closer) - start * 0.5).abs() < 1e-9);
        assert_eq!(
            closer.look_at.map(f64::to_bits),
            camera.look_at.map(f64::to_bits)
        );

        // A camera looking straight down world up has no roll-free orientation, and says so.
        let degenerate = Camera {
            eye: [0.0, 0.0, 1.0],
            look_at: [0.0, 0.0, 0.0],
            ..camera
        };
        assert!(degenerate.view().is_err());
        assert!(degenerate.view().unwrap_err().to_string().contains("+Z"));
    }
}
