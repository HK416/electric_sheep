//! The golden scene: a Cornell box, built in Rust so it needs no asset file.
//!
//! Everything is an axis-aligned `Shape::Box`, so the tessellation is 12 triangles per geom
//! and the whole scene is 96 triangles — inside the "few hundred" ceiling of the scan-based
//! traversal (see `docs/design/renderer.md`). Walls are thin boxes rather than planes so the
//! room is closed and a path-traced ray cannot leak out of a corner.
//!
//! **No two faces in this scene are coplanar, and the two inner boxes are sunk into the
//! floor.** That is not cosmetic: where two triangles share a plane, a ray hits both at
//! exactly the same `t` and the winner is decided by which side of `u + v <= 1` the
//! barycentrics land on — which the CPU and the GPU can answer differently by one ULP. Walls
//! therefore overlap the floor and ceiling instead of abutting them, and the floor/ceiling
//! side faces are buried inside the walls.
//!
//! World frame is spec 3.1's: Z-up, X-forward, Y-left, metres. The room's inner surface spans
//! `x in [-1.2, 4] , y in [-2, 2] , z in [0, 4]` and the camera sits just inside the open
//! `x = -1.2` end looking along +X.

use es_assets::scene::{scene_id, Body, Geom, PhysicsOptions, SceneDesc, Shape};
use es_math::{Pose, Quat, Vec3};

use crate::view::{CameraView, ImageSpec};

/// `T_world_camera` for a camera whose `OpenCV` frame (+Z forward, +X right, +Y down) looks
/// along world +X with world +Z up. A 120-degree rotation about `(-1, 1, -1)/sqrt(3)`.
pub const LOOK_ALONG_X: Quat = Quat::from_xyzw(-0.5, 0.5, -0.5, 0.5);

const WHITE: [f64; 4] = [0.725, 0.71, 0.68, 1.0];
const RED: [f64; 4] = [0.63, 0.065, 0.05, 1.0];
const GREEN: [f64; 4] = [0.14, 0.45, 0.091, 1.0];
const EMISSIVE: [f64; 4] = [1.0, 0.9, 0.78, 1.0];

fn boxx(name: &str, center: Vec3, half: Vec3, rgba: [f64; 4]) -> Geom {
    Geom {
        id: scene_id("geom", &format!("world/{name}")),
        name: name.to_owned(),
        shape: Shape::Box { half_extents: half },
        pose: Pose::new(center, Quat::IDENTITY),
        friction: [1.0, 0.005, 0.000_1],
        contype: 1,
        conaffinity: 1,
        condim: 3,
        density: 1000.0,
        mass: None,
        margin: 0.0,
        gap: 0.0,
        solref: [0.02, 1.0],
        solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
        material: None,
        rgba,
        visual_only: false,
    }
}

/// The golden scene. Geom order fixes the segmentation ids, so reordering this list
/// invalidates every golden.
pub fn cornell_box() -> SceneDesc {
    let v = Vec3::new;
    let geoms = vec![
        boxx("floor", v(1.5, 0.0, -0.15), v(2.7, 2.2, 0.15), WHITE),
        boxx("ceiling", v(1.5, 0.0, 4.15), v(2.7, 2.2, 0.15), WHITE),
        boxx("back", v(4.15, 0.0, 2.0), v(0.15, 2.2, 2.3), WHITE),
        boxx("left", v(1.5, 2.15, 2.0), v(2.7, 0.15, 2.3), RED),
        boxx("right", v(1.5, -2.15, 2.0), v(2.7, 0.15, 2.3), GREEN),
        boxx("tall", v(2.8, 0.8, 0.85), v(0.45, 0.45, 0.9), WHITE),
        boxx("short", v(1.7, -0.8, 0.4), v(0.45, 0.45, 0.45), WHITE),
        boxx(
            "ceiling_light",
            v(2.0, 0.0, 3.88),
            v(0.6, 0.6, 0.02),
            EMISSIVE,
        ),
    ];
    SceneDesc {
        name: "cornell".to_owned(),
        bodies: vec![Body {
            id: scene_id("body", "world"),
            name: "world".to_owned(),
            parent: None,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms,
            sites: Vec::new(),
        }],
        joints: Vec::new(),
        actuators: Vec::new(),
        sensors: Vec::new(),
        tendons: Vec::new(),
        cameras: Vec::new(),
        assets: Vec::new(),
        options: PhysicsOptions::default(),
    }
}

/// The camera the goldens are rendered from.
pub fn cornell_camera(width: u32, height: u32) -> CameraView {
    CameraView {
        pose: Pose::new(Vec3::new(-1.0, 0.0, 2.0), LOOK_ALONG_X),
        spec: ImageSpec::pinhole(width, height, 1.2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::TriScene;

    #[test]
    fn the_room_is_closed_and_has_one_light() {
        let tri = TriScene::from_scene(&cornell_box()).unwrap();
        assert_eq!(tri.tris.len(), 8 * 12);
        assert_eq!(tri.lights.len(), 12);
        assert_eq!(tri.names.len(), 8);
    }

    #[test]
    fn the_camera_looks_along_world_x() {
        let cam = cornell_camera(8, 8);
        let fwd = cam.pose.orientation.rotate(Vec3::new(0.0, 0.0, 1.0));
        assert!((fwd.x - 1.0).abs() < 1e-9, "{fwd:?}");
        // OpenCV +Y is image-down, which must be world-down.
        let down = cam.pose.orientation.rotate(Vec3::new(0.0, 1.0, 0.0));
        assert!((down.z + 1.0).abs() < 1e-9, "{down:?}");
    }
}
