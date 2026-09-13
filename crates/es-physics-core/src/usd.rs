//! `UsdStage` -> [`SceneDesc`]: the physics reading of a parsed `.usda` layer (M4, spec 28.6).
//!
//! Why here and not in `es-usd`: `es-usd` and `es-assets` are both layer 2 and spec 4.2 forbids
//! same-layer dependencies, so the crate that owns [`SceneDesc`] and the crate that owns
//! `UsdStage` cannot see each other. `es-physics-core` (layer 3) already depends on both, so
//! the mapping lives here. Design note: `docs/design/usd-reader.md`; format digest, with the
//! unverified items marked: `docs/api-notes/usd.md`.
//!
//! Everything the file states is mapped; everything it does not state is left alone. No mass
//! is derived from a geom, no collision intent is invented for a prim without
//! `PhysicsCollisionAPI`, and anything unmapped is a [`Warning`], never a silent drop.

use std::collections::{BTreeMap, BTreeSet};

use es_assets::scene::{
    AssetKind, AssetRef, Body, BodyInertial, Geom, Joint, JointKind, PhysicsOptions, SceneDesc,
    SceneError, Shape,
};
use es_assets::scene_id;
use es_math::{units::DEG_TO_RAD, Inertia, Pose, Quat, Vec3};
use es_usd::{Prim, UpAxis, UsdStage, Value};
use thiserror::Error;

/// The fixed rotation from a Y-up right-handed frame to spec 3.1 (Z-up, X-forward).
///
/// The same constant `es-assets`' glTF importer uses, for the same reason: USD `upAxis = "Y"`
/// is the convention glTF is always in. `+Y -> +Z`, `-Z -> +X`, `-X -> +Y`.
const AXIS_FIX: Quat = Quat::from_xyzw(0.5, -0.5, -0.5, 0.5);

/// Domain separator for a mesh content hash.
const MESH_TAG: &str = "es.usd.mesh.v1";

/// Attributes that state something real which [`SceneDesc`] has nowhere to put.
const UNMAPPED_ATTRS: &[&str] = &[
    "physics:velocity",
    "physics:angularVelocity",
    "physics:kinematicEnabled",
    "physics:density",
];

/// Something the file said that the scene cannot carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    /// Prim path, or `/` for a layer-level remark.
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Why a stage could not be turned into a scene.
#[derive(Debug, Error)]
pub enum UsdSceneError {
    /// The file states something self-contradictory, or something this reader refuses.
    #[error("{path}: {message}")]
    Invalid { path: String, message: String },
    /// The mapping produced a structurally invalid scene, which is a bug here, not in the file.
    #[error("the stage does not describe a valid scene: {0}")]
    Scene(#[from] SceneError),
}

fn invalid(path: &str, message: impl Into<String>) -> UsdSceneError {
    UsdSceneError::Invalid {
        path: path.to_owned(),
        message: message.into(),
    }
}

fn note(path: &str, message: impl Into<String>) -> Warning {
    Warning {
        path: path.to_owned(),
        message: message.into(),
    }
}

/// Prim path without its leading `/`, which is the name path [`scene_id`] wants.
fn id_path(path: &str) -> &str {
    path.trim_start_matches('/')
}

/// Parses `.usda` text and maps it onto a scene in one step.
///
/// # Errors
///
/// Parse failures arrive as [`UsdSceneError::Invalid`] carrying the reader's own message —
/// the source line for a syntax error, the refused feature for an unsupported one.
pub fn import_usda(text: &str) -> Result<(SceneDesc, Vec<Warning>), UsdSceneError> {
    let stage = es_usd::parse_usda(text).map_err(|e| match &e {
        es_usd::UsdError::Unsupported { path, .. } => invalid(path, e.to_string()),
        es_usd::UsdError::Syntax { .. } => invalid("/", e.to_string()),
    })?;
    stage_to_scene(&stage)
}

/// Maps a parsed stage onto a validated [`SceneDesc`].
///
/// Conventions (spec 3.1) are applied exactly once, to each prim's *local* transform, which is
/// then composed down the hierarchy — the same scheme `es-assets`' glTF importer uses, and for
/// the same reason: conjugating a rotation distributes over pose composition, so converting
/// per prim and converting once at the end agree. Every length is multiplied by
/// `metersPerUnit`, every inertia by its square, and revolute limits and angular drive targets
/// are converted from the schema's degrees to radians. Body-local quantities — centre of mass,
/// joint anchors and axes, mesh points — are rotated by the same fix, so every frame in the
/// result is Z-up.
///
/// # Errors
///
/// [`UsdSceneError::Invalid`] when the file states something self-contradictory (a joint whose
/// `physics:body1` is not a rigid body, a mesh index out of range, a non-positive
/// `metersPerUnit`), and [`UsdSceneError::Scene`] when the result would not pass
/// [`SceneDesc::validate`].
pub fn stage_to_scene(stage: &UsdStage) -> Result<(SceneDesc, Vec<Warning>), UsdSceneError> {
    let mpu = stage.meta.meters_per_unit;
    if !mpu.is_finite() || mpu <= 0.0 {
        return Err(invalid(
            "/",
            format!("metersPerUnit must be finite and positive, found {mpu}"),
        ));
    }
    let fix = match stage.meta.up_axis {
        UpAxis::Y => AXIS_FIX,
        UpAxis::Z => Quat::IDENTITY,
    };

    let mut warnings: Vec<Warning> = stage
        .warnings
        .iter()
        .map(|message| note("/", message.clone()))
        .collect();

    // Document-order preorder, each prim with its world pose already in spec 3.1 terms.
    let mut placed: Vec<(&Prim, Pose)> = Vec::new();
    for root in &stage.prims {
        place(root, Pose::IDENTITY, fix, mpu, &mut placed, &mut warnings);
    }
    for (prim, _) in &placed {
        for attr in UNMAPPED_ATTRS {
            if prim.attrs.contains_key(*attr) {
                warnings.push(note(
                    &prim.path,
                    format!("`{attr}` has no place in a SceneDesc and is dropped"),
                ));
            }
        }
    }

    let world: BTreeMap<&str, Pose> = placed.iter().map(|(p, w)| (p.path.as_str(), *w)).collect();
    let body_paths: BTreeSet<&str> = placed
        .iter()
        .filter(|(p, _)| p.has_api("PhysicsRigidBodyAPI"))
        .map(|(p, _)| p.path.as_str())
        .collect();

    // In USD it is the joint, not prim nesting, that articulates two bodies: a robot is
    // usually a flat list of links plus joints. So a body's parent is its nearest rigid-body
    // *ancestor prim* when it has one, and otherwise the `physics:body0` of the joint that
    // names it as `physics:body1`.
    let mut parent_of: BTreeMap<&str, &str> = BTreeMap::new();
    for (prim, _) in &placed {
        if joint_kind(&prim.type_name).is_none() {
            continue;
        }
        if let (Some(child), Some(parent)) = (
            rel_target(prim, "physics:body1"),
            rel_target(prim, "physics:body0"),
        ) {
            if body_paths.contains(child) && body_paths.contains(parent) {
                parent_of.insert(child, parent);
            }
        }
    }
    for path in &body_paths {
        if let Some(ancestor) = nearest_body_ancestor(path, &body_paths) {
            parent_of.insert(*path, ancestor);
        }
    }

    let mut bodies: BTreeMap<&str, Body> = BTreeMap::new();
    for (prim, _) in &placed {
        let path = prim.path.as_str();
        if !body_paths.contains(path) {
            continue;
        }
        let parent = parent_of.get(path).copied();
        let pose = match parent {
            Some(p) => world[p].inverse().compose(world[path]),
            None => world[path],
        };
        bodies.insert(
            path,
            Body {
                id: scene_id("body", id_path(path)),
                name: prim.name.clone(),
                parent: parent.map(|p| scene_id("body", id_path(p))),
                pose,
                inertial: inertial(prim, fix, mpu),
                geoms: Vec::new(),
                sites: Vec::new(),
            },
        );
    }

    let mut assets = Vec::new();
    for (prim, _) in &placed {
        if !prim.has_api("PhysicsCollisionAPI") {
            continue;
        }
        let path = prim.path.as_str();
        if prim
            .attrs
            .get("physics:collisionEnabled")
            .and_then(Value::as_bool)
            == Some(false)
        {
            warnings.push(note(
                path,
                "physics:collisionEnabled is false; geom skipped",
            ));
            continue;
        }
        let owner = if body_paths.contains(path) {
            Some(path)
        } else {
            nearest_body_ancestor(path, &body_paths)
        };
        let Some(owner) = owner else {
            warnings.push(note(
                path,
                "PhysicsCollisionAPI on a prim with no rigid-body ancestor; geom skipped",
            ));
            continue;
        };
        let Some((shape, extra)) = shape_of(prim, fix, mpu, &mut assets, &mut warnings)? else {
            continue;
        };
        let pose = world[owner].inverse().compose(world[path]).compose(extra);
        bodies
            .get_mut(owner)
            .expect("owner is a body path, so it is in bodies")
            .geoms
            .push(geom(prim, shape, pose));
    }

    let mut joints = Vec::new();
    for (prim, _) in &placed {
        let Some(kind) = joint_kind(&prim.type_name) else {
            if prim.type_name == "PhysicsScene" {
                warnings.push(note(
                    &prim.path,
                    "PhysicsScene is not mapped; the scene keeps the default PhysicsOptions",
                ));
            }
            continue;
        };
        joints.push(joint(prim, kind, fix, mpu, &body_paths, &mut warnings)?);
    }

    let scene = SceneDesc {
        name: stage
            .meta
            .default_prim
            .clone()
            .unwrap_or_else(|| "usd".to_owned()),
        bodies: bodies.into_values().collect(),
        joints,
        assets,
        options: PhysicsOptions::default(),
        ..SceneDesc::default()
    };
    scene.validate()?;
    Ok((scene, warnings))
}

/// Converts a prim's local transform onto spec 3.1 and composes it under its parent.
fn place<'a>(
    prim: &'a Prim,
    parent: Pose,
    fix: Quat,
    mpu: f64,
    out: &mut Vec<(&'a Prim, Pose)>,
    warnings: &mut Vec<Warning>,
) {
    let mut local_warnings = Vec::new();
    let local = prim.local_xform(&mut local_warnings);
    warnings.extend(
        local_warnings
            .into_iter()
            .map(|message| note(&prim.path, message)),
    );
    let converted = Pose::new(
        fix.rotate(local.position).scale(mpu),
        (fix * local.orientation * fix.conjugate()).normalize(),
    );
    let world = parent.compose(converted);
    out.push((prim, world));
    for child in &prim.children {
        place(child, world, fix, mpu, out, warnings);
    }
}

/// A body-frame point (metres out, file units in).
fn point(v: Vec3, fix: Quat, mpu: f64) -> Vec3 {
    fix.rotate(v).scale(mpu)
}

fn nearest_body_ancestor<'a>(path: &str, bodies: &BTreeSet<&'a str>) -> Option<&'a str> {
    let mut cursor = path;
    while let Some((head, _)) = cursor.rsplit_once('/') {
        if head.is_empty() {
            return None;
        }
        if let Some(hit) = bodies.get(head) {
            return Some(hit);
        }
        cursor = head;
    }
    None
}

fn rel_target<'a>(prim: &'a Prim, name: &str) -> Option<&'a str> {
    prim.rels.get(name)?.first().map(String::as_str)
}

fn joint_kind(type_name: &str) -> Option<JointKind> {
    match type_name {
        "PhysicsRevoluteJoint" => Some(JointKind::Hinge),
        "PhysicsPrismaticJoint" => Some(JointKind::Slide),
        "PhysicsFixedJoint" => Some(JointKind::Fixed),
        _ => None,
    }
}

/// A number attribute, or `fallback` when it is absent.
fn num(prim: &Prim, name: &str, fallback: f64) -> f64 {
    prim.attrs
        .get(name)
        .and_then(Value::as_f64)
        .unwrap_or(fallback)
}

fn token(prim: &Prim, name: &str, fallback: &str) -> String {
    prim.attrs
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or(fallback)
        .to_owned()
}

fn axis_vector(axis: &str) -> Vec3 {
    match axis {
        "X" => Vec3::new(1.0, 0.0, 0.0),
        "Y" => Vec3::new(0.0, 1.0, 0.0),
        _ => Vec3::new(0.0, 0.0, 1.0),
    }
}

/// The shortest rotation taking `+Z` onto `d`. `SceneDesc` capsules and cylinders run along
/// `+Z`; USD lets the prim choose, and this folds that choice into the geom's pose.
///
/// The half-way quaternion: `(z x d, 1 + z.d)` normalised. Pure algebra, no transcendentals
/// (spec 3.4 keeps those out of the physics path even where an importer could afford them).
fn rotation_z_to(d: Vec3) -> Quat {
    let d = d.normalize();
    if d.z < -1.0 + 1e-12 {
        // Antiparallel: the half-way vector vanishes, so pick any perpendicular axis.
        return Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);
    }
    let c = Vec3::new(0.0, 0.0, 1.0).cross(d);
    Quat::from_xyzw(c.x, c.y, c.z, 1.0 + d.z).normalize()
}

/// `PhysicsMassAPI` -> mass properties. Absent means "derive from the geoms" (`None`), which is
/// the backend's job, not ours.
fn inertial(prim: &Prim, fix: Quat, mpu: f64) -> Option<BodyInertial> {
    if !prim.has_api("PhysicsMassAPI") {
        return None;
    }
    let mass = prim.attrs.get("physics:mass").and_then(Value::as_f64)?;
    let com = point(
        prim.attrs
            .get("physics:centerOfMass")
            .and_then(Value::as_vec3)
            .unwrap_or(Vec3::ZERO),
        fix,
        mpu,
    );
    // Inertia is mass * length^2, and its components are along `physics:principalAxes`, so
    // only the frame is rotated.
    let d = prim
        .attrs
        .get("physics:diagonalInertia")
        .and_then(Value::as_vec3)
        .unwrap_or(Vec3::ZERO)
        .scale(mpu * mpu);
    let axes = prim
        .attrs
        .get("physics:principalAxes")
        .and_then(Value::as_quat)
        .unwrap_or(Quat::IDENTITY);
    Some(BodyInertial {
        mass,
        com,
        inertia: Inertia::diagonal(d.x, d.y, d.z),
        frame: (fix * axes * fix.conjugate()).normalize(),
    })
}

/// The geometric primitive plus the extra pose its axis convention needs. `Ok(None)` means the
/// prim type carries no collision shape, and has already been warned about.
fn shape_of(
    prim: &Prim,
    fix: Quat,
    mpu: f64,
    assets: &mut Vec<AssetRef>,
    warnings: &mut Vec<Warning>,
) -> Result<Option<(Shape, Pose)>, UsdSceneError> {
    let shape = match prim.type_name.as_str() {
        "Cube" => Shape::Box {
            half_extents: Vec3::new(1.0, 1.0, 1.0).scale(num(prim, "size", 2.0) * mpu / 2.0),
        },
        "Sphere" => Shape::Sphere {
            radius: num(prim, "radius", 1.0) * mpu,
        },
        "Cylinder" => Shape::Cylinder {
            radius: num(prim, "radius", 1.0) * mpu,
            half_length: num(prim, "height", 2.0) * mpu / 2.0,
        },
        "Capsule" => Shape::Capsule {
            radius: num(prim, "radius", 1.0) * mpu,
            half_length: num(prim, "height", 2.0) * mpu / 2.0,
        },
        "Mesh" => {
            let asset = mesh_asset(prim, fix, mpu)?;
            let shape = Shape::Mesh { asset: asset.id };
            assets.push(asset);
            shape
        }
        other => {
            warnings.push(note(
                &prim.path,
                format!("PhysicsCollisionAPI on `{other}`, which has no shape mapping; skipped"),
            ));
            return Ok(None);
        }
    };
    let extra = match prim.type_name.as_str() {
        "Cylinder" | "Capsule" => Pose::new(
            Vec3::ZERO,
            // The axis token names an axis of the prim's own (already converted) frame.
            rotation_z_to(fix.rotate(axis_vector(&token(prim, "axis", "Z")))),
        ),
        _ => Pose::IDENTITY,
    };
    Ok(Some((shape, extra)))
}

fn geom(prim: &Prim, shape: Shape, pose: Pose) -> Geom {
    Geom {
        id: scene_id("geom", id_path(&prim.path)),
        name: prim.name.clone(),
        shape,
        pose,
        friction: [1.0, 0.005, 0.0001],
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
        rgba: [0.5, 0.5, 0.5, 1.0],
        visual_only: false,
    }
}

/// A `faceVertexCounts` entry, checked to be a small non-negative integer that fits `u32`
/// (spec: a polygon face) rather than whatever a `float` happens to hold. `f64 as i64` is a
/// saturating cast in Rust, so `1e20` lands on `i64::MAX` here and is then rejected by
/// `u32::try_from` instead of silently saturating all the way to `usize::MAX` (S-1).
fn usd_face_vertex_count(count: f64) -> Option<usize> {
    if !count.is_finite() || count.fract() != 0.0 {
        return None;
    }
    let n = u32::try_from(count as i64).ok()?;
    (n >= 3).then_some(n as usize)
}

/// Triangulates the mesh, converts it onto spec 3.1 and hashes the *content*, so the same
/// geometry authored twice gets one hash (spec 5.3, the rule the glTF importer already uses).
fn mesh_asset(prim: &Prim, fix: Quat, mpu: f64) -> Result<AssetRef, UsdSceneError> {
    let points = prim
        .attrs
        .get("points")
        .and_then(Value::as_vec3_array)
        .ok_or_else(|| invalid(&prim.path, "Mesh has no `points` array"))?;
    let counts = prim
        .attrs
        .get("faceVertexCounts")
        .and_then(Value::as_f64_array)
        .ok_or_else(|| invalid(&prim.path, "Mesh has no `faceVertexCounts` array"))?;
    let indices = prim
        .attrs
        .get("faceVertexIndices")
        .and_then(Value::as_f64_array)
        .ok_or_else(|| invalid(&prim.path, "Mesh has no `faceVertexIndices` array"))?;

    let positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| {
            let v = point(*p, fix, mpu);
            [v.x as f32, v.y as f32, v.z as f32]
        })
        .collect();

    let mut tris: Vec<u32> = Vec::new();
    let mut cursor = 0usize;
    for count in counts {
        // S-1 (docs/reviews/M4.md): `count` is a USD `float`, so a file can put `1e20` where a
        // face vertex count belongs. `count as usize` saturates instead of erroring, and
        // `cursor + n` can then wrap in release. Route both through checked conversions.
        let n = usd_face_vertex_count(count).ok_or_else(|| {
            invalid(
                &prim.path,
                format!("faceVertexCounts entry {count} must be an integer in 3..=u32::MAX"),
            )
        })?;
        let next_cursor = cursor
            .checked_add(n)
            .filter(|next| *next <= indices.len())
            .ok_or_else(|| {
                invalid(
                    &prim.path,
                    "faceVertexCounts does not agree with faceVertexIndices",
                )
            })?;
        // Fan triangulation: a USD face is planar and convex by convention.
        for k in 1..n - 1 {
            for offset in [0, k, k + 1] {
                let raw = indices[cursor + offset] as i64;
                let index = usize::try_from(raw)
                    .ok()
                    .filter(|i| *i < positions.len())
                    .ok_or_else(|| {
                        invalid(
                            &prim.path,
                            format!("faceVertexIndices entry {raw} is out of range"),
                        )
                    })?;
                tris.push(index as u32);
            }
        }
        cursor = next_cursor;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(MESH_TAG.as_bytes());
    for p in &positions {
        for v in p {
            hasher.update(&v.to_le_bytes());
        }
    }
    for i in &tris {
        hasher.update(&i.to_le_bytes());
    }
    Ok(AssetRef {
        id: scene_id("asset", &format!("mesh/{}", id_path(&prim.path))),
        name: prim.name.clone(),
        kind: AssetKind::Mesh,
        path: prim.path.clone(),
        hash: *hasher.finalize().as_bytes(),
    })
}

fn joint(
    prim: &Prim,
    kind: JointKind,
    fix: Quat,
    mpu: f64,
    body_paths: &BTreeSet<&str>,
    warnings: &mut Vec<Warning>,
) -> Result<Joint, UsdSceneError> {
    let child = rel_target(prim, "physics:body1")
        .ok_or_else(|| invalid(&prim.path, "joint has no `physics:body1` relationship"))?;
    if !body_paths.contains(child) {
        return Err(invalid(
            &prim.path,
            format!("`physics:body1` is `{child}`, which is not a PhysicsRigidBodyAPI prim"),
        ));
    }
    if let Some(parent) = rel_target(prim, "physics:body0") {
        if !body_paths.contains(parent) {
            warnings.push(note(
                &prim.path,
                format!("`physics:body0` is `{parent}`, which is not a rigid body; the joint attaches to the world"),
            ));
        }
    }

    let local_rot = prim
        .attrs
        .get("physics:localRot1")
        .and_then(Value::as_quat)
        .unwrap_or(Quat::IDENTITY);
    let axis = fix
        .rotate(local_rot.rotate(axis_vector(&token(prim, "physics:axis", "Z"))))
        .normalize();
    let anchor = point(
        prim.attrs
            .get("physics:localPos1")
            .and_then(Value::as_vec3)
            .unwrap_or(Vec3::ZERO),
        fix,
        mpu,
    );

    // Revolute limits and angular drive targets are degrees; prismatic ones are lengths.
    let (scale, dof) = match kind {
        JointKind::Hinge => (DEG_TO_RAD, "angular"),
        JointKind::Slide => (mpu, "linear"),
        _ => (1.0, ""),
    };
    let range = match (
        prim.attrs.get("physics:lowerLimit").and_then(Value::as_f64),
        prim.attrs.get("physics:upperLimit").and_then(Value::as_f64),
    ) {
        (Some(lo), Some(hi)) if kind != JointKind::Fixed && lo <= hi => {
            Some((lo * scale, hi * scale))
        }
        _ => None,
    };

    for key in prim.attrs.keys() {
        if let Some(rest) = key.strip_prefix("drive:") {
            let found = rest.split(':').next().unwrap_or_default();
            if found != dof {
                warnings.push(note(
                    &prim.path,
                    format!("drive on `{found}`, a degree of freedom this joint type does not have; ignored"),
                ));
            }
        }
    }
    let drive = |suffix: &str| {
        prim.attrs
            .get(&format!("drive:{dof}:physics:{suffix}"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    };

    Ok(Joint {
        id: scene_id("joint", id_path(&prim.path)),
        name: prim.name.clone(),
        body: scene_id("body", id_path(child)),
        kind,
        axis,
        anchor,
        range,
        damping: drive("damping"),
        armature: 0.0,
        stiffness: drive("stiffness"),
        friction_loss: 0.0,
        spring_ref: drive("targetPosition") * scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn fixture(name: &str) -> String {
        let path = format!(
            "{}/../../tests/fixtures/usd/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    fn load(name: &str) -> (SceneDesc, Vec<Warning>) {
        import_usda(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn body<'a>(scene: &'a SceneDesc, name: &str) -> &'a Body {
        scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("no body `{name}`"))
    }

    #[test]
    fn pendulum_maps_two_bodies_a_hinge_and_its_drive() {
        let (scene, warnings) = load("pendulum.usda");
        assert!(scene.validate().is_ok());
        assert_eq!(scene.name, "World");
        assert_eq!(scene.bodies.len(), 2);

        let base = body(&scene, "base");
        let link = body(&scene, "link");
        // Already Z-up metres, so the root keeps its authored placement.
        assert_eq!(base.pose.position, Vec3::new(0.0, 0.0, 2.0));
        assert_eq!(base.parent, None);
        // `link` is `base`'s child through the joint, not through prim nesting.
        assert_eq!(link.parent, Some(base.id));
        assert_eq!(link.pose.position, Vec3::new(0.0, 0.0, -0.5));

        let inertial = link.inertial.expect("PhysicsMassAPI");
        assert!(close(inertial.mass, 1.25));
        assert_eq!(inertial.com, Vec3::new(0.0, 0.0, -0.25));

        assert_eq!(
            base.geoms[0].shape,
            Shape::Box {
                half_extents: Vec3::new(0.1, 0.1, 0.1)
            }
        );
        assert_eq!(
            link.geoms[0].shape,
            Shape::Capsule {
                radius: 0.05,
                half_length: 0.25
            }
        );

        assert_eq!(scene.joints.len(), 1);
        let j = &scene.joints[0];
        assert_eq!(j.kind, JointKind::Hinge);
        assert_eq!(j.body, link.id);
        assert_eq!(j.axis, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(j.anchor, Vec3::new(0.0, 0.0, 0.25));
        // Degrees in the file, radians in the scene (spec 3.1).
        let (lo, hi) = j.range.expect("limits");
        assert!(close(lo, -std::f64::consts::FRAC_PI_2), "{lo}");
        assert!(close(hi, std::f64::consts::FRAC_PI_2), "{hi}");
        assert!(close(j.stiffness, 120.0));
        assert!(close(j.damping, 8.0));
        assert!(close(j.spring_ref, 30.0 * DEG_TO_RAD));

        // The one thing the file states that a SceneDesc cannot hold.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].message.contains("kinematicEnabled"));
    }

    #[test]
    fn mesh_is_triangulated_and_content_hashed() {
        let (scene, _) = load("mesh_cube.usda");
        assert_eq!(scene.assets.len(), 1);
        let asset = &scene.assets[0];
        assert_eq!(asset.kind, AssetKind::Mesh);
        assert_eq!(
            scene.bodies[0].geoms[0].shape,
            Shape::Mesh { asset: asset.id }
        );
        assert!(close(scene.bodies[0].inertial.expect("mass api").mass, 2.0));

        // Re-importing the same text reproduces the hash; moving a vertex changes it even
        // though the prim path -- and so the asset id -- does not.
        let (again, _) = load("mesh_cube.usda");
        assert_eq!(again.assets[0].hash, asset.hash);
        let edited = fixture("mesh_cube.usda").replace("(-0.5, -0.5, -0.5)", "(-0.6, -0.5, -0.5)");
        let (moved, _) = import_usda(&edited).expect("still valid");
        assert_ne!(moved.assets[0].hash, asset.hash);
        assert_eq!(moved.assets[0].id, asset.id);
    }

    #[test]
    fn a_broken_mesh_is_an_error_not_a_panic() {
        let edited = fixture("mesh_cube.usda").replace(
            "int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]",
            "int[] faceVertexCounts = [4, 4, 4, 4, 4, 5]",
        );
        assert!(import_usda(&edited).is_err());
        let edited = fixture("mesh_cube.usda").replace("0, 3, 2, 1,", "0, 3, 2, 99,");
        assert!(import_usda(&edited).is_err());
    }

    #[test]
    fn a_face_vertex_count_that_overflows_u32_is_an_error_not_ub() {
        // S-1 (docs/reviews/M4.md): `count as usize` used to saturate to `usize::MAX` and the
        // `cursor + n > indices.len()` guard wrapped past it in release, reaching the indexing
        // below with an out-of-bounds `cursor`. A pathological count must be a hard error.
        let edited = fixture("mesh_cube.usda").replace(
            "int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]",
            "int[] faceVertexCounts = [3, 1e20, 4, 4, 4, 4]",
        );
        let err = import_usda(&edited).expect_err("an out-of-range count must be rejected");
        let UsdSceneError::Invalid { message, .. } = err else {
            panic!("expected Invalid, got {err:?}");
        };
        assert!(message.contains("faceVertexCounts"), "{message}");
    }

    #[test]
    fn y_up_centimetres_converts_to_z_up_metres() {
        let (scene, _) = load("yup_cm.usda");
        let post = body(&scene, "post");
        let arm = body(&scene, "arm");

        // USD (0, 300, 0) cm, Y-up -> spec 3.1 (0, 0, 3) m.
        assert!(close(post.pose.position.x, 0.0), "{:?}", post.pose);
        assert!(close(post.pose.position.y, 0.0), "{:?}", post.pose);
        assert!(close(post.pose.position.z, 3.0), "{:?}", post.pose);
        // The arm hangs 50 cm below, expressed in the post's (also Z-up) frame.
        assert_eq!(arm.parent, Some(post.id));
        assert!(close(arm.pose.position.z, -0.5), "{:?}", arm.pose);
        assert!(close(arm.pose.position.y, 0.0), "{:?}", arm.pose);

        assert_eq!(post.geoms[0].shape, Shape::Sphere { radius: 0.1 });
        assert_eq!(
            arm.geoms[0].shape,
            Shape::Cylinder {
                radius: 0.04,
                half_length: 0.25
            }
        );
        // The cylinder runs along USD +Y, which is +Z after the conversion, so the geom needs
        // no extra rotation at all.
        let along = arm.geoms[0]
            .pose
            .orientation
            .rotate(Vec3::new(0.0, 0.0, 1.0));
        assert!(close(along.z, 1.0), "{along:?}");

        // Prismatic limits are lengths, not angles: 20 cm -> 0.2 m, along USD +Y = +Z.
        let j = &scene.joints[0];
        assert_eq!(j.kind, JointKind::Slide);
        let (lo, hi) = j.range.expect("limits");
        assert!(close(lo, -0.2));
        assert!(close(hi, 0.2));
        assert!(close(j.axis.z, 1.0), "{:?}", j.axis);
    }

    #[test]
    fn a_cylinder_along_x_is_rotated_onto_z() {
        let text = fixture("yup_cm.usda")
            .replace("uniform token axis = \"Y\"", "uniform token axis = \"X\"");
        let (scene, _) = import_usda(&text).expect("valid");
        let arm = body(&scene, "arm");
        // The Y-up fix sends glTF/USD `-X` to `+Y`, so USD `+X` is spec 3.1 `-Y`, and the
        // geom's own `+Z` must end up pointing there.
        let along = arm.geoms[0]
            .pose
            .orientation
            .rotate(Vec3::new(0.0, 0.0, 1.0));
        assert!(close(along.y, -1.0), "{along:?}");
    }

    #[test]
    fn scene_hash_is_stable_across_parses() {
        let (a, _) = load("pendulum.usda");
        let (b, _) = load("pendulum.usda");
        assert_eq!(a.scene_hash(), b.scene_hash());
        assert_ne!(a.scene_hash(), load("yup_cm.usda").0.scene_hash());
    }

    #[test]
    fn a_referenced_layer_is_refused_by_prim_path() {
        let err = import_usda(&fixture("referenced.usda")).expect_err("references are refused");
        let UsdSceneError::Invalid { path, message } = err else {
            panic!("expected Invalid");
        };
        assert_eq!(path, "/World/robot");
        assert!(message.contains("references"), "{message}");
    }

    #[test]
    fn a_malformed_layer_reports_its_line() {
        let err = import_usda(&fixture("malformed.usda")).expect_err("malformed");
        assert!(err.to_string().contains("line 11"), "{err}");
    }

    #[test]
    fn a_stage_with_no_bodies_is_an_empty_valid_scene() {
        let (scene, warnings) = import_usda(
            "#usda 1.0\n(\n upAxis = \"Z\"\n metersPerUnit = 1\n)\ndef Xform \"a\" {}\n",
        )
        .expect("valid");
        assert!(scene.bodies.is_empty());
        assert!(scene.validate().is_ok());
        assert!(warnings.is_empty());
    }

    #[test]
    fn a_non_positive_meters_per_unit_is_refused() {
        let err = import_usda("#usda 1.0\n(\n upAxis = \"Z\"\n metersPerUnit = 0\n)\n")
            .expect_err("zero scale");
        assert!(err.to_string().contains("metersPerUnit"), "{err}");
    }

    #[test]
    fn a_joint_pointing_at_a_non_body_is_an_error() {
        let text = fixture("pendulum.usda").replace("</World/link>", "</World/ghost>");
        assert!(import_usda(&text).is_err());
    }
}
