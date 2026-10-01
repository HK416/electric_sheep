//! `SceneDesc` to MJCF text, for backends that consume MJCF (spec 17.2).
//!
//! A minimal emitter: bodies (with `gravcomp`), joints, primitive and mesh geoms, fixed
//! tendons, `<contact>` pairs and excludes, `motor` / `position` / `velocity` actuators,
//! `jointpos` / `jointvel` sensors and the whole `<option>`. Anything else is
//! [`PhysicsError::Unsupported`] naming the item, because spec 17.2 says a backend declares
//! what it cannot map instead of guessing an approximation.
//!
//! Mesh geoms are emitted *inline*: `<asset><mesh name vertex face>` carries the vertices
//! `es_assets::mesh::load` decoded, and the geom names that mesh. No `file=` path is written --
//! a path would have to reach this function through `PhysicsBackend::load(&SceneDesc)`, which
//! takes no such channel (INV-17), and a machine path inside a hashed struct is exactly what
//! spec 5.3 forbids. A mesh geom whose asset was never loaded is `PhysicsError::Unsupported`
//! naming it, before anything is spawned.
//!
//! Not emitted, because they carry no dynamics into `MuJoCo`: sites, cameras, materials and
//! rgba. A scene that *uses* a site (spatial tendon, site-mounted sensor or actuator) hits the
//! unsupported path above instead.
//!
//! Angles are written in radians (`<compiler angle="radian">`) and quaternions are converted
//! from the spec 3.1 xyzw to MJCF's wxyz.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_assets::gltf::MeshData;
use es_assets::scene::{
    Actuator, ActuatorKind, ActuatorTarget, AssetKind, Body, FrictionCone, Geom, Integrator,
    Jacobian, Joint, JointKind, SceneDesc, Sensor, SensorKind, SensorTarget, Shape, Solver, Tendon,
    TendonKind,
};
use es_core::StableId;
use es_math::{Quat, Vec3};
use es_physics_core::{IndexRange, ModelInfo, PhysicsError};

fn unsupported(what: impl Into<String>) -> PhysicsError {
    PhysicsError::Unsupported(what.into())
}

/// Shortest round-trip decimal form; `{:?}` on `f64` never prints an integer without a point,
/// so `MuJoCo` always parses it as a real.
fn num(v: f64) -> String {
    format!("{v:?}")
}

/// The `f32` sibling of [`num`]: mesh vertices are stored as `f32`, and `{:?}` is again the
/// shortest form that round-trips, so `MuJoCo` reads back the bits the content hash covers.
fn num32(v: f32) -> String {
    format!("{v:?}")
}

fn vec3(v: Vec3) -> String {
    format!("{} {} {}", num(v.x), num(v.y), num(v.z))
}

/// MJCF `quat` is wxyz (spec 3.1 is xyzw). `None` for the identity, which `MuJoCo` defaults to.
fn quat(q: Quat) -> Option<String> {
    (q != Quat::IDENTITY).then(|| format!("{} {} {} {}", num(q.w), num(q.x), num(q.y), num(q.z)))
}

/// XML attribute-value escaping. Names come from an imported file, so they are not trusted to
/// be attribute-safe.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Writes ` name="value"` when `value` is `Some`.
fn attr(out: &mut String, name: &str, value: Option<String>) {
    if let Some(value) = value {
        let _ = write!(out, " {name}=\"{value}\"");
    }
}

fn range_attr(range: Option<(f64, f64)>) -> Option<String> {
    range.map(|(lo, hi)| format!("{} {}", num(lo), num(hi)))
}

/// Non-zero scalars only: `MuJoCo`'s own defaults are zero, so writing them back adds noise.
fn nonzero(v: f64) -> Option<String> {
    (v != 0.0).then(|| num(v))
}

/// Emits `scene` as MJCF.
pub fn scene_to_mjcf(scene: &SceneDesc) -> Result<String, PhysicsError> {
    let joint_names: BTreeMap<StableId, &str> = scene
        .joints
        .iter()
        .map(|j| (j.id, j.name.as_str()))
        .collect();
    let mut joints_by_body: BTreeMap<StableId, Vec<&Joint>> = BTreeMap::new();
    for joint in &scene.joints {
        joints_by_body.entry(joint.body).or_default().push(joint);
    }
    let mut children: BTreeMap<StableId, Vec<&Body>> = BTreeMap::new();
    let mut roots: Vec<&Body> = Vec::new();
    for body in &scene.bodies {
        match body.parent {
            Some(parent) => children.entry(parent).or_default().push(body),
            None => roots.push(body),
        }
    }

    let mut out = String::new();
    let _ = writeln!(out, "<mujoco model=\"{}\">", esc(&scene.name));
    out.push_str("  <compiler angle=\"radian\"/>\n");
    // Every solver option is written out. Dropping one would be a silent semantic change,
    // which is exactly what spec 17.2 forbids.
    let options = &scene.options;
    let _ = writeln!(
        out,
        "  <option timestep=\"{}\" gravity=\"{}\" integrator=\"{}\" cone=\"{}\" \
         jacobian=\"{}\" solver=\"{}\" iterations=\"{}\" ls_iterations=\"{}\" \
         impratio=\"{}\">{}</option>",
        num(options.timestep),
        vec3(options.gravity),
        match options.integrator {
            Integrator::Euler => "Euler",
            Integrator::Rk4 => "RK4",
            Integrator::Implicit => "implicit",
            Integrator::ImplicitFast => "implicitfast",
        },
        match options.cone {
            FrictionCone::Pyramidal => "pyramidal",
            FrictionCone::Elliptic => "elliptic",
        },
        match options.jacobian {
            Jacobian::Auto => "auto",
            Jacobian::Dense => "dense",
            Jacobian::Sparse => "sparse",
        },
        match options.solver {
            Solver::Newton => "Newton",
            Solver::Cg => "CG",
            Solver::Pgs => "PGS",
        },
        options.iterations,
        options.ls_iterations,
        num(options.impratio),
        if options.eulerdamp {
            ""
        } else {
            "<flag eulerdamp=\"disable\"/>"
        }
    );
    write_meshes(&mut out, scene);
    out.push_str("  <worldbody>\n");
    for root in roots {
        if root.name == "world" {
            // The importer's root body *is* MJCF's implicit world body (P30 module docs), so
            // its contents go straight into <worldbody> rather than into a nested <body>.
            for geom in &root.geoms {
                write_geom(&mut out, &root.name, geom, 2, scene)?;
            }
            if let Some(joints) = joints_by_body.get(&root.id) {
                return Err(unsupported(format!(
                    "joint `{}` on the world body",
                    joints[0].name
                )));
            }
            for child in children.get(&root.id).into_iter().flatten() {
                write_body(&mut out, child, &children, &joints_by_body, 2, scene)?;
            }
        } else {
            write_body(&mut out, root, &children, &joints_by_body, 2, scene)?;
        }
    }
    out.push_str("  </worldbody>\n");
    write_contact(&mut out, scene)?;
    write_tendons(&mut out, &scene.tendons, &joint_names)?;
    write_actuators(&mut out, &scene.actuators, &joint_names)?;
    write_sensors(&mut out, &scene.sensors, &joint_names)?;
    out.push_str("</mujoco>\n");
    Ok(out)
}

/// The layout `MuJoCo` gives the model [`scene_to_mjcf`] writes, without loading it: bodies
/// numbered in the order they are written, the world first; each body's joints in the scene's
/// order (a fixed one is no joint); `qpos` / `qvel` addresses summed over them. Only `nq`, `nv`,
/// `nbody`, `qpos`, `dof` and `body` are filled: what reading a recorded state back needs
/// (design note `scene-authoring.md` section 4.8). `tests/layout.rs` holds it equal to
/// `mujoco-cpu`'s on every committed scene.
pub fn layout(scene: &SceneDesc) -> ModelInfo {
    fn walk<'a>(
        b: &'a Body,
        children: &BTreeMap<StableId, Vec<&'a Body>>,
        out: &mut Vec<&'a Body>,
    ) {
        out.push(b);
        for child in children.get(&b.id).into_iter().flatten() {
            walk(child, children, out);
        }
    }
    let mut children: BTreeMap<StableId, Vec<&Body>> = BTreeMap::new();
    for body in &scene.bodies {
        if let Some(parent) = body.parent {
            children.entry(parent).or_default().push(body);
        }
    }
    let mut info = ModelInfo {
        nbody: 1,
        ..ModelInfo::default()
    };
    let mut written = Vec::new();
    for root in scene.bodies.iter().filter(|b| b.parent.is_none()) {
        if root.name == "world" {
            info.body.insert(root.id, IndexRange::new(0, 1));
            for child in children.get(&root.id).into_iter().flatten() {
                walk(child, &children, &mut written);
            }
        } else {
            walk(root, &children, &mut written);
        }
    }
    for body in written {
        info.body.insert(body.id, IndexRange::new(info.nbody, 1));
        info.nbody += 1;
        for joint in scene.joints.iter().filter(|j| j.body == body.id) {
            let (nq, nv) = match joint.kind {
                JointKind::Free => (7, 6),
                JointKind::Ball => (4, 3),
                JointKind::Hinge | JointKind::Slide => (1, 1),
                JointKind::Fixed => continue,
            };
            info.qpos.insert(joint.id, IndexRange::new(info.nq, nq));
            info.dof.insert(joint.id, IndexRange::new(info.nv, nv));
            (info.nq, info.nv) = (info.nq + nq, info.nv + nv);
        }
    }
    info
}

/// `<asset><mesh name vertex face>` for every mesh the scene carries, inline.
///
/// `MuJoCo` builds the collision convex hull and the inertia from exactly these numbers, so the
/// `f32` bits the content hash covers (spec 5.3) are what reaches the solver; `{:?}` is the
/// shortest decimal that round-trips them. A scaled mesh's vertices are written scaled
/// (`SceneDesc::mesh_positions`, packet M17/R3), the ones the renderer draws. An unreferenced
/// mesh is still written: it is what the scene says it carries, and dropping it silently is
/// what spec 17.2 forbids.
fn write_meshes(out: &mut String, scene: &SceneDesc) {
    if scene.meshes.is_empty() {
        return;
    }
    out.push_str("  <asset>\n");
    for (id, data) in &scene.meshes {
        let positions = scene.mesh_positions(*id).unwrap_or_default();
        write_mesh(out, data, &positions);
    }
    out.push_str("  </asset>\n");
}

fn write_mesh(out: &mut String, data: &MeshData, positions: &[[f32; 3]]) {
    let _ = write!(out, "    <mesh name=\"{}\" vertex=\"", esc(&data.name));
    for (i, p) in positions.iter().enumerate() {
        for (k, c) in p.iter().enumerate() {
            if i + k > 0 {
                out.push(' ');
            }
            out.push_str(&num32(*c));
        }
    }
    out.push_str("\" face=\"");
    for (i, index) in data.indices.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "{index}");
    }
    out.push_str("\"/>\n");
}

fn write_body(
    out: &mut String,
    body: &Body,
    children: &BTreeMap<StableId, Vec<&Body>>,
    joints_by_body: &BTreeMap<StableId, Vec<&Joint>>,
    depth: usize,
    scene: &SceneDesc,
) -> Result<(), PhysicsError> {
    let pad = "  ".repeat(depth);
    let _ = write!(out, "{pad}<body name=\"{}\"", esc(&body.name));
    attr(out, "pos", Some(vec3(body.pose.position)));
    attr(out, "quat", quat(body.pose.orientation));
    attr(
        out,
        "gravcomp",
        scene.gravcomp.get(&body.id).map(|g| num(*g)),
    );
    out.push_str(">\n");

    if let Some(inertial) = &body.inertial {
        let m = inertial.inertia.matrix();
        let diagonal = m[0][1] == 0.0 && m[0][2] == 0.0 && m[1][2] == 0.0;
        let _ = write!(out, "{pad}  <inertial pos=\"{}\"", vec3(inertial.com));
        if diagonal {
            attr(out, "quat", quat(inertial.frame));
            attr(
                out,
                "diaginertia",
                Some(format!(
                    "{} {} {}",
                    num(m[0][0]),
                    num(m[1][1]),
                    num(m[2][2])
                )),
            );
        } else if inertial.frame == Quat::IDENTITY {
            attr(
                out,
                "fullinertia",
                Some(format!(
                    "{} {} {} {} {} {}",
                    num(m[0][0]),
                    num(m[1][1]),
                    num(m[2][2]),
                    num(m[0][1]),
                    num(m[0][2]),
                    num(m[1][2])
                )),
            );
        } else {
            // MJCF's `fullinertia` overwrites the inertial frame, so the two cannot be
            // combined; the scene would silently lose the rotation (spec 17.2).
            return Err(unsupported(format!(
                "body `{}`: a rotated inertial frame with a non-diagonal inertia",
                body.name
            )));
        }
        let _ = writeln!(out, " mass=\"{}\"/>", num(inertial.mass));
    }

    for joint in joints_by_body.get(&body.id).into_iter().flatten() {
        write_joint(out, joint, depth + 1);
    }
    for geom in &body.geoms {
        write_geom(out, &body.name, geom, depth + 1, scene)?;
    }
    for child in children.get(&body.id).into_iter().flatten() {
        write_body(out, child, children, joints_by_body, depth + 1, scene)?;
    }
    let _ = writeln!(out, "{pad}</body>");
    Ok(())
}

fn write_joint(out: &mut String, joint: &Joint, depth: usize) {
    let pad = "  ".repeat(depth);
    let kind = match joint.kind {
        JointKind::Hinge => "hinge",
        JointKind::Slide => "slide",
        JointKind::Ball => "ball",
        JointKind::Free => "free",
        // A welded body is spelled in MJCF as a body with no joint at all (P30 module docs).
        JointKind::Fixed => return,
    };
    let _ = write!(
        out,
        "{pad}<joint name=\"{}\" type=\"{kind}\"",
        esc(&joint.name)
    );
    if joint.kind == JointKind::Free {
        // `MuJoCo` ignores every other attribute on a free joint.
        out.push_str("/>\n");
        return;
    }
    if matches!(joint.kind, JointKind::Hinge | JointKind::Slide) {
        attr(out, "axis", Some(vec3(joint.axis)));
        attr(out, "range", range_attr(joint.range));
    }
    attr(out, "pos", Some(vec3(joint.anchor)));
    attr(out, "damping", nonzero(joint.damping));
    attr(out, "armature", nonzero(joint.armature));
    attr(out, "stiffness", nonzero(joint.stiffness));
    attr(out, "springref", nonzero(joint.spring_ref));
    attr(out, "frictionloss", nonzero(joint.friction_loss));
    out.push_str("/>\n");
}

fn write_geom(
    out: &mut String,
    owner: &str,
    geom: &Geom,
    depth: usize,
    scene: &SceneDesc,
) -> Result<(), PhysicsError> {
    // `Some(mesh name)` for a mesh geom, whose vertices are its size, so it writes no `size`.
    let mut mesh = None;
    let (kind, size) = match geom.shape {
        Shape::Plane {
            half_x,
            half_y,
            grid,
        } => (
            "plane",
            // `MuJoCo` requires a positive grid spacing; 0 is the importer's "unset".
            format!(
                "{} {} {}",
                num(half_x),
                num(half_y),
                num(if grid > 0.0 { grid } else { 0.05 })
            ),
        ),
        Shape::Sphere { radius } => ("sphere", num(radius)),
        Shape::Capsule {
            radius,
            half_length,
        } => ("capsule", format!("{} {}", num(radius), num(half_length))),
        Shape::Cylinder {
            radius,
            half_length,
        } => ("cylinder", format!("{} {}", num(radius), num(half_length))),
        Shape::Box { half_extents } => ("box", vec3(half_extents)),
        Shape::Ellipsoid { radii } => ("ellipsoid", vec3(radii)),
        Shape::Mesh { asset } => {
            // The geometry has to be on the scene already: `PhysicsBackend::load` is handed a
            // `&SceneDesc` and nothing else, so a file this function could open does not exist
            // as far as the trait is concerned (INV-17).
            let data = scene.meshes.get(&asset).ok_or_else(|| {
                let named = scene
                    .assets
                    .iter()
                    .find(|a| a.id == asset && a.kind == AssetKind::Mesh)
                    .map_or("<unknown>", |a| a.name.as_str());
                unsupported(format!(
                    "geom `{}`: mesh `{named}` is not loaded (es_assets::mesh::load)",
                    geom.name
                ))
            })?;
            mesh = Some(esc(&data.name));
            ("mesh", String::new())
        }
        Shape::HeightField { .. } => {
            return Err(unsupported(format!("geom `{}`: height field", geom.name)));
        }
    };
    let pad = "  ".repeat(depth);
    let name = geom_label(owner, geom);
    let _ = write!(out, "{pad}<geom name=\"{}\" type=\"{kind}\"", esc(&name));
    attr(out, "size", (!size.is_empty()).then_some(size));
    attr(out, "mesh", mesh);
    attr(out, "pos", Some(vec3(geom.pose.position)));
    attr(out, "quat", quat(geom.pose.orientation));
    attr(
        out,
        "friction",
        Some(format!(
            "{} {} {}",
            num(geom.friction[0]),
            num(geom.friction[1]),
            num(geom.friction[2])
        )),
    );
    let (contype, conaffinity) = if geom.visual_only {
        (0, 0)
    } else {
        (geom.contype, geom.conaffinity)
    };
    attr(out, "contype", Some(contype.to_string()));
    attr(out, "conaffinity", Some(conaffinity.to_string()));
    attr(out, "condim", Some(geom.condim.to_string()));
    // Contact parameter precedence: omitted at the default so the emitted XML stays the
    // shape upstream wrote, emitted otherwise because it decides whose friction wins.
    attr(
        out,
        "priority",
        (geom.priority != 0).then(|| geom.priority.to_string()),
    );
    match geom.mass {
        Some(mass) => attr(out, "mass", Some(num(mass))),
        None => attr(out, "density", Some(num(geom.density))),
    }
    attr(out, "margin", nonzero(geom.margin));
    attr(out, "gap", nonzero(geom.gap));
    attr(
        out,
        "solref",
        Some(format!("{} {}", num(geom.solref[0]), num(geom.solref[1]))),
    );
    attr(
        out,
        "solimp",
        Some(
            geom.solimp
                .iter()
                .map(|v| num(*v))
                .collect::<Vec<_>>()
                .join(" "),
        ),
    );
    out.push_str("/>\n");
    Ok(())
}

/// The name a geom is emitted under. `MuJoCo` requires geom names to be unique across the
/// model, while the scene names a geom the source left unnamed `geom<n>` with `n` counted per
/// owner (`es_assets::scene`), so two bodies with unnamed geoms both carry a `geom1`: an
/// auto-generated name is qualified by its owner, a source name is written verbatim. A
/// `<contact><pair>` refers to geoms through this same function.
fn geom_label(owner: &str, geom: &Geom) -> String {
    let auto_named = geom
        .name
        .strip_prefix("geom")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if auto_named {
        format!("{owner}.{}", geom.name)
    } else {
        geom.name.clone()
    }
}

/// `<contact>`: every `<pair>` (with the parameters it states) and `<exclude>`.
fn write_contact(out: &mut String, scene: &SceneDesc) -> Result<(), PhysicsError> {
    if scene.contact_pairs.is_empty() && scene.contact_excludes.is_empty() {
        return Ok(());
    }
    let geoms: BTreeMap<StableId, String> = scene
        .bodies
        .iter()
        .flat_map(|b| b.geoms.iter().map(move |g| (g.id, geom_label(&b.name, g))))
        .collect();
    let bodies: BTreeMap<StableId, &str> = scene
        .bodies
        .iter()
        .map(|b| (b.id, b.name.as_str()))
        .collect();
    let missing = |id: &StableId| unsupported(format!("contact: {id} is not in the scene"));
    let list = |values: &[f64]| values.iter().map(|v| num(*v)).collect::<Vec<_>>().join(" ");
    out.push_str("  <contact>\n");
    for pair in &scene.contact_pairs {
        let (a, b) = (
            geoms.get(&pair.geom1).ok_or_else(|| missing(&pair.geom1))?,
            geoms.get(&pair.geom2).ok_or_else(|| missing(&pair.geom2))?,
        );
        let _ = write!(out, "    <pair geom1=\"{}\" geom2=\"{}\"", esc(a), esc(b));
        attr(out, "condim", pair.condim.map(|d| d.to_string()));
        attr(out, "friction", pair.friction.map(|v| list(&v)));
        attr(out, "solref", pair.solref.map(|v| list(&v)));
        attr(out, "solimp", pair.solimp.map(|v| list(&v)));
        attr(out, "margin", pair.margin.map(num));
        attr(out, "gap", pair.gap.map(num));
        out.push_str("/>\n");
    }
    for (a, b) in &scene.contact_excludes {
        let (a, b) = (
            bodies.get(a).ok_or_else(|| missing(a))?,
            bodies.get(b).ok_or_else(|| missing(b))?,
        );
        let _ = writeln!(
            out,
            "    <exclude body1=\"{}\" body2=\"{}\"/>",
            esc(a),
            esc(b)
        );
    }
    out.push_str("  </contact>\n");
    Ok(())
}

/// `<tendon>`: fixed tendons. A spatial tendon runs through sites, which this emitter does not
/// write, so it is refused by name.
fn write_tendons(
    out: &mut String,
    tendons: &[Tendon],
    joint_names: &BTreeMap<StableId, &str>,
) -> Result<(), PhysicsError> {
    if tendons.is_empty() {
        return Ok(());
    }
    out.push_str("  <tendon>\n");
    for tendon in tendons {
        let TendonKind::Fixed { joints } = &tendon.kind else {
            return Err(unsupported(format!(
                "tendon `{}`: spatial (this backend emits no sites)",
                tendon.name
            )));
        };
        let _ = write!(out, "    <fixed name=\"{}\"", esc(&tendon.name));
        attr(out, "limited", tendon.range.map(|_| "true".to_owned()));
        attr(out, "range", range_attr(tendon.range));
        attr(out, "stiffness", nonzero(tendon.stiffness));
        attr(out, "damping", nonzero(tendon.damping));
        out.push_str(">\n");
        for (joint, coef) in joints {
            let name = joint_names.get(joint).ok_or_else(|| {
                unsupported(format!(
                    "tendon `{}`: a joint that is not in the scene",
                    tendon.name
                ))
            })?;
            let _ = writeln!(
                out,
                "      <joint joint=\"{}\" coef=\"{}\"/>",
                esc(name),
                num(*coef)
            );
        }
        out.push_str("    </fixed>\n");
    }
    out.push_str("  </tendon>\n");
    Ok(())
}

/// `gear` with its trailing zeros trimmed; `MuJoCo` pads the rest with zeros itself.
fn gear_attr(gear: [f64; 6]) -> String {
    let last = gear.iter().rposition(|v| *v != 0.0).unwrap_or(0);
    gear[..=last]
        .iter()
        .map(|v| num(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_actuators(
    out: &mut String,
    actuators: &[Actuator],
    joint_names: &BTreeMap<StableId, &str>,
) -> Result<(), PhysicsError> {
    if actuators.is_empty() {
        return Ok(());
    }
    out.push_str("  <actuator>\n");
    for actuator in actuators {
        let joint = match actuator.target {
            ActuatorTarget::Joint(id) => joint_names.get(&id).copied().ok_or_else(|| {
                unsupported(format!(
                    "actuator `{}`: target joint is not in the scene",
                    actuator.name
                ))
            })?,
            ActuatorTarget::Tendon(_) => {
                return Err(unsupported(format!(
                    "actuator `{}`: tendon transmission",
                    actuator.name
                )))
            }
            ActuatorTarget::Site(_) => {
                return Err(unsupported(format!(
                    "actuator `{}`: site transmission",
                    actuator.name
                )))
            }
        };
        let kind = match actuator.kind {
            ActuatorKind::Motor => "motor",
            ActuatorKind::Position { .. } => "position",
            ActuatorKind::Velocity { .. } => "velocity",
            ActuatorKind::General { .. } => {
                return Err(unsupported(format!(
                    "actuator `{}`: general gain/bias form",
                    actuator.name
                )))
            }
        };
        let _ = write!(
            out,
            "    <{kind} name=\"{}\" joint=\"{}\"",
            esc(&actuator.name),
            esc(joint)
        );
        match actuator.kind {
            ActuatorKind::Position { kp, kv } => {
                attr(out, "kp", Some(num(kp)));
                attr(out, "kv", nonzero(kv));
            }
            ActuatorKind::Velocity { kv } => attr(out, "kv", Some(num(kv))),
            _ => {}
        }
        attr(out, "gear", Some(gear_attr(actuator.gear)));
        attr(out, "ctrlrange", range_attr(actuator.ctrl_range));
        attr(out, "forcerange", range_attr(actuator.force_range));
        out.push_str("/>\n");
    }
    out.push_str("  </actuator>\n");
    Ok(())
}

fn write_sensors(
    out: &mut String,
    sensors: &[Sensor],
    joint_names: &BTreeMap<StableId, &str>,
) -> Result<(), PhysicsError> {
    if sensors.is_empty() {
        return Ok(());
    }
    out.push_str("  <sensor>\n");
    for sensor in sensors {
        let kind = match sensor.kind {
            SensorKind::JointPos => "jointpos",
            SensorKind::JointVel => "jointvel",
            other => return Err(unsupported(format!("sensor `{}`: {other:?}", sensor.name))),
        };
        let SensorTarget::Joint(id) = sensor.target else {
            return Err(unsupported(format!(
                "sensor `{}`: {:?} target",
                sensor.name, sensor.target
            )));
        };
        let joint = joint_names.get(&id).copied().ok_or_else(|| {
            unsupported(format!(
                "sensor `{}`: target joint is not in the scene",
                sensor.name
            ))
        })?;
        let _ = write!(
            out,
            "    <{kind} name=\"{}\" joint=\"{}\"",
            esc(&sensor.name),
            esc(joint)
        );
        attr(out, "noise", nonzero(sensor.noise));
        attr(out, "cutoff", nonzero(sensor.cutoff));
        out.push_str("/>\n");
    }
    out.push_str("  </sensor>\n");
    Ok(())
}

#[cfg(test)]
// Values must survive the round trip exactly; that exactness is the property under test.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::tests_support::fixture;

    /// Parse, emit, re-parse: the counts and the kinds have to survive the round trip.
    fn round_trip(mjcf: &str) -> (SceneDesc, SceneDesc) {
        let before = es_assets::parse_mjcf(mjcf).unwrap().scene;
        let text = scene_to_mjcf(&before).unwrap();
        let after = es_assets::parse_mjcf(&text)
            .unwrap_or_else(|e| panic!("re-parsing emitted MJCF failed: {e}\n{text}"))
            .scene;
        (before, after)
    }

    #[test]
    fn pendulum_round_trips() {
        let (before, after) = round_trip(&fixture("pendulum.xml"));
        assert_eq!(after.bodies.len(), before.bodies.len());
        assert_eq!(after.joints.len(), 1);
        assert_eq!(after.joints[0].kind, JointKind::Hinge);
        // Degrees in, radians out: the range survives the unit change.
        let (lo, hi) = after.joints[0].range.unwrap();
        assert!((hi - std::f64::consts::FRAC_PI_2).abs() < 1e-12 && (lo + hi).abs() < 1e-12);
        assert_eq!(after.joints[0].armature, before.joints[0].armature);
        let inertial = after.bodies[1].inertial.as_ref().unwrap();
        assert_eq!(inertial.mass, 1.5);
        assert_eq!(
            inertial.inertia.matrix()[2][2],
            before.bodies[1].inertial.as_ref().unwrap().inertia.matrix()[2][2]
        );
    }

    #[test]
    fn arm2_round_trips() {
        let (before, after) = round_trip(&fixture("arm2.xml"));
        assert_eq!(after.bodies.len(), before.bodies.len());
        assert_eq!(after.joints.len(), before.joints.len());
        let geoms: usize = after.bodies.iter().map(|b| b.geoms.len()).sum();
        assert_eq!(
            geoms,
            before.bodies.iter().map(|b| b.geoms.len()).sum::<usize>()
        );
        // Defaults are resolved by the importer, so the emitted file carries them explicitly.
        assert_eq!(after.joints[1].damping, 1.0);
    }

    #[test]
    fn actuators_and_sensors_round_trip() {
        let (before, after) = round_trip(
            r#"<mujoco>
                 <worldbody>
                   <body name="b">
                     <joint name="j" type="slide" axis="1 0 0" range="-1 1"/>
                     <geom name="g" type="box" size="0.1 0.2 0.3" mass="2" priority="2"/>
                   </body>
                 </worldbody>
                 <actuator>
                   <motor name="m" joint="j" gear="10" ctrlrange="-1 1" forcerange="-50 50"/>
                   <position name="p" joint="j" kp="30" kv="2"/>
                   <velocity name="v" joint="j" kv="4"/>
                   <position name="i" joint="j" kp="30" inheritrange="0.5"/>
                 </actuator>
                 <sensor>
                   <jointpos joint="j" noise="0.001"/>
                   <jointvel name="jv" joint="j" cutoff="10"/>
                 </sensor>
               </mujoco>"#,
        );
        assert_eq!(after.actuators.len(), 4);
        assert_eq!(after.sensors.len(), 2);
        assert_eq!(after.actuators[0].gear, before.actuators[0].gear);
        assert_eq!(after.actuators[0].ctrl_range, Some((-1.0, 1.0)));
        assert_eq!(
            after.actuators[1].kind,
            ActuatorKind::Position { kp: 30.0, kv: 2.0 }
        );
        assert_eq!(after.sensors[1].cutoff, 10.0);
        assert_eq!(after.bodies[1].geoms[0].mass, Some(2.0));
        // Contact precedence and the compile-time `inheritrange` rewrite both survive: the
        // joint's own (-1, 1) scaled by 0.5 about its midpoint (packet M6/B1b).
        assert_eq!(after.bodies[1].geoms[0].priority, 2);
        assert_eq!(before.actuators[3].ctrl_range, Some((-0.5, 0.5)));
        assert_eq!(after.actuators[3].ctrl_range, Some((-0.5, 0.5)));
    }

    /// `tests/fixtures/mjcf/mesh_box.xml` with its STL resolved: the emitted file must carry
    /// the vertices themselves, because `PhysicsBackend::load` gets a `&SceneDesc` and no path
    /// channel (INV-17).
    fn mesh_box() -> SceneDesc {
        let dir = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/mjcf"
        ));
        let mut scene = es_assets::parse_mjcf(&fixture("mesh_box.xml"))
            .unwrap()
            .scene;
        es_assets::mesh::load(&mut scene, &dir).expect("the fixture STL loads");
        scene
    }

    #[test]
    fn mesh_geom_emits_inline_vertex_and_face() {
        let scene = mesh_box();
        let text = scene_to_mjcf(&scene).expect("a loaded mesh scene emits");
        assert!(text.contains("<mesh name=\"box\" vertex=\""), "{text}");
        assert!(text.contains("<geom name=\"mesh_geom\" type=\"mesh\" mesh=\"box\""));
        // A mesh geom's vertices are its size, so no `size=` is written for it.
        let geom = text
            .lines()
            .find(|l| l.contains("name=\"mesh_geom\""))
            .unwrap();
        assert!(!geom.contains("size="), "{geom}");
        // Eight corners and twelve triangles, as f32 decimals and 0-based indices.
        let mesh = text.lines().find(|l| l.contains("<mesh ")).unwrap();
        let count = |attr: &str| {
            mesh.split_once(&format!("{attr}=\""))
                .unwrap()
                .1
                .split_once('"')
                .unwrap()
                .0
                .split_whitespace()
                .count()
        };
        assert_eq!(count("vertex"), 24);
        assert_eq!(count("face"), 36);
        assert!(mesh.contains("0.05") && mesh.contains("-0.05"), "{mesh}");
        // And the whole file re-parses, so nothing about it is malformed XML.
        es_assets::parse_mjcf(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    }

    #[test]
    fn unloaded_mesh_is_refused_by_name() {
        let scene = es_assets::parse_mjcf(&fixture("mesh_box.xml"))
            .unwrap()
            .scene;
        let err = scene_to_mjcf(&scene).expect_err("nothing resolved the STL");
        let message = err.to_string();
        assert!(
            message.contains("mesh `box` is not loaded"),
            "{message}, it must name the mesh and the way out"
        );
        assert!(message.contains("es_assets::mesh::load"), "{message}");
        assert!(matches!(err, PhysicsError::Unsupported(_)));
    }

    #[test]
    fn every_unmappable_item_is_named() {
        let cases = [
            (
                r#"<mujoco><asset><mesh name="m" file="m.obj"/></asset><worldbody><body name="b">
                     <geom name="g" type="mesh" mesh="m"/></body></worldbody></mujoco>"#,
                "geom `g`: mesh `m` is not loaded",
            ),
            (
                r#"<mujoco><worldbody><body name="b"><site name="s"/>
                     <geom name="g" type="sphere" size="1"/></body></worldbody>
                   <tendon><spatial name="t"><site site="s"/><site site="s"/></spatial></tendon>
                   </mujoco>"#,
                "tendon `t`",
            ),
            (
                r#"<mujoco><worldbody><body name="b"><site name="s"/>
                     <joint name="j" type="hinge"/>
                     <geom name="g" type="sphere" size="1"/></body></worldbody>
                   <sensor><accelerometer name="a" site="s"/></sensor></mujoco>"#,
                "sensor `a`",
            ),
            (
                r#"<mujoco><worldbody><body name="b">
                     <joint name="j" type="hinge"/>
                     <geom name="g" type="sphere" size="1"/></body></worldbody>
                   <actuator><general name="ga" joint="j" gainprm="5"/></actuator></mujoco>"#,
                "actuator `ga`: general",
            ),
        ];
        for (mjcf, expected) in cases {
            let scene = es_assets::parse_mjcf(mjcf).unwrap().scene;
            let err = scene_to_mjcf(&scene).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains(expected),
                "expected `{expected}` in `{message}`"
            );
            assert!(matches!(err, PhysicsError::Unsupported(_)));
        }
    }

    #[test]
    fn names_are_escaped_and_quaternions_are_wxyz() {
        assert_eq!(esc("a&b\"<c>"), "a&amp;b&quot;&lt;c&gt;");
        assert_eq!(quat(Quat::IDENTITY), None);
        assert_eq!(
            quat(Quat::from_xyzw(0.0, 0.0, 1.0, 0.0)),
            Some("0.0 0.0 0.0 1.0".to_owned())
        );
        assert_eq!(gear_attr([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "2.0");
        assert_eq!(gear_attr([1.0, 0.0, 3.0, 0.0, 0.0, 0.0]), "1.0 0.0 3.0");
    }
}
