//! [`expand`]: a scene document to a [`SceneDesc`].
//!
//! Order is part of the result (`MuJoCo`'s body, dof and actuator order follow the lists): the
//! world body first, then every `[[include]]` in turn (its world-level geoms, sites and cameras
//! join the world, its bodies follow), then the document's own entities — world geoms and
//! lights, textures and materials, bodies, cameras, regions — each in document order. Every
//! element is built with the values the MJCF reader gives the same element, so a document that
//! writes what an MJCF file says reads to the same `SceneDesc`.

// `s`, `b`, `g`, `i` are the scene, body, geom and inertial being built; the inertia components
// keep MJCF's positional order.
#![allow(clippy::many_single_char_names)]

use std::collections::BTreeSet;
use std::path::Path;

use es_core::StableId;
use es_math::units::DEG_TO_RAD;
use es_math::{Inertia, Pose, Quat, Vec3};

use super::{
    refuse, BodyDoc, BuiltinDoc, ColorSpaceDoc, ConeDoc, EsScene, EsSceneError, GeomDoc,
    IntegratorDoc, JacobianDoc, JointKindDoc, MarkDoc, MaterialDoc, ShapeDoc, SolverDoc,
    TexKindDoc, TextureDoc,
};
use crate::scene::{
    scene_id, AssetKind, AssetRef, Body, BodyInertial, Camera, FrictionCone, Geom, Integrator,
    Jacobian, Joint, JointKind, Material, PhysicsOptions, SceneDesc, Shape, Site, Solver,
};
use crate::texture::{Builtin, ColorSpace, Mark, TexKind, Texture, TextureSpec};

/// Thickness of a `[[light]]`'s emissive box, m: the committed scenes' emitters.
const LIGHT_HALF_THICKNESS: f64 = 0.005;

/// Expands `doc`, resolving every file against `base_dir` (the document's directory), then
/// loads the meshes and textures as `es_tools::load_scene` does for any scene file.
pub fn expand(doc: &EsScene, base_dir: &Path) -> Result<SceneDesc, EsSceneError> {
    unique_names(doc)?;
    let mut s = SceneDesc {
        name: doc.name.clone().unwrap_or_else(|| "scene".to_owned()),
        options: physics(doc),
        bodies: vec![Body {
            id: world(),
            name: "world".to_owned(),
            parent: None,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms: Vec::new(),
            sites: Vec::new(),
        }],
        ..SceneDesc::default()
    };
    for inc in &doc.includes {
        super::include::expand_include(&mut s, inc, doc, base_dir)?;
    }
    for t in &doc.textures {
        texture(&mut s, t)?;
    }
    for m in &doc.materials {
        material(&mut s, m)?;
    }
    let mut counter = 0;
    for g in &doc.geoms {
        let geom = geom(&mut s, g, "world", &mut counter, "geom")?;
        s.bodies[0].geoms.push(geom);
    }
    for l in &doc.lights {
        let field = format!("light[{}]", l.name);
        let name = format!("{}_light", l.name);
        let [hx, hy] = l.size;
        let shape = Shape::Box {
            half_extents: Vec3::new(hx, hy, LIGHT_HALF_THICKNESS),
        };
        let mut g = default_geom(scene_id("geom", &format!("world/{name}")), name, shape);
        g.pose = pose(l.pos, l.quat, &field)?;
        let i = l.intensity.unwrap_or(1.0);
        let [r, gr, b] = l.rgb.unwrap_or([1.0; 3]);
        g.rgba = [r * i, gr * i, b * i, 1.0];
        (g.contype, g.conaffinity, g.visual_only) = (0, 0, true);
        s.bodies[0].geoms.push(g);
    }
    for b in &doc.bodies {
        body(&mut s, b)?;
    }
    for c in &doc.cameras {
        let field = format!("camera[{}]", c.name);
        let (body, path) = owner(&s, c.parent.as_deref(), &field)?;
        s.cameras.push(Camera {
            id: scene_id("camera", &format!("{path}/{}", c.name)),
            name: c.name.clone(),
            body: (body != world()).then_some(body),
            pose: pose(c.pos, c.quat, &field)?,
            fovy: c.fovy.unwrap_or(45.0) * DEG_TO_RAD,
        });
    }
    for r in &doc.regions {
        let field = format!("region[{}]", r.name);
        let (body, path) = owner(&s, r.parent.as_deref(), &field)?;
        let site = Site {
            id: scene_id("site", &format!("{path}/{}", r.name)),
            name: r.name.clone(),
            pose: pose(r.pos, r.quat, &field)?,
            size: vec3(r.size, Vec3::new(0.005, 0.005, 0.005)),
        };
        let at = s
            .bodies
            .iter()
            .position(|b| b.id == body)
            .expect("owner found it");
        s.bodies[at].sites.push(site);
    }
    s.validate()?;
    crate::mesh::load(&mut s, base_dir)?;
    Ok(s)
}

pub(super) fn world() -> StableId {
    scene_id("body", "world")
}

/// A second entity of one kind with one name is refused at its field, before anything expands.
fn unique_names(doc: &EsScene) -> Result<(), EsSceneError> {
    let lists: [(&str, Vec<&str>); 7] = [
        (
            "include",
            doc.includes.iter().map(|x| x.name.as_str()).collect(),
        ),
        ("body", doc.bodies.iter().map(|x| x.name.as_str()).collect()),
        (
            "texture",
            doc.textures.iter().map(|x| x.name.as_str()).collect(),
        ),
        (
            "material",
            doc.materials.iter().map(|x| x.name.as_str()).collect(),
        ),
        (
            "camera",
            doc.cameras.iter().map(|x| x.name.as_str()).collect(),
        ),
        (
            "light",
            doc.lights.iter().map(|x| x.name.as_str()).collect(),
        ),
        (
            "region",
            doc.regions.iter().map(|x| x.name.as_str()).collect(),
        ),
    ];
    for (kind, names) in lists {
        let mut seen = BTreeSet::new();
        if let Some(dup) = names.into_iter().find(|n| !seen.insert(*n)) {
            return refuse(
                format!("{kind}[{dup}].name"),
                format!("a second {kind} is named `{dup}`"),
            );
        }
    }
    Ok(())
}

fn physics(doc: &EsScene) -> PhysicsOptions {
    let d = PhysicsOptions::default();
    let Some(p) = &doc.physics else { return d };
    PhysicsOptions {
        timestep: p.timestep.unwrap_or(d.timestep),
        gravity: vec3(p.gravity, d.gravity),
        integrator: p.integrator.map_or(d.integrator, |i| match i {
            IntegratorDoc::Euler => Integrator::Euler,
            IntegratorDoc::Rk4 => Integrator::Rk4,
            IntegratorDoc::Implicit => Integrator::Implicit,
            IntegratorDoc::ImplicitFast => Integrator::ImplicitFast,
        }),
        cone: p.cone.map_or(d.cone, |c| match c {
            ConeDoc::Pyramidal => FrictionCone::Pyramidal,
            ConeDoc::Elliptic => FrictionCone::Elliptic,
        }),
        jacobian: p.jacobian.map_or(d.jacobian, |j| match j {
            JacobianDoc::Dense => Jacobian::Dense,
            JacobianDoc::Sparse => Jacobian::Sparse,
            JacobianDoc::Auto => Jacobian::Auto,
        }),
        solver: p.solver.map_or(d.solver, |s| match s {
            SolverDoc::Pgs => Solver::Pgs,
            SolverDoc::Cg => Solver::Cg,
            SolverDoc::Newton => Solver::Newton,
        }),
        iterations: p.iterations.unwrap_or(d.iterations),
        ls_iterations: p.ls_iterations.unwrap_or(d.ls_iterations),
        eulerdamp: p.eulerdamp.unwrap_or(d.eulerdamp),
        impratio: p.impratio.unwrap_or(d.impratio),
    }
}

pub(super) fn vec3(v: Option<[f64; 3]>, default: Vec3) -> Vec3 {
    v.map_or(default, |[x, y, z]| Vec3::new(x, y, z))
}

/// A quaternion as written, bit for bit, when it is already canonical (unit within 1e-12,
/// `w >= 0`) — the editor writes `SceneDesc`'s own bits back, and `Quat::normalize` is a fixed
/// point only to within a rounding; anything else is normalised.
pub(super) fn quat(q: Option<[f64; 4]>, field: &str) -> Result<Quat, EsSceneError> {
    let Some([x, y, z, w]) = q else {
        return Ok(Quat::IDENTITY);
    };
    let q = Quat::from_xyzw(x, y, z, w);
    let n = q.norm();
    if !n.is_finite() || n == 0.0 {
        return refuse(
            format!("{field}.quat"),
            "not a rotation (zero or not finite)",
        );
    }
    Ok(if w >= 0.0 && (n - 1.0).abs() <= 1e-12 {
        q
    } else {
        q.normalize()
    })
}

pub(super) fn pose(
    pos: Option<[f64; 3]>,
    q: Option<[f64; 4]>,
    field: &str,
) -> Result<Pose, EsSceneError> {
    Ok(Pose {
        position: vec3(pos, Vec3::ZERO),
        orientation: quat(q, field)?,
    })
}

/// The `/`-joined body names from the world down to `id` (the MJCF id scheme's body path).
pub(super) fn body_path(s: &SceneDesc, id: StableId) -> String {
    let mut names = Vec::new();
    let mut cursor = Some(id);
    while let Some(b) = cursor.and_then(|c| s.bodies.iter().find(|b| b.id == c)) {
        names.push(b.name.as_str());
        cursor = b.parent;
    }
    if names.last() != Some(&"world") {
        names.push("world");
    }
    names.reverse();
    names.join("/")
}

/// The body a camera or region hangs from, and its path; absent is the world.
fn owner(
    s: &SceneDesc,
    parent: Option<&str>,
    field: &str,
) -> Result<(StableId, String), EsSceneError> {
    let Some(name) = parent else {
        return Ok((world(), "world".to_owned()));
    };
    match s.bodies.iter().find(|b| b.name == name) {
        Some(b) => Ok((b.id, body_path(s, b.id))),
        None => refuse(
            format!("{field}.parent"),
            format!("no body is named `{name}`"),
        ),
    }
}

/// An MJCF `<geom>` with nothing but its name and shape written.
pub(super) fn default_geom(id: StableId, name: String, shape: Shape) -> Geom {
    Geom {
        id,
        name,
        shape,
        pose: Pose::IDENTITY,
        friction: [1.0, 0.005, 0.0001],
        contype: 1,
        conaffinity: 1,
        condim: 3,
        priority: 0,
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

/// The id of the asset of `kind` named `name`, or the refusal at `field`.
pub(super) fn asset(
    s: &SceneDesc,
    kind: AssetKind,
    name: &str,
    field: &str,
) -> Result<StableId, EsSceneError> {
    match s.assets.iter().find(|a| a.kind == kind && a.name == name) {
        Some(a) => Ok(a.id),
        None => refuse(field, format!("no {} is named `{name}`", kind.tag())),
    }
}

fn geom(
    s: &mut SceneDesc,
    g: &GeomDoc,
    body_path: &str,
    unnamed: &mut u32,
    field: &str,
) -> Result<Geom, EsSceneError> {
    let name = g.name.clone().unwrap_or_else(|| {
        *unnamed += 1;
        format!("geom{unnamed}")
    });
    let field = format!("{field}[{name}]");
    let shape = match &g.shape {
        ShapeDoc::Plane([half_x, half_y, grid]) => Shape::Plane {
            half_x: *half_x,
            half_y: *half_y,
            grid: *grid,
        },
        ShapeDoc::Sphere(radius) => Shape::Sphere { radius: *radius },
        ShapeDoc::Capsule([radius, half_length]) => Shape::Capsule {
            radius: *radius,
            half_length: *half_length,
        },
        ShapeDoc::Cylinder([radius, half_length]) => Shape::Cylinder {
            radius: *radius,
            half_length: *half_length,
        },
        ShapeDoc::Box(h) => Shape::Box {
            half_extents: vec3(Some(*h), Vec3::ZERO),
        },
        ShapeDoc::Ellipsoid(r) => Shape::Ellipsoid {
            radii: vec3(Some(*r), Vec3::ZERO),
        },
        ShapeDoc::Mesh { file, scale } => {
            if scale.is_some_and(|v| v.iter().any(|x| !(x.is_finite() && *x > 0.0))) {
                return refuse(format!("{field}.shape.scale"), "a scale is positive");
            }
            Shape::Mesh {
                asset: mesh_asset(s, file, *scale),
            }
        }
    };
    let mut out = default_geom(
        scene_id("geom", &format!("{body_path}/{name}")),
        name,
        shape,
    );
    out.pose = pose(g.pos, g.quat, &field)?;
    out.mass = g.mass;
    out.density = g.density.unwrap_or(out.density);
    out.friction = g.friction.unwrap_or(out.friction);
    out.condim = g.condim.unwrap_or(out.condim);
    out.contype = g.contype.unwrap_or(out.contype);
    out.conaffinity = g.conaffinity.unwrap_or(out.conaffinity);
    out.priority = g.priority.unwrap_or(out.priority);
    out.margin = g.margin.unwrap_or(out.margin);
    out.gap = g.gap.unwrap_or(out.gap);
    out.solref = g.solref.unwrap_or(out.solref);
    out.solimp = g.solimp.unwrap_or(out.solimp);
    out.rgba = g.rgba.unwrap_or(out.rgba);
    out.visual_only = out.contype == 0 && out.conaffinity == 0;
    if let Some(m) = &g.material {
        out.material = Some(asset(
            s,
            AssetKind::Material,
            m,
            &format!("{field}.material"),
        )?);
    }
    Ok(out)
}

/// The mesh asset of `file` at `scale` (`None` and 1 alike: absent from `mesh_scales`), added on
/// first use and named by the file's stem — and `@` the scale when it has one (`part@0.001`,
/// `part@1,2,1`), so one file at two scales is two `<mesh>`es, as MJCF needs. The URDF reader
/// names its scaled meshes here too (packet M18/K8).
#[allow(clippy::float_cmp)] // 1 exactly is no scale; equal bits are one number in the name
pub(crate) fn mesh_asset(s: &mut SceneDesc, file: &str, scale: Option<[f64; 3]>) -> StableId {
    let scale = scale.filter(|v| *v != [1.0; 3]);
    let same = |a: &&AssetRef| {
        a.kind == AssetKind::Mesh && a.path == file && s.mesh_scales.get(&a.id) == scale.as_ref()
    };
    if let Some(a) = s.assets.iter().find(same) {
        return a.id;
    }
    let base = file.rsplit(['/', '\\']).next().unwrap_or(file);
    let stem = base.rsplit_once('.').map_or(base, |(stem, _)| stem);
    let name = match scale {
        None => stem.to_owned(),
        Some([x, y, z]) if x == y && y == z => format!("{stem}@{x}"),
        Some([x, y, z]) => format!("{stem}@{x},{y},{z}"),
    };
    let a = AssetRef::from_path(AssetKind::Mesh, &name, file);
    let id = a.id;
    s.assets.push(a);
    if let Some(v) = scale {
        s.mesh_scales.insert(id, v);
    }
    id
}

fn body(s: &mut SceneDesc, b: &BodyDoc) -> Result<(), EsSceneError> {
    let field = format!("body[{}]", b.name);
    let parent = match &b.parent {
        None => world(),
        // Bodies before this one only: the document order is the tree's order.
        Some(p) => match s.bodies.iter().find(|x| x.name == *p) {
            Some(x) => x.id,
            None => {
                return refuse(
                    format!("{field}.parent"),
                    format!("no body named `{p}` is defined before `{}`", b.name),
                )
            }
        },
    };
    let path = format!("{}/{}", body_path(s, parent), b.name);
    let id = scene_id("body", &path);
    let inertial = match &b.inertial {
        None => None,
        Some(i) => {
            let f = format!("{field}.inertial");
            let (inertia, frame) = match (i.diaginertia, i.fullinertia) {
                (Some([a, b, c]), None) => (Inertia::diagonal(a, b, c), quat(i.quat, &f)?),
                // `fullinertia` is in the body frame, as in MJCF.
                (None, Some([a, b, c, d, e, g])) => {
                    (Inertia::new(a, b, c, d, e, g), Quat::IDENTITY)
                }
                _ => return refuse(f, "write exactly one of `diaginertia`, `fullinertia`"),
            };
            Some(BodyInertial {
                mass: i.mass,
                com: vec3(i.pos, Vec3::ZERO),
                inertia,
                frame,
            })
        }
    };
    let mut unnamed = 0;
    let mut geoms = Vec::new();
    for g in &b.geoms {
        geoms.push(geom(s, g, &path, &mut unnamed, &format!("{field}.geom"))?);
    }
    if let Some(j) = &b.joint {
        let kind = match j.kind {
            JointKindDoc::Fixed => None,
            JointKindDoc::Free => Some(JointKind::Free),
            JointKindDoc::Ball => Some(JointKind::Ball),
            JointKindDoc::Hinge => Some(JointKind::Hinge),
            JointKindDoc::Slide => Some(JointKind::Slide),
        };
        if let Some(kind) = kind {
            let name = j.name.clone().unwrap_or_else(|| b.name.clone());
            s.joints.push(Joint {
                id: scene_id("joint", &format!("{path}/{name}")),
                name,
                body: id,
                kind,
                axis: vec3(j.axis, Vec3::new(0.0, 0.0, 1.0)).normalize(),
                anchor: vec3(j.pos, Vec3::ZERO),
                range: j.range.map(|[lo, hi]| (lo, hi)),
                damping: j.damping.unwrap_or(0.0),
                armature: j.armature.unwrap_or(0.0),
                stiffness: j.stiffness.unwrap_or(0.0),
                friction_loss: j.frictionloss.unwrap_or(0.0),
                spring_ref: j.springref.unwrap_or(0.0),
            });
        }
    }
    if let Some(g) = b.gravcomp.filter(|g| *g != 0.0) {
        s.gravcomp.insert(id, g);
    }
    s.bodies.push(Body {
        id,
        name: b.name.clone(),
        parent: Some(parent),
        pose: pose(b.pos, b.quat, &field)?,
        inertial,
        geoms,
        sites: Vec::new(),
    });
    Ok(())
}

fn texture(s: &mut SceneDesc, t: &TextureDoc) -> Result<(), EsSceneError> {
    let field = format!("texture[{}]", t.name);
    let d = TextureSpec::default();
    let gridsize = t.gridsize.unwrap_or(d.gridsize);
    if gridsize.contains(&0) {
        return refuse(format!("{field}.gridsize"), "rows and columns must be >= 1");
    }
    if let Some(layout) = &t.gridlayout {
        if layout.chars().count() != (gridsize[0] * gridsize[1]) as usize {
            return refuse(format!("{field}.gridlayout"), "one character per grid cell");
        }
    }
    let spec = TextureSpec {
        kind: t.kind.map_or(d.kind, |k| match k {
            TexKindDoc::TwoD => TexKind::TwoD,
            TexKindDoc::Cube => TexKind::Cube,
            TexKindDoc::Skybox => TexKind::Skybox,
        }),
        colorspace: t.colorspace.map_or(d.colorspace, |c| match c {
            ColorSpaceDoc::Auto => ColorSpace::Auto,
            ColorSpaceDoc::Srgb => ColorSpace::Srgb,
            ColorSpaceDoc::Linear => ColorSpace::Linear,
        }),
        builtin: t.builtin.map_or(d.builtin, |b| match b {
            BuiltinDoc::None => Builtin::None,
            BuiltinDoc::Gradient => Builtin::Gradient,
            BuiltinDoc::Checker => Builtin::Checker,
            BuiltinDoc::Flat => Builtin::Flat,
        }),
        rgb1: t.rgb1.unwrap_or(d.rgb1),
        rgb2: t.rgb2.unwrap_or(d.rgb2),
        mark: t.mark.map_or(d.mark, |m| match m {
            MarkDoc::None => Mark::None,
            MarkDoc::Edge => Mark::Edge,
            MarkDoc::Cross => Mark::Cross,
            MarkDoc::Random => Mark::Random,
        }),
        markrgb: t.markrgb.unwrap_or(d.markrgb),
        random: t.random.unwrap_or(d.random),
        width: t.width.unwrap_or(d.width),
        height: t.height.unwrap_or(d.height),
        file: t.file.clone(),
        gridsize,
        gridlayout: t.gridlayout.clone().unwrap_or(d.gridlayout),
        cubefiles: t.cubefiles.clone().map_or(d.cubefiles, |f| f.map(Some)),
    };
    let a = AssetRef::from_path(
        AssetKind::Texture,
        &t.name,
        t.file.as_deref().unwrap_or_default(),
    );
    s.textures.insert(a.id, Texture { spec, data: None });
    s.assets.push(a);
    Ok(())
}

fn material(s: &mut SceneDesc, m: &MaterialDoc) -> Result<(), EsSceneError> {
    let field = format!("material[{}]", m.name);
    let a = AssetRef::from_path(
        AssetKind::Material,
        &m.name,
        m.texture.as_deref().unwrap_or_default(),
    );
    let id = a.id;
    s.assets.push(a);
    let maps = [
        &m.texture,
        &m.orm,
        &m.metallic_map,
        &m.roughness_map,
        &m.normal_map,
        &m.emissive_map,
    ];
    let explicit = [m.specular, m.shininess, m.metallic, m.roughness, m.emission];
    if maps.iter().all(|t| t.is_none()) && explicit.iter().all(Option::is_none) {
        return Ok(()); // rgba only: drawn as the geom's own colour, as in MJCF
    }
    let slots = [
        "texture",
        "orm",
        "metallic_map",
        "roughness_map",
        "normal_map",
        "emissive_map",
    ];
    let mut ids = [None; 6];
    for ((slot, name), out) in slots.iter().zip(maps).zip(&mut ids) {
        if let Some(name) = name {
            *out = Some(asset(
                s,
                AssetKind::Texture,
                name,
                &format!("{field}.{slot}"),
            )?);
        }
    }
    let unset = |v: Option<f64>| v.filter(|x| *x >= 0.0);
    s.materials.insert(
        id,
        Material {
            rgba: m.rgba.unwrap_or([1.0; 4]),
            emission: m.emission.unwrap_or(0.0),
            specular: m.specular,
            shininess: m.shininess,
            metallic: unset(m.metallic),
            roughness: unset(m.roughness),
            texrepeat: m.texrepeat.unwrap_or([1.0, 1.0]),
            texuniform: m.texuniform.unwrap_or(false),
            rgb: ids[0],
            orm: ids[1],
            metallic_map: ids[2],
            roughness_map: ids[3],
            normal_map: ids[4],
            normal_scale: m.normal_scale,
            emissive_map: ids[5],
            emissive: ids[5].map(|_| [m.emission.unwrap_or(1.0); 3]),
        },
    );
    Ok(())
}
