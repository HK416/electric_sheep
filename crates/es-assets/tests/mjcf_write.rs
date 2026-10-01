//! Plan G, packet G2: the full-fidelity MJCF writer.
//!
//! * Every committed scene under `tests/fixtures/mjcf/**` that `load_scene` reads (parse, then
//!   `mesh::load` for its meshes and textures) comes back from `write_mjcf` -> `parse_mjcf` ->
//!   `mesh::load` field for field, with the same `scene_hash` and asset hashes; the table of
//!   what does not is pinned below with the reason.
//! * The same over generated scenes (a property test).
//! * What MJCF cannot say is an error naming it.
//! * `MuJoCo` loading the export steps exactly as `MuJoCo` loading the original, for SO-101 and
//!   the Shadow Hand (packet H1's method; needs `ES_PYTHON` with `mujoco`, prints `SKIP`
//!   without).
//!
//! ```text
//! ES_PYTHON=.venv/Scripts/python.exe cargo test -p es-assets --test mjcf_write -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use es_assets::mjcf::{write_mjcf, WriteError};
use es_assets::scene::{
    scene_id, Actuator, ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Body, BodyInertial,
    Camera, ContactPair, FrictionCone, Geom, Integrator, Jacobian, Joint, JointKind, Material,
    SceneDesc, Sensor, SensorKind, SensorTarget, Shape, Site, Solver, Tendon, TendonKind,
};
use es_assets::texture::{Builtin, TexKind, Texture, TextureSpec};
use es_math::units::DEG_TO_RAD;
use es_math::{Inertia, Pose, Quat, Vec3};
use proptest::prelude::*;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mjcf")
}

fn xmls(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            xmls(&p, out);
        } else if p.extension().is_some_and(|e| e == "xml") {
            out.push(p);
        }
    }
}

fn scratch(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("es-mjcf-write-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What `load_scene` does for an MJCF file.
fn load(xml: &str, dir: &Path) -> Result<SceneDesc, String> {
    let mut scene = es_assets::parse_mjcf(xml).map_err(|e| e.to_string())?.scene;
    es_assets::mesh::load(&mut scene, dir).map_err(|e| e.to_string())?;
    Ok(scene)
}

/// `None` when `a` and `b` are the same scene; else the first differing line of their dumps.
fn differs(a: &SceneDesc, b: &SceneDesc) -> Option<String> {
    let (da, db) = (format!("{a:#?}"), format!("{b:#?}"));
    if da == db {
        let hashes = |s: &SceneDesc| s.assets.iter().map(|x| x.hash).collect::<Vec<_>>();
        assert_eq!(a.scene_hash(), b.scene_hash(), "equal scenes, other hashes");
        assert_eq!(hashes(a), hashes(b));
        return None;
    }
    let (la, lb): (Vec<&str>, Vec<&str>) = (da.lines().collect(), db.lines().collect());
    let i = la
        .iter()
        .zip(&lb)
        .position(|(x, y)| x != y)
        .unwrap_or(la.len().min(lb.len()));
    let at = |l: &[&str]| l[i.saturating_sub(30)..(i + 1).min(l.len())].join("\n");
    Some(format!(
        "line {i}:\n--- original\n{}\n--- read back\n{}",
        at(&la),
        at(&lb)
    ))
}

/// Writes `scene` under a fresh directory and reads it back as `load_scene` would.
fn round_trip(scene: &SceneDesc, tag: &str) -> Result<(SceneDesc, PathBuf), String> {
    let out = scratch(tag);
    let xml = write_mjcf(scene, &out).map_err(|e| format!("not written: {e}"))?;
    let path = out.join("scene.xml");
    std::fs::write(&path, &xml).unwrap();
    let again = load(&xml, &out).map_err(|e| format!("export not read: {e}\n{xml}"))?;
    Ok((again, path))
}

/// The fixtures that do not round trip, and why. Everything else must.
const EXPECTED: [(&str, &str); 3] = [
    // Parser tests of malformed `<default>` trees: they are refused before any scene exists.
    ("default_cycle.xml", "not read"),
    ("default_duplicate.xml", "not read"),
    // A parser test of `<asset>` spellings whose files are not committed.
    ("assets.xml", "not read"),
];

#[test]
fn every_committed_mjcf_scene_round_trips() {
    let mut files = Vec::new();
    xmls(&fixtures(), &mut files);
    let mut failures = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(fixtures())
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let xml = std::fs::read_to_string(&path).unwrap();
        let outcome = match load(&xml, path.parent().unwrap()) {
            Err(e) => Err(format!("not read: {e}")),
            Ok(original) => match round_trip(&original, "fixture") {
                Err(e) => Err(e),
                Ok((again, _)) => match differs(&original, &again) {
                    Some(d) => Err(format!("differs at {d}")),
                    None => Ok(original.scene_hash()),
                },
            },
        };
        match outcome {
            Ok(hash) => println!("{rel:<45} round trips, scene_hash {}", hex(&hash)),
            Err(why) => {
                println!("{rel:<45} {}", why.lines().next().unwrap_or_default());
                failures.push((rel, why));
            }
        }
    }
    for (rel, why) in &failures {
        let expected = EXPECTED.iter().find(|(f, _)| f == rel);
        assert!(
            expected.is_some_and(|(_, w)| why.starts_with(w)),
            "{rel}: {why}"
        );
    }
    assert_eq!(failures.len(), EXPECTED.len(), "{failures:#?}");
}

fn hex(h: &[u8; 32]) -> String {
    h.iter().fold(String::new(), |s, b| s + &format!("{b:02x}"))
}

#[test]
fn what_mjcf_cannot_say_is_an_error_naming_it() {
    let base = generate(7);
    let expect = |scene: &SceneDesc, needle: &str| {
        let out = scratch("refuse");
        match write_mjcf(scene, &out) {
            Err(WriteError::Inexpressible(what)) => {
                assert!(what.contains(needle), "`{what}` does not name `{needle}`");
            }
            other => panic!("expected a refusal naming {needle}, got {other:?}"),
        }
    };

    let mut s = base.clone();
    let world = s.bodies[0].id;
    s.joints
        .push(joint(world, "world", "spin", JointKind::Hinge));
    expect(&s, "joint `spin` on the world body");

    let mut s = base.clone();
    s.assets.push(AssetRef::from_path(
        AssetKind::HeightField,
        "terrain",
        "t.png",
    ));
    expect(&s, "height field `terrain`");

    let mut s = base.clone();
    let m = AssetRef::from_path(AssetKind::Material, "bumpy", "");
    s.materials.insert(
        m.id,
        Material {
            normal_scale: Some(2.0),
            ..Material::default()
        },
    );
    s.assets.push(m);
    expect(&s, "material `bumpy`: a normal-map scale");

    let mut s = base.clone();
    let g = &mut s.bodies[0].geoms[0];
    g.visual_only = !g.visual_only;
    let name = g.name.clone();
    expect(&s, &format!("geom `{name}`"));

    let mut s = base.clone();
    let mesh = AssetRef::from_path(AssetKind::Mesh, "part", "part.stl");
    s.assets.push(mesh);
    expect(&s, "mesh `part` (not loaded");
}

// ---- generated scenes ----------------------------------------------------------------------

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn u(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn n(&mut self, n: u64) -> usize {
        (self.u() % n) as usize
    }
    fn coin(&mut self) -> bool {
        self.u() & 1 == 1
    }
    /// Uniform in `[lo, hi)`, all 53 bits: the decimal forms are long.
    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.u() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    }
    fn v(&mut self) -> Vec3 {
        Vec3::new(self.f(-1.0, 1.0), self.f(-1.0, 1.0), self.f(-1.0, 1.0))
    }
    fn q(&mut self) -> Quat {
        if self.n(4) == 0 {
            return Quat::IDENTITY;
        }
        Quat::from_xyzw(
            self.f(-1.0, 1.0),
            self.f(-1.0, 1.0),
            self.f(-1.0, 1.0),
            self.f(-1.0, 1.0),
        )
        .normalize()
    }
    fn pose(&mut self) -> Pose {
        Pose::new(self.v(), self.q())
    }
}

fn joint(body: StableIdAlias, path: &str, name: &str, kind: JointKind) -> Joint {
    Joint {
        id: scene_id("joint", &format!("{path}/{name}")),
        name: name.to_owned(),
        body,
        kind,
        axis: Vec3::new(0.0, 0.0, 1.0),
        anchor: Vec3::ZERO,
        range: None,
        damping: 0.0,
        armature: 0.0,
        stiffness: 0.0,
        friction_loss: 0.0,
        spring_ref: 0.0,
    }
}

type StableIdAlias = es_core::StableId;

/// A name: the parser's own for an unnamed element (`geom2`), or a written one.
fn pick_name(r: &mut Rng, tag: &str, counter: &mut u32, explicit: String) -> String {
    if r.coin() {
        *counter += 1;
        format!("{tag}{counter}")
    } else {
        explicit
    }
}

/// A scene in the parser's own form: world first, bodies depth first, unit quaternions and
/// axes, `fovy` from degrees, `visual_only` from the contact bits.
#[allow(clippy::too_many_lines)]
fn generate(seed: u64) -> SceneDesc {
    let r = &mut Rng(seed);
    let mut s = SceneDesc {
        name: format!("gen{seed}"),
        ..SceneDesc::default()
    };
    s.options.timestep = r.f(0.0005, 0.02);
    s.options.gravity = r.v();
    s.options.integrator = [
        Integrator::Euler,
        Integrator::Rk4,
        Integrator::Implicit,
        Integrator::ImplicitFast,
    ][r.n(4)];
    s.options.cone = [FrictionCone::Pyramidal, FrictionCone::Elliptic][r.n(2)];
    s.options.jacobian = [Jacobian::Dense, Jacobian::Sparse, Jacobian::Auto][r.n(3)];
    s.options.solver = [Solver::Pgs, Solver::Cg, Solver::Newton][r.n(3)];
    s.options.iterations = 1 + r.n(200) as u32;
    s.options.ls_iterations = 1 + r.n(60) as u32;
    s.options.eulerdamp = r.coin();
    s.options.impratio = r.f(0.5, 10.0);

    // Builtin textures (no file to load) and the materials that use them.
    let mut textures = Vec::new();
    for i in 0..r.n(3) {
        let a = AssetRef::from_path(AssetKind::Texture, &format!("tex{i}"), "");
        s.textures.insert(
            a.id,
            Texture {
                spec: TextureSpec {
                    kind: [TexKind::TwoD, TexKind::Cube][r.n(2)],
                    builtin: [Builtin::Checker, Builtin::Gradient, Builtin::Flat][r.n(3)],
                    rgb1: [r.f(0.0, 1.0), r.f(0.0, 1.0), r.f(0.0, 1.0)],
                    width: 1 + r.n(64) as u32,
                    height: 1 + r.n(64) as u32,
                    ..TextureSpec::default()
                },
                data: None,
            },
        );
        textures.push((a.id, a.name.clone()));
        s.assets.push(a);
    }
    let mut materials = Vec::new();
    for i in 0..r.n(4) {
        let drawn = r.coin();
        let attr = drawn && !textures.is_empty() && r.coin();
        let tex = (!textures.is_empty()).then(|| textures[r.n(textures.len() as u64)].clone());
        let path = if attr {
            tex.clone().unwrap().1
        } else {
            String::new()
        };
        let a = AssetRef::from_path(AssetKind::Material, &format!("mat{i}"), &path);
        if drawn {
            let mut m = Material {
                rgba: [r.f(0.0, 1.0), r.f(0.0, 1.0), r.f(0.0, 1.0), 1.0],
                emission: r.f(0.0, 2.0),
                specular: r.coin().then(|| r.f(0.0, 1.0)),
                shininess: r.coin().then(|| r.f(0.0, 1.0)),
                metallic: r.coin().then(|| r.f(0.0, 1.0)),
                roughness: r.coin().then(|| r.f(0.0, 1.0)),
                texrepeat: [r.f(0.5, 4.0), r.f(0.5, 4.0)],
                texuniform: r.coin(),
                ..Material::default()
            };
            if let Some((t, _)) = tex {
                m.rgb = Some(t);
                if !attr && r.coin() {
                    m.orm = Some(t);
                    m.emissive_map = Some(t);
                    m.emissive = Some([m.emission; 3]);
                }
            }
            s.materials.insert(a.id, m);
        }
        materials.push(a.id);
        s.assets.push(a);
    }

    // The tree, depth first as the parser produces it: a body is pushed when it is visited.
    let world = Body {
        id: scene_id("body", "world"),
        name: "world".to_owned(),
        parent: None,
        pose: Pose::IDENTITY,
        inertial: None,
        geoms: Vec::new(),
        sites: Vec::new(),
    };
    let mut pending = vec![(world, "world".to_owned(), 0u32)];
    let mut next_body = 0;
    let mut body_counters = std::collections::BTreeMap::<String, u32>::new();
    let mut referable_geoms = Vec::new();
    let mut referable_sites = Vec::new();
    let mut hinges = Vec::new();
    while let Some((body, path, depth)) = pending.pop() {
        let index = s.bodies.len();
        let world = index == 0;
        s.bodies.push(body);
        let owner = s.bodies[index].id;
        let (mut gc, mut sc, mut cc, mut jc) = (0, 0, 0, 0);
        if !world {
            for k in 0..r.n(3) {
                let kind = [
                    JointKind::Hinge,
                    JointKind::Slide,
                    JointKind::Ball,
                    JointKind::Free,
                ][r.n(4)];
                let name = pick_name(r, "joint", &mut jc, format!("j{index}_{k}"));
                let mut j = joint(owner, &path, &name, kind);
                j.axis = r.v().normalize();
                j.anchor = r.v();
                j.range = r.coin().then(|| (r.f(-3.0, 0.0), r.f(0.0, 3.0)));
                j.damping = r.f(0.0, 1.0);
                j.armature = if r.coin() { 0.0 } else { r.f(0.0, 0.1) };
                j.stiffness = r.f(0.0, 5.0);
                j.friction_loss = r.f(0.0, 1.0);
                j.spring_ref = r.f(-1.0, 1.0);
                if kind == JointKind::Hinge && !name.starts_with("joint") {
                    hinges.push(j.id);
                }
                s.joints.push(j);
            }
            if r.coin() {
                let frame = r.q();
                s.bodies[index].inertial = Some(BodyInertial {
                    mass: r.f(0.01, 5.0),
                    com: r.v(),
                    inertia: if frame == Quat::IDENTITY && r.coin() {
                        Inertia::new(
                            1.0,
                            2.0,
                            3.0,
                            r.f(-0.1, 0.1),
                            r.f(-0.1, 0.1),
                            r.f(-0.1, 0.1),
                        )
                    } else {
                        Inertia::diagonal(r.f(0.1, 1.0), r.f(0.1, 1.0), r.f(0.1, 1.0))
                    },
                    frame,
                });
            }
        }
        for k in 0..r.n(4) {
            let name = pick_name(r, "geom", &mut gc, format!("g{index}_{k}"));
            let shape = match r.n(6) {
                0 => Shape::Plane {
                    half_x: r.f(0.0, 5.0),
                    half_y: r.f(0.0, 5.0),
                    grid: r.f(0.0, 1.0),
                },
                1 => Shape::Sphere {
                    radius: r.f(0.01, 1.0),
                },
                2 => Shape::Capsule {
                    radius: r.f(0.01, 1.0),
                    half_length: r.f(0.01, 1.0),
                },
                3 => Shape::Cylinder {
                    radius: r.f(0.01, 1.0),
                    half_length: r.f(0.01, 1.0),
                },
                4 => Shape::Box {
                    half_extents: Vec3::new(r.f(0.01, 1.0), r.f(0.01, 1.0), r.f(0.01, 1.0)),
                },
                _ => Shape::Ellipsoid {
                    radii: Vec3::new(r.f(0.01, 1.0), r.f(0.01, 1.0), r.f(0.01, 1.0)),
                },
            };
            let (contype, conaffinity) = (r.n(3) as u32, r.n(3) as u32);
            let g = Geom {
                id: scene_id("geom", &format!("{path}/{name}")),
                name: name.clone(),
                shape,
                pose: r.pose(),
                friction: [r.f(0.0, 2.0), r.f(0.0, 0.1), r.f(0.0, 0.01)],
                contype,
                conaffinity,
                condim: [1, 3, 4, 6][r.n(4)],
                priority: [-2, -1, 0, 1, 2][r.n(5)],
                density: r.f(100.0, 3000.0),
                mass: r.coin().then(|| r.f(0.01, 2.0)),
                margin: if r.coin() { 0.0 } else { r.f(0.0, 0.01) },
                gap: if r.coin() { 0.0 } else { r.f(0.0, 0.01) },
                solref: [r.f(0.001, 0.1), r.f(0.5, 2.0)],
                solimp: [r.f(0.5, 0.9), r.f(0.9, 0.99), 0.001, 0.5, 2.0],
                material: (!materials.is_empty() && r.coin())
                    .then(|| materials[r.n(materials.len() as u64)]),
                rgba: [r.f(0.0, 1.0), r.f(0.0, 1.0), r.f(0.0, 1.0), r.f(0.0, 1.0)],
                visual_only: contype == 0 && conaffinity == 0,
            };
            if !name.starts_with("geom") {
                referable_geoms.push(g.id);
            }
            s.bodies[index].geoms.push(g);
        }
        for k in 0..r.n(3) {
            let name = pick_name(r, "site", &mut sc, format!("s{index}_{k}"));
            let site = Site {
                id: scene_id("site", &format!("{path}/{name}")),
                name: name.clone(),
                pose: r.pose(),
                size: Vec3::new(r.f(0.001, 0.1), r.f(0.001, 0.1), r.f(0.001, 0.1)),
            };
            if !name.starts_with("site") {
                referable_sites.push(site.id);
            }
            s.bodies[index].sites.push(site);
        }
        for k in 0..r.n(2) {
            let name = pick_name(r, "camera", &mut cc, format!("c{index}_{k}"));
            s.cameras.push(Camera {
                id: scene_id("camera", &format!("{path}/{name}")),
                name,
                body: (!world).then_some(owner),
                pose: r.pose(),
                fovy: r.f(10.0, 120.0) * DEG_TO_RAD,
            });
        }
        if r.coin() && !world {
            s.gravcomp.insert(owner, r.f(0.0, 1.0));
        }
        // Children, pushed in reverse so the stack pops them in order: depth first.
        if depth < 3 {
            let mut kids = Vec::new();
            for _ in 0..r.n(if world { 4 } else { 3 }) {
                next_body += 1;
                let counter = body_counters.entry(path.clone()).or_insert(0);
                let name = pick_name(r, "body", counter, format!("b{next_body}"));
                let child = Body {
                    id: scene_id("body", &format!("{path}/{name}")),
                    name: name.clone(),
                    parent: Some(owner),
                    pose: r.pose(),
                    inertial: None,
                    geoms: Vec::new(),
                    sites: Vec::new(),
                };
                kids.push((child, format!("{path}/{name}"), depth + 1));
            }
            // Reversed, so the stack pops them in order.
            pending.extend(kids.into_iter().rev());
        }
    }

    // The sections that name elements, on names that are unique.
    if hinges.len() >= 2 && r.coin() {
        s.tendons.push(Tendon {
            id: scene_id("tendon", "couple"),
            name: "couple".to_owned(),
            kind: TendonKind::Fixed {
                joints: vec![(hinges[0], r.f(-1.0, 1.0)), (hinges[1], 1.0)],
            },
            range: r.coin().then_some((-0.1, 0.1)),
            stiffness: r.f(0.0, 1.0),
            damping: r.f(0.0, 1.0),
        });
    }
    if referable_sites.len() >= 2 {
        s.tendons.push(Tendon {
            id: scene_id("tendon", "rope"),
            name: "rope".to_owned(),
            kind: TendonKind::Spatial {
                sites: referable_sites[..2].to_vec(),
            },
            range: None,
            stiffness: 0.0,
            damping: r.f(0.0, 1.0),
        });
    }
    for (i, h) in hinges.iter().enumerate() {
        let kind = match r.n(4) {
            0 => ActuatorKind::Motor,
            1 => ActuatorKind::Position {
                kp: r.f(1.0, 100.0),
                kv: r.f(0.0, 10.0),
            },
            2 => ActuatorKind::Velocity { kv: r.f(0.1, 10.0) },
            _ => ActuatorKind::General {
                gain: [r.f(1.0, 10.0), 0.0, 0.0],
                bias: [0.0, r.f(-10.0, 0.0), r.f(-1.0, 0.0)],
            },
        };
        let name = format!("act{i}");
        s.actuators.push(Actuator {
            id: scene_id("actuator", &name),
            name,
            kind,
            target: ActuatorTarget::Joint(*h),
            gear: [r.f(0.5, 2.0), 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrl_range: r.coin().then(|| (-1.0, r.f(0.0, 2.0))),
            force_range: r.coin().then(|| (-r.f(1.0, 9.0), 5.0)),
        });
        let name = format!("pos{i}");
        s.sensors.push(Sensor {
            id: scene_id("sensor", &name),
            name,
            kind: SensorKind::JointPos,
            target: SensorTarget::Joint(*h),
            noise: if r.coin() { 0.0 } else { r.f(0.0, 0.1) },
            cutoff: 0.0,
        });
    }
    if let Some(site) = referable_sites.first() {
        s.sensors.push(Sensor {
            id: scene_id("sensor", "where"),
            name: "where".to_owned(),
            kind: SensorKind::FramePos,
            target: SensorTarget::Site(*site),
            noise: 0.0,
            cutoff: r.f(0.0, 9.0),
        });
    }
    if referable_geoms.len() >= 2 {
        s.contact_pairs.push(ContactPair {
            geom1: referable_geoms[0],
            geom2: referable_geoms[1],
            condim: r.coin().then_some(1),
            friction: r.coin().then(|| [1.0, 1.0, 0.005, 0.0001, r.f(0.0, 0.1)]),
            solref: None,
            solimp: r.coin().then_some([0.8, 0.9, 0.001, 0.5, 2.0]),
            margin: r.coin().then(|| r.f(0.0, 0.01)),
            gap: None,
        });
    }
    let named_bodies: Vec<StableIdAlias> = s
        .bodies
        .iter()
        .filter(|b| b.name.starts_with('b') && !b.name.starts_with("body"))
        .map(|b| b.id)
        .collect();
    if named_bodies.len() >= 2 {
        s.contact_excludes.push((named_bodies[0], named_bodies[1]));
    }
    s.validate().expect("a generated scene is valid");
    s
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn generated_scenes_round_trip(seed in any::<u64>()) {
        let mut scene = generate(seed);
        es_assets::mesh::load(&mut scene, Path::new(".")).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let (again, _) = round_trip(&scene, "gen").map_err(TestCaseError::fail)?;
        if let Some(d) = differs(&scene, &again) {
            return Err(TestCaseError::fail(format!("seed {seed}: {d}")));
        }
    }
}

// ---- MuJoCo parity -------------------------------------------------------------------------

/// Steps two files under the same controls (H1's: 120 ticks at rest, 120 at 40% of each
/// servo's range), compares every tick's `qpos`, and lists the compiled model arrays that are
/// not bit for bit equal -- where any difference comes from.
const PARITY: &str = r#"
import json, sys
import mujoco
import numpy as np

def run(path, ticks, close):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = []
    for t in range(ticks):
        d.ctrl[:] = np.clip(0.0, lo, hi) if t < ticks // 2 else lo + close * (hi - lo)
        mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return m, np.array(out)

a, b, ticks, close = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
allowed = sys.argv[5].split(",")
ma, qa = run(a, ticks, close)
mb, qb = run(b, ticks, close)
fields = []
for name in dir(ma):
    if name.startswith("_") or name.startswith("name") or name in ("names", "paths", "text_data"):
        continue
    x, y = getattr(ma, name, None), getattr(mb, name, None)
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.shape != y.shape or (x.dtype.kind == "f" and not np.array_equal(x.view(np.uint8), y.view(np.uint8))) \
                or (x.dtype.kind != "f" and not np.array_equal(x, y)):
            fields.append(name)
same_shape = qa.shape == qb.shape
diff = np.abs(qa - qb) if same_shape else None
bad = np.nonzero(np.any(qa.view(np.uint64) != qb.view(np.uint64), axis=1))[0] if same_shape else []
print(json.dumps({
    "nq": int(ma.nq), "ticks": ticks,
    "identical": bool(same_shape and len(bad) == 0),
    "first_tick": int(bad[0]) if len(bad) else -1,
    "worst": float(diff.max()) if same_shape else -1.0,
    "fields": fields,
    "unexpected": [f for f in fields if f not in allowed],
}))
"#;

/// `MuJoCo` on the export against `MuJoCo` on the original file. Not bitwise: `SceneDesc` holds
/// the importer's orientations -- a `quat` normalised by division where `MuJoCo` multiplies by
/// the reciprocal, `fromto` / `xyaxes` through `es_math` where `MuJoCo` has its own
/// arithmetic -- and the export writes those, not the file's text. So the oracle is H1's bound
/// on `qpos` plus a pinned list of the compiled arrays allowed to differ: each an orientation,
/// derived from one, or a string address. Any other array -- a dropped damping, a lost
/// material -- fails it.
fn parity(rel: &str, allowed: &[&str]) {
    let python = match std::env::var("ES_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => {
            println!("SKIP parity {rel}: ES_PYTHON is not set");
            return;
        }
    };
    let original = fixtures().join(rel);
    let xml = std::fs::read_to_string(&original).unwrap();
    let scene = load(&xml, original.parent().unwrap()).unwrap();
    let (_, export) = round_trip(&scene, "parity").unwrap();
    let out = Command::new(&python)
        .args(["-c", PARITY])
        .arg(&original)
        .arg(&export)
        .args(["240", "0.4", &allowed.join(",")])
        .output()
        .unwrap_or_else(|e| panic!("`{python}`: {e}"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("No module named 'mujoco'") {
        println!("SKIP parity {rel}: no mujoco in `{python}`");
        return;
    }
    assert!(out.status.success(), "{stderr}");
    let json = String::from_utf8_lossy(&out.stdout);
    println!("parity {rel}: {}", json.trim());
    let worst: f64 = json
        .split_once("\"worst\": ")
        .and_then(|(_, rest)| rest.split([',', '}']).next())
        .and_then(|v| v.trim().parse().ok())
        .expect("a worst difference");
    assert!(
        json.contains("\"unexpected\": []") && (0.0..=1e-9).contains(&worst),
        "MuJoCo steps the export differently from the original: {json}\nexport at {}",
        export.display()
    );
}

#[test]
fn mujoco_steps_the_so101_export_as_the_original() {
    // Three bodies and a site written `quat="0.707107 ..."`, `fromto` capsules, and what
    // MuJoCo derives from their frames.
    parity(
        "so101_pick_place.xml",
        &[
            "body_quat",
            "geom_quat",
            "site_quat",
            "body_invweight0",
            "dof_invweight0",
            "dof_M0",
            "dof_length",
            "actuator_acc0",
            "cam_poscom0",
            "bvh_aabb",
        ],
    );
}

#[test]
fn mujoco_steps_the_shadow_hand_export_as_the_original() {
    // Four middle phalanges' `<inertial quat>`, the `xyaxes` cameras; the asset files moved.
    parity(
        "shadow_hand/shadow_hand_repose.xml",
        &[
            "body_iquat",
            "cam_quat",
            "cam_mat0",
            "mesh_pathadr",
            "tex_pathadr",
        ],
    );
}
