//! The check every command passes before it is applied: the values a scene cannot hold (a
//! negative mass, a zero size), then G1's [`expand`] (names, parents, materials, files), then
//! the semantic mapping report (spec 17.2) of each backend the project runs on. A command whose
//! result fails is **not applied**: the document, the undo stack and the viewport stay at the
//! last scene that expands, and the refusal is shown on the field the person was changing.

use std::path::Path;

use es_assets::esscene::{expand, EsScene, EsSceneError, GeomDoc, ShapeDoc};
use es_assets::scene::{SceneDesc, SceneError};
use es_physics_backend::{mapping_report, BackendKind};

/// Why a command was not applied. `field` is where, as G1 names a document's fields
/// (`body[cube].geom[cube_geom].mass`); `key` is a key of the editor's string tables
/// (`es-editor-model`'s `i18n/*.toml`), whose `{}`s `args` fill in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub field: String,
    pub key: &'static str,
    pub args: Vec<String>,
}

pub const NEGATIVE: &str = "author.refused.negative";
pub const POSITIVE: &str = "author.refused.positive";
pub const UNIT: &str = "author.refused.unit";
pub const ANGLE: &str = "author.refused.angle";
pub const ORDER: &str = "author.refused.order";
pub const EMPTY: &str = "author.refused.empty";
pub const DUPLICATE: &str = "author.refused.duplicate";
pub const PARENT: &str = "author.refused.parent";
pub const MISSING: &str = "author.refused.missing";
pub const CYCLE: &str = "author.refused.cycle";
pub const BACKEND: &str = "author.refused.backend";
/// Anything else, in the reader's own words: its one argument.
pub const OTHER: &str = "author.refused.other";

impl Refusal {
    pub fn new(field: impl Into<String>, key: &'static str, args: Vec<String>) -> Self {
        Self {
            field: field.into(),
            key,
            args,
        }
    }

    /// The field's own name, the last segment of [`Self::field`] (`mass`).
    pub fn leaf(&self) -> &str {
        self.field.rsplit('.').next().unwrap_or(&self.field)
    }
}

/// The scene `doc` expands to against `dir`, or the first reason it cannot be one.
pub fn check(doc: &EsScene, dir: &Path, backends: &[BackendKind]) -> Result<SceneDesc, Refusal> {
    values(doc)?;
    let scene = expand(doc, dir).map_err(from_expand)?;
    for &backend in backends {
        let report = mapping_report(&scene, backend);
        if report.blocked {
            let names: Vec<String> = report.blocking().map(|r| r.feature.to_string()).collect();
            let args = vec![backend.to_string(), names.join(", ")];
            return Err(Refusal::new("scene", BACKEND, args));
        }
    }
    Ok(scene)
}

/// Whether the saved scene of the project at `root` maps onto `backend` (packet M17/G9: ②'s
/// teacher trains on `MuJoCo` Warp, which a project is not checked against while it is built),
/// or why not.
pub fn maps_onto(root: &Path, backend: BackendKind) -> Result<(), Refusal> {
    let path = root.join(crate::SCENE_FILE);
    let other = |why: String| Refusal::new("scene", OTHER, vec![why]);
    let text = std::fs::read_to_string(&path).map_err(|e| other(format!("{e}")))?;
    let doc = EsScene::from_toml(&text).map_err(|e| other(e.to_string()))?;
    check(&doc, root, &[backend]).map(|_| ())
}

/// The name a geom is refused by: its own, or `geom<n>` as G1 names the n-th unnamed one.
pub(crate) fn geom_label(g: &GeomDoc, unnamed: &mut u32) -> String {
    g.name.clone().unwrap_or_else(|| {
        *unnamed += 1;
        format!("geom{unnamed}")
    })
}

/// The values `expand` takes as they are but a scene cannot hold. `NaN` fails every bound.
fn values(doc: &EsScene) -> Result<(), Refusal> {
    let mut unnamed = 0;
    for g in &doc.geoms {
        geom(g, &format!("geom[{}]", geom_label(g, &mut unnamed)))?;
    }
    for b in &doc.bodies {
        let at = format!("body[{}]", b.name);
        name(&b.name, &at)?;
        if let Some(i) = &b.inertial {
            at_least(i.mass, 0.0, &format!("{at}.inertial.mass"), NEGATIVE)?;
        }
        if let Some([lo, hi]) = b.joint.as_ref().and_then(|j| j.range) {
            if lo > hi || lo.is_nan() || hi.is_nan() {
                return Err(Refusal::new(format!("{at}.joint.range"), ORDER, vec![]));
            }
        }
        let mut unnamed = 0;
        for g in &b.geoms {
            geom(g, &format!("{at}.geom[{}]", geom_label(g, &mut unnamed)))?;
        }
    }
    for c in &doc.cameras {
        let at = format!("camera[{}]", c.name);
        name(&c.name, &at)?;
        if let Some(f) = c.fovy {
            if f <= 0.0 || f >= 180.0 || f.is_nan() {
                return Err(Refusal::new(format!("{at}.fovy"), ANGLE, vec![]));
            }
        }
    }
    for l in &doc.lights {
        let at = format!("light[{}]", l.name);
        name(&l.name, &at)?;
        for v in l.size {
            above(v, &format!("{at}.size"))?;
        }
        for v in l.rgb.unwrap_or_default() {
            at_least(v, 0.0, &format!("{at}.rgb"), NEGATIVE)?;
        }
        at_least(
            l.intensity.unwrap_or(1.0),
            0.0,
            &format!("{at}.intensity"),
            NEGATIVE,
        )?;
    }
    for r in &doc.regions {
        let at = format!("region[{}]", r.name);
        name(&r.name, &at)?;
        for v in r.size.unwrap_or([1.0; 3]) {
            above(v, &format!("{at}.size"))?;
        }
    }
    for i in &doc.includes {
        name(&i.name, &format!("include[{}]", i.name))?;
    }
    if let Some(t) = doc.physics.as_ref().and_then(|p| p.timestep) {
        above(t, "physics.timestep")?;
    }
    Ok(())
}

fn name(name: &str, at: &str) -> Result<(), Refusal> {
    if name.trim().is_empty() {
        return Err(Refusal::new(format!("{at}.name"), EMPTY, vec![]));
    }
    Ok(())
}

fn at_least(v: f64, lo: f64, field: &str, key: &'static str) -> Result<(), Refusal> {
    if v >= lo {
        Ok(())
    } else {
        Err(Refusal::new(field, key, vec![]))
    }
}

fn above(v: f64, field: &str) -> Result<(), Refusal> {
    if v > 0.0 {
        Ok(())
    } else {
        Err(Refusal::new(field, POSITIVE, vec![]))
    }
}

fn geom(g: &GeomDoc, at: &str) -> Result<(), Refusal> {
    let shape = format!("{at}.shape");
    let sizes: &[f64] = match &g.shape {
        // Half x and half y may be 0 (an infinite plane); the grid spacing too.
        ShapeDoc::Plane(p) => {
            for v in p {
                at_least(*v, 0.0, &shape, NEGATIVE)?;
            }
            &[]
        }
        ShapeDoc::Sphere(r) => std::slice::from_ref(r),
        ShapeDoc::Capsule(v) | ShapeDoc::Cylinder(v) => v,
        ShapeDoc::Box(v) | ShapeDoc::Ellipsoid(v) => v,
        ShapeDoc::Mesh { scale, .. } => scale.as_ref().map_or(&[][..], |s| &s[..]),
    };
    for v in sizes {
        above(*v, &shape)?;
    }
    if let Some(m) = g.mass {
        at_least(m, 0.0, &format!("{at}.mass"), NEGATIVE)?;
    }
    if let Some(d) = g.density {
        at_least(d, 0.0, &format!("{at}.density"), NEGATIVE)?;
    }
    for v in g.friction.unwrap_or_default() {
        at_least(v, 0.0, &format!("{at}.friction"), NEGATIVE)?;
    }
    for v in g.rgba.unwrap_or_default() {
        if !(0.0..=1.0).contains(&v) {
            return Err(Refusal::new(format!("{at}.rgba"), UNIT, vec![]));
        }
    }
    Ok(())
}

/// The name between the first pair of backticks of `reason`, which is how G1 quotes one.
fn quoted(reason: &str) -> Option<String> {
    let (_, rest) = reason.split_once('`')?;
    Some(rest.split_once('`')?.0.to_owned())
}

/// G1's refusal in the editor's words: by the field it names.
fn from_expand(e: EsSceneError) -> Refusal {
    match e {
        EsSceneError::Field { field, reason } => {
            let name = quoted(&reason);
            let key = match field.rsplit('.').next() {
                Some("name") => DUPLICATE,
                Some("parent") => PARENT,
                Some(
                    "material" | "texture" | "orm" | "metallic_map" | "roughness_map"
                    | "normal_map" | "emissive_map",
                ) => MISSING,
                _ => OTHER,
            };
            match (key, name) {
                (OTHER, _) | (_, None) => Refusal::new(field, OTHER, vec![reason]),
                (key, Some(name)) => Refusal::new(field, key, vec![name]),
            }
        }
        // Two elements whose name paths meet (a body named as an included one).
        EsSceneError::Scene(SceneError::DuplicateId { name, .. }) => {
            Refusal::new(format!("{name}.name"), DUPLICATE, vec![name])
        }
        other => Refusal::new("scene", OTHER, vec![other.to_string()]),
    }
}
