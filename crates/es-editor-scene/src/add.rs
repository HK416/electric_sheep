//! ①'s Add menu, decided (packet M17/G7, `docs/design/scene-authoring.md` section 5.3): what each
//! item makes and where it goes, and a picture made a geom's material.
//!
//! **Where a new thing goes.** Where the view's centre ray first meets what is drawn (the
//! renderer's nearest-hit rule, as a pick), resting on that surface: its lowest point — its
//! shape's half-extent along the surface's normal, a mesh's lowest vertex — sits at the hit.
//! With no hit, at the origin on the ground plane. A camera goes at the view's eye looking where
//! it looks, a light a metre above the point, a robot at the point with its library pose. With
//! snapping on (G6's toggle) the point on a level surface moves to the nearest whole centimetre
//! across it, so the numbers the document gets are the person's, not the pointer's.
//!
//! **One Add is one command**, so one undo step, and it ends up selected. Files are copied into
//! the project first ([`crate::import`]): the check every command passes expands the scene from
//! the files on disk. A refused Add takes back the files it wrote. An included file whose names
//! meet the scene's gets its handle as a name prefix, as a duplicated include does (G5).

// Geometry's own letters: a point's `x y z`, a shape's half-extents `a b c`, a normal `n`.
#![allow(clippy::many_single_char_names)]

use std::path::{Path, PathBuf};

use es_assets::esscene::{
    CameraDoc, GeomDoc, Include, MaterialDoc, RegionDoc, ShapeDoc, TextureDoc,
};
use es_math::units::RAD_TO_DEG;
use es_math::{Quat, Vec3};
use es_render::cpu::nearest_hit_flat;
use es_render::TriScene;
use serde::Deserialize;

use crate::check::{check, Refusal, DUPLICATE, MISSING, OTHER};
use crate::command::{
    new_body, new_camera, new_geom, new_light, new_region, unique, Command, Entity, Record,
};
use crate::gizmo::snap_cm;
use crate::import;
use crate::inspect::{self, ShapeKind};
use crate::model::SceneModel;
use crate::view::{ray, Camera, Ray};

/// How far above the point a new light hangs, metres.
const LIGHT_HEIGHT: f64 = 1.0;
/// The library's list, relative to the repository root.
pub const LIBRARY: &str = "templates/robots.toml";

/// What the Add menu makes.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// A body with a free joint and one geom of this shape: it falls when let go.
    Object(ShapeKind),
    /// Static scenery: one geom on the world.
    Fixed(ShapeKind),
    /// A free body with one mesh geom, from an STL or OBJ file, its vertices times this scale:
    /// the unit the file was drawn in ([`import::UNITS`]).
    Mesh(PathBuf, f64),
    /// An `[[include]]`: a robot of the library or from a file.
    Robot(Robot),
    Camera,
    Light,
    Region,
}

/// A robot file and how it is placed: a library entry, or a person's file ([`Robot::file`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Robot {
    /// The include's name, made unique.
    pub id: String,
    /// The key of its name in the editor's string tables; absent for a person's file.
    pub name: Option<String>,
    /// MJCF, URDF or glTF; in the library, relative to the repository root.
    pub source: PathBuf,
    /// Where it stands from the point it is added at, metres; absent is at the point.
    pub pos: Option<[f64; 3]>,
    /// Its orientation, `[x, y, z, w]`; absent is the file's own.
    pub quat: Option<[f64; 4]>,
    /// Prepended to every name the file declares; absent keeps the file's names.
    pub prefix: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Library {
    kind: String,
    #[serde(rename = "robot")]
    robots: Vec<Robot>,
}

impl Robot {
    /// A person's robot file: named by its stem, at the point, as the file has it.
    pub fn file(path: &Path) -> Self {
        Self {
            id: stem(path),
            name: None,
            source: path.to_path_buf(),
            pos: None,
            quat: None,
            prefix: None,
        }
    }
}

/// The robot library (`templates/robots.toml` under `repo`), its sources made absolute.
pub fn library(repo: &Path) -> Result<Vec<Robot>, String> {
    let path = repo.join(LIBRARY);
    let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let text = std::fs::read_to_string(&path).map_err(|e| fail(&e))?;
    let lib: Library = toml::from_str(&text).map_err(|e| fail(&e))?;
    if lib.kind != "robots" {
        return Err(fail(&"kind must be \"robots\""));
    }
    Ok((lib.robots.into_iter())
        .map(|r| Robot {
            source: repo.join(&r.source),
            ..r
        })
        .collect())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map_or_else(|| "thing".to_owned(), |s| s.to_string_lossy().into_owned())
}

fn arr(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// How far the unturned `shape`'s lowest point along `n` (a unit) is below its origin.
fn extent(shape: &ShapeDoc, n: Vec3) -> f64 {
    let (x, y, z) = (n.x.abs(), n.y.abs(), n.z.abs());
    match shape {
        ShapeDoc::Box([a, b, c]) => a * x + b * y + c * z,
        ShapeDoc::Sphere(r) => *r,
        ShapeDoc::Capsule([r, h]) => r + h * z,
        ShapeDoc::Cylinder([r, h]) => r * (x * x + y * y).sqrt() + h * z,
        ShapeDoc::Ellipsoid([a, b, c]) => {
            ((a * x).powi(2) + (b * y).powi(2) + (c * z).powi(2)).sqrt()
        }
        ShapeDoc::Plane(_) | ShapeDoc::Mesh { .. } => 0.0,
    }
}

/// A new shape of `kind`: 5 cm across, as G5's Add made it.
fn shape(kind: ShapeKind) -> ShapeDoc {
    inspect::reshape(&ShapeDoc::Box([0.025; 3]), kind)
}

/// The MJCF camera that sits at `view`'s eye looking where it looks (MJCF's camera frame looks
/// along -Z with +Y up: the renderer's `OpenCV` frame turned half about X). The quaternion is the
/// renderer's own arithmetic, no trigonometry; `w >= 0`.
fn camera(view: &Camera, name: &str, snap: bool) -> CameraDoc {
    let round = |v: [f64; 3]| if snap { v.map(snap_cm) } else { v };
    let view = Camera {
        eye: round(view.eye),
        look_at: round(view.look_at),
        ..*view
    };
    let q = (view.view()).map_or(Quat::IDENTITY, |v| {
        v.pose.orientation * Quat::from_xyzw(1.0, 0.0, 0.0, 0.0)
    });
    let q = if q.w < 0.0 {
        Quat::from_xyzw(-q.x, -q.y, -q.z, -q.w)
    } else {
        q
    }
    .normalize();
    CameraDoc {
        quat: Some([q.x, q.y, q.z, q.w]),
        pos: Some(view.eye),
        fovy: Some((view.fov_y * RAD_TO_DEG * 1e6).round() / 1e6),
        ..new_camera(name)
    }
}

impl SceneModel {
    /// Where `ray` first meets what is drawn, and the surface's unit normal there, facing back
    /// along the ray: [`Self::hit`]'s triangle, met again in `f64` so a level surface is level.
    pub fn surface(&self, ray: &Ray) -> Option<(Vec3, Vec3)> {
        let tris = TriScene::from_scene(&self.drawn()).ok()?;
        let f = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32];
        let hit = nearest_hit_flat(&tris.tris, f(ray.origin), f(ray.dir), 0.0, f32::INFINITY)?;
        let v = tris.tris[hit.tri as usize].v;
        let [a, b, c] = v.map(|p| Vec3::new(p[0].into(), p[1].into(), p[2].into()));
        let n = (b - a).cross(c - a).normalize();
        let n = if n.dot(ray.dir) > 0.0 { -n } else { n };
        let t = n.dot(a - ray.origin) / n.dot(ray.dir);
        let p = ray.origin + ray.dir.scale(t);
        // On a level surface the height is the surface's own, not the ray's rounding.
        let level = n.x == 0.0 && n.y == 0.0;
        Some((if level { Vec3::new(p.x, p.y, a.z) } else { p }, n))
    }

    /// Adds `item` where `view`'s centre ray meets the scene, as one command; selected after it.
    pub fn add(&mut self, item: &Item, view: &Camera, snap: bool) -> Result<(), Refusal> {
        let centre = ray(
            view,
            [f64::from(view.width) / 2.0, f64::from(view.height) / 2.0],
        );
        let up = Vec3::new(0.0, 0.0, 1.0);
        let (at, n) = centre
            .and_then(|r| self.surface(&r))
            .unwrap_or((Vec3::ZERO, up));
        let level = n.x == 0.0 && n.y == 0.0;
        let at = if snap && level {
            Vec3::new(snap_cm(at.x), snap_cm(at.y), at.z)
        } else {
            at
        };
        let rest = |extent: f64| Some(arr(at + n.scale(extent)));
        let mut written = None;
        let record = match item {
            Item::Object(kind) => {
                let mut b = new_body(&self.unique(kind.base()), shape(*kind));
                b.pos = rest(extent(&shape(*kind), n));
                Record::Body(b)
            }
            Item::Fixed(kind) => {
                let world = &self.scene().bodies[0].geoms;
                let name = unique(kind.base(), &|x| world.iter().any(|g| g.name == x));
                Record::Scenery(GeomDoc {
                    name: Some(name),
                    pos: rest(extent(&shape(*kind), n)),
                    ..new_geom(shape(*kind))
                })
            }
            Item::Mesh(src, unit) => {
                let field = "body.geom.shape";
                let (file, w) = import::file(self.root(), src).map_err(|why| other(field, why))?;
                let mesh = ShapeDoc::Mesh { file, scale: None };
                let mesh = inspect::with_scale(&mesh, [*unit; 3]);
                let mut b = new_body(&self.unique(&stem(src)), mesh);
                match self.lowest(&b, n) {
                    Ok(low) => b.pos = rest(low),
                    Err(r) => {
                        w.undo();
                        return Err(r);
                    }
                }
                written = Some(w);
                Record::Body(b)
            }
            Item::Robot(r) => {
                let (source, w) = import::robot(self.root(), &r.source)
                    .map_err(|why| other("include.source", why))?;
                written = Some(w);
                let [x, y, z] = r.pos.unwrap_or_default();
                Record::Include(Include {
                    name: self.unique(&r.id),
                    source,
                    prefix: r.prefix.clone(),
                    pos: Some(arr(at + Vec3::new(x, y, z))),
                    quat: r.quat,
                    set: None,
                })
            }
            Item::Camera => Record::Camera(camera(view, &self.unique("camera"), snap)),
            Item::Light => {
                let mut l = new_light(&self.unique("light"));
                l.pos = Some(arr(at + up.scale(LIGHT_HEIGHT)));
                Record::Light(l)
            }
            Item::Region => {
                let r = new_region(&self.unique("region"));
                let size = r.size.unwrap_or(inspect::REGION);
                Record::Region(RegionDoc {
                    pos: rest(extent(&ShapeDoc::Box(size), n)),
                    ..r
                })
            }
        };
        let mut done = self.apply(&Command::Add(record.clone()));
        if let (Err(r), Record::Include(inc)) = (&done, &record) {
            if r.key == DUPLICATE && inc.prefix.is_none() {
                let prefix = Some(format!("{}:", inc.name));
                let again = Include {
                    prefix,
                    ..inc.clone()
                };
                done = self.apply(&Command::Add(Record::Include(again)));
            }
        }
        if let (Err(_), Some(w)) = (&done, written) {
            w.undo();
        }
        done
    }

    /// How far below its origin the body `b` (one mesh geom, unturned) reaches along `n`: its
    /// lowest vertex. Reading the mesh is the check a mesh file passes.
    fn lowest(&self, b: &es_assets::esscene::BodyDoc, n: Vec3) -> Result<f64, Refusal> {
        let mut doc = import::empty();
        doc.bodies.push(b.clone());
        let s = check(&doc, self.root(), &[])?;
        let points: Vec<[f32; 3]> = (s.meshes.keys())
            .flat_map(|id| s.mesh_positions(*id).unwrap_or_default().into_owned())
            .collect();
        let low = (points.iter())
            .map(|p| n.dot(Vec3::new(p[0].into(), p[1].into(), p[2].into())))
            .fold(f64::INFINITY, f64::min);
        Ok(if low.is_finite() { -low } else { 0.0 })
    }

    /// The picture `image` as the material of geom `index` of `e` (a body's, or `e` itself for a
    /// geom or a piece of scenery): one command that adds a `[[texture]]` and a `[[material]]`
    /// named by the file — or reuses those the same picture got before — and sets the geom's
    /// `material`.
    pub fn texture(&mut self, e: &Entity, index: usize, image: &Path) -> Result<(), Refusal> {
        let field = format!("{}.material", e.field(self.doc()));
        let (file, w) = import::file(self.root(), image).map_err(|why| other(&field, why))?;
        let mut doc = self.doc().clone();
        let before = (doc.textures.iter())
            .find(|t| t.file.as_deref() == Some(&file))
            .and_then(|t| (doc.materials.iter()).find(|m| m.texture.as_deref() == Some(&t.name)));
        let material = if let Some(m) = before {
            m.name.clone()
        } else {
            let taken = |x: &str| {
                doc.textures.iter().any(|t| t.name == x)
                    || doc.materials.iter().any(|m| m.name == x)
            };
            let name = unique(&stem(image), &taken);
            let json = serde_json::json!({ "name": name, "file": file });
            let t: TextureDoc = serde_json::from_value(json).expect("a texture");
            let json = serde_json::json!({ "name": name, "texture": name });
            let m: MaterialDoc = serde_json::from_value(json).expect("a material");
            doc.textures.push(t);
            doc.materials.push(m);
            name
        };
        let geom = match e {
            Entity::Body(b) => {
                (doc.bodies.iter_mut().find(|x| x.name == *b)).and_then(|b| b.geoms.get_mut(index))
            }
            Entity::Geom { body, index } => (doc.bodies.iter_mut().find(|x| x.name == *body))
                .and_then(|b| b.geoms.get_mut(*index)),
            Entity::Scenery(i) => doc.geoms.get_mut(*i),
            _ => None,
        };
        let Some(geom) = geom else {
            w.undo();
            return Err(Refusal::new(field, MISSING, vec![e.label(self.doc())]));
        };
        geom.material = Some(material);
        let done = self.apply(&Command::Scene(Box::new(doc), Some(e.clone())));
        if done.is_err() {
            w.undo();
        }
        done
    }
}

fn other(field: &str, why: String) -> Refusal {
    Refusal::new(field, OTHER, vec![why])
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// The committed library reads, its files are there, and it names its robots by keys of the
    /// editor's string tables (whose completeness test reads this file for them).
    #[test]
    fn the_library_reads_and_names_its_robots_by_keys() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let lib = super::library(&repo).expect("templates/robots.toml");
        let names: Vec<Option<&str>> = lib.iter().map(|r| r.name.as_deref()).collect();
        let keys = ["author.add.robot.so101", "author.add.robot.shadow_hand"];
        assert_eq!(names, keys.map(Some));
        assert!(lib.iter().all(|r| r.source.is_file()), "{lib:?}");
    }
}
