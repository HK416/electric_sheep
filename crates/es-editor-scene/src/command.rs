//! What a person does to a scene, as data: an [`Entity`] names a thing of the document, a
//! [`Record`] is its fields as the document writes them, and a [`Command`] is one edit — one
//! undo step. [`apply`] changes the documents and nothing else; whether the result is a scene is
//! [`crate::check`]'s to say.
//!
//! Identity is the name (design section 3.1): a rename is a refactor that rewrites every place
//! the scene document and the task specification name the thing, and a geom, which nothing names,
//! is its place in its list.

use std::collections::BTreeSet;

use es_assets::esscene::{
    BodyDoc, CameraDoc, EsScene, GeomDoc, Include, JointDoc, JointKindDoc, LightDoc, Physics,
    RegionDoc, ShapeDoc,
};
use es_script::spec::TaskSpec;

use crate::check::{geom_label, Refusal, CYCLE, EMPTY, MISSING, OTHER};

/// One thing of the scene document.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Entity {
    /// `[physics]`.
    Physics,
    /// An `[[include]]` by its handle.
    Include(String),
    /// The n-th `[[geom]]`: static scenery on the world body (a table, a floor, a bin wall).
    Scenery(usize),
    Body(String),
    /// The n-th `[[body.geom]]` of a body.
    Geom {
        body: String,
        index: usize,
    },
    Camera(String),
    Light(String),
    Region(String),
}

/// An entity's fields, as the document writes them.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Physics(Physics),
    Include(Include),
    Scenery(GeomDoc),
    /// With its geoms.
    Body(BodyDoc),
    Geom(GeomDoc),
    Camera(CameraDoc),
    Light(LightDoc),
    Region(RegionDoc),
}

/// One edit, one undo step.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Appends a body (with its geoms), static scenery, a camera, a light, a region, or an
    /// include (a robot or a mesh file, G7's). Not a geom of a body: set the body for that.
    Add(Record),
    /// Removes the entity; a body goes with the bodies, cameras and regions hanging from it.
    Delete(Entity),
    /// A copy beside the original, under a free name.
    Duplicate(Entity),
    /// Replaces the entity's fields. Its name is not one of them (a rename is [`Self::Rename`]).
    Set(Entity, Record),
    /// The entity's place, relative to what it hangs from: metres and an `[x, y, z, w]`
    /// quaternion, the bits the document stores.
    SetPose {
        entity: Entity,
        pos: [f64; 3],
        quat: [f64; 4],
    },
    /// Hangs a body, a camera or a region from another body (`None`: the world), keeping its
    /// pose relative to what it hangs from.
    Reparent(Entity, Option<String>),
    /// Renames the entity and every reference to it, in the scene and in `task.estask`.
    Rename(Entity, String),
    /// The whole scene document after an edit of several parts (G7's picture import: a texture,
    /// a material and the geom naming it), and what to select after it.
    Scene(Box<EsScene>, Option<Entity>),
}

/// The scene document and the task specification: what a command changes and an undo step
/// restores.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Docs {
    pub scene: EsScene,
    pub spec: Option<TaskSpec>,
}

/// A body with one primitive geom and a free joint: an object that falls when it is let go.
pub fn new_body(name: &str, shape: ShapeDoc) -> BodyDoc {
    BodyDoc {
        name: name.to_owned(),
        parent: None,
        pos: None,
        quat: None,
        gravcomp: None,
        joint: Some(new_joint(JointKindDoc::Free)),
        inertial: None,
        geoms: vec![new_geom(shape)],
    }
}

/// A joint of `kind` named as its body, every other field left to `MuJoCo`'s default.
pub fn new_joint(kind: JointKindDoc) -> JointDoc {
    JointDoc {
        kind,
        name: None,
        axis: None,
        pos: None,
        range: None,
        damping: None,
        armature: None,
        stiffness: None,
        frictionloss: None,
        springref: None,
    }
}

/// A geom of `shape` with every other field left to `MuJoCo`'s default.
pub fn new_geom(shape: ShapeDoc) -> GeomDoc {
    GeomDoc {
        name: None,
        shape,
        pos: None,
        quat: None,
        mass: None,
        density: None,
        friction: None,
        condim: None,
        contype: None,
        conaffinity: None,
        priority: None,
        margin: None,
        gap: None,
        solref: None,
        solimp: None,
        rgba: None,
        material: None,
    }
}

/// A camera fixed to the world, looking straight down (the identity), 45 degrees.
pub fn new_camera(name: &str) -> CameraDoc {
    CameraDoc {
        name: name.to_owned(),
        parent: None,
        pos: Some([0.0, 0.0, 1.0]),
        quat: None,
        fovy: Some(45.0),
    }
}

/// A 40 cm square ceiling light.
pub fn new_light(name: &str) -> LightDoc {
    LightDoc {
        name: name.to_owned(),
        kind: None,
        pos: Some([0.0, 0.0, 1.5]),
        quat: None,
        size: [0.2, 0.2],
        rgb: None,
        intensity: None,
    }
}

/// A 10 cm box region on the world.
pub fn new_region(name: &str) -> RegionDoc {
    RegionDoc {
        name: name.to_owned(),
        parent: None,
        pos: None,
        quat: None,
        size: Some([0.05; 3]),
    }
}

/// A record's `pos` and `quat`.
pub type PoseSlots<'a> = (&'a mut Option<[f64; 3]>, &'a mut Option<[f64; 4]>);

impl Record {
    /// The place fields, when the entity has a place.
    pub fn pose_mut(&mut self) -> Option<PoseSlots<'_>> {
        match self {
            Self::Physics(_) => None,
            Self::Include(x) => Some((&mut x.pos, &mut x.quat)),
            Self::Scenery(x) | Self::Geom(x) => Some((&mut x.pos, &mut x.quat)),
            Self::Body(x) => Some((&mut x.pos, &mut x.quat)),
            Self::Camera(x) => Some((&mut x.pos, &mut x.quat)),
            Self::Light(x) => Some((&mut x.pos, &mut x.quat)),
            Self::Region(x) => Some((&mut x.pos, &mut x.quat)),
        }
    }
}

impl Entity {
    /// Where G1 names this entity's fields: `body[cube]`, `body[cube].geom[cube_geom]`.
    pub fn field(&self, doc: &EsScene) -> String {
        match self {
            Self::Physics => "physics".to_owned(),
            Self::Include(n) => format!("include[{n}]"),
            Self::Scenery(i) => format!("geom[{}]", label(&doc.geoms, *i)),
            Self::Body(n) => format!("body[{n}]"),
            Self::Geom { body, index } => {
                let geoms = doc.bodies.iter().find(|b| b.name == *body);
                let label = geoms.map_or_else(String::new, |b| label(&b.geoms, *index));
                format!("body[{body}].geom[{label}]")
            }
            Self::Camera(n) => format!("camera[{n}]"),
            Self::Light(n) => format!("light[{n}]"),
            Self::Region(n) => format!("region[{n}]"),
        }
    }

    /// The name the hierarchy shows.
    pub fn label(&self, doc: &EsScene) -> String {
        match self {
            Self::Physics => "physics".to_owned(),
            Self::Scenery(i) => label(&doc.geoms, *i),
            Self::Geom { body, index } => (doc.bodies.iter().find(|b| b.name == *body))
                .map_or_else(String::new, |b| label(&b.geoms, *index)),
            Self::Include(n)
            | Self::Body(n)
            | Self::Camera(n)
            | Self::Light(n)
            | Self::Region(n) => n.clone(),
        }
    }
}

/// The n-th geom's name as G1 names it (`geom<k>` for the k-th unnamed one).
fn label(geoms: &[GeomDoc], index: usize) -> String {
    let mut unnamed = 0;
    let mut out = String::new();
    for g in geoms.iter().take(index + 1) {
        out = geom_label(g, &mut unnamed);
    }
    out
}

fn named<'a, T>(list: &'a [T], name: &str, of: impl Fn(&T) -> &str) -> Option<&'a T> {
    list.iter().find(|x| of(x) == name)
}

fn named_mut<'a, T>(list: &'a mut [T], name: &str, of: impl Fn(&T) -> &str) -> Option<&'a mut T> {
    list.iter_mut().find(|x| of(x) == name)
}

/// The entity's fields; `None` when the document has no such entity.
pub fn get(doc: &EsScene, e: &Entity) -> Option<Record> {
    Some(match e {
        Entity::Physics => Record::Physics(doc.physics.clone().unwrap_or_default()),
        Entity::Include(n) => Record::Include(named(&doc.includes, n, |x| &x.name)?.clone()),
        Entity::Scenery(i) => Record::Scenery(doc.geoms.get(*i)?.clone()),
        Entity::Body(n) => Record::Body(named(&doc.bodies, n, |x| &x.name)?.clone()),
        Entity::Geom { body, index } => {
            let b = named(&doc.bodies, body, |x| &x.name)?;
            Record::Geom(b.geoms.get(*index)?.clone())
        }
        Entity::Camera(n) => Record::Camera(named(&doc.cameras, n, |x| &x.name)?.clone()),
        Entity::Light(n) => Record::Light(named(&doc.lights, n, |x| &x.name)?.clone()),
        Entity::Region(n) => Record::Region(named(&doc.regions, n, |x| &x.name)?.clone()),
    })
}

fn missing(doc: &EsScene, e: &Entity) -> Refusal {
    Refusal::new(e.field(doc), MISSING, vec![e.label(doc)])
}

fn cannot(doc: &EsScene, e: &Entity) -> Refusal {
    Refusal::new(e.field(doc), OTHER, vec![format!("{e:?}")])
}

/// Writes `r` over the entity, keeping its name: the record's kind must be the entity's.
fn put(doc: &mut EsScene, e: &Entity, r: Record) -> Result<(), Refusal> {
    let fail = missing(doc, e);
    let slot = match (e, r) {
        (Entity::Physics, Record::Physics(p)) => {
            doc.physics = (p != Physics::default()).then_some(p);
            Some(())
        }
        (Entity::Include(n), Record::Include(x)) => named_mut(&mut doc.includes, n, |x| &x.name)
            .map(|s| {
                *s = Include {
                    name: n.clone(),
                    ..x
                }
            }),
        (Entity::Scenery(i), Record::Scenery(x)) => doc.geoms.get_mut(*i).map(|s| *s = x),
        (Entity::Body(n), Record::Body(x)) => named_mut(&mut doc.bodies, n, |x| &x.name).map(|s| {
            *s = BodyDoc {
                name: n.clone(),
                ..x
            }
        }),
        (Entity::Geom { body, index }, Record::Geom(x)) => {
            let b = named_mut(&mut doc.bodies, body, |x| &x.name);
            b.and_then(|b| b.geoms.get_mut(*index)).map(|s| *s = x)
        }
        (Entity::Camera(n), Record::Camera(x)) => {
            named_mut(&mut doc.cameras, n, |x| &x.name).map(|s| {
                *s = CameraDoc {
                    name: n.clone(),
                    ..x
                }
            })
        }
        (Entity::Light(n), Record::Light(x)) => {
            named_mut(&mut doc.lights, n, |x| &x.name).map(|s| {
                *s = LightDoc {
                    name: n.clone(),
                    ..x
                }
            })
        }
        (Entity::Region(n), Record::Region(x)) => {
            named_mut(&mut doc.regions, n, |x| &x.name).map(|s| {
                *s = RegionDoc {
                    name: n.clone(),
                    ..x
                }
            })
        }
        _ => return Err(cannot(doc, e)),
    };
    slot.ok_or(fail)
}

/// The document's bodies hanging from `name`, `name` itself first.
pub(crate) fn subtree(doc: &EsScene, name: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::from([name.to_owned()]);
    // Parents come before children in a document that expands; one pass per level otherwise.
    for _ in 0..doc.bodies.len() {
        let before = out.len();
        for b in &doc.bodies {
            if b.parent.as_ref().is_some_and(|p| out.contains(p)) {
                out.insert(b.name.clone());
            }
        }
        if out.len() == before {
            break;
        }
    }
    out
}

/// The bodies in their order, but each after the body it hangs from (G1 reads a parent only
/// when it was defined before). A parent the document does not define (an included body) is
/// always there.
// ponytail: quadratic placement; documents hold tens of bodies, not thousands.
fn order(bodies: Vec<BodyDoc>) -> Vec<BodyDoc> {
    let names: BTreeSet<String> = bodies.iter().map(|b| b.name.clone()).collect();
    let mut placed: BTreeSet<String> = BTreeSet::new();
    let mut left = bodies;
    let mut out = Vec::with_capacity(left.len());
    while !left.is_empty() {
        let ready = |b: &BodyDoc| {
            b.parent
                .as_ref()
                .is_none_or(|p| !names.contains(p) || placed.contains(p))
        };
        let i = left.iter().position(ready).unwrap_or(0);
        let b = left.remove(i);
        placed.insert(b.name.clone());
        out.push(b);
    }
    out
}

/// `base` with `_<n>` after it (and any `_<n>` it had dropped) that `taken` does not hold.
pub(crate) fn unique(base: &str, taken: &dyn Fn(&str) -> bool) -> String {
    let stem = match base.rsplit_once('_') {
        Some((stem, n)) if !stem.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) => stem,
        _ => base,
    };
    if !taken(stem) && stem == base {
        return base.to_owned();
    }
    let mut n = 2;
    loop {
        let name = format!("{stem}_{n}");
        if !taken(&name) {
            return name;
        }
        n += 1;
    }
}

/// `s` naming `old` (itself, or `old.<field>`) names `new` after it.
fn swap(s: &mut String, old: &str, new: &str) {
    if s == old {
        new.clone_into(s);
    } else if let Some(rest) = s.strip_prefix(old).filter(|r| r.starts_with('.')) {
        // `robot.` is the specification's own word, not a scene name.
        if old != "robot" {
            *s = format!("{new}{rest}");
        }
    }
}

/// Every place `spec` names the scene thing `old`: the robot, each clause's subject and
/// object, each start item, the observed cameras and sources, the student's views.
pub(crate) fn rename_refs(spec: &mut TaskSpec, old: &str, new: &str) {
    swap(&mut spec.robot, old, new);
    let clauses = std::iter::once(&mut spec.success).chain(spec.failure.as_mut());
    for c in clauses.flat_map(|c| c.clauses.iter_mut()) {
        swap(&mut c.subject, old, new);
        if let Some(o) = &mut c.object {
            swap(o, old, new);
        }
    }
    for item in spec.start.iter_mut().flat_map(|s| s.items.iter_mut()) {
        swap(&mut item.what, old, new);
    }
    if let Some(o) = &mut spec.observe {
        for c in o.cameras.iter_mut().flatten() {
            swap(c, old, new);
        }
        for map in [&mut o.state, &mut o.privileged].into_iter().flatten() {
            for source in map.values_mut() {
                swap(source, old, new);
            }
        }
    }
    let views = spec.student.as_mut().and_then(|s| s.views.as_mut());
    for v in views.into_iter().flatten() {
        swap(v, old, new);
    }
}

/// Applies `cmd` to `d`; returns what to select after it. `taken` says whether a name is in
/// use anywhere in the scene (an included body's too), for the name a copy gets.
pub(crate) fn apply(
    d: &mut Docs,
    cmd: &Command,
    taken: &dyn Fn(&str) -> bool,
) -> Result<Option<Entity>, Refusal> {
    let doc = &mut d.scene;
    match cmd {
        Command::Add(r) => add(doc, r.clone()),
        Command::Delete(e) => delete(doc, e).map(|()| None),
        Command::Duplicate(e) => duplicate(doc, e, taken).map(Some),
        Command::Set(e, r) => put(doc, e, r.clone()).map(|()| Some(e.clone())),
        Command::SetPose { entity, pos, quat } => {
            let mut r = get(doc, entity).ok_or_else(|| missing(doc, entity))?;
            let (p, q) = r.pose_mut().ok_or_else(|| cannot(doc, entity))?;
            (*p, *q) = (Some(*pos), Some(*quat));
            put(doc, entity, r).map(|()| Some(entity.clone()))
        }
        Command::Reparent(e, parent) => reparent(doc, e, parent.clone()).map(|()| Some(e.clone())),
        Command::Scene(scene, select) => {
            doc.clone_from(scene);
            Ok(select.clone())
        }
        Command::Rename(e, new) => {
            let new = new.trim();
            if new.is_empty() {
                return Err(Refusal::new(
                    format!("{}.name", e.field(doc)),
                    EMPTY,
                    vec![],
                ));
            }
            let old = e.label(doc);
            let renamed = rename(doc, e, new)?;
            if let (
                Some(spec),
                Entity::Body(_) | Entity::Include(_) | Entity::Camera(_) | Entity::Region(_),
            ) = (&mut d.spec, e)
            {
                rename_refs(spec, &old, new);
            }
            Ok(Some(renamed))
        }
    }
}

fn add(doc: &mut EsScene, r: Record) -> Result<Option<Entity>, Refusal> {
    Ok(Some(match r {
        Record::Include(x) => {
            let e = Entity::Include(x.name.clone());
            doc.includes.push(x);
            e
        }
        Record::Scenery(x) => {
            doc.geoms.push(x);
            Entity::Scenery(doc.geoms.len() - 1)
        }
        Record::Body(x) => {
            let e = Entity::Body(x.name.clone());
            doc.bodies.push(x);
            e
        }
        Record::Camera(x) => {
            let e = Entity::Camera(x.name.clone());
            doc.cameras.push(x);
            e
        }
        Record::Light(x) => {
            let e = Entity::Light(x.name.clone());
            doc.lights.push(x);
            e
        }
        Record::Region(x) => {
            let e = Entity::Region(x.name.clone());
            doc.regions.push(x);
            e
        }
        Record::Physics(_) | Record::Geom(_) => {
            return Err(Refusal::new("scene", OTHER, vec![format!("{r:?}")]))
        }
    }))
}

fn delete(doc: &mut EsScene, e: &Entity) -> Result<(), Refusal> {
    if get(doc, e).is_none() {
        return Err(missing(doc, e));
    }
    match e {
        Entity::Physics => return Err(cannot(doc, e)),
        Entity::Body(n) => {
            let gone = subtree(doc, n);
            let hangs = |p: &Option<String>| p.as_ref().is_some_and(|p| gone.contains(p));
            doc.bodies.retain(|b| !gone.contains(&b.name));
            doc.cameras.retain(|c| !hangs(&c.parent));
            doc.regions.retain(|r| !hangs(&r.parent));
        }
        Entity::Include(n) => doc.includes.retain(|x| x.name != *n),
        // `get` found both.
        Entity::Scenery(i) => {
            doc.geoms.remove(*i);
        }
        Entity::Geom { body, index } => {
            if let Some(b) = named_mut(&mut doc.bodies, body, |x| &x.name) {
                b.geoms.remove(*index);
            }
        }
        Entity::Camera(n) => doc.cameras.retain(|x| x.name != *n),
        Entity::Light(n) => doc.lights.retain(|x| x.name != *n),
        Entity::Region(n) => doc.regions.retain(|x| x.name != *n),
    }
    Ok(())
}

fn duplicate(
    doc: &mut EsScene,
    e: &Entity,
    taken: &dyn Fn(&str) -> bool,
) -> Result<Entity, Refusal> {
    let r = get(doc, e).ok_or_else(|| missing(doc, e))?;
    // A named geom's name is unique within its list only.
    let fresh = |list: Vec<Option<String>>, name: &Option<String>| {
        name.as_ref()
            .map(|n| unique(n, &|c| list.iter().flatten().any(|x| x == c)))
    };
    match (e, r) {
        (Entity::Physics, _) => return Err(cannot(doc, e)),
        (Entity::Include(n), Record::Include(mut x)) => {
            x.name = unique(n, taken);
            // The copy's names must not meet the original's: they get the copy's handle.
            x.prefix.get_or_insert_with(|| format!("{}:", x.name));
            let at = doc.includes.iter().position(|i| i.name == *n).unwrap_or(0);
            doc.includes.insert(at + 1, x.clone());
            return Ok(Entity::Include(x.name));
        }
        (Entity::Scenery(i), Record::Scenery(mut x)) => {
            x.name = fresh(doc.geoms.iter().map(|g| g.name.clone()).collect(), &x.name);
            doc.geoms.insert(i + 1, x);
            return Ok(Entity::Scenery(i + 1));
        }
        (Entity::Geom { body, index }, Record::Geom(mut x)) => {
            let b = named_mut(&mut doc.bodies, body, |x| &x.name).expect("get found it");
            x.name = fresh(b.geoms.iter().map(|g| g.name.clone()).collect(), &x.name);
            b.geoms.insert(index + 1, x);
            return Ok(Entity::Geom {
                body: body.clone(),
                index: index + 1,
            });
        }
        (Entity::Body(n), Record::Body(mut x)) => {
            x.name = unique(n, taken);
            if let Some(j) = x.joint.as_mut().and_then(|j| j.name.as_mut()) {
                *j = unique(j, taken);
            }
            let at = doc.bodies.iter().position(|b| b.name == *n).unwrap_or(0);
            doc.bodies.insert(at + 1, x.clone());
            return Ok(Entity::Body(x.name));
        }
        (Entity::Camera(n), Record::Camera(mut x)) => {
            x.name = unique(n, taken);
            let at = doc.cameras.iter().position(|c| c.name == *n).unwrap_or(0);
            doc.cameras.insert(at + 1, x.clone());
            return Ok(Entity::Camera(x.name));
        }
        (Entity::Light(n), Record::Light(mut x)) => {
            x.name = unique(n, taken);
            let at = doc.lights.iter().position(|c| c.name == *n).unwrap_or(0);
            doc.lights.insert(at + 1, x.clone());
            return Ok(Entity::Light(x.name));
        }
        (Entity::Region(n), Record::Region(mut x)) => {
            x.name = unique(n, taken);
            let at = doc.regions.iter().position(|c| c.name == *n).unwrap_or(0);
            doc.regions.insert(at + 1, x.clone());
            return Ok(Entity::Region(x.name));
        }
        _ => {}
    }
    Err(cannot(doc, e))
}

fn reparent(doc: &mut EsScene, e: &Entity, parent: Option<String>) -> Result<(), Refusal> {
    let fail = missing(doc, e);
    match e {
        Entity::Body(n) => {
            if parent.as_ref().is_some_and(|p| subtree(doc, n).contains(p)) {
                let field = format!("{}.parent", e.field(doc));
                return Err(Refusal::new(field, CYCLE, vec![n.clone()]));
            }
            let b = named_mut(&mut doc.bodies, n, |x| &x.name).ok_or(fail)?;
            b.parent = parent;
            doc.bodies = order(std::mem::take(&mut doc.bodies));
        }
        Entity::Camera(n) => {
            let c = named_mut(&mut doc.cameras, n, |x| &x.name).ok_or(fail)?;
            c.parent = parent;
        }
        Entity::Region(n) => {
            let r = named_mut(&mut doc.regions, n, |x| &x.name).ok_or(fail)?;
            r.parent = parent;
        }
        _ => return Err(cannot(doc, e)),
    }
    Ok(())
}

/// Renames the entity and what the scene document says of it; the specification is the
/// caller's.
fn rename(doc: &mut EsScene, e: &Entity, new: &str) -> Result<Entity, Refusal> {
    let fail = missing(doc, e);
    let some = |o: Option<()>| o.ok_or(fail.clone());
    Ok(match e {
        Entity::Physics => return Err(cannot(doc, e)),
        Entity::Include(n) => {
            some(
                named_mut(&mut doc.includes, n, |x| &x.name).map(|x| new.clone_into(&mut x.name)),
            )?;
            Entity::Include(new.to_owned())
        }
        Entity::Scenery(i) => {
            some(doc.geoms.get_mut(*i).map(|g| g.name = Some(new.to_owned())))?;
            e.clone()
        }
        Entity::Geom { body, index } => {
            let b = named_mut(&mut doc.bodies, body, |x| &x.name);
            some(
                b.and_then(|b| b.geoms.get_mut(*index))
                    .map(|g| g.name = Some(new.to_owned())),
            )?;
            e.clone()
        }
        Entity::Body(n) => {
            some(named_mut(&mut doc.bodies, n, |x| &x.name).map(|x| new.clone_into(&mut x.name)))?;
            let parents = (doc.bodies.iter_mut().map(|b| &mut b.parent))
                .chain(doc.cameras.iter_mut().map(|c| &mut c.parent))
                .chain(doc.regions.iter_mut().map(|r| &mut r.parent));
            for p in parents.flatten() {
                if p == n {
                    new.clone_into(p);
                }
            }
            Entity::Body(new.to_owned())
        }
        Entity::Camera(n) => {
            some(named_mut(&mut doc.cameras, n, |x| &x.name).map(|x| new.clone_into(&mut x.name)))?;
            Entity::Camera(new.to_owned())
        }
        Entity::Light(n) => {
            some(named_mut(&mut doc.lights, n, |x| &x.name).map(|x| new.clone_into(&mut x.name)))?;
            Entity::Light(new.to_owned())
        }
        Entity::Region(n) => {
            some(named_mut(&mut doc.regions, n, |x| &x.name).map(|x| new.clone_into(&mut x.name)))?;
            Entity::Region(new.to_owned())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_free_name_counts_up_from_the_stem() {
        let taken = |n: &str| ["cube", "cube_2", "box"].contains(&n);
        assert_eq!(unique("cube", &taken), "cube_3");
        assert_eq!(unique("cube_2", &taken), "cube_3");
        assert_eq!(unique("ball", &taken), "ball");
        assert_eq!(unique("ball_7", &taken), "ball_2");
        assert_eq!(unique("_1", &taken), "_1");
    }

    #[test]
    fn a_reference_is_its_name_or_a_field_of_it() {
        let s = |x: &str, old: &str| {
            let mut x = x.to_owned();
            swap(&mut x, old, "cube");
            x
        };
        assert_eq!(s("object", "object"), "cube");
        assert_eq!(s("object.x", "object"), "cube.x");
        assert_eq!(s("object:joint", "object"), "object:joint");
        assert_eq!(s("objects", "object"), "objects");
        assert_eq!(s("robot.joints", "robot"), "robot.joints");
    }
}
