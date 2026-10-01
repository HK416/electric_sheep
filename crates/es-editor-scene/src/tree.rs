//! The hierarchy panel's rows (design section 5): the physics settings, each include folded
//! over the bodies it brings, the static scenery, the bodies as a tree with their geoms, and the
//! cameras, lights and regions — a camera or region under the body it hangs from. And what the
//! viewport draws when some of them are hidden: visibility is the editor's, never the document's.

use std::collections::BTreeSet;
use std::path::Path;

use es_assets::esscene::{expand, EsScene};
use es_assets::scene::SceneDesc;

use crate::command::Entity;

/// What a row is, for its icon and its words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowKind {
    Physics,
    Include,
    /// A body an include brings: shown, not edited here (its overrides are G7's).
    Part,
    Scenery,
    Body,
    Geom,
    Camera,
    Light,
    Region,
}

impl RowKind {
    /// The key of its word in the editor's string tables.
    pub fn key(self) -> &'static str {
        match self {
            Self::Physics => "author.kind.physics",
            Self::Include => "author.kind.include",
            Self::Part => "author.kind.part",
            Self::Scenery => "author.kind.scenery",
            Self::Body => "author.kind.body",
            Self::Geom => "author.kind.geom",
            Self::Camera => "author.kind.camera",
            Self::Light => "author.kind.light",
            Self::Region => "author.kind.region",
        }
    }

    /// Whether the viewport draws it, so an eye can hide it.
    pub fn drawn(self) -> bool {
        matches!(
            self,
            Self::Include | Self::Scenery | Self::Body | Self::Geom | Self::Light
        )
    }
}

/// One line of the hierarchy, in drawing order.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub depth: usize,
    /// The row it folds under.
    pub parent: Option<usize>,
    pub name: String,
    pub kind: RowKind,
    /// `None` for an include's parts.
    pub entity: Option<Entity>,
}

/// What an include brings: its bodies (name, parent name) in its order, and its world geoms.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contents {
    pub bodies: Vec<(String, Option<String>)>,
    pub scenery: Vec<String>,
}

/// Each include's [`Contents`], by expanding the document with that include alone (and the
/// materials its overrides may name). A failing one brings nothing it can show.
pub fn contents(doc: &EsScene, dir: &Path) -> Vec<Contents> {
    doc.includes
        .iter()
        .map(|inc| {
            let alone = EsScene {
                includes: vec![inc.clone()],
                geoms: Vec::new(),
                bodies: Vec::new(),
                cameras: Vec::new(),
                lights: Vec::new(),
                regions: Vec::new(),
                ..doc.clone()
            };
            let Ok(s) = expand(&alone, dir) else {
                return Contents::default();
            };
            let name = |id| {
                s.bodies
                    .iter()
                    .find(|b| Some(b.id) == id)
                    .map(|b| b.name.clone())
            };
            Contents {
                bodies: (s.bodies.iter().skip(1))
                    .map(|b| (b.name.clone(), name(b.parent).filter(|p| p != "world")))
                    .collect(),
                scenery: s.bodies[0].geoms.iter().map(|g| g.name.clone()).collect(),
            }
        })
        .collect()
}

/// Every row of `doc`.
pub fn rows(doc: &EsScene, includes: &[Contents]) -> Vec<Row> {
    let mut out: Vec<Row> = Vec::new();
    let push = |out: &mut Vec<Row>, parent: Option<usize>, name: String, kind, entity| {
        let depth = parent.map_or(0, |p: usize| out[p].depth + 1);
        out.push(Row {
            depth,
            parent,
            name,
            kind,
            entity,
        });
        out.len() - 1
    };
    push(
        &mut out,
        None,
        "physics".into(),
        RowKind::Physics,
        Some(Entity::Physics),
    );
    for (inc, c) in doc.includes.iter().zip(includes) {
        let e = Entity::Include(inc.name.clone());
        let at = push(&mut out, None, inc.name.clone(), RowKind::Include, Some(e));
        let mut parts: Vec<(String, usize)> = Vec::new();
        for (name, parent) in &c.bodies {
            let p = parent
                .as_ref()
                .and_then(|p| parts.iter().find(|(n, _)| n == p));
            let row = push(
                &mut out,
                Some(p.map_or(at, |(_, r)| *r)),
                name.clone(),
                RowKind::Part,
                None,
            );
            parts.push((name.clone(), row));
        }
    }
    for i in 0..doc.geoms.len() {
        let e = Entity::Scenery(i);
        push(&mut out, None, e.label(doc), RowKind::Scenery, Some(e));
    }
    // Bodies in document order: a parent is always defined before its children.
    let mut bodies: Vec<(String, usize)> = Vec::new();
    let under = |bodies: &[(String, usize)], p: &Option<String>| {
        p.as_ref()
            .and_then(|p| bodies.iter().find(|(n, _)| n == p))
            .map(|(_, r)| *r)
    };
    for b in &doc.bodies {
        let e = Entity::Body(b.name.clone());
        let row = push(
            &mut out,
            under(&bodies, &b.parent),
            b.name.clone(),
            RowKind::Body,
            Some(e),
        );
        bodies.push((b.name.clone(), row));
        for index in 0..b.geoms.len() {
            let e = Entity::Geom {
                body: b.name.clone(),
                index,
            };
            push(&mut out, Some(row), e.label(doc), RowKind::Geom, Some(e));
        }
    }
    for c in &doc.cameras {
        let e = Entity::Camera(c.name.clone());
        push(
            &mut out,
            under(&bodies, &c.parent),
            c.name.clone(),
            RowKind::Camera,
            Some(e),
        );
    }
    for l in &doc.lights {
        let e = Entity::Light(l.name.clone());
        push(&mut out, None, l.name.clone(), RowKind::Light, Some(e));
    }
    for r in &doc.regions {
        let e = Entity::Region(r.name.clone());
        push(
            &mut out,
            under(&bodies, &r.parent),
            r.name.clone(),
            RowKind::Region,
            Some(e),
        );
    }
    out
}

/// Which rows a search shows: those whose name holds `query` (any case) and the rows they fold
/// under. An empty query shows every row.
pub fn matches(rows: &[Row], query: &str) -> Vec<bool> {
    let query = query.trim().to_lowercase();
    let mut show = vec![query.is_empty(); rows.len()];
    if query.is_empty() {
        return show;
    }
    for (i, row) in rows.iter().enumerate() {
        if row.name.to_lowercase().contains(&query) {
            let mut at = Some(i);
            while let Some(j) = at {
                show[j] = true;
                at = rows[j].parent;
            }
        }
    }
    show
}

/// `scene` without the geoms of what is hidden: a hidden body hides the bodies under it, an
/// include everything it brings, a light its emitter. For drawing only: never hashed or saved.
pub fn drawn(
    scene: &SceneDesc,
    doc: &EsScene,
    includes: &[Contents],
    hidden: &BTreeSet<Entity>,
) -> SceneDesc {
    let mut out = scene.clone();
    if hidden.is_empty() {
        return out;
    }
    let mut bodies: BTreeSet<String> = BTreeSet::new();
    let mut world: BTreeSet<String> = BTreeSet::new();
    let mut geoms: BTreeSet<(String, usize)> = BTreeSet::new();
    let n_world = out.bodies[0].geoms.len();
    let first_doc_geom = n_world.saturating_sub(doc.geoms.len() + doc.lights.len());
    for e in hidden {
        match e {
            Entity::Include(n) => {
                let at = doc.includes.iter().position(|i| i.name == *n);
                if let Some(c) = at.and_then(|i| includes.get(i)) {
                    bodies.extend(c.bodies.iter().map(|(b, _)| b.clone()));
                    world.extend(c.scenery.iter().cloned());
                }
            }
            Entity::Body(n) => {
                bodies.insert(n.clone());
            }
            Entity::Scenery(i) => {
                if let Some(g) = out.bodies[0].geoms.get(first_doc_geom + i) {
                    world.insert(g.name.clone());
                }
            }
            Entity::Light(n) => {
                world.insert(format!("{n}_light"));
            }
            Entity::Geom { body, index } => {
                geoms.insert((body.clone(), *index));
            }
            Entity::Physics | Entity::Camera(_) | Entity::Region(_) => {}
        }
    }
    // A hidden body hides what hangs from it, down the tree (parents come first).
    let ids: Vec<_> = (out.bodies.iter())
        .filter(|b| bodies.contains(&b.name))
        .map(|b| b.id)
        .collect();
    let mut gone: BTreeSet<_> = ids.into_iter().collect();
    for b in &out.bodies {
        if b.parent.is_some_and(|p| gone.contains(&p)) {
            gone.insert(b.id);
        }
    }
    for (i, b) in out.bodies.iter_mut().enumerate() {
        if i == 0 {
            b.geoms.retain(|g| !world.contains(&g.name));
        } else if gone.contains(&b.id) {
            b.geoms.clear();
        } else {
            let mut index = 0;
            b.geoms.retain(|_| {
                index += 1;
                !geoms.contains(&(b.name.clone(), index - 1))
            });
        }
    }
    out
}
