//! The specification's lists: its clauses, its start items, the cameras it observes, the look
//! of the observed pictures and the observed sources.

use es_assets::scene::SceneDesc;
use es_ir::task::SeedStream;
use es_script::spec::{Clauses, Observe, RenderDoc, RenderPath, Start, StartItem};

use super::edit::{bare, free_bodies, here};
use super::{set_relation, vocab, At, Clause, Relation, TaskSpec};

fn list(spec: &mut TaskSpec, failure: bool) -> &mut Vec<Clause> {
    if failure {
        &mut spec.failure.get_or_insert_with(Clauses::default).clauses
    } else {
        &mut spec.success.clauses
    }
}

fn tidy(spec: &mut TaskSpec) {
    if spec.failure.as_ref().is_some_and(|f| f.clauses.is_empty()) {
        spec.failure = None;
    }
    if spec.observe.as_ref() == Some(&Observe::default()) {
        spec.observe = None;
    }
    let start = spec.start.as_ref();
    // `zero_unset` is no sentence but the committed fixtures' switch: it keeps the table.
    if start.is_some_and(|s| s.items.is_empty() && s.strength.is_none() && s.zero_unset.is_none()) {
        spec.start = None;
    }
}

/// A new clause: the first free body (or subject), still if it can be, else its first relation.
pub fn add_clause(spec: &mut TaskSpec, scene: &SceneDesc, failure: bool) {
    let subject = (free_bodies(scene).into_iter().next())
        .or_else(|| vocab::subjects(scene).into_iter().next())
        .unwrap_or_default();
    let offered = vocab::relations(scene, &subject);
    let (r, o) = (offered.iter())
        .find(|(r, _)| *r == Relation::Still)
        .or(offered.first())
        .copied()
        .unwrap_or((Relation::Still, false));
    let mut c = bare(subject, r);
    set_relation(scene, &mut c, r, o);
    list(spec, failure).push(c);
}

pub fn remove_clause(spec: &mut TaskSpec, at: At) {
    let l = list(spec, at.failure);
    if at.index < l.len() {
        l.remove(at.index);
    }
    tidy(spec);
}

/// Moves a clause one place up (`down = false`) or down in its section.
pub fn move_clause(spec: &mut TaskSpec, at: At, down: bool) {
    let l = list(spec, at.failure);
    let to = if down {
        at.index + 1
    } else {
        at.index.wrapping_sub(1)
    };
    if at.index < l.len() && to < l.len() {
        l.swap(at.index, to);
    }
    tidy(spec);
}

fn item(what: String) -> StartItem {
    StartItem {
        what,
        value: None,
        noise: None,
        range: None,
        draw: None,
        tilt: None,
        tilt_max_deg: None,
        coupled: None,
        dice: None,
        stream: None,
    }
}

/// A new 🎲 placement: the first free body's x where it stands, ± 2 cm.
pub fn add_item(spec: &mut TaskSpec, scene: &SceneDesc) {
    let Some(body) = free_bodies(scene).into_iter().next() else {
        return;
    };
    let what = format!("{body}.x");
    let it = StartItem {
        value: Some(here(scene, &what)),
        noise: Some(0.02),
        dice: Some(true),
        ..item(what)
    };
    spec.start.get_or_insert_with(Start::default).items.push(it);
}

pub fn remove_item(spec: &mut TaskSpec, index: usize) {
    if let Some(s) = spec.start.as_mut().filter(|s| index < s.items.len()) {
        s.items.remove(index);
    }
    tidy(spec);
}

/// The camera named `camera` observed or not (a new one appended, `camera_px` 96 if unset); a
/// camera no longer observed leaves the student's views too.
pub fn set_camera(spec: &mut TaskSpec, camera: &str, on: bool) {
    let o = spec.observe.get_or_insert_with(Observe::default);
    let list = o.cameras.get_or_insert_with(Vec::new);
    if on && !list.iter().any(|c| c == camera) {
        list.push(camera.to_owned());
        o.camera_px.get_or_insert(96);
    }
    if !on {
        list.retain(|c| c != camera);
        if let Some(v) = spec.student.as_mut().and_then(|s| s.views.as_mut()) {
            v.retain(|c| c != camera);
        }
    }
    if o.cameras.as_ref().is_some_and(Vec::is_empty) {
        o.cameras = None;
    }
    tidy(spec);
}

/// The look of the observed pictures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    /// The rasterizer (no `render`).
    Quick,
    /// The viewport's material look: the Task IR has no such sensor path yet (offered disabled).
    Material,
    /// The path tracer at `spp` samples.
    Traced,
}

pub fn look(spec: &TaskSpec) -> Look {
    let o = spec.observe.as_ref();
    match o.and_then(|o| o.render.as_ref()).and_then(|r| r.path) {
        Some(RenderPath::Pt) => Look::Traced,
        _ => Look::Quick,
    }
}

/// Quick drops `render`; traced starts at plan H's 32 samples, 3 bounces and a seed per tick.
pub fn set_look(spec: &mut TaskSpec, to: Look) {
    if to == look(spec) || to == Look::Material {
        return;
    }
    let o = spec.observe.get_or_insert_with(Observe::default);
    o.render = (to == Look::Traced).then(|| RenderDoc {
        path: Some(RenderPath::Pt),
        spp: Some(32),
        bounces: Some(3),
        seed: Some(SeedStream::Tick),
        ..RenderDoc::default()
    });
    tidy(spec);
}

/// Observes `source` on a new channel named after it (`cube.pose` → `cube_pose`), in the
/// privileged table (the teacher's alone) or the state one.
pub fn add_source(spec: &mut TaskSpec, source: &str, privileged: bool) {
    let o = spec.observe.get_or_insert_with(Observe::default);
    let base = source.trim_start_matches("robot.").replace('.', "_");
    let taken = |n: &str| {
        [&o.state, &o.privileged]
            .iter()
            .any(|t| t.as_ref().is_some_and(|t| t.contains_key(n)))
    };
    let name = crate::command::unique(&base, &taken);
    let table = if privileged {
        &mut o.privileged
    } else {
        &mut o.state
    };
    table
        .get_or_insert_with(Default::default)
        .insert(name, source.to_owned());
}

/// Stops observing `channel`; the teacher and the student stop reading it too.
pub fn remove_source(spec: &mut TaskSpec, channel: &str) {
    if let Some(o) = spec.observe.as_mut() {
        for t in [&mut o.state, &mut o.privileged] {
            if let Some(map) = t.as_mut() {
                map.remove(channel);
            }
            if t.as_ref().is_some_and(std::collections::BTreeMap::is_empty) {
                *t = None;
            }
        }
    }
    let lists = [
        spec.teacher.as_mut().and_then(|t| t.state.as_mut()),
        spec.student.as_mut().and_then(|s| s.state.as_mut()),
    ];
    for l in lists.into_iter().flatten() {
        l.retain(|c| c != channel);
    }
    tidy(spec);
}

/// Moves `channel` between the state table and the privileged one.
pub fn swap_source(spec: &mut TaskSpec, channel: &str) {
    let Some(o) = spec.observe.as_mut() else {
        return;
    };
    let from_state = o.state.as_ref().is_some_and(|t| t.contains_key(channel));
    let (from, to) = if from_state {
        (&mut o.state, &mut o.privileged)
    } else {
        (&mut o.privileged, &mut o.state)
    };
    let Some(source) = from.as_mut().and_then(|t| t.remove(channel)) else {
        return;
    };
    if from
        .as_ref()
        .is_some_and(std::collections::BTreeMap::is_empty)
    {
        *from = None;
    }
    to.get_or_insert_with(Default::default)
        .insert(channel.to_owned(), source);
}
