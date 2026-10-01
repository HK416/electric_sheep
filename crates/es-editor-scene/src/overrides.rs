//! An include's overrides in the inspector (packet M17/G7, design section 3.2: game engines'
//! prefab overrides): G1's `[include.set]` targets — a joint's `range`, `damping`, `armature`,
//! `stiffness`, `frictionloss`; an actuator's `kp`, `kv`, `ctrlrange`, `forcerange`; a geom's
//! `rgba`, `material` — addressed by the included file's own names.
//!
//! What the file says of each ([`SceneModel::brought`]) is read from the file alone, before any
//! override and any prefix, so the inspector can show a value the person has not changed and
//! put a changed one back. An edit is G5's `Set` of the include with its new `set`, which
//! [`prune`] has stripped of every override that overrides nothing: clearing a field drops it,
//! and a target with nothing left drops its table.

use es_assets::esscene::{expand, ActuatorSet, GeomSet, Include, IncludeSet, JointSet};
use es_assets::scene::{ActuatorKind, AssetKind, JointKind};

use crate::import;
use crate::model::SceneModel;

/// What an included file declares, by its own names, in its order, with its own values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Brought {
    /// Each joint that can have limits (not a free one), whether its range is in metres (a
    /// slide), and its values.
    pub joints: Vec<(String, bool, JointSet)>,
    /// Each actuator's values; `kp` and `kv` only where it has that gain.
    pub actuators: Vec<(String, ActuatorSet)>,
    pub geoms: Vec<(String, GeomSet)>,
    /// The file's materials, which a geom may name besides the document's.
    pub materials: Vec<String>,
}

impl SceneModel {
    /// What the include `name`'s file declares; `None` when there is no such include or its file
    /// does not read.
    pub fn brought(&self, name: &str) -> Option<Brought> {
        let inc = self.doc().includes.iter().find(|i| i.name == name)?;
        let mut doc = import::empty();
        doc.includes.push(Include {
            name: inc.name.clone(),
            source: inc.source.clone(),
            prefix: None,
            pos: None,
            quat: None,
            set: None,
        });
        let s = expand(&doc, self.root()).ok()?;
        let asset = |id| s.assets.iter().find(|a| a.id == id).map(|a| a.name.clone());
        let joints = (s.joints.iter().filter(|j| j.kind != JointKind::Free))
            .map(|j| {
                let set = JointSet {
                    range: j.range.map(|(lo, hi)| [lo, hi]),
                    damping: Some(j.damping),
                    armature: Some(j.armature),
                    stiffness: Some(j.stiffness),
                    frictionloss: Some(j.friction_loss),
                };
                (j.name.clone(), j.kind == JointKind::Slide, set)
            })
            .collect();
        let actuators = (s.actuators.iter())
            .map(|a| {
                let (kp, kv) = match a.kind {
                    ActuatorKind::Position { kp, kv } => (Some(kp), Some(kv)),
                    ActuatorKind::Velocity { kv } => (None, Some(kv)),
                    _ => (None, None),
                };
                let set = ActuatorSet {
                    kp,
                    kv,
                    ctrlrange: a.ctrl_range.map(|(lo, hi)| [lo, hi]),
                    forcerange: a.force_range.map(|(lo, hi)| [lo, hi]),
                };
                (a.name.clone(), set)
            })
            .collect();
        let geoms = (s.bodies.iter().flat_map(|b| &b.geoms))
            .map(|g| {
                let set = GeomSet {
                    rgba: Some(g.rgba),
                    material: g.material.and_then(asset),
                };
                (g.name.clone(), set)
            })
            .collect();
        let materials = (s.assets.iter())
            .filter(|a| a.kind == AssetKind::Material)
            .map(|a| a.name.clone())
            .collect();
        Some(Brought {
            joints,
            actuators,
            geoms,
            materials,
        })
    }
}

/// `set` without the overrides that override nothing and without empty tables; `None` when
/// nothing is left.
pub fn prune(set: Option<IncludeSet>) -> Option<IncludeSet> {
    let mut s = set?;
    s.joint.retain(|_, v| *v != JointSet::default());
    s.actuator.retain(|_, v| *v != ActuatorSet::default());
    s.geom.retain(|_, v| *v != GeomSet::default());
    (s != IncludeSet::default()).then_some(s)
}
