//! `SceneDesc::validate`: unique ids, resolvable references and an acyclic body tree.

use std::collections::BTreeSet;

use es_core::StableId;
use thiserror::Error;

use super::{ActuatorTarget, AssetKind, SceneDesc, SensorTarget, TendonKind};

/// Why a [`SceneDesc`] is not usable. Structural only — nothing here is a physics judgement.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SceneError {
    #[error("duplicate stable id {id} on `{name}`: two elements share a name path")]
    DuplicateId { id: StableId, name: String },
    #[error("body `{body}` has a parent that is not in the scene")]
    MissingParent { body: String },
    #[error("body `{body}` is its own ancestor")]
    BodyCycle { body: String },
    #[error("{kind} `{name}` references an element that is not in the scene")]
    DanglingRef { kind: &'static str, name: String },
}

impl SceneDesc {
    /// Structural validation: unique ids, resolvable references, acyclic body tree.
    pub fn validate(&self) -> Result<(), SceneError> {
        let mut ids = BTreeSet::new();
        let mut bodies = BTreeSet::new();
        let mut named = |id: StableId, name: &str| -> Result<(), SceneError> {
            if ids.insert(id) {
                Ok(())
            } else {
                Err(SceneError::DuplicateId {
                    id,
                    name: name.to_owned(),
                })
            }
        };
        for body in &self.bodies {
            named(body.id, &body.name)?;
            bodies.insert(body.id);
            for geom in &body.geoms {
                named(geom.id, &geom.name)?;
            }
            for site in &body.sites {
                named(site.id, &site.name)?;
            }
        }
        for joint in &self.joints {
            named(joint.id, &joint.name)?;
        }
        for camera in &self.cameras {
            named(camera.id, &camera.name)?;
        }
        for tendon in &self.tendons {
            named(tendon.id, &tendon.name)?;
        }
        for actuator in &self.actuators {
            named(actuator.id, &actuator.name)?;
        }
        for sensor in &self.sensors {
            named(sensor.id, &sensor.name)?;
        }
        for asset in &self.assets {
            named(asset.id, &asset.name)?;
        }

        for body in &self.bodies {
            if body.parent.is_some_and(|p| !bodies.contains(&p)) {
                return Err(SceneError::MissingParent {
                    body: body.name.clone(),
                });
            }
        }
        self.check_acyclic()?;

        for joint in &self.joints {
            if !bodies.contains(&joint.body) {
                return Err(SceneError::DanglingRef {
                    kind: "joint",
                    name: joint.name.clone(),
                });
            }
        }
        for tendon in &self.tendons {
            let ok = match &tendon.kind {
                TendonKind::Fixed { joints } => joints.iter().all(|(j, _)| ids.contains(j)),
                TendonKind::Spatial { sites } => sites.iter().all(|s| ids.contains(s)),
            };
            if !ok {
                return Err(SceneError::DanglingRef {
                    kind: "tendon",
                    name: tendon.name.clone(),
                });
            }
        }
        for actuator in &self.actuators {
            let (ActuatorTarget::Joint(target)
            | ActuatorTarget::Tendon(target)
            | ActuatorTarget::Site(target)) = actuator.target;
            if !ids.contains(&target) {
                return Err(SceneError::DanglingRef {
                    kind: "actuator",
                    name: actuator.name.clone(),
                });
            }
        }
        for sensor in &self.sensors {
            let target = match sensor.target {
                SensorTarget::Body(id)
                | SensorTarget::Joint(id)
                | SensorTarget::Geom(id)
                | SensorTarget::Site(id)
                | SensorTarget::Camera(id)
                | SensorTarget::Actuator(id) => id,
            };
            if !ids.contains(&target) {
                return Err(SceneError::DanglingRef {
                    kind: "sensor",
                    name: sensor.name.clone(),
                });
            }
        }
        let pairs = self.contact_pairs.iter().map(|p| (p.geom1, p.geom2));
        for (a, b) in pairs.chain(self.contact_excludes.iter().copied()) {
            if !ids.contains(&a) || !ids.contains(&b) {
                return Err(SceneError::DanglingRef {
                    kind: "contact",
                    name: format!("{a} / {b}"),
                });
            }
        }
        if let Some(body) = self.gravcomp.keys().find(|b| !bodies.contains(b)) {
            return Err(SceneError::DanglingRef {
                kind: "gravcomp",
                name: body.to_string(),
            });
        }
        let mesh =
            |id: &StableId| (self.assets.iter()).any(|a| a.id == *id && a.kind == AssetKind::Mesh);
        if let Some(id) = self.mesh_scales.keys().find(|id| !mesh(id)) {
            return Err(SceneError::DanglingRef {
                kind: "mesh scale",
                name: id.to_string(),
            });
        }
        Ok(())
    }

    // ponytail: parent-chain walk per body, O(n * depth). A scene has hundreds of bodies and a
    // depth of ten; swap in a colour DFS if that ever stops being true.
    fn check_acyclic(&self) -> Result<(), SceneError> {
        for body in &self.bodies {
            let mut cursor = body.parent;
            for _ in 0..self.bodies.len() {
                let Some(id) = cursor else {
                    break;
                };
                if id == body.id {
                    return Err(SceneError::BodyCycle {
                        body: body.name.clone(),
                    });
                }
                cursor = self
                    .bodies
                    .iter()
                    .find(|b| b.id == id)
                    .and_then(|b| b.parent);
            }
            if cursor.is_some() {
                return Err(SceneError::BodyCycle {
                    body: body.name.clone(),
                });
            }
        }
        Ok(())
    }
}
