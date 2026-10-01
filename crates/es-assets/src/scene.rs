//! `SceneDesc` — the backend-neutral scene description (P30).
//!
//! Every importer (MJCF, URDF, glTF) produces a `SceneDesc`; every physics backend consumes
//! one. Nothing here computes physics: the mapping from these fields onto a solver is the
//! backend's job (spec 17.2), and unmapped fields are that backend's report, not a parse error.
//!
//! Conventions are the spec 3.1 ones, carried by `es-math`: right-handed Z-up, metres,
//! radians, kilograms, seconds, quaternions xyzw with `w >= 0`.
//!
//! # Stable ids (spec 5.3)
//!
//! Every element carries a [`StableId`] derived from its *name path* with [`scene_id`]:
//!
//! ```text
//! body       body/<body path>                 e.g. body/world/arm/link1
//! joint      joint/<owning body path>/<name>  e.g. joint/world/arm/link1/elbow
//! geom       geom/<owning body path>/<name>
//! site       site/<owning body path>/<name>
//! camera     camera/<owning body path>/<name>
//! actuator   actuator/<name>
//! sensor     sensor/<name>
//! tendon     tendon/<name>
//! asset      asset/<kind>/<name>
//! ```
//!
//! A body path is the `/`-joined chain of body names from the root body down to the body
//! itself — the MJCF importer names the root body `world`, so every element has an owner —
//! so an id depends on names only — not on element order, file layout, or insertion index.
//! Elements the source file leaves unnamed are given the local name `<kind><n>` with `n`
//! counted per owner in document order; such an id *is* order-dependent, which is the honest
//! consequence of the source not naming the element.
//!
//! # Hashing
//!
//! [`SceneDesc::scene_hash`] is blake3 over a canonical encoding: entities sorted by id, `f64`
//! as IEEE bits with `-0.0` normalised to `0.0` and `NaN` to one fixed pattern. Reordering
//! elements in the source file therefore cannot change the hash. Each [`AssetRef`] contributes
//! its own `asset_hash`, so `asset_hash -> scene_hash` of spec 5.3 holds by construction.
//!
//! Layout: the element types are `scene/elements.rs`, the solver options `scene/options.rs`,
//! [`SceneDesc::validate`] `scene/validate.rs` and [`SceneDesc::scene_hash`] with its
//! canonical encoding `scene/hash.rs`; the description itself, [`scene_id`],
//! [`SceneDesc::mesh_positions`] and the tests are here, and every type keeps its
//! `es_assets::scene::` path.

use std::borrow::Cow;
use std::collections::BTreeMap;

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::gltf::MeshData;

mod elements;
mod hash;
mod options;
mod validate;

pub use elements::{
    Actuator, ActuatorKind, ActuatorTarget, AssetKind, AssetRef, Body, BodyInertial, Camera,
    ContactPair, Geom, Joint, JointKind, Material, Sensor, SensorKind, SensorTarget, Shape, Site,
    Tendon, TendonKind,
};
pub use options::{FrictionCone, Integrator, Jacobian, PhysicsOptions, Solver};
pub use validate::SceneError;

/// Derives the id of `path` within `kind`, e.g. `scene_id("joint", "arm/link1/elbow")`.
///
/// The only way a scene element gets an id; see the module docs for the path scheme.
pub fn scene_id(kind: &str, path: &str) -> StableId {
    StableId::from_path(&format!("{kind}/{path}"))
}

/// A complete scene: bodies and their tree, joints, actuation, sensing and solver options.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SceneDesc {
    pub name: String,
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub actuators: Vec<Actuator>,
    pub sensors: Vec<Sensor>,
    pub tendons: Vec<Tendon>,
    pub cameras: Vec<Camera>,
    pub assets: Vec<AssetRef>,
    pub options: PhysicsOptions,
    /// Decoded geometry for the [`AssetKind::Mesh`] entries of `assets`, filled by
    /// [`crate::mesh::load`] and keyed by the asset's id.
    ///
    /// **Not** part of [`SceneDesc::scene_hash`]: what the chain covers is the content digest
    /// `load` writes into the [`AssetRef`] (spec 5.3), so hashing the vertices here as well
    /// would count them twice and would move every primitives-only scene the day the field
    /// was added (spec 28.13 rule 2).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub meshes: BTreeMap<StableId, MeshData>,
    /// MJCF `<contact><pair>`: geom pairs that collide whatever their `contype` /
    /// `conaffinity` say, with the contact parameters the pair overrides (plan H, H1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contact_pairs: Vec<ContactPair>,
    /// MJCF `<contact><exclude>`: body pairs that never collide.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contact_excludes: Vec<(StableId, StableId)>,
    /// MJCF `<body gravcomp>`, by body id: the fraction of the body's weight cancelled by a
    /// passive force. A body absent here has none.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gravcomp: BTreeMap<StableId, f64>,
    /// The `<material>`s that change how a geom is drawn, by asset id (plan H, HT1): those that
    /// name a texture or write one of `specular`, `shininess`, `metallic`, `roughness`,
    /// `emission` explicitly. A material that only carries `rgba` is not here — it rendered as
    /// the geom's own `rgba` before textures existed and still does, so no committed scene
    /// moves. Hashed into [`SceneDesc::scene_hash`] in its own section, present only when
    /// this map is non-empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub materials: BTreeMap<StableId, Material>,
    /// Every `<texture>`, by asset id: its declaration and, once [`crate::mesh::load`] ran,
    /// its texels. **Not** hashed here: the content digest `load` writes into the texture's
    /// [`AssetRef`] is what the chain covers, as for meshes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub textures: BTreeMap<StableId, crate::texture::Texture>,
    /// MJCF `<mesh scale>`, by mesh asset id (packet M17/R3): every vertex is multiplied by it
    /// before anything draws or simulates the mesh ([`SceneDesc::mesh_positions`]). A mesh
    /// absent here has scale 1. `meshes` keeps the file's own vertices, so a writer re-encodes
    /// the file unchanged. Hashed into [`SceneDesc::scene_hash`] in its own section, present
    /// only when this map is non-empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mesh_scales: BTreeMap<StableId, [f64; 3]>,
}

impl SceneDesc {
    /// The vertices of mesh asset `id` as they are drawn and simulated: the decoded file's
    /// times the mesh's scale, in `f64` and rounded to `f32` (`MuJoCo`'s arithmetic); the
    /// decoded ones, borrowed, when it has none. `None` when the mesh is not loaded.
    pub fn mesh_positions(&self, id: StableId) -> Option<Cow<'_, [[f32; 3]]>> {
        let positions = &self.meshes.get(&id)?.positions;
        Some(match self.mesh_scales.get(&id) {
            None => Cow::Borrowed(positions),
            Some(s) => Cow::Owned(
                (positions.iter())
                    .map(|p| [0, 1, 2].map(|k| (f64::from(p[k]) * s[k]) as f32))
                    .collect(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_math::{Pose, Vec3};

    fn body(name: &str, parent: Option<StableId>) -> Body {
        Body {
            id: scene_id("body", name),
            name: name.to_owned(),
            parent,
            pose: Pose::IDENTITY,
            inertial: None,
            geoms: Vec::new(),
            sites: Vec::new(),
        }
    }

    fn scene() -> SceneDesc {
        let a = body("a", None);
        let b = body("b", Some(a.id));
        SceneDesc {
            name: "t".to_owned(),
            joints: vec![Joint {
                id: scene_id("joint", "a/b/j"),
                name: "j".to_owned(),
                body: b.id,
                kind: JointKind::Hinge,
                axis: Vec3::new(0.0, 0.0, 1.0),
                anchor: Vec3::ZERO,
                range: Some((-1.0, 1.0)),
                damping: 0.1,
                armature: 0.0,
                stiffness: 0.0,
                friction_loss: 0.0,
                spring_ref: 0.0,
            }],
            bodies: vec![a, b],
            ..SceneDesc::default()
        }
    }

    #[test]
    fn hash_ignores_element_order() {
        let mut reordered = scene();
        reordered.bodies.reverse();
        assert_eq!(scene().scene_hash(), reordered.scene_hash());
    }

    #[test]
    fn hash_tracks_values() {
        let mut changed = scene();
        changed.joints[0].damping = 0.2;
        assert_ne!(scene().scene_hash(), changed.scene_hash());
    }

    #[test]
    fn negative_zero_hashes_as_zero() {
        let mut a = scene();
        let mut b = scene();
        a.options.timestep = 0.0;
        b.options.timestep = -0.0;
        assert_eq!(a.scene_hash(), b.scene_hash());
    }

    #[test]
    fn validate_accepts_a_tree_and_rejects_dangling_refs() {
        assert!(scene().validate().is_ok());

        let mut orphan = scene();
        orphan.bodies[1].parent = Some(scene_id("body", "ghost"));
        assert!(matches!(
            orphan.validate(),
            Err(SceneError::MissingParent { .. })
        ));

        let mut dangling = scene();
        dangling.joints[0].body = scene_id("body", "ghost");
        assert!(matches!(
            dangling.validate(),
            Err(SceneError::DanglingRef { kind: "joint", .. })
        ));
    }

    #[test]
    fn validate_rejects_duplicate_ids_and_cycles() {
        let mut dup = scene();
        dup.bodies.push(body("a", None));
        assert!(matches!(
            dup.validate(),
            Err(SceneError::DuplicateId { .. })
        ));

        let mut cycle = scene();
        let (a, b) = (cycle.bodies[0].id, cycle.bodies[1].id);
        cycle.bodies[0].parent = Some(b);
        cycle.bodies[1].parent = Some(a);
        assert!(matches!(
            cycle.validate(),
            Err(SceneError::BodyCycle { .. })
        ));
    }

    #[test]
    fn asset_hash_follows_the_path() {
        let a = AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm.stl");
        assert_eq!(
            a.hash,
            AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm.stl").hash
        );
        assert_ne!(
            a.hash,
            AssetRef::from_path(AssetKind::Mesh, "arm", "meshes/arm2.stl").hash
        );
        // The asset feeds the scene hash (spec 5.3).
        let mut with_asset = scene();
        with_asset.assets.push(a);
        assert_ne!(scene().scene_hash(), with_asset.scene_hash());
    }
}
