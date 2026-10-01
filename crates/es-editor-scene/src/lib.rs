//! `es-editor-scene` (layer 12): the editor's scene model (plan G, packet G5;
//! `docs/design/scene-authoring.md` section 5) — the scene document `scene.esscene` (G1) of an
//! editable project, a selection, the [`Command`]s that change it, an undo / redo stack, the
//! check every command passes, the dirty mark, the save and the regeneration of the project's
//! documents (G3b's `generate`). Headless: `es-editor` (layer 13) draws the hierarchy and the
//! inspector over it, and CI judges all of it with `cargo test`.
//!
//! Layer rule (spec 4.2 rule 4): **nothing depends on this crate except `es-editor`**, as for
//! `es-editor-model`, its sibling on layer 12, which it does not know (rule 1: no same-layer
//! dependency). It is a crate of its own because `es-editor-model` is near its line cap (spec
//! 1.5); words a person reads are keys of that crate's string tables, named here as plain
//! `&'static str`.
//!
//! An editable project is a project folder that holds `scene.esscene` and, once the task is
//! said, `task.estask` (G3a); [`make_editable`] writes both from a template's.
//!
//! The viewport's decisions are here too (packet M17/G6): what a click selects and how big the
//! selection is ([`view`]), the move, turn and size handles and the one command a drag of them
//! makes ([`gizmo`]), and the camera the policy sees for the corner ([`policy`]).
//!
//! ①'s Add (packet M17/G7): what each item makes and where it goes ([`add`]), files copied into
//! the project by content ([`import`]), cameras and regions drawn and picked as lines
//! ([`marker`]), and an include's overrides ([`overrides`]).

pub mod add;
pub mod check;
pub mod command;
pub mod copy;
pub mod euler;
pub mod gizmo;
pub mod import;
pub mod inspect;
pub mod marker;
pub mod model;
pub mod overrides;
pub mod policy;
pub mod tree;
pub mod view;

use std::path::Path;

pub use add::{Item, Robot};
pub use check::Refusal;
pub use command::{
    new_body, new_camera, new_geom, new_joint, new_light, new_region, Command, Entity, Record,
};
pub use copy::make_editable;
pub use es_physics_backend::BackendKind;
pub use gizmo::{Drag, Gizmo, Step, Tool};
pub use model::{Regen, SceneModel};
pub use policy::PolicyCamera;
pub use tree::{Row, RowKind};
pub use view::{Camera, Ray};

/// The scene document of an editable project, in its root.
pub const SCENE_FILE: &str = "scene.esscene";
/// The task specification, beside it.
pub const SPEC_FILE: &str = "task.estask";
/// Where the project's documents are regenerated, relative to its root (the `out` the recipes
/// name).
pub const GENERATED_DIR: &str = "generated";
/// The edited, unsaved document, written for what reads a scene from a file (the physics
/// preview, `es render`): beside the scene, so the files it names resolve the same way.
pub const PREVIEW_FILE: &str = ".preview.esscene";

/// Whether the project at `root` is editable: its scene document is there.
pub fn is_editable(root: &Path) -> bool {
    root.join(SCENE_FILE).is_file()
}
