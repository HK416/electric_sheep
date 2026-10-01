//! ① of an editable project (packet M17/G5, `docs/design/scene-authoring.md` section 5): the
//! hierarchy on the left, the viewport's undo / redo / save row, and the inspector on the right,
//! over [`es_editor_scene::SceneModel`]. Over the viewport (packet M17/G6): the selection's tint,
//! its move / turn / size handles, a click to pick, and the policy camera's view in the corner.
//! ①'s Add menu and picture import, cameras and regions drawn as lines, and an include's
//! overrides in the inspector are packet M17/G7's, in this module's children.
//!
//! Drawing only. Which rows there are, what a search keeps, what is hidden, which commands exist,
//! whether one is refused and why, how a size or an angle is shown, what a click selects, where a
//! handle is and what its drag writes — all the scene model's, under test. What is decided here
//! is when a value is handed over: once the person lets go (no pointer button held, no text being
//! typed), as one command, so a drag — of a field or of a handle — is one undo step; a refused
//! value stays in its field with the reason under it and is not tried again until it changes.
//!
//! The children, one job each: `toolbar`, `hierarchy`, `inspector` and `overlay` (the viewport's
//! picking, handles, tint and corner); `add`, `markers` and `overrides` are packet M17/G7's.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use es_assets::esscene::JointKindDoc;
use es_editor_scene::inspect;
use es_editor_scene::view::{project, ray};
use es_editor_scene::{
    BackendKind, Camera, Command, Drag, Entity, Ray, Record, Refusal, RowKind, SceneModel, Tool,
};

use crate::model::scene_view::ScenePreview;
use crate::ui::corner::Corner;

mod add;
mod hierarchy;
mod inspector;
mod markers;
mod overlay;
mod overrides;
mod toolbar;

/// A handle held: the drag, the ray the pointer is on now, and whether a script holds it
/// (`--edit-demo drag`) rather than the pointer.
struct Held {
    drag: Drag,
    now: Ray,
    scripted: bool,
}

/// The selection's triangles for its tint, and the selection and revision they are of.
type Tint = (Option<Entity>, usize, Vec<[[f32; 3]; 3]>);

/// The selected entity's fields as typed so far.
struct Draft {
    entity: Entity,
    /// The model revision it was read at: an edit, an undo or a redo reads it afresh.
    revision: usize,
    record: Record,
    name: String,
    rpy: [f64; 3],
    /// Whether an angle was changed: only then is the quaternion written (its bits kept otherwise).
    rotated: bool,
}

/// ① of one editable project between frames.
pub(crate) struct Author {
    pub(crate) model: SceneModel,
    preview: Option<(usize, Result<Arc<ScenePreview>, String>)>,
    search: String,
    folded: BTreeSet<String>,
    renaming: Option<(Entity, String)>,
    draft: Option<Draft>,
    /// The last refusal and the record it refused.
    refusal: Option<(Refusal, Option<Record>)>,
    saved: Option<Result<(), String>>,
    tool: Tool,
    snap: bool,
    held: Option<Held>,
    tint: Option<Tint>,
    corner: Corner,
    /// The template's bundle (Task IR, Observation IR): the corner's cameras while the project
    /// has no task specification of its own (it trains on the template's documents until G9).
    pub(crate) bundle: Option<[PathBuf; 2]>,
    /// The Add menu's and the overrides' state (packet M17/G7).
    add: add::State,
    /// The task as sentences (packet M17/G8).
    pub(crate) task: crate::ui::sentence::Task,
    /// "Save as template": the name being typed while its dialog is open, and what the last one
    /// did — the folder it wrote, or why not (packet M17/G9).
    pub(crate) save_as: Option<String>,
    pub(crate) saved_as: Option<Result<PathBuf, String>>,
}

fn fold_key(row: &es_editor_scene::Row) -> String {
    format!("{:?}/{}", row.kind, row.name)
}

impl Author {
    /// The project's scene under edit; an include starts folded. `bundle` is the template's Task
    /// IR and Observation IR, for the corner's cameras while the project has no specification.
    pub(crate) fn open(
        root: &Path,
        backends: Vec<BackendKind>,
        bundle: Option<[PathBuf; 2]>,
    ) -> Result<Self, String> {
        let mut model = SceneModel::open(root, backends)?;
        // What `generated/` holds against the saved documents (packet M17/G9).
        model.refresh();
        let folded = (model.rows().iter())
            .filter(|r| r.kind == RowKind::Include)
            .map(fold_key)
            .collect();
        Ok(Self {
            model,
            preview: None,
            search: String::new(),
            folded,
            renaming: None,
            draft: None,
            refusal: None,
            saved: None,
            tool: Tool::Move,
            snap: true,
            held: None,
            tint: None,
            corner: Corner::default(),
            bundle,
            add: add::State::default(),
            task: crate::ui::sentence::Task::default(),
            save_as: None,
            saved_as: None,
        })
    }

    fn apply(&mut self, cmd: &Command, record: Option<Record>) {
        self.refusal = self.model.apply(cmd).err().map(|r| (r, record));
    }

    /// The edited scene as the viewport draws it (what is hidden left out), rebuilt when the
    /// document or what is hidden changes.
    pub(crate) fn preview(&mut self) -> Result<Arc<ScenePreview>, String> {
        let rev = self.model.revision();
        if self.preview.as_ref().is_none_or(|(r, _)| *r != rev) {
            let drawn = self.model.drawn();
            let built = (self.model.scene_file())
                .and_then(|path| ScenePreview::from_scene(path, drawn, rev));
            self.preview = Some((rev, built.map(Arc::new)));
        }
        match &self.preview {
            Some((_, Ok(p))) => Ok(Arc::clone(p)),
            Some((_, Err(e))) => Err(e.clone()),
            None => unreachable!("built above"),
        }
    }

    /// `es-editor --edit-demo`: select the first free body (Shadow Hand's cube) and, from
    /// `edit` on, make it 9 cm and red; `undo` then undoes both (packet M17/G5's captures).
    /// `drag` holds its move handle 5.37 cm along X (snapped: 5 cm) as seen from `camera`, and
    /// `corner` lets that drag go — one command — and selects the front camera, which the corner
    /// then shows (packet M17/G6's captures).
    /// The Add menu's stages (packet M17/G7) are [`Self::add_demo`]'s.
    pub(crate) fn demo(&mut self, stage: &str, camera: &Camera) {
        if self.add_demo(stage, camera) || crate::ui::sentence::demo(self, stage) {
            return;
        }
        let doc = self.model.doc();
        let free = doc.bodies.iter().find(|b| {
            b.joint
                .as_ref()
                .is_some_and(|j| j.kind == JointKindDoc::Free)
                && !b.geoms.is_empty()
        });
        let Some(mut body) = free.cloned() else {
            return;
        };
        let e = Entity::Body(body.name.clone());
        self.model.select(Some(e.clone()));
        if stage == "select" {
            return;
        }
        if stage == "drag" || stage == "corner" {
            let Some(g) = self.model.gizmo(&e, Tool::Move, camera) else {
                return;
            };
            let at = |s: f64| ray(camera, project(camera, g.origin + g.axes[0].scale(s))?);
            let (Some(start), Some(now)) = (at(0.6 * g.size), at(0.6 * g.size + 0.0537)) else {
                return;
            };
            if let Some(drag) = self.model.drag(&e, g.clone(), 0, start) {
                self.held = Some(Held {
                    drag,
                    now,
                    scripted: true,
                });
            }
            if stage == "corner" {
                self.let_go();
                self.model.select(Some(Entity::Camera("front".into())));
            }
            return;
        }
        body.geoms[0].shape = inspect::with_dims(&body.geoms[0].shape, &[0.09, 0.09, 0.09]);
        self.apply(&Command::Set(e.clone(), Record::Body(body.clone())), None);
        body.geoms[0].material = None;
        body.geoms[0].rgba = Some([0.9, 0.15, 0.1, 1.0]);
        self.apply(&Command::Set(e, Record::Body(body)), None);
        if stage == "undo" {
            self.model.undo();
            self.model.undo();
        }
    }
}
