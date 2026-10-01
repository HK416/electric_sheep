//! [`SceneModel`]: an editable project's scene document under edit — the selection, the undo /
//! redo stack, the dirty mark, the save and the regeneration of the project's documents.
//!
//! **Validation policy: a refused command is not applied.** Every command runs on a copy of the
//! documents, the copy is checked ([`crate::check`]), and only a copy that expands replaces the
//! documents and becomes an undo step. So the document, the viewport and every step on the undo
//! stack are always a scene that expands and maps onto the project's backends, and the physics
//! preview and a save never meet a broken one; the person's typed value is not lost, it stays in
//! the inspector's field with the reason beside it.
//!
//! Undo steps hold the documents before and after, whole: a document is a few kilobytes, and a
//! snapshot cannot be undone wrongly.
// ponytail: whole-document snapshots per step; inverse commands if documents grow to megabytes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use es_assets::esscene::{EsScene, Include};
use es_assets::scene::SceneDesc;
use es_ir::task::TaskIr;
use es_physics_backend::BackendKind;
use es_script::spec::{compile_task, generate, TaskSpec};

use crate::check::{check, Refusal};
use crate::command::{self, apply, Command, Docs, Entity, Record};
use crate::tree::{self, Contents, Row};
use crate::{GENERATED_DIR, PREVIEW_FILE, SCENE_FILE, SPEC_FILE};

/// What the last regeneration of the project's documents did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Regen {
    /// Not run since the project was opened.
    NotYet,
    /// The project has no `task.estask`: there is nothing to generate from.
    NoSpec,
    /// These files, written under `generated/`.
    Written(Vec<String>),
    /// Why not; `generated/` was emptied, so nothing stale is left to read.
    Failed(String),
}

#[derive(Debug)]
struct Step {
    before: Docs,
    after: Docs,
}

/// One editable project's scene under edit.
#[derive(Debug)]
pub struct SceneModel {
    root: PathBuf,
    backends: Vec<BackendKind>,
    docs: Docs,
    saved: Docs,
    scene: Arc<SceneDesc>,
    /// What each include brings, and the includes it was read for.
    includes: (Vec<Include>, Vec<Contents>),
    undo: Vec<Step>,
    redo: Vec<Step>,
    selection: Option<Entity>,
    hidden: BTreeSet<Entity>,
    /// Counts every change to what the viewport draws: an edit, an undo, a hidden thing.
    revision: usize,
    /// The revision [`PREVIEW_FILE`] was written for.
    preview: Option<usize>,
    generated: Regen,
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes beside `path` and renames over it: a reader sees the old file or the new one.
fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(fail)?;
    std::fs::rename(&tmp, path).map_err(fail)
}

impl SceneModel {
    /// Opens the project at `root`: its `scene.esscene` and, if there is one, `task.estask`.
    /// `backends` are those the project runs on; every command is checked against each one's
    /// mapping report.
    pub fn open(root: &Path, backends: Vec<BackendKind>) -> Result<Self, String> {
        let scene = EsScene::from_toml(&read(&root.join(SCENE_FILE))?)
            .map_err(|e| format!("{SCENE_FILE}: {e}"))?;
        let spec_path = root.join(SPEC_FILE);
        let spec = if spec_path.is_file() {
            let spec = TaskSpec::from_toml(&read(&spec_path)?);
            Some(spec.map_err(|e| format!("{SPEC_FILE}: {e}"))?)
        } else {
            None
        };
        let desc = check(&scene, root, &backends).map_err(|r| {
            let args = r.args.join(", ");
            format!("{SCENE_FILE}: {}: {args}", r.field)
        })?;
        let includes = (scene.includes.clone(), tree::contents(&scene, root));
        let docs = Docs { scene, spec };
        Ok(Self {
            root: root.to_path_buf(),
            backends,
            saved: docs.clone(),
            docs,
            scene: Arc::new(desc),
            includes,
            undo: Vec::new(),
            redo: Vec::new(),
            selection: None,
            hidden: BTreeSet::new(),
            revision: 0,
            preview: None,
            generated: Regen::NotYet,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn doc(&self) -> &EsScene {
        &self.docs.scene
    }

    pub fn spec(&self) -> Option<&TaskSpec> {
        self.docs.spec.as_ref()
    }

    /// The document's expansion: what the viewport draws and the physics simulates.
    pub fn scene(&self) -> &Arc<SceneDesc> {
        &self.scene
    }

    pub fn revision(&self) -> usize {
        self.revision
    }

    pub fn selection(&self) -> Option<&Entity> {
        self.selection.as_ref()
    }

    pub fn select(&mut self, e: Option<Entity>) {
        self.selection = e;
    }

    /// The selected entity's fields.
    pub fn record(&self, e: &Entity) -> Option<Record> {
        command::get(&self.docs.scene, e)
    }

    pub fn dirty(&self) -> bool {
        self.docs != self.saved
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn generated(&self) -> &Regen {
        &self.generated
    }

    /// Whether `name` is in use anywhere in the scene: a body, joint, geom, site or camera of
    /// the expansion (an included one too), or an entity of the document.
    fn taken(&self, name: &str) -> bool {
        let s = &self.scene;
        let d = &self.docs.scene;
        s.bodies
            .iter()
            .any(|b| b.name == name || b.sites.iter().any(|x| x.name == name))
            || s.joints.iter().any(|j| j.name == name)
            || s.cameras.iter().any(|c| c.name == name)
            || d.includes.iter().any(|i| i.name == name)
            || d.lights.iter().any(|l| l.name == name)
            || d.regions.iter().any(|r| r.name == name)
    }

    /// The bodies `e` may hang from, in scene order: every body but the world and, for a body,
    /// itself and what hangs from it.
    pub fn parents(&self, e: &Entity) -> Vec<String> {
        let own = match e {
            Entity::Body(n) => command::subtree(&self.docs.scene, n),
            _ => BTreeSet::new(),
        };
        (self.scene.bodies.iter().skip(1))
            .map(|b| b.name.clone())
            .filter(|n| !own.contains(n))
            .collect()
    }

    /// `base`, or `base_<n>` when it is taken: the name a new entity is offered.
    pub fn unique(&self, base: &str) -> String {
        command::unique(base, &|n| self.taken(n))
    }

    /// Applies `cmd` if the documents it leaves expand and map onto every backend, and the task
    /// specification still compiles on the scene (packet M17/G8: it always does, or there is
    /// none); refuses it otherwise and changes nothing. A command that changes nothing is no step.
    pub fn apply(&mut self, cmd: &Command) -> Result<(), Refusal> {
        let mut next = self.docs.clone();
        let select = apply(&mut next, cmd, &|n| self.taken(n))?;
        if next == self.docs {
            return Ok(());
        }
        let scene = if next.scene == self.docs.scene {
            Arc::clone(&self.scene)
        } else {
            Arc::new(check(&next.scene, &self.root, &self.backends)?)
        };
        if let Some(spec) = &next.spec {
            if let Err(why) = self.compiles(spec, &next.scene) {
                // A scene edit is refused for breaking the specification, not for one that was
                // already broken (a hand-edited file): that one is the sentences' to mend.
                let current = self.docs.clone();
                let was_fine = (current.spec.as_ref())
                    .is_some_and(|s| self.compiles(s, &current.scene).is_ok());
                if matches!(cmd, Command::Spec(_)) || was_fine {
                    return Err(why);
                }
            }
        }
        let before = std::mem::replace(&mut self.docs, next);
        self.undo.push(Step {
            before,
            after: self.docs.clone(),
        });
        self.redo.clear();
        self.settle(scene);
        if !matches!(cmd, Command::Spec(_)) {
            self.selection = select;
        }
        Ok(())
    }

    /// `spec` compiled on the scene `doc`, read from a file as `compile_task` reads one: the
    /// saved document when `doc` is it, else `doc` written to [`PREVIEW_FILE`].
    pub(crate) fn compiles(&mut self, spec: &TaskSpec, doc: &EsScene) -> Result<TaskIr, Refusal> {
        let other = |why: String| Refusal::new("scene", crate::check::OTHER, vec![why]);
        let file = if *doc == self.saved.scene {
            SCENE_FILE
        } else {
            let text = doc.to_toml().map_err(|e| other(e.to_string()))?;
            write_atomic(&self.root.join(PREVIEW_FILE), &text).map_err(other)?;
            self.preview = (*doc == self.docs.scene).then_some(self.revision);
            PREVIEW_FILE
        };
        let mut spec = spec.clone();
        file.clone_into(&mut spec.scene);
        compile_task(&spec, &self.root).map_err(crate::sentence::refusal)
    }

    /// Back one step; `false` with nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some(step) = self.undo.pop() else {
            return false;
        };
        self.docs = step.before.clone();
        self.redo.push(step);
        self.reexpand();
        true
    }

    /// Forward one undone step; `false` with nothing to redo.
    pub fn redo(&mut self) -> bool {
        let Some(step) = self.redo.pop() else {
            return false;
        };
        self.docs = step.after.clone();
        self.undo.push(step);
        self.reexpand();
        true
    }

    /// The documents of an undo step expanded once already; they still do.
    fn reexpand(&mut self) {
        match check(&self.docs.scene, &self.root, &self.backends) {
            Ok(scene) => self.settle(Arc::new(scene)),
            // A file the scene names changed on disk since: keep drawing the last scene.
            Err(_) => self.revision += 1,
        }
        let doc = &self.docs.scene;
        if self
            .selection
            .as_ref()
            .is_some_and(|e| command::get(doc, e).is_none())
        {
            self.selection = None;
        }
    }

    fn settle(&mut self, scene: Arc<SceneDesc>) {
        if self.includes.0 != self.docs.scene.includes {
            let read = tree::contents(&self.docs.scene, &self.root);
            self.includes = (self.docs.scene.includes.clone(), read);
        }
        self.scene = scene;
        self.revision += 1;
    }

    /// The hierarchy's rows.
    pub fn rows(&self) -> Vec<Row> {
        tree::rows(&self.docs.scene, &self.includes.1)
    }

    /// What each include brings, in include order.
    pub(crate) fn contents(&self) -> &[Contents] {
        &self.includes.1
    }

    pub fn hidden(&self, e: &Entity) -> bool {
        self.hidden.contains(e)
    }

    /// Shows a hidden entity or hides a shown one; the document does not change.
    pub fn toggle_hidden(&mut self, e: &Entity) {
        if !self.hidden.remove(e) {
            self.hidden.insert(e.clone());
        }
        self.revision += 1;
    }

    /// What the viewport draws: the scene without what is hidden.
    pub fn drawn(&self) -> SceneDesc {
        let hidden = (self.hidden.iter())
            .filter(|e| command::get(&self.docs.scene, e).is_some())
            .cloned()
            .collect();
        tree::drawn(&self.scene, &self.docs.scene, &self.includes.1, &hidden)
    }

    /// The file holding the scene as edited, for what reads a scene from a file: the saved
    /// document when nothing is unsaved, else [`PREVIEW_FILE`] beside it, written once per
    /// revision.
    pub fn scene_file(&mut self) -> Result<PathBuf, String> {
        if !self.dirty() {
            return Ok(self.root.join(SCENE_FILE));
        }
        let path = self.root.join(PREVIEW_FILE);
        if self.preview != Some(self.revision) {
            let text = self.docs.scene.to_toml().map_err(|e| e.to_string())?;
            write_atomic(&path, &text)?;
            self.preview = Some(self.revision);
        }
        Ok(path)
    }

    /// Writes `scene.esscene` (and `task.estask`) and regenerates the project's documents.
    pub fn save(&mut self) -> Result<(), String> {
        let scene = self.docs.scene.to_toml().map_err(|e| e.to_string())?;
        let spec = (self.docs.spec.as_ref())
            .map(|s| s.to_toml().map_err(|e| e.to_string()))
            .transpose()?;
        write_atomic(&self.root.join(SCENE_FILE), &scene)?;
        if let Some(spec) = spec {
            write_atomic(&self.root.join(SPEC_FILE), &spec)?;
        }
        self.saved = self.docs.clone();
        let _ = std::fs::remove_file(self.root.join(PREVIEW_FILE));
        self.preview = None;
        self.regenerate();
        Ok(())
    }

    /// `generate` (G3b) from the saved documents into `generated/`, emptied first: after a
    /// failure there is no stale document left to read.
    pub fn regenerate(&mut self) {
        let dir = self.root.join(GENERATED_DIR);
        let _ = std::fs::remove_dir_all(&dir);
        let Some(spec) = &self.saved.spec else {
            self.generated = Regen::NoSpec;
            return;
        };
        let written = generate(spec, &self.root, GENERATED_DIR, SPEC_FILE)
            .map_err(|e| e.to_string())
            .and_then(|docs| {
                std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let mut files = Vec::new();
                for d in docs {
                    write_atomic(&dir.join(&d.file), &d.text)?;
                    files.push(d.file);
                }
                Ok(files)
            });
        self.generated = match written {
            Ok(files) => Regen::Written(files),
            Err(why) => {
                let _ = std::fs::remove_dir_all(&dir);
                Regen::Failed(why)
            }
        };
    }
}
