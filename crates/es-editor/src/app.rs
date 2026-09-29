//! The `eframe::App` glue (packet M12/Y10): the editor's state, opening a path, the frame's
//! pumps, and what is kept between sessions. Everything drawn is [`crate::ui`]'s — the shell
//! (menu bar, step bar, dock, status line) in [`crate::ui::shell`] and the five tabs the editor
//! had before the dock, unchanged, in [`crate::ui::advanced`].
//!
//! Thin on purpose. Nothing here decides *what* is drawn — the view-model did that, headless
//! and under test (`docs/design/editor-shell.md` section 2). Which panes exist and where each
//! step puts them is [`crate::model::layout`]'s, the words are [`crate::model::i18n`]'s, the
//! font is [`crate::model::fonts`]'s and the Open dialog is [`crate::model::dialogs`]'s.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;
use egui::Vec2;
use es_compile::bundle;
use es_compile::PolicyBundle;
use es_eval::run_dir::RunDir;
use es_ir::deployment::DeploymentIr;
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{self, Layout};
use es_ir::task::TaskIr;
use es_ir::NodeId;
use es_render::raster::Camera;

use crate::model::edit::{self, EditIr, EditSession};
use crate::model::fonts;
use crate::model::graph_view::LayeredGraph;
use crate::model::home::StartScreen;
use crate::model::i18n::{self, Lang};
use crate::model::image_view::{BeforeAfter, ImagePair};
use crate::model::inspector::Inspector;
use crate::model::launch::{LaunchModel, State as LaunchState};
use crate::model::layout::{self, Arrangement, Docks, Pane};
use crate::model::palette::Palette;
use crate::model::project::Project;
use crate::model::recent::{self, Kind, Recent, Settings};
use crate::model::replay_view::ReplayView;
use crate::model::run_view;
use crate::model::search::Search;
use crate::model::telemetry_view::{Source, TelemetryModel};
use crate::model::template;
use crate::model::watch::Watch;
use crate::model::workflow::{self, Phase, PhaseState, RunFacts};
use crate::ui::advanced::{Drag, SHOWCASE_CAMERA};

/// Telemetry messages drained per frame (spec 23.3 runs the viewer on a budget).
const PUMP_BUDGET: usize = 256;

/// One opened bundle: the four IRs' view-model plus whatever the image tab could make of it.
pub(crate) struct Opened {
    pub(crate) graph: LayeredGraph,
    pub(crate) task: TaskIr,
    pub(crate) observation: ObservationIr,
    pub(crate) pairs: Vec<ImagePair>,
    pub(crate) image_error: Option<String>,
    pub(crate) textures: BTreeMap<String, (egui::TextureHandle, egui::TextureHandle)>,
}

/// The project the step bar follows (packet M12/Y10): its steps' states and which step is
/// open. `watch` is its run as ③ and ④ start, watch and stop it (packet M12/Y12), and
/// `phases` is what its tick said this frame.
pub(crate) struct OpenProject {
    pub(crate) project: Project,
    pub(crate) phases: [PhaseState; 5],
    pub(crate) phase: Phase,
    pub(crate) watch: Watch,
}

pub struct EditorApp {
    pub(crate) path: String,
    pub(crate) status: String,
    pub(crate) opened: Option<Opened>,
    /// An opened run directory (packet M7/E1). A path is one or the other, never both.
    pub(crate) run: Option<RunDir>,
    /// An opened project folder (packet M12/Y10). Opening anything else closes it.
    pub(crate) project: Option<OpenProject>,
    /// The dock's arrangements, one per step and one for anything else (packet M12/Y10).
    pub(crate) docks: Docks,
    /// ⑤ Results between frames (packet M12/Y13).
    pub(crate) results: crate::ui::results::State,
    /// ③'s checks along the way between frames (packet M13/Z5a).
    pub(crate) previews: crate::ui::train::Previews,
    /// ① and ② between frames: the template's scene, read once per project (packet M12/Y15).
    pub(crate) scene: crate::ui::scene::State,
    /// ② between frames: the program being edited and its try (packet M14/Q4).
    pub(crate) teach: crate::ui::teach::State,
    /// Frames of the selected cell's filmstrip, keyed `<cell>#<index>`.
    pub(crate) run_frames: BTreeMap<String, egui::TextureHandle>,
    /// The scene the replay poses (packet M7/E2). A run directory does not carry one, so it
    /// is typed in - the same `--scene` `es video showcase` takes.
    pub(crate) scene_path: String,
    /// Where the run's frames are (`es eval run --frames <dir>`); `<run>/frames` on open.
    pub(crate) frames_path: String,
    pub(crate) replay: Option<ReplayView>,
    /// Which cell `replay` is playing, so switching rows is visible in the panel.
    pub(crate) replay_cell: String,
    /// The last frame the panel rasterised and what it was drawn for, `(tick, camera)`
    /// (packet M7/E8). Dropped whenever `replay` is.
    pub(crate) replay_texture: Option<((usize, Camera), egui::TextureHandle)>,
    pub(crate) camera: Camera,
    pub(crate) telemetry: TelemetryModel,
    pub(crate) source: Source,
    /// The Live pane's Connect field, and the token beside it (spec 25.1). Typed here,
    /// parsed and dialled by `telemetry_view::attach` (packet M7/E4).
    pub(crate) attach_addr: String,
    pub(crate) attach_token: String,
    /// The Results pane's Launch section (packet M7/E5): the command line the editor is about
    /// to start, and the child once it has. Every string it draws is the model's.
    pub(crate) launch: LaunchModel,
    pub(crate) pan: Vec2,
    pub(crate) zoom: f32,
    /// `Some` while the Design graph is in edit mode (spec 23.4 stage 2).
    pub(crate) edit: Option<EditSession>,
    pub(crate) palette: Palette,
    pub(crate) drag: Option<Drag>,
    pub(crate) selected: Option<NodeId>,
    /// The selected node's parameters (packet M7/E3). Rebuilt when [`EditorApp::inspector_key`]
    /// moves, so that what is typed survives a repaint but never an edit.
    pub(crate) inspector: Option<Inspector>,
    pub(crate) inspector_key: (Option<NodeId>, usize),
    pub(crate) search: Search,
    pub(crate) recent: Recent,
    /// A path asked for from inside a pane, opened once the frame's dock is back in place.
    pub(crate) pending_open: Option<String>,
    /// The reader's language and text size (packet M7/E6), persisted beside the recent list.
    /// Nothing else in this file may consult them: what they change is which *table*
    /// [`i18n::t`] reads, never which sentence is written here.
    pub(crate) settings: Settings,
    /// The start screen (packet M12/Y11): the PC check, the templates, the new-project dialog.
    pub(crate) home: StartScreen,
}

impl std::fmt::Debug for EditorApp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorApp")
            .field("path", &self.path)
            .field("opened", &self.opened.is_some())
            .field("project", &self.project.as_ref().map(|p| &p.project.root))
            .finish_non_exhaustive()
    }
}

impl EditorApp {
    /// `source` is where telemetry comes from. It is a closure so the transport can be
    /// swapped without touching the editor: `es_telemetry::transport` plugs in here.
    pub fn new(source: Source) -> Self {
        Self {
            path: String::new(),
            status: i18n::t(Lang::default(), "status.nothing_open").to_owned(),
            opened: None,
            run: None,
            project: None,
            docks: Docks::default(),
            results: crate::ui::results::State::default(),
            previews: crate::ui::train::Previews::default(),
            scene: crate::ui::scene::State::default(),
            teach: crate::ui::teach::State::default(),
            run_frames: BTreeMap::new(),
            scene_path: String::new(),
            frames_path: String::new(),
            replay: None,
            replay_cell: String::new(),
            replay_texture: None,
            camera: SHOWCASE_CAMERA,
            telemetry: TelemetryModel::default(),
            source,
            attach_addr: String::new(),
            attach_token: String::new(),
            launch: LaunchModel::default(),
            pan: Vec2::new(60.0, 40.0),
            zoom: 1.0,
            edit: None,
            palette: Palette::default(),
            drag: None,
            selected: None,
            inspector: None,
            inspector_key: (None, 0),
            search: Search::default(),
            recent: Recent::default(),
            pending_open: None,
            settings: Settings::default(),
            home: StartScreen::load(),
        }
    }

    /// The word `key` names, in the reader's language.
    pub(crate) fn t(&self, key: &'static str) -> &'static str {
        i18n::t(self.settings.lang, key)
    }

    /// [`i18n::fill`] in the reader's language: a template whose `{}` are filled in order.
    pub(crate) fn fill(&self, key: &'static str, args: &[&str]) -> String {
        i18n::fill(self.settings.lang, key, args)
    }

    /// Installs the system CJK face and the chosen text size (packet M7/E6).
    ///
    /// Called once from `main.rs` and again whenever either setting moves. Both answers are
    /// [`fonts`]'s: which files to look in, in what order, where in the fallback chain the
    /// face goes, and how big each style is. What is decided here is nothing at all — even
    /// the sentence a machine with no CJK font gets is a table entry, filled with the list of
    /// paths that were tried.
    pub fn apply_style(&mut self, ctx: &egui::Context) {
        // The status line is the one sentence that outlives the frame that wrote it, so a
        // language switch with nothing open has to re-read it; anything actually opened has
        // already replaced it with news of its own.
        if self.opened.is_none() && self.run.is_none() && self.project.is_none() {
            self.status = self.t("status.nothing_open").to_owned();
        }
        let mut definitions = egui::FontDefinitions::default();
        if let Err(tried) = fonts::install(&mut definitions) {
            self.status = self.fill("status.font_missing", &[&tried.join(", ")]);
        }
        ctx.set_fonts(definitions);
        let styles = fonts::text_styles(self.settings.text_size);
        ctx.all_styles_mut(|style| style.text_styles.clone_from(&styles));
    }

    /// What the previous session left in `eframe::Storage`: the recent list (packet M7/E3),
    /// the display settings (M7/E6) and the dock's arrangements (M12/Y10). Called before
    /// [`Self::with_path`], so a path on the command line joins the list rather than
    /// replacing it.
    #[must_use]
    pub fn with_storage(mut self, storage: Option<&dyn eframe::Storage>) -> Self {
        if let Some(storage) = storage {
            self.recent =
                Recent::from_json(&storage.get_string(recent::RECENT_KEY).unwrap_or_default());
            // Two `get_string`s and no decision: what an empty or foreign value means is
            // `Settings::from_codes`' (packet M7/E6).
            self.settings = Settings::from_codes(
                &storage.get_string(recent::LANG_KEY).unwrap_or_default(),
                &storage
                    .get_string(recent::TEXT_SIZE_KEY)
                    .unwrap_or_default(),
            );
        }
        // Absent or unreadable is every step's default: `Docks::load`'s decision.
        self.docks = Docks::load(storage);
        self
    }

    /// Which arrangement the dock shows now: the open project's step, or the Advanced one.
    pub(crate) fn arrangement(&self) -> Arrangement {
        Arrangement::of(self.project.as_ref().map(|p| p.phase))
    }

    /// Brings `pane` to the front in the arrangement shown now. Asking for a pane is asking for
    /// the dock, so the start screen gives way even with nothing open.
    pub(crate) fn focus(&mut self, pane: Pane) {
        self.home.workspace = true;
        let arrangement = self.arrangement();
        self.docks.focus(arrangement, pane);
    }

    /// Enters or leaves edit mode. Entering starts a session on the opened bundle's Task IR,
    /// seeded with the positions the read-only view already laid out; leaving drops the
    /// session, and with it the undo history.
    pub(crate) fn set_edit_mode(&mut self, on: bool) {
        if !on {
            self.edit = None;
            self.drag = None;
            self.selected = None;
            return;
        }
        let Some(opened) = &self.opened else {
            "open a bundle before editing".clone_into(&mut self.status);
            return;
        };
        let mut layout = sidecar(Path::new(self.path.trim())).unwrap_or_default();
        for node in &opened.graph.layers[crate::model::graph_view::TASK].nodes {
            if let Some(pos) = node.layout {
                layout.positions.entry(node.id).or_insert(pos);
            }
        }
        let graph = EditIr::Task(opened.task.clone());
        edit::fill_missing_positions(&graph, &mut layout, 6);
        let session = EditSession::new(graph, layout);
        self.palette = Palette::from_registries(&session.registries);
        self.status = format!("editing the Task IR: {} nodes", session.graph.nodes().len());
        self.edit = Some(session);
    }

    /// Writes the `.esgraph` and its `.eslayout` next to the loaded path (spec 14.3).
    pub(crate) fn save_edits(&mut self) {
        let Some(session) = &self.edit else { return };
        let (graph_toml, layout_toml) = match session.save() {
            Ok(pair) => pair,
            Err(e) => {
                self.status = format!("save failed: {e}");
                return;
            }
        };
        let (graph_path, layout_path) =
            save_paths(Path::new(self.path.trim()), session.graph.kind());
        self.status = match fs::write(&graph_path, graph_toml)
            .and_then(|()| fs::write(&layout_path, layout_toml))
        {
            Ok(()) => format!(
                "wrote {} and {}",
                graph_path.display(),
                layout_path.display()
            ),
            Err(e) => format!("save failed: {e}"),
        };
    }

    /// What the status bar says before anything is opened — `es-editor --attach`'s result
    /// (packet M7/E4). Applied after [`Self::with_path`], since attaching is the later news.
    #[must_use]
    pub fn with_status(mut self, status: String) -> Self {
        self.status = status;
        self
    }

    /// Open a project, a bundle or a run directory at startup (`es-editor <path>`).
    #[must_use]
    pub fn with_path(mut self, path: &str) -> Self {
        path.clone_into(&mut self.path);
        self.open();
        self
    }

    /// Opens whatever [`recent::classify`] says the path is: a project folder, a run
    /// directory, a `.esb` container, or a directory of the five per-IR documents. The path
    /// field, the File menu, the command line, the recent list and a dropped file all arrive
    /// here (packets M7/E1, M7/E3, M12/Y10).
    pub(crate) fn open(&mut self) {
        let path = PathBuf::from(self.path.trim());
        self.run = None;
        self.run_frames.clear();
        self.replay = None;
        self.replay_cell.clear();
        self.replay_texture = None;
        self.search.set_hits(Vec::new());
        self.project = None;
        match recent::classify(&path) {
            Kind::Project => self.open_project(&path),
            Kind::Run => match RunDir::open(&path) {
                Ok(run) => {
                    self.status = format!("{}: {}", path.display(), run_view::status(&run));
                    self.frames_path = run.frames_root().display().to_string();
                    self.run = Some(run);
                    self.focus(Pane::AdvancedMetrics);
                    self.recent.push(&path);
                }
                Err(e) => self.status = e.to_string(),
            },
            Kind::Bundle | Kind::Documents => self.open_bundle(&path),
        }
        self.prefill_launch();
    }

    /// A project folder: the step bar reads its latest run off disk, the same way the start
    /// screen's recent cards do, and opens at the first step that is not done. What it had
    /// open before is closed - its panes belong to that path, not to this one.
    fn open_project(&mut self, path: &Path) {
        match Project::open(path) {
            Ok(project) => {
                let facts = project.latest_run().map(|run| RunFacts::read(&run));
                let phases = workflow::phases(facts.as_ref(), None);
                self.status = self.fill("shell.project", &[&project.file.name]);
                self.opened = None;
                self.edit = None;
                self.recent.push(path);
                self.project = Some(OpenProject {
                    watch: Watch::new(&project, template::templates_root()),
                    project,
                    phase: layout::start_phase(&phases),
                    phases,
                });
            }
            Err(e) => self.status = e.to_string(),
        }
    }

    fn open_bundle(&mut self, path: &Path) {
        match load(path) {
            Err(e) => {
                self.status = format!("{}: {e}", path.display());
                self.opened = None;
            }
            Ok((task, observation, learning, deployment)) => {
                let mut graph =
                    LayeredGraph::from_bundle(&task, &observation, &learning, &deployment);
                graph.auto_layout();
                // An `.eslayout` sidecar overrides the automatic positions (spec 14.3). It is
                // read, never written: this view is read-only.
                if let Some(layout) = sidecar(path) {
                    graph.apply_layout(&layout);
                }
                let (pairs, image_error) = match BeforeAfter::sample(&observation) {
                    Ok(p) => (p, None),
                    Err(e) => (Vec::new(), Some(e.to_string())),
                };
                self.status = format!(
                    "{}: {} nodes, {} cross-IR edges, {} diagnostics",
                    path.display(),
                    graph.layers.iter().map(|l| l.nodes.len()).sum::<usize>(),
                    graph.cross_edges.len(),
                    graph.diagnostics.len(),
                );
                self.edit = None;
                self.recent.push(path);
                self.opened = Some(Opened {
                    graph,
                    task,
                    observation,
                    pairs,
                    image_error,
                    textures: BTreeMap::new(),
                });
            }
        }
    }

    /// Hands the session's paths to the Launch section (packet M7/E5). *Which* flag each one
    /// fills, and whether it may overwrite what is already typed, is
    /// [`LaunchModel::prefill`]'s decision and not this file's.
    fn prefill_launch(&mut self) {
        let bundle = self
            .opened
            .is_some()
            .then(|| PathBuf::from(self.path.trim()));
        let run_dir = self.run.as_ref().map(|run| run.dir.clone());
        self.launch.prefill(
            bundle.as_deref(),
            run_dir.as_deref(),
            &self.scene_path,
            &self.attach_addr,
        );
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.telemetry.pump(&mut self.source, PUMP_BUDGET);
        // A launched child's lines and its exit code (packet M7/E5). Once a frame, never
        // blocking: the reader threads are what touch the pipes.
        self.launch.poll();
        // The status bar said "running" when the run started; once it is not, say so there
        // too rather than leaving a finished run looking alive.
        if !matches!(self.launch.state(), LaunchState::Running { .. })
            && self.status.contains("running: pid")
        {
            self.status = self.launch.status_line();
        }
        // The dial's answer, if it arrived this frame (packet M7/E5, un-blocked after E6).
        if let Some(attached) = self.launch.take_attached() {
            match attached {
                Ok(source) => {
                    self.telemetry = TelemetryModel::default();
                    self.source = source;
                    self.status = format!("started and attached: {}", self.launch.status_line());
                }
                Err(e) => self.status = e,
            }
        }
        // ③ and ④: the open project's run (packet M12/Y12).
        crate::ui::train::tick(self, ctx);
        // ②: a try's own child (packet M14/Q4).
        crate::ui::teach::tick(self);

        // A dropped file goes through the same function the path field does (packet M7/E3):
        // one way in means one set of errors out.
        if let Some(path) = ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone())) {
            self.path = path.display().to_string();
            self.open();
        }

        crate::ui::shell::draw(self, ctx);

        // Telemetry is a live stream; repaint even when no input arrives.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }

    /// The recent list, the display settings and the dock's arrangements. `eframe` keeps them
    /// in `%APPDATA%/Electric Sheep editor/data/app.ron` on Windows and in
    /// `~/.local/share/electricsheepeditor/app.ron` on Linux.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(recent::RECENT_KEY, self.recent.to_json());
        let (lang, text_size) = self.settings.codes();
        storage.set_string(recent::LANG_KEY, lang.to_owned());
        storage.set_string(recent::TEXT_SIZE_KEY, text_size.to_owned());
        self.docks.save(storage);
    }
}

// --- loading ---------------------------------------------------------------------------------

type Bundle = (TaskIr, ObservationIr, LearningGraph, DeploymentIr);

/// A `.esb` container (spec 9.6) or a directory holding the per-IR TOML files under the names
/// the bundle format already fixes (spec 14.3: `task.toml`, `observation.toml`, ...).
fn load(path: &Path) -> Result<Bundle, String> {
    if path.extension().is_some_and(|e| e == "esb") {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let b = PolicyBundle::open(&bytes).map_err(|e| e.to_string())?;
        return Ok((b.task, b.observation, b.learning, b.deployment));
    }
    let read = |name: &str| fs::read_to_string(path.join(name)).map_err(|e| format!("{name}: {e}"));
    let task = serial::task_from_toml(&read(bundle::TASK)?).map_err(|e| e.to_string())?;
    let observation =
        serial::observation_from_toml(&read(bundle::OBSERVATION)?).map_err(|e| e.to_string())?;
    let learning =
        serial::learning_from_toml(&read(bundle::LEARNING)?).map_err(|e| e.to_string())?;
    let deployment =
        serial::deployment_from_toml(&read(bundle::DEPLOYMENT)?).map_err(|e| e.to_string())?;
    Ok((task, observation, learning, deployment))
}

/// `<dir>/layout.eslayout`, if it is there and parses. A missing or broken sidecar is not an
/// error: layout is decoration, and the graph is readable without it (spec 4.2 rule 7).
fn sidecar(path: &Path) -> Option<Layout> {
    let file = if path.is_dir() {
        path.join("layout.eslayout")
    } else {
        path.with_extension("eslayout")
    };
    serial::parse_toml(&fs::read_to_string(file).ok()?).ok()
}

/// Where `Save` writes: `<stem>.esgraph` and `<stem>.eslayout` beside the loaded path, or
/// inside it when a directory of per-IR TOML files was opened (spec 14.3).
fn save_paths(path: &Path, kind: es_ir::serial::IrKind) -> (PathBuf, PathBuf) {
    let stem = if path.is_dir() {
        path.join(format!("{kind:?}").to_lowercase())
    } else {
        path.to_path_buf()
    };
    (
        stem.with_extension("esgraph"),
        stem.with_extension("eslayout"),
    )
}
