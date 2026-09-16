//! The egui shell: four tabs over [`crate::model`] (spec 23.2, spec 23.3).
//!
//! Thin on purpose. Nothing here decides *what* is drawn — the view-model did that, headless
//! and under test. This file turns positions into rectangles, and it is the only part CI
//! merely compiles rather than runs, because running it needs a display.
//!
//! The **Graph** tab has two modes. Read-only (spec 23.4 stage 1) draws all four IRs stacked
//! and moves nothing. `Edit` (stage 2) drives one [`EditSession`] over the Task IR: every
//! gesture becomes one [`Edit`], and nothing else. A drag writes a position into the
//! `.eslayout` sidecar, never into an IR (spec 4.2 rule 7).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use es_compile::bundle;
use es_compile::PolicyBundle;
use es_ir::deployment::DeploymentIr;
use es_ir::graph::PortRef;
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{self, Layout};
use es_ir::task::TaskIr;
use es_ir::NodeId;

use crate::model::edit::{self, Edit, EditIr, EditSession};
use crate::model::graph_view::{CrossEdge, LayerView, LayeredGraph, NodeView};
use crate::model::image_view::{BeforeAfter, ImagePair, Rgb8Image};
use crate::model::palette::Palette;
use crate::model::replay_view::{Camera, Projected, ReplayView};
use crate::model::run_view::{Bucket, RunView};
use crate::model::telemetry_view::{Source, TelemetryModel};

const NODE_W: f32 = 178.0;
const NODE_H: f32 = 40.0;
/// Click radius of a port, in graph units.
const PORT_R: f32 = 7.0;
/// Telemetry messages drained per frame (spec 23.3 runs the viewer on a budget).
const PUMP_BUDGET: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Graph,
    Run,
    Telemetry,
    Images,
    Diagnostics,
}

/// One opened bundle: the four IRs' view-model plus whatever the image tab could make of it.
struct Opened {
    graph: LayeredGraph,
    task: TaskIr,
    observation: ObservationIr,
    pairs: Vec<ImagePair>,
    image_error: Option<String>,
    textures: BTreeMap<String, (egui::TextureHandle, egui::TextureHandle)>,
}

/// What the pointer is doing between press and release.
#[derive(Clone, Debug)]
enum Drag {
    /// Moving a node. `origin` is where it was when the drag started, so the single
    /// `Edit::MoveNode` pushed on release has the correct state to undo to.
    Node {
        id: NodeId,
        origin: [f32; 2],
        grab: Vec2,
    },
    /// Pulling a wire out of an output port.
    Link {
        from: PortRef,
    },
    Pan,
}

pub struct EditorApp {
    tab: Tab,
    path: String,
    status: String,
    opened: Option<Opened>,
    /// An opened run directory (packet M7/E1). A path is one or the other, never both.
    run: Option<RunView>,
    /// Frames of the selected cell's filmstrip, keyed `<cell>#<index>`.
    run_frames: BTreeMap<String, egui::TextureHandle>,
    /// The scene the replay poses (packet M7/E2). A run directory does not carry one, so it
    /// is typed in - the same `--scene` `es video showcase` takes.
    scene_path: String,
    /// Where the run's frames are (`es eval run --frames <dir>`); `<run>/frames` on open.
    frames_path: String,
    replay: Option<ReplayView>,
    /// Which cell `replay` is playing, so switching rows is visible in the panel.
    replay_cell: String,
    camera: Camera,
    telemetry: TelemetryModel,
    source: Source,
    pan: Vec2,
    zoom: f32,
    /// `Some` while the Graph tab is in edit mode (spec 23.4 stage 2).
    edit: Option<EditSession>,
    palette: Palette,
    drag: Option<Drag>,
    selected: Option<NodeId>,
}

impl std::fmt::Debug for EditorApp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EditorApp")
            .field("tab", &self.tab)
            .field("path", &self.path)
            .field("opened", &self.opened.is_some())
            .finish_non_exhaustive()
    }
}

impl EditorApp {
    /// `source` is where telemetry comes from. It is a closure so the transport can be
    /// swapped without touching the editor: `es_telemetry::transport` plugs in here.
    pub fn new(source: Source) -> Self {
        Self {
            tab: Tab::Graph,
            path: String::new(),
            status: "no bundle open".to_owned(),
            opened: None,
            run: None,
            run_frames: BTreeMap::new(),
            scene_path: String::new(),
            frames_path: String::new(),
            replay: None,
            replay_cell: String::new(),
            camera: SHOWCASE_CAMERA,
            telemetry: TelemetryModel::default(),
            source,
            pan: Vec2::new(60.0, 40.0),
            zoom: 1.0,
            edit: None,
            palette: Palette::default(),
            drag: None,
            selected: None,
        }
    }

    /// Enters or leaves edit mode. Entering starts a session on the opened bundle's Task IR,
    /// seeded with the positions the read-only view already laid out; leaving drops the
    /// session, and with it the undo history.
    fn set_edit_mode(&mut self, on: bool) {
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
    fn save_edits(&mut self) {
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

    /// Open a bundle or a run directory at startup (`es-editor <bundle.esb|run-dir>`).
    #[must_use]
    pub fn with_path(mut self, path: &str) -> Self {
        path.clone_into(&mut self.path);
        self.open();
        self
    }

    /// A directory that holds `report.json` is a finished run (spec 10.5), anything else is a
    /// bundle: the two are told apart by what is on disk, not by a flag (packet M7/E1).
    fn open(&mut self) {
        let path = PathBuf::from(self.path.trim());
        self.run = None;
        self.run_frames.clear();
        self.replay = None;
        self.replay_cell.clear();
        if RunView::is_run_dir(&path) {
            match RunView::open(&path) {
                Ok(run) => {
                    self.status = format!("{}: {}", path.display(), run.status);
                    self.frames_path = run.frames_root().display().to_string();
                    self.run = Some(run);
                    self.tab = Tab::Run;
                }
                Err(e) => self.status = e.to_string(),
            }
            return;
        }
        match load(&path) {
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
                if let Some(layout) = sidecar(&path) {
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
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.telemetry.pump(&mut self.source, PUMP_BUDGET);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("File");
                ui.add(
                    egui::TextEdit::singleline(&mut self.path)
                        .hint_text("bundle.esb, a directory of the five .toml files, or a run")
                        .desired_width(380.0),
                );
                if ui.button("Open...").clicked() {
                    self.open();
                }
                ui.separator();
                for (tab, name) in [
                    (Tab::Graph, "Graph"),
                    (Tab::Run, "Run"),
                    (Tab::Telemetry, "Telemetry"),
                    (Tab::Images, "Images"),
                    (Tab::Diagnostics, "Diagnostics"),
                ] {
                    ui.selectable_value(&mut self.tab, tab, name);
                }
                ui.separator();
                let mut editing = self.edit.is_some();
                if ui.toggle_value(&mut editing, "Edit").changed() {
                    self.tab = Tab::Graph;
                    self.set_edit_mode(editing);
                }
                if self.edit.is_some() {
                    if ui.button("Save").clicked() {
                        self.save_edits();
                    }
                    let (undo, redo) = self
                        .edit
                        .as_ref()
                        .map_or((false, false), |s| (s.can_undo(), s.can_redo()));
                    if ui.add_enabled(undo, egui::Button::new("Undo")).clicked() {
                        if let Some(s) = self.edit.as_mut() {
                            s.undo();
                        }
                    }
                    if ui.add_enabled(redo, egui::Button::new("Redo")).clicked() {
                        if let Some(s) = self.edit.as_mut() {
                            s.redo();
                        }
                    }
                }
            });
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.separator();
                ui.label(format!("{} telemetry messages", self.telemetry.received));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Graph => self.graph_tab(ui),
            Tab::Run => self.run_tab(ui),
            Tab::Telemetry => self.telemetry_tab(ui),
            Tab::Images => self.images_tab(ui),
            Tab::Diagnostics => self.diagnostics_tab(ui),
        });

        // Telemetry is a live stream; repaint even when no input arrives.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

impl EditorApp {
    fn graph_tab(&mut self, ui: &mut egui::Ui) {
        if self.edit.is_some() {
            self.edit_canvas(ui);
            return;
        }
        let Some(opened) = &self.opened else {
            ui.label("Open a bundle to see the layered graph (spec 23.2).");
            return;
        };
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.hovered() {
            let z = ui.input(|i| i.zoom_delta() * (1.0 + i.smooth_scroll_delta.y * 0.001));
            self.zoom = (self.zoom * z).clamp(0.2, 4.0);
        }
        let origin = response.rect.min + self.pan;
        let zoom = self.zoom;
        let at = |p: [f32; 2]| origin + Vec2::new(p[0], p[1]) * zoom;
        let size = Vec2::new(NODE_W, NODE_H) * zoom;

        for layer in &opened.graph.layers {
            paint_layer(&painter, layer, &at, size, zoom);
        }
        for edge in &opened.graph.cross_edges {
            paint_cross_edge(&painter, &opened.graph, edge, &at, size, zoom);
        }
    }

    /// Edit mode (spec 23.4 stage 2). Every branch below ends in exactly one
    /// [`EditSession::apply`], [`EditSession::undo`] or [`EditSession::redo`] call - this
    /// function decides nothing else, which is what keeps the untested half thin.
    fn edit_canvas(&mut self, ui: &mut egui::Ui) {
        let Self {
            edit: Some(session),
            palette,
            pan,
            zoom,
            drag,
            selected,
            status,
            ..
        } = self
        else {
            return;
        };
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        if response.hovered() {
            *zoom = (*zoom * ui.input(eframe::egui::InputState::zoom_delta)).clamp(0.2, 4.0);
        }
        let origin = response.rect.min + *pan;
        let z = *zoom;
        let to_graph = move |p: Pos2| [(p.x - origin.x) / z, (p.y - origin.y) / z];

        // An owned snapshot of the geometry the pointer is tested against, so that no borrow
        // of `session` is alive across the one `apply` below.
        let mut view = CanvasView::of(session, origin, z);

        let mut pending: Option<Edit> = None;
        if let (true, Some(pos)) = (response.drag_started(), response.interact_pointer_pos()) {
            let started = view.start_drag(pos);
            if let Drag::Node { id, .. } = &started {
                *selected = Some(*id);
            }
            *drag = Some(started);
        }
        match drag.clone() {
            Some(Drag::Pan) => *pan += response.drag_delta(),
            Some(Drag::Node {
                id,
                grab,
                origin: was,
            }) => {
                if let Some(pos) = response.interact_pointer_pos() {
                    let moved = to_graph(pos - grab);
                    if response.drag_stopped() {
                        // Put the node back where the gesture began and record one edit, so
                        // undo returns to the position before the whole drag.
                        session.layout.positions.insert(id, was);
                        pending = Some(Edit::MoveNode {
                            node: id,
                            pos: moved,
                        });
                    } else {
                        session.layout.positions.insert(id, moved);
                        view.positions.insert(id, moved);
                    }
                }
            }
            Some(Drag::Link { from }) => {
                if let (Some(pos), Some(start)) =
                    (response.interact_pointer_pos(), view.port_pos(&from, false))
                {
                    bezier(&painter, start, pos, true, LINK, z);
                    if response.drag_stopped() {
                        if let Some(to) = view.port_at(pos, true) {
                            pending = Some(Edit::Connect { from, to });
                        }
                    }
                }
            }
            None => {}
        }
        if response.drag_stopped() {
            *drag = None;
        }

        ui.input(|i| {
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                if let Some(node) = *selected {
                    pending = Some(Edit::RemoveNode { node });
                }
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Z) {
                session.undo();
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Y) {
                session.redo();
            }
        });

        let ir = session.graph.kind();
        let menu_at = response.interact_pointer_pos().map(to_graph);
        response.context_menu(|ui| {
            ui.label("Add node");
            for (category, entries) in palette.by_category() {
                if entries.first().is_none_or(|e| e.ir != ir) {
                    continue;
                }
                ui.menu_button(category, |ui| {
                    for entry in entries {
                        if ui.button(&entry.kind).clicked() {
                            pending = Some(Edit::AddNode {
                                kind: entry.kind.clone(),
                                params: entry.defaults(),
                                pos: menu_at.unwrap_or([0.0, 0.0]),
                            });
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    }
                });
            }
        });

        if let Some(edit) = pending {
            match session.apply(edit) {
                Ok(()) => *status = format!("{} edits", session.history().len()),
                Err(diags) => {
                    *status = diags.first().map_or_else(
                        || "edit refused".to_owned(),
                        |d| format!("refused: {} {}", d.code, d.message),
                    );
                }
            }
            view = CanvasView::of(session, origin, z);
        }
        if selected.is_some_and(|id| !session.graph.contains(id)) {
            *selected = None;
        }
        view.paint(&painter, *selected);
    }

    /// The Run tab (packet M7/E1): the cell table, the acceptance verdict, and for the
    /// selected cell its Safety Plane timeline and a filmstrip. Every number, every order and
    /// every decoded byte is [`RunView`]'s; this turns them into widgets.
    fn run_tab(&mut self, ui: &mut egui::Ui) {
        if self.run.is_none() {
            ui.label(
                "Open a run directory - one holding report.json - to see its cells (spec 10.5).",
            );
            return;
        }
        // The replay of the selected cell shares the tab (packet M7/E2): the table picks the
        // episode, the panel plays it. Nearly half the window by default, so the canvas is
        // there without dragging the separator first; still resizable either way.
        let height = (ui.available_height() * REPLAY_PANEL_FRACTION).max(REPLAY_PANEL_MIN);
        egui::TopBottomPanel::bottom("replay")
            .resizable(true)
            .min_height(REPLAY_PANEL_MIN)
            .default_height(height)
            .show_inside(ui, |ui| self.replay_panel(ui));
        egui::CentralPanel::default().show_inside(ui, |ui| self.run_table(ui));
    }

    fn run_table(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Self {
            run: Some(run),
            run_frames,
            ..
        } = self
        else {
            return;
        };
        let columns = run.columns();
        let mut sort = None;
        let mut select = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("run-cells").striped(true).show(ui, |ui| {
                for (i, name) in columns.iter().enumerate() {
                    if ui.button(name).clicked() {
                        sort = Some(i);
                    }
                }
                ui.label("traj");
                ui.label("frames");
                ui.end_row();
                let selected = run.selected_cell().map(|c| c.name.clone());
                for row in run.cells() {
                    let is_selected = selected.as_deref() == Some(row.name.as_str());
                    if ui.selectable_label(is_selected, &row.name).clicked() {
                        select = Some(row.name.clone());
                    }
                    ui.label(&row.suite);
                    ui.label(row.seed.map_or_else(|| "--".to_owned(), |s| s.to_string()));
                    for column in columns.iter().skip(3) {
                        ui.label(row.metrics.get(column).map_or_else(dash, metric_text));
                    }
                    ui.label(if row.has_traj { "yes" } else { "--" });
                    ui.label(row.frames.to_string());
                    ui.end_row();
                }
            });

            ui.separator();
            ui.heading(if run.report.passed {
                "Acceptance: passed (spec 10.2)"
            } else {
                "Acceptance: failed (spec 10.2)"
            });
            for line in run.acceptance() {
                let (text, colour) = acceptance_row(line);
                ui.colored_label(colour, text);
            }

            let Some(cell) = run.selected_cell().map(|c| c.name.clone()) else {
                ui.separator();
                ui.label("Select a cell for its Safety Plane timeline and frames (spec 23.3).");
                return;
            };
            ui.separator();
            let timeline = run.timeline(&cell);
            ui.heading(format!("{cell}: {} tick(s)", timeline.rows.len()));
            // One column per ~4 px of the strip; the model folds the ticks into them.
            let n = (ui.available_width() / 4.0) as usize;
            paint_timeline(ui, &timeline.buckets(n));
            for kind in timeline.kind_rows() {
                ui.label(kind.label());
            }

            ui.separator();
            ui.heading("Frames");
            ui.horizontal(|ui| {
                for index in run.filmstrip(&cell, FILMSTRIP) {
                    let key = format!("{cell}#{index}");
                    let texture = run_frames.entry(key.clone()).or_insert_with(|| {
                        let image = run.frame(&cell, index).unwrap_or(Rgb8Image {
                            width: 1,
                            height: 1,
                            data: vec![0, 0, 0],
                        });
                        rgb_texture(&ctx, &key, &image)
                    });
                    ui.vertical(|ui| {
                        ui.label(format!("{index}"));
                        let scale = (160.0 / texture.size_vec2().x).max(1.0);
                        ui.image(egui::load::SizedTexture::new(
                            texture.id(),
                            texture.size_vec2() * scale,
                        ));
                    });
                }
            });
        });
        if let Some(column) = sort {
            run.sort_by(column);
        }
        if let Some(name) = select {
            run.select(&name);
        }
    }

    /// The Replay panel (packet M7/E2): the selected cell's `.estraj`, posed and projected by
    /// [`ReplayView`] and painted as one mesh. The gestures map to the model's pure camera
    /// functions and to `advance`; nothing is decided here.
    fn replay_panel(&mut self, ui: &mut egui::Ui) {
        let dt = f64::from(ui.input(|i| i.stable_dt));
        let Self {
            run,
            replay,
            replay_cell,
            scene_path,
            frames_path,
            camera,
            status,
            ..
        } = self;
        let selected = run
            .as_ref()
            .and_then(RunView::selected_cell)
            .filter(|c| c.has_traj)
            .map(|c| c.name.clone());

        ui.horizontal(|ui| {
            ui.label("Scene");
            ui.add(
                egui::TextEdit::singleline(scene_path)
                    .hint_text("the run's scene: .xml (MJCF) or .urdf")
                    .desired_width(220.0),
            );
            // `es eval run --frames <dir>` writes wherever it was told, which is usually a
            // sibling of the run directory; the model re-scans when this is applied.
            ui.label("Frames");
            let field = ui.add(
                egui::TextEdit::singleline(frames_path)
                    .hint_text("<run>/frames")
                    .desired_width(220.0),
            );
            if field.lost_focus() {
                if let Some(run) = run.as_mut() {
                    run.set_frames_root(frames_path.trim());
                    *status = format!("{}: {}", run.dir.display(), run.status);
                }
            }
            let label = selected
                .as_ref()
                .map_or_else(|| "Replay".to_owned(), |name| format!("Replay {name}"));
            if ui
                .add_enabled(selected.is_some(), egui::Button::new(label))
                .clicked()
            {
                let (Some(run), Some(cell)) = (run.as_ref(), selected.as_ref()) else {
                    return;
                };
                match ReplayView::open(Path::new(scene_path.trim()), &run.traj_path(cell)) {
                    Ok(view) => {
                        *status = format!("{cell}: {} tick(s) replayed", view.ticks());
                        cell.clone_into(replay_cell);
                        *replay = Some(view);
                    }
                    Err(e) => *status = e.to_string(),
                }
            }
            if let Some(view) = replay.as_mut() {
                if ui
                    .button(if view.playing { "Pause" } else { "Play" })
                    .clicked()
                {
                    view.playing = !view.playing;
                }
                if ui.button("|<").clicked() {
                    view.step(-1);
                }
                if ui.button(">|").clicked() {
                    view.step(1);
                }
                ui.add(egui::Slider::new(&mut view.speed, 0.1..=4.0).text("x"));
            }
        });

        let Some(view) = replay.as_mut() else {
            ui.label(
                "Select a cell that has a trajectory, give the scene file, and press Replay \
                 (spec 23.3).",
            );
            return;
        };
        view.advance(dt, REPLAY_RATE_HZ);
        let last = view.ticks().saturating_sub(1);
        ui.horizontal(|ui| {
            ui.label(format!(
                "{replay_cell}  tick {}/{last}  {:.2} s",
                view.tick,
                view.tick as f64 / REPLAY_RATE_HZ
            ));
            ui.add(egui::Slider::new(&mut view.tick, 0..=last).text("tick"));
        });

        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        camera.width = response.rect.width().max(1.0) as u32;
        camera.height = response.rect.height().max(1.0) as u32;
        if response.dragged() {
            let drag = response.drag_delta();
            *camera = camera.orbit(
                f64::from(-drag.x) * ORBIT_PER_POINT,
                f64::from(drag.y) * ORBIT_PER_POINT,
            );
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                *camera = camera.zoom(f64::from(-scroll).mul_add(ZOOM_PER_POINT, 1.0));
            }
        }
        painter.rect_filled(response.rect, 0.0, Color32::from_gray(18));
        painter.add(egui::Shape::mesh(replay_mesh(
            &view.project(view.tick, camera),
            response.rect.min,
        )));
        if view.playing {
            ui.ctx().request_repaint();
        }
    }

    fn telemetry_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Performance (spec 12.4)");
            egui::Grid::new("metrics").striped(true).show(ui, |ui| {
                for (name, value) in self.telemetry.metric_rows() {
                    ui.label(name);
                    ui.label(
                        value.map_or_else(|| "-- (not measured)".to_owned(), |v| format!("{v:.3}")),
                    );
                    ui.end_row();
                }
            });
            ui.separator();
            ui.heading("Streams");
            egui::Grid::new("streams").striped(true).show(ui, |ui| {
                for (key, tick, value) in self.telemetry.latest() {
                    ui.label(format!("stream {}[{}]", key.stream.0, key.index));
                    ui.label(format!("tick {tick}"));
                    ui.label(format!("{value:.4}"));
                    ui.end_row();
                }
            });
            ui.separator();
            ui.heading("Events");
            for e in self.telemetry.events.iter().rev().take(200) {
                ui.label(format!("[{}] {} {:?}", e.tick, e.kind, e.fields));
            }
        });
    }

    fn images_tab(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Some(opened) = &mut self.opened else {
            ui.label("Open a bundle to see the pre/post preprocessing pair (spec 23.3).");
            return;
        };
        if let Some(err) = &opened.image_error {
            ui.label(err.as_str());
        }
        let outputs = opened.observation.outputs.len();
        ui.label(format!(
            "{} of {outputs} observation outputs are images",
            opened.pairs.len()
        ));
        egui::ScrollArea::vertical().show(ui, |ui| {
            for pair in &opened.pairs {
                let (before, after) = opened
                    .textures
                    .entry(pair.name.clone())
                    .or_insert_with(|| (texture(&ctx, pair, true), texture(&ctx, pair, false)));
                ui.heading(&pair.name);
                ui.horizontal(|ui| {
                    for (label, tex) in [("before", &*before), ("after", &*after)] {
                        ui.vertical(|ui| {
                            ui.label(label);
                            let scale = (240.0 / tex.size_vec2().x).max(1.0);
                            ui.image(egui::load::SizedTexture::new(
                                tex.id(),
                                tex.size_vec2() * scale,
                            ));
                        });
                    }
                });
                ui.separator();
            }
        });
    }

    fn diagnostics_tab(&mut self, ui: &mut egui::Ui) {
        let Some(opened) = &self.opened else {
            ui.label("Open a bundle to validate it.");
            return;
        };
        if opened.graph.diagnostics.is_empty() {
            ui.label("No diagnostics: the four IRs validate and agree (spec 11.1).");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for d in &opened.graph.diagnostics {
                ui.label(d.to_string());
            }
        });
    }
}

fn texture(ctx: &egui::Context, pair: &ImagePair, before: bool) -> egui::TextureHandle {
    let img = if before { &pair.before } else { &pair.after };
    let name = format!("{}-{}", pair.name, if before { "before" } else { "after" });
    rgb_texture(ctx, &name, img)
}

fn rgb_texture(ctx: &egui::Context, name: &str, img: &Rgb8Image) -> egui::TextureHandle {
    let color = egui::ColorImage::from_rgb([img.width, img.height], &img.data);
    ctx.load_texture(name, color, egui::TextureOptions::NEAREST)
}

// --- the Run tab -------------------------------------------------------------------------------

/// Frames the filmstrip shows, sampled evenly over the cell by [`RunView::filmstrip`].
const FILMSTRIP: usize = 8;

fn dash() -> String {
    "--".to_owned()
}

/// A metric cell. A histogram has no single number and an unmeasured metric has none at all
/// (spec 10.3): neither is rendered as `0`.
fn metric_text(value: &es_ir::evaluation::MetricValue) -> String {
    match value {
        es_ir::evaluation::MetricValue::Scalar(v) => format!("{v:.4}"),
        es_ir::evaluation::MetricValue::Histogram(h) => {
            format!("histogram, {} cause(s)", h.len())
        }
        es_ir::evaluation::MetricValue::Unavailable { reason } => format!("-- ({reason})"),
    }
}

fn acceptance_row(line: &es_ir::evaluation::AcceptanceResult) -> (String, Color32) {
    match line {
        es_ir::evaluation::AcceptanceResult::Determined {
            criterion,
            observed,
            passed,
        } => (
            format!(
                "{} {} {} ({}, {}): observed {observed:.4} -- {}",
                criterion.metric.name(),
                criterion.comparator.name(),
                criterion.threshold,
                criterion.aggregation.name(),
                criterion.suite.as_deref().unwrap_or("every suite"),
                if *passed { "pass" } else { "FAIL" }
            ),
            if *passed {
                Color32::from_rgb(120, 200, 120)
            } else {
                Color32::from_rgb(230, 120, 110)
            },
        ),
        es_ir::evaluation::AcceptanceResult::Unavailable { metric, reason } => (
            format!("{}: not measured ({reason})", metric.name()),
            Color32::from_gray(160),
        ),
    }
}

/// One colour per `EventSource`, violations as a tick beneath (spec 23.3).
fn paint_timeline(ui: &mut egui::Ui, buckets: &[Bucket]) {
    if buckets.is_empty() {
        ui.label("no events.json: this run recorded no per-tick sources");
        return;
    }
    let (response, painter) =
        ui.allocate_painter(Vec2::new(ui.available_width(), 30.0), Sense::hover());
    let rect = response.rect;
    let w = (rect.width() / buckets.len() as f32).max(1.0);
    for (i, bucket) in buckets.iter().enumerate() {
        let x = rect.left() + i as f32 * rect.width() / buckets.len() as f32;
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(w, 18.0)),
            0.0,
            source_colour(bucket.source),
        );
        if !bucket.counts.is_empty() {
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, rect.top() + 21.0), Vec2::new(w, 8.0)),
                0.0,
                Color32::from_rgb(240, 200, 80),
            );
        }
    }
}

// --- the Replay panel --------------------------------------------------------------------------

/// Where the replay camera starts: the demo's showcase view (`es video showcase --eye`).
const SHOWCASE_CAMERA: Camera = Camera {
    eye: [0.55, -0.45, 0.42],
    look_at: [0.12, -0.02, 0.08],
    // 45 degrees, the showcase default. `to_radians` is not `const`.
    fov_y: std::f64::consts::FRAC_PI_4,
    width: 640,
    height: 400,
};

/// The control rate a recorded `.estraj` tick is worth. 50 Hz is the demo deployment's
/// `rate.control`; a run directory carries no Deployment IR to read it from, and playing at
/// the wrong rate only changes how fast the arm appears to move.
const REPLAY_RATE_HZ: f64 = 50.0;

/// How much of the Run tab the replay panel takes when it first opens, and the least it ever
/// takes: the 3D view has to be visible the moment `Replay` is pressed, not after the user
/// finds the separator.
const REPLAY_PANEL_FRACTION: f32 = 0.45;
const REPLAY_PANEL_MIN: f32 = 320.0;

/// Radians of orbit per point of drag, and zoom per point of scroll.
const ORBIT_PER_POINT: f64 = 0.008;
const ZOOM_PER_POINT: f64 = 0.002;

/// One mesh for the whole frame: three vertices and one triangle per [`Tri2d`], already in
/// paint order, so the painter's algorithm is just the order they are added in.
fn replay_mesh(projected: &Projected, origin: Pos2) -> egui::Mesh {
    let mut mesh = egui::Mesh::default();
    for tri in projected {
        let base = mesh.vertices.len() as u32;
        let colour = Color32::from_rgb(tri.color[0], tri.color[1], tri.color[2]);
        for p in tri.p {
            mesh.colored_vertex(origin + Vec2::new(p[0], p[1]), colour);
        }
        mesh.add_triangle(base, base + 1, base + 2);
    }
    mesh
}

fn source_colour(source: es_eval::runner::EventSource) -> Color32 {
    match source {
        es_eval::runner::EventSource::Policy => Color32::from_rgb(70, 130, 180),
        es_eval::runner::EventSource::Human => Color32::from_rgb(150, 150, 200),
        es_eval::runner::EventSource::Clamped => Color32::from_rgb(220, 170, 60),
        es_eval::runner::EventSource::Fallback => Color32::from_rgb(210, 90, 80),
    }
}

fn paint_layer(
    painter: &egui::Painter,
    layer: &LayerView,
    at: &impl Fn([f32; 2]) -> Pos2,
    size: Vec2,
    zoom: f32,
) {
    let font = FontId::proportional(12.0 * zoom);
    let band = layer.nodes.iter().find_map(|n| n.layout);
    if let Some(p) = band {
        painter.text(
            at(p) - Vec2::new(0.0, 18.0 * zoom),
            Align2::LEFT_BOTTOM,
            format!("{:?} IR", layer.kind),
            font.clone(),
            Color32::from_gray(150),
        );
    }
    for e in &layer.edges {
        let (Some(from), Some(to)) = (node_of(layer, e.from.node), node_of(layer, e.to.node))
        else {
            continue;
        };
        let (Some(a), Some(b)) = (from.layout, to.layout) else {
            continue;
        };
        let p0 = at(a) + Vec2::new(size.x, size.y * 0.5);
        let p3 = at(b) + Vec2::new(0.0, size.y * 0.5);
        bezier(painter, p0, p3, true, Color32::from_gray(120), zoom);
    }
    for node in &layer.nodes {
        let Some(p) = node.layout else { continue };
        let rect = Rect::from_min_size(at(p), size);
        painter.rect_filled(rect, 4.0, Color32::from_rgb(40, 44, 52));
        painter.rect_stroke(
            rect,
            4.0,
            Stroke::new(1.0_f32, Color32::from_gray(90)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.min + Vec2::splat(6.0 * zoom),
            Align2::LEFT_TOP,
            &node.label,
            font.clone(),
            Color32::from_gray(220),
        );
        painter.text(
            rect.left_bottom() + Vec2::new(6.0 * zoom, -4.0 * zoom),
            Align2::LEFT_BOTTOM,
            ports_line(node),
            FontId::proportional(9.0 * zoom),
            Color32::from_gray(140),
        );
    }
}

fn ports_line(node: &NodeView) -> String {
    format!(
        "{} -> {}",
        node.ports.inputs.join(","),
        node.ports.outputs.join(",")
    )
}

fn paint_cross_edge(
    painter: &egui::Painter,
    graph: &LayeredGraph,
    edge: &CrossEdge,
    at: &impl Fn([f32; 2]) -> Pos2,
    size: Vec2,
    zoom: f32,
) {
    let (Some(a), Some(b)) = (graph.position(edge.from), graph.position(edge.to)) else {
        return;
    };
    let p0 = at(a) + Vec2::new(size.x * 0.5, size.y);
    let p3 = at(b) + Vec2::new(size.x * 0.5, 0.0);
    let colour = Color32::from_rgb(120, 170, 255);
    bezier(painter, p0, p3, false, colour, zoom);
    painter.text(
        p0.lerp(p3, 0.5),
        Align2::CENTER_CENTER,
        &edge.label,
        FontId::proportional(10.0 * zoom),
        colour,
    );
}

/// A cubic between two ports: control points offset along the flow direction — horizontally
/// inside a layer, vertically across layers.
fn bezier(
    painter: &egui::Painter,
    p0: Pos2,
    p3: Pos2,
    horizontal: bool,
    colour: Color32,
    zoom: f32,
) {
    let d = if horizontal {
        Vec2::new(((p3.x - p0.x).abs() * 0.5).max(20.0 * zoom), 0.0)
    } else {
        Vec2::new(0.0, ((p3.y - p0.y).abs() * 0.5).max(20.0 * zoom))
    };
    painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
        [p0, p0 + d, p3 - d, p3],
        false,
        Color32::TRANSPARENT,
        Stroke::new(1.5 * zoom, colour),
    ));
}

fn node_of(layer: &LayerView, id: es_ir::NodeId) -> Option<&NodeView> {
    layer.nodes.iter().find(|n| n.id == id)
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

// --- the edit canvas -------------------------------------------------------------------------

const LINK: Color32 = Color32::from_rgb(200, 180, 90);
const PIN_IN: Color32 = Color32::from_rgb(120, 170, 255);

/// One frame of canvas geometry, owned. Built from the session, used for hit-testing and
/// painting, and rebuilt after an edit - so the mutation and the drawing never hold a borrow
/// of the session at the same time.
struct CanvasView {
    origin: Pos2,
    zoom: f32,
    positions: BTreeMap<NodeId, [f32; 2]>,
    kinds: BTreeMap<NodeId, &'static str>,
    /// `(inputs, outputs)` as the node itself declares them.
    ports: BTreeMap<NodeId, (Vec<String>, Vec<String>)>,
    edges: Vec<es_ir::Edge>,
}

impl CanvasView {
    fn of(session: &EditSession, origin: Pos2, zoom: f32) -> Self {
        let kinds = edit::kinds_by_id(&session.graph);
        let ports = kinds
            .keys()
            .map(|id| {
                (
                    *id,
                    (
                        session.graph.port_names(*id, es_ir::Dir::In),
                        session.graph.port_names(*id, es_ir::Dir::Out),
                    ),
                )
            })
            .collect();
        Self {
            origin,
            zoom,
            positions: session.layout.positions.clone(),
            kinds,
            ports,
            edges: session.graph.edges().to_vec(),
        }
    }

    fn rect(&self, id: NodeId) -> Option<Rect> {
        let p = self.positions.get(&id)?;
        Some(Rect::from_min_size(
            self.origin + Vec2::new(p[0], p[1]) * self.zoom,
            Vec2::new(NODE_W, NODE_H) * self.zoom,
        ))
    }

    fn names(&self, id: NodeId, input: bool) -> &[String] {
        self.ports.get(&id).map_or(
            &[][..],
            |(i, o)| {
                if input {
                    i.as_slice()
                } else {
                    o.as_slice()
                }
            },
        )
    }

    fn pin(rect: Rect, i: usize, n: usize, input: bool) -> Pos2 {
        let t = (i as f32 + 1.0) / (n as f32 + 1.0);
        Pos2::new(
            if input { rect.left() } else { rect.right() },
            rect.top() + rect.height() * t,
        )
    }

    fn port_pos(&self, port: &PortRef, input: bool) -> Option<Pos2> {
        let rect = self.rect(port.node)?;
        let names = self.names(port.node, input);
        let i = names.iter().position(|n| *n == port.port)?;
        Some(Self::pin(rect, i, names.len(), input))
    }

    /// The port whose pin is under `pos`, if any.
    fn port_at(&self, pos: Pos2, input: bool) -> Option<PortRef> {
        for id in self.kinds.keys() {
            let Some(rect) = self.rect(*id) else { continue };
            let names = self.names(*id, input);
            for (i, name) in names.iter().enumerate() {
                if Self::pin(rect, i, names.len(), input).distance(pos) <= PORT_R * self.zoom {
                    return Some(PortRef::new(*id, name.clone()));
                }
            }
        }
        None
    }

    /// Output pin first (a wire is pulled from a producer), then the node body, then the
    /// background.
    fn start_drag(&self, pos: Pos2) -> Drag {
        if let Some(from) = self.port_at(pos, false) {
            return Drag::Link { from };
        }
        for id in self.kinds.keys() {
            if let Some(rect) = self.rect(*id) {
                if rect.contains(pos) {
                    return Drag::Node {
                        id: *id,
                        origin: self.positions.get(id).copied().unwrap_or_default(),
                        grab: pos - rect.min,
                    };
                }
            }
        }
        Drag::Pan
    }

    fn paint(&self, painter: &egui::Painter, selected: Option<NodeId>) {
        for edge in &self.edges {
            let (Some(p0), Some(p3)) = (
                self.port_pos(&edge.from, false),
                self.port_pos(&edge.to, true),
            ) else {
                continue;
            };
            bezier(painter, p0, p3, true, Color32::from_gray(140), self.zoom);
        }
        let font = FontId::proportional(12.0 * self.zoom);
        for (id, kind) in &self.kinds {
            let Some(rect) = self.rect(*id) else { continue };
            let fill = if selected == Some(*id) {
                Color32::from_rgb(60, 72, 96)
            } else {
                Color32::from_rgb(40, 44, 52)
            };
            painter.rect_filled(rect, 4.0, fill);
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0_f32, Color32::from_gray(110)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.min + Vec2::splat(6.0 * self.zoom),
                Align2::LEFT_TOP,
                format!("{kind} #{}", id.0),
                font.clone(),
                Color32::from_gray(220),
            );
            for (input, colour) in [(true, PIN_IN), (false, LINK)] {
                let n = self.names(*id, input).len();
                for i in 0..n {
                    painter.circle_filled(
                        Self::pin(rect, i, n, input),
                        PORT_R * 0.5 * self.zoom,
                        colour,
                    );
                }
            }
        }
    }
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
