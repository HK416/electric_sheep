//! The egui shell: four tabs over [`crate::model`] (spec 23.2, spec 23.3).
//!
//! Thin on purpose. Nothing here decides *what* is drawn — the view-model did that, headless
//! and under test. This file turns positions into rectangles, and it is the only part CI
//! merely compiles rather than runs, because running it needs a display.
//!
//! Read-only (spec 23.4 stage 1): the graph pans and zooms, nodes do not move. There is no
//! code path that writes a position back into an IR, because layout is not IR (spec 4.2
//! rule 7) and because editing is stage 2.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use es_compile::bundle;
use es_compile::PolicyBundle;
use es_ir::deployment::DeploymentIr;
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial::{self, Layout};
use es_ir::task::TaskIr;

use crate::model::graph_view::{CrossEdge, LayerView, LayeredGraph, NodeView};
use crate::model::image_view::{BeforeAfter, ImagePair};
use crate::model::telemetry_view::{Source, TelemetryModel};

const NODE_W: f32 = 178.0;
const NODE_H: f32 = 40.0;
/// Telemetry messages drained per frame (spec 23.3 runs the viewer on a budget).
const PUMP_BUDGET: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Graph,
    Telemetry,
    Images,
    Diagnostics,
}

/// One opened bundle: the four IRs' view-model plus whatever the image tab could make of it.
struct Opened {
    graph: LayeredGraph,
    observation: ObservationIr,
    pairs: Vec<ImagePair>,
    image_error: Option<String>,
    textures: BTreeMap<String, (egui::TextureHandle, egui::TextureHandle)>,
}

pub struct EditorApp {
    tab: Tab,
    path: String,
    status: String,
    opened: Option<Opened>,
    telemetry: TelemetryModel,
    source: Source,
    pan: Vec2,
    zoom: f32,
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
            telemetry: TelemetryModel::default(),
            source,
            pan: Vec2::new(60.0, 40.0),
            zoom: 1.0,
        }
    }

    /// Open a bundle at startup (`es-editor <bundle.esb>`).
    #[must_use]
    pub fn with_bundle(mut self, path: &str) -> Self {
        path.clone_into(&mut self.path);
        self.open();
        self
    }

    fn open(&mut self) {
        let path = PathBuf::from(self.path.trim());
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
                self.opened = Some(Opened {
                    graph,
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
                        .hint_text("bundle.esb, or a directory of the five .toml files")
                        .desired_width(380.0),
                );
                if ui.button("Open bundle...").clicked() {
                    self.open();
                }
                ui.separator();
                for (tab, name) in [
                    (Tab::Graph, "Graph"),
                    (Tab::Telemetry, "Telemetry"),
                    (Tab::Images, "Images"),
                    (Tab::Diagnostics, "Diagnostics"),
                ] {
                    ui.selectable_value(&mut self.tab, tab, name);
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
    let color = egui::ColorImage::from_rgb([img.width, img.height], &img.data);
    ctx.load_texture(
        format!("{}-{}", pair.name, if before { "before" } else { "after" }),
        color,
        egui::TextureOptions::NEAREST,
    )
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
