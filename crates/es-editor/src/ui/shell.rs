//! The shell (packet M12/Y10, `docs/design/editor-redesign.md` sections 3 and 6.1): a menu bar,
//! the step bar, the dock and the status line.
//!
//! Which panes exist and where each step puts them, what a step's button says, its colour,
//! whether it opens and where `Next` goes are all [`crate::model::layout`]'s; the states are
//! [`crate::model::workflow`]'s. The step panels are placeholders until packets M12/Y11-Y13
//! fill them; until then the Advanced panes along the bottom do the work.

use eframe::egui;
use egui::{Color32, RichText};
use egui_dock::{DockArea, DockState, TabViewer};

use crate::app::{EditorApp, OpenProject};
use crate::model::dialogs;
use crate::model::fonts;
use crate::model::i18n::{self, Lang};
use crate::model::labels::Browse;
use crate::model::layout::{self, Pane};
use crate::model::workflow::{Phase, PhaseState};
use crate::ui::advanced::short_hash;

/// One frame of the whole window.
pub fn draw(app: &mut EditorApp, ctx: &egui::Context) {
    let mut restyle = false;
    egui::TopBottomPanel::top("menu").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            file_menu(app, ui);
            restyle = view_menu(app, ui);
            ui.separator();
            path_bar(app, ui);
        });
    });
    // With nothing to show, the start screen fills the window between the menu and the status
    // line (packet M12/Y11); the new-project dialog floats over whichever is drawn.
    let start = super::home::shown(app);
    if !start {
        egui::TopBottomPanel::top("steps").show(ctx, |ui| step_bar(app, ui));
    }
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| status_line(app, ui));
    if start {
        egui::CentralPanel::default().show(ctx, |ui| super::home::draw(app, ui));
    } else {
        dock(app, ctx);
    }
    super::home::dialog(app, ctx);

    // Asked for from inside a pane, while the dock was out of `app`; opened now that it is
    // back, so what opening focuses lands in the dock that is kept.
    if let Some(path) = app.pending_open.take() {
        app.path = path;
        app.open();
    }
    // A language or a size that has just been clicked: the font chain and every text style
    // are rebuilt from what `fonts` says, once, rather than every frame.
    if restyle {
        app.apply_style(ctx);
    }
}

/// File: a new project (with the start screen, packet M12/Y11), the three ways to open
/// something, watching a running run, and the paths opened lately (packet M7/E3).
fn file_menu(app: &mut EditorApp, ui: &mut egui::Ui) {
    let mut open: Option<String> = None;
    let mut attach = false;
    ui.menu_button(app.t("menu.file"), |ui| {
        ui.menu_button(app.t("menu.new_project"), |ui| {
            super::home::template_menu(app, ui);
        });
        for (key, browse) in super::home::OPEN_WAYS {
            let item = ui
                .add_enabled(dialogs::AVAILABLE, egui::Button::new(app.t(key)))
                .on_disabled_hover_text(app.t("open.no_dialog.hint"));
            if item.clicked() {
                ui.close_kind(egui::UiKind::Menu);
                open = dialogs::pick(browse).map(|p| p.display().to_string());
            }
        }
        if ui.button(app.t("menu.attach")).clicked() {
            attach = true;
            ui.close_kind(egui::UiKind::Menu);
        }
        ui.separator();
        ui.menu_button(app.t("menu.recent"), |ui| {
            if app.recent.paths.is_empty() {
                ui.weak(app.t("menu.recent_empty"));
            }
            for path in &app.recent.paths {
                if ui.button(path.display().to_string()).clicked() {
                    open = Some(path.display().to_string());
                    ui.close_kind(egui::UiKind::Menu);
                }
            }
        });
    });
    // The address and the Connect button are the Live pane's (packet M7/E4).
    if attach {
        app.focus(Pane::AdvancedLive);
    }
    if let Some(path) = open {
        app.path = path;
        app.open();
    }
}

/// View: the layout back to the step's default, and the two display settings (packet
/// M7/E6). `true` when a setting moved and the style has to be rebuilt.
fn view_menu(app: &mut EditorApp, ui: &mut egui::Ui) -> bool {
    let mut restyle = false;
    ui.menu_button(app.t("menu.view"), |ui| {
        if ui.button(app.t("menu.reset_layout")).clicked() {
            let arrangement = app.arrangement();
            app.docks.reset(arrangement);
            ui.close_kind(egui::UiKind::Menu);
        }
        ui.separator();
        ui.label(app.t("menu.language"));
        // A language is named in its own language and never translated: that is how someone
        // who cannot read the current one finds theirs.
        for lang in Lang::ALL {
            let label = i18n::t(lang, lang.label_key());
            if ui
                .selectable_label(app.settings.lang == lang, label)
                .clicked()
            {
                app.settings.lang = lang;
                restyle = true;
            }
        }
        ui.separator();
        ui.label(app.t("textsize.label"));
        for size in fonts::TextSize::ALL {
            if ui
                .selectable_label(app.settings.text_size == size, app.t(size.label_key()))
                .clicked()
            {
                app.settings.text_size = size;
                restyle = true;
            }
        }
    });
    restyle
}

/// A typed path, for a build without a file dialog and for anyone who has one on the
/// clipboard: the same `open` the menu and a dropped file reach (packet M7/E3).
fn path_bar(app: &mut EditorApp, ui: &mut egui::Ui) {
    let hint = app.t("open.hint");
    ui.add(
        egui::TextEdit::singleline(&mut app.path)
            .hint_text(hint)
            .desired_width(320.0),
    );
    if ui
        .button(app.t("open.open"))
        .on_hover_text(app.t("open.open.hint"))
        .clicked()
    {
        app.open();
    }
    if let Some(picked) = app.browse(ui, Browse::Policy) {
        app.path = picked;
        app.open();
    }
}

/// One button per step, a dot in its state's colour before its name; `Next ▶` and what is
/// left at the far end.
/// Without a project the five are drawn disabled beside `shell.not_a_project`.
fn step_bar(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    ui.horizontal(|ui| {
        let Some(open) = app.project.as_mut() else {
            ui.add_enabled_ui(false, |ui| {
                for phase in Phase::ALL {
                    let _ = ui.button(layout::step_text(lang, phase, &PhaseState::Locked));
                }
            });
            ui.label(i18n::t(lang, "shell.not_a_project"));
            return;
        };
        for (phase, state) in Phase::ALL.into_iter().zip(&open.phases) {
            let text = layout::step_text(lang, phase, state);
            let word = i18n::t(lang, state.key());
            let button = ui
                .add_enabled(
                    layout::can_open(state),
                    egui::Button::selectable(open.phase == phase, (dot(state), text)),
                )
                .on_hover_text(word)
                .on_disabled_hover_text(word);
            if button.clicked() {
                open.phase = phase;
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let next = layout::next_phase(open.phase, &open.phases);
            let pressed = ui
                .add_enabled(
                    next.is_some(),
                    egui::Button::new(i18n::t(lang, "shell.next")),
                )
                .clicked();
            if let (true, Some(next)) = (pressed, next) {
                open.phase = next;
            }
            ui.label(layout::left_text(lang, &open.phases));
        });
    });
}

/// The status line (packet M7/E3): the last news, and while editing the hash of what is in
/// memory and the count of what the IR complains about, both live (spec 5.3, spec 23.4).
fn status_line(app: &EditorApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label(&app.status);
        if let Some(session) = &app.edit {
            ui.separator();
            ui.label(app.fill(
                "status.hash",
                &[
                    &format!("{:?}", session.graph.kind()),
                    &short_hash(session.hash().as_ref()),
                ],
            ));
            ui.separator();
            ui.label(app.fill(
                "status.diagnostics",
                &[&session.diagnostics().len().to_string()],
            ));
        }
        ui.separator();
        ui.label(app.fill("status.messages", &[&app.telemetry.received.to_string()]));
    });
}

/// The dock of the arrangement shown now. It is taken out of `app` for the frame, so the panes
/// can borrow `app` whole, and put back under the arrangement it was taken from.
fn dock(app: &mut EditorApp, ctx: &egui::Context) {
    let arrangement = app.arrangement();
    let mut dock = std::mem::replace(app.docks.get(arrangement), DockState::new(Vec::new()));
    DockArea::new(&mut dock)
        .show_close_buttons(false)
        .show_leaf_close_all_buttons(false)
        .show(ctx, &mut Panes(app));
    *app.docks.get(arrangement) = dock;
}

struct Panes<'a>(&'a mut EditorApp);

impl TabViewer for Panes<'_> {
    type Tab = Pane;

    fn title(&mut self, pane: &mut Pane) -> egui::WidgetText {
        self.0.t(pane.key()).into()
    }

    /// The title changes with the language; the pane, and what egui remembers about it, not.
    fn id(&mut self, pane: &mut Pane) -> egui::Id {
        egui::Id::new(*pane)
    }

    fn ui(&mut self, ui: &mut egui::Ui, pane: &mut Pane) {
        let app = &mut *self.0;
        match pane {
            Pane::Viewport => viewport(app, ui),
            Pane::StepPanel => step_panel(app, ui),
            Pane::Summary => summary(app, ui),
            Pane::Console => console(app, ui),
            Pane::AdvancedGraph => app.graph_tab(ui),
            Pane::AdvancedSees => app.images_tab(ui),
            Pane::AdvancedProblems => app.diagnostics_tab(ui),
            Pane::AdvancedMetrics => app.run_tab(ui),
            Pane::AdvancedLive => app.telemetry_tab(ui),
        }
    }

    fn on_tab_button(&mut self, pane: &mut Pane, response: &egui::Response) {
        if let Some(hint) = pane.hint_key() {
            response.clone().on_hover_text(self.0.t(hint));
        }
    }

    /// No pane can be closed or floated off: every one stays reachable, and View > Reset the
    /// layout puts a dragged one back.
    fn is_closeable(&self, _pane: &Pane) -> bool {
        false
    }

    fn allowed_in_windows(&self, _pane: &mut Pane) -> bool {
        false
    }

    /// The panes lay themselves out, as the tabs did in the window before the dock.
    fn scroll_bars(&self, _pane: &Pane) -> [bool; 2] {
        [false, false]
    }
}

/// The centre of every step. What it shows is packets M12/Y11-Y13's.
fn viewport(app: &EditorApp, ui: &mut egui::Ui) {
    if app.project.is_none() {
        ui.label(app.t("shell.not_a_project"));
        return;
    }
    ui.weak(app.t("shell.coming_soon"));
}

/// The step's own panel, left: its name, where it is, and - until its packet lands - a line
/// saying it is not built yet.
fn step_panel(app: &EditorApp, ui: &mut egui::Ui) {
    let Some(open) = &app.project else {
        ui.label(app.t("shell.not_a_project"));
        return;
    };
    let lang = app.settings.lang;
    let state = state_of(open, open.phase);
    ui.heading(layout::step_text(lang, open.phase, state));
    ui.label(i18n::t(lang, state.key()));
    ui.separator();
    ui.weak(app.t("shell.coming_soon"));
}

/// The project at a glance, right: its name, its folder and the five steps' states.
fn summary(app: &EditorApp, ui: &mut egui::Ui) {
    let Some(open) = &app.project else {
        ui.label(app.t("shell.not_a_project"));
        return;
    };
    let lang = app.settings.lang;
    ui.heading(app.fill("shell.project", &[&open.project.file.name]));
    ui.weak(open.project.root.display().to_string());
    ui.separator();
    egui::Grid::new("summary-steps").show(ui, |ui| {
        for (phase, state) in Phase::ALL.into_iter().zip(&open.phases) {
            ui.horizontal(|ui| {
                ui.label(dot(state));
                ui.label(layout::step_text(lang, phase, state));
            });
            ui.label(i18n::t(lang, state.key()));
            ui.end_row();
        }
    });
}

/// A step's state as a dot in [`layout::colour`]'s colour.
pub(crate) fn dot(state: &PhaseState) -> RichText {
    let [r, g, b] = layout::colour(state);
    RichText::new("\u{25cf}").color(Color32::from_rgb(r, g, b))
}

fn state_of(open: &OpenProject, phase: Phase) -> &PhaseState {
    let i = Phase::ALL.iter().position(|p| *p == phase).unwrap_or(0);
    &open.phases[i]
}

/// `es`'s own lines, as the launched child printed them (packet M7/E5): untranslated, since
/// they are another program's words.
fn console(app: &EditorApp, ui: &mut egui::Ui) {
    if app.launch.lines().next().is_none() {
        ui.weak(app.t("shell.console_empty"));
        return;
    }
    egui::ScrollArea::vertical()
        .id_salt("console")
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for line in app.launch.lines() {
                ui.monospace(line);
            }
        });
}

/// With nothing open, in the Design graph's place: where to open something, and the paths
/// opened lately. The start screen (packet M12/Y11) replaces it with a whole window.
pub(crate) fn nothing_open(app: &mut EditorApp, ui: &mut egui::Ui) {
    ui.add_space(12.0);
    ui.label(app.t("shell.empty"));
    ui.add_space(12.0);
    ui.heading(app.t("home.recent"));
    if app.recent.paths.is_empty() {
        ui.weak(app.t("menu.recent_empty"));
    }
    for path in &app.recent.paths {
        if ui.link(path.display().to_string()).clicked() {
            app.pending_open = Some(path.display().to_string());
        }
    }
}
