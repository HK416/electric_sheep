//! The start screen (packet M12/Y11, `docs/design/editor-redesign.md` section 3): the language,
//! the PC check, one card per template, the recent paths with their miniature step bars, and
//! the file-level ways in - the whole window whenever nothing is open. Also the new-project
//! dialog a card, or File > New project from a template, opens.
//!
//! Drawing only. How each PC-check item is marked, what a card that cannot be created says,
//! where a new project goes and when this screen shows are [`crate::model::home`]'s.

use std::path::Path;

use eframe::egui;
use egui::{Color32, RichText};

use crate::app::EditorApp;
use crate::model::dialogs;
use crate::model::home::{self, Availability, DepsState, Mark, RecentCard};
use crate::model::i18n::{self, Lang, Strings};
use crate::model::labels::Browse;
use crate::model::layout::{self, Pane};
use crate::model::workflow::Phase;
use crate::ui::shell::dot;

/// The three ways to open something that already exists, shared with the File menu. What a
/// folder turns out to be is `recent::classify`'s, so the project and the run folder share a
/// folder chooser; the words say which one the person meant.
pub(crate) const OPEN_WAYS: [(&str, Browse); 3] = [
    ("menu.open_project", Browse::Folder),
    ("menu.open_file", Browse::Policy),
    ("menu.open_run", Browse::Folder),
];

/// A card is this wide, so two sit side by side in a normal window and wrap in a narrow one.
const CARD_WIDTH: f32 = 420.0;

/// Whether the start screen fills the window this frame: [`home::StartScreen::shown`]'s answer.
pub(crate) fn shown(app: &EditorApp) -> bool {
    let open = app.project.is_some() || app.opened.is_some() || app.run.is_some();
    app.home.shown(open, app.telemetry.received > 0)
}

fn rgb([r, g, b]: [u8; 3]) -> Color32 {
    Color32::from_rgb(r, g, b)
}

/// A template's own words. Its `name`, `summary` and `notice` are keys read from a file, so
/// they are not `'static` and go through the table directly.
fn word(lang: Lang, key: &str) -> &str {
    Strings::get(lang).t(key)
}

/// The PC check so far; the first call starts it.
fn check(app: &mut EditorApp) -> DepsState {
    app.home.poll(&app.launch.binary().path).clone()
}

fn ready(state: &DepsState) -> Option<&home::Deps> {
    match state {
        DepsState::Ready(deps) => Some(deps),
        DepsState::Checking | DepsState::Failed(_) => None,
    }
}

pub fn draw(app: &mut EditorApp, ui: &mut egui::Ui) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            language(app, ui);
            pc_check(app, ui);
            ui.add_space(20.0);
            templates(app, ui);
            ui.add_space(20.0);
            recent(app, ui);
            ui.add_space(20.0);
            ways_in(app, ui);
        });
}

/// Each language named in its own words, top right: someone who cannot read the current one
/// finds theirs without knowing the View menu.
fn language(app: &mut EditorApp, ui: &mut egui::Ui) {
    let mut chosen = None;
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            for lang in Lang::ALL.into_iter().rev() {
                let label = i18n::t(lang, lang.label_key());
                if ui
                    .selectable_label(app.settings.lang == lang, label)
                    .clicked()
                {
                    chosen = Some(lang);
                }
            }
            ui.label(format!("\u{1f310} {}", app.t("menu.language")));
        });
    });
    if let Some(lang) = chosen {
        app.settings.lang = lang;
        app.apply_style(ui.ctx());
    }
}

/// One mark per item while the answer is in, a spinner while it is not, and the failure with
/// the `es` it ran when there is none.
fn pc_check(app: &mut EditorApp, ui: &mut egui::Ui) {
    let state = check(app);
    let items = ready(&state).map(home::pc_check).unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        ui.strong(app.t("home.check"));
        match &state {
            DepsState::Checking => {
                ui.spinner();
                ui.label(app.t("home.checking"));
            }
            DepsState::Failed(why) => {
                ui.colored_label(
                    rgb(Mark::Missing.colour()),
                    app.fill("home.check_failed", &[why]),
                );
            }
            DepsState::Ready(_) => {
                for item in &items {
                    let text = match &item.name {
                        Some(name) => app.fill(item.key, &[name]),
                        None => app.t(item.key).to_owned(),
                    };
                    let hover = [Some(app.t(item.mark.key())), item.reason.as_deref()]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("\n");
                    // One unbroken label per item, so the line wraps between items.
                    let text = format!("{} {text}", item.mark.glyph());
                    let text = RichText::new(text).color(rgb(item.mark.colour()));
                    ui.add(egui::Label::new(text).extend()).on_hover_text(hover);
                    ui.add_space(8.0);
                }
            }
        }
    });
    for (key, hint) in home::install_lines(&items) {
        let line = ui.weak(app.t(key));
        if let Some(hint) = hint {
            line.on_hover_text(app.t(hint));
        }
    }
    if matches!(state, DepsState::Failed(_)) {
        let es = app.launch.binary();
        ui.weak(app.fill(
            "home.es_binary",
            &[&es.path.display().to_string(), &es.reason],
        ));
    }
}

/// One card per template: its name, what it does, its notice in plain sight, and Create - or,
/// when this PC cannot run it, the card greyed with what is missing.
fn templates(app: &mut EditorApp, ui: &mut egui::Ui) {
    ui.heading(app.t("home.what"));
    let lang = app.settings.lang;
    let state = check(app);
    let mut pick = None;
    ui.horizontal_wrapped(|ui| {
        for (i, template) in app.home.templates.iter().enumerate() {
            let availability = home::availability(template, ready(&state));
            let can = availability == Availability::Ready;
            egui::Frame::group(ui.style())
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.set_width(CARD_WIDTH);
                        ui.add_enabled_ui(can, |ui| {
                            ui.heading(word(lang, &template.name));
                            ui.label(word(lang, &template.summary));
                        });
                        if let Some(notice) = &template.notice {
                            let notice = format!("\u{26a0} {}", word(lang, notice));
                            ui.label(RichText::new(notice).color(rgb(Mark::Optional.colour())));
                        }
                        ui.add_space(4.0);
                        if ui
                            .add_enabled(can, egui::Button::new(app.t("home.create")))
                            .clicked()
                        {
                            pick = Some(i);
                        }
                        if let Some((mark, why)) = home::availability_text(lang, &availability) {
                            ui.colored_label(rgb(mark.colour()), why);
                        }
                    });
                });
        }
    });
    for (path, why) in &app.home.broken {
        ui.colored_label(
            rgb(Mark::Missing.colour()),
            app.fill("home.broken_template", &[&path.display().to_string(), why]),
        );
    }
    if app.home.templates.is_empty() && app.home.broken.is_empty() {
        ui.weak(app.t("home.no_templates"));
    }
    open_dialog(app, pick);
}

/// The dialog for `templates[index]`, named after it in the reader's language.
fn open_dialog(app: &mut EditorApp, index: Option<usize>) {
    let Some(i) = index else { return };
    let lang = app.settings.lang;
    let name = app
        .home
        .templates
        .get(i)
        .map(|t| word(lang, &t.name).to_owned());
    if let Some(name) = name {
        app.home.open_dialog(i, &name);
    }
}

/// File > New project from a template: one item per template, enabled when it is ready and
/// opening the same dialog as its card.
pub(crate) fn template_menu(app: &mut EditorApp, ui: &mut egui::Ui) {
    let lang = app.settings.lang;
    let state = check(app);
    let mut pick = None;
    for (i, template) in app.home.templates.iter().enumerate() {
        let availability = home::availability(template, ready(&state));
        let mut item = ui.add_enabled(
            availability == Availability::Ready,
            egui::Button::new(word(lang, &template.name)),
        );
        if let Some((_, why)) = home::availability_text(lang, &availability) {
            item = item.on_disabled_hover_text(why);
        }
        if item.clicked() {
            pick = Some(i);
            ui.close_kind(egui::UiKind::Menu);
        }
    }
    if app.home.templates.is_empty() {
        ui.weak(app.t("home.no_templates"));
    }
    open_dialog(app, pick);
}

/// The paths opened lately: a project with its five dots and where it stands, anything else as
/// its path, and a path that is gone greyed.
fn recent(app: &mut EditorApp, ui: &mut egui::Ui) {
    ui.heading(app.t("home.recent"));
    let lang = app.settings.lang;
    let cards = app.home.cards(&app.recent).to_vec();
    if cards.is_empty() {
        ui.weak(app.t("menu.recent_empty"));
    }
    let mut open = None;
    for card in &cards {
        match card {
            RecentCard::Project { path, name, phases } => {
                ui.horizontal(|ui| {
                    let link = ui
                        .link(RichText::new(name).strong())
                        .on_hover_text(path.display().to_string());
                    if link.clicked() {
                        open = Some(path.clone());
                    }
                    ui.spacing_mut().item_spacing.x = 2.0;
                    for state in phases {
                        ui.label(dot(state));
                    }
                    ui.spacing_mut().item_spacing.x = 8.0;
                    let at = layout::start_phase(phases);
                    if let Some((_, state)) =
                        Phase::ALL.into_iter().zip(phases).find(|(p, _)| *p == at)
                    {
                        ui.label(layout::step_text(lang, at, state));
                        ui.weak(i18n::t(lang, state.key()));
                    }
                });
            }
            RecentCard::Other { path, .. } => {
                if ui.link(path.display().to_string()).clicked() {
                    open = Some(path.clone());
                }
            }
            RecentCard::Missing { path } => {
                ui.weak(app.fill("home.gone", &[&path.display().to_string()]));
            }
        }
    }
    if let Some(path) = open {
        app.path = path.display().to_string();
        app.open();
    }
}

/// Opening a project, a file or a run folder, and watching a running run: the File menu's own
/// entries, for someone who has not found the menu yet.
fn ways_in(app: &mut EditorApp, ui: &mut egui::Ui) {
    ui.heading(app.t("home.open_ways"));
    let mut open = None;
    ui.horizontal_wrapped(|ui| {
        for (key, browse) in OPEN_WAYS {
            let button = ui
                .add_enabled(dialogs::AVAILABLE, egui::Button::new(app.t(key)))
                .on_disabled_hover_text(app.t("open.no_dialog.hint"));
            if button.clicked() {
                open = dialogs::pick(browse);
            }
        }
        // The address and the Connect button are the Live pane's (packet M7/E4).
        if ui.button(app.t("menu.attach")).clicked() {
            app.focus(Pane::AdvancedLive);
        }
    });
    ui.weak(app.t("shell.empty"));
    if let Some(path) = open {
        app.path = path.display().to_string();
        app.open();
    }
}

/// The new-project dialog: a name, a folder (typed, or the OS folder chooser), Create. What
/// the folder becomes, and whether Create is offered, are [`home::NewProject`]'s; a refusal is
/// shown in the dialog in a sentence. Creating opens the project, which starts at step three.
pub(crate) fn dialog(app: &mut EditorApp, ctx: &egui::Context) {
    let Some(mut d) = app.home.dialog.take() else {
        return;
    };
    let lang = app.settings.lang;
    let (mut cancel, mut created) = (false, None);
    let modal = egui::Modal::new(egui::Id::new("new-project")).show(ctx, |ui| {
        ui.set_width(720.0);
        ui.heading(app.t("home.new_title"));
        ui.label(word(lang, &d.template.name));
        ui.add_space(8.0);
        egui::Grid::new("new-project-fields")
            .num_columns(2)
            .spacing([12.0, 8.0])
            .show(ui, |ui| {
                ui.label(app.t("home.name"));
                let name = ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(560.0));
                if name.changed() {
                    d.name_changed();
                }
                ui.end_row();
                ui.label(app.t("home.folder"))
                    .on_hover_text(app.t("home.folder.hint"));
                ui.horizontal(|ui| {
                    let folder = ui
                        .add(egui::TextEdit::singleline(&mut d.folder).desired_width(440.0))
                        .on_hover_text(app.t("home.folder.hint"));
                    if folder.changed() {
                        d.folder_typed();
                    }
                    if let Some(picked) = app.browse(ui, Browse::Folder) {
                        d.folder_picked(Path::new(&picked));
                    }
                });
                ui.end_row();
            });
        if let Some(key) = d.blocker() {
            ui.colored_label(rgb(Mark::Missing.colour()), app.t(key));
        }
        if let Some(why) = &d.error {
            ui.colored_label(
                rgb(Mark::Missing.colour()),
                app.fill("home.create_failed", &[why]),
            );
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let create = egui::Button::new(app.t("home.create_button"));
            if ui.add_enabled(d.can_create(), create).clicked() {
                match d.create() {
                    Ok(project) => created = Some(project.root),
                    Err(e) => d.error = Some(e.to_string()),
                }
            }
            if ui.button(app.t("home.cancel")).clicked() {
                cancel = true;
            }
        });
    });
    if let Some(root) = created {
        app.path = root.display().to_string();
        app.open();
    } else if !(cancel || modal.should_close()) {
        app.home.dialog = Some(d);
    }
}
