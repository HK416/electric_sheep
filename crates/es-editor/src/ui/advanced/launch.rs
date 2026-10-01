//! The Results pane's Launch section (packet M7/E5, spec 23.1): the run the editor starts as a
//! child process and attaches to.

use eframe::egui;
use egui::Vec2;

use crate::app::EditorApp;
use crate::model::labels;
use crate::model::launch::{Kind as LaunchKind, State as LaunchState};

/// Width of the Launch section's flag labels, so the text boxes line up. Wider than the
/// flags it replaced: a plain name is longer than `--jobs` and may be Korean (packet M7/E6).
const FLAG_LABEL: Vec2 = Vec2::new(210.0, 18.0);

impl EditorApp {
    /// The Launch section (packet M7/E5, spec 23.1): the editor is a **client**, so this
    /// starts a child process and attaches to it — it hosts nothing.
    ///
    /// Wiring only. Which flags the chosen kind has, what each is called, what the command
    /// line reads as, what the exit code means and how long to wait for the producer's socket
    /// are all [`LaunchModel`]'s, under test (spec 28.10 rule 3). There is no Pause and no
    /// Step: the run speaks no control protocol (spec 23.3), so Kill is the only control.
    pub(super) fn launch_panel(&mut self, ui: &mut egui::Ui) {
        let lang = self.settings.lang;
        let mut start = false;
        let running = matches!(self.launch.state(), LaunchState::Running { .. });
        ui.horizontal(|ui| {
            // The selector says what the command *does*; the hover says what it is, which is
            // the command line itself (packet M7/E6).
            for kind in LaunchKind::ALL {
                ui.selectable_value(&mut self.launch.kind, kind, labels::kind_label(lang, kind))
                    .on_hover_text(kind.label());
            }
            ui.separator();
            let missing = self.launch.missing_required();
            start = ui
                .add_enabled(
                    !running && missing.is_empty(),
                    egui::Button::new(self.t("launch.start")),
                )
                .clicked();
            if ui
                .add_enabled(running, egui::Button::new(self.t("launch.stop")))
                .clicked()
            {
                self.launch.kill();
            }
            ui.separator();
            if missing.is_empty() {
                ui.label(self.launch.status_line());
            } else {
                // What to fill in, by the panel's own words for the fields (packet M7/E6).
                let names: Vec<&str> = missing
                    .iter()
                    .map(|f| labels::launch_label(lang, *f))
                    .collect();
                ui.label(format!("{} {}", self.t("launch.missing"), names.join(", ")));
            }
        });
        // One `horizontal` per flag rather than an `egui::Grid`: a grid caps a cell at the
        // column width it measured last frame, which squeezes a `TextEdit` down to the
        // default interact size and never lets it grow back.
        let mut picked: Option<(crate::model::launch::LaunchField, String)> = None;
        for field in self.launch.fields() {
            ui.horizontal(|ui| {
                // The label is the plain name and the hover is the flag the CLI is given, so
                // the panel still tells anyone who asks exactly what it will type.
                ui.add_sized(
                    FLAG_LABEL,
                    egui::Label::new(labels::launch_label(lang, *field)),
                )
                .on_hover_text(field.flag());
                ui.add(
                    egui::TextEdit::singleline(self.launch.field_mut(*field))
                        .hint_text(field.hint())
                        .desired_width(540.0),
                );
                if let Some(browse) = labels::browses(*field) {
                    if let Some(path) = self.browse(ui, browse) {
                        picked = Some((*field, path));
                    }
                }
            });
        }
        if let Some((field, path)) = picked {
            *self.launch.field_mut(field) = path;
        }
        ui.horizontal(|ui| {
            for flag in self.launch.flags() {
                let label = labels::launch_flag_label(lang, *flag);
                let raw = flag.flag();
                ui.checkbox(self.launch.flag_mut(*flag), label)
                    .on_hover_text(raw);
            }
        });
        // The command line, read-only: what is about to run, in one place, so nobody has to
        // guess which `es` or which flags the panel decided on. It is `es`'s own spelling and
        // stays untranslated - this is the line a person would paste into a terminal.
        ui.label(self.t("launch.command"));
        let mut command = self.launch.command_line();
        ui.add(
            egui::TextEdit::singleline(&mut command)
                .desired_width(f32::INFINITY)
                .interactive(false),
        );
        egui::ScrollArea::vertical()
            .id_salt("launch-lines")
            // A box of its own height inside the form's scroll, not one that fills it.
            .max_height(160.0)
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for line in self.launch.lines() {
                    ui.monospace(line);
                }
            });
        if start {
            self.start_launch();
        }
    }

    /// Start; the attach follows through [`LaunchModel::take_attached`] in `update`, when
    /// the dial the model runs on its own thread has answered. Never here: on a machine where
    /// a refused connect takes seconds, a dial on this thread held the whole window.
    fn start_launch(&mut self) {
        self.launch.start();
        self.status = self.launch.status_line();
    }
}
