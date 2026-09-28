//! The drawing half of the editor (packet M12/Y10): [`shell`] is the menu bar, the step bar,
//! the dock and the status line; [`advanced`] is the five tabs the editor had before the dock;
//! [`home`] is the start screen and the new-project dialog (packet M12/Y11).
//!
//! Compiled only, never run by CI (`docs/design/editor-shell.md` section 2): every branch that
//! chooses what to show is in [`crate::model`], under test. This half turns answers into
//! widgets.

pub mod advanced;
pub mod home;
pub mod results;
pub mod shell;

use eframe::egui;

use crate::app::EditorApp;
use crate::model::dialogs;
use crate::model::labels::Browse;

impl EditorApp {
    /// A **Browse...** button beside a path field (packet M7/E6): the OS's own dialog, which
    /// is what someone who does not know the path expects. `Some` only when a path was
    /// picked, so a cancelled dialog leaves the field exactly as it was.
    ///
    /// Which dialog, which filter and what the button is called are all
    /// [`crate::model::labels`]'s and [`dialogs`]'s. A build without one (Linux, or
    /// `--no-default-features`) draws the button disabled and says why on its hover, rather
    /// than hiding it and leaving someone looking for it.
    pub(crate) fn browse(&self, ui: &mut egui::Ui, browse: Browse) -> Option<String> {
        let label = self.t(browse.label_key());
        if !dialogs::AVAILABLE {
            ui.add_enabled(false, egui::Button::new(label))
                .on_disabled_hover_text(self.t("open.no_dialog.hint"));
            return None;
        }
        ui.button(label)
            .clicked()
            .then(|| dialogs::pick(browse))
            .flatten()
            .map(|path| path.display().to_string())
    }

    /// The Browse button's own words, for the one place a destructured `self` puts
    /// [`Self::browse`] out of reach (the Replay panel).
    pub(crate) fn browse_label(&self, browse: Browse) -> &'static str {
        self.t(browse.label_key())
    }
}
