//! The operating system's own Open dialog (packet M7/E6).
//!
//! Typing a path is fine for whoever knows where the file is. Everyone else expects the
//! window their OS opens for every other program, so `rfd` - **the one dependency this packet
//! adds**, MIT, native on both targets - is wrapped here and named nowhere else: `app.rs`
//! calls [`pick`] and cannot see the crate, which keeps the choice of dialog a decision of
//! `model/` like every other (spec 28.10 rule 3).
//!
//! It is declared under `[target.'cfg(any(windows, target_os = "macos"))'.dependencies]`
//! behind the `file-dialogs` feature, and every function here has a `cfg`-off twin returning
//! `None`. On Linux that twin is what compiles: `rfd`'s Linux backends want GTK or an XDG
//! portal, and a viewer that a headless CI cannot even `cargo check` would cost more than a
//! file dialog is worth there (spec 26.1 makes Linux the primary platform). The typed path,
//! the recent list and drag-and-drop are unchanged on every target, so nothing is *only*
//! reachable through a dialog - [`AVAILABLE`] merely hides a button that would do nothing.

use std::path::PathBuf;

use crate::model::labels::Browse;

/// Whether this build has a native dialog. `false` on Linux and in any
/// `--no-default-features` build; `app.rs` draws no Browse button then and says why in the
/// hover instead.
pub const AVAILABLE: bool = cfg!(all(
    feature = "file-dialogs",
    any(windows, target_os = "macos")
));

/// The dialog `browse` asks for: a folder chooser for [`Browse::Folder`], a file chooser
/// filtered to its extensions otherwise.
///
/// `None` is every ordinary refusal - cancelled, or no dialog in this build - and means the
/// field keeps whatever was typed in it.
pub fn pick(browse: Browse) -> Option<PathBuf> {
    match browse {
        Browse::Folder => pick_dir(),
        _ => pick_file(browse.filter()),
    }
}

/// A file, filtered to `(name, extensions)`.
#[cfg(all(feature = "file-dialogs", any(windows, target_os = "macos")))]
pub fn pick_file(filter: (&str, &[&str])) -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter(filter.0, filter.1)
        .pick_file()
}

/// A directory.
#[cfg(all(feature = "file-dialogs", any(windows, target_os = "macos")))]
pub fn pick_dir() -> Option<PathBuf> {
    rfd::FileDialog::new().pick_folder()
}

/// The `cfg`-off twin: no dialog, so nothing is picked.
#[cfg(not(all(feature = "file-dialogs", any(windows, target_os = "macos"))))]
pub fn pick_file(_filter: (&str, &[&str])) -> Option<PathBuf> {
    None
}

/// The `cfg`-off twin of [`pick_dir`].
#[cfg(not(all(feature = "file-dialogs", any(windows, target_os = "macos"))))]
pub fn pick_dir() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::{pick_file, AVAILABLE};

    use crate::model::labels::{browses, Browse};
    use crate::model::launch::LaunchField;

    /// The stub is the shape of the real thing, and the build that has no dialog says so
    /// rather than offering a button that does nothing. A dialog cannot be opened in a test,
    /// so what is judged here is the wiring: every browsable field reaches a filter, and
    /// [`AVAILABLE`] agrees with which half of the `cfg` compiled.
    #[test]
    fn the_dialog_is_available_exactly_where_it_is_compiled_in() {
        assert_eq!(
            AVAILABLE,
            cfg!(all(feature = "file-dialogs", any(windows, target_os = "macos"))),
            "AVAILABLE is the cfg, not a guess"
        );
        if !AVAILABLE {
            assert!(pick_file(Browse::Policy.filter()).is_none(), "the stub picks nothing");
        }
        // Every field that offers a Browse button has something to filter on, or is a folder.
        for field in LaunchField::ALL {
            let Some(browse) = browses(field) else { continue };
            if browse == Browse::Folder {
                continue;
            }
            let (_, extensions) = browse.filter();
            assert!(!extensions.is_empty(), "{field:?} filters on something");
        }
    }
}
