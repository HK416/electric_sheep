//! Opening something: what a path on disk is, and the ones opened lately.
//!
//! Two decisions, both kept out of `app.rs` (spec 28.10 rule 3):
//!
//! * [`classify`] — a path is a `.esb` container, a directory of the five per-IR documents, or
//!   a finished run (spec 9.6, spec 14.3, spec 10.5). Told apart by **what is on disk**, not by
//!   a flag: someone who typed the wrong one gets the other view's error, not a mode. Every
//!   way of naming a path — the text field, the command line, the recent list, a dropped file
//!   — arrives here.
//! * [`Recent`] — the last ten, most recent first, no duplicates. `app.rs` persists it through
//!   `eframe::App::save` into `eframe::Storage` under [`RECENT_KEY`]; on Windows that file is
//!   `%APPDATA%\Electric Sheep editor\data\app.ron`.

use std::path::{Path, PathBuf};

use crate::model::run_view::RunView;

/// The `eframe::Storage` key the recent list is stored under.
pub const RECENT_KEY: &str = "es-editor.recent";

/// How many paths the list keeps. Ten is a menu someone can read at a glance.
pub const CAP: usize = 10;

/// What is at the end of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A `.esb` policy bundle, or any other single file (spec 9.6).
    Bundle,
    /// A directory of the per-IR TOML documents (spec 14.3).
    Documents,
    /// A finished `es eval run` / `es loop collect` directory (spec 10.5).
    Run,
}

/// Which of the three `path` is.
///
/// A run is recognised by [`RunView::is_run_dir`] — the same one call the Run tab opens with,
/// so the classification and the reader cannot drift apart. Anything that is not a directory
/// is a bundle: a file that turns out not to be one fails with the bundle reader's own error,
/// which says more than "not a directory" would.
pub fn classify(path: &Path) -> Kind {
    if RunView::is_run_dir(path) {
        Kind::Run
    } else if path.is_dir() {
        Kind::Documents
    } else {
        Kind::Bundle
    }
}

/// The paths opened lately, most recent first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Recent {
    pub paths: Vec<PathBuf>,
}

impl Recent {
    /// Moves `path` to the front, whether or not it was already there, and drops the oldest
    /// beyond [`CAP`].
    pub fn push(&mut self, path: &Path) {
        self.paths.retain(|p| p != path);
        self.paths.insert(0, path.to_path_buf());
        self.paths.truncate(CAP);
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.paths).unwrap_or_else(|_| "[]".to_owned())
    }

    /// Never fails: a store written by another version, or by a person with an editor, is
    /// worth less than the editor starting. Rebuilt through [`Recent::push`] so a hand-edited
    /// file cannot smuggle in duplicates or an eleventh entry.
    pub fn from_json(json: &str) -> Self {
        let paths: Vec<PathBuf> = serde_json::from_str(json).unwrap_or_default();
        let mut out = Self::default();
        for path in paths.iter().rev() {
            out.push(path);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{classify, Kind, Recent, CAP};

    use std::fs;
    use std::path::{Path, PathBuf};

    fn fixtures() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning")
    }

    /// Oracle 5: a `.esb`, a directory of the five documents, and a run directory are three
    /// different things, decided from disk.
    #[test]
    fn classify_tells_bundle_documents_and_run_apart() {
        let documents = fixtures();
        assert_eq!(classify(&documents), Kind::Documents);
        for name in [
            "task.toml",
            "observation.toml",
            "learning.toml",
            "deployment.toml",
            "evaluation.toml",
        ] {
            assert!(documents.join(name).is_file(), "{name} is one of the five");
        }

        let run = documents.join("run");
        assert!(run.join("report.json").is_file());
        assert_eq!(classify(&run), Kind::Run);

        // A `.esb` container is a file, and so is anything else that is not a directory.
        let tmp = std::env::temp_dir().join("es-editor-classify");
        fs::create_dir_all(&tmp).expect("a temporary directory");
        let esb = tmp.join("demo.esb");
        fs::write(&esb, b"not really a bundle").expect("write");
        assert_eq!(classify(&esb), Kind::Bundle);
        assert_eq!(classify(&documents.join("task.toml")), Kind::Bundle);
        // A directory that is neither is read as documents, and fails with that reader's
        // error rather than with a third kind nobody can open.
        assert_eq!(classify(&tmp), Kind::Documents);
        assert_eq!(classify(Path::new("no/such/path")), Kind::Bundle);
        fs::remove_dir_all(&tmp).ok();
    }

    /// Oracle 4b: capped at ten, deduplicated to the front, and the JSON round-trips.
    #[test]
    fn recent_is_capped_deduplicated_and_round_trips() {
        let mut recent = Recent::default();
        for i in 0..CAP + 2 {
            recent.push(Path::new(&format!("/bundles/b{i}")));
        }
        assert_eq!(recent.paths.len(), CAP, "capped");
        assert_eq!(
            recent.paths[0],
            PathBuf::from(&format!("/bundles/b{}", CAP + 1)),
            "most recent first"
        );
        assert!(
            !recent.paths.contains(&PathBuf::from("/bundles/b0")),
            "the oldest two fell off"
        );

        let before = recent.paths.len();
        let old = recent.paths[CAP - 1].clone();
        recent.push(&old);
        assert_eq!(recent.paths.len(), before, "a repeat adds no entry");
        assert_eq!(recent.paths[0], old, "and moves to the front");
        assert_eq!(
            recent.paths.iter().filter(|p| **p == old).count(),
            1,
            "deduplicated"
        );

        assert_eq!(Recent::from_json(&recent.to_json()), recent);
        assert_eq!(Recent::from_json("not json at all"), Recent::default());
        assert_eq!(Recent::from_json(""), Recent::default());
        // A store longer than the cap, or with repeats, is fixed on the way in.
        let overfull: Vec<String> = (0..CAP + 5).map(|_| "/bundles/same".to_owned()).collect();
        let repaired = Recent::from_json(&serde_json::to_string(&overfull).expect("json"));
        assert_eq!(repaired.paths, vec![PathBuf::from("/bundles/same")]);
    }
}
