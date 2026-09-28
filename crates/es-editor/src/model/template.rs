//! The templates a project starts from (`docs/design/editor-redesign.md` section 6.2, packet
//! M12/Y5): `templates/<id>.toml` at the repository root, each naming committed documents by
//! path. Nothing here copies a document - a scene path is hash input - so `es` runs with the
//! directory [`find_root`] returns as its working directory.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// How the robot is taught. `Blocks` is the scripted demonstrator (the only one S1 has).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Blocks,
}

/// The three training lengths a person picks from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Length {
    Short,
    Medium,
    Long,
}

/// The four documents `es policy init` builds the collect bundle from.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleDocs {
    pub task: String,
    pub observation: String,
    pub learning: String,
    pub deployment: String,
}

/// `[lengths]`: each preset is a `[run] checkpoint_at` series.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lengths {
    pub short: Vec<u32>,
    pub medium: Vec<u32>,
    pub long: Vec<u32>,
}

/// One `templates/<id>.toml`. `name` and `summary` are i18n keys; every path is relative to
/// the repository root.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    pub kind: String,
    pub id: String,
    pub name: String,
    pub summary: String,
    pub robot: String,
    pub method: Method,
    pub needs: Vec<String>,
    pub cycle: String,
    pub scene: String,
    pub demonstrations: u32,
    pub bundle: BundleDocs,
    pub lengths: Lengths,
}

const KIND: &str = "template";
const DIR: &str = "templates";

impl Template {
    /// The `checkpoint_at` series of a preset; its last element is the run's length.
    pub fn marks(&self, length: Length) -> &[u32] {
        match length {
            Length::Short => &self.lengths.short,
            Length::Medium => &self.lengths.medium,
            Length::Long => &self.lengths.long,
        }
    }
}

/// The directory holding `templates/`: the first ancestor of `exe_dir` that contains both
/// `templates/` and `Cargo.toml`. Pure, for the test; [`templates_root`] calls it with
/// `std::env::current_exe()`'s parent.
///
/// ponytail: templates are found only from a checkout (`target/<profile>/es-editor` sits under
/// the repository); an installable package that ships `templates/` beside the binary is a later
/// item.
pub fn find_root(exe_dir: Option<&Path>) -> Option<PathBuf> {
    exe_dir?
        .ancestors()
        .find(|d| d.join(DIR).is_dir() && d.join("Cargo.toml").is_file())
        .map(Path::to_path_buf)
}

pub fn templates_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    find_root(exe.parent())
}

/// Every `templates/*.toml` that parses, sorted by `id`, and every one that does not, with why.
pub fn load(root: &Path) -> (Vec<Template>, Vec<(PathBuf, String)>) {
    let dir = root.join(DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => return (Vec::new(), vec![(dir, e.to_string())]),
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "toml"))
        .collect();
    paths.sort();
    let (mut ok, mut bad) = (Vec::new(), Vec::new());
    for path in paths {
        let parsed = std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| toml::from_str::<Template>(&text).map_err(|e| e.to_string()))
            .and_then(|t| {
                if t.kind == KIND {
                    Ok(t)
                } else {
                    Err(format!(
                        "kind = {:?}; a template says kind = {KIND:?}",
                        t.kind
                    ))
                }
            });
        match parsed {
            Ok(t) => ok.push(t),
            Err(why) => bad.push((path, why)),
        }
    }
    ok.sort_by(|a: &Template, b: &Template| a.id.cmp(&b.id));
    (ok, bad)
}

#[cfg(test)]
mod tests {
    use super::{find_root, load, Length};
    use std::path::Path;

    #[test]
    fn the_committed_templates_parse_and_name_real_files() {
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
        let (ok, bad) = load(&root);
        assert!(bad.is_empty(), "{bad:?}");
        let cube = ok
            .iter()
            .find(|t| t.id == "cube-into-bin")
            .expect("the cube template");
        for p in [
            &cube.cycle,
            &cube.scene,
            &cube.bundle.task,
            &cube.bundle.observation,
            &cube.bundle.learning,
            &cube.bundle.deployment,
        ] {
            assert!(root.join(p).is_file(), "{p}");
        }
        for l in [Length::Short, Length::Medium, Length::Long] {
            let m = cube.marks(l);
            assert!(
                !m.is_empty() && m.windows(2).all(|w| w[0] < w[1]),
                "{l:?}: {m:?}"
            );
        }
        assert!(cube.marks(Length::Short).last() < cube.marks(Length::Long).last());
    }

    #[test]
    fn find_root_walks_up_and_gives_up_cleanly() {
        let tmp = std::env::temp_dir().join(format!("es-tpl-{}", std::process::id()));
        let deep = tmp.join("target").join("debug");
        std::fs::create_dir_all(&deep).unwrap();
        assert_eq!(find_root(Some(&deep)), None);
        std::fs::create_dir_all(tmp.join("templates")).unwrap();
        std::fs::write(tmp.join("Cargo.toml"), "").unwrap();
        assert_eq!(find_root(Some(&deep)), Some(tmp.clone()));
        assert_eq!(find_root(None), None);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_broken_template_is_reported_not_fatal() {
        let tmp = std::env::temp_dir().join(format!("es-tpl-bad-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("templates")).unwrap();
        std::fs::write(
            tmp.join("templates").join("x.toml"),
            "kind = \"template\"\n",
        )
        .unwrap();
        let (ok, bad) = load(&tmp);
        assert!(ok.is_empty());
        assert_eq!(bad.len(), 1);
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
