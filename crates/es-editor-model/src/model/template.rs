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

/// The documents `es policy init` builds the collect bundle from. No `learning` is spec 8.1's
/// external-policy shape (packet M12/Y5b): the Observation IR has no committed Learning IR.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleDocs {
    pub task: String,
    pub observation: String,
    pub learning: Option<String>,
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

/// `[outcome]` (packet M13/Z4, Z6): what a timed-out attempt is judged by - where the `object`
/// body went against the region the scene geoms of the `target` stem cover
/// ([`crate::model::outcome`]). `object_name` and `target_name` are i18n keys: the two things in
/// the words a failure cause names them by.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSpec {
    pub object: String,
    pub target: String,
    /// The height gain, in metres, that counts as lifted.
    pub lift_m: f64,
    pub object_name: String,
    pub target_name: String,
    /// Absent: an object that ends inside the target is always *too late* (packet M13/R4).
    pub release: Option<Release>,
}

/// `[outcome] release` (packet M13/R4): the scene joint whose opening lets the object go, and
/// the position it must pass for the task to count the object as let go.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Release {
    pub joint: String,
    pub above: f64,
}

/// One `templates/<id>.toml`. `name`, `summary` and `notice` (what the card warns about, such
/// as an experimental route) are i18n keys; every path is relative to the repository root.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    pub kind: String,
    pub id: String,
    pub name: String,
    pub summary: String,
    pub notice: Option<String>,
    pub robot: String,
    pub method: Method,
    pub needs: Vec<String>,
    pub cycle: String,
    pub scene: String,
    pub demonstrations: u32,
    pub bundle: BundleDocs,
    /// The demonstration program a project starts from (packet M14/Q3), which
    /// [`Project::create`](crate::model::project::Project::create) copies into the project as
    /// `teach.toml`. Absent: the cycle's `[collect] expert` name is used as it is.
    pub teach: Option<String>,
    pub lengths: Lengths,
    /// Absent: failures keep the causes the evaluation recorded.
    pub outcome: Option<OutcomeSpec>,
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
    use crate::model::i18n::{Lang, Strings};
    use std::path::Path;

    /// Both cube cards (packet M12/Y5b): the camera-only one builds its collect bundle without a
    /// Learning IR, the cube-pose one with `learning.toml`, and each names its words by keys
    /// both tables hold.
    #[test]
    fn the_committed_templates_parse_and_name_real_files() {
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
        let (ok, bad) = load(&root);
        assert!(bad.is_empty(), "{bad:?}");
        let cards = [
            (
                "cube-into-bin",
                false,
                [
                    "template.cube_into_bin.name",
                    "template.cube_into_bin.summary",
                    "template.cube_into_bin.notice",
                ],
            ),
            (
                "cube-into-bin-hint",
                true,
                [
                    "template.cube_into_bin_hint.name",
                    "template.cube_into_bin_hint.summary",
                    "template.cube_into_bin_hint.notice",
                ],
            ),
        ];
        for (id, has_learning, [name, summary, notice]) in cards {
            let cube = ok.iter().find(|t| t.id == id).expect(id);
            let b = &cube.bundle;
            assert_eq!(b.learning.is_some(), has_learning, "{id}");
            for p in [
                &cube.cycle,
                &cube.scene,
                &b.task,
                &b.observation,
                &b.deployment,
            ]
            .into_iter()
            .chain(b.learning.as_ref())
            {
                assert!(root.join(p).is_file(), "{id}: {p}");
            }
            // Packet M14/Q3: both cards teach with the committed built-in program.
            let teach = std::fs::read_to_string(root.join(cube.teach.as_ref().expect("teach")))
                .expect("the program file");
            assert_eq!(teach, es_env::program::SO101_PICK_PLACE, "{id}");
            assert_eq!(
                (
                    cube.name.as_str(),
                    cube.summary.as_str(),
                    cube.notice.as_deref()
                ),
                (name, summary, Some(notice)),
                "{id}"
            );
            for lang in Lang::ALL {
                for key in [name, summary, notice] {
                    assert_ne!(Strings::get(lang).t(key), key, "{id}: {key}");
                }
            }
            for l in [Length::Short, Length::Medium, Length::Long] {
                let m = cube.marks(l);
                assert!(
                    !m.is_empty() && m.windows(2).all(|w| w[0] < w[1]),
                    "{id} {l:?}: {m:?}"
                );
            }
            assert!(cube.marks(Length::Short).last() < cube.marks(Length::Long).last());
            // Packet M13/Z6: the cube, the bin, two centimetres, named in both languages.
            let o = cube.outcome.as_ref().expect("[outcome]");
            assert_eq!(
                (o.object.as_str(), o.target.as_str(), o.lift_m),
                ("cube", "bin", 0.02),
                "{id}"
            );
            assert_eq!(
                (o.object_name.as_str(), o.target_name.as_str()),
                ("outcome.cube", "outcome.bin"),
                "{id}"
            );
            // Packet M13/R4: let go = the gripper past the Task IR's own 0.85 rad.
            let r = o.release.as_ref().expect("[outcome] release");
            assert_eq!((r.joint.as_str(), r.above), ("gripper", 0.85), "{id}");
            for lang in Lang::ALL {
                for key in [&o.object_name, &o.target_name] {
                    assert_ne!(Strings::get(lang).t(key), key, "{id}: {key}");
                }
            }
        }
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
