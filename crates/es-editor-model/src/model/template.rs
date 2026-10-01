//! The templates a project starts from (`docs/design/editor-redesign.md` section 6.2, packet
//! M12/Y5): `templates/<id>.toml` at the repository root, each naming committed documents by
//! path. Nothing here copies a document - a scene path is hash input - so `es` runs with the
//! directory [`find_root`] returns as its working directory.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// How the robot is taught. `Blocks` is the scripted demonstrator; `Teacher` is a policy trained
/// and chosen in ② whose successful episodes are the demonstrations (packet M16/H7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    Blocks,
    Teacher,
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

/// `[teacher]` (packet M16/H7): the teacher's training recipe, the Evaluation IR its checkpoints
/// are judged by (with `jobs` workers) and the four documents its bundle is built and re-packed
/// on.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherDocs {
    pub recipe: String,
    pub evaluation: String,
    pub task: String,
    pub observation: String,
    pub learning: String,
    pub deployment: String,
    #[serde(default = "one")]
    pub jobs: u32,
}

fn one() -> u32 {
    1
}

/// `[lengths]`: each preset is a `[run] checkpoint_at` series.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lengths {
    pub short: Vec<u32>,
    pub medium: Vec<u32>,
    pub long: Vec<u32>,
}

/// What kind of task `[outcome]` explains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutcomeKind {
    /// Put the object in the target (the cube cards).
    #[default]
    Place,
    /// Turn the object to the target body's orientation (packet M16/H7).
    Reorient,
}

/// `[outcome]` (packet M13/Z4, Z6): what a timed-out attempt is judged by - where the `object`
/// body went against the region the scene geoms of the `target` stem cover
/// ([`crate::model::outcome`]); for `kind = "reorient"`, how far the `object` body's orientation
/// ended from the `target` body's. `object_name` and `target_name` are i18n keys: the two things
/// in the words a failure cause names them by.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSpec {
    #[serde(default)]
    pub kind: OutcomeKind,
    pub object: String,
    pub target: String,
    /// The height gain, in metres, that counts as lifted (`place`).
    #[serde(default)]
    pub lift_m: f64,
    /// The angle under which the task counts the object turned, radians (`reorient`).
    #[serde(default)]
    pub angle_rad: f64,
    /// How far from where it belongs the object counts as dropped, metres (`reorient`).
    #[serde(default)]
    pub drop_m: f64,
    /// The share of attempts that succeeds by chance, when it was measured.
    pub chance: Option<f64>,
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

/// `[editable]` (packet M17/G5): the template's scene as a scene document (`*.esscene`, the
/// robot included by reference) and, when one says the task, its task specification
/// (`*.estask`) — what ①'s "make an editable copy" copies into a project.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditableDocs {
    pub scene: String,
    pub spec: Option<String>,
}

/// `viewport`: where the editor's outside camera starts on the template's scene, metres.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub eye: [f64; 3],
    pub look_at: [f64; 3],
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
    /// `method = "teacher"`'s teacher (packet M16/H7).
    pub teacher: Option<TeacherDocs>,
    /// Absent: the demo's showcase view ([`crate::model::scene_view::SHOWCASE_CAMERA`]).
    pub viewport: Option<Viewport>,
    pub lengths: Lengths,
    /// Absent: failures keep the causes the evaluation recorded.
    pub outcome: Option<OutcomeSpec>,
    /// Absent: ① offers no editable copy.
    pub editable: Option<EditableDocs>,
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
        // The robot library of ①'s Add menu (packet M17/G7) is not a template.
        .filter(|p| !p.ends_with("robots.toml"))
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
            // Packet M17/G5: the scene as a scene document, and no task specification yet.
            let editable = cube.editable.as_ref().expect("[editable]");
            assert!(editable.spec.is_none(), "{id}");
            for p in [
                &cube.cycle,
                &cube.scene,
                &b.task,
                &b.observation,
                &b.deployment,
                &editable.scene,
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

    /// The hand card (packet M16/H7): taught by a teacher policy, every document it names is
    /// committed, its outcome is a reorientation, and its words are in both tables.
    #[test]
    fn the_hand_template_names_its_teacher_and_a_reorientation() {
        use super::{Method, OutcomeKind};
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
        let hand = (load(&root).0.into_iter())
            .find(|t| t.id == "shadow-hand-repose")
            .expect("the hand template");
        assert_eq!(hand.method, Method::Teacher);
        assert!(hand.teach.is_none() && hand.needs.iter().any(|n| n == "mjwarp"));
        let t = hand.teacher.as_ref().expect("[teacher]");
        let b = &hand.bundle;
        // Packet M17/G5: its scene document and task specification, for an editable copy.
        let editable = hand.editable.as_ref().expect("[editable]");
        let spec = editable.spec.as_ref().expect("a task specification");
        for p in [&editable.scene, spec] {
            assert!(root.join(p).is_file(), "{p}");
        }
        for p in [
            &hand.cycle,
            &hand.scene,
            &b.task,
            &b.observation,
            &b.deployment,
            &t.recipe,
            &t.evaluation,
            &t.task,
            &t.observation,
            &t.learning,
            &t.deployment,
        ]
        .into_iter()
        .chain(b.learning.as_ref())
        {
            assert!(root.join(p).is_file(), "{p}");
        }
        let o = hand.outcome.as_ref().expect("[outcome]");
        assert_eq!(o.kind, OutcomeKind::Reorient);
        assert_eq!(
            (
                hand.name.as_str(),
                hand.summary.as_str(),
                hand.notice.as_deref(),
                o.target_name.as_str()
            ),
            (
                "template.shadow_hand.name",
                "template.shadow_hand.summary",
                Some("template.shadow_hand.notice"),
                "outcome.goal"
            )
        );
        assert_eq!((o.angle_rad, o.drop_m, o.chance), (0.1, 0.24, Some(0.06)));
        assert_eq!(hand.marks(Length::Long), [1000, 5000, 20000, 60000]);
        for lang in Lang::ALL {
            let keys = [&hand.name, &hand.summary, &o.object_name, &o.target_name];
            for key in keys.into_iter().chain(hand.notice.as_ref()) {
                assert_ne!(Strings::get(lang).t(key), key, "{key}");
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
