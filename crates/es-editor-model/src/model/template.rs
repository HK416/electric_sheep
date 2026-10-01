//! The templates a project starts from (`docs/design/editor-redesign.md` section 6.2, packet
//! M12/Y5): `templates/<id>.toml` at the repository root, each naming committed documents by
//! path. Nothing here copies a document - a scene path is hash input - so `es` runs with the
//! directory [`find_root`] returns as its working directory.
//!
//! Plan G (packet M17/G9) adds the templates without documents — `empty.toml`, and the ones the
//! person saves under their documents folder ([`load_saved`], [`write_saved`]) — and the
//! authored project, whose ② to ⑤ run its own generated documents in its own folder
//! ([`source`]).

use std::path::{Path, PathBuf};

use es_data::training::Cycle;
use serde::{Deserialize, Serialize};

use crate::model::project::Project;

/// How the robot is taught. `Blocks` is the scripted demonstrator; `Teacher` is a policy trained
/// and chosen in ② whose successful episodes are the demonstrations (packet M16/H7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Lengths {
    pub short: Vec<u32>,
    pub medium: Vec<u32>,
    pub long: Vec<u32>,
}

/// What kind of task `[outcome]` explains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
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
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Release {
    pub joint: String,
    pub above: f64,
}

/// `[editable]` (packet M17/G5): the template's scene as a scene document (`*.esscene`, the
/// robot included by reference) and, when one says the task, its task specification
/// (`*.estask`) — what ①'s "make an editable copy" copies into a project.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EditableDocs {
    pub scene: String,
    pub spec: Option<String>,
}

/// `viewport`: where the editor's outside camera starts on the template's scene, metres.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub eye: [f64; 3],
    pub look_at: [f64; 3],
}

/// One `templates/<id>.toml`. `name`, `summary` and `notice` (what the card warns about, such
/// as an experimental route) are i18n keys; every path is relative to the repository root.
///
/// A template without `bundle` (packet M17/G9: `empty.toml`, and every template the person
/// saves) has no documents of its own: its projects are editable from the start and run what
/// they generate ([`source`]); `cycle` is then empty and `robot` is the scene's.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Template {
    pub kind: String,
    pub id: String,
    pub name: String,
    pub summary: String,
    pub notice: Option<String>,
    #[serde(default)]
    pub robot: String,
    pub method: Method,
    pub needs: Vec<String>,
    #[serde(default)]
    pub cycle: String,
    pub scene: String,
    pub demonstrations: u32,
    pub bundle: Option<BundleDocs>,
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
    /// A template the person saved (packet M17/G9): the template its project was made from,
    /// which a project made from this one names in its `project.toml`. Absent: this one.
    pub base: Option<String>,
    /// Set on the template [`authored`] makes: its documents are a project's `generated/`, and
    /// its collect bundle is rebuilt from them before each run.
    #[serde(skip)]
    pub generated: bool,
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
        match read(&path) {
            Ok(t) => ok.push(t),
            Err(why) => bad.push((path, why)),
        }
    }
    ok.sort_by(|a: &Template, b: &Template| a.id.cmp(&b.id));
    (ok, bad)
}

fn read(path: &Path) -> Result<Template, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let t = toml::from_str::<Template>(&text).map_err(|e| e.to_string())?;
    if t.kind == KIND {
        Ok(t)
    } else {
        Err(format!(
            "kind = {:?}; a template says kind = {KIND:?}",
            t.kind
        ))
    }
}

// --- templates the person saves (packet M17/G9) ----------------------------------------------

/// The file in a saved template's folder; beside it are its documents. Spelt in two literals:
/// whole, it is shaped like a key of the `template` group (as `project::TEACH_FILE`).
pub const SAVED_FILE: &str = concat!("template", ".toml");

/// Every `<dir>/<folder>/template.toml` that parses and names its `base`, its document paths
/// made absolute under its folder, sorted by `id`; and every one that does not, with why.
pub fn load_saved(dir: &Path) -> (Vec<Template>, Vec<(PathBuf, String)>) {
    let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
    let mut paths: Vec<PathBuf> = (entries.map(|e| e.path().join(SAVED_FILE)))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    let (mut ok, mut bad) = (Vec::new(), Vec::new());
    for path in paths {
        let folder = path.parent().unwrap_or(dir).to_path_buf();
        let abs = |p: &mut String| *p = folder.join(&*p).display().to_string();
        match read(&path) {
            Ok(mut t) if t.base.is_some() => {
                abs(&mut t.scene);
                if let Some(e) = t.editable.as_mut() {
                    abs(&mut e.scene);
                    e.spec.iter_mut().for_each(abs);
                }
                ok.push(t);
            }
            Ok(_) => bad.push((path, "a saved template names its `base`".to_owned())),
            Err(why) => bad.push((path, why)),
        }
    }
    ok.sort_by(|a: &Template, b: &Template| a.id.cmp(&b.id));
    (ok, bad)
}

/// "Save as template": `<dir>/template.toml` for a project made from `base`, whose documents
/// `editable` names were copied into `dir` first (`es-editor-scene`'s `copy::documents`). It
/// keeps `base`'s words, needs, lengths and view under the person's `name`, has no documents
/// of its own, and names the template its projects are made from — `base`'s own base, so a
/// project never depends on a folder the person may delete. Written last: its presence is what
/// makes `dir` a template.
pub fn write_saved(
    base: &Template,
    name: &str,
    dir: &Path,
    editable: EditableDocs,
) -> Result<Template, String> {
    let folder = dir.file_name().unwrap_or_default().to_string_lossy();
    let saved = Template {
        id: format!("saved:{folder}"),
        name: name.to_owned(),
        summary: "template.saved.summary".to_owned(),
        notice: None,
        cycle: String::new(),
        scene: editable.scene.clone(),
        bundle: None,
        teach: None,
        teacher: None,
        outcome: None,
        editable: Some(editable),
        base: Some(base.base.clone().unwrap_or_else(|| base.id.clone())),
        generated: false,
        ..base.clone()
    };
    let path = dir.join(SAVED_FILE);
    let fail = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let text = toml::to_string(&saved).map_err(|e| fail(&e))?;
    std::fs::write(&path, text).map_err(|e| fail(&e))?;
    Ok(saved)
}

// --- what ② to ⑤ run (packet M17/G9) ---------------------------------------------------------

/// Why a project's steps ② to ⑤ cannot run: a key of the string tables and the one argument
/// its `{}` takes (empty when it takes none). The `watch.task.*` keys are mended in ①.
pub type Why = (&'static str, String);

/// What an editable project's `generated/` holds against its saved documents, as `es-editor`
/// found it with `es-editor-scene` (which runs the generator this crate does not have).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Generated {
    /// No task specification.
    NoSpec,
    /// What the saved documents generate: these files.
    Fresh(Vec<String>),
    /// ① has changes that are not saved.
    Unsaved,
    /// Not what the saved documents generate: never generated, or changed since.
    Stale,
    /// The saved documents do not generate: why.
    Failed(String),
}

/// The generated documents' folder in an authored project: `generate`'s `out`.
pub const GENERATED: &str = "generated";

/// The documents ② to ⑤ run for `project`, and the folder `es` runs in. `generated` is `None`
/// for a project that is not editable: its template's documents, in the repository root, as
/// before plan G. An editable project without a task of its own keeps its template's documents
/// when the template has some. With a task it runs what `generated/` holds, in its own root —
/// the generated documents name `scene.esscene` and the recipes' `runs/` relative to it — or
/// nothing, with why, while that set is not what its saved documents generate.
pub fn source(
    project: &Project,
    repo_root: Option<PathBuf>,
    generated: Option<&Generated>,
) -> Result<(Template, PathBuf), Why> {
    let root = repo_root.ok_or(("watch.no_checkout", String::new()))?;
    let id = &project.file.template;
    let template = (load(&root).0.into_iter())
        .find(|t| &t.id == id)
        .ok_or_else(|| ("watch.no_template", id.clone()))?;
    let why = |key: &'static str| Err((key, String::new()));
    match generated {
        None => Ok((template, root)),
        Some(Generated::NoSpec) if template.bundle.is_some() => Ok((template, root)),
        Some(Generated::NoSpec) => why("watch.task.none"),
        Some(Generated::Unsaved) => why("watch.task.unsaved"),
        Some(Generated::Stale) => why("watch.task.stale"),
        Some(Generated::Failed(e)) => Err(("watch.task.failed", e.clone())),
        Some(Generated::Fresh(files)) => {
            let t = authored(&template, &project.root, files)?;
            Ok((t, project.root.clone()))
        }
    }
}

/// `base` with an authored project's documents, relative to its `root`: the student arm of
/// `generated/` as the collect bundle, its teacher when there is one, its cycle and its scene.
/// The trained teacher demonstrates unless the cycle names an expert. `base` keeps its words,
/// needs, lengths and view; its outcome is its own task's, so it is not carried over.
pub fn authored(base: &Template, root: &Path, files: &[String]) -> Result<Template, Why> {
    let cycle_file = (files.iter().find(|f| f.starts_with("cycle-")))
        .ok_or(("watch.task.no_cycle", String::new()))?;
    let at = |f: &str| format!("{GENERATED}/{f}");
    let path = root.join(at(cycle_file));
    let cycle = (std::fs::read_to_string(&path).map_err(|e| e.to_string()))
        .and_then(|text| Cycle::parse(&text).map_err(|e| e.to_string()))
        .map_err(|e| ("watch.task.failed", format!("{}: {e}", path.display())))?;
    let name = cycle_file
        .trim_start_matches("cycle-")
        .trim_end_matches(".toml");
    let arm = |kind: &str| at(&format!("{kind}-{name}.toml"));
    let teacher = files
        .iter()
        .any(|f| f == "training-teacher.toml")
        .then(|| TeacherDocs {
            recipe: at("training-teacher.toml"),
            evaluation: at("evaluation-teacher.toml"),
            task: at("task.toml"),
            observation: at("observation-teacher.toml"),
            learning: at("learning-teacher.toml"),
            deployment: at("deployment-teacher.toml"),
            jobs: cycle.eval.jobs,
        });
    let collect = cycle.collect.as_ref();
    let by_teacher = teacher.is_some() && collect.is_some_and(|c| c.expert.is_none());
    Ok(Template {
        method: if by_teacher {
            Method::Teacher
        } else {
            Method::Blocks
        },
        cycle: at(cycle_file),
        scene: cycle.scene.clone(),
        demonstrations: collect.map_or(base.demonstrations, |c| c.episodes),
        bundle: Some(BundleDocs {
            task: at("task.toml"),
            observation: arm("observation"),
            learning: Some(arm("learning")),
            deployment: arm("deployment"),
        }),
        teacher,
        teach: None,
        outcome: None,
        generated: true,
        ..base.clone()
    })
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
            let b = cube.bundle.as_ref().expect("[bundle]");
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
        let b = hand.bundle.as_ref().expect("[bundle]");
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

    /// The empty card (packet M17/G9): no documents of its own — no bundle, cycle, teacher or
    /// outcome — the empty scene as its scene document, no GPU simulator needed to build one,
    /// and its words in both tables.
    #[test]
    fn the_empty_template_has_no_documents_of_its_own() {
        use super::Method;
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
        let empty = (load(&root).0.into_iter())
            .find(|t| t.id == "empty")
            .expect("the empty template");
        assert!(empty.bundle.is_none() && empty.teacher.is_none() && empty.outcome.is_none());
        assert!(empty.cycle.is_empty() && empty.teach.is_none() && empty.base.is_none());
        assert_eq!(empty.method, Method::Teacher);
        let e = empty.editable.as_ref().expect("[editable]");
        let scene = "tests/fixtures/esscene/empty.esscene";
        assert_eq!((e.scene.as_str(), e.spec.as_deref()), (scene, None));
        assert_eq!(empty.scene, scene);
        assert!(root.join(scene).is_file());
        assert!(!empty.needs.iter().any(|n| n == "mjwarp"));
        let words = ["template.empty.name", "template.empty.summary"];
        assert_eq!([empty.name.as_str(), empty.summary.as_str()], words);
        for lang in Lang::ALL {
            for key in words {
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
