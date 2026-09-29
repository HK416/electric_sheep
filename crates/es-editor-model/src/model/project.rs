//! A project folder and its runs (`docs/design/editor-redesign.md` section 6.2, packet M12/Y6).
//!
//! ```text
//! my-project/
//!   project.toml      kind = "project", name, template
//!   untrained.esb     the collect bundle, from the template's `[bundle]` documents
//!   runs/001/         exactly what `es loop cycle --out runs/001` writes, plus
//!     cycle.toml      the recipe this run used, written just before launch
//!     telemetry.txt   the live address, so a re-opened editor can attach again
//! ```
//!
//! The editor runs no learning (spec 23.1): it writes a recipe and hands `es` an argv. The
//! template's documents are named by path, never copied (a scene path is hash input), so `es`
//! runs with the repository root as its working directory and every path this file writes
//! into a recipe or an argv is absolute.

use std::path::{Path, PathBuf};

use es_data::training::{Cycle, Recipe, TrainRef};
use serde::{Deserialize, Serialize};

use crate::model::template::{Length, Template};

pub const PROJECT_FILE: &str = "project.toml";
pub const RUNS_DIR: &str = "runs";
pub const RUN_RECIPE: &str = "cycle.toml";
pub const TELEMETRY_FILE: &str = "telemetry.txt";
pub const COLLECT_BUNDLE: &str = "untrained.esb";

const KIND: &str = "project";

/// The seed `es policy init` defaults to. The collect bundle's weights are never loaded under
/// `--expert`; the seed only names the placeholder.
const BUNDLE_SEED: u64 = 0;

/// Why a project could not be made, opened or given a run. One sentence, shown as it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectError(pub String);

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProjectError {}

fn err(path: &Path, e: impl std::fmt::Display) -> ProjectError {
    ProjectError(format!("{}: {e}", path.display()))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectFile {
    pub kind: String,
    pub name: String,
    pub template: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    /// Absolute: `es` runs in the repository root, so a relative root would name another place.
    pub root: PathBuf,
    pub file: ProjectFile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunFolder {
    pub number: u32,
    pub path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartSettings {
    pub demonstrations: u32,
    pub length: Length,
}

fn absolute(path: &Path) -> Result<PathBuf, ProjectError> {
    std::path::absolute(path).map_err(|e| err(path, e))
}

fn write(path: &Path, bytes: impl AsRef<[u8]>) -> Result<(), ProjectError> {
    std::fs::write(path, bytes).map_err(|e| err(path, e))
}

impl Project {
    /// Makes `root` (refusing one that already holds a `project.toml`), writes `project.toml`
    /// and builds `untrained.esb` from the template's `[bundle]` documents.
    ///
    /// The bundle is built before anything is written, and `project.toml` is written last:
    /// its presence is what makes a folder a project, so a failure leaves no half project.
    pub fn create(
        root: &Path,
        name: &str,
        template: &Template,
        repo_root: &Path,
    ) -> Result<Self, ProjectError> {
        let root = absolute(root)?;
        if Self::is_project_dir(&root) {
            return Err(err(&root.join(PROJECT_FILE), "already a project"));
        }
        let b = &template.bundle;
        let doc = |p: &String| repo_root.join(p);
        let bundle = es_data::training::untrained_bundle(
            &doc(&b.task),
            &doc(&b.observation),
            b.learning.as_ref().map(doc).as_deref(),
            &doc(&b.deployment),
            BUNDLE_SEED,
        )
        .map_err(|e| ProjectError(format!("template {}: {e}", template.id)))?;
        std::fs::create_dir_all(&root).map_err(|e| err(&root, e))?;
        write(&root.join(COLLECT_BUNDLE), bundle)?;
        let file = ProjectFile {
            kind: KIND.to_owned(),
            name: name.to_owned(),
            template: template.id.clone(),
        };
        let text = toml::to_string(&file).map_err(|e| err(&root.join(PROJECT_FILE), e))?;
        write(&root.join(PROJECT_FILE), text)?;
        Ok(Self { root, file })
    }

    pub fn open(root: &Path) -> Result<Self, ProjectError> {
        let root = absolute(root)?;
        let path = root.join(PROJECT_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| err(&path, e))?;
        let file: ProjectFile = toml::from_str(&text).map_err(|e| err(&path, e))?;
        if file.kind != KIND {
            return Err(err(
                &path,
                format!("kind = {:?}; a project says kind = {KIND:?}", file.kind),
            ));
        }
        Ok(Self { root, file })
    }

    pub fn is_project_dir(path: &Path) -> bool {
        path.join(PROJECT_FILE).is_file()
    }

    /// The collect bundle `create` built.
    pub fn bundle(&self) -> PathBuf {
        self.root.join(COLLECT_BUNDLE)
    }

    /// `runs/NNN` directories, ascending; anything else under `runs/` is ignored.
    pub fn runs(&self) -> Vec<RunFolder> {
        let Ok(entries) = std::fs::read_dir(self.root.join(RUNS_DIR)) else {
            return Vec::new();
        };
        let mut runs: Vec<RunFolder> = entries
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let number = name
                    .bytes()
                    .all(|b| b.is_ascii_digit())
                    .then(|| name.parse().ok())??;
                Some(RunFolder {
                    number,
                    path: e.path(),
                })
            })
            .collect();
        runs.sort_by_key(|r| r.number);
        runs
    }

    pub fn latest_run(&self) -> Option<RunFolder> {
        self.runs().pop()
    }

    /// One past the largest existing number, three digits; never an existing directory.
    pub fn next_run_dir(&self) -> PathBuf {
        let mut n = self.latest_run().map_or(1, |r| r.number + 1);
        loop {
            let path = self.root.join(RUNS_DIR).join(format!("{n:03}"));
            if !path.exists() {
                return path;
            }
            n += 1;
        }
    }
}

impl RunFolder {
    pub fn eval_dir(&self) -> PathBuf {
        self.path.join("eval")
    }

    pub fn report_path(&self) -> PathBuf {
        self.eval_dir().join("report.json")
    }

    /// The address the run was started with, if `telemetry.txt` holds one.
    pub fn telemetry_addr(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.path.join(TELEMETRY_FILE)).ok()?;
        Some(text.trim().to_owned()).filter(|a| !a.is_empty())
    }
}

fn read(path: &Path) -> Result<String, ProjectError> {
    std::fs::read_to_string(path).map_err(|e| err(path, e))
}

fn arg(path: &Path) -> String {
    path.display().to_string()
}

/// Writes `run/cycle.toml`: the template's recipe with `[collect] episodes`, `[collect]
/// policy` (absolute path of the project's bundle) and an inline `[train]` whose
/// `[run] checkpoint_at` is the preset and `steps` its last mark. Everything else is the
/// template's, unchanged. Returns the argv for `es` (no program name).
///
/// The inline `[train]` is the template's training recipe with `[dataset]` left out (the
/// cycle overrides it with its own collect output) and two more words changed: `[run]`'s
/// length, and — on the IR route — `[policy] bundle`, which the committed
/// `training.toml` names as `runs/collect-001/untrained.esb`, a file a checkout does not have.
/// The project's bundle is built from the same four documents, so it is the architecture that
/// recipe trains. The `LeRobot` route names its documents instead of a bundle and keeps them.
///
/// `run` is created here and must not exist yet ([`Project::next_run_dir`]): a number is
/// never reused. `telemetry.txt` is written beside the recipe.
pub fn write_run(
    template: &Template,
    repo_root: &Path,
    project: &Project,
    settings: StartSettings,
    run: &Path,
    telemetry: &str,
) -> Result<Vec<String>, ProjectError> {
    let cycle_path = repo_root.join(&template.cycle);
    let mut cycle = Cycle::parse(&read(&cycle_path)?).map_err(|e| err(&cycle_path, e))?;
    let recipe = match &cycle.train.recipe {
        Some(p) => {
            let path = repo_root.join(p);
            Recipe::parse(&read(&path)?).map_err(|e| err(&path, e))?
        }
        None => cycle.training(None, run).map_err(|e| err(&cycle_path, e))?,
    };
    if recipe.init.is_some() || recipe.rl.is_some() {
        return Err(err(
            &cycle_path,
            "its training recipe has [init] or [rl], which an inline [train] cannot carry",
        ));
    }
    let marks = template.marks(settings.length).to_vec();
    let steps = *marks
        .last()
        .ok_or_else(|| ProjectError(format!("template {}: an empty length", template.id)))?;
    let bundle = arg(&project.bundle());
    let mut policy = recipe.policy;
    if policy.bundle.is_some() {
        policy.bundle = Some(bundle.clone());
    }
    cycle.train = TrainRef {
        recipe: None,
        // `es loop cycle` trains on its own collect output whatever `[dataset]` says
        // (`Cycle::training`), so this run's recipe names no other directory.
        dataset: None,
        policy: Some(policy),
        run: Some(es_data::training::Run {
            steps,
            checkpoint_at: marks,
            ..recipe.run
        }),
    };
    let collect = cycle.collect.as_mut().ok_or_else(|| {
        err(
            &cycle_path,
            "it collects nothing, so there is nothing to teach",
        )
    })?;
    collect.episodes = settings.demonstrations;
    collect.policy = bundle;
    // The same checks `es loop cycle` makes before it runs anything, so a preset the route
    // refuses is refused here, not a second after Start.
    cycle.training(None, run).map_err(|e| err(&cycle_path, e))?;

    let text = toml::to_string(&cycle).map_err(|e| err(&cycle_path, e))?;
    let parent = run.parent().unwrap_or(run);
    std::fs::create_dir_all(parent).map_err(|e| err(parent, e))?;
    std::fs::create_dir(run).map_err(|e| err(run, e))?;
    let recipe_path = run.join(RUN_RECIPE);
    write(&recipe_path, text)?;
    write(&run.join(TELEMETRY_FILE), telemetry)?;
    Ok(cycle_argv(&recipe_path, run, telemetry))
}

fn cycle_argv(recipe: &Path, out: &Path, telemetry: &str) -> Vec<String> {
    vec![
        "loop".to_owned(),
        "cycle".to_owned(),
        "--recipe".to_owned(),
        arg(recipe),
        "--out".to_owned(),
        arg(out),
        "--telemetry".to_owned(),
        telemetry.to_owned(),
    ]
}

/// `--from <stage>` argv for resuming `run` (same recipe, same `--out`). `from` is spelled
/// the way `es loop cycle --from` reads it (`es_data::training::Stage::parse`), which is what
/// [`crate::model::workflow`]'s resume points are.
pub fn resume_argv(run: &RunFolder, from: &str, telemetry: &str) -> Vec<String> {
    let mut argv = cycle_argv(&run.path.join(RUN_RECIPE), &run.path, telemetry);
    argv.extend(["--from".to_owned(), from.to_owned()]);
    argv
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::template::{find_root, load};
    use es_data::training::Stage;

    pub(crate) fn repo() -> PathBuf {
        find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout")
    }

    fn template(id: &str) -> Template {
        let (ok, bad) = load(&repo());
        assert!(bad.is_empty(), "{bad:?}");
        ok.into_iter().find(|t| t.id == id).expect(id)
    }

    /// The camera-only card.
    pub(crate) fn cube() -> Template {
        template("cube-into-bin")
    }

    fn hint() -> Template {
        template("cube-into-bin-hint")
    }

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("es-y6-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    pub(crate) fn scratch_project(name: &str) -> Project {
        Project::create(&scratch(name), name, &cube(), &repo()).expect("a project")
    }

    fn inline_marks(cycle: &Cycle) -> Vec<u32> {
        cycle
            .train
            .run
            .as_ref()
            .expect("an inline run")
            .checkpoint_at
            .clone()
    }

    fn after<'a>(argv: &'a [String], flag: &str) -> &'a str {
        &argv.windows(2).find(|w| w[0] == flag).expect(flag)[1]
    }

    #[test]
    fn next_run_dir_never_reuses_a_number() {
        let p = scratch_project("next");
        assert!(p.next_run_dir().ends_with("runs/001"));
        std::fs::create_dir_all(p.root.join("runs/001")).unwrap();
        std::fs::create_dir_all(p.root.join("runs/007")).unwrap();
        std::fs::create_dir_all(p.root.join("runs/notes")).unwrap();
        assert!(p.next_run_dir().ends_with("runs/008"));
        assert_eq!(
            p.runs().iter().map(|r| r.number).collect::<Vec<_>>(),
            [1, 7]
        );
        assert_eq!(p.latest_run().map(|r| r.number), Some(7));
        // A file that happens to carry the next number is skipped too, not written into.
        std::fs::write(p.root.join("runs/008"), "").unwrap();
        assert!(p.next_run_dir().ends_with("runs/009"));
        std::fs::remove_dir_all(&p.root).ok();
    }

    #[test]
    fn create_refuses_an_existing_project() {
        let p = scratch_project("twice");
        assert!(Project::create(&p.root, "again", &cube(), &repo()).is_err());
        assert_eq!(Project::open(&p.root), Ok(p.clone()));
        assert!(Project::is_project_dir(&p.root));
        assert!(p.bundle().is_file());
        assert_eq!(p.file.template, "cube-into-bin");
        std::fs::remove_dir_all(&p.root).ok();
    }

    #[test]
    fn run_recipe_overrides_only_what_the_person_chose() {
        let p = scratch_project("recipe");
        let run = p.next_run_dir();
        let argv = write_run(
            &cube(),
            &repo(),
            &p,
            StartSettings {
                demonstrations: 50,
                length: Length::Short,
            },
            &run,
            "127.0.0.1:7001",
        )
        .unwrap();
        let written: Cycle =
            toml::from_str(&std::fs::read_to_string(run.join("cycle.toml")).unwrap()).unwrap();
        let template: Cycle =
            toml::from_str(&std::fs::read_to_string(repo().join(&cube().cycle)).unwrap()).unwrap();
        assert_eq!(written.collect.as_ref().unwrap().episodes, 50);
        assert_eq!(written.eval, template.eval);
        assert_eq!(written.scene, template.scene);
        assert_eq!(written.showcase, template.showcase);
        let (w, t) = (
            written.collect.as_ref().unwrap(),
            template.collect.as_ref().unwrap(),
        );
        assert_eq!((&w.expert, w.seed, w.frames), (&t.expert, t.seed, t.frames));
        assert_eq!(w.policy, p.bundle().display().to_string());
        // the inline train's marks are the preset, and the length is its last mark
        let marks = cube().marks(Length::Short).to_vec();
        assert_eq!(inline_marks(&written), marks);
        assert_eq!(
            written.train.run.as_ref().unwrap().steps,
            *marks.last().unwrap()
        );
        assert_eq!(argv[..2], ["loop".to_owned(), "cycle".to_owned()]);
        assert!(argv
            .windows(2)
            .any(|w| w[0] == "--telemetry" && w[1] == "127.0.0.1:7001"));
        let folder = p.latest_run().unwrap();
        assert_eq!(folder.path, run);
        assert_eq!(folder.telemetry_addr().as_deref(), Some("127.0.0.1:7001"));
        // A number is never written twice.
        assert!(write_run(
            &cube(),
            &repo(),
            &p,
            StartSettings {
                demonstrations: 50,
                length: Length::Short,
            },
            &run,
            "127.0.0.1:7001",
        )
        .is_err());
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Both cards, through the parser `es loop cycle` itself uses: the camera-only one trains
    /// on `training-lerobot.toml`'s route, the cube-pose one on `training-hint-u3.toml`'s, and each
    /// written recipe resolves to its preset under the route's own mark rule.
    #[test]
    fn a_run_recipe_for_each_template_parses_as_a_cycle() {
        for (template, recipe_file) in [
            (cube(), "training-lerobot.toml"),
            (hint(), "training-hint-u3.toml"),
        ] {
            let root = scratch(&format!("both-{}", template.id));
            let p = Project::create(&root, "both", &template, &repo()).unwrap();
            let committed =
                Cycle::parse(&std::fs::read_to_string(repo().join(&template.cycle)).unwrap())
                    .unwrap();
            assert!(committed
                .train
                .recipe
                .as_deref()
                .is_some_and(|r| r.ends_with(recipe_file)));
            let source = Recipe::parse(
                &std::fs::read_to_string(repo().join(committed.train.recipe.unwrap())).unwrap(),
            )
            .unwrap();
            for length in [Length::Short, Length::Medium, Length::Long] {
                let run = p.next_run_dir();
                write_run(
                    &template,
                    &repo(),
                    &p,
                    StartSettings {
                        demonstrations: 7,
                        length,
                    },
                    &run,
                    "127.0.0.1:7003",
                )
                .unwrap();
                let written =
                    Cycle::parse(&std::fs::read_to_string(run.join(RUN_RECIPE)).unwrap()).unwrap();
                let resolved = written.training(None, &run).unwrap();
                assert_eq!(
                    resolved.marks().unwrap(),
                    template.marks(length),
                    "{length:?}"
                );
                let (r, s) = (&resolved.run, &source.run);
                assert_eq!(
                    (r.batch, r.lr, r.seed, &r.device, &r.interpreter),
                    (s.batch, s.lr, s.seed, &s.device, &s.interpreter),
                    "{}: the rest of [run] is the template's",
                    template.id
                );
                assert_eq!(resolved.policy.lerobot, source.policy.lerobot);
                assert_eq!(resolved.policy.observation, source.policy.observation);
                let bundle = p.bundle().display().to_string();
                let want = source.policy.bundle.as_ref().map(|_| bundle.clone());
                assert_eq!(resolved.policy.bundle, want, "{}", template.id);
                assert_eq!(written.collect.as_ref().unwrap().policy, bundle);
            }
            std::fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn run_recipe_and_argv_keep_a_korean_path_with_spaces() {
        // "\u{bb38}\u{c11c}" and "\u{b0b4} \u{b85c}\u{bd07}" are Korean words; written as escapes
        // because Korean text lives only in the i18n tables.
        let base = scratch("korean");
        let root = base
            .join("\u{bb38}\u{c11c}")
            .join("\u{b0b4} \u{b85c}\u{bd07}");
        let p = Project::create(&root, "k", &cube(), &repo()).unwrap();
        assert_eq!(p.root, root);
        let run = p.next_run_dir();
        let argv = write_run(
            &cube(),
            &repo(),
            &p,
            StartSettings {
                demonstrations: 10,
                length: Length::Short,
            },
            &run,
            "127.0.0.1:7002",
        )
        .unwrap();
        let out = argv.windows(2).find(|w| w[0] == "--out").unwrap()[1].clone();
        assert_eq!(PathBuf::from(out), run);
        let recipe = argv.windows(2).find(|w| w[0] == "--recipe").unwrap()[1].clone();
        assert!(Path::new(&recipe).is_file());
        // ... and the recipe names the bundle under the same path, verbatim.
        let written = Cycle::parse(&std::fs::read_to_string(&recipe).unwrap()).unwrap();
        assert_eq!(
            PathBuf::from(&written.collect.unwrap().policy),
            root.join(COLLECT_BUNDLE)
        );
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn resume_argv_is_the_same_run_from_a_stage_the_cli_reads() {
        let run = RunFolder {
            number: 3,
            path: PathBuf::from("C:/p/runs/003"),
        };
        let argv = resume_argv(&run, "eval", "127.0.0.1:7004");
        assert_eq!(argv[..2], ["loop".to_owned(), "cycle".to_owned()]);
        assert_eq!(PathBuf::from(after(&argv, "--out")), run.path);
        assert_eq!(
            PathBuf::from(after(&argv, "--recipe")),
            run.path.join(RUN_RECIPE)
        );
        assert_eq!(after(&argv, "--telemetry"), "127.0.0.1:7004");
        assert_eq!(Stage::parse(after(&argv, "--from")), Some(Stage::Eval));
    }
}
