//! ② for a template whose demonstrator is a trained policy (`method = "teacher"`, packet
//! M16/H7): the project's teacher runs, each checkpoint's evaluated success, and the chosen
//! teacher - `teacher.esb` - that every run of the project collects with.
//!
//! The editor runs no learning (spec 23.1). Training a teacher is `es train` on the template's
//! recipe, judging a checkpoint is `es eval run` on the template's Evaluation IR, and re-packing
//! the chosen one on the template's documents is `es policy pack`: each is an argv that [`Jobs`]
//! starts in the repository root, one at a time, exactly as `launch.rs` starts any run.
//!
//! ```text
//! teacher/001/            `es train --out teacher/001`, plus
//!   recipe.toml           the template's recipe, `[policy] bundle` = teacher-untrained.esb
//!   eval/<step>/          `es eval run --out`, one per judged checkpoint
//!   repacked/<step>.esb   a checkpoint packed on documents saved since it trained, for its test
//! teacher-untrained.esb   `es policy init` of the template's `[teacher]` documents
//! teacher.esb             the chosen checkpoint; `teacher.toml` names its run and step
//! ```

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use es_compile::{BundleHashes, PolicyBundle};
use es_data::training::Recipe;
use es_ir::evaluation::EvaluationReport;
use serde::{Deserialize, Serialize};

use crate::model::i18n::{fill, t, Lang};
use crate::model::launch::LaunchModel;
use crate::model::project::{Project, ProjectError, RunFolder};
use crate::model::results::card;
use crate::model::template::{TeacherDocs, Template};
use crate::model::train_view::TrainView;

/// The recipe a teacher run ran, written into its folder before launch.
pub const RECIPE: &str = "recipe.toml";
/// The untrained teacher: the template's `[teacher]` documents, as `es policy init` builds them.
pub const UNTRAINED: &str = "teacher-untrained.esb";
/// Which run and step `teacher.esb` is.
pub const CHOICE: &str = "teacher.toml";
const EVAL: &str = "eval";
const CHECKPOINTS: &str = "checkpoints";
const REPORT: &str = "report.json";
/// A checkpoint re-packed on the current documents for its test (packet M17/R7).
const REPACKED: &str = "repacked";

fn fail(path: &Path, e: impl std::fmt::Display) -> ProjectError {
    ProjectError(format!("{}: {e}", path.display()))
}

fn arg(path: &Path) -> String {
    path.display().to_string()
}

fn docs(template: &Template) -> Result<&TeacherDocs, ProjectError> {
    (template.teacher.as_ref())
        .ok_or_else(|| ProjectError(format!("template {}: no [teacher]", template.id)))
}

/// Successes over attempts of one evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Score {
    pub successes: u32,
    pub episodes: u32,
}

impl Score {
    pub fn rate(self) -> f64 {
        f64::from(self.successes) / f64::from(self.episodes.max(1))
    }

    /// "48% (31 of 64)".
    pub fn text(self) -> String {
        format!(
            "{:.0}% ({} / {})",
            self.rate() * 100.0,
            self.successes,
            self.episodes
        )
    }
}

/// One checkpoint of a teacher run and, once judged, its score.
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub step: u32,
    pub score: Option<Score>,
}

pub fn checkpoint(run: &RunFolder, step: u32) -> PathBuf {
    run.path.join(CHECKPOINTS).join(format!("{step}.esb"))
}

pub fn eval_dir(run: &RunFolder, step: u32) -> PathBuf {
    run.path.join(EVAL).join(step.to_string())
}

/// What `<dir>/report.json` says: the report's own success cells, summed ([`card`]).
pub fn score(dir: &Path) -> Option<Score> {
    let text = std::fs::read_to_string(dir.join(REPORT)).ok()?;
    let report: EvaluationReport = serde_json::from_str(&text).ok()?;
    let c = card(&report, None);
    (c.episodes > 0).then_some(Score {
        successes: c.successes,
        episodes: c.episodes,
    })
}

/// Every `checkpoints/<step>.esb` of `run`, by step, with its evaluation's score.
pub fn marks(run: &RunFolder) -> Vec<Mark> {
    let Ok(entries) = std::fs::read_dir(run.path.join(CHECKPOINTS)) else {
        return Vec::new();
    };
    let mut steps: Vec<u32> = (entries.flatten())
        .filter_map(|e| {
            let path = e.path();
            (path.extension()? == "esb").then_some(())?;
            path.file_stem()?.to_str()?.parse().ok()
        })
        .collect();
    steps.sort_unstable();
    (steps.into_iter())
        .map(|step| Mark {
            step,
            score: score(&eval_dir(run, step)),
        })
        .collect()
}

/// `teacher.toml`: which teacher run and checkpoint `teacher.esb` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub run: u32,
    pub step: u32,
}

impl Choice {
    pub fn write(self, project: &Project) -> Result<(), ProjectError> {
        let path = project.root.join(CHOICE);
        let text = toml::to_string(&self).map_err(|e| fail(&path, e))?;
        std::fs::write(&path, text).map_err(|e| fail(&path, e))
    }
}

/// The chosen teacher: `teacher.toml`, but only while `teacher.esb` is there - a pack that
/// failed chose nothing.
pub fn chosen(project: &Project) -> Option<Choice> {
    if !project.teacher_bundle().is_file() {
        return None;
    }
    let text = std::fs::read_to_string(project.root.join(CHOICE)).ok()?;
    toml::from_str(&text).ok()
}

/// The chosen teacher's own score, the student's reference in ⑤.
pub fn chosen_score(project: &Project) -> Option<Score> {
    let c = chosen(project)?;
    let run = (project.teacher_runs().into_iter()).find(|r| r.number == c.run)?;
    score(&eval_dir(&run, c.step))
}

/// Writes `teacher-untrained.esb` from the template's `[teacher]` documents - what `es policy
/// init` writes - and returns its path. Rebuilt each time: the documents may have moved since.
pub fn untrained(
    template: &Template,
    repo: &Path,
    project: &Project,
) -> Result<PathBuf, ProjectError> {
    let t = docs(template)?;
    let doc = |p: &String| repo.join(p);
    let bytes = es_data::training::untrained_bundle(
        &doc(&t.task),
        &doc(&t.observation),
        Some(&doc(&t.learning)),
        &doc(&t.deployment),
        0,
    )
    .map_err(|e| ProjectError(format!("template {}: {e}", template.id)))?;
    let path = project.root.join(UNTRAINED);
    // Unchanged, it is not rewritten: a teacher training now may be reading it.
    if std::fs::read(&path).ok().as_ref() != Some(&bytes) {
        std::fs::write(&path, bytes).map_err(|e| fail(&path, e))?;
    }
    Ok(path)
}

/// A new teacher run: makes `run` ([`Project::next_teacher_dir`]; it must not exist), writes the
/// template's recipe into it with `[policy] bundle` the project's untrained teacher, and returns
/// `es train`'s argv (no program name).
pub fn train(
    template: &Template,
    repo: &Path,
    project: &Project,
    run: &Path,
    telemetry: &str,
) -> Result<Vec<String>, ProjectError> {
    let path = repo.join(&docs(template)?.recipe);
    let text = std::fs::read_to_string(&path).map_err(|e| fail(&path, e))?;
    let mut recipe = Recipe::parse(&text).map_err(|e| fail(&path, e))?;
    recipe.policy.bundle = Some(arg(&untrained(template, repo, project)?));
    let text = toml::to_string(&recipe).map_err(|e| fail(&path, e))?;
    let parent = run.parent().unwrap_or(run);
    std::fs::create_dir_all(parent).map_err(|e| fail(parent, e))?;
    std::fs::create_dir(run).map_err(|e| fail(run, e))?;
    let written = run.join(RECIPE);
    std::fs::write(&written, text).map_err(|e| fail(&written, e))?;
    Ok(vec![
        "train".into(),
        "--recipe".into(),
        arg(&written),
        "--out".into(),
        arg(run),
        "--telemetry".into(),
        telemetry.into(),
    ])
}

/// `es eval run` of one checkpoint on the template's teacher evaluation, into `eval/<step>`.
pub fn evaluate(
    template: &Template,
    repo: &Path,
    run: &RunFolder,
    step: u32,
) -> Result<Vec<String>, ProjectError> {
    eval_argv(template, repo, run, step, &checkpoint(run, step))
}

/// The jobs that test checkpoint `step` (packet M17/R7): its evaluation, and before it, when its
/// documents are not the untrained teacher's - rebuilt from the current ones, which a save in ①
/// may have regenerated since it trained - `es policy pack` of its weights on that teacher into
/// `repacked/<step>.esb`, which is then what is evaluated (`es eval run` refuses a bundle built
/// on other documents, spec 10.4).
pub fn test(
    template: &Template,
    repo: &Path,
    project: &Project,
    run: &RunFolder,
    step: u32,
) -> Result<Vec<Vec<String>>, ProjectError> {
    let untrained = untrained(template, repo, project)?;
    if same_documents(&open(&checkpoint(run, step))?.1, &open(&untrained)?.1) {
        return Ok(vec![evaluate(template, repo, run, step)?]);
    }
    let repacked = run.path.join(REPACKED).join(format!("{step}.esb"));
    remove(&repacked)?;
    Ok(vec![
        pack(&untrained, run, step, &repacked),
        eval_argv(template, repo, run, step, &repacked)?,
    ])
}

fn eval_argv(
    template: &Template,
    repo: &Path,
    run: &RunFolder,
    step: u32,
    policy: &Path,
) -> Result<Vec<String>, ProjectError> {
    let t = docs(template)?;
    Ok(vec![
        "eval".into(),
        "run".into(),
        "--config".into(),
        arg(&repo.join(&t.evaluation)),
        "--policy".into(),
        arg(policy),
        "--scene".into(),
        arg(&repo.join(&template.scene)),
        "--out".into(),
        arg(&eval_dir(run, step)),
        "--jobs".into(),
        t.jobs.to_string(),
    ])
}

/// What choosing a checkpoint took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chose {
    /// Its documents are the template's: `teacher.esb` is its bytes.
    Copied,
    /// Its documents differ (the template's moved since it trained): `es policy pack` of its
    /// weights on the untrained teacher writes `teacher.esb`.
    Pack(Vec<String>),
}

/// Whether two bundles were built on the same Task, Observation and Deployment IR - all a
/// re-pack changes (its weights depend on the Learning IR alone).
pub fn same_documents(a: &BundleHashes, b: &BundleHashes) -> bool {
    (a.task, a.observation, a.deployment) == (b.task, b.observation, b.deployment)
}

/// Makes checkpoint `step` of `run` the project's teacher: `teacher.toml` names it, and
/// `teacher.esb` is its bytes or, when its documents are not the template's, what the returned
/// pack writes. The previous `teacher.esb` is removed first, so a pack that fails leaves no
/// teacher rather than the old one under the new name.
pub fn choose(
    template: &Template,
    repo: &Path,
    project: &Project,
    run: &RunFolder,
    step: u32,
) -> Result<Chose, ProjectError> {
    let untrained = untrained(template, repo, project)?;
    let (bytes, hashes) = open(&checkpoint(run, step))?;
    let same = same_documents(&hashes, &open(&untrained)?.1);
    let target = project.teacher_bundle();
    remove(&target)?;
    Choice {
        run: run.number,
        step,
    }
    .write(project)?;
    if same {
        std::fs::write(&target, bytes).map_err(|e| fail(&target, e))?;
        return Ok(Chose::Copied);
    }
    Ok(Chose::Pack(pack(&untrained, run, step, &target)))
}

/// A bundle's bytes and its hashes.
fn open(path: &Path) -> Result<(Vec<u8>, BundleHashes), ProjectError> {
    let bytes = std::fs::read(path).map_err(|e| fail(path, e))?;
    let bundle = PolicyBundle::open(&bytes).map_err(|e| fail(path, e))?;
    Ok((bytes, bundle.manifest.hashes))
}

/// Gone, or never there: a pack that fails then leaves nothing under the name it was to write.
fn remove(path: &Path) -> Result<(), ProjectError> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(fail(path, e)),
        _ => Ok(()),
    }
}

/// `es policy pack` of checkpoint `step`'s weights on `untrained`, into `out`.
fn pack(untrained: &Path, run: &RunFolder, step: u32, out: &Path) -> Vec<String> {
    let weights = (run.path.join("weights")).join(format!("model-{step}.safetensors"));
    vec![
        "policy".into(),
        "pack".into(),
        "--policy".into(),
        arg(untrained),
        "--weights".into(),
        arg(&weights),
        "--out".into(),
        arg(out),
    ]
}

/// Seconds per iteration a finished teacher run took (packet M17/R7): the wall clock over the
/// iterations of `metrics/env-metrics.json`, which `train_ppo.py` writes beside its loss curve.
pub fn pace(run: &RunFolder) -> Option<f64> {
    let text = std::fs::read_to_string(run.path.join("metrics/env-metrics.json")).ok()?;
    let m: serde_json::Value = serde_json::from_str(&text).ok()?;
    let (wall, n) = (m["wall_clock_s"].as_f64()?, m["iterations"].as_f64()?);
    (wall > 0.0 && n > 0.0).then(|| wall / n)
}

/// The newest of `runs` that measured its [`pace`], by number.
pub fn measured(runs: &[RunFolder]) -> Option<(u32, f64)> {
    runs.iter().rev().find_map(|r| Some((r.number, pace(r)?)))
}

/// The iterations the template's teacher recipe runs, `[run] steps`.
pub fn iterations(template: &Template, repo: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(repo.join(&docs(template).ok()?.recipe)).ok()?;
    Some(Recipe::parse(&text).ok()?.run.steps)
}

/// What the card says about time (packet M17/R7). While `live`, the run training now, has a
/// rate: its iterations done of the total and what is left at its own pace. Else a `measured`
/// run's pace times the recipe's `iterations`. Else, on `generated` documents, that the first
/// run measures it; a built-in template keeps its own sentence.
pub fn time_text(
    lang: Lang,
    live: Option<&TrainView>,
    measured: Option<(u32, f64)>,
    iterations: Option<u32>,
    generated: bool,
) -> String {
    let minutes = |s: f64| format!("{:.0}", (s / 60.0).max(1.0));
    let live = live.and_then(|v| {
        let total = v.total()?;
        Some((v.step()?, total, v.eta(total)?))
    });
    if let Some((done, total, left)) = live {
        let left = minutes(left.as_secs_f64());
        return fill(
            lang,
            "teach.teacher.time.live",
            &[&done.to_string(), &total.to_string(), &left],
        );
    }
    match (measured, iterations) {
        (Some((run, pace)), Some(n)) => fill(
            lang,
            "teach.teacher.time.measured",
            &[
                &minutes(pace * f64::from(n)),
                &format!("{run:03}"),
                &format!("{pace:.2}"),
                &n.to_string(),
            ],
        ),
        _ if generated => t(lang, "teach.teacher.time.first").to_owned(),
        _ => t(lang, "teach.teacher.time").to_owned(),
    }
}

/// What a job of [`Jobs`] is doing, in words: training which teacher, testing which step, or
/// preparing the chosen teacher or a step for its test.
pub fn job_text(lang: Lang, argv: &[String]) -> String {
    let out = (argv.windows(2).find(|w| w[0] == "--out")).map(|w| Path::new(&w[1]));
    let name = (out.and_then(Path::file_stem))
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let repack =
        (out.and_then(Path::parent).and_then(Path::file_name)).is_some_and(|n| n == REPACKED);
    match argv.first().map(String::as_str) {
        Some("train") => fill(lang, "teach.teacher.job.train", &[&name]),
        Some("eval") => fill(lang, "teach.teacher.job.test", &[&name]),
        _ if repack => fill(lang, "teach.teacher.job.repack", &[&name]),
        _ => t(lang, "teach.teacher.job.pack").to_owned(),
    }
}

/// ②'s children, one at a time: a teacher's training, its evaluations, a pack. Its own launch
/// model, as ②'s try has, so the run the step bar follows never sees them.
#[derive(Default)]
pub struct Jobs {
    pub launch: LaunchModel,
    queue: VecDeque<Vec<String>>,
    current: Option<Vec<String>>,
}

impl std::fmt::Debug for Jobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jobs")
            .field("current", &self.current)
            .field("queue", &self.queue)
            .finish_non_exhaustive()
    }
}

impl Jobs {
    /// Queues `argv` behind what runs; the same argv twice is queued once.
    pub fn push(&mut self, argv: Vec<String>) {
        if self.current.as_ref() != Some(&argv) && !self.queue.contains(&argv) {
            self.queue.push_back(argv);
        }
    }

    /// Once a frame: polls the child and, once none runs, starts the next argv in `root`.
    /// `true` when a child ended since the last call: what is on disk changed.
    pub fn tick(&mut self, root: &Path) -> bool {
        self.launch.poll();
        if self.launch.pid().is_some() {
            return false;
        }
        let ended = self.current.take().is_some();
        if let Some(argv) = self.queue.pop_front() {
            self.launch.start_in(&argv, root);
            self.current = Some(argv);
        }
        ended
    }

    /// The argv running now.
    pub fn current(&self) -> Option<&[String]> {
        self.current.as_deref()
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn busy(&self) -> bool {
        self.current.is_some() || !self.queue.is_empty()
    }

    /// The teacher run being trained now: the `--out` of a running `es train`.
    pub fn training(&self) -> Option<PathBuf> {
        let argv = self.current()?;
        (argv.first()? == "train").then_some(())?;
        let out = argv.windows(2).find(|w| w[0] == "--out")?;
        Some(PathBuf::from(&out[1]))
    }

    /// Stop: the queue dropped and the child killed.
    pub fn stop(&mut self) {
        self.queue.clear();
        self.launch.kill();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::project::tests::{repo, template};
    use es_ir::evaluation::{CellResult, MetricSpec, MetricValue};

    pub(crate) fn hand() -> Template {
        template("shadow-hand-repose")
    }

    pub(crate) fn hand_project(name: &str) -> Project {
        let root = std::env::temp_dir().join(format!("es-h7-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        Project::create(&root, name, &hand(), &repo()).expect("a hand project")
    }

    /// A report of `successes` of 64 on the nominal suite, as `es eval run` writes it.
    pub(crate) fn write_report(dir: &Path, successes: u32) {
        let report = EvaluationReport {
            schema_version: 1,
            evaluation_hash: [7; 32],
            execution_hash: [0; 32],
            cells: vec![CellResult {
                suite: "nominal".into(),
                metric: MetricSpec::SuccessRate,
                value: MetricValue::Scalar(f64::from(successes) / 64.0),
                n_episodes: 64,
            }],
            acceptance: Vec::new(),
            passed: false,
            episodes: Vec::new(),
        };
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(REPORT), serde_json::to_string(&report).unwrap()).unwrap();
    }

    /// A teacher run whose checkpoints are the untrained teacher's bytes, the first judged.
    pub(crate) fn fake_run(project: &Project, steps: &[u32]) -> RunFolder {
        let bytes = std::fs::read(untrained(&hand(), &repo(), project).unwrap()).unwrap();
        let path = project.next_teacher_dir();
        std::fs::create_dir_all(path.join(CHECKPOINTS)).unwrap();
        for step in steps {
            std::fs::write(path.join(CHECKPOINTS).join(format!("{step}.esb")), &bytes).unwrap();
        }
        let run = project.teacher_runs().pop().unwrap();
        write_report(&eval_dir(&run, steps[0]), 31);
        run
    }

    /// The recipe a new teacher run writes is the template's with only its bundle moved, and
    /// `es train`'s argv names it, the folder and the address.
    #[test]
    fn a_teacher_run_trains_the_template_recipe_on_the_project_bundle() {
        let p = hand_project("train");
        let run = p.next_teacher_dir();
        let argv = train(&hand(), &repo(), &p, &run, "127.0.0.1:7010").unwrap();
        let recipe = run.join(RECIPE);
        assert_eq!(
            argv,
            [
                "train",
                "--recipe",
                &arg(&recipe),
                "--out",
                &arg(&run),
                "--telemetry",
                "127.0.0.1:7010"
            ]
        );
        let written = Recipe::parse(&std::fs::read_to_string(&recipe).unwrap()).unwrap();
        let path = repo().join(&hand().teacher.unwrap().recipe);
        let mut source = Recipe::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let bundle = p.root.join(UNTRAINED);
        source.policy.bundle = Some(arg(&bundle));
        assert_eq!(written, source);
        assert!(PolicyBundle::open(&std::fs::read(&bundle).unwrap()).is_ok());
        assert_eq!(p.teacher_runs().len(), 1);
        assert!(
            train(&hand(), &repo(), &p, &run, "x").is_err(),
            "never twice"
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Checkpoints are listed by step with their scores; choosing one whose documents are the
    /// template's copies it, and a teacher is chosen only while `teacher.esb` is there.
    #[test]
    fn choosing_a_checkpoint_writes_the_teacher() {
        let p = hand_project("choose");
        let run = fake_run(&p, &[250, 1000, 500]);
        let m = marks(&run);
        assert_eq!(
            m.iter().map(|m| m.step).collect::<Vec<_>>(),
            [250, 500, 1000]
        );
        let judged = Score {
            successes: 31,
            episodes: 64,
        };
        assert_eq!((m[0].score, m[1].score), (Some(judged), None));
        assert_eq!(chosen(&p), None);
        assert_eq!(choose(&hand(), &repo(), &p, &run, 250), Ok(Chose::Copied));
        assert_eq!(
            std::fs::read(p.teacher_bundle()).unwrap(),
            std::fs::read(checkpoint(&run, 250)).unwrap()
        );
        assert_eq!(chosen(&p), Some(Choice { run: 1, step: 250 }));
        assert_eq!(chosen_score(&p), Some(judged));
        std::fs::remove_file(p.teacher_bundle()).unwrap();
        assert_eq!(chosen(&p), None, "no bundle, no teacher");
        assert!(
            choose(&hand(), &repo(), &p, &run, 999).is_err(),
            "no such mark"
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// Different documents need a pack; the argv names the weights of that step.
    #[test]
    fn same_documents_are_the_task_observation_and_deployment() {
        let a = BundleHashes::default();
        let mut b = a;
        assert!(same_documents(&a, &b));
        b.learning = Some([1; 32]);
        assert!(
            same_documents(&a, &b),
            "the Learning IR is the weights' own"
        );
        b.task = Some([2; 32]);
        assert!(!same_documents(&a, &b));
    }

    #[test]
    fn an_evaluation_judges_one_checkpoint_into_its_folder() {
        let run = RunFolder {
            number: 2,
            path: PathBuf::from("C:/p/teacher/002"),
        };
        let argv = evaluate(&hand(), &repo(), &run, 1750).unwrap();
        let after = |flag: &str| &argv.windows(2).find(|w| w[0] == flag).unwrap()[1];
        assert_eq!(argv[..2], ["eval".to_owned(), "run".to_owned()]);
        assert_eq!(PathBuf::from(after("--out")), run.path.join("eval/1750"));
        assert_eq!(
            PathBuf::from(after("--policy")),
            run.path.join("checkpoints/1750.esb")
        );
        assert!(Path::new(after("--config")).is_file());
        assert!(Path::new(after("--scene")).is_file());
        assert_eq!(after("--jobs"), "4");
    }

    /// One child at a time, in order; the running `es train`'s folder is the one trained.
    #[test]
    fn jobs_run_one_at_a_time() {
        let mut jobs = Jobs::default();
        assert!(!jobs.busy() && jobs.training().is_none());
        let train = vec!["train".into(), "--out".into(), "C:/t/001".into()];
        jobs.push(train.clone());
        jobs.push(train.clone());
        jobs.push(vec!["eval".into()]);
        assert_eq!(jobs.queued(), 2, "the same argv once");
        jobs.current = jobs.queue.pop_front();
        assert_eq!(jobs.training(), Some(PathBuf::from("C:/t/001")));
        jobs.push(train);
        assert_eq!(jobs.queued(), 1, "nor while it runs");
        jobs.stop();
        assert_eq!(jobs.queued(), 0);
    }

    /// Each job in words, in both languages, naming its teacher or its step.
    #[test]
    fn every_job_has_words() {
        let run = RunFolder {
            number: 2,
            path: PathBuf::from("C:/p/teacher/002"),
        };
        let train = vec!["train".into(), "--out".into(), arg(&run.path)];
        let test = evaluate(&hand(), &repo(), &run, 1750).unwrap();
        let repack = pack(
            Path::new("u.esb"),
            &run,
            1750,
            &run.path.join("repacked/1750.esb"),
        );
        for lang in Lang::ALL {
            assert!(job_text(lang, &train).contains("002"));
            assert!(job_text(lang, &test).contains("1750"));
            let pack = job_text(lang, &["policy".into(), "pack".into()]);
            assert_ne!(pack, "teach.teacher.job.pack");
            assert!(job_text(lang, &repack).contains("1750"));
        }
    }

    /// Packet M17/R7 (F-1): a checkpoint on the teacher's documents is tested as before; one on
    /// other documents is first packed on the untrained teacher, and its re-pack is tested.
    #[test]
    fn a_checkpoint_on_other_documents_is_repacked_for_its_test() {
        let p = hand_project("test");
        let run = fake_run(&p, &[250]);
        let same = test(&hand(), &repo(), &p, &run, 250).unwrap();
        assert_eq!(same, [evaluate(&hand(), &repo(), &run, 250).unwrap()]);
        // The teacher's documents as if a save had regenerated them since: another task's.
        let other = template("cube-into-bin-hint").bundle.unwrap();
        let mut moved = hand();
        let t = moved.teacher.as_mut().unwrap();
        (t.task, t.observation, t.deployment) = (other.task, other.observation, other.deployment);
        t.learning = other.learning.unwrap();
        let repacked = run.path.join(REPACKED).join("250.esb");
        std::fs::create_dir_all(repacked.parent().unwrap()).unwrap();
        std::fs::write(&repacked, "stale").unwrap();
        let jobs = test(&moved, &repo(), &p, &run, 250).unwrap();
        assert!(!repacked.exists(), "a stale re-pack is never evaluated");
        let weights = run.path.join("weights").join("model-250.safetensors");
        let untrained = p.root.join(UNTRAINED);
        let mut eval = evaluate(&moved, &repo(), &run, 250).unwrap();
        eval[5] = arg(&repacked);
        assert_eq!(
            jobs,
            [
                vec![
                    "policy".into(),
                    "pack".into(),
                    "--policy".into(),
                    arg(&untrained),
                    "--weights".into(),
                    arg(&weights),
                    "--out".into(),
                    arg(&repacked),
                ],
                eval
            ]
        );
        std::fs::remove_dir_all(&p.root).ok();
    }

    /// A live training view of `total` iterations at one second each, `done` of them done.
    fn live(done: u32, total: u32) -> TrainView {
        use es_telemetry::protocol::{Frame, Message, Payload};
        let frame = |stream, wall_ns, payload| {
            Message::Frame(Frame {
                tick: es_core::PhysTick(0),
                wall_ns,
                stream,
                payload,
            })
        };
        let mut view = TrainView::default();
        let begin = Payload::Event {
            kind: "train.begin".into(),
            fields: [("total_steps".to_owned(), total.to_string())].into(),
        };
        view.ingest(&frame(crate::model::live_run::STREAM_EVENTS, 0, begin));
        for step in [1, done] {
            let row = Payload::Scalars(vec![f64::from(step), 0.1, 1e-4, 1.0]);
            let wall = u64::from(step) * 1_000_000_000;
            view.ingest(&frame(crate::model::train_view::STREAM_TRAIN, wall, row));
        }
        view
    }

    /// Packet M17/R7 (F-3): the card's time is the run's own while it trains, a finished run's
    /// measured pace times the recipe before one does, and no number with neither.
    #[test]
    fn the_card_time_is_measured_or_says_the_first_run_measures_it() {
        let p = hand_project("time");
        let run = fake_run(&p, &[250]);
        assert_eq!(measured(&p.teacher_runs()), None, "no measurement");
        let metrics = run.path.join("metrics");
        std::fs::create_dir_all(&metrics).unwrap();
        let text = r#"{"iterations": 3000, "envs": 2048, "wall_clock_s": 2846.7}"#;
        std::fs::write(metrics.join("env-metrics.json"), text).unwrap();
        let (number, pace) = measured(&p.teacher_runs()).unwrap();
        assert_eq!(number, 1);
        assert!((pace - 0.9489).abs() < 1e-4, "{pace}");
        assert!(iterations(&hand(), &repo()).is_some_and(|n| n > 0));
        let digits = |s: &str| s.chars().filter(char::is_ascii_digit).count();
        for lang in Lang::ALL {
            let before = time_text(lang, None, Some((number, pace)), Some(3000), true);
            for part in ["47", "001", "0.95", "3000"] {
                assert!(before.contains(part), "{before}");
            }
            let now = time_text(lang, Some(&live(1200, 3000)), None, Some(3000), true);
            for part in ["1200", "3000", "30"] {
                assert!(now.contains(part), "{now}");
            }
            let starting = time_text(lang, Some(&live(1, 3000)), None, Some(3000), true);
            let first = time_text(lang, None, None, Some(3000), true);
            assert_eq!(starting, first, "no rate yet");
            assert_eq!(digits(&first), 0, "{first}");
            let kept = time_text(lang, None, None, Some(3000), false);
            assert_eq!(
                kept,
                t(lang, "teach.teacher.time"),
                "a template keeps its words"
            );
        }
        std::fs::remove_dir_all(&p.root).ok();
    }
}
