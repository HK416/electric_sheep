//! A project made of runs that ran outside the editor (packet M16/H7): `es-editor --import`.
//!
//! The Shadow Hand's teacher and student were trained and judged from the command line before
//! the editor could start them (plan H, E1/E2). This makes the folder the editor would have made:
//! `project.toml` of the template, each `es train --out` folder as `teacher/NNN` with its
//! evaluations as `teacher/NNN/eval/<step>`, the teacher bundle the student collected with as
//! `teacher.esb` (its checkpoint found by its weights' hash), and each `es loop cycle --out`
//! folder as `runs/NNN` with the recipe it ran as `cycle.toml`.
//!
//! How the outputs get there is [`Mode`]'s: a student run is gigabytes of frames.
//!
//! ```text
//! es-editor --import <project-dir> --template <id> [--name <name>]
//!           [--teacher <es train --out> [--evals <prefix>]]...  (evaluations: <prefix><step>)
//!           [--teacher-bundle <teacher.esb>] [--student <es loop cycle --out>]...
//!           [--mode link|move|copy]                             (default link)
//! ```

use std::path::{Path, PathBuf};

use es_compile::PolicyBundle;

use crate::model::project::{Project, ProjectError, RunFolder, RUN_RECIPE};
use crate::model::teacher::{self, Choice};
use crate::model::template::Template;

/// How an output gets into the project.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Every folder is a link to where it is (a directory junction on Windows, a symbolic link
    /// elsewhere) and every top-level file a copy: nothing is moved, written or duplicated
    /// where the outputs are, and they must stay there.
    #[default]
    Link,
    /// Renamed into the project: instant on the same volume, refused across volumes.
    Move,
    /// Copied: the outputs stay where they are, at the cost of their size again.
    Copy,
}

/// One `es train --out` folder and where its evaluations are: `<evals><step>` per checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherRun {
    pub run: PathBuf,
    pub evals: Option<String>,
}

/// What `es-editor --import` was asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Import {
    pub dest: PathBuf,
    pub name: String,
    pub template: String,
    pub teachers: Vec<TeacherRun>,
    pub bundle: Option<PathBuf>,
    pub students: Vec<PathBuf>,
    pub mode: Mode,
}

fn fail(path: &Path, e: impl std::fmt::Display) -> ProjectError {
    ProjectError(format!("{}: {e}", path.display()))
}

impl Import {
    /// `args` after `--import`, the module's grammar. `--evals` belongs to the `--teacher`
    /// before it; the name defaults to the folder's.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut it = args.iter();
        let dest = PathBuf::from(it.next().ok_or("--import needs a project folder")?);
        let mut spec = Self {
            name: (dest.file_name()).map_or_else(String::new, |n| n.to_string_lossy().into()),
            dest,
            template: String::new(),
            teachers: Vec::new(),
            bundle: None,
            students: Vec::new(),
            mode: Mode::default(),
        };
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--template" => value.clone_into(&mut spec.template),
                "--name" => value.clone_into(&mut spec.name),
                "--teacher" => spec.teachers.push(TeacherRun {
                    run: value.into(),
                    evals: None,
                }),
                "--evals" => {
                    let last = spec
                        .teachers
                        .last_mut()
                        .ok_or("--evals follows a --teacher")?;
                    last.evals = Some(value.clone());
                }
                "--teacher-bundle" => spec.bundle = Some(value.into()),
                "--student" => spec.students.push(value.into()),
                "--mode" => {
                    spec.mode = match value.as_str() {
                        "link" => Mode::Link,
                        "move" => Mode::Move,
                        "copy" => Mode::Copy,
                        other => return Err(format!("--mode {other}: link, move or copy")),
                    }
                }
                other => return Err(format!("{other}: not an --import flag")),
            }
        }
        if spec.template.is_empty() {
            return Err("--import needs --template <id>".into());
        }
        Ok(spec)
    }

    /// Makes the project ([`Project::create`], so it refuses a folder that is one already),
    /// brings every output in, and returns it with the teacher it found, if any.
    pub fn run(
        &self,
        template: &Template,
        repo: &Path,
    ) -> Result<(Project, Option<Choice>), ProjectError> {
        let project = Project::create(&self.dest, &self.name, template, repo)?;
        for t in &self.teachers {
            let dir = project.next_teacher_dir();
            bring_entries(&t.run, &dir, self.mode)?;
            let run = RunFolder {
                number: (project.teacher_runs().last()).map_or(0, |r| r.number),
                path: dir,
            };
            for mark in teacher::marks(&run) {
                let from =
                    PathBuf::from(format!("{}{}", t.evals.as_deref().unwrap_or(""), mark.step));
                if t.evals.is_some() && from.join("report.json").is_file() {
                    let to = teacher::eval_dir(&run, mark.step);
                    std::fs::create_dir_all(to.parent().unwrap_or(&to))
                        .map_err(|e| fail(&to, e))?;
                    bring(&from, &to, self.mode)?;
                }
            }
        }
        let choice = match &self.bundle {
            Some(bundle) => {
                let target = project.teacher_bundle();
                std::fs::copy(bundle, &target).map_err(|e| fail(bundle, e))?;
                let choice = find_checkpoint(&project, &target)?;
                if let Some(c) = choice {
                    c.write(&project)?;
                }
                choice
            }
            None => None,
        };
        for student in &self.students {
            let dir = project.next_run_dir();
            bring_entries(student, &dir, self.mode)?;
            let recipe = dir.join(RUN_RECIPE);
            if !recipe.exists() {
                let cycle = repo.join(&template.cycle);
                std::fs::copy(&cycle, &recipe).map_err(|e| fail(&cycle, e))?;
            }
        }
        Ok((project, choice))
    }
}

/// The teacher checkpoint whose weights `bundle` carries, newest run first.
fn find_checkpoint(project: &Project, bundle: &Path) -> Result<Option<Choice>, ProjectError> {
    let weights = |path: &Path| {
        let bytes = std::fs::read(path).map_err(|e| fail(path, e))?;
        let b = PolicyBundle::open(&bytes).map_err(|e| fail(path, e))?;
        Ok::<_, ProjectError>(*b.learning.policy.weights.hash())
    };
    let want = weights(bundle)?;
    for run in project.teacher_runs().iter().rev() {
        for mark in teacher::marks(run) {
            if weights(&teacher::checkpoint(run, mark.step))? == want {
                return Ok(Some(Choice {
                    run: run.number,
                    step: mark.step,
                }));
            }
        }
    }
    Ok(None)
}

/// `from`'s entries into a new folder `to`: each folder by `mode`, each file copied (a link or
/// a move of the folder's own files would leave nowhere to write `cycle.toml` but the source).
fn bring_entries(from: &Path, to: &Path, mode: Mode) -> Result<(), ProjectError> {
    if !from.is_dir() {
        return Err(fail(from, "not a folder"));
    }
    std::fs::create_dir_all(to).map_err(|e| fail(to, e))?;
    for entry in std::fs::read_dir(from)
        .map_err(|e| fail(from, e))?
        .flatten()
    {
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        if src.is_dir() {
            bring(&src, &dst, mode)?;
        } else {
            std::fs::copy(&src, &dst).map_err(|e| fail(&src, e))?;
        }
    }
    Ok(())
}

/// One folder, by `mode`.
fn bring(from: &Path, to: &Path, mode: Mode) -> Result<(), ProjectError> {
    match mode {
        Mode::Move => std::fs::rename(from, to).map_err(|e| fail(from, e)),
        Mode::Copy => copy_dir(from, to),
        Mode::Link => link_dir(from, to),
    }
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), ProjectError> {
    std::fs::create_dir_all(to).map_err(|e| fail(to, e))?;
    for entry in std::fs::read_dir(from)
        .map_err(|e| fail(from, e))?
        .flatten()
    {
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        if src.is_dir() {
            copy_dir(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst).map_err(|e| fail(&src, e))?;
        }
    }
    Ok(())
}

/// A directory junction: no administrator right or developer mode needed, unlike a Windows
/// symbolic link.
#[cfg(windows)]
fn link_dir(from: &Path, to: &Path) -> Result<(), ProjectError> {
    let from = std::path::absolute(from).map_err(|e| fail(from, e))?;
    let out = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(to)
        .arg(&from)
        .output()
        .map_err(|e| fail(to, e))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(fail(to, String::from_utf8_lossy(&out.stderr).trim()))
    }
}

#[cfg(not(windows))]
fn link_dir(from: &Path, to: &Path) -> Result<(), ProjectError> {
    let from = std::path::absolute(from).map_err(|e| fail(from, e))?;
    std::os::unix::fs::symlink(from, to).map_err(|e| fail(to, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::project::tests::repo;
    use crate::model::results::{finished_runs, RunResults};
    use crate::model::teacher::tests::{hand, hand_project, write_report};
    use crate::model::teacher::{chosen, chosen_score, eval_dir, marks, untrained, Score};

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_command_line_is_read_flag_by_flag() {
        let spec = Import::parse(&args(
            "p/hand --template shadow-hand-repose --teacher t1 --teacher t2 --evals e/t2- \
             --teacher-bundle t.esb --student s1 --mode move",
        ))
        .unwrap();
        assert_eq!(spec.name, "hand");
        assert_eq!(spec.mode, Mode::Move);
        assert_eq!(
            spec.teachers,
            [
                TeacherRun {
                    run: "t1".into(),
                    evals: None
                },
                TeacherRun {
                    run: "t2".into(),
                    evals: Some("e/t2-".into())
                }
            ]
        );
        assert_eq!(spec.students, [PathBuf::from("s1")]);
        for bad in [
            "p",
            "p --template",
            "p --template x --evals e",
            "p --template x --mode hardlink",
            "p --template x --what y",
        ] {
            assert!(Import::parse(&args(bad)).is_err(), "{bad}");
        }
        assert_eq!(
            Import::parse(&args("p --template x")).unwrap().mode,
            Mode::Link
        );
    }

    /// Outputs laid out as plan H's E1/E2 left them - a teacher folder, its evaluations beside
    /// it, the bundle the student collected with, a student cycle folder - in every mode.
    #[test]
    fn outputs_from_elsewhere_become_a_project() {
        for mode in [Mode::Link, Mode::Copy, Mode::Move] {
            let base =
                std::env::temp_dir().join(format!("es-h7-import-{}-{mode:?}", std::process::id()));
            let _ = std::fs::remove_dir_all(&base);
            // The outputs, made through a scratch project's own teacher run.
            let scratch = hand_project(&format!("import-src-{mode:?}"));
            let bytes = std::fs::read(untrained(&hand(), &repo(), &scratch).unwrap()).unwrap();
            let outputs = base.join("outputs");
            let train = outputs.join("teacher-002");
            std::fs::create_dir_all(train.join("checkpoints")).unwrap();
            for step in [250, 500] {
                std::fs::write(train.join(format!("checkpoints/{step}.esb")), &bytes).unwrap();
            }
            std::fs::write(train.join("training.lock"), "{}").unwrap();
            write_report(&outputs.join("eval64-t2-500"), 33);
            std::fs::write(outputs.join("teacher.esb"), &bytes).unwrap();
            let student = outputs.join("student-001");
            let fixture = repo().join("tests/fixtures/visible-learning/run");
            copy_dir(&fixture, &student.join("eval")).unwrap();
            std::fs::write(student.join("loop.jsonl"), "").unwrap();

            let spec = Import {
                dest: base.join("project"),
                name: "hand".into(),
                template: "shadow-hand-repose".into(),
                teachers: vec![TeacherRun {
                    run: train.clone(),
                    evals: Some(format!("{}", outputs.join("eval64-t2-").display())),
                }],
                bundle: Some(outputs.join("teacher.esb")),
                students: vec![student.clone()],
                mode,
            };
            let (p, choice) = spec.run(&hand(), &repo()).unwrap();
            let p = Project::open(&p.root).unwrap();
            let run = p.teacher_runs().pop().expect("teacher/001");
            let m = marks(&run);
            assert_eq!(m.iter().map(|m| m.step).collect::<Vec<_>>(), [250, 500]);
            let judged = Score {
                successes: 33,
                episodes: 64,
            };
            assert_eq!((m[0].score, m[1].score), (None, Some(judged)), "{mode:?}");
            assert!(eval_dir(&run, 500).join("report.json").is_file());
            // Both checkpoints carry the bundle's weights: the first is the one named.
            assert_eq!(choice, Some(Choice { run: 1, step: 250 }));
            assert_eq!(chosen(&p), choice);
            assert_eq!(chosen_score(&p), None, "step 250 was not judged");
            let runs = finished_runs(&p);
            assert_eq!(runs.len(), 1, "{mode:?}");
            assert!(runs[0].path.join(RUN_RECIPE).is_file());
            let results = RunResults::read(&p, &runs[0], Some(&repo())).unwrap();
            assert_eq!(results.teacher, None, "the chosen step was not judged");
            // Link and copy leave the outputs where they are; move takes their folders.
            assert_eq!(
                student.join("eval").is_dir(),
                mode != Mode::Move,
                "{mode:?}"
            );
            assert!(!student.join(RUN_RECIPE).exists(), "nothing written there");
            std::fs::remove_dir_all(&base).ok();
            std::fs::remove_dir_all(&scratch.root).ok();
        }
    }
}
