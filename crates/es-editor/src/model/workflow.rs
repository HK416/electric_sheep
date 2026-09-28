//! The step bar (`docs/design/editor-redesign.md` section 6.3, packet M12/Y6): the state of
//! each of the five steps, as a pure function of what disk says about the latest run
//! ([`RunFacts`]) and what the live connection and the child say ([`LiveFacts`]).
//!
//! ① and ② are done for a template. ③ covers the cycle's collect, expert gate and train; ④
//! its eval and showcase; ⑤ opens once `eval/report.json` exists, whatever happened after it.

use es_data::collect::{read_loop_steps, LoopKind, LoopStep};
use es_data::training::{Cycle, Stage};

use crate::model::live_run::StageRow;
use crate::model::project::{RunFolder, RUN_RECIPE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Scene,
    Teach,
    Train,
    Evaluate,
    Results,
}

impl Phase {
    pub const ALL: [Phase; 5] = [
        Phase::Scene,
        Phase::Teach,
        Phase::Train,
        Phase::Evaluate,
        Phase::Results,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Phase::Scene => "phase.scene",
            Phase::Teach => "phase.teach",
            Phase::Train => "phase.train",
            Phase::Evaluate => "phase.evaluate",
            Phase::Results => "phase.results",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PhaseState {
    Done,
    NotStarted,
    Locked,
    Running {
        stage: String,
        fraction: Option<f32>,
    },
    Failed {
        stage: String,
        code: Option<i32>,
    },
    Interrupted {
        resume_from: Option<String>,
    },
    StoppedByYou {
        resume_from: Option<String>,
    },
}

impl PhaseState {
    /// `run.<state>`: the state's word on the step bar.
    pub fn key(&self) -> &'static str {
        match self {
            PhaseState::Done => "run.done",
            PhaseState::NotStarted => "run.not_started",
            PhaseState::Locked => "run.locked",
            PhaseState::Running { .. } => "run.running",
            PhaseState::Failed { .. } => "run.failed",
            PhaseState::Interrupted { .. } => "run.interrupted",
            PhaseState::StoppedByYou { .. } => "run.stopped_by_you",
        }
    }
}

/// What disk says about one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunFacts {
    pub collected: bool,
    pub trained: bool,
    pub evaluated: bool,
    pub report: bool,
}

impl RunFacts {
    /// Reads the run's `loop.jsonl` and `eval/report.json`. The rows `es loop cycle` appends
    /// to `<out>/loop.jsonl` (`crates/es/src/cmd/cycle.rs`), and what each means here:
    ///
    /// | stage | row | fact |
    /// |---|---|---|
    /// | collect | `collect`, mirrored from the dataset's own ledger | `collected`, once the gate agrees |
    /// | expert-gate | `evaluate` with an `expert` input and a `passed` output, written whether it passed or not | gates `collected`; never `evaluated` |
    /// | train | `train` | `trained` |
    /// | eval | `evaluate` without `expert` (it names a `policy_hash`) | `evaluated` |
    /// | showcase | none | - |
    ///
    /// When the run's own recipe arms the gate (`[collect] expert`), `collected` needs the
    /// latest gate row to say `passed = true`: a dataset the harness never judged, or judged
    /// failing, is not one to resume training on, and `--from` has no expert-gate stage to
    /// resume at (`Stage::parse`). Showcase writes no row, so a run stopped during it reads as
    /// evaluated. A ledger that cannot be read counts as an empty one.
    pub fn read(run: &RunFolder) -> Self {
        let steps = read_loop_steps(&run.path).unwrap_or_default();
        let is_gate =
            |s: &LoopStep| s.kind == LoopKind::Evaluate && s.inputs.contains_key("expert");
        let gated = std::fs::read_to_string(run.path.join(RUN_RECIPE))
            .ok()
            .and_then(|text| Cycle::parse(&text).ok())
            .and_then(|cycle| cycle.collect)
            .is_some_and(|collect| collect.expert.is_some());
        let gate_ok = match steps.iter().rev().find(|s| is_gate(s)) {
            Some(gate) => gate.outputs.get("passed").is_some_and(|p| p == "true"),
            None => !gated,
        };
        let any = |kind: LoopKind| steps.iter().any(|s| s.kind == kind && !is_gate(s));
        Self {
            collected: gate_ok && any(LoopKind::Collect),
            trained: any(LoopKind::Train),
            evaluated: any(LoopKind::Evaluate),
            report: run.report_path().is_file(),
        }
    }

    /// The first stage these facts do not cover, spelled as `es loop cycle --from` reads it;
    /// `None` when nothing is covered, where resuming is starting over.
    fn resume_point(&self) -> Option<String> {
        let stage = if self.evaluated {
            Stage::Showcase
        } else if self.trained {
            Stage::Eval
        } else if self.collected {
            Stage::Train
        } else {
            return None;
        };
        Some(stage.as_str().to_owned())
    }
}

/// What the live connection and the child say. `None` = not attached.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveFacts {
    /// Stage name, end code once it ended.
    pub stages: Vec<(String, Option<i32>)>,
    /// Of the stage in progress.
    pub fraction: Option<f32>,
    pub child: Child,
}

impl LiveFacts {
    /// From the live run's `stage.begin` / `stage.end` rows.
    pub fn new(rows: &[StageRow], fraction: Option<f32>, child: Child) -> Self {
        Self {
            stages: rows
                .iter()
                .map(|r| (r.name.clone(), r.code.map(i32::from)))
                .collect(),
            fraction,
            child,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child {
    Running,
    Exited(i32),
    Killed,
    /// Attached to a run this editor did not start (re-opened, through `telemetry.txt`).
    NotOurs,
}

/// The cycle's stages under ③ and ④, in `Stage::as_str`'s words.
const TRAIN_STAGES: &[&str] = &["collect", "expert-gate", "train"];
const EVALUATE_STAGES: &[&str] = &["eval", "showcase"];

/// ③ = collect, expert-gate, train; ④ = eval, showcase. ① ② are Done for a template.
pub fn phases(run: Option<&RunFacts>, live: Option<&LiveFacts>) -> [PhaseState; 5] {
    use PhaseState::{Done, Interrupted, Locked, NotStarted};
    let empty = RunFacts::default();
    let f = run.unwrap_or(&empty);
    let results = if f.report { Done } else { Locked };
    let (train, evaluate) = match (run, live) {
        (None, None) => (NotStarted, Locked),
        // On disk only: whatever is not there was interrupted - the editor or the run died.
        (Some(_), None) => {
            let interrupted = || Interrupted {
                resume_from: f.resume_point(),
            };
            let train = if f.trained { Done } else { interrupted() };
            let evaluate = match (f.trained, f.evaluated) {
                (false, _) => Locked,
                (true, true) => Done,
                (true, false) => interrupted(),
            };
            (train, evaluate)
        }
        (_, Some(l)) => {
            let train = live_phase(l, TRAIN_STAGES, f.trained, &Done, f);
            let evaluate = live_phase(l, EVALUATE_STAGES, f.evaluated, &train, f);
            (train, evaluate)
        }
    };
    [Done, Done, train, evaluate, results]
}

/// One of ③ ④ while attached. A stage that ended badly fails the phase; a stage in progress is
/// what the child is doing (or died doing); a phase whose stages all ended well - or that a
/// resumed run skipped because disk has it - is done. A phase with nothing yet is locked
/// behind an unfinished one, or is the one the child is about to begin.
fn live_phase(
    l: &LiveFacts,
    stages: &[&str],
    on_disk: bool,
    before: &PhaseState,
    f: &RunFacts,
) -> PhaseState {
    use PhaseState::{Done, Failed, Locked, Running, StoppedByYou};
    let rows: Vec<&(String, Option<i32>)> = l
        .stages
        .iter()
        .filter(|(s, _)| stages.contains(&s.as_str()))
        .collect();
    if let Some((stage, code)) = rows.iter().find(|(_, c)| c.is_some_and(|c| c != 0)) {
        return Failed {
            stage: stage.clone(),
            code: *code,
        };
    }
    let (stage, fraction) = match rows.iter().find(|(_, c)| c.is_none()) {
        Some((stage, _)) => (stage.clone(), l.fraction),
        None if !rows.is_empty() || on_disk => return Done,
        None if *before != Done => return Locked,
        // About to begin: the resume point when it is one of this phase's, else its first.
        None => (
            f.resume_point()
                .filter(|s| stages.contains(&s.as_str()))
                .unwrap_or_else(|| stages[0].to_owned()),
            None,
        ),
    };
    match l.child {
        Child::Running | Child::NotOurs => Running { stage, fraction },
        // A clean exit whose `stage.end` was not seen still finished the stage.
        Child::Exited(0) => Done,
        Child::Exited(code) => Failed {
            stage,
            code: Some(code),
        },
        Child::Killed => StoppedByYou {
            resume_from: f.resume_point(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::{t, Lang};
    use crate::model::project::tests::{cube, repo, scratch_project};
    use crate::model::project::RUN_RECIPE;
    use es_data::collect::{append_loop_step, LoopKind, LoopStep};
    use es_data::training::Stage;

    fn live(stages: &[(&str, Option<i32>)], child: Child) -> LiveFacts {
        LiveFacts {
            stages: stages.iter().map(|(s, c)| ((*s).to_owned(), *c)).collect(),
            fraction: Some(0.5),
            child,
        }
    }
    use PhaseState::*;

    #[test]
    fn no_run_yet() {
        assert_eq!(phases(None, None), [Done, Done, NotStarted, Locked, Locked]);
    }
    #[test]
    fn collecting_is_train_running() {
        let s = phases(
            Some(&RunFacts::default()),
            Some(&live(&[("collect", None)], Child::Running)),
        );
        assert_eq!(
            s[2],
            Running {
                stage: "collect".into(),
                fraction: Some(0.5)
            }
        );
        assert_eq!((s[3].clone(), s[4].clone()), (Locked, Locked));
    }
    #[test]
    fn evaluating_after_training() {
        let l = live(
            &[
                ("collect", Some(0)),
                ("expert-gate", Some(0)),
                ("train", Some(0)),
                ("eval", None),
            ],
            Child::Running,
        );
        let s = phases(Some(&RunFacts::default()), Some(&l));
        assert_eq!(s[2], Done);
        assert!(matches!(&s[3], Running { stage, .. } if stage == "eval"));
    }
    #[test]
    fn a_stage_that_ends_badly_fails_its_phase() {
        let l = live(
            &[("collect", Some(0)), ("expert-gate", Some(4))],
            Child::Exited(4),
        );
        let s = phases(Some(&RunFacts::default()), Some(&l));
        assert_eq!(
            s[2],
            Failed {
                stage: "expert-gate".into(),
                code: Some(4)
            }
        );
        assert_eq!(s[3], Locked);
    }
    #[test]
    fn a_child_that_dies_mid_stage_fails_that_stage() {
        let l = live(&[("collect", Some(0)), ("train", None)], Child::Exited(101));
        assert_eq!(
            phases(Some(&RunFacts::default()), Some(&l))[2],
            Failed {
                stage: "train".into(),
                code: Some(101)
            }
        );
    }
    #[test]
    fn a_kill_is_stopped_by_you_with_a_resume_point() {
        let l = live(&[("collect", Some(0)), ("train", None)], Child::Killed);
        let f = RunFacts {
            collected: true,
            ..Default::default()
        };
        assert_eq!(
            phases(Some(&f), Some(&l))[2],
            StoppedByYou {
                resume_from: Some("train".into())
            }
        );
    }
    #[test]
    fn a_finished_run_on_disk() {
        let f = RunFacts {
            collected: true,
            trained: true,
            evaluated: true,
            report: true,
        };
        assert_eq!(phases(Some(&f), None), [Done, Done, Done, Done, Done]);
    }
    #[test]
    fn a_dead_address_reads_as_interrupted_with_the_next_stage() {
        let f = RunFacts {
            collected: true,
            trained: true,
            ..Default::default()
        };
        let s = phases(Some(&f), None);
        assert_eq!(s[2], Done);
        assert_eq!(
            s[3],
            Interrupted {
                resume_from: Some("eval".into())
            }
        );
        let only_collect = RunFacts {
            collected: true,
            ..Default::default()
        };
        assert_eq!(
            phases(Some(&only_collect), None)[2],
            Interrupted {
                resume_from: Some("train".into())
            }
        );
        assert_eq!(
            phases(Some(&RunFacts::default()), None)[2],
            Interrupted { resume_from: None }
        );
    }
    #[test]
    fn results_open_whenever_a_report_exists() {
        // showcase failed after eval wrote its report: ④ failed, ⑤ still open
        let l = live(
            &[
                ("collect", Some(0)),
                ("train", Some(0)),
                ("eval", Some(0)),
                ("showcase", Some(1)),
            ],
            Child::Exited(1),
        );
        let f = RunFacts {
            collected: true,
            trained: true,
            evaluated: true,
            report: true,
        };
        let s = phases(Some(&f), Some(&l));
        assert_eq!(
            s[3],
            Failed {
                stage: "showcase".into(),
                code: Some(1)
            }
        );
        assert_eq!(s[4], Done);
    }

    /// A run started with `--from` has no live row for the stages it skipped: those are what
    /// disk says. A child that ends before its first stage fails the stage it was going to run.
    #[test]
    fn a_resumed_run_takes_the_skipped_stages_from_disk() {
        let f = RunFacts {
            collected: true,
            trained: true,
            ..Default::default()
        };
        let s = phases(Some(&f), Some(&live(&[("eval", None)], Child::Running)));
        assert_eq!(s[2], Done);
        assert!(matches!(&s[3], Running { stage, .. } if stage == "eval"));
        let s = phases(Some(&f), Some(&live(&[], Child::Exited(1))));
        assert_eq!(
            s[3],
            Failed {
                stage: "eval".into(),
                code: Some(1)
            }
        );
        let s = phases(
            Some(&RunFacts::default()),
            Some(&live(&[], Child::Exited(2))),
        );
        assert_eq!(
            s[2],
            Failed {
                stage: "collect".into(),
                code: Some(2)
            }
        );
    }

    /// Every resume point is a word `es loop cycle --from` reads.
    #[test]
    fn every_resume_point_is_a_from_stage() {
        for (c, t, e) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, true, true),
        ] {
            let f = RunFacts {
                collected: c,
                trained: t,
                evaluated: e,
                report: e,
            };
            let l = live(&[("collect", None)], Child::Killed);
            for state in phases(Some(&f), None)
                .into_iter()
                .chain(phases(Some(&f), Some(&l)))
            {
                if let Interrupted {
                    resume_from: Some(s),
                }
                | StoppedByYou {
                    resume_from: Some(s),
                } = &state
                {
                    assert!(Stage::parse(s).is_some(), "{s}");
                }
            }
        }
    }

    /// The ledger rows `es loop cycle` writes, and what each means to the step bar.
    #[test]
    fn ledger_rows_map_to_facts() {
        let p = scratch_project("facts");
        let path = p.next_run_dir();
        std::fs::create_dir_all(&path).unwrap();
        // The run's recipe arms the expert gate, as both templates' do.
        std::fs::copy(repo().join(cube().cycle), path.join(RUN_RECIPE)).unwrap();
        let run = p.latest_run().unwrap();
        let row = |step: LoopStep| append_loop_step(&run.path, &step).unwrap();
        let gate = |passed: bool| {
            LoopStep::new(LoopKind::Evaluate)
                .input("expert", &"so101-pick-place")
                .output("passed", &passed)
        };
        assert_eq!(RunFacts::read(&run), RunFacts::default());

        row(LoopStep::new(LoopKind::Collect));
        assert_eq!(
            RunFacts::read(&run),
            RunFacts::default(),
            "collected, but the gate has not judged it"
        );
        row(gate(false));
        assert_eq!(
            RunFacts::read(&run),
            RunFacts::default(),
            "a failed gate is no data to train on"
        );
        row(gate(true));
        let collected = RunFacts {
            collected: true,
            ..Default::default()
        };
        assert_eq!(RunFacts::read(&run), collected);
        row(LoopStep::new(LoopKind::Train));
        let trained = RunFacts {
            trained: true,
            ..collected
        };
        assert_eq!(
            RunFacts::read(&run),
            trained,
            "the gate is not an evaluation"
        );
        row(LoopStep::new(LoopKind::Evaluate).input("policy_hash", &"00"));
        assert_eq!(
            RunFacts::read(&run),
            RunFacts {
                evaluated: true,
                ..trained.clone()
            }
        );
        std::fs::create_dir_all(run.eval_dir()).unwrap();
        std::fs::write(run.report_path(), "{}").unwrap();
        assert!(RunFacts::read(&run).report);

        // Without `[collect] expert` there is no gate, and a collect row is enough.
        let bare = RunFolder {
            number: 0,
            path: p.root.join("bare"),
        };
        std::fs::create_dir_all(&bare.path).unwrap();
        append_loop_step(&bare.path, &LoopStep::new(LoopKind::Collect)).unwrap();
        assert!(RunFacts::read(&bare).collected);
        std::fs::remove_dir_all(&p.root).ok();
    }

    #[test]
    fn live_facts_come_from_the_stage_rows() {
        let rows = [
            StageRow {
                name: "collect".into(),
                seconds: Some(1.0),
                code: Some(0),
            },
            StageRow {
                name: "train".into(),
                seconds: None,
                code: None,
            },
        ];
        assert_eq!(
            LiveFacts::new(&rows, Some(0.5), Child::Running),
            live(&[("collect", Some(0)), ("train", None)], Child::Running)
        );
    }

    /// Every step and every state has a word in both tables.
    #[test]
    fn every_phase_and_state_has_words() {
        let states = [
            Done,
            NotStarted,
            Locked,
            Running {
                stage: String::new(),
                fraction: None,
            },
            Failed {
                stage: String::new(),
                code: None,
            },
            Interrupted { resume_from: None },
            StoppedByYou { resume_from: None },
        ];
        for lang in Lang::ALL {
            let keys = Phase::ALL
                .iter()
                .map(|p| p.key())
                .chain(states.iter().map(PhaseState::key));
            for key in keys {
                assert_ne!(t(lang, key), key, "{lang:?}: {key}");
            }
        }
    }
}
