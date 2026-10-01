//! Why an authored project's attempts failed: the clause that was missing at the end (packet
//! M17/R8, `docs/design/scene-authoring.md` section 4.8). Each failed attempt's end state, its
//! trajectory's last row, is read back through the env's own lowering of every clause
//! ([`eval_on_row`]) on the layout `MuJoCo` gives the scene ([`layout`]): no IR change, no hash
//! change. A success clause explains the attempts that ended without it, a failure clause the
//! attempts it ended; `es-editor` words them with G8's sentences and hands them to ⑤. When the
//! success section holds for a while (packet M18/K7), an attempt that ended with every success
//! clause true is explained by how long they had held: counted back from its end row, one row per
//! control tick, with the same per-clause reading.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use es_env::plan::eval_on_row;
use es_env::traj::Trajectory;
use es_eval::episodes::EpisodeRow;
use es_eval::run_dir::RunDir;
use es_ir::serial::evaluation_from_toml;
use es_ir::task::TerminationKind;
use es_physics_backend::layout;
use es_script::spec::{compile_clauses, load_scene, TaskSpec};

use crate::sentence::{self, At, Sentence};
use crate::{GENERATED_DIR, SPEC_FILE};

/// One clause that explains failed attempts, in G8's sentence.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub at: At,
    pub sentence: Sentence,
    /// A success clause: the failed attempts that ended without it. A failure clause: the
    /// attempts it ended.
    pub attempts: u32,
}

/// A run's failed attempts, explained.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Explanation {
    /// Most attempts first, the document's order between equals; a clause that explains no
    /// attempt is not listed.
    pub lines: Vec<Line>,
    /// The failed attempts whose end state was read.
    pub failed: u32,
    /// Each of those attempts' clauses, by cell: the failure clauses that ended it, then the
    /// success clauses it ended without, each in the document's order.
    pub cells: BTreeMap<String, Vec<At>>,
    /// Some attempt was read on its last recorded row, the state before its last step: a run
    /// written before trajectories ended on the end state (review M17 F-7).
    pub before_end: bool,
    /// The success section's `hold_s`, when it has one (packet M18/K7).
    pub hold_s: Option<f64>,
    /// With a hold: each failed attempt whose end row held every success clause, by cell, and for
    /// how long they had held then, in seconds — less than `hold_s`, or it would have succeeded.
    pub held: BTreeMap<String, f64>,
}

/// The failed attempts of `run` (`rows`) explained by the clauses of the specification at
/// `root`. `None` when the project has none that compiles, or when the run did not evaluate the
/// task it compiles to: a run from before the task changed is not explained by clauses it did
/// not have.
pub fn explain(root: &Path, run: &RunDir, rows: &[EpisodeRow]) -> Option<Explanation> {
    let spec = TaskSpec::from_toml(&std::fs::read_to_string(root.join(SPEC_FILE)).ok()?).ok()?;
    let (task, clauses) = compile_clauses(&spec, root).ok()?;
    let hex = (task.task_hash().ok()?.iter()).fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    });
    if evaluated_task(root, run)? != hex {
        return None;
    }
    let (scene, _) = load_scene(&root.join(&spec.scene)).ok()?;
    let model = layout(&scene);
    let nodes: Vec<_> = clauses.iter().map(|&(_, _, node)| node).collect();
    let success: Vec<_> = (clauses.iter())
        .filter(|c| c.0 == TerminationKind::Success)
        .map(|c| c.2)
        .collect();
    // The control ticks the env had counted toward the hold when an attempt ended: the rows in a
    // row, back from its end row, on which every success clause holds. Row 0 is the reset state,
    // which no control tick evaluated.
    let held = |traj: &Trajectory, last: usize| {
        let all = |row| {
            let truth = eval_on_row(&task, &scene, &model, &success, traj, row);
            truth.is_ok_and(|v| v.iter().all(|x| *x != 0.0))
        };
        (1..=last).rev().take_while(|r| all(*r)).count() as u32
    };
    let at = |&(kind, index, _): &(TerminationKind, usize, _)| At {
        failure: kind == TerminationKind::Failure,
        index,
    };
    let mut out = Explanation {
        hold_s: spec.success.hold_s,
        ..Explanation::default()
    };
    let mut counts = vec![0; clauses.len()];
    for row in rows.iter().filter(|r| r.termination != "success") {
        let Ok(traj) = Trajectory::read(&run.traj_path(&row.cell)) else {
            continue;
        };
        let Some(last) = traj.ticks().checked_sub(1) else {
            continue;
        };
        let Ok(truth) = eval_on_row(&task, &scene, &model, &nodes, &traj, last) else {
            continue;
        };
        // A row per frame, and since R2 one more, the end state. Without frames, a row per step.
        let frames = (run.cells().iter().find(|c| c.name == row.cell)).map_or(0, |c| c.frames);
        let captured = if frames > 0 { frames as u64 } else { row.steps };
        out.before_end |= traj.ticks() as u64 != captured + 1;
        out.failed += 1;
        // A failure clause explains an attempt that held it, a success clause one that did not.
        let mut why: Vec<At> = (clauses.iter().zip(&truth).enumerate())
            .filter(|(_, (c, v))| (**v != 0.0) == (c.0 == TerminationKind::Failure))
            .map(|(i, (c, _))| {
                counts[i] += 1;
                at(c)
            })
            .collect();
        why.sort_by_key(|a| !a.failure);
        if out.hold_s.is_some() && why.iter().all(|a| a.failure) {
            let ticks = held(&traj, last);
            (out.held).insert(row.cell.clone(), f64::from(ticks) / spec.control_hz);
        }
        out.cells.insert(row.cell.clone(), why);
    }
    for (c, &attempts) in clauses.iter().zip(&counts).filter(|(_, n)| **n > 0) {
        let at = at(c);
        let section = if at.failure {
            spec.failure.as_ref()?
        } else {
            &spec.success
        };
        let sentence = sentence::clause(&scene, section.clauses.get(at.index)?);
        out.lines.push(Line {
            at,
            sentence,
            attempts,
        });
    }
    // Stable: the document's order between equals.
    out.lines.sort_by_key(|l| std::cmp::Reverse(l.attempts));
    Some(out)
}

/// The Task IR hash the run was evaluated on: the `task` of the generated evaluation document
/// whose hash is the run's report's (an Evaluation IR names its task, spec 10.2).
fn evaluated_task(root: &Path, run: &RunDir) -> Option<String> {
    let dir = std::fs::read_dir(root.join(GENERATED_DIR)).ok()?;
    dir.flatten().find_map(|entry| {
        entry
            .file_name()
            .to_str()?
            .starts_with("evaluation")
            .then_some(())?;
        let ir = evaluation_from_toml(&std::fs::read_to_string(entry.path()).ok()?).ok()?;
        (ir.evaluation_hash().ok()? == run.report.evaluation_hash).then_some(ir.task)
    })
}
