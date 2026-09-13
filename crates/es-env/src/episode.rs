//! Episode recording (§13.1: the recorded episode is what feeds the learning loop) and
//! termination evaluation (§6.3 `Terminate`).
//!
//! Storage is columnar per env and pre-sized from `max_episode_steps`, so a step never
//! allocates.

use std::collections::BTreeMap;

use es_core::{FailureKind, PhysTick};
use es_ir::task::TerminationKind;

use crate::plan::ScalarPlan;
use crate::randomize::ParamScales;

/// Why an episode ended, or that it has not (§6.3: success | failure | timeout).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Termination {
    #[default]
    Running,
    Success,
    Failure,
    Timeout,
}

impl Termination {
    pub fn is_done(self) -> bool {
        self != Self::Running
    }

    fn of(kind: TerminationKind) -> Self {
        match kind {
            TerminationKind::Success => Self::Success,
            TerminationKind::Failure => Self::Failure,
            TerminationKind::Timeout => Self::Timeout,
        }
    }
}

/// Evaluates the task's `Terminate` predicates, then the episode budget.
///
/// Predicates are taken in the order they were lowered (ascending `NodeId`, §6.4) and the first
/// one that holds wins, so two predicates firing on the same tick resolve identically every
/// run. A predicate that cannot be evaluated (a missing port, a division by zero — `Expr::eval`
/// returns `None` rather than a `NaN`) counts as "did not fire".
pub(crate) fn evaluate(
    plan: &ScalarPlan,
    ports: &BTreeMap<String, f64>,
    steps: u32,
    max_steps: u32,
) -> Termination {
    for (kind, expr) in &plan.terminations {
        if expr.eval(ports).is_some_and(|v| v != 0.0) {
            return Termination::of(*kind);
        }
    }
    if max_steps > 0 && steps >= max_steps {
        return Termination::Timeout;
    }
    Termination::Running
}

/// The per-env row widths an episode records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EpisodeShape {
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub nsensordata: usize,
}

/// One finished (or in-progress) episode of one env, in columns.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Episode {
    pub env: u32,
    pub episode: u64,
    pub shape: EpisodeShape,
    /// Physics tick of each step; `steps()` entries.
    pub ticks: Vec<PhysTick>,
    /// `steps() * shape.nq`, row-major by step.
    pub qpos: Vec<f64>,
    /// `steps() * shape.nv`.
    pub qvel: Vec<f64>,
    /// `steps() * shape.nu`.
    pub ctrl: Vec<f64>,
    /// `steps() * shape.nsensordata`.
    pub sensordata: Vec<f64>,
    /// `steps()`.
    pub reward: Vec<f64>,
    /// `steps()`; only the last entry can be true.
    pub done: Vec<bool>,
    /// `steps()`; the failure the backend reported on that step, if any (§18.5).
    pub failure: Vec<Option<FailureKind>>,
    pub termination: Termination,
    /// The model-parameter scales this episode was randomized with (§5 of the design note).
    pub param_scales: ParamScales,
}

impl Episode {
    pub fn steps(&self) -> usize {
        self.ticks.len()
    }

    fn fresh(env: u32, episode: u64, shape: EpisodeShape, max_steps: usize) -> Self {
        let column = |w: usize| Vec::with_capacity(max_steps * w);
        Self {
            env,
            episode,
            shape,
            ticks: Vec::with_capacity(max_steps),
            qpos: column(shape.nq),
            qvel: column(shape.nv),
            ctrl: column(shape.nu),
            sensordata: column(shape.nsensordata),
            reward: Vec::with_capacity(max_steps),
            done: Vec::with_capacity(max_steps),
            failure: Vec::with_capacity(max_steps),
            termination: Termination::Running,
            param_scales: ParamScales::new(),
        }
    }
}

/// One step's worth of one env.
#[derive(Clone, Copy, Debug)]
pub struct StepRow<'a> {
    pub tick: PhysTick,
    pub qpos: &'a [f64],
    pub qvel: &'a [f64],
    pub ctrl: &'a [f64],
    pub sensordata: &'a [f64],
    pub reward: f64,
    pub termination: Termination,
    pub failure: Option<FailureKind>,
}

/// Records every env's current episode; [`finish`](Self::finish) hands one back and starts the
/// next.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeRecorder {
    shape: EpisodeShape,
    max_steps: usize,
    open: Vec<Episode>,
}

impl EpisodeRecorder {
    pub fn new(n_envs: u32, shape: EpisodeShape, max_steps: u32) -> Self {
        let max_steps = max_steps as usize;
        Self {
            shape,
            max_steps,
            open: (0..n_envs)
                .map(|env| Episode::fresh(env, 0, shape, max_steps))
                .collect(),
        }
    }

    pub fn shape(&self) -> EpisodeShape {
        self.shape
    }

    /// The in-progress episode of `env`.
    pub fn open(&self, env: u32) -> &Episode {
        &self.open[env as usize]
    }

    /// Appends one step. Slices shorter than the shape are padded with zeros and longer ones
    /// truncated, so a backend that reports fewer sensors than declared cannot desynchronize
    /// the columns.
    pub fn push(&mut self, env: u32, row: &StepRow<'_>) {
        let shape = self.shape;
        let ep = &mut self.open[env as usize];
        ep.ticks.push(row.tick);
        for (column, values, width) in [
            (&mut ep.qpos, row.qpos, shape.nq),
            (&mut ep.qvel, row.qvel, shape.nv),
            (&mut ep.ctrl, row.ctrl, shape.nu),
            (&mut ep.sensordata, row.sensordata, shape.nsensordata),
        ] {
            column.extend(values.iter().take(width).copied());
            column.resize(column.len() + width.saturating_sub(values.len()), 0.0);
        }
        ep.reward.push(row.reward);
        ep.done.push(row.termination.is_done());
        ep.failure.push(row.failure);
        ep.termination = row.termination;
    }

    /// Records the parameter scales a reset drew for `env`.
    pub fn set_param_scales(&mut self, env: u32, scales: ParamScales) {
        self.open[env as usize].param_scales = scales;
    }

    /// Closes `env`'s episode and opens the next one.
    pub fn finish(&mut self, env: u32) -> Episode {
        let next_id = self.open[env as usize].episode + 1;
        std::mem::replace(
            &mut self.open[env as usize],
            Episode::fresh(env, next_id, self.shape, self.max_steps),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_ir::task::Expr;

    fn shape() -> EpisodeShape {
        EpisodeShape {
            nq: 2,
            nv: 2,
            nu: 1,
            nsensordata: 3,
        }
    }

    fn row(tick: u64, reward: f64, termination: Termination) -> StepRow<'static> {
        StepRow {
            tick: PhysTick(tick),
            qpos: &[1.0, 2.0],
            qvel: &[3.0, 4.0],
            ctrl: &[0.5],
            sensordata: &[7.0, 8.0, 9.0],
            reward,
            termination,
            failure: None,
        }
    }

    #[test]
    fn columns_keep_their_shape_and_finish_rolls_the_episode_id() {
        let mut rec = EpisodeRecorder::new(2, shape(), 8);
        for t in 0..5 {
            rec.push(0, &row(t, f64::from(t as u32), Termination::Running));
        }
        rec.push(0, &row(5, 1.0, Termination::Success));
        assert_eq!(rec.open(1).steps(), 0, "envs are independent");

        let ep = rec.finish(0);
        assert_eq!((ep.env, ep.episode, ep.steps()), (0, 0, 6));
        assert_eq!(ep.qpos.len(), 6 * 2);
        assert_eq!(ep.qvel.len(), 6 * 2);
        assert_eq!(ep.ctrl.len(), 6);
        assert_eq!(ep.sensordata.len(), 6 * 3);
        assert_eq!(ep.reward.len(), 6);
        assert_eq!(ep.failure, vec![None; 6]);
        assert_eq!(ep.termination, Termination::Success);
        assert_eq!(ep.done, [false, false, false, false, false, true]);
        assert_eq!(&ep.qpos[..2], &[1.0, 2.0]);

        let next = rec.open(0);
        assert_eq!((next.episode, next.steps()), (1, 0));
        assert_eq!(next.termination, Termination::Running);
    }

    #[test]
    fn a_short_row_is_padded_rather_than_desynchronizing_the_columns() {
        let mut rec = EpisodeRecorder::new(1, shape(), 4);
        rec.push(
            0,
            &StepRow {
                sensordata: &[1.0],
                ..row(0, 0.0, Termination::Running)
            },
        );
        let ep = rec.finish(0);
        assert_eq!(ep.sensordata, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn the_episode_budget_is_the_last_predicate_consulted() {
        let plan = ScalarPlan {
            terminations: vec![(TerminationKind::Success, Expr::Port("hit".to_owned()))],
            ..ScalarPlan::default()
        };
        let ports = |v: f64| BTreeMap::from([("hit".to_owned(), v)]);
        assert_eq!(evaluate(&plan, &ports(0.0), 3, 10), Termination::Running);
        assert_eq!(evaluate(&plan, &ports(1.0), 3, 10), Termination::Success);
        // Timeout only once no predicate fires.
        assert_eq!(evaluate(&plan, &ports(0.0), 10, 10), Termination::Timeout);
        assert_eq!(evaluate(&plan, &ports(1.0), 10, 10), Termination::Success);
        // A missing port is "did not fire", not a panic and not a NaN.
        assert_eq!(
            evaluate(&plan, &BTreeMap::new(), 0, 0),
            Termination::Running
        );
    }
}
