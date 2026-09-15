//! Evaluation IR execution (§10): perturbation kernels, metric computation, report, lock.
//!
//! `es_ir::evaluation` declares an evaluation; this crate runs one. The split is the same as
//! everywhere else in the project — the IR is a hashable contract, the runtime is what
//! executes it — and the rules that make the result trustworthy are in
//! `docs/design/evaluation-execution.md`:
//!
//! * every perturbation is drawn from `TaskRng(seed_base, suite_id, episode_idx, stream)`, so
//!   changing the policy cannot change the perturbation sequence (§10.4);
//! * a `PerturbationKind` this build cannot realise is [`EvalError::Unsupported`] naming it,
//!   never silently skipped;
//! * a metric nobody measured is `es_ir::evaluation::MetricValue::Unavailable` carrying a
//!   reason, never a fabricated `0.0`;
//! * augmentation stays off (INV-15): a `training_only` `Augment` node is auto-disabled by
//!   the plan (it lowers to an identity pass-through, §10.4), any other one outside the
//!   allow-list is [`EvalError::AugmentationEnabled`], and the graph is never rewritten to
//!   get past either.

pub mod bake;
pub mod domain_gap;
pub mod evidence;
pub mod metrics;
pub mod perturb;
pub mod runner;

pub use bake::ObservationBake;
pub use metrics::compute;
pub use perturb::{LightOverride, PerturbationPlan, ResetOverrides, StepState};
pub use runner::{
    write_artifacts, BackendCaps, Evaluation, EvaluationLock, EventSource, FrameSink, RunConfig,
    Shard, ShardCell, StepEvent,
};

/// Everything that stops an evaluation from producing a report.
///
/// There is no variant meaning "ran it anyway": a condition this runtime cannot reproduce is
/// refused, because a row of the §10.1 table that was not actually run is worse than a
/// missing row.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A `PerturbationKind` this build has no kernel for (§10.2). Never a silent skip.
    #[error("perturbation `{kind}` is not realisable here: {reason}")]
    Unsupported {
        kind: &'static str,
        reason: &'static str,
    },
    /// INV-15. The node is named; the observation graph is not modified.
    #[error(
        "INV-15: observation node {node} is an Augment node and the evaluation's augmentation \
         policy does not allow it; remove the node or add \"{node}\" to the allow-list with a \
         justification"
    )]
    AugmentationEnabled { node: String },
    /// The loaded model and the deployment's joint count disagree. Never broadcast, never
    /// zero-padded: a wrong command is worse than a refused run (§10.1).
    #[error(
        "the model has {nu} actuators, {nq} qpos and {nv} qvel entries; the deployment \
         declares {nj} joints"
    )]
    JointMismatch {
        nu: usize,
        nq: usize,
        nv: usize,
        nj: usize,
    },
    #[error("evaluation IR is not valid: {0}")]
    InvalidIr(String),
    /// A `--jobs N` partition that is not one: an out-of-range shard, or a merge whose workers
    /// do not cover every cell exactly once. Never a partial report (§10.4).
    #[error("shard: {0}")]
    Shard(String),
    #[error("observation plan: {0}")]
    Plan(String),
    #[error("policy: {0}")]
    Policy(String),
    #[error("safety plane: {0}")]
    Safety(String),
    #[error("env: {0}")]
    Env(#[from] es_env::EnvError),
    #[error("writing {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl EvalError {
    pub(crate) fn unsupported(kind: &'static str, reason: &'static str) -> Self {
        Self::Unsupported { kind, reason }
    }
}

/// Lowercase hex of a digest, for the human-readable half of `evaluation.lock`.
pub(crate) fn hex32(d: &[u8; 32]) -> String {
    use std::fmt::Write;
    d.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
