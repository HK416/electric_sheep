//! Failure semantics (§18.5, P16).
//!
//! A quarantined env drops out of reductions; because the accumulator is order independent
//! (§18.4) masking it does not break determinism. A Safety Plane fallback is *normal
//! operation*, not a failure: the env stays [`EnvHealth::Ok`] and the event is recorded.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Health of a single environment (§18.5).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EnvHealth {
    #[default]
    Ok,
    /// State blew up (non-finite or out of bounds) but the env is still being stepped.
    Diverged,
    /// State is contaminated by an earlier divergence; results are not trustworthy.
    Poisoned,
    /// Excluded from reductions and from the dataset until reset.
    Quarantined,
}

impl EnvHealth {
    /// Whether this env contributes to reductions. Only `Quarantined` does not (§18.5).
    pub fn participates_in_reduction(self) -> bool {
        self != Self::Quarantined
    }
}

/// What went wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// A non-finite value appeared in state, observation or reward.
    NanDetected,
    /// State left the physically plausible envelope (§18.5).
    Diverged,
    /// A step, inference or actuator write missed its deadline (§9).
    DeadlineMiss,
    /// A sensor sample never arrived for its scheduled tick (§18.3).
    SensorDrop,
    /// An actuator reported a fault or stopped acknowledging (§18.2).
    ActuatorFault,
    /// The action chunk buffer ran dry before the next inference landed (§8.6, §12.4).
    ChunkUnderrun,
    /// The Safety Plane clamped or rejected an action (§9). Not an env failure.
    SafetyViolation,
    /// The selected backend does not implement a requested feature (§17.2).
    BackendUnsupported,
}

/// What to do about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureAction {
    /// Record the event and keep going.
    Record,
    /// Record and hand control to the configured fallback (§9). The env stays healthy.
    Fallback,
    /// Take the env out of reductions until it resets (§18.5).
    Quarantine,
    /// End this episode, keep the run going.
    AbortEpisode,
    /// Stop the run.
    AbortRun,
}

impl FailureKind {
    /// Every kind, for exhaustive iteration in tests and reports.
    pub const ALL: [Self; 8] = [
        Self::NanDetected,
        Self::Diverged,
        Self::DeadlineMiss,
        Self::SensorDrop,
        Self::ActuatorFault,
        Self::ChunkUnderrun,
        Self::SafetyViolation,
        Self::BackendUnsupported,
    ];

    /// The one action taken for this kind when the policy does not override it.
    ///
    /// The `match` is exhaustive, so adding a kind without deciding its action does not
    /// compile.
    pub fn default_action(self) -> FailureAction {
        match self {
            Self::NanDetected | Self::Diverged => FailureAction::Quarantine,
            Self::DeadlineMiss | Self::SensorDrop | Self::ChunkUnderrun => FailureAction::Record,
            // A fallback is normal operation, so the env stays Ok (§18.5).
            Self::SafetyViolation => FailureAction::Fallback,
            Self::ActuatorFault => FailureAction::AbortEpisode,
            Self::BackendUnsupported => FailureAction::AbortRun,
        }
    }
}

/// Per-run failure handling: the defaults, plus any overrides, plus the quarantine budget.
///
/// `FailureAction` has no variant that disables the Safety Plane; no policy can produce one
/// (INV-12).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FailurePolicy {
    /// Overrides for individual kinds. `BTreeMap`, so iteration is deterministic (§3.4).
    pub overrides: BTreeMap<FailureKind, FailureAction>,
    /// Stop the run once this fraction of envs is quarantined (§18.5 default 5%).
    pub quarantine_rate_limit: f64,
}

impl Default for FailurePolicy {
    fn default() -> Self {
        Self {
            overrides: BTreeMap::new(),
            quarantine_rate_limit: 0.05,
        }
    }
}

impl FailurePolicy {
    pub fn action_for(&self, kind: FailureKind) -> FailureAction {
        self.overrides
            .get(&kind)
            .copied()
            .unwrap_or_else(|| kind.default_action())
    }

    /// Whether the quarantine rate has exceeded the budget and the run must stop.
    pub fn quarantine_budget_exceeded(&self, quarantined: usize, total: usize) -> bool {
        total > 0 && (quarantined as f64 / total as f64) > self.quarantine_rate_limit
    }
}

/// The `es-core` error type.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("tick rate {num}/{den} is invalid: numerator and denominator must be non-zero")]
    InvalidTickRate { num: u64, den: u64 },
    #[error("`{0}` is not a 32-character hex stable id")]
    InvalidStableId(String),
    #[error("{kind:?}: {detail}")]
    Failure { kind: FailureKind, detail: String },
}

impl Error {
    pub fn failure(kind: FailureKind, detail: impl Into<String>) -> Self {
        Self::Failure {
            kind,
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kind_maps_to_exactly_one_default_action() {
        // `default_action` is a total function of a `Copy` enum, so "exactly one" reduces to
        // "defined and stable" for every kind.
        for kind in FailureKind::ALL {
            let action = kind.default_action();
            assert_eq!(action, kind.default_action());
            assert_eq!(FailurePolicy::default().action_for(kind), action);
        }
        assert_eq!(FailureKind::ALL.len(), 8);
    }

    #[test]
    fn safety_violation_falls_back_and_keeps_the_env_healthy() {
        assert_eq!(
            FailureKind::SafetyViolation.default_action(),
            FailureAction::Fallback
        );
        assert!(EnvHealth::Ok.participates_in_reduction());
        assert!(EnvHealth::Diverged.participates_in_reduction());
        assert!(!EnvHealth::Quarantined.participates_in_reduction());
    }

    #[test]
    fn overrides_win() {
        let mut policy = FailurePolicy::default();
        policy
            .overrides
            .insert(FailureKind::SensorDrop, FailureAction::AbortEpisode);
        assert_eq!(
            policy.action_for(FailureKind::SensorDrop),
            FailureAction::AbortEpisode
        );
        assert_eq!(
            policy.action_for(FailureKind::Diverged),
            FailureAction::Quarantine
        );
    }

    #[test]
    fn quarantine_budget() {
        let policy = FailurePolicy::default();
        assert!(!policy.quarantine_budget_exceeded(0, 0));
        assert!(!policy.quarantine_budget_exceeded(5, 100));
        assert!(policy.quarantine_budget_exceeded(6, 100));
    }

    #[test]
    fn serde_round_trip() {
        for kind in FailureKind::ALL {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(serde_json::from_str::<FailureKind>(&json).unwrap(), kind);
            let action = kind.default_action();
            let json = serde_json::to_string(&action).unwrap();
            assert_eq!(
                serde_json::from_str::<FailureAction>(&json).unwrap(),
                action
            );
        }
        for health in [
            EnvHealth::Ok,
            EnvHealth::Diverged,
            EnvHealth::Poisoned,
            EnvHealth::Quarantined,
        ] {
            let json = serde_json::to_string(&health).unwrap();
            assert_eq!(serde_json::from_str::<EnvHealth>(&json).unwrap(), health);
        }
        let mut policy = FailurePolicy::default();
        policy
            .overrides
            .insert(FailureKind::NanDetected, FailureAction::AbortRun);
        let json = serde_json::to_string(&policy).unwrap();
        assert_eq!(
            serde_json::from_str::<FailurePolicy>(&json).unwrap(),
            policy
        );
    }

    #[test]
    fn error_display() {
        let err = Error::failure(FailureKind::DeadlineMiss, "control loop 12 ms over");
        assert!(err.to_string().contains("DeadlineMiss"));
    }
}
