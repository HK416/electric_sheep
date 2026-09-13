//! Metric computation (§10.3) and the §12.4 performance set.
//!
//! The rule that shapes this module: **nothing is invented**. A metric this runtime does not
//! measure is [`MetricValue::Unavailable`] carrying the reason, never `0.0`. A zero in the
//! §10.1 table reads as "the policy never collided"; the truth is that nothing counted
//! collisions, and the two must not be confusable.

use std::collections::BTreeMap;

use es_core::FailureKind;
use es_env::{EnvMetrics, Episode, Termination};
use es_ir::evaluation::{Aggregation, MetricSpec, MetricValue};
use es_safety::{SafetyCounters, ViolationKind};

fn unavailable(reason: &str) -> MetricValue {
    MetricValue::Unavailable {
        reason: reason.to_owned(),
    }
}

/// Computes one metric over the finished episodes of one cell.
///
/// `episodes` must already be in the cell's deterministic order — `(cell, seed)`, which the
/// runner preserves by appending in episode order (§10.4).
pub fn compute(
    spec: &MetricSpec,
    episodes: &[Episode],
    counters: &SafetyCounters,
    env: &EnvMetrics,
) -> MetricValue {
    match spec {
        MetricSpec::SuccessRate => per_episode(spec, episodes).map_or_else(
            || unavailable("no episode finished in this cell"),
            |v| MetricValue::Scalar(mean(&v)),
        ),
        MetricSpec::EpisodeLength | MetricSpec::ActionSmoothness => per_episode(spec, episodes)
            .map_or_else(
                || unavailable("no episode finished in this cell"),
                |v| MetricValue::Scalar(mean(&v)),
            ),
        // §9.3 / §10.3: the fraction of steps the plane clamped, projected or fell back on,
        // over the whole cell. `SafetyCounters::envelope_violation_rate` is the *sliding*
        // fraction the §9.4 rate watchdog reads, which is a different question.
        MetricSpec::EnvelopeViolationRate => {
            if counters.steps == 0 {
                return unavailable("the Safety Plane validated no step");
            }
            MetricValue::Scalar(counters.dirty_steps as f64 / counters.steps as f64)
        }
        MetricSpec::ChunkUnderrunRate => {
            if counters.steps == 0 {
                unavailable("the Safety Plane validated no step")
            } else {
                MetricValue::Scalar(counters.chunk_underrun_rate())
            }
        }
        MetricSpec::FailureModeHistogram => {
            MetricValue::Histogram(failure_histogram(episodes, counters))
        }
        MetricSpec::InterventionRate => {
            unavailable("human intervention is a hardware/HIL signal (§24.2)")
        }
        MetricSpec::CollisionRate => {
            unavailable("PhysicsBackend reports no contacts in this build")
        }
        MetricSpec::DomainGap => unavailable("requires real-log replay (§24.3), not in this build"),
        // §12.4: pass-through only. `EnvMetrics` is `Option` per field precisely so an
        // unmeasured throughput stays unmeasured all the way into the report.
        MetricSpec::PhysicsStepsPerSec => opt(env.physics_steps_per_sec),
        MetricSpec::CameraFramesPerSec => opt(env.camera_frames_per_sec),
        MetricSpec::PixelsPerSec => opt(env.pixels_per_sec),
        MetricSpec::ObservationGbPerSec => opt(env.observation_gb_per_sec),
        MetricSpec::PolicyInferencesPerSec => opt(env.policy_inferences_per_sec),
        MetricSpec::ActionsPerSec => opt(env.actions_per_sec),
        MetricSpec::EndToEndLatencyP50 => opt(env.p50_end_to_end_latency),
        MetricSpec::EndToEndLatencyP95 => opt(env.p95_end_to_end_latency),
        MetricSpec::GpuMemoryPeak => opt(env.gpu_memory_peak),
    }
}

fn opt(v: Option<f64>) -> MetricValue {
    v.map_or_else(
        || unavailable("not instrumented in this build (§12.4)"),
        MetricValue::Scalar,
    )
}

/// The per-episode sample of a metric that has one, for [`Aggregation`] other than `Mean`.
///
/// Returns `None` for the metrics that only exist at cell level (the counter rates and the
/// §12.4 pass-throughs): for those, every aggregation is the same single number.
pub fn per_episode(spec: &MetricSpec, episodes: &[Episode]) -> Option<Vec<f64>> {
    if episodes.is_empty() {
        return None;
    }
    match spec {
        MetricSpec::SuccessRate => Some(
            episodes
                .iter()
                .map(|e| f64::from(u8::from(e.termination == Termination::Success)))
                .collect(),
        ),
        MetricSpec::EpisodeLength => Some(episodes.iter().map(|e| e.steps() as f64).collect()),
        MetricSpec::ActionSmoothness => Some(episodes.iter().map(smoothness).collect()),
        _ => None,
    }
}

/// §10.3: the normalized inverse of the action's first and second differences. One, when the
/// command never moves; toward zero as the trace gets rougher.
fn smoothness(ep: &Episode) -> f64 {
    let nu = ep.shape.nu;
    let steps = ep.steps();
    if nu == 0 || steps < 3 {
        return 1.0;
    }
    let row = |i: usize| &ep.ctrl[i * nu..(i + 1) * nu];
    let (mut d1, mut d2) = (0.0f64, 0.0f64);
    for i in 1..steps {
        for j in 0..nu {
            d1 += (row(i)[j] - row(i - 1)[j]).abs();
        }
    }
    for i in 2..steps {
        for j in 0..nu {
            d2 += (row(i)[j] - 2.0 * row(i - 1)[j] + row(i - 2)[j]).abs();
        }
    }
    let n1 = ((steps - 1) * nu) as f64;
    let n2 = ((steps - 2) * nu) as f64;
    1.0 / (1.0 + d1 / n1 + d2 / n2)
}

/// §10.3 `failure_mode_histogram`: why episodes ended, plus what the Safety Plane saw.
fn failure_histogram(episodes: &[Episode], counters: &SafetyCounters) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    let mut bump = |k: &str, n: u64| {
        if n > 0 {
            *out.entry(k.to_owned()).or_insert(0) += n;
        }
    };
    for ep in episodes {
        match ep.termination {
            Termination::Success => bump("success", 1),
            Termination::Failure => bump("failure", 1),
            Termination::Timeout => bump("timeout", 1),
            Termination::Running => bump("unfinished", 1),
        }
        for f in ep.failure.iter().flatten() {
            bump(failure_name(*f), 1);
        }
    }
    // A fallback is normal operation, not an env failure (§18.5), but it is a failure *mode*
    // of the policy and §10.3 lists it as one of the buckets.
    bump("fallback", counters.fallback_activations);
    for kind in ViolationKind::ALL {
        bump(violation_name(kind), counters.count(kind));
    }
    out
}

fn failure_name(k: FailureKind) -> &'static str {
    match k {
        FailureKind::NanDetected => "nan_detected",
        FailureKind::Diverged => "diverged",
        FailureKind::DeadlineMiss => "deadline_miss",
        FailureKind::SensorDrop => "sensor_drop",
        FailureKind::ActuatorFault => "actuator_fault",
        FailureKind::ChunkUnderrun => "chunk_underrun",
        FailureKind::SafetyViolation => "safety_violation",
        FailureKind::BackendUnsupported => "backend_unsupported",
    }
}

fn violation_name(k: ViolationKind) -> &'static str {
    match k {
        ViolationKind::NonFinite => "violation.non_finite",
        ViolationKind::Position => "violation.position",
        ViolationKind::Velocity => "violation.velocity",
        ViolationKind::Acceleration => "violation.acceleration",
        ViolationKind::Torque => "violation.torque",
        ViolationKind::Workspace => "violation.workspace",
        ViolationKind::RateLimit => "violation.rate_limit",
        ViolationKind::StaleObservation => "violation.stale_observation",
        ViolationKind::InferenceDeadline => "violation.inference_deadline",
        ViolationKind::ChunkUnderrun => "violation.chunk_underrun",
        ViolationKind::HeartbeatLoss => "violation.heartbeat_loss",
        ViolationKind::SensorDropout => "violation.sensor_dropout",
        ViolationKind::ViolationRate => "violation.rate",
        ViolationKind::EstopLatched => "violation.estop_latched",
    }
}

// --- Reduction --------------------------------------------------------------------------

/// Reduces a per-episode sample the way an acceptance criterion asks (§10.2).
///
/// `P95` is the nearest-rank order statistic: no
/// interpolation, so it is exactly reproducible.
pub fn aggregate(values: &[f64], agg: Aggregation) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(match agg {
        Aggregation::Mean => mean(values),
        Aggregation::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        Aggregation::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Aggregation::P95 => {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            let rank = ((sorted.len() as f64) * 0.95).ceil() as usize;
            sorted[rank.clamp(1, sorted.len()) - 1]
        }
    })
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (n-1). `0.0` for fewer than two samples.
pub fn std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let var = values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}

/// Normal-approximation 95% confidence interval of the mean.
///
/// Deliberately not a Welch test: comparing two policies is `es eval compare` (§10.5), which
/// lives in the CLI, above this crate. This is the interval a single report prints next to a
/// cell.
pub fn ci95(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values);
    let half = 1.96 * std(values) / (values.len() as f64).sqrt();
    Some((m - half, m + half))
}
