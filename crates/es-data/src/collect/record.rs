//! What a collection records per frame (design note section 3): the `action_source` a step
//! earned from the plane's counters, the dataset's feature set, and an episode as `LeRobot`
//! columns.

use std::collections::BTreeMap;

use es_compile::PolicyBundle;
use es_safety::{EventSet, SafetyPlane, ViolationKind};

use crate::intervention::{ActionSourceCode, ACTION_SOURCE, INTERVENTION};
use crate::lerobot::meta::{Dtype, FeatureSpec};
use crate::lerobot::{Column, Episode};

/// The raw pre-plane command, beside `action` (spec 13.2).
pub const ACTION_COMMANDED: &str = "action_commanded";

/// The `EventSet` one step raised, as the delta of the plane's own per-kind counters.
///
/// One env means exactly one `validate` per step, so a kind whose count moved is a kind that
/// step raised — the same bitset `SafeAction::events` carries, read from the side the
/// collector can see (`es_env::DomainRunner::emit_actions` keeps the `SafeAction` itself).
pub(super) fn raised(
    before: &[u64; ViolationKind::COUNT],
    after: &[u64; ViolationKind::COUNT],
) -> EventSet {
    let mut set = EventSet::EMPTY;
    for kind in ViolationKind::ALL {
        if after[kind.index()] > before[kind.index()] {
            set.insert(kind);
        }
    }
    set
}

/// `(fallback_activations, clamped_steps)` — the two counters that say what the plane did.
pub(super) fn counters_of<const NJ: usize, const H: usize>(
    plane: &SafetyPlane<NJ, H>,
) -> (u64, u64) {
    let c = plane.counters();
    (c.fallback_activations, c.clamped_steps)
}

/// One frame's `action_source`, from the plane's counter deltas across the step.
///
/// One env per collect run, so exactly one `validate` happens per step and the delta is
/// unambiguous. A clamped human action reports `Clamped`, not `Human`: the frame is still an
/// intervention (the `intervention` column says so) but the actuator did not see what the
/// human asked for, and hiding that would discard the evidence spec 9.3 exists to keep.
pub(super) fn classify(before: (u64, u64), after: (u64, u64), human: bool) -> ActionSourceCode {
    if after.0 > before.0 {
        ActionSourceCode::Fallback
    } else if after.1 > before.1 {
        ActionSourceCode::Clamped
    } else if human {
        ActionSourceCode::Human
    } else {
        ActionSourceCode::Policy
    }
}

/// The feature set of design note section 3, plus one `video` feature per image channel the
/// task declares — and, with no frame sink, one warning for each, because then nothing at all
/// was written for that channel.
pub(super) fn collect_features(
    bundle: &PolicyBundle,
    state: usize,
    nu: usize,
    rendering: bool,
) -> (BTreeMap<String, FeatureSpec>, Vec<String>) {
    let mut f = BTreeMap::new();
    f.insert(
        "observation.state".to_owned(),
        FeatureSpec::new(Dtype::Float32, [state as u64]),
    );
    f.insert(
        "action".to_owned(),
        FeatureSpec::new(Dtype::Float32, [nu as u64]),
    );
    f.insert(
        ACTION_COMMANDED.to_owned(),
        FeatureSpec::new(Dtype::Float32, [nu as u64]),
    );
    f.insert("reward".to_owned(), FeatureSpec::new(Dtype::Float64, [1]));
    f.insert(INTERVENTION.to_owned(), FeatureSpec::new(Dtype::Int64, [1]));
    f.insert(
        ACTION_SOURCE.to_owned(),
        FeatureSpec::new(Dtype::Int64, [1]),
    );

    let mut warnings = Vec::new();
    for (name, channel) in &bundle.task.observation_spec.channels {
        if channel.ty.image.is_none() {
            continue;
        }
        f.insert(
            format!("observation.images.{name}"),
            FeatureSpec::new(Dtype::Video, channel.ty.shape.dims().to_vec()),
        );
        if !rendering {
            warnings.push(format!(
                "camera {name:?}: info.json declares a video feature and read_episode will \
                 produce VideoRef placeholders, but no mp4 was written (no renderer in this \
                 build)"
            ));
        }
    }
    (f, warnings)
}

/// The per-frame columns [`Collector::run`] accumulates itself, beside the ones the episode
/// recorder already holds.
pub(super) struct Recorded<'a> {
    pub(super) sources: &'a [i64],
    pub(super) human: &'a [bool],
    /// The pre-plane command of each frame, `n * nu` (spec 13.2).
    pub(super) commanded: &'a [f64],
    /// The plane's answer of each frame, `n * nu`.
    pub(super) answered: &'a [f64],
}

/// One `es_env::Episode` as `LeRobot` columns (design note section 3).
pub(super) fn to_lerobot(
    index: u32,
    ep: &es_env::Episode,
    shape: (usize, usize, usize),
    frames: &Recorded<'_>,
    fps: f64,
    task: &str,
) -> Episode {
    let (sources, human, commanded) = (frames.sources, frames.human, frames.commanded);
    let (nq, nv, nu) = shape;
    let n = ep.steps();
    let mut state = Vec::with_capacity(n * (nq + nv));
    for i in 0..n {
        state.extend(ep.qpos[i * nq..(i + 1) * nq].iter().map(|v| *v as f32));
        state.extend(ep.qvel[i * nv..(i + 1) * nv].iter().map(|v| *v as f32));
    }
    let columns = BTreeMap::from([
        ("observation.state".to_owned(), Column::F32(state)),
        (
            // The `SafeAction` the plane returned for that step, as `DomainRunner::emit_actions`
            // wrote it: what the plane let through is what is recorded (spec 13.2, `INV-12`).
            // It is `ep.ctrl` bit for bit unless a perturbation moved the command on its way to
            // the actuator (packet M13/Z2), and then the plant's error is not the label.
            "action".to_owned(),
            Column::F32(
                frames.answered[..n * nu]
                    .iter()
                    .map(|v| *v as f32)
                    .collect(),
            ),
        ),
        (
            ACTION_COMMANDED.to_owned(),
            Column::F32(commanded[..n * nu].iter().map(|v| *v as f32).collect()),
        ),
        ("reward".to_owned(), Column::F64(ep.reward[..n].to_vec())),
        (
            INTERVENTION.to_owned(),
            Column::I64(human.iter().map(|h| i64::from(*h)).collect()),
        ),
        (ACTION_SOURCE.to_owned(), Column::I64(sources.to_vec())),
    ]);
    Episode {
        index,
        tasks: vec![task.to_owned()],
        timestamps: (0..n).map(|i| i as f64 / fps).collect(),
        task_index: vec![0; n],
        columns,
        // Regenerated by the reader from `video_path`; no mp4 is written here.
        video: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clamp_is_not_an_intervention_and_an_intervention_is_not_a_failure() {
        // Fallback wins over everything: the actuator saw the watchdog's action.
        assert_eq!(classify((0, 0), (1, 1), true), ActionSourceCode::Fallback);
        // A human action the envelope changed is `Clamped`, not `Human` (design note 2.2).
        assert_eq!(classify((0, 0), (0, 1), true), ActionSourceCode::Clamped);
        assert_eq!(classify((0, 0), (0, 0), true), ActionSourceCode::Human);
        assert_eq!(classify((0, 0), (0, 0), false), ActionSourceCode::Policy);
        // A clamp with nobody intervening is still just a clamp (spec 18.5).
        assert_eq!(classify((3, 7), (3, 8), false), ActionSourceCode::Clamped);
    }
}
