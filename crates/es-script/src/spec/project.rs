//! The sections `es project generate` reads beside the task (packet G3b): who learns it
//! (`[teacher]`, `[student]`), what bounds it when it runs (`[deploy]`), how it is judged
//! (`[evaluate]`) and the cycle that ties them (`[cycle]`).
//!
//! Every default is a value of the two committed document sets the generator reproduces —
//! plan H's Shadow Hand (`tests/fixtures/shadow-hand/`) and plan N's SO-101 views
//! (`tests/fixtures/visible-learning/*-views.toml`) — so a field is written only where a task
//! differs from them. The training recipes are presets with overrides: `training` is any part of
//! a `training.toml`, merged over the preset (`es_data::training::Recipe` refuses an unknown key
//! by name).

use es_data::training::ShowcaseRef;
use serde::{Deserialize, Serialize};

/// `[teacher]`: a state policy trained by reward (PPO) in simulation, reading what the
/// simulator knows. Absent: no teacher documents; the cycle's demonstrations then come from
/// `[cycle] expert`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Teacher {
    /// Its channels, in the order its state vector lays them out: any of `[observe] state` and
    /// `privileged`. Absent: every `state` channel, then every `privileged` one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Vec<String>>,
    /// Merged over the PPO preset (plan H's `training-teacher-v2.toml`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<toml::Table>,
}

/// `[student]`: the camera policy that imitates the demonstrations and is what deploys.
/// Absent: no student documents and no cycle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Student {
    /// The arm's name: its documents are `observation-<name>.toml`, `learning-<name>.toml`, ...
    /// Absent: `student`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Its cameras, from `[observe] cameras`; the first is the one the fusion lists first.
    /// Absent: every camera, in that order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub views: Option<Vec<String>>,
    /// Its `[observe] state` channels, in order — never a `privileged` one (a real robot does
    /// not have it). Absent: every `state` channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<Family>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<Preset>,
    /// The chunk: rows predicted per inference.
    pub horizon: u32,
    /// Rows executed before the next chunk is asked for; `control_hz / execute` is the
    /// replanning rate and must be whole (XIR-023).
    pub execute: u32,
    /// Merged over the family's preset (plan N's `training-views.toml`, `training-mad.toml`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<toml::Table>,
}

/// The student's Learning IR family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// ACT-shaped: one `ImageNet` `ResNet18` per view and the state MLP into a `Concat`, a
    /// one-frame transformer, a regression head (`learning-views.toml`).
    #[default]
    Act,
    /// MAD (arXiv 2505.04619): the views share the first view's encoder and are summed
    /// (`learning-mad.toml`); its preset trains with the single-view loss.
    Mad,
}

/// Conventions that differ between the two committed students and carry no other meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// Plan H's student (M16/H3): a `tanh` head unnormalized by each actuator's ctrlrange
    /// (H6's invariant: the head cannot leave the envelope), the state channels concatenated
    /// and standardized from the scene's ranges as one `state` input, each view fused on a port
    /// named by its camera.
    #[default]
    H3,
    /// The SO-101 demos' (plan U's U3): an unbounded head whose rows are the joint targets,
    /// one state channel passed through as `[-1, 1]` under its own name, the first view fused
    /// on port `image`.
    U3,
}

/// `[deploy]`: the Deployment IR's envelope is the scene's (each actuator's ctrlrange and
/// forcerange, rate limits from the ctrlrange width at the control rate, plan H's rule); what
/// the scene cannot say is here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deploy {
    /// The robot's name in the Deployment IR. Absent: `robot`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The workspace box, world frame (spec 9.3; a joint-space policy is not projected onto
    /// it). Absent: the robot's root body ± 1 m.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<BoxDoc>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxDoc {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// `[evaluate]`: held-out seeds and the acceptance. The teacher is judged on the nominal suite;
/// the student on plan U's six (nominal, light intensity and direction, observation delay,
/// torque noise, backlash) and, in its `-nominal` sibling, on the nominal one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluate {
    /// The seeds are `first_seed ..` — outside every seed the cycle collects on. Absent: 101.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_seed: Option<u64>,
    /// Absent: 16.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episodes: Option<u32>,
    /// Accepted when the nominal success rate is at least this. Absent: 0.5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success_rate: Option<f64>,
}

/// `[cycle]`: `es loop cycle`'s recipe — demonstrate, (keep the successes,) train the student,
/// evaluate it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CycleDoc {
    /// Where the bundles and collections live. Absent: `runs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runs: Option<String>,
    /// A scripted expert demonstrates (`es loop collect --expert`). Absent: the trained
    /// teacher, `<runs>/teacher.esb`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert: Option<String>,
    pub episodes: u32,
    pub seed: u64,
    /// Train on the successful episodes only (`es loop distill --success-only`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success_only: Option<bool>,
    /// Evaluate on `evaluation-<name>-nominal.toml` (the nominal suite) instead of every suite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nominal_only: Option<bool>,
    /// Parallel evaluation workers. Absent: 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jobs: Option<u32>,
    /// A short test of each checkpoint while training runs. Absent: on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub showcase: Option<ShowcaseRef>,
}
