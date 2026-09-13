//! Domain randomization and reset-state distributions (§6.3 `Randomization` / `ResetState`).
//!
//! Target strings are resolved against the scene and the loaded model **once**, at compile
//! time, so the per-reset path has no string work and cannot fail. A target this runtime does
//! not implement is [`EnvError::Unsupported`] naming it — never silently skipped
//! (`docs/design/batch-domains.md` §5).

use std::collections::BTreeMap;

use es_assets::scene::SceneDesc;
use es_core::StableId;
use es_ir::task::{Distribution, TaskIr, TaskNode};
use es_physics_core::backend::ModelInfo;

use crate::rng::EnvRng;
use crate::EnvError;

/// What one draw writes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// Index into one env's `qpos` row.
    Qpos(u32),
    /// Index into one env's `qvel` row.
    Qvel(u32),
    /// Multiplicative scale on a model parameter. Recorded, not yet pushed into the backend:
    /// `PhysicsBackend` has no parameter API (see the ceiling in the design note).
    Scale(Param, StableId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Param {
    BodyMass,
    GeomFriction,
    ActuatorGain,
}

/// One resolved node: a target, a distribution and the RNG stream it draws from.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    target: Target,
    dist: Distribution,
    stream: StableId,
}

/// The per-reset scale factors a plan drew, for the episode record.
pub type ParamScales = BTreeMap<(Param, StableId), f64>;

/// Where a reset writes: one env's state row plus the parameter scales it drew.
#[derive(Debug)]
pub struct ResetBuffer<'a> {
    pub qpos: &'a mut [f64],
    pub qvel: &'a mut [f64],
    pub scales: &'a mut ParamScales,
}

/// Every `ResetState` and `Randomization` node of a task, resolved and ordered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RandomizationPlan {
    entries: Vec<Entry>,
}

impl RandomizationPlan {
    /// Resolves every randomization node against `scene` and `model`.
    ///
    /// Nodes are taken in ascending `NodeId` (§6.4), which fixes the draw order for a given
    /// graph; `ResetState` nodes run before `Randomization` nodes so a randomizer can perturb
    /// the state a reset distribution just laid down.
    pub fn compile(task: &TaskIr, scene: &SceneDesc, model: &ModelInfo) -> Result<Self, EnvError> {
        let mut reset = Vec::new();
        let mut random = Vec::new();
        for node in task.graph.nodes.values() {
            let (target, dist, stream, into) = match node {
                TaskNode::ResetState {
                    target,
                    dist,
                    stream,
                } => (target, dist, stream, &mut reset),
                TaskNode::Randomization {
                    target,
                    dist,
                    stream,
                } => (target, dist, stream, &mut random),
                _ => continue,
            };
            check_distribution(dist, target)?;
            into.push(Entry {
                target: resolve(target, scene, model)?,
                dist: dist.clone(),
                stream: StableId::from_path(stream),
            });
        }
        reset.append(&mut random);
        Ok(Self { entries: reset })
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Draws every entry for one env and writes it into `buf`.
    ///
    /// Each entry gets its own stream keyed by `(seed, env, episode, stream_id)`, so the result
    /// depends on neither the order envs are reset in nor how many envs there are.
    pub fn apply(&self, seed: u64, env: u32, episode: u64, buf: &mut ResetBuffer<'_>) {
        for entry in &self.entries {
            let mut rng = EnvRng::new(seed, env, episode, entry.stream);
            let v = rng.sample(&entry.dist);
            match entry.target {
                Target::Qpos(i) => set(buf.qpos, i, v),
                Target::Qvel(i) => set(buf.qvel, i, v),
                Target::Scale(param, id) => {
                    buf.scales.insert((param, id), v);
                }
            }
        }
    }
}

fn set(row: &mut [f64], index: u32, value: f64) {
    // The index was bounds-checked against `ModelInfo` at compile time; a short row means the
    // caller passed a buffer that does not match the model, which `Env` prevents.
    if let Some(slot) = row.get_mut(index as usize) {
        *slot = value;
    }
}

fn check_distribution(dist: &Distribution, target: &str) -> Result<(), EnvError> {
    match dist {
        Distribution::LogUniform { lo, hi } if *lo <= 0.0 || *hi <= 0.0 => {
            Err(EnvError::Unsupported(format!(
                "LogUniform on \"{target}\" with a non-positive bound ({lo}, {hi})"
            )))
        }
        Distribution::Choice(vs) if vs.is_empty() => Err(EnvError::Unsupported(format!(
            "empty Choice on \"{target}\""
        ))),
        _ => Ok(()),
    }
}

/// The target grammar. Anything else is `Unsupported`, by name.
fn resolve(target: &str, scene: &SceneDesc, model: &ModelInfo) -> Result<Target, EnvError> {
    let unsupported = || EnvError::Unsupported(format!("randomization target \"{target}\""));

    if let Some(i) = index_of(target, "qpos") {
        return in_range(i, model.nq)
            .map(Target::Qpos)
            .ok_or_else(unsupported);
    }
    if let Some(i) = index_of(target, "qvel") {
        return in_range(i, model.nv)
            .map(Target::Qvel)
            .ok_or_else(unsupported);
    }

    let parts: Vec<&str> = target.split('.').collect();
    let [kind, name, field] = parts[..] else {
        return Err(unsupported());
    };
    match (kind, field) {
        ("joint", "qpos" | "qvel") => {
            let joint = scene
                .joints
                .iter()
                .find(|j| j.name == name)
                .ok_or_else(unsupported)?;
            let map = if field == "qpos" {
                &model.qpos
            } else {
                &model.dof
            };
            let start = map.get(&joint.id).ok_or_else(unsupported)?.start;
            Ok(if field == "qpos" {
                Target::Qpos(start)
            } else {
                Target::Qvel(start)
            })
        }
        ("body", "mass") => scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .map(|b| Target::Scale(Param::BodyMass, b.id))
            .ok_or_else(unsupported),
        ("geom", "friction") => scene
            .bodies
            .iter()
            .flat_map(|b| &b.geoms)
            .find(|g| g.name == name)
            .map(|g| Target::Scale(Param::GeomFriction, g.id))
            .ok_or_else(unsupported),
        ("actuator", "gain") => scene
            .actuators
            .iter()
            .find(|a| a.name == name)
            .map(|a| Target::Scale(Param::ActuatorGain, a.id))
            .ok_or_else(unsupported),
        _ => Err(unsupported()),
    }
}

/// `"qpos[3]"` -> `Some(3)`.
fn index_of(target: &str, prefix: &str) -> Option<u32> {
    target
        .strip_prefix(prefix)?
        .strip_prefix('[')?
        .strip_suffix(']')?
        .parse()
        .ok()
}

fn in_range(i: u32, n: u32) -> Option<u32> {
    (i < n).then_some(i)
}

// Bitwise reproducibility is the property under test: these comparisons are deliberate.
#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::env::tests::{fake_model, fake_scene, task_with};
    use es_ir::graph::NodeId;

    fn plan_for(nodes: &[TaskNode]) -> Result<RandomizationPlan, EnvError> {
        RandomizationPlan::compile(&task_with(nodes), &fake_scene(), &fake_model())
    }

    fn randomization(target: &str, dist: Distribution) -> TaskNode {
        TaskNode::Randomization {
            target: target.to_owned(),
            dist,
            stream: format!("s.{target}"),
        }
    }

    fn uniform(lo: f64, hi: f64) -> Distribution {
        Distribution::Uniform { lo, hi }
    }

    #[test]
    fn every_supported_target_resolves() {
        let plan = plan_for(&[
            randomization("qpos[0]", uniform(-0.1, 0.1)),
            randomization("qvel[1]", uniform(-1.0, 1.0)),
            randomization("joint.hinge.qpos", uniform(-0.2, 0.2)),
            randomization("joint.slide.qvel", uniform(-2.0, 2.0)),
            randomization("body.link.mass", uniform(0.9, 1.1)),
            randomization("geom.ball.friction", uniform(0.5, 1.5)),
            randomization("actuator.motor.gain", uniform(0.8, 1.2)),
        ])
        .unwrap();
        assert_eq!(plan.entries.len(), 7);
        assert_eq!(plan.entries[0].target, Target::Qpos(0));
        assert_eq!(plan.entries[1].target, Target::Qvel(1));
        assert!(matches!(
            plan.entries[4].target,
            Target::Scale(Param::BodyMass, _)
        ));
    }

    #[test]
    fn an_unknown_target_is_named_not_skipped() {
        for target in [
            "gravity",
            "qpos[99]",
            "body.nonexistent.mass",
            "body.link.colour",
            "joint.hinge.torque",
        ] {
            let err = plan_for(&[randomization(target, uniform(0.0, 1.0))]).unwrap_err();
            assert!(err.to_string().contains(target), "{target}: {err}");
        }
        let err = plan_for(&[randomization(
            "qpos[0]",
            Distribution::LogUniform { lo: -1.0, hi: 2.0 },
        )])
        .unwrap_err();
        assert!(err.to_string().contains("non-positive"), "{err}");
        let err =
            plan_for(&[randomization("qpos[0]", Distribution::Choice(Vec::new()))]).unwrap_err();
        assert!(err.to_string().contains("empty Choice"), "{err}");
    }

    #[test]
    fn reset_state_runs_before_randomization_whatever_the_node_ids() {
        let task = {
            let mut t = task_with(&[]);
            t.graph.insert(
                NodeId(0),
                randomization("qpos[0]", Distribution::Constant(5.0)),
            );
            t.graph.insert(
                NodeId(1),
                TaskNode::ResetState {
                    target: "qpos[0]".to_owned(),
                    dist: Distribution::Constant(1.0),
                    stream: "reset".to_owned(),
                },
            );
            t
        };
        let plan = RandomizationPlan::compile(&task, &fake_scene(), &fake_model()).unwrap();
        let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
        plan.apply(
            0,
            0,
            0,
            &mut ResetBuffer {
                qpos: &mut qpos,
                qvel: &mut qvel,
                scales: &mut scales,
            },
        );
        assert_eq!(qpos[0], 5.0, "the Randomization node writes last");
    }

    #[test]
    fn draws_depend_on_env_and_episode_but_not_on_reset_order() {
        let plan = plan_for(&[
            randomization("qpos[0]", uniform(-1.0, 1.0)),
            randomization("body.link.mass", uniform(0.5, 1.5)),
        ])
        .unwrap();
        let draw = |env: u32, episode: u64| {
            let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
            plan.apply(
                7,
                env,
                episode,
                &mut ResetBuffer {
                    qpos: &mut qpos,
                    qvel: &mut qvel,
                    scales: &mut scales,
                },
            );
            (qpos[0], *scales.values().next().unwrap())
        };
        assert_eq!(draw(0, 0), draw(0, 0));
        assert_ne!(draw(0, 0), draw(1, 0));
        assert_ne!(draw(0, 0), draw(0, 1));
        let (_, mass) = draw(3, 2);
        assert!((0.5..=1.5).contains(&mass), "{mass}");
    }
}
