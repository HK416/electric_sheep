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
    /// Multiplicative scale on a model parameter: recorded in the episode and pushed into the
    /// backend through `PhysicsBackend::set_params` at reset (packet M11/X4).
    Scale(Param, StableId),
}

pub use es_physics_core::backend::Param;

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

    /// Whether any entry scales a model parameter, i.e. whether a reset has to call
    /// `PhysicsBackend::set_params`. A task without one never does (spec 28.14 rule 1).
    pub fn has_scales(&self) -> bool {
        self.entries
            .iter()
            .any(|e| matches!(e.target, Target::Scale(..)))
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

    // --- visual randomization (packet M11/X5), oracle 1 --------------------------------------

    use crate::env::tests::{camera_scene, sensor_channel};

    /// Every render target of `docs/design/batch-domains.md` section 5.
    const VISUAL: [&str; 18] = [
        "light.intensity",
        "light.direction",
        "light.direction.yaw",
        "light.direction.pitch",
        "light.color",
        "light.color.kelvin",
        "light.ambient",
        "light.radiance",
        "light.sky",
        "geom.ball.rgba",
        "camera.cam.pose.x",
        "camera.cam.pose.y",
        "camera.cam.pose.z",
        "camera.cam.pose.roll",
        "camera.cam.pose.pitch",
        "camera.cam.pose.yaw",
        "camera.cam.fov",
        "geom.cube.rgba",
    ];

    fn visual_plan(nodes: &[TaskNode]) -> Result<RandomizationPlan, EnvError> {
        RandomizationPlan::compile(&task_with(nodes), &camera_scene(), &fake_model())
    }

    fn visual_draw(plan: &RandomizationPlan, seed: u64, env: u32, ep: u64) -> RenderOverrides {
        let mut out = RenderOverrides::default();
        plan.apply_render(seed, env, ep, &mut out);
        out
    }

    fn physical_draw(plan: &RandomizationPlan, env: u32, ep: u64) -> (Vec<u64>, ParamScales) {
        let (mut qpos, mut qvel, mut scales) = (vec![0.0; 2], vec![0.0; 2], ParamScales::new());
        plan.apply(
            3,
            env,
            ep,
            &mut ResetBuffer {
                qpos: &mut qpos,
                qvel: &mut qvel,
                scales: &mut scales,
            },
        );
        (qpos.iter().chain(&qvel).map(|v| v.to_bits()).collect(), scales)
    }

    #[test]
    fn visual_randomization_every_target_parses_and_resolves() {
        for target in VISUAL {
            let dist = if target.ends_with("kelvin") {
                uniform(2500.0, 9000.0)
            } else {
                uniform(0.5, 1.5)
            };
            let plan = visual_plan(&[randomization(target, dist)])
                .unwrap_or_else(|e| panic!("{target}: {e}"));
            assert!(!plan.has_scales(), "{target} is not a physics parameter");
            let ov = visual_draw(&plan, 1, 0, 0);
            assert!(!ov.is_identity(), "{target} drew nothing");
        }
        // The multi-stream targets: two angles, three channels.
        let n = |t: &str| visual_plan(&[randomization(t, uniform(0.5, 1.5))]).unwrap().entries.len();
        assert_eq!(n("light.direction"), 2);
        assert_eq!(n("light.color"), 3);
        assert_eq!(n("geom.ball.rgba"), 3);
        assert_eq!(n("camera.cam.fov"), 1);
        // Each channel from its own stream: a draw of `light.color` is not grey.
        let ov = visual_draw(
            &visual_plan(&[randomization("light.color", uniform(0.5, 1.5))]).unwrap(),
            1,
            0,
            0,
        );
        assert!(ov.color[0] != ov.color[1] && ov.color[1] != ov.color[2], "{:?}", ov.color);
    }

    #[test]
    fn visual_randomization_unknown_targets_are_named_not_skipped() {
        for target in [
            "light",
            "light.flux",
            "light.direction.roll",
            "light.color.hue",
            "geom.nope.rgba",
            "geom.ball.rgb",
            "camera.nope.fov",
            "camera.cam.pose",
            "camera.cam.pose.w",
            "camera.cam.zoom",
            "camera.cam.fov.x",
        ] {
            let err = visual_plan(&[randomization(target, uniform(0.5, 1.5))]).unwrap_err();
            assert!(err.to_string().contains(target), "{target}: {err}");
        }
        // A focal scale must stay positive: an unbounded distribution is refused.
        let err = visual_plan(&[randomization(
            "camera.cam.fov",
            Distribution::Normal {
                mean: 1.0,
                std: 0.1,
            },
        )])
        .unwrap_err();
        assert!(err.to_string().contains("positive"), "{err}");
        let err = visual_plan(&[randomization("camera.cam.fov", uniform(0.0, 1.0))]).unwrap_err();
        assert!(err.to_string().contains("positive"), "{err}");
        // The `Pt` light and sky mean nothing to the rasterizer, and a task with an `Rs` sensor
        // refuses them rather than drawing a value no frame shows (spec 17.2).
        for target in ["light.radiance", "light.sky"] {
            let mut task = task_with(&[randomization(target, uniform(0.5, 1.5))]);
            let scene = camera_scene();
            let (channel, _) =
                sensor_channel(scene.cameras[0].id, es_ir::task::SensorRender::default());
            task.observation_spec
                .channels
                .insert("rgb".to_owned(), channel);
            let err = RandomizationPlan::compile(&task, &scene, &fake_model()).unwrap_err();
            let text = err.to_string();
            assert!(text.contains(target) && text.contains("Rs"), "{text}");
        }
    }

    #[test]
    fn visual_randomization_draws_are_keyed_by_seed_env_episode_stream() {
        let nodes: Vec<TaskNode> = VISUAL
            .iter()
            .filter(|t| !t.ends_with("kelvin"))
            .map(|t| randomization(t, uniform(0.5, 1.5)))
            .collect();
        let plan = visual_plan(&nodes).unwrap();
        assert_eq!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 2, 3));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 2, 4));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 7, 1, 3));
        assert_ne!(visual_draw(&plan, 7, 2, 3), visual_draw(&plan, 8, 2, 3));
        // The stream name is part of the key.
        let renamed = |stream: &str| {
            let plan = visual_plan(&[TaskNode::Randomization {
                target: "light.intensity".to_owned(),
                dist: uniform(0.5, 1.5),
                stream: stream.to_owned(),
            }])
            .unwrap();
            visual_draw(&plan, 7, 2, 3).light.intensity
        };
        assert_eq!(renamed("a"), renamed("a"));
        assert_ne!(renamed("a"), renamed("b"));
    }

    #[test]
    fn visual_randomization_undeclared_targets_move_nothing() {
        let physical = [
            randomization("qpos[0]", uniform(-1.0, 1.0)),
            randomization("body.link.mass", uniform(0.5, 1.5)),
        ];
        let plain = visual_plan(&physical).unwrap();
        assert!(visual_draw(&plain, 3, 0, 0).is_identity());
        assert_eq!(visual_draw(&plain, 3, 0, 0), RenderOverrides::default());
        // Render entries beside the physical ones move none of the physical draws.
        let mut both = physical.to_vec();
        both.extend(
            VISUAL
                .iter()
                .filter(|t| !t.ends_with("kelvin"))
                .map(|t| randomization(t, uniform(0.5, 1.5))),
        );
        let both = visual_plan(&both).unwrap();
        for (env, ep) in [(0, 0), (1, 0), (0, 5)] {
            assert_eq!(physical_draw(&plain, env, ep), physical_draw(&both, env, ep));
        }
        assert_eq!(plain.has_scales(), both.has_scales());
    }

    #[test]
    fn visual_randomization_light_and_camera_arithmetic() {
        let base = [0.3, 0.4, 0.866_025_4];
        let norm = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        // The identity is exact.
        assert_eq!(
            RenderOverrides::default().light_dir(base).map(f64::to_bits),
            base.map(f64::to_bits)
        );
        // Yaw is the evaluation's own kernel: `LightOverride::rotate_dir`.
        let yawed = RenderOverrides {
            light: LightOverride {
                intensity: 1.0,
                yaw_deg: 30.0,
            },
            ..RenderOverrides::default()
        };
        assert_eq!(yawed.light_dir(base), yawed.light.rotate_dir(base));
        // Pitch raises the light towards +Z and keeps it a unit vector and its azimuth.
        let up = RenderOverrides {
            pitch_deg: 20.0,
            ..RenderOverrides::default()
        }
        .light_dir(base);
        assert!(up[2] > base[2], "{up:?}");
        assert!((norm(up) - norm(base)).abs() < 1e-6, "{up:?}");
        assert!((up[1] / up[0] - base[1] / base[0]).abs() < 1e-6, "{up:?}");
        // The colour-temperature table: warm is red-heavy, cool is blue-heavy, clamped at the
        // ends, piecewise linear between its rows.
        let warm = kelvin_rgb(2000.0);
        let cool = kelvin_rgb(10_000.0);
        assert!(warm[0] > warm[2] && cool[2] > cool[0], "{warm:?} {cool:?}");
        assert_eq!(kelvin_rgb(500.0), warm);
        assert_eq!(kelvin_rgb(40_000.0), cool);
        let mid = kelvin_rgb(2500.0);
        let (a, b) = (kelvin_rgb(2000.0), kelvin_rgb(3000.0));
        for c in 0..3 {
            assert!((mid[c] - (a[c] + b[c]) / 2.0).abs() < 1e-12);
        }
        // The camera delta: a translation in the camera's own frame, and a yaw about its +Y.
        let d = CameraDraw {
            offset: [0.1, 0.0, 0.0],
            ..CameraDraw::default()
        };
        assert_eq!(d.pose().position.x, 0.1);
        assert_eq!(d.pose().orientation, es_math::Quat::IDENTITY);
        let turned = CameraDraw {
            rot_deg: [0.0, 0.0, 90.0],
            ..CameraDraw::default()
        }
        .pose()
        .orientation
        .rotate(es_math::Vec3::new(0.0, 0.0, 1.0));
        assert!((turned.x - 1.0).abs() < 1e-6, "{turned:?}");
    }
}
