//! Lowering the reward and termination cones of a Task IR into [`Expr`] (§6.4, §6.5).
//!
//! `Reward` and `Terminate` are graph sinks with an input edge, not expression literals, so the
//! scalar sub-graph feeding each sink is lowered **once** here into the `Expr` form `es-ir`
//! already defines — and evaluated with `Expr::eval`, which is already specified as
//! deterministic. Anything outside the supported node set is named, never approximated
//! (see `docs/design/batch-domains.md` §6).

use std::collections::BTreeMap;

use es_assets::scene::SceneDesc;
use es_ir::graph::{NodeId, PortRef};
use es_ir::task::{
    Aggregation, ArithOp, Expr, JointQuantity, LogicOp, TaskGraph, TaskIr, TaskNode,
    TerminationKind,
};
use es_physics_core::backend::ModelInfo;

use crate::EnvError;

/// Where a lowered leaf reads its value from, as an index into one env's state row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Qpos(u32),
    Qvel(u32),
    Sensor(u32),
    /// Simulation time in ticks; `since_reset` counts from the episode start.
    Time {
        since_reset: bool,
    },
}

/// One `Reward` node, lowered.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RewardTerm {
    pub name: String,
    pub weight: f64,
    pub expr: Expr,
}

/// Every reward and termination cone of a task, plus the leaves they read.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ScalarPlan {
    pub rewards: Vec<RewardTerm>,
    pub terminations: Vec<(TerminationKind, Expr)>,
    /// Port name to state index. `BTreeMap` so filling it is deterministic (§3.4).
    pub bindings: BTreeMap<String, Source>,
}

/// Cones deeper than this are refused rather than risking the stack. Authored graphs are
/// nowhere near it; a generated one that is has a lowering bug.
const MAX_DEPTH: u32 = 64;

impl ScalarPlan {
    /// Lowers every `Reward` and `Terminate` node, in ascending `NodeId` (§6.4: node id is the
    /// final tie-break, so a graph rewrite that keeps ids keeps the semantics).
    pub fn compile(task: &TaskIr, scene: &SceneDesc, model: &ModelInfo) -> Result<Self, EnvError> {
        let mut plan = Self::default();
        let mut ctx = Ctx {
            graph: &task.graph,
            scene,
            model,
            bindings: BTreeMap::new(),
        };
        for (id, node) in &task.graph.nodes {
            match node {
                TaskNode::Reward {
                    name,
                    weight,
                    aggregation,
                    ..
                } => {
                    // A scalar cone has one element, so Sum/Mean/Min/Max all reduce to it.
                    let _: Aggregation = *aggregation;
                    plan.rewards.push(RewardTerm {
                        name: name.clone(),
                        weight: *weight,
                        expr: ctx.lower_input(*id, "value", 0)?,
                    });
                }
                TaskNode::Terminate { kind } => {
                    plan.terminations
                        .push((*kind, ctx.lower_input(*id, "value", 0)?));
                }
                _ => {}
            }
        }
        plan.bindings = ctx.bindings;
        Ok(plan)
    }
}

struct Ctx<'a> {
    graph: &'a TaskGraph,
    scene: &'a SceneDesc,
    model: &'a ModelInfo,
    bindings: BTreeMap<String, Source>,
}

impl Ctx<'_> {
    /// The output port feeding `node`'s input `port`.
    fn producer(&self, node: NodeId, port: &str) -> Option<&PortRef> {
        self.graph
            .edges
            .iter()
            .find(|e| e.to.node == node && e.to.port == port)
            .map(|e| &e.from)
    }

    fn lower_input(&mut self, node: NodeId, port: &str, depth: u32) -> Result<Expr, EnvError> {
        if depth > MAX_DEPTH {
            return Err(EnvError::Unsupported(format!(
                "a reward/terminate cone deeper than {MAX_DEPTH} nodes"
            )));
        }
        let from = self
            .producer(node, port)
            .ok_or_else(|| {
                EnvError::Task(format!(
                    "node {} input \"{port}\" has no incoming edge",
                    node.0
                ))
            })?
            .clone();
        self.lower(&from, depth + 1)
    }

    fn lower(&mut self, from: &PortRef, depth: u32) -> Result<Expr, EnvError> {
        let id = from.node;
        let node = self
            .graph
            .nodes
            .get(&id)
            .ok_or_else(|| EnvError::Task(format!("edge names missing node {}", id.0)))?
            .clone();
        match &node {
            TaskNode::GetJointState {
                body,
                joints,
                quantity,
            } => self.joint_leaf(*body, joints, *quantity),
            TaskNode::GetSensor { sensor, .. } => {
                let range = self.model.sensor.get(sensor).ok_or_else(|| {
                    EnvError::Task(format!("sensor {sensor} is not in the loaded model"))
                })?;
                Ok(self.bind(
                    format!("sensor[{}]", range.start),
                    Source::Sensor(range.start),
                ))
            }
            TaskNode::GetTime { since_reset } => {
                let name = if *since_reset { "time.episode" } else { "time" };
                Ok(self.bind(
                    name.to_owned(),
                    Source::Time {
                        since_reset: *since_reset,
                    },
                ))
            }
            TaskNode::Arith { op, .. } => Ok(Expr::Arith {
                op: *op,
                lhs: Box::new(self.lower_input(id, "a", depth)?),
                rhs: Box::new(self.lower_input(id, "b", depth)?),
            }),
            TaskNode::Compare { op, rhs, .. } => {
                let lhs = Box::new(self.lower_input(id, "a", depth)?);
                let rhs = match rhs {
                    Some(c) => Box::new(Expr::Const(*c)),
                    None => Box::new(self.lower_input(id, "b", depth)?),
                };
                Ok(Expr::Compare { op: *op, lhs, rhs })
            }
            // `Normalize` is an affine map, and the clamp is what makes the declared
            // `Unit::Normalized { lo, hi }` true of the value and not only of the annotation
            // (`TYPE-011`): a reward term outside its own declared range is the bug the unit
            // algebra exists to catch.
            TaskNode::Normalize {
                lo,
                hi,
                out_lo,
                out_hi,
                ..
            } => {
                let (lo, hi) = (lo.first().copied(), hi.first().copied());
                let (Some(lo), Some(hi)) = (lo, hi) else {
                    return Err(EnvError::Task(
                        "Normalize with an empty lo/hi in a reward or termination cone".to_owned(),
                    ));
                };
                if (hi - lo).abs() < f64::EPSILON {
                    return Err(EnvError::Task(format!(
                        "Normalize maps the empty range [{lo}, {hi}]"
                    )));
                }
                let scale = (out_hi - out_lo) / (hi - lo);
                let shifted = Expr::Arith {
                    op: ArithOp::Sub,
                    lhs: Box::new(self.lower_input(id, "value", depth)?),
                    rhs: Box::new(Expr::Const(lo)),
                };
                Ok(Expr::Clamp {
                    value: Box::new(Expr::Arith {
                        op: ArithOp::Add,
                        lhs: Box::new(Expr::Const(*out_lo)),
                        rhs: Box::new(Expr::Arith {
                            op: ArithOp::Mul,
                            lhs: Box::new(shifted),
                            rhs: Box::new(Expr::Const(scale)),
                        }),
                    }),
                    lo: out_lo.min(*out_hi),
                    hi: out_lo.max(*out_hi),
                })
            }
            // `Compare` yields exactly 1.0 or 0.0 (`Expr::eval`), so the four logic ops are
            // arithmetic on those two values -- no new `Expr` variant, and no `es-ir` change.
            TaskNode::Logic { op, .. } => {
                let a = Box::new(self.lower_input(id, "a", depth)?);
                if *op == LogicOp::Not {
                    return Ok(Expr::Arith {
                        op: ArithOp::Sub,
                        lhs: Box::new(Expr::Const(1.0)),
                        rhs: a,
                    });
                }
                let b = Box::new(self.lower_input(id, "b", depth)?);
                let binary = |op| Expr::Arith {
                    op,
                    lhs: a.clone(),
                    rhs: b.clone(),
                };
                Ok(match op {
                    LogicOp::And => binary(ArithOp::Mul),
                    LogicOp::Or => binary(ArithOp::Max),
                    // |a - b| for values that are 0 or 1.
                    LogicOp::Xor => Expr::Arith {
                        op: ArithOp::Sub,
                        lhs: Box::new(binary(ArithOp::Max)),
                        rhs: Box::new(binary(ArithOp::Min)),
                    },
                    LogicOp::Not => unreachable!("handled above"),
                })
            }
            TaskNode::Clamp { lo, hi, .. } => Ok(Expr::Clamp {
                value: Box::new(self.lower_input(id, "value", depth)?),
                lo: lo.first().copied().unwrap_or(f64::MIN),
                hi: hi.first().copied().unwrap_or(f64::MAX),
            }),
            other => Err(EnvError::Unsupported(format!(
                "{} in a reward or termination cone",
                es_ir::graph::IrNode::kind(other)
            ))),
        }
    }

    fn joint_leaf(
        &mut self,
        body: es_core::StableId,
        joints: &[String],
        quantity: JointQuantity,
    ) -> Result<Expr, EnvError> {
        let name = joints.first().ok_or_else(|| {
            EnvError::Task(format!("GetJointState on body {body} names no joint"))
        })?;
        let joint = self
            .scene
            .joints
            .iter()
            .find(|j| &j.name == name)
            .ok_or_else(|| EnvError::Task(format!("no joint named \"{name}\" in the scene")))?;
        let (map, label, src): (_, _, fn(u32) -> Source) = match quantity {
            JointQuantity::Position => (&self.model.qpos, "qpos", Source::Qpos),
            JointQuantity::Velocity => (&self.model.dof, "qvel", Source::Qvel),
            JointQuantity::Torque => {
                return Err(EnvError::Unsupported(
                    "GetJointState(Torque): StateView has no joint torque array".to_owned(),
                ))
            }
        };
        let range = map.get(&joint.id).ok_or_else(|| {
            EnvError::Task(format!("joint \"{name}\" is not in the loaded model"))
        })?;
        let start = range.start;
        Ok(self.bind(format!("{label}[{start}]"), src(start)))
    }

    fn bind(&mut self, name: String, source: Source) -> Expr {
        self.bindings.insert(name.clone(), source);
        Expr::Port(name)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use es_assets::scene::SceneDesc;
    use es_core::StableId;
    use es_physics_core::backend::IndexRange;

    use super::*;

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn scene() -> SceneDesc {
        let path = repo_root().join("tests/fixtures/mjcf/so101_pick_place.xml");
        let xml =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        es_assets::parse_mjcf(&xml)
            .expect("the V0 fixture parses")
            .scene
    }

    fn joint_id(scene: &SceneDesc, name: &str) -> StableId {
        scene
            .joints
            .iter()
            .find(|j| j.name == name)
            .unwrap_or_else(|| panic!("the fixture has a joint named {name}"))
            .id
    }

    fn task_ir() -> TaskIr {
        let path = repo_root().join("tests/fixtures/visible-learning/task.toml");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        es_ir::serial::task_from_toml(&text).expect("V0's task document parses")
    }

    /// The lowering V0's documents need and V1 adds (`Normalize`, `Logic`): the demo task's
    /// reward and termination cones compile, on any machine, with no backend at all.
    #[test]
    fn the_demo_task_cones_lower() {
        let task = task_ir();
        let scene = scene();
        // A model with every joint the cones read, shaped as MuJoCo lays the demo scene out.
        let mut model = ModelInfo {
            nq: 13,
            nv: 12,
            nu: 6,
            nbody: 10,
            ..ModelInfo::default()
        };
        let names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ];
        for (i, name) in names.into_iter().enumerate() {
            let id = joint_id(&scene, name);
            model.qpos.insert(id, IndexRange::new(i as u32, 1));
            model.dof.insert(id, IndexRange::new(i as u32, 1));
        }
        let cube = joint_id(&scene, "cube_free");
        model.qpos.insert(cube, IndexRange::new(6, 7));
        model.dof.insert(cube, IndexRange::new(6, 6));

        let plan = ScalarPlan::compile(&task, &scene, &model).expect("the demo task's cones lower");
        assert_eq!(plan.rewards.len(), 1, "one shaped reward");
        assert_eq!(plan.terminations.len(), 3, "success, failure, timeout");

        // The success predicate is true for a cube inside the bin's x span and at rest, and false
        // for one still on the table -- evaluated through the very `Expr` the env runs.
        // The cube's free joint starts at `qpos[6]`; the gripper's own joint is `qpos[5]`.
        let ports = |x: f64, vx: f64, grip: f64| {
            let mut p = BTreeMap::new();
            for (name, source) in &plan.bindings {
                p.insert(
                    name.clone(),
                    match source {
                        Source::Qpos(6) => x,
                        Source::Qpos(_) => grip,
                        Source::Qvel(_) => vx,
                        Source::Sensor(_) | Source::Time { .. } => 0.0,
                    },
                );
            }
            p
        };
        // What the demonstration commands: the predicate's threshold is 0.6, because a jaw
        // holding the cube stalls near 0.30 (design note section 7.5).
        let (open, closed) = (0.9, 0.30);
        let success = plan
            .terminations
            .iter()
            .find(|(kind, _)| *kind == TerminationKind::Success)
            .expect("a Success node")
            .1
            .clone();
        assert_eq!(
            success.eval(&ports(0.14, 0.0, open)),
            Some(1.0),
            "released in the bin"
        );
        assert_eq!(
            success.eval(&ports(0.14, 0.0, closed)),
            Some(0.0),
            "carried across the bin, still in the jaws"
        );
        assert_eq!(
            success.eval(&ports(0.24, 0.0, open)),
            Some(0.0),
            "on the table"
        );
        assert_eq!(
            success.eval(&ports(0.14, 2.0, open)),
            Some(0.0),
            "still moving"
        );

        // The shaped reward is normalized, whatever the cube does (`TYPE-011`).
        let reward = &plan.rewards[0].expr;
        for x in [-1.0, 0.0, 0.14, 0.3, 5.0] {
            let v = reward
                .eval(&ports(x, 0.0, open))
                .expect("the reward evaluates");
            assert!((0.0..=1.0).contains(&v), "reward {v} for x = {x}");
        }
    }
}
