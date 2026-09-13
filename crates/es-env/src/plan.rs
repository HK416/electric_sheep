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
use es_ir::task::{Aggregation, Expr, JointQuantity, TaskGraph, TaskIr, TaskNode, TerminationKind};
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
