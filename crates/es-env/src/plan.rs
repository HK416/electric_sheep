//! Lowering the reward and termination cones of a Task IR into [`Expr`] (§6.4, §6.5).
//!
//! `Reward` and `Terminate` are graph sinks with an input edge, not expression literals, so the
//! scalar sub-graph feeding each sink is lowered **once** here into the `Expr` form `es-ir`
//! already defines — and evaluated with `Expr::eval`, which is already specified as
//! deterministic. Anything outside the supported node set is named, never approximated
//! (see `docs/design/batch-domains.md` §6).

use std::collections::BTreeMap;

use es_assets::scene::{JointKind, SceneDesc};
use es_ir::graph::{NodeId, PortRef};
use es_ir::task::{
    Aggregation, ArithOp, Expr, JointQuantity, LogicOp, MathFunc, NormKind, ReduceOp, TaskGraph,
    TaskIr, TaskNode, TerminationKind,
};
use es_ir::types::Frame;
use es_physics_core::backend::{ModelInfo, StateView};

use crate::env::at;
use crate::traj::Trajectory;
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
    /// One axis of one body's world position: `xpos[row * 3 + axis]` of the env's row, with
    /// `row` from [`ModelInfo::body`] (spec 6.3 `GetBodyPose`).
    Xpos {
        row: u32,
        axis: u32,
    },
    /// One component of one body's world orientation: `xquat[row * 4 + axis]`, in the order
    /// `StateView` stores it -- `x y z w` (spec 3.1) -- and unit length, because the backend
    /// computes it from `qpos` (spec 6.3 `GetBodyPose.quat`, packet M16/H2).
    Xquat {
        row: u32,
        axis: u32,
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
                        expr: ctx.scalar_input(*id, "value", 0, "Reward")?,
                    });
                }
                TaskNode::Terminate { kind } => {
                    plan.terminations
                        .push((*kind, ctx.scalar_input(*id, "value", 0, "Terminate")?));
                }
                _ => {}
            }
        }
        plan.bindings = ctx.bindings;
        Ok(plan)
    }
}

impl Source {
    /// This leaf's value in env `env` of `state`; `secs(since_reset)` is the time it reads.
    pub(crate) fn read(
        self,
        state: &StateView<'_>,
        model: &ModelInfo,
        env: u32,
        secs: impl Fn(bool) -> f64,
    ) -> f64 {
        match self {
            Source::Qpos(i) => at(state.qpos, env, model.nq, i),
            Source::Qvel(i) => at(state.qvel, env, model.nv, i),
            Source::Sensor(i) => at(state.sensordata, env, model.nsensordata, i),
            // `xpos` is `n_envs * nbody * 3`, env-major (§18.5): the body's row times three,
            // plus the axis.
            Source::Xpos { row, axis } => at(state.xpos, env, model.nbody * 3, row * 3 + axis),
            // `xquat` is `n_envs * nbody * 4`, env-major, `x y z w` (packet M16/H2).
            Source::Xquat { row, axis } => at(state.xquat, env, model.nbody * 4, row * 4 + axis),
            Source::Time { since_reset } => secs(since_reset),
        }
    }
}

/// The value of each node's `value` output on env 0 of `state`, `secs` after its reset: the cone
/// ending there lowered exactly as a `Terminate` input is, and evaluated as the env evaluates it
/// (a predicate is 1.0 or 0.0). What explains an attempt after the fact: each clause's truth on
/// its recorded end state (design note `scene-authoring.md` section 4.8, review M17 F-11).
pub fn eval_nodes(
    task: &TaskIr,
    scene: &SceneDesc,
    model: &ModelInfo,
    nodes: &[NodeId],
    state: &StateView<'_>,
    secs: f64,
) -> Result<Vec<f64>, EnvError> {
    let mut ctx = Ctx {
        graph: &task.graph,
        scene,
        model,
        bindings: BTreeMap::new(),
    };
    let mut exprs = Vec::with_capacity(nodes.len());
    for node in nodes {
        let lanes = ctx.lower(&PortRef::new(*node, "value"), 1)?;
        let n = lanes.len();
        exprs.push(lanes.into_iter().next().filter(|_| n == 1).ok_or_else(|| {
            EnvError::Unsupported(format!("node {} is {n} lanes, not one", node.0))
        })?);
    }
    if state.sensordata.is_empty() {
        if let Some(name) =
            (ctx.bindings.iter()).find_map(|(n, s)| matches!(s, Source::Sensor(_)).then_some(n))
        {
            return Err(EnvError::Unsupported(format!(
                "{name}: the state carries no sensor"
            )));
        }
    }
    let ports: BTreeMap<String, f64> = (ctx.bindings.iter())
        .map(|(name, s)| (name.clone(), s.read(state, model, 0, |_| secs)))
        .collect();
    exprs
        .iter()
        .zip(nodes)
        .map(|(e, node)| {
            e.eval(&ports)
                .ok_or_else(|| EnvError::Task(format!("node {} does not evaluate", node.0)))
        })
        .collect()
}

/// [`eval_nodes`] on row `tick` of `traj`, laid out as `model` says: its `qpos`, `qvel`, and each
/// body's recorded pose at the row `model.body` gives it, `tick / control_rate_hz` seconds after
/// the reset. A trajectory records no `sensordata`, so a cone that reads a sensor is refused, as
/// is a body the trajectory lacks.
pub fn eval_on_row(
    task: &TaskIr,
    scene: &SceneDesc,
    model: &ModelInfo,
    nodes: &[NodeId],
    traj: &Trajectory,
    tick: usize,
) -> Result<Vec<f64>, EnvError> {
    let (n, (nq, nv)) = (traj.ticks(), (model.nq as usize, model.nv as usize));
    let widths = (tick < n).then(|| (traj.qpos(tick), traj.qvel(tick)));
    let Some((qpos, qvel)) = widths.filter(|(q, v)| (q.len(), v.len()) == (nq, nv)) else {
        let why = format!("row {tick} of {n}, `qpos` / `qvel` {nq} / {nv} wide");
        return Err(EnvError::Trajectory(why));
    };
    let poses = traj.poses(tick);
    let (mut xpos, mut xquat) = (
        vec![0.0; model.nbody as usize * 3],
        vec![0.0; model.nbody as usize * 4],
    );
    for (id, range) in &model.body {
        let (Some(p), r) = (poses.get(id), range.start as usize) else {
            return Err(EnvError::Trajectory(format!("body {id} is not recorded")));
        };
        xpos[r * 3..r * 3 + 3].copy_from_slice(&[p.position.x, p.position.y, p.position.z]);
        let q = p.orientation;
        xquat[r * 4..r * 4 + 4].copy_from_slice(&[q.x, q.y, q.z, q.w]);
    }
    let state = StateView {
        n_envs: 1,
        qpos,
        qvel,
        xpos: &xpos,
        xquat: &xquat,
        ..StateView::default()
    };
    let secs = tick as f64 / f64::from(task.config.control_rate_hz);
    eval_nodes(task, scene, model, nodes, &state, secs)
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

    /// Every lane feeding `node`'s input `port`.
    fn lower_input(&mut self, node: NodeId, port: &str, depth: u32) -> Result<Vec<Expr>, EnvError> {
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

    /// The one lane feeding `node`'s input `port`, for a node with no vector meaning.
    ///
    /// A vector arriving here is refused **by name**: silently taking lane 0 would score a
    /// cube's `x` as if it were a distance, which is the class of bug this lowering exists to
    /// make impossible.
    fn scalar_input(
        &mut self,
        node: NodeId,
        port: &str,
        depth: u32,
        kind: &str,
    ) -> Result<Expr, EnvError> {
        let lanes = self.lower_input(node, port, depth)?;
        let n = lanes.len();
        lanes.into_iter().next().filter(|_| n == 1).ok_or_else(|| {
            EnvError::Unsupported(format!(
                "{kind} input \"{port}\" takes one lane, not {n}, in a reward or termination \
                 cone (reduce the vector with Norm first)"
            ))
        })
    }

    /// One [`Expr`] per lane: a scalar leaf is one lane, a `GetBodyPose` position is three.
    fn lower(&mut self, from: &PortRef, depth: u32) -> Result<Vec<Expr>, EnvError> {
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
                Ok(vec![self.bind(
                    format!("sensor[{}]", range.start),
                    Source::Sensor(range.start),
                )])
            }
            TaskNode::GetTime { since_reset } => {
                let name = if *since_reset { "time.episode" } else { "time" };
                Ok(vec![self.bind(
                    name.to_owned(),
                    Source::Time {
                        since_reset: *since_reset,
                    },
                )])
            }
            // Spec 6.3: the pose's two output ports are `pos` (3) and `quat` (4); only `pos`
            // is in `StateView` as three numbers a cone can subtract.
            TaskNode::GetBodyPose { body, relative_to } => {
                self.body_pose_leaf(*body, *relative_to, &from.port)
            }
            TaskNode::GetBodyVelocity { body, relative_to } => {
                self.body_velocity_leaf(*body, *relative_to, &from.port)
            }
            // Packet M17/G3c: a cone's value is one row of lanes, so axis 0 is the only axis.
            TaskNode::Slice {
                axis, start, len, ..
            } => {
                let lanes = self.lower_input(id, "value", depth)?;
                let (s, n) = (*start as usize, *len as usize);
                if *axis != 0 || n == 0 || s + n > lanes.len() {
                    return Err(EnvError::Unsupported(format!(
                        "Slice {{ axis: {axis}, start: {start}, len: {len} }} of {} lanes in a \
                         reward or termination cone",
                        lanes.len()
                    )));
                }
                Ok(lanes[s..s + n].to_vec())
            }
            TaskNode::Concat { parts, axis } => {
                if *axis != 0 {
                    return Err(EnvError::Unsupported(format!(
                        "Concat {{ axis: {axis} }} in a reward or termination cone"
                    )));
                }
                let mut lanes = Vec::new();
                for i in 0..parts.len() {
                    lanes.extend(self.lower_input(id, &format!("in{i}"), depth)?);
                }
                Ok(lanes)
            }
            // Folded in lane order, the association `Norm` and `Dot` use (`DET-020`); `Mean` is
            // that sum divided by the lane count. `unordered = true` lets any order stand, and
            // lane order is one (the validator refuses it in deterministic mode, `DET-030`).
            TaskNode::Reduce { op, axis, .. } => {
                let lanes = self.lower_input(id, "value", depth)?;
                let n = lanes.len();
                let fold_op = match op {
                    ReduceOp::Sum | ReduceOp::Mean => ArithOp::Add,
                    ReduceOp::Min => ArithOp::Min,
                    ReduceOp::Max => ArithOp::Max,
                };
                let folded = fold(fold_op, lanes).filter(|_| *axis == 0).ok_or_else(|| {
                    EnvError::Unsupported(format!(
                        "Reduce {{ axis: {axis} }} over {n} lanes in a reward or termination cone"
                    ))
                })?;
                Ok(vec![if *op == ReduceOp::Mean {
                    arith(ArithOp::Div, folded, Expr::Const(n as f64))
                } else {
                    folded
                }])
            }
            TaskNode::Arith { op, .. } => {
                let a = self.lower_input(id, "a", depth)?;
                let b = self.lower_input(id, "b", depth)?;
                lane_wise(*op, &a, &b)
            }
            // Spec 6.5's own example is `GetBodyPose(a) - GetBodyPose(b) -> Norm(L2) ->
            // Compare`. The sum is folded in lane order (`DET-020`), so the association is
            // `Sqrt(((x*x + y*y) + z*z))` on every backend and every run; `Sqrt` is the IEEE
            // basic operation, not a `DET-010` transcendental (spec 6.6).
            TaskNode::Norm {
                kind: NormKind::L2, ..
            } => {
                let lanes = self.lower_input(id, "value", depth)?;
                let squares = lanes.into_iter().map(|x| arith(ArithOp::Mul, x.clone(), x));
                let sum = fold(ArithOp::Add, squares)
                    .ok_or_else(|| EnvError::Task("Norm over a value with no lanes".to_owned()))?;
                Ok(vec![Expr::Sqrt(Box::new(sum))])
            }
            TaskNode::Norm { kind, .. } => Err(EnvError::Unsupported(format!(
                "Norm {{ kind: {kind:?} }} in a reward or termination cone"
            ))),
            // Packet M16/H2: the products summed in lane order, the association `Norm { L2 }`
            // uses (`DET-020`) -- `((a0*b0 + a1*b1) + a2*b2) + a3*b3` for two quaternions.
            TaskNode::Dot { .. } => {
                let a = self.lower_input(id, "a", depth)?;
                let b = self.lower_input(id, "b", depth)?;
                if a.len() != b.len() || a.is_empty() {
                    return Err(EnvError::Unsupported(format!(
                        "Dot between {} lanes and {} lanes in a reward or termination cone",
                        a.len(),
                        b.len()
                    )));
                }
                Ok(fold(ArithOp::Add, lane_wise(ArithOp::Mul, &a, &b)?)
                    .into_iter()
                    .collect())
            }
            // Lane by lane. `Abs` is `max(x, 0 - x)` and `Sqrt` the IEEE root; both are basic
            // operations, not `DET-010` transcendentals, and `Expr` has no polynomial for the
            // rest, which are refused by name.
            TaskNode::MathFn { func, .. } => {
                let lanes = self.lower_input(id, "value", depth)?;
                match func {
                    MathFunc::Abs => Ok(lanes
                        .into_iter()
                        .map(|x| Expr::Arith {
                            op: ArithOp::Max,
                            lhs: Box::new(x.clone()),
                            rhs: Box::new(Expr::Arith {
                                op: ArithOp::Sub,
                                lhs: Box::new(Expr::Const(0.0)),
                                rhs: Box::new(x),
                            }),
                        })
                        .collect()),
                    MathFunc::Sqrt => {
                        Ok(lanes.into_iter().map(|x| Expr::Sqrt(Box::new(x))).collect())
                    }
                    other => Err(EnvError::Unsupported(format!(
                        "MathFn {{ func: {other:?} }} in a reward or termination cone"
                    ))),
                }
            }
            TaskNode::Compare { op, rhs, .. } => {
                let lhs = Box::new(self.scalar_input(id, "a", depth, "Compare")?);
                let rhs = match rhs {
                    Some(c) => Box::new(Expr::Const(*c)),
                    None => Box::new(self.scalar_input(id, "b", depth, "Compare")?),
                };
                Ok(vec![Expr::Compare { op: *op, lhs, rhs }])
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
                // One lane reads `lo[0]` / `hi[0]` as it always did; a vector reads one bound
                // pair per lane (packet M16/H2), and a count that matches neither is refused.
                let lanes = self.lower_input(id, "value", depth)?;
                if lanes.len() > 1 && (lo.len() != lanes.len() || hi.len() != lanes.len()) {
                    return Err(EnvError::Unsupported(format!(
                        "Normalize over {} lanes with {} lo and {} hi bounds in a reward or \
                         termination cone",
                        lanes.len(),
                        lo.len(),
                        hi.len()
                    )));
                }
                if lanes.is_empty() || lo.is_empty() || hi.is_empty() {
                    return Err(EnvError::Task(
                        "Normalize with an empty lo/hi in a reward or termination cone".to_owned(),
                    ));
                }
                lanes
                    .into_iter()
                    .enumerate()
                    .map(|(i, lane)| affine(lane, lo[i], hi[i], *out_lo, *out_hi))
                    .collect()
            }
            // `Compare` yields exactly 1.0 or 0.0 (`Expr::eval`), so the four logic ops are
            // arithmetic on those two values -- no new `Expr` variant, and no `es-ir` change.
            TaskNode::Logic { op, .. } => {
                let a = Box::new(self.scalar_input(id, "a", depth, "Logic")?);
                if *op == LogicOp::Not {
                    return Ok(vec![Expr::Arith {
                        op: ArithOp::Sub,
                        lhs: Box::new(Expr::Const(1.0)),
                        rhs: a,
                    }]);
                }
                let b = Box::new(self.scalar_input(id, "b", depth, "Logic")?);
                let binary = |op| Expr::Arith {
                    op,
                    lhs: a.clone(),
                    rhs: b.clone(),
                };
                Ok(vec![match op {
                    LogicOp::And => binary(ArithOp::Mul),
                    LogicOp::Or => binary(ArithOp::Max),
                    // |a - b| for values that are 0 or 1.
                    LogicOp::Xor => Expr::Arith {
                        op: ArithOp::Sub,
                        lhs: Box::new(binary(ArithOp::Max)),
                        rhs: Box::new(binary(ArithOp::Min)),
                    },
                    LogicOp::Not => unreachable!("handled above"),
                }])
            }
            TaskNode::Clamp { lo, hi, .. } => Ok(vec![Expr::Clamp {
                value: Box::new(self.scalar_input(id, "value", depth, "Clamp")?),
                lo: lo.first().copied().unwrap_or(f64::MIN),
                hi: hi.first().copied().unwrap_or(f64::MAX),
            }]),
            other => Err(EnvError::Unsupported(format!(
                "{} in a reward or termination cone",
                es_ir::graph::IrNode::kind(other)
            ))),
        }
    }

    /// The three world-position lanes of a body, in `x, y, z` order, or its four orientation
    /// lanes (`quat`, `x, y, z, w`).
    ///
    /// Only `Frame::World`: `StateView` carries `xpos` and `xquat` in the world frame, and a
    /// relative frame is refused naming what was asked for rather than approximated.
    fn body_pose_leaf(
        &mut self,
        body: es_core::StableId,
        relative_to: Frame,
        port: &str,
    ) -> Result<Vec<Expr>, EnvError> {
        let name = self.body_name(body);
        if relative_to != Frame::World {
            return Err(EnvError::Unsupported(format!(
                "GetBodyPose(\"{name}\") relative to {relative_to:?} in a reward or \
                 termination cone"
            )));
        }
        if port != "pos" && port != "quat" {
            return Err(EnvError::Unsupported(format!(
                "GetBodyPose(\"{name}\").{port} in a reward or termination cone"
            )));
        }
        let row = self
            .model
            .body
            .get(&body)
            .ok_or_else(|| {
                EnvError::Unsupported(format!("body \"{name}\" is not in the loaded model"))
            })?
            .start;
        // `quat` is four lanes in `StateView`'s `x y z w` order (packet M16/H2). Nothing
        // lane-wise mixes a quaternion with a position -- the widths differ and `lane_wise`
        // refuses them -- and a dot product of two quaternions is order-free.
        if port == "quat" {
            return Ok((0..4)
                .map(|axis| {
                    self.bind(
                        format!("xquat[{}]", row * 4 + axis),
                        Source::Xquat { row, axis },
                    )
                })
                .collect());
        }
        Ok((0..3)
            .map(|axis| {
                self.bind(
                    format!("xpos[{}]", row * 3 + axis),
                    Source::Xpos { row, axis },
                )
            })
            .collect())
    }

    fn body_name(&self, body: es_core::StableId) -> String {
        self.scene
            .bodies
            .iter()
            .find(|b| b.id == body)
            .map_or_else(|| body.to_string(), |b| b.name.clone())
    }

    /// A free body's velocity (packet M17/G3c). `StateView` has no body-velocity array; what it
    /// carries exactly is the free joint's six `qvel` in `MuJoCo`'s convention, which every
    /// backend's view follows (`physx_ref.py` converts to it): the body origin's linear velocity
    /// in the world frame, then the angular velocity in the body frame. So `linear` is the
    /// first three lanes as they are -- the very ports `GetJointState(Velocity)` binds -- and
    /// `angular` is the last three rotated into the world by the orientation `GetBodyPose.quat`
    /// reads (`xquat`, unit). A body without a free joint (a robot link) would need its
    /// Jacobian, which `StateView` does not carry, and is refused by name.
    fn body_velocity_leaf(
        &mut self,
        body: es_core::StableId,
        relative_to: Frame,
        port: &str,
    ) -> Result<Vec<Expr>, EnvError> {
        let name = self.body_name(body);
        if relative_to != Frame::World || (port != "linear" && port != "angular") {
            return Err(EnvError::Unsupported(format!(
                "GetBodyVelocity(\"{name}\").{port} relative to {relative_to:?} in a reward or \
                 termination cone"
            )));
        }
        let free = self
            .scene
            .joints
            .iter()
            .find(|j| j.body == body && j.kind == JointKind::Free)
            .and_then(|j| self.model.dof.get(&j.id))
            .ok_or_else(|| {
                EnvError::Unsupported(format!(
                    "GetBodyVelocity(\"{name}\") in a reward or termination cone: the body has \
                     no free joint in the loaded model, and StateView carries no other body \
                     velocity"
                ))
            })?;
        let lanes = if port == "linear" { 0..3 } else { 3..6 };
        let v: Vec<Expr> = lanes
            .map(|i| {
                let at = free.start + i;
                self.bind(format!("qvel[{at}]"), Source::Qvel(at))
            })
            .collect();
        if port == "linear" {
            return Ok(v);
        }
        let q = self.body_pose_leaf(body, relative_to, "quat")?;
        Ok(rotate(&q, &v))
    }

    fn joint_leaf(
        &mut self,
        body: es_core::StableId,
        joints: &[String],
        quantity: JointQuantity,
    ) -> Result<Vec<Expr>, EnvError> {
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
        Ok(vec![self.bind(format!("{label}[{start}]"), src(start))])
    }

    fn bind(&mut self, name: String, source: Source) -> Expr {
        self.bindings.insert(name.clone(), source);
        Expr::Port(name)
    }
}

/// `Normalize`'s affine map of one lane, `out_lo + (x - lo) * scale`, clamped to the output
/// range -- the association every committed cone was lowered with.
fn affine(x: Expr, lo: f64, hi: f64, out_lo: f64, out_hi: f64) -> Result<Expr, EnvError> {
    if (hi - lo).abs() < f64::EPSILON {
        return Err(EnvError::Task(format!(
            "Normalize maps the empty range [{lo}, {hi}]"
        )));
    }
    let scale = (out_hi - out_lo) / (hi - lo);
    let shifted = Expr::Arith {
        op: ArithOp::Sub,
        lhs: Box::new(x),
        rhs: Box::new(Expr::Const(lo)),
    };
    Ok(Expr::Clamp {
        value: Box::new(Expr::Arith {
            op: ArithOp::Add,
            lhs: Box::new(Expr::Const(out_lo)),
            rhs: Box::new(Expr::Arith {
                op: ArithOp::Mul,
                lhs: Box::new(shifted),
                rhs: Box::new(Expr::Const(scale)),
            }),
        }),
        lo: out_lo.min(out_hi),
        hi: out_lo.max(out_hi),
    })
}

fn arith(op: ArithOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Arith {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

/// `((l0 op l1) op l2) op ...`: lane order, the one association every reduction here uses.
fn fold(op: ArithOp, lanes: impl IntoIterator<Item = Expr>) -> Option<Expr> {
    lanes.into_iter().reduce(|acc, x| arith(op, acc, x))
}

/// `v` rotated by the unit quaternion `q` (`x y z w`): `v + w t + u × t`, `t = 2 (u × v)`,
/// `u = (x, y, z)` -- polynomial, so no `DET-010` function is involved.
// Written in the formula's symbols.
#[allow(clippy::many_single_char_names)]
fn rotate(q: &[Expr], v: &[Expr]) -> Vec<Expr> {
    let cross = |a: &[Expr], b: &[Expr]| -> Vec<Expr> {
        (0..3)
            .map(|i| {
                let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                arith(
                    ArithOp::Sub,
                    arith(ArithOp::Mul, a[j].clone(), b[k].clone()),
                    arith(ArithOp::Mul, a[k].clone(), b[j].clone()),
                )
            })
            .collect()
    };
    let (u, w) = (&q[..3], &q[3]);
    let t: Vec<Expr> = cross(u, v)
        .into_iter()
        .map(|c| arith(ArithOp::Mul, Expr::Const(2.0), c))
        .collect();
    let ut = cross(u, &t);
    (0..3)
        .map(|i| {
            let wt = arith(ArithOp::Mul, w.clone(), t[i].clone());
            arith(
                ArithOp::Add,
                arith(ArithOp::Add, v[i].clone(), wt),
                ut[i].clone(),
            )
        })
        .collect()
}

/// `Arith` lane by lane, with a one-lane operand broadcast over the other side.
///
/// Anything else -- 3 against 4, say -- is refused by width: the IR's own type check compares
/// shapes, and a cone that got past it with mismatched lanes is a lowering bug, not a
/// reduction this function is allowed to invent.
fn lane_wise(op: ArithOp, a: &[Expr], b: &[Expr]) -> Result<Vec<Expr>, EnvError> {
    let binary = |lhs: &Expr, rhs: &Expr| Expr::Arith {
        op,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    };
    match (a.len(), b.len()) {
        (n, m) if n == m => Ok(a.iter().zip(b).map(|(l, r)| binary(l, r)).collect()),
        (_, 1) => Ok(a.iter().map(|l| binary(l, &b[0])).collect()),
        (1, _) => Ok(b.iter().map(|r| binary(&a[0], r)).collect()),
        (n, m) => Err(EnvError::Unsupported(format!(
            "Arith {op:?} between {n} lanes and {m} lanes in a reward or termination cone"
        ))),
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
                        Source::Sensor(_)
                        | Source::Time { .. }
                        | Source::Xpos { .. }
                        | Source::Xquat { .. } => 0.0,
                    },
                );
            }
            p
        };
        // What the demonstration commands: the predicate's threshold is 0.85, because a jaw
        // holding the 30 mm cube stalls near 0.09 and one that is merely *opening* is still in
        // contact with it (design note sections 7.5 and 7.23).
        let (open, closed, opening) = (0.9, 0.30, 0.6);
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
            success.eval(&ports(0.14, 0.0, opening)),
            Some(0.0),
            "the jaw is opening but has not let go yet"
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

    // --- packet M8/S4d: body positions as lanes, `Norm { L2 }`, and the demo task unmoved ---

    /// What the committed demo task lowered to **before** lanes existed, captured from this
    /// file's own `the_demo_task_cones_lower` on 2026-09-21. The reach cone must not cost the
    /// demo a single byte: same association, same constants, same port names.
    const PINNED_DEMO_REWARDS: &str = "[RewardTerm { name: \"cube_towards_bin\", weight: -1.0, \
expr: Clamp { value: Arith { op: Add, lhs: Const(0.0), rhs: Arith { op: Mul, lhs: Arith { op: \
Sub, lhs: Port(\"qpos[6]\"), rhs: Const(0.09) }, rhs: Const(4.761904761904762) } }, lo: 0.0, \
hi: 1.0 } }]";
    const PINNED_DEMO_TERMS: &str = "[(Success, Arith { op: Mul, lhs: Arith { op: Mul, lhs: \
Arith { op: Mul, lhs: Compare { op: Gt, lhs: Port(\"qpos[6]\"), rhs: Const(0.09) }, rhs: \
Compare { op: Lt, lhs: Port(\"qpos[6]\"), rhs: Const(0.19) } }, rhs: Arith { op: Mul, lhs: \
Compare { op: Gt, lhs: Port(\"qvel[6]\"), rhs: Const(-0.05) }, rhs: Compare { op: Lt, lhs: \
Port(\"qvel[6]\"), rhs: Const(0.05) } } }, rhs: Compare { op: Gt, lhs: Port(\"qpos[5]\"), rhs: \
Const(0.85) } }), (Failure, Compare { op: Gt, lhs: Port(\"qpos[6]\"), rhs: Const(0.55) }), \
(Timeout, Compare { op: Ge, lhs: Port(\"time.episode\"), rhs: Const(36.0) })]";
    const PINNED_DEMO_BINDINGS: &str = "{\"qpos[5]\": Qpos(5), \"qpos[6]\": Qpos(6), \
\"qvel[6]\": Qvel(6), \"time.episode\": Time { since_reset: true }}";

    fn vec_ty(n: u64, unit: es_ir::types::Unit) -> es_ir::types::PortType {
        es_ir::types::PortType {
            elem: es_ir::types::ElemType::F32,
            shape: es_ir::types::Shape::new([n]),
            unit,
            frame: es_ir::types::Frame::World,
            time: es_ir::types::TimeRef::Tick,
            image: None,
        }
    }

    fn body_id(scene: &SceneDesc, name: &str) -> StableId {
        scene
            .bodies
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("the fixture has a body named {name}"))
            .id
    }

    /// `-||pos(link) - pos(target)||` as a `Reward`, plus whatever `tail` adds.
    fn distance_task(scene: &SceneDesc) -> TaskIr {
        use es_ir::task::NormKind;
        let pose = |name: &str| TaskNode::GetBodyPose {
            body: body_id(scene, name),
            relative_to: es_ir::types::Frame::World,
        };
        let mut task = crate::env::tests::task_with(&[
            pose("link"),
            pose("target"),
            TaskNode::Arith {
                op: ArithOp::Sub,
                ty: vec_ty(3, es_ir::types::Unit::Length),
            },
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: vec_ty(3, es_ir::types::Unit::Length),
            },
            TaskNode::Reward {
                name: "reach".to_owned(),
                weight: -1.0,
                aggregation: Aggregation::Sum,
                ty: vec_ty(1, es_ir::types::Unit::Length),
            },
        ]);
        task.graph.connect(NodeId(0), "pos", NodeId(2), "a");
        task.graph.connect(NodeId(1), "pos", NodeId(2), "b");
        task.graph.connect(NodeId(2), "value", NodeId(3), "value");
        task.graph.connect(NodeId(3), "value", NodeId(4), "value");
        task
    }

    /// The cone of spec 6.5's own example, end to end on the fixture backend: two body
    /// positions, a lane-wise subtraction, an `L2` norm, and a reward that is the negated
    /// distance **bitwise** -- with the demo task's own cones byte-identical to before.
    // The bitwise reward is the property under test: this comparison is deliberate.
    #[allow(clippy::float_cmp)]
    #[test]
    fn body_norm_cone_lowers_and_the_demo_task_is_unmoved() {
        use crate::scheduler::{BatchDomains, DomainCfg};
        use crate::Env;
        use es_ir::task::{CmpOp, NormKind, TerminationKind};

        let scene = crate::env::tests::fake_scene();
        let model = crate::env::tests::fake_model();

        // --- the demo task is unmoved ---------------------------------------------------
        let demo = task_ir();
        let demo_scene = self::tests::scene();
        let mut demo_model = ModelInfo {
            nq: 13,
            nv: 12,
            nu: 6,
            nbody: 10,
            ..ModelInfo::default()
        };
        for (i, name) in [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        .into_iter()
        .enumerate()
        {
            let id = joint_id(&demo_scene, name);
            demo_model.qpos.insert(id, IndexRange::new(i as u32, 1));
            demo_model.dof.insert(id, IndexRange::new(i as u32, 1));
        }
        let cube = joint_id(&demo_scene, "cube_free");
        demo_model.qpos.insert(cube, IndexRange::new(6, 7));
        demo_model.dof.insert(cube, IndexRange::new(6, 6));
        let demo_plan = ScalarPlan::compile(&demo, &demo_scene, &demo_model)
            .expect("the demo task's cones lower");
        assert_eq!(format!("{:?}", demo_plan.rewards), PINNED_DEMO_REWARDS);
        assert_eq!(format!("{:?}", demo_plan.terminations), PINNED_DEMO_TERMS);
        assert_eq!(format!("{:?}", demo_plan.bindings), PINNED_DEMO_BINDINGS);

        // --- three lanes in, one reward out ----------------------------------------------
        let task = distance_task(&scene);
        let plan = ScalarPlan::compile(&task, &scene, &model).expect("the body cone lowers");
        assert_eq!(plan.bindings.len(), 6, "two bodies, three axes each");
        // `link` is scene body 1 and `target` is body 2 (`world` is 0), so the rows are 3..6
        // and 6..9 of the env's `xpos`.
        assert_eq!(plan.bindings["xpos[3]"], Source::Xpos { row: 1, axis: 0 });
        assert_eq!(plan.bindings["xpos[8]"], Source::Xpos { row: 2, axis: 2 });

        let mut env = Env::new(
            &task,
            &scene,
            crate::env::tests::FakeBackend::new(),
            &BatchDomains {
                simulation: DomainCfg::new(1, 1),
                observation: DomainCfg::new(1, 10),
                inference: DomainCfg::new(1, 20),
                training: None,
            },
            7,
        )
        .expect("the task compiles against the fixture model");

        // The world the cone is supposed to read: world at the origin, and two bodies a
        // distance apart that no power of two makes exact.
        let (link, target): ([f64; 3], [f64; 3]) = ([0.13, -0.27, 0.31], [0.4, 0.05, -0.17]);
        let mut xpos = vec![0.0; 9];
        xpos[3..6].copy_from_slice(&link);
        xpos[6..9].copy_from_slice(&target);
        env.backend_mut().xpos.copy_from_slice(&xpos);

        let out = env.step(&[0.0]).expect("one control step");
        let (dx, dy, dz) = (
            link[0] - target[0],
            link[1] - target[1],
            link[2] - target[2],
        );
        // That exact association: the squares are summed in lane order (`DET-020`) and the
        // root is IEEE (spec 6.6), so this is an `assert_eq!` on floats on purpose.
        let expected = -((dx * dx + dy * dy) + dz * dz).sqrt();
        assert_eq!(out.rewards[0], expected, "the reward is not -||a - b||");
        assert!(expected < 0.0, "a distance reward is a penalty");

        // --- a vector where a scalar belongs is a named error, not lane 0 -----------------
        let mut vector_into_compare = distance_task(&scene);
        vector_into_compare.graph.insert(
            NodeId(5),
            TaskNode::Compare {
                op: CmpOp::Lt,
                rhs: Some(0.03),
                ty: vec_ty(3, es_ir::types::Unit::Length),
            },
        );
        vector_into_compare.graph.insert(
            NodeId(6),
            TaskNode::Terminate {
                kind: TerminationKind::Success,
            },
        );
        vector_into_compare
            .graph
            .connect(NodeId(0), "pos", NodeId(5), "a");
        vector_into_compare
            .graph
            .connect(NodeId(5), "value", NodeId(6), "value");
        let err = ScalarPlan::compile(&vector_into_compare, &scene, &model)
            .expect_err("three lanes cannot be compared with 0.03");
        let text = err.to_string();
        assert!(
            text.contains("Compare") && text.contains('3'),
            "the refusal must name the node and the width: {text}"
        );

        // An orientation subtracted from a position, another frame, an L1 norm and a body the
        // model does not index are all refused the same way -- by name. The orientation is
        // four lanes since packet M16/H2, so the refusal names the widths.
        for (patch, wanted) in [
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.edges.retain(|e| e.from.node != NodeId(0));
                    t.graph.connect(NodeId(0), "quat", NodeId(2), "a");
                }) as Box<dyn Fn(&mut TaskIr)>,
                "4 lanes and 3 lanes",
            ),
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        NodeId(0),
                        TaskNode::GetBodyPose {
                            body: body_id(&crate::env::tests::fake_scene(), "link"),
                            relative_to: es_ir::types::Frame::LocalOrigin,
                        },
                    );
                }),
                "LocalOrigin",
            ),
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        NodeId(3),
                        TaskNode::Norm {
                            kind: NormKind::L1,
                            ty: vec_ty(3, es_ir::types::Unit::Length),
                        },
                    );
                }),
                "L1",
            ),
        ] {
            let mut broken = distance_task(&scene);
            patch(&mut broken);
            let err = ScalarPlan::compile(&broken, &scene, &model)
                .expect_err("an unsupported cone must be refused");
            assert!(
                err.to_string().contains(wanted),
                "the refusal must name {wanted}: {err}"
            );
        }

        // A one-lane operand broadcasts: `(pos(link) - pos(target)) * time` is three lanes.
        let mut broadcast = distance_task(&scene);
        broadcast
            .graph
            .insert(NodeId(7), TaskNode::GetTime { since_reset: true });
        broadcast.graph.insert(
            NodeId(8),
            TaskNode::Arith {
                op: ArithOp::Mul,
                ty: vec_ty(3, es_ir::types::Unit::Length),
            },
        );
        broadcast.graph.edges.retain(|e| e.to.node != NodeId(3));
        broadcast.graph.connect(NodeId(2), "value", NodeId(8), "a");
        broadcast.graph.connect(NodeId(7), "value", NodeId(8), "b");
        broadcast
            .graph
            .connect(NodeId(8), "value", NodeId(3), "value");
        let plan = ScalarPlan::compile(&broadcast, &scene, &model)
            .expect("a scalar broadcasts over three lanes");
        assert!(
            plan.bindings.contains_key("time.episode"),
            "the broadcast operand is bound once: {:?}",
            plan.bindings
        );
        println!("RAN body_norm_cone_lowers_and_the_demo_task_is_unmoved");
    }

    /// Packet M16/H2: `|q_a . q_b|` from two `GetBodyPose.quat`s (`Dot`, `MathFn { Abs }`),
    /// `sqrt` of it, and a position offset from a constant by a per-lane `Normalize` -- each
    /// evaluated through the `Expr` the env runs, against the same numbers computed by hand.
    // Exact association is the property under test: these comparisons are deliberate.
    #[allow(clippy::float_cmp)]
    #[test]
    fn quaternion_dot_abs_sqrt_and_per_lane_normalize_lower() {
        use es_ir::task::{CmpOp, NormKind, TerminationKind};
        use es_ir::types::Unit;

        let scene = crate::env::tests::fake_scene();
        let model = crate::env::tests::fake_model();
        let pose = |name: &str| TaskNode::GetBodyPose {
            body: body_id(&scene, name),
            relative_to: es_ir::types::Frame::World,
        };
        let q1 = vec_ty(1, Unit::Quaternion);
        let p_ref = [0.25, -0.5, 0.125];
        let mut task = crate::env::tests::task_with(&[
            pose("link"),
            pose("target"),
            TaskNode::Dot {
                ty: vec_ty(4, Unit::Quaternion),
            },
            TaskNode::MathFn {
                func: MathFunc::Abs,
                approx: false,
                ty: q1.clone(),
            },
            TaskNode::Compare {
                op: CmpOp::Ge,
                rhs: Some(0.9),
                ty: q1.clone(),
            },
            TaskNode::Terminate {
                kind: TerminationKind::Success,
            },
            TaskNode::MathFn {
                func: MathFunc::Sqrt,
                approx: false,
                ty: q1.clone(),
            },
            TaskNode::Reward {
                name: "root".to_owned(),
                weight: 1.0,
                aggregation: Aggregation::Sum,
                ty: q1,
            },
            TaskNode::Normalize {
                lo: p_ref.iter().map(|p| p - 1.0).collect(),
                hi: p_ref.iter().map(|p| p + 1.0).collect(),
                out_lo: -1.0,
                out_hi: 1.0,
                ty: vec_ty(3, Unit::Length),
            },
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: vec_ty(3, Unit::Normalized { lo: -1.0, hi: 1.0 }),
            },
            TaskNode::Reward {
                name: "offset".to_owned(),
                weight: -1.0,
                aggregation: Aggregation::Sum,
                ty: vec_ty(1, Unit::Normalized { lo: -1.0, hi: 1.0 }),
            },
        ]);
        let n = NodeId;
        for (from, fp, to, tp) in [
            (0, "quat", 2, "a"),
            (1, "quat", 2, "b"),
            (2, "value", 3, "value"),
            (3, "value", 4, "a"),
            (4, "value", 5, "value"),
            (3, "value", 6, "value"),
            (6, "value", 7, "value"),
            (0, "pos", 8, "value"),
            (8, "value", 9, "value"),
            (9, "value", 10, "value"),
        ] {
            task.graph.connect(n(from), fp, n(to), tp);
        }
        let plan = ScalarPlan::compile(&task, &scene, &model).expect("the cones lower");
        // `link` is body row 1 and `target` row 2: `xquat[4..8]` and `xquat[8..12]`.
        assert_eq!(plan.bindings["xquat[4]"], Source::Xquat { row: 1, axis: 0 });
        assert_eq!(
            plan.bindings["xquat[11]"],
            Source::Xquat { row: 2, axis: 3 }
        );

        // Two unit quaternions (x y z w) whose dot product is negative: `Abs` makes the
        // double cover one orientation.
        let (a, b) = (
            [0.1, -0.2, 0.3, 0.927_361_849_549_570_3],
            [-0.3, 0.1, -0.2, -0.9],
        );
        let pos = [0.5, -0.25, 0.375];
        let mut ports = BTreeMap::new();
        for (i, (qa, qb)) in a.iter().zip(&b).enumerate() {
            ports.insert(format!("xquat[{}]", 4 + i), *qa);
            ports.insert(format!("xquat[{}]", 8 + i), *qb);
        }
        for (i, p) in pos.iter().enumerate() {
            ports.insert(format!("xpos[{}]", 3 + i), *p);
        }
        let dot = ((a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]) + a[3] * b[3];
        assert!(dot < 0.0);
        let root = plan.rewards[0].expr.eval(&ports).expect("evaluates");
        assert_eq!(root, dot.abs().sqrt(), "sqrt(|q_a . q_b|)");
        let success = &plan.terminations[0].1;
        assert_eq!(
            success.eval(&ports),
            Some(1.0),
            "|dot| = {} >= 0.9",
            dot.abs()
        );

        let offset = plan.rewards[1].expr.eval(&ports).expect("evaluates");
        let d: Vec<f64> = (0..3)
            .map(|i| -1.0 + (pos[i] - (p_ref[i] - 1.0)) * 1.0)
            .collect();
        assert_eq!(offset, ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt());
        // pos - p_ref = (0.25, 0.25, 0.25): the offset from the constant, not from the origin.
        assert!((offset - 0.25 * 3.0_f64.sqrt()).abs() < 1e-15, "{offset}");

        // A bound count that matches neither one lane nor the width is refused by name.
        let mut short = task.clone();
        short.graph.insert(
            n(8),
            TaskNode::Normalize {
                lo: vec![0.0, 0.0],
                hi: vec![1.0, 1.0],
                out_lo: -1.0,
                out_hi: 1.0,
                ty: vec_ty(3, Unit::Length),
            },
        );
        let err = ScalarPlan::compile(&short, &scene, &model).expect_err("2 bounds for 3 lanes");
        assert!(err.to_string().contains("3 lanes with 2 lo"), "{err}");

        // Packet M17/R8: the predicate and the offset's norm, read after the fact on the same
        // state, and on that state recorded as a trajectory row, are what the env reads.
        let (qpos, qvel) = ([0.0; 2], [0.0; 2]);
        let mut xpos = vec![0.0; model.nbody as usize * 3];
        let mut xquat = [0.0, 0.0, 0.0, 1.0].repeat(model.nbody as usize);
        xpos[3..6].copy_from_slice(&pos);
        xquat[4..8].copy_from_slice(&a);
        xquat[8..12].copy_from_slice(&b);
        let state = StateView {
            n_envs: 1,
            qpos: &qpos,
            qvel: &qvel,
            xpos: &xpos,
            xquat: &xquat,
            ..StateView::default()
        };
        let nodes = [n(4), n(9)];
        let now = eval_nodes(&task, &scene, &model, &nodes, &state, 0.0).expect("evaluates");
        assert_eq!(now, [1.0, offset]);
        let mut traj = Trajectory::new(&model);
        traj.push(&model, &state, 0).expect("one row");
        let row = eval_on_row(&task, &scene, &model, &nodes, &traj, 0).expect("evaluates");
        assert_eq!(row, now);
        assert!(
            eval_on_row(&task, &scene, &model, &nodes, &traj, 1).is_err(),
            "no row 1"
        );
        println!("RAN quaternion_dot_abs_sqrt_and_per_lane_normalize_lower");
    }

    /// Packet M17/G3c: `Slice`, `Concat`, `Reduce` and `GetBodyVelocity` in reward cones,
    /// each evaluated through the `Expr` the env runs against the same numbers by hand. The
    /// demo scene: six arm joints, then the cube's free joint (`qvel[6..12]`).
    // Exact association is the property under test: these comparisons are deliberate; the
    // rotation is written in the quaternion's symbols.
    #[allow(clippy::float_cmp, clippy::many_single_char_names)]
    #[test]
    fn slice_concat_reduce_and_body_velocity_lower() {
        use es_ir::task::ReduceOp;
        use es_ir::types::{Frame, Unit};

        let scene = scene();
        let mut model = ModelInfo {
            nq: 13,
            nv: 12,
            nbody: scene.bodies.len() as u32,
            ..ModelInfo::default()
        };
        for (row, b) in scene.bodies.iter().enumerate() {
            model.body.insert(b.id, IndexRange::new(row as u32, 1));
        }
        let cube_free = joint_id(&scene, "cube_free");
        model.qpos.insert(cube_free, IndexRange::new(6, 7));
        model.dof.insert(cube_free, IndexRange::new(6, 6));
        let (cube, base) = (body_id(&scene, "cube"), body_id(&scene, "base"));
        let row = |id: StableId| model.body[&id].start as usize;
        let (cr, br) = (row(cube), row(base));

        let p3 = vec_ty(3, Unit::Length);
        let reduce = |op| TaskNode::Reduce {
            op,
            axis: 0,
            unordered: false,
            ty: vec_ty(6, Unit::Length),
        };
        let slice = |ty: &es_ir::types::PortType, start| TaskNode::Slice {
            ty: ty.clone(),
            axis: 0,
            start,
            len: 1,
        };
        let reward = |name: &str| TaskNode::Reward {
            name: name.to_owned(),
            weight: 1.0,
            aggregation: Aggregation::Sum,
            ty: vec_ty(1, Unit::Length),
        };
        let w3 = vec_ty(3, Unit::AngularVelocity);
        let mut task = crate::env::tests::task_with(&[
            TaskNode::GetBodyPose {
                body: cube,
                relative_to: Frame::World,
            },
            TaskNode::GetBodyPose {
                body: base,
                relative_to: Frame::World,
            },
            TaskNode::GetBodyVelocity {
                body: cube,
                relative_to: Frame::World,
            },
            TaskNode::Concat {
                parts: vec![p3.clone(), p3.clone()],
                axis: 0,
            },
            reduce(ReduceOp::Sum),
            reward("sum"),
            reduce(ReduceOp::Mean),
            reward("mean"),
            reduce(ReduceOp::Min),
            reward("min"),
            reduce(ReduceOp::Max),
            reward("max"),
            slice(&p3, 2),
            reward("z"),
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: vec_ty(3, Unit::Velocity),
            },
            reward("speed"),
            slice(&w3, 0),
            reward("w0"),
            slice(&w3, 1),
            reward("w1"),
            slice(&w3, 2),
            reward("w2"),
        ]);
        let n = NodeId;
        for (from, fp, to, tp) in [
            (0, "pos", 3, "in0"),
            (1, "pos", 3, "in1"),
            (3, "value", 4, "value"),
            (4, "value", 5, "value"),
            (3, "value", 6, "value"),
            (6, "value", 7, "value"),
            (3, "value", 8, "value"),
            (8, "value", 9, "value"),
            (3, "value", 10, "value"),
            (10, "value", 11, "value"),
            (0, "pos", 12, "value"),
            (12, "value", 13, "value"),
            (2, "linear", 14, "value"),
            (14, "value", 15, "value"),
            (2, "angular", 16, "value"),
            (16, "value", 17, "value"),
            (2, "angular", 18, "value"),
            (18, "value", 19, "value"),
            (2, "angular", 20, "value"),
            (20, "value", 21, "value"),
        ] {
            task.graph.connect(n(from), fp, n(to), tp);
        }
        let plan = ScalarPlan::compile(&task, &scene, &model).expect("the cones lower");
        // `linear` is the free joint's own `qvel` ports, the ones `GetJointState` binds.
        assert_eq!(plan.bindings["qvel[6]"], Source::Qvel(6));
        assert_eq!(plan.bindings["qvel[11]"], Source::Qvel(11));

        let (c, b) = ([0.13, -0.27, 0.31], [-0.4, 0.05, 0.17]);
        let lin = [0.5, -0.25, 0.125];
        let omega = [0.3, -0.7, 0.2];
        let eval = |q: [f64; 4]| {
            let mut ports = BTreeMap::new();
            for i in 0..3 {
                ports.insert(format!("xpos[{}]", cr * 3 + i), c[i]);
                ports.insert(format!("xpos[{}]", br * 3 + i), b[i]);
                ports.insert(format!("qvel[{}]", 6 + i), lin[i]);
                ports.insert(format!("qvel[{}]", 9 + i), omega[i]);
            }
            for (i, v) in q.iter().enumerate() {
                ports.insert(format!("xquat[{}]", cr * 4 + i), *v);
            }
            plan.rewards
                .iter()
                .map(|r| (r.name.clone(), r.expr.eval(&ports).expect("evaluates")))
                .collect::<BTreeMap<_, _>>()
        };
        let q = [0.1, -0.2, 0.3, 0.927_361_849_549_570_3];
        let r = eval(q);
        let sum = ((((c[0] + c[1]) + c[2]) + b[0]) + b[1]) + b[2];
        assert_eq!(r["sum"], sum, "summed in lane order");
        assert_eq!(r["mean"], sum / 6.0);
        assert_eq!(r["min"], -0.4);
        assert_eq!(r["max"], 0.31);
        assert_eq!(r["z"], c[2]);
        assert_eq!(
            r["speed"],
            ((lin[0] * lin[0] + lin[1] * lin[1]) + lin[2] * lin[2]).sqrt()
        );
        // The body-frame rate in the world: `R(q) omega` with the rotation matrix of `q`.
        let [x, y, z, w] = q;
        let m = [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ];
        for (i, row) in m.iter().enumerate() {
            let want: f64 = (0..3).map(|j| row[j] * omega[j]).sum();
            let got = r[&format!("w{i}")];
            assert!((got - want).abs() < 1e-15, "lane {i}: {got} vs {want}");
        }
        // The identity orientation leaves the lanes as they are, bit for bit.
        let r = eval([0.0, 0.0, 0.0, 1.0]);
        assert_eq!([r["w0"], r["w1"], r["w2"]], omega);

        // Refused by name: a slice past the end, a reduction over a second axis, a body with
        // no free joint, another frame.
        for (patch, wanted) in [
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        n(12),
                        TaskNode::Slice {
                            ty: vec_ty(3, Unit::Length),
                            axis: 0,
                            start: 2,
                            len: 2,
                        },
                    );
                }) as Box<dyn Fn(&mut TaskIr)>,
                "Slice { axis: 0, start: 2, len: 2 } of 3 lanes",
            ),
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        n(4),
                        TaskNode::Reduce {
                            op: ReduceOp::Sum,
                            axis: 1,
                            unordered: false,
                            ty: vec_ty(6, Unit::Length),
                        },
                    );
                }),
                "Reduce { axis: 1 }",
            ),
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        n(2),
                        TaskNode::GetBodyVelocity {
                            body: body_id(&scene, "gripper"),
                            relative_to: Frame::World,
                        },
                    );
                }),
                "\"gripper\") in a reward or termination cone: the body has no free joint",
            ),
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.insert(
                        n(2),
                        TaskNode::GetBodyVelocity {
                            body: cube,
                            relative_to: Frame::LocalOrigin,
                        },
                    );
                }),
                "LocalOrigin",
            ),
        ] {
            let mut broken = task.clone();
            patch(&mut broken);
            let err = ScalarPlan::compile(&broken, &scene, &model)
                .expect_err("an unsupported cone must be refused");
            assert!(err.to_string().contains(wanted), "{wanted}: {err}");
        }
        println!("RAN slice_concat_reduce_and_body_velocity_lower");
    }
}
