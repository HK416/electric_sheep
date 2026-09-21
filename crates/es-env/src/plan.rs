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
    Aggregation, ArithOp, Expr, JointQuantity, LogicOp, NormKind, TaskGraph, TaskIr, TaskNode,
    TerminationKind,
};
use es_ir::types::Frame;
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
    /// One axis of one body's world position: `xpos[row * 3 + axis]` of the env's row, with
    /// `row` from [`ModelInfo::body`] (spec 6.3 `GetBodyPose`).
    Xpos {
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
                let mut sum: Option<Expr> = None;
                for lane in lanes {
                    let square = Expr::Arith {
                        op: ArithOp::Mul,
                        lhs: Box::new(lane.clone()),
                        rhs: Box::new(lane),
                    };
                    sum = Some(match sum {
                        None => square,
                        Some(acc) => Expr::Arith {
                            op: ArithOp::Add,
                            lhs: Box::new(acc),
                            rhs: Box::new(square),
                        },
                    });
                }
                let sum = sum
                    .ok_or_else(|| EnvError::Task("Norm over a value with no lanes".to_owned()))?;
                Ok(vec![Expr::Sqrt(Box::new(sum))])
            }
            TaskNode::Norm { kind, .. } => Err(EnvError::Unsupported(format!(
                "Norm {{ kind: {kind:?} }} in a reward or termination cone"
            ))),
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
                    lhs: Box::new(self.scalar_input(id, "value", depth, "Normalize")?),
                    rhs: Box::new(Expr::Const(lo)),
                };
                Ok(vec![Expr::Clamp {
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
                }])
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

    /// The three world-position lanes of a body, in `x, y, z` order.
    ///
    /// Only `Frame::World` and only the `pos` port: `StateView` carries `xpos` and `xquat`,
    /// and neither a quaternion nor a relative frame is three subtractable numbers. Both are
    /// refused naming what was asked for rather than approximated.
    fn body_pose_leaf(
        &mut self,
        body: es_core::StableId,
        relative_to: Frame,
        port: &str,
    ) -> Result<Vec<Expr>, EnvError> {
        let name = self
            .scene
            .bodies
            .iter()
            .find(|b| b.id == body)
            .map_or_else(|| body.to_string(), |b| b.name.clone());
        if relative_to != Frame::World {
            return Err(EnvError::Unsupported(format!(
                "GetBodyPose(\"{name}\") relative to {relative_to:?} in a reward or \
                 termination cone"
            )));
        }
        if port != "pos" {
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
        Ok((0..3)
            .map(|axis| {
                self.bind(
                    format!("xpos[{}]", row * 3 + axis),
                    Source::Xpos { row, axis },
                )
            })
            .collect())
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
                        Source::Sensor(_) | Source::Time { .. } | Source::Xpos { .. } => 0.0,
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

        // The orientation, another frame, an L1 norm and a body the model does not index are
        // all refused the same way -- by name.
        for (patch, wanted) in [
            (
                Box::new(|t: &mut TaskIr| {
                    t.graph.edges.retain(|e| e.from.node != NodeId(0));
                    t.graph.connect(NodeId(0), "quat", NodeId(2), "a");
                }) as Box<dyn Fn(&mut TaskIr)>,
                "quat",
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
}
