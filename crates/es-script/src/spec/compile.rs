//! [`compile_task`]: a [`TaskSpec`] and its scene to a Task IR, every node one `es-env`
//! lowers (`crates/es-env/src/plan.rs`, `randomize.rs`). Deterministic: the same document and
//! scene give the same bytes (node ids follow the document's order, collections are ordered).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use es_assets::scene::{Body, Joint, JointKind, SceneDesc, TendonKind};
use es_core::StableId;
use es_ir::graph::{IrNode, NodeId};
use es_ir::image::{
    CameraModel, ChannelFormat, ColorSpace, DistortionModel, ImageDType, ImageSpec, Intrinsics,
    ShutterModel,
};
use es_ir::task::{
    ActionSpace, Aggregation, ArithOp, CmpOp, Distribution, JointQuantity, LogicOp, MathFunc,
    NormKind, ObsChannel, ObsSource, ObservationSpec, SceneRef, SensorPath, SensorRender,
    TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind,
};
use es_ir::types::{Align, ElemType, Frame, PortType, Shape, TimeRef, Unit};
// A number written into a document has musl's bits, not the host's (spec 5.3, M17 G3d).
use es_math::approx;

use super::{
    refuse, Clause, Draw, Observe, Relation, RenderDoc, RenderPath, Shaping, SpecError, StartItem,
    TaskSpec,
};

/// `InverseAngle`'s knots: `s = √(8 (1 − d))` at which `1 / (s + 0.1)` is sampled; the last is
/// `√8`, `d = 0` (plan H, `crates/es/tests/shadow_hand.rs`).
const KNOTS: [f64; 9] = [
    0.0,
    0.025,
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    2.828_427_124_746_190_3,
];

/// Compiles `spec` against its scene, `root.join(spec.scene)`.
pub fn compile_task(spec: &TaskSpec, root: &Path) -> Result<TaskIr, SpecError> {
    let (scene, bytes) =
        load_scene(&root.join(&spec.scene)).map_err(|reason| SpecError::Scene {
            path: spec.scene.clone(),
            reason,
        })?;
    let steps = spec.timeout_s * spec.control_hz;
    if !(steps >= 1.0 && steps.fract() == 0.0 && steps <= f64::from(u32::MAX)) {
        return refuse(
            "task-spec",
            "timeout_s",
            format!("{steps} control steps at control_hz is not a whole number"),
        );
    }
    let robot = super::robot::root_body(spec, root, &scene)?;
    let mut c = Compiler::new(spec, &scene, &robot)?;
    if let Some(observe) = &spec.observe {
        c.observe(observe)?;
    }
    // ponytail: every actuator of the scene is the robot's (one robot per task); a scene with a
    // second actuated thing would need the actuators filtered by the robot's subtree.
    c.add(TaskNode::ActionSpec {
        space: ActionSpace::JointPosition,
        dim: scene.actuators.len() as u32,
        control_rate_hz: spec.control_hz as f32,
    });
    c.ends()?;
    if let Some(start) = &spec.start {
        for (i, item) in start.items.iter().enumerate() {
            let at = format!("start[{i}] ({})", item.what);
            c.start_item(&at, item, start.strength)?;
        }
    }
    let task = TaskIr {
        schema_version: es_ir::task::SCHEMA_VERSION,
        scene: SceneRef {
            path: spec.scene.clone(),
            scene_hash: scene.scene_hash(),
            asset_hash: *blake3::hash(&bytes).as_bytes(),
        },
        graph: c.graph,
        observation_spec: ObservationSpec {
            channels: c.channels,
        },
        config: TaskConfig {
            max_episode_steps: steps as u32,
            control_rate_hz: spec.control_hz as f32,
            deterministic: true,
            rng_streams: c.streams,
        },
        control: None,
    };
    let diags = task.validate();
    if !diags.is_empty() {
        return refuse("the compiled Task IR", "validate", format!("{diags:?}"));
    }
    Ok(task)
}

/// What `es_tools::backend::load_scene` does (a layer-11 sibling, so not callable here): the
/// scene by extension, then its mesh files relative to its directory. Also the file's bytes,
/// whose digest is `SceneRef.asset_hash`, as in every committed document.
pub(super) fn load_scene(path: &Path) -> Result<(SceneDesc, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let raw = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let is = |want: &str| {
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case(want))
    };
    let s = |e: &dyn std::fmt::Display| e.to_string();
    if is("esscene") {
        let doc = es_assets::esscene::EsScene::from_toml(raw).map_err(|e| s(&e))?;
        let scene = es_assets::esscene::expand(&doc, dir).map_err(|e| s(&e))?;
        return Ok((scene, bytes));
    }
    let mut scene = if is("urdf") {
        let resolver = es_assets::urdf::PackageResolver::from_env();
        es_assets::urdf::parse_urdf(raw, &resolver)
            .map_err(|e| s(&e))?
            .scene
    } else {
        es_assets::parse_mjcf(raw).map_err(|e| s(&e))?.scene
    };
    es_assets::mesh::load(&mut scene, dir).map_err(|e| s(&e))?;
    Ok((scene, bytes))
}

struct Compiler<'a> {
    spec: &'a TaskSpec,
    scene: &'a SceneDesc,
    root: &'a Body,
    /// The robot's joints, in scene order.
    joints: Vec<&'a Joint>,
    graph: TaskGraph,
    channels: BTreeMap<String, ObsChannel>,
    streams: BTreeSet<String>,
}

impl<'a> Compiler<'a> {
    fn new(spec: &'a TaskSpec, scene: &'a SceneDesc, robot: &str) -> Result<Self, SpecError> {
        let Some(root) = scene.bodies.iter().find(|b| b.name == robot) else {
            return refuse("task-spec", "robot", no("body", &spec.robot));
        };
        // Bodies come parent first (MJCF's tree order, G1's `parent` names one defined before).
        let mut tree = BTreeSet::from([root.id]);
        for b in &scene.bodies {
            if b.parent.is_some_and(|p| tree.contains(&p)) {
                tree.insert(b.id);
            }
        }
        let joints = scene
            .joints
            .iter()
            .filter(|j| tree.contains(&j.body))
            .collect();
        Ok(Self {
            spec,
            scene,
            root,
            joints,
            graph: TaskGraph::new(es_ir::task::SCHEMA_VERSION),
            channels: BTreeMap::new(),
            streams: BTreeSet::new(),
        })
    }

    // --- the graph ------------------------------------------------------------------------

    fn add(&mut self, node: TaskNode) -> NodeId {
        let id = NodeId(self.graph.nodes.len() as u32);
        self.graph.insert(id, node);
        id
    }

    /// A source node, shared: one `GetBodyPose` per body, one `GetTime`, ... however many
    /// clauses and channels read it (the committed documents' shape).
    fn source(&mut self, node: TaskNode) -> NodeId {
        match self.graph.nodes.iter().find(|(_, n)| **n == node) {
            Some((id, _)) => *id,
            None => self.add(node),
        }
    }

    fn out(&self, (id, port): (NodeId, &str)) -> PortType {
        self.graph.nodes[&id]
            .outputs()
            .into_iter()
            .find(|p| p.name == port)
            .map(|p| p.ty)
            .expect("the port exists")
    }

    /// `make(type of from)` fed by `from` at input `to_port`.
    fn feed(
        &mut self,
        from: (NodeId, &str),
        to_port: &str,
        make: impl FnOnce(PortType) -> TaskNode,
    ) -> NodeId {
        let id = self.add(make(self.out(from)));
        self.graph.connect(from.0, from.1, id, to_port);
        id
    }

    fn pair(&mut self, a: (NodeId, &str), b: (NodeId, &str), node: TaskNode) -> NodeId {
        let id = self.add(node);
        self.graph.connect(a.0, a.1, id, "a");
        self.graph.connect(b.0, b.1, id, "b");
        id
    }

    fn compare(&mut self, from: NodeId, op: CmpOp, rhs: f64) -> NodeId {
        self.feed((from, "value"), "a", |ty| TaskNode::Compare {
            op,
            rhs: Some(rhs),
            ty,
        })
    }

    fn logic(&mut self, op: LogicOp, a: NodeId, b: NodeId) -> NodeId {
        let node = TaskNode::Logic {
            op,
            shape: Shape::new([1]),
        };
        self.pair((a, "value"), (b, "value"), node)
    }

    fn normalize(
        &mut self,
        from: (NodeId, &str),
        lo: Vec<f64>,
        hi: Vec<f64>,
        out: [f64; 2],
    ) -> NodeId {
        self.feed(from, "value", |ty| TaskNode::Normalize {
            lo,
            hi,
            out_lo: out[0],
            out_hi: out[1],
            ty,
        })
    }

    fn reward(&mut self, from: NodeId, name: String, weight: f64) -> NodeId {
        self.feed((from, "value"), "value", |ty| TaskNode::Reward {
            name,
            weight,
            aggregation: Aggregation::Sum,
            ty,
        })
    }

    fn scale(&self) -> f64 {
        self.spec
            .reward
            .as_ref()
            .and_then(|r| r.scale)
            .unwrap_or(1.0)
    }

    // --- names ----------------------------------------------------------------------------

    fn body(&self, at: &str, field: &str, name: &str) -> Result<&'a Body, SpecError> {
        let scene = self.scene;
        match scene.bodies.iter().find(|b| b.name == name) {
            Some(b) => Ok(b),
            None => refuse(at, field, no("body", name)),
        }
    }

    fn free_joint(&self, at: &str, field: &str, body: &Body) -> Result<&'a Joint, SpecError> {
        let scene = self.scene;
        match scene
            .joints
            .iter()
            .find(|j| j.body == body.id && j.kind == JointKind::Free)
        {
            Some(j) => Ok(j),
            None => refuse(at, field, format!("body `{}` has no free joint", body.name)),
        }
    }

    /// Where `joint`'s coordinates start in `qpos`.
    fn qpos_of(&self, joint: &Joint) -> usize {
        self.scene
            .joints
            .iter()
            .take_while(|j| j.id != joint.id)
            .map(|j| match j.kind {
                JointKind::Free => 7,
                JointKind::Ball => 4,
                JointKind::Hinge | JointKind::Slide => 1,
                JointKind::Fixed => 0,
            })
            .sum()
    }

    /// `<body>.<field>` when `<body>` is a body of the scene.
    fn dotted(&self, what: &str) -> Option<(&'a Body, String)> {
        let (name, field) = what.rsplit_once('.')?;
        let scene = self.scene;
        let body = scene.bodies.iter().find(|b| b.name == name)?;
        Some((body, field.to_owned()))
    }

    /// A scalar subject: a joint as `GetJointState` reads it (a robot joint through the
    /// robot's root body, as the robot's joint vector; any other through itself), `<body>.x`
    /// of a free body — its free joint's first lane, G3a's form — or any other
    /// `<body>.x|y|z`, one lane of the body's world position or linear velocity
    /// ([`Self::coordinate`]).
    fn scalar(
        &mut self,
        at: &str,
        subject: &str,
        quantity: JointQuantity,
    ) -> Result<NodeId, SpecError> {
        let (owner, joint) = if let Some((body, field)) = self.dotted(subject) {
            let Some(axis) = ["x", "y", "z"].iter().position(|a| *a == field) else {
                return refuse(
                    at,
                    "subject",
                    format!("`{subject}`: a body's coordinate is x, y or z"),
                );
            };
            let scene = self.scene;
            match scene
                .joints
                .iter()
                .find(|j| j.body == body.id && j.kind == JointKind::Free)
            {
                Some(j) if axis == 0 => (j.id, j),
                _ => return self.coordinate(at, body, axis as u64, quantity),
            }
        } else {
            let scene = self.scene;
            let Some(j) = scene.joints.iter().find(|j| j.name == subject) else {
                return refuse(at, "subject", no("joint or `<body>.x|y|z`", subject));
            };
            if !matches!(j.kind, JointKind::Hinge | JointKind::Slide) {
                return refuse(
                    at,
                    "subject",
                    format!("joint `{subject}` is not a hinge or slide"),
                );
            }
            let robot = self.joints.iter().any(|r| r.id == j.id);
            (if robot { self.root.id } else { j.id }, j)
        };
        Ok(self.source(TaskNode::GetJointState {
            body: owner,
            joints: vec![joint.name.clone()],
            quantity,
        }))
    }

    fn pose(&mut self, body: StableId) -> NodeId {
        self.source(TaskNode::GetBodyPose {
            body,
            relative_to: Frame::World,
        })
    }

    fn velocity(&mut self, body: StableId) -> NodeId {
        self.source(TaskNode::GetBodyVelocity {
            body,
            relative_to: Frame::World,
        })
    }

    /// Lane `axis` of a body's world position (`GetBodyPose.pos`, any body) or of its linear
    /// velocity (`GetBodyVelocity.linear`, a free body: `es-env` reads it from the free
    /// joint's `qvel`), by `Slice`.
    fn coordinate(
        &mut self,
        at: &str,
        body: &Body,
        axis: u64,
        quantity: JointQuantity,
    ) -> Result<NodeId, SpecError> {
        let from = if quantity == JointQuantity::Velocity {
            self.free_joint(at, "subject", body)?;
            (self.velocity(body.id), "linear")
        } else {
            (self.pose(body.id), "pos")
        };
        Ok(self.feed(from, "value", |ty| TaskNode::Slice {
            ty,
            axis: 0,
            start: axis,
            len: 1,
        }))
    }

    /// `|value| < bound`, the L2 norm of a vector.
    fn norm_below(&mut self, from: (NodeId, &str), bound: f64) -> NodeId {
        let norm = self.feed(from, "value", |ty| TaskNode::Norm {
            kind: NormKind::L2,
            ty,
        });
        self.compare(norm, CmpOp::Lt, bound)
    }

    /// Region (site) `name` as `(lo, hi)` per world axis: the site's world position ± its
    /// half-extents (`size`). The box is axis-aligned in the **world** frame: a cone takes a
    /// world point only as `Compare`'s literal (the IR has no constant node to rotate a vector
    /// by), and a region the editor writes is unrotated — so a rotated site, or one that moves
    /// (a joint on its body or on an ancestor), is refused by name.
    fn region(&self, at: &str, name: &str) -> Result<([f64; 3], [f64; 3]), SpecError> {
        let scene = self.scene;
        let Some((mut body, site)) = scene
            .bodies
            .iter()
            .find_map(|b| b.sites.iter().find(|s| s.name == name).map(|s| (b, s)))
        else {
            return refuse(at, "object", no("region (site)", name));
        };
        let mut chain = vec![site.pose];
        loop {
            if let Some(j) = scene.joints.iter().find(|j| j.body == body.id) {
                return refuse(
                    at,
                    "object",
                    format!("region `{name}` moves with joint `{}`", j.name),
                );
            }
            chain.push(body.pose);
            let Some(parent) = body.parent else { break };
            let Some(up) = scene.bodies.iter().find(|b| b.id == parent) else {
                let why = format!("body `{}` names a parent the scene lacks", body.name);
                return refuse(at, "object", why);
            };
            body = up;
        }
        if chain
            .iter()
            .any(|p| p.orientation != es_math::Quat::IDENTITY)
        {
            return refuse(
                at,
                "object",
                format!("region `{name}` is rotated; a region is a box along the world axes"),
            );
        }
        // From the world down, as the frames compose.
        let c = chain.iter().rev().fold([0.0; 3], |c, p| {
            [
                c[0] + p.position.x,
                c[1] + p.position.y,
                c[2] + p.position.z,
            ]
        });
        let h = [site.size.x, site.size.y, site.size.z];
        Ok((
            std::array::from_fn(|i| c[i] - h[i]),
            std::array::from_fn(|i| c[i] + h[i]),
        ))
    }

    // --- clauses --------------------------------------------------------------------------

    /// The success, failure and timeout terminations and the sparse terms.
    fn ends(&mut self) -> Result<(), SpecError> {
        let spec = self.spec;
        let scale = self.scale();
        let reward = spec.reward.clone().unwrap_or_default();
        let groups = [
            (
                "success",
                Some(&spec.success),
                LogicOp::And,
                TerminationKind::Success,
                reward.success,
            ),
            (
                "failure",
                spec.failure.as_ref(),
                LogicOp::Or,
                TerminationKind::Failure,
                reward.failure,
            ),
        ];
        for (section, clauses, op, kind, bonus) in groups {
            let Some(clauses) = clauses else { continue };
            if clauses.clauses.is_empty() {
                if section == "success" {
                    return refuse(
                        section,
                        "clauses",
                        "a task needs at least one success clause",
                    );
                }
                continue;
            }
            let mut acc = None;
            for (i, c) in clauses.clauses.iter().enumerate() {
                let at = format!("{section}[{i}] ({} {})", c.subject, c.relation.name());
                let p = self.clause(&at, c)?;
                acc = Some(acc.map_or(p, |a| self.logic(op, a, p)));
            }
            let acc = acc.expect("non-empty");
            let stop = self.add(TaskNode::Terminate { kind });
            self.graph.connect(acc, "value", stop, "value");
            if let Some(w) = bonus {
                self.reward(acc, section.to_owned(), w * scale);
            }
        }
        let time = self.source(TaskNode::GetTime { since_reset: true });
        let late = self.compare(time, CmpOp::Ge, spec.timeout_s);
        let stop = self.add(TaskNode::Terminate {
            kind: TerminationKind::Timeout,
        });
        self.graph.connect(late, "value", stop, "value");
        Ok(())
    }

    /// One clause's predicate (a `[1]` bool) and its shaping terms.
    fn clause(&mut self, at: &str, c: &Clause) -> Result<NodeId, SpecError> {
        if c.relation == Relation::Touches {
            return refuse(
                at,
                "relation",
                "`touches` waits for GetContact lowering in es-env",
            );
        }
        check_fields(at, c)?;
        let scale = self.scale();
        let weight = c.weight.unwrap_or_default() * scale;
        let term = |kind: &str| {
            c.term
                .clone()
                .unwrap_or_else(|| format!("{}_{kind}", c.subject))
        };
        match c.relation {
            // A body inside a region: each world axis of its position strictly inside the box.
            Relation::Inside if c.object.is_some() => {
                let (lo, hi) = self.region(at, c.object.as_deref().expect("matched"))?;
                let body = self.body(at, "subject", &c.subject)?;
                let mut acc = None;
                for axis in 0..3 {
                    let lane = self.coordinate(at, body, axis as u64, JointQuantity::Position)?;
                    let (gt, lt) = (
                        self.compare(lane, CmpOp::Gt, lo[axis]),
                        self.compare(lane, CmpOp::Lt, hi[axis]),
                    );
                    let within = self.logic(LogicOp::And, gt, lt);
                    acc = Some(acc.map_or(within, |prev| self.logic(LogicOp::And, prev, within)));
                }
                Ok(acc.expect("three axes"))
            }
            // A body above (below) another: the difference of their world heights beyond `m`.
            Relation::Above | Relation::Below if c.object.is_some() => {
                let subject = self.body(at, "subject", &c.subject)?;
                let other = self.body(at, "object", c.object.as_deref().expect("matched"))?;
                let z_s = self.coordinate(at, subject, 2, JointQuantity::Position)?;
                let z_o = self.coordinate(at, other, 2, JointQuantity::Position)?;
                let (hi, lo) = if c.relation == Relation::Above {
                    (z_s, z_o)
                } else {
                    (z_o, z_s)
                };
                let ty = self.out((hi, "value"));
                let node = TaskNode::Arith {
                    op: ArithOp::Sub,
                    ty,
                };
                let gap = self.pair((hi, "value"), (lo, "value"), node);
                Ok(self.compare(gap, CmpOp::Gt, c.m.unwrap_or(0.0)))
            }
            // A body (not a joint, not `<body>.<axis>`): its speed `‖v‖`, and its angular rate
            // when `angular` bounds it.
            Relation::Still
                if self.dotted(&c.subject).is_none()
                    && !self.scene.joints.iter().any(|j| j.name == c.subject) =>
            {
                let body = self.body(at, "subject", &c.subject)?;
                self.free_joint(at, "subject", body)?;
                let vel = self.velocity(body.id);
                let slow = self.norm_below((vel, "linear"), c.speed.expect("checked"));
                Ok(match c.angular {
                    Some(rate) => {
                        let calm = self.norm_below((vel, "angular"), rate);
                        self.logic(LogicOp::And, slow, calm)
                    }
                    None => slow,
                })
            }
            Relation::Inside => {
                let [lo, hi] = c.range.expect("checked");
                let x = self.scalar(at, &c.subject, JointQuantity::Position)?;
                let (a, b) = (
                    self.compare(x, CmpOp::Gt, lo),
                    self.compare(x, CmpOp::Lt, hi),
                );
                if let Some([r0, r1]) = c.ramp {
                    let ramp = self.normalize((x, "value"), vec![r0], vec![r1], [0.0, 1.0]);
                    self.reward(ramp, term("ramp"), weight);
                }
                Ok(self.logic(LogicOp::And, a, b))
            }
            Relation::Above | Relation::Below => {
                let x = self.scalar(at, &c.subject, JointQuantity::Position)?;
                let op = if c.relation == Relation::Above {
                    CmpOp::Gt
                } else {
                    CmpOp::Lt
                };
                Ok(self.compare(x, op, c.value.expect("checked")))
            }
            Relation::Still => {
                if c.angular.is_some() {
                    return refuse(
                        at,
                        "angular",
                        "a body's bound; the subject is one coordinate",
                    );
                }
                let v = self.scalar(at, &c.subject, JointQuantity::Velocity)?;
                let speed = c.speed.expect("checked");
                let (a, b) = (
                    self.compare(v, CmpOp::Gt, -speed),
                    self.compare(v, CmpOp::Lt, speed),
                );
                Ok(self.logic(LogicOp::And, a, b))
            }
            Relation::Near | Relation::FartherThan => {
                let m = c.m.expect("checked");
                let s = self.body(at, "subject", &c.subject)?.id;
                let s = self.pose(s);
                let offset = if let Some(o) = &c.object {
                    let o = self.body(at, "object", o)?.id;
                    let o = self.pose(o);
                    let ty = self.out((s, "pos"));
                    let node = TaskNode::Arith {
                        op: ArithOp::Sub,
                        ty,
                    };
                    self.pair((s, "pos"), (o, "pos"), node)
                } else {
                    // No constant node: `p − point` is a per-lane `Normalize` whose map is the
                    // identity shifted by the point, `[p − 1, p + 1] → [−1, 1]` (plan H).
                    if m >= 1.0 {
                        return refuse(at, "m", "against a point, offsets clamp at 1 m per axis");
                    }
                    let p = c.point.expect("checked");
                    let lo = p.iter().map(|v| v - 1.0).collect();
                    let hi = p.iter().map(|v| v + 1.0).collect();
                    self.normalize((s, "pos"), lo, hi, [-1.0, 1.0])
                };
                let dist = self.feed((offset, "value"), "value", |ty| TaskNode::Norm {
                    kind: NormKind::L2,
                    ty,
                });
                if c.shaping.is_some() {
                    let metres = self.normalize((dist, "value"), vec![0.0], vec![1.0], [0.0, 1.0]);
                    self.reward(metres, term("distance"), weight);
                }
                let op = if c.relation == Relation::Near {
                    CmpOp::Lt
                } else {
                    CmpOp::Gt
                };
                Ok(self.compare(dist, op, m))
            }
            Relation::OrientationMatches => {
                let s = self.body(at, "subject", &c.subject)?.id;
                let o = self
                    .body(at, "object", c.object.as_deref().expect("checked"))?
                    .id;
                let (s, o) = (self.pose(s), self.pose(o));
                let ty = self.out((s, "quat"));
                let dot = self.pair((s, "quat"), (o, "quat"), TaskNode::Dot { ty });
                let d = self.math(dot, MathFunc::Abs);
                // `|q·g| ≥ cos(theta/2)`: the rotation between them is at most theta.
                let half = c.within_deg.expect("checked").to_radians() / 2.0;
                let reached = self.compare(d, CmpOp::Ge, approx::sin_cos_f64(half).1);
                if c.shaping.is_some() {
                    self.inverse_angle(d, &term("inverse_angle"), weight);
                }
                Ok(reached)
            }
            Relation::Touches => unreachable!("refused above"),
        }
    }

    fn math(&mut self, from: NodeId, func: MathFunc) -> NodeId {
        self.feed((from, "value"), "value", |ty| TaskNode::MathFn {
            func,
            approx: false,
            ty,
        })
    }

    /// `weight / (s + 0.1)` as one clamped ramp per knot interval (a `Reward` cannot take a
    /// reciprocal: `Arith Div` needs a dimensionless divisor), exact at every knot, plus the
    /// floor `1 / (√8 + 0.1)` paid every step (`time since reset ≥ 0`).
    fn inverse_angle(&mut self, d: NodeId, term: &str, weight: f64) {
        let gap = self.normalize((d, "value"), vec![1.0], vec![0.0], [0.0, 8.0]);
        let s = self.math(gap, MathFunc::Sqrt);
        let f: Vec<f64> = KNOTS.iter().map(|s| 1.0 / (s + 0.1)).collect();
        for k in 0..KNOTS.len() - 1 {
            // 1 at s ≤ s_k, 0 at s ≥ s_{k+1}.
            let ramp = self.normalize((s, "value"), vec![KNOTS[k + 1]], vec![KNOTS[k]], [0.0, 1.0]);
            self.reward(ramp, format!("{term}_{k}"), weight * (f[k] - f[k + 1]));
        }
        let time = self.source(TaskNode::GetTime { since_reset: true });
        let always = self.compare(time, CmpOp::Ge, 0.0);
        self.reward(always, format!("{term}_floor"), weight * f[KNOTS.len() - 1]);
    }

    // --- the reset ------------------------------------------------------------------------

    fn reset(&mut self, target: String, dist: Distribution, stream: String, dice: bool) {
        self.add(if dice {
            TaskNode::Randomization {
                target,
                dist,
                stream: stream.clone(),
            }
        } else {
            TaskNode::ResetState {
                target,
                dist,
                stream: stream.clone(),
            }
        });
        self.streams.insert(stream);
    }

    fn start_item(
        &mut self,
        at: &str,
        item: &StartItem,
        strength: Option<f64>,
    ) -> Result<(), SpecError> {
        let dice = item.dice == Some(true);
        let k = strength.filter(|_| dice);
        let present = [
            ("value", item.value.is_some()),
            ("noise", item.noise.is_some()),
            ("range", item.range.is_some()),
            ("draw", item.draw.is_some()),
            ("tilt", item.tilt.is_some()),
            ("tilt_max_deg", item.tilt_max_deg.is_some()),
            ("coupled", item.coupled.is_some()),
            ("stream", item.stream.is_some()),
        ];
        let only = |allowed: &[&str]| match present.iter().find(|(f, p)| *p && !allowed.contains(f))
        {
            Some((f, _)) => refuse(at, *f, format!("does not apply to `{}`", item.what)),
            None => Ok(()),
        };
        if item.what == "robot.joints" {
            only(&["noise", "coupled"])?;
            let Some(noise) = item.noise else {
                return refuse(at, "noise", "required: the fraction of each joint's range");
            };
            let noise = k.map_or(noise, |k| noise * k);
            let coupled = if item.coupled == Some(true) {
                self.couplings(at)?
            } else {
                Vec::new()
            };
            for j in self.joints.clone() {
                let range = |j: &Joint| match j.range {
                    Some(r) => Ok(r),
                    None => refuse(at, "noise", format!("joint `{}` has no range", j.name)),
                };
                let (stream, (lo, hi)) = match coupled.iter().find(|(_, j0, _)| j0.id == j.id) {
                    Some((j1, _, c)) => {
                        let (lo, hi) = range(j1)?;
                        (*j1, (lo * c, hi * c))
                    }
                    None => (j, range(j)?),
                };
                let dist = between(noise * lo, noise * hi);
                self.reset(
                    format!("joint.{}.qpos", j.name),
                    dist,
                    format!("reset.{}", stream.name),
                    dice,
                );
            }
            return Ok(());
        }
        if let Some((body, field)) = self.dotted(&item.what) {
            let j = self.free_joint(at, "what", body)?;
            let q = self.qpos_of(j);
            let stream = item.stream.clone().unwrap_or_else(|| item.what.clone());
            if field == "orientation" {
                // The quaternion `(w, x, y, z)` from `qpos[q + 3]`, one draw per lane; lanes
                // on one stream share it (`es_env::rng` addresses a draw by its stream).
                let lanes: Vec<(Distribution, String)> = match item.draw {
                    Some(Draw::Yaw) => {
                        only(&["draw", "tilt", "stream"])?;
                        let t = item.tilt.unwrap_or(0.0);
                        // `(1, t, −t·u, u)`, each lane linear in one `u ~ U(−1, 1)`.
                        [(1.0, 1.0), (t, t), (t, -t), (-1.0, 1.0)]
                            .into_iter()
                            .map(|(lo, hi)| (between(lo, hi), stream.clone()))
                            .collect()
                    }
                    Some(Draw::Tilt) => {
                        only(&["draw", "tilt_max_deg", "stream"])?;
                        let theta = item.tilt_max_deg.unwrap_or(f64::NAN);
                        if !(theta > 0.0 && theta < 180.0) {
                            return refuse(
                                at,
                                "tilt_max_deg",
                                "required, in (0, 180); 180 is `any`",
                            );
                        }
                        let m =
                            approx::tan_f64(theta.to_radians() / 2.0) / std::f64::consts::SQRT_2;
                        let ab = Distribution::Uniform { lo: -m, hi: m };
                        vec![
                            (Distribution::Constant(1.0), stream.clone()),
                            (ab.clone(), format!("{stream}.x")),
                            (ab, format!("{stream}.y")),
                            (
                                Distribution::Normal {
                                    mean: 0.0,
                                    std: 1.0,
                                },
                                stream.clone(),
                            ),
                        ]
                    }
                    Some(Draw::Any) => {
                        only(&["draw", "stream"])?;
                        ["w", "x", "y", "z"]
                            .iter()
                            .map(|l| {
                                (
                                    Distribution::Normal {
                                        mean: 0.0,
                                        std: 1.0,
                                    },
                                    format!("{stream}.{l}"),
                                )
                            })
                            .collect()
                    }
                    None => return refuse(at, "draw", "required: `yaw`, `tilt` or `any`"),
                };
                for (lane, (dist, stream)) in lanes.into_iter().enumerate() {
                    self.reset(format!("qpos[{}]", q + 3 + lane), dist, stream, dice);
                }
                return Ok(());
            }
            let Some(axis) = ["x", "y", "z"].iter().position(|a| *a == field) else {
                return refuse(
                    at,
                    "what",
                    format!("`{field}`: a body places by x, y, z or orientation"),
                );
            };
            only(&["value", "noise", "range", "stream"])?;
            let dist = value_or_range(at, item, k)?;
            self.reset(format!("qpos[{}]", q + axis), dist, stream, dice);
            return Ok(());
        }
        let scene = self.scene;
        let Some(j) = scene.joints.iter().find(|j| j.name == item.what) else {
            return refuse(
                at,
                "what",
                no("`robot.joints`, joint or `<body>.x|y|z|yaw`", &item.what),
            );
        };
        only(&["value", "noise", "range", "stream"])?;
        let dist = value_or_range(at, item, k)?;
        let stream = item
            .stream
            .clone()
            .unwrap_or_else(|| format!("reset.{}", j.name));
        self.reset(format!("joint.{}.qpos", j.name), dist, stream, dice);
        Ok(())
    }

    /// `(J1, J0, k)` for every two-joint fixed tendon over the robot's joints: `J0 = k · J1`
    /// holds the tendon `c1·J1 + c0·J0` at zero, `k = −c1 / c0`.
    fn couplings(&self, at: &str) -> Result<Vec<(&'a Joint, &'a Joint, f64)>, SpecError> {
        let robot = |id: StableId| self.joints.iter().copied().find(|j| j.id == id);
        let mut out = Vec::new();
        for t in &self.scene.tendons {
            let TendonKind::Fixed { joints } = &t.kind else {
                continue;
            };
            if !joints.iter().all(|(id, _)| robot(*id).is_some()) {
                continue;
            }
            let [(j1, c1), (j0, c0)] = joints[..] else {
                return refuse(
                    at,
                    "coupled",
                    format!(
                        "tendon `{}` couples {} joints, not two",
                        t.name,
                        joints.len()
                    ),
                );
            };
            out.push((
                robot(j1).expect("checked"),
                robot(j0).expect("checked"),
                -c1 / c0,
            ));
        }
        Ok(out)
    }

    // --- observations ---------------------------------------------------------------------

    fn channel(
        &mut self,
        at: &str,
        name: &str,
        source: ObsSource,
        ty: PortType,
    ) -> Result<(), SpecError> {
        if self.channels.contains_key(name) {
            return refuse(at, name, "a second channel of that name");
        }
        self.channels
            .insert(name.to_owned(), ObsChannel { source, ty });
        Ok(())
    }

    fn bind(&mut self, from: (NodeId, &str), channel: &str) -> PortType {
        let ty = self.out(from);
        let channel = channel.to_owned();
        self.feed(from, "value", |ty| TaskNode::ObservationSpec {
            channel,
            ty,
        });
        ty
    }

    fn observe(&mut self, o: &Observe) -> Result<(), SpecError> {
        let hz = self.spec.control_hz as f32;
        let render = render(o.render.as_ref())?;
        let cameras = o.cameras.clone().unwrap_or_default();
        if !cameras.is_empty() && o.camera_px.is_none() {
            return refuse("observe", "camera_px", "required with cameras");
        }
        for name in cameras {
            let scene = self.scene;
            let Some(cam) = scene.cameras.iter().find(|c| c.name == name) else {
                return refuse("observe", "cameras", no("camera", &name));
            };
            let ty = camera_ty(cam.id, cam.fovy, o.camera_px.expect("checked"), hz);
            let get = self.add(TaskNode::GetSensor {
                sensor: cam.id,
                ty: ty.clone(),
            });
            let channel = format!("rgb_{name}");
            self.bind((get, "value"), &channel);
            let source = ObsSource::Sensor {
                id: cam.id,
                format: ChannelFormat::Rgb,
                render,
            };
            self.channel("observe.cameras", &channel, source, ty)?;
        }
        for (table, map) in [("state", &o.state), ("privileged", &o.privileged)] {
            for (name, what) in map.iter().flat_map(|m| m.iter()) {
                self.state_channel(&format!("observe.{table}.{name}"), name, what)?;
            }
        }
        Ok(())
    }

    fn state_channel(&mut self, at: &str, name: &str, what: &str) -> Result<(), SpecError> {
        let n = self.joints.len();
        let names: Vec<String> = self.joints.iter().map(|j| j.name.clone()).collect();
        let root = self.root.id;
        let joints = |quantity| TaskNode::GetJointState {
            body: root,
            joints: names.clone(),
            quantity,
        };
        match what {
            "robot.joint_pos" | "robot.joint_vel" => {
                // A `JointState` channel naming a body reads the leading `dof` of the row.
                let leading = self.scene.joints.iter().take(n).map(|j| j.id);
                if !leading.eq(self.joints.iter().map(|j| j.id)) || n == 0 {
                    return refuse(
                        at,
                        name,
                        "the robot's joints are not the scene's leading ones",
                    );
                }
                let (quantity, body) = if what == "robot.joint_pos" {
                    (JointQuantity::Position, self.root.id)
                } else {
                    // The root body is the position channel's id (one input buffer per id), so
                    // the velocity channel names the first joint, whose dofs it reads from.
                    (JointQuantity::Velocity, self.joints[0].id)
                };
                let get = self.source(joints(quantity));
                let ty = self.bind((get, "value"), name);
                let source = ObsSource::JointState {
                    body,
                    dof: n as u32,
                    quantity,
                };
                self.channel(at, name, source, ty)
            }
            "robot.previous_action" => {
                let mut initial = Vec::new();
                for a in &self.scene.actuators {
                    let Some((lo, hi)) = a.ctrl_range else {
                        return refuse(at, name, format!("actuator `{}` has no ctrlrange", a.name));
                    };
                    // The centre: Isaac Lab's zero raw action, in actuator units. Written as the
                    // generator it replaces wrote it, so the bits are the committed ones.
                    #[allow(clippy::manual_midpoint)]
                    initial.push((lo + hi) / 2.0);
                }
                let ty = f32v(initial.len() as u64, Unit::Angle, Frame::World);
                let source = ObsSource::PreviousAction {
                    initial: Some(initial),
                };
                self.channel(at, name, source, ty)
            }
            _ => {
                let Some((body, field)) = self.dotted(what) else {
                    return refuse(at, name, no("source", what));
                };
                let (source, get, ports) = match field.as_str() {
                    "pose" => (
                        ObsSource::BodyPose(body.id),
                        self.pose(body.id),
                        ["pos", "quat"],
                    ),
                    "qpos" => {
                        let j = self.free_joint(at, name, body)?;
                        let source = ObsSource::JointState {
                            body: j.id,
                            dof: 7,
                            quantity: JointQuantity::Position,
                        };
                        (source, self.pose(body.id), ["pos", "quat"])
                    }
                    "vel" => {
                        let j = self.free_joint(at, name, body)?;
                        let source = ObsSource::JointState {
                            body: j.id,
                            dof: 6,
                            quantity: JointQuantity::Velocity,
                        };
                        let get = self.source(TaskNode::GetBodyVelocity {
                            body: body.id,
                            relative_to: Frame::World,
                        });
                        (source, get, ["linear", "angular"])
                    }
                    _ => return refuse(at, name, no("source", what)),
                };
                let parts = vec![self.out((get, ports[0])), self.out((get, ports[1]))];
                let cat = self.add(TaskNode::Concat { parts, axis: 0 });
                self.graph.connect(get, ports[0], cat, "in0");
                self.graph.connect(get, ports[1], cat, "in1");
                let ty = self.bind((cat, "value"), name);
                self.channel(at, name, source, ty)
            }
        }
    }
}

fn no(what: &str, name: &str) -> String {
    format!("no {what} named `{name}` in the scene")
}

/// A uniform draw, or the constant when the bounds are the same bits.
fn between(lo: f64, hi: f64) -> Distribution {
    if lo.to_bits() == hi.to_bits() {
        Distribution::Constant(lo)
    } else {
        Distribution::Uniform { lo, hi }
    }
}

/// `value` (± `noise`) or `range`; `k` scales a 🎲 item's spread.
fn value_or_range(at: &str, item: &StartItem, k: Option<f64>) -> Result<Distribution, SpecError> {
    match (item.value, item.range) {
        (Some(v), None) => {
            let n = item.noise.unwrap_or(0.0) * k.unwrap_or(1.0);
            Ok(between(v - n, v + n))
        }
        (None, Some([lo, hi])) if item.noise.is_none() => Ok(match k {
            Some(k) => {
                let (c, h) = (f64::midpoint(lo, hi), (hi - lo) / 2.0);
                Distribution::Uniform {
                    lo: c - k * h,
                    hi: c + k * h,
                }
            }
            None => Distribution::Uniform { lo, hi },
        }),
        (None, Some(_)) => refuse(at, "noise", "a `range` is drawn as written"),
        (Some(_), Some(_)) => refuse(at, "range", "either `value` or `range`"),
        (None, None) => refuse(at, "value", "required: `value` or `range`"),
    }
}

/// Which fields a relation takes, and which it needs.
fn check_fields(at: &str, c: &Clause) -> Result<(), SpecError> {
    let (fields, needs, shaping): (&[&str], &[&str], Option<Shaping>) = match c.relation {
        Relation::Inside if c.object.is_some() => (&["object"], &[], None),
        Relation::Inside => (&["range"], &["range"], Some(Shaping::Ramp)),
        Relation::Above | Relation::Below if c.object.is_some() => (&["object", "m"], &[], None),
        Relation::Above | Relation::Below => (&["value"], &["value"], None),
        Relation::Still => (&["speed", "angular"], &["speed"], None),
        Relation::Near | Relation::FartherThan => {
            (&["object", "point", "m"], &["m"], Some(Shaping::Distance))
        }
        Relation::OrientationMatches => (
            &["object", "within_deg"],
            &["object", "within_deg"],
            Some(Shaping::InverseAngle),
        ),
        Relation::Touches => (&[], &[], None),
    };
    let present = [
        ("object", c.object.is_some()),
        ("point", c.point.is_some()),
        ("range", c.range.is_some()),
        ("value", c.value.is_some()),
        ("m", c.m.is_some()),
        ("speed", c.speed.is_some()),
        ("angular", c.angular.is_some()),
        ("within_deg", c.within_deg.is_some()),
        ("shaping", c.shaping.is_some()),
        ("weight", c.weight.is_some()),
        ("ramp", c.ramp.is_some()),
        ("term", c.term.is_some()),
    ];
    let has = |f: &str| present.iter().any(|(n, p)| *p && *n == f);
    // Where `inside`, `above` / `below` and `near` / `farther_than` measure from: one of two.
    let either = match c.relation {
        Relation::Inside => Some("range"),
        Relation::Above | Relation::Below => Some("value"),
        Relation::Near | Relation::FartherThan => Some("point"),
        _ => None,
    };
    if let Some(alt) = either.filter(|alt| has("object") && has(alt)) {
        return refuse(at, "object", format!("either `object` or `{alt}`"));
    }
    let shaped: &[&str] = match (c.shaping, shaping) {
        (None, _) => &[],
        (Some(s), Some(t)) if s == t && s == Shaping::Ramp => {
            &["shaping", "weight", "term", "ramp"]
        }
        (Some(s), Some(t)) if s == t => &["shaping", "weight", "term"],
        (Some(_), t) => {
            let want = match t {
                Some(Shaping::Distance) => "`distance`",
                Some(Shaping::Ramp) => "`ramp`",
                Some(Shaping::InverseAngle) => "`inverse_angle`",
                None => "no shaping",
            };
            return refuse(
                at,
                "shaping",
                format!("`{}` takes {want}", c.relation.name()),
            );
        }
    };
    if let Some((f, _)) = present
        .iter()
        .find(|(f, p)| *p && !fields.contains(f) && !shaped.contains(f))
    {
        let why = if ["weight", "ramp", "term"].contains(f) {
            "needs `shaping`".to_owned()
        } else {
            format!("does not apply to `{}`", c.relation.name())
        };
        return refuse(at, *f, why);
    }
    for f in needs {
        if !has(f) {
            return refuse(at, *f, "required");
        }
    }
    if c.shaping.is_some() && !has("weight") {
        return refuse(at, "weight", "required with `shaping`");
    }
    if c.shaping == Some(Shaping::Ramp) && !has("ramp") {
        return refuse(at, "ramp", "required with `shaping = \"ramp\"`");
    }
    if matches!(c.relation, Relation::Near | Relation::FartherThan) && has("object") == has("point")
    {
        return refuse(at, "object", "either `object` or `point`");
    }
    Ok(())
}

fn render(doc: Option<&RenderDoc>) -> Result<SensorRender, SpecError> {
    let Some(r) = doc else {
        return Ok(SensorRender::default());
    };
    let path = match (r.path, r.spp, r.bounces) {
        (Some(RenderPath::Pt), Some(spp), Some(bounces)) => SensorPath::Pt { spp, bounces },
        (Some(RenderPath::Pt), None, _) => {
            return refuse("observe.render", "spp", "required on `pt`")
        }
        (Some(RenderPath::Pt), _, None) => {
            return refuse("observe.render", "bounces", "required on `pt`")
        }
        (_, Some(_), _) => return refuse("observe.render", "spp", "`pt` only"),
        (_, _, Some(_)) => return refuse("observe.render", "bounces", "`pt` only"),
        _ => SensorPath::Rs,
    };
    let d = SensorRender::default();
    Ok(SensorRender {
        path,
        exposure: r.exposure.unwrap_or(d.exposure),
        tonemap: r.tonemap.unwrap_or(d.tonemap),
        seed: r.seed.unwrap_or(d.seed),
        svgf: r.svgf.unwrap_or(d.svgf),
    })
}

fn f32v(n: u64, unit: Unit, frame: Frame) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new([n]),
        unit,
        frame,
        time: TimeRef::Tick,
        image: None,
    }
}

/// A `px`×`px` RGB8 pinhole camera of vertical field of view `fovy` (rad), as the renderer
/// delivers it (`es_env::render::image_spec`).
fn camera_ty(camera: StableId, fovy: f64, px: u32, hz: f32) -> PortType {
    let f = f64::from(px) / 2.0 / approx::tan_f64(fovy / 2.0);
    let c = f64::from(px) / 2.0;
    PortType {
        elem: ElemType::U8,
        shape: Shape::new([u64::from(px), u64::from(px), 3]),
        unit: Unit::Pixel,
        frame: Frame::Camera(camera),
        time: TimeRef::Sensor {
            id: camera,
            align: Align::Hold,
        },
        image: Some(ImageSpec {
            width: px,
            height: px,
            channels: ChannelFormat::Rgb,
            dtype: ImageDType::U8,
            color_space: ColorSpace::SRgb,
            camera_model: CameraModel::Pinhole,
            intrinsics: Intrinsics::new(f, f, c, c),
            extrinsics: es_math::Pose::IDENTITY,
            distortion: DistortionModel::None,
            shutter: ShutterModel::Global,
            exposure: Duration::from_millis(2),
            rate_hz: hz,
            depth_scale: None,
        }),
    }
}
