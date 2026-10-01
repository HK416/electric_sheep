//! [`compile_task`]: a [`TaskSpec`] and its scene to a Task IR, every node one `es-env`
//! lowers (`crates/es-env/src/plan.rs`, `randomize.rs`). Deterministic: the same document and
//! scene give the same bytes (node ids follow the document's order, collections are ordered).
//!
//! Layout: the entry points, the [`Compiler`], its graph primitives and its scene lookups are
//! here; each section of the document compiles in its own module: `compile/clauses.rs`
//! (success, failure, timeout and shaping), `compile/start.rs` (the reset) and
//! `compile/observe.rs` (the observation channels).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use es_assets::scene::{Body, Joint, JointKind, SceneDesc};
use es_core::StableId;
use es_ir::graph::{IrNode, NodeId};
use es_ir::task::{
    ActionSpace, Aggregation, CmpOp, JointQuantity, LogicOp, NormKind, ObsChannel, ObservationSpec,
    SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind,
};
use es_ir::types::{Frame, PortType, Shape};

use super::{refuse, SpecError, TaskSpec};

mod clauses;
mod observe;
mod start;

/// Compiles `spec` against its scene, `root.join(spec.scene)`.
pub fn compile_task(spec: &TaskSpec, root: &Path) -> Result<TaskIr, SpecError> {
    compile_clauses(spec, root).map(|(task, _)| task)
}

/// Where one clause's predicate ends: its section (`Success` or `Failure`), its index there and
/// the node, whose `value` is the clause's truth before the fold into `Terminate`.
pub type ClauseNode = (TerminationKind, usize, NodeId);

/// [`compile_task`], and the node each clause's predicate ends in, every clause in document order
/// (design note `scene-authoring.md` section 4.8): a side table, not IR. The fold cannot be undone
/// from the Task IR alone, since a clause can itself be an `And` tree (inside a region).
pub fn compile_clauses(
    spec: &TaskSpec,
    root: &Path,
) -> Result<(TaskIr, Vec<ClauseNode>), SpecError> {
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
    let zero_unset = spec.start.as_ref().and_then(|s| s.zero_unset) == Some(true);
    if !zero_unset {
        c.scene_poses();
    }
    let clauses = std::mem::take(&mut c.clauses);
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
    Ok((task, clauses))
}

/// What `es_tools::backend::load_scene` does (a layer-11 sibling, so not callable here): the
/// scene by extension, then its mesh files relative to its directory. Also the file's bytes,
/// whose digest is `SceneRef.asset_hash`, as in every committed document.
pub fn load_scene(path: &Path) -> Result<(SceneDesc, Vec<u8>), String> {
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
    /// The `qpos` lanes a start item writes.
    placed: BTreeSet<usize>,
    /// Each clause's predicate node ([`compile_clauses`]).
    clauses: Vec<ClauseNode>,
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
            placed: BTreeSet::new(),
            clauses: Vec::new(),
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

    /// The L2 norm of a vector.
    fn norm(&mut self, from: (NodeId, &str)) -> NodeId {
        self.feed(from, "value", |ty| TaskNode::Norm {
            kind: NormKind::L2,
            ty,
        })
    }

    /// `|value| < bound`, the L2 norm of a vector.
    fn norm_below(&mut self, from: (NodeId, &str), bound: f64) -> NodeId {
        let norm = self.norm(from);
        self.compare(norm, CmpOp::Lt, bound)
    }

    /// `p − point` with no constant node: a per-lane `Normalize` whose map is the identity
    /// shifted by the point, `[p − 1, p + 1] → [−1, 1]` (plan H), so each lane clamps at 1 m.
    fn minus_point(&mut self, from: (NodeId, &str), point: [f64; 3]) -> NodeId {
        let lo = point.iter().map(|v| v - 1.0).collect();
        let hi = point.iter().map(|v| v + 1.0).collect();
        self.normalize(from, lo, hi, [-1.0, 1.0])
    }

    /// `weight × distance`, the distance clamped to [0, 1] m.
    fn distance_term(&mut self, dist: NodeId, name: String, weight: f64) {
        let metres = self.normalize((dist, "value"), vec![0.0], vec![1.0], [0.0, 1.0]);
        self.reward(metres, name, weight);
    }

    /// Region (site) `name` as `(centre, half-extents)` per world axis: the site's world
    /// position and its `size`. The box is axis-aligned in the **world** frame: a cone takes a
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
        Ok((c, [site.size.x, site.size.y, site.size.z]))
    }
}

fn no(what: &str, name: &str) -> String {
    format!("no {what} named `{name}` in the scene")
}
