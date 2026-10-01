//! Task IR: scene reference, goals, rewards, termination, randomization, reset, and the
//! `ObservationSpec` declaration (spec 6). Filled in by P20 — no neural nets here (rule 6).
//!
//! This module carries **IR-D** only: a pure dataflow DAG with no side effects (spec 6.2).
//! The IR-C control nodes (`Sequence`, `Branch`, `SubTask`, `Repeat`) are deliberately not
//! `TaskNode` variants; they live in [`crate::control`] and hang off [`TaskIr::control`].
//! Standard pick-and-place / reach / push tasks need IR-D alone.
//!
//! Three prohibitions of spec 6.1 are structural rather than checked: there is no node for a
//! neural network, none for preprocessing (Observation IR owns it) and none for wall-clock
//! access (`DET-002`) — [`TaskNode::GetTime`] reads simulation time only. `DET-020` is
//! likewise structural: every collection here is a `BTreeMap` / `BTreeSet`.
//!
//! Layout: the node set is `task/node.rs`, the spec 7.4 declaration `task/observation_spec.rs`
//! and [`TaskIr::validate`] `task/validate.rs`; the IR itself, its hashes and the test fixtures
//! are here, and every type keeps its `es_ir::task::` path.

use std::collections::BTreeSet;
use std::fmt;

use es_core::StableId;
use serde::{Deserialize, Serialize};

use crate::control::ControlGraph;
use crate::diag::Diagnostic;
use crate::graph::{Graph, IrNode};
use crate::hash::{canonical_hash, CanonWriter};

mod node;
mod observation_spec;
mod validate;

pub use node::TaskNode;
pub use observation_spec::{
    ObsChannel, ObsSource, ObservationSpec, SeedStream, SensorPath, SensorRender, Tonemap,
};

/// Current Task IR schema version.
///
/// `2` since IR-C: [`TaskIr`] carries an optional [`ControlGraph`] and `task_hash` mixes it in
/// (migration note in `docs/design/ir-types.md`).
pub const SCHEMA_VERSION: u32 = 2;

const TASK_TAG: &str = "es.ir.task.v1";
const TASK_GRAPH_TAG: &str = "es.ir.task_graph.v1";

// --- parameter vocabulary and expressions ----------------------------------------------------
//
// Moved to `es-ir-types` for the spec 1.5 context budget (`docs/packets/M4/P-M4-S16.md`);
// nothing here knows the graph, and every path below stays `es_ir::task::..`.

pub use es_ir_types::expr::{
    ActionSpace, Aggregation, ArithOp, CmpOp, Distribution, Expr, JointQuantity, LogicOp, MathFunc,
    NormKind, ReduceOp, TerminationKind,
};

// --- canonical encoding helpers --------------------------------------------------------------

fn wdbg(w: &mut CanonWriter, v: &impl fmt::Debug) {
    w.str(&format!("{v:?}"));
}

fn wid(w: &mut CanonWriter, id: &StableId) {
    w.bytes(id.as_bytes());
}

// --- the IR ------------------------------------------------------------------------------------

/// The scene the task runs in: the authored path plus the hashes that pin its content
/// (spec 5.3 `scene_hash` / `asset_hash`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRef {
    pub path: String,
    pub scene_hash: [u8; 32],
    pub asset_hash: [u8; 32],
}

impl SceneRef {
    fn canonical(&self, w: &mut CanonWriter) {
        w.str(&self.path);
        w.digest(&self.scene_hash);
        w.digest(&self.asset_hash);
    }
}

/// What the episode needs that no single node owns.
///
/// Reset distributions, domain randomization and the termination predicates are *nodes*
/// (spec 6.3 `ResetState`, `Randomization`, `Terminate`); this carries the episode budget that
/// `Terminate(Timeout)` counts against, the control rate, the RNG streams the task declares
/// (spec 6.6 `DET-021`) and whether deterministic mode is required (`DET-030`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskConfig {
    pub max_episode_steps: u32,
    pub control_rate_hz: f32,
    pub deterministic: bool,
    pub rng_streams: BTreeSet<String>,
}

impl TaskConfig {
    fn canonical(&self, w: &mut CanonWriter) {
        w.u32(self.max_episode_steps);
        w.f32(self.control_rate_hz);
        w.bool(self.deterministic);
        w.seq(self.rng_streams.len());
        for s in &self.rng_streams {
            w.str(s);
        }
    }
}

pub type TaskGraph = Graph<TaskNode>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskIr {
    pub schema_version: u32,
    pub scene: SceneRef,
    pub graph: TaskGraph,
    pub observation_spec: ObservationSpec,
    pub config: TaskConfig,
    /// IR-C (spec 6.2). `None` is the IR-D-only task and the default, so every task authored
    /// before IR-C parses unchanged.
    #[serde(default)]
    pub control: Option<ControlGraph>,
}

impl TaskIr {
    /// Semantic identity (spec 11.2 `task_hash`): independent of node ids and node order, so
    /// relabelling the graph leaves it unchanged.
    pub fn task_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let graph = canonical_hash(&self.graph)?;
        let mut w = CanonWriter::new();
        w.str(TASK_TAG);
        w.u32(self.schema_version);
        self.scene.canonical(&mut w);
        self.observation_spec.canonical(&mut w);
        self.config.canonical(&mut w);
        w.digest(&graph);
        match &self.control {
            Some(control) => {
                w.bool(true);
                w.digest(&control.control_hash()?);
            }
            None => w.bool(false),
        }
        w.hash()
    }

    /// Authoring identity (spec 11.2 `task_graph_hash`): the same inputs plus the user-assigned
    /// `NodeId`s, so an editor can tell "the same task, renumbered" from "unchanged".
    pub fn task_graph_hash(&self) -> Result<[u8; 32], Diagnostic> {
        let mut w = CanonWriter::new();
        w.str(TASK_GRAPH_TAG);
        w.digest(&self.task_hash()?);
        w.seq(self.graph.nodes.len());
        for (id, node) in &self.graph.nodes {
            w.u32(id.0);
            w.str(node.kind());
        }
        let mut edges: Vec<(u32, &str, u32, &str)> = self
            .graph
            .edges
            .iter()
            .map(|e| {
                (
                    e.from.node.0,
                    e.from.port.as_str(),
                    e.to.node.0,
                    e.to.port.as_str(),
                )
            })
            .collect();
        edges.sort_unstable();
        w.seq(edges.len());
        for (from, from_port, to, to_port) in edges {
            w.u32(from);
            w.str(from_port);
            w.u32(to);
            w.str(to_port);
        }
        w.hash()
    }
}

/// Fixtures and the proptest generator for the Appendix B.7 properties (P25).
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::{
        ArithOp, CmpOp, Distribution, Expr, JointQuantity, NormKind, ObsChannel, ObsSource,
        ObservationSpec, SceneRef, TaskConfig, TaskGraph, TaskIr, TaskNode, TerminationKind,
        SCHEMA_VERSION,
    };
    use crate::control::{ControlGraph, ControlNode, RepeatUntil, SubTaskRef};
    use crate::graph::NodeId;
    use crate::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
    use es_core::StableId;
    use proptest::prelude::*;
    use std::collections::{BTreeMap, BTreeSet};

    pub fn ty(elem: ElemType, n: u64, unit: Unit, frame: Frame) -> PortType {
        PortType {
            elem,
            shape: Shape::new([n]),
            unit,
            frame,
            time: TimeRef::Tick,
            image: None,
        }
    }

    fn vec3() -> PortType {
        ty(ElemType::F32, 3, Unit::Length, Frame::World)
    }

    fn scalar(unit: Unit) -> PortType {
        ty(ElemType::F32, 1, unit, Frame::World)
    }

    /// `k` reward terms (distance between two bodies), a timeout, one observation channel and
    /// a randomized / reset parameter — the shape of every pick-and-place-class task.
    pub fn task_ir(terms: &[(String, f64, f64)], dof: u32) -> TaskIr {
        let mut g = TaskGraph::new(SCHEMA_VERSION);
        let mut next = 0u32;
        let mut add = |g: &mut TaskGraph, node: TaskNode| {
            let id = NodeId(next);
            next += 1;
            g.insert(id, node);
            id
        };

        for (name, weight, hi) in terms {
            let a = add(
                &mut g,
                TaskNode::GetBodyPose {
                    body: StableId::from_path(&format!("{name}/a")),
                    relative_to: Frame::World,
                },
            );
            let b = add(
                &mut g,
                TaskNode::GetBodyPose {
                    body: StableId::from_path(&format!("{name}/b")),
                    relative_to: Frame::World,
                },
            );
            let sub = add(
                &mut g,
                TaskNode::Arith {
                    op: ArithOp::Sub,
                    ty: vec3(),
                },
            );
            let norm = add(
                &mut g,
                TaskNode::Norm {
                    kind: NormKind::L2,
                    ty: vec3(),
                },
            );
            let normalize = add(
                &mut g,
                TaskNode::Normalize {
                    lo: vec![0.0],
                    hi: vec![*hi],
                    out_lo: -1.0,
                    out_hi: 1.0,
                    ty: scalar(Unit::Length),
                },
            );
            let reward = add(
                &mut g,
                TaskNode::Reward {
                    name: name.clone(),
                    weight: *weight,
                    aggregation: super::Aggregation::Sum,
                    ty: scalar(Unit::Normalized { lo: -1.0, hi: 1.0 }),
                },
            );
            g.connect(a, "pos", sub, "a");
            g.connect(b, "pos", sub, "b");
            g.connect(sub, "value", norm, "value");
            g.connect(norm, "value", normalize, "value");
            g.connect(normalize, "value", reward, "value");
        }

        let time = add(&mut g, TaskNode::GetTime { since_reset: true });
        let over = add(
            &mut g,
            TaskNode::Compare {
                op: CmpOp::Gt,
                rhs: Some(10.0),
                ty: scalar(Unit::Time),
            },
        );
        let stop = add(
            &mut g,
            TaskNode::Terminate {
                kind: TerminationKind::Timeout,
                hold_ticks: None,
            },
        );
        g.connect(time, "value", over, "a");
        g.connect(over, "value", stop, "value");

        let arm = StableId::from_path("arm");
        let joint_ty = ty(
            ElemType::F32,
            u64::from(dof),
            Unit::Angle,
            Frame::Joint(arm),
        );
        let joints = add(
            &mut g,
            TaskNode::GetJointState {
                body: arm,
                joints: (0..dof).map(|i| format!("j{i}")).collect(),
                quantity: JointQuantity::Position,
            },
        );
        let obs = add(
            &mut g,
            TaskNode::ObservationSpec {
                channel: "joint_state".to_owned(),
                ty: joint_ty.clone(),
            },
        );
        g.connect(joints, "value", obs, "value");

        add(
            &mut g,
            TaskNode::Randomization {
                target: "cube.mass".to_owned(),
                dist: Distribution::Uniform { lo: 0.1, hi: 0.5 },
                stream: "domain".to_owned(),
            },
        );
        add(
            &mut g,
            TaskNode::ResetState {
                target: "cube.pos".to_owned(),
                dist: Distribution::Normal {
                    mean: 0.0,
                    std: 0.05,
                },
                stream: "reset".to_owned(),
            },
        );
        add(
            &mut g,
            TaskNode::ActionSpec {
                space: super::ActionSpace::JointPosition,
                dim: dof,
                control_rate_hz: 50.0,
            },
        );

        let mut channels = BTreeMap::new();
        channels.insert(
            "joint_state".to_owned(),
            ObsChannel {
                source: ObsSource::JointState {
                    body: arm,
                    dof,
                    quantity: JointQuantity::Position,
                },
                ty: joint_ty,
            },
        );

        TaskIr {
            schema_version: SCHEMA_VERSION,
            scene: SceneRef {
                path: "scenes/pick.usda".to_owned(),
                scene_hash: [7u8; 32],
                asset_hash: [9u8; 32],
            },
            graph: g,
            observation_spec: ObservationSpec { channels },
            control: None,
            config: TaskConfig {
                max_episode_steps: 400,
                control_rate_hz: 50.0,
                deterministic: true,
                rng_streams: ["domain", "reset"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
            },
        }
    }

    /// A `Sequence` of one stage per reward term, wrapped in a `Repeat` and reached through a
    /// `Branch` — every IR-C kind of spec 6.2, over a task that actually declares those rewards.
    pub fn control_tree(terms: &[(String, f64, f64)]) -> ControlGraph {
        let done = |port: &str| Expr::Compare {
            op: CmpOp::Gt,
            lhs: Box::new(Expr::Port(port.to_owned())),
            rhs: Box::new(Expr::Const(0.5)),
        };
        let mut nodes = BTreeMap::new();
        let stage_ids: Vec<NodeId> = (0..terms.len())
            .map(|i| NodeId(u32::try_from(i).unwrap_or(0) + 4))
            .collect();
        for ((name, weight, _), id) in terms.iter().zip(&stage_ids) {
            nodes.insert(
                *id,
                ControlNode::SubTask {
                    task: SubTaskRef {
                        name: name.clone(),
                        rewards: [name.clone()].into(),
                        observation: ["joint_state".to_owned()].into(),
                        success: Some(done("stage.done")),
                        reset_on_entry: false,
                        weight: *weight,
                    },
                    timeout_ticks: 40,
                },
            );
        }
        nodes.insert(
            NodeId(3),
            ControlNode::Sequence {
                children: stage_ids,
            },
        );
        nodes.insert(
            NodeId(2),
            ControlNode::Repeat {
                body: NodeId(3),
                until: RepeatUntil::Count(2),
            },
        );
        nodes.insert(
            NodeId(1),
            ControlNode::SubTask {
                task: SubTaskRef {
                    name: "abort".to_owned(),
                    rewards: BTreeSet::new(),
                    observation: BTreeSet::new(),
                    success: Some(done("stage.ticks")),
                    reset_on_entry: true,
                    weight: 0.0,
                },
                timeout_ticks: 5,
            },
        );
        nodes.insert(
            NodeId(0),
            ControlNode::Branch {
                condition: done("time.episode"),
                then_: NodeId(2),
                else_: NodeId(1),
            },
        );
        ControlGraph {
            root: NodeId(0),
            nodes,
        }
    }

    /// Valid, connected small Task IRs (Appendix B.7), half of them carrying a control tree so
    /// the five properties cover IR-C as well as IR-D.
    pub fn arbitrary_task_ir() -> impl Strategy<Value = TaskIr> {
        (
            proptest::collection::vec(("term[a-c]{1,3}", 0.1f64..4.0, 0.01f64..2.0), 1..4),
            1u32..8,
            any::<bool>(),
        )
            .prop_map(|(mut terms, dof, control)| {
                terms.sort_by(|a, b| a.0.cmp(&b.0));
                terms.dedup_by(|a, b| a.0 == b.0);
                let mut ir = task_ir(&terms, dof);
                if control {
                    ir.control = Some(control_tree(&terms));
                }
                ir
            })
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{arbitrary_task_ir, task_ir, ty};
    use super::*;
    use crate::codes;
    use crate::graph::{Edge, NodeId, PortRef};
    use crate::types::{ElemType, Frame, Shape, TimeRef, Unit};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn fixture() -> TaskIr {
        task_ir(&[("reach".to_owned(), 1.0, 0.5)], 7)
    }

    /// Shifts every `NodeId` by 100 and reverses the edge list: pure authoring churn.
    fn relabel(ir: &TaskIr) -> TaskIr {
        let mut out = ir.clone();
        out.graph.nodes = ir
            .graph
            .nodes
            .iter()
            .map(|(id, n)| (NodeId(id.0 + 100), n.clone()))
            .collect();
        out.graph.edges = ir
            .graph
            .edges
            .iter()
            .rev()
            .map(|e| Edge {
                from: PortRef::new(NodeId(e.from.node.0 + 100), e.from.port.clone()),
                to: PortRef::new(NodeId(e.to.node.0 + 100), e.to.port.clone()),
            })
            .collect();
        out
    }

    fn codes_of(diags: &[Diagnostic]) -> Vec<&str> {
        diags.iter().map(|d| d.code.as_str()).collect()
    }

    #[test]
    fn valid_fixture_has_no_diagnostics() {
        let diags = fixture().validate();
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn serde_json_round_trip() {
        let ir = fixture();
        let json = serde_json::to_string(&ir).unwrap();
        assert_eq!(serde_json::from_str::<TaskIr>(&json).unwrap(), ir);
    }

    /// S-13: `0` and anything past [`SCHEMA_VERSION`] are rejected; every version this build
    /// still understands loads, from a file as much as from memory.
    #[test]
    fn unsupported_schema_versions_are_rejected() {
        let with = |v: u32| {
            let mut ir = fixture();
            ir.schema_version = v;
            ir
        };
        assert_eq!(codes_of(&with(0).validate()), [codes::TASK_002]);
        assert_eq!(
            codes_of(&with(SCHEMA_VERSION + 1).validate()),
            [codes::TASK_002]
        );
        for v in 1..=SCHEMA_VERSION {
            assert!(with(v).validate().is_empty(), "version {v} is supported");
        }
        // The same guard is what a loaded file meets: `task_from_toml` builds a `TaskIr`.
        let toml = crate::serial::task_to_toml(&with(SCHEMA_VERSION + 1)).unwrap();
        let loaded = crate::serial::task_from_toml(&toml).unwrap();
        assert_eq!(codes_of(&loaded.validate()), [codes::TASK_002]);
    }

    #[test]
    fn cycle_is_graph_001() {
        let mut ir = fixture();
        let ty3 = ty(ElemType::F32, 3, Unit::Length, Frame::World);
        for id in [900u32, 901] {
            ir.graph.insert(
                NodeId(id),
                TaskNode::Arith {
                    op: ArithOp::Add,
                    ty: ty3.clone(),
                },
            );
        }
        ir.graph.connect(NodeId(900), "value", NodeId(901), "a");
        ir.graph.connect(NodeId(901), "value", NodeId(900), "a");
        assert!(codes_of(&ir.validate()).contains(&codes::GRAPH_001));
    }

    #[test]
    fn unit_mismatch_is_type_003() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: ty(ElemType::F32, 3, Unit::Force, Frame::World),
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::GetBodyPose {
                body: StableId::from_path("cube"),
                relative_to: Frame::World,
            },
        );
        ir.graph.connect(NodeId(901), "pos", NodeId(900), "value");
        assert!(codes_of(&ir.validate()).contains(&codes::TYPE_003));
    }

    #[test]
    fn unaligned_time_ref_is_type_014() {
        let mut ir = fixture();
        let sensor = StableId::from_path("cam_front");
        let mut sensor_ty = ty(ElemType::F32, 3, Unit::Length, Frame::World);
        sensor_ty.time = TimeRef::Sensor {
            id: sensor,
            align: crate::types::Align::Reject,
        };
        ir.graph.insert(
            NodeId(900),
            TaskNode::GetSensor {
                sensor,
                ty: sensor_ty,
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::Norm {
                kind: NormKind::L2,
                ty: ty(ElemType::F32, 3, Unit::Length, Frame::World),
            },
        );
        ir.graph.connect(NodeId(900), "value", NodeId(901), "value");
        assert!(codes_of(&ir.validate()).contains(&codes::TYPE_014));
    }

    #[test]
    fn determinism_rules_fire() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::GetRandom {
                stream: String::new(),
                dist: Distribution::Constant(0.0),
                shape: Shape::new([1]),
            },
        );
        ir.graph.insert(
            NodeId(901),
            TaskNode::GetRandom {
                stream: "undeclared".to_owned(),
                dist: Distribution::Constant(0.0),
                shape: Shape::new([1]),
            },
        );
        ir.graph.insert(
            NodeId(902),
            TaskNode::MathFn {
                func: MathFunc::Sin,
                approx: false,
                ty: ty(ElemType::F32, 1, Unit::Angle, Frame::World),
            },
        );
        ir.graph.insert(
            NodeId(903),
            TaskNode::Reduce {
                op: ReduceOp::Sum,
                axis: 0,
                unordered: true,
                ty: ty(ElemType::F32, 3, Unit::Length, Frame::World),
            },
        );
        let diags = ir.validate();
        let got = codes_of(&diags);
        for want in [
            codes::DET_001,
            codes::DET_021,
            codes::DET_010,
            codes::DET_030,
        ] {
            assert!(got.contains(&want), "{want} missing from {got:?}");
        }
    }

    #[test]
    fn observation_channel_must_be_declared() {
        let mut ir = fixture();
        ir.observation_spec.channels.clear();
        assert_eq!(codes_of(&ir.validate()), vec![codes::TASK_001]);
    }

    #[test]
    fn nan_parameter_is_hash_001() {
        let mut ir = fixture();
        ir.graph.insert(
            NodeId(900),
            TaskNode::Reward {
                name: "bad".to_owned(),
                weight: f64::NAN,
                aggregation: Aggregation::Sum,
                ty: ty(ElemType::F32, 1, Unit::Dimensionless, Frame::World),
            },
        );
        assert!(codes_of(&ir.validate()).contains(&codes::HASH_001));
    }

    #[test]
    fn task_hash_ignores_node_ids_but_graph_hash_does_not() {
        let ir = fixture();
        let renamed = relabel(&ir);
        assert_eq!(ir.task_hash().unwrap(), renamed.task_hash().unwrap());
        assert_ne!(
            ir.task_graph_hash().unwrap(),
            renamed.task_graph_hash().unwrap()
        );
    }

    #[test]
    fn task_hash_follows_a_reward_weight() {
        let a = fixture();
        let b = task_ir(&[("reach".to_owned(), 2.0, 0.5)], 7);
        assert_ne!(a.task_hash().unwrap(), b.task_hash().unwrap());
    }

    #[test]
    fn task_hash_follows_the_scene_and_the_observation_spec() {
        let base = fixture();
        let mut scene = base.clone();
        scene.scene.scene_hash = [1u8; 32];
        assert_ne!(base.task_hash().unwrap(), scene.task_hash().unwrap());

        let other_dof = task_ir(&[("reach".to_owned(), 1.0, 0.5)], 6);
        assert_ne!(base.task_hash().unwrap(), other_dof.task_hash().unwrap());
    }

    #[test]
    fn expr_evaluates_deterministically() {
        let ports: BTreeMap<String, f64> = [("d".to_owned(), 0.01)].into_iter().collect();
        let near = Expr::Compare {
            op: CmpOp::Lt,
            lhs: Box::new(Expr::Port("d".to_owned())),
            rhs: Box::new(Expr::Const(0.02)),
        };
        assert_eq!(near.eval(&ports), Some(1.0));
        assert_eq!(
            Expr::Clamp {
                value: Box::new(Expr::Port("d".to_owned())),
                lo: 0.0,
                hi: 0.005,
            }
            .eval(&ports),
            Some(0.005)
        );
        // Missing port, division by zero and an inverted clamp are all `None`, never NaN.
        assert_eq!(Expr::Port("nope".to_owned()).eval(&ports), None);
        assert_eq!(
            Expr::Arith {
                op: ArithOp::Div,
                lhs: Box::new(Expr::Port("d".to_owned())),
                rhs: Box::new(Expr::Const(0.0)),
            }
            .eval(&ports),
            None
        );
        assert_eq!(
            Expr::Clamp {
                value: Box::new(Expr::Const(1.0)),
                lo: 1.0,
                hi: 0.0,
            }
            .eval(&ports),
            None
        );
    }

    proptest! {
        #[test]
        fn generated_tasks_are_valid(ir in arbitrary_task_ir()) {
            let diags = ir.validate();
            prop_assert!(diags.is_empty(), "{diags:?}");
        }

        #[test]
        fn generated_task_hash_survives_relabelling(ir in arbitrary_task_ir()) {
            prop_assert_eq!(ir.task_hash().unwrap(), relabel(&ir).task_hash().unwrap());
        }
    }
}
