//! Normalization: `canon` of spec 11.2 and Appendix B.7.
//!
//! `*_hash` is the identity of a graph *after* normalization, `*_graph_hash` the identity
//! before it. [`canonical_hash`](crate::hash::canonical_hash) already numbers nodes by
//! [`canonical_order`], so normalizing a graph never moves a `*_hash`; what [`canon_graph`]
//! adds is a graph whose stored `NodeId`s are that numbering, which is what a registry or a
//! diff wants to store. Both read the same order, so they cannot disagree.
//!
//! Normalization renames and reorders, and strips nothing: boundary order is an argument
//! order and stays as authored. Deployment IR and Evaluation IR carry no graph, so their
//! normalization is the identity and has no function here.

use std::collections::BTreeMap;

use crate::diag::Diagnostic;
use crate::graph::{Edge, Graph, IrNode, NodeId, PortRef};
use crate::hash::canonical_order;
use crate::learning::LearningGraph;
use crate::observation::ObservationIr;
use crate::task::TaskIr;

/// `NodeId(0) ..` in canonical rank order, with edges sorted so the encoding of two equal
/// graphs is byte-identical.
///
/// Canonical up to interchangeable nodes: two nodes with identical upstream *and* downstream
/// cones tie on rank and may come out in either order, which is exactly the case where the
/// swap is invisible to [`canonical_hash`](crate::hash::canonical_hash).
pub fn canon_graph<N: IrNode + Clone>(g: &Graph<N>) -> Result<Graph<N>, Diagnostic> {
    let mut out = relabel(g, &canon_map(g)?);
    out.edges.sort_unstable();
    Ok(out)
}

/// Canonical Task IR: only the graph is renumbered, every other field is identity.
pub fn canon_task(ir: &TaskIr) -> Result<TaskIr, Diagnostic> {
    Ok(TaskIr {
        graph: canon_graph(&ir.graph)?,
        ..ir.clone()
    })
}

/// Canonical Observation IR. The named outputs point into the graph, so they are renumbered
/// with it.
pub fn canon_observation(ir: &ObservationIr) -> Result<ObservationIr, Diagnostic> {
    let map = canon_map(&ir.graph)?;
    let mut out = relabel_observation(ir, &map);
    out.graph.edges.sort_unstable();
    Ok(out)
}

/// Canonical Learning IR. `inputs` / `outputs` are named tensor ports, not node references,
/// so only `nodes` moves.
pub fn canon_learning(g: &LearningGraph) -> Result<LearningGraph, Diagnostic> {
    Ok(LearningGraph {
        nodes: canon_graph(&g.nodes)?,
        ..g.clone()
    })
}

fn canon_map<N: IrNode>(g: &Graph<N>) -> Result<BTreeMap<NodeId, NodeId>, Diagnostic> {
    Ok(canonical_order(g)?
        .into_iter()
        .enumerate()
        .map(|(rank, id)| (id, NodeId(u32::try_from(rank).unwrap_or(u32::MAX))))
        .collect())
}

/// Applies an id map, keeping edge and boundary order. Ids the map does not mention are left
/// alone; a dangling edge endpoint is reported by the hash, not silently repaired here.
fn relabel<N: Clone>(g: &Graph<N>, map: &BTreeMap<NodeId, NodeId>) -> Graph<N> {
    let at = |id: NodeId| *map.get(&id).unwrap_or(&id);
    let port = |p: &PortRef| PortRef::new(at(p.node), p.port.clone());
    Graph {
        schema_version: g.schema_version,
        nodes: g.nodes.iter().map(|(id, n)| (at(*id), n.clone())).collect(),
        edges: g
            .edges
            .iter()
            .map(|e| Edge {
                from: port(&e.from),
                to: port(&e.to),
            })
            .collect(),
        inputs: g.inputs.iter().map(port).collect(),
        outputs: g.outputs.iter().map(port).collect(),
    }
}

fn relabel_observation(ir: &ObservationIr, map: &BTreeMap<NodeId, NodeId>) -> ObservationIr {
    let at = |id: NodeId| *map.get(&id).unwrap_or(&id);
    ObservationIr {
        graph: relabel(&ir.graph, map),
        outputs: ir
            .outputs
            .iter()
            .map(|(name, out)| {
                (
                    name.clone(),
                    crate::observation::ObservationOutput {
                        port: PortRef::new(at(out.port.node), out.port.port.clone()),
                        ty: out.ty.clone(),
                    },
                )
            })
            .collect(),
        ..ir.clone()
    }
}

#[cfg(any(test, feature = "testing"))]
pub mod testing {
    //! Generators and mutators for the Appendix B.7 properties: a random relabelling, a
    //! layout sidecar, one-step edits, and the algebraic units.

    use super::{relabel, relabel_observation};
    use crate::deployment::DeploymentIr;
    use crate::evaluation::EvaluationIr;
    use crate::graph::{Graph, IrNode, NodeId};
    use crate::learning::LearningGraph;
    use crate::observation::ObservationIr;
    use crate::task::TaskIr;
    use crate::types::{Unit, UnitPowers};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    /// A deterministic, seedable permutation source. `proptest` owns randomness in tests;
    /// this only turns one `u64` into a shuffle, so no global RNG is involved (spec 3.4).
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn shuffle<T>(&mut self, v: &mut [T]) {
            for i in (1..v.len()).rev() {
                v.swap(i, self.next_u32() as usize % (i + 1));
            }
        }
    }

    fn permutation(n: usize, seed: u64) -> Vec<u32> {
        let mut rng = Lcg(seed ^ 0x5eed_1234_9876_abcd);
        // Offset by one so no shuffled graph can keep the ids it started with by accident.
        let mut ids: Vec<u32> = (0..n as u32).map(|i| i * 3 + 1).collect();
        rng.shuffle(&mut ids);
        ids
    }

    fn shuffle_map<N>(g: &Graph<N>, seed: u64) -> BTreeMap<NodeId, NodeId> {
        let perm = permutation(g.nodes.len(), seed);
        g.nodes
            .keys()
            .zip(perm)
            .map(|(id, new)| (*id, NodeId(new)))
            .collect()
    }

    /// Relabels every node by a seeded permutation and permutes the edge list.
    ///
    /// `inputs` / `outputs` keep their order on purpose: the graph boundary is an argument
    /// list, so its order is semantic and reordering it is a real change (spec 11.2).
    pub fn shuffle_ids<N: Clone>(g: &Graph<N>, seed: u64) -> Graph<N> {
        let mut out = relabel(g, &shuffle_map(g, seed));
        Lcg(seed).shuffle(&mut out.edges);
        out
    }

    pub fn shuffle_task_ids(ir: &TaskIr, seed: u64) -> TaskIr {
        TaskIr {
            graph: shuffle_ids(&ir.graph, seed),
            ..ir.clone()
        }
    }

    pub fn shuffle_observation_ids(ir: &ObservationIr, seed: u64) -> ObservationIr {
        let mut out = relabel_observation(ir, &shuffle_map(&ir.graph, seed));
        Lcg(seed).shuffle(&mut out.graph.edges);
        out
    }

    pub fn shuffle_learning_ids(g: &LearningGraph, seed: u64) -> LearningGraph {
        LearningGraph {
            nodes: shuffle_ids(&g.nodes, seed),
            ..g.clone()
        }
    }

    /// A `.eslayout` sidecar (spec 5.1 rule 7): node positions, which live outside the IR.
    pub fn arbitrary_layout() -> impl Strategy<Value = BTreeMap<NodeId, (f32, f32)>> {
        proptest::collection::vec((any::<u32>(), -1e3f32..1e3, -1e3f32..1e3), 0..8).prop_map(|v| {
            v.into_iter()
                .map(|(id, x, y)| (NodeId(id), (x, y)))
                .collect()
        })
    }

    /// The units unit algebra is defined on: the named table plus composites, all built
    /// through `from_powers` so they are normalized. The opaque units (`Quaternion`, `Token`,
    /// `Normalized`, ...) are excluded — `docs/design/ir-types.md` says algebra on them is an
    /// error, which [`super::super::types::Unit::mul`] reports as `TYPE-010`.
    pub fn arbitrary_unit() -> impl Strategy<Value = Unit> {
        (-3i8..=3, -3i8..=3, -3i8..=3, -3i8..=3)
            .prop_map(|(m, kg, s, rad)| Unit::from_powers(UnitPowers::new(m, kg, s, rad)))
    }

    /// Structural edits that need no knowledge of the node type: one added node, one dropped
    /// edge. Both change the node or edge count, so both must move the hash.
    fn graph_edits<N: IrNode + Clone>(g: &Graph<N>) -> Vec<Graph<N>> {
        let mut out = Vec::new();
        if let Some(node) = g.nodes.values().next() {
            let mut added = g.clone();
            let fresh = NodeId(g.nodes.keys().map(|i| i.0).max().unwrap_or(0) + 1);
            added.insert(fresh, node.clone());
            out.push(added);
        }
        if !g.edges.is_empty() {
            let mut dropped = g.clone();
            dropped.edges.pop();
            out.push(dropped);
        }
        out
    }

    /// One-step edits of a Task IR, each semantically different from `ir`.
    pub fn task_edits(ir: &TaskIr) -> Vec<TaskIr> {
        let mut out = vec![
            TaskIr {
                config: crate::task::TaskConfig {
                    control_rate_hz: ir.config.control_rate_hz + 1.0,
                    ..ir.config.clone()
                },
                ..ir.clone()
            },
            TaskIr {
                config: crate::task::TaskConfig {
                    max_episode_steps: ir.config.max_episode_steps + 1,
                    ..ir.config.clone()
                },
                ..ir.clone()
            },
        ];
        let mut scene = ir.clone();
        scene.scene.scene_hash[0] ^= 1;
        out.push(scene);
        out.extend(graph_edits(&ir.graph).into_iter().map(|graph| TaskIr {
            graph,
            ..ir.clone()
        }));
        out
    }

    pub fn observation_edits(ir: &ObservationIr) -> Vec<ObservationIr> {
        let mut out = Vec::new();
        let mut task_ref = ir.clone();
        task_ref.task_ref[0] ^= 1;
        out.push(task_ref);
        if let Some((name, value)) = ir.outputs.iter().next() {
            let mut renamed = ir.clone();
            renamed.outputs.remove(name);
            renamed.outputs.insert(format!("{name}_2"), value.clone());
            out.push(renamed);
        }
        out.extend(
            graph_edits(&ir.graph)
                .into_iter()
                .map(|graph| ObservationIr {
                    graph,
                    ..ir.clone()
                }),
        );
        out
    }

    pub fn learning_edits(g: &LearningGraph) -> Vec<LearningGraph> {
        let mut out = Vec::new();
        let mut hz = g.clone();
        hz.policy.contract.replanning_hz += 1.0;
        out.push(hz);
        let mut window = g.clone();
        window.policy.contract.observation_window += 1;
        out.push(window);
        out.extend(
            graph_edits(&g.nodes)
                .into_iter()
                .map(|nodes| LearningGraph { nodes, ..g.clone() }),
        );
        out
    }

    /// Deployment IR has no graph, so every edit is a parameter edit.
    pub fn deployment_edits(ir: &DeploymentIr) -> Vec<DeploymentIr> {
        let mut out = Vec::new();
        let mut speed = ir.clone();
        // Widening, never narrowing: a test must not tighten a safety envelope (INV-12).
        speed.safety.ee_velocity_max += 1.0;
        out.push(speed);
        let mut deadline = ir.clone();
        deadline.deadlines.observation_age.0 += 1;
        out.push(deadline);
        let mut name = ir.clone();
        name.robot.name.push('x');
        out.push(name);
        let mut horizon = ir.clone();
        horizon.action.horizon += 1;
        out.push(horizon);
        out
    }

    pub fn evaluation_edits(ir: &EvaluationIr) -> Vec<EvaluationIr> {
        let mut out = Vec::new();
        let mut episodes = ir.clone();
        episodes.episodes.n_episodes += 1;
        out.push(episodes);
        let mut task = ir.clone();
        task.task.push('x');
        out.push(task);
        if !ir.acceptance.is_empty() {
            let mut threshold = ir.clone();
            threshold.acceptance[0].threshold += 1.0;
            out.push(threshold);
        }
        if !ir.suites.is_empty() {
            let mut suites = ir.clone();
            suites.suites.pop();
            out.push(suites);
        }
        out
    }
}
