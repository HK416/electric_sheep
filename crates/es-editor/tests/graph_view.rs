//! Oracle for the layered graph view-model (spec 23.2, W8).
//!
//! The bundle is `es-ir`'s own cross-IR fixture (see `common/mod.rs`), so "what the editor
//! shows" is judged against the same five IRs the compiler gate uses.

mod common;

use std::collections::BTreeMap;

use common::Fixture;
use es_editor::model::graph_view::{LayeredGraph, DEPLOYMENT, LEARNING, OBSERVATION, TASK};
use es_ir::codes;
use es_ir::graph::NodeId;
use es_ir::serial::{IrKind, Layout};

fn view(fixture: &Fixture) -> LayeredGraph {
    LayeredGraph::from_bundle(
        &fixture.task,
        &fixture.observation,
        &fixture.learning,
        &fixture.deployment,
    )
}

/// The position of one node of one layer, after a layout has run.
#[track_caller]
fn pos(graph: &LayeredGraph, layer: usize, id: u32) -> [f32; 2] {
    graph.layers[layer]
        .nodes
        .iter()
        .find(|n| n.id == NodeId(id))
        .expect("node exists")
        .layout
        .expect("layout has run")
}

/// Positions are exact constants, never arithmetic results, so they compare bit for bit --
/// which is also what keeps clippy's `float_cmp` happy.
fn bits(at: [f32; 2]) -> [u32; 2] {
    [at[0].to_bits(), at[1].to_bits()]
}

#[test]
fn every_ir_becomes_a_layer_with_its_own_nodes() {
    let fixture = Fixture::new();
    let graph = view(&fixture);

    assert_eq!(graph.layers.len(), 4);
    let kinds: Vec<IrKind> = graph.layers.iter().map(|l| l.kind).collect();
    assert_eq!(
        kinds,
        vec![
            IrKind::Task,
            IrKind::Observation,
            IrKind::Learning,
            IrKind::Deployment
        ]
    );
    assert_eq!(
        graph.layers[TASK].nodes.len(),
        fixture.task.graph.nodes.len()
    );
    assert_eq!(
        graph.layers[OBSERVATION].nodes.len(),
        fixture.observation.graph.nodes.len()
    );
    assert_eq!(
        graph.layers[LEARNING].nodes.len(),
        fixture.learning.nodes.nodes.len()
    );
    // Deployment IR is a record, not a graph: action, envelope, watchdogs, fallback.
    assert_eq!(graph.layers[DEPLOYMENT].nodes.len(), 4);
    assert_eq!(
        graph.layers[TASK].edges.len(),
        fixture.task.graph.edges.len()
    );

    // A node keeps the IR's own stable kind tag and its ports.
    let image_input = &graph.layers[OBSERVATION].nodes[0];
    assert_eq!(image_input.kind, "ImageInput");
    assert_eq!(image_input.label, "ImageInput #0");
    assert_eq!(image_input.ports.outputs, vec!["out".to_owned()]);
    assert!(image_input.ports.inputs.is_empty());
    assert!(
        image_input.layout.is_none(),
        "no position until a layout runs"
    );
}

#[test]
fn cross_edges_mirror_the_joins_es_ir_cross_checks() {
    let fixture = Fixture::new();
    let graph = view(&fixture);

    // Two declared channels, two policy inputs, one action output (spec 7.4, 8.4, 8.5).
    assert_eq!(graph.cross_edges.len(), 5, "{:#?}", graph.cross_edges);

    let pairs: Vec<(usize, usize, &str)> = graph
        .cross_edges
        .iter()
        .map(|e| (e.from.layer, e.to.layer, e.label.as_str()))
        .collect();
    assert!(pairs.contains(&(TASK, OBSERVATION, "rgb_front")));
    assert!(pairs.contains(&(TASK, OBSERVATION, "joint_state")));
    assert!(pairs.contains(&(OBSERVATION, LEARNING, "rgb_front")));
    assert!(pairs.contains(&(OBSERVATION, LEARNING, "joint_state")));
    assert!(pairs.contains(&(LEARNING, DEPLOYMENT, "actions")));

    // Every endpoint names a node that exists in its layer.
    for edge in &graph.cross_edges {
        for end in [edge.from, edge.to] {
            assert!(
                graph.layers[end.layer]
                    .nodes
                    .iter()
                    .any(|n| n.id == end.node),
                "{end:?} is not in layer {}",
                end.layer
            );
        }
    }
}

#[test]
fn a_valid_bundle_has_no_diagnostics() {
    let graph = view(&Fixture::new());
    assert!(graph.diagnostics.is_empty(), "{:#?}", graph.diagnostics);
}

#[test]
fn a_broken_bundle_surfaces_the_code_in_diagnostics() {
    let mut fixture = Fixture::new();
    fixture.learning.policy.contract.action_dim = 7; // XIR-020: against deployment action.dim
    let graph = view(&fixture);
    assert!(
        graph
            .diagnostics
            .iter()
            .any(|d| d.code.as_str() == codes::XIR_020),
        "{:#?}",
        graph.diagnostics
    );
}

#[test]
fn auto_layout_is_deterministic_and_places_every_node() {
    let fixture = Fixture::new();
    let (mut first, mut second) = (view(&fixture), view(&fixture));
    first.auto_layout();
    second.auto_layout();
    assert_eq!(first, second, "the same bundle must lay out identically");
    assert!(first
        .layers
        .iter()
        .all(|l| l.nodes.iter().all(|n| n.layout.is_some())));
}

#[test]
fn auto_layout_ranks_by_dataflow_and_never_overlaps_within_a_rank() {
    let fixture = Fixture::new();
    let mut graph = view(&fixture);
    graph.auto_layout();

    // Observation: ImageInput(0) -> Resize(1) -> Normalize(2); StateInput(3) -> Normalize(4).
    let rank_x = |id: u32| pos(&graph, OBSERVATION, id)[0];
    let row_y = |id: u32| pos(&graph, OBSERVATION, id)[1].to_bits();
    assert!(
        rank_x(0) < rank_x(1) && rank_x(1) < rank_x(2),
        "rank follows the edges"
    );
    assert_eq!(
        rank_x(0).to_bits(),
        rank_x(3).to_bits(),
        "both sources are rank 0"
    );
    assert_ne!(row_y(0), row_y(3), "same rank, different rows");

    // No two nodes of one layer share a position.
    for layer in &graph.layers {
        let mut seen: BTreeMap<[u32; 2], NodeId> = BTreeMap::new();
        for node in &layer.nodes {
            let at = node.layout.expect("layout has run");
            let key = bits(at);
            assert!(
                seen.insert(key, node.id).is_none(),
                "{:?} overlaps {:?} at {at:?}",
                node.id,
                seen[&key]
            );
        }
    }

    // Layers occupy their own bands: no Task node sits at a Learning node's height.
    let extent = |layer: usize| {
        graph.layers[layer]
            .nodes
            .iter()
            .map(|n| n.layout.expect("layout has run")[1])
            .fold((f32::MAX, f32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)))
    };
    for layer in 0..3 {
        assert!(
            extent(layer).1 < extent(layer + 1).0,
            "layer {layer} runs into layer {}",
            layer + 1
        );
    }
}

#[test]
fn an_eslayout_sidecar_overrides_the_automatic_positions() {
    let fixture = Fixture::new();
    let mut graph = view(&fixture);
    graph.auto_layout();

    let mut layout = Layout::default();
    layout.positions.insert(NodeId(1), [-11.0, 22.0]);
    graph.apply_layout(&layout);

    assert_eq!(bits(pos(&graph, OBSERVATION, 1)), bits([-11.0, 22.0]));
    // A node the sidecar does not name keeps its automatic position.
    assert_ne!(bits(pos(&graph, OBSERVATION, 2)), bits([-11.0, 22.0]));
}
