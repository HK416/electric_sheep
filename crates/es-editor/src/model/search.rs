//! Finding a node in a graph of a hundred (spec 23.4 stage 2, "search and large-graph
//! performance").
//!
//! Every decision is here: what a query matches, in what order the hits come back, and which
//! one Enter goes to next. `crate::app` sets the pan from the hit's position and highlights it
//! — geometry, nothing more (spec 28.10 rule 3).
//!
//! Two sources, one matcher. Read-only mode searches the four stacked IRs of a
//! [`LayeredGraph`]; edit mode searches the one IR of an [`EditSession`], which may already
//! hold nodes the layered view was built before. Both report `(layer, node)` in the spec 23.2
//! stacking order, so `app.rs` looks a hit up the same way whichever mode found it.

use es_ir::graph::NodeId;
use es_ir::serial::IrKind;

use crate::model::edit::EditSession;
use crate::model::graph_view::{self, LayeredGraph};

/// Which band of spec 23.2's stack, and which node in it. The two IRs do not share a `NodeId`
/// namespace, so neither half of this pair identifies a node on its own.
pub type Hit = (usize, NodeId);

/// A query, its hits, and which one is current.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Search {
    /// Bound straight to the text box; [`Search::set_hits`] is what makes it mean anything.
    pub query: String,
    hits: Vec<Hit>,
    at: Option<usize>,
}

impl Search {
    /// Case-insensitive substring over every node's kind tag, label and port names, in
    /// layer-then-id order. **An empty query matches nothing** — a search box a person has not
    /// typed in yet should not claim to have found the whole graph.
    pub fn filter(query: &str, graph: &LayeredGraph) -> Vec<Hit> {
        let Some(needle) = needle(query) else {
            return Vec::new();
        };
        let mut hits: Vec<Hit> = Vec::new();
        for (layer, view) in graph.layers.iter().enumerate() {
            for node in &view.nodes {
                let ports = node.ports.inputs.iter().chain(&node.ports.outputs);
                if matches(&needle, node.kind, &node.label, ports) {
                    hits.push((layer, node.id));
                }
            }
        }
        hits.sort_unstable();
        hits
    }

    /// The same over the one IR being edited. The label is the one the canvas paints, so the
    /// same text a person can read is the text they can search for.
    pub fn filter_session(query: &str, session: &EditSession) -> Vec<Hit> {
        let Some(needle) = needle(query) else {
            return Vec::new();
        };
        let layer = band_of(session.graph.kind());
        let mut hits: Vec<Hit> = Vec::new();
        for (id, kind) in session.graph.nodes() {
            let mut ports = session.graph.port_names(id, es_ir::Dir::In);
            ports.extend(session.graph.port_names(id, es_ir::Dir::Out));
            if matches(&needle, kind, &label(kind, id), ports.iter()) {
                hits.push((layer, id));
            }
        }
        hits.sort_unstable();
        hits
    }

    /// Replaces the hits and forgets where the cycle was, so the next Enter starts at the top
    /// of the new list rather than in the middle of the old one.
    pub fn set_hits(&mut self, hits: Vec<Hit>) {
        self.hits = hits;
        self.at = None;
    }

    pub fn hits(&self) -> &[Hit] {
        &self.hits
    }

    pub fn current(&self) -> Option<Hit> {
        self.hits.get(self.at?).copied()
    }

    /// The next hit, cycling back to the first after the last. `None` when nothing matched.
    pub fn advance(&mut self) -> Option<Hit> {
        if self.hits.is_empty() {
            self.at = None;
            return None;
        }
        self.at = Some(self.at.map_or(0, |i| (i + 1) % self.hits.len()));
        self.current()
    }

    /// What the box says beside itself. A count is something a person is told, so it is
    /// decided here (spec 28.10 rule 3).
    pub fn summary(&self) -> String {
        if self.query.trim().is_empty() {
            return String::new();
        }
        if self.hits.is_empty() {
            return "no match".to_owned();
        }
        match self.at {
            Some(i) => format!("{} of {}", i + 1, self.hits.len()),
            None => format!("{} match(es)", self.hits.len()),
        }
    }
}

/// The lowercased query, or `None` if there is nothing to look for.
fn needle(query: &str) -> Option<String> {
    let trimmed = query.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_lowercase())
}

fn matches<'a>(
    needle: &str,
    kind: &str,
    label: &str,
    mut ports: impl Iterator<Item = &'a String>,
) -> bool {
    contains(kind, needle) || contains(label, needle) || ports.any(|p| contains(p, needle))
}

fn contains(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(needle)
}

/// The label `graph_view` gives a node and the canvas paints.
fn label(kind: &str, id: NodeId) -> String {
    format!("{kind} #{}", id.0)
}

/// Which band of spec 23.2's stack an IR is drawn in. Only the three graph IRs reach this —
/// Deployment and Evaluation are records, so `EditIr` has no variant for them.
fn band_of(kind: IrKind) -> usize {
    match kind {
        IrKind::Observation => graph_view::OBSERVATION,
        IrKind::Learning => graph_view::LEARNING,
        IrKind::Task | IrKind::Deployment | IrKind::Evaluation => graph_view::TASK,
    }
}

#[cfg(test)]
mod tests {
    use super::Search;

    use std::path::{Path, PathBuf};

    use es_ir::graph::NodeId;
    use es_ir::serial::Layout;

    use crate::model::edit::{EditIr, EditSession};
    use crate::model::graph_view::{LayeredGraph, LEARNING, TASK};

    fn fixtures() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning")
    }

    fn read(name: &str) -> String {
        std::fs::read_to_string(fixtures().join(name)).expect("the demo bundle's document")
    }

    fn demo() -> LayeredGraph {
        let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task");
        let observation =
            es_ir::serial::observation_from_toml(&read("observation.toml")).expect("observation");
        let learning = es_ir::serial::learning_from_toml(&read("learning.toml")).expect("learning");
        let deployment =
            es_ir::serial::deployment_from_toml(&read("deployment.toml")).expect("deployment");
        LayeredGraph::from_bundle(&task, &observation, &learning, &deployment)
    }

    /// Oracle 4a: the query is case-insensitive, the hits come back in layer-then-id order,
    /// an empty query finds nothing, and Enter cycles.
    #[test]
    fn search_is_case_insensitive_and_ordered() {
        let graph = demo();

        let lower = Search::filter("policy", &graph);
        assert_eq!(lower, Search::filter("POLICY", &graph));
        assert_eq!(lower, Search::filter("  PoLiCy  ", &graph));
        assert!(!lower.is_empty(), "the demo bundle has a policy node");
        assert!(
            lower.iter().all(|(layer, _)| *layer == LEARNING),
            "PolicyHead and PolicyBundle are Learning IR nodes: {lower:?}"
        );

        let mut sorted = lower.clone();
        sorted.sort_unstable();
        assert_eq!(lower, sorted, "layer first, then NodeId");

        assert!(Search::filter("", &graph).is_empty());
        assert!(Search::filter("   ", &graph).is_empty());
        assert!(Search::filter("no-such-node", &graph).is_empty());

        // A port name is searchable too, not only the kind.
        let (id, port) = graph.layers[TASK]
            .nodes
            .iter()
            .find_map(|n| n.ports.inputs.first().map(|p| (n.id, p.clone())))
            .expect("some Task node declares an input");
        assert!(
            Search::filter(&port, &graph).contains(&(TASK, id)),
            "the node declaring `{port}` is found by it"
        );

        // Every node of the Task IR is found by its kind tag.
        for node in &graph.layers[TASK].nodes {
            let hits = Search::filter(node.kind, &graph);
            assert!(
                hits.contains(&(TASK, node.id)),
                "{} #{} is findable by its kind",
                node.kind,
                node.id.0
            );
        }

        let mut search = Search::default();
        search.set_hits(lower.clone());
        assert_eq!(search.current(), None, "nothing is current until Enter");
        let mut seen = Vec::new();
        for _ in 0..lower.len() {
            seen.push(search.advance().expect("a hit"));
        }
        assert_eq!(seen, lower);
        assert_eq!(search.advance(), Some(lower[0]), "Enter cycles");

        search.set_hits(Vec::new());
        assert_eq!(search.advance(), None);
        assert_eq!(search.summary(), "");
    }

    /// The edit-mode source finds the same nodes as the read-only one, plus the ones added
    /// since — which is the reason it exists rather than searching the layered view.
    #[test]
    fn the_edit_session_is_searched_too() {
        let task = es_ir::serial::task_from_toml(&read("task.toml")).expect("task");
        let session = EditSession::new(EditIr::Task(task), Layout::default());
        let graph = demo();
        for node in &graph.layers[TASK].nodes {
            let hits = Search::filter_session(node.kind, &session);
            assert!(hits.contains(&(TASK, node.id)), "{} is findable", node.kind);
        }
        assert!(Search::filter_session("", &session).is_empty());
        assert!(Search::filter_session("NOT-A-KIND", &session).is_empty());
        // Searching by the printed label, which is what the canvas shows.
        let (first, _) = session.graph.nodes()[0];
        let by_label = Search::filter_session(&format!("#{}", first.0), &session);
        assert!(by_label.contains(&(TASK, first)), "{by_label:?}");
        assert!(!by_label.contains(&(TASK, NodeId(9999))));
    }
}
