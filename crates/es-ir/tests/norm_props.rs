//! Appendix B.7: the five normalization invariants, which are gate 2 of spec 28.7.
//!
//! Run with `cargo test -p es-ir --features testing norm`. Without the feature the
//! generators do not exist, so the file compiles to nothing rather than failing to build.

#![cfg(feature = "testing")]

use es_ir::hash::canonical_hash;
use es_ir::norm::{canon_learning, canon_observation, canon_task};
use es_ir::serial::{parse_esgraph, write_esgraph, AnyIr, Layout};
use es_ir::testing::{
    arbitrary_deployment_ir, arbitrary_evaluation_ir, arbitrary_layout, arbitrary_learning_graph,
    arbitrary_observation_ir, arbitrary_task_ir, arbitrary_unit, deployment_edits,
    evaluation_edits, learning_edits, observation_edits, shuffle_learning_ids,
    shuffle_observation_ids, shuffle_task_ids, task_edits,
};
use es_ir::NodeId;
use proptest::prelude::*;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::BTreeMap;

/// `arbitrary_layout` yields the raw position map; the `.eslayout` sidecar itself is
/// `serial::Layout` (spec 5.1 rule 7).
fn to_layout(positions: BTreeMap<NodeId, (f32, f32)>) -> Layout {
    Layout {
        positions: positions
            .into_iter()
            .map(|(id, (x, y))| (id, [x, y]))
            .collect(),
        ..Layout::default()
    }
}

/// `serde_json` is built with `float_roundtrip` workspace-wide, so an `f64` survives exactly.
fn json_round_trip<T: Serialize + DeserializeOwned>(v: &T) -> T {
    serde_json::from_str(&serde_json::to_string(v).expect("serialize")).expect("deserialize")
}

// --- B.7.1 hash(canon(g)) == hash(canon(shuffle_ids(g))) ----------------------------------
//
// Deployment IR and Evaluation IR carry no graph and therefore no node ids, so this property
// is vacuous for them.

proptest! {
    #[test]
    fn norm_hash_independent_of_node_ids_task(ir in arbitrary_task_ir(), seed in any::<u64>()) {
        let shuffled = shuffle_task_ids(&ir, seed);
        prop_assert_eq!(
            canon_task(&ir).unwrap().task_hash().unwrap(),
            canon_task(&shuffled).unwrap().task_hash().unwrap()
        );
    }

    #[test]
    fn norm_hash_independent_of_node_ids_observation(
        ir in arbitrary_observation_ir(),
        seed in any::<u64>(),
    ) {
        let shuffled = shuffle_observation_ids(&ir, seed);
        prop_assert_eq!(
            canon_observation(&ir).unwrap().observation_hash().unwrap(),
            canon_observation(&shuffled).unwrap().observation_hash().unwrap()
        );
    }

    #[test]
    fn norm_hash_independent_of_node_ids_learning(
        g in arbitrary_learning_graph(),
        seed in any::<u64>(),
    ) {
        let shuffled = shuffle_learning_ids(&g, seed);
        prop_assert_eq!(
            canon_learning(&g).unwrap().learning_hash().unwrap(),
            canon_learning(&shuffled).unwrap().learning_hash().unwrap()
        );
    }

    /// Spec 11.2: `*_graph_hash` is the authoring identity, so the *only* hash a relabelling
    /// is allowed to move is that one.
    #[test]
    fn norm_task_graph_hash_sees_node_ids(ir in arbitrary_task_ir(), seed in any::<u64>()) {
        let shuffled = shuffle_task_ids(&ir, seed);
        prop_assert_eq!(ir.task_hash().unwrap(), shuffled.task_hash().unwrap());
        prop_assert_ne!(
            ir.task_graph_hash().unwrap(),
            shuffled.task_graph_hash().unwrap()
        );
    }
}

// --- B.7.2 hash(canon(g)) == hash(canon(move_ui(g))) --------------------------------------
//
// Layout lives in an `.eslayout` sidecar (spec 5.1 rule 7), never in the IR: writing a graph
// with an arbitrary layout attached to a `.esgraph` / `.eslayout` pair and parsing the pair
// back must recover a document that hashes identically to the original, whatever the layout
// says — `hash(parse_esgraph(write_esgraph(&g, &layout))) == hash(&g)`.

proptest! {
    #[test]
    fn norm_hash_independent_of_ui(
        ir in arbitrary_task_ir(),
        obs in arbitrary_observation_ir(),
        lrn in arbitrary_learning_graph(),
        positions in arbitrary_layout(),
    ) {
        let layout = to_layout(positions);

        let (graph_toml, layout_toml) = write_esgraph(&AnyIr::Task(ir.clone()), Some(&layout)).unwrap();
        let (parsed, _) = parse_esgraph(&graph_toml, layout_toml.as_deref()).unwrap();
        let AnyIr::Task(parsed) = parsed else { unreachable!("wrote a Task IR") };
        prop_assert_eq!(parsed.task_hash().unwrap(), ir.task_hash().unwrap());
        prop_assert_eq!(
            canonical_hash(&parsed.graph).unwrap(),
            canonical_hash(&ir.graph).unwrap()
        );

        let (graph_toml, layout_toml) =
            write_esgraph(&AnyIr::Observation(obs.clone()), Some(&layout)).unwrap();
        let (parsed, _) = parse_esgraph(&graph_toml, layout_toml.as_deref()).unwrap();
        let AnyIr::Observation(parsed) = parsed else { unreachable!("wrote an Observation IR") };
        prop_assert_eq!(parsed.observation_hash().unwrap(), obs.observation_hash().unwrap());

        let (graph_toml, layout_toml) =
            write_esgraph(&AnyIr::Learning(lrn.clone()), Some(&layout)).unwrap();
        let (parsed, _) = parse_esgraph(&graph_toml, layout_toml.as_deref()).unwrap();
        let AnyIr::Learning(parsed) = parsed else { unreachable!("wrote a Learning IR") };
        prop_assert_eq!(parsed.learning_hash().unwrap(), lrn.learning_hash().unwrap());
    }
}

// --- B.7.3 hash(canon(deser(ser(g)))) == hash(canon(g)) -----------------------------------

proptest! {
    #[test]
    fn norm_roundtrip_preserves_hash_task(ir in arbitrary_task_ir()) {
        let back = json_round_trip(&ir);
        prop_assert_eq!(back.task_hash().unwrap(), ir.task_hash().unwrap());
        prop_assert_eq!(back.task_graph_hash().unwrap(), ir.task_graph_hash().unwrap());
    }

    #[test]
    fn norm_roundtrip_preserves_hash_observation(ir in arbitrary_observation_ir()) {
        prop_assert_eq!(
            json_round_trip(&ir).observation_hash().unwrap(),
            ir.observation_hash().unwrap()
        );
    }

    #[test]
    fn norm_roundtrip_preserves_hash_learning(g in arbitrary_learning_graph()) {
        let back = json_round_trip(&g);
        prop_assert_eq!(back.learning_hash().unwrap(), g.learning_hash().unwrap());
        prop_assert_eq!(back.policy_hash().unwrap(), g.policy_hash().unwrap());
    }

    #[test]
    fn norm_roundtrip_preserves_hash_deployment(ir in arbitrary_deployment_ir()) {
        prop_assert_eq!(
            json_round_trip(&ir).deployment_hash().unwrap(),
            ir.deployment_hash().unwrap()
        );
    }

    #[test]
    fn norm_roundtrip_preserves_hash_evaluation(ir in arbitrary_evaluation_ir()) {
        prop_assert_eq!(
            json_round_trip(&ir).evaluation_hash().unwrap(),
            ir.evaluation_hash().unwrap()
        );
    }
}

// --- B.7.4 hash(canon(g)) != hash(canon(change_any_param(g))) -----------------------------

proptest! {
    #[test]
    fn norm_param_change_changes_hash_task(
        ir in arbitrary_task_ir(),
        pick in any::<prop::sample::Index>(),
    ) {
        let edits = task_edits(&ir);
        let edited = pick.get(&edits).clone();
        prop_assume!(edited != ir);
        prop_assert_ne!(
            canon_task(&ir).unwrap().task_hash().unwrap(),
            canon_task(&edited).unwrap().task_hash().unwrap()
        );
    }

    #[test]
    fn norm_param_change_changes_hash_observation(
        ir in arbitrary_observation_ir(),
        pick in any::<prop::sample::Index>(),
    ) {
        let edits = observation_edits(&ir);
        let edited = pick.get(&edits).clone();
        prop_assume!(edited != ir);
        prop_assert_ne!(
            canon_observation(&ir).unwrap().observation_hash().unwrap(),
            canon_observation(&edited).unwrap().observation_hash().unwrap()
        );
    }

    #[test]
    fn norm_param_change_changes_hash_learning(
        g in arbitrary_learning_graph(),
        pick in any::<prop::sample::Index>(),
    ) {
        let edits = learning_edits(&g);
        let edited = pick.get(&edits).clone();
        prop_assume!(edited != g);
        prop_assert_ne!(
            canon_learning(&g).unwrap().learning_hash().unwrap(),
            canon_learning(&edited).unwrap().learning_hash().unwrap()
        );
    }

    #[test]
    fn norm_param_change_changes_hash_deployment(
        ir in arbitrary_deployment_ir(),
        pick in any::<prop::sample::Index>(),
    ) {
        let edits = deployment_edits(&ir);
        let edited = pick.get(&edits).clone();
        prop_assume!(edited != ir);
        prop_assert_ne!(
            ir.deployment_hash().unwrap(),
            edited.deployment_hash().unwrap()
        );
    }

    #[test]
    fn norm_param_change_changes_hash_evaluation(
        ir in arbitrary_evaluation_ir(),
        pick in any::<prop::sample::Index>(),
    ) {
        let edits = evaluation_edits(&ir);
        let edited = pick.get(&edits).clone();
        prop_assume!(edited != ir);
        prop_assert_ne!(
            ir.evaluation_hash().unwrap(),
            edited.evaluation_hash().unwrap()
        );
    }
}

// --- B.7.5 unit algebra laws ---------------------------------------------------------------

proptest! {
    /// Over the algebraic units only: `docs/design/ir-types.md` keeps `Quaternion`,
    /// `RotationMatrix`, `Normalized`, `Pixel`, `Luminance`, `Depth`, `Token`, `Current` and
    /// `Voltage` opaque precisely so that algebra on them is an error, not a silent result.
    #[test]
    fn norm_unit_algebra_laws(a in arbitrary_unit(), b in arbitrary_unit()) {
        prop_assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap());
        prop_assert_eq!(a.mul(&b).unwrap().div(&b).unwrap(), a.clone());
        prop_assert_eq!(a.div(&b).unwrap().mul(&b).unwrap(), a);
    }

    #[test]
    fn norm_opaque_units_have_no_algebra(a in arbitrary_unit()) {
        use es_ir::types::Unit;
        for opaque in [
            Unit::Quaternion,
            Unit::RotationMatrix,
            Unit::Token,
            Unit::Normalized { lo: -1.0, hi: 1.0 },
        ] {
            let err = a.mul(&opaque).unwrap_err();
            prop_assert_eq!(err.code.as_str(), es_ir::codes::TYPE_010);
        }
    }
}
