//! Black-box oracle for `docs/packets/M3/W4-domain-gap.md`.
//!
//! `es-eval` cannot depend on `es-data` (both are layer 10, spec §4.2 rule 1 forbids
//! same-layer deps — `cargo xtask layering` checks dev-dependencies too), so unlike most
//! `LeRobot`-adjacent fixtures in this repo, these datasets are not built with
//! `LeRobotWriter`; they are built directly as [`GapInput`], which is exactly the type the
//! `es gap` CLI (`crates/es/src/cmd/gap.rs`, which *can* depend on both crates) produces from
//! a real `LeRobotDataset`. See `docs/design/domain-gap.md` §1 for why.

use std::collections::BTreeMap;

use es_eval::domain_gap::{DomainGap, EpisodeSummary, FeatureSamples, GapInput, GapOptions};
use es_ir::evaluation::MetricValue;

fn channel(dims: usize, values: Vec<f64>) -> FeatureSamples {
    FeatureSamples::new(dims, values)
}

#[test]
fn identical_datasets_are_not_flagged() {
    let mut channels = BTreeMap::new();
    let v: Vec<f64> = (0..64).map(|i| f64::from(i).sin()).collect();
    channels.insert("observation.state".to_owned(), channel(1, v));
    let input = GapInput {
        channels,
        episodes: vec![],
    };

    let report = DomainGap::compute(&input, &input, &GapOptions::default()).expect("compute");
    assert_eq!(report.channels.len(), 1);
    assert!(report.channels[0].ks_d < 1e-9, "{report}");
    assert!(!report.has_flagged(), "{report}");
}

#[test]
fn one_shifted_feature_is_flagged_the_rest_are_not() {
    let matched: Vec<f64> = (0..300).map(|i| f64::from(i % 11)).collect();
    let sim_action: Vec<f64> = (0..300).map(|i| f64::from(i % 6)).collect();
    let real_action: Vec<f64> = (0..300).map(|i| f64::from(i % 6) + 20.0).collect();

    let mut sim_ch = BTreeMap::new();
    sim_ch.insert("observation.state".to_owned(), channel(1, matched.clone()));
    sim_ch.insert("action".to_owned(), channel(1, sim_action));
    let mut real_ch = BTreeMap::new();
    real_ch.insert("observation.state".to_owned(), channel(1, matched));
    real_ch.insert("action".to_owned(), channel(1, real_action));

    let sim = GapInput {
        channels: sim_ch,
        episodes: vec![
            EpisodeSummary {
                success: Some(true),
                length: 40,
                envelope_violation: Some(0.0),
            },
            EpisodeSummary {
                success: Some(true),
                length: 42,
                envelope_violation: Some(0.02),
            },
        ],
    };
    let real = GapInput {
        channels: real_ch,
        episodes: vec![
            EpisodeSummary {
                success: Some(false),
                length: 50,
                envelope_violation: Some(0.2),
            },
            EpisodeSummary {
                success: Some(true),
                length: 55,
                envelope_violation: Some(0.3),
            },
        ],
    };

    let report = DomainGap::compute(&sim, &real, &GapOptions::default()).expect("compute");
    let state = report
        .channels
        .iter()
        .find(|c| c.name == "observation.state")
        .expect("state channel present");
    let action = report
        .channels
        .iter()
        .find(|c| c.name == "action")
        .expect("action channel present");
    assert!(
        !state.flagged,
        "unshifted channel should not flag: {report}"
    );
    assert!(action.flagged, "shifted channel should flag: {report}");
    assert_eq!(report.suspects.len(), 1);
    assert_eq!(report.suspects[0].channel, "action");
    assert!(report.has_flagged());

    // Episode-level numbers are read, not invented, since every episode here has a value.
    assert_eq!(report.episodes.sim_success_rate, MetricValue::Scalar(1.0));
    assert_eq!(report.episodes.real_success_rate, MetricValue::Scalar(0.5));
}

#[test]
fn feature_present_on_only_one_side_is_listed_unmatched_not_scored() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let mut sim_ch = BTreeMap::new();
    sim_ch.insert("observation.state".to_owned(), channel(1, v.clone()));
    sim_ch.insert("observation.sim_only".to_owned(), channel(1, v.clone()));
    let mut real_ch = BTreeMap::new();
    real_ch.insert("observation.state".to_owned(), channel(1, v.clone()));
    real_ch.insert("observation.real_only".to_owned(), channel(1, v));

    let sim = GapInput {
        channels: sim_ch,
        episodes: vec![],
    };
    let real = GapInput {
        channels: real_ch,
        episodes: vec![],
    };

    let report = DomainGap::compute(&sim, &real, &GapOptions::default()).expect("compute");
    assert_eq!(report.channels.len(), 1);
    assert_eq!(
        report.unmatched_sim,
        vec!["observation.sim_only".to_owned()]
    );
    assert_eq!(
        report.unmatched_real,
        vec!["observation.real_only".to_owned()]
    );
}

#[test]
fn episode_columns_absent_on_both_sides_stay_unavailable() {
    let empty = GapInput {
        channels: BTreeMap::new(),
        episodes: vec![EpisodeSummary::default(), EpisodeSummary::default()],
    };
    let report = DomainGap::compute(&empty, &empty, &GapOptions::default()).expect("compute");
    assert!(matches!(
        report.episodes.sim_success_rate,
        MetricValue::Unavailable { .. }
    ));
    assert!(matches!(
        report.episodes.real_envelope_violation_rate,
        MetricValue::Unavailable { .. }
    ));
}
