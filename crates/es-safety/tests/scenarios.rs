//! Safety Plane violation scenario suite — spec 28.7 gate 8.
//!
//! Each `tests/fixtures/safety/*.json` file carries a complete `DeploymentIr` and a step
//! sequence. The runner replays it and asserts the expected `source`, event set and joint
//! vector per step, then the end-of-run counters (spec 10.3).
//!
//! Fixtures are data, never code: nothing here special-cases a scenario by name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use es_core::PhysTick;
use es_ir::deployment::{DeploymentIr, Micros};
use es_safety::{ActionChunk, ActionSource, SafetyPlane, ViolationKind};
use serde::Deserialize;

const TOL: f64 = 1e-9;

/// JSON has no `NaN`; a string spells the non-finite values the plane must survive.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Num {
    Finite(f64),
    Special(String),
}

impl Num {
    fn get(&self) -> f64 {
        match self {
            Self::Finite(v) => *v,
            Self::Special(s) => match s.as_str() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => panic!("unknown numeric literal `{other}`"),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    #[allow(dead_code)]
    spec: String,
    deployment: DeploymentIr,
    #[serde(default)]
    initial_q: Option<Vec<f64>>,
    #[serde(default)]
    initial_qd: Option<Vec<f64>>,
    steps: Vec<Step>,
    #[serde(default)]
    expect_counters: Option<ExpectCounters>,
}

#[derive(Debug, Deserialize)]
struct Step {
    tick: u64,
    obs_age_us: u64,
    /// Absent means "the runtime still holds the previous chunk" (spec 8.6).
    #[serde(default)]
    chunk: Option<Chunk>,
    /// Whether the controller reported in on this tick.
    #[serde(default = "yes")]
    heartbeat: bool,
    /// Sensors that produced a sample. Absent means every configured sensor did.
    #[serde(default)]
    sensors: Option<Vec<String>>,
    #[serde(default)]
    reset_latch: bool,
    expect: Expect,
}

const fn yes() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct Chunk {
    actions: Vec<Vec<Num>>,
    valid: usize,
    /// The caller's sequence number (spec 8.6, P-M1-R3). Absent means "auto-assign the next
    /// one" — the common case, standing in for a policy invocation that always advances its
    /// own counter. A fixture sets this explicitly only to force a *repeated* seq (simulating
    /// a caller that failed to advance it), which is otherwise untestable from JSON alone.
    #[serde(default)]
    seq: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Expect {
    source: ActionSource,
    #[serde(default)]
    events: Vec<ViolationKind>,
    #[serde(default)]
    q: Option<Vec<f64>>,
}

#[derive(Debug, Default, Deserialize)]
struct ExpectCounters {
    #[serde(default)]
    steps: Option<u64>,
    #[serde(default)]
    fallback_activations: Option<u64>,
    #[serde(default)]
    clamped_steps: Option<u64>,
    /// Per-kind counts. Kinds absent from the map must be zero.
    #[serde(default)]
    violations: Option<BTreeMap<ViolationKind, u64>>,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/safety")
}

fn run<const NJ: usize, const H: usize>(sc: &Scenario) {
    let mut plane = SafetyPlane::<NJ, H>::from_ir(&sc.deployment)
        .unwrap_or_else(|e| panic!("{}: from_ir failed: {e}", sc.name));
    if let (Some(q), Some(qd)) = (&sc.initial_q, &sc.initial_qd) {
        let mut q0 = [0.0; NJ];
        let mut qd0 = [0.0; NJ];
        q0.copy_from_slice(q);
        qd0.copy_from_slice(qd);
        plane.observe_state(&q0, &qd0);
    }

    let mode = sc.deployment.execution;
    let sensors: Vec<String> = sc
        .deployment
        .watchdogs
        .0
        .iter()
        .filter_map(|w| match w {
            es_ir::deployment::Watchdog::SensorDropout { sensor, .. } => Some(sensor.clone()),
            _ => None,
        })
        .collect();

    let mut held = ActionChunk::<NJ, H>::empty(mode);
    // Auto-assigned seq for a step that doesn't pin one explicitly — stands in for a caller
    // (the embedded runtime) that increments its own counter on every policy invocation.
    let mut next_seq = 1u64;
    for (i, step) in sc.steps.iter().enumerate() {
        let now = PhysTick(step.tick);
        let at = format!("{} step {i} (tick {})", sc.name, step.tick);

        if step.reset_latch {
            plane.reset_latch();
        }
        if let Some(c) = &step.chunk {
            let mut actions = [[0.0; NJ]; H];
            assert_eq!(c.actions.len(), H, "{at}: chunk must have H = {H} rows");
            for (row, src) in actions.iter_mut().zip(&c.actions) {
                assert_eq!(src.len(), NJ, "{at}: chunk row must have NJ = {NJ} entries");
                for (v, n) in row.iter_mut().zip(src) {
                    *v = n.get();
                }
            }
            let seq = c.seq.unwrap_or_else(|| {
                let s = next_seq;
                next_seq += 1;
                s
            });
            next_seq = next_seq.max(seq + 1);
            held = ActionChunk::new(actions, c.valid, mode).with_seq(seq);
        }
        if step.heartbeat {
            plane.heartbeat(now);
        }
        let seen = step.sensors.clone().unwrap_or_else(|| sensors.clone());
        for s in &seen {
            plane.sensor_seen(s, now);
        }

        let out = plane.validate(&held, Micros(step.obs_age_us), now);

        assert_eq!(out.source, step.expect.source, "{at}: source");
        let got: Vec<ViolationKind> = out.events.iter().collect();
        let mut want = step.expect.events.clone();
        want.sort_unstable();
        assert_eq!(got, want, "{at}: events");
        assert!(
            out.q.iter().all(|v| v.is_finite()),
            "{at}: output is not finite: {:?}",
            out.q
        );
        if let Some(q) = &step.expect.q {
            assert_eq!(q.len(), NJ, "{at}: expected q must have NJ = {NJ} entries");
            for (j, (a, b)) in out.q.iter().zip(q).enumerate() {
                assert!(
                    (a - b).abs() <= TOL,
                    "{at}: q[{j}] = {a}, expected {b} (full {:?})",
                    out.q
                );
            }
        }
    }

    let Some(want) = &sc.expect_counters else {
        return;
    };
    let c = plane.counters();
    if let Some(n) = want.steps {
        assert_eq!(c.steps, n, "{}: counters.steps", sc.name);
    }
    if let Some(n) = want.fallback_activations {
        assert_eq!(
            c.fallback_activations, n,
            "{}: counters.fallback_activations",
            sc.name
        );
    }
    if let Some(n) = want.clamped_steps {
        assert_eq!(c.clamped_steps, n, "{}: counters.clamped_steps", sc.name);
    }
    if let Some(v) = &want.violations {
        for kind in ViolationKind::ALL {
            let expected = v.get(&kind).copied().unwrap_or(0);
            assert_eq!(
                c.count(kind),
                expected,
                "{}: counters.violations[{kind:?}]",
                sc.name
            );
        }
    }
}

#[test]
fn every_violation_scenario_passes() {
    let dir = fixture_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(
        files.len() >= 12,
        "spec 28.7 gate 8 wants the scenario suite in full; found {}",
        files.len()
    );

    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        let sc: Scenario = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{}: {e}", path.file_name().unwrap().to_string_lossy()));
        assert_eq!(
            sc.name,
            path.file_stem().unwrap().to_string_lossy(),
            "fixture name must match its file name"
        );
        // One arm per (NJ, H) the suite uses. Adding a shape means adding an arm, which is
        // the point: the const generics are part of the configuration contract.
        match (sc.deployment.robot.n_joints, sc.deployment.action.horizon) {
            (3, 4) => run::<3, 4>(&sc),
            (6, 4) => run::<6, 4>(&sc),
            (nj, h) => panic!("{}: no runner for NJ = {nj}, H = {h}", sc.name),
        }
    }
    println!("{} safety scenarios passed", files.len());
}

#[test]
fn every_fallback_and_watchdog_kind_is_covered() {
    // Gate 8 is "the violation scenarios in full", so the suite's coverage is itself asserted.
    let dir = fixture_dir();
    let mut kinds: Vec<ViolationKind> = Vec::new();
    let mut fallbacks: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let sc: Scenario = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for step in &sc.steps {
            for k in &step.expect.events {
                if !kinds.contains(k) {
                    kinds.push(*k);
                }
            }
            if let ActionSource::Fallback(f) = step.expect.source {
                let name = format!("{f:?}");
                if !fallbacks.contains(&name) {
                    fallbacks.push(name);
                }
            }
        }
    }
    for kind in ViolationKind::ALL {
        assert!(kinds.contains(&kind), "no scenario exercises {kind:?}");
    }
    // HandoffController and ZeroVelocity are covered by unit tests in `properties.rs`, where
    // the whole matrix of fallbacks is driven from one envelope.
    for want in ["HoldPosition", "EmergencyStop", "RetractToHome"] {
        assert!(
            fallbacks.iter().any(|f| f == want),
            "no scenario exercises the {want} fallback"
        );
    }
}

// --- packet M7/R1: the `ViolationRate` ring is per episode (spec 9.4) ------------------------

/// The `violation_rate` scenario's deployment: a 3-joint rig whose spec 9.4 rate watchdog
/// reads a 4-step window at `max_frac = 0.5`, and whose soft position limit any large command
/// trips. The fixture is the data; this only borrows its document.
fn violation_rate_deployment() -> DeploymentIr {
    let path = fixture_dir().join("violation_rate.json");
    let sc: Scenario = serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    sc.deployment
}

/// A chunk every row of which is far outside the soft position limit, so every step it drives
/// is clamped and therefore dirty.
fn dirty_chunk<const NJ: usize, const H: usize>(
    deploy: &DeploymentIr,
    seq: u64,
) -> ActionChunk<NJ, H> {
    ActionChunk::new([[100.0; NJ]; H], H, deploy.execution).with_seq(seq)
}

/// Drives `steps` dirty steps through a fresh plane and hands back the plane and the sliding
/// rate it reads at the end.
fn fill_the_ring<const NJ: usize, const H: usize>(
    deploy: &DeploymentIr,
    steps: u64,
) -> (SafetyPlane<NJ, H>, f64) {
    let mut plane = SafetyPlane::<NJ, H>::from_ir(deploy).expect("the plane builds");
    plane.observe_state(&[0.0; NJ], &[0.0; NJ]);
    for tick in 0..steps {
        plane.heartbeat(PhysTick(tick));
        plane.validate(&dirty_chunk(deploy, tick + 1), Micros(0), PhysTick(tick));
    }
    let rate = plane.counters().envelope_violation_rate();
    (plane, rate)
}

/// Packet M7/R1 oracle 1 — **`begin_episode` empties the `ViolationRate` ring, and nothing
/// else** (spec 9.4, the sentence added 2026-09-21).
///
/// The ring is the watchdog's sliding input and spec 10.3 judges a full window or nothing, so
/// a window that straddles an episode boundary is half of one stream and half of another
/// (spec 13.1). Every *sum* is the cell's and survives: INV-12 is "no path disables the
/// plane", and re-arming a watchdog is not disarming it.
#[test]
fn window_is_cleared_at_begin_episode() {
    const NJ: usize = 3;
    const H: usize = 4;
    let deploy = violation_rate_deployment();
    let (mut plane, filled) = fill_the_ring::<NJ, H>(&deploy, 8);
    assert!(filled > 0.0, "the fixture never filled the ring: {filled}");

    let before = *plane.counters();
    plane.begin_episode();
    let after = plane.counters();

    assert_eq!(
        after.envelope_violation_rate().to_bits(),
        0.0f64.to_bits(),
        "begin_episode left {} in the ring",
        after.envelope_violation_rate()
    );
    // Every summed counter is the cell's, not the episode's (INV-12): none of them moves.
    assert_eq!(after.steps, before.steps, "steps");
    assert_eq!(after.clamped_steps, before.clamped_steps, "clamped_steps");
    assert_eq!(after.dirty_steps, before.dirty_steps, "dirty_steps");
    assert_eq!(
        after.fallback_activations, before.fallback_activations,
        "fallback_activations"
    );
    for kind in ViolationKind::ALL {
        assert_eq!(
            after.count(kind),
            before.count(kind),
            "violations[{kind:?}]"
        );
    }
    assert!(!plane.is_latched(), "begin_episode left the latch engaged");
    // `chunk_underrun_rate` is a whole-run ratio of the sums, so it is untouched too.
    assert_eq!(
        after.chunk_underrun_rate().to_bits(),
        before.chunk_underrun_rate().to_bits(),
        "chunk_underrun_rate"
    );
}

/// The same statement as a scenario: a plane that tripped `ViolationRate` in episode `k-1`
/// does not trip on tick 0 of episode `k`.
///
/// Without the ring clear, tick 0 of the new episode reads a full window of the previous
/// episode's dirty steps, trips the watchdog and takes the fallback path — which is exactly
/// the divergence packet M7/T8 measured at `nominal-01` tick 24
/// (`docs/design/evaluation-execution.md` 2.7).
#[test]
fn a_tripped_rate_watchdog_does_not_cross_the_episode_boundary() {
    const NJ: usize = 3;
    const H: usize = 4;
    let deploy = violation_rate_deployment();
    let (mut plane, _) = fill_the_ring::<NJ, H>(&deploy, 4);
    // Episode k-1 ends on a tripped watchdog. A trip is self-limiting -- its own step is not
    // counted into the ring (spec 18.5, P-M3-W1-R7), so the rate oscillates around `max_frac`
    // and the trip is every other step; this walks to one rather than assuming which.
    let mut tripped = false;
    for tick in 4..20 {
        let out = plane.validate(&dirty_chunk(&deploy, tick + 1), Micros(0), PhysTick(tick));
        if out.events.contains(ViolationKind::ViolationRate) {
            tripped = true;
            break;
        }
    }
    assert!(tripped, "the fixture never tripped the rate watchdog");

    plane.begin_episode();
    plane.observe_state(&[0.0; NJ], &[0.0; NJ]);
    plane.heartbeat(PhysTick(0));
    let first = plane.validate(&dirty_chunk(&deploy, 10), Micros(0), PhysTick(0));
    assert!(
        !first.events.contains(ViolationKind::ViolationRate),
        "tick 0 of episode k was judged on episode k-1's steps: {:?}",
        first.events.iter().collect::<Vec<_>>()
    );
    // The step is still clamped and still counted: only the sliding rate was re-armed.
    assert!(
        first.events.contains(ViolationKind::Position),
        "{:?}",
        first.events.iter().collect::<Vec<_>>()
    );
    assert!(plane.counters().count(ViolationKind::Position) > 0);
}
