//! [`EmbeddedCore`] driven the way the bare-metal target drives it: two function pointers, a
//! caller-owned scratch buffer, and no allocator (spec 9.6).
//!
//! The `no_std` build is proved by `cargo build -p es-runtime-embedded --no-default-features
//! --target thumbv7em-none-eabihf`; this is the behaviour half, run under `std` because the
//! embedded target has no test harness.

use es_core::PhysTick;
use es_runtime_embedded::{EmbeddedCore, InferFn, ObserveFn};
use es_safety::{
    ActionSpace, Envelope, ExecutionMode, Fallback, FallbackKind, Limit, Micros, SafetyConfig,
    Watchdogs, WorkspaceSpec,
};

const NJ: usize = 3;
const H: usize = 4;
const CAP: usize = 8;

fn config() -> SafetyConfig<NJ> {
    SafetyConfig {
        envelope: Envelope {
            hard: [Limit {
                lower: -10.0,
                upper: 10.0,
            }; NJ],
            soft: [Limit {
                lower: -9.0,
                upper: 9.0,
            }; NJ],
            vel_max: [1e6; NJ],
            acc_max: [1e9; NJ],
            tau_max: [1e6; NJ],
            jerk_max: None,
            d1_max: [1e6; NJ],
            d2_max: [1e6; NJ],
            workspace: WorkspaceSpec::Box {
                min: [-10.0; 3],
                max: [10.0; 3],
            },
            ee_velocity_max: 1.0,
            min_self_distance: 0.0,
            min_env_distance: 0.0,
            contact_force_max: 1.0,
            space: ActionSpace::JointPosition,
            dt_s: 0.001,
            period_us: 1_000,
            execute_chunk: H,
        },
        watchdogs: Watchdogs::default(),
        fallback: Fallback::stationary(FallbackKind::HoldPosition),
    }
}

/// Observation IR stand-in: scale the sensor samples into the policy's input tensor.
fn observe(sensors: &[f32], obs: &mut [f32]) {
    for (o, s) in obs.iter_mut().zip(sensors) {
        *o = s * 0.5;
    }
}

/// Learning IR + policy stand-in: broadcast the first observation component to every row.
fn infer(obs: &[f32], actions: &mut [[f64; NJ]; H]) -> usize {
    let v = f64::from(obs.first().copied().unwrap_or(0.0));
    for row in actions.iter_mut() {
        *row = [v; NJ];
    }
    H
}

/// A policy that failed: zero rows, which the plane reads as a chunk underrun (spec 8.6).
fn infer_nothing(_obs: &[f32], _actions: &mut [[f64; NJ]; H]) -> usize {
    0
}

#[test]
fn the_loop_replans_on_schedule_and_records_every_tick() {
    let mut core =
        EmbeddedCore::<NJ, H, CAP>::from_config(&config(), ExecutionMode::RecedingHorizon, 2);
    core.plane_mut().observe_state(&[0.0; NJ], &[0.0; NJ]);

    let sensors = [4.0f32; NJ];
    let mut obs = [0.0f32; NJ];
    let mut replans = 0;
    for t in 0..6u64 {
        let before = core.chunk_seq();
        let a = core.tick(
            &sensors,
            &mut obs,
            PhysTick(t),
            Micros(0),
            observe as ObserveFn,
            infer as InferFn<NJ, H>,
        );
        if core.chunk_seq() > before {
            replans += 1;
        }
        for q in a.q {
            assert!(
                (q - 2.0).abs() < 1e-12,
                "4.0 * 0.5 reaches the actuator, got {q}"
            );
        }
    }
    // Ticks 0, 2, 4 replan; 1, 3, 5 consume the buffered chunk (spec 8.6).
    assert_eq!(replans, 3);
    assert_eq!(core.chunk_seq(), 3);
    assert_eq!(core.telemetry().len(), 6);
    assert_eq!(
        core.telemetry()
            .drain_since(0)
            .map(|r| r.replanned)
            .collect::<Vec<_>>(),
        vec![true, false, true, false, true, false]
    );
    assert_eq!(core.counters().steps, 6);
}

/// INV-13: an inference that produced nothing still yields an action — the fallback's.
#[test]
fn a_policy_that_returns_no_rows_becomes_the_fallback() {
    let mut core =
        EmbeddedCore::<NJ, H, CAP>::from_config(&config(), ExecutionMode::RecedingHorizon, 1);
    core.plane_mut().observe_state(&[0.25; NJ], &[0.0; NJ]);
    let a = core.tick(
        &[1.0; NJ],
        &mut [0.0f32; NJ],
        PhysTick(0),
        Micros(0),
        observe as ObserveFn,
        infer_nothing as InferFn<NJ, H>,
    );
    assert_eq!(
        a.source,
        es_safety::ActionSource::Fallback(FallbackKind::HoldPosition)
    );
    assert!(a.events.contains(es_safety::ViolationKind::ChunkUnderrun));
    for q in a.q {
        assert!((q - 0.25).abs() < 1e-12, "holds where the robot is");
    }
}

/// The telemetry ring overwrites oldest-first and never grows (spec 9.6).
#[test]
fn the_telemetry_ring_is_bounded() {
    let mut core =
        EmbeddedCore::<NJ, H, CAP>::from_config(&config(), ExecutionMode::RecedingHorizon, 4);
    let mut obs = [0.0f32; NJ];
    for t in 0..(CAP as u64 * 3) {
        core.tick(
            &[1.0; NJ],
            &mut obs,
            PhysTick(t),
            Micros(0),
            observe as ObserveFn,
            infer as InferFn<NJ, H>,
        );
    }
    assert_eq!(core.telemetry().len(), CAP);
    assert_eq!(core.telemetry().capacity(), CAP);
    assert_eq!(core.telemetry().dropped(), CAP as u64 * 2);
}

/// Spec 9.6's "zero heap allocation", for the whole loop rather than just `validate`.
#[test]
fn the_whole_loop_allocates_nothing() {
    if !es_core::alloc_count::counting_enabled() {
        return;
    }
    let cfg = config();
    let sensors = [1.0f32; NJ];
    let mut obs = [0.0f32; NJ];
    es_core::alloc_count::assert_no_alloc(|| {
        let mut core =
            EmbeddedCore::<NJ, H, CAP>::from_config(&cfg, ExecutionMode::RecedingHorizon, 2);
        for t in 0..64u64 {
            let a = core.tick(
                &sensors,
                &mut obs,
                PhysTick(t),
                Micros(0),
                observe as ObserveFn,
                infer as InferFn<NJ, H>,
            );
            assert!(a.q.iter().all(|q| q.is_finite()));
        }
    });
}
