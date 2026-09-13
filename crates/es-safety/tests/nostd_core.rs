//! The `no_std` half of the crate, exercised from a `std` test (spec 9.6, Appendix B.4).
//!
//! `--no-default-features --target thumbv7em-none-eabihf` proves it *builds* without `std`;
//! this proves the same code path *behaves*, because the embedded target has no test harness.
//! Everything here goes through `SafetyPlane::from_config`, which is what the deployment
//! target calls and what `from_ir` delegates to (INV-12: one construction path).

// The sanitizer's output is exact by construction (it copies bounds or writes a literal), so
// bit-exact comparison is the property under test.
#![allow(clippy::float_cmp)]

use es_safety::{
    ActionChunk, ActionSource, Envelope, ExecutionMode, Fallback, FallbackKind, Limit, Micros,
    SafetyConfig, SafetyPlane, SensorWatch, Watchdogs, WorkspaceSpec, MAX_SENSORS, SENSOR_NAME_CAP,
};

const NJ: usize = 3;
const H: usize = 4;

fn envelope() -> Envelope<NJ> {
    Envelope {
        hard: [Limit {
            lower: -2.0,
            upper: 2.0,
        }; NJ],
        soft: [Limit {
            lower: -1.0,
            upper: 1.0,
        }; NJ],
        vel_max: [10.0; NJ],
        acc_max: [100.0; NJ],
        tau_max: [50.0; NJ],
        jerk_max: None,
        d1_max: [0.5; NJ],
        d2_max: [0.5; NJ],
        workspace: WorkspaceSpec::Box {
            min: [-1.0; 3],
            max: [1.0; 3],
        },
        ee_velocity_max: 1.0,
        min_self_distance: 0.0,
        min_env_distance: 0.0,
        contact_force_max: 100.0,
        space: es_safety::ActionSpace::JointPosition,
        dt_s: 0.001,
        period_us: 1000,
        execute_chunk: H,
    }
}

fn config() -> SafetyConfig<NJ> {
    SafetyConfig {
        envelope: envelope(),
        watchdogs: Watchdogs::default(),
        fallback: Fallback::stationary(FallbackKind::HoldPosition),
    }
}

fn chunk(v: f64, seq: u64) -> ActionChunk<NJ, H> {
    ActionChunk::new([[v; NJ]; H], H, ExecutionMode::RecedingHorizon).with_seq(seq)
}

#[test]
fn a_config_built_plane_clamps_to_the_soft_limit() {
    let mut p = SafetyPlane::<NJ, H>::from_config(&config());
    p.observe_state(&[0.0; NJ], &[0.0; NJ]);
    // Far outside: the rate limit bites first, and after enough ticks the soft limit does.
    let mut last = [0.0; NJ];
    for t in 0..64u64 {
        let a = p.validate(&chunk(100.0, t + 1), Micros(0), es_core::PhysTick(t));
        assert_ne!(
            a.source,
            ActionSource::Policy,
            "a huge action is never clean"
        );
        last = a.q;
    }
    for q in last {
        assert!(q <= 1.0 + 1e-12, "clamped to the soft upper limit, got {q}");
    }
}

/// INV-12: a hand-written config cannot turn a clamp stage off. A soft limit wider than its
/// hard limit, or a non-finite bound, is *narrowed* by `from_config`, never honoured.
#[test]
fn a_degenerate_config_is_narrowed_not_skipped() {
    let mut cfg = config();
    cfg.envelope.soft = [Limit {
        lower: f64::NEG_INFINITY,
        upper: f64::INFINITY,
    }; NJ];
    cfg.envelope.vel_max = [f64::NAN; NJ];
    cfg.envelope.d1_max = [f64::INFINITY; NJ];
    cfg.envelope.dt_s = 0.0;

    let p = SafetyPlane::<NJ, H>::from_config(&cfg);
    let e = p.envelope();
    for i in 0..NJ {
        assert_eq!(e.soft[i].lower, e.hard[i].lower);
        assert_eq!(e.soft[i].upper, e.hard[i].upper);
        assert_eq!(e.vel_max[i], 0.0, "NaN velocity bound becomes the tightest");
        assert_eq!(e.d1_max[i], 0.0);
    }
    assert!(e.dt_s > 0.0);

    let mut p = p;
    let a = p.validate(&chunk(5.0, 1), Micros(0), es_core::PhysTick(0));
    for q in a.q {
        assert!(q.is_finite() && (-2.0..=2.0).contains(&q));
    }
}

/// Spec 9.6: zero heap allocation — construction included, not just the hot path.
#[test]
fn construction_and_validation_allocate_nothing() {
    if !es_core::alloc_count::counting_enabled() {
        return;
    }
    let cfg = config();
    es_core::alloc_count::assert_no_alloc(|| {
        let mut p = SafetyPlane::<NJ, H>::from_config(&cfg);
        p.observe_state(&[0.0; NJ], &[0.0; NJ]);
        p.heartbeat(es_core::PhysTick(0));
        p.sensor_seen("wrist_cam", es_core::PhysTick(0));
        for t in 0..32u64 {
            let a = p.validate(&chunk(0.1, t + 1), Micros(0), es_core::PhysTick(t));
            assert!(a.q.iter().all(|q| q.is_finite()));
        }
    });
}

/// The fixed-size tables are bounded and refuse to silently merge two sensors.
#[test]
fn watchdog_sensor_table_is_fixed_size() {
    let mut w = Watchdogs::default();
    let long = "x".repeat(SENSOR_NAME_CAP + 1);
    assert!(!w.arm_sensor(&long, Micros(1)), "no truncation");
    for i in 0..MAX_SENSORS {
        assert!(w.arm_sensor(&format!("sensor_{i}"), Micros(1_000)));
    }
    assert!(!w.arm_sensor("one_too_many", Micros(1_000)));
    assert_eq!(w.sensors().len(), MAX_SENSORS);
    assert_eq!(w.sensors()[0].name(), "sensor_0");
    assert!(
        !SensorWatch::EMPTY.matches(""),
        "an empty slot matches nothing"
    );
}

/// Under `std` the crate's IR value types *are* `es-ir`'s, so `validate`'s signature is
/// unchanged and every existing caller compiles (INV-13). This is what keeps the `no_std`
/// shim in `ir_types.rs` honest.
#[test]
fn ir_value_types_are_the_es_ir_types_under_std() {
    fn takes_micros(m: Micros) -> u64 {
        m.0
    }
    assert_eq!(takes_micros(es_ir::deployment::Micros(7)), 7);

    let mode: ExecutionMode = es_ir::deployment::ExecutionMode::RecedingHorizon;
    let limit: Limit = es_ir::deployment::Limit::symmetric(1.0);
    let space: es_safety::ActionSpace = es_ir::deployment::ActionSpace::JointTorque;
    assert_eq!(mode, es_ir::deployment::ExecutionMode::RecedingHorizon);
    assert_eq!(limit.upper, 1.0);
    assert_eq!(space, es_ir::deployment::ActionSpace::JointTorque);
}
