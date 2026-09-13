//! Properties the Safety Plane must hold for *any* input, plus the allocation-zero check.
//!
//! The scenario suite (`scenarios.rs`) pins specific behaviours; this file pins the two
//! guarantees that must survive inputs nobody enumerated: the output is always finite and
//! inside the envelope, and `validate` never allocates (spec 9.6, P17).

use es_core::PhysTick;
use es_core::TickRate;
use es_ir::deployment::{
    ActionContract, ActionSpace, Deadlines, DeploymentIr, ExecutionMode, FallbackPolicy, Limit,
    Micros, RateLimit, RateSpec, RobotRef, RobotTarget, SafetyEnvelope, Watchdog, WatchdogSet,
    Workspace, SCHEMA_VERSION,
};
use es_safety::{ActionChunk, ActionSource, FallbackKind, SafetyPlane, ViolationKind};
use proptest::prelude::*;

const NJ: usize = 3;
const H: usize = 4;

/// A deliberately wide envelope. Widening is the only way to give a test room: nothing here
/// can switch a constraint off (INV-12).
fn ir(fallback: FallbackPolicy, watchdogs: Vec<Watchdog>) -> DeploymentIr {
    DeploymentIr {
        schema_version: SCHEMA_VERSION,
        robot: RobotRef {
            name: "prop-rig".into(),
            target: RobotTarget::Simulated {
                scene: "scenes/prop.usd".into(),
            },
            n_joints: NJ,
        },
        action: ActionContract {
            space: ActionSpace::JointPosition,
            dim: NJ,
            horizon: H,
            execute_chunk: 3,
        },
        safety: SafetyEnvelope {
            position: vec![
                Limit {
                    lower: -2.0,
                    upper: 2.0
                };
                NJ
            ],
            position_soft_margin: vec![0.1; NJ],
            velocity_max: vec![8.0; NJ],
            acceleration_max: vec![400.0; NJ],
            torque_max: vec![50.0; NJ],
            jerk_max: None,
            action_rate: RateLimit {
                first_diff_max: vec![0.5; NJ],
                second_diff_max: vec![1.0; NJ],
            },
            workspace: Workspace::Box {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            ee_velocity_max: 4.0,
            min_self_distance: 0.01,
            min_env_distance: 0.01,
            contact_force_max: 100.0,
        },
        execution: ExecutionMode::RecedingHorizon,
        deadlines: Deadlines {
            observation_age: Micros(100_000),
            inference_budget: Micros(50_000),
            actuation_budget: Micros(5_000),
        },
        watchdogs: WatchdogSet(watchdogs),
        fallback,
        rate: RateSpec {
            control: TickRate::hz(100),
            inference: TickRate::hz(10),
        },
    }
}

fn plane(fallback: FallbackPolicy) -> SafetyPlane<NJ, H> {
    SafetyPlane::from_ir(&ir(fallback, vec![Watchdog::ChunkUnderrun])).expect("valid config")
}

fn assert_inside(q: [f64; NJ], at: &str) {
    for (j, v) in q.iter().enumerate() {
        assert!(v.is_finite(), "{at}: q[{j}] = {v} is not finite");
        assert!(
            (-2.0..=2.0).contains(v),
            "{at}: q[{j}] = {v} is outside the hard position limits"
        );
    }
}

/// Any `f64` at all: normal, subnormal, huge, zero, and the three non-finite values.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        6 => -1e3f64..1e3f64,
        2 => prop_oneof![Just(0.0), Just(-0.0), Just(f64::MIN_POSITIVE), Just(1e300), Just(-1e300)],
        2 => prop_oneof![
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
        ],
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// INV-13 in executable form: whatever comes in, an executable action comes out.
    #[test]
    fn output_is_always_finite_and_inside_the_envelope(
        rows in prop::collection::vec(
            prop::collection::vec(any_f64(), NJ), H),
        valids in prop::collection::vec(0usize..=H, 1..12),
        ages in prop::collection::vec(0u64..500_000, 1..12),
        tick_gaps in prop::collection::vec(0u64..20, 1..12),
    ) {
        let mut actions = [[0.0; NJ]; H];
        for (dst, src) in actions.iter_mut().zip(&rows) {
            dst.copy_from_slice(src);
        }
        let mut p = plane(FallbackPolicy::ZeroVelocity);
        let mut tick = 0u64;
        let steps = valids.len().min(ages.len()).min(tick_gaps.len());
        for i in 0..steps {
            tick += tick_gaps[i];
            let chunk = ActionChunk::new(actions, valids[i], ExecutionMode::RecedingHorizon);
            let out = p.validate(&chunk, Micros(ages[i]), PhysTick(tick));
            assert_inside(out.q, &format!("step {i}"));
        }
        // Counters stay coherent: every step is accounted for exactly once.
        prop_assert_eq!(p.counters().steps, steps as u64);
        prop_assert!(p.counters().envelope_violation_rate() <= 1.0);
    }
}

#[test]
fn hot_path_allocates_nothing() {
    let mut p = plane(FallbackPolicy::HoldPosition);
    let chunk = ActionChunk::new(
        [[0.05; NJ], [0.10; NJ], [f64::NAN; NJ], [0.20; NJ]],
        4,
        ExecutionMode::RecedingHorizon,
    );
    // Warm up outside the assertion so the first call's lazily-initialised thread locals (if
    // any) are not counted as the hot path's doing.
    p.validate(&chunk, Micros(1_000), PhysTick(0));
    es_core::alloc_count::assert_no_alloc(|| {
        for t in 1..10_000u64 {
            p.heartbeat(PhysTick(t));
            p.sensor_seen("wrist_cam", PhysTick(t));
            let out = p.validate(&chunk, Micros(1_000), PhysTick(t));
            assert_inside(out.q, "alloc loop");
        }
    });
    assert!(es_core::alloc_count::counting_enabled());
}

#[test]
fn zero_velocity_decelerates_within_the_acceleration_limit() {
    let mut p = plane(FallbackPolicy::ZeroVelocity);
    // Build up some motion with a valid chunk, then starve the buffer.
    let moving = ActionChunk::new(
        [[0.3; NJ], [0.6; NJ], [0.9; NJ], [0.0; NJ]],
        3,
        ExecutionMode::RecedingHorizon,
    );
    for t in 0..3u64 {
        p.validate(&moving, Micros(1_000), PhysTick(t));
    }
    let empty = ActionChunk::<NJ, H>::empty(ExecutionMode::RecedingHorizon);
    let mut prev = p.last_safe_action();
    let mut last_step = f64::MAX;
    for t in 3..40u64 {
        let out = p.validate(&empty, Micros(1_000), PhysTick(t));
        assert_eq!(
            out.source,
            ActionSource::Fallback(FallbackKind::ZeroVelocity)
        );
        let step = (out.q[0] - prev[0]).abs();
        assert!(step <= last_step + 1e-12, "deceleration must be monotone");
        last_step = step;
        prev = out.q;
    }
    // It comes to rest rather than drifting.
    assert!(
        last_step < 1e-9,
        "did not reach rest, last step {last_step}"
    );
}

#[test]
fn handoff_holds_position_and_flags_the_caller() {
    let mut p = plane(FallbackPolicy::HandoffController { id: "pid".into() });
    let hold = p.last_safe_action();
    let out = p.validate(
        &ActionChunk::empty(ExecutionMode::RecedingHorizon),
        Micros(1_000),
        PhysTick(0),
    );
    assert_eq!(
        out.source,
        ActionSource::Fallback(FallbackKind::HandoffController)
    );
    assert_eq!(out.q.map(f64::to_bits), hold.map(f64::to_bits));
    assert!(out.events.contains(ViolationKind::ChunkUnderrun));
}

#[test]
fn nan_rejection_is_unconditional() {
    // No watchdog is configured at all, yet a NaN action still falls back (INV-12): there is
    // no `NanInf` entry to leave out.
    let cfg = ir(FallbackPolicy::HoldPosition, Vec::new());
    let mut p = SafetyPlane::<NJ, H>::from_ir(&cfg).unwrap();
    let chunk = ActionChunk::new(
        [[f64::NAN; NJ], [0.1; NJ], [0.1; NJ], [0.1; NJ]],
        4,
        ExecutionMode::RecedingHorizon,
    );
    let out = p.validate(&chunk, Micros(1_000), PhysTick(0));
    assert_eq!(
        out.source,
        ActionSource::Fallback(FallbackKind::HoldPosition)
    );
    assert!(out.events.contains(ViolationKind::NonFinite));
    assert_inside(out.q, "nan");
}

#[test]
fn config_errors_are_reported_not_papered_over() {
    // Joint count disagreeing with NJ is a configuration error, never a silently resized plane.
    let cfg = ir(FallbackPolicy::HoldPosition, vec![Watchdog::ChunkUnderrun]);
    assert!(SafetyPlane::<4, H>::from_ir(&cfg).is_err());
    assert!(SafetyPlane::<NJ, 8>::from_ir(&cfg).is_err());

    let mut bad = cfg.clone();
    bad.safety.velocity_max[1] = f64::NAN;
    assert!(SafetyPlane::<NJ, H>::from_ir(&bad).is_err());

    let mut dup = cfg;
    dup.watchdogs = WatchdogSet(vec![
        Watchdog::EnvelopeViolationRate {
            window: 100_000,
            max_frac: 0.5,
        },
        Watchdog::ChunkUnderrun,
    ]);
    assert!(SafetyPlane::<NJ, H>::from_ir(&dup).is_err());
}

#[test]
fn determinism_two_planes_same_inputs_same_outputs() {
    let cfg = ir(FallbackPolicy::ZeroVelocity, vec![Watchdog::ChunkUnderrun]);
    let mut a = SafetyPlane::<NJ, H>::from_ir(&cfg).unwrap();
    let mut b = SafetyPlane::<NJ, H>::from_ir(&cfg).unwrap();
    let chunk = ActionChunk::new(
        [[9.0; NJ], [-9.0; NJ], [f64::INFINITY; NJ], [0.0; NJ]],
        4,
        ExecutionMode::RecedingHorizon,
    );
    for t in 0..50u64 {
        let x = a.validate(&chunk, Micros(t * 100), PhysTick(t));
        let y = b.validate(&chunk, Micros(t * 100), PhysTick(t));
        assert_eq!(x.q.map(f64::to_bits), y.q.map(f64::to_bits), "tick {t}");
        assert_eq!(x.source, y.source);
        assert_eq!(x.events, y.events);
    }
}
