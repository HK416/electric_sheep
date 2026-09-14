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
use es_safety::{ActionChunk, ActionSource, FallbackKind, SafeAction, SafetyPlane, ViolationKind};
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

// --- The spec 9.4 rate watchdog (P-M3-W1-R1) -------------------------------------------------

/// A plane whose only *configured* watchdog is the rate one, so nothing but `ChunkUnderrun`
/// and `NanInf` (always armed, INV-12) can trip. `window: 8, max_frac: 0.25` means "more than
/// two dirty steps in the last eight".
fn rate_plane(fallback: FallbackPolicy) -> SafetyPlane<NJ, H> {
    SafetyPlane::from_ir(&ir(
        fallback,
        vec![Watchdog::EnvelopeViolationRate {
            window: 8,
            max_frac: 0.25,
        }],
    ))
    .expect("valid config")
}

/// One step commanding `target` on every joint. Each call carries a fresh `seq`, so the plane
/// accepts the chunk and executes row 0 rather than running the previous one off its end.
fn step(p: &mut SafetyPlane<NJ, H>, t: u64, target: f64) -> SafeAction<NJ> {
    let chunk =
        ActionChunk::new([[target; NJ]; H], 1, ExecutionMode::RecedingHorizon).with_seq(t + 1);
    p.validate(&chunk, Micros(1_000), PhysTick(t))
}

/// A step whose every component is `NaN`. `NanInf` is armed unconditionally (INV-12), so this
/// is a genuine violation even on a step the rate watchdog has already failed.
fn nan_step(p: &mut SafetyPlane<NJ, H>, t: u64) -> SafeAction<NJ> {
    step(p, t, f64::NAN)
}

/// A step commanding the pose the plane is already holding at zero velocity: no clamp stage
/// touches it, so it is clean.
fn hold_step(p: &mut SafetyPlane<NJ, H>, t: u64) -> SafeAction<NJ> {
    let target = p.last_safe_action()[0];
    step(p, t, target)
}

/// `1.0` is far outside the per-step velocity budget (`8 rad/s * 10 ms = 0.08`), so the step is
/// clamped — dirty, but a clamp, not a watchdog trip.
fn dirty_step(p: &mut SafetyPlane<NJ, H>, t: u64) -> SafeAction<NJ> {
    let out = step(p, t, 1.0);
    assert_eq!(out.source, ActionSource::Clamped, "step at tick {t}");
    assert!(out.events.contains(ViolationKind::Velocity));
    out
}

/// B-1 of `docs/reviews/M3-W1.md`: a window with fewer than `window` steps in it is not a
/// sample of anything, so it must never trip. The first step here is the HIL cold start — the
/// controller has not answered yet, so there is no chunk — and before the fix it made the rate
/// exactly `1.0`, latching every following step into the fallback.
#[test]
fn a_partial_window_never_trips_the_rate_watchdog() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    let out = p.validate(
        &ActionChunk::empty(ExecutionMode::RecedingHorizon),
        Micros(1_000),
        PhysTick(0),
    );
    assert!(out.events.contains(ViolationKind::ChunkUnderrun));
    // Steps 2..8 fill the rest of the eight-step window. The window is never full, so the
    // watchdog has nothing to judge and every step is the policy's own action.
    for t in 1..8u64 {
        let out = hold_step(&mut p, t);
        assert_eq!(out.source, ActionSource::Policy, "step {}", t + 1);
        assert!(
            !out.events.contains(ViolationKind::ViolationRate),
            "step {}",
            t + 1
        );
    }
    // The eighth step is the one that fills the window, and only then does the rate become a
    // number: one dirty step out of eight, not out of one.
    assert!((p.counters().envelope_violation_rate() - 0.125).abs() < 1e-12);
}

/// Once the window *is* full the watchdog judges it, at the configured threshold and not
/// before: three dirty steps out of eight is `0.375 > 0.25`, two is `0.25` and is not.
#[test]
fn a_full_window_trips_at_the_threshold() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    for t in 0..8u64 {
        assert_eq!(hold_step(&mut p, t).source, ActionSource::Policy);
    }
    // Dirty steps 9, 10 and 11: the window they are judged against holds 0, 1 and 2 dirty
    // steps respectively, so none of them trips.
    for t in 8..11u64 {
        let out = dirty_step(&mut p, t);
        assert!(
            !out.events.contains(ViolationKind::ViolationRate),
            "step {} tripped early",
            t + 1
        );
    }
    // Step 12 reads 3/8 and trips.
    let out = hold_step(&mut p, 11);
    assert!(out.events.contains(ViolationKind::ViolationRate));
    assert_eq!(
        out.source,
        ActionSource::Fallback(FallbackKind::HoldPosition)
    );
}

/// The rate is over the last `window` steps and nothing older: a dirty step that has slid out
/// stops counting, so a run under the threshold recovers on its own.
#[test]
fn the_rate_falls_back_out_of_the_window() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    for t in 0..8u64 {
        hold_step(&mut p, t);
    }
    dirty_step(&mut p, 8);
    // 1/8 is under the threshold, so the plane keeps executing the policy while the bit ages.
    for t in 9..17u64 {
        let out = hold_step(&mut p, t);
        assert_eq!(out.source, ActionSource::Policy, "step {}", t + 1);
        assert!(!out.events.contains(ViolationKind::ViolationRate));
    }
    // Eight clean steps later the bit is gone and the rate reads zero, not the run's history.
    assert!(p.counters().envelope_violation_rate() < 1e-12);
    assert_eq!(p.counters().dirty_steps, 1);
}

/// P-M3-W1-R7, replacing `a_tripped_rate_watchdog_does_not_release_itself`, which pinned the
/// bug this test now forbids: the watchdog's own fallback steps used to refill the window with
/// its own echo, so the trip was a permanent, invisible latch. The window now records a step as
/// dirty only for reasons other than `ViolationRate`, so once the genuine violations age out the
/// plane goes back to executing the policy — within `window` steps of the last real one
/// (spec 18.5: a fallback is normal behaviour, not a failure).
#[test]
fn a_tripped_rate_watchdog_releases_after_a_clean_window() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    for t in 0..8u64 {
        hold_step(&mut p, t);
    }
    for t in 8..11u64 {
        dirty_step(&mut p, t);
    }
    // Step 12 reads 3/8 and trips; the last genuine violation was step 11.
    let out = hold_step(&mut p, 11);
    assert!(out.events.contains(ViolationKind::ViolationRate));
    assert_eq!(
        out.source,
        ActionSource::Fallback(FallbackKind::HoldPosition)
    );
    // Every step from here is clean, so the trip must be gone by step 19 — `window` steps
    // after the last genuine violation — at the very latest.
    let mut released = None;
    for t in 12..19u64 {
        let out = hold_step(&mut p, t);
        if out.events.contains(ViolationKind::ViolationRate) {
            assert_eq!(
                out.source,
                ActionSource::Fallback(FallbackKind::HoldPosition)
            );
            continue;
        }
        assert_eq!(out.source, ActionSource::Policy, "step {}", t + 1);
        released = Some(t);
        break;
    }
    let released = released.expect("the watchdog never released itself");
    // And it stays released: nothing it does afterwards puts a bit back in the window.
    for t in released + 1..40u64 {
        let out = hold_step(&mut p, t);
        assert_eq!(out.source, ActionSource::Policy, "step {}", t + 1);
        assert!(!out.events.contains(ViolationKind::ViolationRate));
    }
    assert!(p.counters().envelope_violation_rate() < 1e-12);
}

/// The other half of R7: the window must stay blind only to the watchdog's *own* event. A
/// policy that keeps emitting `NaN` keeps tripping the always-armed `NanInf` watchdog, and that
/// event is recorded on the fallback step it causes, so the rate stays over the bound and the
/// plane stays in the fallback for as long as the violations continue.
#[test]
fn a_real_violation_during_a_trip_still_counts() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    for t in 0..8u64 {
        hold_step(&mut p, t);
    }
    for t in 8..11u64 {
        dirty_step(&mut p, t);
    }
    assert!(hold_step(&mut p, 11)
        .events
        .contains(ViolationKind::ViolationRate));
    for t in 12..42u64 {
        let out = nan_step(&mut p, t);
        assert!(
            out.events.contains(ViolationKind::NonFinite),
            "step {}",
            t + 1
        );
        assert!(
            out.events.contains(ViolationKind::ViolationRate),
            "step {} released the watchdog while it was still being violated",
            t + 1
        );
    }
}

/// The guard against "fixing" the spec 10.3 counters along with the watchdog's input ring. Only
/// the ring changed: every fallback step the watchdog causes is still one `fallback_activation`,
/// one `ViolationRate` violation and one `dirty_step`, which is what `es-eval`'s episode-level
/// `envelope_violation_rate` (`dirty_steps / steps`) reads.
#[test]
fn the_metric_counters_still_count_every_fallback() {
    let mut p = rate_plane(FallbackPolicy::HoldPosition);
    for t in 0..8u64 {
        hold_step(&mut p, t);
    }
    for t in 8..11u64 {
        dirty_step(&mut p, t);
    }
    let mut fallbacks = 0;
    for t in 11..24u64 {
        if hold_step(&mut p, t)
            .events
            .contains(ViolationKind::ViolationRate)
        {
            fallbacks += 1;
        }
    }
    assert!(fallbacks > 0, "the watchdog never tripped");
    let c = p.counters();
    assert_eq!(c.fallback_activations, fallbacks);
    assert_eq!(c.count(ViolationKind::ViolationRate), fallbacks);
    assert_eq!(c.clamped_steps, 3);
    assert_eq!(c.dirty_steps, fallbacks + 3);
    assert_eq!(c.steps, 24);
}

/// The blocker's worst case: with `FallbackPolicy::EmergencyStop` the rate watchdog's trip
/// latches the e-stop (`plane.rs`'s step 4a), and one clamped step used to be enough to trip
/// it on the very next step. A single clamp is not a rate.
///
/// The first step is a *clamp*, not the packet's `ChunkUnderrun`: that watchdog is armed
/// unconditionally, so an empty first chunk latches the e-stop by itself and would prove
/// nothing about the rate window.
#[test]
fn an_estop_rate_watchdog_does_not_latch_on_step_two() {
    let mut p = rate_plane(FallbackPolicy::EmergencyStop);
    dirty_step(&mut p, 0);
    assert!(!p.is_latched(), "a clamp is not a watchdog trip");
    let out = hold_step(&mut p, 1);
    assert!(!out.events.contains(ViolationKind::ViolationRate));
    assert_eq!(out.source, ActionSource::Policy);
    assert!(
        !p.is_latched(),
        "one dirty step out of a window of 8 latched the e-stop"
    );
}

/// The one real latch survives R7 (spec 9.4 attaches "latch" to `EmergencyStop` and to nothing
/// else, INV-12): a genuine rate trip on a full window still engages the e-stop, every later
/// step short-circuits on it, and only `reset_latch()` clears it.
#[test]
fn an_estop_rate_trip_still_latches() {
    let mut p = rate_plane(FallbackPolicy::EmergencyStop);
    for t in 0..8u64 {
        hold_step(&mut p, t);
    }
    for t in 8..11u64 {
        dirty_step(&mut p, t);
    }
    assert!(!p.is_latched(), "a clamp is not a watchdog trip");
    let out = hold_step(&mut p, 11);
    assert!(out.events.contains(ViolationKind::ViolationRate));
    assert_eq!(
        out.source,
        ActionSource::Fallback(FallbackKind::EmergencyStop)
    );
    assert!(p.is_latched(), "a genuine rate trip must latch the e-stop");
    // Long past the window that tripped it, the latch is still the latch.
    for t in 12..40u64 {
        let out = hold_step(&mut p, t);
        assert_eq!(
            out.source,
            ActionSource::Fallback(FallbackKind::EmergencyStop),
            "step {}",
            t + 1
        );
        assert!(out.events.contains(ViolationKind::EstopLatched));
    }
    assert!(p.is_latched());
    p.reset_latch();
    assert!(!p.is_latched());
}
