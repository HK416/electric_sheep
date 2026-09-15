//! Packet M5/V6: what the Safety Plane's dynamic stages are measured *against*.
//!
//! `scenarios.rs` pins specific behaviours and `properties.rs` pins what must hold for any
//! input; this file pins the one question both of them left open, and that collection and
//! evaluation answered differently until V6 -- whether `velocity_limit`, `acceleration_limit`
//! and `rate_limit` bound the plane's own commands or the servo's following error.
//!
//! See `docs/design/visible-learning.md` section 7.12 and `docs/packets/M5/V6-envelope-semantics.md`.

// The plane copies commands it does not correct, so bit-exact comparison is the property.
#![allow(clippy::float_cmp)]

use es_core::PhysTick;
use es_safety::{ActionChunk, ActionSource, ExecutionMode, Micros, SafetyPlane, ViolationKind};

const NJ: usize = 6;
const H: usize = 16;

/// The demo's committed Deployment IR, read rather than transcribed: this oracle is about
/// *those* numbers (`velocity_max` 3.0 rad/s, `acceleration_max` 20 rad/s^2, first/second
/// action-rate differences 0.08/0.04 rad at 50 Hz), not about numbers a test invented.
fn demo_plane() -> SafetyPlane<NJ, H> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning/deployment.toml");
    let text = std::fs::read_to_string(&path).expect("the demo deployment is in the repo");
    let ir = es_ir::serial::deployment_from_toml(&text).expect("deployment.toml parses");
    SafetyPlane::from_ir(&ir).expect("the demo envelope builds a plane")
}

/// The control period the demo declares, seconds: `rate.control` is 50 Hz.
const DT: f64 = 0.02;
/// `acceleration_max * dt^2` -- the bound the pre-V6 evaluation path put on the *following
/// error*, and the number design note section 7.10 measured the expert against.
const OLD_BOUND: f64 = 20.0 * DT * DT;

/// `0.9 * min(velocity_max * dt, first_diff_max)` and `0.9 * min(acceleration_max * dt^2,
/// second_diff_max)` -- the pacing `es loop collect --expert` derives from the same document
/// (`crates/es/src/cmd/loop.rs::pace`), a tenth held back so nothing lands on a limit.
const STEP_MAX: f64 = 0.9 * 0.06;
const ACCEL_MAX: f64 = 0.9 * OLD_BOUND;

/// One joint's trapezoidal ramp toward `target`, paced to [`STEP_MAX`] / [`ACCEL_MAX`]: the
/// shape of command stream the envelope is supposed to let through untouched.
fn ramp(from: f64, target: f64, steps: usize) -> Vec<f64> {
    let (mut q, mut v) = (from, 0.0f64);
    (0..steps)
        .map(|_| {
            let remaining = target - q;
            let wanted = remaining.signum() * STEP_MAX.min(remaining.abs());
            v = wanted.clamp(v - ACCEL_MAX, v + ACCEL_MAX);
            q += v;
            q
        })
        .collect()
}

fn chunk(row: [f64; NJ], seq: u64) -> ActionChunk<NJ, H> {
    ActionChunk::new([row; H], H, ExecutionMode::RecedingHorizon).with_seq(seq)
}

/// Packet M5/V6, the oracle this packet exists for (design note `visible-learning.md` section
/// 7.12).
///
/// Spec 9.3's table constrains the *policy output* and every row of it says "clamp", so the
/// quantity `velocity_limit`, `acceleration_limit` and `rate_limit` bound has to be one the
/// plane is about to emit: the differences of its own commands. A position-commanded servo
/// always trails its setpoint -- an `sts3215` under load by degrees -- and that following
/// error is what generates its torque, not a velocity the envelope may read. Spec 9.5 says
/// the same thing from the other end: the same `deployment_hash` must give the same safe
/// action in simulation and on hardware, which a bound computed from feedback cannot.
///
/// So: a command stream inside the envelope stays `ActionSource::Policy` **whatever the
/// tracking lag**, and a caller that observes before every `validate` gets the same actions,
/// bit for bit, as one that observes once at the start of the episode. Before V6 the second
/// reading was what `es_eval::runner`, `es_ros2::hil` and `es_runtime_embedded` got, the bound
/// collapsed to 0.008 rad of following error, and the scripted expert scored 0/16 through the
/// evaluation harness it is supposed to define the ceiling of.
#[test]
fn the_envelope_bounds_commands_not_the_following_error() {
    const STEPS: usize = 60;
    // A first-order lag with a 30 % per-tick response: the command leads the joint by up to
    // a couple of steps, two orders of magnitude past what the old reading allowed.
    const ALPHA: f64 = 0.3;
    // Where the arm is when the episode opens -- not zero, so the seed is observable.
    let start = [0.3, -0.2, 0.15, 0.1, 0.0, 0.2];
    let target = [0.8, 0.5, -0.4, 0.6, 0.0, 0.2];

    let ramps: Vec<Vec<f64>> = (0..NJ).map(|j| ramp(start[j], target[j], STEPS)).collect();

    // Two callers, the same commands, different observation discipline.
    let mut every_step = demo_plane();
    let mut once_per_episode = demo_plane();
    let mut q_a = start;
    let mut q_b = start;
    let mut worst_lag = 0.0f64;

    every_step.observe_state(&start, &[0.0; NJ]);
    once_per_episode.observe_state(&start, &[0.0; NJ]);

    for t in 0..STEPS {
        let mut row = [0.0; NJ];
        for (j, r) in ramps.iter().enumerate() {
            row[j] = r[t];
        }
        // The caller that re-observes every tick, which is what every consumer now does.
        let qd: [f64; NJ] = std::array::from_fn(|j| (q_a[j] - start[j]) / DT);
        every_step.observe_state(&q_a, &qd);
        let tick = PhysTick(t as u64);
        let seq = t as u64 + 1;
        let a = every_step.validate(&chunk(row, seq), Micros(0), tick);
        let b = once_per_episode.validate(&chunk(row, seq), Micros(0), tick);

        assert_eq!(
            a.source,
            ActionSource::Policy,
            "step {t}: a paced command was corrected -- {:?}",
            a.events
        );
        for j in 0..NJ {
            assert_eq!(
                a.q[j].to_bits(),
                row[j].to_bits(),
                "step {t} joint {j}: the plane changed a command inside the envelope"
            );
            assert_eq!(
                a.q[j].to_bits(),
                b.q[j].to_bits(),
                "step {t} joint {j}: observing every tick disagrees with observing once"
            );
            worst_lag = worst_lag.max((row[j] - q_a[j]).abs());
            q_a[j] += ALPHA * (a.q[j] - q_a[j]);
            q_b[j] += ALPHA * (b.q[j] - q_b[j]);
        }
    }

    // Non-vacuity: the plant really did trail far past what the old reading allowed, so this
    // test fails if the envelope ever goes back to measuring the following error.
    assert!(
        worst_lag > 10.0 * OLD_BOUND,
        "the plant tracked too well to prove anything: worst lag {worst_lag}"
    );
    println!(
        "RAN the_envelope_bounds_commands_not_the_following_error: {STEPS} clean steps at a \
         worst following error of {worst_lag:.4} rad ({:.0}x the {OLD_BOUND} rad the pre-V6 \
         evaluation path allowed)",
        worst_lag / OLD_BOUND
    );
}

/// The other half: V6 widened nothing. A command stream that asks for more than the envelope
/// allows is still clamped and still counted, measured against the plane's own last command.
#[test]
fn a_command_outside_the_envelope_is_still_clamped() {
    let mut plane = demo_plane();
    let start = [0.3, -0.2, 0.15, 0.1, 0.0, 0.2];
    plane.observe_state(&start, &[0.0; NJ]);
    // A full radian in one tick: 50 rad/s against a 3 rad/s limit.
    let jump = [1.3, -0.2, 0.15, 0.1, 0.0, 0.2];
    let out = plane.validate(&chunk(jump, 1), Micros(0), PhysTick(0));
    assert_eq!(out.source, ActionSource::Clamped);
    assert!(out.events.contains(ViolationKind::Acceleration));
    // Bounded by the acceleration stage, which binds before velocity at these numbers.
    assert!(
        (out.q[0] - start[0]).abs() <= OLD_BOUND + 1e-12,
        "joint 0 moved {} in one tick",
        out.q[0] - start[0]
    );
    assert_eq!(plane.counters().clamped_steps, 1);
}

/// `begin_episode` re-arms the seed: a second episode opens on the arm's real pose, not on
/// where the previous episode's last command left it (the V1c finding, now one call), and a
/// mid-episode measurement never moves the reference the clamp stages use.
#[test]
fn begin_episode_re_arms_the_seed_and_clears_the_latch() {
    let mut plane = demo_plane();
    let first = [0.3, -0.2, 0.15, 0.1, 0.0, 0.2];
    plane.observe_state(&first, &[0.0; NJ]);
    assert_eq!(plane.last_safe_action(), first);
    // Mid-episode measurements are ignored: the command chain owns its own reference.
    let elsewhere = [-0.9, 0.7, -0.6, 0.4, 0.2, 0.5];
    plane.observe_state(&elsewhere, &[0.0; NJ]);
    assert_eq!(plane.last_safe_action(), first);
    // The next episode's first observation is not.
    plane.begin_episode();
    plane.observe_state(&elsewhere, &[0.0; NJ]);
    assert_eq!(plane.last_safe_action(), elsewhere);
    assert!(!plane.is_latched());
}
