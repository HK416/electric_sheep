//! The Safety Plane runtime (spec 9, Appendix B.4).
//!
//! One function matters: [`SafetyPlane::validate`]. It cannot fail, cannot panic and cannot
//! be turned off. See `docs/design/safety-plane.md` for the algorithm in prose.

use es_core::PhysTick;
use es_ir::deployment::{ActionSpace, DeploymentIr, ExecutionMode, Micros};

use crate::config::{Envelope, Fallback, SafetyConfigError, Watchdogs};
use crate::counters::SafetyCounters;
use crate::types::{ActionChunk, ActionSource, EventSet, FallbackKind, SafeAction, ViolationKind};

/// Everything the plane remembers between control ticks. Pre-allocated: no field grows.
#[derive(Clone, Copy, Debug)]
pub struct SafetyState<const NJ: usize, const H: usize> {
    /// The accepted chunk, copied in so the runtime never borrows caller memory (spec 8.6).
    chunk: [[f64; NJ]; H],
    chunk_valid: usize,
    chunk_mode: ExecutionMode,
    /// Index of the next row to execute.
    cursor: usize,
    /// Last emitted action: the hold target of every fallback.
    last_safe: [f64; NJ],
    prev_safe: [f64; NJ],
    /// Velocity estimate implied by the last two emitted actions.
    vel: [f64; NJ],
    prev_vel: [f64; NJ],
    /// Cursor into the retract trajectory; saturates at the last waypoint.
    retract_idx: usize,
    last_chunk_tick: PhysTick,
    last_beat_tick: PhysTick,
    estop_latched: bool,
}

impl<const NJ: usize, const H: usize> SafetyState<NJ, H> {
    fn new(envelope: &Envelope<NJ>) -> Self {
        // Midpoint of each soft limit: inside the envelope by construction. A runtime that
        // knows the real pose calls `observe_state` before the first `validate`.
        let mut last_safe = [0.0; NJ];
        for (q, l) in last_safe.iter_mut().zip(&envelope.soft) {
            *q = f64::midpoint(l.lower, l.upper);
        }
        Self {
            chunk: [[0.0; NJ]; H],
            chunk_valid: 0,
            chunk_mode: ExecutionMode::RecedingHorizon,
            cursor: 0,
            last_safe,
            prev_safe: last_safe,
            vel: [0.0; NJ],
            prev_vel: [0.0; NJ],
            retract_idx: 0,
            last_chunk_tick: PhysTick::ZERO,
            last_beat_tick: PhysTick::ZERO,
            estop_latched: false,
        }
    }
}

/// The policy-independent, deterministic, fail-safe action filter (spec 9.1, Appendix B.4).
///
/// Built only from a validated [`DeploymentIr`]: there is no constructor that omits the
/// envelope and no field that disables it (INV-12).
#[derive(Clone, Debug)]
pub struct SafetyPlane<const NJ: usize, const H: usize> {
    envelope: Envelope<NJ>,
    watchdogs: Watchdogs,
    fallback: Fallback<NJ>,
    state: SafetyState<NJ, H>,
    counters: SafetyCounters,
}

impl<const NJ: usize, const H: usize> SafetyPlane<NJ, H> {
    /// The only constructor (spec 9.2).
    ///
    /// Rejects an IR whose own validator complains, whose joint count or horizon disagrees
    /// with the const generics, whose limits are not finite, or whose violation-rate window
    /// exceeds the pre-allocated ring.
    pub fn from_ir(ir: &DeploymentIr) -> Result<Self, SafetyConfigError> {
        if let Some(d) = ir.validate().first() {
            return Err(SafetyConfigError::InvalidIr(format!("{d}")));
        }
        if ir.robot.n_joints != NJ || ir.action.dim != NJ {
            return Err(SafetyConfigError::JointCount {
                n_joints: ir.robot.n_joints,
                dim: ir.action.dim,
                expected: NJ,
            });
        }
        if ir.action.horizon != H {
            return Err(SafetyConfigError::Horizon {
                horizon: ir.action.horizon,
                expected: H,
            });
        }
        let envelope = Envelope::from_ir(ir)?;
        let watchdogs = Watchdogs::from_ir(ir)?;
        let counters = SafetyCounters::new(watchdogs.window_len());
        Ok(Self {
            state: SafetyState::new(&envelope),
            envelope,
            watchdogs,
            fallback: Fallback::from_ir(ir)?,
            counters,
        })
    }

    pub fn envelope(&self) -> &Envelope<NJ> {
        &self.envelope
    }

    pub fn counters(&self) -> &SafetyCounters {
        &self.counters
    }

    pub fn fallback_kind(&self) -> FallbackKind {
        self.fallback.kind
    }

    /// The last action the plane emitted: the hold target (spec 9.4).
    pub fn last_safe_action(&self) -> [f64; NJ] {
        self.state.last_safe
    }

    /// Whether the emergency stop is engaged (spec 9.4).
    pub fn is_latched(&self) -> bool {
        self.state.estop_latched
    }

    /// Clears the emergency-stop latch. The only way out, and it is never called from inside
    /// `validate` (spec 9.4).
    pub fn reset_latch(&mut self) {
        self.state.estop_latched = false;
    }

    /// Zeroes the statistics. Does not touch the latch or the envelope.
    pub fn reset_counters(&mut self) {
        self.counters.reset();
    }

    /// The controller is alive as of `now` (spec 9.4 `ControllerHeartbeat`).
    pub fn heartbeat(&mut self, now: PhysTick) {
        self.state.last_beat_tick = now;
    }

    /// A sample from `sensor` arrived at `now` (spec 9.4 `SensorDropout`). Unknown names are
    /// ignored: only configured sensors are watched.
    pub fn sensor_seen(&mut self, sensor: &str, now: PhysTick) {
        for s in &mut self.watchdogs.sensors {
            if s.name == sensor {
                s.last_seen = now;
            }
        }
    }

    /// Seeds the measured joint state, so the hold target is where the robot actually is.
    /// Non-finite input is dropped and every value is clamped into the hard limits.
    pub fn observe_state(&mut self, q: &[f64; NJ], qd: &[f64; NJ]) {
        for i in 0..NJ {
            if q[i].is_finite() {
                self.state.last_safe[i] =
                    q[i].clamp(self.envelope.hard[i].lower, self.envelope.hard[i].upper);
            }
            if qd[i].is_finite() {
                self.state.vel[i] =
                    qd[i].clamp(-self.envelope.vel_max[i], self.envelope.vel_max[i]);
            }
        }
        self.state.prev_safe = self.state.last_safe;
        self.state.prev_vel = self.state.vel;
    }

    /// Validates and corrects one chunk step. **Cannot fail** — there is always a safe action,
    /// so propagating an error upward would only invite the caller to ignore it (Appendix B.4,
    /// INV-13). Events are recorded in the returned [`SafeAction`] and in the counters.
    pub fn validate(
        &mut self,
        chunk: &ActionChunk<NJ, H>,
        obs_age: Micros,
        now: PhysTick,
    ) -> SafeAction<NJ> {
        // Step 0: the latch short-circuits everything (spec 9.4).
        if self.state.estop_latched {
            let mut events = EventSet::EMPTY;
            events.insert(ViolationKind::EstopLatched);
            let q = self.scrub(self.state.last_safe);
            // A latched step is still a fallback step (spec 9.4, spec 10.3).
            self.counters.fallback_activations += 1;
            return self.finish(
                q,
                ActionSource::Fallback(FallbackKind::EmergencyStop),
                events,
            );
        }

        // Step 1: accept a new chunk, or keep consuming the stored one.
        self.accept(chunk, now);

        // Step 2: executable length. Rows past K are predictions, not commands (spec 8.5).
        let len = match self.state.chunk_mode {
            ExecutionMode::RecedingHorizon => {
                self.state.chunk_valid.min(self.envelope.execute_chunk)
            }
            _ => self.state.chunk_valid.min(H),
        };

        // Step 3: watchdogs, in the fixed order of docs/design/safety-plane.md.
        let mut events = EventSet::EMPTY;
        let candidate = if self.state.cursor < len {
            Some(self.state.chunk[self.state.cursor])
        } else {
            events.insert(ViolationKind::ChunkUnderrun);
            None
        };
        if let Some(a) = candidate {
            if a.iter().any(|v| !v.is_finite()) {
                events.insert(ViolationKind::NonFinite);
            }
        }
        if let Some(max_age) = self.watchdogs.max_obs_age {
            if obs_age.0 > max_age.0 {
                events.insert(ViolationKind::StaleObservation);
            }
        }
        if let Some(budget) = self.watchdogs.inference_budget {
            if self.micros_since(self.state.last_chunk_tick, now) > budget.0 {
                events.insert(ViolationKind::InferenceDeadline);
            }
        }
        if let Some(timeout) = self.watchdogs.heartbeat_timeout {
            if self.micros_since(self.state.last_beat_tick, now) > timeout.0 {
                events.insert(ViolationKind::HeartbeatLoss);
            }
        }
        for s in &self.watchdogs.sensors {
            let gap = now
                .ticks_since(s.last_seen)
                .unwrap_or(0)
                .saturating_mul(self.envelope.period_us);
            if gap > s.max_gap.0 {
                events.insert(ViolationKind::SensorDropout);
            }
        }
        if let Some((_, max_frac)) = self.watchdogs.violation_rate {
            // Read as of the previous step: this step's own clamp is not recorded yet, which
            // is what keeps the rule non-circular (spec 9.4, spec 10.3).
            if self.counters.envelope_violation_rate() > max_frac {
                events.insert(ViolationKind::ViolationRate);
            }
        }

        // Step 4a: any trip runs the configured fallback (spec 9.4).
        if !events.is_empty() {
            let q = self.run_fallback();
            let q = self.scrub(q);
            let kind = self.fallback.kind;
            if kind == FallbackKind::EmergencyStop {
                self.state.estop_latched = true;
            }
            self.counters.fallback_activations += 1;
            return self.finish(q, ActionSource::Fallback(kind), events);
        }

        // Step 4b: clamp, in the fixed stage order (spec 9.3).
        let mut a = candidate.unwrap_or(self.state.last_safe);
        self.clamp(&mut a, &mut events);
        let a = self.scrub(a);
        self.state.cursor += 1;
        let source = if events.is_empty() {
            ActionSource::Policy
        } else {
            self.counters.clamped_steps += 1;
            ActionSource::Clamped
        };
        self.finish(a, source, events)
    }

    // --- internals ------------------------------------------------------------------------

    fn micros_since(&self, earlier: PhysTick, now: PhysTick) -> u64 {
        now.ticks_since(earlier)
            .unwrap_or(0)
            .saturating_mul(self.envelope.period_us)
    }

    /// Copies `chunk` in if it differs from the stored one. Identity is content equality over
    /// the valid prefix; see the "known ceiling" note in docs/design/safety-plane.md.
    fn accept(&mut self, chunk: &ActionChunk<NJ, H>, now: PhysTick) {
        let same = chunk.valid == self.state.chunk_valid
            && chunk.mode == self.state.chunk_mode
            && chunk.actions[..chunk.valid]
                .iter()
                .zip(&self.state.chunk[..chunk.valid])
                .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        if !same {
            self.state.chunk = chunk.actions;
            self.state.chunk_valid = chunk.valid;
            self.state.chunk_mode = chunk.mode;
            self.state.cursor = 0;
            self.state.last_chunk_tick = now;
        }
    }

    /// Stage order: NaN/Inf, position, velocity, acceleration, torque, workspace, rate limit.
    fn clamp(&self, a: &mut [f64; NJ], events: &mut EventSet) {
        let e = &self.envelope;
        let (dt, st) = (e.dt_s, &self.state);

        // 1. NaN/Inf. The watchdog already caught this, so it is a defensive no-op that keeps
        //    the stage order identical on both paths (spec 9.3 `action_validity`).
        for (i, v) in a.iter_mut().enumerate() {
            if !v.is_finite() {
                *v = st.last_safe[i];
                events.insert(ViolationKind::NonFinite);
            }
        }
        // 2. Position, against the soft limit.
        for (i, v) in a.iter_mut().enumerate() {
            let c = v.clamp(e.soft[i].lower, e.soft[i].upper);
            if c.to_bits() != v.to_bits() {
                *v = c;
                events.insert(ViolationKind::Position);
            }
        }
        // 3. Velocity implied by the step.
        for (i, v) in a.iter_mut().enumerate() {
            let vel = (*v - st.last_safe[i]) / dt;
            if vel.abs() > e.vel_max[i] {
                *v = st.last_safe[i] + vel.signum() * e.vel_max[i] * dt;
                events.insert(ViolationKind::Velocity);
            }
        }
        // 4. Acceleration implied by the velocity change.
        for (i, v) in a.iter_mut().enumerate() {
            let vel = (*v - st.last_safe[i]) / dt;
            let acc = (vel - st.vel[i]) / dt;
            if acc.abs() > e.acc_max[i] {
                let vel = st.vel[i] + acc.signum() * e.acc_max[i] * dt;
                *v = st.last_safe[i] + vel * dt;
                events.insert(ViolationKind::Acceleration);
            }
        }
        // 5. Torque, only where the action is one.
        if e.space == ActionSpace::JointTorque {
            for (i, v) in a.iter_mut().enumerate() {
                let c = v.clamp(-e.tau_max[i], e.tau_max[i]);
                if c.to_bits() != v.to_bits() {
                    *v = c;
                    events.insert(ViolationKind::Torque);
                }
            }
        }
        // 6. Workspace, only for end-effector spaces (no forward kinematics here).
        if e.workspace_applies() {
            let mut p = [a[0], a[1], a[2]];
            if e.project_workspace(&mut p) {
                a[..3].copy_from_slice(&p);
                events.insert(ViolationKind::Workspace);
            }
        }
        // 7. Rate limit: first then second difference (spec 9.3 `rate_limit`).
        for (i, v) in a.iter_mut().enumerate() {
            let d1 = *v - st.last_safe[i];
            if d1.abs() > e.d1_max[i] {
                *v = st.last_safe[i] + d1.signum() * e.d1_max[i];
                events.insert(ViolationKind::RateLimit);
            }
            let d2 = *v - 2.0 * st.last_safe[i] + st.prev_safe[i];
            if d2.abs() > e.d2_max[i] {
                *v = 2.0 * st.last_safe[i] - st.prev_safe[i] + d2.signum() * e.d2_max[i];
                events.insert(ViolationKind::RateLimit);
            }
        }
    }

    /// The fallback action (spec 9.4). Never a network, never dependent on a policy runtime.
    fn run_fallback(&mut self) -> [f64; NJ] {
        let (e, st) = (&self.envelope, &mut self.state);
        match self.fallback.kind {
            FallbackKind::HoldPosition
            | FallbackKind::HandoffController
            | FallbackKind::EmergencyStop => st.last_safe,
            FallbackKind::ZeroVelocity => {
                let mut q = st.last_safe;
                for (i, out) in q.iter_mut().enumerate() {
                    let dv = e.acc_max[i] * e.dt_s;
                    let v = st.vel[i];
                    let v = if v > dv {
                        v - dv
                    } else if v < -dv {
                        v + dv
                    } else {
                        0.0
                    };
                    *out += v * e.dt_s;
                }
                q
            }
            FallbackKind::RetractToHome => {
                let last = self.fallback.trajectory.len() - 1;
                let q = self.fallback.trajectory[st.retract_idx.min(last)];
                st.retract_idx = (st.retract_idx + 1).min(last);
                q
            }
        }
    }

    /// Final scrub: non-finite becomes the hold value, then the hard limits (spec 9.3). After
    /// this the output is inside the envelope no matter which path produced it.
    fn scrub(&self, mut q: [f64; NJ]) -> [f64; NJ] {
        for (i, v) in q.iter_mut().enumerate() {
            if !v.is_finite() {
                *v = self.state.last_safe[i];
            }
            if !v.is_finite() {
                *v = f64::midpoint(self.envelope.hard[i].lower, self.envelope.hard[i].upper);
            }
            *v = v.clamp(self.envelope.hard[i].lower, self.envelope.hard[i].upper);
        }
        q
    }

    /// Common tail: statistics and the state roll-forward, identical on both paths.
    fn finish(&mut self, q: [f64; NJ], source: ActionSource, events: EventSet) -> SafeAction<NJ> {
        for kind in events.iter() {
            self.counters.record(kind);
        }
        self.counters.steps += 1;
        self.counters.window.push(!events.is_empty());

        self.state.prev_vel = self.state.vel;
        for (i, v) in self.state.vel.iter_mut().enumerate() {
            *v = (q[i] - self.state.last_safe[i]) / self.envelope.dt_s;
        }
        self.state.prev_safe = self.state.last_safe;
        self.state.last_safe = q;
        SafeAction { q, source, events }
    }
}
