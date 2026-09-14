//! The values that cross the Safety Plane boundary (spec 9, Appendix B.4).
//!
//! Everything here is `Copy` and fixed size: an action chunk is an array, not a slice, so the
//! runtime never borrows caller memory and never allocates.

use serde::{Deserialize, Serialize};

use crate::ir_types::ExecutionMode;

/// One policy output: `H` predicted actions of `NJ` components each (spec 8.5).
///
/// `valid` is how many leading rows the policy actually filled. How many of those are
/// *executable* is a separate question the plane answers from `execute_chunk` (spec 8.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionChunk<const NJ: usize, const H: usize> {
    pub actions: [[f64; NJ]; H],
    pub valid: usize,
    pub mode: ExecutionMode,
    /// Caller-supplied, monotonic per policy invocation (spec 8.6). The plane treats a chunk
    /// as new iff `seq` is strictly greater than the last one it accepted — content is no
    /// longer consulted (`docs/design/safety-plane.md`, P-M1-R3). `0` from [`Self::new`] /
    /// [`Self::empty`] means "unknown"; that is safe because the plane always accepts the
    /// very first chunk it ever sees regardless of `seq`. A caller that cares about
    /// freshness (the embedded runtime) must call [`Self::with_seq`].
    pub seq: u64,
}

impl<const NJ: usize, const H: usize> ActionChunk<NJ, H> {
    /// A chunk of `valid` rows, clamped to `H`. `seq` defaults to `0` ("unknown"); chain
    /// [`Self::with_seq`] when the caller tracks a real sequence number.
    pub fn new(actions: [[f64; NJ]; H], valid: usize, mode: ExecutionMode) -> Self {
        Self {
            actions,
            valid: valid.min(H),
            mode,
            seq: 0,
        }
    }

    /// An empty chunk: nothing to execute, so the plane reports an underrun (spec 8.6).
    pub fn empty(mode: ExecutionMode) -> Self {
        Self {
            actions: [[0.0; NJ]; H],
            valid: 0,
            mode,
            seq: 0,
        }
    }

    /// Attaches the caller's per-invocation sequence number (spec 8.6).
    #[must_use]
    pub fn with_seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }
}

/// Where the emitted action came from. There is no "unchecked" variant (INV-12).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSource {
    /// The policy's row, emitted unchanged.
    Policy,
    /// The policy's row after at least one envelope stage changed it (spec 9.3).
    Clamped,
    /// A watchdog tripped; this is the fallback's action (spec 9.4).
    Fallback(FallbackKind),
}

/// Which fallback produced the action (spec 9.4). Mirrors `es_ir::deployment::FallbackPolicy`
/// without its payload, which lives in the plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackKind {
    HoldPosition,
    ZeroVelocity,
    RetractToHome,
    /// Hold, and tell the caller to switch to its classical controller.
    HandoffController,
    /// Latched: every later step stays here until `reset_latch`.
    EmergencyStop,
}

/// What the plane observed on one step. Clamps and watchdog trips share this enum so the
/// `failure_mode_histogram` of spec 10.3 is one array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationKind {
    /// A non-finite component: not clampable, so an immediate fallback (spec 9.3).
    NonFinite,
    Position,
    Velocity,
    Acceleration,
    Torque,
    Workspace,
    RateLimit,
    StaleObservation,
    InferenceDeadline,
    ChunkUnderrun,
    HeartbeatLoss,
    SensorDropout,
    /// The sliding envelope-violation fraction exceeded its bound (spec 9.4, spec 10.3).
    ViolationRate,
    /// The emergency-stop latch is engaged (spec 9.4).
    EstopLatched,
}

impl ViolationKind {
    pub const COUNT: usize = 14;

    pub const ALL: [Self; Self::COUNT] = [
        Self::NonFinite,
        Self::Position,
        Self::Velocity,
        Self::Acceleration,
        Self::Torque,
        Self::Workspace,
        Self::RateLimit,
        Self::StaleObservation,
        Self::InferenceDeadline,
        Self::ChunkUnderrun,
        Self::HeartbeatLoss,
        Self::SensorDropout,
        Self::ViolationRate,
        Self::EstopLatched,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The violations of one step, as a bitset over [`ViolationKind`]. Fixed size, `Copy`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventSet(u32);

impl EventSet {
    pub const EMPTY: Self = Self(0);

    pub fn insert(&mut self, kind: ViolationKind) {
        self.0 |= 1 << kind.index();
    }

    pub const fn contains(self, kind: ViolationKind) -> bool {
        self.0 & (1 << kind.index()) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    /// This set with `kind` removed; removing an absent kind is the identity. The spec 9.4 rate
    /// watchdog measures the window through this so it never measures its own trip
    /// (P-M3-W1-R7).
    #[must_use]
    pub const fn without(self, kind: ViolationKind) -> Self {
        Self(self.0 & !(1 << kind.index()))
    }

    /// Every kind in the set, in [`ViolationKind::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = ViolationKind> {
        ViolationKind::ALL
            .into_iter()
            .filter(move |k| self.contains(*k))
    }
}

/// The plane's output: an action that is always executable (INV-13).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafeAction<const NJ: usize> {
    pub q: [f64; NJ],
    pub source: ActionSource,
    pub events: EventSet,
}

impl<const NJ: usize> SafeAction<NJ> {
    /// Whether the action left the plane unchanged (spec 10.3: this step is not a violation).
    pub fn is_clean(&self) -> bool {
        self.source == ActionSource::Policy && self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P-M3-W1-R7: the rate watchdog's window uses this to skip its own echo.
    #[test]
    fn without_removes_exactly_one_kind() {
        let mut s = EventSet::EMPTY;
        s.insert(ViolationKind::Velocity);
        s.insert(ViolationKind::ViolationRate);
        // Removing an absent kind is the identity.
        assert_eq!(s.without(ViolationKind::Torque), s);
        // Other kinds survive.
        let rest = s.without(ViolationKind::ViolationRate);
        assert!(rest.contains(ViolationKind::Velocity));
        assert!(!rest.contains(ViolationKind::ViolationRate));
        // Removing the only present kind empties the set.
        assert_eq!(rest.without(ViolationKind::Velocity), EventSet::EMPTY);
        assert!(EventSet::EMPTY.without(ViolationKind::Velocity).is_empty());
    }
}
