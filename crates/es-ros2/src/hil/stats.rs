//! HIL statistics and their telemetry frame (`docs/design/ros2-boundary.md` section 7.6,
//! spec 24.2).
//!
//! This module and [`super::link`] are the only two that may name a wall clock: every number
//! here is an *observation*, never a Safety Plane input (spec 3.4, spec 3.5). The plane sees
//! integer [`es_core::PhysTick`]s only.

use std::time::{SystemTime, UNIX_EPOCH};

use es_core::PhysTick;
use es_telemetry::{Frame, Payload, StreamId};

/// Telemetry stream for [`HilStats`] frames (design note section 7.6). `"HIL1"` as big-endian
/// ASCII.
pub const HIL_STATS_STREAM: StreamId = StreamId(0x4849_4C31);

/// How many scalars one [`HilStats`] frame carries. Fixed: the field order is the wire
/// contract (design note section 7.6), so a reader indexes by position.
pub const HIL_STATS_FIELDS: usize = 10;

/// The link's counters (design note section 7.6). Counts are exact; the two timing fields are
/// nanosecond observations of the loopback round trip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HilStats {
    pub ticks: u64,
    pub commands: u64,
    pub deadline_miss: u64,
    pub rx_lost: u64,
    pub rx_stale: u64,
    pub rx_invalid: u64,
    pub superseded: u64,
    pub heartbeats: u64,
    /// Last State -> Command round trip, nanoseconds. `0` before the first one.
    pub rtt_last: i64,
    /// RFC 3550 interarrival jitter over that round trip, nanoseconds.
    pub jitter: i64,
}

impl HilStats {
    /// The scalars of one telemetry frame, in the design note's fixed field order.
    #[allow(clippy::cast_precision_loss)]
    pub fn scalars(&self) -> Vec<f64> {
        vec![
            self.ticks as f64,
            self.commands as f64,
            self.deadline_miss as f64,
            self.rx_lost as f64,
            self.rx_stale as f64,
            self.rx_invalid as f64,
            self.superseded as f64,
            self.heartbeats as f64,
            self.rtt_last as f64,
            self.jitter as f64,
        ]
    }

    /// One round-trip sample. RFC 3550's estimator `J += (|D| - J) / 16`, in integer
    /// nanoseconds so nothing accumulates in `f64` (spec 3.4).
    pub(crate) fn observe_rtt(&mut self, rtt_ns: i64) {
        if self.rtt_last != 0 {
            let d = rtt_ns.saturating_sub(self.rtt_last).saturating_abs();
            self.jitter += (d - self.jitter) / 16;
        }
        self.rtt_last = rtt_ns;
    }

    /// The telemetry frame for `now` (design note section 7.6).
    pub fn frame(&self, now: PhysTick) -> Frame {
        Frame {
            tick: now,
            wall_ns: wall_ns(),
            stream: HIL_STATS_STREAM,
            payload: Payload::Scalars(self.scalars()),
        }
    }
}

/// Wall clock nanoseconds since the Unix epoch, for the telemetry frame's timestamp only.
pub(crate) fn wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}
