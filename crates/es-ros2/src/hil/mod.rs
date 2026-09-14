//! Hardware-in-the-loop: an external controller drives the simulation over low-latency UDP
//! (spec 24.2, `docs/design/ros2-boundary.md` section 7).
//!
//! ```text
//! controller under test --UDP Command/Heartbeat--> HilLink --> HilCore<NJ,H> --SafeAction--> plant
//!                       <--UDP State(q, qd)------                  |
//!                                                                  +--> .eshil log, HilStats
//! ```
//!
//! Spec 24.2 withholds tier-1 determinism from HIL, because *when* a datagram arrives is the
//! external controller's business. [`HilCore::tick`] is where that stops: it bins everything
//! accepted since the last tick into the integer inputs of one
//! [`es_runtime_embedded::core_rt::EmbeddedCore::step`] — the loop the robot itself runs
//! (spec 9.5) — and records those inputs in a `.eshil` log. [`replay`] then re-derives every
//! decision offline and compares it byte for byte. That equality is the M3 gate.
//!
//! It is **not** a claim about a physical controller, a real network, RT scheduling, or
//! spec 28.7's gate 14, which stays open (design note section 7.5).
//!
//! Why a module and not an `es-hil` crate: spec 24.2 names one, spec 4.2's layer table does
//! not, and adding a crate to that table is a human decision (design note sections 2 and 10).
//! To keep a later extraction a file move, nothing here imports a ROS module — a test in
//! `tests/hil_gate.rs` enforces it.

mod core;
pub mod link;
pub mod log;
pub mod replay;
pub mod stats;
pub mod wire;

use thiserror::Error;

pub use self::core::HilCore;
pub use self::link::{HilConfig, HilLink};
pub use self::log::HilLogError;
pub use self::replay::{replay, ReplayReport};
pub use self::stats::{HilStats, HIL_STATS_STREAM};
pub use self::wire::{Command, HilMsg, MsgKind, WireError};

/// Why a HIL run could not start, or could not be closed out.
///
/// There is deliberately no variant for "the plane refused an action": the plane cannot refuse
/// (INV-13), and no path here returns anything but a [`es_safety::SafeAction`] toward the
/// plant (INV-12).
#[derive(Debug, Error)]
pub enum HilError {
    #[error("hil log I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The Deployment IR does not describe this `HilCore`'s shape, does not validate, or
    /// cannot be hashed or serialized.
    #[error("deployment ir: {0}")]
    Ir(String),
}
