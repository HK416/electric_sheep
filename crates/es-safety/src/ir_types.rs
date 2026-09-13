//! The four Deployment IR value types the hot path carries, and nothing else.
//!
//! `es-ir` is a `std` crate (serde, `Vec`, `String`, the validator), and spec 9.6 requires the
//! Safety Plane to build without `std`. Under the default `std` feature these names are
//! *re-exports of `es-ir`'s own types*, so `SafetyPlane::validate(.., obs_age: Micros, ..)` has
//! exactly the signature it always had (INV-13) and every caller compiles unchanged. Under
//! `no_std` they are local definitions with the same shape.
//!
//! They stay this small on purpose: a `u64`, a pair of `f64`, and two C-like enums. Anything
//! with a `Vec` in it (`Workspace::ConvexHull`, `Watchdog`, `DeploymentIr`) is *not* shimmed —
//! it is converted once, at construction, into the fixed-size form in [`crate::config`].
//!
//! # ponytail: two definitions of four types, kept in sync by review
//!
//! The ceiling is that a field added to `es_ir::deployment::Limit` will not appear here. The
//! upgrade path is to move these four types into `es-ir-types` (layer 6 has no `std` need for
//! them) and delete this module; that is a separate packet because `es-ir` is out of scope for
//! W2. `tests/nostd_shim.rs` pins the shape under `std`.

#[cfg(feature = "std")]
pub use es_ir::deployment::{ActionSpace, ExecutionMode, HalfSpace, Limit, Micros};

/// A duration in whole microseconds. Integer by construction: `f64` time is forbidden
/// (spec 3.4).
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Micros(pub u64);

/// A closed interval. `lower < upper` is a validated invariant, not an assumption.
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limit {
    pub lower: f64,
    pub upper: f64,
}

/// Half-space `n . x <= d`, the face of a convex polyhedron.
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfSpace {
    pub normal: [f64; 3],
    pub offset: f64,
}

/// Command space of the action the Safety Plane validates (spec 8.5).
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionSpace {
    JointPosition,
    JointVelocity,
    JointTorque,
    EePose,
    EeDelta,
    Gripper,
    Composite,
}

/// How a chunk is consumed (spec 8.5 `ActionExecutionMode`, spec 9.2).
#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionMode {
    /// Execute the whole chunk, then replan.
    OpenLoopChunk,
    /// Execute K, then replan (default).
    RecedingHorizon,
    /// ACT: exponentially weighted average of overlapping predictions.
    TemporalEnsemble { decay: f64 },
    /// Compute the next chunk while the current one runs (spec 8.6).
    RealTimeChunking,
}
