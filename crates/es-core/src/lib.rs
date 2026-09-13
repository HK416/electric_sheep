//! `es-core` (layer 1): ECS, job system, time model, failure semantics, stable ids and
//! allocation primitives. See `docs/ARCHITECTURE.ko.md` §5, §12, §18 and the work packets
//! `docs/packets/M0/P13.md` .. `P17.md`.
//!
//! Layer rule (§4.2): this crate may depend on `es-math` and external crates only.

pub mod alloc_count;
pub mod arena;
pub mod ecs;
pub mod failure;
pub mod id;
pub mod job;
pub mod pool;
pub mod ring;
pub mod time;

pub use failure::{EnvHealth, Error, FailureAction, FailureKind, FailurePolicy};
pub use id::StableId;
pub use time::{PhysTick, SimTime, TickRate};

/// Counts every allocation made by the test binary so [`alloc_count::assert_no_alloc`] can
/// judge allocation-free code paths (P17).
#[cfg(any(test, feature = "alloc-count"))]
#[global_allocator]
static GLOBAL: alloc_count::AllocCounter<std::alloc::System> =
    alloc_count::AllocCounter::new(std::alloc::System);
