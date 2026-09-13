//! `es-core` (layer 1): ECS, job system, time model, failure semantics, stable ids and
//! allocation primitives. See `docs/ARCHITECTURE.ko.md` §5, §12, §18 and the work packets
//! `docs/packets/M0/P13.md` .. `P17.md`.
//!
//! Layer rule (§4.2): this crate may depend on `es-math` and external crates only.
//!
//! # `no_std`
//!
//! Spec §9.6 requires the deployment runtime to be no-std capable with zero heap allocation,
//! and `es-safety` (its Safety Plane) reaches down here for [`PhysTick`] and the telemetry
//! ring. Building with `--no-default-features` therefore keeps exactly two things:
//! [`time::PhysTick`] and [`ring::ArrayRing`]. Everything else — the ECS, the job system,
//! failure policy, stable ids, arenas, pools, the allocation counter, `TickRate`/`SimTime`
//! (whose fallible constructor needs [`Error`]) and the heap-backed [`ring::RingBuffer`] —
//! is behind the default `std` feature. See `docs/design/embedded-runtime.md`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
pub mod alloc_count;
#[cfg(feature = "std")]
pub mod arena;
#[cfg(feature = "std")]
pub mod ecs;
#[cfg(feature = "std")]
pub mod failure;
#[cfg(feature = "std")]
pub mod id;
#[cfg(feature = "std")]
pub mod job;
#[cfg(feature = "std")]
pub mod pool;
pub mod ring;
#[cfg(feature = "std")]
pub mod sizing;
pub mod time;

#[cfg(feature = "std")]
pub use failure::{EnvHealth, Error, FailureAction, FailureKind, FailurePolicy};
#[cfg(feature = "std")]
pub use id::StableId;
pub use time::PhysTick;
#[cfg(feature = "std")]
pub use time::{SimTime, TickRate};

/// Counts every allocation made by the test binary so [`alloc_count::assert_no_alloc`] can
/// judge allocation-free code paths (P17).
#[cfg(any(test, feature = "alloc-count"))]
#[global_allocator]
static GLOBAL: alloc_count::AllocCounter<std::alloc::System> =
    alloc_count::AllocCounter::new(std::alloc::System);
