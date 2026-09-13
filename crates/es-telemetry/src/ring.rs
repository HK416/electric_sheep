//! The telemetry ring buffer (spec 9.6).
//!
//! The implementation lives in `es_core::ring` (layer 1) so that `es-runtime-embedded` (layer
//! 9) can use it too without depending on this crate (layer 10), which would violate spec 4.2
//! layering. Re-exported here so existing callers of `es_telemetry::ring::RingBuffer` see no
//! change.

pub use es_core::ring::RingBuffer;
