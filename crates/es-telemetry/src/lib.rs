//! `es-telemetry` (layer 10): telemetry protocol skeleton (P40). See
//! `docs/ARCHITECTURE.ko.md` spec 23 (editor/debugger), spec 25.1 (security), spec 25.3
//! (versioning), spec 12.4 (performance metrics), spec 9.6 (embedded runtime ring buffer) and
//! `docs/packets/M0/P40.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-math`, `es-core`, `es-ir` and external
//! crates only. It defines the wire *schema*, an in-memory ring buffer, and (as of the M1
//! W8 packet) a loopback TCP transport built on `std::net` alone — no new dependency, no QUIC,
//! no TLS. QUIC/zenoh and TLS are `es-transport` (layer 11, the only crate allowed to link
//! CUDA/HIP) work for a later packet; see [`transport`] for exactly what this stopgap does and
//! does not provide.
//!
//! # Backend-neutral: the Python adapter
//!
//! Spec 23.1: the telemetry protocol is not Electric Sheep specific. A thin Python adapter can
//! emit [`protocol::Frame`]s from `LeRobot`, Isaac Lab, or Newton training loops without linking
//! any ES code, by writing the same length-prefixed-JSON shape [`protocol::encode`] produces.
//! The JSON body of one frame message is:
//!
//! ```json
//! {
//!   "type": "frame",
//!   "tick": 12345,
//!   "wall_ns": 1731500000000000000,
//!   "stream": 3,
//!   "payload": { "kind": "scalar", "value": 0.42 }
//! }
//! ```
//!
//! Only `stream` (an integer stream id), `tick` (an integer step count) and `payload` are
//! required — nothing ES-specific. `payload.kind` is one of `scalar`, `scalars`, `tensor`,
//! `image`, `event`, or `metrics`; see [`protocol::Payload`] for each shape. An adapter that
//! only ever sends `scalar`/`scalars`/`metrics` payloads (loss curves, throughput) needs no
//! tensor or image encoding at all.

pub mod protocol;
pub mod ring;
pub mod transport;

pub use protocol::{
    decode, decode_with_max, encode, encode_with, negotiate, Codec, Frame, Hello, HelloAck,
    Message, Payload, PerfMetrics, ProtoError, StreamId, DEFAULT_MAX_FRAME_BYTES, PROTOCOL_VERSION,
};
pub use ring::RingBuffer;
pub use transport::{Client, ClientStats, Server, ServerStats, TransportError};
