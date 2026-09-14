//! Shared error types for `names` and `attachment` (spec 25.1: bytes and strings that cross a
//! trust boundary get a total, non-panicking decoder and an explicit error, never a default).
//!
//! `cdr::CdrError` lives in [`crate::cdr`] instead of here: it is tightly coupled to that
//! module's byte-level decode logic. This file holds the two error types shared by the
//! higher-level `names` and `attachment` modules, re-exported from there (`pub use`) so callers
//! see `es_ros2::names::NameError` / `es_ros2::attachment::AttachmentError` as documented in the
//! work packet's acceptance criteria.

use thiserror::Error;

/// Why a ROS name, key expression, or liveliness token failed to parse or build
/// (`docs/design/ros2-boundary.md` section 3 "Names").
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NameError {
    /// A ROS name did not start with `/` (relative names are resolved by the caller, not here).
    #[error("ROS name `{0}` is not absolute (must start with '/')")]
    NotAbsolute(String),
    /// Two consecutive `/`, or a trailing `/` (an empty segment).
    #[error("ROS name `{0}` has an empty segment")]
    EmptySegment(String),
    /// A segment is not `[A-Za-z_][A-Za-z0-9_]*`.
    #[error("ROS name `{0}` has an invalid segment")]
    InvalidSegment(String),
    /// A liveliness token did not start with the `@ros2_lv` admin-space chunk.
    #[error("not an rmw_zenoh liveliness token: `{0}`")]
    NotALivelinessToken(String),
    /// Too few `/`-delimited chunks to hold the required fields.
    #[error("liveliness token has too few chunks: `{0}`")]
    TooFewChunks(String),
    /// The entity-kind chunk (`NN`/`MP`/`MS`/`SS`/`SC`) was not one of the known codes.
    #[error("unknown liveliness entity kind `{0}`")]
    UnknownKind(String),
    /// A chunk expected to be a decimal integer (domain id, node id, entity id) was not.
    #[error("expected a decimal integer, got `{0}`")]
    NotAnInteger(String),
    /// The trailing `QoS` chunk did not have the six `:`-separated / eleven-value shape.
    #[error("malformed QoS chunk `{0}`")]
    BadQos(String),
}

/// Why a 33-byte `rmw_zenoh` attachment failed to decode (`docs/api-notes/rmw-zenoh.md`
/// "Attachment: 33 bytes").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AttachmentError {
    /// Attachment payloads are always exactly 33 bytes; this one was not.
    #[error("attachment is {0} bytes, expected 33")]
    BadLength(usize),
    /// Byte 16 must be the LEB128 length prefix `0x10` (16), the GID's fixed size.
    #[error("attachment GID length prefix is 0x{0:02x}, expected 0x10")]
    BadGidLength(u8),
}
