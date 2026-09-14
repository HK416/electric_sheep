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

/// Every runtime and configuration error `es-ros2` can report (`docs/design/ros2-boundary.md`
/// section 4.2's table, plus a few conditions that table leaves unnumbered: config parsing and
/// actuator lookup). [`Ros2Error::code`] is the stable string a caller logs or matches on; the
/// `Display` message (via `thiserror`) is for humans only and may change.
#[derive(Debug, Error)]
pub enum Ros2Error {
    /// TOML syntax, an unknown key, or a missing/malformed required field. Not in section 4.2's
    /// table (that table is about mode/probe conditions); this is the catch-all for "the file
    /// itself does not parse".
    #[error("ros2 config: {0}")]
    Config(String),
    /// Both `[ros2.rmw_zenoh]` and `[ros2.dds_bridge]` are present: modes A and B are mutually
    /// exclusive (spec 24.1).
    #[error("[ros2.rmw_zenoh] and [ros2.dds_bridge] cannot both be configured (A xor B)")]
    BothModesConfigured,
    /// `[ros2.rust_dds]` (mode C) is present: parsed only so this rejection is specific, mode C
    /// is not built.
    #[error("[ros2.rust_dds] (mode C) is not implemented")]
    ModeCNotBuilt,
    /// Mode A's startup probe found a `zenoh-plugin-ros2dds` bridge token already on the
    /// network.
    #[error("mode A startup probe found a bridge token: {0}")]
    ModeAFoundBridge(String),
    /// Mode B's startup probe found no bridge-plugin liveliness token: there is no bridge to
    /// talk to.
    #[error("mode B startup probe found no bridge plugin token")]
    ModeBNoBridge,
    /// Mode B's startup probe found a plain `rmw_zenoh` node token: the two modes would collide
    /// on the same domain.
    #[error("mode B startup probe found an rmw_zenoh token: {0}")]
    ModeBFoundRmwZenoh(String),
    /// A liveliness subscriber armed at startup saw a conflicting token appear later: the node
    /// latches this and fails every later `put`/`send` closed (spec 24.1, design note 4.2).
    #[error("a conflicting liveliness token appeared after startup: {0}")]
    LatchedConflict(String),
    /// A generic [`crate::session::Publisher`] was requested on a topic reserved for
    /// [`crate::actuator::ActuatorPublisher`] (design note section 4.5).
    #[error("`{0}` is a reserved actuator topic; publish through Ros2Node::actuator instead")]
    ReservedActuatorTopic(String),
    /// `Publisher::put` was given a [`crate::msg::Msg`] whose type does not match the one the
    /// publisher was declared with.
    #[error("message type does not match the publisher's declared type")]
    TypeMismatch,
    /// `TRANSIENT_LOCAL` was requested. Refused: the zenoh-ext advanced-pub/sub key literals it
    /// needs are unverified (`docs/api-notes/rmw-zenoh.md` "Payload and data path").
    #[error("TRANSIENT_LOCAL is not supported")]
    TransientLocalUnsupported,
    /// `Ros2Config` has no `[[ros2.actuator]]` entry for the requested topic.
    #[error("no configured actuator topic `{0}`")]
    UnknownActuatorTopic(String),
    /// The requested `ActuatorPublisher::<NJ>`'s `NJ` does not match the configured topic's
    /// joint count (design note section 4.5: "`NJ != joints.len()` fails at construction").
    #[error("actuator topic `{topic}` has {found} configured joints, expected {expected}")]
    ActuatorJointCount {
        topic: String,
        expected: usize,
        found: usize,
    },
    /// An inbound `sensor_msgs/JointState` did not carry every joint the caller asked to
    /// reorder by name (design note section 4.5).
    #[error("JointState is missing joint `{0}`")]
    MissingJoint(String),
    /// A zenoh session, publisher, subscriber or liveliness operation failed. Wraps the
    /// library's own message; only built when the `zenoh` feature is on.
    #[cfg(feature = "zenoh")]
    #[error("zenoh: {0}")]
    Zenoh(String),
}

impl Ros2Error {
    /// The stable error code, exactly the codes of `docs/design/ros2-boundary.md` section 4.2
    /// for the conditions that table lists (`ROS2-001` .. `ROS2-012`); `ROS2-000`, `ROS2-013`
    /// and `ROS2-020` cover conditions the table does not number (config parsing, actuator
    /// lookup, and zenoh transport failures respectively).
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Ros2Error::Config(_) => "ROS2-000",
            Ros2Error::BothModesConfigured => "ROS2-001",
            Ros2Error::ModeCNotBuilt => "ROS2-002",
            Ros2Error::ModeAFoundBridge(_) => "ROS2-003",
            Ros2Error::ModeBNoBridge => "ROS2-004",
            Ros2Error::ModeBFoundRmwZenoh(_) => "ROS2-005",
            Ros2Error::LatchedConflict(_) => "ROS2-006",
            Ros2Error::ReservedActuatorTopic(_) => "ROS2-010",
            Ros2Error::TypeMismatch => "ROS2-011",
            Ros2Error::TransientLocalUnsupported => "ROS2-012",
            Ros2Error::UnknownActuatorTopic(_)
            | Ros2Error::ActuatorJointCount { .. }
            | Ros2Error::MissingJoint(_) => "ROS2-013",
            #[cfg(feature = "zenoh")]
            Ros2Error::Zenoh(_) => "ROS2-020",
        }
    }
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
