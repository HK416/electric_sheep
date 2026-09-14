//! `es-ros2` (layer 11): the ROS 2 boundary — wire codec, key expressions, liveliness and
//! HIL. See `docs/ARCHITECTURE.ko.md` §24.1 (ROS 2 boundary), §25.1 (security), §1.4
//! (oracles), `docs/design/ros2-boundary.md`, `docs/api-notes/ros2-cdr.md`,
//! `docs/api-notes/rmw-zenoh.md`, and the work packet `docs/packets/M3/W1a-ros2-cdr-keyexpr.md`.
//!
//! This packet (W1a) is the half of the ROS 2 boundary that needs no network and no `zenoh`
//! dependency, so it is judged the same way on any machine: [`cdr`] is a hand-rolled
//! little-endian CDR codec for the fixed message subset in [`msg`]; [`names`] builds and parses
//! `rmw_zenoh` topic key expressions and liveliness tokens; [`attachment`] is the 33-byte
//! `rmw_zenoh` sample attachment and its GID derivation. The zenoh session itself (W1b), camera
//! ingest (W1c) and HIL (W1d) are later packets in the same crate.
//!
//! Layer rule (spec 4.2): this crate is layer 11 and depends on `es-core` (layer 1) and
//! `es-safety` (layer 8) among workspace crates. No trait is added here (INV-17): message
//! dispatch is the [`msg::MsgType`] / [`msg::Msg`] enums, not a trait object.
//!
//! W1d adds [`hil`]: the hardware-in-the-loop UDP link, its `.eshil` input log and the
//! byte-identical replay that is the M3 gate (spec 24.2, design note section 7). It is a
//! module rather than an `es-hil` crate because spec 4.2's layer table has no such crate
//! (design note section 2); nothing in it imports a ROS module.
//!
//! W1b adds the zenoh session (this module list's [`config`], [`session`], [`actuator`]), gated
//! behind the `zenoh` cargo feature (off by default, design note section 2) so `es` and any
//! embedded consumer never link it. [`config`] itself needs no feature: parsing a
//! [`config::Ros2Config`] is plain data, checked the same way on every machine.
//!
//! W1c adds [`camera`]: `sensor_msgs/CameraInfo` + `sensor_msgs/Image` become a validated
//! `es_ir::image::ImageSpec`, an HWC byte buffer and a `PhysTick` (design note section 6,
//! spec 7.2, INV-14). It needs no feature and no network either — a message in, a checked
//! frame out.

#[cfg(feature = "zenoh")]
pub mod actuator;
pub mod attachment;
pub mod camera;
pub mod cdr;
pub mod config;
mod error;
pub mod hil;
pub mod msg;
pub mod names;
#[cfg(feature = "zenoh")]
pub mod session;
