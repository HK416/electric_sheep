//! `es-actuator` (layer 3): the actuator transduction model — motor/position/velocity/general
//! force laws, control/force-range clamping, a fixed transport delay, saturation and a small
//! backlash model. See `docs/ARCHITECTURE.ko.md` spec 18.2 (actuators) and
//! `docs/packets/M1/W2-sensor-actuator.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math` and external crates
//! only. `crates/es-assets::scene::ActuatorKind` (layer 2) is a legal dependency but is
//! mirrored instead of imported, to keep this crate's surface self-contained and tiny.

pub mod model;

pub use model::{ActuatorDelay, ActuatorError, ActuatorModel, Backlash, Saturation};
