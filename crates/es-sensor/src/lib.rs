//! `es-sensor` (layer 3): the sensor channel contract and CPU-side sensor realism models
//! (noise, latency, dropout, rolling shutter) that sit between a renderer/physics backend and
//! Observation IR capture. See `docs/ARCHITECTURE.ko.md` spec 18.3 (sensors and realism),
//! spec 15.1 (render -> observation path, channel contract), spec 7.2 (`ImageSpec` fields a
//! camera sensor populates) and `docs/packets/M1/W2-sensor-actuator.md`.
//!
//! Layer rule (spec 4.2): this crate may depend on `es-core`, `es-math` and external crates
//! only. It is below `es-ir` (layer 6) and `es-env` (layer 9), so it defines its own tiny
//! `Unit` string vocabulary (channel.rs) and its own RNG core (noise.rs) instead of depending
//! on either.

pub mod channel;
pub mod noise;
pub mod sensor;

pub use channel::{CameraContract, Channel, ChannelDType, ContractError, Shutter};
pub use noise::{sensor_seed, NoiseRng, RollingShutterSkew, SensorModel, Stage};
pub use sensor::{SensorBank, SensorDesc, SensorError};
