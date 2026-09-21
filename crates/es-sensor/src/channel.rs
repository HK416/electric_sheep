//! The channel contract (spec 15.1, spec 18.3): the fixed set of tensor channels a camera
//! sensor can produce between the renderer and Observation IR capture, and the camera-level
//! timing contract (`CameraContract`) that carries the `ImageSpec` fields (spec 7.2) a sensor
//! must populate: shutter model, exposure, rate.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use es_core::TickRate;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// One channel a camera-like sensor can emit (spec 15.1 render -> observation path).
///
/// `es-sensor` sits below `es-ir` (layer 3 vs layer 6, spec 4.2) and cannot use
/// `es_ir::Unit`, so [`Channel::unit`] returns a plain string name instead. Those names are
/// the contract `es-ir` matches on when it consumes a `CameraContract`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Channel {
    /// Tone-mapped, gamma-encoded color, 3 x u8.
    Rgb8,
    /// Linear-light HDR color, 3 x f32.
    RgbF32Linear,
    /// Depth, 1 x f32. `unit_m` is meters per raw unit (mirrors `ImageSpec::depth_scale`,
    /// spec 7.2); `1.0` means the value is already in meters.
    Depth32 { unit_m: f64 },
    /// Geometric surface normal in camera space, 3 x f32, unit length.
    Normal,
    /// Per-pixel instance/class id, 1 x u32.
    SegmentationId,
    /// Optical flow, 2 x f32 (dx, dy) in pixels.
    Flow,
    /// Point/splat radiance (unverified: assumed 3 x f32 RGB; no alpha channel modeled here).
    PtRadiance,
    /// Temporal history length in frames, 1 x u32 (packet M7/R4): how many accumulated frames
    /// the path tracer's estimate at this pixel rests on. A renderer diagnostic and a mask —
    /// no Observation IR node reads it.
    History,
}

/// A channel's scalar storage type. Mirrors the subset of `ImageSpec::ImageDType` (spec 7.2:
/// `U8 | U16 | F16 | F32`) that channels here actually use, plus `U32` for id maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelDType {
    U8,
    U32,
    F32,
}

impl Channel {
    /// Per-element storage type.
    #[must_use]
    pub fn dtype(self) -> ChannelDType {
        match self {
            Self::Rgb8 => ChannelDType::U8,
            Self::SegmentationId | Self::History => ChannelDType::U32,
            Self::RgbF32Linear
            | Self::Depth32 { .. }
            | Self::Normal
            | Self::Flow
            | Self::PtRadiance => ChannelDType::F32,
        }
    }

    /// Number of scalar components per sample.
    #[must_use]
    pub fn n_components(self) -> u8 {
        match self {
            Self::Rgb8 | Self::RgbF32Linear | Self::Normal | Self::PtRadiance => 3,
            Self::Depth32 { .. } | Self::SegmentationId | Self::History => 1,
            Self::Flow => 2,
        }
    }

    /// Unit name as a plain string, the vocabulary `es-ir` matches on (see module docs).
    /// Fixed set: `"srgb"`, `"linear_rgb"`, `"m"`, `"unit_normal"`, `"id"`, `"px_flow"`,
    /// `"radiance"`, `"frames"`.
    #[must_use]
    pub fn unit(self) -> &'static str {
        match self {
            Self::Rgb8 => "srgb",
            Self::RgbF32Linear => "linear_rgb",
            Self::Depth32 { .. } => "m",
            Self::Normal => "unit_normal",
            Self::SegmentationId => "id",
            Self::Flow => "px_flow",
            Self::PtRadiance => "radiance",
            Self::History => "frames",
        }
    }

    /// Stable per-variant order key; `Depth32`'s `f64` field has no `Ord`, so `Channel` orders
    /// by variant first and, for `Depth32`, by IEEE-754 bit pattern (never arithmetic on it).
    fn sort_key(self) -> (u8, u64) {
        match self {
            Self::Rgb8 => (0, 0),
            Self::RgbF32Linear => (1, 0),
            Self::Depth32 { unit_m } => (2, unit_m.to_bits()),
            Self::Normal => (3, 0),
            Self::SegmentationId => (4, 0),
            Self::Flow => (5, 0),
            Self::PtRadiance => (6, 0),
            Self::History => (7, 0),
        }
    }
}

impl PartialEq for Channel {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for Channel {}

impl PartialOrd for Channel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Channel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

/// Camera shutter model (spec 7.2 `ImageSpec::shutter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shutter {
    Global,
    /// Row-sequential readout over `readout_ticks` physics ticks (spec 18.3).
    Rolling {
        readout_ticks: u64,
    },
}

/// Validation failure building a [`CameraContract`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContractError {
    #[error("camera contract must declare at least one channel")]
    NoChannels,
    #[error("camera contract width and height must be non-zero (got {width}x{height})")]
    ZeroExtent { width: u32, height: u32 },
}

/// The channel contract for one camera sensor (spec 15.1, spec 7.2): which channels it
/// produces and the timing parameters (`rate`, `shutter`, `exposure_ticks`) that feed
/// `ImageSpec`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraContract {
    pub channels: BTreeSet<Channel>,
    pub width: u32,
    pub height: u32,
    pub rate: TickRate,
    pub shutter: Shutter,
    pub exposure_ticks: u64,
}

impl CameraContract {
    /// # Errors
    /// [`ContractError::NoChannels`] if `channels` is empty, [`ContractError::ZeroExtent`] if
    /// `width` or `height` is zero.
    pub fn new(
        channels: BTreeSet<Channel>,
        width: u32,
        height: u32,
        rate: TickRate,
        shutter: Shutter,
        exposure_ticks: u64,
    ) -> Result<Self, ContractError> {
        if channels.is_empty() {
            return Err(ContractError::NoChannels);
        }
        if width == 0 || height == 0 {
            return Err(ContractError::ZeroExtent { width, height });
        }
        Ok(Self {
            channels,
            width,
            height,
            rate,
            shutter,
            exposure_ticks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_and_components() {
        assert_eq!(Channel::Rgb8.dtype(), ChannelDType::U8);
        assert_eq!(Channel::Rgb8.n_components(), 3);
        assert_eq!(Channel::SegmentationId.dtype(), ChannelDType::U32);
        assert_eq!(Channel::SegmentationId.n_components(), 1);
        assert_eq!(Channel::Flow.n_components(), 2);
        assert_eq!(Channel::Depth32 { unit_m: 1.0 }.dtype(), ChannelDType::F32);
    }

    #[test]
    fn unit_names_are_fixed() {
        assert_eq!(Channel::Rgb8.unit(), "srgb");
        assert_eq!(Channel::Depth32 { unit_m: 1.0 }.unit(), "m");
        assert_eq!(Channel::Flow.unit(), "px_flow");
    }

    #[test]
    fn depth_orders_by_bits_not_value() {
        let a = Channel::Depth32 { unit_m: 1.0 };
        let b = Channel::Depth32 { unit_m: 2.0 };
        assert_ne!(a, b);
        assert!(a < b);
        assert!(Channel::Rgb8 < Channel::Depth32 { unit_m: 0.0 });
    }

    #[test]
    fn contract_rejects_empty_channels_and_zero_extent() {
        let rate = TickRate::hz(30);
        assert_eq!(
            CameraContract::new(BTreeSet::new(), 640, 480, rate, Shutter::Global, 0),
            Err(ContractError::NoChannels)
        );
        let mut ch = BTreeSet::new();
        ch.insert(Channel::Rgb8);
        assert_eq!(
            CameraContract::new(ch.clone(), 0, 480, rate, Shutter::Global, 0),
            Err(ContractError::ZeroExtent {
                width: 0,
                height: 480
            })
        );
        assert!(CameraContract::new(ch, 640, 480, rate, Shutter::Global, 100).is_ok());
    }

    #[test]
    fn contract_serde_round_trip() {
        let mut channels = BTreeSet::new();
        channels.insert(Channel::Rgb8);
        channels.insert(Channel::Depth32 { unit_m: 1.0 });
        let contract = CameraContract::new(
            channels,
            320,
            240,
            TickRate::hz(60),
            Shutter::Rolling { readout_ticks: 4 },
            2,
        )
        .unwrap();
        let json = serde_json::to_string(&contract).unwrap();
        let back: CameraContract = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, contract.width);
        assert_eq!(back.height, contract.height);
        assert_eq!(back.channels, contract.channels);
        assert_eq!(back.shutter, contract.shutter);
        assert_eq!(back.exposure_ticks, contract.exposure_ticks);
    }
}
