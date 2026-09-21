//! `es-render` (layer 5, spec 4.2): the camera renderer of spec 15.
//!
//! It produces the spec 15.1 channel contract for many cameras at once, packed into one tile
//! atlas per channel (spec 15.2), through one of two **compute** render paths (spec 15.3):
//!
//! * [`RenderPath::Rs`] — a screen-space scan rasterizer, the default for vision learning;
//! * [`RenderPath::Pt`] — a path tracer with optional `ReSTIR` DI and an a-trous denoise
//!   (spec 28.6). Spec 1.9 item 2 cuts this second under scope pressure, so it is the
//!   minimal honest version and the design doc lists what it skips.
//!
//! `es-gpu` offers compute pipelines only — no graphics pipeline and no ray-tracing
//! extension — so both paths are compute shaders and spec 15.4's TLAS/BLAS is *not*
//! implemented as hardware: they traverse a software [`bvh`] built on the CPU per frame
//! (packet M7/R1), which returns exactly what the earlier flat index-order scan returned.
//!
//! The oracle is [`cpu`], a pure-Rust mirror of both shaders. It generates every golden in
//! `tests/golden/render/`; the GPU never does (spec 1.4).
//!
//! Spec 16.2 places 3DGS as a third render path in this crate. The hook is
//! [`RenderPath`]: a `Splat` variant writes the same channels into the same atlas and needs
//! no change to [`Atlas`], [`Tile`] or the channel contract. `es-splat` owns the asset side.
//!
//! Design note: `docs/design/renderer.md`. Work packets: `docs/packets/M1/W2-tile-atlas.md`,
//! `docs/packets/M4/W8-renderer.md`.
//!
//! # Determinism (spec 3.4)
//!
//! No atomics, no shared memory, no subgroup operations, no dependence on workgroup count.
//! Every transcendental goes through `es_math::approx` / `approx.slang` (spec 3.2
//! `DET-010`). The RNG is counter-based and addressed by `(view, pixel, sample, bounce,
//! stream)`, never global. `BTreeMap` everywhere; no `HashMap`. What this buys is *same
//! device, same binary -> same bits*, plus CPU-to-GPU agreement at the tolerances stated in
//! the design doc — not bit equality across devices, which spec 3.5 tier 1 puts behind the
//! `hardware_capability` term of the execution hash.
#![forbid(unsafe_code)]

pub mod atlas;
pub mod bvh;
pub mod cornell;
pub mod cpu;
pub mod error;
pub mod renderer;
pub mod rng;
pub mod scene;
pub mod ssim;
pub mod view;

pub use atlas::{AtlasLayout, Tile, TileData};
pub use cpu::Frame;
pub use error::RenderError;
pub use renderer::{Atlas, Renderer};
pub use scene::{SceneCache, Tri, TriScene};
pub use ssim::ssim;
pub use view::{
    CameraView, ImageSpec, Intrinsics, RenderConfig, RenderPath, Shading, TileAtlasCfg, Tonemap,
    ViewParams,
};

/// Re-exported so naming a channel does not oblige a caller to depend on `es-sensor` as well
/// (`es-env`'s `render` feature, `docs/packets/M5/V0b-render-in-the-loop.md`).
pub use es_sensor::Channel;

/// Channels the [`RenderPath::Rs`] path writes (spec 15.1). Anything else is
/// [`RenderError::UnsupportedChannel`].
pub const RS_CHANNELS: [Channel; 4] = [
    Channel::Rgb8,
    Channel::Depth32 { unit_m: 1.0 },
    Channel::Normal,
    Channel::SegmentationId,
];

/// Channels the [`RenderPath::Pt`] path writes: linear radiance, the tone-mapped `Rgb8` of
/// packet M7/R3, and the three geometry channels spec 15.3 requires to be bit-identical
/// with `Rs`.
pub const PT_CHANNELS: [Channel; 5] = [
    Channel::PtRadiance,
    Channel::Rgb8,
    Channel::Depth32 { unit_m: 1.0 },
    Channel::Normal,
    Channel::SegmentationId,
];

/// Whether `path` produces `channel`.
pub fn path_produces(path: RenderPath, channel: Channel) -> bool {
    match path {
        RenderPath::Rs => RS_CHANNELS.contains(&channel),
        RenderPath::Pt { .. } => PT_CHANNELS.contains(&channel),
    }
}
