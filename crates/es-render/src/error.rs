//! Errors. Everything this crate refuses, it refuses loudly: a renderer that silently drops
//! a channel or silently resamples a view would break the `ImageSpec` contract of spec 7.2
//! downstream (`INV-14`) where nobody is looking.

use es_sensor::Channel;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("{0}")]
    Gpu(#[from] es_gpu::GpuError),

    /// A `Shape` this crate cannot tessellate (spec 15: meshes need an asset resolver this
    /// packet does not own; height fields need none at all yet).
    #[error("unsupported shape in geom '{geom}': {shape}")]
    UnsupportedShape { geom: String, shape: &'static str },

    /// A channel the selected render path does not produce (see `docs/design/renderer.md`).
    #[error("render path does not produce channel {channel:?}")]
    UnsupportedChannel { channel: Channel },

    /// A view whose resolution differs from the tile. Resampling here would change the
    /// intrinsics (spec 7.2 `OBS-034`), so it is refused instead.
    #[error("view {view} is {width}x{height}, tile is {tile_w}x{tile_h}")]
    ViewTileMismatch {
        view: usize,
        width: u32,
        height: u32,
        tile_w: u32,
        tile_h: u32,
    },

    /// More views than the atlas was configured for.
    #[error("{views} views do not fit an atlas of {capacity} tiles")]
    AtlasTooSmall { views: usize, capacity: usize },

    /// Atlas wider or taller than `maxImageDimension2D` (spec 15.2).
    #[error("atlas {width}x{height} exceeds maxImageDimension2D {limit} (spec 15.2)")]
    AtlasTooLarge { width: u64, height: u64, limit: u32 },

    #[error("invalid render config: {0}")]
    Config(String),
}
