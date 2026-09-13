//! The tile atlas (spec 15.2): N cameras packed into one buffer per channel, with a closed
//! -form tile origin so a downstream consumer reconstructs per-env tensors by arithmetic and
//! never by a host transfer.

use es_sensor::{Channel, ChannelDType};

use crate::error::RenderError;
use crate::view::TileAtlasCfg;

/// `maxImageDimension2D` on the desktop GPUs spec 15.2 targets. The atlas is a buffer here,
/// not a `VkImage`, so the limit does not physically bind — it is enforced anyway, because
/// `es-compile`'s budget model enforces it and a layout this crate accepts but the budget
/// rejects would be a lie.
pub const MAX_IMAGE_DIMENSION_2D: u32 = 16384;

/// Atlas geometry derived from a [`TileAtlasCfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasLayout {
    pub cfg: TileAtlasCfg,
    pub rows: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasLayout {
    pub fn new(cfg: TileAtlasCfg) -> Result<Self, RenderError> {
        if cfg.tile_w == 0 || cfg.tile_h == 0 || cfg.tiles_per_row == 0 || cfg.n_tiles == 0 {
            return Err(RenderError::Config(
                "tile_w, tile_h, tiles_per_row and n_tiles must all be non-zero".to_owned(),
            ));
        }
        let rows = cfg.n_tiles.div_ceil(cfg.tiles_per_row);
        let width = u64::from(cfg.tiles_per_row) * u64::from(cfg.tile_w);
        let height = u64::from(rows) * u64::from(cfg.tile_h);
        if width > u64::from(MAX_IMAGE_DIMENSION_2D) || height > u64::from(MAX_IMAGE_DIMENSION_2D) {
            return Err(RenderError::AtlasTooLarge {
                width,
                height,
                limit: MAX_IMAGE_DIMENSION_2D,
            });
        }
        Ok(Self {
            cfg,
            rows,
            width: width as u32,
            height: height as u32,
        })
    }

    /// Top-left pixel of tile `i` in the atlas. Closed form, not a table: the layout has to
    /// be reproducible without shipping it (spec 15.2).
    pub fn tile_origin(&self, i: u32) -> (u32, u32) {
        (
            (i % self.cfg.tiles_per_row) * self.cfg.tile_w,
            (i / self.cfg.tiles_per_row) * self.cfg.tile_h,
        )
    }

    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Planning size of one channel's atlas, **mirroring** the `render_tile_atlas` item of
    /// `crates/es-compile/src/budget.rs` (spec 20.2) term for term, including its `* 2`
    /// double-buffer factor:
    ///
    /// ```text
    /// rows * tiles_per_row * tile_w * tile_h * components * dtype_bytes * 2
    /// ```
    ///
    /// `es-compile` is layer 7 and cannot be depended on from layer 5 (spec 4.2), so the
    /// arithmetic is duplicated here and pinned by a test. This is *not* what the renderer
    /// allocates — see [`Self::device_bytes`].
    pub fn atlas_bytes(&self, channel: Channel) -> u64 {
        let dtype_bytes = match channel.dtype() {
            ChannelDType::U8 => 1,
            ChannelDType::U32 | ChannelDType::F32 => 4,
        };
        u64::from(self.rows)
            * u64::from(self.cfg.tiles_per_row)
            * u64::from(self.cfg.tile_w)
            * u64::from(self.cfg.tile_h)
            * u64::from(channel.n_components())
            * dtype_bytes
            * 2
    }

    /// What the renderer actually allocates for one channel: single-buffered, and one 32-bit
    /// word per component on the device.
    ///
    /// Every device-side channel is a word buffer — `f32` for the float channels, `u32` for
    /// segmentation, and one word of packed `RGBA8` for [`Channel::Rgb8`] (a compute shader
    /// cannot write three bytes of a word without racing its neighbour; the unused alpha is
    /// dropped on readback). So this is not `atlas_bytes / 2`, and both names exist so that
    /// neither number is quietly reported as the other.
    pub fn device_bytes(&self, channel: Channel) -> u64 {
        self.pixels() * u64::from(words_per_pixel(channel)) * 4
    }
}

/// 32-bit words one pixel of `channel` occupies on the device. See [`AtlasLayout::device_bytes`].
pub fn words_per_pixel(channel: Channel) -> u32 {
    match channel {
        Channel::Rgb8 | Channel::Depth32 { .. } | Channel::SegmentationId => 1,
        Channel::Flow => 2,
        Channel::RgbF32Linear | Channel::Normal | Channel::PtRadiance => 3,
    }
}

/// One camera's slice of one channel, as a tensor: row-major `[h, w, c]`, tightly packed.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub shape: [usize; 3],
    pub data: TileData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TileData {
    U8(Vec<u8>),
    U32(Vec<u32>),
    F32(Vec<f32>),
}

impl Tile {
    pub fn as_u8(&self) -> Option<&[u8]> {
        match &self.data {
            TileData::U8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&[u32]> {
        match &self.data {
            TileData::U32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            TileData::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Raw little-endian bytes, the form the golden files store.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            TileData::U8(v) => v.clone(),
            TileData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TileData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TileData::U8(v) => v.len(),
            TileData::U32(v) => v.len(),
            TileData::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: u32, per_row: u32) -> TileAtlasCfg {
        TileAtlasCfg {
            tile_w: 224,
            tile_h: 224,
            tiles_per_row: per_row,
            n_tiles: n,
        }
    }

    #[test]
    fn tile_origins_are_row_major_and_padded() {
        let l = AtlasLayout::new(cfg(5, 2)).unwrap();
        assert_eq!(l.rows, 3);
        assert_eq!((l.width, l.height), (448, 672));
        assert_eq!(l.tile_origin(0), (0, 0));
        assert_eq!(l.tile_origin(1), (224, 0));
        assert_eq!(l.tile_origin(2), (0, 224));
        assert_eq!(l.tile_origin(4), (0, 448));
    }

    /// The formula must stay term-for-term identical to the `render_tile_atlas` item of
    /// `crates/es-compile/src/budget.rs` (spec 20.2). Recomputed by hand here.
    // The `* 1` and `* 4` are the dtype-bytes term of the mirrored formula; spelling
    // them out is the point of the test.
    #[allow(clippy::identity_op)]
    #[test]
    fn atlas_bytes_mirrors_the_compile_budget_formula() {
        let l = AtlasLayout::new(cfg(512, 23)).unwrap();
        assert_eq!(l.rows, 23);
        let by_hand = 23_u64 * 23 * 224 * 224 * 3 * 1 * 2;
        assert_eq!(l.atlas_bytes(Channel::Rgb8), by_hand);
        assert_eq!(by_hand, 159_258_624);
        // Depth is 1 component of 4 bytes.
        assert_eq!(
            l.atlas_bytes(Channel::Depth32 { unit_m: 1.0 }),
            23_u64 * 23 * 224 * 224 * 1 * 4 * 2
        );
    }

    #[test]
    fn device_bytes_is_one_word_per_component() {
        let l = AtlasLayout::new(cfg(4, 2)).unwrap();
        assert_eq!(l.pixels(), 448 * 448);
        // Rgb8 is one packed RGBA8 word, not three bytes.
        assert_eq!(l.device_bytes(Channel::Rgb8), l.pixels() * 4);
        assert_eq!(l.device_bytes(Channel::Normal), l.pixels() * 12);
    }

    #[test]
    fn atlas_larger_than_max_image_dimension_is_refused() {
        let big = TileAtlasCfg {
            tile_w: 1024,
            tile_h: 1024,
            tiles_per_row: 32,
            n_tiles: 32,
        };
        assert!(matches!(
            AtlasLayout::new(big),
            Err(RenderError::AtlasTooLarge { .. })
        ));
    }

    #[test]
    fn zero_sized_config_is_refused() {
        assert!(AtlasLayout::new(TileAtlasCfg::row(0, 8, 1)).is_err());
        assert!(AtlasLayout::new(TileAtlasCfg::row(8, 8, 0)).is_err());
    }
}
