//! `es-splat` (layer 5): 3D Gaussian Splatting assets — the offline half of spec 16's
//! real-to-sim path.
//!
//! Import a capture ([`import_ply`]), align it to the physics scene in position
//! ([`Similarity`]) and colour ([`ColorAffine`]), and bind it to articulated bodies so the
//! visual follows the physics ([`Binding`], [`skin`]). Nothing here renders: the splat
//! rasteriser of spec 16.3 needs Vulkan and is a separate packet.
//!
//! Design note: `docs/design/splat-real2sim.md`. File layout: `docs/api-notes/
//! gaussian-splat-ply.md` — where every field is marked `unverified`, because there is no
//! reference 3DGS implementation in this workspace to have checked it against.
//!
//! Conventions are spec 3.1's, carried by `es-math`: right-handed Z-up, metres, quaternions
//! xyzw with `w >= 0`. A 3DGS reconstruction is not in that frame; see [`ply`] for the fixed
//! (and `unverified`) conversion.
//!
//! Per spec 1.9 this whole path is cut third under scope pressure, so it stays small: closed
//! -form fits, no iterative refinement, no dependency beyond the three `es-*` crates below.
#![forbid(unsafe_code)]

mod align;
mod bind;
mod ply;

pub use align::{ColorAffine, Similarity};
pub use bind::{skin, Binding, SkinnedSplats};
pub use ply::{import_ply, SplatError};

use es_assets::scene::{scene_id, AssetKind, AssetRef};
use es_math::Vec3;

/// Domain separator for [`SplatScene::asset_hash`]; changing the encoding below must change
/// every stored hash (spec 5.3).
const SPLAT_TAG: &str = "es.splat.v1";

/// The degree-0 spherical-harmonic basis value, `1 / (2 * sqrt(pi))`.
///
/// `rgb = 0.5 + SH_C0 * dc` is the reference implementation's convention (`unverified`).
pub const SH_C0: f32 = 0.282_094_79;

/// Axis-aligned bounds over [`SplatScene::positions`], in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Bounds {
    /// An empty box: `min` above `max`, so any union with a real point is that point.
    fn default() -> Self {
        Self {
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
}

/// A decoded Gaussian splat capture, structure of arrays.
///
/// Every array has length `count` times its component count, and index `i` names the same
/// Gaussian in all of them. Activations are already applied (see `write_ply` for what that
/// costs on a round trip): `scales` are standard deviations in metres, `opacities` are alpha
/// in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct SplatScene {
    /// `3 * count`, metres, spec 3.1 Z-up world frame.
    pub positions: Vec<f32>,
    /// `3 * count`, metres, per-axis standard deviation in the Gaussian's *own* frame.
    pub scales: Vec<f32>,
    /// `4 * count`, xyzw, unit norm, `w >= 0`.
    pub rotations: Vec<f32>,
    /// `count`, alpha in `[0, 1]`.
    pub opacities: Vec<f32>,
    /// `3 * count`, the degree-0 SH coefficient per channel (not RGB; see [`SplatScene::rgb`]).
    pub sh_dc: Vec<f32>,
    /// `3 * ((sh_degree + 1)^2 - 1) * count`, channel-major, or empty for degree 0.
    pub sh_rest: Vec<f32>,
    /// 0..=3.
    pub sh_degree: u32,
    pub bounds: Bounds,
    /// `hash` is [`SplatScene::asset_hash`]. `name` and `path` are the caller's to fill in:
    /// [`import_ply`] is handed bytes, not a file.
    pub asset: AssetRef,
    /// What the file said that this type cannot carry. Never a reason to reject a capture.
    pub warnings: Vec<String>,
}

impl SplatScene {
    /// Number of Gaussians.
    pub fn len(&self) -> usize {
        self.opacities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.opacities.is_empty()
    }

    /// Coefficients of degrees 1..=`sh_degree`, per channel. 0, 3, 8 or 15.
    pub fn sh_rest_per_channel(&self) -> usize {
        let d = self.sh_degree as usize;
        (d + 1) * (d + 1) - 1
    }

    /// Centre of Gaussian `i`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn position(&self, i: usize) -> Vec3 {
        Vec3::new(
            f64::from(self.positions[3 * i]),
            f64::from(self.positions[3 * i + 1]),
            f64::from(self.positions[3 * i + 2]),
        )
    }

    /// Base colour of Gaussian `i`: the degree-0 SH coefficient evaluated, `0.5 + C0 * dc`.
    ///
    /// # Panics
    /// If `i >= self.len()`.
    pub fn rgb(&self, i: usize) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = 0.5 + SH_C0 * self.sh_dc[3 * i + c];
        }
        out
    }

    /// Recomputes [`SplatScene::bounds`] from [`SplatScene::positions`].
    pub fn recompute_bounds(&mut self) {
        let mut b = Bounds::default();
        for p in self.positions.chunks_exact(3) {
            for (axis, v) in p.iter().enumerate() {
                let v = f64::from(*v);
                let (lo, hi) = match axis {
                    0 => (&mut b.min.x, &mut b.max.x),
                    1 => (&mut b.min.y, &mut b.max.y),
                    _ => (&mut b.min.z, &mut b.max.z),
                };
                *lo = lo.min(v);
                *hi = hi.max(v);
            }
        }
        self.bounds = b;
    }

    /// Maps every base colour through `map`, in RGB space, and writes it back as a degree-0
    /// coefficient. Degrees 1..=3 are left alone — see the design note, spec 16.2.
    pub fn apply_color(&mut self, map: &ColorAffine) {
        for i in 0..self.len() {
            let mapped = map.apply(self.rgb(i));
            for (c, m) in mapped.iter().enumerate() {
                self.sh_dc[3 * i + c] = align::rgb_to_dc(*m);
            }
        }
        self.refresh_asset_hash();
    }

    /// `blake3` over the *decoded* Gaussians in file order (spec 5.3), so the same capture
    /// read from an ascii and from a `binary_little_endian` PLY hashes identically.
    ///
    /// Excludes `warnings`, `bounds` (derived) and `asset` (which stores this digest).
    pub fn asset_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(SPLAT_TAG.as_bytes());
        h.update(&(self.len() as u64).to_le_bytes());
        h.update(&self.sh_degree.to_le_bytes());
        for array in [
            &self.positions,
            &self.scales,
            &self.rotations,
            &self.opacities,
            &self.sh_dc,
            &self.sh_rest,
        ] {
            h.update(&(array.len() as u64).to_le_bytes());
            for v in array {
                h.update(&v.to_le_bytes());
            }
        }
        *h.finalize().as_bytes()
    }

    /// Recomputes `asset.hash` after a mutation. Called by every mutator here.
    pub fn refresh_asset_hash(&mut self) {
        self.asset.hash = self.asset_hash();
    }

    /// A reference whose `hash` is the content digest. `AssetKind` has no `Splat` variant —
    /// it lives in `es-assets`, which this packet does not own — so the closest one is used
    /// and `path` carries the file.
    // ponytail: `AssetKind::Mesh` stands in for a splat; the packet that adds
    // `AssetKind::Splat` to `es-assets` should change this line and nothing else, since
    // nothing branches on the variant.
    fn asset_ref(hash: [u8; 32]) -> AssetRef {
        AssetRef {
            id: scene_id("asset", "splat/splat"),
            name: "splat".to_owned(),
            kind: AssetKind::Mesh,
            path: String::new(),
            hash,
        }
    }
}
