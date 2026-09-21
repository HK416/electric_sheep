//! Counter-based RNG (spec 3.4: no global RNG).
//!
//! Same design as `es_env::rng`: a draw is *addressed*, not stepped, so it is a pure function
//! of `(seed, view, pixel, sample, bounce, stream, index)` and neither thread order nor
//! dispatch shape can change it.
//!
//! **32-bit, where `es_env::rng` uses splitmix64.** Slang's `uint64_t` needs `shaderInt64`,
//! which spec 3.3 does not guarantee on every target (`MoltenVK` in particular), and a 32-bit
//! mixer is bit-identical between Rust and Slang with no capability check at all. The mixer
//! is the Murmur3 `fmix32` finalizer. Statistical quality beyond "decorrelated enough for a
//! fixed-bounce diffuse path tracer" is `unverified`; nothing but rendering uses it.
//!
//! `crates/es-render/slang/rng.slang` is the line-for-line mirror. Edit both or neither.

/// Murmur3 `fmix32`.
pub const fn mix32(mut z: u32) -> u32 {
    z ^= z >> 16;
    z = z.wrapping_mul(0x85eb_ca6b);
    z ^= z >> 13;
    z = z.wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

/// Stream key for one path-tracer sample. `stream` separates independent uses at the same
/// coordinates; the table is pinned here and in `docs/design/renderer.md` section 10, and a
/// new use takes a **new** id rather than reusing one at a different `index`:
///
/// | id | use |
/// |---|---|
/// | 0 | the cosine-weighted BSDF bounce direction |
/// | 1 | `ReSTIR` initial candidates (light pick, area sample and the reservoir accept) |
/// | 2 | `ReSTIR` spatial reuse (the reservoir accept per neighbour) |
/// | 3 | `ReSTIR` temporal reuse (the reservoir accept) |
/// | 4 | NEE: which emissive triangle (packet M7/R3) |
/// | 5 | NEE: the uniform point on that triangle |
/// | 6 | NEE: the cosine-weighted sky direction |
///
/// The NEE shadow test has no stream: it draws no random number.
pub fn key(seed: u32, view: u32, px: u32, py: u32, sample: u32, bounce: u32, stream: u32) -> u32 {
    let mut k = mix32(seed ^ view);
    k = mix32(k ^ (px.wrapping_mul(73_856_093) ^ py.wrapping_mul(19_349_663)));
    k = mix32(k ^ sample);
    k = mix32(k ^ bounce);
    mix32(k ^ stream)
}

/// The `i`th draw of a stream, uniform in `[0, 1)`. Top 24 bits, so the value is exact in
/// `f32` and never reaches 1.0.
pub fn uniform(key: u32, i: u32) -> f32 {
    let bits = mix32(key ^ i.wrapping_mul(0x9e37_79b9));
    (bits >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exact equality is the property under test: a draw is a pure function of its
    // coordinates.
    #[allow(clippy::float_cmp)]
    #[test]
    fn draws_are_addressed_not_stepped() {
        let a = key(1, 0, 10, 20, 3, 1, 0);
        assert_eq!(uniform(a, 5), uniform(a, 5));
        assert_ne!(uniform(a, 5), uniform(a, 6));
    }

    #[test]
    fn every_coordinate_separates_the_stream() {
        let base = key(1, 0, 10, 20, 3, 1, 0);
        for other in [
            key(2, 0, 10, 20, 3, 1, 0),
            key(1, 1, 10, 20, 3, 1, 0),
            key(1, 0, 11, 20, 3, 1, 0),
            key(1, 0, 10, 21, 3, 1, 0),
            key(1, 0, 10, 20, 4, 1, 0),
            key(1, 0, 10, 20, 3, 2, 0),
            key(1, 0, 10, 20, 3, 1, 1),
        ] {
            assert_ne!(base, other);
        }
    }

    /// The pixel hash must not collapse `(x, y)` and `(y, x)` onto one stream.
    #[test]
    fn transposed_pixels_differ() {
        assert_ne!(key(0, 0, 3, 7, 0, 0, 0), key(0, 0, 7, 3, 0, 0, 0));
    }

    #[test]
    fn uniform_stays_in_range() {
        for i in 0..10_000 {
            let u = uniform(key(7, 0, i % 97, i / 97, 0, 0, 0), i);
            assert!((0.0..1.0).contains(&u), "{u}");
        }
    }
}
