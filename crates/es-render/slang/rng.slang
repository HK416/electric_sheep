// Counter-based RNG — the line-for-line mirror of `crates/es-render/src/rng.rs` (spec 3.4:
// no global RNG; a draw is addressed, not stepped).
//
// 32-bit, where `es_env::rng` uses splitmix64: Slang's `uint64_t` needs `shaderInt64`, which
// spec 3.3 does not guarantee everywhere, and a 32-bit mixer is bit-identical between Rust
// and Slang with no capability check. The mixer is Murmur3's `fmix32`.
//
// Edit both sides or neither.

#ifndef ES_RENDER_RNG_SLANG
#define ES_RENDER_RNG_SLANG

uint es_mix32(uint z) {
    z ^= z >> 16;
    z *= 0x85ebca6bu;
    z ^= z >> 13;
    z *= 0xc2b2ae35u;
    return z ^ (z >> 16);
}

// stream: 0 = bounce direction, 1 = light selection, 2 = spatial reuse.
uint es_rng_key(uint seed, uint view, uint px, uint py, uint sample, uint bounce, uint stream) {
    uint k = es_mix32(seed ^ view);
    k = es_mix32(k ^ (px * 73856093u ^ py * 19349663u));
    k = es_mix32(k ^ sample);
    k = es_mix32(k ^ bounce);
    return es_mix32(k ^ stream);
}

// The i-th draw of a stream, uniform in [0, 1). Top 24 bits: exact in f32, never 1.0.
float es_rng_uniform(uint key, uint i) {
    uint bits = es_mix32(key ^ (i * 0x9e3779b9u));
    return float(bits >> 8) * (1.0 / 16777216.0);
}

#endif
