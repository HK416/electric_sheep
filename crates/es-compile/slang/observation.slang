// Observation IR GPU kernels (spec 7.6, spec 11.4) — the mirror of
// `crates/es-compile/src/kernels.rs`.
//
// One file, one entry point per kernel id of `KERNEL_IDS`. Every entry is compiled on its
// own with `-entry`, specialised entirely by `-D` defines (dtype, channels, sizes, buffer
// offsets), which is what makes a pipeline a pure function of `(kernel id, defines)`.
//
// Determinism (spec 3.4): no atomics, no shared memory, no subgroup ops, no dependence on
// the workgroup count. Index arithmetic and operator association are copied from the CPU
// kernels line by line, because the CPU kernels are the oracle and the claim is bit
// equality. `NoContraction` is patched onto every float op by `es_gpu::apply_exec_modes`;
// without it a `a * b - c` here may become an fma and the last bits move.
//
// The one transcendental (the sRGB 2.4 power) goes through `approx.slang`, the mirror of
// `es_math::approx`, so the coefficients and the evaluation order are shared.

#include "approx.slang"

// --- bindings -----------------------------------------------------------------------------
//
// Fixed for every kernel so one descriptor layout serves all of them. Unused bindings are
// still bound; a bound-but-unread descriptor is legal and costs nothing.
//
//  0 arena  f32 intermediates, then the f32 plan inputs
//  1 words  u8 plan inputs packed little-endian, then f16/bf16 output bit patterns
//  2 aux    the 256-entry sRGB LUT, then per-`Normalize` mean/std
//  3 rings  the `TemporalWindow` history rings (device-resident state)
//  4 state  per-window (cursor, pushed), uploaded once per run

[[vk::binding(0, 0)]] RWStructuredBuffer<float> arena;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> words;
[[vk::binding(2, 0)]] StructuredBuffer<float> aux;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> rings;
[[vk::binding(4, 0)]] StructuredBuffer<uint> state;

// --- defines ------------------------------------------------------------------------------
//
// Every entry in this file is preprocessed for every compile, so each define needs a default
// or an unrelated entry fails to parse. 1, never 0: a default of 0 would make a constant-
// folded `% ES_DEPTH` a compile error in an entry that is not even being compiled.

#ifndef ES_N
#define ES_N 1
#endif
#ifndef ES_DST
#define ES_DST 1
#endif
#ifndef ES_SRC
#define ES_SRC 1
#endif
#ifndef ES_H
#define ES_H 1
#endif
#ifndef ES_W
#define ES_W 1
#endif
#ifndef ES_C
#define ES_C 1
#endif
#ifndef ES_SW
#define ES_SW 1
#endif
#ifndef ES_SH
#define ES_SH 1
#endif
#ifndef ES_DW
#define ES_DW 1
#endif
#ifndef ES_DH
#define ES_DH 1
#endif
#ifndef ES_RX
#define ES_RX 1
#endif
#ifndef ES_RY
#define ES_RY 1
#endif
#ifndef ES_RW
#define ES_RW 1
#endif
#ifndef ES_RH
#define ES_RH 1
#endif
#ifndef ES_PLANE
#define ES_PLANE 1
#endif
#ifndef ES_AUX
#define ES_AUX 1
#endif
#ifndef ES_LO_BITS
#define ES_LO_BITS 1
#endif
#ifndef ES_HI_BITS
#define ES_HI_BITS 1
#endif
#ifndef ES_CHUNK
#define ES_CHUNK 1
#endif
#ifndef ES_STRIDE_OUT
#define ES_STRIDE_OUT 1
#endif
#ifndef ES_LEAD
#define ES_LEAD 1
#endif
#ifndef ES_SLOT
#define ES_SLOT 1
#endif
#ifndef ES_DEPTH
#define ES_DEPTH 1
#endif
#ifndef ES_RING
#define ES_RING 1
#endif
#ifndef ES_STATE
#define ES_STATE 1
#endif
#ifndef ES_STEPS
#define ES_STEPS 1
#endif
#ifndef ES_STEP_STRIDE
#define ES_STEP_STRIDE 1
#endif
#ifndef ES_NARROW
#define ES_NARROW 1
#endif

// --- helpers ------------------------------------------------------------------------------

/// Byte `at` of the `words` region. Little-endian, as spec 2.1's platforms all are.
uint es_byte(uint at) {
    return (words[at >> 2] >> ((at & 3u) * 8u)) & 0xffu;
}

/// `kernels::src_index`: half-pixel centres, `align_corners = false`, negative edge clamped.
float es_src_index(float scale, uint d) {
    float s = scale * ((float)d + 0.5) - 0.5;
    return s < 0.0 ? 0.0 : s;
}

/// `kernels::srgb_eotf`, same constants and same evaluation order.
float es_srgb_eotf(float x) {
    if (x <= 0.040449936) {
        return x / 12.92;
    }
    float t = (x + 0.055) / 1.055;
    return es_exp(2.4 * es_ln(t));
}

// --- casts --------------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void cast_u8_hwc_to_f32_chw(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint plane = (uint)(ES_H * ES_W);
    uint ch = i / plane;
    uint rem = i - ch * plane;
    uint y = rem / (uint)ES_W;
    uint x = rem - y * (uint)ES_W;
    arena[ES_DST + i] = (float)es_byte((uint)ES_SRC + (y * (uint)ES_W + x) * (uint)ES_C + ch)
        / 255.0;
}

/// f32 -> f16, round-to-nearest-even: the mirror of `half::f16::from_f32`.
///
/// The f32 copy into the arena is what `CpuPlan` does too (`Op::Cast` keeps the wide value
/// so spec 11.5's per-node view still works); the narrowed bits go to `words` because this
/// device is not opened with 16-bit storage.
uint es_f32_to_f16_bits(float value) {
    uint x = asuint(value);
    uint sign = x & 0x80000000u;
    uint exp = x & 0x7f800000u;
    uint man = x & 0x007fffffu;

    if (exp == 0x7f800000u) {
        uint nan_bit = (man == 0u) ? 0u : 0x0200u;
        return (sign >> 16) | 0x7c00u | nan_bit | (man >> 13);
    }

    uint half_sign = sign >> 16;
    int half_exp = (int)(exp >> 23) - 127 + 15;

    if (half_exp >= 0x1f) {
        return half_sign | 0x7c00u;
    }
    if (half_exp <= 0) {
        if (14 - half_exp > 24) {
            return half_sign;
        }
        uint hidden = man | 0x00800000u;
        uint half_man = hidden >> (uint)(14 - half_exp);
        uint round_bit = 1u << (uint)(13 - half_exp);
        if ((hidden & round_bit) != 0u && (hidden & (3u * round_bit - 1u)) != 0u) {
            half_man = half_man + 1u;
        }
        return (half_sign | half_man) & 0xffffu;
    }

    uint biased = ((uint)half_exp) << 10;
    uint half_man = man >> 13;
    uint round_bit = 0x00001000u;
    uint out_bits = half_sign | biased | half_man;
    if ((man & round_bit) != 0u && (man & (3u * round_bit - 1u)) != 0u) {
        out_bits = out_bits + 1u;
    }
    return out_bits & 0xffffu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cast_f32_to_f16(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    float v = arena[ES_SRC + i];
    arena[ES_DST + i] = v;
    words[(uint)ES_NARROW + i] = es_f32_to_f16_bits(v);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cast_f32_to_bf16(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    float v = arena[ES_SRC + i];
    arena[ES_DST + i] = v;

    uint x = asuint(v);
    uint bits;
    if ((x & 0x7fffffffu) > 0x7f800000u) {
        bits = (x >> 16) | 0x0040u;
    } else {
        uint round_bit = 0x00008000u;
        if ((x & round_bit) != 0u && (x & (3u * round_bit - 1u)) != 0u) {
            bits = (x >> 16) + 1u;
        } else {
            bits = x >> 16;
        }
    }
    words[(uint)ES_NARROW + i] = bits & 0xffffu;
}

// --- geometry -----------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void resize_bilinear(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint dplane = (uint)(ES_DH * ES_DW);
    uint ch = i / dplane;
    uint rem = i - ch * dplane;
    uint y = rem / (uint)ES_DW;
    uint x = rem - y * (uint)ES_DW;

    float scale_x = (float)ES_SW / (float)ES_DW;
    float scale_y = (float)ES_SH / (float)ES_DH;

    float sy = es_src_index(scale_y, y);
    uint y0 = (uint)sy;
    uint y1 = min(y0 + 1u, (uint)(ES_SH - 1));
    float ly1 = sy - (float)y0;
    float ly0 = 1.0 - ly1;

    float sx = es_src_index(scale_x, x);
    uint x0 = (uint)sx;
    uint x1 = min(x0 + 1u, (uint)(ES_SW - 1));
    float lx1 = sx - (float)x0;
    float lx0 = 1.0 - lx1;

    uint plane = (uint)ES_SRC + ch * (uint)(ES_SH * ES_SW);
    // PyTorch's association, as in the CPU kernel. Reassociating changes the last bits.
    arena[ES_DST + i] =
        ly0 * (lx0 * arena[plane + y0 * (uint)ES_SW + x0] + lx1 * arena[plane + y0 * (uint)ES_SW + x1])
        + ly1 * (lx0 * arena[plane + y1 * (uint)ES_SW + x0] + lx1 * arena[plane + y1 * (uint)ES_SW + x1]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resize_nearest(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint dplane = (uint)(ES_DH * ES_DW);
    uint ch = i / dplane;
    uint rem = i - ch * dplane;
    uint y = rem / (uint)ES_DW;
    uint x = rem - y * (uint)ES_DW;

    float scale_x = (float)ES_SW / (float)ES_DW;
    float scale_y = (float)ES_SH / (float)ES_DH;
    uint sy = min((uint)(scale_y * (float)y), (uint)(ES_SH - 1));
    uint sx = min((uint)(scale_x * (float)x), (uint)(ES_SW - 1));

    arena[ES_DST + i] = arena[(uint)ES_SRC + ch * (uint)(ES_SH * ES_SW) + sy * (uint)ES_SW + sx];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void crop(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint dplane = (uint)(ES_RH * ES_RW);
    uint ch = i / dplane;
    uint rem = i - ch * dplane;
    uint y = rem / (uint)ES_RW;
    uint x = rem - y * (uint)ES_RW;

    arena[ES_DST + i] = arena[(uint)ES_SRC + ch * (uint)(ES_SH * ES_SW)
        + (y + (uint)ES_RY) * (uint)ES_SW + x + (uint)ES_RX];
}

// --- colour -------------------------------------------------------------------------------

/// `ColorTransform{SRgb -> Linear}` reading a u8 sensor buffer: the LUT is uploaded from
/// `kernels::srgb_to_linear_lut()`, so the table bytes are the CPU's own, not a re-derivation.
[shader("compute")]
[numthreads(64, 1, 1)]
void srgb_to_linear_u8(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint plane = (uint)(ES_H * ES_W);
    uint ch = i / plane;
    uint rem = i - ch * plane;
    uint y = rem / (uint)ES_W;
    uint x = rem - y * (uint)ES_W;
    arena[ES_DST + i] =
        aux[(uint)ES_AUX + es_byte((uint)ES_SRC + (y * (uint)ES_W + x) * (uint)ES_C + ch)];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void srgb_to_linear(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    arena[ES_DST + i] = es_srgb_eotf(arena[ES_SRC + i]);
}

// --- photometric --------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void normalize_mean_std(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint ch = i / (uint)ES_PLANE;
    // Division, not a reciprocal multiply: PyTorch divides and the two differ in the last bit.
    arena[ES_DST + i] =
        (arena[ES_SRC + i] - aux[(uint)ES_AUX + ch]) / aux[(uint)ES_AUX + (uint)ES_C + ch];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void normalize_range(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    // Bit patterns, not decimal literals: a define round-tripped through text would be a
    // second rounding the CPU side never does.
    float lo = asfloat((uint)ES_LO_BITS);
    float hi = asfloat((uint)ES_HI_BITS);
    float span = hi - lo;
    arena[ES_DST + i] = (arena[ES_SRC + i] - lo) / span;
}

// --- structural ---------------------------------------------------------------------------

/// One input of a `Concat`: `ES_OUTER * ES_CHUNK` elements copied into the destination's
/// stride-`ES_STRIDE_OUT` slots starting at `ES_LEAD`. Pure copy, so the join is the same
/// bytes the CPU wrote whatever the dispatch order is.
///
/// `_copy`, not the bare kernel id: `concat` and `stack` are taken by the Slang core module,
/// and `slangc -entry concat` then finds no function. The kernel *id* is still `concat.v1`.
[shader("compute")]
[numthreads(64, 1, 1)]
void concat_copy(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint o = i / (uint)ES_CHUNK;
    uint j = i - o * (uint)ES_CHUNK;
    arena[(uint)ES_DST + o * (uint)ES_STRIDE_OUT + (uint)ES_LEAD + j] = arena[ES_SRC + i];
}

/// `Stack`: the same copy with one chunk per input. A separate entry because
/// `KERNEL_IDS` gives it its own id, and an id is what `compiler_hash` covers.
[shader("compute")]
[numthreads(64, 1, 1)]
void stack_copy(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint o = i / (uint)ES_CHUNK;
    uint j = i - o * (uint)ES_CHUNK;
    arena[(uint)ES_DST + o * (uint)ES_STRIDE_OUT + (uint)ES_LEAD + j] = arena[ES_SRC + i];
}

// --- temporal (spec 7.5) ------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void history_push(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint cursor = state[(uint)ES_STATE];
    uint at = (cursor % (uint)ES_DEPTH) * (uint)ES_SLOT;
    rings[(uint)ES_RING + at + i] = arena[ES_SRC + i];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void window_gather(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= (uint)ES_N) { return; }
    uint k = i / (uint)ES_SLOT;
    uint j = i - k * (uint)ES_SLOT;

    int cursor = (int)state[(uint)ES_STATE];
    int pushed = (int)state[(uint)ES_STATE + 1u];
    int avail = min(pushed, ES_DEPTH);
    // `cursor.saturating_sub(pushed.min(depth) - 1)`, then `.max(oldest)` — `oldest` is
    // already >= 0, so one `max` covers the saturating subtraction of `back` as well.
    int oldest = max(cursor - (avail - 1), 0);
    int back = (int)((uint)(ES_STEPS - 1) - k) * ES_STEP_STRIDE;
    int at = max(cursor - back, oldest);

    arena[ES_DST + i] = rings[(uint)ES_RING + ((uint)at % (uint)ES_DEPTH) * (uint)ES_SLOT + j];
}
