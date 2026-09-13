# Transcendental functions (`es_math::approx`) — design

Spec refs: §3.2 (초월함수), §3.4 (결정적 실행 계약), Appendix D `DET-010`.

## Why a private implementation

SPIR-V `GLSL.std.450` only specifies an ULP bound; the actual bit pattern is vendor- and
driver-dependent. `libm` differs per platform for the same reason. Both break the tier-1
bitwise guarantee of §3.5, so physics / observation / reward kernels must call
`es_math::approx` instead of `f32::sin` & friends, and the Rust and Slang sides must share
coefficients *and* evaluation order.

## Contract

- **Input/output type is `f32`.** Kernels run FP32 (§3.3).
- **Evaluation order is fixed** — Horner, written out as a single expression per polynomial,
  in the same order in `src/approx/coeffs.rs` + `src/approx/mod.rs` and in
  `slang/approx.slang`. No reassociation, no `mul_add` (an FMA would change the result on
  hosts without one, and Slang/SPIR-V must not contract either — `NoContraction` is required
  on the GPU side, §3.4).
- **No `fast-math`, no table lookups, no branch on device properties.**
- **Target ≤ 2 ULP** over the primary domain of each function.

## Range reduction

| fn | primary domain | reduction |
|---|---|---|
| `sin`, `cos` | `|x| ≤ 1e3` | Cody–Waite: `n = floor(x·2/π + 0.5)` (`floor`, not `round` — HLSL/Slang `round` is round-half-to-even while Rust's is round-half-away, so the mirror must not use it), `r = ((x − n·DP1) − n·DP2) − n·DP3`, three `f32` constants carrying ≈ 48 bits of π/2. Quadrant `n & 3` selects the sin- or cos-kernel and the sign. |
| `tan` | `|x| ≤ 6.5`, off the poles | The **same** `reduce_quadrant` as `sin`/`cos` — one reduction, one place to get wrong. `tan` has period π, so only `quadrant & 1` matters: the even quadrants return the kernel, the odd ones `-1.0 / kernel`. The kernel is Cephes `tanf.c`'s own 6-term minimax in `r²`, **not** `sin(x)/cos(x)`: the quotient measures 3 ULP (two kernel errors plus the division), over target. The domain is narrower than `sin`/`cos` because `tan'(x) = 1 + tan²(x)` amplifies the reduction's residual argument error — see the table below. |
| `exp` | `[-88, 88]` (f32 range) | `n = floor(x·log2e + 0.5)`, `r = (x − n·LN2_HI) − n·LN2_LO`, polynomial on `r`, then scale by `2^n` built with `f32::from_bits` (exact). |
| `ln` | `(0, f32::MAX]` | `x = m·2^e` with `m ∈ [√½, √2)` via exponent-field extraction, polynomial in `m − 1`, then `+ e·ln2` split into `LN2_HI/LN2_LO`. |
| `atan2` | all finite `(y, x)` | `atan` on `|y/x|` or `|x/y|`, itself reduced to `[0, tan(π/8)]` by `t → (t−1)/(t+1)`, then quadrant fix-up. Each of `π`, `π/2`, `π/4` is carried as a `_HI`/`_LO` pair (nearest `f32` plus the `f32` residue of the true value); reconstructing as `(PI - a) + PI_LO` rather than `PI - a` is what takes `atan2` from 3 ULP to 2. |
| `sqrt`, `rsqrt` | `[0, ∞)` | None, and no polynomial: `sqrt` is a single correctly-rounded IEEE-754 operation, and SPIR-V's `OpSqrt`/`OpFDiv` are likewise required to be ≤ 0.5 ULP. So `approx::sqrt` is `x.sqrt()` and `approx::rsqrt` is `1.0 / x.sqrt()`. A vendor fast-rsqrt instruction is **not** used — its accuracy is unspecified. |

Outside the primary domain the functions still return a sensible value (IEEE specials are
handled explicitly) but the ULP bound is not claimed. For `sin`/`cos` beyond `|x| ≈ 1e3` the
three-part Cody–Waite reduction runs out of π bits; Payne–Hanek is deliberately **not**
implemented (no kernel in the spec needs it, and it would not be worth the GPU divergence).

## Coefficient provenance

The polynomial coefficients in `src/approx/coeffs.rs` are the single-precision minimax
coefficients from the **Cephes Math Library** (S. L. Moshier, `sinf.c`, `tanf.c`, `expf.c`,
`logf.c`, `atanf.c`), public domain. They were fitted with a Remez exchange against the reduced domains
listed above, which are exactly the reduced domains used here. They are reproduced verbatim
as decimal literals — no rounding, no re-derivation — so that the Rust and Slang files can be
compared literal-by-literal (P09). `clippy::unreadable_literal`, `clippy::excessive_precision`
and `clippy::approx_constant` are allowed in `coeffs.rs` for that reason: digit separators,
trimmed decimals or a swap to `std::f32::consts` would all break the byte-for-byte comparison
with the Slang mirror.

If a coefficient set is ever re-fitted, it must be re-fitted for **both** files in the same
change, and the ULP test below is the gate.

## Accuracy — measured

`cargo test -p es-math approx::` sweeps each function over its primary domain (a dense linear
sweep plus a deterministic pseudo-random sweep, ≥ 200k points each) and compares against the
`f64` `std` implementation rounded to `f32`, which is ≤ 0.5 ULP of the true value over these
domains and is therefore a valid stand-in for MPFR at `f32` precision. The test prints the
measured maximum ULP error per function and fails above 2 ULP.

Measured on x86-64 (Windows, rustc 1.85, debug profile), `cargo test -p es-math approx:: -- --nocapture`:

| fn | domain swept | max ULP (measured) |
|---|---|---|
| `sin` | `[-1e3, 1e3]` and `[-6.5, 6.5]` | 1 |
| `cos` | `[-1e3, 1e3]` and `[-6.5, 6.5]` | 1 |
| `tan` | `[-1.5, 1.5]` and `[-6.5, 6.5]` (primary) | 2 |
| `tan` | `[-1e3, 1e3]` (reported, **not** bounded) | 3 |
| `exp` | `[-88, 88]` | 1 |
| `ln` | `[1e-30, 1e30]` and `[0.5, 2]` | 1 |
| `atan2` | unit circle + random pairs over `[1e-6, 1e6]` | 2 |
| `sqrt` | `[0, 1e30]` | 0 |
| `rsqrt` | `[1e-30, 1e30]` | 1 |

`tan` is the one function whose primary domain is not the full `sin`/`cos` range. Its sweep
filters by magnitude rather than by domain — points where the `f64` reference exceeds `1e3`
are within ~1e-3 of a pole and are skipped — because the amplification factor of `tan` is
`tan'(x)/tan(x) = 1/tan + tan`, so a reduction error that is 0.1 ULP for `sin` is 3 ULP for
`tan` at `|x| ≈ 600`, and unbounded at a pole. The bound claimed is 2 ULP over `|x| ≤ 6.5`;
the wider sweep is printed for the record and asserts nothing. `es-render` uses `tan` only on
`fovy/2 ∈ (0, π/2)`, well inside the bound.

All at or under the 2 ULP target. The numbers are a property of the coefficients and the
evaluation order, not of the host, so they should hold anywhere the same `f32` operations
are correctly rounded.

MPFR (`rug`) as the reference oracle requires GMP and is not buildable on the Windows CI
host; the `f64`-std reference is used instead. Replacing it with a pure-Rust arbitrary
precision crate is **Target / Status: unverified**.

## Rust ↔ Slang bit equality

`slang/approx.slang` mirrors the Rust source line for line. A unit test parses the constant
declarations out of both files and asserts the name/literal pairs are identical byte-for-byte.
That the two then produce **bit-identical results** on a real GPU is
**Target / Status: unverified** — it needs a Vulkan device with the §3.4 execution modes set,
which is M2 work (P09 acceptance only covers the literal comparison).
