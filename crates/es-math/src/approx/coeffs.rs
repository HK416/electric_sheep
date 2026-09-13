//! Minimax polynomial coefficients and range-reduction constants for [`crate::approx`].
//!
//! # Provenance
//!
//! Single-precision minimax coefficients from the **Cephes Math Library** (S. L. Moshier),
//! files `sinf.c`, `expf.c`, `logf.c`, `atanf.c`; public domain. They were fitted by Remez
//! exchange against exactly the reduced domains used in `approx/mod.rs`, and are reproduced
//! here verbatim as decimal literals — no re-derivation, no reformatting.
//!
//! Every constant declared here has a `static const float NAME = <literal>;` twin in
//! `slang/approx.slang` with the **same name and the same literal text**; the test in
//! `approx/slang_mirror.rs` fails if the two drift apart (spec §3.2). That is why digit
//! separators are not used and `clippy::unreadable_literal` is allowed in this file only.
//!
//! Re-fitting a coefficient set means changing both files in the same commit, with
//! `cargo test -p es-math approx::` as the gate.
// Verbatim Cephes literals: digit separators would break the Slang comparison, the extra
// decimals are the published values, and several constants are near a `std::consts` value
// but must stay as written.
#![allow(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::approx_constant
)]

// --- sin / cos ---------------------------------------------------------------------------
// Cody-Waite split of pi/2 into three f32 terms (~48 bits). Cephes `sinf.c` DP1..DP3, x2.
pub const TWO_OVER_PI: f32 = 0.63661977236758134308;
pub const PIO2_1: f32 = 1.5703125;
pub const PIO2_2: f32 = 4.837512969970703125e-4;
pub const PIO2_3: f32 = 7.54978995489188216e-8;

// sin(r)/r - 1 on |r| <= pi/4, in r^2.
pub const SIN_C0: f32 = -1.9515295891e-4;
pub const SIN_C1: f32 = 8.3321608736e-3;
pub const SIN_C2: f32 = -1.6666654611e-1;

// cos(r) - 1 + r^2/2 on |r| <= pi/4, in r^2.
pub const COS_C0: f32 = 2.443315711809948e-5;
pub const COS_C1: f32 = -1.388731625493765e-3;
pub const COS_C2: f32 = 4.166664568298827e-2;

// tan(r)/r - 1 on |r| <= pi/4, in r^2. Cephes `tanf.c`; same reduction as sin/cos above.
pub const TAN_C0: f32 = 9.38540185543E-3;
pub const TAN_C1: f32 = 3.11992232697E-3;
pub const TAN_C2: f32 = 2.44301354525E-2;
pub const TAN_C3: f32 = 5.34112807005E-2;
pub const TAN_C4: f32 = 1.33387994085E-1;
pub const TAN_C5: f32 = 3.33331568548E-1;

// --- exp ---------------------------------------------------------------------------------
pub const LOG2E: f32 = 1.44269504088896341;
// ln(2) split so that n * LN2_HI is exact for every n in the f32 exponent range.
pub const LN2_HI: f32 = 0.693359375;
pub const LN2_LO: f32 = -2.12194440e-4;

pub const EXP_C0: f32 = 1.9875691500e-4;
pub const EXP_C1: f32 = 1.3981999507e-3;
pub const EXP_C2: f32 = 8.3334519073e-3;
pub const EXP_C3: f32 = 4.1665795894e-2;
pub const EXP_C4: f32 = 1.6666665459e-1;
pub const EXP_C5: f32 = 5.0000001201e-1;

// --- ln ----------------------------------------------------------------------------------
pub const SQRT_HALF: f32 = 0.707106781186547524;

pub const LOG_C0: f32 = 7.0376836292e-2;
pub const LOG_C1: f32 = -1.1514610310e-1;
pub const LOG_C2: f32 = 1.1676998740e-1;
pub const LOG_C3: f32 = -1.2420140846e-1;
pub const LOG_C4: f32 = 1.4249322787e-1;
pub const LOG_C5: f32 = -1.6668057665e-1;
pub const LOG_C6: f32 = 2.0000714765e-1;
pub const LOG_C7: f32 = -2.4999993993e-1;
pub const LOG_C8: f32 = 3.3333331174e-1;

// --- atan / atan2 ------------------------------------------------------------------------
pub const TAN_PIO8: f32 = 0.4142135623730950;

pub const ATAN_C0: f32 = 8.05374449538e-2;
pub const ATAN_C1: f32 = -1.38776856032e-1;
pub const ATAN_C2: f32 = 1.99777106478e-1;
pub const ATAN_C3: f32 = -3.33329491539e-1;

// --- shared constants --------------------------------------------------------------------
// Each is split into the nearest f32 and the f32 residue of the true value, so the quadrant
// reconstruction in `atan2` does not inherit the 0.37 ULP error of the rounded constant.
pub const PI: f32 = 3.14159265358979323846;
pub const PI_LO: f32 = -8.742277657e-8;
pub const PIO2: f32 = 1.57079632679489661923;
pub const PIO2_LO: f32 = -4.371138828e-8;
pub const PIO4: f32 = 0.78539816339744830962;
pub const PIO4_LO: f32 = -2.185569414e-8;
