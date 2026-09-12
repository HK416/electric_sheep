// Deterministic transcendentals — GPU mirror of `crates/es-math/src/approx/` (spec 3.2).
//
// This file is a line-for-line mirror of `src/approx/mod.rs` and `src/approx/coeffs.rs`:
// the same range reduction, the same Horner expressions in the same order, and the same
// constant names bound to the same decimal literals. The test in
// `src/approx/slang_mirror.rs` fails if a literal drifts on one side only.
//
// The compile pipeline must decorate these expressions with `NoContraction` and set the
// SPIR-V execution modes of spec 3.4 step 3; without that, bit equality with the CPU path
// does not hold.
//
// Do not edit one side of a coefficient. See docs/design/transcendental.md.

// --- sin / cos ---------------------------------------------------------------------------
static const float TWO_OVER_PI = 0.63661977236758134308;
static const float PIO2_1 = 1.5703125;
static const float PIO2_2 = 4.837512969970703125e-4;
static const float PIO2_3 = 7.54978995489188216e-8;

static const float SIN_C0 = -1.9515295891e-4;
static const float SIN_C1 = 8.3321608736e-3;
static const float SIN_C2 = -1.6666654611e-1;

static const float COS_C0 = 2.443315711809948e-5;
static const float COS_C1 = -1.388731625493765e-3;
static const float COS_C2 = 4.166664568298827e-2;

// --- exp ---------------------------------------------------------------------------------
static const float LOG2E = 1.44269504088896341;
static const float LN2_HI = 0.693359375;
static const float LN2_LO = -2.12194440e-4;

static const float EXP_C0 = 1.9875691500e-4;
static const float EXP_C1 = 1.3981999507e-3;
static const float EXP_C2 = 8.3334519073e-3;
static const float EXP_C3 = 4.1665795894e-2;
static const float EXP_C4 = 1.6666665459e-1;
static const float EXP_C5 = 5.0000001201e-1;

// --- ln ----------------------------------------------------------------------------------
static const float SQRT_HALF = 0.707106781186547524;

static const float LOG_C0 = 7.0376836292e-2;
static const float LOG_C1 = -1.1514610310e-1;
static const float LOG_C2 = 1.1676998740e-1;
static const float LOG_C3 = -1.2420140846e-1;
static const float LOG_C4 = 1.4249322787e-1;
static const float LOG_C5 = -1.6668057665e-1;
static const float LOG_C6 = 2.0000714765e-1;
static const float LOG_C7 = -2.4999993993e-1;
static const float LOG_C8 = 3.3333331174e-1;

// --- atan / atan2 ------------------------------------------------------------------------
static const float TAN_PIO8 = 0.4142135623730950;

static const float ATAN_C0 = 8.05374449538e-2;
static const float ATAN_C1 = -1.38776856032e-1;
static const float ATAN_C2 = 1.99777106478e-1;
static const float ATAN_C3 = -3.33329491539e-1;

// --- shared constants --------------------------------------------------------------------
static const float PI = 3.14159265358979323846;
static const float PI_LO = -8.742277657e-8;
static const float PIO2 = 1.57079632679489661923;
static const float PIO2_LO = -4.371138828e-8;
static const float PIO4 = 0.78539816339744830962;
static const float PIO4_LO = -2.185569414e-8;

// --- implementation ----------------------------------------------------------------------

float es_pow2i(int k) {
    return asfloat((uint)((k + 127) << 23));
}

float es_scale2(float y, int n) {
    n = clamp(n, -250, 250);
    int half = n / 2;
    return y * es_pow2i(half) * es_pow2i(n - half);
}

void es_frexp(float x, out float m, out int e) {
    int adj = 0;
    if (x < 1.17549435e-38) {
        x = x * 16777216.0;
        adj = -24;
    }
    uint bits = asuint(x);
    e = (int)((bits >> 23) & 0xff) - 126 + adj;
    m = asfloat((bits & 0x807fffff) | 0x3f000000);
}

bool es_is_neg(float x) {
    return (asuint(x) >> 31) == 1;
}

void es_reduce_quadrant(float x, out float r, out int quadrant) {
    float n = floor(x * TWO_OVER_PI + 0.5);
    r = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;
    quadrant = ((int)n) & 3;
}

float es_sin_kernel(float r) {
    float z = r * r;
    return ((SIN_C0 * z + SIN_C1) * z + SIN_C2) * z * r + r;
}

float es_cos_kernel(float r) {
    float z = r * r;
    return ((COS_C0 * z + COS_C1) * z + COS_C2) * z * z - 0.5 * z + 1.0;
}

float es_sin(float x) {
    float r;
    int quadrant;
    es_reduce_quadrant(x, r, quadrant);
    if (quadrant == 0) { return es_sin_kernel(r); }
    if (quadrant == 1) { return es_cos_kernel(r); }
    if (quadrant == 2) { return -es_sin_kernel(r); }
    return -es_cos_kernel(r);
}

float es_cos(float x) {
    float r;
    int quadrant;
    es_reduce_quadrant(x, r, quadrant);
    if (quadrant == 0) { return es_cos_kernel(r); }
    if (quadrant == 1) { return -es_sin_kernel(r); }
    if (quadrant == 2) { return -es_cos_kernel(r); }
    return es_sin_kernel(r);
}

float es_exp(float x) {
    if (x > 88.72284) { return 1.0 / 0.0; }
    if (x < -103.97208) { return 0.0; }
    float n = floor(LOG2E * x + 0.5);
    float r = x - n * LN2_HI - n * LN2_LO;
    float z = r * r;
    float y = (((((EXP_C0 * r + EXP_C1) * r + EXP_C2) * r + EXP_C3) * r + EXP_C4) * r + EXP_C5) * z
        + r
        + 1.0;
    return es_scale2(y, (int)n);
}

float es_ln(float x) {
    if (x <= 0.0) { return -1.0 / 0.0; }
    float m;
    int e;
    es_frexp(x, m, e);
    if (m < SQRT_HALF) {
        m = m + m - 1.0;
        e = e - 1;
    } else {
        m = m - 1.0;
    }
    float z = m * m;
    float y = ((((((((LOG_C0 * m + LOG_C1) * m + LOG_C2) * m + LOG_C3) * m + LOG_C4) * m + LOG_C5)
        * m
        + LOG_C6)
        * m
        + LOG_C7)
        * m
        + LOG_C8)
        * m
        * z;
    float fe = (float)e;
    y = y + LN2_LO * fe - 0.5 * z;
    return m + y + LN2_HI * fe;
}

float es_atan_unit(float t) {
    if (t > TAN_PIO8) {
        float x = (t - 1.0) / (t + 1.0);
        float z = x * x;
        float p = (((ATAN_C0 * z + ATAN_C1) * z + ATAN_C2) * z + ATAN_C3) * z * x + x;
        return PIO4 + (p + PIO4_LO);
    } else {
        float z = t * t;
        return (((ATAN_C0 * z + ATAN_C1) * z + ATAN_C2) * z + ATAN_C3) * z * t + t;
    }
}

float es_atan2(float y, float x) {
    float ax = abs(x);
    float ay = abs(y);
    float a;
    if (ax == 0.0 && ay == 0.0) {
        a = 0.0;
    } else if (isinf(ax) && isinf(ay)) {
        a = PIO4;
    } else if (ay <= ax) {
        a = es_atan_unit(ay / ax);
    } else {
        a = (PIO2 - es_atan_unit(ax / ay)) + PIO2_LO;
    }
    if (es_is_neg(x)) { a = (PI - a) + PI_LO; }
    if (es_is_neg(y)) { a = -a; }
    return a;
}

float es_sqrt(float x) {
    return sqrt(x);
}

float es_rsqrt(float x) {
    return 1.0 / sqrt(x);
}
