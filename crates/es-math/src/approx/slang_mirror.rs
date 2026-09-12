//! P09 oracle: the Slang mirror declares the same constants, in the same order, bound to the
//! same literal text as `coeffs.rs`.
//!
//! What this does *not* prove: that the GPU produces the same bits. That needs a Vulkan
//! device with the §3.4 execution modes and is M2 work — `Target / Status: unverified`.

/// `(name, literal)` for every `pub const NAME: f32 = <literal>;` line.
fn rust_constants(src: &str) -> Vec<(String, String)> {
    src.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("pub const ")?;
            let (name, rest) = rest.split_once(": f32 = ")?;
            Some((name.to_owned(), rest.strip_suffix(';')?.to_owned()))
        })
        .collect()
}

/// `(name, literal)` for every `static const float NAME = <literal>;` line.
fn slang_constants(src: &str) -> Vec<(String, String)> {
    src.lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("static const float ")?;
            let (name, rest) = rest.split_once(" = ")?;
            Some((name.to_owned(), rest.strip_suffix(';')?.to_owned()))
        })
        .collect()
}

#[test]
fn coefficients_match_byte_for_byte() {
    let rust = rust_constants(include_str!("coeffs.rs"));
    let slang = slang_constants(include_str!("../../slang/approx.slang"));

    assert!(
        rust.len() >= 30,
        "coeffs.rs parse produced only {} constants — parser or file format changed",
        rust.len()
    );
    assert_eq!(
        rust, slang,
        "Rust and Slang coefficients differ. Every constant must appear in both files with \
         the same name, the same literal text and in the same order (see \
         docs/design/transcendental.md)."
    );
}

#[test]
fn slang_mirror_keeps_the_horner_expressions() {
    let slang = include_str!("../../slang/approx.slang");
    // Spot-check the evaluation order of every polynomial: a reassociated mirror would
    // still pass the literal comparison above.
    for expr in [
        "((SIN_C0 * z + SIN_C1) * z + SIN_C2) * z * r + r",
        "((COS_C0 * z + COS_C1) * z + COS_C2) * z * z - 0.5 * z + 1.0",
        "(((((EXP_C0 * r + EXP_C1) * r + EXP_C2) * r + EXP_C3) * r + EXP_C4) * r + EXP_C5) * z",
        "((((((((LOG_C0 * m + LOG_C1) * m + LOG_C2) * m + LOG_C3) * m + LOG_C4) * m + LOG_C5)",
        "(((ATAN_C0 * z + ATAN_C1) * z + ATAN_C2) * z + ATAN_C3) * z * x + x",
        "((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3",
    ] {
        assert!(slang.contains(expr), "missing from approx.slang: {expr}");
    }
}
