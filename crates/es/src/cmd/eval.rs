//! `es eval compare` (spec 10.5).

use es_ir::evaluation::{EvaluationReport, MetricValue};

use crate::error::CliError;

const HELP: &str = "\
es eval compare <A.json> <B.json>

Prints a per-suite/per-metric table comparing two EvaluationReport JSON files (spec
10.5): A's value, B's value, the delta B-A, and a significance column.

EvaluationReport (spec 10.5) carries only per-cell aggregates (MetricValue::Scalar or
::Histogram), never raw per-episode samples, so significance prints `n/a` unless a report
also carries a non-standard `samples: [f64, ...]` array alongside a cell -- in which case
a two-sided Welch t-test p-value is computed (plain Rust, no stats crate) and flagged
when |p| < 0.05.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("compare") => compare(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es eval: unknown subcommand '{other}'"
        ))),
    }
}

fn load(path: &str) -> Result<(EvaluationReport, serde_json::Value), CliError> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    let report: EvaluationReport = serde_json::from_str(&raw)
        .map_err(|e| CliError::Runtime(format!("{path}: not an EvaluationReport: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| CliError::Runtime(format!("{path}: {e}")))?;
    Ok((report, value))
}

/// The non-standard `cells[i].samples` extension a report may carry; the typed
/// [`EvaluationReport`] has no such field (spec 10.5 only defines aggregates).
fn samples_of(raw: &serde_json::Value, idx: usize) -> Option<Vec<f64>> {
    let arr = raw.get("cells")?.get(idx)?.get("samples")?.as_array()?;
    Some(arr.iter().filter_map(serde_json::Value::as_f64).collect())
}

fn scalar(v: &MetricValue) -> Option<f64> {
    match v {
        MetricValue::Scalar(x) => Some(*x),
        MetricValue::Histogram(_) | MetricValue::Unavailable { .. } => None,
    }
}

fn value_repr(v: &MetricValue) -> String {
    match v {
        MetricValue::Scalar(x) => format!("{x:.6}"),
        MetricValue::Histogram(h) => format!("{h:?}"),
        MetricValue::Unavailable { reason } => format!("unavailable ({reason})"),
    }
}

fn print_row(suite: &str, metric: &str, a: &str, b: &str, delta: &str, sig: &str) {
    println!("{suite:<20} {metric:<26} {a:>14} {b:>14} {delta:>14}  {sig}");
}

fn compare(args: &[String]) -> Result<u8, CliError> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return Ok(0);
    }
    let [a_path, b_path] = args else {
        return Err(CliError::Usage(HELP.to_owned()));
    };
    let (a, a_raw) = load(a_path)?;
    let (b, b_raw) = load(b_path)?;

    print_row("SUITE", "METRIC", "A", "B", "DELTA", "SIGNIFICANT");
    for (ia, ca) in a.cells.iter().enumerate() {
        let Some((ib, cb)) = b
            .cells
            .iter()
            .enumerate()
            .find(|(_, cb)| cb.suite == ca.suite && cb.metric == ca.metric)
        else {
            print_row(
                &ca.suite,
                ca.metric.name(),
                &value_repr(&ca.value),
                "-",
                "n/a",
                "n/a",
            );
            continue;
        };
        match (scalar(&ca.value), scalar(&cb.value)) {
            (Some(av), Some(bv)) => {
                let sig = match (samples_of(&a_raw, ia), samples_of(&b_raw, ib)) {
                    (Some(sa), Some(sb)) if sa.len() > 1 && sb.len() > 1 => {
                        let p = welch_t_test(&sa, &sb);
                        format!("p={p:.4}{}", if p < 0.05 { " *" } else { "" })
                    }
                    _ => "n/a (aggregate-only report)".to_owned(),
                };
                print_row(
                    &ca.suite,
                    ca.metric.name(),
                    &format!("{av:.6}"),
                    &format!("{bv:.6}"),
                    &format!("{:+.6}", bv - av),
                    &sig,
                );
            }
            _ => print_row(
                &ca.suite,
                ca.metric.name(),
                &value_repr(&ca.value),
                &value_repr(&cb.value),
                "n/a",
                "n/a",
            ),
        }
    }
    for cb in &b.cells {
        if !a
            .cells
            .iter()
            .any(|ca| ca.suite == cb.suite && ca.metric == cb.metric)
        {
            print_row(
                &cb.suite,
                cb.metric.name(),
                "-",
                &value_repr(&cb.value),
                "n/a",
                "n/a",
            );
        }
    }
    println!();
    println!("A: passed={}   B: passed={}", a.passed, b.passed);
    Ok(0)
}

// --- Welch's t-test, in plain Rust (no stats crate) -----------------------------------------

/// Two-sided Welch t-test p-value for two independent samples of unequal variance.
fn welch_t_test(a: &[f64], b: &[f64]) -> f64 {
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let var = |xs: &[f64], m: f64| {
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0)
    };
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let se2 = va / na + vb / nb;
    if se2 <= 0.0 {
        return if (ma - mb).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    let t = (ma - mb) / se2.sqrt();
    // Welch-Satterthwaite degrees of freedom.
    let df = se2 * se2 / ((va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0));
    (student_t_sf(t.abs(), df) * 2.0).min(1.0)
}

/// Student-t survival function `P(T > t)` for `t >= 0`, via the regularized incomplete beta
/// function: `sf(t, df) = 0.5 * I_x(df/2, 1/2)` with `x = df / (df + t^2)`.
fn student_t_sf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * reg_incomplete_beta(x, df / 2.0, 0.5)
}

/// The regularized incomplete beta function `I_x(a, b)`, via its continued fraction
/// (Numerical Recipes 6.4), with the standard symmetry swap for faster convergence.
fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(x, a, b) / a
    } else {
        1.0 - front * betacf(1.0 - x, b, a) / b
    }
}

/// Lentz's continued fraction for the incomplete beta function. Parameter names follow
/// Numerical Recipes' own notation, so the short names stay.
#[allow(clippy::many_single_char_names)]
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAXIT: u32 = 200;
    const EPS: f64 = 3e-14;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let mf = f64::from(m);
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Lanczos approximation of `ln(gamma(x))`, g=7, n=9 -- accurate to ~1e-13 for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: keeps the argument in the domain the series was fit for.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let g = 7.0_f64;
    let mut acc = 0.999_999_999_999_809_9_f64;
    for (i, c) in COEF.iter().enumerate() {
        acc += c / (x + i as f64 + 1.0);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}
