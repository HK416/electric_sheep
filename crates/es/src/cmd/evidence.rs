//! `es evidence build` / `es evidence verify` (spec 27.1, spec 28.7 gates 16 and 17).

use std::path::{Path, PathBuf};

use es_eval::evidence::{EvidenceBundle, VerifyReport};
use es_eval::EvaluationLock;
use es_ir::evaluation::EvaluationReport;
use es_ir::hash::HashChain;

use crate::error::CliError;
use crate::util::hex;

const VERIFY_HELP: &str = "\
es evidence verify <bundle.esb> [--against <other.esb>] [--json]

Verifies an evidence bundle (spec 27.1): every container entry hash, every Safety Case
evidence link against the entry it names, every report's execution_hash against the chain
the bundle attests to, the chain against the embedded policy.esb, and traceability
completeness -- every requirement needs at least one evidence of this execution_hash
(spec 28.7 gate 16).

    --against <other.esb>  also print, per component of `HashChain::diff` (spec 5.3), the
                           evidence that a change of that component invalidates
                           (spec 27.1 `revalidation_trigger`). Advisory: it never changes
                           the exit code.
    --json                 print the VerifyReport as JSON instead of a table.

Exit code: 0 when every requirement is covered and nothing raised an error; 1 otherwise;
2 on a usage error.

This does NOT check a signature (spec 25.1 reserves the slot; nothing fills it) and does NOT
re-run the evaluations it carries (spec 28.6, gate 17). See docs/design/safety-case.md.
";

const BUILD_HELP: &str = "\
es evidence build --policy <policy.esb> --chain <chain.json> --report <dir>... \\
                  --case <case.json> --out <evidence.esb>

Seals a deployment bundle, the hash chain of the run that produced the reports (spec 5.3, a
serialized es_ir::hash::HashChain), one or more `es eval run` output directories (each with
report.json and evaluation.lock, spec 10.5) and a Safety Case document into an evidence
bundle (spec 27.1). --report may be repeated; directory order is entry order.

The case must already record the blake3 of every entry it links to; `build` refuses a case
whose graph does not validate, and `verify` refuses the hashes that do not match. Nothing
fills them in for you on purpose -- a tool that writes the hashes it later checks verifies
nothing.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("verify") => verify(&args[1..]),
        Some("build") => build(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{VERIFY_HELP}\n{BUILD_HELP}");
            Ok(0)
        }
        Some(other) => Err(CliError::Usage(format!(
            "es evidence: unknown subcommand '{other}'"
        ))),
    }
}

fn read(path: &Path) -> Result<Vec<u8>, CliError> {
    std::fs::read(path).map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, CliError> {
    let bytes = read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))
}

// --- verify --------------------------------------------------------------------------------

fn verify(args: &[String]) -> Result<u8, CliError> {
    let (mut bundle, mut against, mut json) = (None, None, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(VERIFY_HELP.to_owned())),
            "--json" => json = true,
            "--against" => {
                against = Some(PathBuf::from(it.next().ok_or_else(|| {
                    CliError::Usage(format!("--against: missing value\n\n{VERIFY_HELP}"))
                })?));
            }
            other if other.starts_with("--") => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{VERIFY_HELP}"
                )))
            }
            other if bundle.is_none() => bundle = Some(PathBuf::from(other)),
            other => {
                return Err(CliError::Usage(format!(
                    "unexpected argument '{other}'\n\n{VERIFY_HELP}"
                )))
            }
        }
    }
    let bundle = bundle.ok_or_else(|| CliError::Usage(VERIFY_HELP.to_owned()))?;

    let bytes = read(&bundle)?;
    let other = against.as_deref().map(read).transpose()?;
    let report = EvidenceBundle::verify(&bytes, other.as_deref())
        .map_err(|e| CliError::Runtime(format!("{}: {e}", bundle.display())))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| CliError::Runtime(e.to_string()))?
        );
    } else {
        print_table(&report);
    }
    Ok(u8::from(!report.ok()))
}

fn print_table(r: &VerifyReport) {
    println!("entries:        {}", r.entries);
    println!("execution_hash: {}", hex(&r.execution_hash));
    // Spec 25.1 reserves the slot and nothing fills it; saying so every time is the whole
    // point of spec 27.1's positioning warning.
    println!("signature:      unverified (no signature scheme is implemented)");
    println!();
    println!("REQUIREMENT      SEVERITY   STATUS   EVIDENCE                     STALE");
    for c in &r.coverage {
        println!(
            "{:<16} {:<10} {:<8} {:<28} {}",
            c.requirement,
            format!("{:?}", c.severity).to_lowercase(),
            if c.covered { "covered" } else { "UNCOVERED" },
            if c.evidence.is_empty() {
                "-".to_owned()
            } else {
                c.evidence.join(",")
            },
            if c.stale.is_empty() {
                "-".to_owned()
            } else {
                c.stale.join(",")
            },
        );
    }
    if !r.revalidation.is_empty() {
        println!();
        println!("revalidation (spec 27.1), against the given bundle:");
        for (component, evidence) in &r.revalidation {
            println!(
                "  {:<12} -> {}",
                format!("{component:?}"),
                if evidence.is_empty() {
                    "nothing".to_owned()
                } else {
                    evidence.join(",")
                }
            );
        }
    }
    for d in &r.diagnostics {
        println!();
        print!("{d}");
    }
    println!();
    println!(
        "{}",
        if r.ok() {
            "OK: every requirement is covered by evidence of this execution"
        } else {
            "FAILED: the safety case is not complete (spec 28.7 gate 16)"
        }
    );
}

// --- build ---------------------------------------------------------------------------------

fn build(args: &[String]) -> Result<u8, CliError> {
    let (mut policy, mut chain, mut case, mut out) = (None, None, None, None);
    let mut reports: Vec<PathBuf> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{BUILD_HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(BUILD_HELP.to_owned())),
            "--policy" => policy = Some(PathBuf::from(val()?)),
            "--chain" => chain = Some(PathBuf::from(val()?)),
            "--case" => case = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--report" => reports.push(PathBuf::from(val()?)),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{BUILD_HELP}"
                )))
            }
        }
    }
    let req = |v: Option<PathBuf>, name: &str| {
        v.ok_or_else(|| CliError::Usage(format!("{name} is required\n\n{BUILD_HELP}")))
    };
    let (policy, chain, case, out) = (
        req(policy, "--policy")?,
        req(chain, "--chain")?,
        req(case, "--case")?,
        req(out, "--out")?,
    );
    if reports.is_empty() {
        return Err(CliError::Usage(format!(
            "at least one --report is required\n\n{BUILD_HELP}"
        )));
    }

    let policy_bytes = read(&policy)?;
    let chain: HashChain = read_json(&chain)?;
    let case = read_json(&case)?;
    let runs = reports
        .iter()
        .map(|dir| {
            let report: EvaluationReport = read_json(&dir.join("report.json"))?;
            let lock: EvaluationLock = read_json(&dir.join("evaluation.lock"))?;
            Ok((report, lock))
        })
        .collect::<Result<Vec<_>, CliError>>()?;

    let bytes = EvidenceBundle::build(
        &policy_bytes,
        &chain,
        &runs,
        &case,
        &std::collections::BTreeMap::new(),
    )
    .map_err(|e| CliError::Runtime(e.to_string()))?;
    std::fs::write(&out, &bytes)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    println!(
        "wrote {} ({} bytes, {} runs)",
        out.display(),
        bytes.len(),
        runs.len()
    );
    Ok(0)
}
