//! `es evidence build/verify/keygen/sign/replay` (spec 27.1, spec 25.1, spec 28.7 gates 16
//! and 17).

use std::path::{Path, PathBuf};

use ed25519_dalek::{SigningKey, VerifyingKey};
use es_eval::evidence::{EvidenceBundle, ReplayPlan, SignatureStatus, VerifyReport};
use es_eval::EvaluationLock;
use es_ir::evaluation::EvaluationReport;
use es_ir::hash::HashChain;

use crate::error::CliError;
use crate::util::hex;

const VERIFY_HELP: &str = "\
es evidence verify <bundle.esb> [--against <other.esb>] [--trust <pub.hex>]...
                   [--require-signature] [--json]

Verifies an evidence bundle (spec 27.1): every container entry hash, every Safety Case
evidence link against the entry it names, every report's execution_hash against the chain
the bundle attests to, the chain against the embedded policy.esb, that every report.json is
canonical JSON (spec 28.6 gate 17 prerequisite), and traceability completeness -- every
requirement needs at least one evidence of this execution_hash (spec 28.7 gate 16).

    --against <other.esb>   also print, per component of `HashChain::diff` (spec 5.3), the
                            evidence that a change of that component invalidates
                            (spec 27.1 `revalidation_trigger`). Advisory: it never changes
                            the exit code on its own.
    --trust <pub.hex>       a file holding one hex-encoded ed25519 public key (as printed by
                            `es evidence keygen`/`sign`). Repeatable. A bundle signed by one
                            of these reports `signature: valid`; signed by any other key,
                            `untrusted key`.
    --require-signature     exit 1 unless the signature is `valid` against a --trust key.
                            Without it, signature status is informational only.
    --json                  print the VerifyReport as JSON instead of a table.

Exit code: 0 when every requirement is covered, nothing raised an error, and (with
--require-signature) the signature is valid against a trusted key; 1 otherwise; 2 on a usage
error.

This does NOT re-run the evaluations the bundle carries -- `replay --dry-run` prints the plan
a future `es evidence replay` would execute; the rerun itself needs a backend/runtime and is
M4 work (spec 28.6, gate 17). See docs/design/safety-case.md.
";

const BUILD_HELP: &str = "\
es evidence build --policy <policy.esb> --chain <chain.json> --report <dir>... \\
                  --case <case.json> --out <evidence.esb>

Seals a deployment bundle, the hash chain of the run that produced the reports (spec 5.3, a
serialized es_ir::hash::HashChain), one or more `es eval run` output directories (each with
report.json and evaluation.lock, spec 10.5) and a Safety Case document into an evidence
bundle (spec 27.1). --report may be repeated; directory order is entry order. The bundle is
unsigned -- pipe it through `es evidence sign` to add one.

The case must already record the blake3 of every entry it links to; `build` refuses a case
whose graph does not validate, and `verify` refuses the hashes that do not match. Nothing
fills them in for you on purpose -- a tool that writes the hashes it later checks verifies
nothing.
";

const KEYGEN_HELP: &str = "\
es evidence keygen --out key.eskey

Writes a fresh 32-byte ed25519 signing seed to <key.eskey> and prints its public key (hex).
Add `*.eskey` to .gitignore (already done in this repo) and keep the file out of version
control and off shared disks -- anyone holding it can sign bundles as you (spec 25.1).
";

const SIGN_HELP: &str = "\
es evidence sign --key <key.eskey> --in <evidence.esb> --out <signed.esb>

Re-emits <evidence.esb> with a detached ed25519 signature over every entry (spec 25.1):
`manifest.signature` and `manifest.signer_public_key`. Tampering any entry afterward makes
`es evidence verify` report `signature: invalid`. Prints the signer's public key (hex) so it
can be saved to a --trust file.
";

const REPLAY_HELP: &str = "\
es evidence replay --dry-run <bundle.esb>

Prints the rerun plan for every reports/<i>/ entry (spec 28.6 gate 17): report_index,
evaluation_hash, execution_hash, seeds. `--dry-run` is required -- without it this prints
SKIPPED and exits 3, the same convention as `es eval run` (spec 1.4: refused, never faked),
because the actual rerun needs a PhysicsBackend/PolicyRuntime pair this crate does not link.
";

pub fn dispatch(args: &[String]) -> Result<u8, CliError> {
    match args.first().map(String::as_str) {
        Some("verify") => verify(&args[1..]),
        Some("build") => build(&args[1..]),
        Some("keygen") => keygen(&args[1..]),
        Some("sign") => sign(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("--help" | "-h") | None => {
            println!("{VERIFY_HELP}\n{BUILD_HELP}\n{KEYGEN_HELP}\n{SIGN_HELP}\n{REPLAY_HELP}");
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

// --- key material (spec 25.1: keys never in the repo) ---------------------------------------

/// Decode a hex string (any surrounding whitespace tolerated) into exactly 32 bytes.
fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

fn read_signing_key(path: &Path) -> Result<SigningKey, CliError> {
    let bytes = read(path)?;
    let seed: [u8; 32] = bytes.try_into().map_err(|b: Vec<u8>| {
        CliError::Runtime(format!(
            "{}: a signing key is exactly 32 bytes, this file has {}",
            path.display(),
            b.len()
        ))
    })?;
    Ok(SigningKey::from_bytes(&seed))
}

fn read_trusted_key(path: &Path) -> Result<VerifyingKey, CliError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", path.display())))?;
    let bytes = parse_hex32(&text).ok_or_else(|| {
        CliError::Runtime(format!(
            "{}: expected 64 hex characters (one ed25519 public key)",
            path.display()
        ))
    })?;
    VerifyingKey::from_bytes(&bytes).map_err(|e| {
        CliError::Runtime(format!(
            "{}: not a valid ed25519 public key: {e}",
            path.display()
        ))
    })
}

/// Not a hardened CSPRNG -- see the note on `keygen` (spec 25.1 wants a real key management
/// story; this is the M4 packet's dev-facing stand-in, and it says so on every use).
///
/// ponytail: mixes libstd's OS-seeded `RandomState` (real entropy, but not audited as a CSPRNG)
/// with process/time/address jitter via blake3. Swap for an `OsRng`/`getrandom`-backed source
/// if `es evidence keygen` starts protecting anything beyond dev/CI signing.
fn generate_seed() -> [u8; 32] {
    use std::hash::{BuildHasher, Hasher};
    let mut mix = Vec::with_capacity(32);
    mix.extend_from_slice(
        &std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    mix.extend_from_slice(&u64::from(std::process::id()).to_le_bytes());
    let stack_marker = 0u8;
    mix.extend_from_slice(&(std::ptr::from_ref(&stack_marker) as u64).to_le_bytes());
    mix.extend_from_slice(
        &std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish()
            .to_le_bytes(),
    );
    *blake3::hash(&mix).as_bytes()
}

// --- keygen ----------------------------------------------------------------------------------

fn keygen(args: &[String]) -> Result<u8, CliError> {
    let mut out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(KEYGEN_HELP.to_owned())),
            "--out" => {
                out = Some(PathBuf::from(it.next().ok_or_else(|| {
                    CliError::Usage(format!("--out: missing value\n\n{KEYGEN_HELP}"))
                })?));
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{KEYGEN_HELP}"
                )))
            }
        }
    }
    let out = out.ok_or_else(|| CliError::Usage(KEYGEN_HELP.to_owned()))?;

    let seed = generate_seed();
    std::fs::write(&out, seed).map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    let public = SigningKey::from_bytes(&seed).verifying_key();
    println!("wrote {} (32 bytes)", out.display());
    println!("WARNING: this file is a private signing key -- keep it out of the repo and off");
    println!("shared disks (spec 25.1). `*.eskey` is already in .gitignore.");
    println!("public key: {}", hex(public.as_bytes()));
    Ok(0)
}

// --- sign ------------------------------------------------------------------------------------

fn sign(args: &[String]) -> Result<u8, CliError> {
    let (mut key, mut input, mut out) = (None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| CliError::Usage(format!("{a}: missing value\n\n{SIGN_HELP}")))
        };
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(SIGN_HELP.to_owned())),
            "--key" => key = Some(PathBuf::from(val()?)),
            "--in" => input = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{SIGN_HELP}"
                )))
            }
        }
    }
    let req = |v: Option<PathBuf>, name: &str| {
        v.ok_or_else(|| CliError::Usage(format!("{name} is required\n\n{SIGN_HELP}")))
    };
    let (key, input, out) = (req(key, "--key")?, req(input, "--in")?, req(out, "--out")?);

    let signing_key = read_signing_key(&key)?;
    let bytes = read(&input)?;
    let signed = EvidenceBundle::sign(&bytes, &signing_key)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", input.display())))?;
    std::fs::write(&out, &signed)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", out.display())))?;
    println!("wrote {} ({} bytes)", out.display(), signed.len());
    println!(
        "signer public key: {}",
        hex(signing_key.verifying_key().as_bytes())
    );
    Ok(0)
}

// --- verify --------------------------------------------------------------------------------

fn verify(args: &[String]) -> Result<u8, CliError> {
    let (mut bundle, mut against, mut json, mut require_signature) = (None, None, false, false);
    let mut trust: Vec<PathBuf> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(VERIFY_HELP.to_owned())),
            "--json" => json = true,
            "--require-signature" => require_signature = true,
            "--against" => {
                against = Some(PathBuf::from(it.next().ok_or_else(|| {
                    CliError::Usage(format!("--against: missing value\n\n{VERIFY_HELP}"))
                })?));
            }
            "--trust" => {
                trust.push(PathBuf::from(it.next().ok_or_else(|| {
                    CliError::Usage(format!("--trust: missing value\n\n{VERIFY_HELP}"))
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
    let trusted_keys = trust
        .iter()
        .map(|p| read_trusted_key(p))
        .collect::<Result<Vec<_>, CliError>>()?;
    let report = EvidenceBundle::verify(&bytes, other.as_deref(), &trusted_keys)
        .map_err(|e| CliError::Runtime(format!("{}: {e}", bundle.display())))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| CliError::Runtime(e.to_string()))?
        );
    } else {
        print_table(&report);
    }
    let signature_failed =
        require_signature && !matches!(report.signature, SignatureStatus::Valid(_));
    Ok(u8::from(!report.ok() || signature_failed))
}

fn signature_line(status: &SignatureStatus) -> String {
    match status {
        SignatureStatus::Valid(k) => format!("valid (signer {})", hex(k)),
        SignatureStatus::Invalid => "invalid".to_owned(),
        SignatureStatus::Absent => "absent (unsigned, or a schema_version 1 bundle)".to_owned(),
        SignatureStatus::UntrustedKey(k) => format!("untrusted key (signer {})", hex(k)),
    }
}

fn print_table(r: &VerifyReport) {
    println!("entries:        {}", r.entries);
    println!("execution_hash: {}", hex(&r.execution_hash));
    println!("signature:      {}", signature_line(&r.signature));
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

// --- replay ----------------------------------------------------------------------------------

fn replay(args: &[String]) -> Result<u8, CliError> {
    let (mut bundle, mut dry_run) = (None, false);
    for a in args {
        match a.as_str() {
            "--help" | "-h" => return Err(CliError::Usage(REPLAY_HELP.to_owned())),
            "--dry-run" => dry_run = true,
            other if other.starts_with("--") => {
                return Err(CliError::Usage(format!(
                    "unknown flag '{other}'\n\n{REPLAY_HELP}"
                )))
            }
            other if bundle.is_none() => bundle = Some(PathBuf::from(other)),
            other => {
                return Err(CliError::Usage(format!(
                    "unexpected argument '{other}'\n\n{REPLAY_HELP}"
                )))
            }
        }
    }
    let bundle = bundle.ok_or_else(|| CliError::Usage(REPLAY_HELP.to_owned()))?;

    if !dry_run {
        println!("SKIPPED (no PhysicsBackend/PolicyRuntime wired into `es`; pass --dry-run to see the plan)");
        return Ok(3);
    }

    let bytes = read(&bundle)?;
    let report = EvidenceBundle::verify(&bytes, None, &[])
        .map_err(|e| CliError::Runtime(format!("{}: {e}", bundle.display())))?;
    println!("REPORT  EVALUATION_HASH  EXECUTION_HASH   SEEDS");
    for ReplayPlan {
        report_index,
        evaluation_hash,
        execution_hash,
        seeds,
    } in &report.replayable
    {
        println!(
            "{:<7} {:<16} {:<16} {:?}",
            report_index,
            &hex(evaluation_hash)[..16],
            &hex(execution_hash)[..16],
            seeds
        );
    }
    println!();
    println!(
        "{} plan(s) printed. Rerunning them needs a backend/runtime and is not implemented \
         here (spec 28.6, gate 17); this is the plan a future `es evidence replay` would use.",
        report.replayable.len()
    );
    Ok(0)
}
