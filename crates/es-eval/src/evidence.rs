//! The Safety Case graph and `evidence.esb` (spec 27.1).
//!
//! A provenance bundle records lineage; spec 27.1 says the relation it is missing is
//! *"requirement -> verification evidence"*. This module is that relation: four plain data
//! types, a completeness check (spec 28.7 gate 16), a builder that seals them into the spec
//! 27.1 container, and a verifier that re-derives every claim the container makes about
//! itself.
//!
//! `docs/design/safety-case.md` is the design note. Three rules from it matter when reading
//! this file:
//!
//! * evidence is addressed as a *bundle entry plus its blake3*, never as a path or a URL, so
//!   an edge cannot silently point at something that was swapped;
//! * coverage requires the evidence's `execution_hash` to be the bundle's own — both as the
//!   case records it and as the entry's own spec 10.5 contents (`report.json`,
//!   `evaluation.lock`) record it — so evidence from a different run is *stale*, reported, and
//!   does not count (gate 16);
//! * `revalidation_trigger` hangs off the evidence *kind*, not off the requirement, because
//!   what invalidates a fact is a property of how the fact was produced (spec 27.1).
//!
//! `sign`/`verify` add a detached ed25519 signature over the container (spec 25.1): a
//! `manifest.schema_version` of 2 is the first that can carry it, but a version-1 bundle keeps
//! opening and simply verifies as `signature: Absent` -- there is no migration to write. What
//! this still does not do, stated once here and once in the CLI output: re-run anything.
//! `VerifyReport::replayable` names the reports a future `es evidence replay` would rerun; the
//! rerun itself needs a `PhysicsBackend`/`PolicyRuntime` and is out of scope until M4 finishes
//! it (spec 28.6, gate 17). A green verify means the bundle is self-consistent and, once
//! signed by a trusted key, authentic -- it is still not a replay and not a conformity claim.

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use es_compile::bundle::{
    self, report_entry, BundleError, BundleHashes, BundleKind, BundleManifest, PolicyBundle,
    BUNDLE_SCHEMA_VERSION, CHAIN, EVALUATION_LOCK, POLICY_BUNDLE, REPORT_JSON, SAFETY_CASE,
};
use es_ir::evaluation::EvaluationReport;
use es_ir::hash::{ChangedComponent, HashChain};
use es_ir::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::runner::EvaluationLock;

/// Schema version of `safety_case/case.json` (spec 25.3: every document is versioned).
pub const SCHEMA_VERSION: u32 = 1;

// --- the graph ---------------------------------------------------------------------------

/// How bad it is when the requirement does not hold. Carried for the human reading the
/// report; verification treats every severity the same, because an uncovered requirement is
/// an uncovered requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Minor,
    Serious,
    Critical,
}

/// One safety requirement (spec 27.1: `requirements.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    /// The requirement itself, in the words a technical file would use.
    pub text: String,
    /// Where it comes from: a regulation or standard reference, e.g. `"(EU) 2023/1230 Annex
    /// III 1.2.1"`. Free text on purpose — spec 27.1 notes the harmonised standards for AI
    /// safety functions are still being written, so an enum here would be wrong by next year.
    pub source: String,
    pub severity: Severity,
}

/// The argument a human makes for a requirement (spec 27.1). Nothing machine-checks a claim;
/// it is here so the reasoning is in the artifact and not in someone's mail.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub text: String,
    /// The requirement this claim argues for.
    pub requirement: String,
}

/// How a piece of evidence was produced. The kind decides what invalidates it
/// ([`invalidated_by`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// A spec 10.5 `report.json`.
    EvalReport,
    /// A Safety Plane violation-scenario suite result (spec 9, spec 28.7 gate 8).
    SafetyScenarioSuite,
    /// A spec 10.5 `evaluation.lock`: the conditions, not the numbers.
    Lock,
    /// A golden image or tensor comparison (spec 1.4).
    GoldenTest,
    /// A person reviewed something and signed off in prose. Invalidated by what they read.
    HumanReview,
}

impl EvidenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EvalReport => "eval_report",
            Self::SafetyScenarioSuite => "safety_scenario_suite",
            Self::Lock => "lock",
            Self::GoldenTest => "golden_test",
            Self::HumanReview => "human_review",
        }
    }
}

/// One verification fact, addressed so that it cannot rot (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub id: String,
    pub kind: EvidenceKind,
    /// Entry path inside the bundle, e.g. `reports/0/report.json`.
    pub entry: String,
    /// blake3 of that entry's bytes.
    pub hash: [u8; 32],
    /// The spec 5.3 chain digest of the run that produced it.
    pub execution_hash: [u8; 32],
}

/// `safety_case/case.json` (spec 27.1). `traceability` is the edge set: requirement id ->
/// evidence ids.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyCase {
    pub schema_version: u32,
    pub requirements: Vec<Requirement>,
    #[serde(default)]
    pub claims: Vec<Claim>,
    pub evidence: Vec<Evidence>,
    pub traceability: BTreeMap<String, Vec<String>>,
}

impl SafetyCase {
    /// Structural problems in the graph itself, before any bundle is involved: duplicate ids,
    /// references to something that does not exist, and a requirement nothing is linked to.
    ///
    /// Hash and `execution_hash` agreement need the bundle's entries and are checked by
    /// [`EvidenceBundle::verify`].
    pub fn validate(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        let reqs: BTreeSet<&str> = self.requirements.iter().map(|r| r.id.as_str()).collect();
        let evs: BTreeSet<&str> = self.evidence.iter().map(|e| e.id.as_str()).collect();
        let claims: BTreeSet<&str> = self.claims.iter().map(|c| c.id.as_str()).collect();
        for (kind, unique, all) in [
            ("requirement", reqs.len(), self.requirements.len()),
            ("evidence", evs.len(), self.evidence.len()),
            ("claim", claims.len(), self.claims.len()),
        ] {
            if unique != all {
                out.push(Diagnostic::new(
                    EVID_DUPLICATE_ID,
                    format!("two {kind}s in the safety case share an id"),
                ));
            }
        }

        for c in &self.claims {
            if !reqs.contains(c.requirement.as_str()) {
                out.push(Diagnostic::new(
                    EVID_DANGLING_REF,
                    format!(
                        "claim \"{}\" argues for requirement \"{}\", which the case does not \
                         declare",
                        c.id, c.requirement
                    ),
                ));
            }
        }
        for (req, links) in &self.traceability {
            if !reqs.contains(req.as_str()) {
                out.push(Diagnostic::new(
                    EVID_DANGLING_REF,
                    format!(
                        "traceability names requirement \"{req}\", which the case does not declare"
                    ),
                ));
            }
            for id in links {
                if !evs.contains(id.as_str()) {
                    out.push(
                        Diagnostic::new(
                            EVID_DANGLING_REF,
                            format!(
                                "requirement \"{req}\" is linked to evidence \"{id}\", which the \
                                 case does not declare"
                            ),
                        )
                        .with_hint("every traceability entry must name an item of `evidence`"),
                    );
                }
            }
        }
        if self.requirements.is_empty() {
            out.push(
                Diagnostic::new(
                    EVID_EMPTY_CASE,
                    "the safety case declares no requirements".to_owned(),
                )
                .with_hint(
                    "a case with nothing to cover is vacuously complete; spec 27.1 wants the \
                     requirements a conformity file argues over, not an empty table",
                ),
            );
        }
        for r in &self.requirements {
            if self.traceability.get(&r.id).is_none_or(Vec::is_empty) {
                out.push(
                    Diagnostic::new(
                        EVID_UNCOVERED,
                        format!("requirement \"{}\" has no verification evidence", r.id),
                    )
                    .with_hint(
                        "spec 27.1: a requirement with no evidence is what the Safety Case \
                         graph exists to make visible",
                    ),
                );
            }
        }
        out
    }

    fn evidence_by_id(&self, id: &str) -> Option<&Evidence> {
        self.evidence.iter().find(|e| e.id == id)
    }
}

// Diagnostic codes. These are not in the `es_ir_types::codes` dictionary yet (that crate is
// outside this packet) so they render with an empty title and, per `Diagnostic::new`, the
// default severity Error.
pub const EVID_DUPLICATE_ID: &str = "EVID-001";
pub const EVID_DANGLING_REF: &str = "EVID-002";
pub const EVID_UNCOVERED: &str = "EVID-003";
pub const EVID_ENTRY_MISSING: &str = "EVID-004";
pub const EVID_ENTRY_HASH: &str = "EVID-005";
pub const EVID_EXECUTION_HASH: &str = "EVID-006";
pub const EVID_CHAIN_SLOT: &str = "EVID-007";
pub const EVID_EMPTY_CASE: &str = "EVID-008";
pub const EVID_NOT_CANONICAL: &str = "EVID-009";

/// Which changes invalidate evidence of this kind (spec 27.1 `revalidation_trigger`, in the
/// units of spec 5.3 `HashChain::diff`). The table and its reasoning are in
/// `docs/design/safety-case.md` section 5.
#[must_use]
pub fn invalidated_by(kind: EvidenceKind) -> &'static [ChangedComponent] {
    use ChangedComponent as C;
    match kind {
        // Everything `execution_hash` covers, plus the scene it ran in and the suite it ran.
        // `TaskGraph` is authoring identity and changes no behaviour (spec 5.3).
        EvidenceKind::EvalReport | EvidenceKind::Lock => &[
            C::Asset,
            C::Scene,
            C::Task,
            C::Observation,
            C::Learning,
            C::Policy,
            C::Dataset,
            C::Deployment,
            C::Evaluation,
            C::Compiler,
            C::Runtime,
            C::Hardware,
        ],
        // The envelope, what is commanded, and what the machine does with it. The dataset is
        // upstream of the weights and reaches this only through `Policy`.
        EvidenceKind::SafetyScenarioSuite => &[
            C::Asset,
            C::Scene,
            C::Task,
            C::Observation,
            C::Learning,
            C::Policy,
            C::Deployment,
            C::Compiler,
            C::Runtime,
            C::Hardware,
        ],
        // A golden is a bit-exact claim about a pipeline, not about a policy.
        EvidenceKind::GoldenTest => &[
            C::Asset,
            C::Scene,
            C::Task,
            C::Observation,
            C::Compiler,
            C::Runtime,
            C::Hardware,
        ],
        // What the reviewer read.
        EvidenceKind::HumanReview => &[C::Asset, C::Scene, C::Task, C::Deployment],
    }
}

// --- errors ------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum EvidenceError {
    #[error(transparent)]
    Bundle(#[from] BundleError),
    #[error("{entry}: {message}")]
    Json { entry: String, message: String },
    #[error("the safety case is not valid:\n{0}")]
    Case(String),
    #[error("entry \"{0}\" is reserved by the evidence bundle layout (spec 27.1)")]
    ReservedEntry(String),
}

fn json_err(entry: &str) -> impl Fn(serde_json::Error) -> EvidenceError + '_ {
    move |e| EvidenceError::Json {
        entry: entry.to_owned(),
        message: e.to_string(),
    }
}

/// Canonical pretty JSON plus the trailing newline: routed through `serde_json::Value` (a
/// `BTreeMap` under the hood -- the workspace never turns on `preserve_order`), so object keys
/// come out sorted regardless of the field order the source struct declares them in. This is
/// the form [`EvidenceBundle::verify`] checks every `reports/<i>/report.json` against (the
/// canonical-form check): a bundle entry that is not exactly this is not something `build`
/// could have produced, signature aside.
fn to_json<T: Serialize>(value: &T, entry: &str) -> Result<Vec<u8>, EvidenceError> {
    canonical_json(value).map_err(json_err(entry))
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    let mut text = serde_json::to_string_pretty(&value)?;
    text.push('\n');
    Ok(text.into_bytes())
}

/// The `reports/<i>/` entries for a set of runs, exactly as [`EvidenceBundle::build`] will
/// write them. Public because a `SafetyCase` has to record the blake3 of an entry it links to,
/// which means the author needs the bytes before the bundle exists.
pub fn report_entries(
    reports: &[(EvaluationReport, EvaluationLock)],
) -> Result<BTreeMap<String, Vec<u8>>, EvidenceError> {
    let mut out = BTreeMap::new();
    for (i, (report, lock)) in reports.iter().enumerate() {
        let r = report_entry(i, REPORT_JSON);
        let l = report_entry(i, EVALUATION_LOCK);
        out.insert(r.clone(), to_json(report, &r)?);
        out.insert(l.clone(), to_json(lock, &l)?);
    }
    Ok(out)
}

// --- signing -------------------------------------------------------------------------------

/// What a bundle's ed25519 signature checks to (spec 25.1). `Valid` and `UntrustedKey` both
/// mean the cryptography checked out -- they differ only in whether the caller's
/// `trusted_keys` names that key, which is a policy question `verify` cannot answer on its
/// own. `Absent` is what every bundle built before this packet still reports: a version-1
/// manifest has no `signature` field at all, and that is not a defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignatureStatus {
    /// The signature verifies against `signer_public_key`, and that key is in `trusted_keys`.
    Valid([u8; 32]),
    /// Either the container was changed after signing, or `signature`/`signer_public_key`
    /// does not decode to a valid ed25519 pair at all.
    Invalid,
    /// `manifest.signature` is `None` (every version-1 bundle; an unsigned version-2 one).
    Absent,
    /// The signature verifies, but its key is not one the caller trusts.
    UntrustedKey([u8; 32]),
}

/// blake3 over every non-manifest entry's `(name, hash)` pair, in the sorted order a
/// `BTreeMap` already gives them. This, not the container's own per-entry hashes, is what
/// [`EvidenceBundle::sign`] and [`EvidenceBundle::verify`] exchange over ed25519: the manifest
/// itself is never in `entries` (`bundle::read` parses it out separately), so a signature
/// computed this way cannot be circular -- it authenticates every entry the signer saw,
/// including a `signer_public_key` swap being impossible without invalidating it, without
/// ever needing to hash itself.
fn signing_digest(entries: &BTreeMap<String, Vec<u8>>) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for (name, payload) in entries {
        hasher.update(name.as_bytes());
        hasher.update(blake3::hash(payload).as_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn signature_status(
    manifest: &BundleManifest,
    entries: &BTreeMap<String, Vec<u8>>,
    trusted_keys: &[VerifyingKey],
) -> SignatureStatus {
    let Some(sig_bytes) = manifest.signature.as_deref() else {
        return SignatureStatus::Absent;
    };
    let checked = manifest
        .signer_public_key
        .and_then(|pk| VerifyingKey::from_bytes(&pk).ok().zip(Some(pk)))
        .zip(Signature::from_slice(sig_bytes).ok())
        .filter(|((vk, _), sig)| vk.verify(&signing_digest(entries), sig).is_ok());
    match checked {
        None => SignatureStatus::Invalid,
        Some(((vk, pk), _)) if trusted_keys.contains(&vk) => SignatureStatus::Valid(pk),
        Some(((_, pk), _)) => SignatureStatus::UntrustedKey(pk),
    }
}

// --- the bundle --------------------------------------------------------------------------

/// An `evidence.esb` that has been read: the container's own hashes checked, the chain and the
/// case parsed. Nothing here is *verified* yet — that is [`EvidenceBundle::verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceBundle {
    pub manifest: BundleManifest,
    pub chain: HashChain,
    pub case: SafetyCase,
    pub entries: BTreeMap<String, Vec<u8>>,
}

impl EvidenceBundle {
    /// Seal a policy bundle, the chain of the run that produced the reports, those reports and
    /// a Safety Case into the spec 27.1 container.
    ///
    /// The policy bundle is opened first: an evidence bundle around an artifact that does not
    /// itself verify is worse than none. An invalid case is refused for the same reason.
    /// `extra` rides along unchanged (spec 27.1's `scene/`, `training/`, `validation/` are
    /// later packets and need no format change).
    pub fn build(
        policy_bundle: &[u8],
        chain: &HashChain,
        reports: &[(EvaluationReport, EvaluationLock)],
        case: &SafetyCase,
        extra: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Vec<u8>, EvidenceError> {
        let policy = PolicyBundle::open(policy_bundle)?;
        let errors: Vec<_> = case
            .validate()
            .into_iter()
            .filter(Diagnostic::is_error)
            .collect();
        if !errors.is_empty() {
            return Err(EvidenceError::Case(render(&errors)));
        }

        let mut entries = extra.clone();
        let reserved = [CHAIN, SAFETY_CASE, POLICY_BUNDLE];
        for name in entries.keys() {
            if reserved.contains(&name.as_str()) || name.starts_with("reports/") {
                return Err(EvidenceError::ReservedEntry(name.clone()));
            }
        }
        entries.extend(report_entries(reports)?);
        entries.insert(CHAIN.to_owned(), to_json(chain, CHAIN)?);
        entries.insert(SAFETY_CASE.to_owned(), to_json(case, SAFETY_CASE)?);
        entries.insert(POLICY_BUNDLE.to_owned(), policy_bundle.to_vec());

        let hashes = BundleHashes {
            // `runtime` and `dataset` are the two slots a deployment bundle cannot know
            // (spec 5.3); the attested chain does.
            runtime: Some(chain.runtime),
            dataset: Some(chain.dataset),
            ..policy.manifest.hashes
        };
        Ok(bundle::write(
            &BundleManifest::new(BundleKind::Evidence, hashes),
            &entries,
        )?)
    }

    /// Read and parse. Every container entry's blake3 is checked by `bundle::read`.
    pub fn open(bytes: &[u8]) -> Result<Self, EvidenceError> {
        let raw = bundle::read(bytes)?;
        if raw.manifest.kind != BundleKind::Evidence {
            return Err(BundleError::KindMismatch {
                expected: BundleKind::Evidence,
                found: raw.manifest.kind,
            }
            .into());
        }
        let parse = |name: &'static str| -> Result<&[u8], EvidenceError> {
            Ok(raw
                .entries
                .get(name)
                .map(Vec::as_slice)
                .ok_or(BundleError::MissingEntry(name))?)
        };
        let chain: HashChain = serde_json::from_slice(parse(CHAIN)?).map_err(json_err(CHAIN))?;
        let case: SafetyCase =
            serde_json::from_slice(parse(SAFETY_CASE)?).map_err(json_err(SAFETY_CASE))?;
        Ok(Self {
            manifest: raw.manifest,
            chain,
            case,
            entries: raw.entries,
        })
    }

    /// Re-emit `bytes` with `signature` and `signer_public_key` filled in (spec 25.1, spec
    /// 25.3 `schema_version` 2). `signing_key` is always built from a caller-supplied seed
    /// (`es evidence keygen`, `SigningKey::from_bytes`) -- nothing here generates one, which
    /// is the whole reason `ed25519-dalek`'s `rand_core` feature stays off.
    ///
    /// The container is otherwise untouched: same entries, same manifest hash slots, only the
    /// two signature fields and `schema_version` (bumped by `BundleManifest::new` when the
    /// input predates it) change.
    pub fn sign(bytes: &[u8], signing_key: &SigningKey) -> Result<Vec<u8>, EvidenceError> {
        let raw = bundle::read(bytes)?;
        if raw.manifest.kind != BundleKind::Evidence {
            return Err(BundleError::KindMismatch {
                expected: BundleKind::Evidence,
                found: raw.manifest.kind,
            }
            .into());
        }
        let signature = signing_key.sign(&signing_digest(&raw.entries));
        let manifest = BundleManifest {
            // A version-1 input is upgraded to the version that can carry a signature; a
            // version-2 one is already there.
            schema_version: BUNDLE_SCHEMA_VERSION,
            signature: Some(Vec::from(signature.to_bytes())),
            signer_public_key: Some(signing_key.verifying_key().to_bytes()),
            ..raw.manifest
        };
        Ok(bundle::write(&manifest, &raw.entries)?)
    }

    /// Everything the bundle claims about itself, re-derived (see [`VerifyReport`]).
    ///
    /// `against` is an earlier `evidence.esb` to compare the chain with; it only fills
    /// [`VerifyReport::revalidation`] and never affects [`VerifyReport::ok`], because "this
    /// bundle differs from that one" is not a defect of this bundle. `trusted_keys` decides
    /// [`SignatureStatus::Valid`] vs. [`SignatureStatus::UntrustedKey`] and likewise never
    /// affects `ok()` -- whether to *require* a trusted signature is a caller policy (the CLI's
    /// `--require-signature`), not a property of the bundle.
    pub fn verify(
        bytes: &[u8],
        against: Option<&[u8]>,
        trusted_keys: &[VerifyingKey],
    ) -> Result<VerifyReport, EvidenceError> {
        let b = Self::open(bytes)?;
        let mut diagnostics = b.case.validate();
        let execution_hash = b.chain.execution_hash();

        // 1. The embedded deployment bundle, re-opened: IR validators, the cross-IR pass, the
        //    weights hash and its own chain slots (spec 9.6).
        let policy_bytes = b
            .entries
            .get(POLICY_BUNDLE)
            .ok_or(BundleError::MissingEntry(POLICY_BUNDLE))?;
        let policy = PolicyBundle::open(policy_bytes)?;
        diagnostics.extend(chain_vs_manifest(&b.chain, &policy.manifest.hashes));

        // 2. Every evidence edge points at an entry that is there and unchanged.
        for e in &b.case.evidence {
            let Some(payload) = b.entries.get(&e.entry) else {
                diagnostics.push(Diagnostic::new(
                    EVID_ENTRY_MISSING,
                    format!(
                        "evidence \"{}\" names entry \"{}\", which the bundle does not carry",
                        e.id, e.entry
                    ),
                ));
                continue;
            };
            if blake3::hash(payload).as_bytes() != &e.hash {
                diagnostics.push(
                    Diagnostic::new(
                        EVID_ENTRY_HASH,
                        format!(
                            "evidence \"{}\": entry \"{}\" does not match the hash the safety \
                             case records for it",
                            e.id, e.entry
                        ),
                    )
                    .with_hint("the entry was changed after the case was written"),
                );
            }
        }

        // 3. Every artifact in the bundle is an artifact of *this* execution (spec 5.3). Both
        //    spec 10.5 files carry the hash, so both are parsed: a `report.json` records it as
        //    bytes, an `evaluation.lock` as hex, and an entry the case links to but whose own
        //    contents name another run is not evidence of this one — it goes in `stale`
        //    below, not just in the diagnostics.
        let hex_execution = crate::hex32(&execution_hash);
        let mut foreign: BTreeSet<&String> = BTreeSet::new();
        for (name, payload) in &b.entries {
            if !name.starts_with("reports/") {
                continue;
            }
            let (got, what) = if name.ends_with(REPORT_JSON) {
                let report: EvaluationReport =
                    serde_json::from_slice(payload).map_err(json_err(name))?;
                // Canonical-form check (spec 28.6 gate 17, replay prerequisite): re-serializing
                // the parsed value with sorted keys must reproduce these exact bytes. That is
                // the only thing a hand-edited or differently-indented `report.json` cannot
                // survive, and it is also exactly the form `build` writes -- see `to_json`.
                let canonical = canonical_json(&report).map_err(json_err(name))?;
                if canonical != *payload {
                    diagnostics.push(
                        Diagnostic::new(
                            EVID_NOT_CANONICAL,
                            format!("\"{name}\" is not canonical JSON"),
                        )
                        .with_hint(
                            "re-serializing the parsed report with sorted keys does not \
                             reproduce these bytes -- the entry was hand-edited or re-indented \
                             after `es evidence build` wrote it",
                        ),
                    );
                }
                (crate::hex32(&report.execution_hash), "reports")
            } else if name.ends_with(EVALUATION_LOCK) {
                let lock: EvaluationLock =
                    serde_json::from_slice(payload).map_err(json_err(name))?;
                // The evaluation slot is the suite the lock says it locked; a lock of a
                // different suite is as foreign as one of a different run.
                if b.chain
                    .evaluation
                    .is_some_and(|e| crate::hex32(&e) != lock.evaluation_hash)
                {
                    diagnostics.push(
                        Diagnostic::new(
                            EVID_EXECUTION_HASH,
                            format!(
                                "\"{name}\" locks evaluation_hash {} but the bundle's chain \
                                 records {}",
                                lock.evaluation_hash,
                                b.chain
                                    .evaluation
                                    .map_or_else(|| "none".to_owned(), |e| crate::hex32(&e))
                            ),
                        )
                        .with_hint("the lock is of a different Evaluation IR"),
                    );
                    foreign.insert(name);
                }
                (lock.execution_hash.clone(), "locks")
            } else {
                continue;
            };
            if got != hex_execution {
                diagnostics.push(
                    Diagnostic::new(
                        EVID_EXECUTION_HASH,
                        format!(
                            "\"{name}\" {what} execution_hash {got} but the bundle attests \
                             {hex_execution}"
                        ),
                    )
                    .with_hint("the artifact is of a different run than the bundle's chain.json"),
                );
                foreign.insert(name);
            }
        }

        // 4. Coverage, per requirement (spec 28.7 gate 16).
        let coverage: Vec<_> = b
            .case
            .requirements
            .iter()
            .map(|r| {
                let links = b.case.traceability.get(&r.id).cloned().unwrap_or_default();
                let (mut evidence, mut stale) = (Vec::new(), Vec::new());
                for id in links {
                    let Some(e) = b.case.evidence_by_id(&id) else {
                        continue; // already a dangling-ref diagnostic
                    };
                    let intact = b
                        .entries
                        .get(&e.entry)
                        .is_some_and(|p| blake3::hash(p).as_bytes() == &e.hash);
                    if intact && e.execution_hash == execution_hash && !foreign.contains(&e.entry) {
                        evidence.push(id);
                    } else {
                        stale.push(id);
                    }
                }
                RequirementCoverage {
                    requirement: r.id.clone(),
                    severity: r.severity,
                    covered: !evidence.is_empty(),
                    evidence,
                    stale,
                }
            })
            .collect();
        for c in coverage
            .iter()
            .filter(|c| !c.covered && !c.stale.is_empty())
        {
            diagnostics.push(
                Diagnostic::new(
                    EVID_UNCOVERED,
                    format!(
                        "requirement \"{}\" has only stale evidence ({})",
                        c.requirement,
                        c.stale.join(", ")
                    ),
                )
                .with_hint(
                    "stale means the evidence is of a different execution_hash -- as the case \
                     records it or as the entry itself does -- or its entry no longer matches \
                     its recorded hash",
                ),
            );
        }

        // 5. What a change since `against` would force to be re-run (spec 27.1).
        let mut revalidation = Vec::new();
        if let Some(other) = against {
            let other = Self::open(other)?;
            for component in b.chain.diff(&other.chain) {
                let affected: Vec<String> = b
                    .case
                    .evidence
                    .iter()
                    .filter(|e| invalidated_by(e.kind).contains(&component))
                    .map(|e| e.id.clone())
                    .collect();
                revalidation.push((component, affected));
            }
        }

        // 6. The rerun plan a future `es evidence replay` would execute (spec 28.6 gate 17):
        //    one entry per `reports/<i>/` this bundle actually carries, in index order.
        let mut replayable = Vec::new();
        for i in 0.. {
            let (Some(rp), Some(lp)) = (
                b.entries.get(&report_entry(i, REPORT_JSON)),
                b.entries.get(&report_entry(i, EVALUATION_LOCK)),
            ) else {
                break;
            };
            let name = report_entry(i, REPORT_JSON);
            let report: EvaluationReport = serde_json::from_slice(rp).map_err(json_err(&name))?;
            let lock: EvaluationLock =
                serde_json::from_slice(lp).map_err(json_err(&report_entry(i, EVALUATION_LOCK)))?;
            replayable.push(ReplayPlan {
                report_index: i,
                evaluation_hash: report.evaluation_hash,
                execution_hash: report.execution_hash,
                seeds: lock.seeds,
            });
        }

        Ok(VerifyReport {
            entries: b.entries.len(),
            execution_hash,
            signature: signature_status(&b.manifest, &b.entries, trusted_keys),
            coverage,
            revalidation,
            replayable,
            diagnostics,
        })
    }
}

/// The chain the runtime attested vs. what the deployment bundle sealed. `learning` and
/// `policy` are warnings, not errors: the bundle hashes the declared graph, the runtime chain
/// records what was actually loaded (`docs/design/safety-case.md` section 3).
fn chain_vs_manifest(chain: &HashChain, m: &BundleHashes) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (slot, want, got) in [
        ("task", m.task, chain.task),
        ("observation", m.observation, chain.observation),
        ("deployment", m.deployment, chain.deployment),
        ("compiler", m.compiler, chain.compiler),
    ] {
        if want.is_some_and(|w| w != got) {
            out.push(Diagnostic::new(
                EVID_CHAIN_SLOT,
                format!(
                    "chain.json's \"{slot}\" hash is not the one the embedded policy bundle \
                     records (spec 5.3)"
                ),
            ));
        }
    }
    out
}

fn render(diags: &[Diagnostic]) -> String {
    diags
        .iter()
        .map(|d| format!("  {d}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One row of the gate-16 table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementCoverage {
    pub requirement: String,
    pub severity: Severity,
    /// Evidence that exists, hashes correctly and is of this bundle's execution.
    pub evidence: Vec<String>,
    /// Linked evidence that is not: an entry that no longer matches its recorded hash, or one
    /// of a different run -- either because the case records a foreign `execution_hash` or
    /// because the entry's own `report.json` / `evaluation.lock` contents do.
    pub stale: Vec<String>,
    pub covered: bool,
}

/// The rerun a future `es evidence replay` would perform for one `reports/<i>/` entry (spec
/// 28.6 gate 17). Everything a `PhysicsBackend` + `PolicyRuntime` pair needs to reproduce the
/// cell and compare against `execution_hash`; nothing here runs it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayPlan {
    pub report_index: usize,
    pub evaluation_hash: [u8; 32],
    pub execution_hash: [u8; 32],
    pub seeds: Vec<u64>,
}

/// What [`EvidenceBundle::verify`] found. `ok()` is the CLI's exit code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub entries: usize,
    pub execution_hash: [u8; 32],
    /// Whether the container's ed25519 signature checks out against a caller-trusted key
    /// (spec 25.1). Never part of [`Self::ok`] -- whether a signature is *required* is the
    /// CLI's `--require-signature`, a caller policy, not a property of the bundle.
    pub signature: SignatureStatus,
    pub coverage: Vec<RequirementCoverage>,
    /// Per changed component since `--against`, the evidence that must be re-run (spec 27.1).
    pub revalidation: Vec<(ChangedComponent, Vec<String>)>,
    /// The rerun plan for every `reports/<i>/` entry the bundle carries (spec 28.6 gate 17).
    pub replayable: Vec<ReplayPlan>,
    pub diagnostics: Vec<Diagnostic>,
}

impl VerifyReport {
    /// Gate 16: every requirement covered, and nothing raised an error.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.coverage.iter().all(|c| c.covered)
            && !self.diagnostics.iter().any(Diagnostic::is_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(id: &str) -> Requirement {
        Requirement {
            id: id.to_owned(),
            text: "EE speed stays under 0.25 m/s in a human-detected zone".to_owned(),
            source: "(EU) 2023/1230 Annex III".to_owned(),
            severity: Severity::Critical,
        }
    }

    fn evidence(id: &str) -> Evidence {
        Evidence {
            id: id.to_owned(),
            kind: EvidenceKind::EvalReport,
            entry: report_entry(0, REPORT_JSON),
            hash: [1u8; 32],
            execution_hash: [2u8; 32],
        }
    }

    fn case() -> SafetyCase {
        SafetyCase {
            schema_version: SCHEMA_VERSION,
            requirements: vec![requirement("REQ-07")],
            claims: vec![Claim {
                id: "CLM-01".to_owned(),
                text: "the envelope bounds it and the suite measured it".to_owned(),
                requirement: "REQ-07".to_owned(),
            }],
            evidence: vec![evidence("EV-01")],
            traceability: BTreeMap::from([("REQ-07".to_owned(), vec!["EV-01".to_owned()])]),
        }
    }

    fn codes(diags: &[Diagnostic]) -> Vec<String> {
        diags.iter().map(|d| d.code.as_str().to_owned()).collect()
    }

    #[test]
    fn a_complete_case_validates_clean() {
        assert_eq!(case().validate(), vec![]);
    }

    #[test]
    fn a_requirement_with_no_evidence_is_reported() {
        let mut c = case();
        c.requirements.push(requirement("REQ-08"));
        assert_eq!(codes(&c.validate()), [EVID_UNCOVERED]);

        // An empty list is the same thing as no list.
        c.traceability.insert("REQ-08".to_owned(), vec![]);
        assert_eq!(codes(&c.validate()), [EVID_UNCOVERED]);
    }

    #[test]
    fn a_case_with_no_requirements_is_a_diagnostic() {
        // `VerifyReport::ok()` is "every requirement covered", which is vacuously true over an
        // empty table -- so the empty table itself has to be the finding.
        let c = SafetyCase {
            requirements: vec![],
            claims: vec![],
            traceability: BTreeMap::new(),
            ..case()
        };
        assert_eq!(codes(&c.validate()), [EVID_EMPTY_CASE]);
        assert!(c.validate().iter().any(Diagnostic::is_error));
    }

    #[test]
    fn dangling_references_are_reported() {
        let mut c = case();
        c.traceability
            .insert("REQ-99".to_owned(), vec!["EV-99".to_owned()]);
        c.claims[0].requirement = "REQ-99".to_owned();
        // Dangling claim, dangling traceability key, dangling evidence link.
        assert_eq!(
            codes(&c.validate()),
            [EVID_DANGLING_REF, EVID_DANGLING_REF, EVID_DANGLING_REF]
        );
    }

    #[test]
    fn duplicate_ids_are_reported() {
        let mut c = case();
        c.evidence.push(evidence("EV-01"));
        assert_eq!(codes(&c.validate()), [EVID_DUPLICATE_ID]);
    }

    #[test]
    fn the_revalidation_table_follows_the_spec_27_1_rules() {
        use ChangedComponent as C;
        // Authoring identity changes nothing (spec 5.3): no kind lists it.
        for kind in [
            EvidenceKind::EvalReport,
            EvidenceKind::Lock,
            EvidenceKind::SafetyScenarioSuite,
            EvidenceKind::GoldenTest,
            EvidenceKind::HumanReview,
        ] {
            assert!(
                !invalidated_by(kind).contains(&C::TaskGraph),
                "{kind:?} must not depend on task_graph"
            );
            // A changed envelope or a changed machine reaches every run-produced kind.
            if kind != EvidenceKind::GoldenTest && kind != EvidenceKind::HumanReview {
                assert!(invalidated_by(kind).contains(&C::Deployment), "{kind:?}");
            }
        }
        // Retraining changes `policy_hash`, which is exactly the spec 27.1 example.
        assert!(invalidated_by(EvidenceKind::EvalReport).contains(&C::Policy));
        assert!(!invalidated_by(EvidenceKind::HumanReview).contains(&C::Policy));
        // A golden is a claim about the pipeline, not about the weights.
        assert!(!invalidated_by(EvidenceKind::GoldenTest).contains(&C::Dataset));
    }

    #[test]
    fn a_case_round_trips_through_json() {
        let bytes = to_json(&case(), SAFETY_CASE).expect("serializes");
        assert_eq!(
            serde_json::from_slice::<SafetyCase>(&bytes).expect("parses"),
            case()
        );
    }
}
