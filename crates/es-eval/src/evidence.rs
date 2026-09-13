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
//! * coverage requires the evidence's `execution_hash` to be the bundle's own — evidence from
//!   a different run is *stale*, reported, and does not count (gate 16);
//! * `revalidation_trigger` hangs off the evidence *kind*, not off the requirement, because
//!   what invalidates a fact is a property of how the fact was produced (spec 27.1).
//!
//! What this does not do, stated once here and once in the CLI output: there is no signature
//! (spec 25.1 reserves the slot; nothing fills or checks it) and no replay re-execution
//! (spec 28.6, gate 17). A green verify means the bundle is self-consistent, not authentic.

use std::collections::{BTreeMap, BTreeSet};

use es_compile::bundle::{
    self, report_entry, BundleError, BundleHashes, BundleKind, BundleManifest, PolicyBundle, CHAIN,
    EVALUATION_LOCK, POLICY_BUNDLE, REPORT_JSON, SAFETY_CASE,
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

/// Pretty JSON plus the trailing newline, so a bundle entry and the file `write_artifacts`
/// wrote to disk are byte-identical and hash the same.
fn to_json<T: Serialize>(value: &T, entry: &str) -> Result<Vec<u8>, EvidenceError> {
    let mut text = serde_json::to_string_pretty(value).map_err(json_err(entry))?;
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

    /// Everything the bundle claims about itself, re-derived (see [`VerifyReport`]).
    ///
    /// `against` is an earlier `evidence.esb` to compare the chain with; it only fills
    /// [`VerifyReport::revalidation`] and never affects [`VerifyReport::ok`], because "this
    /// bundle differs from that one" is not a defect of this bundle.
    pub fn verify(bytes: &[u8], against: Option<&[u8]>) -> Result<VerifyReport, EvidenceError> {
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

        // 3. Every report in the bundle is a report of *this* execution (spec 5.3).
        for (name, payload) in &b.entries {
            if !name.ends_with(REPORT_JSON) || !name.starts_with("reports/") {
                continue;
            }
            let report: EvaluationReport =
                serde_json::from_slice(payload).map_err(json_err(name))?;
            if report.execution_hash != execution_hash {
                diagnostics.push(
                    Diagnostic::new(
                        EVID_EXECUTION_HASH,
                        format!(
                            "\"{name}\" reports execution_hash {} but the bundle attests {}",
                            crate::hex32(&report.execution_hash),
                            crate::hex32(&execution_hash)
                        ),
                    )
                    .with_hint("the report is of a different run than the bundle's chain.json"),
                );
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
                    if intact && e.execution_hash == execution_hash {
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
                    "stale means the evidence is of a different execution_hash, or its entry \
                     no longer matches its recorded hash",
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

        Ok(VerifyReport {
            entries: b.entries.len(),
            execution_hash,
            coverage,
            revalidation,
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
    /// Linked evidence that is not: a different run, or an entry that no longer matches.
    pub stale: Vec<String>,
    pub covered: bool,
}

/// What [`EvidenceBundle::verify`] found. `ok()` is the CLI's exit code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    pub entries: usize,
    pub execution_hash: [u8; 32],
    pub coverage: Vec<RequirementCoverage>,
    /// Per changed component since `--against`, the evidence that must be re-run (spec 27.1).
    pub revalidation: Vec<(ChangedComponent, Vec<String>)>,
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
