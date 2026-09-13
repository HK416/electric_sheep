//! The `.esb` container and the `policy.esb` deployment bundle (spec 9.6, spec 10.5,
//! spec 25.3, spec 27.1).
//!
//! Spec 9.6: *"the deployment artifact is a single `policy.esb` bundle"*, and *"the policy
//! weights are not deployed alone — preprocessing and the safety constraints ship with
//! them"*. That is the whole design: a bundle holds the four IRs that decide what the robot
//! does, the checkpoint, and every hash of spec 5.3 needed to recompute `execution_hash`.
//!
//! The container is deliberately boring — see `docs/design/policy-bundle.md` for the byte
//! layout. No archive crate: the format is ~40 lines of little-endian integers, every entry
//! carries its own `blake3`, and entries are sorted by name so two builds of the same inputs
//! are byte-identical. Compression is not here on purpose; safetensors payloads do not
//! compress and an inflate path is attack surface on a deployment artifact (spec 25.1).
//!
//! The evidence bundle of spec 27.1 is the same container with more entries: same magic, same
//! manifest, `kind = Evidence`, plus `safety_case/`, `validation/` and `training/`. Nothing in
//! [`read`] or [`write`] knows the entry names, so that extension needs no format change.

use std::collections::BTreeMap;

use es_ir::cross::{self, IrBundle};
use es_ir::deployment::DeploymentIr;
use es_ir::evaluation::EvaluationIr;
use es_ir::hash::DatasetHash;
use es_ir::learning::LearningGraph;
use es_ir::observation::ObservationIr;
use es_ir::serial;
use es_ir::task::TaskIr;
use es_ir::Diagnostic;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::plan::{CpuPlan, PlanMode};

/// Container magic. The `1` is the container generation, not the manifest's `schema_version`:
/// a new generation is a new magic, so an old reader fails loudly (spec 25.3).
pub const MAGIC: [u8; 4] = *b"ESB1";

/// Manifest schema version (spec 25.3: the bundle format is versioned and old versions stay
/// readable; a *newer* one is refused).
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;

/// A deployment bundle is always compiled in this mode, so `compiler_hash` is comparable
/// between the machine that built the bundle and the one that opens it. The mode reaches
/// [`CpuPlan::compiler_hash`], so it cannot be a silent difference.
pub const BUNDLE_PLAN_MODE: PlanMode = PlanMode::Release;

pub const MANIFEST: &str = "manifest.toml";
pub const TASK: &str = "task.toml";
pub const OBSERVATION: &str = "observation.toml";
pub const LEARNING: &str = "learning.toml";
pub const DEPLOYMENT: &str = "deployment.toml";
pub const WEIGHTS: &str = "weights.safetensors";
pub const EVALUATION: &str = "evaluation.toml";

// --- manifest -------------------------------------------------------------------------------

/// What the bundle is for. The two share a container and a manifest; `Evidence` simply carries
/// more entries (spec 27.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleKind {
    Policy,
    Evidence,
}

/// The spec 5.3 chain, as much of it as a bundle can carry. Every slot is optional because the
/// two bundle kinds fill in different subsets: a `Policy` bundle has no `dataset` (training is
/// not on the deployment path) and no `runtime` (the backend is chosen where the bundle is
/// opened), while an `Evidence` bundle fills both.
///
/// `hardware_capability` is deliberately absent: it describes the machine that runs, not the
/// artifact, so it is computed at load (see `es-runtime-embedded`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleHashes {
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub task: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub learning: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub deployment: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub compiler: Option<[u8; 32]>,
    #[serde(with = "hex32", default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<[u8; 32]>,
    /// `content + schema + split` (spec 5.3). Absent in a deployment bundle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetHash>,
}

/// `manifest.toml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub kind: BundleKind,
    pub hashes: BundleHashes,
    /// RFC 3339 build time, when the builder chose to record one. Not hashed and not required:
    /// a timestamp inside the artifact would break byte-identical rebuilds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_utc: Option<String>,
    /// Spec 25.1 reserves a signature over the container; no crypto is implemented yet, so
    /// this is the slot and nothing verifies it. A reader must not treat `Some` as trust.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Vec<u8>>,
}

impl BundleManifest {
    pub fn new(kind: BundleKind, hashes: BundleHashes) -> Self {
        Self {
            schema_version: BUNDLE_SCHEMA_VERSION,
            kind,
            hashes,
            created_utc: None,
            signature: None,
        }
    }
}

/// Hex for `Option<[u8; 32]>`: a TOML manifest of 32-element integer arrays is unreadable, and
/// the manifest is the one part of a bundle a human inspects.
mod hex32 {
    use super::{Deserialize, Deserializer, Serializer};

    // serde's `with` contract fixes this signature; `Option<&T>` is not an option here.
    #[allow(clippy::ref_option)]
    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            None => s.serialize_none(),
            Some(d) => s.serialize_str(&super::hex(d)),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        let Some(s) = Option::<String>::deserialize(d)? else {
            return Ok(None);
        };
        super::unhex(&s)
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom(format!("'{s}' is not 64 hex characters")))
    }
}

pub fn hex(d: &[u8; 32]) -> String {
    use std::fmt::Write;
    d.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn unhex(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

// --- errors ---------------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BundleError {
    #[error("not an .esb container: magic is {0:?}, expected ESB1")]
    BadMagic([u8; 4]),
    #[error("the container ends inside {0}")]
    Truncated(&'static str),
    #[error("entry name is not UTF-8")]
    BadName,
    #[error("entries are not in sorted order: \"{1}\" follows \"{0}\"")]
    Unsorted(String, String),
    #[error("entry \"{name}\" does not match its recorded hash")]
    EntryHash { name: String },
    #[error("the bundle has no \"{0}\" entry")]
    MissingEntry(&'static str),
    #[error("manifest schema_version {found} is newer than {BUNDLE_SCHEMA_VERSION}")]
    UnsupportedVersion { found: u32 },
    #[error("this is a {found:?} bundle, not a {expected:?} one")]
    KindMismatch {
        expected: BundleKind,
        found: BundleKind,
    },
    #[error("{entry}: {message}")]
    Toml {
        entry: &'static str,
        message: String,
    },
    /// An IR failed its own validator or a cross-IR rule. The diagnostics are rendered because
    /// `Diagnostic` is not `Eq` and a bundle error is a leaf, not something to pattern-match.
    #[error("the bundle's IRs are not valid:\n{0}")]
    Invalid(String),
    /// The recomputed hash for `slot` is not the one the manifest records.
    #[error("hash mismatch in the \"{slot}\" slot of the chain (spec 5.3)")]
    HashMismatch { slot: &'static str },
    #[error("compiling the observation plan failed:\n{0}")]
    Compile(String),
}

fn render(diags: &[Diagnostic]) -> String {
    diags
        .iter()
        .map(|d| format!("  {d}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn errors_only(diags: Vec<Diagnostic>) -> Result<(), BundleError> {
    let errs: Vec<_> = diags.into_iter().filter(Diagnostic::is_error).collect();
    if errs.is_empty() {
        Ok(())
    } else {
        Err(BundleError::Invalid(render(&errs)))
    }
}

fn toml_err(entry: &'static str) -> impl Fn(serial::SerialError) -> BundleError {
    move |e| BundleError::Toml {
        entry,
        message: e.to_string(),
    }
}

// --- container ------------------------------------------------------------------------------

/// A container that has been read and whose every entry hash checked out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    pub manifest: BundleManifest,
    /// Every entry except `manifest.toml`, which is parsed out into `manifest`.
    pub entries: BTreeMap<String, Vec<u8>>,
}

impl Bundle {
    pub fn entry(&self, name: &'static str) -> Result<&[u8], BundleError> {
        self.entries
            .get(name)
            .map(Vec::as_slice)
            .ok_or(BundleError::MissingEntry(name))
    }

    fn text(&self, name: &'static str) -> Result<&str, BundleError> {
        std::str::from_utf8(self.entry(name)?).map_err(|e| BundleError::Toml {
            entry: name,
            message: e.to_string(),
        })
    }
}

/// Serialize a container. `manifest` becomes the `manifest.toml` entry; `entries` supplies the
/// rest. Deterministic: the `BTreeMap` fixes the order and nothing else varies, so identical
/// inputs give identical bytes.
pub fn write(
    manifest: &BundleManifest,
    entries: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<u8>, BundleError> {
    let text = serial::write_toml(manifest).map_err(toml_err(MANIFEST))?;
    let mut all: BTreeMap<&str, &[u8]> = BTreeMap::new();
    all.insert(MANIFEST, text.as_bytes());
    for (name, payload) in entries {
        all.insert(name.as_str(), payload.as_slice());
    }

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&u32::try_from(all.len()).unwrap_or(u32::MAX).to_le_bytes());
    for (name, payload) in &all {
        out.extend_from_slice(&u32::try_from(name.len()).unwrap_or(u32::MAX).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(blake3::hash(payload).as_bytes());
    }
    for payload in all.values() {
        out.extend_from_slice(payload);
    }
    Ok(out)
}

/// A cursor that reports where it ran out instead of panicking on a truncated file. Every
/// length in the header comes from untrusted bytes (spec 25.1).
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], BundleError> {
        let end = self.at.checked_add(n).ok_or(BundleError::Truncated(what))?;
        let s = self
            .b
            .get(self.at..end)
            .ok_or(BundleError::Truncated(what))?;
        self.at = end;
        Ok(s)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, BundleError> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, BundleError> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")))
    }
}

/// Parse a container, verifying every entry's `blake3` against the header. Rejects an
/// out-of-order or duplicated name, so the byte layout of a given set of entries is unique.
pub fn read(bytes: &[u8]) -> Result<Bundle, BundleError> {
    let mut c = Cursor { b: bytes, at: 0 };
    let magic: [u8; 4] = c.take(4, "the magic")?.try_into().expect("4 bytes");
    if magic != MAGIC {
        return Err(BundleError::BadMagic(magic));
    }
    let count = c.u32("the entry count")? as usize;

    let mut header: Vec<(String, u64, [u8; 32])> = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let name_len = c.u32("an entry name length")? as usize;
        let name = std::str::from_utf8(c.take(name_len, "an entry name")?)
            .map_err(|_| BundleError::BadName)?
            .to_owned();
        let len = c.u64("an entry length")?;
        let hash: [u8; 32] = c.take(32, "an entry hash")?.try_into().expect("32 bytes");
        if let Some((prev, ..)) = header.last() {
            if prev >= &name {
                return Err(BundleError::Unsorted(prev.clone(), name));
            }
        }
        header.push((name, len, hash));
    }

    let mut manifest = None;
    let mut entries = BTreeMap::new();
    for (name, len, hash) in header {
        let len = usize::try_from(len).map_err(|_| BundleError::Truncated("a payload"))?;
        let payload = c.take(len, "a payload")?;
        if blake3::hash(payload).as_bytes() != &hash {
            return Err(BundleError::EntryHash { name });
        }
        if name == MANIFEST {
            let text = std::str::from_utf8(payload).map_err(|e| BundleError::Toml {
                entry: MANIFEST,
                message: e.to_string(),
            })?;
            let m: BundleManifest = serial::parse_toml(text).map_err(toml_err(MANIFEST))?;
            if m.schema_version > BUNDLE_SCHEMA_VERSION {
                return Err(BundleError::UnsupportedVersion {
                    found: m.schema_version,
                });
            }
            manifest = Some(m);
        } else {
            entries.insert(name, payload.to_vec());
        }
    }

    Ok(Bundle {
        manifest: manifest.ok_or(BundleError::MissingEntry(MANIFEST))?,
        entries,
    })
}

// --- the policy bundle ------------------------------------------------------------------------

/// A `policy.esb` that has been opened, re-validated and hash-checked (spec 9.6).
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyBundle {
    pub manifest: BundleManifest,
    pub task: TaskIr,
    pub observation: ObservationIr,
    pub learning: LearningGraph,
    pub deployment: DeploymentIr,
    /// Checkpoint bytes, verified against `WeightsRef::hash` (spec 25.1: weights cross a trust
    /// boundary).
    pub weights: Vec<u8>,
    /// Present only in a bundle that also carries an evaluation suite (spec 10.5).
    pub evaluation: Option<EvaluationIr>,
}

impl PolicyBundle {
    /// Build the container bytes for a deployment.
    ///
    /// Validates each IR, runs the cross-IR pass, compiles the observation plan for the
    /// `compiler` slot, and fills every hash the deployment path can know. `runtime` stays
    /// empty: the inference backend is chosen where the bundle is opened.
    pub fn build(
        task: &TaskIr,
        observation: &ObservationIr,
        learning: &LearningGraph,
        deployment: &DeploymentIr,
        weights: &[u8],
    ) -> Result<Vec<u8>, BundleError> {
        let hashes = validate_and_hash(task, observation, learning, deployment, None)?;
        let declared = learning.policy.weights.hash();
        if blake3::hash(weights).as_bytes() != declared {
            return Err(BundleError::HashMismatch { slot: "weights" });
        }
        let entries = BTreeMap::from([
            (
                TASK.to_owned(),
                serial::task_to_toml(task)
                    .map_err(toml_err(TASK))?
                    .into_bytes(),
            ),
            (
                OBSERVATION.to_owned(),
                serial::observation_to_toml(observation)
                    .map_err(toml_err(OBSERVATION))?
                    .into_bytes(),
            ),
            (
                LEARNING.to_owned(),
                serial::learning_to_toml(learning)
                    .map_err(toml_err(LEARNING))?
                    .into_bytes(),
            ),
            (
                DEPLOYMENT.to_owned(),
                serial::deployment_to_toml(deployment)
                    .map_err(toml_err(DEPLOYMENT))?
                    .into_bytes(),
            ),
            (WEIGHTS.to_owned(), weights.to_vec()),
        ]);
        write(&BundleManifest::new(BundleKind::Policy, hashes), &entries)
    }

    /// Open a container: verify every entry hash, re-validate the four IRs, re-run
    /// `es_ir::cross::check`, recompute every hash and compare it with the manifest.
    ///
    /// Nothing here trusts the manifest. It is the artifact's claim; the IRs are the evidence.
    pub fn open(bytes: &[u8]) -> Result<Self, BundleError> {
        let raw = read(bytes)?;
        if raw.manifest.kind != BundleKind::Policy {
            return Err(BundleError::KindMismatch {
                expected: BundleKind::Policy,
                found: raw.manifest.kind,
            });
        }
        let task = serial::task_from_toml(raw.text(TASK)?).map_err(toml_err(TASK))?;
        let observation =
            serial::observation_from_toml(raw.text(OBSERVATION)?).map_err(toml_err(OBSERVATION))?;
        let learning =
            serial::learning_from_toml(raw.text(LEARNING)?).map_err(toml_err(LEARNING))?;
        let deployment =
            serial::deployment_from_toml(raw.text(DEPLOYMENT)?).map_err(toml_err(DEPLOYMENT))?;
        let evaluation = if raw.entries.contains_key(EVALUATION) {
            Some(
                serial::evaluation_from_toml(raw.text(EVALUATION)?)
                    .map_err(toml_err(EVALUATION))?,
            )
        } else {
            None
        };
        let weights = raw.entry(WEIGHTS)?.to_vec();
        if blake3::hash(&weights).as_bytes() != learning.policy.weights.hash() {
            return Err(BundleError::HashMismatch { slot: "weights" });
        }

        let recomputed = validate_and_hash(
            &task,
            &observation,
            &learning,
            &deployment,
            evaluation.as_ref(),
        )?;
        let m = &raw.manifest.hashes;
        for (slot, want, got) in [
            ("task", m.task, recomputed.task),
            ("observation", m.observation, recomputed.observation),
            ("learning", m.learning, recomputed.learning),
            ("policy", m.policy, recomputed.policy),
            ("deployment", m.deployment, recomputed.deployment),
            ("compiler", m.compiler, recomputed.compiler),
        ] {
            // A slot the manifest leaves empty is "not claimed", not "claimed as zero".
            if want.is_some() && want != got {
                return Err(BundleError::HashMismatch { slot });
            }
        }

        Ok(Self {
            manifest: raw.manifest,
            task,
            observation,
            learning,
            deployment,
            weights,
            evaluation,
        })
    }

    /// The observation plan the deployment runs (spec 9.5: the same IR, a CPU/NPU kernel
    /// instead of a GPU one). [`CpuPlan`] is not `Serialize` — it holds resolved buffer
    /// homes and ring state — so the bundle stores the IR and the plan is rebuilt here. The
    /// `compiler` hash checked in [`open`](Self::open) is what makes that sound.
    pub fn compile_plan(&self) -> Result<CpuPlan, BundleError> {
        CpuPlan::compile(&self.observation, BUNDLE_PLAN_MODE)
            .map_err(|d| BundleError::Compile(render(&d)))
    }
}

/// The shared half of `build` and `open`: every IR validator, the cross-IR pass, and the six
/// hashes a deployment bundle can compute from its own contents.
fn validate_and_hash(
    task: &TaskIr,
    observation: &ObservationIr,
    learning: &LearningGraph,
    deployment: &DeploymentIr,
    evaluation: Option<&EvaluationIr>,
) -> Result<BundleHashes, BundleError> {
    errors_only(task.validate())?;
    errors_only(observation.validate())?;
    errors_only(learning.validate())?;
    errors_only(deployment.validate())?;
    if let Some(ev) = evaluation {
        errors_only(ev.validate())?;
    }
    errors_only(cross::check(&IrBundle {
        task,
        observation,
        learning,
        deployment,
        evaluation,
    }))?;

    let plan = CpuPlan::compile(observation, BUNDLE_PLAN_MODE)
        .map_err(|d| BundleError::Compile(render(&d)))?;
    let one = |r: Result<[u8; 32], Diagnostic>| {
        r.map(Some).map_err(|d| BundleError::Invalid(render(&[d])))
    };
    Ok(BundleHashes {
        task: one(task.task_hash())?,
        observation: one(observation.observation_hash())?,
        learning: one(learning.learning_hash())?,
        policy: one(learning.policy_hash())?,
        deployment: one(deployment.deployment_hash())?,
        compiler: Some(plan.compiler_hash()),
        runtime: None,
        dataset: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> BTreeMap<String, Vec<u8>> {
        BTreeMap::from([
            ("b.bin".to_owned(), vec![1, 2, 3]),
            ("a.bin".to_owned(), vec![9; 100]),
        ])
    }

    fn manifest() -> BundleManifest {
        BundleManifest::new(BundleKind::Policy, BundleHashes::default())
    }

    #[test]
    fn container_round_trips_and_is_deterministic() {
        let bytes = write(&manifest(), &entries()).expect("writes");
        assert_eq!(&bytes[..4], b"ESB1");
        assert_eq!(bytes, write(&manifest(), &entries()).expect("writes"));
        let back = read(&bytes).expect("reads");
        assert_eq!(back.manifest, manifest());
        assert_eq!(back.entries, entries());
    }

    #[test]
    fn a_flipped_payload_byte_is_caught() {
        let mut bytes = write(&manifest(), &entries()).expect("writes");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert_eq!(
            read(&bytes),
            // Entries are stored in sorted order, so the last payload is `manifest.toml`.
            Err(BundleError::EntryHash {
                name: MANIFEST.to_owned()
            })
        );
    }

    #[test]
    fn a_foreign_container_is_rejected() {
        assert!(matches!(read(b"ZIP\0rest"), Err(BundleError::BadMagic(_))));
        assert!(matches!(read(b"ESB1"), Err(BundleError::Truncated(_))));
    }

    #[test]
    fn hex_round_trips_and_rejects_junk() {
        let d = [0xABu8; 32];
        assert_eq!(unhex(&hex(&d)), Some(d));
        assert_eq!(unhex("zz"), None);
        assert_eq!(unhex(&"g".repeat(64)), None);
    }

    #[test]
    fn the_manifest_is_human_readable_hex() {
        let m = BundleManifest::new(
            BundleKind::Policy,
            BundleHashes {
                task: Some([0x11; 32]),
                ..BundleHashes::default()
            },
        );
        let text = serial::write_toml(&m).expect("writes");
        assert!(text.contains(&"11".repeat(32)), "{text}");
        assert!(!text.contains("signature"), "None is omitted: {text}");
        assert_eq!(
            serial::parse_toml::<BundleManifest>(&text).expect("parses"),
            m
        );
    }
}
