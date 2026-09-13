# The Safety Case graph and `evidence.esb`

Design note for `es-eval::evidence` and `es evidence verify`. Spec: spec 27.1 (Provenance
Bundle + Safety Case, `evidence.esb`, `traceability.json`, `revalidation_trigger`), spec 5.3
(hash chain, `HashChain::diff`), spec 10.5 (`report.json` / `evaluation.lock`), spec 25.1
(weights and graph files cross a trust boundary; the signature slot), spec 25.2 (the legal
checklist this bundle is the raw material for), spec 28.7 gates 16 and 17.

## 1. Why the graph exists at all

A provenance bundle records *lineage*: this policy came from that dataset, compiled by that
compiler. Spec 27.1 says the part it does not record is the one a conformity file needs:
**"there is no requirement -> verification evidence relation."** The Safety Case graph is that
relation and nothing more. It is three lists and one map:

```
Requirement { id, text, source, severity }        what must hold, and which rule says so
Claim       { id, text, requirement }             the argument a human makes for one requirement
Evidence    { id, kind, entry, hash,              a fact produced by a machine run,
              execution_hash }                      addressed as a bundle entry
Traceability: requirement id -> [evidence id]     the edge set
```

`Evidence.entry` is a path *inside the bundle* (`reports/0/report.json`), `hash` is the blake3
of that entry's bytes, and `execution_hash` is the spec 5.3 chain digest of the run that
produced it. Those three fields are what make the edge checkable rather than decorative: the
evidence cannot be a URL that rots, a file that was swapped, or a run of some other
configuration.

No node type is a trait and nothing is an extension point (INV-17). The graph is data.

## 2. Bundle layout

`evidence.esb` is the same container as `policy.esb` (`docs/design/policy-bundle.md`) with
`kind = Evidence` and more entries. Spec 27.1's tree, as far as M3 W5 fills it:

```
evidence.esb
├── manifest.toml         kind = Evidence; the spec 5.3 hash slots (spec 27.1 manifest.json)
├── chain.json            the full HashChain of the attested run -- see section 3
├── safety_case/case.json requirements + claims + evidence + traceability (spec 27.1)
├── reports/<i>/report.json        spec 10.5 EvaluationReport
├── reports/<i>/evaluation.lock    spec 10.5 EvaluationLock
└── policy/policy.esb     the deployment bundle, embedded whole
```

The policy bundle is embedded **as its own bytes**, one entry, not exploded into
`task/ observation/ learning/ deployment/ policy/`. Re-splitting it would mean the evidence
bundle re-serializes IRs the deployment bundle already sealed, and any difference in
serialization would be invisible. Embedded whole, `verify` calls `PolicyBundle::open` on it and
inherits every check that opening a deployment artifact already performs -- IR validators, the
cross-IR pass, the weights hash, the chain slots.

`scene/`, `training/`, `validation/determinism.json`, `domain_gap.json`, `hazards.json`,
`residual.json` and `signature` from the spec 27.1 tree are not written by this packet. Entries
this packet does not know about ride along via `extra_entries`, so adding them later is not a
format change.

## 3. Why a `chain.json` and not just the manifest

`BundleManifest.hashes` carries the subset of the chain a *deployment* artifact can know. It
has no `asset`, `scene`, `task_graph`, `evaluation` or `hardware` slot, so `execution_hash`
cannot be recomputed from it -- and the only thing that ties a report to this bundle is that
its `execution_hash` is the one the bundle attests to. `chain.json` is the serialized
`es_ir::hash::HashChain` of the run, which gives both: `chain.execution_hash()` to compare
against every report, and component-level fields for `HashChain::diff` under `--against`.

`verify` cross-checks the chain against the embedded policy bundle's manifest slot by slot:

| slot | severity on mismatch | why |
|---|---|---|
| `task`, `observation`, `deployment`, `compiler` | error | both sides compute these the same way, from the same IRs |
| `learning`, `policy` | warning | the bundle hashes the *declared* graph (`learning_hash` / `policy_hash`); the runtime chain records what the `PolicyRuntime` actually loaded (`lowering_hash` / `weights_hash`). They are two honest answers to two different questions, and the gap is a known one, not a silent one |
| `runtime`, `dataset` | not in a policy bundle | filled from the chain into the evidence manifest |

## 4. Completeness -- gate 16

The rule the gate checks, per requirement:

> A requirement is **covered** when at least one evidence linked to it exists as a bundle
> entry, its recorded `hash` equals the blake3 of that entry's bytes, and its
> `execution_hash` equals the bundle's own `chain.execution_hash()` -- both as the *case*
> records it and as the *entry itself* does.

The second half of that last clause is why `verify` parses the entries and does not only read
the case. Every `reports/<i>/` artifact spec 10.5 defines carries the hash of the run that
produced it: `report.json` as `execution_hash` bytes, `evaluation.lock` as hex, plus the
`evaluation_hash` of the suite that was locked. Both are parsed and both are cross-checked
against `chain.json`. Checking only `report.json` left a hole exactly the size of an
`EvidenceKind::Lock`: a lock lifted from another run, with the case rewritten to record that
file's blake3, satisfied every check the case could make about itself and counted as coverage.
An entry whose own contents name a different run (or a different Evaluation IR) is a
diagnostic *and* puts its evidence in `stale`.

Linked evidence that exists and hashes correctly but is of a different run is reported as
**stale**, not as coverage: it is a real measurement of a different machine. A requirement with
only stale evidence is uncovered, and `es evidence verify` exits 1.

`SafetyCase::validate` runs before any of that and rejects the graph itself: duplicate ids,
a claim or traceability key naming a requirement that does not exist, a traceability entry
naming evidence that does not exist, a requirement with no evidence list at all, and a case
with **no requirements** (`EVID-008`). The last one is not pedantry: `VerifyReport::ok()` is
"every requirement is covered", which is vacuously true over an empty table, so without it
`es evidence verify` prints a green gate-16 result over a document that argues nothing.
Building an `evidence.esb` from an invalid case is refused -- an unverifiable artifact is worse
than no artifact (spec 26.1: *what is not verified is not run*).

## 5. `revalidation_trigger` -- the "substantial modification" table

Spec 27.1 hangs `revalidation_trigger` off each requirement; this packet hangs it off the
**evidence kind** instead. The reason is that what invalidates a fact is a property of how the
fact was produced, not of the sentence it supports, and a per-requirement list is a second
place for the same truth to go stale. A requirement is affected exactly when one of its
evidences is.

`ChangedComponent` is the unit (spec 5.3, `HashChain::diff`).

| evidence kind | invalidated by |
|---|---|
| `EvalReport` | every component in `execution_hash` plus `Asset`, `Scene`, `Evaluation` -- i.e. all but `TaskGraph` |
| `Lock` | same as `EvalReport`: it records the conditions that report was produced under |
| `SafetyScenarioSuite` | `Asset`, `Scene`, `Task`, `Observation`, `Learning`, `Policy`, `Deployment`, `Compiler`, `Runtime`, `Hardware` -- the envelope, what is commanded, and what the machine does with it |
| `GoldenTest` | `Asset`, `Scene`, `Task`, `Observation`, `Compiler`, `Runtime`, `Hardware` -- a golden is a bit-exact claim about a pipeline, not about a policy |
| `HumanReview` | `Asset`, `Scene`, `Task`, `Deployment` -- what the reviewer read |

`TaskGraph` invalidates nothing: it is authoring identity (node ids), and spec 5.3 keeps it out
of `execution_hash` for the same reason. `Dataset` does not invalidate a `HumanReview` or a
`GoldenTest` but does invalidate a report, because it is inside `execution_hash`.

`es evidence verify a.esb --against b.esb` runs `a.chain.diff(&b.chain)` and prints, per changed
component, the evidence ids that the table says must be re-run. That is the mechanical half of
"substantial modification" in spec 27.1. It is advisory: `--against` never changes the exit
code, because "this bundle differs from that one" is not a defect of this bundle.

## 6. What verification does not prove

Stated here because spec 27.1 ends with a positioning warning and the CLI must not read as more
than it is.

- **A signature is authentication, not endorsement.** `EvidenceBundle::sign` (M4, W7,
  spec 25.1) adds a detached ed25519 signature over every entry's `(name, hash)` pair, and
  `verify(bytes, against, trusted_keys)` reports `signature: Valid(key) | Invalid | Absent |
  UntrustedKey`. `Valid` only means *the bytes the caller has are the bytes a `trusted_keys`
  holder signed* -- it says nothing about whether that signer checked anything, and it is
  still not part of [`VerifyReport::ok`]: whether a bundle is even required to carry a
  trusted signature is a caller policy (`es evidence verify --require-signature`), not a
  property of the bundle. `Absent` is what every bundle built before W7 still reports (a
  `schema_version: 1` manifest has no signature fields at all, and that is not a defect --
  spec 25.3 keeps old versions readable). A key is only as trustworthy as however
  `--trust pub.hex` was populated; this design does not add a PKI, a revocation list, or
  key rotation, only the primitive those would be built on.
- **A rerun plan is not a rerun.** `VerifyReport::replayable` (spec 28.6 gate 17 prerequisite)
  names, per `reports/<i>/` entry, the `evaluation_hash`/`execution_hash`/`seeds` a rerun
  would need, and `es evidence replay --dry-run` prints it. Nothing here re-executes a cell or
  compares a re-run metric against the recorded one -- that needs a `PhysicsBackend` +
  `PolicyRuntime` pair this crate does not link, which is exactly why plain `es evidence
  replay` prints `SKIPPED` (the same convention as `es eval run` with no backend available)
  instead of pretending to run something. The canonical-form check on `report.json` (byte-for-
  byte equal to `serde_json` re-serialization of its parsed form with sorted keys) only proves
  the entry is *machine-written*, not that re-running it would reproduce the same numbers.
- **No conformity.** The bundle is evidence collection and tracking for writing a technical
  file. It is not a certification, and a green `verify` -- signed or not -- is not a
  conformity statement (spec 27.1).
- **No judgement of the argument.** That every requirement has evidence says nothing about
  whether the requirements are the right ones or the claims follow. That is the human review
  the `HumanReview` evidence kind exists to record, not to replace.
