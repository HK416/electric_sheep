# W7-learning-loop — `es loop collect / intervene / distill` and intervention labels

Spec: §13.1 (the loop is a first-class workflow; each step is a CLI command and an artifact),
§13.2 (the status of intervention data — labels, provenance, how interventions enter the
dataset), §13.3 (loop reproducibility — every step carries hashes), §19.1 (LeRobot format,
intervention labels are episode metadata), §19.2 (Dataset Identity), §19.3 (Training
Identity), §28.5 (M3 W7 row), §9.3/§9.4 and `INV-12` (the Safety Plane is on the only
actuator path, and no code path disables it).

Design note (read first): `docs/design/learning-loop.md`.

## context

```
crates/es-data/src/intervention.rs      (new)
crates/es-data/src/collect.rs           (new: Collector, LoopStep, distill)
crates/es-data/src/lib.rs               (+ `pub mod` lines, re-exports, one DataError variant)
crates/es-data/Cargo.toml               (+ es-env, es-safety, es-policy, es-compile,
                                           es-assets, es-physics-core; es-compile promoted
                                           from a dev-dependency)
crates/es-data/tests/loop_learning.rs   (new: one file, because all four tests share one
                                           ~450-line in-test fixture)
crates/es/src/cmd/loop.rs               (new)
crates/es/src/cmd/mod.rs                (+ one `pub mod` line)
crates/es/src/main.rs                   (+ one dispatch arm, usage text)
crates/es/tests/cli.rs                  (append only)
docs/design/learning-loop.md            (new)
docs/packets/M3/W7-learning-loop.md     (this file)
```

## spec

1. `es_data::intervention`
   - `InterventionSegment { episode, start_frame, end_frame, source: Teleop | Scripted |
     Corrective, operator_id: Option<String>, note: Option<String> }`, `end_frame` inclusive.
   - `ActionSourceCode` — `policy = 0`, `clamped = 1`, `fallback = 2`, `human = 3`; the
     mapping from `es_safety::ActionSource` is the design note's §2.2 table. A clamp or a
     fallback is **not** an intervention (§18.5) but is recorded distinctly.
   - `label(dataset_root, segments) -> Result<LabelReport>`: writes
     `meta/interventions.jsonl`, rewrites the `intervention` column of the affected episodes
     (streaming, one episode at a time), and re-hashes. `action_source` is never rewritten —
     it is collection-time provenance. `dataset_content_hash` changes;
     `dataset_schema_hash` changes only if a column had to be added to `meta/info.json`.
2. `es_data::collect`
   - `Collector::run::<B, F, NJ, H>(&CollectSpec, policy, new_backend, intervener)` drives
     `es_env::Env::step_with_policy` with one `SafetyPlane` built from the bundle's
     Deployment IR — the only actuator path (`INV-12`); an injected action is just another
     chunk and is validated like any other.
   - `Intervener = &mut dyn FnMut(episode, frame, &obs) -> Option<[f64; NJ]>`; an injection at
     control tick `t` labels frames `[t + latency_ticks, t + latency_ticks + execute_chunk)`
     (design note §5.1).
   - Writes episodes through `LeRobotWriter` with the feature set of the design note §3,
     including `action_source` per frame, `meta/interventions.jsonl`, and a `Collect`
     `LoopStep`. Image channels get a `video` feature and a warning; no mp4 is written.
   - `LoopStep { kind: Collect | Intervene | Distill, inputs, outputs, created }` appended to
     `<root>/loop.jsonl`, one JSON object per line, never rewritten.
3. `es_data::distill(inputs, SplitSpec { ratios, seed }, out_root) -> Result<TrainingIdentity>`
   merges datasets (episodes re-indexed, `interventions.jsonl` remapped, feature maps must
   agree), computes `DatasetIdentity` + `Split::deterministic`, writes
   `training_identity.json` and a `Distill` `LoopStep` into the output root **and** every
   input root. The training run itself is PyTorch-side and out of scope; the unknown
   `TrainingIdentity` slots are all-zero digests, never fabricated.
4. CLI
   - `es loop collect --policy policy.esb --scene scene.xml --episodes N --seed S --out <root>
     [--backend mujoco-cpu] [--runtime torch]` — checks backend and runtime availability
     first and prints `SKIPPED (<reason>)` with exit code **3** when either is unavailable,
     exactly as `es eval run` does (§1.4: never fake a run).
   - `es loop intervene --dataset <root> --segments segments.json`
   - `es loop distill --in <root>... --train 0.8 --val 0.1 --test 0.1 --seed S --out <root>`
   - Exit codes: 0 success, 1 runtime failure, 2 usage, 3 skipped.

## oracle

```
cargo fmt --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data -p es
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance

- collect with an in-test `FakeBackend` / `FakePolicy` (built in-test, the
  way `es-eval`'s own oracle does — copied, not imported, so the gate does not test a
  neighbour) plus a scripted intervener produces a dataset whose `action_source` column holds
  exactly `{policy, human}` and whose `meta/interventions.jsonl` segments match the frames the
  intervener injected on.
- `label()` on a written dataset changes `dataset_content_hash` and leaves
  `dataset_schema_hash` unchanged (the schema already carries the two columns); the
  `action_source` column is byte-identical before and after.
- distilling two datasets sums their episode counts, produces a split that
  is an exact disjoint partition, and yields a `TrainingIdentity` identical across two runs.
- `loop.jsonl` of the collected root carries three steps (`collect`, `intervene`, `distill`)
  whose input hashes chain onto the previous step's output hashes (§13.3).
- CLI: `es loop intervene` and `es loop distill` work end to end on in-test datasets;
  `es loop collect` exits 3 with `SKIPPED` when the backend or runtime is unavailable.

## forbidden

- Touching `crates/es-safety`, `crates/es-runtime-embedded`,
  `crates/es-eval/src/{evidence,domain_gap}.rs`, `crates/es-compile`, `crates/es-splat`,
  `crates/es-editor`, `crates/es/src/cmd/{evidence,gap}.rs` — neighbouring packets own them.
- Any branch that bypasses, disables or widens the Safety Plane for collection (`INV-12`),
  including "just for the test": an injected human action goes through `validate` like every
  other chunk.
- Rewriting `action_source` in `label()`, or deriving it from anything but the plane.
- Running the training loop, emitting a non-zero digest for a `TrainingIdentity` slot this
  side of the boundary cannot know, or writing an mp4 / a fabricated image observation.
- Editing the root `Cargo.toml`, editing golden files, or committing.
