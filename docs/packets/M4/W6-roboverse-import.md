# M4 W6 — RoboVerse / MetaSim task conversion

## context

- `crates/es-data/src/roboverse.rs` (new)
- `crates/es-data/src/lib.rs` (`pub mod roboverse;` only)
- `crates/es-data/tests/roboverse.rs` (new)
- `tests/fixtures/roboverse/*.json` (new, hand-written, < 10 KB each)
- `crates/es/src/cmd/import.rs` (add the `roboverse` subcommand; minimal insertion — a
  sibling packet adds a `usd` variant to the same file)
- `crates/es/tests/cli.rs` (append only)
- `docs/api-notes/roboverse.md` (new)
- `docs/packets/M4/W6-roboverse-import.md` (this file)

## spec

- §14.4 (external conversion): `RoboVerse / MetaSim` is one of four named conversion targets
  (LeRobot config, Isaac Lab task config, MJCF, RoboVerse/MetaSim, Gymnasium spec) feeding a
  "Semantic Mapping Report" into the IR; an unmapped item is `severity: error` and blocks
  execution.
- §0.3 line "RoboVerse / MetaSim: simulator-agnostic config, 276 tasks" — the only spec-level
  description of RoboVerse's own shape; everything more specific is `docs/api-notes/roboverse.md`.
- §25.2 — "Isaac Lab / RoboVerse conversion output derivative-work status: needs confirmation".
  This packet does not resolve the license question; it carries the source's `license` field
  through to `Converted::provenance` and `Converted::scene_refs[].license` unconditionally.
- §6 (Task IR, lines 616–709) and §7.4 (Observation IR declaration link) — the two IRs this
  conversion targets, and the boundary (`ObservationSpec` channel <-> Observation IR source
  node) that `es_ir::cross::check`'s `task_observation` pass enforces.
- INV-16 — no pickle-based loading anywhere in this workspace; RoboVerse's own trajectory
  format (`unverified`, believed `.pkl`/`.npz`) is out of scope for this converter by
  construction (only the JSON task *config* is read).

## oracle

```
cargo fmt -p es-data -p es --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data -p es
cargo xtask check-spec-refs
```

## acceptance

- `docs/api-notes/roboverse.md` written before the implementation, marking every field
  `unverified` unless fetched from `roboverse.wiki`/GitHub (fetched: `ScenarioCfg` top-level
  shape, `RobotCfg`'s field names, the three checker class names; unverified: `BaseObjCfg`,
  camera fields, checker constructor fields, randomization shape, trajectory format, license
  resolution).
- `RoboVerseTask` parses JSON only (documented: MetaSim's native Python config needs a
  one-line export step this crate does not implement); unknown top-level fields land in
  `extra` and are reported as warnings, never rejected.
- `convert(&RoboVerseTask) -> Result<Converted, ConvertError>` where `Converted` carries
  `task: TaskIr`, `observation: ObservationIr`, `scene_refs: Vec<SceneAssetRef>`,
  `provenance: Provenance`, `warnings: Vec<String>`, `unmapped: Vec<Unmapped>`.
- Robots/objects become `SceneAssetRef { path, format, license }`; cameras become both a Task
  IR `ObservationSpec` channel and an Observation IR `ImageInput -> Resize` chain; the checker
  becomes `Reward` + `Terminate(Success)` for the three named kinds (`DetectedChecker`,
  `JointPosChecker`, `PositionShiftChecker`), and anything else is `Unmapped { severity:
  Error }`; episode length becomes `TaskConfig::max_episode_steps` plus a
  `Terminate(Timeout)` chain; a recognized `randomization[].distribution.kind` becomes a
  `Randomization` node, an unrecognized one is a warning, not an error.
- Two fixtures: a pick-and-place task with a `DetectedChecker`, and a reach task with a
  `JointPosChecker` plus one randomization term. Both convert with an empty `unmapped` and
  validate clean: `TaskIr::validate()`, `ObservationIr::validate()`, and the `XIR-001`/
  `XIR-002` (Task <-> Observation) diagnostics out of `es_ir::cross::check` are all empty.
- A third fixture with an unrecognized `checker.kind` converts to exactly one `Unmapped`
  entry with `severity: Error`, and `es import roboverse` on it exits 1.
- `task_hash()` / `observation_hash()` are stable across repeated `convert()` calls on the
  same input (no `HashMap`, no wall-clock, no float non-determinism — `BTreeMap`/`BTreeSet`
  only).
- `es import roboverse <task.json> --out <dir>` writes `task.toml`, `observation.toml`,
  `provenance.json`, prints warnings/unmapped items and both hashes, and exits 1 when any
  `unmapped` entry is `severity: Error`.

## forbidden

- `es-usd`, `es-physics-core`, `es-script`, `es-ir`, `es-env`, `es-gpu`, `es-eval` — owned by
  other in-flight packets.
- Any other file under `crates/es/src/cmd/import.rs`'s neighborhood beyond the minimal
  `roboverse` subcommand insertion (a sibling packet owns the `usd` variant in the same file).
- No new crate dependencies (no `serde_yaml`, no PyO3 call-out): YAML/native-Python configs
  are a documented one-line export step, not this crate's problem.
- No pickle-based trajectory reading (INV-16): only the JSON task config is converted here.
