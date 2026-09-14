# M5 V0 — SO-101 cube-into-bin scene and the four IR documents

Design note: `docs/design/visible-learning.md` sections 4 (asset policy) and 5.4 (success predicate);
read section 2.3 first — `<include>` is rejected and a `type="mesh"` geom cannot reach MuJoCo or the
renderer, which is why this packet vendors a primitives-only derivative. Independent of V0b and V4;
blocks V1.

## context

```
tests/fixtures/mjcf/so101_pick_place.xml
tests/fixtures/mjcf/so101_pick_place.LICENSE
tests/fixtures/mjcf/so101_pick_place.PROVENANCE.json
tests/fixtures/visible-learning/task.toml
tests/fixtures/visible-learning/observation.toml
tests/fixtures/visible-learning/learning.toml
tests/fixtures/visible-learning/deployment.toml
crates/es-assets/tests/so101_provenance.rs
crates/es-physics-backend/tests/so101_scene.rs
crates/es/tests/cli.rs
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V0-scene-task.md
docs/packets/M5/V0-scene-task.ko.md
```

Notes: this packet is **fixtures and tests only** — it adds no line to any `src/`, so
`cargo xtask context-budget` cannot move. `crates/es/tests/cli.rs` gains one test that compiles the four
documents through the existing `es task compile`. No `Cargo.toml` changes: the provenance test uses
`std::process::Command` with the system `curl`, not a new HTTP dependency.

## spec

- §1.4: the fixture is judged by an executable cross-check against the upstream model, not by review.
- §4.3, §17.2: a scene item a backend cannot map is `Unsupported` **by name**; the derivative exists so
  that nothing in it is unmappable, and the test proves the whole file loads rather than warns.
- §5.1: Task IR declares `ObservationSpec` and owns no preprocessing and no neural net; Observation IR
  owns the image chain; Learning IR owns the policy. Four separate documents, four hashes.
- §6.3: cube-pose variation is a Task IR `Randomization` node over the cube free joint's `qpos`. Mass,
  friction and actuator-gain targets are **not** declared: `es-env` records the draw but never pushes it
  into the backend (`crates/es-env/src/randomize.rs:24-26`), so declaring one would put a number in the
  episode record that never reached the physics.
- §7.2, INV-14: the `ImageSpec` is declared once, in the Observation IR, at the size the renderer will
  produce. No `Resize` or `Crop` node in this packet, so no intrinsics transform is owed.
- §9.2–§9.4: the Deployment IR carries the joint limits, velocity and rate limits, the execution mode and
  the fallback; V3 tightens them for the demo suite. The envelope is a document, never a switch (INV-12).
- §18.1: the control rate is an integer `TickRate`; nothing in the fixtures is a float duration.
- §25.3: every fixture document carries its schema version.
- §28.3, §28.7 gate 7: this is the scene the gate has been missing since M1.

## oracle

```
cargo fmt --check
cargo clippy -p es-assets -p es-physics-backend --all-targets -- -D warnings
cargo test -p es-assets --test so101_provenance
cargo test -p es-physics-backend --test so101_scene
cargo test -p es --test cli visible_learning
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference — the upstream cross-check, network or a cached copy:

```
ES_MENAGERIE_CACHE=$HOME/cache/menagerie cargo test -p es-assets --test so101_provenance -- --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es-physics-backend --test so101_scene -- --nocapture
```

`so101_provenance.rs` reads `so101_pick_place.PROVENANCE.json` for
`{ repo, path, commit, blake3_so101_xml, derivation }`, obtains upstream `so101.xml` from
`$ES_MENAGERIE_CACHE/<commit>/so101.xml` if present, otherwise
`https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/<commit>/robotstudio_so101/so101.xml`
via `curl -fsSL` into `target/`, and verifies its blake3 against the manifest **before parsing it**
(§25.1: bytes off the network are untrusted). The pin is commit
`ac6b2b09983786f3036cab1000221017fa2193b4`. No cache and no network -> `SKIP so101_provenance: <why>`;
when it ran, `RAN so101_provenance`. A blake3 mismatch is a **failure**, never a skip: it means upstream
moved under the pin.

`so101_scene.rs` renders the fixture through `es_assets::parse_mjcf` -> `mjcf_out::scene_to_mjcf` ->
`MuJoCoCpuBackend::load` and asserts the model shape. No `mujoco` -> `SKIP so101_scene: <why>`.

Fixture parameters, fixed here so V1 and V3 can cite them: image `96x96` `Rgb8` from one fixed overhead
camera; control rate 50 Hz; chunk horizon `H = 16`; `NJ = 6`; `max_episode_steps = 400`. `NJ = 6` and
`H = 16` are already in `es eval run`'s and `es loop collect`'s const-generic dispatch tables
(`crates/es/src/cmd/eval.rs:237-256`, `crates/es/src/cmd/loop.rs:124-144`), so no table row is added.

`tests/so101_provenance.rs`:

- `derivative_has_the_upstream_kinematics` — parse both files; the six joints appear in the same order
  with byte-equal names, and their `axis`, `range`, `damping` and `armature` are exactly equal as `f64`.
- `derivative_has_the_upstream_body_frames` — every upstream body of the chain appears with an exactly
  equal `pos` and an exactly equal orientation after `es_assets::mjcf::orient` normalization.
- `derivative_has_the_upstream_inertials` — equal mass and diagonal inertia per body.
- `derivative_drops_only_visual_mesh_geoms` — every geom the derivative omits is `type="mesh"` in the
  upstream file, and every primitive collision geom upstream is present here.
- `derivative_parses_with_no_warnings` — `parse_mjcf(...).warnings` is empty: the demo scene contains no
  element `SceneDesc` drops (`crates/es-assets/src/mjcf/mod.rs:208-218`).
- `derivative_has_no_include` — the file text contains no `<include`, since both root and body `<include>`
  are `MjcfError::Include` (`mod.rs:247-251`, `:569-573`).
- `link_lengths_are_derived_not_transcribed` — the `Links` values V1's IK consumes are computed from the
  parsed `SceneDesc` and written to `target/so101_links.json`; the test asserts the derivation is a pure
  function of the parse, so no length is ever hand-copied into Rust source.

`tests/so101_scene.rs`:

- `the_demo_scene_loads_in_mujoco` — `nq = 13` (6 arm + 1 jaw excluded by the equality-free derivative,
  plus a 7-dof free joint for the cube; the exact number is asserted against `ModelInfo`, not assumed
  here), `nu = 6`, and every actuator id in `ModelInfo.actuator` resolves.
- `every_scene_item_is_mappable` — `scene_to_mjcf` returns `Ok`: no geom, actuator or sensor in the
  fixture is `PhysicsError::Unsupported`.
- `the_cube_free_joint_is_randomizable` — `RandomizationPlan::compile` of the Task IR resolves every
  declared target to `Target::Qpos`, and two different `(seed, episode)` pairs give two different cube
  poses while the same pair gives the same one.

`crates/es/tests/cli.rs`, one test `visible_learning_documents_compile`: `es task compile` over the four
fixture documents exits 0, and the four hashes it prints are stable across two runs.

## acceptance

The four documents, as authored (no new Rust type is introduced by this packet):

```toml
# task.toml      §6: scene ref, ObservationSpec declaration, reward, Terminate{Success|Failure|Timeout},
#                Randomization over the cube free joint's qpos, ResetState, max_episode_steps = 400
# observation.toml §7: ImageInput(96x96 Rgb8, one camera) + a state chain over qpos/qvel; no Resize,
#                no Crop, no MultiViewPack (rejected by crates/es-compile/src/plan.rs:545)
# learning.toml  §8: VisionEncoder{ResNet18} -> StateEncoder -> Fusion -> TemporalEncoder{Transformer}
#                -> PolicyHead{Regression} -> ActionChunker, plus Normalizer nodes; ArchKind::Act
# deployment.toml §9: NJ = 6, H = 16, 50 Hz, joint/velocity/rate limits, execution mode, fallback
```

- The success predicate is a `Terminate { kind: Success }` cone over the cube free joint's `qpos` and
  `qvel` only — cube centroid inside the bin AABB and `|v|` under the bound (design note section 5.4).
  There is no counter node and no new `TaskNode` variant.
- `so101_pick_place.PROVENANCE.json` names `repo`, `path`, `commit`, `blake3_so101_xml`, the upstream
  licence (`Apache-2.0`) and the derivation rules in prose. `so101_pick_place.LICENSE` is the upstream
  Apache-2.0 text, unmodified.
- The fixture is one flat file: no `<include>`, no `<keyframe>`, no `<equality>`, no mesh geom, no
  `meshdir`.
- 0 lines added to any `src/` directory.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` — no new node, no new `PerturbationKind`, no schema change. `es-ir`
  has 53 lines of §1.5 budget left (design note section 2.10).
- `crates/es-env/src/randomize.rs`, `crates/es-eval/src/perturb.rs` — declaring a mass, friction or gain
  randomization, or making `Target::Scale` reach the backend, is a `PhysicsBackend` parameter API and a
  different packet.
- `crates/es-render`, `crates/es-env/src/expert.rs`, `crates/es-policy`, `crates/es/src/cmd/*.rs` — V0b,
  V1, V2.
- Vendoring `assets/**` (17,230,580 B of STL), `so101.png`, `scene.xml` or `scene_box.xml`.
- Adding an HTTP client crate, or making any test on the PR tier depend on the network.
- Editing goldens, or relaxing an equality in `so101_provenance.rs` to a tolerance to make the derivative
  pass.
