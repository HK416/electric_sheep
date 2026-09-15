# M6 B1 — Go1 scene and the four IR documents

Design note: `docs/design/quadruped-track.md` sections 1–3. Pinned upstream facts:
`docs/api-notes/mujoco-playground-quadruped.md`. The first packet of the quadruped track and
LOCAL-ONLY: no training, no GPU, no server. Blocks the import packet (brax checkpoint →
`safetensors` → Learning IR weights).

## context

```
tests/fixtures/mjcf/go1_primitives.xml
tests/fixtures/mjcf/go1_primitives.LICENSE
tests/fixtures/mjcf/go1_primitives.PROVENANCE.json
tests/fixtures/quadruped/task.toml
tests/fixtures/quadruped/observation.toml
tests/fixtures/quadruped/learning.toml
tests/fixtures/quadruped/deployment.toml
crates/es-assets/tests/go1_provenance.rs
crates/es-assets/src/scene.rs
crates/es-assets/src/mjcf/mod.rs
crates/es-physics-backend/tests/go1_step.rs
crates/es-physics-backend/src/mjcf_out.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/quadruped-track.md
docs/design/quadruped-track.ko.md
docs/packets/M6/B1-go1-scene-and-documents.md
docs/packets/M6/B1-go1-scene-and-documents.ko.md
```

Notes: the four `src/` files change by a few lines each and only for one reason — the
`<option>` block of a Playground-trained model has to survive `SceneDesc` (see **spec** below).
`crates/es/src/cmd/eval.rs` gains exactly one row, `(12, 1)`, in the dispatch table whose own
comment invites it. No `Cargo.toml` change: the provenance test uses the system `curl` through
`std::process::Command`, as `so101_provenance.rs` does, not a new HTTP dependency.

## spec

- §1.4: the scene is judged by an executable cross-check against the pinned upstream files, not
  by review; the physics oracle is MuJoCo (CPU), and it prints `SKIP <why>` rather than passing
  when no interpreter carries `mujoco`.
- §1.9, §8: the policy is trained **externally** by `mujoco_playground` and imported as weights.
  A native training stack was never in scope; the Learning IR is the import surface.
- §17.2: a backend declares what it cannot map instead of guessing. `mjcf_out` writes the whole
  `<option>`, and "the whole `<option>`" now includes `ls_iterations` and `<flag eulerdamp>` —
  a policy trained under `ls_iterations = 5` with `eulerdamp` disabled and stepped at MuJoCo's
  defaults (50, enabled) is being stepped in physics it never saw. That is the one `src/` change
  this packet makes and it is the reason the packet exists at layer 4.
- §4.3, §3.5: `MuJoCoCpuBackend` declares `DeterminismTier::PhysicsMeaning`, not tier 1 — an
  external backend never declares tier 1 — so the parity oracle's tolerance is a declared
  physics-meaning tolerance, not bitwise equality.
- §5.1: Task IR declares the `ObservationSpec` and owns no preprocessing and no neural net;
  Observation IR implements the channel; Learning IR owns the policy. Four documents, four
  hashes, one `es ir check`.
- §7.4, §7.5: the 48-wide channel is declared once, in Task IR, and implemented in Observation
  IR; `temporal.window = { n_steps = 1 }` is layer 2 of the time model and is upstream's
  `history_len = 1` said out loud (XIR-011 checks it against `observation_window`).
- §8.3, §8.4: `StateEncoder{Mlp}` → `PolicyHead{Regression}` → `ActionChunker` →
  `Normalizer{Inverse}`; `replanning_hz = 50` is an integer divisor of the 50 Hz control rate
  (XIR-023).
- §9.2–§9.4, INV-12, INV-13: the Deployment IR carries the joint, velocity, acceleration, torque
  and action-rate limits, the watchdogs and the fallback. Every limit is at least as wide as
  what the policy trained under — widened, never disabled — and `SafetyPlane::validate`'s
  signature is untouched.
- §12.4: wall-clock is reported per 1,000 physics steps as an observation. No `step/s` claim.
- §18.1: control and inference rates are integer `TickRate`s; the deadlines are whole multiples
  of the control period.
- §25.1: the provenance test verifies blake3 **before** parsing bytes off the network.
- §28.9: the milestone this track belongs to.

## oracle

```
cargo fmt --check
cargo clippy -p es-assets -p es-physics-backend -p es --all-targets -- -D warnings
cargo test -p es-assets --test go1_provenance
cargo test -p es-physics-backend --test go1_step
cargo test -p es --test cli quadruped
cargo run -p es -- ir check tests/fixtures/quadruped/task.toml \
    tests/fixtures/quadruped/observation.toml \
    tests/fixtures/quadruped/learning.toml \
    tests/fixtures/quadruped/deployment.toml
cargo xtask ci
```

Reference — the halves that need something this box may not have:

```
# the upstream cross-check: network, or a cached copy
ES_PLAYGROUND_CACHE=$HOME/cache/playground \
  cargo test -p es-assets --test go1_provenance -- --nocapture

# the physics oracle: a Python with mujoco. On the GPU server that is ~/venvs/es (mujoco 3.13):
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-physics-backend --test go1_step -- --ignored --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es --test cli quadruped -- --nocapture
```

`go1_provenance.rs` reads `go1_primitives.PROVENANCE.json` for
`{ repo, path, path_scene, commit, blake3_go1_mjx_feetonly_xml, blake3_scene_flat_terrain_xml,
menagerie_commit, license, derivation }`, obtains both upstream files from
`$ES_PLAYGROUND_CACHE/<commit>/<file>` if present, otherwise from
`raw.githubusercontent.com/google-deepmind/mujoco_playground/<commit>/.../go1/xmls/<file>` via
`curl -fsSL` into `target/`, and verifies their blake3 against the manifest **before parsing
them**. The pin is `124a73fa3303f75a62f8fe04d329b829ed0ebdfb` (release v0.2.0). No cache and no
network → `SKIP go1_provenance: <why>`; having run, `RAN go1_provenance (<file>)`. A blake3
mismatch is a **failure**, never a skip. Two edits are made to upstream's text before parsing,
both importer limitations and neither touching anything asserted on: `<mesh class="go1" ...>` →
`<mesh ...>` (the importer reads `<asset>` before `<default>`) and `<site>` lines are dropped
(the importer wants a 3-vector site `size`, upstream writes MJCF's scalar shorthand).

What the seven provenance tests enumerate:

- `derivative_has_the_upstream_joints_and_actuators` — the twelve hinges in upstream's
  declaration order with byte-equal `axis`, `range`, `damping`, `armature` and `frictionloss`,
  and the twelve `<position>` actuators with equal gains, `ctrlrange`, `forcerange`, `gear` and
  driven joint.
- `derivative_has_the_upstream_body_frames_and_inertials` — trunk plus
  `{FR,FL,RR,RL}_{hip,thigh,calf}`: byte-equal `pos`, byte-equal orientation after the shared
  `mjcf::orient` normalization, the same parent, the same mass and inertia diagonal.
- `derivative_substitutes_only_the_visual_mesh_geoms` — every upstream *primitive* geom is
  present pose-for-pose; everything absent is a mesh and nothing else; **one added primitive per
  dropped mesh in the same body**, so it is a substitution, not a deletion; exactly 13
  substitutions; and the derivative carries no mesh and no mesh asset anywhere.
- `derivative_has_the_upstream_option_block` — `scene.options` equals upstream's, and spelled
  out: `timestep = 0.004`, `iterations = 1`, `ls_iterations = 5`, `eulerdamp` disabled,
  `integrator = Euler`, `cone = pyramidal`.
- `derivative_carries_the_home_keyframe` — the derivative's `home` `qpos` is the 19 pinned
  numbers, and the *upstream scene file's* own `home` is the same 19, so the action's zero-point
  is checked against upstream's bytes rather than trusted; the inlined floor is upstream's too.
- `derivative_parses_with_only_the_enumerated_warnings` — 69 importer warnings, every one of
  them a `<keyframe>`, a geom `priority`/`group`, a material `rgba`, a camera `mode` or a
  `<position inheritrange>`, i.e. exactly the list the manifest enumerates; and the file text
  contains no `<include`, no `meshdir`, no `type="mesh"`, no `<sensor` and no `.stl`.
- `joint_limits_are_derived_not_transcribed` — every hinge has a finite ordered `range` and
  every actuator a two-sided `forcerange` (these are what `deployment.toml` copies), and the
  scene is one free joint plus twelve hinges — the floating base that makes this fixture
  different from every fixture before it.

`go1_step.rs`:

- `the_emitted_mjcf_keeps_the_playground_option_block` — **no Python, runs in PR CI.** The
  re-emitted MJCF carries `timestep="0.004" integrator="Euler" iterations="1"
  ls_iterations="5" cone="pyramidal"` and `<flag eulerdamp="disable"/>`, carries no mesh and no
  `<include>`, and still names `trunk`, `FR_calf_joint`, `RL_hip` and `floor`.
- `go1_stands_from_home_and_agrees_with_mujoco_directly` — `#[ignore]`, needs `ES_PYTHON`.
  `nq/nv/nu = 19/18/12`; reset to `home`, `ctrl` = `home`'s own (a zero action), 250 control
  ticks × 5 substeps = 1250 physics steps; no `StepReport` failure, the state stays finite and
  the trunk stays above 0.20 m. Then the same XML through `mujoco` directly: MuJoCo must report
  `iterations = 1`, `ls_iterations = 5`, `eulerdamp = false`, `timestep = 0.004` — i.e. it read
  our fixture's solver settings and not its own defaults — the reference run must stand too, and
  the worst per-coordinate `|Δqpos|` must be under 1e-3. Wall-clock per 1,000 physics steps is
  printed for both paths.

`crates/es/tests/cli.rs`, five tests matching `quadruped`:

- `regenerate_quadruped_documents` (`#[ignore]`) — the generator. Builds all four documents from
  the parsed scene and writes them, so no hash, joint limit, torque limit or reset pose in the
  fixtures is ever typed by hand.
- `quadruped_documents_validate_and_cross_check` — each of the four `validate()`s clean,
  `cross::check` is empty, the `es ir check` binary exits 0 and prints the hash chain, and the
  documents say what the api-note says: 48 in, 12 out, `history_len = 1`, 50 Hz, chunk 1.
- `quadruped_documents_are_what_the_generator_produces` — the committed bytes equal the
  generator's output, so a hand edit is a failing test.
- `quadruped_bundle_runs_100_ticks_through_the_safety_plane` — the four documents pack into a
  `policy.esb` that re-opens, and 100 control ticks go through the real `SafetyPlane::<12, 1>`
  built from `deployment.toml`, twice: an untrained policy's `tanh`-bounded action is clamped on
  every tick and never latches a fallback and never leaves the position limits, and a
  physically-followable command (0.02 rad/tick) reaches the actuator unchanged on every tick.
  The second pass is what stops a constant-clamping envelope from passing the first.
- `quadruped_eval_run_names_the_observation_gap` — `es eval run` over a one-suite Evaluation IR
  either skips with the documented exit 3 (no `mujoco`/`torch`), or refuses **by name** on the
  known observation gap without writing a report, or — the day the gap closes — exits 0. It
  never panics and never fakes a run (§1.4).

## acceptance

- `tests/fixtures/mjcf/go1_primitives.xml` parses, loads through the MuJoCo CPU backend, and
  differs from upstream only by the substitutions `go1_primitives.PROVENANCE.json` enumerates —
  proved by the seven tests above, not asserted here.
- `<option timestep / integrator / iterations / ls_iterations>` and `<flag eulerdamp>` survive
  `parse_mjcf` → `scene_to_mjcf`. Before this packet, the last two did not.
- The four documents in `tests/fixtures/quadruped/` validate individually and cross-check
  together, and `es ir check` prints a hash chain over them.
- 100 control ticks of an untrained policy pass through the Safety Plane at 12 joints without a
  panic, a fallback or an escape from the envelope.
- Every number in the four documents is derived from `go1_primitives.xml` or from the api-note,
  and the generator test proves the committed files are its output.
- Whatever the oracles could **not** run on this box is stated with its exact server command
  (see **oracle** above), never silently skipped.

## forbidden

- `crates/es-safety` — no change of any kind. The envelope is widened in `deployment.toml`;
  `SafetyPlane::validate`'s signature is untouched (INV-13) and nothing disables the plane
  (INV-12).
- `crates/es-eval`, `crates/es-env` — fixing `joint_state`'s leading-`NJ` convention for a
  floating base, or teaching `input_sources` to capture base velocity, gyro or `qvel`, is the
  packet that unblocks `es eval run` here (design note sections 3.2, 3.3). Not this one.
- `crates/es-ir`, `crates/es-ir-types` — no new node, no new `StateEncoderKind`, no activation
  parameter, no `tanh`. Design note section 3.6 lists those as questions for a human; INV-17
  caps the extension points at seven and this packet adds none.
- `crates/es-policy/src/lower/torch.rs` — swish-vs-ReLU is the import packet's first question
  (design note section 3.4), and changing an activation would move every lowered policy's hash.
- A `priority` field on `Geom`, or any other widening of the MJCF importer beyond the two
  `<option>` values named under **spec**.
- Mesh support in the loader, vendoring the 5 Unitree STLs, or relaxing an equality in
  `go1_provenance.rs` to a tolerance to make the derivative pass.
- Training anything, touching the GPU server, or editing goldens.
- `docs/design/visible-learning.md` and `docs/ARCHITECTURE*.md`.
