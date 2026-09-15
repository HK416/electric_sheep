# M6 B1b — geom `priority` and `<position inheritrange>` through the scene model

Design note: `docs/design/quadruped-track.md` section 3.1. Follow-up to M6/B1, which shipped the
Go1 scene and named these two as the known drops. It closes the physics-parity gap B1's own
oracle then measured: `go1_stands_from_home_and_agrees_with_mujoco_directly` failed at worst
per-coordinate **|Δqpos| = 7.574e-3** against a 1e-3 threshold.

## context

```
crates/es-assets/src/scene.rs
crates/es-assets/src/mjcf/mod.rs
crates/es-assets/src/mjcf/attrs.rs
crates/es-assets/src/mjcf/elements.rs
crates/es-assets/src/gltf.rs
crates/es-assets/src/urdf.rs
crates/es-assets/tests/go1_provenance.rs
crates/es-physics-backend/src/mjcf_out.rs
crates/es-physics-core/src/usd.rs
crates/es-render/src/cornell.rs
crates/es-render/src/scene.rs
crates/es-splat/tests/splat.rs
tests/fixtures/quadruped/{task,observation,learning,deployment}.toml
tests/fixtures/visible-learning/{task,observation,evaluation}.toml
docs/design/quadruped-track.md
docs/design/quadruped-track.ko.md
docs/packets/M6/B1b-geom-priority.md
docs/packets/M6/B1b-geom-priority.ko.md
```

The `gltf.rs` / `urdf.rs` / `usd.rs` / `cornell.rs` / `scene.rs` / `splat.rs` edits are one line
each — `priority: 0`, MuJoCo's default — because `Geom` is a plain struct with no `Default`.
No `Cargo.toml` change, no new trait (INV-17), no new file outside `docs/`.

## the measurement this packet exists for

An ablation on **direct MuJoCo alone**, so the number is not about our code: from
`tests/fixtures/mjcf/go1_primitives.xml`, remove one attribute at a time, step 1,250 physics
steps from `home` with the action at zero, and compare `qpos` against the unmodified file.

| removed | worst per-coordinate \|Δqpos\| |
| --- | --- |
| floor `priority="1"` | **7.574e-3** |
| foot `solimp` | 0 |
| `<position inheritrange>` | 0 |

7.574e-3 is, to the last digit, the disagreement B1's oracle measured between our backend and
direct MuJoCo — so `priority` was the whole gap. MuJoCo's contact-parameter rule: the geom with
the higher `priority` decides `friction`, `condim`, `solref` and `solimp` outright, and equal
priorities mix them elementwise by `max`. The floor's `priority="1"` is upstream saying "every
foot contact uses my 0.6". Dropped, the foot's 0.4 wins the `max` and the stance settles
somewhere else.

## spec

- §5.3: `scene_hash` is "sensitive to every field that can change simulation behaviour", so
  `priority` is encoded **unconditionally**, next to `condim`, not conditionally on being
  non-zero. A hash that agrees for two scenes MuJoCo steps differently is the bug §5.3 forbids.
- §17.2: a backend declares what it cannot map rather than guessing. `mjcf_out` now emits
  `priority` (omitted at the default 0, so the emitted XML keeps upstream's shape).
- §1.4: the judge is MuJoCo (CPU) through `go1_step.rs`, not review.
- §4.3, §3.5: the tolerance stays the backend's declared `DeterminismTier::PhysicsMeaning`
  tolerance; an external backend never declares tier 1, whatever the measured number is.
- §28.9: the milestone.

`SCENE_TAG` stays `es.scene.v1`. It is a domain separator between hash namespaces, not a schema
version: M6/B1 added `ls_iterations` and `eulerdamp` to the same encoding without bumping it, and
this packet follows that. Every `scene_hash` moves anyway, which is what section **hashes** below
is about.

## what `inheritrange` carries

Carried, and it cost ten lines. `<position inheritrange="f">` is a **compile-time rewrite** in
MuJoCo: `ctrlrange` becomes the transmission target's own range scaled by `f` about its midpoint.
The importer already resolves the target joint before it builds the `Actuator`, and the joints
are parsed before `<actuator>` (`ROOT_ORDER`), so `Actuator::ctrl_range` can hold the resolved
pair and no field is added to the scene model at all. Applied for `<position>` only — upstream
allows it on `intvelocity` too, which we do not model, and there it stays an enumerated warning.

Dropping it was not neutral even though the ablation scores it at 0 here: with no `ctrlrange`
the actuator is unclamped, so the policy's position targets are held to a wider range than the
one it trained under. A zero action from `home` never reaches the limit, which is exactly why a
step oracle cannot see it and why it is fixed by argument rather than by measurement.

## hashes

Adding a field to the canonical encoding moves every `scene_hash`, hence every `task_hash` and
everything downstream. Regenerated **only** through their sanctioned generators — no hash in this
repository is ever typed or edited by hand:

```
cargo test -p es --test cli -- --ignored regenerate_quadruped_documents
cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents
```

which rewrote `tests/fixtures/quadruped/{task,observation,learning,deployment}.toml` and
`tests/fixtures/visible-learning/{task,observation,evaluation}.toml` (the last one pins
`task = "<task_hash>"`). No file under `tests/golden/` embeds a scene hash, so no golden is
touched and `cargo xtask verify-goldens` passes unchanged.

## oracle

```
cargo xtask ci
cargo test -p es-assets --test go1_provenance
cargo test -p es-physics-backend --lib mjcf_out
```

On the GPU server (the halves that need `mujoco` / `torch`):

```
ES_PYTHON=$HOME/venvs/es/bin/python \
  cargo test -p es-physics-backend --test go1_step -- --ignored --nocapture
ES_PYTHON=$HOME/venvs/es/bin/python cargo test -p es --test cli --features render -- \
  quadruped dataset_bake expert_passes_the_evaluation_harness collection_and_evaluation
```

## acceptance

- `go1_stands_from_home_and_agrees_with_mujoco_directly` prints
  `RAN go1_stands_from_home_and_agrees_with_mujoco_directly` with worst |Δqpos| under 1e-3.
  **Measured: 8.882e-16** over 1,250 physics steps — the last bit of a double, not merely inside
  the tolerance. Before this packet: 7.574e-3.
- `derivative_carries_the_home_keyframe` compares the floor's `priority="1"` against upstream's
  own bytes, attribute by attribute rather than as one ordered substring, and asserts the value
  reached `SceneDesc`.
- `derivative_parses_with_only_the_enumerated_warnings` no longer enumerates `` `priority` `` or
  `` `inheritrange` ``: they are carried, so they must not warn.
- `actuators_and_sensors_round_trip` round-trips `priority="2"` and an `inheritrange="0.5"`
  position actuator resolved to `(-0.5, 0.5)` from the joint's own `(-1, 1)`.
- The regenerated demo fixtures are proven against MuJoCo on the server by the `es` CLI tests
  above.

## forbidden

- Hand-editing any hash, in a fixture or a golden. Generators only.
- `docs/ARCHITECTURE*.md`, `docs/design/visible-learning.md` — not this packet's to change.
- `crates/es-safety` — untouched. `priority` widens nothing and disables nothing (INV-12,
  INV-13).
- Every other MJCF attribute the importer still drops. `group`, `<keyframe>`, material `rgba`
  and camera `mode` stay enumerated warnings; this packet carries the two the oracle named.
