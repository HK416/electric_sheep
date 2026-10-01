# M17 plan G — scene authoring (S4)

> Packets for `docs/design/scene-authoring.md` (the owner's decisions are its section 1; the
> defaults of its section 9 stand until the owner changes them). The owner started the plan on
> 2026-10-01 ("G1, G2 시작"). Each task is a §1.2 packet: its **Files** block is its `context` for
> `cargo xtask check-scope`.

**Goal:** a person builds a new task from an empty project in the editor — scene, sentences,
generated documents — and trains it, without writing MJCF, TOML IR or Rust.

## Global Constraints

- Everything in `docs/packets/M16/plan-h.md`'s Global Constraints holds (worktree + `--ff-only`,
  one `CARGO_TARGET_DIR` per agent, goldens by generator only, never push, no remote server, no
  learning run inside an agent packet, spec first — `ARCHITECTURE.ko.md` and `.md` in one commit —
  Safety Plane untouched).
- **Every committed document, golden and hash stays as it is.** A scene read from `.esscene`
  hashes by its `SceneDesc` exactly as one read from MJCF.
- Layers (§4.2): readers, writers and include expansion in `es-assets` (2); the task-spec compiler
  in `es-script` (11); CLI in `es`; scene model in `es-editor-scene` (12, G5); widgets in `es-editor`
  (13). No new extension point (INV-17).

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | G1 the scene document · G2 the full MJCF exporter | — |
| 2 | G3a task spec → Task IR · G4 `es scene simulate` and ①'s physics preview (`es render` landed as H8/H9) | G1 |
| 2b | G3b the other documents + `es project generate` · G3c three-axis relations (`Slice`, `Concat`, `Reduce`, `GetBodyVelocity` lowered) | G3a |
| 3 | G5 editor scene model · G6 viewport picking and gizmos | G1, G4, H8 |
| 4 | G7 Add (primitives, meshes, robots, cameras, lights, regions) · G8 sentence editor | G5, G3 |
| 5 | G9 empty-project card, save as template, ② teacher for generated tasks | G3, G8 |
| 6 | GV end to end (orchestrator), review | all |

### Task G1: the scene document (`*.esscene`)

**Files:** `crates/es-assets/src/esscene/**` (new), `crates/es-assets/src/lib.rs` (the module line),
`crates/es-assets/tests/esscene*.rs`, `crates/es-tools/src/backend.rs` (`load_scene` reads
`.esscene`), new fixtures under `tests/fixtures/esscene/`, `docs/ARCHITECTURE.ko.md` / `.md` §14.3
and §6 (`SceneRef`), `docs/design/scene-authoring.md` (+ko) if the schema settles a detail.

- A document model (`EsScene`, serde, TOML; `kind = "scene"`, `schema = 1`) with the design's
  section 3: `[physics]`, `[[include]]` (MJCF / URDF / glTF / USD file, pose, optional name
  prefix — absent = the file's own names, `[include.set]` overrides by the asset's names),
  `[[body]]` (tree by `parent`, pose as pos + xyzw quat, joint fixed / free / hinge / slide / ball
  with axis, range, damping, armature; `[[body.geom]]` shapes box / sphere / capsule / cylinder /
  ellipsoid / plane / mesh with mass or density, friction, condim, contype / conaffinity, rgba or
  material), `[[material]]` / textures (HT1/HT2's fields), `[[camera]]`, `[[light]]` (as the
  renderer takes lights today: the `_light` emissive-geom convention or what `SceneDesc` holds),
  `[[region]]` (a site). Read **and write** (the editor round-trips the document, not the
  expansion).
- `expand(doc, dir) -> SceneDesc`: includes by the existing readers, then the document's own
  entities, in document order; assets resolved and hashed by content.
- **Oracles:** (1) an `.esscene` that includes the SO-101 robot part of
  `tests/fixtures/mjcf/so101_pick_place.xml` (split into a robot file — a fixture) and writes the
  table, cube, bin, cameras and light natively expands to a `SceneDesc` **equal** to
  `parse_mjcf(so101_pick_place.xml)` and the same `scene_hash` / `asset_hash`; the same for the
  Shadow Hand scene (hand included, cube / goal / cameras / light native); (2) read ∘ write is the
  identity on the documents (and a property test over generated documents); (3) every refusal
  names the field (unknown key, dangling parent, unknown material, include not found);
  (4) `load_scene("x.esscene")` gives what `expand` gives; committed hashes unmoved.

### Task G2: the full-fidelity MJCF exporter

**Files:** `crates/es-assets/src/mjcf/write.rs` (new) + its module line, `crates/es-assets/tests/mjcf_write*.rs`,
`crates/es/src/cmd/scene.rs` (new: `es scene export`) + its registration and help, its CLI test.

- `write_mjcf(scene: &SceneDesc, assets_out: &Path) -> String`: everything `SceneDesc` carries —
  bodies, joints, geoms (incl. meshes, written as files beside the XML), sites, cameras, lights /
  emitters, materials and textures (HT1/HT2), actuators, sensors, tendons, contact pairs /
  excludes, gravcomp, options — so that **`parse_mjcf(write_mjcf(s)) == s`** (and the hashes are
  equal). Unlike `es-physics-backend`'s `mjcf_out` (which drops what carries no dynamics) this is
  the export a person opens in MuJoCo's viewer.
- `es scene export <scene> --mjcf <out.xml>` for any scene `load_scene` reads (after G1 lands it
  reads `.esscene` too; until then MJCF / URDF).
- **Oracles:** (1) round trip equality and equal `scene_hash` / `asset_hash` for every committed
  scene under `tests/fixtures/mjcf/**` that parses (SO-101 and its views, Shadow Hand, textured,
  the rest), plus a property test over generated `SceneDesc`s; (2) MuJoCo (Python, skips without
  `ES_PYTHON`) loading the export steps bit-identically to MuJoCo loading the original for the
  SO-101 and Shadow Hand scenes (H1's parity method); (3) committed goldens unmoved.

### Task G3a: the task specification compiles to the Task IR

**Files:** `crates/es-script/src/spec/**` (new: the `*.estask` model and its compiler),
`crates/es-script/src/lib.rs` (the module line), `crates/es-script/Cargo.toml` (`es-assets`,
`toml` if needed), `crates/es-script/tests/estask*.rs`, new fixtures `tests/fixtures/estask/`,
`docs/design/scene-authoring.md` (+ko) §4 if the schema settles a detail.

- The `*.estask` model (serde, TOML, `kind = "task-spec"`, `schema = 1`) of the design's §4:
  `scene`, `robot` (an include), `control_hz`, `[success]` / `[failure]` clause lists, `timeout_s`,
  `[start]` (placements, 🎲 items, strength), `[observe]` (cameras with resolution and render,
  state channels, privileged channels), `[reward]` (levels, shaping). Read and write; unknown keys
  refused by name; every relation of §4.1 except `touches` (refused by name until `GetContact`
  lowers).
- `compile_task(spec, scene_dir) -> TaskIr`: the scene through G1's `load_scene` path, names
  resolved to `StableId`s (a missing name refused with the name), each clause to the nodes of
  §4.1, rewards per §4.1, reset and randomization, the `ObservationSpec` channels and their
  sensors (`render` as the Task IR declares it), the `ActionSpec` from the robot's actuators.
- **Oracles:** (1) a spec for the Shadow Hand repose task compiles to a Task IR whose **`task_hash`
  equals `tests/fixtures/shadow-hand/task-repose.toml`'s** (the committed document is the reference;
  the compiler takes over what `crates/es/tests/shadow_hand.rs` builds by hand); (2) the same for
  the SO-101 three-view task `tests/fixtures/visible-learning/task-views.toml`; if a committed
  document carries a construction the vocabulary cannot express, extend the vocabulary
  minimally and say what (do not change the committed document); (3) each relation on scripted
  states: true / false and its reward (as H2's `the_task_scores_scripted_states`); (4) refusals
  name the clause and the field; (5) committed hashes unmoved.

### Task G3b: the other documents and `es project generate`

**Files:** `crates/es-script/src/spec/**`, `crates/es/src/cmd/project.rs` (new) + its
registration, tests, fixtures. After G3a.

- From the spec and the scene: Observation IRs (teacher state, student views), Learning IRs (the
  state MLP teacher, the three-view ACT student, `tanh` heads — H6's invariant), the Deployment IR
  (envelope from joint and control ranges), Evaluation IRs (held-out seeds, the suites that
  apply, nominal-only sibling), training recipes, the cycle; `es project generate --scene <s>
  --spec <t> --out <dir>` writes them. **Oracle:** the Shadow Hand and SO-101 views specs
  regenerate every committed document of their sets hash-for-hash (bodies byte-for-byte where the
  committed file was itself generated).

### Task G4: `es scene simulate` and ①'s physics preview

**Files:** `crates/es/src/cmd/scene.rs` (a `simulate` verb beside G2's `export`), its test,
`crates/es-editor-model/src/model/{scene_view,viewport}.rs` (the preview's decisions),
`crates/es-editor/src/**` (the button and playback), i18n tables.

- `es scene simulate <scene> --seconds S [--ctrl hold|zero] [--backend mujoco-cpu] --out
  <traj.estraj>`: the scene from its initial pose, every actuator held at its initial target
  (`hold`) or zero, stepped for S seconds, the trajectory written in the `.estraj` format the
  replay reads. No Task IR is needed (a scene alone).
- ①: a "물리 미리보기" button runs it (an argv, as every editor run) and plays the result back in
  the viewport with H8/H9's renderer and the replay's timeline; a plain-language status while it
  runs and if it fails (the mapping report's refusal named).
- **Oracles:** the CLI's trajectory equals stepping the same scene through `MuJoCoCpuBackend`
  directly (bitwise on `mujoco-cpu`); a cube dropped above a plane comes to rest on it; the
  view-model's decisions headless; a screenshot of the preview on the Shadow Hand project.

**G3a as merged (2026-10-01):** both specs compile to the committed `task_hash` (Shadow Hand
`e16b44a9…`, SO-101 views `33f55c29…`); `task_graph_hash` differs (it mixes node ids; the committed
numbering is generator history) — **the orchestrator accepts semantic `task_hash` equality as the
target for G3b too**. `inside` and `still` are one-axis because es-env's cone lowering has no
`Slice`, `Concat`, `Reduce` or `GetBodyVelocity` (G3c). The compiler calls the host's `cos` / `tan`
for the angle threshold, the camera focal length and the tilt bound (G3b settles it).

### Task G3c: three-axis relations

**Files:** `crates/es-env/src/plan.rs` (+ its tests), `crates/es-script/src/spec/compile.rs` (the
clause functions only), `crates/es-script/tests/estask_scripted.rs`, design note §4.1 (+ko).

- es-env lowers `Slice`, `Concat`, `Reduce` and `GetBodyVelocity` in its reward / termination
  cones, with the semantics `es-ir` defines (check `crates/es-ir/src/task.rs`'s output types and
  any CPU reference evaluator), so a relation can read a 3-vector.
- The vocabulary then offers `inside` a scene **region** (an `.esscene` `[[region]]` / MJCF site
  box, all three axes), `still` as the body's linear speed (3-D) and optionally angular, `above`
  / `below` another body, and `<body>.y` / `.z` subjects.
- **Oracles:** each lowered node against hand-computed values on scripted states (as G3a's
  `estask_scripted.rs`); every committed task document's evaluation unchanged (its `task_hash`
  and the reach / SO-101 / Shadow Hand reward goldens and bitwise RL tests); the two reference
  specs still compile to their committed `task_hash`.

**G3b as merged (2026-10-01):** `es project generate` regenerates both reference sets — the Shadow
Hand set and the SO-101 views / cam / MAD arms — IR document for IR document by semantic hash, the
recipes and cycles as equal structs; `visible-learning/deployment.toml` (a hand-tuned envelope) is
the one committed file the scene's ranges do not produce. The host `cos` / `tan` stay (a sin/cos
`tan` moves the 45° cameras' focal length by one ULP); musl's `tan` matched the host's bits at every
probed field of view, so G3d removes the host call without moving a hash.

### Task G3d: generated documents without the host libm

**Files:** `crates/es-math/src/approx*` (a `tan_f64` beside `sin_cos_f64`), `crates/es-script/src/spec/compile.rs`
(the three host calls), the design note §4.3 (+ko). Use `es_math::approx` for the angle threshold's
cosine, the camera focal length's tangent and the tilt bound; **oracle:** every committed
`task_hash` and every G3b hash unmoved, plus a sweep of `tan_f64` against the host over 1°–179° and
the committed fields of view. If any bit moves, stop and report (the owner decides).

### Task G5: the editor's scene model

**Files:** the scene model (new modules, or a new layer-12 crate `es-editor-scene` if
`es-editor-model`'s 10,000-line cap requires it — then §4.2 / Appendix C.8's `LAYERS`, ko first, and
xtask's table in the same packet), `crates/es-editor/src/ui/**` (hierarchy, inspector), i18n.

- An editable project holds `scene.esscene` + `task.estask`. The model: the `EsScene` document (G1),
  a selection, **commands** (add / delete / duplicate entity, set a field, set a pose, re-parent,
  rename — a rename also rewrites the names in `task.estask`), an undo / redo stack, validation after
  every command (G1's `expand` + the mapping report of the project's backend, refusals in plain words
  on the field), save. A template project's scene stays read-only, with "make an editable copy".
- ①: a hierarchy panel (tree, includes folded, search, visibility) and an inspector (pose in metres
  and Euler degrees through `es_math::approx`, shape and size, mass, friction, colour / material /
  texture, joint kind and range), editing through commands; the viewport (H9) re-renders on change.
- **Oracles:** headless: apply / undo / redo returns the identical document (a property test over
  random command sequences); an invalid edit is refused with the field; a rename rewrites the spec
  and the regenerated documents (G3b) still validate; the saved document re-reads equal; a screenshot
  of ① editing the cube's size and colour in the Shadow Hand project copy.

**G5 as merged (2026-10-01):** the scene model is a new layer-12 crate, `es-editor-scene`
(`es-editor-model` stood at 8,942 of its 10,000-line cap); spec §4.2 rule 4 and Appendix C.8 name
both layer-12 crates, which do not know each other. A refused command is not applied, so every
undo step is a scene that expands. "Make an editable copy" keeps the template's `scene_hash`; the
generated IRs differ from the committed ones only where they name the scene file or cite its hash.
Steps ② to ⑤ of an editable project still run the template's documents until G9. Design note §5.1
records the rest.

### Task G6: viewport picking and gizmos (after G5)

**Files:** `crates/es-editor-model` / the scene crate (pick and gizmo decisions), `crates/es-editor`
(drawing, input), `crates/es-render` only if a segmentation readback needs an API.

- Click to select (the segmentation channel of the in-process render gives the geom → its entity),
  translate / rotate / scale gizmos with snapping (cm, 15°), frame-selected; a drag is one command
  (one undo step); the policy camera's view in the viewport's corner (the declared camera, its
  resolution and render path, in process).
- **Oracles:** headless pick → entity and gizmo drag → pose delta on scripted rays; a drag is one
  undo step; screenshots of a gizmo drag and the corner view.

**G3d as merged (2026-10-01):** the compiler takes `cos` and `tan` from `es_math::approx` (musl via
the `libm` crate), so generated documents are host-independent; no committed or G3b-pinned hash
moved. The G3b paragraph's "musl's `tan` matched the host at every probed field of view" holds only
at the committed angles (45°, 70°): over 1°–179° in 0.001° steps 4.1 % of values differ by one ULP
(neither side is correctly rounded), so a document generated before G3d on Windows with such an
angle (e.g. a 68° camera) may differ from a regenerated one in its last bit. The orchestrator kept
musl — spec §5.3's rule for numbers that enter a hash. The fixture generators in
`crates/es/tests/{shadow_hand,views}.rs` still call the host; they agree at their committed angles.
