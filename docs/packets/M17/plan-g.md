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

**G6 as merged (2026-10-01):** picking is the segmentation id at the clicked point read on the CPU
(the renderer's own ray and nearest-hit rule against the drawn triangles; no readback, no
`es-render` change). Anything an include brings selects the include. Snapping is on by default.
The corner draws the policy's declared camera through `es-env`'s single-camera render steps, so
`es-editor` now takes `es-env` with `render`. Design note section 5.2 records the rest. `es-editor` is
at 6,816 lines (past its 6,000 target, under the cap). Wave 3's CI passed on 1bc3bfc.

### Task G7: Add (primitives, files, robot library, camera, light, region)

**Files:** `crates/es-editor-scene` (new modules: what each Add item makes and where it goes, file
import into `assets/`, include overrides), `crates/es-editor/src/ui/**` (the Add menu in a new file,
camera and region markers, the overrides inspector), i18n, a robot library list
`templates/robots.toml`, a test fixture `tests/fixtures/esscene/empty.esscene` (floor, light, one
camera), the design note's section 5.3 (+ko). Not `es-assets`, unless a reader refuses a file the library
needs.

- **①'s Add menu.**
  - **Object:** a box, sphere, cylinder or capsule, as a free body with one geom.
  - **Fixed object:** static scenery.
  - **Mesh file:** an STL or OBJ, as a free body with a mesh geom.
  - **Robot:** from the library (SO-101, Shadow Hand), or from an MJCF, URDF or glTF file. Either
    way it becomes an `[[include]]`.
  - **Camera:** at the viewport's eye, looking where the view looks.
  - **Light** and **region** (a box zone).
- **Where a new thing goes.** It goes where the view's centre ray meets the scene
  (`SceneModel::hit`), resting on that surface; with no hit, at the origin. Its name is
  `unique(stem)`. It ends up selected, and one Add is one undo step.
- **Imports are copied by content** (design note section 3.1, its section 9 default 3).
  - A mesh or texture file goes to `assets/<blake3>.<ext>`.
  - A robot file goes to `assets/<blake3 of the file>/`, together with the files it names (meshes,
    textures, `meshdir`), keeping their relative paths, so its reader resolves them unchanged.
    The library's robots are copied the same way.
  - Copying the same file twice writes nothing new.
  - A file that does not read or expand is refused with the reader's reason, and leaves nothing
    behind.
- **Texture import:** the material picker gains "from an image file". That one command adds a
  `[[texture]]` and a `[[material]]` and sets the selected geom's material.
- **Cameras and regions in the viewport.**
  - They are drawn as line overlays: a camera's frustum, a region's box.
  - They are picked in screen space within a few pixels, and get G6's handles (a region's size
    too).
- **Include overrides:** the inspector edits `[include.set]` (G1's targets: joint `range`,
  `damping`, `armature`, `stiffness`, `frictionloss`; actuator `kp`, `kv`, `ctrlrange`,
  `forcerange`; geom `rgba`, `material`), by the asset's own names. Clearing a field drops its
  override.
- **Oracles** (headless unless noted):
  1. **Each Add item** on the Shadow Hand copy and on `empty.esscene`:
     - the result expands and maps onto `mujoco-cpu`;
     - the new entity is selected;
     - it is one undo step, and undo restores the bytes.
  2. **Placement:** a box added on a scripted centre ray at the SO-101 table rests on it (its
     bottom at the hit height, within 1e-9). A ray that hits nothing places it at the origin.
  3. **Import by content.**
     - Importing `so101.xml` with its meshes twice into a temporary project writes
       `assets/<hash>/…` once.
     - The expansion's per-asset content hashes equal the source's.
     - A broken file is refused, with nothing added and no new file.
     - A document with an imported mesh survives a read and write round trip.
  4. **The library's SO-101**, added to `empty.esscene`, expands to `so101.xml`'s bodies, joints
     and actuators (names, ranges) at its pose.
  5. **Overrides:** setting an included joint's range writes `[include.set.joint.<name>]`, which
     the expansion shows; clearing the field removes the table.
  6. **Picking markers:** screen-space picks of a camera marker and a region box, on scripted
     points.
  7. **Screenshots:** the Add menu; a box added on the table in the SO-101 copy; a robot added to
     an empty scene.

### Task G8: the sentence editor

**Files:** `crates/es-editor-scene` (a new module: the specification as sentence rows, the
vocabulary each subject takes, commands on the specification), `crates/es-editor/src/ui/**` (the
sentence panel in ①, in a new file), i18n, `crates/es-script/src/spec/**` only for a helper the
editor reads (the vocabulary, a clause's words) that changes no compiled byte, the design note's section 4.6
(+ko).

- **The task part of ①.** The specification reads as sentences with a drop-down per slot
  ("[cube] is [inside] [bin]", "fails if [cube] is farther than [24 cm] from [palm]", "fails if
  not done within [8 s]").
  - **Subjects:** the scene's bodies, the robot's joints, `<body>.x/.y/.z`, regions.
  - **Relations:** only those the subject's kind takes (design note sections 4.1, 4.3, 4.4). `touches` is shown
    disabled, with its reason.
  - **Fields:** numbers in the person's units (cm, °, s).
  - **Clauses:** add, remove, reorder; failure clauses; the time limit.
  - **Start items:** `what`, 🎲, noise, and one strength (low / medium / high).
  - **Observe:** the cameras as a checklist of the scene's cameras, `camera_px`, the render look
    (quick / material / PT with samples), the state and privileged sources.
  - **Reward:** the success bonus, and each clause's shaping on or off with a three-level weight.
    The numbers behind the levels are this packet's default and the owner's to change (design note section 9, item 5). A
    weight at no level shows as "custom (value)" and is kept as written.
- **Every edit is one command** on `SceneModel`: a command that replaces the specification. It is
  checked by `compile_task` on the scene as edited and refused in place, naming the clause and the
  field (G3a's refusals). It is one undo step, and the corner follows it (the revision moves).
  G5's policy holds: the model's specification always compiles, or there is none.
- **A project with no `task.estask`** (an SO-101 copy) starts one with "say the task". That is one
  command producing a specification that compiles, from defaults the person then edits.
- **A check as the person goes:** an observed camera that does not see a clause's subject at its
  start position. This is one CPU ray from the camera to the subject (G6's `view` and `hit`); the
  first hit must be the subject. It is shown in words beside the camera.
- **Oracles** (headless unless noted):
  1. **Round trip:** each committed specification (`shadow_hand_repose.estask`,
     `so101_views.estask`) loaded into the sentence model and written back without an edit gives
     an equal `TaskSpec`. Its sentences, in Korean and English, are pinned in the test.
  2. **Every edit kind** is one undo step that compiles:
     - a clause added, removed or moved; a relation; a field;
     - a start item's 🎲, noise and the strength;
     - the cameras, the render look, a reward level.

     Changing the success angle moves `task_hash` and not `scene_hash`.
  3. **Refusals:** a relation the subject does not take is not offered. A missing field, a time
     limit that is not whole ticks, and `touches` are refused, naming the clause and field, and
     nothing changes.
  4. **A new task on the SO-101 copy:** "say the task", a region added (G5's `Add`), then
     "[cube] is inside [region]" and "[cube] is still". It compiles, and `generate` writes the
     documents.
  5. **The camera check** on scripted scenes: a camera looking at the cube passes; one turned
     away or blocked fails.
  6. **Screenshots:** the Shadow Hand copy's sentences in Korean; an edit with a refusal; the
     SO-101 copy with its new task.

G7 and G8 run in parallel, each in new files. Where both must touch a shared file (`lib.rs`,
`author.rs`, the i18n tables), each keeps its hook small and its keys in a block of its own. The
orchestrator rebases the second onto the first.

**G7 as merged (2026-10-01):** ①'s Add menu (objects, fixed objects, STL / OBJ meshes, library
and file robots, cameras, lights, regions) places a new thing where the view's centre ray meets
the scene. Imports go to `assets/<blake3>…`. Camera and region markers are picked on screen, and
`[include.set]` has an inspector. `templates/robots.toml` holds the library, and
`tests/fixtures/esscene/empty.esscene` is the empty scene. Meshes have no scale yet (an STL in
millimetres comes in 1000× too large). Design note section 5.3 records the rest.

**G8 as merged (2026-10-01):** the specification reads as sentences in ①'s Task tab, and
`Command::Spec` replaces it. After every command `SceneModel` compiles the specification, so a
scene edit that would break a sentence is refused. The vocabulary is the compiler's table
(`es-script`'s `spec/vocab.rs`, no compiled byte moved). The weight levels are 0.1 / 1 / 10, the
bonus 25 / 100 / 250, and the strength 0.5 / absent / 1.5; these are the owner's to change. The
camera check found that SO-101's overhead camera does not see the cube's centre at the home pose
(the lower arm is in the way). Design note section 4.6 records the rest. Wave 4's CI passed on
5cc21fe.

### Task G9: an authored project runs end to end

**Files:**
- `crates/es-editor-model` (the start screen's card, an authored project's documents, the
  working directory of the `es` it launches);
- `crates/es-editor-scene` (the defaults "say the task" writes, the paths of the generated
  documents, save as template);
- `crates/es-editor/src/ui/**`;
- `crates/es-script/src/spec/**`, only if `generate` needs an option; no committed
  specification's output may move;
- `templates/empty.toml` (new), i18n, and the design note's section 5.4 (+ko).

- **The empty project.**
  - The start screen gets an "empty project" card beside the templates. It makes an editable
    project from `tests/fixtures/esscene/empty.esscene`, with no specification.
  - Either the template's bundle becomes optional (`templates/empty.toml` with `[editable]` and
    no bundle), or the empty project is a project kind of its own. The packet chooses and records
    the choice.
  - ① works fully. Until there is a specification and its documents are generated, ② to ⑤ say in
    words what is missing.
- **② to ⑤ run the generated documents.** An editable project qualifies when its specification
  compiles and `generated/` holds a `Regen::Written` set from the saved documents. Every step then
  uses the generated documents instead of the template's: the bundle, the teacher (its recipe and
  bundle), the cycle and the evaluations.
  - **Paths.** Generated documents name `scene.esscene` relative to the project root, and recipes
  derive their paths from `runs`. So every `es` the editor launches for such a project either runs
  with the project root as its working directory, or gets absolute paths. Choose one and say why.
  - **Bundles** are built by the `es policy init` line in each recipe's header.
  - **Stale or failed generation.** If the documents differ from the saved ones, or the result is
    `Regen::Failed`, ③ is blocked with the reason and ① offers "save and generate".
  - **Template projects** keep running the template's documents exactly as before.
- **The teacher for generated tasks (②).**
  - "Say the task" also writes `[teacher]`, `[student]` and `[cycle]` defaults, so `generate`
    writes the full set:
    - `[teacher]`: every `[observe] state` and `privileged` channel, and the PPO preset;
    - `[student]`: the observed cameras as its views;
    - `[cycle]`: as G3b defaults it.
  - ② offers the teacher trained by reward for every project with a `[teacher]`. This is plan H's
    card (H7) generalized from the Shadow Hand template, and it needs `mjwarp`, as the hand's does.
- **Save as template.** Any editable project can be saved as a template. The template is written
  under the person's documents, not the repository, and the start screen lists it beside the
  built-in ones. It holds:
  - the scene document;
  - `assets/`, copied whole (a glTF's `.bin` is not in the scene's asset list);
  - the specification;
  - a template file whose `[editable]` names them.

  A project made from it has the source project's `scene_hash`.
- **Oracles** (headless unless noted):
  1. **End to end.** An empty project gets the library's SO-101, a box, a region, "say the task"
     and "[box] is inside [region]", then a save. That gives `Regen::Written` with the full set. As
     the editor would launch them:
     - every generated document passes `es ir check`;
     - every recipe's `es policy init` line builds its bundle;
     - `es loop cycle --dry-run` on the generated cycle passes.
  2. **Argvs.** The argvs the editor builds for ② to ⑤ on that project name the generated
     documents (pinned). A template project's argvs are unchanged (pinned).
  3. **Blocking.** A stale or failed generation blocks ③ with its reason, and saving clears the
     block.
  4. **Template round trip.** Save as template, then make a new project from it. The `scene_hash`
     is equal, the specification is equal, and the generated documents' hashes are equal.
  5. **Screenshots:** the start screen with the empty card; ② of an authored project with the
     teacher card; ③ ready on generated documents.
- **Not in this packet:**
  - a learning run (the orchestrator's GV runs one);
  - ⑤ explaining a failure by its missing clause, the owner's approved S4 direction. That needs
    per-clause truth at each attempt's end, a design item of its own.
- **Line budget:** `es-editor-model` is at 8,969 of 10,000 code lines. G9 must not take it past
  9,600. If it would, stop and report: a split packet comes first.

**G9 as merged (2026-10-01):** the empty project is a template without documents
(`templates/empty.toml`; `Template.bundle` is optional). An editable project with a task runs
② to ⑤ on `generated/`, with `es` in the project's own folder. This holds only while
`generated/` is exactly what the saved documents generate; otherwise ③ is blocked with the
reason. "Say the task" also writes `[teacher]`, `[student]` and `[cycle]`, so a save generates
thirteen documents. ②'s teacher card serves any project with a `[teacher]`, checked against
`mjwarp`'s mapping report first. Save as template writes to
`<documents>/Electric Sheep/templates/<name>/`. Only the dry-runs were measured; no learning run
was made. The default reward of "[box] is inside [area]" is the success bonus alone. Design note
section 5.4 records the rest. CI passed on 832f0db.
