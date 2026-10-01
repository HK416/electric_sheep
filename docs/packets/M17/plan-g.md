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
  in `es-script` (11); CLI in `es`; scene model in `es-editor-model` (12); widgets in `es-editor`
  (13). No new extension point (INV-17).

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | G1 the scene document · G2 the full MJCF exporter | — |
| 2 | G3 task spec + `es project generate` · G4 `es render` / `es scene simulate` | G1 |
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
