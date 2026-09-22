# M10 W2a — mesh geoms load, hash by content and simulate on MuJoCo

Spec: §5.3 (`asset_hash` → `scene_hash`: a mesh's content, not its path, is what the chain
covers), §3.4 / §3.2 (no host libm on the hash path; `str::parse::<f32>` and `from_le_bytes`
only), §4.3 / §17.2 (backend capabilities are declared, a scene a backend cannot map is refused
before anything is spawned), §28.13 rule 2 (a mesh is an addition: the primitives-only scenes'
`scene_hash` do not move; the decoded meshes ride on `SceneDesc`; the resolver is a function) and
wave 1, INV-17. Review: `docs/reviews/M7.md` "R5/R6 (mesh geoms, MJCF textures) … parked";
§28.10 "what is not on the ladder". Design notes: `usd-reader.md` (the content-hash rule this
packet adopts for every importer), `renderer.md` 2.1 (the `Mesh` row: "needs an asset resolver").
Depends on **W0b** (same crate; `scene_hash` platform-stable first). Type B.

## the question

`Shape::Mesh { asset }` already parses from MJCF, URDF, glTF and USD and hashes, but no reader
turns an STL or OBJ into vertices, `AssetRef::hash` is a digest of the *path string*, the MuJoCo
emitter refuses the geom, and the renderer refuses it — so every robot in Menagerie is out of
reach and both campaigns' source policies were trained in scenes we derived by hand. **Can an
MJCF scene with mesh geoms load (STL / OBJ), carry its meshes on `SceneDesc`, hash by content
identically on Windows and Linux, and run on the MuJoCo backend — with the committed
primitives-only scenes' `scene_hash` unmoved?**

Verified before writing (the agent re-checks the lines): both committed scenes have `assets: []`
(`crates/es-assets/tests/so101_provenance.rs:374-378`); `AssetRef.path` is hash input
(`scene.rs:626-632`) so it must never be rewritten; the MJCF importer computes no geom mass
(`scene.rs:80-82`) — the emitter writes `mass=` / `density=` and MuJoCo derives, so a mesh geom
needs **no Rust inertia code**; `PhysicsBackend::load(&SceneDesc)` and `EnvRenderer::new(&SceneDesc)`
are the INV-17 traits a side store could not pass; `mujoco_ref.py` loads via `from_xml_string`,
so an inline `<mesh vertex= face=>` needs no file path on the wire; MJWarp derives its
capabilities from MuJoCo's (`mjwarp.rs:52-58`).

## spec

* `SceneDesc.meshes: BTreeMap<StableId, MeshData>` — `#[serde(default, skip_serializing_if =
  "BTreeMap::is_empty")]`, **not** hashed (the `AssetRef.hash` rewrite is what enters
  `scene_hash`); every `SceneDesc { … }` literal without `..Default::default()` gains
  `meshes: BTreeMap::new()`. `MeshData` is the existing `es_assets::gltf::MeshData`
  (`gltf.rs:97-105`) with `Serialize, Deserialize` added; STL / OBJ fill `positions` and
  `indices` and leave `normals` / `uvs` / `material` `None`. `gltf::mesh_content_hash`
  (`gltf.rs:442`) becomes the crate's one content-hash function (`pub(crate)`): tag ‖ f32 LE
  positions ‖ u32 LE indices — platform-independent by construction.
* `es_assets::mesh::load(scene: &mut SceneDesc, base_dir: &Path) -> Result<(), MeshError>`:
  for every `AssetKind::Mesh` asset not yet in `scene.meshes`, resolve `base_dir.join(asset.path)`
  (`path` is already `join(meshdir, file)`, `elements.rs:53,397`; absolute stays absolute), read,
  dispatch on the extension (`.stl` / `.obj`; else `MeshError::Unsupported { path }`), insert, and
  set `asset.hash = mesh_content_hash(…)`. `lib.rs:7` ("nothing reads a file from disk") gains
  the one exception. `MeshError` names the path and the reason (missing file, bad index, short
  buffer).
* Readers: `es_assets::stl::parse(&[u8])` — binary iff `len == 84 + 50·n` (a header may begin
  with `solid`), else ASCII; vertices deduplicated by exact f32 bits (`BTreeMap<[u32; 3], u32>`,
  first-seen order, deterministic); if a stored facet normal is non-zero and
  `dot(normal, (b−a)×(c−a)) < 0`, swap `b`/`c` (the renderer's normal comes from winding).
  `es_assets::obj::parse(&str)` — `v` (first three numbers), `f` in the forms `a`, `a/b`, `a//c`,
  `a/b/c`; 1-based, negative = relative to the current `v` count; fan-triangulate; everything
  else ignored. Both validate `index < positions.len()`.
* MJCF: `elements.rs parse_asset` reads `scale`, `refpos`, `refquat` and returns
  `MjcfError::Unsupported` for a non-default value (the `<compiler coordinate>` pattern at
  `mod.rs:379-385`) — today `report_unknown` demotes them to warnings, which would become a
  silent geometry error once meshes render. Baking `scale` is a named follow-up.
* Emitter (`mjcf_out.rs`): `<asset><mesh name="…" vertex="x y z …" face="i j k …"/></asset>`
  from `scene.meshes` (f32 via `{:?}`, the shortest round-trip repr), `<geom type="mesh"
  mesh="…">` with no `size`. A mesh geom whose asset is not in `scene.meshes` →
  `PhysicsError::Unsupported("geom `g`: mesh `m` is not loaded (es_assets::mesh::load)")`,
  still before `Process::spawn` (`mujoco.rs:226` vs `:232`). No `file=` paths (that needs a path
  channel through the trait or a machine path in a hashed struct — a follow-up if inline text
  is ever too slow).
* Capabilities: `mujoco.rs:67-73` gains `Feature::ContactMesh` with the quirk "collision uses
  the convex hull of the mesh; the renderer draws the surface"; `mjwarp.rs:58` removes
  `ContactMesh` (unverified there); Newton (`mapping.rs:392`) and the fixture backend unchanged
  — a mesh scene on them is refused by name through `Requirements::from_scene`.
* Fixtures: `tests/fixtures/mjcf/mesh_box.xml` — `<compiler meshdir="meshes"/>`, `<asset><mesh
  name="box" file="box.stl"/>`, a plane, `mesh_box` (freejoint, `type="mesh"`, density 1000, at
  z = 0.3) and `prim_box` (freejoint, `size="0.05 0.05 0.05"`, x = 0.5), one `<camera name="cam">`
  for W2b; `tests/fixtures/mjcf/meshes/box.stl` = binary, 684 B (80-byte header + 12 facets of
  the ±0.05 box), written by `#[ignore] generate_mesh_box_stl` under `ES_GENERATE_GOLDENS=1`.
  Provenance: `so101_pick_place.PROVENANCE.json` gains `"mesh_blake3": { "<name>": "<hex>" }`
  for the 19 upstream STLs (never vendored; fetched into `target/menagerie/<commit>/` by the
  existing `so101_provenance.rs:109-147` pattern extended to a file list);
  `tests/fixtures/mjcf/panda.PROVENANCE.json` — Menagerie `franka_emika_panda` at a pinned
  commit, blake3 of `panda.xml` and of each `.obj` / `.stl`, no vendoring. Panda is **readers +
  hashes only** (its `panda.xml` is believed to carry a finger `<tendon>` / `<equality>` that
  `scene_to_mjcf` refuses — unverified; if a MuJoCo OBJ load is wanted, `universal_robots_ur5e`
  is the candidate). `docs/api-notes/mujoco.md` (+ `.ko.md`) gains a mesh section: binary STL
  layout, OBJ index rules, `<mesh>` attributes, `inertia="legacy"` vs `"exact"` as read from the
  MuJoCo 3.13 docs, the Menagerie commit.

## context

```
crates/es-assets/src/lib.rs
crates/es-assets/src/scene.rs
crates/es-assets/src/mesh.rs
crates/es-assets/src/stl.rs
crates/es-assets/src/obj.rs
crates/es-assets/src/gltf.rs
crates/es-assets/src/mjcf/elements.rs
crates/es-assets/tests/**
crates/es-physics-backend/src/mjcf_out.rs
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/tests/**
crates/es-physics-core/src/usd.rs
crates/es-render/src/cornell.rs
crates/es-splat/src/**
crates/es-usd/src/**
tests/fixtures/mjcf/mesh_box.xml
tests/fixtures/mjcf/meshes/box.stl
tests/fixtures/mjcf/so101_pick_place.PROVENANCE.json
tests/fixtures/mjcf/panda.PROVENANCE.json
docs/api-notes/mujoco.md
docs/api-notes/mujoco.ko.md
docs/packets/M10/W2a-mesh-assets.md
docs/packets/M10/W2a-mesh-assets.ko.md
```

(The `es-physics-core`, `es-render`, `es-splat`, `es-usd` entries are for `SceneDesc` literals
gaining `meshes: BTreeMap::new()` only.)

## oracle

1. `cargo test -p es-assets --test mesh_load` — `binary_stl_dedups_shared_vertices` (two
   hand-written facets sharing an edge → 4 positions, 6 indices),
   `binary_stl_flips_a_facet_against_its_normal`, `ascii_stl_decodes`,
   `obj_negative_indices_and_quad_fans`, `obj_face_forms_slash_and_double_slash`,
   `box_stl_content_hash_is_pinned` (hex constant), `mesh_box_load_rewrites_asset_hash_and_scene_hash`
   (before ≠ after; after == pinned), `committed_scenes_are_unmoved_by_load` (so101 / go1: hash
   before == after; pin the so101 hex too — W0b has made it platform-stable),
   `scale_refpos_refquat_are_refused_by_name`, `missing_file_names_the_path`, `bad_index_is_an_error`.
2. `cargo test -p es-assets --test mjcf_parse assets_are_references_not_files` unchanged.
3. `cargo test -p es-physics-backend` — `mjcf_out::tests::mesh_geom_emits_inline_vertex_and_face`,
   `mjcf_out::tests::unloaded_mesh_is_refused_by_name`, `mujoco::tests::capabilities_are_declared_honestly`
   (`:429` flipped), `a_scene_the_backend_cannot_map_is_refused_before_anything_is_spawned` (the
   mesh case still pre-spawn), the mjwarp capability/mapping agreement test.
4. `ES_PYTHON=… cargo test -p es-physics-backend --test mesh_box` — `mesh_box_loads_in_mujoco`
   (nbody 3, nmesh 1, `mesh_vertnum[0] == 8`, `mesh_facenum[0] == 12`, zero compile warnings),
   `mesh_box_rests_like_the_primitive_box` (2,000 steps at 1 kHz, `|z_mesh − z_prim| < 1e-3 m`,
   both `0.05 ± 2e-3`, no NaN), `mesh_box_mass_and_inertia_match_the_primitive` (`body_mass`
   relative < 1e-9, `body_inertia` relative < 1e-6; MuJoCo 3.13's default `<mesh inertia="legacy">`
   should equal `exact` for a convex closed box — **unverified**: if it differs, record the
   measured gap and the tolerance that holds). Run locally (`ES_PYTHON` venv) **and** on the
   server (CPU queue, minutes) — the same bytes.
5. `--ignored`, on the server: `so101_provenance::upstream_meshes_load_and_hash` (19 hashes ==
   manifest; `derivative_has_the_upstream_kinematics` already proves the twin);
   `menagerie_meshes::so101_upstream_loads_in_mujoco_with_meshes` (`ES_PYTHON`: loads; per-body
   `body_mass` == a direct `from_xml_path` load of upstream within 1e-9 relative; `camera_mount`
   0.012 kg); `panda_provenance::panda_meshes_load_and_hash` (OBJ + STL on real files; hashes
   and triangle counts pinned in the api-note).
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W2a-mesh-assets.md`;
   `cargo xtask context-budget` (`es-assets` expected ≈ +290 → ~3,890).

## acceptance

Oracles 1–4 and 6 green on CI; 5 green with the env set; both committed scenes' `scene_hash`
unmoved; `tests/golden/**` untouched; the api-note section with its Korean sibling; the follow-ups
named in the note: `<mesh scale>` baking, `file=` with a path channel, glTF `MeshData` →
`scene.meshes`, MJWarp `ContactMesh`, textures / materials (M7 R6).

## forbidden

`es-render`, `es-env`, `es-editor`, `es` changes (W2b); rewriting `AssetRef.path`; adding a field
to the `scene_hash` encoding; touching `PhysicsBackend` or any trait (INV-17); vendoring a
Menagerie mesh; Rust mesh mass / inertia code; host libm anywhere on the load path; baking
`scale`; textures / materials; `docs/ARCHITECTURE*.md`; `tests/golden/**`.
