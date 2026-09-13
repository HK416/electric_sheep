# W3-usd-reader — native `.usda` subset reader

Spec: 28.6 (M4 scope: "USD 네이티브"), 1.9 item 6 (this is *cuttable*, replaced by Bake — so it
stays minimal), 2.5 (USD Bake via Python is the supported fallback), 3.1 (Z-up, metres,
quaternion xyzw), 4.2 (layering), 5.3 (`asset_hash` → `scene_hash`), 17.2 (backend semantic
mapping), 1.2 (work packets), 1.4 (oracle first).

Notes: `docs/api-notes/usd.md` (format digest, with every unverified item marked),
`docs/design/usd-reader.md` (code shape).

## context

```
crates/es-usd/**
crates/es-physics-core/src/usd.rs
crates/es-physics-core/src/lib.rs
crates/es-physics-core/Cargo.toml
crates/es/src/cmd/import.rs
crates/es/tests/cli.rs
tests/fixtures/usd/**
docs/api-notes/usd.md
docs/design/usd-reader.md
docs/packets/M4/W3-usd-reader.md
```

`crates/es-usd/**` owns `parse_usda`, `UsdStage`, `Value` and `UsdError`;
`crates/es-physics-core/src/usd.rs` owns `stage_to_scene`. The `es-physics-core` `lib.rs` change
is `pub mod usd;` and one doc line, and the `Cargo.toml` change is the `blake3` dependency the
mesh content hash needs — nothing else. `crates/es/tests/cli.rs` is **append only**, and the
fixtures are hand-written and under 20 KB each.

## spec

1. `es_usd::parse_usda(&str) -> Result<UsdStage, UsdError>` — hand-written tokenizer plus
   recursive-descent prim parser, no new crate dependency. `UsdStage { meta, prims, warnings }`,
   `Prim { path, type_name, specifier, attrs: BTreeMap<String, Value>, rels, api_schemas,
   children }`.
2. `Value` is typed from the *declared* attribute type: `Bool`, `Int`, `Float`, `Double`,
   `Token`, `String`, `Asset`, `Float3`, `Quat`, `Matrix4d`, `Array`, `Rel`. A `quatf`/`quatd`
   literal is `(w, x, y, z)` in the file and is stored as spec 3.1 `xyzw`.
3. Every error carries the source line. Unknown attributes are preserved in `attrs`; unknown
   prim types are kept with a warning. `references` / `payload` / `variantSet` / `inherits` /
   `subLayers` / a `.usdc` or `.usdz` magic are `UsdError::Unsupported { path, feature }`.
4. `UsdStage::resolve_xform(path) -> Pose` composes ancestor local transforms; a local
   transform follows `xformOpOrder` (first-listed is outermost). Scale is dropped with a
   warning.
5. `es_physics_core::usd::stage_to_scene(&UsdStage) -> Result<(SceneDesc, Vec<Warning>),
   UsdSceneError>` — rigid bodies, collision geoms (primitives plus `Shape::Mesh` behind a
   content-hashed `AssetRef`), `PhysicsMassAPI` inertials, revolute/prismatic/fixed joints with
   axis, limits and drive. Y-up → Z-up and `metersPerUnit` scaling applied once each; revolute
   limits and angular drive targets converted from degrees to radians. Ids from prim paths via
   `scene_id`.
6. `es import usd <file.usda> --out scene.json`.
7. English only, `BTreeMap` only, no new extension-point trait (INV-17), no new dependency.

## oracle

```
cargo fmt --check
cargo clippy -p es-usd -p es-physics-core -p es --all-targets -- -D warnings
cargo test -p es-usd -p es-physics-core -p es
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance

- Fixtures in `tests/fixtures/usd/`:
  - `pendulum.usda` — two rigid bodies, one `PhysicsRevoluteJoint` with limits and an angular
    drive; maps to two `Body`s and one `JointKind::Hinge` with the drive as
    `stiffness`/`damping`/`spring_ref` and limits in radians.
  - `mesh_cube.usda` — a `Mesh` with `PhysicsMassAPI`; maps to one `Shape::Mesh` whose
    `AssetRef::hash` is the content hash of the converted points/indices, and a `BodyInertial`.
  - `yup_cm.usda` — `upAxis = "Y"`, `metersPerUnit = 0.01`; a body at USD `(0, 300, 0)` lands
    at spec 3.1 `(0, 0, 3)` m, asserted numerically.
  - `referenced.usda` — uses `references`; `parse_usda` returns `Unsupported` naming the prim
    path.
  - `malformed.usda` — returns a `Syntax` error with the right line number, and does not panic.
- `parse_usda` unit tests cover every `Value` kind, including that `quatf (1, 0, 0, 0)` parses
  to xyzw identity and `quatf (0, 1, 0, 0)` to a 180° x-rotation.
- Every `SceneDesc` produced by `stage_to_scene` passes `SceneDesc::validate()`, and
  `scene_hash` for a fixture is stable across two parses of the same text.
- `es import usd tests/fixtures/usd/pendulum.usda --out <tmp>/scene.json` exits 0, prints the
  scene hash, and the written JSON round-trips back into a `SceneDesc` that validates.

## forbidden

- Adding a USD crate dependency, or any dependency, to `es-usd` (spec 2: pure Rust core, no
  C++; spec 1.9 makes this component cuttable, so it may not grow the dependency graph).
- Touching `crates/es-gpu`, `es-ir`, `es-env`, `es-script`, `es-data`, `es-eval`,
  `es-physics-backend`, `es-policy`, `es-assets`, `xtask`, `.github` — other packets own them.
  In particular `es-usd` must not depend on `es-assets`: both are layer 2 and spec 4.2 forbids
  same-layer deps.
- A new extension-point trait (INV-17), `HashMap` anywhere on the import path (spec 3.4),
  or silently ignoring a composition arc.
- Editing any golden file, or making `stage_to_scene` invent physics the file does not state
  (no derived masses, no invented collision intent).
