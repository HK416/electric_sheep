# Design — native `.usda` reader (M4, spec 28.6)

Format facts and their provenance live in `docs/api-notes/usd.md`. This note is only about the
shape of the code.

## 1. Why this is small on purpose

Spec 1.9 ranks the USD native reader sixth on the cut list: **"USD 네이티브 리더 — Bake로
대체"**. The supported import path is the Python USD Bake of spec 2.5. So the goal here is not
an OpenUSD implementation; it is: *a hand-written subset reader good enough to get a rigid-body
scene out of a flattened `.usda`, that refuses loudly the moment the file leaves the subset.*

Two consequences that shape every decision below:

- **No composition.** `references`, `payload`, `variantSet`, `inherits`, `subLayers` are
  `UsdError::Unsupported` with the prim path. A reader that silently ignored a reference would
  produce a scene missing half its bodies and no error — the worst possible failure for an
  importer.
- **No new dependency.** A hand-written tokenizer is ~300 lines; an OpenUSD binding is a C++
  toolchain (spec 2, pure Rust core). The subset is small enough that the lazy option is also
  the correct one.

## 2. Two crates, one seam

| Crate | Layer | Owns |
|---|---|---|
| `es-usd` | 2 | `.usda` text → `UsdStage`. Knows nothing about physics, bodies or joints. |
| `es-physics-core::usd` | 3 | `UsdStage` → `SceneDesc`. Knows nothing about text. |

The seam is the point of the split. Layer 2 forbids `es-usd` from depending on `es-assets`
(same layer, spec 4.2 "no same-layer deps"), which is exactly why `UsdStage` cannot mention
`SceneDesc` and why the mapping lives one layer up, in `es-physics-core`, which already
depends on both.

It also keeps the parser honest: `UsdStage` is a faithful, lossless-enough mirror of the file
(unknown attributes preserved, unknown prim types kept with a warning), with no interpretation.
Every judgement — what counts as a body, how a `Capsule` axis becomes a pose, what a degree is
— is in the mapper, where it can be read in one sitting.

## 3. `es-usd`

```text
parse_usda(&str) -> Result<UsdStage, UsdError>
UsdStage { meta: StageMeta, prims: Vec<Prim>, warnings: Vec<String> }
Prim { path, type_name, specifier, attrs: BTreeMap<String, Value>, rels, api_schemas, children }
```

- Tokenizer: one pass, tracks the line number of every token, so every `UsdError::Syntax`
  carries a line. `#` to end of line is a comment (except the `#usda` magic).
- Parser: recursive descent over `def`/`over`/`class` blocks. Depth is capped (a fuzzer feeding
  10⁶ `{` must get an error, not a stack overflow).
- `Value` is typed by the *declared* type name, not guessed from the literal: `float 1` and
  `int 1` stay distinguishable, and `quatf (w,x,y,z)` is converted to spec 3.1 `xyzw` at parse
  time so no consumer can get the order wrong.
- `resolve_xform(path) -> Pose` composes ancestors' local transforms. Local transform follows
  `xformOpOrder` exactly, column-vector `M = op[0] * ... * op[n-1]` (api-note 3). Scale is not
  representable in a rigid `Pose`: a non-identity scale is a warning and is dropped, the same
  choice the glTF importer already makes for non-uniform node scale.
- `BTreeMap` only (spec 3.4 forbids `HashMap` iteration dependence). Nothing in this crate
  reads a file from disk; `es` hands it a `&str`.

## 4. `es-physics-core::usd`

```text
stage_to_scene(&UsdStage) -> Result<(SceneDesc, Vec<Warning>), UsdSceneError>
```

Mapping rules, all of them mechanical:

| USD | `SceneDesc` |
|---|---|
| prim with `PhysicsRigidBodyAPI` | `Body`; see the parent rule below |
| prim with `PhysicsCollisionAPI` | `Geom` on its nearest rigid-body ancestor |
| `Cube size` | `Shape::Box { half_extents: size/2 }` |
| `Sphere radius` | `Shape::Sphere` |
| `Cylinder`/`Capsule` `radius`,`height`,`axis` | `Shape::Cylinder`/`Capsule { half_length: height/2 }`, `axis` folded into the geom pose |
| `Mesh points`/`faceVertexIndices`/`faceVertexCounts` | `Shape::Mesh` + content-hashed `AssetRef` |
| `PhysicsMassAPI` | `BodyInertial { mass, com, inertia, frame }` |
| `PhysicsRevoluteJoint` | `JointKind::Hinge`, limits deg → rad |
| `PhysicsPrismaticJoint` | `JointKind::Slide`, limits scaled by `metersPerUnit` |
| `PhysicsFixedJoint` | `JointKind::Fixed` |
| `drive:{angular,linear}:physics:*` | `Joint::stiffness`, `damping`, `spring_ref` |

### The body tree

A USD body's parent prim is usually not its parent *body*: `Scope` and `Xform` prims sit in
between, and a robot is commonly a flat list of links plus joints, with the articulation
carried entirely by `physics:body0`/`body1`. So the parent of a body is

1. its nearest rigid-body **ancestor prim**, when it has one;
2. otherwise the `physics:body0` of the joint that names it as `physics:body1`;
3. otherwise nothing — it is a root.

`body.pose` is then `parent_world⁻¹ ∘ body_world`, computed from the world poses, and a
contradictory file (two joints making each other's parent) is caught by
`SceneDesc::validate`'s cycle check rather than by a rule here.

### Conventions (spec 3.1)

Two conversions, applied once each, to every prim's **local** transform, which is then
composed down the hierarchy — exactly what `es-assets`' glTF importer does, and correct for
the same reason: conjugation is a homomorphism and rotation distributes over `Pose::compose`,
so converting per prim and converting once at the end agree.

1. **Units.** Every length is multiplied by `metersPerUnit`; every inertia by
   `metersPerUnit²`. Revolute limits and angular drive targets are degrees in the schema and
   become radians here.
2. **Axes.** USD `upAxis = "Y"` is right-handed Y-up — the same convention glTF uses — so the
   fix is the same fixed quaternion the glTF importer applies: `+Y → +Z`, `-Z → +X`, `-X → +Y`.
   Positions are rotated by it and orientations conjugated by it. Body-local quantities get the
   same rotation — centre of mass, joint anchor and axis, a `Cylinder`/`Capsule` `axis` token,
   mesh points — so every frame in the result is Z-up, not just the world one.
   `upAxis = "Z"` makes the fix the identity.

### Ids (spec 5.3)

`scene_id(kind, path)` over the prim path with the leading `/` stripped:
`/World/arm/link1` → `body/World/arm/link1`. Prim paths are unique by construction in USD, so
ids are unique without a counter, and a `SceneDesc` from a `.usda` has the same id scheme as
one from MJCF or glTF.

Mesh `AssetRef::hash` is blake3 over the converted `f32` points and `u32` indices — the content,
not the prim path — matching the glTF importer's rule, so the same mesh authored twice hashes
once.

## 5. What this deliberately does not do

- Materials, lights, cameras, shading, primvars, subdivision: parsed into `attrs`, ignored.
- `physics:velocity` / `angularVelocity` / articulation roots / collision groups / filtered
  pairs: `SceneDesc` has no home for initial state, so they are warnings.
- Time samples: a `SceneDesc` is one instant.
- Writing `.usda`. Export is out of scope for M4 W3.
- Handing back the decoded mesh payload. `stage_to_scene` returns the `AssetRef` with the
  triangles' content hash — enough for `scene_hash` (spec 5.3) and for a backend to key a
  cache — but not the vertex arrays themselves, the way `es-assets`' glTF importer returns
  `MeshData`. A consumer that needs the triangles re-reads `points` / `faceVertexIndices` from
  the `UsdStage`; widening the return type is a separate packet.

All of it is a `Warning` or an `Unsupported`, never a silent drop.
