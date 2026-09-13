# API note — OpenUSD `.usda` text format and UsdPhysics, as read by `es-usd`

Pinned digest of the external surface `crates/es-usd` parses. Nothing here is a design
document (see `docs/design/usd-reader.md`); it records **what the format is** and **how far we
believe it**.

Every line is tagged:

- `verified` — read from the linked openusd.org page during the M4 W3 packet (2026-09).
- `unverified` — written from prior knowledge of the format and **not** confirmed against a
  primary source. A fixture proving our parser accepts a shape we invented is not verification.

USD is a cuttable dependency (spec 1.9 item 6): the supported path is the Python **USD Bake**
(spec 2.5), and this reader is the honest-but-minimal native alternative. Anything outside the
subset below is a hard `UsdError::Unsupported` carrying the prim path — never a silent guess.

Sources fetched:

- <https://openusd.org/release/tut_helloworld.html>
- <https://openusd.org/release/glossary.html>
- <https://openusd.org/release/api/class_usd_geom_xformable.html>
- <https://openusd.org/release/api/class_gf_quatf.html>
- <https://openusd.org/release/api/usd_physics_page_front.html>

---

## 1. File shell

| Item | Form | Status |
|---|---|---|
| Magic header | `#usda 1.0` as the first non-blank line | **verified** (hello-world tutorial) |
| Layer metadata | parenthesised block immediately after the header | **verified** that `upAxis`, `metersPerUnit`, `defaultPrim` are layer metadata; the exact `(` ... `)` placement is **unverified** |
| `upAxis` | `token`, `"Y"` or `"Z"`; fallback `"Y"` | value set **verified**; the `"Y"` fallback is **unverified** |
| `metersPerUnit` | `double`; fallback `0.01` (centimetres) | fallback **unverified** |
| `kilogramsPerUnit` | not read; masses are assumed kg | **unverified** |
| `defaultPrim` | `string`/`token` naming a root prim | **unverified** |
| `.usdc`, `.usdz` | binary crate / zip container | existence **verified** (glossary); we never parse them |

The reader warns when `upAxis` or `metersPerUnit` is absent rather than assuming silently,
because the fallback is the one item above that would corrupt a scene by two orders of
magnitude if we guessed wrong.

## 2. Prims

```usda
def Xform "hello"
{
    def Sphere "world"
    {
    }
}
```

**verified** (hello-world tutorial): `def <TypeName> "<name>" { ... }`, nesting by braces.

| Item | Status |
|---|---|
| Specifiers `def`, `over`, `class` | **verified** (glossary: "three possible specifiers") |
| Type name may be omitted (`def "name"`) | **unverified** |
| Prim path is `/` + the `/`-joined chain of prim names | **unverified** |
| Per-prim metadata block `( ... )` between the name and `{` | **unverified** |

Type names we map: `Xform`, `Scope`, `Mesh`, `Cube`, `Sphere`, `Cylinder`, `Capsule`,
`PhysicsRevoluteJoint`, `PhysicsPrismaticJoint`, `PhysicsFixedJoint`. Any other type name is
parsed, kept in the stage, and reported as a warning — not an error, because an unknown prim
type is usually a light or a material that carries no physics.

## 3. Attributes and relationships

Declaration is `[uniform|custom]* <typeName> <name> = <value>`; a relationship is
`rel <name> = </Prim/Path>`. **unverified** as a grammar; the individual spellings below come
from the schema pages listed and are marked there.

Value literals accepted:

| Literal | Example | Status |
|---|---|---|
| bool | `true` | **unverified** |
| int / float / double | `-3`, `1.5`, `1e-3` | **unverified** |
| token / string | `"X"` | **unverified** |
| tuple | `(0, 0, 1)`, `(1, 0, 0, 0)` | **unverified** |
| matrix4d | `( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )` | **unverified** |
| array | `[(0,0,0), (1,0,0)]`, `[0, 1, 2]` | **unverified** |
| asset | `@./mesh.usda@` | **verified** in the reference syntax `@file1.usd@` (glossary) |
| rel target | `</World/link0>` | **unverified** |

Array type names carry `[]`: `float3[] points`, `int[] faceVertexIndices`,
`uniform token[] xformOpOrder`. **unverified**.

Unknown attributes are preserved verbatim in `Prim::attrs` rather than dropped, so a later
packet can widen the mapping without touching the parser.

### Geometry attributes

| Attribute | Prim | Meaning | Status |
|---|---|---|---|
| `points` | `Mesh` | `float3[]` vertex positions | **unverified** |
| `faceVertexIndices` | `Mesh` | `int[]` | **unverified** |
| `faceVertexCounts` | `Mesh` | `int[]`, vertices per face | **unverified** |
| `size` | `Cube` | full side length, default `2.0` | **unverified** |
| `radius` | `Sphere`, `Cylinder`, `Capsule` | default `1.0` | **unverified** |
| `height` | `Cylinder`, `Capsule` | default `2.0`; for `Capsule` this is the *cylindrical* section, excluding the caps | **unverified** |
| `axis` | `Cylinder`, `Capsule` | token `"X"`/`"Y"`/`"Z"`, default `"Z"` | **unverified** |

### Transform ops

**verified** (`UsdGeomXformable`):

- op names `xformOp:translate`, `xformOp:scale`, `xformOp:orient`, `xformOp:transform`, the
  per-component `xformOp:translateX`/`scaleY`/..., `xformOp:rotateX/Y/Z` (degrees) and the six
  Euler variants; custom suffixes are `xformOp:<type>:<suffix>`.
- `xformOpOrder` lists the ops. The doc's words: each successive op is applied "more locally"
  than the preceding one — so the **first-listed op is the outermost**, i.e. with
  column-vector convention `M = op[0] * op[1] * ... * op[n-1]`.
- `"!resetXformStack!"` in `xformOpOrder` means the prim does not inherit its parent's
  transform; everything up to the last occurrence is ignored.

**unverified**: that an `xformOp:*` attribute not named in `xformOpOrder` contributes nothing.
We follow that reading (it matches `GetOrderedXformOps`) and warn when a prim authors ops it
never orders.

We implement `translate`, `orient`, `scale` and `transform` only; the component-wise and Euler
ops are parsed into `attrs` and reported as an unsupported-op warning if ordered.

`xformOp:orient` is a `quatf`/`quatd`. **verified** that `GfQuatf(float real, float i, float
j, float k)` puts the real part first; **unverified** that the `.usda` text serialisation is
therefore `(w, x, y, z)`. We read `(w, x, y, z)` and store spec 3.1 `xyzw`.

## 4. UsdPhysics

All of this section is **verified** against
<https://openusd.org/release/api/usd_physics_page_front.html>, except where noted.

API schemas appear in the prim metadata block, e.g.
`prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]` (the list spelling is
**unverified**).

| Schema | Attributes we read |
|---|---|
| `PhysicsRigidBodyAPI` | marker only; `physics:velocity`, `physics:angularVelocity` (deg/s), `physics:kinematicEnabled` exist and are kept in `attrs` |
| `PhysicsCollisionAPI` | marker only; `physics:collisionEnabled` (bool) |
| `PhysicsMassAPI` | `physics:mass` (float), `physics:centerOfMass` (point3f), `physics:diagonalInertia` (vector3f), `physics:principalAxes` (quatf), `physics:density` (double) |

Joints — `PhysicsRevoluteJoint`, `PhysicsPrismaticJoint`, `PhysicsFixedJoint`:

| Attribute | Type | Note |
|---|---|---|
| `physics:body0`, `physics:body1` | rel | body0 is treated as the parent, body1 as the child (**unverified**) |
| `physics:localPos0`, `physics:localPos1` | point3f | anchor in each body's frame |
| `physics:localRot0`, `physics:localRot1` | quatf | |
| `physics:axis` | token | `"X"`, `"Y"` or `"Z"` |
| `physics:lowerLimit`, `physics:upperLimit` | float | **degrees** for revolute, distance units for prismatic |

`PhysicsDriveAPI` is multi-apply, one instance per degree of freedom:
`drive:<dof>:physics:stiffness`, `:damping`, `:targetPosition`, `:targetVelocity`. We read the
`angular` instance for a revolute joint and `linear` for a prismatic one; a drive on any other
dof token is a warning.

The degree-based revolute limits and targets are the single most valuable verified fact in
this note: reading them as radians would silently shrink every joint range by 57x.

## 5. Rejected outright

Each of these is `UsdError::Unsupported { path, feature }` naming the prim (or `/` for a layer
metadatum), because each one changes what the file *means* and pretending otherwise would
produce a plausible, wrong scene:

| Feature | Why it is refused |
|---|---|
| `references`, `prepend references` | composition; needs a resolver and a layer stack |
| `payload` | ditto, deferred |
| `variantSet` / `variants` | selection-dependent scene |
| `inherits`, `specializes` | composition |
| `.usdc`, `.usdz` inputs | binary crate format, out of scope |
| `subLayers` | composition |
| time samples (`attr.timeSamples = { 0: ... }`) | a scene is a single time |

The route for any of them is USD Bake (spec 2.5): flatten in Python, re-export `.usda`.
