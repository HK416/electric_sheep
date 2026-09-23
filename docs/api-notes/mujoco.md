# `mujoco` (Python), pinned to 3.13.0

Surface used by `crates/es-physics-backend/python/mujoco_ref.py`, the reference process behind
`MuJoCoCpuBackend` (spec 17.1). Checked against the wheel `mujoco==3.13.0` on CPython 3.12 /
3.13 (Windows x86-64) by running the crate's own oracle tests. Not a pinned dependency of the
workspace: the package is optional, and `MuJoCoCpuBackend::is_available()` reports its absence
so CI skips the oracle instead of failing (spec 2.4 — nothing on the runtime path needs
Python).

Anything in the 3.x line should work; refresh this file when the version the CI image installs
changes.

## Calls used

| Call | Signature as used | Notes |
|---|---|---|
| `mujoco.MjModel.from_xml_string(xml)` | `(str) -> MjModel` | Raises `ValueError` on a bad model; the message is forwarded verbatim as `PhysicsError::Backend`. |
| `mujoco.MjData(model)` | `(MjModel) -> MjData` | One per env; `n_envs` is emulated by holding a list of them. |
| `mujoco.mj_step(model, data)` | `(MjModel, MjData) -> None` | One physics tick. Called `n` times per `step(n)`. |
| `mujoco.mj_forward(model, data)` | `(MjModel, MjData) -> None` | After a reset or a state write, so that `sensordata` / `xpos` / `xquat` match `qpos`. |
| `mujoco.mj_resetData(model, data)` | `(MjModel, MjData) -> None` | Back to the model's initial state. |
| `mujoco.mj_id2name(model, objtype, id)` | `(MjModel, mjtObj, int) -> str \| None` | `None` for an unnamed element; the emitter always writes names, so a `None` here is a bug and surfaces as `PhysicsError::Protocol`. |
| `mujoco.mjtObj.mjOBJ_JOINT` / `mjOBJ_ACTUATOR` / `mjOBJ_SENSOR` / `mjOBJ_BODY` | enum | Only these four namespaces are looked up. |

## Fields read

`MjData`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat` — all `numpy` arrays of
`float64`. `xpos` is `(nbody, 3)` and `xquat` is `(nbody, 4)`; **`xquat` is wxyz** and is
reordered to the spec 3.1 xyzw before it crosses the wire.

`MjModel`: `nq`, `nv`, `nu`, `nsensordata`, `nbody`, `njnt`, `nsensor`, `jnt_type`,
`jnt_qposadr`, `jnt_dofadr`, `sensor_adr`, `sensor_dim`, `opt.timestep`.

`jnt_type` is `mjtJoint`: `0 = free` (7 qpos / 6 dof), `1 = ball` (4 / 3), `2 = slide` (1 / 1),
`3 = hinge` (1 / 1). The script carries that table itself rather than deriving widths from
address differences.

## Behaviour worth pinning

- **Sensor noise is off by default.** `noise` and `cutoff` on a sensor are written into the
  MJCF but `MuJoCo` applies noise only with the `sensornoise` enable flag, which this adapter
  does not set. Declared as a `BackendQuirk` on the backend's capabilities (spec 17.2).
- **`autolimits` defaults to true** in 3.x, so a joint `range` with no `limited` attribute is
  limited. The emitter relies on this and writes no `limited`.
- **`fullinertia` overwrites the inertial frame:** `MuJoCo` eigendecomposes it and sets the
  frame from the result, so `quat` and `fullinertia` cannot be combined. The emitter writes
  `quat` + `diaginertia` when the inertia is diagonal, `fullinertia` when the frame is
  identity, and refuses the remaining case by name rather than losing the rotation.
- A plane's third `size` component is the rendering grid spacing and must be positive.

## Meshes (packet M10/W2a)

Sources: the MuJoCo 3.13 [XML reference, `asset/mesh`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh),
[STL (file format) — Wikipedia](https://en.wikipedia.org/wiki/STL_(file_format)),
[fabbers.com STL format](https://www.fabbers.com/tech/STL_Format),
[Wavefront .obj file — Wikipedia](https://en.wikipedia.org/wiki/Wavefront_.obj_file),
gathered as `target/plan-w/w2a-research.md` and re-checked against real files here.

### Binary STL

`80-byte header ‖ u32 LE triangle count ‖ n × 50 bytes`, each facet `3 f32 normal ‖ 3 × 3 f32
vertices ‖ u16 attribute count`, all little-endian. So a binary file is exactly
`84 + 50·n` bytes, and that is the disambiguation `es_assets::stl::parse` uses: the spec says
the header "should never begin with `solid`", but that is a convention exporters break, so
sniffing the first word alone misreads real files. ASCII is the fallback, not the default.

The stored normal is advisory and is frequently `0 0 0`; the truth is the winding, which is
counter-clockwise seen from outside. The reader keeps no normals — the renderer derives them —
and instead swaps a facet's second and third vertices when a non-zero stored normal disagrees
with `(b−a)×(c−a)`.

### ASCII STL

`solid [name]` / `facet normal nx ny nz` / `outer loop` / three `vertex x y z` / `endloop` /
`endfacet` / `endsolid`. Whitespace is free-form. The reader keys on the `normal` and `vertex`
tokens and ignores the rest, which is enough for the grammar and tolerant of the layout.

### OBJ

`v x y z` (a trailing fourth number is `w`, or a colour, depending on the writer; both are
ignored here) and `f` in four forms — `a`, `a/vt`, `a//vn`, `a/vt/vn`. Indices are 1-based and
a negative one counts back from the vertices seen **so far in the file**, not from the final
count. A face may have more than three vertices and is fan-triangulated around its first.
`vt`, `vn`, `o`, `g`, `s`, `mtllib`, `usemtl` and comments are ignored: a mesh without its
material is still the right geometry.

### `<asset><mesh>`

| Attribute | Default | This pipeline |
|---|---|---|
| `file` | — | Read by the importer into `AssetRef.path`; **never emitted** — `scene_to_mjcf` writes `vertex`/`face` inline, because `PhysicsBackend::load` takes a `&SceneDesc` and no path channel (INV-17) and a machine path inside a hashed struct is what spec 5.3 forbids. |
| `vertex` / `face` | — | What the emitter writes. `face` is 0-based, counter-clockwise, `0..nvert-1`. Omitting `face` entirely makes MuJoCo build the mesh from the convex hull of the points; the emitter always writes it. |
| `scale` | `1 1 1` | Non-default is `MjcfError::Unsupported` by name. Baking it is a named follow-up. |
| `refpos` / `refquat` | `0 0 0` / `1 0 0 0` | Likewise refused by name: both transform the vertices before MuJoCo sees them. |
| `inertia` | **`legacy`** | Left at the default. See below. |
| `maxhullvert` | `-1` (unlimited) | Not represented; `report_unknown` warns. `so101.xml` sets `128` model-wide and `64` for three gripper meshes, so an emitted SO-101 runs qhull unbounded where upstream would cap it. |
| `normal` | — | Cannot be given for an STL mesh at all: MuJoCo generates normals for STL itself. |
| `texcoord` | — | Not from STL either. Textures are M7 R6. |

**Collision always uses the convex hull of the mesh**, whatever `inertia` says. The renderer
draws the surface, so for a concave mesh what is drawn and what is collided with differ; this
is declared as a `BackendQuirk` on `Feature::ContactMesh` rather than left implicit.

### `inertia="legacy"` vs `"exact"`, measured

The docs call `legacy` "not recommended" because it overcounts the volume of a *non-convex*
mesh, and keep it as the default for backward compatibility. On a convex closed mesh there is
nothing to overcount, and `mesh_box::legacy_and_exact_inertia_agree_on_a_convex_closed_box`
confirms it: the ±0.05 m box compiles to **bit-identical** `body_mass` and `body_inertia` under
both. The algorithm is not a source of error here.

What *is* a source of error is `f32`: a mesh vertex is `f32` in the file, in `MeshData` and in
the content hash, while a primitive's `size` is `f64`. The nearest `f32` to `0.05` is
`0.05000000074505806`, 1.49e-8 relative, and a volume is three of those — so the mesh box
weighs 4.47e-8 more than the identical primitive box (measured 1.0000000447034845 kg against
1.0000000000000002 kg) and its diagonal inertia is 7.45e-8 off. Both bodies still rest at
0.049892 m after 2,000 steps at 1 kHz, 7.5e-10 m apart. A tolerance tighter than ~1e-7 on a
mesh-against-primitive mass comparison is asking for `f64` vertices, which neither STL nor
`MeshData` carries.

### MuJoCo Menagerie

No single repository licence: the root `LICENSE` concatenates one block per robot, and GitHub
reports `NOASSERTION`. `franka_emika_panda` and `robotstudio_so101` are Apache-2.0;
`universal_robots_ur5e` is a 3-clause BSD from the ROS Industrial Consortium. Nothing is
vendored — `tests/fixtures/mjcf/*.PROVENANCE.json` pins a commit and the blake3 of each file,
and the tests fetch into `target/menagerie/<commit>/` in the upstream directory layout so
`meshdir` resolves. Raw URL: `https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/<commit>/<path>`.

| Model | Commit | Files | Measured |
|---|---|---|---|
| `robotstudio_so101` | `ac6b2b09983786f3036cab1000221017fa2193b4` | 19 binary STL (17.2 MB); `so101.xml` declares 18 of them | 362,996 triangles; every body's mass equals a direct upstream load bit for bit, including the mesh-derived `camera_mount` at 0.012 kg; 11 MB of inline MJCF |
| `franka_emika_panda` | `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7` | `panda.xml` + 8 STL collision + 59 OBJ visual | 136,590 triangles. Readers and hashes only: `panda.xml` has a `<tendon><fixed>` coupling the finger joints, which `scene_to_mjcf` refuses by name. `link5` has no STL — its collision shape is three `link5_collision_*.obj`. |

`universal_robots_ur5e` is the OBJ-only candidate if a MuJoCo load of an OBJ robot is ever
wanted: 19 OBJ files, no `<tendon>`, no `<equality>`. No Menagerie model in either set uses a
non-default `<mesh scale>`.

**Named follow-ups:** baking `<mesh scale>`; a `file=` path channel if inline text ever gets
too slow; carrying glTF's `MeshData` into `scene.meshes` the same way; `Feature::ContactMesh`
on MJWarp, unverified today; textures and materials (M7 R6).

## Protocol

`python -c "<embedded script>"`, one JSON object per line each way. Requests carry `cmd`;
replies are `{"ok": true, ...}` or `{"ok": false, "error": "<ExcType>: <message>"}` — a
modelling error is a value, never a dead process. Arrays are JSON lists of numbers, env-major.
`json.dumps` on a Python float and `serde_json` with `float_roundtrip` are both
shortest-round-trip, so `f64` values cross exactly; the bitwise run-to-run test depends on it.

| Request | Reply |
|---|---|
| `{"cmd":"load","mjcf":str,"n_envs":int,"timestep":float\|null,"seed":int}` | `nq, nv, nu, nsensordata, nbody, joints[{name,qpos:[adr,dim],dof:[adr,dim]}], actuators[name], sensors[{name,adr,dim}], bodies[name]` |
| `{"cmd":"reset","envs":[int]\|null,"state":{...}\|null}` | `{}` |
| `{"cmd":"set_ctrl","ctrl":[float]}` | `{}` |
| `{"cmd":"step","n":int}` | `{"nonfinite":[env]}` |
| `{"cmd":"state"}` | `qpos, qvel, act, sensordata, xpos, xquat` |
| `{"cmd":"set_state","state":{"qpos":[],"qvel":[],"act":[]}}` | `{}` |
| `{"cmd":"quit"}` | *(no reply; the process exits)* |

`ES_PYTHON` selects the interpreter; otherwise `python` then `python3` are tried.
