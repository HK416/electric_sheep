# Scene authoring — build a task from scratch in the editor (S4)

Spec: §5.3 (hash scheme), §6 (Task IR, `SceneRef`), §14 (authoring frontends, storage formats),
§17.2 (semantic mapping report), §23 (editor), §4.2 (layers). Design notes this builds on:
`editor-redesign.md` (§2 decisions, §3 the ① screen, §5 the S4 direction), `renderer.md` §15–16
(textures, PBR), `usd-reader.md`, `python-builder.md`, `node-sdk.md`.

## 1. Why, and the owner's decisions

Owner, 2026-10-01, after the Shadow Hand had to be ported by hand (plan H, H1): "장면에서 구성
요소를 추가 삭제가 가능한가? es-editor의 지향점은 게임엔진 처럼 쉬운 로봇 학습 사용인데 장면을
따로 준비해야 한다면 별로일 것 같아." Today ① is read-only, a scene is a hand-written MJCF a
template points at, and the five IR documents are written by generators in test files. A person
who does not write MJCF and Rust cannot make a new task.

| Question | Decision (owner, 2026-10-01) |
|---|---|
| Scope of the first design | **Editing and task definition together**: scene editing in the viewport, the success/failure sentences, reset and randomization, and generation of the five IR documents — one design, several packets |
| The scene's source of truth | **A new scene document** (`*.esscene`), exported to MJCF; not MJCF itself |

Everything else below is a proposal with a default; section 9 lists what the owner still decides.

## 2. What a person does (the ① and ② screens of `editor-redesign.md` §3, made concrete)

1. **Start**: the start screen's *empty project* card (or *save as template* from any project).
   An empty project opens with a floor, a light and an outside camera.
2. **Build the scene** in ①: *Add* → robot (from the library: SO-101, Shadow Hand; or a URDF /
   MJCF / USD / glTF file), object (box, sphere, cylinder, capsule, or a mesh file), camera,
   light, region (an invisible box or zone a sentence can name: "the bin", "the target area").
   Select in the viewport or the hierarchy; move / rotate / scale with gizmos; the inspector
   edits name, shape, size, mass, friction, colour / material / texture, joint (fixed, free,
   hinge, slide) and its limits. Delete, duplicate, undo / redo. A *physics preview* button
   drops everything for three seconds and plays it back. The policy's camera view is always in
   the viewport's corner (the real observation, H8's looks drawn in process since H9).
3. **Say what success is** in ①'s sentence editor: "[cube] is [inside] [bin]", "[cube] is
   [still] for [1 s]", "[cube] matches [goal]'s orientation within [6°]", "fails if [cube] is
   farther than [24 cm] from [palm]", "fails if not done within [8 s]". What starts where:
   "at the start [cube] is placed at random within [the table area] (🎲)", one randomization
   strength (low / medium / high) plus a 🎲 per property.
4. **Say what the policy sees**: which cameras (and their resolution), which joints; plain
   checks run as you go ("the front camera does not see the cube at its start position").
5. **② Teach** as today: action blocks (S3, needs IK — SO-101 now), a teacher trained by reward
   (the method plan H built: the reward comes from the same sentences), teleoperation (S6).
6. **③ / ④ / ⑤** exactly as for a template project.

Nothing here runs learning or physics inside the editor (§23.1, §4.2 rule 4): previews and
generation are `es` subcommands the editor hands an argv, or pure functions of the documents.
Rendering is the one exception, by the owner's decision of 2026-10-01 ("에디터도 Vulkan 으로
그려줘", after the material look dropped to flat colour whenever the camera moved): the viewport
draws with `es-render` in process on one Vulkan device (packet M16/H9,
`editor-redesign.md` section 5, S4), and `es render` stays for a machine without one.

## 3. The scene document (`*.esscene`)

A TOML file in the project, the person's scene. It is an authoring format like `task.toml`
(§14.3), not a new IR: it is read into the same `SceneDesc` every other reader produces, so
**`scene_hash` and `asset_hash` are those of the `SceneDesc`, independent of the file format**
(today's definition, §5.3). `SceneRef.path` names the `.esscene` (the path is hash input as
today).

```toml
kind = "scene"
schema = 1

[physics]                       # SceneDesc::options; absent = MuJoCo's defaults as today
timestep = 0.008333333333333333
integrator = "implicitfast"

# A robot or any multi-body asset is a *reference* to its own file, placed with a pose and a
# name prefix; the reader expands it into the SceneDesc (section 3.2).
[[include]]
name = "hand"
source = "robots/shadow_hand/shadow_hand.xml"   # MJCF, URDF, USD or glTF
pos = [1.0, 1.25, 0.15]
quat = [0.0, 0.0, 0.0, 1.0]     # xyzw (spec 3.1)

[[body]]
name = "cube"
pos = [1.0, 0.867, 0.177]
joint = "free"
[[body.geom]]
shape = { box = [0.03, 0.03, 0.03] }
mass = 0.216
friction = [1.0, 0.0, 0.0]
material = "block"

[[texture]]
name = "block"
file = "textures/block.png"
kind = "cube"
gridsize = [3, 4]
gridlayout = ".U..LFRB.D.."

[[material]]
name = "block"
texture = "block"
roughness = 0.6

[[camera]]
name = "top"
pos = [1.04, 0.88, 0.56]
quat = [0.0, 0.0, 0.0, 1.0]
fovy = 45.0

[[light]]
name = "ceiling"
kind = "area"
pos = [1.0, 0.9, 0.9]
size = [0.6, 0.6]
intensity = 1.0

[[region]]                      # a site: no collision, no mass; sentences name it
name = "target_area"
size = [0.1, 0.1, 0.01]
pos = [0.25, 0.0, 0.0]
```

### 3.1 Rules that keep the hash chain honest

- **Orientations are stored as quaternions** (the bits that enter `scene_hash`); the inspector
  shows Euler degrees and converts with `es_math::approx` (the asset path's rule, §5.3 /
  M10 W0b), so no host `libm` reaches a hash.
- **Identity is the name path** (`StableId::from_path`, as MJCF bodies today): a body's id is
  its path in the tree. A rename is a refactor: the editor shows which documents name the
  thing and regenerates them (sections 4.2 and 5); it never leaves a dangling id.
- **Assets by content**: a mesh or texture file is copied into the project's `assets/` under
  its blake3 and referenced by that path; `asset_hash` is by content (HT1's rule), so moving a
  project does not move a hash.
- **Additive**: a field left out reads as `SceneDesc`'s default; a document that writes only
  what an existing MJCF says reads to the *same* `SceneDesc` — the oracle in section 8.

### 3.2 Includes (robots and other assets)

An `[[include]]` is expanded by the existing reader of its format (MJCF, URDF, USD, glTF) into
bodies, joints, actuators, sensors, tendons, contact pairs, materials and assets, prefixed with
the include's name, placed under a fixed or free root at the include's pose. The included file
is not copied into the scene document; the project keeps a content-hashed copy under `assets/`.
Per-instance overrides (a joint's range, an actuator's gain, a material) are an `[include.set]`
table addressed by the asset's own names — game engines' prefab overrides. Why a reference, not
a flattened copy: the robot's file stays the one place its kinematics live, a fix to it reaches
every scene that uses it (with the hash moving visibly), and the document stays small enough to
read and diff.

### 3.3 Export

`es scene export <file.esscene> --mjcf <out.xml>` writes the expanded `SceneDesc` as one
self-contained MJCF (cameras, lights, materials, textures, tendons, pairs, gravcomp — everything
`SceneDesc` carries) with its assets beside it. It is a full-fidelity writer in `es-assets`
(layer 2), not `es-physics-backend`'s `mjcf_out` (which by design drops what carries no
dynamics). URDF and USD export are later items.

### 3.4 What G1 settled (the schema as built)

`crates/es-assets/src/esscene/` (`EsScene::from_toml` / `to_toml`, `expand`), oracle
`crates/es-assets/tests/esscene.rs`. Where it differs from the example above, and what the text
above left open:

- **Top level**: `kind = "scene"`, `schema = 1`, `name` (`SceneDesc::name`, hash input; absent is
  `"scene"`), `[physics]` (every field optional, absent is MuJoCo's default; `integrator` is
  `euler|rk4|implicit|implicitfast`), `[[geom]]` (static scenery on the world body: a table, a
  floor, a bin). Every optional field is an `Option`, so writing back writes only what was
  written (read ∘ write is the identity, a property test).
- **Includes**: `name` is the include's handle in the document (a task names its robot by it);
  the name prefix is a separate `prefix`. Absent, the file's own names **and ids** are kept (why
  the mirror oracle holds); present, it is prepended to every name and each id is re-derived by
  the MJCF name path scheme. The file's root bodies go under the world (no wrapper body); a
  non-identity `pos` / `quat` is composed onto the root bodies and world-level elements. The mesh
  and texture paths the file names are re-based onto the document's directory. The file's
  `<option>` and model name are not read (the scene's are the document's `[physics]` and
  `name`). `[include.set.joint.<name>]` (`range`, `damping`, `armature`, `stiffness`,
  `frictionloss`), `[include.set.actuator.<name>]` (`kp`, `kv`, `ctrlrange`, `forcerange`),
  `[include.set.geom.<name>]` (`rgba`, `material` — the file's material or the document's); an
  unknown target is refused by field. MJCF, URDF and glTF are read; **USD is refused**: `es-usd`
  is a layer-2 sibling and yields a stage, not a `SceneDesc` (a USD include is a later item).
- **Bodies**: `parent` names a body defined before (an included one too), absent is the world.
  `joint = { kind, name, axis, pos, range, damping, armature, stiffness, frictionloss, springref }`
  — `kind` is `fixed|free|ball|hinge|slide` (`fixed` and absent weld, emitting no joint, as in
  MJCF), `name` absent is the body's name. `inertial = { mass, pos, quat, diaginertia |
  fullinertia }`, `gravcomp`. A `[[body.geom]]`'s `shape` is `{ plane | sphere | capsule |
  cylinder | box | ellipsoid = sizes }` or `{ mesh = "file" }` (the asset is named by the file's
  stem); an unnamed geom is `geom<n>`, as in MJCF.
- **Textures are named `[[texture]]`s** and a material names one per slot (`texture`, `orm`,
  `metallic_map`, `roughness_map`, `normal_map`, `emissive_map`): one texture is shared by two
  materials (the Shadow Hand's cube and goal), so it cannot be inline. As in MJCF, a material
  that writes only `rgba` is a name, not a drawn material.
- **Cameras**: `fovy` in degrees (MJCF's conversion), absent `parent` is fixed to the world.
- **Lights**: what today's renderer takes — a thin emissive box on the world named
  `<name>_light` (`size` the x / y half-extents, half thickness 0.005 m,
  `rgba = [rgb × intensity, 1]`, no collision). `kind` is `area`, the one kind today.
- **Regions** are sites; `SceneDesc::Site` has no shape, so a region has a `size` (half-extents),
  not a `shape`.
- **Quaternion bits**: a quaternion already canonical (unit within 1e-12, `w ≥ 0`) is stored as
  written, bit for bit; anything else is normalised by `Quat::normalize`. The editor writes
  `SceneDesc`'s bits back, so a round trip moves none.
- **Order**: the world body, the includes in turn, then the document's own entities — world geoms
  and lights, textures and materials, bodies, cameras, regions — each in document order. The
  mirror oracles (SO-101, Shadow Hand) check every value and order of the `SceneDesc`, the
  `scene_hash` (the digest the committed documents carry) and the asset content hashes. The one
  list compared as a set is `assets`: the Shadow Hand file interleaves the cube's texture and
  materials with the hand's while an include's assets come before the document's, and nothing
  reads that order (`scene_hash` sorts it, the loaders and the renderer look up by id). The
  Shadow Hand's floor stays in the hand file: its empty `floor0` body precedes the hand in
  MuJoCo's body order. Asset paths are hash input, so the Shadow Hand document sits beside the
  source file (`tests/fixtures/mjcf/shadow_hand/`).
- `SceneRef.asset_hash` (in the committed documents, blake3 of the scene *file*'s bytes, M11's open
  decision) necessarily differs for a different file; what is equal is the `SceneDesc`'s per-asset
  hashes. What it holds for an `.esscene` (whether it also covers the included files) is G3's to
  decide when it generates documents.

## 4. The task specification (`*.estask`) and `es project generate`

A second person-facing file, beside the scene: **what the task is, in the sentence editor's
terms**. Like `teach.toml` (S3) it is a recipe, not an IR (§5.1 rule 6): it names objects of the
scene and relations from a fixed vocabulary, and `es project generate` compiles it with the
scene into the five IR documents, the training recipes and the cycle. Same inputs, same bytes
(the generator is deterministic and has no clock).

```toml
kind = "task-spec"
schema = 1
scene = "scene.esscene"
robot = "hand"                       # an include; its actuators become the ActionSpec
control_hz = 60

[success]                            # all clauses must hold
clauses = [
  { subject = "cube", relation = "orientation_matches", object = "goal", within_deg = 5.73 },
]
[failure]                            # any clause ends the attempt as a failure
clauses = [
  { subject = "cube", relation = "farther_than", object = "palm_ref", m = 0.24 },
]
timeout_s = 8.0

[start]                              # reset; 🎲 = randomized, strength scales the ranges
strength = "medium"
items = [
  { what = "cube.yaw", dice = true },
  { what = "goal.yaw", dice = true },
  { what = "hand.joints", noise = 0.2 },
]

[observe]
cameras = ["top", "front", "side"]   # the student's views; resolution per camera
camera_px = 96
render = { path = "pt", spp = 32, bounces = 3, exposure = 8 }
state = ["hand.joint_pos", "goal.qpos"]
privileged = ["cube.pose", "cube.vel"]     # the teacher's extra inputs (plan H)

[reward]                             # derived from the sentences; weights are check boxes
success = "a lot"
shaping = ["orientation", "distance"]
```

### 4.1 The vocabulary is what `es-env` lowers

Each relation compiles to Task IR nodes that `es-env`'s reward / termination cone lowering
(`crates/es-env/src/plan.rs`) reads: the sources `GetJointState` (position, velocity),
`GetBodyPose` (`pos`, `quat`), `GetBodyVelocity` (a free body; since G3c), `GetSensor`, `GetTime`
— world frame only; the transforms `Arith`, `Norm{L2}`, `Dot`, `MathFn{Abs,Sqrt}`, `Compare`,
`Logic`, `Normalize`, `Clamp` (one lane) and, since G3c, `Slice`, `Concat`, `Reduce` (axis 0);
the sinks `Reward` and `Terminate`; and the reset's `ResetState` and `Randomization`. Refused by
name inside a cone: `GetContact`, `GetRandom`, `Transform`, `Cross`, `Select`, `Norm{L1,Linf}`,
the other `MathFn`s.

| Relation | Compiles to |
|---|---|
| `inside` a region | per axis: `Slice` of the body's position, `Compare >` the box's low face and `<` its high face, `And`; the three axes `And`ed (section 4.4) |
| `inside` a range | the scalar subject, `Compare >` lo and `<` hi, `And` |
| `above` / `below` a value | the scalar subject, `Compare` |
| `above` / `below` a body (by m) | `Slice` z of both positions, `Arith Sub`, `Compare > m` |
| `near` / `farther_than` (m) | `Arith Sub`, `Norm L2`, `Compare` |
| `still` (for s) | a body: `GetBodyVelocity.linear`'s `Norm L2 <` speed (and `.angular`'s under `angular`); a coordinate: its velocity within ±speed — IR-D has no hold node, so "for 1 s" compiles to "inside and nearly still" (the demo's settling bound, `editor-redesign.md` §5 S4) |
| `orientation_matches` (deg) | `Dot` of quaternions, `MathFn Abs`, `Compare ≥ cos(θ/2)` (plan H's construction) |
| `joint` `above` / `below` (gripper open) | `GetJointState`, `Compare` |
| `touches` | waits for `GetContact` lowering — offered when it lands |

Reward shaping per clause: distance and orientation clauses yield the dense terms plan H used
(`−k·distance`, `1/(√(8(1−d))+0.1)` as a piecewise-linear curve), the success clause a bonus,
the failure clause a penalty; the check boxes choose the weights from three levels. **Why an
attempt failed** is then the clause that was missing (the owner's approved S4 direction): `es
eval run` records each clause's truth at the end of each attempt in `episodes.json`, and ⑤
names it in the words of ①'s sentence. The template `[outcome]` tables (cube-into-bin,
reorient) become generated from the spec.

### 4.2 What `es project generate` writes

From `scene.esscene` + `task.estask`: `task.toml` (Task IR), `observation-teacher.toml`,
`observation-student.toml`, `learning-*.toml` (the template families that exist: the state MLP
teacher, the three-view ACT student — `tanh` heads, H6's invariant), `deployment.toml` (the
envelope from the scene's joint and control ranges), `evaluation*.toml` (held-out seeds, the
suites that apply), `training-*.toml`, `cycle.toml`. Every hash is derived, none typed. The
committed generators in `crates/es/tests/{views,shadow_hand}.rs` become regression oracles:
the SO-101 and Shadow Hand specs regenerate their committed documents byte for byte.

### 4.3 What G3a settled (the schema as built)

`crates/es-script/src/spec/` (`TaskSpec::from_toml` / `to_toml`, `compile_task(spec, root)`),
oracles `crates/es-script/tests/estask*.rs`. The two specs
`tests/fixtures/estask/{shadow_hand_repose,so101_views}.estask` compile to the `task_hash` of
`tests/fixtures/shadow-hand/task-repose.toml` and `tests/fixtures/visible-learning/task-views.toml`
(the semantic hash; `task_graph_hash` differs: the compiler numbers nodes in the spec's order,
the committed documents in their generators' — SO-101's in its history, the gripper clause
appended at 31–33). Where it differs from the example above, and what the text left open:

- **Top level**: `scene` is relative to the project root `compile_task` is given and is written
  into `SceneRef.path` as written; `asset_hash` is blake3 of the scene file's bytes for every
  kind, an `.esscene` too (what an include brings is in `scene_hash` through the `SceneDesc` and
  its per-asset content hashes). `robot` names the robot's **root body**, not an include handle
  (an include can have several roots: the Shadow Hand file's `floor0`; G3b also takes the handle,
  section 4.5); its subtree's joints are
  `robot.joints`, and every actuator of the scene is its action (`JointPosition`, one robot per
  task). `timeout_s × control_hz` must be whole: it is `max_episode_steps`, and
  `time since reset ≥ timeout_s` is the `Timeout` node.
- **Clauses** (`[success]` all, `[failure]` any): `{ subject, relation, ... }`, folded in
  document order with `And` into one `Terminate(Success)` and with `Or` into one
  `Terminate(Failure)`. A **scalar** subject (`inside`, `above`, `below`, `still`) is a joint (a
  robot joint is read through the robot's root body, as the robot's joint vector; another joint
  through itself) or `<body>.x`, the first coordinate of the body's free joint — the one lane
  `GetJointState` reads. A **body** subject (`near`, `farther_than`, `orientation_matches`) is a
  body. Fields: `inside` `range = [lo, hi]`; `above` / `below` `value`; `still` `speed` (the
  velocity within ±speed); `near` / `farther_than` `m` and either `object` (a body: `Arith Sub`,
  `Norm`) or `point = [x, y, z]` (no constant node, so `p − point` is a per-lane `Normalize` over
  `[point − 1, point + 1]`, plan H's construction, and `m` < 1); `orientation_matches` `object`
  and `within_deg` (`|q·g| ≥ cos(θ/2)`). A field a relation does not take, or a missing one, is
  refused naming the clause (`success[1] (cube.x still)`) and the field; unknown keys by name.
- **What did not lower at G3a was refused by name**: `touches` (`GetContact`); `<body>.y` / `.z`,
  and with them `inside` a region, `above` / `below` another body and a 3-axis `still` — the
  cone lowering had no `Slice`, `Concat`, `Reduce` or `GetBodyVelocity`, so `inside` was the
  subject's x span and `still` its x velocity, exactly task.toml's ceiling ("the bin's x span
  plus a settling bound"). G3c lowered the four nodes and made the relations 3-axis (4.4);
  `touches` still waits for `GetContact`.
- **Shaping**, per clause, named where the references needed more than 4.1: `shaping` is the
  term's form, `weight` its weight, `term` its name (absent `<subject>_<shaping>`): `distance`
  (`near` / `farther_than`: `weight × distance`, clamped to [0, 1] m), `ramp` (`inside`, with
  `ramp = [a, b]`: `weight × (s − a)/(b − a)` clamped to [0, 1] — SO-101's `cube_towards_bin`),
  `inverse_angle` (`orientation_matches`: `weight / (s + 0.1)`, `s = √(8(1 − |q·g|))`, as its
  piecewise-linear interpolant at plan H's nine knots, terms `<term>_0..7`, plus `<term>_floor`
  paid every step). `[reward]`: `scale` multiplies every weight (rl_games' `scale_value`, the
  Shadow Hand's 0.01); `success` / `failure` are the sparse terms of those names, fed by the
  folded predicate. Weights are numbers in the document; the check boxes' three levels are the
  sentence editor's (G8) to write as numbers.
- **Start** items `{ what, ... }`: `robot.joints` (`noise` is the fraction of each joint's
  range, Isaac Lab's `reset_dof_pos_noise`; `coupled = true` lets a joint coupled to another by
  a two-joint fixed tendon draw from the other's stream, scaled by the coupling — plan H's
  J0/J1), a joint, `<body>.x|y|z` (`value`, `value ± noise`, or `range`), or
  `<body>.orientation` with `draw` (below). `value` is a `Constant` (also at zero noise), `range`
  a `Uniform` always (task.toml's `Uniform{0, 0}` home pose). `stream` names the draw (absent:
  `what`, or `reset.<joint>`); items on one stream share its draw. `dice = true` (🎲) makes a
  `Randomization` node instead of a `ResetState` (task.toml's cube x / y); `strength` scales
  every 🎲 item's `range` about its centre and its `noise` (absent: as written).
- **Orientation draws** (the owner, 2026-10-01: switching the Shadow Hand from yaw-only to
  flipping the cube is one sentence). A reset node writes one number, so each quaternion lane is
  one draw (lanes on one stream share it), and the backend normalizes what is written — MuJoCo
  and MJWarp, `xquat` at once and `qpos` after the first step (H2b; again for `any` by
  `the_backends_normalize_a_drawn_quaternion`):
  - `draw = "yaw"` (+ `tilt`): plan H's `(1, t, −t·u, u)`, one `u ~ U(−1, 1)`: the resting tilt
    `2·atan(t)` about world X and a yaw `2·atan(u)` over the half turn [−90°, 90°].
  - `draw = "tilt"`, `tilt_max_deg = θ` (< 180): `(1, a, b, g)`, `a, b ~ U(−m, m)`,
    `m = tan(θ/2)/√2`, `g ~ N(0, 1)`. The tilt from vertical never exceeds θ
    (`a² + b² ≤ tan²(θ/2)·(1 + g²)`, reached at `g = 0` and the corners) and every heading is
    drawn (`2·atan(g)`: 68 % within ±90°, 8 % beyond ±120°). This is the closest exact
    construction, not a uniform one (neither over the cap nor in yaw): a tilt bound needs the yaw
    pair `(w, z)` away from zero, and no bounded draw of that pair covers every heading (its set
    is a box, and a convex set whose directions span a half turn reaches the origin), so the yaw
    lane is unbounded instead.
  - `draw = "any"`: four `N(0, 1)` lanes on `<stream>.w|x|y|z`, normalized by the backend:
    uniform over SO(3), to the precision of `EnvRng`'s Box–Muller.
- **Observe**: `cameras` (channel `rgb_<camera>`, `camera_px` square RGB8, the `ImageSpec` the
  renderer delivers at the control rate) and one `render` for all; `state` and `privileged` are
  **tables** `channel = "source"`, not lists — channel names are hash input and the reference
  documents' follow no one rule (`object_vel`, `target_qpos`, `sim_cube_pose`). Sources:
  `robot.joint_pos` (`JointState` on the root body: the leading `dof`, refused unless the
  robot's joints lead the scene's), `robot.joint_vel` (`JointState` on the first joint: one input
  buffer per id), `robot.previous_action` (`initial` each ctrlrange's centre), `<body>.pose`
  (`BodyPose`), `<body>.qpos` / `<body>.vel` (the free joint's 7 / 6). `state` and `privileged`
  (the teacher's extra inputs) are one `ObservationSpec` in the Task IR; G3b reads the split.
- **Numbers from libm**: `within_deg`'s cosine, a camera's focal length and `tilt_max_deg`'s `m`
  come from `es_math::approx` (`sin_cos_f64`, `tan_f64` — the `libm` crate, musl), as the asset
  path's numbers do (§5.3, M10 W0b), so **a generated document has the same bits on every host**
  (G3d, 2026-10-01; G3b had kept the host's `cos` / `tan`). No committed hash moved: at the
  committed angles (the cosine of 0.05 rad, `tan` of 22.5° and 35°) musl and this PC's UCRT give
  the same bits, both correctly rounded. Elsewhere the two are faithful, not equal: over 1°–179° in
  0.001° steps `tan` differs by one ULP at 14,720 of 356,002 arguments (`x` and `x / 2`) and the
  half-angle cosine at 4,902 of 178,001, neither side always the correctly rounded one — a
  document generated on Windows before G3d with, say, a 68° camera differs from today's in its
  focal length's last bit. (`tan` as sin / cos was not taken: it moves the 45° cameras' focal
  length by one ULP, `115.88225099390856` against the committed `…857`.) What still calls the
  host: the fixture generators in `crates/es/tests/{shadow_hand,views}.rs`, which G3b's tests
  compare against and which agree at the committed angles.

### 4.4 What G3c settled (three-axis relations)

`crates/es-env/src/plan.rs` (oracle `slice_concat_reduce_and_body_velocity_lower`) and the clause
functions of `crates/es-script/src/spec/compile.rs` (oracles `crates/es-script/tests/estask*.rs`,
fixture `tests/fixtures/estask/so101_region.esscene`). Both reference specs still compile to
their committed `task_hash`; no committed cone contained the four nodes (they were refused), so
every committed document lowers to the same `Expr` as before.

- **The four nodes in a cone.** A cone's value is one row of lanes, so axis 0 is the only axis;
  another, or a slice past the end, is refused by name. `Slice` takes lanes `start..start+len`,
  `Concat` its inputs `in0, in1, …` in order, `Reduce` folds in lane order — the association
  `Norm` and `Dot` already used (`DET-020`) — with `Mean` that sum divided by the lane count.
  `unordered = true` lowers the same way (lane order is one admissible order; the validator
  refuses it in deterministic mode, `DET-030`).
- **`GetBodyVelocity` is derived exactly from what `StateView` carries**: it has no
  body-velocity array, but a free body's six `qvel` are MuJoCo's free-joint convention — the
  body origin's linear velocity in the world frame, then the angular velocity in the body frame
  — which every backend's view follows (`physx_ref.py` converts PhysX's to it). `linear` is the
  first three lanes as they are (the ports `GetJointState(Velocity)` binds: `cube.x still` and
  `cube still` read the same numbers); `angular` is the last three rotated into the world by
  `xquat` (the orientation `GetBodyPose.quat` reads), `v + w·t + u × t`, `t = 2 u × v` —
  polynomial, no `DET-010` function. On `mujoco-cpu`, a cube tilted 60° spinning at 5 rad/s
  about its own z reads `5 R e_z` within 2e-15 (`a_spinning_cubes_angular_velocity_is_read_in_the_world_frame`;
  read as the world's it would be `(0, 0, 5)`). A body without a free joint would need its
  Jacobian, which `StateView` does not carry: refused by name, as is any frame but the world.
- **A region** is a site — an `.esscene` `[[region]]` or an MJCF `<site>` — and its `size` the
  box's half-extents. The box is **axis-aligned in the world frame**: its centre is the site's
  position plus its bodies' (summed from the world down), and the faces enter the cone as
  `Compare` literals. A cone has no constant node to rotate a vector by (a world point enters
  only as a literal, the reason `near` a point is a per-lane `Normalize`), and a region the
  editor writes is unrotated, so a rotated site — itself or any body above it — is refused by
  name, and so is a site that moves (a joint on its body or above it). Inside is strict: on a
  face is outside. `inside` a region took no shaping until R5 (section 4.7.1).
- **Subjects.** `<body>.x` of a free body stays its free joint's first `qpos` lane (G3a's form;
  the SO-101 hash depends on it). Every other `<body>.x|y|z` — `y`, `z`, and `x` of a body
  without a free joint — is a `Slice` of `GetBodyPose.pos`, for `still` of
  `GetBodyVelocity.linear` (a free body). The two position arrays are one frame but not one
  instant on MuJoCo: `mj_step` computes `xpos` in the kinematics before it integrates, so after
  a step `xpos` trails `qpos` by one physics substep (they agree after a reset); `near` and
  `farther_than` already read `xpos`.
- **`still`** of a subject that names a body — not a joint (a joint name wins: the SO-101's
  `gripper` is both) and not `<body>.<axis>` — is `‖v‖ < speed`, and with `angular` (rad/s,
  new field) also `‖ω‖ < angular`; the body needs a free joint. A coordinate keeps G3a's
  `−speed < v < speed` and refuses `angular`.
- **`above` / `below` a body**: `object` and `m` (the margin, absent 0): above is
  `z_subject − z_object > m`, below `z_object − z_subject > m`.
- `object` excludes `range` (`inside`), `value` (`above` / `below`) and `point` (`near`,
  `farther_than`): "either `object` or …", naming the clause.

### 4.5 What G3b settled (`es project generate`)

`generate(spec, root, out, source)` in `crates/es-script/src/spec/` (`generate.rs`,
`learning.rs`, `recipes.rs`, the sections in `project.rs`, `robot.rs`) and the verb
`es project generate --spec <x.estask> --out <dir> [--scene <s>]` (paths relative to the
current directory, the project root; `--out` is written into the cycle as given). Oracles
`crates/es-script/tests/estask_generate.rs` (hashes, determinism, refusals) and
`crates/es/tests/project.rs` (`es ir check` and `es policy init` on each arm, the recipes and
the cycle dry-run). The spec gains five optional sections; every default is a value of the two
committed sets, so a field is written only where a task differs from them. Every document is
validated and each arm passes the cross-IR check before anything is written.

- **`[teacher]`** (absent: no teacher): `state`, its channels in the state vector's order (any of
  `[observe] state` and `privileged`; absent, every `state` then every `privileged` one) and
  `training`, any part of a `training.toml` merged over the PPO preset (plan H's
  `training-teacher-v2.toml`; `Recipe` refuses an unknown key by name). Documents: plan H's state
  MLP (ELU `[512, 256] → 128`, `tanh` head, one row per tick, the ctrlrange unnormalizer, the
  deadline one control period in whole ms), its deployment (horizon 1) and its evaluation (the
  nominal suite).
- **`[student]`**: `name` (the arm: `observation-<name>.toml`, …; absent `student`), `views`
  (from `[observe] cameras`; absent all), `state` (`[observe] state` channels only — a privileged
  one is refused), `family` (`act`, plan N's ResNet18 per view into a `Concat`; `mad`, the views
  sharing the first encoder and summed, with the single-view loss in its preset), `preset` — the
  conventions that differ between the two committed students and mean nothing else: `h3` (default,
  plan H's: `tanh` head and the ctrlrange unnormalizer, H6's invariant; the state concatenated and
  standardized as one `state` input; every view fused on a port named by its camera) or `u3` (the
  SO-101 demos': an unbounded head whose rows are the targets; one state channel under its own name
  as `[-1, 1]`; the first view on port `image`) — `horizon` and `execute` (required;
  `control_hz / execute` whole, XIR-023), and `training` over the ACT preset (plan N's
  `training-views.toml`) or MAD's (`training-mad.toml`).
- **Observation**: each view is plan U's chain (`Dequantize`, `Normalize [0, 1]`, `Pad 4`,
  `Crop Random`, `ColorJitter 0.2 / 0.2` training only) on the Task IR channel's `ImageSpec`. The
  state statistics come from the scene and the clauses: a joint position by its range, a joint
  velocity by 5 rad/s, the previous action by the ctrlrange, a body's position by the distance
  clause that measures it from a point (that point, that radius: the drop radius) or else its scene
  position ± 1 m, a quaternion as it is, a velocity by 1 m/s and 5 rad/s.
- **`[deploy]`**: `name` and `workspace` (absent: `robot`, the root body ± 1 m). The envelope is
  the scene's — each ctrlrange and forcerange, plan H's rate rule (velocity `2 w hz`, acceleration
  `4 w hz²`, differences `2 w` / `4 w`) — at a control rate that is the scene's timestep in whole
  nanoseconds over the decimation, exactly, and an inference rate that over `execute`.
- **`[evaluate]`**: `first_seed`, `episodes`, `success_rate` (absent 101, 16, 0.5). The student's
  adds plan U's five perturbation suites (the delay one and two control periods in whole ms) and
  has a `-nominal` sibling; the teacher's is the nominal suite.
- **`[cycle]`**: `runs` (absent `runs`), `expert` (absent: the trained teacher,
  `<runs>/teacher.esb`), `episodes`, `seed`, `success_only`, `nominal_only`, `jobs`, `preview`
  (absent on), `showcase`. The recipes' paths derive from `runs` — `<runs>/teacher-untrained.esb`,
  `<runs>/<name>-untrained.esb`, the dataset `<runs>/<name>-001/collect[/successes]/{ds,frames}` —
  and an expert's collection is recorded under the student's bundle (its Task IR). Each recipe's
  header carries the `es policy init` line that builds its bundle from the documents; the verb
  does not build bundles itself.
- **`robot`** may name an `.esscene` include (the orchestrator, 2026-10-01): its one root body
  whose subtree has joints (`hand` → `robot0:hand mount`, past `floor0`); several are refused.
- **Reproduced** (IR documents by their semantic hash, recipes and cycles as the parsed
  `Recipe` / `Cycle`): every document of `tests/fixtures/shadow-hand/` (`evaluation-teacher.toml`
  at `episodes = 16`, `-64` at the spec's 64), the SO-101 three-view arm (`*-views.toml`), its
  one-view arm (`views = ["overhead"]`, `*-cam.toml`) and its MAD set (`family = "mad"`,
  `learning-`, `evaluation-`, `training-`, `cycle-mad.toml`). Twelve of the twenty IR documents
  are byte-equal too; the rest number their nodes in another order. The paths the committed runs
  used that the rule does not give are overrides in the two specs (`teacher-untrained-v2.esb`,
  plan N's `runs/collect-001/`). **Not reproduced**: `visible-learning/deployment.toml`, a
  hand-tuned envelope (3 rad/s, 80 rad/s², a 0.05 rad soft margin) the scene's ranges do not
  give; the SO-101 spec gets plan H's rule.

### 4.6 What G8 settled (the sentence editor)

`crates/es-editor-scene/src/sentence.rs` (sentences, edits, levels, "say the task", the camera
check) and `crates/es-script/src/spec/vocab.rs` (the vocabulary as the compiler reads it), drawn by
`crates/es-editor/src/ui/sentence.rs`; oracles `crates/es-editor-scene/tests/sentences.rs` and
`crates/es-editor/tests/sentences.rs` (the words, pinned in `sentences.txt` beside it: Korean may
live only in documentation and string tables).

- **One command.** `Command::Spec(Option<Box<TaskSpec>>)` replaces the specification.
  `SceneModel::apply` compiles the specification with `compile_task` on the scene as edited (an
  unsaved scene is read from `.preview.esscene`) after **every** command, not only this one: a
  scene edit that would break a sentence (deleting the goal a clause names) is refused too, so
  the model's specification always compiles, or there is none. A scene edit is not refused for a
  specification that was broken before it (a hand-edited file); the sentences mend that one. A
  refusal is G3a's error as a `Refusal`: the field is the clause's place and the key
  (`success[0].within_deg`, `timeout_s`, `observe.state.goal_pose`), the words a key of the tables
  (`author.task.required`, `.ticks`, `.touches`, `.no_success`, else the compiler's reason).
- **The vocabulary is the compiler's.** `vocab::takes(relation, object)` is the table
  `check_fields` reads, moved out of it (no compiled byte changed: G3a's and G3b's oracles pass
  as they were). `vocab::subjects`: every body but the world, every hinge and slide joint,
  `<body>.x|y|z` of every free body. `vocab::relations(scene, subject)`: a joint or a coordinate
  is inside a range, above or below a value, or still (a coordinate only of a free body); a body
  is inside a region, above or below a body, near or farther than a body or a fixed point, still
  (a free body), turned like a body, or touching (shown disabled with its reason). A name that is
  both a joint and a body (SO-101's `gripper`) reads as the joint where both could, as the
  compiler reads it, and takes the body-only relations too. `vocab::regions`: the sites on bodies
  that do not move. A region is an object, never a subject (the compiler reads none as one).
- **Sentences.** A clause, the time limit and a start item are a key of the tables with numbered
  holes (`{0}` the subject, `{1}` the relation, then its fields), so each language puts the slots
  where it reads them; a slot is a name, a relation, a way to draw, a number or a point. Numbers
  show in cm, °, cm/s, °/s, s and % (a resting tilt `t` as its angle `2·atan t`), two decimals at
  most. The document keeps its own units and bits: a slot carries the document's value (absent
  stays absent) and a number is written back (the tilt through `es_math::approx`) only when the
  person changes it. Oracle 1: both committed specifications, read as sentences and written back
  slot by slot, are equal field for field.
- **Edits.** A new subject keeps its relation when it takes it, else gets its first; a new
  relation keeps the fields it takes, fills the ones it needs (a range ± 5 cm about the subject's
  place, a value at it, 5 cm/s, 10°, 5 cm near and 25 cm farther, the first region or another
  body — a free one first) and keeps its shaping when the form stays. Clauses are added (the first
  free body, still), removed and moved within their section; an empty failure section is dropped.
  Start items are added (the first free body's x where it stands, ± 2 cm, 🎲) and removed, and
  their `what`, 🎲, value, range, noise and draw change (`tilt` brings a 30° bound). Observe is a
  checklist of the scene's cameras (a checked one appended, `camera_px` 96 if unset; an unchecked
  one leaves the student's `views` too), `camera_px`, the look and the sources (a new channel is
  named after its source, `cube.pose` → `cube_pose`; a removed one leaves the teacher's and the
  student's `state` too).
- **Levels** (section 9, item 5; the owner's to change). A shaping weight is `0.1`, `1` or `10`,
  signed as its form pays (a distance or a ramp costs, an angle pays); the success bonus is `25`,
  `100`, `250` or none; `[start] strength` is `0.5`, absent or `1.5` (es-script scales every 🎲
  range about its centre, and its noise, by it; medium writes no `strength` because `1.0` is not
  the absent one bit for bit — `c ± 1·h` need not give back `lo` and `hi`). The committed weights
  read as levels: Shadow Hand's rotation `1` medium, cube distance `−10` high, bonus `250` high;
  SO-101's ramp `−1` medium. A number at no level reads "as written (0.37)" and is kept. A ramp
  turned on starts at the range's low edge and ends a range's width past its high one. Weights are
  before `[reward] scale`, which the sentences do not show.
- **The look**: quick (no `render`) or light traced (`pt` at plan H's 32 samples, 3 bounces, a
  seed per tick), samples and bounces as numbers. The viewport's materials look is shown
  disabled: a Task IR sensor has only the rasterizer and the path tracer.
- **Say the task** (a project with no `task.estask`): the robot is the scene document's first
  include, else the root of the first hinge or slide joint's body; `control_hz` is the template's
  Task IR's `control_rate_hz` (SO-101's 50; 50 without one); 8 s; one success clause, the first
  free body still (`speed = 0.05`, SO-101's settling bound); every camera of the scene observed at
  96 px. Oracle 4: on the SO-101 copy, with a region over the bin and "[cube] is inside
  [bin_area]", it compiles and `generate` writes `task.toml`.
- **The camera check.** For each observed camera and each body a clause is about (a coordinate's
  body; a joint is skipped): one ray from the camera to the body's origin where the start puts it
  (a free body's `x|y|z` item: its value, or the middle of its range), inside the square
  picture's field of view, whose nearest hit — `nearest_hit_flat` on the scene's triangles, G6's
  rule — must be one of the body's own shapes. Measured on the SO-101 copy: the overhead camera
  does **not** see the cube at its start — the lower arm at the home pose is on the line to the
  cube's centre (the corner's own frame shows the same). Oracle 5: a camera level with the cube
  sees it; turned away, or behind a board, it does not; the cube's start lifted over the board,
  it does again. One ray to the centre is the packet's rule, so a half-hidden cube reads as
  unseen.
- **The editor.** ①'s right pane has two tabs, the selection's fields and the task. A sentence's
  widgets sit between its words in the reader's order. The edited specification is handed over
  once the person lets go, one undo step; a refused one stays in its sentences with the reason
  above, again under the clause it names, and "put back".

### 4.7 What GV settled (where an unset coordinate starts)

GV's authored project (SO-101, a 5 cm box at (0.22, 0, 0.025), start items `box.x` and `box.y`)
found that `Env::reset` zeroes `qpos` before the reset nodes run, so a free body's coordinate no
start item set started at **zero**: the box at z = 0, half in the floor, popping up over the first
five ticks; its quaternion all zeros, which the backend reads as the identity, losing a rotation
the scene gave it; and under "say the task" (no start items) every free body at the origin,
inside the robot's base. Oracles `crates/es-script/tests/estask.rs` (`scene_pose`).

- **The rule.** For every free joint whose body is outside the robot's subtree, `compile_task`
  emits a constant `ResetState` on each of its seven `qpos` lanes no start item writes (by
  `<body>.x|y|z`, `<body>.orientation` or the joint's name): the value is the scene's own,
  `qpos0` — the body's position `x y z`, then its orientation `w x y z` (MuJoCo's order), read
  bit for bit from `SceneDesc`, no arithmetic. Target `qpos[<lane>]`, stream
  `scene.<body>.pos.<i>` or `scene.<body>.quat.<i>`; the nodes follow the start items' own, in
  the scene's joint order and lane order. A constant draws nothing, so no other draw moves. On
  GV's scene the box gets z = 0.025 and `(1, 0, 0, 0)`; a box turned in the scene keeps its turn;
  with no `[start]` at all every free body starts at its scene pose (measured on `mujoco-cpu`).
- **The opt-out.** `[start] zero_unset = true` keeps the old behaviour: no such node, unset lanes
  at zero. Absent or `false` is the rule. The two committed specifications set it (one comment
  line says why: they reproduce documents committed before the rule), so their `task_hash` and
  G3b's regenerated documents are unchanged; the Shadow Hand template's editable copy copies it.
  It is no sentence (G8): the sentences keep it as read, and a `[start]` table holding only it
  is not dropped.
- **Open for the owner.** Under the opt-out the committed Shadow Hand task's goal (`target`, a
  free body with gravity compensation, only its orientation drawn) starts at the origin (0, 0, 0)
  every episode, not where the scene puts it. Removing `zero_unset` from
  `shadow_hand_repose.estask` would fix that, and would move the committed `task_hash` and with
  it the trained teacher's documents; so101_views' cube would also gain its quaternion (identity,
  as the backend already read it). Left as committed.

### 4.7.1 What R5 settled (shaping toward a region; joints at `qpos0`)

M17's review, N-2 and F-9. Oracles `crates/es-script/tests/estask.rs` (`region_distance`),
`estask_scripted.rs` (`inside_a_region_pays_the_distance_to_its_centre`) and
`crates/es-editor-scene/tests/sentences.rs` (`the_region_distance_toggle_is_one_undo_step`).

- **`inside` a region takes `shaping = "distance"`**: `weight × ‖p − c‖`, `p` the body's world
  position, `c` the region's centre (section 4.4's), the distance clamped to [0, 1] m as `near`'s
  is. It is `near` a point's construction: `GetBodyPose.pos`, a per-lane `Normalize`
  `[c − 1, c + 1] → [−1, 1]`, `Norm L2`, `Normalize [0, 1] → [0, 1]`, `Reward`. Each lane clamps
  at 1 m, which changes nothing: a lane past 1 m puts the distance past 1 m, where the term is
  clamped anyway. The term is `term`, absent `<subject>_distance`. Opt-in: without `shaping` the
  clause compiles as before (G3c's tests unchanged; on GV's project an unshaped region clause has
  the same `task_hash` and the same document bytes as before R5).
- **The sentences.** The region clause shows the shaping toggle and its levels as the other forms
  do (a distance costs: medium is `weight = −1`). A newly picked "inside a region" — a new clause,
  or a relation or subject change that lands on it — starts on at medium, unless the clause had a
  distance term to keep (G8's rule: the shaping stays when the form does, so `near` shaped at high
  that becomes `inside` stays at high). Picking the relation the clause already has changes
  nothing, so a toggle turned off stays off. "Say the task"'s clause is still the first free body
  still; it gains the term only where it lands on a region, a scene with no free body.
- **F-9: no joint starts away from its `qpos0`.** `SceneDesc` has no `ref`: the MJCF parser warns
  that the attribute is not represented and drops it, `.esscene` has no such field, and the
  backends' MuJoCo model is written from `SceneDesc` (`scene_to_mjcf`). So every hinge and slide's
  `qpos0` is 0 in the model the pipeline runs, the pose the file draws, and `Env::reset`'s zero is
  that `qpos0`. No committed scene has a `ref` either (nor a ball joint; the only floating-base
  robot, `go1_primitives.xml`, is in no `.estask`, and M6's `task.toml` is not compiled). Nothing
  is emitted for any committed, GV or library scene, and no code was added. What would change it
  is `ref` in `SceneDesc`, an `es-assets` packet
  (the parser, the `.esscene` field, both MJCF writers, `scene_hash` appending it only when non-zero
  as the appearance block does); then `scene_poses` emits a constant `ResetState` for each unset
  hinge or slide whose `ref` is not 0 (stream `scene.<joint>`), and none under `zero_unset = true`.
  `a_hinges_ref_is_not_represented_and_nothing_is_emitted` fails the day `ref` is parsed.

## 5. The editor (① and ②)

- **Hierarchy panel**: the scene tree (includes folded), search, visibility; drag to re-parent.
- **Inspector**: the selected thing's fields with 🎲 toggles; units in words; invalid values
  refused in place with the reason (`SceneDesc::validate`, the mapping report).
- **Viewport**: picking (the segmentation channel already gives geom ids), translate / rotate /
  scale gizmos with snapping, frame-selected, H8's render modes drawn in process every frame
  (H9), and the policy camera's view in the corner rendered by the same in-process renderer at
  the camera's declared resolution and render path (the real observation; `es render` without a
  device).
- **Commands and undo**: every edit is a command on a scene model in `es-editor-scene` (layer
  12, 5.1), applied to the document, validated, and undoable; the layout (`*.eslayout`, §14.3)
  never enters the scene document.
- **Physics preview**: `es scene simulate --seconds 3 --out <traj>` on `mujoco-cpu`, played back
  in the viewport (the editor runs no physics).
- **Checks** in plain words, computed by `es` or pure functions: the mapping report ("this
  backend cannot simulate tendons"), a camera that cannot see an object at its start positions
  (one segmentation render per camera), an object that falls through the floor in the preview,
  a robot whose joints exceed their limits at the start pose.
- **Save** writes `scene.esscene`; generation runs on demand (and before ③), so the person never
  sees an IR document unless they open the Advanced tabs.

### 5.1 What G5 settled (the scene model)

`crates/es-editor-scene/` (a new layer-12 crate: `SceneModel`, `Command`, `check`, `tree`,
`inspect`, `euler`, `make_editable`), drawn by `crates/es-editor/src/ui/author.rs`, oracles
`crates/es-editor-scene/tests/scene_model.rs`.

- **The crate**: `es-editor-model` stood at 8,942 of its 10,000-line cap and the scene model is
  about 1,400 lines, so it went into a new layer-12 crate `es-editor-scene` (spec §4.2's table and
  rule 4, Appendix C.8, xtask's `LAYERS`). Neither knows the other (rule 1); only `es-editor`
  uses both. A refusal's words are keys of `es-editor-model`'s string tables, and the tables'
  completeness test reads this crate's sources too.
- **An editable project** holds `scene.esscene` (and `task.estask`) in its root. The generated
  documents go to `generated/` (`generate`'s `out` is `"generated"`), emptied before each
  regeneration, so a failure leaves nothing stale. A template names its editable sources as
  `[editable] scene / spec`. "Make an editable copy" copies the template's scene document byte for
  byte and the files it names (includes, meshes, textures) **under the same relative paths**:
  asset paths are hash input, so moving them to `assets/<blake3>` would move the `scene_hash` of a
  scene nobody edited (3.1's rule is left to G7's import). The specification is copied with only
  its `scene` line changed to `scene.esscene` (comments kept). Measured (oracle 5): the Shadow Hand
  copy expands to the template's `scene_hash`, and its generated documents equal the committed
  ones but for what names the scene *file*: the Task IR's `scene.path` and `asset_hash`, the
  Deployment IR's simulated robot target, and through the hash chain an Observation IR's
  `task_ref` and an Evaluation IR's `task` and `observation`; the Learning IRs are unchanged. The
  two SO-101 cube templates copy the scene only (no specification says their `task.toml`).
- **Commands**: `Add` (a body with its geoms, static scenery, a camera, a light, a region; an
  include is G7's), `Delete` (a body with the bodies, cameras and regions hanging from it),
  `Duplicate` (beside the original, `<stem>_<n>`; an include's copy gets a prefix), `Set` (the
  fields; not the name), `SetPose` (for G6's gizmos), `Reparent` (the bodies reordered so a parent
  comes first; a cycle refused), `Rename` (the document's parent references and, in
  `task.estask`, the robot, each clause's subject and object, each start item's `what`, the
  observed cameras and sources, the student's views; `robot.` stays the specification's word).
  Identity is the name; a geom is its place in its list.
- **Validation policy: a refused command is not applied.** The check is: the values (a negative
  mass, density or friction, a zero size, a colour outside [0, 1], the field of view, the order of
  limits, an empty name), then G1's `expand` (a second name, a missing parent, an unknown
  material, a missing file), then the mapping report of `mujoco-cpu` (and `mjwarp` when the
  template trains there). A refusal is the field path as G1 names it, a key of the tables and its
  arguments. Why not apply-with-error: this way the document, the viewport, every undo step and
  the physics preview always hold a scene that expands; the typed value stays in its inspector
  field, its label red and the reason above.
- **Undo**: the whole documents (scene and specification) before and after, per step; a command
  that changes nothing is no step. Dirty is "the documents differ from the saved ones"; save
  writes both atomically (a temporary file, then a rename) and regenerates.
- **Visibility** is the editor's, in memory only (no `.eslayout` is written yet).
- **Inspector**: angles are roll, pitch, yaw in degrees about fixed X, Y, Z, converted with
  `es_math::approx`'s `acos_f64` and `sin_cos_f64`, and a quaternion is written only when an
  angle changes. Sizes show whole (widths, diameters) and are stored halved, exactly. A value is
  handed over once the person lets go (no pointer held, no text being typed), so a drag is one
  undo step.
- **Viewport**: `ScenePreview::from_scene`; a still scene's shot carries the document's revision
  in its tick, so every edit is drawn afresh. An unsaved document is written to
  `.preview.esscene` beside the scene for `es render` and `es scene simulate`; the editor's
  replay loader reads `.esscene` too.
- **Until G9**, ② to ⑤ still run the template's documents; ① says so.

### 5.2 What G6 settled (picking, handles, the corner view)

`crates/es-editor-scene/src/{view,gizmo,policy}.rs` decide; `crates/es-editor/src/ui/author.rs`
(the overlay), `ui/corner.rs` and `gpu.rs` (`Sensor`) draw; oracles
`crates/es-editor-scene/tests/viewport.rs`.

- **Picking is the segmentation channel at the clicked point, read on the CPU.** The ray the
  renderer casts through the point (`es_render::cpu::primary_dir`, in `f64`) meets the drawn
  scene's triangles under the renderer's own nearest-hit rule (`nearest_hit_flat`), and the hit
  triangle's segmentation id is the geom. No readback and no `es-render` change: the viewport
  draws `Rgb8` only, a click is one ray rather than a second channel per frame, and the answer is
  testable on scripted rays. A body's geom selects the body; anything an include brings selects
  the include (which has only its own pose to edit; its parts stay read-only); a document world
  geom is scenery; a light's panel is the light. Cameras and regions draw nothing, so the
  hierarchy selects them. What is hidden is not hit, so the ray goes on behind it; empty space
  selects nothing.
- **Handles.** Move and turn work in the frame the document writes the place in (the parent's;
  the world's for most things), through the thing's own origin, so a move changes exactly one
  coordinate of `pos` and a turn is `dq · q` in that frame. Size works in the shape's own frame:
  a geom, static scenery, a region, a light, and a body with exactly one geom (its geom). A mesh,
  a camera, an include and a body of several geoms have no size handles. A handle is 15 % of its
  distance from the eye long, so it keeps its size on screen. W, E, R pick the tool; F frames.
- **Snapping** (on by default, one checkbox) rounds in the document's units: the moved coordinate
  to a whole centimetre (`(v · 100).round() / 100`, so the document writes `1.05`), the drag's
  turn to a multiple of 15°, a size's whole extent (the inspector's width, height or diameter)
  to a whole centimetre, at least 1 cm. The turn is measured with `es_math::approx`'s `acos_f64`
  (the inspector's `atan2`) and written with its `sin_cos_f64`, so what a drag writes does not
  depend on the host's `libm`.
- **A drag is one command.** While a handle is held the document does not change: the viewport
  draws the selection's tint where the drag puts it, the handles there, and how far it has gone
  (`+5.0 cm`, `+30°`, `6.0 → 9.0 cm`). Letting go applies one `SetPose` (move, turn) or one
  `Set` (size), which is one undo step; a drag that snaps back to where it started makes none.
- **Selection** is a translucent tint over the selection's drawn triangles, drawn over the
  picture (an x-ray, not depth tested). **Frame-selected** keeps the camera's direction and puts
  the sphere about what the selection draws (10 cm at its place when it draws nothing) in the
  middle of the picture, filling most of its height.
- **The corner** shows a camera the policy is given. An editable project with a task
  specification: the student's `views`, else `[observe] cameras`, compiled by `compile_task` on
  the scene as edited (`.preview.esscene` when unsaved). A template, and an editable copy with no
  specification: the cameras the bundle's Observation IR reads (`ImageInput` sensors, node
  order), as the bundle's Task IR declares them. The selected camera when the policy sees it,
  else the first. Measured: the Shadow Hand copy's three corner cameras are declared exactly as
  `task-repose.toml`'s channels (96 × 96, `Pt` 32 spp, 3 bounces, exposure 8, a seed per tick).
- **The corner's frame** is drawn in process by the viewport's worker (H9) through `es-env`'s
  own single-camera steps — `sensor_cfg`, `drawn_frame` with no draws, the `Tick` stream's seed
  for tick 0, one render, one readback — so it is the observation of the scene at its own pose
  (not a reset draw), at the declared size and path, once per scene revision and camera, shown
  with nearest filtering. `es-editor` takes `es-env` with its `render` feature for it. Without a
  device the corner says so: `es render` draws only a free camera, so it is no fallback here.

### 5.3 What G7 settled (Add, imports, markers, overrides)

`crates/es-editor-scene/src/{add,import,marker,overrides}.rs` decide;
`crates/es-editor/src/ui/author/{add,markers,overrides}.rs` draw; oracles
`crates/es-editor-scene/tests/add.rs`. The robot library is `templates/robots.toml`, the empty scene
`tests/fixtures/esscene/empty.esscene`.

- **The menu**: object (box, sphere, cylinder, capsule: a free body with one 5 cm geom), fixed
  object (the same shapes as scenery on the world), mesh file (STL, OBJ: a free body with one mesh
  geom), robot (the library's, or an MJCF, URDF or glTF file: an `[[include]]`), camera, light,
  area. One Add is one command (G5's `Add`), one undo step, and what it made is selected.
- **Where it goes.** The view's centre ray meets the drawn triangles under the renderer's
  nearest-hit rule (`SceneModel::surface`, `hit`'s sibling, which also gives the triangle's
  normal). The point is met again in `f64`, and on a level surface its height is the triangle's
  own, so the table's 0 stays 0. The new thing's lowest point sits at the hit: a primitive's
  half-extent along the normal (it is added unturned), a mesh's lowest vertex, a region's box. With
  no hit it goes at the origin, on the ground plane. With snapping on (G6's toggle) a point on a
  level surface is rounded across the surface to whole centimetres; its height does not move.
  - A camera goes at the view's eye, looking where the view looks: the renderer's look-at
    quaternion turned half about X into MJCF's camera frame, so no trigonometry; with snapping, the
    eye and the target are whole centimetres first; `fovy` is the view's, to 1e-6°.
  - A light hangs 1 m above the point (a ceiling panel, as G5's Add put it).
  - A robot stands at the point plus its library offset, in its library orientation (absent: the
    file's own).
- **Imports by content.**
  - A mesh or a picture goes to `assets/<blake3>.<ext>` (the extension in lower case, since the
    readers choose by it).
  - A robot file goes to `assets/<blake3 of the file>/<its name>`, with the files it names under
    their relative paths: what its reader loads (mesh and texture paths, texture and cube files,
    a `.gltf`'s buffer and image URIs). They are found by expanding the file alone from its own
    folder, so a broken file is refused before anything is written. A file that names something
    outside its folder (`../`, an absolute path) is refused by name.
  - The same bytes already there are not written again; a different file at that path is
    refused.
  - The check every command passes expands from disk, so the files are copied first, and a
    refused Add removes exactly what it wrote (the files and the folders it made). Undo leaves an
    imported file in place (redo needs it); it is a cache by content. G5's editable copy keeps its
    relative paths.
- **Names**: a new body, camera, light or area is `unique(stem)` in G5's name space; scenery is
  unique among the world's geoms; an include's handle is the library id or the file's stem. When an
  include's own names meet the scene's (a second SO-101), the Add is tried once more with the
  prefix `<handle>:`, the prefix a duplicated include gets.
- **Picture import** is the material picker's last entry. It is one command, `Command::Scene` (the
  whole document replaced, the one new command): a `[[texture]]` (2D, the file) and a
  `[[material]]` (that texture), both named by the picture's stem — or the pair the same picture
  made before — and the geom's `material`. PNG only, as the texture loader decodes; anything else
  is refused and its copy taken back.
- **Markers.** Cameras and regions are drawn as lines: a camera's frustum (12 % of its distance
  from the eye deep, so one size on screen; a square picture of its `fovy`; a tick on its top edge)
  and a region's box, the selected one lit. A click within 8 points of those lines picks the camera
  or region, before the ray looks for a drawn geom; G6's handles follow (a region's size too).
- **Overrides.** The inspector of an include lists what its file declares, by the file's names:
  `SceneModel::brought` expands the file alone, without overrides and prefix — every joint but a
  free one, each actuator (`kp`, `kv` only where it has that gain), each geom, the file's
  materials. Each value is the override or else the file's, with ↺ back to the file's. An edit is
  G5's `Set` of the include with its `set` pruned (`overrides::prune`): a cleared field drops its
  override, an emptied target drops its table, and an empty `set` is absent. Ranges show in degrees
  for hinge and ball joints.
- **The library** (`kind = "robots"`, `[[robot]]` with `id`, `name` — an i18n key — `source`
  relative to the repository root, and optional `pos`, `quat`, `prefix`). The template loader
  passes the file by. SO-101 is `so101.xml` with its base on the point; the Shadow Hand is
  `shadow_hand.xml` moved by (-1, -1.25, 0), so its mount hangs 15 cm over the point.
- **`empty.esscene`**: an endless floor (`plane = [0, 0, 0.05]`), a 60 cm ceiling light at 1.5 m,
  and `outside`, a camera 1.2 m back and 1.2 m up, looking down at the origin at 45°.
- **Measured**: every item adds, expands and maps onto `mujoco-cpu` on the Shadow Hand copy and on
  `empty.esscene`; the twelve Shadow Hand meshes, imported, hash as the source's.
- **Left open**: a mesh has no size handles and no scale, since neither `SceneDesc` nor the
  document carries a mesh scale (an STL in millimetres comes in 1000 times too large). A URDF whose
  meshes lie outside its folder is refused, and USD is refused as in G1.

### 5.4 What G9 settled (an authored project runs end to end)

`templates/empty.toml`; `crates/es-editor-model/src/model/template.rs` (`source`, `authored`,
`Generated`, `load_saved`, `write_saved`), `project.rs` and `watch.rs` (`with_source`,
`set_source`, the gate); `crates/es-editor-scene/src/{model,copy,check,sentence}.rs` (`on_disk`,
`refresh`, `Regen::Stale`, `copy::documents`, `check::maps_onto`, the learners "say the task"
writes); `crates/es-editor/src/ui/{scene,author,teacher,teach,results,train,home}.rs` draw and
wire. Oracles `crates/es-editor/tests/authored.rs` (the packet's 1 to 4) and
`crates/es-editor-scene/tests/{sentences,documents}.rs`.

- **The empty project is a template without documents**, not a project kind of its own.
  `Template.bundle` is optional, `cycle` and `robot` may be empty, and `templates/empty.toml`
  names `[editable] scene = empty.esscene` and nothing a task has. Why: every place that resolves
  a project's template keeps one type and one lookup; a project kind would have doubled the
  branches of the start screen, the project file and every step. Its card (sorted by id beside
  the others) makes the folder: the start screen copies the scene in as ①'s editable copy does
  (`make_editable`), then `Project::create` writes `project.toml` and builds no bundle. It needs
  `mujoco`, `torch`, `vulkan` and `render`, not `mjwarp`: building a scene needs no GPU
  simulator, and ② says on the teacher's card what is missing.
- **What ② to ⑤ run** is one function, `template::source(project, repo, generated)`:
  - a template project (not editable): the template's documents, `es` in the repository root,
    exactly as before;
  - an editable project with no task: its template's documents when it has some (an SO-101
    copy), else nothing ("there is no task yet");
  - an editable project with a task: the documents in `generated/` when they are exactly what
    the saved scene and task generate, `es` in the project's own folder.

  "Exactly" is `es-editor-scene`'s `on_disk`: `generate` run in memory and every file compared
  byte for byte (the generator has no clock). `es-editor-model` does not have the generator, so
  `es-editor` carries the answer: from disk when the project opens, then every frame from ①'s
  model (saved, unsaved, the last generation). When it changes, ③'s source, ②'s teacher card
  and program, and ⑤ are read again.
- **The authored template** (`template::authored`) is the project's template — its words, needs,
  lengths and view — with `generated/`'s student arm as the bundle, its `[teacher]` documents, its
  cycle and `scene.esscene`. `method = "teacher"` when there is a teacher and the cycle names no
  expert. It has no `[outcome]`: the template's outcome is its own task's, and explaining a
  failure by the missing clause is S4's next item. `untrained.esb` is rebuilt from the generated
  documents before each run, since they change with every save, as `es policy init` builds it.
- **`es` runs in the project root**, not with absolute paths. The generated documents name
  `scene.esscene` relative to the project (it is hash input, `SceneRef.path`) and the recipes
  derive `runs/…` from `[cycle] runs`, both relative, as G3b's `generate` writes them. The launch
  model already starts every `es` in a folder (`start_in`), so the documents run as written.
  Absolute paths would have meant rewriting generated documents, and a scene path that is hash
  input, per machine. What the editor writes itself stays absolute (run folders, bundles, run
  recipes), as before.
- **Blocking.** ③ does not start, and says why, while ① holds an unsaved task
  (`watch.task.unsaved`), while `generated/` is not what the saved documents make
  (`watch.task.stale`: changed on disk, or never generated), or when generation failed
  (`watch.task.failed`, with the generator's reason). ① is then the step that is not done, so the
  step bar opens there, and its Save button reads "Save and generate". Saving clears the block.
  ② says the same in its panel.
- **"Say the task" says who learns it** (section 4.6's defaults, and):
  - state: the robot's `joint_pos`, `joint_vel` and `previous_action`; privileged: the first free
    body's `pose` and `vel`;
  - `[reward] scale = 0.01` (plan H's) and the medium success bonus (100);
  - `[teacher]` empty: every channel, plan H's PPO preset (2,048 envs, 3,000 iterations on
    `mjwarp`, a checkpoint every 250);
  - `[student]` over every observed camera, reading `joint_pos`, preset `h3` (the `tanh` head),
    16 rows, executing the largest of 1 to 10 that divides `control_hz` (10 at 50 Hz);
  - `[cycle]`: 200 demonstrations from seed 1001 by the trained teacher, `success_only`.

  All of these are the owner's to change (section 9).
- **The teacher card for any project with a `[teacher]`** is plan H's card (H7) on the generated
  recipe, evaluation and documents; its words say "every object", not the cube. Before anything
  runs, the saved scene is checked against `mjwarp`'s mapping report (`check::maps_onto`): a
  blocked feature is said on the card in plain words, and Train is off. While a project is
  built, it is checked against `mujoco-cpu` only.
- **Save as template**: ①'s toolbar, a name, and the folder
  `<documents>/Electric Sheep/templates/<name>/` (`<name> 2`… when taken; never the
  repository). It gets the saved documents (saved first if edited): `scene.esscene`, `task.estask`
  and `assets/` whole, since a glTF's buffers are not in the scene's asset list. Then
  `template.toml` is written last: the project's template's words, needs, lengths and view under
  the person's name, `id = "saved:<folder>"`, no documents, and `base`, the built-in template it
  comes from. The start screen lists saved templates after the built-in ones. A project made from
  one copies the documents the same way and names `base` in its `project.toml`, so it never
  depends on a folder the person may delete. `ES_DOCUMENTS` replaces the documents folder for
  tests and captures.
- **Fixed on the way.** An object from the Add menu has an unnamed free joint, which takes the
  body's name, and the compiler read "[box] is still" as that joint and refused it ("not a hinge
  or slide"). A name is now read as a joint only when it is a hinge's or a slide's, as
  `vocab::relations` already read it. No committed specification's output moved: their free
  joints are named apart from their bodies. The camera check reads names the same way.
- **Measured** (oracle 1, on the empty project with the library's SO-101, a 5 cm box at
  (0.22, 0, 0.025), a target area at (0.22, 0.12, 0.025), "say the task" and "[box] is inside
  [target]"): thirteen documents; `es ir check` passes the teacher arm and the student arm with
  each of its evaluations; both recipe headers' `es policy init` build their bundles;
  `es train --dry-run` of ②'s teacher run plans `train_ppo` on `mjwarp`; and
  `es loop cycle --dry-run` passes on the generated cycle and on the run recipe ③ writes, all in
  the project's folder. The SO-101 scene maps onto `mjwarp` (its meshes are a warning there).
- **Left open.** The teacher card's time sentence is the hand's ("1 to 1.5 hours"); an SO-101
  teacher's run is unmeasured until GV. ⑤ explains failures for a template's own task only. An
  authored project whose cycle names an expert runs, but gets no demonstration program file.

## 6. Where the code goes (layers, §4.2)

| What | Crate (layer) |
|---|---|
| `.esscene` reader and writer, include expansion, full MJCF exporter | `es-assets` (2) |
| task-spec compiler and `es project generate`'s core (pure: scene + spec → documents) | `es-script` (11), the authoring crate; `es-editor-scene` calls it directly |
| `es scene export / simulate`, `es render`, `es project generate` CLI | `es` |
| scene model, commands, undo, checks (G5), picking, handle and corner decisions (G6), sentence editor model (G8) | `es-editor-scene` (12) |
| hierarchy, inspector widgets, gizmos | `es-editor` (13) |

No new extension point (INV-17). `es-editor-model` is over its line target (8,942 of 10,000), so
G5 put the scene model in a new layer-12 crate, `es-editor-scene` (5.1).

## 7. Spec changes (ko first, with the packet that needs them)

- §14.3 storage formats: `*.esscene` (scene document) and `*.estask` (task specification, not an
  IR); §14.1: the scene document is one more frontend whose `scene_hash` equals any other
  frontend's for the same `SceneDesc`.
- §6 (`SceneRef`): the path may name a `.esscene`.
- §23: scene authoring and the sentence editor as the editor's ① (the read-only ① of M12 stays
  for templates opened read-only).

## 8. Oracles (§1.4)

1. **Same scene, same hash**: a `.esscene` that writes what `so101_pick_place.xml` (and the
   Shadow Hand scene) says reads to an equal `SceneDesc` and the same `scene_hash` / `asset_hash`.
2. **Export round trip**: `parse_mjcf(export(scene)) == scene` for every committed scene and
   for generated scenes (property test); MuJoCo loading the export steps bit-identically to the
   backend's own emission (H1's parity oracle).
3. **Generation is a function**: the SO-101 views and Shadow Hand specs regenerate the committed
   documents byte for byte; a changed sentence moves exactly the hashes it should.
4. **Every relation is checked on scripted states** (as H2's tests): its truth and its reward on
   states built to satisfy and to violate it.
5. **Editor commands** (headless): apply / undo / redo returns the identical document; invalid
   edits are refused with the reason; a rename regenerates every naming document.
6. **End to end** (orchestrator): build a new task in the editor from an empty project — e.g. an
   SO-101 pushing a cube into a target area — teach it with a reward-trained teacher, train a
   camera student, and read its result in ⑤, every step watched in the editor.

## 9. Open decisions (defaults in bold, the owner may change them)

1. Robot import: **a reference with overrides** (3.2), or a flattened copy.
2. Identity: **name paths, renames regenerate documents** (3.1), or GUIDs stored per entity
   (rename-proof, but a second identity beside `StableId`).
3. Assets: **copied into the project by content**, or referenced in place.
4. The sentence vocabulary of section 4.1 for the first version: **as listed**, `touches` after
   `GetContact` lowering.
5. Reward: **derived from the sentences with three-level weights**; a free-form reward editor is
   the Advanced graph's (M3 stage 2) and not part of ①.
6. Who learns a said task (G9): **the defaults section 5.4 lists** — the observed channels,
   `[reward] scale = 0.01` and the medium bonus, plan H's PPO preset on `mjwarp`, the `h3` camera
   student, 200 demonstrations from seed 1001 with `success_only`.

## 10. Plan (M17 plan G) — packets in order

| Wave | Packet | Needs |
|---|---|---|
| 1 | G1 `.esscene` schema, reader (includes), writer; spec §14.3 / §6 | — |
| 1 | G2 full-fidelity MJCF exporter + `es scene export` | — |
| 2 | G3 task-spec schema + compiler + `es project generate`; regenerates the SO-101 and Shadow Hand documents byte for byte | G1 |
| 2 | G4 `es render` (one frame of a declared or free camera) and `es scene simulate` | G1 |
| 3 | G5 editor scene model: hierarchy, inspector, commands, undo, save, checks; the `es-editor-model` split | G1 |
| 3 | G6 viewport picking and gizmos; camera corner view | G4, G5, H8 |
| 4 | G7 Add: primitives, mesh / texture import, robot library, camera, light, region; delete, duplicate | G5 |
| 4 | G8 sentence editor (success, failure, start, observe, reward) on G3 | G3, G5 |
| 5 | G9 empty-project card, save as template, ② teacher for generated tasks | G3, G8 |
| 6 | GV end to end (orchestrator), review | all |
