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
  (an include can have several roots: the Shadow Hand file's `floor0`); its subtree's joints are
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
  use the host's `cos` / `tan`, as the generators this replaces did (the first two agree with the
  committed documents on this PC). A correctly rounded implementation would make a generated document
  host-independent (M10's `scene_hash` lesson) — open.

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
  face is outside. `inside` a region takes no shaping yet.
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

## 5. The editor (① and ②)

- **Hierarchy panel**: the scene tree (includes folded), search, visibility; drag to re-parent.
- **Inspector**: the selected thing's fields with 🎲 toggles; units in words; invalid values
  refused in place with the reason (`SceneDesc::validate`, the mapping report).
- **Viewport**: picking (the segmentation channel already gives geom ids), translate / rotate /
  scale gizmos with snapping, frame-selected, H8's render modes drawn in process every frame
  (H9), and the policy camera's view in the corner rendered by the same in-process renderer at
  the camera's declared resolution and render path (the real observation; `es render` without a
  device).
- **Commands and undo**: every edit is a command on a scene model in `es-editor-model` (layer
  12), applied to the document, validated, and undoable; the layout (`*.eslayout`, §14.3) never
  enters the scene document.
- **Physics preview**: `es scene simulate --seconds 3 --out <traj>` on `mujoco-cpu`, played back
  in the viewport (the editor runs no physics).
- **Checks** in plain words, computed by `es` or pure functions: the mapping report ("this
  backend cannot simulate tendons"), a camera that cannot see an object at its start positions
  (one segmentation render per camera), an object that falls through the floor in the preview,
  a robot whose joints exceed their limits at the start pose.
- **Save** writes `scene.esscene`; generation runs on demand (and before ③), so the person never
  sees an IR document unless they open the Advanced tabs.

## 6. Where the code goes (layers, §4.2)

| What | Crate (layer) |
|---|---|
| `.esscene` reader and writer, include expansion, full MJCF exporter | `es-assets` (2) |
| task-spec compiler and `es project generate`'s core (pure: scene + spec → documents) | `es-script` (11), the authoring crate; `es-editor-model` calls it directly |
| `es scene export / simulate`, `es render`, `es project generate` CLI | `es` |
| scene model, commands, undo, checks, sentence editor model | `es-editor-model` (12) |
| hierarchy, inspector widgets, gizmos | `es-editor` (13) |

No new extension point (INV-17). `es-editor-model` is over its line target (≈8,400 of 10,000):
the scene model goes into a new module set sized for a split, and the plan's first editor packet
does the split the budget requires.

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
