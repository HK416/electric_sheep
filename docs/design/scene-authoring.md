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
   the viewport's corner (the real observation, H8's renderer).
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

Nothing here runs learning or physics inside the editor (§23.1, §4.2 rule 4): previews, renders
and generation are `es` subcommands the editor hands an argv, or pure functions of the documents.

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

[[material]]
name = "block"
texture = { file = "textures/block.png", kind = "cube", gridsize = [3, 4], gridlayout = ".U..LFRB.D.." }
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
shape = { box = [0.1, 0.1, 0.01] }
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

Each relation compiles to Task IR nodes `es-env` already lowers (`GetBodyPose`, `GetJointState`,
`GetBodyVelocity`, `Arith`, `Norm`, `Dot`, `MathFn{Abs,Sqrt}`, `Compare`, `Logic`, `Reduce`,
`Normalize`, `Clamp`, `Concat`, `Slice`, `ResetState`, `Randomization`, `Terminate`, `Reward`):

| Relation | Compiles to |
|---|---|
| `inside` (a region) | per-axis `Compare` of the subject's position against the region's box, `And` |
| `above` / `below` (by m) | `Slice` z, `Arith Sub`, `Compare` |
| `near` / `farther_than` (m) | `Arith Sub`, `Norm L2`, `Compare` |
| `still` (for s) | velocity `Norm` under a bound — IR-D has no hold node, so "for 1 s" compiles to "inside and nearly still" (the demo's settling bound, `editor-redesign.md` §5 S4) |
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

## 5. The editor (① and ②)

- **Hierarchy panel**: the scene tree (includes folded), search, visibility; drag to re-parent.
- **Inspector**: the selected thing's fields with 🎲 toggles; units in words; invalid values
  refused in place with the reason (`SceneDesc::validate`, the mapping report).
- **Viewport**: picking (the segmentation channel already gives geom ids), translate / rotate /
  scale gizmos with snapping, frame-selected, H8's render modes, and the policy camera's view in
  the corner rendered by `es render` (the real observation, including its resolution).
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
