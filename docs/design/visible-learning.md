# Plan V — "visible learning": an end-to-end demo anyone can watch — design

Spec: §28.3 and §28.7 gate 7 (Franka pick&place end-to-end), §1.2 (work packets), §1.4 (oracle-first),
§1.5 (context budget), §1.9 (never cut / cut first), §2.3–§2.5 (where Python is allowed), §4.2
(layering), §4.3 (backends), §5.1 (IR boundaries), §5.3 (hash chain), §6.3 (randomization, reset),
§7.2 (`ImageSpec`), §8.3 and §8.7 (Learning IR, policy runtime), §9.2–§9.4 (Safety Plane), §10.1–§10.4
(Evaluation IR), §12.4 (no single `step/s`), §15.2 (tile atlas), §17.2 (backend mapping), §18.1
(integer time), §19.2 (dataset identity), §25.1 (trust boundaries), §26.1–§26.2 (validate, CI tiers).
Invariants: INV-11, INV-12, INV-13, INV-14, INV-16, INV-17.
Packets: `docs/packets/M5/V0-scene-task.md`, `V0b-render-in-the-loop.md`, `V1-expert-dataset.md`,
`V2-act-training.md`, `V3-perturb-safety.md`, `V4-video.md`.

Design-note sections are cited as "section N"; `§N` always means the spec.

## 1. What this is, and what gate it closes

One runnable demo: an SO-101 arm in simulation is shown scripted demonstrations of putting a cube in
a bin, a policy is trained from that dataset, the trained policy is evaluated on randomized scenes,
and the result is a 4x4 grid video with a success-rate overlay and a red frame whenever the Safety
Plane clamps an action.

§28.7 gate 7 ("Franka pick&place end-to-end") was recorded **not met** in `docs/reviews/M1.md:30`
("franka names IR fixtures only; no e2e run exists"). No later review revisits it: a grep for
`gate 7` / `franka` / `pick&place` across `docs/reviews/*.md` returns that one row and nothing in
M2, M3 or M4. Plan V is the packet set that closes it, with two substitutions the user has already
decided: **SO-101, not Franka** (a low-cost arm the sim demo carries over to if one is bought), and
**cube-into-bin, not the generic pick&place**.

| Item | Packet | Oracle | After V |
|---|---|---|---|
| SO-101 scene + Task / Observation / Deployment IR | V0 | upstream-model cross-check, `es task compile`, MuJoCo load | verifiable anywhere (fetch: network) |
| Renderer driven from the env loop, frames to disk, image obs port | V0b | CPU-reference goldens, GPU/CPU agreement | PR tier CPU; GPU on a device |
| Scripted expert + demonstration dataset | V1 | success rate of the expert, LeRobot round trip | needs `mujoco` |
| Learning IR lowering trained in PyTorch, packed to `policy.esb` | V2 | lowered-module identity, loss decrease, `eval run` loads it | needs `torch` (server) |
| Perturbation suite + Safety Plane events per step | V3 | Evaluation IR report, non-vacuous clamp events | needs `mujoco` + `torch` |
| Frame grid -> overlay -> mp4 | V4 | byte-identical mosaic from fixed frames | mosaic anywhere; mp4 needs `cv2` |
| Any wall-clock, throughput or training-time figure | — | a measurement loop that does not exist | `Target / Status: unverified` |
| A policy that generalizes beyond the declared randomization | — | out of scope | `Target / Status: unverified` |

## 2. Findings that changed the plan

Everything in this section was read, not assumed. Line numbers are as of this note.

### 2.1 There is no Rust physics stepper; the backend is a Python subprocess

`PhysicsBackend` (`crates/es-physics-core/src/backend.rs:219-255`) has three production impls
(`mujoco.rs:218`, `mjwarp.rs:175`, `newton.rs:200`) and all three are `Command::new(python).args(["-c",
script])` speaking line-delimited JSON (`crates/es-physics-backend/src/proc.rs:201-229`); floats cross
as decimal JSON (`proc.rs:97-105`). There is no `es-physics-cpu` or `es-physics-gpu` crate;
`crates/es-physics-core/src/lib.rs:7-8` says so ("It contains no engine").

Per control step, `es eval run` pays **three MuJoCo round trips plus one Torch round trip**:
`set_ctrl` (`crates/es-env/src/env.rs:258`), `step` (`env.rs:262`), an unconditional full `state` pull
(`crates/es-physics-backend/src/mujoco.rs:354`), and `policy.infer`
(`crates/es-eval/src/runner.rs:374` -> `crates/es-policy/src/torch_runtime.rs:353`). The code says this
is deliberate and is not a throughput path (`mujoco.rs:5-6`, `torch_runtime.rs:6-8`).

**Consequence.** The 4x4 grid is **16 independent single-env episodes, mosaicked offline** — not a
16-env batch. `Evaluation::run` hardcodes `BatchDomains::single_env()` (`crates/es-eval/src/runner.rs:132`
-> `crates/es-env/src/scheduler.rs:58-65`), `es eval run` has no `--envs` flag
(`crates/es/src/cmd/eval.rs:193-229`), and `MuJoCoCpuBackend` declares `max_envs: 1`
(`mujoco.rs:43`) with `n_envs > 1` emulated by a serial Python loop over independent `MjData`
(`crates/es-physics-backend/python/mujoco_ref.py:138-140`). Raising the batch domain is a separate
concern from this demo and plan V does not touch it.

No packet in plan V states a step rate, an episode time or a training time. `es bench` prints every
field as `unmeasured` (`crates/es/src/cmd/bench.rs:163-196`) and §12.4 abolishes the single figure.

### 2.2 `es-render` is a complete, headless, orphan island

No crate in the workspace depends on `es-render`. It is nonetheless real: `Renderer::new` /
`upload_scene` / `render(&[CameraView]) -> Atlas` (`crates/es-render/src/renderer.rs:175`, `:234`, `:315`),
a closed-form tile atlas for N cameras (`crates/es-render/src/atlas.rs:52-57`, §15.2), GPU->host readback
through `Atlas::read_tile` (`renderer.rs:82-138`) and `Buffer::download` (`crates/es-gpu/src/buffer.rs:152-163`),
and CPU-reference goldens in `tests/golden/render/` that the GPU path must match bit for bit
(`crates/es-render/tests/render.rs:197-209`).

**Headless is confirmed at the Vulkan level, not assumed.** The instance is built with no enabled
extensions (`crates/es-gpu/src/instance.rs:59-64`), the device with none (`crates/es-gpu/src/device.rs:80-82`),
the queue is selected on `COMPUTE` alone (`instance.rs:84`), and there is no surface or swapchain module
(`crates/es-gpu/src/lib.rs:21-28`). `winit` appears only in `es-editor`. Measured on the oracle server
(2026-09-14): `vulkaninfo --summary` reports `NVIDIA GeForce RTX 4090` at `apiVersion 1.4.341` and a
`llvmpipe` fallback, over SSH with no display.

Four gaps block the demo, and the source names all four rather than faking numbers:

| Gap | Evidence |
|---|---|
| `Tile` -> disk | the only `fs::write` is in an `#[ignore]`d golden generator (`crates/es-render/tests/render.rs:173`) |
| `Renderer` -> Observation IR `ImageInput` | `crates/es-eval/src/runner.rs:421-428` returns `EvalError::Plan` for an image input: "Feeding it zeros would produce a number, and a wrong number in the §10.1 table is worse than no table" |
| mesh geoms | `crates/es-render/src/scene.rs:201` refuses `Shape::Mesh` ("needs an asset resolver") |
| animation | `TriScene::from_scene` bakes **static** `world_poses(scene)` into triangles (`scene.rs:56-65`) |

`MultiViewPack` (`crates/es-ir/src/observation.rs:378-381`) is declared but rejected by plan lowering
(`crates/es-compile/src/plan.rs:545`), so the demo uses **one camera**.

### 2.3 The MJCF pipeline cannot carry meshes, includes, keyframes or cameras

The scene path is MJCF text -> `SceneDesc` (`crates/es-assets/src/mjcf/mod.rs`) -> MJCF text again
(`crates/es-physics-backend/src/mjcf_out.rs`) -> `mujoco.MjModel.from_xml_string`. Three narrow points:

- **`<include>` is rejected outright**, at the root (`crates/es-assets/src/mjcf/mod.rs:247-251`) and inside
  a body (`mod.rs:569-573`). Menagerie's `scene_box.xml` is an `<include>` wrapper, so it cannot be used
  as-is. The demo scene must be one flat file.
- **A `type="mesh"` geom is `PhysicsError::Unsupported`** (`mjcf_out.rs:295`, pinned by
  `mjcf_out.rs:551-556`), and `es-render` refuses it too (`scene.rs:201`). A mesh-bearing model therefore
  reaches neither MuJoCo nor the renderer through this pipeline.
- `<keyframe>`, `<equality>`, `<contact>` and `<visual>` are parsed to a warning and dropped
  (`mod.rs:208-218`); sites, cameras, materials and tendons are not re-emitted (`mjcf_out.rs:9-11`).

This decides the asset policy (section 4): a **primitives-only** derivative, not the upstream mesh model.

### 2.4 The data path records no pixels, and writes LeRobot v2.1

`es loop collect` is real, not a stub (`crates/es/src/cmd/loop.rs:146-155`, `:223`), and steps a real `Env`
(`crates/es-data/src/collect.rs:355-357`). It prints `SKIPPED` + exit 3 unless a Python with `mujoco` and a
Python with `torch` are both found (`loop.rs:194-202`).

Per step it records `observation.state` (`qpos ‖ qvel` as f32), `action` (the **post-Safety-Plane** control),
`reward`, `intervention`, `action_source` and `timestamp` (`collect.rs:514-552`). `Episode`
(`crates/es-env/src/episode.rs:73-96`) has no image buffer at all. A declared camera yields a `video` feature
in `info.json` plus a warning, an empty video map and a dangling mp4 reference by design
(`collect.rs:496-510`, `:549-550`).

Format: `codebase_version: "v2.1"`, hardcoded (`crates/es-data/src/lerobot/meta.rs:148`), layout
`data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` (`meta.rs:155`), written with
`parquet 59.3` with `default-features = false` — no `arrow`, **uncompressed**
(`crates/es-data/Cargo.toml:22-24`, `columns.rs:146`). `docs/api-notes/lerobot-dataset.md:3-8` states
plainly that **no `lerobot` version is pinned for the dataset format** and the whole note is reconstructed
from memory. There is no oracle that round-trips a written dataset through the real package: the only
`lerobot` import in the repo is the ACT policy oracle `crates/es-policy/python/act_ref.py`.

**Consequence.** V1 must (a) teach collection to write frames, and (b) add the missing oracle: read the
dataset back with the real `lerobot` on the server. The v2.1 / v3 question is settled by that oracle, not
by argument — see section 12 open question 2.

### 2.5 There is no training loop anywhere, and `lower_act` cannot be evaluated

A grep for `AdamW|optimizer|backward|def train|fn train` finds a digest **slot**
(`crates/es-data/src/identity.rs:194`), a zero placeholder (`collect.rs:654`) and a shader optimizer. There
is no `es learn` and no `es train` subcommand (`crates/es/src/main.rs:48-61`).
`crates/es/src/cmd/loop.rs:5-7` says it out loud: "`es loop train` deliberately does not exist: the
training run is on the Python side of §2.3's split".

Two lowering paths exist and only one is reachable from `eval run`:

| Path | Input | Reached by |
|---|---|---|
| `lower_to_torch(graph: &LearningGraph)` (`crates/es-policy/src/lower/torch.rs:334`) | the IR | `TorchRuntime::load` (`torch_runtime.rs:340`), hence `es eval run --policy` (`crates/es/src/cmd/eval.rs:387-393`) and `es loop collect` (`loop.rs:208`) |
| `lower_act(cfg: &ActConfig)` (`crates/es-policy/src/lerobot.rs:467`) | a LeRobot `config.json` | `TorchRuntime::load_lowered` only, whose single caller is `lower_act` itself (`torch_runtime.rs:278-284`) |

`docs/reviews/M4.md:101-103` records the bypass. The consequence for plan V is sharper than the review
states: **a bundle produced through `lower_act` cannot be run by `es eval run`**, because `load`
re-lowers from the bundle's `LearningGraph`.

**Decision (section 6).** Plan V trains the **IR-owned ACT-shaped graph**, not LeRobot's ACT. The
`LearningGraph` already expresses it — `es_ir::learning::testing::act_like`
(`crates/es-ir/src/learning.rs:1090-1152`) builds `VisionEncoder -> StateEncoder -> Fusion ->
TemporalEncoder -> PolicyHead -> ActionChunker` with `ArchKind::Act`, and `lower_to_torch` lowers
every one of those nodes (`lower/torch.rs:551`, `:587`, `:617`, `:659`, `:756`, `:769-807`). This keeps
§1.4's "the same IR run in PyTorch is the ground truth" literally true — the module Python optimizes is
byte-for-byte the module `eval run` infers with — and it leaves the `lower_act` debt exactly where M4 put
it. Plan V does not fix it; open question 1.

### 2.6 The Safety Plane is already in the loop, per step

`Evaluation::run` builds one `SafetyPlane<NJ, H>` per cell from the Deployment IR
(`crates/es-eval/src/runner.rs:149`) and calls `safety.validate` between `policy.infer` and `env.step`
(`runner.rs:374-385`). `SafetyCounters` and `ViolationKind` are already what metrics read
(`crates/es-eval/src/metrics.rs:13`, `:144`), and `es-data` already classifies a step's action source from
counter deltas — `Policy | Clamped | Fallback | Human` (`crates/es-data/src/collect.rs:461-471`).

**So the red overlay needs no new safety surface.** V3 emits the existing per-step classification
alongside the frame index; nothing calls `validate` differently, nothing is disabled (INV-12), and
`validate`'s signature is untouched (INV-13). `es-safety` gains no dependency, so INV-11 is unaffected.

### 2.7 Randomization: Task IR can move the cube, Evaluation IR cannot

`RandomizationPlan` resolves a target to `Target::Qpos(i)` / `Qvel(i)` and writes it at reset
(`crates/es-env/src/randomize.rs:19-29`, `:109-110`). The cube is a free joint, so its world pose **is**
`qpos` — cube-pose randomization is a Task IR `Randomization` node (§6.3) and works today.

The Evaluation IR side does not: `PerturbationKind::ObjectPose` is `Unsupported` because "`Env::reset`
takes no state override" (`crates/es-eval/src/perturb.rs:196-200`), and every visual kind
(`LightIntensity`, `LightDirection`, `ColorTemperature`, `Occluder`, `CameraExtrinsic`,
`CameraIntrinsic`) is `Unsupported` for want of a renderer (`perturb.rs:180-195`). Five kinds do work:
`ObservationDelay`, `ActionDelay`, `FrameDrop`, `TorqueNoise`, `Backlash` (`perturb.rs:159-179`).

`Target::Scale(BodyMass | GeomFriction | ActuatorGain)` is **drawn and recorded but never pushed into the
backend** — "`PhysicsBackend` has no parameter API" (`randomize.rs:24-26`). Mass and friction
randomization is therefore a recorded lie today, and V0 must not declare it.

**Consequence.** V3's suite is: a nominal cell, the five supported perturbation kinds, and — once V0b
lands a renderer — the two lighting kinds, which become implementable because the reason they were
refused disappears. V3 implements `LightIntensity` and `LightDirection` only; `ColorTemperature`,
`Occluder`, `CameraExtrinsic` and `CameraIntrinsic` stay `Unsupported` (open question 4).

### 2.8 There is no inverse kinematics anywhere

A grep for `inverse_kinematic|jacobian|damped_least|mj_jac|pinv` over `crates/`, `python/` and `xtask/`
returns only MJCF's `<option jacobian=>` solver attribute (`crates/es-assets/src/mjcf/mod.rs:422-426`,
`crates/es-assets/src/scene.rs:370`). `es-math` is `approx`, `conventions`, `reduce`, `scalar`, `simd`;
`es-actuator` is the transduction model only.

But forward kinematics is already free: `StateView` carries `xpos` (`n_envs * nbody * 3`) and `xquat`
(`n_envs * nbody * 4`) for every body (`crates/es-physics-core/src/backend.rs:165-168`), and `ModelInfo.body`
maps a body id to its row (`backend.rs:126-127`).

**Decision (section 5).** The expert uses **closed-form IK**, not damped-least-squares and not `mj_jac`.
SO-101's chain makes this the smaller solution, not a shortcut: `shoulder_pan` is a base yaw and
`shoulder_lift`, `elbow_flex`, `wrist_flex` are three successive pitches in the plane that yaw sweeps
(section 4.2). That is a base rotation plus a planar 3R arm with a chosen approach pitch — a closed form
in tens of lines, with no Jacobian, no iteration, no convergence failure mode and no new dependency. A
DLS solver would need a Jacobian the backend does not expose, and `mj_jac` on the Python side would put
the expert outside the Rust core for no gain.

### 2.9 The oracle server: measured 2026-09-14

Over SSH, read-only; nothing was installed or changed.

| Thing | Result |
|---|---|
| GPU | `NVIDIA GeForce RTX 4090`, 24564 MiB |
| Vulkan | `vulkaninfo` at `~/VulkanSDK/x86_64/bin`, `apiVersion 1.4.341`, headless, no display |
| `ffmpeg` binary | **absent** (`command -v ffmpeg` empty; `ffprobe`, `gst-launch-1.0`, `convert` also absent) |
| FFmpeg shared libs | present: `libavcodec62` .. `libavutil60`, `7:8.0.1-3ubuntu2` |
| `~/venvs/es` | `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `mujoco 3.13.0`, `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `newton 1.6.0`, `diffusers 0.40.0`, `pillow 12.3.0`, `numpy 2.5.2` |
| `~/venvs/es-lerobot` | `lerobot 0.6.1`, `torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `opencv-python-headless 4.13.0.92`, `pillow 12.3.0`, `numpy 2.2.6` |
| `torch.cuda.is_available()` | **`False` in both venvs** — both are `+cpu` builds |
| `import lerobot.datasets` | **`ImportError: 'datasets' is required but not installed`**; `pyarrow` absent too |
| `cv2.VideoWriter` mp4 | `mp4v` opens and writes (10 frames -> 1450 bytes); `avc1` and `H264` **fail** — the only H.264 encoder linked is `h264_v4l2m2m`, which finds no device |
| MuJoCo Menagerie | not present on the server |

Three of these change the plan and two are human prerequisites:

1. **No ffmpeg binary.** The encoder is `cv2.VideoWriter` from `~/venvs/es-lerobot`, codec `mp4v`
   (MPEG-4 Part 2) in an `.mp4` container. Verified above. Not H.264 (open question 5).
2. **Torch is CPU-only on a 4090.** Training an ACT-shaped graph with a ResNet-18 backbone on CPU is
   the schedule risk of plan V. Human prerequisite; section 6.4 keeps the demo's image small so the
   demo is runnable either way, and states no training time.
3. **`lerobot[dataset]` is not installed**, so `lerobot.datasets` cannot even import. V1's dataset
   oracle is `SKIP` until a human installs it. Human prerequisite.

### 2.10 Context budget

`cargo xtask context-budget`, 2026-09-14: every crate `OK`; `es-ir` is at **5,947 / 6,000** code lines,
53 from the target. The budget counts `*.rs` under `src/` only, outside inline test modules
(`xtask/src/context_budget.rs:83-156`) — fixtures, `tests/`, Python scripts and goldens are free.

**Therefore no packet in plan V adds a line to `es-ir` or `es-ir-types`.** Everything the demo needs is
already expressible: `TaskNode::Terminate { kind: TerminationKind::Success }`
(`crates/es-ir-types/src/expr.rs:104-108`), `Randomization` targets (section 2.7), `MetricSpec::SuccessRate`
(`crates/es-eval/src/metrics.rs:32`, `:97-100`), and the six-node ACT-shaped `LearningGraph` (section 2.5).
Each packet's `forbidden` says so.

## 3. Pipeline

```
robotstudio_so101 (upstream, pinned)          V0
  └─ derive, primitives only ─▶ tests/fixtures/mjcf/so101_pick_place.xml
                                     │
                    es-assets ──────▶ SceneDesc ─┬─▶ mjcf_out ─▶ MuJoCo (python subprocess)
                                                 └─▶ TriScene ─▶ es-render                V0b
                                                                    │
 task.toml / observation.toml / deployment.toml (V0) ──▶ policy.esb │
                                     │                              │
                 es loop collect --expert (V1) ◀── ScriptedExpert ──┤
                                     │                              │
                     LeRobot v2.1 dataset + frames                  │
                                     │                              │
             es policy lower ─▶ EsPolicy(nn.Module) ─▶ train_act.py (V2, server)
                                     │
                     model.safetensors ─▶ es policy pack ─▶ trained policy.esb
                                     │
             es eval run --policy --frames (V3) ─▶ report.json + frames/ + events.json
                                     │
                        es video (V4) ─▶ mosaic + overlay ─▶ demo.mp4
```

Four artifacts are the demo's evidence: `report.json` (success rate per suite), the frame directories,
`events.json` (the per-step action source) and `demo.mp4`.

## 4. Asset policy

### 4.1 What is vendored and what is fetched

Upstream is **`robotstudio_so101`** in `google-deepmind/mujoco_menagerie`, **Apache-2.0** ("This model is
released under the Apache License 2.0", `robotstudio_so101/README.md`), requiring MuJoCo 3.1.3 or later.
Pinned commit for that path: **`ac6b2b09983786f3036cab1000221017fa2193b4`** (2026-05-18). The other
candidate, `trs_so_arm100`, is the SO-ARM100; `robotstudio_so101` is the SO-101 the user chose.

The directory is ~17.5 MB: `so101.xml` 16,670 B, `scene.xml`, `scene_box.xml`, `so101.png` 1,010,575 B,
and `assets/` — 19 STL files totalling 17,230,580 B.

**Neither vendoring the directory nor fetching it at runtime is the answer, because section 2.3 says the
mesh model cannot traverse this pipeline at all.** The policy is:

- **Vendor** `tests/fixtures/mjcf/so101_pick_place.xml` (~10 KB, one flat file, no `<include>`), a
  primitives-only derivative: the upstream kinematic chain, joint axes, ranges, inertials and
  `class="collision"` primitive geoms, with every `class="visual"` mesh geom dropped, plus the cube, the
  bin and one camera. Alongside it, `tests/fixtures/mjcf/so101_pick_place.LICENSE` (the upstream
  Apache-2.0 text) and `so101_pick_place.PROVENANCE.json` — upstream repo, path, commit, the blake3 of
  upstream `so101.xml`, and the derivation rules. Dropping the visual meshes is exactly what makes the
  file small enough to vendor; the collision geoms are already primitives. As built, each link also
  gains **one** visual-only primitive (`contype=0 conaffinity=0`, named `<link>_shell`) in place of the
  mesh geoms it lost, so the arm is still recognisable to the renderer; `camera_mount_shell` carries the
  0.012 kg of the mesh it replaces, and every other link has an explicit `<inertial>`, so no added shell
  changes a mass property. The manifest's `derivation` array is the authoritative list, and
  `so101_provenance.rs` is what checks it.
- **Fetch, never vendor, the 17 MB.** V0's provenance oracle fetches upstream `so101.xml` at the pinned
  commit, checks its blake3 against the manifest, and asserts our derivative is kinematically identical:
  same joint names in the same order, same axes, same ranges, same body offsets and orientations, same
  masses and inertias, to an exact `f64` comparison of the parsed `SceneDesc`. No network or no cache ->
  `SKIP` with the reason. Downloaded bytes land under `target/`, never in the tree.

That is the "fetch pinned by commit + blake3" arm of the asset policy, used as an **oracle** rather than
as a build step: nothing in the demo path needs the network.

### 4.2 The kinematic chain, from upstream

Six joints, six `position` actuators, one camera (`wrist_cam`), sites `baseframe` and `gripperframe`, no
keyframe in `so101.xml`. `<compiler angle="radian" meshdir="assets" autolimits="true"/>`,
`<option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20"
impratio="10"/>`.

| Body | pos | quat (wxyz) | joint | axis | ctrlrange |
|---|---|---|---|---|---|
| `base` | `0 0 0` | `1 0 0 0` | `shoulder_pan` | `0 0 1` | -1.91986 .. 1.91986 |
| `shoulder` | `0.0388353 0 0.0624` | `0 0 -1 0` | `shoulder_lift` | `0 0 1` | -1.74533 .. 1.74533 |
| `upper_arm` | `-0.0303992 -0.0182778 -0.0542` | `1 -1 -1 -1` | `elbow_flex` | `0 0 1` | -1.69 .. 1.69 |
| `lower_arm` | `-0.11257 -0.028 0` | `1 0 0 1` | `wrist_flex` | `0 0 1` | -1.658063 .. 1.658063 |
| `wrist` | `-0.1349 0.0052 0` | `1 0 0 -1` | `wrist_roll` | `0 0 1` | -2.7438473 .. 2.84121 |
| `gripper` | `5.55112e-17 -0.0611 0.0181` | `0.0172091 -0.0172091 0.706897 0.706897` | `gripper` | `0 0 1` | -0.174533 .. 1.74533 |
| `moving_jaw_so101_v1` | `0.0202 0.0188 -0.0234` | `1 1 0 0` | (jaw, driven by `gripper`) | | |

`site gripperframe` is `pos="0.012 -0.000218 -0.098127"` in `gripper`. After the fixed body rotations,
`shoulder_lift`, `elbow_flex` and `wrist_flex` are three successive pitches in the vertical plane that
`shoulder_pan` sweeps — the planar 3R of section 2.8. The exact link lengths the IK uses are **derived by
the provenance oracle from the parsed `SceneDesc`, not transcribed by hand** into the Rust source: a
transcribed constant is a number nobody can check.

The demo scene adds, in the same flat file: a free-joint cube (the upstream `scene_box.xml` uses
`size="0.02 0.02 0.03"` at `pos="0.5 0 0.03"` with `condim="3" friction="1 .03 .003"`), a bin made of a
floor plate and four thin box walls, one fixed overhead `<camera>`, one `<light>`, and a geom whose name
ends in `_light` if the path-traced render path needs an emitter (`crates/es-render/src/scene.rs:30`
infers emission from the name suffix, because `SceneDesc` has no emissive material field).

## 5. The scripted expert (V1)

### 5.1 Placement

`crates/es-env/src/expert.rs`. `es-env` is layer 9: it already owns the step loop, already holds
`ModelInfo` and can read `StateView`, and it is below `es-data` (10) and the `es` CLI, both of which need
it. `es-math` (0) knows nothing of a scene; `es-actuator` (3) is the transduction model. A concrete
`ScriptedExpert` struct — **no new trait** (INV-17). It is consumed through the intervener hook
`Collector::run` already takes, which the CLI currently passes as `|_, _, _| None`
(`crates/es/src/cmd/loop.rs:152-153`), so a demonstration is recorded with `action_source = Human` and an
`intervention` column by machinery that already exists and is already tested
(`crates/es-data/tests/loop_learning.rs:569`).

### 5.2 Inverse kinematics

`so101_ik(links: &Links, target: Vec3, approach_pitch: f64) -> Option<[f64; 4]>`:

1. `shoulder_pan = atan2(target.y, target.x)`, rejected if outside the joint range.
2. In the plane that pan sweeps, subtract the wrist offset along the approach direction to get the
   3R wrist centre; solve the 2R triangle for `shoulder_lift` and `elbow_flex` by the law of cosines,
   elbow-up branch; `wrist_flex = approach_pitch - shoulder_lift - elbow_flex`.
3. `None` — never an approximation — when the target is out of reach or any angle leaves its range.
   `None` at a waypoint aborts the episode as a failed demonstration; it is never clamped silently
   (§17.2's rule, applied to kinematics).

`wrist_roll` is held at 0 and `gripper` is commanded by the state machine, so the arm is treated as the
4-DoF positioner it is. Trigonometry uses `es_math::approx` (§3.4 forbids the host `libm` on any path a
golden depends on).

### 5.3 The waypoint state machine

Read from `StateView`: the cube body's `xpos`/`xquat` row and the `gripper` body's row, both via
`ModelInfo.body`. Six states, each a target pose plus a gripper command plus an exit predicate:
`Approach` (above the cube) -> `Descend` -> `Close` (hold for a fixed tick count) -> `Lift` ->
`Transport` (above the bin) -> `Release`. Transitions are on integer tick counts and position error
thresholds; the episode ends on the Task IR's own `Terminate` node, not on the expert's opinion.

Nothing here reads the wall clock or a global RNG. The per-episode variation is the Task IR's cube-pose
randomization (section 2.7), so the same `(task_hash, seed, episode)` produces the same demonstration.

### 5.4 Success

`cube inside the bin volume for N consecutive control steps`, as a Task IR `Terminate` node of kind
`Success` (`crates/es-ir-types/src/expr.rs:105`). The cube's free-joint `qpos` is its world pose, and
`es-env`'s task plan lowers `Source::Qpos` leaves (`crates/es-env/src/plan.rs:19-28`), so the predicate
is a plain `Qpos`/`Qvel` expression with no new node type.

The "for N consecutive steps" part is the one thing the cone cannot count: Task IR-D is a stateless
dataflow DAG. V0 resolves it the cheap way — the bin is deep enough and the velocity bound tight enough
that the predicate is only true once the cube has settled, and `N = 1`. If a settling counter turns out
to be needed it belongs in IR-C (§6, control), not in a new IR-D node: open question 3.

**What V0 measured, which narrows the predicate below "AABB and `|v|`".** Three limits of the *existing*
cone, none of which is an `es-ir` change:

1. **A cone leaf is one scalar, the joint's first index.** `Ctx::joint_leaf`
   (`crates/es-env/src/plan.rs:181-208`) takes `joints.first()` and binds `qpos[range.start]`. For the
   cube's free joint that is `x` alone; `y` and `z` are not addressable from a `TaskNode` at all, because
   no node reads `qpos[i]` by index (only `Randomization` / `ResetState` target *strings* do,
   `randomize.rs:141-153`). The predicate V0 authored is therefore **the bin's x span plus a settling
   bound**, and the scene is laid out so that span discriminates: the cube starts at x ≈ 0.24 and the bin
   interior is x ∈ [0.050, 0.170], with the bin's y span (±0.105) covering the reachable y at that x.
2. **There is no constant leaf and no absolute value.** `TaskNode` has no `Const`, and `Arith::Mul`'s
   right operand is dimensionless under the §5.4 unit algebra (`task.rs` `inputs()`), so neither
   `x - c` nor `v * v` is expressible. `Compare { rhs: Some(c) }` folds the only constant available, so
   `|v| < b` is written as two `Compare`s and an `And`, and the shaped reward is a `Normalize` of the
   cube's x (a reward term must be dimensionless or normalized, `TYPE-011`).
3. **`Normalize` and `Logic` are `TaskNode` variants the cone lowering does not yet handle.**
   `Ctx::lower` (`plan.rs:125-179`) covers `GetJointState`, `GetSensor`, `GetTime`, `Arith`, `Compare`
   and `Clamp`; anything else is `EnvError::Unsupported` by name. Adding `Normalize` and `Logic` there
   is **V1 work in `es-env`** (layer 9) and touches no IR: `Expr` already has `Logic` and the normalize
   is an affine `Arith`. V0's documents are authored against the node set, not against the lowering.

A 3-axis AABB needs one of: a `GetBodyPose` leaf in the cone, a `Slice` leaf over a free joint's 7-wide
`qpos`, or three `jointpos`-style sensors. All three are IR or `es-env` work beyond V0's scope — open
question 3 is widened to cover it.

**The same shape of gap on the observation side.** `capture` (`crates/es-eval/src/runner.rs:411-420`)
resolves an observation input name against `ModelInfo.qpos`, which is keyed by **joint**, or against
`ModelInfo.sensor`; there is no `qvel` path and no body path. The Task IR channel and the Observation IR
`StateInput` must name the *same* id for the cross-IR check (`XIR-002`), and `DEP-031` compares the
Safety Plane envelope against that channel's `dof`, so a 6-joint robot is one channel of `dof = 6` named
by the robot's body — which `capture` then cannot resolve. `crates/es-eval/tests/evaluation.rs:427-440`
side-steps this by naming a body in the Task IR and a joint in the Observation IR, which never passes
`cross::check`. V1 owns the fix (widen `capture` to a joint *set* and to `qvel`, or emit `jointpos` /
`jointvel` sensors into the scene); V0 declares the honest 6-wide state and records the gap here.

## 6. Learning and training (V2)

### 6.1 The graph

The six-node ACT-shaped `LearningGraph` of section 2.5: `VisionEncoder { ResNet18 }` -> `StateEncoder` ->
`Fusion` -> `TemporalEncoder { Transformer }` -> `PolicyHead { Regression }` -> `ActionChunker`, plus
`Normalizer` nodes, which `lower_to_torch` lowers in the IR rather than as generated Python
(`crates/es-policy/src/lower/torch.rs:769-807` — the thing `lower_act` does not do).

It is not LeRobot's ACT. It has no VAE and no DETR decoder, because §8.3's `TemporalEncoder { Transformer }`
carries only a width (the reason `lower_act` exists at all, `crates/es-policy/src/lerobot.rs:9-14`).
Calling it "ACT" in the demo would be a small lie; the design note and the packets call it the
**ACT-shaped Learning IR policy**, and the demo copy should too.

### 6.2 Lower, train, pack

Three steps, with the Rust/Python line exactly where §2.3 puts it:

```
es policy lower   --policy untrained.esb --out build/
es dataset bake   --policy untrained.esb --out baked/ --frames tiles/ ds/     # V2b
<py> python/es/train_act.py --module build/ --baked baked/ --out model.safetensors
es policy pack    --policy untrained.esb --weights model.safetensors --out trained.esb
```

(The `bake` line is V2b's; section 7.9 says why V2's three steps were not enough.)

- `lower` writes `TorchModule.source` verbatim plus `contract.json` — the `weight_keys` and
  `weight_shapes` the lowering declares (`crates/es-policy/src/lower/torch.rs:297`) and the
  `lowering_hash`.
- `bake` runs every recorded frame through the bundle's own Observation IR (V2b, section 7.9).
- `train_act.py` `exec`s that source, builds `EsPolicy()`, reads the **baked** observation set, runs the
  optimizer, and writes `safetensors` **keyed exactly by `contract.json`**. It knows nothing about the IR,
  is not allowed to define a layer, and after V2b implements no Observation IR node either.
- `pack` reads the safetensors, checks every key and shape against the contract, refuses anything else,
  and rewrites the bundle with the new `weights.safetensors` and a recomputed manifest
  (`crates/es-compile/src/bundle.rs:467`, `:513`). `safetensors` only; no pickle path is added anywhere
  (INV-16), and `WeightsSource` gains no variant (`crates/es-policy/src/runtime.rs:27-32`).

`es eval run --policy trained.esb` then works with **no change to the eval path**: `TorchRuntime::load`
re-lowers the same `LearningGraph` and gets the same module (`torch_runtime.rs:340`).

### 6.3 The training oracle

Training is the one part of plan V that §1.4 cannot judge by "matches a reference". It is judged by four
executable facts instead, none of which is "the loss looked good":

1. **The module is the IR's.** `train_act.py` never constructs a layer; the oracle diffs the `es_policy.py`
   it loaded against `lower_to_torch(bundle.learning).source` and requires equality.
2. **The weights fit the contract.** `es policy pack` refuses a missing key, an extra key or a wrong shape;
   the oracle asserts all three refusals.
3. **It learns something.** On a fixed tiny dataset and a fixed seed, final training loss is below a
   pinned fraction of the initial loss. A threshold, not a golden — CPU PyTorch is not bitwise portable.
4. **It round-trips.** The packed bundle loads in `TorchRuntime` and produces a finite action chunk of the
   declared shape for a fixed observation.

Absent `torch`, all four `SKIP` with the reason, per the house rule.

### 6.4 Size, and the CPU-only risk

Section 2.9: both server venvs are `+cpu` builds. The demo's image is therefore small (V0 declares the
`ImageSpec`; 96x96 or 128x128, one camera — `MultiViewPack` is rejected anyway, section 2.2) and the chunk
horizon short. Whether `_backbone("resnet18", 512)` (`crates/es-policy/src/lower/torch.rs:925`) wants
ImageNet-pretrained weights, and therefore a network fetch at training time, is **unverified** and is open
question 6. No training time is stated anywhere (§12.4).

## 7. Rendering in the loop (V0b)

### 7.1 Animation

`TriScene::from_scene` bakes static body poses (section 2.2). V0b adds one function beside it:

```rust
pub fn from_scene_with_poses(scene: &SceneDesc, world: &BTreeMap<StableId, Pose>) -> Result<TriScene, RenderError>
```

— identical to `from_scene` except that a body's world pose comes from `world` when present. The caller
builds that map from `ModelInfo.body` and `StateView::xpos`/`xquat`. **`es-render` gains no dependency**:
it takes a map of poses, not a `StateView`, so layer 5 still knows nothing of layer 3.

Ceiling, stated rather than hidden: the whole scene is re-tessellated and re-uploaded every rendered
frame. `es-render` scans a flat triangle array with no acceleration structure and documents a few-hundred
triangle ceiling (`crates/es-render/src/lib.rs:11-14`); the demo scene is an arm of primitives, a cube and
a five-plate bin. If that ever stops being true the fix is a per-body transform buffer in the shader, not
a bigger CPU loop.

### 7.2 Frames to disk

`Tile` already knows its bytes (`crates/es-render/src/atlas.rs:147-153`) and that layout is already the
golden format (`tests/golden/render/*.bin` + `.json` sidecar). V0b writes exactly that, one pair per
frame: `frames/<cell>/<NNNNNN>.bin` plus one `frames/<cell>/layout.json`.

**No PNG.** There is no PNG or image encoder in the workspace and none in `[workspace.dependencies]`.
Adding one would buy nothing: the only consumer is V4's mosaic, which is numpy. The raw frame is the
byte-identical artifact; PNG would be a second encoding of it. Open question 7 if a human wants `.png`
for eyeballing.

### 7.3 The image observation port

Two edits, both narrow:

- `crates/es-env`: an optional `render` cargo feature (off by default, mirroring `es-ros2/zenoh`) adding
  `es-render` (5) and `es-gpu` (2) — both below layer 9, so `cargo xtask layering` is satisfied. `Env`
  gains `Option<EnvRenderer>`, a concrete struct, not a trait (INV-17). With the feature off, `es-env`
  compiles and behaves exactly as today, and `es-runtime-embedded` never sees Vulkan.
- `crates/es-eval/src/runner.rs:421-428` stops returning `EvalError::Plan` for an image input **when a
  renderer is present**, and feeds the tile as the `ImageInput` port's tensor. With no renderer it keeps
  refusing, with the same message. Nothing is ever zero-filled.

INV-14 is untouched: the renderer produces the tile at exactly the declared size and refuses to resample
(`crates/es-render/src/renderer.rs:292-302`); `Resize`/`Crop` remain Observation IR nodes that transform
intrinsics. `es-render`'s own `ImageSpec` subset (`crates/es-render/src/view.rs:81-95`) is checked against
the Observation IR's declared `ImageSpec` at plan time and a mismatch is an error, not a rescale — the same
"what is not validated is not executed" rule §26.1 states and W1c applied to cameras.

### 7.4 As built (V0b), and two things the plan above got wrong

Both deviations are structural, not preferences:

- **`Env` does not hold the renderer.** `EnvRenderer<'gpu>` borrows the `Gpu`, so a field on `Env` would
  put a lifetime on `Env<B>` — a type `crates/es-data/src/collect.rs:297` and
  `crates/es-eval/src/runner.rs:143` both name, and neither is V0b's to edit. The renderer is therefore
  the caller's, and `EnvRenderer::frame(&ModelInfo, &StateView, env)` takes the state the caller already
  has (the acceptance signature already had that shape). `Env` and `domains.rs` are untouched, which is
  why "with the feature off, byte-identical behaviour" is trivially true rather than tested.
- **`es-eval` has no `render` feature.** It takes a *frame source* —
  `es_eval::runner::FrameSource = dyn FnMut(&ModelInfo, &StateView) -> Result<Vec<u8>, String>` — through
  `Evaluation::run_with_frames`, and `Evaluation::run` passes `None`. Layer 10 therefore never links
  Vulkan even to refuse an image, and the refusal path is testable on any machine. The caller wires
  `EnvRenderer::frame` into that closure; V3 is where the CLI does it.

**Finding for V1/V2.** V0's `observation.toml` declares the `ImageInput` *tensor* as `F32` `[3, 96, 96]`
(CHW) while its `ImageSpec` says `dtype = U8` — so the renderer's `Rgb8` `[96, 96, 3]` tile does not fit
that buffer, and `capture` refuses it naming both sizes. The `ImageSpec` itself matches the renderer
exactly (`the_declared_image_spec_is_checked_not_coerced` checks V0's own file). The fix belongs to
whichever packet next edits the fixture: declare the `ImageInput` as `U8 [96, 96, 3]` and put an
`ObservationNode::Dequantize` after it — `Op::Dequantize` (`crates/es-compile/src/plan.rs:73`) is exactly
"HWC u8 → CHW f32 /255" and already exists. V0b does not edit V0's fixtures.

### 7.5 As built (V1), and the four things V0's scene and documents got wrong

Everything below was measured on the oracle server against
`tests/fixtures/mjcf/so101_pick_place.xml`, and every number in `es_env::expert::demo_cfg` is
one of these measurements. The oracle is `expert_solves_the_pinned_seeds`
(`crates/es/tests/cli.rs`): eight pinned seeds through `es loop collect --expert`, with the
cube's final `x`, `y` **and** `z` read back out of the written dataset — because the Task IR's
own predicate sees `x` alone (section 5.4) and a success rate taken from it would be a rate
about the predicate, not about the expert.

**1. The bin was inside the robot, and out of its reach.** At V0's place — centred on the +X
axis at `x = 0.11` — `bin_wall_nx` intersected `shoulder_holder_col`, the shoulder's own
collision box, by 11 mm: MuJoCo reported the contact at the rest pose and `shoulder_pan` could
not turn at all. Beyond that, *no point above the bin's interior was in the arm's workspace*:
with a 0.16 m tool and SO-101's `wrist_flex` range, nothing at `x < 0.14`, `z > 0.04` has an
elbow-up solution. The bin now sits beside the arm (interior `x ∈ [0.09, 0.19]`,
`y ∈ [-0.15, -0.05]`), which is reachable, clear of the shoulder, and wholly inside the
overhead camera's frustum at wall height — the last of which the rendered golden depends on:
placing it so the wall tops straddled the frustum edge made the GPU frame differ from the CPU
reference in 3 of 27,648 bytes.

**2. The cube is 25 mm, not 30.** One jaw is fixed. The gap is centred on the tool site only
at one opening, and a cube approached with the base yawed presents its diagonal: a 30 mm cube
left about 2 mm of clearance at the near corner of the draw, which the arm loses on the way
down and shoves the cube instead of grasping it. `pos_tol` also came down from 0.035 rad to
0.01 (about 2 mm at the tool) for the same reason.

**3. The success predicate fires while the cube is still in the jaws.** "Cube `x` inside the
bin's span and at rest" is just as true of a cube *carried* across the bin as of one lying in
it, and V1 measured exactly that: 8 of 8 episodes ended `Success` with the cube 10 cm in the
air. The cone gains one more scalar it *can* read — the gripper's own joint — so the predicate
is now "in the bin's x span, at rest, **and not being held**". The threshold is 0.6 rad because
a jaw holding the 25 mm cube stalls at about 0.30 however hard it is told to close; the
demonstration therefore opens to 0.9, not 0.4. The expert also gained a `Lower` stage: released
from carry height the cube lands on the wall as often as in the bin, and — since the predicate
cannot see `z` — the episode ends with the cube still falling.

**4. A step command is not a demonstration.** `execute_chunk = 10` at 5 Hz inference with
`TemporalEnsemble`: the expert's per-tick position target reaches the actuator as a chunk, and
the Safety Plane turns each chunk into motion the envelope allows. A scripted driver that
jumps straight to its IK solution is therefore clamped on nearly every tick — measured: 39 of
40 steps, `Velocity` and `Acceleration` — and V0's `envelope_violation_rate { max_frac = 0.05 }`
latched the fallback about a second into every demonstration and froze the arm for the rest of
it. Two changes, both recorded: the expert emits a **ramped chunk** (`step_max`, `accel_max`,
read out of the Deployment IR it will be recorded through, with anti-windup against the
measured joint), and the demo's `max_frac` is widened to 0.9. Widening the envelope is the
sanctioned move; disabling the plane is not (`INV-12`), every clamp is still counted, and the
clamped steps are still recorded as `action_source = Clamped` — which is V3's material.

**What it measures.** 8/8 pinned seeds end in `Success` and 8/8 put the cube inside the bin's
three-dimensional interior, in about 350 control steps (7 s) of a 900-step (18 s) budget. The
IK itself round-trips through MuJoCo's own forward kinematics to 3.3e-8 m over 120 targets, and
two runs of one seed produce byte-identical `ctrl` rows.

**The dataset oracle answered, and the answer is a refusal.** `lerobot` 0.6.1 with the
`[dataset]` extra installed does not read `codebase_version: "v2.1"` at all — it raises
`BackwardCompatibilityError` and points at its own v2.1 → v3.0 converter. That settles open
question 2 in the direction of a v3 writer, and `docs/api-notes/lerobot-dataset.md` records it;
the writer is deliberately unchanged here, because the packet that changes it is the one that
implements v3.0.

### 7.6 As built (V2), and the six things the plan above did not know

`es policy lower` and `es policy pack` (`crates/es/src/cmd/policy.rs`) are the Rust ends of
section 6.2, and `python/es/train_act.py` is the optimizer between them. `es_policy::lower::Contract`
is the whole of what crosses spec 2.3's split besides the module source. Everything else below
is a finding, and four of the five are things the repo could not have known without running
this leg for the first time.

**1. The lowering emitted a module that could not run.** `VisionEncoder{ResNet18}` lowered to
`self.n0(inputs["rgb_overhead"])`, and the IR port is one image — `[3, 96, 96]`, no batch axis,
because spec 5.2 gives the inference domain its own batch size. Every torchvision backbone is
`nn.BatchNorm2d` all the way down, and that refuses a 3-D input outright: *"expected 4D input
(got 3D input)"*. No test had ever caught it, because the only graph `torch_equivalence.rs`
puts through PyTorch is a state-only MLP (`a_state_only_graph_lowers_to_torch_alone`), and
`es_ir::learning::testing::act_like` declares `[3, 224, 224]` — the same shape, the same
failure. **Every** `lower_to_torch` consumer with a vision encoder was broken, so the fix is one
line in the lowering rather than a workaround in this packet: run it as a one-image batch,
`self.n0(x.unsqueeze(0)).squeeze(0)`, which is exactly the trade the token-less
`TemporalEncoder` arm two branches down already makes. That widens V2's declared context by
`crates/es-policy/src/lower/torch.rs`; the `lowering_hash` of any graph with a vision encoder
moves, and no golden pins it.

**2. The collector had a frame sink and no caller, so V2 wired one.** V1 implemented
`CollectSpec`'s `FrameSink` in `es-data` and tested it there, but nothing ever handed the CLI
one: `crates/es/src/cmd/loop.rs` passed `None`, so every `es loop collect` run printed *"camera
`rgb_overhead`: `info.json` declares a video feature ... but no mp4 was written"* and the
dataset carried no pixels at all. `es loop collect --frames <dir>` now builds the `Gpu` and the
`EnvRenderer` inside the collect call — that borrow is exactly why `Env` does not hold a
renderer (section 7.4) — configures it from the Task IR's own image channel and its `ImageSpec`,
and writes `<dir>/<NNNNNN>.bin` + `.json`, one raw tile per control step, in the format the
render goldens and `es video mosaic` already use. It is behind a new `render` feature on `es`,
off by default, so the ordinary CLI still links no Vulkan (spec 4.2); a build without the
feature refuses `--frames` rather than writing a dataset with a hole in it. Measured on the
oracle server: a 352-step demonstration renders in 1.8 s, and the "no mp4 was written" warning
is gone.

**3. Training re-implements exactly one Observation IR node, and that is a debt.** The tile is
HWC `u8`; the Learning IR input is CHW `f32`. At inference the Observation IR's compiled plan
does that conversion with `Op::Dequantize` (`es_compile::plan::Op::Dequantize`). Python has no
plan runner, so `train_act.py`'s `dequantize` **is** that node, and it is the only place in this
repo where an IR node has a second implementation. It survives only because the demo's
Observation IR is exactly `ImageInput -> Dequantize -> sink`: the moment a second node appears
between the image port and the Learning IR input, training and inference silently disagree. The
real fix is an `es` step that bakes the observation plan over a dataset, which is a packet and
not a line.

**4. `es eval run` still cannot feed an image, so V2 reports no success rate.**
`Evaluation::run` passes `frames: None` and `capture` refuses an image input by name
(`crates/es-eval/src/runner.rs:475-481`) — deliberately, "a wrong number in the spec 10.1 table
is worse than no table". `Evaluation::run_with_frames` and the `renderer_cfg` this packet added
to `cmd/loop.rs` are between them everything V3 needs to close that, but closing it is V3's, and
so is the demo's `evaluation.toml`, which does not exist yet. The strongest honest claim V2
makes about the trained bundle is the packet's own: `es eval run --policy trained.esb` **gets
past `TorchRuntime::load`**, which is the assertion a `lower_act` bundle fails (section 2.5).

**5. `es loop collect --episodes N` only solves episode 0.** Measured on the oracle server:
`--episodes 50 --seed 1` ends `success 1, timeout 49`, while the same seeds run one episode at
a time (`--episodes 1 --seed s`, `s = 1..50`) end 49 successes and one timeout. Something the
scripted expert or the collector carries across an episode boundary is not reset —
`Collector::run`'s loop, `ScriptedExpert`'s waypoint state, or the arm's own pose. This is V1
code and V2 does not touch it; V2's dataset is 50 single-episode runs merged with
`es loop distill`, and the defect wants a V1 follow-up packet. It also means V1's oracle
(`expert_solves_the_pinned_seeds`, eight seeds at one episode each) could not have seen it.

**6. It is `policy_hash`, not `learning_hash`, that a checkpoint moves.** Section 6.2 and the
packet both said `policy_hash`, and that is right, but it is worth pinning: `learning_hash`
covers the graph (`canon_learning`) and does not move, while `policy_hash` hashes `WeightsRef`
into it (`LearningGraph::policy_hash`) and does. `pack_recomputes_the_manifest_rather_than_patching_it`
asserts all four slots: `policy` moves, `learning`, `task` and `observation` do not.

**Open question 6 is answered, and a smaller one opens.** `_backbone` is
`getattr(torchvision.models, name)()` with no `weights=` argument, so training is from scratch
and needs no network at any point. The Learning IR node carries `pretrained = true`, and the
lowering **ignores it** — a silent divergence between what the IR declares and what runs. It is
harmless today (the answer we want is "from scratch") and it should either be honoured or
rejected rather than dropped.

**What was measured, on the oracle server (RTX 4090, `~/venvs/es-lerobot-cuda`, torch
2.11.0+cu129).** 50 demonstrations, 50 of them `Success`, 17,697 frames: 3.4 MB of LeRobot v2.1
parquet and 485 MB of rendered tiles. A five-episode held-out set from seeds 101-105, 4 of 5
`Success`. Training ran 20,000 optimizer steps at `--batch 8` (the lowered module is
single-sample — `reshape(horizon, action_dim)` and `[:execute_chunk]` carry no batch axis — so a
batch is accumulated, one forward at a time), and the L1 chunk loss fell 0.727 -> 0.0507 ->
0.0309 -> 0.0171 at steps 1 / 1,000 / 5,000 / 20,000. Every wall clock is an observation and not
a claim, and no throughput figure is stated anywhere (spec 12.4). The 20k checkpoint is 61 MB and
is **not** committed; it lives on the server under `~/artifacts/plan-v/`, and its blake3 is
recorded in `docs/packets/M5/V2-act-training.md`.

**Deviations from the packet's acceptance, all deliberate.** `train_act.py` takes three flags the
packet did not list — `--checkpoint-at` (which also caps the run, so "20,000 steps" is exact),
`--loss-curve`, and `--frames` for the tiles the collector now writes — and its JSON line carries
five keys beyond the three the packet pinned.
`the_loss_falls` and `the_packed_bundle_round_trips` are one test, because fact 4 needs fact 3's
checkpoint and training twice to keep two names would be waste. `eval_run_accepts_the_trained_bundle`
moved to `crates/es/tests/cli.rs` as `policy_pack_output_is_accepted_by_eval_run`, because the
Evaluation IR fixture lives there and `tests/fixtures/visible-learning/` has no `evaluation.toml`
(that is V3's document). The round-trip also got stronger than the packet asked: it compares
`TorchRuntime::infer` against a *direct* PyTorch forward on a held-out observation at spec 8.9's
tier-4 fp32 tolerance, which covers the safetensors writer, `pack`'s validation, the
`nodes.N` -> `nN` rename and the wire protocol in one number.

### 7.7 As built (V1b): the LeRobot v3.0 export, and what the format actually requires

V1's dataset oracle answered with a refusal (section 7.5), and V1b is the repair: a converter,
not a change of writer. `es loop collect` still writes `codebase_version: "v2.1"` — V2's
training script reads it directly, and moving that format under a concurrent packet would have
been the expensive kind of correct. `es dataset export --lerobot-v3 <root> --out <dir>` is the
new command, `crates/es-data/src/lerobot/v3.rs` is all of it, and
`docs/api-notes/lerobot-dataset.md`'s "LeRobot v3.0" section is the format pinned against
`lerobot` 0.6.1 on the oracle server — read out of the installed package *and* executed against
it, file path by file path.

**Measured 2026-09-15**, `ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python cargo test
-p es-data --test lerobot_v3 -- --nocapture`: `RAN lerobot_v3_export`. `LeRobotDataset` opened
the export, reported `codebase_version 3.0`, 2 episodes / 7 frames, iterated all 7, and handed
back `observation.state` and `action` equal to the parquet we wrote to 1e-5 across every frame,
the camera as a `[3, 4, 6]` CHW tensor, and a pixel sum of 111.6706 against our own 28476/255 =
111.67059. The v2.1 oracle next door still prints its `SKIP`, with the refusal as the reason;
that is the honest state of a format 0.6.1 does not read.

**Four things the format required that the v2.1 note did not say.**

1. **A `shape: [1]` feature is a scalar column, not a length-1 list.** `DatasetInfo.__post_init__`
   turns every `shape` into a tuple and `get_hf_features_from_features` then branches on
   `shape == (1,)` before it branches on `len(shape) == 1`. v2.1's writer emits every feature as
   a 3-level LIST, so `reward`, `timestamp` and the four index columns all change shape on the
   way out. A `[6]` feature is still a list — a variable-size parquet `list<T>` is accepted
   where the declared `fixed_size_list<T>[6]` is, because `datasets` casts it, which is why the
   existing schema builders are reusable at all.
2. **The five bookkeeping columns are mandatory in `features`.** They drive
   `Dataset.from_parquet(..., features=...)`, so a column in the file and not in `info.json` is
   not a harmless extra. v2.1 left this `unverified`; v3.0 answers it.
3. **`meta/tasks.parquet` is read through `pandas`, and the task string must be its index.**
   `dataset_reader.py:352` is `self._meta.tasks.iloc[task_idx].name` — `.name` of a row is the
   *index* value. Two parquet columns are not enough: the file also carries the `pandas`
   key/value metadata in its footer naming `task` as `index_columns`. This is the one place the
   export writes a Python library's private serialization convention, and the api-note quotes
   the exact blob.
4. **Images do not need a video codec, and should not use one.** `dtype: "image"` stores an
   encoded image file inline in the data parquet as `struct<bytes, path>`; `dtype: "video"`
   needs a decoder, and on the oracle server `torchcodec` is installed but does not load
   (`libnppicc.so.12` missing) so it falls back to `pyav`. The export writes `image`, and no
   encoder runs anywhere — not `~/.local/bin/ffmpeg`, not from Rust, not from the oracle's
   Python side.

**No PNG crate, still.** Section 7.2 refused to add an image encoder because the only consumer
was numpy. `lerobot` is a second consumer and it wants a decodable file, so `v3.rs` carries
~70 lines of PNG: 8-bit truecolor, filter 0, and a zlib stream of *stored* deflate blocks. That
costs about 0.1% over the raw frame and buys a file PIL opens. It is checked by
`png_round_trips`, which parses the encoder's own output back — every chunk CRC, the LEN/NLEN
pairs, the Adler-32 — so the encoder has an oracle that needs no interpreter.

**Provenance is a sidecar, not `info.json`.** Spec §19.2 makes the export a derived artifact,
so `meta/es_provenance.json` records the source's `content`/`schema`/`split` hashes and its
`codebase_version`. It is not extra `info.json` keys because `DatasetInfo.from_dict` drops
unknown keys with a warning and `to_dict` would not write them back — provenance there would
not survive a LeRobot-side rewrite.

**Three deliberate omissions.**

- **`meta/stats.json` is not written.** `load_stats` returns `None` when it is absent and no
  read path needs it; it exists for training's normalization. Inventing statistics to fill a
  file is worse than leaving it out, and nothing in plan V trains through `lerobot`.
- **No chunk/file splitting.** One `data/chunk-000/file-000.parquet` and one
  `meta/episodes/chunk-000/file-000.parquet`. `data_files_size_in_mb` is advisory — the reader
  globs `data/*/*.parquet` — and an export currently buffers one dataset in memory, which is
  marked in the source as the ceiling it is.
- **A camera with no frames behind it is dropped, not exported.** `es loop collect` writes no
  pixels today (section 7.5), so `--frames <dir>` is where they come from:
  `<dir>/<name>/<NNNNNN>.bin`, the raw dump `EnvRenderer` already writes, one subdirectory per
  `observation.images.<name>`. Without it the feature is dropped and the command says so,
  rather than declaring an image feature nothing can load.

### 7.8 As built (V3), and the two things the plan above did not know

`es eval run --frames <dir>` is the whole of section 8's input: it renders the Task IR's one
image channel per control step, writes `<dir>/<suite>-<NN>/<NNNNNN>.bin` plus one
`layout.json` per cell, and writes `<out>/events.json` with one
`{ frame, tick, source, events }` record per frame — the directory shape and the JSON shape
`es video mosaic` already read, checked by `eval_run_output_is_what_es_video_mosaic_reads`
without a GPU or a physics backend. A cell is one **episode** of one suite, because
`Evaluation::run` gives every episode its own `Env` (`BatchDomains::single_env()`), which is
why 16 episodes are 16 directories and the 4x4 grid is a grid of episodes.

**1. `capture` guessed which input was an image, and the guess was wrong.** The rule was "any
plan input that is neither a `qpos` nor a `sensor` id is an image". That was invisible while
`Evaluation::run` passed `frames: None` — every such input was refused — and became a bug the
moment V3 supplied a frame source: the demo's `StateInput` names the robot **body**
(`ObsSource::JointState { body, dof: 6 }`, not a joint id and not a sensor id), so the first
real run served a 27,648-byte camera tile into a 6-element `F32` buffer and stopped with a
size mismatch. `input_sources` now resolves every input **once, before the first episode**,
against the documents: an input is an image because the Observation IR says `ImageInput`, and
a `JointState { body, dof }` channel reads the leading `dof` joint positions — the same
convention `joint_state::<NJ>` already uses to feed the Safety Plane. Anything else is an
error at the start of the run rather than a surprise in the middle of the table.
`docs/design/evaluation-execution.md` section 2.3 is the table.

**2. The lighting kinds could not be realised through `EnvRendererCfg`.** Section 2.7 expected
"once V0b lands a renderer, the two lighting kinds become implementable", and they are — but
not where the plan assumed. The `Rs` path shades from `RenderConfig::light_dir` and
`::ambient`, both fixed when a `Renderer` is built; `EnvRendererCfg` carries neither, and V0b
owns that surface. The demo scene's one emissive geom (`ceiling_light`) is no help either: it
sits at `z = 0.8` directly **above** the overhead camera at `z = 0.5`, so it is behind the
camera and out of frame, and on the `Rs` path an emissive geom lights nothing but its own
pixels. So the kernels landed as two pure functions on `es-eval`'s side —
`LightOverride::scene` (a gain on every geom's `rgba`, which is *exactly* a radiance gain
because the Lambert term is linear in `albedo`) and `LightOverride::rotate_dir` (a yaw on
`light_dir`) — and `es eval run` composes `es_render::Renderer` with V0b's own public
`render_config` / `body_poses` / `camera_view` to apply them, rebuilding the renderer only
when the draw changes. `es-render` and `es-env/src/render.rs` are untouched; the `es` crate's
`render` feature gained `es-render` as a direct dependency, which costs the default build
nothing (it is off by default, spec 4.2).

**The envelope was not tightened, because it did not need to be.** Section 8 says the demo's
clamps come from tightening the Deployment IR's limits. The measurement says they come for
free: an ACT chunk is not ramped the way V1's scripted expert is (section 7.5 finding 4), so
the plane clamps on its own, and the non-vacuity rule holds without touching
`deployment.toml`. That is the better outcome — the demo's envelope is the one the
demonstrations were recorded through — and it leaves INV-12 untouched either way.

**What was measured, on the oracle server (RTX 4090, `~/venvs/es/bin/python`, mujoco 3.13.0,
torch 2.14.0+cpu), 2026-09-14.** `evaluation_hash 5d70c21c…3b9ff7`, seeds 101-116 — held out
from the 1-50 the demonstrations were collected on. Success rate is the headline and the only
headline (§12.4):

| checkpoint | nominal success rate | mean episode length |
|---|---|---|
| 1,000 steps | 0.0625 (1/16) | 851.1 |
| 5,000 steps | 0.0000 (0/16) | 900.0 |
| 20,000 steps | 0.1250 (2/16) | 883.5 |

and the 20,000-step checkpoint across the suite, 16 episodes each:

| suite | success rate | mean episode length |
|---|---|---|
| nominal | 0.1250 | 883.5 |
| light_intensity | 0.0625 | 887.4 |
| light_direction | 0.1250 | 883.5 |
| observation_delay | 0.1250 | 883.5 |
| torque_noise | 0.0000 | 900.0 |
| backlash | 0.0000 | 900.0 |

96 episodes, 85,407 rendered frames, 28 minutes. The non-vacuity rule holds — 2 `Success`
episodes and 76,967 `Clamped` steps — but it holds for a reason worth stating plainly rather
than counting as a pass:

**Not one step in any run was `ActionSource::Policy`.** 76,967 `Clamped` and 8,440 `Fallback`
out of 85,407; `envelope_violation_rate` is exactly `1.0` in every cell of every suite. The
policy's raw chunk is outside the envelope on every single step, the rate watchdog trips, and
the fallback holds position about 10% of the time. A `light_direction` row equal to the
`nominal` row to four decimals is the same fact from the other side: what the arm does is
being decided by the Safety Plane, not by the pixels.

**The most likely cause is a train/inference observation mismatch, and it is V2's debt, not a
V3 finding about the policy.** `train_act.py` feeds the graph's state port the **raw**
`observation.state` row and re-implements exactly one Observation IR node, `Dequantize`
(section 7.6 finding 3). But the demo's Observation IR is not `ImageInput -> Dequantize ->
sink`: it is `StateInput -> Normalize{Range −1..1}` and `ImageInput -> Dequantize ->
Normalize{Range 0..1}`. The image branch is safe — `normalize_range(x, 0, 1)` is the identity
— and the state branch is not: at inference the policy receives `(q + 1) / 2`, and in training
it received `q`. An affine shift of the entire proprioceptive input, silently, exactly as
finding 3 predicted would happen "the moment a second node appears". Plan V does not fix it
here: the fix is a plan-bake step in training (or a retrain through the compiled plan), both
of which are V2's, and this packet is forbidden from retraining. It is **open question 11**.

So the honest reading of the table is: the pipeline runs end to end and produces a real §10.1
table, the Safety Plane demonstrably governs the arm, and the *policy* number is not yet a
measurement of ACT — it is a measurement of ACT fed an observation it was not trained on.

**Video.** `es video mosaic --grid 4x4` over the 16 nominal cells gives 900 frames of
384x392 (16 tiles of 96x96 plus an 8-pixel label strip), and `python/es/encode_video.py
--fps 50` gives an 18-second mp4 at the control rate: 10.5 MB `mp4v` for the 20,000-step run,
and 1.5 MB for an H.264 copy through the server's own `ffmpeg 7.0.2` over the same raw
frames. The mp4 is not in the hash chain (section 9); the frames are.

### 7.9 As built (V2b): the training set goes through the Observation IR

Section 7.8's table was not a measurement of ACT. Open question 11 named why, section 7.6's
finding 3 had predicted it a packet earlier, and V2b is the repair the finding already
specified: "an `es` step that bakes the observation plan over a dataset, which is a packet and
not a line."

**The defect, stated once more as a root cause.** The symptom was one missing `Normalize`. The
cause was that one Observation IR node had two implementations — `train_act.py`'s `dequantize`
was `Op::Dequantize` written a second time in Python — and once a node can have two
implementations, the *set* of nodes training applies is a second thing to keep in sync too.
That is how the state branch's `Normalize{Range −1..1}` went missing without anyone deciding to
drop it. Patching training to also normalize the state would have fixed the symptom and left
the mechanism; there is no version of that fix that survives the demo's Observation IR growing
a third node.

**`es dataset bake --policy <bundle.esb> --out <dir> [--frames <tiles>] <root>`.** It runs every
recorded frame through `es_eval::ObservationBake`, which is `capture` with a dataset row where
the physics state was: the same `CpuPlan::compile(obs, PlanMode::Release)`, the same
`input_sources` resolution, the same `f64 -> f32` input encoding, the same `plan.reset()` per
episode. Not "the same logic" — the same functions, called from two places
(`crates/es-eval/src/bake.rs`, `crates/es-eval/src/runner.rs`). It writes one safetensors per
episode carrying `[frames, ...]`-shaped tensors under the Observation IR's **own output names**
plus `action`, and a `manifest.json` with `observation_hash`, `task_hash`, `compiler_hash` and
the dataset's content hash. `train_act.py` reads that and lost `read_dataset`, `read_frames`,
`dequantize` and `plan_inputs` — 321 lines became 269, and the count that matters is that the
number of Observation IR node implementations in this repository went from two to one.

**Three things the plan above did not know.**

**1. `input_sources` had to learn that there may be no model.** At inference an input resolves
against `ModelInfo` first — a joint's `qpos` range, a sensor's range — and only then against the
Task IR's `ObservationSpec`. A recorded dataset has no loaded model: it carries
`observation.state` and tiles. Rather than write a second resolver, `input_sources` takes
`Option<&ModelInfo>`, and with `None` the two model-indexed arms are simply unavailable, so an
observation the bake cannot feed is *one error before the first frame* naming the port — the
same discipline V3 gave the runner (finding 1 of section 7.8), for the same reason. The demo's
`StateInput` names a body and resolves through `ObsSource::JointState { body, dof }` either way,
which is why it bakes at all.

**2. The oracle can be bit identity, and it needs no Python, no GPU and no physics backend.**
`a_baked_frame_is_bit_identical_to_what_capture_serves` runs a real evaluation over a
demo-shaped Observation IR (`StateInput -> Normalize{−1..1}` **and**
`ImageInput -> Dequantize -> Normalize{0..1}`) with a frame source that records the `StateView`
it was handed alongside the tile it served, and a `PolicyRuntime` that records every observation
map it received. The same rows and tiles then go through `ObservationBake`, and every output
tensor of every frame must be byte-equal — 50 frames x 2 ports in the fixture. There is no
tolerance, because a tolerance would mean one of the two paths had grown a conversion of its
own. It also asserts that both nodes *fire* (`(q + 1) / 2` on the state, a changed tile on the
image), so it cannot pass by comparing two copies of the raw input, which is exactly how the old
arrangement would have passed.

**3. `pretrained` is rejected, and open question 6 closes with it.** V2 found the lowering
ignored `VisionEncoder{pretrained}`. Honouring it is one line — `weights="DEFAULT"` in
`_backbone` — and it is the wrong line: `EsPolicy()` would fetch ImageNet weights over the
network at *every* construction, including inside `TorchRuntime::load` at inference, where
`load_state_dict(strict=True)` overwrites all of them a moment later. A lowering that needs the
network to instantiate contradicts §2.5, and weights nobody hashed are outside the chain (§5.3).
So `lower_to_torch` returns `LowerError::Unsupported` naming the flag and both ways out, and the
demo's `learning.toml` now says `pretrained = false` — which is what V2 was doing anyway. It
moves `learning_hash` and `policy_hash`; `task_hash` and `observation_hash` do not move, so the
baked set and `evaluation.toml` are untouched. `es_ir::learning::testing::act_like` keeps
declaring `true`, because that is what ACT is; the two `es-policy` tests that lower it clear the
flag through one shared helper.

**What was measured, on the oracle server (RTX 4090, `~/venvs/es-lerobot-cuda`, torch 2.11.0+cu129
for training, `~/venvs/es/bin/python` mujoco 3.13.0 for evaluation), 2026-09-15.** Every knob is
V2's: the same 50-episode dataset and tiles, `--batch 8 --lr 1e-4 --seed 0`, checkpoints at
1k/5k/20k. One variable moved, and it is the observation. The bake is 17,697 frames, 1.9 GB of
safetensors, `observation_hash f4a50730…55f6e0` — the same hash `evaluation.toml` declares, so the
baked set, the bundle and the evaluation document all name one Observation IR. The lowering is
byte-identical to V2's (`lowering_hash 956abb67…ec6d`), which is the cleanest possible statement
that the architecture did not move.

Training loss, L1 over the action chunk, mean of the 100 steps ending at each mark, beside V2's:

| step | 1 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|
| V2 (raw state) | 0.7267 | 0.1825 | 0.0507 | 0.0309 | 0.0171 |
| V2b (baked) | 0.7261 | 0.2085 | 0.0528 | 0.0325 | 0.0166 |

They are the same curve, and that is the expected result: the state fix is an affine map of one
input, which an optimizer absorbs. **The loss could never have detected this defect**, which is
why the oracle for it is bit identity and not a threshold.

Success rate, 16 held-out episodes on seeds 101-116, beside section 7.8's:

| checkpoint | V3 nominal | V2b nominal | V2b mean episode length |
|---|---|---|---|
| 1,000 steps | 0.0625 (1/16) | 0.1250 (2/16) | 861.8 |
| 5,000 steps | 0.0000 (0/16) | 0.0625 (1/16) | 859.3 |
| 20,000 steps | 0.1250 (2/16) | 0.0000 (0/16) | 900.0 |

Three successes out of 48 episodes either way. The sweep did not get better and it did not get
worse; it moved around, which is what a table decided by something other than the policy looks
like.

and the 20,000-step checkpoint across the suite, 16 episodes each:

| suite | V3 | V2b |
|---|---|---|
| nominal | 0.1250 | 0.0000 |
| light_intensity | 0.0625 | 0.0000 |
| light_direction | 0.1250 | 0.0000 |
| observation_delay | 0.1250 | 0.0000 |
| torque_noise | 0.0000 | 0.0000 |
| backlash | 0.0000 | 0.0000 |

Every V2b cell has `episode_length 900.00` and `envelope_violation_rate 1.0000`.

**`evaluation.toml` asks for `success_rate >= 0.5`. The best cell measured `0.1250` and the
20,000-step suite measured `0.0000`. It fails, and nothing was lowered to make it not fail.** V3's
own non-vacuity gate fails on the 20,000-step bundle too: `visible_learning_demo_run` panics with
*"non-vacuity: no suite produced a single Success episode"*, which is the gate doing its job.
86,400 frames, 77,880 `Clamped`, 8,520 `Fallback` — and, exactly as in V3, **not one step in any
run was `ActionSource::Policy`**, in the nominal sweep either (1k: 12,428 / 1,360 / 0; 5k: 12,388 /
1,360 / 0; 20k: 12,980 / 1,420 / 0).

**So the observation was not what was stopping the arm, and V2b's real result is the next defect,
measured rather than guessed.** Three numbers say what is:

1. **The policy imitates well.** On a baked training frame the chunk's first action tracks the
   recorded action to within 0.02-0.16 rad per joint (t = 312 of episode 0: policy
   `+0.801 −0.727 +0.718 +1.551 −0.008 −0.076`, recorded
   `+0.773 −0.567 +0.661 +1.464 +0.000 −0.050`). Fed the observation it was trained on, ACT
   reproduces the demonstration. That claim could not be made before this packet.
2. **The violation is `Velocity`, on 90% of steps.** Decoding the `ViolationKind` bitsets in
   `events.json`: `Velocity` 77,880 (0.901), `Acceleration` 76,868 (0.890), `RateLimit` 72,467
   (0.839), `Position` 16,093 (0.186), `ViolationRate` 8,520 (0.099). The last one is the
   watchdog that produces every `Fallback`.
3. **The demonstrations' commands lead their own joint positions by four times the envelope's
   per-step budget.** `SafetyPlane` clamps velocity as `(cmd − last_safe) / dt` against
   `velocity_max = 3.0`, so a command may move `3.0 × 0.02 = 0.0600` rad per control step — and
   the recorded `action` column's own step-to-step delta maxes at exactly `0.0600`, because
   `es loop collect` records the **post-plane** command and the scripted expert saturates it.
   Meanwhile `|action[t] − qpos[t]|`, the position servo's steady-state tracking error, is a
   median **0.2413** rad and reaches 0.7831. ACT learns to emit `qpos + 0.24`; the plane will let
   the command advance 0.06 per step; the gap is structural and never closes, so the arm is
   velocity-clamped from the first step of every episode and the violation-rate watchdog latches
   into `hold_position` about a tenth of the time.

None of that is an Observation IR problem and none of it is fixable inside this packet:
`es-safety`, `es-env` and the demo's `deployment.toml` are all in V2b's `forbidden` list, and
INV-12 forbids the one shortcut. It is the next packet, and it is V1/V3-shaped — the
demonstrations' action convention and the Deployment IR's envelope were never checked against each
other. **Open question 12** below asks which end moves.

The honest reading, then: section 7.8's table was a measurement of ACT fed an observation it was
not trained on; this one is a measurement of the Safety Plane refusing a policy that faithfully
imitates demonstrations recorded through a tighter interpretation of the same envelope. The demo
still does not work, but it now fails for a reason with a number attached to it.

**Video.** `es video mosaic --grid 4x4` over the 16 nominal cells and `python/es/encode_video.py
--fps 50`, the same pipeline V3 used, over the retrained checkpoints: 900 frames of 384x392 each,
`demo-{1000,5000,20000}.mp4` at 13.0 / 11.6 / 11.4 MB `mp4v` and 1.9 / 1.7 / 1.7 MB for the H.264
copies through the server's `ffmpeg 7.0.2`. The mp4 is not in the hash chain (section 9); the
frames are.

### 7.10 As built (V1c): what is executed is what is recorded, and where the arm actually stops

Section 7.9 ended on open question 12: the demonstrations' action convention and the Deployment
IR's envelope were never checked against each other, and three ends could move. V1c checked them,
and the first thing it found is that the question's premise was off by one packet.

**1. `action` was already the executed action.** `DomainRunner::emit_actions` copies
`SafeAction::q` into `ctrl`, `Env::step` records `ctrl`, `to_lerobot` writes it as `action`. Open
question 12's option (a) — "record `action` as the next commanded position rather than the servo
target" — describes a change that was already in the code. Nobody could see it, because nothing in
the repository asserted it and the pre-plane command was thrown away, so a `Clamped` frame could
not be read without re-running the plane. V1c pins it (`ctrl` is `SafetyPlane::last_safe_action()`
bit for bit, `f64::to_bits`, no tolerance) and keeps the command in a second column,
`action_commanded`. That moves `dataset_schema_hash`; `task_hash`, `observation_hash`,
`learning_hash` and `lowering_hash` do not move, and `lerobot 0.6.1` reads the extra feature
through V1b's v3.0 export (`RAN lerobot_v3_export`, with `"action_commanded": {"dtype":
"float32"}` in the features it reports back).

**2. "`es loop collect --episodes N` only solves episode 0" was the inference phase, not the
plane and not the expert's `reset`.** Section 7.6's finding 5 named three suspects and all three
were wrong. `es loop collect --expert` resets `ScriptedExpert` from the intervener hook on
`frame == 0`; that hook is `PolicyRuntime::infer`, which runs when a submitted observation is
*released* — `expected_latency_ms` ticks after the submit (section 12.3 of the spec). The demo's
`learning.toml` declares `15.0` ms against a 20 ms control period, so the first call of every
episode is frame 1 and **`frame == 0` never fires at all**. It looked correct only because a
freshly constructed `ScriptedExpert` starts reset: episode 0 needs no reset and gets away with it,
and every episode after it runs with the previous one's stage and latched cube. Keying on the
episode index takes `--episodes 50 --seed 1` from **1/50 to 50/50**, and the held-out five from
0/5 to 5/5 — in one command each. `crates/es-data`'s fixture declared `expected_latency_ms = 0.0`,
which is exactly why V1's own oracle could not have seen it; the regression test now states the
property (`frame_zero_is_not_a_hook_an_intervener_may_reset_on`: with a declared latency, **no**
call of the intervener has frame 0).

A second, smaller instance of the same class: `Collector::run` reset the env, the chunk buffers
and the e-stop latch between episodes but never the plane's hold target, velocity or rate history,
so the plane opened each episode believing the arm was still where the last command left it. It is
now seeded from the measured pose on the first frame of every episode — which is what
`SafetyPlane`'s own documentation says a caller that knows the real pose does before the first
`validate` — and `a_second_episode_repeats_the_first_exactly` fails without it.

**3. The collect path and the evaluation path disagree about what the envelope is measured
against, and that disagreement is not closable from the collector.** `es_eval::runner`,
`es_ros2::hil` and `es_runtime_embedded` call `observe_state` before **every** `validate`, which
re-seeds `last_safe`, `prev_safe`, `vel` and `prev_vel`; the envelope is then a bound on the
**following error**. With the demo's numbers the binding stage is acceleration, and the bound is
`a − q − q̇·dt ≤ acceleration_max · dt² = 0.008` rad. `Collector::run` seeds once per episode, so
inside an episode the envelope bounds *commanded* motion — which is what `ScriptedExpert::chunk`
paces itself to, and its doc comment says so in as many words.

Making the collector re-seed every step was implemented and measured, because that is the change
that would make the two paths agree. It drops the scripted expert to **2/8 on
`expert_solves_the_pinned_seeds`** (threshold `0.875`, a golden) and to **0/16 on the evaluation's
own seeds 101–116**, and a 50-episode collect to 18/50. The reason is structural: the expert plans
a 16-row chunk executed over 10 ticks from the pose at chunk start, and no re-pacing of
`step_max` / `accel_max` / the anti-windup lead satisfies a 0.008 rad following-error bound across
ten open-loop ticks. The two knobs that would are `deployment.toml`'s envelope and
`execute_chunk`, and both are outside V1c. So the asymmetry stays, written down with numbers
instead of guessed at, and open question 12 stays open — with option (a) struck off as already
done and option (c) answered below.

**The diagnostic that decides how to read everything else: open question 12's option (c), run.**
V2b's own 20,000-step checkpoint was re-packed into a bundle whose deployment document is a
*scratch* copy — `velocity_max` 3.0 → 30.0, `acceleration_max` 20.0 → 400.0, `ee_velocity_max`
0.6 → 6.0, both action-rate limits → 1.0, everything else including the watchdogs untouched — and
run over the same 16 nominal episodes. The committed fixture was not edited and nothing was
disabled (INV-12); `weights_hash`, `lowering_hash`, `task_hash` and `observation_hash` all match
V2b's, so only the envelope moved.

| | V2b, committed envelope | the same bundle, widened |
|---|---|---|
| `success_rate` (nominal, 16) | 0.0000 | **0.0000** |
| `episode_length` | 900.00 | 900.00 (16/16 timeout) |
| `envelope_violation_rate` | 1.0000 | **0.0551** |
| `ActionSource::Policy` | 0 / 86,400 | **13,606 / 14,400** |
| `ActionSource::Clamped` | 77,880 | 794 |
| `ActionSource::Fallback` | 8,520 | **0** |

**So the Safety Plane was not what was stopping the arm.** Given an envelope it does not fight,
ACT drives 94.5 % of the steps itself, the watchdog never latches, and the success rate is
unchanged at zero with every episode running out its full 900-step budget. Widening the envelope
buys a policy that moves and still cannot do the task. That is worth knowing before anyone spends
a packet on option (b), and it is why V1c's own retraining below is reported as a measurement of
the demonstrations' convention and not as an attempt to pass `success_rate >= 0.5`.

**What was measured, on the oracle server, 2026-09-15.** Every training knob is V2's and V2b's:
`--batch 8 --lr 1e-4 --seed 0`, checkpoints at 1k/5k/20k, the same 50 training episodes and the
same 5 held out. What moved is the collection: **one** `es loop collect --episodes 50 --seed 1`
command instead of fifty single-episode runs merged by `es loop distill`, with the expert reset
per episode and the plane seeded per episode. 50/50 `Success`, 5/5 on the held-out set, 18,263
frames (against V2's 17,697), baked to 1.9 GB with `observation_hash f4a50730…55f6e0` and
`lowering_hash 956abb67…ec6d` — both byte-identical to V2's and V2b's, so neither the observation
nor the architecture moved.

The demonstrations themselves changed, and this is the packet's own result:

| | V2b's set | V1c's set |
|---|---|---|
| `\|action − qpos\|`, median over all frames | 0.2413 | **0.0027** |
| the same, arm joints 0–4 only | — | 0.0009 |
| the same, gripper (joint 5) | — | 0.1436 |
| frames where `action ≠ action_commanded` | not recorded | 11,580 / 18,263 |
| `action_source` | — | 11,581 clamped, 6,631 human, 50 fallback, 1 policy |

A hundredfold drop in the lead, from seeding the plane with the arm's real pose at the start of
each episode rather than letting `last_safe` start at zero and drift. The 50 fallbacks are one
per episode — the chunk underrun on frame 0, before the first inference result exists — and the
violation-rate watchdog never fires once, against V2b's ~10 % of steps.

Training loss, L1 over the action chunk, mean of the 100 steps ending at each mark:

| step | 1 | 100 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|---|
| V2 (raw state) | 0.7267 | 0.1825 | 0.0507 | 0.0309 | 0.0171 |
| V2b (baked) | 0.7261 | 0.2085 | 0.0528 | 0.0325 | 0.0166 |
| V1c (baked, re-collected) | 0.7476 | 0.2053 | 0.0560 | 0.0319 | **0.0176** |

The same curve for the third time. Success rate, 16 held-out episodes on seeds 101–116:

| checkpoint | V3 nominal | V2b nominal | V1c nominal | V1c mean episode length |
|---|---|---|---|---|
| 1,000 steps | 0.0625 (1/16) | 0.1250 (2/16) | 0.0000 (0/16) | 900.0 |
| 5,000 steps | 0.0000 (0/16) | 0.0625 (1/16) | 0.0000 (0/16) | 900.0 |
| 20,000 steps | 0.1250 (2/16) | 0.0000 (0/16) | **0.0625 (1/16)** | 876.7 |

and the 20,000-step checkpoint across the suite, 16 episodes each:

| suite | V3 | V2b | V1c |
|---|---|---|---|
| nominal | 0.1250 | 0.0000 | 0.0625 |
| light_intensity | 0.0625 | 0.0000 | 0.1250 |
| light_direction | 0.1250 | 0.0000 | 0.0000 |
| observation_delay | 0.1250 | 0.0000 | 0.0000 |
| torque_noise | 0.0000 | 0.0000 | 0.0000 |
| backlash | 0.0000 | 0.0000 | 0.1250 |

Four `Success` episodes in 96, against V2b's zero and V3's two — so V3's own non-vacuity gate,
which panicked on the V2b bundle with *"non-vacuity: no suite produced a single Success
episode"*, now passes: `RAN visible_learning_demo_run`, 1,544 s. **`evaluation.toml` asks for
`success_rate >= 0.5`. The nominal suite measured `0.0625`. It fails, and nothing was lowered to
make it not fail.** Every V1c cell still reports `envelope_violation_rate 1.0000`, and across the
six suites 84,772 frames are 76,412 `Clamped` and 8,360 `Fallback` with **not one
`ActionSource::Policy`** — because the evaluation path's reading of the envelope bounds the
following error at 0.008 rad, and no imitation policy at an L1 of 0.0176 is that precise. Cleaning
up the demonstrations did not change that, and could not have: it is a property of the two
readings, not of the data.

Run through the same widened scratch envelope as the diagnostic above, V1c's 20,000-step
checkpoint reports `envelope_violation_rate 0.4127`, 8,457 `Policy` and 5,943 `Clamped` steps of
14,400 (almost all `violation.position` — the policy commands poses past the joints' soft
limits), no fallback at all, and `success_rate 0.0000`. Both policies, freed of the plane, move
and neither does the task.

**So the honest reading of V1c is: the demonstrations were fixed, the collection loop was fixed,
and the demo still does not work — and for the first time the reason is not downstream of either.
The bottleneck is the policy.** Three packets have now each removed one confound (V2b the
observation, V1c the demonstrations and the episode loop, the two diagnostics the envelope), and
what is left is 50 demonstrations, 20,000 optimizer steps, and an ACT that imitates a training
frame to 0.0176 L1 and generalizes to a held-out cube pose about one time in sixteen.

**Video.** `es video mosaic --grid 4x4` over the 16 nominal cells and `python/es/encode_video.py
--fps 50`, the same pipeline V3 and V2b used: 900 frames of 384x392 each,
`demo-{1000,5000,20000}.mp4` at 12.9 / 12.7 / 11.4 MB `mp4v` and 2.0 / 1.9 / 1.7 MB for the H.264
copies through the server's `ffmpeg 7.0.2`. The mp4 is not in the hash chain (section 9); the
frames are.

### 7.11 As built (V5 phase 1): the fast cycle, and the split that was not available

Packet `docs/packets/M5/V5-fast-cycle.md`. One collect -> train -> evaluate cycle is measured at
roughly an hour, sequential: ~5 min for the nominal 16 episodes, ~28 min for the 6-suite
96-episode run, ~11 min for 20,000 training steps at batch 8. V5 cuts it **without changing a
number**: nothing here is allowed to move a success rate, a loss or a frame.

**1. The episode-level split does not exist, and saying so is the finding.** The obvious design —
one worker per episode — is unavailable, and not for a reason that more code fixes. Inside one
**cell** (one suite, all of its episodes) `Evaluation::run_with_frames` keeps one `Env`, one
`SafetyPlane` and one monotonic chunk `seq` for the whole cell, and `Env::reset` bumps a per-env
episode counter (`crates/es-env/src/env.rs:224`) that keys the task's own `RandomizationPlan`.
Episode 5's initial state is therefore not reproducible without having run episodes 0..4; the safety
counters behind `envelope_violation_rate` and `chunk_underrun_rate` are cell-level sums; and
`env.metrics()` accumulates over the cell. A per-episode worker would have to either re-run the
episodes before its own (no speedup) or produce different numbers (not allowed). Seeking the
episode counter is an `es-env` change, which this packet forbids itself.

A **cell**, by contrast, is genuinely self-contained: its own `Env`, its own `SafetyPlane`, `seq`
from 0, `plan.reset()` on every episode including its first. The two things cells share are the
compiled `CpuPlan` — reset per episode, so a freshly compiled plan and a reset one are the same
plan — and the `PolicyRuntime`, which is feed-forward (`TorchRuntime::infer` keeps no state between
calls; the temporal window lives in the plan). **So `--jobs N` partitions the cells,
round-robin: cell `c` belongs to shard `c % N`.** The demo's six suites are where its 28 minutes
are; the nominal run is one suite and gets nothing, and `N` is clamped to the suite count.

**2. The sequential path is the sharded path with one shard.** `Evaluation::run_shard` runs the
cells a shard owns and judges nothing; `Evaluation::merge` sorts every worker's cells by their
index into `EvaluationIr::suites` (a stable sort, so the metric order inside a cell is untouched)
and computes `judge`, the hash chain, `report.json` and `evaluation.lock` — once, in the parent.
`run_with_frames` is now literally `run_shard((0, 1))` followed by `merge`, so byte-identity
between `--jobs 1` and the old path is by construction rather than by a second implementation
(spec 3.5 tier 1). The workers are processes, not threads, because the physics backend is a Python
subprocess with one env in it and `TorchRuntime` holds another; threads would queue behind the same
two interpreters.

Frames need no merge step at all: a cell writes into `<frames>/<suite>-<NN>/`, cell names are
globally unique and shards own disjoint cells, so the children write into one directory without
colliding. Each hands back its slice of `events.json` and the parent concatenates the (disjoint)
maps.

**3. A lost worker is an error, never a short report.** A merge whose cells are not exactly
`0..suites.len()`, each once, is `EvalError::Shard` naming what it got. Without that check a run
that lost one child would produce a report over five suites carrying a perfectly correct
`evaluation_hash` — the worst available failure mode, because it is indistinguishable from a real
measurement. At the CLI a failed child is one named error carrying the shard, its exit code and its
last line of stderr.

**4. Two of the nine metrics are already not reproducible, and `--jobs` makes that visible.**
`physics_steps_per_sec` and `actions_per_sec` are `EnvMetrics::simulation_wall` divided into a
counter, so a sharded run measures each worker's own clock. They move run-to-run in the sequential
path too; the demo's `evaluation.toml` declares neither and neither does the oracle fixture. This
is recorded rather than fixed — the fix is `es-env`'s, and no number in this note depends on it.

**5. The training flags, and which of them are honest to quote.** `--resident-gpu` moves the whole
baked set onto the device once instead of copying one sample per forward. It changes *where* the
tensors live and nothing else, so the loss curve — and the checkpoint — are **bit-identical** to
the default path at the same seed, which `resident_gpu_does_not_move_the_loss` pins by comparing
two `--loss-curve` files as bytes. `--amp bf16` and `--compile` are opt-in **because** they change
the bits: measured locally on the fixture, 40 steps at batch 4 give `initial_loss`
0.5596105996519327 at fp32 and 0.5595557652413845 under bf16 autocast. `--batch` stays at 8 so the
numbers above stay reproducible; the documented convention for raising it is linear `--lr` scaling
(`--batch 32 --lr 4e-4`).

**6. Measured (phase 2), on the oracle server, 2026-09-15.** `nvidia-smi` read 0% utilization / 55
MiB used before every run below; the 16-core box was otherwise idle except where the run itself is
the reason it was not (next point). V1c's `trained-20000.esb`, `build/` and `baked/` (section 7.10)
are the fixture for every row; nothing here moved a number section 7.9 or 7.10 recorded.

| what | Target | Observed |
| --- | --- | --- |
| 6-suite, 96 episodes, sequential (`--jobs 1`) | ~28 min (V3 baseline) | 25:56 (25:52 on an earlier, since-overwritten run of the same config -- consistent) |
| 6-suite, 96 episodes, `--jobs 6`, before the fix below | ~5-6 min | not run to completion: ~5 frames/s combined against sequential's ~55, 35/96 cells at 85 min, killed rather than waited out (projected > 4 h) |
| 6-suite, 96 episodes, `--jobs 6`, after the fix below | ~5-6 min | **5:49** |
| nominal-only, 16 episodes, sequential, two independent runs | ~5 min | 4:16.49 / 4:16.68, byte-identical to each other (`report.json` and every frame) |
| `report.json` / `events.json`, `--jobs 1` vs `--jobs 6` (fixed) | byte-identical | not bit-identical on the real backend -- see below; the `FakeBackend` oracle stays byte-identical |

The nominal-only config is one suite; `crates/es/src/cmd/eval.rs` clamps
`a.jobs.min(eval_ir.suites.len().max(1) as u32)`, so `--jobs 6` on it runs as `--jobs 1` by
construction -- confirmed by reading the clamp, not by a separate timed run.

**The first `--jobs 6` run was 5x *slower* than sequential.** Each shard is this same binary
re-invoked (`spawn_shards`), and each shard's own `TorchRuntime` subprocess defaults its thread
pool to every core on the box; six of them on 16 cores want on the order of 90 OS threads at once
(`nlwp` 23 per subprocess measured directly), load average sustained ~47, and the box spent its
time context-switching rather than computing -- 35 of 96 cells in 85 minutes, projected past four
hours to finish. Fixed in `crates/es/src/cmd/eval.rs`: `spawn_shards` now sets each shard's
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `TORCH_NUM_THREADS` to
`cores / jobs` (`shard_thread_cap`), skipping any the caller already exported
(`shard_thread_env` -- passthrough wins), both unit-tested without spawning a real subprocess.
After the fix, six shards measured 2-3 threads apiece (`nlwp` 3, ~136% CPU each), load average
fell to ~5.5, and the run finished in 5:49 -- in the packet's targeted range, and confirmed by
`nvidia-smi`/`uptime` before and after that nothing else on the box changed in between.

**That fix costs the real backend's byte-identity, and the packet's forbidden list will not let
it be closed here.** `sharding_the_cells_produces_a_byte_identical_report` (`FakeBackend`, no real
floating point) passes, byte for byte, re-verified after the fix. On the real `mujoco-cpu` +
`torch` stack, though, `--jobs 1`'s own subprocess is left uncapped (unchanged by this packet)
while a `--jobs 6` shard is now capped to `cores/jobs` threads -- a different thread count than
`--jobs 1` uses, and CPU-threaded reductions are not exactly associative. Measured: 6 of the
merged report's 24 cells differ --

| | `--jobs 1` | `--jobs 6` (fixed) |
| --- | --- | --- |
| `light_intensity` `episode_length` | 845.3125 | 845.25 |
| `nominal` `failure_mode_histogram` `violation.position` | 2,592 | 2,430 |
| `light_intensity` `failure_mode_histogram` `violation.position` | 1,677 | 1,936 |

-- and `success_rate` and `envelope_violation_rate` are identical in every one of the 24 cells; a
handful of individual frames differ later in an episode once a step's violation classification
flips. Isolated before blaming the fix: two independent `--jobs 1` runs of the nominal-only config
are byte-identical to each other (table above, frames included -- ruling out inherent nondeterminism
run to run), and exporting `OMP_NUM_THREADS=16` (this box's core count) explicitly before a
`--jobs 1` run reproduces the unset-default run byte for byte (ruling out "explicit vs. default"
as the variable) -- so the divergence tracks thread *count*, not process separation. This is
downstream of `MuJoCoCpuBackend`'s own declared `DeterminismTier::PhysicsMeaning` (section 9: tier
3, not bitwise) and of CPU-threaded kernels generally, not a defect in the merge/shard logic.
Closing it would mean capping `--jobs 1`'s own thread count too, at a value this packet has no
basis to choose without risking the already-committed V1c/V2b/V3 numbers -- forbidden. Left as a
documented, pre-existing limitation of the real backend.

**Training, V2's knobs, all five rows, 20,000 steps at `--checkpoint-at 20000`:**

| flags | wall-clock | `initial_loss` | `final_loss` |
| --- | --- | --- | --- |
| (a) default | 10:53.07 | 0.066787 | 0.017945 |
| (b) `--resident-gpu` | 10:57.00 | 0.066568 | 0.017749 |
| (c) `--resident-gpu --amp bf16` | 12:53.49 | 0.066955 | 0.018015 |
| (d) `--resident-gpu --compile` | 10:00.07 | 0.066878 | 0.017895 |
| (e) `--resident-gpu --batch 64 --lr 8e-4` | 1:24:31 | 0.050043 | **NaN** |

(a) and (b) are **not** bit-identical at `--device cuda`: `cmp` disagrees on both `--loss-curve`
and the checkpoint, diverging from the second optimizer step (the first matches: 0.4806089...
both; the second is 0.48060897 vs 0.48060090, a relative difference around 1e-5, growing from
there). This is ordinary CUDA kernel/algorithm-selection non-determinism -- `train_act.py` sets no
`torch.use_deterministic_algorithms`, and a resident tensor's different memory layout can select a
different cuDNN/cuBLAS kernel than a freshly-copied one. The design note's and packet's
bit-identical claim is accurate for the device the unit oracle actually exercises (CPU, point 7
below) and does not hold on CUDA; both are recorded, not reconciled, since reconciling it is
outside this packet (no code here can add determinism-forcing to `train_act.py` without moving
numbers that are also outside this packet's forbidden list).

(c) is **slower** than (a)/(b), not faster: at `--batch 8` the module runs one sample at a time
(section 7.11 point 5), so each forward is tiny and launch-overhead bound rather than compute
bound, and bf16 autocast's per-op overhead does not pay for itself there. (d) is modestly faster
(~8% over (a)/(b)) -- `torch.compile`'s fusion has something to work with even at this scale, once
its warmup is amortized over 160,000 forward/backward calls. (e) followed the docstring's own
`--batch N` / `--lr` linear-scaling convention (`--batch 64`, 8x, with `--lr 8e-4`, 8x) for the
same 20,000 steps, and diverged to `NaN`; the convention does not hold for this model and this
50-episode dataset at this multiplier, which is a finding about the convention and not a claim
that (e) is equivalent to (a)-(d) (§12.4: nothing here quotes (e) as a `step/s` figure or a
recommendation).

**7. `cargo test -p es-policy --test ir_training -- --ignored --nocapture`, `ES_PYTHON` pointed at
the CUDA training venv, in the server tree:**

```
RAN act_training_uses_baked_observations: loss 0.4456 -> 0.1828 over 40 steps (0.410x), chunk
  [10, 6], max_abs vs a direct forward 0e0 (tol 1e-5), observation_hash
  f4a50730ac95b91734c9678e75d9e6bc1845578bc2985e45b409e80f3355f6e0, torch 2.11.0+cu129
RAN resident_gpu_does_not_move_the_loss: 40 bit-identical steps, 825 bytes of curve
```

`resident_gpu_does_not_move_the_loss` needed one more fix first: its last assertion checked for
the literal substring `"resident_gpu":true`, which Python's default `json.dumps` never emits (it
always spaces the colon: `"resident_gpu": true`) -- a pre-existing bug in the test itself, caught
only now because running it for real needs `torch`, and CI skips it with a printed reason when the
package is missing. Fixed in `crates/es-policy/tests/ir_training.rs`; the test runs at
`--device cpu` (the default `train_act.py` falls back to, unchanged by this packet), which is
exactly why its bit-identical claim does not contradict the CUDA divergence measured above --
they are different devices.

One local limitation worth recording so it is not rediscovered: `--compile` cannot be exercised on
the Windows development box — `torch._inductor` reads its templates with the ANSI codepage and dies
on a `UnicodeDecodeError` under a `cp949` locale. That is torch's bug, not this script's; row (d)
above is its measurement on the Linux server.

### 7.12 As built (V6): one envelope semantics, and the harness that now passes the expert

Packet `docs/packets/M5/V6-envelope-semantics.md`. Section 7.10's finding 3 said the collect path
and the evaluation path disagree about what the Safety Plane's dynamic stages are measured
against, and left it open because V1c was forbidden from `es-safety`. The disagreement is not a
preference. It has an answer in the spec, and the answer decides whether *any* evaluation number
the demo has produced means anything.

**1. The defect, stated exactly.** `Collector::run` called `observe_state` on the first frame of
each episode; `es_eval::runner`, `es_ros2::hil` and `es_runtime_embedded` called it before **every**
`validate`. `observe_state` overwrote `last_safe`, `prev_safe`, `vel` and `prev_vel` — the four
fields stages 3, 4 and 7 of the clamp are differences of. So on the evaluation path
`velocity_limit`, `acceleration_limit` and `rate_limit` were differences between the *command* and
the *measured joint*: the position servo's following error. With the demo's numbers the binding
stage is acceleration and the bound is `a − q − q̇·dt ≤ acceleration_max · dt² = 0.008` rad. The
scripted expert — which paces itself to exactly what the Deployment IR allows, and passes 50/50
through `es loop collect` — scores **0 of 16** through `es eval run`. Every V3, V2b and V1c
evaluation cell was measured against a ceiling of zero, which is why every one of them reports
`envelope_violation_rate 1.0000` and not one step of `ActionSource::Policy`.

**2. The semantics, from the spec, not from convenience.** Spec §9.3's table is titled *constraints
applied to the policy output* (정책 출력에 적용되는 정적·동적 제약) and every dynamic row of it says
**clamp** — `velocity_limit` "관절·EE 속도 상한 / 클램프 + 카운터", `rate_limit` "액션 1차·2차 미분
상한 / 필터링". A measured velocity is not clampable; only the value the plane is about to emit is.
So the quantity each of those rows bounds is a difference of the plane's own commands, and the
measured state enters exactly once, as the seed that puts the first command of an episode where
the arm actually is.

§9.5 says the same thing from the other end, and more strongly: **"`deployment_hash`가 같으면 안전
동작이 같다"** — the same deployment hash must give the same safe action in simulation and on the
robot, and it calls that the core claim of the §27.1 evidence bundle. A clamp computed from
feedback cannot satisfy it. In MuJoCo `qvel` is exact and noiseless; on an SO-101 the same number
comes back over a serial bus at a lower rate with quantization and lag. An envelope that reads it
would emit different actions on the two, for the same `deployment_hash`, and §3.4's determinism
rules would have nothing to grip. The orchestrator's standing default — bound the *measured*
velocity and brake — is therefore **not** taken: the spec is not silent, and §9.5 rules it out.
There is no jump check between command and measurement either, because §9.3's table has no such
row; `position_limit` and `workspace` are the two rows that bound *where* a command may go, and
they are unchanged.

The physical reading is the same one. The SO-101's servos are `STS3215`s driven as MuJoCo
`position` actuators with `kp = 998.22`, `kv = 2.731`, `forcerange = ±2.94 N·m`. The loop is closed
inside the servo: the host sends a goal and the servo makes torque out of the error.
`2.94 / 998.22 = 0.0029` rad is where that torque saturates — **three milliradians of following
error already means full torque**. The pre-V6 evaluation bound of `0.008` rad was therefore a bound
on the servo's error signal at roughly 2.7× its saturation point: it was bounding torque, badly,
in a row that says velocity, while `torque_limit` sits two rows above it and the model's own
`forcerange` already enforces it. `tests/fixtures/visible-learning/deployment.toml` now carries
that derivation for every number it declares, including the three it declares and `es-safety` does
not enforce (`ee_velocity_max`, `contact_force_max`, the distance minima — no FK, no contact query
in the plane).

**3. What changed, and it is four lines.** `SafetyPlane::observe_state` seeds the command chain on
the first call after `SafetyPlane::begin_episode` (or after construction) and returns without
touching anything on every call after that. `begin_episode` clears the e-stop latch *and* re-arms
the seed; it replaces the bare `reset_latch()` at both episode boundaries. Every consumer may now
call `observe_state` before every `validate` — and every consumer does — with the plane, not the
caller, deciding what that means. `validate` keeps its signature (INV-13), no envelope number
moved, nothing is disabled or bypassed (INV-12), and no new trait appeared (INV-17). The collector
loses its `if frame == 0`; the evaluation runner loses its `reset_latch`.

**4. What the numbers say, locally.** `crates/es-safety/tests/envelope_reference.rs` builds a plane
from the demo's committed `deployment.toml`, drives it with the expert's own pacing rule
(`0.9 · min(velocity_max·dt, first_diff_max)` per step, `0.9 · min(acceleration_max·dt²,
second_diff_max)` of change per step) against a plant that trails by up to **0.1774 rad**, twenty-two
times the old bound, and asserts sixty consecutive `ActionSource::Policy` steps whose output equals
the command bit for bit — *and* that a caller observing every tick and one observing once produce
the identical sequence. It fails on pre-V6 code at step 1. Two more pin that V6 widened nothing: a
command asking for 50 rad/s against a 3 rad/s limit is still `Clamped` with `ViolationKind::
Acceleration` and still counted, and `begin_episode` still re-seeds where a second episode opens.

**5. The oracle that was missing, and where it runs.** `expert_passes_the_evaluation_harness`
(`crates/es/tests/cli.rs`) runs `ScriptedExpert` **as the policy** through `es_eval::Evaluation`
— the real runner, the real plane, the real Task IR success predicate — on the same eight pinned
seeds and against the same `0.875` threshold as `expert_solves_the_pinned_seeds`, which are now
one pair of constants shared by both tests. It is a **server oracle**: only `MuJoCoCpuBackend` can
drive the SO-101 scene, and without `mujoco` it prints a reason and skips, exactly like the
collection oracle it mirrors. Two narrowings, both stated in the test: one `Evaluation::run` per
seed with one episode each (a single 8-episode cell would be eight draws of *one* seed, because
`Evaluation::run` keys the Task IR's randomization by an episode counter and seeds the `Env` from
`seeds[0]`), and a constant 96×96 frame, because the expert reads joints and never pixels and that
is what lets this oracle need `mujoco` without also needing a Vulkan device.

**6. Two asymmetries V6 found and did not fix, because neither is the envelope.** (Closed by V6b,
section 7.13 — and the first of the two is stated wrongly below. The collector does *not* replan at
the inference rate either; it replans every tick and temporal-ensembles through a `ChunkBuffer`,
which evaluation had none of. Section 7.13 has the correction and the measurement.)

* **The evaluation runner replans every control tick.** `run_episode` calls the policy once per
  step and gives the plane a fresh `seq` each time, so the cursor resets and only row 0 of every
  chunk ever executes — `action.execute_chunk = 10` is dead on that path, and the deployment's
  `rate.inference = 5 Hz` is not honoured. The collector replans at the inference rate and executes
  ten rows. This is why `ExpertCfg::pace_to` now takes the rows-per-replan as an argument rather
  than reading `execute_chunk`: it is a property of the *consumer*, not of the envelope. For the
  ACT policy it means evaluation queries the network ten times more often than collection did,
  which is a real train/test mismatch and is open question 13.
* **The evaluation runner resets one episode ahead of the collector.** `Env::new` already resets
  (randomization draw 0), and `run_episode` resets again at the top, so evaluation's episode 0 is
  draw 1 of a seed while collection's episode 0 is draw 0. `--seed 1` names a different cube pose
  on the two paths. Fixing it would move every evaluation number ever reported, so it is recorded
  here and the V6 oracle pins the same *seeds* and the same *threshold* rather than the same draws.

**7. `tests/fixtures/hil/v1_small.eshil` replays identically, and that is a measurement.** The log
carries 120 `observe_state` records with 112 distinct values, so a change to what `observe_state`
means could have moved every decision in it. It did not: the HIL test rig's plant is a perfect
position servo (`q := action.q`), so the measured pose always equals the last emitted command and
the two readings coincide. The fixture was **not** regenerated and `v1_fixture_still_replays_
identically` passes unchanged — which is the strongest available statement that V6 changed the
envelope's reference and not its arithmetic. `es-ros2` and `es-runtime-embedded` need no code
change at all: neither has episodes, both build a fresh plane, and both already observe before
validating.

**8. V3's non-vacuity gate.** V3 panics unless the suite produces at least one `Success` episode
*and* at least one `ActionSource::Clamped` step. After V6 a clamp no longer comes from the servo's
following error, so it has to come from the policy's own chunk: consecutive rows of an ACT chunk
that differ by more than `acceleration_max · dt² = 0.008` rad, or a row past a soft joint limit.
Both are reachable in **every** suite including `nominal` — V1c measured the widened-envelope run
as "almost all `violation.position`", which is the soft-limit stage and is untouched by V6 — and
`torque_noise` / `backlash` remain the suites most likely to push the policy into them by moving
the arm out from under its own chunk. Locally the reachability is pinned without a policy at all,
by `a_command_outside_the_envelope_is_still_clamped` (`es-safety`) and
`the_envelope_violation_rate_rises_when_the_policy_leaves_the_envelope` and
`a_tightened_envelope_clamps_and_a_widened_one_does_not` (`es-eval`). Whether the *trained bundle*
still trips it is a phase-2 measurement, and if it stops tripping it that is a finding to report,
not a number to force.

**Not measured yet.** Everything above is local. The server run — the expert through the evaluation
harness on the demo scene, then V1c's 20,000-step bundle re-measured nominal and across the six
suites with `--jobs 6`, no retraining, no knob moved — is phase 2, and the tables here will be
filled from it.

### 7.13 As built (V6b): evaluation executes chunks the way collection does, and draws the same scene

Packet `docs/packets/M5/V6-envelope-semantics.md`, the V6b section. Section 7.12's finding 6
named two asymmetries and left them; the orchestrator closed both before the phase-2 server run,
because re-measuring with either open would have to be redone. **Finding 6's first half was also
wrong, and the truth is worse than what it claimed.**

**1. The correction. Collection does not replan every ten ticks either — it replans every tick
and *ensembles*, and evaluation did neither.** `BatchDomains::single_env()` declares an inference
period of 1, so `es loop collect` submits an observation and receives a chunk on **every** control
tick. What makes `action.execute_chunk` and `rate.inference` mean something there is not the
submit cadence but `es_env::chunk_buffer::ChunkBuffer`: every arrival is stored, `span` decides how
many ticks a chunk may drive (`execute_chunk`, or the whole horizon under `TemporalEnsemble`), and
`action_at` **blends the overlapping chunks** — `w_i = exp(-decay · i)`, ACT temporal ensembling,
which is exactly what the demo's `learning.toml` (`mode = "TemporalEnsemble"`,
`weight_decay = 0.01`) and `deployment.toml` (`[body.execution.temporal_ensemble] decay = 0.01`)
both declare.

`es_eval::runner` had no buffer at all. It handed the plane each raw inference result under a
fresh `seq`, so the plane's cursor reset every tick and **only row 0 of every chunk ever
executed**. The Deployment IR's `action.execute_chunk = 10` was dead on that path, and so was
`execution` — a policy trained against a fifteen-chunk exponential average was evaluated on its
raw last prediction. That is a far bigger train/test gap than "ten times more often", and it is
the one every V3, V2b and V1c evaluation number was taken under.

**2. The fix is one shared function, not two rules.** `DomainRunner::emit_actions`' chunk-to-plane
step is now `es_env::plane_chunk(buffer, feed, tick, mode)` in `chunk_buffer.rs`, with the
`Submitted` bookkeeping promoted to a public `PlaneFeed`. `es_eval::runner` calls the same
function: it pushes every inference result into a `ChunkBuffer` built from the **Deployment IR**
(`action.execute_chunk`, and the blend implied by `execution`, so `TemporalEnsemble { decay }`
becomes `ChunkBlendPolicy::TemporalEnsemble { weight_decay }`), and serves the plane from it.
`infer_chunk` lost its `seq` argument — the seq is stamped once per *result the buffer accepted*,
which is what `SafetyPlane::accept` judges freshness by (§8.6). `PlaneFeed::end_episode` is the
episode boundary on both sides; `DomainRunner::reset_env` now calls it too. The Deployment IR
decides; there is no flag.

`es loop collect` is bit-unchanged — the extraction is a move, and `es-env`'s and `es-data`'s
suites, including V1c's `a_second_episode_repeats_the_first_exactly` and
`emit_actions_writes_the_planes_answer_and_records_the_command`, pass untouched. That is the
point: the shared function is the collector's, and evaluation came to it.

**3. One remaining difference, named.** Collection models the policy contract's
`expected_latency_ms` (15 ms against a 20 ms control period = one tick), so its first chunk applies
at tick 1 and tick 0 is a chunk underrun — the "50 fallbacks, one per episode" of section 7.10.
Evaluation applies at the tick the result was computed from. The Deployment IR has no latency
field (`deadlines.inference_budget` is a watchdog bound, not a schedule), `Evaluation::run` is not
given the Learning IR, and inventing a field or a flag for it is neither this packet's call nor the
IR's current shape. The effect is one frame per 900-step episode. It is written down here and
nowhere else pretends otherwise.

**4. `--seed S` now names one scene.** `Env::new` resets once — draw 0 of `(seed, env, episode)`
(§6.3) — and every episode ends with exactly one reset, `Env::step`'s own on a terminal condition
or the explicit one when the step budget runs out. `Collector::run` relies on that and never resets
before its loop. `run_episode` reset **again** at the top, so evaluation's episode `i` ran on draw
`2i + 1` while collection's ran on draw `i`: `--seed 1` put the cube somewhere else on the two
paths, silently, because both are valid poses. The reset is gone. Nothing else moved: the bottom
reset that closes a budget-exhausted episode stays, `plan.reset()` stays, and cells still get a
fresh `Env` each.

**5. Oracles.** `episode_zero_runs_on_the_first_randomization_draw` (`es-eval`, local, fake
backend) builds the ground truth from `Env` directly and asserts the first state the runner serves
is `Env::new`'s own draw, bit for bit — **measured to fail with the second reset restored**
(`0.5904` against `0.8683`). `the_runner_feeds_the_plane_through_the_collectors_chunk_buffer`
(`es-eval`) and `the_collector_resets_once_per_episode_and_never_before_the_first_observation`
(`es-data`) are the two call-discipline scans, in the style of the seeding one V6 added, so the
extra reset cannot simply move to the other side. `collection_and_evaluation_draw_the_same_scene_
for_a_seed` (`crates/es/tests/cli.rs`) is the cross-path oracle the orchestrator asked for: both
`es_data::Collector` and `es_eval::Evaluation` on the demo scene with seed 1, comparing the `qpos ‖
qvel` each first hands its policy as raw `f64` bits. It needs `MuJoCoCpuBackend` — the SO-101 scene
has no other driver — so it is named here as a **server oracle** beside the two expert ones.

**6. What this invalidates.** Every evaluation number in sections 7.8, 7.9, 7.10 and 7.11 — V3's,
V2b's and V1c's success rates, envelope-violation rates, `ActionSource` histograms, episode lengths
and the two widened-envelope diagnostics. All of them were measured with no chunk buffer, no
temporal ensembling, `execute_chunk` dead, the envelope bounding the following error (V6), and the
cube one draw away from the seed's own (V6b). **Nothing in those tables carries forward.** The
collection numbers do: the demonstrations, the loss curves, `observation_hash`, `lowering_hash`,
`dataset_schema_hash` and the trained checkpoints are all products of the collect and train paths,
neither of which moved. Phase 2 re-measures V1c's committed 20,000-step bundle — no retraining, no
knob moved — and that is the first evaluation table plan V has produced that measures the policy
rather than the harness.

### 7.14 As built (V7a phase 1): the cube's pose enters the state port, and what that is allowed to prove

Packet `docs/packets/M5/V7a-privileged-state-policy.md`. Section 7.13 invalidated every
evaluation number this demo had produced, so the next one has to be worth taking. The vision
demo asks two questions at once — can this graph do the task, and can a from-scratch ResNet18
find a 25 mm cube in 96×96 pixels — and V7a splits them by answering the first with the second
removed: **the cube's pose is put into the state port.**

It is stage 1 of the video, "state policy". Stage 2, vision at a higher resolution with more
demonstrations, is a separate packet.

**The stop rule, written down before the measurement.** If the state policy does not reach
`success_rate ≥ 0.5` nominal on seeds 101–116, the next step is not a bigger model, a longer
schedule or more demonstrations. It is the physics, the contact model or the expert's
trajectories. A policy handed the cube's exact pose, the arm's exact joint angles and 50
demonstrations of a 7-second scripted motion has no information left to lack.

**1. The channel, and the three constraints that shaped it.** `sim_cube_pose` is a second
`ObservationSpec` channel, `ObsSource::JointState { body: <the `cube_free` joint's id>, dof: 7 }`,
typed `f32[7]`. Nothing was added to `es-ir`.

* `es_eval::runner::input_sources` resolves a plan input's source id against `ModelInfo.qpos`
  **before** it falls back to the Task IR channel's `dof`, and `ModelInfo.qpos` is keyed by
  **joint** id. So a channel that names the `cube_free` joint is served `Capture::Qpos(6..13)` —
  the joint's own seven values — while `joint_state`, which names the robot's `base` *body*,
  keeps falling through to `Capture::Joints(6)`, "the leading six of the row". Both arms already
  existed; this packet adds neither.
* **Task IR-D cannot emit a free joint's `qpos`.** `GetJointState`'s output type is
  `f32[joints.len()]`, one scalar per joint *name* — the ceiling section 5.4 recorded, and the
  reason the success predicate sees the cube's `x` alone. The graph therefore shows the value as
  `GetBodyPose{cube, World}` → `Concat{axis 0}` of `pos[3]` and `quat[4]` → `ObservationSpec`,
  which the existing node set expresses and which type-checks edge for edge. The **channel**
  names the joint rather than the body, because the channel decides the *reading*: through the
  joint it is `qpos`, exactly; through the body it would be `xpos ‖ xquat`, which no recorded
  dataset carries.
* **`NormalizeStats::Range` is one `(lo, hi)` per port**, which is the second reason not to
  simply widen `joint_state` from 6 to 13. At the state branch's `±1` the cube's 0.06 m draw
  spans 0.03 of the output range; at `±0.3` — the arm's reach, containing both the draw
  (`x ∈ [0.21, 0.27]`) and the bin's interior (`x ∈ [0.09, 0.19]`, `y ∈ [-0.15, -0.05]`) — it
  spans 0.10. The quaternion's four values leave `[0, 1]` under that range (`w = 1` maps to
  2.17); a `Normalize` is affine and not a clamp, and a box resting flat carries no signal
  there, so it is written down rather than worked around.

**2. Privilege is marked by name, because there is no field to mark it with.** `ObsChannel`
carries `source` and `ty` and nothing else — §7.4 is explicit that the rest belongs to the
Observation IR — so no `provenance` or `sim_only` field was invented. The mark is the `sim_`
prefix, and it is carried unchanged by the Task IR channel, the Observation IR output port, the
Learning IR input, the policy contract and `contract.json`, so it is visible at every hop of the
chain. `sim_cube_pose` and not `sim.cube_pose`: the port name reaches Python as a key of
`forward(**inputs)`, and a dot is a namespace separator in a LeRobot feature name.

Keeping it a *second* port rather than a wider `joint_state` is also what lets the vision packet
drop it again without moving `joint_state`'s hash. The arm's six angles come off real encoders;
the cube's seven numbers come off nothing a robot carries, and the two do not get mixed into one
tensor on the strength of both being floats.

**3. The dataset needed nothing, and that was the finding.** `es_data::collect::to_lerobot`
writes `observation.state` as env 0's whole `qpos` followed by its whole `qvel` — `13 + 12 = 25`
for this scene — so the cube's pose has been in every demonstration since V1. No column was
added, the v2.1 writer and the v3.0 export are untouched, and **`dataset_schema_hash` does not
move**. What moved is which slices of the row the Observation IR reads.
`docs/api-notes/lerobot-dataset.md` now records the layout, because the next three sentences
depend on it.

**4. The bake had to learn the offset, and the refusal matters more than the arm.** A `qpos`
`IndexRange` indexes the recorded row with the same two bounds `capture` indexes
`StateView::qpos_of(0)` with — the row is not a re-encoding of the state, it is the state with
`qvel` appended. So `ObservationBake::frame` gains `Capture::Qpos(r) => row[r]` and
`ObservationBake::new` gains the `Option<&ModelInfo>` V2b denied it, since a `qpos` range has to
come from the model that ran; `es dataset bake` resolves model-free first and only loads the
scene through `MuJoCoCpuBackend` when that is refused.

The refusal is the part that had to be got right. Model-free, `Capture::Joints(dof)` means "the
leading `dof` of the row", and **only one channel can be leading**. With a second `JointState`
channel declared and no model, which one that is is not in the documents — it is in the model —
so `input_sources` names the port and refuses. Without it the demo would have baked `row[..7]`,
six arm angles and the cube's `x`, into the privileged port, and trained on it silently: the
loss would have fallen, and the number would have meant nothing. That is the same class of
defect as section 7.9's, caught before it produced a table rather than after.

**5. What is pinned locally.** `a_baked_frame_is_bit_identical_to_what_capture_serves` now runs
**two** state ports: `j0` through the leading-`dof` reading and `j1` — whose `qpos` range starts
at **1** — through `Capture::Qpos`. Every output tensor of every frame is byte-equal to what the
policy was served, with no tolerance, and the test asserts `q[0] ≠ q[1]` so it cannot pass by
serving one port the other's values. `a_bake_refuses_two_state_channels_without_a_model` pins the
refusal, and that the same pair resolves once the model is handed over.

The two `es dataset bake` CLI tests now need `MuJoCoCpuBackend` and print a skip reason without
it — the demo's second channel is resolved against the scene's `qpos` ranges, and the only
authority on that layout is the backend that loads it. Deriving it from the MJCF here would be a
second implementation of the kind section 7.9 removed. Their fixture row also became the real
`qpos ‖ qvel` width (25) rather than the six values no run of `es loop collect` has ever written.

**6. The hashes that moved.** All derived, all from the declared generator
(`cargo test -p es --test cli -- --ignored regenerate_visible_learning_documents`):

| slot | before (V6b) | after (V7a) |
|---|---|---|
| `task_hash` | `aec2aea9…1bf1` | `6cf826c1…6b7b` |
| `observation_hash` | `f4a50730…f6e0` | `6c18f455…2daf` |
| `learning_hash` | `82faf8c7…1b54` | `5dac0a46…46f0` |
| `policy_hash` | `01583940…88cd` | `c94c2732…4a07` |
| `evaluation_hash` | `5d70c21c…9ff7` | `0259fd44…f041e` |
| `lowering_hash` | `956abb67…ec6d` | `fdd68ec4…0719` |
| `deployment_hash` | `3b2ad568…6db1` | unchanged |
| `compiler_hash` | `f2a02e84…70d6` | unchanged |
| `dataset_schema_hash` | — | unchanged |

`deployment_hash` is in the table so that it can be seen not to have moved: the envelope is V6's
and V7a is forbidden from `es-safety`.

**Not measured yet.** Everything above is local. The server run — bake the existing 50-episode
set through the widened Observation IR, lower, train 20,000 steps on V2b's exact knobs with
`--resident-gpu`, pack, sweep seeds 101–116 on the three checkpoints, then the six suites on the
20,000-step bundle, then the 4×4 mosaic — is phase 2, and the tables here will be filled from it.
The first command of that run is also the first real exercise of `es dataset bake`'s model load,
which has no local coverage because this machine has no `mujoco`.

### 7.18 As built (V10): the scene diagnosis

Packet `docs/packets/M5/V10-scene-diagnosis.md`. Section 7.15's finding 8 named three suspects
for why the demo still cannot be learned now that the harness passes the scripted expert
(V6/V6b, sections 7.12–7.13) and the model is ruled out (V8, section 7.16: LeRobot's own ACT,
a bitwise-verified import, 100,000 steps, 0/16 · 0/16 · 1/16). The three were the
demonstrations, the grasp, and the temporal ensemble. V10 measures all three — no training, no
model change, no fixture and no hash moved — and **all three are sound**. The measurement that
proved the second one sound found a fourth thing nobody had looked at, and that is the finding.

Everything below is the V1c training set: `~/artifacts/plan-v/v1c/ds-train`, `es loop collect
--episodes 50 --seed 1`, 50 episodes, 18,263 frames, `task_hash aec2aea9…`, `observation_hash
f4a50730…`, `dataset` content `51a953d6…`. Measured on the oracle server, 2026-09-15.

**1. The recorded actions reproduce the demonstrations: 50/50.**
`recorded_actions_replay_to_the_same_outcome` (`crates/es/tests/cli.rs`) replays each episode's
recorded action rows open-loop from the same reset, through the same `Env` and the same
`SafetyPlane`, and scores the result against what the demonstration itself recorded (the cube
inside the bin's three-dimensional interior, which is what `expert_solves_the_pinned_seeds`
checks and what the Task IR's x-only predicate cannot — section 5.4). A plain replay loop over
`Env`, not a second `PolicyRuntime`: `INV-17` allows seven extension points and a
recorded-action player is none of them, and the plane still sees every row.

| replaying | cube in the bin | `Termination::Success` | ticks the plane corrected | worst correction |
|---|---|---|---|---|
| the demonstrations' own record | **50 / 50** | — | — | — |
| `action` (the executed `SafeAction`) | **50 / 50** | 49 / 50 | 10,787 / 18,263 | 9.004e-4 rad |
| `action_commanded` (the raw command) | **50 / 50** | 50 / 50 | 11,589 / 18,263 | 3.258e-1 rad |

Three things this pins. **The data is sound** — every demonstration is a reproducible
pick-and-place, and the executed-vs-commanded distinction costs nothing: both columns put all
fifty cubes in the bin. **`action` really is the plane's own output** (section 7.10's claim,
now measured from the other side): re-validating it moves it by at most 0.9 milliradians, which
is `f32` storage and the rate limit rounding, while re-validating the raw command moves it by
up to 0.33 rad. **The 49/50 is the Task IR predicate, not the physics**: one episode's cube is
inside the bin and the x-only cone leaf says otherwise, which is exactly the ceiling section 5.4
recorded. Replaying with a per-episode chunk `seq` instead of a per-run one scores 1/50 — the
plane accepts a chunk only for a `seq` greater than the last it saw and `begin_episode`
deliberately does not reset that, so every episode after the first is a permanent underrun. That
is a property of the replay loop, written down here because it is the first thing the next
person will get wrong.

**2. The expert grasps; it does not push: 50/50.**
`python/es/grasp_probe.py` replays the same fifty action sequences with plain `mujoco` 3.13 — no
`es` runtime in the loop — and logs per tick the contact normal force between each jaw and
`cube_geom`, the cube's height, and the gripper joint's commanded versus measured position. The
reset state and the action rows come from the replay above (`ES_V10_DUMP`), because only that
side knows the Task IR's randomization draw; the cube column of the dump is the self-check.

| | measured |
|---|---|
| demonstrations that lift the cube clear of the table (> 20 mm, its half-height) | **50 / 50** |
| lift, median / max | 122.65 mm / 127.67 mm |
| ticks with **both** jaws in contact, median / max | 161.5 / 474 |
| the grasp, as a fraction of an episode | 47.9 % |
| gripper joint while both jaws hold the cube | 0.0950 rad (commanded −0.05, open 0.9) |
| demonstrations ending with the cube in the bin | 50 / 50 |
| worst cube divergence from the `es` replay | **0.0002 mm** |

A 25 mm cube carried 122 mm above the table, held between two jaws for half the episode, is a
grasp by any definition. The 0.0950 rad is the geometric closure the jaws stall at with the cube
between them — the number section 5.3's `grip_closed = −0.05` overshoots on purpose so the servo
saturates and the hold is firm.

**3. The temporal ensemble lags; it never opens the gripper.**
`the_temporal_ensemble_survives_the_grasp_window` drives one demonstration through
`es_env::plane_chunk` exactly as `es loop collect` and `es eval run` do, and feeds a *second*
`ChunkBuffer` the identical chunks under `HardSwitch`. The second one is the raw command — the
newest chunk's own row for the tick — so the difference between the two is the ensemble and
nothing else. Seed 1, `decay = 0.01`, `CHUNK_SLOTS = 8` overlapping chunks of a 16-row horizon.
The episode is 351 ticks: `Approach` 0, `Descend` 71, `Close` 118, `Lift` 143, `Transport` 202,
`Lower` 268, `Release` 314, and it ends in `Success`.

| joint | max &#124;blend − raw&#124; over ticks 118–338 | raw min | blend min | raw max | blend max |
|---|---|---|---|---|---|
| 0 `shoulder_pan` | 0.22955 | −0.00031 | −0.00031 | 0.77272 | 0.77272 |
| 1 `shoulder_lift` | 0.26197 | −1.37040 | −1.37040 | 0.16505 | 0.16505 |
| 2 `elbow_flex` | 0.20923 | 0.24469 | 0.24469 | 0.71384 | 0.71384 |
| 3 `wrist_flex` | 0.22593 | 1.08776 | 1.08776 | 1.51702 | 1.51702 |
| 4 `wrist_roll` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 5 `gripper` | **0.35651** | **−0.05000** | **−0.05000** | 0.90000 | 0.90000 |

**Every joint's blended command still reaches both ends of what the newest chunk asked for**,
the gripper included: the blend closes to exactly `grip_closed = −0.05` and opens to exactly
`grip_open = 0.9`. The ensemble is a *lag* — up to 0.357 rad on the gripper, about six and a
half ticks of its ramp — and not a loss of range, because the extremes are held long enough for
all eight overlapping chunks to agree on them. The jaw joint measures 0.0678 rad at its
tightest, against the 0.0950 rad measurement 2 reports while the cube is held: the blend is well
inside the closure the geometry allows. The test asserts the equality rather than a tolerance,
so a change to `decay` or `CHUNK_SLOTS` that did open the gripper would fail it.

**4. What the three measurements found on the way: one control step is 5 ms, not 20.**
`grasp_probe.py` takes `--substeps` as a *measurement*, because only the right value keeps its
cube column on the `es` replay's. At `--substeps 1` the worst divergence over fifty episodes is
**0.0002 mm**; at `--substeps 4` it is **202.98 mm**. So one recorded action row is exactly one
MuJoCo step of the scene's own `timestep="0.005"` — **200 Hz** — while
`tests/fixtures/visible-learning/deployment.toml` declares `rate.control = 50` and
`rate.inference = 5`. Nothing wires the two together: `Env::new` loads the scene with
`LoadConfig { rate: None }`, so the physics keeps the MJCF's timestep, and `Env::step` advances
`schedule.domains().inference.period` ticks, which `BatchDomains::single_env()` — the only
thing `Collector::run` and `es_eval::runner` ever build — sets to 1.

Three consequences, all arithmetic:

* **Every dynamic Safety Plane limit is four times looser than the physics it governs.**
  `SafetyPlane`'s `dt_s` comes from the Deployment IR, so `velocity_max · dt = 3.0 · 0.02 =
  0.06` rad is allowed per *5 ms* step: 12 rad/s against a declared 3. `acceleration_max · dt²`
  is looser by sixteen. The envelope is not violated, it is simply measured in the wrong unit.
* **The signal a policy has to resolve is 3 milliradians.** Measured over the fifty episodes,
  `|action[t] − qpos[t−1]|` has a median of **0.00322 rad** and `|action[t] − qpos[t]|` of
  **0.00251 rad**. A 16-row chunk therefore spans about 80 ms and roughly 0.05 rad of travel,
  on an action space whose joints run ±1.7 rad. At `--substeps 4` the same demonstrations show
  0.0480 rad of chunk travel instead of 0.0982, and the per-tick increment drops to 0.00033 —
  the numbers move because the dataset is being read at the rate it was authored for.
* **The demonstrations are four times longer than anyone thinks.** A 351-tick episode is 1.76 s
  of simulated time, not 7 s; `max_episode_steps = 900` is 4.5 s.

The one-tick pairing question that this measurement makes askable is answered and is *not* a
cause: `Env::step` records `ctrl` beside the qpos the step ended in, so `observation.state[t]`
is the result of `action[t]` rather than the row it was computed from, which is the opposite of
LeRobot's pairing — but at 5 ms the two readings differ by 0.7 milliradians
(0.00251 against 0.00322), which is noise. A chunk's L1 against the trivial predictor "repeat
the joints you can already see" is 0.0982 rad over a whole trajectory and 0.0964 rad inside the
grasp window, so the grasp is not a vanishing fraction of the objective either: it is 47.9 % of
the episode and no harder to predict than the rest of it. Both hypotheses are recorded here as
measured and closed.

**Verdict.** The scene, the demonstrations, the expert and the ensemble are not the cause. The
data is a reproducible, two-jaw, 122 mm-lift pick-and-place in all fifty episodes; replaying it
through the full stack puts fifty cubes in the bin; and the chunk blend the policy is trained
against and evaluated through preserves the expert's commands exactly at both ends of every
joint's range. What V10 found instead is that **the whole demo has been running at 200 Hz while
every document declares 50 Hz**, which makes the learning problem four times finer than it was
designed to be — 3 milliradians of commanded motion per step, 80 ms per chunk — and makes every
dynamic envelope number four to sixteen times looser than the step it bounds.

**The smallest next packet** is therefore not a model, a resolution or a demonstration count: it
is to make one control step last one control period. `Env::new` already has the field —
`LoadConfig::rate` is `Some` away from setting the physics rate, and `BatchDomains` already has
a period the schedule honours — so the change is to derive one of the two from the Deployment
IR's `rate.control` and refuse a scene whose timestep cannot divide it. Then re-collect and
retrain with V2's exact knobs and compare against V8's 100,000-step numbers. Nothing else in
plan V should move until that number exists, because every training and evaluation number the
demo has produced was taken at four times the intended rate. Open question 16.

## 8. Safety overlay (V3)

Per rendered frame, V3 appends one record to `events.json`:
`{ frame, tick, source, events }` where `source` is the existing four-way classification from
`SafetyCounters` deltas (section 2.6) and `events` is the `ViolationKind` bitset for that step. The mosaic
draws a red border on a cell whose record says `Clamped` or `Fallback`, and the running success rate comes
from `report.json`'s `MetricSpec::SuccessRate` — both already computed
(`crates/es-eval/src/metrics.rs:32`, `:97-100`).

**The demo must be non-vacuous**, the same rule W1d's gate uses: V3's oracle fails unless the run contains
at least one `Clamped` step and at least one `Success` episode. A demo of a Safety Plane that never fires
demonstrates nothing. The clamp is produced by tightening the Deployment IR's velocity and rate limits for
the demo suite — **changing the envelope, never disabling the plane** (INV-12).

## 9. Determinism, and what the hash chain covers

`execution_hash = H(task, observation, learning, policy, dataset, deployment, compiler, runtime,
hardware_capability)` (§5.3). Plan V's claim is deliberately narrower than "the video is reproducible":

| Artifact | Claim | Why |
|---|---|---|
| `report.json`, `evaluation.lock` | same `evaluation_hash` -> same numbers | already the M2 property; every draw is `EnvRng::new(seed, suite_id, episode_idx, stream)` (`crates/es-eval/src/perturb.rs:9-12`) |
| `frames/**/*.bin` on the **CPU** render path | byte-identical for the same `execution_hash` | the CPU reference is the golden path today (`crates/es-render/tests/render.rs:14-16`, `:197-209`) |
| `frames/**/*.bin` on the **GPU** path | bit-equal to the CPU reference **on a device that passes the existing GPU oracles**; across vendors, `Target / Status: unverified` | the repo refuses to bake one driver's arithmetic into a golden (`render.rs:14-16`) |
| the physics trajectory | `DeterminismTier::PhysicsMeaning`, not bitwise | `MuJoCoCpuBackend` declares tier 3 deliberately (`crates/es-physics-backend/src/mujoco.rs:28-38`) |
| the trained weights | **not** reproducible | CPU PyTorch across versions; section 6.3 rule 3 is a threshold for this reason |
| `demo.mp4` | **not** in the hash chain | the encoder is `cv2`/FFmpeg on the host, outside the chain; the frames are the evidence, the mp4 is a view of them |

So the honest headline is: **same `execution_hash` -> byte-identical frames on the CPU render path**;
everything downstream of the encoder and everything upstream of a trained checkpoint is not, and each
packet says which side of that line it is on.

## 10. Oracles and CI tiers

| Oracle | Checks | Needs | Tier | Missing -> |
|---|---|---|---|---|
| upstream-model cross-check (V0) | our derivative == pinned `so101.xml` kinematics | network or a cached copy | oracle job | `SKIP` + reason |
| `es task compile` on the demo IRs (V0) | the four documents validate and hash | — | PR | — |
| MuJoCo loads the demo scene (V0) | `nq`, `nv`, `nu`, `nbody` as declared | `mujoco` | oracle job | `SKIP` |
| CPU render goldens (V0b) | new frames match the CPU reference | — | PR | — |
| GPU == CPU frame (V0b) | bit equality on this device | Vulkan + `slangc` | GPU job | `SKIP` (GPU class) |
| expert success rate (V1) | >= a pinned fraction over a pinned seed set | `mujoco` | oracle job | `SKIP` |
| LeRobot reads our dataset (V1) | real `lerobot` opens it and agrees on shapes | `lerobot[dataset]` | oracle job | `SKIP` — **not installed today** |
| module identity, contract, loss, round trip (V2) | section 6.3, all four | `torch` | oracle job | `SKIP` |
| Evaluation IR report + non-vacuity (V3) | success rate, >= 1 `Clamped`, >= 1 `Success` | `mujoco` + `torch` | oracle job | `SKIP` |
| mosaic from fixed frames (V4) | byte-identical mosaic + overlay | — | PR | — |
| mp4 encode (V4) | a playable file of the right frame count | `cv2` | oracle job | `SKIP` |

SKIP reasons must avoid the words `gpu`, `vulkan`, `render`, `slangc`, `device` unless the skip really is
a GPU skip: `xtask` classifies those (as `docs/design/ros2-boundary.md` section 8 records).

## 11. Packet order

```
V0 (scene + IR)  ∥  V0b (render in the loop)  ∥  V4 (video assembly)
                          └──────────┬──────────┘
                                     ▼
                                 V1 (expert + dataset)
                                     ▼
                                 V2 (training + pack)
                                     ▼
                                 V3 (perturbation + safety + the demo run)
```

The user's requested order was `V0 ∥ V4 -> V1 -> V2 -> V3`. Two changes, both forced by section 2:

- **`V0b` is new.** Nothing in the repo connects a renderer to the env loop, dumps a frame, or accepts an
  image observation (section 2.2). V1's frames, V2's image inputs, V3's overlay and V4's real input all
  depend on it. It is the largest prerequisite plan V discovered.
- **V4 stays parallel to V0, as asked**, because it is deliberately written as a *consumer*: it takes a
  directory of frames and an `events.json` and produces an mp4, and its oracle is a checked-in synthetic
  frame fixture. It does not need V0b to be written or tested; it needs V0b to produce the *real* demo,
  which happens in V3.

Each packet is budgeted at or under ~1,000 `src/*.rs` lines (section 2.10) and no packet touches `es-ir`.

## 12. Open questions for a human

1. **`lower_act`.** Plan V routes around the M4 debt rather than fixing it (section 2.5). Default: leave it —
   the demo trains the IR-owned graph, and `lower_act` keeps serving gate 5's checkpoint oracle. The
   alternative is widening §8.3's `TemporalEncoder { Transformer }` to carry layers, FFN width and latent
   dim, which is a spec change and lands in `es-ir` with 53 lines of budget left.
2. **LeRobot dataset version.** The repo writes v2.1 (`crates/es-data/src/lerobot/meta.rs:148`) against an
   api-note that pins nothing (`docs/api-notes/lerobot-dataset.md:3-8`), while the server's `lerobot` is
   0.6.1. Default: keep writing v2.1 and let V1's oracle decide — if 0.6.1 refuses it, the finding goes in
   the api-note and a v3 writer becomes its own packet, not a change inside V1.
3. **"For N consecutive steps".** Default `N = 1` plus a velocity bound (section 5.4). If a real settling
   counter is wanted, it is IR-C, not a new IR-D node.
4. **Perturbations V3 leaves `Unsupported`.** `Occluder` needs a scene edit, `CameraExtrinsic` /
   `CameraIntrinsic` need the `ImageSpec` intrinsics to move with the camera (INV-14, the reason they were
   refused), `ColorTemperature` needs a spectral light model the renderer does not have. Default: leave all
   four `Unsupported`; V3 implements `LightIntensity` and `LightDirection` only.
5. **Video codec.** Measured: only `mp4v` encodes through `cv2` on the server (section 2.9). Default: ship
   `mp4v`. **Answered in passing (V3):** `~/.local/bin/ffmpeg 7.0.2-static` is on the server already, so an
   H.264 copy of the same raw mosaic frames costs one pipe and is 7x smaller (1.5 MB against 10.5 MB for 18
   seconds). Nothing was installed and `encode_video.py` is unchanged; the H.264 file is a second view of
   the same frames, not a second pipeline.
6. **Pretrained backbone.** ~~Whether `_backbone("resnet18", 512)` loads ImageNet weights, and therefore
   whether training needs the network, is unverified.~~ **Answered (V2, V2b).** It does not: no `weights=`
   argument is passed, so training is from scratch and needs no network. V2 found the IR's
   `pretrained = true` was being *ignored*; V2b makes `lower_to_torch` refuse it instead, and the demo's
   `learning.toml` now declares `false` (section 7.9).
7. **PNG.** Default: raw `.bin` + sidecar, the existing golden format (section 7.2). Add a PNG encoder only
   if a human wants to open frames in an image viewer.
8. **CPU-only PyTorch on a 4090** (section 2.9). Default: accept it and state no training time. A human
   installing a CUDA build of torch in `~/venvs/es-lerobot` is the single highest-value prerequisite for
   plan V's schedule.
9. **`lerobot[dataset]` is not installed** on the server, so `lerobot.datasets` cannot import. Until a human
   installs it, V1's most important oracle is a permanent `SKIP`.
10. **Gate 7.** Does closing plan V close §28.7 gate 7 as met with the SO-101 / cube-into-bin substitution
    recorded, or does gate 7 stay open for a Franka with two RGB views? Default: record it met, with the
    substitution and the `Target / Status: unverified` rows of section 1 named in the M5 review.
11. ~~**The state port is normalized at inference and not in training** (section 7.8).~~ **Answered
    (V2b).** The default was taken: `es dataset bake` runs the dataset through the same Observation IR
    executor `capture` uses, `train_act.py` reads the baked set, and the demo was retrained on it with
    V2's exact knobs. `observation_hash` did not move, so the packed bundles and `evaluation.toml` stayed
    valid — the cheap alternative (dropping the state `Normalize`) would have moved it. Section 7.9 has
    the new table beside V3's.
12. **The demonstrations' commands and the Deployment IR's envelope were never checked against each
    other, and that is what stops the arm** (section 7.9). Measured: `SafetyPlane` lets a command
    advance `velocity_max * dt = 3.0 * 0.02 = 0.0600` rad per control step, while the recorded
    `action` column leads its own `qpos` by a median **0.2413** rad — the position servo's
    steady-state tracking error, which ACT faithfully learns to emit. So a policy that imitates the
    demonstrations perfectly is velocity-clamped from the first step of every episode,
    `envelope_violation_rate` is `1.0`, and the violation-rate watchdog latches into
    `hold_position` about a tenth of the time. Three ends could move and a human should pick one,
    because they are not equivalent: **(a)** record `action` as the *next commanded position* rather
    than the servo target, which is a V1 change and re-collects the dataset; **(b)** give the
    Learning IR a delta action space so the policy emits `qpos + delta` and the plane sees a ramped
    command, which is an IR change and moves `learning_hash`; **(c)** widen `velocity_max` for the
    demo's Deployment IR to something the demonstrations actually fit, which is the INV-12-legal
    move (widen the envelope, never disable the plane) but makes the demo's envelope no longer the
    one the demonstrations were recorded through. Default, and the cheapest honest one: **(c) first,
    to find out whether the policy can do the task at all**, then (a) or (b) to earn the tight
    envelope back. V2b does none of them — it is forbidden from `es-safety`, `es-env` and
    `deployment.toml` — and its contribution is that the number above exists.

    **Partly answered (V1c, section 7.10), and the question is now sharper.** ~~(a)~~ was already
    done: `action` has always been the post-plane `SafeAction`, and V1c only asserts it and adds
    `action_commanded` beside it. **(c) was run** on V2b's own 20,000-step checkpoint through a
    scratch deployment document: `envelope_violation_rate` falls `1.0000 -> 0.0551`, `Fallback`
    falls `8,520 -> 0`, `ActionSource::Policy` rises `0 -> 13,606 of 14,400` — and `success_rate`
    stays `0.0000`, every episode a 900-step timeout. **The plane was not what was stopping the
    arm.** V1c also found that the collect and evaluation paths measure the envelope against two
    different references (commanded motion against following error), that closing that from the
    collector costs the scripted expert its own oracle (2/8 against a pinned 0.875), and that
    "only episode 0 is solved" was none of section 7.6's three suspects but the inference phase
    (`frame == 0` never fires when a latency is declared; fixed, 1/50 -> 50/50). What is left for a
    human is narrower: **(b)**, a delta action space, or **(c)** as a permanent widening of the
    demo's `deployment.toml` — and (c) now has to be argued on grounds other than success rate,
    because it does not move it.

    **Answered (V6, section 7.12).** The premise of the whole question was wrong in one specific
    way: the `0.2413` rad lead was never measured against `velocity_max` at all on the collect
    path, and on the evaluation path it was measured against `acceleration_max · dt² = 0.008` rad
    — the servo's *following error*, which spec §9.3 does not have a row for and §9.5 forbids the
    plane from reading. V6 makes measurement enter the envelope exactly once per episode, as the
    seed, on every path. `velocity_max` bounds the plane's own commands, which is what the
    scripted expert has always paced itself to and what the demonstrations have always been
    recorded through. **No `deployment.toml` number moved**, and option (c) is withdrawn: the
    envelope was never what the demonstrations did not fit. Option (b), a delta action space, is
    still open on its own merits — it would make the policy's output a ramp by construction rather
    than by imitation — but it is no longer a fix for anything V6 did not fix. What remains is
    section 7.10's own conclusion: given an envelope it does not fight, ACT drives 94.5 % of the
    steps itself and still cannot do the task. **The bottleneck is the policy.**
13. **Evaluation replans ten times more often than collection, and resets one episode ahead of it**
    (section 7.12, finding 6). `es_eval::runner` calls the policy once per *control* tick, so
    `action.execute_chunk = 10` and `rate.inference = 5 Hz` are both dead on that path and the
    network is queried at 50 Hz against demonstrations collected at 5 Hz — a train/test mismatch
    nobody chose. Separately, `Env::new` resets and `run_episode` resets again, so `--seed S` names
    draw 0 of the task's randomization when collecting and draw 1 when evaluating. Default: leave
    both, because either fix moves every evaluation number plan V has ever reported and neither is
    an envelope question. A human who wants the two paths comparable frame for frame should ask for
    the replan cadence first — it is the one that could plausibly be costing the policy its success
    rate.

    **Answered (V6b, section 7.13), and the question was understated.** The orchestrator closed
    both before the phase-2 run. The replan half was also *wrong*: collection replans every control
    tick too (`BatchDomains::single_env()` declares an inference period of 1) — what it has and
    evaluation did not is `ChunkBuffer`, which is where `action.execute_chunk` and ACT temporal
    ensembling live. Evaluation handed the plane each raw result under a fresh `seq` and executed
    row 0 only, so a policy trained against a fifteen-chunk exponential average was evaluated on
    its raw last prediction. `es_eval::runner` now feeds the plane through `es_env::plane_chunk`,
    the collector's own function, with the buffer built from the Deployment IR. The double reset is
    gone. Every evaluation number in sections 7.8–7.11 is invalidated; every collection and
    training number carries forward.
14. **Privilege has no field, and a `sim_` prefix is doing the work of one** (section 7.14).
    `es_ir::task::ObsChannel` is `{ source, ty }` and §7.4 is explicit that nothing else belongs
    there, so V7a marks `sim_cube_pose` by name rather than inventing a `provenance` or
    `sim_only` flag. Nothing validates it: a Deployment IR aimed at a real SO-101 would happily
    reference an Observation IR that reads a channel no robot can supply, and the only thing that
    would stop it is a human reading the port name. Default: **keep the convention** — a field
    with one user and no validator is the speculative abstraction `INV-17` exists to refuse, and
    the vision packet is expected to delete this channel rather than deploy it. A human who wants
    the chain to *enforce* it should ask for a validator rule (Deployment IR in `Real` execution
    mode refuses an Observation IR whose Task IR channels are not all robot-supplied), which is a
    spec change and an `es-ir` packet, not a field.
15. **What the state policy is allowed to conclude** (section 7.14). V7a's stop rule says that if
    the privileged policy does not reach `success_rate ≥ 0.5`, the next suspect is physics,
    contact or the expert's trajectories — not model size. The converse is the open half: if it
    *does* reach it, that is evidence the graph, envelope, expert and training loop are sound, and
    it is **not** evidence that vision will work at 96×96 with 50 demonstrations. Default: treat a
    passing state policy as the go-ahead for stage 2 at a *higher* resolution and a larger set,
    and record the state policy's number as the ceiling the vision policy is measured against.
16. **The demo runs at 200 Hz and every document says 50** (section 7.18). `Env::new` loads the
    scene with `LoadConfig { rate: None }` and `Env::step` advances
    `schedule.domains().inference.period` simulation ticks, which `BatchDomains::single_env()`
    — the only one `es loop collect` and `es eval run` ever build — sets to 1. So one control
    step is one MJCF timestep, 5 ms, while `deployment.toml` declares `rate.control = 50` and
    `rate.inference = 5`. Measured, not inferred: the mujoco probe's own replay tracks the `es`
    replay to 0.0002 mm at one substep and diverges by 202.98 mm at four. Two ends could move
    and they are not equivalent: **(a)** set `LoadConfig::rate` from the Deployment IR's
    `rate.control`, which changes the *physics* timestep and therefore the contact behaviour
    the demonstrations were recorded through; **(b)** derive `BatchDomains::inference.period`
    from `rate.control / <the scene's physics rate>` and refuse a scene whose timestep does not
    divide the control period, which keeps the physics exactly as it is and makes one control
    step four substeps of it. Default, and the one that leaves the scene alone: **(b)**. Either
    way every training and evaluation number plan V has produced was taken at four times the
    intended control rate, so the re-collect and the retrain are part of the same packet, and
    V8's 100,000-step numbers are what the result is compared against.
