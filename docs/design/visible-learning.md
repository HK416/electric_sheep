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
es policy lower --policy untrained.esb --out build/
<py> python/es/train_act.py --module build/ --dataset ds/ --out model.safetensors
es policy pack --policy untrained.esb --weights model.safetensors --out trained.esb
```

- `lower` writes `TorchModule.source` verbatim plus `contract.json` — the `weight_keys` and
  `weight_shapes` the lowering declares (`crates/es-policy/src/lower/torch.rs:297`) and the
  `lowering_hash`.
- `train_act.py` `exec`s that source, builds `EsPolicy()`, reads the LeRobot v2.1 dataset, runs the
  optimizer, and writes `safetensors` **keyed exactly by `contract.json`**. It knows nothing about the IR
  and is not allowed to define a layer.
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
5. **Video codec.** Measured: only `mp4v` encodes on the server (section 2.9). Default: ship `mp4v`. If an
   H.264 file is wanted, a human installs `ffmpeg` (or an OpenCV with `libx264`) — plan V installs nothing.
6. **Pretrained backbone.** Whether `_backbone("resnet18", 512)` loads ImageNet weights, and therefore
   whether training needs the network, is unverified. Default: train from scratch on the demo's small image
   if it does.
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
