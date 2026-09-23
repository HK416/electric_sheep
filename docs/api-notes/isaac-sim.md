# Isaac Sim — installation, headless scripting, MJCF/URDF import, PhysX determinism

Pinned digest for **M11 X2** (`docs/packets/M11/X2-adapter-v2.md`) and the open item M8 review
R8 names ("Isaac Lab as the second source — an `es-usd` scene, the same adapter shape",
`docs/reviews/M8.md`). No crate imports Isaac Sim; since M11 I0 (2026-09-23) one server venv
(`~/venvs/es-isaac`) installs it and `python/physx_smoke.py` drives it. §1–§6 are web-research
prep — every claim there is tagged `verified (fetched)` (read from the cited page on
2026-09-23) or `unverified` (not found in a primary source, or found only in a
secondary/community source). **§7 is measured by us** (M11 I0, `docs/packets/M11/I0-isaac-sim-install.md`)
and overrides §1–§6 where they disagree; **§8 is the `PhysXBackend` built on it** (M11 I1),
also measured. Korean sibling: `isaac-sim.ko.md`.

Sources fetched 2026-09-23:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.0.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/namespace_isaacsim__asset__importer__mjcf.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.0.0/core_api_tutorials/tutorial_core_hello_world.html>
- <https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html>
- <https://github.com/isaac-sim/IsaacSim>, <https://github.com/isaac-sim/IsaacLab/releases>

## 1. Version pinned, install, EULA

| | | status |
|---|---|---|
| pinned version | **5.1.0** (released 2026-01-28) | verified (fetched), pip_installation.html + install_python.html both target `==5.1.0` |
| newer releases seen | Isaac Sim 6.0 GA (2026, exact date not fetched) and an Isaac Lab **v3.0.0-EA** (2026-09-16) "built for Isaac Sim 6.1", GA targeted end of October 2026 | unverified beyond the Isaac Lab release notes; **not** pinned here because the paired Isaac Lab release is itself Early Access, not the latest stable |
| Python | **3.11** (Isaac Sim 5.x). 3.10/3.12 fail pip resolution ("could not find a version that satisfies isaacsim") | verified (fetched) |
| pip install | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` | verified (fetched), both Isaac Lab and Isaac Sim install pages agree |
| GLIBC | **2.35+** (`manylinux_2_35_x86_64` wheel tag); Ubuntu 20.04's GLIBC 2.31 cannot pip-install, binary install only | verified (fetched) |
| Ubuntu | 22.04 / 24.04 | verified (fetched, requirements.html) |
| NVIDIA driver | Linux **580.65.06+**, Windows **580.88+** (minimum tier; same number at "good"/"ideal") | verified (fetched, requirements.html) |
| GPU | minimum tier names GeForce **RTX 4080**; "ideal" names RTX 5080 / RTX PRO 6000 Blackwell; **GPUs without RT cores (A100, H100) are explicitly unsupported** | verified (fetched); RTX 4090 is not named in the tier table |
| RAM / VRAM | 32 GB / 16 GB minimum, 64 GB / 48 GB ideal | verified (fetched) |
| disk | 50 GB SSD minimum, up to 1 TB NVMe "ideal" | verified (fetched) |
| headless, no display | `SimulationApp({"headless": True})` is the documented standalone entry point (§2) and is what every CI/cluster workflow in Isaac Lab's own `train.py --headless` uses | verified (fetched, pattern used throughout Isaac Lab docs); a direct "runs with zero display attached" statement was not found on the requirements page itself |
| EULA | env var **`OMNI_KIT_ACCEPT_EULA`**, values `YES`/`Y`/`1` (case-insensitive), set before `import isaacsim`; otherwise a runtime prompt blocks first launch. License: **NVIDIA Omniverse License Agreement**, <https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html> | verified (fetched) |

**On our GPU server** (`docs/api-notes/gpu-server.md`-class box, RTX 4090 24 GB): VRAM (24 GB)
clears the 16 GB minimum and sits between the RTX 4080 (min) and RTX 5080 (good) consumer
tiers by generation; the 4090 is simply absent from NVIDIA's named tier table, not excluded by
it the way A100/H100 are (no RT cores). Whether it is *validated* — as opposed to merely
meeting the numeric floor — is **unverified**; the vendor list is a compatibility target, not
an exhaustive allow-list, and community reports of RTX 4090 use are common but not a primary
source we're citing here.

Container install is the other documented headless-friendly route: `docker run … -e
"ACCEPT_EULA=Y" …` (verified, fetched from the container-install page linked in search but not
independently refetched here — carried over from `pip_installation.html`'s cross-reference).

## 2. Minimal headless standalone script

**verified (fetched)**, Isaac Sim 5.0/5.1 Hello-World tutorial + Core API Articulation docs.
Module names below are the **current** ones; the pre-4.0 names (`omni.isaac.core`,
`omni.isaac.core.articulations.Articulation`) were renamed to the `isaacsim.*` namespace and
are deprecated aliases in 5.x, not the primary API.

```python
# 1. SimulationApp must exist before any other isaacsim/omni import.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

world = World()                       # wraps the USD stage + physics context
world.scene.add_default_ground_plane()

# ... add a robot prim to the stage (asset reference, or the MJCF/URDF importer, §3) ...
# from isaacsim.core.prims import Articulation
# robot = world.scene.add(Articulation(prim_paths_expr="/World/Robot", name="robot"))

world.reset()                         # required before the first step; initializes prims added to scene

for _ in range(500):
    action = ArticulationAction(
        joint_positions=np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]),
    )
    # robot.apply_action(action)      # or joint_efforts=... for torque control
    world.step(render=False)          # physics tick; render=False skips the render pass

# joint_pos = robot.get_joint_positions()
# joint_vel = robot.get_joint_velocities()

world.reset()                         # back to the scene's initial state
simulation_app.close()
```

Notes, all **verified (fetched)** except where marked:

- `World()` is a singleton wrapper over `SimulationContext` + `Scene`; `world.scene.add(...)`
  registers a prim wrapper so `world.reset()` initializes it.
- `ArticulationAction(joint_positions=…, joint_velocities=…, joint_efforts=…,
  joint_indices=…)` — passing `joint_indices` targets a subset; otherwise the array must match
  the articulation's full DOF count and **order** (§3's ordering caveat applies here too:
  unverified whether it is USD-authoring order or MJCF/URDF source order after import — treat
  as importer-defined per robot, confirm per asset).
- `isaacsim.core.api.World` vs the newer `isaacsim.core.experimental.*` namespace seen in the
  MJCF importer's own example (`stage_utils.open_stage`): 5.1 ships both; `core.api` is the
  tutorial-documented one and the one used above, `core.experimental` is a newer, `unverified`
  as to which is meant to replace which by 6.x.

## 3. MJCF importer

Extension **`isaacsim.asset.importer.mjcf`** (renamed from `omni.isaac.mjcf_importer` pre-4.0).
Enabled by default; `File > Import` in the GUI, or scripted:

```python
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

omni.usd.get_context().new_stage()
import_config = MJCFImporterConfig(mjcf_path="/path/to/model.xml")
importer = MJCFImporter(import_config)
output_usd_path = importer.import_mjcf()          # writes a .usd next to the source by default
result, stage = stage_utils.open_stage(output_usd_path)
```

verified (fetched) from the extension's own tutorial page. **Measured correction (§7):** the
5.1.0 wheel's extension (`isaacsim.asset.importer.mjcf` 2.5.13) exposes no `MJCFImporter` /
`MJCFImporterConfig`; its Python API is the two Kit commands `MJCFCreateImportConfig` and
`MJCFCreateAsset(mjcf_path, import_config, prim_path, dest_path="")` (the extension's own
`docs/api.rst` and tests use them). The class names above belong to a later release.

`MJCFImporterConfig` fields seen: `mjcf_path` (file or directory), `robot_type` (schema hint:
Default/Manipulator/Humanoid/Wheeled/…), `import_scene` (pull in MJCF's own `<option>`/lighting
as simulation settings), `merge_mesh`, `collision_type` (Convex Hull / Convex Decomposition /
Bounding Sphere / Bounding Cube), `allow_self_collision`, `fix_base`, `link_density`,
`run_asset_transformer` — **verified (fetched)**, names only; per-field defaults were not
individually confirmed (**unverified**).

**What it parses**, read off the C++/Python binding namespace's class list (`verified
(fetched)`, `namespace_isaacsim__asset__importer__mjcf.html` — presence of a class is evidence
the element round-trips into USD; absence is evidence it is silently dropped, not proof, since
an element could be folded into another class):

| MJCF element | importer class | reading |
|---|---|---|
| `<body>`, `<inertial>` | `MJCFBody`, `MJCFInertial` | supported |
| `<joint>` | `MJCFJoint` | supported, **with a caveat**: USD/PhysX joints are strictly two-body; a MuJoCo body with multiple joints to the same parent (a common multi-DOF pattern) is collapsed into one PhysX **D6** joint, and a live bug report says this loses per-axis joint metadata (`isaac-sim/IsaacLab#6854`, unverified beyond the issue title) |
| `<geom>` | `MJCFGeom` | supported — includes mesh geoms (STL via `MJCFMesh`) |
| `<contact>` (`<pair>`, `<exclude>`) | `MJCFContact`, `ContactNode` | supported — this is the class that reads `contype`/`conaffinity`-style filtering; per-attribute mapping to PhysX collision groups is **unverified** |
| `<tendon>` | `MJCFTendon` | class exists → some support; whether fixed *and* spatial tendons both import, or only fixed (the one PhysX has a native analogue for), is **unverified** |
| `<equality><connect>` | `MJCFEqualityConnect` | class exists → `connect` (weld-style) equality constraints are read; other equality types (`joint`, `tendon`, `distance`, `weld` with non-identity offset) are **unverified**, and a 2023-era forum report says MuJoCo/Isaac Gym-era equality-connect support was buggy (unverified whether fixed in this extension) |
| `<actuator>` | `MJCFActuator` | supported — the class exists, but **whether `position`/`velocity`/`motor`/`general` actuator types map onto distinct PhysX joint-drive stiffness/damping, or are all flattened to one drive kind, is unverified**; treat every actuator as "some joint drive got attached" until confirmed per-asset |
| `<site>` | `MJCFSite` | supported (becomes a marker Xform, typically) |
| `<material>`, `<texture>` | `MJCFMaterial`, `MJCFTexture` | supported |
| `<sensor>` | *no `MJCFSensor` class in the namespace* | **reading: not imported.** No sensor-related class appears in the fetched namespace listing; MJCF sensors (framepos, jointpos, touch, …) most likely have no USD counterpart written by this extension. Unverified as an explicit negative — absence of a class is suggestive, not a documented exclusion list |
| `friction` (geom `friction="μ_t μ_r μ_s"`) | — | **unverified** whether all three MuJoCo friction coefficients (sliding/torsional/rolling) map onto PhysX's single dynamic+static friction pair, or only the first is used |

Known limitation, **verified (fetched)**: "special characters in link or joint names are not
supported and are replaced with an underscore; if the name then starts with an underscore, an
`a` is prepended" (USD prim-name legality).

Known limitation, **verified (fetched, GitHub issue title)**: multi-DOF joint collapse into a
single D6 loses per-axis metadata (drive gains, limits) that a downstream reader would need to
reconstruct the original per-axis semantics.

## 4. URDF importer

Extension **`isaacsim.asset.importer.urdf`**, analogous shape: `URDFImporterConfig` /
`URDFImporter.import_urdf()`. Config fields named in the docs: `urdf_path`, `usd_path`,
`collision_from_visuals`, `merge_mesh`, plus a joint-drive post-process step where each joint's
drive type (Position/Velocity) and strength (stiffness) are set explicitly — **verified
(fetched)**, names only; this note does not go deeper because URDF is not MJCF's concern for
M11 X2 (that packet's "second source" is an Isaac Lab **policy**, and Isaac Lab's own robots are
already USD, making the importer a fallback path rather than the primary one — unverified
whether M11 X2 will need URDF import at all).

## 5. PhysX determinism

**verified (fetched)**, `IsaacLab/…/features/reproducibility.html`:

- Given identical hardware and an identical Isaac Sim/PhysX version, rigid-body and articulation
  scenes reproduce bitwise run-to-run. **Cross-hardware** reproducibility is **not** guaranteed
  — different GPUs/CPUs hit different floating-point rounding paths.
- PhysX gives **no** determinism guarantee for non-rigid bodies (cloth, soft bodies, deformables).
- `enable_enhanced_determinism`: a `SimulationCfg`/`PhysxCfg` flag, **default `False`**,
  "enables/disables improved determinism at the expense of performance" — exact PhysX-level
  mechanism (extra sync barriers vs. a different solver code path) is **unverified**.
- GPU work scheduling is itself a determinism risk even within one card: "runtime changes to
  simulation parameters may alter the order in which operations take place," causing
  least-significant-bit drift that can accumulate over long rollouts — the documented mitigation
  is to apply domain randomization only at setup time, before stepping begins, never mid-episode
  on the GPU pipeline.
- **Solver**: PhysX 5's default `solver_type` is **TGS** (Temporal Gauss-Seidel, value `1`); the
  alternative is **PGS** (Projected Gauss-Seidel, value `0`). Iteration-count fields
  (`min_position_iteration_count` and its velocity/GPU-buffer-size counterparts) exist on
  `PhysxCfg` but exact field names/defaults beyond `solver_type` were not individually
  confirmed here — **unverified**, follow up against `isaaclab.sim.simulation_cfg` source
  directly before relying on a specific default.
- **GPU vs CPU pipeline**: PhysX 5 GPU-accelerated simulation is the default and can be forced
  off by setting `device="cpu"` in `SimulationCfg`; GPU mode requires pre-sized buffers (it
  cannot dynamically grow them), a operational footgun distinct from the determinism question
  itself — **verified (fetched)**.
- **dt / substeps**: Isaac Lab replaced an explicit `substeps` field with `sim.dt` +
  `decimation`: e.g. `dt = 1/60` with a 2-step decimation is "equivalent to" stepping physics at
  `dt = 1/120` twice per control tick — **verified (fetched)**; see §6 of
  `docs/api-notes/isaac-lab.md` for the reach task's concrete numbers.

## 6. What we don't know yet

- ~~Whether a MJCF-imported scene reproduces MuJoCo CPU numerics~~ — measured in §7: it does
  not, by up to 1.73 rad in 100 steps on SO-101, and §7 names the importer rows that explain it;
  with I1's repairs (§8) the same control is 0.30 rad off over 500 steps.
- Exact actuator-type-to-drive mapping (§3) and exact friction-coefficient mapping — measured
  for `<position>` actuators and geom `friction` in §7.3; `velocity`/`motor`/`general` actuators
  and tendons/equalities are still unmeasured (no fixture carries them).
- Isaac Sim 6.0/6.1 and Isaac Lab 3.0's multi-backend (Warp/Newton) architecture were seen only
  in release-note headlines; nothing about their API surface was fetched, since 3.0 is Early
  Access as of this note's date and this project's own Newton backend
  (`docs/api-notes/newton.md`) is the more relevant reference either way.

## 7. Measured on the oracle server (M11 I0, 2026-09-23)

Everything in this section was run by us; artifacts (logs, `pip freeze`, stage reports, every
trajectory, `run_all.sh`) are under `~/artifacts/plan-x/i0/` on the RTX 4090 server.

### 7.1 EULA, install, versions, cost

- **EULA:** the owner accepted the NVIDIA Omniverse License Agreement / Isaac Sim EULA on
  **2026-09-23** (recorded by the M11 orchestrator before I0 started). Every run sets
  `OMNI_KIT_ACCEPT_EULA=YES`.
- **Host:** Ubuntu **26.04.1** (not on NVIDIA's 22.04/24.04 list), GLIBC 2.43, kernel 7.0.0,
  driver **610.57.04**, RTX 4090 24 GB, Ryzen 7 7700, 60 GB RAM, no sudo.
- **Install** (`uv`, no system packages): `uv venv --python 3.11 ~/venvs/es-isaac` (uv-managed
  CPython **3.11.16**), then `uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url
  https://pypi.nvidia.com --index-strategy unsafe-best-match`, then
  `"isaaclab[isaacsim,all]==2.3.2.post1"` (same index flags), then `rsl-rl-lib` (already
  satisfied). All exit 0.
- **Resolved:** `isaacsim 5.1.0.0` (build `5.1.0-rc.19`, 2025-10-17), PhysX / `omni.physx`
  **107.3.26**, MJCF importer extension **2.5.13**, `isaaclab 2.3.2.post1`, `rsl-rl-lib 3.0.1`
  (the wheel's metadata pins 3.0.1; its bundled `isaaclab_rl/setup.py` says 3.1.2), `torch
  2.7.0+cu126`, `numpy 1.26.0`, `warp-lang 1.17.0`; 205 packages in `pip-freeze.txt`.
- **Time:** venv 4 s, `isaacsim` 154 s, `isaaclab` 20 s (torch already in the `uv` cache). App
  start (headless) ≈ 3.5 s; one 100-step smoke ≈ 4.2 s wall end to end.
- **Disk:** venv **17 GB**; the `uv` cache grew 19 → 35 GB during the install (≈ 16 GB). The
  compat libraries of §7.2: 49 MB.

### 7.2 What Ubuntu 26.04 breaks, and the user-space workarounds

| symptom | cause | workaround (no sudo, nothing system-wide) |
|---|---|---|
| `OSError: libxml2.so.2: cannot open shared object file` in `omni.kit.asset_converter`, `omni.kit.tool.asset_importer`, and the URDF **and MJCF importer** plugins (none load) | 26.04 ships `libxml2.so.16` only | the Ubuntu 24.04 `.deb`s `libxml2_2.9.14+dfsg-1.3ubuntu3.9` and `libicu74_74.2-1ubuntu3.1` from `archive.ubuntu.com`, unpacked with `dpkg-deb -x` into `~/opt/isaac-compat/root`, and `LD_LIBRARY_PATH=~/opt/isaac-compat/root/usr/lib/x86_64-linux-gnu` |
| segfault right after `app ready` in `librtx.scenedb.plugin.so` (`carb.scenerenderer-rtx`) with the default experience (`isaacsim.exp.base.python.kit`), even headless | the RTX renderer on this OS / driver — not diagnosed further | start `SimulationApp` with Isaac Lab's physics-only experience `isaaclab/apps/isaaclab.python.headless.kit` (no scene delegate); `isaaclab.app.AppLauncher(headless=True)` picks the same file. Rendering (cameras, `isaaclab.python.headless.rendering.kit`) is therefore **untested** on this host |

With both: Isaac Lab's `AppLauncher` starts headless, `isaaclab_rl.rsl_rl` and `isaaclab_tasks`
import (the pip wheel bundles them as source extensions, not as separate dists), and 18
`*Reach*` gym ids register (`Isaac-Reach-Franka-v0` among them; no SO-101 task ships).

### 7.3 MJCF importer fidelity (SO-101 and `mesh_box`, from the imported USD stage)

Input: the MJCF text `es_physics_backend::scene_to_mjcf` emits for
`tests/fixtures/mjcf/so101_pick_place.xml` and `mesh_box.xml` (dumped by a throwaway binary
outside the repo that calls `es_assets::parse_mjcf` → `es_assets::mesh::load` →
`scene_to_mjcf`), and the raw fixtures themselves. Import: `fix_base` as stated,
`import_inertia_tensor = true`, `create_physics_scene = false` (the `World` owns the scene),
prim path `/World/robot`. Read back with `python/physx_smoke.py --stage-report`.

| MJCF | USD / PhysX after import | verdict |
|---|---|---|
| hinge `range` (rad) | `PhysicsRevoluteJoint` `lowerLimit`/`upperLimit` in **degrees**, exact (±1.91986 rad → ±109.9999°) | kept |
| joint order | DOF order = MJCF order (`shoulder_pan … gripper`) for SO-101; the smoke still matches by name | kept (this asset) |
| joint `armature` 0.028 | `physxJoint:armature` 0.028 | kept |
| joint `damping` 0.6 | becomes the **drive** damping (0.6) and `physxLimit:X:damping` (0.6); the articulation has no passive joint damping | **changed** — inside the drive's `maxForce` clamp instead of outside the actuator's `forcerange` |
| joint `frictionloss` 0.052 | `physxJoint:jointFriction` 0.0 | **dropped** |
| `<position kp=998.22>` | drive `type = force`, `stiffness = 998.22` as written; USD angular drive gains are per **degree** | **changed** — no rad→deg conversion (57.3× stiffer in N·m/rad if the USD unit holds; the drive saturates either way here) |
| `<position kv=2.731>` | nowhere (the drive damping is the joint damping) | **dropped** |
| `forcerange ±2.94` | drive `maxForce` 2.94 | kept |
| `ctrlrange` | not written | dropped (the caller must clamp targets) |
| body `mass`, `<inertial fullinertia>` | `physics:mass` exact; `diagonalInertia` = MuJoCo's principal moments (reordered) + `principalAxes`; `centerOfMass` exact | kept |
| `mass=` on a non-colliding geom (`camera_mount_shell mass=0.012`, `contype=0`) | body `physics:mass` 0 (PhysX derives it from the colliders at default density) | **dropped** |
| free joint, `fix_base = 0` | the body gets its own `ArticulationRootAPI` (a 0-DOF articulation) and falls correctly | kept |
| free joint, `fix_base = 1` | a `PhysicsFixedJoint rootJoint_<body>` to the world: `fix_base` applies to **every** root, so the free cube is welded | **changed** — the smoke deletes the weld |
| jointless root body (`base`) | welded only with `fix_base = 1`; with 0 the arm would be free | caller's choice; MJCF semantics need 1 |
| `<worldbody>` geoms (table, bin) | one kinematic `RigidBodyAPI` Xform per geom under `/World/robot/worldBody`, which also carries `ArticulationRootAPI` **without** a rigid body → `World.reset()` fails (`'NoneType' object has no attribute 'is_homogeneous'`) | **broken** — the smoke removes that API |
| `plane` | `Plane` collider | kept |
| `box` / `sphere` / `capsule` | `Cube` / `Sphere` / `Capsule` colliders | kept |
| mesh geom from a file (STL) | `Mesh` collider, `physics:approximation = convexHull` | kept (as a hull) |
| inline mesh `vertex=`/`face=` (what `scene_to_mjcf` emits) | the importer looks for a file named after the mesh, logs `Unsupported Format (/meshes/box)`, then `[Fatal] attempted member lookup on NULL TfRefPtr<UsdStage>`, and the process **exits 0** | **broken** — the smoke writes each inline mesh to an OBJ next to a rewritten MJCF |
| `contype = conaffinity = 0` | no collider (visual only) | kept |
| other `contype`/`conaffinity` bitmasks | not mapped: two `PhysicsCollisionGroup`s (`robotCollisionGroup` = `/World/robot`, a prototype group filtered against it); every collider under the robot collides with every other | **dropped** (bitmask semantics) |
| geom `friction` (1.0 / 0.005 / 0.0001) | no physics material authored anywhere → PhysX default material | **dropped** |
| `solref`, `solimp`, `condim`, `margin` | nothing | dropped (no analogue) |
| `<option>` `integrator`, `cone`, `solver`, `iterations`, `impratio` | nothing; the scene is TGS, patch friction, PCM, articulation 32/1 and body 16/1 position/velocity iterations | dropped (no analogue) |
| `<option timestep>`, `gravity` | the smoke's `World(physics_dt=…)` runs 200 Hz / 1000 Hz and gravity 9.81 as in the MJCF (the importer's own `create_physics_scene` path is untested) | caller sets them |
| — | every rigid body gets `physxRigidBody:angularDamping = 0.05` | **added** (MuJoCo has none) |
| `<sensor>` | — | not measured: neither fixture declares one |

Also measured: `World.reset()` advances physics **two** steps before the caller's first
`world.step` (a free body starts at z = 0.3 − 3·g·dt²), so PhysX row k is MuJoCo's row k+2.

### 7.4 Same control, MuJoCo CPU vs PhysX — numbers only, no tolerance claimed

Control: every `<position>` actuator gets `mid + ¼·(hi−lo)·sin(2πk/100 + i)` over its
`ctrlrange` before step k; 100 steps at the MJCF's own timestep; row k is the state after step
k; `max |Δq|` over the 100 rows. MuJoCo 3.13.0 (`~/venvs/es`); PhysX CPU pipeline unless noted.
All from `~/artifacts/plan-x/i0/run_all.sh`.

| scene | pair | max \|Δq\| (rad): pan, lift, elbow, wrist_flex, wrist_roll, gripper | free body max \|Δpos\| |
|---|---|---|---|
| SO-101 emitted (dt 5 ms) | MuJoCo vs PhysX CPU | 0.781, 1.73, 1.05, 1.26, 1.30, 0.851 | cube 3.9e-4 m (MuJoCo's soft contact sinks it 0.11 mm; PhysX holds z = 0.02) |
| SO-101 raw fixture | MuJoCo vs PhysX CPU | 0.781, 1.73, 1.05, 1.26, 1.30, 0.851 | cube 3.9e-4 m |
| SO-101 emitted | MuJoCo vs PhysX GPU pipeline (`cuda:0`) | 0.716, 1.71, 1.13, 1.26, 1.31, 0.853 | cube 3.9e-4 m |
| SO-101 emitted | PhysX CPU vs PhysX GPU | 0.069, 0.261, 0.273, 0.065, 0.0012, 0.0028 | 1.3e-7 m |
| SO-101 emitted | PhysX CPU run vs rerun | 0 (bitwise-identical JSON) | 0 |
| SO-101 | MuJoCo emitted vs MuJoCo raw | ≤ 1.2e-15 | 0 |
| SO-101 | PhysX emitted vs PhysX raw | 2.1e-4, 3.6e-5, 1.8e-4, 1.9e-5, 1.1e-5, 4.8e-7 | 0 |
| `mesh_box` (dt 1 ms, 100 steps, free fall) | MuJoCo vs PhysX, emitted and raw alike | — | 1.99e-3 m (the two reset steps: 0.25046 vs 0.24847 m at row 99) |
| `mesh_box`, 500 steps (lands at ≈ 0.2 s) | MuJoCo vs PhysX | — | 1.55e-2 m (at row 299, just after impact, MuJoCo is at 0.0464 m and PhysX at 0.0500 m; at rest MuJoCo 0.04989 m, PhysX 0.05000 m); mesh and primitive box identical to each other in both engines |

At row 99 of the SO-101 run MuJoCo is at (−0.165, −0.014, 0.142, 0.418, 0.289, 0.550) and
PhysX at (−0.946, −1.740, −0.686, 1.644, 1.131, −0.136); both lag the targets because every
drive sits on its 2.94 N·m limit for most of the run.

Attribution (diagnostics, same script): with the drive gains rewritten from the MJCF
(`--drive-gains rad`: stiffness kp, damping kv + joint damping; `deg`: the same × π/180) the gap
does not shrink (max 0.99, 1.69, … and 0.84, 1.67, … rad). With MuJoCo's passive joint damping
and frictionloss set to 0 (the two rows PhysX drops or moves inside the clamp), MuJoCo matches
the as-imported PhysX run to ≤ 1.5e-3 rad over the first 10 steps on five joints (wrist_flex
0.032) and ≤ 0.02 rad over 20 on all but wrist_flex (0.074): while both engines sit on the
force limit, those two rows are the divergence. The later residual (up to 0.83 rad) is not
attributed.

**For I1:** a `PhysXBackend` built on this importer has to (1) write inline meshes to files,
(2) remove the bodiless `worldBody` articulation root, (3) not weld MJCF free bodies, (4)
author the drives itself (kp in per-degree units, kv, and a decision on passive damping and
frictionloss), (5) author friction materials, (6) own the reset/step count, and (7) start from
the physics-only experience on this host — each a mapping-report row. §8 says what I1 did with
each.

## 8. `PhysXBackend` (M11 I1, 2026-09-23)

`crates/es-physics-backend/src/physx.rs` + `python/physx_ref.py`, packet
`docs/packets/M11/I1-physx-backend.md`. Measured on the same server and venvs as §7; artifacts
under `~/artifacts/plan-x/i1/`. Every number here is a measurement; no tolerance is claimed.

### 8.1 How it runs

- **A subprocess, never an import.** `physx_ref.py` speaks the `proc.rs` JSON-lines protocol
  (`load`, `reset`, `set_ctrl`, `step`, `state`, `set_state`). No `set_params`: per-env model
  fields were not measured, so `ModelParams` is not declared and `set_params` is refused by name.
- **`ES_ISAAC_PYTHON`** names the interpreter. Unset, or unable to `import isaacsim`, is
  `is_available()`'s `Err`, the callers' SKIPPED exit 3. The Rust side sets
  `OMNI_KIT_ACCEPT_EULA=YES` and prepends the §7.2 compat libraries to `LD_LIBRARY_PATH`
  (`ES_ISAAC_COMPAT_LIBS`, else `$HOME/opt/isaac-compat/root/usr/lib/x86_64-linux-gnu` when it
  exists); the script starts `isaaclab.python.headless.kit` itself (`ES_ISAAC_EXPERIENCE`
  overrides). Callers set nothing but `ES_ISAAC_PYTHON`.
- **The script runs from a file.** New finding: `SimulationApp` crashes in the breakpad handler,
  with no log, when the process was started as `python -c <script>` (`sys.argv == ["-c"]`).
  The embedded script is written once to the temp dir, named by its blake3, and run as a file
  (`Process::spawn_command`).
- **stdout is the protocol.** Kit logs to stdout; the script dups fd 1 for the protocol and
  points fd 1 at stderr before importing anything. `ES_PHYSX_STDERR=<file>` keeps Kit's log.
- **Pipeline:** `ES_PHYSX_DEVICE=cpu` (default) or `cuda:0`. Deviation from the packet, which
  asked for `LoadConfig`: `LoadConfig` lives in `es-physics-core`, outside I1's context. The
  pipeline is recorded in the engine version (`isaacsim 5.1.0 physx 107.3.26 cpu|gpu`), so
  `backend_identity` and `evaluation.lock` tell the two apart; `gpu_resident` follows it.
- **Envs:** `n_envs > 1` clones `/World/envs/env_0` with `GridCloner(spacing = 0)` and isolates
  the envs with `filter_collisions`; every env keeps the MJCF's own coordinates.
- **State** goes through `omni.physics.tensors` (articulation views for the hinge dofs and every
  link pose, rigid-body views for free bodies), in MuJoCo's layout rebuilt from the MJCF:
  bodies depth-first with `world` first, joints in body order, a free joint as
  `pos ‖ quat(w,x,y,z)` with its linear velocity at the body origin in the world frame and its
  angular velocity in the body frame (PhysX reports the centre-of-mass velocity and a
  world-frame angular velocity; converted both ways with the body's COM offset). `xquat` is
  `x,y,z,w`. One stage per process: a reload is a new process (≈ 4 s to a loaded stage).

### 8.2 The importer items, fixed or declared

"Fixed" means `physx_ref.py` repairs it and it is a `BackendQuirk` in the capabilities (so in
every `evaluation.lock`); "row" means a mapping-report row (`es backend compare`, the spec 14.4
gate), all of them warnings. Nothing is dropped silently.

| item (§7.3, and three new ones) | how |
|---|---|
| inline `<mesh vertex/face>` is fatal, exit 0 | **fixed**: each inline mesh is written to `<mesh name>.obj` in a temp dir and the MJCF copy says `file=`. New: the importer looks a mesh up **by its name**; a file named otherwise imports no collider at all, and the collider prim is named after the mesh, not the geom. Row `ContactMesh` approximated: the convex hull |
| bodiless `worldBody` articulation root | **fixed**: `ArticulationRootAPI` removed |
| `fix_base` welds MJCF free bodies | **fixed**: `rootJoint_<body>` and the free body's `ArticulationRootAPI` removed, so it is a plain rigid body. A free joint on a body with children (a floating base) is refused at load, by name |
| `World.reset()` runs 2 hidden steps | **fixed**: after `reset()` the script writes MuJoCo's qpos0 (hinges 0, free bodies at their imported pose) and zero velocity, so tick 0 is MuJoCo's tick 0. `mesh_box`'s free fall now agrees with MuJoCo to 1e-6 m until tick 225, just after the impact (I0: 1.99e-3 m off from the start) |
| kp written unconverted into USD's per-degree gain; kv dropped | **fixed**: every drive is written through the tensor API — stiffness = kp, damping = kv, max force = forcerange — which takes SI units (N·m/rad). Checked on a one-hinge probe (kp 1, kv 0.05, no gravity, random targets): max \|Δq\| 1.1e-3 rad over 500 ticks against MuJoCo, which a 57.3× unit error could not stay near. A dof with no position actuator gets no drive. Rows `actuator.pd` / `ActuatorPosition` approximated (a PhysX drive is implicit) |
| `ctrlrange` dropped | **fixed**: the script clamps ctrl to it, as MuJoCo does |
| joint `damping` becomes drive damping, inside the force clamp | **row** `joint.damping` approximated: `-d·q̇` (and a joint spring `-k(q − springref)`) is applied as an explicit joint effort before every physics step, outside the clamp as in MuJoCo, where it is implicit |
| `frictionloss` dropped | **row** `JointFrictionLoss` unsupported: PhysX joint friction is a coefficient, not a dry-friction torque |
| geom friction: no material | **row** `geom.friction` approximated: one material per sliding coefficient, static = dynamic = μ, restitution 0, combine `max` (MuJoCo's rule; `priority` ignored) |
| `contype`/`conaffinity` bitmasks | **row** `geom.contype_conaffinity` unsupported, asked when a colliding geom is neither 1/1 nor 0/0. The importer's collision groups filtered nothing and are removed |
| `solref`, `solimp`, `margin` | **row** `ContactSoftParams` unsupported; spec 17.2's `contact.soft_params` keeps its status, with a note that says they are dropped |
| `condim` 4/6 (torsional, rolling) | **row** `ContactCondim6` unsupported |
| `cone = elliptic` | **row** `ContactElliptic` unsupported: runs pyramidal |
| `<option>` integrator / solver / iterations / impratio | **row** `option.solver` unsupported, on every scene: PhysX TGS, the importer's 32 / 1 articulation iterations |
| `mass=` on a non-colliding geom | **row** `body.mass_from_geoms` unsupported, asked when a body has geoms and no `<inertial>` |
| +0.05 angular damping on every body | **fixed**: angular and linear damping 0, and sleeping off (threshold 0), on every body and articulation |
| physics-only experience | **fixed**: the script picks it |
| armature | kept by the importer (§7.3): `JointArmature` native. spec 17.2's `joint.armature` row keeps its "unsupported" status, with a note that the importer keeps it |
| **new:** the importer's `/collisions` (and `/meshes`, `/visuals`) prototypes | **fixed**: they are defined, active, collision-enabled prims, so PhysX simulates every body's colliders a second time as **static colliders at the world origin** (an overlap query at the origin hits `/collisions/wrist/…`, `/collisions/cube/…`). Deactivated at their root; the instance references into the bodies still compose (34 colliders remain on SO-101, all under `/World`). The instances are made non-instanceable so a collider can carry its own material |
| **new:** a zero free-joint quaternion | **fixed**: the env's reset writes only the position of a free joint it randomizes and leaves the quaternion 0; MuJoCo reads that as the identity, PhysX **silently drops the whole transform**, position included. The script normalizes as `mju_normalize4` does (norm < 1e-15 → identity). Before the fix reach A0's cube never moved and nominal scored 0.0 |

Not mapped — blocked by the mapping report before a process starts (spec 14.4): ball joints,
sensors (the adapter reads none), `velocity` / `general` / site actuators, tendons, height
fields. `JointSlide` is implemented but not exercised by any fixture, so it is not declared.

### 8.3 Oracle 2: `es backend compare`, 500 ticks

`es backend compare --scene <scene> --backends mujoco-cpu,physx --ctrl-random --ticks 500`
(splitmix64 targets in [−1, 1], clamped to `ctrlrange` by both engines), and the Isaac-gated
test `physx_against_mujoco_cpu_is_measured` (every actuator at `mid + ¼(hi−lo)·sin(2πk/100 + i)`,
§7.4's control, 500 ticks, then a PhysX run against a second PhysX run).

| scene | pipeline | control | max \|Δqpos\| | max \|Δqvel\| | energy proxy MuJoCo / PhysX (last tick) | max energy-proxy Δ | divergence tick (1e-6) |
|---|---|---|---|---|---|---|---|
| SO-101 | CPU | random | 7.023e-2 | 3.20 | 6.564 / 7.440 | 5.04 | 0 |
| SO-101 | CPU | sine | 3.03e-1 | 5.90 | 31.41 / 35.47 | 30.0 | 0 |
| `mesh_box` | CPU | none (free fall, lands ≈ 0.2 s) | 1.55e-2 | 2.22 | 2.7e-9 / 1.3e-8 | 4.91 | 225 |
| SO-101 | GPU (`cuda:0`) | random | 7.025e-2 | 3.20 | 6.564 / 7.440 | 5.04 | 0 |
| SO-101 | GPU | sine | 3.54e-1 | 5.60 | 31.41 / 35.51 | 31.1 | 0 |
| `mesh_box` | GPU | none | 1.55e-2 | 2.21 | 2.7e-9 / 8.3e-8 | 4.91 | 225 |

- **Reruns:** the two CPU `es backend compare` runs printed identical output for both scenes,
  and the test's PhysX-against-PhysX rerun is **bitwise** on both scenes (max \|Δqpos\| = max
  \|Δqvel\| = 0, no divergence tick). The GPU pipeline's reruns are bitwise too, in both places
  and on both scenes. CPU pipeline against GPU pipeline under the sine control: SO-101 max
  \|Δqpos\| 0.124 rad, diverging at tick 0; `mesh_box` 6.9e-6 m, diverging at tick 225 (the
  impact). Both pipelines are equally far from MuJoCo.
- **Against §7.4:** under the same sine control, I0's as-imported PhysX was up to 1.73 rad off
  MuJoCo within 100 steps; with §8.2's repairs it is 0.30 rad over 500. SO-101 diverges at
  tick 0 from f32 alone; `mesh_box` stays within 1e-6 until the impact.

### 8.4 Oracle 3: reach A0 through `es eval run --backend physx`

W0b's reach A0 (`~/artifacts/plan-w/w0b/reach/seed0/checkpoints/4000.esb`, trained on
mujoco-cpu), `tests/fixtures/rl/evaluation-reach.toml` (16 episodes × 4 suites), scene
`so101_pick_place.xml`, one binary:

| backend | nominal | observation_delay | torque_noise | backlash | mean episode length (nominal) | wall |
|---|---|---|---|---|---|---|
| mujoco-cpu | **0.5625** | 0.3125 | 0.5 | 0.5 | 129.4 | 13 s |
| physx, CPU pipeline | **0.125** | 0.0 | 0.3125 | 0.0625 | 184.9 | 5 min 43 s |
| physx, GPU pipeline (`cuda:0`) | **0.125** | 0.0625 | 0.0 | 0.0625 | 183.4 | 8 min 1 s |
| mjwarp | refused by its mapping report before anything spawns (`ContactElliptic`, M11 X1) | | | | | |

The run completes; its `evaluation.lock` carries `backend = physx` with every quirk above, and
its `execution_hash` differs from mujoco-cpu's (`2fdd30a0…` against `08851281…`, the
`hardware_capability` slot); the GPU pipeline's run hashes apart again (`e16de409…`, engine
version `… gpu`). A policy trained on MuJoCo keeps a quarter of its nominal success
on PhysX: a sim-to-sim gap, measured and not attributed here (the §8.2 rows that remain —
frictionloss, the soft contact, the solver, explicit damping — are the candidates).

The execution hash does not cover `physx_ref.py` itself: the first A0 run, before the
zero-quaternion fix, scored 0.0 under the same hash. That holds for every out-of-process
backend's script today (`mjwarp_ref.py` too) and is a follow-up for the hash rule.
