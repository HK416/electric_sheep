# Isaac Sim — installation, headless scripting, MJCF/URDF import, PhysX determinism

Pinned digest for **M11 X2** (`docs/packets/M11/X2-adapter-v2.md`) and the open item M8 review
R8 names ("Isaac Lab as the second source — an `es-usd` scene, the same adapter shape",
`docs/reviews/M8.md`). Nothing in this workspace depends on Isaac Sim yet: no crate imports it,
no venv installs it. This is web-research prep, not a verified-by-us oracle like
`docs/api-notes/mujoco.md` — every claim below is tagged `verified (fetched)` (read from the
cited page on 2026-09-23) or `unverified` (not found in a primary source, or found only in a
secondary/community source). Korean sibling: `isaac-sim.ko.md`.

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

verified (fetched) from the extension's own tutorial page.

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

- Whether a MJCF-imported scene, run on GPU PhysX with TGS, reproduces **MuJoCo CPU** numerics
  closely enough for any cross-backend comparison — no oracle run was performed for this note
  (this is a research digest, not a measured result like `docs/api-notes/mujoco.md`'s mesh
  section).
- Exact actuator-type-to-drive mapping (§3) and exact friction-coefficient mapping — both
  load-bearing for M11 X2's adapter if the "second source" scene is ever built by MJCF import
  rather than read from Isaac Lab's native USD assets directly.
- Isaac Sim 6.0/6.1 and Isaac Lab 3.0's multi-backend (Warp/Newton) architecture were seen only
  in release-note headlines; nothing about their API surface was fetched, since 3.0 is Early
  Access as of this note's date and this project's own Newton backend
  (`docs/api-notes/newton.md`) is the more relevant reference either way.
