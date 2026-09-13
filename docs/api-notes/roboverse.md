# RoboVerse / MetaSim task config — shape and provenance

**Pinned version: NONE.** No `metasim` / `roboverse_pack` package is installed in this
workspace; nothing here was read from a running MetaSim. Fields below are marked `verified
(fetched)` when they came from `https://roboverse.wiki/metasim/concept/config.html` on
2026-09-13 (WebFetch, since `metasim.cfg.scenario.ScenarioCfg` / `RobotCfg` source was not
directly reachable — GitHub raw/blob URLs for `metasim/cfg/scenario.py` 404'd), and
`unverified` otherwise (reconstructed from the RoboVerse README, arXiv:2504.18904's abstract,
and spec §0.3/§14.4's one-line description: "simulator-agnostic config, 276 tasks"). A human
must pin a `roboverse_pack`/`metasim` version and correct this file — and
`crates/es-data/src/roboverse.rs` if the shape differs — before anything here is ground truth
(spec §1.7).

`crates/es-data/src/roboverse.rs` accepts **JSON only**, not MetaSim's native Python
`ScenarioCfg`/`TaskCfg` dataclasses and not YAML. A human (or a thin Python script, one call
to `dataclasses.asdict` + `json.dump`) exports the scenario/task config to JSON before this
converter sees it; that step is out of scope here and is not implemented in this crate
(keeps this crate's dependency list at zero new crates — no `serde_yaml`, no PyO3 call-out).

## Top-level shape — mixed verified/unverified

MetaSim's real root config is `ScenarioCfg` (a Python dataclass), which `verified (fetched)`
holds: `robots: list[RobotCfg]`, `objects: list[BaseObjCfg]`, `cameras`, `lights`, `scene`,
plus simulation-runtime fields (`simulator`, `num_envs`, `sim_params`, `decimation`,
`headless`, `renderer`). The docs explicitly say `ScenarioCfg` does **not** carry reward
functions, observation definitions, success checkers, task-level logic, termination
conditions, or algorithm-specific parameters (`verified (fetched)`,
`concept/config.html`) — those live on a separate `TaskCfg` / task-env class
(`BaseTaskEnv`/`RLTaskEnv`, `verified (fetched)`, `concept/task.html`) that carries a
`ScenarioCfg` as one field and implements `_reward`, `_terminated`, `_time_out`,
`_observation`, `_action_space`, `_extra_spec`. This crate models one **flattened JSON
document** that has both halves (`name`, `robots`, `objects`, `cameras`, `episode_length`,
`checker`, `license`) as one `RoboVerseTask` — the export step above is assumed to have
joined `TaskCfg` and its `ScenarioCfg` into one file; no evidence exists of MetaSim itself
ever writing that joined document (`unverified`).

```json
{
  "name": "pick_cube",
  "version": "unverified",
  "license": "Apache-2.0",
  "robots": [ ... ],
  "objects": [ ... ],
  "cameras": [ ... ],
  "episode_length": 250,
  "checker": { ... },
  "randomization": { ... }
}
```

## `robots[]` — mostly `verified (fetched)` field names, `unverified` shapes/defaults

`RobotCfg` (`verified (fetched)`, `concept/config.html`) is simulator-agnostic: one robot
description compiles to whichever backend `ScenarioCfg.simulator` names
(`isaacgym`/`mujoco`/`sapien`/`genesis`/`pybullet`). Fields the fetched page's minimal example
shows:

| field | type | note |
|---|---|---|
| `name` | string | `verified (fetched)` |
| `num_joints` | int | `verified (fetched)` |
| `usd_path` / `mjcf_path` / `urdf_path` | string, one present | `verified (fetched)` — exactly one asset format per robot; this crate reads whichever is present and infers `format` from it |
| `fix_base_link` | bool | `verified (fetched)` |
| `enabled_gravity` | bool | `verified (fetched)` |
| `control_type` | `{joint_name: "position"\|"velocity"\|"effort"}` | `verified (fetched)`: "Dict of joint -> control mode" |
| `actuators` | `{joint_name: BaseActuatorCfg}` | `verified (fetched)`: "Dict of joint -> BaseActuatorCfg"; `BaseActuatorCfg` fields (`stiffness`, `damping`, `effort_limit_sim`) are `unverified` beyond the one example (`stiffness=500, damping=10`) |
| `joint_limits` | unverified shape | `verified (fetched)` that the field exists; per-joint `[lo, hi]` is this crate's guess |
| `default_joint_positions` | optional | `verified (fetched)` name only |
| `curobo_ref_cfg_name` | optional string | `verified (fetched)` name only, motion-planner cross-reference, unused by this converter |

This crate reads `name`, one of `{usd_path, mjcf_path, urdf_path}` (-> `format` = `Usd` /
`Mjcf` / `Urdf`), and `control_type`'s key set as the joint name list — it does **not** parse
`actuators`/`joint_limits` (unmodeled, folded into a warning if present via `extra`). The
robot's asset path becomes a `SceneRefLike` for `es import mjcf|urdf|usd` to actually import
(spec §14.4: this converter maps config shape, not mesh/kinematics — that is the named
external-converter job already owned by `es-assets`).

## `objects[]` — `unverified`

No fetched page enumerated `BaseObjCfg`'s fields. Assumed shape, by analogy with `RobotCfg`
and the general "simulator-agnostic asset description" framing:

| field | type | note |
|---|---|---|
| `name` | string | unverified |
| `usd_path` / `mjcf_path` / `urdf_path` | string, one present | unverified, mirrors `RobotCfg` |
| `pose` | `{pos: [x,y,z], rot: [w,x,y,z]}` | unverified |
| `physics` | `"rigid"` \| `"static"` \| `"articulated"` | unverified; this crate only distinguishes `rigid` (movable, gets a `ResetState`/`Randomization` target) from everything else (fixed scene geometry, `SceneRef` only) |

## `cameras[]` — `unverified`

| field | type | note |
|---|---|---|
| `name` | string | unverified |
| `resolution` | `[width, height]` | unverified |
| `pose` | `{pos, rot}` | unverified, world frame |
| `intrinsics` | `{fx, fy, cx, cy}` | unverified; absent -> this crate synthesizes a nominal pinhole (`fx=fy=width`, centred principal point), same fallback `lerobot_config.rs::nominal_camera` uses |

## `checker` — `unverified` (only class names are `verified (fetched)`, no field shapes)

The RoboVerse/MetaSim README and API surface name at least `DetectedChecker`,
`JointPosChecker`, and `PositionShiftChecker` as success-condition classes (`verified
(fetched)`: these three names appear in the project's own documentation search results), but
no fetched page exposed their constructor fields. This crate accepts a tagged JSON object
`{"kind": "...", ...}` and maps only:

| `kind` | assumed fields (`unverified`) | Task IR mapping |
|---|---|---|
| `"DetectedChecker"` | `{"object": name, "detector": name}` | `Reward`(binary contact/detection proxy via `GetContact`) + `Terminate(Success)` gated on it |
| `"JointPosChecker"` | `{"robot": name, "joint": name, "target": f64, "tolerance": f64}` | `GetJointState` + `Compare(Le, tolerance)` on `|q - target|` -> `Terminate(Success)` |
| `"PositionShiftChecker"` | `{"object": name, "axis": "x"\|"y"\|"z", "distance": f64}` | `GetBodyPose` delta along `axis` compared against `distance` -> `Terminate(Success)` |
| anything else | — | `Unmapped { severity: Error }` (spec §14.4: unmapped items block execution) |

## `randomization` — `unverified`

Spec §0.3 names only "simulator-agnostic config" for RoboVerse, no randomization schema. This
crate accepts an optional `{"target": "<object>.<field>", "distribution": {"kind": "uniform",
"low": f64, "high": f64}}` list (mirrors Task IR's own `Distribution` enum so every entry maps
1:1 to a `TaskNode::Randomization`), `unverified` beyond that shape assumption. A
`distribution.kind` this crate does not recognize is dropped with a warning, not an error —
unlike an unmapped `checker`, a missing randomization term degrades the task rather than
making it meaningless to run.

## Dataset / trajectory format — out of scope by design (INV-16)

RoboVerse's own trajectory storage (`unverified`, believed `.pkl`/`.npz` per common
robomimic-family convention) is **not read by this crate**: `INV-16` bans pickle-based
loading anywhere in this workspace, so a MetaSim trajectory (if pickled) must be re-exported
to LeRobot format by a Python-side script before `es-data::lerobot` reads it. Only the task
*config* (JSON, this file's subject) is converted here.

## License / provenance (spec §25.2)

Spec §25.2 lists "Isaac Lab / RoboVerse conversion output derivative-work status" as
`확인 필요` (needs confirmation) — v1.0 has no answer, only a requirement that the converter
record what it started from. This crate does not resolve the license question; it carries
whatever `license` string (if any) the input JSON names into `Converted::scene_refs[].license`
and `Converted::provenance` (name, version, license), unconditionally, so a downstream
decision has the data to be made with. Absent `license`, the field is `None` and a warning is
emitted — never a fabricated `"unknown"` or a guessed OSS license.
