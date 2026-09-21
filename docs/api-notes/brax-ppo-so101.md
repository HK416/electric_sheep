# brax PPO on our SO-101 scene — the source policy S2b imports

Packet: `docs/packets/M8/S2c-source-policy.md`. Task definition: `docs/design/rl-continuation.md`
section 5 (with the 26-dimensional observation the owner fixed on 2026-09-21). Precedent for the
parameter format: `docs/api-notes/mujoco-playground-quadruped.md` section 2. Korean sibling:
`brax-ppo-so101.ko.md`.

Unlike that quadruped note, **everything below was executed by us**. Every number is
`measured (server, date, path)` — server `renderer-14` (RTX 4090 24 GB, driver 610.57.04,
16 cores), 2026-09-21, artifacts under `~/artifacts/plan-s/s2c/`. One caveat applies to every
throughput number in this note: the GPU was **shared** for the whole session with another job
(the M7 R5 hold-out evaluation, six `es eval run` shards), and a reference float32 matmul
measured **2.1 TFLOP/s** on a card that does far more alone. Read all throughput numbers as
lower bounds measured under contention, not as the 4090's capability.

## 1. Pins, and the one conflict

| package | pinned | resolved |
|---|---|---|
| `playground` (`mujoco_playground`) | **0.2.0** | 0.2.0 |
| `brax` | **0.14.2** | 0.14.2 |
| `jax` / `jaxlib` / `jax-cuda12-{plugin,pjrt}` | **0.9.0** (see below) | 0.9.0 |
| `mujoco` / `mujoco-mjx` | — | 3.13.0 / 3.13.0 |
| `flax` / `optax` / `orbax-checkpoint` | — | 0.12.9 / 0.2.8 / 0.12.4 |
| `numpy` / `blake3` / `jaxopt` / `warp-lang` | — | 2.5.3 / 1.0.9 / 0.8.5 / 1.17.0 |
| Python | 3.12 | 3.12.14 (`~/venvs/es-rl`, uv 0.12.13) |

**The conflict, and how it was resolved.** `playground` 0.2.0 and `brax` 0.14.2 both declare
only `jax` (no upper bound), so `uv` resolved the newest, **0.11.2**. `brax` 0.14.2 calls
`jax.device_put_replicated` (`brax/training/agents/ppo/train.py:756`), which JAX removed:

```
AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.
```

Measured: the attribute is **absent in jax 0.10.0**, **present in 0.9.0 and 0.8.0**. The venv
therefore pins `jax[cuda12]==0.9.0`, the newest JAX that runs the pinned brax. `mujoco-mjx`
3.13.0 works on it unchanged. Anyone recreating the venv must pin JAX; installing
`jax[cuda12]` unpinned reproduces the failure above at the first training step.

```
uv venv --python 3.12 ~/venvs/es-rl
uv pip install --python ~/venvs/es-rl/bin/python "jax[cuda12]==0.9.0" mujoco mujoco-mjx \
    brax==0.14.2 playground==0.2.0 orbax-checkpoint numpy blake3
```

## 2. The task, as implemented

`python/es/rl_source/so101_reach_env.py` is an `mjx_env.MjxEnv` over the committed scene:

* **control** 50 Hz over 200 Hz physics (`ctrl_dt = 0.02`, `sim_dt = 0.005`, `n_substeps = 4`);
* **observation, 26** — `joint_pos[6] ‖ joint_vel[6] ‖ cube_pose[7] ‖ gripper_pose[7]`, each
  pose world-frame `pos[3] ‖ quat[4]` with the quaternion in **xyzw** (§3.1; MuJoCo's `xquat`
  is wxyz and is reordered). `gripper` is the **body**, not the site;
* **action, 6** — normalized position targets, `ctrl = offset + scale · clip(a, −1, 1)` with
  `offset = (hi + lo)/2` and `scale = (hi − lo)/2` per actuator `ctrlrange`;
* **reward** `−dist + 1` on success, `dist = ‖cube_pos − gripper_body_pos‖` (positions only);
* **success** `dist < 0.03` m; **episode** 200 control steps, no early termination;
* **randomization** per episode, reproducing the committed Task IR's nodes
  (`tests/fixtures/visible-learning/task.toml` nodes 28–30): cube `x ~ U(0.21, 0.27)`,
  `y ~ U(−0.03, 0.05)`, `z = 0.02`, every arm joint reset to 0, `qvel = 0`.

Two deviations from the packet's letter, both deliberate:

| # | deviation | why |
|---|---|---|
| D1 | the cube is drawn **once per env**, not once per episode (Playground's default `BraxAutoResetWrapper`, `full_reset=False`) | `full_reset=True` calls `env.reset` — a full `mjx.forward` — for every env on every step. With 4,096 envs and a 2 M-step budget each env sees ~2.4 episodes, so the run is 4,096 independent draws from the same distribution; the re-draw would buy a few percent more variety for a large fraction of the step time on a contended GPU. `eval_brax_so101.py` draws fresh per episode. |
| D2 | success is reported **two ways** — `success_reached` (any step of the episode inside 0.03 m) and `success_final` (the last step is inside) | the task has no early termination, so "the episode succeeded" needs a reading. Both are in `eval.json`; the headline number below is `success_reached`, and for the shipped policy the two are equal. |

## 3. MJX compatibility of the committed scene — measured

**It loads and steps unmodified.** `mjx.put_model` on `tests/fixtures/mjcf/so101_pick_place.xml`
succeeds on `mujoco-mjx` 3.13.0 (`Impl.JAX`, 1.4 s): `integrator="implicitfast"`,
`cone="elliptic"`, `condim` 3 and 6, joint `frictionloss`, `impratio=10` and the `<position>`
actuators are all accepted. Nothing refuses. What it cannot do is **train**:

| measurement (1 env unless stated) | committed scene | derived scene (§3.2) |
|---|---|---|
| static allocation | **1,054 contacts, `efc_J` 5,292 × 12** | **24 contacts, `efc_J` 84 × 12** |
| compile `jit(reset)` / `jit(step)` | 45.4 s / 125.3 s | 15.8 s / 14.9 s |
| one 200-step episode, python loop | did not finish in **45 min** | 3.8 s (52.7 control steps/s) |
| batched, 4,096 envs | 574 env-steps/s (CG; Newton OOMs) | **10,620 env-steps/s** |
| batched, 8,192 envs | out of memory, single 4.10 GiB allocation | not needed |

The cause is not a missing MJX feature, it is `njmax`. MuJoCo's default **Newton** solver forms
a dense `njmax × njmax` constraint matrix per env and factorizes it; at `njmax = 5,292` that is
112 MB per env, and even with the arm pairs excluded (`njmax = 360`, 518 KB/env) 8,192 envs ask
for 4.10 GiB in one allocation and run at 736 env-steps/s.

Two things were tried and rejected:

* **`solver="CG"`** (matrix-free, 3.5× faster at 72 contacts: 2,548 env-steps/s at 4,096 envs)
  **diverges**. With the gripper-vs-cube pairs present, a random full-range policy — which is
  exactly what PPO starts from — drives the cube's free joint to NaN within **11 control steps**
  (`iterations=4 ls_iterations=8`), 60 (10/20) and 40 (10/50). Newton at 10/20 survives the same
  200 steps. CG is out.
* **float64** (`jax_enable_x64`) does not save it: NaN at step 17, with the cube's free-joint
  velocity at 1e52 beforehand. The divergence is the contact impulse, not rounding — the jaws'
  0.75 mm spheres (`condim=6`, `solref="0.01 1"`) sweep through the 25 mm cube inside one 5 ms
  substep when the arm flails.

### 3.1 MJX versus MuJoCo CPU, same scene

One 200-step episode, identical start state and identical control sequence, MJX float32 against
MuJoCo CPU float64 (`python so101_reach_env.py --xml so101_reach_mjx.xml`):

| after | max abs `qpos` difference |
|---|---|
| 1 control step | 6.055e-03 |
| 10 control steps | 9.868e-02 |
| 50 and 200 control steps | 1.619e-01 |

The largest observation-channel difference over the episode is 3.468 (a `joint_vel` channel,
rad/s). This is open-loop chaotic divergence under random full-range targets through a
`kp = 998` position servo; it bounds neither a closed-loop rollout nor the policy's behaviour,
and it is the ordinary float32-vs-float64 gap §3.5 tier 3 allows, not a defect.

### 3.2 The derived scene, and every edit in it

`python/es/rl_source/so101_reach_mjx.xml` **includes** the committed scene and adds one element:

```xml
<mujoco model="so101_reach_mjx">
  <include file="../../../tests/fixtures/mjcf/so101_pick_place.xml"/>
  <contact> … 35 <exclude body1= body2= /> rows … </contact>
</mujoco>
```

Nothing else. **No `<option>` override**: `integrator="implicitfast"`, `cone="elliptic"`,
solver Newton, `iterations=10`, `ls_iterations=50`, `timestep=0.005` and `impratio=10` are the
committed values, and every body, geom, joint, actuator, site and camera is the committed one,
by construction — the file cannot drift from the scene our runtime loads.

| # | edit | what it removes | S4c gap |
|---|---|---|---|
| E1 | 21 `<exclude>` rows, arm link vs arm link | non-adjacent self-collision (shoulder vs wrist, upper_arm vs gripper, …) | the arm may fold through itself in the source env |
| E2 | 7 rows, arm link vs `world` | the arm against the table plane and the five bin geoms | the arm may sweep through the table and the bin |
| E3 | 5 rows, non-gripper links vs `cube` | elbow/wrist/camera-mount bumping the cube | the cube cannot be knocked by the arm's body |
| E4 | 2 rows, `gripper` and `moving_jaw_so101_v1` vs `cube` | **the jaws touching the cube** | the cube is a target pose, not an object: it never moves, and the gripper passes through it |

E4 is the one to read twice. It is what makes the run possible at all (§3, the NaN), and a
policy trained here has never felt the cube. When S4c replays it in our runtime on the committed
scene, contact with the cube is new, and the cube being pushed away is expected, not a bug.
section 5.5 measures what all four cost the trained policy — the answer is "everything", and that is
the honest starting point plan S's continuation is meant to repair.

**Measured cost of the edits, on CPU, where it can be isolated:** stepping the *edited* model
and the *committed* model on MuJoCo CPU from the same state with the same 200 random control
steps gives `max|Δqpos| = 0.000e+00` — bitwise identical. On that trajectory none of the excluded
pairs ever touched, so the edits cost exactly nothing; they only bite when the arm actually
reaches the table, itself, or the cube. Contact counts, by contrast, are what they buy: 1,054 →
24, and 574 → 10,620 env-steps/s.

## 4. The recipe

The server holds no checkout (repo rule: `scp`, never `git clone`). Mirror the two directories
the scripts need, keeping the repo's relative layout — `so101_reach_env.py` resolves the scene
as `../../../tests/fixtures/mjcf/so101_pick_place.xml`, and the derived XML `include`s it by the
same path:

```bash
scp -r python/es/rl_source LJM@192.168.100.14:~/Projects/es-s2c/python/es/
scp tests/fixtures/mjcf/so101_pick_place.xml \
    LJM@192.168.100.14:~/Projects/es-s2c/tests/fixtures/mjcf/
```

```bash
ssh LJM@192.168.100.14
cd ~/Projects/es-s2c/python/es/rl_source
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export XLA_FLAGS=--xla_gpu_deterministic_ops=true

# 0. the scene self-check: MJX vs MuJoCo CPU, and the throughput of a candidate XML
python so101_reach_env.py --xml so101_reach_mjx.xml
python so101_reach_env.py --xml so101_reach_mjx.xml --bench 4096

# 1. train (defaults: --xml so101_reach_mjx.xml, 2 M steps, 4,096 envs)
python train_brax_so101.py --seed 0 --timesteps 2000000 --num-envs 4096 \
    --out ~/artifacts/plan-s/s2c/seed0-run1

# 2. the two oracles
python check_export.py --out ~/artifacts/plan-s/s2c/seed0-run1
python eval_brax_so101.py --out ~/artifacts/plan-s/s2c/seed0-run1 --episodes 64
```

PPO configuration (in `train_brax_so101.py`, in the style of Playground's `PandaPickCube`
manipulation params, with the packet's wider network):

`policy` and `value` MLP **(256, 256)**, `activation=swish`, `normalize_observations=True`,
`unroll_length=10`, `num_minibatches=32`, `num_updates_per_batch=8`, `batch_size=256`,
`discounting=0.97`, `learning_rate=1e-3`, `entropy_cost=2e-2`, `max_grad_norm=1.0`,
`reward_scaling=1.0`, `episode_length=200`, `num_evals=10`, `num_eval_envs=64`,
`deterministic_eval=True`.

## 5. Measured

All rows: server `renderer-14`, 2026-09-21, the GPU shared with the M7 R5 hold-out evaluation
throughout.

### 5.1 Training — `~/artifacts/plan-s/s2c/seed0-run1/`

2,000,000 timesteps, seed 0, 4,096 envs, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`,
**540.6 s wall clock** (9.0 min) with the second determinism run training concurrently on the
same card; an earlier identical pair took 464.0 s. Peak GPU memory ~1.2 GB per run.

| step | reward (episode sum) | steps inside 0.03 m, of 200 | Σ distance |
|---|---|---|---|
| 0 | −55.74 | 0.00 | 55.74 |
| 245,760 | −66.52 | 0.00 | 66.52 |
| 491,520 | −14.60 | 2.20 | 16.80 |
| 737,280 | 184.27 | 188.36 | 4.09 |
| 1,228,800 | 187.25 | 189.50 | 2.25 |
| 1,474,560 | 56.31 | 94.86 | 38.55 |
| 2,211,840 | **188.15** | **190.95** | 2.80 |

The task is solved by ~740 k steps; the dip at 1.47 M is an ordinary PPO wobble and recovers.
A 400 k-step trial run already evaluated at 1.00 success, so the 2 M budget is comfortable
rather than tight. (`curve.json` holds every row.)

### 5.2 Evaluation in the source framework — `eval.json`

`eval_brax_so101.py --episodes 64`, the deterministic policy rebuilt **from `source.npz`** (not
from the checkpoint, so this scores the file S2b imports), fresh per-episode cube draws:

| metric | measured |
|---|---|
| `success_reached` (any step inside 0.03 m) | **1.00** (64/64) — target 0.8 |
| `success_final` (last step inside) | **1.00** |
| final distance, mean / max | 8.41 mm / 14.65 mm |
| return, mean | 188.11 |

### 5.3 Determinism

Two runs, same seed, same flag, `source.npz` **bitwise identical**:

```
blake3 8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3   seed0-run1
blake3 8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3   seed0-run2
```

The same hash also came out of an earlier pair of runs of the same configuration, i.e. four
runs agree. Note the runs were concurrent and the GPU contended, so the reproducibility is not
an artifact of an idle machine.

### 5.4 `check_export.py` — numpy against JAX

| obs set | max abs error | note |
|---|---|---|
| `obs_scaled` (in-distribution) | **1.580e-06** | the gate, tolerance 1e-5 — passes |
| `obs` (`U(−1, 1)`, the packet's draw) | 2.287e-05 | 80.2 % of rows saturate tanh; see §6 |

One measurement that S2b must know: the stored actions are computed under
`jax.default_matmul_precision("highest")`. XLA's **default** float32 matmul on this GPU is
TF32, and that path differs from a true float32 matmul by **1.515e-03**
(`meta.json → oracle.tf32_delta`) — 100× the tier's tolerance, from the hardware path alone.
An importer comparing against TF32-computed actions cannot pass 1e-5 no matter how correct it
is.

### 5.5 The sim-to-sim gap of §3.2, measured on the policy itself

The same `source.npz`, the same 64 evaluation episodes, three scenes — this is the number S4c
will otherwise discover the hard way:

| scene the policy is evaluated in | contacts | `success_reached` | final distance, mean |
|---|---|---|---|
| derived (E1–E4 excluded) — what it trained in | 24 | **1.00** | 8.4 mm |
| derived but with the arm-vs-`world` rows restored (E2 back) | 552 | **0.00** | 71.7 mm |
| the committed scene, nothing excluded | 1,054 | **0.00** | 173.9 mm |

Neither run diverged (no NaN; the returns are ordinary), so this is behaviour, not numerics:
**the trained policy reaches the cube by passing through the table and through itself.** Put
the contacts back and the arm is stopped 7 cm short with the table alone, 17 cm short with
everything. Roughly half the error is E2 (table and bin) and the rest is E1/E3/E4.

That is the honest state of the source policy: it is a valid, reproducible, exactly-importable
function that solves its own environment, and it is *not* a policy that works in ours. For plan
S that is arguably the right starting point — repairing it is exactly what "RL continuation in
our simulator" is for, and the before/after row in S4c's table now has a real "before". It does
mean S2b should not expect the imported policy to score on the committed scene, and S4b should
expect the first continuation iterations to be a re-learning, not a fine-tune.
(`~/artifacts/plan-s/s2c/seed0-run1-committed-scene/eval.json`, `…-keep-world/eval.json`.)

### 5.6 Paths and hashes

| what | where (server `renderer-14`) |
|---|---|
| artifacts | `~/artifacts/plan-s/s2c/seed0-run1/`, `…/seed0-run2/` |
| orbax checkpoint | `…/seed0-run1/checkpoints/000002211840/` (9 checkpoints, 5.5 MB total) |
| logs, benches, NaN probes | `~/artifacts/plan-s/s2c/logs/*.log` |
| venv | `~/venvs/es-rl` (kept) |

| file | blake3 |
|---|---|
| `source.npz` (both runs) | `8c0faf01c4d0e4193815cac4af47262a3daa623d1202d75b2873dfe3bf2283b3` |
| `python/es/rl_source/so101_reach_mjx.xml` | `d193fb75b07f7798d410f024a027d9b70217eef0c5f5042e6b70862ee62460da` |
| `tests/fixtures/mjcf/so101_pick_place.xml` | `94d7fa5fa07e3d0ba23774151bc6cb12339f0882a2e3480cf18e9577043f86ad` |

## 6. The export format S2b reads

`source.npz` — float32 throughout, brax's storage convention (`kernel` is `[in, out]`,
right-multiply; transpose for torch's `[out, in]`):

| key | shape | meaning |
|---|---|---|
| `obs_mean`, `obs_std` | `[26]` | `running_statistics`; apply `(x − mean) / std`, **no clipping** |
| `kernel_0`, `bias_0` | `[26, 256]`, `[256]` | hidden Dense 0, then `swish` |
| `kernel_1`, `bias_1` | `[256, 256]`, `[256]` | hidden Dense 1, then `swish` |
| `mean_kernel`, `mean_bias` | `[256, 6]`, `[6]` | output Dense, first half — brax's `loc` |
| `logstd_kernel`, `logstd_bias` | `[256, 6]`, `[6]` | output Dense, second half |

The deterministic policy, which is the whole of what S2b must reproduce, is

```
x = (obs - obs_mean) / obs_std
x = swish(x @ kernel_0 + bias_0)
x = swish(x @ kernel_1 + bias_1)
a = tanh(x @ mean_kernel + mean_bias)
```

Three facts that are easy to get wrong, each read out of the pinned source:

1. **The output Dense is linear.** `brax.training.networks.MLP` builds `hidden_0 … hidden_n`
   and applies the activation to every layer *but* the last (`activate_final=False`,
   `networks.py:161`). The packet's draft `meta.json` field `activate_output = true` is
   therefore written **`false`**; the only output nonlinearity is the distribution's `tanh`
   (`squash`), which is `NormalTanhDistribution.mode = tanh(loc)`.
2. **The second output half is not a log-std.** brax turns it into a standard deviation with
   `std = softplus(x) + 0.001` (`distribution.py:171`), not `exp(x)`. It is unused at inference;
   S1's `--init-log-std` must apply that formula (and then take a log) rather than read the
   stored numbers as logs.
3. **`obs_std` can be brax's 1e-6 floor.** Three channels of this task are constant — cube `z`
   and two cube-quaternion components, because E4 leaves the cube untouched — so their running
   variance is 0 and `running_statistics.update` clips the std to `std_min_value = 1e-6`
   (`running_statistics.py:233`). Any input that is not on the observation's own scale is
   amplified by 1e6 in those channels. This is why `oracle-1000.npz` carries two sets.

`meta.json` carries the resolved versions, the seed, timesteps and wall clock, `obs_dim = 26`,
`action_dim = 6`, `hidden`, `activation`, `activate_output`, `squash`, the normalizer and
log-std formulas, `obs_layout` (name, start, len, unit per segment) with the quaternion order
spelled out, the joint order, the action `offset`/`scale`/formula, the scene used with its
blake3 *and* the committed scene with its blake3, the env constants, and the full PPO config.

`oracle-1000.npz`:

| key | how it is drawn | what it is for |
|---|---|---|
| `obs`, `actions` | `U(−1, 1)` per channel, numpy seed 0 | the packet's draw, kept verbatim |
| `obs_scaled`, `actions_scaled` | `obs_mean + obs_std · U(−1, 1)`, numpy seed 1 | the same draw on the observation's own scale |

`actions*` are JAX's deterministic actions (`make_inference_fn(params, deterministic=True)`).
The 1e-5 tier is meaningful only on the **scaled** set: on the `U(−1, 1)` set the constant
channels normalize to ~1e6, every `swish` saturates and 99.9 % of actions land on ±1, so two
float32 implementations of the same matmul differ by ~1e-1 on the few rows that are not
saturated. `check_export.py` reports both and gates on the scaled one — a deviation from the
packet's acceptance, taken because the alternative is an oracle that no correct importer can
pass. S2b should compare against `obs_scaled` and treat the uniform set as a saturation probe.
