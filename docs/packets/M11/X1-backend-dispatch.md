# M11 X1 — `--backend`: a policy runs closed-loop on MJWarp, and the backend is in the hash chain

Spec: §28.14 rule 2 (a run on a backend other than `mujoco-cpu` writes
`H("es.backend.v1", name, engine version, float, determinism tier)` into the `hardware_capability`
slot of `execution_hash`; `mujoco-cpu` keeps today's slot) and rule 1 and wave 1, §5.3, §17.1–17.2
(backends declare capabilities; a scene a backend cannot map is refused before anything is spawned),
§3.5 (tier 2 `CrossBackend`: never bitwise, the difference is reported as numbers), §14.4, INV-17.
Review: `docs/reviews/M4.md` gate 12 (MJWarp max |Δqpos| 7.4e-8, Newton 5.9e-4, open loop).
Design notes: `evaluation-execution.md` (+ko) gains a "backend" subsection; `rl-continuation.md`
(+ko) section 2 gains the `[rl] backend` line. Type B with a D row.

## the question

`Env<B: PhysicsBackend>` is generic and `MjWarpBackend` / `NewtonBackend` implement every trait
method, yet `es eval run`, `es loop collect` and `es_native.Rollout` hard-code `MuJoCoCpuBackend`
(`crates/es/src/cmd/eval.rs:455,703,776`, `loop.rs:311,330,593,633`, `crates/es-py/src/rollout.rs:102,173`),
so the only sim-to-sim this repository can do is `es backend compare`'s open-loop trajectory.
**With one dispatch on `BackendKind`, does the same bundle run closed-loop on MJWarp — evaluated,
collected and trained through `Rollout` — with the backend recorded in the hash chain, every
committed `mujoco-cpu` hash unmoved, and Newton / PhysX refused by name?**

## spec

* One function per verb picks the backend: `--backend mujoco-cpu | mjwarp | newton | physx`
  (default `mujoco-cpu`) parses to `BackendKind` (`crates/es-physics-backend/src/mapping.rs:26-49`)
  and calls the existing generic entry point monomorphized on the chosen type —
  `Evaluation::run_shard_with_sink::<B, …>(…, B::new, …)` and `Collector::run_with_sink::<B, …>` —
  never `Box<dyn PhysicsBackend>` inside `Env` (the generic stays; §3.4 hot path). Availability
  (`B::is_available()`) keeps the documented SKIPPED exit 3. `newton` reaches `load` and is refused
  there by its mapping report (actuators and sensors undeclared, contacts not wired) — the refusal
  names the rows; `physx` prints `not implemented (M11/I1)` until I1 lands.
* `es_native.Rollout(…, backend="mujoco-cpu")`: the same dispatch behind an internal enum of
  `Env<MuJoCoCpuBackend>` / `Env<MjWarpBackend>` (a closed enum, not a trait object and not a new
  trait — INV-17). `[rl] backend = "mjwarp"` in a training recipe (absent = `mujoco-cpu`, serialised
  like absence, the `estimator` pattern of M9/R5) reaches `train_ppo.py --backend` and
  `training_hash`.
* Hash chain: `es_physics_backend::backend_identity(caps: &Capabilities, engine_version: &str) ->
  [u8; 32]` = blake3 of the tagged tuple in rule 2; `RunConfig.hardware` is that digest for every
  backend except `mujoco-cpu`, whose slot stays today's value, so every committed `evaluation.lock`
  `execution_hash` is unmoved. The engine version comes from the backend's load reply (add
  `engine_version` to `LoadReply` in `proc.rs` and to the three `*_ref.py` if absent; a missing
  version is an error, not an empty string). `evaluation.lock` already prints `backend`; it gains
  `engine_version`.
* The `es eval run`, `es loop collect` and `train` help text list the four names and what each does
  today. `check_deps` prints each backend's availability.

## context

```
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/check_deps.rs
crates/es/tests/cli.rs
crates/es-eval/src/runner.rs
crates/es-eval/src/lock.rs
crates/es-eval/tests/**
crates/es-physics-backend/src/lib.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/newton.rs
crates/es-physics-backend/python/*_ref.py
crates/es-physics-backend/tests/**
crates/es-py/src/rollout.rs
crates/es-py/src/lib.rs
crates/es-py/tests/**
crates/es-data/src/training.rs
python/es/train_ppo.py
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X1-backend-dispatch.md
docs/packets/M11/X1-backend-dispatch.ko.md
```

(If a listed path does not exist — e.g. the lock type lives elsewhere — use the file that holds it
and name it in the report; do not widen beyond the crates listed.)

## oracle

1. `cargo test -p es-physics-backend backend_identity` — the digest differs across the four names and
   across two engine versions, is stable across calls, and `mujoco-cpu`'s slot is today's value.
2. `cargo test -p es --test cli eval_run_backend_` — `--backend newton` on the demo documents is
   refused naming its mapping rows before any process spawns; `--backend physx` names M11/I1;
   `--backend banana` is a usage error (exit 2); `--backend mjwarp` without `mujoco_warp` exits 3
   SKIPPED. The same for `loop collect`.
3. `cargo test -p es --test cli train_rl_backend_dry_run_plan` — a plan golden addition;
   `backend = "mujoco-cpu"` spelled out hashes as absent; `training-reach.toml`'s `training_hash`
   unmoved.
4. Every committed `evaluation.lock` / report pinned by an existing test is unmoved (the whole
   `es --test cli` and `es-eval` suites green, except the known M10 S-1 test).
5. Server (`ES_PYTHON=~/venvs/es/bin/python`, which has `mujoco_warp`; GPU queue lock): reach A0
   (`~/artifacts/plan-s/s4e/run-4000`) on `evaluation-reach.toml` and the demo U3 checkpoint on
   `evaluation.toml`, each on `mujoco-cpu` and `mjwarp`; `es eval compare` per pair; MJWarp run twice
   (is it run-to-run reproducible? report, do not assume); a 20-iteration `Rollout` PPO smoke on
   `mjwarp`. The table goes into `evaluation-execution.md`'s new subsection.
6. fmt, clippy `-D warnings`, `cargo xtask check-scope docs/packets/M11/X1-backend-dispatch.md`,
   `cargo xtask verify-goldens`.

## acceptance

Oracles 1–6; the measured table with every number beside its `execution_hash`; the design-note
subsections (+ko).

## forbidden

`Box<dyn PhysicsBackend>` inside `Env`; a new trait; any change to `es-safety`; moving a committed
`mujoco-cpu` hash; making Newton "work" by suppressing its mapping report; an MJWarp result reported
as bitwise; PhysX code (I1).
