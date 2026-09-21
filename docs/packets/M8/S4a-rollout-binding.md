# M8 S4a — `es_native.Rollout`: a Python trainer steps our `Env` and the Safety Plane

Spec: §13.4 ("rollouts are `es-env`'s simulation batch domain; the Safety Plane is on"), §2.1
(Python is first-class on the learning path only; the binding is `es-py`, layer 11, which may
depend on `es-env` (9), `es-safety` (8), `es-compile` (7), `es-physics-backend` (4)), §4.2 (no
same-layer deps), §28.11 wave 1, INV-11/12/13, INV-17 (a pyclass is not an extension point).
Design notes: `docs/design/python-builder.md` (+ `.ko.md`) gains a section; the decisions are in
`docs/design/rl-continuation.md` sections 2–3. Precedent: `crates/es-py/src/pybind.rs` (four
builder pyclasses, `#[pyclass(unsendable)]`, feature `python`, `cdylib` + `rlib`).

## the question

`train_ppo.py` (S4b) needs to reset and step N envs, read the **Observation IR's** output ports
(not raw state), push every sampled action through `SafetyPlane::validate`, and read rewards,
dones and the plane's event bits — from Python, through the same code `es eval run` uses. **What
is the smallest binding that gives it that, and how is it shown to be the same rollout the Rust
side produces?**

## spec

* `es_native.Rollout(task_toml: str, observation_toml: str, deployment_toml: str, scene_xml: str,
  seed: int, n_envs: int)` — a `#[pyclass(unsendable)]` over a Rust struct `Rollout` (plain
  struct, `rlib`-visible, tested without Python) that owns `Env<MuJoCoCpuBackend>` built with
  `BatchDomains` sized `n_envs` (the same `TickRate`/`rate.control` derivation `es_eval::runner`
  and `DomainRunner` use), one `SafetyPlane` per env from the Deployment IR, one `CpuPlan` per
  env from the Observation IR (`PlanMode::Release`), and the plan input sources resolved once
  against the documents (reuse `es_env`'s plan/observation helpers if they are public; if the
  only implementation is private in `es-eval`, make that helper `pub` in `es-eval` and depend on
  it — do **not** write a second observation-capture implementation).
* Methods: `reset(envs: list[int] | None) -> None`; `observe() -> dict[str, list[float]]` (port
  name → the plan's output for all envs, row-major `[n_envs, dim]`, f32 as Python floats);
  `act(actions: list[float]) -> (executed: list[float], events: list[int], rewards: list[float],
  dones: list[bool])` — `actions` is `[n_envs × nu]` in actuator units; per env: `observe_state`,
  `heartbeat`, `validate(chunk of one row, age 0, now = step)` as `es_eval::runner::run_episode`
  does, then `Env::step` with the plane's outputs; `events` are `EventSet::bits()`; `dones` from
  `StepOutcome`; `Env::step`'s own reset on a terminal is kept and `begin_episode` is called on
  that env's plane (P-M7-R1 semantics); `model() -> dict` (`nq`, `nv`, `nu`, actuator and joint
  names, `ctrlrange`); `tick() -> int`; `metrics() -> dict` (the nine §12.4 fields, `None` where
  not measured).
* No chunk buffer, no latency: horizon 1, synchronous (`rl-continuation.md` section 3).
* Lists, not numpy: no new Rust dependency. `train_ppo.py` converts on its side.
* Build: `maturin develop --release --features python` in `crates/es-py` (the server venv gets
  `maturin` via `python -m pip install maturin`); `python/es/README.md` documents it.

## context

```
crates/es-py/**
crates/es-eval/src/runner.rs
crates/es-eval/src/lib.rs
python/es/selfcheck.py
python/es/README.md
python/es/README.ko.md
tests/golden/rollout/**
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/packets/M8/S4a-rollout-binding.md
docs/packets/M8/S4a-rollout-binding.ko.md
```

`es-py` (Cargo deps on `es-env`, `es-safety`, `es-eval`, `es-physics-backend`; `rollout.rs` for
the struct, `pybind.rs` for the class), `es-eval` **only** to make an existing private
observation-capture helper `pub` (no logic change), `selfcheck.py --env`, the golden (an
addition), the note, this packet.

## oracle

1. `cargo test -p es-py rollout_matches_es_eval_loop -- --ignored` (`ES_PYTHON`, MuJoCo): the
   Rust `Rollout` driven 100 steps by a fixed scripted ctrl on the committed documents produces
   `qpos` per step bitwise equal to a hand-rolled `Env` + `SafetyPlane` + `CpuPlan` loop written
   in the test the way `run_episode` does it, and identical event bits; writes
   `tests/golden/rollout/so101_100steps.json` when `ES_GENERATE_GOLDENS=1` (otherwise asserts
   against it).
2. `PYTHONPATH=python <venv>/python -m es.selfcheck --env` (after `maturin develop`): the same
   100 steps through the pyclass equal the golden bitwise; prints `RAN` / `SKIP <reason>`.
3. `cargo test -p es-py` without the `python` feature still builds and passes (`rlib` path).
4. `cargo xtask layering` (es-py's new deps are all lower layers), `cargo xtask ci`,
   `cargo xtask check-scope docs/packets/M8/S4a-rollout-binding.md`.

## acceptance

Oracles 1–4 (1 and 2 on the server: tree `~/Projects/es-s4a`, venv `~/venvs/es-lerobot-cuda` +
maturin; delete the tree when done). `python-builder.md` gains "The rollout binding" with the
method table and the no-numpy decision; Korean sibling.

## forbidden

A second observation-capture implementation; any plane state skipped (INV-12) or `validate`
signature change (INV-13); `es-safety` changes; a chunk buffer or latency model (S4b/S4c own
that question); numpy in the Rust crate; a trait (INV-17); `docs/ARCHITECTURE*.md`.
