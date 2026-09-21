# M9 T1 — `JointDelta`: an increment the runtime integrates in one place, and a plane that still sees absolute targets

Spec: §8.5 (the sentence added 2026-09-22: `JointDelta` / `EeDelta` are increments over the
current target; `es-env` integrates; the plane validates absolute targets; the integrator resets
to the plane's seeded pose at the episode boundary), §9.3 (the seed), §9.4, §13.4, §28.12 rule
1–2 and wave 1, INV-11..13 (`es-safety` unchanged), INV-17. Depends on **P-M8-R6** (the `es-ir`
split). Design notes: `docs/design/batch-domains.md` (the runner / chunk buffer sections),
`docs/design/evaluation-execution.md` 2.x, `docs/design/python-builder.md` (the rollout
binding) — each gains the one integration rule; new short section in `rl-continuation.md`.

## the question

`ActionSpace::EeDelta` exists in two enums (`es-ir-types::expr` and `es-ir::deployment`) with no
execution semantics, and `JointDelta` does not exist. **Can a Deployment IR declare a joint-space
increment, can one `es-env` function turn it into the absolute target every consumer (collector,
evaluator, `Rollout`) hands the plane, and does every `JointPosition` document, hash, trajectory
and golden stay byte-identical?**

## spec

* `ActionSpace::JointDelta` added to both enums (the duplication is a finding for the review,
  not this packet's to fix); absent / `JointPosition` is today's canonical form — pin the
  committed `deployment_hash`es before touching anything.
* **One function**, `es_env::control::absolute_target(space, prev: &[f64; NJ], row: &[f64; NJ],
  out: &mut [f64; NJ])` (name and module the agent's, the rule not): for `JointPosition` it
  copies; for `JointDelta` it adds `row` to `prev`. `prev` is the integrator state, owned beside
  the chunk buffer / plane feed (per env), seeded at every episode boundary from the measured
  joint state the plane seeds from (`SafetyPlane::observe_state`'s first call after
  `begin_episode` — the same numbers, read through the same `joint_state` helper), and updated
  from the plane's **executed** output (`SafeAction::q`), never from the raw row — so a clamped
  increment does not accumulate into an unreachable target.
* Consumers: `DomainRunner` (collection), `es_eval::runner::run_episode`, `es_py::Rollout::act`
  call it between the chunk row and `validate`; the plane's inputs, outputs and signature are
  unchanged (INV-13).
* `Normalizer{Inverse}` statistics for a delta policy are in increment units (rad per control
  tick); the Deployment IR's `action.space` says which; `XIR` cross-check: a `JointDelta`
  deployment with an action port whose unit is not an increment unit is refused by name.
* Datasets: `es loop collect` records the *executed absolute* command as today (the dataset
  schema does not change); `action_source` unchanged.

## context

```
crates/es-ir-types/src/expr.rs
crates/es-ir/src/deployment.rs
crates/es-ir/src/cross.rs
crates/es-ir/tests/**
crates/es-env/src/control.rs
crates/es-env/src/chunk_buffer.rs
crates/es-env/src/domains.rs
crates/es-env/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-py/src/rollout.rs
crates/es-py/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/deployment-reach-delta.toml
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M9/T1-delta-actions.md
docs/packets/M9/T1-delta-actions.ko.md
```

## oracle

1. `cargo test -p es-ir committed_deployment_hashes_are_unmoved_by_joint_delta` — the pinned
   hashes; `JointDelta` moves a hash; the unit cross-check refuses by name.
2. `cargo test -p es-env delta_integrates_to_the_absolute_target` — on the fixture backend, a
   scripted absolute sequence and its first-difference sequence driven through the two spaces
   produce **bitwise** the same executed commands, plane events and `qpos`; a clamped increment
   integrates from the executed value, not the raw one (a test where the envelope bites).
3. `cargo test -p es-eval demo_trajectories_are_unmoved` — the committed demo documents
   (`JointPosition`) run to byte-identical `.estraj` / `events.json` on the fixture backend
   before and after (pin the bytes' blake3 first).
4. `cargo test -p es-py rollout_integrates_delta -- --ignored` (server, MuJoCo): `Rollout` over
   `deployment-reach-delta.toml` with increments equals a `Rollout` over `deployment-reach.toml`
   fed the integrated absolutes, bitwise.
5. `git diff --stat main -- crates/es-safety` is empty; `cargo xtask ci`; `check-scope`.

## acceptance

Oracles 1–5; the three notes' one-rule sentences and their Korean siblings.

## forbidden

Any `es-safety` change (INV-11..13); a second integrator (one function, three callers); changing
the dataset schema or any `JointPosition` behaviour; an IK for `EeDelta` (out of this plan);
`docs/ARCHITECTURE*.md`; `tests/golden/**`; INV-17.
