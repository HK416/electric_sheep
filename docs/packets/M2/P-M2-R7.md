# P-M2-R7 — `nu != NJ` is an error, not a silent broadcast

Spec: §9 (Deployment IR / Safety Plane), §10.1 (the evaluation table), §10.4.
Design note: `docs/design/evaluation-execution.md` §2.5.
Invariants: INV-12, INV-13 (untouched — this changes which runs start, not what the plane
decides).
Review finding: `docs/reviews/M2.md` — Should-fix, `crates/es-eval/src/runner.rs:362`.

`safe.q[i.min(NJ - 1)]` drove every actuator beyond `NJ` with a copy of joint `NJ−1`, and
`joint_state` padded missing joints with `0.0`. A model whose actuator count disagreed with
the deployment's `NJ` therefore ran to completion and produced wrong numbers in the §10.1
table, which is worse than no table.

## context

```
crates/es-eval/src/lib.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R7.md
```

## spec

- **`es-eval`** — `EvalError::JointMismatch { nu, nq, nv, nj }`. The review asked for
  `{ nu, nj }`; `nq`/`nv` are in the same variant because removing the `0.0` padding in
  `joint_state` means the model must also carry at least `NJ` `qpos` and `qvel` entries, and
  one variant that names all four is smaller than two variants.
- `run_episode` checks `model.nu == NJ && model.nq >= NJ && model.nv >= NJ` before
  `env.reset`, i.e. at run start, and returns `JointMismatch` otherwise.
- The broadcast becomes `ctrl.copy_from_slice(&safe.q)` and `joint_state` copies `[..NJ]`
  straight out of `qpos` / `qvel` with no `unwrap_or(0.0)`.

## oracle

```
cargo test -p es-eval a_model_with_more_actuators_than_joints_is_refused
```

A backend that loads the fixture model with one extra actuator (`FakeBackend::wide`, `nu = 3`)
against the deployment's `NJ = 2` must make `Evaluation::run` return
`EvalError::JointMismatch { nu: 3, nj: 2, .. }`.

## acceptance

- The 3-actuator run errors; every existing es-eval test (`nu == NJ == 2`) stays green.
- No `min`, no `unwrap_or(0.0)` and no zero padding remains on the action or state path in
  `runner.rs`.
- `cargo test -p es-eval` green.

## forbidden

- Widening or disabling any part of the Safety Plane to accommodate a mismatch (INV-12), and
  any change to `SafetyPlane::validate`'s signature (INV-13).
- Resizing the chunk or the envelope to the model instead of refusing: `NJ` is the
  deployment's contract, and `SafetyPlane::from_ir` already rejects a `deploy`/`NJ`
  disagreement.
- `crates/es-env`, `crates/es-policy`, `crates/es-telemetry`, `crates/es`,
  `crates/es-compile/src/budget.rs`.
