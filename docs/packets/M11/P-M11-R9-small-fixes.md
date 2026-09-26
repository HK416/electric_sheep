# M11 R9 — five small fixes from the M11 review

Spec: §7.2 / INV-14 (a drawn fov moves the episode's intrinsics), §3.5 (determinism tiers),
§1.2 (check-scope), §1.4. Found by `docs/reviews/M11.md` (S-4 and the nits). Each item is
independent and gets its own commit. Type B.

## the question

**Can each of the five be fixed without moving a committed hash, golden, `.estraj`, frame or
`report.json` byte?**

## spec

1. **One focal computation (R2 nit).** `RenderOverrides::image_spec`
   (`crates/es-env/src/randomize.rs`) records `fx_f64 * focal`, while `drawn_frame`
   (`crates/es-env/src/render.rs`) projects with `fx_f32 * (focal as f32)`. The two differ by
   one f32 ULP on the collector sidecar. Make both go through one function, so the recorded
   intrinsics are the ones the frame was projected with. The rendered frame must not move: the
   function computes what `drawn_frame` computes today, and `image_spec` takes its value from
   there. If a committed sidecar or `layout.json` moves, stop and report it.
2. **`FrameSink` sees the render overrides (R2 nit).** The collector recomputes each episode's
   render draws from `RandomizationPlan::apply_render`, because es-data's `FrameSink` only gets
   `(&ModelInfo, &StateView)`. Pass the env's current `RenderOverrides` through the sink, and
   delete the recomputation. The frames must stay bitwise equal (R2's parity oracle).
3. **The run says which determinism tier produced it (S-4).** `es eval run`, `es loop collect`
   and `es train` print one stdout line `determinism tier: <tier> (backend <name>)`. The tier is
   the backend's §3.5 tier, taken from the capability the backend already reports (MJWarp = tier 2).
   `report.json` does not change. A new field would move every committed report.
4. **`check-scope --base <rev>` (M10 nit).** `cargo xtask check-scope <packet.md> --base <rev>`
   checks `git diff --name-only <rev>...HEAD` plus the working tree. Without `--base`, the
   behaviour is unchanged. Add a unit test in `xtask/src/scope.rs`.
5. **Two flaky tests.**
   * `eval_telemetry_publishes_every_tick_in_order` loses a temp `es-policy-*.safetensors` under
     full parallel runs. Find the shared temp name (a fixed name or a PID-only name in a shared
     dir) and make it unique per call. If the name is in production code, fixing it there is in
     scope.
   * `es-telemetry`'s `a_connection_past_max_clients_is_refused` failed once with
     `ConnectionReset`. A reset from a refused connection is also a refusal: accept it,
     together with the refusal the test expects today.

## context

```
crates/es-env/src/randomize.rs
crates/es-env/src/render.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es-data/src/collect.rs
crates/es-data/src/lib.rs
crates/es-data/tests/**
crates/es/src/**
crates/es/tests/cli.rs
crates/es-py/src/**
crates/es-eval/src/**
crates/es-telemetry/src/transport.rs
crates/es-policy/src/**
xtask/src/scope.rs
xtask/src/main.rs
docs/packets/M11/P-M11-R9-small-fixes.md
docs/packets/M11/P-M11-R9-small-fixes.ko.md
```

## oracle

1. Item 1: a test in `crates/es-env/tests/` shows that the `ImageSpec` the collector records and
   the one `drawn_frame` projects with are bitwise equal for a drawn fov. R2's frame parity tests
   still pass.
2. Item 2: R2's collector == evaluator == `Rollout` frame parity tests pass, and the collector no
   longer calls `apply_render`.
3. Item 3: a cli test checks the stdout line for `mujoco-cpu` (tier 1). The committed
   `report.json` fixtures are byte-identical.
4. Item 4: `cargo test -p xtask`. Running `cargo xtask check-scope docs/packets/M11/P-M11-R9-small-fixes.md --base main`
   on this branch passes.
5. Item 5: each test passes 20 times in a row under `cargo test -p es --test cli` with default
   parallelism, and the telemetry test likewise.
6. `cargo xtask ci`, verify-goldens with 0 modifications, fmt, clippy `-D warnings`.

## acceptance

Oracles 1–6. The report gives one line per item, with the commit.

## forbidden

Changing `report.json`'s schema, any committed golden, fixture or hash; `es-safety`; relaxing a
test's assertion beyond what item 5 names.
