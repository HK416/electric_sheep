# W2-embedded-nostd — `no_std` Safety Plane and `no_std` embedded control loop

Spec: §9.6 (`es-runtime-embedded`: single static binary, no-std capable, zero heap allocation;
Observation IR evaluator + Learning IR pre/post + `PolicyRuntime` + the whole Safety Plane +
telemetry ring), §9.5 (sim/real identity — the Safety Plane is *the same code* on both sides,
which is what makes `deployment_hash` equality mean behavioural equality), §28.5 M3 W2 (no-std
Observation IR evaluator, Safety Plane, NPU backend — the NPU backend is a separate packet),
Appendix B.4 (`SafetyPlane` — no_std 가능, 힙 할당 0), §3.4 (determinism), §12.4 (unverified
performance reporting).

Design note: `docs/design/embedded-runtime.md` (read it first — §2 has the per-crate feature
table, §3 the fixed-size caps and the `from_config` sanitiser argument, §8 the `StaticPlan`
design this packet deliberately defers).

## context

```
crates/es-core/Cargo.toml                       (+ `std` / `alloc-count` features)
crates/es-core/src/lib.rs                       (+ cfg_attr no_std, module gating)
crates/es-core/src/time.rs                      (PhysTick stays; TickRate/SimTime gated)
crates/es-core/src/ring.rs                      (+ ArrayRing; RingBuffer gated)
crates/es-safety/Cargo.toml                     (+ `std` feature; libm; drops unused es-math)
crates/es-safety/src/lib.rs                     (+ cfg_attr no_std, exports)
crates/es-safety/src/ir_types.rs                (new)
crates/es-safety/src/config.rs                  (fixed-size config; from_ir behind `std`)
crates/es-safety/src/plane.rs                   (+ from_config; from_ir delegates)
crates/es-safety/src/types.rs                   (ExecutionMode via ir_types)
crates/es-safety/tests/nostd_core.rs            (new)
crates/es-runtime-embedded/Cargo.toml           (+ `std` feature)
crates/es-runtime-embedded/src/lib.rs           (+ cfg_attr no_std, layer split)
crates/es-runtime-embedded/src/core_rt.rs       (new: EmbeddedCore, TickRecord, Observe/InferFn)
crates/es-runtime-embedded/src/runtime.rs       (EmbeddedRuntime built on EmbeddedCore)
crates/es-runtime-embedded/tests/core_loop.rs   (new)
docs/design/embedded-runtime.md                 (new)
docs/packets/M3/W2-embedded-nostd.md            (this file)
```

## spec

1. `es-core` gains a default-on `std` feature. Without it the crate is `no_std` and exposes
   exactly `time::PhysTick` and a new `ring::ArrayRing<T, CAP>` (`[Option<T>; CAP]` inline, no
   `alloc`), with the same sequence numbering / overwrite order / `dropped` semantics as
   `ring::RingBuffer`. `TickRate`, `SimTime` and everything else stay behind `std`; the
   `alloc-count` feature implies `std`.
2. `es-safety` gains a default-on `std` feature. Without it the whole runtime still compiles:
   `SafetyPlane`, `validate`, `SafetyConfig`/`Envelope`/`Watchdogs`/`Fallback`/`SensorWatch`/
   `WorkspaceSpec`, `ActionChunk`, `SafeAction`, `EventSet`, `SafetyCounters`. Only the
   Deployment IR front door (`from_ir`, `SafetyConfigError`) is gated, because `es-ir` is a
   `std` crate.
3. `SafetyPlane::from_config(&SafetyConfig<NJ>) -> Self` is the `no_std` constructor: plain
   `Copy` data, no heap, infallible. `SafetyPlane::from_ir` builds a `SafetyConfig` and
   delegates to it, so there is exactly one construction path (INV-12). `validate`'s signature
   is unchanged (INV-13) — `src/ir_types.rs` re-exports `es-ir`'s `Micros`/`Limit`/
   `ActionSpace`/`ExecutionMode`/`HalfSpace` under `std` and defines identical local copies
   under `no_std`.
4. `Envelope::sanitize` runs once inside `from_config` and only ever **narrows** a degenerate
   config (empty or non-finite hard limit → `[0,0]`; soft limit outside its hard limit → the
   hard limit; non-finite or negative rate/velocity/acceleration/torque bound → `0`;
   non-positive `dt_s` → 1 ms). No repair widens anything (INV-12).
5. The `Vec`/`String` fields become fixed-size tables with hard caps: `MAX_HULL_FACES = 16`,
   `MAX_RETRACT_WAYPOINTS = 32`, `MAX_SENSORS = 8`, `SENSOR_NAME_CAP = 32`. Exceeding one is a
   `from_ir` error (`TooMany`, `SensorNameTooLong`), never a truncation — a sensor name is
   refused rather than shortened so two sensors cannot collapse into one dropout watchdog.
6. `es-runtime-embedded` gains a default-on `std` feature. Without it only
   `core_rt::EmbeddedCore<NJ, H, CAP>` compiles: Safety Plane + chunk cursor + seq +
   `ArrayRing<TickRecord, CAP>`. The observation and inference steps are function-pointer type
   aliases (`ObserveFn`, `InferFn<NJ, H>`), not traits — INV-17's seven extension points are
   untouched — with caller-owned buffers, and `InferFn` returns the number of action rows
   filled so `0` becomes the plane's `ChunkUnderrun` and the configured fallback (INV-13).
7. `EmbeddedRuntime` (the `std` layer) owns an `EmbeddedCore<NJ, H, 1024>` and calls
   `EmbeddedCore::step`, so the `std` and embedded paths cannot drift (§9.5). Its public
   behaviour is unchanged except `telemetry()`, which now returns
   `&ArrayRing<TickRecord, TELEMETRY_TICKS>`.
8. `config::sqrt` is `libm::sqrt` on **both** targets (`f64::sqrt` is `std`-only; IEEE-754
   square root is correctly rounded, so this is bit-identical to the hardware instruction and
   keeps one code path rather than two, §9.5, §3.5 tier 1).
9. The root `Cargo.toml` is untouched and no crate is added. `es-safety` drops the `es-math`
   dependency it never used.
10. **Deferred to a follow-up packet, designed in the note's §8:** the `no_std` Observation IR
    evaluator (`StaticPlan`). `es-compile/src/kernels.rs` is already pure over slices and was
    deliberately left untouched; gating it needs `es-math` and `es-ir` to be `no_std` first and
    needs edits to `es-compile/src/lib.rs` and `bundle.rs`, which W2 does not own. The NPU
    `PolicyRuntime` backend of the §28.5 W2 row is likewise a separate packet; `InferFn` is the
    seam it will use.

## oracle

```
cargo fmt --check
cargo clippy -p es-core -p es-safety -p es-runtime-embedded -p es-compile --all-targets -- -D warnings
cargo test -p es-core -p es-safety -p es-runtime-embedded -p es-compile
cargo build -p es-core -p es-safety -p es-runtime-embedded --no-default-features --target thumbv7em-none-eabihf
cargo clippy -p es-safety -p es-runtime-embedded --no-default-features --target thumbv7em-none-eabihf -- -D warnings
cargo xtask layering
cargo xtask context-budget
cargo check --workspace --all-targets
```

`thumbv7em-none-eabihf` (Cortex-M4F/M7, hard float, no `std`) is the proof target; substitute
`x86_64-unknown-none` where that target is not installed. `cargo build --no-default-features`
without a `--target` only proves the features resolve, not that `std` is absent.

## acceptance

1. Every command above exits 0.
2. `crates/es-safety/tests/nostd_core.rs` passes: `from_config` produces a clamping plane; a
   degenerate config is narrowed rather than honoured; `from_config` **plus** 32 `validate`
   calls run inside `es_core::alloc_count::assert_no_alloc`; the sensor table is bounded and
   refuses over-long names; the IR value types are `es-ir`'s under `std`.
3. `crates/es-runtime-embedded/tests/core_loop.rs` passes: the replan schedule and telemetry
   records are right; an `InferFn` returning `0` yields the fallback with `ChunkUnderrun`; the
   ring is bounded at `CAP` and counts `dropped`; construction + 64 ticks allocate nothing.
4. `es_core::ring::tests::array_ring_matches_ring_buffer` drives `ArrayRing` and `RingBuffer`
   in lock-step and they agree on `len`/`dropped`/`oldest_seq`/`iter_newest`/`drain_since`.
5. The pre-existing `es-safety` property and scenario suites and the `es-runtime-embedded`
   bundle tests pass unmodified — the `std` path did not move.
6. No new trait (INV-17), no `alloc` in a `no_std` path, no `HashMap`, English only.
7. `docs/design/embedded-runtime.md` carries the per-crate feature table, the caps table, the
   size table marked `unverified` with the reason, and the `StaticPlan` design plus its ordered
   blockers.

## forbidden

- `crates/es-eval/**`, `crates/es-compile/src/bundle.rs`, `crates/es-data/**`,
  `crates/es-splat/**`, `crates/es-editor/**`, `crates/es/**` — other in-flight packets.
- `crates/es-compile/src/{lib,kernels,exec,plan}.rs` — the `StaticPlan` packet's, once its
  blockers fall (note §8).
- `crates/es-ir/**`, `crates/es-ir-types/**`, `crates/es-math/**` — the `no_std`-ification of
  those is the blocker list, not this packet.
- The root `Cargo.toml`, and any new crate.
- Changing `SafetyPlane::validate`'s signature or return type (INV-13); adding any second
  plane constructor, an `enabled` flag or a `#[cfg(test)]` bypass (INV-12); making `es-safety`
  depend on `es-policy` (INV-11).
- Editing anything under `tests/golden/`.
