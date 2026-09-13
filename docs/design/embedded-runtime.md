# Embedded runtime — the `no_std` deployment path

Spec: §9.6 (`es-runtime-embedded`: single static binary, no-std capable, zero heap; contains
Observation IR evaluator + Learning IR pre/post + `PolicyRuntime` + the whole Safety Plane +
telemetry ring), §9.5 (sim/real identity: the Safety Plane, the chunker and the preprocessing
are *the same code* on both sides, which is what makes `deployment_hash` equality mean
behavioural equality), §28.5 M3 W2, Appendix B.4 (`es-safety/src/lib.rs` — `no_std` 가능, 힙
할당 0).

Companion notes: `docs/design/safety-plane.md` (the `validate` algorithm),
`docs/design/policy-bundle.md` (`policy.esb`), `docs/design/observation-lowering.md` (the CPU
kernels §8 refers to).

## 1. What "no-std capable" has to mean here

Spec §9.5 is the constraint that shapes everything else. If the embedded build were a *second
implementation* of the Safety Plane, `deployment_hash` equality would prove nothing. So the
rule for this packet was: **`no_std` may remove a front door, never a code path.**

Concretely, for every crate on the deployment path:

- the runtime — `validate`, the clamp stages, the watchdogs, the fallbacks, the chunk cursor,
  the telemetry ring — is compiled from the same source in both builds, with no `cfg` inside
  it;
- what `--no-default-features` removes is only the *authoring-side* front door: the Deployment
  IR reader, the `.esb` bundle reader, the error types that carry a `String`, and the
  `dyn PolicyRuntime` trait object.

That is why `SafetyPlane::from_ir` was turned into a thin wrapper rather than duplicated:
there is one construction path (`from_config`) and `from_ir` funnels into it, so INV-12 (no
code path disables the plane) is a structural property rather than a review promise.

## 2. The split, crate by crate

Each crate gained a `std` feature, on by default. The workspace `Cargo.toml` is untouched —
features are per-crate, and a `cargo build` of the workspace enables `std` everywhere through
normal feature unification.

| crate | `no_std` surface | behind `std` |
|---|---|---|
| `es-core` | `time::PhysTick`, `ring::ArrayRing<T, CAP>` | `ecs`, `job`, `arena`, `pool`, `failure`, `id`, `sizing`, `alloc_count`, `TickRate`, `SimTime`, `ring::RingBuffer` |
| `es-safety` | `SafetyPlane` (incl. `from_config`, `validate`), `SafetyConfig`/`Envelope`/`Watchdogs`/`Fallback`/`SensorWatch`/`WorkspaceSpec`, `ActionChunk`, `SafeAction`, `EventSet`, `SafetyCounters`, `ir_types` | `SafetyPlane::from_ir`, `Envelope/Watchdogs/Fallback::from_ir`, `SafetyConfigError` |
| `es-runtime-embedded` | `core_rt::EmbeddedCore` + `TickRecord` + `ObserveFn`/`InferFn` | `EmbeddedRuntime`, `hardware_capability`, `RuntimeError` |

Dependency consequences: `es-safety` drops `es-ir` and `thiserror` (both `optional`, pulled in
by `std`) and dropped `es-math` outright, which it never used.  `es-runtime-embedded` drops
`es-ir`, `es-compile`, `es-policy`, `blake3` and `thiserror`. `serde` is taken with
`default-features = false` plus `std = ["serde/std"]`.

> Cargo wrinkle worth knowing: a crate-level `default-features = false` is **ignored** when the
> `[workspace.dependencies]` entry does not itself set `default-features`, and cargo only warns.
> `es-core`/`es-safety`/`es-runtime-embedded` therefore name `serde` and their sibling `es-*`
> crates by version/path instead of `workspace = true`. Nothing else changes: features are
> additive, so every other crate still gets `serde/std`.

## 3. `es-safety`: what had to change to reach zero heap

The M1 plane was already allocation-free *per tick*, but it allocated at construction and
borrowed three `std` shapes from the IR. Four of those were structural.

1. **The four IR value types the hot path carries** — `Micros`, `Limit`, `ActionSpace`,
   `ExecutionMode` (plus `HalfSpace`). `src/ir_types.rs` re-exports `es-ir`'s under `std` and
   defines identical local copies under `no_std`. Re-exporting (rather than converting) is
   what keeps `validate`'s signature byte-identical for existing callers, which INV-13
   requires; `tests/nostd_core.rs::ir_value_types_are_the_es_ir_types_under_std` pins it by
   assigning `es_ir::deployment::X` into an `es_safety::X` binding.
   **ponytail ceiling:** two definitions of four trivial types, kept in sync by review. Upgrade
   path: move them into `es-ir-types` and delete the shim. Deferred because `es-ir` is out of
   W2's scope.
2. **`Workspace::ConvexHull { faces: Vec<HalfSpace> }`** → `WorkspaceSpec::ConvexHull { faces:
   [HalfSpace; MAX_HULL_FACES], n_faces }`.
3. **`Watchdogs::sensors: Vec<SensorWatch>` with a `String` name** → a `[SensorWatch;
   MAX_SENSORS]` table whose names are `[u8; SENSOR_NAME_CAP]`. `SensorWatch::new` **refuses**
   a name that does not fit rather than truncating: two sensors sharing a 32-byte prefix must
   not collapse into one watchdog and silently weaken the dropout check.
4. **`Fallback::trajectory: Vec<[f64; NJ]>`** → `[[f64; NJ]; MAX_RETRACT_WAYPOINTS]` + length.
   A `RetractToHome` with no waypoints is now unconstructible (`Fallback::stationary`
   downgrades it to `HoldPosition`, which is safe rather than absent), and `run_fallback` holds
   instead of indexing an empty slice.

| cap | value | what exceeding it does |
|---|---|---|
| `MAX_HULL_FACES` | 16 | `from_ir` → `SafetyConfigError::TooMany` |
| `MAX_RETRACT_WAYPOINTS` | 32 | `from_ir` → `SafetyConfigError::TooMany` |
| `MAX_SENSORS` | 8 | `from_ir` → `SafetyConfigError::TooMany` |
| `SENSOR_NAME_CAP` | 32 bytes | `from_ir` → `SafetyConfigError::SensorNameTooLong` |

These are refusals, not truncations: a deployment the embedded runtime cannot represent must
fail to load, never load with a quieter envelope (INV-12).

### 3.1 `from_config` is infallible, and why that is safe

`from_config` takes plain `Copy` data with no validator behind it, so it cannot return a
`Result` without giving the embedded caller something to ignore — the same argument §B.4 makes
for `validate`. Instead `Envelope::sanitize` runs once at construction and **narrows**:

- a hard limit that is empty or non-finite becomes `[0, 0]`;
- a soft limit that is not a finite sub-interval of its hard limit becomes the hard limit;
- a non-finite or negative `vel_max`/`acc_max`/`tau_max`/`d1_max`/`d2_max` becomes `0` (the
  tightest value, never the loosest);
- a non-positive `dt_s` becomes 1 ms; `period_us` and `execute_chunk` become at least 1.

Every repair tightens. The §9.1 way to make room is still a wider *hard* limit, which is the
INV-12 rule ("widen the envelope") spelled out in data.

### 3.2 `sqrt`

`f64::sqrt` lives in `std`. Rather than `cfg` two square roots, `config::sqrt` calls `libm` on
**both** targets. IEEE-754 square root is correctly rounded, so this is bit-identical to the
hardware instruction the `std` build used to emit — one code path (§9.5) instead of two that
happen to agree. The only caller is the cylinder-workspace projection. `libm` was already in
`Cargo.lock` and is `no_std` by construction.

## 4. `es-core`: `PhysTick` and `ArrayRing`

Only two things are needed below `es-safety`, so only two things are outside `std`.

`TickRate`/`SimTime` stayed behind `std` because `TickRate::rational` reports `es_core::Error`
(`thiserror` + `String`); the plane stores the derived `dt_s`/`period_us` and never needs the
rate itself.

`ring::RingBuffer` owns a `Vec`, so the telemetry ring got a sibling rather than a rewrite:
`ArrayRing<T, CAP>` holds `[Option<T>; CAP]` inline, with the same sequence numbering,
overwrite order and `dropped` count. `ring::tests::array_ring_matches_ring_buffer` drives both
in lock-step for 7 pushes over a capacity-3 ring, so "same behaviour" is a test rather than a
claim. `CAP` is a const generic because on a microcontroller the whole runtime is one `static`.

## 5. `es-runtime-embedded`: one loop, two front ends

`core_rt::EmbeddedCore<NJ, H, CAP>` is the loop:

```text
sensors -> [observe] -> obs -> [infer] -> ActionChunk -> SafetyPlane -> SafeAction
                                                             |
                                                         TickRecord -> ArrayRing<_, CAP>
```

- `EmbeddedCore::step(chunk: Option<ActionChunk>, now, obs_age) -> SafeAction` is the loop
  proper: accept a new chunk iff one was passed (bumping `seq` once per *policy invocation*,
  not per tick, per §8.6), validate, push the telemetry record, advance the cursor.
- `EmbeddedCore::tick(..., observe: ObserveFn, infer: InferFn<NJ, H>)` is the `no_std` entry
  point: it drives the two plug-ins and calls `step`.
- `EmbeddedRuntime::tick` is the `std` entry point: it drives `CpuPlan::run` and
  `dyn PolicyRuntime::infer` itself and calls the same `step`.

The plug-ins are **function pointers**, not traits:

```rust
pub type ObserveFn = fn(sensors: &[f32], obs: &mut [f32]);
pub type InferFn<const NJ: usize, const H: usize> =
    fn(obs: &[f32], actions: &mut [[f64; NJ]; H]) -> usize;
```

No new trait (INV-17 lists seven and `PolicyRuntime` is already one of them), no vtable, no
`Box`, and both buffers belong to the caller so the step cannot allocate. `InferFn` returns the
number of rows it filled: `0` is how an ONNX/Vulkan/NPU failure becomes the plane's
`ChunkUnderrun` and the configured fallback instead of an error a caller could drop (INV-13).
The ONNX / Vulkan / NPU `PolicyRuntime` implementations plug in at the `std` layer — on the
embedded target the integrator supplies the two `fn`s, which is also how a vendor NPU SDK
(C ABI, no Rust trait) gets in.

`EmbeddedRuntime` now *owns* an `EmbeddedCore<NJ, H, 1024>` instead of its own plane, chunk,
cursor, seq and ring, so the std path and the embedded path cannot drift. `EmbeddedCore` is
deliberately not `Clone`: a cloned control loop would fork the plane's state and counters.

## 6. Proof

```
cargo build -p es-safety -p es-runtime-embedded --no-default-features \
    --target thumbv7em-none-eabihf          # Cortex-M4F/M7, hard float, no std
cargo build -p es-core   --no-default-features --target thumbv7em-none-eabihf
cargo clippy -p es-safety -p es-runtime-embedded --no-default-features \
    --target thumbv7em-none-eabihf -- -D warnings
```

all pass. The bare-metal target has no test harness, so behaviour is pinned from `std` tests
that exercise only `no_std` code:

- `crates/es-safety/tests/nostd_core.rs` — `from_config` clamps like a plane should; a
  degenerate config is narrowed, not honoured (INV-12); `from_config` **plus** 32 `validate`
  calls run inside `assert_no_alloc` (spec §9.6's zero-heap claim now covers construction, not
  just the hot path); the sensor table is bounded and never truncates; the IR value types are
  `es-ir`'s under `std`.
- `crates/es-runtime-embedded/tests/core_loop.rs` — the replan schedule and telemetry
  bookkeeping; an `InferFn` returning `0` becomes the fallback; the ring is bounded; the whole
  loop (construction + 64 ticks) runs inside `assert_no_alloc`.
- the pre-existing `es-safety` property/scenario suites and the `es-runtime-embedded` bundle
  tests are unchanged and still green, which is what says the `std` path did not move.

### Size

`.text` of a *linked* image cannot be measured yet: a bare-metal binary needs an entry point,
a linker script and a panic handler, none of which this packet ships (nothing here is a `[[bin]]`).
`llvm-size`/`cargo size` are also not installed in this environment. Release `rlib` sizes for
`thumbv7em-none-eabihf` are recorded instead, as a proxy only — they include metadata and
bitcode, not just code:

| artifact (release, thumbv7em-none-eabihf) | rlib bytes | `.text` |
|---|---|---|
| `libes_safety.rlib` | 441,578 | unverified |
| `libes_runtime_embedded.rlib` | 35,872 | unverified |

**Target / Status: unverified** (spec §12.4). A follow-up packet that adds an example `[[bin]]`
with a linker script and `cortex-m-rt` can report `.text`/`.rodata`/`.bss` properly; the `.bss`
number is the interesting one, since `EmbeddedCore<NJ, H, CAP>` puts the whole telemetry ring
(`CAP * size_of::<Option<TickRecord>>()`) and the whole `SafetyConfig` in static memory.

## 7. What this packet did **not** do

- **NPU `PolicyRuntime` backend** (the third item of the §28.5 W2 row). Out of scope here; the
  `InferFn` slot above is the seam it will use.
- **A linked bare-metal image**, hence no real size or stack-depth numbers (see above).
- **`es-ir` / `es-math` `no_std`**, which is what blocks §8.

## 8. Next packet: the `no_std` Observation IR evaluator (`StaticPlan`)

The §28.5 W2 row also asks for a no-std Observation IR evaluator. `es-compile/src/kernels.rs`
is *already* written the way it needs to be — every function is pure over `&[u8]`/`&[f32]`
slices, single-threaded, in a fixed loop and operator order, with the one transcendental going
through `es_math::approx` (spec §3.4 `DET-010`); the only `Vec` in the file is inside
`#[cfg(test)]`. It was therefore left untouched: gating it is mechanical, but it cannot be done
from inside W2's scope.

**Blockers, in the order they have to fall:**

1. `es-math` (layer 0) is not `no_std`: `simd.rs` uses `std::is_x86_feature_detected!`.
   `kernels.rs` needs `es_math::approx` for the sRGB power curve, so `approx` must become
   `no_std` with the runtime feature dispatch behind `std`.
2. `es-ir` (layer 6) is not `no_std`, and `kernels.rs` needs `es_ir::Rect` while `exec.rs`
   needs `ElemType` and `ResizeFilter`. These are plain value types; the right home is
   `es-ir-types`, which would also let `es-safety` delete `src/ir_types.rs` (§3.1 above).
3. `es-compile`'s own module set: `bundle.rs` (`.esb` reader, `blake3`, `std::io`) and
   `plan.rs`'s `CpuPlan::compile` (`BTreeMap`, `Vec<Step>`, diagnostics with `String`) must go
   behind `std`, which means editing `es-compile/src/lib.rs` and `bundle.rs` — both owned by
   other in-flight packets at the time of writing. The unused `es-assets` dependency should go
   at the same time.

**The design the next packet should implement — `StaticPlan`.** `CpuPlan` is a compile-time
object: `compile` resolves shapes, allocates buffer ids and emits `Vec<Step>`, then `run`
interprets it with a fresh arena and a `BTreeMap` of borrowed inputs per call. Splitting it in
two along the line that already exists:

```rust
// no_std: everything pre-resolved, nothing owned.
pub struct StaticPlan<'a> {
    steps: &'a [Step],            // Op + BufferId operands, already type/shape checked
    buffers: &'a [BufferDesc],    // dtype, element count, byte offset into one arena
    inputs: &'a [(BufferId, u32)],  // plan input slot -> buffer, in a fixed order
    outputs: &'a [(BufferId, u32)],
    arena_bytes: usize,           // the caller's single scratch allocation
}

impl StaticPlan<'_> {
    /// `arena.len() >= self.arena_bytes`; inputs are written into their slots by the caller.
    pub fn run(&self, arena: &mut [u8]) -> Result<(), StaticPlanError>;
}
```

Three properties make it worth doing this way rather than writing a second evaluator:

- **one arena, one caller allocation.** Every buffer is a `(offset, len)` into `arena`, decided
  at compile time, so `run` performs no allocation and no lookup — the §9.6 zero-heap claim
  extends to the observation pipeline.
- **inputs by index, not by name.** `CpuPlan::run` keys inputs by `String`; the static form
  numbers the slots in the plan's own order, which also removes the last `BTreeMap` from the
  tick.
- **the same `Step`s.** `CpuPlan::compile` (still `std`, still in the bundle loader) *emits*
  the `&[Step]`/`&[BufferDesc]` a `StaticPlan` borrows, so a `policy.esb` can be transcoded
  ahead of time into a `static` table for a flash image, and the bytes produced on the robot
  are the bytes the goldens pin — the §9.5 identity claim for the observation path.

Oracle for that packet: the existing `tests/golden/observation/` tensors, run through both
`CpuPlan::run` and `StaticPlan::run`, compared bitwise.
