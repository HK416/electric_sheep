# W3 — Observation IR → GPU lowering (part 2)

Spec: spec 7.6, spec 7.7, spec 11.4, spec 11.5, spec 2.3, spec 3.4, spec 3.5, spec 26.2,
spec 28.3. Design note: `docs/design/observation-lowering.md` §11 (review class C — the
numerics are the artifact). Part 1 is `docs/packets/M1/W3-observation-cpu-ref.md`, whose
`CpuPlan` is this packet's oracle.

## context

```
docs/design/observation-lowering.md      (§11 filled in; .ko.md mirror too)
docs/packets/M1/W3-observation-gpu.md    (new)
crates/es-compile/Cargo.toml             (adds `es-gpu`: layer 2 under layer 7, rule 1)
crates/es-compile/src/lib.rs             (adds `pub mod gpu;` + re-exports)
crates/es-compile/src/gpu/mod.rs         (new — the lowering, no device)
crates/es-compile/src/gpu/plan.rs        (new — compile, pipelines, compiler_hash)
crates/es-compile/src/gpu/exec.rs        (new — upload, dispatch, download)
crates/es-compile/slang/observation.slang (new — one entry point per kernel id)
crates/es-compile/tests/observation_gpu.rs (new)
crates/es-compile/src/exec.rs            (`read_input`, `width`: private -> pub(crate))
```

## forbidden

`budget.rs`, `bundle.rs`, `kernels.rs`, `plan.rs` — owned by part 1 and by the bundle packet.
`exec.rs` changes are limited to the two `pub(crate)` visibility widenings above: the GPU path
must reject a mismatched input with the *same* `ExecError` the CPU path does, and a second
copy of that check is a second contract. `KERNEL_IDS` is not appended to, reordered or
renamed — a new numeric behaviour is a new id and that is part 1's table. `tests/golden/**`
is CI read-only. `es-render`, `es-usd`, `es-script`, `es-data`, `es-eval`, `es-py`, `crates/es`
belong to neighbouring packets.

## spec

- `GpuPlan::compile(&Gpu, &ObservationIr, PlanMode) -> Result<GpuPlan, Vec<Diagnostic>>` calls
  `CpuPlan::compile` and **mirrors** it — the same topological order, the same buffer table,
  the same kernel per node. It is not a second compiler; the only thing it adds is where the
  bytes live and which pipeline runs. Anything the device or `slangc` refuses is `COMPILE-006`.
- One device arena, not a buffer per plan buffer. Five storage bindings serve every kernel:
  `arena` (f32 intermediates, then the f32 plan inputs), `words` (u8 plan inputs, then f16 /
  bf16 output bit patterns), `aux` (the sRGB LUT, then each `Normalize` step's mean/std),
  `rings`, `state` (per-window `cursor`/`pushed`). One descriptor layout covers the plan and
  every buffer offset can therefore be a Slang define.
- One `ComputePipeline` per `(kernel id, Slang entry, defines)`. `KERNEL_IDS` stays the source
  of truth for what a kernel *is*; the defines (dtype, channels, sizes, offsets) are the
  specialisation of spec 2.3. `GpuPlan::pipeline_plan` exposes that list **without a device**,
  so CI with no Vulkan still gates the decision.
- `GpuPlan::run(&mut self, &BTreeMap<String, TensorRef>) -> Result<Outputs, GpuRunError>`:
  upload, record every dispatch on the single queue with a barrier between them, one submit,
  download. `&mut self` because the rings are a stream (spec 7.5 layer 1).
- `GpuPlan::reset()` is the device mirror of `CpuPlan::reset`: zero the rings, zero the
  cursors, i.e. exactly the state `compile` leaves. `es-eval` calls it after every `env.reset`.
- `GpuPlan::compiler_hash()` = the CPU plan's hash plus every pipeline's SPIR-V content hash,
  so editing a `.slang` file moves the `compiler` slot of `execution_hash` even when no kernel
  id did (spec 3.4 item 7, spec 11.2).
- Determinism (spec 3.4): execution modes from
  `Capabilities::deterministic_execution_modes()` — a compile input read off the capability
  query, never a device setting — plus the `NoContraction` `es_gpu::apply_exec_modes` patches
  in. Single queue, one submission per run, no atomics, no shared memory, no subgroup
  operations, no dependence on the workgroup count. `es_math::approx` (`approx.slang`) is the
  only transcendental.

## oracle

```
cargo test -p es-compile -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget
```

Every device test prints `SKIP <reason>` and passes where there is no Vulkan device or no
`slangc` (spec 1.4), and prints `RAN` with the device name where there is. The CPU/GPU
equivalence numbers are printed, not only asserted, so a regression reads as a number.

## acceptance

- Every golden in `tests/golden/observation/` is reproduced by a `GpuPlan`, under the *same*
  comparison `observation_cpu.rs` uses: byte-for-byte, except `srgb_to_linear_lut256` at the
  7 ULP its sidecar declares. Measured: `dequantize`, `resize_bilinear`, `crop`, `normalize`,
  `history_window` bit-equal; the LUT at 7 ULP — which is the CPU kernel's own distance from
  the f64 reference formula, not a GPU error on top of it, because the GPU indexes the CPU's
  uploaded table bytes.
- 50 pseudo-random `(source, target)` size pairs up to 17×17: `CpuPlan::run` vs `GpuPlan::run`
  on the same graph, **bit-for-bit**. Measured 50/50, worst 0 ULP. A mismatch prints the size
  and the ULP distance rather than only failing.
- Two `run`s of one plan over one input are bit-identical (spec 3.5 tier 1).
- `reset(); run(x)` equals a freshly compiled plan's first `run(x)`, and equals the CPU plan's,
  on a fixture that is history-sensitive (asserted, or the test proves nothing).
- CPU vs GPU bit equality on the kernels no golden covers: `resize_nearest`, the elementwise
  `srgb_to_linear` (0 ULP), `concat`, `stack`, `normalize_range`, `cast_f32_to_f16`,
  `cast_f32_to_bf16`.
- Without a GPU: the pipeline list of a chain that uses each kernel once is one pipeline per
  distinct kernel id, and two resizes to the same size share a pipeline while two to different
  sizes do not.

## known limits

- **No fusion.** Spec 11.4's element-wise fusion is not implemented; every node is its own
  dispatch in both plan modes, as on the CPU path. Release-mode fusion and arena aliasing are
  their own packet, which is what `PlanMode` reaching `compiler_hash` protects.
- **Denormals.** The deterministic execution modes flush denormals to zero on the device; the
  CPU kernels do not. A denormal intermediate is the one place the two can disagree.
- `f16`/`bf16` results are written as bit patterns into a `uint` region: the device is not
  opened with 16-bit storage. Widening a narrowed buffer back to f32 on the device is
  unsupported — the arena keeps the f32, so nothing needs it yet.
- `slangc`'s scratch files are named `{pid}-{invocations}` and each `SlangCompiler` starts
  that counter at 0, so two threads of one process compiling the same cache key race. The GPU
  tests serialise around it; the fix belongs in `es-gpu`.
