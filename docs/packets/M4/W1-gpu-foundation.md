# W1-gpu-foundation — `es-gpu`: device, capabilities, Slang, buffers, pipelines

Spec: §2.3 (Slang → SPIR-V, offline + content-hash cache), §3.3–§3.4 (precision, the
eight-item deterministic execution contract), §3.5 (determinism tiers), §4.2 rule 3 (only
`es-gpu` knows Vulkan below layer 3), §11.4–§11.6 (GPU lowering, plans, capability check),
§21 (`TensorTransport`), §22 (multi-GPU), §28.7 gate 3. Design:
`docs/design/gpu-foundation.md`.

## context

```
crates/es-gpu/**
Cargo.toml                      (add `ash` and `gpu-allocator` under [workspace.dependencies] only)
docs/api-notes/ash.md
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/W1-gpu-foundation.md
```

## spec

- `Gpu::open(GpuOptions { prefer_discrete, validation })` opens **one** device with **one**
  compute queue (§3.4 forbids multi-queue in deterministic mode; the API offers no second
  queue). `Gpu::none_available()` is the CI predicate.
- `Capabilities` is filled by *query* (`vkGetPhysicalDeviceProperties2`): every field of
  `VkPhysicalDeviceFloatControlsProperties`, subgroup size and operation classes,
  `shaderFloat64`/`shaderInt64`, `VK_KHR_cooperative_matrix`, workgroup limits, memory heaps,
  device UUID (§22 forbids identifying a device by enumeration order). Nothing writes a
  float control — §3.4 and Appendix C are explicit that it is not a setting.
- `Capabilities::determinism_tier()` → `Bitwise` when f32 denorm-flush, RTE and
  signed-zero/inf/nan-preserve are all supported and independently settable for 32-bit and a
  subgroup size is reported; `CrossBackend` otherwise.
- `Capabilities::deterministic_execution_modes()` → `ExecModes`, the SPIR-V execution modes
  and `NoContraction` policy the **compiler** applies (§3.4 step 3). A mode the device does
  not advertise is not requested — an unsupported capability in a module is invalid SPIR-V;
  the tier reports the shortfall instead.
- `SlangCompiler::compile(source, entry, profile, defines, modes)` invokes `slangc`
  (`ES_SLANGC` or `PATH`) and caches at
  `target/es-slang-cache/<blake3(source+entry+profile+defines+modes+includes+slangc version)>.spv`.
  A hit starts no process. `slangc 2026.8` emits only `DenormFlushToZero`; the other two
  execution modes and the `NoContraction` decorations are patched into the binary by
  `spirv::apply_exec_modes`, and `spirv_has_execution_mode` / `spirv_no_contraction_count`
  read them back out of the binary for the test.
- `Buffer::new(gpu, bytes, Usage::{Storage, Uniform, Staging})` with `upload`/`download`;
  `ComputePipeline::new(gpu, &SpirvModule, &[BindingDesc])`;
  `CommandRecorder::dispatch(pipeline, buffers, groups)` + `submit_and_wait`, with an
  explicit shader-write → shader-read barrier after every dispatch. No atomics, no multi-
  queue, no FP atomic helper (§3.4 forbids them, so they are not offered).
- `kernels/sum.slang` is a fixed-order pairwise tree reduction (no shared memory, no subgroup
  ops, no atomics, no device-derived workgroup count). `kernels/approx_probe.slang` includes
  `crates/es-math/slang/approx.slang` and evaluates `es_sin` / `es_exp`.

## oracle

```
cargo fmt -p es-gpu --check
cargo clippy -p es-gpu --all-targets -- -D warnings
cargo test -p es-gpu -- --nocapture
cargo xtask layering && cargo xtask context-budget && cargo xtask check-spec-refs
```

Every GPU test prints `SKIP <test>: <reason>` and returns when no device (or no `slangc`) is
found, so the suite passes on CI without a GPU and says so. The unit tests — tier derivation
from canned property structs, cache-key separation, the SPIR-V parser/patcher/inspector —
run everywhere.

## acceptance

Measured on an NVIDIA RTX 4060 Laptop GPU, driver 592.82, Vulkan 1.4.325, Slang 2026.8:

- `device_capabilities_are_queried_and_reported` — prints the device, driver, float-controls,
  subgroup and derived tier. This device reports `shaderDenormFlushToZeroFloat32 = false`
  (NVIDIA preserves f32 denormals), so the derived tier is **2 / `CrossBackend`**, and the
  flush-to-zero mode is not requested of the compiler.
- `execution_modes_are_present_in_the_spirv` — all three execution modes are found in the
  compiled binary and `NoContraction` covers every float-arithmetic result (1/1 for
  `sum.slang`).
- `cache_hit_does_not_start_slangc` — first compile invokes `slangc` once, second serves the
  cache with no further invocation and identical words.
- `tree_reduction_is_bit_identical_across_runs_and_matches_the_cpu` — 1M f32, two runs
  bit-identical, and bit-identical to the CPU mirror of the same tree (max ULP 0). The
  `BinnedAcc` (§18.4) exact sum is the accuracy reference: relative difference 2.0e-7, as
  expected for an f32 tree.
- `gate3_cpu_and_slang_transcendentals_agree_bit_for_bit` — **spec §28.7 gate 3 met on this
  device**: 4096 inputs, 0 mismatched, max ULP `sin` 0, `exp` 0.

## forbidden

Any file outside `context`. Other dependency lines in the root `Cargo.toml`. Treating float
controls as a setting, in code or comment (§3.4, Appendix C). A second queue, an FP-atomic
helper, or any API that lets a caller opt out of the barriers. New extension-point traits
(INV-17). `HashMap`/`HashSet` (§3.4). Editing `crates/es-math/slang/approx.slang` or the
coefficient mirror — a gate-3 failure is reported as a max ULP, never fixed by moving the
reference. Graphics pipelines, ray tracing, external memory, multi-GPU (deferred; see the
design doc).
