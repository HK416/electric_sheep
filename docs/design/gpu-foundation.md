# GPU foundation (`es-gpu`, layer 2)

Spec: §2.3 (Slang → SPIR-V, offline compile + content-hash cache), §3.3–§3.4 (precision,
deterministic execution contract), §3.5 (determinism tiers), §4.2 rule 3 (layers ≤ 2 know no
Vulkan symbols, `es-gpu` excepted), §11.4–§11.6 (GPU lowering, release/debug plans,
capability check), §21 (`TensorTransport`), §22 (multi-GPU), §28.7 gate 3.

`es-gpu` is the only crate that links Vulkan. It owns: device open + capability *query*,
Slang invocation + SPIR-V cache, buffers, compute pipelines, one compute queue. It owns no
policy: which kernel to run, how to fuse, what to schedule is §11 (`es-compile`) work.

## Device selection

`Gpu::open(GpuOptions { prefer_discrete, validation })`:

1. Create a `VkInstance` at API 1.3. `validation = true` adds
   `VK_LAYER_KHRONOS_validation`; when the layer is absent the call still succeeds without it
   (CI machines have no SDK) and `Gpu::validation_enabled()` reports what actually happened.
2. Enumerate physical devices. Score: discrete > integrated > virtual > CPU when
   `prefer_discrete`, reversed order otherwise; ties broken by enumeration index. Index order
   is *not* an identity — §22 forbids matching devices by enumeration order, so the device
   UUID is carried in `Capabilities` and is what goes into `hardware_capability` of the
   execution hash (§5.3).
3. Pick the first queue family with `COMPUTE`, create the device with exactly **one** queue
   from it. §3.4 forbids multiple compute queues in deterministic mode, so no other queue is
   ever created; there is no API here to ask for one.
4. `Gpu::none_available()` is the CI predicate: true when the loader is missing, instance
   creation fails, or no physical device is reported. Every GPU test starts with it and
   prints `SKIP` with the reason. `cargo xtask ci` runs the test step with `--nocapture` and
   scans for those `SKIP` lines: with `ES_REQUIRE_GPU=1` — a machine that claims a GPU — any
   of them fails the run, the way `cargo xtask nostd --require` treats a missing target. The
   PR runner has no GPU and leaves the variable unset, so there the skips are only reported.

## Capabilities are queried, never set

§3.4 is explicit and Appendix C repeats it: `VkPhysicalDeviceFloatControlsProperties` is a
capability **query**. Nothing in this crate writes a float-control; the struct is filled from
`vkGetPhysicalDeviceProperties2` and read.

```
Capabilities {
    device_name, device_uuid, device_type, driver_name, driver_info, driver_version,
    api_version,
    float_controls: FloatControls,          // every field of the Vulkan properties struct
    subgroup: Subgroup { size, supported_ops, quad_operations_in_all_stages },
    shader_float64, shader_int64, cooperative_matrix: bool,
    max_workgroup: { count: [u32; 3], size: [u32; 3], invocations },
    memory_heaps: Vec<MemoryHeap>,          // size + device-local flag, for §20 budgets
}
```

`cooperative_matrix` is presence of `VK_KHR_cooperative_matrix` in the device extension list
(§2.4 `VulkanRuntime`); the crate does not use it yet.

### `determinism_tier()`

Derivation (§3.4 step 2, §3.5):

| condition | tier |
|---|---|
| `shader_denorm_flush_to_zero_float32` **and** `shader_rounding_mode_rte_float32` **and** `shader_signed_zero_inf_nan_preserve_float32`, **and** the mode is not forced globally against us — i.e. `denorm_behavior_independence` / `rounding_mode_independence` is `All` or `32BitOnly`, **and** a subgroup size is reported | `Bitwise` (tier 1) |
| otherwise | `CrossBackend` (tier 2) |

`Bitwise` is what §3.5 tier 1 promises *given the same `execution_hash` on the same device and
driver*: the driver and device identity are part of that hash, they are not something the
capability query can establish. A tier-1 claim from this crate means "the four remaining
knobs (float controls, no-contraction, reduction order, single-queue static schedule) can be
pinned on this device", nothing more. The other four items of the eight-item contract
(deterministic reduction, static partitioning, fixed subgroup behaviour, fixed algorithm) are
the caller's; `es-math::reduce` and the kernels here supply them.

### `deterministic_execution_modes()`

Returns `ExecModes` — the SPIR-V execution modes and the `NoContraction` decoration policy the
**compiler** must apply to a module (§3.4 step 3). It is a compile input, not a device
setting; the name says `execution_modes` for that reason and there is deliberately no setter
anywhere on `Gpu`.

```
ExecModes { denorm_flush_to_zero_f32, rounding_mode_rte_f32, signed_zero_inf_nan_preserve_f32,
            no_contraction }
```

`ExecModes::deterministic()` turns all four on; `ExecModes::none()` is the permissive default
for kernels outside the deterministic contract (§3.2 exempts neural-network kernels).

## Slang → SPIR-V

`SlangCompiler::compile(source, entry, profile, defines, exec_modes) -> SpirvModule`.

- `slangc` is found through `ES_SLANGC` or `PATH`. Its `-v` output goes into the cache key,
  because §3.4 item 7 pins the compiler version, not only the source.
- Cache: `target/es-slang-cache/<hash>.spv`, `hash = blake3(source ‖ entry ‖ profile ‖
  defines ‖ exec_modes ‖ slangc version ‖ include dirs ‖ every file under them)`. The include
  walk is **recursive** and sorted, keyed by each file's path relative to its include root, so
  a `#include "sub/helper.slang"` is part of the identity; walk errors propagate rather than
  quietly weakening the key. A hit does not invoke `slangc` at all;
  `SlangCompiler::invocations()` counts real invocations so the test can prove the hit.
  Content-addressed, therefore shareable between ranks (§22) and packageable into a bundle so
  deployment needs no Slang (§11.4).
- Flags: `-target spirv -profile <p> -entry <e> -O0 -fp-mode precise -emit-spirv-directly`
  plus `-denorm-mode-fp32 ftz` when the mode is requested. Exact flags and what each was
  observed to emit: `docs/api-notes/slang.md`.
- What `slangc` does **not** emit is added by patching the module: `RoundingModeRTE`,
  `SignedZeroInfNanPreserve` and the `NoContraction` decorations. The patcher is a hand
  SPIR-V writer (`spirv.rs`, ~200 lines, no dependency) that inserts the capability,
  the `SPV_KHR_float_controls` extension, the `OpExecutionMode`s after the entry point, and
  one `OpDecorate <id> NoContraction` per float-arithmetic result id.
  The patch is keyed on `(entry point, mode, float width)`: it covers *every* `OpEntryPoint`,
  a mode declared at another width does not suppress the requested f32 one, and a module that
  already declares a contradicting mode at that width (`DenormPreserve 32` vs a requested
  `DenormFlushToZero 32`) is refused with `GpuError::Spirv` rather than patched into something
  invalid.
- `spirv_has_execution_mode(&words, mode)` / `spirv_no_contraction_count(&words)` parse the
  header and instruction stream so tests can *prove* the modes are in the binary rather than
  trusting a flag.
- `validate_spirv(&words)` runs **`spirv-val`** (Vulkan SDK, `PATH` or `ES_SPIRV_VAL`) over
  the patched module on every cold compile — a header parse says a word stream is well-formed,
  not that a module is legal, and this crate writes SPIR-V by hand. Missing tool: one printed
  `NOTE` and `Ok(false)`, or an error under `ES_REQUIRE_SPIRV_VAL=1`.

`SpirvModule { words, hash, entry }`. The hash is the `compile_hash` ingredient of §11.4.

## Buffers and pipelines

`gpu-allocator` owns device memory; buffers are `Buffer::new(gpu, bytes, Usage)` with
`Storage` / `Uniform` / `Staging` (host-visible, `CpuToGpu`) / `Readback` (host-cached,
`GpuToCpu`). `upload` copies through a `Staging` buffer and a one-shot command buffer;
`download` copies through a `Readback` buffer (packet M7/R1b): reading a write-combined
`CpuToGpu` mapping ran at 27–44 MiB/s on both oracle GPUs, so a 3.6 MB 1280×720 readback
cost 64–85 ms, while the cached mapping reads at 2.9–3.3 GiB/s (1.1–1.2 ms) on the RTX 3060
(`docs/design/renderer.md` section 8.4 has both GPUs). A device without `GpuToCpu` memory
falls back to `Staging` for downloads and says so once on stderr. A `Buffer` records the `bytes` it was asked for
alongside what was allocated (`bytes.max(4)`, then the allocator's alignment padding), and
`download` returns exactly the requested length — the padding is never handed back as if it
were data.

`ComputePipeline::new(gpu, &SpirvModule, &[BindingDesc])` builds a single descriptor set
(set 0, storage or uniform buffers only). `CommandRecorder` records
`dispatch(pipeline, bindings, groups)` calls on the one compute queue, inserts an explicit
`SHADER_WRITE → SHADER_READ` memory barrier between dispatches, and `submit_and_wait` blocks
on a fence.

There are no atomics, no multi-queue, no async transfer, no timeline semaphores: §3.4 forbids
atomic FP sums and multi-queue physics in deterministic mode, so the API does not offer them.

## Kernels and gate 3

`kernels/sum.slang` — fixed-order pairwise tree reduction, one pass per level, no shared
memory, no subgroup ops, no atomics. The order is a property of the index arithmetic, so it
cannot depend on scheduling. The CPU mirror performs the identical tree in `f32`; `BinnedAcc`
(§18.4) is the accuracy reference, not a bit reference — a `f32` tree and an exact binned sum
are different numbers by construction.

`kernels/approx_probe.slang` — `#include`s `crates/es-math/slang/approx.slang` and evaluates
`es_sin` / `es_exp`; the test compares the result bit-for-bit with `es_math::approx` on the
CPU. This is §28.7 gate 3 (CPU/Slang bit equality). The outcome is reported as bits equal or
as a max-ULP figure with the gate marked unmet — never as a pass by tolerance.

## Deferred

Graphics pipelines and the renderer (§16, `es-render`, layer 5). Ray tracing (§28.6).
Multiple queues and async compute (forbidden under §3.4 anyway). External memory /
`TensorTransport` negotiation (§21) — needs CUDA/HIP symbols, which by §4.2 rule 5 only
`es-transport` may link. Multi-GPU enumeration and roles (§22). Specialization constants for
shape sharing, kernel fusion and the release/debug plan split (§11.4–§11.5) — those are
`es-compile` decisions; this crate only compiles and dispatches what it is given. Timestamp
queries for the §12.4 metric set. Pipeline caches (the SPIR-V cache is content-addressed; a
`VkPipelineCache` is a driver-side extra with no determinism value).
