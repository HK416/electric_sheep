# `slangc` — pinned API digest

Compiler: **Slang `2026.8`** (`slangc.exe` from the Vulkan SDK at `E:\VulkanSDK`, found via
`PATH` or `ES_SLANGC`). The version string goes into every cache key: spec §3.4 item 7 pins
the compiler, not only the source.

Observations below are from `slangc 2026.8` on this machine, checked by disassembling the
output with `spirv-dis`. Anything not checked that way is marked `unverified`.

## Invocation

```
slangc <src>.slang -target spirv -profile glsl_450 -entry main -emit-spirv-directly \
       -O0 -fp-mode precise [-denorm-mode-fp32 ftz] [-I <dir>]... [-D<k>=<v>]... -o <out>.spv
```

| Flag | Why | Status |
|---|---|---|
| `-target spirv` | SPIR-V output (spec §2.3) | verified |
| `-profile glsl_450` | compute profile; SPIR-V 1.3 header is emitted | verified |
| `-entry main` | entry-point function | verified |
| `-emit-spirv-directly` | do not detour through GLSL + glslang (the default since 2024, passed explicitly so a future default change cannot move the bits) | verified |
| `-O0` | no optimizer. Reassociation is what kills bit reproducibility (spec §3.4) | verified (the SPIR-V keeps the source's operation order) |
| `-fp-mode precise` | no fast-math | verified as accepted; it does **not** by itself add `NoContraction` |
| `-denorm-mode-fp32 ftz` | emits `OpCapability DenormFlushToZero`, `OpExtension "SPV_KHR_float_controls"` and `OpExecutionMode %main DenormFlushToZero 32` | verified in `spirv-dis` |
| `-I <dir>` | include path, used for `crates/es-math/slang/approx.slang` | verified |
| `-D<k>=<v>` | preprocessor defines (e.g. `ES_N`) | verified |
| `-o` | output path | verified |

`slangc -v` prints the version on **stderr**, not stdout.

## What `slangc` does not emit

`2026.8` has no flag for the other two execution modes of spec §3.4 step 3 and none for the
`NoContraction` decoration:

* `OpExecutionMode … RoundingModeRTE 32` — no flag found (`slangc -h` lists only the
  `-denorm-mode-fp{16,32,64}` family);
* `OpExecutionMode … SignedZeroInfNanPreserve 32` — same;
* `OpDecorate <id> NoContraction` — `-fp-mode precise` did not produce any in the compiled
  `sum.slang`.

`es-gpu` therefore **patches the compiled module** (`src/spirv.rs`): it adds the missing
`OpCapability` / `OpExtension` / `OpExecutionMode` instructions and one `OpDecorate …
NoContraction` per float-arithmetic result. The patch is idempotent and is verified by the
test `execution_modes_are_present_in_the_spirv`, which parses the final binary rather than
trusting any flag. If a later Slang release grows the flags, the patch stays correct (it
skips what is already there) and the flags can be added to the command line.

`unverified`: whether a Slang attribute (`[require(...)]`, `[SpvExecutionMode(...)]` or
similar) can express `RoundingModeRTE` in source. The source-level route was not found in
`slangc -h` output; the patch was chosen because it is checkable from the binary.

## Cache

`target/es-slang-cache/<blake3>.spv`, overridable with `ES_SLANG_CACHE` (or
`CARGO_TARGET_DIR`). The key hashes: source text, entry, profile, defines, the requested
`ExecModes`, every include directory path **and the contents of the files in it**, and the
`slangc -v` string. A hit does not start a process — `SlangCompiler::invocations()` proves
it in `cache_hit_does_not_start_slangc`.

Content-addressing is what lets ranks share the cache (spec §22) and lets `es task compile`
package it into a bundle so deployment needs no Slang (§11.4).

## Shader conventions

* `[[vk::binding(<binding>, 0)]]` on every resource: descriptor set 0, explicit binding.
* `RWStructuredBuffer<float>` / `StructuredBuffer<float>` both land as
  `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` (SPIR-V 1.3 emits them as `BufferBlock` in the
  `Uniform` storage class).
* The SPIR-V entry point is named `main` regardless of the Slang function name unless
  `-fvk-use-entrypoint-name` is passed; `ComputePipeline` reads the name out of the module
  instead of assuming.
* `[numthreads(64, 1, 1)]` everywhere here, so the workgroup size is a constant of the
  kernel and never derived from a device property (spec §3.4 item 6).
