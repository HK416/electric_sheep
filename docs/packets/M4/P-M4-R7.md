# P-M4-R7 — width- and entry-point-aware SPIR-V patching, validated

Spec: §3.3 (f32 is the precision the modes are requested at), §3.4 step 3 (execution modes
and `NoContraction` are applied by the compiler, via SPIR-V, never by a device setting),
§3.5 tier 1.
Review findings: `docs/reviews/M4.md` S-5 — `es-gpu/src/spirv.rs:146,257`'s "already
present?" guard compared only the **mode word**, ignoring the width operand and the entry
point, so `DenormFlushToZero 16` silently suppressed the requested `32` and a module already
carrying `DenormPreserve 32` would be patched into invalid SPIR-V; `:193` patched only the
**first** entry point (`find`, not `filter`). S-6 — nothing ran `spirv-val` in the one crate
that hand-writes SPIR-V; `pipeline.rs:79` said "validated" when it meant header-parsed.

## context

```
crates/es-gpu/src/spirv.rs
crates/es-gpu/src/slang.rs
crates/es-gpu/src/lib.rs
crates/es-gpu/tests/determinism.rs
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R7.md
```

## spec

- `apply_exec_modes(words, modes) -> Result<Vec<u32>, GpuError>` (was
  `Option<Vec<u32>>`, which had no room to say *why*).
- It collects **every** `OpEntryPoint` id, not the first, and adds each requested mode to
  each of them.
- `declared_modes(words, insts) -> Vec<(entry, mode, width)>` reads the one-literal-operand
  form of `OpExecutionMode` (`i.len == 4`), which is exactly the `SPV_KHR_float_controls`
  shape; `LocalSize` and friends carry a different word count and can never be mistaken for
  it. The "already present?" test is `declared.contains(&(entry, mode, 32))` — per entry
  point, per width.
- `conflicting_mode(mode)`: `DenormFlushToZero ↔ DenormPreserve`, `RoundingModeRTE ↔
  RoundingModeRTZ`. If any entry point already declares the conflicting mode at the width
  being patched, `apply_exec_modes` returns `Err(GpuError::Spirv(..))` **before emitting
  anything** — a module declaring both is invalid SPIR-V, so refusing is the only honest
  answer. `EXEC_MODE_DENORM_PRESERVE` (4459) and `EXEC_MODE_ROUNDING_MODE_RTZ` (4463) are
  added as public constants.
- `validate_spirv(words) -> Result<bool, GpuError>` runs `spirv-val` (`PATH`, or
  `ES_SPIRV_VAL`) over a module written to a uniquely named temp file. `Ok(true)` it ran and
  accepted; `Ok(false)` it is not installed — one `NOTE spirv-val is not on PATH: …` per
  process; `Err` it rejected the module, or it is absent and `ES_REQUIRE_SPIRV_VAL=1`.
- `SlangCompiler::compile` calls `validate_spirv` on the patched words, after
  `apply_exec_modes` and before publishing the cache entry — so a cache hit never pays for
  it and an invalid patch never reaches the cache.

## oracle

```
cargo test -p es-gpu spirv
cargo test -p es-gpu -- --nocapture
```

- `a_mode_at_another_width_does_not_suppress_the_requested_one` — a module carrying
  `DenormFlushToZero 16` comes out with widths `[16, 32]`.
- `every_entry_point_gets_every_mode` — a two-entry module: all three modes land on both ids,
  and re-patching is a no-op.
- `patch_refuses_a_conflicting_mode_at_the_same_width` — `DenormPreserve 32` plus a requested
  `DenormFlushToZero 32` is an `Err` naming 4459; the same pair at *different* widths is
  accepted.
- `patched_kernels_pass_spirv_val` — `sum.slang` and `approx_probe.slang`, patched with
  `ExecModes::deterministic()`, are accepted by `spirv-val`; when the tool is absent the test
  prints `SKIP … no spirv-val on PATH` and says how to make that an error.

## acceptance

- Every `OpEntryPoint` in a patched module carries every requested mode at width 32.
- A mode declared at another width does not suppress the f32 one.
- A contradicting mode at the patched width is an error, not a patch.
- `spirv-val` accepts the determinism kernels after patching. Observed on this machine (RTX
  4060, Vulkan SDK on `PATH`): `RAN spirv-val: patched sum.slang accepted`,
  `RAN spirv-val: patched approx_probe.slang accepted`.
- The existing oracles are unchanged: exec modes 3/3 and `NoContraction` 1/1 in
  `execution_modes_are_present_in_the_spirv`, gate 3 at 4096 inputs / 0 mismatched.

## forbidden

- No SPIR-V crate dependency: `spirv.rs` stays a hand reader/writer with no `rspirv`,
  `spirv-tools` or similar in `Cargo.toml`. `spirv-val` is an external *tool*, invoked, never
  linked.
- No skipping validation to make a test pass, and no path that disables it: the only lever is
  the tool's absence, and `ES_REQUIRE_SPIRV_VAL=1` closes that.
- No widening to f16/f64 modes — §3.3 asks for f32 and only f32 is patched.
- No change to the `NoContraction` pass, the section-boundary insertion logic, or
  `spirv_has_execution_mode`'s public signature.
- No edits to `buffer.rs` or `xtask` (P-M4-R6), or to the cache key (P-M4-R5).
