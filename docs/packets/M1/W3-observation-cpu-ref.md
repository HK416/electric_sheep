# W3 — Observation IR → CPU reference execution plan (part 1)

Spec: spec 7.2, spec 7.3, spec 7.5, spec 7.6, spec 7.7, spec 11.1, spec 11.3, spec 11.5,
spec 3.1, spec 3.4. Design note: `docs/design/observation-lowering.md` (review class C — the
numerics are the artifact; the code is downstream of them).

## context

```
docs/design/observation-lowering.md      (new, written first)
docs/packets/M1/W3-observation-cpu-ref.md (new)
crates/es-compile/Cargo.toml             (adds `half`)
crates/es-compile/src/lib.rs
crates/es-compile/src/plan.rs            (new)
crates/es-compile/src/kernels.rs         (new)
crates/es-compile/src/exec.rs            (new)
crates/es-compile/tests/gen_goldens.rs   (new, #[ignore], run once)
crates/es-compile/tests/observation_cpu.rs (new)
tests/golden/observation/**              (new, CI read-only once committed)
```

## spec

- `plan::CpuPlan::compile(&ObservationIr, PlanMode) -> Result<CpuPlan, Vec<Diagnostic>>` —
  validates through the IR, propagates `ImageSpec`s, topologically orders, resolves each
  node's buffer from its **declared** `PortType`, and lays one flat arena out once.
  `PlanMode::{Debug, Release}` is recorded and hashed; both keep every intermediate
  addressable and neither aliases (rationale: design note §10).
- `kernels` — `resize_bilinear`, `resize_nearest`, `crop`, `srgb_to_linear` (+ the LUT-256 u8
  path), `normalize`, `stack`, `concat`, `history_push`, `window_gather`,
  `cast_u8_hwc_to_f32_chw`, `cast_f32_to_f16`, `cast_f32_to_bf16`. Pure functions over slices,
  single-threaded, fixed loop order, `es_math::approx` for the only transcendental
  (`exp(2.4 * ln t)` in the sRGB EOTF). No `powf`/`exp` from `std` (spec 3.2, `DET-010`).
- `exec::{Tensor, TensorRef, Outputs, ExecError}` and
  `CpuPlan::run(&mut self, &BTreeMap<String, TensorRef>) -> Result<Outputs, ExecError>`.
  `Tensor` is defined here: `es-ir` has no runtime tensor type and must not grow one.
- `CpuPlan::compiler_hash()` — blake3 of (crate version, plan mode, kernel ids in table order),
  for the `compiler` slot of `execution_hash` (spec 5.3, spec 11.2).
- Layouts: u8 HWC at the sensor boundary, f32 CHW downstream; `Dequantize` is the only node
  that changes layout (design note §2).
- Node kinds outside the part-1 subset produce `COMPILE-002`, never a silent no-op. The
  `COMPILE-0xx` codes are local constants here with a `TODO(codes-merge)`, following the way
  the five IR modules held their codes before P27 merged them into `es_ir::codes`.

## oracle

```
cargo test -p es-compile && cargo xtask verify-goldens
```

## acceptance

- Byte-for-byte equality with `tests/golden/observation/*.bin` for: an 8×6 RGB u8 gradient
  dequantised to CHW f32; that image resized to 4×3 bilinear; a crop of it; the 256-entry
  sRGB→linear LUT; a per-channel normalize; a 2-frame history window. Each `.bin` has a
  `.json` sidecar naming shape, dtype, kernel and what it pins.
- Proptests: a resize of a constant image is that constant everywhere (no edge leakage from
  the clamp); `crop(crop(x, a), b) == crop(x, a ∘ b)` in pixels **and** the chained
  `ImageSpec::cropped` intrinsics equal the combined-rectangle ones (`INV-14`).
- A plan compiles from `es_ir::testing::arbitrary_observation_ir` and runs.
- `compiler_hash` differs between `PlanMode::Debug` and `PlanMode::Release`.
- Gate: `cargo fmt -p es-compile --check`, `cargo clippy -p es-compile --all-targets -- -D
  warnings`, `cargo test -p es-compile`, `cargo xtask layering`, `cargo xtask verify-goldens`,
  `cargo xtask context-budget`.

## forbidden

- Any file outside `context`. In particular `crates/es-safety`, `crates/es-env`,
  `crates/es-data` (other packets own them), `crates/es-ir` (its API is consumed, never
  edited), and the root `Cargo.toml`.
- Editing a golden file to make a test pass (spec 1.4). `GOLDEN_UPDATE=1` is not a fix.
- `image`, `ndarray`, or any other array/image dependency: the kernels are the artifact.
- `HashMap`/`HashSet` (spec 3.4), threads, `rayon`, `std::f32::powf`/`exp`/`ln` in a kernel.
- A new trait — the seven extension points of `INV-17` are the whole list, and none of them is
  a kernel or a plan.
- Touching `Intrinsics` arithmetic anywhere in the plan: geometry goes through
  `ImageSpec::resized` / `ImageSpec::cropped` (`INV-14`).
- GPU lowering, fusion, buffer aliasing, `Augment` RNG — later waves.
