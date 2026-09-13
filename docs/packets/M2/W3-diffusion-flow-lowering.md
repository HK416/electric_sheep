# W3 — Diffusion and FlowMatching lowering, tier 4 (`es-policy`)

Spec: spec 8.3 (node set: `DiffusionHead`, `FlowMatchingHead`), spec 8.5 (action modelling),
spec 8.7 (lowering: the head is the `PolicyHandle` part), spec 8.9 (tier 4 tolerance),
spec 3.4 (determinism, no global RNG), spec 5.3 (hash chain), spec 28.4 W3, spec 28.7 gate 12.
Design note: `docs/design/learning-lowering.md` section 8 (review class C — the note is the
artifact, the code is downstream of it).

The offline-verifiable part of the spec 28.4 W3 row. SmolVLA (a `PolicyBundle`) and the ONNX
runtime are **not** in this packet: ONNX is a sibling lowering target with its own `ort`
dependency and its own oracle, and the two policies compared here are torch against the Rust
reference, not torch against ONNX. Gate 12 ("three policies pass tier 4") gets two of its three
from here; the third is the ONNX packet.

## context

```
crates/es-policy/src/lower/torch.rs                 (extended: two head arms + the sampler runtime)
crates/es-policy/src/lib.rs                         (extended: `#[cfg(test)] mod reference`)
crates/es-policy/src/reference.rs                   (new, test-only)
docs/design/learning-lowering.md                    (extended: section 8)
docs/packets/M2/W3-diffusion-flow-lowering.md       (new)
```

## spec

- `PolicyHead { Diffusion { n_steps, scheduler } }` lowers to `_DdpmHead`: the reverse loop
  `x = c1[t] * x + c3[t] * eps_theta(x, cond, t) (+ sigma[t] * z_t)` for `t = n_steps-1 .. 0`
  over a linear beta schedule (`1e-4 .. 0.02`). `Ddpm` and `Ddim` (eta = 0) share the loop and
  differ only in the three coefficient lists, which are computed in Rust f32 by
  `diffusion_schedule` and emitted as Python literals. `DpmSolver` is `LowerError::Unsupported`.
- `PolicyHead { FlowMatching { n_steps } }` lowers to `_FlowHead`: Euler integration of
  `dx/dt = v_theta(x, cond, t)` from noise at `t = 0` to the action at `t = 1`, `dt = 1/n_steps`.
- Both heads share `_Denoiser`, `l1(relu(l0([x_t, cond, temb(t)])))`, with the hidden width and
  the sinusoidal embedding width equal to the conditioning width (spec 8.3 carries neither, so
  they are lowering choices, as `nhead = 8` already is). `cond` must be even.
- **Determinism (spec 3.4).** The initial `x_T` is a *declared graph input* (`noise`), so a
  sampler head declares two input ports and `infer` stays a pure function of its named inputs;
  a one-input sampler head is `LowerError::Shape`. The per-step `z_t` are non-trainable
  checkpoint buffers `nodes.<k>.noise_<t>`, exact keys that `validate_keys` requires. DDIM
  declares none. No seeded generator, no `torch.Generator`, no global RNG.
- `reference.rs` is `#[cfg(test)]` and mirrors the generated module in f32 with a fixed op order
  and `es_math::approx` for `exp`/`sin`/`cos`/`sqrt`. It shares `diffusion_schedule` and the
  noise buffers with the lowering, and mirrors everything else.
- No `ort`/ONNX, no new trait (`INV-17` untouched), no `HashMap`, `BTreeMap` only, no change to
  `es-ir`, `es-safety` or the root manifest.

## oracle

```
cargo fmt -p es-policy --check
cargo clippy -p es-policy --all-targets -- -D warnings
cargo test -p es-policy
ES_PYTHON=<venv>/Scripts/python.exe cargo test -p es-policy
cargo xtask check-spec-refs
```

The tier-4 tests SKIP with a printed reason when no interpreter with `torch` is found and RUN
when `ES_PYTHON` names one — spec 1.4 wants the harness present whether or not the oracle is
installed on a given machine, and a missing wheel must never read as a passing equivalence.

## acceptance

- Lowering is byte-identical across runs for both heads; `n_steps` and the scheduler both move
  `lowering_hash`.
- The generated source contains the sampling loop and the declared `n_steps` literally.
- DDPM declares `n_steps - 1` `noise_<t>` keys (none at `t = 0`); DDIM and FlowMatching declare
  none; a DDPM checkpoint fails `validate_keys` against a FlowMatching module and a missing
  `noise_3` is reported as `missing`.
- torch vs the Rust reference, state 4 / cond 16 / action 2 / horizon 3 / 8 steps, torch
  2.14.0+cpu, against spec 8.9's `abs <= 1e-5`:

  | head | `max_abs` | `max_rel` |
  |---|---|---|
  | `Diffusion { Ddpm }` | 5.96e-8 | 2.60e-6 |
  | `Diffusion { Ddim }` | 3.73e-8 | 4.32e-7 |
  | `FlowMatching` | 1.04e-7 | 1.37e-6 |

- Re-running the same policy is bitwise identical (spec 8.9's last row).

## forbidden

- `crates/es-env`, `crates/es-eval`, `crates/es-compile`, `crates/es-data`, `crates/es`, and
  every other crate: neighbouring packets own them. The root `Cargo.toml` too.
- ONNX / `ort` (its own packet), `PolicyBundle` / SmolVLA / π₀ (spec 8.3 references them whole),
  cross-attention and FiLM conditioning, a UNet denoiser, `DpmSolver`, batched lowering.
- Editing `crates/es-ir`: the sampler contract is expressible with the node set as it stands.
- Weakening `Tolerance::TIER4_FP32`, or making a sampler draw its own randomness.
