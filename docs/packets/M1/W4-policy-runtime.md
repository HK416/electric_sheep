# W4 — `PolicyRuntime` and the `PyTorch` lowering (`es-policy`)

Spec: spec 2.4, spec 8.1, spec 8.3, spec 8.4, spec 8.5, spec 8.7, spec 8.8, spec 8.9, spec 1.4,
spec 5.3, spec 4.2. Design note: `docs/design/learning-lowering.md` (review class C — the node
table and the tolerance are the artifact; the code is downstream of them).

Groundwork for the M1 gate of spec 8.9 ("load a LeRobot ACT checkpoint and get the same action
for the same observation"). This packet builds the harness and the lowering; the checkpoint
fixture and the key remap are the gate packet's.

## context

```
docs/design/learning-lowering.md          (new, written first)
docs/api-notes/torch.md                   (new)
docs/packets/M1/W4-policy-runtime.md      (new)
crates/es-policy/Cargo.toml               (new)
crates/es-policy/src/lib.rs               (new)
crates/es-policy/src/runtime.rs           (new)
crates/es-policy/src/lower/mod.rs         (new)
crates/es-policy/src/lower/torch.rs       (new)
crates/es-policy/src/torch_runtime.rs     (new)
crates/es-policy/src/weights.rs           (new)
crates/es-policy/src/equiv.rs             (new)
crates/es-policy/python/torch_ref.py      (new)
crates/es-policy/tests/torch_equivalence.rs (new)
```

## spec

- `runtime::PolicyRuntime` — object-safe, four methods: `load(&LearningGraph, &WeightsSource)
  -> PolicyInfo`, `infer(&BTreeMap<String, Tensor>) -> BTreeMap<String, Tensor>`, `info()`,
  `runtime_hash() -> [u8; 32]`. One of the seven extension points of `INV-17`. Inputs are
  **already preprocessed** and named per `PolicyContract::inputs`; outputs are the graph's
  declared ports. The runtime adds no pre- or post-processing: normalization, chunking and
  unnormalization are IR nodes and therefore inside the lowered graph (spec 2.4, spec 8.7).
- `WeightsSource::{Safetensors(PathBuf), InMemory(Vec<u8>)}` — no pickle variant, ever
  (`INV-16`). `InferenceBackend` is an **enum** (`Torch | Onnx | Vulkan`), not a trait: it would
  have one implementation today. The `INV-17` trait slot stays reserved.
- `lower::torch::lower_to_torch(&LearningGraph) -> Result<TorchModule, LowerError>` where
  `TorchModule { source, weight_keys, weight_shapes, lowering_hash }`. `source` is a complete
  Python file defining `class EsPolicy(nn.Module)` with `forward(**inputs) -> dict`. Codegen in
  `Graph::topo_order()` order, one member and one forward line per node; an unsupported variant
  is `LowerError::Unsupported(kind)`, never a silent no-op. Output is deterministic —
  `lowering_hash = blake3(tag || source)` and the same graph must give the same bytes.
- `torch_runtime::TorchRuntime` — `PyTorch` behind a Python subprocess, copying the JSON-lines
  pattern of `crates/es-physics-backend/src/proc.rs`. `python/torch_ref.py` is embedded with
  `include_str!`. `is_available()` probes `import torch`. `runtime_hash =
  blake3(tag || "torch" || PROTOCOL_VERSION || torch version)`.
- `weights` — safetensors reader and writer (header length u64 LE + JSON + raw bytes), key and
  shape validation against `TorchModule` (`PolicyError::WeightMismatch { missing, unexpected,
  shape }`), `weights_hash` = blake3 of the file bytes, cross-checked against
  `WeightsRef::hash` **before** the backend is started (spec 5.3).
- `equiv::compare_actions(&Tensor, &Tensor, Tolerance) -> Equivalence { max_abs, max_rel, pass }`
  — the tier-4 harness of spec 8.9, with `TIER4_FP32` (1e-5), `TIER4_VULKAN` (1e-4) and
  `BITWISE` from that table.

## oracle

```
cargo test -p es-policy
ES_PYTHON=<venv with torch>/Scripts/python cargo test -p es-policy   # runs the equivalence test
```

## acceptance

- `torch_mlp_matches_rust`: a `StateEncoder` MLP 4 -> 8 -> 8 into a `RegressionHead` (2 dims,
  horizon 3), fixed-seed synthetic weights written as safetensors, run through `TorchRuntime`
  and compared against an in-test Rust f32 evaluation with a fixed op order, at
  `Tolerance::TIER4_FP32`. **SKIPs with a printed reason when `torch` is absent** — spec 1.4
  wants the harness present whether or not the oracle is installed, and a missing wheel must
  never be reported as a passing equivalence.
- The same input twice through the same loaded runtime is bitwise identical (spec 8.9, last
  row of the table).
- Lowering determinism: two lowerings of one graph are byte-identical, hash included.
- The `es-ir` ACT fixture lowers to exactly one forward line per node, with the documented
  module names and the documented weight keys and shapes.
- Weight validation catches a missing exact key, an empty prefix claim, a wrong shape and a
  stray key, each named.
- The wire protocol encodes and decodes against canned JSON, so it is covered with no Python.
- `WeightsSource` has exactly two variants and the embedded script contains no `import pickle`
  and no `torch.load(` (`INV-16`).
- Gate: `cargo fmt -p es-policy --check`, `cargo clippy -p es-policy --all-targets -- -D
  warnings`, `cargo test -p es-policy`, `cargo xtask layering`, `cargo xtask check-spec-refs`,
  `cargo xtask context-budget`.

## forbidden

- Any file outside `context`. In particular `crates/es-safety` — `es-policy` must never appear
  in its dependency graph in either direction (spec 4.2 rule 8, `INV-11`) — and `crates/es-ir`
  and `crates/es-compile`, whose APIs are consumed and never edited. `crates/es` (the CLI) and
  `docs/ARCHITECTURE*.md` belong to other agents in this wave.
- A pickle path, in Rust or in Python, under any name (`INV-16`).
- Pre- or post-processing inside `PolicyRuntime` (spec 2.4, spec 8.7). If a lowering needs a
  tensor operation, it is an IR node or it does not exist.
- A new trait. `PolicyRuntime` is the one this packet is allowed (`INV-17`); `InferenceBackend`
  stays an enum until a second implementation exists.
- `tch`, `ort`, `safetensors`, `base64`, `ndarray` — the `libtorch`/ONNX backends are M2/M3, and
  the other three are each a few dozen lines that belong in the repository.
- `HashMap`/`HashSet` (spec 3.4), threads, global RNG.
- Re-implementing a pretrained backbone. `torchvision` is referenced, not copied.
- Diffusion/FlowMatching samplers, `PolicyBundle`, `LanguageEncoder`, batching, ONNX export —
  design note section 7 says why each waits.
