# PyTorch — pinned API surface for `TorchRuntime`

What `crates/es-policy` actually calls, so a torch upgrade is a diff against this file rather
than an archaeology session. Companion to `docs/api-notes/mujoco.md`, same shape, same reason.

Spec: spec 1.4 (PyTorch is the Learning IR's reference oracle), spec 2.4 (`TorchRuntime` is the
M1 backend), spec 8.7 (lowering), spec 8.9 (tier-4 equivalence). The lowering contract itself is
`docs/design/learning-lowering.md`.

## Version

| | |
|---|---|
| verified against | **torch 2.14.0+cpu** (`--index-url https://download.pytorch.org/whl/cpu`) |
| Python | 3.12.10 |
| `numpy` | **not installed, not required** — see "No numpy" below |
| `safetensors` package | **not required** — the reader is 15 lines of `struct` + `json` |
| `torchvision` | required **only** for a graph with a `VisionEncoder`; not exercised by CI yet |
| `diffusers` | **0.40.0**, required only by `python/ddpm_ref_check.py` (the DDPM/DDIM oracle); pure Python, `pip install diffusers` |

Install for a local run:

```
python -m venv <short path>            # a deep path fails on Windows without long-path support
<venv>/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
ES_PYTHON=<venv>/Scripts/python cargo test -p es-policy
```

`ES_PYTHON` is the same variable the `MuJoCo` oracle uses, so one venv can serve both. Without
it, `python` then `python3` are tried in order; with no `torch` anywhere, the equivalence test
prints `SKIPPED` and passes, and every other test still runs.

## `diffusers` — the independent DDPM/DDIM oracle

`python/ddpm_ref_check.py` is a second, smaller script with the same stdin/stdout JSON shape.
It exists because the Rust reference of `src/reference.rs` mirrors the lowering, so it cannot
catch a wrong *schedule* (M2 review, `crates/es-policy/src/reference.rs:26`). What it calls:

| symbol | used for |
|---|---|
| `diffusers.DDPMScheduler(num_train_timesteps, beta_schedule, variance_type, clip_sample, clip_sample_range, prediction_type)` | the schedule under test |
| `diffusers.DDIMScheduler(...)` | same, minus `variance_type` (it has none; eta = 0 uses no variance) |
| `.set_timesteps(n)` / `.timesteps` / `.alphas_cumprod` | the subsample and the cumulative alphas |
| `._get_variance(t)` / `._get_variance(t, prev_t)` | per-step variance; private, and the one place a version bump is likely to bite |
| `.step(model_output, t, sample).prev_sample` | the whole sampler loop, for the second test |
| `diffusers.schedulers.scheduling_ddpm.randn_tensor` | monkeypatched to hand DDPM the checkpoint's `noise_<t>` buffer instead of a fresh draw, so both sides consume the same numbers (spec 3.4) |

Two shapes worth knowing: `DDPMScheduler.step` reads `model_output.shape[1]` to detect a
learned variance, so it needs a batch axis the lowered head does not have (the script
reshapes); and `_get_variance` is private API, so a `diffusers` upgrade is a diff against this
table. No `diffusers` -> both tests print `SKIPPED` and pass, as with `torch`.

## The Python API this depends on

Everything is in `crates/es-policy/python/torch_ref.py` and in the generated module.

**Called by the reference process**

| API | used for | notes |
|---|---|---|
| `torch.__version__` | `PolicyInfo::version`, and via it `runtime_hash` | a string; changing it changes the hash chain, which is the intent (spec 5.3) |
| `torch.tensor(list, dtype=torch.float32)` | every tensor that crosses the wire or comes out of a checkpoint | list-of-float construction, not `frombuffer`: no buffer-protocol or numpy dependency |
| `Tensor.reshape(shape)` | applying a declared shape | |
| `Tensor.detach().to(torch.float32).contiguous().reshape(-1).tolist()` | encoding an output | `tolist` widens f32 to Python float exactly, and `struct.pack("<f")` narrows it back to the same bits |
| `nn.Module.load_state_dict(sd, strict=True)` | loading weights | `strict=True` on purpose: a silently partial load is the failure mode this whole packet exists to prevent |
| `nn.Module.eval()` | inference mode | matters for `nn.TransformerEncoderLayer`'s dropout |
| `torch.inference_mode()` | the forward pass | |
| `compile(source, "<es-policy>", "exec")` + `exec` | instantiating the generated module | the *only* code this process executes, and it arrives over the pipe, never from a checkpoint |

**Used by generated modules** (`crates/es-policy/src/lower/torch.rs`)

`nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `nn.Identity`, `nn.TransformerEncoder`,
`nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)`, `nn.Module.register_buffer(...,
persistent=False)`, `torch.cat(tensors, dim=)`, `Tensor.unsqueeze/squeeze/reshape`, slicing.
For vision: `torchvision.models.resnet18` / `resnet34`, whose `.fc` is replaced with an
`nn.Linear`.

**Deliberately not called**

`torch.load`, `torch.save`, `pickle` in any form (`INV-16`); `torch.compile`, `torch.jit`,
`torch.onnx` (M2 owns ONNX); CUDA anything (the oracle is CPU by definition).

### No numpy

The CPU wheel does not pull `numpy` in, and torch prints a `Failed to initialize NumPy` warning
on import when it is absent. It goes to stderr, which the Rust side pipes to null. Nothing here
needs it: tensors are built from Python lists and serialized with `struct`. Keeping it out keeps
the CI venv to one wheel.

## Wire protocol

Line-delimited JSON, one request per line in, one JSON object per line out — the same shape as
`crates/es-physics-backend/python/mujoco_ref.py`. Protocol version **1**
(`es_policy::torch_runtime::PROTOCOL_VERSION`), which is hashed into `runtime_hash`.

```
-> {"cmd": "load",  "source": "<generated python>", "weights_path": "<path>.safetensors"}
<- {"ok": true, "torch_version": "2.14.0+cpu", "protocol": 1}

-> {"cmd": "infer", "inputs": {"joint_state": {"shape": [4], "dtype": "F32",
                                               "data_b64": "…"}}}
<- {"ok": true, "outputs": {"actions": {"shape": [3, 2], "dtype": "F32", "data_b64": "…"}}}

-> {"cmd": "quit"}                      (no reply; the process exits)
```

Any failure is `{"ok": false, "error": "<ExceptionType>: <message>"}` — a modelling mistake is a
typed `PolicyError`, never a dead process or a traceback on stderr.

**Tensor encoding.** `data_b64` is standard base64 (RFC 4648, padded) of the tensor's raw
little-endian f32 bytes, row-major, no batch axis (spec 5.4). `dtype` uses safetensors' spelling
(`"F32"`) so one name means one thing on both sides of the pipe; anything else is rejected.
JSON numbers were the alternative and were rejected: a `[50, 8]` chunk of them is about 20x the
bytes, and it invites a shortest-round-trip argument at the exact place where spec 8.9 wants a
1e-5 comparison.

**Statefulness.** `load` must precede `infer`; a second `load` replaces the model. The process
holds exactly one model, so `TorchRuntime` holds exactly one process.

## Safetensors

The format, in full, as both sides implement it:

```
[0 ..  8)        header length N, u64 little-endian
[8 .. 8+N)       header, UTF-8 JSON object
[8+N .. )        data segment, raw tensor bytes
```

Each header entry is `"<key>": {"dtype": "F32", "shape": [..], "data_offsets": [start, stop]}`,
with the offsets relative to the **start of the data segment**, not the file. The optional
`"__metadata__"` key maps to an object of strings and is not a tensor; both readers skip it.
Rust writes the header from a `BTreeMap`, so the bytes are a function of the content alone
(spec 3.4) and `weights_hash` is stable.

Keys are `nodes.<node_id>.<param path>` (design note section 4); the Python side renames them to
`n<node_id>.<param path>` to match the generated module's members, and `load_state_dict` is
strict, so a mismatch that got past the Rust-side check still fails loudly.

This project reads and writes the format itself rather than depending on the `safetensors`
crate or package. It is ~40 lines each way, it keeps the CI venv to one wheel, and — the actual
reason — the format is a load-bearing part of `INV-16`, so it is worth having in the repository
where it can be read.

## Known gaps

- `torchvision` is referenced by the lowering but no test exercises it; the M1 ACT-checkpoint
  gate (spec 8.9) is where that lands.
- A real LeRobot ACT checkpoint needs a key remap onto this scheme — see the design note,
  section 5.
- `prediction_type` other than `"epsilon"` is `LowerError::Unsupported`: the lowered denoiser
  is `eps_theta`, and `"sample"` / `"v_prediction"` are a different `pred_original_sample`.
- `timestep_spacing` is pinned to `diffusers`' `"leading"` default; `"linspace"` / `"trailing"`
  are not modeled, and neither are `steps_offset != 0`, `rescale_betas_zero_snr`, `thresholding`
  or `DDIMScheduler`'s `eta > 0`.
- No batching. The generated module is written for one sample; `Linear` broadcasts a leading
  axis anyway, but `PolicyHead`'s `reshape(H, A)` pins the convention and would need changing.
- Startup cost is a whole Python interpreter plus a torch import, roughly a second. This is the
  oracle, not a throughput path (spec 2.4); `libtorch` FFI is the same trait with a different
  body.
