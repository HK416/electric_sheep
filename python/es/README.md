# `es` — Python authoring builder (spec 14.2)

Thin Python frontend over the `es_native` pyo3 extension (`crates/es-py`), which itself calls
the language-neutral Rust builder core. See `docs/design/python-builder.md` for the layering.

## The Rust side is canonical

A handful of small formulas in `builder.py` reimplement Rust logic locally instead of calling
into `es_native`, because the values are needed for client-side type inference before a node
has been built (an editor needs to know a port's shape before running the graph once):

- `stable_id(path)` mirrors `es_core::StableId::from_path` (`crates/es-core/src/id.rs`):
  `blake3(path)` truncated to 16 bytes, hex-encoded.
- `_feature_ty(dim, tokens)` and `_chunk_ty(horizon, action_dim)` mirror `feature()`/`chunk()`
  in `crates/es-ir/src/learning.rs`.

**Rust is the source of truth for all three.** If a value here ever disagrees with the Rust
implementation, the Rust implementation is right and this file is the one to fix. `python/es/
selfcheck.py` pins one golden vector per function, matching the literal pinned in
`crates/es-core/src/id.rs`'s `from_path_is_stable_and_distinct` test, so a drift on either side
fails loudly instead of producing IR that validates against the wrong bodies (M4 review S-14).

## Running the self-check

```
maturin develop --release --features python   # from this directory, once, to build es_native
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```

## `encode_video.py`

Unrelated to the builder above: `encode_video.py` is a standalone script (M5 V4, design note
`docs/design/visible-learning.md` sections 2.9, 9) that turns `es video mosaic`'s raw frame
output into an `.mp4`, with `cv2.VideoWriter` (`mp4v` fourcc -- the only codec that opens on the
oracle server; there is no `ffmpeg` binary there). It is the one Python step in that packet; `es
video mosaic` itself is pure Rust. Needs a Python with `opencv-python`:

```
<venv>/bin/python python/es/encode_video.py --frames <mosaic dir> --out demo.mp4 --fps 10
```

## `train_act.py`

Also unrelated to the builder: `train_act.py` is the optimizer half of the spec 2.3 training
split (M5 V2/V2b, design note `docs/design/visible-learning.md` sections 6, 7.6, 7.9). It is the
**only** Python in that packet — `es policy lower`, `es dataset bake` and `es policy pack` are
Rust and need no interpreter.

```
es policy lower --policy untrained.esb --out build/
es dataset bake --policy untrained.esb --out baked/ --frames tiles/ ds/
<venv>/bin/python python/es/train_act.py --module build/ --baked baked/ --out model.safetensors \
    [--epochs N] [--batch N] [--lr F] [--seed N] [--device cuda] \
    [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json] \
    [--resident-gpu] [--amp bf16] [--compile]
es policy pack --policy untrained.esb --weights model.safetensors --out trained.esb
```

It `exec`s the `build/es_policy.py` that `es policy lower` wrote — the module
`es_policy::lower::lower_to_torch` generated from the bundle's own `LearningGraph` — and
optimizes that and nothing else. **It defines no layer**: the architecture comes from the
Learning IR or it does not come, which is what makes spec 1.4's "the same IR run in PyTorch is
the ground truth" literally true rather than approximately true.
`crates/es-policy/tests/ir_training.rs` asserts both halves of that (a byte-equality check
against the lowering, and a source scan of this file).

It writes safetensors keyed exactly as `build/contract.json` declares, so `es policy pack` can
check every key and shape before admitting it into a bundle (spec 25.1). No format that can
execute code on load is read or written anywhere on this path (`INV-16`).

Three things it needs to be told about, all recorded in the design note:

- **it implements no Observation IR node.** `--baked` is the output of `es dataset bake`, which ran
  every recorded frame through the same `CpuPlan` `es eval run` runs at inference. V2 read the
  parquet with `pyarrow` and re-implemented `Op::Dequantize` here, and fed the state port the raw
  `observation.state` row while the Observation IR normalizes it — so the policy trained on one
  observation and was evaluated on another (section 7.9, open question 11). That is why this file
  now reads one format and nothing else;
- a contract input with no baked tensor is a refusal, not a zero-filled port, which retires the
  "silently zero" failure mode V2 had to warn about;
- the lowered module is single-sample, so `--batch N` accumulates N samples into one optimizer
  step rather than running one batched forward.

The three speed flags (M5 V5, design note section 7.11) split by whether they move the numbers:
`--resident-gpu` moves the baked set onto `--device` once instead of per sample and is
**bit-identical** to the default path at the same `--seed`; `--amp bf16` and `--compile` change the
bits and are opt-in for that reason. `--batch` stays at 8 by default because the design note's
measured runs are at 8 — raise it with a linearly scaled `--lr` (`--batch 32 --lr 4e-4`).

Needs a Python with `torch` and `torchvision`. `pyarrow` is no longer read here — `es dataset
bake` does the dataset reading, in Rust.
