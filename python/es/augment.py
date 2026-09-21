"""The Observation IR's `training_only` augmentation, applied by the trainer (spec 7.3, 3.4).

This is the training half of a boundary the compiler draws. `es dataset bake --for-training`
writes the tensor *entering* a port's chain of `training_only` `Augment` nodes -- for the demo
that is the `104x104` canvas a `Pad(4)` produced -- and records the chain in
`training/augmentation.json`. This file is what applies that chain, per sample and per
optimizer step. It invents nothing: the nodes, their parameters and their order all come from
the document, and the evaluation path (`CpuPlan` in `PlanMode::Release`) disables every one of
them structurally, which is INV-15.

**The draws are addressed, not stepped** (spec 3.4: no global RNG). Every value comes from the
counter-based mixer `crates/es-render/src/rng.rs` uses -- Murmur3's `fmix32`, ten lines of
integer arithmetic that Rust, Slang and Python agree on with no floating point anywhere in the
mixer -- keyed by

    (augmentation_seed, sample_index, step, node_index, draw)

so a draw is a pure function of where it is used. `torch.Generator` is deliberately **not**
used: its stream is an implementation detail of a torch version, so a run reproduced on
another torch would silently see different augmentation, and `augmentation.json` would be a
description of something that did not happen. `crates/es-policy/tests/ir_training.rs`
re-implements the ten lines in Rust and compares both against
`tests/golden/train/augment_seed0.json`, bitwise at `f32`.

What is implemented, and what is refused by name (design note `training-recipe.md` 13):

  * `RandomCrop {width, height}`   -- uniform integer offset in `[0, W-w] x [0, H-h]`
  * `ColorJitter {brightness, contrast}` -- `x * (1 + u*b)`, then `(x - mean) * (1 + u*c) + mean`
  * `ColorJitter {saturation, hue}` -- refused: both need a colour model this file does not
    have (`hue` is an HSV rotation), and a silently ignored parameter is worse than a refusal
  * `GaussianNoise {sigma}`        -- additive, Box-Muller over two uniforms per element
  * `RandomErasing`                -- refused: not implemented by packet M7/T6

The scalars are computed in `f64` and rounded to `f32` *before* they touch a tensor, and every
per-element operation is then a single `f32` op with both operands exactly representable --
so the result does not depend on whether a kernel widened an intermediate. See section 13 of
the design note for what that buys and where it stops (the contrast mean's reduction order).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch

MASK = 0xFFFFFFFF
#: `uniform` keeps the top 24 bits, so every draw is exact in `f32` and never reaches 1.0.
SCALE = 1.0 / 16777216.0
#: Weyl-ish stride between draws of one stream, as in `es_render::rng::uniform`.
STRIDE = 0x9E3779B9


def mix32(z):
    """Murmur3's `fmix32`. Takes a Python int or an integer tensor, returns the same kind.

    The masks are what make an `int64` tensor and an arbitrary-precision Python int agree:
    both keep the low 32 bits of every product, which is what the Rust `u32` does by wrapping.
    """
    z = z & MASK
    z = z ^ (z >> 16)
    z = (z * 0x85EBCA6B) & MASK
    z = z ^ (z >> 13)
    z = (z * 0xC2B2AE35) & MASK
    return (z ^ (z >> 16)) & MASK


def key(seed: int, sample: int, step: int, node: int) -> int:
    """The stream for one `(seed, sample, step, node)`.

    The seed is a `u64` (spec 19.3's `seed.json`) and the mixer is 32-bit, so both halves are
    folded in rather than truncated -- two seeds differing only above bit 32 are two streams.
    """
    k = mix32(seed & MASK)
    k = mix32(k ^ ((seed >> 32) & MASK))
    k = mix32(k ^ (sample & MASK))
    k = mix32(k ^ (step & MASK))
    return mix32(k ^ (node & MASK))


def uniform(k, i):
    """Draw `i` of stream `k`, in `[0, 1)`. `f64` here, exact as an `f32` value."""
    bits = mix32(k ^ ((i * STRIDE) & MASK))
    if torch.is_tensor(bits):
        return (bits >> 8).double() * SCALE
    return float(bits >> 8) * SCALE


def f32(value: float) -> float:
    """`value` rounded to the nearest `f32`, as a Python float.

    Every scalar crosses this before it meets a tensor, so a kernel that computes a
    `tensor * scalar` in double and rounds once gives the same bits as one that does not.
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_chains(path) -> tuple:
    """`(chains, seed)` from `training/augmentation.json`; `({}, 0)` for `{"kind": "none"}`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("kind") != "observation-ir":
        return {}, 0
    return doc.get("chains", {}), int(doc.get("seed", 0))


def _refuse(what: str) -> None:
    raise SystemExit(
        "--augmentation: %s. The document declares it and this trainer does not implement "
        "it; packet M7/T6 lists what is implemented (docs/design/training-recipe.md section "
        "13). Remove the node or extend python/es/augment.py -- never train as if it were "
        "not there." % what
    )


def _random_crop(node: dict, x, samples: list, step: int, seed: int):
    w, h = int(node["width"]), int(node["height"])
    n, _, height, width = x.shape
    if w > width or h > height:
        _refuse("RandomCrop %dx%d leaves a %dx%d tensor" % (w, h, width, height))
    out = torch.empty(x.shape[:2] + (h, w), dtype=x.dtype, device=x.device)
    for row in range(n):
        k = key(seed, samples[row], step, int(node["node"]))
        ox = min(int(uniform(k, 0) * (width - w + 1)), width - w)
        oy = min(int(uniform(k, 1) * (height - h + 1)), height - h)
        out[row] = x[row, :, oy : oy + h, ox : ox + w]
    return out


def _color_jitter(node: dict, x, samples: list, step: int, seed: int):
    if float(node.get("saturation", 0.0)) != 0.0 or float(node.get("hue", 0.0)) != 0.0:
        _refuse("ColorJitter sets `saturation` or `hue`")
    brightness, contrast = float(node["brightness"]), float(node["contrast"])
    out = x.clone()
    for row in range(x.shape[0]):
        k = key(seed, samples[row], step, int(node["node"]))
        gain = f32(1.0 + (2.0 * uniform(k, 0) - 1.0) * brightness)
        plane = out[row] * gain
        # The mean of the brightened image, in f64, rounded once to f32 before it is used --
        # the one reduction in this file (design note section 13 records its ceiling).
        mean = f32(float(plane.double().mean()))
        gain_c = f32(1.0 + (2.0 * uniform(k, 1) - 1.0) * contrast)
        out[row] = (plane - mean) * gain_c + mean
    return out


def _gaussian_noise(node: dict, x, samples: list, step: int, seed: int):
    sigma = float(node["sigma"])
    out = x.clone()
    for row in range(x.shape[0]):
        k = key(seed, samples[row], step, int(node["node"]))
        count = out[row].numel()
        draws = torch.arange(2 * count, dtype=torch.int64, device=x.device)
        u = uniform(k, draws)
        # `1 - u` rather than `u`: the draw is in [0, 1), so this one is in (0, 1] and the
        # logarithm is always defined. Box-Muller's first uniform, f64 on both sides.
        radius = torch.sqrt(-2.0 * torch.log(1.0 - u[0::2]))
        angle = torch.cos(2.0 * math.pi * u[1::2])
        out[row] = out[row] + (sigma * radius * angle).float().reshape(out[row].shape)
    return out


KINDS = {
    "RandomCrop": _random_crop,
    "ColorJitter": _color_jitter,
    "GaussianNoise": _gaussian_noise,
}


def apply_chain(chain: list, x, samples: list, step: int, seed: int):
    """Apply one port's chain to a batch `x` of `len(samples)` samples.

    `samples` are the *global* sample indices of the batch -- the index into the trainer's own
    sample list, not the position in the batch -- so a sample seen again at another step draws
    a different value, and the same sample at the same step always draws the same one.
    """
    for node in chain:
        kind = node.get("kind")
        if kind not in KINDS:
            _refuse("%s is not implemented" % kind)
        x = KINDS[kind](node, x, samples, step, seed)
    return x


def demo() -> None:
    """A self-check with no golden and no dataset: the properties the key table promises."""
    assert mix32(0) == 0 and mix32(1) != mix32(2)
    a = key(0, 3, 7, 8)
    assert uniform(a, 0) == uniform(a, 0) and uniform(a, 0) != uniform(a, 1)
    for other in [key(1, 3, 7, 8), key(0, 4, 7, 8), key(0, 3, 8, 8), key(0, 3, 7, 9)]:
        assert other != a, "a coordinate does not separate the stream"
    assert 0.0 <= uniform(a, 5) < 1.0

    x = torch.arange(2 * 3 * 6 * 6, dtype=torch.float32).reshape(2, 3, 6, 6) / 100.0
    chain = [
        {"node": 8, "kind": "RandomCrop", "width": 4, "height": 4},
        {"node": 9, "kind": "ColorJitter", "brightness": 0.2, "contrast": 0.2,
         "saturation": 0.0, "hue": 0.0},
        {"node": 10, "kind": "GaussianNoise", "sigma": 0.05},
    ]
    once = apply_chain(chain, x, [0, 1], 0, 0)
    assert tuple(once.shape) == (2, 3, 4, 4), once.shape
    assert torch.equal(once, apply_chain(chain, x, [0, 1], 0, 0)), "a draw is not addressed"
    assert not torch.equal(once, apply_chain(chain, x, [0, 1], 1, 0)), "the step is ignored"
    assert not torch.equal(once, apply_chain(chain, x, [0, 1], 0, 1)), "the seed is ignored"
    assert not torch.equal(once, apply_chain(chain, x, [2, 3], 0, 0)), "the sample is ignored"
    print("augment.py demo: ok")


if __name__ == "__main__":
    demo()
