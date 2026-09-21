"""Turn torchvision's ImageNet ResNet weights into a named, hashed, licensed artifact.

Packet M7/T5; spec 19.3 (`base_model.lock` -- "the provenance of the pretrained backbone
directly affects the results, it must be included in the provenance; it is also the basis for
license tracking"), spec 2.5 (no network at instantiation), spec 5.3 (nothing outside the hash
chain), spec 29 licence row (owner decision 2026-09-15: torchvision's ImageNet ResNet18
weights, BSD-3, are allowed).

    python/es/fetch_backbone.py --arch resnet18 --out ~/artifacts/plan-v/m7-t5
                                [--expect <blake3>] [--repin]

writes two files under `--out`:

  resnet18-imagenet1k-v1.safetensors   every float tensor of torchvision's `state_dict`,
                                       under torchvision's own key names
  resnet18-imagenet1k-v1.lock.json     where they came from, what they hash to, and the
                                       licence they carry

This is the **one** place in the project where a pickle is read, and it is torchvision that
reads it, inside its own `load_state_dict_from_url`, on the learning path where Python is a
first-class dependency (spec 2.3). `es` never opens a `.pth`: INV-16 is about what this
project's loaders accept, and they accept safetensors. Everything downstream -- `es train`,
`es policy pack`, `train_act.py --init-backbone` -- reads the safetensors file this writes.

`--expect <blake3>` is the pin. `crates/es-data/src/training.rs` holds it as
`RESNET18_IMAGENET1K_V1_BLAKE3` and `es train` verifies every `base_model` against it, so a
run against different tensors than the ones this project measured is not something you can do
by accident. A mismatch here is refused by name and nothing is written; `--repin` writes
anyway and says, loudly, that the constant and the design note have to move with it.

`num_batches_tracked` is deliberately **not** written. It is an int64 count of the batches
upstream training saw, not a weight; `FrozenBatchNorm2d` -- what the lowering builds, and what
`lerobot.rs` builds -- has no such buffer, and the safetensors layout this project reads and
writes is F32 only (`crates/es-policy/src/weights.rs`). The lock file names it as dropped
rather than leaving the reader to notice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# The weights this project is allowed to use, one row per architecture. Adding a row is a
# licence decision (spec 29), which is why the table is explicit instead of `getattr`-ed.
ARCHS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
}

LICENSE = "BSD-3-Clause"
LICENSE_URL = "https://github.com/pytorch/vision/blob/main/LICENSE"

# Not a weight: an int64 batch counter with no `FrozenBatchNorm2d` buffer to land in.
DROPPED = "num_batches_tracked"


def write_safetensors(path: Path, tensors: dict) -> bytes:
    """The layout `crates/es-policy/src/weights.rs` reads, F32, keys sorted.

    Returns the bytes as well as writing them, so the caller hashes what is on disk.
    """
    header, data = {}, bytearray()
    for name in sorted(tensors):
        flat = tensors[name].detach().cpu().float().contiguous().reshape(-1)
        start = len(data)
        data.extend(flat.numpy().tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [start, len(data)],
        }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = struct.pack("<Q", len(blob)) + blob + bytes(data)
    path.write_bytes(payload)
    return payload


def blake3_of(payload: bytes) -> str:
    """blake3, the digest every hash slot in this project is (spec 5.3).

    `blake3` is an optional package here for the same reason it is in `train_act.py`: a
    missing one must say so rather than silently pin a different function.
    """
    try:
        import blake3 as blake3_module
    except ImportError:
        raise SystemExit(
            "fetch_backbone: the `blake3` package is not importable, and the pin this script "
            "writes is a blake3 digest. `python -m pip install blake3`, or set PYTHONPATH to "
            "an interpreter that has it."
        )
    return blake3_module.blake3(payload).hexdigest()


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", default="resnet18", choices=sorted(ARCHS))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--expect",
        default="",
        help="the pinned blake3 of the safetensors file this writes; a mismatch is refused",
    )
    p.add_argument(
        "--repin",
        action="store_true",
        help="write even when the hash differs from --expect, and say that the pin has to move",
    )
    a = p.parse_args(argv)

    import torch
    import torchvision

    enum_name, _, member = ARCHS[a.arch].partition(".")
    weights = getattr(getattr(torchvision.models, enum_name), member)
    # The one network access, and the one pickle read, in this project -- torchvision's own,
    # on the learning path (spec 2.3, INV-16).
    model = getattr(torchvision.models, a.arch)(weights=weights)

    tensors = {k: v for k, v in model.state_dict().items() if not k.endswith(DROPPED)}
    stem = "%s-imagenet1k-v1" % a.arch
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / (stem + ".safetensors")

    # Hashed before it is committed to: a file whose hash disagrees with the pin is never
    # left on disk under a name that says it is the pinned artifact.
    scratch = a.out / (stem + ".safetensors.partial")
    payload = write_safetensors(scratch, tensors)
    digest = blake3_of(payload)
    if a.expect and digest != a.expect:
        if not a.repin:
            scratch.unlink()
            raise SystemExit(
                "fetch_backbone: refusing to write %s.\n"
                "  pinned blake3   %s\n"
                "  computed blake3 %s\n"
                "The pin is `RESNET18_IMAGENET1K_V1_BLAKE3` in crates/es-data/src/training.rs "
                "and every `es train --recipe` naming a `base_model` is verified against it. "
                "Pass --repin to write these tensors anyway; the constant, "
                "docs/api-notes/torchvision.md and docs/design/learning-lowering.md section "
                "5.3 then have to move with them, in the same commit."
                % (target, a.expect, digest)
            )
        sys.stderr.write(
            "fetch_backbone: --repin: the blake3 moved from %s to %s. Move "
            "RESNET18_IMAGENET1K_V1_BLAKE3 and the design note in the same commit.\n"
            % (a.expect, digest)
        )
    scratch.replace(target)

    lock = {
        "source": "torchvision.models.%s" % ARCHS[a.arch],
        "torchvision": torchvision.__version__,
        "torch": torch.__version__,
        "url": weights.url,
        "sha256_upstream": upstream_sha256(weights.url),
        "blake3": digest,
        "file": target.name,
        "bytes": len(payload),
        "tensors": len(tensors),
        "dropped": "*." + DROPPED,
        "license": LICENSE,
        "license_url": LICENSE_URL,
    }
    lock_path = a.out / (stem + ".lock.json")
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(lock, sort_keys=True) + "\n")
    return 0


def upstream_sha256(url: str) -> str:
    """torchvision's own file hash, read off the cached `.pth` it just loaded.

    torchvision puts the first eight hex digits of the sha256 in the file name and checks them
    on download; the full digest is computed here from the cached bytes, because a provenance
    record that carries eight digits is not one anyone can verify. The file is hashed, never
    parsed -- no pickle is opened by this function.
    """
    import torch

    cached = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if not cached.is_file():
        return ""
    h = hashlib.sha256()
    with cached.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
