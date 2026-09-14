"""Optimize the module `es policy lower` emitted, and nothing else (M5 V2; spec 2.3, 8.1).

This script owns the optimizer and nothing above it. It `exec`s `<module dir>/es_policy.py` --
the file `es_policy::lower::lower_to_torch` generated from the bundle's `LearningGraph` -- and
instantiates `EsPolicy()`. It **defines no layer**: no layer constructor, no Module subclass, no
model zoo. The architecture comes from the Learning IR or it does not come, which is what makes
spec 1.4's "the same IR run in PyTorch is the ground truth" literally true here rather than
approximately true.

Nothing it reads from its own command line enters a hash slot (spec 8.1); the training run's
identity is `TrainingIdentity` (spec 19.2), which V2 does not populate.

Usage:

    train_act.py --module <dir> --dataset <root> --out model.safetensors
                 [--frames <dir>] [--epochs N] [--batch N] [--lr F] [--seed N] [--device cpu]
                 [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json]

Prints one JSON line on stdout and nothing else:

    {"initial_loss": f, "final_loss": f, "steps": n, ...}

Three things it deliberately does not do, all for the same reason the rest of the repo does not:

  * it reads and writes no format that can execute code on load (`INV-16`). The checkpoint it
    writes is safetensors, in the same layout `crates/es-policy/src/weights.rs` implements, so
    `es policy pack` and `torch_ref.py` both read it without the `safetensors` package being
    installed;
  * it does not use `lerobot.datasets.LeRobotDataset`: `lerobot` 0.6.1 refuses
    `codebase_version: "v2.1"` outright (design note section 7.5) and this repo writes v2.1. It
    reads the parquet files and `meta/` directly, with `pyarrow`;
  * it does no pre- or post-processing. Normalization, chunking and unnormalizing are IR nodes
    and are already inside `es_policy.py` (spec 8.7).

**The lowered module is single-sample.** `PolicyHead{Regression}` lowers to
`.reshape(horizon, action_dim)` and `ActionChunker` to `[:execute_chunk]`, neither of which
carries a batch axis -- spec 5.2 gives the inference domain its own batch size, so the IR has
none. `--batch N` is therefore N samples accumulated into one optimizer step: the same gradient
a batched forward would produce, one forward at a time.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch


def build_policy(module_dir: Path):
    """`exec` the generated module. This is the only code this script executes that it did not
    ship with, and it came from `es policy lower`, not from a checkpoint."""
    source = (module_dir / "es_policy.py").read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, str(module_dir / "es_policy.py"), "exec"), namespace)  # noqa: S102
    return namespace["EsPolicy"]()


def prod(shape) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return out


# --- the dataset LeRobotWriter wrote (spec 13.2, docs/api-notes/lerobot-dataset.md) ---------


def read_dataset(root: Path):
    """Per episode: the `observation.state` rows and the `action` rows, in frame order."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    template = info["data_path"]
    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states, actions = [], []
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        index = int(episode["episode_index"])
        path = root / template.format(
            episode_chunk=index // chunks_size, episode_index=index, video_key=""
        )
        table = pq.read_table(path)
        states.append(table.column("observation.state").to_pylist())
        actions.append(table.column("action").to_pylist())
    return states, actions


def read_frames(frames_dir: Path, count: int):
    """The raw tiles `es loop collect --frames` wrote: `<NNNNNN>.bin` plus a `.json` sidecar.

    One per recorded control step, in dataset frame order, so tile `i` is row `i` of the
    concatenated episodes. The sidecar is read, never assumed: a shape or dtype this does not
    expect is a refusal, because a wrongly-reshaped image trains a policy that looks fine.
    """
    first = json.loads((frames_dir / "000000.json").read_text(encoding="utf-8"))
    if first.get("dtype") != "u8" or len(first.get("shape", [])) != 3:
        raise SystemExit("%s: expected a 3-D u8 tile, got %r" % (frames_dir, first))
    h, w, c = (int(d) for d in first["shape"])
    tiles = torch.empty((count, h, w, c), dtype=torch.uint8)
    for i in range(count):
        raw = (frames_dir / ("%06d.bin" % i)).read_bytes()
        if len(raw) != h * w * c:
            raise SystemExit(
                "%s/%06d.bin: %d bytes for a %dx%dx%d tile" % (frames_dir, i, len(raw), h, w, c)
            )
        tiles[i] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(h, w, c)
    return tiles


def dequantize(tile: torch.Tensor) -> torch.Tensor:
    """HWC u8 -> CHW f32 / 255.

    This is `ObservationNode::Dequantize` (`es_compile::plan::Op::Dequantize`), which at
    inference runs inside the Observation IR's compiled plan. Training has no plan runner on
    the Python side, so the one op is re-implemented here; it is the only place in this repo
    where an IR node has a second implementation, and it is named in the design note as such.
    If the demo's Observation IR ever grows a second node between the image port and the
    Learning IR input, this stops being true and a real plan-bake step is needed.
    """
    return tile.permute(2, 0, 1).to(torch.float32) / 255.0


def plan_inputs(contract: dict, state_width: int, have_frames: bool):
    """Which graph input port is fed from where.

    An input port whose element count fits inside an `observation.state` row is fed from the
    front of that row -- which is the arm's own `qpos`, the joint state. Anything wider is an
    image: fed from `--frames` when there is one, and otherwise fed zeros and named in the
    output JSON, because a silently-zero input is the failure mode that looks like a trained
    policy. `es loop collect` writes pixels only with `--frames` (design note section 7.6).
    """
    fed, images, zero_filled = [], [], []
    for port, shape in sorted(contract["inputs"].items()):
        shape = [int(d) for d in shape]
        if prod(shape) <= state_width:
            fed.append((port, shape, prod(shape)))
        elif have_frames:
            images.append((port, shape))
        else:
            zero_filled.append((port, shape))
    return fed, images, zero_filled


def make_samples(states, actions, chunk: int):
    """One sample per frame: its `observation.state` row and the next `chunk` recorded actions.

    The last action is repeated when the episode ends inside the horizon; dropping those frames
    would drop exactly the part of the demonstration where the cube is released.
    """
    samples = []
    for state_rows, action_rows in zip(states, actions):
        n = len(state_rows)
        for t in range(n):
            target = [action_rows[min(t + k, n - 1)] for k in range(chunk)]
            samples.append((state_rows[t], target))
    return samples


# --- safetensors out (the same layout crates/es-policy/src/weights.rs reads) ----------------


def write_safetensors(path: Path, tensors: dict) -> None:
    header, data = {}, bytearray()
    for name in sorted(tensors):
        flat = tensors[name].detach().to(torch.float32).cpu().contiguous().reshape(-1)
        start = len(data)
        data.extend(struct.pack("<%df" % flat.numel(), *flat.tolist()))
        header[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [start, len(data)],
        }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(data))


def checkpoint_tensors(model) -> dict:
    """`n<node_id>.<rest>` -> `nodes.<node_id>.<rest>`: the inverse of `torch_ref.py`'s
    `to_state_dict`, so what is written is keyed exactly as `contract.json` declares."""
    out = {}
    for name, tensor in model.state_dict().items():
        if not name.startswith("n"):
            raise ValueError("parameter %r is not a lowered node member" % name)
        head, _, tail = name[1:].partition(".")
        out["nodes." + head + ("." + tail if tail else "")] = tensor
    return out


# --- the loop -------------------------------------------------------------------------------


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description="train the lowered Learning IR module")
    p.add_argument("--module", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument(
        "--frames",
        type=Path,
        help="the <NNNNNN>.bin tiles `es loop collect --frames` wrote, in dataset frame order",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--checkpoint-at",
        default="",
        help="comma-separated optimizer steps to also write a checkpoint at; the largest "
        "also caps the run",
    )
    p.add_argument("--loss-curve", type=Path, help="write the per-step loss as JSON")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    contract = json.loads((a.module / "contract.json").read_text(encoding="utf-8"))
    model = build_policy(a.module).to(device)

    # The chunk width is the module's own: `ActionChunker` slices the head's horizon down to
    # `execute_chunk`, and asking the module beats re-deriving it from the IR.
    model.eval()
    with torch.no_grad():
        probe = {
            port: torch.zeros([int(d) for d in shape], dtype=torch.float32, device=device)
            for port, shape in contract["inputs"].items()
        }
        out = model(**probe)
        if len(out) != 1:
            raise SystemExit("the lowered graph has %d outputs; V2 trains one" % len(out))
        chunk = int(next(iter(out.values())).shape[0])

    states, actions = read_dataset(a.dataset)
    if not states or not states[0]:
        raise SystemExit("%s holds no frames" % a.dataset)
    fed, images, zero_filled = plan_inputs(contract, len(states[0][0]), a.frames is not None)
    samples = make_samples(states, actions, chunk)
    tiles = read_frames(a.frames, len(samples)) if images else None
    if tiles is not None and len(tiles) != len(samples):
        raise SystemExit(
            "%s holds %d frames for %d dataset rows" % (a.frames, len(tiles), len(samples))
        )
    # One shared tensor per pixel-less port: every step would build the same zeros otherwise,
    # and 17k copies of a 96x96 image is a gigabyte of nothing.
    zeros = {
        port: torch.zeros(shape, dtype=torch.float32, device=device) for port, shape in zero_filled
    }

    marks = sorted({int(s) for s in a.checkpoint_at.split(",") if s.strip()})
    per_epoch = len(samples) // max(1, a.batch)
    total = max(marks) if marks else per_epoch * a.epochs
    if total <= 0:
        raise SystemExit("nothing to optimize: %d samples, batch %d" % (len(samples), a.batch))

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    generator = torch.Generator().manual_seed(a.seed)
    model.train()
    losses = []
    order, cursor = [], 0
    for step in range(total):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(a.batch):
            if cursor >= len(order):
                order = torch.randperm(len(samples), generator=generator).tolist()
                cursor = 0
            row = order[cursor]
            state_row, target_rows = samples[row]
            cursor += 1
            inputs = dict(zeros)
            for port, shape in images:
                inputs[port] = dequantize(tiles[row].to(device)).reshape(shape)
            for port, shape, count in fed:
                inputs[port] = torch.tensor(
                    state_row[:count], dtype=torch.float32, device=device
                ).reshape(shape)
            target = torch.tensor(target_rows, dtype=torch.float32, device=device)
            predicted = next(iter(model(**inputs).values()))
            loss = torch.nn.functional.l1_loss(predicted, target) / a.batch
            loss.backward()
            accumulated += float(loss.detach())
        optimizer.step()
        losses.append(accumulated)
        if marks and (step + 1) in marks:
            stem = str(a.out.with_suffix(""))
            write_safetensors(
                Path("%s-%d%s" % (stem, step + 1, a.out.suffix)), checkpoint_tensors(model)
            )

    write_safetensors(a.out, checkpoint_tensors(model))
    if a.loss_curve:
        a.loss_curve.write_text(json.dumps(losses), encoding="utf-8")

    # A single step's loss is noise; the reported pair is the mean of the first and last tenth
    # of the run, so "the loss fell" is a statement about the run and not about one draw.
    window = max(1, len(losses) // 10)
    report = {
        "initial_loss": sum(losses[:window]) / window,
        "final_loss": sum(losses[-window:]) / window,
        "steps": len(losses),
        "samples": len(samples),
        "batch": a.batch,
        "chunk": chunk,
        "zero_filled_inputs": [port for port, _ in zero_filled],
        "image_inputs": [port for port, _ in images],
    }
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
