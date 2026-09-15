"""Optimize the module `es policy lower` emitted, and nothing else (M5 V2/V2b; spec 2.3, 8.1).

This script owns the optimizer and nothing above it. It `exec`s `<module dir>/es_policy.py` --
the file `es_policy::lower::lower_to_torch` generated from the bundle's `LearningGraph` -- and
instantiates `EsPolicy()`. It **defines no layer**: no layer constructor, no Module subclass, no
model zoo. The architecture comes from the Learning IR or it does not come, which is what makes
spec 1.4's "the same IR run in PyTorch is the ground truth" literally true here rather than
approximately true.

Nothing it reads from its own command line enters a hash slot (spec 8.1); the training run's
identity is `TrainingIdentity` (spec 19.2), which V2 does not populate.

Usage:

    train_act.py --module <dir> --baked <dir> --out model.safetensors
                 [--epochs N] [--batch N] [--lr F] [--seed N] [--device cpu]
                 [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json]
                 [--resident-gpu] [--amp bf16] [--compile]

Prints one JSON line on stdout and nothing else:

    {"initial_loss": f, "final_loss": f, "steps": n, ...}

**It implements no Observation IR node, and that is the point of V2b.** `--baked` is the output
of `es dataset bake --policy <bundle.esb> --frames <tiles> <dataset>`, which ran every recorded
frame through the *same* `CpuPlan` `es eval run` runs at inference (`es_eval::ObservationBake`).
V2 read the LeRobot parquet directly and re-implemented `Op::Dequantize` here, and fed the state
port the raw `observation.state` row -- while the demo's Observation IR puts a
`Normalize{Range -1..1}` on it. The policy trained on `q` and was evaluated on `(q + 1) / 2`,
and every success rate in design note section 7.8 is depressed by it (open question 11). Reading
the bake is not a convenience: it is the only way the two paths cannot disagree.

Two more things it deliberately does not do, both for the reason the rest of the repo does not:

  * it reads and writes no format that can execute code on load (`INV-16`). Both the checkpoint
    it writes and the baked set it reads are safetensors, in the layout
    `crates/es-policy/src/weights.rs` implements, so no package beyond `torch` is needed;
  * it does no pre- or post-processing. Normalization, chunking and unnormalizing are IR nodes
    and are already inside `es_policy.py` (spec 8.7) or inside the bake (spec 7.2).

**The lowered module is single-sample.** `PolicyHead{Regression}` lowers to
`.reshape(horizon, action_dim)` and `ActionChunker` to `[:execute_chunk]`, neither of which
carries a batch axis -- spec 5.2 gives the inference domain its own batch size, so the IR has
none. `--batch N` is therefore N samples accumulated into one optimizer step: the same gradient
a batched forward would produce, one forward at a time.

**The three speed flags, and which of them move the numbers** (packet M5/V5):

  * `--resident-gpu` moves the whole baked set onto `--device` once, instead of copying one
    sample per forward. It changes *where* a tensor lives and nothing else -- the batch order,
    the dtype and the arithmetic are untouched -- so the loss curve is **bit-identical** to the
    default path at the same `--seed`, which
    `crates/es-policy/tests/ir_training.rs::resident_gpu_does_not_move_the_loss` pins. It needs
    the baked set to fit in device memory; the size it moved is printed to stderr so a run that
    does not fit says why.
  * `--amp bf16` and `--compile` **do** change the bits: bf16 autocast rounds every matmul's
    inputs and `torch.compile` is free to fuse and reassociate. Both are opt-in for exactly
    that reason -- use them for a sweep, not for a run whose numbers are quoted. The gate that
    stays either way is the fp32 inference-equivalence check in `ir_training.rs`, and it runs
    on a checkpoint trained at the defaults.

**`--channel-weight`, and why it is a flag and not an IR node** (packet M5/V16): the loss is
an unweighted L1 over `chunk x action_dim`, so every action channel gets the same gradient
whatever it means. V16 measured the consequence on the demo: the gripper's "open now" frames
are 7 % of an episode and their observation is frozen, so the fit returns the conditional
median of the ramp and the jaw never crosses the angle at which it stalls on the cube
(design note section 7.24). `--channel-weight 5=5` multiplies channel 5's absolute error by 5.
It takes an **index**, not a name: which channel is the gripper is the scene's knowledge, and
this script owns the optimizer and nothing above it. All-ones is the default and the default
path is bit-identical to the unweighted one -- `(d * w).mean()` with `w = 1` is `l1_loss`.
Like `--batch` and `--lr` it enters no hash slot (spec 8.1), so it belongs in the run summary,
which is where it is printed.

`--batch` defaults to 8 and stays there: the design note's measured runs are at 8, and moving
the default would silently invalidate them. Raising it is a different run, not a faster one --
`--batch N` accumulates N samples per optimizer step, so N x fewer steps cover the same data,
and the convention is to scale the step with it linearly: `--batch 32 --lr 4e-4` for the
`--batch 8 --lr 1e-4` default. Neither the scaling nor the batch size enters a hash slot
(spec 8.1); both belong in whatever records the training run.
"""

from __future__ import annotations

import argparse
import contextlib
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


# --- safetensors, both ways (the layout crates/es-policy/src/weights.rs reads and writes) ----


def read_safetensors(path: Path) -> dict:
    """`name -> tensor`. F32 only, which is what the writer emits and what the bake refuses to
    depart from; any other dtype here would mean two readers of one format."""
    blob = path.read_bytes()
    size = struct.unpack_from("<Q", blob, 0)[0]
    header, base, out = json.loads(blob[8 : 8 + size]), 8 + size, {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != "F32":
            raise SystemExit("%s: %s is %s, not F32" % (path, name, entry["dtype"]))
        a, b = entry["data_offsets"]
        out[name] = torch.frombuffer(
            bytearray(blob[base + a : base + b]), dtype=torch.float32
        ).reshape(entry["shape"])
    return out


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


# --- the baked observation set (`es dataset bake`, spec 7.2, 19.2) ---------------------------


def read_baked(baked: Path, ports: dict) -> tuple:
    """Every episode's baked tensors, in manifest order, plus the total frame count.

    A contract input with no baked tensor is a refusal, not a zero-filled port: a silently-zero
    input is the failure mode that looks like a trained policy (design note section 7.6).
    """
    manifest = json.loads((baked / "manifest.json").read_text(encoding="utf-8"))
    missing = sorted(set(ports) - set(manifest["tensors"]))
    if missing:
        raise SystemExit(
            "%s bakes no tensor for %s; the module's inputs are %s and the bake produced %s. "
            "Re-run `es dataset bake` against the same bundle."
            % (baked, missing, sorted(ports), sorted(manifest["tensors"]))
        )
    episodes = []
    for entry in manifest["episodes"]:
        tensors = read_safetensors(baked / entry["file"])
        n = int(entry["frames"])
        for name, tensor in tensors.items():
            if tensor.shape[0] != n:
                raise SystemExit(
                    "%s: %s has %d rows for %d frames" % (entry["file"], name, tensor.shape[0], n)
                )
        episodes.append(tensors)
    return manifest, episodes


def make_samples(episodes: list, chunk: int) -> list:
    """One sample per frame: `(episode, t, [row indices of the next `chunk` actions])`.

    The last action is repeated when the episode ends inside the horizon; dropping those frames
    would drop exactly the part of the demonstration where the cube is released.
    """
    samples = []
    for index, tensors in enumerate(episodes):
        n = tensors["action"].shape[0]
        for t in range(n):
            samples.append((index, t, [min(t + k, n - 1) for k in range(chunk)]))
    return samples


# --- the loop -------------------------------------------------------------------------------


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description="train the lowered Learning IR module")
    p.add_argument("--module", required=True, type=Path)
    p.add_argument(
        "--baked",
        required=True,
        type=Path,
        help="the output of `es dataset bake`: the dataset run through the Observation IR",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="samples accumulated into one optimizer step; the default is 8 because the "
        "design note's measured runs are at 8. Scale --lr with it linearly",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW step; the convention with --batch is linear scaling, so --batch 32 "
        "goes with --lr 4e-4",
    )
    p.add_argument(
        "--channel-weight",
        action="append",
        default=[],
        metavar="INDEX=WEIGHT",
        help="scale one action channel's absolute error, e.g. --channel-weight 5=5; an index "
        "because channel names are the scene's knowledge, not this script's. Repeatable",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--resident-gpu",
        action="store_true",
        help="move the whole baked set onto --device once instead of per sample; the loss "
        "curve is bit-identical to the default path at the same --seed",
    )
    p.add_argument(
        "--amp",
        choices=["off", "bf16"],
        default="off",
        help="bf16 autocast over the forward and the loss; opt-in because it changes the bits",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the lowered module; opt-in because it changes the bits",
    )
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
    shapes = {port: [int(d) for d in shape] for port, shape in contract["inputs"].items()}
    model = build_policy(a.module).to(device)

    # The chunk width is the module's own: `ActionChunker` slices the head's horizon down to
    # `execute_chunk`, and asking the module beats re-deriving it from the IR.
    model.eval()
    with torch.no_grad():
        probe = {
            port: torch.zeros(shape, dtype=torch.float32, device=device)
            for port, shape in shapes.items()
        }
        out = model(**probe)
        if len(out) != 1:
            raise SystemExit("the lowered graph has %d outputs; V2 trains one" % len(out))
        chunk = int(next(iter(out.values())).shape[0])

    weights = torch.ones(contract["action_dim"], device=device)
    for item in a.channel_weight:
        index, _, value = item.partition("=")
        if not value or not index.strip().isdigit() or int(index) >= len(weights):
            raise SystemExit(
                "--channel-weight wants INDEX=WEIGHT with 0 <= INDEX < %d, got %r"
                % (len(weights), item)
            )
        weights[int(index)] = float(value)

    manifest, episodes = read_baked(a.baked, shapes)
    if a.resident_gpu:
        # Where the tensors live, not what they are: same dtype, same values, same order, so
        # the `.to(device)` in the loop below becomes a no-op and the loss does not move.
        moved = sum(t.numel() * t.element_size() for e in episodes for t in e.values())
        episodes = [{name: t.to(device) for name, t in e.items()} for e in episodes]
        sys.stderr.write("resident on %s: %.1f MiB\n" % (device, moved / (1 << 20)))
    samples = make_samples(episodes, chunk)
    if not samples:
        raise SystemExit("%s holds no frames" % a.baked)

    marks = sorted({int(s) for s in a.checkpoint_at.split(",") if s.strip()})
    per_epoch = len(samples) // max(1, a.batch)
    total = max(marks) if marks else per_epoch * a.epochs
    if total <= 0:
        raise SystemExit("nothing to optimize: %d samples, batch %d" % (len(samples), a.batch))

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    generator = torch.Generator().manual_seed(a.seed)
    # `model` stays the thing whose `state_dict` is written: `torch.compile` returns a wrapper
    # whose parameter names are prefixed, and `checkpoint_tensors` would not recognise them.
    forward = torch.compile(model) if a.compile else model
    amp = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if a.amp == "bf16"
        else contextlib.nullcontext()
    )
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
            index, t, rows = samples[order[cursor]]
            cursor += 1
            tensors = episodes[index]
            inputs = {
                port: tensors[port][t].to(device).reshape(shape) for port, shape in shapes.items()
            }
            target = tensors["action"][rows].to(device)
            with amp:
                predicted = next(iter(forward(**inputs).values()))
                # `(|d| * w).mean()` over [chunk, action_dim]; with every weight 1 this is
                # exactly `l1_loss`, so an unweighted run is bit-identical to V15's.
                loss = ((predicted - target).abs() * weights).mean() / a.batch
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
        # Which of the three speed flags were on, because two of them change the bits and a
        # number whose mode is not recorded is a number nobody can reproduce.
        "resident_gpu": a.resident_gpu,
        "amp": a.amp,
        "compiled": a.compile,
        "chunk": chunk,
        "channel_weight": [float(v) for v in weights.tolist()],
        "ports": sorted(shapes),
        "observation_hash": manifest.get("observation_hash"),
    }
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
