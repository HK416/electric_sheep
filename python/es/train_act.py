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
                 [--schedule constant|warmup_cosine] [--warmup-steps N] [--lr-min F]
                 [--weight-decay F] [--grad-clip F] [--init-backbone weights.safetensors]
                 [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json]
                 [--resident-gpu] [--amp bf16] [--compile]

Prints one JSON line on stdout and nothing else:

    {"initial_loss": f, "final_loss": f, "steps": n, "torch": "...",
     "optimizer": {"kind": "AdamW", "lr": f, "betas": [f, f], ...}, ...}

`es train` reads that line (packet M7/T1): `torch` goes into spec 19.3's `hardware.json`, and
`optimizer` is compared against the `optimizer.json` it declared *before* the run, so the
declaration is checked rather than believed. Nothing this script is told enters a hash slot.

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

**The lowered module takes a batch axis** (packet M7/T3). `forward(**inputs)` wants every input
with a leading `N` and returns the chunk as `[N, execute_chunk, action_dim]`, which
`contract.json`'s `"batch_axis": true` declares; the shapes under `"inputs"` stay per-sample,
because spec 5.2 keeps the batch out of the IR and gives each domain its own size. `--batch N`
is therefore one forward over N stacked samples and one `L1` mean over `[N, K, A]` -- not N
single-sample forwards accumulated, which is what it was through M5 and what made 20,000 steps
kernel-launch bound (spec 28.9 rung 9). The sample order is unchanged: the permutation is still
drawn once per epoch with the same generator, so batch `i` holds exactly the samples the
accumulation loop would have visited. The *numbers* move by one sum order -- `mean` over the
whole batch instead of a sum of per-sample means -- so a loss curve is comparable to M5's in
shape, not bit for bit.

**The three speed flags, and which of them move the numbers** (packet M5/V5):

  * `--resident-gpu` moves the whole baked set onto `--device` once, instead of copying one
    batch per step. It changes *where* a tensor lives and nothing else -- the batch order,
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
the default would silently invalidate them. Raising it is a different run, and now a cheaper
one per sample -- `--batch N` covers N samples per optimizer step, so N x fewer steps cover the
same data. The convention of scaling the step linearly with it (`--batch 32 --lr 4e-4` for the
`--batch 8 --lr 1e-4` default) was measured to diverge at 64 (design note
`visible-learning.md` section 7.11), which is why `--schedule` exists below. Neither the
scaling nor the batch size enters a hash slot (spec 8.1); both belong in whatever records the
training run -- for `es train` that is spec 19.3's `optimizer.json` and `scheduler.json`.

**The learning-rate schedule** (packet M7/T4). `--schedule constant` is the default and is the
old run exactly: `lr_at` is not consulted, the optimizer keeps the step it was built with, and
the loss curve of a run that passes no new flag is bit-identical to the one from before this
packet. `--schedule warmup_cosine` applies

    lr(step) = lr * step / warmup                                       step < warmup
             = lr_min + (lr - lr_min) * 0.5 * (1 + cos(pi * (step - warmup)
                                                       / (total - warmup)))   otherwise

where `total` is the number of optimizer steps this invocation will run. It is the plain
function `lr_at` below, called once per step and written into `param_groups`;
`torch.optim.lr_scheduler` is deliberately not used, because its float sequence is an
implementation detail of a torch version and the schedule has to be reproducible across them.
`tests/golden/train/lr_warmup_cosine.json` pins the first 1,000 values, and
`crates/es-policy/tests/ir_training.rs::lr_schedule_matches_the_golden` compares both this
function and a Rust re-implementation of it against that file.

**The pretrained backbone** (packet M7/T5). `--init-backbone <file>` loads a safetensors file
whose keys are the backbone's own `state_dict` names into every lowered backbone before the first step, mapping
`<key>` to `n<node>.<key>`; `fc` is skipped because the lowering replaced it, and any other
key that is unknown or the wrong shape stops the run. It is a *checkpoint like any other* --
this script still reads and writes nothing that can execute code on load (INV-16), and still
never touches the network. The file is `es train`'s `[policy] base_model`, whose blake3 was
checked against its lock file and against the repository's pin before this script saw it; its
provenance is spec 19.3's `training/base_model.lock`, not anything here.

`frozen` never appears on this command line. It is a field of the Learning IR, so the lowered
module carries it as `requires_grad_(False)` and the optimizer is built over
`[p for p in model.parameters() if p.requires_grad]`. A flag would be a second copy of an IR
decision, and two copies of one fact are one fact and one bug. With nothing frozen the list
is every parameter in the same order, so the default path is bit-identical to the run before
this packet.

`--weight-decay` is AdamW's, defaulting to torch's own `1e-2` **made explicit** so that
`optimizer.json` can name a number this script actually passed rather than one it assumes.
`--grad-clip F` clips the gradient norm to `F` before the step; `0` (the default) is off, and
a run that clipped says so in its summary. The applied learning rates are hashed into
`lr_curve_hash` (blake3 over the `f64` values, little-endian) so that two runs of one schedule
can be told apart by one word.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import struct
import sys
from pathlib import Path

import torch

try:  # only `lr_curve_hash` needs it, and this script's contract is "no package beyond torch"
    import blake3
except ImportError:  # pragma: no cover - reported as an unset hash, never a stopped run
    blake3 = None


def lr_at(step: int, total: int, lr: float, lr_min: float, warmup: int) -> float:
    """The `warmup_cosine` schedule of packet M7/T4, as a plain function of five numbers.

    Not `torch.optim.lr_scheduler`: that sequence is an implementation detail of a torch
    version, and this one is pinned bitwise by `tests/golden/train/lr_warmup_cosine.json`
    against a Rust re-implementation of these same three lines. Keep the arithmetic in this
    order -- the golden is `f64` and the order is what makes the two agree.
    """
    if warmup > 0 and step < warmup:
        return lr * step / warmup
    span = max(1, total - warmup)
    return lr_min + (lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / span))


def lr_curve_hash(applied: list) -> str:
    """blake3 over the learning rates actually applied, as little-endian `f64`.

    `None` when `blake3` is not importable: it is provenance, and a missing optional package
    must not stop a training run that torch alone can finish.
    """
    if blake3 is None:
        return None
    return blake3.blake3(struct.pack("<%dd" % len(applied), *applied)).hexdigest()


def init_backbone(model, tensors: dict) -> list:
    """Load `--init-backbone`'s tensors into every lowered `ResNet` backbone (M7/T5).

    The file holds the backbone's own `state_dict` key names (the provider's, not this project's), so the mapping is
    `n<node>.<key>`; `fc` is skipped by name because the lowering replaced it with
    `Linear(512, out_dim)`, and any *other* key that is unknown or the wrong shape is a
    refusal -- it would mean this file is not this backbone, and a partly-initialised encoder
    is the failure mode that still trains and still looks fine.
    """
    members = [(n, m) for n, m in model.named_children() if type(m).__name__ == "ResNet"]
    if not members:
        raise SystemExit(
            "--init-backbone: the lowered module has no `ResNet` backbone to initialise. "
            "Its Learning IR needs a `VisionEncoder { pretrained = true }`; `es train` "
            "refuses this pairing before it gets here."
        )
    fit = {k: v for k, v in tensors.items() if not k.startswith("fc.")}
    report = []
    for name, member in members:
        own = member.state_dict()
        unknown = sorted(set(fit) - set(own))
        wrong = sorted(k for k in fit if k in own and tuple(own[k].shape) != tuple(fit[k].shape))
        if unknown or wrong:
            raise SystemExit(
                "--init-backbone: the file does not fit %s: %d unknown key(s) %s, "
                "%d shape disagreement(s) %s"
                % (name, len(unknown), unknown[:4], len(wrong), wrong[:4])
            )
        member.load_state_dict(fit, strict=False)
        report.append(
            {
                "member": name,
                "loaded": len(fit),
                # `fc` and nothing else: named, so a reader never has to guess which tensors
                # of this backbone the run started from and which it drew at random.
                "from_scratch": sorted(set(own) - set(fit)),
                "frozen": not any(p.requires_grad for p in member.parameters()),
            }
        )
    return report


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
        "--schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help="constant is the default and is the old run exactly: `lr_at` is not consulted "
        "and the optimizer keeps the step it was built with",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linear warmup from 0 to --lr over this many steps, before the cosine",
    )
    p.add_argument(
        "--lr-min", type=float, default=0.0, help="the floor the cosine decays to"
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW's; torch's own default, made explicit so optimizer.json can name it",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="clip the gradient norm to this before the step; 0 is off",
    )
    p.add_argument(
        "--channel-weight",
        action="append",
        default=[],
        metavar="INDEX=WEIGHT",
        help="scale one action channel's absolute error, e.g. --channel-weight 5=5; an index "
        "because channel names are the scene's knowledge, not this script's. Repeatable",
    )
    p.add_argument(
        "--init-backbone",
        type=Path,
        help="a safetensors file whose keys are a backbone's own `state_dict` names to load into every lowered "
        "backbone before the first step; `es train` passes the `base_model` its recipe "
        "names, after verifying its blake3 against the lock file and the pin",
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
    if not contract.get("batch_axis"):
        raise SystemExit(
            "%s/contract.json does not declare batch_axis: it was written by a lowering that "
            "emits a single-sample module (before packet M7/T3). Re-run `es policy lower`."
            % a.module
        )
    model = build_policy(a.module).to(device)
    # Before the probe and before the optimizer: the ImageNet tensors are the module's
    # initial state, exactly as a resumed checkpoint's would be (packet M7/T5).
    backbones = init_backbone(model, read_safetensors(a.init_backbone)) if a.init_backbone else []

    # The chunk width is the module's own: `ActionChunker` slices the head's horizon down to
    # `execute_chunk`, and asking the module beats re-deriving it from the IR. The probe is one
    # sample *with* the batch axis, so axis 0 is the batch and axis 1 is the chunk.
    model.eval()
    with torch.no_grad():
        probe = {
            port: torch.zeros([1] + shape, dtype=torch.float32, device=device)
            for port, shape in shapes.items()
        }
        out = model(**probe)
        if len(out) != 1:
            raise SystemExit("the lowered graph has %d outputs; V2 trains one" % len(out))
        chunk = int(next(iter(out.values())).shape[1])

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

    # `requires_grad` is where the Learning IR's `frozen` arrives: the lowering emits
    # `requires_grad_(False)` on a `VisionEncoder { frozen = true }`, so `nodes.<k>.*` is out
    # of the optimizer without any flag on this command line carrying a copy of the IR's
    # decision (packet M7/T5). With no frozen node this is every parameter, in the same
    # order, so the default path is the run of before, bit for bit.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("every parameter is frozen; there is nothing to optimize")
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay)
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
    losses, applied_lr = [], []
    order, cursor = [], 0
    for step in range(total):
        # `constant` writes back the number AdamW was built with, which is a no-op on the
        # arithmetic -- that is what makes the default path the old run (packet M7/T4).
        lr_now = (
            a.lr
            if a.schedule == "constant"
            else lr_at(step, total, a.lr, a.lr_min, a.warmup_steps)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        applied_lr.append(lr_now)
        optimizer.zero_grad(set_to_none=True)
        # The same `a.batch` samples the accumulation loop would have visited, in the same
        # order, drawn from the same generator -- and now stacked into one forward (M7/T3).
        picked = []
        for _ in range(a.batch):
            if cursor >= len(order):
                order = torch.randperm(len(samples), generator=generator).tolist()
                cursor = 0
            picked.append(samples[order[cursor]])
            cursor += 1
        inputs = {
            port: torch.stack([episodes[i][port][t] for i, t, _ in picked])
            .to(device)
            .reshape([len(picked)] + shape)
            for port, shape in shapes.items()
        }
        target = torch.stack([episodes[i]["action"][rows] for i, _, rows in picked]).to(device)
        with amp:
            predicted = next(iter(forward(**inputs).values()))
            # `(|d| * w).mean()` over [batch, chunk, action_dim]; with every weight 1 this is
            # exactly `l1_loss`, and the batch mean is the average of the per-sample means the
            # accumulation loop summed -- the same gradient, one sum order later.
            loss = ((predicted - target).abs() * weights).mean()
        loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, a.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
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
    group = optimizer.param_groups[0]
    report = {
        "initial_loss": sum(losses[:window]) / window,
        "final_loss": sum(losses[-window:]) / window,
        "steps": len(losses),
        # The step a divergence started at, rather than a NaN at the end of the run and no
        # way to tell when it happened (packet M7/T4's acceptance asks for exactly this).
        "first_nonfinite_step": next(
            (i for i, v in enumerate(losses) if not math.isfinite(v)), None
        ),
        "samples": len(samples),
        "batch": a.batch,
        # One forward over `batch` samples, not `batch` forwards accumulated (packet M7/T3).
        # It is in the summary because it is what the loss numbers below are a sum order of.
        "batch_axis": True,
        # Which of the three speed flags were on, because two of them change the bits and a
        # number whose mode is not recorded is a number nobody can reproduce.
        "resident_gpu": a.resident_gpu,
        "amp": a.amp,
        "compiled": a.compile,
        # The schedule, and the identity of the sequence it actually applied (packet M7/T4).
        # `es train` writes the first five into spec 19.3's `scheduler.json` and
        # `optimizer.json` *before* the run; these are what the run says it used.
        "schedule": a.schedule,
        "warmup_steps": a.warmup_steps,
        "lr_min": a.lr_min,
        "weight_decay": a.weight_decay,
        "grad_clip": a.grad_clip,
        "lr_curve_hash": lr_curve_hash(applied_lr),
        "chunk": chunk,
        # What the run started from and what it was allowed to move (packet M7/T5). The
        # provenance of the file itself is `es train`'s `training/base_model.lock`; this is
        # the trainer's own account of what it did with it.
        "init_backbone": str(a.init_backbone) if a.init_backbone else None,
        "backbones": backbones,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "channel_weight": [float(v) for v in weights.tolist()],
        "ports": sorted(shapes),
        "observation_hash": manifest.get("observation_hash"),
        # For `es train`'s spec 19.3 slots (packet M7/T1): `optimizer.json` is written before
        # the run starts, from the values `AdamW(params, lr=lr)` leaves at torch's defaults,
        # and `hardware.json` records the library the run actually used. Reporting them here
        # is what lets `es train` compare its declaration against reality instead of trusting
        # it. Neither enters a hash slot from this side (spec 8.1).
        "torch": torch.__version__,
        "optimizer": {
            "kind": "AdamW",
            # The step AdamW was *built* with, not `group["lr"]`: under a schedule the group
            # holds the last applied rate, and what `optimizer.json` declares is the base.
            "lr": a.lr,
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
        },
    }
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
