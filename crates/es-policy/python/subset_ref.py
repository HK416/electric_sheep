"""PyTorch side of packet M15/N8's oracle: `es policy subset` against the full graph.

The subset bundle's lowered module, fed only the cameras it kept, against the *full* bundle's
lowered module fed every camera and told to leave the dropped ones out of the sum --
`forward(keep_views=...)`, packet M15/N7's hook, which is what "the full graph with the dropped
terms removed" computes. Each module loads its own safetensors file, so a tensor the subset
lost, renamed wrongly or changed shows up as a different output, not as a shared object.

Usage:

    python subset_ref.py '<spec json>'

Two modes, both printing one JSON line (`{"ok": true, ...}` or `{"ok": false, "error": str}`):

  init     `exec` the lowered `source`, build `EsPolicy()` under `torch.manual_seed(0)` and
           write its `state_dict` to `weights` as `nodes.<id>.<rest>` -- random weights, in
           the layout `train_act.py` writes.
  compare  load the full module and each case's subset module from their files, feed seeded
           inputs, and report per case whether the subset's actions equal the full module's
           with `keep_views=kept` bitwise, the max absolute difference, and whether they
           differ from the full module's with every camera (a term really was dropped).

`INV-16`: weights cross as safetensors bytes, read with `struct` and `json`, never
`torch.load`, never `pickle`.
"""

import json
import struct
import sys


def write_safetensors(path, tensors):
    header, data = {}, bytearray()
    for name in sorted(tensors):
        flat = tensors[name].detach().to("cpu").float().contiguous().reshape(-1)
        start = len(data)
        data.extend(struct.pack("<%df" % flat.numel(), *flat.tolist()))
        header[name] = {"dtype": "F32", "shape": list(tensors[name].shape), "data_offsets": [start, len(data)]}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)) + blob + bytes(data))


def read_state_dict(path):
    """`nodes.<id>.<rest>` -> `n<id>.<rest>`, the rename of the lowering's design note."""
    import torch

    with open(path, "rb") as handle:
        blob = handle.read()
    size = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + size].decode("utf-8"))
    base = 8 + size
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, stop = entry["data_offsets"]
        values = struct.unpack("<%df" % ((stop - start) // 4), blob[base + start : base + stop])
        out["n" + name[len("nodes.") :]] = torch.tensor(values, dtype=torch.float32).reshape(entry["shape"])
    return out


def module(source, weights=None):
    import torch

    namespace = {}
    with open(source, encoding="utf-8") as handle:
        exec(compile(handle.read(), source, "exec"), namespace)  # noqa: S102 - our own lowering
    torch.manual_seed(0)
    model = namespace["EsPolicy"]()
    if weights is not None:
        model.load_state_dict(read_state_dict(weights), strict=True)
    return model.eval()


def init(spec):
    model = module(spec["source"])
    tensors = {}
    for name, tensor in model.state_dict().items():
        head, _, tail = name[1:].partition(".")
        tensors["nodes." + head + "." + tail] = tensor
    write_safetensors(spec["weights"], tensors)
    return {"ok": True, "tensors": len(tensors)}


def compare(spec):
    import torch

    full = module(spec["source"], spec["weights"])
    g = torch.Generator().manual_seed(1)
    inputs = {name: torch.rand([spec["batch"]] + shape, generator=g) for name, shape in sorted(spec["inputs"].items())}
    with torch.no_grad():
        every = full(**inputs)["actions"]
    cases = []
    for case in spec["cases"]:
        sub = module(case["source"], case["weights"])
        with torch.no_grad():
            want = full(keep_views=tuple(case["kept"]), **inputs)["actions"]
            got = sub(**{k: v for k, v in inputs.items() if k in case["inputs"]})["actions"]
        cases.append(
            {
                "kept": case["kept"],
                "bitwise": bool(torch.equal(got, want)),
                "max_abs": float((got - want).abs().max()),
                "finite": bool(torch.isfinite(got).all()),
                "differs_from_every_view": not bool(torch.equal(got, every)),
            }
        )
    return {"ok": True, "torch": torch.__version__, "cases": cases}


def main():
    try:
        spec = json.loads(sys.argv[1])
        reply = {"init": init, "compare": compare}[spec["mode"]](spec)
    except Exception as exc:  # noqa: BLE001 - reported to the Rust side as an error, not a skip
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
