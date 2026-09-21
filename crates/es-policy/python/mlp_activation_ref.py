"""Hand-written PyTorch reference for packet M8/S2a — the MLP activations and the squash.

This is the *other* side of `lower::torch`'s `StateEncoder{Mlp}` and `PolicyHead{Regression}`
arms: the same network written out by hand, so the comparison is between what the lowering
generates and what a person would have typed, not between the lowering and itself.

Usage:

    python mlp_activation_ref.py <weights.safetensors> '<spec json>'

where the spec is `{"dims": [15, 32, 32], "action_dim": 6, "horizon": 1,
"activation": "Elu", "activate_output": true, "squash": "Tanh", "count": 64}`. It prints one
JSON line, `{"ok": true, "torch": str, "actions": [[...], ...]}` or `{"ok": false,
"error": str}`.

The inputs never cross the process boundary: both sides build them from the same integer
formula over a power of two (see `ramp`), so every value is exact in f32 on both sides. The
weights do, as the safetensors file the lowered module loads — also exact.

`INV-16`: the checkpoint is read as safetensors bytes, never `torch.load` and never `pickle`.
"""

import json
import struct
import sys


def read_safetensors(path):
    """Header length (u64 LE), header JSON, then the raw tensor bytes. No safetensors package."""
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
        raw = blob[base + start : base + stop]
        values = struct.unpack("<%df" % (len(raw) // 4), raw)
        out[name] = torch.tensor(values, dtype=torch.float32).reshape(entry["shape"])
    return out


def ramp(sample, dim):
    """`((37 i + 11 j) % 4096) / 256 - 8`, the observation both sides build independently."""
    return ((sample * 37 + dim * 11) % 4096) / 256.0 - 8.0


def run(weights_path, spec):
    import torch
    from torch import nn

    activations = {"Relu": nn.ReLU, "Elu": nn.ELU, "Swish": nn.SiLU, "Tanh": nn.Tanh}
    make = activations[spec["activation"]]
    dims = spec["dims"]

    layers = []
    for i in range(len(dims) - 1):
        if i > 0:
            layers.append(make())
        layers.append(nn.Linear(dims[i], dims[i + 1]))
    if spec["activate_output"]:
        layers.append(make())
    encoder = nn.Sequential(*layers)
    head = nn.Linear(dims[-1], spec["horizon"] * spec["action_dim"])

    tensors = read_safetensors(weights_path)
    encoder.load_state_dict(
        dict(
            (name[len("nodes.0.") :], value)
            for name, value in tensors.items()
            if name.startswith("nodes.0.")
        )
    )
    head.load_state_dict({"weight": tensors["nodes.1.weight"], "bias": tensors["nodes.1.bias"]})

    actions = []
    with torch.inference_mode():
        for sample in range(spec["count"]):
            x = torch.tensor(
                [[ramp(sample, dim) for dim in range(dims[0])]], dtype=torch.float32
            )
            y = head(encoder(x))
            if spec["squash"] == "Tanh":
                y = torch.tanh(y)
            y = y.reshape(-1, spec["horizon"], spec["action_dim"])
            actions.append(y.reshape(-1).tolist())
    return {"ok": True, "torch": torch.__version__, "actions": actions}


def main():
    try:
        reply = run(sys.argv[1], json.loads(sys.argv[2]))
    except Exception as exc:  # A failure is a typed error on the wire, never a traceback.
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")


main()
