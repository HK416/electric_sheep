"""LeRobot ACT reference process — spec 8.9's M1 gate.

This is the *other* side of `crates/es-policy/src/lerobot.rs`: it runs the real
`lerobot.policies.act.modeling_act.ACTPolicy` on a fixed synthetic observation and prints the
resulting action chunk, so the Rust test can compare it against the same checkpoint run through
our own lowering and `TorchRuntime`.

Usage:

    python act_ref.py <checkpoint dir>

prints one JSON line: {"ok": true, "lerobot": str, "torch": str, "shape": [..], "actions": [..]}
or {"ok": false, "error": str}.

The observation is a fixed integer-derived ramp rather than a seeded draw, so both sides
compute it independently and no tensor crosses the process boundary. Every value is a small
integer over a power of two and is therefore exact in f32 on both sides — a seeded
`torch.Generator` would have had to be reimplemented in Rust, or a 3.7 MB image shipped over a
pipe, to get the same guarantee.

Two notes on what "the real policy" means at the pinned version (see
`docs/api-notes/lerobot-act.md`):

  * `lerobot` 0.6.x moved normalization out of `ACTPolicy` into a separate processor pipeline,
    so `predict_action_chunk` now takes already-normalized inputs and returns normalized
    actions, and the checkpoint's own `normalize_inputs` / `unnormalize_outputs` buffers load as
    *unexpected* keys. This script therefore applies them itself, with LeRobot's own MEAN_STD
    formula, around the call. Our lowering carries the same buffers as nodes 9/10/11, so both
    sides run the identical normalization and the comparison is about the network.
  * `INV-16`: the checkpoint is read through `safetensors`/`from_pretrained`, never `torch.load`
    and never `pickle`.
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


def ramp(count, step, offset, scale, shift):
    """`((i * step + offset) % 4096) / scale + shift` — see the module docstring."""
    return [((i * step + offset) % 4096) / scale + shift for i in range(count)]


def run(path):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = json.load(open(path + "/config.json"))
    camera = next(k for k, v in sorted(config["input_features"].items()) if v["type"] == "VISUAL")
    image_shape = config["input_features"][camera]["shape"]
    state_dim = config["input_features"]["observation.state"]["shape"][0]

    stats = read_safetensors(path + "/model.safetensors")
    buffer = "normalize_inputs.buffer_" + camera.replace(".", "_")
    image_mean, image_std = stats[buffer + ".mean"], stats[buffer + ".std"]
    state_mean = stats["normalize_inputs.buffer_observation_state.mean"]
    state_std = stats["normalize_inputs.buffer_observation_state.std"]
    action_mean = stats["unnormalize_outputs.buffer_action.mean"]
    action_std = stats["unnormalize_outputs.buffer_action.std"]

    # The same ramp the Rust side builds: image in [0, 1), state in [-1, 1).
    pixels = image_shape[0] * image_shape[1] * image_shape[2]
    image = torch.tensor(ramp(pixels, 37, 11, 4096.0, 0.0), dtype=torch.float32)
    image = image.reshape(1, *image_shape)
    state = torch.tensor(ramp(state_dim, 911, 3, 2048.0, -1.0), dtype=torch.float32)
    state = state.reshape(1, state_dim)

    policy = ACTPolicy.from_pretrained(path)
    policy.eval()
    batch = {
        camera: (image - image_mean) / (image_std + 1e-8),
        "observation.state": (state - state_mean) / (state_std + 1e-8),
    }
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)[0][: config["n_action_steps"]]
    actions = chunk * action_std + action_mean

    import importlib.metadata

    return {
        "ok": True,
        "lerobot": importlib.metadata.version("lerobot"),
        "torch": torch.__version__,
        "shape": list(actions.shape),
        "actions": actions.reshape(-1).tolist(),
    }


def main():
    try:
        reply = run(sys.argv[1])
    except Exception as exc:  # Reported on the wire, never as a traceback on stderr.
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()


main()
