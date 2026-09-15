"""LeRobot ACT reference process — spec 8.9's M1 gate.

This is the *other* side of `crates/es-policy/src/lerobot.rs`: it runs the real
`lerobot.policies.act.modeling_act.ACTPolicy` on a fixed synthetic observation and prints the
resulting action chunk, so the Rust test can compare it against the same checkpoint run through
our own lowering and `TorchRuntime`.

Usage:

    python act_ref.py <checkpoint dir> [observation.json]

prints one JSON line: {"ok": true, "lerobot": str, "torch": str, "shape": [..], "actions": [..]}
or {"ok": false, "error": str}.

Without the second argument the observation is a fixed integer-derived ramp rather than a
seeded draw, so both sides compute it independently and no tensor crosses the process boundary.
Every value is a small integer over a power of two and is therefore exact in f32 on both sides
— a seeded `torch.Generator` would have had to be reimplemented in Rust, or a 3.7 MB image
shipped over a pipe, to get the same guarantee.

With it, `{"state": [...], "image": [...]}` — flat, row-major, the shapes the config declares —
is read by both sides instead. That is how a checkpoint trained on *this project's*
demonstrations is compared on a frame it was actually trained on (packet M5/V8): the file is
written with shortest-round-trip decimals, so `float()` here and `f32` there are the same
number, and the exactness guarantee above is unchanged.

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


def run(path, observation=None):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = json.load(open(path + "/config.json"))
    camera = next(k for k, v in sorted(config["input_features"].items()) if v["type"] == "VISUAL")
    image_shape = config["input_features"][camera]["shape"]
    state_dim = config["input_features"]["observation.state"]["shape"][0]

    stats = read_safetensors(path + "/model.safetensors")
    buffer = "normalize_inputs.buffer_" + camera.replace(".", "_")
    if buffer + ".mean" not in stats:
        # LeRobot 0.6.x's second checkpoint layout: normalization moved into a processor
        # pipeline, whose state file `policy_preprocessor.json` names, keyed `<feature>.<stat>`
        # with no prefix. Both layouts are read; `es_policy::lerobot` does the same.
        pipeline = json.load(open(path + "/policy_preprocessor.json"))
        step = next(s for s in pipeline["steps"] if s["registry_name"] == "normalizer_processor")
        stats = read_safetensors(path + "/" + step["state_file"])
        buffer, state, action = camera, "observation.state", "action"
    else:
        state, action = (
            "normalize_inputs.buffer_observation_state",
            "unnormalize_outputs.buffer_action",
        )
    image_mean, image_std = stats[buffer + ".mean"], stats[buffer + ".std"]
    state_mean, state_std = stats[state + ".mean"], stats[state + ".std"]
    action_mean, action_std = stats[action + ".mean"], stats[action + ".std"]

    # The same observation the Rust side builds: either a recorded frame both sides read from
    # one file, or the fixed ramp (image in [0, 1), state in [-1, 1)).
    pixels = image_shape[0] * image_shape[1] * image_shape[2]
    if observation is None:
        flat_image = ramp(pixels, 37, 11, 4096.0, 0.0)
        flat_state = ramp(state_dim, 911, 3, 2048.0, -1.0)
    else:
        given = json.load(open(observation))
        flat_image, flat_state = given["image"], given["state"]
        if len(flat_image) != pixels or len(flat_state) != state_dim:
            raise ValueError(
                "observation has %d pixels and %d state values; the config declares %d and %d"
                % (len(flat_image), len(flat_state), pixels, state_dim)
            )
    image = torch.tensor(flat_image, dtype=torch.float32).reshape(1, *image_shape)
    state = torch.tensor(flat_state, dtype=torch.float32).reshape(1, state_dim)

    # CPU on both sides, always. `config.json` records the device the checkpoint was *trained*
    # on, and `from_pretrained` honours it; our `TorchRuntime` is a CPU subprocess, and an
    # fp32 comparison between two devices would measure cuDNN's kernel choice, not the module.
    policy = ACTPolicy.from_pretrained(path).to("cpu")
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
        reply = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except Exception as exc:  # Reported on the wire, never as a traceback on stderr.
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()


main()
