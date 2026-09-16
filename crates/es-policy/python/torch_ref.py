"""PyTorch reference process for `TorchRuntime` (spec 1.4: the Learning IR oracle).

Line-delimited JSON on stdin, one JSON object per line on stdout. Requests:

    {"cmd": "load",  "source": str, "weights_path": str, "batch_axis": bool}
    {"cmd": "infer", "inputs": {name: {"shape": [int], "dtype": "F32", "data_b64": str}}}
    {"cmd": "quit"}

Every response is {"ok": true, ...} or {"ok": false, "error": str}.

`source` is the file `es_policy::lower::torch` generated from the LearningGraph; it defines
`class EsPolicy(nn.Module)` whose `forward(**inputs)` returns a dict of named tensors.

`batch_axis` says that `forward` wants a leading batch axis on every input and puts one on
every output (packet M7/T3, the shape `train_act.py` trains). Inference here is one sample, so
this process feeds `[1, ..]` and strips the axis off again before replying: spec 5.2 gives the
inference domain its own batch size and the module has no say in it. A `lerobot::lower_act`
module sets it false -- it takes one observation and returns an unbatched chunk.

Three things this script deliberately does not do:

  * it never calls `torch.load` and never imports `pickle` (INV-16). Weights arrive as
    safetensors and are read by the 15-line reader below, so the reference path cannot execute
    code that came from a checkpoint;
  * it needs no `numpy` and no `safetensors` package -- only `torch` -- so the CI venv is one
    wheel. Tensor payloads cross as base64'd little-endian f32 built with `struct`;
  * it does no pre- or post-processing. Those are IR nodes and are inside `source` already
    (spec 8.7). See docs/api-notes/torch.md for the pinned API surface.
"""

import base64
import json
import struct
import sys

try:
    import torch
except ImportError as exc:  # Reported as a protocol response, not a traceback on stderr.
    sys.stdout.write(json.dumps({"ok": False, "error": "import failed: %s" % exc}) + "\n")
    sys.stdout.flush()
    raise SystemExit(1)

PROTOCOL = 1
# The protocol and the safetensors files it reads are f32 only (spec 8.4 `runtime.dtype`).
DTYPE = "F32"


def unpack_f32(raw):
    if len(raw) % 4:
        raise ValueError("%d bytes is not a whole number of f32" % len(raw))
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def read_safetensors(path):
    """Header length (u64 LE), header JSON, then the raw tensor bytes."""
    with open(path, "rb") as handle:
        blob = handle.read()
    size = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8 : 8 + size].decode("utf-8"))
    base = 8 + size
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if entry["dtype"] != DTYPE:
            raise ValueError("%s has dtype %s, expected %s" % (name, entry["dtype"], DTYPE))
        start, stop = entry["data_offsets"]
        values = unpack_f32(blob[base + start : base + stop])
        out[name] = torch.tensor(values, dtype=torch.float32).reshape(entry["shape"])
    return out


def to_state_dict(tensors):
    """`nodes.<node_id>.<rest>` -> `n<node_id>.<rest>`: the rename of design note section 4."""
    state = {}
    for name, tensor in tensors.items():
        if not name.startswith("nodes."):
            raise ValueError("weight key %r is not under `nodes.`" % name)
        head, _, tail = name[len("nodes.") :].partition(".")
        state["n" + head + ("." + tail if tail else "")] = tensor
    return state


def build(source, weights_path):
    namespace = {}
    # `source` is this process's own generated module, not user input. It is the only code
    # this script executes, and it arrives over the pipe rather than inside the weights.
    exec(compile(source, "<es-policy>", "exec"), namespace)
    model = namespace["EsPolicy"]()
    model.load_state_dict(to_state_dict(read_safetensors(weights_path)), strict=True)
    model.eval()
    return model


def decode(name, spec):
    if spec["dtype"] != DTYPE:
        raise ValueError("input %r has dtype %s" % (name, spec["dtype"]))
    values = unpack_f32(base64.b64decode(spec["data_b64"]))
    return torch.tensor(values, dtype=torch.float32).reshape(spec["shape"])


def encode(tensor):
    flat = tensor.detach().to(torch.float32).contiguous().reshape(-1).tolist()
    raw = struct.pack("<%df" % len(flat), *flat)
    return {
        "shape": list(tensor.shape),
        "dtype": DTYPE,
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def infer(model, inputs, batch_axis):
    tensors = dict((name, decode(name, spec)) for name, spec in inputs.items())
    if batch_axis:
        tensors = dict((name, t.unsqueeze(0)) for name, t in tensors.items())
    with torch.inference_mode():
        outputs = model(**tensors)
    if not isinstance(outputs, dict):
        raise TypeError("forward returned %s, expected a dict" % type(outputs).__name__)
    if batch_axis:
        outputs = dict((name, value[0]) for name, value in outputs.items())
    return dict((name, encode(value)) for name, value in outputs.items())


def main():
    model, batch_axis = None, False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "quit":
                return
            if command == "load":
                model = build(request["source"], request["weights_path"])
                batch_axis = bool(request.get("batch_axis"))
                reply = {"ok": True, "torch_version": torch.__version__, "protocol": PROTOCOL}
            elif command == "infer":
                if model is None:
                    raise ValueError("no policy is loaded")
                reply = {"ok": True, "outputs": infer(model, request["inputs"], batch_axis)}
            else:
                raise ValueError("unknown command %r" % (command,))
        except Exception as exc:  # Any failure is a typed error on the wire, never a dead pipe.
            reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


main()
