"""Hand-written PyTorch reference for packet M15/N6 -- `share` and `Sum` (spec 8.3).

The *other* side of `lower::torch`'s `VisionEncoder { share }` and `Fusion { Sum }` arms: the
three-view graph of `crates/es-policy/tests/sum_share.rs` written out by hand as a person would
type it -- **one** ResNet18 applied to each camera, the three features added with `+`, the
sum concatenated with a state MLP, a linear head -- so the comparison is between what the
lowering generates and an independent module, not between the lowering and itself.

Usage:

    python sum_share_ref.py '<spec json>'

The spec names the lowered source, where to write the weight file, the node ids and the sizes
(see `spec()` in the Rust test). It prints one JSON line, `{"ok": true, ...}` or
`{"ok": false, "error": str}`.

Steps, all in this one process so both modules read the very same tensors:

  1. `exec` the lowered source, build `EsPolicy()` under a fixed seed, and write its
     `state_dict` as the safetensors file `TorchRuntime` will load (keys `nodes.<id>.<rest>`,
     the layout `train_act.py::checkpoint_tensors` writes). Its key count per node is reported:
     the sharers must hold none, the owner one ResNet18.
  2. Build the hand-written module and load it from that file.
  3. Feed both the same seeded inputs, in `eval()`, and report bitwise equality and the max
     absolute difference; sample 0's inputs and output go back so the Rust side can put the
     file through `TorchRuntime::load` (`validate_keys` included) and compare.
  4. With `pretrained_source`: `train_act.py`'s own `init_backbone` on the pretrained variant
     of the graph, which must find exactly one backbone -- the owner's -- to initialise.

`INV-16`: weights cross as safetensors bytes, never `torch.load`, never `pickle`.
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


def lowered(source):
    namespace = {}
    exec(compile(source, "<es-policy>", "exec"), namespace)  # noqa: S102 - our own lowering
    return namespace["EsPolicy"]()


def run(spec):
    import torch
    import torchvision
    from torch import nn

    torch.manual_seed(0)
    with open(spec["source"], encoding="utf-8") as handle:
        model = lowered(handle.read())
    model.eval()
    tensors = {}
    for name, tensor in model.state_dict().items():
        head, _, tail = name[1:].partition(".")
        tensors["nodes." + head + ("." + tail if tail else "")] = tensor
    write_safetensors(spec["weights"], tensors)
    per_node = {}
    for key in tensors:
        node = key.split(".")[1]
        per_node[node] = per_node.get(node, 0) + 1

    width, views = spec["width"], spec["views"]
    horizon, action_dim, execute = spec["horizon"], spec["action_dim"], spec["execute"]

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torchvision.models.resnet18(norm_layer=lambda c: nn.GroupNorm(32, c))
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, width)
            self.state = nn.Sequential(
                nn.Linear(spec["state_dim"], spec["hidden"]), nn.ReLU(), nn.Linear(spec["hidden"], width)
            )
            self.head = nn.Linear(2 * width, horizon * action_dim)

        def forward(self, images, state):
            summed = None
            for image in images:
                feature = self.backbone(image)
                summed = feature if summed is None else summed + feature
            x = torch.cat([summed, self.state(state)], dim=-1)
            return self.head(x).reshape(-1, horizon, action_dim)[:, :execute]

    ref = Reference()
    pick = lambda node: {  # noqa: E731
        k[len("nodes.%d." % node) :]: v for k, v in tensors.items() if k.startswith("nodes.%d." % node)
    }
    ref.backbone.load_state_dict(pick(spec["owner"]), strict=True)
    ref.state.load_state_dict(pick(spec["state"]), strict=True)
    ref.head.load_state_dict(pick(spec["head"]), strict=True)
    ref.eval()

    g = torch.Generator().manual_seed(1)
    batch = spec["batch"]
    inputs = {v: torch.rand([batch] + spec["image"], generator=g) for v in views}
    inputs["joint_state"] = torch.randn([batch, spec["state_dim"]], generator=g)
    with torch.no_grad():
        got = model(**inputs)["actions"]
        want = ref([inputs[v] for v in views], inputs["joint_state"])

    reply = {
        "ok": True,
        "torch": torch.__version__,
        "bitwise": bool(torch.equal(got, want)),
        "max_abs": float((got - want).abs().max()),
        "keys_per_node": per_node,
        "reference_backbone_tensors": len(ref.backbone.state_dict()),
        "sample0_inputs": {k: v[0].reshape(-1).tolist() for k, v in inputs.items()},
        "sample0_actions": got[0].reshape(-1).tolist(),
    }

    if spec.get("pretrained_source"):
        # `train_act.py` is run as a file, so it is read and executed here with no `__file__`,
        # exactly as `ir_training.rs` probes `lr_at` out of it.
        namespace = {}
        with open(spec["train_act"], encoding="utf-8") as handle:
            exec(compile(handle.read(), spec["train_act"], "exec"), namespace)  # noqa: S102
        with open(spec["pretrained_source"], encoding="utf-8") as handle:
            pretrained = lowered(handle.read())
        owner = getattr(pretrained, "n%d" % spec["owner"])
        file = {k: torch.randn_like(v) for k, v in owner.state_dict().items()}
        report = namespace["init_backbone"](pretrained, file)
        reply["init_backbone"] = report
        reply["init_backbone_loaded_exactly"] = all(
            torch.equal(owner.state_dict()[k], file[k]) for k in file if not k.startswith("fc.")
        )
    return reply


def main():
    try:
        reply = run(json.loads(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001 - reported to the Rust side as an error, not a skip
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
