"""A PPO actor from brax, rsl_rl or rl_games becomes `weights.safetensors` + `import.json`
(M8/S2b; spec 14.4 "RL policy import", 13.4 "the import never guesses", INV-16).

    import_rl.py --from mujoco-playground|rsl-rl|rl-games --checkpoint <path>
                 [--activation relu|elu|swish|tanh] --out <dir>
    import_rl.py --synth mujoco-playground|rsl-rl|rl-games [--native]
                 [--action position|delta] --out <dir>
    import_rl.py --reference --import <dir> [--resolved mapping-report.json]
                 [--npz oracle-1000.npz] [--rows N] [--seed N] --out oracle.safetensors

This is the **only** place a pickle or an orbax checkpoint is opened (INV-16): `torch.load`
and `brax.training.checkpoint.load` live here and nowhere in the Rust core. What leaves is a
`safetensors` file with framework-neutral keys and a JSON manifest of everything the Rust half
would otherwise have to guess -- and guessing is what `es policy import-rl` refuses to do.

**The neutral form.** `mlp.<i>.weight` `[out, in]` / `mlp.<i>.bias` for each *hidden* Dense,
then `head.weight` / `head.bias` for the output Dense. Every framework here activates every
hidden layer and leaves the output Dense linear, so `import.json`'s `activate_output` is
`false` for all three and the only output nonlinearity is `squash`. (The Rust half's
`StateEncoder{Mlp}` ends at the last *hidden* layer and its `PolicyHead{Regression}` is the
output Dense, so *our* encoder carries `activate_output = true` -- the same network, cut in a
different place. `crates/es-data/src/rl_import.rs` says this again where it does it.)

**`log_std`.** `null` unless the source carries a state-independent one. rsl_rl's `std` and
rl_games' `sigma` are `[action_dim]` parameters, so those two do (`log(std)` and `sigma`
itself). brax's second output half is a *function of the observation*
(`std = softplus(x) + 0.001`, `brax/training/distribution.py:171`), so there is no constant to
import: its rows travel in `import.json` under `source_std` as metadata and `log_std` is
`null`. `python/es/train_ppo.py --init-log-std` is fed only when `log_std` is non-null; it
takes one scalar, so a vector whose entries differ is a human's choice, not the importer's.

**`--reference`** recomputes the imported function in torch from the neutral weights alone and
writes the 1,000-row oracle the `--ignored` Rust test reads: `obs_normalized` (what our
Observation IR hands the module), `reference` (this reconstruction, in actuator units) and,
when `--npz` carries them, `actions_scaled` (the source framework's own deterministic actions,
normalized). Tier (a) of `docs/design/rl-continuation.md` section 4 compares `reference`
against our runtime bitwise; tier (b) compares our runtime against `actions_scaled`. Pass
`--resolved` the `mapping-report.json` the Rust half wrote, so both sides carry the same
action tail -- the adapter may have overridden the checkpoint's `scale`/`offset`, and that
decision is made in one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# The sibling trainer, by path, for the safetensors halves -- a second implementation of the
# format is how two writers quietly start writing two formats (the argument `train_ppo.py`
# makes at its own import).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_act import read_safetensors, write_safetensors  # noqa: E402

FRAMEWORKS = ("mujoco-playground", "rsl-rl", "rl-games")
ACTIVATIONS = ("relu", "elu", "swish", "tanh")
SYNTH_HIDDEN = (8, 8)
SYNTH_OBS, SYNTH_ACTION = 26, 6
# The increment `python/es/rl_source/train_brax_so101.py --action delta` records, in rad per
# control tick (`docs/api-notes/brax-ppo-so101.md` section 7).
DELTA_SCALE = 0.05


def fail(message: str) -> SystemExit:
    return SystemExit("import_rl.py: " + message)


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# --- the three readers ------------------------------------------------------------------------


def read_playground(path: Path, activation: str | None) -> tuple[dict, dict]:
    """S2c's `source.npz` + `meta.json` (`docs/api-notes/brax-ppo-so101.md` section 6), or an
    orbax checkpoint through `brax.training.checkpoint.load`.

    brax stores `kernel` as `[in, out]` (right-multiply); torch wants `[out, in]`, so every
    kernel is transposed exactly once, here.
    """
    npz_path = path / "source.npz" if path.is_dir() else path
    meta_path = (path if path.is_dir() else path.parent) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if npz_path.exists() and npz_path.suffix == ".npz":
        npz = np.load(npz_path)
        n = sum(1 for k in npz.files if k.startswith("kernel_"))
        kernels = [f32(npz[f"kernel_{i}"]) for i in range(n)]
        biases = [f32(npz[f"bias_{i}"]) for i in range(n)]
        head_k, head_b = f32(npz["mean_kernel"]), f32(npz["mean_bias"])
        std_k, std_b = f32(npz["logstd_kernel"]), f32(npz["logstd_bias"])
        obs_mean, obs_std = f32(npz["obs_mean"]), f32(npz["obs_std"])
    else:
        # The orbax route (INV-16: opened here and only here).
        from brax.training import checkpoint  # noqa: PLC0415

        params = checkpoint.load(str(path))
        normalizer, policy = params[0], params[1]
        dense = policy["params"]
        names = sorted(dense, key=lambda k: int(k.split("_")[-1]))
        *hidden, last = names
        kernels = [f32(dense[k]["kernel"]) for k in hidden]
        biases = [f32(dense[k]["bias"]) for k in hidden]
        kernel, bias = f32(dense[last]["kernel"]), f32(dense[last]["bias"])
        half = kernel.shape[1] // 2
        head_k, head_b = kernel[:, :half], bias[:half]
        std_k, std_b = kernel[:, half:], bias[half:]
        obs_mean, obs_std = f32(normalizer.mean), f32(normalizer.std)

    tensors = {}
    for i, (k, b) in enumerate(zip(kernels, biases)):
        tensors[f"mlp.{i}.weight"], tensors[f"mlp.{i}.bias"] = k.T.copy(), b
    tensors["head.weight"], tensors["head.bias"] = head_k.T.copy(), head_b

    action = meta.get("action", {})
    manifest = {
        "framework": "mujoco-playground",
        "versions": meta.get("versions", {}),
        "obs_dim": int(obs_mean.shape[0]),
        "action_dim": int(head_b.shape[0]),
        "hidden": [int(b.shape[0]) for b in biases],
        "activation": activation or meta.get("activation", "swish"),
        # `brax.training.networks.MLP(activate_final=False)`: the output Dense is linear.
        "activate_output": False,
        "squash": meta.get("squash", "tanh"),
        "obs_mean": obs_mean.tolist(),
        "obs_std": obs_std.tolist(),
        "normalizer": "(x - obs_mean) / obs_std, no clipping",
        "log_std": None,
        "source_std": {
            "kind": "softplus_plus_0.001",
            "note": "state-dependent: std = softplus(x @ kernel + bias) + 0.001, so there is "
            "no constant log-std to import. Metadata only; not in weights.safetensors.",
            "kernel": std_k.T.tolist(),
            "bias": std_b.tolist(),
        },
        # The source says what its six numbers *are*; the adapter says it again for our
        # robot, and `es policy import-rl` refuses the two disagreeing (IMP-004). Absent (a
        # checkpoint whose exporter never recorded it) is not a guess either -- it is `null`,
        # and then the adapter alone decides.
        "action_kind": action.get("kind"),
        "action_scale": action.get("scale"),
        "action_offset": action.get("offset"),
        "joint_order": meta.get("joint_order"),
    }
    return tensors, manifest


def _torch_load(path: Path) -> dict:
    """`torch.load` -- the pickle door, open in this file only (INV-16)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _ordered_mlp(state: dict, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """`<prefix>.<i>.weight|bias` in `i` order. `i` is the index inside the framework's own
    `nn.Sequential`, so it counts activations too and is never assumed to be contiguous."""
    idx = sorted(
        {
            int(k[len(prefix) + 1 :].split(".")[0])
            for k in state
            if k.startswith(prefix + ".") and k.endswith(".weight")
        }
    )
    return [
        (f32(state[f"{prefix}.{i}.weight"]), f32(state[f"{prefix}.{i}.bias"])) for i in idx
    ]


def read_rsl_rl(path: Path, activation: str | None) -> tuple[dict, dict]:
    """`model_*.pt`, both layouts rsl_rl has shipped:

    * **pre-5.0** -- `model_state_dict` holding one `ActorCritic`: the actor is an
      `nn.Sequential` named `actor`, so the keys are `actor.<i>.weight`, and the Gaussian's
      standard deviation is a top-level `std` parameter;
    * **>= 5.0** -- `actor_state_dict` holding one `MLPModel` (measured against
      `rsl-rl-lib` 5.5.1, packet M8/S2b): the `nn.Sequential` is named **`mlp`**, the
      normalizer is `obs_normalizer.*`, and the std is `distribution.std_param` (or
      `distribution.log_std_param` when `std_type = "log"`). The packet's draft expected
      `actor.<i>` in this layout too; it is `mlp.<i>`, and both are accepted here.

    The file carries no activation -- rsl_rl takes it as a constructor string and saves only
    tensors -- so `--activation` is required rather than guessed (spec 13.4).
    """
    if activation is None:
        raise fail(
            "--from rsl-rl needs --activation: the checkpoint does not record it "
            "(rsl_rl passes it to the model as a string and saves only tensors). "
            "The default in rsl_rl's own configs is elu."
        )
    raw = _torch_load(path)
    state = None
    for key in ("actor_state_dict", "model_state_dict", "state_dict"):
        if isinstance(raw.get(key), dict):
            state = raw[key]
            break
    if state is None and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state = raw
    if state is None:
        raise fail(
            f"{path}: no actor_state_dict / model_state_dict in {sorted(k for k in raw)}"
        )

    layers = next(
        (found for prefix in ("actor", "mlp") if (found := _ordered_mlp(state, prefix))),
        [],
    )
    if not layers:
        raise fail(f"{path}: no actor.<i>.weight or mlp.<i>.weight keys in {sorted(state)[:8]}")
    *hidden, head = layers
    tensors = {}
    for i, (w, b) in enumerate(hidden):
        tensors[f"mlp.{i}.weight"], tensors[f"mlp.{i}.bias"] = w, b
    tensors["head.weight"], tensors["head.bias"] = head

    # The Gaussian's standard deviation, state-independent in every layout. `log_std` is what
    # `train_ppo.py --init-log-std` would be fed, so it is a log whatever the file stored.
    std = next(
        (state[k] for k in ("std", "distribution.std_param") if k in state),
        None,
    )
    if "distribution.log_std_param" in state:
        log_std = f32(state["distribution.log_std_param"]).tolist()
        std = torch.exp(state["distribution.log_std_param"])
    else:
        log_std = np.log(f32(std)).tolist() if std is not None else None
    # `EmpiricalNormalization` keeps `mean` and `var`; the std it divides by is sqrt(var).
    norm = {k.split(".")[-1].lstrip("_"): v for k, v in state.items() if "obs_normalizer" in k}
    obs_mean = f32(norm["mean"]).tolist() if "mean" in norm else None
    obs_std = None
    if "std" in norm:
        obs_std = f32(norm["std"]).tolist()
    elif "var" in norm:
        obs_std = np.sqrt(f32(norm["var"])).tolist()

    return tensors, {
        "framework": "rsl-rl",
        "versions": _versions("rsl_rl", "rsl-rl-lib", "torch"),
        "obs_dim": int(hidden[0][0].shape[1]) if hidden else int(head[0].shape[1]),
        "action_dim": int(head[1].shape[0]),
        "hidden": [int(b.shape[0]) for _, b in hidden],
        "activation": activation,
        "activate_output": False,
        # rsl_rl's distribution is `Normal(actor(x), std)`: no squash inside the actor.
        "squash": "none",
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "normalizer": "EmpiricalNormalization: (x - mean) / sqrt(var)" if obs_mean else None,
        "log_std": log_std,
        "source_std": None if std is None else {"kind": "std", "value": f32(std).tolist()},
        "action_kind": None,
        "action_scale": None,
        "action_offset": None,
        "joint_order": None,
    }


def read_rl_games(path: Path, activation: str | None) -> tuple[dict, dict]:
    """`.pth`: the `model` dict, `a2c_network.actor_mlp.<i>` + `a2c_network.mu`, with
    `running_mean_std.running_mean|running_var` when the run normalized observations."""
    if activation is None:
        raise fail(
            "--from rl-games needs --activation: the checkpoint stores tensors only "
            "(the activation is a string in the yaml config). rl_games' own default is elu."
        )
    raw = _torch_load(path)
    state = raw["model"] if isinstance(raw.get("model"), dict) else raw

    hidden = _ordered_mlp(state, "a2c_network.actor_mlp")
    if not hidden or "a2c_network.mu.weight" not in state:
        raise fail(f"{path}: no a2c_network.actor_mlp.<i> / a2c_network.mu in {sorted(state)[:8]}")
    tensors = {}
    for i, (w, b) in enumerate(hidden):
        tensors[f"mlp.{i}.weight"], tensors[f"mlp.{i}.bias"] = w, b
    tensors["head.weight"] = f32(state["a2c_network.mu.weight"])
    tensors["head.bias"] = f32(state["a2c_network.mu.bias"])

    # `sigma` IS the log-std: rl_games samples `Normal(mu, torch.exp(sigma))`.
    sigma = state.get("a2c_network.sigma")
    obs_mean = state.get("running_mean_std.running_mean")
    obs_var = state.get("running_mean_std.running_var")
    return tensors, {
        "framework": "rl-games",
        "versions": _versions("rl_games", "rl-games", "torch"),
        "obs_dim": int(hidden[0][0].shape[1]),
        "action_dim": int(tensors["head.bias"].shape[0]),
        "hidden": [int(b.shape[0]) for _, b in hidden],
        "activation": activation,
        "activate_output": False,
        "squash": "none",
        "obs_mean": None if obs_mean is None else f32(obs_mean).tolist(),
        # rl_games' `RunningMeanStd`: (x - mean) / sqrt(var + 1e-5).
        "obs_std": None if obs_var is None else np.sqrt(f32(obs_var) + 1e-5).tolist(),
        "normalizer": None if obs_var is None else "RunningMeanStd: (x - mean)/sqrt(var + 1e-5)",
        "log_std": None if sigma is None else f32(sigma).tolist(),
        "source_std": None if sigma is None else {"kind": "log_std", "value": f32(sigma).tolist()},
        "action_kind": None,
        "action_scale": None,
        "action_offset": None,
        "joint_order": None,
    }


def _versions(module: str, distribution: str, *also: str) -> dict:
    import importlib.metadata as md  # noqa: PLC0415

    out = {}
    for name in (distribution, *also):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "absent"
    if out.get(distribution) == "absent":
        mod = sys.modules.get(module)
        out[distribution] = getattr(mod, "__version__", "absent") if mod else "absent"
    return out


READERS = {
    "mujoco-playground": read_playground,
    "rsl-rl": read_rsl_rl,
    "rl-games": read_rl_games,
}


# --- the reconstruction, shared by `--reference` and `--synth --native` -------------------------


def rebuild(tensors: dict, manifest: dict) -> torch.nn.Module:
    """The imported function in torch, from the neutral weights and the manifest alone.

    The op order is the one `es_policy::lower::lower_to_torch` emits for
    `StateEncoder{Mlp} -> PolicyHead{Regression, squash} -> Normalizer{Inverse, MeanStd}`:
    `nn.Linear`, the activation module, ..., `nn.Linear`, `tanh`, `x * scale + offset`. It has
    to be, or tier (a) of the oracle is comparing two different functions.
    """
    act = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.SiLU,
        "tanh": torch.nn.Tanh,
    }[manifest["activation"]]
    n = len(manifest["hidden"])
    layers: list[torch.nn.Module] = []
    for i in range(n):
        w = tensors[f"mlp.{i}.weight"]
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data, linear.bias.data = _t(w), _t(tensors[f"mlp.{i}.bias"])
        layers += [linear, act()]
    head = torch.nn.Linear(tensors["head.weight"].shape[1], tensors["head.weight"].shape[0])
    head.weight.data, head.bias.data = _t(tensors["head.weight"]), _t(tensors["head.bias"])
    layers.append(head)
    body = torch.nn.Sequential(*layers)

    squash = manifest["squash"]
    dim = manifest["action_dim"]
    scale = _t(manifest["action_scale"] if manifest["action_scale"] else [1.0] * dim)
    offset = _t(manifest["action_offset"] if manifest["action_offset"] else [0.0] * dim)

    class Rebuilt(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            self.register_buffer("scale", scale)
            self.register_buffer("offset", offset)

        def forward(self, x):
            y = self.body(x)
            if squash == "tanh":
                y = torch.tanh(y)
            return y * self.scale + self.offset

    return Rebuilt().eval()


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


# --- the three modes ---------------------------------------------------------------------------


def do_import(args) -> None:
    tensors, manifest = READERS[args.source](Path(args.checkpoint).expanduser(), args.activation)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_safetensors(out / "weights.safetensors", {k: _t(v) for k, v in tensors.items()})
    (out / "import.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shapes = {k: list(np.asarray(v).shape) for k, v in tensors.items()}
    print(
        json.dumps(
            {
                "framework": manifest["framework"],
                "obs_dim": manifest["obs_dim"],
                "action_dim": manifest["action_dim"],
                "hidden": manifest["hidden"],
                "activation": manifest["activation"],
                "squash": manifest["squash"],
                "log_std": "null" if manifest["log_std"] is None else "imported",
                "tensors": shapes,
                "out": str(out),
            }
        )
    )


def synth_tensors(rng) -> dict:
    """26 -> [8, 8] -> 6, seeded, so the committed fixtures are a function of this code."""
    dims = [SYNTH_OBS, *SYNTH_HIDDEN, SYNTH_ACTION]
    out = {}
    for i in range(len(dims) - 1):
        name = "head" if i == len(dims) - 2 else f"mlp.{i}"
        scale = 1.0 / np.sqrt(dims[i])
        out[f"{name}.weight"] = f32(rng.uniform(-scale, scale, (dims[i + 1], dims[i])))
        out[f"{name}.bias"] = f32(rng.uniform(-scale, scale, dims[i + 1]))
    return out


def native_rsl_rl(path: Path):
    """A random-weights actor built and saved by rsl_rl's own classes. Returns the module whose
    `forward` is the deterministic actor, or `None` with a printed reason.

    Two shapes, because rsl_rl has two: >= 5.0's `MLPModel` under `actor_state_dict` (what
    `PPO.save` writes, `algorithms/ppo.py:377`), and the older `ActorCritic` under
    `model_state_dict`. Whichever imports is the one used, and the version is recorded.
    """
    torch.manual_seed(0)
    try:
        from rsl_rl.models import MLPModel  # noqa: PLC0415
        from tensordict import TensorDict  # noqa: PLC0415

        obs = TensorDict({"policy": torch.zeros(1, SYNTH_OBS)}, batch_size=[1])
        model = MLPModel(
            obs=obs,
            obs_groups={"actor": ["policy"]},
            obs_set="actor",
            output_dim=SYNTH_ACTION,
            hidden_dims=list(SYNTH_HIDDEN),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": 1.0,
            },
        )
        torch.save({"actor_state_dict": model.state_dict(), "iter": 0}, path)
        return model.mlp
    except Exception as exc:  # noqa: BLE001 - reported, never silent (spec 1.4)
        print(f"NOTE --native rsl-rl: MLPModel unavailable ({type(exc).__name__}: {exc})")
    try:
        from rsl_rl.modules import ActorCritic  # noqa: PLC0415

        ac = ActorCritic(
            num_actor_obs=SYNTH_OBS,
            num_critic_obs=SYNTH_OBS,
            num_actions=SYNTH_ACTION,
            actor_hidden_dims=list(SYNTH_HIDDEN),
            critic_hidden_dims=list(SYNTH_HIDDEN),
            activation="elu",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP --native rsl-rl: {type(exc).__name__}: {exc}")
        return None
    torch.save({"model_state_dict": ac.state_dict(), "iter": 0}, path)
    return ac.actor


def native_rl_games(path: Path):
    """The same for rl_games: its `A2CBuilder` network under the `model` key its own
    `save`/`torch.save` writes."""
    try:
        from rl_games.algos_torch import model_builder  # noqa: PLC0415

        torch.manual_seed(0)
        # The keys `A2CBuilder.load` reads; a missing one is a `KeyError` at build time, so
        # this is the whole `network:` block of an rl_games yaml, minimally filled.
        params = {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": 0.0},
                    "fixed_sigma": True,
                }
            },
            "mlp": {
                "units": list(SYNTH_HIDDEN),
                "activation": "elu",
                "d2rl": False,
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
        }
        builder = model_builder.ModelBuilder().load(
            {"model": {"name": "continuous_a2c_logstd"}, "network": params}
        )
        # `BaseModel.build` takes one config dict (`algos_torch/models.py:29`), not keywords.
        network = builder.build(
            {
                "actions_num": SYNTH_ACTION,
                "input_shape": (SYNTH_OBS,),
                "num_seqs": 1,
                "value_size": 1,
                "normalize_value": False,
                "normalize_input": False,
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP --native rl-games: {type(exc).__name__}: {exc}")
        return None
    torch.save({"model": network.state_dict(), "epoch": 0}, path)
    # rl_games' deterministic action is `mu_act(mu(actor_mlp(x)))`; its `Network.forward` takes
    # an `obs_dict` and returns the whole distribution, so the actor is assembled from the
    # framework's own three modules rather than called through that wrapper.
    net = network.a2c_network
    return torch.nn.Sequential(net.actor_mlp, net.mu, net.mu_act)


def do_synth(args) -> None:
    """Writes one framework's *native* layout and converts it, so the committed fixture under
    `tests/fixtures/rl/import/` is generated rather than typed (packet M8/S2b).

    `--native` asks the framework itself to build and save the checkpoint, which is what the
    packet's rsl_rl / rl_games oracle needs; it falls back to the layout below, with a printed
    reason, when the framework is not installed. The fixtures committed to this repository are
    the fallback, so CI needs neither package.
    """
    if args.action != "position" and args.synth != "mujoco-playground":
        # rsl_rl and rl_games record no action metadata at all, so there is nothing for
        # `--action` to put in the manifest and the adapter would be the only declaration.
        raise fail(f"--action {args.action} is only meaningful with --synth mujoco-playground")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260921)
    t = synth_tensors(rng)
    n = len(SYNTH_HIDDEN)
    native = out / "native"
    native.mkdir(exist_ok=True)
    obs_mean = f32(rng.normal(0.0, 0.1, SYNTH_OBS))
    obs_std = f32(rng.uniform(0.5, 1.5, SYNTH_OBS))
    actor = None
    if args.native and args.synth != "mujoco-playground":
        actor = (native_rsl_rl if args.synth == "rsl-rl" else native_rl_games)(
            native / ("model_0.pt" if args.synth == "rsl-rl" else "last.pth")
        )

    if actor is not None:
        checkpoint = native / ("model_0.pt" if args.synth == "rsl-rl" else "last.pth")
        do_import(
            argparse.Namespace(
                source=args.synth, checkpoint=str(checkpoint), activation="elu", out=str(out)
            )
        )
        # The link the Rust oracle cannot see: the framework's own forward against the
        # reconstruction the importer builds out of the neutral weights, bitwise.
        tensors = {k: v.numpy() for k, v in read_safetensors(out / "weights.safetensors").items()}
        manifest = json.loads((out / "import.json").read_text())
        body = rebuild(tensors, manifest).body
        probe = _t(np.random.default_rng(0).normal(0, 1, (64, SYNTH_OBS)))
        with torch.no_grad():
            theirs = torch.cat([actor(probe[i : i + 1]) for i in range(64)], 0)
            ours = torch.cat([body(probe[i : i + 1]) for i in range(64)], 0)
        print(
            json.dumps(
                {
                    "native": args.synth,
                    "checkpoint": str(checkpoint),
                    "framework_vs_reconstruction_bitwise": bool(torch.equal(theirs, ours)),
                    "max_abs": float((theirs - ours).abs().max()),
                    "versions": manifest["versions"],
                }
            )
        )
        return

    if args.synth == "mujoco-playground":
        flat = {"obs_mean": obs_mean, "obs_std": obs_std}
        for i in range(n):
            flat[f"kernel_{i}"] = t[f"mlp.{i}.weight"].T.copy()
            flat[f"bias_{i}"] = t[f"mlp.{i}.bias"]
        flat["mean_kernel"] = t["head.weight"].T.copy()
        flat["mean_bias"] = t["head.bias"]
        flat["logstd_kernel"] = f32(rng.normal(0, 0.1, flat["mean_kernel"].shape))
        flat["logstd_bias"] = f32(rng.normal(0, 0.1, SYNTH_ACTION))
        np.savez(native / "source.npz", **flat)
        (native / "meta.json").write_text(
            json.dumps(
                {
                    "framework": "brax",
                    "synthetic": "import_rl.py --synth mujoco-playground",
                    "versions": {"brax": "0.14.2", "jax": "0.9.0"},
                    "activation": "swish",
                    "activate_output": False,
                    "squash": "tanh",
                    "joint_order": None,
                    "action": (
                        {
                            "kind": "joint_delta",
                            "unit": "rad per control tick",
                            "delta_scale": DELTA_SCALE,
                            "offset": [0.0] * SYNTH_ACTION,
                            "scale": [DELTA_SCALE] * SYNTH_ACTION,
                        }
                        if args.action == "delta"
                        else {}
                    ),
                },
                indent=2,
            )
        )
        checkpoint, activation = native, None
    elif args.synth == "rsl-rl":
        state = {f"actor.{2 * i}.weight": _t(t[f"mlp.{i}.weight"]) for i in range(n)}
        state.update({f"actor.{2 * i}.bias": _t(t[f"mlp.{i}.bias"]) for i in range(n)})
        state[f"actor.{2 * n}.weight"] = _t(t["head.weight"])
        state[f"actor.{2 * n}.bias"] = _t(t["head.bias"])
        state["std"] = _t(np.full(SYNTH_ACTION, 1.0, np.float32))
        torch.save({"model_state_dict": state, "iter": 0}, native / "model_0.pt")
        checkpoint, activation = native / "model_0.pt", "elu"
    else:
        state = {f"a2c_network.actor_mlp.{2 * i}.weight": _t(t[f"mlp.{i}.weight"]) for i in range(n)}
        state.update(
            {f"a2c_network.actor_mlp.{2 * i}.bias": _t(t[f"mlp.{i}.bias"]) for i in range(n)}
        )
        state["a2c_network.mu.weight"] = _t(t["head.weight"])
        state["a2c_network.mu.bias"] = _t(t["head.bias"])
        state["a2c_network.sigma"] = _t(np.zeros(SYNTH_ACTION, np.float32))
        state["running_mean_std.running_mean"] = _t(obs_mean)
        state["running_mean_std.running_var"] = _t(obs_std**2)
        torch.save({"model": state, "epoch": 0}, native / "last.pth")
        checkpoint, activation = native / "last.pth", "elu"

    do_import(
        argparse.Namespace(
            source=args.synth, checkpoint=str(checkpoint), activation=activation, out=str(out)
        )
    )
    # A committed fixture must be a function of this file alone: the real readers report the
    # installed package versions, which would move the fixture with whatever torch the
    # generator happened to have. `synthetic` is the provenance line a reader needs instead.
    manifest = json.loads((out / "import.json").read_text())
    manifest["versions"] = {"synthetic": "no framework was installed to produce this"}
    flags = f" --action {args.action}" if args.action != "position" else ""
    manifest["synthetic"] = f"python/es/import_rl.py --synth {args.synth}{flags}"
    (out / "import.json").write_text(json.dumps(manifest, indent=2) + "\n")


def do_reference(args) -> None:
    """Tier (a)/(b) inputs for the `--ignored` Rust oracle, as one safetensors file."""
    src = Path(args.import_dir).expanduser()
    manifest = json.loads((src / "import.json").read_text())
    tensors = {k: v.numpy() for k, v in read_safetensors(src / "weights.safetensors").items()}
    # The action tail the Rust half actually resolved. The adapter may override the
    # checkpoint's `scale`/`offset`, and that rule lives in `crates/es-data/src/rl_import.rs`
    # alone: the reference reads the resolved numbers back out of the mapping report rather
    # than deciding again and risking a different answer.
    if args.resolved:
        report = json.loads(Path(args.resolved).expanduser().read_text())
        manifest["action_scale"] = report["action_scale"]
        manifest["action_offset"] = report["action_offset"]
    model = rebuild(tensors, manifest)

    mean = f32(manifest["obs_mean"] or np.zeros(manifest["obs_dim"]))
    std = f32(manifest["obs_std"] or np.ones(manifest["obs_dim"]))
    out_tensors = {}
    if args.npz:
        npz = np.load(Path(args.npz).expanduser())
        obs = f32(npz["obs_scaled"])
        out_tensors["actions_scaled"] = _t(npz["actions_scaled"])
    else:
        rng = np.random.default_rng(args.seed)
        obs = f32(mean + std * rng.uniform(-1.0, 1.0, (args.rows, mean.shape[0])))

    # The Observation IR's `Normalize{MeanStd}` in numpy f32 -- computed once here so both
    # sides of the bitwise comparison see the same module input, byte for byte.
    normalized = ((obs - mean) / std).astype(np.float32)
    # One row at a time, shape `[1, obs_dim]`: that is exactly what `TorchRuntime` feeds the
    # lowered module (`crates/es-policy/python/torch_ref.py` unsqueezes each sample), and a
    # 1,000-row GEMM is not required to produce the same bits as a 1-row one. Tier (a) claims
    # *bitwise*, so the two sides must be the same call, not merely the same formula.
    with torch.no_grad():
        rows = [model(_t(normalized[i : i + 1])) for i in range(normalized.shape[0])]
    reference = torch.cat(rows, dim=0)
    out_tensors["obs_normalized"] = _t(normalized)
    out_tensors["reference"] = reference
    out = Path(args.out).expanduser()
    write_safetensors(out, out_tensors)
    print(
        json.dumps(
            {
                "rows": int(normalized.shape[0]),
                "obs_dim": int(normalized.shape[1]),
                "keys": sorted(out_tensors),
                "torch": torch.__version__,
                "out": str(out),
            }
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="a PPO actor -> safetensors + import.json")
    ap.add_argument("--from", dest="source", choices=FRAMEWORKS)
    ap.add_argument("--checkpoint")
    ap.add_argument("--activation", choices=ACTIVATIONS)
    ap.add_argument("--synth", choices=FRAMEWORKS)
    ap.add_argument(
        "--native",
        action="store_true",
        help="build and save the checkpoint with the framework's own API, when installed",
    )
    ap.add_argument(
        "--action",
        choices=("position", "delta"),
        default="position",
        help="--synth: whether the synthetic source's action is a position target or a "
        "per-tick joint increment (spec 8.5 JointDelta)",
    )
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--import", dest="import_dir")
    ap.add_argument("--resolved", help="mapping-report.json, for the resolved action tail")
    ap.add_argument("--npz")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.reference:
        if not args.import_dir:
            raise fail("--reference needs --import <dir> (an import.json + weights.safetensors)")
        do_reference(args)
    elif args.synth:
        do_synth(args)
    elif args.source:
        if not args.checkpoint:
            raise fail("--from needs --checkpoint")
        do_import(args)
    else:
        raise fail("one of --from, --synth or --reference is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
