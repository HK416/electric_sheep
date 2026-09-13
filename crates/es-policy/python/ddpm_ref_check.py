"""An *independent* DDPM/DDIM oracle: `diffusers` itself, not our lowering (spec 1.4).

One JSON request on stdin, one JSON object on stdout, as `torch_ref.py` does. Two commands:

    {"cmd": "schedule", ...config}
        -> {"alphas_cumprod": [...], "timesteps": [...], "variances": [...]}
      built by `DDPMScheduler`/`DDIMScheduler` + `set_timesteps` + `_get_variance`, so a
      difference from `es_policy::lower::torch::diffusion_schedule` is our bug, not a
      re-spelling of the same formula.

    {"cmd": "loop", ...config, "cond", "x_t", "weights", "noise"}
        -> {"x": [...]}
      the whole sampler, driven by `scheduler.step(model_output, t, sample)` directly. Only the
      denoiser (one hidden layer, sinusoidal timestep embedding) is rebuilt here, because that
      is the *network*, not the schedule; every coefficient comes from diffusers.

Config keys: num_train_timesteps, beta_schedule, variance_type, clip_sample, clip_sample_range,
n_steps, scheduler ("ddpm" | "ddim").

`{"ok": false, "error": ...}` on any failure, including a missing `diffusers` — the caller
turns that into a SKIP rather than a passing comparison. Nothing here loads a pickle (INV-16):
weights arrive as JSON numbers over the pipe.
"""

import json
import sys


def build(req, cls):
    return cls(
        num_train_timesteps=req["num_train_timesteps"],
        beta_schedule=req["beta_schedule"],
        clip_sample=req["clip_sample"],
        clip_sample_range=req["clip_sample_range"],
        prediction_type="epsilon",
        **({"variance_type": req["variance_type"]} if cls.__name__ == "DDPMScheduler" else {}),
    )


def scheduler_of(req):
    from diffusers import DDIMScheduler, DDPMScheduler

    cls = DDPMScheduler if req["scheduler"] == "ddpm" else DDIMScheduler
    sched = build(req, cls)
    sched.set_timesteps(req["n_steps"])
    return sched


def variances(sched, req):
    """`_get_variance` per inference step. DDIM's takes (t, prev_t); DDPM's takes t alone."""
    out = []
    ts = [int(t) for t in sched.timesteps]
    for i, t in enumerate(ts):
        if req["scheduler"] == "ddpm":
            out.append(float(sched._get_variance(t)))
        else:
            prev = ts[i + 1] if i + 1 < len(ts) else t - req["num_train_timesteps"] // req["n_steps"]
            out.append(float(sched._get_variance(t, prev)))
    return out


def sinusoidal(torch, t, dim):
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32)
    a = t * torch.exp(-9.210340371976184 * i / half)
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def loop(req):
    """The lowered head's sampler, with every coefficient coming from `scheduler.step`."""
    import torch

    import diffusers.schedulers.scheduling_ddpm as ddpm_mod

    sched = scheduler_of(req)
    w = dict((k, torch.tensor(v["data"], dtype=torch.float32).reshape(v["shape"])) for k, v in req["weights"].items())
    cond = torch.tensor(req["cond"], dtype=torch.float32)
    x = torch.tensor(req["x_t"], dtype=torch.float32)
    noise = dict((int(k), torch.tensor(v, dtype=torch.float32)) for k, v in req["noise"].items())

    def denoise(x, t):
        cat = torch.cat([x, cond, sinusoidal(torch, float(t), cond.shape[-1])], dim=-1)
        return torch.nn.functional.linear(
            torch.relu(torch.nn.functional.linear(cat, w["l0.weight"], w["l0.bias"])),
            w["l1.weight"],
            w["l1.bias"],
        )

    # DDPM's ancestral noise is drawn inside `step` via `randn_tensor`; the checkpoint's
    # `noise_<t>` buffer is substituted for it so both sides consume the same draw (spec 3.4).
    original = ddpm_mod.randn_tensor
    held = {}
    ddpm_mod.randn_tensor = lambda shape, **kw: held["z"].reshape(shape)
    try:
        with torch.inference_mode():
            for t in sched.timesteps:
                t = int(t)
                held["z"] = noise.get(t, torch.zeros_like(x))
                # `step` reads `shape[1]` to detect a learned variance, so it needs a batch
                # axis; the lowered head is batch-free (spec 8.5), hence the reshape pair.
                eps = denoise(x, t).reshape(1, -1)
                x = sched.step(eps, t, x.reshape(1, -1)).prev_sample.reshape(-1)
    finally:
        ddpm_mod.randn_tensor = original
    return {"x": [float(v) for v in x.reshape(-1)]}


def main():
    try:
        req = json.loads(sys.stdin.read())
        if req["cmd"] == "schedule":
            sched = scheduler_of(req)
            reply = {
                "ok": True,
                "alphas_cumprod": [float(v) for v in sched.alphas_cumprod],
                "timesteps": [int(t) for t in sched.timesteps],
                "variances": variances(sched, req),
            }
        elif req["cmd"] == "loop":
            reply = dict(loop(req), ok=True)
        else:
            raise ValueError("unknown command %r" % (req.get("cmd"),))
    except Exception as exc:  # A missing wheel is a SKIP on the Rust side, never a pass.
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()


main()
