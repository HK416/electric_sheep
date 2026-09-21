"""PPO over rollouts in *our* `Env`, through *our* Safety Plane (M8/S4b; spec 13.4, 2.3).

The sibling of `train_act.py`, and deliberately its shape: it owns the optimizer and nothing
above it, it `exec`s `<module dir>/es_policy.py` -- the file `es_policy::lower::lower_to_torch`
generated from the bundle's `LearningGraph` -- and it **defines no layer** for the policy. The
architecture comes from the Learning IR or it does not come.

Usage:

    train_ppo.py --module <dir> --rollout-docs <dir> --out model.safetensors
                 --value-out value.safetensors --iterations N
                 --envs N --horizon N --epochs N --minibatches N
                 --gamma F --lam F --clip F --entropy F --value-coef F
                 [--init-log-std F] [--init-weights init.safetensors]
                 [--lr F] [--seed N] [--device cpu] [--grad-clip F] [--weight-decay F]
                 [--schedule constant|warmup_cosine] [--warmup-steps N] [--lr-min F]
                 [--checkpoint-at 0 | 0,50,200] [--loss-curve curve.json]
                 [--progress-every N]

Prints one JSON line on stdout and nothing else -- the same contract `es train` reads for
`train_act.py`: `torch` goes into spec 19.3's `hardware.json` and `optimizer` is compared
against the `optimizer.json` `es train` declared *before* the run.

**`--rollout-docs` is one directory, not four paths** (`es train` writes it out of the policy
bundle): `task.toml`, `observation.toml`, `deployment.toml` and `scene.xml`. They come out of
one bundle on purpose -- a trainer stepping an env declared by anything but the policy's own
documents is measuring a different thing than the evaluation will.

**Three rules this file exists to keep** (`docs/design/rl-continuation.md` sections 1-3):

 1. *PPO is a trainer, not an IR.* The deployed graph is whatever the bundle says; the value
    MLP, the `log_std`, GAE, the clipped objective and the entropy bonus are here and in spec
    19.3's `training/`. They move `training_hash` and never `learning_hash`. The value network
    is written to `--value-out` for resumption and is **never** packed into a bundle -- and
    could not be, because the lowered module declares no such tensor.
 2. *Rollouts are `es-env`, and the Safety Plane is on.* Every step goes through
    `es_native.Rollout`, which runs `observe_state -> heartbeat -> validate -> Env::step` in
    `es_eval::runner::run_episode`'s own order. This file calls no simulator of its own, and
    there is no flag here that turns the plane off (`INV-12`). What the plane did is measured
    rather than hidden: `envelope_violation_rate` is how often it raised an event and
    `executed_ne_sampled_rate` is how often what reached the actuator was not what was
    sampled.
 3. *The Gaussian is around the module's own output, in actuator units.* `lower_to_torch`
    emits one `forward(obs) -> action` that already contains the head's squash and the
    `Normalizer{Inverse}`, so `mu = module(obs)` is the deployed function unchanged and

        a = mu + exp(log_std) * eps

    is rsl_rl's model. `log_std` is state-independent, `[action_dim]` wide, and training-only.
    A brax-imported policy carries `tanh` inside `mu`, so this is not brax's
    `NormalTanhDistribution`; that is accepted and recorded (design note section 2) -- the
    imported *deterministic* policy is reproduced exactly by S2b, and continuation is a new
    training run whose distribution is ours.

**Horizon 1, synchronous** (design note section 3). PPO acts on every control tick: the chunk
handed to the plane carries one row, `expected_latency_ms` is 0, and there is no chunk buffer.
The *evaluation* of the same policy runs under the Deployment IR's declared latency through
`es eval run`, which is what keeps that number honest rather than the trainer's own.

**Determinism** (spec 3.5 tier 1, CPU backend). `torch.use_deterministic_algorithms(True)`,
one `torch.Generator` seeded per run for the action noise and one for the minibatch
permutation, and the env's own RNG seeded through `Rollout(seed=...)`. No global RNG is read.
Two runs of one recipe on one machine give bitwise-equal checkpoints, which
`crates/es/tests/cli.rs::train_rl_two_runs_are_bitwise` is the oracle for.

**What is recorded.** `--loss-curve` is a JSON array with one object per iteration carrying
`loss`, `policy_loss`, `value_loss`, `entropy`, `return`, `episode_len`,
`envelope_violation_rate`, `executed_ne_sampled_rate` and `samples_per_sec`. `--progress-every
N` additionally prints `{"progress": {"step", "loss", "lr", "samples_per_s", "elapsed_s"}}`
every N iterations, which `es train --telemetry` republishes on stream 5 so the editor's Live
tab draws it unchanged (packet M7/E7). Without the flag nothing is printed and nothing is
computed for a watcher who is not there.

No format that can execute code on load is read or written here (`INV-16`): the checkpoints
and the value network are safetensors in the layout `crates/es-policy/src/weights.rs`
implements, read and written by `train_act.py`'s own two functions rather than a second copy
of them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

# The sibling trainer, by path, for the four things both files need: the module loader, the
# two safetensors halves and the checkpoint key mapping. A second implementation of the
# safetensors layout is how two trainers quietly start writing two formats (packet M8/S4b).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_act import (  # noqa: E402
    build_policy,
    checkpoint_tensors,
    init_weights as load_init_weights,
    lr_at,
    lr_curve_hash,
    read_safetensors,
    write_safetensors,
)

try:
    import es_native
except ImportError:  # pragma: no cover - reported as a stopped run, never a silent one
    try:
        from es import es_native  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "es_native is not importable (%s). It is the pyo3 extension this trainer steps "
            "the env through; build it once with `cd python/es && maturin develop --release "
            "--features python` in the interpreter that runs this file." % exc
        )


# --- the two things this file is allowed to define ------------------------------------------


class Value(torch.nn.Module):
    """The value network: an MLP over the concatenation of the Observation IR's output ports.

    It is **not** an IR node and never becomes one (`rl-continuation.md` rule 1). PPO needs a
    baseline, the baseline is not deployed, and a `LearningNode` for it would put a tensor
    into `policy_hash` that no deployment ever evaluates. Two hidden layers of 64 is the
    reference PPO baseline's shape and is deliberately not a flag: a value head that needed
    tuning per task would be an IR decision wearing a command-line flag's clothes.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalized advantage estimation over `[horizon, envs]` tensors.

    `dones[t]` is whether the episode ended *on* step `t`, so it is what cuts the bootstrap
    between `t` and `t + 1` -- the same place `Rollout.act` re-seeded the plane and the plan
    (`begin_episode`), which is what makes the cut the episode boundary and not an arbitrary
    one. Written as the plain backward recursion rather than a vectorised trick: this is the
    definition, it runs once per iteration, and it is the thing a reader checks.
    """
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in range(horizon - 1, -1, -1):
        alive = 1.0 - dones[t]
        nxt = last_value if t == horizon - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values


# --- the rollout ----------------------------------------------------------------------------


def open_rollout(docs: Path, seed: int, n_envs: int):
    """`es_native.Rollout` from the four files `es train` wrote under `--rollout-docs`."""
    read = lambda name: (docs / name).read_text(encoding="utf-8")  # noqa: E731
    try:
        return es_native.Rollout(
            read("task.toml"),
            read("observation.toml"),
            read("deployment.toml"),
            read("scene.xml"),
            seed,
            n_envs,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "%s: --rollout-docs is the directory `es train` writes out of the policy bundle "
            "and holds task.toml, observation.toml, deployment.toml and scene.xml (%s)"
            % (docs, exc)
        )


def observe(roll, ports: list, n_envs: int) -> dict:
    """`{port: [n_envs, dim]}` as f32 tensors, in the port order fixed once by the caller."""
    raw = roll.observe()
    missing = [p for p in ports if p not in raw]
    if missing:
        raise SystemExit(
            "the Observation IR produced %s and the module declares %s; they come from one "
            "bundle, so this means --module and --rollout-docs were taken from two"
            % (sorted(raw), missing)
        )
    return {
        p: torch.tensor(raw[p], dtype=torch.float32).reshape(n_envs, -1) for p in ports
    }


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", required=True, type=Path)
    p.add_argument(
        "--rollout-docs",
        required=True,
        type=Path,
        help="the directory holding task.toml, observation.toml, deployment.toml and "
        "scene.xml, as `es train` writes it out of the policy bundle",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--value-out",
        type=Path,
        help="where the value network is written for resumption. It is training-only state "
        "and is never packed into a bundle",
    )
    p.add_argument("--iterations", type=int, required=True, help="[run] steps")
    p.add_argument("--envs", type=int, required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--minibatches", type=int, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--clip", type=float, required=True)
    p.add_argument("--entropy", type=float, required=True)
    p.add_argument("--value-coef", type=float, required=True)
    p.add_argument(
        "--init-log-std",
        type=float,
        default=None,
        help="the Gaussian's initial log standard deviation. Absent is -0.5; an importer's "
        "own value reaches this flag through `[rl] init_log_std`",
    )
    p.add_argument(
        "--init-weights",
        type=Path,
        help="the intersection `es train` copied out of `[init] policy` (packet M8/S1)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--schedule", choices=["constant", "warmup_cosine"], default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-min", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-at", default="", help="0 | 0,50,200 -- iteration marks")
    p.add_argument("--loss-curve", type=Path, help="one object per iteration, as JSON")
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help='print one {"progress": {...}} line every N iterations; 0 (the default) prints '
        "none",
    )
    a = p.parse_args(argv)

    if a.device != "cpu":
        # Not a refusal: spec 28.11 puts GPU throughput on a later rung, and a run on another
        # device is simply not the bitwise-reproducible one spec 3.5 tier 1 describes.
        sys.stderr.write(
            "note: --device %s; the bitwise tier of spec 3.5 is the CPU backend's\n" % a.device
        )
    # Every RNG this run reads is seeded and local. `use_deterministic_algorithms` turns a
    # non-deterministic kernel into an error rather than a silently different number, which is
    # the only way the two-runs oracle can mean anything.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    contract = json.loads((a.module / "contract.json").read_text(encoding="utf-8"))
    if not contract.get("batch_axis"):
        raise SystemExit(
            "%s/contract.json does not declare batch_axis; re-run `es policy lower`" % a.module
        )
    shapes = {port: [int(d) for d in shape] for port, shape in contract["inputs"].items()}
    ports = sorted(shapes)
    action_dim = int(contract["action_dim"])

    actor = build_policy(a.module).to(device)
    initialised = (
        load_init_weights(actor, read_safetensors(a.init_weights)) if a.init_weights else []
    )

    roll = open_rollout(a.rollout_docs, a.seed, a.envs)
    model = roll.model()
    if int(model["nu"]) != action_dim:
        raise SystemExit(
            "the module emits %d action channels and the scene has %d actuators; the Learning "
            "IR and the Deployment IR come from one bundle, so this is a document mismatch"
            % (action_dim, int(model["nu"]))
        )
    roll.reset(None)

    obs = observe(roll, ports, a.envs)
    obs_dim = sum(int(t.shape[1]) for t in obs.values())
    value = Value(obs_dim).to(device)

    # Training-only state, `[action_dim]` wide and state-independent (design note section 2).
    # `-0.5` when nobody said: exp(-0.5) ~ 0.61 of an actuator unit, wide enough to explore a
    # normalized action space and narrow enough that the plane is clamping rather than latching.
    log_std = torch.nn.Parameter(
        torch.full(
            (action_dim,),
            -0.5 if a.init_log_std is None else float(a.init_log_std),
            device=device,
        )
    )

    trainable = [q for q in actor.parameters() if q.requires_grad]
    if not trainable:
        raise SystemExit("every actor parameter is frozen; there is nothing to optimize")
    optimizer = torch.optim.Adam(
        trainable + list(value.parameters()) + [log_std],
        lr=a.lr,
        weight_decay=a.weight_decay,
    )
    # One stream for the action noise and one for the minibatch order, both local and both
    # seeded: `torch.randn` off the global RNG would make the run depend on whatever else
    # touched it (spec 3.4 forbids exactly that).
    noise_rng = torch.Generator(device="cpu").manual_seed(a.seed)
    order_rng = torch.Generator(device="cpu").manual_seed(a.seed + 1)

    marks = sorted({int(s) for s in a.checkpoint_at.split(",") if s.strip()})
    stem, suffix = str(a.out.with_suffix("")), a.out.suffix
    write_checkpoint = lambda mark: write_safetensors(  # noqa: E731
        Path("%s-%d%s" % (stem, mark, suffix)), checkpoint_tensors(actor)
    )
    # Mark 0 is the state *before* the first update, so it is written before the loop: with
    # `[init]` that is the policy this run continues, bit for bit, which is oracle 3.
    if 0 in marks:
        write_checkpoint(0)

    rows = a.envs * a.horizon
    minibatch = rows // a.minibatches
    curve, applied_lr = [], []
    # Per-env episode bookkeeping that survives an iteration boundary: an episode is not the
    # same thing as a rollout segment, and reporting the segment sum as "return" would make
    # the number a property of `--horizon` (design note section 6).
    episode_return = torch.zeros(a.envs)
    episode_steps = torch.zeros(a.envs)
    started = time.perf_counter()

    for iteration in range(a.iterations):
        lr_now = (
            a.lr
            if a.schedule == "constant"
            else lr_at(iteration, a.iterations, a.lr, a.lr_min, a.warmup_steps)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        applied_lr.append(lr_now)

        # --- collect ------------------------------------------------------------------
        buf_obs = {p: torch.zeros(a.horizon, a.envs, shapes[p][0]) for p in ports}
        buf_flat = torch.zeros(a.horizon, a.envs, obs_dim)
        buf_act = torch.zeros(a.horizon, a.envs, action_dim)
        buf_logp = torch.zeros(a.horizon, a.envs)
        buf_val = torch.zeros(a.horizon, a.envs)
        buf_rew = torch.zeros(a.horizon, a.envs)
        buf_done = torch.zeros(a.horizon, a.envs)
        violations, clamped, finished = 0, 0, []

        with torch.no_grad():
            for t in range(a.horizon):
                flat = torch.cat([obs[p] for p in ports], dim=1)
                mu = next(iter(actor(**obs).values()))[:, 0, :]
                std = log_std.exp()
                eps = torch.randn(mu.shape, generator=noise_rng)
                action = mu + std * eps
                logp = normal_logp(action, mu, log_std)

                for q in ports:
                    buf_obs[q][t] = obs[q]
                buf_flat[t] = flat
                buf_act[t] = action
                buf_logp[t] = logp
                buf_val[t] = value(flat)

                executed, events, rewards, dones = roll.act(
                    action.reshape(-1).double().tolist()
                )
                executed = torch.tensor(executed, dtype=torch.float32).reshape(
                    a.envs, action_dim
                )
                violations += sum(1 for bits in events if bits != 0)
                clamped += int((executed != action).any(dim=1).sum())
                buf_rew[t] = torch.tensor(rewards, dtype=torch.float32)
                buf_done[t] = torch.tensor(
                    [1.0 if d else 0.0 for d in dones], dtype=torch.float32
                )

                episode_return += buf_rew[t]
                episode_steps += 1.0
                for i, done in enumerate(dones):
                    if done:
                        finished.append((float(episode_return[i]), float(episode_steps[i])))
                        episode_return[i] = 0.0
                        episode_steps[i] = 0.0
                obs = observe(roll, ports, a.envs)

            last_value = value(torch.cat([obs[p] for p in ports], dim=1))
            advantages, returns = gae(
                buf_rew, buf_val, buf_done, last_value, a.gamma, a.lam
            )

        # --- update -------------------------------------------------------------------
        flat_obs = {p: buf_obs[p].reshape(rows, -1) for p in ports}
        flat_state = buf_flat.reshape(rows, obs_dim)
        flat_act = buf_act.reshape(rows, action_dim)
        flat_logp = buf_logp.reshape(rows)
        flat_adv = advantages.reshape(rows)
        flat_ret = returns.reshape(rows)
        # Normalized over the whole iteration rather than per minibatch: the scale of the
        # advantage is a property of the data the iteration collected, and re-normalizing per
        # minibatch would make the gradient depend on how the rows were cut.
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        last = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
        for _ in range(a.epochs):
            perm = torch.randperm(rows, generator=order_rng)
            for start in range(0, rows, minibatch):
                idx = perm[start : start + minibatch]
                inputs = {p: flat_obs[p][idx] for p in ports}
                mu = next(iter(actor(**inputs).values()))[:, 0, :]
                logp = normal_logp(flat_act[idx], mu, log_std)
                ratio = (logp - flat_logp[idx]).exp()
                adv = flat_adv[idx]
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1.0 - a.clip, 1.0 + a.clip) * adv
                ).mean()
                value_loss = (value(flat_state[idx]) - flat_ret[idx]).pow(2).mean()
                # The Gaussian's own differential entropy, which is a function of `log_std`
                # alone -- no sampling estimate, so the bonus is exact and deterministic.
                entropy = (log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum()
                loss = policy_loss + a.value_coef * value_loss - a.entropy * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable + list(value.parameters()) + [log_std], a.grad_clip
                    )
                optimizer.step()
                last = {
                    "loss": float(loss.detach()),
                    "policy": float(policy_loss.detach()),
                    "value": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                }

        elapsed = time.perf_counter() - started
        curve.append(
            {
                "loss": last["loss"],
                "policy_loss": last["policy"],
                "value_loss": last["value"],
                "entropy": last["entropy"],
                # The undiscounted reward this iteration's segment collected, per env. It
                # scales with `--horizon`, which is fixed for a run, and that is the price of
                # the property a learning curve actually needs: **every row means the same
                # thing**. The obvious alternative -- the mean return of the episodes that
                # *ended* in this iteration -- is not available in most rows (the demo's
                # episodes are 1800 control steps and a segment is 64), so a curve built on
                # it silently alternates between two quantities and cannot be read.
                "return": float(buf_rew.sum(dim=0).mean()),
                # The mean length of the episodes that ended here, and `--horizon` when none
                # did. Unlike `return` this one is allowed to be occasional: it is read as
                # "when an episode ended, how long was it", not as a curve.
                "episode_len": (
                    sum(n for _, n in finished) / len(finished)
                    if finished
                    else float(a.horizon)
                ),
                # What the plane did, per (env, step) row. Never hidden and never zero by
                # construction: a run whose policy lives entirely inside the envelope reports
                # 0.0 because it measured 0, not because nobody looked (INV-12).
                "envelope_violation_rate": violations / rows,
                "executed_ne_sampled_rate": clamped / rows,
                "samples_per_sec": (iteration + 1) * rows / elapsed if elapsed > 0 else 0.0,
            }
        )
        if a.progress_every > 0 and (iteration + 1) % a.progress_every == 0:
            sys.stdout.write(
                json.dumps(
                    {
                        "progress": {
                            "step": iteration + 1,
                            "loss": last["loss"],
                            "lr": lr_now,
                            "samples_per_s": curve[-1]["samples_per_sec"],
                            "elapsed_s": elapsed,
                        }
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
        if (iteration + 1) in marks:
            write_checkpoint(iteration + 1)

    write_safetensors(a.out, checkpoint_tensors(actor))
    if a.value_out:
        # `value.*` and `log_std`, in one file, under `training/`: the state a resumed run
        # needs and a deployment never does. `es policy pack` would refuse these keys, which
        # is the structural half of "never packed" (design note rule 1).
        tensors = {"log_std": log_std.detach()}
        tensors.update({"value." + k: v for k, v in value.state_dict().items()})
        write_safetensors(a.value_out, tensors)
    if a.loss_curve:
        a.loss_curve.write_text(json.dumps(curve), encoding="utf-8")
        # The nine spec 12.4 fields as the env measured them, beside the curve and **not** in
        # the summary: they are a measurement of the machine, so putting them in the summary
        # would put them in `metrics.json`, which is a `training_hash` slot. A domain this
        # path never runs (render, inference, VRAM) stays `null` rather than a fabricated
        # zero -- `Rollout::metrics` is what decides that, not this file. There is
        # deliberately no single `step/s`.
        (a.loss_curve.parent / "env-metrics.json").write_text(
            json.dumps(
                {
                    "metrics": roll.metrics(),
                    "control_ticks": roll.tick(),
                    "iterations": len(curve),
                    "envs": a.envs,
                    "horizon": a.horizon,
                    "wall_clock_s": time.perf_counter() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    window = max(1, len(curve) // 10)
    group = optimizer.param_groups[0]
    report = {
        "initial_loss": sum(c["loss"] for c in curve[:window]) / window if curve else 0.0,
        "final_loss": sum(c["loss"] for c in curve[-window:]) / window if curve else 0.0,
        "initial_return": sum(c["return"] for c in curve[:window]) / window if curve else 0.0,
        "final_return": sum(c["return"] for c in curve[-window:]) / window if curve else 0.0,
        "steps": len(curve),
        "first_nonfinite_step": next(
            (i for i, c in enumerate(curve) if not math.isfinite(c["loss"])), None
        ),
        "algo": "ppo",
        "envs": a.envs,
        "horizon": a.horizon,
        "rows_per_iteration": rows,
        "epochs": a.epochs,
        "minibatches": a.minibatches,
        "gamma": a.gamma,
        "lam": a.lam,
        "clip": a.clip,
        "entropy_coef": a.entropy,
        "value_coef": a.value_coef,
        "init_log_std": -0.5 if a.init_log_std is None else float(a.init_log_std),
        "final_log_std": [float(v) for v in log_std.detach().tolist()],
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "ports": ports,
        # The plane's own account of the whole run, averaged over iterations. It is in the
        # summary because a run whose actions were clamped half the time trained on a
        # distribution it did not sample (design note open question 2).
        "envelope_violation_rate": (
            sum(c["envelope_violation_rate"] for c in curve) / len(curve) if curve else 0.0
        ),
        "executed_ne_sampled_rate": (
            sum(c["executed_ne_sampled_rate"] for c in curve) / len(curve) if curve else 0.0
        ),
        "schedule": a.schedule,
        "warmup_steps": a.warmup_steps,
        "lr_min": a.lr_min,
        "weight_decay": a.weight_decay,
        "grad_clip": a.grad_clip,
        "lr_curve_hash": lr_curve_hash(applied_lr),
        # Whether, not where. `es train` puts this summary into spec 19.3's `metrics.json`,
        # which is a `training_hash` slot: a path here would be the *output directory* the
        # caller happened to choose, so two runs of one recipe into two directories would
        # have two `training_hash`es and spec 3.5 tier 1 could not hold. What was loaded is
        # `initialised_from` below (names, not paths), and the provenance of the file itself
        # is `training/init.lock`'s job, not this file's.
        "init_weights": a.init_weights is not None,
        "initialised_from": initialised,
        "wrote_value": a.value_out is not None,
        "trainable_parameters": sum(q.numel() for q in trainable),
        "value_parameters": sum(q.numel() for q in value.parameters()),
        "torch": torch.__version__,
        "optimizer": {
            "kind": "Adam",
            "lr": a.lr,
            "betas": list(group["betas"]),
            "eps": group["eps"],
            "weight_decay": group["weight_decay"],
        },
    }
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


def normal_logp(action, mu, log_std):
    """log N(action; mu, exp(log_std)), summed over the action channels.

    Spelled out rather than `torch.distributions.Normal(...).log_prob(...).sum(-1)`: the
    distribution object's arithmetic is free to change between torch versions, and this is
    three operations whose order is the thing the bitwise oracle pins.
    """
    var = (2.0 * log_std).exp()
    return -0.5 * (
        (action - mu).pow(2) / var + 2.0 * log_std + math.log(2.0 * math.pi)
    ).sum(-1)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
