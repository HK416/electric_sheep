# W1 (follow-up) — MJWarp on real hardware: make the live tests run

Spec: spec 17.1 (MJWarp is the batched GPU backend), spec 17.3 (determinism tiers: a GPU backend
declares tier 2 or 3, only `mujoco-cpu` may claim tier 1), spec 3.5 (tier definitions),
spec 12.1 (the simulation batch domain), spec 12.4 (report measured numbers, never a single
figure), spec 1.7 (hallucinated API names are a standing agent failure mode), spec 4.3
(backends are the default path). M1 Wave 1, follow-up to `W1-mjwarp-adapter.md`. Review class B.

`W1-mjwarp-adapter.md` shipped `MjWarpBackend` protocol-complete and **engine-unproven**: the
implementing machine had no CUDA device, so every live test skipped and every API name in
`docs/api-notes/mujoco-warp.md` was marked *unverified*. This packet closes that: a machine with
an RTX 4060 Laptop GPU and `mujoco-warp` installed, and the honest measurements that follow.

## context

```
crates/es-physics-backend/python/mjwarp_ref.py
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/proc.rs          (shared spawn / call / drop; see below)
docs/api-notes/mujoco-warp.md
docs/packets/M1/W1-mjwarp-live.md
```

## forbidden

`mapping.rs` semantics beyond what verification changes, `mujoco.rs`, `mjcf_out.rs`, and every
crate outside `es-physics-backend`. The `es` CLI (`es backend compare`) is a separate packet.

## spec

1. Run the live tests against the installed `mujoco_warp` and make them pass, fixing
   `mjwarp_ref.py` against the **installed** API rather than the guessed one.
2. Measure determinism honestly: two runs from the same written state, bit-identical or not,
   with the observed tolerance recorded.
3. Measure `compare_backends(mujoco-cpu, mjwarp)` over 200 ticks and record `max |dqpos|`.
4. Update `docs/api-notes/mujoco-warp.md`: drop *unverified* from what ran, pin the real
   versions, and keep an explicit list of what is still unchecked.

## oracle

```
ES_PYTHON=<venv python> cargo test -p es-physics-backend -- --nocapture
```

Live tests print `RAN <name>` when they execute and `SKIP <name>: <reason>` otherwise, so the
gate can tell a passing run from a vacuously skipped one:

```
cargo fmt -p es-physics-backend --check
cargo clippy -p es-physics-backend --all-targets -- -D warnings
ES_PYTHON=... cargo test -p es-physics-backend -- --nocapture | grep -E 'SKIP|RAN|qpos|test result'
```

## what was actually wrong

**One bug, and it was not an API name.** Every name the adapter guessed — `put_model`,
`put_data(..., nworld=)`, `step`, `forward`, the `Data` field names, the `nworld`-major layout —
was correct against `mujoco-warp 3.13.0`. What failed was that **`warp` prints its
initialization banner on stdout**, and stdout is the JSON-lines protocol:

```
Protocol("expected value at line 1 column 1 in `Warp 1.17.0 initialized:`")
```

Fix: take the real stdout handle before any import and point `sys.stdout` at `sys.stderr`, which
the Rust side already discards. Two lines. The lesson worth keeping is that a Python library's
*side effects on stdout* are as much a part of its API as its function names, and a protocol
that owns stdout must defend it.

The one genuinely wrong pin was the version: the note claimed `mujoco-warp==0.1.0`; the package
versions in lockstep with `mujoco` and the installed release is **3.13.0**.

## accepted

- `mjwarp_pendulum` **RAN**: loads, steps 100 ticks, finite, `n_envs = 2` independent, resets.
- `mjwarp_against_mujoco_cpu` **RAN**.
- `mjwarp_runs_agree_to_the_declared_tier` **RAN** (new).
- Versions pinned: `mujoco-warp 3.13.0`, `warp-lang 1.17.0`, `mujoco 3.13.0`, CPython 3.12,
  CUDA Toolkit 12.9 / driver 13.1, RTX 4060 Laptop GPU (8 GiB, sm_89).

### measured

| Measurement | Value |
|---|---|
| Run-to-run, 200 ticks, 2 worlds, same written state | **bit-identical**, `max \|delta\| = 0` |
| `max \|dqpos\|` vs `mujoco-cpu`, 200 ticks | **7.41e-8** |
| `max \|dqvel\|` vs `mujoco-cpu`, 200 ticks | 7.76e-7 |
| energy proxy delta | 3.16e-6 |
| divergence tick (1e-6 tolerance) | none |

**The declared tier stays 2 (`CrossBackend`).** Spec 17.3 is explicit that only `mujoco-cpu`
declares tier 1, and one scene reproducing on one GPU with one driver is an observation, not a
guarantee. So `mjwarp_runs_agree_to_the_declared_tier` asserts the *tier-2* contract
(`delta < 1e-9`) and merely prints whether the run happened to be bitwise. A backend that is
reproducible today must not become a test that fails on the next driver update.

## a trap this packet fell into, and the fix

The first passing `mjwarp_against_mujoco_cpu` reported `max |dqpos| = 0.000000e0` — and it was
**meaningless**. `compare_backends` resets and steps without control, and the fixture pendulum
hangs straight down, which is an equilibrium: both backends sat at zero for 200 ticks and the
comparison compared nothing. A green test that proves nothing is worse than a red one.

The compare tests now use a rod that starts **horizontal**, so `qpos = 0` is not an equilibrium
and the 200 ticks actually move. Any future cross-backend fixture needs the same check: assert
the trajectory is non-trivial, or the tolerance is unearned.

## incidental: the third out-of-process backend arrived

`mjwarp.rs` carried a `ponytail:` note saying its duplicated spawn / call / drop trio should
fold into `proc.rs` "when a third out-of-process backend appears". `W4-newton-backend.md` is
that third backend, so the fold happened here: `proc.rs` gained
`Process::spawn_with(script, engine)`, `import_available(modules, what)` and `model_info(...)`
(the shared `LoadReply` → `ModelInfo` name resolution), and `mjwarp.rs` lost 174 lines. Net
change across the two files is **−63 lines**, and `newton.rs` inherits all of it.

## still open

- `contact.condim = 6` unverified.
- Batch widths beyond 2 worlds. `MAX_ENVS = 8192` is declared, not measured (spec 12.4).
- Contact-rich scenes: everything measured is a single hinge with no collisions.
- `docs/api-notes/mujoco-warp.ko.md` still carries the old *unverified* banner and the wrong
  `0.1.0` pin. It is outside this packet's scope; it needs a follow-up `docs:` commit.
