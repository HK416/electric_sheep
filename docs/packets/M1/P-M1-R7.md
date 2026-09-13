# P-M1-R7 — a CI tier with the reference oracles installed

M1 review follow-up (`docs/reviews/M1.md`, Human decision): no CI environment has MuJoCo or
PyTorch installed, so the PR gate is green with zero physics and zero learning cross-checks
ever executed — gates 2, 5 and 6 of spec 28.7 are claimed, not measured. Spec: spec 1.4
(reference oracles are mandatory, not optional), spec 26.2 (PR gate stays under 10 minutes),
spec 28.7 (the gate list this tier exists to actually run).

## context

```
.github/workflows/ci.yml
docs/packets/M1/P-M1-R7.md
```

## spec

- **A second job, `oracles`, added to `.github/workflows/ci.yml`.** The existing `ci` job (PR
  gate, spec 26.2's < 10 min budget) is untouched — `cargo xtask ci` runs exactly as before, on
  every PR and every push.
- **Trigger.** `oracles` runs `if: github.event_name == 'push' || github.event_name ==
  'workflow_dispatch'` only — never on `pull_request` — because installing Python plus the
  `mujoco` and `torch` wheels does not fit the PR job's 10-minute budget and would make every PR
  pay for a check only `main` needs. `workflow_dispatch:` is added to the workflow's top-level
  `on:` block so the job can also be run by hand. `continue-on-error: false` (the default, set
  explicitly): a failure here is a real CI failure, not an informational job.
- **Environment.** `ubuntu-latest`, `actions/setup-python@v5` at `python-version: "3.12"`, then
  `pip install mujoco==3.13.0` (pinned in `docs/api-notes/mujoco.md`) and `pip install
  torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu` (pinned, together with
  the `torch==2.14.0` it pulls in transitively, in `docs/api-notes/torchvision.md` and
  `docs/api-notes/torch.md`) — installing `torchvision` rather than `torch` directly also covers
  `es-compile`'s observation-golden oracle (`docs/packets/M1/P-M1-R1.md`), which needs
  `torchvision.transforms.functional` and runs in this same job. `ES_PYTHON=python` is exported
  for the test step so every oracle in the workspace (the `proc.rs` / `mjwarp.rs` /
  `torch_runtime.rs` interpreter search) finds this exact interpreter instead of falling back to
  a bare `python`/`python3` search.
- **The test step.** `cargo test -p es-physics-backend -p es-policy -p es-compile -- --skip
  mjwarp --nocapture 2>&1 | tee oracles.log` under `set -euo pipefail` (so a test failure fails
  the step even though its output is piped through `tee`), followed by `! grep -q '^SKIP'
  oracles.log`. Every oracle-gated test in these three crates prints a line starting `SKIP` (or
  `SKIPPED`, which also matches `^SKIP`) when its dependency is not found and passes anyway
  (spec 1.4's own convention, e.g. `crates/es-physics-backend/src/mujoco.rs`,
  `crates/es-policy/src/reference.rs`); with `mujoco`/`torch` actually installed and `ES_PYTHON`
  pointed at them, none of the tests this job runs should still hit that path, so the grep
  turning up nothing is what makes the job meaningful rather than a second copy of the PR job.
- **`mujoco_warp` is excluded, not faked.** `mjwarp.rs`'s two oracle tests (`mjwarp_pendulum`,
  `mjwarp_against_mujoco_cpu`) need a GPU (`mujoco_warp` + `warp`), which `ubuntu-latest` does
  not have. `--skip mjwarp` removes them from this run entirely rather than letting them print
  `SKIP` and fail the grep, or installing a GPU-only dependency that cannot actually run here.
  Per spec 12.4, this stays `Target / Status: unverified` — named in this doc, not hidden.

## oracle

```
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8'))"
```

(The job itself is only exercisable by GitHub Actions on a push to `main` or a manual dispatch;
a local review is limited to confirming the YAML parses and the job's steps match this doc.)

## acceptance

- `.github/workflows/ci.yml` parses as valid YAML with two jobs, `ci` and `oracles`.
- The `ci` job's steps are byte-for-byte unchanged from before this packet (still exactly
  `actions/checkout@v4` with `fetch-depth: 0`, `dtolnay/rust-toolchain@stable` with `rustfmt,
  clippy`, `Swatinem/rust-cache@v2`, `cargo xtask ci`).
- The `oracles` job's `if:` restricts it to `push` and `workflow_dispatch`; it installs
  `mujoco==3.13.0` and `torchvision==0.29.0` (the versions in `docs/api-notes/mujoco.md`,
  `docs/api-notes/torch.md` and `docs/api-notes/torchvision.md`), sets `ES_PYTHON`, runs the
  three named crates' tests with `mjwarp` skipped, and fails if any surviving test line starts
  with `SKIP`.

## forbidden

- Any change to the `ci` job's steps, triggers, or `timeout-minutes` — it must stay under the
  spec 26.2 PR budget exactly as it does today.
- `crates/es-compile/src/bundle.rs`, `crates/es-telemetry/src/transport.rs` and their design
  docs — P-M1-R4 and P-M1-R5's scope.
- Marking `mujoco_warp` as verified, or installing `mujoco_warp`/`warp` — no GPU runner is being
  added by this packet.
- Committing.
