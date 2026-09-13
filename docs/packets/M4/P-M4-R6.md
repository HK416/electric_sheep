# P-M4-R6 — a GPU SKIP can fail a required run

Spec: §1.4 (the oracle exists whether or not the machine can run it — and says which),
§1.6 / §26.2 (`xtask` is the single verification entry point, PR tier < 10 min).
Review finding: `docs/reviews/M4.md` S-7 — `es-gpu/tests/determinism.rs:35,46` and
`es-render/tests/render.rs:57` print `SKIP` and then report `ok`, while
`xtask/src/main.rs:45` ran `cargo test` **without `--nocapture`**, so the lines never
reached the log. A GPU-less box reported green for the GPU work. The `! grep -q '^SKIP'`
guard exists in `.github/workflows/ci.yml`, but only over the `oracles` job's three Python
crates. Also S-12 — `es-gpu/src/buffer.rs:144` returned `allocation.size()` bytes, padded by
`gpu-allocator`, so `download_f32().len()` could exceed the requested count; and S-7's second
half — `es-policy/tests/act_checkpoint.rs:113` folded a genuine reference error into the same
SKIP as "lerobot not installed".

## context

```
xtask/src/main.rs
crates/es-gpu/src/buffer.rs
crates/es-gpu/tests/determinism.rs
crates/es-policy/tests/act_checkpoint.rs
.github/workflows/ci.yml
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R6.md
```

## spec

- `xtask/src/main.rs`: `cmd_ci`'s test step becomes `run_tests(root)`, which spawns
  `cargo test --workspace --features es-ir/testing -- --nocapture` with a piped stdout,
  **forwards every line** as it arrives (so the log is unchanged for a human) and collects
  the ones `is_gpu_skip` matches.
- `is_gpu_skip(line: &str) -> bool` is pure and unit-tested: `line` must start with `SKIP`
  at column 0, and the rest must contain one of `gpu`, `vulkan`, `render`, `slangc`,
  `device` case-insensitively. Everything else — `SKIP nostd:`, `SKIP: ES_ACT_CHECKPOINT is
  unset`, `SKIP: no Python with torch`, an indented `SKIP` inside a message — is not a GPU
  skip.
- With `ES_REQUIRE_GPU=1` each collected line is printed as
  `FAIL ES_REQUIRE_GPU=1 but a GPU oracle did not run: …` and the step fails; unset, each is
  printed as `NOTE GPU oracle skipped (ES_REQUIRE_GPU unset): …` and the step passes. Same
  shape as `nostd --require`.
- `crates/es-gpu/src/buffer.rs`: `Buffer` gains `len: u64`, the byte count the caller asked
  for, beside `size` (what was allocated: `bytes.max(4)`, then the allocator's padding).
  `download` truncates to `len`, so it returns exactly the requested number of bytes.
- `crates/es-policy/tests/act_checkpoint.rs`: `is_missing_dependency(&str) -> bool` — true
  only for `cannot start \``, `ModuleNotFoundError:` and `ImportError:`. A reference failure
  that matches SKIPs; anything else `panic!`s, because an installed LeRobot that raised is a
  result, not a missing environment.
- `.github/workflows/ci.yml`: a comment on the PR job's `cargo xtask ci` step recording that
  this runner has no GPU, that `ES_REQUIRE_GPU` is therefore deliberately unset, and what
  setting it to `1` does. The `oracles` job is unchanged.

## oracle

```
cargo test -p xtask
cargo test -p es-gpu -- --nocapture
ES_REQUIRE_GPU=1 cargo xtask ci
```

`xtask`'s own tests cover `is_gpu_skip` on both sides (four GPU skip lines matched, six
non-GPU lines rejected) without needing a GPU. `download_returns_exactly_the_requested_length`
downloads three `f32` from both a `Storage` and a `Staging` buffer and asserts the length and
the contents.

## acceptance

- `cargo xtask ci` prints test output (it runs with `--nocapture`), so a `SKIP` line is
  visible in the log.
- On a machine with a GPU, `ES_REQUIRE_GPU=1 cargo xtask ci` exits 0 and reports no skipped
  GPU oracle; if any GPU oracle skips, it exits non-zero naming the line.
- Without `ES_REQUIRE_GPU`, a GPU-less machine still passes, with one `NOTE` per skip.
- `Buffer::download()` / `download_f32()` return exactly the requested byte / element count.
- A LeRobot reference that raises fails `a_real_lerobot_act_checkpoint_reproduces_its_actions`
  instead of skipping it; only a missing interpreter or a missing module skips.

## forbidden

- No change to what `cargo xtask ci` runs beyond `--nocapture` and the scanner — the step
  list, `context-budget`, `layering`, `nostd`, `check-spec-refs` and `verify-goldens` stay as
  they are.
- No new CI job, and no change to the `oracles` job's existing `tee` + `! grep -q '^SKIP'`.
- No weakening of any SKIP into a silent pass: the scanner only ever turns a skip into a
  failure, never a failure into a skip.
- No edits under `crates/es-render/**`, `crates/es-env/**` or the other crates the parallel
  M4 packets own; `es-policy` is touched only for the SKIP/FAIL split above.
