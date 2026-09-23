# M11 R2 — the collector and the evaluator render a randomized task with its draws

Spec: §28.14 rule 4 (the same draw renders the same frame bitwise wherever it is rendered) and
rule 1, §13.1, §10.4. Found by X5 (2026-09-23): `Rollout` applies each env's `RenderOverrides`,
but `es loop collect --frames` and `es eval run --frames` still render the **undrawn** scene for a
task with render targets. A dataset or a held-out evaluation of a DR task would therefore see
pixels the trainer never saw. Also from X5: two `Cross` builtins in `common.slang` (Möller–Trumbore
and the face normal) may fuse on some drivers, as the camera one did. Type B.

## the question

**Do the collector and the evaluator render each episode with that episode's draws — the same
frame `Rollout` renders at the same `(seed, env, episode, tick)`, bitwise — while every task that
declares no render target renders exactly today's bytes? And do the two remaining `Cross` calls
stay exact under drawn poses?**

## spec

* The collector's frame source (`crates/es/src/cmd/loop.rs`) and the evaluator's
  (`es_eval::runner` frame source, `crates/es/src/cmd/eval.rs`) call `EnvRenderer::frame_with` with
  `env.render_overrides(env)` (X5's API), and write the drawn intrinsics into the frame sidecar as
  `Rollout` does.
* `common.slang`: the Möller–Trumbore and face-normal `cross` become the written-out `es_cross`,
  but only if a test shows the builtin is off by at least 1 ULP under a drawn pose on either GPU.
  If it is exact, record the evidence and leave the code as it is.

## context

```
crates/es/src/cmd/loop.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-env/tests/render_loop.rs
crates/es-render/slang/common.slang
crates/es-render/tests/render.rs
tests/fixtures/rl/**
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/P-M11-R2-collect-eval-draws.md
docs/packets/M11/P-M11-R2-collect-eval-draws.ko.md
```

## oracle

1. `cargo test -p es --test cli --features render dr_collect_eval_frames_match_rollout` (GPU): a
   vision reach task with light, rgba, camera pose and fov targets. The frames at `(episode, tick)`
   from `loop collect --frames`, from `eval run --frames` and from `Rollout` are bitwise equal, and
   the sidecar intrinsics equal the episode's recorded `ImageSpec`.
2. Every existing frame golden and trajectory unmoved (`so101_frame0` and the demo's `.estraj`
   and `events.json` pins).
3. `cargo test -p es-render --test render dr_cross_is_exact` on both GPUs: the result either
   justifies changing `common.slang` or shows it is not needed.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–4; `renderer.md` (+ko) gets one paragraph on where the draws are applied.

## forbidden

Changing a frame of a task without render targets; a second draw implementation; batching (X3b).
