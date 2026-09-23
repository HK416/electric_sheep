# M11 X3b — N envs in one dispatch

Spec: §28.14 wave 2, §15.2 (tile atlas), §3.4, §12.4 (the nine metrics, never a single `step/s`).
Design note: `renderer.md` §1 (tile atlas + channel contract) and 12.4 (cost). Depends on X3. Type B.

## the question

The atlas lays out several **cameras over one scene** (`renderer.md` §1); `Rollout` after X3 renders
each env with its own `EnvRenderer`, one dispatch per env, at 57.5 ms per 96×96 64-spp `Pt` frame on
the RTX 4090 — about 65 h for one reach PPO run. **Can one dispatch render N envs — the same
geometry, per-env body transforms and per-env render overrides — into N tiles, bit-identical to N
single renders, at a per-env cost PT training can afford?**

## spec

* A scene is uploaded once (triangles, BVH per rigid body in local frame, as today's
  tessellation cache already keeps local-frame triangles); per-env transforms and per-env
  overrides (light, colours, camera — X5's) are a small per-tile buffer. The BVH strategy (a
  two-level BVH with per-env instance transforms, or a per-env TLAS rebuild over shared BLASes) is
  the agent's to choose and to justify with numbers; the CPU reference renders the same N tiles by
  the same traversal and the same sample keys.
* Per tile, the sample seed is that env's `frame_seed(stream, base, tick)` — a tile is exactly
  what a single render of that env would produce.
* `Rollout` uses the batched path when the renderer supports it and the env count > 1; the
  single-env path stays and stays the oracle.

## context

```
crates/es-render/src/**
crates/es-render/slang/**
crates/es-render/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/render_loop.rs
crates/es-py/src/rollout.rs
tests/golden/render/batch_*
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M11/X3b-batched-render.md
docs/packets/M11/X3b-batched-render.ko.md
```

## oracle

1. `cargo test -p es-render --test render batched_tiles_equal_single_renders` (GPU): N ∈ {1, 4, 16}
   envs of the SO-101 scene at distinct poses, `Rs` and `Pt` (1 and 16 spp, SVGF on/off): every tile
   == the single render of that env bitwise; CPU reference == GPU as the existing parity rules say
   (0 ULP where they say 0).
2. Every existing golden byte-identical; new `batch_*` goldens from the CPU reference only.
3. `pt_batched_cost` (`--ignored`, both GPUs): per-env ms/frame at N = 1, 4, 16, 64 for `Pt` 4/16/64
   spp ± SVGF, whole-frame (upload, dispatch, readback), beside the X3 single-env numbers.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

Oracles 1–4; `renderer.md` gains section 13 (+ko) with the cost table and the BVH choice's reason.

## forbidden

Changing any existing frame's bytes; non-deterministic reductions; a vendor denoiser; a new trait.
