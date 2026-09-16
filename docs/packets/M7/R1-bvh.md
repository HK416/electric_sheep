# M7 R1 — the 80 ms frame: cached tessellation, persistent buffers, a software BVH

Spec: §15.3 (RS is the vision default), §15.4 (acceleration structures — one TLAS, shared BLAS;
this packet is the software version `es-gpu`'s compute-only device allows), §3.4 (determinism:
no atomics, fixed traversal order, no workgroup-count dependence), §12.4 (`camera_frames_per_sec`,
`pixels_per_sec`; never a single `step/s`), §28.10 rule 1 (a committed document's pixels do not
move). Design note to extend: `docs/design/renderer.md` (+ `.ko.md`) — a new section 8 and edits
to sections 0, 2.1 and 3 where they state the flat-scan ceiling. Predecessors: M4 W8 (the
renderer), M5 V0b/V9 (the env-loop and showcase call sites).

## the question

`es video showcase` measured **80 ms/frame** at 1280×720 for 2,978 triangles on an RTX 4090
(`visible-learning.md` 7.17 item 4), and every 96×96 observation frame inside an evaluation
pays the same per-frame path. Two known causes, unquantified: the whole scene is re-tessellated
(f64, `approx::sin/cos` per vertex) and re-uploaded into a **freshly allocated** buffer every
frame, and the shader scans all triangles per pixel — `O(pixels × triangles)`. **Where does the
time go, and how far does it fall when each cause is removed without moving one output bit?**

## spec

**Step 0 — measure first.** Add an `#[ignore]`d test (or a `--profile` flag to
`es video showcase`, whichever is smaller) that times the four phases of one frame — tessellate,
upload, dispatch+wait, readback — on the demo scene at 1280×720 and at 96×96, over 100 frames,
median and p95. Record the table in the design note **before** changing anything. This is the
baseline the acceptance compares against.

**Step 1 — cached tessellation.** `TriScene::from_scene_with_poses` recomputes every geom's local
vertices per call. Cache the per-geom local tessellation (the `Vec<[Vec3; 3]>` `tessellate(geom)`
returns) keyed by geom id inside a new `SceneCache` (or on `Renderer`), and apply the pose per
frame exactly as today — the same `pose.transform_point` in f64, the same `to_f32`, the same
`face_normal`, the same order — so the world-space triangles are **bit-identical** to the
uncached path. The oracle checks that.

**Step 2 — persistent buffers.** `Renderer::upload_tris` and `render` allocate new `Buffer`s each
call (`Buffer::new` is a Vulkan allocation). Keep the triangle, params, output and BVH buffers
across frames and grow them only when a frame needs more (never shrink). Readback stays as it is.

**Step 3 — a BVH, single level, rebuilt per frame on the CPU.** Over the world-space triangles of
the frame: a binary BVH, median split on the centroid along the longest axis, leaves of ≤ 4
triangles, nodes in a flat array (`[min.xyz, max.xyz, left/first, count]`, f32/u32), built by a
deterministic routine (stable ordering, no `HashMap`, no threads). Upload it beside the
triangles. In `common.slang` (`es_nearest`, and the any-hit/shadow scan `restir.slang` uses) and in
`es_render::cpu::nearest_hit`, replace the flat scan with a stack-based traversal (fixed stack of
64 entries; the build bounds the depth and the test asserts it). **The hit rule is unchanged**:
nearest `t` with a strict `<`, ties broken by the lower *global triangle index* — so the
traversal visits nodes in a fixed order and compares `(t, index)`, and the winner is the flat
scan's winner. Ray–triangle intersection code is untouched (same `Möller–Trumbore`, same
expression order), which is what keeps depth and normal at 0 ULP.

Two-level TLAS/BLAS (a BLAS per geom built once, a TLAS over body instances per frame) is the
§15.4 design and the upgrade path; it is **not** this packet, because per-frame f64 CPU posing is
what keeps the vertices bit-identical to today, and a 3,000-triangle rebuild is not where 80 ms
goes. Leave a `ponytail:` comment naming that upgrade at the build site.

**Step 4 — measure again**, same table, and record it beside the baseline. Report §12.4's
`camera_frames_per_sec` and `pixels_per_sec` for the two sizes; the target `< 5 ms/frame` at
1280×720 stays `Target / Status: unverified` until the server number exists.

## context (allowed scope)

`crates/es-render/src/{scene.rs,renderer.rs,cpu.rs,lib.rs}`, `crates/es-render/src/bvh.rs` (new),
`crates/es-render/slang/{common.slang,restir.slang,pt.slang,raster.slang}` (traversal only),
`crates/es-render/tests/render.rs` (new tests; existing tests and the golden generator untouched),
`crates/es-render/benches/` or the `#[ignore]`d timing test, `crates/es-env/src/render.rs` (use
the cache; no behaviour change), `crates/es/src/cmd/showcase.rs` (use the cache; the per-frame
loop otherwise unchanged), `docs/design/renderer*.md`, `docs/packets/M7/R1-bvh*.md`.

## oracle

1. `cargo test -p es-render cpu_reference_reproduces_the_goldens_bit_for_bit` and every existing
   GPU test — **unchanged files, unchanged results**. `cargo xtask verify-goldens` reports 0
   changed. This is the packet's first and last oracle.
2. `cargo test -p es-render cached_tessellation_is_bit_identical` — `from_scene_with_poses`
   through the cache and through the uncached path give equal `TriScene`s (`PartialEq`, bitwise
   on the `f32`s) for Cornell and for the SO-101 scene at three recorded ticks of a fixture
   `.estraj` (reuse E2's fixture trajectory if it has landed; else the scene's home pose and two
   hand-set joint vectors through `TriScene::from_scene_with_poses` with poses built by the test).
3. `cargo test -p es-render bvh_traversal_is_the_flat_scan` — for 10,000 rays (deterministic
   counter-based directions from `es_render::rng`) over both scenes, `nearest_hit` via the BVH
   returns the same `Option<Hit>` (bitwise `t`, same triangle index) as the retained flat scan
   (keep the old scan as `nearest_hit_flat`, test-only or `pub(crate)`); the any-hit variant
   agrees on the boolean; the maximum traversal depth over all rays is ≤ 64 and printed.
4. `cargo test -p es-render gpu_bvh_matches_the_cpu_bvh` (GPU; `SKIP` without a device) — `Rs`
   and `Pt` 1 spp on the SO-101 scene at 256×256 through the GPU equal the CPU reference at the
   existing tolerances (`Rgb8`/`Seg` bitwise, `Depth`/`Normal` ≤ 1 ULP, printed), and
   `gpu_renders_are_bit_identical_across_runs` still holds with the BVH.
5. `cargo test -p es-render --release -- --ignored frame_profile` prints the four-phase table for
   both sizes; run before and after, both tables in the design note.
6. `cargo xtask ci` green; `cargo xtask check-scope docs/packets/M7/R1-bvh.md` clean.

## acceptance

Oracles 1–6 pass locally (RTX 3060) and 4–5 also on the oracle server (RTX 4090, headless).
`es video showcase` on `~/artifacts/plan-v/v19b`'s `nominal-00` at 1280×720 reproduces V9's
bit-identity oracle (`showcase_replay_of_a_real_run_is_bit_identical`, 96×96, `--ignored`) and
its ms/frame is recorded before and after. The design note section 8 holds both profile tables,
the BVH layout, the traversal order argument for bit-identity, and the two-level upgrade path.

## forbidden

Changing any golden or fixture; changing `Möller–Trumbore` or any shading expression; a
tessellation-count change; atomics, shared memory, subgroup ops or any dependence on workgroup
count (§3.4); a Vulkan extension (`es-gpu` is compute-only and stays so; `crates/es-gpu/**` is
out of scope); `HashMap`; threads in the build; `docs/ARCHITECTURE*.md`; the `RenderConfig`
defaults (R2 owns the look). INV-17: no new trait.
