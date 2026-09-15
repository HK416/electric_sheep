# M5 V9 — the showcase render

Design note: `docs/design/visible-learning.md` **section 7.17**, and 7.1–7.3 (the renderer in
the loop), 7.8 (the V3 mosaic video) and 7.12–7.13 (the harness that passes the expert) for how
it got here. Renderer: `docs/design/renderer.md` sections 2, 3 and 5. Spec: §3.4, §3.5, §4.2,
§15.2, §15.3, §25.1. Read those before this file.

## the question this packet exists to answer

**Can anyone watch this demo?**

No. `es video mosaic` tiles the frames the *policy* reads — 96×96 `Rgb8` tiles from the one
overhead camera the Observation IR declares — and V3's 4×4 grid is a 384×392 mp4 of sixteen
thumbnails seen straight down. Every artifact the demo has produced is either a number, a hash
or that mosaic. The one sentence the owner wants to show people — "the arm picks the cube and
drops it in the bin" — is not in any of them, and cannot be, because the only camera in the
pipeline is the one the network was trained on and it is 96 pixels wide on purpose.

So V9 adds a second, human-facing render of a run that has already happened, from a camera that
exists in no scene, no `ObservationSpec` and no hash.

## the decisions, and what each rests on

### 1. Replay-render, not a second camera in the loop

Two candidates were on the table:

**(a) Replay-render.** Every run records its per-tick state trajectory; a new command re-poses
the scene from that file and renders it with an independent camera.
**(b) A second `ImageSpec` during the run.** `es eval run --showcase WxH` renders a second view
per tick into a sibling directory.

**(a), and the code makes it cheaper rather than more expensive.** `es-render` is already a
pure function of a triangle scene and a camera: `EnvRenderer::frame` is nothing but
`body_poses` → `TriScene::from_scene_with_poses` → `Renderer::render` → `read_tile`
(`crates/es-env/src/render.rs`). The only part of a running env it consumes is
`StateView::xpos` / `xquat` — 7 doubles per body. So "record what the renderer is handed" and
"record what the robot did" are the same file, and once it exists the render is a *replay*
with no physics backend, no policy, no Torch and no MuJoCo in the process at all.

What (b) cannot do, and (a) can:

* re-render a finished run at a different resolution, a different angle or a different render
  path, without re-running a 900-step episode through MuJoCo and a network;
* render **the expert**, which does not go through `es eval run` at all — it goes through
  `es loop collect --expert`, and gets the same file from the same three lines;
* be checked. A second camera in the loop has no reference to be compared against. A replay
  has one: render the trajectory at the run's **own** `ImageSpec`, through the run's **own**
  camera, and the bytes must be the frames the run wrote. That is this packet's oracle and it
  is only available under (a).

And the trajectory is worth having whatever the video is: an evaluation run records
`report.json`, `events.json` and pixels, and nothing that says where the arm and the cube were.

### 2. The file: poses beside `qpos`, and why both

`crates/es-env/src/traj.rs`, `.estraj`:

```text
magic  "ESTRAJ01"          8 bytes
nq, nv, nbody, ticks       4 x u32 little-endian
body ids                   nbody x 16 bytes (`StableId`)
rows                       ticks x (nq + nv + nbody*7) x f64 little-endian
```

`qpos ‖ qvel` is what "what the robot did" means for provenance and is the row the demo's own
`observation.state` already carries. The **body poses** are there because they are what the
renderer eats: re-deriving them from `qpos` at replay time would mean forward kinematics, which
means a physics backend in the replay process and a second implementation of `mj_forward` to
disagree with. Stored, the replay is *the same bytes the renderer was handed* and the oracle
below has no tolerance in it. For the demo scene that is 109 f64 a tick — under 800 KB an
episode, against the 24 MB of 96×96 frames the same episode already writes.

Two details that are load-bearing rather than incidental:

* `Trajectory::poses` builds `Pose` by **struct literal**, not `Pose::new`. What was recorded
  is already `Pose::new`'s output, and normalising a canonical unit quaternion a second time
  moves its last bit (`es_math::Quat::normalize` divides by the norm). Bit-for-bit is the whole
  claim of the file.
* `Trajectory::push` records the tick **where the frame is captured**, not where the control
  tick begins. Under `observation_delay` a dropped observation renders nothing, so trajectory
  index and frame index stay the same number in every suite — which is what makes the oracle
  comparable frame by frame.

Bounded parsing (§25.1): the byte length is implied by the header and checked in full before
anything is allocated. A truncated file, an over-long file, a zero-stride header and an
arithmetic overflow in the length are each refused by name.

### 3. Always recorded, never a flag to remember

`es eval run` writes `<out>/traj/<suite>-<NN>.estraj` and `es loop collect` writes
`<root>/traj/ep-<NNN>.estraj` unconditionally; `--traj <dir>` only moves them. The library
default is `None` (`RunConfig::traj_dir`, `CollectSpec::traj_dir`), so no existing caller or
test changes behaviour.

Under `--jobs N` the shards share one trajectory directory exactly as they share the frame
directory: cell names are globally unique and shards own disjoint cells, so there is nothing to
merge. The collector's files sit **beside** the `LeRobot` dataset, never inside its own files:
`dataset_content_hash` hashes the parquet files the episode metadata names and nothing walks
the root, so no dataset hash moves.

### 4. The camera is on the command line, because the scene cannot grow one

`es video showcase --eye X,Y,Z --look-at X,Y,Z [--fov D]`, world up `+Z`, built into the
`OpenCV` frame of §3.1 by `es_env::render::look_at`. **Not** a `<camera>` added to
`tests/fixtures/mjcf/so101_pick_place.xml`: `scene_hash` feeds `task_hash` feeds
`observation_hash` feeds every trained bundle, so adding a camera to the demo scene would
invalidate the very checkpoints the video exists to show. A camera that changes a hash is not
an independent camera.

`--camera NAME` renders a camera the scene *does* declare instead. That is not a convenience:
with the run's own `--width`/`--height` it is the replay of the run's own observation, which is
how a replay is checked against the run it replays (the oracle below).

`INV-14` is not in play. Nothing is resized: the showcase `ImageSpec` is computed from `--fov`
at `--width`×`--height` like every other camera's, and `Resize`/`Crop` remain Observation IR
nodes. No IR is read, written or hashed by this command; the mp4 was never in the hash chain
(§5.3) and the frames of a showcase render are not either — they are a re-rendering of a run,
not a record of one.

### 5. `Rs` only, and the path tracer is not cut for want of a device

The `Pt` path **runs headless on the oracle server** — `cargo test --release -p es-render` there
passes `gpu_path_tracer_matches_the_cpu_reference_at_1spp`,
`gpu_pt_and_rs_agree_on_geometry` and `gpu_restir_and_svgf_match_the_cpu_within_tolerance` on
the RTX 4090 with no display. `es video showcase` is still `Rs` only, for a reason that has
nothing to do with the device: `PT_CHANNELS` is `PtRadiance` (linear `f32`) plus the three
geometry channels, and `Rgb8` is not among them (`crates/es-render/src/lib.rs`). Turning linear
radiance into an 8-bit frame is a tone-mapping decision, and inventing one here would put a
look-decision in a CLI flag with no oracle behind it. A `--pt` flag is a later packet's, with an
exposure and a curve it can defend; `EnvRendererCfg::path` already carries the variant.

### 6. What did **not** change

No new trait (`INV-17`): `Trajectory` is a struct and the render path is the one `es-render`
already has. No Safety Plane code, no `es-safety` dependency, no envelope number. No IR type,
no schema, no hash. No golden file: the showcase render produces no golden and edits none —
`es_render::cpu`, the generator of every golden in `tests/golden/render/`, is used by the
oracle as a *reference*, not regenerated.

## oracle

Runnable, in this order.

1. **The format round-trips and a truncated file is refused** (§25.1) —
   `cargo test -p es-env --lib traj`. Property tests over `(nq, nv, nbody, ticks)`, plus
   `a_truncated_file_is_refused` (six cut points and one trailing byte) and
   `a_hostile_header_allocates_nothing_it_cannot_read`. `a_replayed_tick_is_the_pose_map_the_
   renderer_was_handed` pins `Trajectory::poses(t) == body_poses(model, state, 0)` exactly.
   No GPU, no backend, runs in CI.

2. **A replay is the run, at the run's own `ImageSpec`** —
   `cargo test --release -p es --features render --test cli --
   a_showcase_replay_reproduces_the_frames_the_policy_saw`. The scripted expert runs through
   `es_eval::Evaluation` — the real runner, the real plane, the real Task IR — with the CPU
   reference rasterizer (`es_render::cpu::rasterize`, the generator of every render golden) as
   its frame source at the demo's own 96×96 `ImageSpec`, and the run records its trajectory.
   Then every tick is re-rendered **from the file** and compared to the frame the run wrote,
   byte for byte. A **server oracle**: it needs `mujoco` and prints a reason and skips without
   it. The CPU rasterizer rather than a Vulkan device is deliberate — the claim is about the
   states, and the renderer is a pure function of them on either path.

3. **The same claim on a real run, through the real command, on the GPU** —
   `ES_SHOWCASE_RUN=<a finished run> cargo test --release -p es --features render --test cli --
   --ignored showcase_replay_of_a_real_run_is_bit_identical`. Runs
   `es video showcase --camera overhead --width 96 --height 96 --cell nominal-00` over a
   finished `es eval run --frames` output and compares every frame to the recorded one.

4. **The video** — `es video showcase` over a run, then `python/es/encode_video.py` (or ffmpeg
   over the same raw frames, as V3 did) produces an mp4 whose frame count equals the recorded
   tick count.

5. `cargo xtask ci`.

## acceptance

* `.estraj` round-trips, refuses a truncated file, and `Trajectory::poses` equals
  `body_poses` bit for bit (oracle 1).
* Every `es eval run` and `es loop collect` writes one trajectory per episode with no flag, and
  no dataset hash, report or lock moves because of it.
* A replay at the run's own `ImageSpec` reproduces the recorded observation frames byte for
  byte, on the CPU reference path (oracle 2) and on the GPU path through the CLI (oracle 3).
* `es video showcase` writes `NNNNNN.bin` + one `layout.json`, one frame per recorded tick, and
  the encoder turns them into an H.264 mp4 with that many frames.
* An `es` built without the `render` feature refuses `es video showcase` by name and links no
  Vulkan.
* `cargo xtask ci` passes.

## forbidden

* **The scene file.** No camera, geom or body may be added to
  `tests/fixtures/mjcf/so101_pick_place.xml`, and no document under
  `tests/fixtures/visible-learning/` may be regenerated: every hash downstream of them names a
  trained bundle this packet has to be able to replay.
* **The Safety Plane, and `es-safety`.** Not touched, not depended on, not widened.
* **Any IR.** No new node, field, schema version or hash term. The showcase camera is a command
  line argument and stays one.
* **Goldens.** `tests/golden/render/*` is read-only; this packet adds none and regenerates
  none.
* **Retraining, and any evaluation table.** V9 renders runs; it does not measure policies. The
  four-episode evaluation document it makes for the video is its own file with its own
  `evaluation_hash`, and is not the pinned 16-episode measurement.
* `es video mosaic`, which keeps doing exactly what it did.
