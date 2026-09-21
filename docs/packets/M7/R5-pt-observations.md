# M7 R5 — path-traced observations: a sensor declares its render path, and a policy is trained on it

Spec: §15.3 ("PT for photorealistic datasets and goldens; RGB between RS and PT is an SSIM
threshold"), §6 / §7.4 (the Task IR *declares* `ObservationSpec`: named channels with a source
and a type; the Observation IR owns preprocessing), §5.3 (nothing outside the hash chain),
§28.10 rule 1 (committed pixels never move; a moved byte is a document change), §3.4 (PT is
deterministic: counter RNG, fixed sample order), §18.3. Owner question 2026-09-21: *is there no
PT-based training?* — there is not, because no document can ask for it: the three places that
build an `EnvRendererCfg` (`es loop collect --frames`, `es eval run --frames`, `es video
showcase`) hard-code `RenderPath::Rs`, and R3 deliberately kept the PT knobs out of
`EnvRendererCfg`. Design notes: `docs/design/renderer.md` section 12, `docs/design/ir-types.md`
(the Task IR's sensor source). Depends on **R3** (NEE, the tone map → `Rgb8`), **R4**
(optional accumulation is *not* used for observations — every frame is independent), **T6**
(`observation-augmented.toml` shows how a second document is added beside the committed one),
**U** (the recipe shape for a re-measurement).

## the question

The path tracer can emit `Rgb8` since R3 and costs ~3 ms for a 96×96 frame at 64 spp on the
oracle server. **Can a Task IR sensor say "render me with the path tracer", so that collection,
evaluation and the showcase all render that sensor the same way, with the committed documents'
hashes and pixels untouched — and does a policy trained on path-traced observations pass the
same harness it was collected under?**

## spec

* **The declaration lives where the renderer config is built from: the Task IR's sensor.**
  `es_ir::task::ObsSource::Sensor { id, format }` gains `render: SensorRender` with
  `#[serde(default)]`:
  ```
  SensorRender { path: Rs | Pt { spp: u32, bounces: u32 }, exposure: f32 (1.0), tonemap: Reinhard | Aces }
  ```
  **Hash rule:** `canonical` writes the `render` block **only when it is not the default** — an
  absent or default `render` is byte-for-byte today's canonical form, so every committed
  `task_hash` (and everything hashed over it) is unmoved; a test asserts the committed
  `task.toml`'s hash against its recorded value. A `Pt` sensor moves `task_hash`, and therefore
  needs its own Evaluation IR document (§13.3: a new comparison) — the packet adds
  `tests/fixtures/visible-learning/task-pt.toml` (the committed `task.toml` with
  `render = { path = "pt", spp = 64, bounces = 3 }` on the camera channel) and
  `evaluation-pt.toml` (the committed `evaluation.toml` naming `task-pt`'s hash), and records both
  hashes in the note. The Observation IR is unchanged: what the sensor *is* (resolution, colour
  space, intrinsics) is `ImageSpec`'s; how the simulation *produces* it is the Task IR's, which is
  why this is not an `ImageSpec` field. Spec text: one paragraph added to §6's `ObservationSpec`
  (Korean, canonical) and its English twin — **the packet's only `ARCHITECTURE` edit, both files
  in one commit**, wording pinned below.
* **One function builds the renderer config from the sensor.** `es_env::render::sensor_cfg(camera,
  spec: &ImageSpec, render: &SensorRender, frames_dir) -> EnvRendererCfg` replaces the three
  hand-built `EnvRendererCfg::rgb(...)` calls (`eval.rs::renderer_cfg`, `loop.rs`, `showcase.rs`
  for the scene camera), mapping `Pt { spp, bounces }` to `RenderPath::Pt { spp, bounces, nee:
  true, restir: false, svgf: false }` with the tone map and exposure; `Rs` maps to exactly today's
  config (bitwise: the frames fixture and every RS golden pin it). `EnvRendererCfg` therefore gains
  `exposure`/`tonemap` **as pass-through fields the sensor declared**, not as CLI knobs — R3's
  refusal to give the observation path knobs stands: the *document* decides. PT frames are
  independent per tick (no accumulation; `RenderConfig.seed` fixed, the sample keys already vary
  per pixel), so a frame is a pure function of the pose — the collector/evaluator parity oracle
  (T7) holds for PT exactly as for RS.
* **Cost is measured, not assumed.** `es loop collect --frames` on `task-pt.toml`, 8 episodes, on
  the RTX 3060 and the 4090: ms/frame of the observation render beside RS's (`Target / Status:
  unverified` until measured). The `es video showcase` of a PT-collected run uses the sensor's
  path for the *scene camera* (`--camera`) and keeps `--path` for the free camera.
* **The re-measurement (server).** U3's configuration on the PT documents: collect 200
  demonstrations with PT frames (`task-pt.toml`), bake `--for-training`, train 20,000 steps at
  row-D settings with `learning-pretrained.toml` + `observation-augmented.toml`, evaluate on
  `evaluation-pt.toml`'s 16 held-out seeds; report `success_rate` beside U3's 0.5625 as
  **row U4** of `visible-learning.md` 7.31 (a new `evaluation_hash`, said out loud). Also the
  SSIM (R3's `ssim`) between the RS and the PT observation of the same tick, on 32 frames, as the
  first §15.3 number at observation resolution.
* **Spec paragraph (pinned).** Korean (`docs/ARCHITECTURE.ko.md`, §6 `ObservationSpec`):
  > **센서의 렌더 경로.** `Sensor` 소스는 `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }`로 시뮬레이션이 그 센서를 어떻게 만드는지 선언한다(패킷 M7/R5). 기본값은 `rs`이며 **부재 = 기본값 = 오늘의 정규형**이라 커밋된 `task_hash`는 움직이지 않는다; `pt`는 `task_hash`를 움직이므로 새 문서다(§13.3). 관측 IR은 이것을 모른다 — 센서가 *무엇*인지는 `ImageSpec`이, 시뮬레이션이 그것을 *어떻게* 만드는지는 Task IR이 말한다.
  English (`docs/ARCHITECTURE.md`):
  > **A sensor's render path.** A `Sensor` source declares how the simulation produces it with `render = { path = "rs" | "pt", spp, bounces, exposure, tonemap }` (packet M7/R5). The default is `rs`, and **absent = default = today's canonical form**, so no committed `task_hash` moves; `pt` moves `task_hash` and is therefore a new document (§13.3). The Observation IR does not know about it — what the sensor *is* belongs to `ImageSpec`, how the simulation *makes* it belongs to the Task IR.

## context

The globs `cargo xtask check-scope` reads, then the same scope in prose:

```
crates/es-ir/src/task.rs
crates/es-ir/src/serial.rs
crates/es-ir/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/**
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/showcase.rs
crates/es/tests/cli.rs
crates/es/tests/video.rs
tests/fixtures/visible-learning/task-pt.toml
tests/fixtures/visible-learning/evaluation-pt.toml
tests/fixtures/visible-learning/training-u4.toml
docs/ARCHITECTURE.ko.md
docs/ARCHITECTURE.md
docs/design/renderer.md
docs/design/renderer.ko.md
docs/design/ir-types.md
docs/design/ir-types.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M7/R5-pt-observations.md
docs/packets/M7/R5-pt-observations.ko.md
```

`es-ir/src/task.rs` (`SensorRender`, the conditional canonical form, the default), `serial.rs`
**only** if the TOML shape needs a serde attribute, `es-ir/tests` (the hash-stability test),
`es-env/src/render.rs` (`sensor_cfg`, the two pass-through fields), `es-env/tests`, the three
CLI call sites, `cli.rs`/`video.rs` (PT collect/eval/showcase tests), the three new fixtures,
the spec paragraph (both files, one commit), the three design notes, this packet.

## oracle

1. `cargo test -p es-ir committed_task_hash_is_unmoved_by_sensor_render` — the committed
   `task.toml` hashes to its recorded `eb6efefa…`; the same document with an explicit default
   `render` hashes identically; `task-pt.toml` differs.
2. `cargo test -p es-env sensor_cfg_rs_is_todays_config` — `sensor_cfg` with the default render
   equals `EnvRendererCfg::rgb(...)` field for field, and its `RenderConfig` renders the frames
   fixture's first frame bitwise.
3. `cargo test -p es --test cli collect_renders_the_sensor_with_the_path_tracer` — 1 expert
   episode on `task-pt.toml` with `--frames` (GPU): the frames are `Rgb8`, non-black, and differ
   from the RS render of the same tick; two runs are bitwise identical (`SKIP` without a device or
   `ES_PYTHON`).
4. `cargo test -p es --test cli eval_refuses_a_pt_policy_on_the_rs_document` — the hash mismatch
   is refused by name (task hash), which is §13.3 doing its job.
5. `cargo test -p es --test video showcase_scene_camera_follows_the_sensor_path`.
6. `cargo xtask verify-goldens` 0 changed; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R5-pt-observations.md`.

## acceptance

Oracles 1–6 (3 on the RTX 3060 and the server). The cost table, the SSIM number and row U4 in the
notes; the PT-collected demonstration frames of one episode saved as a contact-sheet PNG under
the worktree's `target/plan-u/r5/` beside the RS frames of the same ticks. `training-u4.toml`
committed (server paths, as U0–U3's are).

## forbidden

Moving any committed hash or pixel (oracle 1, 2, 6); PT knobs on the CLI (`--path` stays the
free camera's only); accumulation (R4) on the observation path; changing the Observation IR or
`ImageSpec`; `crates/es-render/**` (R3 has what this needs — if it does not, stop and say what);
any `ARCHITECTURE` edit beyond the pinned paragraph. INV-14: intrinsics untouched. INV-17: no new
trait.
