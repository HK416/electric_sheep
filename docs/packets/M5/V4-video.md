# M5 V4 — the 4x4 grid video: mosaic, overlay, encode

Design note: `docs/design/visible-learning.md` sections 7.2, 8 and 9; read section 2.9 first — the oracle
server has **no `ffmpeg` binary**, and only the `mp4v` codec encodes there. Independent of V0 and V0b:
this packet is written as a *consumer* of a frame directory and an `events.json`, and its oracle is a
checked-in synthetic fixture, so it can be built in parallel with V0 exactly as plan V asked. V3 produces
its real input.

## context

```
crates/es/src/cmd/video.rs
crates/es/src/cmd/mod.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
crates/es/tests/video.rs
python/es/encode_video.py
python/es/README.md
python/es/README.ko.md
tests/fixtures/visible-learning/frames/**
tests/fixtures/visible-learning/events.json
tests/golden/video/mosaic_4x4_frame0.bin
tests/golden/video/mosaic_4x4_frame0.json
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V4-video.md
docs/packets/M5/V4-video.ko.md
```

Notes: `video.rs` is new — `es video mosaic`, which is pure Rust and needs no GPU, no Vulkan and no
Python. `python/es/encode_video.py` is the encoder and the only Python; it is not counted by
`cargo xtask context-budget` (`xtask/src/context_budget.rs:148-152`). The fixture is 16 tiny synthetic
frame directories plus an `events.json` covering every `ActionSource`, generated once by the test's
`--ignored` generator and then read-only.

## spec

- §1.4: the mosaic is judged against a golden produced from a fixed fixture; the encode is judged by a
  file that exists, has the right frame count and decodes.
- §2.4: the mosaic and the overlay are Rust and run without Python. Only the container encode is Python,
  and it is a presentation step, not a runtime path.
- §3.4: the mosaic is integer work on bytes — no float blending, no `HashMap` iteration order, no global
  RNG. The same frames and the same `events.json` give the same mosaic, on any machine.
- §5.3, §3.5: the frames are the artifact the hash chain covers; the mp4 is **not** in the chain, because
  the encoder is a host library outside it. Design note section 9 is the table, and `es video` prints that
  sentence rather than implying otherwise.
- §12.4: the overlay shows a success rate and an episode index. It shows no rate per second.
- §25.1: `events.json` and the frame sidecars are read as untrusted input — a frame whose `layout.json`
  disagrees with its byte length, a `frame` index out of range, or a cell count that is not 16 is a named
  error before any allocation.
- §1.5: `es` is at 2,946 code lines; this packet is budgeted under ~350 in `src/`.

## oracle

```
cargo fmt --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es --test video
cargo test -p es --test cli video_
cargo xtask context-budget
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

Reference — the encode, which needs `cv2`:

```
ES_CV2_PYTHON=$HOME/venvs/es-lerobot/bin/python cargo test -p es --test video -- --ignored --nocapture
```

**Measured 2026-09-14 on the oracle server.** `command -v ffmpeg` is empty — there is no `ffmpeg`,
`ffprobe`, `gst-launch-1.0` or `convert` binary — while the FFmpeg shared libraries
(`libavcodec62` .. `libavutil60`, `7:8.0.1-3ubuntu2`) are installed. `~/venvs/es-lerobot` has
`opencv-python-headless 4.13.0.92`, built with `FFMPEG: YES`. A direct test there:
`cv2.VideoWriter` with fourcc `mp4v` opens and writes (10 frames of 64x64 -> 1450 bytes), while `avc1`
and `H264` both fail to open — the only H.264 encoder linked is `h264_v4l2m2m`, which reports
`Could not find a valid device`. **So the codec is `mp4v` (MPEG-4 Part 2) in an `.mp4` container.** Plan V
installs nothing; H.264 is design note open question 5.

`crates/es/tests/video.rs`:

- `mosaic_matches_the_golden` — `es video mosaic` over the checked-in fixture is byte-equal to
  `tests/golden/video/mosaic_4x4_frame0.bin`, with a matching sidecar. This is the whole determinism
  claim of this packet and it runs on the PR tier with no device and no Python.
- `two_runs_are_byte_identical` — the same inputs twice.
- `a_clamped_record_draws_a_red_border` — a cell whose `events.json` record is `Clamped` or `Fallback` has
  the border pixels set to the red constant and the interior untouched; a `Policy` cell has no border.
  The test compares pixel spans, not a visual impression.
- `the_overlay_digits_are_a_fixed_bitmap` — the success-rate and episode text is drawn from an embedded
  bitmap font table, so no system font, no locale and no float formatting can move a pixel.
- `a_short_cell_holds_its_last_frame` — cells with unequal frame counts are padded by repeating the last
  frame, and the mosaic's frame count is the maximum. Truncating to the minimum would hide exactly the
  cells that failed early.
- `mismatched_layouts_are_rejected` — two cells with different `layout.json` sizes is a named error,
  never a resize (§26.1: what is not validated is not executed; INV-14's spirit — this code never
  resamples an image).
- `malformed_inputs_are_rejected_before_allocation` — a byte length disagreeing with the sidecar, a
  `frame` index past the end, a cell count that is not 16, and a `layout.json` whose `shape` product
  overflows: four named errors, no panic. Proptest: arbitrary `events.json` values never panic.
- `encode_produces_a_playable_file` (`--ignored`) — `encode_video.py` over the mosaic frames writes an
  `.mp4` whose frame count, width and height read back through `cv2.VideoCapture` equal what was written.
  No `cv2` -> `SKIP encode_video: <why>`; ran -> `RAN encode_video`.

`crates/es/tests/cli.rs`: `video_usage_errors_exit_2`, `video_mosaic_missing_events_is_exit_1`.

## acceptance

```rust
// crates/es/src/cmd/video.rs
// es video mosaic --frames <dir> --events <events.json> --report <report.json>
//                 --grid 4x4 --out <dir> [--label-height N]
//   reads <frames>/<cell>/NNNNNN.bin + layout.json for every cell,
//   writes <out>/NNNNNN.bin + layout.json: one mosaic frame per timestep.
pub fn dispatch(args: &[String]) -> i32;   // 0 ok, 1 runtime, 2 usage, 3 SKIPPED
```

```
python/es/encode_video.py --frames <mosaic dir> --out demo.mp4 --fps N [--codec mp4v]
```

- The mosaic is `rows x cols` cells in cell-name order (a `BTreeMap`, so the order is the file order, not
  a hash order), each cell's tile copied verbatim. No scaling, no filtering, no colour conversion.
- A cell whose `StepEvent` for that frame is `Clamped` or `Fallback` gets a red border of a fixed width
  drawn over its outermost pixels. `Policy` and `Human` get none.
- A label strip under the grid carries the episode index and the running success rate from
  `report.json`'s `MetricSpec::SuccessRate` (`crates/es-eval/src/metrics.rs:32`, `:97-100`), drawn from an
  embedded bitmap font — no `fontdb`, no system font, no shaping.
- Output frames use the same raw `.bin` + `.json` sidecar layout as `tests/golden/render/*`, so the mosaic
  is verifiable by the same means as a rendered frame (design note section 7.2).
- `encode_video.py` uses `cv2.VideoWriter` with fourcc `mp4v`; it prints one JSON line
  `{"frames": n, "width": w, "height": h, "codec": "mp4v"}` and nothing else on stdout. It reads the
  sidecars; it does not guess a shape.
- `es video` prints, once, that the mp4 is outside the hash chain and the frames are the evidence.
- No new trait, no new external crate (no `png`, no `image`, no `fontdb`, no codec crate), no GPU, no
  Vulkan, ≤ ~350 source lines.

## forbidden

- Adding a PNG, JPEG, font or video codec crate to the workspace, or an `ffmpeg` binary dependency. The
  frame format is the existing raw `.bin` + sidecar; the encoder is `cv2` on the host (design note open
  questions 5 and 7).
- `crates/es-render`, `crates/es-gpu` — this packet never touches a device; the mosaic is a byte copy.
- `crates/es-eval`, `crates/es-env`, `crates/es-data`, `crates/es-policy`, `crates/es-ir` — V0b, V1, V2,
  V3 own those.
- Scaling, cropping or resampling a cell to make a grid fit: mismatched layouts are an error (INV-14's
  rule that no image size changes without an intrinsics transform, applied here by never changing one).
- Truncating the mosaic to the shortest cell.
- Drawing anything derived from the policy rather than from `events.json`: the red border is the Safety
  Plane's decision, and reading it from anywhere else would make the demo's central claim false.
- Editing `tests/golden/video/**` to accommodate a change; regenerate it only through the `--ignored`
  generator and say so in the diff.
