# P-M3-W1-R4 — the es-ros2 oracles run in CI, and a SKIP is not green

Spec: §1.4 (the reference oracle is what judges; a golden that no oracle re-derives is an
assertion, not evidence), §26.2 (CI tiers: PR vs oracle job), §12.4.
Closes **S-3** of `docs/reviews/M3-W1.md`; the es-ros2 half of M4's **S-7**.

## context

```
.github/workflows/ci.yml
xtask/src/main.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R4.md
```

## spec

Four es-ros2 oracles exist and none of them runs in automation:

| Test | Needs | Today |
|---|---|---|
| `gen_goldens::goldens_are_what_rosbags_and_xxhash_produce` | `ES_PYTHON` with `rosbags`, `xxhash` | SKIP everywhere |
| `gen_goldens::robostack_hashes_and_rclpy_bytes_agree_with_the_goldens` | `ES_ROS2_ENV` | SKIP everywhere |
| `gen_camera_goldens` (rosbags + OpenCV part) | `ES_PYTHON` with `rosbags`, `cv2` | SKIP everywhere |
| `gen_camera_goldens` (`--ros` part) | `ES_ROS2_ENV` with `cv_bridge`, `image_geometry` | SKIP everywhere |

`ci.yml:113-127` sets `ES_PYTHON` for `es-physics-backend`/`es-policy`/`es-compile` only, and its
SKIP guard (`:124`) names the same three; `:159-164` sets `ES_ROS2_ENV` for `rmw_zenoh_interop`
alone. So 22 camera goldens, 10 CDR goldens, `rihs01.json` and `gid.json` are checked in and never
re-derived. `xtask`'s `is_gpu_skip` (`main.rs:37`) does not catch them either, because their reasons
carry no GPU word — by design (design note section 8 forbids those words in a SKIP reason), which
means `cargo xtask ci` reports green with every es-ros2 oracle silently absent.

W1c has since **measured** that `ros-kilted-cv-bridge 4.1.0` and `ros-kilted-image-geometry 4.1.0`
co-install in the same prefix as `ros-base` / `rmw-zenoh-cpp` (`docs/packets/M3/W1c-camera-ingest.md`,
"Environment, measured 2026-09-14"), so the caution in `ci.yml:150-152` and in design note section
8's last paragraph is out of date and one prefix serves all four.

Changes:

1. `ci.yml`'s oracle-environment step adds `ros-kilted-cv-bridge=4.1.0
   ros-kilted-image-geometry=4.1.0` to the existing `micromamba create`, and its comment is replaced
   by the measured result.
2. The `install reference oracles` step adds `rosbags==0.11.5 xxhash==4.0.1
   opencv-python-headless==5.0.0.93` (the versions design note section 8 pins).
3. A new step, after the interop one, runs both provenance harnesses with `ES_PYTHON` and
   `ES_ROS2_ENV` set and asserts each printed its `RAN` line:
   `grep -q '^RAN gen_goldens'`, `grep -q '^RAN robostack_hashes_and_rclpy_bytes_agree_with_the_goldens'`
   (`gen_goldens.rs:110,162`) and `grep -q '^RAN gen_camera_goldens'` (`gen_camera_goldens.rs:164`).
   All three lines exist already; no `crates/` edit is needed for them.
4. `xtask/src/main.rs` gains an `ES_REQUIRE_ORACLES=1` mode symmetrical to `ES_REQUIRE_GPU=1`: any
   captured `SKIP` line that `is_gpu_skip` does **not** claim fails the run. Unset, it is printed as
   a `NOTE` exactly as the GPU path does. `is_gpu_skip` itself is unchanged, and its existing xtask
   unit tests gain one case for a non-GPU SKIP line.

Design note section 8's table gains a "Runs in" value of "oracle job" for the four rows, and its
"co-installation is unverified" sentence is replaced with the measured result.

## oracle

```
cargo xtask ci
ES_REQUIRE_ORACLES=1 cargo xtask ci     # fails locally, by design: no rosbags here
cargo test -p xtask
cargo fmt --check
cargo clippy --workspace --all-targets --features es-ros2/zenoh -- -D warnings
```

On the oracle host (Linux, `ES_PYTHON` and `ES_ROS2_ENV` set), the two harnesses must print all
three `RAN` lines and re-derive every checked-in file byte for byte. Record the host, the RoboStack
build strings and the date in this file when they are run.

- `xtask` unit test `a_non_gpu_skip_is_not_a_gpu_skip` — `is_gpu_skip("SKIP gen_camera_goldens: no
  Python interpreter with rosbags + cv2 …")` is `false`, so `ES_REQUIRE_ORACLES=1` would catch it.
- `xtask` unit test `ES_REQUIRE_ORACLES` is off by default: the existing `cargo xtask ci` on a
  machine without the oracles still exits 0.

## acceptance

- `cargo xtask ci` unchanged in behaviour with no environment variables set (exit 0 on this box).
- `ES_REQUIRE_ORACLES=1 cargo xtask ci` fails and names every SKIP it refused.
- The oracle job runs all four es-ros2 oracles and its `grep -q '^RAN …'` guards fail the step if
  any of them skips.
- The PR job's wall time is still inside §26.2's 10 minutes; report the measured cold time in this
  file as an observation, `Target / Status: unverified`.
- No golden file changes (`cargo xtask verify-goldens` still reports 0 changed).

## forbidden

- Every file under `crates/`. The three `RAN` lines already exist; the only Rust this packet
  touches is `xtask/src/main.rs` and its unit tests.
- Editing, regenerating or "fixing" any file under `tests/golden/` or `tests/fixtures/` — if the
  oracle disagrees with a checked-in golden, that is a finding to report, not to patch (§1.4).
- Loosening a SKIP reason so `is_gpu_skip` swallows it, or adding a GPU word to one.
- Docker, sudo, or a privileged install step (design note section 8).
- Any other M3 W1 finding.
