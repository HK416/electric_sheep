# P-M3-W1-R5 — camera boundary codes: ROI bounds, all-zero `r`, `CAM-010`

Spec: §7.2 and Appendix B.2 (INV-14: every size change transforms the intrinsics), §26.1 ("what is
not validated is not executed"), §25.1 (a `CameraInfo` is outside data), §18.3.
Design note `docs/design/ros2-boundary.md` sections 6.1, 6.2 and 6.4.
Closes **S-4** and **S-5** of `docs/reviews/M3-W1.md` and implements human decisions 2, 4 and 5 of
that review. **Do not start this packet before those three decisions are ruled on**; if the ruling
differs, the packet changes with it.

## context

```
crates/es-ros2/src/camera.rs
crates/es-ros2/src/error.rs
crates/es-ros2/tests/camera_ingest.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R5.md
```

## spec

Three boundary conditions the current `camera.rs` gets wrong or conflates.

**1. The ROI is not bounds-checked (S-4).** `roi_and_binning` (`:463-494`) rejects a zero dimension
and a binning that does not divide, then hands the rectangle to `ImageSpec::cropped`
(`es-ir-types/src/image.rs:174`), which subtracts the origin from `cx`/`cy` unconditionally. A
`CameraInfo` with `roi.x_offset = 10_000` on a 640×480 calibration therefore produces a valid-looking
`ImageSpec` for a rectangle outside the sensor, with a negative `cx`. Add, before the crop:
`roi.x_offset + roi.width <= info.width` and `roi.y_offset + roi.height <= info.height`, in `u64` so
the addition cannot wrap, → `CAM-006` naming the rectangle and the calibration size.

**2. `CAM-006` conflates two conditions (S-4).** The zero-dimension branch (`:469-474`) and the
divisibility branch (`:485-489`) both produce `CAM-006`, whose design-note meaning is "the resize
ratio would not be `1 / binning`". Keep one code — `CAM-006` is "the ROI/binning pair is not
usable" — but make the three messages say which of the three it was, and cover each one by a test.

**3. An all-zero `r` (human decision 2).** `check_monocular` (`:379-394`) compares `info.r` against
the identity exactly, so the all-zero `r` that ROS drivers publish for an uncalibrated monocular
camera is `CAM-005`. Accept `r == [0.0; 9]` as the identity — the only two shapes accepted stay
"identity" and "all zero", everything else is still `CAM-005`, and no tolerance is introduced (a
stereo `r` differs from the identity by far more than any epsilon, and "close to the identity" is
still not a thing ROS publishes). `camera_ingest.rs:246`'s
`non_identity_r_or_nonzero_tx_is_rejected` currently asserts `CAM-005` for `r = [0.0; 9]`; that
case moves to the new accepting test.

**4. `CAM-008` covers two unrelated conditions (S-5, human decision 4).** `error.rs:198` maps both
`DeclaredMismatch` and `NotCalibrated` to `CAM-008`, so "no `CameraInfo` has arrived yet" — a
startup race a caller retries through — is indistinguishable from "this camera is calibrated for a
different stream", which stops the line. Give `NotCalibrated` its own `CAM-010`. Design note 6.4
and section 6's code table gain the row; `camera.rs:934`'s and `camera_ingest.rs:699`'s assertions
move with it.

No signature changes: `derive_spec`, `check_declared`, `CameraIngest::on_image` and
`CameraError::code` keep their shapes, and `camera.rs` still writes an intrinsic only at
`derive_spec`'s `k`/`p` indexing (INV-14).

## oracle

```
cargo test -p es-ros2 --test camera_ingest
cargo test -p es-ros2 --lib camera
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

The goldens do not change: every new case mutates a decoded fixture in the test, as
`roi_width_not_divisible_by_binning_is_rejected` already does.

- `an_roi_outside_the_calibration_is_rejected` — `info_b` with `roi.x_offset = 10_000`, and
  separately with `y_offset + height > info.height`: `CAM-006` both times, message naming the
  calibration size. **FAILS before the fix** (both return `Ok` or `CAM-007`).
- `roi_rejections_name_which_rule_failed` — the three `CAM-006` messages (zero dimension, outside
  the calibration, indivisible by binning) are distinct strings.
- `an_all_zero_r_is_the_identity` — `info_a` with `r = [0.0; 9]` derives the same `ImageSpec` as the
  unmodified fixture, matching `matrix("info_a", "intrinsic_matrix")` as the existing test does.
  `r = [0.0; 8] + [1.0]` and a genuinely rotated `r` are still `CAM-005`.
- `no_camera_info_is_its_own_code` — `CameraIngest::on_image` before any `on_camera_info` is
  `CAM-010`; a `CameraInfo` whose calibration contradicts the declared spec is still `CAM-008`.
  Updates `camera_ingest_validates_then_decodes_and_dates_a_frame` (`:699`) and
  `an_image_without_a_camera_info_is_not_executed` (`camera.rs:934`).

## acceptance

- All four tests pass; the first and the last are confirmed to fail on the unfixed code.
- The 19 existing `camera_ingest` tests and the 5 `camera` unit tests still pass, with
  `non_identity_r_or_nonzero_tx_is_rejected` losing exactly the `r = [0.0; 9]` case.
- `CameraError::code()` returns `CAM-001`..`CAM-010`; design note section 6's table and 6.4 list
  `CAM-010`, and `ros2-boundary.ko.md` is updated in the same commit.
- `cargo xtask verify-goldens` still reports 0 changed.
- No new dependency, no `HashMap`, no new trait, ≤ 40 added source lines in `camera.rs`.

## forbidden

- `crates/es-ir`, `crates/es-ir-types` (`ImageSpec::cropped` is correct and stays; no new
  `DistortionModel` variant — that is human decision 1, still "keep rejecting"), `crates/es-safety`,
  `crates/es-sensor`, every other crate.
- `hil`, `session`, `config`, `actuator`, `cdr`, `msg`.
- Editing any file under `tests/golden/` — every new case mutates a decoded fixture in the test.
- Pixel rectification or undistortion, Bayer demosaicing, `CompressedImage`, TF.
- Introducing a tolerance into `check_monocular`, or relaxing the exact `k4..k6 == 0` test in
  `distortion_of`.
- Any other M3 W1 finding.
