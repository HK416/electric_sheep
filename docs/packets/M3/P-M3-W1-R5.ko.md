<!-- Korean translation of docs/packets/M3/P-M3-W1-R5.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R5 — 카메라 경계 코드: ROI 경계, all-zero `r`, `CAM-010`

Spec: §7.2와 Appendix B.2(INV-14: 모든 크기 변경은 intrinsic을 변환한다), §26.1("검증되지 않은 것은
실행되지 않는다"), §25.1(`CameraInfo`는 외부 데이터다), §18.3.
디자인 노트 `docs/design/ros2-boundary.md` section 6.1, 6.2, 6.4.
`docs/reviews/M3-W1.md`의 **S-4**와 **S-5**를 닫고, 같은 리뷰의 사람 판단 항목 2, 4, 5를 구현한다.
**그 세 판단이 내려지기 전에는 이 패킷을 시작하지 말 것**; 판단이 다르면 패킷도 함께 바뀐다.

## context (범위)

```
crates/es-ros2/src/camera.rs
crates/es-ros2/src/error.rs
crates/es-ros2/tests/camera_ingest.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R5.md
```

## spec (사양)

현재 `camera.rs`가 틀리게 처리하거나 뭉뚱그리고 있는 경계 조건 세 가지.

**1. ROI가 bounds check되지 않는다(S-4).** `roi_and_binning`(`:463-494`)은 zero dimension과 나누어
떨어지지 않는 binning을 거부한 뒤, 사각형을 그대로 `ImageSpec::cropped`
(`es-ir-types/src/image.rs:174`)에 넘긴다. 이 함수는 원점을 `cx`/`cy`에서 무조건 뺀다. 따라서
640×480 calibration에 `roi.x_offset = 10_000`인 `CameraInfo`는 센서 바깥 사각형에 대한, 그러나
겉보기에는 멀쩡한 `ImageSpec`을 만들어 내며 `cx`가 음수가 된다. crop 앞에 추가한다:
`roi.x_offset + roi.width <= info.width`와 `roi.y_offset + roi.height <= info.height`를, 덧셈이
wrap할 수 없도록 `u64`로. 실패 시 사각형과 calibration 크기를 지목하는 `CAM-006`.

**2. `CAM-006`이 두 조건을 뭉뚱그린다(S-4).** zero-dimension 분기(`:469-474`)와 divisibility
분기(`:485-489`)가 모두 `CAM-006`을 내는데, 디자인 노트에서 그 코드의 의미는 "resize 비율이
`1 / binning`이 되지 않는다"이다. 코드는 하나로 유지하되 — `CAM-006`은 "ROI/binning 쌍을 쓸 수
없다"이다 — 세 메시지가 셋 중 어느 것이었는지 말하게 하고, 각각을 테스트로 덮는다.

**3. all-zero `r`(사람 판단 2).** `check_monocular`(`:379-394`)은 `info.r`을 identity와 정확히
비교하므로, calibration되지 않은 monocular 카메라에 대해 ROS 드라이버가 publish하는 all-zero `r`이
`CAM-005`가 된다. `r == [0.0; 9]`를 identity로 받아들인다 — 받아들이는 형태는 "identity"와 "전부
0" 두 가지뿐이고 나머지는 여전히 `CAM-005`이며, tolerance는 도입하지 않는다(stereo `r`은 어떤
epsilon보다도 훨씬 크게 identity와 다르고, "identity에 가깝다"는 것은 여전히 ROS가 publish하는
대상이 아니다). 현재 `camera_ingest.rs:246`의 `non_identity_r_or_nonzero_tx_is_rejected`가
`r = [0.0; 9]`에 대해 `CAM-005`를 assert하고 있으므로, 그 케이스는 새로 만드는 수락 테스트로
옮긴다.

**4. `CAM-008`이 무관한 두 조건을 덮는다(S-5, 사람 판단 4).** `error.rs:198`이 `DeclaredMismatch`와
`NotCalibrated`를 모두 `CAM-008`에 매핑하므로, "아직 `CameraInfo`가 오지 않았다" — 호출자가
재시도로 넘어가는 startup race — 와 "이 카메라는 다른 스트림에 맞춰 calibration되어 있다" — 라인을
세워야 하는 상황 — 이 구별되지 않는다. `NotCalibrated`에 고유한 `CAM-010`을 준다. 디자인 노트 6.4와
section 6의 코드 표에 행이 추가되고, `camera.rs:934`와 `camera_ingest.rs:699`의 assertion이 함께
옮겨간다.

시그니처 변경은 없다: `derive_spec`, `check_declared`, `CameraIngest::on_image`,
`CameraError::code`는 모양을 유지하고, `camera.rs`는 여전히 `derive_spec`의 `k`/`p` 인덱싱에서만
intrinsic을 쓴다(INV-14).

## oracle (검증)

```
cargo test -p es-ros2 --test camera_ingest
cargo test -p es-ros2 --lib camera
cargo fmt --check
cargo clippy -p es-ros2 --all-targets --features zenoh -- -D warnings
cargo xtask check-spec-refs
cargo xtask verify-goldens
```

golden은 바뀌지 않는다: 새 케이스는 모두 `roi_width_not_divisible_by_binning_is_rejected`가 이미
그러듯 테스트 안에서 디코딩된 fixture를 변형한다.

- `an_roi_outside_the_calibration_is_rejected` — `roi.x_offset = 10_000`인 `info_b`, 그리고 별도로
  `y_offset + height > info.height`인 경우: 두 경우 모두 `CAM-006`이고 메시지가 calibration 크기를
  지목한다. 수정 전 **FAIL**(둘 다 `Ok`이거나 `CAM-007`을 반환한다).
- `roi_rejections_name_which_rule_failed` — 세 가지 `CAM-006` 메시지(zero dimension, calibration
  바깥, binning으로 나누어떨어지지 않음)가 서로 다른 문자열이다.
- `an_all_zero_r_is_the_identity` — `r = [0.0; 9]`인 `info_a`가 변형하지 않은 fixture와 동일한
  `ImageSpec`을 도출하고, 기존 테스트처럼 `matrix("info_a", "intrinsic_matrix")`와 일치한다.
  `r = [0.0; 8] + [1.0]`과 실제로 회전된 `r`은 여전히 `CAM-005`다.
- `no_camera_info_is_its_own_code` — `on_camera_info` 이전의 `CameraIngest::on_image`는 `CAM-010`;
  declared spec과 모순되는 calibration의 `CameraInfo`는 여전히 `CAM-008`.
  `camera_ingest_validates_then_decodes_and_dates_a_frame`(`:699`)과
  `an_image_without_a_camera_info_is_not_executed`(`camera.rs:934`)를 갱신한다.

## acceptance (수용 기준)

- 네 테스트가 모두 통과하고, 첫 번째와 마지막이 수정 전 코드에서 실패함이 확인된다.
- 기존 `camera_ingest` 테스트 19개와 `camera` 단위 테스트 5개가 계속 통과하며,
  `non_identity_r_or_nonzero_tx_is_rejected`에서는 정확히 `r = [0.0; 9]` 케이스만 빠진다.
- `CameraError::code()`가 `CAM-001`..`CAM-010`을 반환하고, 디자인 노트 section 6의 표와 6.4가
  `CAM-010`을 나열하며, `ros2-boundary.ko.md`가 같은 커밋에서 갱신된다.
- `cargo xtask verify-goldens`가 계속 0 changed를 보고한다.
- 새 의존성 없음, `HashMap` 없음, 새 trait 없음, `camera.rs`에 추가되는 소스 40줄 이하.

## forbidden (금지)

- `crates/es-ir`, `crates/es-ir-types`(`ImageSpec::cropped`은 올바르며 그대로; 새 `DistortionModel`
  variant 없음 — 그것은 사람 판단 1이고 여전히 "계속 거부"다), `crates/es-safety`,
  `crates/es-sensor`, 그 밖의 모든 크레이트.
- `hil`, `session`, `config`, `actuator`, `cdr`, `msg`.
- `tests/golden/` 아래 파일 편집 — 새 케이스는 모두 테스트 안에서 디코딩된 fixture를 변형한다.
- 픽셀 rectification이나 undistortion, Bayer demosaicing, `CompressedImage`, TF.
- `check_monocular`에 tolerance를 도입하거나 `distortion_of`의 정확한 `k4..k6 == 0` 검사를 완화하는
  것.
- 그 밖의 모든 M3 W1 발견 사항.
