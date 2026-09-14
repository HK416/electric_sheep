<!-- Korean translation of docs/packets/M3/P-M3-W1-R4.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-W1-R4 — es-ros2 oracle을 CI에서 실행하고, SKIP을 green으로 두지 않는다

Spec: §1.4(판정하는 것은 reference oracle이다; 어떤 oracle도 재도출하지 않는 golden은 증거가 아니라
주장일 뿐이다), §26.2(CI tier: PR vs oracle job), §12.4.
`docs/reviews/M3-W1.md`의 **S-3**을 닫는다; M4 **S-7**의 es-ros2 몫이다.

## context (범위)

```
.github/workflows/ci.yml
xtask/src/main.rs
docs/design/ros2-boundary.md
docs/design/ros2-boundary.ko.md
docs/packets/M3/P-M3-W1-R4.md
```

## spec (사양)

es-ros2 oracle 네 개가 존재하지만 어느 것도 자동화에서 돌지 않는다.

| 테스트 | 필요 조건 | 현재 |
|---|---|---|
| `gen_goldens::goldens_are_what_rosbags_and_xxhash_produce` | `rosbags`, `xxhash`가 있는 `ES_PYTHON` | 어디서나 SKIP |
| `gen_goldens::robostack_hashes_and_rclpy_bytes_agree_with_the_goldens` | `ES_ROS2_ENV` | 어디서나 SKIP |
| `gen_camera_goldens`(rosbags + OpenCV 부분) | `rosbags`, `cv2`가 있는 `ES_PYTHON` | 어디서나 SKIP |
| `gen_camera_goldens`(`--ros` 부분) | `cv_bridge`, `image_geometry`가 있는 `ES_ROS2_ENV` | 어디서나 SKIP |

`ci.yml:113-127`은 `es-physics-backend`/`es-policy`/`es-compile`에만 `ES_PYTHON`을 설정하고 SKIP
guard(`:124`)도 같은 셋만 지목한다; `:159-164`는 `rmw_zenoh_interop`에만 `ES_ROS2_ENV`를 준다.
그래서 camera golden 22개, CDR golden 10개, `rihs01.json`, `gid.json`이 체크인되어 있을 뿐 한 번도
재도출되지 않는다. `xtask`의 `is_gpu_skip`(`main.rs:37`)도 이를 잡지 못하는데, 그 사유에 GPU 단어가
없기 때문이다 — 그리고 그것은 의도된 것이다(디자인 노트 section 8이 SKIP 사유에 그 단어들을
금지한다). 결과적으로 `cargo xtask ci`는 모든 es-ros2 oracle이 조용히 빠진 채 green을 보고한다.

그 사이 W1c가 `ros-kilted-cv-bridge 4.1.0`과 `ros-kilted-image-geometry 4.1.0`이 `ros-base` /
`rmw-zenoh-cpp`와 같은 프리픽스에 co-install된다는 것을 **측정**했으므로
(`docs/packets/M3/W1c-camera-ingest.md`의 "Environment, measured 2026-09-14"), `ci.yml:150-152`와
디자인 노트 section 8 마지막 문단의 경고는 낡았고 프리픽스 하나로 넷 모두를 감당할 수 있다.

변경 사항:

1. `ci.yml`의 oracle 환경 구성 단계가 기존 `micromamba create`에 `ros-kilted-cv-bridge=4.1.0
   ros-kilted-image-geometry=4.1.0`을 추가하고, 그 주석을 측정 결과로 교체한다.
2. `install reference oracles` 단계가 `rosbags==0.11.5 xxhash==4.0.1
   opencv-python-headless==5.0.0.93`을 추가한다(디자인 노트 section 8이 고정한 버전).
3. interop 단계 뒤에 새 단계를 추가해, `ES_PYTHON`과 `ES_ROS2_ENV`를 설정한 채 두 provenance 하네스를
   실행하고 각자가 `RAN` 라인을 출력했는지 assert한다:
   `grep -q '^RAN gen_goldens'`, `grep -q '^RAN robostack_hashes_and_rclpy_bytes_agree_with_the_goldens'`
   (`gen_goldens.rs:110,162`), `grep -q '^RAN gen_camera_goldens'`(`gen_camera_goldens.rs:164`).
   세 라인 모두 이미 존재하므로 `crates/` 수정은 필요 없다.
4. `xtask/src/main.rs`에 `ES_REQUIRE_GPU=1`과 대칭인 `ES_REQUIRE_ORACLES=1` 모드를 추가한다:
   `is_gpu_skip`이 자기 것이라 주장하지 **않는** 모든 `SKIP` 라인이 run을 실패시킨다. 설정하지
   않으면 GPU 경로와 똑같이 `NOTE`로 출력된다. `is_gpu_skip` 자체는 그대로이고, 기존 xtask 단위
   테스트에 non-GPU SKIP 라인 케이스 하나가 추가된다.

디자인 노트 section 8의 표에서 해당 네 행의 "Runs in" 값이 "oracle job"이 되고,
"co-installation is unverified" 문장은 측정 결과로 교체된다.

## oracle (검증)

```
cargo xtask ci
ES_REQUIRE_ORACLES=1 cargo xtask ci     # 여기서는 의도대로 실패한다: rosbags가 없다
cargo test -p xtask
cargo fmt --check
cargo clippy --workspace --all-targets --features es-ros2/zenoh -- -D warnings
```

oracle 호스트(Linux, `ES_PYTHON`과 `ES_ROS2_ENV` 설정)에서 두 하네스가 세 `RAN` 라인을 모두
출력하고 체크인된 모든 파일을 byte 단위로 재도출해야 한다. 실행했을 때의 호스트, RoboStack 빌드
문자열, 날짜를 이 파일에 기록한다.

- `xtask` 단위 테스트 `a_non_gpu_skip_is_not_a_gpu_skip` — `is_gpu_skip("SKIP gen_camera_goldens: no
  Python interpreter with rosbags + cv2 …")`가 `false`이므로 `ES_REQUIRE_ORACLES=1`이 이를 잡는다.
- `xtask` 단위 테스트: `ES_REQUIRE_ORACLES`는 기본적으로 꺼져 있다 — oracle이 없는 머신에서 기존
  `cargo xtask ci`가 여전히 exit 0이다.

## acceptance (수용 기준)

- 환경 변수를 설정하지 않았을 때 `cargo xtask ci`의 동작이 변하지 않는다(이 머신에서 exit 0).
- `ES_REQUIRE_ORACLES=1 cargo xtask ci`가 실패하고 거부한 모든 SKIP을 지목한다.
- oracle job이 es-ros2 oracle 넷을 모두 실행하고, `grep -q '^RAN …'` guard가 그중 하나라도 skip하면
  단계를 실패시킨다.
- PR job의 wall time이 여전히 §26.2의 10분 안에 있다; 측정한 cold 시간을 관측값으로 이 파일에
  기록한다, `Target / Status: unverified`.
- golden 파일 변경 없음(`cargo xtask verify-goldens`가 계속 0 changed를 보고한다).

## forbidden (금지)

- `crates/` 아래 모든 파일. 세 `RAN` 라인은 이미 존재하며, 이 패킷이 건드리는 Rust는
  `xtask/src/main.rs`와 그 단위 테스트뿐이다.
- `tests/golden/`이나 `tests/fixtures/` 아래 파일을 편집·재생성·"수정"하는 것 — oracle이 체크인된
  golden과 어긋난다면 그것은 패치할 일이 아니라 보고할 발견 사항이다(§1.4).
- `is_gpu_skip`이 삼키도록 SKIP 사유를 느슨하게 만들거나 사유에 GPU 단어를 넣는 것.
- Docker, sudo, 권한이 필요한 설치 단계(디자인 노트 section 8).
- 그 밖의 모든 M3 W1 발견 사항.
