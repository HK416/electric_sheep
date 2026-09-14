# M5 V4 — 4x4 그리드 영상: 모자이크, 오버레이, 인코딩

설계 노트: `docs/design/visible-learning.md` 섹션 7.2, 8, 9; 섹션 2.9를 먼저 읽을 것 — 오라클 서버에는
**`ffmpeg` 바이너리가 없고**, 거기서 인코딩되는 코덱은 `mp4v`뿐이다. V0, V0b와 독립: 이 패킷은 프레임
디렉터리와 `events.json`의 *소비자*로 쓰였고 오라클은 체크인된 합성 픽스처이므로, 플랜 V가 요청한 대로
V0과 정확히 병렬로 만들 수 있다. 실제 입력은 V3가 만든다.

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

메모: `video.rs`는 새 파일 — `es video mosaic`이며 순수 Rust이고 GPU도 Vulkan도 Python도 필요 없다.
`python/es/encode_video.py`가 인코더이자 유일한 Python이고 `cargo xtask context-budget`에 집계되지
않는다 (`xtask/src/context_budget.rs:148-152`). 픽스처는 작은 합성 프레임 디렉터리 16개와 모든
`ActionSource`를 덮는 `events.json`이며, 테스트의 `--ignored` 생성기가 한 번 만든 뒤 읽기 전용이 된다.

## spec

- §1.4: 모자이크는 고정 픽스처에서 만든 골든으로, 인코딩은 존재하고 프레임 수가 맞고 디코드되는 파일로
  판정된다.
- §2.4: 모자이크와 오버레이는 Rust이며 Python 없이 돈다. Python은 컨테이너 인코딩뿐이고, 그것은 표현
  단계이지 런타임 경로가 아니다.
- §3.4: 모자이크는 바이트에 대한 정수 연산이다 — float 블렌딩 없음, `HashMap` 순회 순서 없음, 전역 RNG
  없음. 같은 프레임과 같은 `events.json`은 어떤 머신에서도 같은 모자이크를 준다.
- §5.3, §3.5: 해시 체인이 덮는 산출물은 프레임이고 mp4는 **아니다**. 인코더가 체인 밖 호스트
  라이브러리이기 때문이다. 설계 노트 섹션 9가 그 표이며, `es video`는 다르게 암시하는 대신 그 문장을
  직접 출력한다.
- §12.4: 오버레이는 성공률과 에피소드 인덱스를 보여준다. 초당 무엇도 보여주지 않는다.
- §25.1: `events.json`과 프레임 사이드카는 신뢰할 수 없는 입력으로 읽는다 — `layout.json`이 바이트
  길이와 어긋나는 프레임, 범위를 벗어난 `frame` 인덱스, 16이 아닌 셀 수는 어떤 할당보다 먼저 이름을
  밝히는 에러다.
- §1.5: `es`는 2,946 코드 줄; 이 패킷은 `src/` 기준 약 350줄 이하로 잡는다.

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

레퍼런스 — `cv2`가 필요한 인코딩:

```
ES_CV2_PYTHON=$HOME/venvs/es-lerobot/bin/python cargo test -p es --test video -- --ignored --nocapture
```

**2026-09-14 오라클 서버에서 측정.** `command -v ffmpeg`가 빈 출력이다 — `ffmpeg`, `ffprobe`,
`gst-launch-1.0`, `convert` 바이너리가 없다 — 반면 FFmpeg 공유 라이브러리
(`libavcodec62` .. `libavutil60`, `7:8.0.1-3ubuntu2`)는 설치되어 있다. `~/venvs/es-lerobot`에는
`FFMPEG: YES`로 빌드된 `opencv-python-headless 4.13.0.92`가 있다. 직접 테스트한 결과: fourcc `mp4v`의
`cv2.VideoWriter`는 열리고 기록되지만 (64x64 10프레임 -> 1450바이트), `avc1`과 `H264`는 둘 다 열리지
않는다 — 링크된 유일한 H.264 인코더가 `h264_v4l2m2m`이고 `Could not find a valid device`를 보고한다.
**따라서 코덱은 `mp4v` (MPEG-4 Part 2), 컨테이너는 `.mp4`다.** 플랜 V는 아무것도 설치하지 않으며,
H.264는 설계 노트 미해결 질문 5다.

`crates/es/tests/video.rs`:

- `mosaic_matches_the_golden` — 체크인된 픽스처에 대한 `es video mosaic`이
  `tests/golden/video/mosaic_4x4_frame0.bin`과 바이트 동일하고 사이드카도 일치한다. 이 패킷의 결정성
  주장 전부이며, 장치도 Python도 없이 PR 티어에서 돈다.
- `two_runs_are_byte_identical` — 같은 입력 두 번.
- `a_clamped_record_draws_a_red_border` — `events.json` 레코드가 `Clamped`나 `Fallback`인 셀은 테두리
  픽셀이 붉은 상수로 설정되고 내부는 그대로다; `Policy` 셀에는 테두리가 없다. 테스트는 시각적 인상이
  아니라 픽셀 구간을 비교한다.
- `the_overlay_digits_are_a_fixed_bitmap` — 성공률·에피소드 텍스트는 내장 비트맵 폰트 표에서 그려지므로
  시스템 폰트, 로케일, float 포매팅이 픽셀을 움직일 수 없다.
- `a_short_cell_holds_its_last_frame` — 프레임 수가 다른 셀은 마지막 프레임을 반복해 채우고, 모자이크의
  프레임 수는 최댓값이다. 최솟값으로 자르면 일찍 실패한 셀을 정확히 숨기게 된다.
- `mismatched_layouts_are_rejected` — 크기가 다른 `layout.json`을 가진 두 셀은 리사이즈가 아니라 이름을
  밝히는 에러다 (§26.1: 검증되지 않은 것은 실행되지 않는다; 이 코드가 결코 이미지를 재샘플링하지 않는
  다는 점에서 INV-14의 정신).
- `malformed_inputs_are_rejected_before_allocation` — 사이드카와 어긋난 바이트 길이, 끝을 넘어선 `frame`
  인덱스, 16이 아닌 셀 수, `shape` 곱이 오버플로하는 `layout.json`: 이름 있는 에러 넷, 패닉 없음.
  proptest: 임의의 `events.json` 값이 결코 패닉하지 않는다.
- `encode_produces_a_playable_file` (`--ignored`) — `encode_video.py`가 모자이크 프레임으로 `.mp4`를
  쓰고, `cv2.VideoCapture`로 되읽은 프레임 수·너비·높이가 쓴 값과 같다. `cv2`가 없으면
  `SKIP encode_video: <why>`; 실행되면 `RAN encode_video`.

`crates/es/tests/cli.rs`: `video_usage_errors_exit_2`, `video_mosaic_missing_events_is_exit_1`.

## acceptance

```rust
// crates/es/src/cmd/video.rs
// es video mosaic --frames <dir> --events <events.json> --report <report.json>
//                 --grid 4x4 --out <dir> [--label-height N]
//   모든 셀의 <frames>/<cell>/NNNNNN.bin + layout.json을 읽고,
//   <out>/NNNNNN.bin + layout.json을 쓴다: 타임스텝당 모자이크 프레임 하나.
pub fn dispatch(args: &[String]) -> i32;   // 0 ok, 1 runtime, 2 usage, 3 SKIPPED
```

```
python/es/encode_video.py --frames <모자이크 디렉터리> --out demo.mp4 --fps N [--codec mp4v]
```

- 모자이크는 셀 이름 순서(`BTreeMap`이므로 해시 순서가 아니라 파일 순서)의 `rows x cols` 셀이며, 각 셀의
  타일을 그대로 복사한다. 스케일링도 필터링도 색 변환도 없다.
- 해당 프레임의 `StepEvent`가 `Clamped`나 `Fallback`인 셀은 가장 바깥 픽셀 위에 고정 폭의 붉은 테두리를
  얻는다. `Policy`와 `Human`은 얻지 않는다.
- 그리드 아래 라벨 띠는 에피소드 인덱스와 `report.json`의 `MetricSpec::SuccessRate`
  (`crates/es-eval/src/metrics.rs:32`, `:97-100`)에서 온 진행 중 성공률을 싣고, 내장 비트맵 폰트로
  그린다 — `fontdb` 없음, 시스템 폰트 없음, 셰이핑 없음.
- 출력 프레임은 `tests/golden/render/*`와 같은 원시 `.bin` + `.json` 사이드카 레이아웃을 쓰므로,
  모자이크는 렌더된 프레임과 같은 수단으로 검증 가능하다 (설계 노트 섹션 7.2).
- `encode_video.py`는 fourcc `mp4v`의 `cv2.VideoWriter`를 쓴다; stdout에 JSON 한 줄
  `{"frames": n, "width": w, "height": h, "codec": "mp4v"}` 외에 아무것도 출력하지 않는다. 사이드카를
  읽으며 모양을 추측하지 않는다.
- `es video`는 mp4가 해시 체인 밖이고 증거는 프레임이라는 것을 한 번 출력한다.
- 새 트레이트 없음, 새 외부 크레이트 없음 (`png` 없음, `image` 없음, `fontdb` 없음, 코덱 크레이트 없음),
  GPU 없음, Vulkan 없음, 약 350 소스 줄 이하.

## forbidden

- 워크스페이스에 PNG·JPEG·폰트·비디오 코덱 크레이트를 추가하거나 `ffmpeg` 바이너리 의존을 만드는 것.
  프레임 포맷은 기존 원시 `.bin` + 사이드카이고 인코더는 호스트의 `cv2`다 (설계 노트 미해결 질문 5, 7).
- `crates/es-render`, `crates/es-gpu` — 이 패킷은 장치를 건드리지 않는다; 모자이크는 바이트 복사다.
- `crates/es-eval`, `crates/es-env`, `crates/es-data`, `crates/es-policy`, `crates/es-ir` — V0b, V1, V2,
  V3가 소유한다.
- 그리드를 맞추려고 셀을 스케일·크롭·재샘플링하는 것: 레이아웃 불일치는 에러다 (내부 파라미터 변환 없이
  이미지 크기가 바뀌지 않는다는 INV-14의 규칙을, 여기서는 크기를 아예 바꾸지 않는 방식으로 적용).
- 모자이크를 가장 짧은 셀에 맞춰 자르는 것.
- `events.json`이 아니라 정책에서 파생된 것을 그리는 것: 붉은 테두리는 Safety Plane의 결정이며, 그것을
  다른 데서 읽으면 데모의 핵심 주장이 거짓이 된다.
- 변경을 수용하려고 `tests/golden/video/**`를 편집하는 것; `--ignored` 생성기로만 재생성하고 diff에
  그렇게 명시할 것.
