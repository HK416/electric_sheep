# M10 W1a — 패스 트레이서의 샘플 시드가 틱마다 바뀌고, U4를 U5로 다시 돌린다

스펙: §6(2026-09-22에 추가된 절: `render.seed = "fixed" | "tick"`; `fixed` = 부재 = 오늘의 바이트; `tick`은
에피소드 상대 틱을 시드에 섞고 수집기와 평가기는 같은 `(에피소드, 틱)`에서 여전히 일치한다), §3.4(PT는
결정적: 카운터 RNG, 고정 샘플 순서), §15.3, §28.10 규칙 1(커밋된 픽셀은 움직이지 않는다), §28.13 규칙 3과
파동 1, §13.3(움직인 `task_hash`는 새 비교다). 리뷰: `docs/reviews/M7.md` R13(포즈 지문으로서의 64 spp
그레인), M8 S-11 / R9; `visible-learning.md` 7.32(U4: held-out 0.0, train-seeds 0.0625, 손실은 U3보다 낮음).
설계 노트: `renderer.md` 12절(R5)에 12.8이, `visible-learning.md`에 7.37이 생긴다. **W0b**에 의존(재생성된
문서; `task-pt-tick`은 재생성된 `task-pt`에서 파생). 유형 B, D 행 하나.

## 질문

`RenderConfig::seed`는 상수이고 관측 경로에는 accumulation이 없어 매 호출 `frame == 0`이며 PT 샘플 키는
`(픽셀, 포즈)`의 순수 함수다: 64 spp는 포즈별 고정 텍스처를 남기고, U4의 정책은 — 손실 0.0045, U3의 0.0067보다
낮은데 — held-out 16 중 0, 자기 학습 시드 16 중 1이다. **센서가 `seed = "tick"`을 선언할 수 있어 같은 포즈가
다른 틱에서 다른 그레인을 갖되 모든 `(에피소드, 틱)`이 수집기와 평가기 사이에 비트 재현되면, PT 정책이 그레인
대신 과제를 배우는가 — U3의 0.5625와 U4의 0.0 옆의 U5 행?**

## 명세

* `es_ir::task::SensorRender`에 `seed: SeedStream`(`Fixed` 기본 | `Tick`), `#[serde(default,
  skip_serializing_if = "SeedStream::is_fixed")]`, TOML `seed = "tick"`. `SensorRender::canonical`은
  **`Tick`일 때만** 기존 필드 뒤에 `seed` 항목을 쓴다 — `is_default()`가 이를 포함 — 그래서 커밋된 `task.toml`(`Rs`)과
  재생성된 `task-pt.toml`(`Pt`, seed 부재)의 `task_hash`는 유지된다; 테스트가 둘을 고정한다.
* `es_env::render::EnvRendererCfg`에 패스스루 `seed_stream`; `sensor_cfg`가 센서에서 복사; `render_config`는
  `RenderConfig::seed`를 기본값으로 둔다. `EnvRenderer::frame`(`crates/es-env/src/render.rs:305-334`):
  `Tick`이면 렌더 전에 렌더러의 시드를 `rng::fmix32(cfg.seed ^ tick)`(`crates/es-render/src/rng.rs`가 이미
  내보내는 믹서; private이면 `pub`으로)로 두되 **`tick`은 에피소드 상대 틱** — M7/R1 이후 `events.json`과 수집기의
  스트림 2가 싣는 그 카운트를 그들이 읽는 곳에서 읽는다; `EnvRenderer::frame`의 파일 카운터(에피소드를 가로질러
  증가)가 아니고, `begin_episode`에서 리셋됨이 보여지지 않는 한 `StateView.tick`도 아니다. 렌더러는 `p[13]` 슬롯을
  다시 쓰는 `set_seed(u32)`(또는 `render`가 시드를 받음)를 얻는다; CPU 레퍼런스는 같은 `cfg.seed`를 읽으므로
  CPU/GPU 패리티는 그대로다. `es video showcase`의 장면 카메라는 `sensor_cfg`를 거치므로 문서를 따른다.
* 문서(R5의 5문서 패턴, 확장한 `generate_pt_fixtures`가 생성): `tests/fixtures/visible-learning/task-pt-tick.toml`
  (= 재생성된 `task-pt.toml` + `seed = "tick"`), `observation-pt-tick.toml`, `evaluation-pt-tick.toml`; 서버에
  `observation-augmented-pt-tick.toml`과 train-seeds Evaluation IR; `training-u5.toml`(= `training-u4.toml`의
  레시피를 새 데이터셋·번들로, 경로는 `~/artifacts/plan-w/w1a/`). 해시는 7.37에 기록.
* **서버(GPU 큐 `~/artifacts/plan-w/queue/gpu.lock`, `~/artifacts/plan-w/w1a/`, 트리 `~/Projects/es-w1a`,
  `w1a.sh`를 `nohup`과 스테이지 마커로, ≈ 4.2 h):**
  0. *reading (b)를 먼저, 1분:* `task-pt-tick` 아래 `--frames`로 수집한 PT 에피소드 하나(seed 1)와 같은 시드를
     `--frames`로 평가한 것: 틱 0(리셋 포즈, 같은 상태)의 프레임이 비트 동일 — T7이 `Rs`에서 증명한 수집기/평가기
     패리티를 이제 틱 시드의 `Pt`에서. 다르면 멈추고 보고: U4 결과에 두 번째 원인이 있다.
  1. `es loop collect` 시연 200개, seed 1, `--frames`, `task-pt-tick`(≈ 36분).
  2. `es train --recipe training-u5.toml`(20,000 스텝, ≈ 4분).
  3. held-out 시드 16 × 스위트 6(`--jobs 6`, ≈ 3 h)과 학습 시드 16(≈ 30분).
  U5 행을 `visible-learning.md` 7.37에 U3(0.5625)·U4(0.0 / 0.0625) 옆에: `success_rate`, `envelope_violation_rate`,
  `episode_length`, `passed`, 손실 곡선 끝, `evaluation_hash`, `execution_hash`, wall-clock(P-M9-R5 학습과 겹치면
  "under load" 표기), 그리고 M7 addendum이 필요로 하는 문장: 그레인이 원인이었는가. 프레임 트리는 읽은 뒤 정리.

## context

```
crates/es-ir/src/task.rs
crates/es-ir/tests/**
crates/es-render/src/rng.rs
crates/es-render/src/renderer.rs
crates/es-render/src/view.rs
crates/es-render/tests/**
crates/es-env/src/render.rs
crates/es-env/tests/**
crates/es/src/cmd/loop.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/showcase.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/task-pt-tick.toml
tests/fixtures/visible-learning/observation-pt-tick.toml
tests/fixtures/visible-learning/evaluation-pt-tick.toml
tests/fixtures/visible-learning/training-u5.toml
docs/design/renderer.md
docs/design/renderer.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M10/W1a-pt-seed-tick.md
docs/packets/M10/W1a-pt-seed-tick.ko.md
```

## 오라클

1. `cargo test -p es-ir committed_task_hashes_are_unmoved_by_seed_stream` — `task.toml`과 `task-pt.toml` 해시
   고정; 명시한 `seed = "fixed"`는 부재와 같게 해시; `seed = "tick"`은 해시를 옮김; 모르는 값은 이름으로 거부.
2. `cargo test -p es-env --features render pt_seed_varies_per_tick_and_is_reproducible`(GPU; 없으면 이유와
   함께 SKIP): 데모 장면에서 한 포즈를 `Tick` 아래 틱 0과 1에서 렌더하면 다른 바이트; 같은 `(포즈, 틱)` 두 번은
   비트 동일; `Fixed` 아래 바이트는 변경 전 렌더(시드 슬롯을 건드리지 않은 렌더로 테스트 안에서 고정)와 같다.
3. `cargo test -p es-render` — `set_seed` 뒤 1 spp PT 프레임에서 CPU 레퍼런스와 GPU 일치; 기존 PT 골든
   (`cornell_pt1spp` …) 불변.
4. 서버 스테이지 0(패리티), 그다음 1~3; 7.37행 작성; `renderer.md` 12.8.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W1a-pt-seed-tick.md`.

## 수용

오라클 1~5; 7.37과 12.8과 한국어 짝; M7 addendum의 R13 줄은 리뷰가 갱신한다(이 패킷이 아님).

## 금지

관측 경로의 accumulation·SVGF·디노이저 일체; 벤더 RR(DLSS/FSR); `RenderConfig::seed` 기본값이나 `Fixed`의 바이트
변경; `ImageSpec`(INV-14)이나 Observation IR 건드리기; `docs/ARCHITECTURE*.md`; `tests/golden/**`;
`EnvRendererCfg`를 만드는 두 번째 장소.
