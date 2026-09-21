# M7 E7 — 전체 루프를 살아 있는 채로: 수집과 학습이 발행하고, 에디터가 손실 곡선과 신경망이 보는 것을 그린다

스펙: §23.1(에디터는 돌아가는 프로세스의 클라이언트), §23.3("실행 중에 보이는 것": 실제 관측
이미지, 노드별 통계, 학습 곡선, 예산이 있는 샘플링, 이미지는 따로 제한, 게이트 9 < 1 %),
§13.1(수집 → 학습 → 평가 → 관찰이 하나의 작업 흐름), §12.4(아홉 지표, 절대 `step/s` 하나로 말하지
않는다), §19.3(`metrics.parquet`는 학습 곡선의 정체성 슬롯), §28.10 규칙 3. 2026-09-21 로컬 사이클
시연 뒤 소유자 메모: *에디터에서 손실 곡선 같은 데이터를 보여 달라*, 그리고 *학습 과정에는
렌더링된 화면이 없다 — 그것이 아쉽다.* 설계 노트:
`docs/design/telemetry-protocol.ko.md`(E4의 "프로듀서" 절이 프로듀서 둘과 스트림 하나를 얻는다),
`docs/design/editor-shell.ko.md` 16절. 의존: **E4**(`RunSink`/`Publisher` 모양, 스트림 1–4,
`--attach`), **E5**(Launch 패널의 `--telemetry` 칸), **T1/T2**(`es train`, `es loop cycle`),
**T6**(증강된 학습 텐서가 신경망이 보는 것이다).

## 물음

E4 뒤로는 `es eval run`만이 발행한다. *전체 루프*에서 시작을 누른 사람은 로그 줄이 십 분 동안
흐르는 것을 보고, 그림도 곡선도 보지 못한다. **수집은 평가가 하는 방식으로 자기 에피소드를 발행할
수 있는가, 학습은 자기 손실과 학습률과 처리량과 지금 맞추고 있는 바로 그 텐서를 `N` 스텝마다
1 % 미만의 비용으로 발행할 수 있는가, 그리고 주소 하나가 사이클의 네 단계를 모두 실어서 에디터가
루프를 하나의 살아 있는 실행으로 보여 줄 수 있는가?**

## 사양

* **공유되는 `Publisher` 하나.** E4의 `Publisher`(프레임 만들기, 논블로킹 `Server::publish`)가
  `crates/es/src/cmd/eval.rs`에서 `crates/es/src/cmd/telemetry.rs`로 옮겨 가고,
  모든 스트림 1 이벤트에 넣는 `stage: &'static str` 필드를 얻는다(`"collect"`, `"expert-gate"`,
  `"train"`, `"eval"`, `"showcase"`). 소비자가 소켓 하나에서 사이클의 단계를 구분할 수 있게
  하기 위해서다. `es loop cycle --telemetry <주소> [--telemetry-token] [--telemetry-image-every N]`은
  첫 단계 전에 **한 번** 바인드하고 같은 퍼블리셔를 프로세스 안의 모든 단계에 넘긴다. 독립
  명령은 `es eval run`이 그러듯 자기 것을 바인드한다. 스트림 1의 `stage.begin` / `stage.end`가
  각 단계를 그 벽시계 시간과 함께 감싼다.
* **수집이 발행한다.** `es loop collect --telemetry <주소>`는 수집기의 기존 틱별 훅(`FrameSink`,
  개입자)을 통해, **쓰는 것은 하나도 바꾸지 않고** 발행한다: 스트림 1의
  `episode.begin` / `episode.end { episode, seed, outcome, steps }`, 스트림 2의 제어 틱마다
  `[frame, tick, source, violation bits]` 하나 — `es eval run`과 정확히 같은, 플레인 자신의 판정 —,
  그리고 스트림 4의 `--telemetry-image-every` 틱마다의 관측 이미지. `--frames` 없는 수집은
  이미지를 발행하지 않고 그 사실을 한 번 말한다. 오라클: `--out` 아래 데이터셋이 플래그가 있든
  없든 바이트 단위로 같다.
* **학습이 발행한다.** `python/es/train_act.py --progress-every N`은 `N` 옵티마이저 스텝마다
  JSON 한 줄 `{"progress": {"step", "loss", "lr", "samples_per_s", "elapsed_s"}}`를 최종 요약 줄
  **앞에** stdout으로 찍는다(요약은 마지막 줄로 남고 바이트 단위로 같다). `--sample-every N`은
  `metrics/sample-<step>.bin` + `.json`(`{"shape":[h,w,3],"port":…}`)을 쓴다. 그 안에는
  **지금 묶음의 이미지 입력 하나가, 증강을 거친 뒤, 센서의 색 공간에서 `Rgb8`로** 들어 있다 —
  신경망이 맞추고 있는 텐서를, 보여 주기 위해서만 역정규화한 것 — 그리고 `{"sample": "<경로>"}`를
  찍는다. `es train --telemetry <주소>`는 트레이너의 stdout을 스트리밍 읽기로 바꾸고
  (`Stdio::piped` + `BufReader::lines`, 요약은 여전히 마지막 줄) 발행한다: 진행 줄마다
  스트림 **5**의 `Scalars([step, loss, lr, samples_per_s])`(새 스트림 id는 스키마가 아니라
  데이터다 — 노트의 표가 행 하나를 얻는다), 마크가 포장될 때 스트림 1의
  `checkpoint { step, policy_hash }`, 끝에 §12.4의 정직하게 채울 수 있는 필드로 이루어진 스트림 3의
  `Metrics`, 스트림 4의 샘플 이미지. 플래그가 없으면 `train_act.py`는 `--progress-every`도
  `--sample-every`도 받지 않으므로, 측정된 모든 실행의 계획과 `training.lock`은 움직이지 않는다
  (`plan-ir.txt` 골든 불변).
* **에디터가 그린다.** `model/train_view.rs`: `TrainView`가 스트림 5를
  `Curve { step: Vec<u32>, loss: Vec<f32>, lr: Vec<f32> }`로 접고, `checkpoints: Vec<(u32, String)>`,
  마지막 샘플 이미지, `throughput`을 보관하며 `eta(total_steps)`에 답한다. `LiveRun`(E4)은 단계
  스트립을 얻고, 평가의 `cell.*`를 접듯 수집의 `episode.*`를 접어서 사이클의 수집 에피소드가 Run
  탭의 행이 되게 한다. **Live** 탭(관찰)은 *Training* 절을 얻는다: 손실 곡선(로그 스케일 토글)과
  학습률 곡선을 `ui.painter()` 폴리라인으로 — **플로팅 의존성 없음** — 체크포인트 표시, 현재
  스텝 / 총 / ETA와 함께, 그리고 실행의 카메라 이미지 옆에 샘플 이미지를. Run 탭은 표 위에 단계
  스트립을 보인다. `app.rs`는 배선만 하고, 모든 숫자와 이름은 모델과 i18n 표(E6)에서 온다.
* **Launch 패널.** `es train`과 `es loop cycle` 종류가 `--telemetry`, 토큰,
  `--telemetry-image-every` 칸을 얻으므로(E5의 `LaunchField`는 공유된다), 전체 루프에서 시작을
  누르면 오늘 평가가 그러듯 스스로 붙는다. 세 `launch-*.txt` 골든은 한 번 다시 만든다(픽스처
  입력의 argv는 두 종류의 새 텔레메트리 칸만큼만 달라진다 — 노트에 그렇게 적는다. 그 밖에 골든은
  읽기 전용이다).
* **게이트 9.** 픽스처 베이크에서 `es train` 2,000 스텝을 `--telemetry --progress-every 10
  --sample-every 100`(빨아들이는 클라이언트 하나)이 있는 채와 없는 채로, 각각 세 번, 서버에서.
  `es loop collect` 8 에피소드도 같게. 표는 노트에 E4의 것 옆에. 재기 전까지는
  `Target / Status: unverified`.
* **여기 없는 것.** 노드별 활성값 통계(§23.3) — 낮춰진 모듈에 훅이 없다. 학습 중 롤아웃.
  트레이너를 멈추거나 조종하기. 프로토콜 변경.

## context

`cargo xtask check-scope`가 읽는 glob, 그리고 같은 범위를 산문으로:

```
crates/es/src/cmd/telemetry.rs
crates/es/src/cmd/eval.rs
crates/es/src/cmd/loop.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/cycle.rs
crates/es/src/cmd/mod.rs
crates/es/tests/cli.rs
crates/es-data/src/collect.rs
crates/es-data/src/training.rs
python/es/train_act.py
crates/es-editor/src/model/train_view.rs
crates/es-editor/src/model/live_run.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/launch.rs
crates/es-editor/src/model/labels.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/app.rs
crates/es-editor/i18n/en.toml
crates/es-editor/i18n/ko.toml
tests/golden/editor/launch-*.txt
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E7-live-loop.md
docs/packets/M7/E7-live-loop.ko.md
```

`cmd/telemetry.rs`(새 파일: `stage`를 가진 공유 `Publisher`), `eval.rs`(그것을 쓴다),
`loop.rs`(수집이 발행한다), `train.rs`(스트리밍 stdout, 스트림 5, 체크포인트, 샘플),
`cycle.rs`(모든 단계에 퍼블리셔 하나), `collect.rs`는 **수집기의 훅이 에피소드 결과를 넘겨야 할
때만**, `training.rs`는 **`--telemetry` 뒤의 새 계획 플래그 둘을 위해서만**,
`train_act.py`(`--progress-every`, `--sample-every`), 에디터의 `train_view.rs`(새 파일)와 접기,
`launch.rs`/`labels.rs`/i18n(칸), `app.rs`(배선), Launch 골든, 설계 노트 둘, 이 패킷.

## 오라클

1. `cargo test -p es --test cli train_telemetry_streams_the_curve` — 픽스처 베이크에서
   `es train --telemetry 127.0.0.1:<포트>`, 40 스텝, `--progress-every 10`: 구독한 클라이언트가
   단조 증가하는 스텝의 스트림 5 프레임 4개와 40에서의 `checkpoint` 이벤트 하나를 받는다.
   `training.lock`, 체크포인트, `metrics/loss.json`은 플래그 없는 실행과 바이트 단위로 같다.
   `ES_PYTHON`이 없으면 이름과 함께 `SKIP`.
2. `cargo test -p es --test cli collect_telemetry_publishes_every_episode` — `--frames
   --telemetry`로 전문가 에피소드 2개: 에피소드마다 `episode.begin`/`end`, 제어 틱마다 스트림 2
   프레임 하나, 스트림 4 이미지 최소 하나. 데이터셋 디렉터리는 플래그 없는 실행과 바이트 단위로
   같다.
3. `cargo test -p es --test cli cycle_telemetry_is_one_address` — `--telemetry`를 준
   `es loop cycle --dry-run`이 어느 단계 줄에도 주소를 보이지 않는다(그것은 사이클의 것이지
   단계의 것이 아니다). 살아 있는 실행(T2 오라클 4 모양, `#[ignore]`)은 한 소켓으로
   `collect`, `expert-gate`, `train`, `eval`의 `stage.begin`을 그 순서로 준다.
4. `cargo test -p es-editor train_view_folds_the_curve` — 각본대로 만든 스트림 5와 체크포인트
   이벤트가 트레이너의 `loss.json` 값과 같은 `Curve`(`f32` 비트 단위)를 주고, 표시가 제 스텝에
   있으며, `eta`가 단조다. `live_run_folds_collect_episodes_like_cells`도.
5. `cargo test -p es-editor launch_argv_is_the_golden`을 다시 만든 골든에 대해. i18n 완전성
   테스트도.
6. `cargo xtask ci`(레이어링: 에디터는 의존성을 얻지 않는다. `es-eval`은 그대로),
   `cargo xtask check-scope docs/packets/M7/E7-live-loop.md`.

## 합격

오라클 1–6 (1, 3의 살아 있는 절반, 게이트 9는 오라클 서버에서). 오케스트레이터가 로컬에서 전체
루프의 시작을 누르고(`target/plan-u/demo/cycle.toml`) 지켜본다: 수집 에피소드가 이미지와 함께
행으로 나타나고, Training 절이 내려가는 손실을 샘플 이미지와 나란히 그리고, 그다음 평가 행이
나타난다. 각 단계의 스크린샷은 워크트리의 `target/plan-u/e7/`에.
16절이 스트림 표(1–5), `stage` 필드, 트레이너의 두 플래그, 게이트 9 표를 기록한다.

## 금지

어느 단계든 계산하거나 쓰는 것을 바꾸는 일(바이트 동일성이 오라클 1–2의 주장이다), 어느 실행
경로에서든 블로킹 전송, 프로토콜/스키마 변경(`protocol.rs`는 고정), 플로팅 크레이트나 새 의존성,
`crates/es-eval/**`, `crates/es-telemetry/**`, `crates/es-safety/**`, `docs/ARCHITECTURE*.md`.
INV-17: 싱크는 클로저다. 규칙 3: `app.rs`에서 결정되는 것은 없다.
