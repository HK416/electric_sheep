# M7 E7 — 전체 루프, 실시간으로: 수집과 학습이 발행하고, 에디터가 손실 곡선과 네트워크가 보는 것을 그린다

스펙: §23.1(에디터는 실행 중인 프로세스의 클라이언트다), §23.3("실행 도중 무엇이 보이는가": 실제
관측 이미지, 노드별 통계, 학습 곡선; 예산이 있는 샘플링; 이미지는 별도로 속도 제한됨; 게이트
9 < 1%), §13.1(수집 → 학습 → 평가 → 관찰이 하나의 워크플로다), §12.4(아홉 개의 메트릭, 결코
`step/s`가 아니다), §19.3(`metrics.parquet`이 학습 곡선의 정체성 슬롯이다), §28.10 규칙 3. 로컬
사이클 데모 이후의 오너 노트 2026-09-21: *에디터 안에 손실 곡선 같은 데이터를 보여 달라*와
*학습 프로세스에는 렌더링된 뷰가 없다 — 그것은 아쉬운 일이다.* 설계 노트:
`docs/design/telemetry-protocol.md`(E4의 "Producers" 절이 프로듀서 둘과 스트림 하나를 얻는다),
`docs/design/editor-shell.md` 16절. **E4**(`RunSink`/`Publisher` 패턴, 스트림 1–4, `--attach`),
**E5**(실행 패널의 `--telemetry` 필드), **T1/T2**(`es train`, `es loop cycle`), **T6**(증강된 학습
텐서가 네트워크가 보는 것이다)에 의존한다.

## 질문

E4 이후 `es eval run`만이 발행한다. *전체 루프*에서 Start를 누르는 사람은 로그 줄이 10분 동안
스크롤되는 것을 보며 그림도 곡선도 보지 못한다. **수집이 평가가 하는 것처럼 자신의 에피소드를
발행할 수 있는가, 학습이 자신의 손실, 학습률, 처리량, 그리고 그것이 맞추고 있는 바로 그
텐서를 — `N` 스텝마다, < 1% 비용으로 — 발행할 수 있는가, 그리고 하나의 주소가 사이클의 네
단계 전부를 실어 날라 에디터가 루프를 하나의 라이브 실행으로 보여줄 수 있는가?**

## 사양

* **하나의 `Publisher`, 공유됨.** E4의 `Publisher`(프레임 구성, 논블로킹 `Server::publish`)가
  `crates/es/src/cmd/eval.rs`에서 `crates/es/src/cmd/telemetry.rs`로 옮겨가고, 모든 스트림-1
  이벤트에 넣는 `stage: &'static str` 필드를 얻는다(`"collect"`, `"expert-gate"`, `"train"`,
  `"eval"`, `"showcase"`), 그래서 구독자는 하나의 소켓에서 사이클의 단계들을 구별할 수 있다.
  `es loop cycle --telemetry <addr> [--telemetry-token] [--telemetry-image-every N]`는 첫 단계
  전에 **한 번** 바인드하고 같은 퍼블리셔를 프로세스 내의 모든 단계에 건네준다; 독립 실행
  명령들은 `es eval run`이 하는 것처럼 자신의 것을 바인드한다. 스트림-1 `stage.begin` /
  `stage.end` 이벤트가 각 단계를 그 벽시계 시간과 함께 괄호로 묶는다.
* **수집이 발행한다.** `es loop collect --telemetry <addr>`는 수집기의 기존 틱별 훅(`FrameSink`,
  개입자)을 통해 발행하며 **그것이 쓰는 것에는 아무 변화가 없다**: 스트림 1 `episode.begin` /
  `episode.end { episode, seed, outcome, steps }`, 스트림 2 `es eval run`이 하는 것과 정확히
  똑같이 제어 틱마다 하나의 `[frame, tick, source, violation bits]`(플레인 자신의 판정), 스트림
  4는 `--telemetry-image-every` 틱마다의 관측 이미지. `--frames` 없는 수집은 이미지를 발행하지
  않으며 그것을 한 번 말한다. 오라클: `--out` 아래의 데이터셋은 플래그가 있든 없든 바이트
  단위로 동일하다.
* **학습이 발행한다.** `python/es/train_act.py --progress-every N`은 `N` 옵티마이저 스텝마다,
  최종 요약 줄(이것은 마지막 줄로 남고 바이트 단위로 동일하게 남는다) **앞에** stdout으로
  JSON 한 줄 `{"progress": {"step", "loss", "lr", "samples_per_s", "elapsed_s"}}`을 출력한다;
  `--sample-every N`은 `metrics/sample-<step>.bin` + `.json`(`{"shape":[h,w,3],"port":…}`)을
  쓰는데, 여기에는 증강 이후 현재 배치의 이미지 입력 하나가 센서의 색 공간에서 **`Rgb8`로**
  — 네트워크가 맞추고 있는 바로 그 텐서를, 표시를 위해서만 비정규화한 것 — 담기고,
  `{"sample": "<path>"}`을 출력한다. `es train --telemetry <addr>`는 트레이너의 stdout을
  스트리밍 읽기(`Stdio::piped` + `BufReader::lines`, 요약은 여전히 마지막 줄)로 전환하고
  발행한다: 진행 줄마다 스트림 **5** `Scalars([step, loss, lr, samples_per_s])`(새 스트림
  id는 스키마가 아니라 데이터다 — 노트의 표가 행을 하나 얻는다), 마크가 패킹될 때 스트림
  1 `checkpoint { step, policy_hash }`, 끝에서 실행이 정직하게 채울 수 있는 §12.4 필드를
  가진 스트림 3 `Metrics`(`training_samples_per_sec`; 나머지는 전부 `None`), 스트림 4 샘플
  이미지. 플래그가 없으면 `train_act.py`는 `--progress-every`도 `--sample-every`도 받지
  않으므로, 측정된 모든 실행의 플랜과 `training.lock`은 움직이지 않는다(`plan-ir.txt` 골든
  불변).
* **에디터가 그것을 그린다.** `model/train_view.rs`: `TrainView`가 스트림 5를 `Curve { step:
  Vec<u32>, loss: Vec<f32>, lr: Vec<f32> }`로 접어 넣고, `checkpoints: Vec<(u32, String)>`,
  마지막 샘플 이미지와 `throughput`을 유지하며, `eta(total_steps)`에 답한다. `LiveRun`(E4)이
  단계 스트립을 얻고, eval의 `cell.*`을 접어 넣는 방식 그대로 수집의 `episode.*`를 접어
  넣는다, 그래서 사이클의 수집 에피소드도 Run 탭의 행이 된다. **Live** 탭(관찰)은 *Training*
  섹션을 얻는다: `ui.painter()`를 통한 폴리라인으로 그려지는 손실 곡선(로그 스케일 토글)과
  lr 곡선 — **플로팅 의존성 없음** — 체크포인트 마크, 현재 스텝 / 전체 / ETA, 그리고 실행의
  카메라 이미지 옆의 샘플 이미지와 함께; Run 탭은 표 위에 단계 스트립을 보여준다. `app.rs`가
  배선한다; 모든 숫자와 레이블은 모델과 i18n 표(E6)에서 온다.
* **실행 패널.** `es train`과 `es loop cycle` 종류가 `--telemetry`, 토큰, `--telemetry-image-every`
  필드를 얻는다(E5의 `LaunchField`들이 공유된다), 그래서 전체 루프에서 Start는 오늘 평가가
  하는 것처럼 스스로 붙는다(attach); 세 개의 `launch-*.txt` 골든이 한 번 재생성된다(픽스처
  입력의 argv는 두 종류에서 새로운 기본값 `--telemetry`에 의해서만 바뀐다 — 노트에 그렇게
  말하라; 골든은 그 외에는 읽기 전용이다).
* **게이트 9.** `es train`을 픽스처 베이크에서 `--telemetry --progress-every 10 --sample-every
  100`(드레이닝 클라이언트 하나)이 있는 경우와 없는 경우로 2,000스텝, 각각 세 번, 서버에서;
  `es loop collect` 8 에피소드도 마찬가지. 노트의 표는 E4의 것 옆에; 측정 전까지는 `Target /
  Status: unverified`.
* **여기 없음.** 노드별 활성화 통계(§23.3) — 로워링된 모듈에는 그것을 위한 훅이 없다; 학습
  도중의 롤아웃; 트레이너를 멈추거나 조종하는 것; 프로토콜 변경.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

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

`cmd/telemetry.rs`(신규: `stage`를 가진 공유 `Publisher`), `eval.rs`(그것을 사용), `loop.rs`(수집이
발행), `train.rs`(스트리밍되는 트레이너 stdout, 스트림 5, 체크포인트, 샘플), `cycle.rs`(모든
단계를 위한 퍼블리셔 하나), `collect.rs`는 **오직** 수집기의 훅이 에피소드 결과를 통과시켜야
할 때만, `training.rs`는 **오직** `--telemetry` 뒤의 두 새 플랜 플래그를 위해서만,
`train_act.py`(`--progress-every`, `--sample-every`), 에디터의 `train_view.rs`(신규)와 그
폴드들, `launch.rs`/`labels.rs`/i18n(필드들), `app.rs`(배선), 실행 골든들, 두 설계 노트, 이
패킷.

## 오라클

1. `cargo test -p es --test cli train_telemetry_streams_the_curve` — 픽스처 베이크에서 `es
   train --telemetry 127.0.0.1:<port>`, 40스텝, `--progress-every 10`: 구독한 클라이언트가
   스텝이 엄격히 증가하는 4개의 스트림-5 프레임과 40에서의 `checkpoint` 이벤트 하나를 받는다;
   `training.lock`, 체크포인트, `metrics/loss.json`은 플래그 없는 실행과 바이트 단위로
   동일하다. `ES_PYTHON` 없이는 이름으로 `SKIP`.
2. `cargo test -p es --test cli collect_telemetry_publishes_every_episode` — `--frames
   --telemetry`로 전문가 에피소드 2개: 에피소드마다 `episode.begin`/`end`, 틱마다 스트림-2
   프레임 하나, 최소 스트림-4 이미지 하나; 데이터셋 디렉터리는 플래그 없는 실행과 바이트
   단위로 동일하다.
3. `cargo test -p es --test cli cycle_telemetry_is_one_address` — `--telemetry`가 있는 `es loop
   cycle --dry-run`은 어떤 단계 줄에도 없는 주소를 보여준다(그것은 단계의 것이 아니라
   사이클의 것이다); 라이브 실행(T2 모양의 오라클 4, `#[ignore]`)은 하나의 소켓에서
   `collect`, `expert-gate`, `train`, `eval`에 대한 `stage.begin`을 그 순서대로 전달한다.
4. `cargo test -p es-editor train_view_folds_the_curve` — 스크립팅된 스트림 5 + 체크포인트
   이벤트가 트레이너의 `loss.json` 값과 (비트 단위 `f32`로) 같은 `Curve`를 올바른 스텝의
   마크들과 함께 내놓는다, `eta`는 단조적이다; `live_run_folds_collect_episodes_like_cells`.
5. 재생성된 골든에 대한 `cargo test -p es-editor launch_argv_is_the_golden`; i18n 완전성
   테스트.
6. `cargo xtask ci`(레이어링: 에디터는 새 의존성을 얻지 않는다; `es-eval` 불변); `cargo xtask
   check-scope docs/packets/M7/E7-live-loop.md`.

## 수용 기준

오라클 1–6(1, 3의 라이브 절반과 게이트 9는 오라클 서버에서). 오케스트레이터가 로컬에서
(`target/plan-u/demo/cycle.toml`) 전체 루프에서 Start를 누르고 지켜본다: 수집 에피소드가
이미지와 함께 행으로 나타나고, Training 섹션이 옆의 샘플 이미지와 함께 떨어지는 손실을
그리고, 그다음 평가 행들이 나타난다. 각 단계의 스크린샷은 워크트리의 `target/plan-u/e7/`
아래에. 16절이 스트림 표(1–5), `stage` 필드, 트레이너의 두 플래그와 게이트-9 표를 기록한다.

## 금지

어떤 단계가 계산하거나 쓰는 것을 바꾸는 것(바이트 동일성이 오라클 1–2의 단언이다); 어떤 실행
경로에서든 블로킹 전송; 프로토콜/스키마 변경(`protocol.rs` 동결); 플로팅 크레이트나 그 어떤
새 의존성; `crates/es-eval/**`, `crates/es-telemetry/**`, `crates/es-safety/**`;
`docs/ARCHITECTURE*.md`. INV-17: 싱크는 클로저다. 규칙 3: `app.rs`에서는 아무것도 결정되지
않는다.
