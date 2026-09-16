# M7 E4 — 살아있는 텔레메트리: `es eval run --telemetry`가 발행하고, 에디터가 붙는다

스펙: §23.1(에디터는 실행 중인 프로세스의 클라이언트다; 프로토콜은 백엔드에 무관하다),
§23.3(실행 도중 무엇이 보이는가; 예산이 매겨진 샘플링; "텔레메트리 + 그래프 뷰로 인한 학습
저하 < 1%" — §28.7 게이트 9), §12.4(아홉 개의 지표, 결코 `step/s`가 아님), §25.1(소켓
위의 토큰). 설계 노트: `docs/design/telemetry-protocol.md`(와이어 형태와 TCP 루프백
트랜스포트 — 전부 읽을 것), `docs/design/editor-shell.md` 5절(Telemetry 탭의 모델) — 후자를
13절로 확장한다. 선행: M1 W8(`es_telemetry::transport::{Server, Client}`), M4 에디터
(`TelemetryModel`, `Source` 클로저), E1(Run 탭; 살아있는 실행과 끝난 실행이 같아 보여야
한다).

## the question

`es_telemetry`에는 서버, 클라이언트, 스키마, 테스트 스위트가 있지만 **생산자가 없다**:
`es` 안의 아무것도 프레임을 발행하지 않고, 에디터의 `Source`는 저장된 리플레이일 뿐이다.
**평가가 자신이 아는 것 — 틱별 `StepEvent`, 셀의 진행, 아홉 개의 지표 — 를, 계산하는 것을
바꾸지 않고 발행할 수 있고, 에디터가 실행되는 동안 그것을 보여줄 수 있는가?**

## spec

* **생산자.** `es eval run --telemetry <addr> [--telemetry-token <t>]`는 첫 셀 전에
  `es_telemetry::transport::Server`를 바인딩하고 다음을 발행한다:
  - 스트림 `1` `Event { kind: "cell.begin" | "cell.end", fields: { cell, suite, seed,
    outcome … } }`;
  - 스트림 `2` 제어 틱마다의 `Scalars([tick, source as f64, violation bits as f64])` —
    틱마다 프레임 하나가 §23.3의 예산 문제다: 모든 틱을 발행하되 서버의 기존
    논블로킹 `publish`를 통해서다(느린 클라이언트는 프레임을 떨어뜨리며, 실행은 결코
    기다리지 않는다 — 루프백 테스트가 그것을 단언한다);
  - 스트림 `3` 각 `cell.end`에서의 `Metrics(PerfMetrics)`, 실행이 정직하게 채울 수 있는
    필드로(`actions_per_sec`, `policy_inferences_per_sec`, 플레인의 카운터로부터의
    `chunk_underrun_rate`, 측정되었을 때만 `p50/p95_end_to_end_latency` — 아니면 `None`,
    결코 0이 아님);
  - 스트림 `4` 최대 `--telemetry-image-every N` 틱마다(기본값 `0` = 안 함)의 **현재
    관측 프레임**의 `Image`, 렌더러가 쓴 그대로의 `Rgb8` 바이트 — §23.3의 "실제 관측
    이미지", 스펙이 말하듯 별도로 속도 제한된다.
  훅은 `es_eval::RunConfig` 안의 `Option<&dyn FnMut(Frame)>` 모양 싱크 하나다(또는 CLI가
  비우는 채널) — `es-eval`은 `es-telemetry`의 트랜스포트에 의존해서는 안 된다(둘 다 layer
  10; 같은 레이어 의존 금지: *프레임 구성*은 `crates/es/src/cmd/eval.rs`에 두고,
  `es-eval`에는 `StepEvent`와 셀 컨텍스트를 담는 범용 틱별 콜백만 준다). 플래그가 없을
  때는 동작 변화가 0이다: `report.json`과 `events.json`이 바이트 단위로 같다.
* **소비자.** `es-editor --attach <addr> [--token <t>]`(그리고 Telemetry 탭 안의 필드 +
  Connect 버튼): `Client::connect` + `subscribe`가 `try_recv`를 통해 `Source` 클로저가
  된다; `TelemetryModel`은 스트림 1–4를 E1의 `RunView`가 쓰는 것과 같은
  `CellRow`/`Timeline` 모양으로 접는 `LiveRun` 뷰모델(`live_run.rs`)을 얻어서, Run 탭이
  살아있는 실행과 끝난 실행을 같은 코드로 그린다 — 설계 노트가 공유 타입을 명시한다.
* **게이트 9 측정.** 명목 픽스처(`--jobs 1`)에서 `--telemetry`가 있을 때와 없을 때(비우는
  클라이언트 하나가 붙은 채)의 `es eval run`: 둘의 벽시계 시간, 각각 세 번 실행, `< 1%`
  목표 옆에 관측으로 보고한다(서버 수치가 나오기 전까지는 `Target / Status: unverified`).

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-eval/src/runner.rs
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
crates/es-telemetry/src/transport.rs
crates/es-editor/src/model/live_run.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/design/telemetry-protocol.md
docs/design/telemetry-protocol.ko.md
docs/packets/M7/E4-live-telemetry.md
docs/packets/M7/E4-live-telemetry.ko.md
```

`crates/es-eval/src/runner.rs`(`RunConfig` 안의 틱별 콜백; 그 외에는 아무것도),
`crates/es/src/cmd/eval.rs`(플래그, 프레임 구성, 서버 수명), `crates/es/tests/cli.rs`(새
`eval_telemetry_*` 테스트), `crates/es-telemetry/src/transport.rs`는 **오직** 논블로킹
보장에 테스트에서 보이는 카운터(떨어진 프레임)가 필요할 때만 — 프로토콜 변경 없음,
`crates/es-editor/src/model/live_run.rs`(신규), `model/telemetry_view.rs`, `app.rs`,
`main.rs`, `Cargo.toml`(외부 추가 없음), `docs/design/editor-shell*.md` 13절,
`docs/design/telemetry-protocol*.md`(네 스트림을 이름 붙이는 "생산자" 하위 절),
`docs/packets/M7/E4-live-telemetry*.md`.

## oracle

1. `cargo test -p es --test cli eval_telemetry_publishes_every_tick_in_order` — 루프백: 작은
   픽스처의 평가를 `--telemetry 127.0.0.1:0`으로(바인딩된 포트를 출력), 스트림 1–3을
   구독한 테스트 클라이언트가 모든 셀에 대한 `cell.begin`/`cell.end`와 각 셀 안에서
   엄격히 증가하는 틱을 가진 스트림-2 프레임 하나를 틱마다 받는다; 실행의 `report.json`과
   `events.json`은 플래그 없는 실행과 바이트 단위로 같다.
2. `cargo test -p es --test cli eval_telemetry_never_blocks_the_run` — 접속하고 결코 읽지
   않는 클라이언트: 실행은 끝나고, 그 보고서는 불변이며, 서버의 `stats()`가 떨어진 프레임
   > 0을 보여준다.
3. `cargo test -p es-editor live_run_folds_streams_into_run_rows` — 스크립트화된
   `Vec<Message>`를 `Source`를 통해 흘리면, 같은 실행을 디스크에 쓴 것에 대해
   `RunView::open`이 주는 것과 같은 `CellRow`와 `Timeline`이 나온다(E1의 픽스처를
   메시지로 재생).
4. `cargo test -p es-telemetry`가 불변이고 통과한다(S-4의 플레이크는 범위 밖이다: 발생하면
   재실행하고 그렇게 말할 것).
5. `cargo build -p es-editor`; `cargo xtask ci`(레이어링: `es-eval`은 새 의존성을 얻지
   않는다; `es-editor`는 무엇에든 의존해도 된다); `cargo xtask check-scope docs/packets/M7/E4-live-telemetry.md`.

## acceptance

오라클 1–5. 오케스트레이터가 로컬에서 픽스처에 `es eval run --telemetry 127.0.0.1:7777`을,
그리고 `es-editor --attach 127.0.0.1:7777`을 실행해 셀들이 나타나는 것을 지켜본다.
게이트-9 표는 설계 노트 13절 안의 서버 위에 있다.

## forbidden

평가가 계산하거나 쓰는 것을 바꾸는 것; 같은 레이어 의존 `es-eval → es-telemetry`; 실행
경로 어디에서든의 블로킹 send; 프로토콜/스키마 변경(`protocol.rs`는 그 버전에서 고정되어
있다; 새 스트림 *id*는 스키마가 아니라 데이터다); `crates/es-safety/**`;
`docs/ARCHITECTURE*.md`; 골든. INV-17: 새 트레이트 없음(싱크는 클로저 타입이다).
