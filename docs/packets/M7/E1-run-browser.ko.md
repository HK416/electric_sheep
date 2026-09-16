# M7 E1 — 실행 브라우저: 끝난 `es eval run` / `es loop collect`를 에디터에서 연다

스펙: §23.3(학습 중에 보이는 것: Safety Plane 이벤트 로그, 항별 분해, 청크/레이턴시
히스토그램), §10.5(실행의 아티팩트), §23.1(에디터는 클라이언트다),
§28.10 규칙 3(`app.rs`에서는 아무것도 결정되지 않는다). 확장할 설계 노트:
`docs/design/editor-shell.md`(+ `.ko.md`)에 새 10절. 선행: M5 V3/V4(`report.json`,
`events.json`, `es video mosaic`), V9(`traj/<cell>.estraj`), M4 에디터 stage 2.

## 질문

모든 실행은 `report.json`, `events.json`, `traj/*.estraj`와(`--frames`가 있으면)
`frames/<cell>/NNNNNN.bin`을 쓴다. 이것들의 유일한 독자는 `es video mosaic`와 `jq`를 든
사람이다. **에디터가 실행 디렉터리를 열어 무슨 일이 있었는지 — 셀별로, 틱별로 — 반대편에
아무 프로세스 없이 보여줄 수 있는가?**

## spec

**Run** 탭. `es-editor <run-dir>` 또는 File 필드가 `report.json`을 담은(그리고 선택적으로
`events.json`, `traj/`, `frames/`를 담은) 디렉터리를 연다; 번들 경로는 오늘처럼 여전히
그래프를 열며, 둘은 디스크에 무엇이 있는지로 구분된다.

헤드리스 뷰모델 `crates/es-editor/src/model/run_view.rs`:

* `RunView::open(dir) -> Result<RunView, RunError>`는 `EvaluationReport`(`es_ir::evaluation`)와
  이벤트 맵(`BTreeMap<String, Vec<es_eval::runner::StepEvent>>` — `es-editor`는 `es-eval`,
  layer 12에 의존해도 된다)을 읽고, `traj/<cell>.estraj`와 `frames/<cell>/`는 로드하지 않은
  채 목록만 만든다.
* `RunView::cells() -> Vec<CellRow>`: 셀 이름, 스위트, 보고서가 담고 있으면 시드, 결과
  (보고서의 셀별 메트릭에서: `success_rate`, `episode_length`, `envelope_violation_rate`와
  그 밖에 있는 것 무엇이든 — 모델은 메트릭 이름을 나열할 뿐 하드코딩하지 않는다), 궤적과
  프레임의 존재 여부.
* `RunView::timeline(cell) -> Timeline`: 틱마다 `EventSource`(Policy / Clamped / Fallback /
  Human)와 디코드된 `ViolationKind` 집합(`es_safety::EventSet::from_bits`나 그 동등물 —
  다시 유도하지 말고 디코드할 것), 그리고 종류별 합계와 각 종류의 첫 틱. 어떤 너비에서든
  그리기 위한 버킷 형태 `Timeline::buckets(n)`.
* `RunView::acceptance()`는 `report.acceptance`(rule, value, threshold, passed)를 그대로
  비춘다.
* `RunView::frame(cell, index) -> Option<Rgb8Image>`는 형제 파일 `layout.json`을 이용해
  기존 `Rgb8Image` 타입으로 `.bin` 하나를 로드한다; 요청된 것 이상은 아무것도 캐시하지
  않는다.
* 에러는 빠진 파일의 이름을 말한다; `report.json`만 있는 디렉터리도 여전히 열린다(타임라인과
  필름스트립은 비어 있고, 상태줄이 그렇게 말한다).

`app.rs`가 그리는 것: 셀 테이블(헤더 클릭으로 정렬 가능 — 정렬은 `RunView` 메서드이고
테스트된다), pass/fail 색이 입혀진 수용 행들, 그리고 선택된 셀에 대해 타임라인 스트립
(`EventSource`마다 색 하나, 그 아래 틱으로 표시되는 위반 종류)과 고르게 표본을 뽑은 최대
8장의 필름스트립. 셀을 선택하면 E2의 리플레이가 착륙해 있을 경우 그것을 위해서도 선택된다
(`selected_cell()` 접근자가 결합의 전부다).

## context (허용 범위)

`crates/es-editor/src/model/run_view.rs`(신규), `crates/es-editor/src/model/mod.rs`,
`crates/es-editor/src/lib.rs`(재수출), `crates/es-editor/src/app.rs`(Run 탭과 디렉터리로
여는 분기), `crates/es-editor/src/main.rs`, `crates/es-editor/Cargo.toml`(`es-eval`,
`es-safety`를 의존성으로 — 둘 다 워크스페이스 crate이고 외부 의존성 아님),
`tests/fixtures/visible-learning/run/`(작은 새 픽스처 실행: 셀 4개짜리 `report.json`,
위반 비트를 담은 `Clamped`/`Fallback` 틱 몇 개가 있는 `events.json`, 셀마다 `layout.json`이
딸린 96×96 프레임 하나 — `#[ignore]`가 붙은 생성기 테스트에서 실제 타입으로 생성해 바이트가
그 타입 자신의 것이 되게 한다), `docs/design/editor-shell*.md` 10절,
`docs/packets/M7/E1-run-browser*.md`.

## oracle

1. `cargo test -p es-editor run_view_reproduces_the_report` — 픽스처 실행에 대한
   `RunView::open`: 셀 행이 보고서가 담은 것과 같은 메트릭 값을 지니고(다시 파싱한 float이
   아니라 보고서 자신의 `MetricValue`로 비교), `acceptance()`가 `report.acceptance`와 같으며,
   `passed`가 일치한다.
2. `cargo test -p es-editor timeline_buckets_sum_to_the_events` — 모든 셀에 대해
   `timeline(cell)`의 종류별 합계가 `events.json`을 직접 훑어 얻은 카운트와 같고, `n`이
   `{1, 7, 64}`일 때 `buckets(n)`의 합도 같은 합계가 된다.
3. `cargo test -p es-editor a_report_alone_still_opens` — `report.json`만 있는 디렉터리가
   열린다; `timeline`은 비어 있고 `frame`은 `None`이며, 패닉이 없고, 상태가 무엇이 빠졌는지
   이름을 댄다.
4. `cargo test -p es-editor sorting_is_stable_and_total` — 각 열로 정렬하는 것은 행들의
   순열이고 동점에서 안정적이다.
5. `cargo build -p es-editor`가 탭을 컴파일한다; `cargo xtask ci` 통과; 레이어링 불변.

## acceptance

오라클 1–5 통과. 오케스트레이터가 서버(`~/artifacts/plan-v/v19b/`)에서 복사해 온 실제
실행에 대해 로컬 머신에서 열어 본다: 테이블, 수용 행, `Clamped` 틱이 보이는 타임라인 하나,
필름스트립이 나타난다. 설계 노트 10절이 그 탭, 뷰모델 API, 그리고 의도적으로 하지 않는 것
(살아있는 프로세스 없음, 실행 편집 없음)을 기록한다.

## forbidden

`crates/es-eval/**`, `crates/es-safety/**`, `crates/es-ir/**`(타입을 읽기만, 아무것도
바꾸지 않는다); 기존 네 탭의 동작; `app.rs`에서의 어떤 결정이든(정렬, 버킷화, 디코딩은 모두
모델에 산다); 새 외부 의존성; `docs/ARCHITECTURE*.md`; 기존의 모든 골든.
