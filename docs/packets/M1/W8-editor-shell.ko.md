<!-- Korean translation of docs/packets/M1/W8-editor-shell.md. The English file is the working copy; regenerate this when it changes. -->
# W8 — 에디터 셸 (`es-editor`: 탭, 읽기 전용 계층 그래프, 텔레메트리, 전/후 이미지)

스펙: §23.1(에디터는 실행 중인 프로세스의 클라이언트다), §23.2(계층 그래프 뷰), §23.3(학습
중 확인: 전처리 전/후 이미지, 이벤트 로그, 노드별 통계), §23.4(그래프 기능 3단계 — 읽기
전용이 M1), §14.3(`.eslayout` 사이드카, 절대 IR 안에는 없음), §12.4(메트릭 세트, 절대 단일
`step/s`가 아님), §1.9 cut 5(*편집 가능한* 비주얼 그래프는 잘라낼 수 있음; 읽기 전용은
남긴다), §4.2 규칙 4(아무것도 `es-editor`에 의존하지 않음)와 규칙 7(IR에 레이아웃 없음),
§28.3 W8. 설계 노트: `docs/design/editor-shell.md`.

## context (범위)

```
crates/es-editor/Cargo.toml
crates/es-editor/src/lib.rs
crates/es-editor/src/main.rs
crates/es-editor/src/app.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/model/graph_view.rs
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/image_view.rs
crates/es-editor/tests/common/mod.rs
crates/es-editor/tests/graph_view.rs
Cargo.toml                       (workspace deps only: es-editor, eframe, egui)
docs/design/editor-shell.md
docs/packets/M1/W8-editor-shell.md
```

## spec (사양)

- **새 크레이트 `es-editor`, 계층 12.** 무엇에든 의존할 수 있지만; 아무것도 이것에 의존할
  수 없다(§4.2 규칙 4, `cargo xtask layering`로 강제되며 이미 그 행을 갖고 있다).
- **분리: 헤드리스 뷰모델(테스트됨) + 얇은 egui 레이어(컴파일됨).** CI에는 디스플레이가
  없으므로 모든 결정은 `src/model/**`에 있고 `cargo test`로 판정되며; `src/app.rs`는 위치를
  사각형으로 바꿀 뿐이고 `cargo build`로 판정된다.
- **`model::graph_view`.** `LayeredGraph::from_bundle(task, observation, learning,
  deployment)` → `NodeView { id, kind, label, ports, layout }`로 이루어진 네 개의 `LayerView`
  (Task, Observation, Learning, Deployment)와 IR 고유의 엣지; `es_ir::cross`가 검사하는 조인을
  그대로 반영하는 `cross_edges`(Task `ObservationSpec` → Observation 소스, Observation 출력 →
  Learning 계약 입력, Learning 액션 → 배포된 액션 계약 — *규칙*은 `es_ir::cross`에 그대로
  남고, 이것은 같은 짝짓기를 그릴 뿐이다); `diagnostics` = 각 IR의 `validate()` +
  `cross::check`. Deployment IR은 그래프가 아니라 레코드이므로(§9.2), 그 밴드는 네 개의 합성
  노드다: `ActionContract → SafetyEnvelope → Watchdogs → Fallback`. `auto_layout`은 결정적인
  Sugiyama-lite다(최장 경로 순위 → `x`, 순위 내 정준 `NodeId` 순서 → `y`, 계층당 하나의
  밴드); `apply_layout(&es_ir::serial::Layout)`은 `.eslayout` 사이드카에서 이를 덮어쓴다.
  **레이아웃은 읽히기만 하고 절대 IR에 써넣어지지 않는다**(§4.2 규칙 7, §14.3).
- **`model::telemetry_view`.** `TelemetryModel`은 `Box<dyn FnMut() -> Option<Message>>` 소스로
  부터 `es_telemetry::protocol::Message`를 소비하므로, `es_telemetry::transport`(다른 패킷,
  진행 중)는 이 크레이트를 건드리지 않고 나중에 꽂힌다. `(stream, component)`별로 상한이 있는
  스칼라 히스토리, 최신 `PerfMetrics`, 이벤트 로그를 유지한다; `metric_rows()`는 §12.4 세트를
  고정된 순서로 내놓으며(지연은 p50 + p95), 측정되지 않은 메트릭은 `None`으로 남는다.
- **`model::image_view`.** `BeforeAfter::run(obs, input)`은 Observation IR을
  `es_compile::CpuPlan`으로 컴파일하고 — 이는 로어링이 판정받는 것과 같은 CPU 레퍼런스이지
  (§11.3), 별개의 전처리 구현이 결코 아니다 — 한 프레임에 대해 실행하고 원본 입력을 각 이미지
  출력과 `Rgb8Image`로 짝짓는다. `BeforeAfter::sample(obs)`는 프레임을 갖지 않는 번들을 위해
  결정적 그래디언트를 합성한다.
- **`app` + `main`.** `eframe` 앱, 탭은 `Graph | Telemetry | Images | Diagnostics`, 번들 경로용
  텍스트 필드(`es_compile::PolicyBundle::open`으로 여는 `.esb` 파일, 또는 번들 포맷이 고정한
  이름의 IR별 TOML 파일들이 든 디렉터리). Graph 탭은 뷰모델 위치를 바탕으로 계층 밴드마다
  노드 사각형을 3차 베지어 엣지와 함께 그리고, 팬·줌을 지원하며, **읽기 전용**이다 — 드래그가
  없고, 위치를 IR에 되써넣는 코드 경로도 없다. `es-editor [bundle.esb]`는 시작 시 그것을 연다.
- **`egui-snarl` 없음.** §23.4가 그것을 지정하지만, 읽기 전용 뷰에는 `egui::Painter`만 있으면
  된다; snarl은 대화형 배선을 위해 존재한다. 업그레이드 경로는 설계 노트에 기록되어 있다:
  `LayeredGraph`는 그것을 도입해도 바뀌지 않고, `app.rs`의 페인팅만 교체된다.

제약: 영어만, `BTreeMap`만, 새 trait 없음(INV-17), 소스 라인 ~1500줄 이하, 컴파일 시간을
낮게 유지하는 최소한의 egui 기능, 루트 `Cargo.toml`은 세 개의 워크스페이스 의존성을
추가하는 용도로만 건드림.

## oracle (오라클)

```
cargo fmt -p es-editor --check
cargo clippy -p es-editor --all-targets -- -D warnings
cargo test -p es-editor
cargo build -p es-editor
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- `cargo build -p es-editor`는 디스플레이가 없는 CI 머신에서 성공하고, 어떤 테스트도 창을
  열지 않는다.
- `es-ir` 자체의 cross-IR 픽스처로부터(`es` CLI 테스트가 이미 하듯 `tests/common/mod.rs`에
  복사됨): 각 IR의 노드 수를 가진 네 개의 계층과 네 개의 합성 Deployment 노드; 다섯 개의
  cross-IR 엣지(`rgb_front`와 `joint_state`의 Task→Observation과 Observation→Learning,
  `actions`의 Learning→Deployment), 모든 끝점이 자신의 계층에 존재하는 노드를 가리킴.
- `auto_layout`은 결정적이고(같은 번들에 대한 두 번의 실행이 동일함), 모든 노드를 배치하며,
  데이터플로에 따라 순위를 매기고(`ImageInput < Resize < Normalize`, 두 소스가 순위 0을
  공유), 같은 계층의 두 노드에 같은 위치를 절대 주지 않으며, 각 계층을 자신의 밴드 안에
  유지한다.
- `apply_layout`은 사이드카가 지정한 노드만 정확히 덮어쓰고 나머지는 그대로 둔다.
- 배포된 계약과 어긋나는 `action_dim`을 가진 번들은 `diagnostics`에서 `XIR-020`을 드러내고;
  건드리지 않은 픽스처는 아무것도 드러내지 않는다.
- 미리 준비된 텔레메트리 프레임이 표를 채운다: 컴포넌트별 스칼라 시리즈, 필드를 갖춘 이벤트
  로그, `HelloAck`로부터의 `execution_hash`, §12.4의 열 개 행 중 측정되지 않은 것은 `None`;
  히스토리는 오래된 것부터 상한이 걸리고 `pump`는 예산을 지킨다.
- `ImageInput → Dequantize → Resize(4×3)`를 통과한 8×6 그래디언트는 하나의 `ImagePair`를
  낳는다: `before`는 8×6이고 입력 프레임과 바이트 단위로 동일하며, `after`는 4×3이다.

## forbidden (금지)

- `crates/es-telemetry/src/transport.rs`, `crates/es-core/src/ring.rs`,
  `crates/es/src/cmd/backend.rs`, 그리고 문서 번역 — 다른 패킷들의 범위. 이 크레이트는 아직
  트랜스포트에 의존해서는 안 된다: 메시지 소스는 클로저로 남는다.
- 어떤 IR 타입에든 레이아웃을 써넣거나, `es-ir`에 위치/UI 필드를 추가하는 것(§4.2 규칙 7).
- `es_ir::cross`의 cross-IR 규칙이나 `es_compile`의 전처리 커널을 재구현하는 것 — 짝짓기를
  반영하고, 플랜을 실행하라.
- 어떤 형태의 그래프 편집이든(§23.4 2단계, M3), `egui-snarl`, 네이티브 파일 대화상자
  크레이트, 그 밖의 새로운 외부 의존성, 그리고 세 개의 워크스페이스 의존성 항목을 넘어서는
  루트 `Cargo.toml`에 대한 어떤 변경도 금지.
- 커밋하는 것.
