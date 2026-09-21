<!-- Korean translation of docs/design/editor-shell.md. The English file is the working copy; regenerate this when it changes. -->
# 에디터 셸 — 읽기 전용 계층 그래프, 텔레메트리, 전/후 이미지

`crates/es-editor`(계층 12)를 위한 설계 노트. 스펙: §23(에디터와 통합 디버거),
§23.2(계층 그래프 뷰), §23.3(학습 중 확인), §23.4(그래프 기능 3단계 — 읽기 전용이 M1),
§14.3(`.eslayout` 사이드카), §1.9(cut 5: *편집 가능한* 비주얼 그래프는 잘라내고, 읽기 전용은
남긴다), §12.4(메트릭 세트).

## 1. 이 크레이트가 하는 일

**클라이언트**다(§23.1). 학습을 호스팅하지 않고, 시뮬레이션을 소유하지 않으며, IR 권한도
갖지 않는다: 번들을 열고, 각 IR에 스스로 검증하라고 요청하고, 지시받은 대로 그린다.
§4.2 규칙 4가 이를 구조적으로 강제한다 — 아무것도 `es-editor`에 의존할 수 없으므로,
어떤 것도 뷰에 대한 의존성을 키울 수 없다.

§23.4의 1단계만: 읽기 전용. 편집에는 레이아웃 영속화, undo/redo, 검색, 대형 그래프 성능이
필요하지만 읽기 전용에는 그중 아무것도 필요 없고, §23.4는 디버깅 가치의 대부분이 이미
거기 있다고 말한다. 이는 또한 §1.9 cut 5에서 살아남는 절반이기도 하다.

## 2. 분리: 뷰모델 / egui

| 절반 | 위치 | 테스트 |
|---|---|---|
| 뷰모델 | `src/model/{graph_view,telemetry_view,image_view}.rs` | 완전히, 헤드리스로 |
| egui 셸 | `src/app.rs`, `src/main.rs` | 컴파일만 |

CI에는 디스플레이가 없으므로 무언가를 *결정*하는 모든 것은 `model`에 있고
`cargo test -p es-editor`로 판정되며, `app.rs`는 위치를 사각형으로 바꿀 뿐이고
`cargo build -p es-editor`로 판정된다. 결정을 `app.rs` 밖에 두는 것은 취향이 아니라
규칙이다: 테스트되지 않는 파일은 얇고 그 이상 아무것도 아닌 것만 허용된다.

## 3. `egui-snarl` 없음

§23.4는 노드 그래프에 `egui-snarl`을 지정한다. **읽기 전용** 뷰에는 그것이 필요 없다:
snarl은 대화형 배선 — 드래그, 연결, 연결 해제, 핀 히트 테스트, 편집 시점의 노드 모델 —
을 위해 존재하고, 1단계는 `egui`가 이미 제공하는 `egui::Painter`로 사각형과 3차 베지어
곡선을 그린다. 의존성 하나가 줄고, `es_ir::Graph`와 `LayeredGraph` 외에 노드가 무엇인지에
대한 세 번째 개념도 생기지 않는다.

§23.4 2단계(M3, 편집)를 위한 업그레이드 경로: `LayeredGraph`는 그대로 남고 — snarl은
`app.rs`의 약 120줄짜리 페인팅 코드를 대체하며, 자신의 노드 모델로 `NodeView`/`Ports`를,
위치 저장소로 `Layout`을 사용하게 된다. 뷰모델의 어떤 것도 페인터의 부재로 형태가 잡혀
있지 않으므로, snarl 도입은 파일 하나만 바꾸는 변경이다.

## 4. 계층 그래프 (§23.2)

`LayeredGraph::from_bundle(task, observation, learning, deployment)`는 그 순서대로 네 개의
누적된 `LayerView`를 만들어내며, 각각은 자신의 노드(id, IR 고유의 `IrNode::kind` 태그,
라벨, 포트 이름)와 `es_ir::Graph`에서 그대로 가져온 엣지를 갖는다.

Deployment IR은 그래프가 아니라 레코드이므로(§9.2), 그 밴드는 실행 순서대로 된 네 개의
합성 노드다 — `ActionContract → SafetyEnvelope → Watchdogs → Fallback` — 이것이 §23.2
그림의 액션/안전 부분이다.

**Cross-IR 엣지는 `es_ir::cross`를 그대로 반영한다.** 규칙은 거기 그대로 남고, 이 뷰는
`cross::check`가 진단하는 것과 같은 조인을 그릴 뿐이다:

| 엣지 | 조인 기준 | 규칙 |
|---|---|---|
| Task `ObservationSpec` → Observation 소스 | 센서/바디 id, 또는 `Language` | §7.4 (`XIR-001/002`) |
| Observation 출력 → Learning 계약 입력 | 텐서 이름 | §8.4 (`XIR-010`) |
| Learning 액션 출력 → 배포된 액션 계약 | 포트 이름 | §8.5, §9.2 (`XIR-020..022`) |

`diagnostics`는 각 IR 고유의 `validate()`와 `cross::check`를 합친 것이므로, 그림과 불평
목록이 같은 한 번의 패스에서 나온다. 검증에 실패하는 번들이야말로 누군가 에디터를 열어
들여다보려는 바로 그 대상이므로, `from_bundle`은 절대 실패하지 않는다.

### 레이아웃

`auto_layout`은 Sugiyama-lite다: 최장 경로 순위 → `x`, 그래프의 정준(canonical) `NodeId`
순서에 따른 순위 내 위치 → `y`, 계층당 하나의 수평 밴드. 결정적이며(같은 번들은 항상 같은
레이아웃이 되고, 이는 테스트 항목이다) 겹치지 않는다. 교차 감소는 생략한다 — 그것은 이런
레이아웃을 *보기 좋게* 만드는 것이면서, 동시에 결정성을 유지하려면 신중하고 안정적인
타이브레이크가 필요한 부분이기도 하다.

`apply_layout(&Layout)`은 `.eslayout` 사이드카(§14.3)에서 위치를 덮어쓴다. **에디터는
레이아웃을 읽기만 하고 절대 IR에 써넣지 않는다**(§4.2 규칙 7): `Layout`은
`es_ir::serial`에 있고, 어떤 IR 타입도 위치 필드를 갖지 않으며, IR의 `*_hash`는 위치를
볼 수 없다. 사이드카는 `NodeId` 하나만으로 키가 매겨지는데 네 IR이 이를 위한 네임스페이스를
공유하지 않으므로, 위치는 모든 계층에서 그 id에 적용된다 — IR별 사이드카는 *저장*이
중요해지는 시점인 2단계의 문제다.

## 5. 텔레메트리 (§23.3)

`TelemetryModel`은 `Box<dyn FnMut() -> Option<Message>>`로부터 `es_telemetry::protocol::Message`를
소비한다. 그 간접화가 핵심이다: 트랜스포트는 다른 패킷(`es_telemetry::transport`)의
소관이고, 그것을 꽂는 일은 `main.rs`에서 한 줄이면 된다.

보관하는 것: `(stream, component)`별로 상한이 있는 `(tick, value)` 히스토리, 최신
`PerfMetrics`, 그리고 이벤트 로그. 히스토리는 `es_core::ring`이 아니라 상한이 있는 평범한
`Vec`이다 — 뷰어가 가장 오래된 샘플을 버리는 것은 결정성 경로에 있지 않고, ring은 생산자의
것이다. `pump`는 프레임당 한도가 있다(§23.3은 뷰어를 예산제로 돌린다).

`metric_rows()`는 §12.4 세트를 고정된 순서로 내놓으며, 종단 지연은 p50과 p95로 나눈다.
단일 `step/s` 수치는 금지된다; 측정되지 않은 메트릭은 `0`이 아니라 `--`로 렌더된다.

## 6. 전/후 이미지 (§23.3)

§23.3: *"자기 눈으로 차이를 보는 것이 비전 디버깅의 절반이다."*
`BeforeAfter::run(obs, input)`은 Observation IR을 `es_compile::CpuPlan`으로 컴파일하고 —
이는 GPU 로어링이 판정받는 것과 같은 CPU 레퍼런스이지, 별개의 전처리 구현이 아니다(§11.3) —
한 프레임에 대해 그것을 실행하고, 원본 입력을 각 이미지 출력과 짝짓는다.

표시 디코딩은 CPU 커널이 정의하는 바를 따른다: `u8` 텐서는 센서의 HWC 프레임이고, 그 밖의
것은 파이프라인의 CHW float다. `[0, 1]` 범위를 벗어나는 float 값(`Normalize` 출력)은 텐서
고유의 최소/최대값으로 재조정되어, 정규화된 이미지가 검게 잘리지 않고 보이게 된다; 이미
범위 안에 있는 값은 절대 스케일을 유지한다.

디스크상의 번들은 샘플 프레임을 갖고 있지 않으므로, `BeforeAfter::sample(obs)`는 그래프가
선언한 `ImageInput`의 형태를 한 결정적 그래디언트를 합성한다. 텔레메트리 이미지 스트림이
연결되면(§23.3), 그 프레임이 그래디언트를 대체할 뿐 다른 것은 바뀌지 않는다.

## 7. 의존성과 빌드

워크스페이스에 고정된 `eframe`/`egui` `0.32.3`. 최신 릴리스(0.36.2)가 아니라: 0.32가
워크스페이스 `rust-version`(1.85)과 MSRV가 맞는 마지막 버전이다. 기능은 최소한이다 —
`glow` + `default_fonts` + 두 Linux 윈도잉 백엔드 `x11`/`wayland`; accesskit 없음, wgpu 없음
— 이는 이 크레이트의 증분 재빌드를 약 2.4초로 유지한다. `persistence`는 최근 파일 목록을
위해 이 크레이트가 혼자 추가한다(12절).

`x11`/`wayland`는 실행 가능한 바이너리만을 위한 것이 아니라 Linux에서 필수다: 둘 다 없으면
winit 0.30이 자체 `compile_error!`("The platform you're compiling for is not supported by
winit")에서 멈추므로, Linux에서는 `cargo clippy --workspace`조차 실패한다(§26.1은 Linux
x86_64를 1차 플랫폼으로 정한다). 네이티브 Wayland 세션과 순수 X11 호스트가 각자의 백엔드를
필요로 하므로 둘 다 켠다. 둘 다 빌드 시점에 시스템 `-dev` 패키지를 요구하지 않으며 —
pkg-config 검색 경로를 비워도 `cargo check -p es-editor`가 성공한다 — Windows/macOS에서는
아무 효과가 없다. Linux에서 창을 여는 것은 실행해 보지 않았다(검증 호스트에 디스플레이 없음):
`Status: unverified`. 패킷: `docs/packets/M4/P-M4-R9.md`.

## 8. 여기 없는 것

- 3D 씬 뷰, 상태 스트리밍, pause/step/rewind, 핫패치(§23.3) — 이들은 트랜스포트 반대편에
  실행 중인 프로세스가 필요하다.
- 노드별 실시간 값과 §23.2의 reward → exposure 역추적. 그것들이 걸릴 구조(한 크레이트 안의
  `LayeredGraph` + `TelemetryModel`)는 준비되어 있다; 스트림을 노드에 연결하는 것이 다음
  패킷이다.
- §28.7 gate-9 수치(텔레메트리 + 그래프 뷰 비용이 학습 처리량의 1% 미만): `Target /
  Status: 미검증 (unverified)` — 생산자 없는 뷰어로는 측정할 수 없다.

---

## 9. 2단계: 편집 가능한 그래프 (§23.4, M3 W6)

읽기 전용은 그대로 남았고; 편집은 그 위가 아니라 곁에 추가되었다. `LayeredGraph`는
손대지 않았다 — §3의 예측이 맞아떨어졌다: 편집 모델을 도입하는 것은 `app.rs`에 대한
변경과 두 개의 새 view-model 파일이었다.

| 절반 | 위치 | 테스트 |
|---|---|---|
| edit 모델 | `src/model/edit.rs`, `src/model/palette.rs` | 완전히, 헤드리스로 |
| canvas | `src/app.rs` (`edit_canvas`, `CanvasView`) | 컴파일만 |

`EditSession`은 하나의 IR(`EditIr::{Task, Observation, Learning}` — Deployment와
Evaluation은 그래프가 아니라 레코드다), 그 `.eslayout` `Layout`, 두 노드 레지스트리, 진단
목록, undo/redo 스택을 소유한다. 여섯 개의 편집이 UI가 할 수 있는 모든 것을 다룬다:
`AddNode`, `RemoveNode`, `Connect`, `Disconnect`, `SetParam`, `MoveNode`.

### 편집이 해서는 안 되는 것

- **`MoveNode`는 레이아웃 전용이다.** `Layout::positions`만 쓸 뿐 그 외에는 아무것도
  쓰지 않는다; 테스트 하나는 노드를 세 번 옮기고 `task_hash`가 바뀌지 않았음을
  단언하고, 두 번째 테스트는 `save()`의 `.esgraph` 쪽 절반이 이동 전후로 바이트
  단위로 동일함을 단언한다(§4.2 규칙 7, §14.3).
- **에디터에는 kind별 코드가 없다.** 노드는 `TaskNodeRegistry::create` /
  `LearningNodeRegistry::create`로부터 나오고, 파라미터는 노드를 재직렬화해서 그
  테이블을 factory에 돌려주는 방식으로 교체되며, add-node 메뉴는
  `Palette::from_registries`다. `es-ir`에 노드 kind를 추가하면 여기서는 아무
  편집 없이도 그것이 보인다. `docs/design/node-sdk.md`를 보라.
- **새 추상화 없음.** `INV-17`은 두 개의 노드 factory를 허용하며 SDK는 정확히 그
  둘에 `NodeSchema`를 더한 것이다. 그 결과 Observation IR에는 factory가 없으므로,
  Observation 그래프에 대한 `AddNode`와 `SetParam`은 `FACTORY-001`을 보고하는
  반면 나머지 네 편집은 동작한다.

### 검증, 그리고 "거부됨"이 의미하는 것

모든 편집은 두 검사를 모두 실행한다: `EditIr::validate()`(IR 자신의 전체 패스)가
권고성 진단 목록을 채우고, `Graph::validate_declared_ports()`가 그 편집의 운명을
결정한다. 그 검사에서 **새 오류**를 일으키는 편집 — 포트 타입 불일치(`TYPE-003`), 알
수 없는 포트(`GRAPH-010`), 하나의 입력에 두 번째로 들어오는 엣지(`GRAPH-003`) — 는
되돌려져 반환되며, 히스토리에는 절대 들어가지 않는다. 이미 그래프에 있던 오류는 그대로
남는다: 깨진 그래프에서는 작업을 거부하는 에디터는 아무도 그래프를 고칠 수 없는
에디터이고, 절반만 저작된 그래프야말로 누군가 에디터를 여는 바로 그 대상이다.

`RemoveNode`는 걸려 있는 모든 엣지를 함께 가져가므로, 제거 작업이 나중에
`GRAPH-002`가 그 제거를 거부하게 만들 매달린 끝점을 남길 수 없다.

### Undo

편집별 역연산이 아니라, 편집 하나당 하나의 전체 상태 스냅샷이다. `RemoveNode`를
역연산으로 undo하려면 노드, 그 파라미터, 그 레이아웃 항목, 걸려 있던 모든 엣지를
복원해야 한다 — 은근히 틀릴 기회가 네 번이고, 은근히 틀린 undo는 뚱뚱한 undo보다
나쁘다. 저작된 그래프는 수백 개의 노드다; `edit.rs`의 `ponytail:` 주석이 그것이 더
이상 사실이 아니게 될 때의 업그레이드 경로를 명명한다. 테스트 하나가 세 편집을 undo와
redo로 왕복시키고 IR, 레이아웃, 해시를 비교한다.

### 여전히 `egui-snarl` 없음

편집 기능을 손에 쥔 채로 다시 결정했지만, 답은 바뀌지 않았다. snarl이 대체할
것은 `CanvasView`와 페인팅 루프다: 핀 히트 테스트, 사각형 드래그, 베지어 곡선 그리기
약 190줄인데, 이는 자신만의 노드 모델, 자신만의 레이아웃 저장소, 자신만의 포트
개념을 들여오는 의존성을 받아들일 만큼은 아니다 — 이 세 개념은 이미 여기에
`es_ir::Graph`, `es_ir::serial::Layout`, `es_ir::Port`로 존재한다. 실제로 제대로
만들기 어려운 부분인 제스처-`Edit` 매핑은 snarl이 우리 대신 해주는 것이 아니다:
그것은 `EditSession`이고, 테스트되어 있다.

### 여기 없는 것

- 다중 선택, 복사/붙여넣기, 박스 선택, 미니맵 — 12절. 검색과 파라미터 인스펙터가
  간 곳이고(M7/E3), 나머지 넷을 따져 놓은 곳이다.
- Observation IR 노드 편집, 그리고 Control Graph(IR-C) — 3단계, M4.

---

## 10. Run 탭: 끝난 실행을 연다 (§23.3, §10.5, M7/E1)

모든 `es eval run` / `es loop collect`은 `report.json`, `evaluation.lock`, `events.json`,
`traj/<cell>.estraj`을, 그리고 `--frames`와 함께라면 `frames/<cell>/NNNNNN.bin`을 쓴다.
지금까지 이것들을 읽는 것은 `es video mosaic`과 `jq`를 쥔 사람뿐이었다. **Run** 탭이 그
디렉터리를 연다.

`es-editor <run-dir>`와 File 필드는 경로 하나를 받는다; `report.json`을 가진 디렉터리는
실행이고 나머지는 번들이다(`RunView::is_run_dir`). 플래그가 아니라 디스크에 있는 것으로
구분한다: 실행 디렉터리와 번들 디렉터리는 헷갈릴 수 없고, 잘못 입력한 사람은 모드가 아니라
다른 뷰의 오류를 받는다.

### `model/run_view.rs`

| 호출 | 주는 것 |
|---|---|
| `RunView::open(dir)` | `EvaluationReport`, `events.json`의 `BTreeMap<String, Vec<es_eval::runner::StepEvent>>`, 그리고 `traj/`와 `frames/`의 목록(적재는 하지 않는다) |
| `cells()` | **에피소드**당 `CellRow` 하나: 이름, 스위트, 시드, 리포트 자신의 이름으로 된 그 스위트의 메트릭, 궤적과 프레임의 존재 여부 |
| `columns()` / `sort_by(i)` | 테이블 헤더, 그리고 그중 무엇으로든 안정 정렬 |
| `timeline(cell)` | 틱마다 `EventSource`와 디코딩된 `EventSet`, 그리고 종류별 합계와 각 종류의 첫 틱 |
| `Timeline::buckets(n)` | 같은 틱들을 `n`개의 열로 접은 것 — 어떤 너비에서도 그릴 수 있게 |
| `acceptance()` | `report.acceptance` 그대로 |
| `filmstrip(cell, 8)` / `frame(cell, i)` | 고르게 퍼진 최대 여덟 개의 프레임 인덱스, 그리고 디코딩된 `Rgb8Image` 하나 |
| `set_frames_root(dir)` / `frames_root()` | `<cell>/NNNNNN.bin`을 찾을 위치; 열 때는 `<run>/frames` |
| `selected_cell()` / `select(name)` | 선택. 이것이 리플레이 패널(§11)과의 결합 전부다 |

**"cell"이라 불리는 것이 둘**이고 이 파일은 둘을 구분한다. `es_ir::evaluation::CellResult`는
§10.1 표의 *스위트 × 메트릭* 하나이고, 디스크상의 cell — `events.json`의 키,
`frames/<cell>/`, `traj/<cell>.estraj` — 은 `Evaluation::run_shard`가 `<suite>-<NN>`으로
이름 붙인 *에피소드* 하나다. `CellRow`는 에피소드이며 자기 스위트가 측정한 메트릭을
가지므로, 에피소드가 둘인 스위트의 두 행에는 같은 숫자 셋이 나타난다. 대안 — `CellResult`당
한 행 — 은 리포트 표 그 자체이고, `nominal-01` 에피소드를 보고 싶은 사람이 찾는 것이 아니다.

**메트릭 이름은 리포트의 것이다.** `columns()`는 리포트가 가진 메트릭 이름의 합집합이므로,
`MetricSpec`에 메트릭이 추가되면 여기를 고치지 않아도 나타난다. 하드코딩은 없고,
`Histogram`이나 `Unavailable` 값은 `0`이 아니라 그것 자체로 표시된다.

**시드는 `evaluation.lock`에서 온다.** `report.json`은 시드를 담지 않는다. lock의
`seeds[i]`는 cell 이름의 `NN`으로 고르며, 그것이 `run_shard`가 세는 에피소드 인덱스다.
lock이 없으면 그 칸은 `--`다: 지어낸 시드는 없는 시드보다 나쁘다.

**위반 비트는 다시 유도하지 않고 디코딩한다.** `StepEvent::events`는
`es_safety::EventSet::bits()`다. `EventSet`에는 `from_bits`가 없고 `es-safety`는 이 패킷이
고칠 것이 아니므로, `decode_events`는 인코더가 쓴 것과 같은 표인 `ViolationKind::index()`를
통해 다시 넣고, 테스트가 14개 종류 전부를 왕복시킨다.

**버킷은 연속적이고, 빈틈이 없고, 전체를 덮는다.** 그래서 어떤 `n`에서도 종류별 개수가
타임라인 합계로 다시 더해진다(오라클은 `n ∈ {1, 7, 64}`를 확인한다). 한 버킷은 그것이 덮는
가장 심각한 소스를 보여주므로(`Policy < Human < Clamped < Fallback`), 400틱 에피소드의 클램프
한 틱도 반올림으로 사라지지 않고 보이는 자국으로 남는다.

**실행은 두 개의 시계를 기록하고, 요약은 둘 다 적는다.** `events.json`의 레코드 하나는
*프레임* 하나 — 제어 스텝 하나 — 이고 그 프레임이 돈 `PhysTick`을 싣는다. 데모의 주기에서는
프레임당 물리 틱이 넷이므로, "first at tick 544"는 224칸짜리 스트립 어디에도 없다.
`Timeline::kind_rows()`는 `KindRow { kind, frames, first: FirstSeen { frame, tick } }`을
돌려주고 `KindRow::label()`이 *"Velocity: 2 frame(s), first at frame 1 (tick 1)"* 을 쓴다 —
스트립 자신의 인덱스가 먼저, 물리 틱이 뒤에. `Timeline::heading(cell)`도 같은 단위를 센다
(*"nominal-00: 224 frame(s)"*). 사람에게 어느 시계를 보여줄지는 결정이므로, 그 문구는
`app.rs`가 아니라 테스트가 붙은 모델에 있다(§28.10 규칙 3).

**프레임은 `--frames`가 가리킨 곳에 있다.** `es eval run --frames <dir>`는 지정받은 곳에 쓰고,
그곳은 보통 `<run>/frames`가 아니라 실행 디렉터리의 *형제*다. 그래서 실제 실행의 표는 탭이
그곳을 가리키기 전까지 모든 셀에 `frames 0`을 보여줬다. `set_frames_root(dir)`이 루트를 옮기고
행을 다시 만든다 — 덕분에 홀로 있는 리포트도 외부 프레임 디렉터리에서 셀을 얻는다 — 그리고
`app.rs`에는 `Scene` 옆에 `Frames` 필드 하나가 붙어, 열 때 `frames_root()`로 채워지고 포커스를
잃을 때 적용된다.

**표, acceptance 줄, 스트립, 필름스트립은 하나의 세로 스크롤 영역**이며
(`auto_shrink([false, false])`라 리플레이 패널이 남긴 만큼을 채운다), 필름스트립은 자기만의
가로 스크롤을 갖는다: 160 px 썸네일 여덟 개는 좁은 창보다 넓고, 잘린 프레임은 없는 프레임처럼
보인다.

### 의도적으로 하지 않는 것

- **살아 있는 프로세스 없음.** 이 탭은 끝난 디렉터리를 읽는다. 돌고 있는 `es eval run`에
  붙는 것은 E4의 `--telemetry`이고, 그것은 다른 전송이다.
- **실행을 편집하지 않음.** 아무것도 되쓰지 않는다: 에디터가 다시 쓸 수 있는 아티팩트는
  해시 체인이 보증할 수 없는 아티팩트다(§5.3, §10.5).
- **캐시 없음.** `frame()`은 요청받은 것만 디코딩하고, `app.rs`는 선택된 cell의 필름스트립
  텍스처 여덟 개만 들고 있다가 경로가 바뀌면 버린다.
- **불완전한 실행도 열린다.** 필요한 것은 `report.json`뿐이다. 없는 것은 상태 줄에
  이름으로 적힌다. 사람이 에디터로 여는 실행은 흔히 끝까지 쓰이지 못한 실행이기 때문이다.
  다만 깨진 `events.json`은 추측이 아니라 오류다(§25.1).

픽스처는 `tests/fixtures/visible-learning/run/`이다: 스위트 2 × 에피소드 2, 각각 4틱이며 그중
`Clamped` 둘과 `Fallback` 하나가 실제 위반 비트를 싣고, cell당 96×96 프레임 하나가 있다.
`#[ignore]`된 `generate_fixture_run`이 쓰며, 그것은 `EvaluationReport`와 `EvaluationLock`을
만들어 `es_eval::runner::write_artifacts`와 `FrameSink`에 넘긴다 — 바이트가 그것을 만드는
타입 자신의 것이 되도록. 프레임 `.bin`과 그 `layout.json`만 직접 쓰는데, `es_eval`의 해당
작성기가 비공개이기 때문이다; 형식은 그 작성기의 문서 그대로다.

---

## 11. Replay 패널: 에디터 안에서 재생되는 `.estraj` (§23.3, M7/E2)

§23.3은 "에디터가 장면을 로컬에 복제하고 포즈/관절만 받는" 것을 요구한다. 기록된 궤적이
바로 그 스트림이며, 오프라인이다. `es video showcase`는 장면 파일과 `.estraj`만으로 실행을
다시 렌더링할 수 있음을 이미 증명했다 — 이 패널은 같은 일을 CPU에서, `egui` 캔버스 안에서,
상호작용 가능한 속도로 한다: **물리 없음, GPU 없음, `ffmpeg` 없음, 서버 없음.**

Run 탭의 아래쪽 패널로 산다(§10의 표가 에피소드를 고르고, 이 패널이 그것을 재생한다).
결합 전부는 `RunView::selected_cell()`이고, 패널은 텍스트 필드 하나를 더한다. 실행
디렉터리는 자기 장면 파일을 담지 않기 때문이다 — showcase가 받는 바로 그 `--scene`이다.

**높이는 위젯이 아니라 모델이 답한다.** `replay_view::panel_height(replay, available)`는
아무것도 로드되지 않은 동안 `None`을 돌려주고 — 그때 패널은 컨트롤 줄들뿐이다 —
`ReplayView::is_loaded()`가 참이 되면 탭의 45%를 돌려준다. 바닥은 320 px(볼 만한 캔버스),
천장은 80%(위의 표가 사라지지 않게)다. 첫 판은 45%를 조건 없이 가져갔고, 빈 캔버스가 위의
필름스트립을 잘랐다. `app.rs`는 두 상태에 각자의 패널 id(`replay-controls` / `replay-canvas`)를
주는데, egui가 끌어서 바꾼 높이를 id별로 기억하고 두 상태는 각자의 높이를 원하기 때문이다;
실행이 바뀌면 리플레이가 떨어지고 패널은 스스로 줄어든다.

### `model/replay_view.rs`

| 호출 | 주는 것 |
|---|---|
| `ReplayView::open(scene, traj)` | `SceneDesc`(`es backend`의 `load_scene`처럼 확장자로 MJCF/URDF 구분)와 `Trajectory`; 틱 0을 테셀레이션해 두므로 지원되지 않는 geom은 나중의 빈 캔버스가 아니라 여기서의 오류가 된다 |
| `ticks()` / `qpos(t)` / `scene_at(t)` | 길이, 관절 상태, 그리고 그 틱의 `TriScene` — showcase가 렌더링하는 바로 그 호출 |
| `project(t, &Camera)` | `Vec<Tri2d>`: 화면 점 셋, 각 점의 카메라 공간 `z`, 평면 색 `[u8; 3]`, 무게중심 깊이 키, 원본 삼각형 인덱스, **뒤에서 앞으로 정렬** |
| `Raster::size_for(panel)` / `Raster::draw(&Projected, w, h)` | 그만한 패널에 맞는 래스터 크기와, `Rgb8` + 깊이 프레임 자체 (M7/E8) |
| `Camera::view()` | 그 좌표가 속한 `es_render::CameraView` |
| `Camera::orbit(dyaw, dpitch)` / `zoom(f)` | `look_at`을 중심으로 한 구면 위의 새 카메라; 순수 함수, `#[must_use]`, 내부 상태 없음 |
| `advance(dt, rate_hz)` / `step(±n)` | 재생. 양끝에서 고정되며 `playing`, `tick`, `speed`는 모델의 것이다 |

**투영은 `es_render` 자신의 것을 뒤집은 것이다.** `ViewParams::new(&camera.view())`가 렌더러의
`f32` 카메라를 주고, 월드 정점은 `es_render::cpu::quat_rotate_inv`와 `cpu::primary_dir`가
광선을 쏠 때 쓰는 바로 그 `fx, fy, cx, cy`를 지난다. 여기서 규약을 다시 유도하는 것은 없다
(§3.1: `OpenCV` 카메라 프레임, 이미지 원점 좌상단).

**셰이딩은 `es_shade_lambert`를 네 줄로 옮긴 것**이다. `cpu::shade_lambert`가 비공개이기
때문이다: `ambient + max(dot(n, light), 0) * (1 - ambient)`에 albedo를 곱하고 emission을
더한 뒤, (공개된) `srgb_encode`와 렌더러의 반올림을 거친다. 곱셈과 덧셈은 분리되어 있고
`mul_add`가 아니다: 융합 연산은 한 번 반올림하는데 렌더러는 두 번 반올림한다.
`RenderConfig::rs`를 여기서 만드는 이유는 오로지 광원 방향과 앰비언트 바닥값이 렌더러의
상수이고 그것의 두 번째 사본이 아니게 하기 위해서다. `flat_shade_matches_the_renderer`는
카메라 하나 앞에 삼각형 하나를 두고 `es_render::cpu::rasterize`로 래스터화한 뒤, 투영된
무게중심 아래의 픽셀이 그 색과 같다고 단언한다 — 카메라와 투영과 셰이딩을 한 번에 고정한다.

**근평면 클리핑은 삼각형 단위다.** 정점 하나라도 근평면 위나 뒤에 있는 삼각형은 그 `z`로
나누지 않고 버린다. 나누면 눈을 통과해 이미지 반대편으로 투영된다. 평면에 걸친 삼각형은
쪼개지 않고 사라진다: 쪼개는 클리퍼는 정점을 보간해야 하는데, 이 카메라의 축척에서 팔이
반쯤 뒤에 있는 일은 없다.

### 깊이 버퍼 (M7/E8)

E2는 화가 알고리즘을 내보내며 `ponytail:`로 표시했다. 그 주석이 예고한 바로 그 방식으로
틀렸다: SO-101은 모든 관절에서 서로 관통하는 링크들이라, 무게중심 깊이로 삼각형 전체를
정렬하면 베이스 판이 어깨 위에, 그리퍼 손가락이 손목을 뚫고, 앞팔이 통의 앞벽을 통과해
그려졌다. 소유자 메모 2026-09-21: *폴리곤이 마치 깊이 버퍼가 없는 것처럼 보인다*. 이제 있다.

`Raster::draw(&Projected, w, h) -> Raster { w, h, rgb: Vec<u8>, depth: Vec<f32> }`는 약 40줄의
스캔라인 래스터라이저다: 삼각형마다 세 에지 함수로 부호 있는 면적을 구하고, 경계 상자의
픽셀마다 같은 에지 함수를 픽셀 **중심**(`px + 0.5`, `cpu::primary_dir`가 광선을 쏘는 점)에서
평가해 그 면적으로 나눈다. *부호 있는* 면적으로 나누면 와인딩에 따라 세 무게중심 좌표의
부호가 함께 뒤집히므로 "셋 다 ≥ 0"이 **양면** 모두에 대한 하나의 내부 판정이 된다 — 장면이
필요로 하는 것이 바로 그것이다. `TriScene`의 테셀레이션은 와인딩을 보장하지 않고 `Rs` 경로도
양면을 셰이딩하기 때문이다. 셰이딩은 그대로다: `project`가 이미 계산한 평면 Lambert
`Tri2d::color`와, 삼각형 전체 단위의 같은 근평면 클리핑.

**깊이는 보간되며, `1/z`로 보간된다.** 화면 공간은 카메라 공간 깊이의 역수에 대해 선형이지
깊이 자체에 대해 선형이 아니다. 그래서 `Tri2d`가 정점마다의 `z`를 싣고, 래스터라이저는 세
역수를 무게중심 보간한 뒤 한 번 역수를 취한다. `z`를 직접 보간하면 가장 비스듬히 보이는
삼각형 쌍인 테이블 상판에서 미터 단위로 틀린다. 판정은 엄격히 더 가까울 때만이므로, 공유된
에지는 — 내부 판정이 양쪽을 다 남기기에 두 번 그려진다 — 먼저 온 삼각형의 것이 되고 이음매는
생기지 않는다.

**정렬은 남는다.** 더는 무엇이 보이는지를 결정하지 않지만, 칠하는 순서를 고정하는 것이
정렬이고 그 순서가 같은 깊이에서의 동점을 가른다 — 그래서 이미지는 여전히 (궤적, 틱, 카메라,
w, h)의 순수 함수다. `tests/golden/editor/replay_tick0_order.json`(틱 0의 인덱스 2,754개)도
그 덕에 의미를 유지하고, `cargo xtask verify-goldens`는 골든의 삭제를 위반으로 취급하므로
그것을 퇴역시키는 쪽이 정렬 비용(5 ms 프레임에 대한 `f32` 키 2,754개의 안정 정렬)보다
비쌌을 것이다.

**패널은 메시가 아니라 텍스처를 보여준다.** `app.rs`는 래스터를 `egui::ColorImage`로 틱이나
카메라가 바뀔 때만 한 번 업로드하고 — 키는 `(tick, Camera)`이며, 크기가 `Camera`의 두 필드라
리사이즈도 이 키가 덮는다 — 캔버스에 늘려 그린다. `egui::Mesh`와 `replay_mesh`는 사라졌다.
제스처, 스크러버, 재생/일시정지는 손대지 않았다.

**`size_for`는 축이 아니라 픽셀을 예산으로 쓴다 — 패킷에서의 이탈.** 패킷은 "패널 크기를
960×540으로 상한"이라고 한다. 축마다 적용하면 여기서는 틀린다: Replay 패널은 넓고 낮고
(최대화된 창에서 약 1900 × 280), 가로 960에 맞추는 균일 축소는 960 × 143을 주며, 그러면 패널은
허용된 픽셀의 4분의 1을 2배로 확대한 그림이 된다 — 대체한 메시보다 눈에 띄게 흐리다. 그래서
`size_for`는 패널 자신의 해상도를 **면적 960 × 540 픽셀**에 들어갈 때까지 균일하게 축소한다:
1920 × 1080은 여전히 정확히 960 × 540이 되고, 실제 패널은 같은 비용으로 약 1865 × 278이 된다.
균일한 축소는 타협 대상이 아니다 — `ImageSpec::pinhole`은 정사각 픽셀(`fx == fy`)이라 두 축에
같은 배율을 쓰면 해상도만 다른 같은 화면이지만, 배율이 둘이면 찌그러진다.

**골든은 순서가 아니라 그림이다.** `tests/golden/editor/replay-tick0-320x180.bin`은 픽스처의
틱 0을 showcase 카메라에서 320 × 180 `Rgb8`로 본 것이며, `ES_GENERATE_GOLDENS=1` 뒤의
`#[ignore]`된 `generate_raster_golden`이 한 번 쓰고 그 뒤로는 읽기 전용이다(§1.4). 옆의
`.json`은 그것이 찍힌 카메라를 적어 두고 테스트가 그것을 먼저 단언하므로, 카메라 상수가
흘러가면 설명 없는 픽셀 차이가 아니라 그 단언으로 실패한다.

**측정**(이 장비, `replay_raster_is_fast_enough`, 데모 장면의 삼각형 2,754개, 960 × 540, 20회
중앙값): **릴리스 약 5 ms**(세 번의 실행에서 4.8~5.0 ms, 최소 3.8 ms), 디버그 약 70 ms.
목표는 16 ms였다; 디버그 수치는 그 목표가 아니며,
50 Hz 재생은 어차피 디버그 빌드에서 손으로 확인했다.

### `look_at`은 재사용이 아니라 반복이다

`es_env::render::look_at`은 `es-env`의 `render` 피처 뒤에 있고, 그 피처는 `es-render`와
**`es-gpu`**를 함께 끌어온다: 에디터에서 그것을 켜면 GPU로 아무것도 렌더링하지 않는 뷰어에
Vulkan을 링크하게 되는데, E2가 그것을 금지한다. 그 함수를 피처 밖으로 옮기는 것도 순수한
이동이 아니다 — `es_render::CameraView`를 반환하고 `es_render::ImageSpec::pinhole`을
호출하는데, 둘 다 선택적 의존성의 것이다. 그래서 `Camera::view`는 같은 `es-render` 타입 위에서
그 산술(전방 × 월드 업으로 만든 기저, 그다음 Shepperd 쿼터니언)을 반복하고, 위의 테스트가
그것을 사본의 출처가 아니라 렌더러에 대고 고정한다. `es-env`가 언젠가 `es-render`를 필수
의존성으로 만들면, 이것은 한 줄짜리 위임이 된다.

### 픽스처 궤적

`tests/fixtures/visible-learning/run/traj/nominal-00.estraj`: `tests/fixtures/mjcf/
so101_pick_place.xml`의 48틱, 36 kB이며 `#[ignore]`된 `generate_fixture_traj`가 쓴다.
**물리 백엔드는 관여하지 않고 필요하지도 않다.** `Trajectory`는
`es_physics_core::backend::ModelInfo`로 형태가 정해지고 `StateView`로 채워지는데 둘 다 평범한
구조체다: 생성기는 장면 바디마다 인덱스 하나를 가진 `ModelInfo`를 만들고 `nq`/`nv`는 관절에서
`MuJoCo` 자신의 배치대로 합산한 뒤, 틱마다 장면의 부모 사슬로 모든 바디의 월드 포즈를
합성한다(`shoulder_pan`을 -0.6에서 +0.6 rad까지 쓸어가며; MJCF 힌지는 자식 프레임을 그
`anchor`를 지나는 `axis`에 대해 돌린다: `body.pose * T(a) * R * T(-a)`). 그리고 그 포즈들을
담은 `StateView`를 push한다. 이는 `es_render::scene::world_poses`가 하는 것과 같은 합성이며 —
거기서는 비공개, 여기서는 열두 줄, 테스트 코드에만 있다. `es-physics-core`가 **dev-dependency**인
이유가 정확히 이것이다: 에디터 자신은 그 타입들을 결코 부르지 않는다.

이 궤적은 정책 롤아웃이 아니라 합성된 홈 포즈 스윕이다: 확인해야 할 것은 기록된 포즈
스트림이 장면을 다시 포즈시키고 투영된다는 것이고, (오케스트레이터가 여는) 실제 V19b
`.estraj`는 팔만 다를 뿐 같은 바이트다.

### 여기 없는 것

- **안티에일리어싱, 그림자, 텍스처, 피킹.** 깊이 버퍼는 평면 색의 픽셀당 1 샘플이며, `Rs`
  룩의 나머지는 함께 오지 않았다.
- **에디터 안의 GPU 경로.** E2의 결정은 유효하다: 에디터는 Vulkan을 링크하지 않는다. CPU
  래스터는 그릴 가장 큰 프레임에서 약 5 ms이고, 그 결정을 지키기 쉬운 이유가 그것이다.
- **실행 자신의 제어 주기.** 패널은 데모 배포의 `rate.control`인 50 Hz로 재생한다; 실행
  디렉터리에는 그것을 읽을 Deployment IR이 없고, 주기가 틀려도 팔이 움직여 보이는 속도만
  달라진다. 실행이 자기 주기를 기록하는 순간 `--rate`는 패널의 필드가 된다.
- **카메라 프리셋과 장면 카메라.** `--camera NAME`(showcase의 다른 모드)은 장면 자신의 카메라
  목록이 필요하다; 마우스를 끄는 사람이 원하는 것은 자유 카메라다.

---

## 12. 사람이 하듯 편집하기: 인스펙터, 검색, 열기 (§23.4, M7/E3)

§23.4는 2단계에 필요한 것 세 가지를 지목한다: **파라미터**, **검색**, **대형 그래프
성능**. 그중 둘이 여기 있다. `Edit::SetParam`과 `NodeSchema`는 M4 이래 준비되고 테스트되어
있었지만 그 뒤에 위젯이 없었고 — 노드의 파라미터를 바꾸는 유일한 방법은 TOML 편집이었다 —
무엇이든 여는 유일한 방법은 경로를 타이핑하는 것이었다.

모델 파일 셋, 각각 규칙 하나(§28.10 규칙 3: `app.rs`는 아무것도 결정하지 않는다).

| 모델 | 소유하는 결정 | 테스트 |
|---|---|---|
| `model/inspector.rs` | `ParamType`이 어떤 위젯을 받는지, 사람이 친 글자가 무슨 뜻인지, 그것이 `Edit`이 되는지 | `every_param_type_has_one_widget`, `set_param_round_trips_through_the_inspector`, `a_bad_field_never_emits_an_edit` |
| `model/search.rs` | 질의가 무엇에 맞는지, 어떤 순서인지, 다음 히트가 무엇인지 | `search_is_case_insensitive_and_ordered` |
| `model/recent.rs` | 디스크의 경로가 무엇인지, 최근 연 열 개 | `classify_tells_bundle_documents_and_run_apart`, `recent_is_capped_deduplicated_and_round_trips` |

### 인스펙터

`Inspector::for_node(session, node)`는 노드의 `NodeSchema`를 세션 자신의 레지스트리에서
읽고, 현재 파라미터는 **`Edit::SetParam`이 테이블을 팩토리에 돌려주기 전에 하는 바로 그
재직렬화**로 읽는다 — 그래서 패널이 보여 주는 테이블과 `SetParam`이 덮어쓰는 테이블은
하나이고, `edit.rs`와 마찬가지로 여기에도 종류별 코드는 없다(`docs/design/node-sdk.md`).
`es-ir`에 종류가 하나 추가되면 아무 변경 없이 인스펙트된다.

**`ParamType` 하나에 위젯 하나, 그리고 그 match에는 와일드카드 갈래가 없다.**
`Bool → 체크박스`, `Int → 드래그 정수`, `Float → 드래그 실수`, `String → 텍스트`,
`Enum(vs) → 콤보`, `Shape → [a, b, …]로 파싱되는 텍스트`, `PortType → 인라인 TOML로
파싱되는 텍스트`. `ParamType`에 변형이 추가되면 이 크레이트의 *빌드*가 깨져야 한다.
대안인 와일드카드 갈래는 조용히 편집할 수 없게 된 파라미터이기 때문이다. 일곱 `Widget`은
서로 다르므로 표 테스트가 값을 한다.

**`Field::parse`는 글자가 값이 되는 유일한 자리**이고, 오류가 있는 필드는 절대 편집을
내보내지 않는다: `Inspector::edit`은 이유를 필드에 기록하고 `None`을 돌려주므로, `Int`
칸의 `"abc"`를 세션은 듣지 못한다. `Bool`과 `Enum`은 UI에서 실패할 수 없지만 — 체크박스는
`true`/`false`를, 콤보는 자기가 받은 변형을 쓴다 — 다른 경로로 들어온 글자에 패닉하는 대신
여기서는 실패 가능하다.

**필드는 스키마의 것이되, 노드가 실제로 직렬화한 키로 제한된다.** `None`인 `Option` 필드는
테이블에 없고, `params_with`는 노드에 없는 키를 거부하므로(`FACTORY-003`) 그것을 그리는
것은 죽은 컨트롤을 그리는 일이다. `es-ir`의 `infer_param_type`이 관측된 값 하나만으로
추론한다는 사실의 작은 결과 둘: `ParamType::String`은 그 catch-all이므로(실수 배열, 날짜)
현재 값이 문자열이 아닌 필드는 그것이 출력되는 TOML로 편집되고, `Enum`의 변형 목록은
*예제* 인스턴스가 보여 준 것뿐이므로 스키마가 본 적 없는 변형을 쓰는 값은 숨기는 대신
콤보에 추가된다.

**그냥 클릭하면 선택된다.** 2단계는 `selected`를 `drag_started`에서만 설정했고, 선택이
삭제와 드래그를 위해서만 있던 동안에는 그것으로 충분했다. 인스펙터가 거기 매달린 뒤로는,
읽으려면 먼저 *끌어야* 하는 노드가 이 패킷의 인수를 막은 결함이었다. 이제 클릭은
`CanvasView::hit`을 지난다 — `start_drag`이 쓰는 바로 그 히트 테스트이므로 끌 수 있는 것은
클릭할 수 있다 — 그리고 배경 클릭은 선택을 해제한다. 주 버튼만인 `clicked()`이므로 노드
추가 메뉴를 여는 오른쪽 클릭은 선택을 건드리지 않는다. 이것이 `app.rs`에서 유일하게 자기
테스트를 가진 부분이다(`a_click_hits_the_node_under_it_and_nothing_on_the_background`).
`CanvasView`는 위치와 사각형과 이름이라 디스플레이가 필요 없고, 클릭과 드래그가 하나의
기하를 읽는다는 단언은 `app.rs`를 판정하지 않는다는 2절의 규칙보다 값이 크다.

**편집은 지켜보라고 있는 것이다.** 상태 줄은 IR의 `*_hash`(사람이 한눈에 비교하는 앞
4바이트)와 그 IR이 무엇을 불평하는지의 개수를 나른다. 둘 다 세션에서 가져오고 둘 다
살아 있다. Diagnostics 탭은 세션이 열려 있는 동안 열린 번들의 목록이 아니라 세션의 목록을
보여 준다. `EditSession::apply`가 편집마다 다시 검증하고, 번들의 목록은 디스크 파일의
스냅숏이기 때문이다. `Normalize` 범위를 바꾸면 셋이 한꺼번에 움직인다.

**패널은 선택이 옮겨 가거나 세션이 바뀔 때 다시 만들어진다** — `(selected,
history().len())`를 키로. 그 사이에는 위젯이 자기 글자를 소유하므로 타이핑은 리페인트를
견디고, 패널 뒤에서 일어난 undo가 칸에 낡은 글자를 남기지 않는다.

### 검색

`Search::filter(query, &LayeredGraph)`는 각 노드의 종류 태그, 라벨, 포트 이름에 대한
대소문자 무시 부분 문자열 검색이고, 결과는 레이어-그다음-`NodeId` 순서다. 빈 질의는
**아무것도** 맞히지 않는다. 아무도 타이핑하지 않은 칸이 그래프 전체를 찾아낸 것은 아니기
때문이다. `advance()`는 순환한다.

소스는 둘, 매처는 하나다. 읽기 전용 모드는 쌓인 네 IR을 검색하고, `filter_session`은
편집 중인 한 IR을 검색한다 — 그 IR에는 계층 뷰가 만들어진 뒤에 추가된 노드가 이미 있을 수
있고, 낡은 뷰를 검색하는 쪽은 함수 하나가 덜 들지만 첫 `AddNode` 다음부터 틀린다. 캔버스가
그리는 그 라벨을 써서 같은 `(layer, node)` 쌍을 돌려주므로, 어느 쪽이 찾았든 `app.rs`는
히트를 같은 방법으로 해석한다.

중앙 정렬만 `app.rs`에 남고, 그것이 패킷이 약속한 한 줄이다:
`pan = canvas/2 − (position + node/2) × zoom`. 히트는 읽기 전용 모드에서 테두리가 쳐지고
편집 모드에서는 *선택*된다 — 선택은 인스펙터도 함께 여니, "노드를 찾아 범위를 바꾼다"는
두 동작이다.

### 열기

`recent::classify(path)`는 경로가 무엇인지에 대한 단 하나의 답이다: `RunView::is_run_dir`이
그렇다고 하면 `Run`(§10이 그 호출 하나를 재사용하므로 분류와 판독기는 어긋날 수 없다),
그 밖의 디렉터리는 `Documents`, 디렉터리가 아닌 것은 `Bundle`. `.esb`가 아닌 것으로 밝혀진
파일은 번들 판독기 자신의 오류로 실패하고, 그것이 "디렉터리가 아님"보다 많은 것을 말해
준다. 텍스트 필드, 명령줄, File 메뉴의 최근 목록, 떨군 파일
(`ctx.input(|i| i.raw.dropped_files)`)이 모두 `EditorApp::open`을 지나므로 들어오는 길은
하나이고 나오는 오류도 한 벌이다.

`Recent`는 최근 것부터 열 개를 유지하고 push에서 중복을 제거하며, `eframe::App::save`를
통해 키 **`es-editor.recent`** 아래에 저장된다 — Windows에서는
`%APPDATA%\Electric Sheep editor\data\app.ron`, Linux에서는
`~/.local/share/electricsheepeditor/app.ron` 안의 경로 JSON 배열이다. 이를 위해 `eframe`의
`persistence` 기능이 필요하고, 그것이 이 크레이트가 워크스페이스의 최소 집합 위에 더한
유일한 기능이다(§7); `ron`과 `home`을 `eframe` 뒤로 끌고 올 뿐 새 의존성 항목은 없다. 다른
버전이 쓴 저장소는 `push`를 통해 다시 만들어지므로 손으로 고친 파일이 중복이나 열한 번째
항목을 들여올 수 없고, 읽을 수 없는 저장소는 빈 목록이다. 잃어버린 최근 목록은 에디터가
뜨는 것보다 값이 없기 때문이다.

### 여기 없는 것

- **다중 선택, 복사/붙여넣기, 박스 선택.** 셋 다 새 편집이 아니라 `Edit` *열*이다: 여섯
  편집이 이미 그것을 표현하고, 설계해야 할 것은 undo 스택과 합의하는 선택 모델과
  클립보드다. 둘 다 §23.4가 2단계에 요구하는 것이 아니다.
- **미니맵.** §23.4의 세 번째 2단계 요구는 대형 그래프 *성능*이고, 사람이 체감하는 절반은
  검색이다; 미니맵은 같은 레이아웃의 두 번째 렌더러다. 캔버스가 팬하기에 느려지면 답은
  `CanvasView`의 컬링이지 그것의 작은 사본이 아니다.
- **파일 대화상자.** `rfd`는 의존성이고 네이티브 모달이며 두 번째 입구다; 텍스트 필드와
  최근 목록과 드래그 앤 드롭이 경로가 실제로 도착하는 세 경로를 덮는다.
- **Observation IR 노드의 파라미터 편집.** 그 종류들을 소유하는 팩토리가 없으므로(`INV-17`)
  `NodeSchema`도 인스펙터도 없다 — `Edit::SetParam`이 `FACTORY-001`로 보고하는 그 경계다.

---

## 13. 돌아가는 동안 보는 실행: `--telemetry`와 `--attach` (§23.1, §23.3, M7/E4)

§10은 끝난 실행을 엽다. 이것은 끝나기 전의 같은 실행이다: `es eval run --telemetry
127.0.0.1:7777`이 지금 하는 일을 발행하고, `es-editor --attach 127.0.0.1:7777` — 또는
Telemetry 탭의 **Attach** 필드와 **Connect** 버튼 — 이 그것을 읽는다. §23.1: 에디터는 아무것도
호스팅하지 않고, 돌아가는 프로세스의 클라이언트다.

### 공유하는 타입은 E1의 것이다

`model/live_run.rs`는 네 스트림을 `CellRow`와 `Timeline`로 접는다. 둘 다 `run_view.rs`의
타입이다. 그것이 설계 결정의 전부다: **Run 탭은 표 하나, 스트립 하나, 헤더 한 벌, 선택
하나를** 가지고, 그 행이 어느 쪽에서 왔는지 알지 못한다. `run_table`은 맨 위 `match` 하나로
행을 고르고

| | 끝난 실행 (§10) | 살아 있는 실행 (이 절) |
|---|---|---|
| 행 | `RunView::cells()` | `LiveRun::cells()` |
| 헤더 | `RunView::columns()` | `LiveRun::columns()` |
| 스트립 | `RunView::timeline(cell)` | `LiveRun::timeline(cell)` |
| 제목 | `report.passed` | `LiveRun::status()` |
| 프레임 | `frames/<cell>/NNNNNN.bin` | 가장 최근의 4번 스트림 이미지 |

그 아래는 전부 같은 코드다. 오라클(`live_run_folds_streams_into_run_rows`)은 바로 그
동일성이다: E1의 커밋된 픽스처 실행을 살아 있는 실행이 보냈을 메시지로 재생하고, 그렇게 나온
행·헤더·타임라인이 `RunView::open`이 같은 디렉터리로 만들어 내는 것과 같아야 한다. 같은
`StepEvent` 비트를 접는 곳이 둘인 것 — `RunView::timeline`은 `events.json`을, `LiveRun::timeline`은
와이어를 읽는다 — 은 `run_view.rs`가 E4가 고칠 파일이 아니기 때문이고, 둘이 조용히 엇갈리는 것을
막는 것이 오라클이다.

살아 있는 실행이 가질 수 없는 것이 둘 있다. **합격 판정**이 없다: `report.json`은 마지막 스위트
뒤에 쓰이므로 제목은 `LiveRun::status()`(*"live: nominal-01 running, 2 of 3 cell(s)
finished"*)이고 acceptance 목록은 비어 있다. 그리고 **정렬**이 없다: 살아 있는 표는 행이 아직
도착하는 중이라 셀 이름 순서고, 헤더를 눌러도 디스크에서 열기 전에는 아무 일도 일어나지 않는다.
선택은 양쪽 다 동작하고, 누군가 누를 때까지 선택된 셀은 *지금 돌아가는* 셀이다 — 붙은 에디터가
아무도 건드리지 않아도 살아 있는 스트립을 그린다는 뜻이다.

### 네 개의 스트림

와이어 모양이 속한 곳인 `docs/design/telemetry-protocol.ko.md` §9("프로듀서")에서
이름짓는다. 스트림 id는 스키마가 아니라 데이터다: `protocol.rs`는 그 버전에서 얼어 있다.

| 스트림 | 페이로드 | 언제 |
|---|---|---|
| 1 | `Event { cell.begin \| cell.end \| suite.end }` | 에피소드 경계마다, 그리고 스위트당 한 번 |
| 2 | `Scalars[frame, tick, source, 위반 비트]` | 관측을 뚴 모든 제어 틱 |
| 3 | `Metrics(PerfMetrics)` | `cell.end`마다 |
| 4 | `Image { rgb8 }` | `--telemetry-image-every N` 틱마다(기본 `0`, 안 보냄) |

스트림 2는 `events.json`이 기록하는 바로 그 `StepEvent`를 네 개의 수로 보낸 것이다 — 두 번째
측정이 아니라 같은 기록이고, 그래서 살아 있는 행이 끝난 행과 *같을* 수 있다. `source`는
`es_data::ActionSourceCode`의 번호(`Policy 0, Clamped 1, Fallback 2, Human 3`)라, 데이터셋 컬럼과
와이어가 같은 말을 한다.

### 프로듀서가 하지 않는 것

- **막히기.** 모든 프레임은 `Server::publish`로 나가고, 그것은 클라이언트마다의 16깊이 큐에
  `try_send`하며 가득 찬 큐에서는 *버린다*(`telemetry-protocol.ko.md` §6). 오라클
  `eval_telemetry_never_blocks_the_run`은 한 바이트도 읽지 않는 클라이언트를 붙이고 이미지로
  쌓은 다음, 실행이 리포트가 바뀌지 않은 채 끝나고 서버의 dropped가 0보다 큼을 요구한다.
- **더 계산하기.** 싱크에 건네지는 것은 실행이 이미 가지고 있던 것뿐이다: 플레인이 내놓은
  `StepEvent`, 플랜의 이미지 버퍼(복사가 아니라 빌림), 카운터, `record_cell`이 돌려준
  `CellResult`들. 플래그가 없으면 아무것도 바인드하지 않고 아무것도 달라지지 않는다 —
  `report.json`과 `events.json`은 바이트 단위로 같고, 순서 오라클이 두 실행을 비교해 그것을
  단언한다.
- **두 프로세스 이상에서 발행하기.** `--telemetry`는 `--jobs 1`이 필요하다: `--jobs N` 실행의
  셀은 워커 프로세스에서 일어나고 그중 하나만 주소를 가질 수 있다. 반쪽만 발행하는 대신
  이름을 대며 거절한다.

이 모든 것을 위해 `es-eval`은 `es-telemetry` 의존성을 얻지 않는다 — 둘 다 10층이고 §4.2는
같은 층 의존을 금한다. 평가기는 클로저(`es_eval::runner::RunSink`, 여덟 번째 확장점이 아니라
클로저다, `INV-17`)를 부르고, 둘을 모두 링크하는 유일한 곳인 `crates/es/src/cmd/eval.rs`가
`RunEvent`를 와이어 `Frame`으로 바꿈다.

### 게이트 9: 발행이 실행에 치르는 비용 (§28.7, §23.4)

데모 문서로 돌린 `es eval run`(스위트 하나, 60 제어 틱짜리 에피소드 셋, 매 틱 96×96 프레임
렌더링, 190개 프레임 발행), `--jobs 1`, 모든 스트림을 빨아들이는 구독자 하나를 붙인 채 양쪽
각각 세 번, **번갈아 가며**(일반, 텔레메트리, 일반, …) 측정했다. 측정 중에 바빠지는 기계가 한
팔만이 아니라 두 팔을 함께 움직이게 하기 위해서다. Ubuntu, RTX 4090, 16 코어, 측정 내내 로드
평균 2.9–5.1, GPU 0–17 %(옆 에이전트의 실행):

| | 1회 | 2회 | 3회 | 중앙값 |
|---|---|---|---|---|
| `es eval run`, 릴리즈 | 4.876 s | 4.281 s | 4.266 s | **4.281 s** |
| `es eval run --telemetry`, 릴리즈 | 4.477 s | 4.288 s | 4.255 s | **4.288 s** |
| `es eval run`, 디버그 | 6.610 s | 6.538 s | 6.530 s | **6.538 s** |
| `es eval run --telemetry`, 디버그 | 6.698 s | 6.583 s | 6.577 s | **6.583 s** |

**관측된 오버헤드: 릴리즈 +0.16 %, 디버그 +0.69 %**, §23.3의 *"< 1 %"*에 대해 — 둘 다 게이트를
지키며, 비용의 전부가 `serde_json` 인코딩이고 최적화 안 된 빌드가 그것을 몇 배로 물기 때문에
디버그 수치가 보수적인 쪽이다. 각 팔의 1회차는 차가운 페이지 캐시를 안고 있고, 뒤의 두 쌍은
4.3초 실행에서 50 ms 미만 차이로 기계 자체 잡음과 같은 차수다. 스크립트와 드레이너와 로그는
오라클 서버의 `~/artifacts/plan-v/m7-e4/`에 있다.

이 수치는 일반적인 값이 아니라 이 실행 모양에 대한 관측이다: 프레임을 렌더하고 torch 순전파를
돌리는 제어 틱 옆에서 루프백 소켓으로 190개의 JSON 프레임이 나간다. 더 높은 비율로 발행하는
학습 루프나 텐서 스트림을 구독하는 그래프 뷰는 다른 측정이다 — 그쪽은 `Target / Status:
unverified`다.

### 여기 없는 것

- **`es eval run` 밖의 프로듀서.** `es loop collect`와 `es train`은 아직 아무것도 발행하지
  않는다; 싱크는 `Evaluation::run_shard_with_sink`의 인자이고 그것을 부르는 곳은 거기뿐이다.
- **재접속.** 끊긴 연결은 죽은 `Source`다: `try_recv`는 영원히 아무것도 돌려주지 않고 탭은 가진
  것을 유지한다. 다시 붙는 것은 Connect 버튼이다.
- **살아 있는 실행의 리플레이.** Replay 패널은 에피소드가 끝날 때 쓰이는 `.estraj`를 자세로
  되돌린다; 살아 있는 것을 보려면 두 번째 관절 전송이 필요하고, 그것은 이것이 아니다.
- **이미지 스트림은 기본적으로 꺼져 있다.** 96×96 프레임 하나는 픽셀로 27 kB, JSON으로는 약
  100 kB다; 매 틱 하나씩 발행하는 것이 백프레셔 오라클이 일부러 쓰는 홍수다.

---

## 14. Launch 절: 에디터가 실행을 시작한다 (§23.1, §13.1, M7/E5)

§13은 누군가 시작해 둔 실행을 지켜본다. 이것은 그것을 시작한다 — 그리고 그 호스트가 되지는
**않는다**. §23.1이 설계의 전부다: 에디터는 **실행 중인 프로세스의 클라이언트**다. 명령줄을
짓고, `std::process::Command`에 넘기고, 그다음 터미널을 가진 사람과 똑같은 소켓으로 같은
프로세스에 붙는다. 에디터의 주소 공간 안에서 평가되거나 학습되는 것은 없고, 그렇게 될 수 있는
코드 경로도 없다.

### 이 절이 무엇인가

`model/launch.rs`, `LaunchModel` 하나, 그리고 그것을 그리는 `app.rs`의 70줄 남짓. 종류 선택기
하나(`es eval run` / `es train` / `es loop cycle`), 플래그마다 텍스트 상자 하나, 렌더링된
명령줄을 읽기 전용으로, **Start**, **Kill**, 상태 줄, 그리고 자식의 마지막 200줄.

모든 결정은 모델 안에 있고 `app.rs`에는 없다(§28.10 규칙 3): 어떤 종류가 어떤 플래그를 가지는지
(`fields()`, `flags()`), 각각이 뭐라고 불리는지(`LaunchField::flag`, 이것은 말 그대로 CLI의
철자다), 명령줄이 뭐라고 읽히는지(`command_line()`), 종료 코드가 무엇을 뜻하는지
(`exit_meaning`), 어느 `es`인지(`es_binary()`), 프로듀서의 소켓을 얼마나 기다리는지
(`ATTACH_TRIES` × `ATTACH_DELAY`). `app.rs`는 `fields()` 위의 `for` 루프를 그린다.

### `argv()`는 `argv[0]`을 담지 않는다

`argv()`는 **필드의 순수 함수**다: 환경 없음, 파일시스템 없음, 정규화 없음. 그것이 세 렌더링을
골든 파일로 만들 수 있게 하는 것이다 — `tests/golden/editor/launch-{eval,train,cycle}.txt`, 한
줄에 인자 하나 — 왜냐하면 프로그램이야말로 기계마다 달라지는 바로 그 부분이기 때문이다. 패널은
`binary().path` + `argv()`를 보여준다; 자식은 표시된 문자열이 아니라 `Vec<String>` 그 자체로부터
시작되므로, 표시의 인용부호는 읽기 위한 것이고 그 외의 아무것도 아니다.

모든 플래그에 하나의 규칙: **빈 값은 렌더링되지 않는다.** `--frames`와 `--jobs`는 아무도 타이핑
하지 않았으면 그냥 사라지고, 비워 둔 *필수* 플래그는 없는 플래그로 CLI에 도달한다 — `es`는 그것을
이름으로 종료 코드 2와 함께 거절한다. 대안인 `--config ""`를 넘기는 것은 에디터가 CLI에 없는 에러
메시지를 지어내게 만든다.

### `es_binary()` — 하나의 규칙, 그리고 어느 부분이 답했는지 말한다

| 순서 | 규칙 | `reason` |
|---|---|---|
| 1 | 설정되어 있고 비어 있지 않으면 `ES_BIN` | `ES_BIN` |
| 2 | 에디터 자신의 실행 파일 옆의 `es`(`es.exe`) | `beside the editor` |
| 3 | 맨 `es`, `PATH`가 해석하도록 | `on PATH` |

일상에서 중요한 것은 규칙 2다: `cargo build -p es`와 `cargo build -p es-editor`는 두 바이너리를
같은 `target/<profile>`에 넣으므로, 이 트리에서 빌드된 에디터는 설치된 무엇이 아니라 이 트리에서
빌드된 `es`를 시작한다. 이유는 idle 상태 줄에 보인다. "어느 `es`가 그랬는가"가 놀라운 결과가
불러오는 첫 질문이기 때문이다. `resolve()`는 환경을 두 개의 인자로 받으므로, 순서는 프로세스
전역 상태를 건드리지 않고 테스트로 판정된다.

### 종료 코드는 모델 안에

`eval.rs`, `train.rs`, `cycle.rs`가 같은 넷을 문서화하고, `exit_meaning`은 에디터가 그것을
되풀이하는 유일한 자리다.

| 코드 | 뜻 |
|---|---|
| 0 | 통과 |
| 1 | 실패, 또는 런타임 에러 |
| 2 | 사용법 에러 |
| 3 | **건너뜀**: 이 기계에 없는 백엔드나 런타임 — *아무것도 돌지 않았다*(§1.4) |
| 그 외 | 문서화된 코드 없이 끝남 (죽임당했거나, 크래시) |

3은 누군가의 머릿속이 아니라 표 안에 있어야 하는 행이다: 그것은 실패가 아니고, 그것을 실패처럼
칠하는 패널은 §1.4에 대해 거짓말을 하는 것이다.

**죽인 자식은 실패가 아니라 죽임당한 것으로 보고된다.** 종료 코드 혼자서는 그렇게 말할 수 없다:
윈도우에서 `TerminateProcess`는 1로 나가고 그것은 진짜 실패와 구분되지 않으며, 유닉스에서
시그널은 코드를 아예 남기지 않는다(`-1`로 보고되고, 마지막 행이 덮는다). 그래서 `kill()`은
플래그를 세우고 상태 줄은 `exit 1: killed from here`로 읽힌다 — 패널은 사람이 스스로 끝낸 실행을
두고 실패했다고 말해서는 안 된다.

### Attach는 launch를 따른다

`attach()`는 자식이 `Running`이 되기 전까지 `None`이고, `argv()`가 `--telemetry`를 담지 않는
명령에 대해서도 `None`이다 — 오늘 그것은 모든 `es train`과 모든 `es loop cycle`이다(§13의 "여기
없는 것": `es eval run` 밖에서는 아무것도 발행하지 않는다). 주소가 생기면 `attach_source()`가
E4의 `telemetry_view::attach`를 통해 그것을 걸고, 결과가 Telemetry 탭의 `Source`를 대체한다.
에디터는 클라이언트로 연결한다; 두 번째 경로는 없다.

`es eval run --telemetry`는 *아무것도 열기 전에* 서버를 바인드하지만, 그 "전"도 프로세스 생성과
인자 파싱 이후다. 그래서 `dial()`은 재시도한다: `ATTACH_TRIES` = 20회를 `ATTACH_DELAY` = 100 ms
간격으로, 그다음 주소와 예산을 이름으로 지목하는 에러 하나. 잘못된 형식의 주소는 재시도
**하지 않는다** — 기다린다고 주소가 될 수는 없다. 재시도가 모델 안에 있는 것은 의도적이다;
얼마나 기다릴지 결정하는 패널은 `app.rs` 안의 결정이다.

**접속이 스레드인 이유(E6 뒤에 발견).** 첫 버전은 UI 스레드에서 접속했고 2초를 "유한"이라 불렀다.
Windows에서는 닫힌 로컬 포트로의 `connect`가 약 2초 뒤에야 거절되므로, 20번의 시도는 창을
1분 가까이 붙잡았다 — 그리고 거기 가는 가장 흔한 길은 필수 필드를 비운 채 시작을 누르는 것이었다:
`es`는 즉시 2로 종료하고, 아무것도 듣지 않으며, 에디터는 `TcpStream::connect` 안에 앉아 있었다.
세 가지가 바뀜다: 접속은 자기 스레드에서 돌고 `start()`는 즉시 돌아온다
(`start_returns_at_once_and_the_dial_answers_later`); `es_telemetry::Client::connect`가 양쪽 다리를
모두 제한한다(`CLIENT_CONNECT_TIMEOUT` 1 s, `CLIENT_HANDSHAKE_TIMEOUT` 2 s — 받기만 하고 아무 말도 안 하는
리스너도 포함, `a_client_gives_up_on_a_listener_that_never_answers`); 그리고 `missing_required()`가
비어 있을 때까지 시작은 비활성이며 비어 있는 필드를 패널 자신의 말로 지목한다
(`missing_required_names_the_empty_fields`). Telemetry 탭의 Connect 버튼은 여전히 UI 스레드에서
접속한다 — 한 번만, 같은 시간 제한 안에서.

### 자식은 UI 스레드를 결코 건드리지 않는다

`stdout`과 `stderr`는 파이프되고 각각 자기 스레드가 하나의 `mpsc` 채널로 읽어 넣는다; `poll()`은
프레임마다 한 번 그것을 200줄 링으로 비우고 자식을 `try_wait()`한다. UI 스레드는 파이프를 읽지
않으므로, 하나를 넘치게 하는 자식이 리페인트를 멈춰 세울 수 없고 아무것도 쓰지 않는 자식이 그것을
막을 수 없다. `poll()`이 기다리는 단 한 자리는 `try_wait()`이 종료를 보고한 직후다: 파이프는
EOF이고, 리더 스레드들은 끝나는 중이며, 그들의 마지막 줄들을 기다리는 것이 사용법 에러나 `SKIPPED`
이유가 종료가 떨어진 프레임에 묻혀 사라지지 않게 한다 — 다만 `EXIT_DRAIN` = 50 ms만. 파이프를 물려받은
손자 프로세스(`--jobs` 워커, 물리 서브프로세스)가 자식이 사라진 뒤에도 파이프를 열어 둘 수 있고, 첫
버전의 무제한 `recv()`는 UI 스레드를 그것과 함께 붙잡았을 것이다
(`poll_does_not_wait_for_a_grandchild_holding_the_pipe`); 나중에 도착하는 것은 다음 프레임들이 비운다.

### 미리 채우기, 그리고 덮어쓰지 않는 것

`prefill()`은 **빈** 필드만, 세션이 이미 아는 것으로부터 채운다: `--policy`는 열린 번들의 경로,
`--out`은 열린 실행 디렉터리의 *부모*(다음 실행은 보고 있는 것의 형제다), `--scene`은 Replay
패널의 씬 필드, `--telemetry`는 Telemetry 탭의 attach 주소, 기본값 `127.0.0.1:7777`. 따라서 두
번째 번들을 여는 것이 반쯤 채운 폼을 버리는 일은 결코 없다.

### 여기 없는 것

- **Pause, step, reset, hot-patch**(§23.3). 그것들은 컨트롤 프로토콜을 필요로 하고 실행은 아무
  프로토콜도 말하지 않으므로, `kill()`이 이 패킷이 정직하게 제공할 수 있는 유일한 컨트롤이다.
  아무것도 하지 않는 Pause를 제공하는 것은 제공하지 않는 것보다 나쁘다.
- **job 큐나 둘 이상의 자식.** 하나가 돌고 있는 동안 Start는 비활성이다.
- **원격 호스트.** `std::process::Command`가 로컬이므로 자식은 로컬이다.
- **`ES_BIN`을 넘어서는 환경 편집.** 자식은 에디터의 환경을 상속한다; `PATH`나 `ES_PYTHON`을
  편집하는 패널은 두 번째의, 더 나쁜 셸일 것이다.
- **attach할 때 열린 실행을 비우는 것.** 끝난 실행 디렉터리가 열려 있으면, 새로 시작된 실행이
  발행하는 동안에도 Run 탭의 표는 *그것*을 계속 보여준다 — E4의 Connect 버튼이 이미 가진 것과
  같은 동작이다(`run_table`은 열린 실행을 먼저 매치한다). 여기서 규칙을 지어내는 것보다 E4와의
  일관성을 골랐다; M7 리뷰를 위해 적어 둔다.

### 골든

`tests/golden/editor/launch-{eval,train,cycle}.txt`는 `#[ignore]`된 `generate_launch_goldens`가
한 번 생성했고 그 뒤로는 읽기 전용이다(§1.4). 생성기는 **`ES_GENERATE_GOLDENS=1`이 설정되어
있지 않으면 실행을 거부한다**: `cargo test -- --include-ignored`는 워크스페이스의 모든 무시된
테스트를 쓸어 담고, 그 쓸기 아래에서 자기 골든을 다시 쓰는 생성기는 "골든이 여전히 맞는다"를
동어반복으로 만들 것이다(M7 리뷰 항목). 거부는 실패가 아니라 출력하고 돌아오는 것이어서 쓸기
자체는 여전히 통과하고, 의도적으로 `SKIP` 줄이 아니다 — 기계에 빠진 것은 없고, `cargo xtask ci`의
오라클 스캔이 그것을 레퍼런스 오라클로 세어서는 안 된다.

---

## 15. 전문가가 아닌 사람을 위한 에디터 (§23.1, §23.2, §13.1, M7/E6)

§10–§14는 엔지니어에게 필요한 것은 다 붙였지만 그 외의 사람에게 필요한 것은 하나도 붙이지
않았다. "bundle.esb, 다섯 개의 .toml이 든 폴더, 또는 실행"을 달라는 맨 입력칸, 라벨이 `--config`와
`--out`인 시작 패널, `envelope_violation_rate`가 머리글인 결과 표, "(spec 23.3)"을 인용하는 도움말,
그리고 한글이 없는 기본 글꼴 — 그래서 한국어 경로는 네모로 보였다. 2026-09-21 오너 지시:
*도메인 지식이 없는 사람에게 에디터가 너무 어렵다. UI/UX를 개선하고, 다국어 텍스트가 깨지지
않도록 글꼴을 고르라.*

아래는 전부 이 크레이트의 나머지와 같은 §28.10 규칙 3 — **`app.rs`에서는 아무것도 결정하지
않는다** — 을 문구와 글꼴과 대화상자에 적용한 것이다.

### 문자열 표 규칙

`crates/es-editor/i18n/en.toml`과 `ko.toml`. TOML이 평평하게 유지하도록 키를 따옴표로 감싸고
(`"home.open_project" = "…"`), `model/i18n.rs`에 `include_str!`로 들어간다: `Lang { En, Ko }`,
`Strings::get(lang)`, `t(lang, key)`, 그리고 숫자를 품는 몇 안 되는 템플릿을 위한
`fill(lang, key, args)`. 실행 중에 없는 키는 키 자체로 그려진다. 화면에 `tab.design`이 보이는
에디터가, 아예 뜨지 않는 에디터보다는 낫기 때문이다.

`i18n_tables_are_complete_and_used`가 CI에서 그것을 도달 불가능하게 만들며, 이것이 텍스트에
대한 이 패킷의 오라클 우선 주장 전부다: 두 표의 키 집합이 같고, 모든 키가 크레이트 어딘가에
리터럴로 나타나며, 크레이트 안의 키 모양 리터럴은 모두 키이고, **`*.hint`가 아닌 값에는
"spec "이 들어가지 않는다**. 마지막 규칙이 "(spec 23.3)"을 보이는 텍스트에서 호버로 옮긴다.
원시 지표 이름, CLI 플래그, 절 번호가 있을 곳은 호버다. 오타는 그래서 두 번 실패한다 — 떠도는
리터럴로 한 번, 진짜 키가 쓰이지 않은 것으로 또 한 번.

이 두 표는 저장소에서 **영어가 아닌 소스 텍스트를 담아도 되는 유일한 곳**이기도 하다.
`.githooks/pre-commit`의 `is_doc`에 `*/i18n/*.toml` 경우가 추가되었고, CLAUDE.md의 Conventions
문단이 그 사실을 한 문장으로 적는다. 한국어는 그 밖 어디에도 없다 — 테스트에도, 주석에도.

### 글꼴 규칙

`model/fonts.rs::system_cjk_font()`는 고정된 OS별 목록을 `std::fs`로 훑어 처음 존재하는 파일을
돌려준다:

| OS | 후보, 순서대로 |
|---|---|
| Windows | `C:\Windows\Fonts\malgun.ttf`, `msyh.ttc`, `meiryo.ttc` |
| macOS | `/System/Library/Fonts/AppleSDGothicNeo.ttc`, `PingFang.ttc`, `Supplemental/NotoSansCJK*.ttc` |
| Linux | `/usr/share/fonts/**/NotoSansCJK*.{ttc,otf}`, `NanumGothic.ttf`, `DroidSansFallback*.ttf` |

의도적인 거절이 셋이다. **아무것도 동봉하지 않는다**: CJK 글꼴 한 벌은 10–20 MB이고 패밀리마다
라이선스가 다르며, 이 저장소의 바이너리 파일은 이유가 있어 읽기 전용 골든이다. **글꼴 탐색
크레이트를 쓰지 않는다**: "이 경로가 있는가"는 `std::fs`와 목록이고, 설치된 모든 글꼴을 열거하는
크레이트는 이 에디터에 없는 문제를 푼다 — 리눅스 패턴의 `*`와 `**`는 30줄짜리 매처(`matches`,
`walk`, 네 단계로 제한, 항목을 정렬해서 후보가 둘인 디렉터리도 늘 같은 것으로 풀린다)다.
**언제나 마지막 대체, 결코 처음이 아니다**: `install`은 글꼴을 `Proportional`과 `Monospace`
양쪽의 **끝**에 덧붙인다. 그래서 라틴 문자는 힌팅이 더 나은 egui 자신의 글리프를 유지하고,
egui가 그리지 못하는 문자만 아래로 떨어진다. `font_fallback_is_last`는 만들어진
`FontDefinitions`가 두 패밀리 모두에서 여전히 egui 자신의 목록으로 *시작하는지*를 단언한다.

후보가 하나도 없는 기계는 오류가 아니라 평범한 결과다: `install`이 `Err(candidates)`를 돌려주고,
에디터는 그래도 열리며, 상태 표시줄이 `status.font_missing`을 읽는다 — 찾아본 경로 목록과,
데비안·우분투용 `fonts-noto-cjk`라는 말. 설명이 있는 네모가 설명 없는 네모보다 낫다.

글꼴 옆에서 `TextSize { S, M, L }`가 **본문 15 px / 제목 20 px**를 기준으로 모든 `TextStyle`을
한꺼번에 키우고 줄인다(egui 자신의 기본값은 12.5 px인데, 그건 툴킷을 만든 사람의 눈에 맞춘
선택이다). 두 설정 모두 §12의 최근 목록 옆에서 `eframe::Storage`에 `es-editor.lang`과
`es-editor.text-size`로 보존된다 — 이 크레이트가 쓰는 모든 저장 키를 `recent.rs`가 적어 두므로
둘이 충돌할 수 없고, `Settings::from_codes`가 평범한 문자열을 받으므로 파싱은 화면 없이
판정되며 `app.rs`에는 `get_string` 두 번만 남는다.

### 라벨 표

`model/labels.rs`. 모든 함수가 전역적이고 **와일드카드 갈래가 없다** — `MetricSpec`에 지표가,
`LaunchField`에 플래그가 하나 추가되면 이 크레이트의 빌드가 깨져야 한다. 그 대안은 아무도 끝내
이름 붙이지 않는 열이기 때문이다. `metric_and_launch_labels_are_total`이 두 언어에서
`MetricSpec::ALL`(18개), `LaunchField::ALL`(11개), `LaunchFlag::ALL`(3개), `Kind::ALL`,
`Tab::ALL`을 돌며 라벨이 서로 다르고, `_`를 담지 않고, 원시 이름과 절대 같지 않음을 단언한다.

| 원시 이름 | 쉬운 이름 (en) | 쉬운 이름 (ko) |
|---|---|---|
| `success_rate` | Success rate | 성공률 |
| `intervention_rate` | Human takeovers | 사람이 넘겨받은 비율 |
| `collision_rate` | Collisions | 충돌 비율 |
| `envelope_violation_rate` | Safety limit hits | 안전 한계 위반 |
| `action_smoothness` | Motion smoothness | 움직임의 매끄러움 |
| `episode_length` | Episode length | 에피소드 길이 |
| `failure_mode_histogram` | Failure causes | 실패 원인 |
| `domain_gap` | Gap from the real world | 실제와의 차이 |
| `chunk_underrun_rate` | Motion gaps | 동작이 끊긴 비율 |
| `end_to_end_latency_p50` | Reaction time (typical) | 반응 시간 (보통) |
| `end_to_end_latency_p95` | Reaction time (slowest 5%) | 반응 시간 (느린 5%) |
| `physics_steps_per_sec` | Physics speed | 물리 계산 속도 |
| `camera_frames_per_sec` | Camera speed | 카메라 속도 |
| `pixels_per_sec` | Pixel throughput | 픽셀 처리량 |
| `observation_gb_per_sec` | Sensor throughput | 센서 데이터 처리량 |
| `policy_inferences_per_sec` | Policy speed | 정책 추론 속도 |
| `actions_per_sec` | Action rate | 동작 출력 속도 |
| `gpu_memory_peak` | Peak GPU memory | GPU 메모리 최대 사용량 |

| 원시 이름 | 쉬운 이름 (en) | 쉬운 이름 (ko) |
|---|---|---|
| `--config` | Evaluation settings | 평가 설정 |
| `--policy` | Policy file | 정책 파일 |
| `--scene` | Scene file | 장면 파일 |
| `--out` | Output folder | 결과 폴더 |
| `--frames` | Also save pictures | 사진도 저장할 폴더 |
| `--jobs` | Parallel workers | 동시에 돌릴 개수 |
| `--telemetry` | Watch live at | 실시간으로 볼 주소 |
| `--telemetry-token` | Watch password | 관찰 암호 |
| `--telemetry-image-every` | Send a picture every | 사진 보내는 간격 |
| `--recipe` | Training settings | 학습 설정 |
| `--from` | Start from step | 시작할 단계 |
| `--dry-run` | Check only, do not run | 실행하지 말고 점검만 |
| `--allow-new-evaluation` | Allow a new evaluation | 새 평가 기준 허용 |
| `--skip-expert-gate` | Skip the expert check | 전문가 점검 건너뛰기 |

탭도 같은 길을 간다 — Graph → *Design* / 설계, Run → *Results* / 결과, Telemetry →
*Live* / 관찰, Images → *What the policy sees* / 정책이 보는 것, Diagnostics → *Problems* /
문제 — 옛 이름과 명세 절은 `Tab::hint_key`에 들어간다. 그 때문에 `Tab`이 `app.rs`에서
`labels.rs`로 옮겨졌다: 탭은 이제 이름 하나, 호버 하나, 목적지 하나이고 그것은 모델 데이터이며,
옮기면서 셸이 들고 있던 목록 사본이 사라졌다.

`metric_by_name`은 쉬운 이름과, `report.json`의 열이나 텔레메트리 행이 들고 오는 원시 문자열
사이의 이음매다. 코드베이스에 하나 있는 별칭도 여기가 감당한다:
`TelemetryModel::metric_rows`는 지연 시간을 `PerfMetrics`의 필드 철자대로
`p50_end_to_end_latency`라 적고, `MetricSpec`은 `end_to_end_latency_p50`이라 적는다.

**일부러 번역하지 않은 것.** 그려진 명령줄(터미널에 그대로 붙여 넣는 줄이다), 시작 로그에 찍히는
`es` 자신의 표준 출력, `EsBinary::reason`, IR 진단의 코드와 메시지, 측정되지 않은 지표의
`reason`. 그것들은 다른 크레이트와 다른 패킷의 것이고, 한국어를 지어내는 것은 `es`가 무슨 뜻이었는지
에디터가 추측하는 일이다. 이 패킷의 범위가 닿지 않은 모델 문자열 셋 — `RunView::status`,
`LiveRun::status`, `Timeline::heading` — 도 같은 이유로 아직 영어이며, M7 리뷰 항목으로 적어 둔다.

### 첫 화면 모델

`labels::Step::ALL`은 §13.1의 순환을 순서대로 담는다 — 설계, 수집, 학습, 평가, 관찰. 각각
작업의 낱말(`word.*`), 쉬운 말 한 문장(`home.step.*`), 그리고 버튼이 데려갈 탭(`Step::tab()`;
수집·학습·평가는 모두 결과 탭으로 간다. 시작 패널이 거기 있기 때문이다)을 갖는다.
`home_screen_lists_the_five_steps_in_loop_order`가 순서를, 두 언어 모두에서 각 문장이 문장임을,
그리고 두 단계가 같은 말을 하지 않음을 단언한다. `app.rs`는 `Step::ALL` 위에 격자를 그리고 버튼
셋 — *프로젝트 파일 열기…*, *실행 결과 열기…*, *최근 항목* — 을 놓을 뿐 아무것도 결정하지 않는다.

첫 화면은 여섯 번째 탭이 아니라 **아무것도 열지 않은 설계 탭 자체**다. 사람이 처음 닿는 화면이
거기이고, 일을 시작하려면 먼저 떠나야 하는 탭은 그 일 자체보다 나쁜 첫 화면이다. 거기서는 검색
상자도 감춘다. 아직 찾을 것이 없기 때문이다.

### 대화상자 기능

`rfd 0.17.2`가 **이 패킷이 더하는 단 하나의 의존성**이다(MIT, 두 타깃 모두에서 네이티브).
GTK나 XDG 포털 백엔드가 딸려 오지 않도록 `default-features = false`로 받는다. 모양은 이렇다:

```toml
[features]
default = ["file-dialogs"]
file-dialogs = ["dep:rfd"]

[target.'cfg(any(windows, target_os = "macos"))'.dependencies]
rfd = { workspace = true, optional = true }
```

카고에는 타깃별 기본 기능이 없으므로 `model/dialogs.rs`가
`all(feature = "file-dialogs", any(windows, target_os = "macos"))`로 가르고, 모든 함수에
`None`을 돌려주는 `cfg` 반대편 쌍이 있다. 패킷이 요구한 효과가 정확히 성립한다: 리눅스는 기능이
켜져 있든 아니든 스텁을 컴파일하고 새 요구 사항이 생기지 않으며, `--no-default-features`는 어느
타깃에서나 스텁을 빌드하고, **`app.rs`는 `rfd`를 이름조차 부르지 않는다** — `dialogs::pick(browse)`를
부를 뿐이고, 어떤 필드가 *어떤* 대화상자를 원하는지는 `labels::browses`가 정한다.
`dialogs::AVAILABLE`은 그 `cfg`를 그대로 `const`로 만든 것이라, 대화상자가 없는 빌드는 버튼을
숨기는 대신 흐리게 그리고 `open.no_dialog.hint`를 호버에 단다. 직접 적는 경로, 최근 목록,
끌어다 놓기는 모든 타깃에서 그대로이므로 대화상자로*만* 닿을 수 있는 것은 없다. 이것은 §12의
"여기 없는 것"에 있던 파일 대화상자 항목을 뒤집는다. 그 항목은 에디터에 비전문가 사용자가
보이지 않던 때 쓰인 것이다.

### 손으로 확인한 것

Windows 11, `malgun.ttf`를 찾아 설치. 두 언어의 첫 화면, E4 고정물 번들과 E5 고정물 실행을 열기,
한국어 경로 `C:\Users\User\문서\테스트`를 `--out`에 적어 넣어 입력칸에서도 그려진 명령줄에서도
네모 없이 보이는 것, 결과 머리글 안전 한계 위반에 마우스를 올려 `envelope_violation_rate`가
나오는 것, 글자 크기 크게, 그리고 언어와 크기가 `eframe::Storage`를 통해 재시작을 넘겨 살아남는
것. 스크린샷: `target/plan-u/e6/`.

### 여기 없는 것

- `es` 자신의 CLI 출력 현지화, 그리고 위에 적은 모델 문자열 셋.
- 세 번째 언어. 하나 더하는 일은 파일 하나와 `Lang` 변형 하나이고, 그러면 오라클이 같은 키
  집합으로 그것을 붙든다.
- OS별 글꼴 *설정*. 후보 목록이 고정인 것은 의도다. 글꼴 설정 화면은 두 번째 문제이고, 사람에게
  정말 필요한 한 가지 — "왜 내 글자가 네모지"— 는 상태 표시줄이 답한다.
