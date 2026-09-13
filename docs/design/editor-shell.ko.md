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
`glow` + `default_fonts`, accesskit 없음, wgpu 없음, persistence 없음 — 이는 이 크레이트의
증분 재빌드를 약 2.4초로 유지한다. *실행 가능한* 바이너리(컴파일만 된 것이 아니라)를
원하는 Linux CI는 eframe의 `x11`/`wayland` 기능을 추가한다; 헤드리스 빌드에는 필요 없다.

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

- 검색, 미니맵, 다중 선택, 복사/붙여넣기, 박스 선택(§23.4는 검색과 대형 그래프
  성능을 2단계 요구 사항으로 나열한다; 여섯 개의 편집이 필요로 하는 것은 단일
  선택 캔버스다).
- 파라미터 인스펙터 패널. `Edit::SetParam`과 `NodeSchema`는 둘 다 준비되어 있고
  테스트되어 있다; `ParamType`을 채워 넣는 위젯은 그 뒤에 설계할 모델이 남아
  있지 않은 UI 작업이다.
- Observation IR 노드 편집, 그리고 Control Graph(IR-C) — 3단계, M4.
