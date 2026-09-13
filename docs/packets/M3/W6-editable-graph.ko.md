<!-- Korean translation of docs/packets/M3/W6-editable-graph.md. The English file is the working copy; regenerate this when it changes. -->
# W6-editable-graph — 편집 가능한 그래프 에디터와 Node SDK

Spec: §23.4(그래프 기능 3단계; 2단계 = 편집, M3), §23.2(이 작업이 곁에 추가되는
계층 뷰), §14.3(`*.esgraph` / `*.eslayout` 저장 형식 — 레이아웃은 IR에 들어가지
않는다), §14.5(LLM 생성이 같은 `NodeSchema`를 읽는다), §28.5 W6, §1.9 항목 5
(*편집 가능한* 그래프는 cut 5; 읽기 전용은 남는다), §6.3 / §8.3(노드 집합), 부록 C
(`INV-17`, §4.2 규칙 7).

## context (범위)

```
crates/es-editor/src/model/edit.rs        (new: EditSession, EditIr, Edit, undo/redo, save/load)
crates/es-editor/src/model/palette.rs     (new: Registries, Palette, NodeSchema -> JSON)
crates/es-editor/src/model/mod.rs         (+ two `pub mod` lines)
crates/es-editor/src/app.rs               (+ Edit mode toggle, edit_canvas, CanvasView, save)
crates/es-editor/Cargo.toml               (+ serde, serde_json, toml)
docs/design/node-sdk.md                   (new)
docs/design/editor-shell.md               (+ section 9)
docs/packets/M3/W6-editable-graph.md
```

## spec (사양)

1. `model/edit.rs`, headless. `EditIr`가 `Task(TaskIr) | Observation(ObservationIr) |
   Learning(LearningGraph)`인 `EditSession { graph: EditIr, layout: Layout, registries,
   diagnostics, undo, redo }`. 여섯 개의 편집: `AddNode { kind, params: toml::Value, pos }`,
   `RemoveNode`, `Connect`, `Disconnect`, `SetParam`, `MoveNode`.
   - `apply` / `undo` / `redo`는 결정적이다. 새 노드 id는 `max + 1`이다.
   - **`MoveNode`는 `Layout::positions`만 쓰며** 어떤 `*_hash`도 바꿀 수 없다(§4.2
     규칙 7). `RemoveNode`는 걸려 있는 모든 엣지와 그 노드의 레이아웃 항목도 함께
     떨어뜨린다.
   - 모든 편집 후: `EditIr::validate()`가 권고성 진단 목록을 채우고,
     `Graph::validate_declared_ports()`에서 **새 오류**를 일으키는 편집은 되돌려져
     `Err(Vec<Diagnostic>)`로 반환되며, 히스토리에는 기록되지 않는다. 이미 있던
     오류는 편집을 막지 않는다.
   - Undo/redo는 편집별 역연산이 아니라 전체 상태 스냅샷이다(`ponytail:` 주석이
     업그레이드 경로를 명명한다).
   - `save() -> (esgraph, eslayout)`은 `es_ir::serial::write_esgraph`를 거치고;
     `load(graph, layout?)`는 `parse_esgraph`를 거친다. Deployment나 Evaluation
     파일은 `EditError::NotAGraph`다.
   - 노드 생성과 파라미터 편집은 `TaskNodeRegistry` / `LearningNodeRegistry`의
     `create` + `schema`를 거친다. **`es-editor`에는 kind별 코드가 없다.**
     Observation IR에는 factory가 없으므로(`INV-17`이 둘만 허용한다), 그
     `AddNode` / `SetParam`은 `FACTORY-001`을 보고한다.

2. `model/palette.rs`. `Registries`는 두 레지스트리를 감싸고 등록된 kind 이름들을
   추적한다(레지스트리는 kind를 해석할 수는 있어도 열거할 수는 없다;
   `*NodeFactory::kinds`가 등록 시점의 목록을 준다). `Palette::from_registries`는
   모든 kind를 그 `NodeSchema`와 함께, kind로부터 도출된 카테고리(`Task / sources`,
   `Task / ops`, `Task / declarations`, `Learning / encoders|fusion|heads|action`,
   `* / custom`)로 묶어 나열한다. `PaletteEntry::defaults`는 시작 파라미터
   테이블이고; `Palette::to_json`은 §14.5를 위해 포트, 파라미터 타입, defaults를
   export한다.

3. `docs/design/node-sdk.md`. `TaskNodeFactory` / `LearningNodeFactory`를 구현하는
   방법(예제 struct 하나), 안정성 계약(고정된 kind 목록 + 닫힌 노드 enum ⇒ 서드파티
   kind는 빌트인 노드로 확장되는 저작 단축키이므로 `.esgraph`는 절대 플러그인에
   의존하지 않는다), 그리고 의도적으로 없는 것: 플러그인 로딩 없음, 스크립팅 없음,
   세 번째 factory trait 없음, 노드별 UI 없음.

4. `app.rs`, 컴파일만. 상단 바의 `Edit` 토글이 열린 번들의 Task IR에 대해 세션을
   시작한다; 이어서 Graph 탭이 편집 가능한 캔버스를 그린다: 노드를 드래그하면 →
   놓을 때 `MoveNode` 하나, 출력 핀에서 입력 핀으로 드래그하면 → `Connect`,
   우클릭하면 → 카테고리별 팔레트가 있는 "Add node", `Delete` → `RemoveNode`,
   `Ctrl+Z` / `Ctrl+Y` → undo/redo, `Save` → 로드된 경로 옆에 두 파일. 모든 분기는
   정확히 하나의 `apply` / `undo` / `redo`로 끝난다.

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

- `save` → `load`는 추가된 노드들, 엣지 하나, 바뀐 파라미터 하나를 가진 세션을
  왕복시킨다: IR, 레이아웃, 해시가 모두 같다고 비교된다.
- 세 편집에 대한 undo와 redo는 정확히 같은 IR, 레이아웃, 해시를 복원한다; 새
  편집은 redo를 지운다.
- `MoveNode`는 `task_hash`를 절대 바꾸지 않으며, `save()`의 `.esgraph` 쪽 절반은
  이동 전후로 바이트 단위로 동일하다; 사이드카가 없는 로드는 사이드카가 있는
  로드와 같은 해시를 갖는다.
- 타입이 맞지 않는 `Connect`는 소비 노드에 `TYPE-003`으로 거부되고, 엣지는
  유지되지 않으며, 해시는 바뀌지 않고, 히스토리도 늘어나지 않는다. 알 수 없는
  포트는 `GRAPH-010`; 알 수 없는 kind는 `FACTORY-001`; 알 수 없는 파라미터 키는
  `FACTORY-003`이다.
- 팔레트는 `BUILTIN_TASK_KINDS`와 `BUILTIN_LEARNING_KINDS`에 있는 모든 kind만
  나열하고 그 외에는 아무것도 없으며, 그중 어느 것이든 자신의 `defaults()`로부터
  빌드되고, JSON export는 각 kind의 포트와 파라미터를 싣는다.
- `#[cfg(test)]` 서드파티 factory는 커스텀 kind를 등록하고, 팔레트의
  `Task / custom` 아래에 나타나며, `AddNode`를 통해 인스턴스화되고, 빌트인
  kind를 가로챌 수 없다(`FACTORY-002`).
- `es-editor`는 §1.5 context budget 안에 머무르고 `cargo xtask layering`은
  깨끗하다.

## forbidden (금지)

- `crates/es-editor/**`와 위의 세 문서 파일 바깥의 모든 변경. 특히 `es-ir`:
  `NodeSchema`, factory trait들, 고정된 kind 목록은 P29의 것이고,
  `Graph` / `Layout` / `write_esgraph`는 P28의 것이다.
- 어디에든 새 trait을 만드는 것(`INV-17`): `ObservationNodeFactory` 없음,
  `Editable` 없음, `NodeRenderer` 없음.
- IR 타입 안의 레이아웃, 또는 위치를 볼 수 있는 IR 해시(§4.2 규칙 7).
- `es-editor` 안에 손으로 작성된 kind별 생성, 파라미터 매핑, 또는 포트 목록.
- `egui-snarl`(`docs/design/editor-shell.md` §9에서 다시 결정됨: 이는 약 190줄의
  페인팅 코드를 대체하면서 중복되는 세 개의 개념을 들여올 것이다).
- 어디에든 있는 `HashMap`(§3.4 determinism; `BTreeMap`만 허용), 그리고 `app.rs`
  안에서 CI가 실행할 수 없는 모든 결정.
