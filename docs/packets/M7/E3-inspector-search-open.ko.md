# M7 E3 — 파라미터 인스펙터, 노드 검색, 그리고 사람이 하듯 파일 열기

스펙: §23.4 stage 2(편집: 파라미터, 검색, 대형 그래프의 필요), §14.3(`.eslayout` 사이드카),
§4.2 규칙 7(`es-ir`에 UI 타입 없음), §28.10 규칙 3. 확장할 설계 노트:
`docs/design/editor-shell.md`(+ `.ko.md`) — 9절의 "여기 없는 것" 목록이 줄어들고, 새 12절.
선행: M4 에디터 stage 2(`EditSession`, `Edit::SetParam`, `NodeSchema`, `Palette`), E1(디렉터리/
번들 열기 분기).

## the question

`Edit::SetParam`과 `NodeSchema`는 존재하고 테스트되어 있는데 어떤 위젯도 그것들을 쓰지
않는다; 노드의 파라미터는 오직 TOML을 편집해야만 바꿀 수 있다. 백 개짜리 그래프에서 노드를
찾을 방법이 없고, 무엇이든 여는 유일한 방법은 경로를 타이핑하는 것뿐이다. **사람이 파라미터를
편집하고, 노드를 찾고, 파일을 열 수 있는가 — 모든 결정이 테스트된 모델 안에 있고 새 의존성은
0개인 채로?**

## spec

`crates/es-editor/src/model/`의 헤드리스 모델들:

* `inspector.rs` — `Inspector::for_node(session: &EditSession, node: NodeId) -> Inspector`는
  세션의 레지스트리에서 노드의 `NodeSchema`와 현재 파라미터 테이블(`SetParam`이 쓰는 것과
  같은 재직렬화)을 읽는다. `Inspector::fields() -> &[Field]`, 여기서 `Field { name, ty:
  ParamType, required, text: String, error: Option<String> }`다. 모든 `ParamType`은 정확히
  하나의 `Widget`에 대응한다(`Bool → 체크박스`, `Int → 드래그 정수`, `Float → 드래그
  실수`, `String → 텍스트`, `Enum(vs) → 콤보`, `Shape → "[a, b, …]"로 파싱되는 텍스트`,
  `PortType → 인라인 TOML로 파싱되는 텍스트`); `Field::parse(&self) -> Result<toml::Value,
  String>`가 텍스트가 값이 되는 유일한 곳이며, `Inspector::edit(name, text) -> Option<Edit>`가
  적용할 `SetParam`을 반환하거나 필드에 파싱 오류를 기록한다. `Enum`과 `Bool` 위젯은
  오류를 낼 수 없고, 나머지는 낼 수 있으며, 오류가 있는 필드는 결코 edit을 내지 않는다.
* `search.rs` — `Search::filter(query, &LayeredGraph) -> Vec<(layer, NodeId)>`: 노드의 종류
  태그, 라벨, 포트 이름에 대한 대소문자 구분 없는 부분 문자열 검색이며, 결과는 레이어 다음
  id 순서다; 빈 쿼리는 아무것도 반환하지 않는다; `Search::next(current) -> NodeId`가
  순환한다.
* `recent.rs` — 10개로 상한이 걸린, 최신 것이 먼저 오는, push 시 중복 제거되는
  `Recent { paths: Vec<PathBuf> }`, `to_json`/`from_json`(serde는 이미 의존성이다) — 앱은
  이미 내장되어 있는 `eframe::App::save` / `Storage`를 통해 그것을 영속화한다.
* 열기: `Opened::classify(path) -> Kind::{Bundle, Documents, Run}`는 디스크에 무엇이 있는지로
  결정하며(E1이 run 분기를 도입했다 — 재사용할 것; E1이 아직 착륙하지 않았으면 분류를
  추가하고 E1이 그것을 쓰게 할 것) `.esb` 하나, TOML 다섯 개짜리 디렉터리, `report.json`
  디렉터리에 대해 테스트된다. 드롭된 파일(`egui`의 `raw.dropped_files`)은 텍스트 필드와
  같은 경로를 거친다.

`app.rs`: 편집 모드에서 선택된 노드를 위한 오른쪽 인스펙터 패널(필드마다 위젯 하나, 나쁜
필드 아래의 오류 줄, Enter 또는 포커스 잃음에서 Apply → `EditSession::apply`), 그래프
툴바 안의 검색 상자(Enter가 순환하고, 히트는 중앙에 놓이고 강조된다 — 중앙에 놓는 것은
`pan = f(node position)`, 한 줄짜리다), 최근 목록이 있는 File 메뉴, 그리고 드롭으로 열기.
그 밖의 것은 거기서 결정되지 않는다.

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-editor/src/model/inspector.rs
crates/es-editor/src/model/search.rs
crates/es-editor/src/model/recent.rs
crates/es-editor/src/model/mod.rs
crates/es-editor/src/lib.rs
crates/es-editor/src/app.rs
crates/es-editor/src/main.rs
crates/es-editor/Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M7/E3-inspector-search-open.md
docs/packets/M7/E3-inspector-search-open.ko.md
```

`crates/es-editor/src/model/{inspector.rs,search.rs,recent.rs}`(신규), `model/mod.rs`,
`lib.rs`, `app.rs`, `main.rs`(`Recent`를 영속화하려면 `eframe::NativeOptions`의 기본
영속화가 필요하다 — 무엇이 어디에 저장되는지 노트에 적을 것), `crates/es-editor/Cargo.toml`은
**오직** 이미 존재하는 워크스페이스 크레이트가 필요할 때만(외부 추가 없음),
`docs/design/editor-shell*.md` 12절과 9절의 "여기 없는 것" 목록,
`docs/packets/M7/E3-inspector-search-open*.md`.

## oracle

1. `cargo test -p es-editor every_param_type_has_one_widget` — 모든 `ParamType` 변형에
   대한 테이블 테스트(`Enum`은 값 두 개로)가 각각 하나의 `Widget`을 단언하고 매치가
   완전함을 단언한다(`ParamType` 추가는 동작이 아니라 컴파일을 깬다).
2. `cargo test -p es-editor set_param_round_trips_through_the_inspector` — Task와 Learning
   레지스트리가 노출하는 모든 노드 종류에 대해(`Palette::from_registries`),
   `Palette::defaults`로 노드를 짓고, 인스펙터를 열어, 각 필드의 자기 텍스트를 다시
   입력하고, 나온 `SetParam` 값이 현재 값과 같음을(가짜 edit 없음) 단언한 다음, 필드
   하나를 바꿔 `EditSession::apply`가 그것을 받아들이고 해시된 파라미터에 대해 `task_hash`가
   움직임을 단언한다.
3. `cargo test -p es-editor a_bad_field_never_emits_an_edit` — `Int` 필드에 `"abc"`, `Shape`
   필드에 `"[1,"`: `edit()`은 `None`, `error`가 설정되며, 세션은 불변이다.
4. `cargo test -p es-editor search_is_case_insensitive_and_ordered`와
   `recent_is_capped_deduplicated_and_round_trips`.
5. `cargo test -p es-editor classify_tells_bundle_documents_and_run_apart`.
6. `cargo build -p es-editor`; `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/E3-inspector-search-open.md`.

## acceptance

오라클 1–6. 오케스트레이터가 패널을 통해 데모 번들 안의 `Normalize` 범위를 편집해 진단
목록이 갱신되고 상태줄에서 `task_hash`가 바뀌는 것을 보고, 이름의 일부를 타이핑해 노드를
찾으며, 재시작 뒤 최근 목록에서 번들을 다시 연다. 설계 노트 12절이 모델들과 밖에 남는 것
(다중 선택, 복사/붙여넣기, 미니맵)을 기록한다.

## forbidden

`crates/es-ir/**`(`NodeSchema`/`ParamType`은 읽히기만 하고 결코 바뀌지 않는다; `es-ir`
안의 UI 타입은 규칙 7의 위반이다); 새 외부 의존성(`rfd`, `egui_extras`, …); `app.rs`
안의 어떤 파싱이나 결정이든; 기존 여섯 `Edit`의 의미; `docs/ARCHITECTURE*.md`; 골든.
