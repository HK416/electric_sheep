<!-- Korean translation of docs/design/node-sdk.md. The English file is the working copy; regenerate this when it changes. -->
# Node SDK — 서드파티가 노드 kind를 추가하는 방법

`crates/es-editor`에서 바라본 `es_ir::factory`에 대한 설계 노트. 스펙: §6.3(Task 노드
집합), §8.3(Learning 노드 집합), §14.5(LLM 생성이 같은 스키마를 읽는다), §23.4(편집 가능한
그래프), 부록 C(`INV-17`).

## 1. SDK가 무엇인가

두 개의 trait과 하나의 struct, 이들 모두 이미 `es-ir`에 있다:

| Item | Role |
|---|---|
| `TaskNodeFactory` | `(kind, params) -> TaskNode` (§6.3) |
| `LearningNodeFactory` | `(kind, params) -> LearningNode` (§8.3) |
| `NodeSchema` | 하나의 kind가 받는 것: `inputs`, `outputs`, `Vec<ParamSchema>` |

그것이 SDK의 전부다. `INV-17`은 정확히 일곱 개의 단일 구현 확장 trait을 허용하며 그중
둘이 이것들이다; **에디터를 위해 새로 추가된 것은 없다**. 에디터의 `Registries`
(`model/palette.rs`에 있음)는 두 레지스트리와 각각이 등록된 kind 이름들을 담는 컨테이너일
뿐, 세 번째 추상화가 아니다.

`params`는 노드 자신의 필드들로 이루어진 `toml::Value` 테이블이며, 키는 정확히 노드
enum의 serde 필드 이름과 같다 — `kind` 키는 없으며, 태그는 factory가 제공한다.

## 2. 하나를 구현하기

```rust
use es_ir::factory::{NodeSchema, ParamSchema, ParamType, TaskNodeFactory};
use es_ir::task::{Aggregation, TaskNode};
use es_ir::{codes, Diagnostic, Port};

const GRASP_SCORE: &str = "GraspScore";

struct GraspScoreNodes;

impl TaskNodeFactory for GraspScoreNodes {
    fn kinds(&self) -> &[&'static str] {
        &[GRASP_SCORE]
    }

    fn create(&self, kind: &str, params: &toml::Value) -> Result<TaskNode, Diagnostic> {
        if kind != GRASP_SCORE {
            return Err(Diagnostic::new(codes::FACTORY_001, format!("unknown kind '{kind}'")));
        }
        Ok(TaskNode::Reward {
            name: params.get("name").and_then(toml::Value::as_str).unwrap_or("grasp").to_owned(),
            weight: params.get("weight").and_then(toml::Value::as_float).unwrap_or(1.0),
            aggregation: Aggregation::Sum,
            ty: scalar(),
        })
    }

    fn schema(&self, kind: &str) -> Option<NodeSchema> {
        (kind == GRASP_SCORE).then(|| NodeSchema {
            kind: GRASP_SCORE,
            inputs: vec![Port::new("value", scalar())],
            outputs: vec![],
            params: vec![ParamSchema {
                name: "weight".to_owned(),
                ty: ParamType::Float,
                required: false,
                default: Some(toml::Value::Float(1.0)),
            }],
        })
    }
}
```

등록하면 팔레트, add-node 메뉴, JSON export에 나타난다:

```rust
session.registries.register_task(Box::new(GraspScoreNodes))?;   // FACTORY-002 if the kind is taken
```

같은 factory가 `crates/es-editor/src/model/edit.rs`에 `#[cfg(test)]` 예제
(`GraspScoreNodes`)로 존재하며, 두 개의 테스트가 이것이 팔레트에 도달하고,
`Edit::AddNode`를 통해 인스턴스화되며, 빌트인 kind를 가로챌 수 없음을 증명한다.

## 3. 안정성 계약

**factory는 빌트인 노드를 조합할 뿐, IR 노드 타입을 새로 만들지 않는다.** `TaskNode`와
`LearningNode`는 닫힌(closed) enum이고, `BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS`는
`BUILTIN_TASK_KINDS_HASH` / `BUILTIN_LEARNING_KINDS_HASH` 뒤에 고정되어 있다 —
`crates/es-ir/src/factory.rs`의 두 테스트는 둘 중 하나의 목록이라도 바뀌면 빌드를 실패시키는데,
`IrNode::kind`가 해시 입력이고 이름 변경이 그 kind를 쓰는 모든 그래프의 `task_hash` /
`learning_hash`를 조용히 바꿔버리기 때문이다(§28.7 gate 10). 그래서 서드파티 kind는
**저작 단축키**다: 그래프는 자신의 factory가 만들어낸 빌트인 태그를 기록하고(저장된
`.esgraph`에서 `GraspScore`는 `Reward`가 된다), 플러그인이 있는 상태에서 저작된 번들은
플러그인이 없어도 그대로 로드되고, 해시되고, 컴파일된다.

결과들을, 사람들을 놀라게 하는 빈도 순으로:

1. 커스텀 kind는 IR의 새 노드가 아니므로 lowering, determinism, 해시 체인을 바꿀 수
   없다 — 이 셋은 플러그인이 절대 손댈 수 없는 것들이다.
2. `.esgraph`는 커스텀 kind를 절대 참조하지 않으므로, 그것은 파일의 의존성이 아니다.
3. 커스텀 kind가 무엇으로 확장되는지를 바꾸면 그 변경 이후에 저작된 그래프들이
   바뀌는 것이지, 이미 저장된 그래프들이 바뀌는 것이 아니다.
4. kind 이름은 선착순이다: `register_*`는 가리는(shadow) 대신 `FACTORY-002`를
   보고한다.

## 4. 스키마는 하나의 소스, 세 개의 소비자

`NodeSchema`는 에디터의 add-node 메뉴, 파라미터 에디터, 그리고 —
`Palette::to_json`을 거쳐 — §14.5의 generator 경로에 공급된다. Task IR을 생성하라고 요청받은
LLM은 에디터가 그리는 것과 똑같은 포트, 파라미터 타입, 시작 값을 읽으므로, generator와
에디터가 노드가 무엇을 받는지에 대해 서로 다른 말을 할 수 없다. `Palette::defaults(kind)`는
사용자가 아무것도 건드리기 전에 노드가 생성될 때 쓰이는 파라미터 테이블이며, 테스트 하나가
**모든** 빌트인 kind를 각자의 defaults로부터 빌드하므로, 거짓말을 하는 스키마는 CI를
실패시킨다.

## 5. 의도적으로 제공하지 않는 것

- **플러그인 로딩 없음.** `dlopen` 없음, 동적 레지스트리 없음, 플러그인 매니페스트
  없음. factory는 호스트 바이너리에 컴파일되어 들어가는 Rust 코드다. 런타임에 네이티브
  코드를 로드하는 것은 Safety Plane도 함께 실행되는 프로세스 안에 검토되지 않은
  서드파티를 들이는 셈이 된다.
- **스크립팅 언어 없음.** §14.2가 이미 Python에 대해 이 선을 그어 두었다: 확장이
  임의의 코드를 실행할 수 있다면 IR의 보장은 사라진다. factory는 *노드*를 만들어낼
  뿐이고, 컴파일러와 검증기가 추론할 줄 아는 것은 그 노드 집합이다.
- **세 번째 factory trait 없음.** Observation IR 노드에는 factory가 없으므로,
  Observation 그래프에 대한 `Edit::AddNode`와 `Edit::SetParam`은 `FACTORY-001`을
  보고한다(그 외의 편집은 동작한다). `ObservationNodeFactory`를 추가하는 것은 여덟
  번째 확장 지점이 될 것이다; `INV-17`은 안 된다고 말하므로, 그럴 가치가 생긴다 해도
  고쳐야 할 것은 스펙이지 에디터가 아니다.
- **노드별 UI 없음.** 노드는 자신의 스키마로부터 스스로를 그린다. 커스텀 위젯을
  원하는 factory는 IR 안의 UI 타입을 요구하는 것인데, 이는 §4.2 규칙 7이 금지한다.
