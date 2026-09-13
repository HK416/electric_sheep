<!-- Korean translation of docs/packets/M4/W2-control-graph.md. The English file is the working copy; regenerate this when it changes. -->

# W2 — IR-C: Task IR control graph

Spec: spec 6.2 (IR-D / IR-C 분리; `Sequence`, `Branch`, `SubTask`, `Repeat`; 다단계
조립이 그 용례다), spec 6.3 (`Parallel`은 존재하지 않는다), spec 6.4 (실행 의미론),
spec 6.6 (결정성), spec 23.4 (Control Graph *편집*은 이후의 M4 패킷이다), spec 28.6
(M4 범위), spec 1.2 / spec 1.4 (패킷, 오라클 우선).

Design note: `docs/design/control-graph.md` (type C, 이 패킷의 코드보다 먼저
검토됨).

## context (범위)

```
crates/es-ir/src/control.rs            신규 — IR-C 스키마, 검증, 해싱
crates/es-ir/src/lib.rs                `pub mod control;` + re-export
crates/es-ir/src/task.rs               `TaskIr.control`, SCHEMA_VERSION 1 -> 2, 해시, 검증
crates/es-ir/src/factory.rs            BUILTIN_CONTROL_KINDS + 그 동결된 해시
crates/es-ir/src/serial.rs             TOML 미러가 `control`을 싣는다
crates/es-ir/src/norm.rs               B.7 생성기/편집에서의 control 트리
crates/es-ir-types/src/codes.rs        CTRL-001..006, TYPE-030 (`es_ir::codes`가 이를 re-export)
crates/es-ir/tests/control_graph.rs    검증, serde/TOML 왕복, 재레이블링 불변성
crates/es-env/src/control.rs           신규 — ControlExecutor (CPU 참조 executor)
crates/es-env/src/lib.rs               `pub mod control;` + re-export
crates/es-env/src/env.rs               가산적: control이 있을 때 보상/done을 주도
docs/design/control-graph.md           신규
docs/packets/M4/W2-control-graph.md    이 파일
```

이 crate들 밖의 `TaskIr` 리터럴 지점에 대한 기계적인 `control: None,` 추가는 오직
`cargo check --workspace`를 초록색으로 유지하기 위해서만 이 패킷의 일부다; 그
외에는 아무 변경도 하지 않는다.

## spec (사양)

1. `ControlGraph { root, nodes: BTreeMap<ControlNodeId, ControlNode> }`, `ControlNode`는
   `Sequence { children }`, `Branch { condition, then_, else_ }`, `Repeat { body,
   until }`, `SubTask { task: SubTaskRef, timeout_ticks }` 중 하나. Shape는
   `docs/design/control-graph.md` §2에 고정되어 있다.
2. `TaskIr.control: Option<ControlGraph>`, `#[serde(default)]`. `None`은 예전과
   정확히 똑같이 동작한다. `task::SCHEMA_VERSION`은 2가 되며 마이그레이션 노트는
   `docs/design/ir-types.md`에 들어간다.
3. `SubTask`는 자신의 IR-D 슬라이스를 `NodeId`가 아니라 `Reward` 노드 이름과
   observation 채널 이름으로 이름 붙인다 — 그렇지 않으면 `canon_task` 재레이블링이
   `task_hash`를 움직이게 된다(부록 B.7 속성 1).
4. 검증: 유일하게 존재하는 root, 트리 shape(공유 없음, 사이클 없음, 도달 불가능한
   것 없음), `Repeat` count > 0, `timeout_ticks` > 0, `SubTask` 슬라이스 이름이
   해석됨, 단계 이름이 유일함, `Branch` / `Until` 조건이 불리언임(루트가
   `Expr::Compare`). 코드 `CTRL-001`..`CTRL-006`과 `TYPE-030`.
5. 해싱은 `ControlNode`에 대한 `IrNode` 구현과 `ControlGraph::as_graph()`(트리
   엣지, 인덱스가 매겨진 `child{i}` 포트 이름이라 형제 순서가 의미론적이다)를 통해
   `canonical_hash`를 재사용한다. `params_canonical`은 절대 자식 id를 쓰지 않는다.
6. `BUILTIN_CONTROL_KINDS` + `BUILTIN_CONTROL_KINDS_HASH`는 task와 learning kind
   목록과 같은 방식으로 동결된다(spec 28.7 게이트 10). 별도의 목록이다: control
   노드는 `TaskNode`가 아니므로 `TaskNodeFactory`는 이 kind들을 주장해서는 안 된다.
7. `es_env::control::ControlExecutor`: `new(&TaskIr)`, env별 `Vec<StageState>`,
   `step(env, &mut ports) -> StageOutcome { reward_weight, done, failed,
   transitioned }`. `Sequence`는 sub-task 완료 시 전진하고, `Branch`는 진입 시
   조건을 한 번 평가하며, `Repeat`는 count / `until` 판정식까지 재진입하고,
   `SubTask` 타임아웃은 실패 플래그와 함께 에피소드를 끝낸다. `Env::step`은
   `task.control`이 `Some`일 때만 이를 사용한다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ir -p es-env -p es-editor --all-targets --features es-ir/testing -- -D warnings
cargo test -p es-ir -p es-env -p es-editor --features es-ir/testing
cargo xtask context-budget
cargo xtask check-spec-refs
cargo check --workspace --all-targets
```

## acceptance (수용 기준)

`es-ir`

- 좋은 3단계 트리는 깨끗하게 검증된다; 공유된 자식은 `CTRL-003`을 보고한다;
  사이클은 `CTRL-004`를 보고한다; `Compare`가 아닌 `Branch` 조건은 `TYPE-030`을
  보고한다; 0인 `Repeat` count와 0인 timeout은 `CTRL-005`를 보고한다; 알 수 없는
  reward 이름은 `CTRL-006`을 보고한다.
- Serde JSON과 `.esgraph` TOML 봉투 둘 다 control graph를 가진 Task IR을
  왕복한다.
- `task_hash`는 IR-D 그래프를 재레이블링해도, control 트리를 재레이블링해도
  변하지 않으며, 단계 weight, timeout, 또는 `Sequence`의 형제 순서가 바뀌면
  *실제로* 변한다. 부록 B.7의 다섯 속성은 이제 선택적 control 트리를 방출하는
  `arbitrary_task_ir`에 대해 돈다.
- `BUILTIN_CONTROL_KINDS_HASH`는 실제 목록에 대해 단언(assert)된다.

`es-env`

- `FakeBackend`에서의 3단계 `pick -> move -> place` 태스크는 단계를 순서대로
  방문하며 활성 단계의 보상 항에만 점수를 매긴다.
- 포트 값에 대한 `Branch`는 그 값이 선택하는 arm을 취한다.
- `Repeat { until: Count(3) }`는 body를 세 번 실행한다.
- `timeout_ticks`가 지난 `SubTask`는 `Termination::Failure`로 에피소드를
  끝낸다.
- 같은 태스크와 시드에 대한 두 번의 실행은 단계 경로, 보상, 종료에서 비트
  단위로 동일하다.

## forbidden (금지)

- 새로운 확장-지점 trait(`INV-17`): `ControlNode`는 내부용 `IrNode` 구현만
  얻으며 그 외에는 아무것도 얻지 않는다. `ControlNodeFactory` 없음.
- `SubTaskRef` 안의 IR-D 그래프의 `NodeId`, 그리고 `params_canonical` 안의
  자식 id.
- 에디터 작업: `crates/es-editor/src/model/graph_view.rs`는 IR-C kind를
  계속 모른다(spec 23.4는 Control Graph 편집을 이후 패킷에 둔다).
- 이 경로 어디에도 `HashMap`을 쓰지 않는다(spec 6.6 `DET-020`), wall-clock
  읽기, RNG, 또는 `Branch`와 `RepeatUntil::Until`을 넘어서는 데이터 의존적인
  스케줄링 결정.
- `crates/es-gpu/**`, `crates/es-usd/**`, `crates/es-physics-core/**`,
  `crates/es-script/**`, `crates/es-data/**`, `crates/es-eval/**`,
  `crates/es-physics-backend/**`, `crates/es-policy/**`, `crates/es/**` —
  다른 패킷들이 소유한다; 한 줄짜리 `control: None,` 리터럴 수정만 허용된다.
- `tests/golden/**`을 편집하는 것(spec 1.4, CI 읽기 전용).
