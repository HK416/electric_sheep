<!-- Korean translation of docs/design/control-graph.md. The English file is the working copy; regenerate this when it changes. -->

# Control Graph (IR-C) — 설계

Spec refs: spec 6.2 (IR-D / IR-C 분리), spec 6.3 (`Parallel`은 존재하지 않는다; `Wait` /
`Repeat` / `Condition`은 IR-C이거나 `Compare`+`Select`다), spec 6.4 (실행 의미론), spec 6.5
(표현식), spec 6.6 (결정성 규칙), spec 23.4 (Control Graph *편집*은 M4다), spec 28.6 (M4
범위). Packet: `docs/packets/M4/W2-control-graph.md`.

Type C 산출물: 이 노트는 사람이 검토한 설계이며, 코드가 이를 따르는 것이지 그 반대가
아니다.

## 1. IR-C는 무엇을 위한 것인가

Spec 6.2는 pick&place, reach, push 같은 표준 vision-manipulation 태스크가 IR-D만으로
표현 가능하다는 것을 명시한다. IR-C는 **다단계 조립(multi-stage assembly)**을 위해
존재한다: `pick`, 그다음 `move`, 그다음 `place`, 각 단계는 자신의 보상 항, 자신의 완료
판정 조건, 자신의 데드라인을 가지며, 에피소드 보상은 실제로 실행된 단계들의 합성이다.

그래서 IR-C는 의도적으로 *작다*. 순서화, 데이터에 의존하는 분기 하나, 유계 반복, 그리고
IR-D 그래프의 이름 붙은 슬라이스를 추가한다. 데이터플로는 전혀 추가하지 않는다: 이것이
읽는 모든 값은 IR-D 포트이며, 이미 존재하는 evaluator가 평가한다.

## 2. Schema

```
ControlGraph {
    root:  ControlNodeId,
    nodes: BTreeMap<ControlNodeId, ControlNode>,
}

ControlNode =
    | Sequence { children: Vec<ControlNodeId> }
    | Branch   { condition: Expr, then_: ControlNodeId, else_: ControlNodeId }
    | Repeat   { body: ControlNodeId, until: RepeatUntil }
    | SubTask  { task: SubTaskRef, timeout_ticks: u32 }

RepeatUntil = Count(u32) | Until(Expr)

SubTaskRef {
    name:           String,            // 그래프 안에서 유일함; 이 단계의 정체성
    rewards:        BTreeSet<String>,  // 이 단계가 점수를 매기는 TaskNode::Reward 이름
    observation:    BTreeSet<String>,  // TaskIr::observation_spec 채널의 부분집합
    success:        Option<Expr>,      // 단계 완료 판정식; None = 한 스텝
    reset_on_entry: bool,
    weight:         f64,               // 에피소드 보상에서 이 단계의 계수
}
```

`ControlNodeId`는 `es_ir::graph::NodeId`다 — IR-D가 쓰는 것과 같은 id 타입이며, 자신만의
네임스페이스를 갖는다.

`TaskIr`는 필드 하나를 얻는다:

```
TaskIr { .., control: Option<ControlGraph> }
```

`None`은 IR-D 전용 태스크이며 기본값이므로(`#[serde(default)]`), IR-C 이전에 작성된 모든
태스크는 그대로 파싱된다.

### 2.1 `SubTask`가 node id를 나열하는 대신 이름으로 지목하는 이유

`SubTask`는 부모 IR-D `TaskGraph`의 *슬라이스*이지만, 그 안에 `NodeId`를 담아서는 안
된다. `canon_task`(spec 11.2, 부록 B.7 속성 1)는 IR-D 그래프를 다시 레이블링한다; IR-D id를
들고 있는 control 노드는 그 재레이블링 아래에서 낡아버릴 것이고, `task_hash`는 재레이블링에
불변이지 않게 될 것이다. 이름은 재레이블링 아래에서 안정적이므로, 슬라이스는 다음으로
이름 붙는다:

* 자신의 `Reward` 노드 이름(`TaskNode::Reward { name, .. }`), 그리고
* 자신의 `ObservationSpec` 채널 이름.

`Terminate` 노드는 슬라이스되지 *않는다*: 단계의 완료는 `SubTaskRef::success`이며,
태스크 수준의 `Terminate` 노드는 그대로 — 에피소드 전체에 대한 판정식으로 — 남는다.

## 3. 실행 의미론 (spec 6.4)

Executor는 env당 하나씩 있는 결정적 상태 기계이며, **env 스텝마다 한 번**, 물리 이후이자
보상 이전에 전진한다. 그 안 어디에도 wall clock은 없으며(spec 6.6 `DET-002`), 선언된
두 가지 — `Branch::condition`과 `RepeatUntil::Until` — 이외의 데이터에 의존하는 스케줄링
결정도 없다.

env별 상태는 `(node, index)` 프레임의 스택이다 — 루트에서 활성 `SubTask`까지의 경로 —
그리고 활성 단계에 대한 틱 카운터:

```
descend(n):
    Sequence { children }  -> push (n, 0); descend(children[0])   (비어 있으면 -> advance)
    Branch   { .. }        -> push (n, arm); descend(arm)          arm은 여기서 한 번만 선택된다
    Repeat   { body, .. }  -> push (n, 0); descend(body)
    SubTask  { .. }        -> push (n, 0); stage ticks := 0; 정지 — 이것이 활성 단계다

advance():                                       // 활성 SubTask가 끝났다
    SubTask 프레임을 pop한 뒤, 최상위 프레임부터 순서대로:
    Sequence -> index+1 < len   ? descend(children[index+1]) : pop, 다시 advance
    Repeat   -> 반복 더 있음    ? index+1로 descend(body)     : pop, 다시 advance
    Branch   -> pop, 다시 advance                              // arm은 하나뿐
    스택이 비었음 -> 에피소드 끝
```

**진입은 리셋이 아니라 에피소드의 첫 스텝이다.** `Branch`는 IR-D 포트를 읽는데, 리셋
시점에는 아직 바인딩된 것이 없으므로, `reset_env`는 상태만 지우고 첫 `step`이 보상
콘(cone)이 방금 채운 포트를 갖고 루트에서부터 내려간다. 이 walk는 256 descent 스텝
(`MAX_DESCENT`)으로 유계이므로, 빈 body를 가진 퇴화된 트리가 생성되어도 무한히 도는 대신
env가 종료된다.

`Repeat { until: Count(k) }`는 body를 정확히 `k`번 실행한다(`k > 0`이 검증됨).
`Repeat { until: Until(e) }`는 `e`가 false로 평가되는 동안 body를 재진입하며, 추가로
에피소드 예산으로 유계다 — executor는 결코 참이 될 수 없는 판정식에서 블로킹하지 않는다.

`SubTask`는 다음 순서로 끝난다:

1. `success`가 0이 아닌 값으로 평가되거나(`Expr::eval`; `None` — 없는 포트, 0으로 나누기 —
   는 이미 `episode::evaluate`가 처리하는 것과 정확히 같게 "발동하지 않음"으로 취급된다), 또는
2. `stage ticks >= timeout_ticks`가 되며, 이는 단계가 아니라 **에피소드**를 끝내고 그
   단계의 실패 플래그를 설정한다.

`success: None`은 첫 스텝에서 단계를 완료시킨다; 이것이 "이 슬라이스를 한 스텝만
실행한다"는 퇴화 케이스이며, 사소한 단계들에 대한 `Sequence`가 의미하는 바다.

### 3.1 단계 포트

어떤 IR-C `Expr`을 평가하기 전에도, executor는 보상과 종료 콘이 읽는 것과 같은 포트
맵에 세 개의 포트를 써넣어, 조건식이 기계가 어디에 있는지를 보고 분기할 수 있게 한다:

```
stage.index    부모 Sequence 안에서 활성 SubTask의 인덱스, f64로
stage.ticks    활성 단계에서 보낸 스텝 수, f64로
stage.done     단계가 완료된 스텝에서는 1.0, 그 외에는 0.0
```

`Expr`이 지목할 수 있는 그 외 모든 것은 `ScalarPlan`이 만들어내는 IR-D 바인딩
(`qpos[i]`, `qvel[i]`, `sensor[i]`, `time`, `time.episode`)이다 — IR-C는 새로운
evaluator도 새로운 `Expr` variant도 도입하지 않는다.

**세 단계 포트는 env마다 독립이다.** `Env`는 배치 전체에 대해 포트 맵을 하나만 두고
`bind_ports`에서 env마다 다시 채운다; IR-D 바인딩은 거기서 덮어써지지만 단계 포트는 그
집합에 없으므로, `bind_ports`가 먼저 `control::clear_stage_ports`를 호출한다. 그러지
않으면 해당 env 자신의 `write_ports`보다 **먼저** 실행되는 두 독자 — `episode::evaluate`
안의 task 레벨 `Terminate` 콘과, 그래프 진입 시 한 번 평가되는 `Branch` — 가 이전 env의
단계를 읽게 되고, 배치의 답이 env를 어떤 순서로 스텝했는지에 의존하게 된다
(`docs/reviews/M4.md` B-1, 패킷 `P-M4-R1`). 작성자에게 주는 함의: 그래프에 진입하는
스텝에서는 그 env에 대해 아직 어떤 단계도 실행되지 않았으므로 `stage.*`는 **바인딩되지
않은** 상태이고, 그것을 지목하는 진입 `Branch`는 `None`으로 평가된다 — falsy, 즉 `else_`
갈래다. 그 자리에서는 `stage.*`가 아니라 IR-D 상태로 분기하라.

### 3.2 보상 합성

`control = None`일 때 에피소드 보상은 변하지 않는다: 모든 `Reward` 노드에 대한
`sum(weight * term)` (spec 6.4).

`control = Some(..)`일 때는 합이 **활성 단계**의 보상 항에 대해서만 돌며, 단계 weight로
스케일된다:

```
reward(env) = stage.weight * sum over t in ScalarPlan.rewards, t.name in stage.rewards
                                 of t.weight * t.expr.eval(ports)
```

`rewards` 집합이 비어 있는 단계는 아무 점수도 얻지 못한다. 평가할 수 없는 항은 이전처럼
합을 `NaN`으로 오염시키는 대신 아무것도 기여하지 않는다. 에피소드에 걸쳐 단계별로
합산하는 것이 에피소드가 실제로 실행된 단계들의 합성을 반환하게 만드는 이유다: 선택되지
않은 분기는 정확히 0 스텝을 기여하며 따라서 정확히 0의 보상을 기여한다.

### 3.3 종료

control graph가 있을 때 `done`은 executor에서 온다:

| executor 상태 | `Termination` |
|---|---|
| 스택이 비었음 (모든 단계 완료) | `Success` |
| 어떤 `SubTask`가 `timeout_ticks`에 도달함 | `Failure` |
| 여전히 진행 중 | 에피소드 예산으로 넘어감 (`max_episode_steps` -> `Timeout`) |

태스크 수준의 `Terminate` 노드는 여전히 실행되며 발동하면 여전히 이긴다: 에피소드 전체에
대한 실패 판정식은 어떤 단계도 억누를 수 있는 것이 아니다. `EpisodeRecorder`는 새 열이
필요 없다 — 타임아웃 실패는 마지막 스텝에 `Termination::Failure`로 기록된다.

### 3.4 무작위화와 리셋

리셋과 도메인 무작위화는 IR-D 노드(`ResetState`, `Randomization`, spec 6.3)로 남으며
에피소드 리셋 시에 뽑힌다. 단계는 스스로 그렇게 명시하지 않는 한 이들을 다시 뽑지
**않는다**: `reset_on_entry = false`가 기본값이자 일반적인 경우인데, 다단계 조립은
하나의 연속된 물리적 에피소드이기 때문이다. `reset_on_entry = true`는 스키마에 담기고
전이의 일부로서 executor에 의해 보고되지만, 이를 `Env`에 연결하려면 오늘의
`Env::reset`에는 없는 "에피소드를 닫지 않고 이 env의 리셋 분포를 다시 뽑는" 경로가
필요하며, 미뤄져 있다(§6).

## 4. 결정성 (spec 6.6)

* 모든 컬렉션은 `BTreeMap` / `BTreeSet` / `Vec`다; 어떤 것도 `HashMap`을 순회하지 않는다
  (`DET-020`).
* 단계 시간은 정수 스텝 수다. 초는 절대 누적되지 않는다(spec 18.1).
* RNG(`DET-001`)도 wall clock(`DET-002`)도 이 기계에 전혀 들어오지 않는다 — 둘 중
  어느 것을 위한 노드도 없다.
* 데이터에 의존하는 유일한 스케줄링은 `Branch::condition`과 `RepeatUntil::Until`이며,
  둘 다 이미 결정적으로(정확한 IEEE 비교, 비유한 중간값은 `None`이 됨) 명세된
  `Expr::eval`이 평가한다.
* `Branch`는 조건을 **진입 시 단 한 번만** 평가하며, 선택된 arm은 프레임에 저장된다.
  매 스텝 재평가한다면 경로가 arm 중간의 물리 궤적에 의존하게 되어, 부분 리셋에 걸쳐
  재현 가능하지 않게 될 것이다.

같은 태스크, 시드, 백엔드에 대한 두 번의 실행은 비트 단위로 동일한 단계 경로, 보상,
종료를 만들어낸다. 그것이 `es-env` 테스트가 단언하는 속성이다.

## 5. 해싱

IR-C는 `task_hash`의 일부다. 두 번째 메커니즘을 새로 만드는 대신 IR-D 기계를
재사용한다: `ControlNode`는 `IrNode`를 구현하며, `ControlGraph::as_graph()`는 트리를
`Graph<ControlNode>`로 투영하는데, 그 엣지는 부모 -> 자식 링크다:

```
Sequence  out 포트 "child0" .. "childN-1"   -- 인덱스가 매겨져 있어, 자식 순서가 의미론적이다
Branch    out 포트 "then", "else"
Repeat    out 포트  "body"
SubTask   out 포트 없음
모든 노드는 in 포트 "parent"를 하나씩 갖는다
```

그러면 `canonical_hash`는 변경 없이 적용된다: node id는 다이제스트에 결코 들어가지
않고, 포트 이름이 구조를 실어 나르며, 자식이 뒤바뀐 `Sequence`는 포트 이름이 다르므로
다르게 해시된다. `params_canonical`은 노드 자신의 파라미터만 쓰고 자식 id는 **절대**
쓰지 않는다 — 구조는 엣지의 몫이다.

`TaskIr::task_hash`는 `control.is_some()`과, 존재한다면 control 다이제스트를 섞어
넣는다. 필드를 추가하는 것은 Task IR 스키마 변경이므로, `task::SCHEMA_VERSION`은
`1 -> 2`로 간다; `docs/design/ir-types.md`의 마이그레이션 노트 참고.

`BUILTIN_CONTROL_KINDS`(`Sequence`, `Branch`, `Repeat`, `SubTask`)는
`BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS`와 같은 방식으로(spec 28.7 게이트 10)
동결되며, 자신만의 동결된 다이제스트 `BUILTIN_CONTROL_KINDS_HASH`를 갖는다. 이는
*별도의* 목록이다: `BUILTIN_TASK_KINDS`는 `TaskNodeFactory`가 `TaskNode`로 빌드할 수
있는 kind의 집합이며, control 노드는 `TaskNode`가 아니므로, 이 넷을 그 목록에 추가하면
factory가 만들어낼 수 없는 kind를 주장하는 셈이 될 것이다.

## 6. 검증

| 코드 | 규칙 |
|---|---|
| `CTRL-001` | `root`, 자식, `then_`/`else_` 또는 `body`가 `nodes` 안에 없는 노드를 지목함 |
| `CTRL-002` | 어떤 노드가 `root`에서 도달 불가능함 |
| `CTRL-003` | 어떤 노드가 하나보다 많은 부모를 가짐 — IR-C는 DAG가 아니라 트리다 |
| `CTRL-004` | control graph에 사이클이 있음 |
| `CTRL-005` | `Repeat` count가 0이거나, `SubTask`의 `timeout_ticks`가 0 |
| `CTRL-006` | `SubTask`가 Task IR이 선언하지 않은 `Reward` 노드나 observation 채널을 이름 붙이거나, 두 단계가 이름을 공유함 |
| `TYPE-030` | `Branch` 조건이나 `RepeatUntil::Until` 표현식이 불리언이 아님 |

"불리언"은 추론되는 것이 아니라 구조적이다: `Expr`에는 타입이 없지만, 오직
`Expr::Compare`만 정확히 `0.0` / `1.0`을 만들어내므로, 루트가 `Compare`가 아닌 조건은
0을 기준으로 조용히 문턱값을 매기는 대신 거부된다. `Compare` 위에 씌운
`Expr::Clamp`는 받아들여지지 *않는다* — 사용자가 예측할 수 있는 규칙이 바로 이 좁은
규칙이다.

## 7. 미룬 것

* **에디터 지원.** Control Graph 편집은 spec 23.4에 따라 M4이며 이후 패킷의 몫이다.
  `es-editor`의 계층 그래프 뷰는 여기서 건드리지 않으며 IR-C kind를 알지 못한다.
* **`Wait`.** Spec 6.3은 이를 대체 가능한 것으로 나열한다; `success = time.episode > t`인
  `SubTask`가 이미 이를 표현하므로, 이를 위한 노드는 없다.
* **중첩 `Repeat` 깊이 상한.** `MAX_DESCENT`는 한 스텝의 walk를 유계로 만들지만, 스키마에는
  *선언된* 최대 중첩 깊이가 없고 이를 초과했을 때의 `CTRL-` 코드도 없다; 생성된 그래프가
  필요로 할 때 둘 다 추가할 것이며, 그 전에는 아니다.
* **`Env` 안의 `reset_on_entry` 연결** (§3.4).
* **IR-C를 컴파일러로 lowering하기** (`es-compile`): 여기 있는 executor는 CPU 참조이며,
  `ScalarPlan`이 보상 콘에 대해 그런 것과 같다.
</content>
