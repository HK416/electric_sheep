<!-- Korean translation of docs/packets/M4/P-M4-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R1 — IR-C 포트 맵은 env별이다

Spec: 사양 6.2 / 사양 6.4(IR-C 실행 의미론), 사양 6.6(결정론: 배치 결과는 env가 스텝되는
순서에 의존해서는 안 된다), 사양 5.2(독립적인 배치 도메인). `docs/reviews/M4.md`의 블로커
**B-1**을 닫는다.

## context (범위)

```
crates/es-env/src/control.rs
crates/es-env/src/env.rs
docs/design/control-graph.md
docs/design/control-graph.ko.md
docs/packets/M4/P-M4-R1.md
```

## spec (사양)

`Env`은 배치 전체에 대해 `BTreeMap<String, f64>` 포트 맵을 **하나만** 유지하며,
`bind_ports`에서 env마다 다시 채운다. 모든 IR-D 바인딩(`qpos[i]`, `qvel[i]`, `sensor[i]`,
`time`, `time.episode`)은 그 안에서 덮어써지므로 이들은 이미 env별이다. 세 스테이지 포트
`stage.index` / `stage.ticks` / `stage.done`은 그 집합에 속하지 않는다:
`ControlExecutor::write_ports`가 이들을 삽입할 뿐 아무것도 제거하지 않으므로, env *e*의
경우 이 env 자신의 `write_ports`가 실행되기 **전에** 맵을 읽는 두 지점에서는 여전히 env
*e−1*의 값(env 0이라면 이전 스텝의 env *n−1*의 값)을 그대로 갖고 있다:

1. `episode::evaluate` — 태스크 수준의 `Terminate` 콘(`env.rs`, `evaluate_env`).
2. 그래프 진입 시 한 번 평가되는 `Branch`(`control.rs`, `step` → `run(Some(root))`).

따라서 `stage.*`를 참조하는 어떤 `Terminate`나 진입 `Branch`든 다른 env의 함수가 되어
있었고, env를 스텝하는 순서를 바꾸면 결과가 바뀌었다.

두 리더가 모두 거쳐 가는 한 지점에서의 수정:

- `control.rs`는 세 키의 이름을 한 곳에서 지정하고(`STAGE_INDEX` / `STAGE_TICKS` /
  `STAGE_DONE`), 이들을 제거하는 `pub fn clear_stage_ports(&mut BTreeMap<String, f64>)`를
  공개한다.
- `Env::bind_ports`는 이 env의 IR-D 소스를 바인딩하기 전에 이를 가장 먼저 호출한다. IR-D만
  있는 태스크에서는 아무 효과가 없으며(키가 존재한 적이 없으므로) env당
  `BTreeMap::remove` 세 번의 비용이 든다.

결과는 `control-graph.md` §3.1에 문서화되어 있다: 그래프에 진입하는 스텝에서는 그 env에
대해 아직 아무 스테이지도 실행되지 않았으므로 `stage.*`는 **바인딩되지 않은** 상태이며,
이를 참조하는 진입 `Branch`는 `None`으로 평가된다 — falsy, 즉 `else_` 분기 — 모든 env,
모든 순서에 대해 결정적으로. env별 실행기 스크래치도 대안이었지만, 여기서는 얻는 것이
없다: 같은 '진입 시 바인딩 안 됨' 답이 그로부터도 나오며, 게다가 맵을 중복시킬 것이기
때문이다.

`ControlExecutor`는 자신의 env별 `StageState` 벡터와 `step` 시그니처를 그대로 유지한다;
어떤 스테이지 상태도 이동하지 않았고, IR-C는 여전히 보상과 종료만을 게이팅한다(INV-12 /
INV-13은 변경 없음).

## oracle (오라클)

```
cargo test -p es-env control
```

두 테스트이며, 수정 전에 작성되었다:

- `stage_ports_do_not_leak_into_the_next_env` — env 두 개, `stage.ticks > 0.5`에 대한 루트
  `Branch`, 분기는 `then` / `else`. 어느 env도 스테이지에서 틱을 소비한 적이 없으므로 둘 다
  `else`를 취해야 한다. 옛 코드에서는 env 0이 `else`를, env 1이 `then`을 취했다. env 0의
  `stage.ticks = 1`이 여전히 맵에 남아 있었기 때문이다: 수정 전에는 **실패**, 수정 후에는
  `ok`.
- `a_branch_is_per_env` — R1의 수용 형태: 조건이 실제로 서로 다른 env 두 개(`qpos[0] > 0`,
  반대 방향 토크로 구동). 각 env는 **자기 자신의** 값이 선택하는 분기를 얻으며, 어느
  env가 어느 토크를 갖는지를 바꾸면 스테이지도 함께 바뀐다 — 평가 순서가 놓아둔 자리에
  그대로 남아 있지 않는다.

## acceptance (수용 기준)

- 두 테스트 모두 통과한다; 첫 번째는 수정되지 않은 `bind_ports`에서 실패함이 확인된다.
- 기존의 여덟 `control::tests`와 `es-env`의 나머지(62개 테스트)는 여전히 통과한다,
  `two_runs_are_bitwise_identical` 포함.
- `cargo fmt --check -p es-env`, `cargo clippy -p es-env --all-targets -- -D warnings`,
  `cargo xtask layering`, `cargo xtask context-budget`가 클린하다.
- 새 자유 함수 `control::clear_stage_ports`를 제외하고는 공개 시그니처가 전혀 바뀌지
  않았다.

## forbidden (금지)

`context` 밖의 모든 것 — B-2(`es-eval/src/evidence.rs`)와 그 외 모든 M4 발견 사항은 각자의
패킷에 속한다. `ControlExecutor::step`의 시그니처나 그 `StageState`를 바꾸는 것. `Env`에
두 번째 포트 맵을 주는 것. 스테이지 포트를 추가하거나, 이름을 바꾸거나, `write_ports`가
계산하는 것을 바꾸는 것. `control.rs:158`의 timeout-vs-complete 지적(별개의 발견 사항).
