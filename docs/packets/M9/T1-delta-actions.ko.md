# M9 T1 — `JointDelta`: 런타임이 한곳에서 적분하는 증분, 그리고 여전히 절대 목표를 보는 플레인

스펙: §8.5(2026-09-22에 추가된 문장: `JointDelta` / `EeDelta`는 현재 목표 위의 증분이다;
`es-env`가 적분한다; 플레인은 절대 목표를 검증한다; 적분기는 에피소드 경계에서 플레인이
시드한 포즈로 리셋된다), §9.3(시드), §9.4, §13.4, §28.12 규칙 1–2와 파동 1, INV-11..13
(`es-safety` 불변), INV-17. **P-M8-R6**(`es-ir` 분리)에 의존한다. 설계 노트:
`docs/design/batch-domains.md`(러너 / 청크 버퍼 절), `docs/design/evaluation-execution.md`
2.x, `docs/design/python-builder.md`(롤아웃 바인딩) — 각각 하나의 적분 규칙을 얻는다;
`rl-continuation.md`에 새 짧은 절.

## 질문

`ActionSpace::EeDelta`는 두 열거형(`es-ir-types::expr`와 `es-ir::deployment`)에 실행 의미론
없이 존재하고, `JointDelta`는 아예 존재하지 않는다. **Deployment IR이 관절-공간 증분을
선언할 수 있는가, 하나의 `es-env` 함수가 그것을 모든 소비자(수집기, 평가자, `Rollout`)가
플레인에 건네는 절대 목표로 바꿀 수 있는가, 그리고 모든 `JointPosition` 문서, 해시, 궤적,
골든이 바이트 단위로 동일하게 남는가?**

## 사양

* `ActionSpace::JointDelta`를 두 열거형 모두에 추가한다(그 중복은 이 패킷이 아니라 리뷰가
  다룰 발견이다); 부재 / `JointPosition`이 오늘의 정규 형태다 — 무엇에도 손대기 전에 커밋된
  `deployment_hash`들을 고정하라.
* **함수 하나**, `es_env::control::absolute_target(space, prev: &[f64; NJ], row: &[f64; NJ],
  out: &mut [f64; NJ])`(이름과 모듈은 에이전트의 것, 규칙이 아니다): `JointPosition`에서는
  복사하고, `JointDelta`에서는 `row`를 `prev`에 더한다. `prev`는 적분기 상태로, 청크 버퍼 /
  플레인 피드 곁에(env당) 소유되며, 매 에피소드 경계마다 플레인이 시드하는 그 측정된 관절
  상태로부터 시드되고(`begin_episode` 뒤의 `SafetyPlane::observe_state`의 첫 호출 — 같은
  `joint_state` 헬퍼를 통해 읽는 같은 숫자), 플레인의 **실행된** 출력(`SafeAction::q`)으로부터
  갱신되며, 결코 날것의 행에서 갱신되지 않는다 — 그래서 클램프된 증분이 도달할 수 없는 목표로
  누적되지 않는다.
* 소비자: `DomainRunner`(수집), `es_eval::runner::run_episode`, `es_py::Rollout::act`가 청크
  행과 `validate` 사이에서 그것을 호출한다; 플레인의 입력, 출력, 시그니처는 바뀌지 않는다
  (INV-13).
* 증분 정책의 `Normalizer{Inverse}` 통계는 증분 단위(제어 틱당 rad)로 되어 있다; Deployment
  IR의 `action.space`가 어느 쪽인지 말한다; `XIR` 교차 검사: 단위가 증분 단위가 아닌 액션
  포트를 가진 `JointDelta` 배포는 이름으로 거절된다.
* 데이터셋: `es loop collect`는 오늘처럼 *실행된 절대* 명령을 기록한다(데이터셋 스키마는
  바뀌지 않는다); `action_source`는 불변.

## context

```
crates/es-ir-types/src/expr.rs
crates/es-ir/src/deployment.rs
crates/es-ir/src/cross.rs
crates/es-ir/tests/**
crates/es-env/src/control.rs
crates/es-env/src/chunk_buffer.rs
crates/es-env/src/domains.rs
crates/es-env/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-py/src/rollout.rs
crates/es-py/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/deployment-reach-delta.toml
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/python-builder.md
docs/design/python-builder.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M9/T1-delta-actions.md
docs/packets/M9/T1-delta-actions.ko.md
```

## 오라클

1. `cargo test -p es-ir committed_deployment_hashes_are_unmoved_by_joint_delta` — 고정된 해시;
   `JointDelta`는 해시를 움직인다; 단위 교차 검사는 이름으로 거절한다.
2. `cargo test -p es-env delta_integrates_to_the_absolute_target` — 픽스처 백엔드에서, 스크립트로
   짠 절대 시퀀스와 그 일차 차분 시퀀스를 두 공간으로 몰면 실행된 명령, 플레인 이벤트, `qpos`가
   **비트 단위로** 같다; 클램프된 증분은 날것이 아니라 실행된 값으로부터 적분된다(엔벨로프가
   무는 테스트).
3. `cargo test -p es-eval demo_trajectories_are_unmoved` — 커밋된 데모 문서들(`JointPosition`)은
   전후로 픽스처 백엔드에서 바이트 단위로 동일한 `.estraj` / `events.json`으로 돈다(먼저 그
   바이트의 blake3를 고정하라).
4. `cargo test -p es-py rollout_integrates_delta -- --ignored`(서버, MuJoCo): 증분을 가진
   `deployment-reach-delta.toml` 위의 `Rollout`은 적분된 절대값을 먹인 `deployment-reach.toml`
   위의 `Rollout`과 비트 단위로 같다.
5. `git diff --stat main -- crates/es-safety`가 비어 있음; `cargo xtask ci`; `check-scope`.

## 수용 기준

오라클 1–5; 세 노트의 한 규칙짜리 문장과 그 한국어 자매 문서.

## 금지

`es-safety`에 대한 어떤 변경이든(INV-11..13); 두 번째 적분기(함수 하나, 호출자 셋); 데이터셋
스키마나 `JointPosition`의 동작을 바꾸는 것; `EeDelta`를 위한 IK(이 플랜 밖);
`docs/ARCHITECTURE*.md`; `tests/golden/**`; INV-17.
