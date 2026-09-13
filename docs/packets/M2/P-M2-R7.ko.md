<!-- Korean translation of docs/packets/M2/P-M2-R7.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R7 — `nu != NJ`는 조용한 브로드캐스트가 아니라 오류다

Spec: §9 (Deployment IR / Safety Plane), §10.1 (평가 표), §10.4.
설계 노트: `docs/design/evaluation-execution.md` §2.5.
불변식: INV-12, INV-13 (변경 없음 — 이는 어떤 실행이 시작되는지를 바꾸는 것이지, plane이
무엇을 결정하는지를 바꾸는 것이 아니다).
리뷰 발견 사항: `docs/reviews/M2.md` — Should-fix, `crates/es-eval/src/runner.rs:362`.

`safe.q[i.min(NJ - 1)]`는 `NJ`를 넘어서는 모든 actuator를 관절 `NJ−1`의 복사본으로
구동했고, `joint_state`는 없는 관절을 `0.0`으로 채웠다. 그래서 actuator 수가 배포의
`NJ`와 어긋나는 모델이 그대로 끝까지 실행되어 §10.1 표에 잘못된 숫자를 만들어냈는데,
이는 표가 없는 것보다 나쁘다.

## context (범위)

```
crates/es-eval/src/lib.rs
crates/es-eval/src/runner.rs
crates/es-eval/tests/evaluation.rs
docs/design/evaluation-execution.md
docs/packets/M2/P-M2-R7.md
```

## spec (사양)

- **`es-eval`** — `EvalError::JointMismatch { nu, nq, nv, nj }`. 리뷰는 `{ nu, nj }`를
  요청했다; `nq`/`nv`가 같은 variant에 있는 이유는 `joint_state`에서 `0.0` 패딩을
  제거하면 모델도 최소한 `NJ`개의 `qpos`와 `qvel` 항목을 지녀야 하기 때문이며, 넷
  모두를 이름 붙이는 variant 하나가 둘보다 작기 때문이다.
- `run_episode`는 `env.reset` 이전, 즉 실행 시작 시점에
  `model.nu == NJ && model.nq >= NJ && model.nv >= NJ`를 검사하고, 그렇지 않으면
  `JointMismatch`를 반환한다.
- 브로드캐스트는 `ctrl.copy_from_slice(&safe.q)`가 되고, `joint_state`는
  `unwrap_or(0.0)` 없이 `qpos` / `qvel`에서 곧바로 `[..NJ]`를 복사한다.

## oracle (오라클)

```
cargo test -p es-eval a_model_with_more_actuators_than_joints_is_refused
```

extra actuator를 하나 더 지닌 픽스처 모델을 로드하는 backend(`FakeBackend::wide`,
`nu = 3`)를 배포의 `NJ = 2`에 대해 실행하면 `Evaluation::run`이
`EvalError::JointMismatch { nu: 3, nj: 2, .. }`를 반환해야 한다.

## acceptance (수용 기준)

- 3-actuator 실행은 오류를 낸다; 기존의 모든 es-eval 테스트(`nu == NJ == 2`)는
  green으로 남는다.
- `runner.rs`의 action이나 state 경로에는 더 이상 `min`도, `unwrap_or(0.0)`도, 0
  패딩도 남아 있지 않다.
- `cargo test -p es-eval`가 green이다.

## forbidden (금지)

- 불일치를 수용하기 위해 Safety Plane의 어떤 부분이든 넓히거나 비활성화하는
  것(INV-12), 그리고 `SafetyPlane::validate`의 시그니처에 대한 어떤 변경도(INV-13).
- 모델을 거부하는 대신 chunk나 envelope를 모델에 맞게 크기 조정하는 것: `NJ`는
  배포의 계약이며, `SafetyPlane::from_ir`은 이미 `deploy`/`NJ` 불일치를 거부한다.
- `crates/es-env`, `crates/es-policy`, `crates/es-telemetry`, `crates/es`,
  `crates/es-compile/src/budget.rs`.
