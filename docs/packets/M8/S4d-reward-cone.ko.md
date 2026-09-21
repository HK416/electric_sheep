# M8 S4d — 보상 콘이 몸체에 닿다: `es-env`의 스칼라 계획 안의 `GetBodyPose`, `Norm{L2}`, IEEE `sqrt`

스펙: §6.3(`GetBodyPose`와 `Norm`은 이미 Task IR 노드다), §6.4(실행 의미론), §6.6(`DET-010`,
그리고 2026-09-21에 추가된 문장: `sqrt`는 초월함수가 아니라 IEEE 기본 연산이다 — `Norm{L2}`는
`Expr::Sqrt`로 로워링된다), §5.4(단위 대수), §28.11 파동 3(S4d). 설계 노트:
`docs/design/batch-domains.md` 6절 "보상과 종료"(지어진 대로의 콘: `GetJointState`,
`GetSensor`, `GetTime`, `GetContact` 잎 + `Arith`, `Compare`, `Clamp`, `Normalize`, `Logic`),
`docs/design/rl-continuation.md` 5절(이것이 실행 가능하게 만드는 reach 작업; 26차원 관측).
S4b의 보고서(2026-09-21)가 찾아냈다: 보상 `−‖cube_pos − gripper_pos‖`는 Task IR로는 적을 수
있지만 `es-env`가 실행할 수 없다. 한국어 자매 문서는 같은 커밋에서(오케스트레이터의 것).

## 질문

모든 콘 잎은 스칼라 하나에 바인딩되고(`GetJointState` → 그 관절의 첫 `qpos`, `GetSensor` →
`sensor[start]`), `Source`는 `Qpos | Qvel | Sensor | Time`이며, `es_ir_types::Expr`에는 제곱근이
없다. 그래서 두 몸체 사이의 거리 — reach, place, follow 모든 작업의 보상 — 는 여기서 돌 수
없다. **콘이 몸체의 월드 위치를 세 개의 레인으로 실어 `Arith{Sub}`를 거쳐 IEEE `sqrt`를 가진
`Norm{L2}`로 들어가게 할 수 있는가 — CPU 백엔드 전체에서 비트 단위로, 커밋된 데모 작업의
로워링이 오늘 것과 바이트 단위로 동일하도록?**

## 사양

* **`Expr::Sqrt(Box<Expr>)`**를 `crates/es-ir-types/src/expr.rs`에 추가한다: `eval`은
  `f64::sqrt`다; 음수이거나 유한하지 않은 입력은 다른 모든 비유한값과 마찬가지로 `None`이다
  (결코 `NaN`이 아니다). 이 enum을 매치하는 모든 곳(정규 바이트, 해싱, display, 함수를
  노출한다면 파서)이 그 arm을 얻는다; doc comment는 이것이 `DET-010`의 초월함수가 *아닌* 이유를
  적는다(IEEE 754는 올바르게 반올림된 제곱근을 요구한다 — 하드웨어 명령 하나로, 어디서나 비트
  단위로 동일하다; §6.6). 다른 새 variant는 없다.
* **`Source::Xpos { row, axis }`**를 `crates/es-env/src/plan.rs`에 추가한다:
  `state.xpos[env * nbody * 3 + row * 3 + axis]`를 읽는다, `row`는 `ModelInfo::body`에서 온다
  (모델이 인덱싱하지 않는 몸체 → 그 몸체를 이름 짓는 `EnvError::Unsupported`). `xpos`는 이미
  `StateView`(`crates/es-physics-core/src/backend.rs`) 위에 있고 모든 백엔드가 그것을 채운다.
* **레인.** `lower`는 레인마다 `Expr` 하나를 반환한다: 스칼라 잎은 레인 하나다;
  `GetBodyPose { relative_to: World }`는 레인 세 개다(위치만 — 방향, 그리고 그 외 모든
  `Frame`은 요청받은 것을 이름 짓는 `Unsupported`다); 레인 수가 같은 두 `Arith` 피연산자는
  레인별로 계산되고, 레인 하나짜리 피연산자는 브로드캐스트된다; `Norm { kind: L2 }`는 `n`개의
  레인을 레인 순서대로(DET-020) 합산한 `Sqrt(l₀² + l₁² + …)`로 접는다 — 그 외의
  `NormKind`는 이름으로 `Unsupported`다; `Compare`, `Clamp`, `Normalize`, `Logic`과 모든 싱크는
  정확히 레인 하나를 요구하며, 벡터가 그것들에 닿으면 조용히 첫 레인을 쓰는 대신 이름 붙은
  오류가 된다. 단위 규칙은 오늘 이 파일이 적용하는 것 그대로다(데모 작업의 헤더가 그것들을
  기록한다: `Mul`의 rhs는 무차원, 등).
* **커밋된 것은 아무것도 움직이지 않는다.** 데모 `task.toml`의 보상과 종료 콘은 바이트 단위로
  동일한 `Expr`로 로워링된다 — 무엇에도 손대기 전에 그것들을 고정하고(로워링된 계획의
  canonical/debug 형태, 또는 픽스처 백엔드의 보상 시퀀스) 이후에 단언하라.
* **reach 문서들**(`rl-continuation.md` 5절, 2026-09-21 개정), `tests/fixtures/rl/` 아래:
  `task-reach.toml`(장면 `tests/fixtures/mjcf/so101_pick_place.xml`, 데모의 에피소드별 큐브
  무작위화 재사용; `ObservationSpec` 채널 `joint_pos[6]`, `joint_vel[6]`, `cube_pose[7]`,
  `gripper_pose[7]`는 데모가 `sim_cube_pose`를 바인딩하는 방식으로 바인딩됨; 보상은 스텝마다
  `−‖cube_pos − gripper_pos‖` + 성공 시 `1`; `Terminate`는 거리 `< 0.03` m에서 성공, 타임아웃
  200 제어 스텝); `observation-reach.toml`(`StateInput` ×4 → `Concat` →
  `Normalize{Range}` identity 모양, 26폭, `task_ref` = reach 작업의 해시);
  `deployment-reach.toml`(`action.horizon = 1`, `execute_chunk = 1`, `rate.inference =
  rate.control`를 가진 데모 배포, 무작위 정책이 래치되지 않고 clamp되도록 충분히 넓은
  envelope — INV-12: 넓히기만, 결코 비활성화하지 않기); `evaluation-reach.toml`(홀드아웃 시드
  16개, `nominal` + 상태 정책에 적용되는 데모 스위트들, `success_rate ≥ 0.8` 수용 기준). 각
  문서의 헤더 주석은 데모의 것들이 그러하듯 그것이 무엇인지 말한다.

## context

```
crates/es-ir-types/src/expr.rs
crates/es-ir-types/tests/**
crates/es-env/src/plan.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/task-reach.toml
tests/fixtures/rl/observation-reach.toml
tests/fixtures/rl/deployment-reach.toml
tests/fixtures/rl/evaluation-reach.toml
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M8/S4d-reward-cone.md
docs/packets/M8/S4d-reward-cone.ko.md
```

`expr.rs`(그 variant와 그 arm들), `plan.rs`(그 source, 그 레인들, 두 노드 arm), `env.rs`는
계획이 몸체 테이블을 통과시켜야 할 때에**만**, `es-env/tests`(오라클 2–3), `cli.rs`(오라클 4),
네 문서, 그 노트(6절이 콘을 나열한다), 이 패킷.

## 오라클

1. `cargo test -p es-ir-types expr_sqrt_is_ieee_and_refuses_negative` — `Sqrt(Const(2.0))`는
   정확히 `2f64.sqrt()`로 평가된다; `Sqrt(Const(-1.0))`은 `None`이다; `Sqrt`가 없는 식의 정규
   바이트는 바뀌지 않는다(하나를 고정하라).
2. `cargo test -p es-env body_norm_cone_lowers_and_the_demo_task_is_unmoved` — 테스트가 그
   `xpos`를 설정하는 두-몸체 토이 모델을 가진 픽스처 백엔드 위에서: 보상
   `−Norm{L2}(GetBodyPose(a) − GetBodyPose(b))`는 `-((dx*dx + dy*dy) + dz*dz).sqrt()`와
   **비트 단위로**(그 정확한 결합 순서로) 같다; `Compare`로 들어가는 벡터는 이름으로 거절된다;
   커밋된 데모 작업의 로워링된 콘은 고정된 형태와 같다.
3. `cargo test -p es-env reach_task_executes -- --ignored`(`ES_PYTHON`, MuJoCo): 그 장면 위의
   `task-reach.toml` — `reset` 이후, `Env::step`의 보상은 테스트가 백엔드 자신의 `StateView`로부터
   계산한 `−‖xpos[cube] − xpos[gripper]‖`와 같다(비트 단위); 그리퍼 몸체가 큐브로부터 0.03 m
   이내에 있도록 상태를 설정하면(`supports_state_get_set`), 다음 스텝은 성공으로 종료된다.
4. `cargo test -p es --test cli reach_documents_validate` — 네 문서가 파싱되고, 검증되고,
   교차 검증되며(`XIR_*`), `es task compile`이 그 작업을 받아들이고, 관측의 포트는 26폭이다.
5. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test
   -p es-ir-types -p es-env -p es`, `cargo xtask check-scope docs/packets/M8/S4d-reward-cone.md`.

## 수용 기준

오라클 1–5(3은 서버에서). `batch-domains.md` 6절이 새 잎, 레인 규칙, §6.6 논거와 함께
`Sqrt`를 나열한다; 한국어 자매 문서 갱신.

## 금지

libm을 거치거나 새 `es-math` 함수를 통한 어떤 초월함수든(`sqrt`가 유일한 새 연산이고, 그것은
`f64::sqrt`다); 두 번째 `Expr` variant; 기존 콘이 로워링되는 결과를 바꾸는 것; `es-eval`,
`es-py`, `es-safety`를 건드리는 것; `docs/ARCHITECTURE*.md`(그 문장은 이미 거기 있다);
`docs/design/rl-continuation.md`(S4b가 동시에 그 7절을 쓴다); `tests/golden/**`; INV-17.
