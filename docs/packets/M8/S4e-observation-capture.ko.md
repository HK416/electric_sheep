# M8 S4e — observation 캡처가 관절 속도와 몸체의 포즈를 읽고, reach 실행이 안착한다

스펙: §7.4(`ObservationSpec`가 채널을 선언한다; `ObsSource`가 각각이 어디서 오는지 말한다),
§6.3(`GetJointState`는 `JointQuantity`를 갖는다), §10.1(캡처는 문서들에 대해서만 해석되고,
결코 추측하지 않는다), §3.1(쿼터니언 xyzw), §28.11 파동 3, 패킷 M7/R5의 규칙(새 필드는 부재 =
기본값 = 오늘의 정규 형태). S4d가 찾아냈다(2026-09-21, `tests/fixtures/rl/task-reach.toml`
헤더): `ObsSource`에는 속도 quantity가 없고 `es_eval::runner::input_sources`에는 `qvel`이나
몸체-포즈 읽기가 없어서, reach 관측의 `joint_vel[6]`과 `gripper_pose[7]`은 캡처될 수 없다 —
`es eval run`으로도, 그 코드를 공유하는 `es_native.Rollout`(S4a)으로도. 설계 노트:
`docs/design/evaluation-execution.md` 2.3절 "Observation 캡처"(+ `.ko.md`),
`docs/design/rl-continuation.md` 7절(reach 실행의 행들). 이 패킷의 한국어 자매 문서:
오케스트레이터의 것.

## 질문

**채널이 "이 관절들의 속도"와 "이 몸체의 포즈"를 말할 수 있는가, 그 하나의 캡처 경로가 백엔드
자신의 상태로부터 — `qvel` 행, `xpos ‖ xquat`(xyzw) — 비트 단위로 둘 다를 대접할 수 있는가,
커밋된 모든 문서의 해시와 기존 모든 채널의 바이트가 움직이지 않은 채로, 그리고 그러면 reach
작업이 `es train`의 `[rl]` 경로를 통해 목표까지 학습하는가?**

## 사양

* **`ObsSource::JointState { body, dof, quantity: JointQuantity }`**를 `crates/es-ir/src/task.rs`에
  추가한다, `#[serde(default, skip_serializing_if = …)]`로 `Position`이 기본값이고 정규 바이트는
  `Velocity`에 대해서만 쓰인다(무엇에도 손대기 전에 커밋된 `task_hash eb6efefa…`와 `task-pt`의
  것을 고정하라; `committed_task_hash_is_unmoved_by_sensor_render`가 선례다). `XIR-002`(source
  id당 채널 하나)는 `(id, quantity)`로 키를 잡으므로 위치 채널과 속도 채널이 같은 관절 블록을
  이름 지을 수 있다. 그 채널에 대한 `NodeSchema` 항목이 그 파라미터를 얻는다.
* **캡처**는 `crates/es-eval/src/runner.rs`에서: id가 로드된 모델의 관절인(`ModelInfo::dof`)
  `Velocity` 채널에 대해서는 `Capture::Qvel(range)`, 앞쪽-`dof` 형태(`Joints`의 거울상)에
  대해서는 `Capture::JointsVel(dof)`; `b`가 모델의 자유 관절 몸체가 **아닐** 때
  `ObsSource::BodyPose(b)`에 대해서는 `Capture::BodyPose(row)`(자유 관절 몸체는 오늘의
  `Qpos` arm을 유지하며, 그것이 데모의 `sim_cube_pose`가 쓰는 것이다 — 바이트 불변),
  `StateView`로부터 `xpos[row*3..][..3] ‖ xquat[row*4..][..4]`를 읽어 값 7개, 쿼터니언은
  `StateView`가 문서화하는 순서(xyzw)다. 모델-없는 경로(기록된 프레임, `model == None`)는 두
  새 종류 모두를 이름으로 거절한다 — 데이터셋 행은 그것들을 싣지 않고 아무것도 추측하지 않는다.
* **`Rollout`**(`crates/es-py/src/rollout.rs`)은 같은 `capture`를 호출한다면 바뀔 필요가
  없다; `env_view` / env별 `StateView` 슬라이스가 `xpos`/`xquat`을 싣지 않는다면 그 슬라이스를
  고쳐라.
* **reach 문서들이 재생성된다**(`cargo test -p es --test cli -- --ignored
  regenerate_reach_documents`, S4d의 생성기): `joint_vel` = `JointState { body = base, dof = 6,
  quantity = Velocity }`, `gripper_pose` = `BodyPose(gripper)`; 두 채널에 대한 헤더의 단락은
  이제 참인 것을 말하도록 다시 쓰인다; `observation-reach.toml` / `evaluation-reach.toml`은
  움직인 `task_hash`를 따라간다.
* **reach 실행**(S4b가 미룬 오라클 4): `tests/fixtures/rl/training-reach.toml`(네 reach 문서 +
  그 옆에 추가하는 `learning-reach.toml`로부터 꾸려진 번들 위의 `[rl]`:
  `StateEncoder{Mlp hidden = [64, 64]}` → `PolicyHead{Regression, horizon 1}` →
  액추에이터 `ctrlrange`들 위의 `Normalizer{Inverse}`; `envs = 16`, `horizon = 64`); 서버에서,
  `es eval run`을 통해 `evaluation-reach.toml`에서 `success_rate ≥ 0.8`에 도달하는 예산, 시드
  0 — 또는 도달하지 못한다면 도달한 숫자, 곡선과 함께. `Rollout.metrics()`가 보고하는 그대로의
  아홉 지표로 `rl-continuation.md` 7절에 행을 넣고 나머지는 `Target / Status: unverified`로.

## context

```
crates/es-ir/src/task.rs
crates/es-ir/src/schema.rs
crates/es-ir/src/cross.rs
crates/es-ir/src/**
crates/es-ir/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es-eval/src/bake.rs
crates/es-data/src/roboverse.rs
crates/es-data/tests/**
crates/es-editor/tests/common/mod.rs
crates/es-runtime-embedded/tests/embedded.rs
crates/es-py/src/rollout.rs
crates/es-py/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/**
tests/golden/train/**
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/S4e-observation-capture.md
docs/packets/M8/S4e-observation-capture.ko.md
```

`task.rs`(그 필드, canonical, `es-ir` 안 어디에 있든 `XIR-002`의 키), 스키마,
`runner.rs`(세 `Capture` arm과 그 읽기), `rollout.rs`는 상태 슬라이스에 대해서**만**,
테스트, 재생성된 문서들 + `learning-reach.toml`과 `training-reach.toml`(그리고 `--dry-run`
plan 골든, 추가분), 두 노트, 이 패킷.

## 오라클

1. `cargo test -p es-ir committed_task_hashes_are_unmoved_by_joint_quantity` — 커밋된
   `task.toml`과 `task-pt.toml`의 해시가 고정된 리터럴과 같다; `quantity = "Position"`을
   명시적으로 적은 것은 부재한 것과 같은 해시를 낸다; `Velocity`는 그것을 움직인다; 하나의
   관절 블록 위의 서로 다른 quantity를 가진 두 채널은 `XIR-002`를 통과하고, 같은 quantity를
   두 번 쓰면 여전히 거절된다.
2. `cargo test -p es-eval capture_reads_qvel_and_body_pose` — `qpos`, `qvel`, `xpos`, `xquat`에
   서로 다른 값을 가진 합성 `StateView`: 세 새 arm이 옳은 바이트를 옳은 포트에 안착시키고,
   쿼터니언 순서는 xyzw이며, 모델-없는 경로는 둘 다 이름으로 거절하고, 데모 문서들의 해석은
   달라지지 않는다(커밋된 데모의 `Capture` 맵을 고정하라).
3. `cargo test -p es-py rollout_observes_the_reach_documents -- --ignored`(`ES_PYTHON`, MuJoCo,
   서버): 네 reach 문서 위의 `Rollout`; `reset` 이후 26폭 포트는 테스트 안에서 `StateView`로부터
   읽은 백엔드 자신의 `qpos[0..6]`, `qvel[0..6]`, 큐브 `qpos[6..13]`, 그리퍼 `xpos ‖ xquat`
   (xyzw)와 슬라이스별로 그리고 **비트 단위로** 같다.
4. `cargo test -p es --test cli reach_documents_validate`는 재생성된 문서들 위에서도 여전히
   green이다; `train_rl_dry_run_plan`이 `training-reach.toml`의 plan 골든을 얻는다(추가분).
5. 서버: reach 실행 — `es train --recipe tests/fixtures/rl/training-reach.toml`, 그다음
   `es eval run --config tests/fixtures/rl/evaluation-reach.toml --policy <last checkpoint>`;
   `success_rate`, 예산, 벽시계, 지표들, 곡선, 모두 7절에 서버, 날짜, 경로
   (`~/artifacts/plan-s/s4e/`)와 함께.
6. `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test -p es-ir -p
   es-eval -p es-py -p es`, `cargo xtask verify-goldens`(추가분만), `cargo xtask
   check-scope docs/packets/M8/S4e-observation-capture.md`.

## 수용 기준

오라클 1–6. `evaluation-execution.md` 2.3절이 두 새 캡처와 자유 관절 규칙을 나열한다;
`rl-continuation.md` 7절에 reach 행들이 있다. 목표에 도달하지 못하면, 행들은 무엇이었는지를
말하고, 패킷은 그래도 출하된다 — 측정값이 결과물이지, 숫자가 아니다.

## 금지

기존 채널이 캡처하는 것이나 커밋된 문서의 해시를 바꾸는 것; 문서가 선언하지 않은 채널을
추측하는 것(INV-14의 정신: 이름 붙은 거절, 결코 기본값이 아님); 두 번째 관측-캡처 구현(S4a의
규칙); `python/es/train_ppo.py`와 `crates/es/src/cmd/train.rs`(S4b의 것; 여기서 발견된
트레이너 버그는 보고만 하고 여기서 고치지 않는다); `es-safety`; `docs/ARCHITECTURE*.md`;
`tests/golden/**` 수정; INV-17.
