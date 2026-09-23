# M11 I3 — sim-to-sim, 측정됨: Isaac Lab 정책과 우리 자신의 정책, 세 엔진 위에서

스펙: §28.14 규칙 2, 3, 6, 7과 파동 3, §3.5(계층 2 숫자는 보고되며, 결코 비트 단위라고 주장하지
않는다), §10.4, §12.4. X1(+ 후속), X2, I1, P-M11-R1(모두 머지됨)에 의존. API 노트:
`isaac-lab.md`(reach env cfg, rsl_rl, exporter), `isaac-sim.md` §7–8. 설계 노트:
`rl-continuation.md`(+ko) 7절에 I3 하위 절이 생긴다. 유형 D.

## 질문

앞선 모든 continuation 수치는 "소스 정책이 우리가 갖고 있지 않은 장면에서 학습되었다"로 끝났다
(M8 S-5, M9 S-8). PhysX 백엔드와 어댑터 v2가 있으면, **우리 자신의 장면 위에서 Isaac Lab으로
학습되어 어댑터 v2를 거쳐 임포트된 SO-101 reach 정책은 우리 런타임에서 Isaac Lab에서와 같은
점수를 내는가 — 그리고 그 정책과 우리 자신의 PPO 정책(A0)은 `physx`, `mujoco-cpu`, `mjwarp`를
가로질러 어떻게 움직이는가?**

## 명세

* **Isaac Lab 과제**(`python/es/rl_source/isaac_so101_reach/`, manager 기반 env cfg, Isaac Lab의
  포크가 아니다):
  * 장면은 `scene_to_mjcf`가 내는 그대로의 우리 `so101_pick_place.xml`이다. I1의 수정(같은
    `physx_ref.py` 임포트 함수, 재사용하며 복사하지 않음)을 거쳐 임포트되므로, 정책이 학습하는
    PhysX 모델은 `--backend physx`가 평가하는 그 모델이다.
  * 관측 항은 `tests/fixtures/rl/task-reach-last-action.toml`을 그대로 옮긴다: `joint_pos_rel`,
    `joint_vel_rel`(선언된 scale), 로봇 base frame에서의 cube 자세와 gripper 자세, `last_action`,
    이 순서대로.
  * 액션은 `JointPositionActionCfg(scale, use_default_offset=True)`다.
  * 제어 주기는 20 ms(50 Hz)이며 `deployment-reach.toml`의 것 — decimation × dt — 과 같아야
    한다.
  * 보상은 `-‖cube − gripper‖`에 성공 시(< 0.03 m) +1이며, 에피소드는 거기서나 제어 스텝
    200에서 끝난다.
  * 리셋은 Task IR의 `ResetState` 노드와 같은 범위에서 추출한다.
  정확히 옮길 수 없는 것은 무엇이든 어댑터의 매핑 보고서 곁에 보고서로 나열한다.
* **학습:** Isaac Lab에서 rsl_rl PPO, 시드 3개, headless, GPU 파이프라인, Isaac 쪽 성공률이
  정체할 때까지. 예산, wall clock, held-out 리셋 시드 셋에서의 Isaac 쪽 성공률을 기록한다.
* **임포트:** `import_rl.py --framework rsl-rl`(Isaac Lab 2.3.2가 쓰는 체크포인트 형태 — 어느
  것인지 기록) → `adapter-isaac-so101.toml`(v2: `source_names`, `default_pos`, 항별 scale,
  `use_default_offset`, `policy_dt`) → `es policy import-rl`.
* **평가**는 `evaluation-reach-last-action.toml`(held-out 시드 16개, reach 스위트)에서, 다른
  reach 문서처럼 생성해서. Isaac 시드 3개 × {`physx` CPU 파이프라인, `physx` GPU, `mujoco-cpu`,
  `mjwarp`} 각각에서 돌린다. A0(시드 3개, W0b의 체크포인트)도 같은 네 열을 받는다.
* **귀속 실행(attribution run):** Isaac 정책과 A0를 `mujoco-cpu`에서, 어댑터가 토글할 수 있는
  곳에서 MappingReport의 근사된 행을 하나씩 토글하며(joint damping을 explicit effort로 vs 없음,
  friction combine mode) 돌린다. 이는 엔진 격차 대부분을 설명하는 행이 무엇인지 이름 짓는다.
  판정이 아니라 숫자를 보고한다.

## context

```
python/es/rl_source/isaac_so101_reach/**
python/es/import_rl.py
crates/es-physics-backend/python/physx_ref.py
tests/fixtures/rl/adapter-isaac-so101.toml
tests/fixtures/rl/evaluation-reach-last-action.toml
tests/fixtures/rl/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/api-notes/isaac-lab.md
docs/api-notes/isaac-lab.ko.md
docs/packets/M11/I3-sim-to-sim-measured.md
docs/packets/M11/I3-sim-to-sim-measured.ko.md
```

## 오라클

1. `cargo test -p es --test cli isaac_so101_documents_check` — 새 문서가 검증되고,
   `deployment-reach.toml`과 Cross-IR이 맞으며, 어댑터는 env cfg의 형태를 가진 *합성*
   체크포인트를 임포트한다(Isaac 불필요).
2. 서버: 시드 3개의 Isaac Lab 학습 로그 + 체크포인트(`~/artifacts/plan-x/i3/`); 시드별 Isaac
   쪽 held-out 성공률.
3. 서버: 평가 표 — Isaac 시드 0–2와 A0 시드 0–2 각각에서, physx-CPU / physx-GPU / mujoco-cpu /
   mjwarp 열: `success_rate`, `envelope_violation_rate`, `episode_length`, `execution_hash`.
   MJWarp는 두 번 실행(런 간 재현 안 됨 — X1), physx도 두 번(이쪽은 재현됨).
4. 귀속 행.
5. fmt, clippy, check-scope, verify-goldens.

## 수용

오라클 1~5. `rl-continuation.md`(+ko)에 I3 표와, 질문에 한 문장으로 답하는 문단 하나가 숫자와
함께 생긴다.

## 금지

Isaac 과제를 우리 숫자에 맞을 때까지 튜닝하기(그대로 옮기고 나서 측정한다); 커밋된 해시 옮기기;
새 trait; 추론 시점의 프레임워크 코드; 허용오차 주장하기.
