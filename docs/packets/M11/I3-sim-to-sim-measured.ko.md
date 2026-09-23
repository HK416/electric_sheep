# M11 I3 — sim-to-sim, 측정: Isaac Lab 정책과 우리 정책, 세 엔진에서

사양: §28.14 규칙 2, 3, 6, 7과 wave 3, §3.5 (tier 2 수치는 보고할 뿐 비트 동일을 주장하지 않는다),
§10.4, §12.4. 선행: X1 (+ 후속), X2, I1, P-M11-R1 (모두 병합됨). API 노트: `isaac-lab.md` (reach
env cfg, rsl_rl, exporter), `isaac-sim.md` §7–8. 설계 노트: `rl-continuation.md` (+ko)의 7절에 I3
하위 절이 생긴다. Type D.

## 질문

지금까지의 이어붙이기 수치는 모두 "소스 정책은 우리에게 없는 장면에서 학습되었다"로 끝났다(M8 S-5,
M9 S-8). PhysX 백엔드와 어댑터 v2가 생긴 지금, **우리 장면에서 Isaac Lab으로 학습하고 어댑터 v2로
들여온 SO-101 reach 정책은 우리 런타임에서 Isaac Lab에서와 같은 점수를 내는가 — 그리고 그 정책과
우리 PPO 정책(A0)은 `physx`, `mujoco-cpu`, `mjwarp` 사이에서 어떻게 움직이는가?**

## 사양

* **Isaac Lab 작업** (`python/es/rl_source/isaac_so101_reach/`, manager-based env cfg이며 Isaac Lab의
  포크가 아니다):
  * 장면은 `scene_to_mjcf`가 내보내는 우리 `so101_pick_place.xml`이다. I1의 수정들(같은
    `physx_ref.py` 임포트 함수를 복사하지 않고 재사용)로 임포트하므로, 정책이 학습하는 PhysX 모델이
    `--backend physx`가 평가하는 바로 그 모델이다.
  * 관측 항은 `tests/fixtures/rl/task-reach-last-action.toml`을 그대로 따른다: `joint_pos_rel`,
    `joint_vel_rel`(선언된 scale), 로봇 베이스 좌표계의 큐브 자세와 그리퍼 자세, `last_action`,
    이 순서로.
  * 행동은 `JointPositionActionCfg(scale, use_default_offset=True)`.
  * 제어 주기는 20 ms(50 Hz)이며 `deployment-reach.toml`의 것과 같아야 한다: decimation × dt.
  * 보상은 `-‖cube − gripper‖` + 성공 시 1(< 0.03 m), 에피소드는 거기서 또는 200 제어 스텝에서 끝난다.
  * 리셋은 Task IR의 `ResetState` 노드와 같은 범위에서 뽑는다.
  정확히 따라 할 수 없는 것은 어댑터의 매핑 보고서 곁에 보고서로 나열한다.
* **학습:** Isaac Lab의 rsl_rl PPO, 시드 3개, headless, GPU 파이프라인, Isaac 쪽 성공률이 정체될
  때까지. 예산, 벽시계 시간, 보류된 리셋 시드 집합에서의 Isaac 쪽 성공률을 기록한다.
* **임포트:** `import_rl.py --framework rsl-rl`(Isaac Lab 2.3.2가 쓰는 체크포인트 모양 — 어느 것인지
  기록) → `adapter-isaac-so101.toml`(v2: `source_names`, `default_pos`, 항별 scale,
  `use_default_offset`, `policy_dt`) → `es policy import-rl`.
* **평가**는 `evaluation-reach-last-action.toml`(보류 시드 16개, reach 스위트들)에서 하며, 다른 reach
  문서들처럼 생성한다. Isaac 시드 3개 각각 × {`physx` CPU 파이프라인, `physx` GPU, `mujoco-cpu`,
  `mjwarp`}로 돌린다. A0(시드 3개, W0b의 체크포인트)도 같은 네 열을 받는다.
* **귀속 실행:** Isaac 정책과 A0를 `mujoco-cpu`에서, MappingReport의 근사된 행들을 어댑터가 켜고 끌 수
  있는 곳에서(명시적 effort로서의 조인트 감쇠 대 없음, 마찰 결합 모드) 한 번에 하나씩 바꿔 돌린다.
  엔진 격차의 대부분을 어느 행이 설명하는지 밝힌다. 판정이 아니라 수치를 보고한다.

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

## oracle

1. `cargo test -p es --test cli isaac_so101_documents_check` — 새 문서들이 검증을 통과하고,
   `deployment-reach.toml`과 Cross-IR이 맞으며, 어댑터가 env cfg의 모양을 가진 *합성* 체크포인트를
   임포트한다(Isaac 불필요).
2. 서버: 시드 3개의 Isaac Lab 학습 로그 + 체크포인트(`~/artifacts/plan-x/i3/`); 시드별 Isaac 쪽 보류
   성공률.
3. 서버: 평가 표 — Isaac 시드 0–2와 A0 시드 0–2 각각에 대해 physx-CPU / physx-GPU / mujoco-cpu /
   mjwarp 열: `success_rate`, `envelope_violation_rate`, `episode_length`, `execution_hash`. MJWarp는
   두 번 돌린다(실행마다 재현되지 않는다 — X1), physx도 두 번 돌린다(재현된다).
4. 귀속 행들.
5. fmt, clippy, check-scope, verify-goldens.

## acceptance

오라클 1–5. `rl-continuation.md`(+ko)가 I3 표와, 질문에 수치와 함께 한 문장으로 답하는 문단 하나를
얻는다.

## forbidden

우리 수치와 맞을 때까지 Isaac 작업을 조정하는 것(따라 하고, 그다음 측정한다); 커밋된 해시를 움직이는
것; 새 트레이트; 추론 시의 프레임워크 코드; 허용 오차를 주장하는 것.
