# M9 R5 — 트레이너는 플레인이 실행한 행동으로부터 학습한다 (소유자 결정 B, 2026-09-22)

스펙: §13.4(`es-env`를 통한 롤아웃, 플레인은 켜져 있음; 오늘은 "정책의 샘플이 행동이고, 플레인의
출력이 액추에이터가 받는 것이며, 둘 다 기록된다" — 이 패킷이 바꾸는 것은 *기울기*가 무엇으로부터
계산되는가이지, 무엇이 기록되는가가 아니다), §9.4, INV-11..13(엔벌로프와 플레인은 바뀌지 않는다),
§28.9 규칙 3(변수 하나), §12.4. 리뷰: `docs/reviews/M9.md`의 S-7과 "§13.4의 기본 행동 공간"이라는
사람의 결정 — 오너는 **B**를 택했다: 엔벌로프는 선언된 대로 남고, 트레이너는 실행된 행동을 쓴다.
설계 노트: `docs/design/rl-continuation.md` 2절(샘플링 모델)이 규칙을 얻는다; 7절이 행들을 얻는다.

## 질문

지금까지 모든 RL 실행의 모든 롤아웃 틱이 클램프되었으므로(`executed_ne_sampled_rate = 1.00`),
PPO의 기울기는 env가 결코 실행하지 않은 행동의 로그 확률로부터 계산되어 왔고, 더 작은 잡음
(P-M8-R1)도 증분 공간(T3)도 그것을 바꾸지 못했다. **트레이너가 플레인을 환경의 일부로 다루어 —
버퍼 속 행동이 플레인이 실행한 바로 그것이고, 로그 확률이 그 행동에서 평가된다면 — 엔벌로프를
손대지 않은 채로 reach 정책이 4,000 반복 정점을 넘어 학습하는가?**

## 사양

* `train_ppo.py --estimator executed`(기본값은 `sampled`로 남으므로 커밋된 모든 레시피와 그
  `training_hash`는 움직이지 않는다; 레시피 안의 `[rl] estimator = "executed"`는
  `deny_unknown_fields`이고 `training/config.json`에, 따라서 `training_hash`에 기록된다):
  롤아웃에서 스텝의 행동으로 `executed`(`Rollout.act`에서)를 저장하고, 현재 가우시안 아래
  **실행된 행동에서** `logp`를 계산한다(저장된 옛 로그 확률과 업데이트의 비율 둘 다); 보상,
  done, 가치, GAE는 바뀌지 않는다. `executed_ne_sampled_rate`는 계속 플레인의 클램프율을
  보고한다 — 이는 엔벌로프에 대한 사실이지, 추정량에 대한 사실이 아니다.
* `es-env`, `es-py`, `es-safety`, Deployment IR 어디에도 바뀌는 것이 없다; `deployment-reach.toml`의
  플레인 엔벌로프는 이전의 모든 행이 썼던 바로 그것이다.
* 레시피: `tests/fixtures/rl/training-reach-executed.toml` = `training-reach.toml` + 한 줄;
  `--dry-run` plan 골든은 추가분으로.

## context

```
python/es/train_ppo.py
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/training-reach-executed.toml
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M9/P-M9-R5.md
docs/packets/M9/P-M9-R5.ko.md
```

## 오라클

1. `cargo test -p es --test cli train_rl_estimator_dry_run_plan` — plan 골든(추가분);
   `estimator = "sampled"`를 명시적으로 쓴 것은 부재와 같은 해시를 갖는다; 알 수 없는 값은
   이름으로 거절된다; `training-reach.toml`의 `training_hash`는 움직이지 않는다.
2. `cargo test -p es --test cli train_rl_two_runs_are_bitwise`(`ES_PYTHON`)도
   `estimator = "executed"`로: 두 실행이 비트 단위로 같다.
3. 서버: `training-reach-executed.toml`, 시드 0/1/2, 반복 4,000과 10,000, `evaluation-reach.toml`로
   평가; T3의 절대 A0 행(4,000에서 0.4167; 10,000 시드 0에서 0.3125) 곁에 표 하나로 —
   `success_rate`, `episode_length`, `executed_ne_sampled_rate`, 롤아웃 엔트로피, `return`,
   벽시계, 아홉 지표. 다음 결정이 필요로 하는 문장: 곡선이 4,000을 넘어 유지되는가, 그리고
   시드 세 개에서 10,000에 A0를 이기는가?
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M9/P-M9-R5.md`.

## 수용 기준

오라클 1–4; 2절의 규칙과 7절의 행(+ `.ko.md`). B가 도움이 되지 않으면 행이 그렇게 말하고, 다음
결정(서보 사양 근거가 있는 A, 또는 C)은 오너의 것이다.

## 금지

엔벌로프, `es-safety`, `es-env`의 적분기, 또는 `Rollout`에 대한 어떤 변경이든(INV-11..13; 변수
하나); `executed`를 기본값으로 만드는 것(레시피에서 보이는 선택, 해시됨); pickle; 새 trait.
