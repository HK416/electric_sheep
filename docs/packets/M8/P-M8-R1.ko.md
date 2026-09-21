# M8 R1 — 탐색 잡음: 레시피만 움직일 때 클램프율과 reach 곡선이 하는 일

스펙: §13.4(`es-env`를 통한 롤아웃, 플레인은 켜져 있음; 샘플링된 ≠ 실행된 비율이 보고된다),
§28.9 규칙 3(한 번에 변수 하나), §28.12 파동 0, §12.4. 리뷰: `docs/reviews/M8.md`의 S-3와
"RL 롤아웃에서의 플레인"이라는 사람의 결정 — 오너는(2026-09-22) 추정량, 엔벨로프, 증분 행동
사이에서 결정하기 전에 트레이너의 잡음을 먼저 측정하기로 했다. 유형 D: 코드 없음. 설계 노트:
`docs/design/rl-continuation.md` 7절이 표를 얻는다.

## 질문

S4b와 S4e는 모든 실행의 모든 틱에서 `executed_ne_sampled_rate = 1.00`을 측정했다:
`init_log_std = −0.5`일 때 가우시안의 σ ≈ 0.6 rad/50 Hz 틱은 30 rad/s의 명령 속도이고
플레인은 구조상 그것을 클램프한다; 그러면 엔트로피 보너스가 σ를 키우는 대가를 치르고
(4.87 → 6.06) reach 실행은 4,000 반복 정점 이후 쇠퇴한다. **플레인과 엔벨로프를 건드리지
않은 채, `init_log_std`, 엔트로피 계수, 학습률 스케줄만 움직일 때 클램프율은 얼마나 떨어지고
곡선은 유지되는가?**

## 사양

`tests/fixtures/rl/training-reach.toml`에서 파생된 레시피(`tests/fixtures/rl/noise/` 아래
커밋되고, 헤더 주석이 각각 무엇을 바꾸는지 말한다), 시드 0, `envs = 16`, `horizon = 64`,
`es eval run --config tests/fixtures/rl/evaluation-reach.toml`로 4,000과 10,000 반복에서
평가됨(둘 다에서 체크포인트):

| id | `[rl] init_log_std` | `[rl] entropy` | `[run] schedule` |
|---|---|---|---|
| A0(S4e, 인용됨) | −0.5 | 0.005 | constant |
| A1 | −2.5 | 0.005 | constant |
| A2 | −2.5 | 0.0 | constant |
| A3 | −2.5 | 0.0 | `warmup_cosine`(워밍업 100, `lr_min` = lr/100) |
| A4 | −1.5 | 0.0 | A3과 같음 |

10,000 반복에서 홀드아웃 `success_rate` 기준 A1–A4 중 최선은 시드 1로 재실행한다. 행마다
기록: 반복 1, 4,000, 10,000에서의 `executed_ne_sampled_rate`와 `envelope_violation_rate`
(`metrics/loss-curve.json`에서); 같은 지점에서의 롤아웃 엔트로피; 4,000과 10,000에서의
홀드아웃 `success_rate`와 `episode_length`; 벽시계; `Rollout.metrics()`가 보고하는 그대로의
아홉 §12.4 지표, 나머지는 `Target / Status: unverified`로. 표 하나, 어느 변수가 어느 숫자를
움직였는지 말하는 단락 하나.

## context

```
tests/fixtures/rl/noise/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/P-M8-R1.md
docs/packets/M8/P-M8-R1.ko.md
```

## 오라클

1. 서버: 위의 실행들(`~/artifacts/plan-t/r1/<id>-seed<n>/`), 서버·날짜·경로가 담긴 표;
   `cargo test -p es --test cli train_rl_dry_run_plan`이 여전히 green(골든이 움직이지 않음
   — 노이즈 레시피는 plan 골든에 추가되지 않는다).
2. `cargo xtask check-spec-refs`; `cargo xtask check-scope docs/packets/M8/P-M8-R1.md`.

## 수용 기준

7절의 표에서 모든 칸이 측정되거나 이유와 함께 `unverified`이고, 변수마다 문장 하나씩. A0의
0.5625를 이기거나 4,000을 넘어 유지되는 변형이 하나도 없다면, 표가 그렇게 말한다 — 측정값이
산출물이다.

## 금지

어떤 코드 변경이든(`train_ppo.py`, `es-env`, `es-safety`, `deployment-reach.toml`의 엔벨로프);
Evaluation IR을 바꾸는 것; 위 표를 넘어서 행마다 변수 하나 초과.
