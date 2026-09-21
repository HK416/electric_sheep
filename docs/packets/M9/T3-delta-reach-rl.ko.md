# M9 T3 — 증분 공간에서 PPO로 하는 reach, 최선의 절대-공간 실행 곁에서

스펙: §13.4, §28.9 규칙 3(변수 하나), §28.12 파동 2, §12.4. **T1**, **T2**(증분 배포와
시작점이 될 증분 번들), **P-M8-R1**(최선의 절대-공간 레시피, 대조 행)에 의존한다. 작은
Rust/Python 표면 하나를 가진 유형 D: `Rollout`이 적분한다면(T1) `train_ppo.py`는 새로 필요한
것이 없다; 레시피가 증분 번들을 이름 짓는다.

## 질문

**같은 예산, 시드, Evaluation IR 아래에서, 증분 공간의 PPO는 R1이 찾은 최선의 절대-공간
레시피보다 덜 클램프하고 더 높은 점수를 받는가 — 그리고 T2에서 가져온 증분 정책으로부터의
이어하기는 S4c의 절대값 것이 아무것도 하지 않았던 곳에서 무언가를 하는가?**

## 사양

행, 각각 시드 셋, 4,000과 10,000 반복, `evaluation-reach.toml`(선언된 지연시간은
`es eval run`을 통해 적용된다):

| 행 | 번들 | 레시피 |
|---|---|---|
| 절대, R1의 최선 | `learning-reach.toml` 그래프, `deployment-reach.toml` | R1의 최선(이미 측정된 곳은 인용) |
| 증분, 처음부터 | 같은 그래프, `deployment-reach-delta.toml` | R1의 최선의 `[rl]` 값 |
| 증분, `[init]` = T2의 가져오기 | T2가 가져온 번들 | 동일 |

행마다: 홀드아웃 `success_rate`(평균; 최소 / 최대), `episode_length`,
`executed_ne_sampled_rate`, `envelope_violation_rate`, 롤아웃 엔트로피, 첫 반복의 액터
그래디언트 노름(S4f/R7이 안착했다면 S-13의 탐지기, 아니면 `unverified`), 벽시계, 아홉 지표.
시드 0에서 절대 행과 증분 행 사이의 `es eval compare`.

## context

```
tests/fixtures/rl/training-reach-delta.toml
tests/fixtures/rl/training-reach-delta-continued.toml
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M9/T3-delta-reach-rl.md
docs/packets/M9/T3-delta-reach-rl.ko.md
```

## 오라클

1. `cargo test -p es --test cli train_rl_delta_dry_run_plan` — plan 골든(추가분).
2. 서버: 표, 모든 칸이 측정되거나 이유와 함께 `unverified`; `~/artifacts/plan-t/t3/` 아래의
   산출물.
3. `cargo xtask check-spec-refs`; `check-scope`.

## 수용 기준

7절의 표와 `visible-learning.md`의 행 7.35, 그리고 M9 리뷰가 필요로 하는 문장: RL을 위한
§13.4의 기본 행동 공간이 증분이 되어야 하는가.

## 금지

행 사이에서 Evaluation IR, 엔벨로프, 트레이너를 바꾸는 것; 절대 행과 증분 행 사이에서 변수
하나 초과(같은 `[rl]` 값, 같은 그래프, 같은 시드).
