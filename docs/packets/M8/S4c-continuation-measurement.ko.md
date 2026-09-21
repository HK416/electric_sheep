# M8 S4c — 측정: 이어하기 전후의 가져온 정책, 표 하나

스펙: §28.11 파동 3(유형 D), §13.3(평가는 고정된다; 데이터와 정책만 움직인다), §10.5
(`es eval compare`), §28.9 규칙 2–3, §12.4. **S2b**(가져온 번들), **S1**, **S4b**에 의존.
설계 노트: `docs/design/rl-continuation.md` 7절; `docs/design/visible-learning.md`의 한
행(7.34, "RL 트랙")이 거기를 가리킨다.

## 질문

**우리 시뮬레이션에서 PPO로 가져온 정책을 이어 학습하면 같은 Evaluation IR 위에서 성공률이
오르는가 — 그리고 같은 예산으로 처음부터 학습한 것 대비 얼마나?**

## 사양

하나의 Evaluation IR(`tests/fixtures/rl/evaluation-reach.toml`: 홀드아웃 시드 16개,
nominal + 상태 정책에 적용되는 데모의 perturbation 스위트들), 하나의 장면, 하나의
Deployment IR(선언된 지연시간이 적용된다 — `rl-continuation.md` 3절). 행들, 학습이 관여할
때마다 학습 시드 3개(평균과 최소/최대, §28.9 "신뢰구간"):

| 행 | 정책 | 무엇을 보이는가 |
|---|---|---|
| source | S2c 정책에 대한 brax 자신의 평가(`eval_brax_so101.py`) | 소스 프레임워크의 수치 |
| imported | S2b의 번들, 학습 없음 | 정직한 sim-to-sim 수치 |
| continued | `[init] = imported`, 예산 B의 PPO | 캠페인의 주장 |
| from scratch | 예산 B의 PPO, 무작위 초기화 | 대조군 |
| expert | reach를 위한 스크립트화된 전문가가 있다면 그것; 없으면 생략하고 그렇게 적음 | 하네스 검사(§28.9 규칙 1) |

예산 B는 S4b의 측정된 수렴으로부터 정해진다(§28.11의 오너 결정 3). 표 안의 모든 실행의
`policy_hash`, `training_hash`, `evaluation_hash`, `execution_hash`; 그 옆의
`es eval compare imported.json continued.json` 출력.

## context

```
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
tests/fixtures/rl/**
docs/packets/M8/S4c-continuation-measurement.md
docs/packets/M8/S4c-continuation-measurement.ko.md
```

## 오라클

표는 모든 셀이 측정된 수치(서버, 날짜, `~/artifacts/plan-s/s4c/` 아래의 경로) 아니면 이유와
함께 `Target / Status: unverified`를 갖고 존재한다; 해시가 사슬을 이룬다(모든 행에 걸쳐 같은
`evaluation_hash`); `cargo xtask check-scope`.

## 수용 기준

표, compare 출력, 그것이 말하는 것에 대한 한 문단 — 이어하기가 도움이 되지 않았더라도 포함.

## 금지

행들 사이에서 Evaluation IR을 바꾸는 것(§13.3); 해시 없이 수치를 인용하는 것; 어떤 학습 쪽
변경(S4b가 트레이너를 소유한다).
