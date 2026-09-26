# M11 R11 — 픽셀 액터의 첫 업데이트, 그리고 그것을 포화시키지 않는 학습률

스펙: §13.4, §19.3(`scheduler.json`, `optimizer.json`), §28.14 규칙 7. 설계 노트
`rl-continuation.md` 7절(X7, R10)과 S4c(포화된 `tanh`). R10의 진단으로 찾은 것(2026-09-26).
`rs-pix`에서 리턴은 반복 0에서 반복 1 사이에 −8.9에서 −16.1로 떨어지며, 이는 X7의 실행과 R10의
실행 모두에서 그렇다. 반복 1,000회에 이르면 헤드의 `tanh` 이전 값 `z`의 모든 채널이 |z| = 3을
넘어서고(평균 약 20), 학습되지 않은 액터는 |z| ≈ 0.24다. 행동은 한쪽 구석에 고정되고, `tanh`를
통과하는 그레이디언트는 1 % 미만이다. 크리틱은 그런 상태의 액터를 움직일 수 없으므로, R10의 1단계
규칙은 그보다 앞서는 질문에서 멈췄다.

레시피는 상태 과제의 것이다: `lr = 3e-4`의 Adam, 4 에폭 × 4 미니배치, 그레이디언트 클리핑 없음.
상태 MLP에서는 이것이 통한다. 11M-파라미터 규모의 from-scratch ResNet18에서는, Adam의 초기
스텝들이 모든 파라미터를 그레이디언트 부호 방향으로 대략 `lr`만큼 움직이므로, 출력은 네트워크
크기에 비례해 움직인다. 이것이 가설이다. 이 패킷은 그것을 측정할 뿐 가정하지 않는다. 유형 D
(측정; 레시피만, 코드 없음).

## 질문

**더 낮은 학습률, 워밍업, 그레이디언트 클리핑 중 무엇이 픽셀 액터의 `tanh`를 첫 20회 반복 동안
포화시키지 않게 하는가? 그 설정과 R10의 특권 크리틱을 쓰면, 픽셀만 보는 PPO가 `rs-pix`를
배우는가?**

## 명세

모든 것은 오라클 서버에서, R10의 코드(`f775681` 또는 그 이후)와 헤드의 `Linear`에 건 `diag.py`
훅으로, GPU 스테이지당 GPU 락을 잡고 돈다. 산출물은 `~/artifacts/plan-x/r11/` 아래로 간다. 모든
갈래는 `critic = "privileged"`를 쓴 `rs-pix`, 시드 0이다. 레시피는 R10의
`training-reach-vision-rs-pix-critic.toml`에서 이름이 명시된 필드만 바꾼 것이다. 이들은
`tests/fixtures/rl/` 아래 커밋되며, 2단계에 도달하는 것만 플랜 골든을 얻는다.

* **A단계, 탐침** (20회 반복, `checkpoint_at = [1, 5, 20]`):

  | 갈래 | R10의 레시피 대비 변경점 |
  |---|---|
  | A0 | 없음(`lr = 3e-4`), 붕괴를 재현해야 하는 대조군 |
  | A1 | `lr = 1e-4` |
  | A2 | `lr = 3e-5` |
  | A3 | `lr = 1e-5` |
  | AW | `lr = 3e-4`, `schedule = { kind = "warmup_cosine", warmup = <100 iterations of optimizer steps>, lr_min = 0 }` |
  | AG | `lr = 3e-4`, `grad_clip = 0.5` |

  체크포인트마다, 시드 0에서 16개 env × 64 스텝에 대해 `diag.py`로 측정한 평균 |z|와 |z| > 3의
  비율, 구간 리턴, 학습 리턴을 기록한다. A0는 R10의 붕괴를 재현해야 하며, 그러지 못하면 그 단계는
  무효이고 보고서가 이유를 말한다.
* **규칙 A (지금 확정):** 반복 20회 시점에 |z| > 3의 비율이 0.05 미만이면 그 갈래는 **안정적**이다.
  안정적인 갈래들 중, 반복 20회 시점의 학습률이 가장 큰 것이 B단계로 간다. 동률이면 더 단순한
  쪽(상수 `lr`이 우선)이 간다. 어느 갈래도 안정적이지 않으면 멈춘다.
* **B단계, 스크리닝:** 선택된 갈래, 1,000회 반복. R10의 규칙이 X7의 `rs-pix-s0`에 대해 900–999회
  반복 구간에서 적용된다: 그 갈래의 평균 리턴이 기준선의 평균을 두 표준편차 중 큰 쪽보다 더 크게
  웃돌면 통과한다. 반복 100회와 1,000회 시점의 |z|도 함께 보고한다. 통과하지 못하면 멈춘다.
* **C단계, 행:** 선택된 갈래 × 시드 {0, 1, 2} × `rs-pix`와 `rs-dr-pix`, 4,000회 반복, 그다음
  `evaluation-reach-vision-rs-pix.toml`과 `evaluation-reach-vision-rs-dr-pix.toml`에 대해 시드
  16개로 held-out 평가를 스위트별로 진행한다. 가치 손실, 엔트로피, 리턴, 최종 체크포인트에서의
  |z|, 벽시계, 9지표 집합을 보고한다.
* **예산:** 25 GPU-시간. 추정치가 이를 넘으면, 다음 단계 전에 실행을 멈춘다.

## context

```
tests/fixtures/rl/**
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/packets/M11/P-M11-R11-first-update-saturation.md
docs/packets/M11/P-M11-R11-first-update-saturation.ko.md
```

## 오라클

1. A단계 표, 그리고 규칙 A의 결정과 그 수치.
2. B단계 표(또는 중단), 그리고 규칙 B의 결정과 그 수치.
3. C단계 표(또는 중단).
4. 레시피가 C단계에 도달하면: `cargo test -p es --test cli`의 dry-run 플랜 골든을 추가로만 얻는다.
   수정 0건의 verify-goldens. `cargo xtask check-scope
   docs/packets/M11/P-M11-R11-first-update-saturation.md --base main`.

## 수용

오라클 1–4. `rl-continuation.md` 7절에 R11이 생긴다(영어로; `.ko.md`는 오케스트레이터가 한다),
질문마다 한 문장의 답과 함께.

## 금지

코드 변경(버그를 찾으면 여기서 고치지 않고 보고한다); 어떤 엔벨로프든 또는 `crates/es-safety`;
`Pt` 행; `lr`, `schedule`, `grad_clip`을 넘어서는 어떤 PPO 필드든; C단계에서의 단일 시드 주장;
서버에서의 `git clone`/`pull`; 푸시.
