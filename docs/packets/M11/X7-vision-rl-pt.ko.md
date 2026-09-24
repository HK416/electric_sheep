# M11 X7 — 경로 추적기 위의 vision RL, 랜덤화를 켠 채

스펙: §28.14 규칙 4, 5, 7과 파동 3, §12.4(9지표 집합), §13.4, §3.5. X3, X3b, X5, X6,
P-M11-R1(`es train`이 이미지 입력을 받는다)에 의존. 설계 노트: `rl-continuation.md` 2c(vision
rollout), `renderer.md` 12.9와 13절(비용). 유형 D.

## 질문

**PPO는 경로 추적된 96×96 픽셀에서 — `seed = "tick"`, SVGF, 모든 시각·물리 랜덤화 타깃을 켠 채
— SO-101 reach 과제를 배우는가? 경로 추적기는 래스터라이저 대비 비용이 얼마이며, 어느 학습
렌더러가 다른 쪽으로 더 잘 전이되는가?**

## 명세

* **문서:** `tests/fixtures/rl/task-reach-vision.toml`(X3)에 변형이 생기며, 다른 문서들과 같은
  방식으로 생성한다:
  * `…-pt-dr.toml`: `Pt`, `seed = "tick"`, 행별 `svgf`, 그리고 X5의 light, colour, ambient,
    geom-rgba, camera-pose, fov 타깃과 X4의 mass/friction/gain 타깃을 중간 범위로(범위는
    보고서에 명시).
  * `…-rs-dr.toml`: 같은 것을 `Rs`로.
  * `…-pt.toml`과 `…-rs.toml`: 랜덤화 끔.
* **1단계, 스윕**(서버, GPU lock, X3b의 배치 렌더): 시드 0에서 1,000회 반복에 대해 `Pt` spp ∈
  {4, 8, 16} × SVGF {off, on}. ms/frame, samples/s(9지표 집합)와 return을 기록한다. wall-clock
  시간당 return이 가장 좋은 행이 학습 설정이 된다. 선택 규칙은 스윕을 돌리기 전에 적어 둔다.
* **2단계, 행**, 각각 시드 3개, 같은 예산: `Pt`+DR, `Pt` no-DR, `Rs`+DR, `Rs` no-DR.
* **평가:** `evaluation-reach-vision-*.toml`에서 평가 섭동 스위트와 함께 held-out 시드 16개.
  Cross-render: `Pt`에서 학습된 정책을 `Rs` 문서에서 평가하고 그 반대도. 이 비교가 "어느
  렌더러가 전이되는가"의 답이다.

**소유자 결정 2026-09-24(재실행).** R3가 학습기를 CUDA로 옮긴 뒤, 재실행은 픽셀만 보는 네 행
(`-pix`, 상태에 `cube_pose` 없음)만 다룬다: `pt-dr-pix`, `pt-pix`, `rs-dr-pix`, `rs-pix`, 각각
시드 3개, 4,000회 반복, `--device cuda`, `Pt` 행은 **4 spp**. `cube_pose`가 든 행은 돌리지 않으며,
교란(confound) 발견으로 보고서에 남는다. 1단계는 `pt-dr-pix` 4 spp에서 SVGF off 대 on(1,000회
반복, 시드 0)으로 줄어든다: 마지막 100회 반복의 평균 return이 높은 쪽이 이기되, 그 차이가 두
실행의 같은 100회 반복 표준편차 중 큰 값보다 작으면 SVGF off(더 싼 쪽)가 이긴다. 4 spp 문서는
커밋된 16 spp 문서 옆의 새 파일이다(`x7_pt_pix_variants`). 예산은 약 50–55 GPU-시간이며,
65시간을 넘는 추정이 나오면 실행을 멈춘다.

## context

```
tests/fixtures/rl/**
tests/golden/train/**
crates/es/tests/cli.rs
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/X7-vision-rl-pt.md
docs/packets/M11/X7-vision-rl-pt.ko.md
```

## 오라클

1. `cargo test -p es --test cli vision_reach_dr_documents_check`와 dry-run plan 골든(추가).
2. 서버 1단계 표(ms/frame, samples/s, 1,000회 반복에서의 return, spp × SVGF별).
3. 서버 2단계 표: 4행 × 시드 3개, 스위트별 held-out `success_rate`, cross-render 열, wall
   clock, 9지표 집합.
4. check-scope, verify-goldens.

## 수용

오라클 1~4; `rl-continuation.md`(+ko) 7절에 X7이 생기며 질문마다 한 문장 답이 붙는다.

## 금지

코드 변경(이것은 측정 패킷이다: 버그를 찾으면 새 패킷); 하나의 레시피를 넘어 행별로 PPO
재튜닝하기; 단일 시드 주장.
