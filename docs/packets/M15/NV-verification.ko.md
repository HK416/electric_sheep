<!-- Korean translation of docs/packets/M15/NV-verification.md. The English file is the working copy; regenerate this when it changes. -->
# M15 N-V — 플랜 N의 실험(멀티 카메라)

플랜 N의 E1과 E2를 이 PC(Windows 11, RTX 3060 12 GB, RAM 64 GB)에서, 2026-09-29/30, 실행마다
에디터를 열어 둔 채로 돌렸다. 문서: `tests/fixtures/visible-learning/*-views*`, `*-cam*`,
`*-mad*`(N1, N8b); 씬 `tests/fixtures/mjcf/so101_pick_place_views.xml`(overhead, SO-101의
`camera_mount`에 단 wrist, side).

## 1. 카메라가 보는 것(N1)

리셋 자세에서 팔은 테이블 위로 쭉 뻗어 있다. 커밋된 **overhead** 카메라는 고정한 큐브 위치 다섯 곳에서
큐브 픽셀을 0–29개 보고, 시연자의 접근 웨이포인트에서는 다섯 곳 모두 0개를 본다(그리퍼가 큐브를
가린다). **wrist** 카메라는 접근 웨이포인트에서 큐브 픽셀 448–470개를, **side** 카메라는 리셋에서
66–107개를 본다. 플랜 N 이전 모든 실행의 유일한 카메라였던 overhead 카메라는 로봇이 큐브로 손을 뻗는
동안 큐브를 거의 보지 못한다.

## 2. 발산과 그 원인(P-M15-R2)

첫 E1 학습은 카메라만 쓰는 그래프에서 U3의 설정에서도, 더 순한 설정에서도 매번 NaN이 됐다.

| graph | settings | NaN at step |
|---|---|---|
| one view | lr 4e-4 | 3,189 |
| three views | lr 4e-4 | 774 |
| three views | lr 4e-4, grad_clip 1.0 | 778 |
| three views | lr 1e-4, grad_clip 1.0 | 7,928 |

조사 결과 원인은 데이터나 학습률이 아니라 수치 계산에 있었다. 낮춘 `TemporalEncoder{Transformer}`는
**토큰 하나**짜리 시퀀스를 돌리므로 softmax는 정확히 1이고 q/k 프로젝션의 그래디언트는 정확히 0이다.
그런데 이 GPU에서 torch의 memory-efficient SDPA 커널은 backward에서 dq/dk에 반올림 잡음을
돌려준다(로짓 1에서 2.9e-5, 1e6에서 0.03, 1e9에서 NaN). forward는 정확하다. AdamW는 이 잡음을 lr
크기의 걸음으로 바꾸고, `Wq`/`Wk`가 표류하며(|Wq| 16 → 41), 로짓이 지수적으로 커지다(0.27 → 3e9)
손실이 NaN이 된다. 힌트 카드의 그래프도 같은 방향으로 표류한다. 60,000걸음짜리 두 실행은 문턱에
닿기 전에 끝났을 뿐이다(|Wq| 16 → 39, 16 → 25). Q5의 실행은 ~1,130걸음에서, M7의 U1은 4,517걸음에서
NaN이 됐다(`docs/design/visible-learning.md` 열린 질문 35, 이제 답이 났다). 수정은 CUDA에서 정확한
(math) 어텐션으로 학습하는 것이다(`python/es/train_act.py`, `train_ppo.py`). 그러면 dq/dk가 정확히 0이
되며 CUDA 테스트가 이를 고정한다. 해시는 움직이지 않았고, 추론은 영향을 받은 적이 없다(forward는
정확하다). 이후 E1은 U3 자신의 설정 — lr 4e-4, 클리핑 없음 — 으로 발산 없이 돌았다.

## 3. E1 — 뷰 하나 대 셋, 같은 데이터, 같은 설정

수집은 한 번: 전문가 시연 200개(1초 대기 시연 프로그램), 63,901프레임, 세 카메라가 모두 렌더링했다
(200/200 성공; 전문가 게이트 통과). 두 팔 모두 카메라만 쓰는 IR 경로(`sim_cube_pose`가 없는 U3의
그래프; `Concat` 융합; 뷰마다 ImageNet ResNet18 하나)를 lr 4e-4, 배치 64, warmup-cosine으로 60,000걸음
학습한다. 뷰 하나 팔은 같은 수집물의 `rgb_overhead`만 읽는다. 같은 스위트와 시드로 평가했다.

| training seed 0 | one view (overhead) | three views |
|---|---|---|
| training speed | ~1,100 samples/s | ~440 samples/s (60,000걸음에 2 h 25 min) |
| nominal | 0/16 | 1/16 |
| all suites | 3/96 | 8/96 |
| attempts that lifted the cube | **5/96** | **30/96** |
| ended with the cube in the bin / held over it | 2 / 1 | 8 / 14 |
| `envelope_violation_rate`, nominal | 0.011 | 0.022 |

성공 횟수는 학습 한 번씩이 갖는 실행 간 산포 안에 있다(M10 리뷰 S-4). 그러나 큐브를 들어 올린 횟수는
가깝지 않다. 세 뷰 정책은 큐브를 여섯 배 자주 들었고, 이는 1절의 픽셀 수가 예측하는 바다. 놓는 일이
여전히 병목이다(14번의 시도가 큐브를 빈 위에 쥔 채 끝났다). M14의 실행과 같다. 두 팔의 학습 시드 1:
4절.

## 4. E1, 학습 시드 1

같은 두 레시피를 `seed = 1`로, 같은 수집물에서 돌렸다(2026-09-30, 밤새):

| | nominal | all suites | lifted | ended in the bin / held over it |
|---|---|---|---|---|
| one view, seed 0 | 0/16 | 3/96 | 5/96 | 2 / 1 |
| one view, seed 1 | 0/16 | 1/96 | 6/96 | 1 / 2 |
| three views, seed 0 | 1/16 | 8/96 | 30/96 | 8 / 14 |
| three views, seed 1 | **7/16** | **21/96** | 30/96 | 23 / 2 |

**E1의 답: 두 번째와 세 번째 뷰는 두 시드 모두에서 도움이 된다.** 세 뷰 정책은 큐브를 다섯~여섯 배
자주 들어 올리고(30 대 5–6) 더 자주 성공한다(8과 21 대 3과 1). 시드 1의 nominal 7/16(0.44)은 이
프로젝트가 잰 카메라만 쓰는 수치 중 가장 좋으며, 데모의 합격선 0.5에 성공 한 번이 모자란다. 같은 팔의
두 시드는 성공 8 대 21로 갈린다. M10 리뷰가 잰 실행 간 산포가 여기서도 성립하므로, 성공 횟수에는
시드가 필요하다. 들어 올린 횟수는 안정적이었다(30과 30; 5와 6).

## 5. E2 — MAD, 더 적은 카메라에 배치

MAD 그래프(`learning-mad.toml`: 세 뷰가 공유하는 ResNet18 하나, `Sum` 융합, 이후는 동일),
`single_view = { weight = 0.5 }`, U3의 설정, 60,000걸음(N7b의 인코더 한 번 통과 단일 뷰 스텝으로 전체
실행 2 h 47 min), 시드 0, E1의 수집물에서. 그다음 `es policy subset`으로 각 단일 뷰에 잘라(같은 가중치)
각각을 유도한 Evaluation IR로 판정했다(N8c — 같은 스위트, 시드, 메트릭, 합격 기준):

| the MAD policy, evaluated with | nominal | all suites | lifted | ended in the bin / held over it |
|---|---|---|---|---|
| all three cameras | 4/16 | 12/96 | **41/96** | **29** / 10 |
| **the wrist camera alone** | **2/16** | 5/96 | 11/96 | 8 / 3 |
| the overhead camera alone | 0/16 | 1/96 | 4/96 | 1 / 1 |
| the side camera alone | 0/16 | 5/96 | 12/96 | 8 / 1 |

- 모든 카메라를 쓰면 다른 어느 실행보다 큐브를 자주 들어 올리고 빈에 넣었다(96번 중 41과 29).
- **재학습 없이 카메라 하나에 배치해도 어느 정도는 동작한다.** wrist 하나 — 실제 로봇의 저렴한 구성 —
  는 nominal 2/16에 닿았고, 이는 overhead 카메라 하나로 학습한 두 정책(0/16, 0/16)보다 높다.
  overhead 하나는 그 전용 정책들과 맞먹었다(1/96 대 3/96과 1/96). 이것이 MAD의 주장이며, 이
  프로젝트에서 모방 학습으로는 처음 확인했다. 학습이 한 번이므로 4절의 산포 안에 있다.
- 카메라 하나로 줄이는 것은 공짜가 아니다: 카메라 셋이면 성공 12, 가장 좋은 하나이면 5.

## 6. 플랜 N의 실험 이후 남은 것

- 뷰가 늘면 잡기가 결정적으로 좋아진다. overhead 카메라 하나는 이 과제에 나쁜 센서다.
- 놓는 일이 여전히 가장 약한 부분이다(M14의 H-2): 모든 실행이 일부 시도를 큐브가 빈 위나 안에 있고
  그리퍼가 닫힌 채로 끝낸다.
- 성공률은 비교를 주장하기 전에 학습 시드를 여럿 써야 한다.
