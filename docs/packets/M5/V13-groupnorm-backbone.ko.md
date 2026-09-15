# M5 V13 — 학습 모드가 없는, 처음부터 학습하는 backbone

설계 노트: `docs/design/visible-learning.ko.md` **7.21절**, 그리고 6.1(학습 루프), 7.19(V11,
비교 기준), 7.9(V2b, `pretrained`가 거부되게 된 곳). 로워링 계약:
`docs/design/learning-lowering.ko.md` **5.1절**, 이 패킷이 쓰는 규칙. 스펙: §5.2, §8.3, §8.7,
§1.4. 의존: V2, V2b, V6b, V11.

## 이 패킷이 고치는 발견

V11 체크포인트의 개루프 분석(`target/plan-v/v11/openloop.txt`, `bnrecal.txt`)이 **배포된 함수가
학습된 함수가 아니다**를 찾았고, 이유까지 지목했다.

`lower_to_torch`는 `VisionEncoder { ResNet18, pretrained: false }`를 torchvision의
`resnet18()`로 내리는데, 그 기본 정규화 층이 `BatchNorm2d`다 — 20개. 이 로워링은 단일
샘플이다: spec 8.3의 포트에는 배치 축이 없고(spec 5.2가 추론 도메인에 자기 배치 크기를 준다),
이미지는 `.unsqueeze(0)`을, 헤드는 `.reshape(16, 6)`을 거친다. 그래서
`python/es/train_act.py --batch 8`은 *단일 샘플* forward 여덟 번을 누적하고, 모든
`BatchNorm2d`가 **N = 1** 통계를 적합한다 — 배치 카운터가 달린 instance normalization이다.
그 뒤 `crates/es-policy/python/torch_ref.py:85`가 `model.eval()`을 호출해 running 평균을 끼워
넣는다. V11 자신의 체크포인트를 자기 학습 에피소드 위에서 측정한 10행 chunk L1:

| | chunk L1 |
|---|---|
| `train()` — 학습이 최소화한 것, loss 곡선이 보고한 것 | **0.011** |
| `eval()` — `es eval run`이 실행한 것 | **0.031 – 0.039** |
| 학습 세트 전체를 배치 64로 돌려 running 통계를 재보정한 `eval()` | 0.029 |
| 기준선: 모든 행에서 현재 자세 유지 | 0.048 |

재보정 행이 핵심이다. 가중치 후처리가 할 수 있는 최선이고, 간격을 닫지 못한다. 고칠 자리는
로워링이다.

## 결정

**`pretrained: false`에 대해 backbone을 `norm_layer = lambda c: nn.GroupNorm(32, c)`로 내린다.**

GroupNorm은 눈앞의 샘플 하나를 채널 그룹 단위로 정규화한다. `training` 분기도 running 버퍼도
없으므로 `train()`과 `eval()`이 비트 단위로 같은 함수이고, 단일 샘플 로워링은 그대로 유효하며,
running 통계 버퍼가 가중치 계약에서 빠진다. Diffusion Policy가 정확히 이 이유로 ResNet에 끼워
넣는 선택이다. 그룹 32는 ResNet의 모든 stage 폭(64, 128, 256, 512)을 나눈다.

이것은 IR 변경이 아니라 **로워링** 결정이다: `lowering_hash`와 `compiler_hash`가 움직이고
`learning_hash`는 움직이지 않는다. 스펙 §8.3은 `VisionEncoder`의 정규화를 못 박지 않고,
`docs/design/learning-lowering.ko.md`도 이 패킷 전에는 못 박은 바 없다 — 이제 5.1절이다.
`pretrained: true`는 건드리지 않는다: 이 로워링에서 여전히 거부되고(7.6절, 열린 질문 6),
LeRobot 체크포인트는 `lerobot.rs`를 통해 ImageNet의 BatchNorm을 `FrozenBatchNorm2d`로
유지한다(V8). 고정된 affine 상수에도 학습 모드가 없으므로 그쪽이 맞다.

선언된 가중치 키는 움직이지 않는다 — backbone은 접두사 claim `nodes.0.*`이다. 바뀌는 것은 claim
*아래의* 텐서 집합이다: `running_mean`/`running_var`/`num_batches_tracked` 버퍼 60개가 사라진다.

## 범위

* **`context`** — `crates/es-policy/src/lower/torch.rs`, `crates/es-policy/tests/ir_training.rs`,
  `docs/design/learning-lowering*.md` 5.1절,
  `docs/design/visible-learning*.md` 7.21절과 열린 질문 19,
  `docs/packets/M5/V13-groupnorm-backbone*.md`.
* **`forbidden`** — `crates/es-policy/src/lerobot.rs`(`pretrained: true` / LeRobot 경로는 V8의
  것이고 `FrozenBatchNorm2d`가 맞다), `es-ir`과 스펙 §8.3(스키마 변경 없음),
  `nn.TransformerEncoderLayer`의 dropout(의도된 정규화, 추론에서 꺼짐),
  `tests/fixtures/**`, `docs/ARCHITECTURE*.md`, 7.20절(V12가 쓰고 있다).
* **`INV-17`** — 새 트레이트 없음. 기존 헬퍼에 인자 하나.

## 오라클

1. **Python 불필요.** `cargo test -p es-policy --lib
   lower::torch::tests::a_from_scratch_backbone_has_no_training_mode` — 처음부터 학습하는
   `VisionEncoder`의 생성 소스에 `BatchNorm`이 문자열로도 전혀 없고,
   `norm_layer=lambda c: nn.GroupNorm(32, c)`를 담으며, 선언된 어떤 가중치 키나 shape도 running
   통계를 이름으로 갖지 않는다.
2. **torch 필요.** `cargo test -p es-policy --test ir_training -- --ignored
   the_backbone_computes_the_same_function_in_train_and_eval` — 데모 번들의 로워링된 모듈을 순수
   인터프리터에서 인스턴스화하고, `state_dict()`에 running 통계 버퍼가 없음을 단언하며, 고정
   입력에서 `torch.equal(train()의 backbone(x), eval()의 backbone(x))`를 단언한다. `BatchNorm2d`가
   위반한 바로 그 성질이다. 모듈 전체가 아니라 backbone에 대해 단언하는 이유는
   `nn.TransformerEncoderLayer`의 dropout이 두 번째, *의도된* train/eval 차이이기 때문이다.
   torch가 없으면 이유를 출력하고 SKIP.
3. **다른 것이 깨지지 않았다.** `cargo xtask ci`; 서버에서 `ES_PYTHON`과 함께
   `cargo test -p es-policy --test torch_equivalence`, `--test ir_training -- --ignored`.
4. **체크포인트 자체.** 서버의 `identity.py`가 새로 초기화한 모듈이 아니라 *학습된* V13
   체크포인트에 대해 오라클 2를 반복하고, dropout 차이와 정규화 차이를 분리한다.
5. **End to end.** V11의 baked 세트 위에서 V11의 레시피를 그대로 재학습 — 움직인 변수는 로워링
   하나뿐 — 한 뒤 개루프 L1 표와, 학습 시드(1–16)·홀드아웃 시드(101–116)에 대한 `es eval run`을
   V11의 0/16·0/16 옆에 놓는다.
6. **중단 규칙.** 변수 하나. 여전히 0점이면 그대로 보고하고 이 패킷에서 두 번째 변경에 손대지
   않는다.

## 측정

오라클 서버(RTX 4090, 다른 에이전트의 학습과 공유), `es`는 `~/venvs/es`, 학습은
`~/venvs/es-lerobot-cuda`, 2026-09-16. 아티팩트는 `~/artifacts/plan-v/v13/`, 작은 결과는
`target/plan-v/v13/`로 미러.

### 1 — 로워링과 두 오라클

`cargo xtask ci` 통과. 오라클 1과 `torch_equivalence` / `ir_training` 스위트 통과. 서버에서
*학습된* V13 체크포인트에 대한 `identity.py`(`target/plan-v/v13/identity.txt`):

```
running-statistic buffers in the module: 0 []
BatchNorm in the generated source: 0
contract keys with running statistics: 0
max |train() - eval()| over the BACKBONE alone = 0
max |train() - eval()| over the whole module, dropout active   = 0.0168887
max |train() - eval()| over the whole module, dropout disabled = 0
```

가운데 행은 전부 dropout이다: `nn.TransformerEncoderLayer`의 `nn.Dropout` 멤버 셋 *그리고*
`MultiheadAttention.dropout` — 모듈이 아니라 `self.training`을 읽는 float이라서
`isinstance(m, nn.Dropout)` 스윕만으로는 9.3e-4가 남는다 — 을 모두 0으로 두면 모듈 전체가 두
모드에서 비트 단위로 같다.

`lowering_hash`는 `fdd68ec4…` → `70a8fec7…`로 움직였고, `learning_hash` `5dac0a46…`,
`observation_hash` `4b069f6c…`, `task_hash` `d7a7c061…`는 움직이지 않았다. 선언된 14개
가중치 키는 V11과 바이트 단위로 같고, 체크포인트는 **146개에서 86개 텐서**로 줄었다(running
통계 버퍼 60개 = 20 × 3).

### 2 — 개루프 적합도, V11 자신의 학습 에피소드 셋

| 10행 chunk L1 | V11 (BatchNorm) | **V13 (GroupNorm)** |
|---|---|---|
| `eval()` — 추론 경로 | 0.0313 / 0.0373 / 0.0385 | **0.0132 / 0.0135 / 0.0133** |
| `train()` — loss 곡선이 보고한 것 | 0.0107 / 0.0122 / 0.0111 | 0.0136 / 0.0139 / 0.0139 |
| 기준선: 현재 자세 유지 | 0.0484 / 0.0488 / 0.0481 | 0.0484 / 0.0488 / 0.0481 |
| 학습 loss, 초기 → 최종 | 0.0565 → 0.0132 | 0.0607 → **0.0147** |

`eval()`이 2.4배 내려와 보고된 loss와 일치하고(0.0147에 대해 0.0133), 세 배가 아니게 됐다.
이제 `train()` 쪽이 아주 약간 더 나쁜데 이는 dropout이 예측하는 바다. 간격은 닫혔다.

### 3 — 폐루프

`es eval run --frames --jobs 6`, V11의 스위트, V11의 baked 세트, 변수 하나만 이동.

| | V11 | **V13** |
|---|---|---|
| `success_rate`, 학습 시드 1–16 | 0 / 16 | **0 / 16** |
| `success_rate`, 홀드아웃 시드 101–116 | 0 / 16 | **0 / 16** |
| `envelope_violation_rate`, 학습 / 홀드아웃 | 0.3152 / 0.2735 | 0.5770 / 0.6244 |
| `episode_length` | 900 | 900 |
| **큐브가 조금이라도 움직인 에피소드** | **0 / 16, 0 / 16** | **14 / 16, 14 / 16** |
| 홀드아웃 큐브 변위 | 16개 전부 0.4 mm(정착) | 0.4 – 200.2 mm, 중앙값 약 9 mm |

V11의 정책은 큐브에 도달한 적이 없었다 — 0.4 mm는 리셋 때의 정착이다. V13의 정책은 두 스위트
모두 16개 중 14개에서 도달해 큐브를 밀고, 두 번은 190 mm를 넘긴다. 실패의 종류가 바뀌었다:
"도달하지 못한다"가 "도달하고 손을 닫지 못한다"가 됐다.

### 4 — V12가 다시 수집한 세트에서: 테이블을 떠난 첫 큐브

V12의 페어링 수정 50 Hz bake(`~/artifacts/plan-v/v12/baked`, 50 에피소드, 9,038 프레임, 같은 네
포트)가 준비돼 있어서 같은 모듈과 같은 하이퍼파라미터로 여기에도 학습했다(loss 0.0632 →
**0.0162**; 개루프 `eval()` 0.0137 / 0.0144 / 0.0149 대 `train()` 0.0142 / 0.0148 / 0.0154 —
같은 수렴이고, "현재 자세 유지" 기준선이 0.0484가 아니라 0.0565인 세트에서다. 페어링 수정이
목표를 실제로 더 어렵게 만들었다는 뜻이다).

| 학습 데이터 | `success_rate` 학습 1–16 | 홀드아웃 101–116 | `envelope_violation_rate` | `episode_length` |
|---|---|---|---|---|
| V11의 bake | 0 / 16 | 0 / 16 | 0.5770 / 0.6244 | 900 / 900 |
| **V12의 bake** | **1 / 16** | 0 / 16 | 0.5396 / 0.6392 | 847.75 / 900 |

저 `1 / 16`을 읽기 전에 궤적을 먼저 읽어야 한다. **학습 시드 세 에피소드(03, 11, 15)에서 정책은
큐브를 잡고 118 – 124 mm 들어 올려 통까지 옮긴다** — 큐브 (0.25, 0.00, 0.02) →
(0.13, −0.10, 0.06 … 0.14) — 그리고 900스텝 예산이 다할 때까지 그 자리에서 들고 있어서 셋 다
`timeout`으로 채점된다. 하네스가 실제로 `success`로 채점한 한 에피소드(02, 틱 64에서 종료)는
큐브를 1.6 mm 들었을 뿐 운반이 아니고, 따로 들여다볼 문제이지 이 패킷이 주장하는 바가 아니다.
주장은 세 번의 운반이다: 플랜 V에서 IR 소유 비전 정책이 큐브를 집어 든 첫 사례다.
`es video showcase --cell nominal-03`이 그중 하나를 1280x720, 900틱으로 렌더한다:
`~/artifacts/plan-v/v13/v12/v13-carry-nominal-03-h264.mp4`(0.9 MiB). 홀드아웃에서 성공한
에피소드는 없으므로 홀드아웃 쇼케이스는 없다.

### 판정

학습/배포 간극은 실재했고, 측정됐고, 닫혔다: backbone에 대해 `train()`과 `eval()` 차이 0,
`eval()` 개루프 L1이 2.4배 내려와 보고된 loss와 만난다. V11의 bake에서 `success_rate`는 0/16
그대로였지만 실패의 종류가 바뀌었다 — 팔이 큐브를 스쳐 지나가는 대신 도달한다. V12의 bake에서는
학습 시드 16개 중 3개에서 잡고, 들고, 운반하고, 놓지는 못한다. 홀드아웃은 여전히 0/16이므로
중단 규칙이 발동하고 이 패킷은 두 번째 변수를 움직이지 않는다.

수치가 다음으로 가리키는 것은 도달이 아니라 **놓기**다: 관절 5(그리퍼)가 모든 개루프 표에서
가장 큰 잔차를 갖는다 — 블렌드 0.0162 대 팔 관절 0.0032 – 0.0103, 현재 자세 대비 0.0823 —
그리고 "큐브를 통 위까지 옮겨놓고 열지 않는다"가 정확히 그것이 예측하는 바다. 두 번째로 볼 것은
900스텝 예산이다: 세 에피소드는 예산이 다할 때 큐브를 올바른 자리에서 들고 있었다. 다음 작업의
비교 기준은 V11이 아니라 V13의 수치다.
