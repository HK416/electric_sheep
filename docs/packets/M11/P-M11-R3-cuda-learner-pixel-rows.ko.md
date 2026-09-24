# M11 R3 — CUDA 위의 PPO 학습기, 그리고 큐브를 픽셀로만 보는 비전 행

스펙: §28.14 규칙 5, 7, §3.5(CUDA 학습기는 1단계가 아니며, 비트 단위로 재현되는 것은 CPU다), §12.4.
X7(2026-09-24)에서 나온 것:

- `python/es/train_ppo.py --device cuda`는 첫 순전파에서 죽는다("mat1 is on cpu"). 액터는 장치로
  옮겨지지만 관측과 롤아웃 버퍼는 옮겨지지 않는다. CPU에서는 ResNet18 갱신이 반복 한 번 23.4 s 중
  21.8 s를 차지해, X7 2단계가 약 305 GPU시간으로 추정된다.
- X7의 비전 행은 `cube_pose`가 든 26폭 `state` 포트를 유지한다. 그래서 PPO가 픽셀로 배우는지를
  가릴 수 없다.

유형 B.

## 질문

**`train_ppo.py`가 같은 레시피로 `--device cuda`에서 학습하는가 — CPU 경로의 바이트는 그대로 둔 채로 —
그리고 그때 X7 `pt-dr` 반복 한 번은 몇 초인가? 또 X7의 각 행이 상태에 큐브 자세 없이, 곧 카메라로만
큐브를 볼 수 있게 존재할 수 있는가?**

## 명세

* `train_ppo.py`: 액터나 가치망과 만나는 텐서는 모두 `device`에 있다. 관측(상태와 이미지 포트), 행동,
  로그확률, 이점, 리턴, 미니배치 인덱스가 여기에 든다. 잡음과 순서 생성기는 CPU에 남고(추출값이 바뀌지
  않는다), 추출한 뒤에 장치로 옮긴다. 디스크에 쓰는 것(체크포인트, 지표)은 먼저 CPU로 옮긴다.
  `--device cpu`는 오늘과 같은 바이트를 낸다.
* X7 문서: 네 행마다 `-pix` 형제를 둔다. 그 관측·학습 문서의 상태 포트는 `cube_pose` 없이 관절과 그리퍼
  자세(19폭)만 쓰고, 이미지 포트는 그대로다. X7의 `regenerate_x7_documents`를 확장해 생성하고, plan
  골든을 추가한다.

## context

```
python/es/train_ppo.py
crates/es/tests/cli.rs
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/P-M11-R3-cuda-learner-pixel-rows.md
docs/packets/M11/P-M11-R3-cuda-learner-pixel-rows.ko.md
```

## 오라클

1. 기존 `train_rl_*` cli 테스트와 PPO 비트 단위 테스트가 바뀌지 않고 통과한다(CPU 바이트 불변).
2. 서버(GPU 락): X7 `pt-dr`과 `pt-dr-pix` 행을 `--device cuda`로 3회 반복해 완료한다. 반복당 초를
   렌더, 롤아웃, 학습기로 나눠 기록한다. 이어서 한 시드로 cuda와 cpu를 각각 20회 반복한다. 리턴 곡선이
   PPO 잡음 안에 있어야 하며, 비트 단위 주장이 아니므로 둘 다 보고한다.
3. `cargo test -p es --test cli vision_reach_dr_documents_check`가 `-pix` 행을 포함한다. 그 dry-run plan
   골든은 추가만 한다.
4. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## 수용

오라클 1–4. `rl-continuation.md`(+ko) X7 절에 측정한 cuda 반복당 초와 새 2단계 추정을 적는다.

## 금지

PPO 레시피나 하이퍼파라미터 변경; 커밋된 학습 골든 변경; es-py나 Rust 런타임 변경(필요하면 새 패킷).
