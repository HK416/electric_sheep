# M11 R10 — 특권 크리틱을 쓴 픽셀 PPO, 그리고 ImageNet 인코더 갈래

스펙: §13.4, §19.3(`training/`, `base_model.lock`), §28.14 규칙 1과 7, INV-12. 설계 노트
`rl-continuation.md` 규칙 1–3과 7절(X7). 오케스트레이터가 2026-09-26에, X7의 손실 곡선을 상태 행과
대조해 읽다가 찾은 것:

| run (seed 0) | value loss, iter 0 → 1,000 → end | entropy, iter 0 → end | return, iter 0 → end |
|---|---|---|---|
| 상태 reach (S4e, 4,000 it) | 8.4 → 0.57 → 0.40 | 5.52 → 4.87 | −15.8 → −4.7 |
| `rs-pix` (X7, 4,000 it) | 0.42 → 2.3 → 29.7 | 5.52 → 12.64 | −8.9 → −17.6 |

상태 행의 가치 함수는 적합하고, 엔트로피는 내려간다. 픽셀 행의 것은 적합하지 않고, 엔트로피는 첫
반복부터 오른다. 신호를 전혀 담지 않는 어드밴티지 아래서, 엔트로피 보너스가 `log_std`에 걸리는
유일하게 일관된 그레이디언트다. 원인은 엔벨로프가 아니라 레시피에 있다. X7 아래서 `train_ppo.py`의
`Value`는 평탄화된 관측 위의 64-64 tanh MLP라서, `-pix` 행들에서는 그 첫 레이어가 27,648개의 원시
픽셀을 읽는다(그 자체의 독스트링이 그렇게 말하며, 바로 이 경우를 위한 `ponytail:` 메모를 남겨
둔다). 액터는 그 어드밴티지로부터 배우는 from-scratch ResNet18이다. 같은 엔벨로프가 상태 행을 매
스텝 클램프하는데도(`executed_ne_sampled_rate` 1.0), 그 행은 0.42까지 학습한다. 그러니 S-1은 픽셀
행을 설명하지 않는다.

문헌이 제시하는 해법은 비대칭 액터-크리틱이다(Pinto et al., RSS 2018, arXiv:1710.06542). 액터는
픽셀을 보고, 크리틱은 학습만이 가진 시뮬레이터 상태를 본다. MuJoCo Playground의 vision
`PandaPickCubeCartesian`(arXiv:2502.08844)은 X5와 같은 랜덤화 그룹으로 이를 쓰고, Isaac Lab의
rsl_rl도 `critic` 관측 그룹을 통해 같은 일을 한다. 크리틱은 배포되지 않으므로, 이는
`rl-continuation.md` 규칙 1 안에 그대로 머문다: 가치 네트워크는 학습 전용 상태이고,
`training_hash`만 움직이며 `learning_hash`는 결코 움직이지 않는다. 유형 C (코드 + 측정).

## 질문

**특권 시뮬레이터 상태를 읽는 크리틱을 두면, 픽셀만 보는 PPO가 래스터라이저 위에서 reach 과제를
배우는가? 얼린 ImageNet ResNet18 액터 인코더는 from-scratch 인코더보다 더 빨리 배우는가?**

## 명세

### code

* 학습 레시피(`crates/es-data/src/training.rs`)의 **`[rl] critic`**, 모든 면에서 선례인 `[rl]
  estimator`를 따라: `observation | privileged` enum. 부재는 `observation`을 뜻하며, 이는 오늘의
  동작이고, 직렬화되지 않으므로 커밋된 `training_hash`나 플랜 골든은 어느 것도 움직이지 않는다.
  `train_ppo.py`에는 `--critic privileged`로 전달되고, `optimizer.json`(또는 이미 `estimator`를
  담고 있는 `training/` 파일이 무엇이든)이 그것을 기록한다.
* **`train_ppo.py --critic privileged`**: `Value` MLP(같은 64짜리 은닉층 두 개)는 **이미지가
  아닌** 모든 관측 포트와 각 env의 `Rollout.qpos(env)`를 이어붙인 것을 읽는다. 여기에는 큐브의
  자유 관절이 포함되며, 이것이 특권적인 부분이다. `--critic observation`은 오늘의 코드 경로와
  바이트 단위로 같다. 이미지 포트는 특권 크리틱에 결코 닿지 않는다. 가치 네트워크는 여전히
  `--value-out`에 쓰이며 번들에는 결코 들어가지 않는다. `Rollout`에 크리틱이 필요로 하는
  것(예를 들어 `qvel`)이 없다면 추가하지 않는다. `qpos`에 상태 포트를 더한 것이 설계다.
* **RL 경로의 `--init-backbone`**: `[rl]`을 쓴 `es train`이 `[policy] base_model`을 거부하거나
  `train_ppo.py`가 그것을 불러오지 않는다면, `train_act.py`가 하는 방식대로 배선한다(그 함수를
  재사용하고, 두 번째 사본은 만들지 않는다), `base_model.lock`은 `train_act.py`에서처럼 쓴다.
  이미 되고 있다면, 그렇다고 말하고 아무것도 바꾸지 않는다.
* **문서:** `tests/fixtures/rl/learning-reach-vision-pix-imagenet.toml`, 곧
  `learning-reach-vision-pix.toml`에 `pretrained = true, frozen = true`를 더한 것이며, 손으로
  고치지 않고 다른 문서들과 같은 방식으로 생성한다. ImageNet 입력 정규화가 `Normalized{0,1}`
  포트와 사전학습 백본 사이에 아직 적용되어 있지 않다면, 그렇다고 보고하고 그 갈래를 멈춘다. 이
  패킷에서 IR에 정규화를 추가하지 않는다. 또한 `tests/fixtures/rl/` 아래 갈래별 학습 레시피도
  있으며, 이는 X7의 `rs-pix` 레시피에 `critic = "privileged"`를 더한 것이다(ImageNet 갈래에는
  `base_model`도).

### measurement (오라클 서버, GPU 스테이지당 GPU 락, 마커 파일을 쓰는 `nohup`, 산출물은
`~/artifacts/plan-x/r10/` 아래)

규칙은 무엇이든 돌기 전에 여기에 적어 둔다.

* **1단계, 스크리닝:** `rs-pix`, 시드 0, 1,000회 반복, `--device cuda`.
  * 갈래 P는 from-scratch 인코더를 쓴 특권 크리틱이다.
  * 갈래 PI는 얼린 ImageNet 인코더를 쓴 특권 크리틱이다.
  * 기준선은 X7의 `rs-pix-s0` 곡선의 처음 1,000회 반복이며, 이미 존재하므로 다시 돌리지 않는다.
  * 갈래는 900–999회 반복 구간의 평균 리턴이 같은 구간에서의 기준선 평균을 그 구간의 두
    표준편차 중 큰 쪽보다 더 크게 웃돌면 **통과**한다.
  * 통과한 갈래 중 평균이 더 높은 쪽이 2단계로 간다. 둘이 서로 표준편차 하나 이내라면, 반복당
    더 싼 갈래가 간다. 어느 갈래도 통과하지 못하면 멈춘다: 그것이 결과이며, 다음 패킷은
    state-to-pixel DAgger(arXiv:2412.13662)다.
* **2단계, 행:** 선택된 갈래 × 시드 {0, 1, 2} × `rs-pix`와 `rs-dr-pix`, 4,000회 반복, 그 외에는
  X7의 레시피 그대로.
  * Held-out: `evaluation-reach-vision-rs-pix.toml`과
    `evaluation-reach-vision-rs-dr-pix.toml`에서 그 섭동 스위트와 함께 시드 16개.
  * 실행별 보고: 스위트별 성공률, 최종 가치 손실, 최종 엔트로피, 리턴, 벽시계, 그리고 9지표
    집합.
* **예산:** 총 30 GPU-시간. 추정치가 이를 넘으면, 다음 단계 전에 실행을 멈춘다. `Pt` 행은 이
  패킷에 속하지 않는다: `rs`가 배운다면 `Pt`를 돌릴지는 소유자의 결정이다.

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_ppo.py
python/es/train_act.py
python/es/selfcheck.py
tests/fixtures/rl/**
tests/golden/train/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M11/P-M11-R10-vision-rl-critic.md
docs/packets/M11/P-M11-R10-vision-rl-critic.ko.md
```

## 오라클

1. `cargo test -p es-data`: `critic`이 없는 레시피는 오늘의 것과 바이트 단위로 동일하게
   직렬화되고, `critic = "privileged"`는 `training_hash`를 움직인다. 기존 플랜 골든은 바뀌지
   않으며, 새 골든은 추가로만 생긴다.
2. `cargo test -p es --test cli`는 CPU 테스트 하나를 얻는다: `critic = "privileged"`를 쓴 작은
   `-pix` 레시피에서 2회 반복으로 돈 `es train`을 두 번 돌리면 체크포인트와 가치 파일이 비트
   단위로 같다. 가치 네트워크의 첫 레이어 너비는 `non-image state width + nq`이며, 테스트는
   이를 가치 safetensors에서 읽는다.
3. `train_rl_two_runs_are_bitwise`와 그 외 기존 RL 테스트는 모두 바뀌지 않은 채 통과한다
   (`observation` 경로는 오늘의 바이트 그대로다).
4. 1단계 표, 그다음 2단계 표(또는 중단, 규칙의 수치와 함께).
5. `ES_PYTHON`을 설정한 채 `cargo xtask ci`, 수정 0건의 verify-goldens, 그리고
   `cargo xtask check-scope docs/packets/M11/P-M11-R10-vision-rl-critic.md --base main`
   (P-M11-R9가 병합된 뒤; 그 전에는 diff를 손으로 확인한다).

## 수용

오라클 1–5. `rl-continuation.md`(+ko) 7절에 두 표와 질문마다 한 문장의 답과 함께 R10이 생긴다.

## 금지

`crates/es-safety` 안의 무엇이든, 또는 어떤 배포 문서의 엔벨로프든(S-1은 소유자의 결정이다).
하나의 `-imagenet` 문서를 넘어선 배포된 액터의 아키텍처. 갈래별 PPO 하이퍼파라미터 재튜닝. 2단계에서의
단일 시드 주장. `Pt` 실행. 서버에서의 `git clone`/`pull`(scp만). 푸시.
