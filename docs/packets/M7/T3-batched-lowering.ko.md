# M7 T3 — 배치 축: `lower_to_torch`가 `[N, …]`을 내고, `train_act.py`가 진짜 배치로 학습한다

스펙: §8.7(lowering), §5.2(추론 도메인은 자신만의 배치 크기를 갖는다 — IR은 아무것도
선언하지 않는다), §8.9(tier-4 동등성), §1.4, §28.9 사다리 9단("배치 lowering 안의
샘플별 루프를 제거한다 — 학습 시간의 근본 원인"), §28.10(T3). 수정할 설계 노트:
`docs/design/learning-lowering.md`(+ `.ko.md`) — 2, 3, 5.1절과 새 5.2절 "배치 축".
선행: M5 V2(`train_act.py`), V5(속도 플래그 세 개가 아무것도 움직이지 않았다: 배치 8에서도
모듈이 샘플 하나씩 돈다), V13(GroupNorm; `train()==eval()`이 오라클).

## the question

`--batch 8`은 단일 샘플 forward 여덟 번을 누적한 것이다(`train_act.py`의 독스트링, "the
lowered module is single-sample"): `VisionEncoder`는 `self.n{k}(x.unsqueeze(0)).squeeze(0)`로
lowering되고, 회귀 헤드는 `.reshape(horizon, action_dim)`으로, 청커는 `[:execute_chunk]`로
lowering된다. 20,000스텝은 RTX 4090에서 약 11분이 걸리고 어떤 플래그도 그것을 움직이지
않는다(설계 노트 7.11절). **lowering된 모듈이 `[N, …]`을 받고 트레이너가 `N`을 한 번에
먹이면, 시간은 어디로 가고 함수는 그대로인가?**

## spec

* `lower_to_torch`는 모든 입력을 **선행 배치 축**과 함께 받고 모든 출력도 그것과 함께
  돌려주는 `forward(**inputs)`를 가진 모듈을 낸다: 이미지 포트 `[N, C, H, W]`, 상태
  `[N, D]`, 노이즈 `[N, …]`, 청크 `[N, K, A]`. 노드별로: `VisionEncoder` → `self.n{k}(x)`
  (unsqueeze 없음); `token_count == 0`인 `TemporalEncoder{Transformer}` → `x.unsqueeze(1)`
  … `.squeeze(1)`(토큰 하나짜리 *시퀀스*, 배치가 앞선다 — 예전에는 가짜였던 축이 이제는
  진짜다); `PolicyHead{Regression}` → `.reshape(-1, horizon, action_dim)`; `ActionChunker` →
  `[:, :execute_chunk]`; `Normalizer`는 오늘처럼 브로드캐스트한다; `Diffusion` /
  `FlowMatching` 샘플러는 배치된 상태(`noise.reshape(N, -1)`)로 스텝을 순회하며, 스텝별
  산술은 그 밖에는 **식 단위로 바뀌지 않는다**(샘플러 헤드의 tier-4 허용 오차는 `1e-5`이고,
  옮겨진 식은 곧 옮겨진 ULP다). `Fusion{Concat}`은 `dim=-1`을, `TokenConcat`은 `dim=-2`를
  유지한다.
* `contract.json`은 `inputs` 아래에 IR의 샘플별 shape을 그대로 두고 `"batch_axis": true`를
  더한다(그 키를 모르는 리더는 모듈을 단일 샘플로 취급해 shape에서 요란하게 실패하는데,
  그것이 옳은 실패다).
* `crates/es-policy/python/torch_ref.py`(`TorchRuntime` 쪽)는 `[1, …]`을 먹이고 반환하기
  전에 모든 출력에서 배치 축을 squeeze한다 — 추론은 샘플 하나로 남는다, §5.2에 따라 추론
  도메인의 배치는 모듈이 아니라 런타임의 일이기 때문이다. LeRobot 패스스루
  (`lerobot.rs`, `PolicyBundle`)는 손대지 않는다: 이 lowering을 거친 적이 없다.
* `python/es/train_act.py`: `--batch N`이 **진짜 배치**가 된다 — `N`개의 샘플을 dim 0을
  따라 쌓아 forward 한 번, `[N, K, A]`에 대한 `L1` 평균 — 옵티마이저도, 시드 처리도, 샘플
  순서도 그대로다(순열은 오늘처럼 에포크마다 한 번 뽑히므로, 배치 `i`는 누적 루프가 방문했을
  바로 그 여덟 샘플을 담는다). `--resident-gpu`, `--amp`, `--compile`, `--channel-weight`,
  `--checkpoint-at`, `--loss-curve`는 의미를 유지한다; 독스트링의 "single-sample" 문단들을
  갱신한다. JSON 요약은 `"batch_axis": true`를 얻는다.
* `probe` 텐서를 짓거나 출력을 인덱싱하는 `crates/es-policy/tests/ir_training.rs`와
  `crates/es/tests/cli.rs`의 테스트들은 축에 맞춰 갱신된다(그 수정은 범위 안이다);
  `act_checkpoint` 오라클(LeRobot 경로)은 건드리지 않으며 여전히 `max_abs 0e0`을 읽어야
  한다.
* 서버 측정(수용): V15의 bake된 세트(`~/artifacts/plan-v/v15/baked`, 또는 `ds-train`에서
  다시 bake)에서 20,000스텝, 배치 8, 시드 0, `--resident-gpu`로 이전(main)과 이후(이
  브랜치)를 벽시계 시간과 `final_loss`로 나란히; 그다음 같은 lr에서 배치 64를 관측으로만
  (T4가 스케줄을 담당한다 — 튜닝하지 말 것).

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-policy/src/lower/torch.rs
crates/es-policy/src/lower/mod.rs
crates/es-policy/src/torch_runtime.rs
crates/es-policy/python/torch_ref.py
crates/es-policy/tests/*.rs
python/es/train_act.py
crates/es/tests/cli.rs
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/packets/M7/T3-batched-lowering.md
docs/packets/M7/T3-batched-lowering.ko.md
```

`crates/es-policy/src/lower/torch.rs`, `crates/es-policy/src/lower/mod.rs`,
`crates/es-policy/src/torch_runtime.rs`(계약 읽기가 그 플래그를 필요로 할 때만),
`crates/es-policy/python/torch_ref.py`, `crates/es-policy/tests/ir_training.rs`, lowering된
소스 텍스트를 단언하는 `crates/es-policy/tests/*.rs`, `python/es/train_act.py`,
`crates/es/tests/cli.rs`(모듈을 shape-probe하는 기존 테스트; 새 파일 없음),
`docs/design/learning-lowering*.md`, `docs/packets/M7/T3-batched-lowering*.md`.
`lowering_hash`는 움직인다(소스가 움직였으므로); `learning_hash`는 움직이지 않는다; 둘 다
설계 노트에 기록한다.

## oracle

1. `cargo test -p es-policy --lib lower::torch::tests::the_module_has_a_batch_axis` — ACT
   모양 픽스처와 Diffusion, FlowMatching 픽스처 각각에 대해, 생성된 소스가
   `.unsqueeze(0)` / `.squeeze(0)` 배치 관용구를 담지 않고, 헤드는 선행 `-1`로 reshape하며,
   청커는 `[:, :K]`로 슬라이스하고, `contract.json`이 `batch_axis: true`라고 말한다. Python
   불필요.
2. `cargo test -p es-policy --test ir_training -- --ignored the_batch_is_invariant` —
   torch로: 서로 다른 여덟 입력을 데모 번들의 lowering된 모듈에 하나의 `[8, …]` 배치로
   통과시킨 것과 여덟 번의 `[1, …]` 호출로 통과시킨 것이 CPU에서
   `Tolerance::TIER4_FP32`(`1e-5`) 이내로 같은 출력을 내며, 최대 차이가 출력된다
   (GroupNorm과 Linear는 샘플별이라 작아야 한다; 실제 값을 보고할 것).
3. `cargo test -p es-policy --test ir_training -- --ignored the_backbone_computes_the_same_function_in_train_and_eval`
   — V13의 오라클, 변경 없음, 배치된 모듈에서도 여전히 통과한다.
4. `cargo test -p es-policy --test ir_training -- --ignored act_training_uses_baked_observations
   resident_gpu_does_not_move_the_loss channel_weight_of_one_is_the_unweighted_loss` — 기존
   학습 오라클들이 진짜 배치로도 통과한다(합산 순서가 움직였으니 loss 숫자는 움직일 수 있다:
   그것은 예상된 일이고 설계 노트가 그렇게 말한다; 그것들이 단언하는 *속성*은 유지되어야
   한다).
5. 서버에서 `ES_ACT_CHECKPOINT=… cargo test -p es-policy --test act_checkpoint` —
   `max_abs 0e0`, LeRobot 경로는 손대지 않는다.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T3-batched-lowering.md`.

## acceptance

오라클 1–6 통과(2–5는 오라클 서버에서
`ES_PYTHON=~/venvs/es-lerobot-cuda/bin/python`으로). 전후 표(벽시계 시간, `final_loss`;
`steps/s`는 **보고하지 않는다** — §12.4에 그런 지표는 없다; 옵티마이저 스텝 1,000개당 초와
초당 샘플 수를 보고한다)는 설계 노트 5.2절에 배치-64 관측을 그 아래에 두고 실린다. *이후*
실행의 20k 체크포인트는 U-측정을 위해 `~/artifacts/plan-v/m7-t3/model-20000.safetensors`에
보관한다(여기서 평가하지 않는다 — 그것은 웨이브 5의 몫이다).

## forbidden

`crates/es-ir/**`(IR은 배치 축을 갖지 않는다 — §5.2); `crates/es-policy/src/lerobot.rs`와
LeRobot `act_checkpoint` 경로; `crates/es-safety/**`, `crates/es-eval/**`, `crates/es-env/**`;
축 추가를 넘어서는 샘플러의 스텝별 산술 변경; 학습률 기본값, 옵티마이저, 스케줄(T4);
`pretrained` 처리(T5); 모든 픽스처와 골든; `docs/ARCHITECTURE*.md`. INV-16: safetensors만.
INV-17: 새 트레이트 없음.
