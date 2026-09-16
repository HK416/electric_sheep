# M7 T4 — 학습률 스케줄, 그리고 발산을 멈춘 배치

스펙: §19.3(`scheduler.json`, `optimizer.json`, `precision.json`는 아이덴티티 슬롯이다),
§8.1(옵티마이저는 IR 안에 있지 않다 — 그것은 트레이너의 것이고, 트레이너의 아이덴티티는
§19.3이다), §28.9 "무엇이 각각을 제거할 것인가"("웜업과 lr 스케줄을 갖춘 큰 배치: 선형
스케일링은 발산하는 것으로 측정되었으니, 필요한 것은 관습이 아니라 스케줄이다"),
§28.10(T4). 수정할 설계 노트: `docs/design/training-recipe.md`(+ `.ko.md`, T1의 것) — "스케줄"
절. **T1**(`training/scheduler.json`이 레시피로부터 쓰인다)과 **T3**(진짜 배치; 누적된 단일
샘플 위의 스케줄은 다른 질문에 답하는 것이다)에 의존한다.

## the question

V5는 `--batch 64 --lr 8e-4`(배치 8 / 1e-4에서의 선형 스케일링)를 측정했다: 1시간 24분 뒤
`final_loss`가 NaN이었다(설계 노트 7.11절). T3와 함께라면 배치 64는 forward 한 번이므로, 큰
배치에 대한 벽시계 논거가 처음으로 진짜가 된다. **웜업과 코사인 감쇠를 아이덴티티로 기록해
두면, 배치 64는 같은 수의 샘플을 본 시점에 배치 8과 같거나 그보다 낮은 loss로 학습되는가?**

## spec

* `python/es/train_act.py`는 `--schedule constant|warmup_cosine`(기본값 `constant` — 측정된
  실행들이 재현 가능하게 남도록), `--warmup-steps N`(기본값 `0`), `--lr-min F`(기본값
  `0.0`), `--weight-decay F`(AdamW의 것, torch 기본값인 `1e-2`를 **명시적으로 만든 것** —
  `optimizer.json`이 그것을 이름 붙일 수 있도록), `--grad-clip F`(기본값 `0` = 꺼짐; 클립된
  실행은 그렇게 말한다)를 얻는다. 스케줄은 `step < warmup`일 때 `lr(step) = lr * step /
  warmup`이고, 그다음은 `lr_min + (lr - lr_min) * 0.5 * (1 + cos(pi * (step - warmup) /
  (total - warmup)))`이다 — 스크립트 안의 평범한 함수 `lr_at(step, total, lr, lr_min,
  warmup) -> float`로 구현되어 매 스텝 호출된다(`torch.optim.lr_scheduler`는 쓰지 않는다:
  그 float 시퀀스는 torch 버전의 구현 세부사항이고, 아래 골든은 우리 것을 고정한다).
* JSON 요약은 `schedule`, `warmup_steps`, `lr_min`, `weight_decay`, `grad_clip`과
  `lr_curve_hash`(실제로 적용된 `f64` lr 값들에 대한, 리틀엔디언 blake3)를 보고한다.
* `es train`(T1)은 레시피의 `[run] schedule = { kind = "warmup_cosine", warmup = 500,
  lr_min = 1e-6 }`를 그대로 통과시켜 `training/scheduler.json`으로 쓰고, `optimizer.json`은
  `weight_decay`와 `grad_clip`을 얻는다. 이들 중 무엇이 바뀌어도 `identity_hash`가
  움직인다.
* 서버 측정(수용, V15의 bake된 세트에서, 달리 언급하지 않는 한 각 20,000 옵티마이저 스텝):
  | run | batch | lr | schedule | steps |
  |---|---|---|---|---|
  | A(기준선, T3) | 8 | 1e-4 | constant | 20,000 |
  | B | 64 | 1e-4 | constant | 2,500(같은 수의 샘플) |
  | C | 64 | 4e-4 | warmup_cosine, warmup 250 | 2,500 |
  | D | 64 | 4e-4 | warmup_cosine, warmup 250 | 20,000 |
  `final_loss`, 벽시계 시간, 어떤 스텝이든 non-finite였는지를 보고한다. D의 체크포인트는
  U-측정을 위해 `~/artifacts/plan-v/m7-t4/model-D.safetensors`에 보관한다. 체크포인트를
  **여기서 평가하지 않는다**.

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
python/es/train_act.py
crates/es-policy/tests/ir_training.rs
tests/golden/train/**
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/training.toml
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M7/T4-lr-schedule.md
docs/packets/M7/T4-lr-schedule.ko.md
```

`python/es/train_act.py`, `crates/es-policy/tests/ir_training.rs`(스케줄 테스트),
`tests/golden/train/lr_warmup_cosine.json`(신규: `total = 20000, lr = 4e-4, lr_min = 1e-6,
warmup = 250`에 대한 `lr_at`의 처음 1,000개 값, `ES_PYTHON`을 통해 스크립트의 함수를 호출하는
`#[ignore]`가 붙은 테스트로 한 번 생성됨; 체크인된 테스트는 Python 출력을 골든과 **그리고**
같은 공식의 Rust 재구현과 비교하며, 후자가 Python 없이도 그것을 고정하는 것이다),
`crates/es-data/src/training.rs`와 `crates/es/src/cmd/train.rs`(레시피 필드와 두 JSON
파일), `crates/es/tests/cli.rs`(`train_*` 테스트만), `tests/fixtures/visible-learning/training.toml`
(픽스처 레시피가 그 필드를 보여줘야 한다면 — 그렇다고 밝힐 것), `docs/design/training-recipe*.md`,
`docs/packets/M7/T4-lr-schedule*.md`.

## oracle

1. `cargo test -p es-policy --test ir_training lr_schedule_matches_the_golden` — Rust
   재구현이 골든과 비트 단위로(f64) 같다; `ES_PYTHON`이 있으면 스크립트의 `lr_at`도 비트
   단위로 같다(없으면 이유와 함께 `SKIP`).
2. `cargo test -p es-policy --test ir_training -- --ignored the_default_schedule_is_the_old_run`
   — `--schedule constant`로 40스텝을 돌리면 T3의 오라클 4가 기록한 loss 곡선을
   낸다(비트 단위, 같은 시드, 같은 장치: CPU) — 측정된 실행들이 움직이지 않았다는 뜻이다.
3. `cargo test -p es --test cli train_identity_moves_with_the_schedule` — `warmup`을 바꾸면
   `identity_hash`가 바뀐다; `scheduler.json`과 `optimizer.json`이 그 필드를 담는다.
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T4-lr-schedule.md`.

## acceptance

오라클 1–4; 서버에서의 네 행짜리 표가 설계 노트에 실리고, 각 행의 `training.lock`이
`~/artifacts/plan-v/m7-t4/` 아래에 보관된다. 어떤 행에서든 NaN이 나오면 그것이 발생한
스텝과 함께 있는 그대로의 발견으로 보고된다.

## forbidden

기본 스케줄, lr, 배치, 옵티마이저를 바꾸는 것(측정된 실행들은 재현되어야 한다);
`torch.optim.lr_scheduler`; `crates/es-policy/src/**`(lowering은 T3/T5의 것);
`crates/es-ir/**`; 어떤 체크포인트든 평가하는 것; `docs/ARCHITECTURE*.md`; 새 것 말고 다른
골든. INV-16: safetensors만.
