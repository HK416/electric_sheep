# M7 T6 — 학습 증강: Observation IR의 `training_only` 노드, 트레이너가 적용한다

스펙: §7.3(`Augment` 노드: `RandomCrop / ColorJitter / RandomErasing / GaussianNoise`,
`training_only = true`, Evaluation IR 아래에서 자동 비활성화 — INV-15), §19.3(`augmentation.json`과
`seed.json`의 증강 시드는 아이덴티티 슬롯이다), §5.1(Observation IR이 전처리를 소유한다; 트레이너는
그것을 적용할 뿐 지어내지 않는다), §3.4(전역 RNG 없음), §28.10 규칙 1과 (T6). 확장할 설계 노트:
`docs/design/training-recipe.md`(+ `.ko.md`) — "증강" 절; `docs/design/observation-lowering.md`
(+ `.ko.md`) — 각 경로에서 `training_only`가 뜻하는 것. **T1**(`training/`)과 **T3**(배치 축)에
의존한다. 선행: `crates/es-compile/src/plan.rs`(`Augment` 분기: 아이덴티티 패스스루,
`training_only` 밖에서는 `COMPILE-002`; `Pad`는 아직 거기서 lowering되지 않는다),
`crates/es-ir/src/cross.rs`(XIR-051, INV-15), `crates/es-eval/src/runner.rs::refuse_augmentation`.

## 질문

IR에는 증강 노드가 있지만 모든 경로가 그것들을 *무시한다*: bake는 Release 플랜을 돌리므로
`Augment`는 아이덴티티다; `train_act.py`는 Observation IR을 결코 보지 않는다; `augmentation.json`은
모든 실행에 대해 `{"kind":"none"}`이라고 말한다. **저자가 선언한 `training_only` 서브그래프를
트레이너가 적용할 수 있는가 — 샘플마다 무작위로, 하나의 시드로부터 재현 가능하게, 아이덴티티로
기록되어 — 그러면서 평가 경로는 오늘 만들어내는 바이트를 계속 만들어내는가?**

## 사양

* **경계.** IR 경로에서 `es train`은 번들의 Observation IR을 읽고, 이미지 입력마다 네트워크의
  입력 포트에서 끝나는 **`training_only` `Augment` 노드의 연속된 체인**을 찾는다(체인 안의
  증강이 아닌 노드, 또는 다음 노드가 체인의 다음 노드가 아닌 소비자를 가진 `training_only`
  노드는 이름으로 거부된다). 체인에 **들어가는** 텐서가 학습 텐서다; 체인을 떠나는 텐서가
  Learning IR이 소비하는 것이다.
* **bake가 경계를 쓴다.** `es dataset bake`(`crates/es-eval/src/bake.rs`)는 `--for-training`을
  얻는다: 체인이 비어 있지 않은 포트에 대해 체인의 *입력* 버퍼를 쓰고(플랜이 이미 그것을 갖고
  있다 — `produced[id] = src` — `CpuPlan::buffer_of(node)`로 노출한다), 그
  `manifest.json`은 체인(`"augmentation": [ {id, kind, params}, ... ]`)과 경계 shape을 기록한다.
  플래그가 없을 때, 그리고 `training_only` 노드가 없는 문서에 대해서는 bake가 오늘과 바이트
  단위로 같다(기존 bake 테스트는 불변).
* **평가 경로.** Release 모드의 `CpuPlan`은 `Augment`를 구조적으로 꺼진 채 유지하되(INV-15) 한
  가지 세부 규칙이 있다: 입력이 `width x height`보다 큰 `training_only` **`RandomCrop { width,
  height }`는 결정적 중앙 크롭으로 lowering된다**(오프셋 `((W - width) / 2, (H - height) / 2)`,
  `Crop`의 연산 *그리고 그 intrinsics 변환*을 재사용 — INV-14). `Pad`(replicate 모드)는
  `Pad(4) -> RandomCrop{96, 96}(training_only)` — DrQ의 random shift — 가 평가에서도 학습에서도
  네트워크에 같은 `96x96`을 건네도록 lowering된다; 대칭 패딩이면 그 평가 픽셀은 증강되지 않은
  문서의 것과 비트 단위로 같고, 테스트가 그렇게 말한다. 나머지 세 종류는 아이덴티티로 남는다.
  커밋된 Observation IR 픽스처 중 `Augment` 노드를 가진 것은 없으므로(grep에 걸린 하나,
  `tests/fixtures/quadruped/task.toml`은 Task IR이다 — 확인하고 노트에 그렇게 적을 것) 어떤
  골든도 움직이지 않는다. GPU 관측 lowering(`crates/es-compile/src/gpu/`)은 같은 규칙을
  거울처럼 반영하거나 이름으로 노드를 거부한다(`COMPILE-0xx`) — CPU 플랜이라면 아닐 조용한
  아이덴티티는 결코 아니다.
* **트레이너가 체인을 적용한다.** `train_act.py --augmentation <training/augmentation.json>`는
  샘플마다, 옵티마이저 스텝마다, 체인 순서로 적용한다: `RandomCrop{w,h}` — `[0, W-w] x [0, H-h]`
  안에서 균등하게 뽑힌 정수 오프셋; `ColorJitter{brightness, contrast, saturation, hue}` —
  brightness와 contrast만(`x * (1 + u*b)` 다음 `(x - mean) * (1 + u*c) + mean`, `u`는
  `[-1, 1]` 안), saturation/hue는 0이 아니면 이름으로 거부된다; `GaussianNoise{sigma}` — 가산적;
  `RandomErasing`은 이름으로 거부된다(여기서는 구현되지 않는다; 노트가 그것을 나열한다). 뽑기는
  **트레이너와 Rust 오라클이 둘 다 구현하는 카운터 기반 RNG**에서 온다 — `es_render::rng::{key,
  uniform}`의 `mix32`(정수 산술 열 줄), `(augmentation_seed, sample_index, step, node_index,
  draw)`로 키가 매겨진다; `torch.Generator`는 결코 아닌데, 그 스트림은 torch 버전의
  세부사항이기 때문이다. 가우시안 뽑기는 두 균등분포 위의 Box-Muller이며, 양쪽 모두 `f64`이고,
  캐스트 뒤 `f32`에서 비교된다 — 마지막 비트의 libm 질문은 가정되지 않고 측정된다(T4 10절이
  `cos`에 대해 같은 일을 했다).
* **아이덴티티.** `augmentation.json`은 `{"kind":"observation-ir", "observation_hash": ...,
  "chains": {"<port>": [ {id, kind, ...params} ]}, "seed": <s>}`가 된다; `seed.json.augmentation`은
  실제 `[run] augmentation_seed`(기본값 = `[run] seed`)가 된다. 둘 다 `identity_hash`를
  움직인다. 번들에 체인이 없는 레시피는 오늘의 `{"kind":"none"}`과 설정되지 않은 시드를
  쓴다 — 측정된 실행들의 `training_hash`는 움직이지 않는다.
* **U-측정을 위한 문서.** `tests/fixtures/visible-learning/observation-augmented.toml`: 커밋된
  `observation.toml`에 이미지 경로 위의 `Pad(4, replicate) -> RandomCrop{96,96}(training_only)`와
  `ColorJitter{brightness 0.2, contrast 0.2}(training_only)`를 더한 것. 그 `observation_hash`는
  설계 노트에 커밋된 것 옆에 기록된다; Learning IR은 불변이다(`learning_hash` 불변). Evaluation
  IR이 `observation`을 이름으로 지정하므로, 그것으로 학습된 정책의 평가는 **새로운**
  `evaluation_hash`다(스펙 13.3) — U-측정이 그것을 읽을 곳에 그렇게 적을 것.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
python/es/train_act.py
python/es/augment.py
crates/es-compile/src/plan.rs
crates/es-compile/src/gpu/**
crates/es-compile/src/lib.rs
crates/es-compile/tests/**
crates/es-eval/src/bake.rs
crates/es-eval/tests/**
crates/es-policy/tests/ir_training.rs
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/observation-augmented.toml
tests/golden/train/augment_seed0.json
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/design/observation-lowering.md
docs/design/observation-lowering.ko.md
docs/packets/M7/T6-augmentation.md
docs/packets/M7/T6-augmentation.ko.md
```

`python/es/augment.py`(신규: RNG와 종류들, 트레이너와 오라클 양쪽에서 임포트 가능),
`train_act.py`(`--augmentation`), `es-compile`(`buffer_of`, `Pad`, 중앙 크롭 규칙, GPU
미러링/거부), `es-eval/src/bake.rs`(`--for-training`), `es-data/src/training.rs`
(`augmentation_seed`, 두 슬롯), `es/src/cmd/train.rs`(번들의 Observation IR을 읽고,
`augmentation.json`을 쓰고, 플래그를 전달한다), `es/src/cmd/dataset.rs`(bake 플래그), 테스트,
픽스처, 골든, 두 설계 노트, 이 패킷.

## 오라클

1. `cargo test -p es-policy --test ir_training augmentation_matches_the_golden` — 시드 0에서
   샘플 0..4에 대해 `RandomCrop{8,8} -> ColorJitter{0.2, 0.2} -> GaussianNoise{0.05}`를 거친
   합성 `3x12x12` 이미지: Rust 재구현이 `tests/golden/train/augment_seed0.json`과 `f32`에서
   비트 단위로 같다; `ES_PYTHON`이 있으면 `python/es/augment.py`도 그것과 같다(없으면 이유와
   함께 `SKIP`). 골든은 Python 쪽에서 `#[ignore]`가 붙은 테스트로 한 번 생성된다.
2. `cargo test -p es-compile a_training_only_random_crop_is_a_centre_crop_in_release` — `96x96`
   입력 위의 `Pad(4) -> RandomCrop{96,96}(training_only)`에 대한 플랜은 그 두 노드가 없는
   플랜과 동일한 바이트를 내고, 출력 `ImageSpec`의 intrinsics는 입력의 것이다(INV-14); 입력이
   이미 `width x height`인 `RandomCrop`은 원래 그랬던 대로 아이덴티티다.
3. `cargo test -p es-eval bake_for_training_writes_the_boundary` — 증강된 픽스처에서
   `--for-training` bake는 이미지 포트에 `104x104`와 그 manifest 안의 체인을 쓴다; 플래그가
   없으면 `96x96`과 증강되지 않은 문서의 bake가 쓰는 바이트를 쓴다.
4. `cargo test -p es --test cli train_identity_moves_with_augmentation` — 증강된 번들의 실행은
   체인 항목 두 개와 실제 `seed.json.augmentation`을 가진 `augmentation.json`을 쓴다;
   `identity_hash`는 증강되지 않은 번들의 것과 다르며 `augmentation_seed`와 함께 움직인다;
   증강되지 않은 레시피는 여전히 `{"kind":"none"}`을 쓴다(dry-run 수준, Python 불필요).
5. `cargo test -p es-policy --test ir_training -- --ignored augmented_training_runs` —
   `ES_PYTHON`으로: 증강된 픽스처의 bake에서 40스텝; loss는 유한하다; 하나의
   `augmentation_seed`에서의 두 실행은 CPU에서 바이트 단위로 같은 loss 곡선을 내고, 두 시드는
   그렇지 않다.
6. `cargo xtask ci`(골든: 변경 0건); `cargo xtask check-scope docs/packets/M7/T6-augmentation.md`.

## 수용 기준

오라클 1–6(1의 Python 절반과 5는 오라클 서버에서). 서버에서: 증강된 문서의 T4 행-D 설정(배치
64, `warmup_cosine`)에서의 20,000스텝 실행, U-측정을 위한 그 `training.lock`과 체크포인트를
`~/artifacts/plan-v/m7-t6/` 아래에, 벽시계 시간을 T4의 표 옆에(스텝당 증강 비용은 관측이다).
**여기서 평가하지 않는다.** 설계 노트는 경계 규칙, RNG 키 표, 네 종류의 상태(둘 구현됨, 하나
부분적, 하나 거부됨)와 `observation_hash` 쌍을 기록한다.

## 금지

`crates/es-ir/**`(노드는 이미 존재한다; INV-15의 구조는 그들의 것이다); 증강 안의
`torch.Generator`나 어떤 전역 RNG든; 위의 `training_only` `RandomCrop` 세부 규칙을 제외한 어떤
노드의 Release-플랜 바이트든 바꾸는 것; 커밋된 `observation.toml`, `learning.toml`,
`training.toml`; `docs/ARCHITECTURE*.md`; 새 것 말고 다른 골든. INV-14: intrinsics 변환 없는
크롭 없음. INV-16: safetensors만. INV-17: 새 트레이트 없음.
