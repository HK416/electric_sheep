# M7 T5 — `pretrained = true`: `base_model.lock`을 갖춘 ImageNet ResNet18 가중치

스펙: §8.3(`VisionEncoder { pretrained, frozen }`), §19.3(`base_model.lock` — "사전학습된
backbone의 출처는 결과에 직접 영향을 주므로 프로버넌스에 포함되어야 하며, 라이선스 추적의
근거이기도 하다"), §5.3(해시 체인 밖은 없다), §2.5(인스턴스화 시 네트워크 없음), §29 라이선스
행(오너 결정 2026-09-15: torchvision의 ImageNet ResNet18 가중치, BSD-3는 허용됨), §28.9 L13,
§28.10(T5). 수정할 설계 노트: `docs/design/learning-lowering.md`(+ `.ko.md`) 5.1 → 5.3
"pretrained" 절, 그리고 lock 파일을 위한 `docs/design/training-recipe.md`(T1). **T1**
(`training/`)과 **T3**(배치 축)에 의존한다. 선행: V2b(`pretrained = true`의 거부 — 왜
거부되었는지는 `lower/torch.rs`의 주석을 읽을 것: 생성 시점의 네트워크, 해시 슬롯 없음),
V8/V19(`lerobot.rs`가 `FrozenBatchNorm2d`로 LeRobot의 backbone을 lowering한다 — 전례),
V13(처음부터 학습할 때의 GroupNorm).

## the question

처음부터 학습한 ResNet18은 200개의 시연과 40k스텝 뒤 96×96에서 25mm 큐브를 찾아냈지만
여전히 LeRobot의 ACT에 뒤처지는데, 그 하나뿐인 중요한 아키텍처 차이는 ImageNet으로
사전학습된 backbone이다(설계 노트 7.27절). `pretrained = true`는 그것을 인스턴스화하면
네트워크로 가중치를 받아와야 하고 아무것도 그것을 해시하지 않을 것이므로 거부된다.
**사전학습된 가중치가 이름 붙고, 해시되고, 라이선스가 명시된 아티팩트로 체인에 들어와
`pretrained = true`가 lowering될 수 있게 될 수 있는가 — 그리고 lowering된 함수는 여전히
학습 모드를 갖지 않는가?**

## spec

* **아티팩트.** `python/es/fetch_backbone.py --arch resnet18 --out <dir>`(학습 경로이므로
  Python 허용)는 `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`를 한
  번 인스턴스화하여 `resnet18-imagenet1k-v1.safetensors`(모든 `state_dict` 텐서, torchvision
  자신의 키 이름, BatchNorm 실행 통계량을 **포함하여** — 이들은 고정 상수가 된다)와
  `resnet18-imagenet1k-v1.lock.json`을 쓴다: `{ "source":
  "torchvision.models.ResNet18_Weights.IMAGENET1K_V1", "torchvision": "<버전>", "url": "<
  torchvision이 보고하는 가중치 URL>", "sha256_upstream": "<torchvision 자신의 파일 해시>",
  "blake3": "<쓰인 safetensors의 blake3>", "license": "BSD-3-Clause", "license_url":
  "https://github.com/pytorch/vision/blob/main/LICENSE" }`. pickle 읽기는 학습 경로 위의
  torchvision 내부에서만 일어난다 — `es`는 결코 `.pth`를 열지 않는다; INV-16은 *이
  프로젝트의* 로더가 받아들이는 것에 관한 것이고, 그것들은 safetensors를 받아들인다. 기대되는
  `blake3`는 프로버넌스 테스트 옆에 상수로 **저장소에 고정**된다(`go1_primitives.PROVENANCE.json`이
  업스트림 해시를 고정하는 방식과 같다) — 스크립트는 해시가 그 고정값과 다른 파일을
  `--repin`이 주어지지 않는 한 쓰기를 거부하며 그렇게 말한다.
* **IR 쪽.** `VisionEncoder { pretrained: true }`는 더 이상 거부되지 않는다. 오늘처럼 `fc`가
  교체된 torchvision의 `resnet18(weights=None, norm_layer=FrozenBatchNorm2d)`로 lowering되며,
  프리픽스 클레임 `nodes.<k>.*`가 이제 고정된 버퍼도 포함한다. **생성 시점에 네트워크 없음**
  (`weights=None`) — ImageNet 텐서는 학습된 체크포인트가 그러하듯 정확히 체크포인트를 통해
  들어온다 — `es policy pack --weights`가 그것들을 받아들이고, 학습을 위한 *초기* 체크포인트는
  `es train`(T1)이 만든다: 레시피의 번들이 사전학습된 인코더를 가지면, `es train`은
  레시피에서 `base_model = "<dir>/resnet18-imagenet1k-v1.safetensors"`를 읽어 그 blake3를
  lock 파일 **그리고** 고정된 상수에 대해 검증하고, lock 파일로부터 `training/base_model.lock`을
  쓰며(§19.3, 이제 실제 값), 첫 스텝 전에 `nodes.<k>.` 아래로 로드할 텐서들을
  `train_act.py --init-backbone <file>`에 건넨다. `frozen: true`이면 `train_act.py`는
  `nodes.<k>.*`를 옵티마이저에서 제외한다(그 파라미터들의 `requires_grad = False`);
  `frozen: false`는 미세조정한다. BatchNorm 통계량은 어느 쪽이든 고정된다
  (`FrozenBatchNorm2d`는 학습 모드가 없다) — 이는 유지되는 V13 규칙이다:
  `train() == eval()` 비트 단위로 동일.
* **해시 결과.** `lowering_hash`는 `pretrained: true`인 그래프에 대해서만 움직인다(소스가
  다르므로); `learning_hash`는 움직이지 않는다; `training_hash`는 0이 아닌 `base_model`
  슬롯을 얻는다. 데모의 커밋된 `learning.toml`은 `pretrained = false`를 유지한다 — 이
  패킷은 어떤 픽스처도 바꾸지 않는다; U-측정을 위해 `tests/fixtures/visible-learning/`
  아래에 두 번째 문서 `learning-pretrained.toml`이 추가되며, 그것 자신의 `learning_hash`가
  설계 노트에 기록된다.
* **이름을 대는 거부.** blake3가 lock 파일이나 고정값과 다른 `base_model`(`TRAIN-0xx`);
  레시피가 `base_model`을 이름 대지 않는 번들 안의 `pretrained: true`; `license` 필드가
  비어 있는 lock 파일.

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
python/es/fetch_backbone.py
python/es/train_act.py
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/ir_training.rs
crates/es-policy/tests/backbone_provenance.rs
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/learning-pretrained.toml
docs/api-notes/torchvision.md
docs/api-notes/torchvision.ko.md
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M7/T5-pretrained-backbone.md
docs/packets/M7/T5-pretrained-backbone.ko.md
```

`python/es/fetch_backbone.py`(신규), `python/es/train_act.py`(`--init-backbone`, `frozen`
처리), `crates/es-policy/src/lower/torch.rs`(`pretrained` 분기), `crates/es-policy/tests/ir_training.rs`
(새 테스트), `crates/es-policy/tests/backbone_provenance.rs`(신규, `#[ignore]`: torchvision이
있는 머신에서 스크립트를 통해 내려받아 고정된 blake3를 확인한다), `crates/es-data/src/training.rs`
(`base_model` 레시피 필드와 lock 라이터), `crates/es/src/cmd/train.rs`, `crates/es/tests/cli.rs`
(`train_*` 테스트만), `tests/fixtures/visible-learning/learning-pretrained.toml`(신규),
`docs/api-notes/torchvision*.md`("pretrained weights" 하위 절: URL, 업스트림 sha256, 버전),
`docs/design/learning-lowering*.md`, `docs/design/training-recipe*.md`,
`docs/packets/M7/T5-pretrained-backbone*.md`.

## oracle

1. `cargo test -p es-policy --lib lower::torch::tests::a_pretrained_backbone_lowers_frozen` —
   `pretrained: true`에 대해 생성된 소스가 `weights=None`과 `FrozenBatchNorm2d`를 담고,
   `weights="DEFAULT"`/`IMAGENET1K`를 담지 않으며, 선언된 클레임이 `nodes.<k>.*`를
   포괄한다; `pretrained: false`는 여전히 GroupNorm으로 lowering된다(V13의 테스트 불변).
   Python 불필요.
2. `cargo test -p es-policy --test ir_training -- --ignored the_pretrained_backbone_has_no_training_mode`
   — torch로: `--init-backbone` 이후 사전학습된 모듈에서
   `torch.equal(backbone(x).train(), backbone(x).eval())`; 로드된 텐서가 safetensors
   파일과 비트 단위로 같다.
3. `cargo test -p es-policy --test backbone_provenance -- --ignored` — torchvision이 있는
   머신(서버)에서: 스크립트가 파일을 쓰고, 그 blake3가 고정된 상수와 같으며, lock 파일의
   `license`가 `BSD-3-Clause`다; 변조된 고정값에 대한 `--repin` 없는 실행은 이름을 대며
   거부된다.
4. `cargo test -p es --test cli train_refuses_a_mismatched_base_model`과
   `train_writes_base_model_lock_from_the_lock_file`(dry-run 수준; Python 불필요).
5. `cargo test -p es-policy --test ir_training -- --ignored frozen_excludes_the_backbone_from_the_optimizer`
   — `frozen: true`로 20스텝: `nodes.<k>.*` 텐서는 비트 단위로 불변, 헤드는 움직였다.
6. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M7/T5-pretrained-backbone.md`.

## acceptance

오라클 1–6(2, 3, 5는 오라클 서버에서). 아티팩트와 lock은 서버의 `~/artifacts/plan-v/m7-t5/`에
있다(저장소는 고정값과 lock의 기대 필드만 담고, 45MB 파일은 담지 않는다). 설계 노트 5.3절이
규칙을 기록한다: *사전학습된 backbone은 다른 무엇과도 같은 체크포인트다 — 생성 시점의
네트워크가 아니라 `es policy pack`/`es train`을 통해 들어오며, 그 프로버넌스는
`base_model.lock`이다.*

## forbidden

모듈 생성 시점의 어떤 네트워크 접근이든(`None` 이외의 `weights=`); Rust 안이나
`fetch_backbone.py` 내부의 torchvision 자신 것 외에 `es`의 Python 헬퍼 안의 pickle
리더(INV-16); `crates/es-policy/src/lerobot.rs`; 커밋된 `learning.toml`을 바꾸는 것;
`es-ir` 스키마(`pretrained`/`frozen`은 이미 존재한다); `docs/ARCHITECTURE*.md`; 골든.
INV-17: 새 트레이트 없음.
