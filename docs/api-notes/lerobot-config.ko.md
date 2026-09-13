<!-- Korean translation of docs/api-notes/lerobot-config.md. The English file is the working copy; regenerate this when it changes. -->
# LeRobot 정책 `config.json` — 구조와 정규화 통계

**고정 버전: 없음(NONE).** 이 워크스페이스는 `lerobot`을 어떤 버전으로도 고정하지 않았다. 이
파일을 작성하면서 Python 패키지를 설치하지도, 실제 `config.json`을 읽어보지도 않았다. 아래의
모든 필드는 `lerobot-train`의 `v0.1`/`v0.2` 시대(대략 `docs/api-notes/lerobot-dataset.md`의
`codebase_version: "v2.1"` 데이터셋과 같은 세대) LeRobot에 대한 기억으로 재구성한 것이며,
별도로 명시하지 않는 한 모두 **미검증 (unverified)**이다. 사양 §1.7은 정확히 이 실패 양상
(환각 API)을 지목한다: 사람이 `lerobot` 버전을 고정하고 이 파일을 — 실제 구조가 다르다면
`crates/es-data/src/lerobot_config.rs`도 함께 — 수정하기 전까지는, 여기 적힌 어떤 내용도
참(ground truth)으로 취급해서는 안 된다.

`crates/es-data/src/lerobot_config.rs`가 읽어들이는 필드만 구조체 필드로 모델링되어 있다. 그
외 나머지는 모두 `#[serde(flatten)] extra: BTreeMap<String, Value>` 가방에 담기며, 조용히
버려지거나 거부되는 대신 변환 경고로 보고된다.

## 공통 구조 — 미검증 (unverified)

모든 정책 설정은 `"type"` 판별자를 가진 JSON 객체다:

```json
{ "type": "act", ... }
{ "type": "diffusion", ... }
```

`crates/es-data/src/lerobot_config.rs`는 정확히 `"act"`와 `"diffusion"`만 인식한다. 그 외의
값(`"smolvla"`, `"pi0"`, `"vqbet"` 등)은 추측 대신 `ConfigError::Unsupported(type)`이
된다(사양 §14.4: severity=error는 매핑을 지어내는 대신 실행을 차단한다).

`input_features` / `output_features` — 미검증 (unverified): `name -> { type, shape }` 형태이며,
`type`은 `"VISUAL"`, `"STATE"`, `"ACTION"` 중 하나이고 `shape`는 프레임당 텐서 모양이다
(`VISUAL`의 경우 채널 우선, 예: `[3, 224, 224]`). ACT의 실제 설정 키 집합
(`observation.images.<camera>`, `observation.state`, `action`)은 검증된 출처가 아니라
`docs/api-notes/lerobot-dataset.md`의 특성 이름 규칙에서 가져온 것이다.

`normalization_mapping` — 미검증 (unverified): `{ "VISUAL": "MEAN_STD", "STATE": "MEAN_STD",
"ACTION": "MEAN_STD" }`. LeRobot 소스 곳곳에서 확인되는 값: `MEAN_STD`, `MIN_MAX`,
`IDENTITY`. 이 크레이트는 이 세 가지를 `NormMode::{MeanStd,MinMax,Identity}`로 모델링하지만,
오늘 시점에는 오직 `MeanStd` 형태의 `Normalize`/`Normalizer` 노드만 *생성*한다 —
`IDENTITY`/`MIN_MAX` 설정 값은 (오류 없이) 받아들여져 경고와 함께 `Range` 정규화 노드로 접혀
들어가는데, Observation IR의 `NormalizeStats`에는 identity 변형이 없기 때문이다.

## ACT (`type: "act"`) — 미검증 (unverified)

| 필드 | 타입 | 설명 |
|---|---|---|
| `chunk_size` | int | 예측 지평선 `H` (사양 §8.4) |
| `n_action_steps` | int | 실행 길이 `K` |
| `n_obs_steps` | int | ACT에서는 거의 항상 `1` |
| `vision_backbone` | string | 예: `"resnet18"` |
| `pretrained_backbone_weights` | string \| null | 예: `"ResNet18_Weights.IMAGENET1K_V1"` |
| `dim_model` | int | 트랜스포머 폭 |
| `n_heads` | int | 어텐션 헤드 수 |
| `dim_feedforward` | int | |
| `n_encoder_layers` | int | |
| `n_decoder_layers` | int | |
| `use_vae` | bool | CVAE 인코더 토글 |
| `latent_dim` | int | |
| `temporal_ensemble_coeff` | float \| null | 존재하면 추론 시 시간적 앙상블(temporal ensembling) 사용 |
| `dropout` | float | |
| `kl_weight` | float | |
| `optimizer_lr`, `optimizer_weight_decay`, … | — | 학습 전용; 모델링하지 않고 `extra`에 담아 경고로 보고 |

## Diffusion Policy (`type: "diffusion"`) — 미검증 (unverified)

| 필드 | 타입 | 설명 |
|---|---|---|
| `horizon` | int | 예측 지평선 `H` |
| `n_action_steps` | int | 실행 길이 `K` |
| `n_obs_steps` | int | 보통 `2` |
| `crop_shape` | `[height, width]` \| null | 리사이즈 전에 적용되는 중앙 크롭 |
| `crop_is_random` | bool | 학습 시에는 무작위 오프셋, 평가 시에는 중앙 |
| `use_group_norm` | bool | |
| `down_dims` | `[int, ...]` | U-Net 채널 폭, 예: `[512, 1024, 2048]` |
| `kernel_size` | int | |
| `noise_scheduler_type` | string | 예: `"DDPM"`, `"DDIM"` |
| `num_train_timesteps` | int | |
| `beta_schedule` | string | 예: `"squaredcos_cap_v2"` |
| `prediction_type` | string | 예: `"epsilon"` — 그대로 전달하는 것 이상으로는 모델링하지 않음 |
| `num_inference_steps` | int \| null | 없으면 `num_train_timesteps`를 기본값으로 사용 |

`noise_scheduler_type`은 `es_ir::learning::DiffusionScheduler`로 다음과 같이 매핑된다:
`"DDPM" -> Ddpm`, `"DDIM" -> Ddim`, 그 외에는 인식하지 못한 값을 알리는 경고와 함께
`DpmSolver`로 매핑된다(이 3분할은 LeRobot의 열거형이 아니라 이 크레이트 자체의 선택이다 —
사람이 결국 버전을 고정했을 때 LeRobot에 다른 스케줄러 계열이 있는지는 미검증
(unverified)).

## `crop_shape` / `crop_is_random` -> Observation IR — 설계 노트, LeRobot 필드 아님

`crop_shape`는 `Crop { mode: CropMode::Center, rescale_intrinsics: true, .. }` 노드가
된다(INV-14: 내부 파라미터는 다시 스케일링되며, 결코 오래된 값으로 남지 않는다).
`crop_is_random`은 추가로 같은 `ObservationIr` 그래프에 **연결되지 않은(unwired)**
`Augment { kind: RandomCrop, training_only: true }` 노드를 삽입하는데, 이는 순전히 "학습
샘플은 여기서 무작위 오프셋을 취한다"는 사실을 `Evaluation IR`이 구조적으로 비활성화하는
형태로 기록하기 위한 것이다(사양 §7.3, `INV-15`) — 이는 데이터플로우에 실제로 반영되지
않는데, `CropMode::Random` 샘플 오프셋은 타입 시스템이 추적하는 공칭 기하 구조를 바꾸지
않기 때문이며(`es_ir::observation::CropMode::Random`의 문서 주석 참고), 평가 시점 그래프는
일반 중앙 크롭만 실행하도록 정의되어 있다. `cross_fixture` 스타일의 Augment 노드가 이미
걸려 넘어지는 것으로 알려진 교차 IR 검사는 `XIR-050`이다.

## Dataset `stats` (`meta/stats.json`) — 미검증 (unverified)

`docs/api-notes/lerobot-dataset.md`와 동일한 불확실성을 가진다. 그 문서는 이 파일이
`es-data`의 데이터셋 리더에서 *무시된다*고 기록한다(정규화는 데이터셋 정체성이 아니라
Observation IR의 소관이다). 이 패킷이 첫 번째 소비자다:

```json
{
  "observation.state": { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "action":            { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "observation.images.top": { "mean": [[[0.42]], [[0.41]], [[0.39]]], "std": [[[0.19]], ...] }
}
```

`crates/es-data::lerobot_config::Stats`는 이를 `BTreeMap<String, FeatureStats>`로
모델링하며, `FeatureStats { mean: Vec<f64>, std: Vec<f64>, min: Vec<f64> (기본값 빈 배열),
max: Vec<f64> (기본값 빈 배열) }` 형태다 — 이는 평탄한 구조이며, 비디오 특성의 통계가
실제로 가질 수 있는 픽셀 위치별 중첩 배열이 아니다(이미지 통계가 채널별 `[3]`인지 픽셀별
`[3,1,1]`/전체 해상도인지는 미검증 (unverified); 이 크레이트의 `Stats` 로더는 평탄한
채널별 리스트를 기대하며, 중첩 배열 파일은 파싱에 실패한다 — 실제 파일을 보게 되면 후속
수정이 필요하다).

## `convert()`가 `stats: None`을 다루는 방식

`Stats`가 주어지지 않거나 어떤 특성이 그 안에 항목이 없는 경우: 이미지 특성은 ImageNet
상수(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` — 이 특정 ResNet18
체크포인트가 실제로 이 값들로 학습되었는지는 미검증 (unverified)이지만, 이는 표준
`torchvision` ImageNet 값이다)로 대체되며 경고가 발생한다; 상태/행동 특성은 항등값
(`mean = 0`, `std = 1`)으로 대체되며 경고가 발생하는데, 임의의 관절 상태 벡터에 대해서는
데이터셋에 기반하지 않은 표준 기본값이 없기 때문이다.

## 물리 단위 — 포맷에 없으며 `convert()`가 가정하는 값

LeRobot의 `config.json`과 `stats.json`은 `observation.state` / `action`에 대한 단위나
좌표계 메타데이터를 전혀 담지 않는다(사양 §5.4의 `Unit`/`Frame`에 대응하는 LeRobot측
개념이 없다). `crates/es-data/src/lerobot_config.rs`는 관절 위치 제어(정규화 이전에는
`Unit::Angle`, 테스트 픽스처에서는 `es_ir::task::ActionSpace::JointPosition`/
`es_ir::deployment::ActionSpace::JointPosition`)를 가정하고, 반환하는 경고에서 그 사실을
명시한다 — 이는 LeRobot에서 가장 흔한 설정(`so100`, ALOHA)이지만, 대조할 Task IR이 없는
상태에서 컴파일러가 검증할 수 없는 추측이며, 정책 `config.json` 단독 변환에는 그런 Task IR이
없다.
