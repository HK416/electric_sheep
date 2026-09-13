<!-- Korean translation of docs/api-notes/lerobot-config.md. The English file is the working copy; regenerate this when it changes. -->

# LeRobot 정책 `config.json` — shape와 정규화 통계

**고정 버전: ACT에 한해 `lerobot` 0.6.1** — 아래 ACT 절과, 실제 체크포인트에 대해 작성된
`docs/api-notes/lerobot-act.md`를 참고. 이 페이지의 나머지는 그보다 이전이며 여전히
**`unverified`**다: 실제 `diffusion` `config.json`도, 실제 `meta/stats.json`도 읽어본 적이
없다.

원래 헤더, ACT를 제외한 모든 절에 여전히 해당됨: 이 워크스페이스는 `lerobot`을 고정한 적이
없다; 어떤 Python 패키지도 설치된 적이 없고 실제 `config.json`도 읽힌 적이 없다. 아래의
모든 필드는 `v0.1`/`v0.2` `lerobot-train` 시절(대략
`docs/api-notes/lerobot-dataset.md`의 `codebase_version: "v2.1"` 데이터셋과 같은 세대)
LeRobot에 대한 기억으로부터 재구성한 것이며, 줄에 달리 적혀 있지 않은 한 **`unverified`**다.
spec §1.7은 정확히 이 실패 모드(환각 API)를 지목한다: 사람이 `lerobot` 버전을 고정하고 이
파일을 — 그리고 shape가 실제로 다르다면 `crates/es-data/src/lerobot_config.rs`도 —
바로잡아야, 비로소 여기 있는 내용이 근거로 취급될 수 있다.

`crates/es-data/src/lerobot_config.rs`가 읽는 필드만 구조체 필드로 모델링되어 있다; 나머지는
전부 `#[serde(flatten)] extra: BTreeMap<String, Value>` 가방에 담기며, 조용히 버려지거나
거부되는 대신 변환 경고로 다시 보고된다.

## 공통 shape — `unverified`

모든 정책 config는 `"type"` 판별자를 갖는 JSON 객체다:

```json
{ "type": "act", ... }
{ "type": "diffusion", ... }
```

`crates/es-data/src/lerobot_config.rs`는 정확히 `"act"`와 `"diffusion"`만 인식한다; 그
외의 값(`"smolvla"`, `"pi0"`, `"vqbet"` 등)은 추측이 아니라 `ConfigError::Unsupported(type)`이다
(spec §14.4: severity=error는 매핑을 지어내는 대신 실행을 차단한다).

`input_features` / `output_features` — `unverified`: `name -> { type, shape }`이며,
`type`은 `"VISUAL"`, `"STATE"`, `"ACTION"` 중 하나이고 `shape`는 프레임당 텐서 shape다
(`VISUAL`은 channels-first, 예: `[3, 224, 224]`). ACT의 실제 config가 갖는 정확한 키 집합
(`observation.images.<camera>`, `observation.state`, `action`)은 검증된 출처가 아니라
`docs/api-notes/lerobot-dataset.md`의 feature 이름 관례에서 가져온 것이다.

`normalization_mapping` — `unverified`: `{ "VISUAL": "MEAN_STD", "STATE": "MEAN_STD",
"ACTION": "MEAN_STD" }`. LeRobot 소스의 여러 시점에서 관찰된 값: `MEAN_STD`, `MIN_MAX`,
`IDENTITY`. 이 crate는 세 가지를 `NormMode::{MeanStd,MinMax,Identity}`로 모델링하지만
오늘 실제로 *생성*하는 것은 언제나 `MeanStd` 형태의 `Normalize`/`Normalizer` 노드뿐이다 —
`IDENTITY`/`MIN_MAX` config 값은 (오류 없이) 받아들여져 경고와 함께 `Range` 정규화 노드로
접혀 들어가는데, Observation IR의 `NormalizeStats`에는 identity variant가 없기 때문이다.

## ACT (`type: "act"`) — **검증됨**, `lerobot` 0.6.1

더 이상 `unverified`가 아니다: `docs/api-notes/lerobot-act.md`는 실제
`lerobot/act_aloha_sim_transfer_cube_human` 체크포인트와 그 `config.json`을 프로젝트 venv에서
읽어 이 표를 고정했다. 체크포인트의 키 이름, shape, forward-pass 의미론은 그 파일을
읽으라; 아래 필드는 그중 config 절반이며, 그 체크포인트가 담고 있는 값이다.

| 필드 | 타입 | 그곳에서의 값 | 비고 |
|---|---|---|---|
| `chunk_size` | int | `100` | 예측 horizon `H` (spec §8.4) |
| `n_action_steps` | int | `100` | 실행 길이 `K` |
| `n_obs_steps` | int | `1` | ACT에서는 `1` |
| `vision_backbone` | string | `"resnet18"` | torchvision 모델 이름, `norm_layer=FrozenBatchNorm2d`로 빌드되고 `layer4`에서 잘림 |
| `pretrained_backbone_weights` | string \| null | `"ResNet18_Weights.IMAGENET1K_V1"` | *학습* 기록이다; 체크포인트는 파인튜닝된 백본 가중치를 담고 있다 |
| `replace_final_stride_with_dilation` | bool | `false` | `replace_stride_with_dilation`의 세 번째 항목 |
| `pre_norm` | bool | `false` | post-norm; `false`에서는 인코더에 최종 `LayerNorm`이 없다 |
| `dim_model` | int | `512` | transformer 폭 |
| `n_heads` | int | `8` | attention head 수 |
| `dim_feedforward` | int | `3200` | |
| `feedforward_activation` | string | `"relu"` | |
| `n_encoder_layers` | int | `4` | |
| `n_decoder_layers` | int | `1` | |
| `use_vae` | bool | `true` | CVAE 인코더 토글 — **학습 전용**: `ACT.forward`는 이를 `self.training`으로 게이트하므로 추론 시 latent는 0이다 |
| `latent_dim` | int | `32` | |
| `n_vae_encoder_layers` | int | `4` | 학습 전용 |
| `temporal_ensemble_coeff` | float \| null | `null` | 존재하면 temporal ensembling을 의미하며, 이는 forward pass가 아니라 런타임 스케줄링이다 |
| `dropout` | float | `0.1` | 파라미터를 갖지 않음; `eval()`에서는 항등 |
| `kl_weight` | float | `10.0` | 학습 전용 |
| `device`, `use_amp` | — | `"cuda"`, `false` | 학습 기록; CPU 로드는 경고만 낸다 |
| `optimizer_lr`, `optimizer_weight_decay`, … | — | | 학습 전용; 모델링되지 않고 `extra`에 담겨 경고로 보고됨 |

실제 파일을 읽은 지금, 이 파일의 다른 곳에 있던 기억 기반 재구성 텍스트에 대한 두 가지
정정:

- `input_features` / `output_features`는 정확히 위에서 추측한 shape를 갖는다
  (`VISUAL`/`STATE`/`ACTION`을 갖는 `name -> {type, shape}`), 그리고 ACT의 feature 이름은
  실제로 `observation.images.<camera>`, `observation.state`, `action`이다;
- `VISUAL` feature의 정규화 통계는 픽셀별도 아니고 평평한(flat) `[3]`도 아니라
  **채널별 `[3, 1, 1]`**이다 — `lerobot-act.md` §3 참고. `es-data`의 평평한 `FeatureStats`
  로더는 이 중첩 구조를 파싱하지 못한다; 그 후속 작업은 아직 열려 있다.

## Diffusion Policy (`type: "diffusion"`) — `unverified`

| 필드 | 타입 | 비고 |
|---|---|---|
| `horizon` | int | 예측 horizon `H` |
| `n_action_steps` | int | 실행 길이 `K` |
| `n_obs_steps` | int | 보통 `2` |
| `crop_shape` | `[height, width]` \| null | resize 전에 적용되는 center-crop |
| `crop_is_random` | bool | 학습 시에는 무작위 오프셋, eval 시에는 중앙 |
| `use_group_norm` | bool | |
| `down_dims` | `[int, ...]` | U-Net 채널 폭, 예: `[512, 1024, 2048]` |
| `kernel_size` | int | |
| `noise_scheduler_type` | string | 예: `"DDPM"`, `"DDIM"` |
| `num_train_timesteps` | int | 기본값 `100`; beta 스케줄이 만들어지는 격자 |
| `beta_schedule` | string | 기본값 `"squaredcos_cap_v2"`; `"linear"`가 우리가 lowering하는 다른 하나 |
| `prediction_type` | string | 기본값 `"epsilon"`; `"sample"` / `"v_prediction"`은 그대로 전달되고 lowering에서 거부됨 |
| `num_inference_steps` | int \| null | 없으면 `num_train_timesteps`로 기본값 설정 |
| `clip_sample` | bool | 기본값 `true` — `diffusers`가 `pred_original_sample`을 클램프한다 |
| `clip_sample_range` | float | 기본값 `1.0` |

이 여섯 개는 `diffusers`의 `DDPMScheduler` / `DDIMScheduler` 설정이며 모두 그대로
`HeadKind::Diffusion`에 도달한다: 한 스케줄로 학습된 체크포인트는 다른 스케줄로는
재현되지 않는다. `variance_type`은 LeRobot 필드가 **아니므로**, 변환은 `diffusers` 자체의
기본값인 `fixed_small`을 고정한다. 위 기본값들은 `unverified`다 —
`lerobot/common/policies/diffusion/configuration_diffusion.py`에서 가져온 것이 아니라
기억으로 떠올린 것이며, `es_ir::learning::HeadKind::Diffusion`과 `DiffusionConfig`의
`serde` 기본값이기도 하므로, 해당 필드가 빠진 config도 여전히 변환된다.

`noise_scheduler_type`은 `es_ir::learning::DiffusionScheduler`로 다음과 같이 매핑된다:
`"DDPM" -> Ddpm`, `"DDIM" -> Ddim`, 그 외는 인식되지 않은 값을 지목하는 경고와 함께
`DpmSolver`로 (이 삼분할은 LeRobot의 enum이 아니라 이 crate 자체의 선택이다 — 사람이
결국 고정하게 될 버전에서 LeRobot이 다른 스케줄러 계열을 갖는지는 `unverified`).

## `crop_shape` / `crop_is_random` -> Observation IR — 설계 노트이며 LeRobot 필드가 아님

`crop_shape`는 `Crop { mode: CropMode::Center, rescale_intrinsics: true, .. }` 노드가
된다 (`INV-14`: intrinsics는 항상 재스케일되며 결코 낡은 채로 남지 않는다). `crop_is_random`은
추가로 같은 `ObservationIr` 그래프에 **연결되지 않은(unwired)** `Augment { kind: RandomCrop,
training_only: true }` 노드를 삽입하는데, 순전히 "학습 샘플은 여기서 무작위 오프셋을
취한다"는 사실을 `Evaluation IR`이 구조적으로 비활성화하는 형태로 기록하기 위해서다
(spec §7.3, `INV-15`) — 이는 데이터플로에 실제로 관여하지 않는데, `CropMode::Random` 샘플
오프셋이 타입 시스템이 추적하는 명목상의 기하(geometry)를 바꾸지 않기 때문이며
(`es_ir::observation::CropMode::Random`의 문서 주석 참고), eval-time 그래프는 평범한
center crop을 실행하도록 정의되어 있다. `cross_fixture` 스타일 Augment 노드가 이미 걸리는
것으로 알려진 Cross-IR 검사는 `XIR-050`이다.

## Dataset `stats` (`meta/stats.json`) — `unverified`

`docs/api-notes/lerobot-dataset.md`와 같은 불확실성이며, 그 문서는 이 파일을 `es-data`의
데이터셋 리더가 *무시하는* 것으로 기록한다(정규화는 데이터셋 정체성이 아니라 Observation IR의
소관이다). 이 패킷이 첫 소비자다:

```json
{
  "observation.state": { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "action":            { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "observation.images.top": { "mean": [[[0.42]], [[0.41]], [[0.39]]], "std": [[[0.19]], ...] }
}
```

`crates/es-data::lerobot_config::Stats`는 이를 `BTreeMap<String, FeatureStats>`로,
`FeatureStats { mean: Vec<f64>, std: Vec<f64>, min: Vec<f64> (기본값 빈 배열), max: Vec<f64>
(기본값 빈 배열) }`로 모델링한다 — 평평한 구조이며, 비디오 feature의 통계치가 실제로 담을
수도 있는 픽셀 위치별 중첩 배열이 아니다 (이미지 통계가 채널별 `[3]`인지 픽셀별
`[3,1,1]`/전체 해상도인지는 `unverified`; 이 crate의 `Stats` 로더는 평평한 채널별 리스트를
기대하며 중첩 배열 파일은 파싱에 실패한다 — 실제 파일을 본 뒤 고칠 후속 작업).

## `stats: None`일 때 `convert()`가 하는 일

`Stats`가 주어지지 않거나, 어떤 feature에 대한 항목이 없을 때: 이미지 feature는
ImageNet 상수(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` — 이
특정 ResNet18 체크포인트가 실제로 이 값들로 학습되었는지는 `unverified`이지만, 이는
`torchvision`의 표준 ImageNet 값이다)로 경고와 함께 폴백한다; state/action feature는
항등(`mean = 0`, `std = 1`)으로 경고와 함께 폴백하는데, 임의의 joint-state 벡터에 대한
데이터셋 독립적 표준 기본값이 없기 때문이다.

## 물리 단위 — 포맷에 없으며 `convert()`가 가정함

LeRobot의 `config.json`과 `stats.json`은 `observation.state` / `action`에 대한 단위나
좌표계 메타데이터를 전혀 담지 않는다 (spec §5.4의 `Unit`/`Frame`에는 LeRobot에 대응하는
것이 없다). `crates/es-data/src/lerobot_config.rs`는 관절 위치 제어를 가정하고
(정규화 이전에는 `Unit::Angle`, 테스트 픽스처에서는
`es_ir::task::ActionSpace::JointPosition`/`es_ir::deployment::ActionSpace::JointPosition`),
반환되는 경고에서 그렇게 밝힌다 — 이것이 가장 흔한 LeRobot 구성(`so100`, ALOHA)이지만,
비어 있는 정책 `config.json` 변환에는 참조할 Task IR이 없으므로 컴파일러가 확인할 수 없는
추측이다.
