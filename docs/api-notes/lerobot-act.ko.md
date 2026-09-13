<!-- Korean translation of docs/api-notes/lerobot-act.md. The English file is the working copy; regenerate this when it changes. -->

# LeRobot ACT 체크포인트 — 검증된 레이아웃

**고정 버전: `lerobot` 0.6.1** (`torch` 2.11.0+cpu, `torchvision` 0.26.0+cpu), 실제 체크포인트
`lerobot/act_aloha_sim_transfer_cube_human`에 대해 읽음
(`model.safetensors`의 `blake3`는 `cargo test -p es-policy act -- --nocapture`가 출력한다;
파일은 206,766,560 바이트, 242개 텐서, 모두 `F32`, `__metadata__` 없음).

`docs/api-notes/lerobot-config.md`와 달리, **이 파일에는 `unverified`인 내용이 하나도
없다**: 아래의 모든 키 이름, shape, 기본값은 그 체크포인트와 그 `config.json`에서 직접 읽은
것이며, 모든 동작에 관한 주장은 프로젝트 venv에서
`lerobot.policies.act.modeling_act.ACTPolicy`를 실제로 실행해 확인했다.
`crates/es-policy/src/lerobot.rs`는 정확히 이 파일을 구현하며,
`crates/es-policy/tests/act_checkpoint.rs`는 둘을 정직하게 유지시키는 오라클이다 (spec §1.4).

체크포인트는 절대 커밋하지 않는다. `target/lerobot-cache/`에 다운로드하며, 테스트는
`ES_ACT_CHECKPOINT`를 통해 이를 찾는다. **INV-16**: `allow_patterns`는 상류에 하나가
추가되더라도 어떤 `.bin`/`.pt`/`.ckpt`/`.pkl`(pickle)도 디스크에 올라오지 못하게 막고,
`revision`은 이 파일이 검증된 정확한 커밋을 고정하며, 다운로드 후의 assertion은 같은
불변식에 대한 두 번째의 독립적인 검사다:

```
python -c "
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = 'ba73b2766f1371cdc133ca4efb97eb090d744625'  # resolved via HfApi().model_info(...).sha

path = snapshot_download(
    'lerobot/act_aloha_sim_transfer_cube_human',
    revision=REVISION,
    allow_patterns=['*.safetensors', '*.json'],
    local_dir='target/lerobot-cache/act_aloha_sim_transfer_cube_human',
)
banned = [p for pat in ('*.bin', '*.pt', '*.ckpt', '*.pkl') for p in Path(path).rglob(pat)]
assert not banned, f'pickle-loadable file(s) in snapshot: {banned}'
"
```

## 1. 0.6.1에서 `select_action`이 실제로 무엇인가

`ACTPolicy.select_action`은 `predict_action_chunk` 앞에 놓인 큐다: `n_action_steps`번마다
한 번 모델을 호출하고, 호출마다 액션을 하나씩 꺼낸다. `predict_action_chunk`가 전체
forward pass다 — `self.model(batch)[0]`을, 그 뒤에 `select_action`이 `n_action_steps`만큼
슬라이스한다. **오라클은 이 청크(chunk)를 비교하며**, 이것이 spec §8.9가 허용오차를 정의하는
바로 그 텐서다.

**정규화(normalization)는 0.6.x에서 정책 밖으로 빠져나갔다.** 별도의 프로세서 파이프라인
(`lerobot.processor.normalize_processor`)으로 옮겨갔으므로, `ACTPolicy`는 더 이상
`normalize_inputs` / `unnormalize_outputs` 서브모듈을 갖지 않으며, 이 체크포인트를 로드하면
다음이 로그로 남는다:

```
WARNING:root:Unexpected key(s) when loading model: ['normalize_inputs.buffer_observation_images_top.mean', ...]
```

— 정책이 조용히 무시하는 여덟 개의 버퍼다. 따라서 `predict_action_chunk`는 **이미 정규화된**
입력을 받아 **정규화된** 액션을 반환한다. 통계치는 여전히 체크포인트 안에 있으므로, 우리의
lowering은 이를 노드 9/10/11로 담아두고, `python/act_ref.py`는 참조 호출 주변에서 동일한
것을 적용한다. 공식은 LeRobot의 `MEAN_STD`다:

```
forward:  (x - mean) / (std + 1e-8)          # eps = 1e-8, NormalizerProcessorStep.eps
inverse:  x * std + mean
```

## 2. `config.json` — lowering이 읽는 필드

실제 파일 그대로, 중요한 필드만:

| 필드 | 여기서의 값 | 용도 |
|---|---|---|
| `type` | `"act"` | 판별자; 다른 값이면 거부됨 |
| `input_features` | `observation.images.top: {VISUAL, [3, 480, 640]}`, `observation.state: {STATE, [14]}` | 입력 이름과 shape |
| `output_features` | `action: {ACTION, [14]}` | `action_dim` |
| `normalization_mapping` | `{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}` | `MEAN_STD`만 lowering된다 |
| `chunk_size` | `100` | 예측 horizon `H`, 그리고 `decoder_pos_embed`의 행 수 |
| `n_action_steps` | `100` | 실행 길이 `K` |
| `n_obs_steps` | `1` | `1`만 lowering된다 |
| `vision_backbone` | `"resnet18"` | torchvision 모델 이름; 백본 폭을 512로 고정 |
| `pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | **lowering에서 무시됨** — 체크포인트가 파인튜닝된 백본 가중치를 담고 있으므로 `weights=None`으로 빌드하고 파일에서 로드한다 |
| `replace_final_stride_with_dilation` | `false` | `replace_stride_with_dilation=[False, False, <this>]` |
| `pre_norm` | `false` | post-norm만 lowering된다 |
| `dim_model` | `512` | `d_model`; 카메라 임베딩은 `dim_model // 2`를 사용 |
| `n_heads` | `8` | |
| `dim_feedforward` | `3200` | |
| `feedforward_activation` | `"relu"` | `relu`만 lowering된다 |
| `n_encoder_layers` | `4` | |
| `n_decoder_layers` | `1` | |
| `use_vae` | `true` | 학습 전용, §4 참고 |
| `latent_dim` | `32` | 추론 시 제로 latent의 폭 |
| `n_vae_encoder_layers` | `4` | 학습 전용 |
| `temporal_ensemble_coeff` | `null` | non-null이면 거부됨: 그것은 forward pass가 아니라 런타임 스케줄링이다 |
| `dropout`, `kl_weight`, `optimizer_*`, `device`, `use_amp` | — | 학습/배포용; 무시됨 |

`device: "cuda"`는 파일 안에 있으며 *학습* 기록이다; CPU 박스에서 로드하면
`Device 'cuda' is not available. Switching to 'cpu'`가 로그로 남을 뿐 오류가 아니다.

## 3. `model.safetensors` — 키 이름과 리맵

242개 텐서. `crates/es-policy/src/lerobot.rs`의 `remap_act_keys`는 LeRobot 접두사를 우리의
`nodes.<id>.<param>` 체계(`WEIGHT_PREFIX`)로 매핑하며, longest-prefix-first이므로
`model.encoder.`가 `model.encoder_latent_input_proj.`를 삼켜버리지 않는다:

| LeRobot 접두사 | 우리 노드 | 개수 | shape (이 체크포인트 기준) |
|---|---|---|---|
| `model.backbone.` | `nodes.0.` | 100 | torchvision의 것; 열거가 아니라 접두사로 지목 |
| `model.encoder_img_feat_input_proj.` | `nodes.1.` | 2 | `weight [512, 512, 1, 1]`, `bias [512]` |
| `model.encoder_robot_state_input_proj.` | `nodes.2.` | 2 | `weight [512, 14]`, `bias [512]` |
| `model.encoder_latent_input_proj.` | `nodes.3.` | 2 | `weight [512, 32]`, `bias [512]` |
| `model.encoder_1d_feature_pos_embed.` | `nodes.4.` | 1 | `weight [2, 512]` — latent용 토큰 하나, state용 토큰 하나 |
| `model.encoder.` | `nodes.5.` | 48 | `layers.<i>.{self_attn,linear1,linear2,norm1,norm2}` × 4 |
| `model.decoder_pos_embed.` | `nodes.6.` | 1 | `weight [100, 512]` — DETR object query |
| `model.decoder.` | `nodes.7.` | 20 | 레이어 텐서 18개 × 1 + `norm.{weight,bias}` |
| `model.action_head.` | `nodes.8.` | 2 | `weight [14, 512]`, `bias [14]` |
| `normalize_inputs.buffer_observation_state.` | `nodes.9.` | 2 | `mean [14]`, `std [14]` |
| `normalize_inputs.buffer_observation_images_top.` | `nodes.10.` | 2 | `mean [3, 1, 1]`, `std [3, 1, 1]` |
| `unnormalize_outputs.buffer_action.` | `nodes.11.` | 2 | `mean [14]`, `std [14]` |

이미지 버퍼 이름은 feature 이름의 점을 밑줄로 바꾼 것이다 — `ModuleDict` 키에는 점이 들어갈
수 없어서 생긴 LeRobot 자체의 `nn.ModuleDict` 키 정제 규칙이다. 같은 규칙이 우리의 `forward`
키워드 이름도 정한다 (`observation.images.top` → `observation_images_top`).

**이미지 통계치는 픽셀별이 아니라 채널별 `[3, 1, 1]`이다.** 이는 `lerobot-config.md`의
"Dataset `stats`" 절에서 `VISUAL` feature에 대해 남아 있던 미해결 질문에 답한다.

### 제외됨: 58개 텐서, 전부 학습 전용

| 접두사 | 개수 | 이유 |
|---|---|---|
| `model.vae_encoder.` | 48 | CVAE 인코더, §4 |
| `model.vae_encoder_cls_embed.`, `_robot_state_input_proj.`, `_action_input_proj.`, `_latent_output_proj.` | 7 | 동일 |
| `model.vae_encoder_pos_enc` | 1 | 고정 사인파 버퍼 `[1, 102, 512]`, 동일 |
| `normalize_targets.buffer_action.` | 2 | *학습 타깃*을 정규화한다; 추론 시에는 타깃이 없다 |

### Attention 파라미터 이름

`ACTEncoderLayer`/`ACTDecoderLayer`는 평범한 `nn.MultiheadAttention`, `nn.Linear`,
`nn.LayerNorm`을 갖고 있으므로, 그 `state_dict` 이름은 `torch.nn`의 것이다:
`self_attn.in_proj_weight` `[1536, 512]`, `self_attn.in_proj_bias` `[1536]`,
`self_attn.out_proj.{weight,bias}`, `linear1 [3200, 512]`, `linear2 [512, 3200]`,
`norm1`/`norm2` (그리고 디코더에는 `norm3`과 `multihead_attn.*`가 추가). LeRobot
체크포인트가 접두사 이름 변경만으로 우리의 lowering에 들어맞는 이유가 바로 이것이다.

**인코더에는 최종 norm이 없다.** `ACTEncoder.norm`은 `pre_norm`일 때만 `nn.LayerNorm`이고
그 외에는 `nn.Identity()`다; `pre_norm: false`이므로 파일 안에 `model.encoder.norm.*`이
없다. `ACTDecoder.norm`은 항상 `LayerNorm`이며 항상 존재한다.

## 4. `use_vae`는 학습 전용이다 — 검증됨

`ACT.forward`는 다음 조건에서만 CVAE 분기를 탄다

```python
if self.config.use_vae and ACTION in batch and self.training:
```

따라서 `policy.eval()` 상태에서는 latent가

```python
latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32)
```

`use_vae`와 무관하게 이렇게 된다. 그러므로 모든 `vae_encoder*` 텐서는 추론 시 죽은 가중치이며
리맵에서 제외된다. 이는 또한 lowering에 샘플링 단계도 RNG도 없는 이유이기도 하다: ACT
forward pass는 자신의 입력에 대한 결정적 함수다 (spec §3.4).

## 5. 백본 — `FrozenBatchNorm2d`, 검증됨

```python
backbone_model = getattr(torchvision.models, config.vision_backbone)(
    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
    weights=config.pretrained_backbone_weights,
    norm_layer=FrozenBatchNorm2d,          # torchvision.ops.misc
)
self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
```

파일에서 확인되고 둘 다 검증된 두 가지 귀결:

- `FrozenBatchNorm2d`는 `weight`, `bias`, `running_mean`, `running_var`를 등록하며
  **`num_batches_tracked`는 없다** — 100개 백본 키를 grep하면 `running_mean`은 나오지만
  `num_batches_tracked`는 나오지 않는데, 이것이 한눈에 frozen-BN 체크포인트와
  `nn.BatchNorm2d` 체크포인트를 구별하는 방법이다;
- `IntermediateLayerGetter`는 `layer4`에서 잘라내므로 `avgpool`과 `fc`가 없다.
  **feature map**이 transformer로 들어가는 것이며, 풀링된 벡터가 아니다 — 이것이
  `fc`를 대체해 `[out_dim]`을 반환하는 `lower_to_torch`의 일반적인 `VisionEncoder`와의
  가장 큰 구조적 차이다.

우리 lowering에서는 `weights=`에 `None`을 넘긴다: 체크포인트의 백본은 파인튜닝되어
있으므로, ImageNet 가중치를 내려받는 것은 낭비이자 테스트 안에서의 네트워크 의존성이 될
뿐이다.

## 6. 토큰 레이아웃과 위치 임베딩

인코더 시퀀스는 `[latent, robot_state, *image_feature_map_pixels]`이며 시퀀스 우선
`(S, B, C)`다:

- 토큰 0과 1은 각각 `encoder_latent_input_proj(zeros[32])`와
  `encoder_robot_state_input_proj(state)`에서 오며, `encoder_1d_feature_pos_embed.weight`의
  0번, 1번 행을 취한다;
- 이미지 토큰은 `encoder_img_feat_input_proj(backbone(image)["feature_map"])`을
  `b c h w -> (h w) b c`로 재배열한 것이다. 480×640 입력에 대해 ResNet-18의 `layer4`는
  15×20이므로 이미지 토큰 300개, 전체 시퀀스 302개다.

이들의 위치 임베딩은 `ACTSinusoidalPositionEmbedding2d(dim_model // 2)`이며, 학습되는 것이
아니라 계산되는 것이고, 생성된 모듈의 `_act_pos2d`가 이를 정확히 재현한다. 세 가지 세부
사항이 결과를 좌우하며 셋 다 의도적으로 남겨둔 LeRobot 특유의 동작이다:

- 행/열 인덱스는 1로 채운 마스크에 대한 `cumsum`, 즉 `0..H-1`이 **아니라** `1..H`, `1..W`다;
- 마지막 인덱스에 **`1e-6`을 더한 값**으로 정규화한 뒤 `2π`로 스케일한다;
- 채널 연결(concatenation)에서 `y`(행)가 `x`(열)보다 **먼저** 오며, sin/cos는
  `stack(..., dim=-1).flatten(3)`을 통해 서로 엮인다(interleave).

**위치 임베딩은 query와 key에만 더해지며 value에는 절대 더해지지 않는다** — 인코더의
self-attention과 디코더의 self-/cross-attention 모두에서. `torch.nn`의
`TransformerEncoderLayer`는 이를 표현할 수 없으며, 이것이 생성된 모듈이 여전히 내부에서는
`nn.MultiheadAttention`을 쓰면서도 네 개의 레이어 클래스를 손으로 풀어 쓰는 이유다.

디코더는 `torch.zeros(chunk_size, B, dim_model)`에서 시작해 `decoder_pos_embed.weight.unsqueeze(1)`을
query 임베딩으로 사용하며, **인코더의** 위치 임베딩을 key에 얹은 채 인코더 출력에
cross-attend한다.

## 7. Post-norm 레이어 순서

`pre_norm: false`일 때 각 인코더 레이어는

```
x = norm1(x + self_attn(q=x+pos, k=x+pos, v=x))
x = norm2(x + linear2(relu(linear1(x))))
```

이며, 각 디코더 레이어는 이 둘 사이에
`x = norm2(x + multihead_attn(q=x+dpos, k=enc+epos, v=enc))`를 끼워 넣고, 최종 norm의
번호를 `norm3`으로 다시 매긴다. Dropout(`0.1`)은 모듈 안에 존재하지만 파라미터를 갖지 않고
`eval()`에서는 항등(identity)이므로, lowering은 이를 생략한다.

## 8. 측정 결과

이 체크포인트, `lerobot` 0.6.1, `torch` 2.11.0+cpu에 대한
`cargo test -p es-policy act -- --nocapture`:

```
RAN act_checkpoint: lerobot 0.6.1 torch 2.11.0+cpu shape [100, 14] max_abs 0e0 max_rel 0e0
```

**비트 단위로 동일함** — spec §8.9의 tier-4 `1e-5` 안에 여유롭게 들어가며, 테스트는
허용오차가 아니라 비트 단위 결과를 단언(assert)하므로 향후의 발산이 그 안에 숨을 수 없다.
