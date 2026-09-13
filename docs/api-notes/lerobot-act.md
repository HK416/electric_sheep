# LeRobot ACT checkpoint — verified layout

**Pinned version: `lerobot` 0.6.1** (`torch` 2.11.0+cpu, `torchvision` 0.26.0+cpu), read against
the real checkpoint `lerobot/act_aloha_sim_transfer_cube_human`
(`blake3` of `model.safetensors` is printed by `cargo test -p es-policy act -- --nocapture`;
the file is 206 766 560 bytes, 242 tensors, all `F32`, no `__metadata__`).

Unlike `docs/api-notes/lerobot-config.md`, **nothing in this file is `unverified`**: every key
name, shape and default below was read out of that checkpoint and its `config.json`, and every
behavioural claim was checked by running `lerobot.policies.act.modeling_act.ACTPolicy` in the
project venv. `crates/es-policy/src/lerobot.rs` implements exactly this file;
`crates/es-policy/tests/act_checkpoint.rs` is the oracle that keeps the two honest (spec §1.4).

The checkpoint is never committed. It is downloaded into `target/lerobot-cache/` and the test
finds it through `ES_ACT_CHECKPOINT`:

```
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('lerobot/act_aloha_sim_transfer_cube_human', \
    local_dir='target/lerobot-cache/act_aloha_sim_transfer_cube_human')"
```

## 1. What `select_action` actually is at 0.6.1

`ACTPolicy.select_action` is a queue in front of `predict_action_chunk`: it calls the model
once every `n_action_steps` and pops one action per call. `predict_action_chunk` is the whole
forward pass — `self.model(batch)[0]`, sliced to `n_action_steps` by `select_action`
afterwards. **The oracle compares the chunk**, which is the tensor spec §8.9 defines the
tolerance over.

**Normalization left the policy in 0.6.x.** It moved to a separate processor pipeline
(`lerobot.processor.normalize_processor`), so `ACTPolicy` no longer has `normalize_inputs` /
`unnormalize_outputs` submodules and loading this checkpoint logs

```
WARNING:root:Unexpected key(s) when loading model: ['normalize_inputs.buffer_observation_images_top.mean', ...]
```

— eight buffers the policy silently ignores. `predict_action_chunk` therefore takes
**already-normalized** inputs and returns **normalized** actions. The statistics are still in
the checkpoint, so our lowering carries them as nodes 9/10/11 and `python/act_ref.py` applies
the same ones around the reference call. The formula is LeRobot's `MEAN_STD`:

```
forward:  (x - mean) / (std + 1e-8)          # eps = 1e-8, NormalizerProcessorStep.eps
inverse:  x * std + mean
```

## 2. `config.json` — the fields the lowering reads

The real file, verbatim in the fields that matter:

| field | value here | used for |
|---|---|---|
| `type` | `"act"` | discriminator; anything else is refused |
| `input_features` | `observation.images.top: {VISUAL, [3, 480, 640]}`, `observation.state: {STATE, [14]}` | input names and shapes |
| `output_features` | `action: {ACTION, [14]}` | `action_dim` |
| `normalization_mapping` | `{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}` | only `MEAN_STD` is lowered |
| `chunk_size` | `100` | prediction horizon `H`, and `decoder_pos_embed`'s row count |
| `n_action_steps` | `100` | execution length `K` |
| `n_obs_steps` | `1` | only `1` is lowered |
| `vision_backbone` | `"resnet18"` | torchvision model name; fixes the backbone width at 512 |
| `pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | **ignored by the lowering** — the checkpoint carries fine-tuned backbone weights, so it is built with `weights=None` and loaded from the file |
| `replace_final_stride_with_dilation` | `false` | `replace_stride_with_dilation=[False, False, <this>]` |
| `pre_norm` | `false` | post-norm only is lowered |
| `dim_model` | `512` | `d_model`; the camera embedding uses `dim_model // 2` |
| `n_heads` | `8` | |
| `dim_feedforward` | `3200` | |
| `feedforward_activation` | `"relu"` | only `relu` is lowered |
| `n_encoder_layers` | `4` | |
| `n_decoder_layers` | `1` | |
| `use_vae` | `true` | training-only, see §4 |
| `latent_dim` | `32` | width of the zero latent at inference |
| `n_vae_encoder_layers` | `4` | training-only |
| `temporal_ensemble_coeff` | `null` | non-null is refused: it is runtime scheduling, not a forward pass |
| `dropout`, `kl_weight`, `optimizer_*`, `device`, `use_amp` | — | training/deployment; ignored |

`device: "cuda"` is in the file and is a *training* record; loading on a CPU box logs
`Device 'cuda' is not available. Switching to 'cpu'` and is not an error.

## 3. `model.safetensors` — key names and the remap

242 tensors. `crates/es-policy/src/lerobot.rs`'s `remap_act_keys` maps LeRobot prefixes onto our
`nodes.<id>.<param>` scheme (`WEIGHT_PREFIX`), longest-prefix-first so `model.encoder.` cannot
swallow `model.encoder_latent_input_proj.`:

| LeRobot prefix | our node | count | shapes (this checkpoint) |
|---|---|---|---|
| `model.backbone.` | `nodes.0.` | 100 | torchvision's; a prefix claim, not enumerated |
| `model.encoder_img_feat_input_proj.` | `nodes.1.` | 2 | `weight [512, 512, 1, 1]`, `bias [512]` |
| `model.encoder_robot_state_input_proj.` | `nodes.2.` | 2 | `weight [512, 14]`, `bias [512]` |
| `model.encoder_latent_input_proj.` | `nodes.3.` | 2 | `weight [512, 32]`, `bias [512]` |
| `model.encoder_1d_feature_pos_embed.` | `nodes.4.` | 1 | `weight [2, 512]` — one token for the latent, one for the state |
| `model.encoder.` | `nodes.5.` | 48 | 4 × `layers.<i>.{self_attn,linear1,linear2,norm1,norm2}` |
| `model.decoder_pos_embed.` | `nodes.6.` | 1 | `weight [100, 512]` — the DETR object queries |
| `model.decoder.` | `nodes.7.` | 20 | 1 × 18 layer tensors + `norm.{weight,bias}` |
| `model.action_head.` | `nodes.8.` | 2 | `weight [14, 512]`, `bias [14]` |
| `normalize_inputs.buffer_observation_state.` | `nodes.9.` | 2 | `mean [14]`, `std [14]` |
| `normalize_inputs.buffer_observation_images_top.` | `nodes.10.` | 2 | `mean [3, 1, 1]`, `std [3, 1, 1]` |
| `unnormalize_outputs.buffer_action.` | `nodes.11.` | 2 | `mean [14]`, `std [14]` |

The image buffer name is the feature name with dots replaced by underscores — LeRobot's own
`nn.ModuleDict` key sanitizer, since a `ModuleDict` key may not contain a dot. The same rule
names our `forward` keywords (`observation.images.top` → `observation_images_top`).

**Image statistics are per-channel `[3, 1, 1]`, not per-pixel.** This settles the open question
in `lerobot-config.md`'s "Dataset `stats`" section for `VISUAL` features.

### Dropped: 58 tensors, all training-only

| prefix | count | why |
|---|---|---|
| `model.vae_encoder.` | 48 | CVAE encoder, §4 |
| `model.vae_encoder_cls_embed.`, `_robot_state_input_proj.`, `_action_input_proj.`, `_latent_output_proj.` | 7 | same |
| `model.vae_encoder_pos_enc` | 1 | fixed sinusoidal buffer `[1, 102, 512]`, same |
| `normalize_targets.buffer_action.` | 2 | normalizes the *training target*; there is no target at inference |

### Attention parameter names

`ACTEncoderLayer`/`ACTDecoderLayer` hold plain `nn.MultiheadAttention`, `nn.Linear` and
`nn.LayerNorm`, so their `state_dict` names are `torch.nn`'s: `self_attn.in_proj_weight`
`[1536, 512]`, `self_attn.in_proj_bias` `[1536]`, `self_attn.out_proj.{weight,bias}`,
`linear1 [3200, 512]`, `linear2 [512, 3200]`, `norm1`/`norm2` (and `norm3` plus
`multihead_attn.*` on the decoder). That is why a LeRobot checkpoint drops into our lowering
after a prefix rename and nothing more.

**The encoder has no final norm.** `ACTEncoder.norm` is `nn.LayerNorm` only under `pre_norm`,
otherwise `nn.Identity()`; with `pre_norm: false` there is no `model.encoder.norm.*` in the
file. `ACTDecoder.norm` is always a `LayerNorm` and is always present.

## 4. `use_vae` is training-only — verified

`ACT.forward` takes the CVAE branch under

```python
if self.config.use_vae and ACTION in batch and self.training:
```

so with `policy.eval()` the latent is

```python
latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32)
```

regardless of `use_vae`. Every `vae_encoder*` tensor is therefore dead weight at inference and
is dropped by the remap. This is also why the lowering has no sampling step and no RNG: the ACT
forward pass is a deterministic function of its inputs (spec §3.4).

## 5. The backbone — `FrozenBatchNorm2d`, verified

```python
backbone_model = getattr(torchvision.models, config.vision_backbone)(
    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
    weights=config.pretrained_backbone_weights,
    norm_layer=FrozenBatchNorm2d,          # torchvision.ops.misc
)
self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
```

Two consequences visible in the file and both confirmed:

- `FrozenBatchNorm2d` registers `weight`, `bias`, `running_mean`, `running_var` and **no
  `num_batches_tracked`** — grepping the 100 backbone keys finds `running_mean` and finds no
  `num_batches_tracked`, which is how you tell a frozen-BN checkpoint from a `nn.BatchNorm2d`
  one at a glance;
- `IntermediateLayerGetter` truncates at `layer4`, so `avgpool` and `fc` are absent. The
  **feature map** is what feeds the transformer, never a pooled vector — which is the single
  biggest structural difference from `lower_to_torch`'s generic `VisionEncoder`, whose
  `_backbone` replaces `fc` and returns `[out_dim]`.

`weights=` is passed `None` in our lowering: the checkpoint's backbone is fine-tuned, so
downloading ImageNet weights would be both wasteful and a network dependency in a test.

## 6. Token layout and positional embeddings

The encoder sequence is `[latent, robot_state, *image_feature_map_pixels]`, sequence-first
`(S, B, C)`:

- tokens 0 and 1 come from `encoder_latent_input_proj(zeros[32])` and
  `encoder_robot_state_input_proj(state)`, and take rows 0 and 1 of
  `encoder_1d_feature_pos_embed.weight`;
- the image tokens are `encoder_img_feat_input_proj(backbone(image)["feature_map"])`
  rearranged `b c h w -> (h w) b c`. For a 480×640 input, ResNet-18 `layer4` is 15×20, so
  300 image tokens and a sequence of 302.

Their positional embedding is `ACTSinusoidalPositionEmbedding2d(dim_model // 2)`, computed —
not learned — and reproduced exactly by `_act_pos2d` in the generated module. Three details are
load-bearing and all three are LeRobot quirks kept deliberately:

- row/column indices are `cumsum` over a ones mask, i.e. `1..H` and `1..W`, **not** `0..H-1`;
- they are normalized by the last index **plus `1e-6`**, then scaled to `2π`;
- `y` (rows) comes **before** `x` (columns) in the channel concatenation, and sin/cos are
  interleaved via `stack(..., dim=-1).flatten(3)`.

**The positional embedding is added to the query and key only, never to the value** — in both
the encoder self-attention and the decoder's self- and cross-attention. `torch.nn`'s
`TransformerEncoderLayer` cannot express that, which is why the generated module writes the
four layer classes out by hand while still using `nn.MultiheadAttention` inside them.

The decoder starts from `torch.zeros(chunk_size, B, dim_model)` and uses
`decoder_pos_embed.weight.unsqueeze(1)` as the query embedding, cross-attending to the encoder
output with the **encoder's** positional embedding on the keys.

## 7. Post-norm layer order

With `pre_norm: false` each encoder layer is

```
x = norm1(x + self_attn(q=x+pos, k=x+pos, v=x))
x = norm2(x + linear2(relu(linear1(x))))
```

and each decoder layer inserts `x = norm2(x + multihead_attn(q=x+dpos, k=enc+epos, v=enc))`
between the two, renumbering the final norm to `norm3`. Dropout (`0.1`) is present in the
module but holds no parameters and is the identity under `eval()`, so the lowering omits it.

## 8. Measured result

`cargo test -p es-policy act -- --nocapture` against this checkpoint, `lerobot` 0.6.1,
`torch` 2.11.0+cpu:

```
RAN act_checkpoint: lerobot 0.6.1 torch 2.11.0+cpu shape [100, 14] max_abs 0e0 max_rel 0e0
```

**Bitwise identical** — comfortably inside spec §8.9's tier-4 `1e-5`, and the test asserts the
bitwise result rather than the tolerance so a future divergence cannot hide inside it.
