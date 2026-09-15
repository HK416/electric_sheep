//! Loading a real `LeRobot` ACT checkpoint (spec 1.4, spec 8.9's M1 gate).
//!
//! Spec 8.9 makes one claim falsifiable: *load LeRobot's ACT checkpoint and produce the same
//! action for the same observation*. That is what this module exists for, and
//! `docs/api-notes/lerobot-act.md` pins the checkpoint layout it was written against.
//!
//! Three things are worth knowing before reading on.
//!
//! - **The architecture parameters are not in the Learning IR.** Spec 8.3's
//!   `TemporalEncoder { Transformer }` carries a width and nothing else — no layer count, no
//!   feed-forward width, no latent dimension — so `lower_to_torch` cannot emit DETR-style ACT
//!   from a `LearningGraph` alone. They come from the checkpoint's own `config.json`
//!   ([`ActConfig`]), and [`act_policy`] projects that config back onto the spec 8.4 contract
//!   the compiler does check. Widening the IR node instead is a spec change, not a packet.
//! - **`es-data` is layer 10 and this crate is layer 8** (spec 4.2), so its
//!   `lerobot_config::LeRobotConfig` cannot be used here. [`ActConfig`] reads the same file
//!   with the fields this lowering needs; the two are checked against one real config, not
//!   against each other.
//! - **The VAE encoder is training-only.** `ACT.forward` takes the CVAE branch only under
//!   `self.training`, so at inference the latent is zeros and every `vae_encoder*` tensor is
//!   dead weight. [`remap_act_keys`] drops them, along with `normalize_targets` (the training
//!   target normalizer). `INV-16` still holds throughout: safetensors in, safetensors out,
//!   nothing unpickled.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_ir::learning::{
    ActionExecutionMode, ArchKind, BaseModelRef, PolicyContract, PolicyHandle, RuntimeHints,
    TensorPort, WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use serde::{Deserialize, Serialize};

use crate::lower::torch::lowering_hash;
use crate::lower::{LowerError, TorchModule};
use crate::runtime::PolicyError;
use crate::weights::{parse_header, SafetensorsEntry, WEIGHT_PREFIX};

/// One entry of `input_features` / `output_features`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Feature {
    /// `"VISUAL"`, `"STATE"` or `"ACTION"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The per-frame tensor shape, channels-first for `VISUAL`.
    pub shape: Vec<u64>,
}

/// The fields of a `LeRobot` ACT `config.json` this lowering reads.
///
/// Anything not listed is training configuration (`optimizer_*`, `kl_weight`, `dropout`) or a
/// deployment concern, and is ignored rather than guessed at. `serde` defaults mirror
/// `lerobot.policies.act.configuration_act.ACTConfig`, so a config that omits a field still
/// loads — see `docs/api-notes/lerobot-act.md`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ActConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub input_features: BTreeMap<String, Feature>,
    pub output_features: BTreeMap<String, Feature>,
    #[serde(default)]
    pub normalization_mapping: BTreeMap<String, String>,
    pub chunk_size: u32,
    pub n_action_steps: u32,
    #[serde(default = "one")]
    pub n_obs_steps: u32,
    pub vision_backbone: String,
    #[serde(default)]
    pub replace_final_stride_with_dilation: bool,
    #[serde(default)]
    pub pre_norm: bool,
    pub dim_model: u32,
    pub n_heads: u32,
    pub dim_feedforward: u32,
    #[serde(default = "relu")]
    pub feedforward_activation: String,
    pub n_encoder_layers: u32,
    pub n_decoder_layers: u32,
    #[serde(default)]
    pub use_vae: bool,
    #[serde(default = "thirty_two")]
    pub latent_dim: u32,
    #[serde(default)]
    pub temporal_ensemble_coeff: Option<f64>,
}

fn one() -> u32 {
    1
}

fn relu() -> String {
    "relu".to_owned()
}

fn thirty_two() -> u32 {
    32
}

/// The node ids the lowered module uses, fixed so a remapped checkpoint is portable between
/// runs. Section 3 of `docs/api-notes/lerobot-act.md` is this table.
const BACKBONE: u32 = 0;
const IMG_PROJ: u32 = 1;
const STATE_PROJ: u32 = 2;
const LATENT_PROJ: u32 = 3;
const ENC_POS: u32 = 4;
const ENCODER: u32 = 5;
const DEC_POS: u32 = 6;
const DECODER: u32 = 7;
const HEAD: u32 = 8;
const NORM_STATE: u32 = 9;
const NORM_IMAGE: u32 = 10;
const UNNORM_ACTION: u32 = 11;

/// The `LeRobot` prefix -> our prefix table, longest-source-first so `model.encoder.` is not
/// shadowed by `model.encoder_latent_input_proj.`. Prefixes absent from it are training-only
/// and are dropped.
fn prefix_table(camera: &str) -> Vec<(String, String)> {
    let node = |id: u32| format!("{WEIGHT_PREFIX}{id}.");
    let mut table = vec![
        ("model.backbone.".to_owned(), node(BACKBONE)),
        (
            "model.encoder_img_feat_input_proj.".to_owned(),
            node(IMG_PROJ),
        ),
        (
            "model.encoder_robot_state_input_proj.".to_owned(),
            node(STATE_PROJ),
        ),
        (
            "model.encoder_latent_input_proj.".to_owned(),
            node(LATENT_PROJ),
        ),
        (
            "model.encoder_1d_feature_pos_embed.".to_owned(),
            node(ENC_POS),
        ),
        ("model.encoder.".to_owned(), node(ENCODER)),
        ("model.decoder_pos_embed.".to_owned(), node(DEC_POS)),
        ("model.decoder.".to_owned(), node(DECODER)),
        ("model.action_head.".to_owned(), node(HEAD)),
        (
            "normalize_inputs.buffer_observation_state.".to_owned(),
            node(NORM_STATE),
        ),
        (
            format!("normalize_inputs.buffer_{}.", buffer_name(camera)),
            node(NORM_IMAGE),
        ),
        (
            "unnormalize_outputs.buffer_action.".to_owned(),
            node(UNNORM_ACTION),
        ),
    ];
    table.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
    table
}

/// `LeRobot`'s own `nn.ModuleDict` key sanitizer: dots become underscores, because a
/// `ModuleDict` key may not contain one.
fn buffer_name(feature: &str) -> String {
    feature.replace('.', "_")
}

impl ActConfig {
    /// Parse a `config.json`.
    pub fn parse(json: &str) -> Result<Self, LowerError> {
        let cfg: Self = serde_json::from_str(json)
            .map_err(|e| LowerError::Invalid(format!("config.json: {e}")))?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Everything this lowering refuses rather than approximates. Each one would otherwise be
    /// a silently wrong action chunk, which spec 8.9 exists to prevent.
    fn check(&self) -> Result<(), LowerError> {
        let no = |what: String| Err(LowerError::Unsupported(what));
        if self.kind != "act" {
            return no(format!("LeRobot policy type {:?}", self.kind));
        }
        if self.pre_norm {
            return no("ACT{pre_norm}".to_owned());
        }
        if self.feedforward_activation != "relu" {
            return no(format!("ACT{{{}}}", self.feedforward_activation));
        }
        if self.n_obs_steps != 1 {
            return no(format!("ACT{{n_obs_steps = {}}}", self.n_obs_steps));
        }
        if self.temporal_ensemble_coeff.is_some() {
            // Spec 8.6's `ActionExecutionMode::TemporalEnsemble` is a runtime scheduling
            // decision, not a forward pass; one `infer` cannot observe it.
            return no("ACT{temporal_ensemble_coeff}".to_owned());
        }
        for (feature, mode) in &self.normalization_mapping {
            if mode != "MEAN_STD" {
                return no(format!("ACT{{{feature} = {mode}}}"));
            }
        }
        if self.cameras().len() != 1 {
            return no(format!(
                "ACT{{{} cameras}} (one camera is what the pinned checkpoints carry; more \
                 needs a node-id rule for the extra image normalizers)",
                self.cameras().len()
            ));
        }
        if self.state_dim().is_none() {
            return no("ACT{no observation.state}".to_owned());
        }
        if self.action_dim().is_none() {
            return no("ACT{no action feature}".to_owned());
        }
        if self.dim_model % 2 != 0 {
            return no(format!("ACT{{dim_model = {}}}", self.dim_model));
        }
        Ok(())
    }

    /// The `VISUAL` input features, in `BTreeMap` order (spec 3.4: no map-iteration dependence).
    pub fn cameras(&self) -> Vec<&str> {
        self.input_features
            .iter()
            .filter(|(_, f)| f.kind == "VISUAL")
            .map(|(name, _)| name.as_str())
            .collect()
    }

    fn feature(&self, kind: &str) -> Option<u64> {
        self.input_features
            .values()
            .chain(self.output_features.values())
            .find(|f| f.kind == kind)
            .and_then(|f| f.shape.first().copied())
    }

    pub fn state_dim(&self) -> Option<u64> {
        self.feature("STATE")
    }

    pub fn action_dim(&self) -> Option<u64> {
        self.feature("ACTION")
    }

    /// `[3, H, W]` of the single camera.
    pub fn image_shape(&self) -> Vec<u64> {
        self.cameras()
            .first()
            .and_then(|c| self.input_features.get(*c))
            .map(|f| f.shape.clone())
            .unwrap_or_default()
    }

    /// The backbone's final feature width, which is `encoder_img_feat_input_proj`'s fan-in.
    fn backbone_channels(&self) -> u64 {
        match self.vision_backbone.as_str() {
            "resnet18" | "resnet34" => 512,
            _ => 2048,
        }
    }

    /// The `forward` keyword for one feature: `observation.state` -> `observation_state`,
    /// because a Python keyword argument may not contain a dot.
    fn input_name(feature: &str) -> String {
        buffer_name(feature)
    }
}

/// `LeRobot` key -> our `nodes.<id>.<param>` key, for the keys that survive inference.
///
/// Keys with no entry are absent from the result: `model.vae_encoder*` (training-only, see the
/// module docs) and `normalize_targets.*` (the training target normalizer). A caller that
/// wants to know what was dropped compares the two key sets.
pub fn remap_act_keys<'a>(
    cfg: &ActConfig,
    keys: impl IntoIterator<Item = &'a str>,
) -> BTreeMap<String, String> {
    let camera = cfg
        .cameras()
        .first()
        .copied()
        .unwrap_or_default()
        .to_owned();
    let table = prefix_table(&camera);
    let mut out = BTreeMap::new();
    for key in keys {
        if let Some((src, dst)) = table.iter().find(|(src, _)| key.starts_with(src.as_str())) {
            out.insert(key.to_owned(), format!("{dst}{}", &key[src.len()..]));
        }
    }
    out
}

/// One output tensor and the file its bytes are in: [`remap_checkpoint`] draws from two.
type SourcedEntry<'a> = (String, &'a [u8], SafetensorsEntry);

/// The three normalizers, from `LeRobot` 0.6.x's **separate processor state file**.
///
/// Two checkpoint layouts exist and both are in use (`docs/api-notes/lerobot-act.md`):
///
/// - the pinned upstream one (`lerobot/act_aloha_sim_transfer_cube_human`) carries the
///   statistics inside `model.safetensors` as `normalize_inputs.buffer_<feature>.{mean,std}`,
///   which [`prefix_table`] already maps;
/// - anything `lerobot-train` 0.6.1 writes carries them in
///   `policy_preprocessor_step_<n>_normalizer_processor.safetensors`, keyed
///   `<feature>.{mean,std,min,max,count}` with no prefix at all, because 0.6.x moved
///   normalization out of `ACTPolicy` into a processor pipeline.
///
/// The second file is the caller's `stats`. Its entries are only *added* where the first layout
/// left a gap ([`remap_checkpoint`] uses `or_insert`), so a checkpoint carrying both is read the
/// old way and neither layout needs a flag.
fn normalizer_stats<'a>(
    cfg: &ActConfig,
    stats: Option<&'a [u8]>,
) -> Result<Vec<SourcedEntry<'a>>, PolicyError> {
    let Some(stats) = stats else {
        return Ok(Vec::new());
    };
    let file = parse_header(stats)?;
    let camera = cfg
        .cameras()
        .first()
        .copied()
        .unwrap_or_default()
        .to_owned();
    let mut out = Vec::new();
    for (node, feature) in [
        (NORM_STATE, "observation.state".to_owned()),
        (NORM_IMAGE, camera),
        (UNNORM_ACTION, "action".to_owned()),
    ] {
        for stat in ["mean", "std"] {
            let key = format!("{feature}.{stat}");
            let entry = file.get(&key).ok_or_else(|| {
                PolicyError::Safetensors(format!(
                    "the normalizer state file has no \"{key}\"; it holds {:?}",
                    file.keys().take(8).collect::<Vec<_>>()
                ))
            })?;
            out.push((
                format!("{WEIGHT_PREFIX}{node}.{stat}"),
                stats,
                entry.clone(),
            ));
        }
    }
    Ok(out)
}

/// The `__metadata__` key under which [`remap_checkpoint`] records the config that decides the
/// module.
///
/// Spec 8.3's node parameters do not carry ACT's architecture (the module docs say why), so the
/// only honest place for it is *inside the checkpoint*: the bytes are hashed into
/// `WeightsRef::hash` and therefore into `policy_hash`, so the IR still decides which module
/// runs — transitively, through the hash it declares (spec 5.3). A bundle whose weights were
/// swapped is refused before this is ever read.
pub const ACT_CONFIG_KEY: &str = "es.lerobot.act.config";

/// The ACT config a [`remap_checkpoint`] output carries, or `None` for any other checkpoint.
pub fn embedded_config(bytes: &[u8]) -> Result<Option<ActConfig>, PolicyError> {
    let Some(raw) = crate::weights::metadata(bytes, ACT_CONFIG_KEY)? else {
        return Ok(None);
    };
    ActConfig::parse(&raw)
        .map(Some)
        .map_err(|e| PolicyError::Safetensors(format!("{ACT_CONFIG_KEY}: {e}")))
}

/// Rewrite a `LeRobot` checkpoint into our key scheme, dropping the training-only tensors.
///
/// Byte-level: the header is rebuilt and each surviving tensor's bytes are copied verbatim, so
/// no value is ever decoded, rounded or re-rounded. Output order is `BTreeMap` order, which
/// makes the result a function of the input alone (spec 3.4) and its `blake3` a stable
/// `WeightsRef::hash`.
pub fn remap_checkpoint(
    cfg: &ActConfig,
    bytes: &[u8],
    stats: Option<&[u8]>,
) -> Result<Vec<u8>, PolicyError> {
    let file = parse_header(bytes)?;
    let map = remap_act_keys(cfg, file.keys().map(String::as_str));

    // `(source file, its entry)` per output key, because the normalization statistics may come
    // from a second file — see `normalizer_stats`.
    let mut kept: BTreeMap<String, (&[u8], SafetensorsEntry)> = BTreeMap::new();
    for (from, to) in &map {
        kept.insert(to.clone(), (bytes, file[from].clone()));
    }
    for (name, source, entry) in normalizer_stats(cfg, stats)? {
        kept.entry(name).or_insert((source, entry));
    }

    let mut header = serde_json::Map::new();
    let mut data = Vec::with_capacity(bytes.len());
    for (name, (source, entry)) in kept {
        let header_len = u64::from_le_bytes(source[..8].try_into().map_err(|_| {
            PolicyError::Safetensors("file is shorter than the 8-byte length".into())
        })?) as usize;
        let base = 8 + header_len;
        let (a, b) = entry.offsets;
        let slice = source
            .get(base + a as usize..base + b as usize)
            .ok_or_else(|| {
                PolicyError::Safetensors(format!("entry \"{name}\" runs past the end"))
            })?;
        let start = data.len();
        data.extend_from_slice(slice);
        header.insert(
            name,
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }

    // The config travels with the weights (see `ACT_CONFIG_KEY`). `parse_header` skips
    // `__metadata__`, so this adds no key any validator has to learn about.
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            ACT_CONFIG_KEY: serde_json::to_string(cfg)
                .map_err(|e| PolicyError::Safetensors(e.to_string()))?,
        }),
    );
    let header = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| PolicyError::Safetensors(e.to_string()))?;
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}

// --- the lowering ---------------------------------------------------------------------------

/// The parts of the generated file that do not depend on the config: the backbone provider, the
/// 2-D sinusoidal camera embedding, and the four transformer modules.
///
/// The encoder and decoder layers are written out rather than built from
/// `nn.TransformerEncoderLayer`, for one reason: ACT adds the positional embedding to the
/// **query and key only**, not to the value, and `torch.nn`'s layer has no way to express that.
/// Their parameter names are `torch.nn`'s all the same, which is why a `LeRobot` checkpoint drops
/// straight in. Dropout is absent because `eval()` makes it the identity and it holds no
/// parameters.
const ACT_PY: &str = r#"

def _act_backbone(name, dilation, channels):
    # Referenced, never re-implemented: torchvision is the provider. `FrozenBatchNorm2d` is
    # what LeRobot trained with, so the checkpoint has no `num_batches_tracked` entries.
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    model = getattr(torchvision.models, name)(
        replace_stride_with_dilation=[False, False, dilation],
        weights=None,
        norm_layer=FrozenBatchNorm2d,
    )
    return IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"})


def _act_pos2d(x, half):
    # ACTSinusoidalPositionEmbedding2d: row and column indices normalized into [0, 2*pi],
    # interleaved sin/cos, y before x. The 1e-6 and the 1..H (not 0..H-1) ranges are LeRobot's
    # own, kept because the checkpoint was trained under them.
    ones = torch.ones_like(x[0, :1])
    y = ones.cumsum(1, dtype=torch.float32)
    z = ones.cumsum(2, dtype=torch.float32)
    two_pi = 6.283185307179586
    y = y / (y[:, -1:, :] + 1e-6) * two_pi
    z = z / (z[:, :, -1:] + 1e-6) * two_pi
    inv = 10000 ** (2 * (torch.arange(half, dtype=torch.float32) // 2) / half)
    z = z.unsqueeze(-1) / inv
    y = y.unsqueeze(-1) / inv
    px = torch.stack((z[..., 0::2].sin(), z[..., 1::2].cos()), dim=-1).flatten(3)
    py = torch.stack((y[..., 0::2].sin(), y[..., 1::2].cos()), dim=-1).flatten(3)
    return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)


class _ActEncoderLayer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads)
        self.linear1 = nn.Linear(d, ffn)
        self.linear2 = nn.Linear(ffn, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, pos):
        q = k = x + pos
        x = self.norm1(x + self.self_attn(q, k, value=x)[0])
        return self.norm2(x + self.linear2(torch.relu(self.linear1(x))))


class _ActEncoder(nn.Module):
    # `norm` is `nn.Identity()` under post-norm, so it holds no parameters and is left out.
    def __init__(self, n, d, heads, ffn):
        super().__init__()
        self.layers = nn.ModuleList([_ActEncoderLayer(d, heads, ffn) for _ in range(n)])

    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x, pos)
        return x


class _ActDecoderLayer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads)
        self.multihead_attn = nn.MultiheadAttention(d, heads)
        self.linear1 = nn.Linear(d, ffn)
        self.linear2 = nn.Linear(ffn, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x, enc, dpos, epos):
        q = k = x + dpos
        x = self.norm1(x + self.self_attn(q, k, value=x)[0])
        cross = self.multihead_attn(query=x + dpos, key=enc + epos, value=enc)[0]
        x = self.norm2(x + cross)
        return self.norm3(x + self.linear2(torch.relu(self.linear1(x))))


class _ActDecoder(nn.Module):
    def __init__(self, n, d, heads, ffn):
        super().__init__()
        self.layers = nn.ModuleList([_ActDecoderLayer(d, heads, ffn) for _ in range(n)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, enc, dpos, epos):
        for layer in self.layers:
            x = layer(x, enc, dpos, epos)
        return self.norm(x)


class _ActNorm(nn.Module):
    # LeRobot's MEAN_STD normalizer, statistics carried by the checkpoint rather than by the
    # IR: `(x - mean) / (std + 1e-8)` forward, `x * std + mean` inverse.
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def inverse(self, x):
        return x * self.std + self.mean
"#;

/// Lower an ACT checkpoint's `config.json` to the `PyTorch` module `TorchRuntime` runs.
///
/// The result has the same contract as [`crate::lower_to_torch`]'s: a file defining
/// `class EsPolicy(nn.Module)` whose `forward(**inputs) -> dict` takes the declared input names
/// and returns `{"actions": [n_action_steps, action_dim]}`, already unnormalized, plus the
/// safetensors keys it needs.
pub fn lower_act(cfg: &ActConfig) -> Result<TorchModule, LowerError> {
    cfg.check()?;
    let d = u64::from(cfg.dim_model);
    let ffn = u64::from(cfg.dim_feedforward);
    let latent = u64::from(cfg.latent_dim);
    let chunk = u64::from(cfg.chunk_size);
    let heads = u64::from(cfg.n_heads);
    let channels = cfg.backbone_channels();
    let state = cfg
        .state_dim()
        .expect("check() rejected a config without one");
    let action = cfg
        .action_dim()
        .expect("check() rejected a config without one");
    let camera = cfg.cameras()[0].to_owned();
    let image = cfg.image_shape();
    let img_in = ActConfig::input_name(&camera);
    let state_in = ActConfig::input_name("observation.state");

    let mut keys: Vec<String> = Vec::new();
    let mut shapes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    // The backbone's internal names belong to torchvision, so it is a prefix claim.
    keys.push(format!("{WEIGHT_PREFIX}{BACKBONE}.*"));
    let mut exact = |id: u32, sub: &str, shape: Vec<u64>| {
        let key = format!("{WEIGHT_PREFIX}{id}.{sub}");
        keys.push(key.clone());
        shapes.insert(key, shape);
    };

    exact(IMG_PROJ, "weight", vec![d, channels, 1, 1]);
    exact(IMG_PROJ, "bias", vec![d]);
    exact(STATE_PROJ, "weight", vec![d, state]);
    exact(STATE_PROJ, "bias", vec![d]);
    exact(LATENT_PROJ, "weight", vec![d, latent]);
    exact(LATENT_PROJ, "bias", vec![d]);
    // One token for the latent, one for the robot state.
    exact(ENC_POS, "weight", vec![2, d]);
    for (id, n, cross) in [
        (ENCODER, cfg.n_encoder_layers, false),
        (DECODER, cfg.n_decoder_layers, true),
    ] {
        for i in 0..n {
            let mut attn = vec!["self_attn"];
            if cross {
                attn.push("multihead_attn");
            }
            for a in attn {
                exact(
                    id,
                    &format!("layers.{i}.{a}.in_proj_weight"),
                    vec![3 * d, d],
                );
                exact(id, &format!("layers.{i}.{a}.in_proj_bias"), vec![3 * d]);
                exact(id, &format!("layers.{i}.{a}.out_proj.weight"), vec![d, d]);
                exact(id, &format!("layers.{i}.{a}.out_proj.bias"), vec![d]);
            }
            exact(id, &format!("layers.{i}.linear1.weight"), vec![ffn, d]);
            exact(id, &format!("layers.{i}.linear1.bias"), vec![ffn]);
            exact(id, &format!("layers.{i}.linear2.weight"), vec![d, ffn]);
            exact(id, &format!("layers.{i}.linear2.bias"), vec![d]);
            for norm in 1..=(if cross { 3 } else { 2 }) {
                exact(id, &format!("layers.{i}.norm{norm}.weight"), vec![d]);
                exact(id, &format!("layers.{i}.norm{norm}.bias"), vec![d]);
            }
        }
    }
    // Post-norm leaves the encoder's `norm` an `nn.Identity`; the decoder always has one.
    exact(DECODER, "norm.weight", vec![d]);
    exact(DECODER, "norm.bias", vec![d]);
    exact(DEC_POS, "weight", vec![chunk, d]);
    exact(HEAD, "weight", vec![action, d]);
    exact(HEAD, "bias", vec![action]);
    let channel_stat = vec![3, 1, 1];
    for (id, shape) in [
        (NORM_STATE, vec![state]),
        (NORM_IMAGE, channel_stat.clone()),
        (UNNORM_ACTION, vec![action]),
    ] {
        exact(id, "mean", shape.clone());
        exact(id, "std", shape);
    }

    let mut source = String::new();
    source.push_str(
        "# Generated by es-policy from a LeRobot ACT config.json (spec 8.7, spec 8.9).\n\
         # Do not edit. The checkpoint layout: docs/api-notes/lerobot-act.md\n\
         import torch\n\
         import torch.nn as nn\n",
    );
    source.push_str(ACT_PY);
    let dilation = if cfg.replace_final_stride_with_dilation {
        "True"
    } else {
        "False"
    };
    let _ = write!(
        source,
        r#"

class EsPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.n{BACKBONE} = _act_backbone({backbone:?}, {dilation}, {channels})
        self.n{IMG_PROJ} = nn.Conv2d({channels}, {d}, kernel_size=1)
        self.n{STATE_PROJ} = nn.Linear({state}, {d})
        self.n{LATENT_PROJ} = nn.Linear({latent}, {d})
        self.n{ENC_POS} = nn.Embedding(2, {d})
        self.n{ENCODER} = _ActEncoder({n_enc}, {d}, {heads}, {ffn})
        self.n{DEC_POS} = nn.Embedding({chunk}, {d})
        self.n{DECODER} = _ActDecoder({n_dec}, {d}, {heads}, {ffn})
        self.n{HEAD} = nn.Linear({d}, {action})
        self.n{NORM_STATE} = _ActNorm([{state}])
        self.n{NORM_IMAGE} = _ActNorm([3, 1, 1])
        self.n{UNNORM_ACTION} = _ActNorm([{action}])

    def forward(self, **inputs):
        # inputs: {state_in} [B, {state}], {img_in} [B, {image:?}], both unnormalized --
        # LeRobot normalizes inside the policy, from the buffers below, so the Observation IR
        # must hand these over in raw units. One observation is accepted unbatched, which is
        # what `es eval run` feeds (a spec 7.4 tensor carries no batch axis).
        state = inputs[{state_in:?}]
        image = inputs[{img_in:?}]
        state = self.n{NORM_STATE}(state if state.dim() == 2 else state.unsqueeze(0))
        image = self.n{NORM_IMAGE}(image if image.dim() == 4 else image.unsqueeze(0))
        batch = state.shape[0]
        # use_vae = {use_vae}: the CVAE encoder runs under `self.training` only, so at inference
        # the latent is zeros and all of its tensors are dropped from the checkpoint.
        latent = torch.zeros(batch, {latent}, dtype=torch.float32)
        tokens = torch.stack([self.n{LATENT_PROJ}(latent), self.n{STATE_PROJ}(state)], dim=0)
        pos = self.n{ENC_POS}.weight.unsqueeze(1)
        feature_map = self.n{BACKBONE}(image)["feature_map"]
        cam_pos = _act_pos2d(feature_map, {half})
        feature_map = self.n{IMG_PROJ}(feature_map)
        rows, cols = feature_map.shape[2], feature_map.shape[3]
        feature_map = feature_map.permute(2, 3, 0, 1).reshape(rows * cols, batch, {d})
        cam_pos = cam_pos.permute(2, 3, 0, 1).reshape(rows * cols, 1, {d})
        tokens = torch.cat([tokens, feature_map], dim=0)
        pos = torch.cat([pos, cam_pos], dim=0)
        memory = self.n{ENCODER}(tokens, pos)
        queries = torch.zeros({chunk}, batch, {d}, dtype=torch.float32)
        decoded = self.n{DECODER}(queries, memory, self.n{DEC_POS}.weight.unsqueeze(1), pos)
        actions = self.n{HEAD}(decoded.transpose(0, 1))
        actions = self.n{UNNORM_ACTION}.inverse(actions)
        return {{"actions": actions[0][:{execute}]}}
"#,
        backbone = cfg.vision_backbone,
        n_enc = cfg.n_encoder_layers,
        n_dec = cfg.n_decoder_layers,
        half = d / 2,
        use_vae = cfg.use_vae,
        execute = cfg.n_action_steps,
    );

    Ok(TorchModule {
        lowering_hash: lowering_hash(&source),
        source,
        weight_keys: keys,
        weight_shapes: shapes,
    })
}

/// The spec 8.4 contract this checkpoint satisfies.
///
/// This is the projection the M1 gate is really about: a real `LeRobot` artefact described in the
/// IR's own terms, so `execution_hash` (spec 5.3) has a `policy` slot to hash. `base_model`
/// records the `LeRobot` file the weights came from and `weights` the remapped file actually
/// loaded, so both are in the chain.
pub fn act_policy(
    cfg: &ActConfig,
    source_uri: &str,
    source_hash: [u8; 32],
    weights_hash: [u8; 32],
) -> Result<PolicyHandle, LowerError> {
    cfg.check()?;
    let state = cfg
        .state_dim()
        .expect("check() rejected a config without one");
    let action = cfg
        .action_dim()
        .expect("check() rejected a config without one");
    let camera = cfg.cameras()[0].to_owned();

    let port = |name: String, shape: Vec<u64>, unit: Unit| {
        (
            name.clone(),
            TensorPort::new(
                &name,
                PortType {
                    elem: ElemType::F32,
                    shape: Shape::new(shape),
                    unit,
                    frame: Frame::Policy,
                    time: TimeRef::Tick,
                    image: None,
                },
            ),
        )
    };
    let inputs = [
        port(
            ActConfig::input_name(&camera),
            cfg.image_shape(),
            Unit::Normalized { lo: 0.0, hi: 1.0 },
        ),
        port(
            ActConfig::input_name("observation.state"),
            vec![state],
            Unit::Angle,
        ),
    ]
    .into_iter()
    .collect();

    Ok(PolicyHandle {
        architecture: ArchKind::Act,
        base_model: Some(BaseModelRef {
            uri: source_uri.to_owned(),
            hash: source_hash,
            license: "apache-2.0".to_owned(),
        }),
        weights: WeightsRef::Safetensors {
            path: String::new(),
            hash: weights_hash,
        },
        contract: PolicyContract {
            inputs,
            observation_window: cfg.n_obs_steps,
            action_dim: action as u32,
            horizon: cfg.chunk_size,
            execute_chunk: cfg.n_action_steps,
            // Not in `config.json`: LeRobot has no control-rate field. The lowering runs one
            // forward pass, so nothing here reaches the module; it is contract metadata a
            // Deployment IR would override (spec 9).
            replanning_hz: 0.0,
            execution_mode: ActionExecutionMode::OpenLoopChunk,
            runtime: RuntimeHints {
                dtype: ElemType::F32,
                expected_latency_ms: 0.0,
                deadline_ms: 0.0,
            },
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned `lerobot/act_aloha_sim_transfer_cube_human` config, trimmed to the fields
    /// this module reads (`docs/api-notes/lerobot-act.md` section 2).
    const CONFIG: &str = r#"{
        "type": "act",
        "n_obs_steps": 1,
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "input_features": {
            "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [14]}
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "chunk_size": 100,
        "n_action_steps": 100,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "replace_final_stride_with_dilation": false,
        "pre_norm": false,
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "feedforward_activation": "relu",
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "use_vae": true,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "temporal_ensemble_coeff": null,
        "dropout": 0.1,
        "kl_weight": 10.0
    }"#;

    fn cfg() -> ActConfig {
        ActConfig::parse(CONFIG).unwrap()
    }

    #[test]
    fn the_pinned_config_parses_to_the_documented_shape() {
        let c = cfg();
        assert_eq!(c.cameras(), vec!["observation.images.top"]);
        assert_eq!((c.state_dim(), c.action_dim()), (Some(14), Some(14)));
        assert_eq!(c.image_shape(), vec![3, 480, 640]);
        assert_eq!(
            (c.chunk_size, c.n_action_steps, c.latent_dim),
            (100, 100, 32)
        );
        assert!(c.use_vae && !c.pre_norm);
    }

    #[test]
    fn the_remap_drops_exactly_the_training_only_tensors() {
        let c = cfg();
        let keys = [
            "model.backbone.conv1.weight",
            "model.encoder.layers.0.linear1.weight",
            "model.encoder_latent_input_proj.weight",
            "model.encoder_1d_feature_pos_embed.weight",
            "model.decoder.norm.weight",
            "model.decoder_pos_embed.weight",
            "model.action_head.bias",
            "normalize_inputs.buffer_observation_state.mean",
            "normalize_inputs.buffer_observation_images_top.std",
            "unnormalize_outputs.buffer_action.mean",
            // Training-only, and therefore absent from the result.
            "model.vae_encoder.layers.0.linear1.weight",
            "model.vae_encoder_cls_embed.weight",
            "model.vae_encoder_pos_enc",
            "normalize_targets.buffer_action.mean",
        ];
        let map = remap_act_keys(&c, keys);
        assert_eq!(map.len(), 10, "{map:?}");
        // `model.encoder.` must not swallow `model.encoder_latent_input_proj.`.
        assert_eq!(
            map["model.encoder.layers.0.linear1.weight"],
            "nodes.5.layers.0.linear1.weight"
        );
        assert_eq!(
            map["model.encoder_latent_input_proj.weight"],
            "nodes.3.weight"
        );
        assert_eq!(
            map["model.encoder_1d_feature_pos_embed.weight"],
            "nodes.4.weight"
        );
        assert_eq!(map["model.decoder.norm.weight"], "nodes.7.norm.weight");
        assert_eq!(map["model.decoder_pos_embed.weight"], "nodes.6.weight");
        assert_eq!(map["model.backbone.conv1.weight"], "nodes.0.conv1.weight");
        assert_eq!(
            map["normalize_inputs.buffer_observation_images_top.std"],
            "nodes.10.std"
        );
        assert_eq!(
            map["unnormalize_outputs.buffer_action.mean"],
            "nodes.11.mean"
        );
        for dropped in [
            "model.vae_encoder.layers.0.linear1.weight",
            "normalize_targets.buffer_action.mean",
        ] {
            assert!(!map.contains_key(dropped), "{dropped} survived");
        }
    }

    #[test]
    fn the_lowering_declares_the_keys_the_checkpoint_carries() {
        let m = lower_act(&cfg()).unwrap();
        // 1 backbone prefix claim + 8 projections/embeddings + 4 encoder layers x 12 + 1
        // decoder layer x 18 + decoder norm x 2 + head x 2 + 3 normalizers x 2.
        assert_eq!(m.weight_keys.len(), 1 + 8 + 4 * 12 + 18 + 2 + 2 + 6);
        assert_eq!(m.weight_shapes["nodes.1.weight"], vec![512, 512, 1, 1]);
        assert_eq!(m.weight_shapes["nodes.2.weight"], vec![512, 14]);
        assert_eq!(m.weight_shapes["nodes.3.weight"], vec![512, 32]);
        assert_eq!(m.weight_shapes["nodes.4.weight"], vec![2, 512]);
        assert_eq!(
            m.weight_shapes["nodes.5.layers.3.self_attn.in_proj_weight"],
            vec![1536, 512]
        );
        assert_eq!(
            m.weight_shapes["nodes.5.layers.0.linear1.weight"],
            vec![3200, 512]
        );
        assert_eq!(
            m.weight_shapes["nodes.7.layers.0.multihead_attn.in_proj_bias"],
            vec![1536]
        );
        assert_eq!(m.weight_shapes["nodes.7.layers.0.norm3.weight"], vec![512]);
        assert_eq!(m.weight_shapes["nodes.6.weight"], vec![100, 512]);
        assert_eq!(m.weight_shapes["nodes.8.weight"], vec![14, 512]);
        assert_eq!(m.weight_shapes["nodes.10.mean"], vec![3, 1, 1]);
        // The encoder is post-norm, so it has no final LayerNorm to load.
        assert!(!m.weight_shapes.contains_key("nodes.5.norm.weight"));
        // The backbone is torchvision's to name.
        assert!(!m.weight_shapes.contains_key("nodes.0.*"));
    }

    #[test]
    fn the_lowering_is_byte_identical_across_runs() {
        let a = lower_act(&cfg()).unwrap();
        let b = lower_act(&cfg()).unwrap();
        assert_eq!(a.source, b.source);
        assert_eq!(a.lowering_hash, b.lowering_hash);
        assert_eq!(a.weight_keys, b.weight_keys);
    }

    #[test]
    fn the_generated_source_says_what_it_must() {
        let m = lower_act(&cfg()).unwrap();
        for expected in [
            "_act_backbone(\"resnet18\", False, 512)",
            "_ActEncoder(4, 512, 8, 3200)",
            "_ActDecoder(1, 512, 8, 3200)",
            "inputs[\"observation_images_top\"]",
            "inputs[\"observation_state\"]",
            "torch.zeros(batch, 32, dtype=torch.float32)",
            "_act_pos2d(feature_map, 256)",
            "actions[0][:100]",
            "FrozenBatchNorm2d",
        ] {
            assert!(
                m.source.contains(expected),
                "missing `{expected}`:\n{}",
                m.source
            );
        }
        // INV-16: nothing here can unpickle a checkpoint.
        assert!(!m.source.contains("torch.load"));
        assert!(!m.source.contains("pickle"));
        // The VAE encoder is training-only and must not appear at all.
        assert!(!m.source.contains("vae_encoder"));
    }

    #[test]
    fn a_config_this_lowering_cannot_honour_is_refused() {
        for (field, replacement) in [
            (r#""type": "act""#, r#""type": "diffusion""#),
            (r#""pre_norm": false"#, r#""pre_norm": true"#),
            (
                r#""feedforward_activation": "relu""#,
                r#""feedforward_activation": "gelu""#,
            ),
            (r#""n_obs_steps": 1"#, r#""n_obs_steps": 2"#),
            (
                r#""temporal_ensemble_coeff": null"#,
                r#""temporal_ensemble_coeff": 0.01"#,
            ),
            (r#""VISUAL": "MEAN_STD""#, r#""VISUAL": "MIN_MAX""#),
        ] {
            let json = CONFIG.replace(field, replacement);
            let err = ActConfig::parse(&json).unwrap_err();
            assert!(
                matches!(err, LowerError::Unsupported(_)),
                "{field} -> {replacement}: {err}"
            );
        }
    }

    #[test]
    fn the_policy_handle_carries_the_spec_8_4_contract() {
        let p = act_policy(
            &cfg(),
            "lerobot/act_aloha_sim_transfer_cube_human",
            [1u8; 32],
            [2u8; 32],
        )
        .unwrap();
        assert_eq!(p.architecture, ArchKind::Act);
        assert_eq!((p.contract.action_dim, p.contract.horizon), (14, 100));
        assert_eq!(p.contract.execute_chunk, 100);
        assert_eq!(p.contract.observation_window, 1);
        assert_eq!(p.weights.hash(), &[2u8; 32]);
        assert_eq!(p.base_model.as_ref().unwrap().hash, [1u8; 32]);
        assert_eq!(p.contract.inputs.len(), 2);
        assert!(p.contract.inputs.contains_key("observation_images_top"));
    }

    /// The byte-level repack keeps every surviving tensor's bytes untouched.
    #[test]
    fn a_repacked_checkpoint_keeps_its_values() {
        use crate::weights::{write_safetensors, Checkpoint};
        let mut file: Checkpoint = BTreeMap::new();
        file.insert(
            "model.action_head.bias".to_owned(),
            (vec![2], vec![1.5, -2.5]),
        );
        file.insert(
            "model.vae_encoder_cls_embed.weight".to_owned(),
            (vec![1], vec![9.0]),
        );
        file.insert(
            "normalize_inputs.buffer_observation_state.mean".to_owned(),
            (vec![3], vec![0.25, 0.5, 0.75]),
        );
        let out = remap_checkpoint(&cfg(), &write_safetensors(&file), None).unwrap();
        let header = parse_header(&out).unwrap();
        assert_eq!(
            header.keys().collect::<Vec<_>>(),
            vec!["nodes.8.bias", "nodes.9.mean"]
        );
        let (a, b) = header["nodes.8.bias"].offsets;
        let base = 8 + usize::try_from(u64::from_le_bytes(out[..8].try_into().unwrap())).unwrap();
        let bytes = &out[base + a as usize..base + b as usize];
        assert_eq!(
            bytes,
            [1.5f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat()
        );
        // Deterministic (spec 3.4).
        assert_eq!(
            out,
            remap_checkpoint(&cfg(), &write_safetensors(&file), None).unwrap()
        );
    }
}
