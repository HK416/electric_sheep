//! `LearningGraph` to a `PyTorch` module (spec 8.7), the reference oracle of spec 1.4.
//!
//! The node table, the weight-key scheme and what is deferred live in
//! `docs/design/learning-lowering.md`. This file implements that table and nothing else: one
//! `nn.Module` member per parameterised node, one forward line per node, in
//! `Graph::topo_order()` order. No fusion, no reordering, no cleverness — an oracle that is
//! hard to read against the spec is not an oracle.
//!
//! Determinism is a hard requirement, not a nicety: `lowering_hash` is a component of the
//! compiler identity, so the same graph must produce byte-identical source. Everything here
//! iterates a `BTreeMap` or a `Vec` built in topological order (spec 3.4).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_ir::graph::{IrNode, NodeId};
use es_ir::learning::{
    BetaSchedule, DiffusionScheduler, FusionKind, HeadKind, LearningGraph, LearningNode,
    NormalizeDir, PredictionType, StateEncoderKind, StatsSource, TemporalKind, VarianceType,
    VisionBackbone,
};
use es_ir::types::ElemType;
use es_ir::Diagnostic;

use crate::weights::WEIGHT_PREFIX;

/// Domain separator for [`TorchModule::lowering_hash`].
const LOWERING_TAG: &str = "es.lowering.torch.v1";

/// Attention heads in a lowered `TemporalEncoder { Transformer }`. Spec 8.3's node parameters
/// do not carry a head count, so it is a lowering constant; a checkpoint that disagrees fails
/// the shape check rather than being silently reinterpreted (design note section 3).
const TRANSFORMER_HEADS: u64 = 8;

/// Ends of the linear beta schedule — `diffusers`' `beta_start` / `beta_end` defaults, which
/// `LeRobot` does not override. Spec 8.3 does not carry them, so they stay lowering constants
/// exactly as `TRANSFORMER_HEADS` is (design note section 8).
const BETA_START: f32 = 1e-4;
const BETA_END: f32 = 0.02;

/// `betas_for_alpha_bar`'s cap in `diffusers`.
const MAX_BETA: f32 = 0.999;

/// The parameters of one `DDPMScheduler` / `DDIMScheduler`, unpacked from
/// [`HeadKind::Diffusion`] so [`diffusion_schedule`] takes one argument instead of seven.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DiffusionParams {
    pub n_steps: u32,
    pub scheduler: DiffusionScheduler,
    pub num_train_timesteps: u32,
    pub beta_schedule: BetaSchedule,
    pub variance_type: VarianceType,
    pub clip_sample: bool,
    pub clip_sample_range: f32,
}

impl DiffusionParams {
    /// The scheduler configuration of a `Diffusion` head, `None` for any other head.
    pub(crate) fn from_head(kind: &HeadKind) -> Option<Self> {
        match *kind {
            HeadKind::Diffusion {
                n_steps,
                scheduler,
                num_train_timesteps,
                beta_schedule,
                variance_type,
                clip_sample,
                clip_sample_range,
                ..
            } => Some(Self {
                n_steps,
                scheduler,
                num_train_timesteps,
                beta_schedule,
                variance_type,
                clip_sample,
                clip_sample_range,
            }),
            _ => None,
        }
    }
}

/// The schedule of one sampler, in `diffusers`' own terms (design note section 8.3).
///
/// Per inference step `i`, with `t = timesteps[i]`:
///
/// ```text
/// x0 = (x - sqrt_1mab[i] * eps) / sqrt_ab[i]          # pred_original_sample
/// x0 = clamp(x0, -range, range)                       # iff clip_sample
/// x  = c0[i] * x0 + cx[i] * x + ce[i] * eps + sigma[i] * z_t
/// ```
///
/// `cx` is zero under `Ddim` and `ce` is zero under `Ddpm`; splitting them is what lets one
/// loop serve both while `clip_sample` stays applicable to `x0` (an affine `c1 * x + c3 * eps`
/// cannot express the clamp).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Schedule {
    /// `alphas_cumprod` over the **training** grid, length `num_train_timesteps`.
    pub alpha_bar: Vec<f32>,
    /// The inference timesteps, descending, as `set_timesteps` produces them.
    pub timesteps: Vec<u32>,
    pub sqrt_ab: Vec<f32>,
    pub sqrt_1mab: Vec<f32>,
    pub c0: Vec<f32>,
    pub cx: Vec<f32>,
    pub ce: Vec<f32>,
    /// Zero at `t == 0` and everywhere under `Ddim` (eta = 0), which is what makes DDIM consume
    /// no `noise_<t>` buffer at all.
    pub sigma: Vec<f32>,
    /// `DDPMScheduler::_get_variance` per inference step, before the square root.
    pub variance: Vec<f32>,
}

/// `betas` over the training grid, `diffusers`' `DDPMScheduler.__init__` (`betas_for_alpha_bar`
/// for `squaredcos_cap_v2`), in f32 with a fixed op order (spec 3.4).
fn betas(n: usize, schedule: BetaSchedule) -> Vec<f32> {
    match schedule {
        // torch.linspace(beta_start, beta_end, n).
        BetaSchedule::Linear => (0..n)
            .map(|t| {
                let f = if n == 1 {
                    0.0
                } else {
                    t as f32 / (n - 1) as f32
                };
                BETA_START + (BETA_END - BETA_START) * f
            })
            .collect(),
        // alpha_bar(t) = cos((t + 0.008) / 1.008 * pi / 2)^2, beta_i = 1 - abar(t2)/abar(t1).
        BetaSchedule::SquaredcosCapV2 => {
            let abar = |t: f32| {
                let c = es_math::approx::cos((t + 0.008) / 1.008 * std::f32::consts::FRAC_PI_2);
                c * c
            };
            (0..n)
                .map(|i| {
                    let t1 = i as f32 / n as f32;
                    let t2 = (i + 1) as f32 / n as f32;
                    (1.0 - abar(t2) / abar(t1)).min(MAX_BETA)
                })
                .collect()
        }
    }
}

/// The schedule `diffusers` would build for `p`, in f32 with a fixed op order (spec 3.4).
///
/// `alpha_bar` is a running product, so it is accumulated ascending over the **training** grid;
/// the inference timesteps are then subsampled out of it exactly as `set_timesteps` does under
/// `timestep_spacing = "leading"` (the default of both schedulers):
/// `timesteps = (arange(0, n_steps) * (num_train_timesteps // n_steps))[::-1]`.
pub(crate) fn diffusion_schedule(p: &DiffusionParams) -> Schedule {
    let train = p.num_train_timesteps.max(1) as usize;
    let n = p.n_steps.max(1) as usize;
    let beta = betas(train, p.beta_schedule);
    let mut alpha_bar: Vec<f32> = Vec::with_capacity(train);
    for (t, b) in beta.iter().enumerate() {
        let prev = if t == 0 { 1.0 } else { alpha_bar[t - 1] };
        alpha_bar.push(prev * (1.0 - b));
    }

    let stride = (train / n).max(1);
    let timesteps: Vec<u32> = (0..n).rev().map(|i| (i * stride) as u32).collect();

    let sqrt = es_math::approx::sqrt;
    let mut s = Schedule {
        timesteps: timesteps.clone(),
        alpha_bar: alpha_bar.clone(),
        sqrt_ab: Vec::with_capacity(n),
        sqrt_1mab: Vec::with_capacity(n),
        c0: Vec::with_capacity(n),
        cx: Vec::with_capacity(n),
        ce: Vec::with_capacity(n),
        sigma: Vec::with_capacity(n),
        variance: Vec::with_capacity(n),
    };
    for (i, t) in timesteps.iter().enumerate() {
        let ab = alpha_bar[*t as usize];
        // `previous_timestep`: the next entry of `timesteps`, and `self.one` past the end.
        let prev = match timesteps.get(i + 1) {
            Some(pt) => alpha_bar[*pt as usize],
            None => 1.0,
        };
        let cur_alpha = ab / prev;
        let cur_beta = 1.0 - cur_alpha;
        // `_get_variance`, clamped to 1e-20 as diffusers does before any log. `variance_type`
        // is a `DDPMScheduler` field: `DDIMScheduler` has none and always reports the
        // posterior variance, which under eta = 0 is unused anyway.
        let small = ((1.0 - prev) / (1.0 - ab) * cur_beta).max(1e-20);
        let var = match (p.variance_type, p.scheduler) {
            (VarianceType::FixedLarge, DiffusionScheduler::Ddpm) => cur_beta.max(1e-20),
            _ => small,
        };
        s.sqrt_ab.push(sqrt(ab));
        s.sqrt_1mab.push(sqrt(1.0 - ab));
        s.variance.push(var);
        match p.scheduler {
            DiffusionScheduler::Ddpm => {
                s.c0.push(sqrt(prev) * cur_beta / (1.0 - ab));
                s.cx.push(sqrt(cur_alpha) * (1.0 - prev) / (1.0 - ab));
                s.ce.push(0.0);
                s.sigma.push(if *t == 0 { 0.0 } else { sqrt(var) });
            }
            // eta = 0: x = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev) * eps, deterministic.
            DiffusionScheduler::Ddim => {
                s.c0.push(sqrt(prev));
                s.cx.push(0.0);
                s.ce.push(sqrt(1.0 - prev));
                s.sigma.push(0.0);
            }
            DiffusionScheduler::DpmSolver => unreachable!("rejected as Unsupported before here"),
        }
    }
    s
}

/// The sampler runtime the generated file needs: a sinusoidal timestep embedding and the one
/// denoising network both heads share. Emitted only when a sampler head is present.
const SAMPLER_PY: &str = r"

def _sinusoidal(t, dim):
    # DDPM timestep embedding: [sin(t * w_i), cos(t * w_i)], w_i = exp(-ln(1e4) * i / half).
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32)
    a = t * torch.exp(-9.210340371976184 * i / half)
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class _Denoiser(nn.Module):
    # eps_theta for Diffusion, v_theta for FlowMatching: one hidden layer over
    # [x_t, cond, temb]. Spec 8.3 carries no width for it, so the hidden width is the
    # conditioning width (design note section 8). A UNet is a later packet.
    def __init__(self, x_dim, cond):
        super().__init__()
        self.l0 = nn.Linear(x_dim + cond + cond, cond)
        self.l1 = nn.Linear(cond, x_dim)

    def forward(self, x, cond, temb):
        return self.l1(torch.relu(self.l0(torch.cat([x, cond, temb], dim=-1))))
";

/// The DDPM/DDIM head. `noise` is the initial `x_T` and arrives as a declared graph input; the
/// per-step draws are non-trainable `noise_<t>` buffers in the checkpoint, so a run is a
/// function of its inputs alone (spec 3.4: no global RNG).
const DDPM_PY: &str = r#"

class _DdpmHead(nn.Module):
    # One diffusers DDPM/DDIM step, on the subsampled inference timesteps. `clip` is the
    # clip_sample_range or None. See docs/design/learning-lowering.md section 8.3.
    def __init__(self, x_dim, cond, horizon, action_dim, ts, sa, sb, c0, cx, ce, sigma, clip):
        super().__init__()
        self.net = _Denoiser(x_dim, cond)
        self.cond, self.horizon, self.action_dim = cond, horizon, action_dim
        self.ts, self.sa, self.sb = ts, sa, sb
        self.c0, self.cx, self.ce, self.sigma, self.clip = c0, cx, ce, sigma, clip
        for i, t in enumerate(ts):
            if sigma[i] != 0.0:
                self.register_buffer("noise_%d" % t, torch.zeros(x_dim))

    def forward(self, cond, noise):
        x = noise.reshape(-1)
        for i, t in enumerate(self.ts):
            eps = self.net(x, cond, _sinusoidal(float(t), self.cond))
            x0 = (x - self.sb[i] * eps) / self.sa[i]
            if self.clip is not None:
                x0 = torch.clamp(x0, -self.clip, self.clip)
            x = self.c0[i] * x0 + self.cx[i] * x + self.ce[i] * eps
            if self.sigma[i] != 0.0:
                x = x + self.sigma[i] * getattr(self, "noise_%d" % t)
        return x.reshape(self.horizon, self.action_dim)
"#;

/// The flow-matching head: Euler integration of `dx/dt = v_theta(x, cond, t)` from the noise at
/// `t = 0` to the action at `t = 1`, in `n_steps` equal steps.
const FLOW_PY: &str = r"

class _FlowHead(nn.Module):
    def __init__(self, x_dim, cond, horizon, action_dim, n_steps):
        super().__init__()
        self.net = _Denoiser(x_dim, cond)
        self.cond, self.horizon, self.action_dim = cond, horizon, action_dim
        self.n_steps = n_steps

    def forward(self, cond, noise):
        x = noise.reshape(-1)
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            v = self.net(x, cond, _sinusoidal(i / self.n_steps, self.cond))
            x = x + dt * v
        return x.reshape(self.horizon, self.action_dim)
";

/// A lowered graph: a complete `PyTorch` file plus the checkpoint contract it implies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorchModule {
    /// A self-contained Python file defining `class EsPolicy(nn.Module)` whose
    /// `forward(**inputs) -> dict` takes the graph's declared input names.
    pub source: String,
    /// Safetensors keys this module needs. An entry ending in `.*` is a *prefix claim* over an
    /// opaque sub-module (a torchvision backbone, a `torch.nn` transformer) whose internal
    /// parameter names are not ours to enumerate; every other entry is an exact key and has a
    /// shape in [`weight_shapes`](Self::weight_shapes).
    pub weight_keys: Vec<String>,
    pub weight_shapes: BTreeMap<String, Vec<u64>>,
    /// `blake3(LOWERING_TAG || source)`.
    pub lowering_hash: [u8; 32],
}

impl TorchModule {
    /// Whether `key` ends in `.*`, i.e. covers a sub-module rather than one tensor.
    pub fn is_prefix_claim(key: &str) -> bool {
        key.ends_with(".*")
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LowerError {
    /// A node kind with no `PyTorch` lowering in this packet. The payload is the node kind and
    /// the variant that was unsupported, e.g. `"PolicyHead{Diffusion}"`.
    #[error("no PyTorch lowering for {0} (see docs/design/learning-lowering.md section 7)")]
    Unsupported(String),
    #[error("the graph does not validate: {0}")]
    Invalid(String),
    #[error("node {node}: input port \"{port}\" is fed by neither an edge nor a graph input")]
    Unfed { node: u32, port: String },
    #[error("node {node}: {message}")]
    Shape { node: u32, message: String },
}

/// Lower `graph` to a `PyTorch` module.
///
/// The graph is validated first: an oracle built from an invalid IR proves nothing.
pub fn lower_to_torch(graph: &LearningGraph) -> Result<TorchModule, LowerError> {
    let errors: Vec<String> = graph
        .validate()
        .into_iter()
        .filter(Diagnostic::is_error)
        .map(|d| format!("{} {}", d.code, d.message))
        .collect();
    if !errors.is_empty() {
        return Err(LowerError::Invalid(errors.join("; ")));
    }
    if graph.policy.contract.runtime.dtype != ElemType::F32 {
        return Err(LowerError::Unsupported(format!(
            "dtype {:?}",
            graph.policy.contract.runtime.dtype
        )));
    }

    let mut lo = Lowering::default();
    let order = graph
        .nodes
        .topo_order()
        .map_err(|d| LowerError::Invalid(format!("{} {}", d.code, d.message)))?;

    for id in order {
        let node = &graph.nodes.nodes[&id];
        let args = inputs_of(graph, id, node)?;
        let expr = lo.node_expr(id, node, &args)?;
        let out = node
            .outputs()
            .first()
            .map(|p| p.name.clone())
            .ok_or_else(|| LowerError::Shape {
                node: id.0,
                message: "the node has no output port".to_owned(),
            })?;
        lo.body
            .push(format!("        {} = {expr}", value_of(id, &out)));
    }

    let mut returns = Vec::new();
    for (port, at) in graph.outputs.iter().zip(&graph.nodes.outputs) {
        returns.push(format!("{:?}: {}", port.name, value_of(at.node, &at.port)));
    }

    Ok(lo.finish(&returns))
}

/// The local holding output port `port` of node `id`.
fn value_of(id: NodeId, port: &str) -> String {
    format!("v{}_{}", id.0, port)
}

/// The Python expression feeding each declared input port of `node`, in declared order.
fn inputs_of(
    graph: &LearningGraph,
    id: NodeId,
    node: &LearningNode,
) -> Result<Vec<String>, LowerError> {
    let mut args = Vec::new();
    for port in node.inputs() {
        let edge = graph
            .nodes
            .edges
            .iter()
            .find(|e| e.to.node == id && e.to.port == port.name);
        if let Some(e) = edge {
            args.push(value_of(e.from.node, &e.from.port));
            continue;
        }
        let boundary = graph
            .nodes
            .inputs
            .iter()
            .position(|r| r.node == id && r.port == port.name)
            .and_then(|i| graph.inputs.get(i));
        match boundary {
            Some(declared) => args.push(format!("inputs[{:?}]", declared.name)),
            None => {
                return Err(LowerError::Unfed {
                    node: id.0,
                    port: port.name,
                })
            }
        }
    }
    Ok(args)
}

/// The last dimension of a node's `i`-th declared input port — the feature width every module
/// here is sized by.
fn in_dim(node: &LearningNode, i: usize, id: NodeId) -> Result<u64, LowerError> {
    node.inputs()
        .get(i)
        .and_then(|p| p.ty.shape.dims().last().copied())
        .filter(|d| *d > 0)
        .ok_or_else(|| LowerError::Shape {
            node: id.0,
            message: format!("input port {i} has no usable trailing dimension"),
        })
}

fn unsupported(kind: &str, variant: &impl std::fmt::Debug) -> LowerError {
    LowerError::Unsupported(format!("{kind}{{{variant:?}}}"))
}

#[derive(Default)]
struct Lowering {
    members: Vec<String>,
    body: Vec<String>,
    keys: Vec<String>,
    shapes: BTreeMap<String, Vec<u64>>,
    needs_torchvision: bool,
    needs_sampler: bool,
    needs_ddpm: bool,
    needs_flow: bool,
}

/// What [`Lowering::sampler`] worked out for a sampler head.
struct Sampler {
    cond: u64,
    /// Flattened action-chunk width `H * action_dim`, the sampler's state width.
    x_dim: u64,
    call: String,
}

impl Lowering {
    fn member(&mut self, id: NodeId, ctor: &str) {
        self.members
            .push(format!("        self.n{} = {ctor}", id.0));
    }

    /// Declares an exact key. `sub` is the `state_dict` path below the node's member.
    fn exact(&mut self, id: NodeId, sub: &str, shape: Vec<u64>) {
        let key = format!("{WEIGHT_PREFIX}{}.{sub}", id.0);
        self.keys.push(key.clone());
        self.shapes.insert(key, shape);
    }

    /// Declares a prefix claim over an opaque sub-module.
    fn claim(&mut self, id: NodeId) {
        self.keys.push(format!("{WEIGHT_PREFIX}{}.*", id.0));
    }

    /// Everything a sampler head needs that does not depend on which sampler it is: the shapes,
    /// the two-input check, the denoiser's keys, and the forward call.
    ///
    /// Both heads take **two** inputs — the conditioning feature and the initial noise `x_T`,
    /// which is a declared graph input so a sample is a function of its inputs (design note
    /// section 8). Spec 8.3's node parameters carry no hidden width, so the denoiser's hidden
    /// width and the timestep-embedding width are both the conditioning width.
    fn sampler(
        &mut self,
        id: NodeId,
        node: &LearningNode,
        args: &[String],
        horizon: u32,
        action_dim: u32,
        n_steps: u32,
    ) -> Result<Sampler, LowerError> {
        let shape = |message: String| LowerError::Shape {
            node: id.0,
            message,
        };
        if n_steps == 0 {
            return Err(shape("a sampler needs at least one step".to_owned()));
        }
        let ports = node.inputs();
        if ports.len() != 2 || args.len() != 2 {
            return Err(shape(format!(
                "a Diffusion/FlowMatching head takes two inputs, the conditioning feature and \
                 the initial noise x_T, and declares {} (design note section 8)",
                ports.len()
            )));
        }
        let x_dim = u64::from(horizon) * u64::from(action_dim);
        let noise: u64 = ports[1].ty.shape.dims().iter().product();
        if noise != x_dim {
            return Err(shape(format!(
                "the noise port holds {noise} elements, the action chunk needs {x_dim}"
            )));
        }
        let cond = in_dim(node, 0, id)?;
        if cond % 2 != 0 {
            return Err(shape(format!(
                "the timestep embedding is half sin and half cos, so the conditioning width \
                 must be even, got {cond}"
            )));
        }

        self.needs_sampler = true;
        self.exact(id, "net.l0.weight", vec![cond, x_dim + cond + cond]);
        self.exact(id, "net.l0.bias", vec![cond]);
        self.exact(id, "net.l1.weight", vec![x_dim, cond]);
        self.exact(id, "net.l1.bias", vec![x_dim]);
        Ok(Sampler {
            cond,
            x_dim,
            call: format!("self.n{}({}, {})", id.0, args[0], args[1]),
        })
    }

    /// A `nn.Linear` member plus its two exact keys.
    fn linear(&mut self, id: NodeId, in_dim: u64, out_dim: u64) {
        self.member(id, &format!("nn.Linear({in_dim}, {out_dim})"));
        self.exact(id, "weight", vec![out_dim, in_dim]);
        self.exact(id, "bias", vec![out_dim]);
    }

    // One arm per node kind. Splitting it would hide the table it implements.
    fn node_expr(
        &mut self,
        id: NodeId,
        node: &LearningNode,
        args: &[String],
    ) -> Result<String, LowerError> {
        let k = id.0;
        match node {
            LearningNode::VisionEncoder {
                backbone, out_dim, ..
            } => {
                let name = match backbone {
                    VisionBackbone::ResNet18 => "resnet18",
                    VisionBackbone::ResNet34 => "resnet34",
                    other => return Err(unsupported("VisionEncoder", other)),
                };
                self.needs_torchvision = true;
                self.member(id, &format!("_backbone({name:?}, {out_dim})"));
                self.claim(id);
                // A torchvision backbone is `nn.BatchNorm2d` all the way down, and that
                // refuses a 3-D input outright ("expected 4D input (got 3D input)"). The IR
                // port is one image (spec 8.3 has no batch axis — spec 5.2 gives the
                // inference domain its own batch size), so run it as a one-image batch, the
                // same trade the token-less `TemporalEncoder` arm below makes.
                Ok(format!("self.n{k}({}.unsqueeze(0)).squeeze(0)", args[0]))
            }

            LearningNode::StateEncoder { kind, out_dim, .. } => match kind {
                StateEncoderKind::Identity => Ok(args[0].clone()),
                StateEncoderKind::Mlp { hidden } => {
                    let mut dims = vec![in_dim(node, 0, id)?];
                    dims.extend(hidden.iter().map(|h| u64::from(*h)));
                    dims.push(u64::from(*out_dim));
                    let mut layers = Vec::new();
                    for (i, w) in dims.windows(2).enumerate() {
                        if i > 0 {
                            layers.push("nn.ReLU()".to_owned());
                        }
                        layers.push(format!("nn.Linear({}, {})", w[0], w[1]));
                        let at = layers.len() - 1;
                        self.exact(id, &format!("{at}.weight"), vec![w[1], w[0]]);
                        self.exact(id, &format!("{at}.bias"), vec![w[1]]);
                    }
                    self.member(id, &format!("nn.Sequential({})", layers.join(", ")));
                    Ok(format!("self.n{k}({})", args[0]))
                }
            },

            LearningNode::LanguageEncoder { .. } => {
                Err(LowerError::Unsupported("LanguageEncoder".to_owned()))
            }

            LearningNode::Fusion { kind, out_dim, .. } => {
                let out_dim = u64::from(*out_dim);
                let widths: Result<Vec<u64>, LowerError> =
                    (0..args.len()).map(|i| in_dim(node, i, id)).collect();
                let widths = widths?;
                match kind {
                    FusionKind::Concat => {
                        let cat = format!("torch.cat([{}], dim=-1)", args.join(", "));
                        let sum: u64 = widths.iter().sum();
                        if sum == out_dim {
                            Ok(cat)
                        } else {
                            self.linear(id, sum, out_dim);
                            Ok(format!("self.n{k}({cat})"))
                        }
                    }
                    FusionKind::TokenConcat => {
                        if widths.iter().any(|w| *w != out_dim) {
                            return Err(LowerError::Shape {
                                node: k,
                                message: format!(
                                    "TokenConcat needs every input at width {out_dim}, got {widths:?}"
                                ),
                            });
                        }
                        Ok(format!("torch.cat([{}], dim=-2)", args.join(", ")))
                    }
                    other => Err(unsupported("Fusion", other)),
                }
            }

            LearningNode::TemporalEncoder {
                kind,
                out_dim,
                token_count,
                ..
            } => match kind {
                TemporalKind::None => Ok(args[0].clone()),
                TemporalKind::Transformer => {
                    let d = u64::from(*out_dim);
                    let got = in_dim(node, 0, id)?;
                    if got != d {
                        return Err(LowerError::Shape {
                            node: k,
                            message: format!("a transformer keeps its width: in {got}, out {d}"),
                        });
                    }
                    if d % TRANSFORMER_HEADS != 0 {
                        return Err(LowerError::Shape {
                            node: k,
                            message: format!(
                                "d_model {d} is not a multiple of {TRANSFORMER_HEADS} heads"
                            ),
                        });
                    }
                    self.member(
                        id,
                        &format!(
                            "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model={d}, nhead={TRANSFORMER_HEADS}, batch_first=True), num_layers=1)"
                        ),
                    );
                    self.claim(id);
                    // A pooled feature has no token axis; run it as a one-token sequence.
                    Ok(if *token_count == 0 {
                        format!("self.n{k}({}.unsqueeze(0)).squeeze(0)", args[0])
                    } else {
                        format!("self.n{k}({})", args[0])
                    })
                }
                other => Err(unsupported("TemporalEncoder", other)),
            },

            LearningNode::PolicyHead {
                kind,
                action_dim,
                horizon,
                ..
            } => match kind {
                HeadKind::Regression => {
                    let width = u64::from(*horizon) * u64::from(*action_dim);
                    self.linear(id, in_dim(node, 0, id)?, width);
                    Ok(format!(
                        "self.n{k}({}).reshape({horizon}, {action_dim})",
                        args[0]
                    ))
                }
                HeadKind::Diffusion {
                    n_steps,
                    scheduler,
                    num_train_timesteps,
                    prediction_type,
                    clip_sample,
                    clip_sample_range,
                    ..
                } => {
                    if *scheduler == DiffusionScheduler::DpmSolver {
                        return Err(unsupported("PolicyHead{Diffusion}", scheduler));
                    }
                    // The denoiser is eps_theta; `sample` and `v_prediction` are a different
                    // pred_original_sample and would be a silently wrong chunk (design note 8.3).
                    if *prediction_type != PredictionType::Epsilon {
                        return Err(unsupported("PolicyHead{Diffusion}", prediction_type));
                    }
                    if *num_train_timesteps < *n_steps {
                        return Err(LowerError::Shape {
                            node: id.0,
                            message: format!(
                                "n_steps = {n_steps} inference steps cannot be subsampled out of \
                                 num_train_timesteps = {num_train_timesteps}"
                            ),
                        });
                    }
                    let s = self.sampler(id, node, args, *horizon, *action_dim, *n_steps)?;
                    let sched = diffusion_schedule(
                        &DiffusionParams::from_head(kind).expect("this arm is a Diffusion head"),
                    );
                    for (i, sigma) in sched.sigma.iter().enumerate() {
                        if *sigma != 0.0 {
                            self.exact(id, &format!("noise_{}", sched.timesteps[i]), vec![s.x_dim]);
                        }
                    }
                    self.needs_ddpm = true;
                    let ts = format!(
                        "[{}]",
                        sched
                            .timesteps
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    self.member(
                        id,
                        &format!(
                            "_DdpmHead({}, {}, {horizon}, {action_dim}, {ts}, {}, {}, {}, {}, {}, {}, {})",
                            s.x_dim,
                            s.cond,
                            json_f32s(&sched.sqrt_ab),
                            json_f32s(&sched.sqrt_1mab),
                            json_f32s(&sched.c0),
                            json_f32s(&sched.cx),
                            json_f32s(&sched.ce),
                            json_f32s(&sched.sigma),
                            if *clip_sample {
                                // Debug is shortest-round-trip and always carries a `.`, so it
                                // is a Python float literal, never an int.
                                format!("{clip_sample_range:?}")
                            } else {
                                "None".to_owned()
                            },
                        ),
                    );
                    Ok(s.call)
                }
                HeadKind::FlowMatching { n_steps } => {
                    let s = self.sampler(id, node, args, *horizon, *action_dim, *n_steps)?;
                    self.needs_flow = true;
                    self.member(
                        id,
                        &format!(
                            "_FlowHead({}, {}, {horizon}, {action_dim}, {n_steps})",
                            s.x_dim, s.cond
                        ),
                    );
                    Ok(s.call)
                }
                other => Err(unsupported("PolicyHead", other)),
            },

            LearningNode::PolicyBundle { .. } => {
                Err(LowerError::Unsupported("PolicyBundle".to_owned()))
            }

            // Only `execute_chunk` is a tensor operation; the rest of the node is runtime
            // scheduling (spec 8.6) and is not observable in one `infer`.
            LearningNode::ActionChunker { execute_chunk, .. } => {
                Ok(format!("{}[:{execute_chunk}]", args[0]))
            }

            LearningNode::Normalizer {
                direction, stats, ..
            } => {
                let (first, second) = match stats {
                    StatsSource::MeanStd { mean, std } => (mean, std),
                    StatsSource::MinMax { lo, hi } => (lo, hi),
                    StatsSource::Dataset { .. } => {
                        return Err(LowerError::Unsupported(
                            "Normalizer{Dataset} (resolving a dataset_hash is es-data's job, \
                             layer 10)"
                                .to_owned(),
                        ))
                    }
                };
                // Non-persistent, so the statistics never enter `state_dict` and a checkpoint
                // can never disagree with the IR about them.
                for (suffix, values) in [("a", first), ("b", second)] {
                    self.members.push(format!(
                        "        self.register_buffer(\"n{k}_{suffix}\", torch.tensor({}, dtype=torch.float32), persistent=False)",
                        json_floats(values)
                    ));
                }
                let x = &args[0];
                let (p, q) = (format!("self.n{k}_a"), format!("self.n{k}_b"));
                Ok(match (stats, direction) {
                    (StatsSource::MeanStd { .. }, NormalizeDir::Forward) => {
                        format!("({x} - {p}) / {q}")
                    }
                    (StatsSource::MeanStd { .. }, NormalizeDir::Inverse) => {
                        format!("{x} * {q} + {p}")
                    }
                    (_, NormalizeDir::Forward) => {
                        format!("2.0 * ({x} - {p}) / ({q} - {p}) - 1.0")
                    }
                    (_, NormalizeDir::Inverse) => {
                        format!("({x} + 1.0) * 0.5 * ({q} - {p}) + {p}")
                    }
                })
            }
        }
    }

    fn finish(self, returns: &[String]) -> TorchModule {
        let mut source = String::new();
        source.push_str(
            "# Generated by es-policy from a LearningGraph (spec 8.7). Do not edit.\n\
             # The contract this file implements: docs/design/learning-lowering.md\n\
             import torch\n\
             import torch.nn as nn\n",
        );
        if self.needs_torchvision {
            source.push_str(
                "\n\n\
                 def _backbone(name, out_dim):\n\
                 \x20   # Referenced, never re-implemented: torchvision is the provider.\n\
                 \x20   import torchvision\n\
                 \x20   m = getattr(torchvision.models, name)()\n\
                 \x20   m.fc = nn.Linear(m.fc.in_features, out_dim)\n\
                 \x20   return m\n",
            );
        }
        if self.needs_sampler {
            source.push_str(SAMPLER_PY);
        }
        if self.needs_ddpm {
            source.push_str(DDPM_PY);
        }
        if self.needs_flow {
            source.push_str(FLOW_PY);
        }
        source.push_str(
            "\n\nclass EsPolicy(nn.Module):\n    def __init__(self):\n        super().__init__()\n",
        );
        for m in &self.members {
            let _ = writeln!(source, "{m}");
        }
        source.push_str("\n    def forward(self, **inputs):\n");
        for line in &self.body {
            let _ = writeln!(source, "{line}");
        }
        let _ = writeln!(source, "        return {{{}}}", returns.join(", "));

        TorchModule {
            lowering_hash: lowering_hash(&source),
            source,
            weight_keys: self.keys,
            weight_shapes: self.shapes,
        }
    }
}

/// `blake3(LOWERING_TAG || source)` — [`TorchModule::lowering_hash`], the `compiler` slot of
/// `execution_hash` (spec 5.3). Shared with [`crate::lerobot`], whose ACT lowering produces the
/// same artifact from a checkpoint's `config.json` rather than from a `LearningGraph`.
pub(crate) fn lowering_hash(source: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(LOWERING_TAG.as_bytes());
    h.update(source.as_bytes());
    *h.finalize().as_bytes()
}

/// A Python float list. `serde_json` renders f64 shortest-round-trip, which is both exact and
/// stable, so two lowerings of the same statistics are byte-identical.
fn json_floats(values: &[f64]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_owned())
}

/// The same for f32: shortest round-trip, so the literal Python parses back to the exact f32
/// the schedule computed.
fn json_f32s(values: &[f32]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_ir::learning::testing::act_like;

    fn act() -> LearningGraph {
        act_like(8, 512, 8, 50, 20, 1)
    }

    #[test]
    fn lowering_is_byte_identical_across_runs() {
        let g = act();
        let a = lower_to_torch(&g).unwrap();
        let b = lower_to_torch(&g.clone()).unwrap();
        assert_eq!(a.source, b.source);
        assert_eq!(a.lowering_hash, b.lowering_hash);
        assert_eq!(a.weight_keys, b.weight_keys);
        assert_eq!(a.weight_shapes, b.weight_shapes);
    }

    #[test]
    fn a_different_architecture_moves_the_lowering_hash() {
        let a = lower_to_torch(&act()).unwrap();
        let b = lower_to_torch(&act_like(8, 256, 8, 50, 20, 1)).unwrap();
        assert_ne!(a.lowering_hash, b.lowering_hash);
    }

    #[test]
    fn the_act_fixture_lowers_one_forward_line_per_node() {
        let g = act();
        let m = lower_to_torch(&g).unwrap();

        let body: Vec<&str> = m
            .source
            .split("def forward(self, **inputs):\n")
            .nth(1)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("return"))
            .collect();
        assert_eq!(body.len(), g.nodes.nodes.len(), "{}", m.source);

        for expected in [
            "_backbone(\"resnet18\", 512)",
            "nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 512))",
            "nn.Linear(1024, 512)",
            "nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8",
            "nn.Linear(512, 400)",
        ] {
            assert!(m.source.contains(expected), "missing `{expected}`");
        }
        // Boundary inputs by name, the chunker's slice, and the declared output name.
        assert!(m.source.contains("inputs[\"rgb_front\"]"));
        assert!(m.source.contains("inputs[\"joint_state\"]"));
        assert!(m.source.contains("[:20]"));
        assert!(m.source.contains("return {\"actions\": v5_actions}"));
        assert!(m.source.contains("import torchvision"));
    }

    #[test]
    fn the_act_fixture_declares_the_documented_keys() {
        let m = lower_to_torch(&act()).unwrap();
        assert_eq!(
            m.weight_keys,
            vec![
                "nodes.0.*",
                "nodes.1.0.weight",
                "nodes.1.0.bias",
                "nodes.1.2.weight",
                "nodes.1.2.bias",
                "nodes.2.weight",
                "nodes.2.bias",
                "nodes.3.*",
                "nodes.4.weight",
                "nodes.4.bias",
            ]
        );
        assert_eq!(m.weight_shapes["nodes.1.0.weight"], vec![256, 8]);
        assert_eq!(m.weight_shapes["nodes.2.weight"], vec![512, 1024]);
        assert_eq!(m.weight_shapes["nodes.4.weight"], vec![400, 512]);
        assert_eq!(m.weight_shapes["nodes.4.bias"], vec![400]);
        // Prefix claims carry no shape.
        assert!(!m.weight_shapes.contains_key("nodes.0.*"));
    }

    #[test]
    fn deferred_heads_are_unsupported_not_silently_wrong() {
        use es_ir::learning::ArchKind;
        let mut g = act();
        g.policy.architecture = ArchKind::Discrete;
        let LearningNode::PolicyHead { kind, .. } = g.nodes.nodes.get_mut(&NodeId(4)).unwrap()
        else {
            unreachable!()
        };
        *kind = HeadKind::Discrete { vocab: 256 };
        let err = lower_to_torch(&g).unwrap_err();
        assert!(
            matches!(&err, LowerError::Unsupported(k) if k.starts_with("PolicyHead{Discrete")),
            "{err}"
        );
    }

    // --- sampler heads (design note section 8) ----------------------------------------------

    use crate::reference::{diffusion_head, diffusion_params, sampler_graph, N_STEPS};

    fn sampler(kind: HeadKind) -> LearningGraph {
        sampler_graph(
            kind,
            es_ir::learning::WeightsRef::Safetensors {
                path: "w.safetensors".to_owned(),
                hash: [0u8; 32],
            },
        )
    }

    fn head(scheduler: DiffusionScheduler) -> HeadKind {
        diffusion_head(&diffusion_params(
            scheduler,
            BetaSchedule::SquaredcosCapV2,
            VarianceType::FixedSmall,
        ))
    }

    fn ddpm() -> LearningGraph {
        sampler(head(DiffusionScheduler::Ddpm))
    }

    fn flow() -> LearningGraph {
        sampler(HeadKind::FlowMatching { n_steps: N_STEPS })
    }

    #[test]
    fn sampler_lowering_is_byte_identical_across_runs() {
        for g in [ddpm(), flow()] {
            let a = lower_to_torch(&g).unwrap();
            let b = lower_to_torch(&g.clone()).unwrap();
            assert_eq!(a.source, b.source);
            assert_eq!(a.lowering_hash, b.lowering_hash);
            assert_eq!(a.weight_keys, b.weight_keys);
        }
        // The step count is part of the architecture, so it must move the hash.
        let other = sampler(HeadKind::FlowMatching { n_steps: 9 });
        assert_ne!(
            lower_to_torch(&flow()).unwrap().lowering_hash,
            lower_to_torch(&other).unwrap().lowering_hash
        );
        // So must the scheduler, which only changes the emitted coefficients.
        let ddim = sampler(head(DiffusionScheduler::Ddim));
        assert_ne!(
            lower_to_torch(&ddpm()).unwrap().lowering_hash,
            lower_to_torch(&ddim).unwrap().lowering_hash
        );
    }

    #[test]
    fn the_sampling_loop_is_in_the_source_with_the_declared_step_count() {
        let d = lower_to_torch(&ddpm()).unwrap();
        assert!(d.source.contains("class _DdpmHead"), "{}", d.source);
        assert!(d.source.contains("for i, t in enumerate(self.ts):"));
        assert!(d.source.contains("_sinusoidal(float(t), self.cond)"));
        assert!(d
            .source
            .contains("x0 = torch.clamp(x0, -self.clip, self.clip)"));
        // x_dim = H*A = 6, cond = 16, and the 8 timesteps subsampled out of 100 at stride 12.
        assert!(
            d.source
                .contains("_DdpmHead(6, 16, 3, 2, [84, 72, 60, 48, 36, 24, 12, 0], ["),
            "{}",
            d.source
        );
        assert!(!d.source.contains("class _FlowHead"));
        assert!(!d.source.contains("torchvision"));

        let f = lower_to_torch(&flow()).unwrap();
        assert!(f.source.contains("class _FlowHead"), "{}", f.source);
        assert!(f.source.contains("for i in range(self.n_steps):"));
        assert!(f.source.contains("x = x + dt * v"));
        assert!(f.source.contains("_FlowHead(6, 16, 3, 2, 8)"));
        assert!(!f.source.contains("class _DdpmHead"));
        // Both share one denoiser definition.
        assert!(d.source.contains("class _Denoiser") && f.source.contains("class _Denoiser"));
        assert!(d.source.contains("return {\"actions\": v1_chunk}"));
    }

    #[test]
    fn ddpm_declares_a_noise_buffer_per_stochastic_step_and_ddim_declares_none() {
        let d = lower_to_torch(&ddpm()).unwrap();
        let noise: Vec<&String> = d
            .weight_keys
            .iter()
            .filter(|k| k.contains(".noise_"))
            .collect();
        // t = 0 is the final, noise-free step, so 7 draws for 8 steps, keyed by the real
        // diffusers timestep rather than by the loop index.
        assert_eq!(noise.len(), N_STEPS as usize - 1, "{noise:?}");
        assert!(!noise.iter().any(|k| k.ends_with("noise_0")));
        assert_eq!(d.weight_shapes["nodes.1.noise_12"], vec![6]);

        // The denoiser's own keys: [x_dim + cond + cond] in, cond hidden, x_dim out.
        assert_eq!(d.weight_shapes["nodes.1.net.l0.weight"], vec![16, 38]);
        assert_eq!(d.weight_shapes["nodes.1.net.l0.bias"], vec![16]);
        assert_eq!(d.weight_shapes["nodes.1.net.l1.weight"], vec![6, 16]);
        assert_eq!(d.weight_shapes["nodes.1.net.l1.bias"], vec![6]);

        for g in [sampler(head(DiffusionScheduler::Ddim)), flow()] {
            let m = lower_to_torch(&g).unwrap();
            assert!(!m.weight_keys.iter().any(|k| k.contains(".noise_")));
            assert!(m.weight_keys.iter().all(|k| !k.ends_with(".*")));
        }
    }

    /// A checkpoint for one head must not load into the other, and a missing noise draw must
    /// not be silently replaced by zeros.
    #[test]
    fn the_key_set_binds_the_checkpoint_to_the_head() {
        use crate::weights::{parse_header, validate_keys, write_safetensors};
        let d = lower_to_torch(&ddpm()).unwrap();
        let f = lower_to_torch(&flow()).unwrap();
        let ck = crate::reference::checkpoint(&d.weight_shapes, 7);
        let header = parse_header(&write_safetensors(&ck)).unwrap();
        validate_keys(&d, &header).unwrap();

        let err = validate_keys(&f, &header).unwrap_err();
        let crate::PolicyError::WeightMismatch { unexpected, .. } = &err else {
            panic!("{err}")
        };
        assert_eq!(unexpected.len(), N_STEPS as usize - 1, "{unexpected:?}");

        let mut short = ck;
        short.remove("nodes.1.noise_36");
        let header = parse_header(&write_safetensors(&short)).unwrap();
        let err = validate_keys(&d, &header).unwrap_err();
        let crate::PolicyError::WeightMismatch { missing, .. } = &err else {
            panic!("{err}")
        };
        assert_eq!(missing, &["nodes.1.noise_36"]);
    }

    #[test]
    fn a_sampler_head_without_a_noise_input_does_not_lower() {
        let mut g = ddpm();
        let LearningNode::PolicyHead { inputs, .. } = g.nodes.nodes.get_mut(&NodeId(1)).unwrap()
        else {
            unreachable!()
        };
        inputs.pop();
        g.inputs.pop();
        g.nodes.inputs.pop();
        g.policy.contract.inputs.remove("noise");
        let err = lower_to_torch(&g).unwrap_err();
        assert!(
            matches!(&err, LowerError::Shape { message, .. } if message.contains("initial noise")),
            "{err}"
        );
    }

    #[test]
    fn dpm_solver_is_unsupported_rather_than_approximated() {
        let err = lower_to_torch(&sampler(head(DiffusionScheduler::DpmSolver))).unwrap_err();
        assert!(
            matches!(&err, LowerError::Unsupported(k) if k == "PolicyHead{Diffusion}{DpmSolver}"),
            "{err}"
        );
    }

    /// The schedule is the one the design note writes down: built over the *training* grid and
    /// subsampled to the inference steps, DDPM noisy except at `t = 0`, DDIM noise-free.
    /// `reference::tests::diffusion_schedule_matches_diffusers` is the independent oracle.
    #[test]
    fn the_schedule_matches_the_documented_form() {
        let params = diffusion_params(
            DiffusionScheduler::Ddpm,
            BetaSchedule::SquaredcosCapV2,
            VarianceType::FixedSmall,
        );
        let ddpm_sched = diffusion_schedule(&params);
        assert_eq!(ddpm_sched.alpha_bar.len(), 100);
        assert!(ddpm_sched.alpha_bar.windows(2).all(|w| w[1] < w[0]));
        assert_eq!(ddpm_sched.timesteps, vec![84, 72, 60, 48, 36, 24, 12, 0]);
        assert!(ddpm_sched.sigma[7].abs() < f32::EPSILON);
        assert!(ddpm_sched.sigma[..7].iter().all(|s| *s > 0.0));
        assert!(ddpm_sched.ce.iter().all(|c| *c == 0.0));

        let mut ddim = params;
        ddim.scheduler = DiffusionScheduler::Ddim;
        let ddim_sched = diffusion_schedule(&ddim);
        assert!(ddim_sched.sigma.iter().all(|s| *s == 0.0));
        assert!(ddim_sched.cx.iter().all(|c| *c == 0.0));
        // The last step lands on abar_prev = 1, i.e. the clean sample.
        assert!((ddim_sched.c0[7] - 1.0).abs() < 1e-6 && ddim_sched.ce[7].abs() < 1e-6);
        // fixed_large is beta_t itself, larger than the posterior variance beta~_t.
        let mut large = params;
        large.variance_type = VarianceType::FixedLarge;
        assert!(diffusion_schedule(&large).variance[0] > ddpm_sched.variance[0]);

        let mut one = params;
        one.n_steps = 1;
        assert_eq!(diffusion_schedule(&one).timesteps, vec![0]);
    }

    /// `_sinusoidal` returns `2 * (dim // 2)` values, so an odd conditioning width would make
    /// `_Denoiser.l0` mismatch at run time. It is a lowering error, not a torch traceback.
    #[test]
    fn an_odd_conditioning_width_is_rejected() {
        let mut g = ddpm();
        let LearningNode::StateEncoder { out_dim, .. } = g.nodes.nodes.get_mut(&NodeId(0)).unwrap()
        else {
            unreachable!()
        };
        *out_dim = 15;
        let LearningNode::PolicyHead { inputs, .. } = g.nodes.nodes.get_mut(&NodeId(1)).unwrap()
        else {
            unreachable!()
        };
        inputs[0].ty.shape = es_ir::types::Shape::new(vec![15]);
        let err = lower_to_torch(&g).unwrap_err();
        assert!(
            matches!(&err, LowerError::Shape { message, .. } if message.contains("must be even")),
            "{err}"
        );
    }

    /// `prediction_type` is part of the checkpoint's contract: the denoiser is `eps_theta`, so
    /// `sample` and `v_prediction` would be a silently wrong chunk.
    #[test]
    fn a_non_epsilon_prediction_type_is_unsupported() {
        let mut g = ddpm();
        let LearningNode::PolicyHead { kind, .. } = g.nodes.nodes.get_mut(&NodeId(1)).unwrap()
        else {
            unreachable!()
        };
        let HeadKind::Diffusion {
            prediction_type, ..
        } = kind
        else {
            unreachable!()
        };
        *prediction_type = PredictionType::VPrediction;
        let err = lower_to_torch(&g).unwrap_err();
        assert!(
            matches!(&err, LowerError::Unsupported(k) if k.contains("VPrediction")),
            "{err}"
        );
    }

    #[test]
    fn an_invalid_graph_never_lowers() {
        let mut g = act();
        g.policy.contract.execute_chunk = 80; // LRN-020: K > H
        assert!(matches!(
            lower_to_torch(&g).unwrap_err(),
            LowerError::Invalid(_)
        ));
    }
}
