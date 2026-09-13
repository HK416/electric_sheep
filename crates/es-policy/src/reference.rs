//! The test-only Rust side of the tier-4 comparison for the two sampler heads (spec 8.9).
//!
//! It re-implements what `lower::torch` generates — the same one-hidden-layer denoiser, the same
//! sinusoidal timestep embedding, the same DDPM/DDIM and Euler loops — in f32 with a fixed op
//! order and `es_math::approx` for the transcendentals (spec 3.4). Two things are *shared* with
//! the lowering rather than mirrored, on purpose:
//!
//! - the schedule coefficients ([`diffusion_schedule`]), which the lowering emits into the
//!   generated Python as literals. Recomputing them here would test float formatting;
//! - the per-step noise `z_t`, which lives in the checkpoint as non-trainable `noise_<t>`
//!   buffers. Both sides read the same draw, which is the only honest way to compare an
//!   ancestral sampler at all (spec 3.4 forbids a global RNG on this path).
//!
//! What is therefore measured is the sampling loop, the network, and the embedding — i.e. the
//! lowering. This module is `#[cfg(test)]`: it is an oracle, never a runtime.

use std::collections::BTreeMap;

use es_ir::graph::{Graph, NodeId, PortRef};
use es_ir::learning::{
    ActionExecutionMode, ArchKind, BetaSchedule, DiffusionScheduler, HeadKind, LearningGraph,
    LearningNode, PolicyContract, PolicyHandle, PredictionType, RuntimeHints, StateEncoderKind,
    TensorPort, VarianceType, WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};

use crate::lower::torch::{diffusion_schedule, DiffusionParams};
use crate::weights::Checkpoint;

/// The tiny tier-4 configuration: state 4, action 2, horizon 3, 8 steps, conditioning width 16.
pub const STATE_DIM: u64 = 4;
pub const COND: u64 = 16;
pub const ACTION_DIM: u64 = 2;
pub const HORIZON: u64 = 3;
pub const N_STEPS: u32 = 8;
/// `LeRobot`'s Diffusion Policy `num_train_timesteps`. 100 / 8 = a stride of 12, so the tier-4
/// configuration actually crosses the subsampling path rather than `n_steps == num_train`.
pub const TRAIN_STEPS: u32 = 100;

/// The `diffusers` / `LeRobot` defaults, with one knob per test.
pub fn diffusion_params(
    scheduler: DiffusionScheduler,
    beta_schedule: BetaSchedule,
    variance_type: VarianceType,
) -> DiffusionParams {
    DiffusionParams {
        n_steps: N_STEPS,
        scheduler,
        num_train_timesteps: TRAIN_STEPS,
        beta_schedule,
        variance_type,
        clip_sample: true,
        clip_sample_range: 1.0,
    }
}

pub fn diffusion_head(p: &DiffusionParams) -> HeadKind {
    HeadKind::Diffusion {
        n_steps: p.n_steps,
        scheduler: p.scheduler,
        num_train_timesteps: p.num_train_timesteps,
        beta_schedule: p.beta_schedule,
        variance_type: p.variance_type,
        prediction_type: PredictionType::Epsilon,
        clip_sample: p.clip_sample,
        clip_sample_range: p.clip_sample_range,
    }
}

/// The independent `diffusers` oracle (spec 1.4), embedded so editing it forces a rebuild.
pub const DDPM_CHECK: &str = include_str!("../python/ddpm_ref_check.py");

/// Runs [`DDPM_CHECK`] with `request` on stdin. `Err` is a SKIP reason, never a failure: no
/// interpreter, no `diffusers`, no oracle.
pub fn diffusers_check(request: &serde_json::Value) -> Result<serde_json::Value, String> {
    use std::io::{Read as _, Write as _};
    use std::process::{Command, Stdio};

    let mut tried = Vec::new();
    for python in crate::torch_runtime::python_candidates() {
        let child = Command::new(&python)
            .args(["-c", DDPM_CHECK])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            Err(e) => {
                tried.push(format!("`{python}`: {e}"));
                continue;
            }
        };
        let body = serde_json::to_vec(request).expect("the request is serializable");
        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(&body)
            .map_err(|e| e.to_string())?;
        let mut out = String::new();
        child
            .stdout
            .take()
            .expect("stdout was piped")
            .read_to_string(&mut out)
            .map_err(|e| e.to_string())?;
        let _ = child.wait();
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).map_err(|e| format!("`{python}`: {e} in `{out}`"))?;
        if value["ok"] == serde_json::Value::Bool(true) {
            return Ok(value);
        }
        tried.push(format!("`{python}`: {}", value["error"]));
    }
    Err(format!(
        "no Python with `diffusers` (set ES_PYTHON): {}",
        tried.join("; ")
    ))
}

/// The request body both commands share.
fn check_config(p: &DiffusionParams) -> serde_json::Value {
    serde_json::json!({
        "num_train_timesteps": p.num_train_timesteps,
        "beta_schedule": match p.beta_schedule {
            BetaSchedule::Linear => "linear",
            BetaSchedule::SquaredcosCapV2 => "squaredcos_cap_v2",
        },
        "variance_type": match p.variance_type {
            VarianceType::FixedSmall => "fixed_small",
            VarianceType::FixedLarge => "fixed_large",
        },
        "clip_sample": p.clip_sample,
        "clip_sample_range": p.clip_sample_range,
        "n_steps": p.n_steps,
        "scheduler": match p.scheduler {
            DiffusionScheduler::Ddpm => "ddpm",
            _ => "ddim",
        },
    })
}

// --- the graph ------------------------------------------------------------------------------

fn port(name: &str, shape: &[u64], unit: Unit) -> TensorPort {
    TensorPort::new(
        name,
        PortType {
            elem: ElemType::F32,
            shape: Shape::new(shape.to_vec()),
            unit,
            frame: Frame::Policy,
            time: TimeRef::Tick,
            image: None,
        },
    )
}

fn normalized() -> Unit {
    Unit::Normalized { lo: -1.0, hi: 1.0 }
}

/// `joint_state[4]` -> `Linear(4, 16)` -> sampler head -> `[3, 2]`, with `x_T` entering as the
/// declared graph input `noise`.
pub fn sampler_graph(kind: HeadKind, weights: WeightsRef) -> LearningGraph {
    let state = port("joint_state", &[STATE_DIM], normalized());
    let noise = port("noise", &[HORIZON, ACTION_DIM], Unit::Dimensionless);
    let feat = port("feat", &[COND], Unit::Dimensionless);

    let mut g = Graph::new(1);
    g.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp { hidden: vec![] },
            out_dim: COND as u32,
        },
    );
    g.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![feat, noise.clone()],
            kind,
            action_dim: ACTION_DIM as u32,
            horizon: HORIZON as u32,
        },
    );
    g.connect(NodeId(0), "out", NodeId(1), "feat");
    g.inputs = vec![
        PortRef::new(NodeId(0), "joint_state"),
        PortRef::new(NodeId(1), "noise"),
    ];
    g.outputs = vec![PortRef::new(NodeId(1), "chunk")];

    LearningGraph {
        schema_version: 1,
        inputs: vec![state.clone(), noise.clone()],
        outputs: vec![port("actions", &[HORIZON, ACTION_DIM], normalized())],
        nodes: g,
        policy: PolicyHandle {
            architecture: match kind {
                HeadKind::FlowMatching { .. } => ArchKind::FlowMatching,
                _ => ArchKind::Diffusion,
            },
            base_model: None,
            weights,
            contract: PolicyContract {
                inputs: [(state.name.clone(), state), (noise.name.clone(), noise)]
                    .into_iter()
                    .collect(),
                observation_window: 1,
                action_dim: ACTION_DIM as u32,
                horizon: HORIZON as u32,
                execute_chunk: HORIZON as u32,
                replanning_hz: 10.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 1.0,
                    deadline_ms: 50.0,
                },
            },
        },
    }
}

/// splitmix64, so every weight and every `noise_<t>` draw is a function of the seed alone
/// (spec 3.4: no global RNG, not even in a fixture).
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // [-0.5, 0.5) in exactly representable steps.
        (z >> 40) as f32 / 16_777_216.0 - 0.5
    }
}

/// A checkpoint for every exact key the module declares, including the `noise_<t>` buffers.
pub fn checkpoint(shapes: &BTreeMap<String, Vec<u64>>, seed: u64) -> Checkpoint {
    let mut rng = SplitMix(seed);
    shapes
        .iter()
        .map(|(k, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            let values = (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>();
            (k.clone(), (shape.clone(), values))
        })
        .collect()
}

// --- the network ------------------------------------------------------------------------------

/// `y = W x + b`, `W` row-major `[out, in]`, outputs ascending and inputs ascending within a
/// row, accumulated in f32. No reassociation: a reference is readable, not fast (spec 3.4).
fn affine(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let n_in = x.len();
    assert_eq!(w.len(), b.len() * n_in, "weight shape");
    let mut out = Vec::with_capacity(b.len());
    for (row, bias) in b.iter().enumerate() {
        let mut acc = *bias;
        for (i, xi) in x.iter().enumerate() {
            acc += w[row * n_in + i] * xi;
        }
        out.push(acc);
    }
    out
}

/// `_sinusoidal` of the generated file: `[sin(t w_i), cos(t w_i)]`, `w_i = exp(-ln(1e4) i/half)`.
/// The op order is torch's: scale the index, divide, exponentiate, then scale by `t`.
fn sinusoidal(t: f32, dim: usize) -> Vec<f32> {
    const NEG_LN_1E4: f32 = -9.210_340_4;
    let half = dim / 2;
    let w: Vec<f32> = (0..half)
        .map(|i| es_math::approx::exp(NEG_LN_1E4 * i as f32 / half as f32))
        .collect();
    let mut out: Vec<f32> = w.iter().map(|wi| es_math::approx::sin(t * wi)).collect();
    out.extend(w.iter().map(|wi| es_math::approx::cos(t * wi)));
    out
}

/// `_Denoiser.forward`: `l1(relu(l0([x, cond, temb])))`.
fn denoise(ck: &Checkpoint, x: &[f32], cond: &[f32], temb: &[f32]) -> Vec<f32> {
    let w = |k: &str| ck[k].1.as_slice();
    let mut cat = x.to_vec();
    cat.extend_from_slice(cond);
    cat.extend_from_slice(temb);
    let mut h = affine(w("nodes.1.net.l0.weight"), w("nodes.1.net.l0.bias"), &cat);
    for v in &mut h {
        *v = v.max(0.0);
    }
    affine(w("nodes.1.net.l1.weight"), w("nodes.1.net.l1.bias"), &h)
}

/// `_DdpmHead.forward`: the reverse loop over the subsampled inference timesteps.
pub fn ddpm(ck: &Checkpoint, cond: &[f32], noise: &[f32], p: &DiffusionParams) -> Vec<f32> {
    let s = diffusion_schedule(p);
    let mut x = noise.to_vec();
    for (i, t) in s.timesteps.iter().enumerate() {
        let temb = sinusoidal(*t as f32, cond.len());
        let eps = denoise(ck, &x, cond, &temb);
        for (xi, e) in x.iter_mut().zip(&eps) {
            let mut x0 = (*xi - s.sqrt_1mab[i] * *e) / s.sqrt_ab[i];
            if p.clip_sample {
                x0 = x0.clamp(-p.clip_sample_range, p.clip_sample_range);
            }
            *xi = s.c0[i] * x0 + s.cx[i] * *xi + s.ce[i] * *e;
        }
        if s.sigma[i] != 0.0 {
            let z = ck[&format!("nodes.1.noise_{t}")].1.as_slice();
            for (xi, zi) in x.iter_mut().zip(z) {
                *xi += s.sigma[i] * *zi;
            }
        }
    }
    x
}

/// `_FlowHead.forward`: Euler integration from `t = 0` to `t = 1`.
pub fn flow(ck: &Checkpoint, cond: &[f32], noise: &[f32], n_steps: u32) -> Vec<f32> {
    let n = f64::from(n_steps);
    let dt = (1.0 / n) as f32;
    let mut x = noise.to_vec();
    for i in 0..n_steps {
        let temb = sinusoidal((f64::from(i) / n) as f32, cond.len());
        let v = denoise(ck, &x, cond, &temb);
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * *vi;
        }
    }
    x
}

// --- the tier-4 comparison --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equiv::{action_chunk, compare_actions, Tolerance};
    use crate::lower::lower_to_torch;
    use crate::runtime::{PolicyRuntime, WeightsSource};
    use crate::weights::write_safetensors;
    use crate::{PolicyError, TorchRuntime};
    use es_compile::Tensor;
    use es_ir::Diagnostic;

    fn tensor(shape: &[u64], values: &[f32]) -> Tensor {
        Tensor {
            dtype: ElemType::F32,
            shape: shape.to_vec(),
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn both_sampler_graphs_are_valid_learning_ir() {
        for kind in [
            diffusion_head(&diffusion_params(
                DiffusionScheduler::Ddpm,
                BetaSchedule::SquaredcosCapV2,
                VarianceType::FixedSmall,
            )),
            HeadKind::FlowMatching { n_steps: N_STEPS },
        ] {
            let g = sampler_graph(
                kind,
                WeightsRef::Safetensors {
                    path: "w.safetensors".to_owned(),
                    hash: [0u8; 32],
                },
            );
            let errors: Vec<_> = g
                .validate()
                .into_iter()
                .filter(Diagnostic::is_error)
                .collect();
            assert!(errors.is_empty(), "{kind:?}: {errors:#?}");
        }
    }

    /// The whole point of the packet: torch's sampling loop and the Rust one agree to spec 8.9
    /// tier 4. SKIPs loudly without `torch` — a missing wheel is not a passing equivalence.
    fn tier4(kind: HeadKind, name: &str) {
        if let Err(why) = crate::torch_runtime::is_available() {
            println!("SKIPPED {name}: {why}");
            return;
        }

        let probe = lower_to_torch(&sampler_graph(
            kind,
            WeightsRef::Safetensors {
                path: "w.safetensors".to_owned(),
                hash: [0u8; 32],
            },
        ))
        .unwrap();
        let ck = checkpoint(&probe.weight_shapes, 0x5eed_1234_abcd_0001);
        let bytes = write_safetensors(&ck);
        let path = std::env::temp_dir().join(format!("es-policy-{name}.safetensors"));
        std::fs::write(&path, &bytes).unwrap();

        let graph = sampler_graph(
            kind,
            WeightsRef::Safetensors {
                path: path.to_string_lossy().into_owned(),
                hash: *blake3::hash(&bytes).as_bytes(),
            },
        );
        let mut rt = TorchRuntime::new();
        let info = rt
            .load(&graph, &WeightsSource::Safetensors(path.clone()))
            .expect("torch is available, so the load must succeed");

        let mut rng = SplitMix(0x0bad_f00d_0000_0007);
        let state: Vec<f32> = (0..STATE_DIM).map(|_| rng.next_f32()).collect();
        let x_t: Vec<f32> = (0..HORIZON * ACTION_DIM).map(|_| rng.next_f32()).collect();
        let inputs: BTreeMap<String, Tensor> = [
            ("joint_state".to_owned(), tensor(&[STATE_DIM], &state)),
            ("noise".to_owned(), tensor(&[HORIZON, ACTION_DIM], &x_t)),
        ]
        .into();
        let got = rt.infer(&inputs).expect("forward pass")["actions"].clone();
        assert_eq!(got.shape, vec![HORIZON, ACTION_DIM]);

        let w = |k: &str| ck[k].1.as_slice();
        let cond = affine(w("nodes.0.0.weight"), w("nodes.0.0.bias"), &state);
        let want = match kind {
            HeadKind::Diffusion { .. } => ddpm(
                &ck,
                &cond,
                &x_t,
                &DiffusionParams::from_head(&kind).expect("a Diffusion head"),
            ),
            HeadKind::FlowMatching { n_steps } => flow(&ck, &cond, &x_t, n_steps),
            other => panic!("{other:?} is not a sampler head"),
        };
        let want = action_chunk(HORIZON, ACTION_DIM, &want);

        let e = compare_actions(&got, &want, Tolerance::TIER4_FP32);
        println!(
            "RAN {name} (torch {}): max_abs = {:.3e}, max_rel = {:.3e}",
            info.version, e.max_abs, e.max_rel
        );
        assert!(e.pass, "spec 8.9 tier 4 (abs <= 1e-5): {e:?}");

        // Spec 8.9 last row: the same runtime re-run is bitwise identical. An iterative sampler
        // is where a hidden RNG would show up, so this row matters more here than it does for
        // the regression head.
        let again = rt.infer(&inputs).unwrap();
        assert!(compare_actions(&again["actions"], &got, Tolerance::BITWISE).pass);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torch_ddpm_matches_rust() {
        tier4(
            diffusion_head(&diffusion_params(
                DiffusionScheduler::Ddpm,
                BetaSchedule::SquaredcosCapV2,
                VarianceType::FixedSmall,
            )),
            "torch_ddpm_matches_rust",
        );
    }

    #[test]
    fn torch_ddim_matches_rust() {
        tier4(
            diffusion_head(&diffusion_params(
                DiffusionScheduler::Ddim,
                BetaSchedule::SquaredcosCapV2,
                VarianceType::FixedSmall,
            )),
            "torch_ddim_matches_rust",
        );
    }

    #[test]
    fn torch_flow_matching_matches_rust() {
        tier4(
            HeadKind::FlowMatching { n_steps: N_STEPS },
            "torch_flow_matching_matches_rust",
        );
    }

    /// The largest absolute gap between two `f32` lists, `INFINITY` if the lengths differ.
    fn max_abs(got: &[f32], want: &serde_json::Value) -> f64 {
        let want: Vec<f64> = want
            .as_array()
            .expect("a JSON array")
            .iter()
            .map(|v| v.as_f64().expect("a JSON number"))
            .collect();
        if got.len() != want.len() {
            return f64::INFINITY;
        }
        got.iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0.0, f64::max)
    }

    /// P-M2-R6: the schedule against `diffusers` itself, not against our own mirror of it.
    /// `alphas_cumprod` over the training grid, `set_timesteps`' subsample, and `_get_variance`.
    #[test]
    fn diffusion_schedule_matches_diffusers() {
        let mut ran = 0;
        for scheduler in [DiffusionScheduler::Ddpm, DiffusionScheduler::Ddim] {
            for beta in [BetaSchedule::SquaredcosCapV2, BetaSchedule::Linear] {
                for variance in [VarianceType::FixedSmall, VarianceType::FixedLarge] {
                    let params = diffusion_params(scheduler, beta, variance);
                    let mut request = check_config(&params);
                    request["cmd"] = "schedule".into();
                    let reply = match diffusers_check(&request) {
                        Ok(value) => value,
                        Err(why) => {
                            println!("SKIPPED diffusion_schedule_matches_diffusers: {why}");
                            return;
                        }
                    };
                    let sched = diffusion_schedule(&params);
                    let cumprod = max_abs(&sched.alpha_bar, &reply["alphas_cumprod"]);
                    let var = max_abs(&sched.variance, &reply["variances"]);
                    let label = format!("{scheduler:?}/{beta:?}/{variance:?}");
                    print!("RAN diffusion_schedule_matches_diffusers {label}: ");
                    println!(
                        "alphas_cumprod max_abs = {cumprod:.3e}, variance max_abs = {var:.3e}"
                    );
                    // The subsample is a list of integers: exactly equal, not within 1e-5.
                    let want: Vec<u32> = reply["timesteps"]
                        .as_array()
                        .expect("a JSON array")
                        .iter()
                        .map(|v| v.as_u64().expect("a JSON integer") as u32)
                        .collect();
                    assert_eq!(sched.timesteps, want, "set_timesteps disagrees");
                    let tol = Tolerance::TIER4_FP32.abs;
                    assert!(cumprod <= tol, "alphas_cumprod {cumprod:e} > {tol:e}");
                    assert!(var <= tol, "variance {var:e} > {tol:e}");
                    ran += 1;
                }
            }
        }
        assert_eq!(ran, 8);
    }

    /// The other half of P-M2-R6: the whole lowered head through torch against a *direct*
    /// `scheduler.step` loop, so the reference no longer mirrors the lowering line for line.
    fn diffusers_loop(scheduler: DiffusionScheduler, name: &str) {
        if let Err(why) = crate::torch_runtime::is_available() {
            println!("SKIPPED {name}: {why}");
            return;
        }
        let p = diffusion_params(
            scheduler,
            BetaSchedule::SquaredcosCapV2,
            VarianceType::FixedSmall,
        );
        let kind = diffusion_head(&p);
        let weights = WeightsRef::Safetensors {
            path: "w.safetensors".to_owned(),
            hash: [0u8; 32],
        };
        let probe = lower_to_torch(&sampler_graph(kind, weights)).unwrap();
        let ck = checkpoint(&probe.weight_shapes, 0x5eed_1234_abcd_0002);
        let bytes = write_safetensors(&ck);
        let path = std::env::temp_dir().join(format!("es-policy-{name}.safetensors"));
        std::fs::write(&path, &bytes).unwrap();

        let mut rng = SplitMix(0x0bad_f00d_0000_0011);
        let state: Vec<f32> = (0..STATE_DIM).map(|_| rng.next_f32()).collect();
        let x_t: Vec<f32> = (0..HORIZON * ACTION_DIM).map(|_| rng.next_f32()).collect();
        let w = |k: &str| ck[k].1.clone();
        let cond = affine(&w("nodes.0.0.weight"), &w("nodes.0.0.bias"), &state);

        let mut request = check_config(&p);
        request["cmd"] = "loop".into();
        request["cond"] = cond.clone().into();
        request["x_t"] = x_t.clone().into();
        request["weights"] = serde_json::json!({
            "l0.weight": {"shape": probe.weight_shapes["nodes.1.net.l0.weight"],
                          "data": w("nodes.1.net.l0.weight")},
            "l0.bias": {"shape": probe.weight_shapes["nodes.1.net.l0.bias"],
                        "data": w("nodes.1.net.l0.bias")},
            "l1.weight": {"shape": probe.weight_shapes["nodes.1.net.l1.weight"],
                          "data": w("nodes.1.net.l1.weight")},
            "l1.bias": {"shape": probe.weight_shapes["nodes.1.net.l1.bias"],
                        "data": w("nodes.1.net.l1.bias")},
        });
        let noise: serde_json::Map<String, serde_json::Value> = ck
            .keys()
            .filter_map(|k| {
                k.strip_prefix("nodes.1.noise_")
                    .map(|t| (t.to_owned(), w(k).into()))
            })
            .collect();
        request["noise"] = noise.into();
        let reply = match diffusers_check(&request) {
            Ok(v) => v,
            Err(why) => {
                println!("SKIPPED {name}: {why}");
                let _ = std::fs::remove_file(&path);
                return;
            }
        };

        let graph = sampler_graph(
            kind,
            WeightsRef::Safetensors {
                path: path.to_string_lossy().into_owned(),
                hash: *blake3::hash(&bytes).as_bytes(),
            },
        );
        let mut rt = TorchRuntime::new();
        rt.load(&graph, &WeightsSource::Safetensors(path.clone()))
            .expect("torch is available, so the load must succeed");
        let inputs: BTreeMap<String, Tensor> = [
            ("joint_state".to_owned(), tensor(&[STATE_DIM], &state)),
            ("noise".to_owned(), tensor(&[HORIZON, ACTION_DIM], &x_t)),
        ]
        .into();
        let got = rt.infer(&inputs).expect("forward pass")["actions"].clone();

        let want: Vec<f32> = reply["x"]
            .as_array()
            .expect("an array")
            .iter()
            .map(|v| v.as_f64().expect("a number") as f32)
            .collect();
        let e = compare_actions(
            &got,
            &action_chunk(HORIZON, ACTION_DIM, &want),
            Tolerance::TIER4_FP32,
        );
        println!(
            "RAN {name}: max_abs = {:.3e}, max_rel = {:.3e}",
            e.max_abs, e.max_rel
        );
        assert!(e.pass, "spec 8.9 tier 4 against diffusers: {e:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn torch_ddpm_matches_diffusers_step_loop() {
        diffusers_loop(
            DiffusionScheduler::Ddpm,
            "torch_ddpm_matches_diffusers_step_loop",
        );
    }

    #[test]
    fn torch_ddim_matches_diffusers_step_loop() {
        diffusers_loop(
            DiffusionScheduler::Ddim,
            "torch_ddim_matches_diffusers_step_loop",
        );
    }

    /// Runs without `torch`: the checkpoint must be the one the IR names, and it is checked
    /// before the backend starts (spec 5.3).
    #[test]
    fn a_checkpoint_the_ir_does_not_name_is_rejected() {
        let kind = HeadKind::FlowMatching { n_steps: N_STEPS };
        let module = lower_to_torch(&sampler_graph(
            kind,
            WeightsRef::Safetensors {
                path: "w.safetensors".to_owned(),
                hash: [0u8; 32],
            },
        ))
        .unwrap();
        let bytes = write_safetensors(&checkpoint(&module.weight_shapes, 1));
        let err = TorchRuntime::new()
            .load(
                &sampler_graph(
                    kind,
                    WeightsRef::Safetensors {
                        path: "w.safetensors".to_owned(),
                        hash: [0u8; 32],
                    },
                ),
                &WeightsSource::InMemory(bytes),
            )
            .unwrap_err();
        assert!(matches!(err, PolicyError::WeightsHash { .. }), "{err}");
    }
}
