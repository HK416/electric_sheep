//! Tier-4 policy equivalence against the reference oracle (spec 8.9, spec 1.4).
//!
//! A tiny Learning IR — `StateEncoder` MLP 4 -> 8 -> 8, `RegressionHead` to 2 dims over a
//! horizon of 3 — is lowered to `PyTorch`, loaded with a fixed-seed synthetic checkpoint, and
//! compared against a straightforward f32 evaluation of the same MLP in Rust.
//!
//! **This test SKIPs, loudly, when `torch` is not installed.** Spec 1.4 wants the harness to
//! exist whether or not the oracle is present on a given machine; a missing wheel must not be
//! reported as a passing equivalence. Point `ES_PYTHON` at an interpreter with `torch` to run
//! it for real.

use std::collections::BTreeMap;

use es_compile::Tensor;
use es_ir::graph::{Graph, NodeId, PortRef};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, HeadKind, LearningGraph,
    LearningNode, PolicyContract, PolicyHandle, RuntimeHints, Squash, StateEncoderKind, TensorPort,
    WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_ir::Diagnostic;
use es_policy::runtime::PolicyRuntime;
use es_policy::weights::{write_safetensors, Checkpoint};
use es_policy::{compare_actions, lower_to_torch, PolicyError, Tolerance, TorchRuntime};

const STATE_DIM: u64 = 4;
const FEAT: u64 = 8;
const ACTION_DIM: u64 = 2;
const HORIZON: u64 = 3;

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

/// `joint_state[4]` -> MLP(4, 8, 8) -> `Linear(8, 6)` -> `[3, 2]` -> chunker.
fn mlp_graph(weights: WeightsRef) -> LearningGraph {
    let state = port("joint_state", &[STATE_DIM], normalized());
    let feat = port("feat", &[FEAT], Unit::Dimensionless);
    let chunk = port("chunk", &[HORIZON, ACTION_DIM], normalized());

    let mut g = Graph::new(1);
    g.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![FEAT as u32],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: FEAT as u32,
        },
    );
    g.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![feat],
            kind: HeadKind::Regression,
            action_dim: ACTION_DIM as u32,
            horizon: HORIZON as u32,
            squash: Squash::None,
        },
    );
    g.insert(
        NodeId(2),
        LearningNode::ActionChunker {
            inputs: vec![chunk],
            horizon: HORIZON as u32,
            execute_chunk: HORIZON as u32,
            replan_hz: 10.0,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    g.connect(NodeId(0), "out", NodeId(1), "feat");
    g.connect(NodeId(1), "chunk", NodeId(2), "chunk");
    g.inputs = vec![PortRef::new(NodeId(0), "joint_state")];
    g.outputs = vec![PortRef::new(NodeId(2), "actions")];

    LearningGraph {
        schema_version: 1,
        inputs: vec![state.clone()],
        outputs: vec![port("actions", &[HORIZON, ACTION_DIM], normalized())],
        nodes: g,
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights,
            contract: PolicyContract {
                inputs: [(state.name.clone(), state)].into_iter().collect(),
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

// --- the Rust side of the comparison --------------------------------------------------------

mod reference {
    /// `y = W x + b` with `W` row-major `[out, in]`.
    ///
    /// Fixed op order: outputs ascending, and within a row the inputs ascending, accumulated
    /// in f32. No SIMD, no blocking, no reassociation — the point of a reference is that its
    /// op order is readable, not that it is fast (spec 3.4).
    pub fn affine(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
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

    /// `ReLU` between layers, none after the last. Chosen over `tanh`/`GELU` on purpose: it is
    /// exact in both f32 implementations, so the comparison measures the lowering rather than
    /// the gap between `es_math::approx` and libm (spec 3.2).
    pub fn eval_mlp(layers: &[(&[f32], &[f32])], x: &[f32]) -> Vec<f32> {
        let mut v = x.to_vec();
        for (i, (w, b)) in layers.iter().enumerate() {
            v = affine(w, b, &v);
            if i + 1 < layers.len() {
                for y in &mut v {
                    *y = y.max(0.0);
                }
            }
        }
        v
    }
}

/// xorshift64*, so the "random" weights are a function of the seed and nothing else.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // [-0.5, 0.5), exactly representable steps.
        (self.0 >> 40) as f32 / 16_777_216.0 - 0.5
    }

    fn tensor(&mut self, shape: &[u64]) -> (Vec<u64>, Vec<f32>) {
        let n = shape.iter().product::<u64>() as usize;
        (
            shape.to_vec(),
            (0..n).map(|_| self.next_f32()).collect::<Vec<f32>>(),
        )
    }
}

fn f32s(t: &Tensor) -> Vec<f32> {
    t.data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// --- tests ----------------------------------------------------------------------------------

#[test]
fn the_mlp_graph_is_a_valid_learning_ir() {
    let g = mlp_graph(WeightsRef::Safetensors {
        path: "w.safetensors".to_owned(),
        hash: [0u8; 32],
    });
    let errors: Vec<_> = g
        .validate()
        .into_iter()
        .filter(Diagnostic::is_error)
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
}

/// A state-only graph pulls in no torchvision and declares only exact keys — the case the
/// equivalence test below runs, kept honest without needing `torch`.
#[test]
fn a_state_only_graph_lowers_to_torch_alone() {
    let m = lower_to_torch(&mlp_graph(WeightsRef::Safetensors {
        path: "w.safetensors".to_owned(),
        hash: [0u8; 32],
    }))
    .unwrap();
    assert!(!m.source.contains("torchvision"), "{}", m.source);
    assert!(m.weight_keys.iter().all(|k| !k.ends_with(".*")));
    assert_eq!(
        m.weight_keys,
        vec![
            "nodes.0.0.weight",
            "nodes.0.0.bias",
            "nodes.0.2.weight",
            "nodes.0.2.bias",
            "nodes.1.weight",
            "nodes.1.bias",
        ]
    );
}

/// The hash chain (spec 5.3) is checked before Python is ever started, so this runs everywhere.
#[test]
fn a_checkpoint_the_ir_does_not_name_is_rejected() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let bytes = write_safetensors(&checkpoint(&mut rng));
    let graph = mlp_graph(WeightsRef::Safetensors {
        path: "w.safetensors".to_owned(),
        hash: [0u8; 32], // not the hash of `bytes`
    });
    let err = TorchRuntime::new()
        .load(&graph, &es_policy::WeightsSource::InMemory(bytes))
        .unwrap_err();
    assert!(
        matches!(err, PolicyError::WeightsHash { .. }),
        "{err}: the weights hash must be checked before the backend is even started"
    );
}

fn checkpoint(rng: &mut Rng) -> Checkpoint {
    [
        ("nodes.0.0.weight", rng.tensor(&[FEAT, STATE_DIM])),
        ("nodes.0.0.bias", rng.tensor(&[FEAT])),
        ("nodes.0.2.weight", rng.tensor(&[FEAT, FEAT])),
        ("nodes.0.2.bias", rng.tensor(&[FEAT])),
        ("nodes.1.weight", rng.tensor(&[HORIZON * ACTION_DIM, FEAT])),
        ("nodes.1.bias", rng.tensor(&[HORIZON * ACTION_DIM])),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

#[test]
fn torch_mlp_matches_rust() {
    if let Err(why) = es_policy::torch_runtime::is_available() {
        println!("SKIPPED torch_mlp_matches_rust: {why}");
        return;
    }

    let mut rng = Rng(0x1234_5678_9abc_def1);
    let tensors = checkpoint(&mut rng);
    let bytes = write_safetensors(&tensors);
    let path = std::env::temp_dir().join("es-policy-equivalence.safetensors");
    std::fs::write(&path, &bytes).unwrap();

    let graph = mlp_graph(WeightsRef::Safetensors {
        path: path.to_string_lossy().into_owned(),
        hash: *blake3::hash(&bytes).as_bytes(),
    });

    let mut rt = TorchRuntime::new();
    let info = rt
        .load(&graph, &es_policy::WeightsSource::Safetensors(path.clone()))
        .expect("torch is available, so the load must succeed");
    println!("torch {} loaded the lowered graph", info.version);
    assert_eq!((info.horizon, info.action_dim), (3, 2));
    assert_ne!(rt.runtime_hash(), [0u8; 32]);

    let x: Vec<f32> = (0..STATE_DIM).map(|_| rng.next_f32()).collect();
    let inputs: BTreeMap<String, Tensor> = [(
        "joint_state".to_owned(),
        Tensor {
            dtype: ElemType::F32,
            shape: vec![STATE_DIM],
            data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
    )]
    .into();
    let outputs = rt.infer(&inputs).expect("forward pass");
    let got = &outputs["actions"];
    assert_eq!(got.shape, vec![HORIZON, ACTION_DIM]);

    let w = |k: &str| tensors[k].1.as_slice();
    let feat = reference::eval_mlp(
        &[
            (w("nodes.0.0.weight"), w("nodes.0.0.bias")),
            (w("nodes.0.2.weight"), w("nodes.0.2.bias")),
        ],
        &x,
    );
    let want = reference::affine(w("nodes.1.weight"), w("nodes.1.bias"), &feat);
    let want = es_policy::equiv::action_chunk(HORIZON, ACTION_DIM, &want);

    let e = compare_actions(got, &want, Tolerance::TIER4_FP32);
    println!(
        "RAN torch_mlp_matches_rust: max_abs = {:.3e}, max_rel = {:.3e}, torch = {:?}, rust = {:?}",
        e.max_abs,
        e.max_rel,
        f32s(got),
        f32s(&want)
    );
    assert!(e.pass, "spec 8.9 tier 4 (abs <= 1e-5): {e:?}");

    // Spec 8.9 last row: the same runtime re-run is bitwise identical.
    let again = rt.infer(&inputs).unwrap();
    assert!(compare_actions(&again["actions"], got, Tolerance::BITWISE).pass);

    let _ = std::fs::remove_file(&path);
}
