//! Packet M8/S2a oracles 3 and 4: the `Mlp`'s activation and the `Regression` head's squash
//! reach `PyTorch`, and the default graph still lowers to the byte the demo was measured under.
//!
//! Oracle 3 is pure Rust and runs everywhere: the committed `learning.toml` must lower to the
//! same source it did before the three parameters existed (`lowering_hash` is the `compiler`
//! slot of `execution_hash`, spec 5.3), and every combination must spell the `torch.nn` module
//! the design note's table names.
//!
//! Oracle 4 is `#[ignore]`d and needs `ES_PYTHON`: it compares the lowered module against
//! `python/mlp_activation_ref.py` — the same network written out by hand — **bitwise** on CPU,
//! over 64 observations both sides build from the same integer formula.
//!
//! ```text
//! ES_PYTHON=<venv>/python cargo test -p es-policy lower_mlp_activations_match_torch \
//!     -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use es_compile::Tensor;
use es_ir::graph::{Graph, NodeId, PortRef};
use es_ir::learning::{
    ActionExecutionMode, Activation, ArchKind, HeadKind, LearningGraph, LearningNode,
    PolicyContract, PolicyHandle, RuntimeHints, Squash, StateEncoderKind, TensorPort, WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_policy::runtime::PolicyRuntime;
use es_policy::weights::{write_safetensors, Checkpoint};
use es_policy::{lower_to_torch, TorchRuntime};

/// `lowering_hash` of `tests/fixtures/visible-learning/learning.toml`, as packet M7/T3 left it
/// and as `docs/design/learning-lowering.md` section 5.3's table records it. Every number in
/// `visible-learning.md` section 7 was measured under this source.
const COMMITTED_LOWERING_HASH: &str =
    "3d06811c52b6887f021440d4ce9a1a061e4eb27a0abb97c79822acd4d8a2d394";

/// The oracle-4 graph: `obs[15]` -> `Mlp{hidden = [32], out_dim = 32}` -> `Regression(6, 1)`.
const DIMS: [u64; 3] = [15, 32, 32];
const ACTION_DIM: u64 = 6;
const HORIZON: u64 = 1;
const SAMPLES: usize = 64;

const REF: &str = include_str!("../python/mlp_activation_ref.py");

fn hex(digest: &[u8; 32]) -> String {
    digest.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

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

fn graph(activation: Activation, activate_output: bool, squash: Squash) -> LearningGraph {
    let obs = port("obs", &[DIMS[0]], normalized());
    let feat = port("feat", &[DIMS[2]], Unit::Dimensionless);
    let actions = port("actions", &[HORIZON, ACTION_DIM], normalized());

    let mut g = Graph::new(1);
    g.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![obs.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![DIMS[1] as u32],
                activation,
                activate_output,
            },
            out_dim: DIMS[2] as u32,
        },
    );
    g.insert(
        NodeId(1),
        LearningNode::PolicyHead {
            inputs: vec![feat],
            kind: HeadKind::Regression,
            action_dim: ACTION_DIM as u32,
            horizon: HORIZON as u32,
            squash,
        },
    );
    g.connect(NodeId(0), "out", NodeId(1), "feat");
    g.inputs = vec![PortRef::new(NodeId(0), "obs")];
    g.outputs = vec![PortRef::new(NodeId(1), "chunk")];

    LearningGraph {
        schema_version: 1,
        inputs: vec![obs.clone()],
        outputs: vec![actions],
        nodes: g,
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "w.safetensors".to_owned(),
                hash: [0u8; 32],
            },
            contract: PolicyContract {
                inputs: [(obs.name.clone(), obs)].into_iter().collect(),
                observation_window: 1,
                action_dim: ACTION_DIM as u32,
                horizon: HORIZON as u32,
                execute_chunk: HORIZON as u32,
                replanning_hz: 50.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 1.0,
                    deadline_ms: 20.0,
                },
            },
        },
    }
}

/// Every combination the two parameters and the squash can take, with the `nn.` spelling the
/// design note's table promises for each.
fn combinations() -> Vec<(Activation, &'static str, bool, Squash)> {
    let mut out = Vec::new();
    for (activation, spelling) in [
        (Activation::Relu, "nn.ReLU()"),
        (Activation::Elu, "nn.ELU()"),
        (Activation::Swish, "nn.SiLU()"),
        (Activation::Tanh, "nn.Tanh()"),
    ] {
        for activate_output in [false, true] {
            for squash in [Squash::None, Squash::Tanh] {
                out.push((activation, spelling, activate_output, squash));
            }
        }
    }
    out
}

#[test]
fn lower_mlp_activations_source() {
    // 1. The committed document lowers to the byte-identical source it always did.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/visible-learning/learning.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let committed = es_ir::serial::learning_from_toml(&text).expect("learning.toml parses");
    let module = lower_to_torch(&committed).expect("the demo graph lowers");
    assert_eq!(
        hex(&module.lowering_hash),
        COMMITTED_LOWERING_HASH,
        "the default activation and squash moved the demo's lowering"
    );
    assert!(module.source.contains("nn.ReLU()"), "{}", module.source);
    assert!(!module.source.contains("torch.tanh("), "{}", module.source);

    // 2. Every combination spells what the node asked for, and none of them moves a key.
    let base = lower_to_torch(&graph(Activation::Relu, false, Squash::None)).expect("lowers");
    for (activation, spelling, activate_output, squash) in combinations() {
        let m = lower_to_torch(&graph(activation, activate_output, squash)).expect("lowers");
        let tail = if activate_output {
            format!(", {spelling}")
        } else {
            String::new()
        };
        let expected = format!(
            "nn.Sequential(nn.Linear({}, {}), {spelling}, nn.Linear({}, {}){tail})",
            DIMS[0], DIMS[1], DIMS[1], DIMS[2]
        );
        assert!(
            m.source.contains(&expected),
            "{activation:?}/{activate_output}: expected `{expected}` in\n{}",
            m.source
        );
        assert_eq!(
            m.source.contains(&format!(
                "torch.tanh(self.n1(v0_out)).reshape(-1, {HORIZON}, {ACTION_DIM})"
            )),
            squash == Squash::Tanh,
            "{squash:?} is not what the head lowered to:\n{}",
            m.source
        );
        // An activation carries no weights (spec 8.3), so the checkpoint contract is the same
        // one for all sixteen graphs.
        assert_eq!(
            m.weight_keys, base.weight_keys,
            "{activation:?} moved a key"
        );
        assert_eq!(m.weight_shapes, base.weight_shapes, "{activation:?}");
        // Only the default combination may reproduce the default source.
        assert_eq!(
            m.lowering_hash == base.lowering_hash,
            activation == Activation::Relu && !activate_output && squash == Squash::None,
            "{activation:?}/{activate_output}/{squash:?} lowers to the default source"
        );
    }
    println!(
        "RAN lower_mlp_activations_source: committed lowering_hash {COMMITTED_LOWERING_HASH}, \
         {} combinations",
        combinations().len()
    );
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
}

fn checkpoint(shapes: &BTreeMap<String, Vec<u64>>, seed: u64) -> Checkpoint {
    let mut rng = Rng(seed);
    shapes
        .iter()
        .map(|(name, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            let values: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            (name.clone(), (shape.clone(), values))
        })
        .collect()
}

/// `((37 i + 11 j) % 4096) / 256 - 8` — `mlp_activation_ref.py`'s `ramp`. Integer arithmetic
/// over a power of two, so the observation is bit-identical in both languages without any
/// tensor crossing the process boundary.
fn ramp(sample: usize, dim: usize) -> f32 {
    ((sample * 37 + dim * 11) % 4096) as f32 / 256.0 - 8.0
}

#[derive(serde::Deserialize)]
struct RefReply {
    ok: bool,
    #[serde(default)]
    error: String,
    #[serde(default)]
    torch: String,
    #[serde(default)]
    actions: Vec<Vec<f32>>,
}

fn reference(weights: &str, spec: &str) -> Result<RefReply, String> {
    let python = std::env::var("ES_PYTHON").unwrap_or_else(|_| "python".to_owned());
    let out = Command::new(&python)
        .args(["-c", REF, weights, spec])
        .output()
        .map_err(|e| format!("cannot start `{python}`: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().ok_or_else(|| {
        format!(
            "no output; stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    })?;
    let reply: RefReply = serde_json::from_str(line)
        .map_err(|e| format!("{e} in `{}`", &line[..line.len().min(300)]))?;
    if !reply.ok {
        return Err(reply.error);
    }
    Ok(reply)
}

fn f32s(t: &Tensor) -> Vec<f32> {
    t.data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs ES_PYTHON with torch (spec 1.4 reference oracle)"]
fn lower_mlp_activations_match_torch() {
    if let Err(why) = es_policy::torch_runtime::is_available() {
        panic!("the ignored oracle was asked for and torch is not there: {why}");
    }
    let dir = std::env::temp_dir().join("es-policy-mlp-activations");
    std::fs::create_dir_all(&dir).expect("a scratch directory");

    let mut torch_version = String::new();
    for (activation, _, activate_output, squash) in combinations() {
        let g = graph(activation, activate_output, squash);
        let module = lower_to_torch(&g).expect("lowers");
        let tensors = checkpoint(&module.weight_shapes, 0x1234_5678_9abc_def1);
        let bytes = write_safetensors(&tensors);
        let path = dir.join(format!(
            "{activation:?}-{activate_output}-{squash:?}.safetensors"
        ));
        std::fs::write(&path, &bytes).expect("the weights are written");

        let mut g = g;
        g.policy.weights = WeightsRef::Safetensors {
            path: path.to_string_lossy().into_owned(),
            hash: *blake3::hash(&bytes).as_bytes(),
        };
        let mut rt = TorchRuntime::new();
        rt.load(&g, &es_policy::WeightsSource::Safetensors(path.clone()))
            .expect("torch is available, so the load must succeed");

        let spec = format!(
            r#"{{"dims": [{}, {}, {}], "action_dim": {ACTION_DIM}, "horizon": {HORIZON}, "activation": "{activation:?}", "activate_output": {activate_output}, "squash": "{squash:?}", "count": {SAMPLES}}}"#,
            DIMS[0], DIMS[1], DIMS[2]
        );
        let reply = reference(&path.to_string_lossy(), &spec)
            .unwrap_or_else(|why| panic!("{activation:?}/{activate_output}/{squash:?}: {why}"));
        torch_version = reply.torch;
        assert_eq!(reply.actions.len(), SAMPLES);

        for (sample, want) in reply.actions.iter().enumerate() {
            let x: Vec<f32> = (0..DIMS[0] as usize).map(|d| ramp(sample, d)).collect();
            let inputs: BTreeMap<String, Tensor> = [(
                "obs".to_owned(),
                Tensor {
                    dtype: ElemType::F32,
                    shape: vec![DIMS[0]],
                    data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
            )]
            .into();
            let got = rt.infer(&inputs).expect("forward pass");
            let got = f32s(&got["actions"]);
            assert_eq!(
                got, *want,
                "{activation:?}/{activate_output}/{squash:?} sample {sample}: the lowered \
                 module and the hand-written reference are not bitwise equal"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
    println!(
        "RAN lower_mlp_activations_match_torch: torch {torch_version}, {} combinations x \
         {SAMPLES} observations, bitwise",
        combinations().len()
    );
}
