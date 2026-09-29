//! Packet P-M15-R2: the one-token attention trains under exact attention on CUDA.
//!
//! A lowered `TemporalEncoder { Transformer }` over a pooled feature (`token_count == 0`) runs
//! as a one-token sequence, so its softmax is exactly 1 and the attention's query and key
//! projections have an exactly zero gradient. torch's memory-efficient SDPA kernel returned
//! rounding noise for that zero on CUDA (its forward was exact); `AdamW` normalised the noise into
//! learning-rate-sized steps, the logits grew, and training went NaN
//! (`docs/design/visible-learning.md` open question 35). `train_act.py` now trains on CUDA with
//! the flash, memory-efficient and cuDNN SDPA backends off, which leaves the math one.
//!
//! Python- and CUDA-gated (`ES_PYTHON` with `torch`, and a CUDA device; a loud SKIP
//! otherwise): one optimizer step of `train_act.py`'s own `main` on a small one-token-transformer
//! graph, on CUDA, leaves the q and k rows of the attention's `in_proj_weight` gradient exactly
//! zero and finite while its v rows are not zero, and the summary says `"sdpa": "math"`.

use std::path::PathBuf;
use std::process::Command;

use es_ir::graph::{Graph, NodeId, Port, PortRef};
use es_ir::learning::{
    action_unit, ActionExecutionMode, Activation, ArchKind, ChunkBlendPolicy, HeadKind,
    LearningGraph, LearningNode, PolicyContract, PolicyHandle, RuntimeHints, Squash,
    StateEncoderKind, TemporalKind, WeightsRef,
};
use es_ir::types::{ElemType, Frame, PortType, Shape, TimeRef, Unit};
use es_policy::lower::Contract;
use es_policy::lower_to_torch;

/// The transformer's width: eight heads of eight.
const W: u32 = 64;
const STATE: u64 = 8;
const ACTION_DIM: u32 = 2;
const HORIZON: u32 = 4;
/// The `TemporalEncoder`'s node id, which is its member name `n1` in the lowered module.
const TRANSFORMER: u32 = 1;

fn ty(shape: &[u64], unit: Unit) -> PortType {
    PortType {
        elem: ElemType::F32,
        shape: Shape::new(shape.to_vec()),
        unit,
        frame: Frame::Policy,
        time: TimeRef::Tick,
        image: None,
    }
}

fn feature(name: &str) -> Port {
    Port::new(name, ty(&[u64::from(W)], Unit::Dimensionless))
}

/// State MLP (0) -> one-token transformer (1) -> regression head (2) -> chunker (3).
fn one_token_graph() -> LearningGraph {
    let state = Port::new("joint_state", ty(&[STATE], action_unit()));
    let chunk = |name: &str| {
        Port::new(
            name,
            ty(&[u64::from(HORIZON), u64::from(ACTION_DIM)], action_unit()),
        )
    };
    let mut g = Graph::new(1);
    g.insert(
        NodeId(0),
        LearningNode::StateEncoder {
            inputs: vec![state.clone()],
            kind: StateEncoderKind::Mlp {
                hidden: vec![W],
                activation: Activation::Relu,
                activate_output: false,
            },
            out_dim: W,
        },
    );
    g.insert(
        NodeId(TRANSFORMER),
        LearningNode::TemporalEncoder {
            inputs: vec![feature("seq")],
            kind: TemporalKind::Transformer,
            n_frames: 1,
            out_dim: W,
            token_count: 0,
        },
    );
    g.insert(
        NodeId(2),
        LearningNode::PolicyHead {
            inputs: vec![feature("feat")],
            kind: HeadKind::Regression,
            action_dim: ACTION_DIM,
            horizon: HORIZON,
            squash: Squash::None,
        },
    );
    g.insert(
        NodeId(3),
        LearningNode::ActionChunker {
            inputs: vec![chunk("chunk")],
            horizon: HORIZON,
            execute_chunk: HORIZON,
            replan_hz: 5.0,
            mode: ActionExecutionMode::RecedingHorizon,
            blend: ChunkBlendPolicy::HardSwitch,
            buffer_chunks: 2,
        },
    );
    g.connect(NodeId(0), "out", NodeId(TRANSFORMER), "seq");
    g.connect(NodeId(TRANSFORMER), "out", NodeId(2), "feat");
    g.connect(NodeId(2), "chunk", NodeId(3), "chunk");
    g.inputs = vec![PortRef::new(NodeId(0), "joint_state")];
    g.outputs = vec![PortRef::new(NodeId(3), "actions")];
    LearningGraph {
        schema_version: 1,
        outputs: vec![chunk("actions")],
        nodes: g,
        policy: PolicyHandle {
            architecture: ArchKind::Act,
            base_model: None,
            weights: WeightsRef::Safetensors {
                path: "policy.safetensors".to_owned(),
                hash: [0; 32],
            },
            contract: PolicyContract {
                inputs: [(state.name.clone(), state.clone())].into_iter().collect(),
                observation_window: 1,
                action_dim: ACTION_DIM,
                horizon: HORIZON,
                execute_chunk: HORIZON,
                replanning_hz: 5.0,
                execution_mode: ActionExecutionMode::RecedingHorizon,
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: 5.0,
                    deadline_ms: 50.0,
                },
            },
        },
        inputs: vec![state],
    }
}

/// argv is one JSON spec: `module` (the lowered directory), `train_act`, `dir` (scratch),
/// `node` (the transformer's id). It writes a baked set the way `es dataset bake` lays one out,
/// runs `train_act.py`'s own `main` for one optimizer step on CUDA with `build_policy` wrapped
/// only to keep the module it built, and prints one JSON line: the q, k and v rows' largest
/// absolute gradient, whether the gradient is finite, and the summary's `sdpa`. Without CUDA it
/// prints `{"cuda": false}` and nothing runs.
const ORACLE_PY: &str = r#"
import contextlib, io, json, sys
from pathlib import Path
import torch

spec = json.loads(sys.argv[1])
if not torch.cuda.is_available():
    print(json.dumps({"cuda": False, "torch": torch.__version__}))
    sys.exit(0)
namespace = {}
source = Path(spec["train_act"]).read_text(encoding="utf-8")
exec(compile(source, spec["train_act"], "exec"), namespace)
built, build = [], namespace["build_policy"]
namespace["build_policy"] = lambda directory: built.append(build(directory)) or built[-1]

module, scratch = Path(spec["module"]), Path(spec["dir"])
contract = json.loads((module / "contract.json").read_text(encoding="utf-8"))
frames, draw = 16, torch.Generator().manual_seed(0)
tensors = {p: torch.randn([frames] + s, generator=draw) for p, s in contract["inputs"].items()}
tensors["action"] = torch.randn([frames, contract["action_dim"]], generator=draw)
baked = scratch / "baked"
baked.mkdir(parents=True, exist_ok=True)
namespace["write_safetensors"](baked / "episode_000000.safetensors", tensors)
(baked / "manifest.json").write_text(json.dumps({
    "tensors": sorted(tensors),
    "episodes": [{"file": "episode_000000.safetensors", "frames": frames}],
}), encoding="utf-8")

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    namespace["main"]([
        "--module", str(module), "--baked", str(baked),
        "--out", str(scratch / "model.safetensors"),
        "--device", "cuda", "--checkpoint-at", "1", "--lr", "4e-4",
    ])
summary = json.loads(stdout.getvalue().splitlines()[-1])
attention = getattr(built[0], "n%d" % spec["node"]).layers[0].self_attn
grad, e = attention.in_proj_weight.grad, attention.embed_dim
largest = lambda rows: float(rows.abs().max())
print(json.dumps({
    "cuda": True,
    "device": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "steps": summary["steps"],
    "sdpa": summary.get("sdpa"),
    "dq": largest(grad[:e]),
    "dk": largest(grad[e : 2 * e]),
    "dv": largest(grad[2 * e :]),
    "finite": bool(torch.isfinite(grad).all()),
    "enabled": {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    },
}))
"#;

/// `ES_PYTHON` with `torch`, or the reason to SKIP. Everything after this probe is an error,
/// never a skip (`docs/reviews/M4.md:67`, S-7) -- except the missing CUDA device, which the
/// oracle itself reports.
fn python() -> Result<String, String> {
    let python = std::env::var("ES_PYTHON")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .ok_or("ES_PYTHON is unset")?;
    let out = Command::new(&python)
        .args(["-c", "import torch"])
        .output()
        .map_err(|e| format!("`{python}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`{python}` has no torch: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
        ));
    }
    Ok(python)
}

#[test]
fn one_token_attention_has_an_exactly_zero_query_and_key_gradient_on_cuda() {
    let python = match python() {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP one_token_attention_has_an_exactly_zero_query_and_key_gradient_on_cuda: {why}");
            return;
        }
    };
    let dir = std::env::temp_dir().join(format!("es-exact-attention-{}", std::process::id()));
    let module = dir.join("module");
    std::fs::create_dir_all(&module).expect("a scratch directory");
    let g = one_token_graph();
    let m = lower_to_torch(&g).expect("the one-token graph lowers");
    assert!(
        m.source.contains(".unsqueeze(1)).squeeze(1)"),
        "the transformer must run as a one-token sequence: {}",
        m.source
    );
    std::fs::write(module.join("es_policy.py"), &m.source).expect("write the module");
    std::fs::write(
        module.join("contract.json"),
        serde_json::to_string(&Contract::new(&m, &g)).expect("serialises"),
    )
    .expect("write the contract");
    let train_act = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/es/train_act.py");
    let spec = serde_json::json!({
        "module": module, "train_act": train_act, "dir": dir, "node": TRANSFORMER,
    });
    let out = Command::new(&python)
        .args(["-c", ORACLE_PY, &spec.to_string()])
        .output()
        .expect("run the oracle");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    let reply: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or(""))
        .unwrap_or_else(|e| panic!("{e}: {stdout}\n{stderr}"));
    if reply["cuda"] == false {
        println!(
            "SKIP one_token_attention_has_an_exactly_zero_query_and_key_gradient_on_cuda: \
             torch {} sees no CUDA device",
            reply["torch"]
        );
        return;
    }
    println!("oracle: {reply}");
    assert_eq!(reply["steps"], 1, "{reply}");
    assert_eq!(reply["finite"], true, "a non-finite gradient: {reply}");
    assert!(
        reply["dq"] == 0.0 && reply["dk"] == 0.0,
        "one key makes softmax exactly 1, so the q and k rows' gradient is exactly zero; this \
         run's is not, which is the memory-efficient kernel's backward: {reply}"
    );
    assert!(
        reply["dv"].as_f64().is_some_and(|v| v > 0.0),
        "the v rows had no gradient, so the zero above says nothing: {reply}"
    );
    assert_eq!(
        reply["sdpa"], "math",
        "the summary must say how it trained: {reply}"
    );
    let _ = std::fs::remove_dir_all(&dir);
    println!("RAN one_token_attention_has_an_exactly_zero_query_and_key_gradient_on_cuda");
}
