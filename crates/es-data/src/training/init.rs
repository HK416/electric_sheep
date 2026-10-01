//! Where a policy starts: `[init] policy`'s warm start from a bundle (packet M8/S1), and the
//! untrained bundle `es policy init` builds from four documents (packet M12/Y5).

use std::collections::BTreeMap;
use std::path::Path;

use es_ir::learning::{LearningGraph, LearningNode, WeightsRef};
use es_ir::observation::ObservationIr;
use serde_json::{json, Value};

use super::refuse;
use crate::collect::hex;
use crate::DataError;

// --- starting from a policy (packet M8/S1) -----------------------------------------------

/// `training/init.lock`, the thirteenth file of `<out>/training/` — written only by a recipe
/// that names `[init] policy`, and its digest is what enters `config.json` and therefore
/// `identity_hash` / `training_hash`.
pub const INIT_LOCK: &str = "init.lock";

/// What [`init_from`] decided: the lock to record and the tensors to hand the trainer.
#[derive(Clone, Debug)]
pub struct Init {
    /// `training/init.lock`'s body, sorted names.
    pub lock: Value,
    /// The copied tensors, ready for `es_policy::weights::write_safetensors`.
    pub weights: es_policy::weights::Checkpoint,
    /// `copied.len()`, so the shell can print it without re-reading the lock.
    pub copied: usize,
    pub initialised: usize,
}

/// Compare `[init] policy`'s bundle to the module `target` lowers to, and copy what fits.
///
/// Three buckets and no fourth (the packet's schema): **copied** is a tensor whose name and
/// shape the lowered module declares, **initialised** is a name the module declares and the
/// bundle does not carry, and **`shape_mismatch`** is a name both know at two shapes — recorded
/// and *not* copied, because reshaping a trained tensor is guessing and the failure mode of a
/// guess here is a policy that still trains and still looks fine.
///
/// A `nodes.<k>.*` prefix claim covers a sub-module whose parameter names belong to
/// torchvision or `torch.nn` (`es_policy::weights::validate_keys`). Its tensors are copied by
/// name alone, because there is no declared shape on this side to check them against;
/// `train_act.py --init-weights` checks each one against the real module before loading it,
/// which is where the shape actually lives.
///
/// Zero copied is a refusal. A warm start that shares nothing with what it starts from is not
/// a warm start, and letting it through would write a provenance record whose whole content
/// is "none of this was used".
pub fn init_from(source: &str, bundle: &[u8], target: &LearningGraph) -> Result<Init, DataError> {
    let opened = es_compile::PolicyBundle::open(bundle)
        .map_err(|e| refuse(format!("[init] `policy` {source}: {e}")))?;
    let header = es_policy::weights::parse_header(&opened.weights)
        .map_err(|e| refuse(format!("[init] `policy` {source}: {e}")))?;
    let module = es_policy::lower_to_torch(target)
        .map_err(|e| refuse(format!("the bundle being trained does not lower: {e}")))?;

    let claims: Vec<&str> = module
        .weight_keys
        .iter()
        .filter_map(|k| k.strip_suffix('*'))
        .collect();
    // Safetensors is eight bytes of header length, the header, then the data segment; an
    // entry's offsets are relative to the end of the header. `parse_header` succeeded, so
    // these eight bytes are there and the segment behind them is long enough.
    let data_at = 8 + u64::from_le_bytes(
        opened.weights[..8]
            .try_into()
            .expect("parse_header already read these eight bytes"),
    ) as usize;

    let (mut copied, mut mismatch) = (Vec::new(), Vec::new());
    let mut weights = es_policy::weights::Checkpoint::new();
    for (name, entry) in &header {
        let declared = module.weight_shapes.get(name);
        if declared.is_none() && !claims.iter().any(|p| name.starts_with(p)) {
            // A tensor the module does not declare at all: not copied, and not one of the
            // three buckets either -- the module has no slot to put it in.
            continue;
        }
        if let Some(want) = declared {
            if entry.shape != *want {
                mismatch.push(json!({
                    "name": name, "expected": want, "found": entry.shape,
                }));
                continue;
            }
        }
        if entry.dtype != "F32" {
            return Err(refuse(format!(
                "[init] `policy` {source}: {name} is {}, and every tensor on this path is \
                 F32 (spec 8.4). A second dtype here would be a second reader of one format",
                entry.dtype
            )));
        }
        let (a, b) = entry.offsets;
        let bytes = &opened.weights[data_at + a as usize..data_at + b as usize];
        let values = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weights.insert(name.clone(), (entry.shape.clone(), values));
        copied.push(name.clone());
    }

    let initialised: Vec<String> = module
        .weight_keys
        .iter()
        .filter(|key| match key.strip_suffix('*') {
            Some(prefix) => !header.keys().any(|k| k.starts_with(prefix)),
            None => !header.contains_key(*key),
        })
        .cloned()
        .collect();

    if copied.is_empty() {
        return Err(refuse(format!(
            "[init] `policy` {source} shares no tensor with the module this recipe trains: \
             {} name(s) the module declares are absent from the bundle and {} more disagree \
             about a shape. Starting from a policy that shares nothing is a mistake, not a \
             warm start -- the run would be `steps` steps from scratch under a document \
             saying otherwise",
            initialised.len(),
            mismatch.len()
        )));
    }

    // `copied` and `initialised` come out of a BTreeMap and a key list that is already
    // sorted, but say it rather than rely on it: the lock's digest is a hash slot.
    copied.sort();
    let mut initialised = initialised;
    initialised.sort();
    Ok(Init {
        copied: copied.len(),
        initialised: initialised.len(),
        lock: json!({
            "schema_version": 1,
            "source": source,
            "policy_hash": opened.manifest.hashes.policy.as_ref().map(hex),
            "learning_hash": opened.manifest.hashes.learning.as_ref().map(hex),
            "copied": copied,
            "initialised": initialised,
            "shape_mismatch": mismatch,
        }),
        weights,
    })
}

/// The bundle a cycle's `[collect] policy` names, from four documents (`es policy init`,
/// packet M12/Y5). The demonstrator drives under `--expert`, so the weights are never loaded:
/// they are a placeholder that names the seed, which is what every untrained bundle in this
/// repository has been (nothing on the Rust side initialises weights). What the bundle is for
/// is its documents -- above all the Deployment IR, the Safety Plane the demonstrator runs
/// under. The same documents and seed give the same bytes.
///
/// Without `learning` the Learning IR is [`external_policy`]'s (packet M12/Y5b): the collect
/// bundle of an Observation IR no committed Learning IR takes, such as `observation-v8.toml`.
pub fn untrained_bundle(
    task: &Path,
    observation: &Path,
    learning: Option<&Path>,
    deployment: &Path,
    seed: u64,
) -> Result<Vec<u8>, DataError> {
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| DataError::io(p, e));
    let bad = |p: &Path, e: &dyn std::fmt::Display| refuse(format!("{}: {e}", p.display()));
    let task_ir = es_ir::serial::task_from_toml(&read(task)?).map_err(|e| bad(task, &e))?;
    let observation_ir = es_ir::serial::observation_from_toml(&read(observation)?)
        .map_err(|e| bad(observation, &e))?;
    let deployment_ir =
        es_ir::serial::deployment_from_toml(&read(deployment)?).map_err(|e| bad(deployment, &e))?;
    let weights = format!("es policy init: untrained, never loaded, seed {seed}").into_bytes();
    let hash = *blake3::hash(&weights).as_bytes();
    let learning_ir = match learning {
        Some(p) => {
            let mut ir = es_ir::serial::learning_from_toml(&read(p)?).map_err(|e| bad(p, &e))?;
            ir.policy.weights = WeightsRef::Safetensors {
                path: ir.policy.weights.path().to_owned(),
                hash,
            };
            ir
        }
        None => external_policy(&observation_ir, &deployment_ir, hash),
    };
    es_compile::PolicyBundle::build(
        &task_ir,
        &observation_ir,
        &learning_ir,
        &deployment_ir,
        &weights,
    )
    .map_err(|e| refuse(format!("the four documents do not make a bundle: {e}")))
}

/// Spec 8.1's shape for an external policy, built from the two documents it has to agree with:
/// the contract takes every Observation IR output at the policy tick and returns the Deployment
/// IR's action chunk, and the graph is one `PolicyBundle` node over those ports with an empty
/// boundary. It mirrors the Learning IR `es policy import-lerobot` writes
/// (`crates/es/src/cmd/policy.rs`, `import_lerobot`), which lives in the `es` binary and so is
/// out of this layer's reach; the cadence and the deadline are that function's.
///
/// One number is not: `expected_latency_ms` is one control period, not `import_lerobot`'s
/// re-plan period. `latency_ticks` makes that one tick, which is what `learning.toml`'s 15 ms
/// makes it -- so the scripted demonstrator, which drives through the chunk buffer under this
/// latency, collects and passes the gate under the timing it has always been measured under
/// (`expert_passes_the_evaluation_harness`, the V15 demonstrations).
fn external_policy(
    observation: &ObservationIr,
    deployment: &es_ir::deployment::DeploymentIr,
    weights_hash: [u8; 32],
) -> LearningGraph {
    use es_ir::deployment::ExecutionMode;
    use es_ir::learning::{
        ActionExecutionMode, ArchKind, PolicyContract, PolicyHandle, RuntimeHints, TensorPort,
    };
    use es_ir::types::{ElemType, Frame, PortType, TimeRef};

    let action = deployment.action;
    let inputs: BTreeMap<String, TensorPort> = observation
        .outputs
        .iter()
        .map(|(name, out)| {
            let ty = PortType {
                frame: Frame::Policy,
                time: TimeRef::Tick,
                image: None,
                ..out.ty.clone()
            };
            (name.clone(), TensorPort::new(name.clone(), ty))
        })
        .collect();
    let weights = WeightsRef::Safetensors {
        path: "policy.safetensors".to_owned(),
        hash: weights_hash,
    };
    let mut nodes = es_ir::graph::Graph::new(1);
    nodes.insert(
        es_ir::graph::NodeId(0),
        LearningNode::PolicyBundle {
            inputs: inputs.values().cloned().collect(),
            weights: weights.clone(),
            action_dim: action.dim as u32,
            horizon: action.horizon as u32,
        },
    );
    let ms = |micros: u64| micros as f32 / 1000.0;
    LearningGraph {
        schema_version: 1,
        inputs: Vec::new(),
        nodes,
        outputs: Vec::new(),
        policy: PolicyHandle {
            // Not `Act`: nothing here is an architecture, only a contract.
            architecture: ArchKind::Bundle,
            base_model: None,
            weights,
            contract: PolicyContract {
                inputs,
                observation_window: observation.temporal.window.map_or(1, |w| w.n_steps),
                action_dim: action.dim as u32,
                horizon: action.horizon as u32,
                execute_chunk: action.execute_chunk as u32,
                replanning_hz: (deployment.rate.control.as_hz_f64()
                    / action.execute_chunk.max(1) as f64) as f32,
                execution_mode: match deployment.execution {
                    ExecutionMode::OpenLoopChunk => ActionExecutionMode::OpenLoopChunk,
                    ExecutionMode::RecedingHorizon => ActionExecutionMode::RecedingHorizon,
                    ExecutionMode::TemporalEnsemble { .. } => ActionExecutionMode::TemporalEnsemble,
                    ExecutionMode::RealTimeChunking => ActionExecutionMode::RealTimeChunking,
                },
                runtime: RuntimeHints {
                    dtype: ElemType::F32,
                    expected_latency_ms: ms(deployment.rate.control_period().0),
                    deadline_ms: ms(deployment.deadlines.inference_budget.0),
                },
            },
        },
    }
}
