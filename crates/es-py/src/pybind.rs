//! pyo3 bindings for the `es_native` extension module (`python` feature only, spec 14.2).
//!
//! A thin shell over [`crate::builder`]: every method here either forwards straight to the
//! generic core or adds one of the Pythonic sugar names spec 14.2 shows (`.reward`,
//! `.terminate`, `Learning.act`, `.save`) as a couple of lines wired to `add`/`connect` — none
//! of it constructs a node by hand, so a new node kind never needs a new method here.
//!
//! Params cross the FFI boundary as JSON **text**, not a hand-walked `PyObject` tree: the
//! Python side already has `json.dumps` for exactly this, and no `pythonize`-style crate is on
//! this packet's approved dependency list (`pyo3` only). `python/es/builder.py` is what turns
//! this into the fluent surface of the spec 14.2 snippet.
//!
//! Every builder pyclass is `#[pyclass(unsendable)]`: the node factories they hold (`Box<dyn
//! TaskNodeFactory>` etc.) are not `Send`/`Sync`, and an authoring script builds one IR at a
//! time on one thread anyway (spec 14.2 has no concurrent-authoring use case), so requiring
//! `Send` here would mean wrapping every registry in a `Mutex` for a property nothing needs.

use std::collections::BTreeMap;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use es_ir::graph::NodeId;
use es_ir::Diagnostic;

use crate::builder::{
    DeploymentBuilder, LearningBuilder, ObservationBuilder, SaveError, TaskBuilder,
};
use crate::rollout::{Rollout as RolloutCore, RolloutError};

// Raised for every compiler diagnostic (spec 14.2: "type errors surface as Python exceptions
// carrying the same diagnostics the compiler produces"). `args` is `(text, diagnostics_json)` so
// a caller can either print it as-is or parse the second element back into structured data.
// (A `///` doc comment on this macro invocation is dropped silently by rustdoc; `//` says so.)
create_exception!(es_native, EsDiagnosticError, PyException);

fn parse_json(s: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("invalid JSON: {e}")))
}

fn diags_err(diags: &[Diagnostic]) -> PyErr {
    let text = diags
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let json = serde_json::to_string(diags).unwrap_or_default();
    EsDiagnosticError::new_err((text, json))
}

// Both take their argument by value, not a reference, on purpose: every call site is
// `.map_err(diag_err)` / `.map_err(save_err)`, which needs an `FnOnce(E) -> PyErr` function
// pointer, so the value-taking signature is the one that lets 30-odd call sites stay one line
// instead of a `.map_err(|e| diag_err(&e))` closure each.
#[allow(clippy::needless_pass_by_value)]
fn diag_err(d: Diagnostic) -> PyErr {
    diags_err(std::slice::from_ref(&d))
}

#[allow(clippy::needless_pass_by_value)]
fn save_err(e: SaveError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `already built/saved` — `build`/`save` take the builder's inner state, so a second call
/// needs its own clear error rather than a confusing `None` panic.
fn taken_err() -> PyErr {
    PyValueError::new_err("this builder was already built or saved; construct a new one")
}

// --- Task ------------------------------------------------------------------------------------

/// `es.Task(name, scene=...)` (spec 14.2). `scene_json` is an `es_ir::task::SceneRef`,
/// `config_json` an `es_ir::task::TaskConfig`.
#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Task(Option<TaskBuilder>);

// Not `#[pymethods]`: `inner()` is a plain Rust helper, never a method Python calls.
impl Task {
    fn inner(&mut self) -> PyResult<&mut TaskBuilder> {
        self.0.as_mut().ok_or_else(taken_err)
    }
}

#[pymethods]
impl Task {
    #[new]
    fn new(scene_json: &str, config_json: &str) -> PyResult<Self> {
        let scene = serde_json::from_str(scene_json)
            .map_err(|e| PyValueError::new_err(format!("bad scene: {e}")))?;
        let config = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("bad config: {e}")))?;
        Ok(Self(Some(TaskBuilder::new(scene, config))))
    }

    /// The generic escape hatch every sugar method below reduces to: build one spec 6.3 node.
    fn add(&mut self, kind: &str, params_json: &str) -> PyResult<u32> {
        let params = parse_json(params_json)?;
        self.inner()?
            .add(kind, &params)
            .map(|id| id.0)
            .map_err(diag_err)
    }

    fn connect(&mut self, from: u32, from_port: &str, to: u32, to_port: &str) -> PyResult<()> {
        self.inner()?
            .connect(NodeId(from), from_port, NodeId(to), to_port);
        Ok(())
    }

    /// `.observe(name, source_json, ty_json)` (spec 14.2 `observe_spec`): one Task ->
    /// Observation IR channel. `python/es/builder.py`'s `observe_spec(**channels)` calls this
    /// once per keyword argument.
    fn observe(&mut self, name: &str, source_json: &str, ty_json: &str) -> PyResult<()> {
        let channel =
            serde_json::json!({"source": parse_json(source_json)?, "ty": parse_json(ty_json)?});
        self.inner()?.declare_obs(name, channel).map_err(diag_err)
    }

    /// `.reward(name, weight, value_node, value_port, ty_json)` (spec 14.2): sugar for
    /// `add("Reward", {...})` wired to an already-built value expression.
    fn reward(
        &mut self,
        name: &str,
        weight: f64,
        value: u32,
        value_port: &str,
        ty_json: &str,
    ) -> PyResult<u32> {
        let ty = parse_json(ty_json)?;
        let id = self
            .inner()?
            .add(
                "Reward",
                &serde_json::json!({"name": name, "weight": weight, "aggregation": "Sum", "ty": ty}),
            )
            .map_err(diag_err)?;
        self.inner()?
            .connect(NodeId(value), value_port, id, "value");
        Ok(id.0)
    }

    /// `.terminate(name, when_node, when_port)` (spec 14.2). `TerminationKind` is picked from
    /// `name` by convention (`"success"`/`"fail*"`/else `"timeout"`) so the call needs no extra
    /// argument for it.
    fn terminate(&mut self, name: &str, when: u32, when_port: &str) -> PyResult<u32> {
        let lower = name.to_lowercase();
        let kind = if lower.starts_with("fail") {
            "Failure"
        } else if lower.starts_with("succ") {
            "Success"
        } else {
            "Timeout"
        };
        let id = self
            .inner()?
            .add("Terminate", &serde_json::json!({"kind": kind}))
            .map_err(diag_err)?;
        self.inner()?.connect(NodeId(when), when_port, id, "value");
        Ok(id.0)
    }

    /// Builds and validates (spec 14.2: "the builder does not execute immediately"), returning
    /// the `task.toml` text `save` would also write.
    fn build(&mut self) -> PyResult<String> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        es_ir::serial::task_to_toml(&ir).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `.save(path)` (spec 14.2, spec 14.3 `task.toml`).
    fn save(&mut self, path: &str) -> PyResult<()> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        TaskBuilder::save_toml(&ir, std::path::Path::new(path)).map_err(save_err)
    }
}

// --- Observation ---------------------------------------------------------------------------

/// `es.Observation(name, task=task)` (spec 14.2). `task_ref` is the 32-byte `task_hash` hex.
#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Observation(Option<ObservationBuilder>);

impl Observation {
    fn inner(&mut self) -> PyResult<&mut ObservationBuilder> {
        self.0.as_mut().ok_or_else(taken_err)
    }
}

#[pymethods]
impl Observation {
    #[new]
    fn new(task_ref_hex: &str) -> PyResult<Self> {
        let bytes = hex32(task_ref_hex)?;
        Ok(Self(Some(ObservationBuilder::new(bytes))))
    }

    fn add(&mut self, kind: &str, params_json: &str) -> PyResult<u32> {
        let params = parse_json(params_json)?;
        self.inner()?
            .add(kind, params)
            .map(|id| id.0)
            .map_err(diag_err)
    }

    fn connect(&mut self, from: u32, from_port: &str, to: u32, to_port: &str) -> PyResult<()> {
        self.inner()?
            .connect(NodeId(from), from_port, NodeId(to), to_port);
        Ok(())
    }

    /// `.temporal_window(n_steps=...)` / `.time_align(...)` (spec 14.2) both reduce to setting
    /// the spec 7.5 time model in one shot.
    fn set_temporal(&mut self, temporal_json: &str) -> PyResult<()> {
        let temporal = parse_json(temporal_json)?;
        self.inner()?.set_temporal(temporal).map_err(diag_err)
    }

    /// One named tensor Learning IR binds to, e.g. `"rgb_front"` (spec 7 `outputs`).
    fn declare_output(&mut self, name: &str, node: u32, port: &str, ty_json: &str) -> PyResult<()> {
        let ty = parse_json(ty_json)?;
        self.inner()?
            .declare_obs(name, NodeId(node), port, ty)
            .map_err(diag_err)
    }

    fn build(&mut self) -> PyResult<String> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        es_ir::serial::observation_to_toml(&ir).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        ObservationBuilder::save_toml(&ir, std::path::Path::new(path)).map_err(save_err)
    }
}

// --- Learning --------------------------------------------------------------------------------

/// `es.Learning(name, observation=obs)` (spec 14.2).
#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Learning(Option<LearningBuilder>);

impl Learning {
    fn inner(&mut self) -> PyResult<&mut LearningBuilder> {
        self.0.as_mut().ok_or_else(taken_err)
    }
}

#[pymethods]
impl Learning {
    #[new]
    fn new() -> Self {
        Self(Some(LearningBuilder::new()))
    }

    /// `es.Learning.act(...)` convenience constructor: the ACT (spec 14.2 example) shape —
    /// `VisionEncoder -> StateEncoder -> Fusion -> PolicyHead(regression) -> ActionChunker` —
    /// wired up in one call instead of five. Still routed entirely through `add`/`connect`, so
    /// it is sugar, not a second way to build a node.
    #[staticmethod]
    fn act(
        vision_out_dim: u32,
        state_out_dim: u32,
        action_dim: u32,
        horizon: u32,
    ) -> PyResult<Self> {
        let mut b = LearningBuilder::new();
        let vision = b
            .add(
                "VisionEncoder",
                &serde_json::json!({
                    "inputs": [], "backbone": "ResNet18", "pretrained": true, "frozen": false,
                    "out_dim": vision_out_dim, "token_count": 0
                }),
            )
            .map_err(diag_err)?;
        let state = b
            .add(
                "StateEncoder",
                &serde_json::json!({
                    "inputs": [], "kind": {"Mlp": {"hidden": [state_out_dim]}}, "out_dim": state_out_dim
                }),
            )
            .map_err(diag_err)?;
        let fusion = b
            .add(
                "Fusion",
                &serde_json::json!({
                    "inputs": [], "kind": "Concat",
                    "out_dim": vision_out_dim + state_out_dim, "token_count": 0
                }),
            )
            .map_err(diag_err)?;
        b.connect(vision, "out", fusion, "in0");
        b.connect(state, "out", fusion, "in1");
        let head = b
            .add(
                "PolicyHead",
                &serde_json::json!({
                    "inputs": [], "kind": "Regression", "action_dim": action_dim, "horizon": horizon
                }),
            )
            .map_err(diag_err)?;
        b.connect(fusion, "out", head, "in0");
        let chunker = b
            .add(
                "ActionChunker",
                &serde_json::json!({
                    "inputs": [], "horizon": horizon, "execute_chunk": horizon, "replan_hz": 10.0,
                    "mode": "TemporalEnsemble", "blend": "HardSwitch", "buffer_chunks": 1
                }),
            )
            .map_err(diag_err)?;
        b.connect(head, "out", chunker, "in0");
        Ok(Self(Some(b)))
    }

    fn add(&mut self, kind: &str, params_json: &str) -> PyResult<u32> {
        let params = parse_json(params_json)?;
        self.inner()?
            .add(kind, &params)
            .map(|id| id.0)
            .map_err(diag_err)
    }

    fn connect(&mut self, from: u32, from_port: &str, to: u32, to_port: &str) -> PyResult<()> {
        self.inner()?
            .connect(NodeId(from), from_port, NodeId(to), to_port);
        Ok(())
    }

    fn declare_input(&mut self, name: &str, ty_json: &str, node: u32, port: &str) -> PyResult<()> {
        let ty = parse_json(ty_json)?;
        self.inner()?
            .declare_input(name, ty, NodeId(node), port)
            .map_err(diag_err)
    }

    fn declare_output(&mut self, name: &str, ty_json: &str, node: u32, port: &str) -> PyResult<()> {
        let ty = parse_json(ty_json)?;
        self.inner()?
            .declare_obs(name, ty, NodeId(node), port)
            .map_err(diag_err)
    }

    fn set_policy(&mut self, policy_json: &str) -> PyResult<()> {
        let policy = parse_json(policy_json)?;
        self.inner()?.set_policy(policy).map_err(diag_err)
    }

    fn build(&mut self) -> PyResult<String> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        es_ir::serial::learning_to_toml(&ir).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        LearningBuilder::save_toml(&ir, std::path::Path::new(path)).map_err(save_err)
    }
}

// --- Deployment ------------------------------------------------------------------------------

/// `es.Deployment(name, robot=arm, action=...)` (spec 14.2). Deployment IR is a plain struct,
/// not a graph, so this is direct setters (spec 9) rather than `add`/`connect`.
#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Deployment(Option<DeploymentBuilder>);

impl Deployment {
    fn inner(&mut self) -> PyResult<&mut DeploymentBuilder> {
        self.0.as_mut().ok_or_else(taken_err)
    }
}

#[pymethods]
impl Deployment {
    #[new]
    fn new() -> Self {
        Self(Some(DeploymentBuilder::new()))
    }

    fn set_robot(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .set_robot(parse_json(value_json)?)
            .map_err(diag_err)
    }

    fn set_action(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .set_action(parse_json(value_json)?)
            .map_err(diag_err)
    }

    /// `.envelope(...)` (spec 9.3).
    fn envelope(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .envelope(parse_json(value_json)?)
            .map_err(diag_err)
    }

    fn set_execution(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .set_execution(parse_json(value_json)?)
            .map_err(diag_err)
    }

    fn set_deadlines(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .set_deadlines(parse_json(value_json)?)
            .map_err(diag_err)
    }

    fn set_rate(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .set_rate(parse_json(value_json)?)
            .map_err(diag_err)
    }

    /// `.watchdog(...)` (spec 9.4), one call per watchdog.
    fn watchdog(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .watchdog(parse_json(value_json)?)
            .map_err(diag_err)
    }

    /// `.fallback(...)` (spec 9.4).
    fn fallback(&mut self, value_json: &str) -> PyResult<()> {
        self.inner()?
            .fallback(parse_json(value_json)?)
            .map_err(diag_err)
    }

    fn build(&mut self) -> PyResult<String> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        es_ir::serial::deployment_to_toml(&ir).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn save(&mut self, path: &str) -> PyResult<()> {
        let inner = self.0.take().ok_or_else(taken_err)?;
        let ir = inner.build().map_err(|d| diags_err(&d))?;
        DeploymentBuilder::save_toml(&ir, std::path::Path::new(path)).map_err(save_err)
    }
}

// --- Rollout (packet M8/S4a) --------------------------------------------------------------

/// `es_native.Rollout(task_toml, observation_toml, deployment_toml, scene_xml, seed, n_envs)`
/// — the four documents as **text**, not paths, exactly like the JSON the builders take.
///
/// Lists in, lists out: no numpy here and therefore no numpy in the Rust crate either
/// (`docs/design/rl-continuation.md` section 3). `train_ppo.py` is the one that wants tensors
/// and is also the one that already imports torch, so the conversion belongs on its side —
/// adding `numpy` to `es-py` would put a second array ABI in the runtime for nobody's benefit.
///
/// `unsendable` for the same reason every builder above is: `Env`, `CpuPlan` and the backend
/// process handle are not `Send`, and a trainer steps its envs from the thread that built
/// them.
#[pyclass(unsendable)]
#[derive(Debug)]
pub struct Rollout(Kind);

/// `SafetyPlane<NJ, H>` and `ActionChunk<NJ, H>` are const-generic over the deployment's joint
/// count and horizon, which a document only reveals at run time. This is the same fixed table
/// `es eval run` keeps (`crates/es/src/cmd/eval.rs`, `dispatch_nj_h!`) rather than a
/// type-erased plane, which would be an eighth extension point (`INV-17`).
///
/// ponytail: add a pair here when a new robot/horizon needs a rollout; nothing else changes.
#[derive(Debug)]
enum Kind {
    N1H1(RolloutCore<1, 1>),
    N6H1(RolloutCore<6, 1>),
    N6H16(RolloutCore<6, 16>),
    N7H1(RolloutCore<7, 1>),
    N12H1(RolloutCore<12, 1>),
}

/// Runs `$body` against whichever arm is live. Every method below is `&mut self`, so one macro
/// covers all of them.
macro_rules! on {
    ($self:expr, $r:ident => $body:expr) => {
        match &mut $self.0 {
            Kind::N1H1($r) => $body,
            Kind::N6H1($r) => $body,
            Kind::N6H16($r) => $body,
            Kind::N7H1($r) => $body,
            Kind::N12H1($r) => $body,
        }
    };
}

#[allow(clippy::needless_pass_by_value)]
fn rollout_err(e: RolloutError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// What one [`Rollout::act`] hands back, as the tuple Python unpacks: the executed action, the
/// plane's event bits, the rewards and the dones, each `n_envs` long (`executed` `n_envs * nu`).
type ActTuple = (Vec<f64>, Vec<u32>, Vec<f64>, Vec<bool>);

// `Vec<f64>` / `Option<Vec<u32>>` by value is how a Python list crosses the boundary -- pyo3
// extracts into an owned collection, so there is no borrowed form to take instead.
#[allow(clippy::needless_pass_by_value)]
#[pymethods]
impl Rollout {
    #[new]
    fn new(
        task_toml: &str,
        observation_toml: &str,
        deployment_toml: &str,
        scene_xml: &str,
        seed: u64,
        n_envs: u32,
    ) -> PyResult<Self> {
        // Parsed here only to read `(n_joints, horizon)`; the core parses the document it
        // actually runs, so neither side is handed an IR the other one validated.
        let deploy = es_ir::serial::deployment_from_toml(deployment_toml)
            .map_err(|e| PyValueError::new_err(format!("deployment: {e}")))?;
        macro_rules! build {
            ($nj:literal, $h:literal, $v:ident) => {
                Kind::$v(
                    RolloutCore::<$nj, $h>::new(
                        task_toml,
                        observation_toml,
                        deployment_toml,
                        scene_xml,
                        seed,
                        n_envs,
                    )
                    .map_err(rollout_err)?,
                )
            };
        }
        let kind = match (deploy.robot.n_joints, deploy.action.horizon) {
            (1, 1) => build!(1, 1, N1H1),
            (6, 1) => build!(6, 1, N6H1),
            (6, 16) => build!(6, 16, N6H16),
            (7, 1) => build!(7, 1, N7H1),
            (12, 1) => build!(12, 1, N12H1),
            (nj, h) => {
                return Err(PyValueError::new_err(format!(
                    "unsupported (n_joints={nj}, horizon={h}); es_native.Rollout supports a \
                     fixed table of pairs (crates/es-py/src/pybind.rs) -- add one for this robot"
                )))
            }
        };
        Ok(Self(kind))
    }

    /// `reset(envs=None)`: `None` resets every env.
    #[pyo3(signature = (envs=None))]
    fn reset(&mut self, envs: Option<Vec<u32>>) -> PyResult<()> {
        on!(self, r => r.reset(envs.as_deref()).map_err(rollout_err))
    }

    /// `{port: [n_envs * dim]}` — the Observation IR's output ports, row-major by env.
    fn observe(&mut self) -> PyResult<BTreeMap<String, Vec<f64>>> {
        on!(self, r => r.observe().map_err(rollout_err))
    }

    /// `act([n_envs * nu]) -> (executed, events, rewards, dones)`.
    fn act(&mut self, actions: Vec<f64>) -> PyResult<ActTuple> {
        let a = on!(self, r => r.act(&actions).map_err(rollout_err))?;
        Ok((a.executed, a.events, a.rewards, a.dones))
    }

    /// The loaded model: `nq`, `nv`, `nu`, `n_envs`, actuator and joint names in index order,
    /// and each actuator's `ctrlrange` (`None` where the scene left it unlimited).
    fn model<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let (nq, nv, nu) = on!(self, r => r.dims());
        d.set_item("nq", nq)?;
        d.set_item("nv", nv)?;
        d.set_item("nu", nu)?;
        d.set_item("n_envs", on!(self, r => r.n_envs()))?;
        let actuators = on!(self, r => r.actuators());
        d.set_item(
            "actuators",
            actuators.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "ctrlrange",
            actuators
                .iter()
                .map(|(_, c)| c.map(|(lo, hi)| vec![lo, hi]))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("joints", on!(self, r => r.joints()))?;
        Ok(d)
    }

    /// Control ticks since construction — the plane's clock (packet M5/V17).
    fn tick(&mut self) -> u64 {
        on!(self, r => r.tick())
    }

    /// The nine spec 12.4 fields plus `chunk_underrun_rate`, `None` where nothing measured
    /// them. There is deliberately no single `step/s`.
    fn metrics(&mut self) -> BTreeMap<String, Option<f64>> {
        let m = on!(self, r => r.metrics());
        [
            ("physics_steps_per_sec", m.physics_steps_per_sec),
            ("camera_frames_per_sec", m.camera_frames_per_sec),
            ("pixels_per_sec", m.pixels_per_sec),
            ("observation_gb_per_sec", m.observation_gb_per_sec),
            ("policy_inferences_per_sec", m.policy_inferences_per_sec),
            ("actions_per_sec", m.actions_per_sec),
            ("p50_end_to_end_latency", m.p50_end_to_end_latency),
            ("p95_end_to_end_latency", m.p95_end_to_end_latency),
            ("gpu_memory_peak", m.gpu_memory_peak),
            ("chunk_underrun_rate", m.chunk_underrun_rate),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    /// One env's `qpos`, which is what the golden vector pins.
    fn qpos(&mut self, env: usize) -> Vec<f64> {
        on!(self, r => r.qpos(env))
    }
}

fn hex32(s: &str) -> PyResult<[u8; 32]> {
    if s.len() != 64 {
        return Err(PyValueError::new_err(
            "expected a 64-character hex task_hash",
        ));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| PyValueError::new_err("invalid hex in task_hash"))?;
    }
    Ok(out)
}

/// The `es_native` extension module (spec 14.2): what `import es_native` sees.
#[pymodule]
fn es_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Task>()?;
    m.add_class::<Observation>()?;
    m.add_class::<Learning>()?;
    m.add_class::<Deployment>()?;
    m.add_class::<Rollout>()?;
    m.add("EsDiagnosticError", m.py().get_type::<EsDiagnosticError>())?;
    Ok(())
}
