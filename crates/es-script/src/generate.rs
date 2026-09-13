//! LLM task generation (spec 14.5): generate -> validate -> repair, no evolutionary
//! orchestrator (spec 0.5).
//!
//! [`build_prompt`] hands the model the exact node palette an editor would offer (spec 14.5:
//! "an LLM generating a Task or Learning IR reads the palette rather than a prose list of
//! kinds"), the `es_ir::serial` TOML envelope, and the spec 5.1 boundary rules (Task IR only
//! *declares* `ObservationSpec`; no preprocessing; no neural nets). [`parse_candidate`] pulls
//! the TOML back out of a reply. [`repair_loop`] drives generate -> validate -> repair through
//! the *same* validator every other entry point uses (`TaskIr::validate` /
//! `ObservationIr::validate`), feeding diagnostics back verbatim until a candidate passes or
//! the round budget runs out.
//!
//! The palette here is re-derived from `es_ir::factory`'s registries rather than reused from
//! `crates/es-editor/src/model/palette.rs::Palette::to_json` — es-script is layer 11 and
//! es-editor is layer 12 (spec 4.2), so a dependency the other way is not allowed. Both are
//! thin JSON projections of the identical `NodeSchema` the registries hand back, so they can
//! never disagree about what a node takes even though the code is written twice; see the
//! `palette_lists_every_builtin_kind` test below and `docs/design/llm-task-generation.md`.
//!
//! The `provider` a caller passes to [`repair_loop`] is a plain closure
//! (`&mut dyn FnMut(&Prompt) -> Result<String, GenerateError>`) so tests can script one deploy
//! any external LLM without linking an HTTP client at all. [`AnthropicProvider`] (feature
//! `llm`) is one implementation of that closure shape, over `ureq`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use es_ir::factory::{
    LearningNodeRegistry, NodeSchema, ParamType, TaskNodeRegistry, BUILTIN_LEARNING_KINDS,
    BUILTIN_TASK_KINDS,
};
use es_ir::graph::Port;
use es_ir::observation::ObservationIr;
use es_ir::serial::{observation_from_toml, task_from_toml, SerialError, ES_SCHEMA_VERSION};
use es_ir::task::{SceneRef, TaskIr, TaskNode};
use es_ir::Diagnostic;

/// An unrecognized code; [`Diagnostic::new`] reports any code the dictionary does not know as
/// an error, which is exactly what a generation-time completeness check needs.
pub const GEN_NO_REWARD: &str = "GEN-001";

// --- palette (spec 14.5) ------------------------------------------------------------------

/// The node palette as JSON: every kind `es_ir::factory`'s two registries can build, with
/// ports, parameter types, and a value known to type-check for each. See the module doc for
/// why this is a re-derivation of `es-editor`'s `Palette::to_json` rather than a shared call.
#[derive(Debug)]
pub struct Palette {
    json: String,
}

impl Default for Palette {
    fn default() -> Self {
        Self::from_builtins()
    }
}

impl Palette {
    /// The spec 6.3 / spec 8.3 builtin node sets, no third-party factories.
    pub fn from_builtins() -> Self {
        let task = TaskNodeRegistry::with_builtins();
        let learning = LearningNodeRegistry::with_builtins();
        let mut entries = Vec::new();
        for kind in BUILTIN_TASK_KINDS {
            if let Some(schema) = task.schema(kind) {
                entries.push(schema_json("task", &schema));
            }
        }
        for kind in BUILTIN_LEARNING_KINDS {
            if let Some(schema) = learning.schema(kind) {
                entries.push(schema_json("learning", &schema));
            }
        }
        let json = serde_json::to_string_pretty(&serde_json::json!({
            "es_schema": ES_SCHEMA_VERSION,
            "entries": entries,
        }))
        .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
        Self { json }
    }

    pub fn to_json(&self) -> &str {
        &self.json
    }
}

fn schema_json(ir: &str, schema: &NodeSchema) -> serde_json::Value {
    serde_json::json!({
        "ir": ir,
        "kind": schema.kind,
        "inputs": ports_json(&schema.inputs),
        "outputs": ports_json(&schema.outputs),
        "params": schema.params.iter().map(|p| serde_json::json!({
            "name": p.name,
            "type": param_type_json(&p.ty),
            "required": p.required,
            "default": p.default.as_ref().and_then(|v| serde_json::to_value(v).ok()),
        })).collect::<Vec<_>>(),
    })
}

fn ports_json(ports: &[Port]) -> Vec<serde_json::Value> {
    ports
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "ty": serde_json::to_value(&p.ty).unwrap_or(serde_json::Value::Null),
            })
        })
        .collect()
}

fn param_type_json(ty: &ParamType) -> serde_json::Value {
    match ty {
        ParamType::Bool => serde_json::json!("bool"),
        ParamType::Int => serde_json::json!("int"),
        ParamType::Float => serde_json::json!("float"),
        ParamType::String => serde_json::json!("string"),
        ParamType::Shape => serde_json::json!("shape"),
        ParamType::PortType => serde_json::json!("port_type"),
        ParamType::Enum(variants) => serde_json::json!({ "enum": variants }),
    }
}

// --- prompt --------------------------------------------------------------------------------

/// What the caller wants generated. `constraints` is free-form name/value text (e.g.
/// `"max_episode_steps" -> "500"`) folded into the user turn.
#[derive(Clone, Debug, Default)]
pub struct TaskSpecPrompt {
    pub description: String,
    pub scene: Option<SceneRef>,
    pub constraints: BTreeMap<String, String>,
}

/// A system/user pair ready for any Messages-API-shaped provider.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}

/// Builds the generation prompt: the system turn is the IR grammar (palette + TOML envelope +
/// the spec 5.1 boundary rules); the user turn is the caller's request.
pub fn build_prompt(palette: &Palette, req: &TaskSpecPrompt) -> Prompt {
    let mut system = String::new();
    system.push_str(
        "You write Electric Sheep Task IR (spec section 6) as TOML. Rules, non-negotiable:\n\
         - Task IR only *declares* an ObservationSpec; it never implements preprocessing \
           (that is Observation IR's job).\n\
         - Task IR has no neural networks and no wall-clock reads; use only the node kinds \
           listed in the palette below, one per `kind` tag.\n\
         - Reply with exactly one fenced ```toml code block for the Task IR. If the task also \
           needs an Observation IR, add a second, separate fenced ```toml block for it.\n\
         - Every reward the episode optimizes and every termination condition must be a node \
           from the palette (Reward, Terminate) wired into the graph, not prose.\n\n",
    );
    system.push_str("Node palette (spec 14.5), the ONLY kinds you may use:\n");
    system.push_str(palette.to_json());
    system.push_str(
        "\n\nTOML envelope (es_ir::serial): every file is\n\
         es_schema = 1\n\
         kind = \"task\"  # or \"observation\" for the second block\n\
         [body]\n\
         ...\n\n\
         Task IR body: { schema_version, scene { path, scene_hash, asset_hash }, \
         graph { schema_version, nodes = { \"0\" = { <Kind> = { ..fields.. } }, \"1\" = ... }, \
         edges = [ { from = { node = 0, port = \"...\" }, to = { node = 1, port = \"...\" } } ], \
         inputs = [], outputs = [] }, observation_spec { channels = {} }, \
         config { max_episode_steps, control_rate_hz, deterministic, rng_streams = [] } }. \
         Node ids are small integers assigned by you, referenced as integers in edges.\n",
    );

    let mut user = req.description.clone();
    if let Some(scene) = &req.scene {
        let _ = write!(
            user,
            "\n\nScene: {} (scene_hash {}, asset_hash {})",
            scene.path,
            hex(&scene.scene_hash),
            hex(&scene.asset_hash)
        );
    }
    if !req.constraints.is_empty() {
        user.push_str("\n\nConstraints:\n");
        for (k, v) in &req.constraints {
            let _ = writeln!(user, "- {k}: {v}");
        }
    }
    Prompt { system, user }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

// --- parsing a reply -------------------------------------------------------------------------

/// A [`repair_loop`] round failure: the reply had no parseable Task IR, or the provider itself
/// failed (network error, missing key, non-2xx response).
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("no Task IR TOML block in the reply ({0})")]
    NoTaskCandidate(String),
    #[error("provider failed: {0}")]
    Provider(String),
}

/// Extracts a Task IR (and, if present, an Observation IR) from a model reply. Looks inside
/// ```` ``` ````-fenced blocks (language tag optional); if the reply has no fences at all, the
/// whole trimmed reply is tried as one TOML document.
pub fn parse_candidate(text: &str) -> Result<(TaskIr, Option<ObservationIr>), GenerateError> {
    let blocks = extract_code_blocks(text);
    let mut task = None;
    let mut observation = None;
    let mut last_err: Option<SerialError> = None;
    for block in &blocks {
        if task.is_none() {
            match task_from_toml(block) {
                Ok(t) => {
                    task = Some(t);
                    continue;
                }
                Err(e) => last_err = Some(e),
            }
        }
        if observation.is_none() {
            if let Ok(o) = observation_from_toml(block) {
                observation = Some(o);
            }
        }
    }
    match task {
        Some(t) => Ok((t, observation)),
        None => Err(GenerateError::NoTaskCandidate(last_err.map_or_else(
            || "the reply had no TOML block".to_owned(),
            |e| e.to_string(),
        ))),
    }
}

fn extract_code_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();
    let mut in_block = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if in_block {
                blocks.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            in_block = !in_block;
            continue;
        }
        if in_block {
            current.push_str(line);
            current.push('\n');
        }
    }
    if blocks.is_empty() {
        blocks.push(text.trim().to_owned());
    }
    blocks
}

// --- generate -> validate -> repair (spec 14.5) -----------------------------------------------

/// One round's outcome: the reply's content hash (so a caller can tell two rounds apart
/// without keeping the full text) and the diagnostics fed back for the next round, formatted
/// exactly as `TaskIr::validate` / `ObservationIr::validate` print them.
#[derive(Clone, Debug, Default)]
pub struct RoundReport {
    pub candidate_hash: [u8; 32],
    pub diagnostics: Vec<String>,
}

/// The full repair-loop trace plus the accepted Task IR, if any round validated clean.
#[derive(Clone, Debug, Default)]
pub struct GenerationReport {
    pub rounds: Vec<RoundReport>,
    pub result: Option<TaskIr>,
}

/// Drives generate -> validate -> repair for up to `max_rounds` rounds against `provider`,
/// starting from `initial`. Every round after the first repeats `initial`'s system and user
/// turns plus that round's own diagnostics — no accumulated back-and-forth transcript, so a
/// stateless provider (a single Messages API call, or a human at a prompt) is enough.
///
/// A round is judged by `TaskIr::validate()` plus `ObservationIr::validate()` when a candidate
/// includes one, plus one completeness check this crate adds (a task with no `Reward` node has
/// nothing to optimize, `GEN-001`) — full spec 11.1 Cross-IR Check needs Learning and
/// Deployment IR too, which this generator does not produce (see the design doc).
pub fn repair_loop(
    initial: &Prompt,
    provider: &mut dyn FnMut(&Prompt) -> Result<String, GenerateError>,
    max_rounds: u32,
) -> GenerationReport {
    let mut rounds = Vec::new();
    let mut result = None;
    let mut prompt = initial.clone();

    for _ in 0..max_rounds {
        let reply = match provider(&prompt) {
            Ok(r) => r,
            Err(e) => {
                rounds.push(RoundReport {
                    candidate_hash: [0; 32],
                    diagnostics: vec![e.to_string()],
                });
                break;
            }
        };
        let candidate_hash = *blake3::hash(reply.as_bytes()).as_bytes();

        match parse_candidate(&reply) {
            Ok((task, observation)) => {
                let diags = diagnose(&task, observation.as_ref());
                let diagnostics: Vec<String> = diags.iter().map(ToString::to_string).collect();
                let passed = !diags.iter().any(Diagnostic::is_error);
                rounds.push(RoundReport {
                    candidate_hash,
                    diagnostics: diagnostics.clone(),
                });
                if passed {
                    result = Some(task);
                    break;
                }
                prompt = with_feedback(initial, &diagnostics);
            }
            Err(e) => {
                let msg = e.to_string();
                rounds.push(RoundReport {
                    candidate_hash,
                    diagnostics: vec![msg.clone()],
                });
                prompt = with_feedback(initial, std::slice::from_ref(&msg));
            }
        }
    }

    GenerationReport { rounds, result }
}

/// `TaskIr::validate()`, plus `ObservationIr::validate()` for an optional companion candidate,
/// plus the `GEN-001` completeness check described on [`repair_loop`].
fn diagnose(task: &TaskIr, observation: Option<&ObservationIr>) -> Vec<Diagnostic> {
    let mut diags = task.validate();
    let has_reward = task
        .graph
        .nodes
        .values()
        .any(|n| matches!(n, TaskNode::Reward { .. }));
    if !has_reward {
        diags.push(Diagnostic::new(
            GEN_NO_REWARD,
            "the task graph declares no Reward node; the episode has nothing to optimize",
        ));
    }
    if let Some(obs) = observation {
        diags.extend(obs.validate());
    }
    diags
}

fn with_feedback(initial: &Prompt, diagnostics: &[String]) -> Prompt {
    let mut user = initial.user.clone();
    user.push_str(
        "\n\nYour previous reply did not validate. Fix every diagnostic below and reply again \
         with the corrected TOML block(s):\n\n",
    );
    for d in diagnostics {
        user.push_str(d);
        user.push('\n');
    }
    Prompt {
        system: initial.system.clone(),
        user,
    }
}

// --- Anthropic provider (feature `llm`) -------------------------------------------------------

/// [`repair_loop`]'s provider closure over the Anthropic Messages API
/// (`POST https://api.anthropic.com/v1/messages`, `anthropic-version: 2023-06-01`).
///
/// Reads `ANTHROPIC_API_KEY` from the environment only — never from a file in the repo.
#[cfg(feature = "llm")]
pub struct AnthropicProvider {
    api_key: String,
    model: String,
}

#[cfg(feature = "llm")]
impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

#[cfg(feature = "llm")]
impl AnthropicProvider {
    pub fn from_env() -> Result<Self, GenerateError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| GenerateError::Provider("ANTHROPIC_API_KEY is not set".to_owned()))?;
        Ok(Self {
            api_key,
            model: "claude-sonnet-5".to_owned(),
        })
    }

    /// Use as the `repair_loop` provider via `|p: &Prompt| provider.call(p)`.
    pub fn call(&mut self, prompt: &Prompt) -> Result<String, GenerateError> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 8192,
            "system": prompt.system,
            "messages": [{ "role": "user", "content": prompt.user }],
        });
        let resp: serde_json::Value = ureq::post("https://api.anthropic.com/v1/messages")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(body)
            .map_err(|e| GenerateError::Provider(e.to_string()))?
            .into_json()
            .map_err(|e| GenerateError::Provider(format!("bad JSON response: {e}")))?;
        resp["content"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| GenerateError::Provider(format!("unexpected response shape: {resp}")))
    }
}
