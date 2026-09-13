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
/// failed (network error, missing key, non-2xx response, or a timeout).
#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("no Task IR TOML block in the reply ({0})")]
    NoTaskCandidate(String),
    #[error("provider failed: {0}")]
    Provider(String),
    /// The provider took longer than the configured connect/read timeout (S-8;
    /// [`AnthropicProvider::DEFAULT_TIMEOUT`] / `--timeout`). Kept distinct from `Provider` so a
    /// caller (or a test) can tell "the network is slow" apart from any other failure without
    /// string-matching the message.
    #[error("provider timed out")]
    Timeout,
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

/// S-8 default: `ureq::post` had no read timeout, so a stalled provider hung `es task generate`
/// forever. This bounds both the connect and the read side of one call; override via
/// [`AnthropicProvider::with_timeout`] / `es task generate --timeout`.
#[cfg(feature = "llm")]
pub const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// S-8: a provider error message never contains the API key (checked by redaction below, not
/// merely "the key isn't logged on the happy path"), and is capped at this many bytes so a
/// stalled or misbehaving provider can't blow up logs/output with its entire response body
/// (`unexpected response shape: {resp}` used to print it whole).
#[cfg(feature = "llm")]
const MAX_PROVIDER_ERROR_BYTES: usize = 256;

/// [`repair_loop`]'s provider closure over the Anthropic Messages API
/// (`POST https://api.anthropic.com/v1/messages`, `anthropic-version: 2023-06-01`).
///
/// Reads `ANTHROPIC_API_KEY` from the environment only — never from a file in the repo.
#[cfg(feature = "llm")]
pub struct AnthropicProvider {
    api_key: String,
    model: String,
    endpoint: String,
    agent: ureq::Agent,
}

#[cfg(feature = "llm")]
const ANTHROPIC_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";

#[cfg(feature = "llm")]
impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "llm")]
impl AnthropicProvider {
    /// Same as [`Self::with_timeout`] with [`DEFAULT_TIMEOUT`].
    pub fn from_env() -> Result<Self, GenerateError> {
        Self::with_timeout(DEFAULT_TIMEOUT)
    }

    /// `timeout` bounds both the TCP connect and each socket read of one call to the provider
    /// (S-8) — a stalled provider fails with [`GenerateError::Timeout`] instead of hanging.
    pub fn with_timeout(timeout: std::time::Duration) -> Result<Self, GenerateError> {
        Self::with_endpoint(ANTHROPIC_ENDPOINT, timeout)
    }

    /// Same as [`Self::with_timeout`] but posts to `endpoint` instead of the real Anthropic API.
    /// Exists so the S-8 timeout behavior can be exercised against a local `TcpListener` that
    /// never replies, without making a real network call in tests.
    pub fn with_endpoint(
        endpoint: impl Into<String>,
        timeout: std::time::Duration,
    ) -> Result<Self, GenerateError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| GenerateError::Provider("ANTHROPIC_API_KEY is not set".to_owned()))?;
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(timeout)
            .timeout_read(timeout)
            .build();
        Ok(Self {
            api_key,
            model: "claude-sonnet-5".to_owned(),
            endpoint: endpoint.into(),
            agent,
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
        let resp: serde_json::Value = self
            .agent
            .post(&self.endpoint)
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(body)
            .map_err(|e| self.provider_error(&e))?
            .into_json()
            .map_err(|e| self.provider_error_str(&format!("bad JSON response: {e}")))?;
        resp["content"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| self.provider_error_str(&format!("unexpected response shape: {resp}")))
    }

    /// Maps a `ureq` failure to [`GenerateError::Timeout`] when it was in fact a connect/read
    /// timeout (normalized by `ureq` to `io::ErrorKind::TimedOut`), else a truncated, key-redacted
    /// [`GenerateError::Provider`].
    fn provider_error(&self, err: &ureq::Error) -> GenerateError {
        use std::error::Error as _;
        let timed_out = err
            .source()
            .and_then(|s| s.downcast_ref::<std::io::Error>())
            .is_some_and(|io_err| io_err.kind() == std::io::ErrorKind::TimedOut);
        if timed_out {
            return GenerateError::Timeout;
        }
        self.provider_error_str(&err.to_string())
    }

    fn provider_error_str(&self, message: &str) -> GenerateError {
        GenerateError::Provider(redact_and_truncate(
            message,
            &self.api_key,
            MAX_PROVIDER_ERROR_BYTES,
        ))
    }
}

/// Replaces any occurrence of `secret` with `<redacted>`, then truncates to at most `max_bytes`
/// bytes at a UTF-8 char boundary.
#[cfg(feature = "llm")]
fn redact_and_truncate(message: &str, secret: &str, max_bytes: usize) -> String {
    let redacted = if secret.is_empty() {
        message.to_owned()
    } else {
        message.replace(secret, "<redacted>")
    };
    if redacted.len() <= max_bytes {
        return redacted;
    }
    let mut end = max_bytes;
    while end > 0 && !redacted.is_char_boundary(end) {
        end -= 1;
    }
    redacted[..end].to_owned()
}

#[cfg(all(test, feature = "llm"))]
mod llm_tests {
    use super::*;

    #[test]
    fn redact_and_truncate_removes_the_secret_and_caps_the_length() {
        let msg = format!(
            "provider said: key sk-secret-abc123 was rejected, body: {}",
            "x".repeat(500)
        );
        let out = redact_and_truncate(&msg, "sk-secret-abc123", MAX_PROVIDER_ERROR_BYTES);
        assert!(!out.contains("sk-secret-abc123"), "{out}");
        assert!(out.len() <= MAX_PROVIDER_ERROR_BYTES);
    }

    /// S-8 oracle: a provider that accepts a connection and never replies must fail with
    /// `GenerateError::Timeout`, not hang `es task generate` forever.
    #[test]
    fn a_stalled_provider_times_out_instead_of_hanging() {
        use std::net::TcpListener;
        use std::time::Duration;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local_addr");
        // Accepts the connection and then never writes a response. Not joined: the process
        // ending after this test reaps it, and joining would itself need a timeout.
        let _server = std::thread::spawn(move || {
            // Keep the accepted socket alive for the sleep -- dropping it immediately would
            // reset the connection and the client would see a connection error, not a timeout.
            if let Ok((stream, _)) = listener.accept() {
                std::thread::sleep(Duration::from_secs(5));
                drop(stream);
            }
        });

        std::env::set_var("ANTHROPIC_API_KEY", "test-key-must-not-leak");
        let mut provider = AnthropicProvider::with_endpoint(
            format!("http://{addr}/v1/messages"),
            Duration::from_secs(1),
        )
        .expect("ANTHROPIC_API_KEY is set above");
        let prompt = Prompt {
            system: "sys".to_owned(),
            user: "user".to_owned(),
        };
        let result = provider.call(&prompt);
        assert!(matches!(result, Err(GenerateError::Timeout)), "{result:?}");
    }
}
