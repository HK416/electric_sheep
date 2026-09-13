# LLM task generation (`es-script::generate`, layer 11)

Spec: spec 14.5 (LLM generation goes through the same validator; the loop is generate ->
validate -> repair, diagnostics fed back), spec 14.2 (the Python builder's vocabulary is the
same node set), spec 6 (Task IR: no neural nets; IR-D only), spec 7.4 (`ObservationSpec` is a
declaration, not preprocessing), spec 5.1 (IR boundary rules), spec 0.5 (no evolutionary
orchestrator -- one candidate per round, not a population).

## Why the palette is re-derived, not shared

`crates/es-editor/src/model/palette.rs::Palette::to_json` already exports exactly this JSON,
but es-editor is layer 12 and es-script is layer 11 (spec 4.2): a dependency from es-script on
es-editor would invert the layering. `es_script::generate::Palette::from_builtins()` reads
`TaskNodeRegistry` / `LearningNodeRegistry` (`es_ir::factory`) directly and projects the same
`NodeSchema` fields (`kind`, `inputs`, `outputs`, `params`) to JSON, dropping only the editor's
menu `category` (an LLM needs the type contract, not a UI grouping). Both call sites derive
from the identical `NodeSchema`, so a new node kind, a renamed parameter, or a changed port
type shows up in both without either file being touched -- they can disagree on presentation
(the editor also has menu categories) but never on what a node takes. There is deliberately no
third file trying to keep the two in sync; `crates/es-script/tests/generate.rs`'s
`palette_lists_every_builtin_kind` pins the entry count and shape from this crate's own view.

## Shape

- `TaskSpecPrompt { description, scene: Option<SceneRef>, constraints: BTreeMap<String,
  String> }` -- what the caller wants.
- `build_prompt(&Palette, &TaskSpecPrompt) -> Prompt { system, user }` -- `system` is the full
  grammar: the palette JSON, the `es_ir::serial` TOML envelope shape (`es_schema`, `kind`,
  `[body]`), and the non-negotiable rules (declare `ObservationSpec` only, no preprocessing, no
  neural nets, one node per palette kind). `user` is the description plus scene and
  constraints.
- `parse_candidate(text) -> Result<(TaskIr, Option<ObservationIr>), GenerateError>` -- scans
  for ```` ``` ````-fenced blocks (language tag ignored), or falls back to treating the whole
  trimmed reply as one TOML document when there are no fences. Each block is tried as a Task IR
  first, then as an Observation IR; `es_ir::serial::{task,observation}_from_toml` already
  reject a mismatched envelope `kind`, so this needs no separate sniffing.
- `repair_loop(&Prompt, provider: &mut dyn FnMut(&Prompt) -> Result<String, GenerateError>,
  max_rounds) -> GenerationReport { rounds: Vec<{ candidate_hash, diagnostics }>, result:
  Option<TaskIr> }` -- each round calls `provider`, parses the reply, and judges it with
  `TaskIr::validate()` (the *same* validator `es ir validate` runs) plus one completeness check
  this crate adds, `GEN-001`: a task with no `Reward` node has nothing to optimize, a defect
  `validate()` itself does not catch since a `Reward`-free graph is still structurally sound.
  When an `ObservationIr` candidate is present its own `validate()` runs too. A failing round's
  formatted diagnostics (`Display` on `Diagnostic`, the same spec 5.4 block format `es ir
  validate` prints) are appended verbatim to the *original* prompt's user turn for the next
  round -- no accumulated transcript, so a stateless provider (one Messages API call, or a
  human typing at a prompt) is sufficient.

**Scope cut:** spec 11.1's full Cross-IR Check (`es_ir::cross::check`) also needs a Learning
IR and a Deployment IR, which this generator does not produce -- it targets Task IR (plus an
optional Observation IR), matching what `TaskSpecPrompt` asks for. Wiring an LLM-driven
Learning/Deployment generator through the same loop, and then running the full cross-check
once all four exist, is future work, not this packet's.

## Provider

`repair_loop`'s `provider` is a plain closure, `&mut dyn FnMut(&Prompt) -> Result<String,
GenerateError>`, so tests script a canned sequence of replies with a `Vec<String>` iterator and
the CLI can swap in any backend without `repair_loop` knowing about HTTP at all.

- `--provider stdin` (the CLI default) reads one full reply from stdin per round -- works with
  any external LLM, no API key. A pipe only supplies one reply; a round past a repair reads EOF
  and fails to parse, ending the loop at `--rounds` rather than hanging.
- `AnthropicProvider` (feature `llm` on `es-script`, `es-script-llm` on `es`) calls `POST
  https://api.anthropic.com/v1/messages` via `ureq` (`x-api-key`, `anthropic-version:
  2023-06-01`, body `{model, max_tokens, system, messages: [{role: "user", content}]}`), model
  id `claude-sonnet-5` (current Sonnet-tier model per the bundled `claude-api` skill's model
  table at the time this was written -- re-check that table if this ever looks stale). Reads
  `ANTHROPIC_API_KEY` from the environment only, never from a file in the repo. The feature is
  optional specifically so the default `es-script` / `es` build never links a TLS stack.
  `AnthropicProvider::with_endpoint` posts to a caller-chosen URL instead of the real API --
  the only reason it exists is so a test can point it at a local `TcpListener`.

**Timeouts and error hygiene (M4 review S-8):** the `ureq::Agent` behind `AnthropicProvider` is
built with both `timeout_connect` and `timeout_read` set to `AnthropicProvider::DEFAULT_TIMEOUT`
(60 s), overridable via `with_timeout` / `es task generate --timeout SECS` -- `ureq`'s own
default has *no* read timeout, so a stalled provider used to hang the round (and the whole CLI
invocation) forever. A timeout is reported as `GenerateError::Timeout`, kept distinct from the
catch-all `GenerateError::Provider(String)` so a caller does not have to string-match "timed
out" out of a message to react to it differently (e.g. retry vs. give up). Every other provider
failure's message is redacted (any literal occurrence of the API key is replaced with
`<redacted>`, in case a proxy or error page ever echoes a request header back) and capped at
256 bytes (`redact_and_truncate`) before being wrapped in `GenerateError::Provider` -- the old
`format!("unexpected response shape: {resp}")` printed an unbounded provider response body
verbatim, which both risked a key leak (had one ever appeared in a response) and could dump an
arbitrarily large blob into logs or stdout.

## CLI

`es task generate --prompt "..." [--scene scene.xml] [--rounds N] [--timeout SECS] [--provider
anthropic|stdin] --out <dir>` (`crates/es/src/cmd/generate.rs`, routed from `es task generate`
in `main.rs` before falling through to the existing `cmd::task::dispatch` for `es task
compile`). `--scene` hashes the named file with `blake3` for both `SceneRef::scene_hash` and
`asset_hash` -- a placeholder, not a real asset-import hash (that is `es-assets`' job); a
generation loop typically won't have the imported scene's real content hash yet. On success,
writes `task.toml` under `--out` via `es_ir::serial::task_to_toml`.

`--rounds` defaults to 3 and is checked against `1..=10` before any provider call: 0 rounds can
never produce a result and an unbounded round count times an unbounded per-round timeout was an
unbounded worst-case runtime for one invocation (M4 review S-8, `es/src/cmd/generate.rs:49`
used to accept any `u32`). Outside that range `dispatch` returns `CliError::Usage`, exit code 2
like any other bad argument. `--timeout` (default 60) is seconds, forwarded to
`AnthropicProvider::with_timeout`; `--provider stdin` ignores it, since it never makes a network
call.
