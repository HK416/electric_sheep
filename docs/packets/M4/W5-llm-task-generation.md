# W5 -- LLM task generation

Spec: spec 14.5 (LLM generation through the same validator; generate -> validate -> repair,
diagnostics fed back), spec 14.2 (builder vocabulary), spec 6 (Task IR), spec 7.4
(`ObservationSpec`), spec 0.5 (no evolutionary orchestrator).

Design note: `docs/design/llm-task-generation.md` (reviewed before this packet's code).

## context

```
crates/es-script/src/generate.rs       NEW -- Palette, prompt, parse, repair loop, provider
crates/es-script/src/lib.rs            `pub mod generate;`
crates/es-script/tests/generate.rs     NEW -- integration tests
crates/es-script/Cargo.toml            `llm` feature (ureq, optional, json+tls)
crates/es/src/cmd/generate.rs          NEW -- `es task generate` CLI
crates/es/src/cmd/mod.rs               `pub mod generate;`
crates/es/src/main.rs                  route `es task generate` before `cmd::task::dispatch`
crates/es/Cargo.toml                   `es-script` dependency, `es-script-llm` feature
crates/es/tests/cli.rs                 append-only: one `--provider stdin` CLI test
docs/design/llm-task-generation.md     NEW
docs/packets/M4/W5-llm-task-generation.md   this file
```

## spec

1. `TaskSpecPrompt { description, scene: Option<SceneRef>, constraints: BTreeMap<String,
   String> }` -> `build_prompt(&Palette, &TaskSpecPrompt) -> Prompt { system, user }`. `system`
   carries the node palette as JSON (re-derived from `es_ir::factory`'s two registries, since
   es-script (layer 11) cannot depend on `es-editor`'s `Palette::to_json`, layer 12), the
   `es_ir::serial` TOML envelope shape, and the boundary rules: declare `ObservationSpec` only,
   no preprocessing, no neural nets.
2. `parse_candidate(text) -> Result<(TaskIr, Option<ObservationIr>), GenerateError>` extracts
   the TOML block(s) from a reply, fenced or bare.
3. `repair_loop(&Prompt, provider: &mut dyn FnMut(&Prompt) -> Result<String, GenerateError>,
   max_rounds) -> GenerationReport { rounds: Vec<{ candidate_hash, diagnostics }>, result:
   Option<TaskIr> }` judges each round with `TaskIr::validate()` (+ `ObservationIr::validate()`
   when present, + a `GEN-001` no-Reward completeness check) and feeds the formatted
   diagnostics back into the next round's prompt verbatim.
4. `AnthropicProvider` (feature `llm`) implements the provider closure over the Anthropic
   Messages API via `ureq`, reading `ANTHROPIC_API_KEY` from the environment. `--provider
   stdin` needs no key and works with any external LLM.
5. CLI: `es task generate --prompt "..." [--scene scene.xml] [--rounds 3] [--provider
   anthropic|stdin] --out <dir>`, wired into `main.rs` ahead of the existing `es task compile`
   dispatch without modifying `cmd/task.rs`.

## oracle

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets --all-features -- -D warnings
cargo test -p es-script -p es
cargo xtask check-spec-refs
```

## acceptance

- A scripted provider that first returns a Task IR with no `Reward` node, then a valid one,
  makes `repair_loop` report exactly two rounds; round one's diagnostics contain `GEN-001` and
  round two's are empty; `result` is the second candidate.
- A scripted provider that never returns a valid candidate exhausts `max_rounds` and reports
  `result: None`, with exactly `max_rounds` rounds recorded.
- `parse_candidate` reads a Task IR from a ```` ```toml ```` fenced block, from a bare
  (unfenced) TOML reply, and reads a companion Observation IR when a second fenced block is
  present.
- `build_prompt`'s system turn contains every kind in `BUILTIN_TASK_KINDS` and
  `BUILTIN_LEARNING_KINDS`.
- `es task generate --provider stdin --prompt ... --out <dir>` fed a valid Task IR TOML reply
  on stdin exits 0 and writes `<dir>/task.toml`.
- The default `cargo build -p es-script -p es` (no `llm` / `es-script-llm` feature) does not
  link `ureq` or a TLS backend.

## forbidden

- `crates/es-script/src/mcp.rs`, `crates/es-script/src/tools.rs` -- a separate in-flight
  packet owns the MCP server.
- `crates/es-ir/**` -- no new validator codes there; `GEN-001` is a generation-time
  completeness check local to `es-script::generate`, not part of `TaskIr::validate()`.
- `crates/es/src/cmd/task.rs` -- `es task generate` is routed in `main.rs`, not added as a
  branch inside `cmd::task::dispatch`.
- `crates/es-editor/**` -- the palette is re-derived from `es_ir::factory`, not imported
  (layer 11 cannot depend on layer 12).
- Any new extension-point trait (`INV-17`); `Palette`, `Prompt`, `GenerationReport` etc. are
  plain structs, not `TaskNodeFactory`/`LearningNodeFactory` implementations.
- Committing (`SKIP_DOC_SYNC` is irrelevant here -- neither `ARCHITECTURE.md` file changes).
