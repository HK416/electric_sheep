<!-- Korean translation of docs/packets/M4/W5-llm-task-generation.md. The English file is the working copy; regenerate this when it changes. -->

# W5 -- LLM 태스크 생성

Spec: spec 14.5 (LLM 생성도 같은 validator를 거친다; generate -> validate ->
repair, 진단이 피드백된다), spec 14.2 (빌더 어휘), spec 6 (Task IR), spec 7.4
(`ObservationSpec`), spec 0.5 (진화적 오케스트레이터 없음).

Design note: `docs/design/llm-task-generation.md` (이 패킷의 코드보다 먼저
검토됨).

## context (범위)

```
crates/es-script/src/generate.rs       신규 -- Palette, 프롬프트, 파싱, repair 루프, provider
crates/es-script/src/lib.rs            `pub mod generate;`
crates/es-script/tests/generate.rs     신규 -- 통합 테스트
crates/es-script/Cargo.toml            `llm` feature (ureq, 선택적, json+tls)
crates/es/src/cmd/generate.rs          신규 -- `es task generate` CLI
crates/es/src/cmd/mod.rs               `pub mod generate;`
crates/es/src/main.rs                  `cmd::task::dispatch`보다 먼저 `es task generate`를 라우팅
crates/es/Cargo.toml                   `es-script` 의존성, `es-script-llm` feature
crates/es/tests/cli.rs                 추가 전용: `--provider stdin` CLI 테스트 하나
docs/design/llm-task-generation.md     신규
docs/packets/M4/W5-llm-task-generation.md   이 파일
```

## spec (사양)

1. `TaskSpecPrompt { description, scene: Option<SceneRef>, constraints:
   BTreeMap<String, String> }` -> `build_prompt(&Palette, &TaskSpecPrompt) ->
   Prompt { system, user }`. `system`은 노드 팔레트를 JSON으로 싣는다
   (es-script(layer 11)는 `es-editor`의 `Palette::to_json`(layer 12)에
   의존할 수 없으므로 `es_ir::factory`의 두 레지스트리로부터 다시
   도출됨), `es_ir::serial` TOML 봉투 shape, 그리고 경계 규칙: `ObservationSpec`만
   선언, 전처리 없음, 신경망 없음.
2. `parse_candidate(text) -> Result<(TaskIr, Option<ObservationIr>),
   GenerateError>`는 응답으로부터 TOML 블록(들)을 추출한다, 펜스가
   있든 없든.
3. `repair_loop(&Prompt, provider: &mut dyn FnMut(&Prompt) ->
   Result<String, GenerateError>, max_rounds) -> GenerationReport { rounds:
   Vec<{ candidate_hash, diagnostics }>, result: Option<TaskIr> }`는 각
   라운드를 `TaskIr::validate()`(+ 있다면 `ObservationIr::validate()`, +
   `GEN-001` no-Reward 완전성 검사)로 판정하고, 형식화된 진단을 다음
   라운드의 프롬프트에 그대로 피드백한다.
4. `AnthropicProvider`(feature `llm`)는 `ureq`를 통해 Anthropic Messages
   API 위에서 provider 클로저를 구현하며, `ANTHROPIC_API_KEY`를 환경에서
   읽는다. `--provider stdin`은 키가 필요 없으며 어떤 외부 LLM과도
   동작한다.
5. CLI: `es task generate --prompt "..." [--scene scene.xml] [--rounds 3]
   [--provider anthropic|stdin] --out <dir>`, `cmd/task.rs`를 수정하지 않고
   기존 `es task compile` 디스패치보다 앞서 `main.rs`에 연결됨.

## oracle (오라클)

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets --all-features -- -D warnings
cargo test -p es-script -p es
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- 처음에는 `Reward` 노드가 없는 Task IR을, 그다음에는 유효한 것을 반환하도록
  스크립트된 provider는 `repair_loop`가 정확히 두 라운드를 보고하게 만든다;
  첫 라운드의 진단은 `GEN-001`을 담고 두 번째 라운드는 비어 있다; `result`는
  두 번째 후보다.
- 결코 유효한 후보를 반환하지 않도록 스크립트된 provider는 `max_rounds`를
  소진하고 `result: None`을 보고하며, 정확히 `max_rounds`개의 라운드가
  기록된다.
- `parse_candidate`는 ```` ```toml ```` 펜스 블록으로부터, 펜스 없는 맨
  TOML 응답으로부터 Task IR을 읽으며, 두 번째 펜스 블록이 있으면 동반하는
  Observation IR도 읽는다.
- `build_prompt`의 system 턴은 `BUILTIN_TASK_KINDS`와
  `BUILTIN_LEARNING_KINDS`의 모든 kind를 담는다.
- stdin으로 유효한 Task IR TOML 응답을 받은
  `es task generate --provider stdin --prompt ... --out <dir>`는 0으로
  종료하고 `<dir>/task.toml`을 쓴다.
- 기본 `cargo build -p es-script -p es`(`llm` / `es-script-llm` feature
  없이)는 `ureq`도 TLS 백엔드도 링크하지 않는다.

## forbidden (금지)

- `crates/es-script/src/mcp.rs`, `crates/es-script/src/tools.rs` -- 별도의
  진행 중인 패킷이 MCP 서버를 소유한다.
- `crates/es-ir/**` -- 그곳에는 새 validator 코드가 없다; `GEN-001`은
  `TaskIr::validate()`의 일부가 아니라 `es-script::generate`에 국한된
  생성-시점 완전성 검사다.
- `crates/es/src/cmd/task.rs` -- `es task generate`는 `cmd::task::dispatch`
  안의 분기로 추가되는 것이 아니라 `main.rs`에서 라우팅된다.
- `crates/es-editor/**` -- 팔레트는 임포트되는 것이 아니라 `es_ir::factory`로부터
  다시 도출된다(layer 11은 layer 12에 의존할 수 없다).
- 새로운 확장-지점 trait(`INV-17`); `Palette`, `Prompt`, `GenerationReport`
  등은 `TaskNodeFactory`/`LearningNodeFactory` 구현이 아니라 평범한
  구조체다.
- 커밋하는 것(`SKIP_DOC_SYNC`는 여기서는 무관하다 -- 어느 `ARCHITECTURE.md`
  파일도 바뀌지 않는다).
