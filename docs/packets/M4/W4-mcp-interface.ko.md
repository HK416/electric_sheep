<!-- Korean translation of docs/packets/M4/W4-mcp-interface.md. The English file is the working copy; regenerate this when it changes. -->

# W4 — MCP 인터페이스

Spec: spec 14.5 (약 1759번째 줄: "MCP 인터페이스로 `validate` / `compile` /
`estimate_cost` / `eval`을 노출한다" -- Electric Sheep을 외부 도구를 위한
백엔드로 노출하되, 우리만의 진화-루프 오케스트레이터는 없음), spec 11.1
(Cross-IR Check / Lower, `validate`/`compile`이 재사용함), spec 20 (메모리
예산, `estimate_cost`가 재사용함), spec 10.5 (`eval compare`/`eval run`
아티팩트, `eval`/`eval_run`이 재사용함), spec 25.1 (보안: localhost/stdio
전용, 여기에는 네트워크 리스너가 없음).

Design note: `docs/design/mcp-interface.md`. 고정된 프로토콜 다이제스트:
`docs/api-notes/mcp.md`.

## context (범위)

```
crates/es-script/src/lib.rs             `pub mod mcp; pub mod tools;`
crates/es-script/src/mcp.rs             신규 -- JSON-RPC 2.0 stdio Server
crates/es-script/src/tools.rs           신규 -- validate/compile/estimate_cost/eval/eval_run/
                                         hash_chain 도구 본체
crates/es-script/tests/mcp.rs           신규 -- 프로세스 내 요청-시퀀스 하네스
crates/es/src/cmd/mcp.rs                신규 -- `es mcp` 서브커맨드
crates/es/src/cmd/mod.rs                `pub mod mcp;`
crates/es/src/main.rs                   `Some("mcp") => cmd::mcp::dispatch(&args[1..])`
crates/es/tests/cli.rs                  추가됨 -- 파이프된 stdin으로 spawn된 `es mcp`
docs/api-notes/mcp.md                   신규
docs/design/mcp-interface.md            신규
docs/packets/M4/W4-mcp-interface.md     이 파일
```

`crates/es-script/src/generate.rs`(LLM 태스크 생성, spec 14.5의 나머지 절반)는
동시에 진행 중인 다른 패킷의 파일이며 여기서는 건드리지 않는다.

## spec (사양)

1. `es_script::mcp::Server`: `BufRead`로부터 개행으로 구분된 JSON-RPC 2.0
   요청을 읽고, `Write`에 개행으로 구분된 응답을 쓰며, 입력이 EOF에 도달할
   때까지 계속한다(MCP stdio transport -- 메시지는 개행으로 구분되며 내부에
   개행을 포함해서는 안 된다). JSON-RPC *알림*(`id` 없음)은 성공이든 아니든
   응답을 받지 않는다. 유효한 JSON이 아닌 줄은 패닉이 아니라 `-32700` parse-error
   응답을 받는다.
2. `initialize`는 `protocolVersion`(`"2025-06-18"`), `capabilities: {"tools":
   {}}`, `serverInfo`를 반환한다. `tools/list`는 여섯 개의 도구 스키마(`ToolDef`
   표, 도구당 JSON Schema `inputSchema`)를 반환한다. `tools/call { name,
   arguments }`는 `es_script::tools::{validate, compile, estimate_cost,
   eval_compare, eval_run, hash_chain}`으로 디스패치한다.
3. MCP 스펙 자체의 구분과 일치하는 두 개의 서로 다른 오류 경로: 형식이
   잘못된 `tools/call`에 대한 JSON-RPC 프로토콜 오류(`-32602 Invalid
   params`) -- 알 수 없는 도구 이름, 또는 `ToolError::BadParams`(빠졌거나
   타입이 틀린 필수 인자); `ToolError::Failed`에 대한 `isError: true` 도구
   결과 -- 인자는 파싱되었지만 그것이 지목한 것(잘못된 TOML, 파싱 불가능한
   report)이 동작하지 않았음. 어느 경로도 서버를 죽이지 않는다.
4. 도구 본체는 CLI가 이미 호출하는 것과 같은 라이브러리 함수(`es_ir::serial`,
   `es_compile::CpuPlan`, `es_compile::budget::MemoryBudget`)를 호출하고
   JSON을 반환할 뿐, 절대 `es` 바이너리로 shell out하지 않는다 --
   `es-script`(layer 11)는 어차피 `es`의 바이너리 crate(layer 12)에
   의존할 수 없다.
5. `eval_run`은 항상 `{"status": "SKIPPED", "reason": ...}`을 보고한다:
   실제 실행에는 `PhysicsBackend`와 `PolicyRuntime`이 필요한데(spec 9.6,
   spec 17.1), `es-script`는 둘 다 링크하지 않는다(그 `Cargo.toml` 의존성은
   `es-core`, `es-ir`, `es-ir-types`, `es-compile`, `es-eval`, `es-data`뿐이다)
   -- 백엔드를 쓸 수 없을 때의 `es eval run` 자체의 `SKIPPED`를 그대로
   따른다(spec 1.4: 이 프로세스가 만들어낼 수 없는 결과를 절대 흉내 내지
   않는다).
6. `es mcp`(spec 2.5)는 인자 없이, 네트워크 소켓 없이(spec 25.1) stdin/stdout
   위에서 `Server`를 실행한다.

## oracle (오라클)

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets -- -D warnings
cargo test -p es-script -p es
cargo xtask layering
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `initialize` -> `tools/list` -> `tools/call validate`(최소한의 Task IR에
  대해) -> `tools/call estimate_cost` -> 형식이 잘못된 `tools/call`(필수
  인자 누락)을, `Server::run`에 하나의 개행 구분 시퀀스로 먹이면, 순서대로
  다섯 개의 응답이 나온다: 세 개의 성공한 결과, 그리고 `-32602` 오류로서의
  형식 잘못된 호출 -- `crates/es-script/tests/mcp.rs`가 단언한다.
- 파이프된 stdin/stdout으로 서브프로세스로 spawn된 `es mcp`는 같은
  `initialize` / `tools/list` / `tools/call validate` 시퀀스에 답하며,
  stdin이 닫히면 0으로 종료한다 -- `crates/es/tests/cli.rs`가 단언한다.
- JSON으로 파싱에 실패하는 줄은 `-32700` 응답을 받으며 서버는 그 뒤로도
  요청을 계속 서빙한다(루프를 일찍 끝내지 않는다).
- 잘못된 TOML 문서에 대한 `validate`와 `compile { observation_toml, mode:
  "not-a-mode" }`는 절대 패닉하지 않는다: 전자는 도구-실행 오류
  (`isError: true`)이고, 후자는 `-32602` 프로토콜 오류다(빠졌거나 타입이
  틀린 `mode` 값은 컴파일러가 거부하도록 남겨지는 것이 아니라 인자
  검증에서 잡힌다).

## forbidden (금지)

- `crates/es-script/src/generate.rs` -- 동시에 진행 중인 다른 패킷의 파일.
- 새로운 확장-지점 trait(`INV-17`): 여기서는 필요 없다.
- 어떤 종류든 네트워크 리스너(spec 25.1): stdio 전용.
- `es-script`에서 `es` 바이너리로 shell out하는 것(어차피 layer 순서가 그
  의존을 금지한다; 도구 본체는 라이브러리 crate를 직접 호출한다).
- `HashMap`(이 crate는 `BTreeMap`만 사용하지만, 여기 있는 MCP 도구 표는
  둘 중 어느 것도 필요하지 않다).
- `crates/es-ir/**`, `crates/es-compile/**`, `crates/es-eval/**`,
  `crates/es-data/**`를 편집하는 것 -- 다른 패킷들이 소유한다; 이 패킷은
  그들의 공개 API를 호출할 뿐이다.
- `es-script`의 `Cargo.toml`에 이미 나열된 것(`es-core`, `es-ir`,
  `es-ir-types`, `es-compile`, `es-eval`, `es-data`, `serde`, `serde_json`,
  `blake3`, `thiserror`)을 넘어서는 새 외부 crate 의존성을 추가하는 것;
  IR-kind sniffing은 그것만을 위해 `toml` 의존성을 추가하는 대신
  `es_ir::serial` 자체의 `KindMismatch` 진단을 재사용한다.
