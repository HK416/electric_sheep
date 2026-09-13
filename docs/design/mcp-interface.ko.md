<!-- Korean translation of docs/design/mcp-interface.md. The English file is the working copy; regenerate this when it changes. -->

# MCP 인터페이스 (spec 14.5)

Spec 14.5: 기존 도구들(LLM 태스크-생성 루프, 에디터, 진화적 탐색 하네스)이 Electric
Sheep 자체가 오케스트레이터를 키우지 않고도 Electric Sheep을 백엔드로 쓸 수 있어야
한다(spec 0.5 -- 여기서 번호로 인용하지 않는 것은 이것이 이 문서가 고정하는 절이 아니라
설계 원칙이기 때문이다). 이것이 이름 붙이는 인터페이스는 의도적으로 작다: `validate`,
`compile`, `estimate_cost`, `eval`.

## Shape

`es_script::mcp::Server`는 stdio 위에서 JSON-RPC 2.0을, 개행으로 구분해(한 줄에 메시지
하나, 내부에 개행 없음 -- MCP stdio transport 자체의 프레이밍 규칙,
`docs/api-notes/mcp.md`) 말한다. 메서드는 세 개: `initialize`, `tools/list`,
`tools/call`. resource도, prompt도, sampling도,
`notifications/tools/list_changed`도 없다 -- 여기서 도구 집합은 컴파일 타임에
고정되므로 알릴 것이 없다.

## 네 개가 아니라 여섯 개의 도구

Spec 14.5는 네 개(`validate` / `compile` / `estimate_cost` / `eval`)를 이름 붙인다;
이 패킷은 여섯 개를 노출하며, `eval`을 둘로 나눈다:

- `eval` -- 이미 만들어진 두 `EvaluationReport` JSON 문서를 비교한다(spec 10.5
  `es eval compare`). 순수하게 데이터가 들어가서 순수하게 데이터가 나온다: 백엔드가
  필요 없으므로, 이것이 spec 14.5의 문자 그대로의 `eval`이다.
- `eval_run` -- Evaluation IR을 policy 번들에 대해 실행하는 것에 해당한다(spec 10.5
  `es eval run`). 항상 `{"status": "SKIPPED", "reason": ...}`를 보고한다: 실제 실행에는
  `PhysicsBackend`와 `PolicyRuntime`이 필요한데(spec 9.6, spec 17.1),
  `es-script`(layer 11)는 둘 다 링크하지 않는다 -- 여기 추가한다면 IR/report JSON을
  변환하는 것이 유일한 일인 crate에 `es-physics-backend`와 `es-policy`를 끌어들이게
  되고, 취소도 진행 상황 보고도 내장되어 있지 않은 프로토콜 위에서 이 서버가 오래
  걸리고 자원을 많이 쓰는 작업을 수행할 수 있게 만들어버릴 것이다(이 패킷의 범위
  밖; spec 1.4도 이 프로세스가 만들어낼 수 없는 결과를 흉내 내는 것을 금지한다).
  `hash_chain`이 여섯 번째다: spec 14.5는 이를 이름 붙이지 않지만, 다른 다섯 도구의
  출력 각각이 이미 `*_hash`를 싣고 있으며, 실행을 비교하는 외부 호출자는 직접
  재도출하지 않고도 spec 5.3 해시 체인 슬롯이 필요하다.

## `es`로 shell out하지 않는 이유

`crates/es/src/cmd/{ir,task,bench,eval}.rs`는 이미 `validate`/`compile`/
`estimate_cost`/`eval compare`를 표를 출력하는 CLI 명령으로 구현하고 있다.
`es-script`는 `es` 바이너리 crate에 의존할 수 없으므로(layer 11은 layer 12 아래다),
`crates/es-script/src/tools.rs`는 그 명령들이 호출하는 것과 같은 라이브러리 함수
(`es_ir::serial`, `es_compile::CpuPlan`, `es_compile::budget::MemoryBudget`,
`es_ir::evaluation::EvaluationReport`)를 직접 호출하고, 출력된 표 대신 JSON을
반환한다. 서브프로세스로 shell out하는 방안도 검토했지만 기각했다: `PATH`에 `es`
바이너리가 필요해지고(라이브러리 crate로부터 MCP 클라이언트가 예상하지 못할 숨은
런타임 의존성), 호출당 프로세스 수가 두 배가 되며, 라이브러리 함수가 애초에 피하려는
바로 그 파싱 문제를 다시 끌어들이게 된다.

## 오류 처리

MCP 스펙 자체의 "Protocol Errors"와 "Tool Execution Errors" 구분에 맞춘 두 경로:

- `ToolError::BadParams` -> JSON-RPC `-32602 Invalid params`. `tools/call` 인자
  자체의 shape가 잘못됨: 알 수 없는 도구 이름, 빠진 필수 필드, enum 범위 밖의 값
  (`mode`가 `"debug"`/`"release"`가 아님, `precision`이 `"f32"`/`"f16"`이 아님).
- `ToolError::Failed` -> `result.isError`가 `true`인 성공적인 JSON-RPC 응답. 인자는
  파싱되었지만 그것이 지목한 것이 동작하지 않음: 파싱에 실패하는 TOML, 역직렬화에
  실패하는 `EvaluationReport` JSON, 컴파일에 실패하는 Observation IR.

둘 다 패닉하지 않는다. 아예 JSON이 아닌 줄은 `id: null`과 함께(연결할 요청이 없는
오류에 대한 JSON-RPC 2.0 자체의 규칙) `-32700 Parse error`를 받으며, 서버는 다음
줄을 계속 읽는다.

## Ceiling

- `tools/list`에 페이지네이션이 없다(`cursor`/`nextCursor`): 도구 여섯 개는 응답
  하나에 들어간다. 도구 개수가 클라이언트가 한 번의 호출로 합리적으로 원하는 것을
  넘어서면 추가할 것.
- 도구마다 `outputSchema`가 없고 `inputSchema`만 있다: 모든 도구의 JSON 결과 shape는
  이 파일과 `crates/es-script/src/tools.rs`의 문서 주석에 문서화되어 있을 뿐,
  스키마에 대해 기계적으로 검사되지 않는다. 클라이언트가 소스를 읽지 않고 결과를
  검증해야 한다면 `outputSchema`를 추가할 것.
- `eval`의 비교 표에는 유의성 검정(CLI의 Welch-t-검정, spec 10.5)이 없다: 중복
  구현은 `es-script`가 그것에 의존할 수 없는 `es` 안에 살아야 할 것이고, spec
  14.5는 "compare 표 JSON"만 요구한다. MCP 호출자가 언젠가 이것이 필요해지면 `es`와
  `es-script` 둘 다 호출할 수 있는 라이브러리 crate로 그 검정을 옮길 것.
