<!-- Korean translation of docs/api-notes/mcp.md. The English file is the working copy; regenerate this when it changes. -->

# Model Context Protocol -- `es_script::mcp`를 위한 고정된 표면

`crates/es-script/src/mcp.rs`가 무엇에 맞춰 구현되었는지를 담아, 프로토콜 개정이 고고학적
탐사가 아니라 이 파일에 대한 diff가 되도록 한다. MCP SDK crate는 사용하지 않는다 (이 crate는
layer 11이며 `serde_json` 위에서 JSON-RPC 프레이밍을 직접 구현한다) -- `docs/api-notes/torch.md`,
`docs/api-notes/mujoco.md`와 같은 형태, 같은 이유의 동반 문서다.

Spec: spec 14.5 (약 1759번째 줄, MCP 인터페이스 요구사항).

## 버전

| | |
|---|---|
| 검증 대상 | 프로토콜 버전 **`2025-06-18`**, 2026-09-13에 <https://modelcontextprotocol.io/specification/2025-06-18/basic/transports>와 <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>에서 가져옴 |
| MCP SDK crate | **없음** -- 작업 패킷에 따라 `serde_json` 위에서 JSON-RPC 2.0을 직접 구현 |
| 실제 클라이언트와의 상호운용 | **미검증** -- 작성된 스펙 페이지에 대해서만 확인했을 뿐, 실제 MCP 클라이언트(Claude Desktop, IDE의 MCP 통합, `@modelcontextprotocol/inspector`)에 대해서는 한 번도 확인하지 않았다; spec 12.4의 관례에 따라 측정되지 않은 것은 `Target / Status: 미검증` |

## Transport -- stdio 프레이밍

스펙의 "Transports" 페이지에서 확인한 내용:

- JSON-RPC 메시지는 **반드시** UTF-8로 인코딩되어야 한다.
- stdio: 서버는 stdin에서 JSON-RPC 메시지를 읽고 stdout에 쓴다.
- **메시지는 개행으로 구분되며, 내부에 개행을 포함해서는 안 된다.** `Server::run`에서 유일하게
  중요한 프레이밍 규칙이다: 한 줄을 읽고, JSON-RPC 메시지 하나를 파싱하고, 한 줄을 다시 쓴다.
  stdio에는 `Content-Length:` 헤더 프레이밍이 없다 -- 그것은 Streamable HTTP의 문제이며 여기서는
  구현되지 않는다 (spec 25.1: 이 서버는 stdio 전용이며, 네트워크 리스너가 되는 일은 없다).
- 서버는 로깅을 위해 stderr에 쓸 수 있다(MAY); `es_script::mcp`는 아직 이를 하지 않는다 (여기서는
  그것이 필요한 곳이 없다).
- 서버는 유효한 MCP 메시지가 아닌 어떤 것도 stdout에 써서는 안 된다.

## `initialize`

요청은 `protocolVersion`, `capabilities`, `clientInfo`를 싣는다 (이 서버는 이를 별도로
확인하지 않는다 -- `Server::dispatch`는 `initialize`에 대해 `params`를 완전히 무시하는데, v1
구현은 클라이언트가 무엇을 요청하든 정확히 하나의 동작만 하기 때문이다).

응답 (이 서버의 정확한 형태, `crates/es-script/src/mcp.rs`):

```json
{
  "protocolVersion": "2025-06-18",
  "capabilities": {"tools": {}},
  "serverInfo": {"name": "electric-sheep", "version": "<CARGO_PKG_VERSION>"}
}
```

`capabilities.tools`는 `{"listChanged": true}`가 아니라 빈 객체다: 도구 집합은 런타임에
절대 바뀌지 않으므로 `notifications/tools/list_changed`는 절대 보내지지 않으며,
`listChanged`는 결코 오지 않을 알림을 약속하는 대신 설정되지 않은 채(falsy) 남는다.

## `tools/list`

페이지네이션 없음(`cursor`/`nextCursor`) -- 스펙의 "Listing Tools" 절에서 확인한 대로, 도구
여섯 개에 응답 하나. 각 도구: `name`, `description`, `inputSchema` (JSON Schema,
`properties`/`required`를 갖는 `type: "object"`). `outputSchema`도, `title`도,
`annotations`도 없음 -- 스펙상 선택적이며 여기서는 만들어지지 않는다 (`docs/design/mcp-interface.md`의
"Ceiling" 참고).

## `tools/call`

요청: `{"name": "<tool>", "arguments": {...}}`. 성공 시 응답:

```json
{"content": [{"type": "text", "text": "<JSON-encoded tool result>"}], "isError": false}
```

`structuredContent`(스펙의 대안, 기계-타입 결과 필드)는 만들어지지 않는다 -- `content[0].text`가
전체 JSON 결과를 문자열로 담고, 호출자가 이를 디코드한다. 도구 실행 실패 시: 같은 형태에
`isError: true`이고 `text`에는 오류 메시지가 담긴다.

스펙의 "Error Handling" 절에서 확인한 오류 코드 구분: 알 수 없는 도구 / 잘못된 인자는
**프로토콜 오류**다 (JSON-RPC `error` 객체, 이 서버는 항상 `-32602 Invalid params`를 사용);
실행되었지만 실패한 도구는 정상적인 `result` 안에 `isError: true`를 보고하며, JSON-RPC
오류가 되는 일은 없다. `es_script::tools::ToolError::{BadParams, Failed}`가 정확히 이
구분에 대응한다.

## 실제로 사용되는 JSON-RPC 오류 코드

| 코드 | 의미 | `es_script::mcp`가 보내는 상황 |
|---|---|---|
| `-32700` | Parse error | stdio 한 줄이 유효한 JSON이 아닐 때 |
| `-32601` | Method not found | `dispatch`의 메서드가 `initialize`/`ping`/`tools/list`/`tools/call` 중 어느 것도 아닐 때 |
| `-32602` | Invalid params | 알 수 없는 도구 이름, 또는 도구 본체에서 온 `ToolError::BadParams` |

이 세 가지는 MCP 전용이 아니라 표준 JSON-RPC 2.0의 예약된 코드다; MCP 자체는 "Error
Handling" 절이 프로토콜 오류로 표시하는 것 이상의 추가 코드를 정의하지 않는다.
