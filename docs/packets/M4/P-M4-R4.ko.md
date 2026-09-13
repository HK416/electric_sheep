<!-- Korean translation of docs/packets/M4/P-M4-R4.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R4 — MCP 요청 상한과 generate provider 위생(hygiene)

Spec: 사양 14.5(MCP 인터페이스, LLM 생성), 사양 25.1(stdio 전용 전송이지만, 파이프
반대편의 프로세스는 신뢰할 수 없는 입력이다). `docs/reviews/M4.md`의 should-fix **S-3**과
**S-8**을 닫는다.

## context (범위)

```
crates/es-script/src/mcp.rs
crates/es-script/src/generate.rs
crates/es-script/tests/
crates/es/src/cmd/generate.rs
docs/api-notes/mcp.md
docs/design/mcp-interface.md
docs/design/llm-task-generation.md
docs/packets/M4/P-M4-R4.md
```

## spec (사양)

**S-3** (`es-script/src/mcp.rs:144`): `Server::run`은 `std::io::BufRead::lines()`로 요청을
읽었는데, 이는 길이 상한이 없고(거대한 `toml` 인자가 완전히 버퍼링되어 `String`으로
디코딩되고, `serde_json::Value`로 재인코딩된 뒤, TOML로 파싱되었다 -- 어떤 검사도 실행되기
전에 자기 크기의 여러 배가 할당된다), non-UTF-8 바이트 하나를 `io::Error`로 바꾸어
`line?`이 `run()` 밖으로 전파시켜 세션 전체를 끝내버린다 — 잘못된 형식의 요청이 결코
루프를 크래시시키지 않는다는 모듈 문서의 주장과 모순된다.

수정: `ServerConfig { max_request_bytes: usize }`(`Default` = `MAX_REQUEST_BYTES`, 16 MiB;
오버라이드하려면 `Server::with_config`). `Server::run`의 새 내부 리더(`read_raw_line`)는
`BufRead`에서 raw 바이트를 읽으며 스트리밍되는 동안 누적 길이를 추적한다; 누적 총합이
상한을 넘어서는 순간부터는 그 줄의 나머지 동안 버퍼에 추가하는 것을 멈추고(나머지는
버림), 그래서 메모리 사용량은 줄의 실제 크기가 아니라 상한으로 제한된다. 그 줄의 종료
개행(또는 EOF)에 도달하면:

- 상한 초과 -> `-32600 Invalid Request`, `id: null`, 줄은 버려지고 루프는 계속된다.
- 상한 이하 -> `std::str::from_utf8`로 디코딩된다; 디코딩 실패 -> `-32700 Parse error`,
  `id: null`, 루프는 계속된다; 성공 -> 기존 `handle_line` 경로(trim, 빈 줄 건너뛰기,
  `serde_json::from_str`, dispatch)는 그대로다.

**S-8** (`es-script/src/generate.rs:427`, `es/src/cmd/generate.rs:49,437`):
`AnthropicProvider`/`es task generate`에 있는 세 개의 별개 공백:

1. `ureq::post`는 기본 agent를 사용했는데, 이는 connect timeout은 있지만 **read timeout은
   없다** -- provider가 멈추면 라운드가, 따라서 CLI 호출 전체가 영원히 멈춘다.
2. `--rounds`는 범위 검사 없이 그냥 `u32`로 파싱되었다 -- `0`은 결코 결과를 낼 수 없고,
   한 번의 호출이 만들 수 있는 provider 호출 수(각각 그 자체로 시간상 무한, 1번 참조)에도
   상한이 없었다.
3. `format!("unexpected response shape: {resp}")`(그리고 `GenerateError::Provider`를
   생성하는 다른 두 곳)는 provider 응답/오류 텍스트 전체를 무제한, 무편집(unredacted)
   상태로 메시지에 넣었다.

수정:

- `AnthropicProvider`는 이제 `AgentBuilder::timeout_connect` + `timeout_read`를 통해
  자신만의 `ureq::Agent`를 만들며, 둘 다 `AnthropicProvider::DEFAULT_TIMEOUT`(60초)로
  설정되고, `with_timeout` / `with_endpoint`로 오버라이드할 수 있다(후자는 테스트가 실제
  API 대신 `TcpListener`를 클라이언트가 가리키게도 해준다). source가
  `io::Error(ErrorKind::TimedOut)`로 downcast되는 `ureq` 오류(`ureq`는 connect-timeout과
  read-timeout 실패를 모두 이걸로 정규화한다)는 새 `GenerateError::Timeout` variant로
  매핑되며, `GenerateError::Provider(String)`와는 구분되게 유지되어, 호출자가 "네트워크가
  느렸다"와 그 외 provider 실패를 구분하기 위해 메시지를 문자열 매칭할 필요가 없다.
- `es task generate --rounds`는 어떤 provider가 생성되기 전에 `dispatch`에서 `1..=10`에
  대해 검사된다; 그 범위 밖이면 `CliError::Usage`(exit 2)이며, 이 CLI의 다른 모든 잘못된
  인자와 일치한다. `--timeout SECS`(기본값 60)는 새로 추가되었으며
  `AnthropicProvider::with_timeout`으로 전달된다; `--provider stdin`은 이를 무시한다.
- 모든 `GenerateError::Provider` 메시지는 `redact_and_truncate`를 거친다: API 키가
  리터럴로 등장하는 모든 곳은 `<redacted>`로 대체되며(심층 방어 -- 키가 응답에 나타날
  것으로 예상되지는 않지만, 프록시나 오류 페이지가 요청 헤더를 그대로 되돌려 보낼 수
  있다), 그런 다음 결과는 UTF-8 문자 경계에서 256바이트(`MAX_PROVIDER_ERROR_BYTES`)로
  제한된다.

## oracle (오라클)

```
cargo fmt -p es-script -p es --check
cargo clippy -p es-script -p es --all-targets --all-features -- -D warnings
cargo test -p es-script
cargo test -p es -- generate mcp
```

`AnthropicProvider`와 그 테스트들은 `llm` feature 뒤에 있으므로(crate의 기존 TLS-free
기본 빌드 규칙에 따라 기본적으로 꺼져 있다), S-8 timeout 테스트에는 다음도 필요하다(별도로
실행, 위의 기본 게이트에는 포함되지 않음):

```
cargo test -p es-script --features llm
```

수정 전에 작성되었으며, 수정되지 않은 코드에서 실패함이 확인되었다:

- `crates/es-script/src/mcp.rs`의 `mod tests`:
  - `oversized_line_is_invalid_request_and_the_session_stays_alive` -- 64 MiB짜리
    줄(축소된 상한이 아니라 실제 `MAX_REQUEST_BYTES` 기본값)은 `-32600`을 받고, 다음
    줄의 `initialize`는 여전히 응답을 받는다.
  - `oversized_line_with_a_small_configured_cap_is_invalid_request` -- 작은
    `ServerConfig::max_request_bytes`에서 동일한 형태로, 테스트 실행마다 수십
    메가바이트를 할당하지 않고도 "버려지고, 루프는 살아 있음" 동작을 고정한다.
  - `invalid_utf8_byte_is_a_parse_error_and_the_session_stays_alive` -- 단독 `0xFF`
    바이트는 `-32700`을 받고, 다음 줄의 `ping`은 여전히 응답을 받는다.
- `crates/es-script/src/generate.rs`의 `mod llm_tests`(feature `llm`):
  - `a_stalled_provider_times_out_instead_of_hanging` -- 연결을 accept하고 응답을 절대
    쓰지 않는 `std::net::TcpListener`(accept된 소켓은 sleep 동안 살려 둔다, 즉시
    drop하면 timeout 대신 연결이 리셋되기 때문이다); `AnthropicProvider::with_endpoint`가
    1초 timeout으로 이를 가리킨다; `Err(GenerateError::Timeout)`을 단언한다.
  - `redact_and_truncate_removes_the_secret_and_caps_the_length`.
- `crates/es/src/cmd/generate.rs`의 `mod tests`:
  - `rounds_outside_one_to_ten_is_a_usage_error` -- `--rounds 0`과 `--rounds 11` 둘 다
    provider를 건드리지 않고 `Err(CliError::Usage(_))`(`main` 레벨에서 exit 2)를
    반환한다.
  - `rounds_at_the_boundary_passes_the_usage_check` -- `--rounds 1`과 `--rounds 10`은
    rounds 검사 자체에 의해 거부되지 않는다(알 수 없는 `--provider` 이름을 사용해서
    `dispatch`가 여전히 `Usage`를 반환하도록 하지만, 이번에는 provider-name 분기에서다 —
    rounds 검사가 stdin이나 네트워크 근처에도 가지 않고 이들을 통과시켰음을 확인해준다).

이 CLI 레벨 검사들은 `crates/es/tests/cli.rs`에서 `es` 바이너리를 스폰하는 통합 테스트가
아니라, `crates/es/src/cmd/generate.rs` 안에서 `dispatch()`를 직접 호출하는 유닛
테스트다 -- 그 파일은 이 패킷이 선언한 `context` 밖에 있으며, 이 작업이 진행되는 동안
다른 패킷의 작업이 동시에 편집하고 있었다.

## acceptance (수용 기준)

- 위의 두 오라클 명령 모두 클린하다; `cargo test -p es-script --features llm`이
  통과한다.
- `es-script`는 컴파일되며 기존 테스트 스위트는 `--features llm`이 있든 없든 모두
  통과한다(기본 빌드는 이 패킷 이전과 마찬가지로 TLS-free 상태를 유지한다).
- 기존 함수의 공개 시그니처는 하나도 바뀌지 않았다; `Server::new()`, `Server::run`,
  `AnthropicProvider::from_env`, `AnthropicProvider::call`은 모두 옛 시그니처를
  유지한다. 새로 추가된 공개 표면은 오직: `ServerConfig`, `MAX_REQUEST_BYTES`,
  `Server::with_config`, `GenerateError::Timeout`,
  `AnthropicProvider::{DEFAULT_TIMEOUT, with_timeout, with_endpoint}`,
  `es task generate --timeout`.
- `docs/api-notes/mcp.md`, `docs/design/mcp-interface.md`,
  `docs/design/llm-task-generation.md`는 새 상한/timeout/truncation 동작을 기술한다;
  `docs/ARCHITECTURE.*` 변경은 필요 없다(여기서는 이 crate가 사용하는 표에 이미 예약된
  JSON-RPC 코드 두 개를 추가하는 것 이상으로 고정된 타입 시그니처, 오류 코드 목록,
  불변식을 바꾸는 것이 없다).

## forbidden (금지)

`context` 밖의 모든 것 -- 그 외 모든 M4 리뷰 발견 사항(S-1, S-2, S-4부터 S-7까지, S-9,
S-10)은 각자의 패킷에 속한다; 특히 `es/src/cmd/import.rs`(S-9)와
`crates/es/tests/cli.rs`(공유 통합 테스트 파일, 진행 중인 다른 패킷들이 활발히 편집
중)는 여기서 건드리지 않는다. `SafetyPlane::validate`의 시그니처를 바꾸는 것(이 패킷에는
해당 없음, `AGENTS.md`에 따라 재확인). MCP 도구 집합, `tools/list` 형태, 또는 기존 여섯
개 도구 스키마를 바꾸는 것. agent 레벨 timeout을 넘어서는 `AnthropicProvider`의 wire
포맷(모델, 헤더, 요청/응답 바디 형태)을 바꾸는 것. `--rounds` 상한을 올리거나 없애는
것, 또는 `MAX_REQUEST_BYTES`를 무제한으로 만드는 것.
