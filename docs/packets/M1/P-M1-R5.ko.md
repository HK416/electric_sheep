<!-- Korean translation of docs/docs/packets/M1/P-M1-R5.ko.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R5 — 트랜스포트 핸드셰이크 강화(hardening) (`es-telemetry::transport`)

M1 리뷰 후속 조치(`docs/reviews/M1.md`, Should-fix): 인증 토큰이 가변 시간(variable-time)
`String` 동등 비교로 대조되고 있었고, 핸드셰이크 읽기에 타임아웃이 없어 침묵하는 피어(peer)가
서버 스레드를 영원히 붙잡아 둘 수 있었으며, 동시 연결 수에 상한이 없었다(수락된 TCP 연결마다
스레드 하나, 무제한). Spec: spec 25.1(보안: 토큰 인증, 신뢰할 수 없는 피어에 대한 자원 제한),
spec 25.3(버전 관리, 이 패킷으로 변경되지 않음). 설계 노트: `docs/design/telemetry-protocol.md`.

## context (범위)

```
crates/es-telemetry/src/transport.rs
docs/design/telemetry-protocol.md
docs/packets/M1/P-M1-R5.md
```

테스트는 `transport.rs` 자체의 `#[cfg(test)] mod tests` 안에 있다 — 새 테스트 파일은 없다.

## spec (사양)

- **상수 시간(constant-time) 토큰 비교.** `reject_reason`은 더 이상 `hello.token`을 설정된
  토큰과 `String`/`Option<String>`에 대한 `PartialEq`로 비교하지 않는다(가변 시간이며, 처음
  다른 바이트에서 조기 종료된다). 새로운 `ct_eq(a: &[u8], b: &[u8]) -> bool`은 조기 반환 없이
  `max(a.len(), b.len())`까지의 모든 바이트 위치를 하나의 `u8` 누산기로 접어 넣으며, 길이
  불일치도 `!=`로 비교하는 대신 미리 같은 누산기에 접어 넣는다. 새로운 의존성은 없다
  (`subtle`도 그 무엇도) — 함수 전체가 다섯 줄이다.
- **핸드셰이크 타임아웃.** `Server::bind_with(addr, token, cfg: ServerConfig)`는 새 생성자다;
  `Server::bind`는 시그니처가 변경되지 않았으며 이제 `Self::bind_with(addr, token,
  ServerConfig::default())`를 호출한다. `ServerConfig::handshake_timeout`(기본값
  `HANDSHAKE_TIMEOUT = Duration::from_secs(5)`)은 `serve_client`가 클라이언트의 `Hello`를
  읽기 전에 `TcpStream::set_read_timeout`으로 적용된다; `Hello`를 끝내 완료하지 못한 피어는
  타임아웃이 지나면 버려진다(스레드가 반환된다). 성공적인 `Hello` 읽기 직후 타임아웃은
  해제되어(`set_read_timeout(None)`), 오래 지속되는 인증된 세션의 이후 읽기(`Subscribe`
  루프)는 다시 정상적으로 블록된다.
- **연결 상한.** `ServerConfig::max_clients`(기본값 `DEFAULT_MAX_CLIENTS = 64`)는
  `serve_client`에서, 기존의 버전/토큰 검사 이후(그래서 잘못된 토큰은 여전히 그 자신만의
  구체적인 `Bye` 사유를 받는다) 그리고 클라이언트가 공유 맵에 등록되기 전에 검사된다: 서버가
  이미 `max_clients` 개의 클라이언트를 보유한 상태에서 도착한 연결은 `Bye { reason: "too many
  clients" }`를 받고 `clients`에 결코 추가되지 않는다.
- ponytail: 상한 검사와 등록 삽입은 하나의 원자적 단계가 아니므로, 정확히 용량 경계에서
  도착한 두 연결이 둘 다 등록 전에 검사를 통과하여 `max_clients`를 일시적으로 하나 초과할 수
  있다. 루프백/신뢰된 네트워크용 shim에는 받아들일 만하다(`docs/design/telemetry-protocol.md`
  §6/§8이 이 트랜스포트가 최종본이 아님을 이미 문서화하고 있다); 이것이 정확해야 한다면
  "검사와 삽입"에 걸쳐 하나의 뮤텍스를 유지하는 것이 해법이다.

## oracle (오라클)

```
cargo fmt -p es-telemetry --check
cargo clippy -p es-telemetry --all-targets -- -D warnings
cargo test -p es-telemetry
```

## acceptance (수용 기준)

- 연결한 뒤 아무것도 보내지 않는 클라이언트는 `handshake_timeout + 1s` 이내에 서버가 연결을
  닫는 것을(읽기가 EOF를 반환하는 것을) 관찰한다; 테스트는 빠르게 유지하기 위해
  `Server::bind_with`를 통해 짧게 설정된 타임아웃을 사용한다.
- 잘못된 토큰(`a_bad_token_is_rejected`, 변경 없음)과 설정된 것과 길이가 다른 토큰
  (`a_token_of_different_length_is_still_rejected`, 새로 추가 — `ct_eq`의 길이 불일치 경로를
  실행한다) 모두 `TransportError::Rejected`로 거부된다.
- 65번째 동시 클라이언트(기본값 `max_clients = 64`일 때)는 `"too many"`를 포함하는
  `TransportError::Rejected { reason }`로 거부되는 반면, 처음 64개는 모두 핸드셰이크에
  성공하며 검사가 이루어지는 동안 열린 채로 유지된다.
- 기존의 모든 `transport::tests` 케이스(토큰 없는 핸드셰이크, 토큰이 일치하는 핸드셰이크,
  지원되지 않는 버전, publish/subscribe 팬아웃, 느린 클라이언트 드롭 집계, close/reap)는
  변경 없이 그대로 통과한다.

## forbidden (금지)

- `crates/es-telemetry/src/protocol.rs`, `crates/es-telemetry/src/ring.rs`,
  `crates/es-telemetry/src/lib.rs`, `crates/es-runtime-embedded/*` — 다른 패킷들의 범위다;
  특히 `ServerConfig`는 `es_telemetry::transport::ServerConfig`로 접근 가능하며 의도적으로
  크레이트 루트에 재익스포트되지 않는다, 그것이 `lib.rs`를 건드리기 때문이다.
- `crates/es-compile/*`, `docs/design/policy-bundle.md` — P-M1-R4의 범위다.
- TLS, 상수 시간으로 비교되는 공유 토큰을 넘어서는 실질적인 인증, QUIC, 그리고 spec 28.7
  게이트 9의 `< 1%` 오버헤드가 이 패킷에 의해 *측정*된다는 주장 — 측정되지 않으며, `Target /
  Status: 미검증 (unverified)`로 남는다.
- 새로운 외부 의존성과 새로운 trait(INV-17).
- 커밋하는 것.
