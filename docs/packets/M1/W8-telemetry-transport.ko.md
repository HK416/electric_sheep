<!-- Korean translation of docs/packets/M1/W8-telemetry-transport.md. The English file is the working copy; regenerate this when it changes. -->

# W8 — 텔레메트리 트랜스포트 (`es-core::ring`, `es-telemetry::transport`)

Spec: §23.1–23.2 (에디터는 실행 중인 프로세스의 클라이언트다; 백엔드 중립 프로토콜), §25.1
(보안: 토큰 인증, TLS는 이후, 로컬호스트 기본 바인드), §25.3 (API 버전 관리: 핸드셰이크
협상, N-1 지원), §28.7 게이트 9 (텔레메트리 + 그래프 뷰 오버헤드 < 1%, M1 — 단위 테스트로는
검증 불가능하며 아래에서 `Target / Status: 미검증 (unverified)`로 보고). 설계 노트:
`docs/design/telemetry-protocol.md` (spec 28.3의 M1 선행 조건 문서).

## context (범위)

```
crates/es-core/src/ring.rs
crates/es-core/src/lib.rs
crates/es-telemetry/src/ring.rs
crates/es-telemetry/src/transport.rs
crates/es-telemetry/src/lib.rs
crates/es-telemetry/Cargo.toml
crates/es-runtime-embedded/src/lib.rs
crates/es-runtime-embedded/src/runtime.rs
docs/design/telemetry-protocol.md
docs/packets/M1/W8-telemetry-transport.md
```

(`crates/es-runtime-embedded/src/ring.rs`는 이 패킷이 삭제하는 대상이지, 편집 대상이 아니다.)

## spec (사양)

- **링 버퍼는 복제가 아니라 이동한다.** `RingBuffer<T>`(시퀀스 번호, `push`, `iter_newest`,
  `drain_since`, `dropped`)는 그대로 `es_core::ring`(layer 1)로 이동한다. 이곳이
  `es-telemetry`(layer 10)와 `es-runtime-embedded`(layer 9) 양쪽 모두 서로의 계층 경계를
  넘지 않고 도달할 수 있는 유일한 지점이다(spec 4.2). `es_telemetry::ring`은
  `pub use es_core::ring::RingBuffer;`가 되므로 `es_telemetry::ring::RingBuffer`의 기존
  호출자는 아무것도 바뀌지 않는다. `es-runtime-embedded`의 private `TelemetryRing` 사본과
  그 중복에 대한 `ponytail:` 노트는 삭제된다; `EmbeddedRuntime`은 이제
  `RingBuffer<TickRecord>`를 보유한다(`TickRecord` 타입 자체는 그 크레이트의 틱 루프에
  특화되어 있으므로 `es-runtime-embedded`에 남는다).
- **`es_telemetry::transport::Server`** — `bind(addr: SocketAddr, token: Option<String>) ->
  io::Result<Server>`는 `std::net::TcpListener`를 열고 백그라운드 스레드에서 accept 루프를
  실행한다; 접속이 승인된 클라이언트마다 자신의 reader/writer 스레드 쌍을 가진다.
  `publish(&self, frame: Frame)`은 `frame.stream`을 구독 중인 모든 클라이언트에게 프레임을
  팬아웃한다. 유한한(`CLIENT_QUEUE_CAPACITY = 16`) 클라이언트별 채널과 논블로킹 `try_send`를
  사용한다 — 큐가 가득 차면 프레임을 드롭하고 집계할 뿐, 프로듀서는 결코 대기하지 않는다.
  `stats() -> ServerStats`는 세션 id로 키가 매겨진 클라이언트별 `{ sent, dropped }`를
  보고한다.
- **`es_telemetry::transport::Client`** — `connect(addr, token, client_name) ->
  Result<Client, TransportError>`는 `Hello`/`HelloAck` 핸드셰이크를 수행하거나(또는
  `Bye { reason }`를 읽고 `TransportError::Rejected`를 반환), `subscribe(streams)`는 새
  구독 집합을 보내고, `recv()`는 다음 메시지를 기다리며 블록하고, `try_recv()`는 블록하는
  대신 `TransportError::WouldBlock`을 반환하며, `close()`는 `Bye`를 보내고 세션을
  종료한다.
- **핸드셰이크와 인증 (spec 25.1, 25.3).** 클라이언트는 다른 무엇보다 먼저 `Hello`를
  보내야 한다. 서버는 다음의 경우 거부한다(`Bye { reason }`를 보낸 뒤 연결을 닫는다):
  클라이언트의 `versions_supported`가 서버 자신의 `{PROTOCOL_VERSION,
  PROTOCOL_VERSION - 1}`(N-1, spec 25.3의 `negotiate` — 기존 그대로 재사용)과 아무것도
  공유하지 않을 때; 또는 서버가 `token`을 가지고 있고 클라이언트의 `Hello.token`이 정확히
  일치하지 않을 때. 성공하면 서버는 `HelloAck { version, session_id,
  execution_hash: None }`을 회신한다(이 계층은 해시 체인을 가지고 있지 않으므로 이 필드는
  비어서 시작한다; 해시 체인을 가진 호출자가 이를 감쌀 수 있다). `127.0.0.1:0`으로
  바인딩하는 것이 권장 기본값이다(spec 25.1의 로컬호스트 기본값) — 함수 자체가 이를
  하드코딩하지는 않는데, 호출자가 고정된 포트를 필요로 할 수도 있기 때문이다.
- **프레이밍.** `crate::protocol::{encode, decode}`(길이 프리픽스 JSON)를 변경 없이
  재사용한다; `transport`는 그 프레임들을 `TcpStream`에서 읽고 쓰는 루프만 추가한다.
- **TLS 없음, QUIC 없음 (spec 23.4, 25.1).** 이것은 M1을 위한 루프백/신뢰된 네트워크용
  shim이다; 그 이상은 `Target / Status: 미검증 (unverified)`다. `es-transport`
  (layer 11)가 이후 패킷에서 QUIC + TLS가 안착할 곳이다 — `docs/design/telemetry-protocol.md`에
  문서화되어 있으며, 여기서는 구현되지 않는다.

제약: 영어만 사용, `BTreeMap`만 사용(`HashMap` 금지), 새 외부 의존성 없음(`std`의
`TcpListener`/`TcpStream`만), 새 trait 없음(INV-17), 신규 라인 수 ≤ 약 500, 루트
`Cargo.toml`은 편집하지 않는다.

## oracle (오라클)

```
cargo fmt -p es-core -p es-telemetry -p es-runtime-embedded --check
cargo clippy -p es-core -p es-telemetry -p es-runtime-embedded --all-targets -- -D warnings
cargo test -p es-core -p es-telemetry -p es-runtime-embedded
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- `es-core::ring::RingBuffer<T>`는 기존의 모든 테스트(push/wrap/드롭 집계,
  `drain_since` 클램핑, 할당 없는 `push` 어서션)를 변경 없이 그대로 가지고 있으며,
  `es-telemetry`와 `es-runtime-embedded`는 양쪽 호출 지점에서 동작 변화 없이 이를 대상으로
  컴파일 및 테스트를 통과한다.
- 루프백 핸드셰이크: 공유된(또는 서버에 없다면 부재한) 토큰과 지원되는 버전으로 접속하면
  성공하며 `HelloAck`를 반환한다; 잘못되거나 없는 토큰은 `Bye`로 거부된다; 지원되지 않는
  버전은 불일치를 명시하는 `Bye`로 거부된다.
- 큐(16 깊이)가 전혀 비워지지 않는 클라이언트에게 1000개의 프레임을 publish하면 해당
  클라이언트에 대해 `stats().dropped > 0`이 나오고, 1000프레임 publish 루프는 1초보다
  훨씬 짧은 시간 안에 완료된다 — 즉 `publish`가 가득 찬 큐에서 블록되지 않았다는 뜻이다.
- `es-runtime-embedded`의 기존 테스트 스위트(번들 왕복, 재계획 주기, 청크 재사용 할당 없는
  틱, 텔레메트리 길이)가 새 링 타입에 대해 변경 없이 통과한다.

## forbidden (금지)

- `crates/es-editor`, `crates/es/src/cmd/backend.rs`, 그리고 모든 문서 번역 — 다른
  패킷의 범위다.
- 루트 `Cargo.toml`과 모든 새 외부 의존성(QUIC/zenoh는 이후의 `es-transport` 패킷이다 —
  언급되었을 뿐 여기서 구축되지 않는다).
- TLS, 평문 공유 토큰 이상의 실제 인증, 그리고 spec 28.7 게이트 9의 `< 1%` 오버헤드
  수치가 *측정되었다*고 주장하는 모든 것 — 측정되지 않았으며 `Target / Status: 미검증
  (unverified)`로 보고된다.
- 커밋.
