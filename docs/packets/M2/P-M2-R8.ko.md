<!-- Korean translation of docs/packets/M2/P-M2-R8.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R8 — accept 루프 연결 상한과 절대 핸드셰이크 데드라인

M2 리뷰의 should-fix(`docs/reviews/M2.md`, Should-fix,
`crates/es-telemetry/src/transport.rs:173,255`)를 고친다: 클라이언트 상한은
`thread::spawn`이 이미 실행된 뒤 `serve_client` 안에서 적용되고 있었고, 핸드셰이크
타임아웃은 전체 핸드셰이크에 대한 예산이 아니라 읽기별 `set_read_timeout`이었다 — 그래서
한 번에 한 바이트씩 흘리는 피어는 결코 이를 건드리지 않았고 서버 스레드를 무기한 붙잡아
두었다. 설계 노트: `docs/design/telemetry-protocol.md` §4 (이 패킷에서 갱신됨).

## context (범위)

```
crates/es-telemetry/src/transport.rs   (cap moved into the accept loop, absolute deadline,
                                         Server::stats().{handshaking,threads_live}, tests)
docs/design/telemetry-protocol.md      (§4 resource-limits section)
docs/packets/M2/P-M2-R8.md             (new)
```

## spec (사양)

- **스폰 전 상한 검사.** accept 루프(`listener.incoming()`를 실행하는 단일 스레드)는
  새로 accept된 연결에 대해 `thread::spawn`을 호출하기 *전에* `handshaking.load() +
  clients.lock().len() >= cfg.max_clients`를 검사한다. 상한을 넘으면: accept 루프 자신이
  동기적으로 `Bye { reason: "too many clients" }`를 쓴 다음 소켓을 드롭한다(닫는다) —
  스레드는 스폰되지 않고, `Hello`는 결코 읽히지 않는다. accept 루프는 단일 스레드이므로 이
  검사-후-증가는 (연결마다 동시에 실행되던) `serve_client` 안의 기존 검사와 달리 자기
  자신과의 경합이 없다.
- **절대 핸드셰이크 데드라인.** accept 루프는 admit 시점에 `deadline = Instant::now() +
  cfg.handshake_timeout`을 기록하고 이를 `serve_client`에 전달하며, `serve_client`는
  `read_message_until(stream, buf, Some(deadline))`을 통해 `Hello`를 읽는다: 블록될 수
  있는 모든 소켓 읽기 전에 이 함수는 `deadline`까지 *남은* 시간을 다시 계산해 그것을 해당
  읽기의 타임아웃으로 쓰며, 남은 시간이 없어지면 오류를 낸다. 이는 느리거나 악의적인 피어가
  일으키는 부분 읽기의 개수와 무관하게 전체 핸드셰이크 시간을 경계 짓는다 — 단일
  `set_read_timeout` 호출(이전 형태)은 오직 하나의 읽기만 경계 지을 뿐, 그것들의 합을
  경계 짓지 않는다.
- **관측 가능한 스레드 계정.** `Server`는 두 개의 `Arc<AtomicUsize>` — `handshaking`
  (상한을 통과해 admit됐지만 아직 `Hello`를 통과하지 못한 연결)과 `threads_live`(현재
  실행 중인 모든 `serve_client` 스레드) — 를 보유한다. `ServerStats`는 `Server::stats()`가
  채우는 `handshaking`과 `threads_live` 필드를 새로 얻는다. 두 카운터 모두 스레드/핸드셰이크
  종료 시 작은 RAII `CounterGuard`를 통해 해제되므로, 모든 경로 — 정상 반환, 이른
  `?`/`else return`, 또는 패닉 언와인드 — 가 정확히 한 번씩 스스로를 계산에 반영한다.
  handshaking 가드는 (`HelloAck` 기록 직후가 아니라) 연결이 클라이언트 맵에 삽입된 뒤에야
  해제되므로, 그 사이에 연결이 상한 검사에서 보이지 않게 되는 일은 없다.
- **`ServerConfig`의 형태는 변하지 않는다.** `handshake_timeout`과 `max_clients`는 이미
  존재했으며 상한과 데드라인 모두에 재사용된다; 새 설정 필드는 없다.

제약: 영어만 사용, 새 외부 의존성 없음, 새 트레이트 없음(INV-17), 새
`HashMap` 없음(`BTreeMap`만).

## oracle (오라클)

```
cargo fmt -p es-telemetry --check
cargo clippy -p es-telemetry --all-targets -- -D warnings
cargo test -p es-telemetry
```

- `connections_past_the_cap_get_bye_without_spawning_a_thread`
  (`crates/es-telemetry/src/transport.rs`): `max_clients = 8`로 상한을 채우기 위해 조용한
  소켓 8개를 연 다음 8개를 더 연다; 추가된 8개 각각은 핸드셰이크를 전혀 완료하지 않은 채
  `Bye { reason: "too many clients" }`를 회신받으며, `Server::stats().threads_live`는
  처음 8개가 아직 핸드셰이크 데드라인을 기다리는 동안과 그것이 지난 뒤 모두
  `<= max_clients + 2`로 유지된다.
- `a_dribbling_client_is_dropped_at_the_absolute_handshake_deadline`: 300 ms
  `handshake_timeout`에 대해, 유효한 `Hello`를 30 ms마다 한 바이트씩(개별 읽기 자체의
  타임아웃 안에는 넉넉히 들어가도록) 보낸다; 부분 읽기와 무관하게 연결은
  `handshake_timeout + 1s` 이내에 드롭된다.

## acceptance (수용 기준)

- 기존 `es-telemetry` 테스트가 모두 그린 상태를 유지한다.
  `a_connection_past_max_clients_is_refused`와 `a_silent_client_is_dropped_after_the_
  handshake_timeout`을 포함한다.
- 새 테스트 둘 다 반복된 로컬 실행에서 결정적이며, `cargo test -p es-telemetry` 전체
  스위트는 3초보다 훨씬 짧게 끝난다(측정값: 약 0.8초).
- `docs/design/telemetry-protocol.md` §4는 cap-before-Hello 순서(동작 변경: 상한을 초과한
  연결은 이제 잘못된 토큰을 가졌더라도 "too many clients"를 받는다, 그 `Hello`가 결코 읽히지
  않기 때문이다), 절대 데드라인 메커니즘, 그리고 새로운 `ServerStats` 필드를 서술한다.

## forbidden (금지)

- `docs/reviews/M2.md`의 다른 should-fix나 nit(별도 패킷이다).
- `crates/es-telemetry/src/protocol.rs`, `crates/es-telemetry/src/ring.rs`, 그리고 다른
  모든 크레이트 — 이 패킷은 transport 전용이다.
- 새 의존성, 새 트레이트, TLS/QUIC(여전히 `es-transport`, layer 11, 나중에).
- 커밋.
</content>
