<!-- Korean translation of docs/api-notes/zenoh-rs.md. The English file is the working copy; regenerate this when it changes. -->

# zenoh (Rust) -- `es_ros2::session`을 위한 고정된 표면

`crates/es-ros2`가 `zenoh` cargo feature 뒤에서 호출하는 `zenoh` crate API
(`docs/design/ros2-boundary.md` 섹션 2, 4). rmw_zenoh가 기대하는 key-expression, liveliness,
attachment의 *내용*은 `docs/api-notes/rmw-zenoh.md`에 있다; 이 파일은 전송 라이브러리만
다룬다.

Spec: §24.1(모드 A: key expression, CDR, attachment, liveliness "with `zenoh-rs`"),
§25.1(기본 localhost binding, 기본으로는 TLS 스택을 끌어들이지 않음).

## 버전

| | |
|---|---|
| crate | **`zenoh = "=1.8.0"`**(2026-03-13 릴리스): ROS 2의 `zenoh_cpp_vendor`가 빌드하는 바로 그 버전(zenoh-c 1.8.0 -> `2687c51352121f006e3a603ce07925a8ad0b295c`의 `zenoh 1.8.0`, `docs/api-notes/rmw-zenoh.md`). 최신은 1.10.1(2026-09-07)이다. <https://crates.io/api/v1/crates/zenoh/versions>, 2026-09-14 조회 |
| 아래 시그니처를 읽은 출처 | tag **`1.10.1`**, `https://raw.githubusercontent.com/eclipse-zenoh/zenoh/1.10.1/<path>`; attachment serializer는 `2687c51`(1.8.0)에서. 1.8.0/1.10.1 시그니처 차이가 있으면 W1b에서 컴파일 에러로 드러나며, 그것이 이 파일을 바로잡는다 |
| 명시된 MSRV | 1.8.0과 1.10.1 모두 `rust_version = "1.75.0"`(crates.io per-version API); README: "some of its dependencies may require newer Rust versions". `rust-version = "1.85"`로 1.10.1을 로컬에서 resolve했을 때 cargo의 MSRV-aware resolver는 **1.85를 넘는 의존성을 0개** 골랐다(예: `time-core v0.1.7`, "available: v0.1.9, requires Rust 1.88.0"). **W1b, 2026-09-14: `cargo +1.85 check -p es-ros2 --features zenoh` PASS**(Windows, `1.85-x86_64-pc-windows-msvc`), `.cargo/config.toml`의 `[resolver] incompatible-rust-versions = "fallback"`과 함께 — 단순히 resolve만 된 것이 아니라 1.8.0의 실제 `cargo +1.85` 빌드가 검증됨 |
| license | `EPL-2.0 OR Apache-2.0` |
| `zenoh-ext` | **의존성이 아니다.** rmw_zenoh의 attachment가 필요로 하는 두 serializer 인코딩은 손으로 직접 구현했다(아래) |
| interop peer 버전 | 업스트림 ROS 2 kilted/lyrical/rolling: zenoh 1.8.0(= 우리 pin). RoboStack `ros-kilted-rmw-zenoh-cpp 0.6.6`은 `libzenohc >=1.7.2,<1.7.3`을 링크; `ros-lyrical-rmw-zenoh-cpp 0.10.x`는 `libzenohc >=1.9.0,<1.9.1`을 링크. **W1b, 2026-09-14: wire 호환성 1.8.0(우리) <-> 1.7.2(RoboStack Kilted의 `rmw_zenohd`) VERIFIED** — `rmw_zenoh_interop.rs`의 라이브 오라클(talker/listener/CLI/`topic pub`, 다섯 시나리오 모두)이 그 빌드의 실제 `rmw_zenohd` router를 상대로 통과 |

의존성 라인(W1b):

```toml
zenoh = { version = "=1.8.0", default-features = false, features = ["transport_tcp"], optional = true }
```

## Feature

- 기본값: `auth_pubkey`, `auth_usrpwd`, `transport_compression`, `transport_multilink`,
  `transport_quic`, `transport_quic_datagram`, `transport_tcp`, `transport_tls`,
  `transport_udp`, `transport_unixsock-stream`, `transport_ws`.
- TLS/crypto는 `transport_tls`/`transport_quic`(`rustls`, `ring`, `quinn`, `rcgen`)와
  `transport_multilink -> auth_pubkey -> rsa`에서 온다.
- `default-features = false, features = ["transport_tcp"]`: 272개 패키지, **rustls / ring /
  quinn / rsa / tungstenite 없음**(로컬 `cargo tree` resolve; 기본값은 383개). **W1b,
  2026-09-14: 빌드 VERIFIED**, Windows와 Linux 둘 다(`cargo build -p es-ros2 --features
  zenoh`); `cargo tree -p es-ros2 --features zenoh -e normal --prefix none | grep -cE
  '^(rustls|ring|quinn|rsa) '`는 둘 다에서 `0`을 출력한다.
- `unstable`은 `Sample::source_info`, reliability, zenoh-ext advanced pub/sub를 게이팅한다.
  여기서는 아무것도 그것을 필요로 하지 않는다. `liveliness`는 **stable**이다(`pub mod
  liveliness`에는 cfg gate가 없다).

## 런타임 모델

- zenoh는 자신만의 multi-thread tokio runtime을 만든다(`zenoh-runtime`). current-thread tokio
  아래에서는 panic한다: "Zenoh runtime doesn't support Tokio's current thread scheduler".
- 모든 builder는 `.await`나 `zenoh::Wait::wait()`("by calling the `wait` method in a synchronous
  context")로 resolve된다. `pub trait Wait: Resolvable { fn wait(self) -> Self::To; }`.
  **`es-ros2`는 `.wait()`만 쓴다** -- 공개 API에 async가 없고, 자신만의 tokio 의존성도 없다.

## 사용하는 API

```rust
use zenoh::{Config, Wait};

pub fn open<TryIntoConfig>(config: TryIntoConfig) -> OpenBuilder<TryIntoConfig>;
impl Config {
    pub fn from_json5(input: &str) -> ZResult<Config>;
    pub fn from_file<P: AsRef<Path>>(path: P) -> ZResult<Self>;
    pub fn insert_json5(&mut self, key: &str, value: &str) -> ZResult<()>;
}
// keys used: "mode" (r#""peer""# | r#""client""#), "connect/endpoints", "listen/endpoints",
//            "scouting/multicast/enabled"

impl Session {
    pub fn declare_publisher<'b, TryIntoKeyExpr>(&self, key_expr: TryIntoKeyExpr) -> PublisherBuilder<'_, 'b>;
    pub fn declare_subscriber<'b, TryIntoKeyExpr>(&self, key_expr: TryIntoKeyExpr) -> SubscriberBuilder<'_, 'b, DefaultHandler>;
    pub fn put<'a, 'b: 'a, TryIntoKeyExpr, IntoZBytes>(&'a self, key_expr: TryIntoKeyExpr, payload: IntoZBytes) -> SessionPutBuilder<'a, 'b>;
    pub fn liveliness(&self) -> Liveliness<'_>;
    pub fn close(&self) -> CloseBuilder<Self>;
}
impl Publisher<'_> { pub fn put<IntoZBytes>(&self, payload: IntoZBytes) -> PublisherPutBuilder<'_>; }
// put builders (builders/sample.rs):
//   fn attachment<T: Into<OptionZBytes>>(self, attachment: T) -> Self;
//   fn encoding<T: Into<Encoding>>(self, encoding: T) -> Self;
// subscriber: .callback(move |sample| { ... }).wait()   (tests/session.rs)
//             or the default FIFO handler: subscriber.recv() / recv_async()

impl Sample {
    pub fn key_expr(&self) -> &KeyExpr<'static>;
    pub fn payload(&self) -> &ZBytes;
    pub fn attachment(&self) -> Option<&ZBytes>;
    pub fn timestamp(&self) -> Option<&Timestamp>;
}
impl ZBytes { pub fn to_bytes(&self) -> Cow<'_, [u8]>; }   // From<Vec<u8>>, From<&[u8]>, From<[u8; N]>

impl Liveliness<'_> {
    pub fn declare_token<'b, ..>(&self, key_expr) -> LivelinessTokenBuilder<..>;   // token undeclared on drop
    pub fn declare_subscriber<'b, ..>(&self, key_expr) -> LivelinessSubscriberBuilder<..>; // .history(bool)
    pub fn get<'b, ..>(&self, key_expr) -> LivelinessGetBuilder<..>;               // .timeout(Duration)
}
// z_get_liveliness.rs: let replies = session.liveliness().get(&ke).timeout(t).await;
//                      while let Ok(reply) = replies.recv_async().await { reply.result() ... }
```

## Key-expression 규칙 (`commons/zenoh-keyexpr/src/key_expr/borrowed.rs`)

- "Key expressions may never start or end with `'/'`, nor contain `"//"` or any of the
  following characters: `#$?`"; chunk는 "are not allowed to be empty"다; canon form이어야 한다.
- `*` / `**` / `$*`는 wildcard다; 선언된(query가 아닌) key expression은 이들을 포함해서는
  안 된다.
- `@`로 시작하는 chunk는 리터럴이다: wildcard가 매치하지 않으며, 그대로 이름 붙여야
  한다(`@ros2_lv/**`, `@/*/@ros2_lv/**`).

## Serializer wire format (rmw_zenoh의 attachment 전용)

Zenoh serialization RFC,
<https://github.com/eclipse-zenoh/roadmap/blob/main/rfcs/ALL/Serialization.md>:

- 정수: "fixed-size little endian representation; signed integers use two's complement".
  `i64` = 8바이트 LE.
- Sequence: "first writing the sequence length using LEB128 encoding, then concatenating ...
  Rust should implement serialization of fixed-size array as a variable-length sequence."
  `[1u8, 2, 3] -> 03 01 02 03`. 따라서 `[u8; 16]` = `0x10` + 16바이트. deserializer는 length
  != N이면 거부한다.
- Tuple: 항목들의 연결(concatenation).

## 인프로세스 loopback (테스트; 라우터 없음, 네트워크 없음)

`commons/zenoh-test/src/lib.rs` 패턴: "Dynamic port allocation (`tcp/127.0.0.1:0`)",
"Multicast scouting is disabled"(`scouting/multicast/enabled = false`); 한쪽 peer가 listen하고
다른 쪽이 connect한다(`zenoh/tests/liveliness.rs`: peer2의 토큰을 peer1의 liveliness
subscriber가 본다). `:0`에 바인딩된 포트를 다시 읽으려면 `get_locators_from_session`이
필요한데, 이는 `internal` API를 쓴다 -- **여기서는 활성화되지 않음**. 대신 `es-ros2`
테스트는 `std::net::TcpListener::bind("127.0.0.1:0")`로 포트를 예약하고, 그것을 drop한 다음,
그 포트에서 listen한다(`AddrInUse`면 재시도). **W1b, 2026-09-14:** `session_loopback.rs`의
13개 테스트가 정확히 이 패턴을 사용한다(재시도 루프는 구현되지 않음 — 로컬에서 ~10회
반복 실행하는 동안 필요한 것이 관측되지 않았음); Windows와 Linux 모두에서 전부 통과한다.
1.8.0의 `LivelinessToken`과 `zenoh::pubsub::Publisher`/`Subscriber`는 `Publisher<'static>`를
제외하면 **`es-ros2` 자신의 struct로 새어 들어가는 lifetime parameter가 없다**(`Session`이
내부적으로 `Arc` 기반이므로 `Session::declare_publisher`에서 곧바로 얻어짐) — 이 파일
다른 곳의 `Publisher<'_>` 표기는 1.10.1의 API다; 1.8.0은 `'static`으로 동일하게
컴파일된다.
