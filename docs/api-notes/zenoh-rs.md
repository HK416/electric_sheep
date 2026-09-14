# zenoh (Rust) -- pinned surface for `es_ros2::session`

The `zenoh` crate API that `crates/es-ros2` calls behind its `zenoh` cargo feature
(`docs/design/ros2-boundary.md` sections 2, 4). The key-expression, liveliness and attachment
*contents* rmw_zenoh expects are in `docs/api-notes/rmw-zenoh.md`; this file is only the
transport library.

Spec: §24.1 (mode A: key expression, CDR, attachment, liveliness "with `zenoh-rs`"),
§25.1 (default localhost binding, no TLS stack pulled in by default).

## Version

| | |
|---|---|
| crate | **`zenoh = "=1.8.0"`** (released 2026-03-13): the exact version ROS 2's `zenoh_cpp_vendor` builds (zenoh-c 1.8.0 -> `zenoh 1.8.0` at `2687c51352121f006e3a603ce07925a8ad0b295c`, `docs/api-notes/rmw-zenoh.md`). Latest is 1.10.1 (2026-09-07). <https://crates.io/api/v1/crates/zenoh/versions>, retrieved 2026-09-14 |
| where the signatures below were read | tag **`1.10.1`**, `https://raw.githubusercontent.com/eclipse-zenoh/zenoh/1.10.1/<path>`; the attachment serializer at `2687c51` (1.8.0). Any 1.8.0/1.10.1 signature difference surfaces as a compile error in W1b, which corrects this file |
| declared MSRV | `rust_version = "1.75.0"` for both 1.8.0 and 1.10.1 (crates.io per-version API); README: "some of its dependencies may require newer Rust versions". A local resolve of 1.10.1 with `rust-version = "1.85"` and cargo's MSRV-aware resolver picked **0 dependencies above 1.85** (e.g. `time-core v0.1.7`, "available: v0.1.9, requires Rust 1.88.0"). **W1b, 2026-09-14: `cargo +1.85 check -p es-ros2 --features zenoh` PASSES** (Windows, `1.85-x86_64-pc-windows-msvc`) with `.cargo/config.toml`'s `[resolver] incompatible-rust-versions = "fallback"` — real `cargo +1.85` build of 1.8.0 verified, not just resolved |
| license | `EPL-2.0 OR Apache-2.0` |
| `zenoh-ext` | **not a dependency.** The two serializer encodings rmw_zenoh's attachment needs are hand-rolled (below) |
| interop peer versions | upstream ROS 2 kilted/lyrical/rolling: zenoh 1.8.0 (= our pin). RoboStack `ros-kilted-rmw-zenoh-cpp 0.6.6` links `libzenohc >=1.7.2,<1.7.3`; `ros-lyrical-rmw-zenoh-cpp 0.10.x` links `libzenohc >=1.9.0,<1.9.1`. **W1b, 2026-09-14: wire compatibility 1.8.0 (us) <-> 1.7.2 (RoboStack Kilted's `rmw_zenohd`) VERIFIED** — `rmw_zenoh_interop.rs`'s live oracle (talker/listener/CLI/`topic pub`, all five scenarios) passed against a real `rmw_zenohd` router from that build |

Dependency line (W1b):

```toml
zenoh = { version = "=1.8.0", default-features = false, features = ["transport_tcp"], optional = true }
```

## Features

- Defaults: `auth_pubkey`, `auth_usrpwd`, `transport_compression`, `transport_multilink`,
  `transport_quic`, `transport_quic_datagram`, `transport_tcp`, `transport_tls`,
  `transport_udp`, `transport_unixsock-stream`, `transport_ws`.
- TLS/crypto come from `transport_tls`/`transport_quic` (`rustls`, `ring`, `quinn`, `rcgen`)
  and `transport_multilink -> auth_pubkey -> rsa`.
- `default-features = false, features = ["transport_tcp"]`: 272 packages, **no rustls / ring /
  quinn / rsa / tungstenite** (local `cargo tree` resolve; defaults: 383). **W1b, 2026-09-14:
  build VERIFIED** on Windows and Linux (`cargo build -p es-ros2 --features zenoh`); `cargo tree
  -p es-ros2 --features zenoh -e normal --prefix none | grep -cE '^(rustls|ring|quinn|rsa) '`
  prints `0` on both.
- `unstable` gates `Sample::source_info`, reliability and zenoh-ext advanced pub/sub. Nothing
  here needs it. `liveliness` is **stable** (`pub mod liveliness` has no cfg gate).

## Runtime model

- zenoh builds its own multi-thread tokio runtimes (`zenoh-runtime`). Under a current-thread
  tokio it panics: "Zenoh runtime doesn't support Tokio's current thread scheduler".
- Every builder resolves either with `.await` or with `zenoh::Wait::wait()` ("by calling the
  `wait` method in a synchronous context"). `pub trait Wait: Resolvable { fn wait(self) -> Self::To; }`.
  **`es-ros2` uses `.wait()` only** -- no async in its public API, no tokio dependency of its own.

## API used

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

## Key-expression rules (`commons/zenoh-keyexpr/src/key_expr/borrowed.rs`)

- "Key expressions may never start or end with `'/'`, nor contain `"//"` or any of the
  following characters: `#$?`"; chunks "are not allowed to be empty"; must be canon form.
- `*` / `**` / `$*` are wildcards; a declared (non-query) key expression must not contain them.
- A chunk starting with `@` is verbatim: wildcards do not match it, it must be named
  literally (`@ros2_lv/**`, `@/*/@ros2_lv/**`).

## Serializer wire format (for rmw_zenoh's attachment only)

Zenoh serialization RFC, <https://github.com/eclipse-zenoh/roadmap/blob/main/rfcs/ALL/Serialization.md>:

- Integers: "fixed-size little endian representation; signed integers use two's complement".
  `i64` = 8 bytes LE.
- Sequences: "first writing the sequence length using LEB128 encoding, then concatenating ...
  Rust should implement serialization of fixed-size array as a variable-length sequence."
  `[1u8, 2, 3] -> 03 01 02 03`. So `[u8; 16]` = `0x10` + 16 bytes. The deserializer rejects a
  length != N.
- Tuples: concatenation of the items.

## In-process loopback (tests; no router, no network)

`commons/zenoh-test/src/lib.rs` pattern: "Dynamic port allocation (`tcp/127.0.0.1:0`)",
"Multicast scouting is disabled" (`scouting/multicast/enabled = false`); one peer listens, the
other connects (`zenoh/tests/liveliness.rs`: a token on peer2 is seen by a liveliness subscriber
on peer1). Reading back the port bound to `:0` needs `get_locators_from_session`, which uses
`internal` APIs -- **not enabled here**. `es-ros2` tests instead reserve a port with
`std::net::TcpListener::bind("127.0.0.1:0")`, drop it, and listen on that port (retry on
`AddrInUse`). **W1b, 2026-09-14:** `session_loopback.rs`'s 13 tests use exactly this pattern
(no retry loop implemented — not observed to be needed in ~10 repeated local runs); all pass
on Windows and Linux. `LivelinessToken` and `zenoh::pubsub::Publisher`/`Subscriber` in 1.8.0
carry **no lifetime parameter that leaks into `es-ros2`'s own structs** except
`Publisher<'static>` (obtained straight from `Session::declare_publisher`, since `Session` is
internally `Arc`-backed) — the `Publisher<'_>` notation elsewhere in this file is 1.10.1's API;
1.8.0's compiles the same way with `'static`.
