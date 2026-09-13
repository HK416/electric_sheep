//! Loopback TCP transport for the wire protocol in [`crate::protocol`] (spec 23.1, spec 25.1,
//! spec 25.3).
//!
//! This is the M1 shim, not the final transport: spec 23.4 calls for QUIC on the remote path,
//! and spec 25.1 calls for TLS. Neither is here. `es-transport` (layer 11, the only crate
//! allowed to link CUDA/HIP per `CLAUDE.md`) is where QUIC/zenoh land in a later packet; until
//! then this module gives the editor and the Python adapter something to connect to at all,
//! over `std::net::{TcpListener, TcpStream}` only — no new dependency. Treat everything here as
//! session-local and unauthenticated-in-transit: the `token` is checked once at handshake and
//! sent in clear text, which is fine on a trusted loopback and not fine over an untrusted
//! network. `Target / Status: unverified` for anything beyond that (spec 12.4 / 28.7 gate 9):
//! this module cannot measure "< 1% training overhead" from inside a unit test.
//!
//! # Wire shape
//!
//! One TCP connection per client. After connecting: client sends [`Message::Hello`], server
//! replies [`Message::HelloAck`] (accepted) or [`Message::Bye`] (rejected, then closes). Once
//! accepted, the client may send [`Message::Subscribe`] any number of times (each replaces its
//! subscription set); the server pushes [`Message::Frame`]s for the streams currently
//! subscribed. Every message is the `encode`/`decode` length-prefixed frame from
//! [`crate::protocol`] — no separate handshake wire format.
//!
//! # Backpressure
//!
//! [`Server::publish`] must never block the simulation/training loop that calls it (spec 23.4's
//! "< 1%" gate). Each client has a bounded queue ([`CLIENT_QUEUE_CAPACITY`] deep); publishing
//! tries a non-blocking send and, on a full queue, drops the frame and counts it in
//! [`ClientStats::dropped`] instead of waiting. A slow subscriber loses telemetry, never stalls
//! the producer.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::protocol::{decode, encode, negotiate, Frame, Hello, HelloAck, Message, StreamId};
use crate::PROTOCOL_VERSION;

/// Depth of each client's outgoing frame queue (see "Backpressure" above).
pub const CLIENT_QUEUE_CAPACITY: usize = 16;

/// Default for [`ServerConfig::handshake_timeout`] (spec 25.1): a connected peer that never
/// sends `Hello` must not be able to pin a server thread forever.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Default for [`ServerConfig::max_clients`] (spec 25.1): an accept-loop connection cap so an
/// unbounded number of peers cannot spawn an unbounded number of threads.
pub const DEFAULT_MAX_CLIENTS: usize = 64;

/// Tunables for [`Server::bind_with`]; [`Server::bind`] uses the defaults (spec 25.1).
#[derive(Clone, Copy, Debug)]
pub struct ServerConfig {
    /// A client that connects and then sends nothing is dropped once this much time has passed
    /// without a complete `Hello` arriving.
    pub handshake_timeout: Duration,
    /// Clients registered at once. A connection arriving once the server already holds this
    /// many is refused with `Bye { reason: "too many clients" }` before it can register.
    pub max_clients: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            handshake_timeout: HANDSHAKE_TIMEOUT,
            max_clients: DEFAULT_MAX_CLIENTS,
        }
    }
}

/// Versions this server offers during negotiation: current and N-1 (spec 25.3), or just current
/// when there is no N-1 yet.
fn server_versions() -> Vec<u32> {
    if PROTOCOL_VERSION > 1 {
        vec![PROTOCOL_VERSION - 1, PROTOCOL_VERSION]
    } else {
        vec![PROTOCOL_VERSION]
    }
}

/// Errors from [`Client::connect`] and the per-message calls on [`Client`].
#[derive(Debug, Error)]
pub enum TransportError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("protocol error: {0}")]
    Proto(#[from] crate::protocol::ProtoError),
    #[error("server rejected the connection: {reason}")]
    Rejected { reason: String },
    #[error("connection closed")]
    Closed,
    #[error("no message available right now")]
    WouldBlock,
}

#[derive(Debug, Default)]
struct ClientCounters {
    sent: AtomicU64,
    dropped: AtomicU64,
}

#[derive(Debug)]
struct ClientHandle {
    sender: SyncSender<Frame>,
    subscribed: Arc<Mutex<BTreeSet<StreamId>>>,
    counters: Arc<ClientCounters>,
}

/// Per-client counters (spec 12.4: report what actually happened, never a fabricated number).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientStats {
    pub sent: u64,
    pub dropped: u64,
}

/// Snapshot of every currently connected client's [`ClientStats`], keyed by the session id the
/// server assigned it in [`HelloAck::session_id`], plus accept-loop thread accounting
/// (P-M2-R8) so a caller can see the cap actually holding rather than trusting it blindly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServerStats {
    pub clients: BTreeMap<u64, ClientStats>,
    /// Connections past the accept loop's cap check but not yet past the handshake (no
    /// `HelloAck`/`Bye` sent yet). Counted separately from `clients` because a connection here
    /// holds a cap "slot" and a thread without appearing in the client map.
    pub handshaking: usize,
    /// Threads currently running `serve_client` (handshaking or registered), decremented on
    /// thread exit via a shared [`AtomicUsize`] — see [`CounterGuard`].
    pub threads_live: usize,
}

/// A running telemetry server: one accept loop thread plus one reader/writer thread pair per
/// connected client. Dropping this does not stop the accept loop (see the module's `ponytail`
/// note in [`bind`](Server::bind)) — it is meant to live for the process's lifetime, same as
/// the runtime it instruments.
#[derive(Debug)]
pub struct Server {
    local_addr: SocketAddr,
    clients: Arc<Mutex<BTreeMap<u64, ClientHandle>>>,
    handshaking: Arc<AtomicUsize>,
    threads_live: Arc<AtomicUsize>,
}

/// RAII decrement for a shared counter: created when a slot/thread is claimed, dropped
/// (decrementing) on every exit path — early return, `?`, or panic unwind — so nothing needs
/// to remember to decrement by hand.
struct CounterGuard(Arc<AtomicUsize>);

impl Drop for CounterGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Server {
    /// Binds `addr` (`127.0.0.1:0` picks an ephemeral loopback port) and starts accepting
    /// clients on a background thread, using [`ServerConfig::default`] (spec 25.1). `token`,
    /// when set, is required in every client's [`Hello`]; when `None`, any client is accepted.
    ///
    /// ponytail: the accept thread is fire-and-forget — there is no `Server::shutdown` that
    /// joins it, so the listener stays bound for the process's life even after every `Server`
    /// handle is dropped. Fine for an embedded runtime process that lives as long as the
    /// listener should; add a shutdown flag + a self-connect wakeup if a caller ever needs to
    /// rebind the same address later.
    pub fn bind(addr: SocketAddr, token: Option<String>) -> io::Result<Server> {
        Self::bind_with(addr, token, ServerConfig::default())
    }

    /// Like [`Server::bind`] but with explicit [`ServerConfig`] tunables (handshake timeout,
    /// connection cap) instead of the defaults.
    pub fn bind_with(
        addr: SocketAddr,
        token: Option<String>,
        cfg: ServerConfig,
    ) -> io::Result<Server> {
        let listener = TcpListener::bind(addr)?;
        let local_addr = listener.local_addr()?;
        let clients: Arc<Mutex<BTreeMap<u64, ClientHandle>>> =
            Arc::new(Mutex::new(BTreeMap::new()));
        let token = Arc::new(token);
        let next_id = Arc::new(AtomicU64::new(0));
        let handshaking = Arc::new(AtomicUsize::new(0));
        let threads_live = Arc::new(AtomicUsize::new(0));

        let accept_clients = Arc::clone(&clients);
        let accept_handshaking = Arc::clone(&handshaking);
        let accept_threads_live = Arc::clone(&threads_live);
        thread::spawn(move || {
            for incoming in listener.incoming() {
                let Ok(mut stream) = incoming else { break };
                let id = next_id.fetch_add(1, Ordering::SeqCst);

                // Connection cap (spec 25.1, P-M2-R8): checked here, before spawning a thread
                // for this connection, against every "slot" a connection can occupy — a live
                // handshake plus an already-registered client — so a peer past the cap never
                // gets a thread and is refused before it can even send `Hello`. The accept
                // loop is single-threaded, so this check-then-increment has no race with
                // itself (unlike the old check inside `serve_client`, which ran concurrently
                // per connection).
                let in_flight = accept_handshaking.load(Ordering::SeqCst)
                    + accept_clients.lock().expect("client map lock").len();
                if in_flight >= cfg.max_clients {
                    let _ = write_message(
                        &mut stream,
                        &Message::Bye {
                            reason: "too many clients".to_string(),
                        },
                    );
                    continue;
                }

                accept_handshaking.fetch_add(1, Ordering::SeqCst);
                accept_threads_live.fetch_add(1, Ordering::SeqCst);
                let clients = Arc::clone(&accept_clients);
                let token = Arc::clone(&token);
                let handshaking = Arc::clone(&accept_handshaking);
                let threads_live = Arc::clone(&accept_threads_live);
                // Instant captured at accept, not inside the spawned thread: the deadline is
                // an absolute wall-clock point the handshake read loop checks before every
                // partial read, so a peer dribbling one byte at a time is still cut off on
                // time regardless of how many short reads that takes.
                let deadline = Instant::now() + cfg.handshake_timeout;
                thread::spawn(move || {
                    let _threads_live_guard = CounterGuard(threads_live);
                    serve_client(id, stream, &clients, &token, deadline, handshaking);
                });
            }
        });

        Ok(Server {
            local_addr,
            clients,
            handshaking,
            threads_live,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Fans `frame` out to every client subscribed to `frame.stream`. Never blocks: a client
    /// whose queue is full drops the frame and counts it (see the module doc).
    // By-value per this packet's specified signature: a producer publishing many frames in a
    // loop already owns each one, and the fan-out below clones per subscriber regardless.
    #[allow(clippy::needless_pass_by_value)]
    pub fn publish(&self, frame: Frame) {
        let clients = self.clients.lock().expect("client map lock");
        for handle in clients.values() {
            let subscribed = handle.subscribed.lock().expect("subscribed set lock");
            if !subscribed.contains(&frame.stream) {
                continue;
            }
            drop(subscribed);
            match handle.sender.try_send(frame.clone()) {
                Ok(()) => {
                    handle.counters.sent.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                    handle.counters.dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// A snapshot of every currently connected client's counters, plus accept-loop thread
    /// accounting (P-M2-R8).
    pub fn stats(&self) -> ServerStats {
        let clients = self.clients.lock().expect("client map lock");
        ServerStats {
            clients: clients
                .iter()
                .map(|(id, h)| {
                    (
                        *id,
                        ClientStats {
                            sent: h.counters.sent.load(Ordering::Relaxed),
                            dropped: h.counters.dropped.load(Ordering::Relaxed),
                        },
                    )
                })
                .collect(),
            handshaking: self.handshaking.load(Ordering::SeqCst),
            threads_live: self.threads_live.load(Ordering::SeqCst),
        }
    }
}

/// Handles one accepted connection end to end: handshake, then register-and-relay until the
/// client disconnects. Runs entirely on its own thread; errors just end the connection. The
/// connection cap itself is enforced by the caller (the accept loop) before this function is
/// ever spawned — by the time this runs, the connection already holds a handshaking slot,
/// tracked by `handshaking` and released via [`CounterGuard`] on every exit path.
fn serve_client(
    id: u64,
    mut stream: TcpStream,
    clients: &Arc<Mutex<BTreeMap<u64, ClientHandle>>>,
    token: &Arc<Option<String>>,
    handshake_deadline: Instant,
    handshaking: Arc<AtomicUsize>,
) {
    let mut handshaking_guard = Some(CounterGuard(handshaking));

    // A peer that connects and stays silent — or dribbles one byte at a time — must not pin
    // this thread forever (spec 25.1). `read_message_until` re-checks `handshake_deadline`
    // before every partial read, so the *total* time spent here is bounded regardless of how
    // many short reads it takes; once past the handshake the subscribe-loop read below blocks
    // normally, so the timeout is cleared again immediately after.
    let mut buf = Vec::new();
    let Ok(Message::Hello(hello)) =
        read_message_until(&mut stream, &mut buf, Some(handshake_deadline))
    else {
        return;
    };
    if stream.set_read_timeout(None).is_err() {
        return;
    }

    if let Some(reason) = reject_reason(&hello, token.as_ref().as_ref()) {
        let _ = write_message(&mut stream, &Message::Bye { reason });
        return;
    }
    let version = negotiate(&hello.versions_supported, &server_versions())
        .expect("reject_reason already checked a shared version exists");

    let ack = Message::HelloAck(HelloAck {
        version,
        session_id: id,
        execution_hash: None,
    });
    if write_message(&mut stream, &ack).is_err() {
        return;
    }

    let (sender, receiver) = sync_channel::<Frame>(CLIENT_QUEUE_CAPACITY);
    let subscribed = Arc::new(Mutex::new(BTreeSet::new()));
    let counters = Arc::new(ClientCounters::default());
    clients.lock().expect("client map lock").insert(
        id,
        ClientHandle {
            sender,
            subscribed: Arc::clone(&subscribed),
            counters,
        },
    );
    // Only now is this connection a registered client, counted via `clients.len()` in the
    // accept loop's cap check instead of `handshaking` — release the slot right after
    // insertion (not right after the `HelloAck` write above) so the connection is never
    // invisible to the cap check in between.
    drop(handshaking_guard.take());

    let Ok(mut writer_stream) = stream.try_clone() else {
        clients.lock().expect("client map lock").remove(&id);
        return;
    };
    let writer = thread::spawn(move || {
        while let Ok(frame) = receiver.recv() {
            if write_message(&mut writer_stream, &Message::Frame(frame)).is_err() {
                break;
            }
        }
    });

    // Only `Subscribe` changes server-side state; anything else (Ping/Pong/a stray Hello) is
    // ignored rather than answered, so this thread never writes to `stream` again — the writer
    // thread above owns every post-handshake write, which keeps the two from interleaving
    // bytes on the same socket.
    loop {
        match read_message(&mut stream, &mut buf) {
            Ok(Message::Subscribe { streams }) => {
                *subscribed.lock().expect("subscribed set lock") = streams.into_iter().collect();
            }
            Ok(Message::Bye { .. }) | Err(_) => break,
            Ok(_) => {}
        }
    }

    // Dropping the map entry drops `sender`, which ends the writer thread's `recv()` loop.
    clients.lock().expect("client map lock").remove(&id);
    let _ = writer.join();
}

/// `None` when `hello` may proceed; `Some(reason)` for the [`Message::Bye`] to send instead.
fn reject_reason(hello: &Hello, token: Option<&String>) -> Option<String> {
    if negotiate(&hello.versions_supported, &server_versions()).is_none() {
        return Some(format!(
            "unsupported protocol version: client offered {:?}, server supports {:?}",
            hello.versions_supported,
            server_versions()
        ));
    }
    if let Some(expected) = token {
        let ok = match hello.token.as_deref() {
            Some(got) => ct_eq(got.as_bytes(), expected.as_bytes()),
            None => false,
        };
        if !ok {
            return Some("missing or invalid token".to_string());
        }
    }
    None
}

/// Constant-time byte-slice equality (spec 25.1): folds every position up to `max(a.len(),
/// b.len())` with no early return, so how far a wrong token gets before differing does not
/// change how long the comparison takes. No `subtle` dependency — this is the whole thing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff: u8 = u8::from(a.len() != b.len());
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
    }
    diff == 0
}

/// A connected telemetry client: one TCP connection, past the handshake.
#[derive(Debug)]
pub struct Client {
    stream: TcpStream,
    buf: Vec<u8>,
    pub session_id: u64,
    pub execution_hash: Option<[u8; 32]>,
}

impl Client {
    /// Connects to `addr` and performs the [`Hello`]/[`HelloAck`] handshake. `client_name` is
    /// this process's identity (e.g. `"es-editor"` or `"lerobot-adapter"`), carried in `Hello`
    /// for the server's own logging — it is not the auth token.
    pub fn connect(
        addr: SocketAddr,
        token: Option<String>,
        client_name: impl Into<String>,
    ) -> Result<Client, TransportError> {
        let mut stream = TcpStream::connect(addr)?;
        write_message(
            &mut stream,
            &Message::Hello(Hello {
                versions_supported: vec![PROTOCOL_VERSION],
                token,
                client: client_name.into(),
            }),
        )?;
        let mut buf = Vec::new();
        match read_message(&mut stream, &mut buf)? {
            Message::HelloAck(ack) => Ok(Client {
                stream,
                buf,
                session_id: ack.session_id,
                execution_hash: ack.execution_hash,
            }),
            Message::Bye { reason } => Err(TransportError::Rejected { reason }),
            _ => Err(TransportError::Closed),
        }
    }

    /// Replaces this client's subscription set (spec 23: the editor subscribes to whatever
    /// streams the open graph view needs, and re-subscribes when that changes).
    pub fn subscribe(&mut self, streams: Vec<StreamId>) -> Result<(), TransportError> {
        write_message(&mut self.stream, &Message::Subscribe { streams })?;
        Ok(())
    }

    /// Blocks for the next message.
    pub fn recv(&mut self) -> Result<Message, TransportError> {
        Ok(read_message(&mut self.stream, &mut self.buf)?)
    }

    /// Returns the next already-buffered message, or does one non-blocking read attempt.
    /// [`TransportError::WouldBlock`] means "nothing yet", not an error worth logging.
    pub fn try_recv(&mut self) -> Result<Message, TransportError> {
        if let Ok((msg, used)) = decode(&self.buf) {
            self.buf.drain(0..used);
            return Ok(msg);
        }
        self.stream.set_nonblocking(true)?;
        let mut chunk = [0u8; 4096];
        let read = self.stream.read(&mut chunk);
        self.stream.set_nonblocking(false)?;
        let n = match read {
            Ok(0) => return Err(TransportError::Closed),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Err(TransportError::WouldBlock)
            }
            Err(e) => return Err(e.into()),
        };
        self.buf.extend_from_slice(&chunk[..n]);
        match decode(&self.buf) {
            Ok((msg, used)) => {
                self.buf.drain(0..used);
                Ok(msg)
            }
            Err(crate::protocol::ProtoError::Incomplete { .. }) => Err(TransportError::WouldBlock),
            Err(e) => Err(e.into()),
        }
    }

    /// Tells the server this client is leaving, best-effort (a write failure here just means
    /// the connection was already gone).
    pub fn close(mut self) -> Result<(), TransportError> {
        write_message(
            &mut self.stream,
            &Message::Bye {
                reason: "client closed".to_string(),
            },
        )?;
        Ok(())
    }
}

/// Reads bytes off `stream` into `buf` until one full length-prefixed message can be decoded,
/// then drains those bytes back out of `buf` so it holds only the unconsumed remainder.
fn read_message(stream: &mut TcpStream, buf: &mut Vec<u8>) -> io::Result<Message> {
    read_message_until(stream, buf, None)
}

/// Like [`read_message`], but when `deadline` is set, the *total* time spent waiting for the
/// message to complete is bounded by it, not just each individual read: before every read that
/// would block, the remaining time until `deadline` becomes that read's timeout, and running
/// out of remaining time is itself a timeout error. This is what makes the handshake deadline
/// (P-M2-R8) absolute — a peer sending one byte every few seconds, each within its own read's
/// timeout, still cannot keep the total under `deadline` forever.
fn read_message_until(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    deadline: Option<Instant>,
) -> io::Result<Message> {
    loop {
        match decode(buf) {
            Ok((msg, used)) => {
                buf.drain(0..used);
                return Ok(msg);
            }
            Err(crate::protocol::ProtoError::Incomplete { .. }) => {
                if let Some(deadline) = deadline {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "handshake deadline exceeded",
                        ));
                    }
                    stream.set_read_timeout(Some(remaining))?;
                }
                let mut chunk = [0u8; 4096];
                let n = stream.read(&mut chunk)?;
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "peer closed"));
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }
}

fn write_message(stream: &mut TcpStream, msg: &Message) -> io::Result<()> {
    stream.write_all(&encode(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PerfMetrics;
    use es_core::PhysTick;
    use std::time::{Duration, Instant};

    fn local(server: &Server) -> SocketAddr {
        server.local_addr()
    }

    #[test]
    fn handshake_succeeds_without_a_token() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let client = Client::connect(local(&server), None, "test-client").unwrap();
        assert_eq!(client.session_id, 0);
    }

    #[test]
    fn handshake_succeeds_with_a_matching_token() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), Some("secret".into())).unwrap();
        let client = Client::connect(local(&server), Some("secret".into()), "test-client");
        assert!(client.is_ok(), "{:?}", client.err());
    }

    #[test]
    fn a_bad_token_is_rejected() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), Some("secret".into())).unwrap();
        let err = Client::connect(local(&server), Some("wrong".into()), "test-client")
            .expect_err("wrong token must be rejected");
        assert!(matches!(err, TransportError::Rejected { .. }), "{err:?}");

        let err2 = Client::connect(local(&server), None, "test-client")
            .expect_err("missing token must be rejected");
        assert!(matches!(err2, TransportError::Rejected { .. }), "{err2:?}");
    }

    #[test]
    fn ct_eq_matches_slice_equality_including_different_lengths() {
        assert!(ct_eq(b"same", b"same"));
        assert!(!ct_eq(b"same", b"diff"));
        assert!(!ct_eq(b"short", b"much longer"));
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn a_token_of_different_length_is_still_rejected() {
        let server = Server::bind(
            "127.0.0.1:0".parse().unwrap(),
            Some("a-fairly-long-secret-token".into()),
        )
        .unwrap();
        let err = Client::connect(local(&server), Some("short".into()), "test-client")
            .expect_err("mismatched-length token must be rejected");
        assert!(matches!(err, TransportError::Rejected { .. }), "{err:?}");
    }

    #[test]
    fn a_silent_client_is_dropped_after_the_handshake_timeout() {
        let cfg = ServerConfig {
            handshake_timeout: Duration::from_millis(200),
            ..ServerConfig::default()
        };
        let server = Server::bind_with("127.0.0.1:0".parse().unwrap(), None, cfg).unwrap();
        let mut stream = TcpStream::connect(local(&server)).unwrap();
        // Send nothing. Give the client-side read a generous budget past the server's own
        // timeout, so a read that returns is proof the server closed the connection, not that
        // this stream gave up first.
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut buf = [0u8; 1];
        let n = stream
            .read(&mut buf)
            .expect("server must close a silent client's connection after the handshake timeout");
        assert_eq!(n, 0, "expected EOF, got data");
    }

    #[test]
    fn a_connection_past_max_clients_is_refused() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        // Held alive for the whole test: a dropped `Client` closes its socket, which would let
        // the server reap it and free a slot before the (max_clients + 1)th connect below.
        let _clients: Vec<Client> = (0..DEFAULT_MAX_CLIENTS)
            .map(|_| Client::connect(local(&server), None, "c").unwrap())
            .collect();

        let err = Client::connect(local(&server), None, "one-too-many")
            .expect_err("the connection past max_clients must be refused");
        match err {
            TransportError::Rejected { reason } => assert!(reason.contains("too many"), "{reason}"),
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn an_unsupported_version_is_rejected() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let mut stream = TcpStream::connect(local(&server)).unwrap();
        write_message(
            &mut stream,
            &Message::Hello(Hello {
                versions_supported: vec![9999],
                token: None,
                client: "future-client".into(),
            }),
        )
        .unwrap();
        let mut buf = Vec::new();
        match read_message(&mut stream, &mut buf).unwrap() {
            Message::Bye { reason } => assert!(reason.contains("version"), "{reason}"),
            other => panic!("expected Bye, got {other:?}"),
        }
    }

    /// P-M2-R8 oracle: `max_clients + 8` silent sockets. The first `max_clients` fill the cap
    /// (each occupies a handshaking slot and a thread); the extra 8 arrive once the accept loop
    /// already sees the cap full and must be refused with `Bye` right there — before ever being
    /// read from, and without spawning a thread — so `threads_live` never exceeds the cap by
    /// more than a small constant, even while the first batch is still waiting out its
    /// handshake deadline.
    #[test]
    fn connections_past_the_cap_get_bye_without_spawning_a_thread() {
        let cfg = ServerConfig {
            handshake_timeout: Duration::from_millis(200),
            max_clients: 8,
        };
        let server = Server::bind_with("127.0.0.1:0".parse().unwrap(), None, cfg).unwrap();

        // Fill the cap with silent connections: never send `Hello`, so each sits in
        // `handshaking` until the deadline elapses.
        let filling: Vec<TcpStream> = (0..cfg.max_clients)
            .map(|_| TcpStream::connect(local(&server)).unwrap())
            .collect();
        // Give the accept loop a moment to see and admit each one.
        thread::sleep(Duration::from_millis(50));

        let mut extra: Vec<TcpStream> = (0..8)
            .map(|_| TcpStream::connect(local(&server)).unwrap())
            .collect();
        for stream in &mut extra {
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut buf = Vec::new();
            match read_message(stream, &mut buf).expect("a refused connection still gets a reply") {
                Message::Bye { reason } => assert!(reason.contains("too many"), "{reason}"),
                other => panic!("expected Bye, got {other:?}"),
            }
        }

        // Bounded well below `max_clients + 8`: the extra sockets above never got a thread, even
        // though the filling connections have not hit their handshake deadline yet.
        let stats = server.stats();
        assert!(
            stats.threads_live <= cfg.max_clients + 2,
            "threads_live must stay bounded by the cap, not by the number of connection \
             attempts: {stats:?}"
        );

        // Once the deadline passes, the filling connections are dropped too.
        thread::sleep(cfg.handshake_timeout + Duration::from_millis(500));
        let stats_after = server.stats();
        assert!(
            stats_after.threads_live <= cfg.max_clients + 2,
            "threads_live must stay bounded after the deadline too: {stats_after:?}"
        );
        assert_eq!(
            stats_after.handshaking, 0,
            "every handshake should be done or timed out"
        );

        drop(filling);
    }

    /// P-M2-R8 oracle: a peer sending one byte at a time, each within its own read's timeout,
    /// must still be dropped once the *total* handshake time exceeds `HANDSHAKE_TIMEOUT` — the
    /// bug being fixed is that the old per-read `set_read_timeout` never enforced a total
    /// budget across many short reads.
    #[test]
    fn a_dribbling_client_is_dropped_at_the_absolute_handshake_deadline() {
        let cfg = ServerConfig {
            handshake_timeout: Duration::from_millis(300),
            ..ServerConfig::default()
        };
        let server = Server::bind_with("127.0.0.1:0".parse().unwrap(), None, cfg).unwrap();
        let mut stream = TcpStream::connect(local(&server)).unwrap();

        let hello_bytes = encode(&Message::Hello(Hello {
            versions_supported: vec![PROTOCOL_VERSION],
            token: None,
            client: "dribbler".into(),
        }));
        assert!(
            hello_bytes.len() > 5,
            "the dribble must take more than one byte to matter"
        );

        let start = Instant::now();
        let budget = cfg.handshake_timeout + Duration::from_secs(1);
        // One byte every 30ms is far slower than the 300ms deadline, so the server must cut
        // this off mid-message rather than ever completing the handshake.
        for &byte in &hello_bytes {
            if start.elapsed() > budget || stream.write_all(&[byte]).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(30));
        }

        stream.set_read_timeout(Some(budget)).unwrap();
        let mut buf = [0u8; 1];
        // The server closes with some dribbled bytes still unread in the kernel receive
        // buffer, so the OS may report an abrupt reset/abort here instead of a clean EOF
        // (observed on Windows) — either is fine, since the point is that the connection
        // ends within the deadline, not the exact shutdown flavor.
        match stream.read(&mut buf) {
            Ok(n) => assert_eq!(n, 0, "expected EOF, got data"),
            Err(e) => assert!(
                matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
                ),
                "unexpected read error: {e:?}"
            ),
        }
        assert!(
            start.elapsed() < budget,
            "must be dropped within HANDSHAKE_TIMEOUT + 1s, took {:?}",
            start.elapsed()
        );
    }

    fn frame(stream: StreamId, tick: u64) -> Frame {
        Frame {
            tick: PhysTick(tick),
            wall_ns: tick,
            stream,
            payload: crate::protocol::Payload::Metrics(PerfMetrics::default()),
        }
    }

    #[test]
    fn publish_reaches_a_subscribed_client() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let mut client = Client::connect(local(&server), None, "test-client").unwrap();
        client.subscribe(vec![StreamId(1)]).unwrap();
        // Give the server's reader thread a moment to apply the subscription before publishing.
        thread::sleep(Duration::from_millis(50));

        server.publish(frame(StreamId(1), 1));
        match client.recv().unwrap() {
            Message::Frame(f) => assert_eq!(f.stream, StreamId(1)),
            other => panic!("expected a Frame, got {other:?}"),
        }
    }

    #[test]
    fn a_slow_client_drops_frames_without_blocking_the_publisher() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let mut client = Client::connect(local(&server), None, "slow-client").unwrap();
        client.subscribe(vec![StreamId(0)]).unwrap();
        // Let the subscription land before the flood, and never call recv() again: the
        // client's OS socket buffer plus the server's CLIENT_QUEUE_CAPACITY-deep queue are the
        // only slack, so this must overflow well before 1000 frames.
        thread::sleep(Duration::from_millis(50));

        let start = Instant::now();
        for t in 0..1000u64 {
            server.publish(frame(StreamId(0), t));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "publish must never block on a full client queue, took {elapsed:?}"
        );

        // The writer thread may still be draining its queue into the client's OS buffer;
        // give it a moment before reading the final tally.
        thread::sleep(Duration::from_millis(200));
        let stats = server.stats();
        let client_stats = stats.clients.get(&client.session_id).expect("registered");
        assert!(
            client_stats.dropped > 0,
            "a queue of {CLIENT_QUEUE_CAPACITY} fed 1000 frames with no reader must drop: {stats:?}"
        );
    }

    #[test]
    fn close_tells_the_server_this_client_is_leaving() {
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let client = Client::connect(local(&server), None, "test-client").unwrap();
        let id = client.session_id;
        client.close().unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if !server.stats().clients.contains_key(&id) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("server never reaped the closed client");
    }
}
