//! [`HilLink`]: the UDP front end (`docs/design/ros2-boundary.md` sections 7.1, 7.2 and 7.7).
//!
//! This module and [`super::stats`] are the only two that may name a wall clock, and they use
//! it for two things only: the round-trip/jitter observation and the `sender_mono_ns` header
//! field. Neither reaches [`super::HilCore`], so no wall-clock value is ever a Safety Plane
//! input (spec 3.4, spec 3.5).
//!
//! Security (spec 25.1, design note section 7.7): bind `127.0.0.1` unless explicitly
//! configured otherwise, authenticate every datagram with the 16-byte keyed-blake3 tag, and
//! bound every parse. A datagram that fails any of that is counted and dropped here; it never
//! reaches the plane.

use std::net::{SocketAddr, UdpSocket};
use std::time::Instant;

use es_core::PhysTick;
use es_safety::{SafeAction, SafetyCounters};
use es_telemetry::Server;

use super::stats::HilStats;
use super::wire::{self, HilMsg, MsgKind, BYE_HASH_MISMATCH, BYE_SHAPE_MISMATCH, BYE_SHUTDOWN};
use super::{HilCore, HilError};

/// Round trips tracked at once, one slot per `tick % RTT_SLOTS`. A command answering a State
/// older than this simply has no round-trip sample; it is still submitted to the plane.
const RTT_SLOTS: usize = 256;

/// Datagrams drained in one tick. A bound, not a budget: a peer flooding the socket must not
/// keep the control loop inside `drain` forever.
const MAX_DRAIN: usize = 4096;

/// How the link binds and authenticates (design note section 7.7).
#[derive(Clone)]
pub struct HilConfig {
    /// `127.0.0.1:0` by default in every test and example; anything else is a deliberate
    /// choice by the operator (spec 25.1).
    pub bind: SocketAddr,
    /// The shared 32-byte key, read from a `*.eshilkey` file outside the repo.
    pub key: [u8; 32],
    /// Publish a [`HilStats`] telemetry frame every this many ticks; `0` is read as `1`.
    pub stats_every: u64,
    /// Largest accepted datagram, clamped to [`wire::MAX_DATAGRAM`].
    pub max_datagram: usize,
}

impl std::fmt::Debug for HilConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HilConfig")
            .field("bind", &self.bind)
            .field("key", &"<redacted>")
            .field("stats_every", &self.stats_every)
            .field("max_datagram", &self.max_datagram)
            .finish()
    }
}

/// The UDP front end of a HIL run: drains the socket once per tick, hands what it accepted to
/// [`HilCore`], and answers with the measured `State`.
pub struct HilLink<const NJ: usize, const H: usize> {
    sock: UdpSocket,
    core: HilCore<NJ, H>,
    key: [u8; 32],
    stats_every: u64,
    max_datagram: usize,
    local_addr: SocketAddr,
    session_id: u64,
    /// Bumped for every session this link opens, so two `Hello`s inside one clock tick still
    /// get different ids (design note section 7.7).
    session_counter: u64,
    peer: Option<SocketAddr>,
    /// Highest `seq` accepted from the peer in this session.
    last_seq: u64,
    tx_seq: u64,
    started: Instant,
    /// `(tick, sent at)` per `tick % RTT_SLOTS`, for the State -> Command round trip.
    sent: [(u64, Instant); RTT_SLOTS],
    rx: Vec<u8>,
    tx: Vec<u8>,
}

impl<const NJ: usize, const H: usize> std::fmt::Debug for HilLink<NJ, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HilLink")
            .field("local_addr", &self.local_addr)
            .field("session_id", &self.session_id)
            .field("peer", &self.peer)
            .field("stats", &self.core.stats())
            .finish_non_exhaustive()
    }
}

impl<const NJ: usize, const H: usize> HilLink<NJ, H> {
    /// Binds the socket and takes ownership of `core`.
    pub fn bind(cfg: &HilConfig, core: HilCore<NJ, H>) -> std::io::Result<Self> {
        let sock = UdpSocket::bind(cfg.bind)?;
        sock.set_nonblocking(true)?;
        let local_addr = sock.local_addr()?;
        let started = Instant::now();
        let mut link = Self {
            sock,
            core,
            key: cfg.key,
            stats_every: cfg.stats_every.max(1),
            max_datagram: cfg.max_datagram.clamp(wire::HEADER_LEN, wire::MAX_DATAGRAM),
            local_addr,
            session_id: 0,
            session_counter: 0,
            peer: None,
            last_seq: 0,
            tx_seq: 0,
            started,
            sent: [(u64::MAX, started); RTT_SLOTS],
            rx: Vec::new(),
            tx: Vec::new(),
        };
        // The field is never left uninitialised, even though no session is open yet.
        link.session_id = link.next_session_id();
        Ok(link)
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn stats(&self) -> HilStats {
        self.core.stats()
    }

    /// The Safety Plane's own counters (spec 10.3).
    pub fn counters(&self) -> &SafetyCounters {
        self.core.counters()
    }

    /// One control tick: drain, bin, step, answer (design note section 7.3).
    pub fn tick(&mut self, now: PhysTick, q: &[f64; NJ], qd: &[f64; NJ]) -> SafeAction<NJ> {
        self.drain(now);
        self.core.observe_state(now, q, qd);
        let action = self.core.tick(now);
        if let Some(peer) = self.peer {
            self.send(
                peer,
                &HilMsg::State {
                    tick: now,
                    q: *q,
                    qd: *qd,
                },
            );
            self.sent[now.0 as usize % RTT_SLOTS] = (now.0, Instant::now());
        }
        action
    }

    /// Publishes a [`HilStats`] frame every `stats_every` ticks (design note section 7.6).
    /// Never blocks the loop: [`Server::publish`] drops rather than waits.
    pub fn publish_stats(&self, server: &Server, now: PhysTick) {
        let stats = self.core.stats();
        if stats.ticks % self.stats_every == 0 {
            server.publish(stats.frame(now));
        }
    }

    /// Says goodbye, writes the trailer and returns the decisions hash.
    pub fn finish(mut self) -> Result<[u8; 32], HilError> {
        if let Some(peer) = self.peer {
            self.send(
                peer,
                &HilMsg::Bye {
                    reason: BYE_SHUTDOWN,
                },
            );
        }
        self.core.finish()
    }

    // --- internals --------------------------------------------------------------------

    fn drain(&mut self, now: PhysTick) {
        let mut buf = std::mem::take(&mut self.rx);
        buf.resize(self.max_datagram, 0);
        for _ in 0..MAX_DRAIN {
            match self.sock.recv_from(&mut buf) {
                Ok((n, from)) => self.accept(&buf[..n], from, now),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                // An oversize datagram is reported as an error on some platforms and
                // silently truncated on others; either way it is refused and counted.
                Err(_) => self.core.stats_mut().rx_invalid += 1,
            }
        }
        self.rx = buf;
    }

    fn accept(&mut self, bytes: &[u8], from: SocketAddr, now: PhysTick) {
        let Ok((hdr, msg)) = wire::decode::<NJ, H>(&self.key, bytes) else {
            self.core.stats_mut().rx_invalid += 1;
            return;
        };
        if hdr.kind == MsgKind::Hello {
            self.hello(&hdr, &msg, from, now);
            return;
        }
        if hdr.session_id != self.session_id || self.peer != Some(from) {
            self.core.stats_mut().rx_invalid += 1;
            return;
        }
        if hdr.seq <= self.last_seq {
            self.core.stats_mut().rx_stale += 1;
            return;
        }
        self.core.stats_mut().rx_lost += hdr.seq - self.last_seq - 1;
        self.last_seq = hdr.seq;
        match msg {
            HilMsg::Heartbeat => self.core.on_heartbeat(),
            HilMsg::Command(cmd) => {
                self.observe_rtt(cmd.obs_tick);
                self.core.on_command(cmd);
            }
            HilMsg::Bye { .. } => self.peer = None,
            // Link-to-controller messages coming back the other way: not a valid input.
            _ => self.core.stats_mut().rx_invalid += 1,
        }
    }

    fn hello(&mut self, hdr: &wire::Header, msg: &HilMsg<NJ, H>, from: SocketAddr, now: PhysTick) {
        let HilMsg::Hello {
            nj,
            h,
            deployment_hash,
        } = msg
        else {
            return;
        };
        if hdr.session_id != 0 {
            self.core.stats_mut().rx_invalid += 1;
            return;
        }
        if *nj as usize != NJ || *h as usize != H {
            self.send(
                from,
                &HilMsg::Bye {
                    reason: BYE_SHAPE_MISMATCH,
                },
            );
            return;
        }
        if *deployment_hash != self.core.deployment_hash() {
            self.send(
                from,
                &HilMsg::Bye {
                    reason: BYE_HASH_MISMATCH,
                },
            );
            return;
        }
        self.peer = Some(from);
        // A fresh session per `Hello` (design note section 7.7). Accepting a `Hello` rewinds
        // `last_seq`, so without this a recorded `Hello` + `Command` pair would replay: the
        // rewind would make the recorded commands look new again. Under a fresh id they fail
        // the `hdr.session_id` check in `accept` instead, and never reach the plane.
        self.session_counter = self.session_counter.wrapping_add(1);
        self.session_id = self.next_session_id();
        self.tx_seq = 0;
        // A slot left over from the previous session would otherwise time a round trip that
        // spans the boundary.
        self.sent = [(u64::MAX, self.started); RTT_SLOTS];
        // The controller keeps one `seq` counter across the handshake, so the session starts
        // from the `Hello`'s own `seq` rather than from zero.
        self.last_seq = hdr.seq;
        let (rate_num, rate_den) = self.core.control_rate();
        self.send(
            from,
            &HilMsg::HelloAck {
                rate_num,
                rate_den,
                start_tick: now.0,
            },
        );
    }

    /// A session id for the session about to open. It is not a secret (the tag is), it only has
    /// to be *fresh*, so that a datagram authenticated under an earlier session fails the
    /// `hdr.session_id` check. Never `0` (`Hello`'s reserved value) and never the current id.
    fn next_session_id(&self) -> u64 {
        // No RNG (spec 3.4 forbids a global one): the wall clock, the port and the per-link
        // counter are enough to not repeat, and repeating is all that matters here.
        let id = (super::stats::wall_ns()
            ^ (u64::from(self.local_addr.port()) << 48)
            ^ self.session_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            | 1;
        // Bit 0 stays set either way, so neither branch can produce `0`.
        if id == self.session_id {
            id ^ 2
        } else {
            id
        }
    }

    fn observe_rtt(&mut self, obs_tick: PhysTick) {
        let (tick, at) = self.sent[obs_tick.0 as usize % RTT_SLOTS];
        if tick == obs_tick.0 {
            let ns = at.elapsed().as_nanos().min(i64::MAX as u128) as i64;
            self.core.stats_mut().observe_rtt(ns);
        }
    }

    fn send(&mut self, to: SocketAddr, msg: &HilMsg<NJ, H>) {
        self.tx_seq += 1;
        let mono = self.started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
        let mut buf = std::mem::take(&mut self.tx);
        wire::encode(&self.key, self.session_id, self.tx_seq, mono, msg, &mut buf);
        // A send failure is the network's business, not the control loop's: the tick must not
        // stall and the plane's decision is already made.
        let _ = self.sock.send_to(&buf, to);
        self.tx = buf;
    }
}
