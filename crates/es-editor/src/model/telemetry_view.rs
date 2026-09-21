//! Telemetry tab view-model (spec 23.3): what the editor keeps from a message stream.
//!
//! The editor is a **client** of a running process (spec 23.1) — it hosts nothing. Messages
//! arrive through a [`Source`] closure so the transport stays swappable: today a canned
//! `Vec` in tests or a decoded buffer, tomorrow the `es_telemetry::transport` client, without
//! this file changing.
//!
//! Histories are plain capped `Vec`s, not `es_core::ring`: a viewer dropping the oldest
//! sample is not on the determinism path, and the ring is for the *producer* side.

use std::collections::BTreeMap;
use std::net::SocketAddr;

use es_telemetry::protocol::{HelloAck, Message, Payload, PerfMetrics, StreamId, PROTOCOL_VERSION};
use es_telemetry::transport::{Client, TransportError};

use crate::model::live_run::{LiveRun, RUN_STREAMS};
use crate::model::train_view::TrainView;

/// Where messages come from. `None` means "nothing right now", not "closed" — the app polls
/// once a frame.
pub type Source = Box<dyn FnMut() -> Option<Message>>;

/// One scalar series: a stream plus, for `Scalars`, which component of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SeriesKey {
    pub stream: StreamId,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub tick: u64,
    pub kind: String,
    pub fields: BTreeMap<String, String>,
}

/// Default history depth per series, and event-log depth.
pub const DEFAULT_CAP: usize = 4096;

/// Reads one [`attach`]ed [`Source`] call may make before it answers "nothing right now".
///
/// `Client::try_recv` reads at most one 4 kB chunk per call and reports `WouldBlock` both for
/// *nothing yet* and for *this message is still arriving*. One read per call would drain 4 kB
/// per repaint — at ten repaints a second, less than half of what a run publishes, and an
/// image frame alone is tens of kilobytes — so the producer would drop the rest on the floor
/// (`docs/design/telemetry-protocol.md` §6). Sixty-four reads is "enough to finish a frame";
/// when there is genuinely nothing on the socket they are 64 non-blocking reads that return
/// immediately, once per repaint.
const READS_PER_POLL: usize = 64;

#[derive(Debug)]
pub struct TelemetryModel {
    cap: usize,
    /// `(tick, value)` per series, oldest first, at most `cap` entries.
    pub series: BTreeMap<SeriesKey, Vec<(u64, f64)>>,
    /// Latest metrics frame (spec 12.4). `None` until one arrives — never a fabricated zero.
    pub metrics: Option<PerfMetrics>,
    pub events: Vec<Event>,
    /// Messages ingested since construction, including ones this model keeps nothing from.
    pub received: u64,
    /// `execution_hash` of the observed run (spec 5.3), once the server states it.
    pub execution_hash: Option<[u8; 32]>,
    /// The Run tab's view of the same stream (packet M7/E4): an `es eval run --telemetry`
    /// folded back into the `CellRow`s and `Timeline`s a finished run has.
    pub live: LiveRun,
    /// The Live tab's Training section (packet M7/E7): the same stream's learning curve, its
    /// checkpoint marks and the tensor the network is fitting.
    pub train: TrainView,
}

impl Default for TelemetryModel {
    fn default() -> Self {
        Self::new(DEFAULT_CAP)
    }
}

impl TelemetryModel {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            series: BTreeMap::new(),
            metrics: None,
            events: Vec::new(),
            received: 0,
            execution_hash: None,
            live: LiveRun::default(),
            train: TrainView::default(),
        }
    }

    /// Drain up to `max` messages from a source. Bounded so one slow frame cannot be spent
    /// draining a backlog (spec 23.3: the editor runs on a budget).
    pub fn pump(&mut self, source: &mut Source, max: usize) -> usize {
        let mut n = 0;
        while n < max {
            match source() {
                Some(msg) => {
                    self.ingest(&msg);
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    pub fn ingest(&mut self, msg: &Message) {
        self.received += 1;
        self.live.ingest(msg);
        self.train.ingest(msg);
        let frame = match msg {
            Message::HelloAck(ack) => {
                self.execution_hash = ack.execution_hash;
                return;
            }
            Message::Frame(f) => f,
            _ => return,
        };
        let tick = frame.tick.0;
        match &frame.payload {
            Payload::Scalar(v) => self.push(frame.stream, 0, tick, *v),
            Payload::Scalars(vs) => {
                for (i, v) in vs.iter().enumerate() {
                    self.push(frame.stream, i, tick, *v);
                }
            }
            Payload::Metrics(m) => self.metrics = Some(*m),
            Payload::Event { kind, fields } => {
                if self.events.len() == self.cap {
                    self.events.remove(0);
                }
                self.events.push(Event {
                    tick,
                    kind: kind.clone(),
                    fields: fields.clone(),
                });
            }
            // Tensor and Image frames belong to the 3D view and the image tab, which take
            // their pixels from the observation plan (spec 23.3) rather than from the wire.
            Payload::Tensor { .. } | Payload::Image { .. } => {}
        }
    }

    fn push(&mut self, stream: StreamId, index: usize, tick: u64, value: f64) {
        let series = self.series.entry(SeriesKey { stream, index }).or_default();
        if series.len() == self.cap {
            series.remove(0);
        }
        series.push((tick, value));
    }

    /// The spec 12.4 metric set as a table, in a fixed order, every row always present.
    /// A single `step/s` figure is forbidden (spec 12.4): the rows are independent numbers,
    /// and an unmeasured one is `None`, not `0`. End-to-end latency is two rows, p50 and p95.
    pub fn metric_rows(&self) -> Vec<(&'static str, Option<f64>)> {
        let m = self.metrics.unwrap_or_default();
        vec![
            ("physics_steps_per_sec", m.physics_steps_per_sec),
            ("camera_frames_per_sec", m.camera_frames_per_sec),
            ("pixels_per_sec", m.pixels_per_sec),
            ("observation_gb_per_sec", m.observation_gb_per_sec),
            ("policy_inferences_per_sec", m.policy_inferences_per_sec),
            ("actions_per_sec", m.actions_per_sec),
            ("p50_end_to_end_latency", m.p50_end_to_end_latency),
            ("p95_end_to_end_latency", m.p95_end_to_end_latency),
            ("gpu_memory_peak", m.gpu_memory_peak),
            ("chunk_underrun_rate", m.chunk_underrun_rate),
        ]
    }

    /// Latest value of every series, for the stream table.
    pub fn latest(&self) -> Vec<(SeriesKey, u64, f64)> {
        self.series
            .iter()
            .filter_map(|(k, v)| v.last().map(|(t, x)| (*k, *t, *x)))
            .collect()
    }
}

/// A [`Source`] over a fixed message list — the canned stream the tests feed, and what
/// `es-editor` shows until something is attached.
pub fn replay(messages: Vec<Message>) -> Source {
    let mut it = messages.into_iter();
    Box::new(move || it.next())
}

/// Attaches to a running producer — `es eval run --telemetry <addr>` — and becomes a
/// [`Source`] (spec 23.1: the editor is a client of a running process, it hosts nothing).
///
/// `addr` is `host:port` and an empty `token` is no token, which is the local-development
/// shape spec 25.1 describes. Parsing lives here rather than in `app.rs` so that the field's
/// errors are the model's, with a test (spec 28.10 rule 3).
///
/// The handshake's own `HelloAck` is handed back as the first message, so
/// [`TelemetryModel::execution_hash`] is filled from the run's identity (spec 5.3) exactly as
/// it is from a replayed stream. After that the closure reads until one whole message is in
/// hand or [`READS_PER_POLL`] reads have come back empty: `None` means "nothing right now",
/// which is also what a closed connection looks like — the editor keeps what it has rather
/// than clearing the tab over a dropped socket.
pub fn attach(addr: &str, token: &str) -> Result<Source, String> {
    connect(addr, token).map(source_of)
}

/// The blocking half of [`attach`] - the TCP connect, the handshake and the subscription -
/// which a caller may run off the UI thread. Both legs are bounded by `es_telemetry`'s client
/// timeouts, so on a machine where a refused connect takes seconds (Windows) the caller waits
/// for that bound and not forever.
pub fn connect(addr: &str, token: &str) -> Result<Client, String> {
    let socket: SocketAddr = addr
        .trim()
        .parse()
        .map_err(|e| format!("{addr:?} is not a host:port address: {e}"))?;
    let token = (!token.trim().is_empty()).then(|| token.trim().to_owned());
    let mut client = Client::connect(socket, token, "es-editor").map_err(|e| match e {
        TransportError::Rejected { reason } => format!("{addr} refused the connection: {reason}"),
        other => format!("{addr}: {other}"),
    })?;
    client
        .subscribe(RUN_STREAMS.to_vec())
        .map_err(|e| format!("{addr}: subscribing: {e}"))?;
    Ok(client)
}

/// The non-blocking half: a connected client as the [`Source`] the tab pumps.
pub fn source_of(mut client: Client) -> Source {
    let mut ack = Some(Message::HelloAck(HelloAck {
        version: PROTOCOL_VERSION,
        session_id: client.session_id,
        execution_hash: client.execution_hash,
    }));
    Box::new(move || {
        if let Some(ack) = ack.take() {
            return Some(ack);
        }
        for _ in 0..READS_PER_POLL {
            match client.try_recv() {
                Ok(msg) => return Some(msg),
                // Either nothing has arrived or a message is half here; the only way to tell
                // them apart is to read again (see [`READS_PER_POLL`]).
                Err(TransportError::WouldBlock) => {}
                // A closed or broken connection is "nothing right now" forever: the tab keeps
                // what it has rather than clearing itself over a dropped socket.
                Err(_) => return None,
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_core::PhysTick;
    use es_telemetry::protocol::{Frame, HelloAck, PROTOCOL_VERSION};

    fn frame(tick: u64, stream: u32, payload: Payload) -> Message {
        Message::Frame(Frame {
            tick: PhysTick(tick),
            wall_ns: tick * 1000,
            stream: StreamId(stream),
            payload,
        })
    }

    fn canned() -> Vec<Message> {
        vec![
            Message::HelloAck(HelloAck {
                version: PROTOCOL_VERSION,
                session_id: 1,
                execution_hash: Some([9u8; 32]),
            }),
            frame(1, 0, Payload::Scalar(0.5)),
            frame(2, 0, Payload::Scalar(0.25)),
            frame(2, 1, Payload::Scalars(vec![1.0, 2.0])),
            frame(
                3,
                5,
                Payload::Event {
                    kind: "safety_violation".into(),
                    fields: BTreeMap::from([("limit".into(), "joint_1".into())]),
                },
            ),
            frame(
                4,
                6,
                Payload::Metrics(PerfMetrics {
                    physics_steps_per_sec: Some(1200.0),
                    policy_inferences_per_sec: Some(30.0),
                    ..Default::default()
                }),
            ),
            Message::Ping,
        ]
    }

    #[test]
    fn canned_frames_fill_every_table() {
        let mut m = TelemetryModel::default();
        let mut src = replay(canned());
        assert_eq!(m.pump(&mut src, 100), 7);
        assert_eq!(m.pump(&mut src, 100), 0, "source is drained");

        assert_eq!(m.execution_hash, Some([9u8; 32]));
        assert_eq!(
            m.series[&SeriesKey {
                stream: StreamId(0),
                index: 0
            }],
            vec![(1, 0.5), (2, 0.25)]
        );
        // `Scalars` fans out into one series per component.
        assert_eq!(m.series.len(), 3);
        assert_eq!(
            m.series[&SeriesKey {
                stream: StreamId(1),
                index: 1
            }],
            vec![(2, 2.0)]
        );
        assert_eq!(m.events.len(), 1);
        assert_eq!(m.events[0].kind, "safety_violation");
        assert_eq!(m.events[0].fields["limit"], "joint_1");
        assert_eq!(m.latest().len(), 3);
    }

    #[test]
    fn metric_table_is_the_spec_12_4_set_never_a_single_step_per_sec() {
        let mut m = TelemetryModel::default();
        let rows = m.metric_rows();
        assert_eq!(rows.len(), 10, "9 metrics, latency split into p50 and p95");
        assert!(rows.iter().all(|(_, v)| v.is_none()), "no fabricated zeros");
        assert!(!rows.iter().any(|(name, _)| *name == "steps_per_sec"));

        for msg in canned() {
            m.ingest(&msg);
        }
        let rows = m.metric_rows();
        assert_eq!(rows[0], ("physics_steps_per_sec", Some(1200.0)));
        assert_eq!(rows[4], ("policy_inferences_per_sec", Some(30.0)));
        assert_eq!(rows[8], ("gpu_memory_peak", None));
    }

    #[test]
    fn history_is_capped_oldest_first() {
        let mut m = TelemetryModel::new(3);
        for t in 0..10 {
            m.ingest(&frame(t, 0, Payload::Scalar(t as f64)));
        }
        let s = &m.series[&SeriesKey {
            stream: StreamId(0),
            index: 0,
        }];
        assert_eq!(*s, vec![(7, 7.0), (8, 8.0), (9, 9.0)]);
    }

    #[test]
    fn attach_refuses_what_is_not_an_address_before_opening_a_socket() {
        let Err(e) = attach("not-an-address", "") else {
            panic!("no socket is opened for something that is not an address");
        };
        assert!(e.contains("host:port"), "{e}");
    }

    /// A real loopback round trip: the handshake's identity arrives as the first message, so
    /// the tab shows the run's `execution_hash` without a second code path (spec 5.3).
    #[test]
    fn attach_speaks_to_a_server_and_hands_back_the_handshake() {
        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let server =
            es_telemetry::transport::Server::bind(addr, Some("s3cret".to_owned())).expect("bind");
        let bound = server.local_addr().to_string();

        let Err(e) = attach(&bound, "wrong") else {
            panic!("a bad token is refused (spec 25.1)");
        };
        assert!(e.contains("refused the connection"), "{e}");

        let mut source = attach(&bound, "s3cret").expect("the right token connects");
        let mut model = TelemetryModel::default();
        assert_eq!(model.pump(&mut source, 8), 1, "the HelloAck");
        assert_eq!(model.execution_hash, None, "the run states none");

        // The subscription is the four run streams, so a published frame comes back.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while model.series.is_empty() && std::time::Instant::now() < deadline {
            server.publish(es_telemetry::protocol::Frame {
                tick: PhysTick(3),
                wall_ns: 0,
                stream: StreamId(2),
                payload: Payload::Scalars(vec![0.0, 3.0, 0.0, 0.0]),
            });
            model.pump(&mut source, 8);
        }
        assert!(!model.series.is_empty(), "no frame arrived on stream 2");

        // One `Source` call finishes a whole message, not 4 kB of one: a 32x32 image is some
        // 13 kB of JSON, and a viewer that took four repaints over it would have the rest
        // dropped by the producer's bounded queue.
        server.publish(es_telemetry::protocol::Frame {
            tick: PhysTick(4),
            wall_ns: 0,
            stream: StreamId(4),
            payload: Payload::Image {
                w: 32,
                h: 32,
                format: "rgb8".to_owned(),
                bytes: vec![3u8; 32 * 32 * 3],
            },
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while model.live.image().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(
                model.pump(&mut source, 1) <= 1,
                "one message per pump was asked for"
            );
        }
        let image = model.live.image().expect("the image frame arrived whole");
        assert_eq!((image.width, image.height), (32, 32));
    }

    #[test]
    fn pump_stops_at_the_budget() {
        let mut m = TelemetryModel::default();
        let mut src = replay(canned());
        assert_eq!(m.pump(&mut src, 2), 2);
        assert_eq!(m.received, 2);
    }
}
