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

use es_telemetry::protocol::{Message, Payload, PerfMetrics, StreamId};

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
/// `es-editor` runs on until `es_telemetry::transport` lands.
pub fn replay(messages: Vec<Message>) -> Source {
    let mut it = messages.into_iter();
    Box::new(move || it.next())
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
    fn pump_stops_at_the_budget() {
        let mut m = TelemetryModel::default();
        let mut src = replay(canned());
        assert_eq!(m.pump(&mut src, 2), 2);
        assert_eq!(m.received, 2);
    }
}
