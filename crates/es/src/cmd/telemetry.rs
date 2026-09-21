//! `--telemetry`: the one place a running command becomes a wire `Frame` (spec 23.1, spec
//! 23.3; packets M7/E4, M7/E7).
//!
//! It lives in `es` and not in `es-eval`, `es-data` or `es-telemetry` because those are all
//! layer 10 and spec 4.2 forbids a same-layer dependency: each of them calls a **closure**
//! with what it already had (`INV-17` — a sink is a closure, not an eighth extension point),
//! and this crate, the one that links all three, turns what they say into frames.
//!
//! One [`Publisher`] is one bound socket. `es loop cycle` binds it once and hands the same one
//! to every in-process stage, setting [`Publisher::stage`] as it goes, so a viewer on one
//! address sees collect, the expert gate, training and the evaluation as one run; the
//! standalone commands bind their own. Every stream-1 event carries the stage it came from.
//!
//! **Nothing here waits.** `Server::publish` is a `try_send` into each client's bounded queue
//! and drops on a full one, so a viewer that stops reading loses frames and the run does not
//! notice (`docs/design/telemetry-protocol.md` §6).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use es_data::collect::CollectEvent;
use es_data::ActionSourceCode;
use es_eval::runner::{EventSource, RunEvent, StepEvent};
use es_telemetry::protocol::{Frame, Payload, PerfMetrics, StreamId};
use es_telemetry::transport::Server;

use crate::error::CliError;

/// The stream ids of `docs/design/telemetry-protocol.md` "Producers". Data, not schema:
/// `es_telemetry::protocol` is frozen at its version, and a stream id is a number on the wire
/// that producer and consumer agree on in a design note.
pub(crate) const STREAM_EVENTS: StreamId = StreamId(1);
pub(crate) const STREAM_TICKS: StreamId = StreamId(2);
pub(crate) const STREAM_METRICS: StreamId = StreamId(3);
pub(crate) const STREAM_IMAGE: StreamId = StreamId(4);
/// The training curve (packet M7/E7): `[step, loss, lr, samples_per_s]` per progress line.
pub(crate) const STREAM_TRAIN: StreamId = StreamId(5);

/// The `stage` field every stream-1 event carries. A cycle's stages arrive on one socket, so
/// this is what tells them apart; a standalone command names itself.
pub(crate) const STAGE_COLLECT: &str = "collect";
pub(crate) const STAGE_EXPERT_GATE: &str = "expert-gate";
pub(crate) const STAGE_TRAIN: &str = "train";
pub(crate) const STAGE_EVAL: &str = "eval";
pub(crate) const STAGE_SHOWCASE: &str = "showcase";
pub(crate) const STAGE_CYCLE: &str = "cycle";

/// What one `--telemetry` triple says. Parsed by each command's own parser, because each has
/// one already; bound by [`Publisher::bind`], which is the part that must not differ.
#[derive(Clone, Debug, Default)]
pub(crate) struct TelemetryArgs {
    pub addr: Option<SocketAddr>,
    pub token: Option<String>,
    pub image_every: u64,
}

impl TelemetryArgs {
    /// `--telemetry <addr>`, refused by name when it is not an address.
    pub(crate) fn parse_addr(value: &str, help: &str) -> Result<SocketAddr, CliError> {
        value.parse().map_err(|e| {
            CliError::Usage(format!(
                "--telemetry {value:?} is not an address: {e}\n\n{help}"
            ))
        })
    }

    /// `--telemetry-image-every <N>`.
    pub(crate) fn parse_image_every(value: &str, help: &str) -> Result<u64, CliError> {
        value.parse().map_err(|_| {
            CliError::Usage(format!(
                "--telemetry-image-every {value:?} is not a number\n\n{help}"
            ))
        })
    }

    /// The bound publisher, or `None` when no address was given. Bind **before** anything is
    /// opened: a viewer that attaches on the printed address is then subscribed before the
    /// first episode, which is the difference between watching a run and reading its tail.
    pub(crate) fn bind(&self, stage: &'static str) -> Result<Option<Publisher>, CliError> {
        match self.addr {
            Some(addr) => {
                Publisher::bind(addr, self.token.clone(), self.image_every, stage).map(Some)
            }
            None => Ok(None),
        }
    }
}

/// The one place an event becomes a wire [`Frame`].
pub(crate) struct Publisher {
    server: Server,
    /// The address as it was asked for, for the closing summary line.
    addr: SocketAddr,
    /// Control ticks between two stream-4 images; `0` publishes none.
    image_every: u64,
    /// Which stage of spec 13.1's loop is speaking. Every stream-1 event carries it.
    stage: &'static str,
    /// Wall clock of the open episode, for the spec 12.4 rates at its end.
    began: Instant,
    /// Wall clock of the open stage, for `stage.end`.
    stage_began: Instant,
    /// Observations seen in the open episode, for the image rate limit.
    observations: u64,
    /// The last step this run published. Kept whole rather than as its tick alone because the
    /// tick's type lives in `es-core`, which this crate takes only as a dev-dependency: a
    /// `StepEvent` is what `es-eval` hands over, so nothing here has to name it.
    last: Option<StepEvent>,
}

impl Publisher {
    /// Binds the server and prints the address it got, so `--telemetry 127.0.0.1:0` names its
    /// port.
    pub(crate) fn bind(
        addr: SocketAddr,
        token: Option<String>,
        image_every: u64,
        stage: &'static str,
    ) -> Result<Self, CliError> {
        let server = Server::bind(addr, token)
            .map_err(|e| CliError::Runtime(format!("--telemetry {addr}: {e}")))?;
        println!("telemetry: {}", server.local_addr());
        Ok(Self {
            server,
            addr,
            image_every,
            stage,
            began: Instant::now(),
            stage_began: Instant::now(),
            observations: 0,
            last: None,
        })
    }

    /// Non-blocking by construction (`Server::publish`): this is the whole of what a run pays
    /// for telemetry on the control path.
    fn send(&self, frame: Frame) {
        self.server.publish(frame);
    }

    fn event(&self, kind: &str, mut fields: BTreeMap<String, String>) {
        fields.insert("stage".to_owned(), self.stage.to_owned());
        self.send(Frame {
            // Where the run had got to when this happened; tick zero before the first step.
            tick: self.last.map_or_else(Default::default, |e| e.tick),
            wall_ns: wall_ns(),
            stream: STREAM_EVENTS,
            payload: Payload::Event {
                kind: kind.to_owned(),
                fields,
            },
        });
    }

    /// Opens a stage of `es loop cycle`. Every event until the next one carries this name.
    pub(crate) fn stage_begin(&mut self, stage: &'static str) {
        self.stage = stage;
        self.stage_began = Instant::now();
        self.observations = 0;
        self.event("stage.begin", fields([("name", stage.to_owned())]));
    }

    /// Closes it, with its wall-clock and the exit code the stage returned.
    pub(crate) fn stage_end(&mut self, code: u8) {
        let stage = self.stage;
        self.event(
            "stage.end",
            fields([
                ("name", stage.to_owned()),
                (
                    "seconds",
                    format!("{:.3}", self.stage_began.elapsed().as_secs_f64()),
                ),
                ("code", code.to_string()),
            ]),
        );
    }

    // --- `es eval run` (packet M7/E4) ----------------------------------------------------

    /// The `es_eval::runner::RunSink` body.
    pub(crate) fn on(&mut self, event: RunEvent<'_>) {
        match event {
            RunEvent::CellBegin {
                cell,
                suite,
                seed,
                episode,
            } => {
                self.began = Instant::now();
                self.observations = 0;
                self.event(
                    "cell.begin",
                    fields([
                        ("cell", cell.to_owned()),
                        ("suite", suite.to_owned()),
                        ("seed", seed.to_string()),
                        ("episode", episode.to_string()),
                    ]),
                );
            }
            RunEvent::Observation {
                cell: _,
                tick,
                shape,
                bytes,
            } => {
                // Exactly what the renderer wrote, when that is RGB8; anything else is not an
                // `Rgb8` image and is not relabelled into one (INV-14).
                let [h, w, c] = shape else { return };
                if *c != 3 || bytes.len() != (h * w * c) as usize {
                    self.observations += 1;
                    return;
                }
                let (w, h) = (*w as u32, *h as u32);
                if self.next_image() {
                    self.send(Frame {
                        tick,
                        wall_ns: wall_ns(),
                        stream: STREAM_IMAGE,
                        payload: rgb8(w, h, bytes),
                    });
                }
            }
            RunEvent::Tick { cell: _, event } => {
                self.last = Some(event);
                self.send(Frame {
                    tick: event.tick,
                    wall_ns: wall_ns(),
                    stream: STREAM_TICKS,
                    // The `StepEvent` `events.json` records, as four numbers: the frame
                    // index, the physics tick, the action's source and the violation bits.
                    payload: Payload::Scalars(vec![
                        event.frame as f64,
                        event.tick.0 as f64,
                        f64::from(source_code(event.source)),
                        f64::from(event.events),
                    ]),
                });
            }
            RunEvent::CellEnd { cell, end } => {
                self.event(
                    "cell.end",
                    fields([
                        ("cell", cell.to_owned()),
                        ("outcome", end.outcome.clone()),
                        ("steps", end.steps.to_string()),
                        ("frames", end.frames.to_string()),
                        ("traj", end.traj.to_string()),
                    ]),
                );
                // The spec 12.4 set with only the fields this run measured. An unmeasured
                // metric stays `None`: a zero would be a number nobody took.
                let secs = self.began.elapsed().as_secs_f64();
                let per_sec = |n: u64| (secs > 0.0).then(|| n as f64 / secs);
                self.send(Frame {
                    tick: self.last.map_or_else(Default::default, |e| e.tick),
                    wall_ns: wall_ns(),
                    stream: STREAM_METRICS,
                    payload: Payload::Metrics(PerfMetrics {
                        actions_per_sec: per_sec(end.steps),
                        policy_inferences_per_sec: per_sec(end.inferences),
                        chunk_underrun_rate: Some(end.chunk_underrun_rate),
                        ..PerfMetrics::default()
                    }),
                });
            }
            RunEvent::SuiteEnd { suite, results } => {
                let mut f = fields([("suite", suite.to_owned())]);
                for r in results {
                    f.insert("n_episodes".to_owned(), r.n_episodes.to_string());
                    // The `MetricValue` itself, as JSON: a viewer rebuilds the report's own
                    // value -- `Scalar`, `Histogram` or `Unavailable` -- instead of a string
                    // that has already decided it is a number (spec 10.3).
                    if let Ok(v) = serde_json::to_string(&r.value) {
                        f.insert(format!("metric.{}", r.metric.name()), v);
                    }
                }
                self.event("suite.end", f);
            }
        }
    }

    // --- `es loop collect` (packet M7/E7) ------------------------------------------------

    /// The `es_data::collect::CollectSink` body: an episode is a row of the Run tab's table
    /// exactly as an evaluation's cell is.
    pub(crate) fn on_collect(&mut self, event: CollectEvent) {
        match event {
            CollectEvent::EpisodeBegin { episode, seed } => {
                self.began = Instant::now();
                self.observations = 0;
                self.event(
                    "episode.begin",
                    fields([("episode", episode.to_string()), ("seed", seed.to_string())]),
                );
            }
            CollectEvent::Tick {
                episode: _,
                frame,
                tick,
                source,
                events,
            } => {
                let record = StepEvent {
                    frame: u64::from(frame),
                    tick,
                    source: event_source(source),
                    events,
                };
                self.last = Some(record);
                self.send(Frame {
                    tick,
                    wall_ns: wall_ns(),
                    stream: STREAM_TICKS,
                    payload: Payload::Scalars(vec![
                        f64::from(frame),
                        tick.0 as f64,
                        f64::from(source_code(record.source)),
                        f64::from(events),
                    ]),
                });
            }
            CollectEvent::EpisodeEnd {
                episode,
                outcome,
                steps,
            } => self.event(
                "episode.end",
                fields([
                    ("episode", episode.to_string()),
                    ("outcome", format!("{outcome:?}")),
                    ("steps", steps.to_string()),
                ]),
            ),
        }
    }

    /// One rendered observation tile, rate-limited on its own clock (spec 23.3: the state
    /// stream is every tick, the image stream every Nth, because one 96x96 frame is some
    /// 100 kB of JSON).
    ///
    /// `shape` is the renderer's `[h, w, c]`; anything that is not three-channel `u8` is not
    /// an `Rgb8` image and is not relabelled into one (INV-14).
    ///
    /// Only the render build has a frame source to call this from (`es loop collect
    /// --telemetry-image-every` says so and publishes none without `--frames`), so a build
    /// without the feature never reaches it.
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) fn observation(&mut self, shape: [usize; 3], bytes: &[u8]) {
        let [h, w, c] = shape;
        if c != 3 || bytes.len() != h * w * c {
            self.observations += 1;
            return;
        }
        if self.next_image() {
            self.send(Frame {
                tick: self.last.map_or_else(Default::default, |e| e.tick),
                wall_ns: wall_ns(),
                stream: STREAM_IMAGE,
                payload: rgb8(w as u32, h as u32, bytes),
            });
        }
    }

    /// Counts one observation and says whether this is one of the published ones.
    fn next_image(&mut self) -> bool {
        let n = self.observations;
        self.observations += 1;
        self.image_every != 0 && n % self.image_every == 0
    }

    // --- `es train` (packet M7/E7) --------------------------------------------------------

    /// One `{"progress": {...}}` line of the trainer's stdout, as
    /// `[step, loss, lr, samples_per_s]` on stream 5.
    ///
    /// The frame's `tick` is zero and the step is the first scalar: an optimizer step is not a
    /// physics tick, and putting one in the other's field would be a number nobody can read.
    pub(crate) fn progress(&self, progress: &serde_json::Value) {
        let at = |k: &str| progress.get(k).and_then(serde_json::Value::as_f64);
        let (Some(step), Some(loss)) = (at("step"), at("loss")) else {
            return;
        };
        self.send(Frame {
            // `last` is `None` on a training run, so this is tick zero; naming the type would
            // mean taking `es-core` as a dependency of this crate for one constructor.
            tick: self.last.map_or_else(Default::default, |e| e.tick),
            wall_ns: wall_ns(),
            stream: STREAM_TRAIN,
            payload: Payload::Scalars(vec![
                step,
                loss,
                at("lr").unwrap_or(f64::NAN),
                at("samples_per_s").unwrap_or(f64::NAN),
            ]),
        });
    }

    /// How long this training run is, before the first optimizer step.
    ///
    /// The document's own `[run] steps` and not a number the trainer invents: it is what the
    /// plan's `--checkpoint-at` caps the run at, and it is knowable before Python starts, so
    /// an editor that attaches can draw an ETA from the first progress line onward.
    pub(crate) fn train_begin(&self, total_steps: u32) {
        self.event(
            "train.begin",
            fields([("total_steps", total_steps.to_string())]),
        );
    }

    /// A checkpoint mark was packed: the step and the bundle's own spec 5.3 `policy_hash`.
    pub(crate) fn checkpoint(&self, step: &str, policy_hash: &str) {
        self.event(
            "checkpoint",
            fields([
                ("step", step.to_owned()),
                ("policy_hash", policy_hash.to_owned()),
            ]),
        );
    }

    /// The tensor the network is fitting, as `train_act.py --sample-every` wrote it beside
    /// `metrics/`: `<name>.bin` holding `h * w * 3` `Rgb8` bytes and `<name>.json` its shape.
    ///
    /// A file that is not there, or does not hold what its sidecar says, is skipped in
    /// silence: a sample image is a picture, and a run must not fail over one.
    pub(crate) fn sample(&mut self, path: &Path) {
        let Ok(meta) = std::fs::read_to_string(path.with_extension("json")) else {
            return;
        };
        let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta) else {
            return;
        };
        let dim = |i: usize| meta["shape"].get(i).and_then(serde_json::Value::as_u64);
        let (Some(h), Some(w), Some(3)) = (dim(0), dim(1), dim(2)) else {
            return;
        };
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        if bytes.len() as u64 != h * w * 3 {
            return;
        }
        self.send(Frame {
            tick: self.last.map_or_else(Default::default, |e| e.tick),
            wall_ns: wall_ns(),
            stream: STREAM_IMAGE,
            payload: rgb8(w as u32, h as u32, &bytes),
        });
    }

    /// What the run published and what the subscribers lost, printed once at the end.
    pub(crate) fn summary(&self) -> String {
        let stats = self.server.stats();
        let sent: u64 = stats.clients.values().map(|c| c.sent).sum();
        let dropped: u64 = stats.clients.values().map(|c| c.dropped).sum();
        format!(
            "telemetry: {} closed after {sent} frame(s) delivered to {} client(s), \
             {dropped} dropped",
            self.addr,
            stats.clients.len()
        )
    }
}

/// One `Rgb8` image payload. The bytes are the renderer's own (INV-14): nothing here converts
/// a colour space or resamples.
fn rgb8(w: u32, h: u32, bytes: &[u8]) -> Payload {
    Payload::Image {
        w,
        h,
        format: "rgb8".to_owned(),
        bytes: bytes.to_vec(),
    }
}

fn wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

fn fields<const N: usize>(pairs: [(&str, String); N]) -> BTreeMap<String, String> {
    pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}

/// The wire code of an action's source: `es_data::ActionSourceCode::as_i64`'s table (spec
/// 13.2), so the dataset column and the telemetry stream number the same four outcomes the
/// same way.
fn source_code(source: EventSource) -> u32 {
    match source {
        EventSource::Policy => 0,
        EventSource::Clamped => 1,
        EventSource::Fallback => 2,
        EventSource::Human => 3,
    }
}

/// The same table read the other way, for the collection path — whose per-frame provenance is
/// the dataset's own `action_source` column.
fn event_source(code: ActionSourceCode) -> EventSource {
    match code {
        ActionSourceCode::Policy => EventSource::Policy,
        ActionSourceCode::Clamped => EventSource::Clamped,
        ActionSourceCode::Fallback => EventSource::Fallback,
        ActionSourceCode::Human => EventSource::Human,
    }
}
