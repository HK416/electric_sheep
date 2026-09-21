//! The Run tab, live (spec 23.1, spec 23.3; packet M7/E4): a run that is still going, folded
//! into the shapes the finished one already uses.
//!
//! `es eval run --telemetry <addr>` publishes what it is doing on four streams
//! (`docs/design/telemetry-protocol.md`, "Producers"). This file turns them back into
//! [`CellRow`]s and a [`Timeline`] — E1's own types — so the Run tab has **one** table, one
//! strip and one set of column headers whether the run finished yesterday or is on its third
//! episode right now. Nothing draws here and nothing in `app.rs` decides: the fold is tested
//! against what [`crate::model::run_view::RunView`] makes of the same run on disk.
//!
//! What a live run cannot have is a verdict: `report.json` is written after the last suite, so
//! there is no `acceptance()` here and [`LiveRun::status`] says what is running instead.

use std::collections::BTreeMap;

use es_core::PhysTick;
use es_eval::runner::{EventSource, StepEvent};
use es_ir::evaluation::MetricValue;
use es_telemetry::protocol::{Message, Payload, StreamId};

use crate::model::image_view::Rgb8Image;
use crate::model::run_view::{decode_events, CellRow, FirstSeen, TickRow, Timeline};

/// The producer's four streams (`docs/design/telemetry-protocol.md`, "Producers"). Data and
/// not schema — `es_telemetry::protocol` is frozen at its version, and which number carries
/// what is an agreement between `es eval run --telemetry` and this file.
pub const STREAM_EVENTS: StreamId = StreamId(1);
pub const STREAM_TICKS: StreamId = StreamId(2);
pub const STREAM_METRICS: StreamId = StreamId(3);
pub const STREAM_IMAGE: StreamId = StreamId(4);

/// Every stream a viewer of a run wants, for `Client::subscribe`. Stream 5 is the training
/// curve (packet M7/E7, `crate::model::train_view`): one subscription covers a whole cycle,
/// because a cycle publishes every stage on one socket.
pub const RUN_STREAMS: [StreamId; 5] = [
    STREAM_EVENTS,
    STREAM_TICKS,
    STREAM_METRICS,
    STREAM_IMAGE,
    crate::model::train_view::STREAM_TRAIN,
];

/// One episode as the wire described it.
#[derive(Clone, Debug, Default, PartialEq)]
struct LiveCell {
    suite: String,
    seed: Option<u64>,
    records: Vec<StepEvent>,
    frames: usize,
    has_traj: bool,
    /// The `cell.end` outcome (`Success`, `Timeout`, ...); `None` while it is still running.
    outcome: Option<String>,
}

/// One suite's row of the spec 10.1 table, as `suite.end` stated it.
#[derive(Clone, Debug, Default, PartialEq)]
struct LiveSuite {
    metrics: BTreeMap<String, MetricValue>,
    n_episodes: u32,
}

/// One stage of `es loop cycle`, as `stage.begin` / `stage.end` bracketed it (packet M7/E7).
///
/// The strip above the Run tab's table is this list: a cycle publishes collect, the expert
/// gate, training and the evaluation on **one** socket, and the stage is what tells a row of
/// one from a row of another.
#[derive(Clone, Debug, PartialEq)]
pub struct StageRow {
    /// The producer's own word: `collect`, `expert-gate`, `train`, `eval`, `showcase`.
    pub name: String,
    /// Wall-clock the stage took, once it has ended.
    pub seconds: Option<f64>,
    /// The exit code the stage returned; `None` while it is still running.
    pub code: Option<u8>,
}

impl StageRow {
    pub fn running(&self) -> bool {
        self.code.is_none()
    }
}

/// A run being watched: the same rows and the same timeline the Run tab draws for a finished
/// one.
#[derive(Clone, Debug, Default)]
pub struct LiveRun {
    /// Keyed by cell name, which is also the order the table shows them in — the same order
    /// `RunView` sorts its rows into on open.
    cells: BTreeMap<String, LiveCell>,
    suites: BTreeMap<String, LiveSuite>,
    /// The episode currently running, which is what a stream-2 sample belongs to: the wire
    /// carries numbers, and `cell.begin` / `cell.end` are what name them.
    open: Option<String>,
    /// The latest observation image (stream 4), when the producer was asked for one, and how
    /// many have arrived — the sequence number a viewer keys its texture on, so one image is
    /// uploaded once and not once per repaint.
    image: Option<Rgb8Image>,
    images: u64,
    selected: Option<String>,
    /// The cycle's stages, in the order they began (packet M7/E7). Empty for a run that is
    /// one command rather than a cycle — which is what makes the strip hide itself.
    stages: Vec<StageRow>,
}

impl LiveRun {
    /// Nothing has arrived yet — the Run tab keeps showing whatever is open on disk.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Folds one message. Anything that is not one of the four streams is ignored, so a
    /// producer publishing its own streams beside these does not confuse the table.
    pub fn ingest(&mut self, msg: &Message) {
        let Message::Frame(frame) = msg else {
            return;
        };
        match (frame.stream, &frame.payload) {
            (STREAM_EVENTS, Payload::Event { kind, fields }) => self.event(kind, fields),
            (STREAM_TICKS, Payload::Scalars(v)) => self.tick(v),
            (
                STREAM_IMAGE,
                Payload::Image {
                    w,
                    h,
                    format,
                    bytes,
                },
            ) => {
                self.image = rgb8(*w, *h, format, bytes);
                self.images += 1;
            }
            _ => {}
        }
    }

    fn event(&mut self, kind: &str, fields: &BTreeMap<String, String>) {
        let get = |k: &str| fields.get(k).cloned().unwrap_or_default();
        match kind {
            "cell.begin" => {
                let name = get("cell");
                let cell = self.cells.entry(name.clone()).or_default();
                cell.suite = get("suite");
                cell.seed = fields.get("seed").and_then(|s| s.parse().ok());
                self.open = Some(name);
            }
            "cell.end" => {
                let name = get("cell");
                if let Some(cell) = self.cells.get_mut(&name) {
                    cell.frames = fields
                        .get("frames")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    cell.has_traj = fields.get("traj").is_some_and(|s| s == "true");
                    cell.outcome = Some(get("outcome"));
                }
                self.open = None;
            }
            // An episode of `es loop collect` is a row of the same table (packet M7/E7). It
            // has no suite of its own, so the stage it came from is the column's value: a
            // cycle's Run tab then reads "collect / episode-00" beside "nominal / nominal-00".
            "episode.begin" => {
                let name = episode_cell(&get("episode"));
                let cell = self.cells.entry(name.clone()).or_default();
                cell.suite = get("stage");
                cell.seed = fields.get("seed").and_then(|s| s.parse().ok());
                self.open = Some(name);
            }
            "episode.end" => {
                let name = episode_cell(&get("episode"));
                if let Some(cell) = self.cells.get_mut(&name) {
                    // A collection writes rows, not frames: the count a person wants beside
                    // the episode is the steps it recorded.
                    cell.frames = fields
                        .get("steps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    cell.has_traj = true;
                    cell.outcome = Some(get("outcome"));
                }
                self.open = None;
            }
            "stage.begin" => self.stages.push(StageRow {
                name: get("name"),
                seconds: None,
                code: None,
            }),
            "stage.end" => {
                let name = get("name");
                if let Some(row) = self.stages.iter_mut().rev().find(|s| s.name == name) {
                    row.seconds = fields.get("seconds").and_then(|s| s.parse().ok());
                    row.code = fields.get("code").and_then(|s| s.parse().ok()).or(Some(0));
                }
            }
            "suite.end" => {
                let suite = self.suites.entry(get("suite")).or_default();
                suite.n_episodes = fields
                    .get("n_episodes")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                // The report's own `MetricValue`, rebuilt from the JSON the producer sent:
                // a `Histogram` or an `Unavailable` stays what it is and never becomes a
                // number the run did not measure (spec 10.3).
                for (key, value) in fields {
                    let Some(metric) = key.strip_prefix("metric.") else {
                        continue;
                    };
                    if let Ok(v) = serde_json::from_str::<MetricValue>(value) {
                        suite.metrics.insert(metric.to_owned(), v);
                    }
                }
            }
            _ => {}
        }
    }

    /// One `[frame, tick, source, violation bits]` sample, attributed to the open episode.
    fn tick(&mut self, v: &[f64]) {
        let ([frame, tick, source, events], Some(open)) = (v, self.open.as_ref()) else {
            return;
        };
        let Some(cell) = self.cells.get_mut(open) else {
            return;
        };
        cell.records.push(StepEvent {
            frame: *frame as u64,
            tick: PhysTick(*tick as u64),
            source: event_source(*source as u32),
            events: *events as u32,
        });
    }

    /// One row per episode, in cell-name order — [`crate::model::run_view::RunView::cells`]'s
    /// own shape and order, so the table does not know which end it came from.
    pub fn cells(&self) -> Vec<CellRow> {
        self.cells
            .iter()
            .map(|(name, c)| {
                let suite = self.suites.get(&c.suite);
                CellRow {
                    name: name.clone(),
                    suite: c.suite.clone(),
                    seed: c.seed,
                    metrics: suite.map(|s| s.metrics.clone()).unwrap_or_default(),
                    n_episodes: suite.map_or(0, |s| s.n_episodes),
                    has_traj: c.has_traj,
                    frames: c.frames,
                }
            })
            .collect()
    }

    /// The table's headers: the three identity columns, then one per metric seen so far. The
    /// same rule `RunView::columns` follows, so a metric added to `es-ir` needs no change
    /// here either.
    pub fn columns(&self) -> Vec<String> {
        let mut out = vec!["cell".to_owned(), "suite".to_owned(), "seed".to_owned()];
        let metrics: std::collections::BTreeSet<&String> = self
            .suites
            .values()
            .flat_map(|s| s.metrics.keys())
            .collect();
        out.extend(metrics.into_iter().cloned());
        out
    }

    /// One episode's Safety Plane history so far, decoded exactly as `RunView::timeline`
    /// decodes `events.json` — the bits are `es_safety::EventSet::bits()` either way.
    ///
    /// The fold is repeated rather than shared because `run_view.rs` is not this packet's to
    /// change; the oracle pins the two equal for the same run, so they cannot drift quietly.
    pub fn timeline(&self, cell: &str) -> Timeline {
        let mut out = Timeline::default();
        let Some(live) = self.cells.get(cell) else {
            return out;
        };
        for record in &live.records {
            let events = decode_events(record.events);
            for kind in events.iter() {
                *out.totals.entry(kind).or_default() += 1;
                out.first.entry(kind).or_insert(FirstSeen {
                    frame: record.frame,
                    tick: record.tick.0,
                });
            }
            out.rows.push(TickRow {
                frame: record.frame,
                tick: record.tick.0,
                source: record.source,
                events,
            });
        }
        out
    }

    /// The latest observation image the producer published (stream 4), if any.
    pub fn image(&self) -> Option<&Rgb8Image> {
        self.image.as_ref()
    }

    /// Image frames seen on stream 4, decoded or not: the sequence number of [`Self::image`].
    pub fn images(&self) -> u64 {
        self.images
    }

    /// The cycle's stages, in the order they began. Empty for a run that is one command.
    pub fn stages(&self) -> &[StageRow] {
        &self.stages
    }

    pub fn select(&mut self, name: &str) {
        self.selected = Some(name.to_owned());
    }

    /// The selected row, or — while nothing has been clicked — the episode that is running,
    /// so an attached viewer shows the live strip without anyone touching the table.
    pub fn selected_cell(&self) -> Option<String> {
        self.selected
            .clone()
            .filter(|s| self.cells.contains_key(s))
            .or_else(|| self.open.clone())
    }

    /// The status line, in place of the acceptance heading a finished run has: a live run has
    /// no verdict, because `report.json` is written after the last suite (spec 10.5).
    pub fn status(&self) -> String {
        let done = self.cells.values().filter(|c| c.outcome.is_some()).count();
        match &self.open {
            Some(cell) => format!(
                "live: {cell} running, {done} of {} cell(s) finished",
                self.cells.len()
            ),
            None => format!("live: {done} of {} cell(s) finished", self.cells.len()),
        }
    }
}

/// The wire code of an action's source: `es_data::ActionSourceCode::as_i64`'s table (spec
/// 13.2), which `es eval run --telemetry` writes and this reads back. An unknown code is
/// `Policy` — the quiet one — rather than a panic on a stream from a newer producer.
fn event_source(code: u32) -> EventSource {
    match code {
        1 => EventSource::Clamped,
        2 => EventSource::Fallback,
        3 => EventSource::Human,
        _ => EventSource::Policy,
    }
}

/// The Run tab's name for one collected episode. Zero-padded so the `BTreeMap`'s key order is
/// the episode order, which is the same rule `<suite>-<NN>` follows for an evaluation's cells.
fn episode_cell(index: &str) -> String {
    match index.parse::<u32>() {
        Ok(n) => format!("episode-{n:02}"),
        Err(_) => format!("episode-{index}"),
    }
}

/// An `Image` payload as the image tab's own type, or `None` when it is not what it says it
/// is. Only `rgb8` is decoded: the producer sends the renderer's bytes unconverted (INV-14),
/// and guessing at any other layout here would be a second preprocessing implementation.
pub(crate) fn rgb8(w: u32, h: u32, format: &str, bytes: &[u8]) -> Option<Rgb8Image> {
    let (width, height) = (w as usize, h as usize);
    if format != "rgb8" || bytes.len() != width * height * 3 {
        return None;
    }
    Some(Rgb8Image {
        width,
        height,
        data: bytes.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::run_view::RunView;
    use crate::model::telemetry_view::{replay, TelemetryModel};
    use es_telemetry::protocol::Frame;
    use std::path::{Path, PathBuf};

    fn fixture() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/run")
    }

    fn frame(stream: StreamId, tick: u64, payload: Payload) -> Message {
        Message::Frame(Frame {
            tick: PhysTick(tick),
            wall_ns: 0,
            stream,
            payload,
        })
    }

    fn event(kind: &str, fields: &[(&str, String)]) -> Message {
        frame(
            STREAM_EVENTS,
            0,
            Payload::Event {
                kind: kind.to_owned(),
                fields: fields
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.clone()))
                    .collect(),
            },
        )
    }

    /// What `es eval run --telemetry` would have published while producing the committed
    /// fixture run, read back off the fixture's own artifacts: `events.json` for the ticks,
    /// `report.json` for the suite rows, `evaluation.lock` for the seeds, and the directory
    /// listing for the counts a `cell.end` carries.
    fn messages_for(dir: &Path) -> Vec<Message> {
        let report: es_ir::evaluation::EvaluationReport = serde_json::from_str(
            &std::fs::read_to_string(dir.join("report.json")).expect("report"),
        )
        .expect("report.json");
        let events: BTreeMap<String, Vec<StepEvent>> = serde_json::from_str(
            &std::fs::read_to_string(dir.join("events.json")).expect("events"),
        )
        .expect("events.json");
        let lock: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("evaluation.lock")).expect("lock"),
        )
        .expect("evaluation.lock");
        let seeds: Vec<u64> = lock["seeds"]
            .as_array()
            .expect("seeds")
            .iter()
            .map(|v| v.as_u64().expect("seed"))
            .collect();

        let mut suites: Vec<&String> = report.cells.iter().map(|c| &c.suite).collect();
        suites.dedup();
        let mut out = Vec::new();
        for suite in suites {
            for (episode, seed) in seeds.iter().enumerate() {
                let cell = format!("{suite}-{episode:02}");
                let Some(records) = events.get(&cell) else {
                    continue;
                };
                out.push(event(
                    "cell.begin",
                    &[
                        ("cell", cell.clone()),
                        ("suite", suite.clone()),
                        ("seed", seed.to_string()),
                        ("episode", episode.to_string()),
                    ],
                ));
                for r in records {
                    out.push(frame(
                        STREAM_TICKS,
                        r.tick.0,
                        Payload::Scalars(vec![
                            r.frame as f64,
                            r.tick.0 as f64,
                            f64::from(source_code(r.source)),
                            f64::from(r.events),
                        ]),
                    ));
                }
                let frames = std::fs::read_dir(dir.join("frames").join(&cell)).map_or(0, |d| {
                    d.flatten()
                        .filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
                        .count()
                });
                let traj = dir.join("traj").join(format!("{cell}.estraj")).is_file();
                out.push(event(
                    "cell.end",
                    &[
                        ("cell", cell.clone()),
                        ("outcome", "Success".to_owned()),
                        ("steps", records.len().to_string()),
                        ("frames", frames.to_string()),
                        ("traj", traj.to_string()),
                    ],
                ));
            }
            let mut fields = vec![("suite".to_owned(), suite.clone())];
            for c in report.cells.iter().filter(|c| c.suite == *suite) {
                fields.push(("n_episodes".to_owned(), c.n_episodes.to_string()));
                fields.push((
                    format!("metric.{}", c.metric.name()),
                    serde_json::to_string(&c.value).expect("metric value"),
                ));
            }
            out.push(frame(
                STREAM_EVENTS,
                0,
                Payload::Event {
                    kind: "suite.end".to_owned(),
                    fields: fields.into_iter().collect(),
                },
            ));
        }
        out
    }

    /// The producer's mapping, repeated here so the fixture's `EventSource`s go out as the
    /// numbers `es eval run --telemetry` would have sent.
    fn source_code(source: EventSource) -> u32 {
        match source {
            EventSource::Policy => 0,
            EventSource::Clamped => 1,
            EventSource::Fallback => 2,
            EventSource::Human => 3,
        }
    }

    /// Oracle 3 (packet M7/E4). The same run, once read off disk and once heard on the wire,
    /// gives the same rows, the same headers and the same timeline — which is what lets the
    /// Run tab draw a live run and a finished one with one code path.
    #[test]
    fn live_run_folds_streams_into_run_rows() {
        let dir = fixture();
        let run = RunView::open(&dir).expect("the fixture run opens");

        let mut model = TelemetryModel::default();
        let messages = messages_for(&dir);
        let n = messages.len();
        let mut source = replay(messages);
        assert_eq!(model.pump(&mut source, 10_000), n);

        let live = &model.live;
        assert_eq!(live.cells(), run.cells());
        assert_eq!(live.columns(), run.columns());
        for row in run.cells() {
            assert_eq!(
                live.timeline(&row.name),
                run.timeline(&row.name),
                "timeline of {}",
                row.name
            );
        }
        assert!(
            live.status().contains("cell(s) finished"),
            "{}",
            live.status()
        );
        println!(
            "RAN live_run_folds_streams_into_run_rows: {n} message(s), {} cell(s)",
            live.cells().len()
        );
    }

    #[test]
    fn a_tick_outside_any_cell_is_dropped_not_guessed() {
        let mut live = LiveRun::default();
        live.ingest(&frame(
            STREAM_TICKS,
            1,
            Payload::Scalars(vec![0.0, 1.0, 0.0, 0.0]),
        ));
        assert!(live.is_empty(), "a sample with no open cell names nothing");
        assert_eq!(live.timeline("nominal-00"), Timeline::default());
    }

    #[test]
    fn the_latest_image_is_the_observation_frame_and_a_mislabelled_one_is_refused() {
        let mut live = LiveRun::default();
        let px = vec![7u8; 2 * 3 * 3];
        live.ingest(&frame(
            STREAM_IMAGE,
            4,
            Payload::Image {
                w: 3,
                h: 2,
                format: "rgb8".to_owned(),
                bytes: px.clone(),
            },
        ));
        let image = live.image().expect("an rgb8 frame decodes");
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.data, px);

        // A payload whose bytes do not match its own dimensions is not an image of anything.
        live.ingest(&frame(
            STREAM_IMAGE,
            5,
            Payload::Image {
                w: 3,
                h: 2,
                format: "rgb8".to_owned(),
                bytes: vec![0; 4],
            },
        ));
        assert!(live.image().is_none());
    }

    /// Oracle 4 (packet M7/E7). A collection's episodes become rows of the Run tab's one
    /// table, exactly as an evaluation's cells do — same `CellRow`, same order, same strip —
    /// and a cycle's stage strip is the order its stages began in.
    #[test]
    fn live_run_folds_collect_episodes_like_cells() {
        let mut live = LiveRun::default();
        live.ingest(&event("stage.begin", &[("name", "collect".to_owned())]));
        for episode in 0..2u32 {
            live.ingest(&event(
                "episode.begin",
                &[
                    ("episode", episode.to_string()),
                    ("seed", "4".to_owned()),
                    ("stage", "collect".to_owned()),
                ],
            ));
            for tick in 0..3u64 {
                live.ingest(&frame(
                    STREAM_TICKS,
                    tick,
                    Payload::Scalars(vec![tick as f64, tick as f64, 1.0, 2.0]),
                ));
            }
            live.ingest(&event(
                "episode.end",
                &[
                    ("episode", episode.to_string()),
                    ("outcome", "Success".to_owned()),
                    ("steps", "3".to_owned()),
                ],
            ));
        }
        live.ingest(&event(
            "stage.end",
            &[
                ("name", "collect".to_owned()),
                ("seconds", "12.500".to_owned()),
                ("code", "0".to_owned()),
            ],
        ));
        live.ingest(&event("stage.begin", &[("name", "train".to_owned())]));

        let rows = live.cells();
        assert_eq!(
            rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["episode-00", "episode-01"],
            "zero-padded, so the table's order is the episode order"
        );
        assert!(rows
            .iter()
            .all(|r| r.suite == "collect" && r.seed == Some(4)));
        assert!(rows.iter().all(|r| r.frames == 3 && r.has_traj));
        // The strip is the evaluation's own fold of the same bits: one row per tick, the
        // clamped source and the violation the plane raised.
        let timeline = live.timeline("episode-00");
        assert_eq!(timeline.rows.len(), 3);
        assert_eq!(timeline.rows[0].source, EventSource::Clamped);
        assert!(!timeline.totals.is_empty(), "the events bits decoded");
        // A collection has no suite row, so the table keeps the three identity columns.
        assert_eq!(live.columns(), ["cell", "suite", "seed"]);

        let stages = live.stages();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].name, "collect");
        assert_eq!(stages[0].seconds, Some(12.5));
        assert!(!stages[0].running());
        assert!(stages[1].running(), "train has not ended");
        println!(
            "RAN live_run_folds_collect_episodes_like_cells: {} row(s), {} stage(s)",
            rows.len(),
            stages.len()
        );
    }

    #[test]
    fn the_running_cell_is_selected_until_a_row_is_clicked() {
        let mut live = LiveRun::default();
        live.ingest(&event(
            "cell.begin",
            &[
                ("cell", "nominal-00".to_owned()),
                ("suite", "nominal".to_owned()),
                ("seed", "7".to_owned()),
            ],
        ));
        assert_eq!(live.selected_cell().as_deref(), Some("nominal-00"));
        assert!(
            live.status().contains("nominal-00 running"),
            "{}",
            live.status()
        );
        live.ingest(&event(
            "cell.end",
            &[
                ("cell", "nominal-00".to_owned()),
                ("outcome", "Timeout".to_owned()),
                ("frames", "12".to_owned()),
                ("traj", "true".to_owned()),
            ],
        ));
        assert_eq!(live.cells()[0].frames, 12);
        assert!(live.cells()[0].has_traj);
        assert_eq!(live.selected_cell(), None, "no cell is running any more");
        live.select("nominal-00");
        assert_eq!(live.selected_cell().as_deref(), Some("nominal-00"));
    }
}
