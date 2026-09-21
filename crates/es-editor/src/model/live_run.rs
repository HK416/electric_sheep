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

/// The row's identity (packet M7/R12): the stage a cell ran in and the cell's own name. A
/// cycle publishes every stage on one socket and two stages name their episodes alike — the
/// expert gate's `nominal-00` is not the evaluation's — so the name alone is not a row.
///
/// The one place the key is built, because `app.rs` rebuilds it from a [`CellRow`] to select a
/// row or ask for its timeline. A run read off disk has no stage, and its key is its cell name
/// unchanged, which is what keeps `RunView` and E4's oracle out of this.
pub fn cell_key(stage: &str, cell: &str) -> String {
    if stage.is_empty() {
        cell.to_owned()
    } else {
        format!("{stage} / {cell}")
    }
}

/// One episode as the wire described it.
#[derive(Clone, Debug, Default, PartialEq)]
struct LiveCell {
    /// The cell's own name, without the stage: what the table's first column shows.
    name: String,
    /// The stage it ran in, empty for a run that is one command.
    stage: String,
    /// Where that stage arrived in the cycle — `eval` sorts before `expert-gate`, which is not
    /// the order a cycle runs them in.
    order: usize,
    suite: String,
    seed: Option<u64>,
    records: Vec<StepEvent>,
    frames: usize,
    has_traj: bool,
    /// The `cell.end` outcome (`Success`, `Timeout`, ...); `None` while it is still running.
    outcome: Option<String>,
    /// The viewer attached after this cell had begun, so the row was made by its end event.
    joined_late: bool,
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
    /// Keyed by [`cell_key`], so a stage's rows are contiguous and in cell-name order inside
    /// it — the order `RunView` sorts its rows into, once [`Self::cells`] has put the stages
    /// back in the order they arrived.
    cells: BTreeMap<String, LiveCell>,
    /// One suite row per `(stage, suite)`, keyed by [`cell_key`] too: the expert gate's
    /// `suite.end` is not the evaluation's, even when both suites are called `nominal`.
    suites: BTreeMap<String, LiveSuite>,
    /// The episode currently running, which is what a stream-2 sample belongs to: the wire
    /// carries numbers, and `cell.begin` / `cell.end` are what name them.
    open: Option<String>,
    /// Samples that arrived with no open cell: a viewer that attached mid-episode heard them
    /// before anything named the cell. Claimed by the next end event, discarded by the next
    /// begin (packet M7/R12).
    pending: Vec<StepEvent>,
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
        let count = |k: &str| fields.get(k).and_then(|s| s.parse().ok()).unwrap_or(0);
        match kind {
            "cell.begin" => self.begin(fields, get("cell"), get("suite")),
            "cell.end" => self.end(
                fields,
                get("cell"),
                get("suite"),
                count("frames"),
                fields.get("traj").is_some_and(|s| s == "true"),
            ),
            // An episode of `es loop collect` is a row of the same table (packet M7/E7). It
            // has no suite of its own, so the stage it came from is the column's value: a
            // cycle's Run tab then reads "collect / episode-00" beside "nominal / nominal-00".
            "episode.begin" => self.begin(fields, episode_cell(&get("episode")), get("stage")),
            // A collection writes rows, not frames: the count a person wants beside the
            // episode is the steps it recorded.
            "episode.end" => self.end(
                fields,
                episode_cell(&get("episode")),
                get("stage"),
                count("steps"),
                true,
            ),
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
                let key = cell_key(&self.stage_of(fields), &get("suite"));
                let suite = self.suites.entry(key).or_default();
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

    /// The stage an event belongs to: its own field when it has one — a cycle puts it on every
    /// stream-1 event — else the stage that last began, else empty for a run that is one
    /// command.
    fn stage_of(&self, fields: &BTreeMap<String, String>) -> String {
        match fields.get("stage") {
            Some(stage) if !stage.is_empty() => stage.clone(),
            _ => self
                .stages
                .last()
                .map(|s| s.name.clone())
                .unwrap_or_default(),
        }
    }

    /// Where a stage arrived in the cycle. A stage no `stage.begin` has named yet sorts after
    /// the ones that have, which is also where it will land once its own `stage.begin` does.
    fn stage_order(&self, stage: &str) -> usize {
        self.stages
            .iter()
            .rposition(|s| s.name == stage)
            .unwrap_or(self.stages.len())
    }

    /// Opens a row. `cell.begin` and `episode.begin` differ only in what they call the cell
    /// and what goes in its suite column.
    fn begin(&mut self, fields: &BTreeMap<String, String>, name: String, suite: String) {
        // Samples still waiting for a name belong to a cell nobody will ever name: in a
        // well-ordered stream there are none, and guessing them into this cell would put one
        // episode's history under another's name.
        self.pending.clear();
        let stage = self.stage_of(fields);
        let order = self.stage_order(&stage);
        let key = cell_key(&stage, &name);
        let cell = self.cells.entry(key.clone()).or_insert_with(|| LiveCell {
            name,
            stage,
            order,
            ..LiveCell::default()
        });
        cell.suite = suite;
        cell.seed = fields.get("seed").and_then(|s| s.parse().ok());
        self.open = Some(key);
    }

    /// Closes a row, or makes one: an end event for a cell this viewer never saw begin is a
    /// cell that began before it attached, so the row is built from the event's own fields and
    /// takes the samples that were waiting (packet M7/R12).
    fn end(
        &mut self,
        fields: &BTreeMap<String, String>,
        name: String,
        suite: String,
        frames: usize,
        has_traj: bool,
    ) {
        let stage = self.stage_of(fields);
        let order = self.stage_order(&stage);
        let key = cell_key(&stage, &name);
        let pending = std::mem::take(&mut self.pending);
        let cell = self.cells.entry(key).or_insert_with(|| LiveCell {
            name,
            stage,
            order,
            suite,
            records: pending,
            joined_late: true,
            ..LiveCell::default()
        });
        cell.frames = frames;
        cell.has_traj = has_traj;
        cell.outcome = Some(fields.get("outcome").cloned().unwrap_or_default());
        self.open = None;
    }

    /// One `[frame, tick, source, violation bits]` sample, attributed to the open episode.
    fn tick(&mut self, v: &[f64]) {
        let [frame, tick, source, events] = v else {
            return;
        };
        let record = StepEvent {
            frame: *frame as u64,
            tick: PhysTick(*tick as u64),
            source: event_source(*source as u32),
            events: *events as u32,
        };
        let open = match self.open.as_deref() {
            Some(key) => self.cells.get_mut(key),
            None => None,
        };
        match open {
            Some(cell) => cell.records.push(record),
            None => self.pending.push(record),
        }
    }

    /// One row per `(stage, cell)`, in stage-arrival then cell-name order —
    /// [`crate::model::run_view::RunView::cells`]'s own shape, and its order for a run that has
    /// no stages, so the table does not know which end it came from.
    pub fn cells(&self) -> Vec<CellRow> {
        let mut rows: Vec<&LiveCell> = self.cells.values().collect();
        // The map's own order is already cell-name order inside a stage; a stable sort by the
        // stage's arrival then puts the stages in the order the cycle ran them.
        rows.sort_by_key(|c| c.order);
        rows.into_iter()
            .map(|c| {
                let suite = self.suites.get(&cell_key(&c.stage, &c.suite));
                CellRow {
                    name: c.name.clone(),
                    stage: c.stage.clone(),
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

    /// Whether the row keyed by [`cell_key`] was made by its end event — the cell was already
    /// running when this viewer attached, so what it shows is only the part it heard.
    pub fn joined_late(&self, key: &str) -> bool {
        self.cell(key).is_some_and(|c| c.joined_late)
    }

    /// The row `key` names. `key` is [`cell_key`]'s, which is what `app.rs` holds; a bare cell
    /// name still finds the row while only one stage has run a cell by that name, which is
    /// every run that is one command.
    ///
    /// ponytail: the bare-name path is a linear scan of the rows a run has (tens), and an
    /// ambiguous name finds nothing rather than the wrong stage's row. Key the names too if a
    /// run ever has enough rows for the scan to show.
    fn cell(&self, key: &str) -> Option<&LiveCell> {
        if let Some(cell) = self.cells.get(key) {
            return Some(cell);
        }
        let mut named = self.cells.values().filter(|c| c.name == key);
        let only = named.next()?;
        named.next().is_none().then_some(only)
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
        let Some(live) = self.cell(cell) else {
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
        let late = self.cells.values().filter(|c| c.joined_late).count();
        let late = if late == 0 {
            String::new()
        } else {
            format!(", {late} joined late")
        };
        match &self.open {
            Some(cell) => format!(
                "live: {cell} running, {done} of {} cell(s) finished{late}",
                self.cells.len()
            ),
            None => format!(
                "live: {done} of {} cell(s) finished{late}",
                self.cells.len()
            ),
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

    /// Ticks with nothing interesting in them, for a cell that is only being counted.
    fn ticks(live: &mut LiveRun, n: u64) {
        for tick in 0..n {
            live.ingest(&frame(
                STREAM_TICKS,
                tick,
                Payload::Scalars(vec![tick as f64, tick as f64, 0.0, 0.0]),
            ));
        }
    }

    /// Oracle 1 (packet M7/R12; review R16). A cycle publishes the expert gate and the policy
    /// evaluation on one socket and both call their first episode `nominal-00`. The row's
    /// identity is the pair, so the gate's `success_rate` stays in the gate's row and the
    /// evaluation's row is still empty while it runs.
    #[test]
    fn live_run_keys_rows_by_stage_and_cell() {
        let mut live = LiveRun::default();
        live.ingest(&event("stage.begin", &[("name", "expert-gate".to_owned())]));
        live.ingest(&event(
            "cell.begin",
            &[
                ("cell", "nominal-00".to_owned()),
                ("suite", "nominal".to_owned()),
                ("seed", "1".to_owned()),
                ("stage", "expert-gate".to_owned()),
            ],
        ));
        ticks(&mut live, 3);
        live.ingest(&event(
            "cell.end",
            &[
                ("cell", "nominal-00".to_owned()),
                ("outcome", "Success".to_owned()),
                ("frames", "3".to_owned()),
                ("traj", "true".to_owned()),
                ("stage", "expert-gate".to_owned()),
            ],
        ));
        live.ingest(&event(
            "suite.end",
            &[
                ("suite", "nominal".to_owned()),
                ("n_episodes", "4".to_owned()),
                (
                    "metric.success_rate",
                    serde_json::to_string(&MetricValue::Scalar(1.0)).expect("a metric value"),
                ),
                ("stage", "expert-gate".to_owned()),
            ],
        ));
        live.ingest(&event(
            "stage.end",
            &[("name", "expert-gate".to_owned()), ("code", "0".to_owned())],
        ));
        live.ingest(&event("stage.begin", &[("name", "eval".to_owned())]));
        live.ingest(&event(
            "cell.begin",
            &[
                ("cell", "nominal-00".to_owned()),
                ("suite", "nominal".to_owned()),
                ("seed", "1".to_owned()),
                ("stage", "eval".to_owned()),
            ],
        ));
        ticks(&mut live, 2);

        let rows = live.cells();
        assert_eq!(rows.len(), 2, "one row per (stage, cell): {rows:?}");
        assert_eq!(
            rows.iter().map(|r| r.stage.as_str()).collect::<Vec<_>>(),
            ["expert-gate", "eval"],
            "the order the stages arrived in, not the order their names sort in"
        );
        assert!(rows.iter().all(|r| r.name == "nominal-00"));
        assert_eq!(
            rows[0].metrics.get("success_rate"),
            Some(&MetricValue::Scalar(1.0)),
            "the gate's own suite row"
        );
        assert!(
            rows[1].metrics.is_empty(),
            "the evaluation has measured nothing yet: {:?}",
            rows[1].metrics
        );
        assert!(
            live.status().contains("eval / nominal-00 running"),
            "{}",
            live.status()
        );
        assert_eq!(
            live.timeline(&cell_key("expert-gate", "nominal-00"))
                .rows
                .len(),
            3
        );
        assert_eq!(live.timeline(&cell_key("eval", "nominal-00")).rows.len(), 2);
        println!(
            "RAN live_run_keys_rows_by_stage_and_cell: {} row(s), {} stage(s)",
            rows.len(),
            live.stages().len()
        );
    }

    /// Oracle 2 (packet M7/R12). A viewer that attached after `cell.begin` had gone by still
    /// gets the row: samples with no open cell wait, and the `cell.end` that names them makes
    /// the row out of its own fields and marks it joined late.
    #[test]
    fn a_cell_that_began_before_attach_still_gets_a_row() {
        let mut live = LiveRun::default();
        ticks(&mut live, 3);
        assert!(live.is_empty(), "nothing has named a cell yet");

        live.ingest(&event(
            "cell.end",
            &[
                ("cell", "nominal-01".to_owned()),
                ("outcome", "Timeout".to_owned()),
                ("frames", "3".to_owned()),
                ("traj", "true".to_owned()),
            ],
        ));
        let rows = live.cells();
        assert_eq!(rows.len(), 1, "the end event names the row: {rows:?}");
        assert_eq!(rows[0].name, "nominal-01");
        assert_eq!(rows[0].frames, 3);
        assert!(rows[0].has_traj);
        assert!(live.joined_late(&cell_key("", "nominal-01")));
        assert_eq!(
            live.timeline("nominal-01").rows.len(),
            3,
            "the waiting samples are this cell's"
        );
        assert!(live.status().contains("1 joined late"), "{}", live.status());

        live.ingest(&event(
            "cell.begin",
            &[
                ("cell", "nominal-02".to_owned()),
                ("suite", "nominal".to_owned()),
            ],
        ));
        assert!(
            !live.joined_late("nominal-02"),
            "a cell the viewer saw begin is an ordinary row"
        );
        println!(
            "RAN a_cell_that_began_before_attach_still_gets_a_row: {} row(s)",
            live.cells().len()
        );
    }
}
