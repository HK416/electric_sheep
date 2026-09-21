//! The Run tab's view-model: a finished `es eval run` / `es loop collect` directory, read
//! from disk with no process on the other end (spec 23.3, spec 10.5; packet M7/E1).
//!
//! A run is `report.json` plus, optionally, `events.json`, `traj/<cell>.estraj` and
//! `frames/<cell>/NNNNNN.bin`. Only `report.json` is required: a directory that holds it and
//! nothing else opens, and [`RunView::status`] names what was not there — the run someone
//! opens the editor for is often the one that did not finish writing.
//!
//! Two different things are called a "cell" upstream and this file keeps both straight. A
//! [`es_ir::evaluation::CellResult`] is one *suite x metric* of the spec 10.1 table; a cell on
//! disk (`frames/<cell>`, `traj/<cell>.estraj`, an `events.json` key) is one *episode*, named
//! `<suite>-<NN>` by `Evaluation::run_shard`. [`CellRow`] is the episode, carrying the
//! suite-level metrics its suite measured.
//!
//! Nothing here decides how to draw: sorting, bucketing, sampling and decoding are methods
//! with tests, because `app.rs` is compiled and never run by CI (spec 28.10 rule 3).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use es_eval::runner::{EventSource, StepEvent};
use es_ir::evaluation::{AcceptanceResult, EvaluationReport, MetricValue};
use es_safety::{EventSet, ViolationKind};

use crate::model::image_view::Rgb8Image;

/// Everything that stops a run directory from opening, each naming its file.
#[derive(Debug)]
pub enum RunError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Json { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for RunError {}

/// One episode of the run: what it is called, what its suite measured, and what it left on
/// disk.
#[derive(Clone, Debug, PartialEq)]
pub struct CellRow {
    /// The on-disk cell name, `<suite>-<NN>` for a run that wrote frames or trajectories, and
    /// the bare suite name for a report that stands alone.
    pub name: String,
    /// The stage of `es loop cycle` this cell ran in, for a run heard on the wire (packet
    /// M7/R12). Always empty here: a run read off disk is one run, and the directory says
    /// nothing about the cycle that may have produced it.
    pub stage: String,
    pub suite: String,
    /// The seed this episode ran under, when `evaluation.lock` is beside the report (spec
    /// 10.5). `report.json` alone does not carry seeds, and an invented one is worse than
    /// none.
    pub seed: Option<u64>,
    /// The suite's metrics by the report's own metric names — never a fixed list, so a metric
    /// added to `es-ir` shows up here with no change in the editor.
    pub metrics: BTreeMap<String, MetricValue>,
    /// Episodes that contributed to the suite's numbers (spec 18.5 drops quarantined envs).
    pub n_episodes: u32,
    pub has_traj: bool,
    /// Frames on disk under `frames/<name>/`.
    pub frames: usize,
}

/// One tick of one episode, as `events.json` recorded it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickRow {
    pub frame: u64,
    pub tick: u64,
    pub source: EventSource,
    pub events: EventSet,
}

/// Where a kind was first seen. Two numbers because a run records two clocks: `frame` indexes
/// the records `events.json` holds, which is what the strip is drawn in, and `tick` is the
/// `PhysTick` that frame carried — several physics ticks to one control step (spec 12.1), so
/// the two differ by the ratio and only one of them is a position in the strip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FirstSeen {
    pub frame: u64,
    pub tick: u64,
}

/// One line of the per-kind summary under the strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindRow {
    pub kind: ViolationKind,
    /// Records that carried the kind — frames of the strip, not physics ticks.
    pub frames: usize,
    pub first: FirstSeen,
}

impl KindRow {
    /// The label, decided here rather than in `app.rs`: both clocks, the strip's one first,
    /// so "first at frame 136" can be found by eye on the strip above it.
    pub fn label(&self) -> String {
        format!(
            "{:?}: {} frame(s), first at frame {} (tick {})",
            self.kind, self.frames, self.first.frame, self.first.tick
        )
    }
}

/// One episode's Safety Plane history (spec 9.3, spec 9.4).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline {
    pub rows: Vec<TickRow>,
    /// How many records carried each kind. A kind nobody saw is absent, not zero.
    pub totals: BTreeMap<ViolationKind, usize>,
    /// Where each kind first appeared — where to look, not how often.
    pub first: BTreeMap<ViolationKind, FirstSeen>,
}

/// One column of the timeline strip: the ticks that fall in it, folded.
#[derive(Clone, Debug, PartialEq)]
pub struct Bucket {
    /// The most severe source in the bucket, so a single clamped tick in a wide strip is
    /// still visible (see [`severity`]).
    pub source: EventSource,
    pub counts: BTreeMap<ViolationKind, usize>,
    pub ticks: usize,
}

/// Rank of a source for folding a bucket: the policy's own action is the quiet case, a
/// fallback is the loud one.
pub fn severity(source: EventSource) -> u8 {
    match source {
        EventSource::Policy => 0,
        EventSource::Human => 1,
        EventSource::Clamped => 2,
        EventSource::Fallback => 3,
    }
}

impl Timeline {
    /// The strip's own heading: `"nominal-00: 224 frame(s)"`. Frames, not ticks — the rows are
    /// `events.json` records, one per control step, and [`KindRow::label`] counts the same
    /// thing. The unit is the model's to name (spec 28.10 rule 3).
    pub fn heading(&self, cell: &str) -> String {
        format!("{cell}: {} frame(s)", self.rows.len())
    }

    /// The per-kind summary in `ViolationKind` order, ready to print.
    pub fn kind_rows(&self) -> Vec<KindRow> {
        self.totals
            .iter()
            .map(|(kind, frames)| KindRow {
                kind: *kind,
                frames: *frames,
                first: self.first.get(kind).copied().unwrap_or_default(),
            })
            .collect()
    }

    /// `n` columns over the whole episode, for drawing at any width. Empty for an episode
    /// with no events, and never more columns than there are ticks.
    pub fn buckets(&self, n: usize) -> Vec<Bucket> {
        let n = n.min(self.rows.len());
        if n == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|i| {
                // Contiguous, gapless and total: every tick lands in exactly one bucket, which
                // is what makes the per-kind counts sum back to `totals`.
                let lo = i * self.rows.len() / n;
                let hi = (i + 1) * self.rows.len() / n;
                let rows = &self.rows[lo..hi];
                let mut counts: BTreeMap<ViolationKind, usize> = BTreeMap::new();
                let mut source = EventSource::Policy;
                for row in rows {
                    if severity(row.source) > severity(source) {
                        source = row.source;
                    }
                    for kind in row.events.iter() {
                        *counts.entry(kind).or_default() += 1;
                    }
                }
                Bucket {
                    source,
                    counts,
                    ticks: rows.len(),
                }
            })
            .collect()
    }
}

/// `es_safety::EventSet` from the `u32` a [`StepEvent`] carries.
///
/// The bit positions are [`ViolationKind::index`]'s own, read back through `EventSet::insert`
/// rather than re-derived: the decoding and the encoding are the same table (spec 9.3).
/// `EventSet` has no `from_bits`, and `es-safety` is not this packet's to change.
pub fn decode_events(bits: u32) -> EventSet {
    let mut set = EventSet::EMPTY;
    for kind in ViolationKind::ALL {
        if bits & (1 << kind.index()) != 0 {
            set.insert(kind);
        }
    }
    set
}

/// An opened run directory.
#[derive(Debug)]
pub struct RunView {
    pub dir: PathBuf,
    pub report: EvaluationReport,
    /// `events.json` verbatim: cell name -> frame-ordered records.
    pub events: BTreeMap<String, Vec<StepEvent>>,
    /// What the directory did not hold, for the status line.
    pub status: String,
    /// Where `<cell>/NNNNNN.bin` lives. `<run>/frames` by default, but
    /// `es eval run --frames <dir>` writes wherever it was told, which is usually a sibling of
    /// the run directory - so this is settable ([`Self::set_frames_root`]).
    frames_root: PathBuf,
    /// `evaluation.lock`'s seeds, kept for rebuilding the rows.
    seeds: Vec<u64>,
    rows: Vec<CellRow>,
    selected: Option<String>,
}

impl RunView {
    /// Whether `dir` looks like a run: what tells a run directory from a bundle directory.
    pub fn is_run_dir(dir: &Path) -> bool {
        dir.join("report.json").is_file()
    }

    /// Reads `report.json` and whatever else is there. Only the report is required.
    pub fn open(dir: &Path) -> Result<Self, RunError> {
        let report: EvaluationReport = read_json(&dir.join("report.json"))?;
        let events_path = dir.join("events.json");
        let events: BTreeMap<String, Vec<StepEvent>> = if events_path.is_file() {
            read_json(&events_path)?
        } else {
            BTreeMap::new()
        };
        let mut view = Self {
            dir: dir.to_path_buf(),
            report,
            events,
            status: String::new(),
            frames_root: dir.join("frames"),
            seeds: read_seeds(dir),
            rows: Vec::new(),
            selected: None,
        };
        view.rebuild();
        Ok(view)
    }

    /// Points the filmstrip at another directory - what `es eval run --frames <dir>` wrote.
    /// The rows are rebuilt, so a run whose frames live outside it gains their counts (and,
    /// for a report that stands alone, their cells).
    pub fn set_frames_root(&mut self, root: impl Into<PathBuf>) {
        self.frames_root = root.into();
        self.rebuild();
    }

    pub fn frames_root(&self) -> &Path {
        &self.frames_root
    }

    /// Re-derives the rows and the status line from what is on disk now. Called by
    /// [`Self::open`] and whenever the frames root moves; the sort order goes back to cell
    /// name order, which is the order the table opens in.
    fn rebuild(&mut self) {
        let traj: Vec<String> = stems(&self.dir.join("traj"), Some("estraj"));
        let frame_dirs: Vec<String> = stems(&self.frames_root, None);
        let mut missing = Vec::new();
        if self.events.is_empty() {
            missing.push("events.json");
        }
        if traj.is_empty() {
            missing.push("traj/");
        }
        if frame_dirs.is_empty() {
            missing.push("frames/");
        }
        self.rows = build_rows(
            &self.frames_root,
            &self.report,
            &self.events,
            &self.seeds,
            &traj,
            &frame_dirs,
        );
        self.status = if missing.is_empty() {
            format!("{} cell(s), complete", self.rows.len())
        } else {
            format!("{} cell(s); no {}", self.rows.len(), missing.join(", no "))
        };
    }

    pub fn cells(&self) -> &[CellRow] {
        &self.rows
    }

    /// The report's acceptance lines, unchanged (spec 10.2). The verdict is the report's, not
    /// this view's: an editor that recomputed `passed` could disagree with the artifact.
    pub fn acceptance(&self) -> &[AcceptanceResult] {
        &self.report.acceptance
    }

    /// Table headers in order: the three identity columns, then one per metric the report
    /// carries.
    pub fn columns(&self) -> Vec<String> {
        let mut out = vec!["cell".to_owned(), "suite".to_owned(), "seed".to_owned()];
        let mut metrics: Vec<&String> = self
            .rows
            .iter()
            .flat_map(|r| r.metrics.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        metrics.sort();
        out.extend(metrics.into_iter().cloned());
        out
    }

    /// Sorts the rows by one of [`Self::columns`]. Stable, so clicking a second header keeps
    /// the first one's order inside ties; an index past the last column is a no-op.
    pub fn sort_by(&mut self, column: usize) {
        let Some(name) = self.columns().get(column).cloned() else {
            return;
        };
        match column {
            0 => self.rows.sort_by(|a, b| a.name.cmp(&b.name)),
            1 => self.rows.sort_by(|a, b| a.suite.cmp(&b.suite)),
            2 => self.rows.sort_by_key(|a| a.seed),
            _ => self.rows.sort_by_key(|a| metric_key(a, &name)),
        }
    }

    /// The cell the table has selected, which is also what the replay panel plays (packet
    /// M7/E2).
    pub fn selected_cell(&self) -> Option<&CellRow> {
        let name = self.selected.as_ref()?;
        self.rows.iter().find(|r| r.name == *name)
    }

    pub fn select(&mut self, name: &str) {
        self.selected = Some(name.to_owned());
    }

    /// `traj/<cell>.estraj`, whether or not it exists.
    pub fn traj_path(&self, cell: &str) -> PathBuf {
        self.dir.join("traj").join(format!("{cell}.estraj"))
    }

    /// One episode's Safety Plane history, decoded. Empty when the run wrote no `events.json`.
    pub fn timeline(&self, cell: &str) -> Timeline {
        let mut out = Timeline::default();
        let Some(records) = self.events.get(cell) else {
            return out;
        };
        for record in records {
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

    /// At most `max` frame indices spread evenly over the cell's frames, first and last
    /// included — the filmstrip.
    pub fn filmstrip(&self, cell: &str, max: usize) -> Vec<usize> {
        let n = self
            .rows
            .iter()
            .find(|r| r.name == cell)
            .map_or(0, |r| r.frames);
        if n == 0 || max == 0 {
            return Vec::new();
        }
        if n <= max {
            return (0..n).collect();
        }
        (0..max).map(|i| i * (n - 1) / (max - 1)).collect()
    }

    /// One frame of one cell, decoded through the sibling `layout.json` the run wrote
    /// (`es_eval::runner`'s own directory shape). Nothing is cached: a filmstrip asks for
    /// eight of them and a 96x96 frame is 27 kB.
    pub fn frame(&self, cell: &str, index: usize) -> Option<Rgb8Image> {
        let dir = self.frames_root.join(cell);
        let layout: FrameLayout = read_json(&dir.join("layout.json")).ok()?;
        let [rows, cols, chans] = layout.shape;
        let (rows, cols, chans) = (rows as usize, cols as usize, chans as usize);
        if layout.dtype != "u8" || !(chans == 1 || chans == 3 || chans == 4) {
            return None;
        }
        let data = fs::read(dir.join(format!("{index:06}.bin"))).ok()?;
        if data.len() != rows * cols * chans {
            return None;
        }
        // Grey replicates, RGBA drops alpha: the rule `model::image_view` already displays by.
        let mut rgb = Vec::with_capacity(rows * cols * 3);
        for px in data.chunks_exact(chans) {
            rgb.extend_from_slice(&[px[0], px[(chans - 1).min(1)], px[(chans - 1).min(2)]]);
        }
        Some(Rgb8Image {
            width: cols,
            height: rows,
            data: rgb,
        })
    }
}

/// The `layout.json` beside a cell's frames (`es_eval::runner::CellFrames`).
#[derive(serde::Deserialize)]
struct FrameLayout {
    dtype: String,
    shape: [u64; 3],
}

/// `evaluation.lock`'s seed list (spec 10.5), or empty when it is not beside the report. Only
/// the seeds are read, so a lock from a newer schema still gives them up.
#[derive(serde::Deserialize)]
struct Seeds {
    #[serde(default)]
    seeds: Vec<u64>,
}

fn read_seeds(dir: &Path) -> Vec<u64> {
    read_json::<Seeds>(&dir.join("evaluation.lock")).map_or_else(|_| Vec::new(), |l| l.seeds)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, RunError> {
    let text = fs::read_to_string(path).map_err(|source| RunError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|e| RunError::Json {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// File stems (or directory names, with `ext = None`) directly under `dir`, sorted. A
/// directory that is not there is empty, not an error.
fn stems(dir: &Path, ext: Option<&str>) -> Vec<String> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| match ext {
            Some(ext) => e.path().extension().is_some_and(|x| x == ext),
            None => e.path().is_dir(),
        })
        .filter_map(|e| Some(e.path().file_stem()?.to_string_lossy().into_owned()))
        .collect();
    out.sort();
    out
}

/// Sort key of one metric column: scalars by value, everything else after them (a histogram
/// has no single value and an unmeasured metric has none at all, spec 10.3).
fn metric_key(row: &CellRow, metric: &str) -> (u8, ordered::F64) {
    match row.metrics.get(metric) {
        Some(MetricValue::Scalar(v)) => (0, ordered::F64(*v)),
        Some(_) => (1, ordered::F64(0.0)),
        None => (2, ordered::F64(0.0)),
    }
}

/// `f64` with a total order, so a column of metric values sorts without `partial_cmp` and
/// without a `NaN` losing rows.
mod ordered {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct F64(pub f64);

    impl Eq for F64 {}

    impl PartialOrd for F64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for F64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
}

/// One row per episode on disk, or — for a report that stands alone — one per suite.
fn build_rows(
    frames_root: &Path,
    report: &EvaluationReport,
    events: &BTreeMap<String, Vec<StepEvent>>,
    seeds: &[u64],
    traj: &[String],
    frame_dirs: &[String],
) -> Vec<CellRow> {
    let mut names: Vec<String> = events
        .keys()
        .cloned()
        .chain(traj.iter().cloned())
        .chain(frame_dirs.iter().cloned())
        .collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        names = report
            .cells
            .iter()
            .map(|c| c.suite.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
    }
    names
        .into_iter()
        .map(|name| {
            let suite = suite_of(&name, report);
            let mut metrics = BTreeMap::new();
            let mut n_episodes = 0;
            for cell in report.cells.iter().filter(|c| c.suite == suite) {
                metrics.insert(cell.metric.name().to_owned(), cell.value.clone());
                n_episodes = n_episodes.max(cell.n_episodes);
            }
            CellRow {
                stage: String::new(),
                seed: episode_index(&name, &suite).and_then(|i| seeds.get(i).copied()),
                metrics,
                n_episodes,
                has_traj: traj.contains(&name),
                frames: count_frames(&frames_root.join(&name)),
                suite,
                name,
            }
        })
        .collect()
}

fn count_frames(dir: &Path) -> usize {
    stems(dir, Some("bin")).len()
}

/// The suite a cell belongs to: the longest suite name the report carries that the cell name
/// starts with, since `run_shard` names an episode `<suite>-<NN>` and a suite name may itself
/// hold a dash.
fn suite_of(name: &str, report: &EvaluationReport) -> String {
    report
        .cells
        .iter()
        .map(|c| &c.suite)
        .filter(|s| name == s.as_str() || name.starts_with(&format!("{s}-")))
        .max_by_key(|s| s.len())
        .cloned()
        .unwrap_or_else(|| name.to_owned())
}

/// The `NN` of `<suite>-<NN>`, which is the episode's index into `evaluation.lock`'s seeds.
fn episode_index(name: &str, suite: &str) -> Option<usize> {
    name.strip_prefix(suite)?.strip_prefix('-')?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use es_core::PhysTick;
    use es_eval::runner::{BackendCaps, EvaluationLock, FrameSink};
    use es_ir::evaluation::{AcceptanceCriterion, Aggregation, CellResult, Comparator, MetricSpec};

    /// The committed fixture run, generated by [`generate_fixture_run`] below.
    fn fixture() -> PathBuf {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/run")
    }

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("es-editor-{tag}-{nanos}"));
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    // --- the fixture generator ---------------------------------------------------------------

    /// The four cells of the fixture: two suites, two episodes each.
    const CELLS: [&str; 4] = ["light-00", "light-01", "nominal-00", "nominal-01"];

    /// Regenerates `tests/fixtures/visible-learning/run/`. Run once, explicitly; the result is
    /// then read-only (spec 1.4). `report.json`, `evaluation.lock` and `events.json` are
    /// written by the types that own them (`es_eval::runner`), so the fixture cannot drift
    /// from what a real run writes.
    #[test]
    #[ignore = "fixture generator; run explicitly"]
    fn generate_fixture_run() {
        let dir = fixture();
        fs::create_dir_all(&dir).expect("fixture dir");

        let lock = EvaluationLock {
            schema_version: 1,
            evaluation_hash: "00".repeat(32),
            execution_hash: "11".repeat(32),
            seeds: vec![7, 8],
            backend: BackendCaps {
                name: "fixture".to_owned(),
                determinism: "bitwise".to_owned(),
                float: "fp64".to_owned(),
                max_envs: 1,
                gpu_resident: false,
                supports_reset_subset: true,
                supports_state_get_set: true,
                quirks: Vec::new(),
            },
            created: 0,
        };
        es_eval::runner::write_artifacts(&report(), &lock, &dir).expect("artifacts");

        // Four ticks per cell. `nominal-01` is clamped twice (a velocity clamp, then velocity
        // and acceleration together) and `light-01` falls back once, so the timeline strip has
        // something to draw and the per-kind totals are not all zero.
        let mut sink = FrameSink::new(dir.join("frames"));
        for cell in CELLS {
            let mut events = Vec::new();
            for frame in 0..4u64 {
                let (source, bits) = match (cell, frame) {
                    ("nominal-01", 1) => (EventSource::Clamped, mask(&[ViolationKind::Velocity])),
                    ("nominal-01", 2) => (
                        EventSource::Clamped,
                        mask(&[ViolationKind::Velocity, ViolationKind::Acceleration]),
                    ),
                    ("light-01", 3) => (
                        EventSource::Fallback,
                        mask(&[ViolationKind::StaleObservation]),
                    ),
                    _ => (EventSource::Policy, 0),
                };
                events.push(StepEvent {
                    frame,
                    tick: PhysTick(frame),
                    source,
                    events: bits,
                });
            }
            sink.events.insert(cell.to_owned(), events);
        }
        sink.write_events(&dir.join("events.json")).expect("events");

        // One 96x96 RGB frame per cell, in the `<NNNNNN>.bin` + `layout.json` shape
        // `es_eval::runner` writes (its writer is private; the shape is its documentation).
        for (i, cell) in CELLS.iter().enumerate() {
            let cell_dir = dir.join("frames").join(cell);
            fs::create_dir_all(&cell_dir).expect("cell dir");
            fs::write(
                cell_dir.join("layout.json"),
                "{\"dtype\":\"u8\",\"shape\":[96,96,3]}\n",
            )
            .expect("layout.json");
            let mut data = Vec::with_capacity(96 * 96 * 3);
            for y in 0..96u32 {
                for x in 0..96u32 {
                    data.extend_from_slice(&[(x * 2) as u8, (y * 2) as u8, (i * 60) as u8]);
                }
            }
            fs::write(cell_dir.join("000000.bin"), &data).expect("frame");
        }
        println!("wrote {}", dir.display());
    }

    fn mask(kinds: &[ViolationKind]) -> u32 {
        let mut set = EventSet::EMPTY;
        for kind in kinds {
            set.insert(*kind);
        }
        set.bits()
    }

    /// Two suites x three metrics, and two acceptance lines of which one fails.
    fn report() -> EvaluationReport {
        let cell = |suite: &str, metric: MetricSpec, v: f64| CellResult {
            suite: suite.to_owned(),
            metric,
            value: MetricValue::Scalar(v),
            n_episodes: 2,
        };
        let criterion = |metric, comparator, threshold| AcceptanceCriterion {
            suite: None,
            metric,
            comparator,
            threshold,
            aggregation: Aggregation::Mean,
        };
        EvaluationReport {
            schema_version: 1,
            evaluation_hash: [0; 32],
            execution_hash: [1; 32],
            cells: vec![
                cell("nominal", MetricSpec::SuccessRate, 0.875),
                cell("nominal", MetricSpec::EpisodeLength, 120.0),
                cell("nominal", MetricSpec::EnvelopeViolationRate, 0.0),
                cell("light", MetricSpec::SuccessRate, 0.5),
                cell("light", MetricSpec::EpisodeLength, 150.0),
                cell("light", MetricSpec::EnvelopeViolationRate, 0.25),
            ],
            acceptance: vec![
                AcceptanceResult::Determined {
                    criterion: criterion(MetricSpec::SuccessRate, Comparator::Ge, 0.8),
                    observed: 0.6875,
                    passed: false,
                },
                AcceptanceResult::Determined {
                    criterion: criterion(MetricSpec::EnvelopeViolationRate, Comparator::Le, 0.5),
                    observed: 0.125,
                    passed: true,
                },
            ],
            passed: false,
            episodes: Vec::new(),
        }
    }

    // --- oracles -----------------------------------------------------------------------------

    /// Oracle 1: every number the table shows is the report's own, compared as `MetricValue`.
    #[test]
    fn run_view_reproduces_the_report() {
        let view = RunView::open(&fixture()).expect("open the fixture run");
        let text = fs::read_to_string(fixture().join("report.json")).expect("report.json");
        let report: EvaluationReport = serde_json::from_str(&text).expect("parse report.json");

        assert_eq!(view.cells().len(), CELLS.len());
        for row in view.cells() {
            assert!(CELLS.contains(&row.name.as_str()), "{row:?}");
            let want: BTreeMap<String, MetricValue> = report
                .cells
                .iter()
                .filter(|c| c.suite == row.suite)
                .map(|c| (c.metric.name().to_owned(), c.value.clone()))
                .collect();
            assert!(!want.is_empty(), "{} matched no report cell", row.name);
            assert_eq!(row.metrics, want, "cell {}", row.name);
            assert_eq!(row.n_episodes, 2);
            // `evaluation.lock` carries the seeds; the episode index selects one.
            assert_eq!(row.seed, Some(if row.name.ends_with("00") { 7 } else { 8 }));
            assert_eq!(row.frames, 1, "one frame per cell");
        }
        assert_eq!(view.acceptance(), report.acceptance.as_slice());
        assert_eq!(
            report.passed,
            report
                .acceptance
                .iter()
                .all(|a| matches!(a, AcceptanceResult::Determined { passed: true, .. })),
            "the fixture's own verdict agrees with its lines"
        );
        assert!(!report.passed, "the fixture has one failing line");

        let frame = view.frame("nominal-00", 0).expect("frame 0");
        assert_eq!((frame.width, frame.height), (96, 96));
        assert_eq!(frame.data.len(), 96 * 96 * 3);
        assert!(view.frame("nominal-00", 99).is_none(), "no such frame");
    }

    /// Oracle 2: the timeline is a decoding of `events.json`, and no bucketing loses a tick.
    #[test]
    fn timeline_buckets_sum_to_the_events() {
        let view = RunView::open(&fixture()).expect("open the fixture run");
        let text = fs::read_to_string(fixture().join("events.json")).expect("events.json");
        let raw: BTreeMap<String, Vec<StepEvent>> =
            serde_json::from_str(&text).expect("parse events.json");
        assert!(raw.values().any(|c| c.iter().any(|e| e.events != 0)));

        for (cell, records) in &raw {
            // A direct pass over the file, bit by bit, with no `EventSet` in sight.
            let mut want: BTreeMap<ViolationKind, usize> = BTreeMap::new();
            for record in records {
                for kind in ViolationKind::ALL {
                    if record.events & (1 << kind.index()) != 0 {
                        *want.entry(kind).or_default() += 1;
                    }
                }
            }
            let timeline = view.timeline(cell);
            assert_eq!(timeline.rows.len(), records.len(), "cell {cell}");
            assert_eq!(timeline.totals, want, "cell {cell}");
            for (kind, first) in &timeline.first {
                let seen = records
                    .iter()
                    .find(|r| r.events & (1 << kind.index()) != 0)
                    .expect("a kind that was totalled has a first tick");
                assert_eq!(first.tick, seen.tick.0, "cell {cell} kind {kind:?}");
                assert_eq!(first.frame, seen.frame, "cell {cell} kind {kind:?}");
            }
            for n in [1usize, 7, 64] {
                let buckets = timeline.buckets(n);
                assert!(buckets.len() <= n.min(records.len()));
                assert_eq!(
                    buckets.iter().map(|b| b.ticks).sum::<usize>(),
                    records.len(),
                    "cell {cell}, {n} bucket(s) cover every tick"
                );
                let mut got: BTreeMap<ViolationKind, usize> = BTreeMap::new();
                for bucket in &buckets {
                    for (kind, count) in &bucket.counts {
                        *got.entry(*kind).or_default() += count;
                    }
                }
                assert_eq!(got, want, "cell {cell}, {n} bucket(s)");
            }
        }
        // The loudest source in a bucket is what it shows.
        let folded = view.timeline("nominal-01").buckets(1);
        assert_eq!(folded[0].source, EventSource::Clamped);
        assert_eq!(
            view.timeline("nominal-00").buckets(1)[0].source,
            EventSource::Policy
        );
    }

    /// The per-kind summary labels both clocks, the strip's frame index first: a run records
    /// several physics ticks per control frame, and only the frame is a place on the strip.
    #[test]
    fn a_kind_row_labels_the_frame_and_the_tick() {
        let view = RunView::open(&fixture()).expect("open the fixture run");
        let timeline = view.timeline("nominal-01");
        assert_eq!(timeline.heading("nominal-01"), "nominal-01: 4 frame(s)");
        let rows = timeline.kind_rows();
        assert_eq!(rows.len(), 2, "{rows:?}");
        let velocity = rows
            .iter()
            .find(|r| r.kind == ViolationKind::Velocity)
            .expect("the fixture clamps velocity");
        assert_eq!((velocity.frames, velocity.first.frame), (2, 1));
        assert_eq!(
            velocity.label(),
            "Velocity: 2 frame(s), first at frame 1 (tick 1)"
        );
        assert!(view.timeline("nominal-00").kind_rows().is_empty());
    }

    /// The decoding is the encoding read backwards, for every kind (spec 9.3).
    #[test]
    fn every_violation_kind_round_trips_through_the_bits() {
        for kind in ViolationKind::ALL {
            let mut set = EventSet::EMPTY;
            set.insert(kind);
            assert_eq!(decode_events(set.bits()), set, "{kind:?}");
        }
        assert!(decode_events(0).is_empty());
    }

    /// A real run keeps its frames where `--frames` pointed, which is usually not inside the
    /// run directory: the filmstrip follows [`RunView::set_frames_root`] there.
    #[test]
    fn the_frames_root_can_point_outside_the_run() {
        let outside = scratch("frames-elsewhere");
        let cell = outside.join("nominal-00");
        fs::create_dir_all(&cell).expect("cell dir");
        for name in ["layout.json", "000000.bin"] {
            fs::copy(
                fixture().join("frames/nominal-00").join(name),
                cell.join(name),
            )
            .expect("copy frame");
        }

        // A run directory whose own `frames/` is not there at all.
        let dir = scratch("frames-none");
        fs::copy(fixture().join("report.json"), dir.join("report.json")).expect("copy report");
        let mut view = RunView::open(&dir).expect("open");
        assert!(view.status.contains("no frames/"), "{}", view.status);
        assert_eq!(view.cells().len(), 2, "one row per suite");
        assert!(view.frame("nominal-00", 0).is_none());

        view.set_frames_root(&outside);
        assert_eq!(view.frames_root(), outside.as_path());
        assert!(!view.status.contains("no frames/"), "{}", view.status);
        let row = view
            .cells()
            .iter()
            .find(|r| r.name == "nominal-00")
            .expect("the frames directory named the cell");
        assert_eq!(row.frames, 1);
        assert_eq!(row.suite, "nominal", "the row still joins its suite");
        assert_eq!(view.filmstrip("nominal-00", 8), vec![0]);
        assert_eq!(
            view.frame("nominal-00", 0).expect("frame 0").data.len(),
            96 * 96 * 3
        );
        for dir in [outside, dir] {
            fs::remove_dir_all(dir).ok();
        }
    }

    /// Oracle 3: `report.json` alone opens, and says what is not there.
    #[test]
    fn a_report_alone_still_opens() {
        let dir = scratch("report-alone");
        fs::copy(fixture().join("report.json"), dir.join("report.json")).expect("copy report");
        let view = RunView::open(&dir).expect("a report alone opens");

        assert_eq!(
            view.cells().len(),
            2,
            "one row per suite when nothing else is on disk"
        );
        assert!(view.timeline("nominal").rows.is_empty());
        assert!(view.frame("nominal", 0).is_none());
        assert!(view.filmstrip("nominal", 8).is_empty());
        assert!(view.selected_cell().is_none());
        for missing in ["events.json", "traj/", "frames/"] {
            assert!(view.status.contains(missing), "status: {}", view.status);
        }
        assert_eq!(view.acceptance().len(), 2, "the report's own lines survive");

        // A directory with no report at all is an error that names the file.
        let empty = scratch("empty");
        assert!(!RunView::is_run_dir(&empty));
        let err = RunView::open(&empty).expect_err("no report.json");
        assert!(err.to_string().contains("report.json"), "{err}");
        fs::remove_dir_all(&dir).ok();
        fs::remove_dir_all(&empty).ok();
    }

    /// Oracle 4: every column sorts to a permutation of the rows, and ties keep their order.
    #[test]
    fn sorting_is_stable_and_total() {
        let mut view = RunView::open(&fixture()).expect("open the fixture run");
        let names =
            |v: &RunView| -> Vec<String> { v.cells().iter().map(|r| r.name.clone()).collect() };
        let mut original = names(&view);
        original.sort();
        let columns = view.columns().len();
        assert!(columns >= 4, "{:?}", view.columns());

        for column in 0..columns {
            view.sort_by(column);
            let mut got = names(&view);
            assert_eq!(got.len(), original.len(), "column {column} dropped a row");
            got.sort();
            assert_eq!(got, original, "column {column} is not a permutation");
        }
        // Stability: the rows go into name order, then sort by a column where whole suites
        // tie, so the name order must survive inside each suite.
        view.sort_by(0);
        view.sort_by(1);
        let after = names(&view);
        for suite in ["light", "nominal"] {
            let in_suite: Vec<&String> = after.iter().filter(|n| n.starts_with(suite)).collect();
            let mut sorted = in_suite.clone();
            sorted.sort();
            assert_eq!(in_suite, sorted, "ties reordered inside {suite}");
        }
        // Past the last column nothing happens.
        let before = names(&view);
        view.sort_by(columns + 3);
        assert_eq!(names(&view), before);
    }

    /// The filmstrip samples evenly, ends included, and never more than asked for.
    #[test]
    fn the_filmstrip_samples_evenly() {
        let view = RunView::open(&fixture()).expect("open the fixture run");
        assert_eq!(
            view.filmstrip("nominal-00", 8),
            vec![0],
            "one frame on disk"
        );
        assert!(view.filmstrip("no-such-cell", 8).is_empty());
        assert!(
            Timeline::default().buckets(4).is_empty(),
            "no ticks, no columns"
        );
    }

    /// Selection is the whole coupling to the replay panel (packet M7/E2).
    #[test]
    fn selecting_a_cell_names_its_trajectory() {
        let mut view = RunView::open(&fixture()).expect("open the fixture run");
        view.select("light-01");
        let selected = view.selected_cell().expect("selected");
        assert_eq!(selected.name, "light-01");
        assert_eq!(selected.suite, "light");
        assert!(view.traj_path("light-01").ends_with("light-01.estraj"));
        view.select("no-such-cell");
        assert!(view.selected_cell().is_none());
    }
}
