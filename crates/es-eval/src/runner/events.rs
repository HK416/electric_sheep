//! What a run says about itself as it goes: the live [`RunEvent`]s a [`RunSink`] hears
//! (packet M7/E4) and the per-frame [`StepEvent`] records `events.json` holds.

use es_core::PhysTick;
use es_ir::evaluation::CellResult;
use es_safety::ActionSource;
use serde::{Deserialize, Serialize};

/// Where a run says, as it happens, what it just did (packet M7/E4).
///
/// A plain closure and not a trait (`INV-17` allows seven extension points and this is none
/// of them), taking a *borrowed* description of one moment of the run. Nothing here knows
/// about a socket or a wire frame: `es-telemetry` is layer 10 exactly like this crate and
/// §4.2 forbids a same-layer dependency, so the caller — `es eval run`, which links both —
/// turns a [`RunEvent`] into an `es_telemetry::protocol::Frame`. With no sink the run is
/// byte-for-byte what it was before: `report.json` and `events.json` are written from the
/// same numbers whether anyone is listening or not.
pub type RunSink<'a> = dyn FnMut(RunEvent<'_>) + 'a;

/// One moment of a running evaluation, for a [`RunSink`].
///
/// "Cell" here is the on-disk cell — one *episode* of one suite, named `<suite>-<NN>`, which
/// is what `events.json`, `frames/<cell>/` and `traj/<cell>.estraj` all key on. The §10.1
/// table's row is the *suite*, which is what [`RunEvent::SuiteEnd`] carries.
#[derive(Debug)]
pub enum RunEvent<'a> {
    /// One episode starts, with the identity a viewer needs to name its row.
    CellBegin {
        cell: &'a str,
        suite: &'a str,
        seed: u64,
        episode: u64,
    },
    /// The observation image this tick captured, borrowed from the plan's own input buffer —
    /// the bytes the renderer wrote, not a second render and not a copy. `shape` is the
    /// plan's declared `[h, w, c]`.
    Observation {
        cell: &'a str,
        tick: PhysTick,
        shape: &'a [u64],
        bytes: &'a [u8],
    },
    /// One control tick that captured an observation, as the same [`StepEvent`] `events.json`
    /// records for it — the plane's own verdict, not a re-derivation.
    Tick { cell: &'a str, event: StepEvent },
    /// The episode is over and its files are written.
    CellEnd { cell: &'a str, end: CellEnd },
    /// Every episode of one suite is done **on this shard** and its §10.1 row has been
    /// computed, before the report exists. A worker that owns only a slice of the suite
    /// cannot say what the row came to and stays quiet; `--telemetry` needs `--jobs 1`
    /// anyway, so the only publisher is the one that owns every episode (packet M7/R1).
    SuiteEnd {
        suite: &'a str,
        results: &'a [CellResult],
    },
}

/// What one finished episode leaves behind, for the §12.4 metric set a live viewer shows.
///
/// Every number is one the run already had: nothing is measured for telemetry's sake, and a
/// field the run cannot fill honestly is absent rather than zero (§12.4).
#[derive(Clone, Debug, PartialEq)]
pub struct CellEnd {
    /// Control ticks the episode ran.
    pub steps: u64,
    /// Policy invocations: one per re-plan period (§8.4), which is `steps / replan` rounded up
    /// because the submission condition is `step % replan == 0`.
    pub inferences: u64,
    /// Frames written under `frames/<cell>/`; `0` without `--frames`.
    pub frames: u64,
    /// Whether a `traj/<cell>.estraj` was written beside them.
    pub traj: bool,
    /// `SafetyCounters::chunk_underrun_rate` as it stands at the end of this episode (§10.3).
    pub chunk_underrun_rate: f64,
    /// `es_env::Termination` as it spells itself.
    pub outcome: String,
}

/// Where the emitted action came from, as `es video mosaic` spells it.
///
/// The same four outcomes `es-data` writes into a dataset's `action_source` column
/// (`es_data::ActionSourceCode`, `crates/es-data/src/collect.rs:552`), duplicated rather than
/// shared because `es-data` is layer 10 like this crate and §4.2 forbids a same-layer
/// dependency. `Human` cannot occur here — an evaluation has no teleop — but it is one of the
/// four the overlay reads, so the variant stays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSource {
    Policy,
    Clamped,
    Fallback,
    Human,
}

impl From<ActionSource> for EventSource {
    /// Read off the plane's own per-step output rather than re-derived from counter deltas:
    /// `SafeAction` already says which of the four this step was (§9.3, §9.4).
    fn from(s: ActionSource) -> Self {
        match s {
            ActionSource::Policy => Self::Policy,
            ActionSource::Clamped => Self::Clamped,
            ActionSource::Fallback(_) => Self::Fallback,
        }
    }
}

/// One record per rendered frame, written as `events.json` beside `frames/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepEvent {
    /// Index of the frame this describes inside its cell, dense and ascending from 0.
    pub frame: u64,
    /// Control tick **inside this episode**, counted from the reset that opened it (§10.5,
    /// packet M7/R1) — not the env's cumulative physics clock, which depended on how many
    /// episodes the same `Env` had already run and therefore on how the run was scheduled.
    pub tick: PhysTick,
    pub source: EventSource,
    /// `es_safety::EventSet::bits()` for this step: the `ViolationKind` bitset, `0` when the
    /// step was clean.
    pub events: u32,
}
