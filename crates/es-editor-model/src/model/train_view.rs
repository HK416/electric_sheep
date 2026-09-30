//! The Live tab's Training section (spec 23.3, spec 19.3; packet M7/E7): the learning curve,
//! the marks on it, and the tensor the network is fitting.
//!
//! `es train --telemetry` publishes one stream-5 frame per progress line
//! (`[step, loss, lr, samples_per_s]`), a stream-1 `checkpoint` event per packed mark and the
//! sample image on stream 4 (`docs/design/telemetry-protocol.md` "Producers"). This file folds
//! them into numbers a painter can draw and **decides how the curve is scaled**, because
//! `app.rs` decides nothing (spec 28.10 rule 3): [`TrainView::plot`] hands back points in the
//! unit square and the range they were normalised against, so the shell maps a rectangle and
//! paints a polyline — there is no plotting dependency and there is not going to be one.
//!
//! An `[rl]` run adds stream 6 (`[step, return, episode_len, success, entropy,
//! envelope_violation_rate]`, packet M16/H4) -- the numbers a PPO run is judged by, since its
//! loss says almost nothing -- folded into [`RlCurve`]. A finished `es train` output folder is
//! read into the same view by [`TrainView::open_dir`], so the finished run and the live one are
//! the same plots; `--attach` on top extends them.
//!
//! What it deliberately does not hold: per-node activation statistics (§23.3 wants them; the
//! lowered module has no hook for them) and any control over the run — the trainer speaks no
//! control protocol, so there is nothing to offer.

use std::path::Path;
use std::time::Duration;

use es_eval::run_dir::Rgb8Image;
use es_telemetry::protocol::{Message, Payload, StreamId};

use crate::model::live_run::{rgb8, STREAM_EVENTS, STREAM_IMAGE};

/// The training curve's stream. Data and not schema, like streams 1-4
/// (`docs/design/telemetry-protocol.md` §9): `protocol.rs` names no stream id.
pub const STREAM_TRAIN: StreamId = StreamId(5);

/// A PPO run's learning (packet M16/H4). Data and not schema, as stream 5: an editor that does
/// not know 6 never subscribes to it, and stream 5 stays the four numbers old viewers expect.
pub const STREAM_RL: StreamId = StreamId(6);

/// What `es train` writes that says a folder is its output (spec 19.3).
pub const TRAINING_LOCK: &str = "training.lock";

/// The directories `es train` makes before its trainer starts, so a run still going -- which
/// has no `training.lock` until it ends -- is recognised too.
const RUN_DIRS: [&str; 3] = ["weights", "metrics", "checkpoints"];

/// Whether `dir` is an `es train --out` folder, finished or still running.
pub fn is_train_dir(dir: &Path) -> bool {
    dir.join(TRAINING_LOCK).is_file() || RUN_DIRS.iter().all(|d| dir.join(d).is_dir())
}

/// A PPO run's learning as it arrived: one entry per stream-6 row (or `loss-curve.json` row).
/// A number nobody measured -- a success rate over no finished episode -- is NaN, never zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RlCurve {
    pub step: Vec<u32>,
    pub ret: Vec<f32>,
    pub episode_len: Vec<f32>,
    pub success: Vec<f32>,
    pub entropy: Vec<f32>,
    pub violation: Vec<f32>,
}

/// The learning curve as it arrived: one entry per progress line, oldest first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Curve {
    pub step: Vec<u32>,
    pub loss: Vec<f32>,
    pub lr: Vec<f32>,
}

impl Curve {
    pub fn len(&self) -> usize {
        self.step.len()
    }

    pub fn is_empty(&self) -> bool {
        self.step.is_empty()
    }
}

/// Which curve a caller wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Series {
    Loss,
    Lr,
    Return,
    Success,
    EpisodeLen,
    Entropy,
    Violation,
}

impl Series {
    /// A PPO run's curves, in the order the Live tab draws them.
    pub const RL: [Series; 5] = [
        Series::Return,
        Series::Success,
        Series::EpisodeLen,
        Series::Entropy,
        Series::Violation,
    ];

    /// The string-table key of the curve's plain-language name.
    pub fn key(self) -> &'static str {
        match self {
            Series::Loss => "live.loss",
            Series::Lr => "live.lr",
            Series::Return => "live.return",
            Series::Success => "live.success",
            Series::EpisodeLen => "live.episode_len",
            Series::Entropy => "live.entropy",
            Series::Violation => "live.violation",
        }
    }
}

/// One curve, ready to paint: points in the unit square with `y = 0` at the bottom of the
/// range, the range itself for the two axis labels, and the checkpoint marks as `x`.
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    pub points: Vec<[f32; 2]>,
    /// The values `0.0` and `1.0` on the `y` axis stand for, in the series' own units — not
    /// the log of them, even when `log` was asked for, because that is what a person reads.
    pub min: f32,
    pub max: f32,
    /// Where a packed checkpoint falls along `x`.
    pub marks: Vec<f32>,
    /// The optimizer steps `x = 0.0` and `x = 1.0` stand for.
    pub steps: [u32; 2],
}

impl Plot {
    /// Where optimizer step `step` falls along `x`, clamped to the curve: a checkpoint's mark,
    /// and a checkpoint preview's (packet M13/Z5a).
    pub fn x(&self, step: u32) -> f32 {
        let [x0, x1] = self.steps.map(|s| s as f32);
        let span = if (x1 - x0).abs() < f32::EPSILON {
            1.0
        } else {
            x1 - x0
        };
        ((step as f32 - x0) / span).clamp(0.0, 1.0)
    }
}

/// A training run being watched.
#[derive(Clone, Debug, Default)]
pub struct TrainView {
    curve: Curve,
    rl: RlCurve,
    /// The last step read off a finished run's folder: a live row at or before it is one the
    /// file already holds, and is not drawn twice.
    floor: Option<u32>,
    checkpoints: Vec<(u32, String)>,
    sample: Option<Rgb8Image>,
    samples: u64,
    /// The trainer's own `samples_per_s` from the last progress line.
    throughput: Option<f64>,
    /// The recipe's `[run] steps`, stated once by `es train` before the trainer starts.
    total: Option<u32>,
    /// `(wall_ns, step)` of the first progress frame and the wall clock of the last — the
    /// **producer's** clock, which is what makes the rate the run's rather than the viewer's.
    first: Option<(u64, u32)>,
    last_wall: u64,
    /// Whether the loss curve is drawn on a log scale. A view-model field and not a widget's
    /// own memory, so the choice survives a tab switch and is judged headlessly.
    pub log_scale: bool,
}

impl TrainView {
    /// Nothing has arrived: the section says so rather than drawing an empty box.
    pub fn is_empty(&self) -> bool {
        self.curve.is_empty()
            && self.rl.step.is_empty()
            && self.checkpoints.is_empty()
            && self.total.is_none()
    }

    /// A finished (or still running) `es train --out` folder as the view a live run builds
    /// (packet M16/H4): the packed marks from `training.lock`, and the curve from
    /// `metrics/loss-curve.json` (`[rl]`: one object per iteration) or `metrics/loss.json`
    /// (the IR route: one loss per optimizer step). A run still going has neither file yet,
    /// and `--attach` fills the curve. Neither file carries the learning rate or a success
    /// rate, so those curves stay empty rather than drawn from a stand-in.
    pub fn open_dir(dir: &Path) -> Result<TrainView, String> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name));
        let json = |name: &str, text: String| {
            serde_json::from_str::<serde_json::Value>(&text)
                .map_err(|e| format!("{}: {e}", dir.join(name).display()))
        };
        if !is_train_dir(dir) {
            return Err(format!("{}: not an es train output folder", dir.display()));
        }
        let mut view = TrainView::default();
        if let Ok(text) = read(TRAINING_LOCK) {
            let lock = json(TRAINING_LOCK, text)?;
            for mark in lock["checkpoints"].as_array().into_iter().flatten() {
                if let Some(step) = mark["step"].as_u64() {
                    let hash = mark["policy_hash"].as_str().unwrap_or_default();
                    view.checkpoints.push((step as u32, hash.to_owned()));
                }
            }
        }
        let number = |v: &serde_json::Value| v.as_f64().map_or(f32::NAN, |x| x as f32);
        if let Ok(text) = read("metrics/loss-curve.json") {
            let rows = json("metrics/loss-curve.json", text)?;
            for (i, row) in rows.as_array().into_iter().flatten().enumerate() {
                let step = i as u32 + 1;
                view.curve.step.push(step);
                view.curve.loss.push(number(&row["loss"]));
                view.curve.lr.push(f32::NAN);
                let rl = &mut view.rl;
                rl.step.push(step);
                rl.ret.push(number(&row["return"]));
                rl.episode_len.push(number(&row["episode_len"]));
                rl.success.push(f32::NAN);
                rl.entropy.push(number(&row["entropy"]));
                rl.violation.push(number(&row["envelope_violation_rate"]));
                view.throughput = row["samples_per_sec"].as_f64();
            }
        } else if let Ok(text) = read("metrics/loss.json") {
            let losses = json("metrics/loss.json", text)?;
            for (i, loss) in losses.as_array().into_iter().flatten().enumerate() {
                view.curve.step.push(i as u32 + 1);
                view.curve.loss.push(number(loss));
                view.curve.lr.push(f32::NAN);
            }
        }
        view.floor = view.curve.step.last().copied();
        Ok(view)
    }

    /// Folds one message. Anything that is not the training producer's is ignored, so a
    /// cycle's collect and evaluation frames travel the same socket without confusing this.
    pub fn ingest(&mut self, msg: &Message) {
        let Message::Frame(frame) = msg else {
            return;
        };
        match (frame.stream, &frame.payload) {
            (STREAM_TRAIN, Payload::Scalars(v)) => self.progress(v, frame.wall_ns),
            (STREAM_RL, Payload::Scalars(v)) => self.rl_progress(v),
            (STREAM_EVENTS, Payload::Event { kind, fields }) => match kind.as_str() {
                "train.begin" => {
                    self.total = fields.get("total_steps").and_then(|s| s.parse().ok());
                }
                "checkpoint" => {
                    let step = fields.get("step").and_then(|s| s.parse().ok()).unwrap_or(0);
                    let hash = fields.get("policy_hash").cloned().unwrap_or_default();
                    // Already read off the run's `training.lock`, when its folder is open.
                    if !self.checkpoints.iter().any(|(s, _)| *s == step) {
                        self.checkpoints.push((step, hash));
                    }
                }
                _ => {}
            },
            // The sample image shares stream 4 with the observation frame: which one a frame
            // is, is which stage sent it, and a training run sends only this one.
            (
                STREAM_IMAGE,
                Payload::Image {
                    w,
                    h,
                    format,
                    bytes,
                },
            ) => {
                if let Some(image) = rgb8(*w, *h, format, bytes) {
                    self.sample = Some(image);
                    self.samples += 1;
                }
            }
            _ => {}
        }
    }

    /// One `[step, loss, lr, samples_per_s]`. A frame with fewer numbers is not this stream's
    /// and is dropped rather than padded with zeros.
    fn progress(&mut self, v: &[f64], wall_ns: u64) {
        let [step, loss, lr, throughput] = v else {
            return;
        };
        let step = *step as u32;
        if self.floor.is_some_and(|f| step <= f) {
            return;
        }
        self.first.get_or_insert((wall_ns, step));
        self.last_wall = wall_ns;
        self.curve.step.push(step);
        self.curve.loss.push(*loss as f32);
        self.curve.lr.push(*lr as f32);
        self.throughput = throughput.is_finite().then_some(*throughput);
    }

    /// One `[step, return, episode_len, success, entropy, envelope_violation_rate]`; a
    /// shorter row is not this stream's, and numbers a later producer appends are ignored.
    fn rl_progress(&mut self, v: &[f64]) {
        let [step, ret, len, success, entropy, violation, ..] = *v else {
            return;
        };
        let step = step as u32;
        if self.floor.is_some_and(|f| step <= f) {
            return;
        }
        let rl = &mut self.rl;
        rl.step.push(step);
        rl.ret.push(ret as f32);
        rl.episode_len.push(len as f32);
        rl.success.push(success as f32);
        rl.entropy.push(entropy as f32);
        rl.violation.push(violation as f32);
    }

    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    pub fn rl(&self) -> &RlCurve {
        &self.rl
    }

    /// `(step, policy_hash)` per packed mark, in the order they were packed.
    pub fn checkpoints(&self) -> &[(u32, String)] {
        &self.checkpoints
    }

    /// The last sample image, and how many have arrived — the sequence number a viewer keys
    /// its texture on, so one image is uploaded once and not once per repaint.
    pub fn sample(&self) -> Option<&Rgb8Image> {
        self.sample.as_ref()
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// The trainer's own throughput, `None` until a progress line carries one. Never a zero
    /// standing in for a number nobody measured (spec 12.4).
    pub fn throughput(&self) -> Option<f64> {
        self.throughput
    }

    /// The optimizer step the last progress line reported.
    pub fn step(&self) -> Option<u32> {
        self.curve.step.last().copied()
    }

    /// How many steps the recipe asked for, once the run has said.
    pub fn total(&self) -> Option<u32> {
        self.total
    }

    /// What is left, at the rate the run has been going.
    ///
    /// The rate is the run's own: the `wall_ns` the **producer** stamped on the first and last
    /// progress frames, over the steps between them. An editor that attached late, or that its
    /// window manager stopped repainting, must not invent a speed the run does not have.
    /// `None` before two points and `None` once `total_steps` is reached — a finished run has
    /// no estimate.
    pub fn eta(&self, total_steps: u32) -> Option<Duration> {
        let (wall0, step0) = self.first?;
        let last = *self.curve.step.last()?;
        if last <= step0 || last >= total_steps {
            return None;
        }
        let elapsed_ns = self.last_wall.checked_sub(wall0)? as f64;
        let per_step = elapsed_ns / f64::from(last - step0);
        Duration::try_from_secs_f64(per_step * f64::from(total_steps - last) / 1e9).ok()
    }

    /// One curve in the unit square. `None` until there are two points to draw a line between.
    ///
    /// `log` takes the base-10 log of each loss before normalising, which is how a curve that
    /// falls by two orders of magnitude in its first tenth is readable at all; a non-positive
    /// value cannot be logged and is dropped from the plotted set rather than clamped to
    /// something it is not. The reported `min`/`max` stay the series' own units either way.
    pub fn plot(&self, series: Series, log: bool) -> Option<Plot> {
        let (steps, values): (&[u32], &[f32]) = match series {
            Series::Loss => (&self.curve.step, &self.curve.loss),
            Series::Lr => (&self.curve.step, &self.curve.lr),
            Series::Return => (&self.rl.step, &self.rl.ret),
            Series::Success => (&self.rl.step, &self.rl.success),
            Series::EpisodeLen => (&self.rl.step, &self.rl.episode_len),
            Series::Entropy => (&self.rl.step, &self.rl.entropy),
            Series::Violation => (&self.rl.step, &self.rl.violation),
        };
        let pairs: Vec<(u32, f32)> = steps
            .iter()
            .copied()
            .zip(values.iter().copied())
            .filter(|(_, v)| v.is_finite() && (!log || *v > 0.0))
            .collect();
        if pairs.len() < 2 {
            return None;
        }
        let scaled = |v: f32| if log { v.log10() } else { v };
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (_, v) in &pairs {
            lo = lo.min(scaled(*v));
            hi = hi.max(scaled(*v));
        }
        // A flat curve has no range to normalise against; draw it down the middle rather than
        // dividing by zero.
        let span = if (hi - lo).abs() < f32::EPSILON {
            1.0
        } else {
            hi - lo
        };
        let mut plot = Plot {
            points: Vec::new(),
            min: pairs.iter().map(|(_, v)| *v).fold(f32::INFINITY, f32::min),
            max: pairs
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max),
            marks: Vec::new(),
            steps: [pairs[0].0, pairs[pairs.len() - 1].0],
        };
        plot.points = pairs
            .iter()
            .map(|(s, v)| [plot.x(*s), ((scaled(*v) - lo) / span).clamp(0.0, 1.0)])
            .collect();
        plot.marks = self.checkpoints.iter().map(|(s, _)| plot.x(*s)).collect();
        Some(plot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::telemetry_view::{replay, TelemetryModel};
    use es_core::PhysTick;
    use es_telemetry::protocol::Frame;
    use std::collections::BTreeMap;
    use std::path::Path;

    /// One progress frame, stamped one second per optimizer step so `eta` has a rate to read
    /// — the producer's clock is what it measures, so the fixture has to carry one.
    fn scalars(stream: StreamId, v: Vec<f64>) -> Message {
        Message::Frame(Frame {
            tick: PhysTick(0),
            wall_ns: (v.first().copied().unwrap_or_default() * 1e9) as u64,
            stream,
            payload: Payload::Scalars(v),
        })
    }

    fn event(kind: &str, fields: &[(&str, &str)]) -> Message {
        Message::Frame(Frame {
            tick: PhysTick(0),
            wall_ns: 0,
            stream: STREAM_EVENTS,
            payload: Payload::Event {
                kind: kind.to_owned(),
                fields: fields
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                    .collect::<BTreeMap<_, _>>(),
            },
        })
    }

    /// The loss curve the committed demo run's trainer wrote, as the numbers a `--progress-every`
    /// run would have published: `metrics/loss.json` is a JSON array of per-step losses, and a
    /// progress line is one of them.
    fn demo_losses() -> Vec<f64> {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/visible-learning/run/metrics/loss.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            // The fixture run carries no curve on every checkout; the synthetic fall below is
            // the same assertion about the fold.
            Err(_) => (0..50).map(|i| 1.0 / f64::from(i + 1)).collect(),
        }
    }

    /// Oracle 4 (packet M7/E7). A scripted stream 5 plus `checkpoint` events give a `Curve`
    /// holding exactly the values the trainer reported, bitwise at `f32`, with the marks at
    /// the right steps.
    #[test]
    fn train_view_folds_the_curve() {
        let losses = demo_losses();
        assert!(losses.len() >= 20, "the fixture curve has points");
        let every = 10u32;
        let mut messages = vec![event("train.begin", &[("total_steps", "1000")])];
        let mut expected: Vec<f32> = Vec::new();
        for (i, loss) in losses.iter().enumerate() {
            let step = (i as u32 + 1) * every;
            messages.push(scalars(
                STREAM_TRAIN,
                vec![f64::from(step), *loss, 1e-4, 128.0],
            ));
            expected.push(*loss as f32);
        }
        messages.push(event(
            "checkpoint",
            &[("step", "40"), ("policy_hash", "ab".repeat(32).as_str())],
        ));

        let mut model = TelemetryModel::default();
        let n = messages.len();
        let mut source = replay(messages);
        assert_eq!(model.pump(&mut source, 10_000), n);

        let view = &model.train;
        assert_eq!(view.curve().loss, expected, "bitwise at f32");
        assert_eq!(view.curve().len(), losses.len());
        assert_eq!(view.step(), Some(losses.len() as u32 * every));
        assert_eq!(view.total(), Some(1000));
        assert_eq!(view.throughput(), Some(128.0));
        assert_eq!(view.checkpoints().len(), 1);
        assert_eq!(view.checkpoints()[0].0, 40);

        // The plot is the same points, normalised, with the mark where step 40 falls.
        let plot = view.plot(Series::Loss, false).expect("two points");
        assert_eq!(plot.points.len(), losses.len());
        assert!(plot
            .points
            .iter()
            .all(|[x, y]| (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y)));
        // Both ends are exact by construction: the first and last step normalise to 0 and 1.
        assert_eq!(
            (plot.points[0][0], plot.points[plot.points.len() - 1][0]),
            (0.0, 1.0)
        );
        assert_eq!(plot.marks.len(), 1);
        let want = (40.0 - every as f32) / (view.step().expect("step") - every) as f32;
        assert!((plot.marks[0] - want).abs() < 1e-6, "{:?}", plot.marks);
        assert_eq!(
            plot.x(40).to_bits(),
            plot.marks[0].to_bits(),
            "a preview of the same step marks the same x"
        );
        assert_eq!(
            (plot.x(0), plot.x(u32::MAX)),
            (0.0, 1.0),
            "clamped to the curve"
        );
        assert!(plot.min <= plot.max);
        println!(
            "RAN train_view_folds_the_curve: {} point(s), min {} max {}",
            plot.points.len(),
            plot.min,
            plot.max
        );
    }

    /// The log scale drops what cannot be logged rather than clamping it, and the range it
    /// reports is still in the series' own units.
    #[test]
    fn the_log_scale_drops_a_non_positive_loss_and_reports_real_units() {
        let mut view = TrainView::default();
        for (step, loss) in [(10.0, 1.0), (20.0, 0.0), (30.0, 0.01)] {
            view.ingest(&scalars(STREAM_TRAIN, vec![step, loss, 1e-4, 1.0]));
        }
        let linear = view.plot(Series::Loss, false).expect("three points");
        assert_eq!(linear.points.len(), 3);
        assert_eq!((linear.min, linear.max), (0.0, 1.0));

        let log = view.plot(Series::Loss, true).expect("two positive points");
        assert_eq!(log.points.len(), 2, "the zero is dropped, not clamped");
        assert_eq!((log.min, log.max), (0.01, 1.0), "units, not their logs");
        // Two points, two ends: the smaller loss sits at the bottom.
        assert_eq!((log.points[0][1], log.points[1][1]), (1.0, 0.0));
    }

    /// `eta` needs two points and a rate, says nothing before it has them, and shrinks as the
    /// run goes on.
    #[test]
    fn eta_is_none_until_there_is_a_rate_and_shrinks_after() {
        let mut view = TrainView::default();
        assert_eq!(view.eta(1000), None, "nothing has happened yet");
        view.ingest(&scalars(STREAM_TRAIN, vec![10.0, 1.0, 1e-4, 8.0]));
        assert_eq!(view.eta(1000), None, "one point is not a rate");
        view.ingest(&scalars(STREAM_TRAIN, vec![20.0, 0.5, 1e-4, 8.0]));
        let early = view.eta(1000).expect("two points and a rate");
        view.ingest(&scalars(STREAM_TRAIN, vec![900.0, 0.1, 1e-4, 8.0]));
        let late = view.eta(1000).expect("still running");
        assert!(late < early, "{late:?} is not shorter than {early:?}");
        view.ingest(&scalars(STREAM_TRAIN, vec![1000.0, 0.1, 1e-4, 8.0]));
        assert_eq!(view.eta(1000), None, "a finished run has no estimate");
    }

    /// Packet M16/H4 oracle 2. A recorded `[rl]` run -- stream 5 and stream 6 interleaved, as
    /// `es train --telemetry` sends them -- folds into the loss curve it always did and the
    /// learning curves beside it; a success over no finished episode is not drawn, a short
    /// stream-6 row is dropped, and a longer one keeps its first six numbers.
    #[test]
    #[allow(clippy::float_cmp)] // exact: the fold keeps the numbers it was sent
    fn an_rl_run_folds_its_learning_beside_the_loss() {
        let nan = f64::NAN;
        let mut messages = vec![event("train.begin", &[("total_steps", "4")])];
        for (step, ret, len, success, entropy, violation) in [
            (1.0, 0.5, 40.0, nan, -4.5, 0.5),
            (2.0, 1.5, 64.0, 0.25, -4.0, 0.25),
            (3.0, 2.0, 96.0, 0.5, -2.0, 0.125),
        ] {
            messages.push(scalars(STREAM_TRAIN, vec![step, 0.1 / step, 3e-4, 512.0]));
            messages.push(scalars(
                STREAM_RL,
                vec![step, ret, len, success, entropy, violation],
            ));
        }
        messages.push(scalars(STREAM_RL, vec![4.0, 9.0, 9.0]));
        messages.push(scalars(
            STREAM_RL,
            vec![4.0, 0.75, 32.0, 0.0, 3.5, 1.0, 7.0],
        ));
        let mut model = TelemetryModel::default();
        let mut source = replay(messages);
        model.pump(&mut source, 100);
        let view = &model.train;

        assert_eq!(view.curve().len(), 3, "stream 5 is unchanged");
        let rl = view.rl();
        assert_eq!(rl.step, [1, 2, 3, 4]);
        assert_eq!(rl.ret, [0.5, 1.5, 2.0, 0.75]);
        assert_eq!(rl.episode_len, [40.0, 64.0, 96.0, 32.0]);
        assert_eq!(rl.entropy, [-4.5, -4.0, -2.0, 3.5]);
        assert_eq!(rl.violation, [0.5, 0.25, 0.125, 1.0]);
        assert!(rl.success[0].is_nan());
        assert_eq!(rl.success[1..], [0.25, 0.5, 0.0]);

        let success = view.plot(Series::Success, false).expect("three measured");
        assert_eq!(
            success.points.len(),
            3,
            "the unmeasured first row is not drawn"
        );
        assert_eq!(success.steps, [2, 4]);
        let ret = view.plot(Series::Return, false).expect("four points");
        assert_eq!((ret.min, ret.max), (0.5, 2.0));
        assert_eq!(ret.points[2][1], 1.0, "the peak is the top of the plot");
        for series in Series::RL {
            assert!(view.plot(series, false).is_some(), "{series:?}");
            for lang in crate::model::i18n::Lang::ALL {
                let label = crate::model::i18n::t(lang, series.key());
                assert_ne!(label, series.key(), "{series:?} has no {lang:?} label");
            }
        }
    }

    /// Packet M16/H4 oracle 3. An `es train --out` folder opens as the view a live run builds:
    /// the marks from `training.lock`, the loss and the learning from `loss-curve.json`. The
    /// file has no learning rate and no success rate, so those draw nothing; a live stream
    /// attached on top extends the curves past the file and repeats none of it.
    #[test]
    #[allow(clippy::float_cmp)] // exact: the fold keeps the numbers the file holds
    fn a_train_folder_opens_as_the_finished_live_view() {
        let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
            .join("tests/fixtures/editor/train-rl");
        assert_eq!(
            crate::model::recent::classify(&dir),
            crate::model::recent::Kind::Training
        );
        let mut view = TrainView::open_dir(&dir).expect("the fixture folder");
        let hashes = |c: char| c.to_string().repeat(64);
        assert_eq!(
            view.checkpoints(),
            [(2, hashes('a')), (4, hashes('b'))].as_slice()
        );
        assert_eq!(view.curve().loss, [0.5, 0.25, 0.125, 0.0625]);
        assert_eq!(view.rl().ret, [0.25, 1.5, 2.0, 0.75]);
        assert_eq!(view.rl().episode_len, [40.0, 64.0, 96.0, 32.0]);
        assert_eq!(view.rl().entropy, [-4.5, -4.0, -2.0, 3.5]);
        assert_eq!(view.rl().violation, [0.5, 0.25, 0.125, 1.0]);
        assert_eq!((view.step(), view.throughput()), (Some(4), Some(4096.0)));
        assert!(!view.is_empty());
        assert!(view.plot(Series::Return, false).is_some());
        assert!(view.plot(Series::Lr, false).is_none(), "the file has no lr");
        assert!(
            view.plot(Series::Success, false).is_none(),
            "nor a success rate"
        );

        view.ingest(&scalars(STREAM_TRAIN, vec![4.0, 9.0, 3e-4, 1.0]));
        view.ingest(&scalars(STREAM_RL, vec![4.0, 9.0, 9.0, 1.0, 9.0, 9.0]));
        view.ingest(&event(
            "checkpoint",
            &[("step", "4"), ("policy_hash", "cc".repeat(32).as_str())],
        ));
        assert_eq!(view.curve().len(), 4, "step 4 is the file's already");
        assert_eq!(view.rl().step.len(), 4);
        assert_eq!(view.checkpoints().len(), 2, "mark 4 is the lock's already");
        view.ingest(&scalars(STREAM_TRAIN, vec![5.0, 0.03, 3e-4, 1.0]));
        view.ingest(&scalars(STREAM_RL, vec![5.0, 1.0, 50.0, 0.5, 4.0, 0.5]));
        assert_eq!(view.curve().step.last(), Some(&5));
        assert_eq!(view.rl().step.last(), Some(&5));
        assert!(
            TrainView::open_dir(&dir.join("metrics")).is_err(),
            "not a run"
        );

        // A run still going has the three directories and nothing in them yet: it opens
        // empty, for `--attach` to fill.
        let running = std::env::temp_dir().join(format!("es-train-running-{}", std::process::id()));
        for d in RUN_DIRS {
            std::fs::create_dir_all(running.join(d)).expect("scratch");
        }
        assert_eq!(
            crate::model::recent::classify(&running),
            crate::model::recent::Kind::Training
        );
        let mut live = TrainView::open_dir(&running).expect("a running folder");
        assert!(live.is_empty());
        live.ingest(&scalars(STREAM_RL, vec![1.0, 0.5, 40.0, 0.0, -4.5, 0.5]));
        assert_eq!(live.rl().step, [1]);
        let _ = std::fs::remove_dir_all(&running);
    }
}
