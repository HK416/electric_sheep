//! The traffic light of a running cycle (`docs/design/editor-redesign.md` section 6.4, packet
//! M12/Y7): plain numbers in, one verdict out. [`judge`] is a pure function - the UI reads the
//! clock and the telemetry and hands the numbers over, so every rule is judged headlessly.

/// The colour a verdict is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light {
    Grey,
    Green,
    Amber,
    Red,
}

/// What the light says about the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Starting,
    GoingWell,
    StoppedLearning,
    Slow,
    NotResponding,
    Broken,
    Stopped,
    StoppedByYou,
}

impl Verdict {
    /// A stop the person asked for is not an alarm, so it is grey like a run not yet going.
    pub fn light(self) -> Light {
        match self {
            Verdict::Starting | Verdict::StoppedByYou => Light::Grey,
            Verdict::GoingWell => Light::Green,
            Verdict::StoppedLearning | Verdict::Slow | Verdict::NotResponding => Light::Amber,
            Verdict::Broken | Verdict::Stopped => Light::Red,
        }
    }

    /// `health.<name>`: the verdict's name.
    pub fn key(self) -> &'static str {
        match self {
            Verdict::Starting => "health.starting",
            Verdict::GoingWell => "health.going_well",
            Verdict::StoppedLearning => "health.stopped_learning",
            Verdict::Slow => "health.slow",
            Verdict::NotResponding => "health.not_responding",
            Verdict::Broken => "health.broken",
            Verdict::Stopped => "health.stopped",
            Verdict::StoppedByYou => "health.stopped_by_you",
        }
    }

    /// `health.<name>.advice`: what the person can do about it.
    pub fn advice_key(self) -> &'static str {
        match self {
            Verdict::Starting => "health.starting.advice",
            Verdict::GoingWell => "health.going_well.advice",
            Verdict::StoppedLearning => "health.stopped_learning.advice",
            Verdict::Slow => "health.slow.advice",
            Verdict::NotResponding => "health.not_responding.advice",
            Verdict::Broken => "health.broken.advice",
            Verdict::Stopped => "health.stopped.advice",
            Verdict::StoppedByYou => "health.stopped_by_you.advice",
        }
    }
}

/// One point of the training curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub step: u64,
    pub loss: f64,
    pub samples_per_s: f64,
}

/// Everything [`judge`] looks at, filled by the UI from `LiveRun`, `TrainView` and
/// `LaunchModel`.
#[derive(Clone, Debug, PartialEq)]
pub struct Input<'a> {
    pub since_start_s: f64,
    /// Seconds since the last telemetry message of any stream; `None` = none yet.
    pub since_last_message_s: Option<f64>,
    pub child_alive: bool,
    pub killed: bool,
    pub exit_code: Option<i32>,
    pub stage_codes: &'a [Option<i32>],
    pub total_steps: Option<u64>,
    pub points: &'a [Point],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// `NotResponding` after this long without a message.
    pub silence_s: f64,
    /// `Slow` and `StoppedLearning` are not judged before this many points.
    pub warmup_points: usize,
    /// Latest samples/s below this times the median = `Slow`.
    pub slow_fraction: f64,
    /// Of total steps: no new loss minimum over this = `StoppedLearning`.
    pub plateau_fraction: f64,
}

/// ponytail: first guesses; Y-V calibrates them on a recorded successful run.
pub const THRESHOLDS: Thresholds = Thresholds {
    silence_s: 180.0,
    warmup_points: 20,
    slow_fraction: 1.0 / 3.0,
    plateau_fraction: 0.25,
};

/// The first rule that matches wins: `StoppedByYou` (killed) -> `Broken` (a non-finite loss)
/// -> `Stopped` (a non-zero exit or stage code) -> `NotResponding` (alive and silent past
/// `silence_s`, counted from the start when nothing has arrived) -> `Starting` (nothing has
/// arrived) -> `Slow` -> `StoppedLearning` -> `GoingWell`. `Slow` and `StoppedLearning` need
/// more than `warmup_points` points; `StoppedLearning` also needs `total_steps`.
///
/// `Broken` outranks `Stopped` (packet P-M14-R1): the trainer stops with a non-zero exit
/// *because* its loss became non-finite, and "the loss became invalid" says why the run
/// stopped where "stopped" only says that it did.
pub fn judge(input: &Input<'_>, th: &Thresholds) -> Verdict {
    let failed = |code: &Option<i32>| code.is_some_and(|c| c != 0);
    if input.killed {
        return Verdict::StoppedByYou;
    }
    let points = input.points;
    if points.iter().any(|p| !p.loss.is_finite()) {
        return Verdict::Broken;
    }
    if failed(&input.exit_code) || input.stage_codes.iter().any(failed) {
        return Verdict::Stopped;
    }
    let silent_for = input.since_last_message_s.unwrap_or(input.since_start_s);
    if input.child_alive && silent_for > th.silence_s {
        return Verdict::NotResponding;
    }
    if input.since_last_message_s.is_none() {
        return Verdict::Starting;
    }
    let Some(after_warmup @ [.., latest]) = points.get(th.warmup_points..) else {
        return Verdict::GoingWell;
    };
    let mut rates: Vec<f64> = after_warmup.iter().map(|p| p.samples_per_s).collect();
    rates.sort_by(f64::total_cmp);
    if latest.samples_per_s < th.slow_fraction * rates[rates.len() / 2] {
        return Verdict::Slow;
    }
    if let Some(total) = input.total_steps {
        let cutoff = latest.step as f64 - th.plateau_fraction * total as f64;
        let recent = lowest(points.iter().filter(|p| p.step as f64 > cutoff));
        let before = lowest(points.iter().filter(|p| p.step as f64 <= cutoff));
        if matches!((recent, before), (Some(r), Some(b)) if r >= b) {
            return Verdict::StoppedLearning;
        }
    }
    Verdict::GoingWell
}

fn lowest<'a>(points: impl Iterator<Item = &'a Point>) -> Option<f64> {
    points.map(|p| p.loss).reduce(f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(points: &[Point]) -> Input<'_> {
        Input {
            since_start_s: 600.0,
            since_last_message_s: Some(1.0),
            child_alive: true,
            killed: false,
            exit_code: None,
            stage_codes: &[],
            total_steps: Some(20_000),
            points,
        }
    }

    fn falling(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| Point {
                step: (i as u64 + 1) * 100,
                loss: 1.0 / (i as f64 + 1.0),
                samples_per_s: 40.0,
            })
            .collect()
    }

    #[test]
    fn a_falling_curve_is_going_well() {
        assert_eq!(judge(&base(&falling(100)), &THRESHOLDS), Verdict::GoingWell);
    }

    #[test]
    fn a_kill_is_stopped_by_you() {
        let mut i = base(&[]);
        i.killed = true;
        i.child_alive = false;
        i.exit_code = Some(1);
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::StoppedByYou);
    }

    #[test]
    fn a_bad_exit_or_stage_code_is_stopped() {
        let mut i = base(&[]);
        i.child_alive = false;
        i.exit_code = Some(4);
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::Stopped);
        let codes = [Some(0), Some(3)];
        let mut j = base(&[]);
        j.stage_codes = &codes;
        assert_eq!(judge(&j, &THRESHOLDS), Verdict::Stopped);
    }

    #[test]
    fn nan_is_broken_even_as_the_first_point() {
        let p = [Point {
            step: 1,
            loss: f64::NAN,
            samples_per_s: 1.0,
        }];
        assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Broken);
        let q = [Point {
            step: 1,
            loss: f64::INFINITY,
            samples_per_s: 1.0,
        }];
        assert_eq!(judge(&base(&q), &THRESHOLDS), Verdict::Broken);
    }

    /// Packet P-M14-R1: the trainer exits non-zero because the loss became NaN, and the light
    /// says the cause, not only the stop. A kill still reads as the person's own stop.
    #[test]
    fn a_nonfinite_loss_outranks_a_bad_exit() {
        let mut p = falling(30);
        p.last_mut().unwrap().loss = f64::NAN;
        let mut i = base(&p);
        i.child_alive = false;
        i.exit_code = Some(1);
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::Broken);
        i.killed = true;
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::StoppedByYou);
    }

    #[test]
    fn silence_is_not_responding() {
        let mut i = base(&[]);
        i.since_last_message_s = Some(THRESHOLDS.silence_s + 1.0);
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::NotResponding);
        let mut j = base(&[]);
        j.since_last_message_s = None;
        j.since_start_s = THRESHOLDS.silence_s + 1.0;
        assert_eq!(judge(&j, &THRESHOLDS), Verdict::NotResponding);
    }

    #[test]
    fn edge_series_never_panic_and_never_start_green() {
        let mut i = base(&[]);
        i.since_last_message_s = None;
        i.since_start_s = 5.0;
        assert_eq!(judge(&i, &THRESHOLDS), Verdict::Starting);
        let one = [Point {
            step: 1,
            loss: 2.0,
            samples_per_s: 10.0,
        }];
        assert_eq!(judge(&base(&one), &THRESHOLDS), Verdict::GoingWell);
        let hundred = falling(100);
        let mut none_total = base(&hundred);
        none_total.total_steps = None;
        let _ = judge(&none_total, &THRESHOLDS); // must not panic
    }

    #[test]
    fn a_rate_drop_after_warmup_is_slow() {
        let mut p = falling(60);
        p.last_mut().unwrap().samples_per_s = 5.0; // median 40, 5 < 40/3
        assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Slow);
        let mut early = falling(5);
        early.last_mut().unwrap().samples_per_s = 5.0; // before warmup: not judged
        assert_eq!(judge(&base(&early), &THRESHOLDS), Verdict::GoingWell);
    }

    #[test]
    fn no_new_minimum_over_a_quarter_of_the_run_is_stopped_learning() {
        let mut p = falling(30); // steps 100..3000, minimum at step 3000
        let last = p.last().unwrap().loss;
        for k in 1..=60 {
            // 6000 more steps (> 0.25 * 20_000) without going below `last`
            p.push(Point {
                step: 3000 + k * 100,
                loss: last * 1.05,
                samples_per_s: 40.0,
            });
        }
        assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::StoppedLearning);
    }
}
