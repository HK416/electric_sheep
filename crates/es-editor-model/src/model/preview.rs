//! The short test of each checkpoint, as ③ shows it while the run trains (packet M13/Z4, over
//! Z1).
//!
//! `es loop cycle` with `[eval.preview]` tests every checkpoint as soon as its bundle is on disk:
//! the test's own run goes to `<run>/preview/<step>/` (`episodes.json`, `frames/`, `traj/`), a
//! row per finished test to `<run>/preview/index.jsonl`, and stream 1 says `preview.begin{step,
//! dir}` and `preview.end{step, dir, successes, episodes, code}`. The file is what a re-opened
//! run shows; the events are what a watched run adds to it. A preview judges nothing (spec
//! 13.3): these are only what the person watches.

use std::collections::BTreeMap;
use std::path::{Component, Path};
use std::str::FromStr;

use es_eval::episodes::read_episodes;
use serde::Deserialize;

use crate::model::i18n::{fill, Lang};
use crate::model::results::first_to_play;
use crate::model::telemetry_view::Event;

/// Where `es loop cycle` appends a row per finished preview, relative to the run.
pub const INDEX: &str = "preview/index.jsonl";

/// One checkpoint's test. Deserialised from an `index.jsonl` row, whose other fields (`bundle`,
/// `suite`, `evaluation_hash`, `created`) the screen does not need.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Preview {
    /// The checkpoint's optimizer step.
    pub step: u32,
    /// The test's run, relative to the run: `preview/<step>`.
    pub dir: String,
    pub successes: u32,
    pub episodes: u32,
    /// The `es eval run` child's exit code; `None` while it runs.
    pub code: Option<i32>,
    /// The attempt to play first: the first failure, else the first attempt; `None` until its
    /// `episodes.json` is there.
    #[serde(skip)]
    pub first: Option<String>,
}

fn field<T: FromStr>(e: &Event, key: &str) -> Option<T> {
    e.fields.get(key)?.parse().ok()
}

/// What stream 1 has said, one preview per step, ascending: a `preview.begin` is a test that
/// runs, a `preview.end` one that finished, and the later word about a step wins.
pub fn heard(events: &[Event]) -> Vec<Preview> {
    let mut by_step = BTreeMap::new();
    for e in events {
        let running = match e.kind.as_str() {
            "preview.begin" => true,
            "preview.end" => false,
            _ => continue,
        };
        let (Some(step), Some(dir)) = (field(e, "step"), e.fields.get("dir")) else {
            continue;
        };
        let preview = Preview {
            step,
            dir: dir.clone(),
            successes: field(e, "successes").unwrap_or(0),
            episodes: field(e, "episodes").unwrap_or(0),
            // An end with no readable code still ended.
            code: (!running).then(|| field(e, "code").unwrap_or(-1)),
            first: None,
        };
        by_step.insert(step, preview);
    }
    by_step.into_values().collect()
}

/// `<run>/preview/index.jsonl`, a row per finished test. A line that does not parse is skipped:
/// the file is appended to while the run goes on.
pub fn on_disk(run: &Path) -> Vec<Preview> {
    let Ok(text) = std::fs::read_to_string(run.join(INDEX)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// A `dir` that stays inside the run: neither the wire nor the file is trusted to name
/// anything else.
fn inside(dir: &str) -> bool {
    !dir.is_empty()
        && Path::new(dir)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// Every preview of the run at `run`, newest step first: the file's rows, then what was
/// `heard`, the later word about a step winning; each finished one with its attempt to play
/// first.
pub fn previews(run: &Path, heard: &[Preview]) -> Vec<Preview> {
    let mut by_step = BTreeMap::new();
    for p in on_disk(run).into_iter().chain(heard.iter().cloned()) {
        if inside(&p.dir) {
            by_step.insert(p.step, p);
        }
    }
    by_step
        .into_values()
        .rev()
        .map(|mut p| {
            if p.code.is_some() {
                let rows = read_episodes(&run.join(&p.dir)).ok().flatten();
                p.first = first_to_play(rows.as_deref(), &[]);
            }
            p
        })
        .collect()
}

/// The preview ③ shows (packet M13/Z5a): the one picked while it is there, else the newest
/// that finished, else the newest still running - its headline alone. `previews` is newest
/// first, as [`previews`] hands them back.
pub fn shown(previews: &[Preview], chosen: Option<u32>) -> Option<&Preview> {
    (previews.iter().find(|p| Some(p.step) == chosen))
        .or_else(|| previews.iter().find(|p| p.code.is_some()))
        .or(previews.first())
}

/// A preview's one line: still running, run to the end (how many attempts succeeded), or its
/// child failed - said so, with no numbers, since a half-run test counted nothing.
pub fn headline(lang: Lang, p: &Preview) -> String {
    let step = p.step.to_string();
    match p.code {
        None => fill(lang, "watch.preview.running", &[&step]),
        Some(0) => fill(
            lang,
            "watch.preview.done",
            &[&p.successes.to_string(), &p.episodes.to_string(), &step],
        ),
        Some(_) => fill(lang, "watch.preview.failed", &[&step]),
    }
}

/// How far from a preview's mark on the loss curve, in points, a click still picks it.
pub const REACH: f32 = 8.0;

/// The preview whose mark is nearest a click on the curve, when it is within [`REACH`]:
/// `marks` are `(step, x)` and `x` the click, both across the curve in `0..=1`, and `width` the
/// curve's width in points.
pub fn picked(marks: &[(u32, f32)], x: f32, width: f32) -> Option<u32> {
    marks
        .iter()
        .map(|&(step, at)| (step, (at - x).abs() * width))
        .filter(|(_, off)| *off <= REACH)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(step, _)| step)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use es_eval::episodes::{write_episodes, EpisodeRow};

    pub(crate) fn event(kind: &str, fields: &[(&str, &str)]) -> Event {
        Event {
            tick: 0,
            kind: kind.to_owned(),
            fields: fields
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
        }
    }

    /// `preview.end` as `es loop cycle`'s preview thread publishes it (`Voice::preview_end`).
    pub(crate) fn end(step: &str, successes: &str, code: &str) -> Event {
        event(
            "preview.end",
            &[
                ("step", step),
                ("dir", &format!("preview/{step}")),
                ("successes", successes),
                ("episodes", "4"),
                ("code", code),
                ("stage", "preview"),
            ],
        )
    }

    pub(crate) fn begin(step: &str) -> Event {
        event(
            "preview.begin",
            &[
                ("step", step),
                ("dir", &format!("preview/{step}")),
                ("stage", "preview"),
            ],
        )
    }

    /// An `index.jsonl` row as `es loop cycle` writes it (`cmd/cycle.rs`, `preview`).
    pub(crate) fn index_row(step: u32, successes: u32, code: i32) -> String {
        serde_json::json!({
            "step": step,
            "dir": format!("preview/{step}"),
            "bundle": format!("train/checkpoints/{step}.esb"),
            "suite": "nominal",
            "evaluation_hash": "ab".repeat(32),
            "successes": successes,
            "episodes": 4,
            "code": code,
            "created": 1_790_000_000u64,
        })
        .to_string()
    }

    fn row(ep: u64, termination: &str) -> EpisodeRow {
        EpisodeRow {
            suite: "nominal".into(),
            cell: format!("nominal-{ep:02}"),
            episode: ep,
            seed: 101 + ep,
            termination: termination.into(),
            steps: 10,
            changed_steps: 0,
            histogram: [(termination.to_owned(), 1)].into(),
        }
    }

    fn done(step: u32, successes: u32, code: i32, first: Option<&str>) -> Preview {
        Preview {
            step,
            dir: format!("preview/{step}"),
            successes,
            episodes: 4,
            code: Some(code),
            first: first.map(str::to_owned),
        }
    }

    fn running(step: u32) -> Preview {
        Preview {
            successes: 0,
            episodes: 0,
            code: None,
            ..done(step, 0, 0, None)
        }
    }

    /// A fixture event stream: begins and ends in the order the preview thread sends them,
    /// between the cycle's other events; a malformed one is skipped, and a step begun again
    /// runs again.
    #[test]
    fn previews_from_a_fixture_event_stream() {
        let events = [
            event("stage.begin", &[("stage", "train")]),
            begin("1000"),
            event("checkpoint", &[("step", "5000")]),
            end("1000", "1", "0"),
            begin("5000"),
            event("preview.begin", &[("dir", "preview/7")]),
        ];
        assert_eq!(heard(&events), [done(1000, 1, 0, None), running(5000)]);
        let again = [end("1000", "1", "0"), begin("1000")];
        assert_eq!(heard(&again), [running(1000)]);
        assert!(heard(&[]).is_empty());
    }

    /// A fixture directory: what a re-opened run shows from disk alone, then with what was
    /// heard since - newest first, the later word winning, and each finished preview's first
    /// failure to play.
    #[test]
    fn previews_from_a_fixture_directory() {
        let run = std::env::temp_dir().join(format!("es-z4-preview-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&run);
        std::fs::create_dir_all(run.join("preview/1000")).unwrap();
        std::fs::create_dir_all(run.join("preview/5000")).unwrap();
        write_episodes(
            &[
                row(0, "success"),
                row(1, "timeout"),
                row(2, "success"),
                row(3, "timeout"),
            ],
            &run.join("preview/1000"),
        )
        .unwrap();
        write_episodes(&[row(0, "success")], &run.join("preview/5000")).unwrap();
        let index = [
            index_row(1000, 2, 0),
            "{\"step\": 5000, \"di".to_owned(),
            index_row(5000, 0, 1),
        ];
        std::fs::write(run.join(INDEX), index.join("\n") + "\n").unwrap();

        assert_eq!(
            previews(&run, &[]),
            [
                done(5000, 0, 1, Some("nominal-00")),
                done(1000, 2, 0, Some("nominal-01"))
            ],
            "a re-opened run: the file alone, the half-written line skipped"
        );
        let heard = heard(&[end("5000", "1", "0"), begin("20000")]);
        assert_eq!(
            previews(&run, &heard),
            [
                running(20000),
                done(5000, 1, 0, Some("nominal-00")),
                done(1000, 2, 0, Some("nominal-01"))
            ]
        );
        // A dir that would leave the run is not followed, from the wire or the file.
        let escape = Preview {
            dir: "../elsewhere".into(),
            ..done(30000, 4, 0, None)
        };
        assert_eq!(previews(&run, &[escape]).len(), 2);
        assert!(previews(&run.join("absent"), &[]).is_empty());
        std::fs::remove_dir_all(&run).ok();
    }

    /// Which preview ③ shows: the chosen step while it is listed; otherwise the newest that
    /// finished, a failed one included; with none finished, the newest running.
    #[test]
    fn the_shown_preview_is_the_chosen_else_the_newest_finished() {
        let list = [
            running(20000),
            done(5000, 1, 0, None),
            done(1000, 3, 0, None),
        ];
        let step = |chosen| shown(&list, chosen).map(|p| p.step);
        assert_eq!(step(Some(1000)), Some(1000));
        assert_eq!(step(Some(20000)), Some(20000), "a running one, when picked");
        assert_eq!(step(None), Some(5000));
        assert_eq!(step(Some(7)), Some(5000), "a step no longer listed");
        let failed = [running(20000), done(5000, 0, 1, None)];
        assert_eq!(shown(&failed, None).map(|p| p.step), Some(5000));
        let none_finished = [running(20000), running(5000)];
        assert_eq!(shown(&none_finished, None).map(|p| p.step), Some(20000));
        assert_eq!(shown(&[], Some(1000)), None);
    }

    /// The headline names the step every time, and the numbers only of a test that ran to the
    /// end.
    #[test]
    fn a_headline_says_running_done_or_failed() {
        for lang in Lang::ALL {
            let finished = headline(lang, &done(1000, 1, 0, None));
            assert!(
                finished.contains("1000") && finished.contains('4') && !finished.contains("{}"),
                "{finished}"
            );
            let going = headline(lang, &running(5000));
            assert!(going.contains("5000") && !going.contains("{}"), "{going}");
            let failed = headline(lang, &done(5000, 3, 2, None));
            assert!(
                failed.contains("5000") && !failed.contains('3') && !failed.contains('4'),
                "no numbers from a test that did not finish: {failed}"
            );
            assert_ne!(going, failed);
        }
        assert_eq!(
            headline(Lang::En, &done(1000, 1, 0, None)),
            "Check: 1 of 4 succeeded — at step 1000"
        );
    }

    /// A click picks the nearest mark within reach, in points: two marks 0.1 apart on a
    /// 200-point curve are 20 points apart.
    #[test]
    fn a_click_picks_the_nearest_mark_within_reach() {
        let marks = [(1000, 0.2), (5000, 0.3)];
        assert_eq!(picked(&marks, 0.2, 200.0), Some(1000), "on the mark");
        assert_eq!(picked(&marks, 0.23, 200.0), Some(1000), "6 points off");
        assert_eq!(picked(&marks, 0.27, 200.0), Some(5000), "nearer the other");
        assert_eq!(picked(&marks, 0.25, 200.0), None, "10 points from both");
        assert_eq!(
            picked(&marks, 0.24, 100.0),
            Some(1000),
            "4 points off on a narrower curve"
        );
        assert_eq!(picked(&[], 0.2, 200.0), None);
    }
}
