//! Plain words for the things the editor used to name the way the code names them
//! (packet M7/E6, spec 23.2).
//!
//! A results table headed `envelope_violation_rate` and a launch field labelled `--config`
//! are readable to whoever wrote them and to nobody else. Every one of those names has a
//! plain word here and keeps its raw spelling one hover away - the hover is where the
//! technical identifier belongs, because that is what someone reaches for once they already
//! know what the column means.
//!
//! **Every table is total and has no wildcard arm.** A metric added to
//! [`MetricSpec`](es_ir::evaluation::MetricSpec), a flag added to [`LaunchField`] or a tab
//! added to [`Tab`] must break this crate's *build*; the alternative - a `_ =>` returning the
//! raw name - is a column nobody ever gets around to naming. `metric_and_launch_labels_are_total`
//! judges the same thing from the other side, over `MetricSpec::ALL`.
//!
//! The home screen's five steps are here too, for the same reason `app.rs` decides nothing
//! else (spec 28.10 rule 3): their order is spec 13.1's loop, not a layout.

use es_ir::evaluation::MetricSpec;

use crate::model::i18n::{t, Lang};
use crate::model::launch::{Kind, LaunchField, LaunchFlag};

// --- tabs --------------------------------------------------------------------------------

/// The tabs, named for what a person does in them rather than for the view-model behind
/// them. The old name and the spec section live in [`Tab::hint_key`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tab {
    #[default]
    Design,
    Results,
    Live,
    Sees,
    Problems,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Design,
        Tab::Results,
        Tab::Live,
        Tab::Sees,
        Tab::Problems,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Tab::Design => "tab.design",
            Tab::Results => "tab.results",
            Tab::Live => "tab.live",
            Tab::Sees => "tab.sees",
            Tab::Problems => "tab.problems",
        }
    }

    /// The hover: the name the tab had before this packet, and the spec section it serves.
    pub fn hint_key(self) -> &'static str {
        match self {
            Tab::Design => "tab.design.hint",
            Tab::Results => "tab.results.hint",
            Tab::Live => "tab.live.hint",
            Tab::Sees => "tab.sees.hint",
            Tab::Problems => "tab.problems.hint",
        }
    }
}

// --- the home screen ----------------------------------------------------------------------

/// The five steps of spec 13.1's loop, in the order they happen.
///
/// The home screen is a strip of these: the word, one sentence of what it means, and a button
/// to the tab that does it. The order is the loop's, which is why it is a `const` here and
/// not a `for` loop in `app.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Design,
    Collect,
    Train,
    Evaluate,
    Watch,
}

impl Step {
    pub const ALL: [Step; 5] = [
        Step::Design,
        Step::Collect,
        Step::Train,
        Step::Evaluate,
        Step::Watch,
    ];

    /// The workflow's own word for the step.
    pub fn word_key(self) -> &'static str {
        match self {
            Step::Design => "word.design",
            Step::Collect => "word.collect",
            Step::Train => "word.train",
            Step::Evaluate => "word.evaluate",
            Step::Watch => "word.watch",
        }
    }

    /// One sentence saying what the step is for, in words that assume no domain knowledge.
    pub fn sentence_key(self) -> &'static str {
        match self {
            Step::Design => "home.step.design",
            Step::Collect => "home.step.collect",
            Step::Train => "home.step.train",
            Step::Evaluate => "home.step.evaluate",
            Step::Watch => "home.step.watch",
        }
    }

    /// Where the step's button goes. Collecting, training and evaluating all start from the
    /// Results tab, because that is where the Start panel is; watching is the Live tab.
    pub fn tab(self) -> Tab {
        match self {
            Step::Design => Tab::Design,
            Step::Collect | Step::Train | Step::Evaluate => Tab::Results,
            Step::Watch => Tab::Live,
        }
    }
}

// --- the stages of a cycle -----------------------------------------------------------------

/// The plain word for one stage of `es loop cycle`, as the wire names it (packet M7/E7).
///
/// Three of the five are already the workflow's own words, so they share [`Step`]'s keys
/// rather than getting a second set that could drift from them. A name this build has no word
/// for renders as the producer's own, which is better than an empty chip.
pub fn stage_label(lang: Lang, name: &str) -> &'static str {
    match name {
        "collect" => t(lang, "word.collect"),
        "expert-gate" => t(lang, "word.expert_gate"),
        "train" => t(lang, "word.train"),
        "eval" => t(lang, "word.evaluate"),
        "showcase" => t(lang, "word.showcase"),
        _ => "",
    }
}

// --- metrics -----------------------------------------------------------------------------

/// The key of a metric's plain name. Total over `MetricSpec::ALL` and deliberately without a
/// wildcard arm.
pub fn metric_key(metric: MetricSpec) -> &'static str {
    match metric {
        MetricSpec::SuccessRate => "metric.success_rate",
        MetricSpec::InterventionRate => "metric.intervention_rate",
        MetricSpec::CollisionRate => "metric.collision_rate",
        MetricSpec::EnvelopeViolationRate => "metric.envelope_violation_rate",
        MetricSpec::ActionSmoothness => "metric.action_smoothness",
        MetricSpec::EpisodeLength => "metric.episode_length",
        MetricSpec::FailureModeHistogram => "metric.failure_mode_histogram",
        MetricSpec::DomainGap => "metric.domain_gap",
        MetricSpec::ChunkUnderrunRate => "metric.chunk_underrun_rate",
        MetricSpec::EndToEndLatencyP50 => "metric.end_to_end_latency_p50",
        MetricSpec::EndToEndLatencyP95 => "metric.end_to_end_latency_p95",
        MetricSpec::PhysicsStepsPerSec => "metric.physics_steps_per_sec",
        MetricSpec::CameraFramesPerSec => "metric.camera_frames_per_sec",
        MetricSpec::PixelsPerSec => "metric.pixels_per_sec",
        MetricSpec::ObservationGbPerSec => "metric.observation_gb_per_sec",
        MetricSpec::PolicyInferencesPerSec => "metric.policy_inferences_per_sec",
        MetricSpec::ActionsPerSec => "metric.actions_per_sec",
        MetricSpec::GpuMemoryPeak => "metric.gpu_memory_peak",
    }
}

/// A metric's plain name.
pub fn metric_label(lang: Lang, metric: MetricSpec) -> &'static str {
    t(lang, metric_key(metric))
}

/// The metric a report column or a telemetry row is named after.
///
/// A run's `report.json` and the live stream both carry the metric as the string
/// `MetricSpec::name()` prints, so the column header is matched back to the enum rather than
/// re-humanised from the string. The two latency rows are the exception:
/// `TelemetryModel::metric_rows` spells them the way `PerfMetrics`' fields are spelt, and
/// both spellings mean one metric.
pub fn metric_by_name(name: &str) -> Option<MetricSpec> {
    match name {
        "p50_end_to_end_latency" => return Some(MetricSpec::EndToEndLatencyP50),
        "p95_end_to_end_latency" => return Some(MetricSpec::EndToEndLatencyP95),
        _ => {}
    }
    MetricSpec::ALL.into_iter().find(|m| m.name() == name)
}

/// A results-table column header: one of the three fixed columns, a metric's plain name, or -
/// for a column this build has no word for - the raw name, which is better than nothing.
pub fn column_label(lang: Lang, column: &str) -> &'static str {
    match column {
        "cell" => t(lang, "column.cell"),
        "suite" => t(lang, "column.suite"),
        "seed" => t(lang, "column.seed"),
        _ => metric_by_name(column).map_or("", |m| metric_label(lang, m)),
    }
}

// --- the Start panel ------------------------------------------------------------------------

/// What a launch field is called in plain words. Total over [`LaunchField`].
pub fn launch_key(field: LaunchField) -> &'static str {
    match field {
        LaunchField::Config => "launch.config",
        LaunchField::Policy => "launch.policy",
        LaunchField::Scene => "launch.scene",
        LaunchField::Out => "launch.out",
        LaunchField::Frames => "launch.frames",
        LaunchField::Jobs => "launch.jobs",
        LaunchField::Telemetry => "launch.telemetry",
        LaunchField::TelemetryToken => "launch.telemetry_token",
        LaunchField::TelemetryImageEvery => "launch.telemetry_image_every",
        LaunchField::Recipe => "launch.recipe",
        LaunchField::From => "launch.from",
    }
}

/// As [`launch_key`], for the flags that take no value. Total over [`LaunchFlag`].
pub fn launch_flag_key(flag: LaunchFlag) -> &'static str {
    match flag {
        LaunchFlag::DryRun => "launch.dry_run",
        LaunchFlag::AllowNewEvaluation => "launch.allow_new_evaluation",
        LaunchFlag::SkipExpertGate => "launch.skip_expert_gate",
    }
}

/// What the command selector says. The command itself (`es eval run`) stays in the hover.
pub fn kind_key(kind: Kind) -> &'static str {
    match kind {
        Kind::Eval => "launch.kind.eval",
        Kind::Train => "launch.kind.train",
        Kind::Cycle => "launch.kind.cycle",
    }
}

pub fn launch_label(lang: Lang, field: LaunchField) -> &'static str {
    t(lang, launch_key(field))
}

pub fn launch_flag_label(lang: Lang, flag: LaunchFlag) -> &'static str {
    t(lang, launch_flag_key(flag))
}

pub fn kind_label(lang: Lang, kind: Kind) -> &'static str {
    t(lang, kind_key(kind))
}

/// Which field a **Browse** button beside it should open a *folder* chooser for, rather than
/// a file chooser. `--out` and `--frames` name directories the run writes into; the rest name
/// files that already exist, and the two flags that name neither get no button at all.
pub fn browses(field: LaunchField) -> Option<Browse> {
    match field {
        LaunchField::Config | LaunchField::Recipe => Some(Browse::Toml),
        LaunchField::Policy => Some(Browse::Policy),
        LaunchField::Scene => Some(Browse::Scene),
        LaunchField::Out | LaunchField::Frames => Some(Browse::Folder),
        LaunchField::Jobs
        | LaunchField::Telemetry
        | LaunchField::TelemetryToken
        | LaunchField::TelemetryImageEvery
        | LaunchField::From => None,
    }
}

/// What a **Browse** button offers. The extension lists are the ones the CLI documents, so
/// the dialog cannot suggest a file `es` would refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Browse {
    /// A `.esb` bundle, which is also what the top bar's Open opens.
    Policy,
    /// One of the per-IR or settings documents.
    Toml,
    /// An MJCF or URDF scene.
    Scene,
    Folder,
}

impl Browse {
    /// What the button beside the field says. A folder chooser and a file chooser are
    /// different enough that a person should be told which one is about to open.
    pub fn label_key(self) -> &'static str {
        match self {
            Browse::Folder => "open.browse_folder",
            Browse::Policy | Browse::Toml | Browse::Scene => "open.browse",
        }
    }

    /// `(what the filter is called, the extensions it allows)`. The name is a raw one on
    /// purpose: a file dialog's filter row is where extensions belong.
    pub fn filter(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Browse::Policy => ("Policy bundle (*.esb)", &["esb"]),
            Browse::Toml => ("Settings (*.toml)", &["toml"]),
            Browse::Scene => ("Scene (*.xml, *.urdf)", &["xml", "urdf"]),
            Browse::Folder => ("", &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        browses, column_label, kind_label, launch_flag_label, launch_label, metric_by_name,
        metric_label, Browse, Step, Tab,
    };

    use std::collections::BTreeSet;

    use es_ir::evaluation::MetricSpec;

    use crate::model::i18n::{t, Lang};
    use crate::model::launch::{Kind, LaunchField, LaunchFlag};

    /// Oracle 2 (packet M7/E6). Every metric, every launch field and every flag has a plain
    /// name in both languages, no two of them share one, and none of them fell back to its key.
    #[test]
    fn metric_and_launch_labels_are_total() {
        assert_eq!(MetricSpec::ALL.len(), 18, "the whole set is covered");
        for lang in Lang::ALL {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for metric in MetricSpec::ALL {
                let label = metric_label(lang, metric);
                assert!(!label.is_empty(), "{metric:?} has a name");
                assert_ne!(
                    label,
                    super::metric_key(metric),
                    "{metric:?} is translated, not left as its key"
                );
                assert!(
                    !label.contains('_'),
                    "{metric:?} still reads as an identifier: {label}"
                );
                assert!(seen.insert(label), "{label} is used twice in {lang:?}");
                // The raw name is what the hover shows, and it is never the label.
                assert_ne!(label, metric.name(), "{metric:?}");
                assert_eq!(metric_by_name(metric.name()), Some(metric), "round-trips");
            }

            for field in LaunchField::ALL {
                let label = launch_label(lang, field);
                assert!(!label.is_empty() && !label.starts_with("--"), "{field:?}");
                assert!(seen.insert(label), "{label} is used twice in {lang:?}");
            }
            for flag in LaunchFlag::ALL {
                let label = launch_flag_label(lang, flag);
                assert!(!label.is_empty() && !label.starts_with("--"), "{flag:?}");
                assert!(seen.insert(label), "{label} is used twice in {lang:?}");
            }
            for kind in Kind::ALL {
                let label = kind_label(lang, kind);
                assert!(!label.contains("es "), "{kind:?} shows a command: {label}");
                assert!(seen.insert(label), "{label} is used twice in {lang:?}");
            }
            for tab in Tab::ALL {
                assert!(seen.insert(t(lang, tab.key())), "two tabs share a name");
                assert!(!t(lang, tab.hint_key()).is_empty(), "{tab:?} has a hover");
            }

            // The three fixed columns and the metric columns all humanise.
            for column in ["cell", "suite", "seed"] {
                assert!(!column_label(lang, column).is_empty(), "{column}");
            }
            assert_eq!(
                column_label(lang, MetricSpec::SuccessRate.name()),
                metric_label(lang, MetricSpec::SuccessRate)
            );
            assert!(column_label(lang, "invented_by_a_future_run").is_empty());
        }

        // The telemetry tab spells the two latency rows its own way; both reach one metric.
        assert_eq!(
            metric_by_name("p50_end_to_end_latency"),
            Some(MetricSpec::EndToEndLatencyP50)
        );
        assert_eq!(
            metric_by_name("p95_end_to_end_latency"),
            Some(MetricSpec::EndToEndLatencyP95)
        );
        assert_eq!(metric_by_name("step_per_sec"), None);

        // A cycle's five stages have a plain word each, and an unknown one says nothing
        // rather than guessing (packet M7/E7).
        for lang in Lang::ALL {
            for stage in ["collect", "expert-gate", "train", "eval", "showcase"] {
                assert!(
                    !super::stage_label(lang, stage).is_empty(),
                    "{stage} in {lang:?}"
                );
            }
            assert!(super::stage_label(lang, "invented-by-a-future-run").is_empty());
        }

        // A folder-valued flag offers a folder chooser and a file-valued one a file chooser.
        assert_eq!(browses(LaunchField::Out), Some(Browse::Folder));
        assert_eq!(browses(LaunchField::Policy), Some(Browse::Policy));
        assert_eq!(browses(LaunchField::Jobs), None, "a number is typed");
        for field in LaunchField::ALL {
            let Some(browse) = browses(field) else {
                continue;
            };
            let (name, extensions) = browse.filter();
            assert_eq!(
                name.is_empty(),
                extensions.is_empty(),
                "{field:?}: a filter has both or neither"
            );
        }
    }

    /// Oracle 5 (packet M7/E6). The home screen's strip is spec 13.1's loop, in order, and
    /// every step says what it is for in both languages.
    #[test]
    fn home_screen_lists_the_five_steps_in_loop_order() {
        assert_eq!(
            Step::ALL,
            [
                Step::Design,
                Step::Collect,
                Step::Train,
                Step::Evaluate,
                Step::Watch
            ],
            "collect -> train -> evaluate -> watch, after describing the task"
        );
        for lang in Lang::ALL {
            let mut words = BTreeSet::new();
            let mut sentences = BTreeSet::new();
            for step in Step::ALL {
                let word = t(lang, step.word_key());
                let sentence = t(lang, step.sentence_key());
                assert!(!word.is_empty(), "{step:?} has a word in {lang:?}");
                assert!(
                    sentence.len() > 20 && sentence.ends_with('.'),
                    "{step:?} in {lang:?} is one sentence: {sentence}"
                );
                assert!(!sentence.contains("spec "), "{step:?} cites the spec");
                assert!(words.insert(word), "{word} is used twice");
                assert!(sentences.insert(sentence), "two steps say the same thing");
            }
        }
        // Every button lands somewhere a person can act, and the first step is the tab the
        // home screen itself is on.
        assert_eq!(Step::Design.tab(), Tab::Design);
        assert_eq!(Step::Watch.tab(), Tab::Live);
        for step in Step::ALL {
            assert!(Tab::ALL.contains(&step.tab()), "{step:?} goes somewhere");
        }
    }
}
