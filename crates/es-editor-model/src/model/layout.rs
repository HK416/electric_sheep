//! The dock under the step bar (`docs/design/editor-redesign.md` section 6.1, packet M12/Y10):
//! which panes exist, where each step puts them, what the person's own arrangement was, and
//! how the step bar reads a step's state. `ui/shell.rs` draws all of it and decides none of it.
//!
//! Every pane is in every arrangement it belongs to exactly once, and none can be closed: a
//! pane dragged somewhere odd is still there, and View > Reset the layout puts it back.

use std::collections::BTreeMap;

use egui_dock::{DockState, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::model::i18n::{self, Lang};
use crate::model::recent::LAYOUT_KEY;
use crate::model::workflow::{Phase, PhaseState};

/// One dock tab. The first four are the shell's own; the five Advanced ones are the tabs the
/// editor had before the dock, each drawn unchanged as one pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Pane {
    Viewport,
    StepPanel,
    Summary,
    Console,
    AdvancedGraph,
    AdvancedSees,
    AdvancedProblems,
    AdvancedMetrics,
    AdvancedLive,
}

use Pane::{
    AdvancedGraph, AdvancedLive, AdvancedMetrics, AdvancedProblems, AdvancedSees, Console,
    StepPanel, Summary, Viewport,
};

/// The Advanced panes, in the order the old tab bar had them.
const ADVANCED: [Pane; 5] = [
    AdvancedGraph,
    AdvancedMetrics,
    AdvancedLive,
    AdvancedSees,
    AdvancedProblems,
];

impl Pane {
    pub const ALL: [Pane; 9] = [
        Viewport,
        StepPanel,
        Summary,
        Console,
        AdvancedGraph,
        AdvancedSees,
        AdvancedProblems,
        AdvancedMetrics,
        AdvancedLive,
    ];

    pub fn key(self) -> &'static str {
        match self {
            Viewport => "shell.viewport",
            StepPanel => "shell.step_panel",
            Summary => "shell.summary",
            Console => "shell.console",
            AdvancedGraph => "advanced.graph",
            AdvancedSees => "advanced.sees",
            AdvancedProblems => "advanced.problems",
            AdvancedMetrics => "advanced.metrics",
            AdvancedLive => "advanced.live",
        }
    }

    /// The hover of an Advanced pane: the tab it used to be, and its spec section.
    pub fn hint_key(self) -> Option<&'static str> {
        match self {
            Viewport | StepPanel | Summary | Console => None,
            AdvancedGraph => Some("tab.design.hint"),
            AdvancedSees => Some("tab.sees.hint"),
            AdvancedProblems => Some("tab.problems.hint"),
            AdvancedMetrics => Some("tab.results.hint"),
            AdvancedLive => Some("tab.live.hint"),
        }
    }

    /// Whether the dock scrolls the pane up and down. The rest size themselves to the pane: the
    /// viewport's canvas, players and tiles, the log's own stick-to-bottom list, the graph's
    /// panning canvas, and the run table's panels, each of which scrolls inside.
    pub fn scrolls(self) -> bool {
        match self {
            StepPanel | Summary | AdvancedSees | AdvancedProblems | AdvancedLive => true,
            Viewport | Console | AdvancedGraph | AdvancedMetrics => false,
        }
    }
}

/// The panes of one arrangement, by where they sit. The first pane of a group is its open tab.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub left: Vec<Pane>,
    pub centre: Vec<Pane>,
    pub right: Vec<Pane>,
    pub bottom: Vec<Pane>,
}

/// The default arrangement for a phase: the step panel left, the viewport in the centre, the
/// summary right, and along the bottom the log and the Advanced panes - the one the step leans
/// on most first among those.
pub fn default_layout(phase: Phase) -> Layout {
    let first = match phase {
        Phase::Scene | Phase::Teach => AdvancedGraph,
        Phase::Train | Phase::Evaluate => AdvancedLive,
        Phase::Results => AdvancedMetrics,
    };
    let mut bottom = vec![Console, first];
    bottom.extend(ADVANCED.into_iter().filter(|p| *p != first));
    Layout {
        left: vec![StepPanel],
        centre: vec![Viewport],
        right: vec![Summary],
        bottom,
    }
}

/// What a path that is not a project opens into: the Advanced panes fill the centre, in the
/// old tab bar's order, with the log below. A step's panes mean nothing without a project.
pub fn advanced_layout() -> Layout {
    Layout {
        left: Vec::new(),
        centre: ADVANCED.to_vec(),
        right: Vec::new(),
        bottom: vec![Console],
    }
}

impl Layout {
    /// Every pane, sorted.
    pub fn panes(&self) -> Vec<Pane> {
        let mut panes: Vec<Pane> = [&self.left, &self.centre, &self.right, &self.bottom]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        panes.sort();
        panes
    }

    /// As a dock: the bottom spans the width, the left and right flank the centre, each side a
    /// fifth of the width. The bottom takes 30% when it holds panes to work in and 15% when it
    /// is only the log. (`egui_dock`'s fraction is the left or upper child's share.)
    pub fn dock(&self) -> DockState<Pane> {
        let mut dock = DockState::new(self.centre.clone());
        let tree = dock.main_surface_mut();
        let mut centre = NodeIndex::root();
        if !self.bottom.is_empty() {
            let upper = if self.bottom == [Console] { 0.85 } else { 0.7 };
            [centre, _] = tree.split_below(centre, upper, self.bottom.clone());
        }
        if !self.left.is_empty() {
            [centre, _] = tree.split_left(centre, 0.2, self.left.clone());
        }
        if !self.right.is_empty() {
            tree.split_right(centre, 0.75, self.right.clone());
        }
        dock
    }
}

/// Which arrangement the dock shows: one per step of a project, one for anything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrangement {
    Step(Phase),
    Advanced,
}

impl Arrangement {
    /// The open project's current step, or the Advanced arrangement when none is open.
    pub fn of(phase: Option<Phase>) -> Self {
        phase.map_or(Self::Advanced, Self::Step)
    }

    pub fn layout(self) -> Layout {
        match self {
            Self::Step(phase) => default_layout(phase),
            Self::Advanced => advanced_layout(),
        }
    }

    /// What the arrangement is stored under.
    fn name(self) -> &'static str {
        match self {
            Self::Step(phase) => phase.key(),
            Self::Advanced => "advanced",
        }
    }

    fn named(name: &str) -> Option<Self> {
        Phase::ALL
            .map(Self::Step)
            .into_iter()
            .chain([Self::Advanced])
            .find(|a| a.name() == name)
    }
}

/// The person's arrangements, persisted under [`LAYOUT_KEY`]. An arrangement nobody has
/// touched is not stored: it is its default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Docks(BTreeMap<String, DockState<Pane>>);

impl Docks {
    /// What the last session left, as eframe stores it. Absent or unreadable is empty, which
    /// is every default; one arrangement that does not hold exactly its own panes - stored by
    /// a version with another set - starts over rather than losing a pane for good.
    pub fn load(storage: Option<&dyn eframe::Storage>) -> Self {
        let mut docks: Self = storage
            .and_then(|s| eframe::get_value(s, LAYOUT_KEY))
            .unwrap_or_default();
        docks.0.retain(|name, dock| {
            Arrangement::named(name).is_some_and(|a| {
                let mut held: Vec<Pane> = dock.iter_all_tabs().map(|(_, p)| *p).collect();
                held.sort();
                held == a.layout().panes()
            })
        });
        docks
    }

    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, LAYOUT_KEY, self);
    }

    pub fn get(&mut self, arrangement: Arrangement) -> &mut DockState<Pane> {
        self.0
            .entry(arrangement.name().to_owned())
            .or_insert_with(|| arrangement.layout().dock())
    }

    pub fn reset(&mut self, arrangement: Arrangement) {
        self.0
            .insert(arrangement.name().to_owned(), arrangement.layout().dock());
    }

    /// Brings `pane` to the front of its group, wherever the person put it.
    pub fn focus(&mut self, arrangement: Arrangement, pane: Pane) {
        let dock = self.get(arrangement);
        if let Some((surface, node, tab)) = dock.find_tab(&pane) {
            dock.set_active_tab((surface, node, tab));
            dock.set_focused_node_and_surface((surface, node));
        }
    }
}

// --- the step bar ------------------------------------------------------------------------

fn index(phase: Phase) -> usize {
    Phase::ALL.iter().position(|p| *p == phase).unwrap_or(0)
}

/// Where a project opens: the first step that is not done, or the results once all are.
pub fn start_phase(states: &[PhaseState; 5]) -> Phase {
    Phase::ALL
        .into_iter()
        .zip(states)
        .find(|(_, s)| **s != PhaseState::Done)
        .map_or(Phase::Results, |(p, _)| p)
}

/// What `Next` goes to: the step after `phase`, when it can be opened.
pub fn next_phase(phase: Phase, states: &[PhaseState; 5]) -> Option<Phase> {
    let next = index(phase) + 1;
    let phase = *Phase::ALL.get(next)?;
    can_open(&states[next]).then_some(phase)
}

/// A step can be opened unless it waits for the one before it.
pub fn can_open(state: &PhaseState) -> bool {
    *state != PhaseState::Locked
}

/// How many steps are not done yet.
fn left(states: &[PhaseState; 5]) -> usize {
    states.iter().filter(|s| **s != PhaseState::Done).count()
}

/// What the step bar says is left: how many steps, or that every one is done.
pub fn left_text(lang: Lang, states: &[PhaseState; 5]) -> String {
    match left(states) {
        0 => i18n::t(lang, "shell.all_done").to_owned(),
        n => i18n::fill(lang, "shell.left", &[&n.to_string()]),
    }
}

/// The colour of the dot before a step's name: green done, blue running, red failed, amber
/// when it waits for the person to resume, grey when it has not begun and darker grey when it
/// cannot. Mid tones, so the dot reads on a light theme and a dark one alike; the name beside
/// it keeps the theme's own text colour.
pub fn colour(state: &PhaseState) -> [u8; 3] {
    match state {
        PhaseState::Done => [70, 170, 90],
        PhaseState::Running { .. } => [60, 140, 230],
        PhaseState::Failed { .. } => [220, 80, 70],
        PhaseState::Interrupted { .. } | PhaseState::StoppedByYou { .. } => [230, 165, 40],
        PhaseState::NotStarted => [160, 160, 160],
        PhaseState::Locked => [110, 110, 110],
    }
}

/// "③ Train 62%": the step's number, its name, and how far a running step has got.
pub fn step_text(lang: Lang, phase: Phase, state: &PhaseState) -> String {
    const NUMBERS: [char; 5] = ['\u{2460}', '\u{2461}', '\u{2462}', '\u{2463}', '\u{2464}'];
    let name = i18n::t(lang, phase.key());
    let number = NUMBERS[index(phase)];
    match state {
        PhaseState::Running {
            fraction: Some(f), ..
        } => format!("{number} {name} {:.0}%", f * 100.0),
        _ => format!("{number} {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::t;
    use PhaseState::{Done, Failed, Interrupted, Locked, NotStarted, Running, StoppedByYou};

    #[test]
    fn every_phase_has_a_layout_with_the_viewport_in_the_centre() {
        for p in Phase::ALL {
            let l = default_layout(p);
            assert_eq!(l.centre, vec![Pane::Viewport], "{p:?}");
            assert!(l.left.contains(&Pane::StepPanel));
        }
    }

    #[test]
    fn every_pane_has_a_label_in_both_languages() {
        for p in Pane::ALL {
            for lang in [Lang::En, Lang::Ko] {
                let s = t(lang, p.key());
                assert!(
                    !s.is_empty() && s != p.key() && !s.contains('_'),
                    "{p:?} {lang:?}"
                );
            }
        }
        let keys: std::collections::BTreeSet<_> = Pane::ALL.iter().map(|p| p.key()).collect();
        assert_eq!(keys.len(), Pane::ALL.len(), "one name per pane");
        for p in Pane::ALL {
            let advanced = p.hint_key().is_some();
            assert_eq!(advanced, format!("{p:?}").starts_with("Advanced"), "{p:?}");
        }
    }

    #[test]
    fn the_side_panes_scroll_and_the_viewport_does_not() {
        assert!(Pane::StepPanel.scrolls() && Pane::Summary.scrolls() && !Pane::Viewport.scrolls());
    }

    fn arrangements() -> Vec<Arrangement> {
        Phase::ALL
            .map(Arrangement::Step)
            .into_iter()
            .chain([Arrangement::Advanced])
            .collect()
    }

    fn held(dock: &DockState<Pane>) -> Vec<Pane> {
        let mut panes: Vec<Pane> = dock.iter_all_tabs().map(|(_, p)| *p).collect();
        panes.sort();
        panes
    }

    /// No pane is lost and none is shown twice: a step holds all nine, the Advanced
    /// arrangement the five old tabs and the log, and the dock built from a layout holds
    /// exactly what the layout lists.
    #[test]
    fn every_arrangement_holds_its_panes_once() {
        for a in arrangements() {
            let layout = a.layout();
            let panes = layout.panes();
            let mut unique = panes.clone();
            unique.dedup();
            assert_eq!(panes, unique, "{a:?} lists a pane twice");
            assert_eq!(held(&layout.dock()), panes, "{a:?}");
            match a {
                Arrangement::Step(_) => assert_eq!(panes, Pane::ALL.to_vec(), "{a:?}"),
                Arrangement::Advanced => {
                    assert!(panes.contains(&Pane::Console));
                    assert!(!panes.contains(&Pane::Viewport));
                    let advanced = Pane::ALL.iter().filter(|p| p.hint_key().is_some());
                    assert!(advanced.clone().all(|p| layout.centre.contains(p)));
                }
            }
        }
        assert_eq!(Arrangement::of(None), Arrangement::Advanced);
        assert_eq!(
            Arrangement::of(Some(Phase::Train)),
            Arrangement::Step(Phase::Train)
        );
    }

    #[derive(Default)]
    struct Memory(BTreeMap<String, String>);

    impl eframe::Storage for Memory {
        fn get_string(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.0.insert(key.to_owned(), value);
        }
        fn flush(&mut self) {}
    }

    /// The person's arrangement survives a restart; a store that is absent, unreadable or
    /// short of a pane gives that arrangement its default back.
    #[test]
    fn a_stored_arrangement_round_trips_and_a_bad_one_falls_back() {
        let train = Arrangement::Step(Phase::Train);
        let default = train.layout().dock();
        let mut docks = Docks::default();
        // A person's change: the summary dragged into the step panel's group.
        let dock = docks.get(train);
        let (surface, node, tab) = dock.find_tab(&Pane::Summary).expect("the summary");
        let summary = dock.remove_tab((surface, node, tab)).expect("removed");
        let (step, _) = dock.find_main_surface_tab(&Pane::StepPanel).expect("step");
        dock.set_focused_node_and_surface((surface, step));
        dock.push_to_focused_leaf(summary);
        let moved = |d: &DockState<Pane>| d.find_main_surface_tab(&Pane::Summary).map(|x| x.0);
        assert_eq!(moved(docks.get(train)), Some(step));
        assert_ne!(moved(&default), Some(step));

        let mut store = Memory::default();
        docks.save(&mut store);
        let mut back = Docks::load(Some(&store));
        assert_eq!(moved(back.get(train)), Some(step), "kept across a restart");
        assert_eq!(held(back.get(train)), Pane::ALL.to_vec());
        back.reset(train);
        assert_eq!(moved(back.get(train)), moved(&default), "reset");

        for (what, store) in [("absent", None), ("unreadable", Some("not ron at all"))] {
            let mut memory = Memory::default();
            if let Some(text) = store {
                eframe::Storage::set_string(&mut memory, LAYOUT_KEY, text.to_owned());
            }
            let mut docks = Docks::load(Some(&memory));
            assert_eq!(moved(docks.get(train)), moved(&default), "{what}");
        }
        assert_eq!(moved(Docks::load(None).get(train)), moved(&default));

        // Stored by a version that had one pane fewer: that arrangement starts over.
        let mut short = Docks::default();
        let dock = short.get(train);
        let found = dock.find_tab(&Pane::AdvancedLive).expect("live");
        dock.remove_tab(found);
        let mut store = Memory::default();
        short.save(&mut store);
        assert_eq!(
            held(Docks::load(Some(&store)).get(train)),
            Pane::ALL.to_vec()
        );
    }

    /// Focusing a pane makes it the active tab of its group, wherever the person put it.
    #[test]
    fn focus_makes_a_pane_the_active_tab() {
        let mut docks = Docks::default();
        for a in arrangements() {
            for pane in a.layout().panes() {
                docks.focus(a, pane);
                let (_, active) = docks.get(a).find_active_focused().expect("a focused pane");
                assert_eq!(*active, pane, "{a:?}");
            }
        }
    }

    fn fresh() -> [PhaseState; 5] {
        [Done, Done, NotStarted, Locked, Locked]
    }

    #[test]
    fn the_step_bar_reads_the_states() {
        assert_eq!(
            start_phase(&fresh()),
            Phase::Train,
            "a new project opens at training"
        );
        assert_eq!(start_phase(&[Done, Done, Done, Done, Done]), Phase::Results);
        let interrupted = [
            Done,
            Done,
            Done,
            Interrupted {
                resume_from: Some("eval".into()),
            },
            Locked,
        ];
        assert_eq!(start_phase(&interrupted), Phase::Evaluate);

        assert_eq!(next_phase(Phase::Scene, &fresh()), Some(Phase::Teach));
        assert_eq!(
            next_phase(Phase::Train, &fresh()),
            None,
            "evaluate is locked"
        );
        assert_eq!(
            next_phase(Phase::Results, &[Done, Done, Done, Done, Done]),
            None
        );
        assert!(!can_open(&Locked));
        assert!(can_open(&NotStarted) && can_open(&Done));

        assert_eq!(left(&fresh()), 3);
        assert_eq!(left(&[Done, Done, Done, Done, Done]), 0);
        for lang in Lang::ALL {
            assert!(left_text(lang, &fresh()).contains('3'));
            assert_eq!(
                left_text(lang, &[Done, Done, Done, Done, Done]),
                t(lang, "shell.all_done")
            );
        }

        let running = Running {
            stage: "train".into(),
            fraction: Some(0.62),
        };
        for lang in Lang::ALL {
            let text = step_text(lang, Phase::Train, &running);
            assert!(text.contains(t(lang, Phase::Train.key())), "{text}");
            assert!(text.contains("62%") && text.contains('\u{2462}'), "{text}");
            let still = step_text(lang, Phase::Train, &NotStarted);
            assert!(!still.contains('%'), "{still}");
        }

        let states = [
            Done,
            NotStarted,
            Locked,
            running,
            Failed {
                stage: "train".into(),
                code: Some(1),
            },
            Interrupted { resume_from: None },
            StoppedByYou { resume_from: None },
        ];
        let colours: std::collections::BTreeSet<[u8; 3]> = states.iter().map(colour).collect();
        assert!(
            colours.len() >= 5,
            "done, waiting, locked, running and failed differ"
        );
        assert_ne!(colour(&Done), colour(&states[4]));
    }
}
