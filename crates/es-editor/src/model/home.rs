//! The start screen's model (`docs/design/editor-redesign.md` section 3, packet M12/Y9): what
//! this PC has, which templates it can run, and what the recent paths are now.
//!
//! The PC check is `es --check-deps --json` (packet M12/Y1), started by [`DepsProbe`] on its own
//! thread because the probe imports Python modules and can take half a minute. A template that
//! this PC cannot run is not hidden: [`availability`] names what is missing, and the card is
//! drawn disabled with those names in plain words ([`need_label`]).
//!
//! What the start screen itself decides (packet M12/Y11) is here too: how each PC-check item is
//! marked ([`pc_check`]), what a card that cannot be created says ([`availability_text`]), where
//! a new project goes ([`NewProject`]), and when the start screen fills the window
//! ([`StartScreen::shown`]). `ui/home.rs` only draws the answers.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};

use serde::Deserialize;

use crate::model::i18n::{self, Lang};
use crate::model::project::{Project, ProjectError};
use crate::model::recent::{self, Recent};
use crate::model::template::{load, templates_root, Template};
use crate::model::workflow::{phases, PhaseState, RunFacts};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Python {
    pub found: bool,
    pub path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Modules {
    pub mujoco: bool,
    pub torch: bool,
    pub lerobot: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Backend {
    pub name: String,
    pub available: bool,
    pub reason: Option<String>,
}

/// What `es --check-deps --json` prints (packet M12/Y1).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Deps {
    pub schema: u32,
    pub python: Python,
    pub modules: Modules,
    pub vulkan_loader: bool,
    pub render: bool,
    pub backends: Vec<Backend>,
}

/// The one schema this editor reads. Another number is another format, refused rather than
/// half-read.
const SCHEMA: u32 = 1;

pub fn parse_deps(json: &str) -> Result<Deps, String> {
    let deps: Deps = serde_json::from_str(json.trim())
        .map_err(|e| format!("es --check-deps --json printed something else: {e}"))?;
    if deps.schema != SCHEMA {
        return Err(format!(
            "es --check-deps --json speaks schema {}; this editor reads schema {SCHEMA}",
            deps.schema
        ));
    }
    Ok(deps)
}

/// `es --check-deps --json` on its own thread; [`DepsProbe::poll`] reads its one answer without
/// blocking.
#[derive(Debug)]
pub struct DepsProbe {
    rx: Receiver<DepsState>,
    state: DepsState,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DepsState {
    Checking,
    Ready(Deps),
    Failed(String),
}

impl DepsProbe {
    pub fn start(es: &Path) -> Self {
        Self::spawn(es, &["--check-deps", "--json"])
    }

    /// [`Self::start`] with the command line as a parameter, so the oracle runs the platform
    /// shell down the same path a real probe takes.
    fn spawn(program: &Path, args: &[&str]) -> Self {
        let (tx, rx) = mpsc::channel();
        let mut command = Command::new(program);
        command.args(args).stdin(Stdio::null());
        let shown = program.display().to_string();
        std::thread::spawn(move || {
            let _ = tx.send(probe(command, &shown));
        });
        Self {
            rx,
            state: DepsState::Checking,
        }
    }

    pub fn poll(&mut self) -> &DepsState {
        if let Ok(answer) = self.rx.try_recv() {
            self.state = answer;
        }
        &self.state
    }
}

/// Runs the probe to the end. A child that cannot start, exits non-zero, or prints anything but
/// the JSON is `Failed` with why - never `Ready` on a guess.
fn probe(mut command: Command, shown: &str) -> DepsState {
    let out = match command.output() {
        Ok(out) => out,
        Err(e) => return DepsState::Failed(format!("{shown}: {e}")),
    };
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return DepsState::Failed(format!(
            "{shown} --check-deps --json ended with {}: {}",
            out.status,
            stderr.trim()
        ));
    }
    match parse_deps(&String::from_utf8_lossy(&out.stdout)) {
        Ok(deps) => DepsState::Ready(deps),
        Err(why) => DepsState::Failed(why),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability {
    Ready,
    /// The template's `needs` names this PC lacks, in the template's order.
    Missing(Vec<String>),
    /// The PC check has not answered (or failed): nothing is claimed either way.
    Unknown,
}

/// Every `needs` name this editor knows: its label key, and where the probe reports it.
type Need = (&'static str, &'static str, fn(&Deps) -> bool);
const NEEDS: [Need; 5] = [
    ("mujoco", "deps.mujoco", |d| d.modules.mujoco),
    ("torch", "deps.torch", |d| d.modules.torch),
    ("lerobot", "deps.lerobot", |d| d.modules.lerobot),
    ("vulkan", "deps.vulkan", |d| d.vulkan_loader),
    ("render", "deps.render", |d| d.render),
];

fn need(name: &str) -> Option<&'static Need> {
    NEEDS.iter().find(|n| n.0 == name)
}

/// The i18n key naming a `needs` item in plain words, or `None` for a name this editor does
/// not know - drawn as the name itself.
pub fn need_label(name: &str) -> Option<&'static str> {
    need(name).map(|n| n.1)
}

/// `needs` names: "mujoco" | "torch" | "lerobot" (modules), "vulkan" (loader), "render". A name
/// the probe does not report is not vouched for: it counts as missing.
pub fn availability(template: &Template, deps: Option<&Deps>) -> Availability {
    let Some(deps) = deps else {
        return Availability::Unknown;
    };
    let missing: Vec<String> = template
        .needs
        .iter()
        .filter(|name| !need(name).is_some_and(|n| (n.2)(deps)))
        .cloned()
        .collect();
    if missing.is_empty() {
        Availability::Ready
    } else {
        Availability::Missing(missing)
    }
}

/// One recent path, as the start screen lists it.
// At most `recent::CAP` cards exist, so boxing `phases` would buy nothing.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum RecentCard {
    /// `phases` is what disk says about the latest run: a card is drawn without attaching.
    Project {
        path: PathBuf,
        name: String,
        phases: [PhaseState; 5],
    },
    Other {
        path: PathBuf,
        kind: recent::Kind,
    },
    Missing {
        path: PathBuf,
    },
}

/// A path that is gone is `Missing`; a folder [`Project::open`] reads is a `Project`; anything
/// else is what [`recent::classify`] says - a broken `project.toml` is `Kind::Project` there,
/// which the start screen draws as a project it cannot open.
pub fn recent_cards(recent: &Recent) -> Vec<RecentCard> {
    recent
        .paths
        .iter()
        .map(|p| {
            let path = p.clone();
            if !p.exists() {
                RecentCard::Missing { path }
            } else if let Ok(project) = Project::open(p) {
                let facts = project.latest_run().map(|run| RunFacts::read(&run));
                RecentCard::Project {
                    path,
                    name: project.file.name,
                    phases: phases(facts.as_ref(), None),
                }
            } else {
                RecentCard::Other {
                    path,
                    kind: recent::classify(p),
                }
            }
        })
        .collect()
}

// --- the start screen (packet M12/Y11) --------------------------------------------------------

/// How one item of the PC-check line is marked: green when this PC has it, red when a template
/// needs it and it is not here, amber for an optional simulator that is not here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Have,
    Missing,
    Optional,
}

impl Mark {
    /// The same mid tones as the step bar's dots (`layout::colour`), so one green means one thing.
    pub fn colour(self) -> [u8; 3] {
        match self {
            Mark::Have => [70, 170, 90],
            Mark::Missing => [220, 80, 70],
            Mark::Optional => [230, 165, 40],
        }
    }

    /// A shape as well as a colour, for whoever cannot tell red from green.
    pub fn glyph(self) -> &'static str {
        match self {
            Mark::Have => "\u{2714}",
            Mark::Missing => "\u{2716}",
            Mark::Optional => "\u{25cb}",
        }
    }

    /// The hover: what the mark means.
    pub fn key(self) -> &'static str {
        match self {
            Mark::Have => "home.mark.have",
            Mark::Missing => "home.mark.missing",
            Mark::Optional => "home.mark.optional",
        }
    }
}

/// One item of the PC-check line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckItem {
    /// Its plain name; `home.optional` takes the simulator's name as its `{}`.
    pub key: &'static str,
    pub name: Option<String>,
    pub mark: Mark,
    /// Why an optional simulator is not here, in `es`'s own words (not translated: they are
    /// another program's).
    pub reason: Option<String>,
}

/// The PC-check line: Python, every `needs` name the editor knows, then each physics backend as
/// an optional extra.
pub fn pc_check(deps: &Deps) -> Vec<CheckItem> {
    let item = |key, have: bool| CheckItem {
        key,
        name: None,
        mark: if have { Mark::Have } else { Mark::Missing },
        reason: None,
    };
    let mut items = vec![item("deps.python", deps.python.found)];
    items.extend(NEEDS.iter().map(|n| item(n.1, (n.2)(deps))));
    items.extend(deps.backends.iter().map(|b| CheckItem {
        key: "home.optional",
        name: Some(b.name.clone()),
        mark: if b.available {
            Mark::Have
        } else {
            Mark::Optional
        },
        reason: b.reason.clone(),
    }));
    items
}

/// Whether the line needs the sentence on how to install what is missing: only for a red item.
pub fn needs_install(items: &[CheckItem]) -> bool {
    items.iter().any(|i| i.mark == Mark::Missing)
}

/// What a card (or a File-menu item) says instead of offering Create, and in which colour: the
/// missing items in plain words (red), or that the PC check has not answered (amber). `None`
/// when the template is ready.
pub fn availability_text(lang: Lang, availability: &Availability) -> Option<(Mark, String)> {
    match availability {
        Availability::Ready => None,
        Availability::Unknown => Some((Mark::Optional, i18n::t(lang, "home.unknown").to_owned())),
        Availability::Missing(names) => {
            let words: Vec<&str> = names
                .iter()
                .map(|n| need_label(n).map_or(n.as_str(), |k| i18n::t(lang, k)))
                .collect();
            let text = i18n::fill(lang, "home.missing", &[&words.join(", ")]);
            Some((Mark::Missing, text))
        }
    }
}

/// Where new projects go by default, under the person's Documents folder.
const PROJECTS_DIR: &str = "Electric Sheep";

/// `<home>/Documents`: `USERPROFILE` on Windows, `HOME` elsewhere.
///
/// ponytail: an environment variable and a fixed `Documents`, not the platform's known-folder
/// call (a dependency and `unsafe` this crate forbids); a Documents folder redirected elsewhere
/// (to a cloud drive) is not followed. The dialog's folder picker is the way out; the
/// known-folder call comes with a dependency the orchestrator approves.
pub fn documents_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(PathBuf::from(home).join("Documents"))
}

/// `<documents>/Electric Sheep/<name>`.
pub fn default_folder(documents: Option<&Path>, name: &str) -> PathBuf {
    let base = documents.map_or_else(|| PathBuf::from(PROJECTS_DIR), |d| d.join(PROJECTS_DIR));
    free(&base, name)
}

/// `<base>/<name>`, or `<name> 2`, `<name> 3`... when that is taken: a second project from the
/// same template does not land on the first one.
fn free(base: &Path, name: &str) -> PathBuf {
    let name = name.trim();
    let mut path = base.join(name);
    let mut n = 1;
    while path.exists() {
        n += 1;
        path = base.join(format!("{name} {n}"));
    }
    path
}

/// The new-project dialog: a template, a name and a folder.
#[derive(Clone, Debug, PartialEq)]
pub struct NewProject {
    pub template: Template,
    /// The directory holding `templates/`, which `Project::create` reads the documents from.
    pub root: PathBuf,
    pub name: String,
    pub folder: String,
    documents: Option<PathBuf>,
    /// Once the person has typed or picked a folder, the name no longer moves it.
    chosen: bool,
    /// Why the last Create failed, as `Project::create` said it.
    pub error: Option<String>,
}

impl NewProject {
    /// `name` is the template's plain name in the reader's language; the folder follows it.
    pub fn new(template: Template, root: PathBuf, name: &str, documents: Option<PathBuf>) -> Self {
        Self {
            folder: default_folder(documents.as_deref(), name)
                .display()
                .to_string(),
            template,
            root,
            name: name.to_owned(),
            documents,
            chosen: false,
            error: None,
        }
    }

    pub fn name_changed(&mut self) {
        self.error = None;
        if !self.chosen {
            self.folder = default_folder(self.documents.as_deref(), &self.name)
                .display()
                .to_string();
        }
    }

    pub fn folder_typed(&mut self) {
        self.error = None;
        self.chosen = true;
    }

    /// What the OS folder picker returned. An empty folder is the project's; one that already
    /// holds files gets a new folder inside it, so a project never spills into, say, Documents.
    pub fn folder_picked(&mut self, dir: &Path) {
        let empty = std::fs::read_dir(dir).is_ok_and(|mut e| e.next().is_none());
        let folder = if empty {
            dir.to_path_buf()
        } else {
            free(dir, &self.name)
        };
        self.folder = folder.display().to_string();
        self.folder_typed();
    }

    /// Why Create is refused before it is pressed, as a key: the folder already holds a project,
    /// which `Project::create` would refuse in its own (untranslated) words.
    pub fn blocker(&self) -> Option<&'static str> {
        Project::is_project_dir(Path::new(self.folder.trim())).then_some("home.already_project")
    }

    pub fn can_create(&self) -> bool {
        !self.name.trim().is_empty() && !self.folder.trim().is_empty() && self.blocker().is_none()
    }

    pub fn create(&self) -> Result<Project, ProjectError> {
        Project::create(
            Path::new(self.folder.trim()),
            self.name.trim(),
            &self.template,
            &self.root,
        )
    }
}

/// The start screen's state: the PC check, the templates, the new-project dialog, and whether
/// the person asked for the workspace with nothing open.
#[derive(Debug, Default)]
pub struct StartScreen {
    probe: Option<DepsProbe>,
    /// The directory holding `templates/`; `None` when none was found above the editor.
    pub root: Option<PathBuf>,
    pub templates: Vec<Template>,
    /// Every template that did not parse, with why.
    pub broken: Vec<(PathBuf, String)>,
    pub dialog: Option<NewProject>,
    /// Set when a pane is asked for (File > Watch a running run): the dock shows even though
    /// nothing is open.
    pub workspace: bool,
    cards: Option<(Recent, Vec<RecentCard>)>,
}

impl StartScreen {
    /// The templates beside this editor's checkout (`template::templates_root`).
    pub fn load() -> Self {
        Self::at(templates_root())
    }

    fn at(root: Option<PathBuf>) -> Self {
        let (templates, broken) = root.as_deref().map(load).unwrap_or_default();
        Self {
            root,
            templates,
            broken,
            ..Self::default()
        }
    }

    /// The PC check's answer so far. The probe is started on the first call, with `es`, so an
    /// editor that never shows the start screen or the New project menu never runs it.
    pub fn poll(&mut self, es: &Path) -> &DepsState {
        self.probe
            .get_or_insert_with(|| DepsProbe::start(es))
            .poll()
    }

    /// The recent cards, read again only when the list changes: reading them opens every
    /// project on it, which is not something to do every frame.
    pub fn cards(&mut self, recent: &Recent) -> &[RecentCard] {
        if self.cards.as_ref().is_none_or(|(r, _)| r != recent) {
            self.cards = Some((recent.clone(), recent_cards(recent)));
        }
        self.cards.as_ref().map_or(&[], |(_, c)| c)
    }

    /// Opens the dialog for `templates[index]`, named `name` (its plain name).
    pub fn open_dialog(&mut self, index: usize, name: &str) {
        if let (Some(template), Some(root)) = (self.templates.get(index), &self.root) {
            self.dialog = Some(NewProject::new(
                template.clone(),
                root.clone(),
                name,
                documents_dir(),
            ));
        }
    }

    /// The start screen fills the window unless something is open, a run is streaming in, or
    /// the person asked for a pane.
    pub fn shown(&self, open: bool, streaming: bool) -> bool {
        !(open || streaming || self.workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::i18n::{Lang, Strings};
    use crate::model::template::{find_root, load};
    use std::time::{Duration, Instant};

    const READY: &str = r#"{"schema":1,"python":{"found":true,"path":"p"},
  "modules":{"mujoco":true,"torch":true,"lerobot":true},"vulkan_loader":true,"render":true,
  "backends":[{"name":"mujoco-cpu","available":true}]}"#;

    fn templates() -> Vec<Template> {
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
        load(&root).0
    }

    fn cube() -> Template {
        templates()
            .into_iter()
            .find(|t| t.id == "cube-into-bin")
            .expect("the committed cube template")
    }

    #[test]
    fn ready_deps_make_the_cube_template_ready() {
        let d = parse_deps(READY).unwrap();
        assert_eq!(availability(&cube(), Some(&d)), Availability::Ready);
        assert_eq!(availability(&cube(), None), Availability::Unknown);
    }

    #[test]
    fn a_missing_module_or_render_names_what_is_missing() {
        let mut d = parse_deps(READY).unwrap();
        d.modules.lerobot = false;
        d.render = false;
        assert_eq!(
            availability(&cube(), Some(&d)),
            Availability::Missing(vec!["lerobot".into(), "render".into()])
        );
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(parse_deps("es --check-deps (spec 2.5)").is_err());
        assert!(parse_deps("").is_err());
        // Another schema is another format: refused, not half-read.
        assert!(parse_deps(&READY.replace("\"schema\":1", "\"schema\":2")).is_err());
    }

    #[test]
    fn recent_cards_mark_what_is_gone() {
        let r = Recent {
            paths: vec![PathBuf::from("Z:/nowhere/at/all")],
        };
        assert!(matches!(recent_cards(&r)[0], RecentCard::Missing { .. }));
    }

    /// A project folder is a card with its name; a run, a documents folder, and a folder whose
    /// `project.toml` is broken are the file-level kinds `recent::classify` says they are.
    #[test]
    fn recent_cards_read_a_project_and_classify_the_rest() {
        let tmp = std::env::temp_dir().join(format!("es-home-{}", std::process::id()));
        let (good, broken) = (tmp.join("my robot"), tmp.join("broken"));
        std::fs::create_dir_all(&good).unwrap();
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(
            good.join("project.toml"),
            "kind = \"project\"\nname = \"Cube try 2\"\ntemplate = \"cube-into-bin\"\n",
        )
        .unwrap();
        std::fs::write(broken.join("project.toml"), "kind = \"project\"\n").unwrap();
        let run =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/visible-learning/run");
        let r = Recent {
            paths: vec![good.clone(), run.clone(), broken.clone()],
        };
        assert_eq!(
            recent_cards(&r),
            vec![
                // No run yet: ① ② done, ③ not started (`workflow::phases(None, None)`).
                RecentCard::Project {
                    path: good,
                    name: "Cube try 2".into(),
                    phases: phases(None, None),
                },
                RecentCard::Other {
                    path: run,
                    kind: recent::Kind::Run
                },
                // A `project.toml` that does not parse is still a project folder to `classify`.
                RecentCard::Other {
                    path: broken,
                    kind: recent::Kind::Project
                },
            ]
        );
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Every `needs` name a committed template uses has a plain-language label in both
    /// languages; a name the editor does not know has none and is never vouched for.
    #[test]
    fn every_committed_need_has_a_label_and_an_unknown_one_is_missing() {
        let all = templates();
        assert!(!all.is_empty());
        for need in all.iter().flat_map(|t| &t.needs) {
            let key = need_label(need).unwrap_or_else(|| panic!("no label for {need}"));
            for lang in Lang::ALL {
                assert_ne!(Strings::get(lang).t(key), key, "{need}");
            }
        }
        assert_eq!(need_label("warp-drive"), None);
        let mut t = cube();
        t.needs = vec!["mujoco".into(), "warp-drive".into()];
        let d = parse_deps(READY).unwrap();
        assert_eq!(
            availability(&t, Some(&d)),
            Availability::Missing(vec!["warp-drive".into()])
        );
    }

    /// The platform's "run this one command line" shell, as `launch.rs`'s oracle uses it.
    fn shell(script: &str) -> (PathBuf, [String; 2]) {
        if cfg!(windows) {
            ("cmd".into(), ["/C".into(), script.into()])
        } else {
            ("sh".into(), ["-c".into(), script.into()])
        }
    }

    fn probe(script: &str) -> DepsProbe {
        let (program, [a, b]) = shell(script);
        DepsProbe::spawn(&program, &[&a, &b])
    }

    /// Polls until the probe leaves `Checking`, or gives up after 30 s.
    fn settle(p: &mut DepsProbe) -> DepsState {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if *p.poll() != DepsState::Checking {
                return p.poll().clone();
            }
            assert!(Instant::now() < deadline, "the probe never answered");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// `poll` answers at once while the child still runs; a child that prints something other
    /// than the JSON, exits non-zero, or cannot start at all ends in `Failed`, never `Ready`.
    #[test]
    fn the_probe_never_blocks_and_every_failure_is_failed() {
        let mut slow = probe(if cfg!(windows) {
            "ping -n 3 127.0.0.1"
        } else {
            "sleep 2; echo not json"
        });
        let started = Instant::now();
        assert_eq!(*slow.poll(), DepsState::Checking);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "poll blocked"
        );
        assert!(matches!(settle(&mut slow), DepsState::Failed(_)));

        let DepsState::Failed(why) = settle(&mut probe("exit 3")) else {
            panic!("a non-zero exit is a failure")
        };
        assert!(why.contains('3'), "{why}");

        let nowhere = Path::new("Z:/nowhere/es-that-does-not-exist");
        let DepsState::Failed(why) = settle(&mut DepsProbe::start(nowhere)) else {
            panic!("a binary that cannot start is a failure")
        };
        assert!(why.contains("es-that-does-not-exist"), "{why}");
    }

    /// The good path through the same spawn: the child's stdout is parsed into `Ready`.
    #[test]
    fn the_probe_reads_what_the_child_prints() {
        let file = std::env::temp_dir().join(format!("es-home-deps-{}.json", std::process::id()));
        std::fs::write(&file, READY.replace('\n', "")).unwrap();
        let cat = if cfg!(windows) { "type" } else { "cat" };
        let state = settle(&mut probe(&format!("{cat} {}", file.display())));
        std::fs::remove_file(&file).ok();
        assert_eq!(state, DepsState::Ready(parse_deps(READY).unwrap()));
    }

    /// The PC-check line (packet M12/Y11): Python, then every `needs` item, then each simulator
    /// as an optional extra. Red only for what a template needs; an absent simulator is amber
    /// and keeps `es`'s reason.
    #[test]
    fn pc_check_marks_what_is_here_missing_and_optional() {
        let all = pc_check(&parse_deps(READY).unwrap());
        assert!(all.iter().all(|i| i.mark == Mark::Have), "{all:?}");
        assert!(!needs_install(&all));
        assert_eq!(all[0].key, "deps.python");
        assert_eq!(all.len(), 1 + NEEDS.len() + 1);

        let mut d = parse_deps(READY).unwrap();
        d.modules.lerobot = false;
        d.backends.push(Backend {
            name: "physx".into(),
            available: false,
            reason: Some("no Isaac Sim".into()),
        });
        let items = pc_check(&d);
        let lerobot = items.iter().find(|i| i.key == "deps.lerobot").unwrap();
        assert_eq!(lerobot.mark, Mark::Missing);
        let physx = items.last().unwrap();
        assert_eq!(
            (physx.key, physx.name.as_deref(), physx.mark),
            ("home.optional", Some("physx"), Mark::Optional)
        );
        assert_eq!(physx.reason.as_deref(), Some("no Isaac Sim"));
        assert!(needs_install(&items));
        // An absent simulator alone asks nobody to install anything.
        d.modules.lerobot = true;
        assert!(!needs_install(&pc_check(&d)));

        for lang in Lang::ALL {
            for mark in [Mark::Have, Mark::Missing, Mark::Optional] {
                assert_ne!(Strings::get(lang).t(mark.key()), mark.key(), "{mark:?}");
            }
        }
    }

    /// A card that cannot be created says why in plain words; one that can says nothing.
    #[test]
    fn availability_text_names_what_is_missing_in_plain_words() {
        for lang in Lang::ALL {
            assert_eq!(availability_text(lang, &Availability::Ready), None);
            let (mark, unknown) = availability_text(lang, &Availability::Unknown).unwrap();
            assert_ne!(unknown, "home.unknown");
            assert_eq!(mark, Mark::Optional, "waiting is not an error");
            let missing = Availability::Missing(vec!["lerobot".into(), "warp-drive".into()]);
            let (mark, text) = availability_text(lang, &missing).unwrap();
            assert_eq!(mark, Mark::Missing);
            assert!(text.contains(i18n::t(lang, "deps.lerobot")), "{text}");
            // A name the editor has no word for is shown as itself, never dropped.
            assert!(text.contains("warp-drive"), "{text}");
            assert!(!text.contains("{}"), "{text}");
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("es-home-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The folder follows the name under `<Documents>/Electric Sheep` until the person types or
    /// picks one; a taken folder gets a number; a picked folder that holds files gets the
    /// project in a new folder inside it.
    #[test]
    fn the_new_project_folder_follows_the_name_until_chosen() {
        let docs = scratch("docs");
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let base = docs.join("Electric Sheep");
        let mut d = NewProject::new(cube(), root, "Cube try", Some(docs.clone()));
        assert_eq!(d.folder, base.join("Cube try").display().to_string());
        assert!(d.can_create());

        std::fs::create_dir_all(base.join("My arm")).unwrap();
        "My arm".clone_into(&mut d.name);
        d.name_changed();
        assert_eq!(d.folder, base.join("My arm 2").display().to_string());

        "  ".clone_into(&mut d.name);
        d.name_changed();
        assert!(!d.can_create(), "a blank name creates nothing");

        "typed by hand".clone_into(&mut d.folder);
        d.folder_typed();
        "Cube try".clone_into(&mut d.name);
        d.name_changed();
        assert_eq!(d.folder, "typed by hand", "a typed folder stays put");

        let empty = docs.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        d.folder_picked(&empty);
        assert_eq!(d.folder, empty.display().to_string());
        d.folder_picked(&docs);
        assert_eq!(d.folder, docs.join("Cube try").display().to_string());
        std::fs::remove_dir_all(&docs).ok();
    }

    /// Create makes the project at the chosen folder, and a project with no run opens at step
    /// three: the template did steps one and two. A second Create on the same folder is refused
    /// with a sentence, and the dialog's next default does not collide with it.
    #[test]
    fn creating_a_project_opens_it_at_train() {
        use crate::model::layout::start_phase;
        use crate::model::workflow::Phase;

        let docs = scratch("create");
        let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).unwrap();
        let mut screen = StartScreen::at(Some(root));
        let index = screen.templates.iter().position(|t| t.id == cube().id);
        screen.open_dialog(index.unwrap(), "My cube");
        let mut d = screen.dialog.take().expect("the dialog opened");
        d.documents = Some(docs.clone());
        d.name_changed();

        let project = d.create().expect("created");
        assert_eq!(project.file.name, "My cube");
        let reopened = Project::open(Path::new(&d.folder)).unwrap();
        let facts = reopened.latest_run().map(|run| RunFacts::read(&run));
        assert_eq!(start_phase(&phases(facts.as_ref(), None)), Phase::Train);

        assert!(d.create().is_err(), "the same folder twice");
        // ...which the dialog says in its own words before Create is pressed.
        assert_eq!(d.blocker(), Some("home.already_project"));
        assert!(!d.can_create());
        d.name_changed();
        assert!(d.folder.ends_with("My cube 2"), "{}", d.folder);
        assert_eq!(d.blocker(), None);
        assert!(d.can_create());
        std::fs::remove_dir_all(&docs).ok();
    }

    /// The start screen fills the window only while there is nothing else to show.
    #[test]
    fn the_start_screen_gives_way_to_anything_open() {
        let mut screen = StartScreen::at(None);
        assert!(screen.templates.is_empty() && screen.broken.is_empty());
        assert!(screen.shown(false, false));
        assert!(
            !screen.shown(true, false),
            "a project, a file or a run is open"
        );
        assert!(!screen.shown(false, true), "a run is streaming in");
        screen.workspace = true;
        assert!(!screen.shown(false, false), "a pane was asked for");
        screen.open_dialog(0, "x");
        assert_eq!(screen.dialog, None, "no template, no dialog");
    }
}
