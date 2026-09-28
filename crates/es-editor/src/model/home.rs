//! The start screen's model (`docs/design/editor-redesign.md` section 3, packet M12/Y9): what
//! this PC has, which templates it can run, and what the recent paths are now.
//!
//! The PC check is `es --check-deps --json` (packet M12/Y1), started by [`DepsProbe`] on its own
//! thread because the probe imports Python modules and can take half a minute. A template that
//! this PC cannot run is not hidden: [`availability`] names what is missing, and the card is
//! drawn disabled with those names in plain words ([`need_label`]).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};

use serde::Deserialize;

use crate::model::recent::{self, Recent};
use crate::model::template::Template;

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
// Packet M12/Y6's merge adds `phases: [PhaseState; 5]` to `Project`, read from the latest run.
#[derive(Clone, Debug, PartialEq)]
pub enum RecentCard {
    Project { path: PathBuf, name: String },
    Other { path: PathBuf, kind: recent::Kind },
    Missing { path: PathBuf },
}

/// `project.toml`, read just far enough for a card.
///
/// ponytail: a stand-in for packet M12/Y6's `project::Project::open`, which replaces it when
/// both are merged; the fields and `deny_unknown_fields` match Y6's `ProjectFile` so the two
/// accept the same files.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectFile {
    kind: String,
    name: String,
    #[allow(dead_code)]
    template: String,
}

fn project_name(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("project.toml")).ok()?;
    let file: ProjectFile = toml::from_str(&text).ok()?;
    (file.kind == "project").then_some(file.name)
}

/// A path that is gone is `Missing`; a folder with a readable `project.toml` is a `Project`;
/// anything else - a broken `project.toml` included - is what [`recent::classify`] says.
pub fn recent_cards(recent: &Recent) -> Vec<RecentCard> {
    recent
        .paths
        .iter()
        .map(|p| {
            let path = p.clone();
            if !p.exists() {
                RecentCard::Missing { path }
            } else if let Some(name) = project_name(p) {
                RecentCard::Project { path, name }
            } else {
                RecentCard::Other {
                    path,
                    kind: recent::classify(p),
                }
            }
        })
        .collect()
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
                RecentCard::Project {
                    path: good,
                    name: "Cube try 2".into()
                },
                RecentCard::Other {
                    path: run,
                    kind: recent::Kind::Run
                },
                RecentCard::Other {
                    path: broken,
                    kind: recent::Kind::Documents
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
}
