# M12 plan Y — the editor's workflow shell (S1) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Each task below is also a §1.2 work packet: its
> **Files** block is the packet's `context`, and nothing outside it may change.

**Goal:** A person with no background opens the editor, picks the cube template, presses one
button, watches the run with a traffic light, and reads a plain-language result — without a
terminal, a recipe path or a TOML file.

**Architecture:** egui stays; `egui_dock` adds Unity-style docking under a fixed step bar.
Every decision lives in a headless-tested `model/` file; `ui/` only draws. The editor still
runs no learning: it writes a run recipe into a project folder and starts `es loop cycle` as a
child it attaches to. Four small backend changes feed it: `--check-deps --json`,
`episodes.json`, `es policy init`, and the cube template's vision-route cycle recipe.

**Tech Stack:** Rust 1.85, egui/eframe 0.32.3, `egui_dock` (new, version pinned by Y10),
`rfd` 0.17.2 (present), serde/serde_json/toml, the workspace crates `es-data`, `es-eval`,
`es-render`, `es-ir`, `es-telemetry`.

**Spec:** `docs/design/editor-redesign.md` (§6 is S1). Read it with this plan; where the two
differ, the design note wins and the difference is a plan bug to report.

## Global Constraints

- MSRV `rust-version = "1.85"`; `egui`/`eframe` stay `0.32.3`. The only new dependency is
  `egui_dock` (Y10). Nothing else is added without the orchestrator.
- Layering (§4.2): `es-editor` may newly depend on `es-data` (layer 10). No same-layer
  dependency; nothing may depend on `es-editor`. `cargo xtask` layering must stay green.
- **Nothing is decided in drawing code** (`editor-shell.md` §2): every branch that chooses
  what to show lives in `crates/es-editor/src/model/` and is judged by `cargo test -p es-editor`.
- **Every visible word** goes through `crates/es-editor/i18n/en.toml` and `ko.toml`
  (`editor-shell.md` §15): same keys in both, every key used, no value outside a `*.hint`
  containing `spec `. Korean text lives nowhere else — not in tests, not in comments.
- **Label mappings are total with no wildcard arm** over `FailureKind`, `ViolationKind`,
  `PerturbationKind`, `es_env::Termination` and every new enum a label is drawn from.
- `es-ir` is not modified in S1. The Safety Plane is untouched (INV-11..13); no path runs
  without it (INV-12). No new trait (INV-17).
- Goldens are read-only (§1.4): a golden changes only through its documented generator, never
  by hand, and a task that moves code must leave every existing golden byte-identical.
- Conventional Commits, English, **no trailers**; hooks active (`core.hooksPath=.githooks`).
- Worktree agents: first `git merge --ff-only main`; **never push**; report the branch.
- Builds in the main tree use `CARGO_TARGET_DIR=target/alt` (the owner's running
  `es-editor.exe` locks `target/`).
- Per task: `cargo fmt --all -- --check`, `cargo clippy -p <crates> --all-targets -- -D warnings`,
  the touched tests, `cargo xtask check-scope`. The orchestrator runs one `cargo xtask ci` per
  merged wave.
- This machine has no `ES_PYTHON`: every oracle that needs Python must **skip with a printed
  reason**, exactly like the existing ones. Real runs are collected in Y-V, later.

## Review Focus

1. **A project folder whose path has Korean letters and spaces** (`C:\Users\User\문서\내 로봇`):
   the run recipe, `--out`, `--recipe` and the child's working directory carry it verbatim.
   Test: Y6 `run_recipe_and_argv_keep_a_korean_path_with_spaces`.
2. **Start pressed twice, or while a run is going**: one child, one run directory. Test: Y6
   `next_run_dir_never_reuses_a_number` and `start_in_while_running_is_a_no_op`.
3. **The editor reopened after it (or the run) died**: `telemetry.txt` is there but nobody
   answers — the run is *interrupted* with a resume point, never *running*. Test: Y6
   `a_dead_address_reads_as_interrupted_with_the_next_stage`.
4. **A curve with no points, a NaN first point, one point; a run the person stopped**: the
   light never panics, never shows green before data, and says *stopped by you* rather than
   *stopped* for a kill. Test: Y7 `edge_series_never_panic_and_never_start_green`,
   `a_kill_is_stopped_by_you`.
5. **An old run without `episodes.json`; a previous run with another `evaluation_hash`**: no
   tiles and bars from `report.json` for the first, no number at all for the second. Test: Y8
   `old_run_falls_back_to_suite_metrics` and `different_evaluation_hash_is_not_comparable`.

---

## Waves

| Wave | Tasks (parallel inside a wave) | Needs |
|---|---|---|
| 1 | Y1 `--check-deps --json` · Y2 `episodes.json` · Y3 raster → `es-render` · Y4 run reading → `es-eval` · Y5 cube template + `es policy init` | — |
| 2 | Y6 project + workflow · Y7 traffic light · Y8 results model · Y9 home model | wave 1 |
| 3 | Y10 docking shell | wave 2 |
| 4 | Y11 start screen · Y12 ③/④ progress · Y13 ⑤ results | Y10 |
| 5 | Y14 docs and the Y-V checklist | wave 4 |
| later | **Y-V** real runs (local venv or the server), threshold calibration | owner's go |

---

### Task Y1: `es --check-deps --json`

**Files:**
- Modify: `crates/es/src/cmd/check_deps.rs`
- Modify: `crates/es/src/main.rs` (the `--check-deps` arm only)
- Test: `crates/es/tests/cli.rs` (next to `check_deps_always_exits_zero`)
- Docs: none (the help line in `TOP_HELP` gains `[--json]`)

**Interfaces:**
- Produces (stdout, one line, exit 0 always):

```json
{"schema":1,
 "python":{"found":true,"path":"C:/…/python.exe"},
 "modules":{"mujoco":true,"torch":true,"lerobot":false},
 "vulkan_loader":true,
 "render":false,
 "backends":[{"name":"mujoco-cpu","available":true},
             {"name":"mjwarp","available":false,"reason":"…"}]}
```

`python.path` is absent when `found` is false. `render` is `cfg!(feature = "render")` of the
`es` binary itself — `--frames` needs it (`crates/es/src/cmd/eval.rs`), and the cube template
records frames, so the start screen must know.

- [ ] **Step 1: Write the failing test** in `crates/es/tests/cli.rs`:

```rust
#[test]
fn check_deps_json_is_one_object_with_every_field() {
    let out = bin().args(["--check-deps", "--json"]).output().expect("run es");
    assert_eq!(out.status.code(), Some(0));
    let text = stdout(&out);
    let v: serde_json::Value = serde_json::from_str(text.trim()).expect("one JSON object");
    assert_eq!(v["schema"], 1);
    assert!(v["python"]["found"].is_boolean());
    for m in ["mujoco", "torch", "lerobot"] {
        assert!(v["modules"][m].is_boolean(), "modules.{m}");
    }
    assert!(v["vulkan_loader"].is_boolean());
    assert!(v["render"].is_boolean());
    let names: Vec<&str> = v["backends"].as_array().unwrap().iter()
        .map(|b| b["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"mujoco-cpu"));
    for b in v["backends"].as_array().unwrap() {
        if b["available"] == false {
            assert!(b["reason"].is_string(), "an unavailable backend says why");
        }
    }
}
```

- [ ] **Step 2: Run it, expect FAIL** —
  `cargo test -p es --test cli check_deps_json_is_one_object_with_every_field` (today's text is
  not JSON).
- [ ] **Step 3: Implement.** Gather the facts once into a private `struct Deps` (python,
  mujoco, torch, lerobot, vulkan, render, backends: `Vec<(String, Result<(), String>)>`), then
  `run(json: bool)` prints either today's text (unchanged, byte for byte) or
  `serde_json::to_string(&serde_json::json!({...}))`. `main.rs`:
  `Some("--check-deps") => Ok(cmd::check_deps::run(args.iter().any(|a| a == "--json")))`.
- [ ] **Step 4: Run both** check-deps tests; expect PASS (the text test proves the text is
  unchanged).
- [ ] **Step 5: Commit** `feat(es): --check-deps --json for the editor's start screen`.

**Acceptance:** both tests pass; text output unchanged. **Forbidden:** changing the text
format; probing a GPU driver (the Vulkan check stays a file-existence heuristic).

---

### Task Y2: `episodes.json` — one row per evaluated episode

**Files:**
- Create: `crates/es-eval/src/episodes.rs`
- Modify: `crates/es-eval/src/lib.rs` (module + re-exports)
- Modify: `crates/es-eval/src/runner.rs` (extract `cell_name`; make `resolve_seeds`
  `pub(crate)`; no behaviour change)
- Modify: `crates/es/src/cmd/eval.rs` (write the file after `Evaluation::merge`)
- Test: in-module tests in `episodes.rs`; `crates/es/tests/cli.rs` (Python-gated cross-check)

**Interfaces:**
- Consumes: `runner::{Shard, ShardCell}`, `metrics::CellSummary` (`histogram`, `steps`,
  `dirty_steps`), `es_ir::evaluation::EvaluationIr`.
- Produces:

```rust
pub const EPISODES_FILE: &str = "episodes.json";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRow {
    /// `EvaluationIr::suites[cell].name`.
    pub suite: String,
    /// `<suite>-<NN>`: the key of `frames/<cell>/`, `traj/<cell>.estraj`, `events.json`.
    pub cell: String,
    /// Index into the resolved seed list (§10.2).
    pub episode: u64,
    pub seed: u64,
    /// `success` | `failure` | `timeout` | `unfinished` — the one termination bucket.
    pub termination: String,
    /// Steps the Safety Plane validated, and how many of them it changed.
    pub steps: u64,
    pub changed_steps: u64,
    /// This episode's own §10.3 `failure_mode_histogram` buckets.
    pub histogram: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeIndex {
    pub schema_version: u32, // 1
    pub episodes: Vec<EpisodeRow>,
}

/// Rows in `(cell, episode)` order, whatever order the shards arrive in.
pub fn episode_rows(ir: &EvaluationIr, shards: &[Shard]) -> Vec<EpisodeRow>;
pub fn write_episodes(rows: &[EpisodeRow], dir: &Path) -> Result<(), EvalError>;
/// `Ok(None)` when the file is absent (a run from before this packet).
pub fn read_episodes(dir: &Path) -> Result<Option<Vec<EpisodeRow>>, EvalError>;
```

and in `runner.rs`: `pub fn cell_name(suite: &str, episode: u64) -> String` (the
`format!("{}-{idx:02}", suite.name)` at `runner.rs:710`, now called from both places — first
confirm that `idx` there is the episode index and say so in the commit message).

- [ ] **Step 1: Write the failing tests** in `episodes.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn summary(termination: &str, steps: u64, changed: u64) -> metrics::CellSummary {
        let mut s = metrics::CellSummary::default();
        s.histogram.insert(termination.to_owned(), 1);
        s.steps = steps;
        s.dirty_steps = changed;
        s.n_episodes = 1;
        s
    }

    fn unit(cell: u32, episode: u64, termination: &str) -> ShardCell {
        ShardCell { cell, episode, summary: summary(termination, 100, 3) }
    }

    #[test]
    fn rows_come_out_in_cell_episode_order_with_names_and_seeds() {
        let ir = two_suite_ir(); // suites "nominal", "light_intensity"; seeds base 101, 2 episodes
        let a = Shard { cells: vec![unit(1, 0, "failure"), unit(0, 1, "success")], ..Default::default() };
        let b = Shard { cells: vec![unit(0, 0, "timeout"), unit(1, 1, "success")], ..Default::default() };
        let rows = episode_rows(&ir, &[a, b]);
        let key: Vec<_> = rows.iter().map(|r| (r.cell.as_str(), r.seed, r.termination.as_str())).collect();
        assert_eq!(key, [
            ("nominal-00", 101, "timeout"), ("nominal-01", 102, "success"),
            ("light_intensity-00", 101, "failure"), ("light_intensity-01", 102, "success"),
        ]);
        assert_eq!(rows[0].changed_steps, 3);
    }

    #[test]
    fn the_file_round_trips_and_an_absent_file_is_none() {
        let dir = std::env::temp_dir().join(format!("es-episodes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), None);
        let rows = episode_rows(&two_suite_ir(), &[Shard { cells: vec![unit(0, 0, "success")], ..Default::default() }]);
        write_episodes(&rows, &dir).unwrap();
        assert_eq!(read_episodes(&dir).unwrap(), Some(rows));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
```

`two_suite_ir()` builds a minimal `EvaluationIr` the way `runner.rs`'s own tests do (reuse their
helper if one exists; otherwise build it with `SeedPlan::Base(101)` and `n_episodes = 2`).
If `CellSummary` has no `Default`, derive it (no other change to the type).

- [ ] **Step 2: Run** `cargo test -p es-eval episodes`; expect FAIL (module missing).
- [ ] **Step 3: Implement** `episode_rows` (flatten, sort by `(cell, episode)`, suite name from
  `ir.suites[cell]`, seed from `resolve_seeds(ir)[episode]`, termination = the one key of
  `success|failure|timeout|unfinished` present in the histogram), the writer (pretty JSON +
  trailing newline, like `write_artifacts`) and the reader. In `crates/es/src/cmd/eval.rs`,
  right after `es_eval::write_artifacts(...)`:
  `es_eval::episodes::write_episodes(&es_eval::episodes::episode_rows(&eval_ir, &shards), &a.out)`.
  The shard-worker branch (`--shard-out`) returns before this and writes nothing new.
- [ ] **Step 4: Add the Python-gated cross-check** to `crates/es/tests/cli.rs`, beside an
  existing `es eval run` test that already skips without Python: after the run, for every
  suite, `count(termination == "success") / n` equals the report's `success_rate` value for
  that suite **exactly**, and the row count equals `n_episodes × suites`. Two independently
  computed numbers checking each other.
- [ ] **Step 5: Run** `cargo test -p es-eval` and `cargo test -p es --test cli eval_run` (the
  new one prints its skip reason here); expect PASS / SKIP.
- [ ] **Step 6: Commit** `feat(es-eval): episodes.json, one row per evaluated episode`.

**Acceptance:** tests pass; `report.json`, `evaluation.lock` and `events.json` byte-identical
(no existing golden or test changes). **Forbidden:** adding a field to `report.json` (it would
move every committed report); computing any metric here.

---

### Task Y3: the replay rasterizer moves to `es-render`

**Files:**
- Create: `crates/es-render/src/raster.rs`
- Modify: `crates/es-render/src/lib.rs` (`pub mod raster;`)
- Modify: `crates/es-editor/src/model/replay_view.rs`
- Modify: `crates/es-editor/src/app.rs` (imports only)

**Interfaces:**
- Moves, unchanged in behaviour, from `replay_view.rs` into `es_render::raster`: `Camera`
  (with `view`, `orbit`, `zoom`, `PITCH_LIMIT`), `Tri2d`, `Projected`, `BACKGROUND`,
  `MAX_PIXELS`, `Raster` (with `size_for`, `draw`), `project_scene`, `shading`, `flat_colour`,
  `edge`, `span`, `dot`, `vec3`, `quat_from_basis`, and the error variants they raise (a
  `RasterError` in `es-render`; `ReplayError` wraps it).
- Stays in `replay_view.rs` (it needs `es_env::traj`, layer 9): `ReplayView`, `ReplayError`,
  `panel_height` and the panel constants, `load_scene`.
- `flat_colour` must call `es_render::cpu::shade_lambert` directly instead of re-implementing
  it **if and only if** the golden stays byte-identical; if it does not, keep the copy, add a
  `ponytail:` comment naming the difference, and report it.

- [ ] **Step 1:** Run `cargo test -p es-editor replay` on `main`; record the passing test list.
- [ ] **Step 2:** Move the code and its unit tests (tests that need only the rasterizer move
  with it; tests that open an `.estraj` stay).
- [ ] **Step 3:** Run `cargo test -p es-render raster`, `cargo test -p es-editor replay`,
  `cargo xtask verify-goldens`; expect the same passing list and
  `tests/golden/editor/replay-tick0-320x180.bin`, `.json` and `replay_tick0_order.json`
  untouched.
- [ ] **Step 4: Commit** `refactor(es-render): the editor's CPU replay rasterizer moves to es-render`.

**Acceptance:** goldens byte-identical; no behaviour change. **Forbidden:** any change to
pixels, triangle order or the camera maths; a re-export shim in the editor.

---

### Task Y4: reading a finished run moves to `es-eval`

**Files:**
- Create: `crates/es-eval/src/run_dir.rs`
- Modify: `crates/es-eval/src/lib.rs`
- Modify: `crates/es-editor/src/model/run_view.rs`, `image_view.rs`, `live_run.rs`,
  `train_view.rs`, `recent.rs`, `crates/es-editor/src/app.rs` (imports and call sites only)

**Interfaces:**
- Moves to `es_eval::run_dir`: `RunError`, `CellRow`, `TickRow`, `FirstSeen`, `KindRow`
  (data only), `Timeline` (data + `kind_rows`, `buckets`), `Bucket`, `severity`,
  `decode_events`, `RunView` (renamed **`RunDir`**; `is_run_dir`, `open`, `set_frames_root`,
  `frames_root`, `cells`, `acceptance`, `sort_by`, `selected_cell`, `select`, `traj_path`,
  `timeline`, `filmstrip`, `frame`) and the private helpers.
- Moves to `es_eval::run_dir`: **`Rgb8Image`** (`{ width, height, data }`), the in-memory
  form of the frames `es-eval` writes. The editor imports it from there.
- Stays in `es-editor` as free functions, because a person reads them:
  `Timeline::heading` → `run_view::timeline_heading(&Timeline, cell)`, `KindRow::label` →
  `run_view::kind_label(&KindRow)`, `RunView::columns` → `run_view::columns(&RunDir)`.
  `run_view.rs` keeps only these and whatever else draws words.

- [ ] **Step 1:** Run `cargo test -p es-editor run_view live_run recent`; record the list.
- [ ] **Step 2:** Move the code and the tests that need no editor type.
- [ ] **Step 3:** Run `cargo test -p es-eval run_dir`, `cargo test -p es-editor`,
  `cargo xtask verify-goldens`, `cargo xtask layering`; expect PASS and no golden change.
- [ ] **Step 4: Commit** `refactor(es-eval): reading a finished run moves from the editor to es-eval`.

**Acceptance:** same tests pass; layering green (`es-eval` gains no new crate dependency).
**Forbidden:** behaviour changes; moving strings a person reads into `es-eval`.

---

### Task Y5: the cube template, `es policy init`, and the template reader

**Files:**
- Create: `templates/cube-into-bin.toml`
- Create: `tests/fixtures/visible-learning/cycle-vision.toml`
- Create: `tests/golden/train/plan-cycle-vision.txt` (only through the generator of Step 4)
- Modify: `crates/es/src/cmd/policy.rs` (`init` subcommand)
- Modify: `crates/es-data/src/training.rs` (a library function `untrained_bundle` the CLI and
  the editor share, if `es-compile`'s `PolicyBundle::build` plus existing weight
  initialisation cannot be called directly by both)
- Create: `crates/es-editor/src/model/template.rs`; Modify: `crates/es-editor/src/model/mod.rs`,
  `crates/es-editor/Cargo.toml` (`es-data` dependency)
- Test: `crates/es/tests/cli.rs`; in-module tests in `template.rs`

**Interfaces:**
- Produces, CLI:
  `es policy init --task T --observation O --learning L --deployment D [--seed N] --out X.esb`
  — deterministic bytes for the same documents and seed; exit 0/1/2 as the other `policy`
  subcommands. It exists because a fresh checkout cannot make the bundle a cycle's
  `[collect] policy` names (its Deployment IR is the plane the demonstrator runs under).
- Produces, library (callable by the editor; put it where `es` and `es-editor` can both reach):

```rust
pub fn untrained_bundle(task: &Path, observation: &Path, learning: &Path,
                        deployment: &Path, seed: u64) -> Result<Vec<u8>, DataError>;
```

- Produces, template file (`templates/cube-into-bin.toml`):

```toml
kind = "template"
id = "cube-into-bin"
name = "template.cube_into_bin.name"        # es-editor i18n key
summary = "template.cube_into_bin.summary"  # es-editor i18n key
robot = "SO-101"
method = "blocks"
needs = ["mujoco", "torch", "lerobot", "vulkan", "render"]
cycle = "tests/fixtures/visible-learning/cycle-vision.toml"
scene = "tests/fixtures/mjcf/so101_pick_place.xml"
demonstrations = 200

# The bundle `es policy init` builds for [collect] policy.
[bundle]
task        = "tests/fixtures/visible-learning/task.toml"
observation = "tests/fixtures/visible-learning/observation.toml"
learning    = "tests/fixtures/visible-learning/learning.toml"
deployment  = "tests/fixtures/visible-learning/deployment.toml"

# Training length presets: the `[run] checkpoint_at` series; the last mark is the length.
[lengths]
short  = [VALUES CHOSEN IN STEP 2]
medium = [VALUES CHOSEN IN STEP 2]
long   = [VALUES CHOSEN IN STEP 2]
```

- Produces, `model/template.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method { Blocks }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Length { Short, Medium, Long }

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Template { pub kind: String, pub id: String, pub name: String, pub summary: String,
    pub robot: String, pub method: Method, pub needs: Vec<String>, pub cycle: String,
    pub scene: String, pub demonstrations: u32, pub bundle: BundleDocs, pub lengths: Lengths }

impl Template {
    /// The `checkpoint_at` series of a preset; its last element is the run's length.
    pub fn marks(&self, length: Length) -> &[u32];
}

/// The directory holding `templates/`: the first ancestor of `exe_dir` that contains both
/// `templates/` and `Cargo.toml`. Pure, for the test; `templates_root()` calls it with
/// `std::env::current_exe()`'s parent.
pub fn find_root(exe_dir: Option<&Path>) -> Option<PathBuf>;
pub fn templates_root() -> Option<PathBuf>;

/// Every `templates/*.toml` that parses, sorted by `id`, and every one that does not, with why.
pub fn load(root: &Path) -> (Vec<Template>, Vec<(PathBuf, String)>);
```

A `ponytail:` comment on `find_root` names the ceiling: templates are found only from a
checkout (an installable package is a later item).

- [ ] **Step 1: Resolve the three open facts and write them into the commit message:**
  (a) how the V19b vision route collected — the collect bundle's documents (the `[bundle]`
  block above assumes the IR-route documents, since the policy never acts under `--expert`;
  confirm from `docs/design/visible-learning.md` / `training-recipe.md`); (b) whether
  `es-data`'s existing `init_weights` (`crates/es-data/src/training.rs:1166`) or
  `es-compile` already initialises weights deterministically — reuse it, write nothing new
  if so; (c) the LeRobot route's rule for `[run] steps` and `checkpoint_at`
  (`training.rs:467`, `:757`: `lerobot-train` saves at one `--save_freq`, so the marks are a
  multiple series).
- [ ] **Step 2: Write `cycle-vision.toml`** (the committed `cycle.toml` with `[train] recipe =
  "tests/fixtures/visible-learning/training-lerobot.toml"`, `[eval] config =
  "tests/fixtures/visible-learning/evaluation-v8.toml"`, and a header comment saying why) and
  choose the three length presets as multiple series that `training.rs` accepts, with
  `medium` equal to `training-lerobot.toml`'s own marks.
- [ ] **Step 3: Failing tests.** `crates/es/tests/cli.rs`:

```rust
#[test]
fn policy_init_is_deterministic_and_the_bundle_opens() {
    let dir = scratch_dir("policy-init");
    let args = |out: &str| vec![
        "policy".to_owned(), "init".to_owned(),
        "--task".to_owned(), "tests/fixtures/visible-learning/task.toml".to_owned(),
        "--observation".to_owned(), "tests/fixtures/visible-learning/observation.toml".to_owned(),
        "--learning".to_owned(), "tests/fixtures/visible-learning/learning.toml".to_owned(),
        "--deployment".to_owned(), "tests/fixtures/visible-learning/deployment.toml".to_owned(),
        "--out".to_owned(), dir.join(out).display().to_string(),
    ];
    for out in ["a.esb", "b.esb"] {
        let o = bin().current_dir(repo_root()).args(args(out)).output().unwrap();
        assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    }
    let (a, b) = (std::fs::read(dir.join("a.esb")).unwrap(), std::fs::read(dir.join("b.esb")).unwrap());
    assert_eq!(a, b);
    es_compile::PolicyBundle::open(&a).expect("the bundle opens");
}

#[test]
fn cycle_vision_dry_run_matches_its_golden() {
    let o = bin().current_dir(repo_root())
        .args(["loop", "cycle", "--recipe", "tests/fixtures/visible-learning/cycle-vision.toml",
               "--out", "runs/y5-dry", "--dry-run"])
        .output().unwrap();
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    assert_golden("tests/golden/train/plan-cycle-vision.txt", &stdout(&o));
}
```

  Use the helpers the file already has for the repository root, goldens and generation
  (`ES_GENERATE_GOLDENS=1`), exactly as `plan-cycle.txt`'s test does.
  `template.rs`:

```rust
#[test]
fn the_committed_templates_parse_and_name_real_files() {
    let root = find_root(Some(Path::new(env!("CARGO_MANIFEST_DIR")))).expect("a checkout");
    let (ok, bad) = load(&root);
    assert!(bad.is_empty(), "{bad:?}");
    let cube = ok.iter().find(|t| t.id == "cube-into-bin").expect("the cube template");
    for p in [&cube.cycle, &cube.scene, &cube.bundle.task, &cube.bundle.observation,
              &cube.bundle.learning, &cube.bundle.deployment] {
        assert!(root.join(p).is_file(), "{p}");
    }
    for l in [Length::Short, Length::Medium, Length::Long] {
        let m = cube.marks(l);
        assert!(!m.is_empty() && m.windows(2).all(|w| w[0] < w[1]), "{l:?}: {m:?}");
    }
    assert!(cube.marks(Length::Short).last() < cube.marks(Length::Long).last());
}

#[test]
fn find_root_walks_up_and_gives_up_cleanly() {
    let tmp = std::env::temp_dir().join(format!("es-tpl-{}", std::process::id()));
    let deep = tmp.join("target").join("debug");
    std::fs::create_dir_all(&deep).unwrap();
    assert_eq!(find_root(Some(&deep)), None);
    std::fs::create_dir_all(tmp.join("templates")).unwrap();
    std::fs::write(tmp.join("Cargo.toml"), "").unwrap();
    assert_eq!(find_root(Some(&deep)), Some(tmp.clone()));
    assert_eq!(find_root(None), None);
    std::fs::remove_dir_all(&tmp).unwrap();
}

#[test]
fn a_broken_template_is_reported_not_fatal() {
    let tmp = std::env::temp_dir().join(format!("es-tpl-bad-{}", std::process::id()));
    std::fs::create_dir_all(tmp.join("templates")).unwrap();
    std::fs::write(tmp.join("templates").join("x.toml"), "kind = \"template\"\n").unwrap();
    let (ok, bad) = load(&tmp);
    assert!(ok.is_empty());
    assert_eq!(bad.len(), 1);
    std::fs::remove_dir_all(&tmp).unwrap();
}
```

- [ ] **Step 4:** Run; expect FAIL. Implement `init`, `untrained_bundle`, `template.rs`;
  generate the golden once with `ES_GENERATE_GOLDENS=1` and read it before committing (it
  must show collect → expert-gate → train (LeRobot route) → eval (`evaluation-v8.toml`) →
  showcase).
- [ ] **Step 5:** Run the new tests and `cargo test -p es --test cli plan_cycle`; expect PASS.
- [ ] **Step 6: Commit** `feat: the cube template, es policy init, and the editor's template reader`.

**Acceptance:** the tests above; the golden reviewed by the orchestrator. **Forbidden:**
copying committed documents (their paths are hash input); changing `cycle.toml`,
`training-lerobot.toml` or any committed IR document; a committed binary bundle.

---

### Task Y6: project folder, run recipes, and the step bar's state

**Files:**
- Create: `crates/es-editor/src/model/project.rs`, `crates/es-editor/src/model/workflow.rs`
- Modify: `crates/es-editor/src/model/mod.rs`, `model/launch.rs` (`start_in`,
  `free_local_port`), `model/recent.rs` (`Kind::Project`, `classify`)
- Modify: `crates/es-editor/i18n/en.toml`, `ko.toml` (`phase.*`, `run.*` keys)

**Interfaces:**
- Consumes: `template::{Template, Length}` (Y5), `es_data::training::Cycle` (existing,
  `Serialize`/`Deserialize`), `es_data::collect::{read_loop_steps, LoopKind}`,
  `es_data::untrained_bundle` (Y5), `live_run::StageRow` (existing).
- Produces, `project.rs`:

```rust
pub const PROJECT_FILE: &str = "project.toml";
pub const RUNS_DIR: &str = "runs";
pub const RUN_RECIPE: &str = "cycle.toml";
pub const TELEMETRY_FILE: &str = "telemetry.txt";
pub const COLLECT_BUNDLE: &str = "untrained.esb";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectFile { pub kind: String, pub name: String, pub template: String }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project { pub root: PathBuf, pub file: ProjectFile }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunFolder { pub number: u32, pub path: PathBuf }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartSettings { pub demonstrations: u32, pub length: Length }

impl Project {
    /// Makes `root` (refusing one that already holds a `project.toml`), writes `project.toml`
    /// and builds `untrained.esb` from the template's `[bundle]` documents.
    pub fn create(root: &Path, name: &str, template: &Template, repo_root: &Path)
        -> Result<Self, ProjectError>;
    pub fn open(root: &Path) -> Result<Self, ProjectError>;
    pub fn is_project_dir(path: &Path) -> bool;
    /// `runs/NNN` directories, ascending; anything else under `runs/` is ignored.
    pub fn runs(&self) -> Vec<RunFolder>;
    pub fn latest_run(&self) -> Option<RunFolder>;
    /// One past the largest existing number, three digits; never an existing directory.
    pub fn next_run_dir(&self) -> PathBuf;
}

impl RunFolder {
    pub fn eval_dir(&self) -> PathBuf;          // path/eval
    pub fn report_path(&self) -> PathBuf;       // path/eval/report.json
    pub fn telemetry_addr(&self) -> Option<String>;
}

/// Writes `run/cycle.toml`: the template's recipe with `[collect] episodes`, `[collect]
/// policy` (absolute path of the project's bundle) and an inline `[train]` whose
/// `[run] checkpoint_at` is the preset and `steps` its last mark. Everything else is the
/// template's, unchanged. Returns the argv for `es` (no program name).
pub fn write_run(template: &Template, repo_root: &Path, project: &Project,
                 settings: StartSettings, run: &Path, telemetry: &str)
    -> Result<Vec<String>, ProjectError>;

/// `--from <stage>` argv for resuming `run` (same recipe, same `--out`).
pub fn resume_argv(run: &RunFolder, from: &str, telemetry: &str) -> Vec<String>;
```

- Produces, `launch.rs`:

```rust
impl LaunchModel {
    /// Starts `es` with `args` in `cwd` (the template's repository root), exactly like
    /// `start_program` otherwise; a no-op while a child runs; attach-follows-launch reads the
    /// `--telemetry` value out of `args`.
    pub fn start_in(&mut self, args: &[String], cwd: &Path);
}
/// A port the OS says is free on 127.0.0.1 now (bind `:0`, read it, drop).
pub fn free_local_port() -> u16;
```

- Produces, `workflow.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase { Scene, Teach, Train, Evaluate, Results }
impl Phase { pub const ALL: [Phase; 5]; pub fn key(self) -> &'static str; }

#[derive(Clone, Debug, PartialEq)]
pub enum PhaseState {
    Done,
    NotStarted,
    Locked,
    Running { stage: String, fraction: Option<f32> },
    Failed { stage: String, code: Option<i32> },
    Interrupted { resume_from: Option<String> },
    StoppedByYou { resume_from: Option<String> },
}

/// What disk says about one run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunFacts { pub collected: bool, pub trained: bool, pub evaluated: bool,
                      pub report: bool }
impl RunFacts { pub fn read(run: &RunFolder) -> Self; }

/// What the live connection and the child say. `None` = not attached.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveFacts {
    pub stages: Vec<(String, Option<i32>)>,   // stage name, end code once it ended
    pub fraction: Option<f32>,                // of the stage in progress
    pub child: Child,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child { Running, Exited(i32), Killed, NotOurs }

/// ③ = collect, expert-gate, train; ④ = eval, showcase. ① ② are Done for a template.
pub fn phases(run: Option<&RunFacts>, live: Option<&LiveFacts>) -> [PhaseState; 5];
```

- [ ] **Step 1: Failing tests** in `project.rs` (temp dirs; a fake `Template` built in the
  test with paths into the real fixtures under `CARGO_MANIFEST_DIR/../..`):

```rust
#[test]
fn next_run_dir_never_reuses_a_number() {
    let p = scratch_project("next");
    assert!(p.next_run_dir().ends_with("runs/001"));
    std::fs::create_dir_all(p.root.join("runs/001")).unwrap();
    std::fs::create_dir_all(p.root.join("runs/007")).unwrap();
    std::fs::create_dir_all(p.root.join("runs/notes")).unwrap();
    assert!(p.next_run_dir().ends_with("runs/008"));
    assert_eq!(p.runs().iter().map(|r| r.number).collect::<Vec<_>>(), [1, 7]);
}

#[test]
fn create_refuses_an_existing_project() {
    let p = scratch_project("twice");
    assert!(Project::create(&p.root, "again", &cube(), &repo()).is_err());
}

#[test]
fn run_recipe_overrides_only_what_the_person_chose() {
    let p = scratch_project("recipe");
    let run = p.next_run_dir();
    let argv = write_run(&cube(), &repo(), &p,
        StartSettings { demonstrations: 50, length: Length::Short }, &run, "127.0.0.1:7001").unwrap();
    let written: es_data::training::Cycle =
        toml::from_str(&std::fs::read_to_string(run.join("cycle.toml")).unwrap()).unwrap();
    let template: es_data::training::Cycle =
        toml::from_str(&std::fs::read_to_string(repo().join(&cube().cycle)).unwrap()).unwrap();
    assert_eq!(written.collect.as_ref().unwrap().episodes, 50);
    assert_eq!(written.eval, template.eval);
    assert_eq!(written.scene, template.scene);
    // the inline train's marks are the preset, and the length is its last mark
    let marks = cube().marks(Length::Short).to_vec();
    assert_eq!(inline_marks(&written), marks);
    assert_eq!(argv[..2], ["loop".to_owned(), "cycle".to_owned()]);
    assert!(argv.windows(2).any(|w| w[0] == "--telemetry" && w[1] == "127.0.0.1:7001"));
}

#[test]
fn run_recipe_and_argv_keep_a_korean_path_with_spaces() {
    // "\u{bb38}\u{c11c}" and "\u{b0b4} \u{b85c}\u{bd07}" are Korean words; written as escapes
    // because Korean text lives only in the i18n tables.
    let root = std::env::temp_dir().join(format!("es-y6-{}", std::process::id()))
        .join("\u{bb38}\u{c11c}").join("\u{b0b4} \u{b85c}\u{bd07}");
    let p = Project::create(&root, "k", &cube(), &repo()).unwrap();
    let run = p.next_run_dir();
    let argv = write_run(&cube(), &repo(), &p,
        StartSettings { demonstrations: 10, length: Length::Short }, &run, "127.0.0.1:7002").unwrap();
    let out = argv.windows(2).find(|w| w[0] == "--out").unwrap()[1].clone();
    assert_eq!(PathBuf::from(out), run);
    let recipe = argv.windows(2).find(|w| w[0] == "--recipe").unwrap()[1].clone();
    assert!(Path::new(&recipe).is_file());
}
```

  In `launch.rs`:

```rust
#[test]
fn start_in_while_running_is_a_no_op() {
    let mut m = LaunchModel::default();
    let (shell, args) = long_running_shell(); // the platform shell the existing tests use
    m.start_program(&shell, &args);
    let pid = m.pid();
    m.start_in(&["loop".into(), "cycle".into()], Path::new("."));
    assert_eq!(m.pid(), pid, "a second start while running starts nothing");
    m.kill();
}
```

  (Use the helpers the existing launch tests already have for a shell child and its pid; add a
  `pid()` accessor only if none exists.)

  In `workflow.rs` — the whole state table:

```rust
fn live(stages: &[(&str, Option<i32>)], child: Child) -> LiveFacts {
    LiveFacts { stages: stages.iter().map(|(s, c)| ((*s).to_owned(), *c)).collect(),
                fraction: Some(0.5), child }
}
use PhaseState::*;

#[test]
fn no_run_yet() {
    assert_eq!(phases(None, None), [Done, Done, NotStarted, Locked, Locked]);
}
#[test]
fn collecting_is_train_running() {
    let s = phases(Some(&RunFacts::default()), Some(&live(&[("collect", None)], Child::Running)));
    assert_eq!(s[2], Running { stage: "collect".into(), fraction: Some(0.5) });
    assert_eq!((s[3].clone(), s[4].clone()), (Locked, Locked));
}
#[test]
fn evaluating_after_training() {
    let l = live(&[("collect", Some(0)), ("expert-gate", Some(0)), ("train", Some(0)), ("eval", None)], Child::Running);
    let s = phases(Some(&RunFacts::default()), Some(&l));
    assert_eq!(s[2], Done);
    assert!(matches!(&s[3], Running { stage, .. } if stage == "eval"));
}
#[test]
fn a_stage_that_ends_badly_fails_its_phase() {
    let l = live(&[("collect", Some(0)), ("expert-gate", Some(4))], Child::Exited(4));
    let s = phases(Some(&RunFacts::default()), Some(&l));
    assert_eq!(s[2], Failed { stage: "expert-gate".into(), code: Some(4) });
    assert_eq!(s[3], Locked);
}
#[test]
fn a_child_that_dies_mid_stage_fails_that_stage() {
    let l = live(&[("collect", Some(0)), ("train", None)], Child::Exited(101));
    assert_eq!(phases(Some(&RunFacts::default()), Some(&l))[2],
               Failed { stage: "train".into(), code: Some(101) });
}
#[test]
fn a_kill_is_stopped_by_you_with_a_resume_point() {
    let l = live(&[("collect", Some(0)), ("train", None)], Child::Killed);
    let f = RunFacts { collected: true, ..Default::default() };
    assert_eq!(phases(Some(&f), Some(&l))[2], StoppedByYou { resume_from: Some("train".into()) });
}
#[test]
fn a_finished_run_on_disk() {
    let f = RunFacts { collected: true, trained: true, evaluated: true, report: true };
    assert_eq!(phases(Some(&f), None), [Done, Done, Done, Done, Done]);
}
#[test]
fn a_dead_address_reads_as_interrupted_with_the_next_stage() {
    let f = RunFacts { collected: true, trained: true, ..Default::default() };
    let s = phases(Some(&f), None);
    assert_eq!(s[2], Done);
    assert_eq!(s[3], Interrupted { resume_from: Some("eval".into()) });
    let only_collect = RunFacts { collected: true, ..Default::default() };
    assert_eq!(phases(Some(&only_collect), None)[2], Interrupted { resume_from: Some("train".into()) });
    assert_eq!(phases(Some(&RunFacts::default()), None)[2], Interrupted { resume_from: None });
}
#[test]
fn results_open_whenever_a_report_exists() {
    // showcase failed after eval wrote its report: ④ failed, ⑤ still open
    let l = live(&[("collect", Some(0)), ("train", Some(0)), ("eval", Some(0)), ("showcase", Some(1))], Child::Exited(1));
    let f = RunFacts { collected: true, trained: true, evaluated: true, report: true };
    let s = phases(Some(&f), Some(&l));
    assert_eq!(s[3], Failed { stage: "showcase".into(), code: Some(1) });
    assert_eq!(s[4], Done);
}
```

- [ ] **Step 2:** Run `cargo test -p es-editor project workflow launch`; expect FAIL.
- [ ] **Step 3:** Implement. `RunFacts::read` uses `read_loop_steps` on the run directory
  (collect → `LoopKind::Collect`, train → `LoopKind::Train`, eval → `LoopKind::Evaluate` with
  the non-expert policy; check which ledger rows the cycle writes, including the expert gate's,
  and document the mapping in a comment) and `report_path().is_file()`. The resume point is
  the first stage the facts do not cover, spelled the way `es loop cycle --from` spells it.
  `Kind::Project` joins `recent::classify`, checked **before** `RunView::is_run_dir`.
- [ ] **Step 4:** Run; expect PASS. Run the i18n and labels tests (new `phase.*` keys).
- [ ] **Step 5: Commit** `feat(es-editor): project folders, run recipes and the step bar's state`.

**Acceptance:** the tests above. **Forbidden:** drawing code; editing the template's cycle
recipe in place; writing anywhere outside the project folder except the OS temp dir in tests.

---

### Task Y7: the traffic light

**Files:**
- Create: `crates/es-editor/src/model/health.rs`; Modify: `model/mod.rs`, `i18n/en.toml`,
  `i18n/ko.toml` (`health.*` keys: one name and one advice line per verdict)

**Interfaces:**
- Consumes: nothing but plain numbers (the UI fills them from `LiveRun`, `TrainView` and
  `LaunchModel`).
- Produces:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light { Grey, Green, Amber, Red }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict { Starting, GoingWell, StoppedLearning, Slow, NotResponding, Broken,
                   Stopped, StoppedByYou }
impl Verdict {
    pub fn light(self) -> Light;
    pub fn key(self) -> &'static str;        // "health.<name>"
    pub fn advice_key(self) -> &'static str; // "health.<name>.advice"
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point { pub step: u64, pub loss: f64, pub samples_per_s: f64 }

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
    pub silence_s: f64,        // NotResponding after this long without a message
    pub warmup_points: usize,  // Slow and StoppedLearning are not judged before this
    pub slow_fraction: f64,    // latest samples/s below this x the median = Slow
    pub plateau_fraction: f64, // of total steps: no new loss minimum over this = StoppedLearning
}
/// ponytail: first guesses; Y-V calibrates them on a recorded successful run.
pub const THRESHOLDS: Thresholds = Thresholds { silence_s: 180.0, warmup_points: 20,
    slow_fraction: 1.0 / 3.0, plateau_fraction: 0.25 };

pub fn judge(input: &Input<'_>, th: &Thresholds) -> Verdict;
```

Order (the first that matches): `StoppedByYou` (killed) → `Stopped` (exit code ≠ 0, or any
stage code ≠ 0) → `Broken` (any non-finite loss) → `NotResponding` (alive, and the time since
the last message — or since start when there is none — exceeds `silence_s`) → `Starting` (no
message yet) → `Slow` → `StoppedLearning` → `GoingWell`.

- [ ] **Step 1: Failing tests** (`health.rs`):

```rust
fn base<'a>(points: &'a [Point]) -> Input<'a> {
    Input { since_start_s: 600.0, since_last_message_s: Some(1.0), child_alive: true,
            killed: false, exit_code: None, stage_codes: &[], total_steps: Some(20_000), points }
}
fn falling(n: usize) -> Vec<Point> {
    (0..n).map(|i| Point { step: (i as u64 + 1) * 100, loss: 1.0 / (i as f64 + 1.0),
                           samples_per_s: 40.0 }).collect()
}

#[test] fn a_falling_curve_is_going_well() {
    assert_eq!(judge(&base(&falling(100)), &THRESHOLDS), Verdict::GoingWell);
}
#[test] fn a_kill_is_stopped_by_you() {
    let mut i = base(&[]); i.killed = true; i.child_alive = false; i.exit_code = Some(1);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::StoppedByYou);
}
#[test] fn a_bad_exit_or_stage_code_is_stopped() {
    let mut i = base(&[]); i.child_alive = false; i.exit_code = Some(4);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::Stopped);
    let codes = [Some(0), Some(3)];
    let mut j = base(&[]); j.stage_codes = &codes;
    assert_eq!(judge(&j, &THRESHOLDS), Verdict::Stopped);
}
#[test] fn nan_is_broken_even_as_the_first_point() {
    let p = [Point { step: 1, loss: f64::NAN, samples_per_s: 1.0 }];
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Broken);
    let q = [Point { step: 1, loss: f64::INFINITY, samples_per_s: 1.0 }];
    assert_eq!(judge(&base(&q), &THRESHOLDS), Verdict::Broken);
}
#[test] fn silence_is_not_responding() {
    let mut i = base(&[]); i.since_last_message_s = Some(THRESHOLDS.silence_s + 1.0);
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::NotResponding);
    let mut j = base(&[]); j.since_last_message_s = None; j.since_start_s = THRESHOLDS.silence_s + 1.0;
    assert_eq!(judge(&j, &THRESHOLDS), Verdict::NotResponding);
}
#[test] fn edge_series_never_panic_and_never_start_green() {
    let mut i = base(&[]); i.since_last_message_s = None; i.since_start_s = 5.0;
    assert_eq!(judge(&i, &THRESHOLDS), Verdict::Starting);
    let one = [Point { step: 1, loss: 2.0, samples_per_s: 10.0 }];
    assert_eq!(judge(&base(&one), &THRESHOLDS), Verdict::GoingWell);
    let mut none_total = base(&falling(100)); none_total.total_steps = None;
    let _ = judge(&none_total, &THRESHOLDS); // must not panic
}
#[test] fn a_rate_drop_after_warmup_is_slow() {
    let mut p = falling(60);
    p.last_mut().unwrap().samples_per_s = 5.0; // median 40, 5 < 40/3
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::Slow);
    let mut early = falling(5);
    early.last_mut().unwrap().samples_per_s = 5.0; // before warmup: not judged
    assert_eq!(judge(&base(&early), &THRESHOLDS), Verdict::GoingWell);
}
#[test] fn no_new_minimum_over_a_quarter_of_the_run_is_stopped_learning() {
    let mut p = falling(30); // steps 100..3000, minimum at step 3000
    let last = p.last().unwrap().loss;
    for k in 1..=60 { // 6000 more steps (> 0.25 * 20_000) without going below `last`
        p.push(Point { step: 3000 + k * 100, loss: last * 1.05, samples_per_s: 40.0 });
    }
    assert_eq!(judge(&base(&p), &THRESHOLDS), Verdict::StoppedLearning);
}
```

- [ ] **Step 2:** Run `cargo test -p es-editor health`; expect FAIL.
- [ ] **Step 3:** Implement `judge` (median over the points after warm-up; "no new minimum"
  = the minimum of the points with `step > last_step - plateau_fraction * total` is not below
  the minimum of the points before them; with `total_steps == None`, skip that rule).
- [ ] **Step 4:** Run; expect PASS; run the i18n test.
- [ ] **Step 5: Commit** `feat(es-editor): the traffic light of a running cycle`.

**Acceptance:** tests pass. **Forbidden:** reading telemetry or a clock inside `judge` (it is
a pure function); claiming the thresholds are measured.

---

### Task Y8: the results model

**Files:**
- Create: `crates/es-editor/src/model/results.rs`; Modify: `model/mod.rs`, `model/labels.rs`
  (`cause_key`, `perturbation_key`, totality test), `i18n/en.toml`, `ko.toml` (`results.*`,
  `cause.*`, `cause.*.advice`, `perturb.*`)

**Interfaces:**
- Consumes: `es_eval::episodes::EpisodeRow` (Y2), `es_eval::run_dir::RunDir` (Y4),
  `es_ir::evaluation::{EvaluationReport, EvaluationIr, PerturbationKind, MetricValue,
  MetricSpec}`, `es_safety::ViolationKind`, `es_core::FailureKind` (check the path).
- Produces:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cause { Timeout, FailureCondition, Unfinished, SafetyLimit, SafetyFallback,
                 MotionGaps, TooLate, Unstable, SensorDrop, ActuatorFault, BackendUnsupported }

/// A histogram bucket's cause. `success` is `None`. Total over every name `es-eval` writes:
/// the four terminations, every `FailureKind`, `fallback`, every `ViolationKind` — the match
/// arms name each one; an unknown name is `None` and is shown raw.
pub fn cause_of(bucket: &str) -> Option<Cause>;

#[derive(Clone, Debug, PartialEq)]
pub struct Card { pub passed: bool, pub successes: u32, pub episodes: u32 }
pub fn card(report: &EvaluationReport, rows: Option<&[EpisodeRow]>) -> Card;

#[derive(Clone, Debug, PartialEq)]
pub enum Comparison { NoPrevious, NotComparable, Delta { success_points: f64 } }
/// Success fraction difference in percentage points, only for an equal `evaluation_hash`.
pub fn compare(current: &EvaluationReport, previous: Option<&EvaluationReport>) -> Comparison;

/// Failed episodes per cause, most first, then `Cause` order; an episode with two causes
/// counts once under each.
pub fn causes(rows: &[EpisodeRow]) -> Vec<(Cause, u32)>;

#[derive(Clone, Debug, PartialEq)]
pub struct Situation { pub suite: String, pub kinds: Vec<PerturbationKind>,
                       pub successes: u32, pub episodes: u32 }
/// Per suite in the evaluation's own order. With rows: counted from them. Without (an old
/// run): from the report's `success_rate` value x `n_episodes`, rounded. `kinds` empty =
/// nominal; `ir == None` leaves `kinds` empty and the UI shows the raw suite name.
pub fn situations(report: &EvaluationReport, rows: Option<&[EpisodeRow]>,
                  ir: Option<&EvaluationIr>) -> Vec<Situation>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFilter { All, Successes, Failures }
#[derive(Clone, Debug, PartialEq)]
pub struct Tile { pub cell: String, pub suite: String, pub success: bool,
                  pub cause: Option<Cause> }
pub fn tiles(rows: &[EpisodeRow], filter: TileFilter) -> Vec<Tile>;
```

`labels.rs` gains `cause_key(Cause) -> &'static str`, `cause_advice_key(Cause)`, and
`perturbation_key(PerturbationKind) -> &'static str`, each a match with one arm per variant.

- [ ] **Step 1: Failing tests** (`results.rs`):

```rust
fn row(suite: &str, ep: u64, termination: &str, extra: &[(&str, u64)]) -> EpisodeRow {
    let mut histogram: BTreeMap<String, u64> = [(termination.to_owned(), 1)].into();
    for (k, n) in extra { histogram.insert((*k).to_owned(), *n); }
    EpisodeRow { suite: suite.into(), cell: format!("{suite}-{ep:02}"), episode: ep,
                 seed: 100 + ep, termination: termination.into(), steps: 10,
                 changed_steps: 0, histogram }
}

#[test] fn every_bucket_es_eval_writes_has_a_cause_or_is_success() {
    assert_eq!(cause_of("success"), None);
    for t in ["failure", "timeout", "unfinished", "fallback"] { assert!(cause_of(t).is_some(), "{t}"); }
    for k in es_safety::ViolationKind::ALL { assert!(cause_of(violation_bucket(k)).is_some(), "{k:?}"); }
    for f in FAILURE_KINDS { assert!(cause_of(f).is_some(), "{f}"); }
}
#[test] fn causes_count_failed_episodes_most_first() {
    let rows = [row("nominal", 0, "timeout", &[]), row("nominal", 1, "failure", &[("fallback", 2)]),
                row("nominal", 2, "timeout", &[]), row("nominal", 3, "success", &[])];
    assert_eq!(causes(&rows), [(Cause::Timeout, 2), (Cause::FailureCondition, 1), (Cause::SafetyFallback, 1)]);
}
#[test] fn different_evaluation_hash_is_not_comparable() {
    let (a, mut b) = (report_with(0.5, "aa"), report_with(0.25, "aa"));
    assert_eq!(compare(&a, Some(&b)), Comparison::Delta { success_points: 25.0 });
    b.evaluation_hash = hash("bb");
    assert_eq!(compare(&a, Some(&b)), Comparison::NotComparable);
    assert_eq!(compare(&a, None), Comparison::NoPrevious);
}
#[test] fn old_run_falls_back_to_suite_metrics() {
    let report = report_with(0.75, "aa"); // one suite "nominal", n_episodes 4
    let s = situations(&report, None, None);
    assert_eq!((s[0].successes, s[0].episodes), (3, 4));
    assert!(tiles(&[], TileFilter::All).is_empty());
    assert_eq!(card(&report, None), Card { passed: report.passed, successes: 3, episodes: 4 });
}
#[test] fn tiles_filter_and_name_their_cause() {
    let rows = [row("nominal", 0, "success", &[]), row("nominal", 1, "timeout", &[])];
    assert_eq!(tiles(&rows, TileFilter::Failures).len(), 1);
    assert_eq!(tiles(&rows, TileFilter::Failures)[0].cause, Some(Cause::Timeout));
    assert_eq!(tiles(&rows, TileFilter::Successes)[0].cell, "nominal-00");
}
```

  `violation_bucket` and `FAILURE_KINDS` must be the **names `es-eval` writes**: make
  `metrics::violation_name` / `failure_name` public (or re-export them) rather than retyping
  strings. `report_with(success_rate, hash_hex)` builds an `EvaluationReport` with one suite
  `nominal`, `n_episodes: 4`, a `success_rate` cell, and `passed: true`.
  In `labels.rs`, extend `metric_and_launch_labels_are_total` (or add a sibling) so every
  `Cause` and every `PerturbationKind` has a distinct label in both languages, no `_`, never the
  raw name.
- [ ] **Step 2:** Run `cargo test -p es-editor results labels`; expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expect PASS; run the i18n test.
- [ ] **Step 5: Commit** `feat(es-editor): the results model — verdict, causes, situations, tiles`.

**Acceptance:** tests pass. **Forbidden:** a wildcard arm in `cause_of`'s known names,
`cause_key` or `perturbation_key`; a comparison number across different `evaluation_hash`es;
recomputing any §10.1 metric (the report's are the numbers).

---

### Task Y9: the home model

**Files:**
- Create: `crates/es-editor/src/model/home.rs`; Modify: `model/mod.rs`, `i18n/en.toml`,
  `ko.toml` (`home.*`, `deps.*`)

**Interfaces:**
- Consumes: Y1's JSON; `template::{Template, templates_root, load}` (Y5);
  `project::Project` and `workflow::{RunFacts, phases}` (Y6) for recent-project cards.
- Produces:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Deps { pub schema: u32, pub python: Python, pub modules: Modules,
                  pub vulkan_loader: bool, pub render: bool, pub backends: Vec<Backend> }
// Python { found: bool, path: Option<String> }, Modules { mujoco, torch, lerobot: bool },
// Backend { name: String, available: bool, reason: Option<String> }

pub fn parse_deps(json: &str) -> Result<Deps, String>;

/// `es --check-deps --json` on its own thread (the probe can take ~30 s).
pub struct DepsProbe { /* receiver */ }
#[derive(Clone, Debug, PartialEq)]
pub enum DepsState { Checking, Ready(Deps), Failed(String) }
impl DepsProbe { pub fn start(es: &Path) -> Self; pub fn poll(&mut self) -> &DepsState; }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Availability { Ready, Missing(Vec<String>), Unknown }
/// `needs` names: "mujoco" | "torch" | "lerobot" (modules), "vulkan" (loader), "render".
pub fn availability(template: &Template, deps: Option<&Deps>) -> Availability;

/// One recent path, as the start screen lists it.
#[derive(Clone, Debug, PartialEq)]
pub enum RecentCard {
    Project { path: PathBuf, name: String, phases: [PhaseState; 5] },
    Other { path: PathBuf, kind: recent::Kind },
    Missing { path: PathBuf },
}
pub fn recent_cards(recent: &Recent) -> Vec<RecentCard>;
```

- [ ] **Step 1: Failing tests** (`home.rs`):

```rust
const READY: &str = r#"{"schema":1,"python":{"found":true,"path":"p"},
  "modules":{"mujoco":true,"torch":true,"lerobot":true},"vulkan_loader":true,"render":true,
  "backends":[{"name":"mujoco-cpu","available":true}]}"#;

#[test] fn ready_deps_make_the_cube_template_ready() {
    let d = parse_deps(READY).unwrap();
    assert_eq!(availability(&cube(), Some(&d)), Availability::Ready);
    assert_eq!(availability(&cube(), None), Availability::Unknown);
}
#[test] fn a_missing_module_or_render_names_what_is_missing() {
    let mut d = parse_deps(READY).unwrap();
    d.modules.lerobot = false; d.render = false;
    assert_eq!(availability(&cube(), Some(&d)),
               Availability::Missing(vec!["lerobot".into(), "render".into()]));
}
#[test] fn garbage_is_an_error_not_a_panic() {
    assert!(parse_deps("es --check-deps (spec 2.5)").is_err());
}
#[test] fn recent_cards_mark_what_is_gone() {
    let r = Recent { paths: vec![PathBuf::from("Z:/nowhere/at/all")] };
    assert!(matches!(recent_cards(&r)[0], RecentCard::Missing { .. }));
}
```

- [ ] **Step 2:** Run `cargo test -p es-editor home`; expect FAIL.
- [ ] **Step 3:** Implement (`DepsProbe` spawns `es --check-deps --json` with `Command`,
  reads stdout on a thread, sends one result through `mpsc`; `poll` never blocks).
- [ ] **Step 4:** Run; expect PASS; i18n test.
- [ ] **Step 5: Commit** `feat(es-editor): the start screen's model — PC check, templates, recent projects`.

**Acceptance:** tests pass. **Forbidden:** running the probe on the UI thread; hiding a
template that is not ready (it is shown disabled with what is missing).

---

### Task Y10: the docking shell

**Files:**
- Modify: `Cargo.toml` (workspace dependency `egui_dock`), `crates/es-editor/Cargo.toml`
- Create: `crates/es-editor/src/ui/mod.rs`, `ui/shell.rs`, `ui/advanced.rs`
- Modify: `crates/es-editor/src/app.rs` (becomes the `eframe::App` glue: state, `update`
  dispatch, storage), `src/lib.rs`, `src/main.rs`, `model/recent.rs` (`LAYOUT_KEY`),
  `i18n/en.toml`, `ko.toml` (`shell.*`, `menu.*`, `advanced.*`)
- Create: `crates/es-editor/src/model/layout.rs` (which dock tabs exist per phase; tested)

**Interfaces:**
- Consumes: `workflow::{Phase, PhaseState, phases}` (Y6); every existing tab function.
- Produces:
  - `model/layout.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pane { Viewport, StepPanel, Summary, Console,
                AdvancedGraph, AdvancedSees, AdvancedProblems, AdvancedMetrics, AdvancedLive }
impl Pane { pub const ALL: [Pane; 9]; pub fn key(self) -> &'static str; }
/// The default arrangement for a phase: (left, centre, right, bottom) panes.
pub fn default_layout(phase: Phase) -> Layout;
```

  - `ui/shell.rs`: `pub fn draw(app: &mut EditorApp, ctx: &egui::Context)` — menu bar (File:
    new project from template, open project, open file, open run folder, attach, recent; View:
    reset layout, language, text size), the step bar (one button per `Phase`, coloured from
    `PhaseState`, the `Next ▶` button and the "what is left" text), the `egui_dock` area, the
    status line.
  - `ui/advanced.rs`: today's `graph_tab`, `run_tab`, `telemetry_tab`, `images_tab`,
    `diagnostics_tab` moved out of `app.rs` unchanged, each drawn as one dock tab.
  - The old top tab bar and the old home screen are deleted; `labels::Tab`'s role is taken by
    `Pane` (delete `Tab` and its tests once nothing uses it; `labels::Step` goes with the old
    home in Y11).
  - Opening a non-project path (a bundle, documents, a run folder) shows the dock with the
    Advanced panes and the step bar disabled, with `shell.not_a_project`.

- [ ] **Step 1: Pin the crate.** Find the newest `egui_dock` whose `egui` requirement is
  `0.32` and whose MSRV ≤ 1.85 (`cargo info egui_dock@<v>`, its `Cargo.toml`); add it to the
  workspace with `default-features = false` plus `serde` if needed for layout persistence.
  **If none exists**, stop and report: the fallback (egui `SidePanel`s, no drag-to-re-dock) is
  a scope change the orchestrator takes to the owner.
- [ ] **Step 2: Failing test** (`layout.rs`):

```rust
#[test] fn every_phase_has_a_layout_with_the_viewport_in_the_centre() {
    for p in Phase::ALL {
        let l = default_layout(p);
        assert_eq!(l.centre, vec![Pane::Viewport], "{p:?}");
        assert!(l.left.contains(&Pane::StepPanel));
    }
}
#[test] fn every_pane_has_a_label_in_both_languages() {
    for p in Pane::ALL {
        for lang in [Lang::En, Lang::Ko] {
            let s = t(lang, p.key());
            assert!(!s.is_empty() && s != p.key() && !s.contains('_'), "{p:?} {lang:?}");
        }
    }
}
```

- [ ] **Step 3:** Run; expect FAIL; implement `layout.rs`; run; expect PASS.
- [ ] **Step 4:** Move the tab functions into `ui/advanced.rs`; build the shell; persist the
  dock state under `recent::LAYOUT_KEY` (`"es-editor.layout"`), falling back to the phase
  default when absent or unreadable.
- [ ] **Step 5:** `cargo build -p es-editor` (target/alt), `cargo clippy -p es-editor
  --all-targets -- -D warnings`, `cargo test -p es-editor`; then run it by hand and take a
  screenshot of: the empty shell, a bundle opened (Advanced panes), each phase button.
- [ ] **Step 6: Commit** `feat(es-editor): the docking shell and the step bar`.

**Acceptance:** tests pass; screenshots attached to the report; every existing feature still
reachable from an Advanced pane. **Forbidden:** decisions in `ui/`; removing a feature;
another dependency.

---

### Task Y11: the start screen

**Files:**
- Create: `crates/es-editor/src/ui/home.rs`; Modify: `ui/mod.rs`, `ui/shell.rs` (dispatch),
  `app.rs` (the `DepsProbe`, the new-project dialog state), `model/labels.rs` (delete `Step`
  and `home_screen_lists_the_five_steps_in_loop_order`), `model/dialogs.rs` (a folder picker
  for the new project, if not already there), `i18n/*.toml`

**Interfaces:**
- Consumes: `home::{DepsProbe, DepsState, availability, recent_cards}`, `template::load`,
  `project::Project::create`, `recent::Recent`.
- Produces: `pub fn draw(app: &mut EditorApp, ui: &mut egui::Ui)`; full-window when no
  project or file is open.

Layout (design note §3): the PC-check line (one item per `needs` name in plain words, green /
red, optional backends amber, "checking…" while `Checking`, the failure text while `Failed`);
"What would you like to do?" with one card per loaded template (disabled with
`home.missing` + the missing items when `Missing`) — and nothing for S3–S5's cards; recent
projects with their miniature step bars; the file-level ways in. A template card opens a
small dialog: project name (default `template` name), folder (default
`<Documents>/Electric Sheep/<name>`; the OS folder picker through `dialogs`), Create. Creating
opens the project at ③ (① and ② are `Done`). A template that failed to parse is listed under
the cards with its path and reason.

- [ ] **Step 1:** Implement; `cargo test -p es-editor` (i18n completeness catches every new
  key); clippy.
- [ ] **Step 2:** By hand, on this machine (no Python): the PC check shows Python, the
  learning tools and MuJoCo as missing; the cube card is disabled and lists them; a recent
  path that no longer exists shows as gone. Creating a project is exercised by Y6's tests
  (including the Korean path with a space), not by hand here. Screenshots.
- [ ] **Step 3: Commit** `feat(es-editor): the start screen`.

**Acceptance:** screenshots; tests pass. **Forbidden:** a card for a template or feature that
does not run end to end.

---

### Task Y12: ③ and ④ — starting, watching, stopping

**Files:**
- Create: `crates/es-editor/src/ui/train.rs`; Modify: `ui/mod.rs`, `ui/shell.rs`, `app.rs`
  (the run controller: which run, its `LaunchModel`, its `LiveRun`/`TrainView`, the start
  instant, the done-notification latch), `i18n/*.toml`

**Interfaces:**
- Consumes: `project::{write_run, resume_argv, RunFolder}`, `launch::{LaunchModel::start_in,
  free_local_port}`, `workflow::phases`, `health::{judge, Input, THRESHOLDS}`,
  `live_run::LiveRun`, `train_view::TrainView`, `labels::stage_label`.
- Produces: `pub fn draw_train(app, ui)` and `pub fn draw_evaluate(app, ui)` for the step
  panel, centre and summary panes.

Behaviour (design note §6.4):
1. **Settings** before a run: demonstrations (default `template.demonstrations`), length
   (short / medium / long). **Start** → `next_run_dir`, `write_run` with
   `127.0.0.1:{free_local_port()}`, write `telemetry.txt`, `start_in(argv, repo_root)`.
2. **Stage cards** from `LiveRun::stages()`, plain names via `labels::stage_label`; progress
   = `episode.end` count ÷ demonstrations while collecting, step ÷ `TrainView::total()` while
   training; time left only from `TrainView::eta()`.
3. **Centre**: while collecting, the newest image-stream frame and the success counter from
   `episode.end` outcomes; while training, the sample image and the loss curve (the existing
   `paint_curve`).
4. **Right**: the light (`judge` with the numbers above) with its name and advice line; the
   curve; samples/s; GPU memory only when a metric row carries it.
5. **Stop** → `LaunchModel::kill`. **Evaluate what it has learned so far** → kill; set the
   run's `cycle.toml` `[eval] checkpoint` to the newest mark the ledger or a `checkpoint`
   event reports; `start_in(resume_argv(run, "eval", addr), repo_root)`. (Whether `--from`
   accepts that is Y-V item 3; until then the button is present and its failure is shown by
   the light.)
6. **Done**: the first frame on which ④ becomes `Done` sends
   `egui::ViewportCommand::RequestUserAttention(egui::UserAttentionType::Informational)` once
   and shows `train.done`; if the current phase is ③ or ④, switch to ⑤.
7. **Re-opened**: dial `telemetry.txt`'s address once; on failure the phases read
   *interrupted* and the panel offers [Resume] (`resume_argv` with the resume point) and
   [Start over] (a new run).
8. **④** is the same screen with the eval stages, plus a tile strip filled from
   `cell.end` rows of `LiveRun` as they arrive.

- [ ] **Step 1:** Implement; tests + clippy.
- [ ] **Step 2:** By hand without Python: Start → the child fails at the first stage that
  needs Python → the light turns *stopped* with the stage's plain name and the console holds
  `es`'s own message; Stop on a running child shows *stopped by you*. Screenshots.
- [ ] **Step 3: Commit** `feat(es-editor): starting and watching a run`.

**Acceptance:** screenshots of the failure path and the kill path. **Forbidden:** a whole-run
ETA; blocking the UI thread on a child, a socket or a file larger than a recipe.

---

### Task Y13: ⑤ — results

**Files:**
- Create: `crates/es-editor/src/ui/results.rs`; Modify: `ui/mod.rs`, `ui/shell.rs`,
  `app.rs`, `model/dialogs.rs` (`save_file(default_name, filter)`), `i18n/*.toml`

**Interfaces:**
- Consumes: `results::{card, compare, causes, situations, tiles, Tile, TileFilter}`,
  `es_eval::episodes::read_episodes`, `es_eval::run_dir::RunDir`, `replay_view::ReplayView`,
  `es_render::raster::{Camera, Raster}`, `labels::{cause_key, cause_advice_key,
  perturbation_key, metric_label}`, the run's Evaluation IR (from its `cycle.toml`
  `[eval] config`, repository-relative).
- Produces: `pub fn draw(app, ui)`.

Behaviour (design note §6.5): the run list; the card (pass/fail, "x of n", each acceptance line
in plain words with ✓/✗, the comparison with the previous run of the project or
`results.not_comparable`); causes with advice; the player (outside orbit camera / the policy's
eye from `frames/<cell>/` / side by side; 0.5×/1×/2×; the existing timeline marks); situation
bars (label = the joined `perturbation_key`s, `results.nominal` when none, the raw suite name
on hover); tiles with the filter; folded: the existing metrics table and the hashes; buttons:
export the policy (the newest checkpoint bundle, `dialogs::save_file`), run again (③ with the
same settings). An old run without `episodes.json` shows `results.old_run` where the tiles
would be.

- [ ] **Step 1:** Implement; tests + clippy.
- [ ] **Step 2:** By hand: open `tests/fixtures/visible-learning/run` (the committed E5 fixture
  run) inside a scratch project copy → the old-run path; then the same with a hand-written
  `episodes.json` beside it in the scratch copy (never in the fixture) → tiles, causes, bars.
  Screenshots.
- [ ] **Step 3: Commit** `feat(es-editor): the results screen`.

**Acceptance:** screenshots of both paths. **Forbidden:** modifying the committed fixture run.

---

### Task Y14: documents, and the checklist for later

**Files:**
- Modify: `docs/design/editor-shell.md` + `.ko.md` (new §17: what S1 changed, pointing at
  `editor-redesign.md`), `docs/design/editor-redesign.md` + `.ko.md` (status line; anything
  the implementation settled — the length presets, the ledger mapping, the `egui_dock`
  version), `CLAUDE.md` (project status: M12 plan Y implemented, Y-V pending)
- Create: `docs/packets/M12/YV-verification.md` + `.ko.md`

The Y-V checklist, each item with its exact command, where it may run (this PC with a venv,
or the server — the owner decides), and what counts as pass:

1. `es loop cycle --recipe runs/…/cycle.toml` of the cube template runs end to end on the
   LeRobot route; numbers recorded (held-out success, envelope rate), V19b beside them.
2. `episodes.json`'s success count equals `success_rate` on that run (Y2's gated test, run
   for real).
3. `--from eval` after `[eval] checkpoint` changes (the "evaluate so far" button).
4. The traffic light: record that run's progress frames to a fixture, add
   `a_recorded_successful_run_is_green_throughout` to `health.rs`, and tune `THRESHOLDS`
   until it passes without weakening the synthetic tests.
5. The length presets' wall-clock times on the machine used, for the start screen's estimate.
6. The whole flow by hand on Windows: start → template → Start → watch → results, screenshots.

- [ ] **Step 1:** Write the documents (the Korean siblings may be delegated to a Sonnet
  agent, then reviewed).
- [ ] **Step 2:** `cargo xtask check-spec-refs`; the pre-commit pairing check.
- [ ] **Step 3: Commit** `docs: the editor's workflow shell (M12 plan Y) and its verification list`.

---

## Self-review (done while writing)

- **Spec coverage:** §6.1 shell/dock/code layout → Y10; moves → Y3, Y4; §6.2 project/template
  → Y5, Y6; §6.3 step state → Y6; §6.4 progress/light → Y7, Y12; §6.5 results → Y2, Y8, Y13;
  §6.6 backend → Y1, Y2, Y5; §6.7 errors → Y6 (interrupted), Y7 (stopped/stopped by you),
  Y8 (old run), Y9 (missing deps, broken template), Y13 (no report); §6.8 oracles → each
  task's tests + Y-V; §8's five open facts → Y5 step 1, Y10 step 1, Y-V 1–5.
- **Placeholders:** the length presets are chosen in Y5 step 2 from `training.rs`'s rule;
  `THRESHOLDS` are declared first guesses with the calibration task named (Y-V 4).
- **Type names:** `es_eval::run_dir::RunDir` (a finished evaluation's files, Y4) and
  `project::RunFolder` (a numbered `runs/NNN` folder of a project, Y6) are different things
  with deliberately different names; `RunFolder::eval_dir()` is what a `RunDir` opens.
