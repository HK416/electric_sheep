# Editor redesign — a workflow shell for people who are not experts

Design note for `crates/es-editor` (layer 12) and the small backend changes it needs. Spec:
§23.1 (the editor is a client), §23.3 (watching a run), §13.1 and §13.3 (the loop and its
fixed evaluation), §10 (Evaluation IR), §28.9 (the expert gate), §1.4 (oracle first), §1.5
(context budget), §4.2 (layering). It builds on `editor-shell.md` §10–§16, which it
reorganises rather than replaces.

Status: agreed with the owner on 2026-09-28 (screens and decomposition). **S1 implemented the same
day** (plan Y, `docs/packets/M12/plan-y.md`, Y1–Y15; review `docs/reviews/M12.md`); not closed until the
real runs of `docs/packets/M12/YV-verification.md`. S2–S6 are directions, each to get its own
design pass before it is planned; S2 waits on the `es-editor` split (section 9).

## 1. Why

Owner directive, 2026-09-28: redesign the editor completely.

1. A layout that someone with no background can approach, the way people approach Unity or
   Unreal.
2. Anyone can carry out (vision-based) robot learning.
3. Anyone can see, easily, how a learning run is going.
4. Anyone can see, easily, what a learning run produced.

`editor-shell.md` §15 already replaced the jargon with plain words. What it did not change is
the *shape*: five engineer tabs, and a launch panel that asks for a recipe path. A person who
does not know that a recipe exists cannot start. This redesign changes the shape.

## 2. Decisions taken with the owner

| Question | Decision |
|---|---|
| Where does a beginner start? | **Both**: templates, and a project built from scratch |
| Where do a new task's demonstrations come from? | **Three ways**: action blocks, mouse/gamepad teleoperation in the viewport, reward-only RL (a physical leader arm is out) |
| Where do runs execute? | **This PC first**; a remote server is a later sub-project (attaching to an address already works) |
| Technology | **egui kept**, docking added (`egui_dock`); no web UI, no Bevy |
| Overall layout | **Step bar + docking** (option B): Unity-style panels, with a fixed step bar that says where you are and what comes next |
| After picking a template | Opens with ① and ② already done; "Start training" is one click away |
| Randomization UI | One strength setting (low / medium / high) **plus** a 🎲 per property |
| When are demonstrations produced in bulk? | In ③, together with training, as one button; ② only defines the method and checks it |
| What the viewport shows during training | **Checkpoint previews**: a few test episodes per checkpoint, with video (new) |
| How much the results screen guides | One line of advice per failure cause, plus **"Train again, focused on failures"** |
| Decomposition | S1 → S2 → S3 → S4 → S5 → S6, S1 first |
| Crate budget | Move code the editor holds on others' behalf back to its owners (section 6.1) |

## 3. The screens (all sub-projects)

The shell has three fixed parts — a **step bar** on top, a **dock area**, a **status line** at the
bottom — and the start screen, which is full-window.

```
[① Scene] [② Teach] [③ Train 62%] [④ Evaluate] [⑤ Results]      ⚠ 1 thing left   [Next ▶]
┌ step panel ──┐┌ viewport (always centre) ──────────┐┌ inspector / summary ─┐
│ changes with ││                    ┌─────────────┐ ││ changes with the    │
│ the step     ││                    │ what the    │ ││ step                │
│              ││                    │ policy sees │ ││                     │
└──────────────┘└────────────────────┴─────────────┘┘└─────────────────────┘
 status: what is running · time left · problems                (console, collapsed)
```

Panels can be dragged and re-docked; each step has a default arrangement and the person's
changes are remembered. The five tabs of today (Design graph, Results table, Live, What the
policy sees, Problems) survive as **Advanced** dock tabs — nothing is removed.

**Start screen.** A "check this PC" line (GPU, Python learning tools, MuJoCo, optional ones in
amber); "What would you like to do?" with template cards and an empty-project card; recent
projects, each with its own miniature step bar ("③ training 62%", "✓ done · 56%"); and the
file-level ways in (open a file, open a run folder, attach to a running run). **Only cards that
actually run end to end are shown**; a template or the empty project appears when its
sub-project lands.

**① Scene.** Left: what is in the scene (robot, objects, camera, lights) and an Add list
(robot: SO-101 or a URDF/MJCF/USD file; objects: primitives or a mesh file; camera; light).
Centre: 3D viewport with move / rotate / scale gizmos and a physics preview, and, always in its
corner, **what the policy's camera sees** at the policy's resolution. Right: the selected
thing's properties with 🎲 toggles, and a **sentence editor** for success and failure
("[cube] is [inside] [bin] for [1 s]"; "fails if not done within [36 s]"). Bottom: plain-word
checks ("the camera sees only half of the bin").

**② Teach.** Three tabs in the step panel:
- *Action blocks*: object-relative blocks ("move above [cube] by [5 cm]", "close gripper",
  "move above [bin]"), a timeline, preview, and 🎲 "try another position".
- *Teleoperate*: drag the end effector (IK) with mouse or gamepad, record; each recording is
  judged by ①'s success condition; a count against a target ("11 of 50 successful").
- *Reward*: check boxes for what earns points ("success, a lot", "hand closer to [cube], a
  little"), and one sentence on how it learns: a *teacher* that knows object positions
  practises first; a *student* that sees only the camera learns from the teacher.

All three end in **"Check first"**: run the demonstrator a few times and show how often it
succeeds, with a link to each failure. ③ is locked until it passes (§28.9 rule 1, in plain
words).

**③ Train.** One button runs *make data → (teacher practice) → train*. Stage cards with
progress and time left; the centre shows checkpoint previews (S2) and what the policy sees;
the right shows the **traffic light** and the loss curve (dots on the curve open that
checkpoint's preview); Stop, and "Evaluate what it has learned so far".

**④ Evaluate.** The same progress screen; the episode tiles of ⑤ fill in as they finish.
When done: a notification, and the editor moves to ⑤.

**⑤ Results.** A verdict card (pass / fail, "9 of 16", each acceptance line in plain words,
the change from the previous run — or "evaluation conditions differ, not comparable"); why it
failed, with one line of advice per cause; a player (outside camera / the policy's eye / side
by side, event marks on its timeline); success per situation; every test episode as a tile;
the detailed metrics and the hashes, folded. Buttons: export the policy, run again, **train
again focused on failures** (S2).

## 4. What the code has today, and what is missing

| Screen | Present | Missing |
|---|---|---|
| Shell / start | `es --check-deps` (text); recent list; plain labels and i18n | a docking crate (egui 0.32.3 has none here); `--check-deps --json`; **any notion of a project** |
| ③ Train | stage, progress, total steps, ETA, loss, sample image, checkpoint events — all on the wire (`editor-shell.md` §13, §16) | the traffic-light rules; **checkpoint previews** (no evaluation runs during training) |
| ⑤ Results | verdict, acceptance, 18 metrics, CPU 3D replay, timeline marks | **per-episode outcome on disk** (it exists only on the wire, `cell.end`); which failure predicate fired; no failure set |
| ① Scene | CPU raster + orbit camera (`replay_view`); `scene_to_mjcf`; position / colour / light / camera randomization (§6.3, M11) | picking and gizmos; a generator from scene to the five documents (only `#[ignore]`d test generators today); a one-frame camera render CLI; a "held for" node; contact conditions (`GetContact` is not lowered by `es-env`) |
| ② Action blocks | the expert gate of `es loop cycle` | the only demonstrator is `so101-pick-place` (`es-env/src/expert.rs`) with a closed-form SO-101 IK; no data-driven waypoint program |
| ② Reward | PPO (`[rl]`); `es loop collect --policy <bundle> --frames` renders images from a state policy's rollout | success-only episode filtering; a teacher→student recipe; collision penalties |
| ② Teleoperate | — | an interactive simulation session (MuJoCo runs as a Python subprocess); a command channel from the editor to that session |

Two facts that shape S1:

- **No camera-only route has passed on the cube task yet.** *(Corrected 2026-09-28 by packet
  Y5; an earlier draft said the opposite.)* M5/V19b's held-out 15/16 was LeRobot ACT on
  `observation-v19` (`07fad282…`, not committed): a 13-value state of the six joint angles
  **and** the simulator-privileged `sim_cube_pose`, beside the camera. The IR route of M7 (U3,
  0.5625) reads the same privileged pose. The camera-and-joints route (`observation-v8.toml`)
  scored 0–1/16 in V8, V11 and V12 on older harnesses and has not been measured on the current
  one. Which route the template uses is the owner's choice (section 8).
- **"Train again focused on failures" must not train on the evaluation's seeds.** That would
  make the next comparison meaningless (§13.3). It widens the *training* randomization towards
  the perturbations that failed and collects with fresh seeds.

## 5. Sub-projects

| # | Scope | Serves goal | Needs |
|---|---|---|---|
| **S1** | Docking shell, step bar, start screen, project folder, the cube template, ③/④ progress and traffic light, ⑤ results, per-episode outcome on disk; today's tabs become Advanced tabs | 1, 3, 4; 2 for templates | — |
| **S2** | Checkpoint previews; which failure predicate fired (task-specific cause names); train again focused on failures | 3, 4 | S1 |
| **S3** | Action blocks on a template's scene (SO-101): a data-driven waypoint program, a concrete waypoint expert | 2 | S1 |
| **S4** | Scene from scratch: viewport editing, scene and document generator, one-frame camera render, success/failure sentences; the empty-project card | 2 | S1, S3 |
| **S5** | Reward: RL state teacher → image demonstrations → vision student; the "reach" template | 2 | S1 |
| **S6** | Teleoperation in the viewport | 2 | S1, S4 |
| later | general IK for imported robots; remote execution; contact conditions; an installable package | — | — |

Directions already fixed for S2–S6 (each still gets its own design pass):

- **S2.** The cheapest hook for a preview is the `checkpoint` telemetry event, which fires
  right after a checkpoint bundle is written (`crates/es/src/cmd/train.rs`, `PolicyPack`);
  `es loop cycle` runs a short evaluation on it beside the trainer, without touching the
  trainer. Failure-focused re-training is a new cycle whose collect randomization is widened
  towards the failing perturbation kinds, fresh seeds, `[init] policy` from the last run, and
  the same Evaluation IR.
- **S3.** The waypoint program is a data file (a demonstrator recipe, not an IR — Task IR owns
  no policy, §5.1 rule 6) executed by a concrete struct beside `ScriptedExpert` — not a trait
  (INV-17). SO-101 first, through the IK that exists; its hash goes into the dataset's
  provenance.
- **S4.** The editor edits a scene model and writes MJCF through `scene_to_mjcf`; an
  `es project generate` command turns scene + choices into the five documents (so the CLI and
  the editor share it); an `es render` command renders one frame of a declared camera through
  `es-render`, so the preview is the real observation and the editor still creates no Vulkan
  device. The sentence editor offers only what `es-env` lowers. "Held for 1 s" compiles to
  "inside and nearly still" (the demo's settling bound) because IR-D has no hold node; the
  lowering gains multi-lane `GetJointState`, `Slice` and `GetBodyVelocity` (`es-env`, not
  `es-ir`). Contact conditions wait for `GetContact` lowering.
- **S5.** The pieces exist: `[rl]` trains a state teacher, `es loop collect --policy teacher
  --frames` renders the Task IR's camera during its rollout. Missing: keeping only successful
  episodes, and a cycle that chains teacher → collect → student.
- **S6.** A long-lived `es` session process that steps the simulation with the Safety Plane
  on (INV-12), takes end-effector targets from the editor, and records with
  `action_source = Human`. The telemetry transport is one-way today; S6 adds the upstream
  channel.

## 6. S1 in detail

### 6.1 Structure

**Shell.** Step bar (fixed) + dock area + status line (fixed); the start screen is full-window.
Dock tabs: viewport; step panel (left, per step); summary / inspector (right); console;
Advanced (design graph, what the policy sees, problems, the raw metric table). Each step has a
default layout; the person's layout is persisted through `eframe::Storage`, under a key
spelled in `recent.rs` like the others.

**Docking crate.** `egui_dock`, pinned to a release that supports egui 0.32 and the workspace
MSRV (1.85). Confirming that release is the plan's first task. If none exists, the fallback is
egui's own resizable, collapsible `SidePanel`s, and only drag-to-re-dock is lost.

**Code layout.** The rule of `editor-shell.md` §2 stays: **nothing is decided in the drawing
code**. New view-models, each headless-tested:

| File | Owns |
|---|---|
| `model/workflow.rs` | the step bar: the state of each step from the project, the runs on disk, and the live run |
| `model/project.rs` | reading a project folder, numbering and creating a run |
| `model/health.rs` | the traffic light |
| `model/results.rs` | verdict, comparison, per-situation bars, tiles |
| `model/home.rs` | templates and their availability, the `--check-deps --json` reading |

`app.rs` (2,456 lines) is split by screen into `ui/` (shell, home, train, results, advanced),
all drawing.

**Code the editor holds on its owners' behalf goes back to them.** `es-editor` counts 6,128
lines by §1.5's measure today (code lines: no blank, comment-only or test lines; warn 6,000,
cap 10,000), and S1 adds roughly 2,000–2,500, so the cap does not force this; the owner chose
it for ownership and to remove a duplicate:

- `replay_view`'s CPU rasterizer (its shading is `es_render::cpu::shade_lambert` re-implemented)
  moves to a `raster` module of `es-render`; the trajectory posing, which needs `es-env`
  (layer 9), stays in the editor;
- `run_view`'s reading of a finished run moves to `es-eval`, which writes those files
  (`report.json`, `events.json`, `frames/`), together with `Rgb8Image`, the frame format's
  in-memory form. Strings a person reads stay in the editor.

No layer changes. (An earlier draft of this note said 8,238 lines; that is the count with
blank and comment lines, not §1.5's.)

### 6.2 Project and template

```
my-project/
  project.toml        # kind = "project", name, template = "cube-into-bin"
  runs/
    001/              # exactly what `es loop cycle --out runs/001` writes
      cycle.toml      # the recipe this run used, written by the editor just before launch
      telemetry.txt   # the live address, so a re-opened editor can attach again
      loop.jsonl, collect/, train/, eval/, showcase/
    002/ ...
```

`project.toml` holds a name and a template id in S1; S3 and S4 add the fields for authored
documents when they need them.

**Templates** live in `templates/<id>.toml` at the repository root: a name and description
(i18n keys), the robot, the teaching method, and **paths to the committed documents** —
`tests/fixtures/visible-learning/task.toml`, `observation-v8.toml`, the LeRobot ACT training
recipe, `evaluation-v8.toml` — plus a vision-route cycle recipe, which is new (the committed
`cycle.toml` is the privileged IR route). The documents are referenced, not copied: the scene
path is hash input (`SceneRef::canonical`), so a copy would move every hash downstream and the
template would no longer be the documents the numbers were measured on. `es` therefore runs
with the repository root as its working directory. That only works from a checkout; the code
carries a `ponytail:` comment naming the ceiling (an installable package is a later item).

### 6.3 Step state

`workflow.rs` is a pure function of (project, runs on disk, live run):

- ① and ② are **done** for a template and read-only: ① shows the scene through the CPU
  raster; ② shows a summary ("Action blocks: the SO-101 demonstrator, ready").
- ③ and ④ are *not started / running (stage, %) / done / failed (stage, exit code) /
  interrupted*, from the latest run's `loop.jsonl` and the live stages. ③ covers collect,
  expert gate and train; ④ covers eval and showcase.
- ⑤ opens once `eval/report.json` exists; a run list (001, 002, …) selects another run.

**Start settings in ③** are two: number of demonstrations, and training length
(short / medium / long). The editor writes `runs/NNN/cycle.toml` — the template's recipe with
these overrides, the train section inline — and launches it.

### 6.4 ③ / ④ progress, and the traffic light

1. **Start** writes `cycle.toml` and `telemetry.txt`, then runs
   `es loop cycle --recipe runs/NNN/cycle.toml --out runs/NNN --telemetry 127.0.0.1:<free port>`
   through `launch.rs` (`Kind::Cycle`), and attach-follows-launch connects.
2. **Stage cards** come from `stage.begin`/`stage.end` under plain names. Progress is
   `episode.end` count ÷ the demonstration count while collecting, and step ÷ `train.begin`'s
   total while training. Time left is shown only where a rate exists; **no whole-run ETA is
   invented**.
3. **Centre**: while collecting, the latest rendered demonstration frame and "demonstrations
   succeeded 190 / 200"; while training, the sample the policy sees and a large loss curve.
   S2 replaces this with checkpoint previews.
4. **Done**: `ViewportCommand::RequestUserAttention` (egui's own; no dependency) and an in-app
   notice; if the person is on ③ or ④, the editor moves to ⑤.
5. **Re-opened**: attach to `telemetry.txt`'s address; if that fails the run is
   *interrupted*, with [Resume] (`--from <stage>`) and [Start over].

Buttons: **Stop** ends the child. **Evaluate what it has learned so far** stops training and
runs `--from eval` on the newest checkpoint already written; whether `--from` accepts the
changed `[eval] checkpoint` is an item the plan verifies by running it.

**Traffic light** (`health.rs`), the first rule that matches wins:

| # | Light | Condition | Advice |
|---|---|---|---|
| 1 | red — *stopped* | child exit code ≠ 0, or a `stage.end` code ≠ 0 | per stage ("the demonstrator failed too often"), [console] [start again] |
| 2 | red — *broken* | loss is NaN or infinite | restart with a lower learning rate |
| 3 | amber — *not responding* | the process is alive but no telemetry for T s | wait, or stop |
| 4 | amber — *slow* | samples/s below a third of the run's median, after warm-up | check other GPU programs |
| 5 | amber — *stopped learning* | the loss has not fallen over a window | more data |
| 6 | grey — *starting* | no progress frame has arrived yet, and fewer than T s have passed | — |
| 7 | green — *going well* | none of the above | — |

T and rule 5's window are **not guessed**: the curve of the real acceptance run is recorded,
and "replaying a successful run is green from start to finish" is a test the thresholds must
pass. Synthetic series cover each rule firing. GPU memory is not on the training wire today,
so it is shown only when present.

### 6.5 ⑤ Results

Reads `runs/NNN/eval/`: `report.json`, the new `episodes.json`, `frames/`, `traj/*.estraj`.

- **Verdict.** Pass / fail is `report.passed`. The headline is successes over all episodes
  ("9 of 16"). Below it, each acceptance line in plain words with ✓/✗, through `labels.rs`.
- **Comparison.** With the project's previous run only when `evaluation_hash` is equal; if it
  differs, "evaluation conditions differ — not comparable" and no number (§13.3). [Compare
  with another run] picks another.
- **Why it failed.** From `episodes.json`'s termination and failure kinds, in plain words:
  timeout → *not done in time*; `safety_violation` → *hit a safety limit*; fallback → *the
  safety fallback stopped it*; `nan_detected` / `diverged` → *the simulation became unstable*;
  `chunk_underrun` → *motion gaps*; `deadline_miss` → *reacted too late*; a failure predicate →
  *met a failure condition*. The mapping is total over `FailureKind` and the termination kinds,
  with no wildcard arm (the `labels.rs` rule). One line of advice per cause, from the string
  tables. Task-specific causes ("dropped the cube") are S2.
- **Per situation.** `episodes.json` grouped by suite, "successes / episodes" bars. The label is
  built from the suite's **perturbation kinds** — a total mapping over the twelve
  `PerturbationKind`s (a suite with none is *nominal*), joined when a suite has several — not
  from the author's free-form suite name, which is shown in the hover.
- **Tiles and player.** Every episode as a tile (all / successes / failures), the last recorded
  frame as its thumbnail. A tile plays in the centre viewport through the existing replay
  (moved to `es-render`, section 6.1) with its timeline marks; camera: outside (orbit) / the policy's
  eye (recorded frames) / side by side; speed 0.5× / 1× / 2×.
- **Folded**: the 18 metrics (today's results table) and the settings and hashes of the run.
- **Buttons**: export the policy (a save dialog copying the `.esb`); run again (③ with the same
  settings). *Train again focused on failures* joins them in S2.

### 6.6 Backend changes (all small)

- `es --check-deps --json`: the same facts as today's text, machine-readable, for the start
  screen.
- `es eval run` (and so the cycle's eval stage) writes `episodes.json`: per episode — suite,
  cell, seed, termination, failure-kind counts, Safety Plane violations and fallbacks, steps,
  the frames directory and the trajectory path when recorded.
- The vision-route cycle recipe for the cube template, and `templates/cube-into-bin.toml`.
- `es policy init`: builds the bundle a cycle's `[collect] policy` names from the committed
  documents. A fresh checkout has no way to make one today (`untrained.esb` has come from
  test helpers and server artifacts), and the collect stage needs it for its Deployment IR —
  the Safety Plane the demonstrator runs under — even when the policy itself never acts.

### 6.7 Errors

No path crashes the editor; each says what happened in plain words and offers the next action.

| Situation | Shown |
|---|---|
| no `es` binary | red on the PC-check line, with `es_binary()`'s reason |
| no Python / LeRobot / MuJoCo | the template card disabled, "needs: LeRobot learning tools", and how to install |
| broken `project.toml` | a notice; the editor stays on the start screen |
| the run died, or the editor was closed mid-run | *interrupted*, [Resume] / [Start over] |
| eval ended with code 0 but no `report.json` | red — "no result file" (never treated as success) |
| an old run without `episodes.json` | ⑤ hides the tiles, draws the bars from `report.json`'s suite metrics, and says why |

### 6.8 Oracles (§1.4)

All under `cargo xtask ci`:

- **Headless model tests**: `workflow` (the whole state table); `health` (synthetic series fire
  each rule; **the recorded successful run replays green throughout**); `results` (verdict,
  comparison, hash mismatch, bars, tiles, old-run fallback); `home` (`--check-deps --json`
  parsing, template availability); `project` (create, read, run numbering).
- **`episodes.json`**: its episode count equals the report's `n_episodes`, and **successes ÷
  episodes equals the `success_rate` metric exactly** — two independently computed numbers
  checking each other.
- **`--check-deps --json`**: a schema test.
- **The moves of section 6.1** leave every existing golden byte-identical
  (`tests/golden/editor/replay-tick0-320x180.bin`, `replay_tick0_order.json`, the launch
  goldens).
- **Strings**: the existing i18n completeness test (en/ko key parity, every key used) and the
  label totality test, extended to failure causes and perturbation kinds.
- **Budget**: `es-editor` ≤ 10,000 source lines.
- **By hand** (a UI is not judged in CI): on Windows, start → template → Start training →
  watch → results, with screenshots; and **one real end-to-end run of the template's cycle,
  its numbers recorded** — whether they match V19b is reported, not required.

### 6.9 Not in S1

Checkpoint previews, task-specific failure causes, failure-focused re-training (S2); editing
action blocks (S3); editing a scene (S4); RL and the reach template (S5); teleoperation (S6);
remote execution; an installable package.

## 7. What this design keeps

- **The editor hosts no learning** (§23.1): every run is an `es` child it attaches to.
- **The Safety Plane is on every path** (INV-12): nothing here adds a way to run without it;
  S6's session runs it like evaluation does.
- **The evaluation stays fixed across a loop** (§13.3): comparisons require an equal
  `evaluation_hash`; failure-focused re-training never collects on evaluation seeds.
- **IR boundaries** (§5.1): the editor writes documents through `es-ir`'s own types and
  validators; a demonstrator recipe is not an IR; no UI type enters `es-ir`.
- **No new extension point** (INV-17): the waypoint expert of S3 is a concrete struct.
- **The string-table and label rules** of `editor-shell.md` §15 apply to every new word.

## 8. To verify in the S1 plan

1. an `egui_dock` release for egui 0.32 and MSRV 1.85;
2. `es loop cycle` runs the LeRobot route end to end (the recipe's `[collect] policy` on that
   route included);
3. `--from eval` after `[eval] checkpoint` is changed;
4. the traffic-light thresholds, from the recorded acceptance run;
5. how short / medium / long map onto the LeRobot route: settled by Y5 — `[run] steps` > 0 and
   every mark a multiple of the first (`lerobot-train` saves at one `--save_freq`); the presets
   are [2500, 5000], [10000, 20000], [20000, 40000, 60000];
6. **owner decision:** which observation route the cube template uses (section 4) — and, for the
   camera-only route, how the expert gate gets a bundle it accepts (packet Y5 found that no four
   committed documents make one: `XIR-040` under `evaluation-v8.toml`, `XIR-010` for
   `observation-v8.toml` with `learning.toml`).

   *Decided 2026-09-28: both, as two cards — `cube-into-bin` (camera only, marked experimental)
   and `cube-into-bin-hint` (cube pose given, marked practice). `es policy init` without
   `--learning` builds the camera-only gate's bundle (packet Y5b).*

## 9. What the implementation settled (2026-09-28)

Facts plan Y found or fixed that the sections above did not say; the review `docs/reviews/M12.md`
has the findings and the open decisions.

- **Docking:** `egui_dock` 0.17.0, the last release on egui 0.32 and MSRV 1.85 (0.18 moves to egui
  0.33, 0.19+ to Rust 1.92). Dock states persist per arrangement (one per step, one Advanced).
- **Language:** language and text size moved to the View menu; the start screen carries its own
  language switch (review H-5).
- **Templates need `render`:** a default `es` build has no renderer (`"render": false`), so both
  cube cards stay disabled until `es` is built with `--features render`; the start screen says so
  (review H-1).
- **The traffic light** (section 6.4) as built, first match wins: *stopped by you* (grey) →
  *stopped* → *broken* → *not responding* → *starting* (grey, before any message) → *slow* →
  *stopped learning* → *going well*. A failed acceptance is **not** *stopped*: `es loop cycle` ends
  eval (and showcase, which repeats eval's code) with 1 when the acceptance fails, and ④ reads
  "finished — did not pass", grey, with ⑤ open.
- **Run folders:** `write_run` creates `runs/NNN` exclusively and is refused while a run is going;
  Start is refused while a child runs, a start is queued, or a step is still running. The ledger
  maps to the step bar as: `collect` row → collected (and, with an expert, only once the latest gate
  row says passed); `train` row → trained; an `evaluate` row without `expert` → evaluated;
  showcase writes no row. `--from` accepts `collect | train | eval | showcase`.
- **`episodes.json`** holds suite, cell, episode, seed, termination, plane steps and changed steps,
  and the episode's own failure-mode buckets; frames and trajectory are found by the cell name
  (`frames/<cell>/`, `traj/<cell>.estraj`), not stored.
- **Length presets:** camera-only [2500, 5000], [10000, 20000], [20000, 40000, 60000] (the LeRobot
  route's marks are a multiple series); hint [1000, 5000], [1000, 5000, 20000],
  [1000, 5000, 20000, 60000] — both mean 5,000 / 20,000 / 60,000 steps.
- **Budget:** `es-editor` is at 9,715 of 10,000 after S1. S2 needs a split first (review S-1, H-2).
