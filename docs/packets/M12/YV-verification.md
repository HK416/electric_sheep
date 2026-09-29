# M12 Y-V — what plan Y still has to see run for real

Plan Y (`plan-y.md`, design note `docs/design/editor-redesign.md`) was implemented without a
single real learning run: this machine had no Python environment with MuJoCo and LeRobot, and
the owner ruled out the remote server for the duration (2026-09-28). Every oracle that needs
Python skips with a printed reason. This list is what those skips left unproven, in the order
worth running them. **Where it runs — a local venv on this PC or the server — is the owner's
decision; nothing here may be started on the server without it.**

## 0. Prerequisites

- A Python with `mujoco`, `torch` (CUDA for anything but the short preset) and `lerobot`, named
  by `ES_PYTHON`. Check: `es --check-deps --json` → `modules` all `true`.
- `es` built with the renderer: `cargo build -p es --features render` (a default build reports
  `"render": false`, and the start screen keeps both cube cards disabled). Whether `render`
  should become a default feature of `es` is an open owner decision (see the M12 review).
- The editor started from the checkout (`target/<profile>/es-editor.exe`), so it finds
  `templates/` and the `es` beside it.

## 1. Both cube templates run end to end

For each card — `cube-into-bin` (camera only, LeRobot route, `cycle-vision.toml`) and
`cube-into-bin-hint` (cube pose given, IR route, `cycle.toml`) — create a project from the
start screen, choose **short**, press Start, and let it finish.

Pass: every stage ends (collect → expert gate → train → eval → showcase), ⑤ opens by itself,
and the run folder holds `loop.jsonl`, `eval/report.json`, `eval/episodes.json`. Record, per
card: held-out success, `envelope_violation_rate`, wall-clock per stage. No success rate is
required — the camera-only card has never passed this task (design note section 4); the
numbers are reported as they come.

Then **medium** on at least the hint card, to see a number comparable with M7's U-series.

## 2. `episodes.json` agrees with `report.json`

`ES_PYTHON=… cargo test -p es --test cli eval_run_episodes_json_agrees_with_the_report` runs
instead of skipping. And on the run folders of item 1: for every suite, successes counted from
`episodes.json` ÷ its rows = the report's `success_rate` exactly.

## 3. "Evaluate what it has learned so far"

Start a run, wait for the first checkpoint mark, press the button. Expected risk (Y12): after a
kill in the middle of training `training.lock` may be missing, and `es loop cycle --from eval`
may refuse. Pass: either the evaluation runs on the newest mark and ⑤ opens, or the refusal is
understood and the button is changed (for example: disabled until a mark exists, or a cycle
change that makes `--from eval` accept a killed training) — a decision recorded in the review.

Also check, on Windows, that **Stop kills the whole process tree**: `LaunchModel::kill` ends
`es.exe`, and a Python trainer it started may survive (Y12). Pass: no `python` process of that
run is left after Stop; if one is, fix the launch model (a job object) as its own packet.

## 4. The traffic light's thresholds

Record the progress points (`step, loss, samples/s`, arrival times) of one successful run of
item 1 into a fixture, add `a_recorded_successful_run_is_green_throughout` to
`crates/es-editor/src/model/health.rs`, and tune `THRESHOLDS` until it passes **without**
weakening any synthetic test. Record the chosen values and the run they came from in the test.

## 5. What publishing costs the run

Every run the editor starts adds `--telemetry-image-every 50` (Y12). Measure the collect and
train stages with and without it on the same run settings (`editor-shell.md` section 13's gate 9
method). Pass: under 1 % (spec 23.4's gate); otherwise raise the interval.

## 6. The length presets in minutes

Wall-clock of short / medium / long on the machine used, for each card. If the start screen
should show an estimate, it is added from these numbers (never guessed).

## 7. The two cards collect the same demonstrations

Y5b gave the camera-only collect bundle a declared latency of one control period (20 ms), so
that the demonstrator runs exactly as under the IR-route bundle. Check: the collect datasets of
the two cards at the same seed and episode count have identical `qpos`/`ctrl` columns.

## 8. The whole flow by hand

On Windows, with the editor as a first-time user sees it: start → template card → Create →
Start (short) → watch ③ and ④ → ⑤ → play a failed episode → Run again. Screenshots of each
screen in both languages under `target/plan-y/yv/`, and every place the reader had to guess
written down as a follow-up.

## Results, 2026-09-29 (this PC: Windows 11, RTX 3060 12 GB, 64 GB RAM, 4 GB page file)

Environment, set up for this run: `.venv` with Python 3.12.10, `torch 2.11.0+cu128` and
`torchvision 0.26.0+cu128` (the server's `+cu129` build has no Windows wheel; 2.11 is what
`lerobot 0.6.1` allows, `torch<2.12`), `lerobot[training] 0.6.1`, `mujoco 3.13.0`; `es` built with
`--features render`. `torchcodec 0.11.1` cannot load without FFmpeg's shared libraries; LeRobot
falls back to `pyav` and nothing here decodes video (the export writes `image`). Every run below
was made by the editor's own model (`Project::create`, `write_run`) and launched with the argv the
editor's Start uses, `--telemetry-image-every 50` included.

**Item 1 — both cards run end to end: yes, after three fixes.**

| | hint card (IR route) | camera-only card (LeRobot route) |
|---|---|---|
| collect | 200/200 successful demonstrations, 103,881 frames, 26.4 min | 200/200, 26.2 min |
| expert gate | passed, 2.1 min | passed, 3.4 min (the `XIR-040` gate of Y5 is gone: Y5b holds on a real run) |
| train (short, 5,000 steps) | 2.3 min, ~480 samples/s | export + 5,000 LeRobot steps at ~16 steps/s, ~11 min |
| eval (96 episodes, 6 workers) | 7.3 min | 9 min |
| showcase | 0.5 min | 0.5 min |
| held-out nominal | **0/16**; all suites 6/96 | **0/16**; all suites 0/96 |
| `envelope_violation_rate` (nominal) | **0.996** | 0.12 |
| verdict | did not pass (exit 1) | did not pass (exit 1) |

Neither number is a surprise at the short preset: the hint recipe was written for 20,000 steps
and U3 passed only with the pretrained backbone and augmentation; the camera-only route had never
passed. The hint card's plane changed 99.6 % of the policy's steps — M11's open human decision
"the envelope for a learning policy", now seen from the editor.

What the first real runs found, each fixed and merged before the numbers above:
- **P-M12-R1** — the v3 export wrote one 2.9 GB row group; `pyarrow` cannot read a nested column
  past 2 GB from one row group, so `lerobot-train` refused the dataset. One row group per episode.
- **P-M12-R2** — the LeRobot route published no training progress, so the editor's light read
  *not responding* while training was healthy. `lerobot-train`'s bar and metric lines now feed
  stream 5.
- **P-M12-R3** — the editor never noticed a closed telemetry connection, so a run that had ended
  (or been resumed elsewhere) read *not responding* forever. Closed now ends the attachment and
  `telemetry.txt` is re-dialled every 10 s.
- **Unexplained, once:** the camera card's first evaluation lost one worker's MuJoCo process
  (`backend process died: … (os error 232)`), while two agents were compiling on the same PC. The
  same shard alone and the whole evaluation re-run on a quiet PC both completed; commit peaked at
  41.7 GB of a 67.9 GB limit. The cause is not proven: the backend's own last words are discarded
  (`proc.rs` nulls stderr, and a failed write never reads the error line the script leaves on
  stdout) — the follow-up is to surface them, then see it again.

**Item 2 — `episodes.json` agrees with `report.json`: yes**, exactly, in all six suites of both
runs (successes ÷ rows = `success_rate`).

**Item 5 (partly)** — the lengths in minutes, short preset, this PC: hint ≈ 39 min end to end;
camera-only ≈ 50 min without the interruptions.

Not yet run: items 3 (evaluate-so-far, the process tree on Stop), 4 (the light's thresholds on a
recorded run), 6 (medium/long), 7 (the two cards' demonstrations compared), 8 (the whole flow by
hand).
