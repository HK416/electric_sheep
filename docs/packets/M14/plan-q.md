# M14 plan Q — S3: teaching with action blocks

> Packets for sub-project S3 of `docs/design/editor-redesign.md` (sections 3, 5, 11). The owner
> granted autonomy on 2026-09-29 ("남은 구현을 진행"): design follows the agreed direction; only
> spec, Safety Plane and hash decisions go back to the owner. Each task is a §1.2 packet: its
> **Files** block is its `context` for `cargo xtask check-scope`.

**Goal:** in ② a person sees the demonstration as a short list of blocks in plain words ("move
above the cube by 4.5 cm", "close the gripper", "move above the bin"), changes one, presses
"한 번 해 보기" and watches the demonstrator do it, then trains on demonstrations of *their*
program — without a TOML file or a line of code.

**The one idea.** Today's demonstrator (`es_env::ScriptedExpert`, `--expert so101-pick-place`)
is already a list of waypoints — a state machine over seven stages whose targets come from one
`ExpertCfg` of hand-tuned numbers (`demo_cfg`). S3 makes that list data: a **demonstration
program** file, executed by the same struct. The built-in `so101-pick-place` becomes the
committed program `templates/teach/so101-pick-place.toml`, and the refactor is proven by a golden
of today's chunks, bit for bit.

## Global Constraints

- Everything in `plan-z.md`'s Global Constraints still holds (MSRV 1.85, egui 0.32.3, i18n and
  label totality, decisions in `es-editor-model`, goldens by generator only, no trailers,
  worktree + `--ff-only`, never push, `CARGO_TARGET_DIR=target/alt`, no remote server, no
  learning run inside an agent packet).
- **Not an IR.** A demonstration program is the demonstrator's recipe, like `ExpertCfg` today
  (Task IR owns no policy, §5.1 rule 6). No IR type changes; `es-ir` is not touched.
- **No new trait (INV-17).** The program is data run by the existing concrete `ScriptedExpert`.
- **Safety Plane:** untouched. Every demonstration still goes through the plane (INV-12), paced
  to the Deployment IR's envelope by `ExpertCfg::pace_to` exactly as today.
- **Hashes:** no existing hash definition moves. A dataset collected from a program is
  different *data* (its content hash says so); the ledger's collect step records the program's
  blake3 as new provenance beside the existing `expert` input.
- **Determinism (§3.4):** the program's numbers are parsed once; the executor keeps today's
  `es_math::approx` transcendentals. A block's degrees are converted with `DEG_TO_RAD` exactly
  as `demo_cfg` does, so the built-in program reproduces today's radians bit for bit.

## Review Focus

1. **The refactor moves nothing:** Q1's golden of today's chunk sequence (recorded *before* the
   refactor, in its own commit) is byte-identical after it, and the Python-gated
   `expert_solves_the_pinned_seeds` still solves 8 of 8.
2. **A place resolves exactly:** "bin" is the world position of the lowest geom of stem `bin`
   (`bin_floor`, `0.14 -0.1`), the same `f64` values `demo_cfg` writes as literals.
3. **An unreachable block is said before a run:** ② marks a move block whose target leaves
   SO-101's reach for some object position the Task IR can draw (the corners of its
   `Randomization` range), in plain words, and "한 번 해 보기" still runs it (the demonstrator
   then fails that demonstration by name, spec 17.2) — never a clamped approximation.
4. **The person's file, not the template's:** `Project::create` copies the template's program
   into the project; editing it never writes under `templates/`.

---

## Waves

| Wave | Tasks | Needs |
|---|---|---|
| 1 | Q1 the program and its executor (es-env) | — |
| 2 | Q2 `--expert <file>` (es, es-data) · Q3 the teach model (es-editor-model) | Q1 |
| 3 | Q4 the ② screen (es-editor) | Q3 |
| 4 | Q5 real runs (orchestrator), docs, review | wave 3 |

---

### Task Q1: the demonstration program, run by `ScriptedExpert`

**Files:** `crates/es-env/src/expert.rs` (and a new `crates/es-env/src/program.rs` if it keeps
`expert.rs` readable), `crates/es-env/src/lib.rs` (exports), `crates/es-env/tests/expert.rs`,
`tests/golden/expert/**` (generator only), `templates/teach/so101-pick-place.toml`.

**Format** (TOML, `kind = "demonstration"`):
```toml
kind   = "demonstration"
robot  = "SO-101"
object = "cube"          # the free-joint body a block calls "object"; latched at episode start

[[blocks]]               # move: go to a point, gripper held as given
move  = "object"         # "object" | a place (a geom stem in the scene, e.g. "bin") | [x, y]
above = 0.045            # metres above the target's own height  -- or --
# height = 0.14          # metres above the world origin (exactly one of the two)
pitch = -85              # tool pitch, degrees; -90 is straight down
grip  = "open"           # "open" | "closed"

[[blocks]]               # grip: hold the previous block's pose, set the gripper, wait
grip = "closed"
wait = 5.0               # seconds
```
- A move block is reached when the four arm joints are within `pos_tol` of its IK solution
  (today's rule); a grip block advances after `wait` seconds, counted in the demonstrator's own
  steps — one per re-plan, every `replan_every` control ticks (`ExpertCfg::pace_to`): 5 s at a
  50 Hz control rate re-planning every 10 ticks is 25 steps, today's `close_ticks`. A `wait` that
  is not a whole number of steps is refused by name. After the last block the last command is
  held (today's `Done`).
- `"object"`'s point is the object's position latched at the episode's first step (today's
  `grasp`); a place's point is the world position of the **lowest** geom of that stem (the bin's
  floor), x and y. `above` adds to the target's own z (the object's centre; a place's lowest
  geom's centre); `height` replaces it.
- The robot is `SO-101` only (`so101_ik`); another value is refused by name. The tuning that is
  not the person's to change — `pos_tol`, `grip_open`/`grip_closed` angles, the pace — stays in
  code (`demo_cfg`'s values), so the file says only *what* to do.
- `templates/teach/so101-pick-place.toml` is today's seven stages as blocks: approach (object,
  above 0.045, -85°, open), descend (object, above -0.005, -85°, open), close (closed, 5 s),
  lift (object, height 0.14, -45°, closed), transport (bin, height 0.14, -45°, closed), lower
  (bin, height 0.06, -85°, closed), release (open, 5 s). `--expert so101-pick-place` runs
  that file (compiled in with `include_str!`), so there is one source of the demo's numbers.

**Oracle.**
1. First commit, before any refactor: an ignored generator
   `generate_expert_chunks_golden` (+ its check test) that drives today's `ScriptedExpert` on
   the demo scene through a deterministic **perfect-follower** state sequence — the joints take
   the last executed row of each chunk, the cube stays where it was drawn, for three pinned cube
   positions and the full episode length — and writes every chunk's `f64` bits to
   `tests/golden/expert/so101-pick-place-chunks.json`. No Python.
2. The refactor's commit leaves that golden byte-identical (`verify-goldens` shows no change).
3. Unit tests: the format (both `above` and `height` refused; an unknown key, an unknown place,
   a non-whole `wait`, an empty program refused by name); a place's point for `bin` equals
   `(0.14, -0.1)` exactly; the committed file parses and equals the compiled-in built-in.
4. Python-gated (skips without `ES_PYTHON`): the existing `expert_solves_the_pinned_seeds`.

### Task Q2: `--expert <program.toml>`

**Files:** `crates/es/src/cmd/loop.rs`, `crates/es/src/cmd/eval.rs`, `crates/es-data/src/collect.rs`,
`crates/es-data/src/training.rs` (the cycle's `[collect] expert` passes a path through),
`crates/es/tests/cli.rs`, `tests/golden/train/**` (generator only).

- `es loop collect` and `es eval run` (the expert gate) accept `--expert <name | path.toml>`; a
  value naming an existing `.toml` file is a program, anything else a built-in name (today only
  `so101-pick-place`). One resolver for both commands.
- The collect ledger step keeps `expert` (the name, or the path as given) and adds
  `expert_program` = the blake3 of the program's bytes (the built-in's bytes for a name).
- `Cycle` `[collect] expert` may be a path; the dry-run plan prints it verbatim. The committed
  cycles' goldens do not move.

**Oracle.** Unit: the resolver (name, file, missing file, a file that does not parse); the ledger
fields. CLI: `es eval run --expert templates/teach/so101-pick-place.toml` accepted by the gate's
argument check. Python-gated: `es loop collect --episodes 2 --expert <the committed file>`
writes the same dataset `content` as `--expert so101-pick-place`.

### Task Q3: the teach model (`es-editor-model`)

**Files:** a new `crates/es-editor-model/src/model/teach.rs`, `model/{mod.rs, project.rs,
template.rs, labels.rs, workflow.rs}`, `crates/es-editor-model/i18n/*`, `templates/*.toml` (the
`teach` field).

- A template names its program (`teach = "templates/teach/so101-pick-place.toml"`);
  `Project::create` copies it to `<project>/teach.toml`; `write_run` sets `[collect] expert` to
  that file. A project made before S3 (no `teach.toml`) keeps the template's built-in name.
- `Teach` (the ② state): the program, dirty flag, load/save (`teach.toml`, written atomically),
  reset to the template's; add a move block or a grip block after the selected one, delete,
  move up/down, edit one field; every edit re-validates.
- Plain words for each block, from its fields (i18n, both languages): e.g. ko "[큐브] 위 4.5 cm로
  이동 · 손목 -85° · 집게 열림", "집게 닫기 · 5초 기다리기". The object and places are named
  with the template's `[outcome]` names where they match (Z6's `outcome.cube`, `outcome.bin`).
- Validation per block, each an i18n key: unreachable for some object position the Task IR can
  draw (IK at the corners of the cube's `Randomization` range, and at the place for a place
  target); a pitch outside what SO-101 can hold there; a first block that is not a move; no
  block that closes the gripper; no block that opens it after closing. A warning never blocks
  saving or trying; it is shown on the block.
- "한 번 해 보기": the argv of an `es eval run --expert <teach.toml>` on a derived Evaluation IR
  (the nominal suite, one explicit seed, no acceptance — as Z1's `preview_evaluation` derives
  one) into `<project>/try/<n>/`, and the result read back (success or not, the cell to play).
  🎲 "다른 위치로" picks the next seed. ② refuses to try while a run is going (`may_start`).

**Oracle.** Headless tests: the template's program copied at create; `write_run`'s recipe names
it; each edit operation; the labels for every block shape in both languages (no `{}` left);
each validation on a crafted program (an unreachable `above`, no close, a grip first); the try
argv and derived IR parse (the IR through `es_ir::serial`, one suite, one seed, no acceptance);
reading a fixture try folder.

### Task Q4: the ② Teach screen (`es-editor`)

**Files:** a new `crates/es-editor/src/ui/teach.rs`, `crates/es-editor/src/ui/{mod.rs, shell.rs}`,
`crates/es-editor/src/app.rs` (hooks only).

Left: the block list — each block one row in plain words with its warning mark; select to edit;
add (move / grip), delete, up, down; "되돌리기" (the template's program); Save. Right (inspector):
the selected block's fields — target (object / a place / a point), above-or-height with a
centimetre slider, wrist angle in degrees, gripper open/closed, wait in seconds. Centre: "한 번
해 보기" and "🎲 다른 위치로", then the shared player (`ui/player.rs`) on the try's cell, and
"성공" / "실패 — <cause>" (the outcome classes of Z4a apply to a try as to any run). ② stays
read-only for a template whose program the editor cannot open (a pre-S3 project), with the reason.

**Oracle.** Build, clippy, tests; screenshots of ② with a warning and after a try (in-app hook,
never committed).

### Task Q5: real runs and documents (orchestrator)

On a quiet PC: a project from the hint card; change "move above the cube" from 4.5 cm to 6 cm;
try once; one short run from ③; the numbers (the try's time, the run's demonstrations and
their success, the verdict) in `docs/packets/M14/QV-verification.md`; the design note's S3
section rewritten to what was built; the review.
