# M18 plan K — file-level structure, and what M17 left

> The owner, 2026-10-02: "남은 작업 수행 및 코드 구조 개선 요청". Asked how far, they chose
> **file level only**: no crate split and no spec change. Every source file over 1,000 counted
> code lines is split into modules, and the crates stay as they are. Each task is a §1.2 packet; its
> **Files** block is its `context`.

## Global constraints

- **Pure moves.** No behaviour change and no renamed public item.
  - Every path other crates use keeps working. The parent file stays at its path, Rust's
    `foo.rs` + `foo/part.rs` layout, so the many file paths cited in `docs/` stay true; it
    re-exports what moved.
  - Tests move with the code they test, or stay where they are; none is weakened.
  - No golden, hash or document moves.
- **Split by responsibility,** not by line count: each new module is one cohesive job, named for it,
  aiming at ≤ about 700 counted lines. A file that already reads as one job may stay larger; say
  why.
- **Isolation.** Worktree per agent with its own `CARGO_TARGET_DIR=target/wt-<packet>`, at most
  three agents at once. Never push. No learning runs. Conventional Commits, English, no trailers.
  Docs keep their Korean sibling.
- **Gate:**
  - `cargo fmt --all --check`.
  - `clippy -D warnings --all-targets` on the touched crates.
  - The touched crates' tests, and the `es` CLI tests that cover them (with `ES_PYTHON`).
  - xtask `layering`, `context-budget`, `check-spec-refs` and `verify-goldens`.
  - The orchestrator runs `cargo xtask ci` once per merged wave.

## The files (counted code lines, tests excluded, 2026-10-02)

| file | lines | packet |
|---|---|---|
| `es-data/src/training.rs` | 2,204 | K1 |
| `es-data/src/collect.rs` | 1,070 | K1 |
| `es-editor/src/ui/advanced.rs` | 1,709 | K2 |
| `es-editor/src/ui/author.rs` | 1,039 | K2 |
| `es-ir/src/task.rs` | 1,254 | K3 |
| `es-ir/src/learning.rs` | 1,059 | K3 |
| `es-script/src/spec/compile.rs` | 1,194 | K3 |
| `es-eval/src/runner.rs` | 1,483 | K4 |
| `es/src/cmd/train.rs` | 1,206 | K4 |
| `es/src/cmd/eval.rs` | 1,134 | K4 |
| `es-import/src/rl_import.rs` | 1,421 | K5 |
| `es-render/src/cpu.rs` | 1,317 | K5 |
| `es-editor-scene/src/sentence.rs` | 1,042 | K6 |
| `es-assets/src/scene.rs` | 1,036 | K6 |

## Waves

| wave | packets |
|---|---|
| 1 | K1 (es-data) · K2 (es-editor) · K3 (es-ir, es-script's compiler) |
| 2 | K4 (es-eval, es's train and eval verbs) · K5 (es-import, es-render's CPU path) · K6 (es-editor-scene's sentences, es-assets' scene model) |
| 3 | K7 the hold node (M17's F-8) · K8 URDF mesh scale |

- **K7, holding for a while (M17 F-8).** Design: `docs/design/scene-authoring.md` section 4.9 (a
  `hold_ticks` on the `Terminate` sink, not a new IR-D node). The paragraph below was the first sketch. "[box] is still for [1 s]" compiles today to "still now" (IR-D
  has no hold node), so a box momentarily slow while tumbling counts.
  - A stateful node (true while its input has held for `n` control ticks, per env, reset with the
    episode) is a spec change to the Task IR's node set (§6, ko first).
  - It is used only by specifications that ask for it, so no committed hash moves.
  - `es-env` lowers it in reward and termination cones with an integer counter.
  - `es-script`'s `still` takes `for_s`, which must be whole ticks.
  - G8's sentence gets the "for [1 s]" slot.
  - R8's after-the-fact reading explains a hold clause by its inner predicate on the end row, and
    says so.
  - Design first, as a short section of `docs/design/scene-authoring.md` (4.9); the orchestrator
    writes it before the packet runs.
- **K8, URDF mesh scale.** A URDF `<mesh scale>` maps onto R3's `SceneDesc::mesh_scales` instead of
  warning.

## As merged (2026-10-02)

All eight packets are merged. `cargo xtask ci` is green at `ba7c7fe` with `ES_PYTHON`. No golden or committed hash moved.

**Files (K1–K6).** The 14 files over 1,000 counted lines are split by responsibility. Each parent
stays at its path and re-exports what moved, so no other crate and no cited path changed.

| packet | files | largest module now |
|---|---|---|
| K1 `f1e32b4`, `c3d9d2a` | `es-data` `training.rs` (9 modules), `collect.rs` (5) | `collect.rs` 514 |
| K2 `1db60d9`, `72c2c0e` | `es-editor` `ui/advanced.rs` (9), `ui/author.rs` (5 new) | `author/inspector.rs` 469 |
| K3 `0dc231c`, `3e6cf85`, `c1168a7` | `es-ir` `task.rs`, `learning.rs`; `es-script` `compile.rs` | `task/node.rs` 582 |
| K4 `361142a`, `4456cfa`, `3e229b8` | `es-eval` `runner.rs`; `es` `cmd/train.rs`, `cmd/eval.rs` | `runner.rs` 629 |
| K5 `c86dab7`, `2e74a21` | `es-render` `cpu.rs`; `es-import` `rl_import.rs` | `rl_import.rs` 480 |
| K6 `16a706d`, `3935fbb` | `es-assets` `scene.rs`; `es-editor-scene` `sentence.rs` | `scene/hash.rs` 480 |

**How each split was checked.**
- **Line diff:** exact line ranges were copied, and a multiset line diff against the original was
  empty except for imports, module docs, re-exports and `pub(super)`.
- **Test names:** the same before and after.
- **Source scans:** every test that reads source as text was kept true. Two `es-eval` tests count
  calls in `run_episode`, so it stayed in `runner.rs`.
- **Renderer:** the CPU path reproduces every render golden bit for bit.
- **Editor:** six pairs of screenshots, taken before and after the split, are pixel-identical.
- **Hashes:** a probe over 15 fixtures found `scene_hash`, `Debug` and TOML byte-identical.

**Crate totals.** They rose slightly with the new import lines. `es-ir` crossed its 6,000-line
target (6,031), and the four crates already over it stay WARN. The owner chose file level only, so
no crate was split.

**Features (K7, K8).**
- **K7 `72a7312`…`ba7c7fe` closes M17's F-8.** `Terminate` takes an optional `hold_ticks`
  (spec §6.3, ko first; `TASK-004` refuses 0).
  - **How it holds:** an integer counter per env lives where the episode budget is counted, so
    IR-D stays a pure DAG.
  - **The task specification:** `[success] hold_s` and `[failure] hold_s`. ① reads it as "아래가
    모두 [1초] 동안 맞으면 성공".
  - **R8's explanations:** a too-short hold is explained as "held for only X s".
  - **Measured:** a dropped box succeeds at the bounce without a hold, and with a 1 s hold only
    once it rests (steps 61–66 instead of 12–14).
  - **Hashes:** all 20 committed Task IRs (46 `Terminate` nodes) keep their hashes.
- **K8 `45e4717` closes M17's URDF item.** A URDF `<mesh scale>` maps onto
  `SceneDesc::mesh_scales` by the scene document's own function, so the names and the vertices
  equal MJCF's. Unscaled URDFs keep their hashes.

**Defaults K7 chose (the owner's to change).**
- The success bonus is paid on every tick of a hold. The bonus reads the instantaneous fold, and
  paying it once would need a new node.
- An absent hold shows as "0 s".
- A specification with no failure section shows no failure hold slot.
