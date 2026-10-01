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
