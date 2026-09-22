# M10 W3b — `es` under the target: the artifact verbs move to `es-tools`

Spec: §1.5, §4.2 (the row added 2026-09-22: `es-tools` at layer 11 — the `es` verbs that read
artifacts: video, showcase, backend, evidence, gap, bench), §26.2 (one entry point: `es` stays the
only binary; `es-tools` is a library it calls), §28.13 wave 2. Review: `docs/reviews/M8.md` R6.
Precedent: P-M8-R6 and **W3a** (this wave's first split). Depends on **W0a** and **W1a** having
merged (`cmd/eval.rs`, `cmd/showcase.rs` are touched there). Type B.

## the question

`es` reports 7,485 code lines. Its `cmd/` modules form a star around `error` / `util` (35 lines);
`video` + `showcase` + `backend` (1,232, the `render` feature seam — `showcase` is the only module
behind it) and `evidence` + `gap` + `bench` (871) have no in-crate edges beyond `error` / `util`
(and `showcase` → `backend`, `eval`). **Does moving those six verbs to a layer-11 library crate
`es-tools`, with `CliError` / `hex` re-homed there and re-exported by `es`, bring `es` under the
target with `crates/es/tests/cli.rs` (12,800 lines, runs the binary) unchanged?**

## spec

* New crate `crates/es-tools` (layer 11, library): `video.rs`, `showcase.rs`, `backend.rs`,
  `evidence.rs`, `gap.rs`, `bench.rs` moved byte-for-byte except `use` lines; `error.rs`
  (`CliError`) and `util.rs` (`hex`) move with them and `es` re-exports them (`pub use
  es_tools::{error, util}` or the equivalent) so the remaining `cmd/*` modules compile
  unchanged. `showcase`'s use of `cmd::eval` (`renderer_cfg` or similar): move the smallest
  shared helper into `es-tools` and have `es::cmd::eval` call it from there (a `use` change, no
  body change) — state what moved.
* The `render` feature moves: `es-tools/Cargo.toml` `render = ["es-env/render", "dep:es-gpu",
  "dep:es-render"]`, `es/Cargo.toml` `render = ["es-tools/render"]`; `es` drops the optional
  `es-gpu` / `es-render` deps if nothing else in it needs them.
* `crates/es/src/main.rs`: the dispatcher calls `es_tools::video::run(...)` etc.; `TOP_HELP`
  unchanged; every command's argv, output and exit codes unchanged (the CLI tests are the
  oracle).
* `xtask/src/layering.rs` `LAYERS` gains `("es-tools", 11)`; `es` itself stays exempt.

## context

```
crates/es-tools/**
crates/es/src/main.rs
crates/es/src/cmd/mod.rs
crates/es/src/cmd/error.rs
crates/es/src/cmd/util.rs
crates/es/src/error.rs
crates/es/src/util.rs
crates/es/src/cmd/video.rs
crates/es/src/cmd/showcase.rs
crates/es/src/cmd/backend.rs
crates/es/src/cmd/evidence.rs
crates/es/src/cmd/gap.rs
crates/es/src/cmd/bench.rs
crates/es/src/cmd/eval.rs
crates/es/Cargo.toml
Cargo.toml
Cargo.lock
xtask/src/layering.rs
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M10/W3b-es-tools-split.md
docs/packets/M10/W3b-es-tools-split.ko.md
```

(`editor-shell.md` only where the launch panel names the `es` binary's verbs — one sentence if
anything.)

## oracle

1. `cargo xtask context-budget` — `es` < 6,000 (expected ≈ 5,380), `es-tools` ≈ 2,100.
2. `cargo test -p es --test cli` and `--test video` — unchanged files, green; `cargo test -p
   es-tools`.
3. `cargo build -p es --features render` and without; `cargo xtask layering`; `check-spec-refs`;
   `verify-goldens` (`tests/golden/video/**`, `editor/**` untouched).
4. `git diff --stat main -- crates ':!crates/es' ':!crates/es-tools'` — Cargo files only.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W3b-es-tools-split.md`.

## acceptance

Oracles 1–5.

## forbidden

Changing any verb's argv, output, exit code or help text; a second binary; editing
`crates/es/tests/**`; touching `cmd/{train,cycle,loop,dataset,policy,import,ir,task,generate,
check_deps,mcp,telemetry}.rs` beyond a `use` line for the re-homed `error` / `util`;
`tests/golden/**`; `docs/ARCHITECTURE*.md`.
