# P-M3-R4 — the `no_std` build is a CI step

Spec: §1.6 (`xtask` is the single verification entry point), §26.2 (PR-tier CI), W2
(`docs/packets/M3/W2-embedded-nostd.md`).
Review finding: `docs/reviews/M3.md` Should-fix — "no CI gate on the `no_std` split":
nothing under `.github/workflows/` or `xtask/` builds
`--no-default-features --target thumbv7em-none-eabihf`, so the property W2 establishes rots
silently on the next `es-core` change.

## context

```
xtask/src/main.rs
xtask/src/nostd.rs
.github/workflows/ci.yml
docs/packets/M0/P01.md
docs/packets/M3/P-M3-R4.md
```

## spec

- `xtask/src/nostd.rs`: `cargo xtask nostd` runs
  `cargo build -p es-core -p es-safety -p es-runtime-embedded --no-default-features --target
  thumbv7em-none-eabihf`. The target-detection check is a pure function,
  `target_installed(installed: &str, target: &str) -> bool`, over the text
  `rustup target list --installed` prints, so it is unit-testable without shelling out.
- When the target is not installed, `nostd::run` prints
  `SKIP nostd: target thumbv7em-none-eabihf not installed (rustup target add
  thumbv7em-none-eabihf)` and returns success — the same "SKIP is not a failure" convention
  as the `oracles` job's mujoco/mjwarp/torch skips. `run` takes a `require: bool`; with
  `require = true` the same missing-target condition prints `FAIL nostd: ...` to stderr and
  returns failure instead of SKIPping.
- `main.rs`: a new `nostd` match arm reads an optional `--require` argv token and calls
  `nostd::run(&root, require)`; `cmd_ci` calls `nostd::run(root, false)` right after
  `layering::run(root)` (SKIP-tolerant, because most dev machines and the base PR toolchain
  do not carry the embedded target).
- `.github/workflows/ci.yml`: the PR job's toolchain step gains
  `targets: thumbv7em-none-eabihf` so the runner actually has the target, and a second step
  after `cargo xtask ci` runs `cargo xtask nostd --require` — the simpler of the two options
  the review named (a separate required step, rather than teeing `cargo xtask ci`'s output
  and grepping it for `^SKIP nostd`), so a SKIP in CI is a hard failure instead of a silent
  green.
- `docs/packets/M0/P01.md`: note that this packet adds `nostd` to the dispatcher the same
  way P02-P06 did, so the skeleton packet's own description of "who adds what" stays
  accurate.

## oracle

```
cargo test -p xtask nostd::
cargo xtask nostd
```

`cargo xtask nostd` passes locally (the target is installed in this environment); the unit
tests cover both the installed and not-installed branches of the parser.

## acceptance

- `cargo xtask nostd` builds `es-core`, `es-safety`, `es-runtime-embedded` for
  `thumbv7em-none-eabihf` with `--no-default-features` and exits 0 when the target is
  installed.
- `cargo xtask nostd` (no target installed) prints a line starting `SKIP nostd:` and exits 0.
- `cargo xtask nostd --require` (no target installed) prints a line starting `FAIL nostd:`
  to stderr and exits non-zero.
- `cargo xtask ci` runs `nostd` after `layering` and before `check-spec-refs`.
- `.github/workflows/ci.yml`'s PR job toolchain step lists `thumbv7em-none-eabihf` under
  `targets`, and a step after `cargo xtask ci` runs `cargo xtask nostd --require`.
- Unit tests exercise `target_installed` directly (installed, missing, and a
  trailing-whitespace/empty-output line) without invoking `rustup`.

## forbidden

- No shelling out to `rustup` from unit tests — `target_installed` takes the listing text as
  a plain `&str` argument.
- No change to what `cargo xtask ci` builds beyond adding the one `nostd` step — this packet
  does not touch `context_budget`, `layering`, `spec_refs`, or `goldens`.
- No edits under `crates/**` other than what W2 already established; this packet only wires
  the existing build command into `xtask` and CI.
