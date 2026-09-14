# P-M4-R9 — Linux windowing features for eframe, so the ubuntu PR gate compiles `es-editor`

Closes the red `PR gate (spec 26.2)` job on `main` (GitHub Actions run 34780514440, commit
`c0fd56b`, ubuntu-latest). The workspace pins `eframe` with `default-features = false` and only
`glow` + `default_fonts` (`Cargo.toml:52-55`), which drops eframe's default `x11` / `wayland`
features. With neither, `winit 0.30.13` stops at its own
`compile_error!("The platform you're compiling for is not supported by winit")`
(`winit-0.30.13/src/platform_impl/mod.rs:78`). `cargo xtask ci` runs
`cargo clippy --workspace --all-targets`, so on Linux the gate fails before a single test runs
— on the platform §26.1 names as primary. Section 7 of `docs/design/editor-shell.md` said a
headless build does not need these features; that holds on Windows, where the crate was
verified, and not on Linux.

Not a review item: found while bringing up a Linux verification host (renderer-14, Ubuntu
26.04, RTX 4090, rustc 1.98.1) and reproduced there with
`cargo clippy -p es-editor --all-targets -- -D warnings` (exit 101, the same `compile_error!`).

Spec: §26.1 (Linux x86_64 is the primary platform), §26.2 (the PR tier runs clippy, fmt and unit
tests), §23 (editor), §1.4 (the oracle — the existing ubuntu PR gate — already fails; this
packet makes it pass without weakening it). No editor behavior changes.

## context

```
Cargo.toml
docs/design/editor-shell.md
docs/design/editor-shell.ko.md
docs/packets/M4/P-M4-R9.md
docs/packets/M4/P-M4-R9.ko.md
```

`Cargo.toml` changes only the `[workspace.dependencies]` `eframe` feature list and its comment;
the two design notes change only section 7; the packet and its Korean sibling are new.

## spec

- `[workspace.dependencies] eframe` keeps `default-features = false`, `glow` and
  `default_fonts`, and adds `x11` and `wayland` — eframe's own Linux defaults and nothing else
  from its default set (no `accesskit`, no `web_screen_reader`, no persistence).
- Both, not one. `x11` alone compiles, but a native Wayland session (the Ubuntu default) would
  then depend on XWayland being installed; `wayland` alone drops plain X11 hosts.
- No new system build dependency. With the pkg-config search path emptied
  (`PKG_CONFIG_LIBDIR=/nonexistent`, under which `pkg-config --exists wayland-client` returns 1),
  `cargo check -p es-editor` still succeeds: the X11 side is `x11-dl` / `x11rb` /
  `xkbcommon-dl`, and `wayland-sys` is built with `dlopen` (switched on by glutin and
  smithay-clipboard), so its build script links nothing. The ubuntu runner therefore
  needs no `apt-get install` step, and `.github/workflows/ci.yml` does not change.
- On Windows and macOS both features are no-ops: every crate they switch on is a Linux/BSD-only
  target dependency of winit and glutin.
- Section 7 of `docs/design/editor-shell.md` (and its Korean sibling) states the above instead
  of "building headless does not need them".

## oracle

Before the change, the first command fails on Linux with winit's `compile_error!`. After it,
on Linux (renderer-14, `ES_REQUIRE_GPU=1`, Python oracles installed):

```
cargo clippy -p es-editor --all-targets -- -D warnings
PKG_CONFIG_LIBDIR=/nonexistent PKG_CONFIG_PATH= cargo check -p es-editor
cargo tree -p es-editor -i winit -e features
cargo xtask ci
cargo xtask check-scope docs/packets/M4/P-M4-R9.md
```

On Windows:

```
cargo clippy -p es-editor --all-targets -- -D warnings
cargo xtask ci
```

## acceptance

- Every Linux command above exits 0, and `cargo xtask ci` passes all of its steps (fmt, clippy,
  tests, context-budget, layering, nostd, spec-refs, goldens).
- `cargo tree -p es-editor -i winit -e features` on Linux lists both `winit feature "x11"` and
  `winit feature "wayland"`.
- The Windows gate still passes.
- The GitHub `PR gate (spec 26.2)` job is green on the next push. The sibling
  `Reference oracle tier` job fails for unrelated reasons; see forbidden.
- Opening the editor window on Linux: `Target / Status: unverified` — the verification host has
  no display reachable over SSH and no Xvfb.

## forbidden

- `crates/es-editor/**` — no source change; the defect is in the workspace feature set.
- `.github/workflows/ci.yml` — the `Reference oracle tier` job's SKIPs (no slangc, newton,
  diffusers or ACT checkpoint on the runner) are a separate problem with its own design choice
  (install them or narrow the job), not touched here.
- Moving `eframe` / `egui` past 0.32.3, or the workspace `rust-version`: section 7 of
  `docs/design/editor-shell.md` pins both.
- Any other packet's scope.
