# M12 R5 — `render` is a default feature of `es`

Owner decision H-1, 2026-09-29: a default `cargo build -p es` has no renderer, so the start
screen keeps both cube templates disabled (`"render": false`) even with every Python tool
installed. `es-render` is already built for `es-editor`, so the cost to a default build is small.

## spec

- `crates/es/Cargo.toml`: `render` joins `default`. `--no-default-features` still builds without
  it, and every `#[cfg(not(feature = "render"))]` path keeps working there.
- Tests that assert the *absence* of the renderer in a default build (for example the
  `--frames needs the render feature` refusal) move under `#[cfg(not(feature = "render"))]` or run
  the binary built without default features; none is deleted.
- `es --check-deps --json` then reports `"render": true` on a default build; the start screen's
  render install line disappears for a default build without any editor change.
- Documents that say "the default build of `es` does not pull `es-render` in" (the `eval.rs`
  refusal text, `docs/design/*`, `python/es/README*`, CLAUDE.md if it says so) are corrected.

## context

```
crates/es/Cargo.toml
crates/es/src/**
crates/es/tests/**
python/es/README.md
python/es/README.ko.md
docs/design/**
Cargo.lock
docs/packets/M12/P-M12-R5-render-default.md
docs/packets/M12/P-M12-R5-render-default.ko.md
```

## oracle

1. `cargo build -p es` then `target/.../es --check-deps --json` → `"render": true`.
2. `cargo test -p es` (default features) and `cargo test -p es --no-default-features` both pass.
3. `cargo xtask ci` stages that build `es` (fmt, clippy `-D warnings` on the workspace, layering).

## forbidden

Changing any renderer behaviour or hash; connecting to any remote server.
