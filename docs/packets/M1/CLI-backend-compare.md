# CLI-backend-compare — `es backend compare`

Spec: §17.2 (backend semantic mapping, `es backend compare`), §17.3 (determinism tiers),
§14.4 (an unmapped row with `severity: error` blocks execution), §3.5 (determinism-tier
comparison metrics).

## context

```
crates/es/Cargo.toml
crates/es/src/cmd/mod.rs
crates/es/src/cmd/backend.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
docs/packets/M1/CLI-backend-compare.md
```

## spec

`es backend compare` (the CLI wiring for `es-physics-backend::mapping`, whose
`mapping_report`/`compare_backends` already do the real work) takes a scene (`--scene
<file.xml|urdf>`, or `--task <task.toml>` as an alias that reads the Task IR's
`SceneRef::path`) and a comma-separated `--backends` list drawn from
`BackendKind::{mujoco-cpu,mjwarp,newton,physx}`.

- Parses the scene with the importer matching the file extension: `es_assets::parse_mjcf`
  for `.xml`, `es_assets::urdf::parse_urdf` with `PackageResolver::from_env()` for
  `.urdf`. `--task` reads the `SceneRef`; a hash-only ref (`path` empty) is a runtime
  error telling the user to pass `--scene` directly.
- Prints `mapping_report(scene, kind)` for every requested backend first, unconditionally
  -- this must run even for a backend with no adapter (`newton`, `physx`) or one that
  turns out to be unavailable, so the semantic-mapping table is always visible.
- Then, per requested backend: a mapping report that is `blocked` (spec 14.4) is never
  probed for availability and is reported `SKIPPED (blocked by mapping report, ...)`
  naming the blocking features. Otherwise `MuJoCoCpuBackend::is_available()` /
  `MjWarpBackend::is_available()` decide `available` vs `SKIPPED (<reason>)`; `newton`/
  `physx` always print `SKIPPED (not implemented (M2/M3))` (no adapter exists).
- For every pair of backends that both ended up `available`, constructs both instances,
  pre-loads each with a `LoadConfig { n_envs: --envs, seed: --seed, rate: None }` (so
  `compare_backends`, which only loads when `model_info()` is `None`, picks these up
  instead of its own `LoadConfig::default()`), builds a control sequence from the
  loaded `ModelInfo::nu` (all-zero by default, or seeded pseudo-random -- a small inline
  splitmix64, no `rand` dependency -- under `--ctrl-random`), runs `compare_backends`,
  and prints its `Display` table (which repeats each side's mapping report, by that
  type's own `Display` impl -- accepted duplication rather than reaching into
  `es-physics-backend` to change it, which is out of this packet's scope).
- Exit code: 0 once every requested backend ran a comparison or was skipped for an
  environment reason (unavailable / not implemented); 1 when any requested backend's
  mapping report is `blocked` (spec 14.4), or any comparison's `max_dqpos` exceeds
  `--tol` (default `1e-6`); 2 on a usage error (missing `--backends`/`--scene`/`--task`,
  an unknown backend name, a bad flag value).
- Never panics on an unavailable or blocked backend -- both are ordinary `SKIPPED` rows,
  not an `Err`.

## oracle

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
```

## acceptance

- `backend compare --scene tests/fixtures/mjcf/pendulum.xml --backends mujoco-cpu,mjwarp`
  prints both backends' mapping reports; `mjwarp` is `blocked` (the fixture's
  `option cone="elliptic"` has no pyramidal-only mapping on `mjwarp`, spec 17.2) and is
  always reported `SKIPPED`, so the process exits 1 regardless of whether a Python
  `mujoco` happens to be installed.
- The same file with `--backends mujoco-cpu` alone (not blocked) exits 0, `mujoco-cpu`
  reported `SKIPPED` for an environment reason in a CI Python with no `mujoco` package.
- `--backends not-a-real-backend` exits 2: backend names are validated during flag
  parsing, before the scene file is ever read.
- No subcommand path calls `.unwrap()`/`.expect()` on CLI input, file contents, or a
  spawned backend's output.

## forbidden

- `crates/es-ir/**`, `crates/es-compile/**`, `crates/es-runtime-embedded/**`, `xtask/**`,
  the root `Cargo.toml`, and any `docs/ARCHITECTURE*.md` -- owned by concurrently
  in-flight packets.
- `crates/es-physics-backend/**`, `crates/es-physics-core/**`, `crates/es-assets/**` --
  this packet only consumes their public API (`mapping_report`, `compare_backends`,
  `BackendKind`, `MuJoCoCpuBackend`, `MjWarpBackend`, `parse_mjcf`, `urdf::parse_urdf`).
- `clap` or any other CLI-argument-parsing crate; `rand` or any other RNG crate for
  `--ctrl-random` (a inline seeded splitmix64 is enough).
- `HashMap`/`HashSet` -- `BTreeMap`/`BTreeSet` or a small `Vec` only, per repo
  convention.
- Committing.
