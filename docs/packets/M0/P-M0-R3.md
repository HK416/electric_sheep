# P-M0-R3 — MJCF `<default>` class-graph hardening

Spec: spec 3.1 (conventions), spec 17.2 (backend semantic mapping), spec 4.3 (MuJoCo-family
backends). Follow-up to the M0 review (`docs/reviews/M0.md`, two Should-fix items on
`crates/es-assets/src/mjcf/mod.rs`). Review class A. Builds on P32.

## context

```
crates/es-assets/src/mjcf/mod.rs
tests/fixtures/mjcf/default_cycle.xml
tests/fixtures/mjcf/default_duplicate.xml
docs/packets/M0/P-M0-R3.md
```

## spec

MJCF is untrusted input, so the class graph must be made acyclic at construction and the walk
over it must be bounded anyway.

- **Duplicate class names are an error**, matching MuJoCo. `MjcfError::DuplicateClass
  { line, class }`. Previously the second `<default class="x">` overwrote the first one's
  parent link while merging attributes, so the earlier inheritance was lost with no
  diagnostic. Erroring rather than warning was chosen because a silently-changed parent link
  changes `scene_hash` with no way for a caller to notice.
- `main` is the one exception: it is the implicit root class, so repeated **top-level**
  `<default>` sections still merge into it, exactly as before.
- A **nested** `<default>` therefore has to name a class no one has declared. An unnamed
  nested `<default>` would default to the name `main` and make the root class a descendant of
  itself, so it is rejected by the same duplicate check (`duplicate default class \`main\``),
  as is a nested `<default class="main">`.
- Together these make the class graph a forest rooted at `main`: a nested class's parent is
  its strictly-enclosing class, and no name repeats.
- `resolve` keeps a bound anyway — a parent chain longer than the class table has revisited a
  class, and returns `MjcfError::ClassCycle { line, class }` instead of pushing to `chain`
  until the process is out of memory. This is a backstop, unreachable from a parsed file.
- Both new variants carry the line, like every other `MjcfError`.

## oracle

```
cargo test -p es-assets mjcf
```

## acceptance

- `default_cycle.xml` (the review's `<default class="a"><default class="a"/></default>`)
  returns `line 6: duplicate default class \`a\`` — not a hang, not an OOM.
- `default_duplicate.xml` returns `line 14: duplicate default class \`x\``.
- Both fixture tests parse inside a worker thread and fail the test if no result arrives
  within 5 s, so a regression to the unbounded walk is reported as a failure rather than
  hanging CI.
- `an_unnamed_nested_default_is_rejected`: nested `<default>` with no `class` is an error.
- `repeated_top_level_defaults_still_merge_into_main`: two top-level `<default>` sections
  parse with no error and no warning.
- `a_cyclic_class_graph_cannot_loop_resolve`: a hand-built `a -> b -> a` class table makes
  `resolve` return `ClassCycle`.
- `arm2.xml` and every other existing MJCF fixture still parse to the same values, and the
  mutation fuzzer (`fuzz_mjcf_importer_never_panics_on_mutated_fixtures`) covers the two new
  fixtures too.

## forbidden

Any file outside `context`. Changing existing fixtures. `crates/es-assets/src/urdf.rs`,
`gltf.rs`, `scene.rs` or the `mjcf` submodules (`attrs.rs`, `elements.rs`, `orient.rs`).
Adding a trait or an XML dependency. `HashMap` / `HashSet` (spec 3.4). Resolving or reading
any referenced file.
