# M11 R4 — a run refuses a scene that is not the one its Task IR pins

Spec: §5.3 (`scene_hash` is a hash-chain input; an equal `execution_hash` means equal
conditions), §6 (`SceneRef { path, scene_hash, asset_hash }`), §10.4, §3.5. Found by I3
(2026-09-24): `es eval run --scene` loads whatever scene file it is given. On the server, the
reach A0 policy was run on four edited scenes (no damping, no frictionloss, and so on), and all
four runs reported the committed `execution_hash` `08851281…`. No runtime code compares the
loaded scene's `SceneDesc::scene_hash()` with the Task IR's `task.scene.scene_hash`: a grep of
`crates/` finds only the Evaluation IR lock copying the declared value
(`es-eval/src/runner.rs`). Type B.

## the question

**Is there one check — in `es_env::Env::new`, which every runner (`es eval run`,
`es loop collect`, `es train`/`Rollout`, `es video`) goes through — that refuses a scene whose
content hash differs from the Task IR's declared `scene_hash`, while every committed document
still runs with its committed scene and no hash moves?**

## spec

* `Env::new` compares `scene.scene_hash()` with `task.scene.scene_hash`. On a mismatch it
  returns an `EnvError` that names both hashes (short hex) and the scene path. If the spec has
  an error code for this, use it; if not, do not add a code. The message is enough, and a new
  code is a spec change, which you propose in your report.
* Tests that build a `TaskIr` with a placeholder `scene_hash` (for example `[7u8; 32]`) and
  then run an `Env` must now declare the real hash. Test-only helpers may set
  `task.scene.scene_hash = scene.scene_hash()` explicitly. There is no opt-out flag in
  production code: a diagnostic run on an edited scene needs its own Task IR, and that is what
  makes it a different condition.
* `asset_hash` is out of scope. Say in the report whether it could be checked the same way.

## context

```
crates/es-env/src/env.rs
crates/es-env/src/lib.rs
crates/es-env/tests/**
crates/es-eval/tests/**
crates/es-data/tests/**
crates/es/tests/**
crates/es-py/tests/**
crates/es-py/src/**
crates/es/src/cmd/generate.rs
python/es/builder.py
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/P-M11-R4-scene-hash-check.md
docs/packets/M11/P-M11-R4-scene-hash-check.ko.md
```

Extended 2026-09-24 by the coordinator: `crates/es/src/cmd/generate.rs` and `python/es/builder.py`, so a
generated Task IR pins the loaded scene's hash and the Python builder says a zero hash is refused.

## oracle

1. `cargo test -p es-env scene_hash_mismatch_is_refused`: the SO-101 task with its committed
   scene builds; the same task with an edited copy of the scene (one damping value changed)
   is refused, and the error names both hashes.
2. `cargo test -p es --test cli eval_run_refuses_an_edited_scene`: `es eval run --scene
   <edited copy>` with a committed evaluation exits non-zero with that message.
3. Every existing test passes: `cargo xtask ci`. No committed hash, golden, `.estraj` or report
   moves (verify-goldens).
4. fmt, clippy `-D warnings`, check-scope.

## acceptance

Oracles 1–4, and one paragraph in `batch-domains.md` (+ko) on where the scene is checked.

## forbidden

An opt-out flag or environment variable; changing any committed Task IR or scene; changing
`scene_hash`'s definition.
