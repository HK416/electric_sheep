# P-M4-R0 — `SlangCompiler` scratch-file race

Spec: spec 2.3, spec 11.4 (Slang → SPIR-V content-hash cache), spec 3.4 (deterministic
execution contract — the compiled bytes for one cache key are the same regardless of who
compiled them). Follow-up to three `es-compile` GPU test failures, filed as a concurrency
hole in `es-gpu` rather than fixed at the `es-compile` call site. Review class A.

## context

```
crates/es-gpu/src/slang.rs
crates/es-compile/tests/observation_gpu.rs
docs/packets/M4/P-M4-R0-slang-scratch.md
```

## spec

`SlangCompiler::compile` named its `slangc` scratch input/output
`{hash}.{pid}-{invocations}.slang` / `.raw.spv` directly under the cache dir. `invocations`
is a per-instance counter starting at 0, so two threads of one process — or two
`SlangCompiler` instances in one process, e.g. a shared compiler plus a fresh one per
call — compiling the *same* cache key at the same time produce the same tag and delete each
other's scratch file mid-compile (`os error 2`). `crates/es-compile/tests/observation_gpu.rs`
worked around this with a process-wide `Mutex` serialising every GPU test.

The fix:

- A process-wide `static COUNTER: AtomicU64` hands out a globally unique `n` per `compile`
  call, independent of which `SlangCompiler` instance or thread makes the call.
- Each call gets its own scratch directory, `std::env::temp_dir().join(format!("es-slang-
  {pid}-{n}"))` — the OS temp dir, not the cache dir, so a scratch file is never adjacent to
  another call's — via an RAII `ScratchDir` guard that `remove_dir_all`s itself on drop,
  success or failure. The files inside are further named by cache-key hash, `n` and the
  thread id, for a readable name if a guard is ever skipped by a hard process kill.
- The cache entry is published by writing to `<hash>.spv.tmp-<n>` (unique per call) and
  `rename`-ing to `<hash>.spv`. Two calls racing on the same key both compile independently
  and both reach the rename; determinism (spec 3.4) means their bytes are identical, so
  whichever rename lands second harmlessly overwrites the first with the same content — the
  cache dir still ends up with exactly one `.spv` for that key. No lock, no dedup logic.
- `invocations` moved from `Cell<usize>` to `AtomicUsize`, which is what makes `&SlangCompiler`
  `Sync` and lets one instance be compiled through from multiple threads at all (the oracle's
  "shared compiler" arm would not have compiled otherwise).
- The `es-compile` workaround (`static DEVICE: Mutex<()>` in `observation_gpu.rs`, and the
  lock acquisition in `on_gpu`) is removed now that concurrent compiles of the same key are
  safe; GPU tests in that file run without cross-test serialisation again.

## oracle

```
cargo test -p es-gpu --lib slang::tests::eight_concurrent_compiles_of_the_same_key_do_not_clobber_each_other
cargo test -p es-compile --test observation_gpu
```

The first is the oracle written before the fix (fails with `os error 2` or a words mismatch
on the old code): 8 threads compile the same source through a mix of one shared
`SlangCompiler` and fresh per-call instances, all against one cache dir. It asserts every
compile succeeds, every returned `SpirvModule::words` is identical, and the cache dir holds
exactly one `.spv` for that key afterward. It prints `SKIP` and returns (no panic) when
`slangc` is not on `PATH` / `ES_SLANGC` (spec 1.4). The second confirms the `es-compile`
workaround's removal did not reintroduce the failure under this crate's own (formerly
serialised) GPU tests.

## acceptance

- `eight_concurrent_compiles_of_the_same_key_do_not_clobber_each_other` passes when `slangc`
  is available (verified on this machine, RTX 4060 present) and SKIPs cleanly otherwise.
- `cargo test -p es-compile --test observation_gpu` passes with the `DEVICE` mutex and its
  doc comment removed, run repeatedly (checked 3x) with no flake.
- `cargo fmt -p es-gpu -p es-compile --check` and
  `cargo clippy -p es-gpu -p es-compile --all-targets -- -D warnings` are clean.
- No new dependency. No change to `SlangCompiler`'s public API (`new`, `with_include`,
  `version`, `invocations`, `cache_dir`, `compile`, `compile_file` keep their signatures).

## forbidden

Any file outside `context`, in particular `crates/es-render/src/renderer.rs`'s
`COMPILE_LOCK` — a matching workaround for the same root cause, but a separate crate's file
and out of this packet's scope; removing it is a follow-up packet. Adding a lock or mutex to
`SlangCompiler` itself (that would just reintroduce serialised compilation). Changing the
on-disk cache file naming scheme (`<hash>.spv`) that other tools may already depend on.
Touching `default_cache_dir`, `cache_key`, or anything in `crates/es-gpu/src/spirv.rs`.
