# P-M4-R5 — the Slang cache key covers every include

Spec: §2.3 (offline Slang → SPIR-V with a content-hash cache), §3.4 item 7 (the compiler is
part of the identity), §11.4 (`compile_hash`), §22 (a cache shared between ranks).
Review finding: `docs/reviews/M4.md` S-4 — `es-gpu/src/slang.rs:236-256` hashed include
directories with a **non-recursive** `read_dir` and swallowed read errors, so a header
included as `sub/helper.slang` contributed nothing to the key and a stale `.spv` was served
after it changed. A content-addressed cache that misses an input is worse than no cache: the
bundle `es task compile` packages (§11.4) is then named by a hash that does not describe it.

## context

```
crates/es-gpu/src/slang.rs
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R5.md
```

## spec

- `SlangCompiler::cache_key` returns `Result<String, GpuError>`. Every error from walking an
  include directory propagates; nothing is swallowed.
- A free function `hash_include_tree(&mut Hasher, dir, prefix, depth)` walks each include
  directory **recursively**. Entries are sorted (`Vec<PathBuf>` + `sort`) so the key does not
  depend on `read_dir` order; each file contributes its path *relative to the include root*
  (`sub/helper.slang`, not just `helper.slang`) followed by its bytes, so moving a file
  between subdirectories moves the key.
- `MAX_INCLUDE_DEPTH = 32`: deeper is an `io::ErrorKind::InvalidInput`, not a stack overflow.
  A symlink loop is the only realistic way to reach it.
- `compile` propagates with `?`; no other behaviour changes.

## oracle

```
cargo test -p es-gpu slang
```

Two tests:

- `cache_key_covers_files_in_include_subdirectories` — no `slangc` needed: a temp include
  root holding `sub/helper.slang`, key taken, helper rewritten, key taken again, keys differ;
  then the root is deleted and `cache_key` returns `Err`.
- `editing_an_included_subdirectory_file_recompiles` — end to end through `slangc` (SKIPs
  when it is absent, spec §1.4): a kernel that `#include "sub/helper.slang"` is compiled,
  compiled again (cache hit, `invocations() == 1`), the helper is edited, compiled a third
  time — `invocations() == 2`, and both the hash and the SPIR-V words differ from the first
  compile.

## acceptance

- Editing a file in an include *subdirectory* changes the cache key and causes a real
  `slangc` invocation. Observed: 2 invocations for three compiles.
- An unreadable or missing include directory makes `compile` fail rather than produce a key.
- The unchanged-input case still hits the cache: `cache_hit_does_not_start_slangc` is
  untouched and still reports one invocation for two compiles.
- `docs/api-notes/slang.md` and `docs/design/gpu-foundation.md` describe the recursive walk
  and the error propagation.

## forbidden

- No new dependency for the walk (`walkdir` and friends are not needed for `read_dir` plus
  recursion).
- No caching of the include hash across calls: the point of the key is that it is recomputed
  from what is on disk right now.
- No change to the cache layout, the file naming, or the concurrent-publish logic that
  `P-M4-R0-slang-scratch` established.
- No edits to `spirv.rs`, `buffer.rs` or `xtask` — those are R6 and R7.
