# M10 W0a — the policy runtime's thread count enters `runtime_hash`

Spec: §5.3 (the sentence added 2026-09-22: `runtime_hash` covers the policy runtime's intra-op
thread count; the pool is not pinned), §10.4 (byte-identity of an evaluation), §28.13 rule 1 and
wave 0, §3.5 tier 1. Review: `docs/reviews/M8.md` S-1 and the human decision "the worker thread
pool" — the owner chose **option 2 (say it in the hash)** on 2026-09-22. Design note:
`docs/design/evaluation-execution.md` 2.7 (the three options; this packet writes the decision
paragraph under them). Type B with one D row.

## the question

`es eval run --jobs 8` and `--jobs 1` produce different `report.json` bytes on the committed demo
documents because each worker's Torch pool is capped at `cores / N` and Torch's CPU inference is
not bitwise across intra-op thread counts; `execution_hash` does not know. **If the torch
subprocess reports the thread count it actually runs with and `TorchRuntime::runtime_hash`
covers it, do two reports taken at different counts get two `execution_hash`es while the same
count keeps every byte, and does nothing else move?**

## spec

* `python/torch_ref.py`: the handshake message gains `"threads": torch.get_num_threads()`
  (the intra-op pool as Torch resolved it from `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / its
  default). `PROTOCOL_VERSION` 1 → 2 (`crates/es-policy/src/torch_runtime.rs`): the wire
  schema changed, and the protocol number is already a `runtime_hash` input.
* `TorchRuntime::runtime_hash()` = `blake3(RUNTIME_TAG ‖ "torch" ‖ protocol ‖ version ‖ threads
  as u32 LE)`. `runtime_hash_of` (`crates/es-policy/src/runtime.rs:143-150`) either gains a
  `threads: u32` parameter that every other runtime passes as `1`, or `TorchRuntime` computes
  its own digest with the extra field — the agent's choice; the `FakePolicy` / `CpuPlan`
  digests may move (nothing pins them) but must stay deterministic.
* `EvaluationLock` (`crates/es-eval/src/runner.rs:322-328`) gains `runtime_threads:
  Option<u32>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`), filled from a new
  `PolicyRuntime`-independent accessor — the evaluator asks the runtime for the count it
  reported (a method on `TorchRuntime`, not on the trait: INV-17 forbids widening the trait
  for one runtime; the CLI knows which runtime it built). The committed
  `tests/fixtures/visible-learning/run/evaluation.lock` (placeholders) keeps deserialising.
* `es eval run`'s help (`crates/es/src/cmd/eval.rs:56-81`): the byte-identity claim reads "at
  the same policy-runtime thread count, which `evaluation.lock` records and `execution_hash`
  covers".
* **Nothing about the pool changes**: `shard_thread_env`, the `cores / N` cap, and `--jobs 1`
  inheriting the ambient pool are as they were; no committed number moves.

## context

```
python/torch_ref.py
crates/es-policy/src/torch_runtime.rs
crates/es-policy/src/runtime.rs
crates/es-policy/tests/**
crates/es-eval/src/runner.rs
crates/es-eval/tests/**
crates/es/src/cmd/eval.rs
crates/es/tests/cli.rs
docs/design/evaluation-execution.md
docs/design/evaluation-execution.ko.md
docs/packets/M10/W0a-thread-pool-hash.md
docs/packets/M10/W0a-thread-pool-hash.ko.md
```

## oracle

1. `cargo test -p es-policy torch_runtime_hash_covers_the_thread_count` (`ES_PYTHON`; skips
   with a printed reason without it): two `TorchRuntime`s spawned with `OMP_NUM_THREADS=1` and
   `=2` exported to the child report 1 and 2 and hash differently; two at the same count hash
   the same; `PROTOCOL_VERSION == 2` is refused by an older `torch_ref.py` by name (the existing
   version-mismatch refusal).
2. `cargo test -p es-eval` — the `--jobs 1 / 2 / 4` byte-identity tests on `FakePolicy` stay
   green (`runtime_threads` is `None` for a runtime that has no pool).
3. Server (Linux, 16 cores; CPU queue `~/artifacts/plan-w/queue/cpu.lock`; artifacts
   `~/artifacts/plan-w/w0a/`), on the **committed** demo documents and U0's bundle, the
   nominal suite only (`evaluation.toml` restricted as `evaluation-execution.md` 2.7 did):
   `--jobs 1`, `--jobs 8`, `OMP_NUM_THREADS=2 --jobs 1`. One table: `evaluation_hash`,
   `execution_hash`, `runtime_threads` from the lock, blake3 of `report.json`, wall-clock.
   Expected: `evaluation_hash` equal in all three; rows 2 and 3 equal in every column but
   wall-clock; row 1 differs in `execution_hash`, `runtime_threads` and the report digest —
   the 2.7 finding, now visible in the chain. Written under 2.7 as "Decision (M10/W0a)".
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M10/W0a-thread-pool-hash.md`.

## acceptance

Oracles 1–4; the 2.7 decision paragraph and its Korean sibling; the help text.

## forbidden

Pinning or changing the pool (`shard_thread_cap`, `THREAD_ENV_VARS`, `--jobs 1`'s behaviour);
the `hardware_capability` slot (L24 is another packet); widening `PolicyRuntime` (INV-17);
`docs/ARCHITECTURE*.md` (the sentence is in); `tests/golden/**`; any committed fixture.
