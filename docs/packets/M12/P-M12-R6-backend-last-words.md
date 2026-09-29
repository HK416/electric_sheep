# M12 R6 — a backend process that dies says why

Review finding S-6 (`docs/reviews/M12.md`): one evaluation worker's MuJoCo process died once with
`backend process died: 파이프가 닫히는 중입니다. (os error 232)` and the cause is unknown, because
`crates/es-physics-backend/src/proc.rs` discards stderr (`Stdio::null()`) and a failed write never
reads the error line the Python side leaves on stdout (`mujoco_ref.py` writes
`{"ok": false, "error": "import failed: …"}` and exits when its imports fail — a write after that
fails with os error 232 and the line is lost).

## spec

- `Process::call`: when the write fails, or the read returns EOF, collect the process's last
  words before reporting: whatever complete line is already on stdout (a protocol error line is
  decoded and its `error` text used), the exit status (`try_wait`, after a short bounded wait),
  and a bounded tail of stderr.
- stderr is captured into a bounded ring (last 64 lines or 16 KiB) by a reader thread, so an
  unread pipe can never block the Python side (the reason it was nulled); the ring is read only
  on failure.
- The error stays `PhysicsError::ProcessDied`; its text becomes e.g. `backend process died (exit
  code 1): import failed: DLL load failed while importing … — stderr: …`. Nothing changes on the
  success path; no hash, no timing on the hot path beyond the reader thread.
- Same for `crates/es-policy`'s torch runtime process if it shares the pattern (`the backend
  process died`), in the same way.

## context

```
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/tests/**
crates/es-policy/src/torch_runtime.rs
crates/es-policy/src/runtime.rs
crates/es-policy/tests/**
docs/packets/M12/P-M12-R6-backend-last-words.md
docs/packets/M12/P-M12-R6-backend-last-words.ko.md
```

## oracle

1. A test that spawns a stand-in process (the platform's own interpreter, as the existing proc
   tests do, or `ES_PYTHON` when set) which prints a protocol error line and exits 1 before the
   first request: `call` reports the decoded error text and the exit code, not os error 232.
2. A test whose stand-in writes > 1 MB to stderr and then answers: no deadlock, answer read.
3. The existing backend and policy tests pass; with `ES_PYTHON` set the MuJoCo ones run.
4. fmt, clippy `-D warnings`, `check-scope`.

## forbidden

Changing the protocol or `mujoco_ref.py`'s behaviour; any hash change; connecting to any remote
server.
