# M5 V1b — LeRobot v3.0 export

Design note: `docs/design/visible-learning.md` section 7.7; read section 7.5 first — V1's oracle
measured the refusal this packet closes. Depends on V1 (`9e18236`).

`es loop collect` writes `codebase_version: "v2.1"`, and `lerobot` 0.6.1 raises
`BackwardCompatibilityError` on it before looking at a single byte of our parquet. The v2.1 writer
is **deliberately unchanged** — V2's training script reads it directly — so this packet adds a
converter, `es dataset export --lerobot-v3 <root> --out <dir>`, whose output the installed
`lerobot` opens.

## context

```
crates/es-data/src/lerobot/v3.rs
crates/es-data/src/lerobot/mod.rs
crates/es-data/src/lib.rs
crates/es-data/tests/lerobot_v3.rs
crates/es-data/python/lerobot_read_ref.py
crates/es/src/cmd/dataset.rs
crates/es/tests/cli.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M5/V1b-lerobot-v3-export.md
docs/packets/M5/V1b-lerobot-v3-export.ko.md
```

Notes: `v3.rs` is new and carries the whole converter — the v3.0 schema, the parquet writers and a
PNG encoder; `mod.rs` and `lib.rs` get the module and a re-export only. `columns.rs` and `meta.rs`
are **not** touched: v2.1 is what they describe. `lerobot_read_ref.py` gains flattened per-frame
columns and the image shape so the oracle can check every frame instead of the two ends.
`dataset.rs` gains the `export` subcommand next to the existing `info`.

## spec

- §1.4: the oracle is the `lerobot` package opening the directory we wrote and handing back
  frames, not our own reader agreeing with itself. Pass/fail is a command.
- §1.9: LeRobot compatibility is a never-cut item; V1 measured that we currently have none against
  0.6.1, and this is the repair.
- §2.5: the converter is Rust and runs without Python. The interpreter appears only in the oracle.
- §19.1: the on-disk format is `lerobot` 0.6.1's v3.0 — `meta/info.json`, `meta/tasks.parquet`,
  `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`,
  `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`. Field by field in
  `docs/api-notes/lerobot-dataset.md`, "LeRobot v3.0".
- §19.2: `dataset_hash = H(content, schema, split)`. The export is a **derived artifact**: it must
  record the source dataset's `content_hash` and `schema_hash` so provenance survives the copy.
  `meta/es_provenance.json` carries them, because `lerobot`'s own `DatasetInfo.from_dict` drops
  unknown `info.json` keys with a warning and would silently lose them on any rewrite.
- §25.1: the export is written, never read back into a trust decision; the Python oracle reads it
  out of process.
- §3.4: no float time accumulation — `timestamp` is copied from the source, not re-derived.
- §1.5: `es-data` is at 3,515 code lines; this packet is budgeted under ~450.

## oracle

```
cargo fmt --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data --test lerobot_v3
cargo test -p es --test cli dataset_export
cargo xtask context-budget
cargo xtask check-spec-refs
```

The oracle that needs an interpreter:

```
ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python \
  cargo test -p es-data --test lerobot_v3 -- --nocapture
```

1. **`lerobot_v3_export` — the real package reads the export.** The test writes a V1-style v2.1
   dataset (two episodes, `observation.state`, `action`, `reward`, `action_source`,
   `intervention`, and one `observation.images.*` camera) plus the raw frames V0b's renderer
   dumps, runs the converter, and hands the result to
   `crates/es-data/python/lerobot_read_ref.py`, which opens it with `LeRobotDataset(root=...)`
   and iterates **every** frame. The test then asserts, against the columns it wrote:
   the episode count, the total frame count, the per-episode lengths, every
   `observation.state` and `action` value (flattened, frame by frame), and the camera tensor's
   shape and pixel sum. No `ES_LEROBOT_PYTHON`, no `lerobot`, or a `lerobot` that refuses →
   `SKIP lerobot_v3_export: <why>`; ran → `RAN lerobot_v3_export: <json>`. A refusal is a
   failure, not a skip: the skip path covers a missing interpreter, never a rejected dataset.
2. **`png_round_trips`** — the PNG encoder writes stored-deflate IDAT; the test parses its own
   output back (magic, IHDR, the stored blocks, the Adler-32 and every chunk CRC) and compares
   the pixels with the input. This is the check that fails if the encoder breaks without an
   interpreter anywhere.
3. **`export_layout_is_v3`** — the converter's output has `meta/info.json` with
   `codebase_version: "v3.0"`, the four path templates, `features` carrying the five reserved
   bookkeeping columns and the camera as `dtype: "image"`, and the three parquet files exist and
   are non-empty.
4. **`provenance_records_the_source_identity`** — `meta/es_provenance.json`'s `content` and
   `schema` equal `DatasetIdentity::compute` on the source dataset (§19.2).
5. **`images_without_frames_are_dropped_not_dangled`** — exporting a dataset that declares a
   camera with no `--frames` directory drops the feature and reports it, rather than writing an
   `info.json` that promises pixels nobody can load.
6. `crates/es/tests/cli.rs`: `dataset_export_writes_a_v3_dataset` — `es dataset export
   --lerobot-v3 <root> --out <dir>` exits 0 and prints the episode and frame counts;
   a missing `--out` is exit 2 (usage), a missing root is exit 1 (runtime).

## acceptance

```rust
// crates/es-data/src/lerobot/v3.rs
/// What the converter produced, for the CLI to print.
pub struct ExportReport {
    pub episodes: u32,
    pub frames: u64,
    /// Camera keys written as `dtype: "image"`.
    pub cameras: Vec<String>,
    /// Camera keys dropped — a declared camera with no frames behind it.
    pub dropped: Vec<String>,
}

/// Converts a v2.1 dataset to `lerobot` 0.6.1's v3.0 layout (api-note "LeRobot v3.0").
///
/// `frames` is the root of the raw frame dump `es_env::render::EnvRenderer` writes, one
/// subdirectory per camera named after the suffix of `observation.images.<name>`, each holding
/// `<NNNNNN>.bin` in dataset-global frame order. `None` drops every image feature.
pub fn export_v3(
    src: &LeRobotDataset,
    out: &Path,
    frames: Option<&Path>,
) -> Result<ExportReport, DataError>;
```

- Image storage is **PNG bytes embedded in the data parquet** as
  `struct<bytes: binary, path: string>` under `dtype: "image"` — the one storage `lerobot` 0.6.1
  decodes with neither `torchcodec` (which does not load on the oracle server) nor `ffmpeg`.
  No mp4 is written and no video encoder is called, from Rust or from Python.
- One `data/chunk-000/file-000.parquet` holding every episode, and one
  `meta/episodes/chunk-000/file-000.parquet`. Chunk/file splitting at
  `data_files_size_in_mb` is not implemented; the reader globs `data/*/*.parquet` and does not
  care, and the ceiling is marked in the source.
- No new external dependency: the same `parquet 59.3` low-level column API `columns.rs` already
  uses, and a PNG encoder in ~70 lines (stored-deflate, CRC-32, Adler-32).
- No new trait (INV-17), no `HashMap`, no float time, ≤ ~450 source lines.

## forbidden

- `crates/es-data/src/lerobot/columns.rs`, `crates/es-data/src/lerobot/meta.rs` — the v2.1
  reader/writer stays exactly as it is, `codebase_version` included. V2's training script reads
  it.
- `crates/es-data/src/collect.rs` — `es loop collect` keeps writing v2.1.
- `crates/es-policy`, `crates/es-eval`, `crates/es-env`, `crates/es/src/cmd/eval.rs` — V2 and V3.
- Training scripts and anything under `scripts/` or `python/` other than `lerobot_read_ref.py`.
- Adding `arrow`, a compression codec, an image crate or a deflate crate to `es-data`.
- A Rust or Python re-implementation of LeRobot's reader in place of the real package.
- Writing `meta/stats.json` with invented numbers: absent is honest (`load_stats` returns `None`),
  wrong is not.

## as built

Measured on the oracle server, 2026-09-15. Design note section 7.7 has the reasoning; this is
what changed against the packet above.

**Oracle results.** `lerobot_v3_export`: **RAN**. `LeRobotDataset(repo_id=..., root=<export>)`
opened it, reported `codebase_version 3.0`, **2 episodes / 7 frames**, iterated all 7
(`"iterated": 7`), and returned `observation.state`
`[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 1.0, ..., 3.0]` and
`action` `[0.0, -0.5, ..., -5.5, 1.0, 0.5, ..., -3.0]` — equal to the parquet we wrote, frame by
frame, to 1e-5. The camera came back as `[3, 4, 6]` (CHW) with pixel sum **111.67059516906738**
against our own 28476/255 = **111.670588**. `png_round_trips`, `png_spans_several_stored_blocks`,
`export_layout_is_v3`, `provenance_records_the_source_identity`,
`images_without_frames_are_dropped_not_dangled` and `dataset_export_writes_a_v3_dataset` all pass
with no interpreter.

**Deviations from the plan above**, each forced by something the package actually does:

- `ExportReport` carries no reason string per dropped camera — the only reason is "no frames
  behind it", and the CLI prints it once per camera rather than storing it.
- `export_v3` refuses a dataset with no `meta/tasks.jsonl` and a non-integer `fps`, rather than
  guessing: `total_tasks: 0` makes `lerobot` skip `meta/tasks.parquet` and then raise inside
  `get_item`, and `DatasetInfo` declares `fps: int`.
- The PNG encoder gained a second unit test (`png_spans_several_stored_blocks`) because a
  96x96 frame is 27 KB and a 160x160 one crosses the 65535-byte stored-block boundary the first
  test never reached.
- `crates/es-data/python/lerobot_read_ref.py` (shared with the v2.1 oracle) gained
  `iterated`, `states_flat`, `actions_flat`, `cameras`, `image_shape`, `image_sum` and `tasks`.
  Everything it prints is flat on purpose: the Rust side scans the JSON without a parser.
- `es dataset export` takes `--frames <dir>`, which the packet's signature implied but the CLI
  line did not spell out.
- **The budget was wrong: ~450 estimated, 702 actual** (`es-data` 3,515 -> 4,154, `es` 3,596 ->
  3,659; both still far under the §1.5 cap). The estimate did not price three parquet writers
  with hand-built schemas, and `columns.rs` being `forbidden` means ~35 lines of schema
  builders are duplicated rather than shared. Folding them together is a refactor for whoever
  owns both files next, not incidental work here.

**Human questions.**

1. `meta/stats.json` is not written, so a `lerobot`-side *training* run off an export would
   have no normalization statistics. Loading is unaffected. If someone wants to train through
   `lerobot` rather than through V2's script, that is a packet.
2. `meta/tasks.parquet` depends on the `pandas` footer metadata convention, which is pandas'
   serialization detail rather than a documented LeRobot format field. It is pinned to
   `pandas` 2.3.3 / `pyarrow` 25.0.1 and verified by the oracle; a pandas major version could
   move it.
3. Nothing yet writes the `--frames` directory from `es loop collect` — V1 left the renderer
   out of the CLI path. Until that is wired, an export of a real demonstration set drops its
   camera.
