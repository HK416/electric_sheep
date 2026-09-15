# LeRobot dataset format — on-disk layout

**Pinned version: NONE.** Nothing in this workspace pins `lerobot`; no Python package was
installed and no real dataset was read while writing this file. Every statement below is
reconstructed from memory of LeRobot `v2.1` (`codebase_version: "v2.1"`) and is therefore
marked `unverified` unless a line says otherwise. Spec §1.7 names exactly this failure mode
(환각 API): **a human must pin a `lerobot` version and correct this file before anything
here is treated as ground truth.**

Only what `crates/es-data` reads and writes is described. Video decoding, streaming and
LeRobotDataset `v3` multi-episode packing (spec §19.1) are out of scope for W7.

## Directory layout — `unverified`

```
<root>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── stats.json                  (v2.0)  or
│   └── episodes_stats.jsonl        (v2.1)
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── observation.images.<camera>/
            └── episode_000000.mp4
```

`es-data` reads and writes `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl` and
the `data/` parquet files. `meta/stats.json` / `meta/episodes_stats.jsonl` are **ignored**
(read: skipped; write: not produced) — normalization statistics belong to the Observation IR
(spec §7), not to dataset identity, and including a float aggregate in `content_hash` would
make identity depend on a summary rather than on the samples.

## `meta/info.json` — `unverified`

| field | type | note |
|---|---|---|
| `codebase_version` | string | e.g. `"v2.1"`. Enters `schema_hash`. |
| `robot_type` | string \| null | optional; not hashed |
| `fps` | number | integer in practice; read as `f64`. Enters `schema_hash`. |
| `total_episodes` | int | |
| `total_frames` | int | |
| `total_tasks` | int | |
| `total_videos` | int | `unverified`, may be absent; not used |
| `total_chunks` | int | |
| `chunks_size` | int | episodes per chunk directory, typically `1000` |
| `data_path` | string | template, see below |
| `video_path` | string \| null | template; absent for video-less datasets |
| `features` | object | `name -> {dtype, shape, names}` |
| `splits` | object | `unverified`, e.g. `{"train": "0:100"}`. **Not read** — `es-data` owns splits (spec §19.2) |

Unknown fields are preserved on read and written back out unchanged (`Info::extra`), so a
round-trip through `es-data` does not silently drop keys this note got wrong.

### Path templates — `unverified`

```
data_path  = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
```

Placeholders are Python `str.format` fields. `es-data` supports exactly
`{episode_chunk}`, `{episode_index}`, `{video_key}`, each with an optional `:0Nd` zero-pad
spec; anything else is a `DataError::Unsupported`. `episode_chunk = episode_index /
chunks_size` (integer division).

### `features` entries — `unverified`

```json
"observation.state": { "dtype": "float32", "shape": [6], "names": ["shoulder_pan", "..."] }
"action":            { "dtype": "float32", "shape": [6], "names": [...] }
"observation.images.top": {
  "dtype": "video", "shape": [480, 640, 3],
  "names": ["height", "width", "channels"],
  "info": { "video.fps": 30.0, "video.codec": "av1", "...": "..." }
}
"timestamp":     { "dtype": "float32", "shape": [1], "names": null }
"frame_index":   { "dtype": "int64",   "shape": [1], "names": null }
"episode_index": { "dtype": "int64",   "shape": [1], "names": null }
"index":         { "dtype": "int64",   "shape": [1], "names": null }
"task_index":    { "dtype": "int64",   "shape": [1], "names": null }
```

`dtype` values seen: `float32`, `float64`, `int64`, `bool`, `string`, `video`, `image`.
`es-data` supports `float32` / `float64` / `int64` / `bool` as parquet columns and
`video` / `image` as video-only features with no parquet column; `string` is
`DataError::Unsupported`. The `info` sub-object on video features is `unverified` in both
presence and key spelling; it is carried through verbatim and not hashed beyond its JSON.

**Especially uncertain (`unverified`):** whether the five bookkeeping features
(`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`) appear in `features`
at all, and whether `timestamp` is `float32` or `float64`. `es-data` treats these five
names as *reserved*: they are never written as list columns and never appear in
`Episode::columns` (see "Column shapes" below).

## `meta/episodes.jsonl` — `unverified`

One JSON object per line, in ascending `episode_index`:

```json
{"episode_index": 0, "tasks": ["pick up the cube"], "length": 120}
```

## `meta/tasks.jsonl` — `unverified`

```json
{"task_index": 0, "task": "pick up the cube"}
```

`task_index` is the value of the parquet `task_index` column. `es-data` assigns indices in
first-seen order across `write_episode` calls; it does **not** verify that a caller's
`task_index` column agrees with its `Episode::tasks` strings.

## Parquet files — partly verified

Verified against `parquet 59.3.0` (the crate, not against a real LeRobot file): the physical
encodings below are what `es-data` writes and what it accepts on read. That a genuine
LeRobot file uses these encodings is `unverified`.

### Column shapes

- A feature with `shape` `[d]` (or any shape whose element count is `d`) is a **3-level
  list**, one list per frame, `d` items per list:

  ```
  optional group <feature name> (LIST) {
    repeated group list {
      optional <float|double|int64|boolean> item;
    }
  }
  ```

  `max_def_level = 3`, `max_rep_level = 1`. `es-data` writes every item non-null
  (`def = 3`) and rejects a file whose lists are ragged or contain nulls.

- The five reserved names are **plain (non-list) optional primitives**, `max_def_level = 1`:
  `timestamp` (`float`/`double`), `frame_index`, `episode_index`, `index`, `task_index`
  (`int64`). This is `unverified` — LeRobot may write them as length-1 lists like every
  other feature; if the human's pinned version disagrees, only
  `crates/es-data/src/lerobot/columns.rs` changes.

- `index` is the dataset-global frame counter and `episode_index` is constant within a file;
  both are **derived** on write (`index = frames_written_so_far + frame`) and **discarded**
  on read, so they never enter `Episode`. `frame_index` is likewise derived as `0..length`.
  Round-trip equality is therefore over `timestamps`, `task_index`, `columns` and the video
  refs — everything that is not a pure function of position.

- Column *order* in the file is not significant: on read, leaves are located by
  `ColumnDescriptor::path().string()`, not by index. `es-data` writes one row group per
  episode with columns in `BTreeMap` order.

- Compression is `UNCOMPRESSED` (parquet's default `WriterProperties`). Reading a compressed
  LeRobot file needs the relevant `parquet` codec feature turned on in
  `crates/es-data/Cargo.toml`; none is enabled today, so a snappy/zstd-compressed real file
  will fail to read. `unverified` which codec LeRobot actually uses.

## `videos/` — not read

No video file is opened, decoded or written in W7. A frame's image feature is surfaced as
`VideoRef { path, frame_index }`, where `path` is the `video_path` template rendered for the
episode and camera, relative to the dataset root, and `frame_index` is the frame's position
within the episode. Whether one mp4 holds exactly one episode's frames starting at 0 is
`unverified` (it is what the `video_path` template implies).

## Mapping to `es_ir::PortType` (spec §5.4)

`FeatureSpec::port_type()`:

| feature | `elem` | `shape` | `unit` | `frame` | `time` | `image` |
|---|---|---|---|---|---|---|
| `float32` | `F32` | feature `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `float64` | `F64` | feature `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `int64` | `I32` | feature `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `bool` | `Bool` | feature `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `video` / `image` | `U8` | feature `shape` | `Pixel` | `Policy` | `Tick` | `None` |

Three deliberate lossy edges, all forced by what the format does *not* record:

- **`int64 -> ElemType::I32`.** `ElemType` (spec §5.4) has no 64-bit integer. The in-memory
  `Column::I64` keeps full width; only the advertised `PortType` narrows.
- **`Frame::Policy`.** A LeRobot dataset records no coordinate frame for a state or action
  vector, and `Frame::Policy` is the spec's "no frame checking" frame. Inventing
  `Frame::World` would assert something the file does not say.
- **`image: None`.** `ImageSpec` (resolution is known, but color space, camera model,
  intrinsics, extrinsics, distortion, shutter and exposure are not) cannot be filled from
  `info.json`. Spec §19.1 requires `ImageSpec` *in the dataset*; LeRobot v2.1 has no place
  to put it, so recovering it is a later packet (an Electric Sheep sidecar, or the
  `features[..].info` object once a real file pins its schema).

`Unit::Dimensionless` rather than `Unit::Angle`/`Unit::Length`: the `names` list
(`"shoulder_pan"`, …) is not a unit declaration, and `is_policy_input()` accepts
`Dimensionless`, which is what these tensors are used as.

## Measured against the real package, 2026-09-15 — `verified`

`ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python cargo test -p es-data --test
lerobot_oracle -- --nocapture`, on the oracle server (packet `docs/packets/M5/V1`), against a
dataset this crate wrote (`observation.state`, `action`, `reward`, `action_source`,
`intervention`; two episodes; no video feature):

| Thing | Result |
|---|---|
| `lerobot` | 0.6.1, with `datasets` 4.8.5 and `pyarrow` 25.0.1 — the `[dataset]` extra **is** installed now |
| `import lerobot.datasets.lerobot_dataset` | works |
| `LeRobotDataset(repo_id=..., root=<ours>)` | **refuses**: `BackwardCompatibilityError: The dataset you requested is in 2.1 format. We introduced a new format since v3.0 which is not backward compatible with v2.1.` |
| `lerobot.datasets.dataset_metadata.CODEBASE_VERSION` | `"v3.0"` |

So the finding the packet asked for is a refusal, and it is decisive for design-note open
question 2: **`lerobot` 0.6.1 does not read `codebase_version: "v2.1"` at all**, so nothing
about our parquet, our `meta/*.jsonl` or our column dtypes was exercised — the version gate
fires first. Nothing here upgrades that verdict for the rest of this file: every `unverified`
above is still unverified.

What this does *not* say: that our v2.1 writer is wrong. v2.1 is a format `lerobot` itself
shipped; 0.6.1 simply dropped backward compatibility and offers
`python -m lerobot.scripts.convert_dataset_v21_to_v30` for datasets on the hub. The choice
between writing v3.0 directly and shipping a converter is a packet of its own (spec 19.1), and
`crates/es-data/src/lerobot/meta.rs`'s `codebase_version` is deliberately left alone here.

Until then `crates/es-data/tests/lerobot_oracle.rs` prints `SKIP lerobot_oracle: <why>` with
that refusal as the reason, on every machine.

---

# LeRobot v3.0 — on-disk layout — `verified`

**Pinned version: `lerobot` 0.6.1**, with `datasets` 4.8.5, `pyarrow` 25.0.1, `pandas` 2.3.3,
`cv2` 4.13.0, on the oracle server. Read on 2026-09-15 from the installed package, packet
`docs/packets/M5/V1b-lerobot-v3-export.md`:

```
~/venvs/es-lerobot-cuda/lib/python3.12/site-packages/lerobot/datasets/
    utils.py             (path templates, DatasetInfo, chunk/file sizes)
    dataset_metadata.py  (CODEBASE_VERSION, image_keys/video_keys, load path)
    dataset_reader.py    (get_item, the hf_dataset features, the tasks lookup)
    io_utils.py          (load_nested_dataset, load/write tasks, episodes, stats)
    feature_utils.py     (get_hf_features_from_features)
    lerobot_dataset.py   (LeRobotDataset.__init__, the docstring layout)
~/venvs/es-lerobot-cuda/lib/python3.12/site-packages/lerobot/scripts/convert_dataset_v21_to_v30.py
```

There is no `lerobot/datasets/v30/` directory in 0.6.1; v3.0 *is* the format, and
`convert_dataset_v21_to_v30.py` is the upgrade path for datasets already on the hub.

Everything below was additionally **executed** against 0.6.1: a dataset in exactly this shape,
written with the physical encodings `crates/es-data/src/lerobot/v3.rs` uses, opens with
`LeRobotDataset(repo_id=..., root=...)` and yields frames.

## Directory layout

```
<root>/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── stats.json                        (optional)
│   ├── es_provenance.json                (ours, not LeRobot's -- see below)
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/                               (only for `dtype: "video"` features)
    └── <video_key>/
        └── chunk-000/
            └── file-000.mp4
```

Constants, `datasets/utils.py:88-107`:

| name | value |
|---|---|
| `DEFAULT_CHUNK_SIZE` | `1000` (max files per chunk directory) |
| `DEFAULT_DATA_FILE_SIZE_IN_MB` | `100` |
| `DEFAULT_VIDEO_FILE_SIZE_IN_MB` | `200` |
| `INFO_PATH` | `meta/info.json` |
| `STATS_PATH` | `meta/stats.json` |
| `DEFAULT_TASKS_PATH` | `meta/tasks.parquet` |
| `DEFAULT_EPISODES_PATH` | `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| `DEFAULT_DATA_PATH` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| `DEFAULT_VIDEO_PATH` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| `DEFAULT_IMAGE_PATH` | `images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png` |
| `CODEBASE_VERSION` | `"v3.0"` (`datasets/dataset_metadata.py:60`) |

The placeholders changed from v2.1: **there is no `{episode_chunk}` and no `{episode_index}` in
`data_path` or `video_path`.** A file holds many episodes, and which file an episode is in comes
out of `meta/episodes/*.parquet`, not out of arithmetic on the episode index.
`load_nested_dataset` (`io_utils.py:63-83`) simply globs `data/*/*.parquet`, so the chunk/file
numbering only has to be self-consistent with what the episodes table says.

`DEFAULT_IMAGE_PATH` is a *writer-side* staging path (`image_writer.py`); the reader never
resolves it. Image pixels reach the reader through the data parquet — see below.

## `meta/info.json`

Parsed by `DatasetInfo.from_dict` (`utils.py:114-196`). Unknown keys are **dropped with a
`logger.warning`**; missing optional keys take the dataclass default.

| field | type | required | note |
|---|---|---|---|
| `codebase_version` | string | yes | must parse as `3.0`; `2.1` raises `BackwardCompatibilityError`, `> 3.0` raises `ForwardCompatibilityError` |
| `fps` | int | yes | `__post_init__` rejects `<= 0` |
| `features` | object | yes | `name -> {dtype, shape, names}`; drives the whole parquet schema |
| `total_episodes` | int | no (0) | `0` means "do not load `meta/episodes`" |
| `total_frames` | int | no (0) | |
| `total_tasks` | int | no (0) | `0` means "do not load `meta/tasks.parquet`", and then `get_item` raises on `meta.tasks.iloc[...]` |
| `chunks_size` | int | no (1000) | must be `> 0` |
| `data_files_size_in_mb` | int | no (100) | must be `> 0` |
| `video_files_size_in_mb` | int | no (200) | must be `> 0` |
| `data_path` | string | no (default) | |
| `video_path` | string \| null | no (default) | `null` for a dataset with no `dtype: "video"` feature |
| `robot_type` | string \| null | no | |
| `splits` | object | no (`{}`) | |
| `tools` | list \| null | no | OpenAI-style tool schemas; omitted when unset |

`total_videos` and `total_chunks` — v2.1 fields — are *not* v3.0 fields and are dropped with a
warning.

### `features` -> the parquet schema

`DatasetInfo.__post_init__` coerces every `shape` from list to **tuple**, and
`get_hf_features_from_features` (`feature_utils.py:43-83`) then maps, in this order:

| condition | `datasets` feature | arrow type |
|---|---|---|
| `dtype == "video"` | *skipped* | no column at all |
| `dtype == "image"` | `datasets.Image()` | `struct<bytes: binary, path: string>` |
| `shape == (1,)` | `datasets.Value(dtype)` | plain scalar |
| `len(shape) == 1` | `datasets.List(Value(dtype), length=n)` | `fixed_size_list<item: T>[n]` |
| `len(shape) in 2..=5` | `Array2D`..`Array5D` | nested fixed-size lists |

Two consequences that bite:

- **A `shape: [1]` feature is a scalar column, not a length-1 list.** v2.1's writer emits every
  feature as a 3-level LIST; for v3.0 `reward`, `timestamp`, `frame_index`, `episode_index`,
  `index` and `task_index` must be plain primitives.
- The five bookkeeping columns **must appear in `features`**, because `features` is what
  `Dataset.from_parquet(..., features=...)` casts the file to. This was the open question v2.1
  left (`unverified` above); for v3.0 it is answered: they are required.

A parquet variable-size `list<item: T>` is accepted where `fixed_size_list<item: T>[n]` is
declared — `datasets` casts it — which is why the 3-level LIST `columns.rs` already writes is
reusable. Measured, not assumed.

### Image features without ffmpeg or torchcodec

`dtype: "image"` pixels live **inline in the data parquet** as
`struct<bytes: binary, path: string>`, where `bytes` is an encoded image file (PNG here) and
`path` is null. `hf_transform_to_torch` (`io_utils.py:266-293`) turns the PIL image into a
`float32` `(C, H, W)` tensor in `[0, 1]`. Nothing in that path touches `torchcodec`, `pyav` or
`ffmpeg`.

`dtype: "video"` is the other option and does need a decoder: `dataset_reader._query_videos`
calls `decode_video_frames`. On the oracle server `torchcodec` is installed but **cannot load**
(`libnppicc.so.12: cannot open shared object file`), so it falls back to `pyav`. `es dataset
export --lerobot-v3` therefore writes `image`, never `video`, and calls no encoder — not
`~/.local/bin/ffmpeg`, not from Rust, not from the Python side of the oracle.

The cost is size: a stored-deflate PNG is roughly the raw frame plus 0.1%.
`data_files_size_in_mb` is advisory (the reader globs), so a single large data file is legal;
splitting is an optimisation, not a correctness requirement.

## `meta/episodes/chunk-XXX/file-XXX.parquet`

Loaded by `load_episodes` (`io_utils.py:212-218`) with **no declared features** — the arrow
types are inferred from the file — then every `stats/*` column is dropped. Columns the reader
actually uses:

| column | type | used by |
|---|---|---|
| `episode_index` | int64 | `filter_episodes`, `_check_cached_episodes_sufficient` |
| `length` | int64 | `meta.episodes[i]["length"]` |
| `dataset_from_index` | int64 | `dataset_reader._get_query_indices` (delta-timestamp windows) |
| `dataset_to_index` | int64 | same; exclusive end |
| `tasks` | list\<string\> | episode-level task labels |
| `data/chunk_index` | int64 | `DatasetMetadata.get_data_file_path` |
| `data/file_index` | int64 | same |
| `meta/episodes/chunk_index` | int64 | the *writer*'s append path |
| `meta/episodes/file_index` | int64 | same |
| `videos/<key>/chunk_index`, `videos/<key>/file_index`, `videos/<key>/from_timestamp` | int64/float | only for `dtype: "video"` features |
| `stats/<feature>/<stat>` | — | optional per-episode statistics; dropped on load |

The column *names* contain `/`. They are flat top-level parquet fields whose name happens to
contain a slash, not nested groups.

`dataset_from_index` / `dataset_to_index` are cumulative over episodes in file order, and match
the `index` column in the data parquet.

## `meta/tasks.parquet`

`load_tasks` is `pd.read_parquet(...)` followed by `tasks.index.name = "task"`
(`io_utils.py:184-187`), and the reader resolves a frame's task with
`self._meta.tasks.iloc[task_idx].name` (`dataset_reader.py:352`) — i.e. **the task string must
be the pandas index, not a column value**. So the file carries two parquet columns,
`task_index` (int64) and `task` (string), *plus* the `pandas` key/value metadata in the footer
that tells pyarrow which one is the index:

```json
{"index_columns": ["task"],
 "column_indexes": [{"name": null, "field_name": null, "pandas_type": "unicode",
                     "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}],
 "columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64",
              "numpy_type": "int64", "metadata": null},
             {"name": "task", "field_name": "task", "pandas_type": "unicode",
              "numpy_type": "object", "metadata": null}],
 "pandas_version": "2.3.3"}
```

Row order must agree with `task_index`, because the lookup is positional (`iloc`).

## `meta/stats.json`

**Optional.** `load_stats` returns `None` when the file is absent (`io_utils.py:161-175`) and
nothing on the read path requires it. It is `{feature: {mean|std|min|max|count: [...]}}`,
consumed by training's normalization. `es dataset export --lerobot-v3` does not write it:
LeRobot compatibility is about the dataset being *readable*, and inventing statistics would be
worse than omitting them. If a human wants to train LeRobot-side off an export, that is the
follow-up.

## Parquet physical encodings that were accepted

Measured against 0.6.1 / pyarrow 25.0.1, writing with `parquet 59.3`'s low-level column API:

- `UNCOMPRESSED` pages (no codec feature enabled in `crates/es-data/Cargo.toml`).
- No `ARROW:schema` footer metadata. pyarrow infers arrow types from the parquet schema and
  `datasets` casts to the declared features.
- 3-level LIST (`optional group X (LIST) { repeated group list { optional T item; } }`) where a
  `fixed_size_list` is declared.
- `optional group X { optional byte_array bytes; optional byte_array path (String); }` for an
  image column, with `path` written as null on every row.
- Plain `optional` primitives for the five bookkeeping columns and for any `shape: [1]` feature.
- One row group for the whole file. `write_table_one_row_group_per_episode` is what LeRobot's
  own writer does (`io_utils.py:295-309`) and is a random-access optimisation, not a
  requirement.

## `meta/es_provenance.json` — ours, not LeRobot's

Spec §19.2 makes the export a derived artifact, so it records where it came from:

```json
{"source_root": "...", "source_codebase_version": "v2.1",
 "content": "<64 hex>", "schema": "<64 hex>", "split": "<64 hex>",
 "exported_by": "es dataset export --lerobot-v3"}
```

The three hashes are `DatasetIdentity::compute` over the source with a display-only all-train
split, exactly as `es dataset info` prints them. It is a sidecar rather than extra `info.json`
keys because `DatasetInfo.from_dict` drops unknown keys (with a warning) and `to_dict` would
not write them back — provenance in `info.json` would not survive a LeRobot-side rewrite.

## The columns `es loop collect` writes — ours, not LeRobot's

`observation.state`, `action` and `reward` are LeRobot's own names. Three more are Electric
Sheep's, declared in `features` like any other so that both the v2.1 writer and the v3.0
exporter carry them without a special case:

| Column | dtype | shape | Meaning |
|---|---|---|---|
| `action_source` | `int64` | `[1]` | `es_data::ActionSourceCode`: `Policy` / `Human` / `Clamped` / `Fallback` (spec §13.2) |
| `intervention` | `int64` | `[1]` | 1 where a human or a scripted intervener drove the tick |
| `action_commanded` | `float32` | `[nu]` | the pre-plane command of that tick — **packet M5/V1c** |

**`action` is the executed action, and `action_commanded` is what was asked for.** `action` is
the `SafetyPlane::validate` result that `DomainRunner::emit_actions` copied into `ctrl`: what
reached the actuator, after every clamp (`INV-12`). `action_commanded` is the row the chunk
buffer served for the same tick, before the plane judged it. The two are equal on a tick the
plane passed through unchanged, and on a tick the buffer had no row for at all — where there was
no command, and `action_source` reads `Fallback`.

A consumer training a policy wants `action`: it is the only column whose values the Deployment
IR's envelope will let that policy reproduce. `action_commanded` is provenance, so that a
`Clamped` frame can be read without re-running the plane.

**This moved `dataset_schema_hash`** (and therefore `content`, §19.1/§19.2): a dataset collected
before packet M5/V1c has no `action_commanded` column and hashes differently from one collected
after, even from the same seed. That is the intended behaviour — the schema is part of a training
run's input identity — and it is why V1c re-collects rather than patching the existing set. Both
columns cross `es dataset export --lerobot-v3` unchanged (`export_layout_is_v3`), and `lerobot`
0.6.1 reads a dataset that carries the extra feature (`lerobot_v3_export`).

### `observation.state` is `qpos ‖ qvel`, and packet M5/V7a depends on it

`es_data::collect::to_lerobot` writes the row as the **whole** of env 0's `qpos` followed by
the whole of its `qvel` — width `nq + nv`, never a joint subset. For the SO-101 demo scene that
is `13 + 12 = 25`: six arm hinges at `qpos[0..6]`, the cube's free joint at `qpos[6..13]`
(position `xyz` then quaternion `wxyz`), and the matching `qvel` behind them. `features`
declares it as one `float32` feature of that width, and both the v2.1 writer and the v3.0
exporter carry it with no special case.

Two consequences, both load-bearing:

* **A `qpos` `IndexRange` indexes the recorded row directly.** `es_eval::runner::capture` reads
  `StateView::qpos_of(0)[r]` at inference; `es_eval::ObservationBake` reads `row[r]` from the
  parquet. Same two bounds, no translation — the row is not a re-encoding of the state, it is
  the state with `qvel` appended. That is what lets the demo's privileged `sim_cube_pose`
  channel bake bit-identically to what inference serves
  (`a_baked_frame_is_bit_identical_to_what_capture_serves`).
* **`observation.state` already carried the cube's pose before V7a wanted it.** No column was
  added and `dataset_schema_hash` did not move for this packet; what moved is which slices of
  the row the Observation IR reads.

The `names` list in the v2.1 example above (`["shoulder_pan", ...]`) illustrates LeRobot's own
convention and is *not* what this writer emits for the demo: a 25-wide row of `qpos ‖ qvel` has
no one-name-per-joint reading.
