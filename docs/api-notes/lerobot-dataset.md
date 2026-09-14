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
