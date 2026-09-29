# M12 R1 — `es dataset export --lerobot-v3` writes one row group per episode

Found by Y-V item 1 on 2026-09-29, the first real run of the camera-only cube template on this
Windows PC (`lerobot 0.6.1`, `datasets 4.8.5`, `pyarrow 25.0.1`). Spec §19.2 (the export is a
derived artifact); api-note `docs/api-notes/lerobot-dataset.md`; design note
`docs/design/visible-learning.md` 7 (the V1b export section).

## the defect

`lerobot-train` refused the export of 200 demonstrations (103,881 frames):
`pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked
array outputs` (`datasets/packaged_modules/parquet/parquet.py` → `pyarrow._dataset`).

Root cause, measured: `crates/es-data/src/lerobot/v3.rs` `write_data` writes the whole dataset as
**one row group** (`writer.next_row_group()` once). The image column is
`struct<bytes: binary, path: string>` holding stored-deflate PNGs (≈ the raw 96×96×3 frame), so
that single row group is 2.9 GB, and pyarrow cannot materialise a nested column past 2 GB from
one row group — even `pq.read_table` fails on the file. Rewriting the same rows in row groups of
1,000 (`ParquetFile.iter_batches` → `ParquetWriter`) makes `LeRobotDataset` load all 103,881
frames. The api-note's "one row group for the whole file … a random-access optimisation, not a
requirement" is wrong at this size; LeRobot's own writer writes one row group per episode
(`io_utils.py`, `write_table_one_row_group_per_episode`).

## spec

- `write_data` writes **one row group per episode**, in episode order, each holding exactly that
  episode's rows. Column order, types, definition levels, values, `path = null` on image rows,
  and `meta/info.json` are unchanged; only the row-group boundaries move.
- The per-column statistics (`meta/episodes` stats, the returned map) stay computed over **all**
  rows, byte-identical to today's.
- Nothing else changes: one data file, no PNG compression change, no chunk/file splitting.

## context

```
crates/es-data/src/lerobot/v3.rs
crates/es-data/tests/lerobot_v3.rs
docs/api-notes/lerobot-dataset.md
docs/api-notes/lerobot-dataset.ko.md
docs/design/visible-learning.md
docs/design/visible-learning.ko.md
docs/packets/M12/P-M12-R1-v3-row-groups.md
docs/packets/M12/P-M12-R1-v3-row-groups.ko.md
```

## oracle

1. A failing test first, in `crates/es-data/tests/lerobot_v3.rs`: export a fixture of at least
   three episodes of different lengths; the data file has `num_row_groups == episodes`, row group
   `i` has `rows == length(episode i)`, and reading every row back (all row groups, in order)
   gives exactly the rows today's export gives (compare against the values the existing tests
   already assert, not against a new golden).
2. The existing `lerobot_v3` tests and `cargo test -p es-data` pass unchanged; `cargo test -p es
   --test cli` export tests pass (Python-gated ones skip or run).
3. With `ES_PYTHON` set (this machine now has `.venv`: lerobot 0.6.1): the existing Python-gated
   LeRobot read oracle, if there is one, runs instead of skipping.
4. fmt, clippy `-D warnings`, `cargo xtask verify-goldens`, `check-scope`.

## acceptance

Oracles 1–4; the api-note and design-note sentences corrected (+ko), saying what was measured.

## forbidden

Changing the image encoding, the schema, `info.json` or the statistics; splitting into several
files; any change outside the context; connecting to any remote server.
