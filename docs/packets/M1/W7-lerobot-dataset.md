# W7 — LeRobot dataset read/write + dataset identity

Spec: §19.1 (포맷), §19.2 (Dataset Identity), §19.3 (Training Identity), §5.3
(`dataset_hash = H(content, schema, split)`), §13.2 (개입 데이터의 지위), §8.8, §2.5
(읽기는 Rust 네이티브), §28.4 W7.

## context

```
crates/es-data/Cargo.toml
crates/es-data/src/lib.rs
crates/es-data/src/identity.rs
crates/es-data/src/lerobot/mod.rs
crates/es-data/src/lerobot/meta.rs
crates/es-data/src/lerobot/columns.rs
crates/es-data/tests/lerobot.rs
docs/api-notes/lerobot-dataset.md
docs/packets/M1/W7-lerobot-dataset.md
```

## spec

**Reading is Rust-native (§2.5): no Python at runtime, on any path in this packet.**

1. `docs/api-notes/lerobot-dataset.md` is written *first* and every field the author could
   not check against a pinned `lerobot` install is marked `unverified` (§1.7). No `lerobot`
   version is pinned in this workspace, so essentially the whole format description is
   unverified and says so at the top.

2. `LeRobotDataset::open(root) -> Result<Self, DataError>` parses `meta/info.json`,
   `meta/episodes.jsonl` and `meta/tasks.jsonl` and exposes `info()`, `features() ->
   &BTreeMap<String, FeatureSpec>`, `episodes() -> &[EpisodeMeta]`, `tasks()`.
   `FeatureSpec::port_type() -> es_ir::PortType` maps the dataset schema onto the spec §5.4
   type system; the mapping and its three lossy edges are tabulated in the api-note.

3. `read_episode(i) -> Result<Episode, DataError>` reads one parquet file through the
   low-level `parquet` column API. `Episode` holds `timestamps`, `task_index`, columnar
   `BTreeMap<String, Column>` (`F32`/`F64`/`I64`/`Bool`, flat row-major, `length *
   elem_count` values) and `video: BTreeMap<String, Vec<VideoRef>>`. **No video is decoded**
   — `VideoRef { path, frame_index }` records where the frame lives; decoding is a later
   packet. `frame_index`, `episode_index` and `index` are pure functions of position and are
   derived on write / discarded on read.

4. `LeRobotWriter::create(root, info)` + `write_episode(&Episode)` + `finish()` produce the
   same layout (`data/chunk-XXX/episode_XXXXXX.parquet`, `meta/*.jsonl`, `meta/info.json`
   with totals recomputed), such that `open(write(x))` yields back `x` for every non-video
   feature. No mp4 is written.

5. `DatasetIdentity` (§19.2):
   - `content` = blake3 over the episode parquet bytes in episode index order, **streamed**
     (`blake3::Hasher::update_reader`), never loading a dataset into memory;
   - `schema` = `es_ir::CanonWriter` over `codebase_version`, `fps` and the features sorted
     by name (name, dtype, shape, `names`);
   - `split` = `CanonWriter` over `Split { train, val, test }` episode index lists;
   - `to_dataset_hash() -> es_ir::DatasetHash` feeds the §5.3 chain.
   `Split::deterministic(n_episodes, fractions, seed)` is a splitmix64-seeded Fisher-Yates
   shuffle — no global RNG (§3.4) — and partitions `0..n` exactly.

6. `TrainingIdentity` (§19.3) carries the twelve artifacts of the §19.3 listing as digests
   (plus `BaseModel { source, hash, license }`, called out by the spec as the one that must
   be in provenance), hashed with `CanonWriter`; `policy_hash(checkpoint)` follows.

## oracle

```
cargo fmt -p es-data --check
cargo clippy -p es-data --all-targets -- -D warnings
cargo test -p es-data
cargo xtask layering
cargo xtask context-budget
```

## acceptance

- A synthetic 3-episode dataset (2 state features + 1 action + 1 video feature, differing
  episode lengths) written by `LeRobotWriter` and read back by `LeRobotDataset` compares
  equal episode by episode, including video refs and task indices.
- Changing one sample value changes `content_hash` and leaves `schema_hash` unchanged;
  adding a feature changes `schema_hash`.
- Changing the split changes `split_hash` and leaves the other two unchanged.
- `Split::deterministic` is stable for a seed (proptest) and its three lists are a partition
  of `0..n` with no overlap and no loss (proptest).
- A malformed `meta/info.json` and a missing `meta/` both surface as typed `DataError`
  variants, not panics.
- `es-data` stays well under the §1.5 budget and adds no new trait (INV-17) and no pickle
  path (INV-16).

## forbidden

- No Python, at build time or runtime.
- No video decoding, no mp4 writing, no `ImageSpec` invention (see the api-note).
- No new extension-point trait (INV-17); no `HashMap`/`HashSet` (§3.4).
- No edits to the root `Cargo.toml`, to `crates/es-safety`, `crates/es-compile`,
  `crates/es-env` (neighbouring packets own them), or to any crate below layer 10.
- Do not enable `parquet`'s `arrow` feature or any compression codec without a measured
  reason; the compile-time budget is the point.
- Do not re-derive `DatasetHash` locally — it lives in `es-ir` (§5.3) and is imported.
