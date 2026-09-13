<!-- Korean translation of docs/packets/M1/W7-lerobot-dataset.md. The English file is the working copy; regenerate this when it changes. -->

# W7 — LeRobot 데이터셋 읽기/쓰기 + 데이터셋 아이덴티티

Spec: §19.1 (포맷), §19.2 (Dataset Identity), §19.3 (Training Identity), §5.3
(`dataset_hash = H(content, schema, split)`), §13.2 (개입 데이터의 지위), §8.8, §2.5
(읽기는 Rust 네이티브), §28.4 W7.

## context (범위)

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

## spec (사양)

**읽기는 Rust 네이티브다(§2.5): 이 패킷의 어떤 경로에서도 런타임에 Python을 쓰지 않는다.**

1. `docs/api-notes/lerobot-dataset.md`를 *먼저* 작성하고, 저자가 고정된 `lerobot` 설치본에 대해
   확인할 수 없었던 모든 필드는 `미검증 / unverified`로 표시한다(§1.7). 이 워크스페이스에는
   `lerobot` 버전이 고정되어 있지 않으므로, 사실상 포맷 설명 전체가 미검증이며 문서 맨 위에 그렇게
   명시한다.

2. `LeRobotDataset::open(root) -> Result<Self, DataError>`는 `meta/info.json`,
   `meta/episodes.jsonl`, `meta/tasks.jsonl`을 파싱하고 `info()`, `features() ->
   &BTreeMap<String, FeatureSpec>`, `episodes() -> &[EpisodeMeta]`, `tasks()`를 노출한다.
   `FeatureSpec::port_type() -> es_ir::PortType`은 데이터셋 스키마를 spec §5.4의 타입 시스템에
   매핑하며, 그 매핑과 세 가지 손실 지점은 api-note에 표로 정리되어 있다.

3. `read_episode(i) -> Result<Episode, DataError>`는 저수준 `parquet` 컬럼 API를 통해 parquet
   파일 하나를 읽는다. `Episode`는 `timestamps`, `task_index`, 컬럼 형태의 `BTreeMap<String,
   Column>`(`F32`/`F64`/`I64`/`Bool`, flat row-major, `length * elem_count`개 값)과
   `video: BTreeMap<String, Vec<VideoRef>>`를 갖는다. **비디오는 디코딩하지 않는다** —
   `VideoRef { path, frame_index }`는 프레임이 있는 위치만 기록하며, 디코딩은 이후 패킷의 몫이다.
   `frame_index`, `episode_index`, `index`는 위치의 순수 함수이며, 쓸 때 유도되고 읽을 때 버려진다.

4. `LeRobotWriter::create(root, info)` + `write_episode(&Episode)` + `finish()`는 동일한
   레이아웃을 생성하며(`data/chunk-XXX/episode_XXXXXX.parquet`, `meta/*.jsonl`, 합계를 재계산한
   `meta/info.json`), 그 결과 모든 비디오 아닌 피처에 대해 `open(write(x))`가 `x`를 그대로
   돌려준다. mp4는 기록하지 않는다.

5. `DatasetIdentity`(§19.2):
   - `content` = 에피소드 인덱스 순서로 나열한 에피소드 parquet 바이트에 대한 blake3 값이며,
     데이터셋을 메모리에 전부 올리지 않고 **스트리밍**으로 계산한다(`blake3::Hasher::update_reader`);
   - `schema` = `codebase_version`, `fps`, 그리고 이름순으로 정렬한 피처들(name, dtype, shape,
     `names`)에 대한 `es_ir::CanonWriter`;
   - `split` = `Split { train, val, test }` 에피소드 인덱스 리스트에 대한 `CanonWriter`;
   - `to_dataset_hash() -> es_ir::DatasetHash`가 §5.3의 해시 체인에 공급된다.
   `Split::deterministic(n_episodes, fractions, seed)`는 splitmix64로 시드를 부여한 Fisher-Yates
   셔플이며 — 전역 RNG를 쓰지 않고(§3.4) — `0..n`을 정확히 분할한다.

6. `TrainingIdentity`(§19.3)는 §19.3 목록의 열두 가지 산출물을 다이제스트 형태로 담으며(스펙이
   provenance에 반드시 들어가야 한다고 지목하는 `BaseModel { source, hash, license }`도 포함),
   `CanonWriter`로 해시한다. 이어서 `policy_hash(checkpoint)`가 뒤따른다.

## oracle (오라클)

```
cargo fmt -p es-data --check
cargo clippy -p es-data --all-targets -- -D warnings
cargo test -p es-data
cargo xtask layering
cargo xtask context-budget
```

## acceptance (수용 기준)

- `LeRobotWriter`로 기록하고 `LeRobotDataset`으로 다시 읽은 합성 3-에피소드 데이터셋(상태 피처
  2개 + 액션 1개 + 비디오 피처 1개, 에피소드 길이는 서로 다름)이 비디오 참조와 task 인덱스를
  포함해 에피소드별로 동일하게 비교된다.
- 샘플 값 하나를 바꾸면 `content_hash`가 바뀌고 `schema_hash`는 그대로 유지된다. 피처를 추가하면
  `schema_hash`가 바뀐다.
- 분할을 바꾸면 `split_hash`가 바뀌고 나머지 둘은 그대로 유지된다.
- `Split::deterministic`은 같은 시드에 대해 안정적이며(proptest), 세 리스트는 겹침도 누락도 없이
  `0..n`을 분할한다(proptest).
- 형식이 잘못된 `meta/info.json`과 `meta/`가 없는 경우 모두 panic이 아니라 타입이 지정된
  `DataError` variant로 드러난다.
- `es-data`는 §1.5 예산을 여유 있게 지키며, 새 trait을 추가하지 않고(INV-17) pickle 경로도
  추가하지 않는다(INV-16).

## forbidden (금지)

- 빌드 타임이든 런타임이든 Python을 쓰지 않는다.
- 비디오 디코딩, mp4 기록, `ImageSpec` 임의 생성을 하지 않는다(api-note 참고).
- 새 확장 지점 trait을 추가하지 않는다(INV-17). `HashMap`/`HashSet`을 쓰지 않는다(§3.4).
- 루트 `Cargo.toml`, `crates/es-safety`, `crates/es-compile`, `crates/es-env`(이웃 패킷들이
  소유)를 수정하지 않으며, 레이어 10 미만의 어떤 크레이트도 수정하지 않는다.
- 측정된 근거 없이 `parquet`의 `arrow` feature나 어떤 압축 코덱도 켜지 않는다. 컴파일 타임
  예산을 지키는 것이 핵심이다.
- `DatasetHash`를 로컬에서 다시 유도하지 않는다 — 이는 `es-ir`(§5.3)에 있으며 그곳에서 import한다.
