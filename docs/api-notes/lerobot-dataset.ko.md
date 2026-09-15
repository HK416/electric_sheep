<!-- Korean translation of docs/api-notes/lerobot-dataset.md. The English file is the working copy; regenerate this when it changes. -->

# LeRobot 데이터셋 포맷 — 디스크 상 레이아웃

**고정 버전: 없음(NONE).** 이 워크스페이스는 `lerobot`을 고정하지 않았다. 이 파일을 작성하는 동안 Python 패키지를
설치하지도, 실제 데이터셋을 읽지도 않았다. 아래의 모든 서술은 LeRobot `v2.1`(`codebase_version: "v2.1"`)에
대한 기억을 바탕으로 재구성한 것이며, 별도로 명시하지 않는 한 모두 `미검증 / unverified`로 표시한다. 스펙
§1.7은 바로 이 실패 양상을 지목한다(환각 API): **사람이 `lerobot` 버전을 고정하고 이 파일을 수정하기 전까지는
여기 적힌 어떤 내용도 사실로 취급해서는 안 된다.**

여기서는 `crates/es-data`가 읽고 쓰는 부분만 다룬다. 비디오 디코딩, 스트리밍, LeRobotDataset `v3`의
multi-episode packing(스펙 §19.1)은 W7의 범위 밖이다.

## 디렉터리 레이아웃 — `미검증 / unverified`

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

`es-data`는 `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`과 `data/` 아래의 parquet
파일을 읽고 쓴다. `meta/stats.json` / `meta/episodes_stats.jsonl`은 **무시한다**(읽기: 건너뜀; 쓰기:
생성하지 않음) — 정규화 통계는 데이터셋 아이덴티티가 아니라 Observation IR(스펙 §7)의 소관이며, 부동소수점
집계값을 `content_hash`에 포함시키면 아이덴티티가 샘플이 아니라 요약값에 종속되기 때문이다.

## `meta/info.json` — `미검증 / unverified`

| 필드 | 타입 | 비고 |
|---|---|---|
| `codebase_version` | string | 예: `"v2.1"`. `schema_hash`에 포함된다. |
| `robot_type` | string \| null | 선택적; 해시에 포함되지 않음 |
| `fps` | number | 실제로는 정수; `f64`로 읽는다. `schema_hash`에 포함된다. |
| `total_episodes` | int | |
| `total_frames` | int | |
| `total_tasks` | int | |
| `total_videos` | int | `미검증 / unverified`, 없을 수 있음; 사용하지 않음 |
| `total_chunks` | int | |
| `chunks_size` | int | 청크 디렉터리당 에피소드 수, 보통 `1000` |
| `data_path` | string | 템플릿, 아래 참고 |
| `video_path` | string \| null | 템플릿; 비디오가 없는 데이터셋에서는 없음 |
| `features` | object | `name -> {dtype, shape, names}` |
| `splits` | object | `미검증 / unverified`, 예: `{"train": "0:100"}`. **읽지 않음** — 분할은 `es-data`가 소유한다(스펙 §19.2) |

알 수 없는 필드는 읽을 때 보존되고 쓸 때 변경 없이 그대로 다시 기록되므로(`Info::extra`), `es-data`를
왕복해도 이 문서가 잘못 파악한 키가 조용히 유실되지는 않는다.

### 경로 템플릿 — `미검증 / unverified`

```
data_path  = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
```

플레이스홀더는 Python `str.format` 필드다. `es-data`가 지원하는 것은 정확히 `{episode_chunk}`,
`{episode_index}`, `{video_key}`뿐이며, 각각 선택적으로 `:0Nd` 제로 패딩 지정을 붙일 수 있다. 그 밖의
것은 `DataError::Unsupported`가 된다. `episode_chunk = episode_index / chunks_size`(정수 나눗셈).

### `features` 엔트리 — `미검증 / unverified`

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

관측된 `dtype` 값: `float32`, `float64`, `int64`, `bool`, `string`, `video`, `image`. `es-data`는
`float32` / `float64` / `int64` / `bool`을 parquet 컬럼으로 지원하고, `video` / `image`는 parquet
컬럼이 없는 video-only 피처로 지원한다. `string`은 `DataError::Unsupported`다. 비디오 피처의 `info`
하위 객체는 존재 여부와 키 표기 모두 `미검증 / unverified`다. 그대로 보존되며 JSON 형태 이상으로 해시되지
않는다.

**특히 불확실한 부분(`미검증 / unverified`):** 다섯 개의 부기용 피처(`timestamp`, `frame_index`,
`episode_index`, `index`, `task_index`)가 애초에 `features`에 나타나는지 여부, 그리고 `timestamp`가
`float32`인지 `float64`인지. `es-data`는 이 다섯 이름을 *예약됨*으로 취급한다: 이들은 리스트 컬럼으로
기록되지 않고 `Episode::columns`에도 나타나지 않는다(아래 "컬럼 형태" 참고).

## `meta/episodes.jsonl` — `미검증 / unverified`

한 줄에 JSON 객체 하나씩, `episode_index` 오름차순으로 나열한다:

```json
{"episode_index": 0, "tasks": ["pick up the cube"], "length": 120}
```

## `meta/tasks.jsonl` — `미검증 / unverified`

```json
{"task_index": 0, "task": "pick up the cube"}
```

`task_index`는 parquet의 `task_index` 컬럼 값이다. `es-data`는 `write_episode` 호출에 걸쳐 처음
등장한 순서대로 인덱스를 부여한다. 호출자가 넘긴 `task_index` 컬럼이 그 `Episode::tasks` 문자열과
일치하는지는 **검증하지 않는다**.

## Parquet 파일 — 부분 검증

`parquet 59.3.0`(크레이트 기준이며, 실제 LeRobot 파일 기준은 아니다)에 대해 검증했다: 아래의 물리적
인코딩은 `es-data`가 기록하는 방식이자 읽을 때 받아들이는 방식이다. 실제 LeRobot 파일이 이 인코딩을
사용하는지는 `미검증 / unverified`다.

### 컬럼 형태

- `shape`가 `[d]`인(또는 원소 개수가 `d`인 임의의 shape) 피처는 **3레벨 리스트**로, 프레임당 리스트
  하나에 리스트당 `d`개 항목을 담는다:

  ```
  optional group <feature name> (LIST) {
    repeated group list {
      optional <float|double|int64|boolean> item;
    }
  }
  ```

  `max_def_level = 3`, `max_rep_level = 1`. `es-data`는 모든 항목을 non-null로 기록하며(`def =
  3`), 리스트 길이가 들쭉날쭉하거나 null을 포함한 파일은 거부한다.

- 다섯 개의 예약된 이름은 **일반(비-리스트) optional 원시 타입**이며, `max_def_level = 1`이다:
  `timestamp`(`float`/`double`), `frame_index`, `episode_index`, `index`, `task_index`(`int64`).
  이는 `미검증 / unverified`다 — LeRobot이 다른 모든 피처와 마찬가지로 이들을 길이-1 리스트로 기록할
  수도 있다. 사람이 고정한 버전이 이와 다르다면, 바뀌는 것은 `crates/es-data/src/lerobot/columns.rs`뿐이다.

- `index`는 데이터셋 전역 프레임 카운터이고 `episode_index`는 파일 내에서 상수다. 둘 다 쓸 때
  **유도**되고(`index = frames_written_so_far + frame`) 읽을 때 **버려지므로**, 결코 `Episode`에
  들어가지 않는다. `frame_index` 역시 `0..length`로 마찬가지로 유도된다. 따라서 왕복 동등성은
  `timestamps`, `task_index`, `columns`, 그리고 비디오 참조에 대해서만 성립한다 — 즉 위치의 순수
  함수가 아닌 모든 것에 대해서.

- 파일 내 컬럼 *순서*는 의미가 없다: 읽을 때 리프는 인덱스가 아니라 `ColumnDescriptor::path().string()`으로
  위치를 찾는다. `es-data`는 에피소드당 하나의 row group을 `BTreeMap` 순서의 컬럼으로 기록한다.

- 압축은 `UNCOMPRESSED`다(parquet의 기본 `WriterProperties`). 압축된 LeRobot 파일을 읽으려면
  `crates/es-data/Cargo.toml`에서 해당 `parquet` 코덱 feature를 켜야 한다. 현재는 아무것도 켜져
  있지 않으므로, snappy/zstd로 압축된 실제 파일은 읽기에 실패한다. LeRobot이 실제로 어떤 코덱을
  쓰는지는 `미검증 / unverified`다.

## `videos/` — 읽지 않음

W7에서는 어떤 비디오 파일도 열거나 디코딩하거나 기록하지 않는다. 프레임의 이미지 피처는
`VideoRef { path, frame_index }`로 노출되며, 여기서 `path`는 데이터셋 루트 기준 상대 경로로
렌더링된 해당 에피소드·카메라의 `video_path` 템플릿이고, `frame_index`는 에피소드 내에서 그
프레임의 위치다. 하나의 mp4가 정확히 한 에피소드의 프레임만을 0부터 담는지는 `미검증 / unverified`다
(`video_path` 템플릿이 함의하는 바는 그렇다).

## `es_ir::PortType`으로의 매핑 (spec §5.4)

`FeatureSpec::port_type()`:

| 피처 | `elem` | `shape` | `unit` | `frame` | `time` | `image` |
|---|---|---|---|---|---|---|
| `float32` | `F32` | 피처 `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `float64` | `F64` | 피처 `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `int64` | `I32` | 피처 `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `bool` | `Bool` | 피처 `shape` | `Dimensionless` | `Policy` | `Tick` | `None` |
| `video` / `image` | `U8` | 피처 `shape` | `Pixel` | `Policy` | `Tick` | `None` |

의도적인 손실 지점 세 가지로, 모두 이 포맷이 기록하지 *않는* 것 때문에 강제된 것이다:

- **`int64 -> ElemType::I32`.** `ElemType`(spec §5.4)에는 64비트 정수가 없다. 메모리 상의
  `Column::I64`는 전체 폭을 유지하며, 좁아지는 것은 외부에 알려지는 `PortType`뿐이다.
- **`Frame::Policy`.** LeRobot 데이터셋은 상태나 액션 벡터에 대한 좌표계를 기록하지 않으며,
  `Frame::Policy`는 스펙에서 "프레임 검사를 하지 않음"을 뜻하는 프레임이다. `Frame::World`를
  임의로 지정하면 파일이 말하지 않는 것을 단정하는 셈이 된다.
- **`image: None`.** `ImageSpec`(해상도는 알 수 있지만 색공간, 카메라 모델, 내부/외부 파라미터,
  왜곡, 셔터, 노출은 알 수 없다)은 `info.json`으로부터 채울 수 없다. spec §19.1은 *데이터셋 안에*
  `ImageSpec`이 있을 것을 요구한다. LeRobot v2.1에는 이를 담을 자리가 없으므로, 이를 복원하는 일은
  이후 패킷의 몫이다(Electric Sheep 사이드카, 또는 실제 파일이 스키마를 고정한 뒤의
  `features[..].info` 객체).

`Unit::Angle`/`Unit::Length`가 아니라 `Unit::Dimensionless`인 이유: `names` 리스트
(`"shoulder_pan"` 등)는 단위 선언이 아니며, `is_policy_input()`은 `Dimensionless`를 받아들이는데
이 텐서들이 실제로 그렇게 쓰이기 때문이다.

## 실제 패키지로 측정, 2026-09-15 — `검증됨 / verified`

오라클 서버에서(패킷 `docs/packets/M5/V1`) 이 크레이트가 쓴 데이터셋(`observation.state`, `action`,
`reward`, `action_source`, `intervention`, 에피소드 2개, 비디오 피처 없음)에 대해
`ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python cargo test -p es-data --test
lerobot_oracle -- --nocapture`를 실행한 결과:

| 항목 | 결과 |
|---|---|
| `lerobot` | 0.6.1, `datasets` 4.8.5, `pyarrow` 25.0.1 — `[dataset]` extra가 이제 설치되어 있다 |
| `import lerobot.datasets.lerobot_dataset` | 성공 |
| `LeRobotDataset(repo_id=..., root=<ours>)` | **거부**: `BackwardCompatibilityError: The dataset you requested is in 2.1 format. We introduced a new format since v3.0 which is not backward compatible with v2.1.` |
| `lerobot.datasets.dataset_metadata.CODEBASE_VERSION` | `"v3.0"` |

즉 패킷이 요구한 발견은 "거부"이며, 설계 노트 미해결 질문 2에 대한 결론이다: **`lerobot` 0.6.1은
`codebase_version: "v2.1"`을 아예 읽지 않는다.** 버전 게이트가 먼저 걸리므로 우리의 parquet, `meta/*.jsonl`,
컬럼 dtype 중 어느 것도 검증되지 않았다. 이 문서의 나머지 `미검증` 표시는 그대로 유지된다.

이것이 "우리의 v2.1 writer가 틀렸다"는 뜻은 아니다. v2.1은 `lerobot`이 직접 배포했던 포맷이고, 0.6.1이
하위 호환을 끊으면서 허브 데이터셋용으로 `python -m lerobot.scripts.convert_dataset_v21_to_v30`을 제공할
뿐이다. v3.0을 직접 쓸 것인지 변환기를 낼 것인지는 별도 패킷의 문제이며(스펙 §19.1), 여기서
`crates/es-data/src/lerobot/meta.rs`의 `codebase_version`은 의도적으로 손대지 않는다.

그때까지 `crates/es-data/tests/lerobot_oracle.rs`는 모든 머신에서 위 거부 사유를 그대로 실어
`SKIP lerobot_oracle: <why>`를 출력한다.

---

# LeRobot v3.0 — 디스크 레이아웃 — `검증됨`

**고정 버전: `lerobot` 0.6.1**, `datasets` 4.8.5, `pyarrow` 25.0.1, `pandas` 2.3.3, `cv2` 4.13.0,
오라클 서버. 2026-09-15에 설치된 패키지에서 직접 읽었다. 패킷은
`docs/packets/M5/V1b-lerobot-v3-export.md`:

```
~/venvs/es-lerobot-cuda/lib/python3.12/site-packages/lerobot/datasets/
    utils.py             (경로 템플릿, DatasetInfo, 청크/파일 크기)
    dataset_metadata.py  (CODEBASE_VERSION, image_keys/video_keys, 로드 경로)
    dataset_reader.py    (get_item, hf_dataset features, tasks 조회)
    io_utils.py          (load_nested_dataset, tasks/episodes/stats 읽기·쓰기)
    feature_utils.py     (get_hf_features_from_features)
    lerobot_dataset.py   (LeRobotDataset.__init__, 독스트링의 레이아웃)
~/venvs/es-lerobot-cuda/lib/python3.12/site-packages/lerobot/scripts/convert_dataset_v21_to_v30.py
```

0.6.1에 `lerobot/datasets/v30/` 디렉터리는 없다. v3.0이 곧 포맷 자체이고,
`convert_dataset_v21_to_v30.py`는 이미 허브에 있는 데이터셋의 업그레이드 경로다.

아래 내용은 문서로만 옮긴 것이 아니라 0.6.1에 대해 **실행**해 확인했다. 정확히 이 모양으로,
`crates/es-data/src/lerobot/v3.rs`가 쓰는 물리 인코딩으로 쓴 데이터셋이
`LeRobotDataset(repo_id=..., root=...)`으로 열리고 프레임을 돌려준다.

## 디렉터리 레이아웃

```
<root>/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── stats.json                        (선택)
│   ├── es_provenance.json                (우리 것, LeRobot의 것이 아님 — 아래 참조)
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/                               (`dtype: "video"` 피처가 있을 때만)
    └── <video_key>/
        └── chunk-000/
            └── file-000.mp4
```

상수, `datasets/utils.py:88-107`:

| 이름 | 값 |
|---|---|
| `DEFAULT_CHUNK_SIZE` | `1000` (청크 디렉터리당 최대 파일 수) |
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

v2.1에서 플레이스홀더가 바뀌었다. **`data_path`와 `video_path`에 `{episode_chunk}`도
`{episode_index}`도 없다.** 한 파일이 여러 에피소드를 담고, 어떤 에피소드가 어느 파일에 있는지는
에피소드 인덱스 산술이 아니라 `meta/episodes/*.parquet`에서 나온다.
`load_nested_dataset`(`io_utils.py:63-83`)은 그냥 `data/*/*.parquet`을 글롭하므로, 청크/파일 번호는
에피소드 테이블이 말하는 것과 자기 일관적이기만 하면 된다.

`DEFAULT_IMAGE_PATH`는 *라이터 쪽* 스테이징 경로(`image_writer.py`)이고 리더는 그것을 해석하지
않는다. 이미지 픽셀은 데이터 parquet을 통해 리더에 도달한다 — 아래 참조.

## `meta/info.json`

`DatasetInfo.from_dict`(`utils.py:114-196`)이 파싱한다. 모르는 키는 **`logger.warning`과 함께
버려지고**, 없는 선택 키는 데이터클래스 기본값을 쓴다.

| 필드 | 타입 | 필수 | 메모 |
|---|---|---|---|
| `codebase_version` | string | 예 | `3.0`으로 파싱되어야 한다. `2.1`은 `BackwardCompatibilityError`, `> 3.0`은 `ForwardCompatibilityError` |
| `fps` | int | 예 | `__post_init__`이 `<= 0`을 거부 |
| `features` | object | 예 | `name -> {dtype, shape, names}`. parquet 스키마 전체를 결정한다 |
| `total_episodes` | int | 아니오 (0) | `0`이면 `meta/episodes`를 읽지 않는다 |
| `total_frames` | int | 아니오 (0) | |
| `total_tasks` | int | 아니오 (0) | `0`이면 `meta/tasks.parquet`을 읽지 않고, 그러면 `get_item`이 `meta.tasks.iloc[...]`에서 예외 |
| `chunks_size` | int | 아니오 (1000) | `> 0` |
| `data_files_size_in_mb` | int | 아니오 (100) | `> 0` |
| `video_files_size_in_mb` | int | 아니오 (200) | `> 0` |
| `data_path` | string | 아니오 (기본값) | |
| `video_path` | string \| null | 아니오 (기본값) | `dtype: "video"` 피처가 없으면 `null` |
| `robot_type` | string \| null | 아니오 | |
| `splits` | object | 아니오 (`{}`) | |
| `tools` | list \| null | 아니오 | OpenAI 형식 도구 스키마. 없으면 생략된다 |

v2.1 필드인 `total_videos`와 `total_chunks`는 v3.0 필드가 *아니며* 경고와 함께 버려진다.

### `features` -> parquet 스키마

`DatasetInfo.__post_init__`이 모든 `shape`를 리스트에서 **튜플**로 바꾸고,
`get_hf_features_from_features`(`feature_utils.py:43-83`)가 이 순서로 매핑한다:

| 조건 | `datasets` 피처 | arrow 타입 |
|---|---|---|
| `dtype == "video"` | *건너뜀* | 컬럼 자체가 없다 |
| `dtype == "image"` | `datasets.Image()` | `struct<bytes: binary, path: string>` |
| `shape == (1,)` | `datasets.Value(dtype)` | 평범한 스칼라 |
| `len(shape) == 1` | `datasets.List(Value(dtype), length=n)` | `fixed_size_list<item: T>[n]` |
| `len(shape) in 2..=5` | `Array2D`..`Array5D` | 중첩 고정 크기 리스트 |

물리는 두 가지 결과가 있다:

- **`shape: [1]` 피처는 길이 1 리스트가 아니라 스칼라 컬럼이다.** v2.1 라이터는 모든 피처를 3-레벨
  LIST로 쓴다. v3.0에서는 `reward`, `timestamp`, `frame_index`, `episode_index`, `index`,
  `task_index`가 평범한 프리미티브여야 한다.
- 다섯 부기 컬럼은 **`features`에 반드시 나타나야 한다.** `features`가 곧
  `Dataset.from_parquet(..., features=...)`이 파일을 캐스트하는 대상이기 때문이다. v2.1에서 남겨둔
  미해결 질문(위의 `미검증`)이 v3.0에서는 답이 나왔다: 필수다.

선언이 `fixed_size_list<item: T>[n]`인 자리에 parquet 가변 길이 `list<item: T>`가 와도 받아들인다 —
`datasets`가 캐스트한다 — 그래서 `columns.rs`가 이미 쓰는 3-레벨 LIST를 재사용할 수 있다. 가정이
아니라 측정이다.

### ffmpeg·torchcodec 없이 읽히는 이미지 피처

`dtype: "image"` 픽셀은 **데이터 parquet 안에** `struct<bytes: binary, path: string>`로 들어간다.
`bytes`는 인코딩된 이미지 파일(여기서는 PNG)이고 `path`는 null이다.
`hf_transform_to_torch`(`io_utils.py:266-293`)가 PIL 이미지를 `[0, 1]` 범위의 `float32` `(C, H, W)`
텐서로 바꾼다. 이 경로는 `torchcodec`·`pyav`·`ffmpeg` 어느 것도 건드리지 않는다.

다른 선택지인 `dtype: "video"`는 디코더가 필요하다. `dataset_reader._query_videos`가
`decode_video_frames`를 부른다. 오라클 서버에서 `torchcodec`은 설치되어 있지만 **로드되지 않고**
(`libnppicc.so.12: cannot open shared object file`) `pyav`로 폴백한다. 그래서
`es dataset export --lerobot-v3`는 `video`가 아니라 `image`를 쓰고, 인코더를 전혀 호출하지 않는다 —
`~/.local/bin/ffmpeg`도, Rust에서도, 오라클의 Python 쪽에서도.

대가는 크기다. stored-deflate PNG는 원시 프레임 + 약 0.1%다. `data_files_size_in_mb`는 권고값이므로
(리더는 글롭한다) 큰 데이터 파일 하나도 합법이다. 분할은 최적화이지 정확성 요건이 아니다.

## `meta/episodes/chunk-XXX/file-XXX.parquet`

`load_episodes`(`io_utils.py:212-218`)가 **피처 선언 없이** 읽고 — arrow 타입은 파일에서 추론된다 —
`stats/*` 컬럼을 모두 버린다. 리더가 실제로 쓰는 컬럼:

| 컬럼 | 타입 | 사용처 |
|---|---|---|
| `episode_index` | int64 | `filter_episodes`, `_check_cached_episodes_sufficient` |
| `length` | int64 | `meta.episodes[i]["length"]` |
| `dataset_from_index` | int64 | `dataset_reader._get_query_indices` (delta-timestamp 윈도) |
| `dataset_to_index` | int64 | 동일. 끝은 배타적 |
| `tasks` | list\<string\> | 에피소드 단위 태스크 레이블 |
| `data/chunk_index` | int64 | `DatasetMetadata.get_data_file_path` |
| `data/file_index` | int64 | 동일 |
| `meta/episodes/chunk_index` | int64 | *라이터*의 append 경로 |
| `meta/episodes/file_index` | int64 | 동일 |
| `videos/<key>/chunk_index`, `videos/<key>/file_index`, `videos/<key>/from_timestamp` | int64/float | `dtype: "video"` 피처에만 |
| `stats/<feature>/<stat>` | — | 선택적 에피소드별 통계. 로드 시 버려진다 |

컬럼 *이름*에 `/`가 들어간다. 중첩 그룹이 아니라, 이름에 슬래시가 들어간 평평한 최상위 parquet
필드다.

`dataset_from_index` / `dataset_to_index`는 파일 순서로 에피소드에 대해 누적되며, 데이터 parquet의
`index` 컬럼과 일치한다.

## `meta/tasks.parquet`

`load_tasks`는 `pd.read_parquet(...)` 뒤에 `tasks.index.name = "task"`이고(`io_utils.py:184-187`),
리더는 프레임의 태스크를 `self._meta.tasks.iloc[task_idx].name`(`dataset_reader.py:352`)으로 푼다 —
즉 **태스크 문자열은 컬럼 값이 아니라 pandas 인덱스여야 한다.** 그래서 파일은 parquet 컬럼 두 개,
`task_index`(int64)와 `task`(string)를 담고, *그 위에* 어느 쪽이 인덱스인지 pyarrow에게 알려주는
푸터의 `pandas` key/value 메타데이터를 담는다:

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

조회가 위치 기반(`iloc`)이므로 행 순서는 `task_index`와 일치해야 한다.

## `meta/stats.json` — `검증됨 / verified`

**읽기에는 선택 사항, 학습에는 필수.** 파일이 없으면 `load_stats`는 `None`을 돌려주고
(`io_utils.py:161-175`) 읽기 경로에서 그것을 요구하는 곳은 없다. 그러나 `lerobot-train`은 모든 정책
피처를 `LeRobotDataset.meta.stats`로 정규화하므로, 이 파일이 없는 데이터셋으로는 학습할 수 없다.
패킷 `M5/V8`이 이 절의 이전 판이 말한 그 후속 작업이고, `es dataset export --lerobot-v3`는 이제
이것을 쓴다.

**모양.** `{feature: {stat: 중첩 리스트}}`. `load_stats`는 `load_json` 다음
`cast_stats_to_numpy`이고, 그것은 `flatten_dict` → `np.atleast_1d(np.array(v))` →
`unflatten_dict`이다(`io_utils.py:148-158`). 따라서 어떤 JSON 숫자 트리든 받아들여지고, 정규화기가
색인하는 것은 *리스트의 모양*이다. 세 가지 모양이 나오며
`compute_stats._validate_stat_value`는 그 밖의 것을 받지 않는다.

| 피처 | 통계 모양 | 예 |
|---|---|---|
| `shape: [n]`, `n > 1` | `[n]` | `"observation.state": {"mean": [숫자 6개]}` |
| `shape: [1]`(스칼라 컬럼) | `[1]` | `"timestamp": {"mean": [0.42]}` |
| `dtype: "image"`/`"video"` | `[3, 1, 1]`(또는 `[1,1,1]`) | `{"mean": [[[0.79]], [[0.76]], [[0.70]]]}` |
| 전부 | `count`는 항상 `[1]` | |

**키.** `min`, `max`, `mean`, `std`, `count`, 그리고 `q01`, `q10`, `q50`, `q90`, `q99`.
`NormalizerProcessorStep`은 `MEAN_STD`에 `mean`/`std`, `MIN_MAX`에 `min`/`max`, `QUANTILES`에
`q01`/`q99`, `QUANTILE10`에 `q10`/`q90`을 읽고, 자기 모드가 원하는 짝이 없으면 이름을 짚어
예외를 던진다. **ACT는 `VISUAL`·`STATE`·`ACTION` 모두 `MEAN_STD`이므로**
(`ACTConfig.normalization_mapping`, 0.6.1에서 측정), 익스포트는 앞의 다섯 개를 쓰고 **분위수는
생략한다**. 분위수는 LeRobot 자신의 코드에서도 5000 구간 히스토그램 추정치이고
(`RunningQuantileStats`), 근사의 근사를 하나 더 만드는 것은 정직하게 없는 키보다 나쁘다.

**LeRobot이 계산하는 방식, 그리고 익스포트가 그것과 맞는 이유.** `compute_episode_stats`는 에피소드
하나를 줄인다 — 벡터 컬럼은 `axis=0`, 이미지는 `axis=(0,2,3)`과 `/255`. `aggregate_stats`는
에피소드들을 병렬 분산 공식(`(var_i + (mean_i − mean)²)`에 `count_i` 가중)으로 합치는데, 이것은
*정확하다*. 합쳐진 평균과 모집단 표준편차는 전체 데이터셋을 한 번에 훑어 구한 값과 같고, 그것이
`crates/es-data/src/lerobot/v3.rs`가 하는 일이다. `std`는 양쪽 다 모집단(`ddof = 0`)이다.

**정직한 차이 하나: 이미지는 표본이다.** `sample_images`는 에피소드당
`estimate_num_samples(n) = max(100, min(int(n**0.75), 10_000))` 프레임을 뽑고 300 px를 넘으면
다운샘플한다. 그래서 긴 에피소드에서 LeRobot의 이미지 통계는 추정치이고 익스포트의 것은 모든
프레임의 모든 픽셀에 대한 정확한 값이다. 이미지 피처의 `count`도 다르다 — LeRobot은 표본 프레임
수를, 익스포트는 프레임 수를 센다. 표본이 전수가 될 만큼 짧은 데이터셋에서는 둘이 부동소수점
정밀도까지 일치하며, `crates/es-data/tests/lerobot_v3.rs::lerobot_v3_stats`가 그 비교다.
`2026-09-15` 측정: 모든 피처·모든 키에 걸쳐 최대 불일치 `5.5e-08`.

## 받아들여진 parquet 물리 인코딩

0.6.1 / pyarrow 25.0.1에 대해 `parquet 59.3`의 저수준 컬럼 API로 써서 측정했다:

- `UNCOMPRESSED` 페이지 (`crates/es-data/Cargo.toml`에 코덱 피처가 켜져 있지 않다).
- `ARROW:schema` 푸터 메타데이터 없음. pyarrow가 parquet 스키마에서 arrow 타입을 추론하고
  `datasets`가 선언된 피처로 캐스트한다.
- `fixed_size_list`가 선언된 자리에 3-레벨 LIST
  (`optional group X (LIST) { repeated group list { optional T item; } }`).
- 이미지 컬럼에 `optional group X { optional byte_array bytes; optional byte_array path (String); }`,
  `path`는 모든 행에서 null.
- 다섯 부기 컬럼과 모든 `shape: [1]` 피처에 평범한 `optional` 프리미티브.
- 파일 전체에 로우 그룹 하나. `write_table_one_row_group_per_episode`는 LeRobot 자체 라이터가 하는
  것이고(`io_utils.py:295-309`) 임의 접근 최적화이지 요건이 아니다.

## `meta/es_provenance.json` — 우리 것이고 LeRobot의 것이 아니다

스펙 §19.2에 따라 내보낸 것은 파생 산출물이므로 출처를 기록한다:

```json
{"source_root": "...", "source_codebase_version": "v2.1",
 "content": "<64 hex>", "schema": "<64 hex>", "split": "<64 hex>",
 "exported_by": "es dataset export --lerobot-v3"}
```

세 해시는 표시 전용 all-train 스플릿으로 원본에 대해 계산한 `DatasetIdentity::compute`이며,
`es dataset info`가 출력하는 것과 정확히 같다. `info.json`의 추가 키가 아니라 사이드카인 이유는
`DatasetInfo.from_dict`가 모르는 키를 (경고와 함께) 버리고 `to_dict`가 그것을 되쓰지 않기 때문이다 —
`info.json`에 넣은 출처는 LeRobot 쪽 재기록을 살아남지 못한다.

## `es loop collect`가 쓰는 컬럼 — 우리 것이고 LeRobot의 것이 아니다

`observation.state`, `action`, `reward`는 LeRobot 자신의 이름이다. 나머지 셋은 Electric Sheep의
것이며, 다른 피처와 똑같이 `features`에 선언되므로 v2.1 라이터도 v3.0 익스포터도 특수 분기 없이
그대로 실어 나른다:

| 컬럼 | dtype | shape | 의미 |
|---|---|---|---|
| `action_source` | `int64` | `[1]` | `es_data::ActionSourceCode`: `Policy` / `Human` / `Clamped` / `Fallback` (스펙 §13.2) |
| `intervention` | `int64` | `[1]` | 사람 또는 스크립트 개입자가 그 틱을 몰았으면 1 |
| `action_commanded` | `float32` | `[nu]` | 그 틱의 플레인 통과 **이전** 명령 — **패킷 M5/V1c** |

**`action`은 실행된 액션이고 `action_commanded`는 요청된 값이다.** `action`은
`DomainRunner::emit_actions`가 `ctrl`에 복사한 `SafetyPlane::validate`의 결과 — 모든 클램프를
거친 뒤 액추에이터에 도달한 값이다(`INV-12`). `action_commanded`는 같은 틱에 청크 버퍼가 내어준
행, 즉 플레인이 판정하기 전의 값이다. 플레인이 손대지 않고 통과시킨 틱에서 둘은 같고, 버퍼에 그
틱의 행이 아예 없었던 틱 — 명령 자체가 없었고 `action_source`가 `Fallback`을 읽는 틱 — 에서도
같다.

정책을 학습시키는 소비자가 원하는 것은 `action`이다. Deployment IR의 엔벌로프가 그 정책에게
재현하도록 허용하는 값을 가진 유일한 컬럼이기 때문이다. `action_commanded`는 출처 기록이며,
플레인을 다시 돌리지 않고도 `Clamped` 프레임을 읽을 수 있게 한다.

**이로써 `dataset_schema_hash`가 이동했다**(따라서 `content`도, §19.1/§19.2). 패킷 M5/V1c 이전에
수집된 데이터셋에는 `action_commanded` 컬럼이 없으므로, 같은 시드에서 수집했더라도 이후의 것과
해시가 다르다. 이는 의도된 동작이며 — 스키마는 학습 실행의 입력 정체성의 일부다 — V1c가 기존
세트를 패치하는 대신 재수집하는 이유다. 두 컬럼 모두 `es dataset export --lerobot-v3`를 그대로
통과하고(`export_layout_is_v3`), `lerobot` 0.6.1은 추가 피처를 실은 데이터셋을 읽는다
(`lerobot_v3_export`).

### `observation.state`는 `qpos ‖ qvel`이고, 패킷 M5/V7a는 이에 의존한다

`es_data::collect::to_lerobot`는 이 행을 env 0의 `qpos` **전체** 뒤에 `qvel` **전체**를 붙여
쓴다. 폭은 `nq + nv`이고, 관절의 일부를 고른 적은 한 번도 없다. SO-101 데모 씬에서는
`13 + 12 = 25`다: `qpos[0..6]`에 팔의 여섯 힌지, `qpos[6..13]`에 큐브의 free 조인트
(위치 `xyz` 다음 쿼터니언 `wxyz`), 그 뒤에 대응하는 `qvel`. `features`는 이를 그 폭의 `float32`
피처 하나로 선언하며, v2.1 라이터도 v3.0 익스포터도 특수 분기 없이 실어 나른다.

여기서 두 가지 귀결이 나오고, 둘 다 설계를 떠받친다:

* **`qpos`의 `IndexRange`가 기록된 행을 그대로 인덱싱한다.** 추론에서
  `es_eval::runner::capture`는 `StateView::qpos_of(0)[r]`을 읽고, `es_eval::ObservationBake`는
  parquet에서 `row[r]`을 읽는다. 같은 두 경계이며 변환이 없다 — 이 행은 상태의 재인코딩이
  아니라 상태에 `qvel`을 덧붙인 것이다. 데모의 특권 채널 `sim_cube_pose`가 추론이 내어주는
  값과 비트 단위로 동일하게 bake되는 근거가 바로 이것이다
  (`a_baked_frame_is_bit_identical_to_what_capture_serves`).
* **`observation.state`는 V7a가 필요로 하기 전부터 이미 큐브의 포즈를 싣고 있었다.** 이 패킷은
  컬럼을 추가하지 않았고 `dataset_schema_hash`도 움직이지 않았다. 움직인 것은 Observation IR이
  그 행의 어느 구간을 읽는가다.

위 v2.1 예시의 `names` 목록(`["shoulder_pan", ...]`)은 LeRobot 자신의 관례를 보여줄 뿐,
데모에서 이 라이터가 내보내는 것이 아니다. `qpos ‖ qvel` 25칸 행에는 관절당 이름 하나라는
읽기가 존재하지 않는다.
