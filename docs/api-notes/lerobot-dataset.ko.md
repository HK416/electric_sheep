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
