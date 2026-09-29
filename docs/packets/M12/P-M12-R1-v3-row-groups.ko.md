<!-- Korean translation of docs/packets/M12/P-M12-R1-v3-row-groups.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R1 — `es dataset export --lerobot-v3`는 에피소드당 row group 하나를 쓴다

2026-09-29, 이 Windows PC에서 카메라 전용 큐브 템플릿을 처음 실제로 돌린 Y-V 항목 1에서
발견했다(`lerobot 0.6.1`, `datasets 4.8.5`, `pyarrow 25.0.1`). 스펙 §19.2(내보낸 것은 파생
산출물이다), api-note `docs/api-notes/lerobot-dataset.md`, 설계 노트
`docs/design/visible-learning.md` 7절(V1b 내보내기 절).

## 결함

`lerobot-train`이 시연 200개(103,881 프레임)의 내보내기를 거부했다:
`pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked
array outputs` (`datasets/packaged_modules/parquet/parquet.py` → `pyarrow._dataset`).

측정한 근본 원인: `crates/es-data/src/lerobot/v3.rs`의 `write_data`가 데이터셋 전체를 **row group
하나**로 쓴다(`writer.next_row_group()`을 한 번만 부른다). 이미지 컬럼은 stored-deflate PNG(원시
96×96×3 프레임과 거의 같은 크기)를 담는 `struct<bytes: binary, path: string>`이므로, 그 row group
하나가 2.9 GB가 되고, pyarrow는 row group 하나에서 2 GB를 넘는 중첩 컬럼을 구체화하지 못한다 —
`pq.read_table`조차 그 파일에서 실패한다. 같은 행을 1,000행짜리 row group으로 다시 쓰면
(`ParquetFile.iter_batches` → `ParquetWriter`) `LeRobotDataset`이 103,881 프레임을 모두 로드한다.
api-note의 "파일 전체에 row group 하나 … 임의 접근 최적화이지 요건이 아니다"는 이 크기에서
틀렸다. LeRobot 자체 라이터는 에피소드당 row group 하나를 쓴다(`io_utils.py`,
`write_table_one_row_group_per_episode`).

## 스펙

- `write_data`는 **에피소드당 row group 하나**를 에피소드 순서대로 쓰고, 각 row group에는 정확히
  그 에피소드의 행만 담는다. 컬럼 순서, 타입, definition level, 값, 이미지 행의 `path = null`,
  `meta/info.json`은 바뀌지 않는다. row group 경계만 옮겨진다.
- 컬럼별 통계(`meta/episodes` 통계, 반환되는 맵)는 계속 **모든** 행에 대해 계산하며, 오늘의 것과
  바이트 단위로 같다.
- 그 밖에는 아무것도 바뀌지 않는다: 데이터 파일 하나, PNG 압축 변경 없음, 청크/파일 분할 없음.

## 컨텍스트

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

## 오라클

1. 먼저 실패하는 테스트를 `crates/es-data/tests/lerobot_v3.rs`에 둔다: 길이가 서로 다른 에피소드
   세 개 이상의 픽스처를 내보낸다. 데이터 파일은 `num_row_groups == episodes`이고, row group
   `i`는 `rows == length(episode i)`이며, 모든 행을 다시 읽으면(모든 row group을 순서대로) 오늘의
   내보내기가 주는 행과 정확히 같다(새 골든이 아니라 기존 테스트가 이미 단언하는 값과 비교한다).
2. 기존 `lerobot_v3` 테스트와 `cargo test -p es-data`가 그대로 통과하고, `cargo test -p es --test
   cli`의 내보내기 테스트가 통과한다(Python 게이트가 걸린 것은 건너뛰거나 실행된다).
3. `ES_PYTHON`을 설정하면(이 머신에는 이제 `.venv`가 있다: lerobot 0.6.1) 기존의 Python 게이트
   LeRobot 읽기 오라클이 있다면 건너뛰지 않고 실행된다.
4. fmt, clippy `-D warnings`, `cargo xtask verify-goldens`, `check-scope`.

## 수용 기준

오라클 1–4. api-note와 설계 노트의 문장을 고치고(+ko), 무엇을 측정했는지 적는다.

## 금지

이미지 인코딩, 스키마, `info.json`, 통계를 바꾸는 것. 여러 파일로 나누는 것. 컨텍스트 밖의 어떤
변경도. 원격 서버에 접속하는 것.
