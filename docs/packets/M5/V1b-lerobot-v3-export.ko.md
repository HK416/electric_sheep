# M5 V1b — LeRobot v3.0 내보내기

설계 노트: `docs/design/visible-learning.md` 7.7절. 먼저 7.5절을 읽을 것 — 이 패킷이 닫는 거부를
V1의 오라클이 측정했다. V1(`9e18236`)에 의존한다.

`es loop collect`는 `codebase_version: "v2.1"`을 쓰고, `lerobot` 0.6.1은 우리 parquet을 한 바이트도
보기 전에 `BackwardCompatibilityError`를 낸다. v2.1 라이터는 **의도적으로 그대로 둔다** — V2의 학습
스크립트가 그것을 직접 읽는다 — 그래서 이 패킷은 변환기를 추가한다.
`es dataset export --lerobot-v3 <root> --out <dir>`의 출력은 설치된 `lerobot`이 연다.

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

메모: `v3.rs`는 신규이고 변환기 전체 — v3.0 스키마, parquet 라이터, PNG 인코더 — 를 담는다.
`mod.rs`와 `lib.rs`는 모듈 선언과 재수출만 얻는다. `columns.rs`와 `meta.rs`는 **건드리지 않는다**:
그것들이 기술하는 것이 v2.1이다. `lerobot_read_ref.py`는 평탄화된 프레임별 컬럼과 이미지 shape를
얻어, 오라클이 양 끝이 아니라 모든 프레임을 확인할 수 있게 한다. `dataset.rs`는 기존 `info` 옆에
`export` 하위 명령을 얻는다.

## spec

- §1.4: 오라클은 우리가 쓴 디렉터리를 `lerobot` 패키지가 열어 프레임을 돌려주는 것이지, 우리 리더가
  스스로에게 동의하는 것이 아니다. 합불은 명령이다.
- §1.9: LeRobot 호환성은 절대 잘라내지 않는 항목이다. V1은 0.6.1에 대해 우리에게 호환성이 전혀 없음을
  측정했고, 이것이 그 수리다.
- §2.5: 변환기는 Rust이고 Python 없이 돈다. 인터프리터는 오라클에만 나타난다.
- §19.1: 디스크 포맷은 `lerobot` 0.6.1의 v3.0 — `meta/info.json`, `meta/tasks.parquet`,
  `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`,
  `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`. 필드 단위 기술은
  `docs/api-notes/lerobot-dataset.md`의 "LeRobot v3.0"에 있다.
- §19.2: `dataset_hash = H(content, schema, split)`. 내보낸 것은 **파생 산출물**이므로 원본 데이터셋의
  `content_hash`와 `schema_hash`를 기록해 출처가 복사를 살아남아야 한다. `meta/es_provenance.json`이
  그것을 담는다 — `lerobot`의 `DatasetInfo.from_dict`는 모르는 `info.json` 키를 경고와 함께 버리므로
  재기록이 한 번이라도 일어나면 조용히 잃는다.
- §25.1: 내보낸 것은 쓰기만 하고 신뢰 판단으로 되읽지 않는다. Python 오라클은 프로세스 밖에서 읽는다.
- §3.4: float 시간 누적 없음 — `timestamp`는 원본에서 복사하며 다시 유도하지 않는다.
- §1.5: `es-data`는 3,515 코드 라인이고, 이 패킷의 예산은 ~450줄 이하다.

## oracle

```
cargo fmt --check
cargo clippy -p es-data -p es --all-targets -- -D warnings
cargo test -p es-data --test lerobot_v3
cargo test -p es --test cli dataset_export
cargo xtask context-budget
cargo xtask check-spec-refs
```

인터프리터가 필요한 오라클:

```
ES_LEROBOT_PYTHON=$HOME/venvs/es-lerobot-cuda/bin/python \
  cargo test -p es-data --test lerobot_v3 -- --nocapture
```

1. **`lerobot_v3_export` — 진짜 패키지가 내보낸 것을 읽는다.** 테스트는 V1 형태의 v2.1 데이터셋
   (에피소드 2개, `observation.state`, `action`, `reward`, `action_source`, `intervention`,
   그리고 `observation.images.*` 카메라 하나)과 V0b 렌더러가 덤프하는 원시 프레임을 쓰고, 변환기를
   돌린 뒤 결과를 `crates/es-data/python/lerobot_read_ref.py`에 넘긴다. 스크립트는
   `LeRobotDataset(root=...)`로 열어 **모든** 프레임을 순회한다. 테스트는 자신이 쓴 컬럼에 대해
   에피소드 수, 총 프레임 수, 에피소드별 길이, 모든 `observation.state`·`action` 값(프레임 단위로
   평탄화), 카메라 텐서의 shape와 픽셀 합을 단언한다. `ES_LEROBOT_PYTHON`이 없거나 `lerobot`이
   없으면 `SKIP lerobot_v3_export: <why>`, 돌았으면 `RAN lerobot_v3_export: <json>`. 거부는 스킵이
   아니라 실패다: 스킵 경로는 인터프리터 부재만 덮고, 거절된 데이터셋은 절대 덮지 않는다.
2. **`png_round_trips`** — PNG 인코더는 stored-deflate IDAT를 쓴다. 테스트는 자기 출력을 되파싱해
   (매직, IHDR, stored 블록, Adler-32, 모든 청크 CRC) 픽셀을 입력과 비교한다. 인터프리터 없이도
   인코더가 깨지면 실패하는 검사다.
3. **`export_layout_is_v3`** — 출력의 `meta/info.json`이 `codebase_version: "v3.0"`, 네 개의 경로
   템플릿, 예약 부기 컬럼 다섯 개와 `dtype: "image"` 카메라를 담은 `features`를 갖고, parquet 파일
   세 개가 존재하며 비어있지 않다.
4. **`provenance_records_the_source_identity`** — `meta/es_provenance.json`의 `content`와 `schema`가
   원본 데이터셋에 대한 `DatasetIdentity::compute`와 같다(§19.2).
5. **`images_without_frames_are_dropped_not_dangled`** — `--frames` 디렉터리 없이 카메라를 선언한
   데이터셋을 내보내면 해당 피처를 버리고 보고한다. 아무도 못 읽을 픽셀을 약속하는 `info.json`을
   쓰지 않는다.
6. `crates/es/tests/cli.rs`: `dataset_export_writes_a_v3_dataset` — `es dataset export
   --lerobot-v3 <root> --out <dir>`가 0으로 끝나고 에피소드·프레임 수를 출력한다. `--out` 누락은
   종료 코드 2(사용법), 없는 root는 1(런타임)이다.

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

- 이미지 저장은 **데이터 parquet에 임베드된 PNG 바이트**이며 `dtype: "image"` 아래
  `struct<bytes: binary, path: string>`이다 — `lerobot` 0.6.1이 `torchcodec`(오라클 서버에서 로드
  실패)도 `ffmpeg`도 없이 디코드하는 유일한 저장 방식이다. mp4는 쓰지 않고, Rust에서도 Python에서도
  비디오 인코더를 호출하지 않는다.
- 모든 에피소드를 담은 `data/chunk-000/file-000.parquet` 하나와
  `meta/episodes/chunk-000/file-000.parquet` 하나. `data_files_size_in_mb`에서의 청크/파일 분할은
  구현하지 않는다 — 리더는 `data/*/*.parquet`을 글롭하므로 신경 쓰지 않으며, 그 한계는 소스에
  표시한다.
- 새 외부 의존성 없음: `columns.rs`가 이미 쓰는 `parquet 59.3` 저수준 컬럼 API와, ~70줄짜리 PNG
  인코더(stored-deflate, CRC-32, Adler-32)뿐이다.
- 새 트레이트 없음(INV-17), `HashMap` 없음, float 시간 없음, 소스 ~450줄 이하.

## forbidden

- `crates/es-data/src/lerobot/columns.rs`, `crates/es-data/src/lerobot/meta.rs` — v2.1 리더/라이터는
  `codebase_version`을 포함해 지금 그대로 둔다. V2의 학습 스크립트가 읽는다.
- `crates/es-data/src/collect.rs` — `es loop collect`는 계속 v2.1을 쓴다.
- `crates/es-policy`, `crates/es-eval`, `crates/es-env`, `crates/es/src/cmd/eval.rs` — V2와 V3의 것.
- 학습 스크립트, 그리고 `lerobot_read_ref.py` 외의 `scripts/`·`python/` 아래 모든 것.
- `es-data`에 `arrow`, 압축 코덱, 이미지 크레이트, deflate 크레이트를 추가하는 것.
- 진짜 패키지 대신 LeRobot 리더를 Rust나 Python으로 다시 구현하는 것.
- 지어낸 숫자로 `meta/stats.json`을 쓰는 것: 없는 것은 정직하지만(`load_stats`는 `None`을 돌려준다)
  틀린 것은 아니다.

## as built

2026-09-15 오라클 서버에서 측정. 근거는 설계 노트 7.7절에 있고, 여기는 위 패킷과 달라진 점이다.

**오라클 결과.** `lerobot_v3_export`: **RAN**. `LeRobotDataset(repo_id=..., root=<export>)`가
열어서 `codebase_version 3.0`, **에피소드 2개 / 프레임 7개**를 보고했고, 7프레임을 모두 순회했으며
(`"iterated": 7`), `observation.state`
`[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 1.0, ..., 3.0]`과
`action` `[0.0, -0.5, ..., -5.5, 1.0, 0.5, ..., -3.0]`을 돌려줬다 — 우리가 쓴 parquet과 프레임
단위로 1e-5 이내 일치. 카메라는 `[3, 4, 6]`(CHW), 픽셀 합 **111.67059516906738** 대 우리 계산
28476/255 = **111.670588**. `png_round_trips`, `png_spans_several_stored_blocks`,
`export_layout_is_v3`, `provenance_records_the_source_identity`,
`images_without_frames_are_dropped_not_dangled`, `dataset_export_writes_a_v3_dataset`는 모두
인터프리터 없이 통과한다.

**위 계획과의 차이**, 각각 패키지가 실제로 하는 동작이 강제한 것:

- `ExportReport`는 버려진 카메라마다 사유 문자열을 담지 않는다 — 사유는 "뒤에 프레임이 없다" 하나뿐이고,
  CLI가 카메라마다 한 번씩 출력한다.
- `export_v3`는 `meta/tasks.jsonl`이 없는 데이터셋과 정수가 아닌 `fps`를 추측하지 않고 거부한다:
  `total_tasks: 0`이면 `lerobot`이 `meta/tasks.parquet`을 건너뛰고 `get_item`에서 예외를 내며,
  `DatasetInfo`는 `fps: int`를 선언한다.
- PNG 인코더는 단위 테스트를 하나 더 얻었다(`png_spans_several_stored_blocks`). 96x96 프레임은
  27 KB이고 160x160은 첫 테스트가 닿지 못한 65535바이트 stored 블록 경계를 넘는다.
- `crates/es-data/python/lerobot_read_ref.py`(v2.1 오라클과 공유)는 `iterated`, `states_flat`,
  `actions_flat`, `cameras`, `image_shape`, `image_sum`, `tasks`를 얻었다. 출력이 전부 평평한 것은
  의도다 — Rust 쪽은 파서 없이 JSON을 훑는다.
- `es dataset export`는 `--frames <dir>`를 받는다. 패킷의 시그니처는 암시했지만 CLI 줄에는 적혀
  있지 않았다.
- **예산이 틀렸다: ~450줄 추정, 실제 702줄**(`es-data` 3,515 -> 4,154, `es` 3,596 -> 3,659. 둘 다
  여전히 §1.5 상한보다 한참 아래). 추정은 손으로 만든 스키마를 가진 parquet 라이터 셋의 값을 매기지
  못했고, `columns.rs`가 `forbidden`이라 스키마 빌더 ~35줄은 공유가 아니라 중복이다. 그 둘을 합치는
  것은 다음에 두 파일을 함께 소유하는 쪽의 리팩터링이지, 여기서 곁다리로 할 일이 아니다.

**사람에게 묻는 질문.**

1. `meta/stats.json`을 쓰지 않으므로, 내보낸 것으로 `lerobot` 쪽에서 *학습*하면 정규화 통계가 없다.
   읽기에는 영향이 없다. V2 스크립트가 아니라 `lerobot`으로 학습하려면 그것은 별도 패킷이다.
2. `meta/tasks.parquet`은 `pandas` 푸터 메타데이터 관행에 의존하는데, 이는 문서화된 LeRobot 포맷
   필드가 아니라 pandas의 직렬화 세부사항이다. `pandas` 2.3.3 / `pyarrow` 25.0.1에 고정하고
   오라클로 검증했지만, pandas 메이저 버전이 바뀌면 움직일 수 있다.
3. `es loop collect`에서 `--frames` 디렉터리를 쓰는 것은 아직 아무것도 없다 — V1이 렌더러를 CLI
   경로 밖에 뒀다. 그것이 연결되기 전까지, 실제 시연 데이터셋을 내보내면 카메라가 버려진다.
