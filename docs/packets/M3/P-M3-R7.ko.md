<!-- Korean translation of docs/packets/M3/P-M3-R7.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R7 — 메타데이터 불일치는 패닉을 일으키지 않는다

Spec: §13.2 (개입 라벨), §19.2 (데이터셋 식별), W7
(`docs/packets/M3/W7-learning-loop.md`).
리뷰 지적사항: `docs/reviews/M3.md` Should-fix, `crates/es-data/src/intervention.rs:280` —
`m[i]`가 `meta.length`로 크기가 정해진 마스크를 `ep.len()`을 도는 `i`로 인덱싱했다;
에피소드 메타데이터가 자신의 parquet과 어긋나는 데이터셋은 크레이트의 나머지 부분이 이미
`DataError`로 취급하는 불일치를 보고하는 대신 거기서 패닉을 일으킨다.

## context (범위)

```
crates/es-data/src/intervention.rs
crates/es-data/tests/loop_learning.rs
docs/packets/M3/P-M3-R7.md
```

## spec (사양)

- `label`의 에피소드별 루프는 마스크의 길이 출처(`dataset.episodes()`에서 온
  `meta.length`)를 `ep.len()`(parquet의 `read_episode`가 실제로 반환한 값)과
  **어느 한쪽으로 다른 쪽을 인덱싱하기 전에** 비교하고, 둘이 다르면 에피소드와 두 길이를
  모두 명시한 `DataError::Inconsistent`를 반환한다 — 리뷰가 명시적으로 기각한
  `m.get(i).copied().unwrap_or(false)`가 아니다: 불일치는 잘못된 데이터셋이며, 기본값으로
  덮어버리면 실제 개입 프레임이 조용히 사라질 뿐 손상을 보고하지 못한다(스펙 13.2의
  요점 자체가 신뢰할 수 있는 라벨이다). `DataError::Inconsistent`는 기존의
  `String` 페이로드 변형 그대로 유지한다(이미 `es-data` 전체에서 십여 곳에 쓰이며, 같은
  함수 안에서 세 줄 앞에도 쓰인다) — 새로운 구조체 형태 변형을 만드는 대신 호출부 한 곳이
  설명적인 메시지를 포맷하는 편이 `Inconsistent`의 기존 모든 호출부를 뜯어고치는 것보다
  작은 diff다.
- 이 가드는 에피소드당 한 번, `masks`를 인덱싱하는 `values` 벡터보다 먼저 실행되므로,
  해당 에피소드가 개입 구간을 하나도 갖지 않더라도 그 에피소드의 프레임 범위에 속하는
  모든 인덱스를 커버한다.

## oracle (오라클)

```
cargo test -p es-data label_on_mismatched_metadata_is_an_error_not_a_panic
```

`collect_into`로 데이터셋을 만들고, `meta/episodes.jsonl`을 직접 편집하여 에피소드 0의
선언된 `length`를 `collect_into`가 실제로 쓴 parquet 파일보다 짧게 만든 다음,
`es_data::label(&root, &[])`를 호출하고 패닉이 아니라 `DataError::Inconsistent`를
반환하는지 검증한다.

## acceptance (수용 기준)

- `meta/episodes.jsonl`의 길이가 parquet보다 짧은 데이터셋에 대한 `label`은 패닉이 아니라
  `Err(DataError::Inconsistent { .. })`(기존의 `String` 페이로드 변형)를 반환한다.
- 마스크 인덱싱 경로에 `unwrap`, `expect`, `.get(..).unwrap_or(..)`가 새로 도입되지
  않았다.
- `cargo test -p es-data`는 기존 `loop_learning.rs`와 `lerobot.rs` 픽스처를 포함해 계속
  통과한다.

## forbidden (금지)

- `DataError` enum의 형태를 바꾸지 않는다(새로운 구조체 형태 `Inconsistent` 변형 금지) —
  `es-data` 전체의 기존 `DataError::Inconsistent(String)` 호출부를 전부 고쳐야 하는데
  행동상 얻는 것이 없다.
- `crates/es-eval`, `crates/es-splat`, 그 외 다른 `crates/es-*` 크레이트는 수정하지
  않는다.
- 테스트는 `crates/es-data/tests/loop_learning.rs`에 추가만 한다 — 그 파일의 기존
  테스트나 픽스처는 수정하지 않는다.
