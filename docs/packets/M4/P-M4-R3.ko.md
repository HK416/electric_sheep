<!-- Korean translation of docs/packets/M4/P-M4-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R3 — USD 리더의 신뢰 경계 강화 (review S-1, S-2)

Spec: 사양 1.9 항목 6(네이티브 USD 리더는 범위 압박 시 가장 먼저 잘려나가는 것이지만,
존재하는 동안에는 신뢰할 수 없는 `.usda` 텍스트를 읽으므로 사양 1.4의 오라클 우선 규칙이
그 실패 모드에도 적용된다). `docs/reviews/M4.md`의 should-fix S-1과 S-2의 후속 조치.
Review class A.

## context (범위)

```
crates/es-usd/src/lib.rs
crates/es-usd/src/parse.rs
crates/es-physics-core/src/usd.rs
crates/es/src/cmd/import.rs
crates/es/Cargo.toml
crates/es/tests/cli.rs
docs/packets/M4/P-M4-R3.md
```

## spec (사양)

세 개의 구멍이며, 모두 "공격자가 제어하는 `.usda` 파일"이고, 어느 것도 의미론 버그가
아니다:

- **S-1** (`es-physics-core/src/usd.rs`, 이전 `:514`) — `mesh_asset`은 `faceVertexCounts`
  entry를 `count as usize`로 읽었다. Rust의 float-to-int 캐스트는 래핑이 아니라
  saturate하므로 `1e20`은 `usize::MAX`가 되었으며, 그 자체로는 패닉이 아니다 — 하지만
  이후 `cursor + n > indices.len()`에서 비교가 실행되기 전에 *덧셈*(`cursor + n`) 자체가
  오버플로했고, 이는 release에서 wrap되어 가드를 통과할 수 있으며, 유효 범위와 전혀
  무관한 `cursor`로 `indices[cursor + offset]`에 도달하게 된다. 수정: `usd_face_vertex_count`는
  유한한 정수이면서 `u32`에 들어맞고 3 이상인 것이 아니면 무엇이든 거부한다
  (`UsdSceneError::Invalid`, prim 경로 첨부), 그리고 진행 중인 `cursor`는 raw `+` 대신
  `checked_add(n).filter(|c| *c <= indices.len())`을 통해 전진한다.
- **S-2** (`es-usd/src/parse.rs`, 이전 `:635,650`) — `MAX_DEPTH`는 `def` 안에 `def`가
  중첩되는 prim 중첩(`prim`/`prim_body`, `:472`에서 검사됨)만 센다. `raw_value`와
  `raw_sequence`는 자체 카운터 없이 모든 `[...]`/`(...)` 리터럴마다 서로 재귀 호출하므로,
  충분히 깊은 `[[[[...]]]]`는 네이티브 스택을 오버플로한다 — `UsdError`가 아니라
  abort이며, 호출자가 잡을 수 없다. 수정: `Parser`에 `value_depth: usize` 필드를 두고, 두
  재귀 지점이 이제 모두 호출하는 `nested_sequence` 래퍼 안에서 새 `MAX_VALUE_DEPTH = 64`에
  대해 증가시키고 검사하며, 초과되면 해당 줄과 함께 `UsdError::Syntax`를 낸다. 별개로,
  `lex`(`:180`, 이 패킷 이전에는 번호가 없었다)는 `es/src/cmd/import.rs`(`:297`)가
  디스크에서 읽은 것이 무엇이든 크기 검사 전혀 없이 레이어 전체를 `Vec<char>`로
  확장한다 — 큰 파일은 토큰 하나가 생기기도 전에 상한 없는 할당을 유발한다. 수정: 새로운
  `es_usd::MAX_USDA_BYTES`(64 MiB)를 `strip_header`/`lex`가 실행되기 전, `parse_usda` 맨
  위에서 `text.len()`에 대해 검사하고, 같은 상수를 CLI의 `usd_import`가 `read_to_string`을
  호출하기 전 `std::fs::metadata(&path).len()`에 대해 검사한다 — 읽은 뒤 거부하는 것이
  아니라 파일 크기로 거부한다.

하지 않은 것: `Vec<char>` 대신 `char_indices`/바이트로 다시 렉싱하는 것. `lex_string`,
`lex_delimited`, `lex_number`는 모두 `chars: &[char]`를 요소 위치로 인덱싱하고 `char` 값을
직접 비교한다; 토크나이저 전체를 바이트 오프셋으로 바꾸는 것은 이들 모두를 건드려야
하는데, 얻는 메모리 이득은 새 크기 상한이 이미 제한하는 것이다(64 MiB의 UTF-8은 많아야
~64M개의 `char`가 되며, 어느 쪽이든 고정된 상한이다). should-fix 패킷의 범위를 벗어난다;
가능한 후속 작업으로 표시해두며 여기서는 시도하지 않았다.

## oracle (오라클)

```
cargo test -p es-usd -p es-physics-core
cargo test -p es usd
```

수정이 반드시 깔끔한 `Err`로 바꿔야 하는 세 케이스이며, 어느 것도 크래시가 아니다:

- `usd::tests::a_face_vertex_count_that_overflows_u32_is_an_error_not_ub`
  (`es-physics-core/src/usd.rs`) — 기존 `mesh_cube.usda` 픽스처에 대한
  `faceVertexCounts = [3, 1e20, ...]`(테스트 내에서 편집, 새 픽스처 파일 없음)는 패닉이나
  범위 밖 인덱스가 아니라 `faceVertexCounts`를 지목하는 `UsdSceneError::Invalid`를
  반환한다.
- `parse::tests::deep_value_nesting_errors_instead_of_overflowing`
  (`es-usd/src/parse.rs`) — `[` 100 KB(테스트 내에서 생성, 픽스처 파일 아님)는 프로세스를
  abort하는 대신 `UsdError::Syntax`를 반환한다. 기본 `cargo test` 스택에서 실행되며
  (`--test-threads=1` 같은 특별 처리 불필요) 같은 테스트 바이너리 안에서 1초를 훨씬 밑도는
  시간에 완료된다. 이는 "타임아웃이 있는 스레드로 감싸기"의 인프로세스 형태다: 실제 스택
  오버플로였다면 `Err`를 반환하는 대신 테스트 바이너리 전체를 abort시켰을 것이므로, 테스트가
  통과한다는 것 자체가 증거다.
- `parse::tests::oversized_layer_is_rejected_before_the_char_vec`
  (`es-usd/src/parse.rs`)와 `import_usd_refuses_an_oversized_file` (`es/tests/cli.rs`) —
  `MAX_USDA_BYTES + 1`바이트 레이어(테스트 내에서 `"a".repeat(...)` / `vec![b'a'; ...]`로
  구성, 픽스처 파일 없음, `tests/fixtures/usd/`를 200 KB 미만으로 유지)는 파서 자체에
  의해서도, CLI의 `usd_import`에 의해서도(읽기 전에 파일 메타데이터 길이로) 거부된다.

## acceptance (수용 기준)

- `cargo fmt -p es-usd -p es-physics-core -p es --check`가 클린하다.
- `cargo clippy -p es-usd -p es-physics-core -p es --all-targets -- -D warnings`가
  클린하다.
- `cargo test -p es-usd -p es-physics-core` — 19개(`es-physics-core`) + 15개(`es-usd`)
  테스트가 통과하며, 기존 테스트는 모두 변경 없이 여전히 green이고, 새 테스트 세 개가
  추가되었다.
- `cargo test -p es usd` — 기존의 `import_usd_*` CLI 테스트 두 개에 더해 새
  `import_usd_refuses_an_oversized_file`이 통과한다.
- `tests/fixtures/usd/`에 새 픽스처 파일이 추가되지 않았다(세 악의적 입력 모두 테스트
  내에서 생성됨); 디렉터리는 패킷 이전 크기를 유지한다.
- `UsdError`나 `UsdSceneError`의 variant 형태에 변화가 없다 — 두 악의적 입력 모두 새 오류
  종류가 아니라 메시지를 담은 기존 `Syntax`/`Invalid` variant로 매핑된다.

## forbidden (금지)

`es-script`, `es-eval`, `es-data`, `es-env`, `es-ir`, `es-gpu`, `es-render`,
`es-core`(이 crate들에 대해 진행 중인 다른 리뷰 후속 작업들). `es/src/cmd/import.rs`의
RoboVerse 분기(`roboverse_import`, S-9/S-10) — 이 패킷은 `usd_import`의 읽기만 건드린다.
`es-usd::parse::lex`를 `Vec<char>`에서 벗어나게 다시 렉싱하는 것(위에서 언급, 시도하지
않음). `mesh_asset`이 수행하는 매핑에서 count/cursor 검사를 넘어서는 어떤 변경이든 —
geometry 의미론은 범위 밖이다.
