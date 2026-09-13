<!-- Korean translation of docs/packets/M1/P-M1-R6.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R6 — 노드 종류 목록 동결

M1 리뷰의 Should-fix(`docs/reviews/M1.md`, Should-fix,
`crates/es-ir/src/factory.rs:93,103`; 또한 spec 28.7 게이트 10, "부분적으로" 충족된
것으로 기재됨)를 고친다: `BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS`는 doc 주석에서만
"동결(frozen)"되었다고 선언되어 있었다 — 종류가 이름 변경, 재정렬, 추가, 삭제되어도
CI에서는 아무것도 실패하지 않았는데, 그렇게 하는 것은 그것을 쓰는 모든 그래프의
`task_hash` / `learning_hash`를 조용히 바꾸는 일이었음에도 그랬다.

Spec: spec 6.3, spec 8.3, spec 28.7 게이트 10.

## context (범위)

```
crates/es-ir/src/factory.rs   (kinds_hash helper, two frozen [u8; 32] consts, test module)
docs/packets/M1/P-M1-R6.md    (new)
```

## forbidden (금지)

`crates/es-compile`, `crates/es-telemetry`, `crates/es-eval`, `crates/es-data`,
`.github/**` — 동시 진행 중인 M1 후속 작업이 소유한다. 새 trait 없음 — `kinds_hash`는
확장 지점이 아니라 private free function이다(`INV-17` 영향 없음). `docs/design/ir-types.md`는
이 패킷에서 편집되지 않는다; 그 마이그레이션 노트 요구사항은 다음에 목록을 바꿀 사람을
위해 문서화되어 있다.

## spec (사양)

- `kinds_hash(kinds: &[&str]) -> [u8; 32]`: 종류들을 정렬하고(그래서 소스 배열 안의
  선언 순서는 계약의 일부가 아니다), `\n`으로 이어붙여, `blake3::hash`로 해시한다.
- `BUILTIN_TASK_KINDS_HASH` / `BUILTIN_LEARNING_KINDS_HASH`: 해시를 실행해 그 결과를
  붙여넣어(아래 참고) 한 번 계산된 하드코딩된 `[u8; 32]` 16진수 리터럴. 각각에 대한
  doc 주석: "이 해시를 바꾸는 것은 스키마 변경이다: `schema_version`을 올리고
  docs/design/ir-types.md에 마이그레이션 노트를 추가할 것".
- `factory.rs`의 테스트가 실제 `BUILTIN_TASK_KINDS` / `BUILTIN_LEARNING_KINDS`에 대해
  `kinds_hash`를 다시 계산하여 하드코딩된 상수와의 동등성을 단언하므로, 이름 변경,
  구성원이 바뀌는 재정렬, 추가, 삭제는 `cargo test -p es-ir`를 실패시킨다.

계산된 해시(정렬된 종류 목록을 `\n`으로 이어붙인 것에 대한 blake3), 각각 16진수 32바이트:

```
BUILTIN_TASK_KINDS_HASH     = 079283852d9d3faf85c7da5d48fa0c88210ab5c37c82effc3bb236e2603e144c
BUILTIN_LEARNING_KINDS_HASH = a17d0653f6c0160cfce352a486ca040bbdbdf4b52139a12ffcf0920e1caffe51
```

권위 있는(authoritative) 형태는 `factory.rs` 안의 `[u8; 32]` 16진 배열 리터럴이다
(같은 순서로 바이트마다 하나의 `0xNN`).

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-ir --all-targets -- -D warnings
cargo test -p es-ir
```

## acceptance (수용 기준)

- `cargo test -p es-ir factory::frozen_kind_lists::`가 통과한다: 두 하드코딩된 해시
  모두 새로 계산한 `kinds_hash`와 일치한다.
- `BUILTIN_TASK_KINDS`나 `BUILTIN_LEARNING_KINDS` 중 한 항목을 수동으로 이름 변경하면
  (커밋하지 않고 — 로컬 확인용으로만) 해당 테스트가 실패하며, 이는 게이트 10이 이제
  주석이 아니라 기계적으로 검사됨을 확인해 준다.
- `cargo test -p es-ir` 전체가 여전히 통과한다(`factory_basic.rs`의 기존 커버리지
  테스트는 영향받지 않는다 — 여전히 해시가 아니라 목록 자체를 직접 실행한다).
