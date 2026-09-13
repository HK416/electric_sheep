<!-- Korean translation of docs/packets/M3/P-M3-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R3 — PLY 리더의 reserve 상한 설정

Spec: §1.4(오라클 우선), §3.4/부록 D(신뢰할 수 없는 입력으로부터의 무제한 할당 금지),
§16(이 크레이트의 범위). `docs/reviews/M3.md`의 Should-fix `ply.rs:441`과
`W3-splat-importer.md`(이 패킷은 그 범위 안에 머문다)에 대한 후속 조치다.

`import_ply`의 ascii 경로는 줄 단위 필드 개수 검사(`fields.len() != header.props.len()`)가
실행되기도 전에, 헤더가 *선언한* `element vertex` 개수로부터 곧바로 `3 * count`,
`4 * count`, `per_rest * count`개의 원소를 reserve했다. 악의적인 ascii 헤더는 파일이
실제로 담을 수 있는 것보다 훨씬 많은 정점 수를 선언할 수 있다 — 파일 자체의 비어 있지
않은 줄 수는 줄당 2바이트라는 느슨한 기준으로만 `count`를 제한한다 — 그래서 SH 3차(59개
프로퍼티) 전체 헤더에 대해 `element vertex 500000`을 선언한 ~1 MB짜리 파일은 첫 레코드가
검증되기도 전에 약 118 MB를 요청했다. 바이너리 경로는 이미 안전했다: `Truncated` 검사
(`available < count * stride`)가 reserve *이전에* 실행되어, 파일이 실제로 담을 수 없는
어떤 `count`든 거부한다.

## context (범위)

```
crates/es-splat/src/ply.rs
crates/es-splat/tests/splat.rs
crates/es-splat/Cargo.toml
docs/packets/M3/P-M3-R3.md
```

## spec (사양)

1. **`reserve_count(format, props, stride, count, available) -> usize`** — `ply.rs`의
   private 헬퍼로, `import_ply`가 모든 `SplatScene` 배열의 `Vec::with_capacity`에서 헤더의
   raw `count` 대신 사용한다. `available` 바이트가 실제로 담을 수 있는 전체 정점 레코드
   수로 reserve를 제한한다: ascii의 경우 `count.min(available / (2 * props))`(레코드
   하나는 프로퍼티당 최소 숫자 1개와 구분자 1개가 필요하다), binary의 경우
   `count.min(available / stride)`(기존 `Truncated` 검사에 이미 함의되어 있던 것을
   부수적인 것이 아니라 명시적이고 직접적인 상한으로 만든다).
2. `import_ply`의 제어 흐름, 오류 variant, 또는 각 정점을 디코딩하는 루프는 변경하지
   않는다: `reserve_count`는 사전 용량(capacity)만 정하므로, 보수적인 추정치보다 실제
   레코드가 더 많은 파일도 일반적인 `Vec` 증가를 통해 여전히 올바르게 디코딩된다.
3. `es-splat`의 `[dev-dependencies]`에 `alloc-count` feature를 가진 `es-core`가
   추가된다(`es-safety`와 `es-runtime-embedded`가 이미 쓰고 있는 것과 같은 패턴). 이를
   통해 테스트가 `cfg(test)` 아래에서 할당 횟수를 관찰할 수 있다.

## oracle (오라클)

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask check-spec-refs
```

구체적으로: `ply::reserve_tests::ascii_reserve_is_bounded_by_file_size_not_declared_count`는
`reserve_count`의 출력을 직접 고정한다 — 악의적인 매개변수(`count = 500_000`,
`available = 1_000_000`, `props = 59`)에 대해 여섯 배열 전체에 걸친 최악의 reserve(예약된
정점당 59개의 `f32` 슬롯)는 8 MB 미만이며, 이는 수정 전 `3 * count` / `per_rest * count`
호출이 함의하던 ~118 MB와 대비된다. `tests/splat.rs`의
`a_hostile_ascii_header_does_not_amplify_the_reserve`는 줄당 토큰 하나로
`element vertex 500000`을 선언하는, 생성된 ~1 MB짜리 ascii PLY를 임포트하여
`SplatError::BadFieldCount { vertex: 0, found: 1, expected: 59 }`를 검증하고, 더불어
(`es_core::alloc_count::allocation_count`를 통해) 임포트가 예약 없이 push마다 증가하는
경로가 필요로 할 수십만 번이 아니라 수백 번 수준으로 할당한다는 것을 검증한다 — 바이트
상한 자체는 위의 순수 함수 테스트가 고정하는데, 이 카운터는 바이트가 아니라 호출 횟수를
세기 때문이다.

## acceptance (수용 기준)

- 악의적인 헤더 테스트가 통과한다; 바이트 단위로 정확한 라운드트립 테스트와 ascii/binary
  교차 디코드 테스트는 영향을 받지 않는다(이들은 정상적인 입력에 대해 `reserve_count`를
  실행하는데, 정상적인 파일은 실제 레코드당 항상 `2 * props`바이트보다 많은 여유가
  있으므로 이 값은 실제 `count`와 같아진다).
- `binary_reserve_is_bounded_by_stride`는 binary 경로의 상한이 변하지 않았음을
  고정한다(`available / stride`이며, 기존 `Truncated` 검사가 이미 보장하던 것과 일치).

## forbidden (금지)

- `SplatError` variant, 줄 단위 필드 개수 검사, 또는 binary `Truncated` 검사의 조건을
  변경하는 것 — 이 패킷은 무엇이 유효한 입력으로 간주되는지가 아니라 사전에 *얼마나*
  reserve하는지만 바꾼다.
- `crates/es-splat/**`와 이 패킷 파일 바깥의 그 무엇이든. `es-core`의 `alloc-count`
  feature는 dev-dependency로 소비될 뿐 결코 수정되지 않는다.
- 커밋. 오라클은 실행되고 보고될 뿐, 랜딩(land)되지 않는다.
