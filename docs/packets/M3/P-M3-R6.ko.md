<!-- Korean translation of docs/packets/M3/P-M3-R6.md. The English file is the working copy; regenerate this when it changes. -->

# P-M3-R6 — 공선(collinear) 대응점은 퇴화된 피팅이다

Spec: §16.2(위치 정렬, `Similarity::fit`), §1.4(오라클 우선). `docs/reviews/M3.md`의
Should-fix `align.rs:97`과 `W3-splat-importer.md`(이 패킷은 그 범위 안에 머문다)에 대한
후속 조치다.

`Similarity::fit`의 문서는 이미 "퇴화된 입력은 오류이며, 조용히 항등원으로 처리되지
않는다"고 약속했고, 공선이 아닌 세 점이 회전을 결정한다고 명시했다. 실제로 구현된 검사는
`src_spread <= 0.0` 하나뿐이었는데, 이는 (모든 방향으로 퍼짐이 0인) 일치하는 점들은
잡아내지만 공선인 점들은 잡아내지 못한다: 한 직선 위의 네 점은 양의 `src_spread`를
가지지만 그 직선을 *축으로 하는* 회전은 잘 정의되지 않으므로, Horn의 방법은 약속된 오류
대신 유한하고 임의적인 회전을 조용히 반환했다.

## context (범위)

```
crates/es-splat/src/align.rs
crates/es-splat/tests/splat.rs
docs/packets/M3/P-M3-R6.md
```

## spec (사양)

1. **`Similarity::fit`의 랭크(rank) 검사** — 기존의 일치점 검사(`src_spread <= 0.0` →
   `SplatError::DegenerateFit`, 변경 없음) 다음에, 노름(norm)이 가장 큰 중심화된 `src`
   점(`axis`, 고정되고 순서에 무관한 선택: 반복 순서상 실행 중 최댓값 노름을 갱신하며
   처음 도달한 점)과, 모든 중심화된 점이 `axis`에 수직인 성분의 제곱합(`perp_spread`, 두
   번째 주성분 퍼짐)을 계산한다. `perp_spread`가 `src_spread`에 비해 무시할 만한
   수준일 때(`<= src_spread * f64::EPSILON.sqrt()`), 모든 대응점은 중심점을 지나는
   `axis` 직선 위에 있다: `SplatError::DegenerateFit`.
2. 전체 연산 순서는 고정된다(`axis`/`axis_norm2`는 `s`, `src_spread`와 같은 패스에서
   누적하고; 수직-퍼짐 패스는 `src`를 순서대로 한 번만 순회한다), 이는 파일에 기존부터
   있던 "데이터에 의존하는 반복 횟수 없음, 재결합(reassociation) 없음"이라는 원칙과
   일치한다 — 이는 성능이 아니라 재현성(reproducibility)에 관한 속성이다.
3. 일치점 검사, `TooFewPoints`/`PointCountMismatch`/`NonFiniteSample`, 또는
   `horn_rotation`/`jacobi_eigen`에는 변경이 없다.

## oracle (오라클)

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask check-spec-refs
```

구체적으로: `tests/splat.rs`의 `similarity_fit_refuses_collinear_input`은 한 직선 위의
네 점(좌표축에 정렬됨, `src == dst`)을 피팅하여 `SplatError::DegenerateFit`을 검증한 뒤,
좌표축에 정렬되지 않은 공선 `src`를 (역시 공선인) 다른 `dst`에 대해 다시 검사한다 — 즉 이
검사가 "항등원"이나 "좌표축을 따르는 경우"의 특수 사례가 아님을 확인한다. 기존의
`similarity_fit_refuses_degenerate_input`(일치점)과
`similarity_fit_recovers_a_known_transform`(일반적으로 공선이 아닌 점 10개짜리 의사난수
시드 64개)은 변경 없이 그대로 통과한다.

## acceptance (수용 기준)

- 공선인 네 점은 유한한 회전이 아니라 `SplatError::DegenerateFit`이 된다.
- 일치점 케이스와 64-시드 복원 속성은 영향받지 않는다: 무작위 3차원 점들은 사실상 결코
  공선이 되지 않으므로, 진정으로 well-posed한 입력에서는 `perp_spread`가 엡실론
  임계값보다 훨씬 위에 머문다.

## forbidden (금지)

- 일치점 검사의 임계값이나 오류 variant를 변경하는 것.
- 새로운 오류 variant를 추가하는 것 — 공선 입력은 문서 자체의 표현("퇴화된 입력은
  오류다")에 따라 `SplatError::DegenerateFit`을 재사용한다.
- `ColorAffine::fit`, `Binding::bind`/`skin`, 또는 `ply.rs` 안의 그 무엇이든 — 이 패킷은
  오직 `align.rs`의 랭크 검사만 다룬다.
- `crates/es-splat/**`와 이 패킷 파일 바깥의 그 무엇이든.
- 커밋. 오라클은 실행되고 보고될 뿐, 랜딩(land)되지 않는다.
