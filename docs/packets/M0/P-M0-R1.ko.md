<!-- Korean translation of docs/packets/M0/P-M0-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M0-R1 — `BinnedAcc` 포이즌(poison) 순서 문제 (M0 리뷰 블로커)

Spec: §18.4 (결정적 리덕션), §3.4 (결정적 실행 계약), Appendix B.8 (signature), Appendix D
`DET-011`. M0 리뷰(`docs/reviews/M0.md`, Blocker)의 후속 작업. 리뷰 등급 A.

## context

```
crates/es-math/src/reduce.rs
docs/design/deterministic-reduce.md
docs/packets/M0/P-M0-R1.md
```

## spec

- `add`는 어떤 bin 연산도 수행하기 **전에** 비유한(non-finite) 입력을 전용 포이즌 상태
  (poison state)에 기록하고 반환한다. 비유한 값이 `bins`에 들어가는 일은 결코 없다. 이전
  코드는 `bins[0] += v`를 쓴 뒤 `rescale_to(0)`을 호출했는데, 이전 적재로 인덱스가 이미 더
  성긴(coarser) 값으로 설정되어 있으면 포이즌이 bin 배열 바닥으로 밀려나 사라졌다 —
  `add(1e-300); add(NAN)`은 `0.0`으로 끝났고, `add(NAN); add(1e-300)`은 `NaN`으로 끝났다.
- 이 상태는 `u8` 비트 집합(`+Inf` / `-Inf` / `NaN`)이므로, `merge`는 이를 `|`로 결합한다:
  교환적이고 결합적이며 멱등적(idempotent)이다. 할당이 없고 여전히 `Copy`다.
- `finish`는 bin 합보다 먼저 IEEE `sum` 의미론으로 이를 해석한다: `NaN`이 우선하고, `+Inf`와
  `-Inf`가 함께 있으면 `NaN`이며, 무한대가 하나만 있으면 유한 항들을 압도하고, `NaN`은
  `f64::NAN` 상수로 반환되므로 순열이 달라져도 `to_bits()`가 안정적이다.
- 어느 한쪽이 비어 있을 때도 `merge`는 포이즌을 전파해야 한다 — 빈 accumulator가 포이즌된
  것과 병합되면 포이즌되고, 포이즌된 것이 빈 것과 병합되면 계속 포이즌 상태로 남는다.
- 이제 비유한 입력에 대해서도 재현성을 주장한다; 설계 문서의 "유한 입력에 대해서만" 이라는
  단서는 `docs/design/deterministic-reduce.md`의 표로 대체된다.

## oracle

```
cargo test -p es-math reduce
```

수정 전에 작성됨: 새로 추가된 두 proptest는 이전 구현에서 실패한다.

## acceptance

- Proptest `poisoned_permutation_is_bit_identical`: `NaN`, `±Inf`, `±f64::MIN_POSITIVE`,
  가장 작은 서브노멀(subnormal) 값과 정규 값들로 이루어진 벡터에 대해, 원래 순서와 뒤섞은
  (shuffle) 순서의 `finish().to_bits()`가 동일하다.
- Proptest `poisoned_split_merge_tree_is_bit_identical`: 같은 벡터들을 임의의 분할-병합
  트리로 리덕션한 결과가 선형 리덕션과 같은 비트를 낸다.
- `non_finite_poisons_whatever_the_order`: `[1e-300, NAN]`은 어느 순서든 `NaN`이다;
  `[+Inf, -Inf]`는 어느 순서든 `NaN`이다; `[1.0, +Inf]`는 `+Inf`다; `[1e300, -Inf]`는 `-Inf`다.
- `poison_survives_a_merge_from_either_side`: 기본(default) accumulator로 병합하는 경우도
  포함한다.
- 기존의 유한 입력 속성들(순열, 분할-병합, 교환성, 결합성, 정확도)은 변경 없이 그대로
  통과한다.

## forbidden

`context` 밖의 모든 파일. `DeterministicAcc` trait 시그니처 변경이나 확장 지점(extension
point) 추가(INV-17). `KahanAcc` 추가. accumulator에서의 힙 할당. `finish`가 `Result`를
반환하도록 만드는 것. `crates/es-math/src/approx.rs`, `scalar.rs`, `simd.rs`를 건드리는 것.
