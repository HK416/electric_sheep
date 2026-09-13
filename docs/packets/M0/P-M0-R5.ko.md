<!-- Korean translation of docs/packets/M0/P-M0-R5.md. The English file is the working copy; regenerate this when it changes. -->

# P-M0-R5 — `BinnedAcc` 항 개수(term-count) 상한 assertion

Spec: §18.4 (결정적 리덕션), §3.4, Appendix D `DET-011`. M0 리뷰(`docs/reviews/M0.md`,
`reduce.rs:33-35`에 대한 Nit)의 후속 작업. 리뷰 등급 C. P-M0-R1을 기반으로 한다.

## context

```
crates/es-math/src/reduce.rs
docs/design/deterministic-reduce.md
docs/packets/M0/P-M0-R5.md
```

## spec

- 정확성 논증은 문서화만 되어 있고 한 번도 검사된 적 없는 어떤 항 개수 아래에서만 성립한다.
  정확한 한계는 다음과 같다: 한 번의 적재는 bin에 `2^W` 양자(quanta) 미만을 더하며(슬라이스가
  bin의 상단 경계인 `2^W` 양자보다 작다), bin은 `2^53` 양자 미만을 담고 있는 동안에만
  정확하다. 따라서 `W = 26`일 때 `MAX_TERMS = 2^(53 - W) = 2^27 = 134_217_728`이다.
- `terms: u32`는 유한한 적재 횟수를 센다. 0과 비유한 입력은 어떤 bin도 건드리지 않으며
  세지 않는다. 카운터는 래핑(wrapping)되지 않고 포화(saturate)한다.
- `merge`는 두 카운트를 **더한다**: 병합이 bin 내용을 더하는 것이기 때문에, 예산은
  accumulator당이 아니라 리덕션당이다. 덧셈은 교환적이고 결합적이므로 카운터가 `DET-011`
  불변식을 깨뜨릴 수 없고, 카운트는 다중집합(multiset)의 함수이므로 어떤 순열에서는 assert가
  발동하고 다른 순열에서는 조용한 일이 있을 수 없다.
- `add`와 `merge` 모두 `debug_assert!(terms <= MAX_TERMS, "... term ceiling ...")`을
  수행한다. 디버그 전용이다: 카운터는 릴리스 커널에서 어떤 비용도 발생시키면 안 된다. 상한을
  넘으면 bin이 반올림되고 순서 독립성이 무너지므로, 그 위반은 디버그 실행과 테스트에서
  반드시 시끄럽게(loud) 드러나야 한다.
- `2^27`번의 실제 적재를 수행하지 않고도 상한에 도달할 수 있도록 `#[cfg(test)]
  set_terms_for_test` 헬퍼가 존재한다.

## oracle

```
cargo test -p es-math reduce
```

## acceptance

- `depositing_past_the_term_ceiling_trips_the_assert`: `#[should_panic(expected = "term
  ceiling")]`이며, 카운터를 `MAX_TERMS`로 시딩(seed)한 뒤 `add`를 한 번 호출한다.
- `merging_past_the_term_ceiling_trips_the_assert`: 같은 상황을 `merge`를 통해 재현하여,
  예산이 accumulator들 사이에 공유됨을 증명한다.
- 두 테스트 모두 `#[cfg(debug_assertions)]`이므로, 릴리스 테스트 실행에서는 실패하지 않는다.
- `docs/design/deterministic-reduce.md`는 한계값 자체뿐 아니라 그 유도 과정도 서술한다.
- 기존의 모든 `reduce` 테스트가 그대로 통과한다; 어떤 테스트도 눈에 띄게 느려지지 않는다.

## forbidden

`context` 밖의 모든 파일. 상한을 런타임 `assert!`나 `Result`로 바꾸는 것. `W`, `K`, 또는
`DeterministicAcc` 시그니처를 변경하는 것. 확장 지점(extension point) 추가(INV-17).
accumulator에서의 힙 할당.
