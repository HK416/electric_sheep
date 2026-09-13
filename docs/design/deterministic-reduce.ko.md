<!-- Korean translation of docs/design/deterministic-reduce.md. The English file is the working copy; regenerate this when it changes. -->

# 결정적 리덕션 (`es_math::reduce`) — 설계

Spec refs: §18.4 (결정적 리덕션), §3.4 (결정적 실행 계약), Appendix B.8 (signature),
Appendix D `DET-011`.

## 핵심이 되는 불변식

§18.4에 따르면: **`merge`는 결합적(associative)이고 교환적(commutative)이어야 한다.**
그렇다면 결과는 서브그룹 순서, 워크그룹 병합 순서, SM 개수에 의존하지 않으며, CPU와
GPU에서의 같은 리덕션이 같은 비트를 낸다. 아래 내용은 전부 이것을 성립시키기 위한 것이다.

`KahanAcc`는 의도적으로 **제공하지 않는다**: 보정합(compensated summation)도 여전히
순서에 의존한다(보정 항이 도착 순서에 의존하므로) — 즉 이 불변식을 만족하지 못한다.

## 표현

Appendix B.8이 형태를 고정한다:

```rust
pub trait DeterministicAcc<T>: Default + Clone {
    fn add(&mut self, v: T);
    fn merge(&mut self, other: &Self);
    fn finish(&self) -> T;
}
pub struct BinnedAcc<const K: usize = 3> { bins: [f64; K], index: Option<i32> }
```

`index`는 지금까지 담긴 값 중 가장 큰 크기(magnitude)의 bin 인덱스이며, `None`은 비어
있음을 뜻한다. 인덱스가 `i`인 accumulator의 bin `j`는 다음 지수 구간을 담당한다.

```
[ 2^(-W·(i+j+1)) , 2^(-W·(i+j)) )        W = 26 bits per bin, K = 3 bins (spec default)
```

따라서 모든 accumulator의 모든 bin 경계는 하나의 전역 격자 `{ 2^(-W·n) }` 위에 놓인다.
이것이 (아래에서 볼) 재스케일을 정확하게 만드는 성질이다.

## 적재(Deposit) — 반올림 없이 정확하게

Demmel–Nguyen 방식의 Binned/RFA 합산이되, `1.5·2^e` 분할 기법이 아니라 **유효숫자
(significand)에 대한 절단(truncation)**으로 슬라이싱한다.

```
add(x):  grow index so that |x| < 2^(-W·index)
         r = x
         for j in 0..K:
             part    = truncate r at exponent (-W·(index+j+1))   // clear low bits, toward zero
             bins[j] += part
             r       -= part                                      // exact: part is a bit-prefix of r
```

두 연산 모두 정확하다.

- `part`는 `r`에서 하위 유효숫자 비트를 지운 것이므로, `r - part`는 정확히 표현 가능하다
  — `r`의 유효숫자 중 버려진 접미(suffix) 부분이기 때문이다.
- `bins[j] += part`가 정확한 이유는, bin `j`에 더해지는 모든 값이 `2^(-W·(index+j+1))`의
  배수이면서 `2^(-W·(index+j))`보다 작기 때문이다 — 합은 같은 양자(quantum)의 배수로
  유지되고, `|bin| < 2^(53-W)` 양자인 동안 `f64`는 이를 정확히 표현한다.

정확한 덧셈은 자명하게 결합적이고 교환적이므로, 반올림 오차에 대한 논증이 아니라 구성
자체로 불변식이 성립한다. 절단은 0 방향으로 이루어지며 `x`와 인덱스에만 의존할 뿐
accumulator의 현재 내용에는 의존하지 않는다 — 따라서 어떤 round-to-nearest 동률 처리도
도착 순서에 의존하지 않는다.

**W의 트레이드오프.** `W`는 정밀도를 사는 대신 항(term) 예산을 쓴다.

| W | 최대 항 아래로 보장되는 비트 수 (`W·(K-1)`) | bin이 반올림되기 전까지의 항 개수 (`2^(53-W)`) |
|---|---|---|
| 20 | 40 | 8.6·10^9 |
| **26** | **52** — `f64` 자체가 지니는 정밀도 | **1.3·10^8** |
| 32 | 64 | 2.1·10^6 |

`W = 26`이 선택된 지점이다: `f64` 유효숫자를 온전히 커버하면서, 항 개수 상한은 약
1.3·10^8이 된다. 이 상한은 병합까지 포함한 리덕션 *전체*에 걸린다 — `merge`는 bin
내용을 더하는 것이므로, 작업을 여러 accumulator로 나눈다고 예산이 늘어나지는 않는다.
이를 넘으면 정확성 논증이 깨지고 bin이 반올림되기 시작한다 — 그만큼 큰 리덕션에는 더
큰 `K`(그리고 그에 비례해 작은 `W`)가 필요하며, 그래서 `K`가 const 매개변수로 되어
있다.

## 재스케일과 병합

더 작은 인덱스로 재스케일하는 것(더 큰 값이 들어온 경우)은 bin들을 `d`칸만큼 아래로
밀고 바닥으로 빠지는 것을 버린다. 모든 경계가 공통 격자 위에 있으므로, 이는 처음부터
모든 이전 입력을 *더 새롭고 더 성긴* 경계에서 절단했던 것과 정확히 같다 — 따라서 결과는
최종 인덱스에만 의존하고, 값들이 도착한 순서에는 의존하지 않는다.

`merge`는 양쪽을 `min(index)`로 재스케일한 뒤 bin을 원소별로 더한다: 정렬된 bin에 대한
`+`가 정확하므로 결합적이고 교환적이다.

## finish

`finish`는 가장 작은 구간부터 가장 큰 구간까지, 고정된 순서로 bin들을 합산한다. 이것이
전체 알고리즘에서 유일한 반올림이며, bin 내용의 순수 함수이므로 재현 가능하다.

## 잃는 것

`K`개의 bin은 최상단 경계 아래로 `W·K = 78` 진 이진 자릿수를 포괄한다. 인덱스가 `W`
단위로 양자화되므로, 최대 *항* 아래로 보장되는 커버리지는 `W·(K-1) = 52` 비트이고, 그
아래의 기여분은 잘려나간다. 이것이 binned summation의 의도된 트레이드오프다 — 순서
독립성을 사는 대신, `K`가 그 손잡이다 (`BinnedAcc<4>`는 accumulator당 `f64` 하나를
더 써서 26 자릿수를 더 확보한다).

절단은 0 방향이므로, 같은 부호 값의 긴 합은 편향 없는 것이 아니라 0 쪽으로 약간
편향된다. 결정적 round-to-nearest는 이 편향을 없애지만 순서 의존적인 동률 처리를 다시
들여오게 되므로 쓰지 않는다.

비유한(non-finite) 입력은 accumulator를 오염시킨다(bin 0에 그대로 더해져
`merge`/`finish`를 거쳐 전파된다) — 재현성은 유한한 입력에 대해서만 주장한다.

## 오라클

`cargo test -p es-math reduce::` — 무작위 `f64` 벡터에 대한 proptest로, (a) 원래 순서,
(b) 임의의 순열, (c) 같은 원소들에 대한 임의의 분할-병합 트리에서 `finish()`의 비트가
동일함을 확인한다. CPU `BinnedAcc`와 GPU 서브그룹 리덕션 사이의 비트 동등성은
**Target / Status: 미검증**이다 (M2 Vulkan 경로가 필요하다).
