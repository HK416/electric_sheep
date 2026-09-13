<!-- Korean translation of docs/design/transcendental.md. The English file is the working copy; regenerate this when it changes. -->

# 초월함수 (`es_math::approx`) — 설계

Spec refs: §3.2 (초월함수), §3.4 (결정적 실행 계약), Appendix D `DET-010`.

## 자체 구현이 필요한 이유

SPIR-V `GLSL.std.450`은 ULP 상한만 규정할 뿐, 실제 비트 패턴은 벤더·드라이버에 따라 달라진다.
`libm`도 같은 이유로 플랫폼마다 다르다. 둘 다 §3.5의 tier-1 비트 단위 보장을 깨뜨리므로,
physics / observation / reward 커널은 `f32::sin` 등을 직접 호출하는 대신 `es_math::approx`를
호출해야 하며, Rust 쪽과 Slang 쪽은 계수와 *연산 순서*를 모두 공유해야 한다.

## 계약

- **입출력 타입은 `f32`다.** 커널은 FP32로 실행된다 (§3.3).
- **연산 순서는 고정된다** — Horner 방식으로, 다항식마다 하나의 식으로 풀어 써서
  `src/approx/coeffs.rs` + `src/approx/mod.rs`와 `slang/approx.slang`에서 동일한 순서를
  유지한다. 재결합(reassociation)도, `mul_add`도 쓰지 않는다 (FMA가 있으면 호스트에 FMA
  유닛이 없을 때 결과가 달라지고, Slang/SPIR-V 쪽도 축약(contract)해서는 안 된다 — GPU
  쪽에는 `NoContraction`이 요구된다, §3.4).
- **`fast-math` 금지, 테이블 조회 금지, 디바이스 속성에 따른 분기 금지.**
- **각 함수의 주 정의역(primary domain)에서 목표는 ≤ 2 ULP.**

## 범위 축소 (Range reduction)

| fn | 주 정의역 | 축소 방법 |
|---|---|---|
| `sin`, `cos` | `|x| ≤ 1e3` | Cody–Waite: `n = floor(x·2/π + 0.5)` (`floor`이며 `round`이 아니다 — HLSL/Slang의 `round`은 round-half-to-even이고 Rust는 round-half-away이므로, 미러 구현은 이를 쓸 수 없다), `r = ((x − n·DP1) − n·DP2) − n·DP3`, π/2를 약 48비트 정밀도로 나눠 담은 `f32` 상수 세 개를 사용한다. 사분면 `n & 3`이 sin- 또는 cos-커널과 부호를 결정한다. |
| `exp` | `[-88, 88]` (f32 범위) | `n = floor(x·log2e + 0.5)`, `r = (x − n·LN2_HI) − n·LN2_LO`, `r`에 대한 다항식을 계산한 뒤 `f32::from_bits`로 만든 (정확한) `2^n`을 곱해 스케일한다. |
| `ln` | `(0, f32::MAX]` | 지수 필드 추출을 통해 `m ∈ [√½, √2)`인 `x = m·2^e`로 분해하고, `m − 1`에 대한 다항식을 계산한 뒤 `LN2_HI/LN2_LO`로 나눈 `+ e·ln2`를 더한다. |
| `atan2` | 유한한 모든 `(y, x)` | `|y/x|` 또는 `|x/y|`에 대해 `atan`을 계산하며, 이는 다시 `t → (t−1)/(t+1)`로 `[0, tan(π/8)]`까지 축소한 뒤 사분면을 보정한다. `π`, `π/2`, `π/4` 각각은 `_HI`/`_LO` 쌍(참값에 가장 가까운 `f32`와 그 잔차인 `f32`)으로 유지되며, `PI - a`가 아니라 `(PI - a) + PI_LO`로 재구성하는 것이 `atan2`를 3 ULP에서 2 ULP로 낮추는 지점이다. |
| `sqrt`, `rsqrt` | `[0, ∞)` | 축소 없음, 다항식도 없음: `sqrt`는 단일 연산으로 올바르게 반올림되는 IEEE-754 연산이며, SPIR-V의 `OpSqrt`/`OpFDiv` 역시 ≤ 0.5 ULP가 요구된다. 따라서 `approx::sqrt`는 `x.sqrt()`이고 `approx::rsqrt`는 `1.0 / x.sqrt()`이다. 벤더의 fast-rsqrt 명령은 정확도가 규정되어 있지 않으므로 **쓰지 않는다**. |

주 정의역을 벗어나도 함수는 (IEEE 특수값을 명시적으로 처리하여) 여전히 합리적인 값을
반환하지만, 그 경우 ULP 상한은 보장되지 않는다. `sin`/`cos`의 경우 `|x| ≈ 1e3`를
넘어서면 3단 Cody–Waite 축소가 π 비트 정밀도를 소진한다. Payne–Hanek 축소는 의도적으로
**구현하지 않는다** (사양의 어떤 커널도 이를 필요로 하지 않고, GPU 분기(divergence) 비용을
감수할 가치도 없다).

## 계수 출처

`src/approx/coeffs.rs`의 다항식 계수는 **Cephes Math Library**(S. L. Moshier, `sinf.c`,
`expf.c`, `logf.c`, `atanf.c`)의 단정밀도 minimax 계수이며 퍼블릭 도메인이다. 이 계수들은
위에 나열한, 여기서 쓰이는 것과 정확히 같은 축소된 정의역에 대해 Remez 교환 알고리즘으로
피팅된 것이다. 반올림도, 재유도도 없이 소수 리터럴 그대로 옮겨 적었으므로, Rust 파일과
Slang 파일을 리터럴 단위로 비교할 수 있다 (P09). `coeffs.rs`에서
`clippy::unreadable_literal`, `clippy::excessive_precision`, `clippy::approx_constant`를
허용하는 이유도 이 때문이다 — 자릿수 구분자, 소수점 축약, `std::f32::consts`로의 치환
어느 것이든 Slang 미러와의 바이트 단위 비교를 깨뜨린다.

계수 세트를 다시 피팅해야 한다면 **두 파일 모두** 같은 변경에서 다시 피팅해야 하며,
아래의 ULP 테스트가 그 게이트다.

## 정확도 — 실측

`cargo test -p es-math approx::`는 각 함수를 주 정의역 전체에 대해 스윕(조밀한 선형
스윕과 결정적 의사난수 스윕 각 ≥ 200k점)하고, 이 정의역들에서 참값의 ≤ 0.5 ULP인 `f64`
`std` 구현을 `f32`로 반올림한 값과 비교한다 — 즉 `f32` 정밀도에서 MPFR을 대신할 유효한
대안이다. 테스트는 함수별 실측 최대 ULP 오차를 출력하고 2 ULP를 넘으면 실패한다.

x86-64(Windows, rustc 1.85, debug 프로파일)에서 `cargo test -p es-math approx:: --
--nocapture`로 측정:

| fn | 스윕한 정의역 | 실측 최대 ULP |
|---|---|---|
| `sin` | `[-1e3, 1e3]` and `[-6.5, 6.5]` | 1 |
| `cos` | `[-1e3, 1e3]` and `[-6.5, 6.5]` | 1 |
| `exp` | `[-88, 88]` | 1 |
| `ln` | `[1e-30, 1e30]` and `[0.5, 2]` | 1 |
| `atan2` | 단위원 + `[1e-6, 1e6]` 범위의 무작위 쌍 | 2 |
| `sqrt` | `[0, 1e30]` | 0 |
| `rsqrt` | `[1e-30, 1e30]` | 1 |

모두 2 ULP 목표 이내다. 이 수치는 계수와 연산 순서의 속성이지 호스트의 속성이 아니므로,
같은 `f32` 연산이 올바르게 반올림되는 곳이라면 어디서나 성립해야 한다.

참조 오라클인 MPFR(`rug`)은 GMP를 필요로 하며 Windows CI 호스트에서는 빌드할 수 없어,
대신 `f64`-std 참조를 사용한다. 이를 순수 Rust 임의정밀도 크레이트로 교체하는 것은
**Target / Status: 미검증**이다.

## Rust ↔ Slang 비트 동등성

`slang/approx.slang`은 Rust 소스를 한 줄 한 줄 그대로 옮긴 것이다. 유닛 테스트가 두
파일에서 상수 선언을 파싱해 이름/리터럴 쌍이 바이트 단위로 동일한지 검사한다. 이 둘이
실제 GPU에서 **비트 단위로 동일한 결과**를 내는지는 **Target / Status: 미검증**이다 —
§3.4의 실행 모드가 설정된 Vulkan 디바이스가 필요하며, 이는 M2 작업이다 (P09의 acceptance는
리터럴 비교만을 다룬다).
