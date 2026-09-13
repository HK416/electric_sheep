<!-- Korean translation of docs/packets/M2/W3-diffusion-flow-lowering.md. The English file is the working copy; regenerate this when it changes. -->

# W3 — Diffusion 및 FlowMatching lowering, tier 4 (`es-policy`)

명세: spec 8.3 (노드 집합: `DiffusionHead`, `FlowMatchingHead`), spec 8.5 (액션 모델링),
spec 8.7 (lowering: 헤드는 `PolicyHandle` 부분이다), spec 8.9 (tier 4 허용오차), spec 3.4
(결정성, 전역 RNG 금지), spec 5.3 (해시 체인), spec 28.4 W3, spec 28.7 gate 12. 설계 노트:
`docs/design/learning-lowering.md` 8절 (리뷰 유형 C — 노트가 산출물이며, 코드는 그 하위
결과물이다).

spec 28.4 W3 행에서 오프라인으로 검증 가능한 부분이다. SmolVLA(`PolicyBundle`)와 ONNX
런타임은 이 패킷에 **포함되지 않는다**: ONNX는 자신의 `ort` 의존성과 자신의 오라클을 가진
형제 lowering 대상이며, 여기서 비교하는 두 정책은 torch 대 ONNX가 아니라 torch 대 Rust
참조 구현이다. gate 12("세 정책이 tier 4를 통과한다")의 세 개 중 둘은 여기서 나오고, 나머지
하나는 ONNX 패킷에서 나온다.

## context (범위)

```
crates/es-policy/src/lower/torch.rs                 (extended: two head arms + the sampler runtime)
crates/es-policy/src/lib.rs                         (extended: `#[cfg(test)] mod reference`)
crates/es-policy/src/reference.rs                   (new, test-only)
docs/design/learning-lowering.md                    (extended: section 8)
docs/packets/M2/W3-diffusion-flow-lowering.md       (new)
```

## spec (사양)

- `PolicyHead { Diffusion { n_steps, scheduler } }`는 `_DdpmHead`로 lowering된다: 선형 beta
  스케줄(`1e-4 .. 0.02`) 위에서 `t = n_steps-1 .. 0`에 대해 역방향 루프
  `x = c1[t] * x + c3[t] * eps_theta(x, cond, t) (+ sigma[t] * z_t)`를 수행한다. `Ddpm`과
  `Ddim`(eta = 0)은 이 루프를 공유하며 세 계수 리스트에서만 차이가 나는데, 이 리스트들은
  Rust f32로 `diffusion_schedule`이 계산해 Python 리터럴로 내보낸다. `DpmSolver`는
  `LowerError::Unsupported`다.
- `PolicyHead { FlowMatching { n_steps } }`는 `_FlowHead`로 lowering된다: `t = 0`의
  노이즈에서 `t = 1`의 액션까지 `dx/dt = v_theta(x, cond, t)`를 오일러 적분하며,
  `dt = 1/n_steps`다.
- 두 헤드 모두 `_Denoiser`, `l1(relu(l0([x_t, cond, temb(t)])))`를 공유하며, hidden 너비와
  sinusoidal 임베딩 너비는 conditioning 너비와 같다(spec 8.3은 둘 다 싣고 있지 않으므로,
  `nhead = 8`과 마찬가지로 lowering 상의 선택이다). `cond`는 짝수여야 한다.
- **결정성 (spec 3.4).** 초기값 `x_T`는 *선언된 그래프 입력*(`noise`)이므로, 샘플러 헤드는
  입력 포트 두 개를 선언하고 `infer`는 여전히 이름 붙은 입력들의 순수 함수로 남는다; 입력이
  하나뿐인 샘플러 헤드는 `LowerError::Shape`다. 스텝별 `z_t`는 학습되지 않는 체크포인트 버퍼
  `nodes.<k>.noise_<t>`이며, `validate_keys`가 요구하는 exact 키다. DDIM은 아무것도 선언하지
  않는다. 시드가 걸린 생성기도, `torch.Generator`도, 전역 RNG도 없다.
- `reference.rs`는 `#[cfg(test)]`이며, 고정된 연산 순서와 `exp`/`sin`/`cos`/`sqrt`에 대한
  `es_math::approx`를 사용해 생성된 모듈을 f32로 미러링한다. lowering과 `diffusion_schedule`
  및 노이즈 버퍼를 공유하며, 나머지 전부도 미러링한다.
- `ort`/ONNX 없음, 새 trait 없음(`INV-17` 그대로), `HashMap` 없음, `BTreeMap`만 사용,
  `es-ir`, `es-safety`, 루트 매니페스트에 대한 변경 없음.

## oracle (오라클)

```
cargo fmt -p es-policy --check
cargo clippy -p es-policy --all-targets -- -D warnings
cargo test -p es-policy
ES_PYTHON=<venv>/Scripts/python.exe cargo test -p es-policy
cargo xtask check-spec-refs
```

tier-4 테스트는 `torch`가 설치된 인터프리터를 찾지 못하면 이유를 출력하며 SKIP하고,
`ES_PYTHON`이 그런 인터프리터를 가리키면 RUN한다 — spec 1.4는 오라클이 특정 머신에 설치되어
있든 아니든 하네스 자체는 존재하기를 원하며, wheel이 없는 상태가 결코 동등성 통과로 읽혀서는
안 된다.

## acceptance (수용 기준)

- 두 헤드 모두 lowering이 실행마다 바이트 단위로 동일하다; `n_steps`와 스케줄러 모두
  `lowering_hash`를 바꾼다.
- 생성된 소스는 샘플링 루프와 선언된 `n_steps`를 문자 그대로 담고 있다.
- DDPM은 `noise_<t>` 키를 `n_steps - 1`개 선언한다(`t = 0`에서는 없음); DDIM과
  FlowMatching은 아무것도 선언하지 않는다; DDPM 체크포인트는 FlowMatching 모듈에 대해
  `validate_keys`를 통과하지 못하며, `noise_3`이 없으면 `missing`으로 보고된다.
- torch 대 Rust 참조 구현, state 4 / cond 16 / action 2 / horizon 3 / 8 steps, torch
  2.14.0+cpu, spec 8.9의 `abs <= 1e-5` 기준:

  | 헤드 | `max_abs` | `max_rel` |
  |---|---|---|
  | `Diffusion { Ddpm }` | 5.96e-8 | 2.60e-6 |
  | `Diffusion { Ddim }` | 3.73e-8 | 4.32e-7 |
  | `FlowMatching` | 1.04e-7 | 1.37e-6 |

- 동일한 정책을 재실행하면 비트 단위로 동일하다(spec 8.9의 마지막 행).

## forbidden (금지)

- `crates/es-env`, `crates/es-eval`, `crates/es-compile`, `crates/es-data`, `crates/es`,
  그리고 그 외의 모든 크레이트: 인접 패킷들의 소관이다. 루트 `Cargo.toml`도 마찬가지다.
- ONNX / `ort`(별도 패킷), `PolicyBundle` / SmolVLA / π₀(spec 8.3이 이들을 통째로
  참조한다고 명시), cross-attention과 FiLM conditioning, UNet denoiser, `DpmSolver`,
  배치화된 lowering.
- `crates/es-ir` 수정: 샘플러 계약은 현재의 노드 집합으로 표현 가능하다.
- `Tolerance::TIER4_FP32`를 완화하는 것, 또는 샘플러가 자기 자신의 난수를 뽑게 만드는 것.
