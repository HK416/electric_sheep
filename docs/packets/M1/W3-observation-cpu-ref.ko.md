<!-- Korean translation of docs/packets/M1/W3-observation-cpu-ref.md. The English file is the working copy; regenerate this when it changes. -->

# W3 — Observation IR → CPU 레퍼런스 실행 계획 (part 1)

Spec: spec 7.2, spec 7.3, spec 7.5, spec 7.6, spec 7.7, spec 11.1, spec 11.3,
spec 11.5, spec 3.1, spec 3.4. 설계 노트: `docs/design/observation-lowering.md`
(리뷰 등급 C — 수치가 곧 산출물이며, 코드는 그것에 종속된다).

## context (범위)

```
docs/design/observation-lowering.md      (new, written first)
docs/packets/M1/W3-observation-cpu-ref.md (new)
crates/es-compile/Cargo.toml             (adds `half`)
crates/es-compile/src/lib.rs
crates/es-compile/src/plan.rs            (new)
crates/es-compile/src/kernels.rs         (new)
crates/es-compile/src/exec.rs            (new)
crates/es-compile/tests/gen_goldens.rs   (new, #[ignore], run once)
crates/es-compile/tests/observation_cpu.rs (new)
tests/golden/observation/**              (new, CI read-only once committed)
```

## spec (사양)

- `plan::CpuPlan::compile(&ObservationIr, PlanMode) -> Result<CpuPlan, Vec<Diagnostic>>` —
  IR을 통과시켜 검증하고, `ImageSpec`들을 전파하고, 위상 정렬하고, 각
  노드의 버퍼를 그 **선언된** `PortType`으로부터 해석하고, 하나의 평평한
  arena를 한 번에 배치한다. `PlanMode::{Debug, Release}`는 기록되고
  해시되며, 둘 다 모든 중간값을 주소 지정 가능하게 유지하고 둘 다
  aliasing하지 않는다 (근거: 설계 노트 §10).
- `kernels` — `resize_bilinear`, `resize_nearest`, `crop`, `srgb_to_linear`
  (+ LUT-256 u8 경로), `normalize`, `stack`, `concat`, `history_push`,
  `window_gather`, `cast_u8_hwc_to_f32_chw`, `cast_f32_to_f16`,
  `cast_f32_to_bf16`. 슬라이스에 대한 순수 함수, 단일 스레드, 고정된 루프
  순서, 유일한 초월함수(sRGB EOTF의 `exp(2.4 * ln t)`)에는
  `es_math::approx`를 사용. 커널 안에서 `std`의 `powf`/`exp`는 금지
  (spec 3.2, `DET-010`).
- `exec::{Tensor, TensorRef, Outputs, ExecError}`와
  `CpuPlan::run(&mut self, &BTreeMap<String, TensorRef>) -> Result<Outputs, ExecError>`.
  `Tensor`는 여기서 정의된다: `es-ir`에는 런타임 텐서 타입이 없으며 새로
  하나를 만들어서도 안 된다.
- `CpuPlan::compiler_hash()` — (크레이트 버전, 계획 모드, 표 순서대로의
  커널 id들)에 대한 blake3이며, `execution_hash`의 `compiler` 슬롯을
  위한 것이다 (spec 5.3, spec 11.2).
- 레이아웃: sensor 경계에서는 u8 HWC, 그 이후로는 f32 CHW; `Dequantize`가
  레이아웃을 바꾸는 유일한 노드다 (설계 노트 §2).
- part-1 서브셋 밖의 노드 종류는 조용한 no-op이 아니라 `COMPILE-002`를
  낸다. `COMPILE-0xx` 코드들은 여기서는 `TODO(codes-merge)`가 붙은 로컬
  상수이며, 이는 다섯 개의 IR 모듈이 P27에서 `es_ir::codes`로 병합되기
  전에 자신의 코드를 갖고 있던 방식을 따른 것이다.

## oracle (오라클)

```
cargo test -p es-compile && cargo xtask verify-goldens
```

## acceptance (수용 기준)

- 다음에 대해 `tests/golden/observation/*.bin`과 바이트 단위로 일치함:
  8×6 RGB u8 그래디언트를 CHW f32로 dequantize한 것; 그 이미지를 4×3으로
  bilinear resize한 것; 그것의 crop; 256개 항목의 sRGB→linear LUT; 채널별
  normalize; 2-프레임 history window. 각 `.bin`에는 shape, dtype, kernel,
  그것이 무엇을 고정하는지를 명명하는 `.json` 사이드카가 있다.
- Proptest: 상수 이미지의 resize는 어디서나 그 상수다(clamp로 인한 가장자리
  누출 없음); `crop(crop(x, a), b) == crop(x, a ∘ b)`가 픽셀 단위로
  성립하며, **동시에** 연쇄된 `ImageSpec::cropped` intrinsics가 합쳐진
  사각형의 것과 같다 (`INV-14`).
- `es_ir::testing::arbitrary_observation_ir`로부터 계획이 컴파일되고
  실행된다.
- `compiler_hash`는 `PlanMode::Debug`와 `PlanMode::Release` 사이에서
  다르다.
- 게이트: `cargo fmt -p es-compile --check`, `cargo clippy -p es-compile
  --all-targets -- -D warnings`, `cargo test -p es-compile`,
  `cargo xtask layering`, `cargo xtask verify-goldens`,
  `cargo xtask context-budget`.

## forbidden (금지)

- `context` 밖의 어떤 파일이든. 특히 `crates/es-safety`, `crates/es-env`,
  `crates/es-data` (다른 패킷들이 소유함), `crates/es-ir` (그 API는
  소비될 뿐 절대 편집되지 않음), 그리고 루트 `Cargo.toml`.
- 테스트를 통과시키기 위해 golden 파일을 편집하는 것 (spec 1.4).
  `GOLDEN_UPDATE=1`은 해결책이 아니다.
- `image`, `ndarray`, 또는 그 외 어떤 배열/이미지 의존성이든: 커널
  자체가 산출물이다.
- `HashMap`/`HashSet` (spec 3.4), 스레드, `rayon`, 커널 안의
  `std::f32::powf`/`exp`/`ln`.
- 새 trait — `INV-17`의 일곱 확장 지점이 전부이며, 그중 어느 것도 커널이나
  계획이 아니다.
- 계획 어디에서든 `Intrinsics` 산술을 건드리는 것: 기하학은
  `ImageSpec::resized` / `ImageSpec::cropped`를 거친다 (`INV-14`).
- GPU lowering, fusion, 버퍼 aliasing, `Augment` RNG — 이후 웨이브의 몫이다.
