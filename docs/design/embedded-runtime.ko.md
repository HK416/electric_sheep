<!-- Korean translation of docs/design/embedded-runtime.md. The English file is the working copy; regenerate this when it changes. -->

# 임베디드 런타임 — `no_std` 배포 경로

Spec: §9.6 (`es-runtime-embedded`: 단일 정적 바이너리, no-std 가능, 힙 할당 0; Observation
IR 평가기 + Learning IR 전·후처리 + `PolicyRuntime` + Safety Plane 전체 + 텔레메트리
링버퍼를 포함), §9.5 (시뮬/실기 동일성: Safety Plane, chunker, 전처리가 양쪽에서 *같은
코드*라는 것, 이것이 `deployment_hash` 동일성이 행위적 동일성을 의미하게 만드는 이유),
§28.5 M3 W2, 부록 B.4 (`es-safety/src/lib.rs` — `no_std` 가능, 힙 할당 0).

동반 문서: `docs/design/safety-plane.md` (`validate` 알고리즘), `docs/design/policy-bundle.md`
(`policy.esb`), `docs/design/observation-lowering.md` (§8이 참조하는 CPU 커널).

## 1. 여기서 "no-std 가능"이 의미해야 하는 것

Spec §9.5는 나머지 모든 것의 형태를 결정하는 제약이다. 임베디드 빌드가 Safety Plane의
*두 번째 구현*이라면 `deployment_hash` 동일성은 아무것도 증명하지 못한다. 그래서 이
패킷의 규칙은 다음과 같았다: **`no_std`는 정문(front door)을 제거할 수는 있어도, 코드
경로를 제거해서는 안 된다.**

구체적으로, 배포 경로 위의 모든 crate에 대해:

- 런타임 — `validate`, clamp 단계들, watchdog들, fallback들, chunk 커서, 텔레메트리
  링버퍼 — 은 두 빌드 모두에서 같은 소스로부터 컴파일되며, 그 안에 `cfg`가 없다;
- `--no-default-features`가 제거하는 것은 *authoring 쪽* 정문뿐이다: Deployment IR
  리더, `.esb` 번들 리더, `String`을 담는 에러 타입들, `dyn PolicyRuntime` trait
  object.

`SafetyPlane::from_ir`가 중복 구현이 아니라 얇은 래퍼로 만들어진 이유가 바로 그것이다:
생성 경로는 하나(`from_config`)뿐이고 `from_ir`는 그리로 흘러들어가므로, INV-12(어떤
코드 경로도 plane을 비활성화하지 않는다)는 리뷰의 약속이 아니라 구조적 속성이 된다.

## 2. crate별 분리

각 crate는 기본적으로 켜져 있는 `std` feature를 얻었다. 워크스페이스 `Cargo.toml`은
손대지 않았다 — feature는 crate 단위이며, 워크스페이스를 `cargo build`하면 일반적인
feature unification을 통해 모든 곳에서 `std`가 켜진다.

| crate | `no_std` 표면 | `std` 뒤에 있는 것 |
|---|---|---|
| `es-core` | `time::PhysTick`, `ring::ArrayRing<T, CAP>` | `ecs`, `job`, `arena`, `pool`, `failure`, `id`, `sizing`, `alloc_count`, `TickRate`, `SimTime`, `ring::RingBuffer` |
| `es-safety` | `SafetyPlane`(`from_config`, `validate` 포함), `SafetyConfig`/`Envelope`/`Watchdogs`/`Fallback`/`SensorWatch`/`WorkspaceSpec`, `ActionChunk`, `SafeAction`, `EventSet`, `SafetyCounters`, `ir_types` | `SafetyPlane::from_ir`, `Envelope/Watchdogs/Fallback::from_ir`, `SafetyConfigError` |
| `es-runtime-embedded` | `core_rt::EmbeddedCore` + `TickRecord` + `ObserveFn`/`InferFn` | `EmbeddedRuntime`, `hardware_capability`, `RuntimeError` |

의존성 결과: `es-safety`는 `es-ir`와 `thiserror`를 떨어뜨린다(둘 다 `optional`이며
`std`가 끌어들인다), 그리고 애초에 쓰지 않던 `es-math`도 완전히 떨어뜨린다.
`es-runtime-embedded`는 `es-ir`, `es-compile`, `es-policy`, `blake3`, `thiserror`를
떨어뜨린다. `serde`는 `default-features = false`와 `std = ["serde/std"]`로 취급된다.

> Cargo의 알아둘 만한 함정: crate 레벨의 `default-features = false`는
> `[workspace.dependencies]` 항목 자체가 `default-features`를 설정하지 않으면
> **무시되며**, cargo는 경고만 낸다. 그래서 `es-core`/`es-safety`/`es-runtime-embedded`는
> `workspace = true` 대신 `serde`와 형제 `es-*` crate들을 버전/경로로 직접 지정한다.
> 그 외에는 아무것도 바뀌지 않는다: feature는 가산적(additive)이므로 다른 모든
> crate는 여전히 `serde/std`를 얻는다.

## 3. `es-safety`: 힙 0에 도달하기 위해 바뀌어야 했던 것

M1 plane은 이미 틱당(per tick) 할당이 없었지만, 생성 시점에 할당했고 IR로부터 세 가지
`std` 형태를 빌려 왔다. 그중 넷이 구조적인 문제였다.

1. **핫 패스가 나르는 네 개의 IR 값 타입** — `Micros`, `Limit`, `ActionSpace`,
   `ExecutionMode`(그리고 `HalfSpace`). `src/ir_types.rs`는 `std`에서는 `es-ir`의
   것을 재수출(re-export)하고, `no_std`에서는 동일한 로컬 복사본을 정의한다.
   (변환이 아니라) 재수출하는 것이 기존 호출자들에 대해 `validate`의 시그니처를
   바이트 단위로 동일하게 유지하는 방법이며, 이는 INV-13이 요구하는 것이다;
   `tests/nostd_core.rs::ir_value_types_are_the_es_ir_types_under_std`는
   `es_ir::deployment::X`를 `es_safety::X` 바인딩에 대입함으로써 이를 고정한다.
   **ponytail ceiling:** 사소한 타입 네 개의 정의가 두 벌 존재하며, 리뷰로 동기화를
   유지한다. 업그레이드 경로: 그것들을 `es-ir-types`로 옮기고 shim을 삭제한다.
   `es-ir`가 W2의 범위 밖이라서 미룬 것이다.
2. **`Workspace::ConvexHull { faces: Vec<HalfSpace> }`** → `WorkspaceSpec::ConvexHull
   { faces: [HalfSpace; MAX_HULL_FACES], n_faces }`.
3. **`String` 이름을 가진 `Watchdogs::sensors: Vec<SensorWatch>`** → 이름이
   `[u8; SENSOR_NAME_CAP]`인 `[SensorWatch; MAX_SENSORS]` 테이블. `SensorWatch::new`는
   맞지 않는 이름을 잘라내는 대신 **거부한다**: 32바이트 프리픽스를 공유하는 두 센서가
   하나의 watchdog으로 붕괴되어 dropout 검사를 조용히 약화시켜서는 안 되기 때문이다.
4. **`Fallback::trajectory: Vec<[f64; NJ]>`** → `[[f64; NJ]; MAX_RETRACT_WAYPOINTS]` +
   길이. waypoint가 없는 `RetractToHome`은 이제 구성 자체가 불가능하며
   (`Fallback::stationary`는 그것을 안전한 `HoldPosition`으로 강등시킨다, 부재
   상태로 두지 않는다), `run_fallback`은 빈 슬라이스를 인덱싱하는 대신 정지 상태를
   유지한다.

| 상한 | 값 | 초과 시 일어나는 일 |
|---|---|---|
| `MAX_HULL_FACES` | 16 | `from_ir` → `SafetyConfigError::TooMany` |
| `MAX_RETRACT_WAYPOINTS` | 32 | `from_ir` → `SafetyConfigError::TooMany` |
| `MAX_SENSORS` | 8 | `from_ir` → `SafetyConfigError::TooMany` |
| `SENSOR_NAME_CAP` | 32 bytes | `from_ir` → `SafetyConfigError::SensorNameTooLong` |

이것들은 절단(truncation)이 아니라 거부(refusal)다: 임베디드 런타임이 표현할 수 없는
배포는 로드에 실패해야 하며, 더 조용한 envelope로 로드되어서는 안 된다(INV-12).

### 3.1 `from_config`는 infallible이며, 그것이 안전한 이유

`from_config`는 뒤에 검증기가 없는 평범한 `Copy` 데이터를 받으므로, 임베디드 호출자에게
무시할 거리를 주지 않고서는 `Result`를 반환할 수 없다 — §B.4가 `validate`에 대해 펴는
논리와 동일하다. 대신 `Envelope::sanitize`가 생성 시점에 한 번 실행되어 **좁히기만
한다**:

- 비어 있거나 finite하지 않은 hard limit는 `[0, 0]`이 된다;
- hard limit의 finite한 부분구간이 아닌 soft limit는 hard limit이 된다;
- finite하지 않거나 음수인 `vel_max`/`acc_max`/`tau_max`/`d1_max`/`d2_max`는 `0`이
  된다(가장 느슨한 값이 아니라 가장 엄격한 값);
- 양수가 아닌 `dt_s`는 1 ms가 되고, `period_us`와 `execute_chunk`는 최소 1이 된다.

모든 수리는 조인다(tighten). 여유를 만드는 §9.1의 방법은 여전히 더 넓은 *hard* limit
뿐이며, 이는 데이터로 표현된 INV-12 규칙("envelope를 넓혀라")이다.

### 3.2 `sqrt`

`f64::sqrt`는 `std`에 있다. 두 개의 제곱근을 `cfg`로 나누는 대신, `config::sqrt`는
**양쪽** 타겟에서 `libm`을 호출한다. IEEE-754 제곱근은 올바르게 반올림되므로, 이는
`std` 빌드가 방출하던 하드웨어 명령어와 비트 단위로 동일하다 — 우연히 일치하는 두
경로가 아니라 하나의 코드 경로(§9.5)다. 유일한 호출자는 실린더 workspace 투영이다.
`libm`은 이미 `Cargo.lock`에 있었고 태생적으로 `no_std`다.

## 4. `es-core`: `PhysTick`과 `ArrayRing`

`es-safety` 아래에서 필요한 것은 두 가지뿐이므로, `std` 밖에 있는 것도 두 가지뿐이다.

`TickRate`/`SimTime`은 `std` 뒤에 남았다. `TickRate::rational`이 `es_core::Error`
(`thiserror` + `String`)를 보고하기 때문이다; plane은 rate 자체가 아니라 파생된
`dt_s`/`period_us`를 저장하며 rate 자체는 결코 필요로 하지 않는다.

`ring::RingBuffer`는 `Vec`을 소유하므로, 텔레메트리 링버퍼는 재작성이 아니라 형제를
얻었다: `ArrayRing<T, CAP>`는 같은 시퀀스 번호 매기기, 덮어쓰기 순서, `dropped` 카운트를
가진 채로 `[Option<T>; CAP]`을 인라인으로 담는다. `ring::tests::array_ring_matches_ring_buffer`는
용량 3짜리 링에 대해 7번의 push를 통해 둘을 lock-step으로 구동하므로, "같은 행위"는
주장이 아니라 테스트다. `CAP`이 const generic인 이유는 마이크로컨트롤러에서는 런타임
전체가 하나의 `static`이기 때문이다.

## 5. `es-runtime-embedded`: 하나의 루프, 두 개의 프런트엔드

`core_rt::EmbeddedCore<NJ, H, CAP>`가 그 루프다:

```text
sensors -> [observe] -> obs -> [infer] -> ActionChunk -> SafetyPlane -> SafeAction
                                                             |
                                                         TickRecord -> ArrayRing<_, CAP>
```

- `EmbeddedCore::step(chunk: Option<ActionChunk>, now, obs_age) -> SafeAction`가
  본연의 루프다: (전달됐을 때만) 새 chunk를 받아들이고(§8.6에 따라 틱마다가 아니라
  *policy 호출마다* `seq`를 한 번 올린다), validate하고, 텔레메트리 레코드를 push하고,
  커서를 진행시킨다.
- `EmbeddedCore::tick(..., observe: ObserveFn, infer: InferFn<NJ, H>)`가 `no_std`
  진입점이다: 두 개의 플러그인을 구동하고 `step`을 호출한다.
- `EmbeddedRuntime::tick`이 `std` 진입점이다: `CpuPlan::run`과
  `dyn PolicyRuntime::infer`를 직접 구동하고 같은 `step`을 호출한다.

플러그인은 trait이 아니라 **함수 포인터**다:

```rust
pub type ObserveFn = fn(sensors: &[f32], obs: &mut [f32]);
pub type InferFn<const NJ: usize, const H: usize> =
    fn(obs: &[f32], actions: &mut [[f64; NJ]; H]) -> usize;
```

새 trait 없이(INV-17은 일곱 개를 나열하며 `PolicyRuntime`은 이미 그중 하나다), vtable도
없고, `Box`도 없으며, 두 버퍼 모두 호출자 소유이므로 step은 할당할 수 없다. `InferFn`은
채운 행의 개수를 반환한다: `0`은 ONNX/Vulkan/NPU 실패가 호출자가 버릴 수 있는 에러가
아니라 plane의 `ChunkUnderrun`과 설정된 fallback이 되는 방법이다(INV-13). ONNX / Vulkan
/ NPU `PolicyRuntime` 구현들은 `std` 레이어에서 끼워 넣어진다 — 임베디드 타겟에서는
통합자가 두 개의 `fn`을 공급하며, 이것이 (Rust trait 없이 C ABI만 있는) 벤더 NPU SDK가
들어오는 방법이기도 하다.

`EmbeddedRuntime`은 이제 자신만의 plane, chunk, 커서, seq, 링버퍼를 갖는 대신
`EmbeddedCore<NJ, H, 1024>`를 *소유한다*. 그래서 `std` 경로와 임베디드 경로가 서로
어긋날 수 없다. `EmbeddedCore`는 의도적으로 `Clone`이 아니다: 복제된 제어 루프는
plane의 상태와 카운터를 분기시킬 것이다.

## 6. 증명

```
cargo build -p es-safety -p es-runtime-embedded --no-default-features \
    --target thumbv7em-none-eabihf          # Cortex-M4F/M7, hard float, no std
cargo build -p es-core   --no-default-features --target thumbv7em-none-eabihf
cargo clippy -p es-safety -p es-runtime-embedded --no-default-features \
    --target thumbv7em-none-eabihf -- -D warnings
```

모두 통과한다. bare-metal 타겟에는 테스트 하네스가 없으므로, 행위는 `no_std` 코드만
운동시키는 `std` 테스트로부터 고정된다:

- `crates/es-safety/tests/nostd_core.rs` — `from_config`는 plane이 그래야 하듯이
  clamp한다; 퇴화된 설정은 존중되지 않고 좁혀진다(INV-12); `from_config`에 **더해**
  32번의 `validate` 호출이 `assert_no_alloc` 안에서 실행된다(spec §9.6의 힙 0 주장이
  이제 핫 패스뿐 아니라 생성까지 포괄한다); 센서 테이블은 유계이며 결코 절단하지
  않는다; IR 값 타입들은 `std`에서는 `es-ir`의 것이다.
- `crates/es-runtime-embedded/tests/core_loop.rs` — replan 스케줄과 텔레메트리 기록;
  `0`을 반환하는 `InferFn`은 fallback이 된다; 링버퍼는 유계다; 전체 루프(생성 + 64틱)가
  `assert_no_alloc` 안에서 실행된다.
- 기존의 `es-safety` property/scenario suite들과 `es-runtime-embedded` 번들
  테스트들은 변경되지 않았고 여전히 green이다. 이것이 `std` 경로가 움직이지 않았음을
  말해준다.

### 크기

*링크된* 이미지의 `.text`는 아직 측정할 수 없다: bare-metal 바이너리는 진입점, 링커
스크립트, 패닉 핸들러가 필요한데, 이 패킷은 그중 아무것도 배포하지 않는다(여기에는
`[[bin]]`이 없다). `llvm-size`/`cargo size`도 이 환경에 설치되어 있지 않다. 대신
`thumbv7em-none-eabihf`용 릴리스 `rlib` 크기를 프록시로만 기록한다 — 이것들은 코드뿐
아니라 메타데이터와 bitcode도 포함한다:

| 산출물 (release, thumbv7em-none-eabihf) | rlib bytes | `.text` |
|---|---|---|
| `libes_safety.rlib` | 441,578 | 미검증 (unverified) |
| `libes_runtime_embedded.rlib` | 35,872 | 미검증 (unverified) |

**Target / Status: 미검증 (unverified)** (spec §12.4). 링커 스크립트와 `cortex-m-rt`가
있는 예제 `[[bin]]`을 추가하는 후속 패킷이라면 `.text`/`.rodata`/`.bss`를 제대로 보고할
수 있다; `.bss` 숫자가 흥미로운데, `EmbeddedCore<NJ, H, CAP>`가 텔레메트리 링버퍼
전체(`CAP * size_of::<Option<TickRecord>>()`)와 `SafetyConfig` 전체를 static 메모리에
두기 때문이다.

## 7. 이 패킷이 **하지 않은** 것

- **NPU `PolicyRuntime` 백엔드**(§28.5 W2 행의 세 번째 항목). 여기서는 범위 밖이다;
  위의 `InferFn` 슬롯이 그것이 쓰게 될 이음매다.
- **링크된 bare-metal 이미지**, 따라서 실제 크기나 스택 깊이 숫자 없음(위 참조).
- **`es-ir` / `es-math`의 `no_std`화**, 이것이 §8을 막고 있는 것이다.

## 8. 다음 패킷: `no_std` Observation IR 평가기(`StaticPlan`)

§28.5 W2 행은 no-std Observation IR 평가기도 요구한다. `es-compile/src/kernels.rs`는
필요한 방식으로 *이미* 작성되어 있다 — 모든 함수는 `&[u8]`/`&[f32]` 슬라이스에 대해
순수하고, 단일 스레드이며, 고정된 루프와 연산자 순서를 가지고, 유일한 초월함수는
`es_math::approx`를 거친다(spec §3.4 `DET-010`); 파일 안의 유일한 `Vec`는
`#[cfg(test)]` 안에 있다. 그래서 손대지 않고 남겨두었다: 이것을 게이팅하는 것은
기계적인 일이지만, W2의 범위 안에서는 할 수 없다.

**막힌 것들, 풀려야 하는 순서대로:**

1. `es-math`(layer 0)는 `no_std`가 아니다: `simd.rs`가
   `std::is_x86_feature_detected!`를 쓴다. `kernels.rs`는 sRGB 파워 커브를 위해
   `es_math::approx`가 필요하므로, `approx`는 런타임 feature 디스패치를 `std` 뒤로
   두면서 `no_std`가 되어야 한다.
2. `es-ir`(layer 6)는 `no_std`가 아니며, `kernels.rs`는 `es_ir::Rect`를,
   `exec.rs`는 `ElemType`과 `ResizeFilter`를 필요로 한다. 이것들은 평범한 값
   타입이며, 제자리는 `es-ir-types`다. 이렇게 하면 `es-safety`도 `src/ir_types.rs`를
   삭제할 수 있게 된다(위 §3.1).
3. `es-compile` 자체의 모듈 집합: `bundle.rs`(`.esb` 리더, `blake3`, `std::io`)와
   `plan.rs`의 `CpuPlan::compile`(`BTreeMap`, `Vec<Step>`, `String`을 쓰는 진단)은
   `std` 뒤로 가야 하며, 이는 `es-compile/src/lib.rs`와 `bundle.rs`를 편집해야
   한다는 뜻이다 — 작성 시점 기준으로 둘 다 다른 진행 중인 패킷 소유다. 쓰이지 않는
   `es-assets` 의존성도 같은 시점에 제거되어야 한다.

**다음 패킷이 구현해야 할 설계 — `StaticPlan`.** `CpuPlan`은 컴파일타임 객체다:
`compile`은 shape를 해석하고 버퍼 id를 할당하고 `Vec<Step>`을 방출하며, 그 다음
`run`은 매 호출마다 새 arena와 빌린 입력의 `BTreeMap`으로 그것을 해석한다. 이미
존재하는 그 경계선을 따라 둘로 나누면:

```rust
// no_std: 모든 것이 미리 해석되어 있고, 아무것도 소유하지 않는다.
pub struct StaticPlan<'a> {
    steps: &'a [Step],            // Op + BufferId 피연산자, 이미 타입/shape 검사됨
    buffers: &'a [BufferDesc],    // dtype, 원소 개수, 하나의 arena 안 바이트 오프셋
    inputs: &'a [(BufferId, u32)],  // plan 입력 슬롯 -> 버퍼, 고정된 순서
    outputs: &'a [(BufferId, u32)],
    arena_bytes: usize,           // 호출자의 단일 scratch 할당
}

impl StaticPlan<'_> {
    /// `arena.len() >= self.arena_bytes`; 입력은 호출자에 의해 자신의 슬롯에 쓰인다.
    pub fn run(&self, arena: &mut [u8]) -> Result<(), StaticPlanError>;
}
```

이것을 두 번째 평가기를 새로 쓰는 대신 이렇게 하는 것이 가치 있게 만드는 세 가지
속성:

- **하나의 arena, 하나의 호출자 할당.** 모든 버퍼는 컴파일타임에 결정된 `arena` 안의
  `(offset, len)`이므로, `run`은 할당도 조회도 하지 않는다 — §9.6의 힙 0 주장이
  observation 파이프라인까지 확장된다.
- **이름이 아니라 인덱스로 된 입력.** `CpuPlan::run`은 입력을 `String`으로 키
  삼는다; static 형태는 plan 자체의 순서로 슬롯 번호를 매기며, 이는 틱에서 마지막
  남은 `BTreeMap`도 제거한다.
- **같은 `Step`들.** (여전히 `std`이고 여전히 번들 로더 안에 있는) `CpuPlan::compile`이
  `StaticPlan`이 빌리는 `&[Step]`/`&[BufferDesc]`를 *방출*한다. 그래서 `policy.esb`는
  플래시 이미지를 위한 `static` 테이블로 미리 트랜스코딩될 수 있고, 로봇에서 생성되는
  바이트가 골든이 고정하는 바이트가 된다 — observation 경로에 대한 §9.5 동일성 주장.

그 패킷의 오라클: 기존의 `tests/golden/observation/` 텐서를, `CpuPlan::run`과
`StaticPlan::run` 양쪽으로 실행해서 비트 단위로 비교한다.
