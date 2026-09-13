<!-- Korean translation of docs/packets/M3/W2-embedded-nostd.md. The English file is the working copy; regenerate this when it changes. -->

# W2-embedded-nostd — `no_std` Safety Plane과 `no_std` 임베디드 제어 루프

Spec: §9.6 (`es-runtime-embedded`: 단일 정적 바이너리, no-std 가능, 힙 할당 0;
Observation IR 평가기 + Learning IR 전·후처리 + `PolicyRuntime` + Safety Plane 전체 +
텔레메트리 링버퍼), §9.5 (시뮬/실기 동일성 — Safety Plane이 양쪽에서 *같은 코드*라는
것, 이것이 `deployment_hash` 동일성이 행위적 동일성을 의미하게 만드는 이유), §28.5 M3
W2 (no-std Observation IR 평가기, Safety Plane, NPU 백엔드 — NPU 백엔드는 별도
패킷), 부록 B.4 (`SafetyPlane` — no_std 가능, 힙 할당 0), §3.4 (결정성), §12.4
(미검증 성능 보고).

설계 노트: `docs/design/embedded-runtime.md`(먼저 읽을 것 — §2에 crate별 feature
표, §3에 고정 크기 상한과 `from_config` sanitizer 논거, §8에 이 패킷이 의도적으로
미루는 `StaticPlan` 설계가 있다).

## context (범위)

```
crates/es-core/Cargo.toml                       (+ `std` / `alloc-count` features)
crates/es-core/src/lib.rs                       (+ cfg_attr no_std, module gating)
crates/es-core/src/time.rs                      (PhysTick stays; TickRate/SimTime gated)
crates/es-core/src/ring.rs                      (+ ArrayRing; RingBuffer gated)
crates/es-safety/Cargo.toml                     (+ `std` feature; libm; drops unused es-math)
crates/es-safety/src/lib.rs                     (+ cfg_attr no_std, exports)
crates/es-safety/src/ir_types.rs                (new)
crates/es-safety/src/config.rs                  (fixed-size config; from_ir behind `std`)
crates/es-safety/src/plane.rs                   (+ from_config; from_ir delegates)
crates/es-safety/src/types.rs                   (ExecutionMode via ir_types)
crates/es-safety/tests/nostd_core.rs            (new)
crates/es-runtime-embedded/Cargo.toml           (+ `std` feature)
crates/es-runtime-embedded/src/lib.rs           (+ cfg_attr no_std, layer split)
crates/es-runtime-embedded/src/core_rt.rs       (new: EmbeddedCore, TickRecord, Observe/InferFn)
crates/es-runtime-embedded/src/runtime.rs       (EmbeddedRuntime built on EmbeddedCore)
crates/es-runtime-embedded/tests/core_loop.rs   (new)
docs/design/embedded-runtime.md                 (new)
docs/packets/M3/W2-embedded-nostd.md            (this file)
```

## spec (사양)

1. `es-core`는 기본적으로 켜져 있는 `std` feature를 얻는다. 그것이 없으면 crate는
   `no_std`이며 정확히 `time::PhysTick`과 새로운 `ring::ArrayRing<T, CAP>`
   (`[Option<T>; CAP]`을 인라인으로, `alloc` 없이)만 노출한다. `ring::RingBuffer`와
   같은 시퀀스 번호 매기기 / 덮어쓰기 순서 / `dropped` 의미론을 갖는다. `TickRate`,
   `SimTime`, 그 밖의 모든 것은 `std` 뒤에 남는다; `alloc-count` feature는 `std`를
   내포한다.
2. `es-safety`는 기본적으로 켜져 있는 `std` feature를 얻는다. 그것이 없어도 런타임
   전체는 여전히 컴파일된다: `SafetyPlane`, `validate`,
   `SafetyConfig`/`Envelope`/`Watchdogs`/`Fallback`/`SensorWatch`/`WorkspaceSpec`,
   `ActionChunk`, `SafeAction`, `EventSet`, `SafetyCounters`. Deployment IR
   정문(`from_ir`, `SafetyConfigError`)만 게이팅되는데, `es-ir`가 `std` crate이기
   때문이다.
3. `SafetyPlane::from_config(&SafetyConfig<NJ>) -> Self`가 `no_std` 생성자다: 평범한
   `Copy` 데이터, 힙 없음, infallible. `SafetyPlane::from_ir`는 `SafetyConfig`를
   만들고 그것에 위임하므로, 생성 경로는 정확히 하나뿐이다(INV-12). `validate`의
   시그니처는 변경되지 않는다(INV-13) — `src/ir_types.rs`는 `std`에서는 `es-ir`의
   `Micros`/`Limit`/`ActionSpace`/`ExecutionMode`/`HalfSpace`를 재수출하고, `no_std`
   에서는 동일한 로컬 복사본을 정의한다.
4. `Envelope::sanitize`는 `from_config` 안에서 한 번 실행되며 퇴화된 설정을 오직
   **좁히기만** 한다(비어 있거나 finite하지 않은 hard limit → `[0,0]`; hard limit
   바깥의 soft limit → hard limit; finite하지 않거나 음수인 rate/velocity/
   acceleration/torque bound → `0`; 양수가 아닌 `dt_s` → 1 ms). 어떤 수리도
   무엇도 넓히지 않는다(INV-12).
5. `Vec`/`String` 필드는 고정 상한을 가진 고정 크기 테이블이 된다: `MAX_HULL_FACES =
   16`, `MAX_RETRACT_WAYPOINTS = 32`, `MAX_SENSORS = 8`, `SENSOR_NAME_CAP = 32`.
   하나를 초과하는 것은 `from_ir` 에러(`TooMany`, `SensorNameTooLong`)이며, 결코
   절단이 아니다 — 센서 이름은 짧게 잘리는 대신 거부되어, 두 센서가 하나의 dropout
   watchdog으로 붕괴할 수 없게 한다.
6. `es-runtime-embedded`는 기본적으로 켜져 있는 `std` feature를 얻는다. 그것이
   없으면 `core_rt::EmbeddedCore<NJ, H, CAP>`만 컴파일된다: Safety Plane + chunk
   커서 + seq + `ArrayRing<TickRecord, CAP>`. observation과 inference 단계는
   trait이 아니라 함수 포인터 타입 별칭(`ObserveFn`, `InferFn<NJ, H>`)이다 —
   INV-17의 일곱 확장 지점은 그대로 유지된다 — 버퍼는 호출자가 소유하며, `InferFn`은
   채운 action 행의 개수를 반환하므로 `0`은 plane의 `ChunkUnderrun`과 설정된
   fallback이 된다(INV-13).
7. `EmbeddedRuntime`(`std` 레이어)은 `EmbeddedCore<NJ, H, 1024>`를 소유하고
   `EmbeddedCore::step`을 호출하므로, `std` 경로와 임베디드 경로는 서로 어긋날 수
   없다(§9.5). 공개 행위는 `telemetry()`를 제외하면 변경되지 않으며,
   `telemetry()`는 이제 `&ArrayRing<TickRecord, TELEMETRY_TICKS>`를 반환한다.
8. `config::sqrt`는 **양쪽** 타겟에서 `libm::sqrt`다(`f64::sqrt`는 `std` 전용이다;
   IEEE-754 제곱근은 올바르게 반올림되므로, 이는 하드웨어 명령어와 비트 단위로
   동일하며 두 개가 아니라 하나의 코드 경로를 유지한다, §9.5, §3.5 tier 1).
9. 루트 `Cargo.toml`은 손대지 않으며 어떤 crate도 추가되지 않는다. `es-safety`는
   애초에 쓰지 않던 `es-math` 의존성을 떨어뜨린다.
10. **후속 패킷으로 미뤄짐, 설계는 노트의 §8에:** `no_std` Observation IR
    평가기(`StaticPlan`). `es-compile/src/kernels.rs`는 이미 슬라이스에 대해
    순수하며 의도적으로 손대지 않은 채 남겨졌다; 이것을 게이팅하려면 먼저
    `es-math`와 `es-ir`가 `no_std`가 되어야 하고, `es-compile/src/lib.rs`와
    `bundle.rs`에 대한 편집이 필요한데, 이는 W2가 소유하지 않는다. §28.5 W2 행의
    NPU `PolicyRuntime` 백엔드도 마찬가지로 별도 패킷이다; `InferFn`이 그것이 쓸
    이음매다.

## oracle (오라클)

```
cargo fmt --check
cargo clippy -p es-core -p es-safety -p es-runtime-embedded -p es-compile --all-targets -- -D warnings
cargo test -p es-core -p es-safety -p es-runtime-embedded -p es-compile
cargo build -p es-core -p es-safety -p es-runtime-embedded --no-default-features --target thumbv7em-none-eabihf
cargo clippy -p es-safety -p es-runtime-embedded --no-default-features --target thumbv7em-none-eabihf -- -D warnings
cargo xtask layering
cargo xtask context-budget
cargo check --workspace --all-targets
```

`thumbv7em-none-eabihf`(Cortex-M4F/M7, hard float, `std` 없음)가 증명 타겟이다;
그 타겟이 설치되어 있지 않으면 `x86_64-unknown-none`으로 대체한다. `--target` 없는
`cargo build --no-default-features`는 feature가 해석된다는 것만 증명할 뿐, `std`가
부재한다는 것을 증명하지 않는다.

## acceptance (수용 기준)

1. 위의 모든 명령이 0으로 종료한다.
2. `crates/es-safety/tests/nostd_core.rs`가 통과한다: `from_config`가 clamp하는
   plane을 만든다; 퇴화된 설정은 존중되지 않고 좁혀진다; `from_config`에 **더해**
   32번의 `validate` 호출이 `es_core::alloc_count::assert_no_alloc` 안에서
   실행된다; 센서 테이블은 유계이며 너무 긴 이름을 거부한다; IR 값 타입은 `std`
   에서는 `es-ir`의 것이다.
3. `crates/es-runtime-embedded/tests/core_loop.rs`가 통과한다: replan 스케줄과
   텔레메트리 기록이 올바르다; `0`을 반환하는 `InferFn`은 `ChunkUnderrun`과 함께
   fallback을 낳는다; 링버퍼는 `CAP`에서 유계이고 `dropped`를 센다; 생성 + 64틱이
   아무것도 할당하지 않는다.
4. `es_core::ring::tests::array_ring_matches_ring_buffer`가 `ArrayRing`과
   `RingBuffer`를 lock-step으로 구동하며 `len`/`dropped`/`oldest_seq`/
   `iter_newest`/`drain_since`에서 서로 일치한다.
5. 기존의 `es-safety` property/scenario suite들과 `es-runtime-embedded` 번들
   테스트들이 수정 없이 통과한다 — `std` 경로가 움직이지 않았다는 뜻이다.
6. 새 trait 없음(INV-17), `no_std` 경로에 `alloc` 없음, `HashMap` 없음, 영어만.
7. `docs/design/embedded-runtime.md`가 crate별 feature 표, 상한 표, 이유와 함께
   `unverified`로 표시된 크기 표, 그리고 `StaticPlan` 설계와 그 순서 있는
   blocker들을 싣고 있다.

## forbidden (금지)

- `crates/es-eval/**`, `crates/es-compile/src/bundle.rs`, `crates/es-data/**`,
  `crates/es-splat/**`, `crates/es-editor/**`, `crates/es/**` — 다른 진행 중인
  패킷들의 것.
- `crates/es-compile/src/{lib,kernels,exec,plan}.rs` — `StaticPlan` 패킷의 것,
  그것의 blocker들이 풀리고 나면(노트 §8).
- `crates/es-ir/**`, `crates/es-ir-types/**`, `crates/es-math/**` — 그것들의
  `no_std`화는 blocker 목록이지, 이 패킷의 일이 아니다.
- 루트 `Cargo.toml`, 그리고 어떤 새 crate도.
- `SafetyPlane::validate`의 시그니처나 반환 타입을 바꾸는 것(INV-13); 두 번째
  plane 생성자, `enabled` 플래그, `#[cfg(test)]` 우회를 추가하는 것(INV-12);
  `es-safety`가 `es-policy`에 의존하게 만드는 것(INV-11).
- `tests/golden/` 아래의 무엇이든 편집하는 것.
