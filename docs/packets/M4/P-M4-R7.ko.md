<!-- Korean translation of docs/packets/M4/P-M4-R7.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R7 — 너비와 entry point를 인식하는 SPIR-V 패칭, 검증됨

Spec: §3.3(모드가 요청되는 정밀도는 f32다), §3.4 step 3(execution mode와
`NoContraction`은 컴파일러가 SPIR-V를 통해 적용하며, 결코 디바이스 설정으로 적용하지
않는다), §3.5 tier 1.
리뷰 발견 사항: `docs/reviews/M4.md` S-5 — `es-gpu/src/spirv.rs:146,257`의 "이미
존재하는가?" 가드는 **모드 워드**만 비교하고 너비 오퍼랜드와 entry point는 무시했다.
그래서 `DenormFlushToZero 16`이 요청된 `32`를 조용히 억제했고, 이미
`DenormPreserve 32`를 지닌 모듈은 잘못된 SPIR-V로 패치되곤 했다; `:193`은 **첫 번째**
entry point만 패치했다(`filter`가 아니라 `find`). S-6 — SPIR-V를 직접 작성하는 유일한
크레이트에서 `spirv-val`을 실행하는 곳이 없었다; `pipeline.rs:79`는 헤더만 파싱해놓고
"검증됨(validated)"이라고 말하고 있었다.

## context (범위)

```
crates/es-gpu/src/spirv.rs
crates/es-gpu/src/slang.rs
crates/es-gpu/src/lib.rs
crates/es-gpu/tests/determinism.rs
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R7.md
```

## spec (사양)

- `apply_exec_modes(words, modes) -> Result<Vec<u32>, GpuError>` (이전에는
  `Option<Vec<u32>>`였으며, *왜* 실패했는지 말할 여지가 없었다).
- 첫 번째가 아니라 **모든** `OpEntryPoint` id를 수집하며, 요청된 각 모드를 그 각각에
  추가한다.
- `declared_modes(words, insts) -> Vec<(entry, mode, width)>`는 `OpExecutionMode`의
  리터럴 오퍼랜드 하나짜리 형태(`i.len == 4`)를 읽으며, 이는 정확히
  `SPV_KHR_float_controls`의 형태다; `LocalSize`류는 워드 수가 다르므로 결코 이것으로
  오인될 수 없다. "이미 존재하는가?" 검사는 `declared.contains(&(entry, mode, 32))`다
  — entry point별, 너비별로.
- `conflicting_mode(mode)`: `DenormFlushToZero ↔ DenormPreserve`,
  `RoundingModeRTE ↔ RoundingModeRTZ`. 어떤 entry point든 패치 중인 너비에서 이미
  상충하는 모드를 선언하고 있다면, `apply_exec_modes`는 **아무것도 내보내기 전에**
  `Err(GpuError::Spirv(..))`를 반환한다 — 둘 다 선언한 모듈은 유효하지 않은 SPIR-V이므로,
  거부하는 것만이 정직한 답이다. `EXEC_MODE_DENORM_PRESERVE`(4459)와
  `EXEC_MODE_ROUNDING_MODE_RTZ`(4463)가 공개 상수로 추가된다.
- `validate_spirv(words) -> Result<bool, GpuError>`는 고유한 이름의 임시 파일에 쓰인
  모듈에 대해 `spirv-val`(`PATH` 또는 `ES_SPIRV_VAL`)을 실행한다. `Ok(true)`는
  실행되어 수락했다는 뜻; `Ok(false)`는 설치되어 있지 않다는 뜻이며 — 프로세스당
  `NOTE spirv-val is not on PATH: …`를 한 번 남긴다; `Err`는 모듈을 거부했거나, 없는
  상태에서 `ES_REQUIRE_SPIRV_VAL=1`이라는 뜻이다.
- `SlangCompiler::compile`은 `apply_exec_modes` 이후, 캐시 엔트리를 발행하기 전에
  패치된 워드에 대해 `validate_spirv`를 호출한다 — 그래서 캐시 히트는 결코 이 비용을
  치르지 않으며, 유효하지 않은 패치는 결코 캐시에 도달하지 않는다.

## oracle (오라클)

```
cargo test -p es-gpu spirv
cargo test -p es-gpu -- --nocapture
```

- `a_mode_at_another_width_does_not_suppress_the_requested_one` — `DenormFlushToZero
  16`을 지닌 모듈은 너비 `[16, 32]`를 갖고 나온다.
- `every_entry_point_gets_every_mode` — entry가 두 개인 모듈: 세 모드 모두 두 id
  모두에 적용되며, 다시 패치해도 아무 일도 일어나지 않는다(no-op).
- `patch_refuses_a_conflicting_mode_at_the_same_width` — `DenormPreserve 32`에
  `DenormFlushToZero 32`를 요청하면 4459를 명시하는 `Err`가 된다; 같은 쌍이라도 *다른*
  너비에서는 수락된다.
- `patched_kernels_pass_spirv_val` — `ExecModes::deterministic()`로 패치된
  `sum.slang`과 `approx_probe.slang`은 `spirv-val`에 의해 수락된다; 도구가 없으면
  테스트는 `SKIP … no spirv-val on PATH`를 출력하고 이를 오류로 만드는 방법을
  알려준다.

## acceptance (수용 기준)

- 패치된 모듈의 모든 `OpEntryPoint`는 요청된 모든 모드를 너비 32에서 지닌다.
- 다른 너비에서 선언된 모드가 f32 모드를 억제하지 않는다.
- 패치 중인 너비에서 상충하는 모드는 패치가 아니라 오류다.
- 패치 후 `spirv-val`은 결정론 커널들을 수락한다. 이 머신(RTX 4060, `PATH`에 Vulkan
  SDK)에서 관찰됨: `RAN spirv-val: patched sum.slang accepted`,
  `RAN spirv-val: patched approx_probe.slang accepted`.
- 기존 오라클들은 변경되지 않았다: `execution_modes_are_present_in_the_spirv`에서
  exec mode 3/3과 `NoContraction` 1/1, gate 3은 입력 4096개 / 불일치 0.

## forbidden (금지)

- SPIR-V 크레이트 의존성 없음: `spirv.rs`는 `Cargo.toml`에 `rspirv`, `spirv-tools`
  등이 없는, 손으로 작성한 리더/라이터로 남는다. `spirv-val`은 외부 *도구*이며, 호출될
  뿐 링크되지 않는다.
- 테스트를 통과시키려고 검증을 건너뛰지 않으며, 이를 비활성화하는 경로도 없다: 유일한
  레버는 도구의 부재이며, `ES_REQUIRE_SPIRV_VAL=1`이 그것마저 닫는다.
- f16/f64 모드로의 확장 없음 — §3.3은 f32를 요구하며 오직 f32만 패치된다.
- `NoContraction` 패스, 섹션 경계 삽입 로직, `spirv_has_execution_mode`의 공개
  시그니처에는 변경이 없다.
- `buffer.rs`나 `xtask`(P-M4-R6), 또는 캐시 키(P-M4-R5)에는 손대지 않는다.
