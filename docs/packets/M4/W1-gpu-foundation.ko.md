<!-- Korean translation of docs/packets/M4/W1-gpu-foundation.md. The English file is the working copy; regenerate this when it changes. -->

# W1-gpu-foundation — `es-gpu`: 디바이스, capability, Slang, 버퍼, 파이프라인

Spec: §2.3 (Slang → SPIR-V, 오프라인 + content-hash 캐시), §3.3–§3.4 (정밀도, 여덟 항목짜리
결정적 실행 계약), §3.5 (결정성 계층), §4.2 규칙 3 (layer 3 아래에서는 `es-gpu`만
Vulkan을 안다), §11.4–§11.6 (GPU lowering, 계획, capability 검사), §21
(`TensorTransport`), §22 (multi-GPU), §28.7 게이트 3. Design:
`docs/design/gpu-foundation.md`.

## context (범위)

```
crates/es-gpu/**
Cargo.toml                      ([workspace.dependencies] 아래에만 `ash`와 `gpu-allocator` 추가)
docs/api-notes/ash.md
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/W1-gpu-foundation.md
```

## spec (사양)

- `Gpu::open(GpuOptions { prefer_discrete, validation })`는 **하나**의 compute 큐를
  가진 **하나**의 디바이스를 연다(§3.4는 결정적 모드에서 multi-queue를 금지한다;
  API는 두 번째 큐를 제공하지 않는다). `Gpu::none_available()`이 CI 판정식이다.
- `Capabilities`는 *조회*로 채워진다(`vkGetPhysicalDeviceProperties2`):
  `VkPhysicalDeviceFloatControlsProperties`의 모든 필드, subgroup 크기와 연산
  클래스, `shaderFloat64`/`shaderInt64`, `VK_KHR_cooperative_matrix`, workgroup
  한계, 메모리 heap, 디바이스 UUID(§22는 열거 순서로 디바이스를 식별하는 것을
  금지한다). 어떤 것도 float control을 쓰지 않는다 — §3.4와 부록 C는 이것이
  설정이 아니라고 명시한다.
- `Capabilities::determinism_tier()` → f32 denorm-flush, RTE, signed-zero/inf/nan-preserve가
  모두 지원되고 32비트에 대해 독립적으로 설정 가능하며 subgroup 크기가 보고될 때
  `Bitwise`; 그 외에는 `CrossBackend`.
- `Capabilities::deterministic_execution_modes()` → `ExecModes`, **컴파일러**가
  적용하는 SPIR-V 실행 모드와 `NoContraction` 정책(§3.4 3단계). 디바이스가 광고하지
  않는 모드는 요청되지 않는다 — 모듈 안의 지원되지 않는 capability는 유효하지 않은
  SPIR-V다; 그 대신 계층이 부족분을 보고한다.
- `SlangCompiler::compile(source, entry, profile, defines, modes)`는 `slangc`를
  호출하며(`ES_SLANGC` 또는 `PATH`)
  `target/es-slang-cache/<blake3(source+entry+profile+defines+modes+includes+slangc version)>.spv`에
  캐시한다. 히트는 프로세스를 시작하지 않는다. `slangc 2026.8`은 `DenormFlushToZero`만
  방출한다; 나머지 두 실행 모드와 `NoContraction` 데코레이션은
  `spirv::apply_exec_modes`가 바이너리에 패치해 넣으며,
  `spirv_has_execution_mode` / `spirv_no_contraction_count`가 테스트를 위해 이를
  바이너리에서 다시 읽어낸다.
- `upload`/`download`를 갖는 `Buffer::new(gpu, bytes, Usage::{Storage, Uniform,
  Staging})`; `ComputePipeline::new(gpu, &SpirvModule, &[BindingDesc])`;
  `CommandRecorder::dispatch(pipeline, buffers, groups)` + `submit_and_wait`,
  모든 디스패치 뒤에 명시적인 shader-write → shader-read 배리어. atomic도,
  multi-queue도, FP atomic 헬퍼도 없다(§3.4가 이를 금지하므로 제공되지 않는다).
- `kernels/sum.slang`은 고정 순서 pairwise 트리 리덕션이다(shared 메모리 없음,
  subgroup 연산 없음, atomic 없음, 디바이스에서 유도된 workgroup count 없음).
  `kernels/approx_probe.slang`은 `crates/es-math/slang/approx.slang`을
  `#include`하고 `es_sin` / `es_exp`를 평가한다.

## oracle (오라클)

```
cargo fmt -p es-gpu --check
cargo clippy -p es-gpu --all-targets -- -D warnings
cargo test -p es-gpu -- --nocapture
cargo xtask layering && cargo xtask context-budget && cargo xtask check-spec-refs
```

모든 GPU 테스트는 디바이스(또는 `slangc`)가 없으면 `SKIP <test>: <reason>`을
출력하고 반환하므로, 스위트는 GPU 없는 CI에서도 통과하며 그렇게 밝힌다.
단위 테스트 — 미리 준비된 property 구조체로부터의 계층 도출, 캐시 키 분리,
SPIR-V 파서/패처/검사기 — 는 어디서나 실행된다.

## acceptance (수용 기준)

NVIDIA RTX 4060 Laptop GPU, 드라이버 592.82, Vulkan 1.4.325, Slang 2026.8에서
측정됨:

- `device_capabilities_are_queried_and_reported` — 디바이스, 드라이버,
  float-controls, subgroup, 도출된 계층을 출력한다. 이 디바이스는
  `shaderDenormFlushToZeroFloat32 = false`를 보고하므로(NVIDIA는 f32 denormal을
  보존한다), 도출된 계층은 **2 / `CrossBackend`**이며, flush-to-zero 모드는
  컴파일러에 요청되지 않는다.
- `execution_modes_are_present_in_the_spirv` — 세 실행 모드 모두 컴파일된
  바이너리에서 발견되며, `NoContraction`은 모든 float 산술 결과를 커버한다
  (`sum.slang`에 대해 1/1).
- `cache_hit_does_not_start_slangc` — 첫 컴파일은 `slangc`를 한 번 호출하고, 두
  번째는 추가 호출 없이 캐시로부터 동일한 워드를 제공한다.
- `tree_reduction_is_bit_identical_across_runs_and_matches_the_cpu` — 100만 개의
  f32, 두 번의 실행이 비트 동일하며, 같은 트리의 CPU 미러와도 비트 동일하다
  (최대 ULP 0). `BinnedAcc`(§18.4)의 정확한 합은 정확도 참조다: 상대 차이
  2.0e-7, f32 트리에 대해 기대되는 대로.
- `gate3_cpu_and_slang_transcendentals_agree_bit_for_bit` — **이 디바이스에서
  spec §28.7 게이트 3 충족**: 입력 4096개, 불일치 0개, 최대 ULP `sin` 0, `exp` 0.

## forbidden (금지)

`context` 밖의 모든 파일. 루트 `Cargo.toml`의 다른 의존성 줄. float control을
코드나 주석에서 설정으로 취급하는 것(§3.4, 부록 C). 두 번째 큐, FP-atomic
헬퍼, 또는 호출자가 배리어를 옵트아웃할 수 있게 하는 어떤 API. 새로운
확장-지점 trait(INV-17). `HashMap`/`HashSet`(§3.4). `crates/es-math/slang/approx.slang`이나
계수 미러를 편집하는 것 — 게이트 3 실패는 최대 ULP로 보고될 뿐, 참조를 옮겨서
고치는 일은 없다. 그래픽 파이프라인, 레이 트레이싱, 외부 메모리, multi-GPU
(미뤄짐; 설계 문서 참고).
