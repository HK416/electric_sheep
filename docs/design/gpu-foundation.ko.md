<!-- Korean translation of docs/design/gpu-foundation.md. The English file is the working copy; regenerate this when it changes. -->

# GPU foundation (`es-gpu`, layer 2)

Spec: §2.3 (Slang → SPIR-V, 오프라인 컴파일 + content-hash 캐시), §3.3–§3.4 (정밀도, 결정적
실행 계약), §3.5 (결정성 계층), §4.2 규칙 3 (layer ≤ 2는 Vulkan 심볼을 모른다, `es-gpu`는
예외), §11.4–§11.6 (GPU lowering, release/debug 계획, capability 검사), §21
(`TensorTransport`), §22 (multi-GPU), §28.7 게이트 3.

`es-gpu`는 Vulkan을 링크하는 유일한 crate다. 이것이 소유하는 것: 디바이스 open +
capability *조회*, Slang 호출 + SPIR-V 캐시, 버퍼, compute 파이프라인, compute 큐 하나.
정책은 소유하지 않는다: 어떤 커널을 실행할지, 어떻게 융합할지, 무엇을 스케줄할지는
§11(`es-compile`)의 일이다.

## 디바이스 선택

`Gpu::open(GpuOptions { prefer_discrete, validation })`:

1. API 1.3으로 `VkInstance`를 생성한다. `validation = true`는
   `VK_LAYER_KHRONOS_validation`을 추가한다; 레이어가 없으면 호출은 그것 없이도
   성공하며(CI 머신에는 SDK가 없다) `Gpu::validation_enabled()`가 실제로 무슨 일이
   일어났는지 보고한다.
2. 물리 디바이스를 열거한다. 점수: `prefer_discrete`일 때 discrete > integrated >
   virtual > CPU, 그렇지 않으면 역순; 동점은 열거 인덱스로 깨진다. 열거 순서는 아이덴티티가
   *아니다* — §22는 열거 순서로 디바이스를 매칭하는 것을 금지하므로, 디바이스 UUID가
   `Capabilities`에 실려 실행 해시(§5.3)의 `hardware_capability`로 들어간다.
3. `COMPUTE`를 갖는 첫 번째 큐 패밀리를 골라, 그로부터 정확히 **하나**의 큐로 디바이스를
   만든다. §3.4는 결정적 모드에서 다중 compute 큐를 금지하므로, 다른 큐는 절대 생성되지
   않는다; 여기에는 그것을 요청할 API가 없다.
4. `Gpu::none_available()`은 CI 판정식이다: loader가 없거나, 인스턴스 생성이 실패하거나,
   보고된 물리 디바이스가 없을 때 참이다. 모든 GPU 테스트는 이것으로 시작해 이유와 함께
   `SKIP`을 출력한다. `cargo xtask ci`는 테스트 단계를 `--nocapture`로 실행하고 그
   `SKIP` 줄들을 스캔한다: `ES_REQUIRE_GPU=1`이면(GPU가 있다고 주장하는 머신) 그중
   하나라도 있으면 실행이 실패하는데, 이는 `cargo xtask nostd --require`가 타깃 부재를
   다루는 방식과 같다. PR 러너에는 GPU가 없고 이 변수를 설정하지 않으므로, 거기서는
   스킵이 보고만 될 뿐이다.

## Capability는 조회될 뿐, 절대 설정되지 않는다

§3.4는 명시적이고 부록 C도 이를 반복한다: `VkPhysicalDeviceFloatControlsProperties`는
capability **조회**다. 이 crate 안 어디에도 float-control을 쓰는 곳은 없다; 구조체는
`vkGetPhysicalDeviceProperties2`로부터 채워지고 읽힐 뿐이다.

```
Capabilities {
    device_name, device_uuid, device_type, driver_name, driver_info, driver_version,
    api_version,
    float_controls: FloatControls,          // Vulkan properties 구조체의 모든 필드
    subgroup: Subgroup { size, supported_ops, quad_operations_in_all_stages },
    shader_float64, shader_int64, cooperative_matrix: bool,
    max_workgroup: { count: [u32; 3], size: [u32; 3], invocations },
    memory_heaps: Vec<MemoryHeap>,          // 크기 + device-local 플래그, §20 예산을 위해
}
```

`cooperative_matrix`는 디바이스 확장 목록 안의 `VK_KHR_cooperative_matrix`의 존재
여부다(§2.4 `VulkanRuntime`); 이 crate는 아직 이를 사용하지 않는다.

### `determinism_tier()`

도출 방식 (§3.4 2단계, §3.5):

| 조건 | 계층 |
|---|---|
| `shader_denorm_flush_to_zero_float32` **그리고** `shader_rounding_mode_rte_float32` **그리고** `shader_signed_zero_inf_nan_preserve_float32`, **그리고** 그 모드가 우리에게 불리하게 전역적으로 강제되어 있지 않음 — 즉 `denorm_behavior_independence` / `rounding_mode_independence`가 `All`이거나 `32BitOnly`, **그리고** subgroup 크기가 보고됨 | `Bitwise` (계층 1) |
| 그 외 | `CrossBackend` (계층 2) |

`Bitwise`는 §3.5 계층 1이 *같은 디바이스와 드라이버에서 같은 `execution_hash`가
주어졌을 때* 약속하는 것이다: 드라이버와 디바이스 아이덴티티는 그 해시의 일부이며,
capability 조회가 확립할 수 있는 것이 아니다. 이 crate가 tier-1을 주장한다는 것은
"나머지 네 손잡이(float control, no-contraction, 리덕션 순서, 단일 큐 정적 스케줄)를
이 디바이스에서 고정할 수 있다"는 뜻일 뿐, 그 이상은 아니다. 여덟 항목짜리 계약의
나머지 네 항목(결정적 리덕션, 정적 분할, 고정된 subgroup 동작, 고정된 알고리즘)은
호출자의 몫이다; `es-math::reduce`와 여기 있는 커널들이 그것을 제공한다.

### `deterministic_execution_modes()`

`ExecModes`를 반환한다 — **컴파일러**가 모듈에 적용해야 하는 SPIR-V 실행 모드와
`NoContraction` 데코레이션 정책이다 (§3.4 3단계). 이는 디바이스 설정이 아니라 컴파일
입력이다; 그래서 이름이 `execution_modes`이며 `Gpu`에는 의도적으로 setter가 어디에도
없다.

```
ExecModes { denorm_flush_to_zero_f32, rounding_mode_rte_f32, signed_zero_inf_nan_preserve_f32,
            no_contraction }
```

`ExecModes::deterministic()`은 네 개를 모두 켠다; `ExecModes::none()`은 결정적 계약
밖에 있는 커널을 위한 허용적인 기본값이다 (§3.2는 신경망 커널을 예외로 둔다).

## Slang → SPIR-V

`SlangCompiler::compile(source, entry, profile, defines, exec_modes) -> SpirvModule`.

- `slangc`는 `ES_SLANGC` 또는 `PATH`를 통해 찾는다. 그 `-v` 출력은 캐시 키에 들어가는데,
  §3.4 항목 7이 소스뿐 아니라 컴파일러 버전도 고정하기 때문이다.
- 캐시: `target/es-slang-cache/<hash>.spv`, `hash = blake3(source ‖ entry ‖ profile ‖
  defines ‖ exec_modes ‖ slangc version ‖ include dirs ‖그 아래의 모든 파일)`. include
  순회는 **재귀적**이고 정렬되며, 각 파일의 경로를 그것의 include root에 대한 상대
  경로로 키로 삼는다. 그래서 `#include "sub/helper.slang"`도 아이덴티티의 일부가
  된다; 순회 오류는 키를 조용히 약화시키는 대신 그대로 전파된다. 히트는 `slangc`를
  전혀 호출하지 않는다; `SlangCompiler::invocations()`가 실제 호출 횟수를 세어 테스트가
  히트를 증명할 수 있게 한다. 콘텐츠 어드레싱이므로 랭크 간에 공유 가능하고(§22),
  배포가 Slang을 필요로 하지 않도록 번들에 담을 수 있다(§11.4).
- 플래그: `-target spirv -profile <p> -entry <e> -O0 -fp-mode precise
  -emit-spirv-directly`, 모드가 요청되면 `-denorm-mode-fp32 ftz`도 추가. 정확한 플래그와
  각각이 실제로 무엇을 방출하는 것으로 관찰되었는지: `docs/api-notes/slang.md`.
- `slangc`가 방출하지 **않는** 것은 모듈을 패치해 추가한다: `RoundingModeRTE`,
  `SignedZeroInfNanPreserve`, 그리고 `NoContraction` 데코레이션. 패처는 손으로 짠
  SPIR-V writer(`spirv.rs`, 약 200줄, 의존성 없음)이며, capability, `SPV_KHR_float_controls`
  확장, 진입점 뒤의 `OpExecutionMode`들, 그리고 float 산술 결과 id마다 하나씩의
  `OpDecorate <id> NoContraction`을 삽입한다.
  이 패치는 `(진입점, 모드, float 너비)`로 키가 매겨진다: *모든* `OpEntryPoint`를
  다루고, 다른 너비에 선언된 모드는 요청된 f32 모드를 억제하지 않으며, 이미 그
  너비에 모순되는 모드를 선언한 모듈(요청된 `DenormFlushToZero 32`에 대한
  `DenormPreserve 32`)은 패치되어 유효하지 않은 것으로 만들어지는 대신
  `GpuError::Spirv`로 거부된다.
- `spirv_has_execution_mode(&words, mode)` / `spirv_no_contraction_count(&words)`는
  헤더와 명령 스트림을 파싱하므로, 테스트는 플래그를 신뢰하는 대신 그 모드들이 바이너리
  안에 있음을 *증명*할 수 있다.
- `validate_spirv(&words)`는 콜드 컴파일마다 패치된 모듈에 대해 **`spirv-val`**
  (Vulkan SDK, `PATH` 또는 `ES_SPIRV_VAL`)을 실행한다 — 헤더 파싱은 워드 스트림이
  잘 구성되어 있다는 것을 말할 뿐 모듈이 합법적이라는 것을 말하지 않으며, 이
  crate는 SPIR-V를 손으로 작성한다. 도구가 없으면: 출력된 `NOTE`와 함께 `Ok(false)`,
  또는 `ES_REQUIRE_SPIRV_VAL=1` 아래에서는 오류.

`SpirvModule { words, hash, entry }`. 이 해시가 §11.4의 `compile_hash` 구성 요소다.

## 버퍼와 파이프라인

`gpu-allocator`가 디바이스 메모리를 소유한다; 버퍼는 `Storage` / `Uniform` /
`Staging`(host-visible)을 갖는 `Buffer::new(gpu, bytes, Usage)`다. `upload` /
`download`는 staging 버퍼와 일회성 커맨드 버퍼를 통해 복사한다. `Buffer`는 요청받은
`bytes`를 실제로 할당된 것(`bytes.max(4)`에 할당자의 정렬 패딩을 더한 것)과 함께
기록하며, `download`는 정확히 요청된 길이를 반환한다 — 패딩이 데이터인 양 돌려주는
일은 없다.

`ComputePipeline::new(gpu, &SpirvModule, &[BindingDesc])`는 단일 descriptor set을
만든다(set 0, storage 또는 uniform 버퍼만). `CommandRecorder`는 하나의 compute 큐에
`dispatch(pipeline, bindings, groups)` 호출을 기록하고, 디스패치 사이에 명시적
`SHADER_WRITE → SHADER_READ` 메모리 배리어를 삽입하며, `submit_and_wait`는 fence로
블로킹한다.

atomic도, multi-queue도, 비동기 전송도, 타임라인 세마포어도 없다: §3.4는 결정적
모드에서 atomic FP 합산과 multi-queue 물리를 금지하므로 API는 이를 제공하지 않는다.

## 커널과 게이트 3

`kernels/sum.slang` — 고정 순서 pairwise 트리 리덕션, 레벨당 한 패스, shared 메모리
없음, subgroup 연산 없음, atomic 없음. 순서는 인덱스 산술의 속성이므로 스케줄링에
의존할 수 없다. CPU 미러는 `f32`로 동일한 트리를 수행한다; `BinnedAcc`(§18.4)는
정확도 참조이지 비트 참조가 아니다 — `f32` 트리와 정확한 binned 합은 구성상 다른
숫자다.

`kernels/approx_probe.slang` — `crates/es-math/slang/approx.slang`을 `#include`하고
`es_sin` / `es_exp`를 평가한다; 테스트는 결과를 CPU의 `es_math::approx`와 비트 단위로
비교한다. 이것이 §28.7 게이트 3이다(CPU/Slang 비트 동등성). 결과는 비트가 같음 또는
최대-ULP 수치와 게이트 미충족 표시로 보고되며 — 허용오차에 의한 통과로는 절대
보고되지 않는다.

## 미룬 것

그래픽 파이프라인과 렌더러 (§16, `es-render`, layer 5). 레이 트레이싱 (§28.6). 여러
큐와 비동기 compute (어차피 §3.4에서 금지됨). 외부 메모리 / `TensorTransport` 협상
(§21) — CUDA/HIP 심볼이 필요한데, §4.2 규칙 5에 따라 `es-transport`만 이를 링크할 수
있다. Multi-GPU 열거와 역할 (§22). 형태 공유, 커널 융합, release/debug 계획 분리를
위한 specialization constant (§11.4–§11.5) — 이들은 `es-compile`의 결정 사항이다; 이
crate는 주어진 것을 컴파일하고 디스패치할 뿐이다. §12.4 지표 집합을 위한 타임스탬프
쿼리. 파이프라인 캐시 (SPIR-V 캐시는 콘텐츠 어드레스 방식이다; `VkPipelineCache`는
결정성 가치가 없는 드라이버 쪽 부가물이다).
