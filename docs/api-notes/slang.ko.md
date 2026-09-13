<!-- Korean translation of docs/api-notes/slang.md. The English file is the working copy; regenerate this when it changes. -->

# `slangc` — 고정된 API 다이제스트

컴파일러: **Slang `2026.8`** (`E:\VulkanSDK`의 Vulkan SDK에 있는 `slangc.exe`, `PATH` 또는
`ES_SLANGC`로 찾음). 버전 문자열은 모든 캐시 키에 들어간다: spec §3.4 항목 7은 소스뿐 아니라
컴파일러도 고정한다.

아래 관측은 이 머신의 `slangc 2026.8`에서 나온 것이며, `spirv-dis`로 출력을 역어셈블해
확인했다. 그렇게 확인하지 않은 것은 모두 `미검증 (unverified)`로 표시한다.

## 호출

```
slangc <src>.slang -target spirv -profile glsl_450 -entry main -emit-spirv-directly \
       -O0 -fp-mode precise [-denorm-mode-fp32 ftz] [-I <dir>]... [-D<k>=<v>]... -o <out>.spv
```

| 플래그 | 이유 | 상태 |
|---|---|---|
| `-target spirv` | SPIR-V 출력 (spec §2.3) | 검증됨 |
| `-profile glsl_450` | compute 프로파일; SPIR-V 1.3 헤더가 방출된다 | 검증됨 |
| `-entry main` | 진입점 함수 | 검증됨 |
| `-emit-spirv-directly` | GLSL + glslang을 거치지 않는다 (2024년 이후 기본값이지만, 향후 기본값이 바뀌어도 결과가 흔들리지 않도록 명시적으로 전달) | 검증됨 |
| `-O0` | 최적화기 없음. 비트 재현성을 죽이는 것은 재결합(reassociation)이다 (spec §3.4) | 검증됨 (SPIR-V가 소스의 연산 순서를 유지한다) |
| `-fp-mode precise` | fast-math 없음 | 받아들여짐이 검증됨; 이것만으로는 `NoContraction`을 추가하지 **않는다** |
| `-denorm-mode-fp32 ftz` | `OpCapability DenormFlushToZero`, `OpExtension "SPV_KHR_float_controls"`, `OpExecutionMode %main DenormFlushToZero 32`를 방출한다 | `spirv-dis`로 검증됨 |
| `-I <dir>` | include 경로, `crates/es-math/slang/approx.slang`에 사용됨 | 검증됨 |
| `-D<k>=<v>` | 전처리기 정의 (예: `ES_N`) | 검증됨 |
| `-o` | 출력 경로 | 검증됨 |

`slangc -v`는 버전을 stdout이 아니라 **stderr**에 출력한다.

## `slangc`가 방출하지 않는 것

`2026.8`은 spec §3.4 3단계의 나머지 두 실행 모드에 대한 플래그도, `NoContraction`
데코레이션에 대한 플래그도 없다:

* `OpExecutionMode … RoundingModeRTE 32` — 플래그를 찾지 못함 (`slangc -h`는
  `-denorm-mode-fp{16,32,64}` 계열만 나열한다);
* `OpExecutionMode … SignedZeroInfNanPreserve 32` — 마찬가지;
* `OpDecorate <id> NoContraction` — `-fp-mode precise`는 컴파일된 `sum.slang`에서 이를
  전혀 만들어내지 않았다.

그래서 `es-gpu`는 **컴파일된 모듈을 패치한다** (`src/spirv.rs`): 빠진 `OpCapability` /
`OpExtension` / `OpExecutionMode` 명령을 추가하고, float 산술 결과마다 `OpDecorate …
NoContraction` 하나씩을 추가한다. 이 패치는 멱등(idempotent)이며, 어떤 플래그도 신뢰하지 않고
최종 바이너리를 파싱하는 테스트 `execution_modes_are_present_in_the_spirv`로 검증된다. 이후의
Slang 릴리스가 해당 플래그를 갖추게 되어도 이 패치는 계속 올바르게 동작하며(이미 있는 것은
건너뛴다) 그 시점에 커맨드라인에 플래그를 추가하면 된다.

이 패치는 모드 단위가 아니라 `(entry point, mode, float width)` 단위다:

* **모든** `OpEntryPoint`에 대해 실행되므로, 멀티 엔트리 모듈은 모든 엔트리에서 모드를
  얻는다;
* `DenormFlushToZero 16`은 요청된 `DenormFlushToZero 32`로 치지 않는다 — width
  피연산자를 비교한다;
* 요청된 width에서 이미 *모순되는* 모드를 선언한 모듈(요청된 `DenormFlushToZero 32`에
  대한 `DenormPreserve 32`, 또는 요청된 `RoundingModeRTE 32`에 대한 `RoundingModeRTZ 32`)은
  잘못된 모듈로 패치되는 대신 `GpuError::Spirv`로 **거부된다**.

## 검증

`spirv.rs`는 SPIR-V를 손으로 작성하므로, 그 결과는 콜드 컴파일마다 **`spirv-val`**
(Vulkan SDK)로 검사된다 — 캐시 히트는 이 비용을 절대 치르지 않는다. 실행 파일은
`PATH`상의 `spirv-val` 또는 `ES_SPIRV_VAL`이다. 이것이 없으면 `validate_spirv`는
프로세스당 `NOTE spirv-val is not on PATH: …`를 한 번 출력하고 `Ok(false)`를
반환한다; `ES_REQUIRE_SPIRV_VAL=1`이면 그 부재는 대신 오류가 된다.
`patched_kernels_pass_spirv_val`은 `sum.slang`과 `approx_probe.slang` 둘 다 패치
후에 유효하게 나오는지 단언한다.

`미검증 (unverified)`: Slang 속성(`[require(...)]`, `[SpvExecutionMode(...)]` 또는 유사한 것)이
소스 안에서 `RoundingModeRTE`를 표현할 수 있는지 여부. 소스 레벨 경로는 `slangc -h` 출력에서
발견되지 않았다; 바이너리에서 확인 가능하다는 이유로 패치 방식을 선택했다.

## Cache

`target/es-slang-cache/<blake3>.spv`, `ES_SLANG_CACHE`(또는 `CARGO_TARGET_DIR`)로
오버라이드 가능. 키는 다음을 해시한다: 소스 텍스트, entry, profile, defines, 요청된
`ExecModes`, 모든 include 디렉터리 경로 **그리고 그로부터 재귀적으로 도달 가능한 모든
파일의 내용**, 그리고 `slangc -v` 문자열. 캐시 히트는 프로세스를 시작하지 않는다 —
`SlangCompiler::invocations()`가 `cache_hit_does_not_start_slangc`에서 이를 증명한다.

include walk은 정렬되어 있고, 각 파일의 경로를 include 루트에 대한 상대 경로로 키
삼으며(그래서 파일을 하위 디렉터리 사이로 옮기면 키도 바뀐다), **자신의 오류를
전파한다**: 읽을 수 없는 include 트리는, 오래된 `.spv`를 내어줄 더 약한 키를 조용히
만들어내는 대신 컴파일을 실패시킨다. `editing_an_included_subdirectory_file_recompiles`가
그 오라클이다 — `sub/helper.slang`을 `#include`하는 커널을, 그 사이에 helper를
편집해가며 세 번 컴파일하면 정확히 두 번의 `slangc` 호출이 나와야 한다.

콘텐츠 어드레싱 덕분에 랭크들이 캐시를 공유할 수 있고 (spec §22), `es task compile`이 이를
번들에 담을 수 있어 배포에 Slang이 필요 없다 (§11.4).

## 셰이더 관례

* 모든 리소스에 `[[vk::binding(<binding>, 0)]]`: descriptor set 0, 명시적 바인딩.
* `RWStructuredBuffer<float>` / `StructuredBuffer<float>` 모두 `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`로
  귀결된다 (SPIR-V 1.3은 이들을 `Uniform` storage class 안의 `BufferBlock`으로 방출한다).
* `-fvk-use-entrypoint-name`을 넘기지 않는 한 SPIR-V 진입점은 Slang 함수 이름과 무관하게
  `main`으로 명명된다; `ComputePipeline`은 이를 가정하지 않고 모듈에서 이름을 읽어낸다.
* 여기서는 어디서나 `[numthreads(64, 1, 1)]`이므로, 워크그룹 크기는 커널의 상수이며 디바이스
  속성에서 유도되는 일이 절대 없다 (spec §3.4 항목 6).
</content>
