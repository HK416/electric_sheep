<!-- Korean translation of docs/api-notes/ash.md. The English file is the working copy; regenerate this when it changes. -->

# `ash` — 고정된 API 다이제스트

버전: **`ash = "0.38.0"`** (crate 0.38.0, 헤더 1.3.281), 디바이스 메모리를 위한
**`gpu-allocator = "0.28.0"`** (`default-features = false`, feature `vulkan`)와 함께. 둘 다
워크스페이스 의존성이며, spec §4.2 규칙 3에 따라 `es-gpu`만 이들을 이름으로 지목할 수 있다;
`cargo xtask layering`이 이를 강제한다.

이 머신에 대해 검증됨: NVIDIA RTX 4060 Laptop GPU, 드라이버 592.82, `E:\VulkanSDK`의 Vulkan
SDK. 거기서 실행해 보지 않은 것은 모두 `unverified`로 표시한다.

## Instance

| 항목 | 사용된 값 |
|---|---|
| API 버전 | `vk::make_api_version(0, 1, 3, 0)` |
| Layers | `VK_LAYER_KHRONOS_validation`, `GpuOptions::validation`이 **참이고** 레이어가 존재할 때만; 부재는 오류가 아니다 (`Gpu::validation_enabled()`가 진실을 보고한다) |
| Instance extensions | 없음 |
| Loader | `unsafe { ash::Entry::load() }` (crate feature `loaded`, 기본으로 켜짐) |

## Physical device

아래는 전부 조회(query)이며 아무것도 설정하지 않는다 (spec §3.4).

| 호출 | 구조체 |
|---|---|
| `get_physical_device_properties2` | `pNext` 체인에 네 개의 구조체를 담은 `vk::PhysicalDeviceProperties2` |
| ↳ float controls | `vk::PhysicalDeviceFloatControlsProperties` (Vulkan 1.2 코어; 모든 필드가 `caps::FloatControls`로 그대로 미러링된다) |
| ↳ subgroup | `vk::PhysicalDeviceSubgroupProperties` (1.1 코어) |
| ↳ driver | `vk::PhysicalDeviceDriverProperties` (1.2 코어) — `driverName`, `driverInfo` |
| ↳ id | `vk::PhysicalDeviceIDProperties` (1.1 코어) — `deviceUUID`, spec §22이 열거 순서 대신 요구하는 디바이스 아이덴티티 |
| `get_physical_device_features` | `shaderFloat64`, `shaderInt64` (spec §3.3 정밀도 표) |
| `get_physical_device_memory_properties` | heaps → `caps::MemoryHeap` (spec §20 예산) |
| `enumerate_device_extension_properties` | `VK_KHR_cooperative_matrix`의 존재 여부 (spec §2.4 `VulkanRuntime`; 아직 사용되지 않음) |

`VkShaderFloatControlsIndependence`의 원시 값, `caps::Independence::from_raw`가 미러링함:
`0 = 32_BIT_ONLY`, `1 = ALL`, `2 = NONE`.

ash 0.38은 문자열 getter를 `props.device_name_as_c_str()`,
`driver.driver_name_as_c_str()`, `ext.extension_name_as_c_str()`로 표기하며, 각각
`Result<&CStr, FromBytesUntilNulError>`를 반환한다.

## Device

`vk::QueueFlags::COMPUTE`를 갖는 첫 번째 패밀리에서 큐 하나, 우선순위 `1.0`. 디바이스 확장은
아무것도 활성화하지 않는다. 활성화되는 feature: 지원될 때 `shaderFloat64` / `shaderInt64`.

spec §3.4는 결정적 모드에서 다중 compute 큐를 금지하므로, 두 번째 큐는 절대 생성되지 않으며
API에는 애초에 그것을 요청할 방법이 없다.

## Command submission

`vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER`, primary 버퍼,
`vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT`, recorder당 `vkQueueSubmit` 한 번, 새로 만든
`VkFence`를 `u64::MAX`로 기다린 뒤 파괴. 타임라인 세마포어 없음, 동시에 진행 중인 두 번째
제출 없음: 스케줄은 정적이고 순차적이다 (spec §3.4 항목 4).

디스패치 사이의 배리어: `vk::MemoryBarrier` `SHADER_WRITE → SHADER_READ | SHADER_WRITE`,
스테이지 `COMPUTE_SHADER → COMPUTE_SHADER`.

## Descriptors

Set 0만 사용, `STORAGE_BUFFER` / `UNIFORM_BUFFER`, 각각 `descriptorCount = 1`,
`ShaderStageFlags::COMPUTE`. 한 세트는 크기가 `ComputePipeline::MAX_DISPATCHES = 256`인 풀에서
**디스패치당** 하나씩 할당된다: descriptor 쓰기는 호스트에서 일어나므로, 기록된 두 디스패치
사이에서 세트를 재사용하면 둘 다 마지막 바인딩으로 실행되어 버린다. `reset_descriptors()`
(즉 `vkResetDescriptorPool`)는 그 세트를 사용한 제출이 완료된 뒤에만 합법적이다.

## `gpu-allocator` 0.28

```rust
Allocator::new(&AllocatorCreateDesc {
    instance, device, physical_device,
    debug_settings: AllocatorDebugSettings::default(),
    buffer_device_address: false,
    allocation_sizes: AllocationSizes::default(),
})
```

`allocate(&AllocationCreateDesc { name, requirements, location, linear: true,
allocation_scheme: AllocationScheme::GpuAllocatorManaged })`. storage/uniform에는
`MemoryLocation::GpuOnly`, staging에는 `CpuToGpu`. `allocation.mapped_ptr()`는 정확히
호스트-가시(host-visible) 메모리에 대해서만 `Some`이며, 이것이 `Buffer::upload`/`download`가
memcpy와 staging 복사 중 무엇을 쓸지 정하는 방법이다.

allocator는 복제된 `Instance`/`Device` 핸들을 들고 있으므로, 디바이스보다 **먼저** drop되어야
한다: `Gpu`는 이를 `ManuallyDrop`에 담아 두고 자신의 `Drop`에서 가장 먼저 drop한다.

## 사용하지 않는 것

그래픽 파이프라인, 렌더 패스, 스왑체인, 레이 트레이싱, 타임라인 세마포어, 외부 메모리
(`VK_KHR_external_memory_*`, spec §21 — CUDA/HIP 심볼이 필요한데 §4.2 규칙 5에 따라
`es-transport`만 이를 링크할 수 있다), `VkPipelineCache`, 쿼리 풀. 각각이 왜 미뤄졌는지는
`docs/design/gpu-foundation.md`를 참고.
