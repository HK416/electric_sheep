# `ash` — pinned API digest

Version: **`ash = "0.38.0"`** (crate 0.38.0, headers 1.3.281), with
**`gpu-allocator = "0.28.0"`** (`default-features = false`, feature `vulkan`) for device
memory. Both are workspace dependencies and, by spec §4.2 rule 3, only `es-gpu` may name
them; `cargo xtask layering` enforces it.

Verified against this machine: NVIDIA RTX 4060 Laptop GPU, driver 592.82, Vulkan SDK at
`E:\VulkanSDK`. Anything not exercised there is marked `unverified`.

## Instance

| Item | Value used |
|---|---|
| API version | `vk::make_api_version(0, 1, 3, 0)` |
| Layers | `VK_LAYER_KHRONOS_validation`, only when `GpuOptions::validation` **and** the layer is present; absence is not an error (`Gpu::validation_enabled()` reports the truth) |
| Instance extensions | none |
| Loader | `unsafe { ash::Entry::load() }` (crate feature `loaded`, on by default) |

## Physical device

All of the following are queries; nothing is set (spec §3.4).

| Call | Struct |
|---|---|
| `get_physical_device_properties2` | `vk::PhysicalDeviceProperties2` with four structs in the `pNext` chain |
| ↳ float controls | `vk::PhysicalDeviceFloatControlsProperties` (Vulkan 1.2 core; every field is mirrored into `caps::FloatControls`) |
| ↳ subgroup | `vk::PhysicalDeviceSubgroupProperties` (1.1 core) |
| ↳ driver | `vk::PhysicalDeviceDriverProperties` (1.2 core) — `driverName`, `driverInfo` |
| ↳ id | `vk::PhysicalDeviceIDProperties` (1.1 core) — `deviceUUID`, the device identity spec §22 requires instead of enumeration order |
| `get_physical_device_features` | `shaderFloat64`, `shaderInt64` (spec §3.3 precision table) |
| `get_physical_device_memory_properties` | heaps → `caps::MemoryHeap` (spec §20 budget) |
| `enumerate_device_extension_properties` | presence of `VK_KHR_cooperative_matrix` (spec §2.4 `VulkanRuntime`; not used yet) |

`VkShaderFloatControlsIndependence` raw values, mirrored by `caps::Independence::from_raw`:
`0 = 32_BIT_ONLY`, `1 = ALL`, `2 = NONE`.

ash 0.38 spells the string getters `props.device_name_as_c_str()`,
`driver.driver_name_as_c_str()`, `ext.extension_name_as_c_str()`, each returning
`Result<&CStr, FromBytesUntilNulError>`.

## Device

One queue, from the first family with `vk::QueueFlags::COMPUTE`, priority `1.0`. No device
extensions are enabled. Enabled features: `shaderFloat64` / `shaderInt64` when supported.

Spec §3.4 forbids multiple compute queues in deterministic mode, so a second queue is never
created and the API offers no way to ask for one.

## Command submission

`vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER`, primary buffers,
`vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT`, one `vkQueueSubmit` per recorder, a fresh
`VkFence` waited with `u64::MAX` then destroyed. No timeline semaphores, no second submit in
flight: the schedule is static and serial (spec §3.4 item 4).

Barrier between dispatches: `vk::MemoryBarrier` `SHADER_WRITE → SHADER_READ | SHADER_WRITE`,
stages `COMPUTE_SHADER → COMPUTE_SHADER`.

## Descriptors

Set 0 only, `STORAGE_BUFFER` / `UNIFORM_BUFFER`, `descriptorCount = 1` each,
`ShaderStageFlags::COMPUTE`. One set is allocated **per dispatch** from a pool sized
`ComputePipeline::MAX_DISPATCHES = 256`: descriptor writes happen on the host, so a set
reused between two recorded dispatches would run both with the last binding.
`reset_descriptors()` (i.e. `vkResetDescriptorPool`) is legal only after the submit that used
the sets has completed.

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
allocation_scheme: AllocationScheme::GpuAllocatorManaged })`. `MemoryLocation::GpuOnly` for
storage/uniform, `CpuToGpu` for staging. `allocation.mapped_ptr()` is `Some` exactly for
host-visible memory, which is how `Buffer::upload`/`download` decide between a memcpy and a
staging copy.

The allocator holds cloned `Instance`/`Device` handles, so it must be dropped **before** the
device: `Gpu` keeps it in a `ManuallyDrop` and drops it first in its own `Drop`.

## Not used

Graphics pipelines, render passes, swapchains, ray tracing, timeline semaphores, external
memory (`VK_KHR_external_memory_*`, spec §21 — needs CUDA/HIP symbols that by §4.2 rule 5
only `es-transport` may link), `VkPipelineCache`, query pools. See
`docs/design/gpu-foundation.md` for why each is deferred.
