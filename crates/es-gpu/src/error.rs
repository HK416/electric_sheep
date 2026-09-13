//! One error type for the crate. No GPU failure is recoverable here — the caller either
//! falls back to CPU (spec §3.3 Apple/MoltenVK row) or reports the missing capability at
//! compile time (§11.6).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    /// No Vulkan loader, or the loader reports no usable instance.
    #[error("no Vulkan loader: {0}")]
    LoaderMissing(String),
    /// The loader works but no physical device has a compute queue.
    #[error("no Vulkan device with a compute queue")]
    NoDevice,
    #[error("Vulkan call failed: {0}")]
    Vulkan(#[from] ash::vk::Result),
    #[error("device memory: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),
    #[error("slangc not found (set ES_SLANGC): {0}")]
    SlangcMissing(String),
    #[error("slangc failed: {0}")]
    SlangcFailed(String),
    #[error("SPIR-V: {0}")]
    Spirv(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A descriptor pool ran out; see `ComputePipeline::MAX_DISPATCHES`.
    #[error("{0}")]
    Exhausted(String),
}
