//! `es-gpu` (layer 2, spec §4.2): the only crate below layer 3 that links Vulkan.
//!
//! What it owns: opening a device and *querying* its capabilities (spec §3.4 step 1),
//! compiling Slang to SPIR-V with a content-hash cache (§2.3, §11.4), buffers, compute
//! pipelines and one compute queue.
//!
//! What it does not own: which kernel to run, how to fuse, what to schedule — that is
//! `es-compile` (§11). Graphics, ray tracing, `TensorTransport` (§21) and multi-GPU (§22)
//! are deferred; see `docs/design/gpu-foundation.md`.
//!
//! # Determinism
//!
//! Vulkan float-controls are a **capability query**, not a setting (spec §3.4, Appendix C).
//! Nothing here writes one. [`Capabilities::determinism_tier`] reads the query to say which
//! tier of §3.5 the device can carry, and [`Capabilities::deterministic_execution_modes`]
//! returns the SPIR-V execution modes the *compiler* must apply. The eight-item contract of
//! §3.4 needs five more things this crate only partly supplies: the single queue and the
//! static schedule are here, the deterministic reduction is `es_math::reduce`, and the fixed
//! algorithm and fixed subgroup behaviour are properties of the kernel.

pub mod buffer;
pub mod caps;
pub mod device;
pub mod error;
pub mod instance;
pub mod pipeline;
pub mod slang;
pub mod spirv;

pub use buffer::{Buffer, Usage};
pub use caps::{
    Capabilities, DeterminismTier, ExecModes, FloatControls, Independence, MaxWorkgroup,
    MemoryHeap, Subgroup,
};
pub use device::Gpu;
pub use error::GpuError;
pub use instance::GpuOptions;
pub use pipeline::{BindingDesc, BindingKind, CommandRecorder, ComputePipeline};
pub use slang::{SlangCompiler, SpirvModule};
pub use spirv::{
    apply_exec_modes, spirv_entry_points, spirv_float_arithmetic_count, spirv_has_execution_mode,
    spirv_no_contraction_count, DECORATION_NO_CONTRACTION, EXEC_MODE_DENORM_FLUSH_TO_ZERO,
    EXEC_MODE_ROUNDING_MODE_RTE, EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
};
