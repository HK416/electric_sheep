//! `es-compile` (layer 7): the compiler pipeline of spec 11.
//!
//! First piece, M1 W3: the **CPU reference execution plan** for Observation IR (spec 11.3).
//! It is the oracle, not a performance path — the Slang/SPIR-V lowering of spec 11.4 is
//! judged against the bits it produces, and it runs with no Slang and no GPU so CI and macOS
//! can gate on it.
//!
//! Read `docs/design/observation-lowering.md` before changing a kernel: it pins the resize
//! convention, the layouts, and what is still `unverified` against `LeRobot` (spec 7.7).
#![forbid(unsafe_code)]

pub mod budget;
pub mod bundle;
pub mod exec;
pub mod kernels;
pub mod plan;

pub use budget::{
    BudgetDomains, BudgetInputs, BudgetItem, BudgetViolation, MemoryBudget, MemoryReport,
    ModelSizes, Precision, TileAtlasCfg,
};
pub use bundle::{
    Bundle, BundleError, BundleHashes, BundleKind, BundleManifest, PolicyBundle,
    BUNDLE_SCHEMA_VERSION,
};
pub use exec::{ExecError, Outputs, Tensor, TensorRef};
pub use plan::{BufferDesc, BufferId, CpuPlan, Home, Op, PlanMode, Step};
