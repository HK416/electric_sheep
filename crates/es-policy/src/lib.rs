//! `es-policy` (layer 8): policy inference behind the `PolicyRuntime` extension point
//! (spec 2.4, `INV-17`), and the `PyTorch` lowering that is the Learning IR's reference
//! oracle (spec 1.4, spec 8.7).
//!
//! Read `docs/design/learning-lowering.md` before changing anything here: it pins the node
//! table, the safetensors key scheme and the tier-4 tolerance (spec 8.9), and the code is
//! downstream of it.
//!
//! Two rules shape the whole crate.
//!
//! - **The IR owns pre- and post-processing, not the runtime** (spec 2.4, spec 8.7).
//!   [`PolicyRuntime::infer`] takes tensors that are already preprocessed and named per
//!   `PolicyContract::inputs`, and returns the graph's declared outputs. Normalization,
//!   chunking and unnormalization are IR nodes, so they are inside the lowered graph; the
//!   runtime adds nothing and has no knobs.
//! - **No pickle** (`INV-16`). [`WeightsSource`] has exactly two variants, neither of which
//!   executes code on load.
//!
//! Layer rule (spec 4.2): this crate may use `es-math`, `es-core`, `es-ir` and `es-compile`.
//! It must never depend on `es-safety` — safety is independent of policy (rule 8, `INV-11`).
#![forbid(unsafe_code)]

pub mod equiv;
/// Loading a real `LeRobot` ACT checkpoint — spec 8.9's M1 gate.
pub mod lerobot;
pub mod lower;
/// The tier-4 Rust reference for the sampler heads. Test-only: it is an oracle, not a runtime.
#[cfg(test)]
mod reference;
pub mod runtime;
pub mod torch_runtime;
pub mod weights;

pub use equiv::{compare_actions, Equivalence, Tolerance};
pub use lerobot::{act_policy, lower_act, remap_act_keys, remap_checkpoint, ActConfig};
pub use lower::{lower_to_torch, LowerError, TorchModule};
pub use runtime::{InferenceBackend, PolicyError, PolicyInfo, PolicyRuntime, WeightsSource};
pub use torch_runtime::TorchRuntime;
pub use weights::{SafetensorsEntry, WEIGHT_PREFIX};
