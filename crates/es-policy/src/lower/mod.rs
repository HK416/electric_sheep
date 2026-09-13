//! Lowering a `LearningGraph` to an executable artifact (spec 8.7).
//!
//! One target so far: `PyTorch`, which is the reference oracle (spec 1.4). ONNX (M2) and
//! Vulkan (M3) are siblings of [`torch`], not layers on top of it.

pub mod torch;

pub use torch::{lower_to_torch, LowerError, TorchModule};
