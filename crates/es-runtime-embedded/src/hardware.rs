//! The `hardware_capability` slot of `execution_hash` (spec 5.3).
//!
//! Spec 5.3 only needs the *identity* of the capability report, not the report. What goes in
//! is everything that could make a bitwise-identical re-run impossible on this machine and
//! that this crate can see without a GPU or a driver:
//!
//! - the target triple, as `ARCH`/`OS`/`FAMILY` plus the pointer width — `std::env::consts`
//!   rather than a `TARGET` env var, which needs a build script to exist at all;
//! - the CPU features that change which `es-math` / `wide` code path runs, detected at
//!   runtime (so a binary built for a baseline but run on a wider machine hashes differently,
//!   which is the point);
//! - the feature *list itself*, in a fixed order, so adding a feature to the probe changes
//!   every hash deliberately rather than silently.
//!
//! Not included, and deliberately: GPU limits and driver versions. This runtime has no
//! `es-gpu` dependency (spec 9.6 excludes the renderer); a deployment that adds a Vulkan
//! `PolicyRuntime` folds the device identity into its own `runtime_hash`.

use es_ir::HardwareCapability;

const TAG: &str = "es.hardware_capability.embedded.v1";

/// `blake3(tag || triple || [feature, present]*)`.
pub fn hardware_capability() -> HardwareCapability {
    let mut h = blake3_of_str(TAG);
    h.update(std::env::consts::ARCH.as_bytes());
    h.update(std::env::consts::OS.as_bytes());
    h.update(std::env::consts::FAMILY.as_bytes());
    h.update(&(usize::BITS).to_le_bytes());
    for (name, present) in features() {
        h.update(name.as_bytes());
        h.update(&[u8::from(present)]);
    }
    HardwareCapability(*h.finalize().as_bytes())
}

fn blake3_of_str(s: &str) -> blake3::Hasher {
    let mut h = blake3::Hasher::new();
    h.update(s.as_bytes());
    h
}

/// The probe list, in a fixed order. Empty on an architecture with no stable detection macro:
/// the triple still separates it from every other target.
fn features() -> Vec<(&'static str, bool)> {
    #[cfg(target_arch = "x86_64")]
    {
        vec![
            ("sse2", is_x86_feature_detected!("sse2")),
            ("sse4.2", is_x86_feature_detected!("sse4.2")),
            ("avx", is_x86_feature_detected!("avx")),
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
        ]
    }
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
        ]
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_stable_within_a_process() {
        assert_eq!(hardware_capability(), hardware_capability());
        assert_ne!(hardware_capability(), HardwareCapability([0u8; 32]));
    }
}
