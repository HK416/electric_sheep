//! What the device says about itself, and what that implies for determinism.
//!
//! Spec §3.4: `VkPhysicalDeviceFloatControlsProperties` is a **capability query**, not a
//! setting. Nothing in this module writes a float control; everything here is filled from
//! `vkGetPhysicalDeviceProperties2` and read. The knobs that actually change shader
//! behaviour are SPIR-V execution modes, described by [`ExecModes`] and applied by the
//! compiler (`crate::slang`).

use serde::{Deserialize, Serialize};

/// Determinism tiers of spec §3.5.
///
/// `es_physics_core::caps::DeterminismTier` (layer 3) is the same enum. This crate is layer
/// 2 and may not depend on layer 3 (spec §4.2 rule 1), so the tier is mirrored here and kept
/// honest by `tier_names_match_the_spec`: the `snake_case` names are the contract that
/// crosses the crate boundary, in the hash chain and in `es backend compare` output.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismTier {
    /// 0: same `*_hash`, so the definition is the same. No numerical guarantee.
    #[default]
    SemanticEqual,
    /// 1: same `execution_hash` on the same device and driver gives bit-identical results.
    Bitwise,
    /// 2: CPU vs GPU, or one backend swapped for another, within a defined tolerance.
    CrossBackend,
    /// 3: physics-metric tolerance against `MuJoCo` or against measurement.
    PhysicsMeaning,
    /// 4: the same IR run in `PyTorch` vs the native runtime (spec §8.9).
    PolicyEquivalent,
}

/// `VkShaderFloatControlsIndependence`: which float widths can carry an execution mode
/// independently of the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Independence {
    /// Only 32-bit types can be set independently.
    ThirtyTwoBitOnly,
    /// Every width can be set independently.
    All,
    /// No width can be set independently of the others.
    #[default]
    None,
}

impl Independence {
    /// From the raw `VkShaderFloatControlsIndependence` value.
    pub fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::ThirtyTwoBitOnly,
            1 => Self::All,
            _ => Self::None,
        }
    }

    /// Whether a 32-bit execution mode can be requested without dragging fp16/fp64 along.
    pub fn covers_f32(self) -> bool {
        matches!(self, Self::ThirtyTwoBitOnly | Self::All)
    }
}

/// Every field of `VkPhysicalDeviceFloatControlsProperties`, as queried (spec §3.4 step 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FloatControls {
    pub denorm_behavior_independence: Independence,
    pub rounding_mode_independence: Independence,
    pub shader_signed_zero_inf_nan_preserve_float16: bool,
    pub shader_signed_zero_inf_nan_preserve_float32: bool,
    pub shader_signed_zero_inf_nan_preserve_float64: bool,
    pub shader_denorm_preserve_float16: bool,
    pub shader_denorm_preserve_float32: bool,
    pub shader_denorm_preserve_float64: bool,
    pub shader_denorm_flush_to_zero_float16: bool,
    pub shader_denorm_flush_to_zero_float32: bool,
    pub shader_denorm_flush_to_zero_float64: bool,
    pub shader_rounding_mode_rte_float16: bool,
    pub shader_rounding_mode_rte_float32: bool,
    pub shader_rounding_mode_rte_float64: bool,
    pub shader_rounding_mode_rtz_float16: bool,
    pub shader_rounding_mode_rtz_float32: bool,
    pub shader_rounding_mode_rtz_float64: bool,
}

/// Subgroup properties. Spec §3.4 item 5 wants subgroup behaviour *fixed*, which means the
/// size has to be known and the kernels must not branch on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Subgroup {
    pub size: u32,
    /// Operation-class names (`basic`, `vote`, `arithmetic`, ...) in a fixed order.
    pub supported_ops: Vec<String>,
    pub quad_operations_in_all_stages: bool,
}

/// Workgroup limits, straight from `VkPhysicalDeviceLimits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MaxWorkgroup {
    pub count: [u32; 3],
    pub size: [u32; 3],
    pub invocations: u32,
}

/// One memory heap, for the §20 memory budget report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryHeap {
    pub size_bytes: u64,
    pub device_local: bool,
}

/// Everything queried about the physical device. Goes into `hardware_capability` of the
/// execution hash (spec §5.3), so it is serializable and ordered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Capabilities {
    pub device_name: String,
    /// Device UUID. Spec §22 forbids identifying a device by enumeration order.
    pub device_uuid: [u8; 16],
    pub device_type: String,
    pub driver_name: String,
    pub driver_info: String,
    pub driver_version: u32,
    pub api_version: (u32, u32, u32),
    pub float_controls: FloatControls,
    pub subgroup: Subgroup,
    pub shader_float64: bool,
    pub shader_int64: bool,
    /// `VK_KHR_cooperative_matrix` present (spec §2.4 `VulkanRuntime`). Unused so far.
    pub cooperative_matrix: bool,
    pub max_workgroup: MaxWorkgroup,
    pub memory_heaps: Vec<MemoryHeap>,
}

impl Capabilities {
    /// Which tier of spec §3.5 this device can carry, derived from the query (spec §3.4
    /// step 2).
    ///
    /// Tier 1 needs the three f32 execution modes of §3.4 step 3 to be supported *and* the
    /// 32-bit width to be independently settable, plus a known subgroup size (§3.4 item 5).
    /// Anything less is tier 2: the kernel still runs, with a cross-backend tolerance
    /// instead of a bit guarantee.
    ///
    /// Support is a necessary condition, not a sufficient one: tier 1 additionally requires
    /// the same `execution_hash` on the same device and driver, which no property query can
    /// establish. The remaining four items of the eight-item contract (deterministic
    /// reduction, static partitioning, fixed subgroup behaviour, fixed algorithm) belong to
    /// the caller.
    pub fn determinism_tier(&self) -> DeterminismTier {
        let fc = &self.float_controls;
        let f32_modes_supported = fc.shader_denorm_flush_to_zero_float32
            && fc.shader_rounding_mode_rte_float32
            && fc.shader_signed_zero_inf_nan_preserve_float32;
        let independent = fc.denorm_behavior_independence.covers_f32()
            && fc.rounding_mode_independence.covers_f32();
        if f32_modes_supported && independent && self.subgroup.size > 0 {
            DeterminismTier::Bitwise
        } else {
            DeterminismTier::CrossBackend
        }
    }

    /// The SPIR-V execution modes and `NoContraction` policy the **compiler** must apply
    /// (spec §3.4 step 3) for this device to reach the tier above.
    ///
    /// This is a compile input, not a device setting: there is no way to write a float
    /// control onto a `VkDevice`, and this crate offers no setter that pretends otherwise.
    ///
    /// A mode the device does not advertise is left out rather than demanded: a SPIR-V
    /// module declaring an unsupported float-control capability is invalid, and the point of
    /// step 2 is to *refuse* the deterministic claim, not to force the mode. Which is why
    /// `determinism_tier()` reports tier 2 for exactly those devices.
    pub fn deterministic_execution_modes(&self) -> ExecModes {
        let fc = &self.float_controls;
        ExecModes {
            denorm_flush_to_zero_f32: fc.shader_denorm_flush_to_zero_float32,
            rounding_mode_rte_f32: fc.shader_rounding_mode_rte_float32,
            signed_zero_inf_nan_preserve_f32: fc.shader_signed_zero_inf_nan_preserve_float32,
            no_contraction: true,
        }
    }
}

/// SPIR-V execution modes + decoration policy a module must be compiled with (spec §3.4
/// step 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExecModes {
    pub denorm_flush_to_zero_f32: bool,
    pub rounding_mode_rte_f32: bool,
    pub signed_zero_inf_nan_preserve_f32: bool,
    /// `NoContraction` on every float arithmetic result, so the driver cannot fuse
    /// multiply-add. On its own this does *not* make execution deterministic — §3.4 needs
    /// all eight contract items.
    pub no_contraction: bool,
}

impl ExecModes {
    /// All four on: the §3.4 step 3 set.
    pub fn deterministic() -> Self {
        Self {
            denorm_flush_to_zero_f32: true,
            rounding_mode_rte_f32: true,
            signed_zero_inf_nan_preserve_f32: true,
            no_contraction: true,
        }
    }

    /// Nothing requested. For kernels outside the deterministic contract — spec §3.2 exempts
    /// neural-network kernels, whose bit reproducibility is a separate contract.
    pub fn none() -> Self {
        Self::default()
    }

    /// Stable bytes for the cache key (spec §2.3): the SPIR-V depends on these.
    pub fn key_bytes(self) -> [u8; 4] {
        [
            u8::from(self.denorm_flush_to_zero_f32),
            u8::from(self.rounding_mode_rte_f32),
            u8::from(self.signed_zero_inf_nan_preserve_f32),
            u8::from(self.no_contraction),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device that supports everything.
    fn full() -> Capabilities {
        Capabilities {
            float_controls: FloatControls {
                denorm_behavior_independence: Independence::All,
                rounding_mode_independence: Independence::All,
                shader_signed_zero_inf_nan_preserve_float32: true,
                shader_denorm_flush_to_zero_float32: true,
                shader_rounding_mode_rte_float32: true,
                ..FloatControls::default()
            },
            subgroup: Subgroup {
                size: 32,
                ..Subgroup::default()
            },
            ..Capabilities::default()
        }
    }

    #[test]
    fn tier_one_needs_the_three_f32_modes() {
        assert_eq!(full().determinism_tier(), DeterminismTier::Bitwise);

        for drop_one in [
            |c: &mut Capabilities| c.float_controls.shader_denorm_flush_to_zero_float32 = false,
            |c: &mut Capabilities| c.float_controls.shader_rounding_mode_rte_float32 = false,
            |c: &mut Capabilities| {
                c.float_controls.shader_signed_zero_inf_nan_preserve_float32 = false;
            },
        ] {
            let mut caps = full();
            drop_one(&mut caps);
            assert_eq!(caps.determinism_tier(), DeterminismTier::CrossBackend);
        }
    }

    #[test]
    fn tier_one_needs_independent_f32_control_and_a_subgroup_size() {
        let mut caps = full();
        caps.float_controls.denorm_behavior_independence = Independence::None;
        assert_eq!(caps.determinism_tier(), DeterminismTier::CrossBackend);

        let mut caps = full();
        caps.float_controls.rounding_mode_independence = Independence::None;
        assert_eq!(caps.determinism_tier(), DeterminismTier::CrossBackend);

        let mut caps = full();
        caps.subgroup.size = 0;
        assert_eq!(caps.determinism_tier(), DeterminismTier::CrossBackend);
    }

    #[test]
    fn thirty_two_bit_only_is_enough_for_tier_one() {
        let mut caps = full();
        caps.float_controls.denorm_behavior_independence = Independence::ThirtyTwoBitOnly;
        caps.float_controls.rounding_mode_independence = Independence::ThirtyTwoBitOnly;
        assert_eq!(caps.determinism_tier(), DeterminismTier::Bitwise);
    }

    #[test]
    fn independence_raw_values_follow_vulkan() {
        assert_eq!(Independence::from_raw(0), Independence::ThirtyTwoBitOnly);
        assert_eq!(Independence::from_raw(1), Independence::All);
        assert_eq!(Independence::from_raw(2), Independence::None);
        assert!(!Independence::None.covers_f32());
    }

    #[test]
    fn tier_names_match_the_spec() {
        // Mirror check for the local `DeterminismTier` (see the module comment): the
        // serialized names are the spec §3.5 contract.
        let names: Vec<String> = [
            DeterminismTier::SemanticEqual,
            DeterminismTier::Bitwise,
            DeterminismTier::CrossBackend,
            DeterminismTier::PhysicsMeaning,
            DeterminismTier::PolicyEquivalent,
        ]
        .iter()
        .map(|t| serde_json::to_string(t).unwrap())
        .collect();
        assert_eq!(
            names,
            [
                "\"semantic_equal\"",
                "\"bitwise\"",
                "\"cross_backend\"",
                "\"physics_meaning\"",
                "\"policy_equivalent\"",
            ]
        );
    }

    #[test]
    fn unsupported_modes_are_not_requested() {
        assert_eq!(
            full().deterministic_execution_modes(),
            ExecModes::deterministic()
        );
        // NVIDIA preserves f32 denormals and reports no flush-to-zero support: the mode is
        // dropped from the compile request, and the tier drops with it.
        let mut caps = full();
        caps.float_controls.shader_denorm_flush_to_zero_float32 = false;
        let modes = caps.deterministic_execution_modes();
        assert!(!modes.denorm_flush_to_zero_f32);
        assert!(modes.rounding_mode_rte_f32 && modes.no_contraction);
        assert_eq!(caps.determinism_tier(), DeterminismTier::CrossBackend);
    }

    #[test]
    fn exec_mode_key_bytes_separate_the_modes() {
        assert_eq!(ExecModes::none().key_bytes(), [0, 0, 0, 0]);
        assert_eq!(ExecModes::deterministic().key_bytes(), [1, 1, 1, 1]);
    }
}
