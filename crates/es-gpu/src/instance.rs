//! Instance creation, physical-device selection and the capability *query* (spec §3.4
//! step 1).
//!
//! Spec §4.2 rule 3: this crate is the only place below layer 3 that knows Vulkan symbols.

use std::ffi::CStr;

use ash::{vk, Entry, Instance};

use crate::caps::{Capabilities, FloatControls, Independence, MaxWorkgroup, MemoryHeap, Subgroup};
use crate::error::GpuError;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";
const COOPERATIVE_MATRIX: &CStr = c"VK_KHR_cooperative_matrix";

/// How to open the device.
#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    /// Prefer a discrete GPU over an integrated one. Selection never uses enumeration order
    /// as an identity (spec §22) — the chosen device's UUID is reported in [`Capabilities`].
    pub prefer_discrete: bool,
    /// Request `VK_LAYER_KHRONOS_validation`. Absent layer is not an error; check
    /// `Gpu::validation_enabled()`.
    pub validation: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            prefer_discrete: true,
            validation: false,
        }
    }
}

/// Load the loader and create an instance. Returns whether validation was actually enabled.
pub(crate) fn create_instance(
    entry: &Entry,
    options: GpuOptions,
) -> Result<(Instance, bool), GpuError> {
    let app = vk::ApplicationInfo::default()
        .application_name(c"electric-sheep")
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let mut layers: Vec<*const i8> = Vec::new();
    let mut validation = false;
    if options.validation {
        // SAFETY: `entry` is a loaded Vulkan loader; the call only reads loader state.
        let available = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        if available
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
        {
            layers.push(VALIDATION_LAYER.as_ptr());
            validation = true;
        }
    }

    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layers);
    // SAFETY: `info` and the slices it points at outlive the call; no extension structs are
    // chained.
    let instance = unsafe { entry.create_instance(&info, None) }?;
    Ok((instance, validation))
}

/// Pick a physical device and a compute queue family.
///
/// Deterministic given the same machine: the score is a property of the device, and ties
/// break on enumeration index only after every device property has been compared.
pub(crate) fn select_device(
    instance: &Instance,
    options: GpuOptions,
) -> Result<(vk::PhysicalDevice, u32), GpuError> {
    // SAFETY: `instance` is live for the duration of the call.
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let mut best: Option<(u32, usize, vk::PhysicalDevice, u32)> = None;
    for (index, &pd) in devices.iter().enumerate() {
        // SAFETY: `pd` came from this instance's enumeration.
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(family) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        // SAFETY: same.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let score = device_score(props.device_type, options.prefer_discrete);
        let family = u32::try_from(family).unwrap_or(0);
        if best.is_none_or(|(s, i, _, _)| score > s || (score == s && index < i)) {
            best = Some((score, index, pd, family));
        }
    }
    best.map(|(_, _, pd, q)| (pd, q)).ok_or(GpuError::NoDevice)
}

fn device_score(kind: vk::PhysicalDeviceType, prefer_discrete: bool) -> u32 {
    let discrete = u32::from(kind == vk::PhysicalDeviceType::DISCRETE_GPU);
    let integrated = u32::from(kind == vk::PhysicalDeviceType::INTEGRATED_GPU);
    let other = u32::from(kind == vk::PhysicalDeviceType::VIRTUAL_GPU) * 2
        + u32::from(kind == vk::PhysicalDeviceType::CPU);
    if prefer_discrete {
        discrete * 8 + integrated * 4 + other
    } else {
        integrated * 8 + discrete * 4 + other
    }
}

fn device_type_name(kind: vk::PhysicalDeviceType) -> &'static str {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn cstr_to_string(raw: Result<&CStr, std::ffi::FromBytesUntilNulError>) -> String {
    raw.map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Subgroup operation classes, in a fixed order so the capability record is deterministic.
fn subgroup_ops(flags: vk::SubgroupFeatureFlags) -> Vec<String> {
    const TABLE: &[(vk::SubgroupFeatureFlags, &str)] = &[
        (vk::SubgroupFeatureFlags::BASIC, "basic"),
        (vk::SubgroupFeatureFlags::VOTE, "vote"),
        (vk::SubgroupFeatureFlags::ARITHMETIC, "arithmetic"),
        (vk::SubgroupFeatureFlags::BALLOT, "ballot"),
        (vk::SubgroupFeatureFlags::SHUFFLE, "shuffle"),
        (
            vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE,
            "shuffle_relative",
        ),
        (vk::SubgroupFeatureFlags::CLUSTERED, "clustered"),
        (vk::SubgroupFeatureFlags::QUAD, "quad"),
    ];
    TABLE
        .iter()
        .filter(|(f, _)| flags.contains(*f))
        .map(|(_, name)| (*name).to_owned())
        .collect()
}

/// Query everything the device says about itself. Nothing here sets anything (spec §3.4).
pub(crate) fn query_capabilities(instance: &Instance, pd: vk::PhysicalDevice) -> Capabilities {
    let mut float_controls = vk::PhysicalDeviceFloatControlsProperties::default();
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut float_controls)
        .push_next(&mut subgroup)
        .push_next(&mut driver)
        .push_next(&mut id);
    // SAFETY: `pd` belongs to `instance`; the chained structs live until the call returns
    // and each is written by the driver, not read.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    // SAFETY: same.
    let features = unsafe { instance.get_physical_device_features(pd) };
    // SAFETY: same.
    let memory = unsafe { instance.get_physical_device_memory_properties(pd) };
    // SAFETY: same.
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();

    let props = props2.properties;
    let limits = props.limits;
    let api = props.api_version;

    Capabilities {
        device_name: cstr_to_string(props.device_name_as_c_str()),
        device_uuid: id.device_uuid,
        device_type: device_type_name(props.device_type).to_owned(),
        driver_name: cstr_to_string(driver.driver_name_as_c_str()),
        driver_info: cstr_to_string(driver.driver_info_as_c_str()),
        driver_version: props.driver_version,
        api_version: (
            vk::api_version_major(api),
            vk::api_version_minor(api),
            vk::api_version_patch(api),
        ),
        float_controls: FloatControls {
            denorm_behavior_independence: Independence::from_raw(
                float_controls.denorm_behavior_independence.as_raw(),
            ),
            rounding_mode_independence: Independence::from_raw(
                float_controls.rounding_mode_independence.as_raw(),
            ),
            shader_signed_zero_inf_nan_preserve_float16: float_controls
                .shader_signed_zero_inf_nan_preserve_float16
                != 0,
            shader_signed_zero_inf_nan_preserve_float32: float_controls
                .shader_signed_zero_inf_nan_preserve_float32
                != 0,
            shader_signed_zero_inf_nan_preserve_float64: float_controls
                .shader_signed_zero_inf_nan_preserve_float64
                != 0,
            shader_denorm_preserve_float16: float_controls.shader_denorm_preserve_float16 != 0,
            shader_denorm_preserve_float32: float_controls.shader_denorm_preserve_float32 != 0,
            shader_denorm_preserve_float64: float_controls.shader_denorm_preserve_float64 != 0,
            shader_denorm_flush_to_zero_float16: float_controls.shader_denorm_flush_to_zero_float16
                != 0,
            shader_denorm_flush_to_zero_float32: float_controls.shader_denorm_flush_to_zero_float32
                != 0,
            shader_denorm_flush_to_zero_float64: float_controls.shader_denorm_flush_to_zero_float64
                != 0,
            shader_rounding_mode_rte_float16: float_controls.shader_rounding_mode_rte_float16 != 0,
            shader_rounding_mode_rte_float32: float_controls.shader_rounding_mode_rte_float32 != 0,
            shader_rounding_mode_rte_float64: float_controls.shader_rounding_mode_rte_float64 != 0,
            shader_rounding_mode_rtz_float16: float_controls.shader_rounding_mode_rtz_float16 != 0,
            shader_rounding_mode_rtz_float32: float_controls.shader_rounding_mode_rtz_float32 != 0,
            shader_rounding_mode_rtz_float64: float_controls.shader_rounding_mode_rtz_float64 != 0,
        },
        subgroup: Subgroup {
            size: subgroup.subgroup_size,
            supported_ops: subgroup_ops(subgroup.supported_operations),
            quad_operations_in_all_stages: subgroup.quad_operations_in_all_stages != 0,
        },
        shader_float64: features.shader_float64 != 0,
        shader_int64: features.shader_int64 != 0,
        cooperative_matrix: extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(COOPERATIVE_MATRIX)),
        max_workgroup: MaxWorkgroup {
            count: limits.max_compute_work_group_count,
            size: limits.max_compute_work_group_size,
            invocations: limits.max_compute_work_group_invocations,
        },
        memory_heaps: memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .map(|h| MemoryHeap {
                size_bytes: h.size,
                device_local: h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_wins_when_preferred_and_loses_when_not() {
        let d = device_score(vk::PhysicalDeviceType::DISCRETE_GPU, true);
        let i = device_score(vk::PhysicalDeviceType::INTEGRATED_GPU, true);
        let c = device_score(vk::PhysicalDeviceType::CPU, true);
        assert!(d > i && i > c);

        let d = device_score(vk::PhysicalDeviceType::DISCRETE_GPU, false);
        let i = device_score(vk::PhysicalDeviceType::INTEGRATED_GPU, false);
        assert!(i > d);
    }

    #[test]
    fn subgroup_ops_are_named_in_a_fixed_order() {
        let flags = vk::SubgroupFeatureFlags::QUAD | vk::SubgroupFeatureFlags::BASIC;
        assert_eq!(subgroup_ops(flags), ["basic", "quad"]);
        assert!(subgroup_ops(vk::SubgroupFeatureFlags::empty()).is_empty());
    }
}
