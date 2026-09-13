//! The open device: one logical device, **one** compute queue, one command pool, one
//! allocator.
//!
//! Spec §3.4 forbids multiple compute queues in deterministic mode, so no second queue is
//! ever created and there is no API to ask for one.

use std::cell::RefCell;
use std::mem::ManuallyDrop;

use ash::{vk, Device, Entry, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::caps::{Capabilities, DeterminismTier, ExecModes};
use crate::error::GpuError;
use crate::instance::{create_instance, query_capabilities, select_device, GpuOptions};

/// An open Vulkan compute device.
pub struct Gpu {
    // Dropped last (field order is drop order): the loader must outlive the instance.
    allocator: RefCell<ManuallyDrop<Allocator>>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    device: Device,
    instance: Instance,
    _entry: Entry,
    caps: Capabilities,
    validation: bool,
}

impl std::fmt::Debug for Gpu {
    // Vulkan handles print as raw pointers and say nothing; the identity that matters is the
    // device and the tier it can carry.
    #[allow(clippy::missing_fields_in_debug)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("device_name", &self.caps.device_name)
            .field("tier", &self.determinism_tier())
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

impl Gpu {
    /// Open a device (spec §3.4 step 1: everything about it is queried, nothing is set).
    pub fn open(options: GpuOptions) -> Result<Self, GpuError> {
        // SAFETY: loading the Vulkan loader; no invariants beyond "do not call twice
        // concurrently with dlclose", which we do not do.
        let entry =
            unsafe { Entry::load() }.map_err(|e| GpuError::LoaderMissing(format!("{e}")))?;
        let (instance, validation) = create_instance(&entry, options)?;
        match Self::open_on(entry, instance.clone(), validation, options) {
            Ok(gpu) => Ok(gpu),
            Err(e) => {
                // SAFETY: `open_on` failed, so no `Gpu` owns this instance; the clone is
                // the only live reference. (A device created just before a later failure
                // in `open_on` leaks until process exit — an open failure is fatal for the
                // caller anyway.)
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn open_on(
        entry: Entry,
        instance: Instance,
        validation: bool,
        options: GpuOptions,
    ) -> Result<Self, GpuError> {
        let (physical_device, queue_family) = select_device(&instance, options)?;
        let caps = query_capabilities(&instance, physical_device);

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_float64(caps.shader_float64)
            .shader_int64(caps.shader_int64);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&features);
        // SAFETY: `physical_device` belongs to `instance`; the create-info slices outlive
        // the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
        // SAFETY: family and index were just created with the device.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `device` is live.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })?;

        Ok(Self {
            allocator: RefCell::new(ManuallyDrop::new(allocator)),
            command_pool,
            queue,
            device,
            instance,
            _entry: entry,
            caps,
            validation,
        })
    }

    /// CI predicate: true when no Vulkan device can be opened. Every GPU test starts here
    /// and prints `SKIP` with the reason.
    pub fn none_available() -> bool {
        Self::open(GpuOptions::default()).is_err()
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Tier of spec §3.5 this device can carry (see [`Capabilities::determinism_tier`]).
    pub fn determinism_tier(&self) -> DeterminismTier {
        self.caps.determinism_tier()
    }

    /// The execution modes a module must be compiled with. A compile input, never a device
    /// setting (spec §3.4).
    pub fn deterministic_execution_modes(&self) -> ExecModes {
        self.caps.deterministic_execution_modes()
    }

    /// Whether the validation layer was actually enabled (it is absent on machines without
    /// the SDK, which is not an error).
    pub fn validation_enabled(&self) -> bool {
        self.validation
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn allocator(&self) -> &RefCell<ManuallyDrop<Allocator>> {
        &self.allocator
    }

    /// Allocate a primary command buffer from the single pool.
    pub(crate) fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer, GpuError> {
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `info` names this device's pool and outlives the call.
        let buffers = unsafe { self.device.allocate_command_buffers(&info) }?;
        Ok(buffers[0])
    }

    pub(crate) fn free_command_buffer(&self, cmd: vk::CommandBuffer) {
        // SAFETY: `cmd` came from this pool and is no longer executing (every submit path
        // here waits on a fence or on the queue before freeing).
        unsafe { self.device.free_command_buffers(self.command_pool, &[cmd]) };
    }

    /// Record, submit and wait — the only submission path. One queue, one fence, no
    /// overlapping work (spec §3.4: deterministic scheduling is a static, single-queue
    /// schedule).
    pub(crate) fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), GpuError> {
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        // SAFETY: `cmd` is a recorded, ended primary buffer from this device's pool; the
        // fence belongs to this device and is waited on before either is destroyed.
        unsafe {
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let result = self
                .device
                .queue_submit(self.queue, &[submit], fence)
                .and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX));
            self.device.destroy_fence(fence, None);
            result?;
        }
        Ok(())
    }

    /// Record a one-shot command buffer, submit it and wait.
    pub(crate) fn one_shot<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(&Device, vk::CommandBuffer),
    {
        let cmd = self.allocate_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was just allocated and is not recording.
        unsafe { self.device.begin_command_buffer(cmd, &begin) }?;
        record(&self.device, cmd);
        // SAFETY: `cmd` is recording.
        unsafe { self.device.end_command_buffer(cmd) }?;
        let result = self.submit_and_wait(cmd);
        self.free_command_buffer(cmd);
        result
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: everything below belongs to this device, nothing is in flight (every
        // submission path waits), and each handle is destroyed exactly once, children
        // before parents.
        unsafe {
            let _ = self.device.device_wait_idle();
            ManuallyDrop::drop(&mut self.allocator.borrow_mut());
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
