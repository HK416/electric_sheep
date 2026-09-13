//! Compute pipelines and the one command recorder.
//!
//! There are no graphics pipelines, no second queue, no atomics helper and no async
//! transfer: spec §3.4 forbids atomic FP sums and multi-queue work in deterministic mode, so
//! the API does not offer them. Dispatches are separated by an explicit memory barrier, so a
//! recorded sequence executes in the order it was recorded.

use std::cell::Cell;
use std::ffi::CString;

use ash::vk;

use crate::buffer::Buffer;
use crate::device::Gpu;
use crate::error::GpuError;
use crate::slang::SpirvModule;
use crate::spirv;

/// One descriptor binding in set 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingDesc {
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    Storage,
    Uniform,
}

impl BindingKind {
    fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::Storage => vk::DescriptorType::STORAGE_BUFFER,
            Self::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        }
    }
}

/// A compute pipeline plus a descriptor pool it hands out one set per dispatch from.
pub struct ComputePipeline<'gpu> {
    gpu: &'gpu Gpu,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    bindings: Vec<BindingDesc>,
    used_sets: Cell<u32>,
}

impl std::fmt::Debug for ComputePipeline<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("bindings", &self.bindings)
            .field("used_sets", &self.used_sets.get())
            .finish()
    }
}

impl<'gpu> ComputePipeline<'gpu> {
    /// Descriptor sets in the pool. One dispatch takes one set, because a descriptor set is
    /// written on the host and a recorded command buffer reads it at submit time — reusing
    /// one set for two dispatches with different buffers would silently run both with the
    /// last binding.
    ///
    /// ponytail: fixed-size pool; call [`reset_descriptors`](Self::reset_descriptors) after
    /// a wait, or raise this, if a plan needs more dispatches per pipeline.
    pub const MAX_DISPATCHES: u32 = 256;

    pub fn new(
        gpu: &'gpu Gpu,
        module: &SpirvModule,
        bindings: &[BindingDesc],
    ) -> Result<Self, GpuError> {
        let device = gpu.device();
        let info = vk::ShaderModuleCreateInfo::default().code(&module.words);
        // SAFETY: `module.words` is a validated SPIR-V module (it came from `slangc` and
        // through the header check in `spirv`), and outlives the call.
        let shader = unsafe { device.create_shader_module(&info, None) }?;

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.kind.descriptor_type())
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        // SAFETY: `set_info` and the slice it points at outlive the call.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_info, None) }?;

        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: same.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        // Slang names the SPIR-V entry point `main` unless told otherwise, so take the name
        // from the module rather than assuming it matches the Slang function name.
        let entry_name = spirv::spirv_entry_points(&module.words)
            .into_iter()
            .next()
            .unwrap_or_else(|| module.entry.clone());
        let entry = CString::new(entry_name)
            .map_err(|e| GpuError::Spirv(format!("entry point name: {e}")))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: the create-info and every handle it names belong to this device and
        // outlive the call. No pipeline cache: the SPIR-V cache is content-addressed and a
        // driver cache adds nothing to determinism.
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| GpuError::Vulkan(e))?[0];

        let sizes: Vec<vk::DescriptorPoolSize> = [BindingKind::Storage, BindingKind::Uniform]
            .iter()
            .filter_map(|kind| {
                let count =
                    u32::try_from(bindings.iter().filter(|b| b.kind == *kind).count()).unwrap_or(0);
                (count > 0).then(|| {
                    vk::DescriptorPoolSize::default()
                        .ty(kind.descriptor_type())
                        .descriptor_count(count * Self::MAX_DISPATCHES)
                })
            })
            .collect();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(Self::MAX_DISPATCHES)
            .pool_sizes(&sizes);
        // SAFETY: `pool_info` and its slice outlive the call.
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        Ok(Self {
            gpu,
            module: shader,
            set_layout,
            layout,
            pipeline,
            pool,
            bindings: bindings.to_vec(),
            used_sets: Cell::new(0),
        })
    }

    /// Give every descriptor set back. Only legal once the work that used them has finished
    /// (i.e. after `submit_and_wait`).
    pub fn reset_descriptors(&self) -> Result<(), GpuError> {
        // SAFETY: the pool belongs to this device; the caller guarantees no submitted
        // command buffer still references its sets.
        unsafe {
            self.gpu
                .device()
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }?;
        self.used_sets.set(0);
        Ok(())
    }

    fn allocate_set(&self, buffers: &[&Buffer<'_>]) -> Result<vk::DescriptorSet, GpuError> {
        if buffers.len() != self.bindings.len() {
            return Err(GpuError::Exhausted(format!(
                "pipeline has {} bindings, dispatch passed {}",
                self.bindings.len(),
                buffers.len()
            )));
        }
        if self.used_sets.get() >= Self::MAX_DISPATCHES {
            return Err(GpuError::Exhausted(format!(
                "more than {} dispatches on one pipeline without reset_descriptors",
                Self::MAX_DISPATCHES
            )));
        }
        let device = self.gpu.device();
        let layouts = [self.set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: `info` names this pipeline's pool and layout and outlives the call.
        let set = unsafe { device.allocate_descriptor_sets(&info) }?[0];
        self.used_sets.set(self.used_sets.get() + 1);

        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.handle())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = self
            .bindings
            .iter()
            .zip(&infos)
            .map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b.binding)
                    .descriptor_type(b.kind.descriptor_type())
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: `writes` and the `infos` they point at outlive the call; every buffer is
        // alive for at least as long as the recorder that dispatches with it.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }
}

impl Drop for ComputePipeline<'_> {
    fn drop(&mut self) {
        let device = self.gpu.device();
        // SAFETY: every handle belongs to this device and is destroyed exactly once; no
        // submission is in flight because every submit path waits before returning.
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.module, None);
        }
    }
}

/// Records dispatches on the single compute queue and submits them once.
pub struct CommandRecorder<'gpu> {
    gpu: &'gpu Gpu,
    cmd: vk::CommandBuffer,
    dispatches: u32,
}

impl std::fmt::Debug for CommandRecorder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandRecorder")
            .field("dispatches", &self.dispatches)
            .finish()
    }
}

impl<'gpu> CommandRecorder<'gpu> {
    pub fn new(gpu: &'gpu Gpu) -> Result<Self, GpuError> {
        let cmd = gpu.allocate_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` was just allocated from this device's pool and is not recording.
        unsafe { gpu.device().begin_command_buffer(cmd, &begin) }?;
        Ok(Self {
            gpu,
            cmd,
            dispatches: 0,
        })
    }

    /// Bind `buffers` (one per [`BindingDesc`], in declaration order) and dispatch
    /// `groups` workgroups. A full shader-write → shader-read barrier is recorded after
    /// every dispatch, so consecutive dispatches see each other's writes and the execution
    /// order is the recording order.
    pub fn dispatch(
        &mut self,
        pipeline: &ComputePipeline<'_>,
        buffers: &[&Buffer<'_>],
        groups: [u32; 3],
    ) -> Result<(), GpuError> {
        let set = pipeline.allocate_set(buffers)?;
        let device = self.gpu.device();
        // SAFETY: `self.cmd` is recording on this device; the pipeline, layout and set all
        // belong to it and stay alive until `submit_and_wait` returns.
        unsafe {
            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(self.cmd, groups[0], groups[1], groups[2]);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.dispatches += 1;
        Ok(())
    }

    /// End, submit to the one queue, and block until the fence signals.
    pub fn submit_and_wait(self) -> Result<(), GpuError> {
        // SAFETY: `self.cmd` is recording and was begun in `new`.
        unsafe { self.gpu.device().end_command_buffer(self.cmd) }?;
        let result = self.gpu.submit_and_wait(self.cmd);
        self.gpu.free_command_buffer(self.cmd);
        result
    }
}
